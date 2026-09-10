"""Compare execution modes with identical origins, labels and optimization steps."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import statistics
import time

import torch

from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.train.staged_dynamics_support import parameter_partition, prediction_loss
from emotion_ssm.utils.dynamics_execution import synchronize_gradients_batched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--modes', nargs='+', default=['reference', 'optimized', 'compiled'])
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--pause-training-for-timing', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    # A single integration callable is used with grad/no-grad and several
    # trainable parameter partitions. These are explicit, finite variants.
    torch._dynamo.config.cache_size_limit = 64
    run, output = Path(args.run), Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path = output.with_suffix('.source.pt')
    if not source_path.exists():
        ck = torch.load(run/'diagnostic/last.pt', map_location='cpu', weights_only=False)
        torch.save(ck, source_path)
    ck = torch.load(source_path, map_location='cpu', weights_only=False)
    bank = torch.load(run/f"diagnostic/origins_block{ck['run_state']['block']}_rank0.pt", map_location='cpu', weights_only=False)
    indices = torch.linspace(0, len(bank['states'].fast)-1, 16).long()
    instances = {}
    report = dict(checkpoint_step=ck['global_step'], global_batch=32, per_rank_origins=16,
                  horizons=bank['horizons'], precision='fp32_no_tf32', modes={},
                  updates=args.updates, source_checkpoint=str(source_path))

    def persist():
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')

    def emit(**values):
        print(json.dumps(values), flush=True)

    def compute(core, observer, params, chosen):
        core.zero_grad(set_to_none=True)
        loss, predicted = prediction_loss(core, bank, chosen, 'cuda', observer, 1.)
        loss.backward()
        return loss, torch.stack([state.z for state in predicted], 1), torch.cat([
            torch.zeros_like(p).flatten() if p.grad is None else p.grad.detach().flatten() for p in params])

    reference_result = None
    for mode in args.modes:
        emit(event='warmup_started', mode=mode)
        began = time.perf_counter()
        observer = TokenObserver(ck['construction']['observer']).cuda().eval()
        core = UnifiedEmotionStateCore.from_config(ck['construction']['state']).cuda()
        observer.load_state_dict(ck['models']['observer']); core.load_state_dict(ck['models']['state'])
        params = parameter_partition(observer, core, 'fixed')
        core.configure_execution(mode)
        instances[mode] = (core, observer, params)
        result = compute(core, observer, params, indices)
        compute(core, observer, params, indices)
        torch.cuda.synchronize()
        saved = [value.detach().clone() for value in result]
        details = dict(warmup_seconds=time.perf_counter()-began)
        if reference_result is None:
            reference_result = saved
        else:
            differences = [float((a-b).abs().max()) for a, b in zip(reference_result, saved)]
            relative = float((saved[2]-reference_result[2]).norm()/reference_result[2].norm().clamp_min(1e-12))
            details.update(loss_max_abs=differences[0], forecast_max_abs=differences[1],
                           gradient_max_abs=differences[2], gradient_relative_l2=relative)
            torch.testing.assert_close(saved[0], reference_result[0], rtol=2e-5, atol=3e-6)
            torch.testing.assert_close(saved[1], reference_result[1], rtol=5e-5, atol=3e-6)
            torch.testing.assert_close(saved[2], reference_result[2], rtol=3e-4, atol=2e-5)
            if relative > 2e-4:
                raise AssertionError('Gradient relative difference exceeds execution tolerance')
        report['modes'][mode] = details
        persist(); emit(event='warmup_passed', mode=mode, **details)

    # Compilation runs alongside the live job. Only the short steady-state
    # timing section pauses verified workers, always resumed in finally.
    paused = []
    try:
        if args.pause_training_for_timing:
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit() or entry.stat().st_uid != os.getuid():
                    continue
                try:
                    argv = (entry/'cmdline').read_bytes().replace(b'\x00', b' ').decode()
                except (OSError, UnicodeError):
                    continue
                if ' -u -m emotion_ssm.train.dynamics_staged_v3 ' in argv and str(run) in argv:
                    pid = int(entry.name)
                    os.kill(pid, signal.SIGSTOP); paused.append(pid)
            emit(event='training_paused_for_timing', pids=paused)
        for mode, (core, observer, params) in instances.items():
            timings = []
            for _ in range(4):
                torch.cuda.synchronize(); start = time.perf_counter()
                compute(core, observer, params, indices)
                torch.cuda.synchronize(); timings.append(time.perf_counter()-start)
            report['modes'][mode]['step_seconds'] = timings
            report['modes'][mode]['median_seconds'] = statistics.median(timings)
            persist(); emit(event='timing_complete', mode=mode, median_seconds=statistics.median(timings))
    finally:
        for pid in paused:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        emit(event='training_resumed', pids=paused)
    report['timing_paused_worker_pids'] = paused

    # All modes receive the same sequence and fresh optimizer state. This
    # checks numerical drift over updates, not equality with an old experiment.
    generator = torch.Generator().manual_seed(93027)
    sequence = [torch.randint(len(bank['states'].fast), (16,), generator=generator) for _ in range(args.updates)]
    baseline_weights = baseline_output = None
    for mode, (core, observer, params) in instances.items():
        core.load_state_dict(ck['models']['state'])
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.001)
        began = time.perf_counter()
        for number, chosen in enumerate(sequence):
            compute(core, observer, params, chosen)
            synchronize_gradients_batched(params, 1, 'cuda')
            torch.nn.utils.clip_grad_norm_(params, 5.)
            optimizer.step()
            if (number+1) % 5 == 0:
                emit(event='update_equivalence_progress', mode=mode, updates=number+1)
        weights = torch.cat([p.detach().flatten() for p in params])
        with torch.no_grad():
            loss, output_states = prediction_loss(core, bank, indices, 'cuda', observer, 1.)
            prediction = torch.stack([s.z for s in output_states], 1)
        if baseline_weights is None:
            baseline_weights, baseline_output = weights, prediction
        else:
            delta = float((weights-baseline_weights).abs().max())
            drift = float((prediction-baseline_output).abs().max())
            report['modes'][mode].update(after_updates_parameter_max_abs=delta,
                                        after_updates_forecast_max_abs=drift)
            if delta > 1e-4 or drift > 3e-5:
                raise AssertionError('Multi-update execution drift exceeds tolerance')
        report['modes'][mode]['after_updates_loss'] = float(loss)
        report['modes'][mode]['update_check_seconds_with_other_training'] = time.perf_counter()-began
        persist(); emit(event='update_equivalence_passed', mode=mode)
    original = report['modes']['reference']['median_seconds']
    for mode, values in report['modes'].items():
        values['speedup_vs_reference'] = original/values['median_seconds']
    report['passed'] = True
    persist(); emit(event='complete', output=str(output), modes=report['modes'])


if __name__ == '__main__':
    main()
