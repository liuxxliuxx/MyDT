"""Replay the same validation dialogues under reference and compiled execution."""
import argparse
import json
from pathlib import Path
import time

import torch

from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.train.dynamics_v3 import DialogueCollection
from emotion_ssm.train.dynamics_staged_v3 import source_balanced_indices, evaluate_bank, fit_shrink, endpoint_metrics
from emotion_ssm.train.staged_dynamics_support import EncodedDialogueCache, build_origin_bank, memory_cat, parameter_partition


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', required=True)
    args = parser.parse_args(); root = Path(args.run)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    ck = torch.load(root/'execution_benchmark.source.pt', map_location='cpu', weights_only=False)
    config = ck['config']; output = root/'execution_validation.json'
    report = {'checkpoint_step': ck['global_step'], 'protocol': 'same_weights_full_fixed_live_gap_validation', 'modes': {}}
    observer = TokenObserver(ck['construction']['observer']).cuda().eval()
    teacher = TokenObserver(ck['construction']['observer']).cuda().eval()
    core = UnifiedEmotionStateCore.from_config(ck['construction']['state']).cuda()
    for name, model in [('observer', observer), ('teacher', teacher), ('state', core)]:
        model.load_state_dict(ck['models'][name])
    teacher.requires_grad_(False); parameter_partition(observer, core, 'fixed')
    data = DialogueCollection(config['data']['token_roots'], 'val')
    selected = source_balanced_indices(data, config['staged']['validation_dialogues_per_domain'], config['train']['seed']+17011)
    cache = EncodedDialogueCache(observer, teacher, config['paths']['feature_cache'], 'cuda', 32)
    references = [torch.load(root/f'diagnostic/reference_val_rank{rank}.pt', map_location='cpu', weights_only=False) for rank in [0, 1]]
    fixed = dict(references[0])
    fixed['states'] = memory_cat([bank['states'] for bank in references])
    for name in ['targets', 'valid', 'domain']:
        fixed[name] = torch.cat([bank[name] for bank in references])
    fixed['current'] = {name: sum(bank['current'][name] for bank in references) for name in references[0]['current']}
    train = torch.load(root/f"diagnostic/origins_block{ck['run_state']['block']}_rank0.pt", map_location='cpu', weights_only=False)
    mean, shrink = fit_shrink(train, torch.device('cuda'))
    baseline_states = {}
    for mode in ['reference', 'compiled']:
        core.configure_execution(mode)
        print(json.dumps({'event': 'validation_mode_started', 'mode': mode}), flush=True)
        values = {}; started = time.perf_counter()
        for label in ['fixed', 'live', 'gap']:
            gap = config['staged']['block_gap_seconds'] if label == 'gap' else 0
            bank = fixed if label == 'fixed' else build_origin_bank(observer, core, cache, data, selected,
                config['train']['forecast_seconds'], config['staged']['origin_stride'], gap,
                progress=lambda done, total, origins: print(json.dumps(dict(mode=mode, protocol=label, dialogues=done, total=total)), flush=True))
            values[label] = evaluate_bank(core, bank, mean, shrink, 'cuda', 64)
            if mode == 'reference':
                baseline_states[label] = bank['states'].z.clone()
            else:
                difference = float((bank['states'].z-baseline_states[label]).abs().max())
                values[label]['origin_max_abs_difference'] = difference
                torch.testing.assert_close(bank['states'].z, baseline_states[label], rtol=5e-5, atol=3e-6)
                for metric in ['mse', 'macro_mse']:
                    before = report['modes']['reference'][label]['methods']['learned'][metric]
                    after = values[label]['methods']['learned'][metric]
                    if abs(after-before) > max(1e-9, abs(before)*2e-5):
                        raise AssertionError(f'{label} {metric} changed beyond numerical tolerance')
            if label == 'live':
                values[label]['semantic'] = endpoint_metrics(observer, core, bank, 'cuda', 64)
            print(json.dumps({'event': 'validation_protocol_complete', 'mode': mode, 'protocol': label,
                              'mse': values[label]['methods']['learned']['macro_mse']}), flush=True)
        values['elapsed_with_live_training_seconds'] = time.perf_counter()-started
        report['modes'][mode] = values
        output.write_text(json.dumps(report, indent=2))
    report['passed'] = True; report['dialogues'] = len(selected)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'event': 'complete', 'passed': True, 'dialogues': len(selected)}), flush=True)


if __name__ == '__main__':
    main()
