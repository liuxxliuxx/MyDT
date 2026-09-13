"""At a complete checkpoint, benchmark execution and resume the same experiment."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def compare_reports(folder, modes):
    reports = {mode:json.loads((folder/mode/'report.json').read_text()) for mode in modes}
    reference = reports['reference']
    for mode, report in reports.items():
        if report['ranks'] != reference['ranks']:
            raise ValueError('Input/weight/gradient/optimizer/state/RNG equality failed: '+mode)
        for old, new in zip(reference['rows'], report['rows']):
            if old['metrics'] != new['metrics']:
                raise ValueError('Per-step metrics differ: '+mode)
    reference_seconds = reference['mean_step_seconds']
    if 'reference_repeat' in reports:
        reference_seconds = (reference_seconds + reports['reference_repeat']['mean_step_seconds']) / 2
    speed = {mode:dict(seconds=report['mean_step_seconds'],
                      data_wait_seconds=report['mean_data_wait_seconds'],
                      speedup=reference_seconds/report['mean_step_seconds'],
                      time_reduction_percent=100*(1-report['mean_step_seconds']/reference_seconds))
             for mode, report in reports.items()}
    result = dict(passed=True, compared_steps=reference['steps'], gradients_only=reference['gradients_only'],
                  reference_seconds=reference_seconds, bracketed_reference='reference_repeat' in reports,
                  equality='bitwise input/model/gradient/optimizer/scaler/state/RNG hashes and identical per-step metrics',
                  speed=speed)
    write_json(folder/'comparison.json', result)
    return result


def run(args):
    run_dir = Path(args.run_dir).resolve()
    code = Path(__file__).resolve().parents[1]
    if run_dir.name != 'avatar_old9750_20260911_gpu23' or code.parent != run_dir:
        raise ValueError('This handoff is restricted to the authorized Avatar experiment')
    root = run_dir.parents[1]
    env = dict(os.environ, PYTHONPATH=str(code), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               CUDA_VISIBLE_DEVICES='GPU-5b497823-4a84-bde7-5670-2172ee96245d,GPU-9117039b-5194-5d46-d4a7-088ff06ce552',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    status_path = run_dir/('throughput_arm_status.json' if args.mode == 'arm' else 'throughput_status.json')
    journal = dict(pid=os.getpid(), status='waiting_for_complete_checkpoint', minimum_step=args.minimum_step,
                   current_training_preserved=True, code=str(code), mode=args.mode)
    write_json(status_path, journal)

    def execute(module, arguments, log):
        command = [sys.executable, '-u', '-B', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                   '--module', module, *arguments]
        with Path(log).open('ab') as handle:
            child = subprocess.Popen(command, cwd=code, env=env, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT)
            journal.update(child_pid=child.pid, command=command, log=str(log))
            write_json(status_path, journal)
            result = child.wait()
        if result:
            raise RuntimeError(f'{module} exited with code {result}; see {log}')

    def benchmark(folder, checkpoint, gradients_only):
        folder.mkdir(parents=True, exist_ok=True)
        modes = ['reference', 'metrics', 'prefetch', 'combined', 'reference_repeat']
        for mode in modes:
            implementation = 'reference' if mode == 'reference_repeat' else mode
            journal.update(status='benchmarking', benchmark_mode=mode, gradients_only=gradients_only)
            options = ['--checkpoint', str(checkpoint), '--output', str(folder/mode), '--mode', implementation,
                       '--steps', str(args.benchmark_steps), '--warmup', str(min(5, args.benchmark_steps-1))]
            if gradients_only:
                options += ['--gradients-only']
            elif mode == 'combined':
                options += ['--save-continuation']
            execute('scripts.benchmark_avatar_throughput', options, folder/(mode+'.log'))
        result = compare_reports(folder, modes)
        journal.update(benchmark=result)
        write_json(status_path, journal)
        return folder/'combined/continuation.pt'

    if args.mode == 'arm':
        journal.update(status='waiting_for_probe_completion')
        write_json(status_path, journal)
        while True:
            current = json.loads((run_dir/'throughput_status.json').read_text())
            if current.get('status') == 'probe_complete':
                break
            probe_pid = current['pid']
            stat = Path('/proc')/str(probe_pid)/'stat'
            if not stat.exists() or stat.read_text().split(') ')[1][0] == 'Z':
                if json.loads((run_dir/'throughput_status.json').read_text()).get('status') == 'probe_complete':
                    break
                raise RuntimeError('Initial probe exited before its equality gate passed')
            time.sleep(1.)
        folder = run_dir/'throughput_probe'
        journal.update(status='repeating_reference_probe')
        execute('scripts.benchmark_avatar_throughput', [
            '--checkpoint', str(run_dir/'smoke/last.pt'), '--output', str(folder/'reference_repeat'),
            '--mode', 'reference', '--steps', '12', '--warmup', '5', '--gradients-only'],
            folder/'reference_repeat.log')
        result = compare_reports(folder, ['reference','metrics','prefetch','combined','reference_repeat'])
        journal.update(status='armed', probe_comparison=result, completed=time.time())
        write_json(status_path, journal)
        args.mode = 'handoff'
        return run(args)

    if args.mode == 'probe':
        benchmark(run_dir/'throughput_probe', run_dir/'smoke/last.pt', True)
        journal.update(status='probe_complete', completed=time.time())
        write_json(status_path, journal)
        return

    # Read only completed validation announcements: best.pt has also finished
    # copying before this line is printed by the existing trainer.
    completed = 0
    log_path = run_dir/'formal.log'
    with log_path.open(errors='replace') as log:
        while completed < args.minimum_step:
            position = log.tell()
            line = log.readline()
            if not line.endswith('\n'):
                # Preserve a partially written validation announcement.
                log.seek(position)
                time.sleep(.25)
                continue
            try:
                completed = max(completed, int(json.loads(line).get('validation_step', 0)))
            except (ValueError, TypeError):
                pass
    checkpoint = run_dir/'formal/last.pt'
    # Match the exact original Avatar process IDs and code paths before stopping.
    original = json.loads((run_dir/'process.json').read_text())
    pipeline = json.loads((run_dir/'pipeline_status.json').read_text())
    if pipeline.get('stage') != 'formal' or pipeline.get('pid') != original['pid']:
        raise ValueError('Original Avatar pipeline changed; automatic handoff refused')
    old_pids = [original['pid'], pipeline['child_pid']]
    for pid in old_pids:
        command = (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if str(run_dir) not in command:
            raise ValueError('PID no longer belongs to the intended Avatar run')
    journal.update(status='switching_at_checkpoint', source_step=completed, old_pids=old_pids)
    write_json(status_path, journal)
    # Stop the old supervisor first so its expected SIGTERM cannot mark a new
    # training process failed or launch test/OOD concurrently.
    os.kill(old_pids[0], signal.SIGTERM)
    os.kill(old_pids[1], signal.SIGTERM)
    deadline = time.time()+60
    while time.time() < deadline:
        remaining = []
        for pid in old_pids:
            stat = Path('/proc')/str(pid)/'stat'
            if stat.exists() and stat.read_text().split(') ')[1][0] != 'Z':
                remaining.append(pid)
        if not remaining:
            break
        time.sleep(.25)
    else:
        raise RuntimeError('Old Avatar processes did not exit; refusing duplicate allocation')

    resume_checkpoint = checkpoint
    optimized = False
    try:
        resume_checkpoint = benchmark(run_dir/'throughput_benchmark', checkpoint, False)
        optimized = True
    except Exception as error:
        # A failed execution-equivalence check must never leave training stopped.
        journal.update(optimization_error=str(error), status='restoring_original_execution')
        write_json(status_path, journal)
    config = json.loads((run_dir/'formal_config.json').read_text())
    config['paths']['resume'] = str(resume_checkpoint)
    config['generation_execution'] = dict(defer_metrics=optimized, prefetch_batches=int(optimized))
    target_config = run_dir/'resume_throughput_config.json'
    write_json(target_config, config)
    journal.update(status='training', optimized=optimized, checkpoint=str(resume_checkpoint), continued=time.time())
    write_json(status_path, journal)
    write_json(run_dir/'pipeline_status.json', dict(status='running', stage='formal', pid=os.getpid(),
               execution='metrics-prefetch-v1' if optimized else 'original', log=str(run_dir/'formal_optimized.log'),
               physical_gpus=[2,3], checkpoint=str(resume_checkpoint)))
    execute('emotion_ssm.train.generation_v3', ['--config', str(target_config)], run_dir/'formal_optimized.log')
    for split in ('test','ood'):
        journal.update(status='evaluating', split=split)
        execute('emotion_ssm.evaluate_v3', ['--checkpoint', str(run_dir/'formal/best.pt'), '--split', split,
                '--output', str(run_dir/'formal'/f'{split}.json')], run_dir/(f'evaluate_{split}.log'))
    journal.update(status='complete', completed=time.time())
    write_json(status_path, journal)
    write_json(run_dir/'pipeline_status.json', dict(status='complete', stage='evaluated', formal_steps=30000,
               best_checkpoint=str(run_dir/'formal/best.pt'), evaluated_splits=['test','ood']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--minimum-step', type=int, default=1000)
    p.add_argument('--benchmark-steps', type=int, default=20)
    p.add_argument('--mode', choices=['probe','handoff','arm'], default='handoff')
    args = p.parse_args()
    try:
        run(args)
    except Exception as error:
        path = Path(args.run_dir)/('throughput_arm_status.json' if args.mode == 'arm' else 'throughput_status.json')
        old = json.loads(path.read_text()) if path.exists() else {}
        old.update(status='failed', error=str(error), failed=time.time())
        write_json(path, old)
        raise
