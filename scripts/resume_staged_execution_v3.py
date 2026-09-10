"""Continue existing staged checkpoints with an accepted execution mode."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--code', required=True)
    parser.add_argument('--execution', choices=['optimized', 'compiled'], required=True)
    args = parser.parse_args()
    run, code = Path(args.run).resolve(), Path(args.code).resolve()
    root = run.parent.parent
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='2,3', PYTHONPATH=os.pathsep.join([str(code), str(root)]),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        TORCHINDUCTOR_CACHE_DIR=str(run/'inductor_cache'), TORCHINDUCTOR_COMPILE_THREADS='4')

    def status(**values):
        values.update(time=datetime.datetime.now().astimezone().isoformat(), execution=args.execution)
        (run/'pipeline_status.json').write_text(json.dumps(values, indent=2))
        print(json.dumps(values), flush=True)

    for stage in ['diagnostic', 'formal']:
        state_file = run/stage/'training_status.json'
        if state_file.exists() and json.loads(state_file.read_text()).get('status') == 'complete':
            continue
        if stage == 'formal' and not (run/'diagnostic/best.pt').exists():
            status(status='diagnostic_complete_gate_not_passed', reason='No accepted diagnostic checkpoint')
            return
        config = run/f'{stage}_config.json'
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                   '-m', 'emotion_ssm.train.dynamics_staged_v3', '--config', str(config),
                   '--execution', args.execution]
        checkpoint = run/stage/'last.pt'
        if checkpoint.exists():
            command += ['--resume', str(checkpoint)]
        with (run/f'{stage}.log').open('a', encoding='utf-8') as log:
            log.write('\n'+json.dumps(dict(event='execution_switch', execution=args.execution,
                checkpoint=str(checkpoint) if checkpoint.exists() else None,
                time=datetime.datetime.now().astimezone().isoformat()))+'\n')
            log.flush()
            process = subprocess.Popen(command, cwd=code, env=environment, stdout=log, stderr=subprocess.STDOUT)
            status(status='running', stage=stage, pid=process.pid, physical_gpus=[2, 3],
                   log=str(run/f'{stage}.log'))
            exit_code = process.wait()
        if exit_code:
            status(status='failed', stage=stage, exit_code=exit_code)
            raise SystemExit(exit_code)
    status(status='complete', deployment_checkpoint=str(run/'formal/best.pt') if (run/'formal/best.pt').exists() else None)


if __name__ == '__main__':
    main()
