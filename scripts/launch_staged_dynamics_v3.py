"""Server1 GPU2/3 diagnostic followed by a gated formal dynamics run."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--code-dir', required=True)
    parser.add_argument('--init', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root, output, code = Path(args.root), Path(args.run_dir), Path(args.code_dir)
    output.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.init, map_location='cpu', weights_only=False)
    config = copy.deepcopy(ck['config'])
    config['data']['token_roots'] = [str(root/'artifacts'/folder) for folder in [
        'v3_1_retrain_20260908_gpu23/emotiontalk_tokens',
        'v3_1_supervision_fix_20260909_gpu23/iemocap_tokens',
        'v3_1_retrain_20260908_gpu23/dualtalk_tokens']]
    config['data']['dualtalk_tokens'] = config['data']['token_roots'][2]
    config['paths'].update(dynamics_checkpoint=str(Path(args.init).resolve()), resume='',
        feature_cache=str(root/'artifacts/v3_2_1_frozen_features'), baseline='', observation_checkpoint='')
    config['train'].update(global_chunks_per_step=32, cpu_threads=4, amp=False,
                           compile_adaptive_flow=False, device='cuda:0')
    config['staged'] = dict(calibration_steps=100, fixed_steps=400, joint_rounds=2,
        joint_steps_per_round=50, readapt_steps_per_round=200, bank_refresh_steps=200,
        bank_dialogues_per_domain=32, validation_dialogues_per_domain=8,
        validation_every=100, checkpoint_every=100, log_every=10, compile_flow=False)
    config['paths']['output'] = str(output/'diagnostic')
    (output/'diagnostic_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    formal = copy.deepcopy(config)
    formal['paths'].update(output=str(output/'formal'), dynamics_checkpoint=str(output/'diagnostic/best.pt'))
    formal['staged'].update(calibration_steps=1000, fixed_steps=4000, joint_rounds=4,
        joint_steps_per_round=250, readapt_steps_per_round=1000, bank_refresh_steps=1000,
        bank_dialogues_per_domain=64, validation_every=250, checkpoint_every=250)
    (output/'formal_config.json').write_text(json.dumps(formal, indent=2), encoding='utf-8')
    if args.prepare_only:
        print(json.dumps({'diagnostic_steps': 1000, 'formal_steps': 10000, 'run_dir': str(output)}))
        return
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='2,3', PYTHONPATH=os.pathsep.join([str(code), str(root)]),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    def status(**values):
        values['time'] = datetime.datetime.now().astimezone().isoformat()
        (output/'pipeline_status.json').write_text(json.dumps(values, indent=2), encoding='utf-8')
        print(json.dumps(values), flush=True)
    for stage in ['diagnostic', 'formal']:
        if stage == 'formal' and not (output/'diagnostic/best.pt').exists():
            status(status='diagnostic_complete_gate_not_passed',
                   reason='No checkpoint passed fixed-origin, fresh-origin and block-gap forecast guards; formal run not started.')
            return
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                   '-m', 'emotion_ssm.train.dynamics_staged_v3', '--config', str(output/f'{stage}_config.json')]
        with (output/f'{stage}.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=code, env=environment, stdout=log, stderr=subprocess.STDOUT)
            status(status='running', stage=stage, pid=process.pid, physical_gpus=[2, 3],
                   log=str(output/f'{stage}.log'))
            exit_code = process.wait()
        if exit_code:
            status(status='failed', stage=stage, exit_code=exit_code)
            raise SystemExit(exit_code)
    status(status='complete', deployment_checkpoint=str(output/'formal/best.pt')
           if (output/'formal/best.pt').exists() else None)


if __name__ == '__main__':
    main()
