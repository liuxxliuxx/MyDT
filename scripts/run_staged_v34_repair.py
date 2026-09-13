"""Technical preflight, origin cross-audit, 1000-step diagnostic, then a fresh full repair run."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def require_free_gpus():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    allowed={row.split(',')[1].strip() for row in rows.splitlines() if row.split(',')[0].strip() in ('2','3')}
    if len(allowed)!=2:raise RuntimeError('Cannot resolve physical GPU2 and GPU3')
    for attempt in range(10):
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        occupied=[row for row in processes.splitlines() if row.split(',')[0].strip() in allowed]
        if not occupied:return
        if attempt==9:raise RuntimeError('GPU2/3 already occupied; refuse concurrent NCCL jobs: '+str(occupied))
        time.sleep(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-config',required=True);p.add_argument('--initial-checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--comparison-root',required=True);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise ValueError('Only physical GPU2,3 authorized')
    import fcntl
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    scripts=Path(__file__).resolve().parent
    def call(name,command,log_path):
        require_free_gpus()
        with log_path.open('a',encoding='utf-8') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            state=dict(status='running',stage=name,pid=os.getpid(),child_pid=child.pid,started=time.time(),log=str(log_path),command=command)
            atomic(root/'pipeline_status.json',state)
            code=child.wait()
        state.update(status='failed' if code else 'completed_stage',exit_code=code,updated=time.time())
        atomic(root/'pipeline_status.json',state)
        if code:raise SystemExit(code)
    ddp=[sys.executable,'-u','-B','-m','torch.distributed.run','--standalone','--nproc_per_node=2']
    preflight=root/'gpu_preflight_compiled'
    if not (preflight/'result.json').exists():
        call('gpu_preflight',ddp+[str(scripts/'verify_staged_v34_ddp.py'),'--output',str(preflight)],root/'gpu_preflight_compiled.log')
    if json.loads((preflight/'result.json').read_text())['status']!='passed':raise RuntimeError('GPU/DDP preflight failed')
    cross=root/'origin_cross'
    if not (cross/'summary.json').exists():
        comparison=Path(args.comparison_root)
        call('origin_cross',[sys.executable,'-u','-B',str(scripts/'diagnose_staged_v34_origins.py'),
            '--reference',args.initial_checkpoint,'--candidate',str(comparison/'best_vector.pt'),
            '--bank-root',str(comparison),'--output',str(cross)],root/'origin_cross.log')
    config=json.loads(Path(args.base_config).read_text())
    config['paths'].update(dynamics_checkpoint=str(Path(args.initial_checkpoint).resolve()),resume='')
    config['train'].update(seed=6666,global_chunks_per_step=32,amp=False,cpu_threads=2)
    from emotion_ssm.train.staged_v34.trainer import DEFAULTS
    config['staged']={**DEFAULTS,**config.get('staged',{})}
    config['staged'].update(calibration_steps=1000,fixed_steps=4000,joint_rounds=4,joint_steps_per_round=250,
        readapt_steps_per_round=1000,bank_refresh_steps=1000,bank_dialogues_per_domain=64,
        calibration_dialogues_per_domain=16,validation_dialogues_per_domain=8,origin_stride=4,
        cache_batch=64,prediction_batch=128,gold_query_budget=64,tbptt_events=32,
        validation_every=1000,checkpoint_every=250,archive_every=1000,gradient_every=1000,
        objective_mode='weighted_joint',endpoint_label_weight=.25,class_weight_power=.5,class_weight_cap=3.,
        semantic_keep_weight=.1,neutral_margin_weight=0.,affine=False,enable_partner=True,
        schedule='constant',origin_refresh_steps=250,origin_probe_steps=100,replay_dialogue_batch=8,
        flow_replay_origins=True,train_protocol_cycle=['clean','train_mixed'])
    jobs=[('01_diagnostic',dict(calibration_steps=100,fixed_steps=400,joint_rounds=2,
        joint_steps_per_round=50,readapt_steps_per_round=200,bank_refresh_steps=200,
        validation_every=250,archive_every=250,gradient_every=250)),('02_formal',{})]
    atomic(root/'experiment_protocol.json',dict(revision='v3.4-online-origins-semantic-neutral-v1',
        initialization=args.initial_checkpoint,physical_gpus=[2,3],seed=6666,jobs=jobs,
        initial_state='each run starts independently from old9750; old optimizer is not resumed',
        observer_teacher_label_heads='frozen; correction coverage locked',
        quality='validation research run; best_semantic is diagnostic, best_deploy requires all semantic/vector gates'))
    for name,change in jobs:
        local=copy.deepcopy(config);output=root/name;output.mkdir(exist_ok=True)
        local['paths']['output']=str(output);local['staged'].update(change)
        path=output/'requested_config.json'
        if path.exists() and json.loads(path.read_text())!=local:raise ValueError('Existing repair protocol changed')
        atomic(path,local)
        status=output/'training_status.json'
        if status.exists() and json.loads(status.read_text()).get('status')=='complete':continue
        command=ddp+['--module','emotion_ssm.train.staged_v34.trainer','--config',str(path),'--execution','compiled']
        if (output/'last.pt').exists():command+=['--resume',str(output/'last.pt')]
        call(name,command,output/'train.nohup.log')
        # A quality gate controls deployment, not continuation of a finite research comparison.
        validation=json.loads((output/'validation.json').read_text())
        atomic(output/'quality_status.json',dict(deployable=validation['acceptance']['gate_passed'],
            checks=validation['acceptance']['checks'],research_training_complete=True))
    atomic(root/'pipeline_status.json',dict(status='complete',time=time.time(),physical_gpus=[2,3]))
    previous=root/'previous_suite.json'
    if previous.exists():
        saved=json.loads(previous.read_text())
        if saved['status']=='queued_after_repair':
            require_free_gpus()
            env=os.environ.copy();env.update(saved['environment'])
            with Path(saved['log']).open('a') as log:
                child=subprocess.Popen(saved['command'],cwd=saved['cwd'],env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            saved.update(status='resumed_after_repair',pid=child.pid,resumed=time.time());atomic(previous,saved)


if __name__=='__main__':main()
