"""Sequential, two-GPU execution of all three authorized experiment steps."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)


def run_reporting(root,name=None,stage=None):
    """Record report failures so a finished trainer cannot leave a stale status."""
    preflight=name is None;phase='preflight_reporting' if preflight else 'reporting'
    log_path=root/'report_preflight.log' if preflight else root/name/'report.log'
    command=[sys.executable,'-B',str(Path(__file__).with_name('report_staged_v33_suite.py'))]
    command+=['--check-environment'] if preflight else ['--root',str(root)]
    with log_path.open('a',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        state=dict(status=phase,suite_pid=os.getpid(),child_pid=process.pid,
            experiment=name or '__suite_preflight__',stage=0 if preflight else stage,
            time=time.time(),log=str(log_path),command=command)
        atomic(root/'pipeline_status.json',state)
        result=process.wait()
    state.update(status=('failed_' if result else 'completed_')+phase,exit_code=result,time=time.time())
    atomic(root/'pipeline_status.json',state)
    if result:raise SystemExit(result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base-config',required=True)
    parser.add_argument('--initial-checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--execution',default='compiled');args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise RuntimeError('This suite is authorized for physical GPU2,3 only')
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(root/'suite.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('This suite already has an active owner')
    run_reporting(root)
    base=json.loads(Path(args.base_config).read_text());base['paths'].update(dynamics_checkpoint=str(Path(args.initial_checkpoint).resolve()),resume='')
    base['train'].update(global_chunks_per_step=32,seed=6666,amp=False,cpu_threads=4)
    base['staged'].update(calibration_steps=1000,fixed_steps=4000,joint_rounds=4,joint_steps_per_round=250,
        readapt_steps_per_round=1000,bank_refresh_steps=1000,bank_dialogues_per_domain=64,
        calibration_dialogues_per_domain=16,validation_dialogues_per_domain=8,validation_every=1000,
        checkpoint_every=250,archive_every=1000,gradient_every=1000,update_refresh_steps=250,
        cache_batch=64,prediction_batch=128,gold_query_budget=64,tbptt_events=32,
        objective_mode='weighted_joint',endpoint_label_weight=1.,affine=False,enable_partner=True,schedule='constant',
        acceptance_tolerance=.02,semantic_absolute_tolerance=.02,minimum_fixed_improvement=.005,
        deployment_domains=[2],long_horizon_min=16,missing_seed=88173)
    jobs=[('01_protocol_diagnostic',1,dict(calibration_steps=100,fixed_steps=400,joint_rounds=2,
                joint_steps_per_round=50,readapt_steps_per_round=200,bank_refresh_steps=200,
                validation_every=250,gradient_every=250,archive_every=250)),
          ('02_vector_only',2,dict(objective_mode='vector_only',endpoint_label_weight=0.)),
          ('02_joint025',2,dict(endpoint_label_weight=.25)),
          ('02_joint100',2,dict(endpoint_label_weight=1.)),
          ('03_affine_joint100',3,dict(affine=True,endpoint_label_weight=1.)),
          ('03_affine_self_only',3,dict(affine=True,enable_partner=False,endpoint_label_weight=1.))]
    suite=dict(created=time.time(),pid=os.getpid(),initial_checkpoint=base['paths']['dynamics_checkpoint'],
        protocol='v3.3-query-balanced-endpoints-missing-replay-v1',physical_gpus=[2,3],
        jobs=[dict(name=n,step=step,overrides=change) for n,step,change in jobs],
        controls='same initialization, query budgets, manifests, seeds, learning rates and trainable core',
        research_quality_gate='quality failures are reported, never silently converted to deployment approval')
    existing=root/'suite.json'
    if existing.exists():
        old=json.loads(existing.read_text())
        if old['initial_checkpoint']!=suite['initial_checkpoint'] or old['jobs']!=suite['jobs']:
            raise ValueError('Existing suite protocol differs; use a new output directory')
    else:atomic(existing,suite)
    for name,stage,overrides in jobs:
        output=root/name;output.mkdir(exist_ok=True);config=copy.deepcopy(base)
        config['paths']['output']=str(output);config['staged'].update(overrides)
        config_path=output/'requested_config.json'
        if config_path.exists() and json.loads(config_path.read_text())!=config:
            raise ValueError('Requested experiment configuration changed')
        atomic(config_path,config)
        status_file=output/'training_status.json'
        complete=status_file.exists() and json.loads(status_file.read_text()).get('status')=='complete'
        if not complete:
            command=[sys.executable,'-u','-B','-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                '--module','emotion_ssm.train.staged_v33.trainer','--config',str(config_path),'--execution',args.execution]
            if (output/'last.pt').exists():command.extend(['--resume',str(output/'last.pt')])
            with (output/'train.nohup.log').open('a',encoding='utf-8') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                atomic(root/'pipeline_status.json',dict(status='training',suite_pid=os.getpid(),child_pid=process.pid,
                    experiment=name,stage=stage,started=time.time(),log=str(output/'train.nohup.log'),command=command))
                result=process.wait()
            if result:
                atomic(root/'pipeline_status.json',dict(status='failed',experiment=name,stage=stage,exit_code=result,time=time.time()))
                raise SystemExit(result)
        # All steps include a final causal coupling and long-integration audit.
        if not (output/'mechanisms.json').exists():
            command=[sys.executable,'-u','-B',str(Path(__file__).with_name('evaluate_staged_v33_mechanisms.py')),
                '--checkpoint',str(output/'best_vector.pt'),'--output',str(output/'mechanisms.json'),'--execution',args.execution]
            with (output/'mechanisms.log').open('a') as log:
                atomic(root/'pipeline_status.json',dict(status='mechanism_evaluation',experiment=name,stage=stage,time=time.time()))
                result=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
            if result:
                atomic(root/'pipeline_status.json',dict(status='failed_mechanisms',experiment=name,stage=stage,exit_code=result,time=time.time()))
                raise SystemExit(result)
        run_reporting(root,name,stage)
    atomic(root/'pipeline_status.json',dict(status='complete',time=time.time(),experiments=len(jobs),stages=3))


if __name__=='__main__':main()
