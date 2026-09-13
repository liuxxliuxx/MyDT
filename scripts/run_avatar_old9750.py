"""Train a dyadic Avatar from the verified old9750 dynamics, alongside other jobs."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from emotion_ssm.config_v3 import default_config, validate_config, write_config
from emotion_ssm.train.staged_dynamics_support import weight_digest
from emotion_ssm.utils.checkpoint_v3 import read_checkpoint, manifest_provenance


GPUS = ['GPU-5b497823-4a84-bde7-5670-2172ee96245d','GPU-9117039b-5194-5d46-d4a7-088ff06ce552']
EXPECTED_WEIGHTS = 'b1f656f069b3b86cdb18f9c59021d84746b606c84a93dd33d5e76d1283db860e'


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def run(args):
    root=Path(args.project_root).resolve();out=Path(args.output).resolve()
    if root not in out.parents or out.parent.name!='runs':
        raise ValueError('Avatar output must be a new direct child of project/runs')
    code=Path(__file__).resolve().parents[1]
    if (out/'pipeline_status.json').exists():raise ValueError('Refusing a duplicate training pipeline')
    out.mkdir(parents=True,exist_ok=True)
    source=root/'runs/v3_2_1_staged_20260910_gpu23/formal/best.pt'
    snapshot=out/'initial_dynamics_step009750.pt'
    shutil.copyfile(source,snapshot)
    payload=read_checkpoint(snapshot)
    if payload['global_step']!=9750 or weight_digest(payload['models'])!=EXPECTED_WEIGHTS:
        raise ValueError('Dynamics checkpoint differs from the evaluated old9750 weights')
    if sha256(source)!=sha256(snapshot):raise ValueError('Source checkpoint changed during snapshot')
    cfg=default_config()
    cfg['observer']=copy.deepcopy(payload['construction']['observer'])
    cfg['state']=copy.deepcopy(payload['construction']['state'])
    cfg['state'].pop('observation_dim')
    cfg['data']=copy.deepcopy(payload['config']['data'])
    cfg['data']['dualtalk_raw']=str(root/'datasets/dualtalk')
    cfg['data']['dualtalk_tokens']=str(root/'artifacts/v3_1_retrain_20260908_gpu23/dualtalk_tokens')
    cfg['data']['streaming_source']=copy.deepcopy(payload['construction']['adapter_source_binding']['feature_source'])
    cfg['generation'].update(variant='dyadic',train_observer=False,train_state=False,
        dynamics_execution='optimized',gradient_checkpointing=True,offload_extractor=True)
    cfg['train'].update(seed=6666,max_steps=30000,global_chunks_per_step=32,tbptt_seconds=32,
        lr=1e-4,weight_decay=.01,amp=True,clip_grad=5.,coordinate_weight=0.,future_weight=0.,masked_weight=0.,
        validation_max_dialogues=0,validate_every=1000,log_every=10,cpu_threads=2,
        experiment_scope='old9750_frozen_observer_and_dynamics_generator_training')
    cfg['paths'].update(dynamics_checkpoint=str(snapshot),baseline=str(root/'model/dualtalk_baseline.pth'),
        observation_checkpoint='',resume='',output=str(out/'formal'))
    validate_config(cfg)
    actual=manifest_provenance(cfg)
    for path,metadata in payload['provenance'].items():
        if actual[path]['sha256']!=metadata['sha256']:raise ValueError('Upstream dataset manifest changed: '+path)
    manifests={path:dict(sha256=entry['sha256'],split_counts={k:len(v) for k,v in entry['splits'].items()})
               for path,entry in actual.items()}
    receipt=dict(reference=str(source),snapshot=str(snapshot),source_step=9750,weight_hash=EXPECTED_WEIGHTS,
        checkpoint_sha256=sha256(snapshot),physical_gpus=[2,3],gpu_uuids=GPUS,
        variant='dyadic',trainable='generator backbone and FiLM; original audio convolutions frozen',
        frozen='observer, adapters, label heads, teacher, complete dynamics',
        upstream_optimizer_restored=False,seed=6666,formal_steps=30000,global_valid_new_blocks=32,
        clock='25 new frames each second, at most 3 seconds generator history',
        selection='full validation generation_total',token_manifests=manifests,
        initialization_generator=str(root/'model/dualtalk_baseline.pth'),
        initial_existing_jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True))
    write_json(out/'experiment_protocol.json',receipt)
    del payload
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(GPUS),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(code),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    def stage(name, module, arguments):
        command=[sys.executable,'-u','-B','-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                 '--module',module,*arguments]
        log=out/(name+'.log')
        with log.open('wb') as handle:
            child=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
            write_json(out/'pipeline_status.json',dict(status='running',stage=name,pid=os.getpid(),child_pid=child.pid,
                started=time.time(),command=command,log=str(log),physical_gpus=[2,3]))
            print(json.dumps(dict(stage=name,pid=child.pid,log=str(log))),flush=True)
            result=child.wait()
        if result:
            write_json(out/'pipeline_status.json',dict(status='failed',stage=name,returncode=result,log=str(log)))
            raise RuntimeError('Avatar stage failed: '+name)
    smoke=copy.deepcopy(cfg)
    smoke['paths']['output']=str(out/'smoke')
    smoke['train'].update(max_steps=2,validate_every=2,log_every=1,validation_max_dialogues=2)
    write_config(out/'smoke_config.json',smoke);write_config(out/'formal_config.json',cfg)
    stage('smoke','emotion_ssm.train.generation_v3',['--config',str(out/'smoke_config.json')])
    checkpoint=read_checkpoint(out/'smoke/last.pt')
    system=checkpoint['models']['system']
    condition={name:{key[len(prefix):]:value for key,value in system.items() if key.startswith(prefix)}
        for name,prefix in [('observer','observer.'),('teacher','teacher.'),('state','state_model.')]}
    if weight_digest(condition)!=EXPECTED_WEIGHTS:raise ValueError('Smoke changed the frozen old9750 condition')
    metrics=[json.loads(line) for line in (out/'smoke/train_metrics.jsonl').read_text().splitlines()]
    if checkpoint['global_step']!=2 or checkpoint['run_state']['world_size']!=2:raise ValueError('Incomplete double-GPU smoke')
    for row in metrics:
        if row['global_valid_blocks']!=32 or row['generator_grad_norm']<=0 or row['state_grad_norm'] or row['observer_grad_norm']:
            raise ValueError('Unexpected smoke gradient permissions or optimization budget')
    write_json(out/'smoke_gate.json',dict(passed=True,steps=2,world_size=2,global_valid_blocks=32,
        frozen_condition_hash=weight_digest(condition),generator_gradient_positive=True,
        validation=checkpoint['metrics'],max_peak_reserved_mib=max(row['peak_reserved_mib'] for row in metrics)))
    del checkpoint,system,condition
    stage('formal','emotion_ssm.train.generation_v3',['--config',str(out/'formal_config.json')])
    for split in ('test','ood'):
        stage('evaluate_'+split,'emotion_ssm.evaluate_v3',['--checkpoint',str(out/'formal/best.pt'),
              '--split',split,'--output',str(out/'formal'/f'{split}.json')])
    write_json(out/'pipeline_status.json',dict(status='complete',finished=time.time(),formal_steps=30000,
        best_checkpoint=str(out/'formal/best.pt'),evaluated_splits=['test','ood']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
