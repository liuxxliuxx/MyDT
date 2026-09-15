"""v3.4: query-balanced staged training with fixed and replayed deployment controls."""
import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
import torch
import torch.distributed as dist
from emotion_ssm.config_v3 import read_config,write_config
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.train.dynamics_v3 import DialogueCollection,_distributed_device
from emotion_ssm.train.dynamics_staged_v3 import DEFAULTS as OLD_DEFAULTS,stage_plan,source_balanced_indices,cpu_models
from emotion_ssm.train.staged_dynamics_support import lock_snapshot,assert_locked,weight_digest
from emotion_ssm.utils.checkpoint import capture_rng_state,restore_rng_state
from emotion_ssm.utils.checkpoint_v3 import read_checkpoint,save_checkpoint,manifest_provenance
from . import REVISION
from .data import FeatureViews,build_bank,QuerySampler,origin_drift
from .semantics import class_audit
from emotion_ssm.train.staged_v33.missing import digest
from .optimization import PersistentOptimizers
from .objectives import loss,supports
from emotion_ssm.train.staged_v33.baselines import fit,refit_output_calibration
from .evaluation import evaluate,deployment_gate,semantic_score
from .diagnostics import gradient_diagnostic

DEFAULTS=dict(OLD_DEFAULTS,missing_seed=88173,tbptt_events=32,gold_query_budget=64,
    objective_mode='weighted_joint',task_weights=dict(ce=1.,vad=1.,intensity=1.),
    schedule='constant',affine=False,affine_max_offset=.5,enable_partner=True,
    calibration_dialogues_per_domain=16,baseline_ridge=.001,validation_every=1000,
    checkpoint_every=250,archive_every=1000,gradient_every=1000,
    update_refresh_steps=250,deployment_domains=[2],long_horizon_min=16,
    semantic_absolute_tolerance=.02,class_weight_power=.5,class_weight_cap=3.,
    semantic_keep_weight=.1,neutral_margin_weight=0.,neutral_margin=.2,
    flow_replay_origins=True,replay_dialogue_batch=8,origin_refresh_steps=250,
    origin_probe_steps=100,origin_drift_mse=1e-5,origin_component_drift_mse=2e-5,
    origin_drift_class_flip=.02,long_semantic_tolerance=.01,vector_guard_tolerance=.05,
    minimum_long_f1_improvement=.005,train_protocol_cycle=['clean','train_mixed'])


def barrier():
    if dist.is_initialized():dist.barrier()


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def run(request,execution_override=None,stop_after=None):
    config=copy.deepcopy(request);device,rank,size=_distributed_device(config)
    output=Path(config['paths']['output']);output.mkdir(parents=True,exist_ok=True)
    resume=config['paths'].get('resume');source=read_checkpoint(resume or config['paths']['dynamics_checkpoint'])
    if resume:
        if source['kind']!='dynamics_staged_v34' or source['config']['train'].get('staged_revision')!=REVISION:
            raise ValueError('Old optimizer semantics cannot resume v3.4; use explicit initialization')
        if source['provenance']!=manifest_provenance(config):raise ValueError('Resume source/split provenance changed')
        if source['run_state']['world_size']!=size:raise ValueError('Exact resume needs the same rank budget')
        for key in ('history_age_sampling','train_protocol_cycle','block_gap_period','block_gap_seconds','full_origin_replay'):
            if config.get('staged',{}).get(key,DEFAULTS.get(key))!=source['config'].get('staged',{}).get(key,DEFAULTS.get(key)):
                raise ValueError('Long-history protocol changed; initialize a new experiment')
        config=copy.deepcopy(source['config']);config['paths']['output']=str(output)
    settings={**DEFAULTS,**config.get('staged',{})};config['staged']=settings
    if settings['objective_mode']!='weighted_joint' or settings['endpoint_label_weight']<=0:
        raise ValueError('v3.4 repair requires positive true-endpoint semantic supervision')
    if not settings['flow_replay_origins']:raise ValueError('v3.4 requires current detached flow-training origins')
    if min(settings['origin_refresh_steps'],settings['origin_probe_steps'])<1:raise ValueError('Origin refresh/probe interval must be positive')
    if settings['schedule'] not in ('constant','cosine'):raise ValueError('Unknown learning-rate schedule')
    torch.set_num_threads(int(config['train'].get('cpu_threads',4)))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed=int(config['train']['seed']);torch.manual_seed(seed);random.seed(seed)
    observer=TokenObserver(source['construction']['observer']).to(device).eval()
    teacher=TokenObserver(source['construction']['observer']).to(device).eval().requires_grad_(False)
    construction=copy.deepcopy(source['construction']['state'])
    if settings['affine']:
        construction.update(flow_kind='adaptive_affine_dyadic_v2',affine_max_offset=settings['affine_max_offset'])
    core=UnifiedEmotionStateCore.from_config(construction).to(device)
    models=dict(observer=observer,teacher=teacher,state=core)
    for name,model in models.items():
        weights=source['models'].get(name,source['models']['observer'] if name=='teacher' else {})
        missing,unexpected=model.load_state_dict(weights,strict=False)
        allowed={'adaptive_flow.affine_fast','adaptive_flow.affine_slow'} if name=='state' and settings['affine'] and not resume else set()
        if set(missing)-allowed or unexpected:raise ValueError(f'Incompatible initialization: {name}, {missing}, {unexpected}')
    if dist.is_initialized():
        for model in models.values():
            for value in model.state_dict().values():dist.broadcast(value,0)
    execution=execution_override or config.get('execution',{}).get('mode','reference')
    core.configure_execution(execution)
    optimizers=PersistentOptimizers(observer,core,settings)
    initial_models=source['run_state']['initial_models'] if resume else cpu_models(models)
    locks=source['run_state']['locks'] if resume else lock_snapshot(core)
    frozen=weight_digest({'observer':observer,'teacher':teacher},exclude_heads=True)
    config['train'].update(staged_revision=REVISION,amp=False,max_steps=sum(n for _,n in stage_plan(settings)))
    config['state']={k:v for k,v in core.get_config().items() if k!='observation_dim'}
    config['observer']=observer.construction();config['execution']=dict(mode=execution,revision=REVISION)
    config['paths']['resume']=''
    if rank==0:write_config(output/'config.json',config)
    training=DialogueCollection(config['data']['token_roots'],'train');validation=DialogueCollection(config['data']['token_roots'],'val')
    calibration_ids=source_balanced_indices(training,settings['calibration_dialogues_per_domain'],seed+22271)
    validation_ids=source_balanced_indices(validation,settings['validation_dialogues_per_domain'],seed+17011)
    cohort=dict(train=[training.identity(i) for i in calibration_ids],validation=[validation.identity(i) for i in validation_ids],
        seed=seed,provenance=manifest_provenance(config));cohort['sha256']=digest(cohort)
    if resume and source['run_state']['cohort']!=cohort:raise ValueError('Fixed calibration cohort changed')
    if rank==0:atomic_json(output/'calibration_manifest.json',cohort)
    views=FeatureViews(observer,teacher,config['paths']['feature_cache'],device,settings)
    began=time.monotonic();global_step=source['global_step'] if resume else 0
    def emit(event,**values):
        record=dict(event=event,rank=rank,step=global_step,elapsed=time.monotonic()-began,**values)
        with (output/f'rank{rank}.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        if rank==0:print(json.dumps(record,ensure_ascii=False,allow_nan=False),flush=True)
    def bank(collection,indices,label,protocol='clean',expected=None):
        path=output/(label+'.pt');value=None
        signature=digest(dict(weights=weight_digest({'observer':observer,'state':core}),
            construction=core.get_config(),cohort=[collection.identity(i) for i in indices],protocol=protocol,
            partner=settings['enable_partner'],missing_seed=settings['missing_seed'],
            period=settings['block_gap_period'],gap=settings['block_gap_seconds'],
            stride=settings['origin_stride'],tbptt=settings['tbptt_events'],horizons=config['train']['forecast_seconds'],revision=REVISION))
        if rank==0:
            if path.exists():
                value=torch.load(path,map_location='cpu',weights_only=False)
                if value.get('signature')!=signature:value=None
            if value is None:
                last=[time.monotonic()]
                def progress(done,total,origins):
                    if done==total or time.monotonic()-last[0]>25:
                        emit('bank_progress',label=label,protocol=protocol,dialogues=done,total=total,origins=origins);last[0]=time.monotonic()
                value=build_bank(observer,core,views,collection,indices,config['train']['forecast_seconds'],
                    settings['origin_stride'],protocol,settings['enable_partner'],progress)
                value['signature']=signature
                temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)
        barrier()
        if value is None:value=torch.load(path,map_location='cpu',weights_only=False)
        if value['signature']!=signature:raise ValueError('Origin producer differs between ranks/resume')
        if expected is not None and value['producer']!=expected:raise ValueError('Origin replay identity changed')
        if settings.get('history_age_sampling'):
            if 'history_age' not in value:raise ValueError('Rebuild the origin bank to include physical history ages')
            if rank==0:
                from emotion_ssm.utils.history_coverage import bank_coverage
                atomic_json(output/(label+'_history_coverage.json'),bank_coverage(value,settings['history_age_sampling'].get('age_edges',[0,8,16,32,64])))
        return value
    saved=cpu_models(models)
    for name,model in models.items():model.load_state_dict(initial_models[name])
    reference=bank(validation,validation_ids,'fixed_validation')
    calibration=bank(training,calibration_ids,'fixed_calibration')
    # Initial calibration fits stay fixed across every block/checkpoint.
    fixed_fit=fit(calibration,core,settings,device)
    for name,model in models.items():model.load_state_dict(saved[name])
    del saved
    initial_metrics=source['run_state']['initial_metrics'] if resume else None
    best_vector=source['run_state']['best_vector'] if resume else math.inf
    best_deploy=source['run_state']['best_deploy'] if resume else math.inf
    best_semantic=source['run_state']['best_semantic'] if resume else math.inf
    if resume:
        optimizers.load_state_dict(source['optimizer'])
        restore_rng_state(source['run_state']['ranks'][rank]['rng'])
    def validate(phase):
        assert_locked(core,locks)
        if weight_digest({'observer':observer,'teacher':teacher},exclude_heads=True)!=frozen:
            raise RuntimeError('Frozen affect coordinates or teacher changed')
        metrics=dict(step=global_step,phase=phase,candidate_hash=weight_digest({'observer':observer,'state':core}),
            correction_parameters_unchanged=True,cohort_sha256=cohort['sha256'])
        fixed_current_fit=refit_output_calibration(fixed_fit,calibration,core,settings,device)
        metrics['fixed']=evaluate(observer,core,reference,fixed_current_fit,settings,device)
        for protocol,view in [('live','clean'),('update_gap','update_gap'),('sensor_gap','sensor_gap')]:
            train=bank(training,calibration_ids,'calibration_'+protocol,view)
            current_fit=fit(train,core,settings,device)
            current=bank(validation,validation_ids,'validation_'+protocol,view)
            metrics[protocol]=evaluate(observer,core,current,current_fit,settings,device)
            if metrics[protocol]['query_hash']!=metrics['fixed']['query_hash']:
                raise ValueError('Fixed/live/gap target keys or valid counts changed')
            del train,current
        metrics['acceptance']=deployment_gate(metrics,initial_metrics or metrics,settings)
        if rank==0:
            atomic_json(output/'validation.json',metrics);atomic_json(output/f'validation_step{global_step:06d}.json',metrics)
        emit('validation',phase=phase,metrics_file=f'validation_step{global_step:06d}.json',
            fixed=metrics['fixed']['methods'],live=metrics['live']['methods'],
            update_gap=metrics['update_gap']['methods']['learned'],sensor_gap=metrics['sensor_gap']['methods']['learned'],
            semantic=metrics['live']['semantic']['learned'],acceptance=metrics['acceptance'])
        return metrics
    emit('initialized',revision=REVISION,world_size=size,physical_gpu_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
        execution=execution,initial_checkpoint_step=source['global_step'],plan=stage_plan(settings),
        vector_query_budget=config['train']['global_chunks_per_step'],gold_task_query_budget=settings['gold_query_budget'],
        objective=settings['objective_mode'],label_weight=settings['endpoint_label_weight'],affine=settings['affine'],
        enable_partner=settings['enable_partner'],cohort_sha256=cohort['sha256'])
    if initial_metrics is None:initial_metrics=validate('initial')
    metrics=source.get('metrics',initial_metrics) if resume else initial_metrics
    resume_block=source['run_state']['block'] if resume else -1
    resume_step=source['run_state']['block_step'] if resume else 0
    budget=int(config['train']['global_chunks_per_step']);gold_budget=int(settings['gold_query_budget'])
    if budget<size or gold_budget<size:raise ValueError('Global query budgets must cover all ranks')
    def save(names,block,block_step,bank_models,train_bank,sampler,refresh_step):
        ranks=[None]*size;local=dict(rng=capture_rng_state(),sampler=sampler.state_dict())
        if size>1:dist.all_gather_object(ranks,local)
        else:ranks=[local]
        run_state=dict(world_size=size,initial_models=initial_models,initial_metrics=initial_metrics,
            locks=locks,cohort=cohort,best_vector=best_vector,best_deploy=best_deploy,best_semantic=best_semantic,
            block=block,block_step=block_step,bank_producer_models=bank_models,
            bank_producer=train_bank['producer'],refresh_step=refresh_step,ranks=ranks,
            initial_construction=source['construction'],sampler_protocol='assigned-domain-horizon-task-v1',
            class_audit=train_bank['class_audit'],train_protocol=train_bank['protocol'])
        if rank==0:
            for name in names:
                save_checkpoint(output/name,models,config,{**source['construction'],'observer':observer.construction(),'state':core.get_config()},
                    'dynamics_staged_v34',step=global_step,optimizer=optimizers,metrics=metrics,run_state=run_state)
            atomic_json(output/'training_status.json',dict(step=global_step,total=config['train']['max_steps'],
                block=block,block_step=block_step,phase=stage_plan(settings)[block][0],
                optimizer_updates=optimizers.updates,best_vector=None if not math.isfinite(best_vector) else best_vector,
                best_deploy=None if not math.isfinite(best_deploy) else best_deploy,
                best_semantic=None if not math.isfinite(best_semantic) else -best_semantic,
                gate_passed=metrics['acceptance']['gate_passed'],status='running',revision=REVISION))
        barrier()
    for block,(phase,length) in enumerate(stage_plan(settings)):
        if block<resume_block:continue
        continuing=bool(resume and block==resume_block);begin=resume_step if continuing else 0
        if begin==length:continue
        online=phase in ('calibration','joint')
        cycle=settings['train_protocol_cycle'];protocol=cycle[block%len(cycle)]
        selected=source_balanced_indices(training,settings['bank_dialogues_per_domain'],seed+9001,block)
        if continuing:
            refresh_step=source['run_state']['refresh_step'];bank_models=source['run_state']['bank_producer_models']
            current=cpu_models(models)
            for name,model in models.items():model.load_state_dict(bank_models[name])
            if source['run_state']['train_protocol']!=protocol:raise ValueError('Resume masking protocol changed')
            train_bank=bank(training,selected,f'origins_block{block}',protocol=protocol,expected=source['run_state']['bank_producer'])
            for name,model in models.items():model.load_state_dict(current[name])
        else:
            refresh_step=0;bank_models=cpu_models(models)
            train_bank=bank(training,selected,f'origins_block{block}',protocol=protocol)
        train_bank['class_audit']=class_audit(train_bank,settings)
        if rank==0:atomic_json(output/f'class_audit_block{block}.json',train_bank['class_audit'])
        sampler=QuerySampler(train_bank,seed+block*101,chronological=online,age_sampling=settings.get('history_age_sampling'))
        if continuing:sampler.load_state_dict(source['run_state']['ranks'][rank]['sampler'])
        support=supports(sampler,device);optimizers.phase(phase)
        emit('phase_started',phase=phase,block=block,length=length,origins=len(train_bank['keys']),
            producer=train_bank['producer'],coverage=train_bank['coverage'],support={k:v.cpu().tolist() for k,v in support.items()},
            tbptt_protocol='current-replay-update-only' if online else 'current-replay-detached-flow-only',
            train_protocol=protocol,class_weight_fit=train_bank['class_audit']['scope'],
            optimizer_updates=optimizers.updates)
        seconds=[]
        for block_step in range(begin+1,length+1):
            refresh=block_step>refresh_step+settings['origin_refresh_steps']
            if not refresh and block_step>1 and (block_step-1)%settings['origin_probe_steps']==0:
                report=[None]
                if rank==0:
                    report[0]=origin_drift(observer,core,views,training,selected,train_bank,
                        config['train']['forecast_seconds'],settings['origin_stride'],protocol,settings)
                if size>1:dist.broadcast_object_list(report,src=0)
                refresh=report[0]['refresh'];emit('origin_drift',block=block,phase=phase,diagnostic=report[0])
            if refresh:
                bank_models=cpu_models(models);refresh_step=block_step-1
                previous_sampler=sampler.state_dict()
                train_bank=bank(training,selected,f'origins_block{block}',protocol=protocol)
                train_bank['class_audit']=class_audit(train_bank,settings)
                sampler=QuerySampler(train_bank,seed+block*101,chronological=online,age_sampling=settings.get('history_age_sampling'));sampler.load_state_dict(previous_sampler)
                support=supports(sampler,device)
                emit('prefix_refreshed',block=block,block_step=block_step,producer=train_bank['producer'],
                    mask_plan_hash=digest([p['sha256'] for p in train_bank['plans']]))
            started=time.monotonic();optimizers.phase(phase);optimizers.zero_grad()
            vectors,gold=sampler.sample(budget,gold_budget,rank,size)
            total,parts,stats=loss(core,observer,train_bank,vectors,gold,settings,support,online)
            if not bool(torch.isfinite(total)):raise FloatingPointError('Non-finite global training objective')
            total.backward();norm=optimizers.step();global_step+=1
            seconds.append(time.monotonic()-started)
            if global_step%settings['log_every']==0 or block_step==1:
                emit('train',phase=phase,block=block,block_step=block_step,losses=stats,grad_norm=norm,
                    mean_step_seconds=sum(seconds)/len(seconds),sampling=sampler.diagnostics(),
                    optimizer_updates=optimizers.updates.copy(),learning_rates={n:o.param_groups[0]['lr'] for n,o in optimizers.optimizers.items()})
                seconds=[]
            if global_step%settings['gradient_every']==0:
                # A diagnostic must not discard already completed optimizer steps.
                # This snapshot's metrics retain their own (possibly older) step.
                save(['last.pt'],block,block_step,bank_models,train_bank,sampler,refresh_step)
                diagnostic=gradient_diagnostic(observer,core,train_bank,settings,budget,gold_budget,rank,size,seed+33971,online)
                emit('gradient_diagnostic',diagnostic=diagnostic)
            names=[]
            if global_step%settings['validation_every']==0 or block_step==length:
                metrics=validate(phase);score=metrics['fixed']['methods']['learned']['macro_mse']
                is_best=score<best_vector;metrics['is_best']=is_best
                if is_best:best_vector=score;names.append('best_vector.pt')
                deploy_score=semantic_score(metrics,settings)
                if deploy_score<best_semantic:
                    best_semantic=deploy_score;names.append('best_semantic.pt')
                if metrics['acceptance']['gate_passed'] and deploy_score<best_deploy:
                    best_deploy=deploy_score;names.append('best_deploy.pt')
                emit('selection',is_best=is_best,gate_passed=metrics['acceptance']['gate_passed'],best_vector=best_vector,
                    best_deploy=None if not math.isfinite(best_deploy) else -best_deploy,
                    best_semantic=None if not math.isfinite(best_semantic) else -best_semantic)
            if global_step%settings['archive_every']==0:names.append(f'step{global_step:06d}.pt')
            if names or global_step%settings['checkpoint_every']==0 or block_step==length or global_step==stop_after:
                save(['last.pt',*names],block,block_step,bank_models,train_bank,sampler,refresh_step)
            if global_step==stop_after:return dict(step=global_step,stopped_for_test=True)
        resume=False
    if rank==0:
        atomic_json(output/'training_status.json',dict(status='complete',step=global_step,total=config['train']['max_steps'],
            best_vector=best_vector,best_deploy=None if not math.isfinite(best_deploy) else best_deploy,
            best_semantic=None if not math.isfinite(best_semantic) else -best_semantic,
            gate_passed=metrics['acceptance']['gate_passed'],revision=REVISION))
    emit('complete',best_vector=best_vector,gate_passed=metrics['acceptance']['gate_passed'])
    return dict(step=global_step,best_vector=best_vector)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--resume')
    parser.add_argument('--execution',choices=['reference','optimized','compiled']);args=parser.parse_args()
    config=read_config(args.config)
    if args.resume:config['paths']['resume']=args.resume
    try:run(config,args.execution)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
