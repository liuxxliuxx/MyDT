import copy
import contextlib
import io
import json
import socket
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_staged_dynamics_v3 import setup_models
from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.train.staged_v33.data import FeatureViews,build_bank,QuerySampler,replay_origins
from emotion_ssm.train.staged_v33.missing import make_plan,mask_sensor_features
from emotion_ssm.train.staged_v33.objectives import loss,supports
from emotion_ssm.train.staged_v33.optimization import PersistentOptimizers,global_mean,synchronize
from emotion_ssm.train.staged_v33.baselines import fit
from emotion_ssm.train.staged_v33.evaluation import evaluate,deployment_gate
from emotion_ssm.train.staged_v33.trainer import DEFAULTS,run
from emotion_ssm.train.staged_v33.diagnostics import gradient_diagnostic
from emotion_ssm.utils.checkpoint_v3 import read_checkpoint,save_checkpoint,require_training_revision


@pytest.fixture(autouse=True)
def small_threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def case(tmp_path,stride=1):
    config,o,t,c,_,collection=setup_models(tmp_path)
    settings=dict(DEFAULTS,cache_batch=4,prediction_batch=4,origin_stride=stride,tbptt_events=2,
                  block_gap_period=4,block_gap_seconds=1,gold_query_budget=6)
    cache=FeatureViews(o,t,tmp_path/'views',torch.device('cpu'),settings)
    bank=build_bank(o,c,cache,collection,[0,1],[1,2],stride)
    return config,o,t,c,collection,cache,bank,settings


def test_gold_queries_do_not_depend_on_vector_stride(tmp_path):
    _,o,t,c,col,cache,bank,s=case(tmp_path,stride=2)
    dense=build_bank(o,c,cache,col,[0,1],[1,2],1)
    assert bank['coverage']['cached_queries']==dense['coverage']['cached_queries']==4
    assert bank['valid'].sum()<dense['valid'].sum()
    assert any(not bool(bank['valid'][row].any()) for row,q in bank['endpoints'])
    # No future teacher affect is required for a real human label query.
    for index in (0,1):cache.get(col,index)['valid'].zero_()
    label_only=build_bank(o,c,cache,col,[0,1],[1,2],2)
    assert not label_only['valid'].any() and len(label_only['endpoints'])==4


def test_sampling_balances_actual_queries_and_restores_rng(tmp_path):
    _,_,_,_,_,_,bank,_=case(tmp_path)
    bank['domain']=torch.tensor([0]*2+[1]*5+[2])
    sampler=QuerySampler(bank,19)
    for _ in range(3):sampler.sample(32,32)
    assert sampler.domain_counts=={0:32,1:32,2:32}
    state=copy.deepcopy(sampler.state_dict());a=sampler.sample(32,32)
    other=QuerySampler(bank,99);other.load_state_dict(state)
    assert other.sample(32,32)==a
    assert sum(sampler.domain_counts.values())==128


def test_sensor_gap_never_recovers_raw_history_and_replay_matches(tmp_path):
    _,o,t,c,col,cache,_,s=case(tmp_path)
    clean=cache.get(col,0);update=cache.get(col,0,'update_gap');sensor=cache.get(col,0,'sensor_gap')
    assert update['plan']!=sensor['plan']
    raw=col[0]['packets'][3]['roles'][0]
    masked=mask_sensor_features(raw,0,sensor['plan'])
    assert not masked['audio_mask'].any() and not masked['prosody_mask'].any()
    # Audio acquired during a real outage remains unavailable in a later context window.
    later=copy.deepcopy(raw);later['now']=torch.tensor(5.);later['dt']=torch.tensor(1.)
    assert not mask_sensor_features(later,0,sensor['plan'])['audio_mask'].any()
    for protocol in ('clean','update_gap','sensor_gap','train_mixed'):
        bank=build_bank(o,c,cache,col,[0,1],[1,2],1,protocol)
        actual=replay_origins(o,c,bank,list(range(len(bank['keys']))),'cpu')
        torch.testing.assert_close(actual.z,bank['states'].z,atol=2e-6,rtol=2e-5)
    assert make_plan('d',99,'train_mixed',55,period=17,gap=5)==make_plan('d',99,'train_mixed',55,period=17,gap=5)


def test_sensor_gap_masks_shared_text_by_sender(tmp_path):
    _,_,_,_,col,_,_,_=case(tmp_path)
    features=col[0]['packets'][3]['roles'][1]
    features['text_tokens']=torch.ones(2,4);features['text_mask']=torch.ones(2,dtype=torch.bool)
    features['text_roles']=torch.tensor([0,1]);features['text_times']=torch.tensor([3.5,3.5])
    features['text_fresh_mask']=torch.ones(2,dtype=torch.bool)
    plan=dict(intervals=[dict(kind='sensor_gap',start=3.,end=4.,roles=[0],modes=['T'])])
    value=mask_sensor_features(features,1,plan)
    assert value['text_mask'].tolist()==[True,False]


def test_objective_uses_assigned_horizon_and_task(tmp_path):
    _,o,t,c,col,cache,bank,s=case(tmp_path)
    optimizer=PersistentOptimizers(o,c,s);optimizer.phase('fixed')
    sampler=QuerySampler(bank,2);v,g=sampler.sample(4,6);support=supports(sampler,'cpu')
    total,terms,stats=loss(c,o,bank,v,g,s,support)
    assert sum(sum(r) for r in stats['vector']['counts'])==8
    for task in ('ce','vad','intensity'):
        assert sum(sum(r) for r in stats[task]['counts'])==2
    assert stats['vector_mse']==pytest.approx(stats['vector']['value']/4)
    with pytest.raises(ValueError,match='does not cover'):
        loss(c,o,bank,[x for x in v if x[1]==0],g,s,support)
    total.backward();optimizer.step()


def test_optimizers_keep_moments_and_unused_parameters_frozen(tmp_path):
    _,o,_,c,_,_,_,s=case(tmp_path)
    optimizer=PersistentOptimizers(o,c,s);optimizer.phase('fixed')
    parameter=optimizer.named['flow'][0][1];unused=optimizer.named['flow'][-1][1];before=unused.clone()
    parameter.square().sum().backward();optimizer.step()
    assert unused.grad is None;torch.testing.assert_close(unused,before,atol=0,rtol=0)
    step=optimizer.optimizers['flow'].state[parameter]['step'].clone()
    optimizer.phase('calibration');optimizer.phase('readapt')
    assert torch.equal(optimizer.optimizers['flow'].state[parameter]['step'],step)
    saved=copy.deepcopy(optimizer.state_dict());other=PersistentOptimizers(o,c,s);other.load_state_dict(saved)
    assert other.updates==optimizer.updates


def test_affine_zero_initialization_role_symmetry_and_long_stability():
    torch.manual_seed(11)
    old=UnifiedEmotionStateCore(observation_dim=4,relation_dim=3,hidden_dim=8,flow_kind='adaptive_dyadic_v1')
    new=UnifiedEmotionStateCore(**{**old.get_config(),'flow_kind':'adaptive_affine_dyadic_v2','affine_max_offset':.5})
    missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
    assert set(missing)=={'adaptive_flow.affine_fast','adaptive_flow.affine_slow'} and not unexpected
    state=old.initialize(2);state.fast.normal_(0,.1);state.slow.normal_(0,.1)
    for a,b in zip(old.forecast(state,[1.,3.7]),new.forecast(state,[1.,3.7])):torch.testing.assert_close(a.z,b.z,atol=0,rtol=0)
    with torch.no_grad():new.adaptive_flow.affine_fast.fill_(.2);new.adaptive_flow.affine_slow.fill_(.2)
    a=new.forecast(state,[2.5])[0];b=new.forecast(state.role_swap(),[2.5])[0].role_swap()
    torch.testing.assert_close(a.z,b.z,atol=2e-6,rtol=2e-5)
    a.z.square().mean().backward();assert new.adaptive_flow.affine_slow.grad.abs().sum()>0
    with torch.no_grad():
        long=new.forecast(state,[30.,300.,1800.])
        assert all(torch.isfinite(x.z).all() and x.z.norm(dim=-1).max()<20 for x in long)
    restored=UnifiedEmotionStateCore.from_config(new.get_config());restored.load_state_dict(new.state_dict())
    torch.testing.assert_close(restored.forecast(state,[1.])[0].z,new.forecast(state,[1.])[0].z)


def test_metrics_batch_size_and_true_endpoint_weights(tmp_path):
    _,o,t,c,_,_,bank,s=case(tmp_path)
    baseline=fit(bank,c,s,'cpu')
    one=evaluate(o,c,bank,baseline,{**s,'prediction_batch':1},'cpu')
    many=evaluate(o,c,bank,baseline,{**s,'prediction_batch':8},'cpu')
    for name in one['methods']:
        assert one['methods'][name]['mse']==pytest.approx(many['methods'][name]['mse'],rel=1e-5,abs=1e-10)
        semantic=one['semantic'][name]
        assert semantic['query_weighted']['emotion_count']==4
        assert semantic['endpoint_balanced']['emotion_count']==2
    assert one['query_hash']==many['query_hash']


@pytest.mark.parametrize('phase',['fixed','joint'])
def test_gradient_diagnostic_preserves_sampler_rng_and_weights(tmp_path,phase):
    _,o,_,c,_,_,bank,s=case(tmp_path);opt=PersistentOptimizers(o,c,s);opt.phase(phase)
    before=torch.get_rng_state().clone();weights={k:v.clone() for k,v in c.state_dict().items()}
    online=phase=='joint'
    sampler=QuerySampler(bank,88);v,g=sampler.sample(4,6)
    _,terms,_=loss(c,o,bank,v,g,s,supports(sampler,'cpu'),online)
    parameters=opt.parameters();expected={}
    for name,term in terms.items():
        values=torch.autograd.grad(term,parameters,retain_graph=True,allow_unused=True)
        expected[name]=float(torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                                      for p,g in zip(parameters,values)]).norm())
    d=gradient_diagnostic(o,c,bank,s,4,6,0,1,88,online=online)
    for name,norm in expected.items():assert d['raw_task_gradients'][name]['norm']==pytest.approx(norm,rel=1e-6,abs=1e-10)
    assert d['graph_protocol']=='fresh_forward_per_task_v1'
    assert d['raw_task_gradients']['vector']['norm']>0 and torch.equal(before,torch.get_rng_state())
    assert all(torch.equal(v,weights[k]) for k,v in c.state_dict().items())
    assert all(p.grad is None for p in opt.parameters())


def test_four_stages_resume_exactly_without_external_weights(tmp_path):
    config,o,t,c,_,_,_,settings=case(tmp_path)
    initial=tmp_path/'source.pt';save_checkpoint(initial,dict(observer=o,teacher=t,state=c),config,
        dict(observer=o.construction(),state=c.get_config()),'dynamics_v3')
    config['train']['cpu_threads']=1
    config['paths'].update(dynamics_checkpoint=str(initial),feature_cache=str(tmp_path/'features'))
    config['staged']=dict(settings,calibration_steps=2,fixed_steps=2,bank_refresh_steps=2,joint_rounds=1,
        joint_steps_per_round=2,readapt_steps_per_round=2,bank_dialogues_per_domain=2,
        calibration_dialogues_per_domain=2,validation_dialogues_per_domain=2,validation_every=2,
        checkpoint_every=1,archive_every=2,gradient_every=3,log_every=1,update_refresh_steps=1)
    with contextlib.redirect_stdout(io.StringIO()):run(config)
    final=read_checkpoint(Path(config['paths']['output'])/'last.pt')
    midway=Path(config['paths']['output'])/'step000004.pt';require_training_revision(read_checkpoint(midway))
    initial.rename(tmp_path/'moved.pt');config['paths']['resume']=str(midway)
    with contextlib.redirect_stdout(io.StringIO()):run(config)
    after=read_checkpoint(Path(config['paths']['output'])/'last.pt')
    for name,weights in final['models'].items():
        for key,value in weights.items():torch.testing.assert_close(value,after['models'][name][key],atol=0,rtol=0)
    assert final['optimizer']['updates']==after['optimizer']['updates']=={'update':4,'flow':6}


def _ddp_worker(rank,size,url,path):
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=url,rank=rank,world_size=size)
    p=torch.nn.Parameter(torch.tensor(.7,dtype=torch.float64));unused=torch.nn.Parameter(torch.tensor(.4,dtype=torch.float64))
    optimizer=torch.optim.AdamW([p,unused],lr=.01)
    domains=torch.tensor([0,0,1,1,2,2,2]);h=torch.tensor([0,1,0,1,0,1,1]);x=torch.arange(1,8,dtype=torch.float64)
    chosen=torch.arange(rank,7,size);cell=domains[chosen]*2+h[chosen]
    sums=torch.zeros(6,dtype=torch.float64).index_add(0,cell,(p*x[chosen]-1).square()).view(3,2)
    counts=torch.zeros(6,dtype=torch.float64).index_add(0,cell,torch.ones(len(chosen),dtype=torch.float64)).view(3,2)
    term,_=global_mean(sums,counts,torch.ones(3,2,dtype=torch.bool),True)
    # Only rank zero has this supervised task.
    label_sums=(p-2).square().reshape(1,1) if rank==0 else torch.zeros(1,1,dtype=torch.float64)+p*0
    labels,_=global_mean(label_sums,torch.tensor([[float(rank==0)]],dtype=torch.float64))
    (term+.25*labels).backward();synchronize([p,unused]);gradient=p.grad.clone();optimizer.step()
    if rank==0:torch.save(dict(loss=term.detach()+.25*labels.detach(),grad=gradient,parameter=p.detach(),unused_none=unused.grad is None),path)
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo not installed')
def test_two_rank_global_counts_match_single_rank_update(tmp_path):
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    output=tmp_path/'ddp.pt';mp.spawn(_ddp_worker,args=(2,f'tcp://127.0.0.1:{port}?use_libuv=0',str(output)),nprocs=2,join=True)
    result=torch.load(output,weights_only=False)
    p=torch.nn.Parameter(torch.tensor(.7,dtype=torch.float64));optimizer=torch.optim.AdamW([p],lr=.01)
    x=torch.arange(1,8,dtype=torch.float64);error=(p*x-1).square()
    expected=(error[:5].sum()+error[5:].mean())/6+.25*(p-2).square()
    expected.backward();gradient=p.grad.clone();optimizer.step()
    torch.testing.assert_close(result['loss'],expected.detach());torch.testing.assert_close(result['grad'],gradient)
    torch.testing.assert_close(result['parameter'],p.detach());assert result['unused_none']
