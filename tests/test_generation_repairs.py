import copy
import torch
import pytest
from emotion_ssm.utils.generation_losses import (boundary_frame,boundary_loss_terms,matching_boundary,
    plan_objective_counts,loss_settings,visual_loss_terms)
from emotion_ssm.models.condition_router import ConditionRouter
from emotion_ssm.models.visual_affect_teacher import FrozenFlameAffect,observer_digest,FLAME_SCHEMA
from emotion_ssm.train.generation_v3 import train_segment,configure_generation_from_config,evaluate
from emotion_ssm.train.generation_sampling import ConversationStates
from emotion_ssm.config_v3 import default_config
from emotion_ssm.utils.checkpoint_v3 import require_generation_objective_match
from test_v3_generation import small_model,packet


def frozen(model):
    model.teacher=copy.deepcopy(model.observer)
    cfg=default_config();cfg['generation'].update(train_observer=False,train_state=False)
    cfg['generation_losses']={'boundary_weight':.1}
    params=configure_generation_from_config(model,cfg)
    return cfg,params


def test_boundary_gradients_masks_and_metadata():
    a=torch.ones(2,25,56,requires_grad=True);b=torch.zeros(2,25,56,requires_grad=True)
    valid=torch.ones(2,25,dtype=torch.bool);valid[1]=False
    truth=torch.zeros_like(a);truth[1]=float('nan')
    with torch.no_grad():a[1]=float('nan');b[1]=float('nan')
    edge=boundary_frame(packet(1),a,truth,valid)
    assert matching_boundary(packet(2),25,edge) is edge
    assert matching_boundary(packet(3),25,edge) is None
    swapped=packet(2);swapped['roles']=('B','A')
    assert matching_boundary(swapped,25,edge) is None
    sse,count=boundary_loss_terms(b,truth,valid,edge)
    assert count==56 and sse==56
    sse.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    assert a.grad[0,-1].abs().sum()>0 and b.grad[0,0].abs().sum()>0
    assert not a.grad[1].any() and not b.grad[1].any()


def test_boundary_training_cross_steps_rollback_and_counts():
    torch.set_num_threads(1);model=small_model();cfg,params=frozen(model)
    optimizer=torch.optim.AdamW(params,lr=0.)
    bank=ConversationStates();target=torch.zeros(1,25,56);valid=torch.ones(1,25,dtype=torch.bool)
    rows=[(packet(t),target,valid) for t in (1,2,3)]
    assert plan_objective_counts(rows,settings=cfg['generation_losses'])==(112,3)
    bank,m=train_segment(model,rows,optimizer,state=bank,tbptt_steps=2)
    assert m['boundary_loss']==pytest.approx(m['boundary_velocity_mse'],rel=1e-6)
    assert m['train_total_loss']==pytest.approx(m['generation_total_with_boundary'],rel=1e-6)
    edge=next(iter(bank.boundary_for_loss.values()));assert edge.prediction.grad_fn is None
    snapshot=edge.prediction.clone()
    bank2,m2=train_segment(model,[(packet(4),target,valid)],optimizer,state=bank,tbptt_steps=2)
    assert m2['error_elements']['boundary_velocity']==56
    assert next(iter(bank.boundary_for_loss.values())).end_frame==75
    torch.testing.assert_close(edge.prediction,snapshot)
    assert next(iter(bank2.boundary_for_loss.values())).end_frame==100


def test_teacher_is_flame_only_frozen_and_input_differentiable():
    torch.set_num_threads(1)
    teacher=FrozenFlameAffect(small_model().observer)
    teacher.train();assert not teacher.training and not teacher.observer.training
    motion=torch.randn(2,25,56,requires_grad=True);valid=torch.ones(2,25,dtype=torch.bool)
    out=teacher(motion,valid)
    out['affect'][:,0].sum().backward()
    assert torch.isfinite(motion.grad).all() and motion.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    with pytest.raises(ValueError,match='independent'):teacher.require_validated()
    with pytest.raises(ValueError,match='domain'):teacher(motion,valid,domain_id=1)
    with pytest.raises(ValueError,match='schema'):teacher(motion,valid,motion_schema='other')


def test_visual_losses_reach_generated_not_reference_and_exclude_nan():
    torch.manual_seed(8)
    pred={k:torch.randn(shape,requires_grad=True) for k,shape in
          [('affect',(2,8)),('emotion_logits',(2,7)),('vad',(2,3)),('intensity',(2,))]}
    ref={k:torch.randn_like(v,requires_grad=True) for k,v in pred.items()}
    with torch.no_grad():
        for values in (pred,ref):
            for value in values.values():value[1]=float('nan')
    terms=visual_loss_terms(pred,ref,torch.tensor([True,False]))
    sum(terms.values()).backward()
    assert all(v.grad is None for v in ref.values())
    assert all(torch.isfinite(v.grad).all() and v.grad[0].abs().sum()>0 for v in pred.values())


def test_routing_preserves_state_and_oracle_cannot_deploy():
    torch.set_num_threads(1)
    model=small_model().eval();other=copy.deepcopy(model)
    other.condition_router=ConditionRouter(model.state_model.context_dim,model.state_model.context_layout(),{'mode':'film_off'})
    a,sa,da=model(packet());b,sb,db=other(packet())
    torch.testing.assert_close(sa.emotion.z,sb.emotion.z)
    assert not torch.equal(a,b) and not db['film_enabled']
    route=ConditionRouter(model.state_model.context_dim,model.state_model.context_layout(),
                          dict(mode='oracle_visual_pseudo',semantic_heads=['emotion']))
    with pytest.raises(ValueError,match='diagnostic'):route(da['context'])
    override=torch.randn(1,7,requires_grad=True)
    value,_=route(da['context'],override=override,diagnostic=True);value.sum().backward()
    assert override.grad is None and route.projector.weight.grad is not None
    with pytest.raises(ValueError,match='selection'):
        ConditionRouter(4,{},dict(mode='actual_semantic',semantic_heads=['bogus']))


def test_changed_objective_requires_new_optimizer_protocol():
    cfg=default_config();other=copy.deepcopy(cfg)
    other['generation_losses']={'boundary_weight':.1}
    with pytest.raises(ValueError):require_generation_objective_match(cfg,other)
    cfg['generation_losses']={'boundary_weight':0}
    require_generation_objective_match(default_config(),cfg)


def test_observe_only_prefix_exact_state_and_next_output():
    torch.set_num_threads(1);model=small_model().eval();full=warm=None
    with torch.no_grad():
        for t in range(1,11):
            _,full,_=model(packet(t),full)
            out,warm,_=model(packet(t),warm,observe_only=True)
            assert out is None
        a,sa,_=model(packet(11),full);b,sb,_=model(packet(11),warm)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    torch.testing.assert_close(sa.emotion.z,sb.emotion.z,atol=0,rtol=0)


def test_added_modules_checkpoint_roundtrip(tmp_path):
    from test_v3_checkpoint import tiny_config
    from emotion_ssm.utils.checkpoint_v3 import build_avatar,save_checkpoint,load_avatar
    torch.set_num_threads(1);cfg,construction=tiny_config()
    cfg['generation']['condition_routing']={'mode':'mean','train_mean':[.2]*26,'mean_provenance':{'split':'train','sha256':'test-fixture'}}
    # Context width comes from the actual state construction.
    from emotion_ssm.models.state_core import UnifiedEmotionStateCore
    d=UnifiedEmotionStateCore(**cfg['state']).context_dim
    cfg['generation']['condition_routing']['train_mean']=[.2]*d
    cfg['generation']['visual_teacher']={'enabled':True}
    model=build_avatar(cfg,construction=construction,initialize=False).eval()
    file=tmp_path/'complete.pt';save_checkpoint(file,{'system':model},cfg,model.construction_info,'streaming_avatar_v3')
    recovered,_,_=load_avatar(file)
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,recovered.state_dict()[name],atol=0,rtol=0)
    assert hasattr(recovered,'visual_teacher')
    assert recovered.condition_router.mean_provenance['split']=='train'


def distributed_rows(rank):
    rows=[]
    for t in range(1,(4 if rank==0 else 2)+1):
        mask=torch.ones(1,25,dtype=torch.bool)
        if rank==1:mask[:, -2:]=False
        rows.append((packet(t,session=str(rank)),torch.zeros(1,25,56),mask))
    return rows


def _boundary_worker(rank,url,path):
    import torch.distributed as dist
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=url,rank=rank,world_size=2)
    torch.manual_seed(771);model=small_model();_,params=frozen(model)
    optimizer=torch.optim.AdamW(params,lr=.001,foreach=False)
    _,metrics=train_segment(model,distributed_rows(rank),optimizer,state=ConversationStates(),tbptt_steps=2)
    if rank==0:torch.save(dict(model=model.state_dict(),metrics=metrics),path)
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason='Gloo unavailable')
def test_real_two_rank_boundary_loss_and_update_match_single(tmp_path):
    import socket
    import torch.multiprocessing as mp
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    output=tmp_path/'rank.pt'
    mp.spawn(_boundary_worker,args=(f'tcp://127.0.0.1:{port}?use_libuv=0',str(output)),nprocs=2,join=True)
    result=torch.load(output,weights_only=False)
    torch.set_num_threads(1);torch.manual_seed(771);model=small_model();_,params=frozen(model)
    optimizer=torch.optim.AdamW(params,lr=.001,foreach=False)
    _,metrics=train_segment(model,distributed_rows(0)+distributed_rows(1),optimizer,state=ConversationStates(),tbptt_steps=2)
    for key in ('generation_total','boundary_loss','train_total_loss'):
        assert metrics[key]==pytest.approx(result['metrics'][key],rel=1e-6)
    assert metrics['error_elements']==result['metrics']['error_elements']
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,result['model'][key],atol=2e-7,rtol=2e-5)


def test_visual_generation_branch_learns_with_frozen_upstream(monkeypatch):
    torch.set_num_threads(1);model=small_model();model.visual_teacher=FrozenFlameAffect(model.observer)
    # Unit test the optimization path, separately from the real-human gate tests.
    monkeypatch.setattr(model.visual_teacher,'require_validated',lambda *args:None)
    monkeypatch.setattr(model.visual_teacher,'require_coordinate',lambda *args:None)
    cfg,params=frozen(model)
    cfg['generation_losses'].update(visual_feature_weight=.2,visual_class_distill_weight=.1,
        visual_vad_distill_weight=.1,vad_dimensions=[0],visual_intensity_distill_weight=.1)
    params=configure_generation_from_config(model,cfg)
    before={k:v.clone() for k,v in model.visual_teacher.state_dict().items()}
    optimizer=torch.optim.AdamW(params,lr=.001)
    state,m=train_segment(model,[(packet(1),torch.randn(1,25,56),torch.ones(1,25,dtype=torch.bool))],optimizer,tbptt_steps=2)
    assert m['visual_valid_windows']==1 and m['visual_feature_loss']>0
    assert m['film_grad_norm_before_clip']>0
    assert all(torch.equal(v,before[k]) for k,v in model.visual_teacher.state_dict().items())
    assert all(p.grad is None for p in model.visual_teacher.parameters())


def test_calibration_smoke_freezes_semantic_coordinates(tmp_path):
    from emotion_ssm.train.visual_teacher_calibration import calibrate,load_rows
    from emotion_ssm.models.visual_affect_teacher import affect_coordinate_digest
    torch.set_num_threads(1);observer=small_model().observer;coordinate=affect_coordinate_digest(observer)
    empty,status=calibrate(observer,[],[])
    assert empty is None and status['status']=='awaiting_independent_human_labels'
    def row(source,emotion):
        return dict(source_id=source,annotation_origin='human_manual',annotator_ids=['unit-test-fixture'],
                    motion=torch.randn(1,25,56),mask=torch.ones(1,25,dtype=torch.bool),domain_id=2,
                    emotion=emotion,vad=torch.tensor([.1,.2,.3]),vad_mask=torch.ones(3,dtype=torch.bool),
                    intensity=.2,intensity_mask=True)
    tr=[row('train-a',1),row('train-b',2)];va=[row('val-a',1),row('val-b',2)]
    teacher,report=calibrate(observer,tr,va,steps=2)
    assert affect_coordinate_digest(teacher.observer)==coordinate
    assert all(not p.requires_grad for p in teacher.parameters())
    assert report['steps']==2 and report['input_gradient_finite_nonzero']
    assert report['different_emotion_pairs']==1


def test_selection_keeps_internal_proxy_separate_from_independent_gold():
    from emotion_ssm.utils.generation_selection import conditioned_acceptance,selection_score
    reference=dict(boundary_velocity_mse=1.,jaw_mse=1.,neck_mse=1.,expression_mse=1.)
    policy=dict(maximum_relative_regression={k:.05 for k in reference},semantic_metric='f1',minimum_improvement=.01)
    result=conditioned_acceptance(reference,reference,policy)
    assert not result['accepted'] and result['status']=='awaiting_independent_semantics'
    gold=dict(identity='fixture',split='val',independent_human_rows=2,independent_of_training_teacher=True,improvements={'f1':.02})
    assert conditioned_acceptance(reference,reference,policy,gold,'fixture')['accepted']
    assert not conditioned_acceptance({**reference,'boundary_velocity_mse':1.06},reference,policy,gold,'fixture')['accepted']
    with pytest.raises(ValueError):conditioned_acceptance(reference,reference,policy,gold,'other-checkpoint')
    assert selection_score({'generation_total':2.,'generation_total_with_boundary':3.})==2
