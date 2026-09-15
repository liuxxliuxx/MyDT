import copy
import torch

from emotion_ssm.config_v3 import default_config
from emotion_ssm.train.generation_v3 import (
    configure_generation_from_config, generation_auxiliary_enabled, train_segment,
)
from test_v3_generation import small_model, packet


def test_frozen_upstream_remains_identical_while_generator_learns():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model=small_model('dyadic')
    model.teacher=copy.deepcopy(model.observer).requires_grad_(False)
    cfg=default_config()
    cfg['generation'].update(train_observer=False,train_state=False)
    before={name:copy.deepcopy(module.state_dict()) for name,module in
        [('observer',model.observer),('state',model.state_model),('teacher',model.teacher),('generator',model.generator)]}
    params=configure_generation_from_config(model,cfg)
    assert not generation_auxiliary_enabled(model,cfg['train'])
    assert all(not p.requires_grad for m in (model.observer,model.state_model,model.teacher) for p in m.parameters())
    optimizer=torch.optim.AdamW(params,lr=1e-3)
    examples=[(packet(t,event=t==1),torch.zeros(1,25,56),torch.ones(1,25,dtype=torch.bool)) for t in (1,2,3)]
    state,metrics=train_segment(model,examples,optimizer,tbptt_steps=2)
    assert state.emotion.elapsed.item()==3
    assert metrics['global_valid_blocks']==3 and metrics['valid_frames']==75
    assert metrics['generator_grad_norm']>0
    assert metrics['observer_grad_norm']==metrics['state_grad_norm']==0
    for name,module in [('observer',model.observer),('state',model.state_model),('teacher',model.teacher)]:
        assert not module.training
        for key,value in module.state_dict().items():assert torch.equal(value,before[name][key])
    assert any(not torch.equal(v,before['generator'][k]) for k,v in model.generator.state_dict().items())


def test_default_generation_retains_joint_training_policy():
    model=small_model('dyadic')
    model.teacher=copy.deepcopy(model.observer)
    cfg=default_config()
    configure_generation_from_config(model,cfg)
    assert generation_auxiliary_enabled(model,cfg['train'])
    assert any(p.requires_grad for p in model.observer.parameters())
    assert any(p.requires_grad for p in model.state_model.parameters())
    cfg['train'].update(coordinate_weight=0.,future_weight=0.,masked_weight=0.)
    assert not generation_auxiliary_enabled(model,cfg['train'])
