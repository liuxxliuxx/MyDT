import copy
import contextlib
import io
import json
from pathlib import Path
from dataclasses import fields
import pytest
import torch
from test_staged_v33 import case as old_case
from emotion_ssm.models.state_core import EmotionMemory
from emotion_ssm.train.staged_v33.data import build_bank as sequential_bank
from emotion_ssm.train.staged_v33.baselines import fit
from emotion_ssm.train.staged_v34.data import build_bank,step_batch,origin_drift,QuerySampler
from emotion_ssm.train.staged_v34.semantics import class_audit,confusion_metrics
from emotion_ssm.train.staged_v34.objectives import loss,supports
from emotion_ssm.train.staged_v34.optimization import PersistentOptimizers
from emotion_ssm.train.staged_v34.evaluation import evaluate,deployment_gate
from emotion_ssm.train.staged_v34.trainer import run,DEFAULTS
from emotion_ssm.train.staged_v34.diagnostics import gradient_diagnostic
from emotion_ssm.utils.checkpoint_v3 import save_checkpoint,read_checkpoint,require_training_revision


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def case(tmp_path):
    config,o,t,c,col,cache,bank,s=old_case(tmp_path)
    s={**DEFAULTS,**s,'replay_dialogue_batch':8,'long_horizon_min':1}
    cache.settings=s
    bank=build_bank(o,c,cache,col,[0,1],[1,2],1)
    bank['class_audit']=class_audit(bank,s)
    return config,o,t,c,col,cache,bank,s


@pytest.mark.parametrize('protocol',['clean','update_gap','sensor_gap','train_mixed'])
def test_batched_full_replay_matches_sequential(tmp_path,protocol):
    _,o,t,c,col,cache,bank,s=case(tmp_path)
    before=sequential_bank(o,c,cache,col,[0,1],[1,2],1,protocol)
    after=build_bank(o,c,cache,col,[0,1],[1,2],1,protocol)
    for key in ('keys','endpoints','replay_rows','coverage','producer'):assert before[key]==after[key]
    for key in ('states','prefixes'):
        for field in fields(EmotionMemory):
            torch.testing.assert_close(getattr(before[key],field.name),getattr(after[key],field.name),rtol=2e-5,atol=2e-6)
    for key in ('targets','valid','history','current_gold','current_valid'):
        torch.testing.assert_close(before[key],after[key],rtol=2e-5,atol=2e-6)
    for k,v in before['current'].items():assert after['current'][k]==pytest.approx(v,rel=1e-5,abs=1e-8)


def test_unequal_length_padding_and_fractional_dt(tmp_path):
    _,o,t,c,col,cache,bank,s=case(tmp_path)
    rows=[copy.deepcopy(cache.get(col,i)) for i in [0,1]]
    # One dialogue ends early; its padding must not modify baseline, elapsed or event IDs.
    rows[1]['times']=rows[1]['times'][:2]
    rows[0]['dt']=[.2,.7,1.5,.1,1.2][:len(rows[0]['times'])]
    for i,dt in enumerate(rows[0]['dt']):rows[0]['observations']['action_duration'][i].clamp_(max=dt)
    batched=c.initialize(2)
    single=[c.initialize(1),c.initialize(1)]
    with torch.no_grad():
        for tick in range(len(rows[0]['times'])):
            batched,_,_=step_batch(o,c,rows,batched,tick,'cpu')
            for i,row in enumerate(rows):
                if tick<len(row['times']):single[i],_,_=step_batch(o,c,[row],single[i],tick,'cpu')
            for f in fields(EmotionMemory):
                torch.testing.assert_close(getattr(batched,f.name),torch.cat([getattr(x,f.name) for x in single]),atol=2e-6,rtol=2e-5)


def test_class_weights_unique_train_labels_and_bounded(tmp_path):
    *_,bank,s=case(tmp_path)
    audit=class_audit(bank,s);weights=torch.tensor(audit['weights']);counts=torch.tensor(audit['counts'])
    assert weights.max()<=3 and torch.equal(weights[counts==0],torch.zeros_like(weights[counts==0]))
    assert audit['endpoints']==2 and audit['horizon_queries']==4 and audit['fit_split']=='train'
    changed=copy.deepcopy(bank);changed['endpoints'].append(changed['endpoints'][0])
    with pytest.raises(ValueError,match='Repeated'):class_audit(changed,s)
    unweighted=class_audit(bank,{**s,'class_weight_power':0})
    assert all(w==1 for w,n in zip(torch.tensor(unweighted['weights']).flatten(),counts.flatten()) if n)


@pytest.mark.parametrize('phase',['calibration','fixed','joint','readapt'])
def test_future_gradients_reach_only_active_partition(tmp_path,phase):
    _,o,t,c,col,cache,bank,s=case(tmp_path)
    opt=PersistentOptimizers(o,c,s);opt.phase(phase)
    sampler=QuerySampler(bank,77);v,g=sampler.sample(4,6)
    total,parts,stats=loss(c,o,bank,v,g,s,supports(sampler,'cpu'),phase in ('calibration','joint'))
    total.backward()
    active='update' if phase in ('calibration','joint') else 'flow'
    assert opt.active==[active]
    assert sum(float(p.grad.abs().sum()) for _,p in opt.named[active] if p.grad is not None)>0
    assert all(p.grad is None and not p.requires_grad for _,p in opt.named['flow' if active=='update' else 'update'])
    assert all(p.grad is None for name,p in o.named_parameters() if not name.startswith(('event_head.','action_head.')))
    assert all(p.grad is None for p in t.parameters())
    assert stats['label_weight']>0
    saved={k:v.clone() for k,v in c.state_dict().items()}
    diagnostic=gradient_diagnostic(o,c,bank,s,4,6,0,1,77,active=='update')
    assert diagnostic['raw_task_gradients']['ce']['norm']>0
    assert all(torch.equal(v,saved[k]) for k,v in c.state_dict().items())


def test_flow_change_detected_even_with_frozen_correction(tmp_path):
    _,o,t,c,col,cache,bank,s=case(tmp_path)
    opt=PersistentOptimizers(o,c,s);opt.phase('fixed')
    report=origin_drift(o,c,cache,col,[0,1],bank,[1,2],1,'clean',s)
    assert not report['refresh'] and report['z_mse']<1e-12
    with torch.no_grad():
        for name,p in opt.named['flow']:p.add_(.1)
    report=origin_drift(o,c,cache,col,[0,1],bank,[1,2],1,'clean',s)
    assert report['refresh']


def test_neutral_errors_and_domain_gate(tmp_path):
    cm=torch.zeros(7,7);cm[0,4]=10;cm[4,4]=3
    stats=confusion_metrics(cm)
    assert stats['nonneutral_to_neutral']==1 and stats['neutral_recall']==1
    assert stats['per_class_recall'][1] is None
    _,o,t,c,col,cache,bank,s=case(tmp_path)
    result=evaluate(o,c,bank,fit(bank,c,s,'cpu'),s,'cpu')
    initial={k:copy.deepcopy(result) for k in ('fixed','live','update_gap','sensor_gap')}
    initial['correction_parameters_unchanged']=True
    current=copy.deepcopy(initial)
    # Eliminate real neutral recognition: a reduced prediction-neutral share cannot pass.
    key=next(k for k,v in current['live']['semantic_cells']['learned'].items() if v['macro_f1'] is not None)
    initial['live']['semantic_cells']['learned'][key]['neutral_recall']=.8
    current['live']['semantic_cells']['learned'][key]['neutral_recall']=0.
    gate=deployment_gate(current,initial,{**s,'minimum_long_f1_improvement':0})
    assert not gate['gate_passed'] and not gate['checks'][f'live/{key}/neutral_recall']


def test_full_pipeline_resume_with_all_phase_refresh(tmp_path):
    config,o,t,c,_,_,_,s=case(tmp_path)
    initial=tmp_path/'source.pt';save_checkpoint(initial,dict(observer=o,teacher=t,state=c),config,
        dict(observer=o.construction(),state=c.get_config()),'dynamics_v3')
    config['train']['cpu_threads']=1
    config['paths'].update(dynamics_checkpoint=str(initial),feature_cache=str(tmp_path/'features'))
    config['staged']=dict(s,calibration_steps=2,fixed_steps=2,bank_refresh_steps=2,joint_rounds=1,
        joint_steps_per_round=2,readapt_steps_per_round=2,bank_dialogues_per_domain=2,
        calibration_dialogues_per_domain=2,validation_dialogues_per_domain=2,validation_every=2,
        checkpoint_every=1,archive_every=2,gradient_every=3,log_every=1,origin_refresh_steps=1,origin_probe_steps=1)
    with contextlib.redirect_stdout(io.StringIO()):run(config)
    final=read_checkpoint(Path(config['paths']['output'])/'last.pt')
    midway=Path(config['paths']['output'])/'step000004.pt';require_training_revision(read_checkpoint(midway))
    initial.rename(tmp_path/'moved.pt');config['paths']['resume']=str(midway)
    with contextlib.redirect_stdout(io.StringIO()):run(config)
    after=read_checkpoint(Path(config['paths']['output'])/'last.pt')
    for name,weights in final['models'].items():
        for key,value in weights.items():torch.testing.assert_close(value,after['models'][name][key],atol=0,rtol=0)
    assert final['optimizer']['updates']==after['optimizer']['updates']=={'update':4,'flow':4}
    records=[json.loads(x) for x in (Path(config['paths']['output'])/'rank0.jsonl').read_text().splitlines()]
    assert {r['block'] for r in records if r['event']=='prefix_refreshed'}=={0,1,2,3}
    assert after['run_state']['class_audit']['fit_split']=='train'
