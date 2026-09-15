"""Exercise the complete new experiment entry, with real tiny generator weights."""
import copy
import contextlib
import io
import json
import pytest
import torch
from test_v3_checkpoint import tiny_config
from test_generation_long_history import Clips
from emotion_ssm.utils.checkpoint_v3 import build_avatar,save_checkpoint,read_checkpoint,load_avatar
from emotion_ssm.train.generation_v3 import run
from scripts.run_generation_repair_suite import build_suite


def test_four_controls_complete_and_restore_new_objective(tmp_path,monkeypatch):
    torch.set_num_threads(1);cfg,construction=tiny_config()
    construction['generator_audio']['mask_time_prob']=.05
    initial=build_avatar(cfg,construction=construction,initialize=False)
    source=tmp_path/'common.pt';save_checkpoint(source,{'system':initial},cfg,initial.construction_info,'streaming_avatar_v3')
    suite=build_suite(cfg,source,tmp_path/'suite',steps=2,seeds=[6666])
    monkeypatch.setattr('emotion_ssm.train.generation_v3.GenerationTokenDataset',lambda tokens,raw,split:Clips(split,(5,5,5,5)))
    results=[]
    for spec in suite['runs']:
        config=json.loads(open(spec['config'],encoding='utf-8').read())
        config['train'].update(device='cpu',global_chunks_per_step=4,amp=False,validate_every=1,log_every=1)
        config['generation_stability'].update(conversations_per_rank=2,blocks_per_conversation=2)
        config['generation']['gradient_checkpointing']=False
        with contextlib.redirect_stdout(io.StringIO()):result=run(config)
        output=tmp_path/'suite'/spec['name']
        assert result['steps']==2 and (output/'best_reconstruction.pt').exists()
        payload=read_checkpoint(output/'last.pt')
        assert payload['global_step']==2 and payload['run_state']['optimizer_budget']=={'global_new_blocks':4,'step':2}
        assert (output/'reconstruction_frontier.json').exists()
        assert not (output/'best_conditioned.pt').exists()
        recovered,_,_=load_avatar(output/'best_reconstruction.pt')
        assert recovered.condition_router.mode==config['generation']['condition_routing']['mode']
        with pytest.raises(ValueError,match='already contains'):run(config)
        # Resume a complete immutable candidate; no external initial file read.
        config['paths']['resume']=str(output/'last.pt')
        config['paths']['avatar_initialization']='removed.pt'
        with contextlib.redirect_stdout(io.StringIO()):run(config)
        results.append(payload)
    assert {r['config']['generation_losses']['boundary_weight'] for r in results}=={0.,.1}
    assert {r['config']['generation']['condition_routing']['mode'] for r in results}=={'film_off','full_state'}


def test_new_upstream_is_explicitly_packaged_before_generator_retraining(tmp_path):
    from scripts.prepare_avatar_from_dynamics import prepare
    torch.set_num_threads(1);cfg,construction=tiny_config()
    model=build_avatar(cfg,construction=construction,initialize=False)
    avatar=tmp_path/'avatar.pt';save_checkpoint(avatar,{'system':model},cfg,model.construction_info,'streaming_avatar_v3',step=12)
    with torch.no_grad():next(model.state_model.event_projection.parameters()).add_(.2)
    dynamics=tmp_path/'dynamics.pt'
    save_checkpoint(dynamics,dict(observer=model.observer,teacher=model.teacher,state=model.state_model),cfg,
                    dict(observer=model.observer.construction(),state=model.state_model.get_config()),'dynamics_v3',step=9)
    output=tmp_path/'combined.pt';result=prepare(avatar,dynamics,output)
    avatar.rename(tmp_path/'moved_avatar.pt');dynamics.rename(tmp_path/'moved_dynamics.pt')
    recovered,_,payload=load_avatar(output)
    assert result['film_reset_to_identity'] and not result['optimizer_restored']
    assert payload['optimizer'] is None and payload['global_step']==0
    assert not recovered.generator.film.affine.weight.any()
    for key,value in model.state_model.state_dict().items():
        torch.testing.assert_close(value,recovered.state_model.state_dict()[key],atol=0,rtol=0)
    for key,value in model.generator.baseline.state_dict().items():
        torch.testing.assert_close(value,recovered.generator.baseline.state_dict()[key],atol=0,rtol=0)
