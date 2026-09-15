import copy
import json
import pytest
import torch
from emotion_ssm.data.continuous_dialogues import ContinuousDialogueDataset,PROTOCOL
from emotion_ssm.train.history_sampling import HistoryWindowCursor
from emotion_ssm.train.generation_v3 import train_segment
from emotion_ssm.train.generation_sampling import ConversationStates
from test_v3_generation import small_model,packet
from test_generation_repairs import frozen


class Clips:
    prefetch_rng_neutral=True
    def __init__(self,split='train',lengths=(3,3)):
        self.split=split;self.names=[f'{split}-source{i}-clip' for i in range(len(lengths))]
        self.source_ids=[f'{split}-source{i}' for i in range(len(lengths))]
        self.lengths=lengths;self.manifest_digest='fixture-'+split;self.feature_digest='fixture'
        self.feature_sources={'dualtalk':'synthetic-fixture'}
    def __len__(self):return len(self.names)
    def packets(self,index):
        for t in range(1,self.lengths[index]+1):
            p=packet(t,self.names[index]);p['start_time']=0.
            for prefix in ('target','partner'):
                f=p[prefix+'_features']
                f['audio_starts']=f['audio_times']-.2
                f['audio_fresh_mask']=torch.ones(1,4,dtype=torch.bool)
                f['prosody_tokens']=torch.zeros(1,4,8);f['prosody_times']=f['audio_times'].clone()
                f['prosody_mask']=f['audio_mask'].clone();f['audio_history_complete']=torch.tensor([True])
                f['text_positions']=torch.tensor([[2*t-2,2*t-1]])
                f['text_roles']=torch.tensor([[0,1]])
            yield p,p['partner_blendshape']*.3,torch.ones(1,25,dtype=torch.bool)


def timeline(data,gap=0.):
    return dict(protocol=PROTOCOL,split=data.split,token_manifest_digest=data.manifest_digest,conversations=[
        dict(session_id='continuous',source_id='source-original',roles=['personA','personB'],segments=[
            dict(name=data.names[0],start=10.,end=13.,roles=['personA','personB'],verified=True,evidence='fixture-only'),
            dict(name=data.names[1],start=13.+gap,end=16.+gap,roles=['personA','personB'],verified=True,evidence='fixture-only')])])


def test_continuity_requires_source_role_and_clock_evidence():
    data=Clips();meta=timeline(data)
    for key,value in [('verified',False),('roles',['personB','personA']),('start',12.)]:
        bad=copy.deepcopy(meta);bad['conversations'][0]['segments'][1][key]=value
        with pytest.raises(ValueError):ContinuousDialogueDataset(data,bad)
    meta['split']='val'
    with pytest.raises(ValueError):ContinuousDialogueDataset(data,meta)


@pytest.mark.parametrize('gap',[0.,1.4])
def test_continuous_timeline_preserves_text_audio_and_physical_gaps(gap):
    torch.set_num_threads(1);base=Clips();data=ContinuousDialogueDataset(base,timeline(base,gap))
    rows=list(data.packets(0));model=small_model().eval();state=None
    assert sum(int(v.any()) for _,_,v in rows)==6
    with torch.no_grad():
        for p,t,v in rows:
            _,state,d=model(p,state,observe_only=True)
            assert p['roles']==('personA','personB')
            assert d['observation_inputs'][0]['audio_tokens'].shape[1]==d['observation_inputs'][0]['prosody_tokens'].shape[1]
    assert state.time==pytest.approx(16+gap)
    assert state.emotion.elapsed.item()==pytest.approx(6+gap)
    assert state.audio_history['personA']['audio_tokens'].shape[0]>=24
    assert rows[-1][0]['target_features']['text_mask'].sum()==12
    assert len(rows)==data.lengths[0]


def test_age_window_sampling_resume_budget_and_prefix_replay():
    torch.set_num_threads(1);data=Clips(lengths=(45,45,45,45));cfg,params=frozen(model:=small_model())
    sampler=HistoryWindowCursor(data,seed=44,conversations=2,blocks_per_conversation=4,
        config={'natural_probability':0.})
    rows=sampler.take_valid(8);saved=sampler.state_dict();expected=sampler.take_valid(8)
    restored=HistoryWindowCursor(data,seed=44,conversations=2,blocks_per_conversation=4,
        config={'natural_probability':0.},saved=saved)
    from test_generation_throughput import assert_identical
    assert_identical(expected,restored.take_valid(8))
    assert sum(int(v.any()) for _,_,v in rows)==8
    assert len({p['training_source_id'] for p,_,_ in rows})==2
    for p,_,_ in rows:
        if 'training_prefix' in p:
            assert all(x['time']<p['time'] for x in p['training_prefix'])
            assert all('target_flame' not in x and 'target_blendshape' not in x for x in p['training_prefix'])
    optimizer=torch.optim.AdamW(params,lr=0.)
    bank,m=train_segment(model,rows,optimizer,state=ConversationStates(),tbptt_steps=4)
    assert m['global_valid_blocks']==8 and not bank.states and not bank.boundary_for_loss
    assert sampler.coverage()[-2]['sources']==4 and sampler.coverage()[-1]['windows']==0


def test_long_age_sampler_keeps_domain_horizon_quota_and_resume(tmp_path):
    from test_staged_v34 import case
    from emotion_ssm.train.staged_v33.data import QuerySampler
    from emotion_ssm.utils.history_coverage import endpoint_coverage,bank_coverage
    _,_,_,_,col,_,bank,_=case(tmp_path)
    setting=dict(age_edges=[0,2,4,8],natural_probability=.5)
    sampler=QuerySampler(bank,99,age_sampling=setting)
    sampler.sample(32,32);saved=copy.deepcopy(sampler.state_dict());expected=sampler.sample(32,32)
    again=QuerySampler(bank,99,age_sampling=setting);again.load_state_dict(saved)
    assert expected==again.sample(32,32) and again.age_visits
    assert sum(again.domain_counts.values())==64
    assert bank_coverage(bank)
    audit=endpoint_coverage([(col.identity(i),col[i]['packets']) for i in range(len(col))],[1,2])
    assert audit['missing_long_data'] and any(c['real_endpoint_queries'] for c in audit['cells'])


def test_history_index_cache_preserves_sampling_and_detects_corruption(tmp_path):
    data=Clips(lengths=(10,12,14,16));settings={'candidate_cache_directory':str(tmp_path)}
    original=HistoryWindowCursor(data,seed=44,conversations=2,config=settings)
    natural=HistoryWindowCursor(data,seed=44,conversations=2)
    saved_packets=data.packets
    def forbidden(_):
        raise AssertionError('A warm candidate index must not load all dialogues')
    data.packets=forbidden
    cached=HistoryWindowCursor(data,seed=44,conversations=2,config=settings)
    data.packets=saved_packets
    assert cached.candidates==original.candidates==natural.candidates
    for _ in range(3):
        sampled=[]
        for cursor in (original,cached,natural):
            sampled.append([(p['training_record_name'],p['history_window_origin'],p['time'])
                            for p,_,_ in cursor.take_valid(8)])
        assert sampled[0]==sampled[1]==sampled[2]
    path=next(tmp_path.glob('*.json'));record=json.loads(path.read_text())
    record['candidates'][0][1]+=1;path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='identity/content mismatch'):
        HistoryWindowCursor(data,seed=44,conversations=2,config=settings)


def test_context_outages_remain_missing_in_later_cached_windows(tmp_path):
    from test_staged_v34 import case
    from emotion_ssm.train.staged_v33.missing import make_plan,mask_sensor_features
    _,o,t,c,col,cache,_,s=case(tmp_path)
    for identity in ('dialogue-a','dialogue-b'):
        plan=make_plan(identity,180,'train_context',17,64,16)
        assert plan==make_plan(identity,180,'train_context',17,64,16)
        assert all(x['kind']=='sensor_gap' for x in plan['intervals'])
    from emotion_ssm.train.staged_v34.data import build_bank
    bank=build_bank(o,c,cache,col,[0,1],[1,2],1,protocol='train_context')
    assert bank['protocol']=='train_context' and bank['valid'].any()
    clean=cache.get(col,0);masked=cache.get(col,0,'train_context')
    torch.testing.assert_close(clean['gold'],masked['gold'])
    # Explicit interval probes cached overlapping context, after the outage ends.
    f=col[0]['packets'][-1]['roles'][0]
    plan={'intervals':[dict(kind='sensor_gap',start=0.,end=4.,roles=[0,1],modes=['A','T','V'])]}
    later=copy.deepcopy(f);later['now']=torch.tensor(9.);later['dt']=torch.tensor(1.)
    a=mask_sensor_features(f,0,plan);b=mask_sensor_features(later,0,plan)
    torch.testing.assert_close(a['audio_mask'],b['audio_mask'])


def test_condition_eval_reuses_features_and_training_only_donors(tmp_path):
    from test_v3_checkpoint import tiny_config
    from emotion_ssm.utils.checkpoint_v3 import build_avatar
    from emotion_ssm.train.generation_condition_eval import fit_training_conditions,evaluate_conditions
    torch.set_num_threads(1);cfg,construction=tiny_config()
    model=build_avatar(cfg,construction=construction,initialize=False).eval()
    train=Clips('train',(7,7));val=Clips('val',(7,7))
    # Match tiny observer feature widths (audio/text=8 already supplied).
    bank=fit_training_conditions(model,train,train.names)
    result=evaluate_conditions(model,val,val.names,bank,history_ablations=True)
    assert result['matching_coverage']==1 and result['candidate_blocks']==6
    assert result['metrics']['true']['evaluated_chunks']==6
    assert result['sensitivity']['true']['output_mse']==0
    assert result['independent_semantics'] is None
    assert 'uncoupled_history' in result['metrics']
    assert result['sensitivity']['state_recent32']['condition_mse']==pytest.approx(0.,abs=1e-12)
    with pytest.raises(ValueError,match='train only'):fit_training_conditions(model,val,val.names)
    with pytest.raises(ValueError,match='anchor'):evaluate_conditions(model,val,val.names,bank,score_start_seconds=1)


def test_suite_budgets_routes_and_pending_gates(tmp_path):
    from scripts.run_generation_repair_suite import build_suite
    from emotion_ssm.config_v3 import default_config
    from emotion_ssm.train.generation_v3 import generation_execution_settings
    suite=build_suite(default_config(),'common.pt',tmp_path,seeds=[6666])
    assert len(suite['runs'])==4
    configs=[json.loads(open(row['config'],encoding='utf-8').read()) for row in suite['runs']]
    for cfg in configs:
        assert cfg['train']['max_steps']==1000 and cfg['train']['global_chunks_per_step']==32
        assert cfg['generation']['variant']=='dyadic'
        assert cfg['paths']['avatar_initialization']=='common.pt'
        assert generation_execution_settings(cfg)==dict(prefetch_batches=1,defer_metrics=True)
    pending=build_suite(default_config(),'common.pt',tmp_path/'pending',seeds=[6666],boundary_weights=[.1],routes=['actual_semantic'])
    assert pending['runs'][0]['status']=='awaiting_validated_visual_teacher'


def test_current_full_origins_ignore_stale_prefixes_and_preserve_tbptt(tmp_path):
    from test_staged_v34 import case
    from emotion_ssm.train.staged_v34.data import replay_current_origins,build_bank
    from emotion_ssm.train.staged_dynamics_support import memory_index
    _,o,t,c,col,cache,bank,s=case(tmp_path)
    with torch.no_grad():
        for p in c.event_projection.parameters():p.add_(.25)
        bank['prefixes'].fast.add_(99.)
    fresh=build_bank(o,c,cache,col,[0,1],[1,2],1)
    rows=list(range(len(bank['keys'])))
    detached=replay_current_origins(o,c,bank,rows,'cpu',with_grad=False)
    trained=replay_current_origins(o,c,bank,rows,'cpu',with_grad=True)
    torch.testing.assert_close(detached.z,fresh['states'].z,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(trained.z,detached.z,atol=2e-6,rtol=2e-5)
    assert detached.z.grad_fn is None
    trained.z.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in c.parameters())
