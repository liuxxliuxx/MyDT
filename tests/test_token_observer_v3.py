from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from emotion_ssm.data.packets_v3 import (TOKEN_PROTOCOL, empty_role_features, collate_role_features,
                                         validate_packet, endpoint_label, PacketFrameDataset)
from emotion_ssm.models.token_observer import TokenObserver, TokenObserverConfig, SUBSETS
from emotion_ssm.preprocess.tokens_v3 import LocalTokenFeatures, build_role_features, _upstream_dialogue
from emotion_ssm.train.observation_v3 import collate_observation_samples, observation_objective, DialogueBalancedBatches


@pytest.fixture(autouse=True)
def single_thread():
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model():
    torch.manual_seed(7)
    return TokenObserver(TokenObserverConfig(audio_dim=8,text_dim=8,model_dim=16,affect_dim=8,
                                             num_layers=1,num_heads=2,dropout=0.))


def features(seed=0):
    generator=torch.Generator().manual_seed(seed)
    result=empty_role_features(8,8,now=1.)
    for mode,dim in (("audio",8),("au",35),("flame",56),("text",8)):
        result[mode+"_tokens"]=torch.randn(5,dim,generator=generator)
        result[mode+"_mask"]=torch.ones(5,dtype=torch.bool)
        result[mode+"_times"]=torch.linspace(.2,1.,5)
    result.update(audio_starts=torch.arange(5,dtype=torch.float64)*.2,
                  audio_fresh_mask=torch.ones(5,dtype=torch.bool),
                  prosody_tokens=torch.randn(5,8,generator=generator),prosody_mask=result["audio_mask"].clone(),
                  prosody_times=result["audio_times"].clone(),text_positions=torch.arange(5))
    result.update(text_roles=torch.tensor([0,0,1,1,0]),text_fresh_mask=torch.ones(5,dtype=torch.bool),
                  fresh_observation=torch.ones(3,dtype=torch.bool),modality_mask=torch.ones(3,dtype=torch.bool),
                  context_available=torch.tensor(True),event_present=torch.tensor(True),action_present=torch.tensor(True),
                  action_duration=torch.tensor(1.))
    return result


def test_mask_replacement_precedes_context_attention():
    observer=model().eval()
    batch=collate_role_features([features(0),features(1)])
    corruption={mode:torch.zeros_like(batch[mode+"_mask"]) for mode in ("audio","au","flame","text")}
    corruption["audio"][:,1:4]=True
    corruption["text"][:,2:4]=True
    changed=copy.deepcopy(batch)
    changed["audio_tokens"][:,1:4]+=100
    changed["text_tokens"][:,2:4]*=-12
    first=observer.encode(batch,corruption=corruption)
    second=observer.encode(changed,corruption=corruption)
    torch.testing.assert_close(first["observation"].aff,second["observation"].aff,atol=0,rtol=0)
    torch.testing.assert_close(first["reconstruction"]["audio"],second["reconstruction"]["audio"],atol=0,rtol=0)


def test_relative_role_query_changes_text_affect_without_absolute_slot_bias():
    observer=model().eval()
    first=features()
    second=copy.deepcopy(first)
    second["text_roles"]=1-first["text_roles"]
    out=observer(collate_role_features([first,second]),"T")
    assert (out.aff[0]-out.aff[1]).abs().max()>1e-5
    # Exchanging storage slots alone changes no target-relative inputs.
    reverse=observer(collate_role_features([second,first]),"T")
    torch.testing.assert_close(out.aff,reverse.aff.flip(0))


def test_old_context_is_not_new_action_or_event():
    observer=model().eval()
    sample=empty_role_features(8,8)
    sample["text_tokens"]=torch.randn(3,8)
    sample["text_mask"]=torch.ones(3,dtype=torch.bool)
    sample["text_roles"]=torch.tensor([0,1,0])
    sample["text_times"]=torch.zeros(3)
    sample["text_fresh_mask"]=torch.zeros(3,dtype=torch.bool)
    out=observer(collate_role_features([sample]))
    assert out.context_available.item()
    assert not out.fresh_observation.any()
    assert out.aff.norm()>0
    assert not out.event.any() and not out.action.any()
    assert out.action_duration.item()==0


def test_missing_all_modes_zeroes_observation_and_losses_are_finite():
    observer=model()
    batch=collate_role_features([empty_role_features(8,8)])
    output=observer(batch)
    assert output.aff.count_nonzero()==0
    assert output.event.count_nonzero()==0
    assert output.action.count_nonzero()==0
    loss=observer.masked_loss(batch)["total"]
    assert torch.isfinite(loss)
    loss.backward()


def test_all_seven_subsets_share_affect_and_supervision_gradients():
    observer=model()
    batch=collate_role_features([features(1),features(3)])
    label={"start":0.,"end":1.,"emotion":3,"intensity":.5,"intensity_mask":True,
           "vad":[.4,0,0],"vad_mask":[True,False,False]}
    samples=[{"features":features(i),"targets":[label]} for i in (1,3)]
    data=collate_observation_samples(samples)
    for subset in SUBSETS:
        assert observer(batch,subset).aff.shape==(2,8)
        loss=observation_objective(observer,data,subset)["total"]
        assert torch.isfinite(loss)
        loss.backward()
    assert observer.fusion.layers[0].self_attn.in_proj_weight.grad.abs().sum()>0
    assert observer.adapters["flame"][0][0].weight.grad.abs().sum()>0


def test_target_flame_prohibition_and_future_timestamp():
    packet={"protocol":TOKEN_PROTOCOL,"start":0.,"end":1.,"dt":1.,"roles":[features(),features()],
            "targets":[[],[]],"avatar_role":0}
    with pytest.raises(ValueError,match="Avatar"):
        validate_packet(packet)
    packet["roles"][0]["flame_mask"].zero_()
    validate_packet(packet)
    packet["roles"][1]["audio_times"][0]=2.
    with pytest.raises(ValueError,match="Future"):
        validate_packet(packet)


class _LocalConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv1d(1,6,400,320)
    def forward(self,x):
        return self.conv(x[:,None])


class _FakeText(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(64,8)
    def get_input_embeddings(self):
        return self.emb


class _LocalAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(conv_kernel=[400],conv_stride=[320],hidden_size=8,feat_extract_norm="group")
        self.feature_extractor=_LocalConv()
        self.feature_projection=nn.Linear(6,8)
        self.context=nn.Linear(8,8)
    def _get_feat_extract_output_lengths(self,length):
        return (length-400)//320+1
    def forward(self,wave,attention_mask=None):
        hidden=self.feature_projection(self.feature_extractor(wave).transpose(1,2))
        return SimpleNamespace(last_hidden_state=hidden+self.context(hidden.mean(1,keepdim=True)))


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.source={"audio_dim":8,"text_dim":8,"audio_model":"test-local-conv","text_model":"test-lexical"}
        self.audio=_LocalAudio()
        self.text=_FakeText()
        self.tokenizer=lambda value,**kwargs:{"input_ids":[ord(c)%64 for c in value]}
    def construction(self):
        return {"test":"fake"}


def extractor():
    return LocalTokenFeatures({},backbone=_FakeBackbone())


def test_local_audio_has_no_neighbor_or_future_context_leakage_and_handles_tail():
    local=extractor()
    audio=torch.randn(3*8000)
    changed=audio.clone()
    changed[8000:16000]*=-8
    first=local.audio_tokens(audio,0.)
    second=local.audio_tokens(changed,0.)
    torch.testing.assert_close(first[0][[0,2]],second[0][[0,2]],atol=0,rtol=0)
    short=local.audio_tokens(torch.ones(120),0.)
    assert not short[1].any()
    assert short[2].max()<=.007501


def test_fractional_real_server_endpoint_uses_floor_samples_and_float64_clock():
    local=extractor()
    start,end=176.,176.26597046774344
    # The old round() path included sample 4255 and stamped 176.2660064697.
    wave=torch.randn(round((end-start)*16000))
    changed=wave.clone(); changed[-1]=1000.
    first=build_role_features(local,wave,[],start,end,"A",0)
    second=build_role_features(local,changed,[],start,end,"A",0)
    assert first["audio_times"].dtype==torch.float64
    assert float(first["audio_times"].max())==pytest.approx(176.2659375,abs=1e-12)
    assert float(first["audio_times"].max())<end
    torch.testing.assert_close(first["audio_tokens"],second["audio_tokens"],atol=0,rtol=0)
    packet={"protocol":TOKEN_PROTOCOL,"start":start,"end":end,"dt":end-start,"roles":[first,second],"targets":[[],[]]}
    validate_packet(packet)
    packet["roles"][1]["audio_times"][-1]=end+2e-6
    with pytest.raises(ValueError,match="Future"):
        validate_packet(packet)
    # Text available two microseconds later must not be rounded into this packet.
    words=[{"id":"future","text":"sad","role":"A","end":176.25,"available_at":end+2e-6}]
    assert not local.text_tokens(words,end,start,"A")["text_mask"].any()
    assert local.text_tokens(words,end+3e-6,start,"A")["text_mask"].any()


def test_late_text_visibility_and_shared_storage():
    local=extractor()
    words=[{"id":"1","text":"sad","role":"A","end":.4,"available_at":1.2},
           {"id":"2","text":"kind","role":"B","end":1.1,"available_at":1.6}]
    assert not local.text_tokens(words,1.,0.,"A")["text_mask"].any()
    first=local.text_tokens(words,1.4,1.,"A")
    second=local.text_tokens(words,2.,1.4,"B")
    assert first["text_tokens"].untyped_storage().data_ptr()==second["text_tokens"].untyped_storage().data_ptr()
    assert first["text_roles"].eq(0).all()
    assert second["text_roles"][:3].eq(1).all()
    assert not local.text_tokens(words,3.,2.,"B")["text_fresh_mask"].any()


def test_build_features_visual_silence_and_text_only_events():
    local=extractor()
    words=[{"id":"1","text":"hi","role":"A","end":.8,"available_at":1.}]
    values=build_role_features(local,torch.zeros(16000),words,0.,1.,"A",2,torch.randn(25,56),"flame")
    assert values["action_present"] and values["action_duration"]==1
    assert not values["audio_mask"].any()
    assert values["event_present"]
    late=build_role_features(local,torch.zeros(16000),words,0.,1.,"A",2)
    assert late["event_present"] and not late["action_present"]
    online=local(torch.zeros(1,16000),words,1.,0.,"A",False)
    assert online["domain_id"].item()==2
    assert online["audio_tokens"].ndim==3


def test_endpoint_masks_do_not_invent_intensity_or_vad():
    et=endpoint_label({"start_time":0.,"end_time":2.,"sentiment_score":1.,"intensity_abs":1.},"emotiontalk")
    assert et["vad_mask"]==[True,False,False] and et["intensity_mask"]
    ie=endpoint_label({"start_time":0.,"end_time":2.,"vad":[.1,.2,.3]},"iemocap")
    assert ie["vad_mask"]==[True,True,True] and not ie["intensity_mask"]


def test_upstream_second_packets_apply_label_once_and_mask_unknown_identity(tmp_path,monkeypatch):
    folder=tmp_path/"dialogues"/"d1"
    folder.mkdir(parents=True)
    utterances=[{"utterance_id":"u1","speaker_id":"a","start_time":0.,"end_time":1.6,"text":"sad","emotion":"sad"},
                {"utterance_id":"u2","speaker_id":"b","start_time":.2,"end_time":2.4,"text":"kind","emotion":"happy"}]
    (folder/"labels.json").write_text(json.dumps({"utterances":utterances}))
    torch.save({"utterance_ids":["u1","u2"],"au_sequences":[torch.ones(20,35),torch.ones(20,35)],"normalized":True},folder/"face_au_features.pt")
    (folder/"provenance.json").write_text(json.dumps({"face_identity_verified":False}))
    wave=tmp_path/"audio.wav"; wave.write_bytes(b"dummy")
    monkeypatch.setattr("emotion_ssm.preprocess.tokens_v3._load_wave",lambda path:torch.randn(48000))
    source={"input_root":str(tmp_path),"dataset":"iemocap","_audio_mapping":{"u1":str(wave),"u2":str(wave)}}
    packets,audit=_upstream_dialogue(source,"d1",extractor())
    assert [p["dt"] for p in packets]==pytest.approx([1.,1.,.4])
    assert [[len(labels) for labels in p["targets"]] for p in packets]==[[0,0],[1,0],[0,1]]
    assert not any(r["au_mask"].any() for p in packets for r in p["roles"])
    assert not packets[0]["roles"][0]["text_mask"].any()
    assert audit["unverified_visual_masked"]


def fake_source(language="shared"):
    return {"audio_model":"fake-audio-"+language,"text_model":"fake-text-"+language,"audio_dim":8,"text_dim":8,
            "audio_revision":"test-audio-revision","text_revision":"test-text-revision","preprocessing":TOKEN_PROTOCOL,
            "normalization":"per-500ms-zscore-with-raw-prosody-v2","token_audio":"independent-500ms-full-backbone-v1",
            "token_text":"lexical-input-embedding-explicit-order-v2","token_ms":500,"token_clock":"float64-floor-sample-end-v2",
            "audio_prosody":"log-rms-log-std-mean-zcr-log-peak-log-crest-voice-energy-cv-v1",
            "prosody_dim":8,"audio_history_seconds":16.}


def write_cache(root):
    root.mkdir()
    packets=[]
    for index in range(2):
        fs=[features(index),features(index+5)]
        for f in fs:
            f["now"]+=index
            for mode in ("audio","prosody","au","flame","text"):
                f[mode+"_times"]+=index
            f["audio_starts"]+=index
        label={"start":float(index),"end":float(index+1),"emotion":3,"intensity":.5,"intensity_mask":True,
               "vad":[.4,0,0],"vad_mask":[True,False,False]}
        packets.append({"protocol":TOKEN_PROTOCOL,"dialogue_id":"emotiontalk:train","start":float(index),"end":float(index+1),
                        "dt":1.,"roles":fs,"targets":[[label],[label]]})
    torch.save({"protocol":TOKEN_PROTOCOL,"cache_id":"test","packets":packets},root/"sample.pt")
    manifest={"protocol":TOKEN_PROTOCOL,"feature_sources":{"emotiontalk":fake_source()},
              "splits":{"train":["emotiontalk:train"],"val":["emotiontalk:val"],"test":[],"ood":[]},
              "dialogues":{name:{"path":"sample.pt","cache_id":"test","packets":2,"dataset":"emotiontalk"} for name in ("emotiontalk:train","emotiontalk:val")}}
    (root/"manifest.json").write_text(json.dumps(manifest))


def test_a0_training_checkpoint_roundtrip_and_contiguous_sampler(tmp_path):
    from emotion_ssm.config_v3 import default_config
    from emotion_ssm.train.observation_v3 import run
    from emotion_ssm.utils.checkpoint_v3 import load_observer
    root=tmp_path/"cache"; write_cache(root)
    data=PacketFrameDataset(root)
    sampler=DialogueBalancedBatches([data],2,world_size=2)
    assert list(sampler)==[[0,1]]
    cfg=default_config()
    cfg["observer"]=model().construction()
    cfg["train"].update(device="cpu",observation_steps=2,validate_every=1,observer_batch_size=2,amp=False,validation_max_dialogues=1)
    cfg["data"]["token_roots"]=[str(root)]
    cfg["paths"]["output"]=str(tmp_path/"output")
    path=run(cfg)
    restored,payload=load_observer(path,teacher=True)
    assert payload["config"]["train"]["max_steps"]==2
    assert payload["experiment"]["max_steps"]==2
    restored.eval()
    rebuilt=TokenObserver(payload["construction"]["observer"])
    rebuilt.load_state_dict(payload["models"]["teacher"]); rebuilt.eval()
    batch=collate_role_features([features()])
    torch.testing.assert_close(restored(batch).aff,rebuilt(batch).aff,atol=0,rtol=0)


def test_complete_token_extractor_recovers_without_external_pretrained_paths():
    from transformers import HubertConfig,RobertaConfig
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer=Tokenizer(WordLevel({"<unk>":0,"<pad>":1,"<s>":2,"</s>":3,"sad":4,"kind":5},unk_token="<unk>"))
    tokenizer.pre_tokenizer=Whitespace()
    construction={"audio":HubertConfig(hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,
        conv_dim=(8,8),conv_kernel=(10,8),conv_stride=(5,8),num_conv_pos_embedding_groups=2,num_conv_pos_embeddings=8).to_dict(),
        "text":RobertaConfig(vocab_size=6,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16).to_dict(),
        "tokenizer":tokenizer.to_str(),"special_tokens":{"unk_token":"<unk>","pad_token":"<pad>","bos_token":"<s>","eos_token":"</s>"}}
    source={"audio_model":"missing-audio-path","text_model":"missing-text-path","audio_dim":8,"text_dim":8}
    first=LocalTokenFeatures(source,construction=construction)
    second=LocalTokenFeatures(first.source,construction=first.construction())
    second.load_state_dict(first.state_dict())
    wave=torch.randn(1,16000)
    words=[{"id":"x","text":"sad","role":"A","end":.7,"available_at":.8}]
    a,b=first(wave,words,1.,0.,"A"),second(wave,words,1.,0.,"A")
    for key in a:
        torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)


def test_resume_exact_optimizer_rng_and_cursor(tmp_path):
    from emotion_ssm.config_v3 import default_config
    from emotion_ssm.train.observation_v3 import run
    root=tmp_path/"cache"; write_cache(root)
    cfg=default_config(); cfg["observer"]=model().construction()
    cfg["observer"]["dropout"]=.1
    cfg["train"].update(device="cpu",observation_steps=2,validate_every=1,observer_batch_size=2,amp=False,validation_max_dialogues=1)
    cfg["data"]["token_roots"]=[str(root)]
    cfg["paths"]["output"]=str(tmp_path/"full")
    run(cfg)
    cfg["paths"]["output"]=str(tmp_path/"split")
    cfg["train"]["observation_steps"]=1
    run(cfg)
    cfg["paths"]["resume"]=str(tmp_path/"split"/"last.pt")
    cfg["train"]["observation_steps"]=2
    run(cfg)
    first=torch.load(tmp_path/"full"/"last.pt",weights_only=False)
    second=torch.load(tmp_path/"split"/"last.pt",weights_only=False)
    for key,value in first["models"]["observer"].items():
        torch.testing.assert_close(value,second["models"]["observer"][key],atol=0,rtol=0)


def test_unlabelled_flame_calibration_exports_calibrated_teacher(tmp_path,monkeypatch):
    from emotion_ssm.config_v3 import default_config
    from emotion_ssm.train.observation_v3 import run
    root=tmp_path/"cache"; write_cache(root)
    cfg=default_config(); cfg["observer"]=model().construction()
    cfg["train"].update(device="cpu",observation_steps=1,calibration_steps=1,validate_every=1,observer_batch_size=2,amp=False,validation_max_dialogues=1)
    cfg["data"]["token_roots"]=[str(root)]
    cfg["paths"]["output"]=str(tmp_path/"a0")
    pretrained=run(cfg)
    dual=tmp_path/"dual";write_cache(dual)
    manifest=json.loads((dual/"manifest.json").read_text())
    manifest["splits"]={k:[v.replace("emotiontalk:","dualtalk:") for v in values] for k,values in manifest["splits"].items()}
    manifest["dialogues"]={k.replace("emotiontalk:","dualtalk:"):v for k,v in manifest["dialogues"].items()}
    manifest["feature_sources"]={"dualtalk":fake_source()}
    (dual/"manifest.json").write_text(json.dumps(manifest))
    payload=torch.load(dual/"sample.pt",weights_only=False)
    for packet in payload["packets"]:
        packet["targets"]=[[],[]]
        for role in packet["roles"]:
            role["domain_id"]=torch.tensor(2)
            role["au_mask"].zero_()
    torch.save(payload,dual/"sample.pt")
    cfg["data"]["dualtalk_tokens"]=str(dual)
    cfg["paths"].update(output=str(tmp_path/"calibration"),observation_checkpoint=pretrained)
    cfg["train"]["stage"]="calibration"
    result=run(cfg)
    checkpoint=torch.load(result,weights_only=False)
    assert checkpoint["config"]["train"]["max_steps"]==1
    assert checkpoint["experiment"]["max_steps"]==1
    assert checkpoint["metrics"]["domain2/V"]["teacher_samples"]>0
    assert checkpoint["metrics"]["selection_loss"]<float("inf")
    assert checkpoint["construction"]["adapter_source_domain"]==0
    assert checkpoint["config"]["data"]["adapter_source_domain"]==0
    assert checkpoint["metrics"]["adapter_source_domain"]==0
    for name,value in checkpoint["models"]["observer"].items():
        torch.testing.assert_close(value,checkpoint["models"]["teacher"][name],atol=0,rtol=0)
    assert "coordinate_teacher" in checkpoint["models"]
    for mode in ("audio","text"):
        for suffix in ("0.weight","0.bias","1.weight","1.bias"):
            torch.testing.assert_close(checkpoint["models"]["coordinate_teacher"][f"adapters.{mode}.0.{suffix}"],
                                       checkpoint["models"]["coordinate_teacher"][f"adapters.{mode}.2.{suffix}"],atol=0,rtol=0)
    # Resume needs neither the old A0 file nor another semantic adapter copy.
    def must_not_copy(*args,**kwargs):
        raise AssertionError("Adapter copy must not run during calibration resume")
    monkeypatch.setattr("emotion_ssm.train.observation_v3.initialize_dualtalk_adapters",must_not_copy)
    cfg["paths"].update(resume=str(tmp_path/"calibration"/"last.pt"),observation_checkpoint="missing-A0.pt")
    cfg["train"]["calibration_steps"]=2
    run(cfg)


def labelled_provenance(sources):
    return {"provenance":{name:{"feature_sources":{name:source},"splits":{"train":[name+":dialogue"]},"sha256":"digest-"+name}
                          for name,source in sources.items()}}


def test_calibration_transfers_english_iemocap_adapter_and_preserves_flame():
    from emotion_ssm.train.observation_v3 import initialize_dualtalk_adapters
    observer=model()
    payload=labelled_provenance({"emotiontalk":fake_source("chinese"),"iemocap":fake_source("english"),"dualtalk":fake_source("english")})
    visual=copy.deepcopy(observer.adapters["flame"][2].state_dict())
    binding=initialize_dualtalk_adapters(observer,payload,fake_source("english"))
    assert binding["source_domain"]==1 and binding["source_dataset"]=="iemocap"
    for mode in ("audio","text"):
        for key,value in observer.adapters[mode][1].state_dict().items():
            torch.testing.assert_close(value,observer.adapters[mode][2].state_dict()[key],atol=0,rtol=0)
    for key,value in visual.items():
        torch.testing.assert_close(value,observer.adapters["flame"][2].state_dict()[key],atol=0,rtol=0)


def test_calibration_emotiontalk_only_uses_compatible_domain_zero():
    from emotion_ssm.train.observation_v3 import select_labelled_adapter_source
    source=fake_source("chinese")
    binding=select_labelled_adapter_source(labelled_provenance({"emotiontalk":source,"dualtalk":source}),source)
    assert binding["source_domain"]==0
    assert binding["matching_labelled_domains"]==[0]


@pytest.mark.parametrize("change",[{"audio_model":"different-same-dimension-model"},{"text_revision":"different-weights"}])
def test_calibration_rejects_same_dimension_incompatible_source(change):
    from emotion_ssm.train.observation_v3 import select_labelled_adapter_source
    source=fake_source("english")
    with pytest.raises(ValueError,match="fully compatible"):
        select_labelled_adapter_source(labelled_provenance({"iemocap":source,"dualtalk":{**source,**change}}),{**source,**change})


def test_calibration_domain_must_exist_in_a0_training_split():
    from emotion_ssm.train.observation_v3 import select_labelled_adapter_source
    source=fake_source("english")
    payload=labelled_provenance({"iemocap":source,"dualtalk":source})
    payload["provenance"]["iemocap"]["splits"]["train"]=[]
    with pytest.raises(ValueError,match="No labelled"):
        select_labelled_adapter_source(payload,source)
