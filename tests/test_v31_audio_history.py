"""Causal acoustic history, independent atoms and raw/cache agreement."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from emotion_ssm.data.packets_v3 import (TOKEN_PROTOCOL, merge_audio_history, prefix_role_features,
                                        collate_role_features, validate_packet)
from emotion_ssm.preprocess.tokens_v3 import LocalTokenFeatures, build_role_features
from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.models.streaming_v3 import StreamingAvatarV3
from emotion_ssm.models.token_observer import TokenObserver


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class FakeAudio(nn.Module):
    def __init__(self,norm="group"):
        super().__init__()
        self.config=SimpleNamespace(conv_kernel=[400],conv_stride=[320],hidden_size=8,feat_extract_norm=norm)
        self.conv=nn.Conv1d(1,8,400,320)
        self.context=nn.Linear(8,8)
        self.calls=[]

    def _get_feat_extract_output_lengths(self,lengths):
        return (lengths-400)//320+1

    def forward(self,wave,attention_mask=None):
        self.calls.append((wave.shape,attention_mask))
        h=self.conv(wave[:,None]).transpose(1,2)
        # This deliberately carries the complete input unit's context so the
        # independence test would catch concatenating adjacent acoustic units.
        h=h+self.context(h.mean(1,keepdim=True))
        return SimpleNamespace(last_hidden_state=h)


class FakeText(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(128,8)
    def get_input_embeddings(self):
        return self.embed


class FakeBackbone(nn.Module):
    def __init__(self,norm="group"):
        super().__init__()
        self.source={"audio_dim":8,"text_dim":8,"audio_model":"fake-full-audio","text_model":"fake-lexical"}
        self.audio,self.text=FakeAudio(norm),FakeText()
        self.tokenizer=lambda word,**kw:{"input_ids":[ord(c)%128 for c in word]}
    def construction(self):
        return {"test":"v31-full-atoms"}


def extractor(norm="group"):
    torch.manual_seed(3)
    return LocalTokenFeatures({},backbone=FakeBackbone(norm))


def test_independent_complete_backbone_atoms_and_prosody_mask_alignment():
    local=extractor()
    wave=torch.randn(24000)
    first=local.audio_tokens(wave,0.)
    changed=wave.clone();changed[8000:16000]=changed[8000:16000].flip(0)*7
    second=local.audio_tokens(changed,0.)
    assert local.source["preprocessing"]==TOKEN_PROTOCOL
    assert local.source["token_ms"]==500
    assert first[0].shape==(3,8) and first[3].shape==(3,8)
    torch.testing.assert_close(first[0][[0,2]],second[0][[0,2]],atol=0,rtol=0)
    torch.testing.assert_close(first[3][[0,2]],second[3][[0,2]],atol=0,rtol=0)
    assert local.backbone.audio.calls[0][0]==(3,8000)
    assert local.backbone.audio.calls[0][1] is None
    assert first[2].tolist()==[.5,1.,1.5]
    layered=extractor("layer")
    layered.audio_tokens(wave[:16000],0.)
    assert layered.backbone.audio.calls[0][1].shape==(2,8000)


def test_raw_energy_survives_normalization_and_short_silent_audio_is_missing():
    local=extractor()
    wave=torch.randn(8000)*.02
    low=local.audio_tokens(wave,0.)
    high=local.audio_tokens(wave*10,0.)
    torch.testing.assert_close(low[0],high[0],atol=5e-4,rtol=5e-4)
    assert float(high[3][0,0]-low[3][0,0])==pytest.approx(2.302585,abs=.004)
    assert not local.audio_tokens(torch.zeros(16000),0.)[1].any()
    assert not local.audio_tokens(torch.ones(120),0.)[1].any()
    assert local.audio_tokens(torch.randn(450),0.)[1].all()
    assert torch.isfinite(local.audio_tokens(torch.zeros(0),0.)[3]).all()


def test_batched_preprocessing_matches_per_packet_and_causal_fractional_endpoint():
    local=extractor()
    wave=torch.randn(50560)  # 3.16 seconds: three full blocks and one tail.
    all_atoms=local.audio_tokens(wave,0.)
    for start,end in ((0.,1.),(1.,2.),(2.,3.),(3.,3.16)):
        a,b=round(start*16000),round(end*16000)
        direct=build_role_features(local,wave[a:b],[],start,end,"A",0)
        cached=build_role_features(local,wave[a:b],[],start,end,"A",0,
                                  audio_precomputed=tuple(v[int(start*2):int(__import__('math').ceil(end*2))] for v in all_atoms))
        for key in ("audio_tokens","audio_times","audio_starts","audio_mask","prosody_tokens"):
            torch.testing.assert_close(direct[key],cached[key],atol=5e-4,rtol=5e-4)
    start,end=176.,176.26597046774344
    tail=torch.randn(round((end-start)*16000));changed=tail.clone();changed[-1]=999.
    one=build_role_features(local,tail,[],start,end,"A",0)
    two=build_role_features(local,changed,[],start,end,"A",0)
    assert one["audio_times"].max()<end
    torch.testing.assert_close(one["audio_tokens"],two["audio_tokens"],atol=0,rtol=0)


def test_sentence_endpoint_reads_history_and_old_sound_does_not_repeat_action():
    local=extractor()
    previous=None
    for second in range(4):
        wave=torch.randn(16000) if second<3 else torch.zeros(16000)
        current=build_role_features(local,wave,[],second,second+1,"A",0)
        previous=merge_audio_history(previous,current)
    assert len(previous["audio_tokens"])==8
    assert previous["modality_mask"][0]
    assert not previous["fresh_observation"][0]
    assert not previous["action_present"] and previous["action_duration"]==0
    endpoint=prefix_role_features(previous,3.2)
    assert endpoint["audio_mask"].sum()==6
    assert float(endpoint["audio_starts"][endpoint["audio_mask"]].min())==0
    assert not endpoint["audio_fresh_mask"].any()
    assert not endpoint["action_present"]
    assert not (endpoint["audio_mask"] & (endpoint["audio_times"]>3.2)).any()
    # A full later window contains only its most recent sixteen seconds.
    for second in range(4,19):
        previous=merge_audio_history(previous,build_role_features(local,torch.randn(16000),[],second,second+1,"A",0))
    assert previous["audio_starts"].min()==3.
    assert len(previous["audio_tokens"])==32


def test_text_has_lexical_order_distinct_from_availability_and_late_arrivals():
    local=extractor()
    words=[{"id":"w10","text":"B","role":"A","start":.2,"end":.3,"available_at":1.},
           {"id":"w2","text":"A","role":"A","start":.1,"end":.2,"available_at":1.},
           {"id":"late","text":"C","role":"B","start":.3,"end":.4,"available_at":2.}]
    first=local.text_tokens(words,1.,0.,"A")
    expected=local.backbone.text.get_input_embeddings()(torch.tensor([ord('A'),ord('B')])).half()
    torch.testing.assert_close(first["text_tokens"],expected)
    assert first["text_positions"].tolist()==[0,1]
    later=local.text_tokens(words,2.,1.,"A")
    assert later["text_fresh_mask"].tolist()==[False,False,True]
    assert first["text_positions"].tolist()==[0,1]


class TinyGenerator(nn.Module):
    def __init__(self,context_dim):
        super().__init__()
        self.context_dim=context_dim
        self.output=nn.Linear(context_dim,56)
    def forward(self,a,b,v,c,enabled=True):
        return self.output(c) if enabled else v*0


def test_raw_stream_uses_same_audio_history_and_target_flame_is_not_observed():
    local=extractor()
    obs=TokenObserver(dict(audio_dim=8,text_dim=8,model_dim=16,affect_dim=8,num_heads=2,num_layers=1,dropout=0.))
    core=UnifiedEmotionStateCore(observation_dim=8,relation_dim=4,hidden_dim=16)
    model=StreamingAvatarV3(TinyGenerator(core.context_dim),obs,core,features=local).eval()
    state=None
    expected=[None,None]
    waves=torch.randn(2,48000)
    waves[:,32000:]=0
    saved=None
    for i in range(3):
        packet={"session_id":"d","roles":("A","B"),"time":float(i+1),
                "target_audio":waves[0:1,i*16000:(i+1)*16000],
                "partner_audio":waves[1:2,i*16000:(i+1)*16000],
                "target_speech_active":i<2,"partner_speech_active":i<2,
                "partner_blendshape":torch.zeros(1,25,56),"partner_visual_mask":torch.zeros(1,25,dtype=torch.bool),
                "target_blendshape":torch.randn(1,25,56)*100}
        with torch.no_grad():
            out,state,d=model(packet,state)
        if i==0:
            saved=out.clone()
        state=state.detach()
        for r,role in enumerate(("A","B")):
            cur=build_role_features(local,waves[r,i*16000:(i+1)*16000],[],float(i),float(i+1),role,2)
            expected[r]=merge_audio_history(expected[r],cur)
            actual=d["observation_inputs"][r]
            for key in ("audio_tokens","audio_times","audio_starts","audio_mask","prosody_tokens"):
                torch.testing.assert_close(actual[key][0],expected[r][key],atol=5e-4,rtol=5e-4,check_dtype=False)
        assert not d["observation_inputs"][0]["flame_mask"].any()
        if i==2:
            assert d["target_modalities"][0,0]
            assert not d["target_fresh"].any()
            assert d["observations"][0].action_duration.item()==0
    assert len(state.audio_history["A"]["audio_tokens"])==6
    assert state.history[0]["target_audio"].shape[1]==48000
    assert saved.shape==(1,25,56)


def test_stream_preserves_current_block_text_clock_but_late_text_cannot_rewrite_past():
    local=extractor()
    obs=TokenObserver(dict(audio_dim=8,text_dim=8,model_dim=16,affect_dim=8,num_heads=2,num_layers=1,dropout=0.))
    core=UnifiedEmotionStateCore(observation_dim=8,relation_dim=4,hidden_dim=16)
    model=StreamingAvatarV3(TinyGenerator(core.context_dim),obs,core,features=local).eval()
    initial=model.initial_state("d",("A","B"))
    first=model._words({"time":1.,"words":[{"id":"on-time","text":"A","role":"A",
                        "start":.1,"end":.3,"available_at":.8}]},initial)
    assert first["on-time"]["available_at"]==.8
    from dataclasses import replace
    state=replace(initial,time=1.,words=first)
    second=model._words({"time":2.,"words":[{"id":"late","text":"B","role":"A",
                         "start":.2,"end":.3,"available_at":.4}]},state)
    assert second["late"]["available_at"]==2.
    assert "late" not in state.words


def test_raw_stream_detects_quiet_short_valid_audio_independently_of_padding():
    local=extractor()
    obs=TokenObserver(dict(audio_dim=8,text_dim=8,model_dim=16,affect_dim=8,num_heads=2,num_layers=1,dropout=0.))
    core=UnifiedEmotionStateCore(observation_dim=8,relation_dim=4,hidden_dim=16)
    model=StreamingAvatarV3(TinyGenerator(core.context_dim),obs,core,features=local).eval()
    wave=torch.zeros(1,16000)
    wave[:,:1024]=torch.randn(1,1024)*2.5e-4
    assert wave.square().mean().sqrt()<model.speech_rms_threshold
    assert wave[:,:1024].square().mean().sqrt()>model.speech_rms_threshold
    packet={"session_id":"d","roles":("A","B"),"time":1.,
            "target_audio":wave,"partner_audio":wave.clone(),
            "target_audio_length":1024,"partner_audio_length":1024,
            "partner_blendshape":torch.zeros(1,25,56),"partner_visual_mask":torch.zeros(1,25,dtype=torch.bool)}
    with torch.no_grad():
        out,_,first=model(packet)
    assert first["target_fresh"][0,0]
    assert first["observations"][0].action_duration.item()==pytest.approx(1024/16000)
    changed=dict(packet)
    changed["target_audio"]=wave.clone();changed["target_audio"][:,1024:]=9999
    with torch.no_grad():
        second_out,_,second=model(changed)
    torch.testing.assert_close(first["target_aff"],second["target_aff"],atol=0,rtol=0)
    torch.testing.assert_close(out,second_out,atol=0,rtol=0)
