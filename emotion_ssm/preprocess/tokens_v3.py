"""Build source-bound, second-clock tokens without contextual-hidden leakage.

CLI: python -m emotion_ssm.preprocess.tokens_v3 --config configuration.json
Raw datasets, old pooled caches and CTC alignment files are read-only inputs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from emotion_ssm.data.packets_v3 import (TOKEN_PROTOCOL, SUPERVISION_REVISION, AUDIO_HISTORY_SECONDS, empty_role_features,
                                       endpoint_label, validate_packet, merge_audio_history)
from emotion_ssm.data.protocol import DOMAINS, fingerprint


class LocalTokenFeatures(nn.Module):
    """Independent half-second pretrained acoustic units and lexical embeddings.

    The complete frozen acoustic backbone contextualizes only the samples of
    one atom. Different atoms never share normalization or self-attention. A
    masked atom therefore cannot leak through an unmasked cached neighbor.
    """
    def __init__(self, source, construction=None, local_files_only=True, backbone=None):
        super().__init__()
        from emotion_ssm.models.features import FrozenFeatures
        self.backbone = backbone if backbone is not None else FrozenFeatures(
            source, construction=construction, local_files_only=local_files_only)
        self.source = dict(self.backbone.source)
        self.source.update(preprocessing=TOKEN_PROTOCOL, token_audio="independent-500ms-full-backbone-v1",
                           token_text="lexical-input-embedding-explicit-order-v2", token_ms=500,
                           normalization="per-500ms-zscore-with-raw-prosody-v2",
                           token_clock="float64-floor-sample-end-v2",
                           audio_prosody="log-rms-log-std-mean-zcr-log-peak-log-crest-voice-energy-cv-v1",
                           prosody_dim=8, audio_history_seconds=AUDIO_HISTORY_SECONDS)
        self.requires_grad_(False)
        self.eval()
        self._text_cache_key = None
        self._text_cache = None

    def train(self, mode=True):
        return super().train(False)

    def construction(self):
        return self.backbone.construction()

    @torch.no_grad()
    def forward(self, waveform, words, now, since, role, speech_active=True, audio_length=None):
        from emotion_ssm.data.packets_v3 import collate_role_features
        length = waveform.shape[-1] if audio_length is None else int(audio_length)
        if waveform.ndim != 2 or len(waveform) != 1 or not 0 <= length <= waveform.shape[-1]:
            raise ValueError("Local token inference expects one valid waveform [1,T]")
        values = build_role_features(self, waveform[0,:length], words, float(since), float(now), role, 2)
        if not speech_active:
            values["audio_mask"].zero_()
            values["prosody_mask"].zero_()
            values["audio_fresh_mask"].zero_()
            values["modality_mask"][0] = False
            values["fresh_observation"][0] = False
            values["action_present"] = torch.tensor(False)
            values["action_duration"] = torch.tensor(0.)
        return collate_role_features([values], waveform.device)

    @torch.no_grad()
    def audio_tokens(self, waveform, start, sample_mask=None):
        audio = self.backbone.audio
        device = next(audio.parameters()).device
        waveform = torch.as_tensor(waveform, dtype=torch.float32, device=device).flatten()
        length, window = len(waveform), 8000
        count = max(1, math.ceil(length/window))
        padded = F.pad(waveform, (0, count*window-length)).view(count, window)
        valid_lengths = (length - torch.arange(count, device=device)*window).clamp(0, window)
        from emotion_ssm.utils.audio import minimum_waveform_length
        valid = valid_lengths >= minimum_waveform_length(audio)
        if sample_mask is not None:
            sample_mask = F.pad(torch.as_tensor(sample_mask, device=device).bool(), (0, count*window-length)).view(count, window)
            valid &= sample_mask.sum(-1) == valid_lengths
        positions = torch.arange(window, device=device)[None] < valid_lengths[:, None]
        mean = (padded*positions).sum(-1, keepdim=True)/valid_lengths[:, None].clamp_min(1)
        variance = ((padded-mean).square()*positions).sum(-1, keepdim=True)/valid_lengths[:, None].clamp_min(1)
        normalized = ((padded-mean)/variance.sqrt().clamp_min(1e-6))*positions
        rms = ((padded.square()*positions).sum(-1)/valid_lengths.clamp_min(1)).sqrt()
        # Silence does not become speech after normalization. This is an energy
        # floor, not a claim that all valid non-silent tokens are spoken words.
        valid &= rms >= 1e-4
        values = waveform.new_zeros(count, self.source["audio_dim"])
        # Equal-length groups avoid padding-sensitive group normalization and
        # still batch the usual full half-second atoms. Layer-normalized models
        # receive their configured attention mask. Pool only valid hidden frames.
        for valid_length in valid_lengths[valid].unique():
            indices=(valid & (valid_lengths==valid_length)).nonzero().flatten()
            # A dialogue can contain thousands of atoms. Bound activation
            # memory while amortizing the frozen Transformer over 32 atoms.
            for selected in indices.split(32):
                local=normalized[selected,:int(valid_length)]
                kwargs={}
                if getattr(audio.config,"feat_extract_norm","group")!="group":
                    kwargs["attention_mask"]=torch.ones_like(local,dtype=torch.long)
                hidden=audio(local,**kwargs).last_hidden_state
                output_length=audio._get_feat_extract_output_lengths(valid_length).clamp(min=0,max=hidden.shape[1])
                hidden_mask=torch.arange(hidden.shape[1],device=device)<output_length
                values[selected]=(hidden*hidden_mask[None,:,None]).sum(1)/hidden_mask.sum().clamp_min(1)
        peak=(padded.abs()*positions).max(-1).values
        crossing=((padded[:,1:]*padded[:,:-1]<0) & positions[:,1:] & positions[:,:-1]).sum(-1)
        zcr=crossing/(valid_lengths-1).clamp_min(1)
        frames=padded.view(count,25,320)
        frame_positions=positions.view(count,25,320)
        frame_counts=frame_positions.sum(-1)
        frame_rms=((frames.square()*frame_positions).sum(-1)/frame_counts.clamp_min(1)).sqrt()
        frame_valid=frame_counts>0
        frame_count=frame_valid.sum(-1).clamp_min(1)
        frame_mean=(frame_rms*frame_valid).sum(-1)/frame_count
        frame_std=(((frame_rms-frame_mean[:,None]).square()*frame_valid).sum(-1)/frame_count).sqrt()
        log_rms=rms.clamp_min(1e-7).log()
        log_peak=peak.clamp_min(1e-7).log()
        prosody=torch.stack((log_rms,variance[:,0].sqrt().clamp_min(1e-7).log(),mean[:,0],zcr,
                            log_peak,log_peak-log_rms,((frame_rms>=1e-4)&frame_valid).sum(-1)/frame_count,
                            frame_std/frame_mean.clamp_min(1e-7)),dim=-1)
        prosody=torch.where(valid[:,None],prosody,0.)
        ends = float(start) + (torch.arange(count, device=device)*window+valid_lengths).double()/16000.
        return values.cpu().half(), valid.cpu(), ends.cpu(), prosody.cpu().half()

    @torch.no_grad()
    def text_tokens(self, words, now, since, query_role, max_tokens=256):
        device = next(self.backbone.text.parameters()).device
        # Arrival time and lexical order have different meanings. Preserve
        # source/acoustic order among simultaneously available words rather than
        # lexicographically sorting IDs such as word-10 before word-2.
        ordered = sorted(enumerate(words), key=lambda pair: (float(pair[1]["available_at"]),
                         float(pair[1].get("start",pair[1]["end"])),pair[0]))
        ordered=[word for _,word in ordered]
        key = tuple((str(w.get("id", "")),str(w["text"]),str(w["role"]),float(w["end"]),float(w["available_at"])) for w in ordered)
        if key != self._text_cache_key:
            all_ids, all_roles, all_times = [], [], []
            for word in ordered:
                if float(word["available_at"]) < float(word["end"]):
                    raise ValueError("Text availability precedes its acoustic endpoint")
                token_ids = self.backbone.tokenizer(str(word["text"]), add_special_tokens=False)["input_ids"]
                all_ids.extend(token_ids)
                all_roles.extend([str(word["role"])]*len(token_ids))
                all_times.extend([float(word["available_at"])]*len(token_ids))
            embeddings = (self.backbone.text.get_input_embeddings()(torch.tensor(all_ids,device=device)).cpu().half()
                          if all_ids else torch.zeros(0,self.source["text_dim"],dtype=torch.float16))
            self._text_cache = (embeddings, all_roles, torch.tensor(all_times,dtype=torch.float64))
            self._text_cache_key = key
        embeddings, all_roles, all_times = self._text_cache
        stop = int((all_times <= now).sum())
        start = max(0,stop-max_tokens)
        if stop == start:
            return {"text_tokens": torch.zeros(1, self.source["text_dim"],dtype=torch.float16), "text_mask": torch.zeros(1, dtype=torch.bool),
                    "text_roles": torch.zeros(1, dtype=torch.long), "text_times": torch.tensor([float(now)],dtype=torch.float64),
                    "text_fresh_mask": torch.zeros(1, dtype=torch.bool),"text_positions":torch.zeros(1,dtype=torch.long)}
        # Slices share one lexical storage across both roles and all packets;
        # torch.save preserves sharing instead of writing a context per second.
        return {"text_tokens": embeddings[start:stop], "text_mask": torch.ones(stop-start,dtype=torch.bool),
                "text_roles": torch.tensor([int(r != str(query_role)) for r in all_roles[start:stop]]),
                "text_times": all_times[start:stop], "text_fresh_mask": all_times[start:stop] > since,
                "text_positions":torch.arange(stop-start,dtype=torch.long)}


def build_role_features(extractor, waveform, words, start, end, role, domain,
                        visual=None, visual_type="au", visual_mask=None, sample_mask=None, audio_precomputed=None):
    values = empty_role_features(extractor.source["audio_dim"], extractor.source["text_dim"], end, end-start, domain)
    # The right edge of a token represents a half-open sample interval. Never
    # use round() here: a fractional endpoint may round into the next sample.
    valid_samples=math.floor((float(end)-float(start))*16000+1e-9)
    waveform=torch.as_tensor(waveform).flatten()[:valid_samples]
    if sample_mask is not None:
        sample_mask=torch.as_tensor(sample_mask).flatten()[:valid_samples]
    a, am, at, prosody = (extractor.audio_tokens(waveform, start, sample_mask)
                          if audio_precomputed is None else audio_precomputed)
    if len(a)<1 or (at.double()>end+1e-9).any():
        raise ValueError("Precomputed atoms must be causal and belong to the current packet")
    starts=float(start)+torch.arange(len(a),dtype=torch.float64)*.5
    values.update(audio_tokens=a, audio_mask=am, audio_times=at,audio_starts=starts,
                  audio_fresh_mask=am.clone(),prosody_tokens=prosody,prosody_mask=am.clone(),prosody_times=at)
    values.update(extractor.text_tokens(words, end, start, role))
    if visual is not None and len(visual):
        visual = torch.as_tensor(visual).float()
        if visual_type not in ("au", "flame"):
            raise ValueError("Visual adapter type must be declared AU or FLAME")
        valid = torch.isfinite(visual).all(-1)
        if visual_mask is not None:
            valid &= torch.as_tensor(visual_mask).bool()
        values[visual_type + "_tokens"] = torch.nan_to_num(visual)
        values[visual_type + "_mask"] = valid
        values[visual_type + "_times"] = torch.linspace(start, end, len(visual)+1,dtype=torch.float64)[1:]
    available = torch.tensor([bool(am.any()), bool(values["au_mask"].any() | values["flame_mask"].any()), bool(values["text_mask"].any())])
    new_own = values["text_fresh_mask"] & (values["text_roles"] == 0) & values["text_mask"]
    new_any = values["text_fresh_mask"] & values["text_mask"]
    fresh = available.clone()
    fresh[2] = new_any.any()
    # Shared partner text supplies new evidence about SELF through context, but
    # is not SELF's external speaking action or its own lexical event.
    action = bool(am.any() | available[1])
    audio_duration = min(end-start, float(((at-starts).clamp_min(0)*am).sum()))
    visual_duration = end-start if bool(available[1]) else 0.
    values.update(modality_mask=available, fresh_observation=fresh,
                  context_available=available[2], event_present=new_own.any(),
                  action_present=torch.tensor(action),
                  action_duration=torch.tensor(max(audio_duration, visual_duration) if action else 0.))
    return values


def _load_wave(path):
    import librosa
    return torch.from_numpy(librosa.load(str(path), sr=16000)[0])


def _upstream_dialogue(source, name, extractor):
    from emotion_ssm.data.common import load_pt, read_json
    from emotion_ssm.data.feature_store import resolve_utterances, resolve_speaker_id, unpack_face_data
    root = Path(source["input_root"])
    folder = root / "dialogues" / name
    if not folder.is_dir():
        folder = root / name
    labels = read_json(folder / "labels.json")
    utterances = resolve_utterances(labels)
    speakers = sorted({resolve_speaker_id(item, name) for item in utterances})
    if len(speakers) > 2 or not utterances:
        raise ValueError(f"{name}: invalid dyadic utterance list")
    while len(speakers) < 2:
        speakers.append(f"{name}:missing_partner")
    mapping = source["_audio_mapping"]
    uids = [str(item.get("utterance_id", item.get("id", i))) for i, item in enumerate(utterances)]
    aus, confs, masks, normalized = unpack_face_data(load_pt(folder / "face_au_features.pt"), uids)
    provenance = read_json(folder / "provenance.json") if (folder / "provenance.json").exists() else {}
    verified = source["dataset"] != "iemocap" or provenance.get("face_identity_verified", False)
    if not normalized:
        stats = load_pt(root / "metadata" / f"fold_{source.get('fold', 5)}_face_stats.pt")
        aus = [(torch.as_tensor(au) - stats["mean"]) / stats["std"].clamp_min(1e-6) for au in aus]
    end = max(float(i["end_time"]) for i in utterances)
    samples = math.ceil(end*16000)
    waves, coverage = torch.zeros(2, samples), torch.zeros(2, samples, dtype=torch.uint8)
    frames = math.ceil(end*25)
    visuals, visual_masks = torch.zeros(2, frames, 35), torch.zeros(2, frames, dtype=torch.bool)
    words, grouped_labels, signatures = [], [[], []], []
    for index, (item, uid) in enumerate(zip(utterances, uids)):
        role = speakers.index(resolve_speaker_id(item, name))
        start, stop = float(item["start_time"]), float(item["end_time"])
        if not 0 <= start < stop:
            raise ValueError(f"{name}/{uid}: invalid source times")
        path = mapping.get(f"{name}/{uid}", mapping.get(uid))
        if path is None:
            raise ValueError(f"No explicit WAV mapping for {name}/{uid}")
        path = Path(path).resolve(strict=True)
        wave = _load_wave(path)
        a = round(start*16000)
        count = min(len(wave), max(0, math.floor(stop*16000+1e-9)-a))
        waves[role, a:a+count] += wave[:count]
        coverage[role, a:a+count] += 1
        signatures.append([str(path), path.stat().st_size, path.stat().st_mtime_ns])
        text = str(item.get("text", item.get("transcript", ""))).strip()
        if text:
            available = float(item.get("text_available_at", stop))
            if available < stop:
                raise ValueError("Upstream transcript is not allowed before utterance endpoint")
            words.append({"id": uid, "text": text, "start": start, "end": stop,
                          "available_at": available, "role": role})
        grouped_labels[role].append(endpoint_label(item, source["dataset"]))
        au = torch.as_tensor(aus[index]).float()
        valid = torch.as_tensor(masks[index]).bool() & (torch.as_tensor(confs[index]) > 0) & torch.isfinite(au).all(-1)
        fa, fb = round(start*25), min(frames, round(stop*25))
        if verified and fb > fa and len(au):
            indices = torch.linspace(0, len(au)-1, fb-fa).round().long()
            # Same-role overlapping face tracks have no unique visible identity.
            overlap = visual_masks[role, fa:fb].clone()
            visuals[role, fa:fb] = torch.nan_to_num(au[indices])
            visual_masks[role, fa:fb] = valid[indices] & ~overlap
    ambiguous = coverage > 1
    waves[ambiguous] = 0
    valid_sample_count=math.floor(end*16000+1e-9)
    all_audio=[extractor.audio_tokens(waves[r,:valid_sample_count],0.,~ambiguous[r,:valid_sample_count]) for r in (0,1)]
    packets = []
    audio_history=[None,None]
    for block in range(math.ceil(end)):
        start, stop = float(block), min(end, block+1.)
        a, b = round(start*16000), math.floor(stop*16000+1e-9)
        fa, fb = round(start*25), round(stop*25)
        roles = [build_role_features(extractor, waves[r, a:b], words, start, stop, r, DOMAINS[source["dataset"]],
                                    visuals[r, fa:fb], "au", visual_masks[r, fa:fb], ~ambiguous[r, a:b],
                                    audio_precomputed=tuple(value[int(start*2):math.ceil(stop*2)] for value in all_audio[r])) for r in (0, 1)]
        roles=[merge_audio_history(audio_history[r],roles[r],extractor.source["audio_history_seconds"]) for r in (0,1)]
        audio_history=roles
        targets = [[label for label in grouped_labels[r] if start < label["end"] <= stop+1e-6] for r in (0, 1)]
        packet = dict(protocol=TOKEN_PROTOCOL, dialogue_id=source["dataset"]+":"+name,
                      start=start, end=stop, dt=stop-start, roles=roles, targets=targets,
                      source_domain=DOMAINS[source["dataset"]], text_protocol="offline_transcript_endpoint")
        validate_packet(packet)
        packets.append(packet)
    return packets, {"same_role_overlap_samples": int(ambiguous.sum()), "unverified_visual_masked": not verified,
                     "au_times": "uniform_within_declared_utterance", "raw_signature": signatures,
                     "label_digest": fingerprint(labels), "speakers": speakers,
                     "audio_history_seconds":extractor.source["audio_history_seconds"],
                     "endpoint_utterances":len(utterances),
                     "utterances_longer_than_audio_history":sum(float(i["end_time"])-float(i["start_time"])>
                         extractor.source["audio_history_seconds"] for i in utterances),
                     "endpoint_audio_policy":"past16s-completed-independent500ms-atoms; final <500ms atom may be unavailable at in-block endpoint"}


def _dualtalk_dialogue(source, name, split, extractor):
    from emotion_ssm.data.dualtalk import partner_stem
    from emotion_ssm.data.timed_dualtalk import load_flame
    raw_split = "train" if split in ("train", "val") else split
    folder = Path(source["input_root"]) / raw_split
    cache = torch.load(Path(source["timed_root"]) / raw_split / (name+".pt"), map_location="cpu", weights_only=False)
    if "aligned_words" not in cache:
        raise ValueError("Old timed cache lacks explicit aligned_words; do not substitute pooled text")
    roles = ("speaker1" if name.endswith("speaker1") else "speaker2", "speaker2" if name.endswith("speaker1") else "speaker1")
    stems = (name, partner_stem(name))
    waves = [_load_wave(folder/(stem+".wav")) for stem in stems]
    flames = [load_flame(folder/(stem+".npz")) for stem in stems]
    frames = min(*(len(f) for f in flames), int(min(len(w) for w in waves)/16000*25))
    valid_sample_count=math.floor(frames/25*16000+1e-9)
    all_audio=[extractor.audio_tokens(waves[r][:valid_sample_count],0.) for r in (0,1)]
    packets = []
    audio_history=[None,None]
    for fa in range(0, frames, 25):
        fb = min(fa+25, frames)
        start, stop = fa/25, fb/25
        a, b = round(start*16000), math.floor(stop*16000+1e-9)
        # Directed generation protocol: slot 0 is Avatar. Its current FLAME
        # appears exclusively below in target_flame, never in roles[0].
        features = [build_role_features(extractor, waves[r][a:b], cache["aligned_words"], start, stop, roles[r], 2,
                                       None if r == 0 else flames[r][fa:fb], "flame",
                                       audio_precomputed=tuple(value[int(start*2):math.ceil(stop*2)] for value in all_audio[r])) for r in (0, 1)]
        features=[merge_audio_history(audio_history[r],features[r],extractor.source["audio_history_seconds"]) for r in (0,1)]
        audio_history=features
        truths = torch.stack([flame[fa:fb] for flame in flames])
        packet = dict(protocol=TOKEN_PROTOCOL, dialogue_id="dualtalk:"+name, start=start, end=stop, dt=stop-start,
                      roles=features, targets=[[], []], avatar_role=0, source_domain=2,
                      target_flame=torch.nan_to_num(truths), target_frame_mask=torch.isfinite(truths).all(-1),
                      text_protocol=cache["text_protocol"], raw_stems=stems)
        validate_packet(packet)
        packets.append(packet)
    files = [folder/(stem+suffix) for stem in stems for suffix in (".wav", ".npz")]
    return packets, {"alignment_cache_id": cache["cache_id"],"alignment_protocol":cache["text_protocol"],
                     "aligned_word_count":len(cache["aligned_words"]),"text_missing_roles":[role for role in roles if not any(str(w["role"])==role for w in cache["aligned_words"])],
                     "audio_history_seconds":extractor.source["audio_history_seconds"],
                     "raw_signature": [[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in files]}


def _source_splits(source):
    root = Path(source["input_root"])
    if source["dataset"] == "dualtalk":
        manifest = json.loads(Path(source.get("split_manifest", Path(source["timed_root"])/"splits.json")).read_text(encoding="utf-8"))
        output = {s: list(manifest[s]) for s in ("train", "val")}
        for split in ("test", "ood"):
            output[split] = sorted(p.stem for p in (root/split).glob("*.npz"))
        return output
    split_root = root/"splits"
    if source.get("fold") is not None:
        split_root = split_root / f"fold_{source['fold']}"
    output = {}
    for split in ("train", "val", "test"):
        paths = [split_root/(split+"_dialogues.txt"), split_root/(split+".txt")]
        if split == "val":
            paths += [split_root/"valid_dialogues.txt", split_root/"validation_dialogues.txt"]
        path = next((p for p in paths if p.exists()), None)
        output[split] = [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()] if path else []
    if not output["train"] or not output["val"]:
        raise ValueError(f"{source['dataset']}: explicit train and validation source lists are required")
    return output


def run(config):
    output = Path(config["output_root"]).resolve()
    for source in config["sources"]:
        if output == Path(source["input_root"]).resolve() or Path(source["input_root"]).resolve() in output.parents:
            raise ValueError("Token output must not be inside an input dataset")
    output.mkdir(parents=True, exist_ok=True)
    (output/"dialogues").mkdir(exist_ok=True)
    manifest = {"protocol": TOKEN_PROTOCOL, "supervision_revision": SUPERVISION_REVISION,
                "feature_sources": {}, "splits": {s: [] for s in ("train","val","test","ood")}, "dialogues": {}, "data_gaps": [],"quality":{}}
    errors = []
    for raw_source in config["sources"]:
        source = copy.deepcopy(raw_source)
        dataset = source["dataset"]
        if dataset not in DOMAINS:
            raise ValueError("Unknown dataset")
        if "audio_manifest" in source:
            source["_audio_mapping"] = json.loads(Path(source["audio_manifest"]).read_text(encoding="utf-8"))
        specification = {k: source[k] for k in ("audio_model", "text_model")}
        specification.update(audio_dim=source.get("audio_dim",768), text_dim=source.get("text_dim",768))
        extractor = LocalTokenFeatures(specification, local_files_only=source.get("local_files_only", True)).to(source.get("device", config.get("device", "cpu")))
        manifest["feature_sources"][dataset] = extractor.source
        quality=manifest["quality"].setdefault(dataset,{"role_packets":0,"modality_available_packets":{"A":0,"V":0,"T":0},
            "tokens":{m:{"total":0,"valid":0} for m in ("audio","prosody","au","flame","text")},"text_missing_dialogues":[],"text_missing_role_streams":0,
            "endpoint_utterances":0,"utterances_longer_than_audio_history":0})
        if dataset=="dualtalk":
            report=Path(source["timed_root"])/"preparation_report.json"
            if report.exists():
                inherited=json.loads(report.read_text(encoding="utf-8"))
                quality["inherited_alignment_report"]={"path":str(report),"digest":fingerprint(inherited),
                    "text_policy":"inherit_verified_aligned_words_only; failed and unsupported text remains missing"}
        split_ids = _source_splits(source)
        assigned = {}
        for split, names in split_ids.items():
            shard_rank, shard_world = int(config.get("shard_rank",0)), int(config.get("shard_world",1))
            if not 0 <= shard_rank < shard_world:
                raise ValueError("Invalid token extraction shard")
            names = names[shard_rank::shard_world]
            manifest["extraction_shard"] = {"rank":shard_rank,"world":shard_world}
            if config.get("max_dialogues") is not None:
                names = names[:int(config["max_dialogues"])]
                manifest["diagnostic_subset"] = True
            for name in names:
                key = dataset+":"+name
                if key in assigned and assigned[key] != split:
                    raise ValueError(f"Dialogue {key} occurs in multiple splits")
                assigned[key] = split
                filename = fingerprint(key)+".pt"
                path = output/"dialogues"/filename
                try:
                    # Always rebuild source signature from raw inputs; stale
                    # cache is never accepted merely because dimensions match.
                    packets, audit = (_dualtalk_dialogue(source,name,split,extractor) if dataset == "dualtalk"
                                      else _upstream_dialogue(source,name,extractor))
                    # Raw label bytes may be unchanged while their validity
                    # rules change. Bind the interpretation into cache identity.
                    audit["supervision_revision"] = SUPERVISION_REVISION
                    cache_id = fingerprint({"protocol": TOKEN_PROTOCOL, "source": extractor.source,
                                            "audit": audit, "dialogue": key})
                    if path.exists() and not config.get("overwrite", False):
                        old = torch.load(path, map_location="cpu", weights_only=False)
                        if old.get("cache_id") != cache_id:
                            raise ValueError("Token source changed; choose a fresh output or set overwrite=true")
                    else:
                        temporary = path.with_suffix(".tmp")
                        torch.save({"protocol": TOKEN_PROTOCOL, "cache_id": cache_id, "feature_source": extractor.source,
                                    "packets": packets, "audit": audit}, temporary)
                        temporary.replace(path)
                    manifest["splits"][split].append(key)
                    manifest["dialogues"][key] = {"path": "dialogues/"+filename, "cache_id": cache_id,
                                                    "packets": len(packets), "dataset": dataset}
                    has_text=False
                    for packet in packets:
                        for role in packet["roles"]:
                            quality["role_packets"]+=1
                            for m in ("audio","prosody","au","flame","text"):
                                quality["tokens"][m]["total"]+=len(role[m+"_mask"])
                                quality["tokens"][m]["valid"]+=int(role[m+"_mask"].sum())
                            for i,m in enumerate(("A","V","T")):
                                quality["modality_available_packets"][m]+=int(role["modality_mask"][i])
                            has_text |= bool(role["text_mask"].any())
                    if not has_text:
                        quality["text_missing_dialogues"].append(key)
                    quality["text_missing_role_streams"]+=len(audit.get("text_missing_roles",[]))
                    quality["endpoint_utterances"]+=audit.get("endpoint_utterances",0)
                    quality["utterances_longer_than_audio_history"]+=audit.get("utterances_longer_than_audio_history",0)
                    if audit.get("same_role_overlap_samples",0) or audit.get("unverified_visual_masked",False):
                        manifest["data_gaps"].append({"dialogue":key, **{k:v for k,v in audit.items() if k in ("same_role_overlap_samples","unverified_visual_masked")}})
                    print(json.dumps({"dataset": dataset,"split":split,"completed":len(manifest["dialogues"]),"dialogue":name,"packets":len(packets)}),flush=True)
                except (ValueError, KeyError, FileNotFoundError) as error:
                    errors.append({"dialogue":key,"error":str(error)})
                    print(json.dumps(errors[-1]),flush=True)
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    manifest["errors"] = errors
    manifest["digest"] = fingerprint(manifest)
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    if errors or not manifest["dialogues"]:
        raise RuntimeError("Token preprocessing incomplete; inspect manifest errors")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
