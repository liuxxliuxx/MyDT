"""V3 source-balanced token A0 training with endpoint-only emotion supervision."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import math
import time
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from emotion_ssm.data.packets_v3 import PacketFrameDataset, collate_role_features, collate_endpoint_labels, prefix_role_features
from emotion_ssm.models.token_observer import TokenObserver, SUBSETS


TOKEN_SOURCE_KEYS=("audio_model","text_model","audio_dim","text_dim","audio_revision","text_revision",
                   "preprocessing","normalization","token_audio","token_text","token_ms","token_clock")


def select_labelled_adapter_source(payload,target_source):
    """Match an A0 labelled domain by full extractor provenance, never by size.

    Domain 2 has no emotion labels. Its unsupervised reconstruction adapter must
    not define the fixed emotion teacher for subsequent visual calibration.
    """
    from emotion_ssm.data.protocol import fingerprint
    missing=[key for key in TOKEN_SOURCE_KEYS if key not in target_source]
    if missing:
        raise ValueError(f"DualTalk token source is incomplete: {missing}")
    candidates={}
    for root,provenance in payload.get("provenance",{}).items():
        names=provenance.get("splits",{}).get("train",[])
        for dataset,domain in (("emotiontalk",0),("iemocap",1)):
            source=provenance.get("feature_sources",{}).get(dataset)
            if source is None or not any(str(name).startswith(dataset+":") for name in names):
                continue
            if any(key not in source for key in TOKEN_SOURCE_KEYS):
                raise ValueError(f"A0 {dataset} token provenance is incomplete")
            if dataset in candidates and candidates[dataset]["source"]!=source:
                raise ValueError(f"A0 contains conflicting {dataset} extractor sources")
            candidates[dataset]={"domain":domain,"source":source,"manifest_root":root,
                                 "manifest_sha256":provenance.get("sha256","")}
    matching=[(name,item) for name,item in candidates.items() if item["source"]==target_source]
    if not matching:
        raise ValueError("No labelled A0 domain has a fully compatible audio/text token extractor for DualTalk; "
                         "regenerate DualTalk tokens with a trained source, do not infer compatibility from dimensions")
    # If both labelled corpora deliberately share the exact same extractor,
    # prefer IEMOCAP consistently; record all matches rather than hiding a tie.
    dataset,chosen=max(matching,key=lambda item:item[1]["domain"])
    return {"source_domain":chosen["domain"],"target_domain":2,"source_dataset":dataset,
            "feature_source":copy.deepcopy(target_source),"feature_source_digest":fingerprint(target_source),
            "source_manifest_root":chosen["manifest_root"],"source_manifest_sha256":chosen["manifest_sha256"],
            "matching_labelled_domains":sorted(item["domain"] for _,item in matching),
            "eligibility_protocol":"labelled_domain_present_in_A0_training_provenance"}


def initialize_dualtalk_adapters(observer,payload,target_source):
    binding=select_labelled_adapter_source(payload,target_source)
    source,target=binding["source_domain"],binding["target_domain"]
    for mode in ("audio","text","prosody"):
        observer.adapters[mode][target].load_state_dict(copy.deepcopy(observer.adapters[mode][source].state_dict()),strict=True)
    return binding


def collate_observation_samples(samples):
    features = collate_role_features([item["features"] for item in samples])
    endpoint_features, labels = [], []
    for item in samples:
        for target in item["targets"]:
            endpoint_features.append(prefix_role_features(item["features"],float(target["end"])))
            labels.append(target)
    return {"features": features, "endpoint_features": collate_role_features(endpoint_features) if labels else None,
            "labels": collate_endpoint_labels(labels) if labels else None,
            "samples": len(samples), "endpoints": len(labels),
            "dialogues": len({str(item.get("dialogue_id", "")) for item in samples}),
            "roles": [int(item.get("role", -1)) for item in samples]}


def _available_subsets(features):
    available = (bool(features["audio_mask"].any()),
                 bool(features["au_mask"].any()) or bool(features["flame_mask"].any()),
                 bool(features["text_mask"].any()))
    return [name for name, selected in SUBSETS.items()
            if any(present and requested for present, requested in zip(available, selected))]


def _active_subset(features, subset):
    return subset in _available_subsets(features)


def _valid_supervision(label):
    return (0 <= int(label.get("emotion", -1)) < 7 or bool(label.get("intensity_mask", False))
            or any(label.get("vad_mask", [False] * 3)))


def build_training_index(datasets, progress=None):
    """Read TRAIN once; validation/test labels never influence priors or sampling."""
    rows, domains = [], {}
    offset = 0
    for dataset in datasets:
        lookup = {triple: offset + index for index, triple in enumerate(dataset.index)}
        for dialogue_index, name in enumerate(dataset.dialogues.ids):
            payload = dataset.dialogues[dialogue_index]
            for packet_index, packet in enumerate(payload["packets"]):
                for role in (0, 1):
                    features, labels = packet["roles"][role], packet["targets"][role]
                    domain = str(int(features["domain_id"]))
                    stats = domains.setdefault(domain, {"class_counts": [0] * 7, "endpoint_count": 0,
                        "unknown_class_count": 0, "raw_class_counts": {}, "vad_sum": [0.] * 3,
                        "vad_count": [0] * 3, "intensity_sum": 0., "intensity_count": 0,
                        "packet_count": 0, "dialogues": set(), "source": name.partition(":")[0]})
                    stats["packet_count"] += 1
                    stats["dialogues"].add(name)
                    supervised = set()
                    for label in labels:
                        if _valid_supervision(label):
                            endpoint = prefix_role_features(features, float(label["end"]))
                            supervised.update(_available_subsets(endpoint))
                        stats["endpoint_count"] += 1
                        category = int(label.get("emotion", -1))
                        raw = str(label.get("raw_emotion", label.get("original_emotion", category)))
                        stats["raw_class_counts"][raw] = stats["raw_class_counts"].get(raw, 0) + 1
                        if 0 <= category < 7:
                            stats["class_counts"][category] += 1
                        else:
                            stats["unknown_class_count"] += 1
                        for dimension, valid in enumerate(label.get("vad_mask", [False] * 3)):
                            if valid:
                                stats["vad_sum"][dimension] += float(label["vad"][dimension])
                                stats["vad_count"][dimension] += 1
                        if label.get("intensity_mask", False):
                            stats["intensity_sum"] += float(label["intensity"])
                            stats["intensity_count"] += 1
                    rows.append({"index": lookup[(dialogue_index, packet_index, role)], "domain": domain,
                                 "dialogue": name, "role": role, "supervised": sorted(supervised),
                                 "available": _available_subsets(features)})
            if progress is not None:
                progress(dialogue_index+1,len(dataset.dialogues.ids),len(rows))
        offset += len(dataset)
    _finalize_training_statistics(domains)
    return {"protocol": "v3.1-train-only-endpoint-index", "domains": domains, "rows": rows,
            "unknown_label_policy": "No guessed mapping; retain independently valid VAD/intensity supervision",
            "class_weight_policy": "inverse_sqrt_train_frequency_normalized_within_domain", "split": "train"}


def _finalize_training_statistics(domains):
    for stats in domains.values():
        counts = torch.tensor(stats["class_counts"], dtype=torch.float64)
        present = counts > 0
        weights = torch.zeros_like(counts)
        if present.any():
            weights[present] = counts[present].sum().sqrt() / counts[present].sqrt()
            weights[present] /= weights[present].mean()
        stats["class_weights"] = weights.tolist()
        stats["majority_class"] = int(counts.argmax()) if counts.sum() else None
        stats["vad_mean"] = [total / count if count else None for total, count in zip(stats["vad_sum"], stats["vad_count"])]
        stats["intensity_mean"] = stats["intensity_sum"] / stats["intensity_count"] if stats["intensity_count"] else None
        stats["dialogues"] = sorted(stats["dialogues"])


INDEX_CACHE_REVISION = "a0-train-endpoints-source-cache-v3.1.1"


def training_index_cache_binding(dataset):
    """Bind to the manifest and its actual TRAIN artifacts, never validation labels."""
    root = dataset.dialogues.root
    manifest_path = root/"manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if list(dataset.dialogues.ids) != list(manifest["splits"]["train"]):
        raise ValueError("A0 training-index caches may only index the declared train split")
    signatures = []
    for name in dataset.dialogues.ids:
        relative = manifest["dialogues"][name]["path"]
        stat = (root/relative).stat()
        signatures.append([name, relative, stat.st_size, stat.st_mtime_ns])
    binding = {"revision": INDEX_CACHE_REVISION, "split": "train", "rows": len(dataset),
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "protocol": manifest["protocol"], "train_artifacts": signatures}
    digest = hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return root/".a0-training-index-v31"/(digest+".pt"), binding


def _read_index_cache(path, binding):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("binding") != binding:
        raise ValueError("Stale A0 index cache: train manifest/artifacts or indexing semantics changed")
    index = payload.get("index", {})
    if (index.get("split") != "train" or index.get("protocol") != "v3.1-train-only-endpoint-index"
            or len(index.get("rows", [])) != binding["rows"]):
        raise ValueError("Malformed A0 train-only index cache")
    return index


def cached_source_training_index(dataset, *, build=True, wait_seconds=14400, poll_seconds=.5, emit=None):
    """CPU/file rendezvous; never launches a distributed or CUDA collective.

    Rank zero builds a source once and atomically publishes it. Other ranks wait
    for that exact source signature on the shared token filesystem. Later seeds
    and FLAME calibration reuse the same source-local cache without rescanning.
    """
    path, binding = training_index_cache_binding(dataset)
    log = emit if emit is not None else lambda message: print(json.dumps(message),flush=True)
    started, last_report = time.monotonic(), -float("inf")
    if path.exists():
        index = _read_index_cache(path,binding)
        log({"stage":"training_index","status":"cache_hit","source":str(dataset.dialogues.root),"rows":len(index["rows"])})
        return index
    if build:
        log({"stage":"training_index","status":"building_source","source":str(dataset.dialogues.root),
             "train_dialogues":len(dataset.dialogues.ids),"train_rows":len(dataset)})
        def progress(completed,total,rows):
            nonlocal last_report
            now = time.monotonic()
            if completed==1 or completed==total or now-last_report>=30:
                last_report=now
                log({"stage":"training_index","status":"indexing_cpu","source":str(dataset.dialogues.root),
                     "dialogues":completed,"total_dialogues":total,"rows":rows,"seconds":now-started})
        index = build_training_index([dataset],progress=progress)
        # Refuse to publish an index spanning concurrent preprocessing edits.
        _, after = training_index_cache_binding(dataset)
        if after != binding:
            raise ValueError("Training artifacts changed during A0 indexing; rebuild from a stable manifest")
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(f".tmp.{os.getpid()}")
        torch.save({"binding":binding,"index":index},temporary)
        temporary.replace(path)
        log({"stage":"training_index","status":"cache_ready","source":str(dataset.dialogues.root),
             "rows":len(index["rows"]),"seconds":time.monotonic()-started})
        return index
    while not path.exists():
        now=time.monotonic()
        if now-started>wait_seconds:
            raise TimeoutError(f"Timed out waiting for rank-zero CPU training index: {path}")
        if now-last_report>=30:
            last_report=now
            log({"stage":"training_index","status":"waiting_for_rank_zero_cpu_cache",
                 "source":str(dataset.dialogues.root),"seconds":now-started,"distributed_collective":False})
        time.sleep(poll_seconds)
    return _read_index_cache(path,binding)


def cached_training_index(datasets, *, build=True, wait_seconds=14400, emit=None):
    combined, offset = None, 0
    for dataset in datasets:
        source = cached_source_training_index(dataset,build=build,wait_seconds=wait_seconds,emit=emit)
        if combined is None:
            combined={key:value for key,value in source.items() if key not in ("rows","domains")}
            combined.update(rows=[],domains={})
        for row in source["rows"]:
            row["index"]+=offset
        combined["rows"].extend(source["rows"])
        offset+=len(dataset)
        for domain, stats in source["domains"].items():
            if domain not in combined["domains"]:
                combined["domains"][domain]=copy.deepcopy(stats)
                continue
            target=combined["domains"][domain]
            if set(target["dialogues"]) & set(stats["dialogues"]):
                raise ValueError("A0 training roots repeat the same source dialogue")
            for field in ("endpoint_count","unknown_class_count","intensity_sum","intensity_count","packet_count"):
                target[field]+=stats[field]
            for field in ("class_counts","vad_sum","vad_count"):
                target[field]=[a+b for a,b in zip(target[field],stats[field])]
            target["dialogues"].extend(stats["dialogues"])
            for label,count in stats["raw_class_counts"].items():
                target["raw_class_counts"][label]=target["raw_class_counts"].get(label,0)+count
    if combined is None:
        raise ValueError("A0 requires training token sources")
    _finalize_training_statistics(combined["domains"])
    return combined


class MixedDialogueDataset(torch.utils.data.Dataset):
    """Bounded dialogue LRU so a mixed batch does not reread every row from disk."""
    def __init__(self, datasets, cache_dialogues=12):
        self.datasets, self.cache_size = datasets, int(cache_dialogues)
        self.lengths = [len(dataset) for dataset in datasets]
        self.cache = OrderedDict()

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):
        source = 0
        while index >= self.lengths[source]:
            index -= self.lengths[source]
            source += 1
        dataset = self.datasets[source]
        dialogue, packet, role = dataset.index[index]
        key = (source, dialogue)
        if key not in self.cache:
            self.cache[key] = dataset.dialogues[dialogue]
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        item = self.cache[key]["packets"][packet]
        return {"features": item["roles"][role], "targets": item["targets"][role],
                "dialogue_id": dataset.dialogues.ids[dialogue], "packet_index": packet, "role": role}


def _global_ratio(numerator, denominator):
    """DDP averages gradients: world scaling recovers the global valid mean."""
    count = torch.as_tensor(denominator, device=numerator.device, dtype=torch.float64).detach().clone()
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if world > 1:
        dist.all_reduce(count)
    return numerator * world / count.clamp_min(1e-12).to(numerator.dtype), count


def globally_weighted_supervision(model, features, labels, subset=None, training_statistics=None, reference=None):
    zero = reference if reference is not None else next(model.parameters()).sum() * 0.
    numerator = {key: zero for key in ("emotion", "intensity", "vad")}
    denominator = {key: zero.detach() for key in numerator}
    if labels is not None:
        encoded = model.encode(features, subset)
        predicted = model.decode_affect(encoded["observation"].aff)
        valid = encoded["valid"] & labels.get("endpoint_mask", encoded["valid"])
        emotion = labels["emotion"].long()
        supported = valid & (emotion >= 0) & (emotion < 7)
        row_weight = predicted["emotion_logits"].new_ones(len(valid))
        priors = (training_statistics or {}).get("domains", {})
        for domain, statistics in priors.items():
            selected = features["domain_id"].eq(int(domain)) & supported
            if selected.any():
                weights = row_weight.new_tensor(statistics["class_weights"])
                row_weight[selected] = weights[emotion[selected]]
        if supported.any():
            ce = F.cross_entropy(predicted["emotion_logits"][supported].float(), emotion[supported], reduction="none")
            numerator["emotion"] = (ce * row_weight[supported]).sum()
            denominator["emotion"] = row_weight[supported].sum()
        intensity_valid = valid & labels["intensity_mask"].bool()
        if intensity_valid.any():
            numerator["intensity"] = (predicted["intensity"][intensity_valid].float() - labels["intensity"][intensity_valid]).square().sum()
            denominator["intensity"] = intensity_valid.sum()
        vad_valid = valid[:, None] & labels["vad_mask"].bool()
        if vad_valid.any():
            numerator["vad"] = (predicted["vad"].float()[vad_valid] - labels["vad"][vad_valid]).square().sum()
            denominator["vad"] = vad_valid.sum()
    losses, counts = {}, {}
    # Every rank executes every reduction, including ranks with zero labels.
    for key in numerator:
        losses[key], counts[key] = _global_ratio(numerator[key], denominator[key])
    return {**losses, **{key + "_weight_count": value for key, value in counts.items()},
            "total": sum(losses.values())}


def _to(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k:_to(v,device) for k,v in value.items()}
    return value


def observation_objective(model, batch, subset=None, teacher=None, weights=None):
    weights = weights or {}
    features = batch["features"]
    distributed = dist.is_available() and dist.is_initialized()
    masking = weights.get("masking")
    reconstruction = model.masked_loss(features, subset=None if masking else subset, teacher=teacher,
                                        distributed=distributed, masking=masking)
    losses = {"masked_"+key:value for key,value in reconstruction.items() if key != "total"}
    total = float(weights.get("masked_weight",1.))*reconstruction["total"]
    labels = globally_weighted_supervision(model, batch.get("endpoint_features"), batch.get("labels"), subset,
                                          weights.get("training_statistics"), reference=total * 0.)
    total = total + float(weights.get("label_weight", 1.)) * labels["total"]
    losses.update({"label_" + key: value for key, value in labels.items() if key != "total"})
    losses["total"] = total
    return losses


def calibration_objective(model,batch,teacher,weights=None):
    """Fixed A/T targets, with global valid-row/element means on every rank."""
    weights=weights or {}
    features=batch["features"]
    with torch.no_grad():
        target=teacher.encode_clean(features,"AT")
    current=model.encode(features,"V")
    valid=current["valid"] & target["valid"] & features["flame_mask"].any(1)
    zero=current["raw_aff"].sum()*0
    coordinate_sum=(1-F.cosine_similarity(current["observation"].aff[valid].float(),
                    target["observation"].aff[valid].float())).sum() if valid.any() else zero
    action_valid=valid & target["observation"].action_present
    action_sum=F.smooth_l1_loss(current["observation"].action[action_valid].float(),
                    target["observation"].action[action_valid].float(),reduction="sum") if action_valid.any() else zero
    # Calibration only learns the FLAME route. Avoid computing discarded
    # summary/distillation/diversity graphs from the general masked objective.
    corruption=model.sample_corruption(features)
    masked=model.encode(features,"V",corruption)
    chosen=masked["masked"]["flame"]
    flame_target=features["flame_tokens"].detach().float()
    flame_target=F.layer_norm(flame_target,(flame_target.shape[-1],))
    reconstruction_sum=F.smooth_l1_loss(masked["reconstruction"]["flame"][chosen].float(),
                    flame_target[chosen],reduction="sum") if chosen.any() else zero
    # Each synchronous calibration batch executes the same three reductions,
    # including ranks with no teacher-compatible rows or masked FLAME tokens.
    coordinate,coordinate_count=_global_ratio(coordinate_sum,valid.sum())
    action,action_count=_global_ratio(action_sum,action_valid.sum()*current["observation"].action.shape[-1])
    reconstruction,reconstruction_count=_global_ratio(reconstruction_sum,chosen.sum()*flame_target.shape[-1])
    return {"coordinate":coordinate,"action_teacher":action,"masked_reconstruction":reconstruction,
            "coordinate_weight_count":coordinate_count,"action_weight_count":action_count,
            "reconstruction_weight_count":reconstruction_count,
            "total":coordinate+.1*action+float(weights.get("masked_weight",.2))*reconstruction}


class _TrainingObjective(nn.Module):
    def __init__(self, observer, teacher, weights):
        super().__init__()
        self.observer, self.teacher, self.weights = observer, teacher, weights

    def forward(self,batch,subset):
        if self.weights.get("stage")=="calibration":
            return calibration_objective(self.observer,batch,self.teacher,self.weights)
        return observation_objective(self.observer,batch,subset,self.teacher,self.weights)


class DialogueBalancedBatches(Sampler):
    """Multi-dialogue/role batches with real endpoint supervision on every rank.

    At least half of each local batch is drawn from actual supervised endpoints
    compatible with this step's modality subset, when any exist. The other rows
    supply source-balanced unsupervised packets, including unlabelled DualTalk.
    No utterance label is ever copied to an unlabelled second.
    """
    def __init__(self,datasets,batch_size,rank=0,world_size=1,seed=6666,epoch=0,
                 training_index=None,supervised_fraction=.5,subset_offset=0,modality_dropout=True,
                 active_dialogues=12,refresh_batches=32,masking=None):
        self.datasets,self.batch_size=datasets,int(batch_size)
        self.rank,self.world_size,self.seed,self.epoch=rank,world_size,seed,epoch
        self.index=training_index or build_training_index(datasets)
        self.rows=self.index["rows"]
        if not self.rows:
            raise ValueError("A0 has no training token packets")
        self.count=max(1,math.ceil(len(self.rows)/(self.batch_size*self.world_size)))
        self.label_count=max(1,min(self.batch_size,math.ceil(self.batch_size*supervised_fraction)))
        self.subset_offset,self.modality_dropout=int(subset_offset),bool(modality_dropout)
        self.masking = masking
        self.active_dialogues=max(2,int(active_dialogues))
        self.refresh_batches=max(1,int(refresh_batches))
        self.pools={}
        for subset in SUBSETS:
            self.pools[subset]={kind:{} for kind in ("supervised","available")}
            for row in self.rows:
                for kind in ("supervised","available"):
                    if subset in row[kind]:
                        self.pools[subset][kind].setdefault(row["domain"],{}).setdefault(row["dialogue"],[]).append(row)

    def __len__(self):
        return self.count

    def subset_for_batch(self,index):
        if self.masking and self.modality_dropout:
            from emotion_ssm.models.context_masking import sample_training_subset
            # Independent deterministic RNG: DataLoader lookahead and resume
            # must not consume the model's stochastic masking stream.
            generator = torch.Generator().manual_seed(self.seed+104729*(self.subset_offset+index+1))
            requested = sample_training_subset(self.masking, generator)
        else:
            requested=list(SUBSETS)[(self.subset_offset+index)%len(SUBSETS)] if self.modality_dropout else "AVT"
        # An entirely absent modality cannot be trained by relabelling it present.
        return requested if self.pools[requested]["available"] else "AVT"

    def __iter__(self):
        rng=random.Random(self.seed+self.epoch)
        width=self.batch_size*self.world_size
        names_by_domain={}
        for row in self.rows:
            names_by_domain.setdefault(row["domain"],set()).add(row["dialogue"])
        names_by_domain={domain:sorted(names) for domain,names in names_by_domain.items()}
        for names in names_by_domain.values():
            rng.shuffle(names)
        active={}
        # Tiny fixtures cannot supply cross-dialogue diversity; no invented rows.
        if len(self.rows)<=width and len({row["dialogue"] for row in self.rows})==1:
            ordered=[row["index"] for row in self.rows]
            ordered += [ordered[i%len(ordered)] for i in range(width-len(ordered))]
            yield ordered[self.rank*self.batch_size:(self.rank+1)*self.batch_size]
            return
        for batch_index in range(self.count):
            # Reuse a small multi-dialogue working set across consecutive batches.
            # Its size fits the loader LRU, avoiding one disk load per sample.
            if batch_index%self.refresh_batches==0:
                per_domain=max(2,self.active_dialogues//len(names_by_domain))
                window=batch_index//self.refresh_batches
                active={domain:{names[(window*per_domain+i)%len(names)] for i in range(min(per_domain,len(names)))}
                        for domain,names in names_by_domain.items()}
            subset=self.subset_for_batch(batch_index)
            used_rows,used_dialogues=set(),set()
            selected=[]
            for slot in range(width):
                labelled=slot<self.label_count*self.world_size
                pool=self.pools[subset]["supervised" if labelled else "available"]
                if not pool:
                    pool=self.pools[subset]["available"]
                if not pool:
                    raise ValueError(f"No real available {subset} observations; cannot construct this modality batch")
                domains=sorted(pool)
                domain=domains[(batch_index+slot)%len(domains)]
                dialogues={name:rows for name,rows in pool[domain].items() if name in active[domain]} or pool[domain]
                candidates=[name for name,rows in dialogues.items() if any(row["index"] not in used_rows for row in rows)]
                unseen=[name for name in candidates if name not in used_dialogues]
                names=unseen or candidates or list(dialogues)
                name=rng.choice(names)
                available=[row for row in dialogues[name] if row["index"] not in used_rows] or dialogues[name]
                preferred=[row for row in available if row["role"]==(slot//self.world_size)%2]
                row=rng.choice(preferred or available)
                selected.append(row["index"])
                used_rows.add(row["index"]); used_dialogues.add(row["dialogue"])
            yield selected[self.rank::self.world_size]


@torch.no_grad()
def _evaluate_calibration(observer, roots, device, max_dialogues=16, batch_size=16, teacher=None, calibration=False):
    """Single-rank deterministic evaluation, no distributed sampler duplicates.

    Emotion confusion and squared-error numerators are aggregated globally over
    valid labels. Masked reconstruction is intentionally not the semantic score.
    """
    from emotion_ssm.data.packets_v3 import TokenPacketDataset
    observer.eval()
    totals={}
    pending=[]

    def consume(samples):
        if not samples:
            return
        features=collate_role_features([item[0] for item in samples],device)
        labels=collate_endpoint_labels([item[1] for item in samples],device)
        domains=[int(item[0]["domain_id"]) for item in samples]
        for subset in SUBSETS:
            encoded=observer.encode(features,subset)
            predictions=observer.decode_affect(encoded["observation"].aff)
            target=teacher.encode(features,"AT") if teacher is not None and subset in ("V","AVT") else None
            masked_reconstruction=None
            if subset=="V" and features["flame_mask"].any():
                corruption={m:torch.zeros_like(features[m+"_mask"]) for m in ("audio","au","flame","text")}
                for row,mask in enumerate(features["flame_mask"]):
                    indices=mask.nonzero().flatten()
                    corruption["flame"][row,indices[:max(1,round(len(indices)*.4))]]=True
                masked_output=observer.encode(features,"V",corruption)
                target_flame=F.layer_norm(features["flame_tokens"].float(),(features["flame_tokens"].shape[-1],))
                masked_reconstruction=F.smooth_l1_loss(masked_output["reconstruction"]["flame"],target_flame,reduction="none")
            for row,domain in enumerate(domains):
                key=f"domain{domain}/{subset}"
                record=totals.setdefault(key,{"confusion":torch.zeros(7,7,dtype=torch.int64),"vad_sse":0.,"vad_count":0,
                                             "intensity_sse":0.,"intensity_count":0,"affect":[],
                                             "teacher_error":0.,"teacher_count":0,"teacher_targets":[],"teacher_predictions":[],
                                             "flame_mask_loss_sum":0.,"flame_mask_elements":0})
                if not encoded["valid"][row]:
                    continue
                record["affect"].append(encoded["observation"].aff[row].cpu())
                if masked_reconstruction is not None:
                    mask=corruption["flame"][row]
                    record["flame_mask_loss_sum"]+=float(masked_reconstruction[row][mask].sum())
                    record["flame_mask_elements"]+=int(mask.sum())*features["flame_tokens"].shape[-1]
                if target is not None and target["valid"][row]:
                    record["teacher_error"]+=float(1-F.cosine_similarity(encoded["observation"].aff[row:row+1],target["observation"].aff[row:row+1]))
                    record["teacher_count"]+=1
                    record["teacher_targets"].append(target["observation"].aff[row].cpu())
                    record["teacher_predictions"].append(encoded["observation"].aff[row].cpu())
                label=int(labels["emotion"][row])
                if 0<=label<7:
                    record["confusion"][label,int(predictions["emotion_logits"][row].argmax())]+=1
                mask=labels["vad_mask"][row].bool()
                record["vad_sse"]+=float((predictions["vad"][row][mask]-labels["vad"][row][mask]).square().sum())
                record["vad_count"]+=int(mask.sum())
                if labels["intensity_mask"][row]:
                    record["intensity_sse"]+=float((predictions["intensity"][row]-labels["intensity"][row]).square())
                    record["intensity_count"]+=1

    seen={}
    for root in roots:
        dataset=TokenPacketDataset(root,"val")
        for index,name in enumerate(dataset.ids):
            domain=name.partition(":")[0]
            if max_dialogues and seen.get(domain,0)>=max_dialogues:
                continue
            seen[domain]=seen.get(domain,0)+1
            for packet in dataset[index]["packets"]:
                for role in (0,1):
                    labels_for_role=packet["targets"][role]
                    if not labels_for_role:
                        labels_for_role=[{"emotion":-1,"intensity":0.,"intensity_mask":False,"vad":[0.,0.,0.],"vad_mask":[False,False,False]}]
                    for label in labels_for_role:
                        sample=prefix_role_features(packet["roles"][role],float(label["end"])) if "end" in label else packet["roles"][role]
                        pending.append((sample,label))
                        if len(pending)>=batch_size:
                            consume(pending)
                            pending=[]
    consume(pending)
    metrics={}
    scores=[]
    for key,record in totals.items():
        confusion=record["confusion"].float()
        support=confusion.sum(1)
        represented=support>0
        precision=confusion.diag()/confusion.sum(0).clamp_min(1)
        recall=confusion.diag()/support.clamp_min(1)
        f1=2*precision*recall/(precision+recall).clamp_min(1e-8)
        aff_values=record.pop("affect")
        if not aff_values:
            metrics[key]={"samples":0,"emotion_samples":0,"missing":True,"embedding_std":None,"embedding_cosine":None}
            continue
        aff=torch.stack(aff_values)
        metrics[key]={"samples":len(aff),"emotion_samples":int(confusion.sum()),
                      "macro_f1":float(f1[represented].mean()) if represented.any() else None,
                      "uar":float(recall[represented].mean()) if represented.any() else None,
                      "vad_mse":record["vad_sse"]/record["vad_count"] if record["vad_count"] else None,
                      "intensity_mse":record["intensity_sse"]/record["intensity_count"] if record["intensity_count"] else None,
                      "embedding_std":float(aff.std(0,unbiased=False).mean()),
                      "embedding_cosine":float((aff.sum(0).square().sum()-len(aff))/(len(aff)*(len(aff)-1))) if len(aff)>1 else None,
                      "confusion":record["confusion"].tolist()}
        if record["teacher_count"]:
            teacher_aff=torch.stack(record["teacher_targets"])
            mean=F.normalize(teacher_aff.mean(0),dim=0)
            metrics[key]["teacher_cosine_error"]=record["teacher_error"]/record["teacher_count"]
            metrics[key]["teacher_samples"]=record["teacher_count"]
            metrics[key]["constant_teacher_mean_error"]=float((1-teacher_aff@mean).mean())
        if record["flame_mask_elements"]:
            metrics[key]["flame_masked_reconstruction"]=record["flame_mask_loss_sum"]/record["flame_mask_elements"]
        # Report and select using all deployment-important subsets, not AVT alone.
        if key.rsplit("/",1)[1] in ("A","AT","AVT") and represented.any():
            scores.append(1-metrics[key]["macro_f1"])
    if calibration:
        visual=metrics.get("domain2/V",{})
        metrics["selection_loss"]=visual.get("teacher_cosine_error",float("inf"))+.2*visual.get("flame_masked_reconstruction",0.)
        metrics["selection_protocol"]="fixed_AT_teacher_to_FLAME_V_cosine_error_plus_0.2_masked_reconstruction"
        metrics["visual_exceeds_constant_mean"]=(visual.get("teacher_cosine_error",float("inf"))<visual.get("constant_teacher_mean_error",-1))
    else:
        metrics["selection_loss"]=sum(scores)/len(scores) if scores else float("inf")
        metrics["selection_protocol"]="mean_domain_A_AT_AVT_one_minus_endpoint_macro_f1"
    return metrics


def confusion_metrics(confusion):
    matrix = torch.as_tensor(confusion, dtype=torch.float64)
    support, predicted = matrix.sum(1), matrix.sum(0)
    active = support > 0
    recall = matrix.diag() / support.clamp_min(1)
    precision = matrix.diag() / predicted.clamp_min(1)
    f1 = 2 * recall * precision / (recall + precision).clamp_min(1e-12)
    return {"macro_f1": float(f1[active].mean()) if active.any() else None,
            "uar": float(recall[active].mean()) if active.any() else None,
            "accuracy": float(matrix.diag().sum() / matrix.sum()) if matrix.sum() else None,
            "support": support.long().tolist(), "predicted_counts": predicted.long().tolist(),
            "supported_classes": int(active.sum()), "predicted_classes": int((predicted > 0).sum()),
            "supported_predicted_classes": int(((predicted > 0) & active).sum())}


@torch.no_grad()
def clone_dropout_diagnostics(observer, features, subset="AVT", row=0, clones=8):
    """Repeated identical evidence separates dropout variance from data variance."""
    repeated = {key: value[row:row+1].expand(clones, *value.shape[1:]).clone()
                for key, value in features.items()}
    clean = observer.encode_clean(repeated, subset)["observation"].aff.float()
    modes = [(module, module.training) for module in observer.modules()]
    device = clean.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            observer.train()
            stochastic = observer.encode(repeated, subset)["observation"].aff.float()
    finally:
        for module, training in modes:
            module.training = training
    return {"identical_input_clean_std": float(clean.std(0, unbiased=False).mean()),
            "identical_input_dropout_std": float(stochastic.std(0, unbiased=False).mean()),
            "clones": clones, "protocol": "same_tokens_roles_times_repeated; dropout_enabled_only_in_second_view"}


@torch.no_grad()
def evaluate(observer, roots, device, max_dialogues=16, batch_size=16, teacher=None,
             calibration=False, training_statistics=None, model_name="student"):
    if calibration:
        return _evaluate_calibration(observer, roots, device, max_dialogues, batch_size, teacher, True)
    from emotion_ssm.data.packets_v3 import TokenPacketDataset
    if training_statistics is None:
        index = build_training_index([PacketFrameDataset(root, "train") for root in roots])
        training_statistics = {key: value for key, value in index.items() if key != "rows"}
    observer.eval()
    totals, pending, seen = {}, [], {}

    def consume(samples):
        if not samples:
            return
        features = collate_role_features([item[0] for item in samples], device)
        labels = collate_endpoint_labels([item[1] for item in samples], device)
        domains = features["domain_id"].long()
        real_masks = torch.stack((features["audio_mask"].any(1), features["au_mask"].any(1) | features["flame_mask"].any(1), features["text_mask"].any(1)), 1)
        for subset, selected in SUBSETS.items():
            encoded = observer.encode_clean(features, subset)
            predictions = observer.decode_affect(encoded["observation"].aff)
            for domain in domains.unique().tolist():
                key = f"domain{domain}/{subset}"
                stats = training_statistics.get("domains", {}).get(str(domain), {})
                record = totals.setdefault(key, {"confusion": torch.zeros(7, 7, dtype=torch.long),
                    "samples": 0, "seen_samples": 0, "modality_counts": torch.zeros(3, dtype=torch.long),
                    "full_subset_samples": 0, "vad_sse": 0., "vad_baseline_sse": 0., "vad_count": 0,
                    "vad_dimension_sse": [0.] * 3, "vad_dimension_baseline_sse": [0.] * 3, "vad_dimension_count": [0] * 3,
                    "intensity_sse": 0., "intensity_baseline_sse": 0., "intensity_count": 0,
                    "aff_sum": None, "aff_square_sum": None, "raw_sum": None, "raw_square_sum": None})
                rows = domains.eq(domain)
                record["seen_samples"] += int(rows.sum())
                record["modality_counts"] += real_masks[rows].long().sum(0).cpu()
                requirement = torch.tensor(selected, device=device, dtype=torch.bool)
                record["full_subset_samples"] += int((real_masks[rows] | ~requirement).all(1).sum())
                valid = rows & encoded["valid"]
                if not valid.any():
                    continue
                if "clone" not in record and subset in ("A", "AT", "AVT"):
                    record["clone"] = clone_dropout_diagnostics(observer, features, subset, int(valid.nonzero()[0]))
                affect, raw = encoded["observation"].aff[valid].double().cpu(), encoded["raw_aff"][valid].double().cpu()
                for prefix, values in (("aff", affect), ("raw", raw)):
                    total, square = values.sum(0), values.square().sum(0)
                    record[prefix + "_sum"] = total if record[prefix + "_sum"] is None else record[prefix + "_sum"] + total
                    record[prefix + "_square_sum"] = square if record[prefix + "_square_sum"] is None else record[prefix + "_square_sum"] + square
                record["samples"] += len(affect)
                emotion = labels["emotion"].long()
                classified = valid & (emotion >= 0) & (emotion < 7)
                if classified.any():
                    predicted = predictions["emotion_logits"][classified].argmax(-1)
                    counts = torch.bincount(emotion[classified] * 7 + predicted, minlength=49).reshape(7, 7).cpu()
                    record["confusion"] += counts
                for dimension in range(3):
                    allowed = valid & labels["vad_mask"][:, dimension].bool()
                    mean = stats.get("vad_mean", [None] * 3)[dimension]
                    if allowed.any() and mean is not None:
                        truth = labels["vad"][allowed, dimension].double()
                        error = float((predictions["vad"][allowed, dimension].double() - truth).square().sum())
                        constant = float((truth - mean).square().sum())
                        count = int(allowed.sum())
                        record["vad_sse"] += error; record["vad_baseline_sse"] += constant; record["vad_count"] += count
                        record["vad_dimension_sse"][dimension] += error
                        record["vad_dimension_baseline_sse"][dimension] += constant
                        record["vad_dimension_count"][dimension] += count
                allowed = valid & labels["intensity_mask"].bool()
                mean = stats.get("intensity_mean")
                if allowed.any() and mean is not None:
                    truth = labels["intensity"][allowed].double()
                    record["intensity_sse"] += float((predictions["intensity"][allowed].double() - truth).square().sum())
                    record["intensity_baseline_sse"] += float((truth - mean).square().sum())
                    record["intensity_count"] += int(allowed.sum())

    for root in roots:
        dataset = TokenPacketDataset(root, "val")
        for index, name in enumerate(dataset.ids):
            domain = name.partition(":")[0]
            if max_dialogues and seen.get(domain, 0) >= max_dialogues:
                continue
            seen[domain] = seen.get(domain, 0) + 1
            for packet in dataset[index]["packets"]:
                for role in (0, 1):
                    labels = packet["targets"][role] or [{"emotion": -1, "intensity": 0., "intensity_mask": False,
                                                          "vad": [0.] * 3, "vad_mask": [False] * 3}]
                    for label in labels:
                        features = prefix_role_features(packet["roles"][role], float(label["end"])) if "end" in label else packet["roles"][role]
                        pending.append((features, label))
                        if len(pending) >= batch_size:
                            consume(pending); pending = []
    consume(pending)
    metrics, scores = {}, []
    for key, record in totals.items():
        domain, subset = key.split("/")
        stats = training_statistics.get("domains", {}).get(domain.replace("domain", ""), {})
        count = record["samples"]
        result = {**confusion_metrics(record["confusion"]), "samples": count,
                  "seen_samples": record["seen_samples"], "emotion_samples": int(record["confusion"].sum()),
                  "missing": count == 0, "confusion": record["confusion"].tolist(),
                  "modality_counts": dict(zip(("A", "V", "T"), record["modality_counts"].tolist())),
                  "full_requested_subset_samples": record["full_subset_samples"],
                  "requested_subset": subset, "model": model_name, "evaluation_mode": "clean_dropout_disabled"}
        majority = stats.get("majority_class")
        constant = torch.zeros(7, 7, dtype=torch.long)
        if majority is not None:
            constant[:, majority] = record["confusion"].sum(1)
        result["constant_class"] = {"predicted_class": majority, "source": "training_class_counts_only", **confusion_metrics(constant)}
        for task in ("vad", "intensity"):
            elements = record[task + "_count"]
            result[task + "_mse"] = record[task + "_sse"] / elements if elements else None
            result[task + "_constant_train_mean_mse"] = record[task + "_baseline_sse"] / elements if elements else None
            result[task + "_elements"] = elements
        result["vad_dimensions"] = [{"elements": n, "mse": s/n if n else None, "constant_train_mean_mse": b/n if n else None}
                                     for s, b, n in zip(record["vad_dimension_sse"], record["vad_dimension_baseline_sse"], record["vad_dimension_count"])]
        if count:
            variance = (record["aff_square_sum"] / count - (record["aff_sum"] / count).square()).clamp_min(0)
            raw_variance = (record["raw_square_sum"] / count - (record["raw_sum"] / count).square()).clamp_min(0)
            result["embedding_std"] = float(variance.sqrt().mean())
            result["embedding_variance_trace"] = float(variance.sum())
            result["raw_embedding_std"] = float(raw_variance.sqrt().mean())
            result["embedding_cosine"] = float((record["aff_sum"].square().sum()-record["aff_square_sum"].sum()) / (count*(count-1))) if count > 1 else None
        else:
            result.update(embedding_std=None, embedding_variance_trace=None, raw_embedding_std=None, embedding_cosine=None)
        if "clone" in record:
            result["clone_dropout"] = record["clone"]
        metrics[key] = result
        if subset in ("A", "AT", "AVT") and result["macro_f1"] is not None:
            scores.append(1-result["macro_f1"])
    metrics.update(selection_loss=sum(scores)/len(scores) if scores else float("inf"),
                   selection_protocol="independent_endpoint_macro_f1_A_AT_AVT; train_only_constant_baselines",
                   evaluation_model=model_name, evaluation_split="val", validation_dialogues=seen,
                   training_statistics=training_statistics,
                   semantic_evidence="true_labels_only; EMA agreement is not semantic validation")
    return metrics


def semantic_gate(metrics, *, allow_untrained=False):
    """Reject trivial classification and clean collapse without a fixed F1 target."""
    failures, checks, limitations = [], {}, []
    domains = metrics.get("training_statistics", {}).get("domains", {})
    for domain, prior in domains.items():
        if sum(prior.get("class_counts", [])) == 0 and not sum(prior.get("vad_count", [])):
            limitations.append(f"domain{domain}: no independent emotion/VAD labels; no semantic claim")
            continue
        for subset in ("A", "AT", "AVT"):
            key = f"domain{domain}/{subset}"
            result = metrics.get(key, {})
            reasons = []
            if result.get("missing", True):
                reasons.append("No valid observations in the validation subset")
            elif not result.get("emotion_samples", 0) and not result.get("vad_elements", 0):
                reasons.append("No independent valid validation supervision")
            if result.get("emotion_samples", 0):
                if result.get("supported_classes", 0) < 2:
                    reasons.append("Validation contains fewer than two emotion classes; semantic discrimination unverified")
                elif result.get("supported_predicted_classes", 0) < 2:
                    reasons.append("Predictions collapsed to one supported emotion class")
                baseline = result.get("constant_class", {})
                for name in ("macro_f1", "uar"):
                    value, constant = result.get(name), baseline.get(name)
                    if value is None or constant is None or not math.isfinite(value) or value <= constant + 1e-8:
                        reasons.append(f"{name} does not exceed the train-majority constant baseline")
            trace = result.get("embedding_variance_trace")
            if result.get("samples", 0) >= 2 and (trace is None or not math.isfinite(trace) or trace <= 1e-12):
                reasons.append("Deterministic normalized affect has numerical-zero between-example variance")
            if result.get("vad_elements", 0):
                error, baseline = result.get("vad_mse"), result.get("vad_constant_train_mean_mse")
                if baseline is not None and baseline > 1e-12 and (not math.isfinite(error) or error >= baseline):
                    reasons.append("VAD does not improve on the train-mean predictor")
            counts = result.get("modality_counts", {})
            unavailable = [name for name, count in counts.items() if not count]
            if unavailable:
                limitations.append(f"{key}: modalities {','.join(unavailable)} missing; requested fusion name is not proof of their contribution")
            checks[key] = {"passed": not reasons, "reasons": reasons}
            failures.extend(f"{key}: {reason}" for reason in reasons)
    if not checks:
        failures.append("No labelled domain can establish independent A0 quality")
    passed = not failures
    return {"passed": passed, "allowed": passed or bool(allow_untrained), "bypassed": bool(allow_untrained and not passed),
            "checks": checks, "failures": failures, "limitations": sorted(set(limitations)),
            "protocol": "independent_endpoint_vs_train_constants_and_clean_variance_v3.1",
            "threshold_policy": "No absolute F1 threshold; reject constant predictors and numerical collapse"}


def run(config):
    from emotion_ssm.config_v3 import read_config,validate_config,write_config
    from emotion_ssm.utils.checkpoint_v3 import save_checkpoint,read_checkpoint,restore_training,load_observer,require_training_revision
    from emotion_ssm.utils.checkpoint import capture_rng_state,restore_rng_state
    config=read_config(config) if isinstance(config,(str,Path)) else validate_config(copy.deepcopy(config))
    resumed=read_checkpoint(config["paths"]["resume"]) if config["paths"].get("resume") else None
    if resumed is not None:
        require_training_revision(resumed)
        if resumed["config"]["train"].get("masking") != config["train"].get("masking"):
            raise ValueError("Masking policy changed; start a new observation experiment")
    train=config["train"]
    calibration=train.get("stage")=="calibration"
    max_steps=int(train.get("calibration_steps" if calibration else "observation_steps",train["max_steps"]))
    train["max_steps"]=max_steps
    rank=int(os.environ.get("RANK",0)); world=int(os.environ.get("WORLD_SIZE",1)); local=int(os.environ.get("LOCAL_RANK",0))
    distributed=world>1
    device=torch.device(f"cuda:{local}" if torch.cuda.is_available() and str(train["device"]).startswith("cuda") else "cpu")
    if device.type=="cuda":
        torch.cuda.set_device(device)
    owned_group=distributed and not dist.is_initialized()
    seed=int(train["seed"])
    random.seed(seed+rank); torch.manual_seed(seed+rank)
    if train.get("deterministic",False):
        torch.backends.cudnn.benchmark=False
    roots=list(config["data"]["token_roots"])
    extra=config["data"].get("dualtalk_tokens")
    if extra and extra not in roots:
        roots.append(extra)
    if calibration:
        if not extra:
            raise ValueError("FLAME calibration requires data.dualtalk_tokens")
        roots=[extra]
    datasets=[PacketFrameDataset(root,"train") for root in roots]
    if rank==0:
        print(json.dumps({"stage":"training_index","status":"reading_train_only","roots":roots}),flush=True)
    index_started=time.monotonic()
    # CPU preparation can take tens of minutes on uncached corpora. All ranks
    # finish the atomic-file rendezvous before initializing NCCL; no watchdog
    # observes a broadcast sitting idle while rank zero scans the dataset.
    training_index=cached_training_index(datasets,build=rank==0,
                        wait_seconds=float(train.get("observer_index_wait_seconds",14400)))
    if rank==0:
        print(json.dumps({"stage":"training_index","status":"complete","seconds":time.monotonic()-index_started,
                          "rows":len(training_index["rows"]),"endpoints":sum(item["endpoint_count"] for item in training_index["domains"].values())}),flush=True)
    if owned_group:
        dist.init_process_group("nccl" if device.type=="cuda" else "gloo",
                                timeout=timedelta(seconds=int(train.get("distributed_timeout_seconds",3600))))
    training_statistics={key:value for key,value in training_index.items() if key!="rows"}
    cache_size=max(6,int(train.get("observer_cache_dialogues",12)))
    combined=MixedDialogueDataset(datasets,cache_dialogues=cache_size)
    adapter_binding=None
    if calibration:
        target_source=datasets[0].dialogues.feature_sources.get("dualtalk")
        if target_source is None:
            raise ValueError("Calibration requires an explicitly declared DualTalk feature source")
        if resumed is not None:
            # Recovery is authoritative. Do not load external A0 weights or
            # repeat the adapter copy over the stored calibrated parameters.
            adapter_binding=resumed["construction"].get("adapter_source_binding")
            if adapter_binding is None:
                raise ValueError("Calibration checkpoint lacks labelled adapter-source binding; start a new calibration experiment")
            if adapter_binding["feature_source"]!=target_source:
                raise ValueError("Calibration resume source differs from its stored adapter binding")
            observer=TokenObserver(resumed["construction"]["observer"]).to(device)
        else:
            observer,a0_payload=load_observer(config["paths"]["observation_checkpoint"],device,teacher=True)
            adapter_binding=initialize_dualtalk_adapters(observer,a0_payload,target_source)
        config["data"]["adapter_source_domain"]=adapter_binding["source_domain"]
        config["data"]["adapter_source_binding"]=copy.deepcopy(adapter_binding)
    else:
        observer=TokenObserver(config["observer"]).to(device)
    # Identical model initialization; stochastic data/masking differs per rank.
    if distributed:
        for parameter in observer.parameters():
            dist.broadcast(parameter.data,0)
    teacher=copy.deepcopy(observer).requires_grad_(False).eval()
    if calibration:
        observer.requires_grad_(False)
        observer.adapters["flame"].requires_grad_(True)
        observer.reconstruct["flame"].requires_grad_(True)
    objective=_TrainingObjective(observer,teacher,{**train,"training_statistics":training_statistics})
    wrapped=DistributedDataParallel(objective,device_ids=[local] if device.type=="cuda" else None,
                                     find_unused_parameters=True,broadcast_buffers=False) if distributed else objective
    requested_lr=float(train.get("observation_lr",train["lr"]))
    lr_cap=float(train.get("observation_lr_cap",1e-4))
    if not 0<requested_lr or not 0<lr_cap<=1e-3:
        raise ValueError("A0 learning rate must be positive and its explicit cap at most 1e-3")
    learning_rate=min(requested_lr,lr_cap)
    ema_decay=float(train.get("observation_ema_decay",.99))
    if not 0<=ema_decay<1:
        raise ValueError("observation_ema_decay must lie in [0,1)")
    optimizer=torch.optim.AdamW([p for p in observer.parameters() if p.requires_grad],lr=learning_rate,
                               weight_decay=float(train["weight_decay"]),foreach=False)
    output=Path(config["paths"]["output"])
    if rank==0:
        output.mkdir(parents=True,exist_ok=True)
        write_config(output/"config.json",config)
        (output/"training_statistics.json").write_text(json.dumps(training_statistics,indent=2),encoding="utf-8")
    step,epoch,best,next_batch=0,0,float("inf"),0
    best_gate_passed=False
    amp=bool(train.get("amp",True) and device.type=="cuda")
    amp_dtype=torch.bfloat16 if device.type=="cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler("cuda",enabled=amp and amp_dtype==torch.float16)
    if config["paths"].get("resume"):
        payload=resumed
        if not calibration and "ema" not in payload["models"]:
            raise ValueError("A0 resume requires explicit student/EMA/export identities; use old weights only as initialization")
        restore_training(payload,{"observer":observer,"coordinate_teacher" if calibration else "ema":teacher},optimizer=optimizer,scaler=scaler,config=config,restore_rng=False)
        state=payload["run_state"]
        if state.get("world_size",world)!=world:
            raise ValueError("Exact A0 resume requires the same rank count")
        step=int(payload["global_step"]); epoch=int(state.get("epoch",0)); best=float(state.get("best",best)); next_batch=int(state.get("next_batch",0))
        best_gate_passed=bool(state.get("best_gate_passed",False))
        if "rank_rng" in state:
            restore_rng_state(state["rank_rng"][rank])
        else:
            raise ValueError("A0 checkpoint lacks per-rank RNG for exact resume")
    batch_size=int(train.get("observer_batch_size",16))
    try:
        while step<max_steps:
            sampler=DialogueBalancedBatches(datasets,batch_size,rank,world,seed,epoch,training_index=training_index,
                        supervised_fraction=float(train.get("observer_supervised_fraction",.5)),subset_offset=step-next_batch,
                        modality_dropout=bool(train.get("modality_dropout",True)),active_dialogues=cache_size,
                        refresh_batches=int(train.get("observer_refresh_batches",32)),masking=train.get("masking"))
            loader=DataLoader(combined,batch_sampler=sampler,collate_fn=collate_observation_samples,num_workers=0,
                              generator=torch.Generator().manual_seed(seed+epoch))
            for batch_index,batch in enumerate(loader):
                if batch_index<next_batch:
                    continue
                observer.eval() if calibration else observer.train()
                batch=_to(batch,device)
                subset=sampler.subset_for_batch(batch_index)
                requested_subset=subset if train.get("masking") else (list(SUBSETS)[step%len(SUBSETS)] if train.get("modality_dropout",True) else "AVT")
                warmup=int(train.get("observation_warmup_steps",0))
                for group in optimizer.param_groups:
                    group["lr"]=learning_rate*min(1.,(step+1)/max(1,warmup))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type,enabled=amp,dtype=amp_dtype if device.type=="cuda" else torch.bfloat16):
                    losses=wrapped(batch,subset)
                finite=torch.isfinite(losses["total"]).to(device=device,dtype=torch.int32)
                if distributed:
                    dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not int(finite):
                    raise FloatingPointError("Non-finite V3 observation objective")
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                norm=nn.utils.clip_grad_norm_(observer.parameters(),float(train["clip_grad"]))
                finite=torch.isfinite(norm).to(device=device,dtype=torch.int32)
                if distributed:
                    dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not int(finite):
                    # A skipped scaled update is not a completed optimization
                    # step. Stop all ranks before optimizer/EMA/cursor advance.
                    raise FloatingPointError("Non-finite A0/calibration gradients; no optimizer step was counted")
                scaler.step(optimizer); scaler.update()
                with torch.no_grad():
                    for target,current in zip(teacher.parameters(),observer.parameters()):
                        if not calibration:
                            target.lerp_(current,1-ema_decay)
                step+=1
                next_batch=batch_index+1
                if step%int(train.get("log_every",10))==0 or step==1:
                    keys=sorted(losses)
                    values=torch.stack([losses[key].detach().float() for key in keys])
                    if distributed:
                        dist.all_reduce(values); values/=world
                    stats={key:float(value) for key,value in zip(keys,values)}
                    stats.update(step=step,epoch=epoch,subset=subset,requested_subset=requested_subset,
                                 grad_norm=float(norm),world_size=world,learning_rate=optimizer.param_groups[0]["lr"],
                                 requested_learning_rate=requested_lr,learning_rate_cap=lr_cap,ema_decay=ema_decay,
                                 local_packets=batch["samples"],local_endpoints=batch["endpoints"],
                                 local_dialogues=batch["dialogues"],local_roles=sorted(set(batch["roles"])),
                                 supervision_protocol="global_valid_weighted_endpoint_sums")
                    if adapter_binding is not None:
                        stats["adapter_source_domain"]=adapter_binding["source_domain"]
                    if rank==0:
                        print(json.dumps(stats),flush=True)
                        with (output/"metrics.jsonl").open("a",encoding="utf-8") as stream:
                            stream.write(json.dumps(stats)+"\n")
                        (output/"training_status.json").write_text(json.dumps({"stage":"calibration" if calibration else "observation",
                            "step":step,"max_steps":max_steps,"status":"running","world_size":world}),encoding="utf-8")
                if step%int(train["validate_every"])==0 or step==max_steps:
                    rank_rng=[None for _ in range(world)]
                    if distributed:
                        dist.all_gather_object(rank_rng,capture_rng_state())
                    else:
                        rank_rng[0]=capture_rng_state()
                    if distributed:
                        dist.barrier()
                    if rank==0:
                        if calibration:
                            metrics=evaluate(observer,roots,device,int(train.get("validation_max_dialogues",16)),batch_size,
                                             teacher=teacher,calibration=True)
                            selected_model="student"
                            passed=False
                        else:
                            evaluations={name:evaluate(model,roots,device,int(train.get("validation_max_dialogues",16)),batch_size,
                                               training_statistics=training_statistics,model_name=name)
                                         for name,model in (("student",observer),("ema",teacher))}
                            gates={name:semantic_gate(value) for name,value in evaluations.items()}
                            selected_model=min(evaluations,key=lambda name:(not gates[name]["passed"],evaluations[name]["selection_loss"]))
                            metrics={**evaluations[selected_model],**evaluations,"selected_model":selected_model,
                                     "semantic_gate":gates[selected_model],"student_gate":gates["student"],"ema_gate":gates["ema"]}
                            passed=gates[selected_model]["passed"]
                        metrics["step"]=step
                        if adapter_binding is not None:
                            metrics["adapter_source_domain"]=adapter_binding["source_domain"]
                            metrics["adapter_source_binding"]=copy.deepcopy(adapter_binding)
                        print(json.dumps({"validation":metrics}),flush=True)
                        improved=(passed and not best_gate_passed) or (passed==best_gate_passed and metrics["selection_loss"]<best)
                        # Always export a reviewable checkpoint, even when smoke
                        # quality is untrained. The separate gate blocks Phase B.
                        improved=improved or not (output/"best.pt").exists()
                        if improved:
                            best=metrics["selection_loss"]
                            best_gate_passed=passed
                        # Downstream targets must include the calibrated visual
                        # route. Keep the old AT anchor separately for resuming
                        # calibration, where it must remain fixed.
                        saved_models=({"observer":observer,"teacher":observer,"coordinate_teacher":teacher} if calibration
                                      else {"observer":observer,"ema":teacher,"teacher":observer if selected_model=="student" else teacher})
                        construction={"observer":observer.construction(),"exported_model":selected_model,
                                      "training_statistics":training_statistics}
                        if adapter_binding is not None:
                            construction.update(adapter_source_domain=adapter_binding["source_domain"],
                                                adapter_source_binding=copy.deepcopy(adapter_binding))
                        kwargs=dict(models=saved_models,config=config,construction=construction,
                                    kind="calibration_v3" if calibration else "observation_v3",step=step,optimizer=optimizer,metrics=metrics,
                                    run_state={"epoch":epoch,"next_batch":next_batch,"best":best,"best_gate_passed":best_gate_passed,
                                               "world_size":world,"rank_rng":rank_rng,"training_statistics":training_statistics,
                                               "sampling_protocol":"mixed_active_dialogues_real_endpoints_v3.1"},scaler=scaler)
                        save_checkpoint(output/"last.pt",**kwargs)
                        if improved:
                            save_checkpoint(output/"best.pt",**kwargs)
                        (output/"validation.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
                    if distributed:
                        dist.barrier()
                if step>=max_steps:
                    break
            epoch+=1
            next_batch=0
    finally:
        if owned_group:
            dist.destroy_process_group()
    if rank==0:
        (output/"training_status.json").write_text(json.dumps({"stage":"calibration" if calibration else "observation",
            "step":step,"max_steps":max_steps,"status":"complete","world_size":world,
            "semantic_gate_passed":best_gate_passed if not calibration else None,
            "semantic_gate_required":bool(train.get("require_a0_gate",True)) and not calibration}),encoding="utf-8")
    return str(output/"best.pt")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    run(args.config)


if __name__=="__main__":
    main()
