"""Frozen feature caches and fixed-origin banks for staged dynamics training.

Feature caches never contain trainable event/action outputs. Origin banks do:
their producer fingerprint therefore covers every observer and state weight.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import fields
from pathlib import Path

import torch
from torch.nn import functional as F

from emotion_ssm.data.packets_v3 import collate_role_features
from emotion_ssm.models.state_core import EmotionMemory, StateObservation
from emotion_ssm.train.dynamics_v3 import future_matches, future_endpoint_queries, unit_state_readout

CACHE_REVISION = 'frozen-head-inputs-v1-fp32'
LOCKED_CORE = ('baseline', 'fast_correction_logits', 'slow_correction_logits',
               'event_strength', 'coupling_strength')
UPDATE_PREFIXES = ('event_projection.', 'influence.', 'relation_drive.')
HEAD_PREFIXES = ('event_head.', 'action_head.')
OBS_FIELDS = ('aff', 'reliability', 'modality_mask', 'event_present', 'action_present',
              'action_duration', 'fresh_observation')


def weight_digest(models, exclude_heads=False):
    result = hashlib.sha256()
    for prefix, model in sorted(models.items()):
        values = model if isinstance(model, dict) else model.state_dict()
        for key, value in sorted(values.items()):
            if exclude_heads and prefix == 'observer' and key.startswith(HEAD_PREFIXES):
                continue
            tensor = value.detach().contiguous().cpu()
            result.update((prefix + '.' + key + str(tensor.shape) + str(tensor.dtype)).encode())
            result.update(tensor.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def memory_cat(values):
    return EmotionMemory(**{f.name: torch.cat([getattr(v, f.name) for v in values]) for f in fields(EmotionMemory)})


def memory_index(value, index):
    return EmotionMemory(**{f.name: getattr(value, f.name)[index] for f in fields(EmotionMemory)})


def parameter_partition(observer, core, phase):
    """Observation overwrite rates and input gains cannot grow in any phase."""
    if phase not in {'calibration', 'fixed', 'joint', 'readapt'}:
        raise ValueError('Unknown staged dynamics phase')
    observer.requires_grad_(False)
    core.requires_grad_(False)
    if phase in {'calibration', 'joint'}:
        observer.event_head.requires_grad_(True)
        observer.action_head.requires_grad_(True)
        for name, parameter in core.named_parameters():
            if name.startswith(UPDATE_PREFIXES):
                parameter.requires_grad_(True)
    if phase in {'fixed', 'readapt', 'joint'}:
        for name, parameter in core.named_parameters():
            if name not in LOCKED_CORE and not name.startswith(UPDATE_PREFIXES):
                parameter.requires_grad_(True)
    return [p for model in (observer, core) for p in model.parameters() if p.requires_grad]


def lock_snapshot(core):
    return {name: dict(core.named_parameters())[name].detach().cpu().clone() for name in LOCKED_CORE}


def assert_locked(core, reference):
    for name, value in reference.items():
        actual = dict(core.named_parameters())[name]
        if actual.requires_grad or not torch.equal(actual.detach().cpu(), value):
            raise RuntimeError(f'Observation-coverage/reference guard violated: {name}')


def state_anchor_loss(current, teacher, valid, diagnostics):
    """Supervise posterior z, not a frozen observer or each memory component.

    Coverage/confidence are detached. Missing observations contribute no anchor.
    This objective is never the checkpoint selection criterion.
    """
    target = teacher.detach().float()
    weight = diagnostics['evidence'].detach().float() * valid.float()
    error = (current.z.float() - target).square().sum(-1)
    return (error * weight).sum() / weight.sum().clamp_min(1.)


class EncodedDialogueCache:
    def __init__(self, observer, teacher, directory, device, batch_size=32):
        self.observer, self.teacher, self.device = observer, teacher, device
        self.batch_size = int(batch_size)
        self.fingerprint = weight_digest({'observer': observer, 'teacher': teacher}, exclude_heads=True)
        self.directory = Path(directory) / self.fingerprint[:20]
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    def get(self, collection, index, subset='AVT'):
        domain, sample = collection.index[index]
        dataset = collection.datasets[domain]
        identity = collection.identity(index)
        manifest = dataset.root / 'manifest.json'
        manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        metadata = {'revision': CACHE_REVISION, 'encoder': self.fingerprint,
                    'manifest': manifest_hash, 'identity': identity, 'subset': subset}
        key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        path = self.directory / (key + '.pt')
        if path.exists():
            result = torch.load(path, map_location='cpu', weights_only=False)
            if result['metadata'] != metadata:
                raise ValueError('Frozen feature cache provenance mismatch')
            self.hits += 1
            return result
        dialogue = collection[index]
        packets = dialogue['packets']
        output = {name: [] for name in (*OBS_FIELDS, 'event_input', 'action_input')}
        gold, valid = [], []
        with torch.no_grad():
            for start in range(0, len(packets), max(1, self.batch_size // 2)):
                features = [role for packet in packets[start:start + max(1, self.batch_size // 2)] for role in packet['roles']]
                batch = collate_role_features(features, self.device)
                # Cache precision is explicit and independent of training AMP.
                with torch.autocast(device_type=self.device.type, enabled=False):
                    encoded = self.observer.encode(batch, subset=subset)
                    target = self.teacher.encode(batch, subset='AVT')
                obs = encoded['observation']
                for name in OBS_FIELDS:
                    output[name].append(getattr(obs, name).detach().cpu())
                output['event_input'].append(encoded['head_inputs']['event'].detach().cpu())
                output['action_input'].append(encoded['head_inputs']['action'].detach().cpu())
                gold.append(target['observation'].aff.detach().cpu())
                valid.append((target['valid'] & batch['fresh_observation'].bool().any(-1)).cpu())
        result = {'metadata': metadata, 'domain': domain, 'identity': identity,
                  'times': [float(p['end']) for p in packets],
                  'dt': [float(p['dt']) for p in packets],
                  'labels': [copy.deepcopy(p.get('targets', [[], []])) for p in packets],
                  'gold': torch.cat(gold).reshape(len(packets), 2, -1),
                  'valid': torch.cat(valid).reshape(len(packets), 2),
                  'observations': {name: torch.cat(rows).reshape(len(packets), 2, *rows[0].shape[1:])
                                   for name, rows in output.items()}}
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
        torch.save(result, temporary)
        temporary.replace(path)
        self.misses += 1
        return result


def cached_pair(observer, encoded, tick, device, missing=False):
    pair = []
    for role in (0, 1):
        values = {name: encoded['observations'][name][tick, role:role+1].to(device) for name in OBS_FIELDS}
        event_input = encoded['observations']['event_input'][tick, role:role+1].to(device)
        action_input = encoded['observations']['action_input'][tick, role:role+1].to(device)
        values['event'] = observer.event_head(event_input) * values['event_present'][:, None]
        values['action'] = observer.action_head(action_input) * values['action_present'][:, None]
        values['event_id'] = torch.full((1,), tick, dtype=torch.long, device=device)
        if missing:
            for name in ('modality_mask', 'fresh_observation', 'event_present', 'action_present', 'action_duration'):
                values[name] = torch.zeros_like(values[name])
        pair.append(StateObservation(**values))
    return pair


def bank_from_rows(memories, records, encoded_records, horizons, producer, protocol, cpu=True):
    states = memory_cat(memories)
    n, d = len(memories), states.fast.shape[-1]
    targets = torch.zeros(n, len(horizons), 2, d)
    valid = torch.zeros(n, len(horizons), 2, dtype=torch.bool)
    endpoint_rows = []
    for row, (record, tick) in enumerate(records):
        encoded = encoded_records[record]
        for h, future in future_matches(encoded['times'], tick, horizons):
            column = horizons.index(h)
            targets[row, column] = encoded['gold'][future]
            valid[row, column] = encoded['valid'][future]
        queries = encoded.get('_queries')
        if queries is None:
            packets = [{'targets': labels} for labels in encoded['labels']]
            queries = future_endpoint_queries(encoded['times'], packets, horizons)
            encoded['_queries'] = queries
        for query in queries.get(tick, []):
            endpoint_rows.append((row, copy.deepcopy(query)))
    return {'states': states.to('cpu') if cpu else states, 'targets': targets, 'valid': valid,
            'domain': torch.tensor([encoded_records[r]['domain'] for r, _ in records]),
            'keys': [(encoded_records[r]['identity'], encoded_records[r]['times'][t]) for r, t in records],
            'endpoints': endpoint_rows, 'producer': producer, 'protocol': protocol,
            'horizons': list(horizons)}


@torch.no_grad()
def build_origin_bank(observer, core, cache, collection, indices, horizons, stride=4,
                      gap_seconds=0, progress=None):
    producer = weight_digest({'observer': observer, 'state': core})
    memories, records, encoded_records = [], [], []
    totals = torch.zeros(5, dtype=torch.float64, device=cache.device)
    for number, index in enumerate(indices):
        encoded = cache.get(collection, index)
        encoded_records.append(encoded)
        state = core.initialize(1, cache.device)
        for tick, dt in enumerate(encoded['dt']):
            # A declared periodic block gap, never data-dependent masking.
            missing = gap_seconds > 0 and encoded['times'][tick] % 32 >= 32-gap_seconds
            pair = cached_pair(observer, encoded, tick, cache.device, missing)
            diagnostics = {}
            state = core.advance(state, pair, dt, diagnostics=diagnostics)
            mask = encoded['valid'][tick].to(cache.device)[None]
            truth = encoded['gold'][tick].to(cache.device)[None]
            totals[0] += (state.z[mask]-truth[mask]).square().sum()
            totals[1] += (diagnostics['input_conditioned_prior'][mask]-truth[mask]).square().sum()
            totals[2] += mask.sum()*state.fast.shape[-1]
            totals[3] += diagnostics['correction_gain'].sum()
            totals[4] += diagnostics['correction_gain'].numel()
            if tick % stride == stride-1 and any(bool(encoded['valid'][j].any())
                    for _, j in future_matches(encoded['times'], tick, horizons)):
                memories.append(state.detach().to('cpu'))
                records.append((number, tick))
        if progress:
            progress(number+1, len(indices), len(memories))
    if not memories:
        raise ValueError('No valid fixed-origin future pairs in selected training/validation data')
    result = bank_from_rows(memories, records, encoded_records, list(horizons), producer,
                            f'complete_dialogues_stride{stride}_gap{gap_seconds}s')
    result['current'] = dict(zip(('posterior_sse', 'input_conditioned_prior_sse', 'elements',
                                'correction_gain_sum', 'correction_gain_count'), totals.cpu().tolist()))
    # Count each teacher packet-role once, not once per future horizon.
    gold = torch.cat([e['gold'][e['valid']] for e in encoded_records])
    result['teacher_sum'], result['teacher_count'] = gold.double().sum(0), len(gold)
    return result


def prediction_loss(core, bank, indices, device, observer=None, label_weight=0.):
    origins = memory_index(bank['states'], indices).to(device)
    predictions = core.forecast(origins, bank['horizons'])
    target, valid = bank['targets'][indices].to(device), bank['valid'][indices].to(device)
    per_h = []
    for column, state in enumerate(predictions):
        mask = valid[:, column]
        error = (state.z.float()-target[:, column].detach().float()).square().sum(-1)
        per_h.append((error*mask).sum()/mask.sum().clamp_min(1))
    loss = torch.stack(per_h).mean()
    if observer is not None and label_weight:
        # True endpoint times are propagated in a batch with one dt per row.
        mapping = {}
        for local, original in enumerate(indices.tolist()):
            mapping.setdefault(original, []).append(local)
        endpoint = [(local, query) for row, query in bank['endpoints']
                    for local in mapping.get(row, [])]
        if endpoint:
            chosen = torch.tensor([row for row, _ in endpoint], device=device)
            seconds = torch.tensor([query['seconds'] for _, query in endpoint], device=device)
            states = core._propagate(memory_index(origins, chosen), seconds)
            roles = torch.tensor([query['role'] for _, query in endpoint], device=device)
            affect = states.z[torch.arange(len(endpoint), device=device), roles]
            decoded = observer.decode_affect(unit_state_readout(affect))
            if getattr(core, '_execution_optimized', False):
                labels = [query['label'] for _, query in endpoint]
                label_loss = vectorized_endpoint_loss(decoded, labels)
                if label_loss is not None:
                    loss = loss + label_weight*label_loss
                return loss, predictions
            terms = []
            for row, (_, query) in enumerate(endpoint):
                label = query['label']; terms_row = []
                emotion = int(label.get('emotion', -1))
                if 0 <= emotion < 7:
                    terms_row.append(F.cross_entropy(decoded['emotion_logits'][row:row+1],
                                      torch.tensor([emotion], device=device)))
                mask = torch.tensor(label.get('vad_mask', [False]*3), device=device, dtype=torch.bool)
                if mask.any():
                    truth = torch.tensor(label['vad'], device=device)[mask]
                    terms_row.append(F.mse_loss(decoded['vad'][row, mask], truth))
                if label.get('intensity_mask', False):
                    terms_row.append((decoded['intensity'][row]-float(label['intensity'])).square())
                if terms_row:
                    terms.append(torch.stack(terms_row).sum())
            if terms:
                loss = loss + label_weight*torch.stack(terms).mean()
    return loss, predictions


def vectorized_endpoint_loss(decoded, labels):
    """Same mean of per-query CE + masked-coordinate VAD mean + intensity.

    Missing labels do not enter either numerator or denominator. Constructing
    masks on CPU avoids one GPU-to-CPU truth-value check per endpoint.
    """
    device = decoded['vad'].device
    emotions = [int(label.get('emotion', -1)) for label in labels]
    emotion_valid = [0 <= value < 7 for value in emotions]
    vad_masks = [label.get('vad_mask', [False]*3) for label in labels]
    intensity_valid = [bool(label.get('intensity_mask', False)) for label in labels]
    row_valid = [a or any(b) or c for a, b, c in zip(emotion_valid, vad_masks, intensity_valid)]
    if not any(row_valid):
        return None
    emotion = torch.tensor([value if valid else 0 for value, valid in zip(emotions, emotion_valid)], device=device)
    ce = F.cross_entropy(decoded['emotion_logits'], emotion, reduction='none')
    ce = ce * torch.tensor(emotion_valid, device=device)
    mask = torch.tensor(vad_masks, device=device, dtype=torch.bool)
    truth = torch.tensor([label.get('vad', [0., 0., 0.]) for label in labels], device=device)
    delta = torch.where(mask, decoded['vad']-truth, torch.zeros_like(decoded['vad']))
    vad = delta.square().sum(-1)/mask.sum(-1).clamp_min(1)
    mask = torch.tensor(intensity_valid, device=device)
    truth = torch.tensor([float(label.get('intensity', 0.)) for label in labels], device=device)
    delta = torch.where(mask, decoded['intensity']-truth, torch.zeros_like(decoded['intensity']))
    terms = ce + vad + delta.square()
    return terms[torch.tensor(row_valid, device=device)].mean()
