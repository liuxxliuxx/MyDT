"""Read-only retrospective decay evaluation of retained staged checkpoints.

Run from the matching code snapshot. JSON goes to stdout, progress to stderr;
this tool never writes server files or changes a training process.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch

from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.train.dynamics_v3 import DialogueCollection
from emotion_ssm.train.dynamics_staged_v3 import source_balanced_indices
from emotion_ssm.train.staged_dynamics_support import (
    CACHE_REVISION, build_origin_bank, memory_cat, weight_digest,
)


def read_events(path):
    raw = path.read_bytes()
    rows = []
    for line in raw.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass  # An active writer may not have finished the final line.
    return rows, hashlib.sha256(raw).hexdigest()


class ReadOnlyCache:
    def __init__(self, observer, teacher, directory, device):
        self.device = device
        self.fingerprint = weight_digest({'observer': observer, 'teacher': teacher}, exclude_heads=True)
        self.directory = Path(directory) / self.fingerprint[:20]
        self.loaded = {}

    def get(self, collection, index, subset='AVT'):
        domain, _ = collection.index[index]
        dataset = collection.datasets[domain]
        manifest_hash = hashlib.sha256((dataset.root/'manifest.json').read_bytes()).hexdigest()
        metadata = dict(revision=CACHE_REVISION, encoder=self.fingerprint,
                        manifest=manifest_hash, identity=collection.identity(index), subset=subset)
        key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        if key not in self.loaded:
            path = self.directory/(key+'.pt')
            if not path.exists():
                raise FileNotFoundError('Read-only audit requires an existing feature cache: '+str(path))
            value = torch.load(path, map_location='cpu', weights_only=False)
            if value['metadata'] != metadata:
                raise ValueError('Cache provenance mismatch')
            self.loaded[key] = value
        return self.loaded[key]


def merge_banks(banks):
    result = dict(banks[0])
    result['states'] = memory_cat([b['states'] for b in banks])
    for name in ('targets', 'valid', 'domain'):
        result[name] = torch.cat([b[name] for b in banks])
    result['keys'] = sum([b['keys'] for b in banks], [])
    return result


def aggregate(sums):
    sums = sums.cpu()
    error, count = sums[..., 0], sums[..., 1]
    per_h = []
    for h in range(error.shape[1]):
        valid = count[:, h] > 0
        if valid.any():
            per_h.append((error[valid, h]/count[valid, h]).mean())
    return dict(mse=float(error.sum()/count.sum()),
                macro_mse=float(torch.stack(per_h).mean()),
                elements=int(count.sum()), sums=sums.tolist())


@torch.no_grad()
def decay_metrics(core, bank, device):
    state = bank['states'].to(device)
    truth, valid = bank['targets'].to(device), bank['valid'].to(device)
    domains = bank['domain'].to(device)
    result = {}
    for method in ('pure_decay', 'hold'):
        sums = torch.zeros(3, len(bank['horizons']), 2, dtype=torch.float64, device=device)
        for h, seconds in enumerate(bank['horizons']):
            prediction = core.decay_only(state, seconds).z if method == 'pure_decay' else state.z
            error = (prediction.double()-truth[:, h].double()).square().sum(-1)
            for domain in range(3):
                mask = valid[:, h] & (domains == domain)[:, None]
                sums[domain, h, 0] = (error*mask).sum()
                sums[domain, h, 1] = mask.sum()*truth.shape[-1]
        result[method] = aggregate(sums)
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--stage', choices=['diagnostic', 'formal'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, nargs='+', help='Evaluate only these retained checkpoint steps')
    args = parser.parse_args()
    root, stage = Path(args.run), args.stage
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, log_hash = read_events(root/stage/'rank0.jsonl')
    metrics = [r['metrics'] for r in rows if r.get('event') == 'validation']
    by_hash = {}
    for m in metrics:
        by_hash.setdefault(m['candidate_hash'], []).append(m)
    candidates, checkpoints = {}, []
    paths = [root/stage/'best.pt', root/stage/'last.pt']
    if stage == 'diagnostic':
        paths.append(root/'execution_benchmark.source.pt')
    for path in paths:
        if not path.exists():
            continue
        payload = torch.load(path, map_location='cpu', weights_only=False)
        checkpoints.append(dict(path=str(path), step=payload['global_step']))
        variants = {'models': payload['models']}
        for name in ('initial_models', 'bank_producer_models'):
            if name in payload.get('run_state', {}):
                variants[name] = payload['run_state'][name]
        for name, models in variants.items():
            digest = weight_digest({'observer': models['observer'], 'state': models['state']})
            if digest not in by_hash and name == 'models' and payload.get('metrics', {}).get('candidate_hash') == digest:
                by_hash[digest] = [payload['metrics']]
            if digest not in by_hash:
                continue
            if digest not in candidates:
                candidates[digest] = dict(models=models, config=payload['config'],
                    construction=payload['construction'], sources=[])
            candidates[digest]['sources'].append(str(path.relative_to(root))+':'+name)
    references = merge_banks([torch.load(root/stage/f'reference_val_rank{rank}.pt',
        map_location='cpu', weights_only=False) for rank in (0, 1)])
    report = dict(stage=stage, captured_at=datetime.datetime.now().astimezone().isoformat(),
                  checkpoints=checkpoints, evaluation=[], protocol='existing_checkpoint_exact_decay_readonly_v1')
    for digest, candidate in sorted(candidates.items(), key=lambda kv: min(m['step'] for m in by_hash[kv[0]])):
        logs = sorted(by_hash[digest], key=lambda m: m['step'])
        if args.steps and not any(m['step'] in args.steps for m in logs):
            continue
        print(json.dumps(dict(stage=stage, steps=[m['step'] for m in logs], event='decay_audit_started')), file=sys.stderr, flush=True)
        config = candidate['config']
        observer = TokenObserver(candidate['construction']['observer']).to(device).eval()
        teacher = TokenObserver(candidate['construction']['observer']).to(device).eval()
        core = UnifiedEmotionStateCore.from_config(candidate['construction']['state']).to(device).eval()
        for name, model in [('observer', observer), ('teacher', teacher), ('state', core)]:
            model.load_state_dict(candidate['models'][name], strict=True)
            model.requires_grad_(False)
        # Avoid JIT cache writes; use the accepted same-math execution path.
        core.configure_execution('optimized')
        cache = ReadOnlyCache(observer, teacher, config['paths']['feature_cache'], device)
        collection = DialogueCollection(config['data']['token_roots'], 'val')
        selected = source_balanced_indices(collection, config['staged']['validation_dialogues_per_domain'], config['train']['seed']+17011)
        item = dict(candidate_hash=digest, steps=[m['step'] for m in logs], sources=candidate['sources'],
                    fast_tau=(1/core.rates()[0]).cpu().tolist(), slow_tau=(1/core.rates()[1]).cpu().tolist(), protocols={})
        for protocol in ('fixed', 'live', 'gap'):
            bank = references if protocol == 'fixed' else build_origin_bank(observer, core, cache, collection,
                selected, config['train']['forecast_seconds'], config['staged']['origin_stride'],
                config['staged']['block_gap_seconds'] if protocol == 'gap' else 0)
            # Replay order differs from rank concatenation; compare sample keys as sets.
            if sorted(bank['keys']) != sorted(references['keys']):
                raise AssertionError('Validation origin identity changed')
            measured = decay_metrics(core, bank, device)
            expected = logs[-1][protocol]['methods']['hold']
            delta = abs(measured['hold']['macro_mse']-expected['macro_mse'])
            if delta > 2e-8 or measured['hold']['elements'] != expected['elements']:
                raise AssertionError(f'{stage} {item["steps"]} {protocol}: replay hold mismatch {delta}')
            measured['hold_log_abs_difference'] = delta
            item['protocols'][protocol] = measured
            print(json.dumps(dict(stage=stage, steps=item['steps'], protocol=protocol,
                pure_decay=measured['pure_decay']['macro_mse'], hold_check=delta)), file=sys.stderr, flush=True)
        report['evaluation'].append(item)
        del observer, teacher, core, cache
    rows, log_hash = read_events(root/stage/'rank0.jsonl')
    report['events'] = rows
    report['log_sha256'] = log_hash
    report['completed_at'] = datetime.datetime.now().astimezone().isoformat()
    report['status'] = json.loads((root/stage/'training_status.json').read_text())
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
