"""Export immutable evaluation weights and source/feature fingerprints for another host."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def main():
    root = Path('/home/s21_yhr/lzh/MyDualTalk')
    source = root / 'runs/avatar_stability_20260913_gpu23'
    output = root / 'runs/avatar_comparison_20260913_s2_export'
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source / 'code'))
    import torch
    torch.set_num_threads(1)
    checkpoint = source / 'formal/best.pt'
    snapshot = output / 'current_step004000.pt'
    if not snapshot.exists():
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if payload['global_step'] != 4000:
            raise ValueError('Expected the explicitly selected step4000 checkpoint')
        payload = dict(payload, optimizer=None, scaler=None, run_state={}, rng_state=None)
        torch.save(payload, snapshot.with_suffix('.tmp'))
        snapshot.with_suffix('.tmp').replace(snapshot)
    payload = torch.load(snapshot, map_location='cpu', weights_only=False, mmap=True)
    if payload['global_step'] != 4000:
        raise ValueError('Snapshot step mismatch')
    config = payload['config']
    (output / 'source_config.json').write_text(json.dumps(config, indent=2))
    (output / 'source_construction.json').write_text(json.dumps(payload['construction'], indent=2))
    tokens = Path(config['data']['dualtalk_tokens'])
    manifest = json.loads((tokens / 'manifest.json').read_text())
    files = {}

    def register(path):
        path = Path(path).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError('Source file unavailable/outside project: ' + str(path))
        relative = path.relative_to(root).as_posix()
        if relative not in files:
            files[relative] = dict(bytes=path.stat().st_size, sha256=sha256(path),
                                   mtime_ns=path.stat().st_mtime_ns)
        return relative

    common = [register(snapshot), register(output / 'source_config.json'), register(output / 'source_construction.json')]
    for path in (source / 'code').rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.md', '.json') and '.pytest_cache' not in path.parts:
            common.append(register(path))
    for token_root in config['data']['token_roots']:
        common.append(register(Path(token_root) / 'manifest.json'))
    groups = {'evaluation': [], 'training': []}
    raw = {}
    # Raw audio/FLAME are already present on server2. Fingerprints permit
    # relocation without trusting timestamps from an independently copied tree.
    for split in ('val', 'test', 'ood', 'train'):
        group = 'training' if split == 'train' else 'evaluation'
        for index, name in enumerate(manifest['splits'][split]):
            path = (tokens / manifest['dialogues'][name]['path']).resolve()
            groups[group].append(register(path))
            # Audit signatures name the exact raw files used for each token cache.
            data = torch.load(path, map_location='cpu', weights_only=False)
            for filename, size, modified in data.get('audit', {}).get('raw_signature', []):
                p = Path(filename)
                if (p.stat().st_size, p.stat().st_mtime_ns) != (size, modified):
                    raise ValueError('Source raw inputs changed after feature extraction: ' + filename)
                relative = p.resolve().relative_to(root).as_posix()
                if relative not in raw:
                    raw[relative] = dict(bytes=size, source_mtime_ns=modified, sha256=sha256(p))
            del data
            if index % 500 == 0:
                print(json.dumps(dict(stage='fingerprints', split=split, completed=index,
                                      total=len(manifest['splits'][split]))), flush=True)
    record = dict(created=time.time(), source_root=str(root), source_run=str(source),
                  current_step=4000, files=files, raw_files=raw,
                  token_root=config['data']['dualtalk_tokens'], token_roots=config['data']['token_roots'],
                  manifests_preserved_verbatim=True, groups=groups, common=common)
    record_path = output / 'transfer_manifest.json'
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    smoke = []
    for split in ('val', 'train'):
        for name in manifest['splits'][split][:16]:
            smoke.append((tokens / manifest['dialogues'][name]['path']).resolve().relative_to(root).as_posix())
    (output / 'smoke_files.txt').write_text('\n'.join(sorted(set(common + smoke + [record_path.relative_to(root).as_posix()]))) + '\n')
    for group, names in groups.items():
        selected = sorted(set(common + names + [record_path.relative_to(root).as_posix()]))
        (output / (group + '_files.txt')).write_text('\n'.join(selected) + '\n')
    receipt = dict(status='ready', output=str(output), current_step=4000,
                   evaluation_bytes=sum(files[p]['bytes'] for p in set(common + groups['evaluation'])),
                   training_bytes=sum(files[p]['bytes'] for p in set(groups['training'])),
                   raw_files=len(raw), files=len(files), completed=time.time())
    (output / 'export_status.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
