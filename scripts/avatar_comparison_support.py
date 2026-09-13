"""Audited path relocation and shared-input Avatar evaluation helpers."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


class VerifiedRelocation:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.project = self.root.parent.parent
        self.source = self.root / 'source'
        self.export = self.source / 'runs/avatar_comparison_20260913_s2_export'
        self.manifest = json.loads((self.export / 'transfer_manifest.json').read_text())
        self.original = Path(self.manifest['source_root'])
        self.verified = {}

    def relocate(self, original):
        return self.source / Path(original).relative_to(self.original)

    def verify(self, path, record):
        path = Path(path)
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if self.verified.get(str(path)) == signature:
            return signature
        if stat.st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError('Relocated content does not match server1: ' + str(path))
        if (path.stat().st_size, path.stat().st_mtime_ns) != signature:
            raise ValueError('File changed during verification: ' + str(path))
        self.verified[str(path)] = signature
        return signature

    def token_path(self, path):
        path = Path(path).resolve()
        relative = path.relative_to(self.source).as_posix()
        self.verify(path, self.manifest['files'][relative])
        return path

    def raw_signature(self, signature):
        relocated = []
        for filename, size, modified in signature:
            relative = Path(filename).relative_to(self.original).as_posix()
            expected = self.manifest['raw_files'][relative]
            if (size, modified) != (expected['bytes'], expected['source_mtime_ns']):
                raise ValueError('Raw source changed after token extraction: ' + filename)
            path = self.project / relative
            actual_size, actual_time = self.verify(path, expected)
            relocated.append((str(path), actual_size, actual_time))
        return relocated

    def install(self):
        import emotion_ssm.data.packets_v3 as packets
        original = packets.TokenPacketDataset
        if getattr(original, '_verified_relocation', False):
            raise ValueError('Relocation must be installed once per worker')
        verifier = self

        class RelocatedPackets(original):
            _verified_relocation = True

            def __getitem__(self, index):
                name = self.ids[index]
                verifier.token_path(self.root / self.manifest['dialogues'][name]['path'])
                data = super().__getitem__(index)
                data = dict(data)
                data['audit'] = dict(data.get('audit', {}))
                data['audit']['raw_signature'] = verifier.raw_signature(data['audit'].get('raw_signature', []))
                return data

        packets.TokenPacketDataset = RelocatedPackets

    def config(self, config):
        config = copy.deepcopy(config)
        config['data']['token_roots'] = [str(self.relocate(p)) for p in config['data']['token_roots']]
        config['data']['dualtalk_tokens'] = str(self.relocate(config['data']['dualtalk_tokens']))
        config['data']['dualtalk_raw'] = str(self.project / 'datasets/dualtalk')
        return config


def identity(records):
    return sorted([dict(dialogue=r['dialogue'], elements=r['elements'],
                        evaluated_chunks=r['evaluated_chunks'], valid_frames=r['valid_frames'])
                   for r in records], key=lambda r: r['dialogue'])


def select_smoke_names(dataset, maximum=16):
    return dataset.names[:maximum]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)
