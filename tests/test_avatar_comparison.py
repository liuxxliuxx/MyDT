import hashlib
import json
from pathlib import Path
import pytest

from scripts.avatar_comparison_support import VerifiedRelocation, identity


def fixture_relocation(tmp_path):
    project = tmp_path / 'project'
    root = project / 'runs/comparison'
    export = root / 'source/runs/avatar_comparison_20260913_s2_export'
    export.mkdir(parents=True)
    raw = project / 'datasets/dualtalk/train/a.wav'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'original-wave')
    record = dict(bytes=raw.stat().st_size, source_mtime_ns=12345,
                  sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    manifest = dict(source_root='/original', files={}, raw_files={'datasets/dualtalk/train/a.wav':record})
    (export / 'transfer_manifest.json').write_text(json.dumps(manifest))
    return VerifiedRelocation(root), raw, record


def test_content_verified_relocation_preserves_original_manifest(tmp_path):
    verifier, raw, record = fixture_relocation(tmp_path)
    signature = [('/original/datasets/dualtalk/train/a.wav', record['bytes'],12345)]
    before = (verifier.export / 'transfer_manifest.json').read_bytes()
    remapped = verifier.raw_signature(signature)
    assert remapped == [(str(raw),raw.stat().st_size,raw.stat().st_mtime_ns)]
    assert signature[0][2] == 12345
    assert (verifier.export / 'transfer_manifest.json').read_bytes() == before


def test_changed_raw_content_is_rejected(tmp_path):
    verifier, raw, record = fixture_relocation(tmp_path)
    signature = [('/original/datasets/dualtalk/train/a.wav',record['bytes'],12345)]
    verifier.raw_signature(signature)
    raw.write_bytes(b'modified-wave')
    with pytest.raises(ValueError, match='does not match'):
        verifier.raw_signature(signature)


def test_stale_source_signature_is_rejected(tmp_path):
    verifier, _, record = fixture_relocation(tmp_path)
    with pytest.raises(ValueError, match='changed after token'):
        verifier.raw_signature([('/original/datasets/dualtalk/train/a.wav',record['bytes'],12346)])


def test_identity_comparison_keeps_effective_element_counts():
    rows = [dict(dialogue='b', elements={'expression':50}, evaluated_chunks=1, valid_frames=1),
            dict(dialogue='a', elements={'expression':100}, evaluated_chunks=1, valid_frames=2)]
    assert identity(rows) == identity(list(reversed(rows)))
    changed = [dict(rows[0], elements={'expression':49}),rows[1]]
    assert identity(rows) != identity(changed)
