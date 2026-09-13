"""Create a local code-only archive; this command does not access any server."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    repo = Path(__file__).resolve().parents[1]
    output = repo / 'runs/avatar_stability_20260913_gpu23'
    output.mkdir(parents=True, exist_ok=True)
    files = ['wav2vec.py', 'emotion_ssm/train/generation_v3.py', 'emotion_ssm/train/generation_sampling.py',
             'emotion_ssm/train/generation_stability.py', 'emotion_ssm/train/generation_prefetch.py',
             'emotion_ssm/utils/reconstruction.py', 'scripts/run_avatar_stability.py',
             'tests/test_generation_stability.py']
    checked = ['DualTalk.py', 'emotion_ssm/config_v3.py', 'emotion_ssm/models/streaming_v3.py',
               'emotion_ssm/models/state_core.py', 'emotion_ssm/models/conditioned_dualtalk.py',
               'emotion_ssm/data/packets_v3.py']
    checks = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in checked}
    hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in files}
    selection = json.loads((repo / 'runs/avatar_old9750_20260911_gpu23/decline_probe_results.json').read_text(encoding='utf-8-sig'))['selection']
    with zipfile.ZipFile(output / 'code_overlay.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(repo / name, name)
        archive.writestr('deployment_manifest.json', json.dumps(dict(unchanged=checks, overlay=hashes), indent=2))
        archive.writestr('diagnostic_selection.json', json.dumps(selection, indent=2))
    (output / 'local_overlay_sha256.json').write_text(json.dumps(hashes, indent=2))
    print(output / 'code_overlay.zip')


if __name__ == '__main__':
    main()
