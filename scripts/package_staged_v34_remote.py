"""Package only reviewed Python sources and tests; no data, checkpoints or credentials."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    project=Path(__file__).resolve().parents[1]
    output=project/'runs/v3_4_semantic_repair_20260911_gpu23';output.mkdir(parents=True,exist_ok=True)
    sources=sorted((project/'emotion_ssm').rglob('*.py'))
    sources += [project/'DualTalk.py',project/'wav2vec.py']
    sources += [project/'scripts'/name for name in ('compare_staged_v33_checkpoints.py','verify_staged_v34_ddp.py',
        'diagnose_staged_v34_origins.py','run_staged_v34_repair.py')]
    sources += [project/'tests'/name for name in ('test_staged_v34.py','test_staged_v33.py','test_staged_dynamics_v3.py',
        'test_v3_dynamics_training.py','test_v3_state_core.py')]
    manifest={}
    with zipfile.ZipFile(output/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            name=path.relative_to(project).as_posix();data=path.read_bytes()
            if '__pycache__' in name:continue
            compile(data,name,'exec')
            archive.writestr(name,data);manifest[name]=hashlib.sha256(data).hexdigest()
        archive.writestr('source_manifest.json',json.dumps(manifest,indent=2))
    (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(manifest),zip=str(output/'source.zip'),bytes=(output/'source.zip').stat().st_size)))


if __name__=='__main__':main()
