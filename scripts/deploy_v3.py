"""Build/install a reviewed code-only archive, keeping server data untouched."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def allowed(name):
    parts = PurePosixPath(name).parts
    if not parts or any(p in ("..", ".git", "__pycache__") or p.startswith(".codex") for p in parts):
        return False
    if name in ("DualTalk.py", "wav2vec.py", "requirements-test.txt", "requirements-v2.txt", "README.md", "emotion_ssm/README.md"):
        return True
    if parts[0] in ("emotion_ssm", "tests") and name.endswith(".py"):
        return True
    if parts[0] == "scripts" and ("v3" in parts[-1]) and name.endswith(".py"):
        return True
    return name in ("docs/v3_training.md", "docs/v32_adaptive_dynamics_2026-09-09.md",
                    "docs/server3_v32_run_2026-09-09.md")


def build(root, archive):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    candidates = [root / name for name in ("DualTalk.py", "wav2vec.py", "requirements-test.txt", "requirements-v2.txt",
                                         "README.md", "emotion_ssm/README.md", "docs/v3_training.md",
                                         "docs/v32_adaptive_dynamics_2026-09-09.md",
                                         "docs/server3_v32_run_2026-09-09.md")]
    for folder in ("emotion_ssm", "tests", "scripts"):
        candidates.extend((root / folder).rglob("*.py"))
    paths = sorted({path for path in candidates if path.is_file() and allowed(path.relative_to(root).as_posix())})
    manifest = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, path.relative_to(root).as_posix())
        bundle.writestr("deployment_manifest.json", json.dumps(manifest, indent=2))
    print(json.dumps({"archive": str(archive), "files": len(paths), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}))


def install(root, archive, run_id):
    root = Path(root).resolve()
    if Path(run_id).name != run_id or not run_id.startswith("v3_"):
        raise ValueError("Invalid deployment run identifier")
    backup = root / "runs" / run_id / "deployment_backup"
    changed = []
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("deployment_manifest.json"))
        if set(bundle.namelist()) != set(manifest) | {"deployment_manifest.json"}:
            raise ValueError("Archive inventory differs from its manifest")
        contents = {}
        # Verify every member before mutating any server file.
        for name, digest in manifest.items():
            target = (root / name).resolve()
            if not allowed(name) or root not in target.parents:
                raise ValueError(f"Refusing unexpected server path: {name}")
            value = bundle.read(name)
            if hashlib.sha256(value).hexdigest() != digest:
                raise ValueError(f"Bundle checksum mismatch: {name}")
            contents[name] = value
        for name, value in contents.items():
            target = root / name
            if target.exists() and target.read_bytes() == value:
                continue
            if target.exists():
                previous = backup / name
                if previous.exists():
                    raise ValueError(f"Backup already exists; use a new deployment id: {previous}")
                previous.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, previous)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".v3-upload")
            temporary.write_bytes(value)
            temporary.replace(target)
            changed.append(name)
    audit = root / "runs" / run_id / "deployment.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps({"manifest": manifest, "changed": changed, "backup": str(backup)}, indent=2))
    print(json.dumps({"installed_files": len(changed), "backup": str(backup), "audit": str(audit)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "install"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.mode == "build":
        build(args.root, args.archive)
    else:
        install(args.root, args.archive, args.run_id)


if __name__ == "__main__":
    main()
