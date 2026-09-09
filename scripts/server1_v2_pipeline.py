"""Server-1 deployment run: rebuild features, then train on physical GPU 2/3."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

GPU_UUIDS = ("GPU-5b497823-4a84-bde7-5670-2172ee96245d", "GPU-9117039b-5194-5d46-d4a7-088ff06ce552")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(GPU_UUIDS):
        raise RuntimeError("Launch with the explicitly approved physical GPU 2/3 UUIDs")
    stamp = root / "runs" / args.run_id
    stamp.mkdir(parents=True, exist_ok=True)
    artifacts = root / "artifacts" / args.run_id
    artifacts.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    def status(stage, **values):
        item = {"stage": stage, "updated": datetime.datetime.now().isoformat(), **values}
        (stamp/"status.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        print(json.dumps(item), flush=True)

    def run(command, name, extra_env=None):
        status(name, command=command)
        with (stamp/(name+".log")).open("a") as log:
            subprocess.run(command, env={**environment, **(extra_env or {})}, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    try:
        inputs = {"emotiontalk": root/"datasets/emotiontalk/processed", "iemocap": root/"artifacts/features/iemocap"}
        models = {"emotiontalk": ("/home/s21_yhr/lzh/Emotiontalk_work/models/chinese-hubert-base",
                                 "/home/s21_yhr/lzh/Emotiontalk_work/models/chinese-macbert-base"),
                  "iemocap": ("facebook/wav2vec2-base-960h", "roberta-base")}
        outputs = {name: artifacts/(name+"_v2") for name in inputs}
        commands = []
        for name, input_root in inputs.items():
            mapping = {}
            for labels_path in sorted((input_root/"dialogues").glob("*/labels.json")):
                dialogue = labels_path.parent.name
                for item in json.loads(labels_path.read_text())["utterances"]:
                    if name == "emotiontalk":
                        wav = Path("/home/s21_yhr/lzh/Emotiontalk_work/raw")/item["audio"]
                    else:
                        wav = (root/"datasets/iemocap/raw"/("Session"+str(int(dialogue[3:5]))) /
                               "sentences/wav"/dialogue/(item["utterance_id"]+".wav"))
                    if not wav.is_file():
                        raise FileNotFoundError(wav)
                    mapping[dialogue+"/"+item["utterance_id"]] = str(wav)
            manifest = artifacts/(name+"_audio_manifest.json")
            manifest.write_text(json.dumps(mapping, indent=2))
            commands.append([sys.executable, "-B", "-m", "emotion_ssm.preprocess.emotion_features",
                "--input-root", str(input_root), "--output-root", str(outputs[name]),
                "--audio-manifest", str(manifest), "--audio-model", models[name][0],
                "--text-model", models[name][1], "--dataset", name, "--device", "cuda:0"])
        status("preprocessing", physical_gpus=[2,3], feature_outputs={k:str(v) for k,v in outputs.items()},
               missing_iemocap_face_identity="visual masked; old AU tracks have no verified role identity")
        processes = []
        for (name, _), command, gpu in zip(inputs.items(), commands, GPU_UUIDS):
            log = (stamp/("preprocess_"+name+".log")).open("a")
            process = subprocess.Popen(command, env={**environment,"CUDA_VISIBLE_DEVICES":gpu},
                                       stdout=log, stderr=subprocess.STDOUT)
            log.close()
            processes.append((name, process))
        failures = [(name, code) for name, process in processes if (code := process.wait()) != 0]
        if failures:
            raise RuntimeError(f"Feature preparation failed: {failures}")
        # Start all four pilot controls after the newly trained observation/dynamics stages.
        run([sys.executable, "scripts/run_v2_experiments.py", "--source", "all", "--mode", "pilot",
             "--stages", "all", "--seeds", "6666", "--nproc", "2", "--device", "cuda:0",
             "--baseline", str(root/"model/dualtalk_baseline.pth"), "--data-root", str(root/"datasets/dualtalk"),
             "--emotiontalk-root", str(outputs["emotiontalk"]), "--iemocap-root", str(outputs["iemocap"]),
             "--output-root", str(stamp/"training"), "--artifacts-root", str(artifacts/"generation")], "training")
        status("completed", physical_gpus=[2,3], mode="1000-step pilot, all four controls")
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
