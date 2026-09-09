"""Stream raw A/A/FLAME and available external text from a full checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from emotion_ssm.data.dualtalk import speech_is_active
from emotion_ssm.data.timed_dualtalk import load_flame
from emotion_ssm.utils.generation_checkpoint import load_generation
from emotion_ssm.utils.paths import ensure_output_directory


@torch.no_grad()
def run(checkpoint, target_audio, partner_audio, partner_flame, output_dir, words=None,
        device="cpu", session_id="demo", roles=("avatar", "user"), **unused):
    import librosa
    model, cfg, metadata = load_generation(checkpoint, device)
    model.eval()
    output_dir = ensure_output_directory(output_dir, [Path(p).parent for p in (target_audio, partner_audio, partner_flame)])
    waves = [torch.from_numpy(librosa.load(str(p), sr=16000)[0])[None].to(device)
             for p in (target_audio, partner_audio)]
    visual = load_flame(partner_flame).to(device)
    frames = min(len(visual), min(v.shape[1] for v in waves) // 640)
    available = json.loads(Path(words).read_text(encoding="utf-8")) if words else []
    state, outputs, diagnostics = None, [], []
    for start in range(0, frames, 25):
        stop = min(start+25, frames)
        a, b = start*640, stop*640
        audio = [v[:, a:b] for v in waves]
        valid = torch.isfinite(visual[start:stop]).all(-1)[None]
        packet = {"session_id": session_id, "roles": tuple(roles), "time": stop/25,
                  "target_audio": audio[0], "partner_audio": audio[1],
                  "target_speech_active": speech_is_active(audio[0].cpu().numpy(), cfg.DUALTALK.SPEECH_RMS_THRESHOLD),
                  "partner_speech_active": speech_is_active(audio[1].cpu().numpy(), cfg.DUALTALK.SPEECH_RMS_THRESHOLD),
                  "partner_blendshape": torch.nan_to_num(visual[start:stop])[None],
                  "partner_visual_mask": valid, "words": [w for w in available if float(w["available_at"]) <= stop/25]}
        generated, state, observed = model(packet, state)
        outputs.append(generated[0].cpu())
        diagnostics.append({k: v.cpu().tolist() if torch.is_tensor(v) else v for k, v in observed.items()})
    if not outputs:
        raise ValueError("Inputs contain no complete 25-FPS frame")
    values = torch.cat(outputs).numpy()
    np.savez(output_dir / "generated.npz", exp=values[:, :50], pose=np.concatenate([values[:, 53:56], values[:, 50:53]], -1))
    torch.save(state.detach(), output_dir / "stream_state.pt")
    (output_dir / "observations.json").write_text(json.dumps({"protocol": "external_asr" if words else "text_missing",
        "checkpoint_protocol": metadata["protocol"], "blocks": diagnostics}, indent=2), encoding="utf-8")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "target-audio", "partner-audio", "partner-flame", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--words", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--session-id", default="demo")
    parser.add_argument("--roles", nargs=2, default=["avatar", "user"])
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
