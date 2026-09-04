from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import librosa
import numpy as np
import torch

from emotion_ssm.config import load_config
from emotion_ssm.data.dualtalk import normalize_waveform, speech_is_active
from emotion_ssm.train.dualtalk import _build_system
from emotion_ssm.train.dynamics_core import load_component_state
from emotion_ssm.utils.paths import ensure_output_directory


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Run emotion-conditioned DualTalk on one pair")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-audio", type=Path, required=True)
    parser.add_argument("--partner-audio", type=Path, required=True)
    parser.add_argument("--partner-flame", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="emotion_conditioned")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = load_config(args.config, args.opts)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    system = _build_system(cfg, device)
    system.load_state_dict(load_component_state(args.checkpoint, "system"), strict=True)
    system.eval()

    target_wave, _ = librosa.load(str(args.target_audio), sr=16000)
    partner_wave, _ = librosa.load(str(args.partner_audio), sr=16000)
    with np.load(str(args.partner_flame)) as partner_data:
        partner_exp = partner_data["exp"].copy()
        partner_pose = partner_data["pose"].copy()
        partner_shape = partner_data["shape"].copy() if "shape" in partner_data else None
    partner_blendshape = np.concatenate(
        [partner_exp, partner_pose[:, 3:], partner_pose[:, :3]], axis=-1
    )
    audio_length = min(len(target_wave), len(partner_wave))
    target = normalize_waveform(target_wave[:audio_length])[None].to(device)
    partner = normalize_waveform(partner_wave[:audio_length])[None].to(device)
    partner_bs = torch.from_numpy(partner_blendshape).float()[None].to(device)
    duration = torch.tensor([audio_length / 16000.0], device=device)
    target_active = torch.tensor(
        [speech_is_active(target_wave[:audio_length], cfg.DUALTALK.SPEECH_RMS_THRESHOLD)],
        device=device,
    )
    partner_active = torch.tensor(
        [speech_is_active(partner_wave[:audio_length], cfg.DUALTALK.SPEECH_RMS_THRESHOLD)],
        device=device,
    )
    context, state, evidence = system.conditioner(
        target,
        partner,
        duration,
        target_speech_active=target_active,
        partner_speech_active=partner_active,
    )
    generated = system.generator(target, partner, partner_bs, context, True)[0].cpu().numpy()

    output_dir = ensure_output_directory(
        args.output_dir,
        [
            cfg.DATA.ROOT,
            cfg.DATA.EMOTIONTALK_ROOT,
            cfg.DATA.IEMOCAP_RAW_ROOT,
            cfg.DATA.DUALTALK_ROOT,
        ],
    )
    blendshape_path = output_dir / f"{args.name}.npy"
    flame_path = output_dir / f"{args.name}.npz"
    state_path = output_dir / f"{args.name}_emotion.pt"
    np.save(blendshape_path, generated)
    pose = np.concatenate([generated[:, 53:56], generated[:, 50:53]], axis=-1)
    flame_payload = {"exp": generated[:, :50], "pose": pose}
    if partner_shape is not None:
        flame_payload["shape"] = partner_shape
    np.savez(flame_path, **flame_payload)
    torch.save(
        {
            "context": context.cpu(),
            "z": state.z.cpu(),
            "relation": state.relation.cpu(),
            "target_audio_aff": evidence["target_aff"].cpu(),
            "partner_audio_aff": evidence["partner_aff"].cpu(),
        },
        state_path,
    )

    render_path = output_dir / f"{args.name}.mp4"
    if cfg.DUALTALK.RENDER_COMMAND:
        command = cfg.DUALTALK.RENDER_COMMAND.format(
            blendshape=str(blendshape_path),
            flame=str(flame_path),
            audio=str(args.target_audio),
            output=str(render_path),
        )
        subprocess.run(shlex.split(command), check=True)
        if not render_path.exists():
            raise RuntimeError(f"Renderer completed but did not create {render_path}")
    summary = {
        "blendshape": str(blendshape_path),
        "flame": str(flame_path),
        "emotion_state": str(state_path),
        "render": str(render_path) if render_path.exists() else None,
    }
    (output_dir / f"{args.name}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
