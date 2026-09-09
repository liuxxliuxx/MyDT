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
    if cfg.DUALTALK.PROTOCOL_VERSION == 2:
        from emotion_ssm.infer.streaming_demo import run
        return run(args.checkpoint, args.target_audio, args.partner_audio, args.partner_flame, args.output_dir, device=device)
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
    chunk_frames = int(cfg.DUALTALK.CHUNK_FRAMES)
    chunk_samples = int(round(chunk_frames / cfg.DUALTALK.FPS * 16000))
    num_chunks = min(
        len(target_wave) // chunk_samples,
        len(partner_wave) // chunk_samples,
        len(partner_blendshape) // chunk_frames,
    )
    if num_chunks < 1:
        raise ValueError("Inputs are shorter than one configured DualTalk chunk")

    state = None
    generated_chunks = []
    contexts = []
    state_values = []
    relation_values = []
    conditioning_values = []
    activity_values = []
    for chunk_index in range(num_chunks):
        sample_start = chunk_index * chunk_samples
        sample_stop = sample_start + chunk_samples
        frame_start = chunk_index * chunk_frames
        frame_stop = frame_start + chunk_frames
        target_raw = target_wave[sample_start:sample_stop]
        partner_raw = partner_wave[sample_start:sample_stop]
        target = normalize_waveform(target_raw)[None].to(device)
        partner = normalize_waveform(partner_raw)[None].to(device)
        partner_bs = torch.from_numpy(
            partner_blendshape[frame_start:frame_stop]
        ).float()[None].to(device)
        duration = torch.tensor([chunk_samples / 16000.0], device=device)
        target_active = torch.tensor(
            [speech_is_active(target_raw, cfg.DUALTALK.SPEECH_RMS_THRESHOLD)],
            device=device,
        )
        partner_active = torch.tensor(
            [speech_is_active(partner_raw, cfg.DUALTALK.SPEECH_RMS_THRESHOLD)],
            device=device,
        )
        context, state, conditioning_aff = system._condition_chunk(
            {
                "target_audio": target,
                "partner_audio": partner,
                "dt": duration,
                "target_speech_active": target_active,
                "partner_speech_active": partner_active,
            },
            state,
        )
        generated_chunk = system.generator(
            target, partner, partner_bs, context, True
        )[0]
        generated_chunks.append(generated_chunk.cpu())
        contexts.append(context[0].cpu())
        state_values.append(state.z[0].cpu())
        relation_values.append(state.relation[0].cpu())
        conditioning_values.append(conditioning_aff[0].cpu())
        activity_values.append(
            torch.stack([target_active[0], partner_active[0]]).cpu()
        )
    generated = torch.cat(generated_chunks, dim=0).numpy()

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
            "context": torch.stack(contexts),
            "z": torch.stack(state_values),
            "relation": torch.stack(relation_values),
            "conditioning_aff": torch.stack(conditioning_values),
            "speech_active": torch.stack(activity_values),
            "chunk_frames": chunk_frames,
            "chunk_seconds": chunk_samples / 16000.0,
            "final_target_state_aff": system.conditioner.state_model.state_to_aff(
                state.z[:, 0]
            ).cpu(),
            "final_partner_state_aff": system.conditioner.state_model.state_to_aff(
                state.z[:, 1]
            ).cpu(),
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
