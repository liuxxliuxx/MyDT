from emotion_ssm.config import parse_config_args
from emotion_ssm.train.dynamics_trainer import run_dynamics_stage


def main() -> None:
    cfg, _ = parse_config_args("Phase B: directional dyadic coupling")
    if not cfg.TRAIN.PHASE_A_CHECKPOINT:
        raise ValueError("TRAIN.PHASE_A_CHECKPOINT is required for Phase B")
    run_dynamics_stage(
        cfg,
        stage_name="phase_b_coupling",
        required_checkpoint_name="phase_b_best.pt",
        enable_partner=True,
        use_counterfactual=True,
        previous_stage_checkpoint=cfg.TRAIN.PHASE_A_CHECKPOINT,
    )


if __name__ == "__main__":
    main()
