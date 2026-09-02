from emotion_ssm.config import parse_config_args
from emotion_ssm.train.dynamics_trainer import run_dynamics_stage


def main() -> None:
    cfg, _ = parse_config_args("Phase A: self dynamics and multi-horizon prediction")
    run_dynamics_stage(
        cfg,
        stage_name="phase_a_dynamics",
        required_checkpoint_name="phase_a_best.pt",
        enable_partner=False,
        use_counterfactual=False,
    )


if __name__ == "__main__":
    main()
