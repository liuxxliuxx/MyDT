"""Baseline control using the same v2 stream, sampler and optimizer as all controls."""
from emotion_ssm.config import parse_config_args
from emotion_ssm.train.generation import run_generation


def main():
    cfg, _ = parse_config_args("Matched-budget DualTalk baseline control")
    if cfg.DUALTALK.PROTOCOL_VERSION != 2:
        raise ValueError("Legacy baseline training is disabled; evaluate archived checkpoints explicitly")
    cfg = cfg.clone()
    cfg.defrost()
    cfg.DUALTALK.VARIANT = "none"
    cfg.freeze()
    return run_generation(cfg)


if __name__ == "__main__":
    main()
