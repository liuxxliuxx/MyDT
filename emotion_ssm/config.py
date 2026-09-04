from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from yacs.config import CfgNode as CN


def get_cfg_defaults() -> CN:
    cfg = CN()

    cfg.SEED = 6666
    cfg.DEVICE = "cuda"
    cfg.DETERMINISTIC = False

    cfg.DATA = CN()
    cfg.DATA.ROOT = "datasets"
    cfg.DATA.EMOTIONTALK_ROOT = "datasets/emotiontalk/processed"
    cfg.DATA.IEMOCAP_RAW_ROOT = "datasets/iemocap/raw"
    cfg.DATA.IEMOCAP_FEATURE_ROOT = "artifacts/features/iemocap"
    cfg.DATA.DUALTALK_ROOT = "datasets/dualtalk"
    cfg.DATA.IEMOCAP_FOLD = 5
    cfg.DATA.WINDOW_LENGTH = 33
    cfg.DATA.WINDOW_STRIDE = 4
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = True
    cfg.DATA.DOMAIN_RATIO = [2, 1]
    cfg.DATA.SOURCES = ["emotiontalk", "iemocap"]

    cfg.PREPROCESS = CN()
    cfg.PREPROCESS.AUDIO_MODEL = "facebook/wav2vec2-base-960h"
    cfg.PREPROCESS.TEXT_MODEL = "roberta-base"
    cfg.PREPROCESS.LOCAL_FILES_ONLY = False
    cfg.PREPROCESS.OPENFACE_BIN = "FeatureExtraction"
    cfg.PREPROCESS.FFMPEG_BIN = "ffmpeg"
    cfg.PREPROCESS.DEVICE = "cuda:0"
    cfg.PREPROCESS.BATCH_SIZE = 32
    cfg.PREPROCESS.CONFIDENCE_THRESHOLD = 0.8
    cfg.PREPROCESS.SKIP_OPENFACE = False

    cfg.MODEL = CN()
    cfg.MODEL.AUDIO_DIM = 768
    cfg.MODEL.FACE_DIM = 35
    cfg.MODEL.TEXT_DIM = 768
    cfg.MODEL.MODEL_DIM = 256
    cfg.MODEL.OBSERVATION_DIM = 128
    cfg.MODEL.STATE_DIM = 128
    cfg.MODEL.RELATION_DIM = 64
    cfg.MODEL.INFLUENCE_DIM = 128
    cfg.MODEL.INFLUENCE_CHANNELS = 32
    cfg.MODEL.NUM_DOMAINS = 2
    cfg.MODEL.NUM_LAYERS = 2
    cfg.MODEL.NUM_HEADS = 4
    cfg.MODEL.DROPOUT = 0.1
    cfg.MODEL.NUM_TIMESCALES = 8
    cfg.MODEL.TAU_MIN = 0.5
    cfg.MODEL.TAU_MAX = 300.0
    cfg.MODEL.SPEAKER_DELTA_SCALE = 0.1

    cfg.LOSS = CN()
    cfg.LOSS.UNIFY = 1.0
    cfg.LOSS.SMOOTH_L1 = 1.0
    cfg.LOSS.EMOTION = 1.0
    cfg.LOSS.INTENSITY = 0.5
    cfg.LOSS.VAD = 0.5
    cfg.LOSS.RELIABILITY = 0.1
    cfg.LOSS.SPEAKER = 0.05
    cfg.LOSS.DOMAIN = 0.05
    cfg.LOSS.DECORRELATION = 0.01
    cfg.LOSS.NEXT_STATE = 1.0
    cfg.LOSS.TRAJECTORY = 0.5
    cfg.LOSS.CORRECTION = 0.05
    cfg.LOSS.COUNTERFACTUAL = 0.2
    cfg.LOSS.COUNTERFACTUAL_MARGIN = 0.2
    cfg.LOSS.GENERATION_STATE = 0.1

    cfg.TRAIN = CN()
    cfg.TRAIN.OUTPUT_ROOT = "runs"
    cfg.TRAIN.EXPERIMENT_NAME = "experiment"
    cfg.TRAIN.BATCH_SIZE = 128
    cfg.TRAIN.SEQUENCE_BATCH_SIZE = 8
    cfg.TRAIN.EPOCHS = 50
    cfg.TRAIN.STOP_AFTER_EPOCHS = 0
    cfg.TRAIN.LR = 3e-4
    cfg.TRAIN.STATE_LR_SCALE = 0.1
    cfg.TRAIN.FINETUNE_OBSERVATION = False
    cfg.TRAIN.WEIGHT_DECAY = 1e-4
    cfg.TRAIN.GRAD_CLIP = 5.0
    cfg.TRAIN.GRAD_ACCUMULATION = 1
    cfg.TRAIN.EMA_DECAY = 0.996
    cfg.TRAIN.GRL_WARMUP_EPOCHS = 10.0
    cfg.TRAIN.AMP = True
    cfg.TRAIN.LOG_INTERVAL = 20
    cfg.TRAIN.SAVE_EVERY = 1
    cfg.TRAIN.RESUME = ""
    cfg.TRAIN.OBSERVATION_CHECKPOINT = ""
    cfg.TRAIN.EMA_TEACHER_CHECKPOINT = ""
    cfg.TRAIN.EMOTION_HEADS_CHECKPOINT = ""
    cfg.TRAIN.PHASE_A_CHECKPOINT = ""
    cfg.TRAIN.PHASE_B_CHECKPOINT = ""
    cfg.TRAIN.DRY_RUN = False
    cfg.TRAIN.DRY_RUN_TRAIN_BATCHES = 2
    cfg.TRAIN.DRY_RUN_VAL_BATCHES = 1

    cfg.DYNAMICS = CN()
    cfg.DYNAMICS.HORIZONS = [1, 2, 4, 8, 16, 32]
    cfg.DYNAMICS.CORRECTION_MODE = "teacher_forced"
    cfg.DYNAMICS.ENABLE_PARTNER = True
    cfg.DYNAMICS.FIXED_RELATION = False
    cfg.DYNAMICS.SYMMETRIC_COUPLING = False
    cfg.DYNAMICS.DISABLE_LONG_TIMESCALES = False
    cfg.DYNAMICS.RANDOM_ROLE_SWAP = True
    cfg.DYNAMICS.RANDOM_PARTNER = False

    cfg.COUNTERFACTUAL = CN()
    cfg.COUNTERFACTUAL.TOP_K = 8
    cfg.COUNTERFACTUAL.INTENSITY_TOLERANCE = 0.2
    cfg.COUNTERFACTUAL.TURN_TOLERANCE = 0.15
    cfg.COUNTERFACTUAL.MAX_ACTION_COSINE = 0.8

    cfg.DUALTALK = CN()
    cfg.DUALTALK.BASELINE_CHECKPOINT = ""
    cfg.DUALTALK.PHASE_B_CHECKPOINT = ""
    cfg.DUALTALK.FEATURE_DIM = 256
    cfg.DUALTALK.BLENDSHAPE_DIM = 56
    cfg.DUALTALK.FILM_SCALE = 0.1
    cfg.DUALTALK.FREEZE_STATE_EPOCHS = 10
    cfg.DUALTALK.FPS = 25
    cfg.DUALTALK.CHUNK_FRAMES = 200
    cfg.DUALTALK.AUDIO_MODEL = "facebook/wav2vec2-base-960h"
    cfg.DUALTALK.LOCAL_FILES_ONLY = False
    cfg.DUALTALK.JOINT_FINETUNE_LR_SCALE = 0.1
    cfg.DUALTALK.RENDER_COMMAND = ""

    return cfg


def load_config(path: Path, overrides: Optional[list] = None) -> CN:
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(path))
    if overrides:
        cfg.merge_from_list(overrides)
    cfg.freeze()
    return cfg


def parse_config_args(description: str) -> Tuple[CN, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="YACS overrides, for example TRAIN.BATCH_SIZE 64",
    )
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(args.config))
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.defrost()
    if args.resume is not None:
        cfg.TRAIN.RESUME = str(args.resume)
    if args.dry_run:
        cfg.TRAIN.DRY_RUN = True
    cfg.freeze()
    return cfg, args
