import os
from pathlib import Path


_VSA_FOLDER = os.path.join("/home", "ilia", "Data", "vsa")

STT_MODEL = os.path.join(
    _VSA_FOLDER,
    "older",
    "stt_model_exp.pt",
)

WAV2VEC_MODEL = os.path.join(
    _VSA_FOLDER,
    "older",
    "wav2vec_model_exp.pt",
)

WAV2VEC_TRIAL_MODELS = os.path.join(
    _VSA_FOLDER,
    "wav2vec_trials",
)

OPTUNA_CHECKPOINTS = os.path.join(
    _VSA_FOLDER,
    "optuna_checkpoints",
)

LIBRI_SPEECH = os.path.join(
    _VSA_FOLDER,
    "train-clean-100",
)

MEDDIALOG_AUDIO = os.path.join(
    _VSA_FOLDER,
    "meddialog-audio",
)


class ExperimentConfig:
    """All settings used by the STT experiment runner."""

    # original, frozen, full, conformer, or all
    EXPERIMENT = "all"

    WAV2VEC_EPOCHS = 10
    CONFORMER_EPOCHS = 50
    BATCH_SIZE = 2
    FROZEN_HEAD_LEARNING_RATE = 1e-4
    FULL_HEAD_LEARNING_RATE = 1e-4
    FULL_TRANSFORMER_LEARNING_RATE = 1e-5
    CONFORMER_LEARNING_RATE = 3e-4
    WAV2VEC_WEIGHT_DECAY = 1e-4
    CONFORMER_WEIGHT_DECAY = 1e-3
    CONFORMER_DROPOUT = 0.1
    FROZEN_GRADIENT_ACCUMULATION = 8
    FULL_GRADIENT_ACCUMULATION = 16
    CONFORMER_GRADIENT_ACCUMULATION = 8
    HEAD_ONLY_EPOCHS = 1
    WAV2VEC_EARLY_STOPPING_PATIENCE = 3
    CONFORMER_EARLY_STOPPING_PATIENCE = 5
    LR_WARMUP_FRACTION = 0.10
    LR_HOLD_FRACTION = 0.40
    TIME_MASK_PROBABILITY = 0.05
    TIME_MASK_LENGTH = 10
    CHANNEL_MASK_PROBABILITY = 0.008
    CHANNEL_MASK_LENGTH = 64
    NUM_WORKERS = 16
    GRADIENT_CLIP = 1.0
    MIXED_PRECISION = True

    PROTOCOL_VERSION = "v1"
    OUTPUT_DIR = Path(_VSA_FOLDER) / "experiments" / PROTOCOL_VERSION
    AVAILABLE_MODELS_DIR = Path(_VSA_FOLDER) / "available_models"
    RESUME_FROM = None
    AUTO_RESUME = True
    SKIP_COMPLETED_EXPERIMENTS = True
    RANDOM_SEED = 42

    # Optional CSV containing filename and voice_id. Set REQUIRE_VOICE_MANIFEST
    # to reject the deterministic dialogue-disjoint fallback split.
    VOICE_MANIFEST = None
    REQUIRE_VOICE_MANIFEST = False
