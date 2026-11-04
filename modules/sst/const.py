import os
from pathlib import Path


_VSA_FOLDER = os.path.join("/home", "ilia", "Data", "vsa")

STT_MODEL = os.path.join(
    _VSA_FOLDER,
    "stt_model_exp.pt",
)

WAV2VEC_MODEL = os.path.join(
    _VSA_FOLDER,
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

    EPOCHS = 10
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 16
    GRADIENT_CLIP = 1.0
    MIXED_PRECISION = True

    OUTPUT_DIR = Path(_VSA_FOLDER) / "experiments"
    RESUME_FROM = None
    AUTO_RESUME = True
    SKIP_COMPLETED_EXPERIMENTS = True
    RANDOM_SEED = 42

    # CSV containing filename and voice_id. Required for MedDialog because
    # its published metadata does not expose the assigned TTS voice.
    VOICE_MANIFEST = None
