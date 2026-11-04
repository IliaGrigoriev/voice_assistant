import os


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

LIBRI_SPEECH = os.path.join(
    _VSA_FOLDER,
    "train-clean-100",
)

MEDDIALOG_AUDIO = os.path.join(
    _VSA_FOLDER,
    "meddialog-audio",
)
