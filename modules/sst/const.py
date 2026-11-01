import os


_VSA_FOLDER = os.path.join("/home", "ilia", "Data", "vsa")

STT_MODEL = os.path.join(
    _VSA_FOLDER,
    "stt_model.pt",
)

LIBRI_SPEECH = os.path.join(
    _VSA_FOLDER,
    "train-clean-100",
)
