"""Configuration for transcription clustering."""

from pathlib import Path

VSA_ROOT = Path("/home/ilia/Data/vsa")
METADATA_PATH = VSA_ROOT / "meddialog-audio" / "metadata.csv"
OUTPUT_ROOT = VSA_ROOT / "Summary"
MODEL_CACHE = OUTPUT_ROOT / "model_cache"

SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 128
MIN_CLUSTER_SIZE = 15
MIN_SAMPLES = 5
