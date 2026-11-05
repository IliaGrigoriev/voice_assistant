"""Create vector representations of transcriptions."""

from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class TranscriptionEmbedder:
    def __init__(self, model_name: str, cache_folder: Path, batch_size: int) -> None:
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def encode(self, texts: list[str]) -> np.ndarray:
        print(f"Embedding {len(texts):,} transcriptions on {self.device}")
        model = SentenceTransformer(
            self.model_name,
            device=self.device,
            cache_folder=str(self.cache_folder),
        )
        return np.asarray(model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ), dtype=np.float32)
