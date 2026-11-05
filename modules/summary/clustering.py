"""Cluster transcription vectors and write one CSV per cluster."""

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN

from data_reader import Transcription


class TranscriptionClusterer:
    def __init__(self, output_root: Path, min_cluster_size: int,
                 min_samples: int) -> None:
        self.output_root = output_root
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        labels = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
            cluster_selection_method="leaf",
            copy=True,
        ).fit_predict(embeddings)
        return self._assign_unclustered(embeddings, labels)

    @staticmethod
    def _assign_unclustered(
        embeddings: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Assign every HDBSCAN outlier to the nearest cluster centroid."""
        labels = labels.copy()
        cluster_ids = sorted(set(int(label) for label in labels if label != -1))
        if not cluster_ids:
            return np.zeros(len(labels), dtype=int)

        centroids = np.stack([
            embeddings[labels == cluster_id].mean(axis=0)
            for cluster_id in cluster_ids
        ])
        centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.maximum(centroid_norms, 1e-12)

        unclustered = np.flatnonzero(labels == -1)
        if unclustered.size:
            vectors = embeddings[unclustered]
            vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(vector_norms, 1e-12)
            nearest = np.argmax(vectors @ centroids.T, axis=1)
            labels[unclustered] = np.asarray(cluster_ids)[nearest]
        return labels

    def write(self, transcriptions: list[Transcription],
              labels: np.ndarray) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = self.output_root / timestamp
        suffix = 1
        while output_dir.exists():
            output_dir = self.output_root / f"{timestamp}_{suffix:02d}"
            suffix += 1
        output_dir.mkdir(parents=True)

        for cluster_id in sorted(set(int(label) for label in labels)):
            filename = f"cluster_{cluster_id:04d}.csv"
            with (output_dir / filename).open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["sentence_id", "cluster_id", "transcription"],
                )
                writer.writeheader()
                for transcription, label in zip(transcriptions, labels):
                    if int(label) == cluster_id:
                        writer.writerow({
                            "sentence_id": transcription.sentence_id,
                            "cluster_id": cluster_id,
                            "transcription": transcription.text,
                        })
        return output_dir
