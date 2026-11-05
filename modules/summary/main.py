"""Run transcription embedding and clustering."""

from clustering import TranscriptionClusterer
from const import (
    EMBEDDING_BATCH_SIZE,
    METADATA_PATH,
    MIN_CLUSTER_SIZE,
    MIN_SAMPLES,
    MODEL_CACHE,
    OUTPUT_ROOT,
    SENTENCE_MODEL,
)
from data_reader import MetadataReader
from embedder import TranscriptionEmbedder


def main() -> None:
    transcriptions = MetadataReader(METADATA_PATH).read()
    embeddings = TranscriptionEmbedder(
        SENTENCE_MODEL, MODEL_CACHE, EMBEDDING_BATCH_SIZE
    ).encode([transcription.text for transcription in transcriptions])

    clusterer = TranscriptionClusterer(
        OUTPUT_ROOT,
        MIN_CLUSTER_SIZE,
        MIN_SAMPLES,
    )
    labels = clusterer.cluster(embeddings)
    output_dir = clusterer.write(transcriptions, labels)
    cluster_count = len(set(int(label) for label in labels if label != -1))
    print(f"Wrote {cluster_count} clusters to {output_dir}")


if __name__ == "__main__":
    main()
