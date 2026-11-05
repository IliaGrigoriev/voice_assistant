# Summary

Summary is the semantic-clustering component of the voice assistant project.
It organizes the MedDialog-Audio transcriptions into groups of sentences with
similar meaning. These groups are intended to become the source material for
later summarization, labeling, retrieval, or other downstream processing.

## Processing pipeline

1. `MetadataReader` reads `filename` and `transcription` from the
   MedDialog-Audio `metadata.csv` file.
2. `TranscriptionEmbedder` converts every non-empty transcription into a
   normalized vector with `sentence-transformers/all-MiniLM-L6-v2`.
3. `TranscriptionClusterer` applies HDBSCAN directly to those vectors using
   cosine distance.
4. Any HDBSCAN outlier is assigned to the closest cluster centroid using
   cosine similarity, ensuring that every transcription belongs to a cluster.
5. The clusterer writes every cluster to its own CSV file.

No principal component analysis or other dimensionality reduction is used.
CUDA is selected automatically for embedding when PyTorch can access a GPU.

## Project files

- `const.py` contains paths, model selection, batch size, and clustering
  parameters.
- `data_reader.py` reads and validates the metadata.
- `embedder.py` creates the sentence vectors.
- `clustering.py` performs density-based clustering and writes the results.
- `main.py` connects the components and runs the pipeline.

## Input

The default metadata file is:

```text
/home/ilia/Data/vsa/meddialog-audio/metadata.csv
```

It must contain these columns:

- `filename`, used as the sentence ID;
- `transcription`, containing the sentence text.

Rows with empty transcriptions are ignored.

## Output

Each execution creates a timestamped directory under:

```text
/home/ilia/Data/vsa/Summary
```

Clusters are written as `cluster_0000.csv`, `cluster_0001.csv`, and so on.
There is no noise file because every sentence is assigned to a cluster.

Every CSV contains only:

```csv
sentence_id,cluster_id,transcription
```

## Configuration

Edit `const.py` to change the input path, output path, Sentence Transformer
model, embedding batch size, minimum cluster size, or minimum samples. The
pipeline intentionally has no command-line argument parser.

## Running

From the `modules` directory:

```bash
source .va/bin/activate
python summary/main.py
```

Required packages are included in `modules/reqs.txt`.
