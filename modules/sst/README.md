# Speech-to-Text Experiments

This directory contains the character-level CTC speech-recognition pipeline.
It compares pretrained Wav2Vec2 variants with a Conformer trained from scratch,
using the same text normalization, decoder, data splits, and evaluation metrics.

The main research question is:

> How much does medical-domain fine-tuning improve recognition, and how does it
> compare with training a Conformer from scratch?

## Experiments

The experiment suite is defined by the `Experiments` class in
`experiments.py`. Each comparison has its own method.

| Configuration value | Experiment method | Model | Purpose |
| --- | --- | --- | --- |
| `original` | `original_wav2vec_experiment()` | Original pretrained Wav2Vec2 and original CTC head | Baseline without medical fine-tuning |
| `frozen` | `frozen_wav2vec_experiment()` | Frozen Wav2Vec2 encoder and new trainable CTC head | Low-cost medical fine-tuning |
| `full` | `fully_finetuned_wav2vec_experiment()` | Fully trainable Wav2Vec2 and new CTC head | Main medical-domain model |
| `conformer` | `conformer_from_scratch_experiment()` | CNN and Conformer initialized from scratch | Pretraining comparison |
| `all` | All four methods in order | Complete comparison suite | Produce the final comparison data |

The original Wav2Vec2 baseline is evaluation-only, so its training time is
zero. The other three experiments train on medical speech and are evaluated on
both medical speech and LibriSpeech.

### Why it is called `frozen`

In the `frozen` experiment, the pretrained Wav2Vec2 encoder parameters have
`requires_grad = False`, so backpropagation does not update them. The newly
added character-level CTC classifier head remains trainable. In other words,
`frozen` is shorthand for **frozen encoder plus a new trainable head**, not a
completely frozen model. This provides a low-cost comparison because it trains
far fewer parameters than full Wav2Vec2 fine-tuning.

## Measurements

Every experiment reports:

- CTC loss
- Word error rate (WER)
- Character error rate (CER)
- Training time
- Peak training and evaluation GPU allocation
- Inference time and milliseconds per utterance
- Total and trainable parameter counts
- Model size in bytes
- Performance on the fixed MedDialog and LibriSpeech test sets

WER and CER are corpus-level error rates. Greedy CTC decoding collapses
consecutive repeated tokens and removes the blank token.

## Code organization

| File | Class | Responsibility |
| --- | --- | --- |
| `main.py` | `Main` | Minimal executable entry point |
| `const.py` | `ExperimentConfig` | Paths, hyperparameters, and experiment selection |
| `experiment_runner.py` | `ExperimentRunner` | Select, skip, stop, and resume experiments |
| `experiments.py` | `Experiments` | Construct models and implement the four experiments |
| `evaluator.py` | `ExperimentEvaluator` | Cross-domain evaluation and result serialization |
| `trainer.py` | `Trainer` | CTC training, validation, metrics, and checkpoints |
| `wav2vec.py` | `Wav2Vec`, `OriginalWav2Vec` | Pretrained baseline and fine-tuning model |
| `custom_conformer.py` | `CustomConformer` | Log-Mel, CNN, Conformer, and CTC output model |
| `text.py` | `TextProcessor` | Shared normalization, CTC decoding, WER, and CER |
| `data/dataloader.py` | `Dataloader` | LibriSpeech and MedDialog loading and collation |
| `hypo_tune.py` | `HypoTune`, `Wav2VecHypoTune` | Resumable Optuna studies |
| `stt_model.py` | `STTModel` | Common model interface |

## Execution flow

Running `main.py` follows this sequence:

1. `Main` creates an `ExperimentRunner` using `ExperimentConfig`.
2. `ExperimentRunner` selects one experiment or the complete suite.
3. Completed experiments are skipped when configured to do so.
4. `Experiments` constructs the model, optimizer, scheduler, and `Trainer`.
5. The model trains on the MedDialog training split when training is required.
6. Validation loss, WER, and CER are calculated after every epoch.
7. The final model is evaluated on both MedDialog and LibriSpeech test data.
8. Checkpoints and JSON results are written under the experiment output folder.

## Configuration

All settings are controlled in `const.py`; `main.py` has no command-line
training arguments.

```python
class ExperimentConfig:
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

    VOICE_MANIFEST = None
```

`EXPERIMENT` accepts `original`, `frozen`, `full`, `conformer`, or `all`.

Set `RESUME_FROM` only when explicitly resuming one experiment from a specific
checkpoint. When `AUTO_RESUME` is enabled, the runner normally discovers
`interrupted.pt` without requiring this setting.

## Running the suite

Run from the `modules` directory because the project currently uses imports and
relative storage paths based on that working directory.

```bash
cd /home/ilia/Code/voice_assistant/modules
source .va/bin/activate
python3 sst/main.py
```

The selected model may download pretrained TorchAudio weights the first time it
is constructed. LibriSpeech test data is downloaded when it is not already
available.

## Datasets and leak-free splits

### LibriSpeech

The loader uses the standard dataset divisions:

- Training: `train-clean-100`
- Validation: `dev-clean`
- Test: `test-clean`

These fixed divisions make evaluation reproducible.

### MedDialog-Audio

The experiments use the clean, 16 kHz MedDialog recordings. The upstream CSV
does not identify which of the eight TTS voices generated each recording.
Therefore, the loader deliberately refuses to guess voice identity from a
filename, speaker role, or acoustic feature.

Provide a CSV that covers every clean recording:

```csv
filename,voice_id
00001_1o00.wav,tara
00001_2o00.wav,leo
```

Then set:

```python
VOICE_MANIFEST = Path("/path/to/meddialog_voice_manifest.csv")
```

The loader assigns complete voice identities to training, validation, or test.
No voice can occur in more than one split. The sorted group assignment and
fixed seed make the split reproducible.

## Shared normalization and decoding

All models use the vocabulary:

```text
<blank>, space, apostrophe, a-z
```

`TextProcessor` is the single source of truth for training targets and
evaluation references. It:

1. converts text to lowercase;
2. collapses whitespace;
3. removes characters outside the shared vocabulary;
4. greedily decodes CTC predictions;
5. calculates edit distances for WER and CER.

The original Wav2Vec2 bundle uses the equivalent label order: blank, word
separator, apostrophe, and A-Z. `OriginalWav2Vec` validates this order before
using its pretrained output head.

## Training behavior

The trainer uses:

- AdamW optimization
- cosine-annealing learning-rate scheduling
- configurable gradient clipping
- optional CUDA mixed precision
- CTC loss with zero-infinity handling
- waveform-length masking for both architectures
- deterministic epoch shuffling

The frozen Wav2Vec2 experiment passes only the new classifier parameters to the
optimizer. Full Wav2Vec2 fine-tuning updates the encoder and classifier. The
Conformer initializes all parameters from scratch.

## Checkpoints and result files

The default layout is:

```text
/home/ilia/Data/vsa/experiments/
├── original_wav2vec/
│   └── results.json
├── frozen_wav2vec/
│   ├── epoch_0001.pt
│   ├── epoch_0001.metrics.json
│   ├── ...
│   └── results.json
├── fully_finetuned_wav2vec/
├── conformer_from_scratch/
├── results.json
└── suite_state.json
```

Every epoch checkpoint contains:

- model state
- optimizer state
- learning-rate scheduler state
- mixed-precision scaler state
- epoch and batch recovery positions
- accumulated training time
- validation metrics
- parameter counts

The adjacent `.metrics.json` file contains the serializable metadata without
the large tensor state dictionaries. Per-experiment `results.json` files
contain the final cross-domain comparison metrics. The root `results.json`
contains the complete suite.

## Stopping and resuming

Press `Ctrl+C` or send `SIGTERM` to stop safely. The active trainer writes:

```text
<experiment>/interrupted.pt
```

This checkpoint stores the next epoch and batch, partial epoch loss, model,
optimizer, scheduler, and mixed-precision state. On the next invocation:

- `AUTO_RESUME = True` loads `interrupted.pt` automatically;
- deterministic shuffling reconstructs the same batch order;
- already processed batches are skipped;
- completed experiments with `results.json` are skipped when
  `SKIP_COMPLETED_EXPERIMENTS = True`.

After an experiment finishes successfully, its stale interruption checkpoint
is removed. `suite_state.json` records whether the suite completed or stopped.

Only force termination such as `SIGKILL`, power loss during a checkpoint write,
or filesystem failure can prevent the latest state from being saved. The last
completed epoch checkpoint remains available in those cases.

## Optuna recovery

`HypoTune` and `Wav2VecHypoTune` store study state in SQLite and training state
under `OPTUNA_CHECKPOINTS`. If a trial is interrupted:

1. `Trainer` writes its `interrupted.pt` checkpoint;
2. the Optuna trial records the checkpoint path in its user attributes;
3. reopening the study re-enqueues the interrupted parameter set;
4. the new trial resumes its model and optimizer from the saved batch.

Completed and pruned trials count toward the configured total trial budget.
Failed or interrupted trials do not consume that budget. Out-of-memory trials
are pruned so the study can continue.

## Interpreting the comparison

Use the domain-prefixed values in each result:

```text
medical_wer
medical_cer
medical_loss
librispeech_wer
librispeech_cer
librispeech_loss
```

The main comparison is the improvement from `original_wav2vec` to
`fully_finetuned_wav2vec` on medical speech. LibriSpeech metrics show whether
medical fine-tuning reduces general-domain performance. Compare training time,
memory, and accuracy against `frozen_wav2vec` to quantify the cost of full
fine-tuning, and against `conformer_from_scratch` to measure the value of
pretraining.

## Limitations

- The vocabulary supports lowercase English characters, spaces, and
  apostrophes only.
- Decoding is greedy and does not use a language model or beam search.
- A complete MedDialog voice manifest is required for leak-free evaluation.
- Checkpoint recovery occurs after completed optimizer steps, not inside a
  forward or backward operation.
- GPU memory is reported using PyTorch peak allocated memory, not total driver
  or system GPU usage.
