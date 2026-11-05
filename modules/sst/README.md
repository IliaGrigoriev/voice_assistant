# Speech-to-Text Experiments

This directory contains the character-level CTC speech-recognition pipeline.
It compares pretrained Wav2Vec2 variants with a Conformer trained from scratch,
using the same text normalization, decoder, data splits, and evaluation metrics.

The main research question is:

> How much does medical-domain fine-tuning improve recognition, and how does it
> compare with training a Conformer from scratch?

The literature basis and planned parameter-selection protocol are documented
separately in [PARAMETER_SELECTION.md](PARAMETER_SELECTION.md). That document
distinguishes the target recipe from settings that are already implemented.

The code implements the `v1` protocol described in that document. The
versioned output directory prevents checkpoints from older optimizer and
scheduler configurations from being resumed accidentally.

## Experiments

The experiment suite is defined by the `Experiments` class in
`experiments.py`. Each comparison has its own method.

| Value | Model | Training data | Test data | Purpose |
| --- | --- | --- | --- | --- |
| `original` | Original pretrained Wav2Vec2 with its pretrained CTC head | None | MedDialog and LibriSpeech | Measure medical performance before adaptation and establish the general-domain reference |
| `frozen` | Frozen Wav2Vec2 encoder with a new trainable CTC head | MedDialog | MedDialog and LibriSpeech | Measure low-cost medical adaptation and general-domain retention |
| `full` | Wav2Vec2 with a new head and conservative encoder adaptation | MedDialog | MedDialog and LibriSpeech | Produce the main medical model and quantify catastrophic forgetting |
| `conformer` | CNN-Conformer-CTC initialized from scratch | MedDialog | MedDialog only | Measure the value of Wav2Vec2 pretraining when target-domain data is held constant |
| `all` | All four experiments in the order above | As listed | As listed | Produce the complete comparison |

The original Wav2Vec2 baseline is evaluation-only, so its training time is
zero. The other three experiments train only on medical speech. LibriSpeech is
a retention benchmark for Wav2Vec2, not a headline benchmark for the Conformer.

### Experiment 1: original Wav2Vec2

This is the unchanged `WAV2VEC2_ASR_BASE_960H` model. Its MedDialog result
shows how well a general-domain ASR system transfers without medical training.
Its LibriSpeech result is the reference against which forgetting is measured.

### Experiment 2: frozen Wav2Vec2 plus a new head

The pretrained encoder is locked and a new character-level CTC projection is
trained on MedDialog. This isolates how much domain adaptation can be achieved
by remapping existing acoustic representations without modifying them.

### Why it is called `frozen`

In the `frozen` experiment, the pretrained Wav2Vec2 encoder parameters have
`requires_grad = False`, so backpropagation does not update them. The newly
added character-level CTC classifier head remains trainable. In other words,
`frozen` is shorthand for **frozen encoder plus a new trainable head**, not a
completely frozen model. This provides a low-cost comparison because it trains
far fewer parameters than full Wav2Vec2 fine-tuning.

### Experiment 3: fully fine-tuned Wav2Vec2

This is the main medical-domain model. The planned recipe first trains the new
CTC head, then updates the Transformer with a lower learning rate while keeping
the convolutional feature extractor frozen. MedDialog measures adaptation;
LibriSpeech measures the cost of that adaptation in general-domain accuracy.

### Experiment 4: Conformer from scratch

The Conformer starts with random weights and trains only on the same MedDialog
training split used by the Wav2Vec2 adaptations. It is evaluated only on
MedDialog. This makes the medical comparison meaningful: pretrained Wav2Vec2
versus an architecture trained from scratch with the same target-domain data.
It is excluded from the LibriSpeech ranking because Wav2Vec2 has already seen
substantially more LibriSpeech ASR training.

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
- Performance on each experiment's designated fixed test sets

WER and CER are corpus-level error rates. Greedy CTC decoding collapses
consecutive repeated tokens and removes the blank token.

## Code organization

| File | Class | Responsibility |
| --- | --- | --- |
| `main.py` | `Main` | Minimal executable entry point |
| `const.py` | `ExperimentConfig` | Paths, hyperparameters, and experiment selection |
| `PARAMETER_SELECTION.md` | — | Paper-backed parameter choices and search protocol |
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
7. Wav2Vec2 models are evaluated on MedDialog and LibriSpeech; the Conformer is
   evaluated on MedDialog only under the planned protocol.
8. Checkpoints and JSON results are written under the experiment output folder.

## Configuration

All settings are controlled in `const.py`; `main.py` has no command-line
training arguments.

```python
class ExperimentConfig:
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
    NUM_WORKERS = 16
    GRADIENT_CLIP = 1.0
    MIXED_PRECISION = True

    PROTOCOL_VERSION = "v1"
    OUTPUT_DIR = Path(_VSA_FOLDER) / "experiments" / PROTOCOL_VERSION
    RESUME_FROM = None
    AUTO_RESUME = True
    SKIP_COMPLETED_EXPERIMENTS = True
    RANDOM_SEED = 42

    VOICE_MANIFEST = None
    REQUIRE_VOICE_MANIFEST = False
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
With the default `REQUIRE_VOICE_MANIFEST = False`, the loader uses a stable
hash of each dialogue ID to create 80/10/10 train, validation, and test splits.
Both sides of a dialogue remain in the same split, preventing dialogue-text
leakage, but actual TTS voices may occur in multiple splits.

When `metadata.csv` and the clean-audio directory already exist locally, the
loader uses them directly instead of asking Hugging Face to scan all 21,069
files. It contacts Hugging Face only when those required paths are absent.
Constructed split objects are also cached at class level and reused by all
later experiments and evaluators in the same Python session.

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

When a manifest is supplied, the loader assigns complete voice identities to
training, validation, or test, so no voice occurs in more than one split. Set
`REQUIRE_VOICE_MANIFEST = True` to prohibit the dialogue-based fallback. Both
strategies are deterministic.

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

The original Wav2Vec2 bundle contains the equivalent labels—blank, word
separator, apostrophe, and A-Z—but stores them in frequency order rather than
alphabetical order. `OriginalWav2Vec` validates the label set and reorders its
logits into the shared project vocabulary before evaluation.

## Training behavior

The trainer uses:

- AdamW optimization
- per-update warmup, hold, and linear-decay scheduling
- configurable gradient accumulation
- configurable gradient clipping
- optional CUDA mixed precision
- CTC loss with zero-infinity handling
- waveform-length masking for both architectures
- deterministic epoch shuffling

The frozen Wav2Vec2 experiment passes only the new classifier parameters to the
optimizer. Full Wav2Vec2 first trains the head alone, then updates the
Transformer with a separate lower learning rate while keeping convolutional
features frozen. The Conformer initializes all parameters from scratch.

When mixed precision is enabled, model forward and backward operations use
AMP, while CTC log-probabilities and loss are explicitly calculated in FP32.
PyTorch does not provide a CUDA FP16 CTC-loss kernel, and the FP32 calculation
also improves numerical stability.

## Checkpoints and result files

The default layout is:

```text
/home/ilia/Data/vsa/experiments/v1/
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

The primary table compares all four models on MedDialog. The secondary
retention table contains only the three Wav2Vec2 configurations on LibriSpeech.
The improvement from `original_wav2vec` to `fully_finetuned_wav2vec` measures
medical adaptation, while their LibriSpeech difference measures catastrophic
forgetting. Comparing frozen and full Wav2Vec2 quantifies the value and cost of
updating pretrained representations. Comparing full Wav2Vec2 with the medical-
only Conformer measures the value of pretraining.

## Limitations

- The vocabulary supports lowercase English characters, spaces, and
  apostrophes only.
- Decoding is greedy and does not use a language model or beam search.
- A complete MedDialog voice manifest is required for truly voice-disjoint
  evaluation; the default fallback is dialogue-disjoint only.
- Checkpoint recovery occurs after completed optimizer steps, not inside a
  forward or backward operation.
- GPU memory is reported using PyTorch peak allocated memory, not total driver
  or system GPU usage.
