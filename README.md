# Voice Assistant

An experimental local voice assistant that combines speech recognition with
natural-language intent parsing. The project can train or fine-tune two
character-level CTC speech models, transcribe audio through a FastAPI service,
and visualize semantic action graphs in a React frontend.

## Features

- Custom CNN + Conformer speech-to-text model
- Fine-tuning based on `WAV2VEC2_ASR_BASE_960H`
- Optuna hyperparameter studies stored in SQLite
- Resumable total trial budgets after interrupted runs
- LibriSpeech and MedDialog-Audio data loaders
- Per-epoch resumable checkpoints with sidecar metric files
- WER/CER evaluation on fixed dialogue- or voice-disjoint test data
- Classifier-only, full Wav2Vec2, and from-scratch Conformer modes
- Local FastAPI endpoints for transcription and intent recognition
- React interface for recording audio, uploading audio, and selecting an exact
  `.pt` model checkpoint
- AMR and spaCy-based semantic action graph generation

## Project structure

```text
voice_assistant/
├── frontend/                       React and Vite frontend
├── modules/
│   ├── comms_main.py               FastAPI service
│   ├── actions_from_intent/        Intent and action-graph pipeline
│   ├── sst/
│   │   ├── README.md               STT experiment and recovery guide
│   │   ├── data/dataloader.py      LibriSpeech and MedDialog loaders
│   │   ├── stt_model.py            Shared STT interface
│   │   ├── custom_conformer.py     Custom Conformer implementation
│   │   ├── wav2vec.py              Wav2Vec2 implementation
│   │   ├── trainer.py              CTC training loop
│   │   ├── experiments.py          Four experiment definitions
│   │   ├── evaluator.py            Cross-domain metrics and result files
│   │   ├── experiment_runner.py    Selection, stopping, and suite recovery
│   │   ├── hypo_tune.py            Optuna objectives and study handling
│   │   ├── main.py                 Training entry point
│   │   └── const.py                Dataset and checkpoint paths
│   ├── reqs.txt                    Python dependencies
│   └── stt_optuna.db               Optuna study database
└── README.md
```

## Requirements

- Linux
- Python 3 with a PyTorch-compatible version
- Node.js and npm
- FFmpeg-compatible audio support through TorchAudio
- An NVIDIA GPU is strongly recommended for training Wav2Vec2
- Enough disk space for the chosen speech dataset and checkpoints

The current paths are configured for this machine under:

```text
/home/ilia/Data/vsa
```

To use another location, update `_VSA_FOLDER` in
`modules/sst/const.py` before downloading data or training.

## Python setup

Run Python commands from `modules/`. Several imports and the relative Optuna
database path depend on this working directory.

```bash
cd /home/ilia/Code/voice_assistant/modules
python3 -m venv .va
source .va/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r reqs.txt
```

The AMR model used by intent recognition is downloaded automatically on first
use.

## Frontend setup

```bash
cd /home/ilia/Code/voice_assistant/frontend
npm install
npm run dev
```

Vite serves the development frontend at `http://localhost:5173` by default.
The frontend calls `http://localhost:8000` unless `VITE_API_BASE_URL` is set.

To create a production build:

```bash
npm run build
```

## Start the backend

The default Conformer checkpoint must exist before the backend starts:

```text
/home/ilia/Data/vsa/stt_model_exp.pt
```

Start FastAPI from `modules/`:

```bash
cd /home/ilia/Code/voice_assistant/modules
source .va/bin/activate
python3 comms_main.py
```

The service listens on `http://127.0.0.1:8000`.

### API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/recognize?text=...` | Convert text into a semantic action graph |
| `POST` | `/transcribe?model_path=...` | Transcribe an uploaded audio file |
| `GET` | `/stt-models` | List server-side `.pt` checkpoints |
| `POST` | `/stt-models/upload` | Upload and validate a selected `.pt` checkpoint |

The frontend's **Choose model file** control uploads the selected checkpoint to:

```text
/home/ilia/Data/vsa/uploaded_models
```

Only `.pt` files of at most 2 GB are accepted. The backend inspects checkpoint
keys to distinguish a `CustomConformer` checkpoint from a `Wav2Vec` checkpoint.
Paths outside `/home/ilia/Data/vsa` are rejected.

The backend caches loaded models in memory. Restart it after replacing a file
at an already loaded path.

## Speech datasets

Dataset identifiers used by the experiment configuration are:

```python
Dataloader.MEDDIALOG_AUDIO
Dataloader.LIBRISPEECH
```

### MedDialog-Audio

The loader downloads only the clean subset and `metadata.csv` from
[aline-gassenn/MedDialog-Audio](https://huggingface.co/datasets/aline-gassenn/MedDialog-Audio).
It excludes the white-noise and hospital-background-noise variants.

The clean subset contains:

- 21,068 audio files
- 10,534 synthetic medical dialogues
- 136.68 hours of transcribed audio
- Eight synthetic TTS voices

The files and Hugging Face cache are stored under:

```text
/home/ilia/Data/vsa/meddialog-audio
```

For more reliable Hugging Face downloads, authenticate before the first run:

```bash
hf auth login
```

The dataset uses the CC BY-NC 4.0 license. Review its dataset card before use.

### LibriSpeech

LibriSpeech is stored under:

```text
/home/ilia/Data/vsa/train-clean-100
```

The loader uses `train-clean-100` for training and `dev-clean` for validation.

## Train and tune models

Run training from `modules/`:

```bash
cd /home/ilia/Code/voice_assistant/modules
source .va/bin/activate
python3 sst/main.py
```

Configure the experiment runner in `modules/sst/const.py`:

```python
class ExperimentConfig:
    EXPERIMENT = "all"  # original, frozen, full, conformer, or all
    PROTOCOL_VERSION = "v1"
    WAV2VEC_EPOCHS = 10
    CONFORMER_EPOCHS = 50
    BATCH_SIZE = 2
    FROZEN_HEAD_LEARNING_RATE = 1e-4
    FULL_HEAD_LEARNING_RATE = 1e-4
    FULL_TRANSFORMER_LEARNING_RATE = 1e-5
    CONFORMER_LEARNING_RATE = 3e-4
    NUM_WORKERS = 16
    GRADIENT_CLIP = 1.0
    MIXED_PRECISION = True
    RESUME_FROM = None
    VOICE_MANIFEST = None
    REQUIRE_VOICE_MANIFEST = False
```

The four experiments are implemented as separate methods on `Experiments`:
the untouched pretrained Wav2Vec2 baseline, frozen
Wav2Vec2 with a new CTC head, complete Wav2Vec2 fine-tuning, and a Conformer
trained from scratch. `main.py` has no command-line training arguments; all
experiment parameters and paths come from `ExperimentConfig`.

The published MedDialog metadata does not map filenames to the eight TTS voice
identities. Supply that mapping to prevent voice leakage:

```python
VOICE_MANIFEST = Path("/path/to/meddialog_voice_manifest.csv")
```

The CSV must contain `filename` and `voice_id` columns and cover every clean
audio file. When supplied, validation and test each hold out a complete voice
identity. Without it, the default deterministic fallback keeps complete
dialogues within an 80/10/10 split but cannot prevent TTS voices from appearing
in more than one split. Set `REQUIRE_VOICE_MANIFEST = True` to disable that
fallback.

After updating `const.py`, run:

```bash
python3 sst/main.py
```

Press `Ctrl+C` (or send `SIGTERM`) to stop safely. The active experiment saves
`interrupted.pt`, including the model, optimizer, scheduler, AMP scaler, epoch,
and next batch. With `AUTO_RESUME = True`, the next invocation resumes that
batch using the same deterministic shuffle. Completed experiments are skipped
when `SKIP_COMPLETED_EXPERIMENTS = True`.

Optuna uses the same interruption checkpoints under `OPTUNA_CHECKPOINTS`.
Interrupted parameter sets are re-enqueued automatically when the study is
opened again, while completed/pruned trials continue to count toward the total
trial budget.

Each epoch writes a `.pt` bundle containing model, optimizer, scheduler, AMP
scaler, timing, memory, parameter counts, and metrics. A human-readable
`.metrics.json` file is saved beside it. Validation reports loss, WER, and CER;
the fixed test split is evaluated after training. Experiment `results.json`
files additionally report designated-domain performance, training time, peak
GPU allocation, inference milliseconds per utterance, parameter
counts, and in-memory model size. LibriSpeech retention is reported for the
Wav2Vec2 experiments; the medical-only Conformer is evaluated on MedDialog.

See [`modules/sst/PARAMETER_SELECTION.md`](modules/sst/PARAMETER_SELECTION.md)
for the paper-backed protocol and search ranges.

MedDialog uses deterministic splits grouped by a `voice_id`, `voice`,
`tts_voice`, or `speaker_id` metadata column. The public metadata does not
identify which of its eight TTS voices generated each file, so the loader fails
closed until a `voice_id` column is added; it does not guess identity from the
speaker role or acoustic features. LibriSpeech uses its standard
train-clean-100, dev-clean, and test-clean splits.

### Optuna studies

The SQLite database is `modules/stt_optuna.db` when training is launched from
`modules/`.

| Study | Model and dataset |
| --- | --- |
| `stt_optimization` | Original LibriSpeech Conformer study |
| `meddialog_optimization` | MedDialog Conformer study |
| `wav2vec_meddialog_optimization` | MedDialog Wav2Vec2 study |

`N_TRIALS` and `WAV2VEC_N_TRIALS` are total study budgets, not additional
trials on every invocation. If two of five trials are complete or pruned, the
next run schedules three remaining trials. Failed or interrupted trials do not
consume the budget.

An interrupted study can resume, but an individual trial cannot resume from a
half-finished epoch because model and optimizer state are not saved per batch or
epoch.

The Wav2Vec study searches:

- Learning rate: `1e-6` to `5e-5`
- Weight decay: `1e-6` to `1e-2`
- Batch size: `1`, `2`, or `4`
- Epochs: `1` to `5`

Long MedDialog recordings can exceed a 16 GB GPU at batch sizes 2 or 4. Such
trials are caught and marked as pruned so the study can continue. Batch size 1
is the known safe setting on the development machine.

### Inspect study parameters

`show_hypo_params()` in `modules/sst/main.py` prints the best completed trial,
validation loss, and parameter values. Set its `study_name` argument to inspect
a specific study.

Optuna training uses the same normalized transcripts and WER/CER evaluator as
direct training.

## Checkpoint locations

```text
/home/ilia/Data/vsa/stt_model_exp.pt
/home/ilia/Data/vsa/wav2vec_model_exp.pt
/home/ilia/Data/vsa/wav2vec_trials/trial_XXXX_complete.pt
/home/ilia/Data/vsa/wav2vec_trials/trial_XXXX_pruned.pt
/home/ilia/Data/vsa/uploaded_models/*.pt
```

- `stt_model_exp.pt` is the final custom Conformer checkpoint and the backend's
  default model.
- `wav2vec_model_exp.pt` is written only after the Wav2Vec study finishes and a
  fresh final model completes training with the best parameters.
- Wav2Vec trial checkpoints are saved after completed trials and after normal
  Optuna pruning. Failed or out-of-memory trial models are not saved.
- Trial checkpoint files contain model `state_dict` values. Hyperparameters and
  losses remain in the Optuna SQLite database.

## GPU behavior

Training selects CUDA automatically when available:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Check the active environment:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Monitor training with:

```bash
watch -n 1 nvidia-smi
```

Optuna's sampler runs on CPU, while each model-training objective runs on the
selected CUDA device. Dataset decoding and DataLoader workers remain primarily
CPU-side.

## Intent recognition

The intent pipeline combines AMR parsing and spaCy analysis to produce a
semantic action graph. See
[`modules/actions_from_intent/README.md`](modules/actions_from_intent/README.md)
for its pipeline, usage, and role mappings.

## Troubleshooting

### The model upload button returns `Not Found`

The frontend is connected to an older backend process that does not include
`POST /stt-models/upload`. Restart `python3 comms_main.py`.

### A selected checkpoint is not used

Confirm that the transcription response shows the expected `model_path` and
architecture. The backend caches checkpoints by path, so restart it if the file
at that path was overwritten.

### Optuna reports an existing study

This is expected. Studies use `load_if_exists=True` so they resume from
`stt_optuna.db`.

### Hugging Face prints `Fetching 21069 files` with `0.00B`

The client is checking the local snapshot. `0.00B` means it is not downloading
the dataset again.

### CUDA allocation warnings appear but training continues

An optional CUDA workspace allocation may have fallen back to a lower-memory
implementation. If training continues, no action is required. A fatal OOM is
reported as a pruned trial; the next trial then starts after model cleanup and
CUDA cache clearing.

## Current limitations

- Training uses character-level CTC with a small English vocabulary.
- Resume points are at epoch boundaries, not individual batches.
- Uploaded model files are copied to server storage rather than referenced in
  place, because browsers cannot expose arbitrary local filesystem paths.
- This is an experimental system and is not intended for clinical use.
