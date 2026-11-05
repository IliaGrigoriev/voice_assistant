from pathlib import Path
import hashlib
import warnings

import torch
import torchaudio
from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download

from const import LIBRI_SPEECH, MEDDIALOG_AUDIO

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data   import DataLoader as TorchDataLoader

try:
    from ..text import TextProcessor
except ImportError:
    from text import TextProcessor

class Dataloader:

    LIBRISPEECH = "librispeech"
    MEDDIALOG_AUDIO = "meddialog-audio"
    SHUFFLE_SEED = 42
    _MEDDIALOG_SPLIT_CACHE = {}

    # Constructor
    def __init__(self, VOCABULARY, dataset_name=LIBRISPEECH,
                 voice_manifest_path=None, require_voice_manifest=False):
        self._VOCABULARY = VOCABULARY
        self._text = TextProcessor(VOCABULARY)
        self._dataset_name = dataset_name
        self._voice_manifest_path = voice_manifest_path
        self._require_voice_manifest = require_voice_manifest
        self._meddialog_splits = None

        supported_datasets = {self.LIBRISPEECH, self.MEDDIALOG_AUDIO}
        if dataset_name not in supported_datasets:
            raise ValueError(
                f"Unknown dataset {dataset_name!r}; choose one of "
                f"{sorted(supported_datasets)}"
            )

    # Get batches
    # ----------------------------------------------------------------
    def get_batches(
        self,
        batch_size,
        num_workers=16,
        dataset_url="train-clean-100",
        shuffle=True,
        download=False,
    ):
        if self._dataset_name == self.MEDDIALOG_AUDIO:
            raise ValueError(
                "Use the train, validation, or test batch methods for "
                "MedDialog-Audio"
            )

        dataset = torchaudio.datasets.LIBRISPEECH(
            root=str(LIBRI_SPEECH),
            url=dataset_url,
            download=download,
        )

        return TorchDataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = shuffle,
            collate_fn  = self._collate_batch,
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
            persistent_workers = num_workers > 0,
            prefetch_factor = 2 if num_workers > 0 else None,
            generator=(torch.Generator().manual_seed(self.SHUFFLE_SEED)
                       if shuffle else None),
        )

    def get_train_batches(self, batch_size, num_workers=16):
        if self._dataset_name == self.MEDDIALOG_AUDIO:
            return self._get_meddialog_batches(
                split="train",
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
            )

        return self.get_batches(
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_url="train-clean-100",
            shuffle=True,
        )

    def get_validation_batches(self, batch_size, num_workers=16):
        if self._dataset_name == self.MEDDIALOG_AUDIO:
            return self._get_meddialog_batches(
                split="validation",
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )

        return self.get_batches(
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_url="dev-clean",
            shuffle=False,
            download=True,
        )

    def get_test_batches(self, batch_size, num_workers=16):
        if self._dataset_name == self.MEDDIALOG_AUDIO:
            return self._get_meddialog_batches(
                split="test", batch_size=batch_size,
                num_workers=num_workers, shuffle=False,
            )
        return self.get_batches(
            batch_size=batch_size, num_workers=num_workers,
            dataset_url="test-clean", shuffle=False, download=True,
        )

    def _load_meddialog_splits(self):
        manifest_key = (
            str(Path(self._voice_manifest_path).resolve())
            if self._voice_manifest_path is not None else None
        )
        if self._voice_manifest_path is None and self._require_voice_manifest:
            raise ValueError(
                "ExperimentConfig.VOICE_MANIFEST is required for MedDialog. "
                "Set it to a CSV containing filename and voice_id for every "
                "recording. The public metadata does not provide voice IDs, "
                "and continuing without them would leak TTS voices across "
                "training, validation, and test splits."
            )
        if manifest_key in self._MEDDIALOG_SPLIT_CACHE:
            self._meddialog_splits = self._MEDDIALOG_SPLIT_CACHE[manifest_key]
            return self._meddialog_splits

        if self._meddialog_splits is None:
            dataset_root = Path(MEDDIALOG_AUDIO)
            metadata_path = dataset_root / "metadata.csv"
            audio_root = dataset_root / "noise-free audio"
            if not metadata_path.is_file() or not audio_root.is_dir():
                dataset_root = Path(snapshot_download(
                    repo_id="aline-gassenn/MedDialog-Audio",
                    repo_type="dataset",
                    allow_patterns=[
                        "metadata.csv",
                        "noise-free audio/**/*.wav",
                    ],
                    local_dir=MEDDIALOG_AUDIO,
                ))
                metadata_path = dataset_root / "metadata.csv"
                audio_root = dataset_root / "noise-free audio"
            samples = load_dataset(
                "csv",
                data_files=str(metadata_path),
                split="train",
                cache_dir=MEDDIALOG_AUDIO,
            )

            if self._voice_manifest_path is not None:
                manifest = load_dataset(
                    "csv", data_files=str(self._voice_manifest_path),
                    split="train", cache_dir=MEDDIALOG_AUDIO,
                )
                required = {"filename", "voice_id"}
                if not required.issubset(manifest.column_names):
                    raise ValueError(
                        "Voice manifest must contain filename and voice_id"
                    )
                voice_by_filename = dict(zip(
                    manifest["filename"], manifest["voice_id"]
                ))
                missing_voices = sorted(
                    set(samples["filename"]) - set(voice_by_filename)
                )
                if missing_voices:
                    raise ValueError(
                        f"Voice manifest is missing {len(missing_voices)} "
                        f"files; first missing file: {missing_voices[0]}"
                    )
                samples = samples.add_column(
                    "voice_id",
                    [voice_by_filename[name] for name in samples["filename"]],
                )

            if "filename" not in samples.column_names:
                raise ValueError(
                    "MedDialog metadata does not contain a 'filename' "
                    f"column. Available columns: {samples.column_names}"
                )
            if "transcription" not in samples.column_names:
                raise ValueError(
                    "MedDialog metadata does not contain a 'transcription' "
                    f"column. Available columns: {samples.column_names}"
                )

            audio_paths = {
                path.name: path
                for path in audio_root.rglob("*.wav")
            }
            missing_files = sorted(
                set(samples["filename"]) - set(audio_paths)
            )
            if missing_files:
                raise FileNotFoundError(
                    f"Could not find {len(missing_files)} MedDialog audio "
                    f"files; first missing file: {missing_files[0]}"
                )

            samples = samples.add_column(
                "audio",
                [str(audio_paths[name]) for name in samples["filename"]],
            ).cast_column(
                "audio",
                Audio(sampling_rate=16_000),
            )
            # Never guess a TTS identity from speaker role, filename, or F0.
            # A wrong guess silently leaks the same voice across evaluation.
            voice_columns = (
                "voice_id", "voice", "tts_voice", "speaker_id"
            )
            voice_column = next(
                (name for name in voice_columns
                 if name in samples.column_names), None
            )
            if voice_column is not None:
                groups = sorted(set(samples[voice_column]))
                if len(groups) < 3:
                    raise ValueError(
                        "At least three synthetic voice groups are required "
                        "for leak-free train/validation/test splits"
                    )
                test_groups = {groups[-1]}
                validation_groups = {groups[-2]}
                train_groups = set(groups) - validation_groups - test_groups
                group_values = samples[voice_column]
                self._meddialog_splits = {
                    "train": samples.select([
                        index for index, value in enumerate(group_values)
                        if value in train_groups
                    ]),
                    "validation": samples.select([
                        index for index, value in enumerate(group_values)
                        if value in validation_groups
                    ]),
                    "test": samples.select([
                        index for index, value in enumerate(group_values)
                        if value in test_groups
                    ]),
                }
            else:
                warnings.warn(
                    "MedDialog has no voice IDs; using deterministic "
                    "dialogue-disjoint 80/10/10 splits. Patient and doctor "
                    "recordings from one dialogue stay together, but the "
                    "same TTS voice may occur in multiple splits.",
                    RuntimeWarning,
                )

                def assign_split(row):
                    stem = Path(row["filename"]).stem
                    dialogue_id = stem.split("_")[0].split("-")[0]
                    digest = hashlib.sha256(
                        f"{self.SHUFFLE_SEED}:{dialogue_id}".encode("utf-8")
                    ).digest()
                    bucket = int.from_bytes(digest[:8], "big") % 100
                    split = ("train" if bucket < 80 else
                             "validation" if bucket < 90 else "test")
                    return {"dataset_split": split}

                samples = samples.map(assign_split)
                split_values = samples["dataset_split"]
                self._meddialog_splits = {
                    split: samples.select([
                        index for index, value in enumerate(split_values)
                        if value == split
                    ])
                    for split in ("train", "validation", "test")
                }
            self._MEDDIALOG_SPLIT_CACHE[manifest_key] = self._meddialog_splits

        return self._meddialog_splits

    def _get_meddialog_batches(
        self,
        split,
        batch_size,
        num_workers,
        shuffle,
    ):
        splits = self._load_meddialog_splits()
        dataset = splits[split]

        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_batch,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            generator=(torch.Generator().manual_seed(self.SHUFFLE_SEED)
                       if shuffle else None),
        )
    
    # Collate batch
    # ----------------------------------------------------------------
    def _collate_batch(self, samples):
        waveforms     = []
        encoded_items = []

        for sample in samples:
            if isinstance(sample, dict):
                audio = sample["audio"]
                if isinstance(audio, dict):
                    waveform = torch.as_tensor(audio["array"])
                    sample_rate = audio["sampling_rate"]
                else:
                    decoded_audio = audio.get_all_samples()
                    waveform = decoded_audio.data
                    sample_rate = decoded_audio.sample_rate
                transcript = sample["transcription"]
            else:
                waveform, sample_rate, transcript, *_ = sample

            if sample_rate != 16_000:
                raise ValueError(
                    f"Expected 16 kHz audio, got {sample_rate}"
                )

            waveform = waveform.to(dtype=torch.float32)
            if waveform.ndim == 2:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze()
            CHAR_TO_ID = {
                character: index
                for index, character in enumerate(self._VOCABULARY)
            }
            normalized_transcript = self._text.normalize(transcript)
            encoded_transcript = torch.tensor(
                [CHAR_TO_ID[character] for character in normalized_transcript],
                dtype=torch.long,
            )

            waveforms.append(waveform)
            encoded_items.append(encoded_transcript)

        waveform_lengths = torch.tensor(
            [waveform.numel() for waveform in waveforms],
            dtype=torch.long,
        )

        transcript_lengths = torch.tensor(
            [transcript.numel() for transcript in encoded_items],
            dtype=torch.long,
        )

        # Recordings have different durations, so their waveform tensors have
        # different lengths and cannot be stacked into one rectangular batch.
        # Pad shorter waveforms with silence (0.0) to the longest waveform in
        # this batch, producing a [batch_size, max_waveform_length] tensor.
        # waveform_lengths preserves each original length so later processing
        # can distinguish real audio samples from the added padding.
        padded_waveforms = pad_sequence(
            waveforms,
            batch_first=True,
            padding_value=0.0,
        )

        encoded_transcripts = torch.cat(encoded_items)

        return {
            "waveforms"           : padded_waveforms,
            "waveform_lengths"    : waveform_lengths,
            "encoded_transcripts" : encoded_transcripts,
            "transcript_lengths"  : transcript_lengths,
            "transcripts"         : [
                self._text.normalize(
                    sample["transcription"] if isinstance(sample, dict)
                    else sample[2]
                )
                for sample in samples
            ],
        }
