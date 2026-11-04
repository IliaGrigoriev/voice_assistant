from pathlib import Path

import torch
import torchaudio
from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download

from const import LIBRI_SPEECH, MEDDIALOG_AUDIO

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data   import DataLoader as TorchDataLoader

class Dataloader:

    LIBRISPEECH = "librispeech"
    MEDDIALOG_AUDIO = "meddialog-audio"

    # Constructor
    def __init__(self, VOCABULARY, dataset_name=LIBRISPEECH):
        self._VOCABULARY = VOCABULARY
        self._dataset_name = dataset_name
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
                "Use get_train_batches() or get_validation_batches() for "
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

    def _load_meddialog_splits(self):
        if self._meddialog_splits is None:
            dataset_root = snapshot_download(
                repo_id="aline-gassenn/MedDialog-Audio",
                repo_type="dataset",
                allow_patterns=[
                    "metadata.csv",
                    "noise-free audio/**/*.wav",
                ],
                local_dir=MEDDIALOG_AUDIO,
            )
            metadata_path = Path(dataset_root) / "metadata.csv"
            samples = load_dataset(
                "csv",
                data_files=str(metadata_path),
                split="train",
                cache_dir=MEDDIALOG_AUDIO,
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

            audio_root = Path(dataset_root) / "noise-free audio"
            audio_paths = {
                path.name: str(path)
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
                [audio_paths[name] for name in samples["filename"]],
            ).cast_column(
                "audio",
                Audio(sampling_rate=16_000),
            )
            print(samples)
            print(samples.features)
            self._meddialog_splits = samples.train_test_split(
                test_size=0.1,
                seed=42,
            )

        return self._meddialog_splits

    def _get_meddialog_batches(
        self,
        split,
        batch_size,
        num_workers,
        shuffle,
    ):
        splits = self._load_meddialog_splits()
        dataset = (
            splits["train"] if split == "train" else splits["test"]
        )

        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_batch,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
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
            transcript = transcript.lower()

            CHAR_TO_ID = {
                character: index
                for index, character in enumerate(self._VOCABULARY)
            }
            normalized_transcript = "".join(
                character
                for character in transcript
                if character in CHAR_TO_ID
            )
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
        }
