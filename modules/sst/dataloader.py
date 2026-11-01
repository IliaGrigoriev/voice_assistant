import torch
import torchaudio

from const import LIBRI_SPEECH

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data   import DataLoader as TorchDataLoader

class Dataloader:

    # Constructor
    def __init__(self, VOCABULARY):
        self._VOCABULARY = VOCABULARY

    # Get batches
    # ----------------------------------------------------------------
    def get_batches(
        self,
        batch_size,
        num_workers=16,
        dataset_url="train-clean-100",
        shuffle=True,
    ):
        dataset = torchaudio.datasets.LIBRISPEECH(
            root=str(LIBRI_SPEECH),
            url=dataset_url,
            download=False,
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
        return self.get_batches(
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_url="train-clean-100",
            shuffle=True,
        )

    def get_validation_batches(self, batch_size, num_workers=16):
        return self.get_batches(
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_url="dev-clean",
            shuffle=False,
        )
    
    # Collate batch
    # ----------------------------------------------------------------
    def _collate_batch(self, samples):
        waveforms     = []
        encoded_items = []

        for waveform, sample_rate, transcript, *_ in samples:
            if sample_rate != 16_000:
                raise ValueError(
                    f"Expected 16 kHz audio, got {sample_rate}"
                )

            waveform   = waveform.squeeze(0)
            transcript = transcript.lower()

            CHAR_TO_ID = {
                character: index
                for index, character in enumerate(self._VOCABULARY)
            }
            encoded_transcript = torch.tensor(
                [CHAR_TO_ID[character] for character in transcript],
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
