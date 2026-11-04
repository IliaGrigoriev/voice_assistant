import optuna
import torch
import torch.nn as nn
from tqdm import tqdm

VOCABULARY = ["<blank>", " ", "'", *"abcdefghijklmnopqrstuvwxyz"]
VOCAB_SIZE = len(VOCABULARY)
HOP_LENGTH = 160
BLANK_ID = VOCABULARY.index("<blank>")


class Trainer:

    def __init__(self, model, optimizer, device, hop_length, blank_id):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.hop_length = hop_length
        self.criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def _calculate_output_lengths(self, waveform_lengths):
        lengths = waveform_lengths // self.hop_length + 1
        lengths = (lengths + 1) // 2
        lengths = (lengths + 1) // 2
        return lengths.to(dtype=torch.long)

    def _calculate_loss(self, batch):
        waveforms = batch["waveforms"].to(self.device, non_blocking=True)
        waveform_lengths = batch["waveform_lengths"].to(
            self.device,
            non_blocking=True,
        )
        targets = batch["encoded_transcripts"].to(
            self.device,
            non_blocking=True,
        )

        if self.model.uses_waveform_lengths:
            logits, output_lengths = self.model(
                waveforms,
                waveform_lengths,
            )
        else:
            logits = self.model(waveforms)
            output_lengths = self._calculate_output_lengths(
                batch["waveform_lengths"]
            )
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        return self.criterion(
            log_probs,
            targets,
            output_lengths.cpu(),
            batch["transcript_lengths"],
        )

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, unit="batch", leave=False):
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._calculate_loss(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.inference_mode()
    def validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0

        for batch in loader:
            total_loss += self._calculate_loss(batch).item()

        return total_loss / len(loader)

    def fit(self, train_loader, validation_loader, epochs, trial=None):
        validation_loss = float("inf")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            validation_loss = self.validate_epoch(validation_loader)
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"train loss: {train_loss:.4f}, "
                f"validation loss: {validation_loss:.4f}"
            )

            if trial is not None:
                trial.report(validation_loss, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return validation_loss
