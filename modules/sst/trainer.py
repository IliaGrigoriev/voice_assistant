import json
import time
from contextlib import nullcontext
from pathlib import Path

import optuna
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from .text import TextProcessor
except ImportError:
    from text import TextProcessor

VOCABULARY = ["<blank>", " ", "'", *"abcdefghijklmnopqrstuvwxyz"]
VOCAB_SIZE = len(VOCABULARY)
HOP_LENGTH = 160
BLANK_ID = VOCABULARY.index("<blank>")


class TriStageLRScheduler:
    """Per-update warmup, hold, and linear-decay scheduler."""

    def __init__(self, optimizer, total_steps, warmup_fraction=0.1,
                 hold_fraction=0.4):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(1, round(total_steps * warmup_fraction))
        self.hold_steps = round(total_steps * hold_fraction)
        self.decay_steps = max(
            1, self.total_steps - self.warmup_steps - self.hold_steps
        )
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0
        self._apply_factor(self._factor(0))

    def _factor(self, step):
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if step < self.warmup_steps + self.hold_steps:
            return 1.0
        decay_progress = (
            step - self.warmup_steps - self.hold_steps + 1
        ) / self.decay_steps
        return max(0.0, 1.0 - decay_progress)

    def _apply_factor(self, factor):
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def step(self):
        self.step_count += 1
        self._apply_factor(self._factor(self.step_count))

    def state_dict(self):
        return {"step_count": self.step_count}

    def load_state_dict(self, state):
        self.step_count = state["step_count"]
        self._apply_factor(self._factor(self.step_count))


class Trainer:
    """CTC trainer with reproducible evaluation and resumable checkpoints."""

    def __init__(self, model, optimizer, device, hop_length, blank_id,
                 scheduler=None, gradient_clip=1.0, mixed_precision=False,
                 checkpoint_dir=None, seed=42, gradient_accumulation=1,
                 epoch_callback=None, early_stopping_patience=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.hop_length = hop_length
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.mixed_precision = mixed_precision and device.type == "cuda"
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
        self.text = TextProcessor(VOCABULARY, blank_id)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        self.last_metrics = {}
        self.seed = seed
        self.gradient_accumulation = max(1, gradient_accumulation)
        self.epoch_callback = epoch_callback
        self.early_stopping_patience = early_stopping_patience
        self.best_validation_wer = float("inf")
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        self.completed_batches = 0
        self.epoch_completed = False
        self.partial_loss = 0.0
        self.partial_batches = 0

    @property
    def parameter_counts(self):
        return {
            "parameters_total": sum(p.numel() for p in self.model.parameters()),
            "parameters_trainable": sum(p.numel() for p in self.model.parameters()
                                        if p.requires_grad),
        }

    @property
    def model_size_bytes(self):
        tensors = list(self.model.parameters()) + list(self.model.buffers())
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def _calculate_output_lengths(self, waveform_lengths):
        lengths = waveform_lengths // self.hop_length + 1
        lengths = (lengths + 1) // 2
        lengths = (lengths + 1) // 2
        return lengths.to(dtype=torch.long)

    def _forward(self, batch):
        waveforms = batch["waveforms"].to(self.device, non_blocking=True)
        lengths = batch["waveform_lengths"].to(self.device, non_blocking=True)
        if self.model.uses_waveform_lengths:
            return self.model(waveforms, lengths)
        return self.model(waveforms), self._calculate_output_lengths(
            batch["waveform_lengths"])

    def _loss(self, logits, output_lengths, batch):
        targets = batch["encoded_transcripts"].to(self.device, non_blocking=True)
        # CTC loss has no CUDA FP16 implementation and is numerically
        # sensitive. Keep the model forward pass under AMP, but calculate
        # log-probabilities and CTC itself in FP32.
        log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
        return self.criterion(
            log_probs, targets,
            output_lengths.cpu(), batch["transcript_lengths"])

    def _autocast(self):
        return (torch.amp.autocast("cuda", dtype=torch.float16)
                if self.mixed_precision else nullcontext())

    def train_epoch(self, loader, epoch, start_batch=0):
        self.model.train()
        total_loss = self.partial_loss if start_batch else 0.0
        sampler = getattr(loader, "sampler", None)
        generator = getattr(sampler, "generator", None)
        if generator is not None:
            generator.manual_seed(self.seed + epoch)
        self.current_epoch = epoch
        self.completed_batches = start_batch
        self.epoch_completed = False
        processed = self.partial_batches if start_batch else 0
        self.optimizer.zero_grad(set_to_none=True)
        pending_loss = 0.0
        pending_batches = 0
        for batch_index, batch in enumerate(
            tqdm(loader, unit="batch", leave=False)
        ):
            if batch_index < start_batch:
                continue
            with self._autocast():
                logits, output_lengths = self._forward(batch)
                loss = self._loss(logits, output_lengths, batch)
            self.scaler.scale(loss / self.gradient_accumulation).backward()
            pending_loss += loss.item()
            pending_batches += 1
            update_due = (
                pending_batches == self.gradient_accumulation
                or batch_index + 1 == len(loader)
            )
            if update_due:
                self.scaler.unscale_(self.optimizer)
                if pending_batches < self.gradient_accumulation:
                    correction = self.gradient_accumulation / pending_batches
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                if self.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler:
                    self.scheduler.step()
                total_loss += pending_loss
                processed += pending_batches
                pending_loss = 0.0
                pending_batches = 0
                self.partial_loss = total_loss
                self.partial_batches = processed
                self.completed_batches = batch_index + 1
        average = total_loss / max(1, processed)
        self.partial_loss = 0.0
        self.partial_batches = 0
        return average

    @torch.inference_mode()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        inference_seconds = 0.0
        sample_count = 0
        empty_prediction_count = 0
        blank_prediction_count = 0
        prediction_frame_count = 0
        counts = {"word_errors": 0, "words": 0, "char_errors": 0, "chars": 0}
        for batch in loader:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            with self._autocast():
                logits, output_lengths = self._forward(batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_seconds += time.perf_counter() - started
            sample_count += len(batch["transcripts"])
            total_loss += self._loss(logits, output_lengths, batch).item()
            hypotheses = []
            for tokens, length in zip(
                logits.argmax(-1).cpu(), output_lengths.cpu()
            ):
                valid_tokens = tokens[:int(length)]
                prediction_frame_count += valid_tokens.numel()
                blank_prediction_count += int(
                    (valid_tokens == self.text.blank_id).sum()
                )
                hypothesis = self.text.decode(valid_tokens)
                empty_prediction_count += not hypothesis
                hypotheses.append(hypothesis)
            current = self.text.error_counts(batch["transcripts"], hypotheses)
            for name in counts:
                counts[name] += current[name]
        return {
            "loss": total_loss / max(1, len(loader)),
            **self.text.rates(counts),
            "inference_seconds": inference_seconds,
            "inference_ms_per_utterance": (
                1000 * inference_seconds / max(1, sample_count)
            ),
            "empty_prediction_rate": (
                empty_prediction_count / max(1, sample_count)
            ),
            "blank_frame_rate": (
                blank_prediction_count / max(1, prediction_frame_count)
            ),
            "model_size_bytes": self.model_size_bytes,
            **self.parameter_counts,
        }

    def save_checkpoint(self, epoch, metrics, elapsed_seconds,
                        filename=None, next_epoch=None, next_batch=0):
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / (filename or f"epoch_{epoch:04d}.pt")
        payload = {
            "epoch": epoch, "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(), "metrics": metrics,
            "elapsed_seconds": elapsed_seconds, **self.parameter_counts,
            "next_epoch": next_epoch if next_epoch is not None else epoch + 1,
            "next_batch": next_batch,
            "partial_loss": self.partial_loss,
            "partial_batches": self.partial_batches,
            "best_validation_wer": self.best_validation_wer,
            "epochs_without_improvement": self.epochs_without_improvement,
        }
        torch.save(payload, path)
        serializable = {key: value for key, value in payload.items()
                        if not key.endswith("state_dict")}
        path.with_suffix(".metrics.json").write_text(
            json.dumps(serializable, indent=2), encoding="utf-8")
        return path

    def resume(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.last_metrics = checkpoint.get("metrics", {})
        self.partial_loss = checkpoint.get("partial_loss", 0.0)
        self.partial_batches = checkpoint.get("partial_batches", 0)
        self.best_validation_wer = checkpoint.get(
            "best_validation_wer", float("inf")
        )
        self.epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement", 0
        )
        return (checkpoint.get("next_epoch", checkpoint["epoch"] + 1),
                checkpoint.get("next_batch", 0),
                checkpoint.get("elapsed_seconds", 0.0))

    def load_best_model(self):
        if self.checkpoint_dir is None:
            return None
        path = self.checkpoint_dir / "best.pt"
        if not path.is_file():
            return None
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def fit(self, train_loader, validation_loader, epochs, test_loader=None,
            trial=None, resume_from=None):
        start_epoch, start_batch, previous_time = (
            (1, 0, 0.0) if not resume_from else self.resume(resume_from)
        )
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        validation = {"loss": float("inf"), "wer": 1.0, "cer": 1.0}
        metrics = dict(self.last_metrics)
        try:
          for epoch in range(start_epoch, epochs + 1):
            if self.epoch_callback:
                self.epoch_callback(epoch)
            train_loss = self.train_epoch(
                train_loader, epoch,
                start_batch if epoch == start_epoch else 0,
            )
            validation = self.evaluate(validation_loader)
            validation_improved = (
                validation["wer"] < self.best_validation_wer
            )
            if validation_improved:
                self.best_validation_wer = validation["wer"]
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            stop_early = (
                self.early_stopping_patience is not None
                and self.epochs_without_improvement
                >= self.early_stopping_patience
            )
            elapsed = previous_time + time.perf_counter() - started
            learning_rates = {
                group.get("name", f"group_{index}"): group["lr"]
                for index, group in enumerate(self.optimizer.param_groups)
            }
            metrics = {
                "epoch": epoch, "train_loss": train_loss,
                **{f"validation_{key}": value for key, value in validation.items()},
                "learning_rates": learning_rates,
                "optimizer_updates": (
                    self.scheduler.step_count if self.scheduler else None
                ),
                "gradient_accumulation": self.gradient_accumulation,
                "best_validation_wer": self.best_validation_wer,
                "epochs_without_improvement": self.epochs_without_improvement,
                "stopped_early": stop_early,
                "gpu_peak_memory_bytes": (torch.cuda.max_memory_allocated(self.device)
                                           if self.device.type == "cuda" else 0),
                "training_seconds": elapsed, **self.parameter_counts,
                "model_size_bytes": self.model_size_bytes,
            }
            self.last_metrics = metrics
            checkpoint = self.save_checkpoint(epoch, metrics, elapsed)
            if validation_improved:
                self.save_checkpoint(
                    epoch, metrics, elapsed, filename="best.pt"
                )
            self.epoch_completed = True
            print(f"Epoch {epoch}/{epochs}: {metrics}; checkpoint={checkpoint}")
            if trial is not None:
                trial.report(validation["wer"], step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if stop_early:
                print(
                    "Early stopping: validation WER did not improve for "
                    f"{self.early_stopping_patience} epochs"
                )
                break
        except KeyboardInterrupt:
            elapsed = previous_time + time.perf_counter() - started
            path = self.save_checkpoint(
                self.current_epoch, self.last_metrics, elapsed,
                filename="interrupted.pt",
                next_epoch=(self.current_epoch + 1 if self.epoch_completed
                            else self.current_epoch),
                next_batch=(0 if self.epoch_completed
                            else self.completed_batches),
            )
            print(f"Training interrupted safely; resume from {path}")
            raise
        if test_loader is not None:
            test_metrics = self.evaluate(test_loader)
            elapsed = previous_time + time.perf_counter() - started
            final = {**metrics,
                     **{f"validation_{key}": value for key, value in validation.items()},
                     **{f"test_{key}": value for key, value in test_metrics.items()}}
            self.save_checkpoint(epochs, final, elapsed)
            self.last_metrics = final
            print(f"Fixed test split: {test_metrics}")
        return validation["loss"]
