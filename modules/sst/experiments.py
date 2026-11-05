import math
import shutil

import torch

from custom_conformer import CustomConformer
from data.dataloader import Dataloader
from evaluator import ExperimentEvaluator
from trainer import (
    BLANK_ID, HOP_LENGTH, Trainer, TriStageLRScheduler,
    VOCAB_SIZE, VOCABULARY,
)
from wav2vec import OriginalWav2Vec, Wav2Vec


class Experiments:
    """Definitions of the four ASR experiments."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.evaluator = ExperimentEvaluator(config, device)

    @property
    def registry(self):
        return {
            "original": ("original_wav2vec", self.original_wav2vec_experiment),
            "frozen": ("frozen_wav2vec", self.frozen_wav2vec_experiment),
            "full": ("fully_finetuned_wav2vec",
                     self.fully_finetuned_wav2vec_experiment),
            "conformer": ("conformer_from_scratch",
                          self.conformer_from_scratch_experiment),
        }

    @property
    def loader_options(self):
        return self.evaluator.loader_options

    def _medical_data(self):
        return Dataloader(
            VOCABULARY, Dataloader.MEDDIALOG_AUDIO,
            voice_manifest_path=self.config.VOICE_MANIFEST,
            require_voice_manifest=self.config.REQUIRE_VOICE_MANIFEST,
        )

    def _wav2vec(self):
        return Wav2Vec(
            VOCAB_SIZE,
            time_mask_probability=self.config.TIME_MASK_PROBABILITY,
            time_mask_length=self.config.TIME_MASK_LENGTH,
            channel_mask_probability=self.config.CHANNEL_MASK_PROBABILITY,
            channel_mask_length=self.config.CHANNEL_MASK_LENGTH,
        )

    def _resume_path(self, name):
        interrupted = self.config.OUTPUT_DIR / name / "interrupted.pt"
        resume_from = self.config.RESUME_FROM
        if (resume_from is None and self.config.AUTO_RESUME
                and interrupted.is_file()):
            resume_from = interrupted
            print(f"Automatically resuming {name} from {interrupted}")
        return resume_from, interrupted

    def _train_and_evaluate(
        self, name, purpose, model, optimizer, epochs,
        gradient_accumulation, include_librispeech,
        epoch_callback=None, early_stopping_patience=None,
    ):
        model.to(self.device)
        data = self._medical_data()
        train_loader = data.get_train_batches(**self.loader_options)
        validation_loader = data.get_validation_batches(**self.loader_options)
        updates_per_epoch = math.ceil(
            len(train_loader) / gradient_accumulation
        )
        scheduler = TriStageLRScheduler(
            optimizer,
            total_steps=updates_per_epoch * epochs,
            warmup_fraction=self.config.LR_WARMUP_FRACTION,
            hold_fraction=self.config.LR_HOLD_FRACTION,
        )
        trainer = Trainer(
            model=model, optimizer=optimizer, device=self.device,
            hop_length=HOP_LENGTH, blank_id=BLANK_ID,
            scheduler=scheduler, gradient_clip=self.config.GRADIENT_CLIP,
            mixed_precision=self.config.MIXED_PRECISION,
            checkpoint_dir=self.config.OUTPUT_DIR / name,
            seed=self.config.RANDOM_SEED,
            gradient_accumulation=gradient_accumulation,
            epoch_callback=epoch_callback,
            early_stopping_patience=early_stopping_patience,
        )
        resume_from, interrupted = self._resume_path(name)
        trainer.fit(
            train_loader, validation_loader, epochs,
            resume_from=resume_from,
        )
        interrupted.unlink(missing_ok=True)
        final_training_metrics = dict(trainer.last_metrics)
        best_checkpoint = trainer.load_best_model()
        self.publish_best_model(name, trainer.checkpoint_dir)
        metrics = {
            **final_training_metrics,
            "selected_epoch": (
                best_checkpoint["epoch"] if best_checkpoint else
                final_training_metrics.get("epoch")
            ),
            "selected_validation_metrics": (
                best_checkpoint.get("metrics", {}) if best_checkpoint else {}
            ),
            **self.evaluator.evaluate_domains(
                trainer, include_librispeech=include_librispeech
            ),
        }
        return self.evaluator.write_results(name, purpose, metrics)

    def publish_best_model(self, name, checkpoint_dir):
        """Copy an experiment's best checkpoint into the selectable models."""
        source = checkpoint_dir / "best.pt"
        if not source.is_file():
            print(f"No best checkpoint to publish for {name}: {source}")
            return None

        destination_dir = self.config.AVAILABLE_MODELS_DIR
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{name}.pt"
        shutil.copy2(source, destination)
        print(f"Published best model for {name}: {destination}")
        return destination

    def original_wav2vec_experiment(self):
        model = OriginalWav2Vec().to(self.device)
        trainer = Trainer(
            model=model, optimizer=None, device=self.device,
            hop_length=HOP_LENGTH, blank_id=BLANK_ID,
            mixed_precision=self.config.MIXED_PRECISION,
        )
        metrics = {
            "training_seconds": 0.0,
            **self.evaluator.evaluate_domains(
                trainer, include_librispeech=True
            ),
        }
        metrics["gpu_peak_memory_bytes"] = metrics[
            "evaluation_gpu_peak_memory_bytes"
        ]
        return self.evaluator.write_results(
            "original_wav2vec", "Baseline without medical fine-tuning",
            metrics,
        )

    def frozen_wav2vec_experiment(self):
        model = self._wav2vec().set_classifier_only(True).to(self.device)
        optimizer = torch.optim.AdamW(
            [{
                "params": list(model.classifier.parameters()),
                "lr": self.config.FROZEN_HEAD_LEARNING_RATE,
                "name": "ctc_head",
            }],
            weight_decay=self.config.WAV2VEC_WEIGHT_DECAY,
        )
        return self._train_and_evaluate(
            "frozen_wav2vec", "Low-cost medical fine-tuning",
            model, optimizer, self.config.WAV2VEC_EPOCHS,
            self.config.FROZEN_GRADIENT_ACCUMULATION, True,
            early_stopping_patience=(
                self.config.WAV2VEC_EARLY_STOPPING_PATIENCE
            ),
        )

    def fully_finetuned_wav2vec_experiment(self):
        model = self._wav2vec().set_full_finetuning(head_only=True)
        model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.optimizer_groups(
                self.config.FULL_HEAD_LEARNING_RATE,
                self.config.FULL_TRANSFORMER_LEARNING_RATE,
            ),
            weight_decay=self.config.WAV2VEC_WEIGHT_DECAY,
        )

        def configure_stage(epoch):
            head_only = epoch <= self.config.HEAD_ONLY_EPOCHS
            model.set_full_finetuning(head_only=head_only)
            stage = "head-only" if head_only else "head + Transformer"
            print(f"Full Wav2Vec2 stage for epoch {epoch}: {stage}")

        return self._train_and_evaluate(
            "fully_finetuned_wav2vec", "Main medical-domain model",
            model, optimizer, self.config.WAV2VEC_EPOCHS,
            self.config.FULL_GRADIENT_ACCUMULATION, True,
            epoch_callback=configure_stage,
            early_stopping_patience=(
                self.config.WAV2VEC_EARLY_STOPPING_PATIENCE
            ),
        )

    def conformer_from_scratch_experiment(self):
        model = CustomConformer(
            VOCAB_SIZE, HOP_LENGTH, self.device,
            dropout=self.config.CONFORMER_DROPOUT,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            [{
                "params": list(model.parameters()),
                "lr": self.config.CONFORMER_LEARNING_RATE,
                "name": "conformer",
            }],
            weight_decay=self.config.CONFORMER_WEIGHT_DECAY,
        )
        return self._train_and_evaluate(
            "conformer_from_scratch", "From-scratch medical comparison",
            model, optimizer, self.config.CONFORMER_EPOCHS,
            self.config.CONFORMER_GRADIENT_ACCUMULATION, False,
            early_stopping_patience=(
                self.config.CONFORMER_EARLY_STOPPING_PATIENCE
            ),
        )
