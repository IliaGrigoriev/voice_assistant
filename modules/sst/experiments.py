import torch

from custom_conformer import CustomConformer
from data.dataloader import Dataloader
from evaluator import ExperimentEvaluator
from trainer import BLANK_ID, HOP_LENGTH, Trainer, VOCAB_SIZE, VOCABULARY
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

    def _trainer(self, model, experiment_name, train=True):
        model = model.to(self.device)
        optimizer = None
        scheduler = None
        if train:
            parameters = [parameter for parameter in model.parameters()
                          if parameter.requires_grad]
            optimizer = torch.optim.AdamW(
                parameters, lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, self.config.EPOCHS)
            )
        return Trainer(
            model=model, optimizer=optimizer, device=self.device,
            hop_length=HOP_LENGTH, blank_id=BLANK_ID, scheduler=scheduler,
            gradient_clip=self.config.GRADIENT_CLIP,
            mixed_precision=self.config.MIXED_PRECISION,
            checkpoint_dir=self.config.OUTPUT_DIR / experiment_name,
            seed=self.config.RANDOM_SEED,
        )

    def _run_trained_experiment(self, name, purpose, model):
        medical = Dataloader(
            VOCABULARY, Dataloader.MEDDIALOG_AUDIO,
            voice_manifest_path=self.config.VOICE_MANIFEST,
        )
        trainer = self._trainer(model, name)
        interrupted = self.config.OUTPUT_DIR / name / "interrupted.pt"
        resume_from = self.config.RESUME_FROM
        if (resume_from is None and self.config.AUTO_RESUME
                and interrupted.is_file()):
            resume_from = interrupted
            print(f"Automatically resuming {name} from {interrupted}")
        trainer.fit(
            medical.get_train_batches(**self.loader_options),
            medical.get_validation_batches(**self.loader_options),
            self.config.EPOCHS,
            resume_from=resume_from,
        )
        interrupted.unlink(missing_ok=True)
        metrics = {
            **trainer.last_metrics,
            **self.evaluator.evaluate_domains(trainer),
        }
        trainer.save_checkpoint(
            self.config.EPOCHS, metrics,
            metrics.get("training_seconds", 0.0),
        )
        return self.evaluator.write_results(name, purpose, metrics)

    def original_wav2vec_experiment(self):
        trainer = self._trainer(
            OriginalWav2Vec(), "original_wav2vec", train=False
        )
        metrics = {
            "training_seconds": 0.0,
            **self.evaluator.evaluate_domains(trainer),
        }
        metrics["gpu_peak_memory_bytes"] = metrics[
            "evaluation_gpu_peak_memory_bytes"
        ]
        return self.evaluator.write_results(
            "original_wav2vec", "Baseline without medical fine-tuning",
            metrics,
        )

    def frozen_wav2vec_experiment(self):
        return self._run_trained_experiment(
            "frozen_wav2vec", "Low-cost medical fine-tuning",
            Wav2Vec(VOCAB_SIZE).set_classifier_only(True),
        )

    def fully_finetuned_wav2vec_experiment(self):
        return self._run_trained_experiment(
            "fully_finetuned_wav2vec", "Main medical-domain model",
            Wav2Vec(VOCAB_SIZE),
        )

    def conformer_from_scratch_experiment(self):
        return self._run_trained_experiment(
            "conformer_from_scratch", "From-scratch pretraining comparison",
            CustomConformer(VOCAB_SIZE, HOP_LENGTH, self.device),
        )
