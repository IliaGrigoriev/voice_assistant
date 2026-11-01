import gc

import optuna
import torch

from dataloader import Dataloader
from stt_model import STTModel
from trainer import BLANK_ID, HOP_LENGTH, Trainer, VOCAB_SIZE, VOCABULARY


class HypoTune:

    def __init__(self, device=None, num_workers=16):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_workers = num_workers

    def objective(self, trial):
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        )
        dropout = trial.suggest_float("dropout", 0.05, 0.4)
        weight_decay = trial.suggest_float(
            "weight_decay", 1e-6, 1e-2, log=True
        )
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
        epochs = trial.suggest_int("epochs", 20, 80, step=10)

        model = STTModel(
            vocab_size=VOCAB_SIZE,
            hop_length=HOP_LENGTH,
            device=self.device,
            dropout=dropout,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        data = Dataloader(VOCABULARY)
        train_loader = data.get_train_batches(
            batch_size=batch_size,
            num_workers=self.num_workers,
        )
        validation_loader = data.get_validation_batches(
            batch_size=batch_size,
            num_workers=self.num_workers,
        )
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=self.device,
            hop_length=HOP_LENGTH,
            blank_id=BLANK_ID,
        )

        try:
            return trainer.fit(
                train_loader=train_loader,
                validation_loader=validation_loader,
                epochs=epochs,
                trial=trial,
            )
        except torch.cuda.OutOfMemoryError as error:
            raise optuna.TrialPruned("GPU ran out of memory") from error
        finally:
            del trainer, optimizer, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def tune(self, n_trials=20):
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1,
            ),
            storage="sqlite:///stt_optuna.db",
            study_name="stt_optimization",
            load_if_exists=True,
        )
        study.optimize(self.objective, n_trials=n_trials)
        return study
