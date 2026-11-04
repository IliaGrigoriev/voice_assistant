import gc
from pathlib import Path

import optuna
import torch

from const import WAV2VEC_TRIAL_MODELS
from custom_conformer import CustomConformer
from data.dataloader import Dataloader
from trainer import BLANK_ID, HOP_LENGTH, Trainer, VOCAB_SIZE, VOCABULARY
from wav2vec import Wav2Vec


class HypoTune:

    n_startup_trials = 5
    n_warmup_steps = 5

    def __init__(
        self,
        device=None,
        num_workers=16,
        dataset_name=Dataloader.LIBRISPEECH,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_workers = num_workers
        self.dataset_name = dataset_name

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

        model = CustomConformer(
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
        data = Dataloader(
            VOCABULARY,
            dataset_name=self.dataset_name,
        )
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

    def tune(
        self,
        n_trials=20,
        study_name="stt_optimization",
        storage="sqlite:///stt_optuna.db",
        seed_study_name=None,
    ):
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=self.n_warmup_steps,
                interval_steps=1,
            ),
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )

        if seed_study_name is not None and n_trials > 0 and not study.trials:
            seed_study = optuna.load_study(
                study_name=seed_study_name,
                storage=storage,
            )
            study.enqueue_trial(seed_study.best_params)

        budgeted_states = {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        }
        finished_trials = sum(
            trial.state in budgeted_states
            for trial in study.trials
        )
        remaining_trials = max(0, n_trials - finished_trials)
        print(
            f"Study {study_name!r}: {finished_trials}/{n_trials} "
            f"trials finished; {remaining_trials} remaining"
        )
        study.optimize(self.objective, n_trials=remaining_trials)
        return study


class Wav2VecHypoTune(HypoTune):

    n_startup_trials = 1
    n_warmup_steps = 1

    @staticmethod
    def _save_trial_model(model, trial, status):
        checkpoint_folder = Path(WAV2VEC_TRIAL_MODELS)
        checkpoint_folder.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_folder / (
            f"trial_{trial.number:04d}_{status}.pt"
        )
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved Wav2Vec trial model to {checkpoint_path}")

    def objective(self, trial):
        learning_rate = trial.suggest_float(
            "learning_rate",
            1e-6,
            5e-5,
            log=True,
        )
        weight_decay = trial.suggest_float(
            "weight_decay",
            1e-6,
            1e-2,
            log=True,
        )
        batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
        epochs = trial.suggest_int("epochs", 1, 5)

        model = None
        optimizer = None
        trainer = None
        try:
            model = Wav2Vec(vocab_size=VOCAB_SIZE)
            model.to(self.device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            data = Dataloader(
                VOCABULARY,
                dataset_name=self.dataset_name,
            )
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
            validation_loss = trainer.fit(
                train_loader=train_loader,
                validation_loader=validation_loader,
                epochs=epochs,
                trial=trial,
            )
            self._save_trial_model(model, trial, "complete")
            return validation_loss
        except optuna.TrialPruned:
            self._save_trial_model(model, trial, "pruned")
            raise
        except torch.cuda.OutOfMemoryError:
            raise optuna.TrialPruned("GPU ran out of memory") from None
        finally:
            del trainer, optimizer, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
