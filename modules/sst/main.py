import torch

from const import STT_MODEL, WAV2VEC_MODEL
from custom_conformer import CustomConformer
from data.dataloader import Dataloader
from hypo_tune import HypoTune, Wav2VecHypoTune
from trainer import BLANK_ID, HOP_LENGTH, Trainer, VOCAB_SIZE, VOCABULARY
from wav2vec import Wav2Vec


DATASET = Dataloader.MEDDIALOG_AUDIO
STUDY_NAME = "meddialog_optimization"
SEED_STUDY_NAME = "stt_optimization"
N_TRIALS = 5
WAV2VEC_STUDY_NAME = "wav2vec_meddialog_optimization"
WAV2VEC_N_TRIALS = 5


def show_hypo_params(study_name=STUDY_NAME):
    import optuna

    study = optuna.load_study(
        study_name=study_name,
        storage="sqlite:///stt_optuna.db",
    )

    print("Best trial:", study.best_trial.number)
    print("Best validation loss:", study.best_value)
    print("Best parameters:")

    for name, value in study.best_params.items():
        print(f"  {name}: {value}")

def train_custom_conformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study = HypoTune(
        device=device,
        dataset_name=DATASET,
    ).tune(
        n_trials=N_TRIALS,
        study_name=STUDY_NAME,
        seed_study_name=SEED_STUDY_NAME,
    )
    best = study.best_params
    print("Best validation loss:", study.best_value)
    print("Best parameters:", best)

    model = CustomConformer(
        vocab_size=VOCAB_SIZE,
        hop_length=HOP_LENGTH,
        device=device,
        dropout=best["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best["learning_rate"],
        weight_decay=best["weight_decay"],
    )

    data = Dataloader(VOCABULARY, dataset_name=DATASET)
    train_loader = data.get_train_batches(batch_size=best["batch_size"])
    validation_loader = data.get_validation_batches(
        batch_size=best["batch_size"]
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        hop_length=HOP_LENGTH,
        blank_id=BLANK_ID,
    )
    trainer.fit(
        train_loader      = train_loader,
        validation_loader = validation_loader,
        epochs            = best["epochs"]
    )
    torch.save(model.state_dict(), STT_MODEL)


def fine_tune_wav2vec():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study = Wav2VecHypoTune(
        device=device,
        dataset_name=DATASET,
    ).tune(
        n_trials=WAV2VEC_N_TRIALS,
        study_name=WAV2VEC_STUDY_NAME,
    )
    best = study.best_params
    print("Best validation loss:", study.best_value)
    print("Best parameters:", best)

    model = Wav2Vec(vocab_size=VOCAB_SIZE).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best["learning_rate"],
        weight_decay=best["weight_decay"],
    )

    data = Dataloader(VOCABULARY, dataset_name=DATASET)
    train_loader = data.get_train_batches(batch_size=best["batch_size"])
    validation_loader = data.get_validation_batches(
        batch_size=best["batch_size"]
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        hop_length=HOP_LENGTH,
        blank_id=BLANK_ID,
    )
    trainer.fit(
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=best["epochs"],
    )
    torch.save(model.state_dict(), WAV2VEC_MODEL)


if __name__ == "__main__":
    fine_tune_wav2vec()
    # train_custom_conformer()
    # show_hypo_params()
