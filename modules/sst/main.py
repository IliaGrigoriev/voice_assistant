import torch

from const import STT_MODEL
from dataloader import Dataloader
from hypo_tune import HypoTune
from stt_model import STTModel
from trainer import BLANK_ID, HOP_LENGTH, Trainer, VOCAB_SIZE, VOCABULARY


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study = HypoTune(device=device).tune(n_trials=20)
    best = study.best_params
    print("Best validation loss:", study.best_value)
    print("Best parameters:", best)

    model = STTModel(
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

    data = Dataloader(VOCABULARY)
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


if __name__ == "__main__":
    train()
