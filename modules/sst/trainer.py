import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader import Dataloader
from stt_model import STTModel
from const import STT_MODEL

VOCABULARY = ["<blank>", " ", "'", *"abcdefghijklmnopqrstuvwxyz"]
VOCAB_SIZE = len(VOCABULARY) # 29
HOP_LENGTH = 160 # 10 ms at a 16 kHz sample rate

# Calculate output lengths
# -----------------------------------------------------------------------
def calculate_output_lengths(waveform_lengths, hop_length=160):
    # MelSpectrogram uses center=True by default, so the number of valid
    # spectrogram frames is floor(audio_samples / hop_length) + 1.
    lengths = waveform_lengths // hop_length + 1

    # Conv2d(kernel_size=3, stride=2, padding=1) changes length L to
    # floor((L + 1) / 2), equivalent to (L + 1) // 2.
    lengths = (lengths + 1) // 2
    lengths = (lengths + 1) // 2

    return lengths.to(dtype=torch.long)

# Main
# -----------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Build model
    model = STTModel(
        vocab_size = VOCAB_SIZE,
        hop_length = HOP_LENGTH,
    ).to(device)

    # Compute the loss
    BLANK_ID = VOCABULARY.index("<blank>")  # 0
    criterion = nn.CTCLoss(
        blank         = BLANK_ID,
        zero_infinity = True,
    )

    # Prepare data for training
    data_loader = Dataloader(VOCABULARY)
    batches = data_loader.get_batches(batch_size = 8)

    # Model training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2,
    )  

    NUMBER_OF_EPOCHS = 10
    for epoch in range(NUMBER_OF_EPOCHS):
        model.train() # set the model into the training mode
        total_loss = 0.0
        # Train
        progress = tqdm(
            batches,
            desc=f"Epoch {epoch + 1}/{NUMBER_OF_EPOCHS}",
            unit="batch",
        )
        for batch in progress:
            waveforms = batch["waveforms"].to(device, non_blocking=True)
            targets = batch["encoded_transcripts"].to(device, non_blocking=True)
            
            # Length tensors can remain on CPU for maximum CTCLoss compatibility.
            waveform_lengths = batch["waveform_lengths"]
            target_lengths   = batch["transcript_lengths"]

            # Pass features through the rest of the model.
            logits = model(waveforms)

            # Calculate the log-probabilities for each character.
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            output_lengths = calculate_output_lengths(
                waveform_lengths,
                HOP_LENGTH,
            )
            loss = criterion(
                log_probs,
                targets,
                output_lengths,
                target_lengths,
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            batch_loss = loss.item()
            total_loss += batch_loss
            progress.set_postfix(loss=f"{batch_loss:.4f}")

        average_loss = total_loss / len(batches)
        print(
            f"Epoch {epoch + 1}/{NUMBER_OF_EPOCHS}, "
            f"loss: {average_loss:.4f}"
        )
    torch.save(model.state_dict(), STT_MODEL)
