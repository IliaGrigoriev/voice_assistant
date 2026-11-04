import torch.nn as nn
import torchaudio

try:
    from .stt_model import STTModel
except ImportError:
    from stt_model import STTModel


class Wav2Vec(STTModel):
    """Wav2Vec2 encoder with the project's character-level CTC head."""

    uses_waveform_lengths = True

    def __init__(self, vocab_size):
        super().__init__()

        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.encoder = bundle.get_model()
        # Replace the bundle's 29-label ASR head with the shared project head.
        self.encoder.aux = None
        self.classifier = nn.Linear(768, vocab_size)

    def forward(self, waveform, lengths=None):
        feature_layers, output_lengths = self.encoder.extract_features(
            waveform,
            lengths,
        )
        logits = self.classifier(feature_layers[-1])
        return logits, output_lengths
