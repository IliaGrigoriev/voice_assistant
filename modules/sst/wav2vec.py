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

    def set_classifier_only(self, enabled=True):
        """Freeze/unfreeze the pretrained encoder for the requested mode."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not enabled
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True
        return self

    def forward(self, waveform, lengths=None):
        feature_layers, output_lengths = self.encoder.extract_features(
            waveform,
            lengths,
        )
        logits = self.classifier(feature_layers[-1])
        return logits, output_lengths


class OriginalWav2Vec(STTModel):
    """Unmodified LibriSpeech Wav2Vec2 ASR baseline.

    The pretrained bundle's label order is blank, word separator, apostrophe,
    and A-Z, which is identical to the shared vocabulary after lower-casing
    and replacing the word separator with a space.
    """

    uses_waveform_lengths = True

    def __init__(self):
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        labels = tuple(bundle.get_labels())
        expected = ("-", "|", "'", *tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        if labels != expected:
            raise ValueError(f"Unexpected Wav2Vec2 label order: {labels}")
        self.model = bundle.get_model()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, waveform, lengths=None):
        return self.model(waveform, lengths)
