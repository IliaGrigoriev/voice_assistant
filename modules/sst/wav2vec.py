import torch
import torch.nn as nn
import torchaudio

try:
    from .stt_model import STTModel
except ImportError:
    from stt_model import STTModel


class Wav2Vec(STTModel):
    """Wav2Vec2 encoder with the project's character-level CTC head."""

    uses_waveform_lengths = True

    def __init__(self, vocab_size, time_mask_probability=0.0,
                 time_mask_length=10, channel_mask_probability=0.0,
                 channel_mask_length=64):
        super().__init__()

        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.encoder = bundle.get_model()
        # Replace the bundle's 29-label ASR head with the shared project head.
        self.encoder.aux = None
        self.classifier = nn.Linear(768, vocab_size)
        self.time_mask_probability = time_mask_probability
        self.time_mask_length = time_mask_length
        self.channel_mask_probability = channel_mask_probability
        self.channel_mask_length = channel_mask_length

    def set_classifier_only(self, enabled=True):
        """Freeze/unfreeze the pretrained encoder for the requested mode."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not enabled
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True
        return self

    def set_full_finetuning(self, head_only=False):
        """Freeze CNN features permanently and optionally freeze Transformer."""
        for parameter in self.encoder.feature_extractor.parameters():
            parameter.requires_grad = False
        for parameter in self.encoder.encoder.parameters():
            parameter.requires_grad = not head_only
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True
        return self

    def optimizer_groups(self, head_lr, transformer_lr):
        return [
            {
                "params": list(self.classifier.parameters()),
                "lr": head_lr,
                "name": "ctc_head",
            },
            {
                "params": list(self.encoder.encoder.parameters()),
                "lr": transformer_lr,
                "name": "transformer",
            },
        ]

    @staticmethod
    def _mask_spans(features, probability, length, dimension):
        if probability <= 0 or length <= 0:
            return features
        size = features.shape[dimension]
        expected_spans = probability * size / length
        span_count = int(expected_spans)
        if torch.rand((), device=features.device) < expected_spans - span_count:
            span_count += 1
        if span_count == 0:
            return features
        if size <= length:
            return features
        for batch_index in range(features.shape[0]):
            starts = torch.randint(
                0, size - length + 1, (span_count,),
                device=features.device,
            )
            for start in starts:
                slices = [batch_index, slice(None), slice(None)]
                slices[dimension] = slice(start, start + length)
                features[tuple(slices)] = 0
        return features

    def forward(self, waveform, lengths=None):
        features, output_lengths = self.encoder.feature_extractor(
            waveform, lengths
        )
        hidden, attention_mask = self.encoder.encoder._preprocess(
            features, output_lengths
        )
        if self.training:
            hidden = hidden.clone()
            hidden = self._mask_spans(
                hidden, self.time_mask_probability,
                self.time_mask_length, dimension=1,
            )
            hidden = self._mask_spans(
                hidden, self.channel_mask_probability,
                self.channel_mask_length, dimension=2,
            )
        hidden = self.encoder.encoder.transformer(
            hidden, attention_mask=attention_mask
        )
        return self.classifier(hidden), output_lengths


class OriginalWav2Vec(STTModel):
    """Unmodified LibriSpeech Wav2Vec2 ASR baseline.

    The pretrained bundle uses a frequency-ordered vocabulary. Its output
    logits are reordered to the project's shared blank/space/apostrophe/a-z
    vocabulary before decoding and CTC evaluation.
    """

    uses_waveform_lengths = True

    def __init__(self):
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        labels = tuple(bundle.get_labels())
        shared_labels = ("-", "|", "'", *tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        missing = set(shared_labels) - set(labels)
        if missing:
            raise ValueError(
                f"Wav2Vec2 labels are missing required tokens: {sorted(missing)}"
            )
        self.register_buffer(
            "_shared_label_indices",
            torch.tensor(
                [labels.index(label) for label in shared_labels],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.model = bundle.get_model()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, waveform, lengths=None):
        logits, output_lengths = self.model(waveform, lengths)
        return logits.index_select(-1, self._shared_label_indices), output_lengths
