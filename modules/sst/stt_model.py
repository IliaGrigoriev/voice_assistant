from abc import ABC, abstractmethod

import torch.nn as nn


class STTModel(nn.Module, ABC):
    """Interface implemented by speech-to-text architectures."""

    uses_waveform_lengths = False

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, waveform):
        """Return character logits, optionally with output lengths."""
