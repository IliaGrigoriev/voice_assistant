import torch
import torch.nn as nn
import torchaudio


class STTModel(nn.Module):

    # Constructor
    def __init__(self, vocab_size, hop_length):
        super().__init__()
        self._mel = self.audio_to_log_mel(hop_length)
        self._cnn, self._cnn_projection = self.build_cnn()
        self._transformer = self.transformer()
        self._output = self.hidden_to_char_logits(vocab_size)

    # --------------------------------------------------
    # 1. Audio → Log-Mel spectrogram
    # --------------------------------------------------
    def audio_to_log_mel(self, hop_length):
        return torchaudio.transforms.MelSpectrogram(
            sample_rate = 16_000,
            n_fft       = 400, # 25 ms
            hop_length  = hop_length,
            n_mels      = 80
        )

    # --------------------------------------------------
    # 2. CNN: local features + 4× subsampling
    # --------------------------------------------------
    def build_cnn(self):
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels  = 1,
                out_channels = 32,
                kernel_size  = 3,
                stride       = 2,
                padding      = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels  = 32,
                out_channels = 32,
                kernel_size  = 3,
                stride       = 2,
                padding      = 1
            ),
            nn.ReLU()
        )

        # 80 Mel bins → ~20 after 4×subsampling
        cnn_projection = nn.Linear(
            32 * 20,
            256
        )
        return cnn, cnn_projection

    # --------------------------------------------------
    # 3. Transformer
    # --------------------------------------------------
    def transformer(self):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = 256,
            nhead           = 4,
            dim_feedforward = 1024,
            dropout         = 0.1,
            batch_first     = True
        )

        return nn.TransformerEncoder(
            encoder_layer,
            num_layers = 6
        )

    # --------------------------------------------------
    # 4. Hidden representation → character logits
    # --------------------------------------------------
    def hidden_to_char_logits(self, vocab_size):
        return nn.Linear(
            256,
            vocab_size
        )

    # --------------------------------------------------
    # 5. Forward
    # --------------------------------------------------
    def forward(self, waveform):

        # waveform:
        # [B, audio_samples]
        # ----- Log-Mel -----
        x = self._mel(waveform)

        # [B, 80, T]
        x = torch.log(x + 1e-6)

        # CNN expects:
        # [batch, channels, height, width]
        x = x.unsqueeze(1)

        # [B, 1, 80, T]
        # ----- CNN -----

        x = self._cnn(x)

        # approximately:
        # [B, 32, 20, T/4]

        # Put time dimension second
        x = x.permute(0, 3, 1, 2)

        # [B, T/4, 32, 20]
        B, T, C, F = x.shape
        x = x.reshape(B, T, C * F)

        # [B, T/4, 640]
        x = self._cnn_projection(x)

        # [B, T/4, 256]
        # ----- Transformer -----
        x = self._transformer(x)

        # [B, T/4, 256]

        # ----- Character prediction -----
        logits = self._output(x)

        # [B, T/4, vocab_size] # [batch, time, vocab_size]
        return logits
