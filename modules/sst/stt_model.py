import torch
import torch.nn as nn
import torchaudio


class STTModel(nn.Module):

    # Constructor
    def __init__(self, vocab_size, hop_length, device):
        super().__init__()
        self._device = device
        self._mel = self.audio_to_log_mel(hop_length)
        self._cnn, self._cnn_projection = self.build_cnn()
        self._conformer = self.conformer_layer()
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
    # 3. Conformer encoder
    # --------------------------------------------------
    def conformer_layer(self):
        return torchaudio.models.Conformer(
            input_dim                  = 256,
            num_heads                  = 4,
            ffn_dim                    = 1024,
            num_layers                 = 6,
            depthwise_conv_kernel_size = 31,
            dropout                    = 0.1
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
        # ----- Conformer -----
        input_lengths = torch.full(
            (B,),
            T,
            dtype  = torch.long,
            device = self._device
        )
        x, _ = self._conformer(x, input_lengths)

        # [B, T/4, 256]

        # ----- Character prediction -----
        logits = self._output(x)

        # [B, T/4, vocab_size] # [batch, time, vocab_size]
        return logits
