# STT Code Parameters Explained

This document explains what the parameters in the basic PyTorch STT
model configure.

## 1. Mel spectrogram

``` python
self.mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=16_000,
    n_fft=400,
    hop_length=160,
    n_mels=80
)
```

### `sample_rate=16_000`

The audio contains 16,000 samples per second. LibriSpeech uses 16 kHz
audio.

### `n_fft=400`

Number of audio samples considered for one Fourier analysis window.

At 16 kHz:

\[ 400 / 16000 = 0.025s = 25ms \]

So we analyze roughly 25 ms chunks.

### `hop_length=160`

How far we move before calculating the next spectrum.

\[ 160 / 16000 = 10ms \]

For example:

``` text
window 1: samples   0–399
window 2: samples 160–559
window 3: samples 320–719
...
```

The windows overlap.

### `n_mels=80`

Compress the FFT frequency representation into **80 Mel frequency
bands**.

------------------------------------------------------------------------

## 2. First convolution

``` python
nn.Conv2d(
    in_channels=1,
    out_channels=32,
    kernel_size=3,
    stride=2,
    padding=1
)
```

### `in_channels=1`

The spectrogram is treated as a single-channel image:

``` text
[B, 1, 80, 1000]
     ↑
   channel
```

This is analogous to a grayscale image having one channel.

### `out_channels=32`

The CNN learns **32 different filters**, producing 32 feature maps.

``` text
1 input channel
       ↓
32 learned feature maps
```

`32` is a model-design choice. It could instead be 16, 64, etc.

### `kernel_size=3`

Each convolution filter examines a `3 × 3` region at once.

For the spectrogram, this means approximately:

``` text
3 frequency positions
×
3 time positions
```

### `stride=2`

The kernel moves two positions at a time rather than one.

This performs subsampling.

Approximately:

``` text
80 × 1000
    ↓
40 × 500
```

### `padding=1`

Adds one artificial cell around the edges.

Without padding, convolution would shrink the dimensions slightly more.

The convolution output size is:

\[ `\left`{=tex}`\lfloor`{=tex} `\frac{N+2P-K}{S}`{=tex}
`\right`{=tex}`\rfloor`{=tex}+1 \]

where:

``` text
K = 3    kernel size
S = 2    stride
P = 1    padding
```

------------------------------------------------------------------------

## 3. ReLU

``` python
nn.ReLU()
```

ReLU applies:

\[ f(x)=`\max`{=tex}(0,x) \]

For example:

``` text
-2.1 → 0
-0.4 → 0
 0.7 → 0.7
 3.2 → 3.2
```

It introduces nonlinearity into the neural network.

------------------------------------------------------------------------

## 4. Second convolution

``` python
nn.Conv2d(
    in_channels=32,
    out_channels=32,
    kernel_size=3,
    stride=2,
    padding=1
)
```

This is almost identical to the first convolution.

`in_channels=32` because the previous convolution produced 32 channels.

The second `stride=2` produces another 2× reduction:

``` text
Time:
1000 → 500 → 250

Frequency:
80 → 40 → 20
```

Together, the two convolution layers give roughly **4× temporal
subsampling**.

------------------------------------------------------------------------

## 5. CNN projection

``` python
self.cnn_projection = nn.Linear(
    32 * 20,
    256
)
```

`nn.Linear(input_size, output_size)` performs a learned linear
transformation.

After the CNN, each time step contains:

``` text
32 channels × 20 frequency positions
= 640 values
```

Therefore:

``` python
nn.Linear(640, 256)
```

means:

> Take those 640 values and learn a 256-dimensional representation.

The `256` becomes the model dimension:

``` python
d_model = 256
```

------------------------------------------------------------------------

## 6. Transformer layer

``` python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=256,
    nhead=4,
    dim_feedforward=1024,
    dropout=0.1,
    batch_first=True
)
```

### `d_model=256`

Every time step is represented by a vector containing 256 values.

The Transformer therefore receives:

``` text
[B, T, 256]
```

This is why the preceding CNN projection produces exactly 256 values.

### `nhead=4`

The Transformer uses four attention heads.

Because:

``` text
256 / 4 = 64
```

each head has a head dimension of 64.

Conceptually:

``` text
256-dimensional representation
           ↓
 ┌─────────┼─────────┬─────────┐
 ↓         ↓         ↓         ↓
head 1    head 2    head 3    head 4
 64        64        64        64
```

### `dim_feedforward=1024`

Each Transformer encoder block contains a feed-forward neural network.

Approximately:

``` text
256
 ↓
1024
 ↓
256
```

A traditional choice is:

\[ 4 `\times `{=tex}d\_{model} \]

and here:

\[ 4 `\times 256`{=tex} = 1024 \]

### `dropout=0.1`

During training, approximately 10% of certain activations are randomly
zeroed.

This is a regularization technique intended to reduce overfitting.

During inference:

``` python
model.eval()
```

dropout is automatically disabled.

### `batch_first=True`

Tells PyTorch that tensors are arranged as:

``` text
[batch, time, features]
```

For example:

``` text
[B, 250, 256]
```

rather than:

``` text
[time, batch, features]
```

------------------------------------------------------------------------

## 7. Number of Transformer layers

``` python
self.transformer = nn.TransformerEncoder(
    encoder_layer,
    num_layers=6
)
```

`encoder_layer` defines one Transformer encoder block.

`num_layers=6` creates six such blocks:

``` text
Transformer block
      ↓
Transformer block
      ↓
Transformer block
      ↓
Transformer block
      ↓
Transformer block
      ↓
Transformer block
```

Six layers is a model-size choice, not a mathematical requirement.

------------------------------------------------------------------------

## 8. Output layer

``` python
self.output = nn.Linear(
    256,
    vocab_size
)
```

This converts each 256-dimensional Transformer representation into one
score for every possible output token.

For example, the vocabulary might contain:

``` text
<blank>
a
b
c
...
z
space
apostrophe
```

If there are 29 tokens:

``` text
256 → 29
```

for every time step.

The resulting scores are called **logits**.

------------------------------------------------------------------------

## 9. CTC configuration

``` python
ctc_loss = nn.CTCLoss(
    blank=0,
    zero_infinity=True
)
```

### `blank=0`

Token ID `0` represents the special CTC blank.

For example:

``` text
0  = <blank>
1  = a
2  = b
3  = c
...
27 = space
28 = apostrophe
```

### `zero_infinity=True`

Sometimes a CTC alignment is impossible, for example when the target
transcription is too long relative to the number of input time steps.

This can result in infinite loss.

`zero_infinity=True` replaces those infinite losses and gradients with
zero rather than allowing `inf` or `NaN` values to disrupt training.

------------------------------------------------------------------------

## 10. AdamW optimizer

``` python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)
```

### `model.parameters()`

Passes all trainable parameters in the model to AdamW.

This includes parameters from:

-   CNN layers
-   CNN projection
-   Transformer
-   Q/K/V projections inside attention
-   feed-forward layers
-   output layer

### `lr=3e-4`

This is the learning rate.

\[ 3 `\times 10`{=tex}\^{-4} = 0.0003 \]

It controls the approximate scale of parameter updates during training.

------------------------------------------------------------------------

## Main model-design parameters

The most important values that we are choosing ourselves are:

``` python
n_mels = 80

cnn_channels = 32
cnn_kernel = 3
cnn_stride = 2

d_model = 256

attention_heads = 4
feedforward_size = 1024
transformer_layers = 6
dropout = 0.1

learning_rate = 3e-4
```

These are the main architectural and training hyperparameters that we
can later experiment with.

Other arguments mostly describe the input data format or tell PyTorch
how to perform the corresponding operation.
