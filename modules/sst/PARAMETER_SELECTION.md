# Parameter Selection and Literature Rationale

This document defines how training parameters are selected for the four
speech-to-text experiments. The code implements these starting configurations
under protocol version `v1`; the search spaces define subsequent Optuna
work rather than values applied simultaneously.

## Selection principles

Parameters are chosen in three stages:

1. use published Wav2Vec2 and Conformer recipes to define defensible starting
   values;
2. reject unstable configurations with short pilot runs;
3. optimize the remaining ranges using only the fixed MedDialog validation
   split.

MedDialog test data and LibriSpeech `test-clean` must not influence parameter
selection. The primary selection metric is MedDialog validation WER, with CER
as a tie-breaker. CTC loss is diagnostic rather than the selection objective,
because a blank-collapsed model can have a finite loss while producing empty
transcriptions.

## Reference papers

### Wav2Vec2 fine-tuning

Baevski et al. add a randomly initialized output projection, train only that
projection for the first 10,000 updates, keep the convolutional feature encoder
frozen during fine-tuning, and use a tri-stage learning-rate schedule. Their
lower-resource experiments compare learning rates of `2e-5` and `3e-5`; the
960-hour BASE experiment uses `1e-4` together with a much larger effective
audio batch than this project can fit on one GPU.

- Baevski et al., [*wav2vec 2.0: A Framework for Self-Supervised Learning of
  Speech Representations*](https://proceedings.neurips.cc/paper_files/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf), NeurIPS 2020.
- The corresponding [Fairseq Wav2Vec2 ASR
  implementation](https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2_asr.py)
  exposes classifier-only warmup, feature-gradient scaling, time/channel
  masking, dropout, and LayerDrop.

The original recipe also applies time and channel masking during fine-tuning,
following the same general regularization idea as SpecAugment:

- Park et al., [*SpecAugment: A Simple Data Augmentation Method for Automatic
  Speech Recognition*](https://arxiv.org/abs/1904.08779), Interspeech 2019.

### Domain adaptation

Domain-specific Wav2Vec2 experiments show that adaptation must be selected on
in-domain development data and that a fine-tuned system does not automatically
improve every target condition:

- Ferreira and Oliveira, [*Domain Specific Wav2vec 2.0 Fine-tuning For The
  SE&R 2022 Challenge*](https://arxiv.org/abs/2207.14418), 2022.

This supports using MedDialog validation WER as the primary objective and
reporting LibriSpeech only as a retention measurement.

### Conformer training

The custom model follows the convolution-plus-attention design introduced by:

- Gulati et al., [*Conformer: Convolution-augmented Transformer for Speech
  Recognition*](https://arxiv.org/abs/2005.08100), Interspeech 2020.

Domain-adaptation work on Conformer systems also reports that source/target
utterance characteristics affect the best architecture and fine-tuning
parameters:

- Wang et al., [*Hyper-parameter Adaptation of Conformer ASR Systems for
  Elderly and Dysarthric Speech Recognition*](https://arxiv.org/abs/2306.15265),
  Interspeech 2023.

### Optimizer and objective

- Loshchilov and Hutter, [*Decoupled Weight Decay
  Regularization*](https://openreview.net/forum?id=Bkg6RiCqY7), ICLR 2019,
  motivates AdamW rather than coupling L2 regularization to Adam updates.
- Graves et al., [*Connectionist Temporal Classification: Labelling
  Unsegmented Sequence Data with Recurrent Neural
  Networks*](https://www.cs.toronto.edu/~graves/icml_2006.pdf), ICML 2006,
  defines the CTC training objective used by every trainable model.

## Experiment 1: original Wav2Vec2

This experiment has no training parameters. It evaluates the unchanged
`WAV2VEC2_ASR_BASE_960H` checkpoint with the shared normalization and greedy
CTC decoder.

Evaluation:

- MedDialog test: baseline medical-domain accuracy;
- LibriSpeech `test-clean`: reference general-domain accuracy;
- inference latency, model size, parameter count, and GPU allocation.

## Experiment 2: frozen Wav2Vec2 with a new head

“Frozen” means that the complete pretrained encoder has
`requires_grad = False`; only the new character-level CTC head is optimized.

### Starting configuration

| Parameter | Choice | Rationale |
| --- | ---: | --- |
| Head learning rate | `1e-4` | A new linear head can tolerate a larger rate than a pretrained encoder |
| Weight decay | `1e-4` | Conservative AdamW regularization |
| Gradient clipping | `1.0` | Limits unstable CTC updates |
| Batch size | `2` | Fits long MedDialog recordings on the available GPU |
| Gradient accumulation | `8` | Raises the effective batch without increasing activation memory |
| Warmup fraction | `0.10` | Avoids an abrupt initial update |
| Maximum epochs | `10` | Upper bound; validation WER controls early stopping |

### Search space

```text
head learning rate:      3e-5, 1e-4, 3e-4
weight decay:            0, 1e-4, 1e-3, 1e-2
gradient accumulation:   4, 8, 16
warmup fraction:         0.0, 0.05, 0.10
```

## Experiment 3: fully fine-tuned Wav2Vec2

The failed run that used one `1e-4` learning rate for the entire model produced
`WER = CER = 1.0` for every epoch. Its training loss barely changed and its
validation loss later diverged. That configuration must not be reused.

The replacement protocol separates the randomly initialized head from the
pretrained acoustic network.

### Stage A: classifier-only adaptation

- Freeze the convolutional feature extractor and Transformer.
- Train only the new CTC head.
- Use head learning rate `1e-4`.
- Run for one MedDialog epoch as a practical single-GPU warmup.

The one-epoch choice is a compute-aware approximation, not a reproduction of
the paper’s 10,000 optimizer updates. With gradient accumulation, minibatches
and optimizer updates must be reported separately.

### Stage B: conservative Transformer adaptation

- Keep the convolutional feature extractor frozen permanently.
- Unfreeze the Transformer context network.
- Continue training the CTC head.
- Use separate optimizer parameter groups.

| Parameter | Starting choice | Rationale |
| --- | ---: | --- |
| Head learning rate | `1e-4` | The new projection still learns faster |
| Transformer learning rate | `1e-5` | Conservative update to pretrained ASR representations |
| Feature-extractor learning rate | `0` | Matches the original Wav2Vec2 freezing strategy |
| Gradient accumulation | `16` | About 32 recordings per optimizer update |
| Gradient clipping | `1.0` | Protects against unstable full-model gradients |
| Warmup | First `10%` of all fine-tuning updates | Original fine-tuning recipe uses warmup; with ten epochs this coincides with Stage A |
| Constant phase | Next `40%` | Preserves a useful peak learning rate |
| Linear decay | Final `50%` | Matches the published tri-stage schedule |
| Time-mask probability | `0.05` | Published 100-hour Wav2Vec2 setting |
| Time-mask length | `10` | Published fine-tuning span length |
| Channel-mask probability | `0.008` | Published 100-hour setting |

At an average duration near 23 seconds, batch size 2 with accumulation 16 is
approximately 736 seconds of audio per optimizer update. This is still below
the roughly 1,600-second effective batch reported for Wav2Vec2 BASE, but is far
closer than updating on each two-recording minibatch.

### Search space

```text
Transformer learning rate: 5e-6, 1e-5, 2e-5, 3e-5
head learning rate:        5e-5, 1e-4, 3e-4
head-only duration:        0.5, 1.0, 2.0 epochs
gradient accumulation:    8, 16
warmup fraction:           0.05, 0.10
weight decay:              1e-4, 1e-3, 1e-2
```

## Experiment 4: Conformer from scratch

The Conformer is initialized randomly and trained only on MedDialog. It is
evaluated only on MedDialog because the Wav2Vec2 checkpoint has already seen
substantially more LibriSpeech ASR training; ranking a medical-only Conformer on
LibriSpeech would not be a balanced comparison.

Architecture parameters remain fixed during the first optimization study:

```text
layers:             6
attention heads:    4
hidden dimension:   256
FFN dimension:      1024
convolution kernel: 31
```

Keeping the architecture fixed separates optimizer selection from neural
architecture search.

### Starting configuration

| Parameter | Choice |
| --- | ---: |
| Learning rate | `3e-4` |
| Weight decay | `1e-3` |
| Dropout | `0.1` |
| Warmup fraction | `0.10` |
| Gradient clipping | `1.0` |
| Gradient accumulation | `8` |
| Maximum epochs | `50` |
| Early-stopping patience | `5` validation epochs |

### Search space

```text
learning rate:          1e-4 to 1e-3, logarithmic
weight decay:           1e-6 to 1e-2, logarithmic
dropout:                0.05, 0.10, 0.20, 0.30
gradient accumulation:  4, 8, 16
warmup fraction:        0.05, 0.10
```

## Pilot and pruning protocol

Long failed runs should be avoided. Each candidate first receives a short
pilot:

1. train for 500–1,000 optimizer updates;
2. decode the same small, fixed validation subset;
3. record the blank-token proportion and several reference/prediction pairs;
4. prune if predictions are almost entirely empty, loss is non-finite, or GPU
   memory is exhausted;
5. continue promising trials for one complete epoch before allocating the full
   budget.

Optuna trials are ranked by MedDialog validation WER. Median pruning may be
used only after the classifier-only phase, because early WER during that phase
is not directly comparable with an already-unfrozen trial.

## Final model selection and reporting

For each trainable experiment:

1. select hyperparameters using MedDialog validation WER only;
2. use validation CER to break near-ties;
3. retrain the selected configuration using three random seeds where compute
   permits;
4. report mean and standard deviation;
5. evaluate the selected final checkpoints once on their designated test sets.

The final tables are:

- **Medical comparison:** all four models on MedDialog test;
- **Retention comparison:** original, frozen, and fully fine-tuned Wav2Vec2 on
  LibriSpeech `test-clean`;
- **Efficiency comparison:** training time, peak GPU allocation, inference
  latency, trainable parameters, and model size.

The Conformer is intentionally absent from the LibriSpeech retention table.
