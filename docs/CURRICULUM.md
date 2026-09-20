# Curriculum

Thirty-three lessons, in six parts. Each row is one Markdown file under
`content/lessons/` and one directory under `content/exercises/`. The `Needs GPU`
column marks the exercises that run on RunPod rather than in the app container.

## Part 1 — Foundations

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `01-the-learning-problem` | The learning problem | Empirical risk minimisation, a held-out estimate of risk, and a simulated bias-variance decomposition | No |
| `02-linear-regression` | Linear regression | Least squares through normal equations, QR, and the SVD, and the conditioning that separates them | No |
| `03-gradient-descent` | Gradient descent | Exact line search, fixed steps, heavy ball, and Nesterov on a quadratic, against the predicted rates | No |
| `04-regularisation` | Ridge, lasso, and the proximal view | Ridge paths in closed form, soft thresholding, and ISTA and FISTA for the lasso | No |
| `05-logistic-regression` | Logistic regression | Stable log-sum-exp, the gradient and Hessian, gradient descent against Newton, and what separability does | No |
| `06-exponential-families` | Exponential families and GLMs | The natural parameterisation, the mean and variance from the log-partition, and Fisher scoring for Poisson regression | No |

## Part 2 — Classical machine learning

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `07-model-selection` | Generalisation and model selection | k-fold cross-validation, learning curves, and a Rademacher-style bound evaluated by sampling | No |
| `08-generative-classifiers` | Generative classifiers | Gaussian discriminant analysis, naive Bayes, and the discriminative counterpart they reduce to | No |
| `09-support-vector-machines` | Support vector machines | The soft-margin dual, the KKT conditions as a test, and a simplified SMO | No |
| `10-kernels` | Kernels and the representer theorem | Kernel ridge regression, the representer theorem checked numerically, and random Fourier features | No |
| `11-decision-trees` | Decision trees | CART with exact split search, impurity for both tasks, and cost-complexity pruning | No |
| `12-ensembles` | Bagging, forests, and boosting | The bagging variance identity, a random forest, and gradient boosting as functional gradient descent | No |
| `13-pca-and-svd` | PCA and the SVD | PCA three ways, power iteration, the Eckart-Young bound, and probabilistic PCA by EM | No |
| `14-clustering-and-em` | Clustering and expectation maximisation | Lloyd's algorithm, k-means++, EM for a Gaussian mixture, and the ELBO that proves it monotone | No |

## Part 3 — Neural networks

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `15-multilayer-perceptrons` | Multilayer perceptrons | A forward pass, a two-layer approximation of a sawtooth, and the width a given error needs | No |
| `16-backpropagation` | Backpropagation as reverse-mode AD | A tape-based autodiff engine with vector-Jacobian products, checked against finite differences | No |
| `17-optimisers` | Optimisers for deep networks | SGD with momentum, Adam, AdamW, and the decoupled weight-decay difference made visible | No |
| `18-initialisation-and-norm` | Initialisation and normalisation | Signal propagation under Xavier and He, and batch norm and layer norm forward and backward | No |
| `19-deep-regularisation` | Regularisation in deep networks | Inverted dropout forward and backward, label smoothing, and a double-descent curve | No |

## Part 4 — Vision and sequences

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `20-convolutions` | Convolutional networks | `conv2d` forward and backward through `im2col`, receptive fields, and pooling | No |
| `21-residual-networks` | Residual networks | A residual block, the gradient-flow argument, and a small ResNet trained on CIFAR-10 | Yes |
| `22-recurrent-networks` | Recurrent networks and BPTT | An RNN cell, truncated backpropagation through time, and the spectral radius that decides the gradient | No |
| `23-gated-recurrence` | LSTMs and GRUs | An LSTM cell forward and backward, and the constant error carousel demonstrated | No |
| `24-seq2seq-attention` | Sequence to sequence with attention | Additive and multiplicative attention, and the alignment a copy task produces | No |

## Part 5 — Transformers

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `25-self-attention` | Self-attention | Scaled dot-product attention, multiple heads, causal masking, and the variance argument for the scale | No |
| `26-the-transformer-block` | The transformer block | A full pre-norm block with SwiGLU, sinusoidal and rotary position information, and the residual stream | No |
| `27-training-a-transformer` | Training a transformer | A character-level GPT trained end to end, with warmup, clipping, and mixed precision | Yes |

## Part 6 — Large language models

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `28-tokenisation` | Tokenisation | Byte-level BPE training, encoding, decoding, and what the merge order costs | No |
| `29-scaling-laws` | Scaling laws | A FLOP and parameter counter for a transformer, and a compute-optimal frontier fitted from runs | No |
| `30-inference-and-decoding` | Inference and decoding | A KV cache, greedy, temperature, top-k and nucleus sampling, and speculative decoding's acceptance rule | No |
| `31-parameter-efficient-tuning` | Parameter-efficient fine-tuning | A LoRA layer, the merge that makes it free at inference, and the rank-versus-quality trade | No |
| `32-alignment` | Alignment: reward models, PPO, and DPO | The Bradley-Terry reward loss, the DPO objective derived from the KL-regularised optimum, and both gradients | No |
| `33-mixtures-of-experts` | Mixtures of experts | Top-k routing, the load-balancing auxiliary loss, capacity factors, and what dropping tokens costs | No |
