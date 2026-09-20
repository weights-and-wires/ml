---
title: Training a transformer
slug: 27-training-a-transformer
part: "Part 5 — Transformers"
summary: The causal language-modelling objective, the AdamW recipe that fits it, and the flop count that says what a run costs.
minutes: 70
gpu: true
objectives:
  - Derive why an untrained model's loss sits at log V, and read perplexity as an effective branching factor.
  - "Justify each element of the AdamW training recipe: the warmup-into-cosine schedule, global-norm clipping, weight tying, and excluding norms and biases from decay."
  - Explain why fp16 needs loss scaling and bf16 does not, from the two formats' exponent widths.
  - Derive the 6ND training-flops estimate from the forward and backward passes.
prereqs:
  - 26-the-transformer-block
  - 17-optimisers
exercise: 27-training-a-transformer
---

# Training a transformer

A transformer block computes a function. Turning a stack of them into a
language model is an optimisation problem, and by the time models reach
billions of parameters and trillions of tokens, every choice in the training
recipe is load-bearing: the wrong learning-rate schedule wastes a training run
that costs real money, and the wrong numeric format silently corrupts
gradients without raising an error. This lesson derives the objective, the
sanity checks that catch a broken setup before it burns compute, and the
handful of recipe choices — an optimiser variant, a schedule, a clipping rule,
a tying decision, a numeric format — that every modern transformer training
run shares, together with the reasoning that makes each one non-negotiable
rather than a hyperparameter to sweep.

The lesson closes with the flop-counting argument that turns a parameter count
and a token count into an estimate of how much compute an entire run costs,
setting up the compute-optimal tradeoffs of lesson 29.

## What you already know

- The transformer block computes $\operatorname{Attn}$ and a feed-forward
  network, wrapped in residual connections (lesson 26).
- Cross-entropy loss for a categorical prediction is $-\log p(\text{true class})$.
- AdamW maintains first and second moment estimates of the gradient, applies
  bias correction, and decays weights independently of the gradient step
  (lesson 17).
- A matrix multiply of shapes $(m,k)$ by $(k,n)$ costs $2mnk$ flops, and a
  transformer block's forward cost is about twice its parameter count, in
  flops per token (lesson 26).
- IEEE floating-point formats split their bits between an exponent, which sets
  the representable range, and a mantissa, which sets the precision within
  that range.

## The objective

A causal language model factors the joint probability of a token sequence
$x_1, \dots, x_T$ autoregressively:

$$
p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t}) .
$$

Maximising the log-likelihood of the data is the same problem as minimising
the average cross-entropy between the model's predicted next-token
distribution and the token that actually came next, at every position:

$$
L = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_{t+1} \mid x_{\le t}) .
$$

Training feeds the model the true prefix $x_{\le t}$ at every position rather
than its own previous predictions — **teacher forcing** — which is what makes
this a single forward pass over the whole sequence rather than $T$ sequential
ones: the model computes logits for every position in parallel, and the loss
at position $t$ compares those logits against $x_{t+1}$, the sequence shifted
by one. Nothing about teacher forcing evaluates whether the model can sustain
its own predictions across many steps; that gap between training on ground
truth and generating from its own output is exposure bias, a separate concern
from the objective itself.

### The sanity anchor

At initialisation, a well-behaved model's logits are near zero for every
token, so its predicted distribution over the vocabulary $V$ is close to
uniform. The cross-entropy of a uniform distribution against any single true
class is $-\log(1/V) = \log V$, independent of which class is true. A training
run whose first loss reading is not within a small margin of $\log V$ has a
bug somewhere before the loss — an output layer that is not near-zero at
initialisation, a vocabulary size mismatched between the data pipeline and the
model, or logits scaled by an accidental factor — and it is worth checking
before spending a single step of compute chasing it later.

### Perplexity

Perplexity is $\exp(L)$, the loss exponentiated back out of log space. Writing
the loss as an average of per-token terms and exponentiating turns the sum
into a product:

$$
\operatorname{PP} = \exp\!\left(\frac{1}{T}\sum_t -\log p_t\right)
= \left(\prod_t \frac{1}{p_t}\right)^{1/T} ,
$$

the geometric mean of each step's reciprocal probability. If a model assigned
probability exactly $1/k$ to the true token at every step — as an untrained
model does with $k = V$ — perplexity would equal $k$ exactly, by the sanity
anchor above. That reading survives approximately away from that special
case: a perplexity of $20$ means the model is, on average, as uncertain at
each step as a uniform choice among twenty tokens, even though no real model
is uniform over any twenty-token subset. Perplexity is reported instead of raw
loss because it is dimensionless in tokens and comparable across tokenisations
with different vocabulary sizes only when those vocabularies are close in
size; comparing perplexities across very different tokenisers compares
different branching factors, not different models.

## The recipe

**AdamW with $\beta_2 = 0.95$.** The second moment estimate's exponential
moving average has an effective memory of roughly $1/(1-\beta_2)$ steps: the
default $\beta_2 = 0.999$ integrates gradient statistics over about a thousand
steps, which is far longer than the gradient distribution stays stationary
under the large batches and aggressive learning rates language-model
pretraining uses. Lowering $\beta_2$ to $0.95$ shortens that memory to about
twenty steps, letting the second-moment estimate track a gradient scale that
is shifting every few hundred steps rather than smoothing over it.

**Linear warmup into cosine decay.** Adam's bias-corrected second moment,
$\hat{v}_t = v_t / (1-\beta_2^t)$, divides by a quantity that starts at
$1-\beta_2 \approx 0.05$ and grows to $1$: at step one, a single noisy gradient
sample is divided by a small number and can swing the effective step size by
more than an order of magnitude relative to its steady-state value. A linear
ramp from zero keeps the learning rate small while $t$ is small and the
correction is least trustworthy, then hands off to a smooth cosine decay from
the peak rate down to a floor across the remaining, fixed compute budget. The
schedule is smooth by construction — no discontinuous drop the optimiser has
to absorb in one step — and, unlike a fixed decay rate, it is defined relative
to a known total step count, so the anneal always finishes exactly as the
budget runs out.

**Global-norm clipping at $1.0$.** Before the optimiser step, compute the norm
of the gradient across every parameter, $g = \big(\sum_i \lVert \nabla_i
\rVert^2\big)^{1/2}$, and rescale the whole gradient by $\min(1, \, 1/g)$ when
it exceeds the threshold. Rescaling every coordinate by the same factor
preserves the gradient's direction and only shrinks its magnitude, unlike
clipping each coordinate independently, which would distort the direction of
the step. Typical gradient norms during stable training sit below the
threshold, so clipping is a no-op almost always and only intervenes on the
rare spike — a batch dominated by a rare token, or a numerically large
logit — that would otherwise translate into a destructive update at the
current learning rate.

**Weight tying.** The input embedding table and the output projection to
logits are both linear maps between the vocabulary and $d_{\text{model}}$;
tying sets them to the same matrix (transposed for the output side), which
removes an entire $V \times d_{\text{model}}$ block of parameters and is
motivated by more than the saved memory: a token's input embedding and the
direction that makes it likely as an output play complementary roles in the
same geometry, and sharing the matrix forces that geometry to serve both
directions at once.

**No decay on norms and biases.** Weight decay is an isotropic prior pulling
every weight toward zero, appropriate for the large matrices that do the
model's feature transformations, where shrinkage is a real regulariser.
Normalisation gains are initialised at one and biases at zero specifically so
that a sublayer starts close to the identity; decaying a norm gain toward zero
directly fights that initialisation; and biases and gains together are a tiny
fraction of the parameter count, so excluding them costs nothing towards
regularising the model as a whole. Implementations split parameters into two
groups by whether a tensor has more than one dimension and decay only the
matrices.

## Numeric precision

Training in fp32 throughout is safe and slow. Two lower-precision formats
trade some of that safety for throughput, and they fail differently.

| Format | Exponent bits | Mantissa bits | Representable range |
|---|---|---|---|
| fp32 | 8 | 23 | $\approx 10^{-38}$ to $10^{38}$ |
| fp16 | 5 | 10 | $\approx 6\times 10^{-5}$ to $65504$ |
| bf16 | 8 | 7 | $\approx 10^{-38}$ to $10^{38}$ |

bf16 keeps fp32's 8-bit exponent and truncates only the mantissa, so it spans
the same range as fp32 at coarser precision within that range. fp16 keeps more
mantissa bits but narrows the exponent to 5 bits, and gradients in a deep
network routinely fall below fp16's smallest representable magnitude,
particularly late in training or in layers many blocks from the loss —
values that bf16 still represents, at reduced precision, and that fp16
flushes to exact zero, deleting the gradient signal outright.

**Loss scaling** is the fix for fp16's range problem specifically. Multiplying
the loss by a constant $s$ before calling backward multiplies every gradient
by the same $s$, by linearity of differentiation, shifting small gradients up
into fp16's representable range before they underflow; dividing the resulting
gradients by $s$ again before the optimiser step restores the correct scale
after the numbers have survived the pass through fp16 storage. bf16 needs no
such correction, because nothing in its exponent range is at risk of
underflowing in the first place — the entire technique exists to patch a
narrow-exponent format, not a low-precision one.

## Fitting a bigger step in fixed memory

**Gradient accumulation** runs $k$ microbatches through the forward and
backward pass, summing (or averaging) their gradients without an optimiser
step in between, then takes one step against the accumulated gradient. The
result is mathematically the same update an actual batch of size
$k \times \text{microbatch}$ would produce, at the cost of $k$ forward and
backward passes instead of one, because only one microbatch's activations
occupy memory at a time. The effective batch size a run trains at is the
microbatch size times the accumulation steps times the number of parallel
devices, and it is this effective size, not the microbatch size, that the
learning-rate schedule and the warmup length are tuned against.

**Activation checkpointing** trades the same currency the other way.
A standard backward pass needs every layer's forward activations to compute
local gradients, so peak memory grows with depth. Checkpointing stores only a
subset of activations — at block boundaries, typically — and recomputes the
rest by re-running the forward pass for that segment during the backward pass.
It costs roughly one extra forward pass in compute, since every recomputed
segment is a repeated forward computation, in exchange for memory that no
longer scales with the full depth, which is what makes it possible to fit a
deeper or wider model in the same device memory a shallower one needed.

## The compute of a training run

A block's forward pass costs about two flops per parameter per token — one
multiply and one add through each weight the token passes through, as derived
in lesson 26. The backward pass computes gradients with respect to both the
layer's inputs, to keep propagating error to earlier layers, and its weights,
to know how to update them; each of those is itself a matrix multiply the same
size as the forward one, so backward costs about twice the forward pass:
$4N$ flops per token against the forward's $2N$. Summed,

$$
C \approx (2N + 4N) \cdot D = 6ND,
$$

flops for a model of $N$ non-embedding parameters trained on $D$ tokens. This
ignores the attention score computation's $O(T)$ term per token, exactly as
lesson 26's per-block estimate does, and it ignores every elementwise
operation — activation functions, normalisation, the softmax — because a
matrix multiply's flops dominate an elementwise pass over the same tensor by a
factor of the matrix's inner dimension.

## Cost and scaling

| Quantity | Scaling |
|---|---|
| Forward flops per token | $\approx 2N$ |
| Backward flops per token | $\approx 4N$ |
| Total training flops | $\approx 6ND$ |
| Optimiser state memory (AdamW) | $2N$ extra values ($m$ and $v$), on top of $N$ parameters and $N$ gradients |
| Effective batch size | microbatch $\times$ accumulation steps $\times$ devices |

AdamW's memory cost is easy to undercount: a naive fp32 implementation holds
four copies of every parameter — the parameter, its gradient, and the two
moment estimates — before any activation memory is counted, which is why
optimiser state, not parameters, is often the first thing to run out of room
on a single device.

## Where this breaks

**A loss that never approaches $\log V$.** The single most common early
failure, and the sanity anchor exists to catch it before a run is left
training on a broken setup for hours.

**fp16 without loss scaling.** Gradients underflow silently — no error, no
NaN, only a model that stops learning in the layers whose gradients vanished
first, usually the earliest ones.

**Clipping set too aggressively.** A clip threshold well below the typical
gradient norm turns every step into a fixed-magnitude update in the
gradient's direction, discarding the information the gradient's magnitude
carries about how far to move; clipping should intervene on outliers, not on
every step.

**Decaying norms and biases anyway.** A framework's default weight-decay group
sometimes ignores parameter dimensionality; decaying a layer norm's gain
towards zero shrinks that sublayer's output on every forward pass, and the
symptom looks like an unrelated instability far from its cause.

## The exercise

You implement the objective, the schedule, the clipping rule, and one AdamW
step, then train a tiny transformer with them. Write `causal_lm_loss(logits,
targets)`, `perplexity(loss)`, `warmup_cosine_lr(step, warmup, total, peak,
floor)`, `clip_global_norm(grads, max_norm)`, `training_flops(params, tokens)`,
and `train_step(model_fn, params, batch, opt_state, config)`, which runs one
forward pass, computes the loss, backpropagates, clips the gradient, and
applies one decoupled-weight-decay AdamW update using the learning rate the
schedule gives the current step. Build a small causal transformer from
lesson 26's pieces in this file's `__main__` block, training it on a short,
repeating, built-in character string for a few hundred steps.

The checks verify that the loss on uniform logits equals $\log V$ exactly,
that perplexity is the exponential of the loss, that the schedule agrees with
both of its pieces at the warmup boundary and reaches its floor exactly at the
final step, that clipping produces exactly the threshold norm when the
gradient exceeds it and changes nothing when it does not, that the flop
formula matches a hand computation, and that twenty steps of `train_step` on a
batch small enough to memorise drive the loss well below $\log V$.
