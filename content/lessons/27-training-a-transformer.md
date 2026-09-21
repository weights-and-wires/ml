---
title: Training a transformer
slug: 27-training-a-transformer
part: "Part 5 — Transformers"
summary: "The causal language-modelling objective, the AdamW recipe that fits it, and the flops that say what a run costs."
minutes: 60
gpu: true
objectives:
  - Explain why an untrained model's loss sits at log V, and read perplexity as an effective branching factor.
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
language model is an optimisation problem. Once a model reaches billions of
parameters and trillions of tokens, every choice in the training recipe
carries weight. The wrong learning-rate schedule wastes a run that costs
real money, and the wrong numeric format corrupts gradients without raising
an error. This lesson derives the objective, the sanity check that catches a
broken setup before it burns compute, and the handful of recipe choices that
every modern transformer training run shares. It also gives the reasoning
that makes each one non-negotiable rather than a hyperparameter to sweep. It closes
with the flop-counting argument that turns a parameter count and a token
count into an estimate of what a run costs, setting up the compute-optimal
trade-offs of lesson 29.

## What you already know

- The transformer block computes $\operatorname{Attn}$ and a feed-forward
  network, wrapped in residual connections (lesson 26).
- Cross-entropy loss for a categorical prediction is $-\log p(\text{true class})$.
- AdamW maintains first and second moment estimates of the gradient, applies
  bias correction, and decays weights independently of the gradient step
  (lesson 17).
- A transformer block's forward cost is about twice its parameter count, in
  flops per token (lesson 26).
- IEEE floating-point formats split their bits between an exponent, which
  sets the representable range, and a mantissa, which sets the precision
  within that range.

## The objective

A causal language model factors the joint probability of a token sequence
$x_1, \dots, x_T$ autoregressively:

$$
p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t}) .
$$

Maximising that log-likelihood is the same problem as minimising the average
cross-entropy between the model's predicted next-token distribution and the
token that actually came next, at every position:

$$
L = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_{t+1} \mid x_{\le t}) .
$$

Training feeds the model the true prefix $x_{\le t}$ at every position rather
than its own previous predictions — **teacher forcing**. The model
therefore computes logits for every position in one forward pass, and the
loss at position $t$ compares those logits against $x_{t+1}$, the sequence
shifted by one. Teacher forcing never checks whether the model can sustain its own
predictions across many steps; that gap between training on ground truth and
generating from the model's own output is exposure bias, a separate concern
from the objective.

### The sanity anchor

At initialisation, a well-behaved model's logits sit near zero for every
token, so its predicted distribution over the vocabulary $V$ is close to
uniform. The cross-entropy of a uniform distribution against any single true
class is $-\log(1/V) = \log V$, regardless of which class is true. A training
run whose first loss reading is not close to $\log V$ has a bug before the
loss. It might be an output layer that is not near-zero at initialisation, a
mismatched vocabulary size, or logits scaled by an accidental factor — worth
checking before a single step of compute goes into chasing it later.

### Perplexity

Perplexity is $\exp(L)$, the loss carried back out of log space. Because it
is the exponential of an average of per-token log-probabilities, it equals
the geometric mean of each step's reciprocal probability. A model assigning
probability exactly $1/k$ to the true token at every step, as an untrained
model does with $k = V$, has perplexity exactly $k$. That reading survives
loosely away from that special case: a perplexity of $20$ means the model is,
on average, about as uncertain at each step as a uniform choice among twenty
tokens. Report perplexity instead of raw loss for comparisons across
vocabularies of similar size; comparing perplexities across very different
tokenisers compares different branching factors, not different models.

## The recipe

**AdamW with $\beta_2 = 0.95$.** The second-moment estimate's exponential
moving average has an effective memory of about $1/(1-\beta_2)$ steps. The
default $\beta_2 = 0.999$ integrates gradient statistics over roughly a
thousand steps, far longer than the gradient distribution stays stationary
under the large batches and aggressive learning rates pretraining uses.
Lowering $\beta_2$ to $0.95$ shortens that memory to about twenty steps, so
the second-moment estimate tracks a gradient scale that shifts every few
hundred steps.

**Linear warmup into cosine decay.** Adam's bias-corrected second moment
divides by a quantity that starts small and grows to one. At step one, a
single noisy gradient sample can therefore swing the effective step size by
an order of magnitude relative to its steady-state value. A linear ramp from zero keeps
the learning rate small while that correction is least trustworthy, then
hands off to a smooth cosine decay from the peak rate to a floor across the
remaining, fixed compute budget. The schedule is continuous by construction,
and because it is defined relative to a known total step count, the anneal
finishes exactly as the budget runs out.

**Global-norm clipping at $1.0$.** Before the optimiser step, compute the
norm of the gradient across every parameter, $g = \big(\sum_i \lVert
\nabla_i \rVert^2\big)^{1/2}$, and rescale the whole gradient by
$\min(1, 1/g)$ when it exceeds the threshold. Rescaling every coordinate by
the same factor preserves the gradient's direction and only shrinks its
magnitude, unlike clipping each coordinate independently, which would distort
the step's direction. Gradient norms during stable training sit below the
threshold almost always, so clipping only intervenes on the rare spike that
would otherwise become a destructive update at the current learning rate.

**Weight tying.** The input embedding table and the output projection to
logits are both linear maps between the vocabulary and $d_{\text{model}}$, so
tying sets them to the same matrix, transposed for the output side. Beyond
removing an entire $V \times d_{\text{model}}$ block of parameters, a token's
input embedding and the direction that makes it likely as an output play
complementary roles in the same geometry, and sharing the matrix forces that
geometry to serve both.

**No decay on norms and biases.** Weight decay pulls every weight toward
zero, appropriate for the large matrices that do the model's feature
transformations. Normalisation gains start at one and biases at zero so a
sublayer starts close to the identity, and decaying a gain toward zero fights
that initialisation directly. Biases and gains together are a tiny fraction
of the parameter count, so excluding them costs nothing toward regularising
the model. Implementations split parameters into two groups by whether a
tensor has more than one dimension, and decay only the matrices.

## Numeric precision

Training in fp32 throughout is safe and slow. Two lower-precision formats
trade safety for throughput, and they fail differently.

| Format | Exponent bits | Mantissa bits | Representable range |
|---|---|---|---|
| fp32 | 8 | 23 | $\approx 10^{-38}$ to $10^{38}$ |
| fp16 | 5 | 10 | $\approx 6\times 10^{-5}$ to $65504$ |
| bf16 | 8 | 7 | $\approx 10^{-38}$ to $10^{38}$ |

bf16 keeps fp32's 8-bit exponent and truncates only the mantissa, so it spans
the same range as fp32 at coarser precision. fp16 keeps more mantissa bits but
narrows the exponent to 5 bits. Gradients in a deep network routinely fall
below fp16's smallest representable magnitude, particularly late in training
or in layers many blocks from the loss. bf16 still represents those values,
at reduced precision. fp16 instead flushes them to exact zero, deleting the
gradient signal outright.

**Loss scaling** fixes fp16's range problem specifically. Multiplying the
loss by a constant $s$ before calling backward multiplies every gradient by
that same $s$, by linearity of differentiation. This shifts small gradients
up into fp16's representable range before they underflow. Dividing the
gradients by $s$ again before the optimiser step restores the correct
scale. bf16 needs
no such correction, because nothing in its exponent range risks underflowing
in the first place — the technique patches a narrow-exponent format, not a
low-precision one.

**Gradient accumulation** buys a wider effective batch with extra compute
instead of extra memory: it sums the gradients from $k$ microbatches before
one optimiser step, matching the update a real batch of size $k \times
\text{microbatch}$ would produce. It is this effective batch size —
microbatch times accumulation steps times devices — that a schedule's warmup
length is tuned against.

## The compute of a training run

A block's forward pass costs about two flops per parameter per token, one
multiply and one add through each weight the token passes through, as derived
in lesson 26. The backward pass computes gradients for both the layer's
inputs, to keep propagating error backward, and its weights, to know how to
update them. Each of these is a matrix multiply the same size as the forward
one, so backward costs about twice the forward pass: $4N$ flops per token
against the forward's $2N$. Summed,

$$
C \approx (2N + 4N) \cdot D = 6ND,
$$

flops for a model of $N$ non-embedding parameters trained on $D$ tokens. This
ignores attention's $O(T)$ term per token, exactly as lesson 26's per-block
estimate does. It also ignores every elementwise operation, because a matrix
multiply's flops dominate an elementwise pass over the same tensor by a
factor of the matrix's inner dimension.

## Cost and scaling

| Quantity | Scaling |
|---|---|
| Forward flops per token | $\approx 2N$ |
| Backward flops per token | $\approx 4N$ |
| Total training flops | $\approx 6ND$ |
| Optimiser state memory (AdamW) | $2N$ extra values ($m$ and $v$), on top of $N$ parameters and $N$ gradients |
| Effective batch size | microbatch $\times$ accumulation steps $\times$ devices |

AdamW's memory cost is easy to undercount. A naive fp32 implementation holds
four copies of every parameter — the parameter, its gradient, and the two
moment estimates — before any activation memory is counted. That is why
optimiser state, not parameters, is often the first thing to run out of room
on a single device.

## Where this breaks

**A loss that never approaches $\log V$.** The single most common early
failure, which the sanity anchor exists to catch before a run trains for
hours on a broken setup.

**fp16 without loss scaling.** Gradients underflow silently — no error, no
NaN, only a model that stops learning in the layers whose gradients vanished
first, usually the earliest ones.

**Clipping set too aggressively.** A clip threshold well below the typical
gradient norm turns every step into a fixed-magnitude update, discarding the
information the gradient's magnitude carries about how far to move. Clipping
should intervene on outliers, not on every step.

**Decaying norms and biases anyway.** A framework's default weight-decay
group sometimes ignores parameter dimensionality. Decaying a layer norm's
gain toward zero then shrinks that sublayer's output on every forward pass —
a symptom that looks like an unrelated instability far from its cause.

## The exercise

You implement the objective, the schedule, the clipping rule, and one AdamW
step, then train a tiny transformer with them. Write `causal_lm_loss(logits,
targets)`, `perplexity(loss)`, `warmup_cosine_lr(step, warmup, total, peak,
floor)`, `clip_global_norm(grads, max_norm)`, `training_flops(params, tokens)`,
and `train_step(model_fn, params, batch, opt_state, config)`. This last
function runs one forward pass, computes the loss, backpropagates, and clips
the gradient. It then applies one decoupled-weight-decay AdamW update, using
the learning rate the schedule gives the current step. Build a small causal transformer from
lesson 26's pieces in this file's `__main__` block, training it on a short,
repeating, built-in character string for a few hundred steps.

The checks verify that the loss on uniform logits equals $\log V$ exactly,
and that perplexity is the exponential of the loss. They verify that the
schedule agrees with both of its pieces at the warmup boundary, and reaches
its floor exactly at the final step. They verify that clipping produces
exactly the threshold norm when the gradient exceeds it, and changes
nothing when it does not. They also verify that the flop formula matches a hand computation, and
that twenty steps of `train_step` on a batch small enough to memorise drive
the loss well below $\log V$.
