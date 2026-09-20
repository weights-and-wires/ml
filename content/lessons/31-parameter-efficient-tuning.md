---
title: Parameter-efficient fine-tuning
slug: 31-parameter-efficient-tuning
part: "Part 6 — Large language models"
summary: Why full fine-tuning's memory cost is prohibitive, and how a low-rank update gets almost all the benefit for a fraction of the parameters.
minutes: 55
gpu: false
objectives:
  - Count full fine-tuning's memory cost per parameter and identify where each byte goes.
  - Derive LoRA's parameter and memory saving from the rank r and the layer's shape.
  - Prove that merging BA into W reproduces the adapted layer's output exactly, at zero inference cost.
  - Show that a gradient step on A and B moves the effective weight in the direction a full fine-tuning step would, restricted to what the current factorisation can reach.
prereqs:
  - 27-training-a-transformer
exercise: 31-parameter-efficient-tuning
---

# Parameter-efficient fine-tuning

Fine-tuning a seven-billion-parameter model by updating every weight needs, by
the accounting this lesson opens with, on the order of a hundred gigabytes of
optimizer state alone — before a single activation is stored. That figure does
not fall much no matter how cleverly you schedule the computation, because it is
a count of bytes per trainable parameter, and full fine-tuning makes every
parameter trainable.

The question parameter-efficient fine-tuning asks is whether every parameter
needs to move at all. An empirical observation from outside this literature —
that fine-tuning's useful update often lives in a much lower-dimensional space
than the full parameter count suggests — turns into a method once you decide to
represent the update itself as a low-rank matrix rather than search for one.
That method, LoRA, is this lesson's centre: it derives the saving in parameters
and memory, shows why the adapted model is identical to the base model before
training starts, and proves that the update it converges to is not an arbitrary
low-rank compromise, but the specific projection a full fine-tuning gradient
would produce if it were restricted to what a rank-$r$ factorisation can express.

## What you already know

- A dense layer $y = xW^{\top}$ backpropagates a gradient $\nabla_W L =
  (\nabla_y L)^{\top} x$, one outer product per example.
- Adam keeps two running moments per parameter, a mean and an uncentered
  variance of the gradient, both updated every step.
- Mixed-precision training keeps an fp16 or bf16 copy of each weight for fast
  matrix multiplication alongside an fp32 master copy that absorbs the small
  updates without rounding them away.
- The best rank-$r$ approximation of a matrix in Frobenius norm is its
  truncated SVD, with squared error equal to the sum of the squared singular
  values discarded — the Eckart-Young theorem.
- The chain rule for a bilinear map: for $f(A, B) = BA$, a small change
  $dA, dB$ gives $d(BA) = B\, dA + dB\, A$.

## Full fine-tuning's memory, counted

Training a matrix $W$ with Adam under mixed precision keeps five quantities per
parameter: an fp16 copy of $W$ for the forward and backward matmuls (2 bytes),
an fp16 gradient produced by backpropagation (2 bytes), an fp32 master copy of
$W$ that the optimizer actually updates (4 bytes), and Adam's two fp32 running
moments (4 bytes each). That totals

$$
2 + 2 + 4 + 4 + 4 = 16 \text{ bytes per trainable parameter.}
$$

This figure does not improve if you skip mixed precision and keep everything in
fp32: a single fp32 copy of the parameter (4 bytes), its fp32 gradient (4
bytes), and the two fp32 moments (4 bytes each) sum to the same 16 bytes.
Mixed precision buys faster matrix multiplication on hardware built for
low-precision arithmetic; it does not reduce the memory the optimizer needs.
The only way to shrink this number is to shrink the count of parameters it
applies to — a fact the rest of this lesson exploits directly.

## The intrinsic-dimension observation

Before LoRA, an empirical result about fine-tuning more broadly suggested
something odd: many pretrained models can be fine-tuned to near-full performance
by optimizing within a randomly chosen low-dimensional subspace of the full
parameter space, far smaller than the number of parameters the model has. The
update fine-tuning needs to make, in other words, has a low **intrinsic
dimension** — most directions in parameter space are not needed to reach a good
solution.

That observation does not by itself say which low-dimensional subspace to use,
or how to find it cheaply. LoRA's answer is not to search for a subspace at all,
but to constrain the *shape* of the update directly: instead of a rank-$d$
update — full fine-tuning, unconstrained — search only over rank-$r$ updates,
for $r$ far smaller than the layer's dimensions, and let gradient descent choose
which rank-$r$ update to use.

## Low-rank adapters

Freeze a pretrained weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times
d_{\text{in}}}$ and replace the fine-tuned weight with

$$
W_{\text{eff}} = W + \frac{\alpha}{r} BA , \qquad
B \in \mathbb{R}^{d_{\text{out}} \times r}, \;\; A \in \mathbb{R}^{r \times
d_{\text{in}}}, \;\; r \ll \min(d_{\text{in}}, d_{\text{out}}) .
$$

Only $A$ and $B$ are trained; $W$ never receives a gradient. The scalar
$\alpha / r$ rescales the update so that its magnitude stays roughly constant as
$r$ changes, which lets a learning rate tuned at one rank carry over
approximately to another without retuning — $\alpha$ is chosen once, and the
division by $r$ cancels the tendency of $BA$'s scale to grow with $r$ purely
from summing more rank-one terms.

Initializing $B$ to zero, with $A$ drawn from an ordinary random initialization,
makes $BA = 0$ at step zero exactly, so $W_{\text{eff}} = W$: the adapted model
computes the identical function the pretrained model did before a single
gradient step is taken. Training then moves $B$ away from zero, and the update
grows from nothing rather than starting from an arbitrary random perturbation of
a model that already works.

The saving is direct. A full fine-tune of $W$ trains $d_{\text{in}}
d_{\text{out}}$ parameters; a LoRA adapter trains

$$
r \, d_{\text{in}} + r \, d_{\text{out}} = r (d_{\text{in}} + d_{\text{out}})
$$

parameters, for $A$ and $B$ together. For $d_{\text{in}} = d_{\text{out}} =
4096$ and $r = 8$ — one attention projection in a model the size of a
seven-billion-parameter Llama-class transformer — full fine-tuning trains
$4096^2 = 16{,}777{,}216$ parameters against LoRA's $8 \times 8192 = 65{,}536$,
a factor of exactly $256$ fewer. At 16 bytes per trainable parameter, that one
matrix's optimizer state falls from 256 MiB to 1 MiB. The frozen base weight
itself still has to sit in memory to run the forward and backward pass through
it, at whatever precision it is stored in, but it needs no gradient and no
optimizer state at all.

## Which matrices to adapt

LoRA can be applied to any weight matrix in the network — the four attention
projections $W_Q, W_K, W_V, W_O$, and the feed-forward matrices alike. Given a
fixed total parameter budget, the empirical answer is to spread a small rank
across more matrices rather than concentrate a large rank on few: adapting all
of the attention projections, and often the feed-forward matrices too, at a
modest rank consistently outperforms adapting only one or two matrix types at a
correspondingly larger rank for the same parameter count. Adapting only $W_V$,
the original paper's smallest configuration, still recovers much of full
fine-tuning's benefit, which is itself evidence for how little of the update
budget the task actually needs.

## Merging at inference

Because $\alpha A / r$ and $B$ are ordinary matrices once training finishes, the
adapted weight $W_{\text{eff}} = W + (\alpha/r) BA$ can be computed once and
stored in place of $W$. Every subsequent forward pass runs through
$W_{\text{eff}}$ directly, with the same architecture, the same latency, and no
trace that an adapter was ever there — LoRA is free at inference once merged,
unlike methods that add sequential computation to every forward pass.

The cost shows up when a serving system has to support many adapters on one
base model. Merging bakes in a single adapter, so switching adapters per
request means either re-merging before every request — an $O(d_{\text{in}}
d_{\text{out}})$ matrix multiply, cheap once but not free at request volume —
or keeping $W$ unmerged and adding each request's own $(\alpha/r) x A^{\top}
B^{\top}$ term at serving time, which costs $O(r(d_{\text{in}} +
d_{\text{out}}))$ extra flops per token instead of nothing, in exchange for
serving a different adapter to every request in the same batch off one resident
copy of $W$.

## Quantized and other adapters

**QLoRA** freezes the base model in 4-bit precision and trains LoRA adapters in
a higher-precision format on top of it, backpropagating through the frozen,
dequantized weights without ever updating them. Its NF4 data type places
quantization levels at the quantiles of a standard normal distribution rather
than at uniformly spaced values, so that a set of weights that is
approximately normally distributed — the common case after pretraining —
is quantized with roughly equal expected error in every bin, instead of wasting
resolution on values that occur rarely. Because each block of weights needs its
own scale factor to quantize accurately, and those scale factors are numerous
enough to cost real memory in their own right, QLoRA quantizes the scale
factors themselves a second time — **double quantization** — recovering a
further fraction of a bit per parameter at the cost of one extra dequantization
step.

**Adapters** insert a small bottleneck feed-forward module, a down-projection,
a nonlinearity, and an up-projection, directly into the computation graph after
each sublayer, training only the new modules while the backbone stays frozen.
Unlike LoRA's parallel, additive update, an adapter sits in series with the
existing computation, so it cannot be merged away: every inference pass pays
its extra sequential layers. **Prefix tuning** prepends trainable continuous
vectors to the keys and values at every attention layer, leaving the backbone's
weights untouched and optimizing only the prefix; it changes what every layer
attends over rather than what any weight matrix computes, and it costs
additional context length, and therefore additional attention compute and
cache, at every inference call. **$(IA)^3$** learns a per-channel vector that
rescales keys, values, and feed-forward activations elementwise, adding only
$O(d)$ parameters per rescaled activation — far fewer than LoRA's $O(rd)$ — and,
since an elementwise rescaling commutes with the linear map on either side of
it, folds into the surrounding weights just as cleanly as LoRA does.

## The rank-versus-quality curve

Downstream task performance rises quickly as $r$ grows from one, then flattens:
most tasks reach performance close to full fine-tuning at a rank in the single
digits to the low tens, well below what the layer's width might suggest is
needed. Pushing $r$ toward $\min(d_{\text{in}}, d_{\text{out}})$ buys almost
nothing once the plateau is reached, because the earlier saving in trainable
parameters was never really about approximating $W$ itself — it was about
supplying enough directions to reach the low-dimensional update the intrinsic-
dimension observation predicted was there all along.

## Cost and scaling

The saving compounds: more parameters means more matrices worth adapting, and
the ratio between full and LoRA training cost holds regardless of which layer
it is applied to.

| Quantity | Full fine-tuning | LoRA ($r=8$) |
|---|---|---|
| Trainable parameters, one $4096 \times 4096$ matrix | 16,777,216 | 65,536 |
| Optimizer memory (16 bytes/parameter) | 256 MiB | 1 MiB |
| Trainable parameters, all attention projections, 32 layers | 2,147,483,648 | 8,388,608 |
| Optimizer memory, same scope | 32 GiB | 128 MiB |

The ratio is exactly 256 at every scope, because every matrix involved has the
same shape: adapting more layers multiplies both sides of the comparison by the
same factor. The frozen base weights still have to be loaded to run forward and
backward passes through them — LoRA does not touch that cost — but the
optimizer memory that scales with the number of trainable parameters falls by
close to three orders of magnitude, which is the difference between needing a
cluster and needing one accelerator.

## Where this breaks

**Serving many adapters at once.** The free-at-inference property is a property
of one merged adapter; a system serving hundreds of customer-specific adapters
off a shared base model gives that property up in exchange for flexibility, and
must budget the extra per-request flops accordingly.

**Very small ranks on hard tasks.** The rank-quality plateau is task dependent:
tasks that need the model to acquire genuinely new capability, rather than
recombine what pretraining already encoded, can need a rank large enough that
the parameter saving becomes far less dramatic.

**Quantization error compounds with adaptation error.** QLoRA's 4-bit base
model already discards information the full-precision model had; a LoRA
adapter trained against that quantized base is fitting a slightly different
function than one trained against the full-precision weights, and the two
errors do not add in a way that is easy to bound in advance.

**Catastrophic forgetting is not solved by fewer parameters.** Fine-tuning any
subset of parameters, however small, can still overwrite pretrained behaviour
the fine-tuning data never exercised; a small trainable footprint bounds how
much can change, not what changes.

## The exercise

You build LoRA's forward pass, its parameter and memory accounting, its
inference-time merge, and the low-rank approximation result the whole method
leans on.

Write `lora_forward(x, W, A, B, alpha, r)`, computing $xW^{\top} +
(\alpha/r)\, x A^{\top} B^{\top}$, and `lora_parameter_count(d_in, d_out, r)`,
returning $r(d_{\text{in}} + d_{\text{out}})$. Write `merge_lora(W, A, B, alpha,
r)`, returning $W + (\alpha/r) BA$, and `optimiser_memory_bytes(params, mode)`,
applying the 16-bytes-per-parameter figure derived above to a count of
trainable parameters for `mode` in `"full"` or `"lora"` — the same rate in both
cases, since the saving comes entirely from the parameter count, not a cheaper
rate. Write `best_rank_approximation(delta_w, r)`, returning the closest
rank-$r$ matrix to `delta_w` in Frobenius norm by truncated SVD.

The checks verify that a zero-initialized $B$ makes `lora_forward` exactly
reproduce the base layer, that `merge_lora` gives outputs identical to
`lora_forward` to machine precision, that the parameter count matches the
hand-computed $r(d_{\text{in}} + d_{\text{out}})$ and that the saving against
full fine-tuning matches the formula, that `best_rank_approximation` achieves
the Eckart-Young error for the chosen rank, and, tying the exercise back to this
lesson's central claim, that a gradient step on $A$ and $B$ moves
$W_{\text{eff}}$ in exactly the direction the full gradient with respect to
$W_{\text{eff}}$ would, projected onto the tangent space the current
factorization can reach.
