---
title: Parameter-efficient fine-tuning
slug: 31-parameter-efficient-tuning
part: "Part 6 — Large language models"
summary: Why full fine-tuning's memory cost is prohibitive, and how a low-rank update gets almost all the benefit for a fraction of the parameters.
minutes: 45
gpu: false
objectives:
  - Count full fine-tuning's memory cost per parameter, and identify where each byte goes.
  - Derive LoRA's parameter and memory saving from the rank r and the layer's shape.
  - Show that merging BA into W reproduces the adapted layer's output exactly, at zero inference cost.
  - Show that a gradient step on A and B moves the effective weight along the full fine-tuning gradient, projected onto what the current factorization can reach.
prereqs:
  - 27-training-a-transformer
exercise: 31-parameter-efficient-tuning
---

# Parameter-efficient fine-tuning

Fine-tuning a seven-billion-parameter model by updating every weight needs,
by the accounting this lesson opens with, on the order of a hundred gigabytes
of optimizer state alone, before a single activation is stored. That figure
doesn't fall no matter how cleverly you schedule the computation, because
it's a count of bytes per trainable parameter, and full fine-tuning makes
every parameter trainable.

Parameter-efficient fine-tuning asks whether every parameter needs to move at
all. An empirical observation from outside this literature — that
fine-tuning's useful update often lives in a much lower-dimensional space
than the full parameter count suggests — turns into a method once you
represent the update itself as a low-rank matrix instead of searching for
one. That method, LoRA, is this lesson's center. It derives the saving in
parameters and memory, shows why the adapted model matches the base model
exactly before training starts, and shows that the update it converges to is
the specific projection a full fine-tuning gradient would produce if
restricted to what a rank-$r$ factorization can express.

## What you already know

- A dense layer $y = xW^{\top}$ backpropagates a gradient $\nabla_W L =
  (\nabla_y L)^{\top} x$, one outer product per example.
- Adam keeps two running moments per parameter, a mean and an uncentered
  variance of the gradient, both updated every step.
- Mixed-precision training keeps a fast fp16 or bf16 copy of each weight
  alongside an fp32 master copy that absorbs small updates without rounding
  them away.
- The best rank-$r$ approximation of a matrix in Frobenius norm is its
  truncated SVD, with squared error equal to the sum of the discarded
  squared singular values — the Eckart-Young theorem.
- The chain rule for a bilinear map: for $f(A, B) = BA$, a small change
  $dA, dB$ gives $d(BA) = B\, dA + dB\, A$.

## Full fine-tuning's memory, counted

Training a matrix $W$ with Adam under mixed precision keeps five quantities
per parameter: an fp16 copy of $W$ for the forward and backward matmuls (2
bytes), an fp16 gradient (2 bytes), an fp32 master copy of $W$ that the
optimizer updates (4 bytes), and Adam's two fp32 running moments (4 bytes
each), for $2+2+4+4+4=16$ bytes per trainable parameter. Skipping mixed
precision doesn't help: an fp32 copy, its fp32 gradient, and the two fp32
moments sum to the same 16 bytes. Mixed precision buys faster matrix
multiplication on hardware built for low-precision arithmetic — it doesn't
reduce the memory the optimizer needs. The only way to shrink this number is
to shrink the count of parameters it applies to.

## Low-rank adapters

The intuition behind LoRA: many pretrained models can be fine-tuned to near
full performance while optimizing within a low-dimensional subspace of the
full parameter space, so the useful update has a low **intrinsic dimension**.
Rather than search for that subspace, LoRA constrains the *shape* of the
update directly. Freeze a pretrained weight matrix $W \in
\mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and replace the fine-tuned
weight with

$$
W_{\text{eff}} = W + \frac{\alpha}{r} BA , \qquad
B \in \mathbb{R}^{d_{\text{out}} \times r}, \;\; A \in \mathbb{R}^{r \times
d_{\text{in}}}, \;\; r \ll \min(d_{\text{in}}, d_{\text{out}}) .
$$

Only $A$ and $B$ are trained; $W$ never receives a gradient. The scalar
$\alpha/r$ rescales the update so its magnitude stays roughly constant as $r$
changes, letting a learning rate tuned at one rank carry over to another
without retuning. Initializing $B$ to zero, with $A$ drawn from an ordinary
random initialization, makes $BA = 0$ at step zero, so $W_{\text{eff}} = W$:
the adapted model computes the same function the pretrained model did before
a single gradient step.

The saving is direct. A full fine-tune of $W$ trains $d_{\text{in}}
d_{\text{out}}$ parameters; a LoRA adapter trains $r(d_{\text{in}} +
d_{\text{out}})$ parameters, for $A$ and $B$ together. For $d_{\text{in}} =
d_{\text{out}} = 4096$ and $r = 8$ — one attention projection in a
seven-billion-parameter Llama-class transformer — full fine-tuning trains
$4096^2 = 16{,}777{,}216$ parameters against LoRA's $8 \times 8192 =
65{,}536$, a factor of exactly 256 fewer. At 16 bytes per trainable
parameter, that one matrix's optimizer state falls from 256 MiB to 1 MiB. The
frozen base weight still has to sit in memory to run the forward and
backward pass through it, but it needs no gradient and no optimizer state.

LoRA can adapt any weight matrix in the network. Given a fixed parameter
budget, spreading a small rank across more matrices — all four attention
projections, and often the feed-forward matrices too — consistently
outperforms concentrating a large rank on one or two matrix types.

## Merging at inference

Because $\alpha A / r$ and $B$ are ordinary matrices once training finishes,
the adapted weight $W_{\text{eff}} = W + (\alpha/r) BA$ can be computed once
and stored in place of $W$. Every later forward pass runs through
$W_{\text{eff}}$ directly, at the same latency, with no trace an adapter was
ever there. The cost shows up when a serving system supports many adapters
on one base model: merging bakes in a single adapter, so switching per
request means re-merging before every request, an $O(d_{\text{in}}
d_{\text{out}})$ matmul, or keeping $W$ unmerged and adding each request's
own $(\alpha/r) x A^{\top} B^{\top}$ term at serving time, at
$O(r(d_{\text{in}} + d_{\text{out}}))$ extra flops per token instead of
nothing.

## Why the gradient still points the right way

LoRA doesn't train an arbitrary rank-$r$ update — gradient descent on $A$ and
$B$ moves $W_{\text{eff}}$ in a specific, structured direction. With loss
$L$ and $G = \partial L / \partial W_{\text{eff}}$ the gradient a full
fine-tune of $W$ would receive, the chain rule through $BA$ gives

$$
\frac{\partial L}{\partial A} = \frac{\alpha}{r} B^{\top} G , \qquad
\frac{\partial L}{\partial B} = \frac{\alpha}{r} G A^{\top} .
$$

A gradient step on $A$ and $B$ changes $W_{\text{eff}}$ by $-\eta
(\alpha/r)^2 \big(G A^{\top} A + B B^{\top} G\big)$, to first order in the
step size $\eta$. That's the full gradient $G$, acted on by $A^{\top} A$ on
the right and $B B^{\top}$ on the left — projection-like operators built
from the current factorization. LoRA's update follows $G$ exactly where the
current $A$ and $B$ already span, and none of it where they don't; widening
$r$ only widens what "where they span" means.

## Quantized and other adapters

**QLoRA** freezes the base model in 4-bit precision and trains LoRA adapters
in higher precision on top of it, backpropagating through the frozen,
dequantized weights without updating them. Its NF4 data type places
quantization levels at the quantiles of a standard normal distribution, so
weights that are approximately normally distributed after pretraining are
quantized with roughly equal expected error in every bin. Because the
per-block scale factors needed for accurate quantization are numerous enough
to cost real memory, QLoRA quantizes them a second time too.

**Adapters** insert a small bottleneck feed-forward module into the
computation graph after each sublayer, training only the new modules. Unlike
LoRA's parallel, additive update, an adapter sits in series with existing
computation, so it can't be merged away — every inference pass pays its
extra layers. **Prefix tuning** prepends trainable vectors to the keys and
values at every attention layer, changing what each layer attends over
rather than what any weight computes, at the cost of extra context length
and attention compute on every call. **$(IA)^3$** learns a per-channel vector
that rescales keys, values, and feed-forward activations, adding only
$O(d)$ parameters per rescaled activation and folding into the surrounding
weights just as cleanly as LoRA.

Downstream task performance rises quickly as $r$ grows from one, then
flattens: most tasks reach performance close to full fine-tuning at a rank in
the single digits to the low tens. Pushing $r$ further buys almost nothing,
because the saving was never about approximating $W$ — it was about
supplying enough directions to reach the low-dimensional update that was
there all along.

## Cost and scaling

The saving compounds: more matrices worth adapting hold the same ratio
between full and LoRA training cost.

| Quantity | Full fine-tuning | LoRA ($r=8$) |
|---|---|---|
| Trainable parameters, one $4096 \times 4096$ matrix | 16,777,216 | 65,536 |
| Optimizer memory (16 bytes/parameter) | 256 MiB | 1 MiB |
| Trainable parameters, all attention projections, 32 layers | 2,147,483,648 | 8,388,608 |
| Optimizer memory, same scope | 32 GiB | 128 MiB |

The ratio is exactly 256 at every scope, because every matrix involved has
the same shape: adapting more layers multiplies both sides by the same
factor. The frozen base weights still have to be loaded to run forward and
backward passes — LoRA doesn't touch that cost — but the optimizer memory
that scales with trainable parameters falls by close to three orders of
magnitude, the difference between needing a cluster and needing one
accelerator.

## Where this breaks

**Serving many adapters at once.** The free-at-inference property belongs to
one merged adapter; a system serving hundreds of customer-specific adapters
off a shared base model gives that property up for flexibility, and must
budget the extra per-request flops.

**Very small ranks on hard tasks.** The rank-quality plateau is task
dependent: tasks that need the model to acquire genuinely new capability,
rather than recombine what pretraining already encoded, can need a rank
large enough that the parameter saving becomes far less dramatic.

**Quantization error compounds with adaptation error.** QLoRA's 4-bit base
model already discards information the full-precision model had, so a LoRA
adapter trained against that quantized base fits a slightly different
function than one trained against full precision, and the two errors don't
add in a way that's easy to bound in advance.

**Catastrophic forgetting isn't solved by fewer parameters.** Fine-tuning any
subset of parameters, however small, can still overwrite pretrained behavior
the fine-tuning data never exercised; a small trainable footprint bounds how
much can change, not what changes.

## The exercise

You build LoRA's forward pass, its parameter and memory accounting, its
inference-time merge, and the low-rank approximation result the whole method
leans on.

Write `lora_forward(x, W, A, B, alpha, r)`, computing $xW^{\top} +
(\alpha/r)\, x A^{\top} B^{\top}$, and `lora_parameter_count(d_in, d_out,
r)`, returning $r(d_{\text{in}} + d_{\text{out}})$. Write `merge_lora(W, A, B,
alpha, r)`, returning $W + (\alpha/r) BA$, and `optimiser_memory_bytes(params,
mode)`, applying the 16-bytes-per-parameter figure derived above for `mode`
in `"full"` or `"lora"` — the same rate in both cases, since the saving comes
entirely from the parameter count. Write `best_rank_approximation(delta_w,
r)`, returning the closest rank-$r$ matrix to `delta_w` in Frobenius norm by
truncated SVD.

The checks verify that a zero-initialized $B$ makes `lora_forward` exactly
reproduce the base layer, that `merge_lora` gives outputs identical to
`lora_forward` to machine precision, that the parameter count matches the
hand-computed $r(d_{\text{in}} + d_{\text{out}})$ and the saving against full
fine-tuning matches the formula, that `best_rank_approximation` achieves the
Eckart-Young error for the chosen rank, and, tying the exercise back to this
lesson's central claim, that a gradient step on $A$ and $B$ moves
$W_{\text{eff}}$ in exactly the direction the full gradient with respect to
$W_{\text{eff}}$ would, projected onto the tangent space the current
factorization can reach.
