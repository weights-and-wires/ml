---
title: Inference and decoding
slug: 30-inference-and-decoding
part: "Part 6 — Large language models"
summary: Why generation is memory bound, what the KV cache buys, and the sampling rules that turn logits into text.
minutes: 50
gpu: false
objectives:
  - Show why decode is memory bound and prefill is compute bound, using the arithmetic-intensity argument.
  - State the KV cache's size formula, and the quadratic-to-linear change it makes to generating a sequence.
  - State the nucleus-sampling definition precisely, and give temperature's limiting behavior at zero and infinity.
  - Prove that the speculative-decoding acceptance rule reproduces the target distribution exactly, for any draft model.
prereqs:
  - 25-self-attention
  - 27-training-a-transformer
exercise: 30-inference-and-decoding
---

# Inference and decoding

Training a transformer is one forward and one backward pass, batched as large
as the accelerator allows, running until the loss stops falling. Serving one
is a different computation: one token at a time, for one sequence a user is
waiting on, with a cost dominated not by how many flops the chip can do but by
how fast it can read its own weights from memory. Every serving system's
design choice — the KV cache, the shape of attention, running a second,
smaller model alongside the real one — follows from taking that cost
seriously.

The second half of this lesson asks a different question: given the model's
next-token distribution, which token do you emit? Greedy decoding and
sampling with temperature, top-k, or nucleus truncation are different
answers, and none is correct outright — each trades off determinism,
diversity, and a tendency toward repetition. Speculative decoding closes the
lesson by showing that a cheap, inaccurate model can speed up an expensive,
accurate one without changing the output distribution at all.

## What you already know

- Scaled dot-product attention, $\operatorname{softmax}(QK^{\top}/\sqrt{d_k})V$,
  computes keys and values once per token and reuses them for every later
  query.
- A transformer block's forward pass costs $2N$ flops per token, from the flop
  accounting of a dense matmul.
- Autoregressive generation samples $y_t \sim p(\cdot \mid y_{<t})$ one token
  at a time, conditioning on everything generated so far.
- Softmax is shift invariant: $\operatorname{softmax}(z) =
  \operatorname{softmax}(z - c)$ for any constant $c$.
- A probability distribution is a nonnegative vector that sums to one.

## Prefill and decode

Serving a request has two phases. **Prefill** processes the whole prompt in
one forward pass, producing the first output token and populating every
layer's cache. **Decode** then produces one token at a time, each a forward
pass over a sequence of length one, conditioned on everything cached so far.

An accelerator has a peak compute rate $P$, in flops per second, and a peak
memory bandwidth $M$, in bytes per second. A computation's arithmetic
intensity is its flops divided by the bytes it reads from memory, and it's
compute bound when that intensity exceeds $P/M$, memory bound when it falls
short. High-end accelerators put $P/M$ at roughly 100 to 300 flops per byte
moved.

Take one dense weight matrix $W \in \mathbb{R}^{d \times d}$ at $b$ bytes per
element. Passing a batch of $B$ tokens through it costs $2Bd^2$ flops and
reads $W$ once, $bd^2$ bytes, so its arithmetic intensity is

$$
I(B) = \frac{2Bd^2}{bd^2} = \frac{2B}{b} ,
$$

independent of $d$. Decode processes one token per sequence, $B = 1$, so its
intensity is fixed at $2/b$ regardless of model size — in bf16, exactly one
flop per byte, far below any accelerator's $P/M$. Decode is memory bound at
every model scale. Prefill batches the whole prompt, $B = T_{\text{prompt}}$,
so its intensity grows with prompt length and crosses into compute-bound
territory past a few hundred tokens. That's why a long prompt's prefill costs
noticeably more per token than a short one, while decode's per-token cost
barely changes with model width — only with how much has to be read. Batching
many concurrent requests' single-token steps together shares one weight read
across all of them, which is why production serving systems batch requests
rather than serving one sequence per accelerator.

## The KV cache

Without caching, generating token $t$ means recomputing keys and values for
every one of the $t$ tokens seen so far. Each projection over $t$ tokens costs
$O(t \, d_{\text{model}}^2)$ flops, and summing over $t = 1, \dots, T$ gives
$O(T^2 d_{\text{model}}^2)$ — quadratic in sequence length.

The fix is to keep every key and value vector computed so far. At step $t$,
the model computes $Q$, $K$, and $V$ only for the new token, appends the new
$K$ and $V$ to a per-layer cache, and attends the new query against
everything cached. The projections now cost $O(d_{\text{model}}^2)$ per step,
$O(T d_{\text{model}}^2)$ in total — linear instead of quadratic.

The price is memory. Caching keys and values for $n_{\text{kv}}$ heads of
width $d_{\text{head}}$, across $n_{\text{layer}}$ layers and $T$ cached
positions, at `bytes` bytes per element, costs

$$
\text{cache bytes} = 2 \cdot n_{\text{layer}} \cdot n_{\text{kv}} \cdot
d_{\text{head}} \cdot T \cdot \text{bytes} ,
$$

the factor of two for keys and values. A 24-layer model with 8 key-value
heads of width 64, caching a 2,048-token context in bf16, needs
$2 \times 24 \times 8 \times 64 \times 2048 \times 2 = 96$ MiB per sequence —
growing linearly with every token generated and every concurrent request
served.

## Fewer key-value heads

The cache formula scales with $n_{\text{kv}}$, the number of key-value heads,
not $n_{\text{head}}$, the number of query heads — the two need not match.
**Multi-query attention** sets $n_{\text{kv}} = 1$: every query head attends
against one shared key and value projection. **Grouped-query attention**
generalizes this, partitioning the query heads into $n_{\text{kv}}$ groups
that each share one key-value head, recovering ordinary multi-head attention
at $n_{\text{kv}} = n_{\text{head}}$. Because the cache formula is linear in
$n_{\text{kv}}$, grouping into $g = n_{\text{head}} / n_{\text{kv}}$ groups
shrinks the cache by exactly $g$, at the cost of every query in a group
sharing keys and values instead of learning its own.

## Choosing a token

Every decoding rule starts from the same distribution, $p(\cdot \mid
y_{<t})$, and differs only in what it does with it. **Greedy decoding** takes
$\operatorname*{arg\,max}_y p(y \mid y_{<t})$ at every step. It's cheap and
deterministic, and it doesn't maximize the sequence's overall probability:
committing to the best token at step one can rule out a continuation whose
joint probability would have been higher.

**Temperature** rescales the logits before the softmax:
$\operatorname{softmax}(z/\tau)$. As $\tau \to 0^{+}$, every non-maximal
logit's exponent diverges to $-\infty$, so the distribution collapses onto
the argmax — temperature zero is greedy decoding. As $\tau \to \infty$, every
scaled logit goes to zero and the softmax converges to uniform — sampling
with no regard for the model at all.

**Top-k sampling** keeps the $k$ highest-probability tokens and renormalizes
them to sum to one before sampling. **Nucleus, or top-p, sampling** instead
keeps the smallest set $S$ of tokens, taken in decreasing order of
probability, whose cumulative probability reaches a threshold $p$:

$$
S = \{ y_{(1)}, \dots, y_{(m)} \}, \quad
m = \min \Big\{ j : \textstyle\sum_{i=1}^{j} p_{(i)} \geq p \Big\} ,
$$

then renormalizes over $S$. Unlike top-k, the size of $S$ adapts to how
peaked the distribution is: a confident prediction gets a small nucleus, a
uniform-looking one gets a large one.

**Repetition** is the failure mode the rules above are patched against: once
a phrase has occurred, attending back to it makes continuing it look
probable, and low-temperature decoding can loop forever. A repetition
penalty divides the logit of any already-generated token by a factor greater
than one before the softmax, damping the tokens a loop would reinforce.

## Speculative decoding

Decode is memory bound, so a small draft model $q$ is nearly as slow, per
token, as the large target model $p$ it's meant to accelerate — both spend
most of their time waiting on memory bandwidth. Speculative decoding exploits
that: let $q$ generate several tokens cheaply, then run $p$ once, in
parallel, over the whole draft continuation — a compute-bound, prefill-shaped
batch — and accept or reject each drafted token against $p$'s judgment.

For a drafted token $x$, accept it with probability $\min(1, p(x)/q(x))$. On
rejection, discard every draft token from this point on and sample instead
from the residual distribution $r(y) = \max(0, p(y) - q(y)) / Z$, where
$Z = \sum_z \max(0, p(z) - q(z))$.

This reproduces $p$ exactly, for any $q$ with full support, however badly $q$
approximates $p$. The probability that a fixed token $x$ is the one that
ultimately emerges is the probability it's drafted and accepted, plus the
probability rejection happens and $x$ is then drawn from the residual:

$$
P(\text{output} = x) = q(x) \min\!\left(1, \frac{p(x)}{q(x)}\right)
+ \max(0, p(x) - q(x)) .
$$

The first term simplifies to $\min(q(x), p(x))$. The identity $\min(p, q) +
\max(p - q, 0) = p$ holds for any two reals — check the two cases $p \geq q$
and $p < q$ directly — so the sum above equals $p(x)$ exactly, for every $x$.
A worse draft model only lowers the acceptance rate, forcing more residual
sampling and giving up more of the speedup, never the correctness of the
output distribution.

## Cost and scaling

The KV cache grows linearly in every dimension of the formula that defines
it, which is what makes long contexts expensive to serve rather than merely
slow to compute.

| Model shape | Context | Cache per sequence |
|---|---|---|
| 12 layers, 8 kv heads, $d_{\text{head}}=64$ | 2,048 tokens | 48 MiB |
| 24 layers, 8 kv heads, $d_{\text{head}}=64$ | 8,192 tokens | 384 MiB |
| 32 layers, 8 kv heads, $d_{\text{head}}=128$ | 32,768 tokens | 4.0 GiB |
| 32 layers, 1 kv head (MQA), $d_{\text{head}}=128$ | 32,768 tokens | 0.5 GiB |

The last two rows are the same model and context, differing only in
$n_{\text{kv}}$: multi-query attention cuts the cache by the factor of eight
that dropping from eight key-value heads to one predicts exactly. Serving a
hundred concurrent long-context sequences on the third row already asks for
400 GiB of cache alone, before a single weight is loaded.

## Where this breaks

**Cache growth outruns memory.** Past some context length, the KV cache no
longer fits alongside the weights, forcing eviction policies, cache
quantization, or a hard context limit that has nothing to do with what the
model could otherwise attend to.

**Speculative decoding without a bandwidth gap.** The technique pays off only
because decode is memory bound. At a large enough serving batch, decode is
already compute bound, the draft model's extra forward passes stop being
free, and speculative decoding can slow generation down instead of speeding
it up.

**A weak or slow draft model.** If $q$ rarely agrees with $p$, acceptance is
rare and most tokens fall through to residual sampling, which costs as much
as not drafting at all; if $q$ itself isn't cheap relative to $p$, there's no
bandwidth slack left to exploit.

**Sampling without a repetition penalty.** Nucleus and top-k sampling reduce
repetition but don't eliminate it: a peaked distribution can put almost all
of its nucleus mass on the token that continues a loop, and only an explicit
penalty on already-used tokens breaks it reliably.

## The exercise

You build the memory model and the decoding rules this lesson derived.

Write `kv_cache_bytes(n_layer, n_kv, d_head, seq_len, bytes_per_elt)` from the
formula above, and a `KVCache` class with `append(layer, k, v)`, storing one
new token's key and value for a given layer, and `get(layer)`, returning
every cached key and value for that layer so far. Write
`apply_temperature(logits, temperature)`, returning the tempered softmax
distribution; `top_k_filter(probs, k)` and `nucleus_filter(probs, p)`, each
zeroing out the excluded probability mass and renormalizing what remains to
sum to one; `greedy_decode(logits)`, returning the index of the largest
logit; and `speculative_accept(p, q, draft_token, rng)`, implementing the
accept-or-resample rule and returning the token that rule selects.

The checks verify the cache formula against a hand-computed value and against
the exact reduction multi-query attention predicts, that nucleus filtering
keeps precisely the smallest set reaching $p$ with its kept mass renormalized
to one, that temperature at a small value converges to greedy's choice, that
incremental decoding through `KVCache` produces the same attention output as
a full recompute over the whole sequence at once, and — the one this lesson's
proof was for — that `speculative_accept`, run many times with a draft
distribution deliberately built to disagree with the target, reproduces the
target distribution to within sampling error regardless.
