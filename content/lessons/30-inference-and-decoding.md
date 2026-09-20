---
title: Inference and decoding
slug: 30-inference-and-decoding
part: "Part 6 — Large language models"
summary: Why generation is memory bound, what the KV cache buys, and the sampling rules that turn logits into text.
minutes: 60
gpu: false
objectives:
  - Derive the arithmetic-intensity argument that makes prefill compute bound and decode memory bound.
  - Derive the KV cache's size formula and the quadratic-to-linear change it makes to generating a sequence.
  - State the nucleus-sampling definition precisely, and derive temperature's limiting behaviour at zero and infinity.
  - Prove that the speculative-decoding acceptance rule reproduces the target distribution exactly, for any draft model.
prereqs:
  - 25-self-attention
  - 27-training-a-transformer
exercise: 30-inference-and-decoding
---

# Inference and decoding

Training a transformer is one forward and one backward pass, batched as large as
the accelerator allows, running until the loss stops falling. Serving one is a
different computation: one token at a time, one sequence a user is waiting on,
with a cost dominated not by how many flops the chip can do but by how fast it
can read its own weights from memory. Every serving system's design — the KV
cache, the choice of attention shape, running a second, smaller model alongside
the real one — follows from taking that cost seriously.

The second half of this lesson asks a different question: given the model's
next-token distribution, which token do you emit? Greedy decoding, beam search,
and sampling with temperature, top-k, or nucleus truncation are four different
answers, and none is "the correct one" outright — each trades off determinism,
diversity, and a tendency toward degenerate repetition. Speculative decoding
closes the lesson by showing that a cheap, inaccurate model can speed up an
expensive, accurate one without changing the output distribution at all, a
stronger guarantee than it sounds.

## What you already know

- Scaled dot-product attention, $\operatorname{softmax}(QK^{\top}/\sqrt{d_k})V$,
  and that keys and values are computed once per token and reused by every later
  query.
- A transformer block's forward pass costs $2N$ flops per token, from the flop
  accounting of a dense matmul.
- Autoregressive generation samples $y_t \sim p(\cdot \mid y_{<t})$ one token at a
  time, conditioning on everything generated so far.
- Softmax is shift invariant: $\operatorname{softmax}(z) =
  \operatorname{softmax}(z - c)$ for any constant $c$.
- A probability distribution is a nonnegative vector summing to one.

## Prefill and decode

Serving a request has two phases. **Prefill** processes the prompt, all
$T_{\text{prompt}}$ tokens at once, in a single forward pass that produces the
first output token and populates every layer's cache. **Decode** then produces
one token at a time, each one a forward pass over a sequence of length one,
conditioned on everything cached so far. The two phases run the same weights
through the same architecture, and they sit on opposite sides of a hardware
boundary.

An accelerator has a peak compute rate $P$, in flops per second, and a peak
memory bandwidth $M$, in bytes per second. A computation's **arithmetic
intensity** is its flops divided by the bytes it must move from memory, and the
computation is compute bound when that intensity exceeds $P/M$, memory bound
when it falls short. Current high-end accelerators put $P/M$ at roughly 100 to
300 flops per byte moved.

Take one dense weight matrix $W \in \mathbb{R}^{d \times d}$, stored at $b$ bytes
per element. Passing a batch of $B$ tokens through it costs $2Bd^2$ flops and
requires reading $W$ once, $bd^2$ bytes, so its arithmetic intensity is

$$
I(B) = \frac{2Bd^2}{bd^2} = \frac{2B}{b} ,
$$

independent of $d$ entirely. Decode processes one token per sequence, $B = 1$,
so its intensity is fixed at $2/b$ regardless of model size — in bf16, $b=2$
bytes, an intensity of exactly $1$ flop per byte, far below any accelerator's
$P/M$. Decode is memory bound at every model scale, and the bottleneck is the
time to stream the weights out of memory, not the matmuls themselves. Prefill
processes the whole prompt as one batch, $B = T_{\text{prompt}}$, so its
intensity grows linearly with prompt length and crosses into compute-bound
territory once $T_{\text{prompt}}$ exceeds $b \cdot (P/M) / 2$ — a few hundred
tokens, for the numbers above. This is why a long prompt's prefill takes
noticeably longer per token than a short one, while decode's per-token cost
barely changes with model width at all, only with how much has to be read.

Decode need not stay memory bound forever. Batching many independent sequences'
single-token steps together shares one weight read across all of them, exactly
the trick that makes prefill compute bound, which is why production serving
systems batch concurrent requests rather than serving one sequence per
accelerator at a time.

## The KV cache

Without caching, generating token $t$ means recomputing keys and values for
every one of the $t$ tokens seen so far, since nothing from the previous step
was kept. Each linear projection over $t$ tokens costs $O(t \, d_{\text{model}}^2)$
flops, and summing that over $t = 1, \dots, T$ gives
$O(T^2 d_{\text{model}}^2)$ — quadratic in the length of the sequence generated.

The fix is to keep every key and value vector computed so far. At step $t$, the
model computes $Q$, $K$, and $V$ only for the new token, appends the new $K$ and
$V$ to a per-layer cache, and attends the new query against everything cached,
old and new. The projections now cost $O(d_{\text{model}}^2)$ per step,
$O(T d_{\text{model}}^2)$ in total, and generation is linear in $T$ instead of
quadratic. The attention scores themselves, $O(t \, d_{\text{head}})$ work at
step $t$, still sum to $O(T^2 d_{\text{head}})$ — unavoidable, since every new
token must be compared against every previous one — but $d_{\text{head}} \ll
d_{\text{model}}$, so this term stays far below the projection cost the cache
just removed, for sequence lengths well short of $d_{\text{model}}$.

The price is memory. Caching keys and values for $n_{\text{kv}}$ heads of width
$d_{\text{head}}$, across $n_{\text{layer}}$ layers and $T$ cached positions, at
$\text{bytes}$ bytes per element, costs

$$
\text{cache bytes} = 2 \cdot n_{\text{layer}} \cdot n_{\text{kv}} \cdot
d_{\text{head}} \cdot T \cdot \text{bytes} ,
$$

the factor of two for keys and values separately. A 24-layer model with 8
key-value heads of width 64, caching a 2048-token context in bf16, needs

$$
2 \cdot 24 \cdot 8 \cdot 64 \cdot 2048 \cdot 2 = 100{,}663{,}296 \text{ bytes} = 96 \text{ MiB}
$$

per sequence — for that one sequence alone, growing linearly with every token
generated and every concurrent request served.

## Fewer key-value heads

The cache formula scales with $n_{\text{kv}}$, the number of key-value heads, not
$n_{\text{head}}$, the number of query heads — the two need not be equal.
**Multi-query attention** sets $n_{\text{kv}} = 1$: every query head attends
against one shared key and value projection. **Grouped-query attention**
generalises this, partitioning the $n_{\text{head}}$ query heads into
$n_{\text{kv}}$ groups that each share one key-value head, recovering ordinary
multi-head attention at $n_{\text{kv}} = n_{\text{head}}$ and multi-query
attention at $n_{\text{kv}} = 1$. Since the cache formula is linear in
$n_{\text{kv}}$, grouping queries into $g = n_{\text{head}} / n_{\text{kv}}$
groups shrinks the cache by exactly a factor of $g$, at the cost of every query
in a group sharing the same keys and values rather than learning its own —
a quality trade-off measured empirically, not derived, and typically small for
$g$ up to eight or so.

## Choosing a token

Every decoding rule starts from the same object, a distribution $p(\cdot \mid
y_{<t})$ over the vocabulary, and differs only in what it does with it.

**Greedy decoding** takes $\operatorname*{arg\,max}_y p(y \mid y_{<t})$ at every
step. It is cheap and deterministic, and it does not maximise the probability of
the sequence it produces. Take a two-step example: $p(a) = 0.4$, $p(b) = 0.6$ at
step one, so greedy commits to $b$; suppose $p(x \mid a) = 0.99$ but every
continuation after $b$ has probability at most $0.5$. Greedy's sequence has
probability at most $0.6 \times 0.5 = 0.30$, while $(a, x)$ has probability
$0.4 \times 0.99 = 0.396$ — a sequence greedy can never reach, because its
first step already discarded $a$.

**Beam search** keeps the $k$ highest cumulative-probability partial sequences
at each step, expands each by the whole vocabulary, and keeps the top $k$ of the
resulting $k \lvert V \rvert$ candidates by cumulative log-probability. It costs
$O(T k \lvert V \rvert)$ time and $O(k)$ sequences held in memory, recovering
greedy at $k=1$ and, at $k = \lvert V \rvert^T$, exhaustive search. Because every
extra token multiplies the sequence probability by a factor at most one, raw
cumulative log-probability only ever falls as a sequence grows, which biases
beam search toward shorter completions regardless of quality. Dividing the
cumulative log-probability by the sequence length before comparing candidates —
scoring by the average per-token log-probability rather than the sum — removes
this mechanical bias and lets beams of different lengths compete on quality
instead of brevity.

**Temperature** rescales the logits before the softmax:
$\operatorname{softmax}(z/\tau)$. As $\tau \to 0^{+}$, divide numerator and
denominator by $\exp(z_{\max}/\tau)$:

$$
\operatorname{softmax}(z/\tau)_i =
\frac{\exp((z_i - z_{\max})/\tau)}{\sum_j \exp((z_j - z_{\max})/\tau)} .
$$

For $i$ at the maximum the numerator is $1$; for every other $i$, $z_i - z_{\max}
< 0$ and the exponent diverges to $-\infty$ as $\tau \to 0^{+}$, sending that
term to zero. The limit is the one-hot indicator of the argmax: temperature zero
is greedy decoding. As $\tau \to \infty$, every $z_i/\tau \to 0$, every
$\exp(z_i/\tau) \to 1$, and the softmax converges to the uniform distribution —
sampling with no regard for the model at all.

**Top-k sampling** keeps only the $k$ highest-probability tokens and renormalises
their probabilities to sum to one before sampling. **Nucleus, or top-p,
sampling** instead keeps the smallest set $S$ of tokens, taken in decreasing
order of probability, whose cumulative probability reaches a threshold $p$:

$$
S = \Big\{ y_{(1)}, \dots, y_{(m)} \Big\}, \quad
m = \min \Big\{ j : \textstyle\sum_{i=1}^{j} p_{(i)} \geq p \Big\} ,
$$

where $p_{(1)} \geq p_{(2)} \geq \dots$ is the sorted distribution, then
renormalises over $S$. Unlike top-k, the size of $S$ adapts to how peaked the
distribution is: a confident next-token prediction gets a small nucleus, a
uniform-looking one gets a large one.

**Repetition** is the failure mode all of the above are patched against: once a
phrase has occurred, attending back to it makes continuing it look highly
probable, and greedy or low-temperature decoding can fall into a loop it never
escapes. A repetition penalty divides the logit of any token already generated
by a factor greater than one before the softmax, damping exactly the tokens a
loop would otherwise reinforce.

## Speculative decoding

Decode is memory bound, so a small draft model $q$ is nearly as slow, per token,
as the large target model $p$ it is meant to accelerate — both spend most of
their time waiting on the same memory bandwidth. Speculative decoding uses that
fact: let $q$ generate several tokens autoregressively, cheaply, then run $p$
once, in parallel, over the whole draft continuation — a prefill-shaped,
compute-bound batch of the size arithmetic intensity rewards — and accept or
reject each drafted token against $p$'s judgement.

For a drafted token $x$, accept it with probability $\min(1, p(x)/q(x))$. On
acceptance, move to the next drafted token. On rejection, discard every draft
token from this point on and sample instead from the residual distribution

$$
r(y) = \frac{\max(0, p(y) - q(y))}{\sum_z \max(0, p(z) - q(z))} .
$$

This reproduces $p$ exactly, for any $q$ with full support, however badly $q$
approximates $p$. For a fixed token $x$, write $Z = \sum_z \max(0, p(z) - q(z))$
for the residual's normalising constant. The probability that $x$ is the token
that ultimately emerges is the probability it is drafted and accepted, plus the
probability rejection happens and $x$ is then drawn from the residual:

$$
P(\text{output} = x) = q(x) \min\!\left(1, \frac{p(x)}{q(x)}\right)
+ Z \cdot \frac{\max(0, p(x) - q(x))}{Z} .
$$

The first term is $q(x) \min(1, p(x)/q(x)) = \min(q(x), p(x))$, since
multiplying through by $q(x)$ turns the $1$ branch into $q(x)$ and the
$p(x)/q(x)$ branch into $p(x)$. The identity

$$
\min(p, q) + \max(p - q, 0) = p
$$

holds for any two reals: if $p \geq q$, the left side is $q + (p - q) = p$; if
$p < q$, it is $p + 0 = p$. Applied pointwise with $p = p(x)$, $q = q(x)$,

$$
P(\text{output} = x) = \min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x)
$$

exactly, for every $x$. Nothing about $q$'s quality enters this identity — a
worse draft model only lowers the acceptance rate, forcing more residual
sampling and giving up more of the speedup, never the correctness of the output
distribution.

## Cost and scaling

The KV cache grows linearly in every dimension of the formula that defines it,
which is what makes long contexts expensive to serve rather than merely slow to
compute.

| Model shape | Context | Cache per sequence |
|---|---|---|
| 12 layers, 8 kv heads, $d_{\text{head}}=64$ | 2,048 tokens | 48 MiB |
| 24 layers, 8 kv heads, $d_{\text{head}}=64$ | 8,192 tokens | 384 MiB |
| 32 layers, 8 kv heads, $d_{\text{head}}=128$ | 32,768 tokens | 4.0 GiB |
| 32 layers, 1 kv head (MQA), $d_{\text{head}}=128$ | 32,768 tokens | 0.5 GiB |

The last two rows are the same model and context, and differ only in
$n_{\text{kv}}$: multi-query attention cuts the cache by the factor of eight
that dropping from eight key-value heads to one predicts exactly. Serving a
hundred concurrent long-context sequences on the third row already asks for 400
GiB of cache alone, before a single weight is loaded — the reason long-context
serving is a memory-capacity problem as much as a compute one.

## Where this breaks

**Cache growth outruns memory.** Past some context length, the KV cache no
longer fits alongside the weights, forcing eviction policies, cache
quantization, or a hard limit on context length that has nothing to do with
what the model could otherwise attend to.

**Speculative decoding without a bandwidth gap.** The technique only pays off
because decode is memory bound; at a large enough serving batch, decode is
already compute bound, the draft model's extra forward passes stop being free,
and speculative decoding can slow generation down instead of speeding it up.

**A weak or slow draft model.** If $q$ rarely agrees with $p$, acceptance is
rare and most tokens fall through to residual sampling, which costs as much as
not drafting at all; if $q$ itself is not cheap relative to $p$, there is no
bandwidth slack left to exploit.

**Sampling without a repetition penalty.** Nucleus and top-k sampling reduce
repetition but do not eliminate it: a sufficiently peaked distribution can put
almost all of its nucleus mass on the token that continues a loop, and only an
explicit penalty on already-used tokens breaks it reliably.

## The exercise

You build the memory model and the decoding rules this lesson derived.

Write `kv_cache_bytes(n_layer, n_kv, d_head, seq_len, bytes_per_elt)` from the
formula above, and a `KVCache` class with `append(layer, k, v)`, storing one new
token's key and value for a given layer, and `get(layer)`, returning every
cached key and value for that layer so far. Write `apply_temperature(logits,
temperature)`, returning the tempered softmax distribution; `top_k_filter(probs,
k)` and `nucleus_filter(probs, p)`, each zeroing out the excluded probability
mass and renormalising what remains to sum to one; `greedy_decode(logits)`,
returning the index of the largest logit; and `speculative_accept(p, q,
draft_token, rng)`, implementing the accept-or-resample rule and returning the
token that rule selects.

The checks verify the cache formula against a hand-computed value and against
the exact reduction multi-query attention predicts, that nucleus filtering
keeps precisely the smallest set reaching $p$ with its kept mass renormalised to
one, that temperature at a small value converges to greedy's choice, that
incremental decoding through `KVCache` produces the same attention output as a
full recompute over the whole sequence at once, and — the one this lesson's
proof was for — that `speculative_accept`, run many times with a draft
distribution deliberately built to disagree with the target, reproduces the
target distribution to within sampling error regardless.
