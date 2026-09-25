---
title: Inference and decoding
slug: 30-inference-and-decoding
part: "Part 6 — Large language models"
summary: Why generation is limited by memory speed, how a KV cache helps, and the rules that turn scores into text.
minutes: 55
gpu: false
objectives:
  - Explain why generating one token at a time is limited by memory speed, not arithmetic.
  - Compute the size of a KV cache, and build one that stores keys and values token by token.
  - Build greedy, temperature, top-k, and nucleus decoding from a vector of scores.
  - Build speculative decoding's accept-or-replace rule, and explain why it keeps the big model's output distribution.
prereqs:
  - 25-self-attention
  - 27-training-a-transformer
exercise: 30-inference-and-decoding
---

# Inference and decoding

Training a transformer runs huge batches forward and backward until the loss
stops falling. Serving one is a different job. A user waits while the model
writes one token at a time. Each step is small, and the chip spends most of
its time reading weights from memory rather than doing arithmetic.

Every trick in a serving system follows from that fact. The KV cache avoids
repeating work. Sharing key and value heads shrinks memory. A small helper
model can even speed up a big one.

The second half of this lesson asks a simpler question. The model gives you a
probability for every possible next token. Which one do you pick? You build
four rules for that choice, and then speculative decoding, which gets a
speedup without changing what the big model would have said.

## What you already know

- Attention compares a query with every earlier key, then averages the values
  using those weights.
- A softmax turns a vector of scores into probabilities that are positive and
  sum to one.
- Subtracting the same number from every score doesn't change the softmax.
- A language model generates text one token at a time, each conditioned on
  everything before it.
- A forward pass costs about two flops per parameter per token.

## Prefill and decode

Serving a request has two phases:

- **Prefill** reads the whole prompt in one forward pass. It produces the
  first output token and fills the cache.
- **Decode** then produces one token per forward pass.

Think of a chef with a huge cookbook on a high shelf. Fetching the book is
slow. Cooking from it is fast. If the chef fetches the book to make one dish,
most of the time goes to fetching. If the chef makes a hundred dishes per
fetch, the fetch cost is shared. The weights are the cookbook. Memory
bandwidth is the ladder.

A chip has a peak compute rate, in flops per second, and a peak memory speed,
in bytes per second. *Arithmetic intensity* is the flops a job does per byte
it reads. When intensity is low, the chip waits on memory. The job is *memory
bound*. When intensity is high, it waits on arithmetic and is *compute bound*.
Big accelerators need roughly 100 to 300 flops per byte to stay busy.

Push a batch of $B$ tokens through one $d \times d$ weight matrix stored at
$b$ bytes per number. That's $2Bd^2$ flops and one read of $bd^2$ bytes:

$$
I = \frac{2Bd^2}{bd^2} = \frac{2B}{b}
$$

Read it as: intensity is twice the batch size divided by bytes per number. The
matrix size cancels. Decode has $B = 1$, so in 16-bit numbers ($b = 2$) the
intensity is one flop per byte. That's far below what any accelerator needs.
Decode is memory bound at every model size. Prefill has $B$ equal to the
prompt length, so a long prompt keeps the chip busy. Servers batch many users'
decode steps together so one weight read serves them all.

## The KV cache

In attention, every token produces a *key* and a *value*. Each later token
compares its *query* with all earlier keys and averages the earlier values.
Earlier keys and values never change. Without a cache, step $t$ would
recompute keys and values for all $t$ earlier tokens. Over $T$ steps that work
grows with $T^2$.

The cache is a notebook. At each step you compute the key and value for the
new token only, append them to a per-layer list, and attend against the whole
list. The work per step stays constant, so the total grows with $T$ instead of
$T^2$.

The price is memory. The cache stores keys and values for $n_{\text{kv}}$
heads of width $d_{\text{head}}$, in each of $n_{\text{layer}}$ layers, for
$T$ tokens:

$$
\text{bytes} = 2 \cdot n_{\text{layer}} \cdot n_{\text{kv}} \cdot
d_{\text{head}} \cdot T \cdot \text{bytes per number}
$$

Read it left to right: 2 because you store both keys and values, times
layers, times heads, times numbers per head, times tokens, times bytes per
number. In code it's one multiplication:

```python
size = 2 * n_layer * n_kv * d_head * seq_len * bytes_per_elt
```

### Worked example

A 24-layer model with 8 key-value heads of width 64, a 2,048-token context,
and 2-byte numbers:

- $2 \times 24 = 48$
- $48 \times 8 = 384$
- $384 \times 64 = 24{,}576$
- $24{,}576 \times 2048 = 50{,}331{,}648$
- $50{,}331{,}648 \times 2 = 100{,}663{,}296$ bytes, which is 96 MiB

That's for one sequence. A hundred users need a hundred caches.

For the exercise's `KVCache`, the stored shapes matter. Each call to `append`
adds a `k` and a `v` of shape `(n_kv, d_head)` for one token. After $T$
appends, `get` returns `K` and `V` of shape `(T, n_kv, d_head)`, one row per
token in the order you appended them. `np.stack(list_of_arrays, axis=0)`
builds exactly that.

## Share key and value heads

The cache formula counts key-value heads, $n_{\text{kv}}$, not query heads.
The two don't have to match:

- **Multi-query attention** uses one key-value head that every query head
  shares, so $n_{\text{kv}} = 1$.
- **Grouped-query attention** splits the query heads into $n_{\text{kv}}$
  groups. Each group shares one key-value head.

Because the formula is a plain product, going from 8 key-value heads to 1
shrinks the cache by exactly 8. The cost is that queries in a group can't
learn their own keys and values.

## Choose the next token

The model's last layer produces *logits*: one raw score per token in the
vocabulary. Every decoding rule starts from those scores.

### Greedy

Pick the token with the highest score. That's `np.argmax(logits)`. It's fast
and always gives the same answer. It isn't guaranteed to find the most likely
sentence, because the best first word can lead to worse later words.

### Temperature

Temperature $\tau$ is a dial for how adventurous the model is. Divide the
logits by $\tau$ before the softmax:

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

Read it as: scale every score $z_i$ by $1/\tau$, exponentiate, and divide by
the total so the results sum to one. A small $\tau$ stretches the gaps between
scores, so the top token takes almost all the probability. As $\tau$
approaches 0, this becomes greedy decoding. A large $\tau$ squashes the gaps,
and as $\tau$ grows without limit the probabilities become equal. For
stability, subtract the largest scaled score before calling `np.exp`:

```python
s = logits / temperature
e = np.exp(s - s.max())
probs = e / e.sum()
```

Worked example with logits $[2, 1, 0]$:

| $\tau$ | Scaled scores | Probabilities |
|---|---|---|
| 1 | $[2, 1, 0]$ | $[0.6652, 0.2447, 0.0900]$ |
| 2 | $[1, 0.5, 0]$ | $[0.5065, 0.3072, 0.1863]$ |
| 0.5 | $[4, 2, 0]$ | $[0.8668, 0.1173, 0.0159]$ |

At $\tau = 1$: after subtracting the max you get $[0, -1, -2]$. Exponentials
are $[1, 0.3679, 0.1353]$, summing to $1.5032$. Dividing gives the first row.

### Top-k

Keep the $k$ most likely tokens, set the rest to zero, and divide by the kept
total so it sums to one again. With probabilities $[0.4, 0.3, 0.2, 0.1]$ and
$k = 2$, you keep $0.4$ and $0.3$, whose total is $0.7$. The result is
$[0.5714, 0.4286, 0, 0]$.

### Nucleus, or top-p

Top-k keeps the same number of tokens whether the model is sure or unsure.
Nucleus sampling adapts. Sort the tokens from most to least likely. Keep the
shortest run from the top whose running total reaches $p$. Then renormalise.

Worked example with probabilities $[0.1, 0.5, 0.05, 0.2, 0.15]$ and $p = 0.7$:

- Sorted largest first: $0.5$ (index 1), $0.2$ (index 3), $0.15$, $0.1$,
  $0.05$
- Running totals: $0.5, 0.7, 0.85, 0.95, 1.0$
- The first total that reaches $0.7$ is the second one, so keep indices 1
  and 3
- Renormalise: $0.5 / 0.7 = 0.7143$ and $0.2 / 0.7 = 0.2857$

The numpy tools are `np.argsort(-probs)` for the order, `np.cumsum` for the
running totals, and `np.searchsorted(cumulative, p)` for the first position
that reaches $p$.

### Repetition

Once a phrase appears, attention makes repeating it look likely, and
low-temperature decoding can loop forever. A *repetition penalty* shrinks the
score of any token already generated before the softmax.

## Speculative decoding

Decode is memory bound, so a small *draft* model is almost free next to the
big *target* model. Speculative decoding uses that. The draft model guesses a
few tokens. The target model then checks all of them in one pass, which is
shaped like prefill and keeps the chip busy. Each guess is accepted or
replaced.

Call the target's probabilities $p$ and the draft's $q$. For a drafted token
$x$, accept it with probability

$$
\min\left(1, \frac{p(x)}{q(x)}\right)
$$

Read it as: if the target likes $x$ at least as much as the draft did, always
accept. Otherwise accept with the ratio of the two. On rejection, draw a
replacement from the *residual*: take $\max(0, p - q)$ for every token and
divide by its total. That's the probability the target wanted but the draft
didn't give.

```python
accept = rng.uniform() < min(1.0, p[x] / q[x])
residual = np.maximum(p - q, 0); residual /= residual.sum()
```

Why is this safe? The chance of emitting $x$ is the chance it was drafted and
kept, $\min(q(x), p(x))$, plus the chance of a rejection that lands on $x$,
which works out to $\max(0, p(x) - q(x))$. For any two numbers, the minimum
plus the positive part of their difference equals the first one. So the total
is exactly $p(x)$. A bad draft model makes you reject more and lose the
speedup. It never changes the output distribution.

### Worked example

Take $p = [0.7, 0.2, 0.1]$ and $q = [0.1, 0.1, 0.8]$:

- Acceptance chances: token 0 gets $\min(1, 7) = 1$, token 1 gets $\min(1, 2)
  = 1$, token 2 gets $\min(1, 0.125) = 0.125$.
- Accepted mass is $\min(p, q) = [0.1, 0.1, 0.1]$, totaling $0.3$. So you
  reject 70% of the time.
- Residual: $\max(0, p - q) = [0.6, 0.1, 0]$, normalised to $[0.857, 0.143,
  0]$.
- Output chance of token 0: $0.1 + 0.7 \times 0.857 = 0.7$. Token 1: $0.1 +
  0.7 \times 0.143 = 0.2$. Token 2: $0.1 + 0 = 0.1$. That's $p$ exactly.

## Build it step by step

The checks test the pieces in this order.

1. **`kv_cache_bytes(n_layer, n_kv, d_head, seq_len, bytes_per_elt=2)`.**
   Returns an integer. It's the product from the cache formula. Common
   mistake: forgetting the 2 for keys and values. Check against 100,663,296.
2. **`greedy_decode(logits)`.** Takes shape `(vocab,)`, returns a plain
   `int`. Use `np.argmax` and wrap it in `int()`.
3. **`apply_temperature(logits, temperature)`.** Takes shape `(vocab,)`,
   returns probabilities of the same shape. Divide, subtract the max,
   exponentiate, normalise. Common mistake: skipping the max subtraction.
   At $\tau = 0.001$ the scaled scores reach the thousands and `np.exp`
   overflows to `inf`.
4. **`top_k_filter(probs, k)`.** Returns shape `(vocab,)`. Get the top
   indices with `np.argsort(-probs)[:k]`, copy those entries into
   `np.zeros_like(probs)`, then divide by the sum. Common mistake: forgetting
   to renormalise.
5. **`nucleus_filter(probs, p)`.** Returns shape `(vocab,)`. Sort, take
   `np.cumsum`, find the cutoff with `np.searchsorted(..., side="left")`, keep
   indices up to and including it, renormalise. Common mistake: an
   off-by-one that drops the token that pushes the total past $p$. The input
   isn't sorted, so sort it yourself.
6. **`KVCache`.** In `__init__`, make one empty list of keys and one of values
   per layer. `append(layer, k, v)` adds the `(n_kv, d_head)` arrays to that
   layer's lists. `get(layer)` returns `(K, V)`, each `(T, n_kv, d_head)`,
   from `np.stack(..., axis=0)`. Common mistake: using `np.concatenate`, which
   gives `(T * n_kv, d_head)`. The checks compare cached attention against a
   full recompute.
7. **`speculative_accept(p, q, draft_token, rng)`.** Returns a plain `int`.
   Draw one `rng.uniform()`, compare with the acceptance chance, and on
   rejection sample with `rng.choice(len(p), p=residual)`. Common mistake:
   using `np.random` instead of the given `rng`, or sampling from $p$ instead
   of the residual on rejection. Both break the distribution check.

## Cost and scaling

The cache grows in direct proportion to every term in its formula. That makes
long contexts expensive to serve, not only slow.

| Model shape | Context | Cache per sequence |
|---|---|---|
| 12 layers, 8 kv heads, head width 64 | 2,048 tokens | 48 MiB |
| 24 layers, 8 kv heads, head width 64 | 8,192 tokens | 384 MiB |
| 32 layers, 8 kv heads, head width 128 | 32,768 tokens | 4.0 GiB |
| 32 layers, 1 kv head, head width 128 | 32,768 tokens | 0.5 GiB |

The last two rows differ only in key-value heads, and the cache shrinks by
exactly 8. A hundred users on the third row need 400 GiB of cache before a
single weight is loaded.

The decoding rules cost little. Greedy and temperature take one pass over the
vocabulary. Top-k and nucleus sort it once, which takes about $V \log V$ steps
for a vocabulary of $V$ tokens.

## Where this breaks

**The cache outgrows memory.** Past some context length, the cache no longer
fits beside the weights. Servers then evict old entries, store the cache in
fewer bits, or cap the context.

**No memory gap to exploit.** Speculative decoding helps only because decode
is memory bound. With a very large serving batch, decode becomes compute
bound, the draft's extra work stops being free, and speculation can slow
things down.

**A weak or slow draft.** If the draft rarely agrees with the target, most
tokens are rejected and you gain nothing. If the draft isn't much cheaper
than the target, there's no slack to use.

**Loops.** Top-k and nucleus sampling reduce repetition but don't stop it. A
peaked distribution can put nearly all its nucleus on the token that
continues a loop. An explicit repetition penalty breaks it reliably.

## The exercise

You build the memory model and the decoding rules from this lesson. Write
`kv_cache_bytes`, and a `KVCache` class whose `append(layer, k, v)` stores one
token's keys and values and whose `get(layer)` returns everything stored so
far, stacked in order. Write `greedy_decode`, `apply_temperature`,
`top_k_filter`, and `nucleus_filter`. Finish with `speculative_accept(p, q,
draft_token, rng)`.

The checks verify the cache size against a hand count and that it shrinks
exactly 8 times with one key-value head instead of 8. They check greedy
choice, temperature against a manual softmax and its greedy limit, and the
exact tokens top-k and nucleus keep. They check that token-by-token attention
through your `KVCache` matches a full recompute, and that
`speculative_accept` reproduces the target distribution over 20,000 draws
even with a badly matched draft.
