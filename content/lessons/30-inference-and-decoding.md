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

Training pushes huge batches forward and backward until the loss stops falling. Serving writes one token at a time while a person waits. Each step is small, and the chip spends it reading weights from memory.

A cache stores keys and values so later steps reuse them. Sharing key and value heads shrinks that store. A small draft model can propose tokens for the large model to check in one pass.

You still end every step with one score per vocabulary token. Here the vocabulary has five tokens. You build greedy choice, temperature, top-k, and nucleus sampling, then the accept-or-replace rule for speculative decoding.

## What you already know

This lesson uses five ideas from earlier work:

- Attention compares a query with every earlier key, then averages the values.
- A softmax turns scores into positive numbers that sum to one.
- Subtracting one number from every score leaves a softmax unchanged.
- A language model writes one token at a time, using every token before it.
- A forward pass costs about two flops per parameter per token.

## See why decode waits on memory

Take batch size $B = 1$ and $b = 2$ bytes per number. A square matrix does $2Bd^2$ flops and reads $bd^2$ bytes. Cancel $d^2$ and the ratio is 1 flop per byte. That ratio is the arithmetic intensity. Below roughly 100 to 300 flops per byte, a big accelerator is memory bound and waits on the read.

$$
I = \frac{2Bd^2}{bd^2} = \frac{2B}{b}
$$

Intensity is twice the batch, divided by bytes per number. The width cancels, so a wider model stays memory bound. The same $B$ and $b$ give 1 again. Prefill reads the whole prompt in one pass and fills the cache. Decode writes one token per pass. A long prompt has a large $B$. Servers batch many users so one weight read serves them all. The usual mistake expects width to make decode compute bound.

> [!KEY]
> Decode reads every weight to write one token, about one flop per byte. It is memory bound at every model size. Batching users shares that read.

## Size a KV cache by hand

Each token produces a key and a value, and later tokens reuse both.

Take 24 layers, 8 key-value heads, head width 64, 2048 tokens, and 2 bytes. Store keys and values: $2 \times 24 = 48$, $48 \times 8 = 384$, $384 \times 64 = 24576$, $24576 \times 2048 = 50331648$, and $50331648 \times 2 = 100663296$ bytes. That is 96 MiB for one sequence.

This store is the KV cache. Each step computes the key and value for that token only, appends them, and attends over the list. Work per step stays flat, so total work grows with length $T$ rather than with $T^2$.

$$
\text{bytes} = 2 \cdot n_{\text{layer}} \cdot n_{\text{kv}} \cdot
d_{\text{head}} \cdot T \cdot \text{bytes per number}
$$

The leading 2 is keys plus values. The same inputs give 100663296 again. The product has this shape:

```python
size = 2 * n_layer * n_kv * d_head * seq_len * bytes_per_elt
```

`append` stores `k` and `v` of shape `(n_kv, d_head)`. After $T$ calls, `get` returns `K` and `V` of shape `(T, n_kv, d_head)` from `np.stack(..., axis=0)`. The usual mistake is `np.concatenate`, which yields `(T * n_kv, d_head)`.

> [!WARNING]
> Stack on axis 0. Concatenate drops the token axis, and later attention reads the wrong shape.

> [!KEY]
> A KV cache computes each token's key and value once and reuses them. You trade repeated arithmetic for memory.

## Share key and value heads

The byte count uses key-value heads, $n_{\text{kv}}$. Query heads can differ. Multi-query attention sets $n_{\text{kv}} = 1$, so every query head shares one key-value head. Grouped-query attention uses $n_{\text{kv}}$ groups, and each group shares one key-value head.

Going from 8 key-value heads to 1 shrinks the cache by 8. The usual mistake sizes the cache from the query-head count.

## Pick one token from five scores

The last layer emits logits, one score per vocabulary id. For `[0.5, -2.0, 3.5, 3.4, -1.0]` the largest entry is 3.5, at index 2. Greedy decoding returns `int(np.argmax(logits))`. A strong first token can still lead into a weaker sentence. The usual mistake returns a numpy scalar.

Temperature $\tau$ (tau, pronounced TOW, as in cow) rescales the scores before a softmax. A softmax exponentiates each score and divides by the total, so the results are positive and sum to one:

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

Each score $z_i$ is divided by $\tau$, exponentiated, and divided by the sum. Below 1, the top token takes almost all the mass, and near 0 the choice matches greedy decoding. Above 1, the scores flatten.

For logits `[2, 1, 0]` at $\tau = 1$, subtract the max to get `[0, -1, -2]`. Exponentials `[1, 0.3679, 0.1353]` sum to 1.5032, and dividing gives `[0.6652, 0.2447, 0.0900]`. Three temperatures compare as follows:

| Temperature | Scaled scores | Probabilities |
|---|---|---|
| 1 | $[2, 1, 0]$ | $[0.6652, 0.2447, 0.0900]$ |
| 2 | $[1, 0.5, 0]$ | $[0.5065, 0.3072, 0.1863]$ |
| 0.5 | $[4, 2, 0]$ | $[0.8668, 0.1173, 0.0159]$ |

Stable softmax has this shape:

```python
s = logits / temperature
probs = np.exp(s - s.max())
probs = probs / probs.sum()
```

The usual mistake skips the max subtraction. At $\tau = 0.001$, `np.exp` overflows to `inf`.

> [!KEY]
> A temperature below 1 sharpens the distribution toward the greedy token. A temperature above 1 flattens it.

Top-k keeps the $k$ largest probabilities, zeros the rest, and divides by the kept total. For `[0.4, 0.3, 0.2, 0.1]` and $k = 2$, keep 0.4 and 0.3. They sum to 0.7, so the result is `[0.5714, 0.4286, 0, 0]`. `np.argsort(-probs)[:k]` returns the indices.

> [!TIP]
> Negate the probabilities before `np.argsort` to put the largest first, then slice `[:k]`.

The usual mistake zeros the tail and skips the division.

Nucleus sampling, or top-p, keeps the shortest prefix of a sorted list whose running total reaches $p$, including the token that crosses $p$. For `[0.1, 0.5, 0.05, 0.2, 0.15]` and $p = 0.7$, the sorted values are 0.5, 0.2, 0.15, 0.1, 0.05 at indices 1, 3, 4, 0, 2. Running totals are 0.5, 0.7, 0.85, 0.95, 1.0. The second total hits 0.7, so keep indices 1 and 3. Then $0.5 / 0.7 = 0.7143$ and $0.2 / 0.7 = 0.2857$.

Use `np.argsort`, `np.cumsum`, and `np.searchsorted(cumulative, p, side="left")`. The input arrives unsorted. The usual mistake drops the token that reaches $p$.

> [!WARNING]
> Keep the token that pushes the running total up to $p$. The `np.searchsorted` index is included.

A low temperature can repeat a phrase. A repetition penalty shrinks the logit of a token already written, before the softmax.

## Accept or replace a drafted token

Decode is memory bound, so a small draft model is cheap beside the target. The draft proposes a few tokens. The target scores them in one pass. Each proposal is accepted or replaced.

Let $p$ be the target and $q$ the draft. Accept drafted token $x$ with probability

$$
\min\left(1, \frac{p(x)}{q(x)}\right)
$$

Accept for sure when the target assigns $x$ at least as much probability as the draft. Otherwise accept with the ratio. On a rejection, draw from the residual $\max(0, p - q)$, divided by its total.

For $p = [0.7, 0.2, 0.1]$ and $q = [0.1, 0.1, 0.8]$, acceptance chances are 1, 1, and 0.125. Accepted mass totals 0.3, so 70 percent of draws are rejected. The residual `[0.6, 0.1, 0]` normalises to `[0.857, 0.143, 0]`, and token 0's chance is $0.1 + 0.7 \times 0.857 = 0.7$. Tokens 1 and 2 land on 0.2 and 0.1, which is $p$. The minimum plus the positive part of the difference equals the first number, so the output is $p(x)$ for every token. A weak draft adds rejections and leaves that distribution alone. The draw has this shape:

```python
accept = rng.uniform() < min(1.0, p[x] / q[x])
residual = np.maximum(p - q, 0.0)
residual = residual / residual.sum()
```

The usual mistake draws the replacement from $p$, or calls `np.random` instead of `rng`.

> [!KEY]
> Accept a draft with probability $\min(1, p(x) / q(x))$. On a rejection, sample the normalised $\max(0, p - q)$. The output follows the target.

> [!RECAP]
> Keep these five facts in front of you while you code:
>
> - Decode intensity is $2B / b$, about one flop per byte when $B = 1$.
> - The cache holds $2 \cdot n_{\text{layer}} \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot T$ numbers. `get` stacks them to `(T, n_kv, d_head)`.
> - Temperature divides the logits by $\tau$. Subtract the maximum before `np.exp`.
> - Top-k and nucleus zero some tokens, then renormalise. Nucleus keeps the token that reaches $p$.
> - Speculative decoding accepts with $\min(1, p(x) / q(x))$, else samples $\max(0, p - q)$ with `rng`.

## Build it step by step

The checks call your functions in this order:

1. `kv_cache_bytes(n_layer, n_kv, d_head, seq_len, bytes_per_elt=2)` returns the product, and 100663296 is the check value. The usual mistake drops the leading 2:

```python
def kv_cache_bytes(n_layer, n_kv, d_head, seq_len, bytes_per_elt=2):
    # 2 * layers * heads * width * tokens * bytes
    return size
```

2. `greedy_decode(logits)` takes `(vocab,)` and returns `int(np.argmax(logits))`. The usual mistake returns a numpy scalar:

```python
def greedy_decode(logits):
    return index  # a plain int
```

3. `apply_temperature(logits, temperature)` returns `(vocab,)` probabilities. Divide, subtract the max, exponentiate, and normalise. The usual mistake skips the max:

```python
def apply_temperature(logits, temperature):
    # exp(logits / temperature - max) / sum
    return probs
```

4. `top_k_filter(probs, k)` keeps `np.argsort(-probs)[:k]` and divides by that sum. The usual mistake skips the division:

```python
def top_k_filter(probs, k):
    # copy the top k into zeros, then divide
    return filtered
```

5. `nucleus_filter(probs, p)` sorts, takes `np.cumsum`, and keeps through `np.searchsorted(..., side="left")`. The usual mistake drops the token that crosses $p$:

```python
def nucleus_filter(probs, p):
    # keep through the searchsorted index, then renormalise
    return filtered
```

6. `KVCache.__init__` makes an empty key list and value list per layer. `append(layer, k, v)` stores `(n_kv, d_head)`. `get(layer)` returns `(T, n_kv, d_head)` from `np.stack` on axis 0. The usual mistake concatenates:

```python
def get(self, layer):
    # stack keys and values on axis 0
    return K, V
```

7. `speculative_accept(p, q, draft_token, rng)` compares `rng.uniform()` with the acceptance chance and, on a rejection, calls `rng.choice(len(p), p=residual)`. The usual mistake samples from $p$ or from `np.random`:

```python
def speculative_accept(p, q, draft_token, rng):
    # rng only, residual on rejection
    return token
```

## Cost and scaling

The cache grows with every factor in its formula, so a long context is expensive to serve. One sequence costs the following:

| Model shape | Context | Cache per sequence |
|---|---|---|
| 12 layers, 8 kv heads, head width 64 | 2048 tokens | 48 MiB |
| 24 layers, 8 kv heads, head width 64 | 8192 tokens | 384 MiB |
| 32 layers, 8 kv heads, head width 128 | 32768 tokens | 4.0 GiB |
| 32 layers, 1 kv head, head width 128 | 32768 tokens | 0.5 GiB |

The last two rows differ only in key-value heads, and the cache shrinks by 8. A hundred users on the third row need 400 GiB of cache before a weight is loaded. Greedy choice and temperature walk the vocabulary once. Top-k and nucleus sort it, about $V \log V$ steps.

## Where this breaks

Past some context length the cache no longer fits beside the weights. Servers evict old entries, store fewer bits, or cap the context. A very large batch makes decode compute bound, so a draft can slow the step. A draft that rarely matches the target, or that costs almost as much, adds work and little speed. A peaked nucleus can still continue a loop. A repetition penalty breaks it.

## The exercise

`kv_cache_bytes` returns the cache size. `KVCache.append(layer, k, v)` stores one token, and `KVCache.get(layer)` stacks everything stored so far. `greedy_decode`, `apply_temperature`, `top_k_filter`, and `nucleus_filter` turn scores into an id or a distribution. `speculative_accept(p, q, draft_token, rng)` accepts or replaces one drafted token.

The checks match the cache size to a hand count, and one key-value head shrinks it by 8. They check greedy choice, temperature, and the tokens top-k and nucleus keep. Attention through `KVCache` matches a full recompute. Over 20000 draws, `speculative_accept` matches the target, including when the draft disagrees.
