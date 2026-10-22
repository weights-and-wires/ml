---
title: "Why scores are divided by the square root of the width"
slug: 098-scaling-the-scores
part: "Part 6 — Transformers"
summary: Dot products grow with the vector width and saturate the softmax, so dividing by the square root of the width keeps scores at unit scale.
minutes: 40
gpu: false
objectives:
  - Explain with numbers why unscaled dot products make the softmax nearly one-hot and its gradient almost zero.
  - Argue from adding variances why the spread of a dot product grows as the square root of the width.
  - Explain why dividing by the square root of d_k fixes this, and why dividing by d_k or by a constant does not.
  - Say how queries, keys, and values are made from X by three matrices, and write scaled dot-product attention on them.
prereqs:
  - 093-attention-as-soft-lookup
  - 097-attention-scores-and-weights
exercise: 098-scaling-the-scores
---

# Why scores are divided by the square root of the width

You built a lookup in the previous chapter, "Letting tokens look at each other: scores and softmax weights", and it worked on vectors of width 2. Suppose you make the model wider, so each query and key has 64 numbers instead of 2, hoping for a richer match. The attention suddenly becomes all-or-nothing: each word puts nearly all of its weight on one other word, and training stalls. Nothing about the idea changed, only the width. Why would width alone break it?

The cause is arithmetic. A dot product adds up one product per entry, so more entries means a bigger sum. A softmax is sensitive to the size of its inputs, so bigger scores make it behave like a hard pick, which gradient descent cannot learn from. This chapter shows the problem with numbers, finds how fast the scores grow, and fixes it with one division.

## What you already know

- A score is the dot product of a query and a key, and `q @ k.T` gives every score at once.
- A softmax turns a row of scores into weights that sum to 1, and a stable one subtracts the row maximum first.
- A blend `weights @ v` averages the value rows with those weights.
- A hard pick, where one weight is 1 and the rest are 0, has no useful gradient.

## The symptom: a spike

Take two tokens, "cat" and "sat", and give every vector 64 entries. To keep the arithmetic visible, fill each vector with one repeated number. This is a deliberately clean case, not typical data. The "cat" query has 0.25 in every entry, the "cat" key has 1 in every entry, and the "sat" key has 0.5 in every entry.

One product is $0.25 \times 1 = 0.25$, and there are 64 of them, so the score for the "cat" key is $64 \times 0.25 = 16$. For the "sat" key each product is $0.25 \times 0.5 = 0.125$, so the score is $64 \times 0.125 = 8$. The scores are $[16, 8]$.

The exponentials are $e^{16} = 8886110.52$ and $e^{8} = 2980.96$, which add to 8889091.48. The weights are $0.9997$ and $0.0003$. "Sat" gets a three-thousandth of a share, which is a spike, not a blend.

The spike is more than ugly. Look at how a weight reacts when its own score moves. If a weight is $w$, nudging its score changes it at a rate of $w(1 - w)$. At the spike, $0.9997 \times 0.0003 = 0.0003$. Raising the "cat" score by 1 hardly moves anything, because the weight is already pinned near 1. A gradient that small passes almost nothing back to the queries and keys. Compare the slope at weights $[0.7311, 0.2689]$, which is $0.7311 \times 0.2689 = 0.1966$, nearly 600 times larger. (The chapter "The gradient and the cost of attention" derives this rule.)

## How fast do scores grow?

Constant vectors show the problem but hide its cause, so use typical ones. Suppose each entry of $q$ and $k$ is random, with mean 0 and variance 1, and the entries are independent. **Variance** is the average squared distance from the mean, and its square root, the standard deviation, is the typical size of the spread.

One product $q_i k_i$ has mean 0 and variance 1. A dot product is a sum of $d_k$ such products. The products are independent, and variances of independent terms add, so the sum has variance $d_k$. The typical size of a score is its standard deviation, $\sqrt{d_k}$.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Line chart of the standard deviation of the dot product of random vectors against key width, growing like the square root of the width, and flat at one after dividing by the square root">
<line x1="70.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-muted" />
<line x1="70.0" y1="200.0" x2="70.0" y2="30.0" class="stroke-muted" />
<line x1="70.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-grid" />
<text x="62.0" y="204.0" text-anchor="end" class="fig-small">0</text>
<line x1="70.0" y1="160.0" x2="560.0" y2="160.0" class="stroke-grid" />
<text x="62.0" y="164.0" text-anchor="end" class="fig-small">4</text>
<line x1="70.0" y1="120.0" x2="560.0" y2="120.0" class="stroke-grid" />
<text x="62.0" y="124.0" text-anchor="end" class="fig-small">8</text>
<line x1="70.0" y1="80.0" x2="560.0" y2="80.0" class="stroke-grid" />
<text x="62.0" y="84.0" text-anchor="end" class="fig-small">12</text>
<line x1="70.0" y1="40.0" x2="560.0" y2="40.0" class="stroke-grid" />
<text x="62.0" y="44.0" text-anchor="end" class="fig-small">16</text>
<text x="70.0" y="218.0" text-anchor="middle" class="fig-small">1</text>
<text x="192.5" y="218.0" text-anchor="middle" class="fig-small">4</text>
<text x="315.0" y="218.0" text-anchor="middle" class="fig-small">16</text>
<text x="437.5" y="218.0" text-anchor="middle" class="fig-small">64</text>
<text x="560.0" y="218.0" text-anchor="middle" class="fig-small">256</text>
<text x="315.0" y="238.0" text-anchor="middle" class="fig-small">key width d_k (log scale)</text>
<text transform="rotate(-90 24 115)" x="24.0" y="115.0" text-anchor="middle" class="fig-small">std</text>
<polyline points="70.0,190.0 85.3,189.1 100.6,188.1 115.9,187.0 131.2,185.9 146.6,184.6 161.9,183.2 177.2,181.7 192.5,180.0 207.8,178.2 223.1,176.2 238.4,174.1 253.8,171.7 269.1,169.2 284.4,166.4 299.7,163.3 315.0,160.0 330.3,156.4 345.6,152.4 360.9,148.1 376.2,143.4 391.6,138.3 406.9,132.7 422.2,126.6 437.5,120.0 452.8,112.8 468.1,104.9 483.4,96.3 498.8,86.9 514.1,76.6 529.4,65.5 544.7,53.3 560.0,40.0" fill="none" class="stroke-b dashed" stroke-width="2"/>
<polyline points="70.0,189.9 192.5,179.6 315.0,160.6 437.5,120.6 560.0,38.5" fill="none" class="stroke-a" stroke-width="2.5"/>
<circle cx="70.0" cy="189.9" r="4.5" class="fill-a"/>
<circle cx="70.0" cy="189.9" r="4.5" class="fill-c"/>
<circle cx="192.5" cy="179.6" r="4.5" class="fill-a"/>
<circle cx="192.5" cy="189.8" r="4.5" class="fill-c"/>
<circle cx="315.0" cy="160.6" r="4.5" class="fill-a"/>
<circle cx="315.0" cy="190.2" r="4.5" class="fill-c"/>
<circle cx="437.5" cy="120.6" r="4.5" class="fill-a"/>
<circle cx="437.5" cy="190.1" r="4.5" class="fill-c"/>
<circle cx="560.0" cy="38.5" r="4.5" class="fill-a"/>
<circle cx="560.0" cy="189.9" r="4.5" class="fill-c"/>
<polyline points="70.0,189.9 192.5,189.8 315.0,190.2 437.5,190.1 560.0,189.9" fill="none" class="stroke-c" stroke-width="2.5"/>
<rect x="80.0" y="36.0" width="10.0" height="10.0" rx="4" class="fill-a" />
<text x="96.0" y="45.0" text-anchor="start" class="fig-small">measured std of q·k</text>
<line x1="80.0" y1="60.0" x2="90.0" y2="60.0" class="stroke-b dashed" stroke-width="2"/>
<text x="96.0" y="64.0" text-anchor="start" class="fig-small">sqrt(d_k)</text>
<rect x="80.0" y="70.0" width="10.0" height="10.0" rx="4" class="fill-c" />
<text x="96.0" y="79.0" text-anchor="start" class="fig-small">after dividing by sqrt(d_k)</text>
</svg>
Figure 1. The measured spread of q·k for random vectors (blue dots, 2000 samples each) follows the square root of the width (orange dashed). Dividing by that square root gives the green line, which sits at 1 for every width.
```

Figure 1 checks this by simulation. At width 4 the spread is 2.04, at 16 it is 3.94, at 64 it is 7.94, and at 256 it is 16.15, matching $\sqrt{d_k}$ = 2, 4, 8, 16. The result needs the assumptions we stated, zero mean and independent entries. Real learned vectors are not exactly like that, but the trend, scores growing with width, is what matters.

```figure
<svg viewBox="0 0 600 226" role="img" aria-label="Grouped bars of the average largest softmax weight over eight random keys at key widths 4, 64, and 256, unscaled versus scaled">
<line x1="60.0" y1="190.0" x2="570.0" y2="190.0" class="stroke-muted" />
<line x1="60.0" y1="190.0" x2="60.0" y2="40.0" class="stroke-muted" />
<line x1="60.0" y1="155.0" x2="570.0" y2="155.0" class="stroke-grid" />
<text x="52.0" y="159.0" text-anchor="end" class="fig-small">0.25</text>
<line x1="60.0" y1="120.0" x2="570.0" y2="120.0" class="stroke-grid" />
<text x="52.0" y="124.0" text-anchor="end" class="fig-small">0.50</text>
<line x1="60.0" y1="85.0" x2="570.0" y2="85.0" class="stroke-grid" />
<text x="52.0" y="89.0" text-anchor="end" class="fig-small">0.75</text>
<line x1="60.0" y1="50.0" x2="570.0" y2="50.0" class="stroke-grid" />
<text x="52.0" y="54.0" text-anchor="end" class="fig-small">1.00</text>
<rect x="100.0" y="117.3" width="50.0" height="72.7" rx="4" class="fill-a" />
<text x="125.0" y="111.3" text-anchor="middle" class="fig-small fig-mono">0.52</text>
<rect x="158.0" y="142.5" width="50.0" height="47.5" rx="4" class="fill-c" />
<text x="183.0" y="136.5" text-anchor="middle" class="fig-small fig-mono">0.34</text>
<text x="154.0" y="208.0" text-anchor="middle" class="fig-small">d_k = 4</text>
<rect x="260.0" y="67.3" width="50.0" height="122.7" rx="4" class="fill-a" />
<text x="285.0" y="61.3" text-anchor="middle" class="fig-small fig-mono">0.88</text>
<rect x="318.0" y="139.9" width="50.0" height="50.1" rx="4" class="fill-c" />
<text x="343.0" y="133.9" text-anchor="middle" class="fig-small fig-mono">0.36</text>
<text x="314.0" y="208.0" text-anchor="middle" class="fig-small">d_k = 64</text>
<rect x="420.0" y="58.0" width="50.0" height="132.0" rx="4" class="fill-a" />
<text x="445.0" y="52.0" text-anchor="middle" class="fig-small fig-mono">0.94</text>
<rect x="478.0" y="139.9" width="50.0" height="50.1" rx="4" class="fill-c" />
<text x="503.0" y="133.9" text-anchor="middle" class="fig-small fig-mono">0.36</text>
<text x="474.0" y="208.0" text-anchor="middle" class="fig-small">d_k = 256</text>
<rect x="300.0" y="12.0" width="10.0" height="10.0" rx="4" class="fill-a" />
<text x="316.0" y="21.0" text-anchor="start" class="fig-small">unscaled</text>
<rect x="400.0" y="12.0" width="10.0" height="10.0" rx="4" class="fill-c" />
<text x="416.0" y="21.0" text-anchor="start" class="fig-small">divided by sqrt(d_k)</text>
<text transform="rotate(-90 24 115)" x="24.0" y="115.0" text-anchor="middle" class="fig-small">largest weight</text>
</svg>
Figure 2. One query against eight random keys. Without scaling, the largest weight climbs from 0.52 to 0.94 as the width grows, so the attention turns into a hard pick. With the division, it stays near 0.35 at every width.
```

What does a growing spread do to a softmax? Figure 2 averages the largest weight over many random queries and eight keys. A flat attention would give each key 0.125, and a hard pick gives 1. Unscaled, the largest weight is 0.52 at width 4, 0.88 at width 64, and 0.94 at width 256. The wider the model, the closer to a hard pick, and the smaller the gradient.

## The fix, and why not something else

Divide every score by $\sqrt{d_k}$. Dividing a quantity by $c$ divides its spread by $c$, so the spread becomes $\sqrt{d_k} / \sqrt{d_k} = 1$ at any width. The softmax then sees inputs of the same size whether the vectors have 4 numbers or 4096. In the constant example, 16 and 8 divided by $\sqrt{64} = 8$ become 2 and 1, and the weights are the healthy $[0.7311, 0.2689]$ from the previous chapter.

Why not divide by something else? Dividing by $d_k$ over-corrects. The spread becomes $\sqrt{d_k}/d_k = 1/\sqrt{d_k}$, which shrinks as the width grows. In the constant example the scores become $[0.25, 0.125]$ and the weights $[0.5312, 0.4688]$, almost a plain average, so a clear match is crushed. Dividing by a fixed constant such as 8 works at width 64 but fails at width 256, because the spread keeps growing. Only $\sqrt{d_k}$ tracks the growth exactly.

> [!KEY]
> Dot products between random vectors have a spread of $\sqrt{d_k}$, because the variances of $d_k$ independent terms add. Dividing the scores by $\sqrt{d_k}$ restores a spread of 1 at every width, so the softmax stays soft and its gradient stays alive.

## Where queries, keys, and values come from

So far the queries and keys were given. In a transformer, each token starts as a row of a matrix $X$ with one row per token. Three learned matrices turn each row into its three roles:

$$
Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V
$$

Read aloud: the query for every token is the token row times one matrix, the key is the same row times a second matrix, and the value is the same row times a third. With $d$ numbers per token and $d_k$ per query and key, $W_Q$ and $W_K$ are $d \times d_k$ and $W_V$ is $d \times d_v$, so $Q$ and $K$ are $T \times d_k$ and $V$ is $T \times d_v$. Three different matrices let one token ask a different question from the label it shows. The exercise starts from $Q$, $K$, and $V$ already made. With these, the whole layer is:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V
$$

Read aloud: build the table of dot products, shrink it by $\sqrt{d_k}$, turn each row into weights, and blend the values with them. Figure 3 follows the shapes through the layer.

```figure
<svg viewBox="0 0 600 220" role="img" aria-label="Data flow of scaled dot-product attention, from the input X through Q, K, V, the score table, the division by the square root of d_k, a row-wise softmax, and the product with V">
<defs><marker id="c098-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<rect x="10.0" y="100.0" width="70.0" height="40.0" rx="4" class="fill-soft-a stroke-a" />
<text x="45.0" y="117.0" text-anchor="middle" class="fig-bold">X</text>
<text x="45.0" y="133.0" text-anchor="middle" class="fig-small fig-mono">(T, d)</text>
<rect x="150.0" y="30.0" width="70.0" height="40.0" rx="4" class="fill-soft-a stroke-a" />
<text x="185.0" y="47.0" text-anchor="middle" class="fig-bold">Q</text>
<text x="185.0" y="63.0" text-anchor="middle" class="fig-small fig-mono">(T, d_k)</text>
<line x1="80.0" y1="120.0" x2="148.0" y2="50.0" class="stroke-muted" marker-end="url(#c098-arrow)"/>
<text x="118.0" y="40.0" text-anchor="middle" class="fig-small">W_Q</text>
<rect x="150.0" y="100.0" width="70.0" height="40.0" rx="4" class="fill-soft-a stroke-a" />
<text x="185.0" y="117.0" text-anchor="middle" class="fig-bold">K</text>
<text x="185.0" y="133.0" text-anchor="middle" class="fig-small fig-mono">(T, d_k)</text>
<line x1="80.0" y1="120.0" x2="148.0" y2="120.0" class="stroke-muted" marker-end="url(#c098-arrow)"/>
<text x="118.0" y="108.0" text-anchor="middle" class="fig-small">W_K</text>
<rect x="150.0" y="170.0" width="70.0" height="40.0" rx="4" class="fill-soft-a stroke-a" />
<text x="185.0" y="187.0" text-anchor="middle" class="fig-bold">V</text>
<text x="185.0" y="203.0" text-anchor="middle" class="fig-small fig-mono">(T, d_v)</text>
<line x1="80.0" y1="120.0" x2="148.0" y2="190.0" class="stroke-muted" marker-end="url(#c098-arrow)"/>
<text x="118.0" y="190.0" text-anchor="middle" class="fig-small">W_V</text>
<rect x="255.0" y="60.0" width="80.0" height="40.0" rx="4" class="fill-soft-b stroke-b" />
<text x="295.0" y="77.0" text-anchor="middle" class="fig-bold">Q Kᵀ</text>
<text x="295.0" y="93.0" text-anchor="middle" class="fig-small fig-mono">(T, T)</text>
<line x1="220.0" y1="50.0" x2="253.0" y2="72.0" class="stroke-muted" marker-end="url(#c098-arrow)"/>
<line x1="220.0" y1="120.0" x2="253.0" y2="90.0" class="stroke-muted" marker-end="url(#c098-arrow)"/>
<rect x="360.0" y="60.0" width="70.0" height="40.0" rx="4" class="fill-soft-b stroke-b" />
<text x="395.0" y="85.0" text-anchor="middle" class="fig-bold">÷ √d_k</text>
<line x1="335.0" y1="80.0" x2="358.0" y2="80.0" class="stroke-muted" marker-end="url(#c098-arrow)"/>
<rect x="455.0" y="60.0" width="70.0" height="40.0" rx="4" class="fill-soft-b stroke-b" />
<text x="490.0" y="77.0" text-anchor="middle" class="fig-bold">softmax</text>
<text x="490.0" y="93.0" text-anchor="middle" class="fig-small fig-mono">each row</text>
<line x1="430.0" y1="80.0" x2="453.0" y2="80.0" class="stroke-muted" marker-end="url(#c098-arrow)"/>
<rect x="450.0" y="150.0" width="110.0" height="50.0" rx="4" class="fill-soft-c stroke-c" />
<text x="505.0" y="172.0" text-anchor="middle" class="fig-bold">weights · V</text>
<text x="505.0" y="188.0" text-anchor="middle" class="fig-small fig-mono">(T, d_v)</text>
<line x1="490.0" y1="100.0" x2="500.0" y2="148.0" class="stroke-muted" marker-end="url(#c098-arrow)"/>
<line x1="220.0" y1="190.0" x2="448.0" y2="180.0" class="stroke-muted" marker-end="url(#c098-arrow)"/>
</svg>
Figure 3. The data flow of one attention layer for T tokens. The score table is T by T, one weight for each pair of tokens, and every row of it is a softmax. The values are blended last.
```

```python
scale = 1.0 / np.sqrt(q.shape[-1])
scores = (q @ k.T) * scale                 # (T, T)
weights = softmax(scores, axis=-1)         # each row sums to 1
output = weights @ v                       # (T, d_v)
```

> [!TIP]
> If your attention weights look almost one-hot at initialisation, check for a missing scale before you suspect the data.

> [!WARNING]
> Scale by the key width `q.shape[-1]`, which is $d_k$, not by the sequence length. A wrong denominator gives weights that look reasonable but a model that trains worse.

> [!RECAP]
> Before you build:
> - Raw scores have a spread of $\sqrt{d_k}$, so multiply them by $1/\sqrt{d_k}$.
> - After scaling, the spread is about 1 at every width, and the softmax is not saturated.
> - Attention is `softmax(q @ k.T * scale) @ v`, with the softmax along each row.
> - You return the weights as well as the output, because later chapters read them.

## Build it step by step

1. **`attention_score_scale(d_k)`.** It takes the width and returns $1/\sqrt{d_k}$ as a float. For $d_k = 64$ it is 0.125. The usual mistake is returning $1/d_k$.
2. **`score_std(d_k, n=2000, seed=0)`.** Make `rng = np.random.default_rng(seed)`, draw `q` and `k` as `(n, d_k)` standard normals, take `np.sum(q * k, axis=1)`, and return the standard deviation. The usual mistake is a full matrix product `q @ k.T`, which computes $n^2$ dot products instead of the $n$ you want.
3. **`scaled_dot_product_attention(q, k, v, mask=None)`.** Return `(output, weights)` with shapes `(T_q, d_v)` and `(T_q, T_k)`. Read $d_k$ from `q.shape[-1]`. Leave `mask` alone in this chapter. The usual mistake is applying the softmax along the wrong axis, so rows no longer sum to 1.

## Where this breaks

The scale assumes entries with roughly zero mean and unit variance. If the learned projections drift so that scores become large anyway, the softmax saturates again even with the division. Some models add a normalisation of queries and keys to guard against this. Dividing by $\sqrt{d_k}$ also does nothing about the sentence length: more keys means more terms in the softmax sum, which is a separate effect.

## The exercise

You write `attention_score_scale`, `score_std`, and `scaled_dot_product_attention`. The `softmax` from the previous chapter is already in the file. The checks confirm the scale for several widths, that the measured spread is near $\sqrt{d_k}$ before scaling and near 1 after, the output shapes and row sums, the exact weights $[0.7311, 0.2689]$ and $[0.6225, 0.3775]$ for the 64-wide example, agreement with the written formula, and that at width 256 no row is nearly one-hot.
