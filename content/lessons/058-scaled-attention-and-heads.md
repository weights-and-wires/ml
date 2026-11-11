---
title: "Scaled attention, many heads, and the causal mask"
slug: 058-scaled-attention-and-heads
part: "Part 6 — Transformers"
summary: Tokens score each other with dot products divided by the square root of the width, read through several heads, and a mask hides the future.
minutes: 75
gpu: false
objectives:
  - Compute dot-product scores, softmax weights, and the blended output for a token reading other tokens, with a softmax that survives a score of 900.
  - Argue from adding variances why the spread of a dot product grows as the square root of the width, and why dividing by that square root keeps the softmax soft.
  - Build multi-head attention by cutting the projections into column blocks, attending in each, concatenating, and applying W_O.
  - Explain why a language model needs a causal mask, and why it is added before the softmax and not after.
  - Show with a worked example that attention without a mask ignores the order of its inputs.
prereqs:
  - 056-attention-as-soft-lookup
  - 057-attention-gradients-and-alignment
exercise: 058-scaled-attention-and-heads
---

# Scaled attention, many heads, and the causal mask

Take the sentence "the cat sat because it was tired". Which word does "it" stand for? You answered "cat" without effort, but look at what that took: you compared "it" with every earlier word, decided that "cat" fits best, and let the meaning of "cat" flow into your reading of "it". A model that stores one vector per word cannot do this. The vector for "it" is the same whether the sentence is about a cat, a printer, or a storm. What "it" needs is a way to choose, from the content of the words, how much to read from each one.

You already have the tool: a query scored against keys, a softmax, and a blend of values, as in [attention as a soft lookup](/lessons/056-attention-as-soft-lookup). The difference is the setting. In a transformer the words of one sentence read each other, so every word is both a reader and something to be read. Three traps appear when you build that layer. Make each vector 64 numbers wide instead of 2 and the attention turns all-or-nothing, and training stalls, with nothing changed but the width. A word that needs two things at once, such as the noun a verb agrees with and the noun nearest to it, has only one budget of weights to split. And a model that predicts the next word can cheat by reading the word it is asked to predict.

This chapter builds the layer and fixes each trap in turn. A division by the square root of the width keeps the softmax soft. Several narrow heads give each word several budgets. A mask removes the future.

```figure
<svg viewBox="0 0 600 236" role="img" aria-label="The sentence the cat sat because it, with arrows from the word it to every word, thicker for larger attention weights">
<defs><marker id="c043-arrow-s" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<path d="M534,150 Q300.0,-51.599999999999994 64,150" fill="none" class="stroke-b" stroke-width="3.4" marker-end="url(#c043-arrow-s)" opacity="0.85"/>
<path d="M534,150 Q360.0,-1.1999999999999886 184,150" fill="none" class="stroke-b" stroke-width="11.1" marker-end="url(#c043-arrow-s)" opacity="0.85"/>
<path d="M534,150 Q420.0,49.2 304,150" fill="none" class="stroke-b" stroke-width="5.4" marker-end="url(#c043-arrow-s)" opacity="0.85"/>
<path d="M534,150 Q480.0,99.6 424,150" fill="none" class="stroke-b" stroke-width="3.2" marker-end="url(#c043-arrow-s)" opacity="0.85"/>
<rect x="10.0" y="152.0" width="100.0" height="30.0" rx="4" class="fill-soft-a stroke-a" />
<text x="60.0" y="172.0" text-anchor="middle" class="fig-bold">the</text>
<text x="60.0" y="203.0" text-anchor="middle" class="fig-small fig-mono">score 0.1</text>
<text x="60.0" y="222.0" text-anchor="middle" class="fig-small fig-mono">weight 0.09</text>
<rect x="130.0" y="152.0" width="100.0" height="30.0" rx="4" class="fill-soft-a stroke-a" />
<text x="180.0" y="172.0" text-anchor="middle" class="fig-bold">cat</text>
<text x="180.0" y="203.0" text-anchor="middle" class="fig-small fig-mono">score 2.0</text>
<text x="180.0" y="222.0" text-anchor="middle" class="fig-small fig-mono">weight 0.57</text>
<rect x="250.0" y="152.0" width="100.0" height="30.0" rx="4" class="fill-soft-a stroke-a" />
<text x="300.0" y="172.0" text-anchor="middle" class="fig-bold">sat</text>
<text x="300.0" y="203.0" text-anchor="middle" class="fig-small fig-mono">score 1.0</text>
<text x="300.0" y="222.0" text-anchor="middle" class="fig-small fig-mono">weight 0.21</text>
<rect x="370.0" y="152.0" width="100.0" height="30.0" rx="4" class="fill-soft-a stroke-a" />
<text x="420.0" y="172.0" text-anchor="middle" class="fig-bold">because</text>
<text x="420.0" y="203.0" text-anchor="middle" class="fig-small fig-mono">score 0.0</text>
<text x="420.0" y="222.0" text-anchor="middle" class="fig-small fig-mono">weight 0.08</text>
<rect x="490.0" y="152.0" width="100.0" height="30.0" rx="4" class="fill-soft-b stroke-b" />
<text x="540.0" y="172.0" text-anchor="middle" class="fig-bold">it</text>
<text x="540.0" y="203.0" text-anchor="middle" class="fig-small fig-mono">score -0.3</text>
<text x="540.0" y="222.0" text-anchor="middle" class="fig-small fig-mono">weight 0.06</text>
<text x="300.0" y="16.0" text-anchor="middle" class="fig-small">How much does 'it' read from each word? Thicker arrow, bigger weight.</text>
</svg>
Figure 1. Five words, and the word "it" deciding how much to read from each. The weights add up to 1, and the thick arrow goes to "cat". The word "it" also scores itself (0.06), which is why the five weights add to 1 but only four arrows are drawn. Nothing in the picture says "cat" by rule: the numbers come from scores, which come from the vectors.
```

## What you already know

- Attention is score, softmax, blend: a query is dotted with each key, a softmax turns the scores into weights that sum to 1, and the output is the weighted blend of the value rows.
- A softmax gives the same answer if every score is shifted by a constant, and $e^{-\infty} = 0$.
- A weight of exactly 0 means "reads nothing from that token".
- Variances of independent numbers add when you add the numbers.

## Every word reads every word

Read Figure 1 as a budget. The word "it" has 100 percent of its attention to hand out. It gives 0.57 to "cat", 0.21 to "sat", and small shares to the rest, so whatever "it" reads is a mixture of the other words, mostly "cat". Averaging all the words would give "it" the same blend whatever it is looking for, and a hard pick moves in jumps, so training would have nothing to follow. The soft pick gives the best word the biggest share, and the shares move smoothly when the scores move.

In a transformer, three learned matrices produce a query, a key, and a value from each token's row of the token matrix $X$: $Q = X W_Q$, $K = X W_K$, and $V = X W_V$. Why three vectors and not one? With one vector, a word's score with itself, its squared length, would tend to be the biggest in its row, so every word would mostly read itself. The question "it" asks ("which noun do I refer to?") is not the label "it" shows ("I am a pronoun"), and neither is the content it passes on. This chapter takes $Q$, $K$, and $V$ as given.

Keep to two candidates, so every number is visible. The query for "it" is $q = [1, 1]$. The key for "cat" is $[1.5, 0.5]$ and the key for "sat" is $[0.5, 0.5]$. The values are $[1, 0]$ for "cat" and $[0, 1]$ for "sat". The scores are dot products: $1 \times 1.5 + 1 \times 0.5 = 2$ for "cat" and $1 \times 0.5 + 1 \times 0.5 = 1$ for "sat". The exponentials are $e^2 = 7.3891$ and $e^1 = 2.7183$, which add to 10.1074, so the weights are $0.7311$ and $0.2689$. The output is $0.7311 \times [1, 0] + 0.2689 \times [0, 1] = [0.7311, 0.2689]$, mostly the "cat" value with a trace of "sat".

For stacked queries and keys, the steps become matrix operations, and the softmax runs along each row:

```python
scores = q @ k.T                     # (T_q, T_k), one dot product per pair
weights = softmax(scores, axis=-1)   # each row sums to 1
output = weights @ v                 # (T_q, d_v), each row a blend of value rows
```

A softmax computed as written breaks on large scores, because $e^{900}$ overflows to infinity and infinity divided by infinity is `nan`. The cure is the shift you already know: subtract the largest score in each row first, so the largest entry becomes 0 and its exponential is exactly 1. For scores $[900, 0]$ you get $[0, -900]$, then $[1, 0]$ after the exponential, and the weights are $[1, 0]$.

```python
shifted = x - x.max(axis=axis, keepdims=True)
exp = np.exp(shifted)
weights = exp / exp.sum(axis=axis, keepdims=True)
```

## The symptom: a spike at width 64

Next, widen the vectors. Take the same two tokens with 64 entries each, and fill each vector with one repeated number so the arithmetic stays visible. This is a clean case, not typical data. The "cat" query has 0.25 in every entry, the "cat" key has 1 in every entry, and the "sat" key has 0.5 in every entry.

One product is $0.25 \times 1 = 0.25$, and there are 64 of them, so the score for the "cat" key is $64 \times 0.25 = 16$. For the "sat" key each product is $0.25 \times 0.5 = 0.125$, so the score is $64 \times 0.125 = 8$. The scores are $[16, 8]$, and the exponentials are $e^{16} = 8886110.52$ and $e^{8} = 2980.96$. The weights are $0.9997$ and $0.0003$. "Sat" gets a three-thousandth of a share, which is a spike, not a blend.

The spike is more than ugly. When a weight is $w$, nudging its own score changes it at a rate of $w(1 - w)$. At the spike that rate is $0.9997 \times 0.0003 = 0.0003$, so almost nothing flows back to the queries and keys. At weights $[0.7311, 0.2689]$ the rate is $0.1966$, nearly 600 times larger.

## How fast do scores grow?

Constant vectors hide the cause, so use typical ones. Suppose each entry of $q$ and $k$ is random, with mean 0 and variance 1, and the entries are independent. **Variance** is the average squared distance from the mean, and its square root, the standard deviation, is the typical size of the spread.

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
Figure 2. The measured spread of q·k for random vectors (blue dots, 2000 samples each) follows the square root of the width (orange dashed). Dividing by that square root gives the green line, which sits at 1 for every width.
```

Figure 2 checks this by simulation. At width 4 the spread is 2.04, at 16 it is 3.94, at 64 it is 7.94, and at 256 it is 16.15, matching $\sqrt{d_k}$ = 2, 4, 8, 16. The result needs zero mean and independent entries. Learned vectors only roughly obey them, but the trend holds.

What does a growing spread do to a softmax? Average the largest weight over many random queries and eight keys. A flat attention would give each key 0.125, and a hard pick gives 1. Unscaled, the largest weight is 0.52 at width 4, 0.88 at width 64, and 0.94 at width 256. The wider the model, the closer to a hard pick, and the smaller the gradient.

## The fix, and why not something else

Divide every score by $\sqrt{d_k}$. Dividing a quantity by $c$ divides its spread by $c$, so the spread becomes $\sqrt{d_k} / \sqrt{d_k} = 1$ at any width. The softmax then sees inputs of the same size at any width. In the constant example, 16 and 8 divided by $\sqrt{64} = 8$ become 2 and 1, and the weights are the healthy $[0.7311, 0.2689]$ from the tiny example. On random vectors the largest weight stays near 0.35 at every width.

Why not divide by something else? Dividing by $d_k$ over-corrects. The spread becomes $1/\sqrt{d_k}$, which shrinks as the width grows. In the constant example the weights become $[0.5312, 0.4688]$, almost a plain average, so a clear match is crushed. A fixed constant such as 8 works at width 64 but fails at 256, because the spread keeps growing. Only $\sqrt{d_k}$ tracks the growth exactly. The whole layer is:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V
$$

Read aloud: build the table of dot products, shrink it by $\sqrt{d_k}$, turn each row into weights, and blend the values with them.

```python
scale = 1.0 / np.sqrt(q.shape[-1])
scores = (q @ k.T) * scale                 # (T_q, T_k)
weights = softmax(scores, axis=-1)         # each row sums to 1
output = weights @ v                       # (T_q, d_v)
```

> [!KEY]
> Dot products between random vectors have a spread of $\sqrt{d_k}$, because the variances of $d_k$ independent terms add. Dividing the scores by $\sqrt{d_k}$ restores a spread of 1 at every width, so the softmax stays soft and its gradient stays alive.

> [!TIP]
> If your attention weights look almost one-hot at initialisation, check for a missing scale before you suspect the data.

## Heads: several questions at the same cost

Back to the second trap. Read "the keys to the old cabinet are missing". To read "are" correctly, the model needs "keys", because the verb must agree with a plural subject. It also wants "cabinet", the nearest noun. If "are" puts 0.7 on "keys", it has only 0.3 left for everything else, and splitting the budget evenly mixes the two values into one vector that cannot be pulled apart again.

The idea in Figure 3 is to cut the width into slices and run a separate attention on each slice. With width $d = 4$ and two heads, each head owns $d/2 = 2$ columns. Each head has its own table of weights, so it can ask its own question. The cost is the same as one wide head, because two heads of width 2 do the same total attention arithmetic as one head of width 4.

```figure
<svg viewBox="0 0 620 236" role="img" aria-label="Two attention heads each reading two columns of the input, producing their own two by two weight tables, which are concatenated and multiplied by W_O">
<defs><marker id="c043-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="70.0" y="18.0" text-anchor="middle" class="fig-bold">X (2 tokens, d = 4)</text>
<rect x="20.0" y="30.0" width="34.0" height="28.0" rx="4" class="fill-soft-a stroke-a" />
<text x="37.0" y="48.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="54.0" y="30.0" width="34.0" height="28.0" rx="4" class="fill-soft-a stroke-a" />
<text x="71.0" y="48.0" text-anchor="middle" class="fig-small fig-mono">0</text>
<rect x="88.0" y="30.0" width="34.0" height="28.0" rx="4" class="fill-soft-d stroke-d" />
<text x="105.0" y="48.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="122.0" y="30.0" width="34.0" height="28.0" rx="4" class="fill-soft-d stroke-d" />
<text x="139.0" y="48.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="20.0" y="58.0" width="34.0" height="28.0" rx="4" class="fill-soft-a stroke-a" />
<text x="37.0" y="76.0" text-anchor="middle" class="fig-small fig-mono">0</text>
<rect x="54.0" y="58.0" width="34.0" height="28.0" rx="4" class="fill-soft-a stroke-a" />
<text x="71.0" y="76.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="88.0" y="58.0" width="34.0" height="28.0" rx="4" class="fill-soft-d stroke-d" />
<text x="105.0" y="76.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="122.0" y="58.0" width="34.0" height="28.0" rx="4" class="fill-soft-d stroke-d" />
<text x="139.0" y="76.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<text x="88.0" y="102.0" text-anchor="middle" class="fig-small">cols 0-1 → head 0</text>
<text x="88.0" y="118.0" text-anchor="middle" class="fig-small">cols 2-3 → head 1</text>
<line x1="160.0" y1="58.0" x2="198.0" y2="44.0" class="stroke-muted" marker-end="url(#c043-arrow)"/>
<line x1="160.0" y1="58.0" x2="198.0" y2="160.0" class="stroke-muted" marker-end="url(#c043-arrow)"/>
<text x="260.0" y="16.0" text-anchor="middle" class="fig-bold">head 0 weights</text>
<rect x="200.0" y="24.0" width="60.0" height="34.0" rx="4" class="fill-b" fill-opacity="0.57"/>
<text x="230.0" y="45.0" text-anchor="middle" class="fig-small fig-mono">0.6698</text>
<rect x="260.0" y="24.0" width="60.0" height="34.0" rx="4" class="fill-b" fill-opacity="0.33"/>
<text x="290.0" y="45.0" text-anchor="middle" class="fig-small fig-mono">0.3302</text>
<rect x="200.0" y="58.0" width="60.0" height="34.0" rx="4" class="fill-b" fill-opacity="0.33"/>
<text x="230.0" y="79.0" text-anchor="middle" class="fig-small fig-mono">0.3302</text>
<rect x="260.0" y="58.0" width="60.0" height="34.0" rx="4" class="fill-b" fill-opacity="0.57"/>
<text x="290.0" y="79.0" text-anchor="middle" class="fig-small fig-mono">0.6698</text>
<text x="260.0" y="108.0" text-anchor="middle" class="fig-small">mostly itself</text>
<text x="260.0" y="134.0" text-anchor="middle" class="fig-bold">head 1 weights</text>
<rect x="200.0" y="142.0" width="60.0" height="34.0" rx="4" class="fill-b" fill-opacity="0.45"/>
<text x="230.0" y="163.0" text-anchor="middle" class="fig-small fig-mono">0.5000</text>
<rect x="260.0" y="142.0" width="60.0" height="34.0" rx="4" class="fill-b" fill-opacity="0.45"/>
<text x="290.0" y="163.0" text-anchor="middle" class="fig-small fig-mono">0.5000</text>
<rect x="200.0" y="176.0" width="60.0" height="34.0" rx="4" class="fill-b" fill-opacity="0.45"/>
<text x="230.0" y="197.0" text-anchor="middle" class="fig-small fig-mono">0.5000</text>
<rect x="260.0" y="176.0" width="60.0" height="34.0" rx="4" class="fill-b" fill-opacity="0.45"/>
<text x="290.0" y="197.0" text-anchor="middle" class="fig-small fig-mono">0.5000</text>
<text x="260.0" y="226.0" text-anchor="middle" class="fig-small">both equally</text>
<line x1="330.0" y1="60.0" x2="356.0" y2="96.0" class="stroke-muted" marker-end="url(#c043-arrow)"/>
<line x1="330.0" y1="176.0" x2="356.0" y2="134.0" class="stroke-muted" marker-end="url(#c043-arrow)"/>
<text x="450.0" y="82.0" text-anchor="middle" class="fig-bold">concatenate side by side</text>
<rect x="360.0" y="92.0" width="46.0" height="28.0" rx="4" class="fill-soft-a stroke-a" />
<text x="383.0" y="110.0" text-anchor="middle" class="fig-small fig-mono">0.6698</text>
<rect x="406.0" y="92.0" width="46.0" height="28.0" rx="4" class="fill-soft-a stroke-a" />
<text x="429.0" y="110.0" text-anchor="middle" class="fig-small fig-mono">0.3302</text>
<rect x="452.0" y="92.0" width="46.0" height="28.0" rx="4" class="fill-soft-d stroke-d" />
<text x="475.0" y="110.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="498.0" y="92.0" width="46.0" height="28.0" rx="4" class="fill-soft-d stroke-d" />
<text x="521.0" y="110.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="360.0" y="120.0" width="46.0" height="28.0" rx="4" class="fill-soft-a stroke-a" />
<text x="383.0" y="138.0" text-anchor="middle" class="fig-small fig-mono">0.3302</text>
<rect x="406.0" y="120.0" width="46.0" height="28.0" rx="4" class="fill-soft-a stroke-a" />
<text x="429.0" y="138.0" text-anchor="middle" class="fig-small fig-mono">0.6698</text>
<rect x="452.0" y="120.0" width="46.0" height="28.0" rx="4" class="fill-soft-d stroke-d" />
<text x="475.0" y="138.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="498.0" y="120.0" width="46.0" height="28.0" rx="4" class="fill-soft-d stroke-d" />
<text x="521.0" y="138.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<line x1="546.0" y1="120.0" x2="566.0" y2="120.0" class="stroke-muted" marker-end="url(#c043-arrow)"/>
<rect x="568.0" y="96.0" width="38.0" height="48.0" rx="4" class="fill-soft-c stroke-c" />
<text x="587.0" y="125.0" text-anchor="middle" class="fig-bold">W_O</text>
<text x="450.0" y="168.0" text-anchor="middle" class="fig-small">still (2, 4); W_O mixes the strips</text>
</svg>
Figure 3. Two heads on two tokens. Head 0 mostly reads the token itself (0.6698 on the diagonal), head 1 reads both tokens equally. Their outputs sit side by side and W_O combines them. Notice that each head works on only two of the four columns.
```

Here is the example in Figure 3 with real numbers. Two tokens have rows $[1, 0, 1, 1]$ and $[0, 1, 1, 1]$, and for simplicity all four weight matrices are the identity, so queries, keys, and values all equal the rows.

**Head 0** reads columns 0 and 1: the rows $[1, 0]$ and $[0, 1]$. The scores are the dot products, $[[1, 0], [0, 1]]$, divided by $\sqrt{2} = 1.4142$ and not by $\sqrt{4}$, because each head works in width $d_k = 2$. That gives $[[0.7071, 0], [0, 0.7071]]$. The softmax of $[0.7071, 0]$ uses $e^{0.7071} = 2.0281$ and $e^0 = 1$, which add to 3.0281, so the weights are $0.6698$ and $0.3302$. Head 0 mostly reads the token itself.

**Head 1** reads columns 2 and 3: the rows $[1, 1]$ and $[1, 1]$. Every score is $2 / 1.4142 = 1.4142$, so the weights are $[0.5, 0.5]$ and the head averages both tokens. Same sentence, two different readings.

The head outputs are $[0.6698, 0.3302]$ and $[1, 1]$ for token 0, and $[0.3302, 0.6698]$ and $[1, 1]$ for token 1. Put them side by side to restore width 4, then multiply by one more matrix:

$$
\operatorname{MultiHead}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O
$$

Read aloud: attend in each slice, glue the results side by side, and mix them with $W_O$. Why the last matrix? Side by side, the strips cannot talk to each other: head 0's finding lives in columns 0 and 1 and head 1's in columns 2 and 3. $W_O$ is a $d \times d$ matrix, so every output column can combine findings from every head.

Cutting columns $i \cdot d_k$ to $(i+1) \cdot d_k$ out of $Q = X W_Q$ gives the same numbers as multiplying $X$ by the matching columns of $W_Q$. So slicing one big matrix is the same as giving each head its own small $d \times d_k$ projection, with one matrix product instead of many.

```python
d_k = d // heads
Q, K, V = x @ Wq, x @ Wk, x @ Wv
heads_out = [attend(Q[:, i*d_k:(i+1)*d_k], K[:, i*d_k:(i+1)*d_k], V[:, i*d_k:(i+1)*d_k], mask)
             for i in range(heads)]
out = np.concatenate(heads_out, axis=-1) @ Wo
```

> [!KEY]
> One head gives each token one set of weights, so its questions compete for one budget. Several heads, each with its own table of weights, give each token several independent readings for the same cost, and $W_O$ lets the readings combine.

## The causal mask: hiding the future

Back to the third trap. A language model is trained to predict every position of a sentence at once, and the answer for position $t$ is the word at position $t+1$, which is sitting right there in the input. If position $t$ can read position $t+1$, the cheapest strategy is to copy it, and the model learns nothing about language. So position $t$ may read columns $0$ to $t$ of the attention table and must not read the columns after $t$. Think of the table as a grid, one row per reading position and one column per position being read. The allowed part is the lower triangle, and Figure 4 shows how to enforce it.

```figure
<svg viewBox="0 0 560 280" role="img" aria-label="Left, a five by five causal mask with zeros on and below the diagonal and minus infinity above. Right, the attention weights after softmax, zero above the diagonal and each row summing to one">
<defs><marker id="c043-arrow2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="140.0" y="20.0" text-anchor="middle" class="fig-bold">Mask M added to the scores</text>
<rect x="30.0" y="30.0" width="44.0" height="44.0" rx="4" class="fill-soft-c stroke-c" />
<text x="52.0" y="57.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="74.0" y="30.0" width="44.0" height="44.0" rx="4" class="fill-sunk stroke-muted" />
<text x="96.0" y="57.0" text-anchor="middle" class="fig-mono">−∞</text>
<rect x="118.0" y="30.0" width="44.0" height="44.0" rx="4" class="fill-sunk stroke-muted" />
<text x="140.0" y="57.0" text-anchor="middle" class="fig-mono">−∞</text>
<rect x="162.0" y="30.0" width="44.0" height="44.0" rx="4" class="fill-sunk stroke-muted" />
<text x="184.0" y="57.0" text-anchor="middle" class="fig-mono">−∞</text>
<rect x="206.0" y="30.0" width="44.0" height="44.0" rx="4" class="fill-sunk stroke-muted" />
<text x="228.0" y="57.0" text-anchor="middle" class="fig-mono">−∞</text>
<rect x="30.0" y="74.0" width="44.0" height="44.0" rx="4" class="fill-soft-c stroke-c" />
<text x="52.0" y="101.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="74.0" y="74.0" width="44.0" height="44.0" rx="4" class="fill-soft-c stroke-c" />
<text x="96.0" y="101.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="118.0" y="74.0" width="44.0" height="44.0" rx="4" class="fill-sunk stroke-muted" />
<text x="140.0" y="101.0" text-anchor="middle" class="fig-mono">−∞</text>
<rect x="162.0" y="74.0" width="44.0" height="44.0" rx="4" class="fill-sunk stroke-muted" />
<text x="184.0" y="101.0" text-anchor="middle" class="fig-mono">−∞</text>
<rect x="206.0" y="74.0" width="44.0" height="44.0" rx="4" class="fill-sunk stroke-muted" />
<text x="228.0" y="101.0" text-anchor="middle" class="fig-mono">−∞</text>
<rect x="30.0" y="118.0" width="44.0" height="44.0" rx="4" class="fill-soft-c stroke-c" />
<text x="52.0" y="145.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="74.0" y="118.0" width="44.0" height="44.0" rx="4" class="fill-soft-c stroke-c" />
<text x="96.0" y="145.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="118.0" y="118.0" width="44.0" height="44.0" rx="4" class="fill-soft-c stroke-c" />
<text x="140.0" y="145.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="162.0" y="118.0" width="44.0" height="44.0" rx="4" class="fill-sunk stroke-muted" />
<text x="184.0" y="145.0" text-anchor="middle" class="fig-mono">−∞</text>
<rect x="206.0" y="118.0" width="44.0" height="44.0" rx="4" class="fill-sunk stroke-muted" />
<text x="228.0" y="145.0" text-anchor="middle" class="fig-mono">−∞</text>
<rect x="30.0" y="162.0" width="44.0" height="44.0" rx="4" class="fill-soft-c stroke-c" />
<text x="52.0" y="189.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="74.0" y="162.0" width="44.0" height="44.0" rx="4" class="fill-soft-c stroke-c" />
<text x="96.0" y="189.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="118.0" y="162.0" width="44.0" height="44.0" rx="4" class="fill-soft-c stroke-c" />
<text x="140.0" y="189.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="162.0" y="162.0" width="44.0" height="44.0" rx="4" class="fill-soft-c stroke-c" />
<text x="184.0" y="189.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="206.0" y="162.0" width="44.0" height="44.0" rx="4" class="fill-sunk stroke-muted" />
<text x="228.0" y="189.0" text-anchor="middle" class="fig-mono">−∞</text>
<rect x="30.0" y="206.0" width="44.0" height="44.0" rx="4" class="fill-soft-c stroke-c" />
<text x="52.0" y="233.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="74.0" y="206.0" width="44.0" height="44.0" rx="4" class="fill-soft-c stroke-c" />
<text x="96.0" y="233.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="118.0" y="206.0" width="44.0" height="44.0" rx="4" class="fill-soft-c stroke-c" />
<text x="140.0" y="233.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="162.0" y="206.0" width="44.0" height="44.0" rx="4" class="fill-soft-c stroke-c" />
<text x="184.0" y="233.0" text-anchor="middle" class="fig-mono">0</text>
<rect x="206.0" y="206.0" width="44.0" height="44.0" rx="4" class="fill-soft-c stroke-c" />
<text x="228.0" y="233.0" text-anchor="middle" class="fig-mono">0</text>
<text x="18.0" y="56.0" text-anchor="end" class="fig-small">t0</text>
<text x="18.0" y="100.0" text-anchor="end" class="fig-small">t1</text>
<text x="18.0" y="144.0" text-anchor="end" class="fig-small">t2</text>
<text x="18.0" y="188.0" text-anchor="end" class="fig-small">t3</text>
<text x="18.0" y="232.0" text-anchor="end" class="fig-small">t4</text>
<text x="420.0" y="20.0" text-anchor="middle" class="fig-bold">Weights after the softmax</text>
<rect x="310.0" y="30.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.78"/>
<text x="332.0" y="56.0" text-anchor="middle" class="fig-small fig-mono">1.00</text>
<rect x="354.0" y="30.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.08"/>
<text x="376.0" y="56.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="398.0" y="30.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.08"/>
<text x="420.0" y="56.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="442.0" y="30.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.08"/>
<text x="464.0" y="56.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="486.0" y="30.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.08"/>
<text x="508.0" y="56.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="310.0" y="74.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.63"/>
<text x="332.0" y="100.0" text-anchor="middle" class="fig-small fig-mono">0.78</text>
<rect x="354.0" y="74.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.23"/>
<text x="376.0" y="100.0" text-anchor="middle" class="fig-small fig-mono">0.22</text>
<rect x="398.0" y="74.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.08"/>
<text x="420.0" y="100.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="442.0" y="74.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.08"/>
<text x="464.0" y="100.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="486.0" y="74.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.08"/>
<text x="508.0" y="100.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="310.0" y="118.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.52"/>
<text x="332.0" y="144.0" text-anchor="middle" class="fig-small fig-mono">0.62</text>
<rect x="354.0" y="118.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.18"/>
<text x="376.0" y="144.0" text-anchor="middle" class="fig-small fig-mono">0.14</text>
<rect x="398.0" y="118.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.24"/>
<text x="420.0" y="144.0" text-anchor="middle" class="fig-small fig-mono">0.23</text>
<rect x="442.0" y="118.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.08"/>
<text x="464.0" y="144.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="486.0" y="118.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.08"/>
<text x="508.0" y="144.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="310.0" y="162.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.21"/>
<text x="332.0" y="188.0" text-anchor="middle" class="fig-small fig-mono">0.18</text>
<rect x="354.0" y="162.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.17"/>
<text x="376.0" y="188.0" text-anchor="middle" class="fig-small fig-mono">0.12</text>
<rect x="398.0" y="162.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.41"/>
<text x="420.0" y="188.0" text-anchor="middle" class="fig-small fig-mono">0.47</text>
<rect x="442.0" y="162.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.24"/>
<text x="464.0" y="188.0" text-anchor="middle" class="fig-small fig-mono">0.23</text>
<rect x="486.0" y="162.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.08"/>
<text x="508.0" y="188.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="310.0" y="206.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.23"/>
<text x="332.0" y="232.0" text-anchor="middle" class="fig-small fig-mono">0.21</text>
<rect x="354.0" y="206.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.38"/>
<text x="376.0" y="232.0" text-anchor="middle" class="fig-small fig-mono">0.42</text>
<rect x="398.0" y="206.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.15"/>
<text x="420.0" y="232.0" text-anchor="middle" class="fig-small fig-mono">0.10</text>
<rect x="442.0" y="206.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.22"/>
<text x="464.0" y="232.0" text-anchor="middle" class="fig-small fig-mono">0.21</text>
<rect x="486.0" y="206.0" width="44.0" height="44.0" rx="4" class="fill-b stroke-muted" fill-opacity="0.12"/>
<text x="508.0" y="232.0" text-anchor="middle" class="fig-small fig-mono">0.06</text>
<line x1="262.0" y1="142.0" x2="300.0" y2="142.0" class="stroke-muted" marker-end="url(#c043-arrow2)"/>
<text x="140.0" y="268.0" text-anchor="middle" class="fig-small">row = the reading token, column = the token read</text>
<text x="420.0" y="268.0" text-anchor="middle" class="fig-small">each row sums to 1; zeros above the diagonal</text>
</svg>
Figure 4. A five-token causal mask and the weights it produces from random queries and keys. The left grid holds 0 where reading is allowed and minus infinity where it is not. On the right, the upper triangle is exactly 0 and each row still sums to 1. Row 0 is [1, 0, 0, 0, 0] because token 0 can only read itself.
```

Why minus infinity, and why add it to the scores? The softmax turns a score $s$ into $e^{s}$, and $e^{-\infty} = 0$. So a score of $-\infty$ makes the weight exactly 0, and the softmax divides by the sum over the remaining entries only, so the allowed weights still add up to 1. Take the row of scaled scores $[2, 1]$ for "cat" and a future "sat". Adding the mask row $[0, -\infty]$ gives $[2, -\infty]$, and the weights are $[1, 0]$.

For $T$ tokens the mask $M$ is 0 on and below the diagonal and $-\infty$ above it. The attention becomes $\operatorname{softmax}(Q K^{\top} / \sqrt{d_k} + M)\, V$. In numpy:

```python
mask = np.zeros((T, T))
mask[np.triu_indices(T, k=1)] = -np.inf   # strictly above the diagonal
```

**Why not zero the future weights after the softmax?** It looks equivalent and is not. Softmax the row $[2, 1]$ first and you get $[0.7311, 0.2689]$. Zero the future entry and you have $[0.7311, 0]$. Two things went wrong. The row sums to 0.7311, not 1, so the output is shrunk. And worse, the 0.7311 was computed with the future score in the denominator, so it still carries information about the future. Adding $-\infty$ before the softmax removes the future from the sum itself, giving $[1, 0]$.

> [!WARNING]
> Use `k=1`, not `k=0`. With `k=0` the diagonal is also $-\infty$, so token 0 has nothing to read. Its row is all $-\infty$, the softmax computes $-\infty - (-\infty)$ when it subtracts the row maximum, and the whole row becomes `nan`.

### Order is not in the lookup

Swap the two tokens of the head example, so the rows are $[0, 1, 1, 1]$ then $[1, 0, 1, 1]$. Every pairwise score depends only on the two vectors involved, not on where they sit. So the weight tables are the same tables with rows and columns swapped, and the output rows come out swapped as well: $[0.3302, 0.6698, 1, 1]$ first, then $[0.6698, 0.3302, 1, 1]$. Without a mask, shuffling the input only shuffles the output the same way. "Dog bites man" and "man bites dog" give each word the same blend, until position is added to the vectors. A causal mask does leak a little order, since each token reads only the tokens before it, but not which of them came first. [Position encodings](/lessons/061-position-encodings) add the missing order.

> [!TIP]
> To test a multi-head implementation, compare it with a loop that does one head at a time using the same weights. It catches the commonest bug: slicing rows or interleaving columns instead of taking contiguous column blocks.

> [!KEY]
> A token reads the others by scoring them with dot products, and dividing the scores by $\sqrt{d_k}$ keeps the softmax soft at any width. Heads give each token several budgets of attention for the same cost, and an additive $-\infty$ mask before the softmax makes the future's weights exactly 0 and renormalises the rest, which zeroing after the softmax cannot do.

> [!RECAP]
> Before you build:
> - A score is the dot product of one query with one key, and `q @ k.T` gives all of them at once, shape `(T_q, T_k)`. Softmax along each row, after subtracting the row maximum, so a score of 900 gives `[1, 0]` rather than `nan`.
> - Raw scores have a spread of $\sqrt{d_k}$, so multiply them by $1/\sqrt{d_k}$. Attention is `softmax(q @ k.T * scale) @ v`, and you return the weights as well as the output.
> - A head is ordinary attention on a block of $d/h$ columns, scaled by $\sqrt{d/h}$. Slice columns, never rows, from `x @ Wq`, `x @ Wk`, and `x @ Wv`, concatenate on the last axis, and multiply by `Wo`.
> - The causal mask is 0 on and below the diagonal and $-\infty$ strictly above it, added to the scores before the softmax, and every head gets the same mask.
> - Without a mask, permuting the input rows permutes the output rows.

## Build it step by step

1. **`dot_scores(q, k)`.** It takes queries `(T_q, d_k)` and keys `(T_k, d_k)` and returns `(T_q, T_k)`, with no scaling. One matrix product does it. The usual mistake is forgetting the transpose of `k`, which gives a shape error or, for square inputs, a silently wrong table.
2. **`softmax(x, axis=-1)`.** It keeps the shape of `x`. Subtract the maximum along `axis` with `keepdims=True`, take `np.exp`, and divide by the sum along `axis`. The usual mistake is skipping the subtraction and getting `nan` at large scores.
3. **`blend_values(weights, v)`.** It takes weights `(T_q, T_k)` and values `(T_k, d_v)` and returns `(T_q, d_v)`. One matrix product again. The usual mistake is writing `v @ weights`, which blends the wrong axis.
4. **`attention_score_scale(d_k)`.** It takes the width and returns $1/\sqrt{d_k}$ as a float. For $d_k = 64$ it is 0.125. The usual mistake is returning $1/d_k$.
5. **`score_std(d_k, n=2000, seed=0)`.** Make `rng = np.random.default_rng(seed)`, draw `q` and `k` as `(n, d_k)` standard normals, take `np.sum(q * k, axis=1)`, and return the standard deviation. The usual mistake is a full matrix product `q @ k.T`, which computes $n^2$ dot products instead of the $n$ you want.
6. **`scaled_dot_product_attention(q, k, v, mask=None)`.** Return `(output, weights)` with shapes `(T_q, d_v)` and `(T_q, T_k)`. Read $d_k$ from `q.shape[-1]`, not from the sequence length. When `mask` is given, add it to the scaled scores before the softmax. The usual mistake is applying the softmax along the wrong axis, so rows no longer sum to 1.
7. **`causal_mask(T)`.** Return a `(T, T)` array that is 0 on and below the diagonal and `-np.inf` above. Use `np.triu_indices(T, k=1)`. The usual mistake is `k=0`, which makes row 0 `nan`.
8. **`split_heads(x, heads)`.** Take `(T, d)` and return a list of `heads` arrays, each `(T, d // heads)`, as column blocks. Raise `ValueError` when `heads` does not divide `d`. The usual mistake is slicing rows, `x[i*d_k:(i+1)*d_k]`.
9. **`multi_head_attention(x, Wq, Wk, Wv, Wo, heads, mask=None)`.** Project once, split each projection with `split_heads`, call `scaled_dot_product_attention` on the matching slices with the same `mask`, concatenate on the last axis, and multiply by `Wo`. The usual mistake is applying the mask in only one head, or dividing by $\sqrt{d}$ instead of $\sqrt{d/h}$ (the attention function already reads the slice width).

## Where this breaks

The scale assumes entries with roughly zero mean and unit variance. If the learned projections drift so that scores become large anyway, the softmax saturates again even with the division, and some models normalise queries and keys to guard against it. A mask row that is entirely $-\infty$ gives `nan`, which a causal mask never produces, since the diagonal stays open. Heads also do not guarantee that different heads learn different jobs. Training can leave several heads doing nearly the same thing, which wastes width.

## The exercise

You write `dot_scores`, `softmax`, `blend_values`, `attention_score_scale`, `score_std`, `scaled_dot_product_attention`, `causal_mask`, `split_heads`, and `multi_head_attention`. The checks confirm softmax weights $[0.7311, 0.2689]$ that stay finite at 900, a measured spread near $\sqrt{d_k}$ before scaling and near 1 after, the exact weights for the 64-wide example, and that at width 256 no row is nearly one-hot. For the second half, they confirm the mask structure, column-block slicing, the two-head worked example, and agreement with a head-by-head reference for two and four heads with and without a mask. They also confirm that later tokens never change earlier outputs in any head, and that shuffling the input rows shuffles the outputs.
