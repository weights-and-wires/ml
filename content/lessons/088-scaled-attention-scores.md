---
title: "Attention scores, and why we scale them"
slug: 088-scaled-attention-scores
part: "Part 6 — Transformers"
summary: A token scores the others with dot products, turns the scores into softmax weights, and divides by the square root of the width to keep them soft.
minutes: 50
gpu: false
objectives:
  - Explain why a token needs to read from other tokens, and why a fixed average or a hard pick is not good enough.
  - Compute dot-product scores, softmax weights, and the blended output for a two-token example by hand, with a softmax that survives a score of 900.
  - Argue from adding variances why the spread of a dot product grows as the square root of the width, and why dividing by that square root keeps the softmax soft.
  - Write scaled dot-product attention for one sequence.
prereqs:
  - 084-attention-as-soft-lookup
  - 087-alignment-and-copy-decoder
exercise: 088-scaled-attention-scores
---

# Attention scores, and why we scale them

Take the sentence "the cat sat because it was tired". Which word does "it" stand for? You answered "cat" without effort, but look at what that took: you compared "it" with every earlier word, decided that "cat" fits best, and let the meaning of "cat" flow into your reading of "it". A model that stores one vector per word cannot do this. The vector for "it" is the same whether the sentence is about a cat, a printer, or a storm.

Averaging all the word vectors does not help, because an average gives "it" the same blend whatever it is looking for. What "it" needs is a way to choose, from the content of the words, how much to read from each one.

This chapter builds that choice from three small pieces: a score for each pair of words, a softmax that turns scores into shares, and a blend of values using those shares. Then it fixes a trap: make each vector 64 numbers wide instead of 2 and the attention turns all-or-nothing, and training stalls. Nothing changed but the width. You will see why, and why one division repairs it.

## What you already know

- A dot product multiplies matching entries and adds them. It is large and positive when two vectors point the same way.
- A softmax turns a list of scores into positive weights that add up to 1.
- A decoder can score each encoder state and read a weighted blend, as in the chapter on attention as a soft lookup.
- Variances of independent numbers add when you add the numbers.

## The picture: one word reading the others

```figure
<svg viewBox="0 0 600 236" role="img" aria-label="The sentence the cat sat because it, with arrows from the word it to every word, thicker for larger attention weights">
<defs><marker id="c097-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<path d="M534,150 Q300.0,-51.599999999999994 64,150" fill="none" class="stroke-b" stroke-width="3.4" marker-end="url(#c097-arrow)" opacity="0.85"/>
<path d="M534,150 Q360.0,-1.1999999999999886 184,150" fill="none" class="stroke-b" stroke-width="11.1" marker-end="url(#c097-arrow)" opacity="0.85"/>
<path d="M534,150 Q420.0,49.2 304,150" fill="none" class="stroke-b" stroke-width="5.4" marker-end="url(#c097-arrow)" opacity="0.85"/>
<path d="M534,150 Q480.0,99.6 424,150" fill="none" class="stroke-b" stroke-width="3.2" marker-end="url(#c097-arrow)" opacity="0.85"/>
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

Read Figure 1 as a budget. The word "it" has 100 percent of its attention to hand out. It gives 0.57 of that to "cat", 0.21 to "sat", and small shares to the rest. Whatever "it" reads is a mixture of the other words, mostly "cat". Where do the scores come from, and how do they become shares?

Two simpler ideas fail in ways worth seeing. Averaging gives every word the same share, so the blend is identical no matter what "it" is looking for. Picking the single best word, the argmax, does depend on content, but it is a hard switch. If the score for "cat" changes from 2.0 to 2.1, nothing in the output changes, until the score crosses another word's score and the output jumps. A function that is flat almost everywhere has a gradient of zero almost everywhere, so training has nothing to follow. We want a soft pick: the best word gets the biggest share, a close second gets a smaller share, and the shares move smoothly when the scores move.

## Query, key, and value

Think of a library: you walk in with a question, compare it with the label on each book's spine, and read a blend of the contents weighted by relevance. Attention gives each word three vectors with these roles. The **query** is what this word is looking for. The **key** is the label it shows to others. The **value** is the content it hands over when it is read. Why three vectors and not one? With one vector, a word's score with itself, its squared length, would tend to be the biggest in its row, so every word would mostly read itself. The question "it" asks ("which noun do I refer to?") is not the label "it" shows ("I am a pronoun"), and neither is the content it passes on. In a transformer, three learned matrices produce them from each token's row of $X$: $Q = X W_Q$, $K = X W_K$, $V = X W_V$. Here we take them as given.

## A tiny example with real numbers

Keep to two candidates, so every number is visible. The query for "it" is $q = [1, 1]$. The key for "cat" is $[1.5, 0.5]$ and the key for "sat" is $[0.5, 0.5]$. The values are $[1, 0]$ for "cat" and $[0, 1]$ for "sat".

```figure
<svg viewBox="0 0 620 262" role="img" aria-label="Diagram of one query scored against two keys, turned into weights by softmax, and used to blend two value rows into one output">
<defs><marker id="c097-arrow3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="110.0" y="22.0" text-anchor="middle" class="fig-bold">Query</text>
<text x="110.0" y="38.0" text-anchor="middle" class="fig-small">'it' asks</text>
<text x="250.0" y="22.0" text-anchor="middle" class="fig-bold">Keys</text>
<text x="250.0" y="38.0" text-anchor="middle" class="fig-small">labels on shelf</text>
<text x="355.0" y="22.0" text-anchor="middle" class="fig-bold">Score</text>
<text x="355.0" y="38.0" text-anchor="middle" class="fig-small">query · key</text>
<text x="440.0" y="22.0" text-anchor="middle" class="fig-bold">Weight</text>
<text x="440.0" y="38.0" text-anchor="middle" class="fig-small">softmax</text>
<text x="525.0" y="22.0" text-anchor="middle" class="fig-bold">Values</text>
<text x="525.0" y="38.0" text-anchor="middle" class="fig-small">what is stored</text>
<rect x="40.0" y="80.0" width="140.0" height="56.0" rx="4" class="fill-soft-b stroke-b" />
<text x="110.0" y="104.0" text-anchor="middle" class="fig-bold">it</text>
<text x="110.0" y="124.0" text-anchor="middle" class="fig-small fig-mono">[1, 1]</text>
<rect x="190.0" y="60.0" width="120.0" height="40.0" rx="4" class="fill-soft-a stroke-a" />
<text x="250.0" y="78.0" text-anchor="middle" class="fig-bold">cat</text>
<text x="250.0" y="94.0" text-anchor="middle" class="fig-small fig-mono">[1.5, 0.5]</text>
<rect x="190.0" y="130.0" width="120.0" height="40.0" rx="4" class="fill-soft-a stroke-a" />
<text x="250.0" y="148.0" text-anchor="middle" class="fig-bold">sat</text>
<text x="250.0" y="164.0" text-anchor="middle" class="fig-small fig-mono">[0.5, 0.5]</text>
<text x="355.0" y="84.0" text-anchor="middle" class="fig-bold">2</text>
<text x="440.0" y="84.0" text-anchor="middle" class="fig-bold fig-mono">0.7311</text>
<line x1="312.0" y1="80.0" x2="338.0" y2="80.0" class="stroke-muted" marker-end="url(#c097-arrow3)"/>
<line x1="372.0" y1="80.0" x2="404.0" y2="80.0" class="stroke-muted" marker-end="url(#c097-arrow3)"/>
<line x1="476.0" y1="80.0" x2="492.0" y2="80.0" class="stroke-muted" marker-end="url(#c097-arrow3)"/>
<text x="355.0" y="154.0" text-anchor="middle" class="fig-bold">1</text>
<text x="440.0" y="154.0" text-anchor="middle" class="fig-bold fig-mono">0.2689</text>
<line x1="312.0" y1="150.0" x2="338.0" y2="150.0" class="stroke-muted" marker-end="url(#c097-arrow3)"/>
<line x1="372.0" y1="150.0" x2="404.0" y2="150.0" class="stroke-muted" marker-end="url(#c097-arrow3)"/>
<line x1="476.0" y1="150.0" x2="492.0" y2="150.0" class="stroke-muted" marker-end="url(#c097-arrow3)"/>
<line x1="180.0" y1="108.0" x2="188.0" y2="88.0" class="stroke-muted" />
<line x1="180.0" y1="108.0" x2="188.0" y2="148.0" class="stroke-muted" />
<rect x="495.0" y="60.0" width="70.0" height="40.0" rx="4" class="fill-soft-c stroke-c" />
<text x="530.0" y="78.0" text-anchor="middle" class="fig-bold">cat</text>
<text x="530.0" y="94.0" text-anchor="middle" class="fig-small fig-mono">[1, 0]</text>
<rect x="495.0" y="130.0" width="70.0" height="40.0" rx="4" class="fill-soft-c stroke-c" />
<text x="530.0" y="148.0" text-anchor="middle" class="fig-bold">sat</text>
<text x="530.0" y="164.0" text-anchor="middle" class="fig-small fig-mono">[0, 1]</text>
<line x1="565.0" y1="80.0" x2="582.0" y2="80.0" class="stroke-muted"/>
<line x1="565.0" y1="150.0" x2="582.0" y2="150.0" class="stroke-muted"/>
<line x1="582.0" y1="80.0" x2="582.0" y2="196.0" class="stroke-muted" marker-end="url(#c097-arrow3)"/>
<rect x="400.0" y="198.0" width="200.0" height="36.0" rx="4" class="fill-soft-b stroke-b" />
<text x="500.0" y="214.0" text-anchor="middle" class="fig-bold">output</text>
<text x="500.0" y="229.0" text-anchor="middle" class="fig-small fig-mono">0.7311·[1,0] + 0.2689·[0,1]</text>
<text x="500.0" y="252.0" text-anchor="middle" class="fig-small fig-mono">= [0.7311, 0.2689]</text>
</svg>
Figure 2. The whole lookup for one query. Scores 2 and 1 become weights 0.7311 and 0.2689, and both value rows, scaled by those weights, are added into the single output. Follow the left to right flow: score, then share, then blend.
```

First, the **scores**, one dot product per key. For "cat": $1 \times 1.5 + 1 \times 0.5 = 2$. For "sat": $1 \times 0.5 + 1 \times 0.5 = 1$. The query agrees more with the "cat" key, so "cat" scores higher.

Second, the **weights**, by softmax. The exponentials are $e^2 = 7.3891$ and $e^1 = 2.7183$, and they add to 10.1074. So the weights are $7.3891 / 10.1074 = 0.7311$ for "cat" and $2.7183 / 10.1074 = 0.2689$ for "sat". They add to 1 because we divided by their sum. Why the exponential, and not each score divided by the total? Scores can be negative, and a negative share makes no sense, while $e^s$ is always positive. 

Third, the **blend**: $0.7311 \times [1, 0] + 0.2689 \times [0, 1] = [0.7311, 0.2689]$. The output is mostly the "cat" value, with a trace of "sat".

For one query $q$ and keys $k_1, \dots, k_n$, the score of key $j$ is a dot product, the weights come from the softmax, and the output blends the value rows $v_j$:

$$
s_j = q \cdot k_j, \qquad w_j = \frac{e^{s_j}}{\sum_{m} e^{s_m}}, \qquad \text{output} = \sum_j w_j \, v_j
$$

Read aloud: the score is how well the query and key agree, the weight is "my share of the exponentials", and the output adds up each value scaled by its share. For stacked queries and keys, the steps become matrix operations:

```python
scores = q @ k.T                     # (T_q, T_k), one dot product per pair
weights = softmax(scores, axis=-1)   # each row sums to 1
output = weights @ v                 # (T_q, d_v), each row a blend of value rows
```

## A softmax that does not overflow

A softmax computed as written breaks on large scores. In floating point, $e^{900}$ overflows to infinity, and then infinity divided by infinity is `nan`. The cure is that a softmax gives the same answer if you subtract the same number from every score, because the factor $e^{-c}$ appears on the top and the bottom and cancels. So subtract the largest score first. The largest entry becomes 0, its exponential is exactly 1, and nothing can overflow. For scores $[900, 0]$ you get $[0, -900]$, then $[1, 0]$ after the exponential, and the weights are $[1, 0]$.

```python
shifted = x - x.max(axis=axis, keepdims=True)
exp = np.exp(shifted)
weights = exp / exp.sum(axis=axis, keepdims=True)
```

> [!KEY]
> Every word scores every other word with a dot product between a query and a key, a softmax turns the scores into shares that sum to 1, and the output is the blend of the values using those shares. Unlike an average, the shares depend on content, and unlike a hard pick, they change smoothly.

## The symptom: a spike at width 64

Now widen the vectors. Take the same two tokens with 64 entries each, and fill each vector with one repeated number so the arithmetic stays visible. This is a clean case, not typical data. The "cat" query has 0.25 in every entry, the "cat" key has 1 in every entry, and the "sat" key has 0.5 in every entry.

One product is $0.25 \times 1 = 0.25$, and there are 64 of them, so the score for the "cat" key is $64 \times 0.25 = 16$. For the "sat" key each product is $0.25 \times 0.5 = 0.125$, so the score is $64 \times 0.125 = 8$. The scores are $[16, 8]$, and the exponentials are $e^{16} = 8886110.52$ and $e^{8} = 2980.96$. The weights are $0.9997$ and $0.0003$. "Sat" gets a three-thousandth of a share, which is a spike, not a blend.

The spike is more than ugly. When a weight is $w$, nudging its own score changes it at a rate of $w(1 - w)$. At the spike, $0.9997 \times 0.0003 = 0.0003$. Raising the "cat" score by 1 hardly moves anything, because the weight is already pinned near 1, so almost nothing flows back to the queries and keys. At weights $[0.7311, 0.2689]$ the rate is $0.1966$, nearly 600 times larger.

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
Figure 3. The measured spread of q·k for random vectors (blue dots, 2000 samples each) follows the square root of the width (orange dashed). Dividing by that square root gives the green line, which sits at 1 for every width.
```

Figure 3 checks this by simulation. At width 4 the spread is 2.04, at 16 it is 3.94, at 64 it is 7.94, and at 256 it is 16.15, matching $\sqrt{d_k}$ = 2, 4, 8, 16. The result needs the assumptions we stated, zero mean and independent entries. Learned vectors only roughly obey them, but the trend holds.

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
Figure 4. One query against eight random keys. Without scaling, the largest weight climbs from 0.52 to 0.94 as the width grows, so the attention turns into a hard pick. With the division, it stays near 0.35 at every width.
```

What does a growing spread do to a softmax? Figure 4 averages the largest weight over many random queries and eight keys. A flat attention would give each key 0.125, and a hard pick gives 1. Unscaled, the largest weight is 0.52 at width 4, 0.88 at width 64, and 0.94 at width 256. The wider the model, the closer to a hard pick, and the smaller the gradient.

## The fix, and why not something else

Divide every score by $\sqrt{d_k}$. Dividing a quantity by $c$ divides its spread by $c$, so the spread becomes $\sqrt{d_k} / \sqrt{d_k} = 1$ at any width. The softmax then sees inputs of the same size at any width. In the constant example, 16 and 8 divided by $\sqrt{64} = 8$ become 2 and 1, and the weights are the healthy $[0.7311, 0.2689]$ from the tiny example. The green bars in Figure 4 show the same effect on random vectors: the largest weight stays near 0.35 at every width.

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

> [!RECAP]
> Before you build:
> - A score is the dot product of one query with one key, and `q @ k.T` gives all of them at once, shape `(T_q, T_k)`.
> - Softmax along each row turns scores into positive weights that sum to 1. Subtract the row maximum first, so a score of 900 gives `[1, 0]` rather than `nan`.
> - Raw scores have a spread of $\sqrt{d_k}$, so multiply them by $1/\sqrt{d_k}$ to keep the spread near 1 at every width.
> - Attention is `softmax(q @ k.T * scale) @ v`, with the softmax along each row, and you return the weights as well as the output.

## Build it step by step

1. **`dot_scores(q, k)`.** It takes queries `(T_q, d_k)` and keys `(T_k, d_k)` and returns `(T_q, T_k)`, with no scaling. One matrix product does it. The usual mistake is forgetting the transpose of `k`, which gives a shape error or, for square inputs, a silently wrong table.
2. **`softmax(x, axis=-1)`.** It keeps the shape of `x`. Subtract the maximum along `axis` with `keepdims=True`, take `np.exp`, and divide by the sum along `axis`. The usual mistake is skipping the subtraction and getting `nan` at large scores.
3. **`blend_values(weights, v)`.** It takes weights `(T_q, T_k)` and values `(T_k, d_v)` and returns `(T_q, d_v)`. One matrix product again. The usual mistake is writing `v @ weights`, which blends the wrong axis.
4. **`attention_score_scale(d_k)`.** It takes the width and returns $1/\sqrt{d_k}$ as a float. For $d_k = 64$ it is 0.125. The usual mistake is returning $1/d_k$.
5. **`score_std(d_k, n=2000, seed=0)`.** Make `rng = np.random.default_rng(seed)`, draw `q` and `k` as `(n, d_k)` standard normals, take `np.sum(q * k, axis=1)`, and return the standard deviation. The usual mistake is a full matrix product `q @ k.T`, which computes $n^2$ dot products instead of the $n$ you want.
6. **`scaled_dot_product_attention(q, k, v, mask=None)`.** Return `(output, weights)` with shapes `(T_q, d_v)` and `(T_q, T_k)`. Read $d_k$ from `q.shape[-1]`, not from the sequence length. Leave `mask` as `None` here. A later chapter explains it, and if it is given you add it to the scaled scores before the softmax. The usual mistake is applying the softmax along the wrong axis, so rows no longer sum to 1.

## Where this breaks

The scale assumes entries with roughly zero mean and unit variance. If the learned projections drift so that scores become large anyway, the softmax saturates again even with the division, and some models normalise queries and keys to guard against it. Two other limits remain: a word cannot ask two questions at once, and the lookup does not know where a word sits in the sentence.

## The exercise

You write `dot_scores`, `softmax`, `blend_values`, `attention_score_scale`, `score_std`, and `scaled_dot_product_attention`. The checks confirm the unscaled scores of the worked examples, softmax weights $[0.7311, 0.2689]$ that stay finite at 900 and sum to 1, one-hot and uniform blends, the scale for several widths, a measured spread near $\sqrt{d_k}$ before scaling and near 1 after, the exact weights for the 64-wide example, agreement with the written formula, and that at width 256 no row is nearly one-hot.
