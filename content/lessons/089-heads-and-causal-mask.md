---
title: "Many heads and the causal mask"
slug: 089-heads-and-causal-mask
part: "Part 6 — Transformers"
summary: Several narrow attentions run side by side so a token can ask several questions at once, and an additive minus-infinity mask hides the future.
minutes: 40
gpu: false
objectives:
  - Explain why one attention pattern per token is limiting, and how heads give each token several.
  - Build multi-head attention by cutting the projections into column blocks, attending in each, concatenating, and applying W_O.
  - Explain why a language model needs a causal mask, and why it is added before the softmax and not after.
  - Show with a worked example that attention without a mask ignores the order of its inputs.
prereqs:
  - 088-scaled-attention-scores
exercise: 089-heads-and-causal-mask
---

# Many heads and the causal mask

Read "the keys to the old cabinet are missing". To read "are" correctly, the model needs "keys", because the verb must agree with a plural subject. It also wants "cabinet", the nearest noun, to know what the keys belong to. In the attention you built in the last two chapters, each word has one list of weights that adds up to 1. If "are" puts 0.7 on "keys", it has only 0.3 left for everything else, and the two readings compete for the same budget. Splitting the budget 0.5 and 0.5 does not help much either, because the two values are then mixed into one vector that cannot be pulled apart again. How can a word ask more than one question at a time?

A second problem appears when the model is a language model that predicts the next word. During training it predicts every position of a sentence at once, and the answer for position $t$ is the word at position $t+1$, which is sitting right there in the input. If position $t$ can read position $t+1$, the cheapest strategy is to copy it, and the model learns nothing about language. We must stop each position from reading the future.

This chapter solves both. Heads give each word several budgets of attention. A mask removes the future.

## What you already know

- Scaled dot-product attention computes `softmax(q @ k.T / sqrt(d_k)) @ v`, and each row of weights sums to 1.
- Queries, keys, and values are made from the token matrix `x` with three weight matrices.
- A softmax gives the same answer if every score is shifted by a constant, and $e^{-\infty} = 0$.
- A weight of exactly 0 means "reads nothing from that token".

## Heads: several questions at the same cost

```figure
<svg viewBox="0 0 620 236" role="img" aria-label="Two attention heads each reading two columns of the input, producing their own two by two weight tables, which are concatenated and multiplied by W_O">
<defs><marker id="c099-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
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
<line x1="160.0" y1="58.0" x2="198.0" y2="44.0" class="stroke-muted" marker-end="url(#c099-arrow)"/>
<line x1="160.0" y1="58.0" x2="198.0" y2="160.0" class="stroke-muted" marker-end="url(#c099-arrow)"/>
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
<line x1="330.0" y1="60.0" x2="356.0" y2="96.0" class="stroke-muted" marker-end="url(#c099-arrow)"/>
<line x1="330.0" y1="176.0" x2="356.0" y2="134.0" class="stroke-muted" marker-end="url(#c099-arrow)"/>
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
<line x1="546.0" y1="120.0" x2="566.0" y2="120.0" class="stroke-muted" marker-end="url(#c099-arrow)"/>
<rect x="568.0" y="96.0" width="38.0" height="48.0" rx="4" class="fill-soft-c stroke-c" />
<text x="587.0" y="125.0" text-anchor="middle" class="fig-bold">W_O</text>
<text x="450.0" y="168.0" text-anchor="middle" class="fig-small">still (2, 4); W_O mixes the strips</text>
</svg>
Figure 1. Two heads on two tokens. Head 0 mostly reads the token itself (0.6698 on the diagonal), head 1 reads both tokens equally. Their outputs sit side by side and W_O combines them. Notice that each head works on only two of the four columns.
```

The idea in Figure 1 is to cut the width into slices and run a separate attention on each slice. With width $d = 4$ and two heads, each head owns $d/2 = 2$ columns. Each head has its own table of weights, so it can ask its own question. The cost is the same as one wide head, because two heads of width 2 do the same total attention arithmetic as one head of width 4.

Here is the example in Figure 1 with real numbers. Two tokens have rows $[1, 0, 1, 1]$ and $[0, 1, 1, 1]$, and for simplicity all four weight matrices are the identity, so queries, keys, and values all equal the rows.

**Head 0** reads columns 0 and 1: the rows $[1, 0]$ and $[0, 1]$. The scores are the dot products, $[[1, 0], [0, 1]]$, divided by $\sqrt{2} = 1.4142$ and not by $\sqrt{4}$, because each head works in width $d_k = 2$. That gives $[[0.7071, 0], [0, 0.7071]]$. The softmax of $[0.7071, 0]$ uses $e^{0.7071} = 2.0281$ and $e^0 = 1$, which add to 3.0281, so the weights are $0.6698$ and $0.3302$. Head 0 mostly reads the token itself.

**Head 1** reads columns 2 and 3: the rows $[1, 1]$ and $[1, 1]$. Every score is $2 / 1.4142 = 1.4142$, so the weights are $[0.5, 0.5]$ and the head averages both tokens. Same sentence, two different readings.

The head outputs are $[0.6698, 0.3302]$ and $[1, 1]$ for token 0, and $[0.3302, 0.6698]$ and $[1, 1]$ for token 1. Put them side by side to restore width 4, then multiply by one more matrix:

$$
\operatorname{MultiHead}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O
$$

Read aloud: attend in each slice, glue the results side by side, and mix them with $W_O$. Why the last matrix? Side by side, the strips cannot talk to each other: head 0's finding lives in columns 0 and 1 and head 1's in columns 2 and 3. $W_O$ is a $d \times d$ matrix, so every output column can combine findings from every head.

Where do the slices come from? Cutting columns $i \cdot d_k$ to $(i+1) \cdot d_k$ out of $Q = X W_Q$ gives the same numbers as multiplying $X$ by the matching columns of $W_Q$. So slicing one big matrix is the same as giving each head its own small $d \times d_k$ projection, and it needs one matrix product instead of many.

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

Back to the second problem. Think of the attention table as a grid, one row per reading position and one column per position being read. Position $t$ may read columns $0$ to $t$ and must not read columns after $t$. That is the lower triangle of the grid, and Figure 2 shows how to enforce it.

```figure
<svg viewBox="0 0 560 280" role="img" aria-label="Left, a five by five causal mask with zeros on and below the diagonal and minus infinity above. Right, the attention weights after softmax, zero above the diagonal and each row summing to one">
<defs><marker id="c099-arrow2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
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
<line x1="262.0" y1="142.0" x2="300.0" y2="142.0" class="stroke-muted" marker-end="url(#c099-arrow2)"/>
<text x="140.0" y="268.0" text-anchor="middle" class="fig-small">row = the reading token, column = the token read</text>
<text x="420.0" y="268.0" text-anchor="middle" class="fig-small">each row sums to 1; zeros above the diagonal</text>
</svg>
Figure 2. A five-token causal mask and the weights it produces from random queries and keys. The left grid holds 0 where reading is allowed and minus infinity where it is not. On the right, the upper triangle is exactly 0 and each row still sums to 1. Row 0 is [1, 0, 0, 0, 0] because token 0 can only read itself.
```

Why minus infinity, and why add it to the scores? The softmax turns a score $s$ into $e^{s}$, and $e^{-\infty} = 0$. So a score of $-\infty$ makes the weight exactly 0, and the softmax divides by the sum over the remaining entries only, so the allowed weights still add up to 1. Take the row of scaled scores $[2, 1]$ for "cat" and a future "sat". Adding the mask row $[0, -\infty]$ gives $[2, -\infty]$, and the weights are $[1, 0]$.

For $T$ tokens the mask $M$ is 0 on and below the diagonal and $-\infty$ above it. The attention becomes $\operatorname{softmax}(Q K^{\top} / \sqrt{d_k} + M)\, V$. In numpy:

```python
mask = np.zeros((T, T))
mask[np.triu_indices(T, k=1)] = -np.inf   # strictly above the diagonal
```

**Why not zero the future weights after the softmax?** It looks equivalent and is not. Softmax the row $[2, 1]$ first and you get $[0.7311, 0.2689]$. Zero the future entry and you have $[0.7311, 0]$. Two things went wrong. The row sums to 0.7311, not 1, so the output is shrunk. And worse, the 0.7311 was computed with the future score in the denominator, so it still carries information about the future. Adding $-\infty$ before the softmax removes the future from the sum itself, giving $[1, 0]$.

```figure
<svg viewBox="0 0 600 226" role="img" aria-label="Two pairs of bars: adding minus infinity before the softmax gives weights 1 and 0 summing to 1, while zeroing the future weight after the softmax gives 0.7311 and 0 summing to 0.7311">
<line x1="40.0" y1="170.0" x2="560.0" y2="170.0" class="stroke-muted" />
<text x="155.0" y="24.0" text-anchor="middle" class="fig-bold">mask before softmax</text>
<rect x="80.0" y="50.0" width="60.0" height="120.0" rx="4" class="fill-c" />
<text x="110.0" y="44.0" text-anchor="middle" class="fig-small fig-mono">1.0000</text>
<text x="110.0" y="188.0" text-anchor="middle" class="fig-small">cat (allowed)</text>
<rect x="160.0" y="170.0" width="60.0" height="1.0" rx="4" class="fill-c" />
<text x="190.0" y="164.0" text-anchor="middle" class="fig-small fig-mono">0.0000</text>
<text x="190.0" y="188.0" text-anchor="middle" class="fig-small">sat (future)</text>
<text x="155.0" y="210.0" text-anchor="middle" class="fig-bold fig-mono">row sum = 1.0000</text>
<text x="415.0" y="24.0" text-anchor="middle" class="fig-bold">zero after softmax</text>
<rect x="340.0" y="82.3" width="60.0" height="87.7" rx="4" class="fill-bad" />
<text x="370.0" y="76.3" text-anchor="middle" class="fig-small fig-mono">0.7311</text>
<text x="370.0" y="188.0" text-anchor="middle" class="fig-small">cat (allowed)</text>
<rect x="420.0" y="170.0" width="60.0" height="1.0" rx="4" class="fill-bad" />
<text x="450.0" y="164.0" text-anchor="middle" class="fig-small fig-mono">0.0000</text>
<text x="450.0" y="188.0" text-anchor="middle" class="fig-small">sat (future)</text>
<text x="415.0" y="210.0" text-anchor="middle" class="fig-bold fig-mono">row sum = 0.7311</text>
<text x="300.0" y="140.0" text-anchor="middle" class="fig-small">vs</text>
</svg>
Figure 3. The same row of scores [2, 1], where the second token is in the future. Masking before the softmax gives [1, 0], which sums to 1 and has no trace of the future. Zeroing after gives [0.7311, 0], which sums to 0.7311 and was still shaped by the future score.
```

> [!WARNING]
> Use `k=1`, not `k=0`. With `k=0` the diagonal is also $-\infty$, so token 0 has nothing to read. Its row is all $-\infty$, the softmax computes $-\infty - (-\infty)$ when it subtracts the row maximum, and the whole row becomes `nan`.

> [!KEY]
> A language model is trained to predict every next word at once, so each position must not read later positions. An additive $-\infty$ mask before the softmax makes those weights exactly 0 and renormalises the rest, which zeroing after the softmax cannot do.

## Order is not in the lookup

Swap the two tokens of the head example, so the rows are $[0, 1, 1, 1]$ then $[1, 0, 1, 1]$. Every pairwise score depends only on the two vectors involved, not on where they sit. So the weight tables are the same tables with rows and columns swapped, and the output rows come out swapped as well: $[0.3302, 0.6698, 1, 1]$ first, then $[0.6698, 0.3302, 1, 1]$. Without a mask, shuffling the input only shuffles the output the same way. "Dog bites man" and "man bites dog" give each word the same blend, until position is added to the vectors. (A causal mask does leak a little order, since each token reads only the tokens before it, but not which of them came first.) Adding position is what the chapter "Where is each token? Sinusoidal position clocks" does.

> [!TIP]
> To test a multi-head implementation, compare it with a loop that does one head at a time using the same weights. It catches the commonest bug: slicing rows or interleaving columns instead of taking contiguous column blocks.

> [!RECAP]
> Before you build:
> - A head is ordinary attention on a block of $d/h$ columns, scaled by $\sqrt{d/h}$.
> - Slice columns, never rows, from the projections `x @ Wq`, `x @ Wk`, and `x @ Wv`.
> - Concatenate head outputs on the last axis and multiply by `Wo`.
> - The causal mask is 0 on and below the diagonal and $-\infty$ strictly above it, added to the scores before the softmax, and every head gets the same mask.
> - Without a mask, permuting the input rows permutes the output rows.

## Build it step by step

1. **`causal_mask(T)`.** Return a `(T, T)` array that is 0 on and below the diagonal and `-np.inf` above. Use `np.triu_indices(T, k=1)`. The usual mistake is `k=0`, which makes row 0 `nan`.
2. **`split_heads(x, heads)`.** Take `(T, d)` and return a list of `heads` arrays, each `(T, d // heads)`, as column blocks. Raise `ValueError` when `heads` does not divide `d`. The usual mistake is slicing rows, `x[i*d_k:(i+1)*d_k]`.
3. **`multi_head_attention(x, Wq, Wk, Wv, Wo, heads, mask=None)`.** Project once, split each projection with `split_heads`, call `scaled_dot_product_attention` on the matching slices with the same `mask`, concatenate on the last axis, and multiply by `Wo`. The usual mistake is applying the mask in only one head, or dividing by $\sqrt{d}$ instead of $\sqrt{d/h}$ (the provided attention function already reads the slice width).

## Where this breaks

A mask row that is entirely $-\infty$ gives `nan`, because the shifted softmax computes $-\infty - (-\infty)$. A causal mask never produces one, since the diagonal stays open. Heads also do not guarantee that different heads learn different jobs. Training can leave several heads doing nearly the same thing, which wastes width.

## The exercise

You write `causal_mask`, `split_heads`, and `multi_head_attention`. The `softmax` and `scaled_dot_product_attention` functions from the earlier chapters are already in the file. The checks confirm the mask structure, exact zeros above the diagonal after a softmax, column-block slicing, one head matching single-head attention times `Wo`, the two-head worked example above, agreement with a head-by-head reference for two and four heads with and without a mask, that later tokens never change earlier outputs in any head, and that shuffling the input rows shuffles the outputs.
