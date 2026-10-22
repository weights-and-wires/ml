---
title: "Letting tokens look at each other: scores and softmax weights"
slug: 097-attention-scores-and-weights
part: "Part 6 — Transformers"
summary: A token scores every other token with a dot product, turns the scores into weights with a stable softmax, and reads a weighted blend of their values.
minutes: 40
gpu: false
objectives:
  - Explain why a token needs to read from other tokens, and why a fixed average or a hard pick is not good enough.
  - Compute dot-product scores, softmax weights, and the blended output for a two-token example by hand.
  - Explain query, key, and value as a soft dictionary lookup, and why a token needs three separate vectors.
  - Write a softmax that survives a score of 900 by subtracting the maximum first.
prereqs:
  - 093-attention-as-soft-lookup
  - 096-alignment-and-copy-decoder
exercise: 097-attention-scores-and-weights
---

# Letting tokens look at each other: scores and softmax weights

Take the sentence "the cat sat because it was tired". Which word does "it" stand for? You answered "cat" without effort, but look at what that took: you compared "it" with every earlier word, decided that "cat" fits best, and let the meaning of "cat" flow into your reading of "it". A model that stores one vector per word cannot do this. The vector for "it" is the same whether the sentence is about a cat, a printer, or a storm.

Reading the sentence one word at a time does not fix this cleanly. By the time a recurrent net reaches "it", everything it knows about "cat" has been squeezed through every word in between, and the detail can be gone. Averaging all the word vectors does not fix it either, because an average gives "it" the same blend whatever it is looking for. What "it" needs is a way to choose, from the content of the words, how much to read from each one.

This chapter builds that choice from three small pieces: a score for each pair of words, a softmax that turns scores into shares, and a blend of values using those shares. The previous chapters did this between a decoder and an encoder. Here the words in one sentence do it to each other, which is called self-attention.

## What you already know

- A dot product multiplies matching entries and adds them. It is large and positive when two vectors point the same way.
- A softmax turns a list of scores into positive weights that add up to 1.
- A matrix product `A @ B` takes dot products of the rows of `A` with the columns of `B`.
- A decoder can score each encoder state and read a weighted blend, as in the chapter on attention as a soft lookup.

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

Read Figure 1 as a budget. The word "it" has 100 percent of its attention to hand out. It gives 0.57 of that to "cat", 0.21 to "sat", and small shares to the rest. Whatever "it" reads is then a mixture of the other words, mostly "cat". The whole job of this chapter is to turn that picture into arithmetic: where do the scores come from, and how do they become shares?

## Why not average, and why not pick one?

Two simpler ideas come to mind, and both fail in ways worth seeing.

Averaging gives every word the same share, 1/5 each in Figure 1. The blend is identical no matter what "it" is looking for, so "it" in a sentence about a printer would read the same mixture. The shares must depend on content.

Picking the single best word, the argmax, does depend on content, but it is a hard switch. If the score for "cat" changes from 2.0 to 2.1, nothing at all changes in the output, until the score crosses the other word's score, where the output jumps. A function that is flat almost everywhere has a gradient of zero almost everywhere, so training has nothing to follow. We want a pick that is soft: the best word gets the biggest share, a close second gets a smaller share, and the shares move smoothly when the scores move.

## Query, key, and value

Think of a library. You walk in with a question. Each book has a label on its spine, and inside it has contents. You compare your question with the labels, decide how relevant each book is, and then read a blend of the contents, weighted by relevance. The three roles are separate: what you ask, what the label says, and what is inside.

Attention gives each word three vectors with these roles. The **query** is what this word is looking for. The **key** is the label it shows to others. The **value** is the content it hands over when it is read. Why three vectors and not one? If a single vector played all the roles, the score between two words would be the same in both directions, and a word's score with itself, which is its squared length, would tend to be the biggest in its row, so every word would mostly read itself. The question "it" asks ("which noun do I refer to?") is not the label "it" shows ("I am a pronoun"), and neither is the content it passes on. Three learned projections of the word let each role differ. The next chapter shows where the three vectors come from. Here we take them as given.

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

Second, the **weights**, by softmax. The exponentials are $e^2 = 7.3891$ and $e^1 = 2.7183$, and they add to 10.1074. So the weights are $7.3891 / 10.1074 = 0.7311$ for "cat" and $2.7183 / 10.1074 = 0.2689$ for "sat". They add to 1 because we divided by their sum. Why the exponential, and not just each score divided by the total of the scores? Scores can be negative, and a negative share makes no sense; $e^s$ is positive for any score. It also lets a bigger score win by a growing margin, and it is smooth, so the shares move gently when a score moves.

Third, the **blend**: $0.7311 \times [1, 0] + 0.2689 \times [0, 1] = [0.7311, 0.2689]$. The output is mostly the "cat" value, with a trace of "sat". Figure 1 differs because it has five words, so the shares are spread over more candidates, but the mechanism is the same.

## The formulas, read aloud

For one query $q$ and keys $k_1, \dots, k_n$, the score of key $j$ is a dot product, and the weights come from the softmax:

$$
s_j = q \cdot k_j, \qquad w_j = \frac{e^{s_j}}{\sum_{m} e^{s_m}}
$$

Read aloud: the score is how well the query and key agree, and the weight is "my share of the exponentials". The output is the blend of the value rows $v_j$:

$$
\text{output} = \sum_j w_j \, v_j
$$

Read aloud: add up each value, scaled by how much attention it got. Stack all the queries as rows of $Q$ and all the keys as rows of $K$, and the three steps become matrix operations. In numpy:

```python
scores = q @ k.T                 # (T_q, T_k), one dot product per pair
weights = softmax(scores, axis=-1)   # each row sums to 1
output = weights @ v             # (T_q, d_v), each row a blend of value rows
```

## A softmax that does not overflow

A softmax computed as written breaks on large scores. In floating point, $e^{900}$ overflows to infinity, and then infinity divided by infinity is `nan`. Figure 3 shows the cure. Softmax gives the same answer if you subtract the same number from every score, because the factor $e^{-c}$ appears on the top and the bottom and cancels. So subtract the largest score first. The largest entry becomes 0, its exponential is exactly 1, and nothing can overflow. For scores $[900, 0]$ you get $[0, -900]$, then $[1, 0]$ after the exponential, and the weights are $[1, 0]$.

```figure
<svg viewBox="0 0 600 240" role="img" aria-label="Left, bars of scores 2 and 1 and the weights 0.7311 and 0.2689. Right, a naive softmax at score 900 gives nan while subtracting the maximum gives 1 and 0">
<defs><marker id="c097-arrow2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="150.0" y="22.0" text-anchor="middle" class="fig-bold">Scores [2, 1] become weights</text>
<line x1="30.0" y1="190.0" x2="280.0" y2="190.0" class="stroke-muted" />
<rect x="40.0" y="114.0" width="40.0" height="76.0" rx="4" class="fill-a" />
<text x="60.0" y="108.0" text-anchor="middle" class="fig-small fig-mono">2.0</text>
<rect x="95.0" y="152.0" width="40.0" height="38.0" rx="4" class="fill-a" />
<text x="115.0" y="146.0" text-anchor="middle" class="fig-small fig-mono">1.0</text>
<rect x="175.0" y="134.4" width="40.0" height="55.6" rx="4" class="fill-b" />
<text x="195.0" y="128.4" text-anchor="middle" class="fig-small fig-mono">0.7311</text>
<rect x="230.0" y="169.6" width="40.0" height="20.4" rx="4" class="fill-b" />
<text x="250.0" y="163.6" text-anchor="middle" class="fig-small fig-mono">0.2689</text>
<text x="87.0" y="208.0" text-anchor="middle" class="fig-small">scores</text>
<text x="202.0" y="208.0" text-anchor="middle" class="fig-small">weights</text>
<text x="60.0" y="226.0" text-anchor="middle" class="fig-small">cat</text>
<text x="115.0" y="226.0" text-anchor="middle" class="fig-small">sat</text>
<text x="195.0" y="226.0" text-anchor="middle" class="fig-small">cat</text>
<text x="250.0" y="226.0" text-anchor="middle" class="fig-small">sat</text>
<text x="450.0" y="22.0" text-anchor="middle" class="fig-bold">Scores [900, 0], naive versus stable</text>
<rect x="320.0" y="40.0" width="260.0" height="42.0" rx="4" class="fill-soft-b stroke-b" />
<text x="450.0" y="58.0" text-anchor="middle" class="fig-small">naive: e^900 overflows to inf</text>
<text x="450.0" y="75.0" text-anchor="middle" class="fig-small fig-mono">inf / (inf + 1) = nan</text>
<rect x="320.0" y="110.0" width="260.0" height="42.0" rx="4" class="fill-soft-c stroke-c" />
<text x="450.0" y="128.0" text-anchor="middle" class="fig-small">subtract the max first: [0, -900]</text>
<text x="450.0" y="145.0" text-anchor="middle" class="fig-small fig-mono">e^0 = 1, e^-900 = 0, so [1, 0]</text>
<text x="450.0" y="185.0" text-anchor="middle" class="fig-small">Same weights in theory, only one</text>
<text x="450.0" y="202.0" text-anchor="middle" class="fig-small">survives in floating point.</text>
</svg>
Figure 3. Left, scores 2 and 1 next to the weights they produce. Right, scores 900 and 0: the naive route produces nan, and subtracting the maximum first gives the correct [1, 0]. Both routes agree on paper, but only one survives in floating point.
```

```python
shifted = x - x.max(axis=axis, keepdims=True)
exp = np.exp(shifted)
weights = exp / exp.sum(axis=axis, keepdims=True)
```

> [!WARNING]
> Keep `keepdims=True` on both the `max` and the `sum`. Without it, the shapes broadcast against the wrong axis and a `(T, T)` table is divided in the wrong direction, with no error message.

> [!KEY]
> "It" can look back at "cat" because every word scores every other word with a dot product between a query and a key, a softmax turns the scores into shares that sum to 1, and the output is the blend of the values using those shares. Unlike an average, the shares depend on content. Unlike a hard pick, they change smoothly, so gradient descent can learn them.

> [!TIP]
> After the softmax, check that each row sums to 1. A failed check usually means the softmax ran along the wrong axis.

> [!RECAP]
> Before you build:
> - A score is the dot product of one query with one key, and `q @ k.T` gives all of them at once, shape `(T_q, T_k)`.
> - Softmax along each row turns a row of scores into weights that are positive and sum to 1.
> - Subtract the row maximum before `np.exp`, so a score of 900 gives `[1, 0]` rather than `nan`.
> - The output is `weights @ v`, a weighted average of the value rows.
> - Nothing is divided by anything yet. The next chapter explains why that has to change.

## Build it step by step

1. **`dot_scores(q, k)`.** It takes queries `(T_q, d_k)` and keys `(T_k, d_k)` and returns `(T_q, T_k)`. One matrix product does it. The usual mistake is forgetting the transpose of `k`, which gives a shape error or, for square inputs, a silently wrong table.
2. **`softmax(x, axis=-1)`.** It keeps the shape of `x`. Subtract the maximum along `axis` with `keepdims=True`, take `np.exp`, and divide by the sum along `axis`. The usual mistake is skipping the subtraction and getting `nan` at large scores.
3. **`blend_values(weights, v)`.** It takes weights `(T_q, T_k)` and values `(T_k, d_v)` and returns `(T_q, d_v)`. One matrix product again. The usual mistake is writing `v @ weights`, which blends the wrong axis.

## Where this breaks

The mechanism has two honest limits, both addressed in the following chapters. First, with wide vectors the scores get large, and the softmax behaves like the hard pick we rejected. Second, with only one set of weights per word, a word cannot ask two different questions at once. And because each pair's score depends only on the two vectors, the lookup does not know where a word sits in the sentence.

## The exercise

You write `dot_scores`, `softmax`, and `blend_values`. The checks confirm the scores of the worked example ($[2, 1]$, and $[[16, 8], [8, 4]]$ for 64-wide vectors) with no scaling, the weights $[0.7311, 0.2689]$, a finite softmax at 900, rows that sum to 1 along either axis, rows of very different size that each stay finite, a one-hot blend that returns a value row exactly, and the full pipeline from scores to the output $[0.7311, 0.2689]$.
