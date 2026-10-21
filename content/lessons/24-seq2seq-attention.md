---
title: Sequence to sequence with attention
slug: 24-seq2seq-attention
part: "Part 4 — Vision and sequences"
summary: "Let a decoder look back at every encoder state with attention, instead of squeezing the whole input into one vector."
minutes: 65
gpu: false
objectives:
  - Explain why one fixed-size summary vector loses information on long inputs.
  - Compute additive and multiplicative attention scores, weights, and the context vector in numpy.
  - Build the backward pass of additive attention and check it against finite differences.
  - Explain attention as a soft dictionary lookup with a query, keys, and values.
prereqs:
  - 23-gated-recurrence
exercise: 24-seq2seq-attention
---

# Sequence to sequence with attention

A translation model has two halves. An encoder reads the source. A decoder writes the output. In the plain design, the encoder hands over only its last hidden state, one vector for the whole input.

That is like translating a paragraph from memory. Short sentences survive. A long sentence does not fit in one fixed vector, and the translation gets worse as the source gets longer. The gradients can be fine. The summary is still too small.

Attention lets the decoder look back. At every output step it scores every encoder state and blends the matches. You build two scorers, one backward pass, and a copy model with hand-set weights.

## What you already know

You already have these tools:

- An encoder turns each source position into a hidden state.
- A decoder writes one output position at a time from its own state.
- A dot product multiplies matching entries and adds them. It is large when two vectors point the same way.
- A softmax turns a list of scores into positive weights that add up to one.
- The slope of tanh at u is 1 minus tanh(u) squared.

## Two keys and one query

Picture a library desk. The query is your request, each key is a spine label, and each value is the book's pages. You take a blend, mostly from the labels that match.

Use two source positions. The query is [1, 0]. The keys are [1, 0] and [0, 1]. The values are [10, 0] and [0, 10].

The first dot product is 1 × 1 + 0 × 0 = 1. The second is 1 × 0 + 0 × 1 = 0. A softmax on [1, 0] uses e^1 = 2.718 and e^0 = 1. The sum is 3.718. The weights are 2.718 / 3.718 = 0.7311 and 1 / 3.718 = 0.2689. The blend is 0.7311 × [10, 0] + 0.2689 × [0, 10] = [7.311, 2.689]. The match leans toward the first value.

This blend is attention. The weights are the attention weights. The blend is the context vector. The Greek letter alpha, written α, is one weight. The same two steps end every kind of attention:

$$
\alpha_i = \frac{e^{e_i}}{\sum_{j} e^{e_j}}, \qquad
c = \sum_{i} \alpha_i \, \text{values}_i
$$

Alpha_i is the softmax of the scores, and c is the weighted sum of the value rows. Scores 1 and 0 again give [7.311, 2.689].

Subtract the largest score before the exponential, or a large score overflows:

```python
weights = np.exp(scores - scores.max())
weights = weights / weights.sum()
context = weights @ values
```

The usual mistake is to forget the subtraction, or to skip dividing by the sum.

The score can be a plain dot, or a dot with a matrix in the middle when the lengths differ:

$$
e_i = s^{\top} h_i \qquad \text{or, with a weight matrix,} \qquad e_i = s^{\top} W h_i
$$

The first form dots query s with key h_i. The second inserts W, shaped `(d_s, d_h)`. An identity W leaves the scores at 1 and 0. You score every key at once:

```python
scores = keys @ query                 # dot form
scores = keys @ (W.T @ query)         # general form
```

The usual mistake is `if W:` rather than `if W is None:`. An array does not behave like a missing value.

> [!KEY]
> Attention scores a query against every key, turns the scores into weights with a softmax, and returns the weighted sum of the values.

In a translator the query is the decoder state, and the keys and values are the encoder states. Here `query` is `(d_s,)`, `keys` are `(T_x, d_h)`, and `values` are `(T_x, d_v)`.

> [!KEY]
> The decoder builds a fresh context from every encoder state at each output step. The whole input no longer has to fit in one vector.

## Scores from a tiny network

Dot products are one way to score. Additive attention uses a tiny network instead. Keep the same query, keys, and values. Let both projections be the identity, and let the learned vector v be [1, 1].

Position 1 adds [1, 0] + [1, 0] = [2, 0]. Then tanh(2) = 0.9640 and tanh(0) = 0, so the score is 0.9640 × 1 + 0 × 1 = 0.9640. Position 2 adds [1, 0] + [0, 1] = [1, 1]. Then tanh(1) = 0.7616 twice, and 0.7616 + 0.7616 = 1.5232.

Subtract the max, 1.5232. The shifted scores are −0.5592 and 0. Then e^{−0.5592} = 0.5717 and e^0 = 1. The sum is 1.5717. The weights are 0.5717 / 1.5717 = 0.3637 and 1 / 1.5717 = 0.6363. Position 2 wins. Two moderate tanh entries add up to more than one large entry.

Flip v to [−1, −1] and the weights swap to 0.6363 and 0.3637, so the matching key wins. The copy task uses a large negative v.

This scorer is additive attention. The vector v sets what counts as a match.

$$
e_i = v^{\top} \tanh\left(W_1 s + W_2 h_i\right)
$$

W_1, shaped `(d, d_s)`, multiplies the query. W_2, shaped `(d, d_h)`, multiplies the key. Add, apply tanh, and dot with v. Identities and v = [1, 1] give the scores and weights you computed. You score every key in one pass:

```python
u = W1 @ query + keys @ W2.T      # (T_x, d)
scores = np.tanh(u) @ v           # (T_x,)
```

`W1 @ query` has shape `(d,)` and is added to every row. The usual mistake is swapping W_1 and W_2. W_1 goes with the query.

> [!TIP]
> Write the shape of each result in a comment, as the lines above do. A mismatch usually means the two matrices were swapped, or a `.T` was dropped.

> [!KEY]
> In additive attention, the vector v sets what counts as a match. Equality of the query and the key is not enough.

## One gradient on the context

A gradient is the slope of the loss with respect to each input. Suppose the context gradient is [1, 0], and the weights are the additive ones, 0.3637 and 0.6363. The values are still [10, 0] and [0, 10].

Each value row is scaled by its weight. The outer product gives rows [0.3637, 0] and [0.6363, 0]. Each weight's slope is its value dotted with [1, 0], so the weight gradients are [10, 0]. Their weighted average is 0.3637 × 10 + 0.6363 × 0 = 3.637. The score gradients are 0.3637 × (10 − 3.637) = 2.314 and 0.6363 × (0 − 3.637) = −2.314. They sum to 0.

Write t for tanh(u). The backward formulas are:

$$
d\text{values} = \alpha \, \text{dcontext}^{\top}, \qquad
d\alpha = \text{values} \cdot \text{dcontext}
$$

$$
de_i = \alpha_i \left( d\alpha_i - \sum_j \alpha_j \, d\alpha_j \right)
$$

Each value row gets the context gradient times its weight, an outer product. Each weight's gradient is its value dotted with the context gradient. The softmax step subtracts the weighted average, then multiplies by the weight. That is the pair 2.314 and −2.314.

Then the scoring network:

$$
dv = t^{\top} de, \qquad
du = \left(de \, v^{\top}\right) \odot \left(1 - t^2\right)
$$

$$
dW_1 = \Big(\sum_i du_i\Big) s^{\top}, \quad
d\text{query} = W_1^{\top} \sum_i du_i, \quad
dW_2 = du^{\top} \text{keys}, \quad
d\text{keys} = du \, W_2
$$

v's gradient adds each tanh row times its score gradient. The query was added to every row, so sum `du` before multiplying by the query. Each key touches only its own row. One piece in numpy is:

```python
dscores = weights * (dweights - np.sum(weights * dweights))
du = np.outer(dscores, v) * (1.0 - t ** 2)
dW1 = np.outer(du.sum(axis=0), query)
```

The usual mistake is forgetting 1 − t², or passing the full `du` into `np.outer`.

> [!WARNING]
> Sum `du` on axis 0 before `np.outer`. The full `(T_x, d)` array flattens inside `np.outer` and the gradient gets the wrong shape.

## Where the decoder looked

Stack one weight row per decoder step. That T_y by T_x array is the alignment matrix. Row t says where the decoder looked while writing output t. On translation the matrix often traces the word pairs, with no labels supplied.

For a copy task, output t must equal input t, so the ideal matrix is the identity. With queries and keys at 10 times the identity, identity projections, and v = −10 on every entry, the match scores about −10 and every other position scores about −20. The leftover weight is about e^{−10} = 4.54 × 10^{−5}. A positive v makes the other positions win.

> [!RECAP]
> Carry these facts into the exercise:
> - Every attention ends the same way: softmax the scores, then `weights @ values`.
> - Additive scores are `tanh(W1 @ query + keys @ W2.T) @ v`. W_1 goes with the query.
> - Multiplicative scores are `keys @ query`, or `keys @ (W.T @ query)` when you pass W.
> - The backward pass runs values, weights, the softmax step, the tanh step, then the two projections.
> - In the copy task, a negative v makes the matching position win, so the alignment is near the identity.

## Build it step by step

The checks test the functions in this order:

1. **`additive_attention(query, keys, values, W1, W2, v)`.** Return `(context, weights)` with shapes `(d_v,)` and `(T_x,)`. Compute `u`, the scores, `softmax(scores)`, and `weights @ values`. The usual mistake is swapping `W1` and `W2`. `W1` multiplies the query.
2. **`multiplicative_attention(query, keys, values, W=None)`.** Return the same shapes. Use `keys @ query` when `W is None`, and `keys @ (W.T @ query)` otherwise. The usual mistake is `if W:`, which fails on an array.
3. **Convex combination.** There is no separate function. The checks confirm the context equals `weights @ values`, a blend of the value rows.
4. **`attention_backward(dcontext, query, keys, values, W1, W2, v)`.** Rerun the forward pass, then return `(dquery, dkeys, dvalues, dW1, dW2, dv)`, each shaped like its input. Use `np.outer`, `.T @`, and `.sum(axis=0)`. The usual mistake is dropping the 1 − t² factor, or forgetting to sum `du` before the outer product.
5. **Additive copies a dot.** There is no separate function. Tiny identity matrices and `v` equal to the query make tanh nearly linear, so the weights match a scaled dot product.
6. **`decode_with_attention(source_tokens, vocab_size)`.** Return `(decoded, alignment)`: a list of T_x ints and a `(T_x, T_x)` array. Use `10 * np.eye(T_x)` for queries and keys, one-hot rows of `np.eye(vocab_size)` as values, identities for `W1` and `W2`, and `v = -10 * np.ones(T_x)`. At step `t`, call `additive_attention` on query row `t`, store the weights, and append `int(np.argmax(context))`. The usual mistake is a positive `v`.

## Cost and scaling

One decoder step looks at T_x encoder states, with attention width d. A full output of T_y steps multiplies the work by T_y. The costs are:

| Kind | Work per decoder step | Parameters |
|---|---|---|
| Additive | about T_x · d · (d_s + d_h) | d(d_s + d_h) + d |
| Multiplicative, with W | about T_x · d_s · d_h | d_s d_h |
| Multiplicative, dot only | about T_x · d_h | none |

Total work grows with T_x × T_y. Double both lengths and the work goes up four times. The dot form is one matrix product and has no tanh.

## Where this breaks

Attention fixes what the decoder can see. It still fails in familiar ways:

- The decoder can read the same source word at many steps, and the output repeats a phrase. Coverage adds a running total of past weights into the score.
- The encoder is still one step at a time. Attention does not speed that chain up. The next lesson replaces the encoder with attention.
- Attention only mixes what the encoder states hold. A dropped detail does not come back.
- Work grows with both lengths, so long inputs together with long outputs get expensive.

## The exercise

You build `additive_attention`, `multiplicative_attention`, `attention_backward` for the additive form, and `decode_with_attention`. The copy decoder uses weights you set by hand. There is no training loop.

The checks confirm positive weights that sum to one, and a context equal to `weights @ values`. They compare `attention_backward` with finite differences on all six inputs. They also match additive attention to a scaled dot product in the copy setup, and they require `decode_with_attention` to reproduce the source with alignment within 0.01 of the identity.
