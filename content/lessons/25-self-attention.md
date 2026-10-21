---
title: Self-attention
slug: 25-self-attention
part: "Part 5 — Transformers"
summary: Build scaled dot-product attention, multi-head attention, and a causal mask in numpy, then backpropagate through them.
minutes: 60
gpu: false
objectives:
  - Compute scaled dot-product attention and explain why the scores are divided by the square root of the key width.
  - Build multi-head attention by slicing the projections into heads and joining the results.
  - Build a causal mask and explain why it is added before the softmax.
  - Compute the flop count of one attention layer and explain why it grows with the square of the sequence length.
prereqs:
  - 24-seq2seq-attention
exercise: 25-self-attention
---

# Self-attention

In the last lesson a decoder looked at a separate encoder. Self-attention uses the same lookup inside one sentence. Every token supplies a query, a key, and a value, and collects a blend of the values that match its query.

A recurrent net reads one token at a time. Token 10 waits for token 9. Self-attention has no such chain. Every position does its lookup at the same moment, so the whole sentence can run in parallel.

You build the scaled lookup, the heads, the mask, and the flop count.

## What you already know

You already have these tools:

- A dot product multiplies matching entries and adds them. It is large when two vectors point the same way.
- A softmax turns a list of scores into positive weights that add up to one.
- A matrix product `A @ B` multiplies rows of `A` by columns of `B`.
- A gradient is the list of slopes of the loss with respect to each input. The backward pass pushes it from the output back to the inputs.

## A dot product that is too large

Picture two tokens, cat and sat. Each query and each key is 64 copies of one number. That width is d_k.

The cat query is 0.25 in every entry. The cat key is 1 in every entry. One product is 0.25 × 1 = 0.25. Sixty-four of them sum to 64 × 0.25 = 16. The sat key is 0.5 in every entry. One product is 0.25 × 0.5 = 0.125, and 64 × 0.125 = 8.

The sat query is 0.125 in every entry. Its dot with the cat key is 64 × 0.125 = 8. Its dot with the sat key is 64 × 0.0625 = 4. The unscaled score matrix, one row per query, is:

$$
\begin{bmatrix} 16 & 8 \\ 8 & 4 \end{bmatrix}
$$

Softmax the first row, [16, 8]. You get e^{16} = 8886110.52 and e^8 = 2980.96. The sum is 8889091.48. Then 8886110.52 / 8889091.48 = 0.9997 and 2980.96 / 8889091.48 = 0.0003. That is a spike. A small nudge of 16 barely moves the weights, so the gradient is nearly 0.

The width caused the spike. Variance is the average squared distance from the mean. With mean 0 and variance 1 in each entry, the dot product has variance d_k, so a typical score grows like the square root of d_k. At width 64, scores near 8 are ordinary. Dividing by 64 would crush a match. Dividing by 8 brings a typical score near 1.

Divide the matrix by 8:

$$
\begin{bmatrix} 2 & 1 \\ 1 & 0.5 \end{bmatrix}
$$

Softmax of [2, 1]: e^2 = 7.3891 and e^1 = 2.7183. The sum is 10.1074. The weights are 7.3891 / 10.1074 = 0.7311 and 2.7183 / 10.1074 = 0.2689. Sat keeps a real share.

This division is scaled dot-product attention. Three learned matrices turn each token row of X into a query, a key, and a value. The whole lookup is:

$$
Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V
$$

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V
$$

Q K transpose is the table of dots. Divide by the square root of d_k, softmax each row, and mix the values. These queries and keys are Q and K. With value rows [1, 0] and [0, 1], the first output row is [0.7311, 0.2689].

The other scaled row, [1, 0.5], softmaxes to 0.6225 and 0.3775. You write the layer as follows:

```python
scores = Q @ K.T / np.sqrt(Q.shape[-1])   # (T, T)
weights = softmax(scores, axis=-1)        # each row sums to 1
output = weights @ V                      # (T, d_v)
```

Subtract the row max before the exponential. Scores [900, 0] become [0, −900], and the weights are [1, 0]. The usual mistake is to leave the dots unscaled.

> [!KEY]
> Without the square-root scale, scores grow with the key width. The softmax then puts almost all of its weight on one position, and its gradient vanishes.

> [!TIP]
> After the softmax, check that each row sums to 1. A failed check means the softmax ran on the wrong axis.

## Narrow slices, side by side

One head has one match rule. Multi-head attention runs several narrower attentions side by side.

Take width d = 4 and two heads, so each head is d_k = 2 columns wide. A query row [1, 0, 0, 1] splits into [1, 0] and [0, 1]. Head 0 reads columns 0 and 1. Head 1 reads columns 2 and 3. Each head divides by the square root of 2, which is 1.4142, not by the square root of 4.

On the 2 by 2 identity, divide the diagonal 1 by 1.4142 to get 0.7071. Softmax of [0.7071, 0] uses e^{0.7071} = 2.0281 and e^0 = 1. The sum is 3.0281, so the weights are 0.6698 and 0.3302.

You place the head outputs side by side and multiply by one more matrix, W_O:

$$
\operatorname{MultiHead}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O
$$

Each head is ordinary attention on its slice. Concatenation restores width d, and W_O mixes the heads. One head is single-head attention followed by W_O.

```python
d_k = d // heads
lo, hi = i * d_k, (i + 1) * d_k
out_i, _ = scaled_dot_product_attention(
    Q[:, lo:hi], K[:, lo:hi], V[:, lo:hi], mask
)
out = np.concatenate(head_outputs, axis=-1) @ Wo
```

The usual mistake is slicing rows instead of columns.

> [!KEY]
> Each head runs ordinary attention on its own d / h columns. You get several match rules for the same work as one wide head.

## A cover over future tokens

A language model predicts the next token. Position t must not read any position after t.

Take the scaled cat row, scores [2, 1]. Add a mask entry of 0 on the legal key and minus infinity on the future key. The row becomes [2, −∞]. Because e^{−∞} = 0, the weights are [1, 0], and the output is the first value alone, [1, 0]. For two positions the mask is:

$$
M = \begin{bmatrix} 0 & -\infty \\ 0 & 0 \end{bmatrix}
$$

Add M to the scaled scores, then softmax, then multiply by the values. M is 0 on legal keys, including the token itself, and −∞ on future keys. The allowed weights still sum to 1.

```python
mask = np.zeros((T, T))
mask[np.triu_indices(T, k=1)] = -np.inf
```

Add the mask before the softmax. Softmax [2, 1] first and the weights are 0.7311 and 0.2689. Zero the future weight afterwards and you have [0.7311, 0], which sums to 0.7311. The future score already changed the first weight through the denominator. The usual mistake is `k=0`, which masks the diagonal, leaves position 0 with no key, and returns NaN.

> [!WARNING]
> Use `k=1`, not `k=0`. With `k=0` the diagonal is masked, position 0 has nothing to look at, and its row becomes NaN.

> [!KEY]
> Add the mask to the scores before the softmax. Future positions then get weight 0, and the allowed weights still sum to 1.

## Order is not in the lookup

Swap the two rows of X. Queries, keys, and values swap the same way, and so does the output when there is no mask. "Cat sat" and "sat cat" produce the same blends in swapped order. Self-attention has no order of its own. The next lesson adds position to X.

## One gradient through the scale

Use the softened weights [0.7311, 0.2689], value rows [1, 0] and [0, 1], and an upstream gradient [1, −1] on that output row. Because the values are the identity, the weight gradient is also [1, −1].

The weighted row sum is 0.7311 − 0.2689 = 0.4622. Then 0.7311 × (1 − 0.4622) = 0.3932 and 0.2689 × (−1 − 0.4622) = −0.3932. A softmax row sends back a score gradient that sums to 0.

Going backward, one line at a time:

1. `output = weights @ V`, so `dV = weights.T @ dout` and `dweights = dout @ V.T`.
2. For each row, `dscores = weights * (dweights - rowsum)`, with `rowsum` the sum of `weights * dweights`.
3. `scores = Q @ K.T / sqrt(d_k)`, so `dQ = dscores @ K / sqrt(d_k)` and `dK = dscores.T @ Q / sqrt(d_k)`.

The plug-in is [0.3932, −0.3932]. A masked position has weight 0, so its score gradient is 0. The usual mistake is dropping `1 / sqrt(d_k)` on `dQ` and `dK`.

> [!RECAP]
> Carry these facts into the exercise:
> - Scores are Q K transpose, divided by the square root of d_k, and the softmax runs along each row.
> - Subtract each row's maximum before `np.exp`, so large scores stay finite.
> - The causal mask is 0 on and below the diagonal and −∞ strictly above it, added before the softmax.
> - Multi-head attention slices columns into width d / h, attends, concatenates, and multiplies by W_O.
> - The backward pass undoes `weights @ V`, then the softmax, then the scaled product, and keeps the 1 / sqrt(d_k) factor.

## Build it step by step

The checks test the functions in this order:

1. **`causal_mask(T)`.** Return `(T, T)`, with 0 on and below the diagonal and `-np.inf` past it. Use `np.triu_indices(T, k=1)`. The usual mistake is `k=0`, which masks the diagonal and makes position 0 NaN.
2. **`softmax(x, axis=-1)`.** Keep the input shape. Use `np.max`, `np.exp`, and `np.sum` along `axis`, all with `keepdims=True`. The usual mistake is dropping `keepdims`, so the division broadcasts on the wrong axis.
3. **`scaled_dot_product_attention(q, k, v, mask=None)`.** `q` is `(T_q, d_k)`, `k` is `(T_k, d_k)`, and `v` is `(T_k, d_v)`. Return `(output, weights)`, shapes `(T_q, d_v)` and `(T_q, T_k)`. Read d_k from `q.shape[-1]`. Add the mask before the softmax. The usual mistake is adding it afterwards.
4. **`multi_head_attention(x, Wq, Wk, Wv, Wo, heads, mask=None)`.** `x` is `(T, d)` and each weight matrix is `(d, d)`. Return `(T, d)`. Project, loop over column slices, call attention with the same mask on every head, concatenate on the last axis, then multiply by `Wo`. The usual mistake is slicing rows.
5. **`attention_backward(dout, q, k, v, mask=None)`.** Recompute the forward pass, then the three backward steps. Return `(dq, dk, dv)` with the same shapes as `q`, `k`, and `v`. The usual mistake is forgetting `1 / sqrt(d_k)` on `dq` and `dk`.
6. **`attention_flops(T, d, heads)`.** Return the integer `8*T*d**2 + 4*T**2*d`. The `heads` argument is accepted and unused.

## Cost and scaling

A flop is one multiply or one add. A p by q matrix times a q by r matrix costs 2pqr flops. One layer splits as follows:

| Piece | Flops | Grows with |
|---|---|---|
| Projections Q, K, V, and W_O | 8 T d² | T times d² |
| Scores plus the mix with V | 4 T² d | T² times d |

The total is:

$$
\text{flops} = 8 T d^2 + 4 T^2 d
$$

The head count is absent. For T = 16 and d = 4, you get 8 × 16 × 16 + 4 × 256 × 4 = 6144.

```python
flops = 8 * T * d**2 + 4 * T**2 * d
```

Once T is larger than 2d, the second term wins, and doubling the length roughly quadruples the cost. Each head also stores a T by T table for the backward pass.

## Where this breaks

The lookup fails in four familiar ways:

- The T squared term dominates at thousands of tokens, in time and in memory. Tiled attention avoids storing the full table.
- Shuffled input gives shuffled output. Position has to be added to the input.
- One layer can mix any two positions in one step. It cannot chain two lookups. Stacking layers gives that chain.
- A mask row of all −∞ makes the softmax divide 0 by 0 and return NaN. A causal mask leaves the diagonal open.

## The exercise

You build `causal_mask`, `softmax`, `scaled_dot_product_attention`, `multi_head_attention`, `attention_backward`, and `attention_flops`. Every array is one sequence, with no batch axis.

The checks confirm the mask, a finite softmax at 900 that sums to 1, and valid attention rows. Under the causal mask, positions after t do not change earlier outputs. One head matches single-head attention times `Wo`. Shuffling the input shuffles the output. `attention_backward` matches finite differences, and `attention_flops` matches the formula for every head count.
