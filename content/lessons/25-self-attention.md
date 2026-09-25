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

In the previous lesson, a decoder looked back at a separate encoder. Self-attention
uses the same lookup, but every word in a sentence looks at every other word in
the same sentence. Each word asks a question, every word offers an answer, and
each word collects a blend of the answers that fit its question best.

This matters because a recurrent network reads one word at a time. It can't
start word 10 until it has finished word 9. Self-attention has no such chain.
Every position does its lookup at the same moment, so a GPU can compute the
whole sentence in parallel.

In this lesson you build that lookup in numpy, step by step. You add a fix that
keeps the numbers well behaved, split the work into several heads, hide the
future from each word, and count what it all costs.

## What you already know

- Attention scores a query against every key, turns the scores into weights
  with a softmax, and returns the weighted average of the values.
- The softmax turns a list of numbers into positive weights that sum to one.
- A matrix product `A @ B` multiplies rows of `A` by columns of `B`.
- The backward pass pushes a gradient from a layer's output back to its inputs.
- A gradient is the list of slopes of the loss with respect to each input.

## Scaled dot-product attention

Think of a library. Your query is what you're looking for. Each book has a key,
its label on the spine, and a value, its contents. You compare your query with
every label, and you take more from the books whose labels match best.

In self-attention the sequence has $T$ positions (tokens). Each position is a
row of $X$, a $T \times d$ matrix, where $d$ is the model width. Three learned
weight matrices turn each row into a query, a key, and a value:

$$
Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V
$$

Read aloud: the query matrix $Q$ is the input $X$ times the weight matrix
$W_Q$, and the same for keys and values. Each result has one row per position.

```python
Q = X @ Wq   # (T, d_k)
K = X @ Wk   # (T, d_k)
V = X @ Wv   # (T, d_v)
```

Then attention itself is one line of maths:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V
$$

Read it piece by piece:

- $Q K^{\top}$ is a $T \times T$ table of scores. Entry $(i, j)$ is the dot
  product of query $i$ with key $j$: how well position $j$ matches what
  position $i$ is looking for.
- $d_k$ is the width of each query and key. You divide every score by
  $\sqrt{d_k}$.
- The softmax runs along each row, so each position's weights are positive and
  sum to one.
- Multiplying by $V$ gives each position a weighted average of the value rows.

```python
scores = Q @ K.T / np.sqrt(Q.shape[-1])   # (T, T)
weights = softmax(scores, axis=-1)        # each row sums to 1
output = weights @ V                      # (T, d_v)
```

> [!TIP]
> After the softmax, check `np.allclose(weights.sum(axis=-1), 1)`. If it fails,
> your softmax ran along the wrong axis.

### Why divide by the square root of d_k

A dot product adds up $d_k$ products. If each query and key entry is a random
number with average 0 and spread 1, the sum has a variance of $d_k$. Variance
measures how spread out a number is, so the typical size of a score grows like
$\sqrt{d_k}$. With $d_k = 64$, scores of size 8 are common.

Large scores are bad for the softmax. One score wins by a mile, its weight
goes to almost 1, and every other weight goes to almost 0. At that point the
softmax is flat: nudging a score barely changes the output, so the gradient is
close to zero and learning stalls. Dividing by $\sqrt{d_k}$ brings the typical
score back to size 1, whatever the width.

> [!KEY]
> Without the $\sqrt{d_k}$ scale, scores grow with the key width. The softmax
> then puts almost all its weight on one position, and its gradient vanishes.

### The softmax, safely

The softmax of a row $z$ is $e^{z_j}$ divided by the sum of $e^{z}$ over the
row. Computing $e^{900}$ overflows to infinity. The fix: subtract the row's
largest value first. That doesn't change the answer, because the same factor
cancels from top and bottom, but now the biggest exponent is $e^0 = 1$.

```python
shifted = z - z.max(axis=axis, keepdims=True)
e = np.exp(shifted)
p = e / e.sum(axis=axis, keepdims=True)
```

### A worked example

Take $T = 2$ positions with $d_k = 2$, and let $Q = K = V = X$, where

$$
X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

1. Scores: $Q K^{\top}$ is the identity matrix. Divide by $\sqrt{2} \approx 1.4142$
   to get $\begin{bmatrix} 0.7071 & 0 \\ 0 & 0.7071 \end{bmatrix}$.
2. Softmax of row 0, $[0.7071, 0]$: $e^{0.7071} = 2.0281$ and $e^0 = 1$. The
   sum is $3.0281$, so the weights are $[0.6698, 0.3302]$. Row 1 is the mirror
   image, $[0.3302, 0.6698]$.
3. Output: row 0 is $0.6698 \cdot [1, 0] + 0.3302 \cdot [0, 1] = [0.6698, 0.3302]$.
   Row 1 is $[0.3302, 0.6698]$.

Each position mostly keeps itself, but blends in a third of its neighbour.

## Multi-head attention

One attention step has one idea of "similar". A word might care about several
things at once: which word is its subject, which word came right before it,
which words share its topic. Multi-head attention runs $h$ smaller attentions,
called heads, side by side. Each head can learn a different idea of similarity.

Here is the recipe. Project the whole input once with $(d, d)$ matrices. Cut
the columns of $Q$, $K$, and $V$ into $h$ equal slices of width $d_k = d / h$.
Run attention on each slice. Glue the $h$ outputs back together along the
columns, and multiply by one more matrix, $W_O$, which mixes the heads:

$$
\operatorname{MultiHead}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O
$$

Read aloud: head $i$ is ordinary attention on the $i$-th slice of columns.
Concatenate means placing the head outputs side by side, which gives a
$T \times d$ matrix again. $W_O$ is $d \times d$.

```python
d_k = d // heads
lo, hi = i * d_k, (i + 1) * d_k
out_i, _ = scaled_dot_product_attention(Q[:, lo:hi], K[:, lo:hi], V[:, lo:hi], mask)
out = np.concatenate(head_outputs, axis=-1) @ Wo
```

Each head divides its scores by $\sqrt{d_k}$ using its own slice width, not the
full $d$. With one head, the slice is the whole matrix, so multi-head attention
with `heads=1` equals single-head attention followed by $W_O$.

Splitting into heads costs nothing extra. You do the same projections, and the
$h$ narrow score tables together take the same work as one wide one.

> [!KEY]
> Each head runs ordinary attention on its own $d / h$ columns. You get several
> ideas of similarity for the same number of flops as one wide head.

## Causal masking

A language model predicts the next word. When it writes word 3, words 4 and
onward don't exist yet. So in training, position $t$ must not peek at any
position after $t$. It's like a test where you cover the answers below the
line you're on.

You block the future by adding a mask $M$ to the scores before the softmax:

$$
\operatorname{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}} + M\right) V,
\qquad
M_{t,j} = \begin{cases} 0 & j \le t \\ -\infty & j > t \end{cases}
$$

Read aloud: the mask is 0 wherever position $t$ may look (itself and
everything before it) and minus infinity wherever it looks ahead. Since
$e^{-\infty} = 0$, a forbidden position gets weight exactly 0, and the softmax
shares the full weight of 1 among the allowed positions.

```python
mask = np.zeros((T, T))
mask[np.triu_indices(T, k=1)] = -np.inf   # strictly above the diagonal
```

> [!WARNING]
> Use `k=1`, not `k=0`. With `k=0` the diagonal is masked too, so position 0
> has nothing left to look at and its softmax returns NaN.

For $T = 3$, the mask is

$$
M = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix}
$$

Apply a $T = 2$ mask to the worked example. Row 0 becomes $[0.7071, -\infty]$,
so its weights are $[1, 0]$ and its output is $[1, 0]$. Row 1 has nothing
masked, so it stays $[0.3302, 0.6698]$.

Why before the softmax and not after? If you zeroed the weights after the
softmax, the denominator would still include the future positions. The weights
left behind would sum to less than 1, and the future would still have leaked
into them through that sum.

> [!KEY]
> Add the mask to the scores before the softmax. Future positions then get
> weight exactly 0, and the allowed weights still sum to one.

## Order doesn't exist yet

Shuffle the rows of $X$. Each of $Q$, $K$, and $V$ is computed row by row, so
their rows shuffle the same way. The softmax works row by row too. The result:
without a mask, the output is the same set of rows, shuffled in the same
order. Self-attention on its own can't tell "dog bites man" from "man bites
dog". The next lesson fixes this by adding position information to $X$.

## Backward pass

To train, you need the gradient of the loss with respect to $Q$, $K$, and $V$.
Call the upstream gradient on the output `dout` (shape $T \times d_v$). Work
backwards through each forward line, one at a time:

1. `output = weights @ V`. So `dV = weights.T @ dout` and
   `dweights = dout @ V.T`.
2. `weights = softmax(scores)`. For each row, the softmax gradient is
   `dscores = weights * (dweights - rowsum)`, where `rowsum` is the sum over
   the row of `weights * dweights`.
3. `scores = Q @ K.T / sqrt(d_k)`. So `dQ = dscores @ K / sqrt(d_k)` and
   `dK = dscores.T @ Q / sqrt(d_k)`.

The mask has no gradient of its own. Masked positions have weight 0, so their
`dscores` is 0 too.

> [!RECAP]
> - Scores are $Q K^{\top} / \sqrt{d_k}$, a $(T_q, T_k)$ table, and the softmax
>   runs along each row.
> - Subtract each row's maximum before `np.exp`, so large scores never overflow.
> - The causal mask is 0 on and below the diagonal and $-\infty$ above it, added
>   before the softmax.
> - Multi-head attention slices columns into heads of width $d / h$, attends on
>   each, concatenates, and multiplies by $W_O$.
> - The backward pass undoes `weights @ V`, then the softmax, then the scaled
>   product, keeping the $1 / \sqrt{d_k}$ factor.

## Build it step by step

The checks test your functions in this order.

1. **`causal_mask(T)`** returns a $(T, T)$ float array: 0 on and below the
   diagonal, `-np.inf` strictly above. Start with `np.zeros((T, T))` and set
   the upper triangle with `np.triu_indices(T, k=1)`. Common mistake: using
   `k=0`, which also masks the diagonal, so position 0 has nothing to look at
   and its row becomes all NaN.
2. **`softmax(x, axis=-1)`** takes any shape and returns the same shape. Use
   `np.max`, `np.exp`, and `np.sum` along `axis`, all with `keepdims=True`.
   Common mistake: forgetting `keepdims`, so the division broadcasts along the
   wrong axis.
3. **`scaled_dot_product_attention(q, k, v, mask=None)`** takes `q` of shape
   $(T_q, d_k)$, `k` of $(T_k, d_k)$, and `v` of $(T_k, d_v)$. It returns a
   tuple `(output, weights)` with shapes $(T_q, d_v)$ and $(T_q, T_k)$. Take
   $d_k$ from `q.shape[-1]`. Common mistake: adding the mask after the softmax.
4. **`multi_head_attention(x, Wq, Wk, Wv, Wo, heads, mask=None)`** takes `x`
   of shape $(T, d)$ and four $(d, d)$ matrices, and returns $(T, d)$. Project,
   loop over heads slicing columns, call your attention function with the same
   mask for every head, concatenate with `np.concatenate(..., axis=-1)`, then
   multiply by `Wo`. Common mistake: slicing rows instead of columns.
5. **`attention_backward(dout, q, k, v, mask=None)`** recomputes the forward
   pass, then follows the three backward steps above. It returns `(dq, dk, dv)`
   with the same shapes as `q`, `k`, and `v`. Common mistake: forgetting the
   `1 / sqrt(d_k)` factor on `dq` and `dk`.
6. **`attention_flops(T, d, heads)`** returns the integer
   `8*T*d**2 + 4*T**2*d`. The `heads` argument is accepted but not used.

## Cost and scaling

A flop is one multiply or one add. Multiplying a $p \times q$ matrix by a
$q \times r$ matrix takes $2pqr$ flops. Count the pieces of one layer:

| Piece | Flops | Grows with |
|---|---|---|
| Four projections: $Q$, $K$, $V$, and $W_O$ | $4 \times 2Td^2 = 8Td^2$ | length times width squared |
| Scores $QK^{\top}$ plus weighted sum with $V$ | $2 \times 2T^2 d = 4T^2 d$ | length squared times width |

Add them to get the total:

$$
\text{flops} = 8 T d^2 + 4 T^2 d
$$

Read aloud: eight times length times width squared, plus four times length
squared times width. The number of heads isn't in it. For $T = 16$ and $d = 4$,
that's $8 \cdot 16 \cdot 16 + 4 \cdot 256 \cdot 4 = 2048 + 4096 = 6144$.

```python
flops = 8 * T * d**2 + 4 * T**2 * d
```

Once $T$ is larger than $2d$, the second term wins. Doubling the sequence
length then roughly quadruples the cost. Memory grows the same way: each head
keeps a $T \times T$ weight table for the backward pass.

## Where this breaks

- **Long sequences get expensive fast.** The $T^2$ term dominates at thousands
  of tokens, in both time and memory. People use tricks that compute attention
  in tiles without storing the full table, or approximate attention with
  cheaper patterns.
- **No sense of order.** As you saw, shuffled input gives shuffled output.
  Every real model adds position information to the input.
- **One layer, one lookup.** A single layer can copy or average information
  between any two positions in one step. It can't chain two lookups, like
  "find the word before the word that matches me". Stacking layers gives the
  model that chain.
- **A fully masked row.** If a row of the mask is all $-\infty$, the softmax
  divides 0 by 0 and returns NaN. A causal mask always leaves the diagonal
  open, so this only happens with other masks.

## The exercise

You build self-attention from its parts: `causal_mask`, `softmax`,
`scaled_dot_product_attention`, `multi_head_attention`, `attention_backward`,
and `attention_flops`. Everything works on a single sequence with no batch
dimension.

The checks confirm that the mask has the right pattern, that `softmax` stays
finite at logits of 900 and sums to one along the axis you ask for, and that
attention weights form valid rows. They check that changing later positions
never changes earlier outputs under the causal mask, that one-head multi-head
attention matches single-head attention times `Wo`, and that shuffling the
input shuffles the output the same way. Finally, they compare
`attention_backward` against finite differences, a numerical estimate of the
gradient, and check that `attention_flops` matches the formula for every head
count.
