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

A translation model has two parts. An **encoder** reads the source sentence
with a recurrent network. A **decoder** writes the output sentence with a
second recurrent network. In the simplest design, the encoder hands over only
its final hidden state: one vector meant to summarise the whole input.

That's like reading a long paragraph once and then translating it from memory,
without looking back. It works for short sentences. For long ones, a single
fixed-size vector can't hold everything, and translation quality drops sharply
as sentences get longer. This happens even when the gradients are fine.

**Attention** lets the decoder look back. At every output step, the decoder
looks over all of the encoder's hidden states and picks the ones that matter
right now. In this lesson you build two ways to score that choice, the
backward pass for one of them, and a small hand-built model that copies its
input perfectly.

## What you already know

- An encoder RNN turns inputs $x_1, \dots, x_{T_x}$ into hidden states
  $h_1, \dots, h_{T_x}$, one per source position.
- The **softmax** turns a list of scores into positive weights that add up to
  one: $\operatorname{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}$.
- A **weighted average** of vectors, with weights that are positive and sum
  to one, always lies "between" those vectors.
- The derivative of $\tanh(u)$ is $1 - \tanh(u)^2$.
- A dot product `a @ b` is large when two vectors point the same way.

## Attention as a soft lookup

A Python dictionary lookup takes a **query**, finds the **key** that matches
exactly, and returns its **value**. Attention does a soft version:

1. Score the query against every key. A higher score means a better match.
2. Turn the scores into weights with a softmax.
3. Return the weighted average of the values.

Instead of picking one winner, attention blends every value, mostly from the
best matches. Because every step is smooth, you can train it with ordinary
backpropagation.

> [!KEY]
> Attention scores a query against every key, turns the scores into weights with
> a softmax, and returns the weighted average of the values.

In a translation model, the query is the decoder's state $s_t$ at output step
$t$: "what do I need now?" The keys and values are the encoder states $h_i$:
"here's what's at position $i$". In this exercise the shapes are:

- `query`, shape `(d_s,)`, one decoder state.
- `keys`, shape `(T_x, d_h)`, one row per source position.
- `values`, shape `(T_x, d_v)`, one row per source position. Often the same
  as the keys.

The last two steps are the same for every kind of attention:

$$
\alpha_i = \frac{e^{e_i}}{\sum_{j} e^{e_j}}, \qquad
c = \sum_{i} \alpha_i \, \text{values}_i
$$

Read it aloud: the weight $\alpha_i$ for position $i$ is the softmax of the
scores $e_i$. The **context vector** $c$ is the weighted sum of the value
rows. In numpy, subtracting the maximum first so that `np.exp` can't
overflow:

```python
weights = np.exp(scores - scores.max())
weights = weights / weights.sum()
context = weights @ values
```

The starter file gives you this `softmax` already.

> [!KEY]
> The decoder builds a fresh context from all the encoder states at every output
> step. It no longer has to squeeze the whole input into one vector.

## Multiplicative attention

The simplest score is the dot product of the query with each key:

$$
e_i = s^{\top} h_i \qquad \text{or, with a weight matrix,} \qquad e_i = s^{\top} W h_i
$$

Read it aloud: the dot form multiplies the query by each key and sums, which
needs the query and key to have the same length. The **general** form puts a
learned matrix $W$, shape `(d_s, d_h)`, in between, so the lengths can differ.
In numpy, scoring all keys at once:

```python
scores = keys @ query              # dot form
scores = keys @ (W.T @ query)      # general form
```

### Work a dot-product example by hand

Take two source positions with two-number keys, and values that make the
result easy to read:

$$
s = [1, 0], \qquad
\text{keys} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \qquad
\text{values} = \begin{bmatrix} 10 & 0 \\ 0 & 10 \end{bmatrix}
$$

- Scores: $e_1 = 1 \times 1 + 0 \times 0 = 1$ and $e_2 = 0$.
- Weights: $e^1 = 2.718$ and $e^0 = 1$, total $3.718$, so
  $\alpha = [0.7311, 0.2689]$.
- Context: $0.7311 \times [10, 0] + 0.2689 \times [0, 10] = [7.311, 2.689]$.

The query matches the first key, so the context leans toward the first value.

## Additive attention

Additive attention scores each key with a tiny neural network. It projects the
query and the key into a shared space of size $d$, adds them, squashes with
$\tanh$, and takes a dot product with a learned vector $v$:

$$
e_i = v^{\top} \tanh\left(W_1 s + W_2 h_i\right)
$$

Read it aloud: multiply the query by $W_1$, shape `(d, d_s)`. Multiply key $i$
by $W_2$, shape `(d, d_h)`. Add the two, apply $\tanh$ entry by entry, and dot
the result with $v$, shape `(d,)`. That gives one number per key. In numpy,
for all keys at once:

```python
u = W1 @ query + keys @ W2.T      # shape (T_x, d)
scores = np.tanh(u) @ v           # shape (T_x,)
```

Here `W1 @ query` has shape `(d,)`, and numpy adds it to every row of
`keys @ W2.T`. Then you apply the same softmax and weighted sum as before.

> [!TIP]
> Write the shape of each result in a comment, as the code above does. When a
> shape doesn't match the comment, you have usually swapped `W1` and `W2` or
> dropped a `.T`.

### Work an additive example by hand

Use the same query, keys, and values, with $W_1$ and $W_2$ both the identity
and $v = [1, 1]$.

- Position 1: $u_1 = [1, 0] + [1, 0] = [2, 0]$, $\tanh(u_1) = [0.9640, 0]$,
  score $0.9640$.
- Position 2: $u_2 = [1, 0] + [0, 1] = [1, 1]$, $\tanh(u_2) = [0.7616, 0.7616]$,
  score $1.5232$.
- Weights: softmax of $[0.9640, 1.5232]$ is $[0.3637, 0.6363]$.

Surprise: the second position wins, even though the query matches the first
key. Two moderate entries after $\tanh$ add up to more than one large entry.
Flip $v$ to $[-1, -1]$ and the scores become $[-0.9640, -1.5232]$, giving
weights $[0.6363, 0.3637]$. Now the matching key wins. You use exactly this
trick in the copy task at the end.

> [!KEY]
> In additive attention, the learned vector $v$ decides what counts as a good
> match. A key that equals the query doesn't win on its own.

## The backward pass of additive attention

The backward pass takes `dcontext`, the gradient on the context vector, shape
`(d_v,)`, and returns a gradient for every input. Walk the forward pass in
reverse:

$$
d\text{values} = \alpha \, \text{dcontext}^{\top}, \qquad
d\alpha = \text{values} \cdot \text{dcontext}
$$

$$
de_i = \alpha_i \left( d\alpha_i - \sum_j \alpha_j \, d\alpha_j \right)
$$

Read it aloud. Each value row gets the context gradient scaled by its weight;
that's an **outer product**, `np.outer`. Each weight's gradient is its value
row dotted with the context gradient. The softmax backward step subtracts the
weighted average of those gradients, then scales by the weight.

Then back through the scoring network. Write $t = \tanh(u)$:

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

Read it aloud: $v$'s gradient sums each $\tanh$ row times its score gradient.
The gradient on $u$ passes back through $\tanh$. The query was added to every
row of $u$, so its gradient uses the sum of all rows of $du$. The keys each
touched only their own row. In numpy:

```python
dscores = weights * (dweights - np.sum(weights * dweights))
du = np.outer(dscores, v) * (1.0 - t ** 2)
dW1 = np.outer(du.sum(axis=0), query)
```

> [!WARNING]
> Use `du.sum(axis=0)`, shape `(d,)`, for both `dW1` and `dquery`. If you pass
> the full `(T_x, d)` array to `np.outer`, it still runs, because `np.outer`
> flattens its inputs, but `dW1` gets the wrong shape.

## The alignment matrix

Run attention once per decoder step and stack the weight rows. You get a
$T_y \times T_x$ **alignment matrix**: row $t$ says where the decoder looked
while writing output $t$. Nobody tells the model which words line up. On
translation, the trained matrix still often traces the word-by-word
correspondence, so people plot it to see what the model uses.

For a copy task, where output $t$ should equal input $t$, the ideal alignment
matrix is the identity: step $t$ looks only at position $t$.

> [!RECAP]
> - Every form of attention ends the same way: softmax the scores, then take
>   `weights @ values`.
> - Additive scores are `np.tanh(W1 @ query + keys @ W2.T) @ v`. `W1` goes with
>   the query.
> - Multiplicative scores are `keys @ query`, or `keys @ (W.T @ query)` when you
>   pass a `W`.
> - The backward pass runs in reverse: values and weights, the softmax step, the
>   $\tanh$ step, then the two projections.
> - In the copy task, a negative $v$ makes the matching position win, so the
>   alignment matrix is close to the identity.

## Build it step by step

The checks test the functions in this order.

1. **`additive_attention(query, keys, values, W1, W2, v)`.** Returns
   `(context, weights)` with shapes `(d_v,)` and `(T_x,)`. Compute `u`,
   `scores`, `softmax(scores)`, and `weights @ values`. Common mistake:
   swapping `W1` and `W2`; here `W1` goes with the query.
2. **`multiplicative_attention(query, keys, values, W=None)`.** Same return
   shapes. Use `keys @ query` when `W is None`, otherwise
   `keys @ (W.T @ query)`. Common mistake: testing `if W:` instead of
   `if W is None:`, which fails on an array.
3. **Convex combination.** No new function. The checks confirm your context
   equals `weights @ values`.
4. **`attention_backward(dcontext, query, keys, values, W1, W2, v)`.** Rerun
   the forward pass inside, then return `(dquery, dkeys, dvalues, dW1, dW2, dv)`
   in that order, each shaped like its input. Use `np.outer`, `.T @`, and
   `.sum(axis=0)`. Common mistake: forgetting the $1 - t^2$ factor.
5. **Additive emulates dot.** No new function. With tiny identity matrices for
   `W1` and `W2`, and `v` set to the query, $\tanh$ acts like a straight line,
   and additive attention gives the same weights as a scaled dot product. This
   passes when steps 1 and 2 are right.
6. **`decode_with_attention(source_tokens, vocab_size)`.** Returns
   `(decoded, alignment)`: a list of $T_x$ ints and a `(T_x, T_x)` array. Use
   `10 * np.eye(T_x)` as both the queries and the keys, one-hot rows of
   `np.eye(vocab_size)` as the values, identity matrices for `W1` and `W2`, and
   `v = -10 * np.ones(T_x)`. For each step `t`, call your
   `additive_attention` with query row `t`, store the weights, and append
   `int(np.argmax(context))`. The matching position gives one saturated
   $\tanh$ entry, score about $-10$; every other position gives two, score
   about $-20$, so its weight is about $e^{-10} \approx 4.5 \times 10^{-5}$.
   Common mistake: a positive `v`, which makes the wrong positions win.

## Cost and scaling

The table covers one decoder step against $T_x$ encoder states, with
attention size $d$.

| Kind | Work per decoder step | Parameters |
|---|---|---|
| Additive | about $T_x \cdot d \, (d_s + d_h)$ | $d (d_s + d_h) + d$ |
| Multiplicative, general | about $T_x \cdot d_s d_h$ | $d_s d_h$ |
| Multiplicative, dot | about $T_x \cdot d_h$ | none |

A full output of $T_y$ steps multiplies the work by $T_y$. Total attention
work grows with $T_x \times T_y$. Double both lengths and the work goes up four
times. Dot-product attention is the fastest in practice, because it's one big
matrix product with no $\tanh$.

## Where this breaks

**Repeating itself.** Nothing stops the decoder from looking at the same
source word at many steps. The symptom is a repeated phrase. **Coverage**
methods keep a running total of past weights per position and feed it into the
score.

**The encoder is still sequential.** Attention fixes what the decoder can
reach. The encoder still builds $h_1, \dots, h_{T_x}$ one step at a time. The
next lesson builds the encoder out of attention instead.

**Attention can't recover what the encoder dropped.** It only mixes what the
encoder states hold. If the encoder lost a detail, looking harder won't bring
it back.

**Cost grows with both lengths.** Long inputs and long outputs together make
the $T_x \times T_y$ cost add up.

## The exercise

You build `additive_attention` and `multiplicative_attention`, the backward
pass `attention_backward` for the additive form, and `decode_with_attention`,
a copy-task decoder whose weights you set by hand, with no training.

The checks confirm both forms return weights that are positive and sum to
one, and a context that's a weighted average of the values. They compare
`attention_backward` with finite differences for all six inputs. They confirm
additive attention matches a scaled dot product when set up to copy one. Last,
they confirm `decode_with_attention` reproduces the source exactly and that
its alignment matrix is within 0.01 of the identity.
