---
title: Parameter-efficient fine-tuning
slug: 31-parameter-efficient-tuning
part: "Part 6 — Large language models"
summary: Why full fine-tuning needs so much memory, and how a small low-rank update gets most of the benefit.
minutes: 50
gpu: false
objectives:
  - Count the memory full fine-tuning needs per parameter, and where each byte goes.
  - Build a LoRA layer and count how many parameters it trains.
  - Merge a LoRA update into the base weight and check the outputs match exactly.
  - Compute the best rank-r approximation of a matrix with the SVD.
  - Compute LoRA's gradients for A and B, and explain which direction they move the weight.
prereqs:
  - 27-training-a-transformer
exercise: 31-parameter-efficient-tuning
---

# Parameter-efficient fine-tuning

Fine-tuning means taking a model that's already trained and training it a
little more on your own data. The simple way updates every weight. For a
seven-billion-parameter model, that needs over a hundred gigabytes of memory
for the optimiser alone. Most people don't have that hardware.

Parameter-efficient fine-tuning asks a practical question. Does every weight
need to move? Experiments say no. The useful change is usually simple, and a
small number of new parameters can describe it.

This lesson builds LoRA, short for low-rank adaptation, the most widely used
method of this kind. You count what it saves, show that it starts out
identical to the original model, merge it away at the end so it costs nothing
to run, and see how its gradient relates to ordinary fine-tuning.

## What you already know

- A linear layer computes `y = x @ W.T`, where `W` has shape
  `(d_out, d_in)`.
- The Adam optimiser keeps two running averages for every parameter it
  trains.
- The *rank* of a matrix is the number of independent directions it can
  produce. An outer product of two vectors has rank 1.
- The singular value decomposition, or SVD, writes a matrix as
  `U @ diag(s) @ Vt`, with the singular values `s` sorted from largest to
  smallest.
- The chain rule lets you pass a gradient back through a matrix product.

## Count full fine-tuning's memory

Training one parameter with Adam in mixed precision stores five numbers:

| What | Precision | Bytes |
|---|---|---|
| Working copy of the weight | 16-bit | 2 |
| Gradient | 16-bit | 2 |
| Master copy of the weight | 32-bit | 4 |
| Adam's first running average | 32-bit | 4 |
| Adam's second running average | 32-bit | 4 |
| **Total** | | **16** |

*Mixed precision* means doing the fast matrix math in 16-bit numbers while
keeping a careful 32-bit copy for updates. Skipping it doesn't save memory:
32-bit weight, gradient, and two averages also add up to 16 bytes. So a
7-billion-parameter model needs $7 \times 10^9 \times 16 = 112$ GB.

The only way to shrink this is to train fewer parameters. The code is one
line: `16 * params`, whether you're charging a full model or an adapter.

## Build a low-rank adapter

Picture the pretrained weight as a large printed map. Fine-tuning redraws
the whole map. LoRA leaves the map alone and lays a thin transparent sheet on
top with a few corrections. The sheet is cheap because it's built from a small
number of directions.

Freeze the pretrained weight $W$, of shape $d_{\text{out}} \times d_{\text{in}}$.
Add two small trainable matrices: $B$ of shape $d_{\text{out}} \times r$ and
$A$ of shape $r \times d_{\text{in}}$. The rank $r$ is small, often 4 to 16.
The layer now uses

$$
W_{\text{eff}} = W + \frac{\alpha}{r} BA
$$

Read it piece by piece:

- $W$ is the frozen original weight. It never changes.
- $BA$ is the correction. It has the same shape as $W$ but rank at most $r$.
- $\alpha / r$ is a fixed scale. $\alpha$ is a number you choose. Dividing by
  $r$ keeps the size of the update steady when you change $r$, so the same
  learning rate keeps working.

You never need to build $W_{\text{eff}}$ during training. Apply each piece to
the input $x$, of shape `(n, d_in)`, in turn:

$$
\hat{y} = xW^{\top} + \frac{\alpha}{r} (xA^{\top}) B^{\top}
$$

In words: the usual layer output, plus the input squeezed down to $r$
numbers by $A$, then expanded back up to $d_{\text{out}}$ by $B$, then
scaled. In numpy:

```python
base = x @ W.T                       # (n, d_out)
update = (x @ A.T) @ B.T             # (n, r) then (n, d_out)
```

**Start at zero.** Initialise $B$ to all zeros and $A$ to small random
numbers. Then $BA = 0$, so at step zero the adapted layer gives exactly the
same output as the original. Training starts from the pretrained model, not
from a damaged copy. (If both were zero, neither would ever get a gradient.)

**Count the savings.** Full fine-tuning of $W$ trains $d_{\text{in}}
d_{\text{out}}$ numbers. LoRA trains the entries of $A$ and $B$:

$$
\text{LoRA parameters} = r (d_{\text{in}} + d_{\text{out}})
$$

For one $4096 \times 4096$ attention matrix with $r = 8$, full training is
$4096^2 = 16{,}777{,}216$ parameters. LoRA is $8 \times (4096 + 4096) =
65{,}536$. That's 256 times fewer. At 16 bytes each, optimiser memory for that
matrix falls from 256 MiB to 1 MiB. The frozen $W$ still sits in memory, but
it needs no gradient and no optimiser state.

### Worked example

Take $d_{\text{in}} = d_{\text{out}} = 2$, $r = 1$, and $\alpha = 2$, so the
scale is $\alpha / r = 2$. Let

$$
W = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
A = \begin{bmatrix} 1 & 2 \end{bmatrix}, \quad
B = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad
x = \begin{bmatrix} 1 & 1 \end{bmatrix}
$$

- Base output: $xW^{\top} = [1, 1]$
- Squeeze: $xA^{\top} = 1 \cdot 1 + 1 \cdot 2 = 3$
- Expand: $3 \cdot B^{\top} = [3, 0]$
- Scale: $2 \times [3, 0] = [6, 0]$
- Output: $[1, 1] + [6, 0] = [7, 1]$

LoRA here trains $1 \times (2 + 2) = 4$ numbers, the same as the full $2
\times 2$ matrix. The saving only appears when $r$ is much smaller than the
layer's width.

## Merge at inference

After training, $A$ and $B$ are ordinary fixed matrices. So you can compute
$W_{\text{eff}}$ once and store it in place of $W$:

```python
W_merged = W + (alpha / r) * (B @ A)   # (d_out, d_in), same as W
```

Every forward pass then uses one matrix, at the same speed as the original
model. In the worked example, $BA = \begin{bmatrix} 1 & 2 \\ 0 & 0
\end{bmatrix}$, so $W_{\text{eff}} = \begin{bmatrix} 3 & 4 \\ 0 & 1
\end{bmatrix}$. Then $x W_{\text{eff}}^{\top} = [3 + 4, 0 + 1] = [7, 1]$, the
same answer as before. Matrix multiplication distributes over addition, so
the two routes always agree, up to rounding.

The catch comes when one server hosts many adapters on one base model.
Merging bakes in a single adapter. To switch per request, you either re-merge
or keep $W$ unmerged and pay for the small extra term on every token.

## Find the best low-rank approximation

Why should a rank-$r$ correction be enough? Any matrix can be written with
the SVD as a sum of rank-1 pieces, each weighted by a singular value. The
singular values are sorted, so the first pieces matter most. The best rank-$r$
approximation keeps the first $r$ pieces and drops the rest. This is the
Eckart-Young theorem. The leftover error, measured as the square root of the
sum of squared entries (the *Frobenius norm*), is

$$
\text{error} = \sqrt{s_{r+1}^2 + s_{r+2}^2 + \cdots}
$$

Read it as: the error is built from exactly the singular values you threw
away. If those are small, a low-rank update loses almost nothing. In numpy:

```python
U, s, Vt = np.linalg.svd(M, full_matrices=False)
# keep the first r columns of U, the first r values of s, the first r rows of Vt
```

Worked example: $M = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$ has
singular values 3 and 1. The best rank-1 approximation is $\begin{bmatrix} 3
& 0 \\ 0 & 0 \end{bmatrix}$. The error is $\sqrt{1^2} = 1$.

## See where the gradient goes

Let $G$ be the gradient that ordinary full fine-tuning would give for
$W_{\text{eff}}$. It has the same shape as $W$. The chain rule through $BA$
gives the gradients for the two small matrices:

$$
\frac{\partial L}{\partial A} = \frac{\alpha}{r} B^{\top} G, \qquad
\frac{\partial L}{\partial B} = \frac{\alpha}{r} G A^{\top}
$$

Read it as: to get $A$'s gradient, pass $G$ back through $B$. To get $B$'s,
pass it back through $A$. Both carry the same scale $\alpha / r$. Shapes
check out: $B^{\top} G$ is $(r, d_{\text{out}}) \times (d_{\text{out}},
d_{\text{in}}) = (r, d_{\text{in}})$, the shape of $A$.

```python
dA = (alpha / r) * (B.T @ G)   # (r, d_in)
dB = (alpha / r) * (G @ A.T)   # (d_out, r)
```

Take one small step of size $\eta$ on both. To first order, $W_{\text{eff}}$
changes by

$$
\Delta W_{\text{eff}} \approx -\eta \left(\frac{\alpha}{r}\right)^2 \left(G A^{\top} A + B B^{\top} G\right)
$$

In words: LoRA still moves downhill along the full gradient $G$, but filtered.
$A^{\top} A$ keeps only the input directions $A$ already covers. $B B^{\top}$
keeps only the output directions $B$ already covers. Directions outside what
$A$ and $B$ span get no update. A larger $r$ widens the filter.

## Other adapter methods

A few relatives are worth knowing by name:

- **QLoRA** stores the frozen base model in 4-bit numbers and trains LoRA on
  top. That cuts the memory for $W$ itself by about four times.
- **Adapters** insert small extra layers after each sublayer. They can't be
  merged away, so every inference pays for them.
- **Prefix tuning** adds trainable vectors to the keys and values in every
  attention layer. It costs extra context length on every call.

Quality rises fast as $r$ grows from 1, then levels off. Most tasks get close
to full fine-tuning at a rank somewhere from single digits to the low tens.

## Build it step by step

The checks test the functions in this order. `x` is `(n, d_in)`, `W` is
`(d_out, d_in)`, `A` is `(r, d_in)`, and `B` is `(d_out, r)`.

1. **`lora_forward(x, W, A, B, alpha, r)`.** Returns `(n, d_out)`. Compute the
   base output, the squeeze-then-expand update, and add them with the
   `alpha / r` scale. Common mistake: writing `x @ W` instead of `x @ W.T`.
   The first check sets `B` to zero and demands an exact match with the base
   layer.
2. **`lora_parameter_count(d_in, d_out, r)`.** Returns an integer. It's the
   formula above. Common mistake: returning `d_in * d_out`, the full count.
3. **`optimiser_memory_bytes(params, mode)`.** Returns `16 * params` for
   `mode` equal to `"full"` or `"lora"`. The rate is the same in both. Raise a
   `ValueError` for any other mode. Common mistake: charging LoRA a different
   rate per parameter.
4. **`merge_lora(W, A, B, alpha, r)`.** Returns a `(d_out, d_in)` matrix. Use
   `B @ A`, not `A @ B`, which has shape `(r, r)`. The checks compare `x @
   merged.T` with `lora_forward` and run finite differences through your
   merge to confirm the gradient formulas.
5. **`best_rank_approximation(delta_w, r)`.** Returns a matrix the same shape
   as `delta_w`. Call `np.linalg.svd` with `full_matrices=False`, keep the
   first `r` pieces, and multiply them back together. `U[:, :r] * s[:r]`
   scales each column without building a diagonal matrix. Common mistake:
   keeping the last `r` singular values instead of the first.

## Cost and scaling

The saving holds at every scope, because every matrix in the table has the
same shape.

| Quantity | Full fine-tuning | LoRA ($r = 8$) |
|---|---|---|
| Trainable parameters, one $4096 \times 4096$ matrix | 16,777,216 | 65,536 |
| Optimiser memory, one matrix | 256 MiB | 1 MiB |
| Trainable parameters, all attention matrices, 32 layers | 2,147,483,648 | 8,388,608 |
| Optimiser memory, same scope | 32 GiB | 128 MiB |

The frozen base weights still need memory, and the forward and backward
passes still run through them. LoRA doesn't change that. It cuts only the
part that scales with trainable parameters, but that's the difference between
a cluster and one accelerator.

The extra compute during training is small: the update path costs about $r
(d_{\text{in}} + d_{\text{out}})$ multiplies per example, against
$d_{\text{in}} d_{\text{out}}$ for the base layer. After merging, it's zero.

## Where this breaks

**Many adapters at once.** A server running hundreds of customer adapters
on one base model can't merge them all. It pays the small extra term for
every request instead.

**Hard tasks at tiny ranks.** If a task needs a genuinely new skill rather
than a recombination of what the model already knows, it may need a larger
rank, and the saving shrinks.

**Errors that stack.** QLoRA's 4-bit base already loses some information.
An adapter trained on top fits that slightly different model, and the two
errors are hard to predict together.

**Forgetting.** Training few parameters limits how much the model can change,
not what changes. Fine-tuning can still overwrite skills your data never
tested. Mixing some general data into fine-tuning helps.

## The exercise

You build a LoRA layer and its accounting. Write `lora_forward(x, W, A, B,
alpha, r)` for the adapted forward pass, `lora_parameter_count(d_in, d_out,
r)` for $r(d_{\text{in}} + d_{\text{out}})$, and
`optimiser_memory_bytes(params, mode)` for the 16-bytes-per-parameter rule.
Write `merge_lora(W, A, B, alpha, r)` for $W + (\alpha / r) BA$, and
`best_rank_approximation(delta_w, r)` for the truncated SVD.

The checks verify that a zero `B` reproduces the base layer exactly, the
parameter count and memory ratio, and that the merged weight matches
`lora_forward` to machine precision. They check that your rank-$r$
approximation reaches the Eckart-Young error. Finally, they run finite
differences through `merge_lora` to confirm the gradients for $A$ and $B$,
and confirm that one small step moves $W_{\text{eff}}$ in the filtered
direction from the last formula.
