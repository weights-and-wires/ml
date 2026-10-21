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

Fine-tuning trains a model a little more on your data. Updating every weight of a seven-billion-parameter model needs about 112 GB for the optimiser alone.

The useful change is often simple, so a few extra parameters can describe it. The original weights stay frozen.

You build low-rank adaptation, LoRA. You count the bytes, check that step zero matches the original layer, and fold the update into one matrix. You also see which directions a gradient step can move.

## What you already know

This lesson uses five ideas from earlier work:

- A linear layer computes `y = x @ W.T`, and `W` has shape `(d_out, d_in)`.
- Adam keeps two running averages for every parameter it trains.
- Rank is the number of independent directions in a matrix. An outer product of two vectors has rank 1.
- The singular value decomposition writes a matrix as `U @ diag(s) @ Vt`, with `s` largest first.
- The chain rule carries a derivative back through a matrix product.

## Count sixteen bytes per trained parameter

A gradient is the change in the loss when you nudge one input. Adam in mixed precision stores five numbers per trained parameter. Mixed precision does the fast math in 16-bit numbers and keeps a 32-bit copy for the update.

The five numbers add up as follows:

| What | Precision | Bytes |
|---|---|---|
| Working copy | 16-bit | 2 |
| Gradient | 16-bit | 2 |
| Master copy | 32-bit | 4 |
| First running average | 32-bit | 4 |
| Second running average | 32-bit | 4 |
| Total | | 16 |

Skip mixed precision and a 32-bit weight, gradient, and two averages are still 16 bytes. So $7 \times 10^9 \times 16 = 112$ GB. The rate is `16 * params` for a full model or an adapter. The usual mistake charges the adapter a different rate.

> [!KEY]
> Adam costs 16 bytes per trained parameter. The bill shrinks when you train fewer parameters.

## Compute a 2 by 2 update by hand

Take input width 2, output width 2, rank $r = 1$, and scale numerator $\alpha = 2$. The Greek letter $\alpha$ (alpha, pronounced AL-fuh) is that numerator, so $\alpha / r = 2$. Let $W$ be the identity, $A$ the row `[1, 2]`, $B$ the column `[1, 0]`, and $x$ the row `[1, 1]`.

The base output is $x W^{\top} = [1, 1]$. The squeeze $x A^{\top}$ is $1 + 2 = 3$. Times $B^{\top}$ that is `[3, 0]`. Scale by 2 to get `[6, 0]`. Add the base: `[7, 1]`.

This correction is LoRA. $W$ stays frozen, with shape $d_{\text{out}} \times d_{\text{in}}$. $B$ is $d_{\text{out}} \times r$ and $A$ is $r \times d_{\text{in}}$. The rank $r$ is often 4 to 16. The effective weight is

$$
W_{\text{eff}} = W + \frac{\alpha}{r} BA
$$

$BA$ matches the shape of $W$ and has rank at most $r$. Dividing by $r$ keeps the update a similar size when $r$ changes. During training you apply the pieces to a batch `x` of shape `(n, d_in)`:

$$
\hat{y} = xW^{\top} + \frac{\alpha}{r} (xA^{\top}) B^{\top}
$$

The second term squeezes $x$ to $r$ numbers, expands with $B$, and scales. The same 2 by 2 numbers give `[7, 1]` again. Here $BA$ is the matrix with first row `[1, 2]` and second row `[0, 0]`, so $W_{\text{eff}}$ is `[[3, 4], [0, 1]]`, and $x W_{\text{eff}}^{\top} = [7, 1]$. The forward pass has this shape:

```python
base = x @ W.T              # (n, d_out)
low_rank = (x @ A.T) @ B.T  # (n, r) then (n, d_out)
```

> [!WARNING]
> `W` has shape `(d_out, d_in)`, so the layer is `x @ W.T`. On a square `W`, `x @ W` runs and returns different numbers.

Start $B$ at zero and $A$ at small random values. Then $BA = 0$, and step zero matches the base layer. If both start at zero, the gradient into each one stays zero.

Full training of $W$ updates $d_{\text{in}} d_{\text{out}}$ numbers. LoRA updates $A$ and $B$:

$$
\text{LoRA parameters} = r (d_{\text{in}} + d_{\text{out}})
$$

This toy trains $1 \times (2 + 2) = 4$ numbers, the same as the full matrix. The saving appears when $r$ is far below the width. One $4096 \times 4096$ matrix at $r = 8$ is $4096^2 = 16777216$ parameters in full, and $8 \times 8192 = 65536$ in LoRA, 256 times fewer. At 16 bytes each, optimiser memory falls from 256 MiB to 1 MiB. Frozen $W$ still sits in memory, with no gradient and no optimiser state. The usual mistake returns `d_in * d_out` from the LoRA counter.

## Fold the factors into one matrix

After training, $A$ and $B$ are fixed. Form the effective weight once:

```python
W_merged = W + (alpha / r) * (B @ A)  # (d_out, d_in)
```

> [!WARNING]
> Multiply `B @ A`, the shape of `W`. The product `A @ B` runs too, and its shape is `(r, r)`.

Later passes use one matrix, at the original speed. The merged 2 by 2 weight already matched the split pass. A server with many adapters on one base model cannot bake them all in. Re-merge per request, or keep the small extra product.

> [!KEY]
> After training, store $W + \frac{\alpha}{r} BA$. The merged layer matches the split outputs and runs at the original speed.

## Keep the largest singular values

The decomposition writes a matrix as rank-1 pieces weighted by singular values, largest first. The best rank-$r$ approximation, the Eckart-Young one, keeps the first $r$ pieces. The error is the Frobenius norm, the square root of the summed squares:

$$
\text{error} = \sqrt{s_{r+1}^2 + s_{r+2}^2 + \cdots}
$$

The error uses only the singular values you dropped. For $M = \operatorname{diag}(3, 1)$, the singular values are 3 and 1. The best rank-1 matrix keeps 3 and zeros the rest, and the error is $\sqrt{1^2} = 1$. The formula gives that 1 again. The factors have this shape:

```python
U, s, Vt = np.linalg.svd(M, full_matrices=False)
# first r columns of U, first r values of s, first r rows of Vt
```

> [!TIP]
> `U[:, :r] * s[:r]` scales columns by broadcasting. You can skip `np.diag(s)`.

The usual mistake keeps the smallest singular values.

> [!KEY]
> The best rank-$r$ approximation keeps the $r$ largest singular values. Small dropped values mean the update loses little.

## See which directions the step can move

Let $G$ be the gradient full fine-tuning would give $W_{\text{eff}}$. It has the shape of $W$. The chain rule through $BA$ gives

$$
\frac{\partial L}{\partial A} = \frac{\alpha}{r} B^{\top} G, \qquad
\frac{\partial L}{\partial B} = \frac{\alpha}{r} G A^{\top}
$$

$A$'s gradient passes $G$ through $B$, and $B$'s passes $G$ through $A$. Both carry $\alpha / r$. Shapes agree: $B^{\top} G$ is $(r, d_{\text{in}})$, the shape of $A$. The products have this shape:

```python
dA = (alpha / r) * (B.T @ G)  # (r, d_in)
dB = (alpha / r) * (G @ A.T)  # (d_out, r)
```

One step of size $\eta$ (eta, pronounced AY-tuh) changes the effective weight, to first order, by

$$
\Delta W_{\text{eff}} \approx -\eta \left(\frac{\alpha}{r}\right)^2 \left(G A^{\top} A + B B^{\top} G\right)
$$

The step follows $G$, through a filter. $A^{\top} A$ keeps input directions already in $A$. $B B^{\top}$ keeps output directions already in $B$. A larger $r$ widens the filter. The usual mistake drops $\alpha / r$ from one of the two gradients.

> [!KEY]
> LoRA follows the full gradient $G$ inside the directions $A$ and $B$ already cover.

QLoRA stores the frozen base in 4-bit numbers and trains LoRA on top, cutting memory for $W$ about four times. Adapters are extra layers that stay in the forward pass. Prefix tuning adds vectors to the keys and values and spends context length on every call. Quality rises as $r$ grows from 1, then levels off, often by the low tens.

> [!RECAP]
> Keep these five facts in front of you while you code:
>
> - Training costs 16 bytes per trained parameter, for a full model or an adapter.
> - The forward pass is `x @ W.T` plus `(alpha / r)` times `(x @ A.T) @ B.T`. A zero $B$ matches the base.
> - LoRA trains $r (d_{\text{in}} + d_{\text{out}})$ numbers.
> - Merging stores $W + \frac{\alpha}{r} BA$, the same shape as $W$.
> - The best rank-$r$ matrix keeps the first $r$ pieces of the SVD.

## Build it step by step

`x` is `(n, d_in)`, `W` is `(d_out, d_in)`, `A` is `(r, d_in)`, and `B` is `(d_out, r)`. The checks call your functions in this order:

1. `lora_forward(x, W, A, B, alpha, r)` returns `(n, d_out)`. Add the base to the scaled low-rank path. The usual mistake writes `x @ W`. A zero `B` must match the base layer:

```python
def lora_forward(x, W, A, B, alpha, r):
    # x @ W.T + (alpha / r) * (x @ A.T) @ B.T
    return y
```

2. `lora_parameter_count(d_in, d_out, r)` returns $r (d_{\text{in}} + d_{\text{out}})$. The usual mistake returns `d_in * d_out`:

```python
def lora_parameter_count(d_in, d_out, r):
    # r * (d_in + d_out)
    return count
```

3. `optimiser_memory_bytes(params, mode)` returns `16 * params` for `"full"` and for `"lora"`. Raise `ValueError` for any other mode. The usual mistake changes the rate:

```python
def optimiser_memory_bytes(params, mode):
    # 16 * params for both modes
    return nbytes
```

4. `merge_lora(W, A, B, alpha, r)` returns `(d_out, d_in)` from `B @ A`. Checks compare `x @ merged.T` with `lora_forward`, and finite differences confirm the gradients:

```python
def merge_lora(W, A, B, alpha, r):
    # W + (alpha / r) * (B @ A)
    return merged
```

5. `best_rank_approximation(delta_w, r)` returns the shape of `delta_w`. Call `np.linalg.svd` with `full_matrices=False` and keep the first $r$ pieces. The usual mistake keeps the smallest values:

```python
def best_rank_approximation(delta_w, r):
    # (U[:, :r] * s[:r]) @ Vt[:r]
    return approx
```

## Cost and scaling

Every matrix in this comparison is $4096 \times 4096$. The trainable bill compares as follows:

| Quantity | Full fine-tuning | LoRA, rank 8 |
|---|---|---|
| Parameters, one matrix | 16777216 | 65536 |
| Optimiser memory, one matrix | 256 MiB | 1 MiB |
| Parameters, attention, 32 layers | 2147483648 | 8388608 |
| Optimiser memory, same scope | 32 GiB | 128 MiB |

Frozen weights still occupy memory, and both passes still run through them. LoRA cuts the part that scales with the trainable count. The update path costs about $r (d_{\text{in}} + d_{\text{out}})$ multiplies per example, against $d_{\text{in}} d_{\text{out}}$ for the base. After the merge, that extra path is gone.

## Where this breaks

Hundreds of adapters on one base model cannot all be merged. Each request pays the small extra product. A task outside what the base model already represents may need a larger rank, and the saving shrinks. A 4-bit base already drops information, and an adapter fits that altered model. Few trainable parameters limit how far the weights move. They do not pick which skill moves, so fine-tuning can overwrite a skill the data never tests. Mixing in some general data helps.

## The exercise

`lora_forward(x, W, A, B, alpha, r)` is the adapted forward pass. `lora_parameter_count(d_in, d_out, r)` returns $r(d_{\text{in}} + d_{\text{out}})$. `optimiser_memory_bytes(params, mode)` applies the 16-byte rule. `merge_lora(W, A, B, alpha, r)` returns $W + (\alpha / r) BA$. `best_rank_approximation(delta_w, r)` is the truncated SVD.

A zero `B` matches the base layer. The parameter count and the memory ratio match. The merged weight matches `lora_forward`. The rank-$r$ error matches Eckart-Young. Finite differences through `merge_lora` confirm the gradients, and one small step moves $W_{\text{eff}}$ along the filtered direction.
