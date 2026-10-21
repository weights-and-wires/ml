---
title: Initialisation and normalisation
slug: 18-initialisation-and-norm
part: "Part 3 — Neural networks"
summary: "Choose Xavier and He scales, measure signal size through fifty layers, and build batch norm and layer norm."
minutes: 65
gpu: false
objectives:
  - Compute the Xavier and He weight scales, and explain which activation each suits.
  - Build a simulation that measures how signal size changes through fifty layers.
  - Build the forward and backward passes of batch normalisation and layer normalisation.
  - Explain why layer norm does not depend on the batch and batch norm does.
prereqs:
  - 15-multilayer-perceptrons
  - 16-backpropagation
exercise: 18-initialisation-and-norm
---

# Initialisation and normalisation

Before training, every weight needs a starting value. Draw them at random, and the size of those draws sets whether a signal survives. Too small, and each layer shrinks it. Too large, and the numbers overflow.

Fifty layers make a small shrink fatal. Pass on seven tenths of the signal at every layer and about one part in fifty million is left. The gradient, the list of partial derivatives of the loss, shrinks the same way on the way back. Early layers then get almost no update.

You choose the Xavier and He scales, measure signal size through a deep stack, and build two normalisation layers that put the signal back to a fixed spread.

## What you already know

You already have the following:

- A linear layer outputs a weighted sum of its inputs.
- ReLU replaces each negative number with zero and keeps the positive ones.
- Variance is the average squared distance from the mean. It measures spread.
- Backpropagation carries the gradient from the loss back through every layer.
- A finite difference checks a gradient by nudging an input and watching the output.

## A sum that grows with width

Take two inputs, each of variance 1, and two weights of standard deviation 1. One output is a sum of two products. Independent terms add their variances, so the output variance is about 2.

That factor is the whole story of scale. With $n_{\text{in}}$ inputs of some variance, and weights of mean 0 and standard deviation $s$, one output has variance

$$
\text{Var}(\text{output}) = n_{\text{in}} \, s^2 \cdot \text{Var}(\text{input})
$$

Count the inputs, multiply by the weight variance $s^2$, then by the input variance. For the two inputs above, $n_{\text{in}} s^2 = 2$, and the output variance is twice the input variance.

To keep the spread unchanged, you need $n_{\text{in}} s^2 = 1$. Any other factor is raised to the fiftieth power in a fifty-layer stack:

```python
h = rng.normal(size=(num_trials, n_in))
W = rng.normal(scale=s, size=(n_in, n_out))
print(np.mean((h @ W) ** 2))
```

The usual mistake is to draw a wide layer with standard deviation 1, so the signal grows with $n_{\text{in}}$ at every layer.

> [!KEY]
> Each linear layer multiplies signal variance by $n_{\text{in}} s^2$. Choose $s$ so that factor is 1.

## Split the difference between forward and back

A forward pass needs $n_{\text{in}} s^2 = 1$. A backward pass needs $n_{\text{out}} s^2 = 1$. For a square layer of width 100, both agree on $s^2 = 1/100$, so $s = 0.1$.

When the two widths differ, **Xavier initialisation** (also called Glorot) uses the average of the two counts:

$$
s_{\text{Xavier}} = \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}
$$

The square root is the standard deviation you draw from. For width 100 in and out, $s = \sqrt{2/200} = 0.1$, and $n_{\text{in}} s^2 = 1$. It suits activations that are nearly linear near zero, such as tanh:

```python
s = np.sqrt(2.0 / (n_in + n_out))
```

The usual mistake is to return $2 / (n_{\text{in}} + n_{\text{out}})$, the variance, instead of its square root.

## Put back what ReLU throws away

ReLU keeps only the positive half, so it cuts the mean square roughly in half. Width 100 with Xavier still has $n_{\text{in}} s^2 = 1$, and each ReLU then multiplies the size by about $1/2$. After 50 layers that is $0.5^{50} \approx 9 \times 10^{-16}$. The signal is gone.

Double the weight variance and the half from ReLU cancels it. You want $n_{\text{in}} s^2 = 2$, so $s = \sqrt{2/100} \approx 0.1414$. Each ReLU layer then multiplies by $2/2 = 1$, and after 50 layers the size is still about 1.

That choice is **He initialisation**:

$$
s_{\text{He}} = \sqrt{\frac{2}{n_{\text{in}}}}
$$

The standard deviation is the square root of 2 over the fan-in. Only $n_{\text{in}}$ appears, because the backward fix is left to later tools:

```python
s = np.sqrt(2.0 / n_in)
```

The usual mistake is to use He in front of tanh. The variance then doubles at every layer.

> [!KEY]
> ReLU halves the mean square. He doubles the weight variance, so the signal keeps its size at any depth.

After ReLU the values are no longer centred on zero. The next layer cares about the mean square, $\mathrm{mean}(h^2)$, not the spread around the mean. For the input, and for a linear layer, the mean is zero, so the two match. `forward_variance` records $\mathrm{mean}(h^2)$ at the input and after every layer, which is `depth + 1` numbers:

```python
h = h @ W
if activation == "relu":
    h = np.maximum(h, 0.0)
variances.append(float(np.mean(h**2)))
```

The usual mistake is to apply ReLU to the raw input, before the first multiply.

## Rescale a group, then let the layer choose

Take one row of features, $[1, 2, 3]$, with scale 1 and shift 0. The mean is $(1+2+3)/3 = 2$. The gaps from the mean are $-1, 0, 1$. The variance is $(1+0+1)/3 = 0.6667$. Divide by $\sqrt{0.6667 + 10^{-5}} \approx 0.8165$ and you get about $[-1.2247,\, 0,\, 1.2247]$.

That rescaling is **normalisation**. For a group of $M$ numbers, subtract the mean, divide by the standard deviation, then apply a learned scale $\gamma$ and shift $\beta$:

$$
\mu = \frac{1}{M}\sum_{i=1}^{M} x_i, \qquad
\sigma^2 = \frac{1}{M}\sum_{i=1}^{M} (x_i - \mu)^2, \qquad
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad
y_i = \gamma \, \hat{x}_i + \beta
$$

Mu, $\mu$, is the group mean. Sigma squared, $\sigma^2$, divides by $M$, not $M-1$. Epsilon, $\epsilon$, such as $10^{-5}$, keeps the square root off zero. Gamma and beta are one number per feature. The same row gives about $[-1.2247,\, 0,\, 1.2247]$. Add 37.5 to every entry and the output does not change, because the mean moves with the row:

```python
mean = x.mean(axis=axis, keepdims=True)
var = x.var(axis=axis, keepdims=True)
xhat = (x - mean) / np.sqrt(var + eps)
out = gamma * xhat + beta
```

The usual mistake is to divide the variance by $M - 1$. `np.var` already divides by $M$.

The matrix `x` has shape `(N, D)`: $N$ examples, $D$ features. **Batch normalisation** groups each column, so it reduces over `axis=0` and $M = N$. **Layer normalisation** groups each row, so it reduces over `axis=1` and $M = D$. In both, `gamma` and `beta` have shape `(D,)`.

> [!KEY]
> Batch norm and layer norm share the arithmetic. Batch norm reduces over `axis=0`. Layer norm reduces over `axis=1`.

`keepdims=True` leaves a length-1 axis so the mean lines up with `x`.

> [!TIP]
> Print `mean.shape`. It is `(1, D)` for batch norm and `(N, 1)` for layer norm.

Shift one example by a constant and layer norm is unchanged, as in the row above. Batch norm is not. That example moves the column means shared by the whole batch.

A batch of one has column variance 0, so batch norm has nothing to divide by except $\epsilon$. At test time you often score one example, so batch norm stores running averages of $\mu$ and $\sigma^2$ from training. Layer norm uses only that example's own features, in training and at test time. Recurrent nets and transformers use it for that reason.

> [!KEY]
> Layer norm looks only at one example. Its output does not change with the batch size.

## Send the gradient back through the group

The loss sends back `dout`, the same shape as the output. From $y = \gamma \hat{x} + \beta$, the shift's gradient is the sum of `dout`, and the scale's gradient is the sum of `dout * xhat`. Both sums run over examples, `axis=0`, for batch norm and for layer norm.

On the row above, take `dout = [1, 0, 0]`. Then `dbeta` is `[1, 0, 0]`, and `dgamma` is about `[-1.2247, 0, 0]`.

The input is harder. Each $x_i$ changes the output directly, through the mean, and through the variance. Let $d\hat{x} = \mathrm{dout} \cdot \gamma$, and let each sum run over the same group as the forward pass:

$$
dx = \frac{1}{M\sqrt{\sigma^2 + \epsilon}}\Big(M\, d\hat{x} - \textstyle\sum d\hat{x} - \hat{x} \sum \big(d\hat{x} \cdot \hat{x}\big)\Big)
$$

Start from $M$ times $d\hat{x}$. Subtract the group sum, which undoes the mean. Subtract $\hat{x}$ times the sum of $d\hat{x} \cdot \hat{x}$, which undoes the variance. Divide by $M$ times the standard deviation. Only `axis` and $M$ change between the two layers. $M$ is $N$ for batch norm and $D$ for layer norm:

```python
dxhat = dout * gamma
dx = (istd / M) * (
    M * dxhat
    - dxhat.sum(axis=axis, keepdims=True)
    - xhat * (dxhat * xhat).sum(axis=axis, keepdims=True)
)
```

The usual mistake is to sum `dgamma` and `dbeta` over the normalised axis. They are per feature, so they always sum over `axis=0`.

> [!WARNING]
> For layer norm, `dx` uses `axis=1`. `dgamma` and `dbeta` still sum over `axis=0`.

> [!RECAP]
> Hold these five points before you write code:
>
> - A layer multiplies variance by $n_{\text{in}} s^2$. Xavier uses $\sqrt{2/(n_{\text{in}}+n_{\text{out}})}$. He uses $\sqrt{2/n_{\text{in}}}$ before a ReLU.
> - Record $\mathrm{mean}(h^2)$ for the input and after every layer, so the result has length `depth + 1`.
> - Normalise by subtracting the mean and dividing by $\sqrt{\sigma^2+\epsilon}$, then apply $\gamma$ and $\beta$. Use `keepdims=True`.
> - Batch norm reduces over `axis=0`. Layer norm reduces over `axis=1`.
> - `dx` uses that group axis. `dgamma` and `dbeta` always sum over `axis=0`.

## Build it step by step

The checks call the functions in this order:

1. **`xavier_scale(n_in, n_out)`.** Return a float, `np.sqrt(2 / (n_in + n_out))`. The usual mistake is to skip the square root.
2. **`he_scale(n_in)`.** Return `np.sqrt(2 / n_in)`. The usual mistake is to include `n_out`.
3. **`forward_variance(depth, widths, scale, activation, num_trials, seed)`.** Use `np.random.default_rng(seed)`. Draw `h` with shape `(num_trials, widths[0])` and record `np.mean(h**2)`. For each of `depth` layers, draw `W` of shape `(widths[l], widths[l + 1])` with `rng.normal(scale=scale)`, multiply, and apply `np.maximum(h, 0)` only when `activation` is `"relu"`. Return length `depth + 1`. The usual mistake is a ReLU on the input.
4. **`batchnorm_forward(x, gamma, beta, eps)`.** `x` is `(N, D)`. Reduce over `axis=0`. Return `(out, cache)` and keep `xhat`, the inverse standard deviation, `gamma`, and the axis in the cache. The usual mistake is dropping `keepdims=True`.
5. **`layernorm_forward(x, gamma, beta, eps)`.** The same arithmetic over `axis=1`. The usual mistake is reducing over columns.
6. **`batchnorm_backward(dout, cache)`.** Return `(dx, dgamma, dbeta)` with shapes `(N, D)`, `(D,)`, and `(D,)`. Use the `dx` formula with `axis=0` and $M = N$. The usual mistake is a sign error on the variance term.
7. **`layernorm_backward(dout, cache)`.** The same formula with `axis=1` and $M = D$. The usual mistake is summing `dgamma` over `axis=1`.

## Cost and scaling

You pick a scale once. Normalisation then adds a small cost on every pass:

| Operation | Time | Extra memory |
|---|---|---|
| Xavier or He scale | once | none |
| Forward normalisation | a few passes over `(N, D)` | one cache about the size of `x` |
| Backward normalisation | a few more passes over `(N, D)` | none beyond that cache |
| `forward_variance` | one matrix multiply per layer | one `(num_trials, width)` array at a time |

Time grows in a straight line with the number of activations. It stays small next to the linear layers' matrix multiplies.

## Where this breaks

Each of these failures has a direct change:

- Xavier in front of ReLU halves the signal at every layer. Use He for ReLU.
- A very small batch makes batch-norm statistics noisy. Use layer norm, or group norm.
- Batch norm at test time must use its stored running averages. One prediction must not depend on who else shares the batch.
- A constant group has $\sigma^2 = 0$. Only $\epsilon$ keeps the division finite.
- Past a hundred layers or so, a good scale and a norm layer can still lose the gradient. A residual shortcut, in lesson 21, gives it a path around the block.

## The exercise

You implement `xavier_scale`, `he_scale`, `forward_variance`, `batchnorm_forward`, `batchnorm_backward`, `layernorm_forward`, and `layernorm_backward`. Each forward returns `(out, cache)`, and the matching backward reads that cache.

The checks verify both scale formulas, and that one linear layer multiplies variance by $n s^2$. They check mean 0 and variance 1 on the right axis, and both backward passes against finite differences for $x$, $\gamma$, and $\beta$. They check that a constant shift of one example changes batch norm and not layer norm. The last check runs fifty ReLU layers: He stays flat, and Xavier halves the signal at every layer.
