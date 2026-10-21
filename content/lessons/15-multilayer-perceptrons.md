---
title: Multilayer perceptrons
slug: 15-multilayer-perceptrons
part: "Part 3 — Neural networks"
summary: "Stack bent layers, run a two-input forward pass by hand, and count pieces, parameters, and flops."
minutes: 55
gpu: false
objectives:
  - Compute ReLU, GELU, and their derivatives, and check the derivatives with finite differences.
  - Build a forward pass for any list of affine-then-activation layers.
  - Explain why stacked layers with no nonlinearity collapse into one linear layer.
  - Build a ReLU network whose number of linear pieces doubles with every layer.
  - Count the parameters and flops of a network by hand.
prereqs:
  - 05-logistic-regression
exercise: 15-multilayer-perceptrons
---

# Multilayer perceptrons

Logistic regression draws one straight boundary. It forms one weighted sum and squashes it. Many problems need a boundary that bends.

A multilayer perceptron (MLP) stacks layers. Each layer forms a weighted sum, then bends every entry with an activation. Enough bends can follow a curve that one straight cut cannot.

You can run a forward pass by hand, see why the bend matters, and count pieces, parameters, and flops.

## What you already know

Five earlier ideas are enough to start:

- A linear map is `W @ x + b`.
- Logistic regression is one linear map, then a sigmoid.
- A derivative is a slope. For $x^2$ at $x = 3$ the slope is 6, so a nudge of 0.01 changes the output by about 0.06.
- A finite difference estimates a slope. On $x^2$ at $h = 0.01$, $(9.0601 - 8.9401)/0.02 = 6$.
- The chain rule multiplies slopes. The slope of $f(g(x))$ is $f'(g(x))$ times $g'(x)$.

## Run one network on two inputs

Take $x = [2, -3]$. The first weight matrix has rows $[1, 0]$, $[0, 1]$, and $[1, 1]$, with bias $[0, 0, -1]$ and a ReLU bend. The second weight row is $[1, 1, 1]$, with bias $0.5$ and no bend.

First pre-activation, row by row: $2$, then $-3$, then $2 + (-3) - 1 = -2$. So $[2, -3, -2]$. ReLU replaces negatives with 0 and leaves $[2, 0, 0]$. The second layer is $2 + 0 + 0 + 0.5 = 2.5$.

That walk is a **forward pass**. The value before the bend is the **pre-activation**. `W` has shape `(out, in)`.

$$
h_{\text{out}} = \sigma(W h_{\text{in}} + b) .
$$

Multiply by $W$, add $b$, and apply $\sigma$ to each entry. The same network, ReLU then identity, still returns 2.5:

```python
z = h @ layer["W"].T + layer["b"]
h = activation(layer["activation"], z)
```

The usual mistake is `W @ h` when `W` is `(out, in)` and `h` is a row. On this example the pre-activations have shapes `(3,)` and `(1,)`.

> [!TIP]
> Print `z.shape` here. You need `(3,)` and then `(1,)`. Any other shape means `W` faces the wrong way.

## Drop the bend and watch the stack collapse

Take $W_1$ with rows $(1, 0)$ and $(0, 2)$, bias $[1, 0]$, then $W_2 = [3, 1]$, bias $4$, and $x = [2, -3]$, with no bend.

First map: $[2, -6]+[1, 0] = [3, -6]$. Second: $9 - 6 + 4 = 7$. Combined weight $[3, 2]$ and bias $7$ also give $6 - 6 + 7 = 7$.

$$
W_2 (W_1 x + b_1) + b_2 = (W_2 W_1)\, x + (W_2 b_1 + b_2) .
$$

One product, one bias. A narrow middle layer caps the rank at that width:

```python
W, b = W2 @ W1, W2 @ b1 + b2
```

The usual mistake is expecting a curve from linear layers alone.

> [!KEY]
> Without an activation between them, any stack of affine layers is one affine layer.

## Cut off the negative numbers

The pre-activation $[2, -3, -2]$ became $[2, 0, 0]$. A nudge of the negative entries does nothing, so those slopes are 0. A nudge of the positive entry passes through, so that slope is 1.

$$
\operatorname{relu}(x) = \max(x, 0), \qquad
\operatorname{relu}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases} .
$$

Positive $x$ stays, negative $x$ becomes 0. The slope is 1 or 0. At 0 it is undefined, and the checks skip that point.

Plug in $[2, -3, -2]$. Values $[2, 0, 0]$, slopes $[1, 0, 0]$:

```python
np.maximum(x, 0.0)
(x > 0).astype(float)
```

The usual mistake is a boolean mask. Cast with `.astype(float)`. Each ReLU unit adds a kink where its pre-activation crosses 0, so the network is piecewise linear.

## Soften the cut with GELU

The **Gaussian error linear unit** matches ReLU far from 0 and curves near 0. At $x = 1$, $\sqrt{2/\pi} = 0.797885$ and $1 + 0.044715 = 1.044715$. Their product is $u = 0.833562$. Then $\tanh(u) = 0.682384$, and $\tfrac12(1 + 0.682384) = 0.841192$. At $x = -1$ the same steps give $-0.158808$, a dip below 0 that ReLU does not make.

$$
\operatorname{gelu}(x) = \tfrac{1}{2}\, x \left(1 + \tanh(u)\right), \qquad
u = \sqrt{2/\pi}\,\left(x + 0.044715\, x^3\right) .
$$

$u$ uses $x$ and $x^3$. Add 1 to $\tanh(u)$, halve, and multiply by $x$. Large positive $x$ returns about $x$. Large negative $x$ returns about 0. At $x = 1$ you recover $0.841192$.

Write $t = \tanh(u)$. The slope of $\tanh$ is $1-t^2$.

$$
\operatorname{gelu}'(x) = \tfrac{1}{2}(1 + t) + \tfrac{1}{2}\, x\, (1 - t^2)\, \sqrt{2/\pi}\,\left(1 + 3 \cdot 0.044715\, x^2\right) .
$$

At $x = 1$ the first term is $0.841192$. Then $1-t^2 = 0.534352$ and the slope of $u$ is $0.797885 \times 1.134145 = 0.904917$. The second term is $0.5 \times 0.534352 \times 0.904917 = 0.241772$. The sum is $1.082964$:

```python
u = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
t = np.tanh(u)
```

The usual mistake is dropping the `3 *` in the slope of $u$, or dropping $\tfrac12(1+t)$.

> [!TIP]
> Check a derivative with `(f(x + h) - f(x - h)) / (2 * h)` at `h = 1e-5`.

## Ask how wide one layer must be

One wide ReLU layer can get close to any continuous function on a bounded region. That is the **universal approximation theorem**. It does not say training finds the weights.

If the slope never exceeds $L$ on $[0, 1]$, about $L/\epsilon$ units keep the error under $\epsilon$.

A line of slope 4, with error under $0.5$, needs about $4/0.5 = 8$ units:

```python
width = L / epsilon
```

The usual mistake is reading the theorem as a promise about gradient descent.

> [!KEY]
> One hidden layer can approximate a continuous function on a bounded region. The width can be enormous.

## Fold the line into a sawtooth

The **tent map** folds $[0, 1]$ in half:

$$
T(x) = 2 \min(x, 1 - x) = 2\operatorname{relu}(x) - 4\operatorname{relu}(x - \tfrac{1}{2}) \quad \text{on } [0, 1] .
$$

At $x = 0.75$, $2\cdot 0.75 - 4\cdot 0.25 = 0.5$, and $T(0.5) = 1$.

A few values line up as follows:

| $x$ | 0 | 0.125 | 0.25 | 0.375 | 0.5 | 0.75 |
|---|---|---|---|---|---|---|
| $T(x)$ | 0 | 0.25 | 0.5 | 0.75 | 1 | 0.5 |
| $T(T(x))$ | 0 | 0.5 | 1 | 0.5 | 0 | 1 |

Two folds give $2^2 = 4$ pieces. After $k$ folds, $k$ layers of width 2 give $2^k$ pieces. One hidden layer needs about $2^k$ units. Depth multiplies kinks. Width adds them.

Fuse $2h_1 - 4h_2$ into the next layer. Both rows are $[2, -4]$ and the bias is $[0, -0.5]$, so the layer probes $y$ and $y - \tfrac12$:

```python
# Later hidden layer: W = [[2, -4], [2, -4]], b = [0, -0.5].
```

The usual mistake is an extra combine layer, which breaks the count `depth + 1`.

> [!KEY]
> $k$ layers of width 2 give $2^k$ pieces. One hidden layer needs about $2^k$ units to match.

## Count parameters and flops

A layer with `W` of shape `(out, in)` has `out * in` weights and `out` biases. A **flop** is one floating-point operation. Charge $2 \cdot \text{in} \cdot \text{out}$ for the product, `out` for the bias, then 1 per output for ReLU, 9 for GELU, and 0 for identity.

A 4-8-8-3 network with ReLU, ReLU, then identity has parameters $(32+8) + (64+8) + (24+3) = 40+72+27 = 139$.

Flops: $2\cdot4\cdot8 + 8 + 8 = 80$, then $2\cdot8\cdot8 + 8 + 8 = 144$, then $2\cdot8\cdot3 + 3 + 0 = 51$. The total is $80+144+51 = 275$:

```python
sum(layer["W"].size + layer["b"].size for layer in params)
```

The usual mistake is reading `W.shape` backwards. Write `out_dim, in_dim = W.shape`. The name `in` is a Python keyword.

> [!WARNING]
> `W` has shape `(out, in)`. Swap them and both counts come out wrong.

> [!RECAP]
> Keep these five facts in hand:
> - A layer is `h @ W.T + b`, with `W` shaped `(out, in)`, then ReLU, GELU, or identity.
> - ReLU's slope is `(x > 0).astype(float)`. GELU's slope has two terms in $t = \tanh(u)$.
> - Later tent layers use weights `[[2, -4], [2, -4]]` and bias `[0, -0.5]`.
> - Count pieces with a slope and an intercept, split at a root, then merge matches.
> - Parameters are `out * in + out`. Flops add $2\cdot\text{in}\cdot\text{out}$, the bias, then 1 or 9 per output.

## Build it step by step

Build the nine functions in this order:

1. **`relu(x)`** returns `np.maximum(x, 0.0)`, same shape. The usual mistake is Python `max` on an array. Write it like this:

    ```python
    def relu(x):
        # Elementwise maximum with 0.
    ```

2. **`relu_grad(x)`** returns 1 where `x > 0` and 0 where `x < 0`, as floats. The value at 0 is unchecked. The usual mistake is a boolean array. Write it like this:

    ```python
    def relu_grad(x):
        # (x > 0).astype(float)
    ```

3. **`gelu(x)`** returns the tanh approximation, same shape. The usual mistake is dropping the cubic term. Write it like this:

    ```python
    def gelu(x):
        # 0.5 * x * (1 + tanh(u)), with u from x and x**3.
    ```

4. **`gelu_grad(x)`** returns both derivative terms from one `t = np.tanh(u)`. The checks use finite differences. The usual mistake is a missing `3 *`. Write it like this:

    ```python
    def gelu_grad(x):
        # 0.5 * (1 + t) plus the bracket's slope.
    ```

5. **`mlp_forward(params, x)`** maps `(in,)` to `(out,)` through dicts with `"W"`, `"b"`, and `"activation"`. Use `h @ W.T + b`. The usual mistake is `W @ h`. Write it like this:

    ```python
    def mlp_forward(params, x):
        # Loop. Identity returns the pre-activation unchanged.
    ```

6. **`sawtooth_network(depth)`** returns `depth + 1` layers. First `W` is `(2, 1)`, both rows `[1]`, bias `[0, -0.5]`, `"relu"`. Middle weights are `[[2, -4], [2, -4]]` with bias `[0, -0.5]`. Output weight is `[[2, -4]]` with `"identity"`. The usual mistake is an extra layer. Write it like this:

    ```python
    def sawtooth_network(depth):
        # depth ReLU layers of width 2, then one identity layer.
    ```

7. **`count_linear_regions_1d(params, x_min, x_max)`** counts pieces. Track a slope and an intercept per unit. Pre-activation uses `W @ slopes` and `W @ intercepts + b`. Split at an interior root $x = -\text{intercept}/\text{slope}$. A negative ReLU midpoint gets slope 0. Identity keeps the line. Merge equal neighbours. The usual mistake is skipping the merge. Write it like this:

    ```python
    def count_linear_regions_1d(params, x_min, x_max):
        # Split on interior roots, zero inactive ReLUs, merge, return the count.
    ```

8. **`parameter_count(params)`** sums `W.size + b.size`. The usual mistake is forgetting biases. Write it like this:

    ```python
    def parameter_count(params):
        # Add both arrays on every layer.
    ```

9. **`flop_count(params)`** reads `out_dim, in_dim = W.shape` and adds the three charges. Return an int. The usual mistake is swapping the shape, or charging `"identity"`. Write it like this:

    ```python
    def flop_count(params):
        # 2 * in * out + out + activation_cost * out.
    ```

## Cost and scaling

For $L$ layers of width $m$, the costs change as follows:

| Change | Parameters | Flops per input |
|---|---|---|
| Double the width $m$ | About 4 times | About 4 times |
| Double the depth $L$ | About 2 times | About 2 times |
| Double the batch size | Unchanged | 2 times |

Inference stores the weights and one layer of activations. Training also keeps every layer's activations for the backward pass.

## Where this breaks

Four limits show up in practice:

- Gradient descent rarely builds the sawtooth, even though depth can represent it.
- A ReLU unit that is negative on every input has slope 0, so its weights stay put.
- Slopes can shrink or grow across many layers. Later lessons cover initialisation, normalisation, and residuals.
- A single wide hidden layer can be enormous. Depth is usually cheaper.

## The exercise

You implement `relu`, `relu_grad`, `gelu`, `gelu_grad`, `mlp_forward`, `sawtooth_network`, `count_linear_regions_1d`, `parameter_count`, and `flop_count`.

Checks compare both derivatives with finite differences. GELU at $-8$ is near 0, and at $8$ it is near 8. `mlp_forward` on $[2, -3]$ returns $[2.5]$. Two identity layers match one composed affine map, with rank at most the middle width. `sawtooth_network(k)` has $2^k$ pieces and matches the closed form. A 4-8-8-3 network has 139 parameters and 275 flops.
