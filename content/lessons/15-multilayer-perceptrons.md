---
title: Multilayer perceptrons
slug: 15-multilayer-perceptrons
part: "Part 3 — Neural networks"
summary: Stack layers with ReLU or GELU between them, see why the nonlinearity matters, and build a deep sawtooth network by hand.
minutes: 55
gpu: false
objectives:
  - Compute ReLU, GELU, and their derivatives, and check them with finite differences.
  - Build a forward pass for any list of affine-then-activation layers.
  - Explain why stacked layers with no nonlinearity collapse into one linear layer.
  - Build a ReLU network whose number of linear pieces doubles with every layer.
  - Count the parameters and flops of a network by hand.
prereqs:
  - 05-logistic-regression
exercise: 15-multilayer-perceptrons
---

# Multilayer perceptrons

Logistic regression computes one weighted sum of the inputs and squashes it.
That draws a single straight boundary. Many real problems need a curved
boundary, or a function that bends in many places.

A multilayer perceptron (MLP) gets there by stacking layers. Each layer takes a
weighted sum and then bends the result with a simple function called an
**activation**. Stack enough of these and the network can bend a line into
almost any shape.

In this lesson you build the building blocks: two activations, a forward pass,
and a network you design by hand, with no training, that shows why depth is so
powerful. You also learn to count what a network costs.

## What you already know

- A linear layer computes `W @ x + b`: a matrix `W`, an input vector `x`, and a bias vector `b`.
- Logistic regression is one linear layer followed by the sigmoid.
- A derivative measures how much a function's output changes when you nudge its input.
- A finite difference, $(f(x+h) - f(x-h)) / 2h$ for a small $h$, approximates a derivative.
- The chain rule: the derivative of $f(g(x))$ is $f'(g(x)) \cdot g'(x)$.

## Layers and the forward pass

A **layer** does two things. First an **affine map**: multiply by a weight
matrix and add a bias. Then an activation, applied to every entry separately.

$$
h_{\text{out}} = \sigma(W h_{\text{in}} + b)
$$

Read it aloud: take the input vector $h_{\text{in}}$, multiply by the weight
matrix $W$, add the bias $b$, and apply the activation $\sigma$ to each entry.
In this exercise, $W$ has shape `(out, in)`: one row per output unit. The
number $W h_{\text{in}} + b$ before the activation is the **pre-activation**.

A network is a list of layers. The **forward pass** feeds the input through
them one after another:

```python
h = x
for layer in params:
    z = h @ layer["W"].T + layer["b"]      # pre-activation, shape (out,)
    h = activation(layer["activation"], z)
```

Writing `h @ W.T` instead of `W @ h` gives the same answer for a vector, and it
also works when `h` is a batch of rows.

> [!TIP]
> Print `z.shape` inside the loop while you test. On the worked example that follows, you must see `(3,)` and then `(1,)`. An error or any other shape means `W` is the wrong way round.

### Worked example by hand

A first layer with `W = [[1, 0], [0, 1], [1, 1]]`, `b = [0, 0, -1]`, and ReLU.
A second layer with `W = [[1, 1, 1]]`, `b = [0.5]`, and no activation. Input
`x = [2, -3]`.

- Pre-activation of layer 1: $[2, -3, 2 + (-3) - 1] = [2, -3, -2]$.
- After ReLU: $[2, 0, 0]$.
- Layer 2: $2 + 0 + 0 + 0.5 = 2.5$.

Your `mlp_forward` must return `[2.5]` for this input.

## Why you need a nonlinearity

Suppose you drop the activation and stack two affine maps:

$$
W_2 (W_1 x + b_1) + b_2 = (W_2 W_1)\, x + (W_2 b_1 + b_2)
$$

Read it aloud: applying one affine map after another is the same as applying a
single affine map, with weight matrix $W_2 W_1$ and bias $W_2 b_1 + b_2$. So a
hundred linear layers are no more powerful than one. Worse, if the middle layer
is narrow, the combined matrix can be no richer than that middle width allows.
Its **rank**, the number of independent directions it can output, is at most
the smallest width in the stack.

```python
W, b = W2 @ W1, W2 @ b1 + b2      # the whole network in one layer
```

> [!KEY]
> Without an activation between them, any stack of affine layers is one affine layer. The activation is what breaks this collapse.

## ReLU

The **rectified linear unit** keeps positive numbers and replaces negative
numbers with 0.

$$
\operatorname{relu}(x) = \max(x, 0), \qquad
\operatorname{relu}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}
$$

Read it aloud: ReLU returns $x$ when it is positive and 0 otherwise. Its slope
is 1 on the right and 0 on the left. At exactly 0 the slope is not defined.
Code usually picks 0 there, and the checks do not test that point.

```python
np.maximum(x, 0.0)
(x > 0).astype(float)
```

A ReLU network is **piecewise linear**: it is made of straight segments joined
at kinks. Each unit adds a kink where its pre-activation crosses 0.

## GELU

The **Gaussian error linear unit** is a smooth cousin of ReLU used in most
transformers. It behaves like ReLU far from 0 but curves gently near 0. The
exercise uses the standard tanh approximation:

$$
\operatorname{gelu}(x) = \tfrac{1}{2}\, x \left(1 + \tanh(u)\right), \qquad
u = \sqrt{2/\pi}\,\left(x + 0.044715\, x^3\right)
$$

Read it aloud: compute an inner value $u$ from $x$ and $x^3$. Take its $\tanh$,
which lies between -1 and 1. Add 1 so you get a number between 0 and 2, halve
it, and multiply by $x$. For large positive $x$, $\tanh(u) \approx 1$ and the
output is about $x$. For large negative $x$, $\tanh(u) \approx -1$ and the
output is about 0.

Its derivative comes from the product rule and the chain rule. Write
$t = \tanh(u)$. The derivative of $\tanh$ is $1 - t^2$:

$$
\operatorname{gelu}'(x) = \tfrac{1}{2}(1 + t) + \tfrac{1}{2}\, x\, (1 - t^2)\, \sqrt{2/\pi}\,\left(1 + 3 \cdot 0.044715\, x^2\right)
$$

> [!TIP]
> Test any derivative you write against a central difference, `(f(x + h) - f(x - h)) / (2 * h)` with `h = 1e-5`. The two agree to about `1e-6` or better when your formula is right.

Read it aloud: the first term is the derivative of the $x$ factor times the
bracket. The second term is $x$ times the derivative of the bracket, which is
$\frac{1}{2}$ times $\tanh$'s slope $1 - t^2$ times the slope of $u$.

```python
u = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
t = np.tanh(u)
```

### Worked example by hand

At $x = 1$: $u = 0.79788 \times 1.044715 = 0.83356$, and $t = \tanh(0.83356) = 0.68238$.

- $\operatorname{gelu}(1) = 0.5 \times 1 \times 1.68238 = 0.84119$.
- $\operatorname{gelu}'(1) = 0.84119 + 0.5 \times (1 - 0.46564) \times 0.79788 \times 1.134145 = 0.84119 + 0.24177 = 1.08296$.
- $\operatorname{gelu}(-1) = -0.15881$. GELU dips slightly below 0, unlike ReLU.

## What one hidden layer can do

The **universal approximation theorem** says, in words: a network with one
hidden layer and a nonlinear activation such as ReLU can get as close as you
like to any continuous function on a bounded region, if the hidden layer is
wide enough.

Two cautions come with it. It does not say how wide "wide enough" is, and for
bumpy functions the width can be huge. It also says nothing about whether
training will find those weights. For a function whose slope never exceeds $L$
on $[0, 1]$, about $L / \epsilon$ ReLU units are enough to stay within error
$\epsilon$, because each unit adds one kink to a piecewise-linear fit.

> [!KEY]
> One hidden layer can approximate any continuous function on a bounded region, but the width it needs can be enormous.

## Depth: the sawtooth

Depth can buy far more than width. The **tent map** folds the interval
$[0, 1]$ in half:

$$
T(x) = 2 \min(x, 1 - x) = 2\operatorname{relu}(x) - 4\operatorname{relu}(x - \tfrac{1}{2}) \quad \text{on } [0, 1]
$$

Read it aloud: $T$ rises from 0 at $x = 0$ to 1 at $x = \frac{1}{2}$, then
falls back to 0 at $x = 1$. You build it with two ReLU units: one that "sees"
$x$ and one that "sees" $x - \frac{1}{2}$. Check the right half: for
$x = 0.75$, $2(0.75) - 4(0.25) = 0.5$, and $2\min(0.75, 0.25) = 0.5$.

Apply the tent map $k$ times in a row and you get a **sawtooth** with $2^k$
straight pieces. Each fold doubles the number of teeth. A network with $k$
layers of width 2 computes it with about $6k$ parameters. A one-hidden-layer
network needs roughly $2^k$ units to make that many kinks, because each unit
adds at most one kink.

> [!KEY]
> Depth multiplies kinks, and width only adds them. With $k$ layers of width 2 you get $2^k$ straight pieces, which one hidden layer matches only with about $2^k$ units.

Depth 2 by hand, $T(T(x))$ at a few points:

| $x$ | 0 | 0.125 | 0.25 | 0.375 | 0.5 | 0.75 |
|---|---|---|---|---|---|---|
| $T(x)$ | 0 | 0.25 | 0.5 | 0.75 | 1 | 0.5 |
| $T(T(x))$ | 0 | 0.5 | 1 | 0.5 | 0 | 1 |

That is two teeth, four straight pieces, which is $2^2$.

To stack tents inside the layer format, fuse the "combine" step
$2h_1 - 4h_2$ of one tent into the weights of the next layer. The next layer
must probe $y$ and $y - \frac{1}{2}$, where $y = 2h_1 - 4h_2$, so both of its
weight rows are $[2, -4]$ and its bias is $[0, -\frac{1}{2}]$.

## Counting parameters and flops

A layer with weights of shape `(out, in)` has `out * in` weights and `out`
biases. A **flop** is one floating-point operation. The exercise charges:

- $2 \times \text{in} \times \text{out}$ for the matrix-vector product, one multiply and one add per weight.
- $\text{out}$ for the bias add.
- Per output element: 1 for ReLU, 9 for GELU, 0 for identity.

For a 4-8-8-3 network with ReLU, ReLU, identity:

- Parameters: $(32 + 8) + (64 + 8) + (24 + 3) = 139$.
- Flops: $(64 + 8 + 8) + (128 + 8 + 8) + (48 + 3 + 0) = 80 + 144 + 51 = 275$.

```python
sum(layer["W"].size + layer["b"].size for layer in params)
```

> [!WARNING]
> `W` has shape `(out, in)`, so read it as `out_dim, in_dim = W.shape`, in that order. You can't name a variable `in`, because it's a Python keyword.

> [!RECAP]
> - A layer computes `h @ W.T + b`, with `W` of shape `(out, in)`, then applies its named activation. `"identity"` leaves the values alone.
> - ReLU is `np.maximum(x, 0.0)` with slope `(x > 0).astype(float)`. The GELU slope has two terms, both built from $t = \tanh(u)$.
> - One tent probes $x$ and $x - \tfrac{1}{2}$. Each later layer fuses $2h_1 - 4h_2$ into weights `[[2, -4], [2, -4]]` and bias `[0, -0.5]`.
> - To count regions, track every unit as a slope and an intercept on each interval, split where a pre-activation crosses 0, and merge equal neighbours at the end.
> - A layer has `out * in + out` parameters and costs $2 \cdot \text{in} \cdot \text{out} + \text{out}$ flops, plus 1 per output for ReLU or 9 for GELU.

## Build it step by step

Build the functions in the order the checks test them.

1. **`relu(x)` and `relu_grad(x)`.** Any shape in, same shape out. Use `np.maximum` and a boolean mask cast with `.astype(float)`. Common mistake: returning a boolean array from `relu_grad`.
2. **`gelu(x)` and `gelu_grad(x)`.** Same shape in and out. Compute `u` and `t = np.tanh(u)` once, then combine them as in the formulas above. Common mistake: forgetting the `3 *` in the slope of `u`, or dropping the first term $\frac{1}{2}(1 + t)$. The check compares your derivative to finite differences.
3. **`mlp_forward(params, x)`.** Takes a list of layer dicts and a vector `x` of shape `(in,)`. Returns shape `(out,)`. Loop over layers, compute `h @ W.T + b`, and apply the named activation: `"relu"`, `"gelu"`, or `"identity"`. Common mistake: using `W @ h` with the wrong shape, or applying an activation to the final `"identity"` layer.
4. **`sawtooth_network(depth)`.** Returns `depth + 1` layers. The first layer has `W` of shape `(2, 1)` with both rows `[1]` and bias `[0, -0.5]`. Each of the `depth - 1` middle layers has `W = [[2, -4], [2, -4]]` and bias `[0, -0.5]`. The output layer has `W = [[2, -4]]`, bias `[0]`, and `"identity"`. Common mistake: adding a separate combine layer with no activation, which breaks the `depth + 1` layer count.
5. **`count_linear_regions_1d(params, x_min, x_max)`.** Keep a list of intervals. On each interval, every unit is an exact straight line in $x$, stored as a slope and an intercept, both arrays. For each layer, compute pre-activation slopes `W @ slopes` and intercepts `W @ intercepts + b`. Split the interval where any pre-activation is 0, at $x = -\text{intercept} / \text{slope}$, only if that point lies strictly inside the interval. On each piece, test the midpoint: a ReLU unit that is off gets slope and intercept 0. At the end, merge neighbouring pieces whose output formula is the same, and return the count. Common mistake: counting a kink that does not change the output, so you skip the final merge.
6. **`parameter_count(params)`.** Sum `W.size + b.size`. Common mistake: forgetting the biases.
7. **`flop_count(params)`.** Read `out_dim, in_dim = W.shape` and add the three charges above for each layer. Return an `int`. Common mistake: swapping `in` and `out` when you read the shape, or charging an activation cost for `"identity"`.

## Cost and scaling

The cost of one forward pass grows with the number of weights. The table shows
how it changes for a network of $L$ layers of width $m$.

| Change | Parameters | Flops per input |
|---|---|---|
| Double the width $m$ | about 4 times | about 4 times |
| Double the depth $L$ | about 2 times | about 2 times |
| Double the batch size | unchanged | 2 times |

Memory for inference is the weights plus one layer of activations. For
training, you also keep every layer's activations for the backward pass, which
you see in the next lesson.

## Where this breaks

**Linear pieces grow, but training may not find them.** The sawtooth shows
what depth can represent. Gradient descent from random weights rarely finds
such a precise construction.

**Dead ReLUs.** If a unit's pre-activation is negative for every input, its
slope is 0 everywhere and it never learns again. Careful initialisation, a
smaller learning rate, or GELU help.

**Deep stacks are hard to train.** Gradients can shrink or grow as they pass
through many layers. Later lessons cover initialisation, normalisation, and
residual connections that fix this.

**Wide shallow networks cost a lot.** The universal approximation theorem
allows one hidden layer, but the width needed can be enormous. Depth is
usually the cheaper route.

## The exercise

You implement `relu`, `relu_grad`, `gelu`, `gelu_grad`, `mlp_forward`,
`sawtooth_network`, `count_linear_regions_1d`, `parameter_count`, and
`flop_count` in `starter.py`.

The checks compare both derivatives to finite differences and test GELU at
$\pm 8$. They run `mlp_forward` on the worked example above, and confirm that
two identity layers equal one composed affine map. They check that
`sawtooth_network(k)` has exactly $2^k$ linear pieces and matches the
closed-form sawtooth to machine precision. Finally, they compare your counts to
139 parameters and 275 flops on a 4-8-8-3 network.
