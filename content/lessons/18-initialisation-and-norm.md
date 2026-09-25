---
title: Initialisation and normalisation
slug: 18-initialisation-and-norm
part: "Part 3 — Neural networks"
summary: Pick starting weights that keep a deep network's signal steady, then build batch norm and layer norm to keep it that way.
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

Before training starts, every weight in a network needs a value. You pick them
at random. How large those random numbers are turns out to matter a great
deal. Too small, and the signal shrinks a little at every layer until nothing
reaches the end. Too large, and it grows until the numbers overflow.

In a shallow network you barely notice. In a network with fifty layers, a
shrink of 30% per layer leaves almost nothing: $0.7^{50}$ is about one in
fifty million. The gradients shrink the same way on the way back, so the early
layers never learn.

This lesson has two halves. First you work out the right starting scale, and
you confirm it with a simulation. Then you build normalisation layers, which
rescale the signal inside the network so that a slightly wrong starting scale
stops mattering.

## What you already know

- A linear layer computes $h \, W$, where each output is a weighted sum of the inputs.
- ReLU sets negative values to zero and keeps positive ones.
- The variance of a set of numbers is the average squared distance from their mean.
- Backpropagation passes gradients from the output back through every layer.
- A gradient can be checked against a finite difference.

## Why the scale of the weights matters

Think of a line of people passing a message by whispering. If each person
whispers a little quieter than they heard, the message fades out by the end of
the line. If each speaks a little louder, it turns into shouting. You want each
person to pass it on at the same volume.

In a network, "volume" is the variance of the activations. Take one layer with
$n_{\text{in}}$ inputs. Each output is a sum of $n_{\text{in}}$ terms, each an
input times a weight. Suppose the inputs have variance 1 and the weights are
drawn with mean 0 and standard deviation $s$. Then each output has variance

$$
\text{Var}(\text{output}) = n_{\text{in}} \, s^2 \cdot \text{Var}(\text{input})
$$

Read it aloud: the output variance is the number of inputs, times the weight
variance $s^2$, times the input variance. A sum of $n_{\text{in}}$ independent
terms has $n_{\text{in}}$ times the variance of one term, and each term has
variance $s^2 \cdot \text{Var}(\text{input})$.

To keep the variance the same from layer to layer, you need
$n_{\text{in}} \, s^2 = 1$. Any other value multiplies the variance by that
factor at every layer, and fifty layers raise it to the fiftieth power.

```python
h = rng.normal(size=(num_trials, n_in))        # variance 1
W = rng.normal(scale=s, size=(n_in, n_out))
print(np.mean((h @ W) ** 2), n_in * s**2)       # these agree
```

### Xavier initialisation

Xavier (also called Glorot) initialisation balances the forward and backward
pass. The forward pass wants $n_{\text{in}} s^2 = 1$, and the backward pass
wants $n_{\text{out}} s^2 = 1$. Xavier takes the compromise:

$$
s_{\text{Xavier}} = \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}
$$

Read it aloud: the standard deviation is the square root of 2 over the sum of
the number of inputs and outputs. When the two are equal, this is
$\sqrt{1 / n_{\text{in}}}$, exactly what the forward pass needs. It suits
activations that are close to linear near zero, such as tanh.

```python
s = np.sqrt(2.0 / (n_in + n_out))
```

### He initialisation

ReLU throws away the negative half of its input. That halves the average
squared value, so each ReLU layer multiplies the signal's size by
$n_{\text{in}} s^2 / 2$. He initialisation doubles the weight variance to
make up for it:

$$
s_{\text{He}} = \sqrt{\frac{2}{n_{\text{in}}}}
$$

Read it aloud: the standard deviation is the square root of 2 over the number
of inputs. Then $n_{\text{in}} s^2 / 2 = 1$, and the signal keeps the same
size through any number of ReLU layers.

```python
s = np.sqrt(2.0 / n_in)
```

### Worked example: fifty layers

Take a width of 100 at every layer:

- Xavier gives $s = \sqrt{2 / 200} = 0.1$, so $n_{\text{in}} s^2 = 1$. With
  ReLU, each layer multiplies the size by $1/2$. After 50 layers, that is
  $0.5^{50} \approx 9 \times 10^{-16}$. The signal is gone.
- He gives $s = \sqrt{2 / 100} \approx 0.1414$, so
  $n_{\text{in}} s^2 = 2$. With ReLU, each layer multiplies by
  $2 / 2 = 1$. After 50 layers, the size is still about 1.

Your `forward_variance` function runs this experiment for real.

### Measuring signal size

After a ReLU, the activations are no longer centred on zero, because they are
all zero or positive. What the next layer's variance depends on is the average
squared value, $\text{mean}(h^2)$, not the variance around the mean. For the
input and for any linear layer's output, the mean is zero, so the two are
equal. The simulation records $\text{mean}(h^2)$ at every layer and calls it
the variance.

```python
variances.append(float(np.mean(h**2)))
```

## Normalisation layers

Getting the starting scale right only helps at the start. As training changes
the weights, the signal can drift again. A normalisation layer rescales the
activations inside the network, at every step, so they always have mean 0 and
variance 1.

Think of the volume leveller on a music app. However loud or quiet a song was
recorded, the leveller brings it to the same volume. Then the layer lets the
network choose its own volume with two learned numbers per feature: a scale
$\gamma$ (gamma) and a shift $\beta$ (beta).

For a group of $M$ numbers $x_1, \dots, x_M$, the layer computes

$$
\mu = \frac{1}{M}\sum_{i=1}^{M} x_i, \qquad
\sigma^2 = \frac{1}{M}\sum_{i=1}^{M} (x_i - \mu)^2, \qquad
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad
y_i = \gamma \, \hat{x}_i + \beta
$$

Read it aloud, left to right:

- $\mu$ (mu) is the mean of the group.
- $\sigma^2$ (sigma squared) is the variance of the group. It divides by $M$,
  not $M - 1$, which is what `np.var` does by default.
- $\hat{x}_i$ ("x hat") is each value minus the mean, divided by the standard
  deviation. The small $\epsilon$, such as $10^{-5}$, prevents division by zero.
- $y_i$ is the output: the normalised value times $\gamma$, plus $\beta$.

The only question is which numbers form a group.

### Batch norm and layer norm

Your activations are a matrix `x` of shape `(N, D)`: $N$ examples in the
batch, $D$ features each.

- **Batch normalisation** groups each feature across the batch. It normalises
  each column, so it reduces over `axis=0`, and $M = N$.
- **Layer normalisation** groups each example across its features. It
  normalises each row, so it reduces over `axis=1`, and $M = D$.

In both, $\gamma$ and $\beta$ have shape `(D,)`, one per feature.

```python
mean = x.mean(axis=axis, keepdims=True)
var = x.var(axis=axis, keepdims=True)
istd = 1.0 / np.sqrt(var + eps)
xhat = (x - mean) * istd
out = gamma * xhat + beta
```

The `keepdims=True` keeps the reduced axis with length 1, so broadcasting
lines up the mean with the right rows or columns.

### Worked example: layer norm on one row

Take one example with features $[1, 2, 3]$, $\gamma = 1$, and $\beta = 0$:

1. Mean: $\mu = (1 + 2 + 3) / 3 = 2$.
2. Differences from the mean: $[-1, 0, 1]$.
3. Variance: $\sigma^2 = (1 + 0 + 1) / 3 = 0.6667$.
4. Standard deviation: $\sqrt{0.6667 + 0.00001} \approx 0.8165$.
5. Normalised: $[-1, 0, 1] / 0.8165 \approx [-1.2247, 0, 1.2247]$.

Add 37.5 to every feature of that row and you get the same output, because the
mean shifts by 37.5 too. Layer norm doesn't care about a constant shift of a
whole example. Batch norm does, because shifting one example changes the
column means that every example shares.

### Why layer norm doesn't depend on the batch

Batch norm's output for one example depends on every other example in the
batch. With a batch of two, its statistics are very noisy. With a batch of
one, the variance of each column is zero and the layer outputs nothing useful.
At test time you often predict for one example at a time, so batch norm keeps
running averages of $\mu$ and $\sigma^2$ from training to use instead.

Layer norm uses only the features of the example itself. It behaves the same
for any batch size, in training and at test time. That is why transformers and
recurrent networks use it.

## The backward pass

To train through a normalisation layer, you need the gradient of the loss with
respect to its input $x$ and to $\gamma$ and $\beta$. Call the incoming
gradient `dout`, the same shape as the output.

The gradients for $\gamma$ and $\beta$ follow from $y = \gamma \hat{x} + \beta$.
Each feature's $\gamma$ and $\beta$ touch every row, so you sum over the batch
axis. This is `axis=0` for both batch norm and layer norm:

```python
dbeta = dout.sum(axis=0)
dgamma = (dout * xhat).sum(axis=0)
```

The gradient for $x$ is harder, because each $x_i$ affects the output three
ways: directly, through the mean, and through the variance. Let
$d\hat{x} = \text{dout} \cdot \gamma$, and let "sum" mean a sum over the
normalised group, the same axis the forward pass reduced over. Then

$$
dx = \frac{1}{M\sqrt{\sigma^2 + \epsilon}}\Big(M\, d\hat{x} - \textstyle\sum d\hat{x} - \hat{x} \sum \big(d\hat{x} \cdot \hat{x}\big)\Big)
$$

Read it aloud: start from $M$ times the gradient at $\hat{x}$. Subtract its
sum over the group, which undoes the effect of the mean. Subtract $\hat{x}$
times the sum of $d\hat{x} \cdot \hat{x}$, which undoes the effect of the
variance. Scale the result by one over $M$ times the standard deviation.

```python
dxhat = dout * gamma
dx = (istd / M) * (M * dxhat
                   - dxhat.sum(axis=axis, keepdims=True)
                   - xhat * (dxhat * xhat).sum(axis=axis, keepdims=True))
```

Batch norm and layer norm share this formula. Only `axis` and $M$ change.

## Build it step by step

The checks test the functions in this order.

1. **`xavier_scale(n_in, n_out)`.** Returns a float. One line with `np.sqrt`.
   The common mistake is returning the variance $2 / (n_{\text{in}} +
   n_{\text{out}})$ instead of its square root.
2. **`he_scale(n_in)`.** Same pattern, with `2 / n_in`.
3. **`forward_variance(depth, widths, scale, activation, num_trials, seed)`.**
   Make `rng = np.random.default_rng(seed)`. Draw the input `h` with shape
   `(num_trials, widths[0])`, and record `np.mean(h**2)`. For each of `depth`
   layers, draw a weight matrix of shape `(widths[l], widths[l + 1])` with
   `rng.normal(scale=scale, ...)`, compute `h @ W`, apply `np.maximum(..., 0)`
   if the activation is `"relu"`, and record `np.mean(h**2)`. Return an array
   of length `depth + 1`. The common mistake is applying ReLU to the input
   before the first layer.
4. **`batchnorm_forward(x, gamma, beta, eps)`.** Takes `x` of shape `(N, D)`.
   Returns `(out, cache)`. Reduce over `axis=0`. Put `xhat`, the inverse
   standard deviation, `gamma`, and the axis in the cache. A shared helper that
   takes `axis` saves you writing this twice.
5. **`layernorm_forward(x, gamma, beta, eps)`.** The same, over `axis=1`. The
   common mistake is forgetting `keepdims=True`, which breaks broadcasting.
6. **`batchnorm_backward(dout, cache)`.** Returns `(dx, dgamma, dbeta)` with
   shapes `(N, D)`, `(D,)`, and `(D,)`. Use the formula above with `axis=0`.
7. **`layernorm_backward(dout, cache)`.** The same formula with `axis=1` and
   $M = D$. The common mistake is summing `dgamma` and `dbeta` over `axis=1`.
   They are per-feature, so they always sum over `axis=0`.

## Cost and scaling

Initialisation costs nothing during training: you pick the scale once.
Normalisation layers add a small cost to every forward and backward pass.

| Operation | Time | Extra memory |
|---|---|---|
| Choose Xavier or He scale | once, no cost | none |
| Normalisation forward pass | a few passes over the `(N, D)` activations | the cache, about one more `(N, D)` array |
| Normalisation backward pass | a few more passes over `(N, D)` | none beyond the cache |
| `forward_variance` simulation | one matrix multiply per layer | one `(num_trials, width)` array at a time |

The time for normalisation grows in a straight line with the number of
activations. It is small next to the matrix multiplies of the linear layers.

## Where this breaks

- **Wrong scale for the activation.** Xavier with ReLU shrinks the signal by
  half per layer. Use He for ReLU and its relatives.
- **Very small batches with batch norm.** The batch statistics are too noisy.
  Use layer norm or group norm instead.
- **Forgetting test mode.** Batch norm must use its stored running averages at
  test time, or one prediction depends on which other examples happen to share
  its batch.
- **Constant features.** When every value in a group is equal, $\sigma^2 = 0$
  and only $\epsilon$ keeps the division finite.
- **Very deep networks.** Good initialisation and normalisation still aren't
  always enough past a hundred layers or so. Residual connections, which add
  each block's input back to its output, give gradients a shortcut. Lesson 21
  covers them.

## The exercise

You implement `xavier_scale`, `he_scale`, `forward_variance`,
`batchnorm_forward`, `batchnorm_backward`, `layernorm_forward`, and
`layernorm_backward`. Each forward function returns `(out, cache)`, and its
backward function takes that cache back.

The checks verify both scale formulas, that one linear layer multiplies the
variance by $n \, s^2$, that each normalisation gives mean 0 and variance 1
along the right axis, and that both backward passes match finite differences
for $x$, $\gamma$, and $\beta$. They check that layer norm ignores a constant
shift of one example while batch norm does not. The last check runs a
fifty-layer ReLU network and expects He to keep the signal flat and Xavier to
halve it at every layer.
