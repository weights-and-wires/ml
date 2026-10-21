---
title: Regularisation in deep networks
slug: 19-deep-regularisation
part: "Part 3 — Neural networks"
summary: "Build dropout, weight decay, and label smoothing, and the ridge penalty that matches early stopping."
minutes: 60
gpu: false
objectives:
  - Build inverted dropout, and explain why it needs no change at test time.
  - Build a weight-decay step, and explain what it does to weights feeding a normalisation layer.
  - Compute the ridge penalty that a fixed number of gradient-descent steps corresponds to.
  - Build the label-smoothed loss and its gradient, and explain where its minimum sits.
prereqs:
  - 18-initialisation-and-norm
exercise: 19-deep-regularisation
---

# Regularisation in deep networks

A deep network often has far more weights than training examples. It can fit the training labels, noise included. That fit fails on examples it has not seen.

Regularisation is any device that stops the fit from hugging the training set, so error on new data falls. Lesson 4 did this with a penalty that pulls weights toward zero. Deep networks keep that idea and add a few more.

You build dropout, weight decay, the ridge penalty that matches early stopping, and label smoothing. Three of the four pull toward a simpler fit in the same spirit as that penalty.

## What you already know

You already have the following:

- Overfitting means low error on the training set and high error on new data.
- Ridge adds a penalty on the squared size of the weights, which shrinks them.
- Softmax turns a row of scores into probabilities that sum to 1.
- Cross-entropy is the negative log of the probability on the right class.
- A normalisation layer rescales its input to mean 0 and variance 1.

## Switch units off, then scale the rest

Take activations $x = [2, 4, 6, 8]$ and drop each one with probability $0.5$. Suppose the keep-mask comes out $[1, 0, 1, 1]$. The product is $[2, 0, 6, 8]$. Divide by $1 - 0.5$ and you get $[4, 0, 12, 16]$.

That rule is **inverted dropout**. The mask is 0 or 1, same shape as $x$, and a 1 means the unit stayed on. Multiply, then divide by the chance of staying on:

$$
\text{out} = \frac{x \cdot \text{mask}}{1 - p}
$$

Here $p$ is the chance of dropping a unit, not of keeping it. A unit survives when its random draw is below $1 - p$. The same mask and $p$ give $[4, 0, 12, 16]$:

```python
mask = (rng.random(x.shape) < 1.0 - p).astype(x.dtype)
out = x * mask / (1.0 - p)
```

The usual mistake is to divide by $p$, or to treat $p$ as the keep probability.

Each entry survives with probability $1 - p$, so $x \cdot \text{mask}$ averages $(1 - p)\, x$. Dividing by $1 - p$ puts the average back at $x$. Training already matches the input on average, so at test time you return $x$ and a mask of ones. No second scale.

> [!KEY]
> Inverted dropout multiplies the kept units by $1/(1-p)$ in training. Test time returns $x$ unchanged.

> [!WARNING]
> $p$ is the drop probability. A unit stays on when `rng.random` is below $1 - p$.

The backward pass is the same multiply. Dropped units get gradient 0. Kept units pass `dout` through, scaled by $1/(1-p)$:

```python
dx = dout * mask / (1.0 - p)
```

The usual mistake is to forget the $1/(1-p)$ on the way back.

> [!TIP]
> Try `np.ones` of a large length. The mask mean sits near $1-p$, and the output mean sits near 1.

## Shrink every weight by the same fraction

Take $w = 1$, gradient $g = 0.1$, learning rate $\eta = 0.1$, and decay rate $0.01$. The gradient piece removes $0.1 \times 0.1 = 0.01$. The decay piece removes $0.1 \times 0.01 \times 1 = 0.001$. The new weight is $0.989$.

That extra subtraction is **weight decay** for plain gradient descent. The learning rate is the step size. You take the usual step, then also step toward zero:

$$
w_{\text{new}} = w - \eta \, g - \eta \, \lambda \, w
$$

Lambda, $\lambda$, is the decay rate. For plain gradient descent this matches ridge, because the gradient of $\tfrac{\lambda}{2}\|w\|^2$ is $\lambda w$. The same numbers give $0.989$:

```python
new_w = w - lr * grad - lr * weight_decay * w
```

The usual mistake is to forget the learning rate on the decay term.

There is a twist when those weights feed batch norm or layer norm. The norm divides out an overall scale, so doubling the weights leaves the output unchanged. The loss does not care how large they are. Decay cannot simplify the function there. Smaller weights do mean that one gradient step turns them through a larger angle, so on that path decay acts like a larger learning rate.

> [!KEY]
> When weights feed a normalisation layer, their size does not change the output. Decay there acts like a larger learning rate.

## Stop early, and match a ridge penalty

Gradient descent from $w = 0$ moves the steep, well-supported directions first and the flat, noisy ones last. Stopping early leaves the flat directions near zero. Ridge shrinks those directions the most. A step count works like a penalty.

Make that exact on one coordinate. The loss is $\tfrac{1}{2}\lambda (w - w^*)^2$, with curvature $\lambda$ and best weight $w^*$. Take $\lambda = 2$, step size $\eta = 0.1$, and $w^* = 1$. The gradient at 0 is $\lambda(0 - w^*) = -2$. One step gives $0 - 0.1 \times (-2) = 0.2$.

After $k$ steps from zero, gradient descent has moved this far toward $w^*$:

$$
w_k = \big(1 - (1 - \eta\lambda)^k\big)\, w^*
$$

Ridge with penalty $\gamma$ instead multiplies $w^*$ by $\lambda / (\lambda + \gamma)$. Set them equal and solve:

$$
\gamma = \frac{\lambda \, (1 - \eta\lambda)^k}{1 - (1 - \eta\lambda)^k}
$$

The base $(1 - \eta\lambda)^k$ is the fraction of the way still left to go. Gamma is the curvature times that base, divided by one minus the base. The word eigenvalue in the code is this curvature, one coordinate of a quadratic.

With $\lambda = 2$ and $\eta = 0.1$, the base is $0.8^k$. The two rules agree as follows:

| Steps $k$ | Base $0.8^k$ | Penalty $\gamma$ | Both weights |
|---|---|---|---|
| 0 | 1 | infinite | 0 |
| 1 | 0.8 | $2 \times 0.8 / 0.2 = 8$ | 0.2 |
| 10 | 0.10737 | 0.24058 | 0.89263 |

At $k = 0$ the denominator is 0, so $\gamma$ is infinite: no training is an infinitely strong penalty. As $k$ grows, the base and $\gamma$ both go to 0:

```python
base = (1.0 - step * eigenvalue) ** iterations
gamma = eigenvalue * base / (1.0 - base)
```

The usual mistake is to divide by zero when `iterations` is 0. Return `float("inf")` there. If the denominator is 0 or negative, return 0.0.

> [!KEY]
> $k$ steps of gradient descent from zero match ridge with penalty $\gamma$. Fewer steps mean a larger $\gamma$.

## Ask for a little less certainty

Cross-entropy rewards pushing the right class ever closer to probability 1. The scores keep growing. Spread a small amount $\epsilon$ of the target across all $K$ classes instead.

Take scores $z = [2, 0, 0]$, right class 0, $K = 3$, and $\epsilon = 0.1$. Softmax uses $e^2 = 7.389$, so the probabilities are $[7.389, 1, 1] / 9.389 = [0.7870, 0.1065, 0.1065]$.

The softened target is **label smoothing**:

$$
y^{\text{smooth}} = (1 - \epsilon)\,\text{onehot} + \frac{\epsilon}{K}
$$

Take the one-hot row, scale it by $1 - \epsilon$, and add $\epsilon / K$ to every class. The right class gets $1 - \epsilon + \epsilon/K$. Every other class gets $\epsilon/K$. The row still sums to 1. Here that is $[0.9333, 0.0333, 0.0333]$.

The loss on one row, with $p = \mathrm{softmax}(z)$, is ordinary cross-entropy against that target. Average over the $N$ rows:

$$
L_{\text{row}} = -\sum_{c=1}^{K} y^{\text{smooth}}_c \log p_c
$$

Each class contributes its target weight times the log probability. Flip the sign, then average over rows. On this row the loss is $0.3729$. With $\epsilon = 0$ it would be $-\log 0.7870 = 0.2395$:

```python
y_smooth = (1.0 - eps) * np.eye(K)[targets] + eps / K
shifted = logits - logits.max(axis=1, keepdims=True)
log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
loss = float((-(y_smooth * log_p).sum(axis=1)).mean())
```

The usual mistake is to sum over rows instead of taking the mean. Also subtract the row max before the exponential, or a large score overflows.

The gradient in the scores has the same shape as plain softmax cross-entropy:

$$
\frac{\partial L}{\partial z} = \frac{p - y^{\text{smooth}}}{N}
$$

Predicted probabilities minus the smoothed targets, divided by the batch size, because the loss is a mean. On this one row the gradient is $[-0.1463, 0.0732, 0.0732]$. It is zero when $p$ equals the smoothed target, not when $p$ is one-hot:

```python
grad = (p - y_smooth) / N
```

The usual mistake is to leave out the division by $N$.

> [!KEY]
> Label smoothing moves the best prediction off a one-hot target. Pushing the scores ever higher no longer lowers the loss.

> [!RECAP]
> Hold these five points before you write code:
>
> - Dropout zeros each entry with probability $p$ and divides the rest by $1-p$. Test time returns $x$ and a mask of ones.
> - The backward pass multiplies `dout` by that same mask and by $1/(1-p)$.
> - A weight-decay step subtracts `lr * weight_decay * w` on top of the gradient step.
> - The smoothed target is $(1-\epsilon)$ times one-hot, plus $\epsilon/K$. The gradient is $(p - y^{\text{smooth}}) / N$.
> - The early-stopping penalty is $\lambda b / (1-b)$ with $b = (1-\eta\lambda)^k$. It is infinite at $k = 0$.

## Build it step by step

The checks call the functions in this order:

1. **`weight_decay_step(w, grad, lr, weight_decay)`.** Return `w - lr * grad - lr * weight_decay * w`, same shape as `w`. The usual mistake is to drop `lr` from the decay term.
2. **`dropout_forward(x, p, rng, training)`.** Return `(out, mask)`. If `training` is false, return `x` and `np.ones_like(x)`. Otherwise draw the mask from `rng.random(x.shape)`, so it depends only on the shape. The usual mistake is dividing by `p`.
3. **`dropout_backward(dout, mask, p)`.** Return `dout * mask / (1 - p)`, same shape as `dout`. The usual mistake is to skip the scale.
4. **`label_smoothing_loss(logits, targets, eps)`.** `logits` is `(N, K)` and `targets` is `(N,)` of integers. Build the smoothed target with `np.eye(K)[targets]`, take a max-shifted log-softmax, and return the mean over rows as a float. The usual mistake is a sum instead of a mean.
5. **`label_smoothing_grad(logits, targets, eps)`.** Return `(p - y_smooth) / N` with shape `(N, K)`. The usual mistake is to omit `/ N`.
6. **`early_stopping_equivalent_ridge(eigenvalue, step, iterations)`.** Return a float. If `iterations` is 0, return `float("inf")`. Otherwise compute the base and the ratio. If the denominator is 0 or less, return 0.0. The usual mistake is the division at 0 steps.

## Cost and scaling

All four are cheap next to the network itself:

| Method | Extra time per step | Extra memory |
|---|---|---|
| Dropout | one random draw and one multiply per activation | a mask the size of the activations |
| Weight decay | one multiply per weight | none |
| Early stopping | a pass over held-out data now and then | one copy of the best weights |
| Label smoothing | a few operations per score | none |

## Where this breaks

Each of these failures has a direct change:

- Dropout changes the spread of activations between training and test, which confuses batch norm's running averages. Many convolutional nets skip dropout, or use it only after the last norm.
- A drop rate near 1 leaves almost nothing on, and training stalls. Rates from 0.1 to 0.5 are the usual range.
- Decaying biases, and the norm scales $\gamma$ and $\beta$, usually hurts. Leave them out of the decay.
- A jumpy held-out loss can stop training too soon. Wait through several checks with no improvement.
- Smoothing stops the probabilities from sitting at 0 and 1, so they no longer match how often the prediction is right in a simple way. Skip it when you need that match.
- Past the point where training error hits zero, test error can fall again. That pattern is double descent.

## The exercise

You implement `weight_decay_step`, `dropout_forward`, `dropout_backward`, `label_smoothing_loss`, `label_smoothing_grad`, and `early_stopping_equivalent_ridge`.

The checks verify the decay step, dropout's identity at test time, and that the average dropout output matches the input. They check `dropout_backward` against a finite difference with the mask held fixed. They check the smoothing gradient against a finite difference of your loss, and that the gradient vanishes where softmax equals the smoothed target. The last checks want an infinite penalty at 0 steps and a near-zero penalty after many steps. They also match 25 steps of gradient descent to the predicted ridge solution.
