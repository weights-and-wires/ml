---
title: Regularisation in deep networks
slug: 19-deep-regularisation
part: "Part 3 — Neural networks"
summary: Build dropout, weight decay, and label smoothing, and compute the ridge penalty that stopping training early is worth.
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

A deep network usually has far more weights than training examples. It can
memorise the training set, labels and noise and all. Memorising is not
learning: a model that memorises does badly on examples it hasn't seen.

Regularisation is any trick that stops a model from fitting the training data
too closely, so it does better on new data. In lesson 4 you met ridge
regression, which adds a penalty that pulls weights toward zero. Deep networks
use that idea and a few others.

In this lesson you build four of them: dropout, weight decay, early stopping,
and label smoothing. You also see that three of the four pull the model toward
something simpler in the same way ridge does.

## What you already know

- Overfitting means the model fits the training data well but new data badly.
- Ridge regression adds $\lambda \|w\|^2$ to the loss, which shrinks the weights.
- Softmax turns a row of scores into probabilities that sum to 1.
- Cross-entropy loss is the negative log of the probability given to the right class.
- Normalisation layers rescale their input to mean 0 and variance 1.

## Dropout

Picture a team where any member might be off sick on a given day. Nobody can
rely on one colleague to do everything, so the whole team learns to cover
each job. Dropout does this to a network. During training, it switches off
each unit at random, so the network can't rely on any single unit.

Here is how inverted dropout works. For each entry of the activations $x$,
flip a biased coin. With probability $p$, set it to zero. Otherwise keep it,
and divide it by $1 - p$:

$$
\text{out} = \frac{x \cdot \text{mask}}{1 - p}
$$

Read it aloud: the mask is an array of 0s and 1s the same shape as $x$. Each
entry is 1 with probability $1 - p$. Multiply $x$ by the mask, then divide by
the chance of keeping a unit.

```python
mask = (rng.random(x.shape) < 1 - p).astype(x.dtype)
out = x * mask / (1 - p)
```

Why divide? Each entry survives with probability $1 - p$, so on average
$x \cdot \text{mask}$ is $(1 - p)\, x$. Dividing by $1 - p$ brings the average
back to $x$. Because the average output already matches the input during
training, you don't need to change anything at test time. At test time, dropout
returns $x$ as it is, with a mask of all ones.

### Worked example: dropout

Take $x = [2, 4, 6, 8]$, $p = 0.5$, and suppose the mask comes out as
$[1, 0, 1, 1]$:

1. $x \cdot \text{mask} = [2, 0, 6, 8]$.
2. Divide by $1 - p = 0.5$: $[4, 0, 12, 16]$.

The input averages 5 and this output averages 8, because this mask kept three
of four. Over many masks, the output averages 5.

### The backward pass

Dropout is a multiply by a fixed number for each entry. The gradient passes
back through the same multiply:

```python
dx = dout * mask / (1 - p)
```

Units that were dropped get zero gradient. Units that were kept get the
incoming gradient scaled by $1 / (1 - p)$.

## Weight decay

Weight decay shrinks every weight a little on every step. For plain gradient
descent, the step is

$$
w_{\text{new}} = w - \eta \, g - \eta \, \lambda \, w
$$

Read it aloud: take the normal gradient step, the learning rate $\eta$ times
the gradient $g$ of the loss. Then also subtract the learning rate times the
decay rate $\lambda$ times the weights. That second term pulls each weight a
small fraction of the way toward zero.

```python
new_w = w - lr * grad - lr * weight_decay * w
```

For plain gradient descent this is the same as ridge: the gradient of
$\tfrac{\lambda}{2}\|w\|^2$ is $\lambda w$. For example, with $w = 1.0$,
$g = 0.1$, $\eta = 0.1$, and $\lambda = 0.01$, the new weight is
$1.0 - 0.01 - 0.001 = 0.989$.

### Weight decay before a normalisation layer

There is a twist in modern networks. Suppose a layer's weights feed straight
into batch norm or layer norm. The norm layer divides out any overall scale,
so doubling those weights gives exactly the same output. The loss doesn't care
how large the weights are.

So weight decay can't make the model "simpler" there. What it does instead is
keep the weights small, and smaller weights mean a given gradient step turns
them by a larger angle. In that setting, weight decay acts like a larger
learning rate, not like a penalty on complexity.

## Early stopping

Early stopping means you watch the loss on held-out data during training, and
you stop when it starts getting worse. It sounds unrelated to ridge, but it
does almost the same thing.

Here is the intuition. Gradient descent starting from $w = 0$ learns the
steep, well-supported directions of the problem first, and the flat, noisy ones
last. Stopping early leaves the flat directions close to zero. Ridge also
shrinks the flat directions the most. So a step count works like a ridge
penalty.

You can make that exact for one direction. Suppose the loss along it is
$\tfrac{1}{2}\lambda (w - w^*)^2$, where $\lambda$ is the curvature (how steep
the bowl is) and $w^*$ is the best weight. Gradient descent from 0 with
learning rate $\eta$ reaches, after $k$ steps,

$$
w_k = \big(1 - (1 - \eta\lambda)^k\big)\, w^*
$$

Ridge with penalty $\gamma$ (gamma) gives $w = \frac{\lambda}{\lambda + \gamma}
\, w^*$. Set the two equal and solve for $\gamma$:

$$
\gamma = \frac{\lambda \, (1 - \eta\lambda)^k}{1 - (1 - \eta\lambda)^k}
$$

Read it aloud: compute the base $(1 - \eta\lambda)^k$, the fraction of the way
gradient descent still has to go. The equivalent ridge penalty is the curvature
times that base, divided by one minus the base.

```python
base = (1.0 - step * eigenvalue) ** iterations
gamma = eigenvalue * base / (1.0 - base)
```

Two limits make sense. At $k = 0$, the base is 1 and the denominator is 0, so
$\gamma$ is infinite: no training means the weight stays at zero, like an
infinitely strong penalty. As $k$ grows, the base goes to 0 and so does
$\gamma$: training forever means no regularisation at all.

The word eigenvalue in the code means the curvature along one independent
direction of a quadratic loss. The checks use a loss where each coordinate is
its own direction.

### Worked example: early stopping

Take $\lambda = 2$, $\eta = 0.1$, and $w^* = 1$. Then $1 - \eta\lambda = 0.8$.

| Steps $k$ | Base $0.8^k$ | $\gamma$ | Gradient descent $w_k$ | Ridge $\lambda/(\lambda+\gamma)$ |
|---|---|---|---|---|
| 0 | 1 | infinite | 0 | 0 |
| 1 | 0.8 | $2 \times 0.8 / 0.2 = 8$ | 0.2 | $2/10 = 0.2$ |
| 10 | 0.10737 | 0.24058 | 0.89263 | 0.89263 |

The last two columns agree, which is the whole point.

## Label smoothing

A classifier trained with cross-entropy is rewarded for pushing the right
class's probability ever closer to 1. It never quite gets there, so it keeps
growing its scores. The model ends up overconfident.

Label smoothing asks for a little less certainty. Instead of a target of 1 on
the right class and 0 elsewhere, it spreads a small amount $\epsilon$ of the
target evenly over all $K$ classes:

$$
y^{\text{smooth}} = (1 - \epsilon)\,\text{onehot} + \frac{\epsilon}{K}
$$

Read it aloud: take the one-hot target (1 on the right class, 0 elsewhere),
scale it by $1 - \epsilon$, and add $\epsilon / K$ to every class. The right
class gets $1 - \epsilon + \epsilon/K$, and each other class gets
$\epsilon / K$. The row still sums to 1.

```python
y_smooth = (1 - eps) * np.eye(K)[targets] + eps / K
```

The loss is the cross-entropy against this target, averaged over the $N$
rows of the batch. For one row with probabilities $p = \text{softmax}(z)$:

$$
L_{\text{row}} = -\sum_{c=1}^{K} y^{\text{smooth}}_c \log p_c
$$

Read it aloud: for each class $c$, multiply the target weight by the log of
the predicted probability, add them up, and flip the sign. Then average over
rows. Compute $\log p$ with the log-softmax trick: subtract each row's maximum
score before exponentiating, so large scores don't overflow.

```python
shifted = logits - logits.max(axis=1, keepdims=True)
log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
loss = -(y_smooth * log_p).sum(axis=1).mean()
```

The gradient with respect to the scores has the same neat form as plain
softmax cross-entropy:

$$
\frac{\partial L}{\partial z} = \frac{p - y^{\text{smooth}}}{N}
$$

Read it aloud: predicted probabilities minus smoothed targets, divided by the
batch size $N$ because the loss is a mean.

This gradient is zero when $p$ equals $y^{\text{smooth}}$. So the best
prediction is no longer one-hot. A near-certain prediction scores worse than
one that matches the smoothed target.

### Worked example: label smoothing

Take one row with scores $z = [2, 0, 0]$, right class 0, $K = 3$, and
$\epsilon = 0.1$:

1. Softmax: $e^2 = 7.389$, so $p = [7.389, 1, 1] / 9.389 = [0.7870, 0.1065, 0.1065]$.
2. Smoothed target: $[0.9 + 0.0333, 0.0333, 0.0333] = [0.9333, 0.0333, 0.0333]$.
3. Loss: $-(0.9333 \log 0.7870 + 2 \times 0.0333 \log 0.1065) = 0.3729$.
4. Gradient with $N = 1$: $p - y^{\text{smooth}} = [-0.1463, 0.0732, 0.0732]$.

Without smoothing, the loss would be $-\log 0.7870 = 0.2395$.

## Build it step by step

The checks test the functions in this order.

1. **`weight_decay_step(w, grad, lr, weight_decay)`.** Returns the new `w`,
   same shape. One line. The common mistake is forgetting to multiply the
   decay by `lr`.
2. **`dropout_forward(x, p, rng, training)`.** Returns `(out, mask)`. If
   `training` is `False`, return `x` and `np.ones_like(x)`. Otherwise draw the
   mask from `rng.random(x.shape)`, so it depends only on the shape. The common
   mistake is dividing by `p` instead of `1 - p`.
3. **`dropout_backward(dout, mask, p)`.** Returns `dx`, same shape as `dout`.
   One line.
4. **`label_smoothing_loss(logits, targets, eps)`.** Takes `logits` of shape
   `(N, K)` and integer `targets` of shape `(N,)`. Returns a float. Build the
   smoothed target with `np.eye(K)[targets]`, compute log-softmax with the
   max-shift, and average over rows. The common mistake is summing over rows
   instead of taking the mean.
5. **`label_smoothing_grad(logits, targets, eps)`.** Returns an `(N, K)` array.
   Compute softmax with the same max-shift and return
   `(p - y_smooth) / N`. The common mistake is leaving out the `/ N`.
6. **`early_stopping_equivalent_ridge(eigenvalue, step, iterations)`.** Returns
   a float. Return `float("inf")` when `iterations` is 0. Otherwise compute the
   base and the ratio. If the denominator is 0 or less, return 0.0. The common
   mistake is dividing by zero at `iterations == 0`.

## Cost and scaling

All four regularisers are cheap next to the network itself.

| Method | Extra time per step | Extra memory |
|---|---|---|
| Dropout | one random draw and one multiply per activation | the mask, same size as the activations |
| Weight decay | one multiply per weight | none |
| Early stopping | one pass over held-out data every so often | one saved copy of the best weights |
| Label smoothing | a few extra operations per score | none |

Early stopping can save time overall, because you stop training sooner.

## Where this breaks

- **Dropout with batch norm.** Dropout changes the spread of activations
  between training and test, which confuses batch norm's running averages.
  Many convolutional networks skip dropout, or use it only after the last
  normalisation layer.
- **Dropout rate too high.** With $p$ near 1, almost nothing survives and
  training stalls. Rates of 0.1 to 0.5 are typical.
- **Weight decay on the wrong weights.** Decaying biases and normalisation
  $\gamma$ and $\beta$ usually hurts. People exclude them.
- **Noisy held-out loss.** Early stopping on a jumpy curve can stop too soon.
  Wait for several checks without improvement before stopping.
- **Label smoothing and calibrated probabilities.** Smoothing keeps the model
  from being overconfident, but its probabilities no longer match how often
  it is right in a simple way. Avoid it when you need exact probabilities.
- **Parameter counts don't tell the whole story.** Classical theory says more
  parameters means more overfitting. Deep networks often do the opposite: as
  they grow past the point where they fit the training set exactly, their test
  error can fall again. This is called double descent. It is one reason people
  rely on these regularisers rather than keeping networks small.

## The exercise

You implement `weight_decay_step`, `dropout_forward`, `dropout_backward`,
`label_smoothing_loss`, `label_smoothing_grad`, and
`early_stopping_equivalent_ridge`.

The checks verify the weight-decay step, that dropout returns its input at test
time, that its average output matches the input over many draws, and that
`dropout_backward` matches a finite difference with the mask held fixed. They
check your label-smoothing gradient against a finite difference of your loss,
and that the gradient vanishes where softmax equals the smoothed target. The
last checks confirm the early-stopping penalty is infinite at zero steps and
near zero after many steps, and that gradient descent for 25 steps lands
exactly where your predicted ridge penalty says it should.
