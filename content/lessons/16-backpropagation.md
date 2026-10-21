---
title: Backpropagation as reverse-mode AD
slug: 16-backpropagation
part: "Part 3 — Neural networks"
summary: "Record a calculation on a tape and walk the chain rule backwards so one pass returns every gradient."
minutes: 70
gpu: false
objectives:
  - Explain how backpropagation applies the chain rule from the loss back to every input.
  - Build the backward rule for addition, multiplication, matrix products, ReLU, and sum.
  - Compute the softmax cross-entropy loss and its gradient with respect to the logits.
  - Build a topological sort so every gradient is complete before it flows further.
  - Check any gradient against a finite-difference estimate.
prereqs:
  - 15-multilayer-perceptrons
exercise: 16-backpropagation
---

# Backpropagation as reverse-mode AD

Training needs a slope of the loss for every weight. A gradient is that list of slopes. Nudge one input by 0.01 and the output moves by about 0.03, and that slope is 3. Hand-written slopes do not reach millions of weights.

Backpropagation records each operation, then walks those steps backwards with the chain rule. One walk returns every slope, for about the cost of one more forward pass.

You can build that walk. A value remembers its operation, and a backward method returns slope to the inputs.

## What you already know

Five earlier ideas are enough to start:

- The chain rule multiplies slopes. If $y = f(u)$ and $u = g(x)$, then $dy/dx = (dy/du)\cdot(du/dx)$.
- A forward pass is a chain of matrix products, bias adds, and ReLUs.
- Softmax turns scores into probabilities that add to 1. Scores of 0 and 0 become 0.5 and 0.5.
- A finite difference $(f(x+h)-f(x-h))/(2h)$ estimates a slope.
- Broadcasting stretches a bias of shape `(m,)` across an `(n, m)` matrix.

## Walk one product and one sum backwards

Take $a = 3$ and $b = 2$. Multiply first: $p = a \times b = 6$. Then add: $c = p + a = 9$. The tape holds a multiply, then an add.

The slope of $c$ in itself is 1. Addition passes it through, so $p$ receives 1 and $a$ receives 1. Multiplication gives each input the incoming slope times the other input. From $p$, $a$ receives $1 \times 2 = 2$ and $b$ receives $1 \times 3 = 3$. The value $a$ was used twice, so $1 + 2 = 3$. The slope in $b$ is 3.

Expand to check. $c = ab + a$. With $b = 2$ this is $3a$, so the slope in $a$ is $b + 1 = 3$. The slope in $b$ is $a = 3$.

This tape is **reverse-mode automatic differentiation**. The map from an output slope to input slopes is a **vector-Jacobian product** (VJP). You never build the full table of partial derivatives. Write the incoming slope as $\bar{z}$.

One multiply has a second trap. At $a = 3$, $a \times a = 9$. Each factor contributes $1 \times 3 = 3$, so the total is $3 + 3 = 6$, which is $2a$. Overwrite instead of add, and you keep only 3.

For an add and an elementwise product, the rules are

$$
z = x + y:\quad \bar{x} \mathrel{+}= \bar{z},\ \ \bar{y} \mathrel{+}= \bar{z}
\qquad\qquad
z = x \odot y:\quad \bar{x} \mathrel{+}= \bar{z} \odot y,\ \ \bar{y} \mathrel{+}= \bar{z} \odot x .
$$

The $\mathrel{+}=$ adds onto the slope already stored. Plug in $\bar{c} = 1$. The add gives $\bar{p} \mathrel{+}= 1$ and $\bar{a} \mathrel{+}= 1$. The multiply gives $\bar{a} \mathrel{+}= 2$ and $\bar{b} \mathrel{+}= 3$. Stored $\bar{a}$ is 3:

```python
self.grad += _unbroadcast(out.grad, self.value.shape)
self.grad += _unbroadcast(out.grad * other.value, self.value.shape)
```

The usual mistake is `=` instead of `+=`.

> [!KEY]
> Gradients accumulate. Every backward rule uses `+=`.

## Pass a slope through each operation

**Broadcasting** sums the copies. Add bias $[10, 20]$ onto two rows. Suppose the incoming slope has rows $(1, 2)$ and $(3, 4)$. The bias slope is the column sum, $[1+3,\ 2+4] = [4, 6]$. Call `_unbroadcast` in `__add__` and `__mul__` before every `+=`:

```python
bias_grad = out.grad.sum(axis=0)
```

The usual mistake is a slope of shape `(n, m)` on a bias of shape `(m,)`.

> [!KEY]
> A copied value's gradient is the sum of the gradients of the copies.

A matrix product uses a transpose. Take $A = [1, 2]$ and $B = [3, 4]$. Then $Z = 1\cdot 3 + 2\cdot 4 = 11$. Slopes in $A$ are 3 and 4. Slopes in $B$ are 1 and 2.

For $Z = AB$, with shapes `(n, k)` and `(k, m)`,

$$
\bar{A} \mathrel{+}= \bar{Z} B^\top, \qquad \bar{B} \mathrel{+}= A^\top \bar{Z} .
$$

With $\bar{Z} = 1$ you recover $\bar{A} = [3, 4]$ and $\bar{B} = [1, 2]$:

```python
self.grad += out.grad @ other.value.T
other.grad += self.value.T @ out.grad
```

The usual mistake is a transpose on the wrong factor. A gradient has the same shape as its value.

ReLU passes slope where the input was positive. Take $x = [-1, 2]$ and incoming slope $[5, 7]$. The mask is $[0, 1]$, so the outgoing slope is $[0, 7]$.

A sum gives every entry weight 1. Take $x = [1, 2, 3]$, sum 6. Incoming slope 4 means every entry receives 4.

$$
z = \operatorname{relu}(x):\quad \bar{x} \mathrel{+}= \bar{z} \odot [x > 0]
\qquad\qquad
z = \textstyle\sum_i x_i:\quad \bar{x} \mathrel{+}= \bar{z} \cdot \mathbf{1} .
$$

Plug in. ReLU returns $[0, 7]$. Sum returns $[4, 4, 4]$:

```python
self.grad += out.grad * (self.value > 0)
self.grad += np.ones_like(self.value) * out.grad
```

The usual mistake is masking ReLU with the output. The rule names the input.

> [!TIP]
> When a transpose is in doubt, keep the only product whose shapes fit.

## Finish a node before you leave it

A node must collect every contribution before it passes slope onward.

Take $a = 2$, $b = a \times a = 4$, $c = a + a = 4$, and $d = b + c = 8$. Then $d = a^2 + 2a$, so the slope at 2 is $4 + 2 = 6$.

Walk it. $d$ sends 1 to $b$ and 1 to $c$. The square sends $1 \times 2$ on each factor, so $b$ contributes 4 to $a$. The add contributes 2. The total is 6. Stop after the square and you have only 4.

A **topological order** lists each node after its inputs. Visit children, then append, and skip an id you have already seen:

```python
# if id(node) not in visited: visit children, then topo.append(node)
```

One valid order is $a$, $b$, $c$, $d$. Reversed, you run $d$, $c$, $b$, $a$. Both 4 and 2 land on $a$ before it passes slope on.

The usual mistake is running a node after only one user has written a slope. Set the output slope to ones, then call `_backward_fn` on `reversed(topo)`.

> [!KEY]
> Reverse topological order gives each node its full gradient before that node passes any slope on.

## Turn scores into a loss

**Logits** are one score per class. Softmax turns them into probabilities. Cross-entropy is minus the log of the correct class. The loss is the mean over $n$ rows.

Take $z = [2, 1, 0]$, class 0. Subtract the row max 2, leaving $[0, -1, -2]$. Exponentials: $1$, $0.367879$, $0.135335$. Sum: $1.503214$. Divide: $1/1.503214 = 0.665241$, $0.367879/1.503214 = 0.244728$, and $0.135335/1.503214 = 0.090030$.

The loss is $-\log 0.665241 = 0.407606$. Subtract 1 on class 0 and the slope is $[-0.334759,\ 0.244728,\ 0.090030]$.

$$
p_{ic} = \frac{e^{z_{ic}}}{\sum_{c'} e^{z_{ic'}}}, \qquad
\text{loss} = -\frac{1}{n} \sum_{i=1}^{n} \log p_{i y_i} .
$$

Subtracting the row maximum cancels in the fraction. Do it anyway, or `np.exp` overflows. With $n = 1$ you recover loss $0.407606$:

```python
m = z.max(axis=1, keepdims=True)
log_probs = (z - m) - np.log(np.exp(z - m).sum(axis=1, keepdims=True))
loss = -log_probs[np.arange(n), labels].mean()
```

The usual mistake is `log_probs[:, labels]`, an `(n, n)` block. Use `np.arange(n)` with `labels`.

$$
\frac{\partial\, \text{loss}}{\partial z} = \frac{1}{n}\left(p - \text{onehot}(y)\right) .
$$

Divide by $n$ because the loss is a mean. With $n = 1$ the slope is $[-0.334759,\ 0.244728,\ 0.090030]$ again:

```python
g = probs.copy()
g[np.arange(n), labels] -= 1.0
g /= n
```

The usual mistake is a missing `/ n`.

> [!KEY]
> The gradient of mean softmax cross-entropy in the logits is $(p - \text{onehot}(y)) / n$.

> [!WARNING]
> `log_probs[:, labels]` is an `(n, n)` block. Index one entry per row.

## Check a slope with two nudges

Fix $b = 2$, so $c = 3a$. At $a = 3$ and $h = 0.001$ the two values are $3 \times 3.001 = 9.003$ and $3 \times 2.999 = 8.997$. The gap $0.006$ divided by $0.002$ is $3$.

That two-sided nudge is a **central difference**:

$$
\frac{\partial\, \text{loss}}{\partial w} \approx \frac{\text{loss}(w + h) - \text{loss}(w - h)}{2h} .
$$

Near $h = 10^{-5}$ you usually get about 6 digits. A tiny $h$ loses the gap to rounding. Two passes per weight make this a test:

```python
(loss(w + h) - loss(w - h)) / (2 * h)
```

The usual mistake is a one-sided nudge. Its error shrinks like $h$, not like $h^2$.

> [!RECAP]
> Keep these five facts in hand:
> - Every backward rule uses `+=`. Addition and multiplication call `_unbroadcast` first.
> - A matrix product multiplies by the other factor, transposed. ReLU masks with `self.value > 0`. Sum spreads the slope over ones.
> - `backward` visits children first, sets the output slope to ones, and runs `_backward_fn` in reverse.
> - Softmax cross-entropy subtracts the row max and sends $(p - \text{onehot}(y))/n$ to the logits.
> - On $c = ab + a$ at $a = 3$, $b = 2$, and $h = 0.001$, the central difference returns 3.

## Build it step by step

`_unbroadcast` is already written. Build these nine pieces:

1. **`Tensor.__init__`** is already written. It stores a float `value` and a zero `grad` of the same shape. The fields are as follows:

    ```python
    def __init__(self, value, children=()):
        # Float value, zero grad, children saved.
    ```

2. **`__add__`** coerces with `Tensor._coerce`, builds `out` from the sum, and adds `_unbroadcast(out.grad, shape)` into each `grad`. The usual mistake is a bias slope of shape `(n, m)`:

    ```python
    def __add__(self, other):
        # children=(self, other). += unbroadcast(out.grad).
    ```

3. **`__mul__`** gives each side `out.grad` times the other value, then unbroadcasts. The usual mistake is using a value's own entries:

    ```python
    def __mul__(self, other):
        # += unbroadcast(out.grad * other value).
    ```

4. **`__matmul__`** uses `out.grad @ other.value.T` and `self.value.T @ out.grad` on 2-dimensional tensors. The usual mistake is the wrong transpose:

    ```python
    def __matmul__(self, other):
        # No unbroadcast. Gradients match value shapes.
    ```

5. **`relu`** multiplies `out.grad` by `self.value > 0`. The usual mistake is masking with the output:

    ```python
    def relu(self):
        # One child. Mask with the input.
    ```

6. **`sum`** returns a scalar and spreads `out.grad` with `np.ones_like`. The usual mistake is returning a vector:

    ```python
    def sum(self):
        # out.grad on every entry.
    ```

7. **`backward`** builds `topo` depth-first, sets `self.grad` to ones, and calls `_backward_fn` on `reversed(topo)`. `a * a` at 3 and the diamond at 2 both expect 6. The usual mistake is `=`:

    ```python
    def backward(self):
        # Children first, then append. Ones, then reverse.
    ```

8. **`softmax_cross_entropy(logits, labels)`** returns a scalar whose only child is `logits`. Backward adds `out.grad * (probs - onehot) / n`. The usual mistake is a missing `/ n` or a missing row max:

    ```python
    def softmax_cross_entropy(logits, labels):
        # Stable log-sum-exp. One child: logits.
    ```

9. **`mlp_loss_and_grad(params, X, y)`** returns `(float(loss.value), grads)`. `W` is `(in, out)`, so the layer is `h @ W + b`. Use `.relu()` on every layer except the last, then copy both grads. The usual mistake is a ReLU on the logits:

    ```python
    def mlp_loss_and_grad(params, X, y):
        # Fresh tensors. Hidden ReLU only. Copy both grads.
    ```

## Cost and scaling

For $P$ parameters, the three routes line up as follows:

| Method | Time for all $P$ gradients | Extra memory |
|---|---|---|
| Backpropagation, reverse mode | About 2 to 3 forward passes | Every stored activation |
| Forward mode | About $P$ forward passes | Small |
| Finite differences | $2P$ forward passes | Small |

With one loss and large $P$, reverse mode wins. Memory grows with depth times batch size. **Checkpointing** stores some activations and recomputes the rest.

## Where this breaks

Five failures show up in practice:

- `=` drops every contribution except the last. Use `+=`.
- A `grad` keeps growing across calls. This exercise builds fresh tensors, so each `grad` starts at zero.
- ReLU has no slope at 0. Engines pick 0. A finite difference on that kink disagrees.
- Large logits make `np.exp` overflow. Subtract the row maximum first.
- Large batches store too many activations. Use smaller batches, or checkpoint.

## The exercise

You implement `Tensor.__add__`, `Tensor.__mul__`, `Tensor.__matmul__`, `Tensor.relu`, `Tensor.sum`, `Tensor.backward`, `softmax_cross_entropy`, and `mlp_loss_and_grad`. `Tensor.__init__` and `_unbroadcast` are already written.

Checks: a tensor starts with a zero gradient of the same shape. Each operation matches finite differences, including a broadcast bias. `a * a` at 3 gives 6. The diamond at 2 gives 6. The softmax gradient equals $(p - \text{onehot}(y))/n$. `mlp_loss_and_grad` matches finite differences on a small two-layer net.
