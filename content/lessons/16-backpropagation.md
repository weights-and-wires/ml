---
title: Backpropagation as reverse-mode AD
slug: 16-backpropagation
part: "Part 3 — Neural networks"
summary: Build a tiny autodiff engine that records a computation and runs the chain rule backwards to get every gradient in one pass.
minutes: 70
gpu: false
objectives:
  - Explain how backpropagation applies the chain rule from the loss back to every input.
  - Build the backward rule for addition, multiplication, matrix products, ReLU, and sum.
  - Compute the softmax-cross-entropy loss and its gradient with respect to the logits.
  - Build a topological sort so every gradient is complete before it flows further.
  - Check any gradient against a finite-difference estimate.
prereqs:
  - 15-multilayer-perceptrons
exercise: 16-backpropagation
---

# Backpropagation as reverse-mode AD

To train a network, you need the gradient of the loss with respect to every
weight. The **gradient** is the list of partial derivatives: for each weight,
how much the loss changes when you nudge that weight a little. In earlier
lessons you worked out gradients by hand, one model at a time. That does not
scale to networks with millions of weights.

**Backpropagation** does the work for you. During the forward pass, it records
every operation, like writing each step of a recipe on a card. Then it walks
the cards backwards, applying the chain rule at each step. One backward pass
gives the gradient for every weight at once, for about the cost of one more
forward pass.

This is called **reverse-mode automatic differentiation** (AD). In this lesson
you build a small version of the engine inside PyTorch: a `Tensor` class that
records operations and runs them in reverse.

## What you already know

- The chain rule: if $y = f(u)$ and $u = g(x)$, then $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$.
- A forward pass through an MLP is a chain of matrix products, bias adds, and ReLUs.
- Softmax turns a row of scores into probabilities that add up to 1.
- A finite difference, $(f(x+h) - f(x-h)) / 2h$, estimates a derivative.
- Numpy broadcasting stretches a `(m,)` bias across every row of an `(n, m)` matrix.

## The graph and the tape

Every value in the computation becomes a node. A node remembers three things:

- `value`: the number or array it holds.
- `grad`: the gradient of the final loss with respect to this node. It starts as zeros of the same shape.
- `_children`: the nodes it was computed from, plus a small function `_backward_fn` that knows how to pass gradient back to them.

Take $c = a \times b + a$ with $a = 3$ and $b = 2$. The forward pass builds two
nodes: $p = a \times b = 6$ and $c = p + a = 9$.

The backward pass starts at the output. The gradient of $c$ with respect to
itself is 1. Then each node hands gradient to its children:

- $c = p + a$: addition passes the gradient through unchanged. So $p$ gets 1 and $a$ gets 1.
- $p = a \times b$: multiplication gives each input the gradient times the *other* input. So $a$ gets $1 \times 2 = 2$ and $b$ gets $1 \times 3 = 3$.
- $a$ was used twice, so its contributions add: $1 + 2 = 3$.

You can check this by hand: $c = ab + a$, so $\frac{\partial c}{\partial a} = b + 1 = 3$ and $\frac{\partial c}{\partial b} = a = 3$.

That last step is the most important rule in the engine: **gradients
accumulate**. Every backward function uses `+=`, never `=`.

## The backward rule for each operation

Each operation needs one backward rule. The rule takes the gradient arriving
at the output, written $\bar{z}$ ("z-bar"), and turns it into gradients for the
inputs. This is called a **vector-Jacobian product**. The **Jacobian** is the
full table of partial derivatives of the output with respect to the input. You
never build that table. Each rule computes its product with $\bar{z}$ directly.

### Addition and multiplication

For $z = x + y$ and $z = x \odot y$, where $\odot$ means elementwise
multiplication:

$$
z = x + y:\quad \bar{x} \mathrel{+}= \bar{z},\ \ \bar{y} \mathrel{+}= \bar{z}
\qquad\qquad
z = x \odot y:\quad \bar{x} \mathrel{+}= \bar{z} \odot y,\ \ \bar{y} \mathrel{+}= \bar{z} \odot x
$$

Read it aloud: addition copies the incoming gradient to both inputs.
Multiplication gives each input the incoming gradient times the other input.

```python
self.grad += _unbroadcast(out.grad, self.value.shape)                  # add
self.grad += _unbroadcast(out.grad * other.value, self.value.shape)    # mul
```

**Broadcasting needs one extra step.** When you add a bias `b` of shape `(m,)`
to `X` of shape `(n, m)`, numpy copies `b` into all $n$ rows. Each copy
contributes to the loss, so the bias gradient is the sum of the incoming
gradient over those rows. The provided `_unbroadcast(grad, shape)` does that
summing for you. Call it in `__add__` and `__mul__` before every `+=`.

### Matrix product

For $Z = A B$, with $A$ of shape `(n, k)` and $B$ of shape `(k, m)`:

$$
\bar{A} \mathrel{+}= \bar{Z} B^\top, \qquad \bar{B} \mathrel{+}= A^\top \bar{Z}
$$

Read it aloud: the gradient for $A$ is the incoming gradient times $B$
transposed. The gradient for $B$ is $A$ transposed times the incoming gradient.
A quick way to remember: the shapes only fit one way. $\bar{Z}$ is `(n, m)`,
and you need `(n, k)` for $\bar{A}$, so multiply by $B^\top$, which is `(m, k)`.

```python
self.grad += out.grad @ other.value.T
other.grad += self.value.T @ out.grad
```

### ReLU and sum

ReLU lets gradient through where its input was positive and blocks it
elsewhere. Sum turns an array into one number, and every entry contributed
with weight 1, so each entry gets the incoming gradient.

$$
z = \operatorname{relu}(x):\quad \bar{x} \mathrel{+}= \bar{z} \odot [x > 0]
\qquad\qquad
z = \textstyle\sum_i x_i:\quad \bar{x} \mathrel{+}= \bar{z} \cdot \mathbf{1}
$$

Read it aloud: $[x > 0]$ is 1 where $x$ is positive and 0 elsewhere.
$\mathbf{1}$ is an array of ones with the shape of $x$.

```python
self.grad += out.grad * (self.value > 0)          # relu
self.grad += out.grad * np.ones_like(self.value)  # sum
```

## The order of the backward pass

A node may be used by several later nodes. You must collect all of its
gradient before it passes gradient on to its own children. Otherwise it
forwards an incomplete number.

A **topological order** solves this. It lists nodes so that every node comes
after all the nodes it was built from. Walk that list in reverse and every
node is finished before you reach its children.

You build the order with a **depth-first search**: visit a node's children
first, then append the node. Keep a set of visited nodes so a shared node is
added only once.

```python
def build(node):
    if id(node) not in visited:
        visited.add(id(node))
        for child in node._children:
            build(child)
        topo.append(node)
```

After building `topo`, set the output's `grad` to ones and call each node's
`_backward_fn` in `reversed(topo)`.

### The diamond

Take $b = a \times a$, $c = a + a$, and $d = b + c$, with $a = 2$. The node
$a$ feeds both $b$ and $c$, and both feed $d$. By hand,
$d = a^2 + 2a$, so $\frac{dd}{da} = 2a + 2 = 6$. With the right order, $d$
passes 1 to both $b$ and $c$. Then $b$ passes $1 \times 2 + 1 \times 2 = 4$
to $a$, and $c$ passes $1 + 1 = 2$, for a total of 6. The checks use this exact graph.

## Softmax cross-entropy

A classifier outputs one score per class, called **logits**. Softmax turns
logits $z$ into probabilities $p$. Cross-entropy is the loss: minus the log of
the probability given to the correct class. For $n$ examples with labels
$y_i$, the loss is the mean:

$$
p_{ic} = \frac{e^{z_{ic}}}{\sum_{c'} e^{z_{ic'}}}, \qquad
\text{loss} = -\frac{1}{n} \sum_{i=1}^{n} \log p_{i y_i}
$$

Read it aloud: for row $i$, exponentiate every score and divide by the row
total to get probabilities. Take the probability of the correct class, take
its log, flip the sign, and average over the $n$ rows.

To avoid overflow, subtract each row's largest score first. It cancels in the
fraction, so the answer is unchanged:

```python
m = z.max(axis=1, keepdims=True)
log_probs = (z - m) - np.log(np.exp(z - m).sum(axis=1, keepdims=True))
loss = -log_probs[np.arange(n), labels].mean()
```

The gradient with respect to the logits is short and worth remembering:

$$
\frac{\partial\, \text{loss}}{\partial z} = \frac{1}{n}\left(p - \text{onehot}(y)\right)
$$

Read it aloud: take the probability matrix, subtract 1 at each row's correct
class, and divide by $n$. The $\frac{1}{n}$ is there because the loss is a
mean, not a sum. `onehot(y)` is a matrix with a 1 at the correct class of each
row and 0 elsewhere.

```python
g = probs.copy()
g[np.arange(n), labels] -= 1.0
g /= n
```

### Worked example by hand

One example, logits $z = [2, 1, 0]$, correct class 0.

- Exponentials: $e^2 = 7.389$, $e^1 = 2.718$, $e^0 = 1$. The total is $11.107$.
- Probabilities: $p = [0.6652, 0.2447, 0.0900]$.
- Loss: $-\log 0.6652 = 0.4076$.
- Gradient: $[0.6652 - 1, 0.2447, 0.0900] / 1 = [-0.3348, 0.2447, 0.0900]$.

The gradient pushes the correct score up (negative gradient) and the others
down.

## Checking gradients with finite differences

To check any gradient, nudge one input by a small $h$ both ways and measure
the change in the loss:

$$
\frac{\partial\, \text{loss}}{\partial w} \approx \frac{\text{loss}(w + h) - \text{loss}(w - h)}{2h}
$$

Read it aloud: evaluate the loss slightly above and slightly below $w$,
subtract, and divide by the gap $2h$. With $h$ around $10^{-5}$, the estimate
is usually correct to 6 or more digits. Much smaller $h$ makes it worse,
because rounding errors in the subtraction take over. This costs two forward
passes per weight, so it is a test, not a training method.

## Build it step by step

Build the pieces in the order the checks test them.

1. **`Tensor.__init__`.** Already written. It stores `value` as a float array and `grad` as zeros of the same shape. Read it so you know the fields.
2. **`__add__(self, other)`.** Coerce `other` with `Tensor._coerce`. Build `out = Tensor(self.value + other.value, children=(self, other))`. Define an inner `_backward()` that adds `_unbroadcast(out.grad, ...)` to each operand's `grad`, and assign it to `out._backward_fn`. Return `out`. Common mistake: skipping `_unbroadcast`, so a `(m,)` bias gets a `(n, m)` gradient.
3. **`__mul__(self, other)`.** Same pattern. Each operand gets `out.grad` times the other operand's `value`, unbroadcast. Common mistake: multiplying by the operand's own value.
4. **`__matmul__(self, other)`.** Both are 2-D. Use `out.grad @ other.value.T` and `self.value.T @ out.grad`. No unbroadcast is needed. Common mistake: putting the transpose on the wrong side. Check the shapes.
5. **`relu(self)`.** One child. Multiply `out.grad` by the mask `self.value > 0`. Common mistake: masking with the output instead of the input. For ReLU both give the same mask, but reading the input is the habit to keep.
6. **`sum(self)`.** Returns a scalar `Tensor`. Backward spreads `out.grad` over `np.ones_like(self.value)`.
7. **`backward(self)`.** Build `topo` with the depth-first search above, set `self.grad = np.ones_like(self.value)`, and call `_backward_fn` for each node in `reversed(topo)`. Common mistake: using `=` instead of `+=` in any backward function. The checks for `a * a` and the diamond catch it.
8. **`softmax_cross_entropy(logits, labels)`.** Compute the loss in numpy with the stable log-sum-exp. Return a scalar `Tensor` with `children=(logits,)`. Its backward adds `out.grad * (probs - onehot) / n` to `logits.grad`. Common mistake: forgetting to divide by `n`, or computing `np.exp(z)` without subtracting the row maximum.
9. **`mlp_loss_and_grad(params, X, y)`.** Wrap each `W` and `b` in a `Tensor`. Here `W` has shape `(in, out)`, so each layer is `h @ W + b`, the transpose of the lesson 15 convention. Apply `.relu()` after every layer except the last. Pass the last output to `softmax_cross_entropy`, call `.backward()` once, and return `(float(loss.value), grads)`, where `grads` copies each `W.grad` and `b.grad`. Common mistake: applying ReLU to the logits.

## Cost and scaling

Reverse mode is cheap in time and hungry for memory. The table compares the
ways to get a gradient for a network with $P$ parameters.

| Method | Time for all $P$ gradients | Extra memory |
|---|---|---|
| Backpropagation (reverse mode) | about 2 to 3 forward passes | every stored activation |
| Forward mode | about $P$ forward passes | small |
| Finite differences | $2P$ forward passes | small |

Forward mode pushes a nudge from one input forward to the output, so you need
one pass per input. Reverse mode pulls the gradient from the single loss back
to every input at once. With one loss and millions of weights, reverse mode
wins by a huge margin.

The price is memory. The backward pass needs the forward values, such as `A`
and `B` in a matrix product, so you keep every layer's activations until the
backward pass uses them. Memory grows with depth times batch size.
**Checkpointing** trades time for memory: it stores only some activations and
recomputes the rest during the backward pass.

## Where this breaks

**Overwriting instead of accumulating.** Using `=` in a backward function gives
wrong gradients whenever a value is used twice. Always use `+=`.

**Forgetting to zero gradients.** A `grad` keeps growing across calls to
`backward`. Real libraries make you reset it before each step. In this
exercise you build fresh `Tensor` objects on every call, so each one starts at
zero.

**Kinks.** ReLU has no derivative at exactly 0. Engines pick 0, which is fine in
practice, but a finite-difference check that lands on a kink disagrees.

**Overflow in softmax.** Large logits make `np.exp` return infinity. Subtract
the row maximum first.

**Running out of memory.** Deep networks with large batches store too many
activations. Use smaller batches or checkpointing.

## The exercise

You implement `Tensor.__add__`, `Tensor.__mul__`, `Tensor.__matmul__`,
`Tensor.relu`, `Tensor.sum`, `Tensor.backward`, `softmax_cross_entropy`, and
`mlp_loss_and_grad` in `starter.py`. The helper `_unbroadcast` is provided.

The checks confirm that a `Tensor` starts with a zero gradient and that each
operation's gradient matches finite differences, including a broadcast bias.
They test that `a * a` gives 6 at $a = 3$, that the diamond gives 6 at
$a = 2$, and that the softmax-cross-entropy gradient equals
$(p - \text{onehot}(y)) / n$. The last check compares `mlp_loss_and_grad` with
finite differences on a small two-layer network.
