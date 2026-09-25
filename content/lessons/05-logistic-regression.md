---
title: Logistic regression
slug: 05-logistic-regression
part: "Part 1 — Foundations"
summary: Turn a linear score into a probability, then fit it with gradient descent and with Newton's method.
minutes: 60
gpu: false
objectives:
  - Compute the sigmoid and the logistic loss without overflow for scores as large as 800.
  - Build the gradient and the Hessian of the loss in numpy, and check them against finite differences.
  - Fit the model with gradient descent and with Newton's method, and compare how many steps each needs.
  - Explain why perfectly separable data makes the weights grow without limit, and how a ridge penalty stops it.
prereqs:
  - 02-linear-regression
  - 03-gradient-descent
exercise: 05-logistic-regression
---

# Logistic regression

Many questions have a yes-or-no answer. Is this email spam? Will this
patient return to hospital? Linear regression predicts a number on a
sliding scale, and nothing stops that number from being 7 or -3. A yes-or-no
question needs a probability between 0 and 1 instead.

Logistic regression keeps the linear score from lesson 2 and passes it
through a squashing function that turns any number into a probability. It
then picks the weights that make the observed labels as likely as possible.
The result is one of the most used models in practice, and the building
block for every classifier later in the course.

In this lesson you build the whole thing: a probability function that never
overflows, a loss, its first and second derivatives, and two ways to
minimise the loss. You also see one surprising way the method fails.

## What you already know

- A linear model computes a score for each example as $Xw$, one number per row of $X$.
- Gradient descent from lesson 3 walks downhill on a loss by stepping against the gradient.
- The gradient of a function is the vector of its partial derivatives, one per weight.
- A ridge penalty from lesson 4 adds a cost for large weights and keeps a solve well behaved.
- Computers store numbers with limited range, so `np.exp(800)` returns `inf`.

## From a score to a probability

Think of a light dimmer whose knob can turn forever in both directions.
Every knob position maps to a brightness between fully off and fully on. Far to the left is almost off, far to the right is almost on, and the
middle is half bright. The *sigmoid* function does exactly that.

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Read it aloud: take the score $z$, flip its sign, raise $e$ to that power,
add one, and take one over the result. When $z$ is large and positive,
$e^{-z}$ is tiny, so $\sigma(z)$ is close to 1. When $z$ is large and
negative, $e^{-z}$ is huge, so $\sigma(z)$ is close to 0. At $z = 0$ you get
exactly $0.5$.

In numpy, the direct version is one line:

```python
p = 1.0 / (1.0 + np.exp(-z))
```

The model uses the sigmoid on each example's score. Here $x_i$ is row $i$ of
$X$, a vector of $d$ features, and $w$ is the weight vector of length $d$.
The score is $z_i = x_i^{\top} w$, the dot product of the two. The predicted
probability that the label is 1 is $p_i = \sigma(z_i)$. This lesson has no
separate bias $b$. To get one, add a column of ones to $X$, and its weight
plays the role of $b$.

## The loss

The loss scores how badly the probabilities match the labels. Each label
$y_i$ is 0 or 1. A good model gives high probability to the label that
happened. The *log loss*, also called the negative log-likelihood, punishes
confident wrong answers hard and confident right answers barely at all.

The exercise uses this form of it, summed over all $n$ examples:

$$
L(w) = \sum_{i=1}^{n} \Bigl[ \operatorname{softplus}(z_i) - y_i z_i \Bigr]
+ \frac{\lambda}{2} \lVert w \rVert^2,
\qquad \operatorname{softplus}(z) = \log(1 + e^{z})
$$

Read it aloud piece by piece:

- $z_i = x_i^{\top} w$ is the score for example $i$.
- $\operatorname{softplus}(z_i)$ is a smooth version of "the score, but never below zero".
- Subtracting $y_i z_i$ rewards a large score when the label is 1.
- The sum adds up one term per example. It is a sum, not an average.
- $\frac{\lambda}{2} \lVert w \rVert^2$ is the ridge penalty. $\lVert w \rVert^2$ is the squared length of $w$, the sum of its squared entries. $\lambda$, called `ridge` in the code, sets how strong the penalty is. With $\lambda = 0$ the penalty disappears.

This form is the usual cross-entropy loss, rewritten so it needs only the
score and not the probability. In numpy, ignoring overflow for a moment:

```python
z = X @ w
loss = np.sum(np.log(1 + np.exp(z)) - y * z) + 0.5 * ridge * np.dot(w, w)
```

At $w = 0$ every score is 0 and every term is $\log 2 \approx 0.693$,
whatever the label. That gives you a quick test: ten examples at $w = 0$
must give a loss of $10 \log 2$.

## Keep the numbers finite

A confident model produces large scores. When $z = 800$, `np.exp(800)` is
`inf`, and `np.log(1 + inf)` is `inf`. The loss becomes useless exactly when
the model is doing well.

The fix is a rewrite that gives the same value but only ever raises $e$ to a
number that is zero or negative:

$$
\operatorname{softplus}(z) = \max(z, 0) + \log\bigl(1 + e^{-|z|}\bigr)
$$

Read it aloud: take the positive part of $z$, then add a small correction
that uses $e$ raised to minus the size of $z$. Since $-|z| \le 0$, the
exponential lies between 0 and 1 and can never overflow. This trick is
called *log-sum-exp*. In numpy:

```python
softplus = np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
```

`np.log1p(a)` computes $\log(1 + a)$ accurately when $a$ is tiny.

The sigmoid needs the same care. For $z \ge 0$, use $1 / (1 + e^{-z})$. For
$z < 0$, use the equal expression $e^{z} / (1 + e^{z})$. Each branch only
exponentiates a number that is zero or negative. In numpy, you build a
boolean mask such as `pos = z >= 0` and fill the two parts of the output
separately.

## The gradient and the Hessian

The *gradient* tells you, for each weight, how fast the loss changes when
you nudge that weight. Its formula looks like the linear regression one,
with the probability in place of the prediction:

$$
\nabla L(w) = X^{\top}\bigl(\sigma(Xw) - y\bigr) + \lambda w
$$

Read it aloud: compute each example's probability, subtract its label to get
an error between -1 and 1, and let $X^{\top}$ add up those errors, weighted
by each feature. Then add $\lambda w$ from the penalty. The result has one
entry per weight, so its shape is `(d,)`.

```python
p = sigmoid(X @ w)
grad = X.T @ (p - y) + ridge * w
```

The *Hessian* is the matrix of second derivatives. It says how the gradient
itself changes, so it measures how curved the loss is in each direction:

$$
\nabla^2 L(w) = X^{\top} D X + \lambda I, \qquad
D = \operatorname{diag}\bigl(p_1(1-p_1), \ldots, p_n(1-p_n)\bigr)
$$

Read it aloud: give each example a weight $p_i(1 - p_i)$, which is largest
(0.25) when the model is unsure and near 0 when it is confident. Scale each
row of $X$ by that weight, multiply by $X^{\top}$, and add $\lambda$ to each
diagonal entry. $D$ is a diagonal matrix, which means all its entries off
the diagonal are zero. $I$ is the identity matrix of size $d$. The result has
shape `(d, d)`.

You never need to build the $n$-by-$n$ matrix $D$. Scale the rows instead:

```python
s = p * (1 - p)
H = (X * s[:, None]).T @ X + ridge * np.eye(X.shape[1])
```

Every weight $p_i(1 - p_i)$ is zero or positive, so the Hessian is
*positive semidefinite*. In plain words, the loss curves upward or stays flat
in every direction, never downward. That makes the loss a bowl with no false
bottoms: any point where the gradient is zero is the lowest point.

## Newton's method

Gradient descent takes small fixed steps downhill. Newton's method looks at
the curvature too. At each point, it fits a bowl-shaped quadratic to the loss
and jumps straight to that bowl's bottom:

$$
w \leftarrow w - H^{-1} g, \qquad g = \nabla L(w), \quad H = \nabla^2 L(w)
$$

Read it aloud: find the step $\delta$ that solves $H \delta = g$, then
subtract it from $w$. Where the loss is steep but not curved, the step is
long. Where it is sharply curved, the step is short. In numpy, solve the
system instead of inverting $H$:

```python
delta = np.linalg.solve(H, g)
w = w - delta
```

Near the answer, each Newton step roughly doubles the number of correct
digits. On the exercise's problem, Newton finishes in under ten steps where
fixed-step gradient descent needs hundreds. The price is a $d$-by-$d$ solve
at every step.

## Work one example by hand

Take two examples with two features each. The first column is all ones, so
its weight acts as a bias:

$$
X = \begin{bmatrix} 1 & 0 \\ 1 & 2 \end{bmatrix}, \quad
y = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad
w = \begin{bmatrix} 0.5 \\ -1 \end{bmatrix}, \quad \lambda = 0
$$

The following table shows every intermediate value, rounded to four decimal
places:

| Quantity | Example 1 | Example 2 |
|---|---|---|
| Score $z_i = x_i^{\top} w$ | $0.5$ | $0.5 - 2 = -1.5$ |
| Probability $p_i = \sigma(z_i)$ | $0.6225$ | $0.1824$ |
| $\operatorname{softplus}(z_i)$ | $0.9741$ | $0.2014$ |
| Loss term $\operatorname{softplus}(z_i) - y_i z_i$ | $0.4741$ | $0.2014$ |
| Error $p_i - y_i$ | $-0.3775$ | $0.1824$ |
| Curvature weight $p_i(1 - p_i)$ | $0.2350$ | $0.1491$ |

The loss is $0.4741 + 0.2014 = 0.6755$.

The gradient is $X^{\top}(p - y)$. Its first entry adds the errors:
$-0.3775 + 0.1824 = -0.1951$. Its second entry weights them by the second
feature: $0 \times (-0.3775) + 2 \times 0.1824 = 0.3649$. So
$g = [-0.1951, 0.3649]$.

The Hessian is $X^{\top} D X$:

$$
H = \begin{bmatrix} 0.2350 + 0.1491 & 2 \times 0.1491 \\
2 \times 0.1491 & 4 \times 0.1491 \end{bmatrix}
= \begin{bmatrix} 0.3842 & 0.2983 \\ 0.2983 & 0.5966 \end{bmatrix}
$$

One Newton step solves $H \delta = g$ and gives
$w \approx [2.1065, -2.4148]$. Run your functions on these numbers and
compare.

## When the classes separate perfectly

Suppose one straight line puts every positive example on one side and every
negative one on the other. That data is *linearly separable*. You might
expect this to be the easy case. It is the broken one.

On separable data, doubling the weights keeps every example on the right
side and makes every probability more confident. Each loss term shrinks
toward zero. Doubling again helps again. The loss approaches zero but never
reaches it, because $\sigma$ never outputs exactly 0 or 1. So the best
weights do not exist: gradient descent keeps making $w$ longer forever, and
the loss keeps creeping down.

The ridge penalty fixes this. Its cost $\frac{\lambda}{2} \lVert w \rVert^2$
grows as $w$ grows, so at some length the penalty outweighs the tiny gain
from more confidence. The weights settle at a finite size. The penalty also
adds $\lambda$ to the Hessian's diagonal, which keeps the Newton solve well
posed. That is why `fit_newton` uses a tiny default ridge of `1e-8`.

## Build it step by step

Build the functions in this order. Each one uses the ones before it.

1. **`sigmoid(z)`**. Takes an array of any shape and returns an array of the
   same shape. Make a float copy, build a mask `z >= 0`, and fill the two
   parts with the two formulas from "Keep the numbers finite". Use
   `np.empty_like` for the output. Common mistake: computing `np.exp(-z)` on
   the whole array first, which overflows for $z = -800$ even if you discard
   that part afterwards.
2. **`log_loss(w, X, y, ridge)`**. Takes `w` of shape `(d,)`, `X` of shape
   `(n, d)`, `y` of shape `(n,)`, and returns one float. Compute `z = X @ w`,
   the stable softplus, `np.sum(softplus - y * z)`, and add
   `0.5 * ridge * np.dot(w, w)`. Common mistake: using the mean instead of
   the sum, or forgetting the factor of one half on the penalty.
3. **`gradient(w, X, y, ridge)`**. Returns shape `(d,)`. Use your `sigmoid`,
   then `X.T @ (p - y) + ridge * w`. Common mistake: writing `y - p`, which
   flips the sign and sends gradient descent uphill.
4. **`hessian(w, X, y, ridge)`**. Returns shape `(d, d)`. Compute
   `s = p * (1 - p)`, scale the rows with `X * s[:, None]`, multiply by `X`,
   and add `ridge * np.eye(d)`. Common mistake: `X * s` without `[:, None]`,
   which tries to scale columns and fails or broadcasts wrongly.
5. **`fit_gradient_descent(X, y, lr, max_iter, ridge, tol)`**. Start from
   `w = np.zeros(d)`. In each loop pass, append the current loss to
   `history`, compute the gradient, stop if `np.linalg.norm(g) < tol`, and
   otherwise update `w = w - lr * g`. Return `(w, history)`. Common mistake:
   recording the loss after the update, so `history[0]` is not the loss at
   $w = 0$.
6. **`fit_newton(X, y, max_iter, ridge, tol)`**. The same loop, but the
   update solves `np.linalg.solve(H, g)` and subtracts the result. Common
   mistake: calling `np.linalg.inv`, which is slower and less accurate than
   `solve`.

## Cost and scaling

The cost depends on $n$, the number of examples, and $d$, the number of
features. The following table shows how each piece grows:

| Piece | Work grows like | In plain words |
|---|---|---|
| Loss or gradient | $n \times d$ | One pass over the data. |
| Hessian | $n \times d^2$ | $d$ times the gradient's work. |
| Newton solve | $d^3$ | Small when $d$ is in the hundreds, heavy in the tens of thousands. |
| Memory for the Hessian | $d^2$ | A 10,000-feature model needs 100 million numbers. |

Newton's method wins when $d$ is small, because a handful of expensive steps
beats hundreds of cheap ones. When $d$ is large, gradient descent and its
relatives are the only practical choice.

## Where this breaks

**Separable or nearly separable data.** The weights grow every step while
the validation score stops improving. Add a ridge penalty or stop training
early.

**One class is rare.** If 99% of labels are 0, most curvature weights
$p_i(1 - p_i)$ are tiny, and the Newton solve becomes fragile. Weight each
example's loss term by the inverse of its class frequency.

**Overflow that hides.** A naive loss returns `inf` or `nan` without an
error. A training loop that never checks `np.isfinite` reports a broken
model as one that failed to learn. Use the stable formulas.

**Too many features for Newton.** When $d$ is close to $n$ or larger,
building and solving the $d$-by-$d$ Hessian costs more than it saves. Use
gradient descent or a quasi-Newton method such as L-BFGS, which approximates
the curvature without storing the full Hessian.

## The exercise

You implement `sigmoid`, `log_loss`, `gradient`, `hessian`,
`fit_gradient_descent`, and `fit_newton`. Both fitting functions start from
$w = 0$ and return the fitted weights with a list of the loss recorded before
every step.

The checks confirm that `sigmoid` and `log_loss` stay finite and correct at
scores of $\pm 800$, that the loss equals $n \log 2$ at $w = 0$, and that the
ridge term adds exactly $\frac{\lambda}{2} \lVert w \rVert^2$. They compare
`gradient` and `hessian` with finite differences and confirm the Hessian is
symmetric and positive semidefinite. They confirm `fit_newton` reaches the
same answer as `fit_gradient_descent` in under ten steps where gradient
descent needs hundreds, and that it ends at a point with a zero gradient.
The last check builds separable data and confirms the unregularised weights
keep growing while the ridge-penalised ones stay small.
