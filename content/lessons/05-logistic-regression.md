---
title: Logistic regression
slug: 05-logistic-regression
part: "Part 1 — Foundations"
summary: The Bernoulli log-likelihood, its gradient and Hessian, Newton's method as reweighted least squares, and what separable data does to all of it.
minutes: 55
gpu: false
objectives:
  - Derive the negative log-likelihood, its gradient X^T(sigma(Xw) - y), and its Hessian X^T D X.
  - Evaluate the loss through log-sum-exp and explain why the naive formula overflows.
  - Identify Newton's method with iteratively reweighted least squares and state its local convergence rate.
  - Explain why linearly separable data sends the maximum likelihood estimate to infinity, and how a ridge penalty fixes it.
prereqs:
  - 02-linear-regression
  - 03-gradient-descent
exercise: 05-logistic-regression
---

# Logistic regression

Linear regression assumes the target is a noisy linear function of the
features. That assumption is meaningless for a label in $\{0, 1\}$: nothing
about $Xw$ is confined to an interval, let alone to two points. Logistic
regression keeps the linear predictor $Xw$ but routes it through a link
function before it has to mean a probability, and fits it by maximum
likelihood instead of least squares.

The resulting loss is convex, so gradient descent from lesson 3 already
solves it. What this lesson adds is Newton's method, which uses the same loss
surface to take far fewer steps, and a numerical hazard that the squared loss
never had: the log-likelihood involves $\log(1 + e^z)$, and evaluated the
obvious way, it overflows on exactly the confident predictions a well-trained
model produces. The lesson closes on a failure mode with no analogue in
lesson 2 either: on data a linear boundary separates exactly, the maximum
likelihood estimate does not exist.

## What you already know

- A convex function's Hessian is positive semidefinite everywhere, and the
  function is strictly convex wherever the Hessian is positive definite.
- Gradient descent's fixed-step rate is governed by the Hessian's condition
  number; Newton's method already appears in lesson 3 as the update that
  eliminates that dependence on a quadratic.
- $X^{\top}X \succeq 0$ for any $X$, and it is positive definite exactly when
  $X$ has full column rank.
- Floating-point arithmetic overflows to infinity above roughly $1.8 \times
  10^{308}$ in double precision, and underflows silently to zero below
  roughly $10^{-308}$.
- The chain rule applied to a scalar function of $Xw$ gives a gradient of the
  form $X^{\top} \, (\text{something evaluated at each row})$.

## The model and its likelihood

Take $X \in \mathbb{R}^{n \times d}$ with rows $x_i^{\top}$ and labels
$y \in \{0, 1\}^n$. Logistic regression models

$$
p(y_i = 1 \mid x_i; w) = \sigma(x_i^{\top} w), \qquad
\sigma(z) = \frac{1}{1 + e^{-z}} ,
$$

so the log-odds of the positive class are exactly the linear predictor:
$\log \dfrac{p}{1-p} = x_i^{\top} w$. This is the **logit link**, and it is
the reason the model is linear in $w$ despite $p$ being bounded. Each label is
a Bernoulli draw with that success probability, and the labels are
conditionally independent given $X$, so the likelihood factorises:

$$
p(y \mid X; w) = \prod_{i=1}^{n} \sigma(x_i^{\top} w)^{y_i} \,
\bigl(1 - \sigma(x_i^{\top} w)\bigr)^{1 - y_i} .
$$

## The negative log-likelihood

Write $z_i = x_i^{\top} w$ and take the negative log of the likelihood. Using
$1 - \sigma(z) = \sigma(-z)$ and $\log \sigma(z) = -\log(1 + e^{-z})$, the
$i$-th term is

$$
- y_i \log \sigma(z_i) - (1 - y_i) \log\bigl(1 - \sigma(z_i)\bigr)
= y_i \log(1 + e^{-z_i}) + (1 - y_i) \log(1 + e^{z_i}) .
$$

Both cases collapse into one expression. Check it against $y_i = 1$: then
$\log(1 + e^{z_i}) - z_i = \log\bigl((1 + e^{z_i})/e^{z_i}\bigr) =
\log(1 + e^{-z_i})$, which is the $y_i = 1$ branch above. So

$$
L(w) = \sum_{i=1}^{n} \Bigl[ \log\bigl(1 + e^{z_i}\bigr) - y_i z_i \Bigr],
\qquad z = Xw .
$$

This is the empirical risk under the **logistic loss**. Every term is
$\operatorname{softplus}(z_i) - y_i z_i$, where
$\operatorname{softplus}(z) = \log(1 + e^z)$ is a smooth approximation to
$\max(0, z)$ — its derivative is $\sigma(z)$, which is exactly what makes the
gradient below come out clean.

## Numerically stable evaluation

$\operatorname{softplus}$ computed as `log(1 + exp(z))` overflows for
$z \gtrsim 709$: `exp(z)` alone exceeds the largest representable double, so
the sum and the log both receive `inf` where the true value is a perfectly
ordinary number near $z$. This is not a corner case; a confidently correct
prediction produces exactly this $z$, so a model that is working overflows a
naive implementation while a broken one does not.

The fix is the log-sum-exp identity applied to the two terms $\{0, z\}$:

$$
\log(1 + e^z) = \log\bigl(e^0 + e^z\bigr)
= m + \log\bigl(e^{-m} + e^{z - m}\bigr), \qquad m = \max(0, z) .
$$

Subtracting $m$ before exponentiating leaves both arguments to $\exp$ at or
below zero, so the exponentials can only underflow to $0$, never overflow.
Expanded for the two cases,

$$
\operatorname{softplus}(z) = \max(z, 0) + \log\!\bigl(1 + e^{-|z|}\bigr) .
$$

The same branch-on-sign trick stabilises $\sigma$ itself: for $z \geq 0$
compute $1/(1 + e^{-z})$, and for $z < 0$ compute $e^{z}/(1 + e^{z})$. Either
branch only ever exponentiates a nonpositive number.

## The gradient

Differentiate $L$ term by term. Since $\operatorname{softplus}'(z) =
\sigma(z)$,

$$
\frac{\partial}{\partial z_i} \Bigl[ \operatorname{softplus}(z_i) - y_i z_i
\Bigr] = \sigma(z_i) - y_i .
$$

Each $z_i$ depends on $w$ through $z_i = x_i^{\top} w$, so
$\partial z_i / \partial w = x_i$, and the chain rule sums the contributions:

$$
\nabla L(w) = \sum_{i=1}^{n} \bigl(\sigma(z_i) - y_i\bigr) x_i
= X^{\top} \bigl( \sigma(Xw) - y \bigr) .
$$

The gradient has the same shape as the normal equations' residual term in
lesson 2, with the residual $y - Xw$ replaced by $y - \sigma(Xw)$: logistic
regression is least squares with the linear prediction swapped for the
probability it induces.

## The Hessian and convexity

Differentiate the gradient's $i$-th contribution again. Since
$\sigma'(z) = \sigma(z)(1 - \sigma(z))$,

$$
\frac{\partial}{\partial z_i} \bigl[\sigma(z_i) - y_i\bigr] = \sigma(z_i)
\bigl(1 - \sigma(z_i)\bigr) =: p_i(1 - p_i) ,
$$

and summing the outer products $x_i x_i^{\top}$ weighted by this scalar gives

$$
\nabla^2 L(w) = \sum_{i=1}^{n} p_i(1-p_i) \, x_i x_i^{\top} = X^{\top} D X,
\qquad D = \operatorname{diag}\bigl(p_1(1-p_1), \dots, p_n(1-p_n)\bigr) .
$$

Every diagonal entry of $D$ lies in $[0, 1/4]$, so $D \succeq 0$, and for any
$v \in \mathbb{R}^d$,

$$
v^{\top} X^{\top} D X v = \sum_{i=1}^{n} p_i(1-p_i) (x_i^{\top} v)^2 \geq 0 .
$$

The Hessian is positive semidefinite everywhere, so $L$ is convex and every
stationary point of the gradient is a global minimiser. It is positive
definite, and $L$ strictly convex, exactly when no nonzero $v$ makes every
term vanish: with every $p_i \in (0, 1)$ that happens if and only if $X$ has
full column rank. Unlike the linear regression Hessian $X^{\top}X$, this one
depends on $w$ through $D$, so the loss surface is convex but not quadratic.

## Newton's method and IRLS

Newton's method replaces the fixed or line-searched step of gradient descent
with the step that exactly minimises the local quadratic model of $L$:

$$
w \leftarrow w - \bigl(X^{\top} D X\bigr)^{-1} X^{\top} \bigl(\sigma(Xw) - y
\bigr) .
$$

This step has a second identity. Write $g = \sigma(Xw) - y$ and define the
**working response** $z^{\ast} = Xw - D^{-1} g$, with entries
$z_i^{\ast} = x_i^{\top} w + (y_i - p_i)/\bigl(p_i(1-p_i)\bigr)$. Then

$$
w - (X^{\top}DX)^{-1} X^{\top} g
= (X^{\top}DX)^{-1} \Bigl[ X^{\top}DXw - X^{\top}g \Bigr]
= (X^{\top}DX)^{-1} X^{\top} D z^{\ast} ,
$$

which is precisely the weighted least squares solution of
$\min_w (z^{\ast} - Xw)^{\top} D (z^{\ast} - Xw)$. Each Newton step is a
weighted linear regression of a working response against $X$, with weights
$D$ recomputed from the current $w$ — **iteratively reweighted least
squares** (IRLS). Newton's method and IRLS are the same algorithm seen from
two directions: one as a second-order optimisation step, the other as
repeated weighted regression.

Near a minimiser with a nonsingular Hessian that is Lipschitz continuous,
Newton's method converges quadratically: $\lVert w_{k+1} - w^{\ast} \rVert
\leq C \lVert w_k - w^{\ast} \rVert^2$ for some constant $C$. Each step
roughly squares the number of correct digits, which is why the exercise's
Newton run finishes in single digits of iterations against the hundreds that
fixed-step gradient descent needs on the same problem — at the price of an
$O(d^3)$ solve per step in place of gradient descent's $O(nd)$.

## Where separability breaks the MLE

Suppose the training data is linearly separable: some $w_0$ satisfies
$x_i^{\top} w_0 > 0$ for every $i$ with $y_i = 1$ and $x_i^{\top} w_0 < 0$ for
every $i$ with $y_i = 0$. Scale it by $t > 0$ and let $t \to \infty$. Every
correctly classified logit $t \, x_i^{\top} w_0$ diverges to
$+\infty$ or $-\infty$ in the direction that drives its loss term to zero, so

$$
L(t w_0) \longrightarrow 0 \quad \text{as } t \to \infty .
$$

The infimum of $L$ is zero, but it is not attained at any finite $w$: no
gradient is ever exactly zero away from $w = 0$, since $\sigma$ never reaches
$0$ or $1$ exactly. Gradient descent and Newton's method both chase this
infimum, so their iterates satisfy $\lVert w_k \rVert \to \infty$ while
$L(w_k) \to 0$ — the loss converges while the parameter that produced it
does not. The Hessian degrades along the way: as $\lvert z_i \rvert \to
\infty$, $p_i(1-p_i) \to 0$, so $D \to 0$ and $X^{\top}DX$ approaches
singularity, which is what makes the unregularised Newton solve unreliable on
data that is separable or close to it.

Adding a ridge penalty $\frac{\lambda}{2} \lVert w \rVert_2^2$ for $\lambda >
0$ changes the objective to $L(w) + \frac{\lambda}{2}\lVert w \rVert_2^2$,
whose Hessian is $X^{\top}DX + \lambda I \succ 0$ for every $w$, regardless of
the rank of $X$ or how separable the data is. The penalty term grows without
bound as $\lVert w \rVert \to \infty$, so the regularised objective is
coercive: it has a unique, finite minimiser, and the solve at every Newton
step stays well posed.

## Multiclass: the softmax

With $K$ classes, replace the single vector $w$ by $w_1, \dots, w_K \in
\mathbb{R}^d$ and model

$$
p(y = k \mid x; W) = \frac{e^{x^{\top} w_k}}{\sum_{j=1}^{K} e^{x^{\top} w_j}} ,
$$

the **softmax** of the $K$ logits. When $K = 2$ this reduces to $\sigma$: the
softmax probability of class $1$ is $e^{z_1}/(e^{z_0}+e^{z_1}) = \sigma(z_1 -
z_0)$, so only the difference of the two logits matters. That is the general
pattern. Adding any constant vector $c \in \mathbb{R}^d$ to every $w_k$
leaves every ratio $e^{x^{\top}(w_k + c)} / e^{x^{\top}(w_j + c)}$ unchanged,
because the factor $e^{x^{\top}c}$ cancels top and bottom. The map from
$(w_1, \dots, w_K)$ to the probabilities it produces is therefore
**over-complete by one**: the $Kd$ nominal parameters carry only $(K-1)d$
degrees of freedom, and fixing one class's vector, typically $w_K = 0$, as a
reference removes exactly the redundant direction without losing any
expressiveness.

## Cost and scaling

Let $n$ be the sample count and $d$ the feature count.

| Quantity | Cost | Notes |
|---|---|---|
| Gradient $X^{\top}(\sigma(Xw) - y)$ | $O(nd)$ | One matrix-vector product plus one matrix-transpose-vector product. |
| Hessian $X^{\top}DX$ | $O(nd^2)$ | Dominates a gradient descent step by a factor of $d$. |
| Newton solve $(X^{\top}DX)^{-1}g$ | $O(d^3)$ once $X^{\top}DX$ is formed | Via Cholesky, since the Hessian is positive definite away from separability. |
| One Newton step overall | $O(nd^2 + d^3)$ | Dominated by forming the Hessian when $n \gg d$. |
| $k$ steps of gradient descent | $O(knd)$ | Cheaper per step, but $k$ scales with the condition number of the Hessian along the path. |

Newton's method trades a per-step cost that is $d$ times larger for a step
count that is typically one or two orders of magnitude smaller near the
optimum, which is why it wins whenever $d$ is small enough that $O(nd^2 +
d^3)$ per step is affordable. Gradient descent remains the only option once
$d$ is large enough that forming, let alone factoring, $X^{\top}DX$ is not.

## Where this breaks

**Separable and near-separable data.** As shown above, the unregularised MLE
diverges. In practice this shows up as coefficients that grow every epoch and
a loss that keeps falling without the validation metric improving. Ridge or
early stopping are the standard fixes; both bound $\lVert w \rVert$ instead of
letting it chase an infimum that no finite point attains.

**Perfect or near-perfect class imbalance.** When one class is rare, most
$p_i(1-p_i)$ terms are tiny, so $X^{\top}DX$ is poorly conditioned even
without separability, and Newton's step becomes numerically fragile in the
same way. Reweighting the loss by inverse class frequency changes $D$ without
changing what the model can represent.

**Naive loss evaluation.** Skipping log-sum-exp does not fail loudly: it
returns `inf` or `nan` silently, and a training loop that does not check for
finiteness reports a broken model as one that failed to learn.

**Newton on wide problems.** When $d$ is comparable to or larger than $n$,
forming and factoring $X^{\top}DX$ costs more than gradient descent can lose
by taking more steps. First-order methods, or quasi-Newton methods that
approximate the Hessian without forming it, take over.

## The exercise

You implement the loss and both fitting routines, then use them to reproduce
the two central claims of the lesson numerically.

Write a numerically stable `sigmoid` and a numerically stable `log_loss` that
matches $L(w) = \sum_i [\operatorname{softplus}(x_i^{\top}w) - y_i
x_i^{\top}w] + \frac{\text{ridge}}{2}\lVert w \rVert^2$, both finite for
logits as large in magnitude as $800$. Write `gradient` and `hessian` from the
formulas derived above, then `fit_gradient_descent` and `fit_newton`, each
returning the fitted coefficients and the loss recorded before every step.

The checks verify stability at $\pm 800$, check `gradient` and `hessian`
against finite differences, confirm the Hessian is positive semidefinite,
and confirm `fit_newton` reaches the same optimum as
`fit_gradient_descent` in under ten iterations where gradient descent needs
hundreds. The last check builds a linearly separable dataset and confirms
that the unregularised coefficient norm keeps growing while the ridge
penalised run's does not — the numerical signature of the divergence this
lesson derives.
