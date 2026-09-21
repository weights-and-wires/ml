---
title: Support vector machines
slug: 09-support-vector-machines
part: "Part 2 — Classical machine learning"
summary: Maximise the margin instead of the likelihood, and read the dual's KKT conditions as a classifier of every point.
minutes: 45
gpu: false
objectives:
  - State the geometric margin and the hard-margin and soft-margin primals it defines.
  - State the SVM dual, including the box constraint and the equality constraint, and see why eliminating the primal variables produces it.
  - Read the KKT conditions as a partition of every training point into margin, support, and violating.
  - Compute the hinge loss and its subgradient, the primal's unconstrained equivalent.
  - Explain why the dual, not the primal, is what admits kernels.
prereqs:
  - 03-gradient-descent
  - 05-logistic-regression
exercise: 09-support-vector-machines
---

# Support vector machines

On separable data, logistic regression's maximum-likelihood weights grow without bound. Lesson 5 showed why: pushing every point's margin further from the boundary keeps lowering the loss, and the sigmoid never saturates enough to stop the climb. Every separating hyperplane fits the training data perfectly, so nothing in the objective prefers one over another.

The support vector machine replaces that objective with one that has an actual best answer. Among every hyperplane that separates the two classes, it picks the one farthest from the nearest point of either class. That is a well-posed geometric question, and it stays well posed even when you let a few points cross the boundary.

This lesson states the SVM's primal and dual, reads the dual's KKT conditions as a label for every training point, and derives the two-variable update behind Sequential Minimal Optimization (SMO), the algorithm the exercise has you build.

## What you already know

- A convex program's KKT conditions — stationarity, primal and dual feasibility, and complementary slackness — are necessary and sufficient for optimality when Slater's condition holds.
- Strong duality: for a convex problem satisfying Slater's condition, the primal and dual optimal values agree.
- A subgradient at a non-differentiable point of a convex function $f$ is any $g$ with $f(z) \ge f(x) + g^{\top}(z-x)$ for every $z$.
- Gradient descent converges on convex, Lipschitz-gradient objectives; subgradient descent converges more slowly on merely convex ones.

## The margin and the primal

For a hyperplane $\{x : w^{\top}x + b = 0\}$ and a labelled point $(x_i, y_i)$ with $y_i \in \{-1, +1\}$, the **geometric margin**

$$
\gamma_i = \frac{y_i(w^{\top}x_i+b)}{\lVert w \rVert}
$$

is the signed distance from $x_i$ to the hyperplane, positive when the point sits on the correct side. Rescaling $w$ and $b$ by the same positive constant leaves $\gamma_i$ unchanged, because the $\lVert w \rVert$ in the denominator cancels it. The SVM chooses $(w,b)$ to maximise the smallest $\gamma_i$ over the training set — the widest gap between the two classes.

Fix the scale of $(w,b)$ so the closest point has $y_i(w^{\top}x_i+b)=1$. Under that normalisation the margin is $1/\lVert w \rVert$, so maximising it is the same as minimising $\lVert w \rVert$, or, for a smooth objective, minimising $\tfrac12\lVert w\rVert^2$. That gives the **hard-margin primal**:

$$
\min_{w,b} \ \tfrac{1}{2}\lVert w \rVert^2
\qquad \text{s.t.} \qquad
y_i(w^{\top}x_i + b) \ge 1 \quad \text{for every } i .
$$

This is a convex quadratic program, and it only has a solution when the data is linearly separable.

Real data rarely is. The **soft-margin primal** adds one slack variable $\xi_i \ge 0$ per point, measuring how far it may fall short of the margin:

$$
\min_{w,b,\xi} \ \tfrac{1}{2}\lVert w \rVert^2 + C\sum_{i=1}^n \xi_i
\qquad \text{s.t.} \qquad
y_i(w^{\top}x_i+b) \ge 1 - \xi_i, \quad \xi_i \ge 0 .
$$

A point with $\xi_i = 0$ meets the original margin. A point with $0 < \xi_i \le 1$ sits inside the margin but on the correct side. A point with $\xi_i > 1$ is misclassified. $C$ trades margin width against total violation: a large $C$ pushes every $\xi_i$ toward 0 and recovers the hard margin, and a small $C$ lets the objective ignore violations and drives $w$ toward 0. $C$ plays the role that $1/\lambda$ plays in ridge regression — both control the same trade-off between fit and the size of $w$.

## The dual

Introduce a multiplier $\alpha_i \ge 0$ for each margin constraint and $\mu_i \ge 0$ for each $\xi_i \ge 0$, and form the Lagrangian. Setting its gradient with respect to $w$, $b$, and every $\xi_i$ to zero gives three stationarity conditions:

$$
w = \sum_i \alpha_i y_i x_i, \qquad \sum_i \alpha_i y_i = 0, \qquad \mu_i = C - \alpha_i .
$$

Dual feasibility needs $\mu_i \ge 0$, so $\mu_i = C - \alpha_i \ge 0$ together with $\alpha_i \ge 0$ gives the **box constraint** $0 \le \alpha_i \le C$. Substituting all three conditions back into the Lagrangian eliminates $w$, $b$, and $\xi$ entirely; the algebra is mechanical, and the destination is what matters. Writing $Q_{ij} = y_iy_j\, x_i^{\top}x_j$, the **dual problem** is

$$
\max_{\alpha} \ \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j}\alpha_i\alpha_j Q_{ij}
\qquad \text{s.t.} \qquad
0 \le \alpha_i \le C, \quad \sum_i \alpha_i y_i = 0 .
$$

The dual is a concave quadratic program in $n$ variables with one equality constraint and $n$ box constraints — simpler than the primal's mix of a norm objective and $2n$ inequalities. The slack variables have disappeared into the box on $\alpha$: allowing margin violations shows up only as a ceiling on how much influence any one point can have.

## The KKT conditions as a label for every point

Complementary slackness, combined with the box constraint, splits every training point into exactly three cases.

- **$\alpha_i = 0$.** Correctly classified and outside the margin: $y_i(w^{\top}x_i+b) \ge 1$. It contributes nothing to $w = \sum_i \alpha_i y_i x_i$.
- **$0 < \alpha_i < C$.** Exactly on the margin boundary: $y_i(w^{\top}x_i+b) = 1$. This is a **support vector** in the strict sense, the source of the model's name, and any one of these points lets you recover $b$ from $w$.
- **$\alpha_i = C$.** A **margin violator**: $\xi_i$ takes whatever value $y_i(w^{\top}x_i+b) = 1 - \xi_i$ requires, so the point may still be correctly classified inside the margin, or fully misclassified.

Only points with $\alpha_i > 0$ enter $w$, and that is typically a small fraction of the training set: the solution is sparse in $\alpha$.

## The hinge loss

The soft-margin primal has an unconstrained equivalent: at the optimum, each $\xi_i$ takes the smallest value its constraint allows, $\xi_i = \max(0, 1 - y_i(w^{\top}x_i+b))$, the **hinge loss** of point $i$. Substituting removes the constraints entirely:

$$
\min_{w,b} \ \tfrac{1}{2}\lVert w \rVert^2 + C \sum_i \max\big(0,\, 1 - y_i(w^{\top}x_i+b)\big) .
$$

Write $z_i = y_i(w^{\top}x_i+b)$ and $\ell(z) = \max(0, 1-z)$, convex and piecewise linear with a single kink at $z=1$. Away from the kink, the subgradient of the total objective is

$$
\nabla_w = w - C\!\!\sum_{i:\, z_i < 1}\!\! y_i x_i, \qquad
\frac{\partial}{\partial b} = -C\!\!\sum_{i:\, z_i < 1}\!\! y_i ,
$$

a sum over exactly the points inside or violating the margin, the same set the KKT analysis flagged as having $\alpha_i > 0$. The hinge loss's active set and the dual's support set are two views of the same condition, $y_i(w^{\top}x_i+b) < 1$.

## Why the dual admits kernels

The dual objective and the prediction rule $\hat y(x) = \operatorname{sign}\big(\sum_i \alpha_i y_i x_i^{\top}x + b\big)$ need the data only through inner products: $x_i^{\top}x_j$ between training points, and $x_i^{\top}x$ at prediction time. Neither ever needs an individual $x_i$ alone. Replace every inner product with a **kernel** $k(x_i,x_j) = \phi(x_i)^{\top}\phi(x_j)$, and you compute the same dual and the same prediction as if every point had first passed through $\phi$, without ever forming $\phi(x)$ — the point of the trick when $\phi$'s target space has far more dimensions than you could build, or infinitely many, as the next lesson makes precise. The primal has no equivalent move: its objective is written in terms of $w$, a vector in that same space, and computing with $w$ directly costs exactly the dimension the trick exists to avoid.

## SMO's two-variable update

The dual's equality constraint $\sum_i \alpha_i y_i = 0$ rules out optimising one $\alpha_i$ at a time, since changing it alone moves the sum away from zero. The smallest update that respects the constraint changes two variables together. **SMO** picks a pair $(\alpha_i, \alpha_j)$ and solves for their joint optimum in closed form, repeatedly.

Write $f(x) = \sum_k \alpha_k y_k x_k^{\top}x + b$ for the current decision function and $E_i = f(x_i) - y_i$ for its error at $x_i$. Holding every other $\alpha_k$ fixed reduces the dual, restricted to $(\alpha_i,\alpha_j)$, to a one-dimensional quadratic whose unclipped optimum is

$$
\alpha_j^{\text{new}} = \alpha_j^{\text{old}} + \frac{y_j(E_i - E_j)}{\eta},
\qquad
\eta = K_{ii} + K_{jj} - 2K_{ij} ,
$$

where $\eta$ is the squared distance between $\phi(x_i)$ and $\phi(x_j)$ in feature space, and is strictly positive whenever the two points differ there. This value is not yet feasible: the line $\alpha_i y_i + \alpha_j y_j = \text{const}$ meets the box $[0,C]^2$ in a segment $[L,H]$,

$$
y_i \ne y_j: \ L = \max(0, \alpha_j^{\text{old}} - \alpha_i^{\text{old}}), \ H = \min(C, C + \alpha_j^{\text{old}} - \alpha_i^{\text{old}}),
$$

$$
y_i = y_j: \ L = \max(0, \alpha_i^{\text{old}} + \alpha_j^{\text{old}} - C), \ H = \min(C, \alpha_i^{\text{old}} + \alpha_j^{\text{old}}) ,
$$

and clipping $\alpha_j^{\text{new}}$ into $[L,H]$ before recovering $\alpha_i^{\text{new}} = \alpha_i^{\text{old}} + y_iy_j(\alpha_j^{\text{old}} - \alpha_j^{\text{new}})$ keeps the step inside the feasible region. Each step solves its two-variable subproblem exactly, so the dual objective never decreases across a sequence of steps, the property the exercise checks directly. Repeating the update over pairs that violate the KKT conditions converges to the global optimum, because the dual is concave and the feasible region is convex.

## Cost and scaling

| Quantity | Cost | Memory |
|---|---|---|
| Gram matrix $K$ | $O(n^2 d)$ once | $O(n^2)$ |
| One SMO step | $O(n)$, from a cached error array | $O(1)$ beyond $K$ |
| A full SMO pass over $n$ points | $O(n^2)$ | — |
| Prediction on an unseen point | $O(n_{sv}\, d)$, using only the support vectors | $O(n_{sv}\, d)$ |
| Generic QP solver on the dual | $O(n^3)$ | $O(n^2)$ |

The Gram matrix is the wall: it costs $O(n^2)$ memory regardless of how you solve the dual, which puts a few tens of thousands of points near the limit of what fits in memory on one machine. SMO trades the $O(n^3)$ factorisation a generic QP solver needs for many cheap $O(n)$ steps, and prediction is cheap because only support vectors and violators carry nonzero weight in $\alpha$.

## Where this breaks

**Choosing $C$.** $C$ is a hyperparameter tuned by the cross-validation of lesson 7. Too large re-introduces the hard margin's sensitivity to a single point; too small ignores the data almost entirely.

**Quadratic memory.** The $O(n^2)$ Gram matrix makes the exact method impractical past roughly $10^5$ points on commodity hardware. Practical systems subsample, approximate the kernel (the subject of the next lesson), or minimise the hinge loss's primal form by (sub)gradient descent, which scales linearly in $n$ at the cost of the kernel trick.

**Kernel choice.** Which kernel to use, and with what hyperparameters, is a model-selection problem like choosing $C$, and the two interact: a more flexible kernel typically wants a smaller effective $C$.

**Slow convergence from a poor working-set choice.** Scanning pairs in a fixed order, as the exercise does for simplicity, takes far more steps to converge than choosing the pair that most violates the KKT conditions, which is what production libraries such as libsvm do.

**No probabilities.** The dual gives you a signed distance to the margin, not a calibrated $p(y\mid x)$. Platt scaling, fitting a one-dimensional logistic regression on the SVM's output, is the standard patch.

## The exercise

You build the pieces of a soft-margin SVM solver from the results above: `hinge_loss_and_subgradient`, evaluating the unconstrained objective and a subgradient at a given $(w,b)$; `kkt_violation`, which reads off the three-way classification above and returns the worst violation across the training set; `smo_step`, one clipped two-variable update; and `fit_smo`, which drives a sequence of those updates from $\alpha=0$ to convergence.

The checks verify the hinge subgradient against a finite-difference approximation away from the kink, and confirm that `smo_step` never lets $\alpha$ leave $[0,C]$ or the equality constraint drift from zero, and that the dual objective never decreases across a sequence of steps. The last two checks run `fit_smo` to convergence on a small, separable problem and confirm that its KKT violation falls below tolerance, and that the $w$ it recovers agrees with a reference solution from a generic constrained optimiser working on the same dual.
