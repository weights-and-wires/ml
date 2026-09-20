---
title: Support vector machines
slug: 09-support-vector-machines
part: "Part 2 — Classical machine learning"
summary: Maximise the margin instead of the likelihood, derive the dual in full, and read its KKT conditions as a classifier of every training point.
minutes: 65
gpu: false
objectives:
  - Derive the hard-margin and soft-margin primals from the geometric margin.
  - Derive the dual in full, including the box constraint and the equality constraint it comes with.
  - Read the KKT conditions as a partition of every point into margin, support, and violating.
  - Derive the hinge loss as the primal's unconstrained equivalent, and its subgradient.
  - Explain why the dual, and not the primal, is what admits kernels.
prereqs:
  - 03-gradient-descent
  - 05-logistic-regression
exercise: 09-support-vector-machines
---

# Support vector machines

On separable data, logistic regression's maximum likelihood estimate does not
exist as a finite point: lesson 5 showed that the weights grow without bound,
because pushing every margin further from the boundary always lowers the
loss a little more. Every separating hyperplane fits the training data
perfectly, and nothing in the logistic objective prefers one over another
except how far it has managed to push the weights before the optimiser
stops.

The support vector machine replaces that objective with one that has an
actual maximiser: among every hyperplane that separates the data, pick the
one that maximises the distance from the boundary to the nearest point of
either class. That is a well-posed geometric problem before it is a
statistical one, and it stays well posed once you drop separability and let
some points cross the boundary. This lesson derives the primal from the
margin, derives its dual by eliminating every primal variable in closed
form, reads the dual's KKT conditions as a classification of each training
point, and derives Sequential Minimal Optimization's two-variable update
from the constraint that makes a one-variable update impossible.

## What you already know

- A convex function's global minimisers are exactly its stationary points, and
  a convex program's KKT conditions — stationarity, primal and dual
  feasibility, and complementary slackness — are necessary and sufficient for
  optimality under a constraint qualification such as Slater's condition.
- Strong duality: for a convex problem satisfying Slater's condition, the
  primal and dual optimal values coincide, and any primal-dual pair satisfying
  the KKT conditions is optimal for both.
- Logistic regression's loss keeps falling as the weights grow on separable
  data, because the sigmoid saturates rather than reaching 1.
- A convex function need not be differentiable everywhere; a subgradient at a
  non-differentiable point is any $g$ with $f(z) \ge f(x) + g^{\top}(z-x)$ for
  every $z$.
- Gradient descent converges on convex, Lipschitz-gradient objectives, and
  subgradient descent converges more slowly on merely convex ones.

## The geometric margin

For a hyperplane $\{x : w^{\top}x + b = 0\}$ and a labelled point $(x_i, y_i)$
with $y_i \in \{-1, +1\}$, define the **functional margin**
$\hat\gamma_i = y_i(w^{\top}x_i + b)$, positive exactly when the point is
classified correctly. The functional margin scales with $(w, b)$: replacing
$(w,b)$ by $(cw, cb)$ for $c > 0$ multiplies every $\hat\gamma_i$ by $c$
without changing which side of the hyperplane any point falls on. The
**geometric margin** removes that ambiguity by dividing out the scale of $w$:

$$
\gamma_i = \frac{y_i(w^{\top}x_i+b)}{\lVert w \rVert} ,
$$

which is the signed Euclidean distance from $x_i$ to the hyperplane. Unlike
the functional margin, $\gamma_i$ is invariant to rescaling $(w,b)$, because
the $\lVert w \rVert$ in the denominator absorbs exactly the factor $c$ that
rescaling introduces upstairs.

Define the margin of the whole training set as
$\gamma = \min_i \gamma_i$, the distance from the hyperplane to its closest
point. The learning problem is to choose $(w,b)$ maximising $\gamma$ subject
to every point being on the correct side.

### Canonical form

Because $\gamma$ is scale-invariant, you are free to fix the scale of $(w,b)$
without losing any hyperplane the maximisation could reach. Fix it so that the
closest point has functional margin exactly 1: $\min_i y_i(w^{\top}x_i+b) = 1$.
Under that normalisation, $\gamma = 1/\lVert w \rVert$, and maximising $\gamma$
is the same problem as minimising $\lVert w \rVert$ — or, squaring for a
differentiable objective and scaling by $\tfrac12$ for a clean gradient,
minimising $\tfrac12\lVert w \rVert^2$.

## The hard-margin primal

Putting the canonical scale directly into the constraints:

$$
\min_{w,b} \ \tfrac{1}{2}\lVert w \rVert^2
\qquad \text{s.t.} \qquad
y_i(w^{\top}x_i + b) \ge 1 \quad \text{for every } i .
$$

This is a convex quadratic program: the objective is a positive semidefinite
quadratic form and every constraint is affine in $(w,b)$. It has a solution
only when the data is linearly separable, since otherwise no $(w,b)$ satisfies
every constraint at once.

## The soft-margin primal

Real data is rarely separable, and even when it is, the separating hyperplane
found by insisting on zero violations can be dictated by a single point close
to the other class. Introduce one slack variable $\xi_i \ge 0$ per point,
measuring how far it is allowed to fall short of the margin:

$$
\min_{w,b,\xi} \ \tfrac{1}{2}\lVert w \rVert^2 + C\sum_{i=1}^n \xi_i
\qquad \text{s.t.} \qquad
y_i(w^{\top}x_i+b) \ge 1 - \xi_i, \quad \xi_i \ge 0 \quad \text{for every } i .
$$

A point with $\xi_i = 0$ meets the original margin requirement; one with
$0 < \xi_i \le 1$ is inside the margin but still on the correct side; one with
$\xi_i > 1$ is misclassified. $C > 0$ trades margin width against total
violation: as $C \to \infty$ every $\xi_i$ is forced to 0 and the problem
recovers the hard margin (when it is feasible); as $C \to 0$ the penalty on
violations vanishes and the objective is minimised by $w = 0$, the widest
possible margin with no regard for correctness. $C$ plays the role
$1/\lambda$ plays in ridge regression: both control the same trade between
fitting the data and controlling the norm of $w$, one as a multiplier on the
loss, the other as a divisor on the penalty.

## The Lagrangian and the dual

Introduce multipliers $\alpha_i \ge 0$ for the margin constraints and
$\mu_i \ge 0$ for $\xi_i \ge 0$:

$$
\mathcal{L}(w,b,\xi,\alpha,\mu) = \tfrac{1}{2}\lVert w \rVert^2 + C\sum_i \xi_i
- \sum_i \alpha_i\big[y_i(w^{\top}x_i+b) - 1 + \xi_i\big] - \sum_i \mu_i \xi_i .
$$

### Stationarity

Differentiate with respect to each primal variable and set the result to zero.

$$
\nabla_w \mathcal{L} = w - \sum_i \alpha_i y_i x_i = 0
\ \Longrightarrow\
w = \sum_i \alpha_i y_i x_i ,
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = -\sum_i \alpha_i y_i = 0
\ \Longrightarrow\
\sum_i \alpha_i y_i = 0 ,
$$

$$
\frac{\partial \mathcal{L}}{\partial \xi_i} = C - \alpha_i - \mu_i = 0
\ \Longrightarrow\
\mu_i = C - \alpha_i .
$$

Dual feasibility requires $\mu_i \ge 0$, which combined with $\mu_i = C -
\alpha_i$ gives $\alpha_i \le C$; combined with $\alpha_i \ge 0$ this is the
**box constraint** $0 \le \alpha_i \le C$. The slack variables have vanished
into a bound on $\alpha$ rather than appearing in the dual directly — the
price of allowing margin violations shows up only as a ceiling on how much
influence any single point can have.

### Eliminating the primal variables

Substitute $\mu_i = C - \alpha_i$ into $\mathcal{L}$. The slack terms collect
as $\sum_i \xi_i(C - \alpha_i - \mu_i)$, which is identically zero at
stationarity, so every $\xi_i$ disappears from the Lagrangian entirely:

$$
\mathcal{L} = \tfrac{1}{2}\lVert w \rVert^2 - \sum_i \alpha_i y_i(w^{\top}x_i + b) + \sum_i \alpha_i .
$$

The $b$ term is $-b\sum_i \alpha_i y_i$, which vanishes by the second
stationarity condition. Substitute $w = \sum_i \alpha_i y_i x_i$ into the
remaining two terms. Writing $Q_{ij} = y_i y_j x_i^{\top}x_j$,

$$
\tfrac{1}{2}\lVert w \rVert^2 = \tfrac{1}{2}\sum_{i,j}\alpha_i\alpha_j Q_{ij},
\qquad
\sum_i \alpha_i y_i w^{\top}x_i = \sum_i \alpha_i y_i x_i^{\top}\sum_j \alpha_j y_j x_j = \sum_{i,j}\alpha_i \alpha_j Q_{ij} ,
$$

so the first term is exactly half the second, and

$$
\mathcal{L} = \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j}\alpha_i \alpha_j y_i y_j x_i^{\top}x_j .
$$

Every primal variable is gone. The **dual problem** is

$$
\max_{\alpha} \ \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j (x_i^{\top}x_j)
\qquad \text{s.t.} \qquad
0 \le \alpha_i \le C, \quad \sum_i \alpha_i y_i = 0 ,
$$

a concave quadratic program in $n$ variables with one equality constraint and
$n$ box constraints — a dramatically simpler feasible region than the primal's
mix of a norm objective and $2n$ inequality constraints, and, as the last
section of this lesson argues, one that generalises to kernels for free.

## The KKT conditions as a classification of every point

Complementary slackness gives two conditions per point:

$$
\alpha_i\big[y_i(w^{\top}x_i+b) - 1 + \xi_i\big] = 0, \qquad \mu_i \xi_i = 0 .
$$

Since $\mu_i = C - \alpha_i$, the second condition is
$(C-\alpha_i)\xi_i = 0$. Reading the two together against the box constraint
splits every training point into exactly three categories.

- **$\alpha_i = 0$.** Then $\mu_i = C > 0$, forcing $\xi_i = 0$, and the first
  condition holds regardless of the margin term. The point is not required to
  lie on the margin at all: $y_i(w^{\top}x_i+b) \ge 1$, correctly classified
  and outside the margin. It contributes nothing to $w = \sum_i \alpha_i y_i
  x_i$.
- **$0 < \alpha_i < C$.** Then $\mu_i > 0$ forces $\xi_i = 0$, and $\alpha_i
  \ne 0$ forces the margin term to vanish: $y_i(w^{\top}x_i+b) = 1$ exactly.
  This point sits precisely on the margin boundary — a **support vector** in
  the strict sense, and the source of the model's name, since these are the
  only points whose exact position matters. Any one of them lets you recover
  $b$ from $w$.
- **$\alpha_i = C$.** Then $\mu_i = 0$ and $\xi_i$ is unconstrained by the
  second condition, so it takes whatever value the margin constraint
  requires: $y_i(w^{\top}x_i+b) = 1 - \xi_i$. If $\xi_i = 0$ the point is
  still on the margin; if $0 < \xi_i \le 1$ it is inside the margin but
  correctly classified; if $\xi_i > 1$ it is misclassified. These are the
  **margin violators**.

Only points with $\alpha_i > 0$ enter $w$, and typically only a small fraction
of the training set does: the prediction rule
$\hat y(x) = \operatorname{sign}\big(\sum_i \alpha_i y_i x_i^{\top}x + b\big)$
depends only on the support vectors and violators, which is why the solution
is sparse in $\alpha$ even though the primal's $w$ is a dense combination of
every one of them.

## The hinge loss

The soft-margin primal has an unconstrained equivalent. At the optimum, each
$\xi_i$ takes the smallest value consistent with its constraint — increasing
$\xi_i$ beyond that only adds to the objective for no benefit — so
$\xi_i = \max(0, 1 - y_i(w^{\top}x_i+b))$, the **hinge loss** of point $i$.
Substituting removes the constraints entirely:

$$
\min_{w,b} \ \tfrac{1}{2}\lVert w \rVert^2 + C \sum_i \max\big(0,\, 1 - y_i(w^{\top}x_i+b)\big) .
$$

Write $z_i = y_i(w^{\top}x_i+b)$ and $\ell(z) = \max(0, 1-z)$. $\ell$ is
convex and piecewise linear, with a single kink at $z=1$:

$$
\ell'(z) = \begin{cases} -1 & z < 1 \\ 0 & z > 1 \end{cases},
\qquad
\partial \ell(1) = [-1, 0] .
$$

Away from the kink, the subgradient of the total objective with respect to
$w$ is $w - C\sum_{i : z_i < 1} y_i x_i$ and with respect to $b$ is
$-C\sum_{i: z_i<1} y_i$ — a sum over exactly the points inside or violating
the margin, which is the same set the KKT analysis singled out as having
$\alpha_i > 0$. That correspondence is not a coincidence: the hinge loss's
active set and the dual's support set are two views of the same condition,
$y_i(w^{\top}x_i+b) < 1$.

## Why the dual admits kernels

Look at what the dual objective and the prediction rule actually need from
the data: $x_i^{\top}x_j$ for every pair of training points, and
$x_i^{\top}x$ for a new point $x$ at prediction time. Neither ever needs an
individual $x_i$ on its own. Replacing every inner product
$x_i^{\top}x_j$ with a **kernel** $k(x_i,x_j) = \phi(x_i)^{\top}\phi(x_j)$
computes the same dual and the same prediction rule as if you had first
mapped every point through some feature map $\phi$, without ever forming
$\phi(x)$ — which matters most when $\phi$ maps into a space with far more
dimensions than you could afford to instantiate, or infinitely many, as the
next lesson makes precise.

The primal has no equivalent move available. Its objective and constraints
are written in terms of $w$, a vector living in whatever space $\phi$ maps
into, and computing with $w$ directly requires forming it — which costs
exactly the dimension the kernel trick exists to avoid. The dual's
elimination of $w$ in favour of $\sum_i \alpha_i y_i x_i$ is what makes the
inner-product-only structure visible in the first place; that elimination is
the entire reason to derive the dual, beyond the fact that its feasible
region is simpler.

## SMO's two-variable update

The dual's equality constraint $\sum_i \alpha_i y_i = 0$ rules out optimising
one $\alpha_i$ at a time: changing $\alpha_i$ alone while holding every other
$\alpha_k$ fixed moves the sum away from zero unless $y_i = 0$, which never
holds. The smallest update that can respect the constraint changes two
variables at once, compensating one against the other. **Sequential Minimal
Optimization** (SMO) does exactly that, repeatedly, choosing a pair
$(\alpha_i, \alpha_j)$ and solving for the pair's joint optimum in closed
form.

Fix every $\alpha_k$ for $k \ne i,j$. The constraint pins
$\alpha_i y_i + \alpha_j y_j = \zeta$ to whatever value the current
$\alpha_i, \alpha_j$ give it, so $\alpha_i$ is an affine function of
$\alpha_j$. Substituting that relationship turns the dual objective,
restricted to this pair, into a one-dimensional quadratic in $\alpha_j$ alone.
Writing $f(x) = \sum_k \alpha_k y_k x_k^{\top}x + b$ for the current decision
function and $E_i = f(x_i) - y_i$ for its error at $x_i$, differentiating that
quadratic and setting the result to zero gives the unclipped update

$$
\alpha_j^{\text{new}} = \alpha_j^{\text{old}} + \frac{y_j(E_i - E_j)}{\eta},
\qquad
\eta = K_{ii} + K_{jj} - 2K_{ij} ,
$$

where $\eta$ is the squared distance between $\phi(x_i)$ and $\phi(x_j)$ in
feature space, and is strictly positive whenever the two points are distinct
in that space. This unclipped value is not yet feasible: the line
$\alpha_i y_i + \alpha_j y_j = \zeta$ intersects the box $[0,C]^2$ in a segment
$[L, H]$ that depends on whether $y_i = y_j$,

$$
y_i \ne y_j: \ L = \max(0, \alpha_j^{\text{old}} - \alpha_i^{\text{old}}), \ H = \min(C, C + \alpha_j^{\text{old}} - \alpha_i^{\text{old}}),
$$

$$
y_i = y_j: \ L = \max(0, \alpha_i^{\text{old}} + \alpha_j^{\text{old}} - C), \ H = \min(C, \alpha_i^{\text{old}} + \alpha_j^{\text{old}}) ,
$$

and clipping $\alpha_j^{\text{new}}$ into $[L,H]$ before recovering
$\alpha_i^{\text{new}} = \alpha_i^{\text{old}} + y_iy_j(\alpha_j^{\text{old}} -
\alpha_j^{\text{new}})$ is what keeps the step inside the feasible region.
Because each step solves its two-variable subproblem exactly, the dual
objective never decreases across a sequence of steps — the property the
exercise checks directly — and repeating the update, sweeping over pairs that
violate the KKT conditions, converges to the global optimum because the dual
is concave and the feasible region is convex.

## Cost and scaling

| Quantity | Cost | Memory |
|---|---|---|
| Gram matrix $K$ | $O(n^2 d)$ once | $O(n^2)$ |
| One SMO step | $O(n)$ (recompute $f(x_i), f(x_j)$ from a cached error array) | $O(1)$ beyond $K$ |
| A full SMO pass over $n$ points | $O(n^2)$ | — |
| Prediction on a new point | $O(n_{sv} \, d)$, using only the support vectors | $O(n_{sv} \, d)$ |
| Generic QP solver on the dual | $O(n^3)$ | $O(n^2)$ |

The Gram matrix is the wall: it costs $O(n^2)$ memory regardless of how the
dual is solved, which puts a training set of a few tens of thousands of
points near the limit of what fits in memory on one machine. SMO avoids ever
forming the $O(n^3)$ factorisation a generic QP solver would use, trading it
for many cheap $O(n)$ steps — a good trade whenever the number of steps to
convergence stays well below $n$, which working-set heuristics that prioritise
the most-violated pairs are designed to ensure. Prediction is cheap exactly
because complementary slackness makes $\alpha$ sparse: only the support
vectors and violators carry nonzero weight, so a model trained on a huge
dataset can still predict in time proportional to a much smaller support set.

## Where this breaks

**Choosing $C$.** $C$ is a hyperparameter like any regularisation strength,
selected by the cross-validation of lesson 7, and the SVM offers no default
that works across problems: too large re-introduces the hard margin's
sensitivity to a single point, too small ignores the data almost entirely.

**Quadratic memory.** The $O(n^2)$ Gram matrix makes the exact method
impractical past roughly $10^5$ points on commodity hardware. Practical
systems either subsample, approximate the kernel — random Fourier features
and the Nyström method, both covered next lesson — or abandon the dual
entirely and minimise the hinge loss's primal form directly by (sub)gradient
descent, which scales linearly in $n$ at the cost of losing the kernel trick.

**Kernel choice.** The dual's inner-product structure makes kernels available,
but which kernel to use, and with what hyperparameters, is a model selection
problem exactly like choosing $C$ — and the two interact, since a more
flexible kernel typically wants a smaller effective $C$ to avoid overfitting
the training set's support vectors.

**Slow convergence from a poor working-set choice.** SMO's convergence rate
depends heavily on which pair is chosen at each step. Scanning pairs in a
fixed order, as the exercise does for simplicity, can take far more steps
than choosing the pair that most violates the KKT conditions, which is what
production implementations such as libsvm do, at the cost of a more
complicated selection rule.

**No probabilities.** The dual gives you a signed distance to the margin, not
a calibrated $p(y\mid x)$. Platt scaling — fitting a one-dimensional logistic
regression on top of the SVM's output — is the standard patch, and it is
exactly logistic regression again, now solving a problem the SVM's own
training never touched.

## The exercise

You build the pieces of a soft-margin SVM solver from the derivations above.
Implement `hinge_loss_and_subgradient`, evaluating the unconstrained
objective and a subgradient at a given $(w,b)$; `kkt_violation`, which reads
off the three-way classification above and returns the worst violation across
the training set; `smo_step`, one clipped two-variable update; and `fit_smo`,
which drives a sequence of those updates from $\alpha=0$ to convergence.

The checks verify the hinge subgradient against a finite-difference
approximation at points away from the kink, confirm that `smo_step` never
lets $\alpha$ leave $[0,C]$ or the equality constraint drift from zero, and
confirm that the dual objective never decreases across a sequence of steps —
the property that makes convergence possible at all. The last two checks run
`fit_smo` to convergence on a small, separable problem and confirm that its
KKT violation falls below tolerance and that the $w$ it recovers agrees with
a reference solution obtained from a generic constrained optimiser working on
the same dual.
