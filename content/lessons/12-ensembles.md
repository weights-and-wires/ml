---
title: Bagging, forests, and boosting
slug: 12-ensembles
part: "Part 2 — Classical machine learning"
summary: The variance identity that explains bagging, the decorrelation that random forests add to it, and boosting derived as gradient descent in function space.
minutes: 60
gpu: false
objectives:
  - Derive the variance of an average of correlated estimators and read off why decorrelation, not more trees, is the lever bagging has left.
  - Derive the in-bag fraction of the bootstrap and explain out-of-bag error as a held-out estimate that costs nothing extra.
  - Derive gradient boosting as stagewise functional gradient descent, and show it reduces to fitting residuals under squared error.
  - Derive AdaBoost's weight update as the exponential-loss instance of the same framework.
prereqs:
  - 11-decision-trees
exercise: 12-ensembles
---

# Bagging, forests, and boosting

A single decision tree is a low-bias, high-variance estimator: it can
represent an arbitrarily fine partition of feature space, and a small change
to the training sample can flip which split the greedy search takes at the
root, rearranging every split beneath it. Two families of methods turn a
collection of such unstable estimators into one accurate predictor, by two
opposite routes. Bagging averages many trees fit independently, trading
variance for none of the bias, and works only as well as the trees disagree
with each other. Boosting fits trees sequentially, each one correcting the
errors of the ensemble built so far, trading bias for a controlled amount of
variance. Both are instances of a single question — how to combine many weak
estimators — answered in opposite directions.

This lesson derives the variance identity that governs bagging and explains
why random forests add feature subsampling on top of it, then derives boosting
from the ground up as gradient descent carried out in the space of functions
rather than the space of parameters.

## What you already know

- A decision tree, grown to purity, drives training error to zero and is the
  high-variance estimator this lesson stabilises.
- $\operatorname{Var}(aX + bY) = a^2\operatorname{Var}(X) + b^2\operatorname{Var}(Y)
  + 2ab\operatorname{Cov}(X, Y)$ for any two random variables.
- The gradient of a scalar loss with respect to a vector of predictions is a
  vector of the same shape, one entry per prediction.
- $(1 - 1/n)^n \to e^{-1}$ as $n \to \infty$.

## The bagging variance identity

Let $\hat{f}_1, \ldots, \hat{f}_B$ be estimators fit on $B$ independent
resamples of the training data, each with the same variance
$\operatorname{Var}(\hat{f}_b) = \sigma^2$ and the same pairwise correlation
$\operatorname{Corr}(\hat{f}_i, \hat{f}_j) = \rho$ for $i \neq j$, so that
$\operatorname{Cov}(\hat{f}_i, \hat{f}_j) = \rho \sigma^2$. Bagging predicts
with the average $\bar{f} = \frac{1}{B} \sum_b \hat{f}_b$. Expanding the
variance of the sum,

$$
\operatorname{Var}(\bar{f}) = \frac{1}{B^2} \left(
\sum_{b=1}^{B} \operatorname{Var}(\hat{f}_b)
+ \sum_{i \neq j} \operatorname{Cov}(\hat{f}_i, \hat{f}_j)
\right)
= \frac{1}{B^2} \left( B\sigma^2 + B(B-1)\rho\sigma^2 \right)
= \frac{\sigma^2}{B} + \frac{B-1}{B}\rho\sigma^2 .
$$

Regrouping the two terms by their dependence on $B$,

$$
\operatorname{Var}(\bar{f}) = \rho\sigma^2 + \frac{1 - \rho}{B}\sigma^2 .
$$

The second term is the part averaging kills: it vanishes as $B \to \infty$
exactly as fast as the variance of a mean of $B$ i.i.d. terms does. The first
term does not depend on $B$ at all. However many trees you average, the
variance of the ensemble cannot fall below $\rho\sigma^2$. Bagging alone
cannot make an ensemble of highly correlated trees behave like an ensemble of
independent ones; it can only average out the part of the variance that
resampling actually decorrelates. The lever left to pull is $\rho$ itself —
lowering the correlation between trees lowers the floor, which is exactly what
random forests do.

### The bootstrap and the in-bag fraction

Each $\hat{f}_b$ is fit on a bootstrap resample: $n$ draws with replacement
from the $n$ training points. A given training point $i$ is excluded from a
single draw with probability $1 - 1/n$, so it is excluded from all $n$ draws
of one bootstrap sample with probability $(1 - 1/n)^n$. As $n \to \infty$,

$$
\left(1 - \frac{1}{n}\right)^n = \exp\left(n \log\left(1 - \frac{1}{n}\right)\right)
\to \exp(-1) ,
$$

using $\log(1 - 1/n) = -1/n - O(1/n^2)$. A bootstrap sample therefore includes
a given point with probability converging to $1 - e^{-1} \approx 0.632$, and
leaves it out — out-of-bag — with probability converging to $e^{-1} \approx
0.368$. Both fractions are already close to their limits by $n$ in the low
hundreds.

### Out-of-bag error

Every training point is out-of-bag for roughly a third of the trees, by
construction. Averaging the predictions of only those trees for which point
$i$ was out-of-bag gives a prediction for $i$ made by estimators that never
saw it during fitting — a held-out prediction, obtained without holding
anything out of training. Averaging the resulting error over all $n$ points
gives the **out-of-bag (OOB) error**, an estimate of generalisation error that
costs nothing beyond the trees you were fitting anyway, and that tracks
$k$-fold cross-validation closely once $B$ is large enough that every point
has been out-of-bag for a reasonable number of trees.

### Random forests: decorrelation by feature subsampling

The variance identity says the only route past $\rho\sigma^2$ is reducing
$\rho$. A random forest does this by restricting each split search, at every
node of every tree, to a random subset of $m < d$ features rather than all
$d$. When two features are both strongly predictive, ordinary bagged trees
tend to split on the stronger one near the root of nearly every tree,
producing highly correlated trees regardless of which bootstrap sample each
saw. Denying most trees access to the stronger feature at a given split forces
some of them to rely on the second-best feature instead, which decorrelates
the resulting predictions directly, at the cost of making each individual tree
a slightly worse fit — a bias-variance trade paid for in $\rho$ rather than in
$\sigma^2$.

## Boosting as functional gradient descent

Bagging fits every estimator to the same target and lets averaging do the
work. Boosting instead builds an additive model

$$
F_M(x) = \sum_{m=1}^{M} \eta \, h_m(x) ,
$$

one term at a time, where each $h_m$ is a weak learner and $\eta$ is a
shrinkage factor. Treat the vector of predictions
$F = (F(x_1), \ldots, F(x_n))$ as the variable of a generic optimisation
problem: minimise $L(F) = \sum_i \ell(y_i, F(x_i))$ over all of
$\mathbb{R}^n$, ignoring for a moment that $F$ has to come from evaluating a
sum of weak learners.

Ordinary gradient descent on $L(F)$ would move
$F \leftarrow F - \eta \nabla_F L(F)$, where the $i$-th entry of the gradient
is $\partial \ell(y_i, F(x_i)) / \partial F(x_i)$, evaluated at the current
$F$. That step is not reachable in general, because it moves every
$F(x_i)$ independently and nothing constrains the result to be a function
that generalises to unseen $x$. Boosting's move is to fit a weak learner
$h_m$ — a shallow tree — to the negative gradient

$$
g_i = -\left.\frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)}\right|_{F = F_{m-1}} ,
$$

by least squares, treating $g$ as a regression target. The fitted $h_m$ is
the best approximation, within the weak-learner class, to the direction
ordinary gradient descent would have taken, and $F_m = F_{m-1} + \eta h_m$
takes a step of length $\eta$ along it. This is functional gradient descent:
the same algorithm as gradient descent, with the projection onto a hypothesis
class standing in for the unconstrained step.

A line search chooses the step size instead of fixing it in advance:
$\gamma_m = \operatorname{arg\,min}_\gamma \sum_i \ell(y_i, F_{m-1}(x_i) +
\gamma h_m(x_i))$, and the update becomes $F_m = F_{m-1} + \eta \gamma_m h_m$,
with $\eta \leq 1$ still shrinking the found optimum. Squared error makes the
line search unnecessary, as the next section shows.

### Squared error reduces to fitting residuals

Take $\ell(y, F) = \tfrac{1}{2}(y - F)^2$. The negative gradient is
$g_i = y_i - F_{m-1}(x_i)$, the residual of the current model — gradient
boosting under squared error fits every weak learner to the residual left by
the ensemble so far, with no further transformation. Write $h = h_m(X)$ for
the fitted stump's predictions on the training rows. Because $h$ is the
least-squares fit of $g$ restricted to each leaf, the fitted value on every
leaf is exactly the mean of $g$ over that leaf, so the residual $g - h$ is
orthogonal to $h$ within every leaf and hence overall:
$\sum_i h_i (g_i - h_i) = 0$, so $\sum_i g_i h_i = \sum_i h_i^2$. This is
the same normal-equations orthogonality that made ordinary least squares a
projection in lesson 2, specialised to the indicator basis of a partition.

That identity pins down the line search in closed form. Expanding the loss
after a step of size $\gamma$,

$$
\sum_i \tfrac{1}{2}(g_i - \gamma h_i)^2
= \tfrac{1}{2}\sum_i g_i^2 - \gamma \sum_i g_i h_i + \tfrac{1}{2}\gamma^2 \sum_i h_i^2 ,
$$

and substituting $\sum_i g_i h_i = \sum_i h_i^2$ turns the derivative
$-\sum_i g_i h_i + \gamma \sum_i h_i^2$ into $(\gamma - 1)\sum_i h_i^2$, which
is zero exactly at $\gamma = 1$. The stump fit already found the loss-minimising
step; there is nothing left for a line search to do, which is why textbook
gradient boosting for squared error is described as fitting residuals with no
mention of a step size. Substituting $\gamma = 1$ into the expansion gives the
loss after one round of shrinkage $\eta$ in closed form:

$$
L_m = L_{m-1} - \eta\left(1 - \tfrac{\eta}{2}\right) \sum_i h_i^2 .
$$

For any $\eta \in (0, 2)$ the coefficient $\eta(1 - \eta/2)$ is strictly
positive, so $L_m \leq L_{m-1}$ with equality only when the fitted stump is
identically zero — training loss decreases every round, and the exact amount
it decreases by is a closed-form function of $\eta$ and the stump's own
squared output.

### AdaBoost: the exponential-loss instance

Take $y \in \{-1, +1\}$ and $\ell(y, F) = e^{-yF}$. The current per-point
weight is $w_i = e^{-y_i F_{m-1}(x_i)}$, and the weak learner is restricted to
$h(x) \in \{-1, +1\}$. For a step of size $\gamma$ along a fixed $h_m$, split
the weighted exponential loss by whether $h_m$ agrees with $y_i$:

$$
L(\gamma) = \sum_{i:\, y_i h_m(x_i) = 1} w_i e^{-\gamma} + \sum_{i:\, y_i h_m(x_i) = -1} w_i e^{\gamma}
= W_c e^{-\gamma} + W_w e^{\gamma} ,
$$

with $W_c$ and $W_w$ the weight mass on the correctly and incorrectly
classified points. Differentiating and setting the result to zero,
$-W_c e^{-\gamma} + W_w e^{\gamma} = 0$, gives $e^{2\gamma} = W_c / W_w$ and

$$
\gamma_m = \frac{1}{2} \log \frac{W_c}{W_w} = \frac{1}{2} \log \frac{1 - \epsilon_m}{\epsilon_m} ,
\qquad \epsilon_m = \frac{W_w}{W_c + W_w} ,
$$

the weighted misclassification rate. Choosing $h_m$ to minimise $L(\gamma_m)$
is equivalent to choosing the classifier that minimises $\epsilon_m$, so
AdaBoost's weak-learner search — a weighted classification problem — is the
same weighted-least-squares-style fit used for squared error, specialised to
$\pm 1$ outputs. The multiplicative reweighting

$$
w_i \leftarrow w_i \, e^{-\gamma_m y_i h_m(x_i)}
$$

follows directly from $F_m = F_{m-1} + \gamma_m h_m$ and
$w_i = e^{-y_i F_m(x_i)}$: a correctly classified point has its weight scaled
by $e^{-\gamma_m} = \sqrt{\epsilon_m / (1 - \epsilon_m)} < 1$, and a
misclassified point by $e^{\gamma_m} = \sqrt{(1 - \epsilon_m)/\epsilon_m} > 1$
— exactly the up-weighting of mistakes that AdaBoost is usually introduced
through, recovered here as one instance of the general functional gradient
framework rather than as a standalone rule.

### Shrinkage and subsampling

$\eta < 1$ trades rounds for stability: the closed-form loss decrease above
shows every round contributes less progress at a smaller $\eta$, so reaching a
given training loss needs proportionally more rounds, and the ensemble spends
longer in the low-variance region near its initialisation before the
individual stumps' idiosyncrasies accumulate. Subsampling a fraction of the
rows before fitting each round's weak learner — stochastic gradient boosting
— adds the same kind of decorrelation between successive learners that
feature subsampling adds between the trees of a random forest, at the cost of
a noisier estimate of the gradient at each step.

## Cost and scaling

| Operation | Cost | Notes |
|---|---|---|
| One bootstrap resample | $O(n)$ | $n$ draws from a discrete uniform distribution. |
| Fitting $B$ bagged trees, depth-limited to $O(\log n)$ | $O(Bnd \log n)$ | Independent of each other; embarrassingly parallel. |
| Out-of-bag error, given the $B$ fitted trees | $O(Bn)$ | One prediction per tree per training point. |
| One boosting round, depth-limited weak learner | $O(nd \log n)$ | Same split search as a single tree, on the current residual. |
| $M$ boosting rounds | $O(Mnd \log n)$ | Strictly sequential: round $m$ needs $F_{m-1}$. |

Bagging parallelises across trees and boosting does not, because every
boosting round's target depends on every earlier round's fit — the sequential
dependency is the price of correcting errors rather than merely averaging over
them.

## Where this breaks

**Bagging plateaus.** The variance floor $\rho\sigma^2$ means adding trees
past the point where $\rho$ stops falling wastes computation without reducing
error; the marginal tree helps only through the vanishing $(1-\rho)\sigma^2/B$
term.

**Boosting overfits when unregularised.** Fitting the residual exactly, with
$\eta = 1$ and unlimited rounds, drives training loss to zero and can drive
generalisation error up well before that point, because later rounds fit
successively smaller and more sample-specific patterns in the residual.
Shrinkage, row subsampling, and early stopping against a validation set are
the standard controls.

**AdaBoost is sensitive to label noise.** The exponential loss grows without
bound as a prediction moves further from the wrong side of the margin, so a
single mislabelled point can dominate the weight distribution after a few
rounds, pulling every subsequent weak learner toward fitting that one point.
Losses that grow linearly, or that cap their growth, are the usual response.

**Neither method fixes a systematically biased weak learner.** Bagging cannot
lower bias at all — averaging correlated or independent copies of a biased
estimator returns something with the same bias. Boosting can lower bias
because each round targets the current residual, but does so at the cost of
increasing variance round over round, which is exactly why it needs shrinkage
where bagging does not.

## The exercise

You implement the mechanics behind both families. Write `bootstrap_indices`,
which draws one bootstrap resample, and `bagging_variance(rho, sigma2, B)`,
the closed-form identity derived above. Write `fit_bagged_stumps`, which fits
an ensemble of one-split regression trees to independent bootstrap resamples,
and `out_of_bag_error`, which scores that ensemble using only the trees for
which each point was held out. Finally write `gradient_boost(X, y, loss_grad,
n_rounds, learning_rate)`, stagewise functional gradient descent with
depth-limited trees as the weak learner, where `loss_grad(y, F)` returns the
negative gradient — the regression target for the next round — given the
current predictions `F`.

The checks confirm `bagging_variance` against its two extremes and against a
Monte Carlo simulation of correlated Gaussian estimators, that the empirical
in-bag fraction of `bootstrap_indices` converges to $1 - 1/e$, that
`out_of_bag_error` tracks a genuinely held-out test error, that the first
round of `gradient_boost` under squared error targets the raw labels
themselves (the residual against a zero initial model), that training loss
decreases monotonically round over round, and that the loss reduction of a
single round matches the closed-form $\eta(1 - \eta/2)\sum_i h_i^2$ identity
exactly across several learning rates.
