---
title: Generalisation and model selection
slug: 07-model-selection
part: "Part 2 — Classical machine learning"
summary: Generalisation bounds are too loose to pick a model, so this lesson builds the tools that estimate risk instead.
minutes: 45
gpu: false
objectives:
  - State the finite-sample generalisation bound's shape and explain why it's too loose to pick a model.
  - Derive the closed-form leave-one-out residual for a linear smoother from the hat matrix.
  - Explain the bias-variance trade-off in k as k ranges from 2 to n.
  - Explain why hyperparameter selection without a nested split is optimistically biased.
prereqs:
  - 01-the-learning-problem
exercise: 07-model-selection
---

# Generalisation and model selection

Every model in this course minimises an empirical risk
$\hat R(h) = \frac1n \sum_i \ell(h(x_i), y_i)$. Nothing in that minimisation
refers to the risk that matters, $R(h) = \mathbb{E}_{(x,y)}[\ell(h(x), y)]$
over the true data distribution. Lesson 1 introduced the generalisation gap
$R(h) - \hat R(h)$ as the quantity that separates the two, and used a
held-out set to estimate it once.

This lesson asks how large that gap can be in the worst case, and shows the
answer is too loose to act on directly. What a practitioner uses instead is
cross-validation, the leave-one-out shortcut, and learning curves — and the
reason they take the shape they do traces straight back to the bounds this
lesson opens with.

## What you already know

- Empirical risk minimisation, and that $\hat R(h)$ evaluated on the data
  used to choose $h$ is an optimistic estimate of $R(h)$.
- Hoeffding's inequality: for i.i.d. $Z_1, \ldots, Z_n \in [0,1]$ with mean
  $\mu$, $P(|\bar Z - \mu| \geq \epsilon) \leq 2e^{-2n\epsilon^2}$.
- A union bound: $P(\bigcup_j A_j) \leq \sum_j P(A_j)$.
- The ridge regression hat matrix $H = X(X^{\top}X + \lambda I)^{-1}X^{\top}$
  and the fitted values $\hat y = Hy$.
- The bias-variance decomposition of a risk estimate into a systematic
  offset and a spread across resamples of the data.

## How loose the worst case is

Fix a hypothesis class $\mathcal{H}$ with $M$ elements and a loss bounded in
$[0,1]$. Combining Hoeffding's inequality with a union bound over all $M$
hypotheses gives: with probability at least $1-\delta$, every
$h \in \mathcal{H}$ satisfies

$$
|\hat R(h) - R(h)| \leq \sqrt{\frac{\log(2M/\delta)}{2n}} .
$$

The gap shrinks at the rate $O(1/\sqrt n)$ and grows only logarithmically in
the size of the class — a shape every bound in this lesson shares. It's
also useless once $\mathcal{H}$ is infinite, which every parametric model in
this course has been.

Two extensions replace $\log M$ with a measure of richness that survives an
infinite class. Rademacher complexity,

$$
\hat{\mathfrak{R}}_S(\mathcal{F}) = \mathbb{E}_{\sigma}\left[
\sup_{f \in \mathcal{F}} \frac1n \sum_{i=1}^n \sigma_i f(z_i)\right] ,
$$

measures how well a function class can correlate with pure random noise
$\sigma_i \in \{-1, +1\}$: a class of one function scores close to $0$, and
a class rich enough to match every sign pattern scores $1$, its maximum. A
symmetrisation argument turns this into a bound of the same $O(1/\sqrt n)$
shape, with $\log M$ replaced by $\mathfrak{R}_n(\mathcal{F})$. For
classifiers, the analogous measure is the VC dimension, a single number
describing the largest sample a class can label every possible way; the
sample size needed to hold the gap to $\epsilon$ scales linearly in it.
Every version of the bound carries the same message: richness enters
through one number, and the gap shrinks at the same $1/\sqrt n$ rate
regardless of which number that is.

## Why these bounds can't pick a model

Every bound above is a uniform statement — it holds simultaneously for
every hypothesis in the class, which is what a worst-case argument over a
supremum requires. That worst case is almost never realised by the
particular hypothesis a learning algorithm returns, so the bounds are loose
by orders of magnitude in practice: a VC bound for a modestly sized neural
network routinely exceeds $1$, a statement about a probability that can't
be true.

What the bounds are good for is qualitative. The gap shrinks like
$1/\sqrt n$, it grows with a single measure of model richness, and
regularisation — restricting $\mathcal{H}$ or penalising complexity —
directly controls the gap these bounds describe. None of that says which of
ten ridge penalties to ship. For that, a practitioner replaces a bound on
the risk with an estimate of it.

## k-fold cross-validation

Partition the data into $k$ roughly equal folds. For each fold, fit on the
other $k-1$ folds and evaluate on the held-out one; the cross-validated
estimate is the average of the $k$ held-out errors, with every point scored
exactly once by a model that never saw it during training.

Each fold trains on $n(k-1)/k$ points rather than $n$, so if more training
data lowers the true risk — the typical case — cross-validation is biased
upward as an estimate of the risk of the model you'll actually deploy. The
bias shrinks as $k$ grows. Variance moves the other way: the $k$ training
sets overlap heavily, so their held-out estimates are correlated, and
averaging correlated quantities reduces variance less than averaging
independent ones would. At $k=n$ — leave-one-out — every training set
differs from every other by exactly one point, the maximum possible
overlap, so leave-one-out has the smallest bias of any $k$ and among the
largest variance. Values of $k$ between 5 and 10 are the standard
compromise: little enough bias to trust, few enough folds to avoid
leave-one-out's variance and its $n$-fold cost.

## The leave-one-out shortcut for linear smoothers

Refitting $n$ times is the literal definition of leave-one-out, and for a
linear smoother — any estimator of the form $\hat y = Hy$ for a matrix $H$
that doesn't depend on $y$, which covers ridge regression and ordinary
least squares — it's also unnecessary. Every leave-one-out residual is
available from a single fit:

$$
y_i - \hat y_i^{(-i)} = \frac{r_i}{1 - h_{ii}} ,
$$

where $r_i = y_i - \hat y_i$ is the ordinary residual and $h_{ii}$ is the
$i$-th diagonal entry of $H$, the point's leverage. The identity holds
because refitting after replacing $y_i$ with the leave-one-out prediction
reproduces that same prediction — a fixed point of the linear smoother —
and solving that fixed-point equation for the prediction gives the formula
directly. One fit and the diagonal of $H$ replace $n$ refits.

## Nested cross-validation and selection bias

Using the same $k$-fold split to choose a hyperparameter, such as the
$\lambda$ that minimises cross-validated error, and then reporting that
minimum as the model's expected performance double-counts the data:
$\lambda$ was chosen because it scored well on exactly the same folds that
report the score, and some $\lambda$ looks better than it is purely from
cross-validation's own sampling noise. The reported number is
optimistically biased, and the bias grows with the number of
hyperparameters compared, for the same union-bound reason a larger
$\mathcal{H}$ widens the finite-class bound earlier in this lesson.

Nested cross-validation separates the two roles. An outer $k$-fold split
estimates generalisation performance; within each outer training fold, an
inner cross-validation loop selects the hyperparameter using only that
fold's data, before the outer fold is scored on data the inner loop never
saw. The outer score estimates the risk of the entire fit-and-select
procedure, not the risk of whichever hyperparameter got lucky.

## Learning curves

A learning curve plots training and validation error against training set
size. Training error typically rises with $n$, since fitting more points
exactly gets harder; validation error falls; both trend toward the same
asymptote as $n \to \infty$. The gap between them at a given $n$ is an
empirical stand-in for the generalisation gap this lesson opened with.

The shape is a diagnostic. Curves that have converged to a high error
together mean more data won't help — only a richer hypothesis class or
better features close the gap. A gap still wide at the largest available
$n$ means the model overfits relative to the data on hand, and more data,
stronger regularisation, or a smaller hypothesis class all help.

## Cost and scaling

Let $n$, $d$, and $k$ be the sample count, feature count, and fold count.

| Procedure | Cost | Notes |
|---|---|---|
| $k$-fold cross-validation, linear model | $O(nd^2)$ | Same order as one fit; the $k$ folds partition the work. |
| Leave-one-out by brute force | $O(n^2d^2)$ | $n$ refits, each $O(nd^2)$. |
| Leave-one-out by the hat-matrix shortcut | $O(nd^2)$ | One fit; the rest reads off a diagonal. |
| Nested cross-validation, $k$ outer, $k'$ inner, $m$ hyperparameters | $O(kk'm \cdot nd^2)$ | Multiplies selection cost by the outer loop. |

The shortcut is what makes leave-one-out affordable past a handful of
points; without it, its $n$-fold cost is why 5- or 10-fold cross-validation
is the default.

## Where this breaks

**Non-i.i.d. data.** Every bound and cross-validation scheme here assumes
exchangeable data. Time series and grouped data violate it: a random
$k$-fold split leaks future information into training folds, or splits
correlated points across train and test, and both inflate the apparent
accuracy. Forward-chaining splits and group-aware folds are the standard
fixes.

**Small $n$ with many hyperparameters.** Nested cross-validation needs
enough data for the inner loop to select sensibly and the outer loop to
score reliably; with $n$ in the tens, both loops run on samples too small
for either estimate to be trustworthy.

**Reporting cross-validation error as a substitute for a true test set.**
Once a project has cross-validated its way through several rounds of
feature engineering, the folds have played the role a held-out set plays
after repeated reuse in lesson 1: informing choices, and so no longer
measuring generalisation on data those choices never saw.

## The exercise

You build the estimation tools this lesson introduces, plus the Monte
Carlo estimator for Rademacher complexity, and use them to reproduce its
two quantitative claims.

Write `k_fold_indices`, which partitions `range(n)` into `k` folds, and
`cross_val_score`, which uses it to compute the $k$ held-out mean squared
errors for a `fit`/`predict` pair. Write `loo_linear_smoother`, which
computes ridge regression's leave-one-out mean squared error through the
hat-matrix shortcut derived above, and `learning_curve`, which reports
training and validation error across a sequence of training sizes. Write
`empirical_rademacher`, which estimates $\hat{\mathfrak{R}}_S(\mathcal{F})$
by averaging the supremum over hypotheses of $\frac1n \sigma \cdot f$
across many random sign vectors $\sigma$.

The checks confirm `k_fold_indices` partitions the data with no point
repeated or dropped, that `loo_linear_smoother` matches brute-force
leave-one-out refitting to near machine precision, that
`empirical_rademacher` is close to zero for a single hypothesis, grows
sharply for a class of hundreds, and reaches its maximum of $1$ for a class
that shatters the sample, and that `cross_val_score`, swept over a grid of
ridge penalties on data with a known best penalty, selects one whose true
held-out risk beats both no regularisation and heavy over-regularisation by
a wide margin.
