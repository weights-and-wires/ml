---
title: Generalisation and model selection
slug: 07-model-selection
part: "Part 2 — Classical machine learning"
summary: Generalisation bounds, why they are too loose to pick a model, then cross-validation, the leave-one-out shortcut, and learning curves as the tools that do.
minutes: 50
gpu: false
objectives:
  - State the finite-class and Rademacher generalisation bounds, with their constants, and explain what each is for.
  - Derive the closed-form leave-one-out residual for a linear smoother from the hat matrix.
  - Explain the bias-variance trade-off in k as k ranges from 2 to n.
  - Explain why hyperparameter selection without a nested split is optimistically biased.
prereqs:
  - 01-the-learning-problem
exercise: 07-model-selection
---

# Generalisation and model selection

Every model in this course minimises an empirical risk $\hat R(h) = \frac1n
\sum_i \ell(h(x_i), y_i)$. Nothing in that minimisation refers to the risk
that matters, $R(h) = \mathbb{E}_{(x,y)}[\ell(h(x), y)]$ over the true data
distribution. Lesson 1 introduced the **generalisation gap** $R(h) -
\hat R(h)$ as the quantity that separates the two, and used a held-out set to
estimate it once. This lesson asks how large that gap can be in the worst
case, as a function of the hypothesis class's richness and the sample size;
and, since the answer turns out to be too loose to act on directly, what a
practitioner uses instead.

The bounds are not a detour from the practical half that follows. They are
the reason $k$-fold cross-validation looks the way it does, why leave-one-out
has a variance problem despite using almost all the data every time, and why
skipping a nested split silently inflates a reported accuracy.

## What you already know

- Empirical risk minimisation, and that $\hat R(h)$ evaluated on the data
  used to choose $h$ is an optimistic estimate of $R(h)$.
- Hoeffding's inequality: for i.i.d. $Z_1, \dots, Z_n \in [0,1]$ with mean
  $\mu$, $P(|\bar Z - \mu| \geq \epsilon) \leq 2 e^{-2n\epsilon^2}$.
- A union bound: $P(\bigcup_j A_j) \leq \sum_j P(A_j)$.
- The ridge regression hat matrix $H = X(X^{\top}X + \lambda I)^{-1}X^{\top}$
  and the fitted values $\hat y = Hy$.
- The bias-variance decomposition of a risk estimate into a systematic
  offset and a spread across resamples of the data.

## A finite-class bound from Hoeffding and a union bound

Fix a hypothesis class $\mathcal{H}$ with $|\mathcal{H}| = M$ elements and a
loss bounded in $[0,1]$. For a single, fixed $h$, Hoeffding's inequality
applied to the i.i.d. losses $\ell(h(x_i), y_i)$ gives

$$
P\bigl(|\hat R(h) - R(h)| \geq \epsilon\bigr) \leq 2 e^{-2n\epsilon^2} .
$$

The catch is that a learning algorithm does not evaluate one fixed $h$: it
picks the $h \in \mathcal{H}$ that minimises $\hat R$, so the bound has to
hold simultaneously for every hypothesis the algorithm might have chosen. The
union bound over the $M$ events $\{|\hat R(h) - R(h)| \geq \epsilon\}$ gives

$$
P\Bigl(\exists h \in \mathcal{H} : |\hat R(h) - R(h)| \geq \epsilon\Bigr)
\leq 2M e^{-2n\epsilon^2} .
$$

Setting the right side to $\delta$ and solving for $\epsilon$: with
probability at least $1 - \delta$, every $h \in \mathcal{H}$ satisfies

$$
|\hat R(h) - R(h)| \leq \sqrt{\frac{\log(2M/\delta)}{2n}} .
$$

The gap shrinks at the rate $O(1/\sqrt{n})$ and grows only logarithmically in
the size of the class, which is already the qualitative shape every bound in
this lesson takes. It is also useless the moment $\mathcal{H}$ is infinite,
which every parametric model in this course has been.

## Rademacher complexity

**Rademacher complexity** replaces $\log M$ with a quantity defined directly
on the function class and, crucially, on the sample. For a class $\mathcal{F}$
of functions bounded in $[-1,1]$ and a sample $S = (z_1, \dots, z_n)$,

$$
\hat{\mathfrak{R}}_S(\mathcal{F})
= \mathbb{E}_{\sigma} \left[ \sup_{f \in \mathcal{F}} \frac1n \sum_{i=1}^{n}
\sigma_i f(z_i) \right] ,
$$

where $\sigma_1, \dots, \sigma_n$ are independent, uniform $\pm 1$ signs. It
measures how well the class can correlate with pure noise: a class of one
function scores close to $0$, since $\mathbb{E}[\sigma_i] = 0$ regardless of
$f$, while a class rich enough to match every sign pattern scores $1$, its
maximum, because some $f \in \mathcal{F}$ agrees with $\sigma$ on every draw.

The population version $\mathfrak{R}_n(\mathcal{F}) =
\mathbb{E}_S[\hat{\mathfrak{R}}_S(\mathcal{F})]$ controls the generalisation
gap through a **symmetrisation** argument. Introduce an independent ghost
sample $S'$ with the same distribution as $S$:

$$
\mathbb{E}_S \left[ \sup_f \bigl(\mathbb{E}[f] - \hat{\mathbb{E}}_S[f]\bigr)
\right]
\leq \mathbb{E}_{S,S'} \left[ \sup_f \bigl(\hat{\mathbb{E}}_{S'}[f] -
\hat{\mathbb{E}}_S[f]\bigr) \right]
= \mathbb{E}_{S,S',\sigma} \left[ \sup_f \frac1n \sum_i \sigma_i
\bigl(f(z_i') - f(z_i)\bigr) \right] \leq 2 \mathfrak{R}_n(\mathcal{F}) .
$$

The middle equality holds because swapping $z_i$ and $z_i'$ for every $i$
with $\sigma_i = -1$ leaves the joint distribution of $(S, S')$ unchanged, so
multiplying each paired difference by an independent sign changes nothing in
expectation; the final step splits the supremum of a difference, by the
triangle inequality, into two copies of $\hat{\mathfrak{R}}_S$. With
probability at least $1-\delta$, for every $f \in \mathcal{F}$,

$$
\mathbb{E}[f] \leq \hat{\mathbb{E}}_S[f] + 2\mathfrak{R}_n(\mathcal{F}) +
\sqrt{\frac{\log(1/\delta)}{2n}} .
$$

Unlike the finite-class bound, this one is distribution-dependent — computed
under the data's actual distribution, not a worst case — and its empirical
version $\hat{\mathfrak{R}}_S$ can be estimated directly from a batch of
data, which is exactly what this lesson's exercise does by Monte Carlo.

## VC dimension and the growth function

For a binary hypothesis class, the relevant measure of richness is
combinatorial. The **growth function** $\Pi_{\mathcal{H}}(n)$ is the largest
number of distinct label patterns $\mathcal{H}$ can produce on any $n$
points:

$$
\Pi_{\mathcal{H}}(n) = \max_{x_1, \dots, x_n} \bigl| \{ (h(x_1), \dots,
h(x_n)) : h \in \mathcal{H} \} \bigr| .
$$

Since there are only $2^n$ possible label patterns, $\Pi_{\mathcal{H}}(n)
\leq 2^n$ always. The **VC dimension** $d_{VC}$ is the largest $n$ for which
equality holds — the largest sample size $\mathcal{H}$ can **shatter**, that
is, label every possible way. The Sauer-Shelah lemma says that once $n$
exceeds $d_{VC}$, growth stops being exponential and becomes polynomial:

$$
\Pi_{\mathcal{H}}(n) \leq \sum_{i=0}^{d_{VC}} \binom{n}{i} \leq \left(
\frac{en}{d_{VC}} \right)^{d_{VC}} \qquad \text{for } n \geq d_{VC} .
$$

Substituting the growth function in place of $M$ in a union-bound argument
like the one above, and optimising the resulting bound, gives a
generalisation gap of order $O\bigl(\sqrt{d_{VC} \log(n/d_{VC}) / n}\bigr)$,
and inverting it gives the standard **sample complexity** result: to hold the
gap to $\epsilon$ with confidence $1-\delta$ takes

$$
n = O\!\left( \frac{d_{VC} + \log(1/\delta)}{\epsilon^2} \right)
$$

samples. This lesson does not derive that bound in full; the shape is the
point. Richness enters through one number, $d_{VC}$, and the sample size
needed to control the gap scales linearly in it.

## Why these bounds cannot pick a model

Every bound in this lesson is a **uniform** statement: it holds
simultaneously for every hypothesis in the class, which is what a worst-case
argument over a supremum requires. That worst case is almost never realised
by the particular hypothesis a learning algorithm returns, so the bounds are
loose by orders of magnitude in practice — a VC bound for a modestly sized
neural network routinely exceeds $1$, a vacuous statement about a
probability. They are also computed for the worst distribution consistent
with the assumptions, never the one the data actually came from.

What they are good for is qualitative: the generalisation gap shrinks like
$1/\sqrt{n}$, it grows with a single measure of model richness, and
regularisation — restricting $\mathcal{H}$ or penalising complexity — is not
a heuristic but the thing these bounds say directly controls the gap. None of
that says which of ten ridge penalties to ship. For that, a practitioner
replaces a bound on the risk with an estimate of it.

## k-fold cross-validation

Partition the data into $k$ roughly equal folds. For each fold $j$, fit on
the other $k-1$ folds and evaluate on fold $j$; the cross-validated estimate
is the average of the $k$ held-out errors. Every point is scored exactly once,
by a model that never saw it during training.

Each fold trains on $n(k-1)/k$ points rather than $n$. If the learning curve
of the method is decreasing — more training data lowers the true risk, the
typical case — then a model fit on fewer points has higher risk than one fit
on all $n$, so cross-validation is **biased upward** as an estimate of the
risk of the model you will actually deploy, and the bias shrinks as $k$
grows, since each training fold then holds a larger fraction of the data.

The **variance** moves the other way. The $k$ held-out estimates come from
training sets that overlap in $n(k-2)/k$ points out of $n(k-1)/k$, so they
are correlated, and averaging correlated quantities reduces variance less
than averaging $k$ independent ones would. At $k = n$ — **leave-one-out**
(LOO) — every training set differs from every other by exactly one point,
the maximum possible overlap, so the $n$ LOO estimates are about as
correlated as they can be. LOO has the smallest bias of any $k$, since each
fold trains on $n-1$ of $n$ points, and among the largest variance, since
that near-total overlap means the average of $n$ estimates behaves like the
average of far fewer independent ones. Values of $k$ between $5$ and $10$ are
the standard compromise: little enough bias to trust, few enough folds to
avoid LOO's variance and its $n$-fold cost.

## The leave-one-out shortcut for linear smoothers

Refitting $n$ times is the literal definition of leave-one-out, and for a
**linear smoother** — any estimator of the form $\hat y = Hy$ for a matrix
$H$ that does not depend on $y$, which covers ridge regression, ordinary
least squares, and kernel ridge regression — it is also unnecessary.

Fix $\lambda$ and let $H = X(X^{\top}X + \lambda I)^{-1}X^{\top}$, so
$\hat y = Hy$ and the residual is $r = y - \hat y$. Write $\hat y_i^{(-i)}$
for the fitted value at $x_i$ from a model trained on every point except $i$.
Construct a modified target vector $y^{(i)}$, equal to $y$ except that its
$i$-th entry is replaced by $t := \hat y_i^{(-i)}$. The coefficient vector
that minimises the (possibly ridge-penalised) squared loss on $(X, y^{(i)})$
is exactly $\hat\beta^{(-i)}$, the leave-one-out coefficient: at
$\beta = \hat\beta^{(-i)}$ the $i$-th term contributes zero loss and a zero
gradient, since $x_i^{\top}\hat\beta^{(-i)} = t$ by definition, so the
stationarity condition for the full, modified problem reduces to the
stationarity condition for the original leave-one-out problem, which has the
unique solution $\hat\beta^{(-i)}$.

Because $H$ does not depend on $y$, applying the same smoother to $y^{(i)}$
therefore reproduces $t$ at coordinate $i$:

$$
t = \bigl(H y^{(i)}\bigr)_i = \sum_{j \neq i} H_{ij} y_j + H_{ii} \, t .
$$

Solving for $t$, using $\sum_j H_{ij} y_j = \hat y_i$,

$$
t (1 - H_{ii}) = \hat y_i - H_{ii} y_i
\quad\Longrightarrow\quad
y_i - t = \frac{y_i - \hat y_i}{1 - H_{ii}} = \frac{r_i}{1 - h_{ii}} ,
$$

writing $h_{ii}$ for the $i$-th diagonal entry of $H$, the point's
**leverage**. Every leave-one-out residual is recovered from one fit and the
diagonal of $H$, at the cost of forming $H$ once rather than refitting $n$
times.

## Nested cross-validation and selection bias

Using the same $k$-fold split to choose a hyperparameter, such as the
$\lambda$ that minimises cross-validated error, and then reporting that
minimum as the model's expected performance double-counts the data: $\lambda$
was chosen because it scored well on exactly the folds now reporting the
score, and cross-validation's own sampling noise means some $\lambda$ looks
better than it is purely by chance. The reported number is therefore
optimistically biased, and the bias grows with the number of hyperparameters
compared, for the same union-bound reason a larger $\mathcal{H}$ widens the
finite-class bound earlier in this lesson.

**Nested cross-validation** separates the two roles. An outer $k$-fold split
estimates generalisation performance; within each outer training fold, an
inner cross-validation loop selects the hyperparameter using only that
fold's data, before the outer fold is scored on data the inner loop never
saw. The outer score now estimates the risk of the entire fit-and-select
procedure, not the risk of whichever hyperparameter got lucky.

## Learning curves

A **learning curve** plots training and validation error against training
set size. Training error typically rises with $n$, since fitting more points
exactly gets harder; validation error falls; both trend toward the same
asymptote as $n \to \infty$. The gap between them at a given $n$ is an
empirical stand-in for the generalisation gap this lesson opened with, and it
shrinks at the same qualitative $O(1/\sqrt{n})$ rate the bounds predict.

The shape is a diagnostic. Curves that have converged to a high error
together mean more data will not help — only a richer hypothesis class or
better features close the gap. A gap still wide at the largest available $n$
means the model overfits relative to the data on hand, and more data,
stronger regularisation, or a smaller hypothesis class all help.

## Cost and scaling

Let $n$, $d$, and $k$ be the sample count, feature count, and fold count.

| Procedure | Cost | Notes |
|---|---|---|
| $k$-fold cross-validation, linear model | $O(nd^2)$ | Same order as one fit; the $k$ folds partition the work. |
| Leave-one-out by brute force | $O(n^2 d^2)$ | $n$ refits, each $O(nd^2)$. |
| Leave-one-out by the hat-matrix shortcut | $O(nd^2)$ | One fit; the rest reads off a diagonal. |
| Nested cross-validation, $k$ outer, $k'$ inner, $m$ hyperparameters | $O(k k' m \cdot nd^2)$ | Multiplies selection cost by the outer loop. |

The shortcut is what makes leave-one-out affordable past a handful of points;
without it, its $n$-fold cost is why $5$- or $10$-fold cross-validation is
the default.

## Where this breaks

**Non-i.i.d. data.** Every bound and cross-validation scheme here assumes
exchangeable data. Time series and grouped data violate it: a random
$k$-fold split leaks future information into training folds, or splits
correlated points across train and test, and both inflate the apparent
accuracy. Forward-chaining splits and group-aware folds are the standard
fixes.

**Small $n$ with many hyperparameters.** Nested cross-validation needs enough
data for the inner loop to select sensibly and the outer loop to score
reliably; with $n$ in the tens, both loops run on samples too small for
either estimate to be trustworthy.

**Reporting cross-validation error as a substitute for a true test set.**
Once a project has cross-validated its way through several rounds of feature
engineering, the folds have played the role a held-out set plays after
repeated reuse in lesson 1: informing choices, and so no longer measuring
generalisation on data those choices never saw.

## The exercise

You build the estimation tools this lesson introduces, plus the Monte Carlo
estimator for Rademacher complexity, and use them to reproduce its two
quantitative claims.

Write `k_fold_indices`, which partitions `range(n)` into `k` folds, and
`cross_val_score`, which uses it to compute the $k$ held-out mean squared
errors for a `fit`/`predict` pair. Write `loo_linear_smoother`, which
computes ridge regression's leave-one-out mean squared error through the
hat-matrix shortcut derived above, and `learning_curve`, which reports
training and validation error across a sequence of training sizes. Write
`empirical_rademacher`, which estimates $\hat{\mathfrak{R}}_S(\mathcal{F})$ by
averaging the supremum over hypotheses of $\frac1n \sigma \cdot f$ across many
random sign vectors $\sigma$.

The checks confirm `k_fold_indices` partitions the data with no point
repeated or dropped, that `loo_linear_smoother` matches brute-force
leave-one-out refitting to near machine precision, that
`empirical_rademacher` is close to zero for a single hypothesis, grows
sharply for a class of hundreds, and reaches its maximum of $1$ for a class
that shatters the sample, and that `cross_val_score`, swept over a grid of
ridge penalties on data with a known best penalty, selects one whose true
held-out risk beats both no regularisation and heavy over-regularisation by a
wide margin.
