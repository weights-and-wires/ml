---
title: The learning problem
slug: 01-the-learning-problem
part: "Part 1 — Foundations"
summary: The supervised setup, risk versus empirical risk, and the bias-variance decomposition, derived in full.
minutes: 45
gpu: false
objectives:
  - State the supervised learning problem under i.i.d. sampling, and define risk, empirical risk, and empirical risk minimisation precisely.
  - Prove that the training estimate of risk is biased downward, and identify the source of the bias.
  - Derive the bias-variance decomposition of squared error at a point, including why the cross term vanishes.
  - Split excess risk into an approximation term and an estimation term, and explain what a held-out set buys and what it costs.
exercise: 01-the-learning-problem
---

# The learning problem

A model that fits the data you gave it is not yet a model that works. Every
method in this course — a linear fit, a kernel machine, a transformer — is an
answer to a single question: given a finite sample, how do you choose a
function that performs well on inputs you have not seen? "Performs well on the
sample" is easy to state and easy to satisfy, including by memorising the
sample outright. "Performs well in general" is the thing you actually want,
and it is not observable from the sample alone.

This lesson sets up the machinery the rest of the course reuses without
redefining it: the data-generating distribution, the risk of a hypothesis, the
empirical risk you can actually compute, and the gap between them. That gap
has two faces. One is a systematic optimism — training error underestimates
true error, and the proof of that fact is three lines. The other is a
variance-versus-bias trade-off that governs every model-capacity decision you
will make later: ridge's penalty strength, a tree's depth, a network's width.
Both are consequences of the same fact, that the hypothesis you fit and the
data you fit it on are not independent.

## What you already know

- Expectation is linear, and for a random variable $Z$ its variance is
  $\operatorname{Var}(Z) = \mathbb{E}[(Z - \mathbb{E}Z)^2] = \mathbb{E}[Z^2] - (\mathbb{E}Z)^2$.
- Independent, identically distributed (i.i.d.) sampling means each draw comes
  from the same fixed distribution, independently of every other draw.
- The law of large numbers: an average of i.i.d. terms with finite variance
  converges to its expectation as the number of terms grows.
- $\mathbb{E}[UV] = \mathbb{E}[U]\mathbb{E}[V]$ when $U$ and $V$ are independent.
- $\lVert \cdot \rVert$ denotes the Euclidean norm unless stated otherwise.

## The supervised setup

Fix an input space $\mathcal{X} \subseteq \mathbb{R}^d$ and an output space
$\mathcal{Y}$. There is a joint distribution $P$ over $\mathcal{X} \times
\mathcal{Y}$ that you do not observe. You observe a training set
$S = \{(x_i, y_i)\}_{i=1}^n$ drawn i.i.d. from $P$. Everything that follows
depends on that i.i.d. assumption; break it — by drift, by feedback from your
own predictions, by a sampling scheme that favours some inputs — and the
guarantees below stop applying, though the algorithms will still run and
produce numbers.

A **hypothesis class** $\mathcal{H}$ is a set of candidate functions
$h : \mathcal{X} \to \mathcal{Y}'$, where $\mathcal{Y}'$ is $\mathcal{Y}$
itself for classification or a superset such as $\mathbb{R}$ for regression
with a score. A **loss function** $\ell : \mathcal{Y}' \times \mathcal{Y} \to
\mathbb{R}_{\geq 0}$ scores a prediction against a label; $\ell(h(x), y) = (h(x)
- y)^2$ for squared loss, $\ell(h(x), y) = \mathbb{1}[h(x) \neq y]$ for
zero-one loss.

The **risk** of $h$ is its expected loss under the true distribution:

$$
R(h) = \mathbb{E}_{(X, Y) \sim P}\big[\ell(h(X), Y)\big] .
$$

$R$ is the quantity you care about and cannot compute, because $P$ is unknown.
The **empirical risk** on a sample $S$ is the one quantity you can compute:

$$
\hat{R}_n(h) = \frac{1}{n} \sum_{i=1}^{n} \ell(h(x_i), y_i) .
$$

For a fixed $h$ — one chosen before looking at $S$, or independently of it —
each summand $\ell(h(x_i), y_i)$ is an independent draw from the same
distribution, so $\hat{R}_n(h)$ is an average of i.i.d. terms with mean
$R(h)$:

$$
\mathbb{E}_S[\hat{R}_n(h)] = \frac{1}{n}\sum_{i=1}^n \mathbb{E}[\ell(h(x_i), y_i)] = R(h) .
$$

$\hat{R}_n(h)$ is an unbiased estimator of $R(h)$, for any single, fixed $h$.
Keep that qualifier in view; it is the entire content of the next section.

## Empirical risk minimisation

You do not evaluate one fixed hypothesis. You search a class $\mathcal{H}$ for
the one that fits the sample best:

$$
\hat{h} = \operatorname*{arg\,min}_{h \in \mathcal{H}} \hat{R}_n(h) .
$$

This is **empirical risk minimisation** (ERM), and it is the algorithmic
content of least squares, logistic regression, and the loss functions that
train every network in this course. Its justification is the law of large
numbers: for each fixed $h$, $\hat{R}_n(h) \to R(h)$ as $n \to \infty$, so
minimising a good proxy for $R$ over a class ought to find something close to
$\operatorname*{arg\,min}_{h \in \mathcal{H}} R(h)$.

The word "ought" is doing work. $\hat{h}$ is not fixed — it is chosen using
$S$, to fit $S$ — so the unbiasedness argument above does not apply to
$\hat{R}_n(\hat{h})$. That failure is not a technicality.

## Training risk is optimistic

Write $h^\star = \operatorname*{arg\,min}_{h \in \mathcal{H}} R(h)$ for the
best hypothesis in the class under the true risk. $h^\star$ is fixed: it
depends on $P$ and $\mathcal{H}$, not on the sample $S$. Because $\hat{h}$
minimises $\hat{R}_n$ over all of $\mathcal{H}$, and $h^\star \in \mathcal{H}$,

$$
\hat{R}_n(\hat{h}) \leq \hat{R}_n(h^\star) \qquad \text{for every draw of } S .
$$

Take expectations over $S$ on both sides. The right side is the empirical risk
of a fixed hypothesis, so the unbiasedness result applies directly:
$\mathbb{E}_S[\hat{R}_n(h^\star)] = R(h^\star)$. And because $h^\star$ minimises
the true risk over $\mathcal{H}$ while $\hat{h} \in \mathcal{H}$ for every
draw of $S$, $R(h^\star) \leq R(\hat{h})$ pointwise, hence also in expectation.
Chaining the three inequalities:

$$
\mathbb{E}_S[\hat{R}_n(\hat{h})] \;\leq\; \mathbb{E}_S[\hat{R}_n(h^\star)] \;=\; R(h^\star) \;\leq\; \mathbb{E}_S[R(\hat{h})] .
$$

So $\mathbb{E}_S[\hat{R}_n(\hat{h})] \leq \mathbb{E}_S[R(\hat{h})]$: on average,
the training risk of the hypothesis you actually picked underestimates its own
true risk. The mechanism is search, not any property of a particular loss or
model: $\hat{h}$ was chosen, among other reasons, because it happened to look
good on the very sample used to score it. The gap $\mathbb{E}_S[R(\hat{h})] -
\mathbb{E}_S[\hat{R}_n(\hat{h})]$ is the **generalisation gap**, and it grows
with the richness of $\mathcal{H}$: a larger class gives ERM more ways to find
a hypothesis that fits sampling noise in $S$ specifically, which is exactly the
freedom that makes the two inequalities above loose.

## The bias-variance decomposition

The optimism result above holds for any loss. Squared loss admits a sharper,
pointwise decomposition that separates two distinct causes of error. Fix a
query point $x$ and suppose $y = f(x) + \varepsilon$, where $f(x) =
\mathbb{E}[Y \mid X = x]$ and $\varepsilon$ is mean-zero noise with variance
$\sigma^2$, independent of the training set $S$. Let $\hat{f}$ be a predictor
fit on $S$; $\hat{f}(x)$ is a random variable through the randomness of $S$.

First, separate the label noise from the fit. Because $\varepsilon$ is
independent of $S$ and hence of $\hat{f}(x)$, and $\mathbb{E}[\varepsilon] = 0$:

$$
\mathbb{E}\big[(y - \hat{f}(x))^2\big]
= \mathbb{E}\big[(f(x) - \hat{f}(x) + \varepsilon)^2\big]
= \mathbb{E}\big[(f(x) - \hat{f}(x))^2\big]
+ 2\,\mathbb{E}\big[(f(x)-\hat{f}(x))\big]\,\mathbb{E}[\varepsilon]
+ \mathbb{E}[\varepsilon^2] .
$$

The cross term vanishes because $\mathbb{E}[\varepsilon] = 0$, leaving

$$
\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \mathbb{E}_S\big[(f(x) - \hat{f}(x))^2\big] + \sigma^2 . \tag{1}
$$

Second, split the remaining term over the randomness of $S$. Let $\bar{f}(x) =
\mathbb{E}_S[\hat{f}(x)]$ be the average prediction over training sets. Add and
subtract $\bar{f}(x)$:

$$
\mathbb{E}_S\big[(f(x) - \hat{f}(x))^2\big]
= \mathbb{E}_S\big[\big((f(x) - \bar{f}(x)) + (\bar{f}(x) - \hat{f}(x))\big)^2\big] .
$$

Expand the square. The first factor is a constant with respect to $S$, so the
cross term is

$$
2\big(f(x) - \bar{f}(x)\big)\,\mathbb{E}_S\big[\bar{f}(x) - \hat{f}(x)\big]
= 2\big(f(x) - \bar{f}(x)\big) \cdot 0 = 0 ,
$$

because $\mathbb{E}_S[\hat{f}(x)] = \bar{f}(x)$ by definition. What survives is

$$
\mathbb{E}_S\big[(f(x) - \hat{f}(x))^2\big] = \underbrace{(f(x) - \bar{f}(x))^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_S\big[(\hat{f}(x) - \bar{f}(x))^2\big]}_{\text{variance}} . \tag{2}
$$

Combining (1) and (2):

$$
\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \operatorname{Bias}[\hat{f}(x)]^2 + \operatorname{Var}[\hat{f}(x)] + \sigma^2 . \tag{3}
$$

Every term is nonnegative, and $\sigma^2$ does not depend on $\hat{f}$ at all:
it is the noise floor no estimator can remove. Bias measures how far the
average fit is from the truth — driven by the hypothesis class being too
restrictive to represent $f$. Variance measures how much the fit moves from
one training set to another — driven by the class being flexible enough to
chase sampling noise. Restricting $\mathcal{H}$ (fewer polynomial terms,
smaller trees, a stronger ridge penalty) trades variance for bias; enlarging it
trades the other way. No choice of $\mathcal{H}$ or $n$ moves $\sigma^2$.

## Approximation and estimation

The same trade-off appears at the level of the whole hypothesis class, not
just one predictor. Let $h_{\text{Bayes}}$ be the unconstrained risk minimiser
over all measurable functions, and $h^\star = \operatorname*{arg\,min}_{h \in
\mathcal{H}} R(h)$ as before. The excess risk of $\hat{h}$ splits as

$$
R(\hat{h}) - R(h_{\text{Bayes}})
= \underbrace{\big(R(h^\star) - R(h_{\text{Bayes}})\big)}_{\text{approximation error}}
+ \underbrace{\big(R(\hat{h}) - R(h^\star)\big)}_{\text{estimation error}} .
$$

Approximation error is the cost of restricting the search to $\mathcal{H}$ at
all; it does not depend on $n$ and does not vanish as you collect more data,
because $h^\star$ is fixed by $\mathcal{H}$ and $P$. It plays the role bias
played above. Estimation error is the cost of not knowing $P$ and instead
optimising $\hat{R}_n$ over a finite sample; it shrinks as $n$ grows and plays
the role variance played above. A larger $\mathcal{H}$ can only decrease
approximation error — it contains everything the smaller class contained — and
can only increase estimation error, because ERM now has more room to fit noise.
Choosing $\mathcal{H}$, or a regularisation strength that acts like choosing
$\mathcal{H}$, is choosing a point on this trade-off, not eliminating it.

## What a held-out set buys and costs

Section 3 showed $\hat{R}_n(\hat{h})$ is a biased-downward estimate of
$R(\hat{h})$, precisely because $\hat{h}$ and the data scoring it are the same
data. The fix follows from the same argument that made the fixed-hypothesis
case unbiased: partition $S$ into a training set $S_{\text{tr}}$ and a
held-out set $S_{\text{te}}$ that are independent draws, fit $\hat{h}$ on
$S_{\text{tr}}$ only, and evaluate on $S_{\text{te}}$. Relative to
$S_{\text{te}}$, $\hat{h}$ is a fixed hypothesis — it was chosen without
looking at $S_{\text{te}}$ — so

$$
\mathbb{E}_{S_{\text{te}}}\big[\hat{R}_{n_{\text{te}}}(\hat{h})\big] = R(\hat{h})
$$

holds exactly, by the same one-line argument as the fixed-$h$ case. That is
what a held-out set buys: an honest, unbiased estimate of the risk of the
model you actually shipped.

It is not free. First, the estimate has variance
$\operatorname{Var}(\ell(\hat{h}(X), Y)) / n_{\text{te}}$, which shrinks only
as $1/n_{\text{te}}$; a small test set gives an unbiased but noisy answer.
Second, every point moved into $S_{\text{te}}$ is a point removed from
$S_{\text{tr}}$, which increases the estimation error of $\hat{h}$ itself —
you now fit on less data to get an honest score of a worse model. The split
fraction trades the precision of the risk estimate against the quality of the
fitted model, and neither extreme is free: an all-training split gives you a
biased estimate of a good model, an all-test split gives you an unbiased
estimate of a model fit on nothing.

## Cost and scaling

| Quantity | Cost | Scaling |
|---|---|---|
| $\hat{R}_n(h)$ for a fixed $h$ | $n$ loss evaluations | $O(n)$ |
| Train/test split | one shuffle of indices | $O(n)$ |
| Polynomial features, degree $p$ | $n(p+1)$ evaluations | $O(np)$ |
| Least-squares fit in $p+1$ features | one normal-equations solve | $O(np^2 + p^3)$ |
| Bias-variance Monte Carlo, $T$ resampled sets, $q$ query points | $T$ fits plus $T q$ predictions | $O(T(np^2 + p^3 + qp))$ |

The Monte Carlo estimate of bias and variance in the exercise is the most
expensive item here only because it repeats the fit $T$ times; each individual
fit is the same $O(np^2 + p^3)$ least-squares cost as lesson 2 will make cheap
and stable. Doubling the number of resampled training sets halves the Monte
Carlo error in the bias and variance estimates, at the cost of $T$ appearing
linearly in the runtime — a $\sqrt{T}$ return on a $T$ expense, the usual
Monte Carlo trade.

## Where this breaks

**The i.i.d. assumption fails in deployment.** Every result here — the
unbiasedness of $\hat{R}_n(h)$ for fixed $h$, the held-out estimate's
unbiasedness — depends on training and evaluation data coming from the same
distribution as the data the model will face. Distribution shift breaks the
premise, not the arithmetic; a held-out set drawn the same way as training data
is silent about performance under shift.

**Reusing the held-out set for model selection reintroduces the bias.** If you
try many models and pick the one with the best $S_{\text{te}}$ score, $S_{\text{te}}$
is no longer independent of the selected model, and the argument in section 6
no longer applies — $S_{\text{te}}$ has become part of the search, exactly as
$S_{\text{tr}}$ was for $\hat{h}$. This is why a validation set used for
tuning and a test set used once for the final number must be different sets.

**Small held-out sets give unbiased but useless estimates.** An unbiased
estimator with enormous variance answers the right question with the wrong
number. When $n$ is small, $k$-fold cross-validation, covered in lesson 7,
trades some of that variance away by averaging several splits.

**Heavy-tailed losses slow the law of large numbers.** The unbiasedness of
$\hat{R}_n(h)$ never used a variance bound, but its usefulness as an estimate
of $R(h)$ does: a loss with infinite or very large variance needs a much
larger $n$ before $\hat{R}_n(h)$ concentrates near $R(h)$.

## The exercise

You implement the two computable objects this lesson defines and the
experiment that verifies the bias-variance decomposition numerically rather
than trusting equation (3) on faith.

Write `empirical_risk(predictions, targets, loss)`, which averages a
vectorised loss over a batch, and `train_test_split(x, y, test_size, seed)`,
which partitions indices reproducibly. Write `polynomial_features(x, degree)`,
building the design matrix $[1, x, x^2, \dots, x^{\text{degree}}]$ column by
column, and `fit_polynomial(x, y, degree)`, which fits it by least squares.
These four are the tools; the fifth function is the experiment.
`bias_variance_decomposition(true_function, noise_std, x_query, degree,
n_train, n_trials, x_domain, seed)` resamples `n_trials` independent training
sets of size `n_train` from `true_function` plus Gaussian noise, fits a degree-
`degree` polynomial to each, and returns the squared bias, the variance, and
the noise floor at every point in `x_query`, estimated from the empirical
mean and spread of the resulting predictions.

The checks confirm each piece before asking anything of the whole: that
`polynomial_features` builds the right matrix, that `fit_polynomial` recovers
an exact polynomial's coefficients when there is no noise to fight, and that
`train_test_split` partitions every index exactly once. They then confirm the
two facts this lesson exists to establish — that the bias, variance, and noise
your function returns sum to the total expected squared error measured
independently, that bias falls and variance rises as `degree` increases on a
fixed training-set size, and that a model's risk on its own training split
underestimates its risk on a held-out split.
