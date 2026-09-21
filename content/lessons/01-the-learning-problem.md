---
title: The learning problem
slug: 01-the-learning-problem
part: "Part 1 — Foundations"
summary: The supervised setup, risk versus empirical risk, and why training error is an optimistic guide to true performance.
minutes: 35
gpu: false
objectives:
  - State the supervised learning problem under i.i.d. sampling, and define risk, empirical risk, and empirical risk minimisation.
  - Explain why the training estimate of risk is biased downward, and identify where that bias comes from.
  - State the bias-variance decomposition of squared error at a point, and explain what each term means.
  - Explain what a held-out set buys you, and what it costs.
exercise: 01-the-learning-problem
---

# The learning problem

A model that fits the data you gave it is not a model that works. Every method in this course — a linear fit, a kernel machine, a transformer — answers one question. Given a finite sample, how do you choose a function that performs well on inputs you have not seen? Fitting the sample is easy; you can even memorise it outright. Performing well in general is what you want, and you cannot read that off the sample alone.

This lesson sets up the vocabulary the rest of the course reuses: the data-generating distribution, the risk of a hypothesis, the empirical risk you can compute, and the gap between the two. That gap has two faces. Training error is optimistic, underestimating true error for a reason you can state in one sentence. The bias-variance trade-off governs every model-capacity decision you make later, from ridge's penalty strength to a tree's depth to a network's width. Both faces trace back to the same cause: the hypothesis you fit and the data you fit it on are not independent.

## What you already know

- Expectation is linear, and for a random variable $Z$, $\operatorname{Var}(Z) = \mathbb{E}[(Z - \mathbb{E}Z)^2] = \mathbb{E}[Z^2] - (\mathbb{E}Z)^2$.
- Independent, identically distributed (i.i.d.) sampling means each draw comes from the same fixed distribution, independently of every other draw.
- The law of large numbers: an average of i.i.d. terms with finite variance converges to its expectation as the number of terms grows.
- $\mathbb{E}[UV] = \mathbb{E}[U]\mathbb{E}[V]$ when $U$ and $V$ are independent.
- $\lVert \cdot \rVert$ denotes the Euclidean norm unless stated otherwise.

## The supervised setup

Fix an input space $\mathcal{X} \subseteq \mathbb{R}^d$ and an output space $\mathcal{Y}$. A joint distribution $P$ over $\mathcal{X} \times \mathcal{Y}$ generates the data, and you never observe $P$ itself. You observe a training set $S = \{(x_i, y_i)\}_{i=1}^n$ drawn i.i.d. from $P$. Everything below depends on that assumption: break it, through drift, feedback from your own predictions, or a sampling scheme that favours some inputs, and the guarantees stop applying, even though your code keeps producing numbers.

A **hypothesis class** $\mathcal{H}$ is a set of candidate functions $h : \mathcal{X} \to \mathcal{Y}'$, where $\mathcal{Y}'$ is $\mathcal{Y}$ itself for classification, or a superset such as $\mathbb{R}$ for a regression score. A **loss function** $\ell$ scores a prediction against a label. Squared loss is $\ell(h(x), y) = (h(x) - y)^2$; zero-one loss is $\ell(h(x), y) = \mathbb{1}[h(x) \neq y]$.

The **risk** of $h$ is its expected loss under the true distribution:

$$
R(h) = \mathbb{E}_{(X, Y) \sim P}\big[\ell(h(X), Y)\big] .
$$

$R$ is the number you care about, and you cannot compute it, because $P$ is unknown. The **empirical risk** on a sample $S$ is the number you can compute instead:

$$
\hat{R}_n(h) = \frac{1}{n} \sum_{i=1}^{n} \ell(h(x_i), y_i) .
$$

Take a hypothesis $h$ fixed in advance, chosen without looking at $S$. Each term $\ell(h(x_i), y_i)$ is then an independent draw from the same distribution, so $\hat{R}_n(h)$ averages i.i.d. terms with mean $R(h)$:

$$
\mathbb{E}_S[\hat{R}_n(h)] = R(h) .
$$

$\hat{R}_n(h)$ is an unbiased estimate of $R(h)$, but only for a hypothesis you fixed before touching $S$. That qualifier is the entire content of the next section.

## Empirical risk minimisation

You rarely evaluate one fixed hypothesis. You search a class $\mathcal{H}$ for whichever member fits the sample best:

$$
\hat{h} = \operatorname*{arg\,min}_{h \in \mathcal{H}} \hat{R}_n(h) .
$$

This is **empirical risk minimisation** (ERM), the algorithmic content of least squares, logistic regression, and every loss function that trains a network in this course. The law of large numbers justifies it: for each fixed $h$, $\hat{R}_n(h) \to R(h)$ as $n \to \infty$, so minimising a close proxy for $R$ ought to land near the hypothesis that minimises $R$ itself.

"Ought to" is doing work: $\hat{h}$ is not fixed. You chose it using $S$, to fit $S$, so the unbiasedness result above does not cover $\hat{R}_n(\hat{h})$.

## Training risk is optimistic

Write $h^\star = \operatorname*{arg\,min}_{h \in \mathcal{H}} R(h)$ for the best hypothesis in the class under the true risk — fixed, unlike $\hat h$, because it depends on $P$ and $\mathcal{H}$, never on the sample. Because $\hat{h}$ minimises $\hat{R}_n$ over $\mathcal{H}$ and $h^\star$ is a member of $\mathcal{H}$, $\hat{R}_n(\hat{h}) \leq \hat{R}_n(h^\star)$ for every draw of $S$. Take expectations: $\mathbb{E}_S[\hat{R}_n(h^\star)] = R(h^\star)$, since $h^\star$ is fixed, and $R(h^\star) \leq R(\hat h)$ always, since $h^\star$ minimises the true risk over $\mathcal{H}$. Chained together,

$$
\mathbb{E}_S[\hat{R}_n(\hat{h})] \;\leq\; R(h^\star) \;\leq\; \mathbb{E}_S[R(\hat{h})] .
$$

On average, the training risk of the hypothesis you actually picked underestimates its own true risk. The cause is search, not any property of a particular loss or model: $\hat{h}$ was chosen partly because it happened to look good on the very sample used to score it. The gap $\mathbb{E}_S[R(\hat{h})] - \mathbb{E}_S[\hat{R}_n(\hat{h})]$ is the **generalisation gap**, and it grows with the richness of $\mathcal{H}$. A larger class gives ERM more ways to find a hypothesis that fits sampling noise specific to $S$.

## The bias-variance decomposition

Squared loss admits a sharper, pointwise version of the same story. Fix a query point $x$, and suppose $y = f(x) + \varepsilon$, where $f(x) = \mathbb{E}[Y \mid X = x]$ and $\varepsilon$ is mean-zero noise with variance $\sigma^2$, independent of the training set $S$. Let $\hat{f}$ be a predictor fit on $S$, so $\hat{f}(x)$ is random through $S$, and write $\bar{f}(x) = \mathbb{E}_S[\hat{f}(x)]$ for its average across training sets. Split off the label noise, then split the fit's error around its own average:

$$
\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{(f(x) - \bar{f}(x))^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_S\big[(\hat{f}(x) - \bar{f}(x))^2\big]}_{\text{variance}} + \sigma^2 . \tag{1}
$$

Both splits drop a cross term for the same reason: each pairs a quantity that is fixed, once you condition correctly, with something that averages to zero. $\mathbb{E}[\varepsilon] = 0$ kills the noise cross term; $\mathbb{E}_S[\hat{f}(x) - \bar{f}(x)] = 0$, by the definition of $\bar f(x)$ as that average, kills the other.

Every term in equation (1) is nonnegative, and $\sigma^2$ does not depend on $\hat{f}$ at all: it is the noise floor no estimator can remove. Bias measures how far the average fit sits from the truth, driven by a hypothesis class too restrictive to represent $f$. Variance measures how much the fit moves from one training set to another, driven by a class flexible enough to chase sampling noise. Restricting $\mathcal{H}$ — fewer polynomial terms, a shallower tree, a stronger ridge penalty — trades variance for bias. Enlarging $\mathcal{H}$ trades the other way. No choice of $\mathcal{H}$ or $n$ moves $\sigma^2$.

## Held-out risk

The training-risk result and the bias-variance split share a cause: scoring a fit on the data used to produce it flatters the fit. Splitting the sample fixes this, because it restores the one condition the first result needed — a hypothesis scored on data it never touched. Fit $\hat{h}$ on a training portion $S_{\text{tr}}$ only, and evaluate it on an independent held-out portion $S_{\text{te}}$. Relative to $S_{\text{te}}$, $\hat{h}$ is a fixed hypothesis, so

$$
\mathbb{E}_{S_{\text{te}}}\big[\hat{R}_{n_{\text{te}}}(\hat{h})\big] = R(\hat{h})
$$

holds exactly, by the same one-line argument used for a fixed $h$. That is what a held-out set buys: an honest, unbiased estimate of the risk of the model you shipped.

It costs two things. The estimate's variance shrinks only as $1/n_{\text{te}}$, so a small test set is unbiased but noisy. And every point moved into $S_{\text{te}}$ is a point removed from $S_{\text{tr}}$, which worsens the fit itself — an honest score costs you a slightly weaker model. An all-training split gives a biased estimate of a good model; an all-test split gives an unbiased estimate of a model fit on nothing. Neither extreme is free.

## Cost and scaling

The exercise's costs scale with the sample size $n$, the polynomial degree $p$, and the number of resampled sets $T$ used to estimate bias and variance.

| Quantity | Cost | Scaling |
|---|---|---|
| $\hat{R}_n(h)$ for a fixed $h$ | $n$ loss evaluations | $O(n)$ |
| Train/test split | one shuffle of indices | $O(n)$ |
| Polynomial features, degree $p$ | $n(p+1)$ evaluations | $O(np)$ |
| Least-squares fit in $p+1$ features | one normal-equations solve | $O(np^2 + p^3)$ |
| Bias-variance Monte Carlo, $T$ resampled sets, $q$ query points | $T$ fits plus $T q$ predictions | $O(T(np^2 + p^3 + qp))$ |

The Monte Carlo estimate costs the most only because it repeats the fit $T$ times; each fit is the same $O(np^2 + p^3)$ least-squares cost that lesson 2 makes cheap and numerically stable. Quadrupling $T$ halves the Monte Carlo error, at a runtime cost linear in $T$ — a $\sqrt{T}$ return on a $T$ expense, the usual Monte Carlo trade.

## Where this breaks

**The i.i.d. assumption fails in deployment.** Every unbiasedness result above depends on training and evaluation data coming from the same distribution the model will face. Distribution shift breaks that premise, not the arithmetic. A held-out set drawn the same way as the training data says nothing about performance under shift.

**Reusing the held-out set for model selection reintroduces the bias.** Try many models and pick the one with the best score on $S_{\text{te}}$, and $S_{\text{te}}$ is no longer independent of the selected model — it has become part of the search, exactly the way $S_{\text{tr}}$ was for $\hat{h}$. A validation set used for tuning and a test set used once for the final number have to be different sets.

**Small held-out sets give unbiased but useless estimates.** An unbiased estimator with enormous variance answers the right question with the wrong number. When $n$ is small, $k$-fold cross-validation, covered in lesson 7, trades some of that variance away by averaging several splits.

**Heavy-tailed losses slow convergence.** Unbiasedness never needed a variance bound, but usefulness as a stand-in for $R(h)$ does. A loss with infinite or very large variance needs a much larger $n$ before $\hat{R}_n(h)$ settles near $R(h)$.

## The exercise

You implement the two computable objects this lesson defines, and an experiment that checks the bias-variance decomposition numerically rather than trusting equation (1) on faith.

Write `empirical_risk(predictions, targets, loss)`, which averages a vectorised loss over a batch, and `train_test_split(x, y, test_size, seed)`, which partitions indices reproducibly. Write `polynomial_features(x, degree)`, building the design matrix $[1, x, x^2, \dots, x^{\text{degree}}]$ column by column, and `fit_polynomial(x, y, degree)`, which fits it by least squares. These four are the tools. The fifth function is the experiment: `bias_variance_decomposition(true_function, noise_std, x_query, degree, n_train, n_trials, x_domain, seed)` resamples `n_trials` independent training sets of size `n_train` from `true_function` plus Gaussian noise, fits a degree-`degree` polynomial to each, and returns the squared bias, the variance, and the noise floor at every point in `x_query`, estimated from the empirical mean and spread of the resulting predictions.

The checks confirm each piece before asking anything of the whole: that `polynomial_features` builds the right matrix, that `fit_polynomial` recovers an exact polynomial's coefficients when there is no noise to fight, and that `train_test_split` partitions every index exactly once. They then confirm the two facts this lesson exists to establish: that the bias, variance, and noise your function returns sum to the total expected squared error measured independently, that bias falls and variance rises as `degree` increases on a fixed training-set size, and that a model's risk on its own training split underestimates its risk on a held-out split.
