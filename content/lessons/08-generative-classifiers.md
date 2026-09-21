---
title: Generative classifiers
slug: 08-generative-classifiers
part: "Part 2 — Classical machine learning"
summary: Model p(x | y) instead of p(y | x), and the Gaussian version turns out to be logistic regression in disguise.
minutes: 45
gpu: false
objectives:
  - Derive the shared-covariance GDA posterior and show it's exactly logistic in x.
  - State how an unequal-covariance model changes the decision boundary from a hyperplane to a quadratic surface.
  - State the maximum likelihood estimates for the class means, the pooled covariance, and the priors.
  - State the conditional independence assumption behind naive Bayes, and derive its estimator, including Laplace smoothing.
  - Explain why a correctly specified generative model needs less data than a discriminative one, and why that reverses when the model is wrong.
prereqs:
  - 05-logistic-regression
exercise: 08-generative-classifiers
---

# Generative classifiers

Logistic regression models $p(y \mid x)$ and nothing else: fit the
parameters of one conditional distribution, and the training data's
marginal distribution over $x$ never enters the objective. That's a
deliberate economy. It costs you the ability to answer any question
that isn't "what is the label" — what a typical input from class $k$ looks
like, for instance, or how to detect an input unlike anything the classes
generated.

A generative classifier answers a harder question first. It models the
joint distribution $p(x, y) = p(x \mid y)p(y)$, fitting a distribution over
$x$ for each class separately, then obtains $p(y \mid x)$ from Bayes' rule
as a byproduct. This lesson builds the two classical instances of this
idea — Gaussian discriminant analysis and naive Bayes. It also proves the
fact that makes them worth knowing alongside logistic regression. Under a
shared covariance, the Gaussian generative model's posterior is the same
logistic function logistic regression fits directly, with $w$ and $b$
given in closed form. The two methods agree on the hypothesis class and differ only
in how they find a point in it, and that difference decides how much data
each one needs.

## What you already know

- Bayes' rule, $p(y \mid x) = p(x \mid y)p(y)/p(x)$, and that $p(x)$ is the
  same normalising constant across every class.
- The multivariate Gaussian log-density,
  $\log\mathcal{N}(x;\mu,\Sigma) = -\tfrac12(x-\mu)^{\top}\Sigma^{-1}(x-\mu)
  - \tfrac12\log\det\Sigma - \tfrac{d}{2}\log 2\pi$.
- The sigmoid $\sigma(z) = 1/(1+e^{-z})$, and that logistic regression fits
  $p(y=1\mid x) = \sigma(w^{\top}x+b)$ by maximum likelihood.
- Maximum likelihood estimation: write the log-likelihood, differentiate,
  and set the gradient to zero.

## Generative versus discriminative

Both families choose a parameterised $p(y \mid x)$ and fit it by maximum
likelihood on i.i.d. data, but they factorise the likelihood differently:

$$
\text{discriminative:}\ \max_\theta \sum_i \log p_\theta(y_i \mid x_i)
\qquad
\text{generative:}\ \max_\theta \sum_i \log p_\theta(x_i \mid y_i) + \log
p_\theta(y_i) .
$$

The generative objective is the harder modelling problem. It commits to a
distribution over $x$, which the discriminative objective never has to get
right. But it's the easier optimisation problem when the class of
$p(x\mid y)$ is chosen well: the Gaussian discriminant analysis estimates
below are sums and averages in closed form. Logistic regression's maximum
likelihood estimate needs an iterative solver instead.

## Gaussian discriminant analysis

Take $y \in \{0, 1\}$ and model each class-conditional density as Gaussian
with a covariance shared between the classes: $p(x\mid y=k) =
\mathcal{N}(x;\mu_k,\Sigma)$ and $p(y=k) = \pi_k$, with $\pi_0+\pi_1=1$.
Sharing $\Sigma$ is the assumption that does the work below.

Applying Bayes' rule and taking a log-odds, the posterior is affine in $x$:

$$
\log\frac{p(y=1\mid x)}{p(y=0\mid x)} = w^{\top}x + b, \qquad
w = \Sigma^{-1}(\mu_1-\mu_0), \qquad
b = \log\frac{\pi_1}{\pi_0} - \tfrac12\mu_1^{\top}\Sigma^{-1}\mu_1 +
\tfrac12\mu_0^{\top}\Sigma^{-1}\mu_0 .
$$

This falls out of expanding the two Gaussian log-densities. The quadratic
term $x^{\top}\Sigma^{-1}x$ and the log-determinant appear in both classes
and cancel, because $\Sigma$ is the same matrix in numerator and
denominator. Only terms linear in $x$ remain. An affine log-odds is
exactly the statement $p(y=1\mid x) = \sigma(w^{\top}x+b)$, so GDA with a
shared covariance and logistic regression fit the same functional form.
They generally land on different $w$ and $b$, because they get there by
different estimation procedures. One reaches them by two Gaussian fits and
a subtraction, the other by direct conditional maximum likelihood.

Drop the shared-covariance assumption and let each class keep its own
$\Sigma_k$. The quadratic terms no longer cancel, so the log-odds becomes a
quadratic function of $x$. The decision boundary is a general conic — an
ellipse, a parabola, or a hyperbola — instead of a hyperplane. This is
quadratic discriminant analysis. It has strictly more expressive power than
GDA and strictly more parameters to estimate — $K$ full covariance matrices
instead of one. It pays for that expressiveness in variance whenever $n$
isn't large relative to $Kd^2$.

## The maximum likelihood estimates

Fitting $\mu_k$, the shared $\Sigma$, and $\pi_k$ by maximum likelihood on
$n$ i.i.d. examples, with $n_k$ examples of class $k$, gives closed-form
estimates for all three:

$$
\hat{\mu}_k = \frac1{n_k}\sum_{i:y_i=k} x_i, \qquad
\hat{\pi}_k = \frac{n_k}{n}, \qquad
\hat{\Sigma} = \frac1n \sum_k \sum_{i:y_i=k}
(x_i-\hat{\mu}_k)(x_i-\hat{\mu}_k)^{\top} .
$$

Each class mean is the sample mean of its own examples, because only that
class's likelihood terms depend on it. Each prior is the class frequency,
the maximiser of $\sum_k n_k\log\pi_k$ subject to the priors summing to
one. The shared covariance pools the centred outer products across every
class and divides by the total count $n$, not $n-K$. That makes it a
biased estimate of the population covariance, for the same reason the
maximum likelihood variance of a single Gaussian is biased: the estimate
is centred on $\hat{\mu}_k$ rather than the true $\mu_k$.

## Naive Bayes

A full covariance costs $O(d^2)$ parameters, unaffordable once $d$ is in
the thousands — a bag-of-words vocabulary, for instance. Naive Bayes buys
back the cost with an assumption instead of a restriction on the family:
the features are conditionally independent given the class,
$p(x\mid y=k) = \prod_j p(x_j\mid y=k)$.

This assumption is essentially always false — "credit" and "card" are far
from independent even conditioned on the label. But it collapses one
joint distribution over $\mathbb{R}^d$ into $d$ independent univariate
ones. Gaussian naive Bayes is the diagonal-covariance case of the model
above: fit an independent $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$ to feature
$j$ within class $k$, using only that class's data for that feature:

$$
\hat{\mu}_{kj} = \frac1{n_k}\sum_{i:y_i=k} x_{ij}, \qquad
\hat{\sigma}_{kj}^2 = \frac1{n_k}\sum_{i:y_i=k}(x_{ij}-\hat{\mu}_{kj})^2 .
$$

Unlike GDA's pooled $\hat{\Sigma}$, nothing here is shared across classes,
so an outlier class can't distort another class's model.

## The multinomial case and Laplace smoothing

For count data — word counts, canonically — model $x\mid y=k$ as a
multinomial over a vocabulary of size $V$ with parameter vector $\theta_k$.
Maximising the log-likelihood gives $\hat\theta_{kj} = c_{kj} / \sum_j
c_{kj}$, where $c_{kj}$ is the total count of word $j$ across class $k$'s
documents.

That estimate assigns exactly zero probability to a word absent from class
$k$'s training documents. One such word in a test document sends the
entire class score to $-\infty$, regardless of every other word's
evidence. Laplace smoothing adds a pseudocount $\alpha$ to every entry
before normalising:

$$
\hat\theta_{kj} = \frac{c_{kj}+\alpha}{\sum_j c_{kj} + \alpha V} .
$$

This is the posterior mean of $\theta_k$ under a symmetric Dirichlet prior
with concentration $\alpha$ — smoothing is a Bayesian estimate wearing a
numerical fix's clothes. It guarantees every word gets positive
probability in every class, at the cost of biasing rare words toward
uniform.

Conditional independence being false doesn't stop the classifier from
working. Naive Bayes only needs the class ranking to come out right, not
the class probabilities. Its overconfidence from double-counting
correlated words tends to be roughly symmetric across classes. As a
result, the ranking survives even though the probabilities it's computed
from shouldn't be trusted.

## Why the model you believe wins with little data

Both GDA and logistic regression converge, as $n \to \infty$, to their
respective population optima. But they get there at different rates and
land in different places depending on whether the Gaussian assumption is
true.

When the data really is two shared-covariance Gaussians, GDA's parameters
are the maximum likelihood estimate of a correctly specified model. Its
induced $(w, b)$ reach the Bayes-optimal boundary faster than logistic
regression's direct estimate. That's because GDA also uses the structure
of $x$'s marginal distribution that logistic regression discards. With few
examples relative to the number of features, that head start matters.

When the assumption is false, GDA still converges, but to the best
Gaussian-with-shared-covariance approximation of the truth. That
boundary's bias is fixed by the misspecification, and it doesn't shrink
with $n$. Logistic regression converges more slowly, but to the
Bayes-optimal boundary itself, because a linear log-odds is all it ever
assumed. The true $p(x\mid y)$ never needs to look Gaussian for that
assumption to hold at the boundary. With few examples, reach for the model
whose assumptions you believe; with many, the destination matters more
than the speed.

## Cost and scaling

Every estimate here is closed form; none needs an iterative solver.

| Quantity | Cost | Memory |
|---|---|---|
| GDA fit ($\hat{\mu}_k$, $\hat{\Sigma}$, $\hat{\pi}_k$) | $O(nd + nd^2)$ | $O(Kd + d^2)$ |
| GDA predict, $n'$ points | $O(d^3 + n'd^2)$ | $O(d^2)$ |
| Gaussian naive Bayes fit and predict | $O(nd)$ and $O(n'Kd)$ | $O(Kd)$ |
| Multinomial naive Bayes fit and predict | $O(\operatorname{nnz}(X))$ and $O(n'Kd)$ | $O(KV)$ |

GDA pays $O(d^3)$ once, to factorise the shared covariance, then reuses
that factorisation for every prediction rather than inverting $\Sigma$
directly. Naive Bayes never forms a $d \times d$ matrix at all. That's the
entire point of the independence assumption, and the reason it scales to
vocabularies where $d^2$ wouldn't fit in memory.

## Where this breaks

**Non-Gaussian features.** Categorical or heavily skewed features violate
the Gaussian assumption directly. Categorical naive Bayes, with a
multinomial or Bernoulli likelihood per feature, is the standard fix; GDA
has no analogous fix short of transforming the features or leaving the
family.

**Singular or ill-conditioned $\hat{\Sigma}$.** Once $d$ approaches $n_k$
for the smallest class, $\hat{\Sigma}$ is rank deficient, and the
factorisation the posterior needs fails outright. Shrinkage toward a
multiple of the identity restores invertibility at the cost of a biased
covariance, the same trade ridge regression makes.

**Correlated features under naive Bayes.** Duplicating a feature doesn't
change the true class-conditional distribution, but naive Bayes treats the
duplicate as extra evidence, double-counting it and shifting the decision
boundary.

**Outliers under maximum likelihood.** Both $\hat{\mu}_k$ and
$\hat{\Sigma}$ are sample moments: a far-away point moves a mean linearly
in its distance and a covariance quadratically. A discriminative model
whose loss saturates for confidently correct points, like logistic
regression, is far less disturbed by a point already on the right side of
the boundary. The exercise measures this directly.

## The exercise

You build shared-covariance Gaussian discriminant analysis end to end,
then both flavours of naive Bayes.

Implement `fit_gda`, which returns the class means, the pooled covariance,
and the priors. Implement `gda_log_posterior`, which evaluates the class
posterior from those three quantities using a Cholesky factorisation
rather than an explicit matrix inverse. Finally, implement
`gda_to_logistic`, which extracts the equivalent logistic $(w, b)$ from a
two-class GDA model using the derivation above. Then implement
`fit_naive_bayes_gaussian`, the diagonal-covariance estimator, and
`fit_naive_bayes_multinomial` with Laplace smoothing.

The checks verify each maximum likelihood estimate against a direct
computation from the data. They also confirm that `gda_to_logistic`
reproduces `gda_log_posterior`'s decision function to machine precision,
and confirm that the multinomial estimator's smoothed probabilities
normalise. The last two checks run the small-sample comparison the
previous section predicts. Averaged over many training sets, GDA beats a
directly fit logistic regression when the data really is two
shared-covariance Gaussians. GDA loses to it once a handful of correctly
classified outliers drag the Gaussian estimates away from the truth
without moving the Bayes-optimal boundary at all.
