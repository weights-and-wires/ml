---
title: Generative classifiers
slug: 08-generative-classifiers
part: "Part 2 — Classical machine learning"
summary: Model p(x | y)p(y) instead of p(y | x), and discover that the Gaussian version reduces exactly to logistic regression.
minutes: 50
gpu: false
objectives:
  - Derive the shared-covariance GDA posterior and show it is exactly logistic in x.
  - Extend the derivation to unequal covariances and show the boundary becomes quadratic.
  - Derive the maximum likelihood estimates for the class means, the pooled covariance, and the priors.
  - State the conditional independence assumption behind naive Bayes and derive its estimator, including Laplace smoothing.
  - Explain the asymptotic argument for why a correctly specified generative model is more sample efficient than a discriminative one, and why that reverses when the model is wrong.
prereqs:
  - 05-logistic-regression
exercise: 08-generative-classifiers
---

# Generative classifiers

Logistic regression models $p(y \mid x)$ and nothing else: fit the parameters
of one conditional distribution, and the training data's marginal
distribution over $x$ never enters the objective. That is a deliberate
economy, and it costs you the ability to answer any question that is not
"what is the label", such as what a typical input from class $k$ looks like,
or how to detect an input unlike anything the classes generated.

A generative classifier answers a harder question first. It models the joint
distribution $p(x, y) = p(x \mid y) p(y)$, fitting a distribution over $x$ for
each class separately, then obtains $p(y \mid x)$ from Bayes' rule as a
byproduct. This lesson develops the two classical instances of this idea —
Gaussian discriminant analysis and naive Bayes — and proves the fact that
makes them worth knowing alongside logistic regression rather than instead of
it: under a shared covariance, the Gaussian generative model's posterior is
the same logistic function logistic regression fits directly, with $w$ and
$b$ given in closed form. The two methods agree on the hypothesis class and
differ only in how they choose a point in it, and that difference has
consequences for how much data each one needs.

## What you already know

- Bayes' rule, $p(y \mid x) = p(x \mid y) p(y) / p(x)$, and that $p(x)$ is the
  same normalising constant across every class.
- The multivariate Gaussian density and its log form,
  $\log \mathcal{N}(x; \mu, \Sigma) = -\tfrac{1}{2}(x-\mu)^{\top}\Sigma^{-1}(x-\mu)
  - \tfrac{1}{2}\log\det\Sigma - \tfrac{d}{2}\log 2\pi$.
- The sigmoid $\sigma(z) = 1/(1+e^{-z})$ and that logistic regression fits
  $p(y=1\mid x) = \sigma(w^{\top}x + b)$ by maximum likelihood.
- Maximum likelihood estimation: write the log-likelihood, differentiate, and
  set the gradient to zero.
- A Lagrange multiplier turns a maximisation subject to $\sum_i \theta_i = 1$
  into an unconstrained one.

## Generative versus discriminative

Both families choose a parameterised $p(y \mid x)$ and fit it by maximum
likelihood on i.i.d. data. They differ in the factorisation:

$$
\text{discriminative:} \quad \max_{\theta} \sum_i \log p_\theta(y_i \mid x_i)
\qquad
\text{generative:} \quad \max_{\theta} \sum_i \log p_\theta(x_i \mid y_i) + \log p_\theta(y_i) .
$$

The generative objective is a strictly harder modelling problem — it commits
to a distribution over $x$, which the discriminative objective never has to
get right — but it is also a much easier optimisation problem when the class
of $p(x \mid y)$ is chosen well: the two Gaussian discriminant analysis
estimates below are sums and averages, computed in closed form, while
logistic regression's maximum likelihood estimate requires an iterative
solver with no closed form.

## Gaussian discriminant analysis

Take $y \in \{0, 1\}$ and model each class-conditional density as Gaussian
with a covariance shared between the classes:

$$
p(x \mid y=k) = \mathcal{N}(x; \mu_k, \Sigma), \qquad p(y=k) = \pi_k ,
$$

for $k \in \{0, 1\}$, with $\pi_0 + \pi_1 = 1$. Sharing $\Sigma$ is the
assumption that does the work in what follows; without it, the model is
quadratic discriminant analysis, covered below.

### The posterior is logistic

Apply Bayes' rule and take a log-odds:

$$
\log \frac{p(y=1\mid x)}{p(y=0\mid x)}
= \log \frac{\pi_1}{\pi_0} + \log \frac{\mathcal{N}(x;\mu_1,\Sigma)}{\mathcal{N}(x;\mu_0,\Sigma)} .
$$

Expand the Gaussian log-densities. The quadratic term $x^{\top}\Sigma^{-1}x$
and the log-determinant appear in both and cancel, because $\Sigma$ is the
same matrix in numerator and denominator:

$$
\log \frac{\mathcal{N}(x;\mu_1,\Sigma)}{\mathcal{N}(x;\mu_0,\Sigma)}
= -\tfrac{1}{2}(x-\mu_1)^{\top}\Sigma^{-1}(x-\mu_1) + \tfrac{1}{2}(x-\mu_0)^{\top}\Sigma^{-1}(x-\mu_0) .
$$

Expanding both quadratic forms and cancelling the shared $x^{\top}\Sigma^{-1}x$
term leaves

$$
(\mu_1 - \mu_0)^{\top}\Sigma^{-1}x - \tfrac{1}{2}\mu_1^{\top}\Sigma^{-1}\mu_1 + \tfrac{1}{2}\mu_0^{\top}\Sigma^{-1}\mu_0 .
$$

So the log-odds is affine in $x$:

$$
\log \frac{p(y=1\mid x)}{p(y=0\mid x)} = w^{\top}x + b ,
\qquad
w = \Sigma^{-1}(\mu_1 - \mu_0),
\qquad
b = \log\frac{\pi_1}{\pi_0} - \tfrac{1}{2}\mu_1^{\top}\Sigma^{-1}\mu_1 + \tfrac{1}{2}\mu_0^{\top}\Sigma^{-1}\mu_0 .
$$

An affine log-odds is exactly the statement $p(y=1\mid x) = \sigma(w^{\top}x+b)$:
solving $\log\frac{p}{1-p} = z$ for $p$ gives $p = \sigma(z)$ by definition. GDA
with a shared covariance and logistic regression fit the same functional form
for $p(y\mid x)$. They can, and generally do, land on different $w$ and $b$,
because they get there by different estimation procedures — one by two
Gaussian fits and a sum, the other by direct conditional maximum likelihood
— and the rest of this lesson is about when that difference matters.

### Unequal covariances: quadratic discriminant analysis

Drop the shared-covariance assumption and let each class keep its own
$\Sigma_k$. The quadratic terms in the log-odds no longer cancel:

$$
\log \frac{p(y=1\mid x)}{p(y=0\mid x)}
= -\tfrac{1}{2}x^{\top}(\Sigma_1^{-1} - \Sigma_0^{-1})x + (\Sigma_1^{-1}\mu_1 - \Sigma_0^{-1}\mu_0)^{\top}x + c ,
$$

for a constant $c$ collecting the remaining terms. This is a quadratic form in
$x$, so the decision boundary $\{x : p(y=1\mid x) = \tfrac{1}{2}\}$ is a
general conic — an ellipse, a parabola, or a hyperbola, depending on the sign
pattern of the eigenvalues of $\Sigma_1^{-1} - \Sigma_0^{-1}$ — rather than a
hyperplane. QDA has strictly more expressive power than GDA and strictly more
parameters to estimate: $K$ full covariance matrices instead of one, which
costs $O(Kd^2)$ parameters against $O(d^2)$, and pays for that expressiveness
in variance whenever $n$ is not large relative to $Kd^2$.

## The maximum likelihood estimates

Fit $\mu_k$, $\Sigma$ (shared), and $\pi_k$ by maximising the joint
log-likelihood over $n$ i.i.d. examples with $n_k$ examples of class $k$:

$$
\ell(\mu, \Sigma, \pi) = \sum_{i=1}^n \log \pi_{y_i}
+ \sum_{i=1}^n \Big[ -\tfrac{1}{2}(x_i-\mu_{y_i})^{\top}\Sigma^{-1}(x_i-\mu_{y_i}) - \tfrac12\log\det\Sigma \Big] + \text{const} .
$$

**Class means.** Only the terms with $y_i = k$ depend on $\mu_k$. Differentiate
that sum and set it to zero:

$$
\nabla_{\mu_k} \ell = \Sigma^{-1}\sum_{i: y_i=k}(x_i - \mu_k) = 0
\quad\Longrightarrow\quad
\hat{\mu}_k = \frac{1}{n_k}\sum_{i: y_i = k} x_i .
$$

Each class mean is the sample mean of its own examples, independent of
$\Sigma$.

**Priors.** The prior terms are $\sum_k n_k \log \pi_k$, maximised subject to
$\sum_k \pi_k = 1$. With a multiplier $\lambda$,

$$
\frac{\partial}{\partial \pi_k}\Big[\sum_k n_k \log\pi_k - \lambda\big(\textstyle\sum_k \pi_k - 1\big)\Big] = \frac{n_k}{\pi_k} - \lambda = 0
\quad\Longrightarrow\quad \pi_k = n_k/\lambda .
$$

Summing over $k$ gives $\lambda = n$, so $\hat{\pi}_k = n_k / n$, the class
frequency.

**Shared covariance.** Substitute $\hat\mu_k$ and collect the terms in
$\Sigma$. Using $\operatorname{tr}(A^{\top}BA) = \operatorname{tr}(BAA^{\top})$
to rewrite the quadratic forms as a trace lets you apply the standard
identities $\nabla_\Sigma \log\det\Sigma = \Sigma^{-1}$ and
$\nabla_\Sigma \operatorname{tr}(\Sigma^{-1}S) = -\Sigma^{-1}S\Sigma^{-1}$:

$$
\ell(\Sigma) = -\tfrac{n}{2}\log\det\Sigma - \tfrac{1}{2}\operatorname{tr}\Big(\Sigma^{-1}\sum_i (x_i - \hat\mu_{y_i})(x_i-\hat\mu_{y_i})^{\top}\Big) + \text{const} .
$$

Writing $S = \sum_i (x_i-\hat\mu_{y_i})(x_i-\hat\mu_{y_i})^{\top}$, the
gradient is $-\tfrac{n}{2}\Sigma^{-1} + \tfrac12\Sigma^{-1}S\Sigma^{-1}$, which
vanishes at

$$
\hat\Sigma = \frac{1}{n}\sum_{k}\sum_{i: y_i=k} (x_i - \hat\mu_k)(x_i-\hat\mu_k)^{\top} = \frac{S}{n} .
$$

The maximum likelihood covariance pools the centred outer products across
every class and divides by the total count $n$, not $n - K$: it is a biased
estimate of the population covariance, in exactly the way the maximum
likelihood variance of a single Gaussian is biased, and for the same reason —
the estimate is centred on $\hat\mu_k$ rather than the true $\mu_k$, which
makes the residuals slightly too small.

## Naive Bayes

Modelling a full covariance costs $O(d^2)$ parameters, which is unaffordable
once $d$ is in the thousands — a bag-of-words vocabulary, for instance. Naive
Bayes buys back the cost with an assumption instead of a restriction on the
family: the features are conditionally independent given the class,

$$
p(x \mid y=k) = \prod_{j=1}^d p(x_j \mid y=k) .
$$

This assumption is essentially always false — the words "credit" and "card"
are far from independent even conditioned on the label — but it collapses the
estimation problem from one joint distribution over $\mathbb{R}^d$ to $d$
independent univariate distributions, each fit from every example regardless
of what any other feature did. Gaussian naive Bayes is the diagonal-covariance
case of the model above: fit an independent $\mathcal{N}(\mu_{kj},
\sigma_{kj}^2)$ to feature $j$ within class $k$, and both its MLEs fall out of
the single-Gaussian MLE applied feature by feature and class by class:

$$
\hat\mu_{kj} = \frac{1}{n_k}\sum_{i: y_i=k} x_{ij},
\qquad
\hat\sigma_{kj}^2 = \frac{1}{n_k}\sum_{i: y_i=k} (x_{ij} - \hat\mu_{kj})^2 .
$$

Unlike GDA's pooled $\hat\Sigma$, nothing is shared across classes here: class
$k$'s variance for feature $j$ uses only class $k$'s data, so an outlier class
cannot distort another class's model.

### The multinomial case, and Laplace smoothing

For count data — word counts, in the canonical case — model $x \in \mathbb{N}^V$
over a vocabulary of size $V$ as $x \mid y=k$ drawn from a multinomial with
parameter vector $\theta_k \in \Delta^{V-1}$, so that
$p(x \mid y=k) \propto \prod_j \theta_{kj}^{x_j}$. The log-likelihood
contribution of class $k$, as a function of $\theta_k$ alone and dropping the
multinomial coefficient, which does not depend on $\theta_k$, is

$$
\sum_{i: y_i=k} \sum_j x_{ij} \log \theta_{kj} = \sum_j c_{kj} \log\theta_{kj},
\qquad c_{kj} = \sum_{i: y_i=k} x_{ij} ,
$$

the total count of word $j$ across every document in class $k$. Maximising
$\sum_j c_{kj}\log\theta_{kj}$ subject to $\sum_j \theta_{kj} = 1$ is the same
Lagrangian computation as the class priors above, and gives
$\hat\theta_{kj} = c_{kj} / \sum_j c_{kj}$.

That estimate assigns exactly zero probability to any word absent from class
$k$'s training documents, and one such word in a test document sends the
entire class score to $-\infty$ regardless of every other word's evidence.
**Laplace smoothing** adds a pseudocount $\alpha$ to every entry before
normalising:

$$
\hat\theta_{kj} = \frac{c_{kj} + \alpha}{\sum_j c_{kj} + \alpha V} .
$$

This is the posterior mean of $\theta_k$ under a symmetric
$\operatorname{Dirichlet}(\alpha, \ldots, \alpha)$ prior — smoothing is a Bayes
estimate wearing a numerical fix's clothes — and it guarantees every word gets
positive probability in every class, at the cost of biasing rare words toward
uniform.

### Why the false assumption still works

Naive Bayes is a decision rule, $\hat y(x) = \operatorname*{arg\,max}_k
\big[\log\pi_k + \sum_j \log p(x_j\mid y=k)\big]$, and an arg max only needs
the *ranking* of the class scores to be right, not their absolute values.
Conditional independence is false for text, so the estimated
$p(x\mid y=k)$ is a poor estimate of the true class-conditional density — it
double-counts correlated words as though they were independent evidence, and
the resulting scores are badly overconfident. But that overconfidence is
close to symmetric across classes when the correlation structure among
features is similar within each class, so the arg max survives even though
the probabilities it is computed from do not. This is also why naive Bayes
probability outputs are calibrated poorly and should not be trusted as
probabilities, even on problems where its predictions are accurate.

The other half of the answer is sample size. Full multinomial modelling of a
document over a vocabulary of size $V$ needs $O(2^V)$ parameters to capture
every interaction; the naive assumption needs $O(V)$. With $V$ in the tens of
thousands and training documents in the thousands, the full model is not
merely inefficient, it has no hope of being estimated at all, while the naive
one is a sum away.

## The asymptotic argument

Both GDA and logistic regression converge, as $n \to \infty$, to their
respective population optima. Two facts from the statistics of M-estimators
decide how fast, and to what:

- When the generative model's assumptions hold — the data really is a mixture
  of Gaussians with a shared covariance — GDA's parameters are the maximum
  likelihood estimate of a correctly specified low-dimensional model, and its
  induced $(w, b)$ converge to the Bayes-optimal decision boundary at the
  parametric rate $O(1/\sqrt n)$ with a smaller constant than logistic
  regression's direct estimate of the same boundary, because GDA also uses the
  marginal structure of $x$ that logistic regression discards. Ng and
  Jordan's analysis (2001) makes this precise: the discriminative model needs
  $O(d)$ examples to approach its asymptote, the generative one needs
  $O(\log d)$, so with few examples relative to the dimension, the generative
  model has already covered more of the distance to its limit.
- When the assumptions do not hold, GDA still converges — to the best
  Gaussian-with-shared-covariance approximation of the truth, which need not
  be the Bayes-optimal boundary at all. Its bias is fixed by the
  misspecification and does not shrink with $n$. Logistic regression converges
  more slowly, at the same $O(1/\sqrt n)$ rate but with the discriminative
  model's larger constant, yet its limit is the Bayes-optimal boundary itself,
  because a linear log-odds is all logistic regression ever assumed, and nothing
  about the true $p(x\mid y)$ needs to match a Gaussian for that assumption to
  hold at the boundary.

The practical reading: with few examples, reach for the model whose
assumptions you believe, because it gets you closer, faster, to wherever it is
headed. With many examples, the destination matters more than the speed, and a
discriminative model that only commits to the shape of the boundary — rather
than the entire generating process — is the safer bet.

## Cost and scaling

Every estimate below is closed form; nothing here needs an iterative solver.

| Quantity | Cost | Memory |
|---|---|---|
| GDA fit ($\hat\mu_k$, $\hat\Sigma$, $\hat\pi_k$) | $O(nd + nd^2)$ | $O(Kd + d^2)$ |
| GDA predict, $n'$ points | $O(d^3 + n'd^2)$ (factorise $\Sigma$ once, then a triangular solve per point) | $O(d^2)$ |
| Gaussian naive Bayes fit and predict | $O(nd)$ and $O(n'Kd)$ | $O(Kd)$ |
| Multinomial naive Bayes fit and predict | $O(\operatorname{nnz}(X))$ and $O(n'Kd)$ (or $O(n' \operatorname{nnz})$ sparse) | $O(KV)$ |

GDA pays $O(d^3)$ once, to factorise the shared covariance, then reuses that
factorisation for every prediction — the same Cholesky-and-triangular-solve
pattern as the linear regression lesson, and for the same numerical reason:
never invert $\Sigma$ directly. Naive Bayes never forms a $d\times d$ matrix at
all, which is the entire point of the independence assumption and the reason
it scales to vocabularies where $d^2$ would not fit in memory.

## Where this breaks

**Non-Gaussian features.** Categorical or heavily skewed features violate the
Gaussian assumption directly. Categorical naive Bayes (a multinomial or
Bernoulli likelihood per feature) is the standard fix; for GDA there is no
analogous fix short of transforming the features or leaving the family.

**Singular or ill-conditioned $\hat\Sigma$.** Once $d$ approaches $n_k$ for the
smallest class, $\hat\Sigma$ is rank deficient or numerically singular, and the
Cholesky factorisation the posterior needs fails outright. Shrinkage —
blending $\hat\Sigma$ with a multiple of the identity — restores invertibility
at the cost of a biased covariance, the same variance-for-bias trade ridge
regression makes.

**Correlated features under naive Bayes.** Duplicating a feature does not
change the true class-conditional distribution, but naive Bayes treats the
duplicate as independent new evidence and effectively double-counts it,
shifting the decision boundary. Highly correlated feature blocks are the
naive assumption's worst case.

**Outliers under maximum likelihood.** Both $\hat\mu_k$ and $\hat\Sigma$ are
sample moments, and a single far-away point moves a mean linearly in its
distance and a covariance quadratically. A discriminative model whose loss
saturates for confidently correct points — logistic regression's included —
is far less disturbed by a point that is already on the right side of the
boundary, however far it sits from the class centre. The exercise measures
this difference directly.

**Class imbalance.** $\hat\pi_k = n_k/n$ is an unbiased estimate of the prior
under the sampling scheme that generated the training set, which is not
always the deployment scheme. If the training data over- or under-samples a
class relative to production, correcting $\hat\pi_k$ before applying Bayes'
rule is necessary and easy to forget, because both GDA and naive Bayes bury
the prior inside a single joint computation.

## The exercise

You build shared-covariance Gaussian discriminant analysis end to end, then
both flavours of naive Bayes. Implement `fit_gda`, which returns the class
means, the pooled covariance, and the priors; `gda_log_posterior`, which
evaluates the class posterior from those three quantities using a Cholesky
factorisation rather than an explicit matrix inverse; and `gda_to_logistic`,
which extracts the equivalent logistic $(w, b)$ from a two-class GDA model
using the derivation above. Then implement `fit_naive_bayes_gaussian`, the
diagonal-covariance estimator, and `fit_naive_bayes_multinomial` with Laplace
smoothing.

The checks verify each maximum likelihood estimate against a direct
computation from the data, confirm that `gda_to_logistic` reproduces
`gda_log_posterior`'s decision function to machine precision, and confirm
that the multinomial estimator's smoothed probabilities normalise. The last
two checks run the small-sample comparison the asymptotic argument predicts:
averaged over many training sets, GDA beats a directly fit logistic
regression when the data really is two shared-covariance Gaussians, and loses
to it once a handful of correctly classified outliers drag the Gaussian
estimates away from the truth without moving the Bayes-optimal boundary at
all.
