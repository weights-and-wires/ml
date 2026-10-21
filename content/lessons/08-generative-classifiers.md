---
title: Generative classifiers
slug: 08-generative-classifiers
part: "Part 2 — Classical machine learning"
summary: Describe each class, classify with Bayes' rule, and see a shared Gaussian become logistic regression.
minutes: 60
gpu: false
objectives:
  - Compute class means, a shared covariance, and class priors from labelled data.
  - Build a Gaussian classifier that returns log probabilities for every class.
  - Turn a two-class Gaussian model into the weights and bias of a logistic regression.
  - Build Gaussian and word-count naive Bayes, including Laplace smoothing.
  - Explain when a generative model beats logistic regression and when it loses.
prereqs:
  - 05-logistic-regression
exercise: 08-generative-classifiers
---

# Generative classifiers

Logistic regression draws a boundary. It never builds a picture of a typical point in each class. You sometimes want that picture.

A generative classifier learns each class on its own, then asks which class was more likely to produce a new point. With a bell-shaped cloud per class, the boundary is the same kind of straight cut as logistic regression.

You will fit that Gaussian model, turn it into logistic weights, and fit a lighter naive Bayes model for many features, including word counts.

## What you already know

These five ideas are the ones this lesson uses:

- Logistic regression predicts $p(y = 1 \mid x) = \sigma(w^{\top}x + b)$. The sigmoid sends a score of 0 to 0.5.
- Class probabilities are non-negative and add to 1, as with 0.2 and 0.8.
- Bayes' rule turns $p(x \mid y)$ into $p(y \mid x)$ by multiplying by the prior $p(y)$ and dividing by $p(x)$.
- A log keeps a tiny probability usable. A probability of 0.001 has natural log about $-6.9$.
- The mean of the numbers 1 and 3 is $(1 + 3) / 2 = 2$.

## Compare classes with Bayes' rule

Two warehouses ship parcels. Half come from each. At a weight of 3 kg, warehouse 0 has density 0.2 and warehouse 1 has density 0.05.

Unnormalised scores are $0.5 \times 0.2 = 0.1$ and $0.5 \times 0.05 = 0.025$. The total is 0.125. Warehouse 0 gets $0.1 / 0.125 = 0.8$. Warehouse 1 gets 0.2.

That comparison is *Bayes' rule*. For class $k$, with prior $\pi_k = p(y = k)$ and class density $p(x \mid y = k)$,

$$
p(y = k \mid x) = \frac{\pi_k \, p(x \mid y = k)}{\sum_{j} \pi_j \, p(x \mid y = j)} .
$$

The numerator is prior times density. The denominator makes the class probabilities add to 1. In logs, that division is a log-sum-exp:

$$
\log p(y = k \mid x) = \log \pi_k + \log p(x \mid y = k) - \log\sum_j \exp(\log \pi_j + \log p(x \mid y = j)).
$$

The same densities return posteriors 0.8 and 0.2. Numpy subtracts a log-sum-exp from `np.log(priors) + log_density`.

The usual mistake is normalising along the wrong axis, so a row no longer sums to 1.

> [!TIP]
> `np.exp(log_post).sum(axis=1)` is 1 for every row, up to rounding.

Building $p(x \mid y)$ is *generative*. Logistic regression models $p(y \mid x)$ only.

> [!KEY]
> A generative classifier multiplies a prior by a class density. Bayes' rule turns that into $p(y \mid x)$.

## Fit one shared Gaussian cloud

Gaussian discriminant analysis, or GDA, gives every class a bell cloud in $d$ dimensions. Class $k$ has a mean vector $\mu_k$. All classes share one covariance $\Sigma$.

Take one feature and four points. Class 0 has 1 and 3. Class 1 has 5 and 7.

Priors are $2/4 = 0.5$ and $0.5$. Means are $(1+3)/2 = 2$ and $(5+7)/2 = 6$. Gaps from each point's own class mean are $-1, 1, -1, 1$. Squared gaps sum to 4. Divide by $n = 4$ to get $\Sigma = 1$.

Those are the *maximum likelihood estimates*:

$$
\pi_k = \frac{n_k}{n}, \qquad
\mu_k = \frac{1}{n_k}\sum_{i:\, y_i = k} x_i, \qquad
\Sigma = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu_{y_i})(x_i - \mu_{y_i})^{\top} .
$$

$n_k$ is the class count. Divide $\Sigma$ by $n$, not by $n - 1$ or $n_k$. The four points return $\pi = 0.5$, means 2 and 6, and $\Sigma = 1$.

Numpy adds `diff.T @ diff` for `diff = X[y == k] - means[k]`, then divides by `n` once.

The usual mistake is dividing by $n - 1$, by $n - K$, or by each $n_k$.

> [!WARNING]
> Subtract each point's own class mean. The overall mean folds the gap between centres into the spread, and $\Sigma$ comes out too big.

> [!KEY]
> Fitting GDA is counting and averaging. The pooled covariance divides by $n$.

The log density at a point $x$ is

$$
\log p(x \mid y = k) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log\det\Sigma - \frac{1}{2}(x - \mu_k)^{\top}\Sigma^{-1}(x - \mu_k) .
$$

The last term is squared *Mahalanobis distance*. A larger $\det\Sigma$ means a wider cloud.

At $x = 3$ and $d = 1$, the constant is $-\frac12\log(2\pi) \approx -0.9189$, and $\log\det\Sigma = 0$. Class 0 has distance $1$, so the log density is $-0.9189 - 0.5 = -1.4189$. Class 1 has distance $9$, so $-0.9189 - 4.5 = -5.4189$. Add $\log 0.5 \approx -0.6931$ to get $-2.1121$ and $-6.1121$.

Do not invert $\Sigma$. A lower-triangular *Cholesky factor* $L$ satisfies $L L^{\top} = \Sigma$. Solving $L z = x - \mu_k$ makes the Mahalanobis term $z^{\top} z$. And $\log\det\Sigma = 2\sum_i \log L_{ii}$.

Here $\Sigma = 1$, so $L = 1$, $z = x - \mu_k$, and class 0 has $z^2 = 1$. That is the same $-1.4189$.

Numpy calls `cholesky` and `solve_triangular`.

The usual mistake is dropping the $1/2$ on the Mahalanobis term. Unlike the $d$ term, it does not cancel across classes.

The log-sum-exp of $-2.1121$ and $-6.1121$ is about $-2.0939$. Probabilities are about 0.982 and 0.018, so $x = 3$ is class 0.

## Turn two classes into logistic regression

With two classes and one shared $\Sigma$, the quadratic pieces $x^{\top}\Sigma^{-1}x$ cancel. The posterior is a sigmoid of a straight score:

$$
p(y = 1 \mid x) = \sigma(w^{\top}x + b), \qquad
w = \Sigma^{-1}(\mu_1 - \mu_0), \qquad
b = \log\frac{\pi_1}{\pi_0} - \frac{1}{2}\mu_1^{\top}\Sigma^{-1}\mu_1 + \frac{1}{2}\mu_0^{\top}\Sigma^{-1}\mu_0 .
$$

$w$ runs from $\mu_0$ to $\mu_1$, with less weight on a wide direction. The bias holds the log prior ratio and the two centre terms.

On the four points, $w = 4$ and $b = 0 - 18 + 2 = -16$. At $x = 3$ the score is $-4$, and $\sigma(-4) \approx 0.018$, the class-1 posterior. At $x = 4$ the score is 0 and the probability is 0.5.

Numpy is `np.linalg.inv(cov) @ (mu1 - mu0)` for $w$.

The usual mistake is `means[0] - means[1]`, because $w$ must use $\mu_1 - \mu_0$.

> [!KEY]
> Two classes and one shared covariance make GDA a logistic regression. The boundary is a straight cut.

Separate covariances leave a curve, called quadratic discriminant analysis. This lesson shares one covariance.

## Drop the covariance with naive Bayes

Ten thousand words make a covariance of about $10^8$ entries. *Naive Bayes* treats features as independent given the class, so the log density is a sum. The assumption is often false and the probabilities too sharp, but the class ranking is often still useful.

For continuous features, each class and feature gets its own one-dimensional Gaussian:

$$
\mu_{kj} = \text{mean of feature } j \text{ over class } k, \qquad
\sigma^2_{kj} = \frac{1}{n_k}\sum_{i:\, y_i = k}(x_{ij} - \mu_{kj})^2 .
$$

On class 0, with values 1 and 3, the mean is 2 and the sum of squared gaps is 2. Divide by $n_k = 2$ to get variance 1. Dividing by $n = 4$ gives the wrong 0.5. `np.var` divides by $n_k` by default. A constant feature has variance 0, so you floor every variance.

Numpy is `np.maximum(X[y == k].var(axis=0), var_floor)`.

The usual mistake is `ddof=1`, which divides by $n_k - 1$.

Each text row is word counts over $V$ words. A count of 0 would zero the whole row's probability. *Laplace smoothing* adds $\alpha$ to every word:

$$
\theta_{kj} = \frac{c_{kj} + \alpha}{\sum_{j'} c_{kj'} + \alpha V} .
$$

$c_{kj}$ is the total count of word $j$ in class $k$. The denominator adds $\alpha$ once per word.

With $V = 3$ and $\alpha = 1$, counts $[2, 0, 1]$ become $[3, 1, 2]$, total 6, so $\theta_0 = [1/2, 1/6, 1/3]$. The unseen word gets $1/6$.

Numpy is `np.log((counts + alpha) / (counts + alpha).sum())`.

The usual mistake is adding $\alpha$ only in the denominator, or smoothing each row before the sum.

> [!KEY]
> Laplace smoothing adds $\alpha$ to every count before you normalise. No word gets probability 0.

## See when the extra picture helps

Move the point 7 out to 100. The class-1 mean jumps from 6 to $(5 + 100) / 2 = 52.5$, and the boundary moves with it. A logistic gradient only sees $p - y$. A point with probability 0.99 and label 1 contributes 0.01, so it barely pulls.

If the clouds really share one Gaussian covariance, GDA reaches a good boundary with fewer rows. Outliers pull the averages, and more rows do not fix a wrong shape. With little data, trust the assumptions you believe. With many rows, fewer assumptions usually win.

Keep these checks in mind:

> [!RECAP]
> - The log posterior is the log prior plus the log density, minus a log-sum-exp across classes.
> - GDA fits priors $n_k / n$, one mean per class, and one covariance divided by $n$.
> - With Cholesky $L$, the Mahalanobis term is $z^{\top} z$ where $L z = x - \mu_k$, and $\log\det\Sigma = 2\sum_i \log L_{ii}$.
> - For two classes, $w = \Sigma^{-1}(\mu_1 - \mu_0)$, and $b$ adds the log prior ratio to the two centre terms.
> - Gaussian naive Bayes floors a variance per class and feature. Word counts add $\alpha$, then normalise.

## Build it step by step

Build the five functions in the order the checks test them:

1. **`fit_gda(X, y)`**. Takes `X` of shape `(n, d)` and integer labels `y` of shape `(n,)`. Returns `means` `(K, d)`, `cov` `(d, d)`, and `priors` `(K,)`. Set `K = int(y.max()) + 1`. The common mistake is dividing the covariance by anything other than `n`. One class adds this block:

```python
means[k] = X[mask].mean(axis=0)   # shape (d,)
cov += diff.T @ diff               # shape (d, d), divide by n at the end
```

2. **`gda_log_posterior(X, means, cov, priors)`**. Returns shape `(n, K)`. Factor once with `cholesky(cov, lower=True)`, solve `(X - means[k]).T`, and subtract a row log-sum-exp. The common mistake is a missing $1/2$ on the Mahalanobis term. One class looks like this:

```python
z = solve_triangular(L, (X - means[k]).T, lower=True)  # shape (d, n)
maha = np.sum(z ** 2, axis=0)                           # shape (n,)
```

3. **`gda_to_logistic(means, cov, priors)`**. Two classes only. Returns `w` of shape `(d,)` and `b` as a Python float. The common mistake is swapping the class means. The weight uses this difference:

```python
w = np.linalg.inv(cov) @ (means[1] - means[0])  # shape (d,)
```

4. **`fit_naive_bayes_gaussian(X, y, var_floor)`**. Returns `means` `(K, d)`, `variances` `(K, d)`, and `priors` `(K,)`. Use `.var(axis=0)` and `np.maximum` with the floor. The common mistake is `ddof=1`. One class looks like this:

```python
variances[k] = np.maximum(X[mask].var(axis=0), var_floor)  # shape (d,)
```

5. **`fit_naive_bayes_multinomial(X, y, alpha)`**. Takes counts `X` of shape `(n, V)`. Returns `log_theta` `(K, V)` and `log_priors` `(K,)`. Sum the class rows, add `alpha`, divide by the new total, and take `np.log`. The common mistake is smoothing each row before the sum. One class looks like this:

```python
smoothed = X[mask].sum(axis=0) + alpha          # shape (V,)
log_theta[k] = np.log(smoothed / smoothed.sum())
```

## Cost and scaling

Times below use $n$ rows, $d$ features, $K$ classes, and $m$ queries:

| Step | Time | Memory |
|---|---|---|
| Fit GDA | grows like $n d^2$ | $Kd + d^2$ |
| Predict with GDA | $d^3$ once, then $m K d^2$ | $d^2$ |
| Fit Gaussian naive Bayes | grows like $n d$ | $Kd$ |
| Fit word-count naive Bayes | grows like the nonzero counts | $KV$ |

Doubling $d$ makes the factor about eight times slower. Naive Bayes never builds that matrix.

## Where this breaks

**Features that are not bell-shaped.** Skew or categories break the Gaussian. Take a log, or switch to a matching naive Bayes model.

**A covariance you cannot factor.** When $d$ nears $n$, Cholesky fails. Add a small multiple of the identity to `cov`.

**Copied features.** Naive Bayes counts a duplicated column twice, and the probabilities get sharper. Drop near-duplicates first.

**Outliers.** One extreme row pulls the averages. The last check shows GDA losing to logistic regression. Drop the row, or use a discriminative model.

## The exercise

You implement `fit_gda`, `gda_log_posterior`, `gda_to_logistic`, `fit_naive_bayes_gaussian`, and `fit_naive_bayes_multinomial`.

The checks confirm shapes, priors that sum to 1, and estimates that match a direct maximum likelihood computation. They check that each log-posterior row exponentiates to a distribution and matches an independent Gaussian density. They check that `w` and `b` reproduce GDA's decisions, and that Laplace smoothing normalises every row. The last checks show GDA beating logistic regression on Gaussian data, and losing once outliers break that assumption.
