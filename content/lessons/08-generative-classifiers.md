---
title: Generative classifiers
slug: 08-generative-classifiers
part: "Part 2 — Classical machine learning"
summary: Learn what each class looks like, then classify with Bayes' rule. The Gaussian version turns out to be logistic regression.
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

Logistic regression learns the boundary between classes directly. It never asks what a typical cat photo or a typical spam email looks like. It only asks which side of a line a point falls on.

A generative classifier works the other way round. It first learns a description of each class on its own: where the class sits, how spread out it is, and how common it is. To classify a new point, it asks each class "how likely were you to produce this point?" and picks the class with the best answer.

This lesson builds two generative classifiers. The first, Gaussian discriminant analysis, models each class as a bell-shaped cloud. You will see that it produces exactly the same kind of boundary as logistic regression. The second, naive Bayes, makes a strong simplifying assumption that lets it scale to thousands of features, such as the words in an email.

## What you already know

- Logistic regression predicts $p(y = 1 \mid x) = \sigma(w^{\top}x + b)$, where $\sigma(z) = 1 / (1 + e^{-z})$ is the sigmoid.
- A probability distribution over classes has non-negative entries that add up to 1.
- Bayes' rule turns $p(x \mid y)$ into $p(y \mid x)$: multiply by the prior $p(y)$ and divide by $p(x)$.
- Working with log probabilities avoids tiny numbers underflowing to zero.
- The mean of a set of vectors is their sum divided by how many there are.

## The idea: describe each class, then compare

Think of a sorting office that gets parcels from two warehouses. Each warehouse ships parcels of a typical weight and size. When an unlabelled parcel arrives, you ask which warehouse would most plausibly have shipped it. You also account for the fact that one warehouse ships far more parcels than the other.

That is Bayes' rule. Write $K$ for the number of classes, labelled $0, 1, \ldots, K-1$. For each class $k$ you need two things:

- The **prior** $\pi_k = p(y = k)$: how common class $k$ is before you look at $x$.
- The **class-conditional density** $p(x \mid y = k)$: how likely class $k$ is to produce a point that looks like $x$.

Then the **posterior**, the probability of class $k$ after seeing $x$, is

$$
p(y = k \mid x) = \frac{\pi_k \, p(x \mid y = k)}{\sum_{j} \pi_j \, p(x \mid y = j)} .
$$

Read it aloud: take class $k$'s prior times its density at $x$. Do the same for every class. Divide class $k$'s number by the total, so all the posteriors add up to 1.

In code you work with logs. The numerator becomes a sum, and the division becomes a subtraction of a **log-sum-exp**, the log of the total:

```python
joint = np.log(priors)[None, :] + log_density        # shape (n, K)
log_post = joint - logsumexp(joint, axis=1)[:, None]  # each row sums to 1 after exp
```

A model like this is called **generative** because it describes how to generate $x$ for each class. Logistic regression is **discriminative**: it models $p(y \mid x)$ directly and ignores how $x$ is spread out.

## Gaussian discriminant analysis

Gaussian discriminant analysis (GDA) says each class is a cloud of points shaped like a bell curve in $d$ dimensions, a **multivariate Gaussian**. Each class gets its own centre, the **mean** $\mu_k$, a vector of length $d$. All classes share one **covariance matrix** $\Sigma$, a $d \times d$ matrix that says how spread out the cloud is along each direction and how features move together.

The log density of a Gaussian at a point $x$ is

$$
\log p(x \mid y = k) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log\det\Sigma - \frac{1}{2}(x - \mu_k)^{\top}\Sigma^{-1}(x - \mu_k) .
$$

Read it piece by piece:

- $-\frac{d}{2}\log(2\pi)$ is a constant that only depends on the number of features.
- $\log\det\Sigma$ is the log of the determinant of $\Sigma$. A wider cloud has a larger determinant, so each point in it is a little less likely.
- $(x - \mu_k)^{\top}\Sigma^{-1}(x - \mu_k)$ is the **Mahalanobis distance** squared. It is the squared distance from $x$ to the centre, measured in units of the cloud's spread. Far points get a large number, so a very negative log density.

### Compute it without inverting the matrix

Inverting $\Sigma$ is slow and loses accuracy. Instead, use the **Cholesky factor**: a lower-triangular matrix $L$ with $L L^{\top} = \Sigma$. Two facts make it useful:

- If you solve $L z = x - \mu_k$ for $z$, then the Mahalanobis term equals $z^{\top} z$, the plain sum of squares of $z$.
- $\log\det\Sigma = 2\sum_i \log L_{ii}$, twice the sum of the logs of the diagonal of $L$.

In numpy and scipy, for all $n$ points at once:

```python
L = cholesky(cov, lower=True)                         # (d, d)
log_det = 2.0 * np.sum(np.log(np.diag(L)))
z = solve_triangular(L, (X - means[k]).T, lower=True) # (d, n)
maha = np.sum(z**2, axis=0)                           # (n,)
```

## Fit the model: counting and averaging

Fitting GDA needs no gradient descent. The **maximum likelihood estimates**, the parameters that make the training data most probable, are plain averages. Write $n_k$ for the number of training points in class $k$.

$$
\pi_k = \frac{n_k}{n}, \qquad
\mu_k = \frac{1}{n_k}\sum_{i:\, y_i = k} x_i, \qquad
\Sigma = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu_{y_i})(x_i - \mu_{y_i})^{\top} .
$$

Read them aloud:

- The prior is the fraction of training points in class $k$.
- The mean is the average of the points in class $k$.
- The **pooled covariance** subtracts from every point the mean of its own class, then averages the outer products over all $n$ points. It divides by $n$, not by $n - 1$ and not by $n_k$.

In numpy, for each class:

```python
mask = y == k
priors[k] = mask.sum() / n
means[k] = X[mask].mean(axis=0)
diff = X[mask] - means[k]
cov += diff.T @ diff      # add this class's sum; divide by n once at the end
```

### A worked example

Take one feature ($d = 1$) and four points. Class 0 has $x = 1$ and $x = 3$. Class 1 has $x = 5$ and $x = 7$.

- Priors: $\pi_0 = 2/4 = 0.5$ and $\pi_1 = 0.5$.
- Means: $\mu_0 = (1 + 3)/2 = 2$ and $\mu_1 = (5 + 7)/2 = 6$.
- Covariance: the differences from each point's own class mean are $-1, 1, -1, 1$. Their squares add up to $4$. Divide by $n = 4$ to get $\Sigma = 1$.

Now classify $x = 3$. The constant is $-\frac{1}{2}\log(2\pi) = -0.9189$, and $\log\det\Sigma = 0$.

| Class | Mahalanobis term | Log density | Plus $\log\pi_k = -0.6931$ |
|---|---|---|---|
| 0 | $(3-2)^2 = 1$ | $-0.9189 - 0.5 = -1.4189$ | $-2.1121$ |
| 1 | $(3-6)^2 = 9$ | $-0.9189 - 4.5 = -5.4189$ | $-6.1121$ |

The log-sum-exp of the last column is $-2.1121 + \log(1 + e^{-4}) = -2.0939$. Subtract it to get log posteriors of $-0.0181$ for class 0 and $-4.0181$ for class 1. As probabilities, that is $0.982$ and $0.018$. The point at $x = 3$ is almost certainly class 0.

## GDA is logistic regression in disguise

With two classes and a shared covariance, something neat happens. When you take the log of the posterior ratio, the quadratic parts $x^{\top}\Sigma^{-1}x$ from the two classes are identical and cancel. What is left is linear in $x$. So the posterior is a sigmoid of a linear score:

$$
p(y = 1 \mid x) = \sigma(w^{\top}x + b), \qquad
w = \Sigma^{-1}(\mu_1 - \mu_0), \qquad
b = \log\frac{\pi_1}{\pi_0} - \frac{1}{2}\mu_1^{\top}\Sigma^{-1}\mu_1 + \frac{1}{2}\mu_0^{\top}\Sigma^{-1}\mu_0 .
$$

Read it aloud:

- The weight vector points from the class 0 centre to the class 1 centre, adjusted by the inverse covariance so that directions with a lot of spread count for less.
- The bias has three parts: the log of how much more common class 1 is, minus half of class 1's centre measured through $\Sigma^{-1}$, plus half of class 0's.

In numpy:

```python
P = np.linalg.inv(cov)                     # fine here: d is small, done once
w = P @ (mu1 - mu0)
b = np.log(pi1 / pi0) - 0.5 * mu1 @ P @ mu1 + 0.5 * mu0 @ P @ mu0
```

Check it on the worked example. $w = (6 - 2)/1 = 4$ and $b = 0 - 0.5 \cdot 36 + 0.5 \cdot 4 = -16$. At $x = 3$ the score is $4 \cdot 3 - 16 = -4$, and $\sigma(-4) = 0.018$. That matches the class 1 posterior from the table. At $x = 4$, halfway between the centres, the score is 0 and the probability is exactly 0.5.

If each class had its own covariance, the quadratic parts would not cancel. The boundary would then curve, and you would get quadratic discriminant analysis. This lesson only uses the shared version.

## Naive Bayes

A $d \times d$ covariance gets huge when $d$ is large. With 10,000 words as features, it has 100 million entries. **Naive Bayes** avoids it by assuming that, once you know the class, every feature is independent of every other. This is called **conditional independence**. The density of $x$ then becomes a product of one small density per feature, and its log is a sum.

The assumption is usually false. The words "free" and "prize" appear together in spam far more than chance would predict. Naive Bayes still ranks classes well in practice, even though its probabilities come out overconfident.

### Gaussian naive Bayes

For continuous features, give each class and each feature its own one-dimensional Gaussian. You need a mean and a variance for every (class, feature) pair:

$$
\mu_{kj} = \text{mean of feature } j \text{ over class } k, \qquad
\sigma^2_{kj} = \frac{1}{n_k}\sum_{i:\, y_i = k}(x_{ij} - \mu_{kj})^2 .
$$

The variance divides by $n_k$, the class size. That is what `np.var` does by default. Unlike GDA, nothing is shared: each class gets its own spread. If a feature is constant inside a class, its variance is zero, and dividing by it later breaks. So you raise every variance to at least a small floor:

```python
means[k] = X[mask].mean(axis=0)
variances[k] = np.maximum(X[mask].var(axis=0), var_floor)
```

### Word-count naive Bayes and Laplace smoothing

For text, each example is a row of word counts over a vocabulary of $V$ words. The multinomial model gives each class $k$ a probability $\theta_{kj}$ for each word $j$. Write $c_{kj}$ for the total count of word $j$ across all class $k$ documents. The raw estimate is $c_{kj}$ divided by the total word count of the class.

That raw estimate has a trap. A word that never appeared in class $k$ gets probability zero. One such word in a test email sends the log score to minus infinity, no matter how strong the other evidence is. **Laplace smoothing** fixes this by pretending you saw every word $\alpha$ extra times:

$$
\theta_{kj} = \frac{c_{kj} + \alpha}{\sum_{j'} c_{kj'} + \alpha V} .
$$

Read it aloud: add $\alpha$ to every word's count in the class. The denominator is the new total, the old total plus $\alpha$ for each of the $V$ words. Each class's probabilities then add up to 1. The prior is the class fraction, as before.

```python
counts = X[mask].sum(axis=0)           # (V,) total counts for class k
smoothed = counts + alpha
log_theta[k] = np.log(smoothed / smoothed.sum())
log_priors[k] = np.log(mask.sum() / n)
```

Work a small example with $V = 3$ and $\alpha = 1$. Class 0 has two documents with counts $[2, 0, 1]$ and $[0, 0, 0]$. Summed, that is $[2, 0, 1]$. Add 1 to get $[3, 1, 2]$, which totals 6. So $\theta_0 = [0.5, 0.1667, 0.3333]$. Word 2 never appeared in class 0, yet it still gets probability $1/6$ instead of zero.

## When the generative model wins

GDA uses more information than logistic regression. It also models how $x$ is spread out, not just the boundary. When the data really is Gaussian with a shared covariance, that extra information pays off. GDA reaches a good boundary with fewer training examples.

When the assumption is wrong, the extra information misleads. A few far-away outliers drag the class means and blow up the covariance, because both are plain averages. GDA's boundary then settles in the wrong place, and more data does not fix it. Logistic regression only assumes the log-odds are linear, so it barely moves for a point already far on the correct side. The rule of thumb: with little data, trust the model whose assumptions you believe. With a lot of data, the less assuming model usually wins.

## Build it step by step

Work through the functions in the order the checks test them.

1. **`fit_gda(X, y)`.** Takes `X` of shape `(n, d)` and integer labels `y` of shape `(n,)`. Returns `means` `(K, d)`, `cov` `(d, d)`, and `priors` `(K,)`. Get `K` as `int(y.max()) + 1`. Loop over classes with a boolean mask, use `.mean(axis=0)`, and add `diff.T @ diff` into `cov`. The most common mistake is dividing the covariance by `n - 1`, by `n - K`, or by each `n_k`. Divide the total once by `n`.
2. **`gda_log_posterior(X, means, cov, priors)`.** Returns an `(n, K)` array of log posteriors. Factor once with `cholesky(cov, lower=True)`. For each class, call `solve_triangular` on `(X - means[k]).T`, square and sum over axis 0, and build the log density from the formula. Add `np.log(priors[k])`. Then subtract a stable log-sum-exp across axis 1: take the row maximum out before calling `np.exp`. The common mistake is forgetting the `-0.5 * log_det` or the $d/2 \log(2\pi)$ term. They cancel in the posterior, but a missing factor of 0.5 on the Mahalanobis term does not.
3. **`gda_to_logistic(means, cov, priors)`.** Two classes only. Returns `w` of shape `(d,)` and `b` as a Python `float`. Use the formulas above directly. The common mistake is flipping the sign: `w` uses `means[1] - means[0]`, because the sigmoid gives the probability of class 1.
4. **`fit_naive_bayes_gaussian(X, y, var_floor)`.** Returns `means` `(K, d)`, `variances` `(K, d)`, and `priors` `(K,)`. Use `.mean(axis=0)`, `.var(axis=0)`, and `np.maximum` with the floor. The common mistake is using `ddof=1`. Keep the default, which divides by $n_k$.
5. **`fit_naive_bayes_multinomial(X, y, alpha)`.** Returns `log_theta` `(K, V)` and `log_priors` `(K,)`. Sum counts with `.sum(axis=0)` over the class's rows, add `alpha`, normalise, then take `np.log`. The common mistake is smoothing each document before summing, or adding `alpha` only to the denominator.

## Cost and scaling

Every estimate is a single pass of averaging, so fitting is fast. The table uses $n$ training points, $d$ features, $K$ classes, and $m$ query points.

| Step | Time | Memory |
|---|---|---|
| Fit GDA | grows with $n d^2$ (one outer product per point) | $K d + d^2$ |
| Predict with GDA | $d^3$ once to factor, then $m K d^2$ | $d^2$ |
| Fit Gaussian naive Bayes | grows with $n d$ | $K d$ |
| Fit word-count naive Bayes | grows with the number of nonzero counts | $K V$ |

GDA's cost is dominated by the $d \times d$ covariance. Doubling the features makes it four times as big and eight times as slow to factor. Naive Bayes never builds that matrix, which is why it handles vocabularies of tens of thousands of words.

## Where this breaks

**Features that are not bell-shaped.** Skewed or categorical features break the Gaussian assumption. Transform the features, such as taking a log of a skewed quantity, or use a naive Bayes model that suits the feature type.

**A covariance that cannot be factored.** When $d$ gets close to the number of training points, the covariance becomes singular and the Cholesky factor fails. Add a small multiple of the identity matrix to `cov`. This is the same trade that ridge regression makes.

**Correlated features in naive Bayes.** Copy a feature twice, and naive Bayes counts its evidence twice. The probabilities get more extreme and the boundary shifts. Remove near-duplicate features before fitting.

**Outliers.** Means and covariances are averages, so one extreme point pulls them hard. The last check in the exercise shows GDA losing to logistic regression for exactly this reason. Remove outliers or use a discriminative model.

## The exercise

You build shared-covariance GDA from scratch, then both kinds of naive Bayes. Implement `fit_gda` to return the class means, pooled covariance, and priors. Implement `gda_log_posterior` to return log posteriors computed through a Cholesky factor. Implement `gda_to_logistic` to return the logistic `w` and `b` for a two-class model. Then implement `fit_naive_bayes_gaussian` and `fit_naive_bayes_multinomial`.

The checks confirm output shapes and that the priors sum to one. They compare your estimates against a direct computation, and check that each row of your log posterior exponentiates to a valid distribution matching an independent Gaussian density. They confirm that your logistic `w` and `b` reproduce GDA's decisions to machine precision, and that Laplace smoothing normalises every row. The last two checks show GDA beating logistic regression on Gaussian data, and losing once outliers break the assumption.
