---
title: Generalisation and model selection
slug: 07-model-selection
part: "Part 2 — Classical machine learning"
summary: Estimate error on new data with cross-validation, leave-one-out, and learning curves, then pick a setting.
minutes: 60
gpu: false
objectives:
  - Build k-fold cross-validation and use it to pick a ridge penalty.
  - Compute leave-one-out error for ridge regression from one fit, using the hat matrix.
  - Build a learning curve and read whether you need more data or a richer model.
  - Estimate Rademacher complexity for a small set of models by sampling random signs.
  - Explain why the winning score among many settings is too good to report.
prereqs:
  - 01-the-learning-problem
exercise: 07-model-selection
---

# Generalisation and model selection

You fit a model on last year's exam and it scores perfectly. This year's paper is the one that matters. The gap between those two scores is the *generalisation gap*.

You also have several ridge penalties, and you must pick one before new marks arrive. A penalty that wins on papers you already used can be lucky.

You will bound that gap, score flexibility with random signs, and estimate error with cross-validation, leave-one-out, and a learning curve.

## What you already know

These five ideas are the ones this lesson uses:

- Training error is the average loss on the data you fit, and on 100 points it often beats the error on new points.
- *True risk* is the average loss on new data from the same source, which the training set cannot give you.
- Ridge regression with penalty $\lambda = 1$ fits $w = (X^{\top}X + I)^{-1}X^{\top}y$.
- Mean squared error, or MSE, averages squared gaps, so gaps of 1 and 3 give $(1 + 9) / 2 = 5$.
- Examples are *i.i.d.* when each is drawn independently from the same distribution, as with two flips of one coin.

## Bound the gap for a short list of models

You try two scoring rules on four answers. Each loss sits between 0 and 1. You allow the claim to fail with probability 0.5.

Twice the number of rules, divided by 0.5, is 8. The natural log of 8 is about 2.079. Divide by $2 \times 4 = 8$ to get 0.260. The square root is about 0.510, so the gap is at most about 0.51.

That guarantee is a *finite-class generalisation bound*:

$$
\text{gap} \le \sqrt{\frac{\log(2M / \delta)}{2n}}
$$

$M$ is the number of models, $\delta$ is the failure chance, and $n$ is the number of i.i.d. examples. The log is natural, and the loss must sit between 0 and 1.

The same inputs return 0.510. At $n = 1000$, $M = 100$, and $\delta = 0.05$, the bound is about 0.064. Four times the data halves it.

Numpy is `np.sqrt(np.log(2 * M / delta) / (2 * n))`.

The usual mistake is `np.log10`, which gives about 0.336 on these inputs.

> [!KEY]
> The gap shrinks like $1/\sqrt{n}$. Extra models widen it only through a logarithm.

Infinite weights need a different score.

## Score flexibility with random signs

Model 1 outputs $[1, 1]$ on two examples. Model 2 outputs $[1, -1]$. Give each example a random sign, $+1$ or $-1$.

For signs $[1, -1]$, model 1 scores $(1 - 1) / 2 = 0$ and model 2 scores $(1 + 1) / 2 = 1$.

All four sign patterns give these scores:

| Signs | Model 1 | Model 2 | Best |
|---|---|---|---|
| $[-1, -1]$ | $-1$ | $0$ | $0$ |
| $[-1, 1]$ | $0$ | $-1$ | $0$ |
| $[1, -1]$ | $0$ | $1$ | $1$ |
| $[1, 1]$ | $1$ | $0$ | $1$ |

The last column averages to $(0 + 0 + 1 + 1) / 4 = 0.5$.

That average is the *empirical Rademacher complexity*. Row $j$ of $F$ holds model $j$ on $n$ examples:

$$
\hat{R} = \text{average over many random } \sigma \text{ of }
\max_{j} \frac{1}{n} \sum_{i=1}^{n} \sigma_i F_{ji}
$$

You dot each row with a sign vector $\sigma$, divide by $n$, keep the max, and average over draws of $\sigma$.

The formula returns that same 0.5. One model averages near 0. A set that matches every sign pattern scores 1.

Numpy is `(F @ signs.T).max(axis=0).mean() / n`, with `signs` of shape `(num_draws, n)`.

The usual mistake is `max(axis=1)`, the best draw per model rather than the best model per draw.

> [!WARNING]
> `F @ signs.T` has shape `(number of models, num_draws)`. Max over `axis=0`.

A bound above 1 does not constrain a loss between 0 and 1. Estimate the risk when you pick a penalty.

## Estimate the risk with k-fold cross-validation

Shuffle seven indices to $[3, 1, 4, 0, 6, 2, 5]$. Cut them into groups of 3, 2, and 2: $[3, 1, 4]$, $[0, 6]$, $[2, 5]$. Sort each group. Every index from 0 to 6 appears once. Each group sits out once. Held-out MSEs of 4, 2, and 2 average to $(4 + 2 + 2) / 3 = 8/3$.

This is *k-fold cross-validation*. One shuffle, $k$ folds, and the average held-out MSE:

$$
\text{CV} = \frac{1}{k}\sum_{f=1}^{k} \text{MSE}_f
$$

$\text{MSE}_f$ is the mean squared error of a model that never saw fold $f$. Scores 4, 2, and 2 give $8/3$.

Numpy shuffles with `rng.permutation(n)`, then `np.array_split`, then `np.sort` on each piece.

The usual mistake is `np.split`. For $n = 7$ and $k = 3$ it errors. `array_split` returns sizes 3, 2, and 2.

> [!TIP]
> `np.array_equal(np.sort(np.concatenate(folds)), np.arange(n))` is true only when every index appears once.

Each fold trains on $(k - 1) / k$ of the rows. For $k = 5$ that is 0.8, a little pessimistic. At $k = n$ you get *leave-one-out*, less bias and a noisier average. Folds from 5 to 10 are the usual compromise.

> [!KEY]
> Every example is scored once, by a model that never saw it.

## Read leave-one-out error from one ridge fit

Leave-one-out refits once per point. Ridge can do it in one fit.

Take $\lambda = 1$, inputs $1, 2, 3$, and targets $1, 2, 2$. Then $X^{\top}X = 1 + 4 + 9 = 14$, so $14 + \lambda = 15$. And $X^{\top}y = 1 + 4 + 6 = 11$, so $w = 11/15$.

Fitted values are $11/15$, $22/15$, and $33/15$. Residuals are $4/15$, $8/15$, and $-1/5$. Diagonal weights are $1/15$, $4/15$, and $9/15$. Divide each residual by one minus its weight: $2/7$, $8/11$, and $-1/2$. The mean of the squares is about 0.2869.

Call this matrix the *hat matrix*, and call $h_{ii}$ the *leverage*. The scaled residuals are the leave-one-out residuals:

$$
H = X (X^{\top}X + \lambda I)^{-1} X^{\top}
$$

$$
y_i - \hat{y}_i^{(-i)} = \frac{r_i}{1 - h_{ii}}, \qquad r = y - Hy
$$

$H$ is $n$ by $n$. Dividing by $1 - h_{ii}$ undoes the pull of a high leverage. The formula returns $2/7$, $8/11$, and $-1/2$.

Numpy builds $H$ as `X @ np.linalg.inv(X.T @ X + lam * np.eye(d)) @ X.T`. Divide `y - H @ y` by `1 - np.diag(H)`.

The usual mistake is dropping `lam * np.eye(d)`. At $\lambda = 1$ the divisor is then 14, not 15.

> [!KEY]
> For ridge, divide each ordinary residual by one minus its leverage. One fit is enough.

## Keep the choice off the score you report

Twenty penalties share five folds, and the winner averages 0.20. One of twenty tends to be lucky, so 0.20 is too good.

Use five outer folds to score. Inside each outer training set, inner cross-validation picks the penalty. The outer fold scores that choice. An outer average of 0.35 is the number you report.

This is *nested cross-validation*. The inner loop chooses and the outer loop scores the recipe, both by averaging MSE. The outer plugs in as 0.35, not the lucky 0.20.

The usual mistake is publishing 0.20 after it has already chosen the penalty.

> [!KEY]
> The best score among many settings is too optimistic. Score the choice on data it never saw.

## Read a learning curve

Hold out 2 of 10 points. Shuffle the other 8 once. Train on the first 2, then the first 4, then all 8.

At size 2 you might see training MSE 0.1 and validation MSE 0.9. At size 8, 0.4 and 0.5. Training error rose. Validation error fell.

That plot is a *learning curve*. One validation set and one shuffle stop it jumping.

If both curves meet near 0.7, you need a richer model or better features. A gap near 0.1 against 0.9 is too flexible a fit. More data, a stronger penalty, or a simpler model can help.

Numpy holds the pool in `order[n_val:]` and scores with `np.mean((y_hat - y_tr) ** 2)`.

The usual mistake is a new shuffle at every size, so larger sets are not supersets.

Keep these checks in mind:

> [!RECAP]
> - The finite-class gap is $\sqrt{\log(2M/\delta) / (2n)}$, too loose to pick a penalty.
> - k-fold cross-validation shuffles once, splits with `np.array_split`, and averages held-out MSE.
> - For ridge, the leave-one-out residual is $r_i / (1 - h_{ii})$.
> - A learning curve reuses one validation set and one shuffled pool.
> - The random-sign score keeps the best model per draw, then divides by $n$.

## Build it step by step

Build the five functions in the order the checks test them:

1. **`k_fold_indices(n, k, rng)`**. Returns `k` sorted integer arrays that partition `0` to `n - 1`. Sizes differ by at most one. Use `rng.permutation`, `np.array_split`, and `np.sort`. The common mistake is `np.split` when `k` does not divide `n`. The pieces have these shapes:

```python
order = rng.permutation(n)         # shape (n,)
chunks = np.array_split(order, k)  # k arrays
```

2. **`cross_val_score(fit, predict, X, y, k)`**. Takes `X` of shape `(n, d)` and `y` of shape `(n,)`. Returns fold MSEs of shape `(k,)`. Use `np.random.default_rng(0)` inside, and `np.setdiff1d` for the training rows. The common mistake is an outside generator. Each fold does this:

```python
pred = predict(model, X[test])  # shape (n_test,)
mse = np.mean((pred - y[test]) ** 2)
```

3. **`loo_linear_smoother(X, y, lam)`**. Returns one float. Divide residuals by `1 - np.diag(H)`, square, and average. The common mistake is forgetting `lam * np.eye(d)`. The matrices look like this:

```python
gram = X.T @ X + lam * np.eye(d)  # shape (d, d)
H = X @ np.linalg.inv(gram) @ X.T  # shape (n, n)
```

4. **`learning_curve(fit, predict, X, y, train_sizes, val_fraction, seed)`**. Returns two arrays, one entry per size. Shuffle with `np.random.default_rng(seed)`, hold out `int(round(val_fraction * n))` rows, and train on `pool[:size]`. The common mistake is a new shuffle per size. One size uses these indices:

```python
val_idx = order[:n_val]  # shape (n_val,)
train_idx = pool[:size]  # shape (size,)
```

5. **`empirical_rademacher(losses_matrix, rng, num_draws)`**. Takes `losses_matrix` of shape `(num_models, n)` and returns one float. Draw signs of shape `(num_draws, n)` with `rng.choice`, take `max` on `axis=0`, average, and divide by `n`. The common mistake is the other axis. The product has this shape:

```python
signs = rng.choice(np.array([-1.0, 1.0]), size=(num_draws, n))
scores = losses_matrix @ signs.T  # shape (num_models, num_draws)
```

## Cost and scaling

One ridge fit costs about $n \times d^2$ work, for $n$ rows and $d$ features. The methods compare as follows:

| Method | Work | Memory |
|---|---|---|
| k-fold cross-validation | about $k$ ridge fits | the training rows |
| Leave-one-out by refitting | $n$ fits | the training rows |
| Leave-one-out shortcut | about one fit, plus $H$ | $n^2$ entries in $H$ |
| Nested cross-validation | outer times inner times settings | one fold at a time |
| Random-sign estimate | models times $n$ times draws | the sign matrix |

Those $n$ refits are why 5-fold or 10-fold is the default.

## Where this breaks

**Rows that are not i.i.d.** A random time-series split trains on the future and tests on the past. Scans from one patient leak the same way. Split a series forward in time, and keep each group in one fold.

**Too many settings.** With a few dozen rows, nested loops are too small to trust. Compare fewer settings.

**Reused folds.** Choosing features from one cross-validation score makes that score part of the fit. Keep a final test set and use it once.

**Leverage near 1.** The shortcut divides by nearly 0. This is common at $\lambda = 0$ when $d$ is close to $n$. A positive $\lambda$ keeps every leverage below 1.

## The exercise

You implement `k_fold_indices`, `cross_val_score`, `loo_linear_smoother`, `learning_curve`, and `empirical_rademacher`. Callers pass `fit(X_train, y_train)` and `predict(model, X_test)`.

The checks confirm a full partition into folds whose sizes differ by at most one, and the same `k` finite scores from every `cross_val_score` call. They match `loo_linear_smoother` to three brute-force refits, and they check that validation error falls as the training set grows. They check that `empirical_rademacher` is near 0 for one model, larger for 300 models, and 1 for every sign pattern. The last check picks a ridge penalty that beats both no penalty and a very large penalty on fresh data.
