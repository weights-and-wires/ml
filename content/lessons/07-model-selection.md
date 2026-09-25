---
title: Generalisation and model selection
slug: 07-model-selection
part: "Part 2 — Classical machine learning"
summary: Estimate how well a model does on unseen data with cross-validation and leave-one-out, and use that to pick settings.
minutes: 60
gpu: false
objectives:
  - Build k-fold cross-validation and use it to pick a ridge penalty.
  - Compute leave-one-out error for ridge regression from a single fit, using the hat matrix.
  - Build a learning curve and read what it says about more data versus a better model.
  - Estimate the Rademacher complexity of a small set of models by sampling random signs.
  - Explain why choosing a setting and scoring it on the same folds gives a score that is too good.
prereqs:
  - 01-the-learning-problem
exercise: 07-model-selection
---

# Generalisation and model selection

Every model so far picks its weights by doing well on the training data.
What you care about is how it does on data it has never seen. A student who
memorises last year's exam can score perfectly on it and still fail this
year's. The gap between those two scores is the *generalisation gap*, and
lesson 1 measured it once with a held-out set.

This lesson asks two practical questions. How big can that gap be? And how
do you pick between models, such as ten different ridge penalties, when you
can't see the future data?

Theory gives the first answer, but its numbers are far too loose to act on.
So you build the tools people use instead: k-fold cross-validation, a fast
shortcut for leave-one-out, and learning curves. You also build a small
estimator for how flexible a set of models is.

## What you already know

- The training error is the average loss on the data you fit, and it tends to look better than the truth.
- The *true risk* is the average loss on new data from the same source, which you can't compute directly.
- Ridge regression fits $w = (X^{\top}X + \lambda I)^{-1}X^{\top}y$, where $\lambda$ is the penalty strength.
- The mean squared error, or MSE, is the average of the squared differences between predictions and targets.
- Data is *i.i.d.* when each example is drawn independently from the same distribution.

## How big can the gap be

Picture flipping a coin 100 times to test whether it is fair. With a single
coin, getting 70 heads is rare. With a thousand coins, some coin will
probably land 70 heads by luck. Model choice works the same way. The more
models you try, the more likely one of them looks good on your data by
accident.

Theory turns this into a formula. Suppose you pick from $M$ candidate
models, your loss is always between 0 and 1, and your $n$ training examples
are i.i.d. Then, with probability at least $1 - \delta$, every one of the $M$
models has a gap of at most:

$$
\text{gap} \le \sqrt{\frac{\log(2M / \delta)}{2n}}
$$

Read it aloud: take the log of twice the number of models divided by the
allowed failure chance $\delta$, divide by twice the sample size, and take the
square root. Two lessons follow. The gap shrinks like one over $\sqrt{n}$, so
four times the data halves it. And trying more models widens it, but only
slowly, through a log.

```python
gap = np.sqrt(np.log(2 * M / delta) / (2 * n))
```

With $n = 1000$, $M = 100$, and $\delta = 0.05$, the bound is about $0.064$.

This formula needs a finite number of models. A linear model has infinitely
many possible weight vectors. Theory handles that with a measure of how
flexible a set of models is, such as the VC dimension or the Rademacher
complexity. The next section builds the second one.

## Measure flexibility with random signs

A flexible set of models can fit anything, even pure noise. The *Rademacher
complexity* measures exactly that. Flip a fair coin for each of your $n$
examples to get a random sign $\sigma_i$ of $+1$ or $-1$. Then ask how well
the best model in your set lines up with those random signs.

Each row of a matrix $F$ holds one model's outputs on the $n$ examples, so
$F$ has shape (number of models, $n$). The estimate is:

$$
\hat{R} = \text{average over many random } \sigma \text{ of }
\max_{j} \frac{1}{n} \sum_{i=1}^{n} \sigma_i F_{ji}
$$

Read it aloud: draw a random sign vector $\sigma$. For each model $j$, take
the dot product of its row with $\sigma$ and divide by $n$. Keep the largest
of those values. Repeat for many sign vectors and average the results.

```python
signs = rng.choice(np.array([-1.0, 1.0]), size=(num_draws, n))
estimate = (F @ signs.T).max(axis=0).mean() / n
```

> [!WARNING]
> `F @ signs.T` has shape (number of models, `num_draws`). Take the max over `axis=0`, across models, to keep the best model for each draw. Then average over the draws.

A set with one model scores close to 0, because on average its dot product
with random signs is 0. A set that can match every possible sign pattern
scores exactly 1, the maximum.

### Work the random-sign estimate by hand

Take two models on $n = 2$ examples, with outputs $f_1 = [1, 1]$ and
$f_2 = [1, -1]$. There are only four sign vectors, so you can list them all
instead of sampling.

| $\sigma$ | $\frac{1}{2}\,\sigma \cdot f_1$ | $\frac{1}{2}\,\sigma \cdot f_2$ | Largest |
|---|---|---|---|
| $[-1, -1]$ | $-1$ | $0$ | $0$ |
| $[-1, 1]$ | $0$ | $-1$ | $0$ |
| $[1, -1]$ | $0$ | $1$ | $1$ |
| $[1, 1]$ | $1$ | $0$ | $1$ |

The average of the last column is $0.5$. With many sampled draws, your
function returns a number close to $0.5$ on this input.

## Why theory can't pick your model

These bounds hold for every model in the set at once. They have to cover the
worst possible case, and the model you trained is almost never that case. So
in practice the bounds are loose by a huge margin. For a modest neural
network, a bound on the error can come out larger than 1, which says nothing
at all.

The bounds are still useful as a guide. More data helps, at a rate of one
over $\sqrt{n}$. Flexibility hurts. Regularisation limits flexibility. None
of that tells you which of ten ridge penalties to use. For that, you stop
bounding the true risk and start estimating it.

## k-fold cross-validation

Think of a class that splits into five study groups. Each group takes turns
sitting out, and the other four teach a model. The group that sat out then
tests it. Every student is tested once, always by a model that never saw
them.

That is *k-fold cross-validation*. Shuffle the $n$ indices, cut them into $k$
*folds* of nearly equal size, and for each fold:

1. Fit on the other $k - 1$ folds.
2. Predict on the held-out fold.
3. Record the MSE on that fold.

The result is $k$ numbers, and their average estimates the true risk.

```python
order = rng.permutation(n)
folds = [np.sort(chunk) for chunk in np.array_split(order, k)]
```

`np.array_split` handles sizes that don't divide evenly. With $n = 7$ and
$k = 3$, it gives folds of size 3, 2, and 2.

> [!TIP]
> Check your folds with `np.array_equal(np.sort(np.concatenate(folds)), np.arange(n))`. It is `True` only when every index appears exactly once.

The choice of $k$ is a trade-off:

- Each fold trains on only $\frac{k-1}{k}$ of the data. With less data a
  model is usually a bit worse, so the estimate comes out slightly too
  pessimistic. A larger $k$ shrinks that bias.
- The training sets overlap heavily, so their scores move together. A larger
  $k$ means more overlap and a noisier average. It also means more fits.

At $k = n$ you get *leave-one-out*: the least bias and among the most noise.
Values from 5 to 10 are the usual compromise.

> [!KEY]
> In cross-validation, every example is scored once, by a model that never saw it. The average of the $k$ fold scores is your estimate of the true risk.

## The leave-one-out shortcut

Leave-one-out needs $n$ fits, one per held-out example. For ridge regression
you can get all $n$ results from a single fit.

Ridge predictions on the training data are a matrix times the targets,
$\hat{y} = Hy$. The matrix $H$ is the *hat matrix*, because it puts the hat
on $y$:

$$
H = X (X^{\top}X + \lambda I)^{-1} X^{\top}
$$

Read it aloud: form $X^{\top}X$, add $\lambda$ to its diagonal, invert it,
and sandwich it between $X$ and $X^{\top}$. $H$ has shape $n$ by $n$. Its
diagonal entry $h_{ii}$ is the *leverage* of example $i$: how much that
example's own target pulls on its own prediction.

The shortcut says the leave-one-out residual for example $i$ is:

$$
y_i - \hat{y}_i^{(-i)} = \frac{r_i}{1 - h_{ii}}, \qquad r = y - Hy
$$

Read it aloud: $\hat{y}_i^{(-i)}$ is the prediction for example $i$ from a
model fitted without it. Take the ordinary residual $r_i$ from the full fit,
and divide by one minus that example's leverage. A high-leverage example
pulled the fit toward itself, so its ordinary residual looks too small. The
division scales it back up. The leave-one-out MSE is the mean of the squares
of these residuals.

```python
H = X @ np.linalg.inv(X.T @ X + lam * np.eye(d)) @ X.T
loo = (y - H @ y) / (1 - np.diag(H))
```

> [!KEY]
> For ridge, one fit gives every leave-one-out residual. Divide each ordinary residual by one minus its leverage.

### Work the shortcut by hand

Take one feature, three examples, and $\lambda = 1$:

$$
X = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \quad
y = \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}
$$

Then $X^{\top}X + \lambda = 1 + 4 + 9 + 1 = 15$, so the inverse is
$1/15 \approx 0.0667$. Each entry of $H$ is $x_i x_j / 15$. The following
table shows each step, rounded to four decimal places:

| Example | Leverage $h_{ii}$ | Fitted $\hat{y}_i$ | Residual $r_i$ | LOO residual $r_i/(1-h_{ii})$ |
|---|---|---|---|---|
| 1 | $1/15 = 0.0667$ | $0.7333$ | $0.2667$ | $0.2857$ |
| 2 | $4/15 = 0.2667$ | $1.4667$ | $0.5333$ | $0.7273$ |
| 3 | $9/15 = 0.6$ | $2.2$ | $-0.2$ | $-0.5$ |

The fitted values come from $w = X^{\top}y / 15 = 11/15$. Refitting three
times without each example gives the same three LOO residuals. The
leave-one-out MSE is $(0.2857^2 + 0.7273^2 + 0.5^2)/3 = 0.2869$.

## Choose a setting without fooling yourself

Suppose you run cross-validation for twenty ridge penalties and pick the one
with the lowest score. That score is too good. Remember the thousand coins:
among twenty penalties, one tends to look lucky on these particular folds.
The more settings you compare, the more optimistic the winning score.

*Nested cross-validation* keeps the two jobs apart. An outer loop splits the
data into folds for scoring. Inside each outer training set, an inner
cross-validation picks the penalty using only that data. The outer fold then
scores the result on examples the inner loop never saw. The outer score
estimates how well the whole pick-then-fit recipe works.

> [!KEY]
> The best score among many settings is too optimistic. To report an honest number, score the choice itself on data the choice never saw.

## Learning curves

A *learning curve* plots training error and validation error against the
training-set size. With few examples, a model fits them closely, so
training error is low and validation error is high. As the set grows,
training error rises, validation error falls, and the two approach each
other.

Reading the curve tells you what to do next:

- Both curves meet at a high error: more data won't help. You need a more
  flexible model or better features.
- A wide gap remains at the largest size: the model overfits. More data,
  stronger regularisation, or a simpler model all help.

To make the curves smooth, hold out one fixed validation set, shuffle the
rest once, and train on the first `size` examples of that shuffle. Each
training set then contains the smaller ones.

> [!RECAP]
> - The theoretical gap shrinks like $1/\sqrt{n}$, but the bounds are too loose to pick a model.
> - k-fold cross-validation shuffles once, splits with `np.array_split`, and records the MSE on each held-out fold.
> - For ridge, the leave-one-out residual is $r_i / (1 - h_{ii})$, where $h_{ii}$ is diagonal entry $i$ of the hat matrix.
> - A learning curve uses one fixed validation set and trains on growing prefixes of one shuffled pool.
> - The random-sign estimate takes the best model for each draw, averages over draws, and divides by $n$.

## Build it step by step

Build the functions in the order the checks test them.

1. **`k_fold_indices(n, k, rng)`**. Returns a list of `k` sorted integer
   arrays that together hold every index from 0 to `n - 1` once. Use
   `rng.permutation(n)`, `np.array_split`, and `np.sort`. Common mistake:
   using `np.split`, which fails when `k` doesn't divide `n`.
2. **`cross_val_score(fit, predict, X, y, k)`**. Returns an array of shape
   `(k,)`. Create `rng = np.random.default_rng(0)` inside the function so
   every call uses the same folds. For each fold, find the training indices
   with `np.setdiff1d`, call `model = fit(X[train], y[train])`, then
   `predict(model, X[test])`, and store the MSE. Common mistake: passing in
   or reusing an outside random generator, which makes two calls disagree.
3. **`loo_linear_smoother(X, y, lam)`**. Returns one float. Build the hat
   matrix, take `np.diag`, divide the residuals by `1 - leverage`, square,
   and take `np.mean`. Common mistake: forgetting `lam * np.eye(d)`, which
   breaks the match at `lam > 0`.
4. **`learning_curve(fit, predict, X, y, train_sizes, val_fraction, seed)`**.
   Returns two arrays, each with one entry per size. Draw
   `order = np.random.default_rng(seed).permutation(n)`, take the first
   `int(round(val_fraction * n))` indices as the validation set and the rest
   as the pool. For each size, train on `pool[:size]` and record the MSE on
   those training examples and on the validation set. Common mistake:
   reshuffling for each size, which makes the curve jumpy.
5. **`empirical_rademacher(losses_matrix, rng, num_draws)`**. Returns one
   float. Draw a `(num_draws, n)` array of signs with `rng.choice`, compute
   `losses_matrix @ signs.T` to get shape `(num_models, num_draws)`, take the
   `max` over `axis=0`, then the mean, then divide by `n`. Common mistake:
   taking the max over the wrong axis, which returns the best draw instead of
   the best model.

## Cost and scaling

The cost depends on $n$ examples, $d$ features, and $k$ folds. For ridge
regression, one fit costs about $n \times d^2$ work. The following table
compares the methods:

| Method | Work compared with one fit | In plain words |
|---|---|---|
| k-fold cross-validation | about $k$ fits | Five folds cost about five fits. |
| Leave-one-out by refitting | $n$ fits | Slow once $n$ is in the thousands. |
| Leave-one-out by the shortcut | about one fit, plus the $n$-by-$n$ hat matrix | Fast for small $n$, but $H$ needs $n^2$ memory. |
| Nested cross-validation | outer folds times inner folds times settings | Grows fast, so keep the grid small. |
| Random-sign estimate | models times $n$ times draws | Cheap for the sizes in this exercise. |

The shortcut is why leave-one-out is practical for linear models. Without
it, its $n$ refits are why 5-fold or 10-fold cross-validation is the default.

## Where this breaks

**Data that isn't i.i.d.** Every method here assumes the examples are
independent. Time series break that: a random split lets the model train on
the future and test on the past. Grouped data, such as several scans from
each patient, breaks it too. Use forward-in-time splits for time series and
keep each group inside one fold.

**Small data with many settings.** With only a few dozen examples, nested
cross-validation has too little data in each loop for either estimate to be
trustworthy. Compare fewer settings.

**Reusing folds until they stop being fair.** After many rounds of
choosing features and settings by the same cross-validation score, that
score has shaped your choices. It no longer measures performance on unseen
data. Keep a final test set that you use once, at the end.

**Leverage close to 1.** If an example has $h_{ii}$ near 1, the shortcut
divides by a number near zero. With $\lambda = 0$ and $d$ close to $n$, this
happens often. A positive $\lambda$ keeps every leverage below 1.

## The exercise

You implement `k_fold_indices`, `cross_val_score`, `loo_linear_smoother`,
`learning_curve`, and `empirical_rademacher`. The `fit` and `predict`
arguments follow the pattern `model = fit(X_train, y_train)` and
`y_pred = predict(model, X_test)`, so your functions work with any model.

The checks confirm `k_fold_indices` uses every index once in folds whose
sizes differ by at most one, and that `cross_val_score` returns the same
`k` finite scores on every call. They confirm `loo_linear_smoother` matches
brute-force refitting for three penalties, and that `learning_curve` shows
validation error falling as the training set grows. They confirm
`empirical_rademacher` is near 0 for one model, grows for 300 models, and
reaches 1 for a set that matches every sign pattern. The last check uses
`cross_val_score` to pick a ridge penalty and confirms it clearly beats both
no penalty and a very large one on fresh test data.
