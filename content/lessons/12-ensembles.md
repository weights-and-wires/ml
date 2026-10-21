---
title: Bagging, forests, and boosting
slug: 12-ensembles
part: "Part 2 — Classical machine learning"
summary: "Average jumpy trees, or add stumps that fix leftover error, and estimate test error from missed rows."
minutes: 60
gpu: false
objectives:
  - Compute the variance of an average of correlated models, and explain why more trees stop helping.
  - Draw bootstrap samples, and estimate test error from the rows each tree missed.
  - Build gradient boosting so each stump fits the gap the current sum still misses.
  - Place random forests and AdaBoost in those same two ideas.
prereqs:
  - 11-decision-trees
exercise: 12-ensembles
---

# Bagging, forests, and boosting

A single decision tree is jumpy. Change a few training rows and the first split can flip, and every split under it flips too.

Averaging a crowd beats most single guesses. Bagging trains many trees on slightly different copies of your rows and averages them. Boosting is a relay. Each stump fits the gap the sum still misses, and you add only a fraction of it.

You can tell when extra trees stop helping, score rows a tree never saw, and run gradient boosting so training loss falls each round.

## What you already know

Five earlier ideas are enough to start:

- A pure-leaf tree fits its training rows and still misses other rows.
- A stump is one split and two leaves.
- Variance is an average squared gap. For 1 and 3, the mean is 2 and the variance is $(1+1)/2 = 1$.
- Correlation is 1 when two lists move together and 0 when they do not. The lists 1, 2, 3 and 2, 4, 6 have correlation 1.
- A gradient is a slope. For $w^2$ at $w = 3$ the slope is 6. A nudge of 0.01 raises it by about 0.06.

## Average two noisy guesses

Two models each have variance 4, and correlation $\rho = 0.5$. The shared piece is $0.5 \times 4 = 2$, and averaging leaves it alone. The private 2, split across both models, leaves 1. The average has variance $2 + 1 = 3$.

This split is the **variance of an average**.

$$
\operatorname{Var}(\text{average}) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2 .
$$

The first term, $\rho\sigma^2$, is the shared variance. It ignores $B$. The second term splits the private variance across $B$ models.

Plug in $0.5$, $4$, and $2$. You get 3 again. With $\rho = 0$ the result is $4/2 = 2$. With $\rho = 1$, or with $B = 1$, it stays 4. At $\rho = 0.3$, $\sigma^2 = 1$, and $B = 50$ you get $0.3 + 0.014 = 0.314$. At $B = 5000$ you get $0.30014$. More trees then only cost time. Lower $\rho$ to lower the floor:

```python
var_avg = rho * sigma2 + (1 - rho) * sigma2 / B
```

The usual mistake is dividing the shared term by $B$ as well.

> [!KEY]
> Averaging shrinks only the private part of the variance. The shared part $\rho\sigma^2$ stays, however large $B$ gets.

## Draw a bootstrap sample

You have $n = 4$ rows. Draw `idx = [0, 0, 2, 3]`, with replacement. Row 0 appears twice, rows 2 and 3 once, and row 1 never. The in-bag fraction is $3/4$. Row 1 is **out-of-bag**. This draw is a **bootstrap** sample: $n$ draws from $n$ rows, with replacement.

A given row misses one draw with probability $1 - 1/n$. It misses all $n$ draws with probability

$$
\left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx 0.368 .
$$

It is the one-draw miss chance, $n$ times. For large $n$ that is about $1/e$, so about 36.8% of rows are missed and 63.2% appear. At $n = 4$, $(3/4)^2 = 0.5625$ and $0.5625^2 = 0.3164$. Your draw missed 1 of 4 rows, one outcome of that chance:

```python
idx = rng.integers(0, n, size=n)
```

The usual mistake is `rng.choice(n, n, replace=False)`. That shuffles, so nothing is out-of-bag.

> [!TIP]
> Build the in-bag mask with `in_bag[idx] = True`. A repeated index writes the same slot twice, so you do not need `np.unique`.

> [!KEY]
> For large $n$, a bootstrap sample holds about 63.2% of the distinct rows. The rest are out-of-bag for that tree.

## Score the rows a tree missed

Average, for each row, only the trees that missed it. That mean squared error is the **out-of-bag error**.

Take $x = [1, 2, 3, 4]$ and $y = [0, 10, 10, 0]$. The draw `[0, 0, 1, 1]` splits at 1.5, with leaf values 0 and 10. Out-of-bag rows 2 and 3 both get 10. The draw `[2, 2, 3, 3]` splits at 3.5, with leaf values 10 and 0. Out-of-bag rows 0 and 1 both get 10.

The held-out scores line up as follows:

| Row | $y$ | Prediction | Squared error |
|---|---|---|---|
| 0 | 0 | 10 | 100 |
| 1 | 10 | 10 | 0 |
| 2 | 10 | 10 | 0 |
| 3 | 0 | 10 | 100 |

The mean is $(100 + 0 + 0 + 100) / 4 = 50$.

Tree 1 predicts 0 on row 3, but that row was in its bag. Averaging both trees there gives 5, which is not held out.

Add each stump's out-of-bag predictions into `total`, and add 1 into `count`. Then:

```python
pred = total[count > 0] / count[count > 0]
oob = np.mean((pred - y[count > 0]) ** 2)
```

The usual mistake is averaging every stump on every row. That is training error.

> [!KEY]
> Score each row with only the trees that missed it. The mean squared error of those scores is a held-out estimate.

> [!WARNING]
> Some rows stay in every bag and keep `count == 0`. Dividing there returns `nan`. Select `count > 0` first.

## Hide features so the trees disagree

If one feature dominates, every bagged stump splits on it and $\rho$ stays high. A **random forest** shows each split only a random subset of columns. Some trees miss the strong feature. Each tree is a bit worse, and $\rho$ falls.

The same identity, with $\sigma^2 = 1$ and $B = 50$, gives $0.804$ at $\rho = 0.8$ and $0.216$ at $\rho = 0.2$. The usual mistake is calling another bootstrap a forest. The change is which columns a split may see. This exercise bags rows only.

## Add one stump for the leftover error

Take $x = [1, 2, 3, 4]$ and $y = [1, 3, 8, 10]$, with $F = 0$. Squared labels 1, 9, 64, and 100 sum to 174. The loss, half the mean, is $0.5 \times 174/4 = 21.75$. The best stump splits at 2.5. Leaf means are $(1+3)/2 = 2$ and $(8+10)/2 = 9$, so $h = [2, 2, 9, 9]$. At $\eta = 0.5$, $F = [1, 1, 4.5, 4.5]$. Residual squares 0, 4, 12.25, and 30.25 sum to 46.5, and the loss is $0.5 \times 46.5/4 = 5.8125$.

This is **gradient boosting**, or functional gradient descent. You step $F$ by a stump fitted to the negative gradient:

$$
g_i = -\frac{\partial\, \ell(y_i, F_i)}{\partial F_i} .
$$

$g_i$ is minus the slope of $\ell$ in that row's prediction. The caller passes `loss_grad(y, F)`, and you set $F \leftarrow F + \eta h$. For squared error the slope is the residual:

$$
g_i = y_i - F_i .
$$

On round 1, $F = 0$, so $g = y = [1, 3, 8, 10]$, the target you fitted:

```python
target = loss_grad(y, F)
F = F + learning_rate * _predict_stump(_fit_stump(X, target), X)
```

The usual mistake is fitting every round to $y$. After round 1 the target is the residual.

Each leaf predicts the mean of its targets, so one round drops the mean loss by

$$
L_{\text{before}} - L_{\text{after}} = \eta\left(1 - \frac{\eta}{2}\right) \cdot \frac{1}{n}\sum_i h_i^2 .
$$

It is $\eta(1 - \eta/2)$ times the mean of $h^2$. Between $\eta = 0$ and $\eta = 2$ the loss falls, unless $h$ is all zeros. Here $h^2$ is 4, 4, 81, and 81, mean $170/4 = 42.5$. Then $0.5 \times 0.75 \times 42.5 = 15.9375$, and $21.75 - 5.8125 = 15.9375$:

```python
drop = eta * (1 - eta / 2) * np.mean(h ** 2)
```

The usual mistake is $\eta = 1$ for many rounds. The identity still holds, but later stumps fit noise. A small $\eta$, such as 0.1, keeps each step short.

> [!KEY]
> Under squared error, each round fits a stump to the residual $y - F$ and adds $\eta$ times that stump.

AdaBoost uses the loss $e^{-yF}$ for labels $-1$ and $+1$. At $y = 1$, $F = 0$ gives 1, and $F = 0.5$ gives $e^{-0.5} = 0.607$. At $y = -1$ and $F = 0.5$ the loss is $e^{0.5} = 1.649$, so a wrong row gets a larger weight next round. You do not code that rule. One mislabelled row can dominate because the loss grows so fast.

> [!RECAP]
> Keep these five facts in hand:
> - The variance of an average is $\rho\sigma^2 + (1-\rho)\sigma^2 / B$. Only the second term shrinks as $B$ grows.
> - A bootstrap sample is `rng.integers(0, n, size=n)`. For large $n$, about 36.8% of rows are out-of-bag.
> - Out-of-bag error averages only the stumps that missed each row, and it skips rows with `count == 0`.
> - Gradient boosting starts from $F = 0$, fits a stump to `loss_grad(y, F)`, and adds `learning_rate` times that stump.
> - Under squared error the first target is $y$, and one round drops the loss by $\eta(1-\eta/2)$ times the mean of $h^2$.

## Build it step by step

`_fit_stump` and `_predict_stump` are already written. Build the five functions in this order:

1. **`bagging_variance(rho, sigma2, B)`** takes three numbers and returns a float. Return $\rho\sigma^2 + (1-\rho)\sigma^2 / B$. The usual mistake is dividing the first term by `B`. Write it like this:

    ```python
    def bagging_variance(rho, sigma2, B):
        # Return rho * sigma2 + (1 - rho) * sigma2 / B as a float.
    ```

2. **`bootstrap_indices(n, rng)`** returns integers of shape `(n,)`, each from 0 to `n - 1`. Call `rng.integers(0, n, size=n)`. The usual mistake is sampling without replacement. Write it like this:

    ```python
    def bootstrap_indices(n, rng):
        # n draws with replacement on 0, ..., n - 1.
    ```

3. **`fit_bagged_stumps(X, y, n_estimators, rng)`** returns stump dicts and index arrays, one pair per estimator. Draw `idx`, fit `_fit_stump(X[idx], y[idx])`, and keep both. The usual mistake is dropping the indices. Write it like this:

    ```python
    def fit_bagged_stumps(X, y, n_estimators, rng):
        # For each estimator, store the stump and its bootstrap idx.
    ```

4. **`out_of_bag_error(X, y, stumps, in_bag_indices)`** returns a float. Keep `total` and `count` of shape `(n,)`. Add each stump only on its out-of-bag rows. Average squared error where `count > 0`. The usual mistake is using every stump on every row. Write it like this:

    ```python
    def out_of_bag_error(X, y, stumps, in_bag_indices):
        # Accumulate on the out-of-bag mask, then mean squared error.
    ```

5. **`gradient_boost(X, y, loss_grad, n_rounds, learning_rate)`** returns `F` of shape `(n,)` and `n_rounds` dicts. Start from `np.zeros(n)`. Each round, fit a stump to `loss_grad(y, F)` and add `learning_rate` times its predictions. Append `{"target": target, "prediction": F.copy()}`. The usual mistake is appending `F` without `.copy()`. Write it like this:

    ```python
    def gradient_boost(X, y, loss_grad, n_rounds, learning_rate):
        # Step F from zeros. Copy F into each history dict.
    ```

## Cost and scaling

The costs for $B$ trees or $M$ rounds, $n$ rows, and $d$ features line up as follows:

| Task | How the work grows | Notes |
|---|---|---|
| One bootstrap sample | Proportional to $n$ | $n$ random draws. |
| Bagging $B$ stumps | About $B \cdot d \cdot n \log n$ | Independent trees can run together. |
| Out-of-bag error | Proportional to $B \cdot n$ | One prediction per tree per row. |
| Boosting $M$ rounds | $M$ times one stump fit | Round $m$ reads $F$ from round $m-1$. |

## Where this breaks

Four failures show up in practice:

- Past the $\rho\sigma^2$ floor, more trees only add time. A random column subset at each split lowers $\rho$.
- A large learning rate and many rounds make boosting fit noise. Keep `learning_rate` small and stop when validation stops improving.
- AdaBoost's exponential loss lets one mislabelled row take most of the weight. A slower-growing loss is more robust.
- Averaging a biased model stays biased. Boosting can cut bias, and each round adds variance, so the learning rate stays small.

## The exercise

You implement `bagging_variance`, `bootstrap_indices`, `fit_bagged_stumps`, `out_of_bag_error`, and `gradient_boost`.

The checks hit the edges $\rho = 0$, $\rho = 1$, and $B = 1$, plus a correlated simulation. `bootstrap_indices` must have shape `(n,)`, stay in range, repeat rows, and land near in-bag fraction $1 - 1/e$. `fit_bagged_stumps` returns one stump and one index array per estimator. `out_of_bag_error` tracks a real held-out error. For `gradient_boost`, the first target is $y$, loss falls every round, and one round's drop matches $\eta(1-\eta/2)$ times the mean of $h^2$.
