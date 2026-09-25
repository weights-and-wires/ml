---
title: Bagging, forests, and boosting
slug: 12-ensembles
part: "Part 2 — Classical machine learning"
summary: Combine many weak trees into one strong model, by averaging (bagging) or by fixing mistakes in turn (boosting).
minutes: 60
gpu: false
objectives:
  - Compute the variance of an average of correlated models, and explain why adding trees stops helping.
  - Draw bootstrap samples, and use out-of-bag rows to estimate test error for free.
  - Build gradient boosting, where each new stump fits what the model still gets wrong.
  - Explain how random forests and AdaBoost fit into the same two ideas.
prereqs:
  - 11-decision-trees
exercise: 12-ensembles
---

# Bagging, forests, and boosting

A single decision tree is jumpy. Change a few training rows and the root split can flip, and then every split beneath it changes too. Its predictions have high **variance**: they swing a lot from one training set to the next.

There is an old trick for jumpy guesses: ask a crowd and average the answers. If you ask 100 people to guess the weight of a cow, the average is usually closer than most single guesses. **Bagging** does this with trees. It trains many trees on slightly different copies of the data and averages them.

**Boosting** takes the opposite route. Instead of a crowd guessing at once, it is a team taking turns. Each new small tree looks at what the team still gets wrong and fixes a bit of it. This lesson builds both.

## What you already know

- A decision tree grown until its leaves are pure fits the training data perfectly and generalises badly.
- A stump is a tree with one split and two leaves.
- The variance of a random quantity measures how much it spreads around its average.
- The correlation $\rho$ between two quantities is 1 when they always move together and 0 when they are unrelated.
- A gradient points in the direction a function increases fastest, so its negative points downhill.

## Why averaging helps, and where it stops

Suppose you have $B$ models. Each one's prediction has variance $\sigma^2$. Any two of them have correlation $\rho$. Bagging predicts with their average. The variance of that average is

$$
\operatorname{Var}(\text{average}) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2 .
$$

Read it aloud in two pieces. The second piece, $(1-\rho)\sigma^2 / B$, shrinks as you add more models $B$. The first piece, $\rho\sigma^2$, does not depend on $B$ at all. It is a floor. However many trees you average, the variance cannot drop below $\rho\sigma^2$.

```python
var_avg = rho * sigma2 + (1 - rho) * sigma2 / B
```

Try some numbers. With $\rho = 0.3$, $\sigma^2 = 1$, and $B = 50$, you get $0.3 + 0.7/50 = 0.3 + 0.014 = 0.314$. Going to $B = 5000$ only gets you to $0.30014$. Three extreme cases are worth remembering:

- $\rho = 0$: the models are independent, and the variance is $\sigma^2 / B$.
- $\rho = 1$: the models are identical, and averaging does nothing. The variance stays $\sigma^2$.
- $B = 1$: one model, variance $\sigma^2$, whatever $\rho$ is.

The lesson: past a point, more trees do not help. What helps is making the trees less alike, so $\rho$ falls.

## The bootstrap

To train many different trees, you need many different training sets. You only have one. The **bootstrap** fakes new ones: draw $n$ rows from your $n$ training rows, at random, **with replacement**. With replacement means a row can be picked more than once, and some rows are not picked at all.

```python
idx = rng.integers(0, n, size=n)   # n random row numbers from 0 to n-1
X_boot, y_boot = X[idx], y[idx]
```

How many distinct rows does one bootstrap sample contain? A given row is missed by one draw with probability $1 - 1/n$. It is missed by all $n$ draws with probability

$$
\left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx 0.368 .
$$

Read it aloud: the chance of missing a row once, multiplied by itself $n$ times. For large $n$ this is close to $1/e$. So about 36.8% of rows are left out of each sample, and about 63.2% ($1 - 1/e$) are in it. Those left-out rows are called **out-of-bag** for that tree.

```python
in_bag_fraction = len(np.unique(idx)) / n   # close to 0.632 for large n
```

### A worked example

With $n = 4$ rows, suppose the draw is `idx = [0, 0, 2, 3]`. Row 0 appears twice, rows 2 and 3 once, and row 1 never. The in-bag fraction is $3/4$, and row 1 is out-of-bag for this tree. You can mark it with a boolean mask:

```python
in_bag = np.zeros(4, dtype=bool)
in_bag[idx] = True       # [True, False, True, True]
oob = ~in_bag            # [False, True, False, False]
```

## Out-of-bag error: a free test set

Each row is out-of-bag for roughly a third of the trees. Those trees never saw that row. So if you average only their predictions for that row, you get an honest, held-out prediction.

Do that for every row, and compute the mean squared error against the true labels. The result is the **out-of-bag (OOB) error**. It estimates test error without setting aside any data, and it tracks cross-validation closely once you have many trees.

The recipe for each tree $b$:

1. Build the out-of-bag mask for tree $b$.
2. Add tree $b$'s predictions on those rows to a running `total`.
3. Add 1 to a running `count` for those rows.

Then the OOB prediction for each row is `total / count`, over rows with `count > 0`, and the OOB error is the mean of the squared differences from `y`.

## Random forests: make the trees disagree

Suppose one feature is very strong. Almost every bagged tree splits on it at the root, so the trees look alike and $\rho$ stays high. A **random forest** fixes this: at every split, each tree may only look at a random subset of the features. Some trees then have to use the second-best feature. Each tree gets a little worse, but they disagree more, so $\rho$ drops and the floor $\rho\sigma^2$ drops with it. The exercise uses plain bagging, but this is the one change that turns bagging into a forest.

## Boosting: fix the mistakes in turn

Boosting builds a model $F$ as a sum of small trees, added one at a time. Start with $F = 0$ for every row. Then repeat:

1. Work out, for every row, which way to nudge $F$ to lower the loss.
2. Fit a stump $h$ to those nudges.
3. Add a small step of that stump: $F \leftarrow F + \eta\,h$.

The **learning rate** $\eta$ (eta) is a number such as $0.1$ that keeps each step small.

The nudge in step 1 is the negative gradient of the loss with respect to each prediction:

$$
g_i = -\frac{\partial\, \ell(y_i, F_i)}{\partial F_i} .
$$

Read it aloud: for row $i$, ask how the loss $\ell$ changes when you change that row's prediction $F_i$, and go the opposite way. It is ordinary gradient descent, except the step must come from a stump, so it generalises to new rows.

In the exercise, the caller passes this as a function, `loss_grad(y, F)`, which returns the whole vector $g$.

### Squared error means fitting the residuals

Take the loss $\ell = \tfrac12 (y - F)^2$. Its negative gradient is

$$
g_i = y_i - F_i ,
$$

which is the **residual**: what the model still gets wrong on row $i$. So under squared error, each round fits a stump to the leftover error. In the first round $F = 0$, so the target is the raw label $y$ itself.

```python
target = loss_grad(y, F)             # y - F under squared error
stump = _fit_stump(X, target)
F = F + learning_rate * _predict_stump(stump, X)
```

### How much one round helps

Measure the training loss as the mean over rows, $L = \frac{1}{n}\sum_i \tfrac12 (y_i - F_i)^2$. Because each stump leaf predicts the mean of the targets in that leaf, one round with learning rate $\eta$ lowers the loss by exactly

$$
L_{\text{before}} - L_{\text{after}} = \eta\left(1 - \frac{\eta}{2}\right) \cdot \frac{1}{n}\sum_i h_i^2 ,
$$

where $h_i$ is the stump's prediction on row $i$. Read it aloud: the drop equals $\eta(1 - \eta/2)$ times the average squared stump output. For any $\eta$ between 0 and 2, that factor is positive, so the loss falls every round unless the stump predicts all zeros.

```python
drop = eta * (1 - eta / 2) * np.mean(h ** 2)
```

### A worked round

Take $x = [1, 2, 3, 4]$ and $y = [1, 3, 8, 10]$, starting from $F = 0$. The starting loss is $\tfrac12 \cdot (1 + 9 + 64 + 100)/4 = 21.75$.

The target is $y$. The best stump splits at $x \le 2.5$ and predicts each side's mean, so $h = [2, 2, 9, 9]$. With $\eta = 0.5$, the new predictions are $F = [1, 1, 4.5, 4.5]$.

The residuals are $[0, 2, 3.5, 5.5]$. Their squares sum to $0 + 4 + 12.25 + 30.25 = 46.5$, so the new loss is $\tfrac12 \cdot 46.5 / 4 = 5.8125$. The formula agrees: the mean of $h^2$ is $(4 + 4 + 81 + 81)/4 = 42.5$, and $0.5 \times 0.75 \times 42.5 = 15.9375 = 21.75 - 5.8125$.

### AdaBoost in one paragraph

AdaBoost is boosting with a different loss, the exponential loss $e^{-yF}$ for labels $y$ of $-1$ or $+1$. It keeps a weight on each row. After each round, it scales a correctly classified row's weight down and a misclassified row's weight up, so the next weak learner focuses on the mistakes. That rule falls out of the same "follow the negative gradient" recipe. You do not implement it here.

## Build it step by step

The checks test the functions in this order. `_fit_stump` and `_predict_stump` are already written for you.

1. **`bagging_variance(rho, sigma2, B)`.** Takes three numbers and returns a float. It is the one-line formula above. Common mistake: dividing the whole thing by `B`, instead of only the `(1 - rho)` part.
2. **`bootstrap_indices(n, rng)`.** Returns an integer array of shape `(n,)` with values from `0` to `n - 1`. Use `rng.integers(0, n, size=n)`. Common mistake: using `rng.choice(n, n, replace=False)`, which is a shuffle, not a bootstrap.
3. **`fit_bagged_stumps(X, y, n_estimators, rng)`.** Returns two lists of length `n_estimators`: the stump dicts and the index arrays. For each estimator, draw `idx`, fit `_fit_stump(X[idx], y[idx])`, and keep `idx`. Common mistake: forgetting to return the indices, which `out_of_bag_error` needs.
4. **`out_of_bag_error(X, y, stumps, in_bag_indices)`.** Returns a float. Keep `total` and `count` arrays of shape `(n,)`. For each stump, build the boolean out-of-bag mask, and add that stump's predictions on those rows. Average only the rows with `count > 0`. Common mistake: averaging over all stumps for every row, which reports training error, not held-out error.
5. **`gradient_boost(X, y, loss_grad, n_rounds, learning_rate)`.** Returns the final `F` of shape `(n,)` and a list of `n_rounds` dicts. Start from `np.zeros(n)`. Each round, compute `target`, fit a stump, update `F`, and append `{"target": target, "prediction": F.copy()}`. Common mistake: appending `F` without `.copy()`, or updating `F` in place, so every history entry ends up the same.

## Cost and scaling

The following table compares the work with $B$ bagged trees or $M$ boosting rounds, $n$ rows, and $d$ features.

| Task | How the work grows | Notes |
|---|---|---|
| One bootstrap sample | Proportional to $n$ | $n$ random draws. |
| Bagging $B$ stumps | $B$ times one stump fit, about $B \cdot d \cdot n \log n$ | Trees are independent, so they run in parallel. |
| Out-of-bag error | Proportional to $B \cdot n$ | One prediction per tree per row. |
| Boosting $M$ rounds | $M$ times one stump fit | Strictly one after another. |

Bagging parallelises across trees. Boosting cannot, because each round needs the previous round's predictions.

## Where this breaks

- **Bagging plateaus.** Once the $\rho\sigma^2$ floor dominates, more trees only cost time. Use feature subsampling to lower $\rho$.
- **Boosting overfits.** With a large learning rate and many rounds, later stumps fit noise. Use a small `learning_rate`, fewer rounds, and stop when a validation score stops improving.
- **AdaBoost and label noise.** The exponential loss grows very fast for badly wrong rows, so a mislabelled row can grab most of the weight. Losses that grow more slowly are more robust.
- **Bias stays.** Averaging copies of a biased model gives a biased average. Boosting can lower bias, but it adds variance with each round, which is why it needs a small learning rate.

## The exercise

You implement the core of both families. Write `bagging_variance` and `bootstrap_indices`, then `fit_bagged_stumps`, which fits one stump per bootstrap sample, and `out_of_bag_error`, which scores the ensemble using only the stumps that did not see each row. Finally write `gradient_boost`, which starts from zero and adds `learning_rate` times a stump fitted to `loss_grad(y, F)` each round.

The checks confirm `bagging_variance` at its three extremes and against a random simulation. They check that `bootstrap_indices` has the right shape and range and an in-bag fraction near $1 - 1/e$, that `fit_bagged_stumps` returns one stump and one index array per estimator, and that `out_of_bag_error` lands near a true held-out error. For `gradient_boost`, they check that the first target is the raw labels, that training loss falls every round, and that one round's drop matches $\eta(1-\eta/2)$ times the mean of $h^2$ for several learning rates.
