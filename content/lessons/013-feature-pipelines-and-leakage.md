---
title: "Feature pipelines and data leakage"
slug: 013-feature-pipelines-and-leakage
part: "Part 2 — Classical models"
summary: Fit every preprocessing step on training rows only, refit it inside each fold, and catch the leaks that make pure noise look predictable.
minutes: 70
gpu: false
objectives:
  - Build standardising and imputing steps that store training-only statistics, and explain why transform never reads the test rows.
  - Encode categories with one-hot columns that have an unknown bucket, and with smoothed out-of-fold target means.
  - Refit feature selection and scaling inside each cross-validation fold, and show that selecting first scores well on pure noise.
  - Recognise target leakage, duplicate rows across folds, and time leakage by how they look in a score.
prereqs:
  - 012-cross-validation-and-leave-one-out
  - 001-loss-and-honest-scores
exercise: 013-feature-pipelines-and-leakage
---

# Feature pipelines and data leakage

You have 50 patients and 500 measurements for each, such as blood markers and gene activity. You want to predict a yes-or-no outcome. Fitting a model on all 500 columns would memorise 50 rows, so you first keep the 10 columns that correlate best with the outcome. Then you run five-fold cross-validation, as in the [previous chapter](/lessons/012-cross-validation-and-leave-one-out), and it reports 86 percent accuracy. You celebrate.

Here is the catch. The outcomes are coin flips and the measurements are random numbers, so no model can beat 50 percent. The cross-validation was honest, but you had already used every outcome, held-out ones included, to choose the columns. The held-out rows were not hidden at all.

This chapter shows where that kind of mistake hides and how to design it out. You will build preprocessing steps that learn their numbers from training rows only, and a scorer that refits the whole pipeline inside every fold. Along the way you will encode categories, fill in missing values, and learn to read a suspiciously good score as a symptom.

## What you already know

- A model scored on rows it never saw gives an honest estimate. See [honest scores](/lessons/001-loss-and-honest-scores).
- In k-fold cross-validation, every row is held out once while a model trains on the other $k - 1$ folds. See [cross-validation](/lessons/012-cross-validation-and-leave-one-out).
- A column's mean is its average, and its standard deviation is the typical distance from that mean. Standardising a column subtracts the mean and divides by the standard deviation.
- The correlation of two columns is a number between $-1$ and $1$ that says how closely they rise and fall together.
- A nearest-centroid classifier averages each class's rows into one centre and labels a point by the closer centre.

## The leak, on pure noise

Figure 1 runs the opening experiment 200 times. Each run makes 50 rows with 500 random columns and random 0 or 1 labels. The classifier is the nearest-centroid classifier, and the score is five-fold cross-validated accuracy. The orange line picks the best columns on all 50 rows and then cross-validates. The blue line picks the best columns again inside each fold, using only that fold's training rows.

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="Cross-validated accuracy on pure noise rises to 0.96 when columns are chosen on all rows, and stays at 0.5 when they are chosen inside each fold">
<line x1="70" y1="206.7" x2="570" y2="206.7" class="stroke-grid"/>
<text x="58" y="210.7" text-anchor="end" class="fig-small">0.5</text>
<line x1="70" y1="173.3" x2="570" y2="173.3" class="stroke-grid"/>
<text x="58" y="177.3" text-anchor="end" class="fig-small">0.6</text>
<line x1="70" y1="140" x2="570" y2="140" class="stroke-grid"/>
<text x="58" y="144" text-anchor="end" class="fig-small">0.7</text>
<line x1="70" y1="106.7" x2="570" y2="106.7" class="stroke-grid"/>
<text x="58" y="110.7" text-anchor="end" class="fig-small">0.8</text>
<line x1="70" y1="73.3" x2="570" y2="73.3" class="stroke-grid"/>
<text x="58" y="77.3" text-anchor="end" class="fig-small">0.9</text>
<line x1="70" y1="40" x2="570" y2="40" class="stroke-grid"/>
<text x="58" y="44" text-anchor="end" class="fig-small">1.0</text>
<line x1="70" y1="240" x2="570" y2="240" class="stroke-muted"/><line x1="70" y1="30" x2="70" y2="240" class="stroke-muted"/>
<text x="70" y="258" text-anchor="middle" class="fig-small">1</text>
<text x="153.3" y="258" text-anchor="middle" class="fig-small">2</text>
<text x="236.7" y="258" text-anchor="middle" class="fig-small">5</text>
<text x="320" y="258" text-anchor="middle" class="fig-small">10</text>
<text x="403.3" y="258" text-anchor="middle" class="fig-small">20</text>
<text x="486.7" y="258" text-anchor="middle" class="fig-small">50</text>
<text x="570" y="258" text-anchor="middle" class="fig-small">100</text>
<text x="320" y="282" text-anchor="middle">columns kept out of 500</text>
<text x="14" y="135" transform="rotate(-90 14 135)" text-anchor="middle">cross-validated accuracy</text>
<line x1="70" y1="206.7" x2="570" y2="206.7" class="stroke-muted dashed"/>
<polyline points="70.0,145.3 153.3,129.3 236.7,105.9 320.0,86.9 403.3,71.9 486.7,56.9 570.0,52.9" fill="none" class="stroke-b" stroke-width="2.5"/>
<polyline points="70.0,205.4 153.3,204.8 236.7,206.6 320.0,206.0 403.3,206.6 486.7,206.3 570.0,207.3" fill="none" class="stroke-a" stroke-width="2.5"/>
<circle cx="320" cy="86.9" r="6" class="fill-b"/>
<text x="332" y="108" class="fig-bold">selected first: 0.86 at 10 columns</text>
<circle cx="320" cy="206" r="6" class="fill-a"/>
<text x="332" y="194" class="fig-bold">selected inside folds: 0.50</text>
<text x="500" y="222" class="fig-small">chance</text>
</svg>
Figure 1. On labels that are pure coin flips, choosing columns before cross-validating (orange) scores 0.86 with 10 columns and 0.96 with 100. Choosing them inside each fold (blue) stays at 0.50, which is the truth. Each point averages 200 seeded datasets.
```

Look at the orange line. One column out of 500 already gives 0.68, ten give 0.86, and a hundred give 0.96. The blue line sits at 0.50 for every count. Only the order of two steps changed between the lines.

Why does the orange line climb? With 500 random columns and 50 rows, some columns agree with the labels by luck, the way one of 500 coin-flippers will guess a 50-flip sequence better than the rest. Choosing on all rows finds those lucky columns, and the lucky columns include luck about the held-out rows. The model then "predicts" a held-out label because that very label helped pick the column.

> [!KEY]
> A step that looks at the labels, such as choosing columns by their correlation with the outcome, is part of the model. It has to run inside the cross-validation loop, on training rows only. Otherwise the held-out rows have already voted on the model they are about to grade.

## Fit on the training rows, transform everything

The cure is a rule, and it applies to every preprocessing step, not only selection. Anything that computes a number from data is a fitted object. It has two jobs. `fit` looks at training rows and stores numbers. `transform` applies the stored numbers to any rows, and never looks at the rows to decide anything.

Take standardising. The obvious approach is to standardise the whole table once and then split it. Why not? Because the mean and standard deviation of all rows include the held-out rows, and a held-out row then nudges the very numbers used to rescale it. For a mean the nudge is small, and for column selection it is enormous, but one rule covers both.

Work one column by hand. The training values are $2, 4, 6, 8$.

1. The mean is $(2 + 4 + 6 + 8)/4 = 5$.
2. The distances from the mean are $-3, -1, 1, 3$. Their squares are $9, 1, 1, 9$, which average to $5$, so the standard deviation is $\sqrt{5} = 2.2361$.
3. A test value of $10$ arrives. You use the stored numbers, so it becomes $(10 - 5)/2.2361 = 2.2361$.

If you had added the test value to the pool, the mean would have moved to 6, and the same 10 would map to 1.41. The test row would have changed the scale used to read it.

$$
z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}
$$

Read this aloud: subtract the training mean from the value, then divide by the training standard deviation, so the result counts how many training standard deviations the value sits from the training average. The subscript is the whole point. In code, the object keeps two arrays.

```python
class Standardiser:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)           # then swap any 0 for 1
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_
```

You can test the rule directly. Fit, record `mean_` and `std_`, then transform two test sets that differ by 1000 in every entry. The stored arrays must not move, and the outputs must differ by exactly $1000 / \text{std}$.

Figure 2 shows the flow for one fold. The training rows feed `fit`, which produces stored numbers. The stored numbers feed `transform` for both the training rows and the held-out rows. No arrow runs from the held-out rows into `fit`.

```figure
<svg viewBox="0 0 620 250" role="img" aria-label="Training rows feed fit, which stores mean 5 and standard deviation 2.2361. Both the training rows and the held-out row 10 pass through transform using those stored numbers. No arrow goes from the held-out row into fit.">
<defs>
<marker id="c13-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker>
</defs>
<rect x="20" y="30" width="150" height="56" rx="6" class="fill-soft-a stroke-a"/>
<text x="95" y="53" text-anchor="middle" class="fig-bold">training rows</text>
<text x="95" y="73" text-anchor="middle" class="fig-small">2, 4, 6, 8</text>
<rect x="230" y="30" width="160" height="56" rx="6" class="fill-soft-b stroke-b"/>
<text x="310" y="53" text-anchor="middle" class="fig-bold">fit</text>
<text x="310" y="73" text-anchor="middle" class="fig-small">mean 5, std 2.2361</text>
<rect x="450" y="30" width="150" height="56" rx="6" class="fill-soft-c stroke-c"/>
<text x="525" y="53" text-anchor="middle" class="fig-bold">stored numbers</text>
<text x="525" y="73" text-anchor="middle" class="fig-small">mean_, std_</text>
<line x1="170" y1="58" x2="228" y2="58" class="stroke-ink" marker-end="url(#c13-arrow)"/>
<line x1="390" y1="58" x2="448" y2="58" class="stroke-ink" marker-end="url(#c13-arrow)"/>
<rect x="230" y="160" width="160" height="56" rx="6" class="fill-soft-b stroke-b"/>
<text x="310" y="183" text-anchor="middle" class="fig-bold">transform</text>
<text x="310" y="203" text-anchor="middle" class="fig-small">(x - 5) / 2.2361</text>
<line x1="525" y1="86" x2="525" y2="188" class="stroke-ink"/>
<line x1="525" y1="188" x2="392" y2="188" class="stroke-ink" marker-end="url(#c13-arrow)"/>
<rect x="20" y="160" width="150" height="56" rx="6" class="fill-soft-a stroke-a"/>
<text x="95" y="183" text-anchor="middle" class="fig-bold">held-out row</text>
<text x="95" y="203" text-anchor="middle" class="fig-small">10, then 2.2361</text>
<line x1="170" y1="188" x2="228" y2="188" class="stroke-ink" marker-end="url(#c13-arrow)"/>
<line x1="95" y1="86" x2="95" y2="158" class="stroke-bad dashed"/>
<text x="105" y="128" class="fig-small">no arrow into fit</text>
</svg>
Figure 2. The stored numbers come from the training rows and only from them. The held-out value 10 goes through `transform` and becomes 2.2361. The dashed line marks the path that must not exist.
```

## Filling in missing values

Real tables have holes. A missing entry shows up as `NaN`, and most models cannot take one. The obvious fix is to fill each hole with the column's average. That is an imputer, and it follows the same rule, because the average is a number computed from data.

Take the training column $1, 2, \text{NaN}, 9$. The observed entries are $1, 2, 9$. Their mean is $4$ and their median, the middle value when sorted, is $2$. The one large value 9 drags the mean but not the median, so the median is the safer fill when a column has outliers. `fit` stores one fill value per column, skipping `NaN` when it computes it. `transform` writes that stored value into every hole, in training rows and test rows alike. A hole in a test row gets 4 (or 2), never an average of the test column.

```python
class Imputer:
    def fit(self, X):
        self.fill_ = np.array([np.nanmean(col) for col in X.T])  # or nanmedian
        return self

    def transform(self, X):
        X = X.copy()
        rows, cols = np.nonzero(np.isnan(X))
        X[rows, cols] = self.fill_[cols]
        return X
```

Chain the steps in the order you would run them on unseen data: impute first, then standardise on the filled values, each fitted on training rows only.

## Encoding categories

A column of words, such as `"red"`, `"green"`, and `"blue"`, has to become numbers before a model can use it. Replacing the words by 0, 1, and 2 is the obvious approach, and it fails because it tells the model that blue is twice green. The usual fix is one-hot encoding: one column per category, with a 1 in the column of the row's category and 0 elsewhere.

A second problem is the category that shows up at prediction time and was never in training. If the encoder only knows three colors, `"purple"` has no column. Reserve a last column, an unknown bucket, and send every unseen value to it. With the fitted categories `["red", "green", "blue"]`, the values `["green", "purple"]` become

| value | red | green | blue | unknown |
|---|---|---|---|---|
| green | 0 | 1 | 0 | 0 |
| purple | 0 | 0 | 0 | 1 |

Each row sums to 1, and the number of columns is the number of fitted categories plus one. The fitted category list is the stored statistic here, and it comes from training rows.

One-hot is wasteful for thousands of categories, such as zip codes. Target encoding replaces each category by a number derived from the target: the average outcome among training rows of that category. A category whose rows tend to have outcome 1 gets a value near 1.

A raw category average is fragile. A category with one row has an average equal to that row's label, which is a coin flip dressed up as a measurement. Smoothing pulls each average toward the overall average, harder when the category has few rows. With $n_c$ rows in category $c$, a sum of targets $S_c$, an overall training mean $p$, and a smoothing strength $m$ (a count of imaginary rows that all have the overall mean),

$$
\text{enc}(c) = \frac{S_c + m\,p}{n_c + m}
$$

Read it aloud: add $m$ pretend rows at the overall mean to the category's real rows, then average. When $n_c$ is large the real rows dominate, and when $n_c$ is small the pretend rows do. The same value is the mixture $\frac{n_c}{n_c + m}\,\text{mean}_c + \frac{m}{n_c + m}\,p$.

Work a table by hand. Six training rows have categories A, A, A, B, C, C and targets $1, 1, 0, 0, 1, 0$. The overall mean is $p = 3/6 = 0.5$. Take $m = 2$.

- A has $S = 2$ and $n = 3$, so $\text{enc}(A) = (2 + 1)/(3 + 2) = 0.6$.
- B has $S = 0$ and $n = 1$, so $\text{enc}(B) = (0 + 1)/(1 + 2) = 0.3333$. The plain average would be 0, and smoothing keeps one row from deciding.
- C has $S = 1$ and $n = 2$, so $\text{enc}(C) = (1 + 1)/(2 + 2) = 0.5$.
- An unseen category D gets the overall mean, 0.5.

There is a second trap. If you encode the training rows with this table, each row's own label is inside its own encoding. That is a leak, and it is worst for rare categories. The fix is the out-of-fold encoding. Split the rows into folds, and encode each fold with a table learned from the other folds. A row's label then never contributes to its own value.

Run that on the six rows with two folds, rows $\{0, 3, 4\}$ and $\{1, 2, 5\}$, and $m = 2$. For the first fold, the training rows are 1, 2, and 5, which are A with target 1, A with target 0, and C with target 0. Their overall mean is $1/3$. Then $\text{enc}(A) = (1 + 2/3)/(2 + 2) = 0.4167$, $\text{enc}(C) = (0 + 2/3)/(1 + 2) = 0.2222$, and B is unseen, so it gets $0.3333$. The second fold works the same way with the training rows 0, 3, and 4, and the final out-of-fold encodings of all six rows are

$$
[0.4167,\; 0.7778,\; 0.7778,\; 0.3333,\; 0.2222,\; 0.7778].
$$

Rows 0 and 1 are both A with target 1, yet they get 0.4167 and 0.7778. Neither value knows its own label, so they differ only because their folds differ.

The leak is large enough to measure. Make 300 rows with a category that has 150 possible values, about two rows each, and give them random 0 or 1 targets. Encode with the table from all 300 rows and the encoding correlates 0.65 with the target, which is pure leakage. Encode out of fold and the correlation is $-0.04$. The small negative sign is a side effect of holding a fold out: a fold with many 1s leaves the training mean a little lower, and rare categories mostly echo that mean.

> [!WARNING]
> A target encoding fitted on all rows and then fed into cross-validation scores well on noise. Learn the table inside each training fold, or use out-of-fold values and still fit the final table on training rows only.

## Selection and scaling inside the folds

Go back to Figure 1 and to the cure. The scorer needs to treat the entire preprocessing recipe as one unit that it can refit. Let a pipeline be a function `pipeline_fit(X_train, y_train)` that does everything it needs to do on those rows, then returns a `predict(X_test)` function. Imputing, standardising, choosing columns, and the classifier itself all happen inside that call.

```python
def cv_score(pipeline_fit, X, y, folds):
    scores = np.empty(len(folds))
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(len(y)), test)
        predict = pipeline_fit(X[train], y[train])      # everything is learned here
        scores[f] = np.mean(predict(X[test]) == y[test])
    return scores
```

The shape guards the rule. The held-out rows never enter `pipeline_fit`, so nothing inside it, from a mean to a column choice, can see them. The leaky version differs in one line: it calls `select_top_k(X, y, 10)` once, on all rows, before it hands anything to `cv_score`. The safe version calls `select_top_k(X_train, y_train, 10)` inside `pipeline_fit`.

Hastie, Tibshirani, and Friedman call this the wrong and right way to cross-validate. The wrong way picks predictors on the full data and then cross-validates the classifier alone, which reports a low error on labels with no signal.

> [!KEY]
> Cross-validation scores a recipe, not a model. The recipe is everything that touches the labels or learns a statistic, from imputing through selection to the final fit. Refit the whole recipe on each training fold, and the fold's held-out rows stay hidden.

## Other ways a score lies

Selection is one leak among several. The others are about what the rows contain and where they came from.

**Target leakage.** A feature is computed from the outcome, or from something that happens only after it. A column holding "days in hospital" predicts readmission, but you only know the number after the stay ends. A column "refund issued" predicts churn because refunds follow cancellations. Ask of every column whether you know its value at the moment you would make the prediction.

**Duplicate rows across folds.** If the same record appears twice, a random split can put one copy in training and one in the held-out fold. A model that memorises scores perfectly on the twin. Take 25 random rows, each appearing twice, with labels that are pure noise. A one-nearest-neighbour classifier scores 0.90 under random five-fold splits, because the nearest neighbour of a held-out row is usually its twin. If you keep each pair in one fold, the score drops to 0.42, which is chance, give or take the small sample. Count the held-out rows that have an exact twin in training. A count above zero is a warning.

**Time leakage.** When rows are ordered in time, a random split trains on the future to predict the past. Take 60 days of a rising series. Predicting a held-out day by the mean of the day before and the day after gives a mean squared error of 0.31 on random test days, because the neighbours sit in training. A forecast of the last 12 days from the last training day has a mean squared error of 34.12 on the same series. The two predictors differ, but only the second is one you could run in practice. Split by time, so every test row comes after every training row.

| Leak | What the score looks like | How to catch it |
|---|---|---|
| Selection or scaling before the split | Good on noise, falls on a true hold-out | Refit the pipeline inside each fold |
| Target encoding on all rows | One category column dominates | Encode out of fold |
| Target leakage | One feature is nearly the label | Ask when each value becomes known |
| Duplicates across folds | Perfect memorising scores | Count held-out rows with a twin in training |
| Time leakage | Random-split score beats forward-split score | Split forward in time |

> [!TIP]
> A score that is too good is information. When a model is much better than the problem seems to allow, shuffle the labels and rerun your whole pipeline. If it still scores above chance, there is a leak in the pipeline, and you have a test that finds it.

## Build it step by step

> [!RECAP]
> - A fitted step learns numbers from training rows in `fit` and applies them in `transform`, and `transform` never reads the rows it is given to decide anything.
> - Imputing, standardising, one-hot encoding, and target encoding are all fitted steps with stored numbers.
> - Smoothed target encoding must be out of fold, or a row's own label leaks into its own value.
> - Cross-validation scores a whole recipe, so the recipe refits inside every fold, including selection.
> - Pure-noise labels are the test: a fold-safe pipeline scores about 0.5, and a leaky one does not.

You write the pieces in the order the checks test them. Two helpers are provided: `k_fold_indices` from the previous chapter, and `fit_centroid`, a nearest-centroid classifier.

1. **`Standardiser`** with `fit`, `transform`, and `fit_transform`. `fit` takes `X` of shape `(n, d)` and stores `mean_` and `std_`, each of shape `(d,)`, dividing by $n$. A constant column has a standard deviation of 0, so store 1 there. `transform` returns `(X - mean_) / std_`. The common mistake is to recompute the mean inside `transform`, which passes on training rows and fails when the test rows move.

2. **`Imputer(strategy)`** with `fit` and `transform`. `fit` stores `fill_` of shape `(d,)`, skipping `NaN` with `np.nanmean` or `np.nanmedian`. `transform` returns a copy with every `NaN` replaced by its column's fill. The common mistake is to edit the input array in place.

3. **`one_hot(fit_categories, values)`** returns an array of shape `(n, m + 1)`, where `m` is the number of fitted categories. Build a dictionary from category to column, and send a value that is not in it to the last column. The common mistake is to drop unseen values to all zeros, so the row sums to 0.

4. **`TargetEncoder(smoothing)`** with `fit(categories, y)` and `transform(categories)`. `fit` stores `prior_`, the mean of `y`, and `table_`, a dictionary from category to $(S_c + m p)/(n_c + m)$. `transform` looks each category up and returns `prior_` when it is missing. The common mistake is to divide by $n_c$ and not by $n_c + m$.

5. **`target_encode_oof(categories, y, folds, smoothing)`** returns an array of shape `(n,)`. For each test fold, fit a `TargetEncoder` on the other rows and transform the fold with it. The common mistake is to fit one encoder on all rows, which passes the shape checks and leaks.

   ```python
   train = np.setdiff1d(np.arange(n), test)
   out[test] = TargetEncoder(smoothing).fit(cats[train], y[train]).transform(cats[test])
   ```

6. **`select_top_k(X, y, k)`** returns the `k` column numbers with the largest absolute correlation with `y`, sorted, shape `(k,)`. Centre `X` and `y`, compute each column's correlation, and take `np.argsort(-corr)[:k]`. The common mistake is to drop the absolute value, so a strong negative column loses to a weak positive one.

7. **`cv_score(pipeline_fit, X, y, folds)`** returns the accuracy of each fold, shape `(k,)`. The loop is the one shown above. The common mistake is to build the pipeline once outside the loop and reuse it.

8. **`leaky_cv_accuracy`** and **`safe_cv_accuracy`** `(X, y, n_keep, folds)` each return the mean fold accuracy of a nearest-centroid classifier on the kept columns. The leaky one calls `select_top_k` on all rows once. The safe one calls it inside `pipeline_fit`. The common mistake is to write the safe one with the selection above the function it hands to `cv_score`, which is the leak again.

9. **`count_cross_fold_duplicates(X, folds)`** returns an integer. For each fold, collect the bytes of every training row in a set with `row.tobytes()`, then count the held-out rows that are in it. Sum over folds. The common mistake is to compare held-out rows only with each other.

## Where this breaks

**Pipelines inside a search.** If you also tune a setting, such as the number of kept columns, the tuning loop is one more fitted step. Score it on an outer loop, as in [nested cross-validation](/lessons/014-learning-curves-and-winners-curse).

**Missingness that carries meaning.** A blank income field may mean "declined to say", and that fact can predict the outcome. A filled-in average hides it, so add a 0-or-1 column that marks which entries were missing.

**Groups.** Rows from one patient, one user, or one machine are not independent. Keep each group in one fold, the way the duplicate demo does, or the group id leaks through its rows.


## The exercise

You implement `Standardiser`, `Imputer`, `one_hot`, `TargetEncoder`, `target_encode_oof`, `select_top_k`, `cv_score`, `leaky_cv_accuracy`, `safe_cv_accuracy`, and `count_cross_fold_duplicates`. The starter's closing program runs the hand example through an imputer and a standardiser, then prints the leaky and the fold-safe accuracy on one pure-noise dataset.

The ten checks confirm that `Standardiser` stores the training mean and standard deviation and that shifting the test rows by 1000 moves the output but never the stored numbers. They confirm that `Imputer` fills with the training mean or median, that `one_hot` has an unknown bucket, and that `TargetEncoder` and `target_encode_oof` reproduce the hand tables and keep a row's own label out of its value. One check requires the out-of-fold encoding of noise labels to stay within 0.2 of zero correlation. Another requires `cv_score` to refit on training rows only. The check on pure noise requires `leaky_cv_accuracy` to average above 0.75 and `safe_cv_accuracy` to stay between 0.42 and 0.58 over 10 seeded datasets. The last check counts duplicates across folds.

## Further reading

- Stanford CS229, Machine Learning, course page at `https://cs229.stanford.edu/`. The course description lists "practical advice" for applying learning algorithms. The syllabus is behind a login, so this list cites the course and gives no lecture number.
- Trevor Hastie, Robert Tibshirani, and Jerome Friedman, *The Elements of Statistical Learning*, second edition, section 7.10.2, "The wrong and right way to do cross-validation". The book is available at `https://hastie.su.domains/ElemStatLearn/`.
- Shachar Kaufman, Saharon Rosset, Claudia Perlich, and Ori Stitelman, "Leakage in data mining: formulation, detection, and avoidance", ACM Transactions on Knowledge Discovery from Data, 2012. It names target leakage and describes how to detect it.
