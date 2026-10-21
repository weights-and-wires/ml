---
title: The learning problem
slug: 01-the-learning-problem
part: "Part 1 — Foundations"
summary: Score guesses by hand, hold flats back, fit a curve, and split error into bias, variance, and noise.
minutes: 45
gpu: false
objectives:
  - Score a set of guesses with any per-example loss, then average those scores.
  - Hold some rows back with a seeded split and measure error only on those rows.
  - Fit a polynomial by least squares and explain why the training score flatters it.
  - Estimate squared bias, variance, and noise in a simulation, and say what each part means.
exercise: 01-the-learning-problem
---

# The learning problem

A 20 square metre flat rents for £200 a week. A 30 square metre flat rents for £300. A 40 square metre flat rents for £400.

The rule "weekly rent equals ten times the size" matches all three. It predicts £500 for a 50 square metre flat whose rent is £650, so it fails on a flat it never saw. That failure is the subject of this course.

How do you pick a rule that works on flats you have not seen? By the end you can score a guess by hand and hold flats back so the score is honest. You can fit a simple curve, and explain why a wiggly curve looks perfect on its training flats and fails on the held-back ones.

## What you already know

You already use these five calculations:

- The average of 2, 4, and 6 is 4.
- An error of 2, squared, is 4. Two errors of 1, squared and added, are 2. Big errors hurt more.
- Powers: if $x$ is 3, then $x^0$ is 1, $x^1$ is 3, and $x^2$ is 9.
- A weighted sum: `2*3 + 1*5 = 11`. numpy writes the same idea as `a @ b` when `a` and `b` are arrays.
- A function can be passed as a value, for example `lambda p, t: (p - t) ** 2`.

## Score one guess

You guess 1, 2, 3, and 4. The targets are 1.5, 2.5, 2, and 5.

The first miss is $1-1.5=-0.5$, and $(-0.5)\times(-0.5)=0.25$. The second miss is $2-2.5=-0.5$, and that square is 0.25 too.

The third miss is $3-2=1$, so the square is 1. The fourth miss is $4-5=-1$, so the square is 1.

The four misses and their squares are as follows:

|Guess|Target|Miss|Square|
|---|---|---|---|
|1|1.5|-0.5|0.25|
|2|2.5|-0.5|0.25|
|3|2|1|1|
|4|5|-1|1|

Those squares are called the squared loss. On row three, $\hat{y}=3$ (y-hat) is the guess and $y=2$ is the target.

The formula is as follows:

$$
(\hat{y}-y)^2
$$

Row three gives $(3-2)^2=1$.

Square every row with numpy:

```python
squared_error = lambda p, t: (p - t) ** 2
```

Adding the raw misses $-0.5+-0.5+1+-1$ gives $-1$, because the signs cancel.

Predict categories 1, 0, and 1 against targets 1, 1, and 0. The scores are 0, then 1, then 1. Two of three are wrong, so the average is $2/3$.

That 0-or-1 score is called zero-one loss. Use it for categories, and use squared loss for numbers. With $\ell$ (ell) as the score, the formula is as follows:

$$
\ell=\begin{cases}0&\text{if }\hat{y}=y\\1&\text{if }\hat{y}\neq y\end{cases}
$$

The second pair has $\hat{y}=0$ and $y=1$, so the formula gives 1.

Turn mismatches into ones:

```python
zero_one = lambda p, t: (p != t).astype(float)
```

Subtracting makes $0-1=-1$ and $1-0=1$, so both misses no longer count as 1.

## Average those scores

Add the four squares: $0.25+0.25=0.5$, then $0.5+1=1.5$, then $1.5+1=2.5$. Divide by 4 to get $2.5/4=0.625$.

That average is the empirical risk. Here $n=4$, and $\ell(\hat{y}_i,y_i)$ is the score of guess $i$.

The four scores are 0.25, 0.25, 1, and 1. Their average is $\hat{R}$ (R-hat).

The formula is as follows:

$$
\hat{R}=\frac{1}{n}\sum_{i=1}^{n}\ell(\hat{y}_i,y_i)
$$

The sum is 2.5 and the fraction is $1/4$, so you get 0.625 again.

Take the mean in numpy:

```python
risk = np.mean(loss(predictions, targets))
```

`np.sum` skips the division and stops at 2.5.

The true risk is that average over every flat you might meet. You cannot compute it, because you never see every flat.

If the rule ignored these four flats, 0.625 is a fair stand-in. Over many draws it lands on the true risk, and it gets closer as you add flats.

> [!KEY]
> Empirical risk is the average score on the flats you have. It stands in for the true risk, which you cannot compute.

## Why the training score flatters you

You memorise last year's paper and score 100. On a different paper you score 60, because part of the 100 is luck.

Searching for the lowest average on the flats you fit is empirical risk minimisation. Least squares, logistic regression, and neural network training all search this way.

On average the training score is lower than the true risk. That difference is the generalisation gap. A wiggly rule widens the gap, and more training flats shrink it.

> [!KEY]
> Training error flatters the rule. A more flexible rule widens the gap to the true risk.

This comparison needs one source of flats, and one flat must not determine the next. The short name is i.i.d., for independent and identically distributed.

Picture one deck. Identically distributed means every card comes from that deck. Independent means you replace the card and shuffle, so the three of hearts says nothing about the next one.

If rents change after you collect them, later flats are a different deck. The printed score is no longer the error you meet.

## Hold some flats back

You have $n=20$ flats and hold back a fraction 0.25. Then $0.25\times20=5$, so you score 5 flats and fit on 15.

The 15 flats you fit are the training set. The 5 you only score are the test set, or held-out set. The score is fair, or unbiased: right on average, neither high nor low.

> [!KEY]
> The test score estimates the true risk only because the fit never used the test flats.

Five test flats are noisy on one split. Each flat held back is one fewer to fit, so the rule gets a little worse.

Shuffle positions $0,1,\ldots,19$, not the two arrays apart. The same positions keep each size with its rent, and a fixed seed repeats the shuffle.

Shuffle positions in numpy:

```python
order = np.random.default_rng(seed).permutation(n)
```

Two separate permutations pair a size with the wrong rent.

The test count rounds down: $\lfloor0.25\times20\rfloor=5$. The symbol $\lfloor\cdot\rfloor$ means round down to a whole number.

Of 21 flats, $0.25\times21=5.25$, and the floor is 5. For these 20 flats, 15 train.

## Fit a simple curve

Size $x=2$ with weights 1, 3, and 0.5 uses powers $2^0=1$, $2^1=2$, and $2^2=4$. The guess is $1+3\times2+0.5\times4=1+6+2=9$.

This is a polynomial of degree $p=2$. The weights are $w_0=1$, $w_1=3$, and $w_2=0.5$.

The formula is as follows:

$$
\hat{y}=w_0+w_1x+w_2x^2+\dots+w_px^p
$$

Those weights at $x=2$ give 9 again. Weight $w_0$ multiplies $x^0=1$, the intercept. The darts story uses the word bias for a different idea.

A degree-1 polynomial is a straight line. A degree-9 polynomial can wiggle.

Sizes 2 and $-1$ at degree 2 make two rows. Row one is 1, 2, 4. Row two is $(-1)^0=1$, $(-1)^1=-1$, $(-1)^2=1$.

That table is the design matrix $\phi$ (phi, said "fie"):

$$
\phi=\begin{bmatrix}1&2&4\\1&-1&1\end{bmatrix}
$$

Build one column per power:

```python
phi = x[:, None] ** np.arange(p + 1)[None, :]  # (n, p + 1)
y_hat = phi @ w                                 # (n,)
```

`np.arange(p)` drops $x^2$, so the width is 2 instead of 3.

> [!TIP]
> Print `phi.shape` after you build it. It must be `(n, p + 1)`.

numpy takes $0^0=1$. A 0 in that column deletes the intercept.

Least squares minimises the sum of squared training misses. Lesson 2 derives the solve:

```python
w = np.linalg.lstsq(phi, y, rcond=None)[0]
```

The call returns a tuple. Item 0 is the weights, shape `(p + 1,)`. The whole tuple is the wrong object.

## Why simple curves and wiggly curves fail differently

Four darts land at 0.8, 1.2, 1.0, and 1.4. The bullseye is 1.0.

Add the hits: $0.8+1.2=2$, then $2+1=3$, then $3+1.4=4.4$. Then $4.4/4=1.1$. The gap $1.1-1=0.1$ squares to $0.01$.

Distances from 1.1 are $-0.3$, $0.1$, $-0.1$, and $0.3$. Squares are 0.09, 0.01, 0.01, and 0.09. Their sum is $0.09+0.01+0.01+0.09=0.20$, and $0.20/4=0.05$.

`np.var` divides by 4, the number of darts, not by 3. The wobble has standard deviation $\sigma$ (sigma, said "sig-ma") $=0.2$, and $0.2^2=0.04$. Then $0.01+0.05+0.04=0.10$.

The 0.01 is squared bias, the 0.05 is variance, and the 0.04 is noise. A line misses the same way every time: high bias. A wiggly curve scatters: high variance.

Noise stays, and no curve removes it. Stiffer raises bias and lowers variance. Wigglier does the reverse.

All three pieces are zero or positive.

Here $f(x)=1.0$ is the bullseye, $\bar{f}(x)=1.1$ is the average dart (f bar), and $\hat{f}(x)$ is one dart. Targets add a nudge of mean 0 and standard deviation $\sigma$.

The formula is as follows:

$$
\text{expected\,error}=\big(\bar{f}(x)-f(x)\big)^2+\text{average\,of\,}\big(\hat{f}(x)-\bar{f}(x)\big)^2+\sigma^2
$$

$(1.1-1)^2=0.01$, the spread is 0.05, and $\sigma^2=0.04$. The sum is 0.10 again.

Repeating the draw and averaging is a Monte Carlo estimate. Store one row per trial and one column per query point, and average down the trials with `axis=0`:

```python
mean_pred = predictions.mean(axis=0)  # (q,)
bias_sq = (mean_pred - true_function(x_query)) ** 2
variance = predictions.var(axis=0)    # divide by the trial count
noise = np.full_like(x_query, noise_std ** 2)
```

`axis=1` returns one number per trial.

> [!WARNING]
> Use `axis=0`: one row per trial, one column per query point. The result must have shape `(q,)`.

> [!KEY]
> Expected squared error is squared bias plus variance plus noise. Stiff means high bias, wiggly means high variance, and noise stays.

Remember:

> [!RECAP]
> - Return empirical risk as `float(np.mean(scores))`.
> - Score a held-out set. The training score is too kind.
> - Shuffle positions once, then index `x` and `y` together.
> - Column $j$ of the design matrix is $x^j$, shape `(n, p + 1)`.
> - Reduce bias and variance on `axis=0`. Noise is $\sigma^2$.

## Build it step by step

The checks call the functions in this order.

1. **`empirical_risk(predictions, targets, loss)`** takes two `(n,)` arrays and a loss, and returns one Python `float`:

   ```python
   scores = loss(predictions, targets)  # (n,)
   # Return float(mean(scores)).
   ```

   A numpy scalar fails the check.

2. **`polynomial_features(x, degree)`** returns shape `(n, degree + 1)`, and column `j` is `x ** j`:

   ```python
   powers = np.arange(degree + 1)
   # x[:, None] ** powers[None, :]
   ```

   `np.arange(degree)` drops the last column.

3. **`fit_polynomial(x, y, degree)`** returns weights of shape `(degree + 1,)` from `(n,)` arrays:

   ```python
   phi = polynomial_features(x, degree)
   # np.linalg.lstsq(phi, y, rcond=None)[0]
   ```

   The whole tuple is not the weight vector.

4. **`train_test_split(x, y, test_size, seed)`** returns `(x_train, y_train, x_test, y_test)`. Take `int(np.floor(test_size * n))` test rows:

   ```python
   order = np.random.default_rng(seed).permutation(len(x))
   # Prefix tests. The rest trains. Index x and y alike.
   ```

   Separate shuffles of `x` and `y` break pairs.

5. **`bias_variance_decomposition(true_function, noise_std, x_query, degree, n_train, n_trials, x_domain, seed)`** returns three shape `(q,)` arrays:

   ```python
   predictions = np.empty((n_trials, len(x_query)))
   # Each row: uniform draw, noise, fit, predict at x_query.
   # Reduce on axis 0. Noise is noise_std ** 2.
   ```

   `axis=1` collapses query points. Skipping the square stores 0.2, not 0.04.

## Cost and scaling

Scoring and splitting grow like the number of flats. The fit also grows with the square of the number of weights, and the simulation repeats it once per trial.

Here $n$ counts flats, $p$ is the degree, $T$ counts trials, and $q$ counts query points. Growth is as follows:

|Step|Work grows with|Notes|
|---|---|---|
|Empirical risk|$n$|One mean.|
|Train/test split|$n$|One shuffle.|
|Polynomial features|$n\times p$|One column per power.|
|Least-squares fit|$n\times p^2$|Usual degrees.|
|Bias-variance simulation|$T$ fits|Memory $T\times q$.|

Doubling $T$ doubles the run time. Precision improves only about 1.4 times, because simulation error shrinks with the square root of the trial count.

## Where this breaks

**Collected rents go stale.** Later flats are another deck, so the test score misses the error you meet. Fit again on recent rents.

**You pick the best test score.** That score turns optimistic. Choose rules on a validation set, and touch the test set once.

**The test set is tiny.** Ten flats are fair on average and wild on one split. Use cross-validation from lesson 7.

**A few huge misses dominate.** The average settles slowly. Look at the spread of per-flat scores as well as the mean.

**A high power blows up.** $x^{25}$ is tiny or huge, so the curve swings between training flats. That swing is high variance.

## The exercise

You build `empirical_risk`, `polynomial_features`, `fit_polynomial`, `train_test_split`, and `bias_variance_decomposition`.

`empirical_risk` must average squared loss and zero-one loss. `polynomial_features` must build $1, x, x^2, \ldots$ up to the degree. `fit_polynomial` must recover an exact polynomial with no noise.

`train_test_split` must use every row once, keep pairs, and repeat for one seed. A wiggly fit must score lower on training rows than on held-out rows.

Bias, variance, and noise must sum to a separate Monte Carlo total. Noise must equal `noise_std ** 2`. A higher degree must lower bias and raise variance.
