---
title: The learning problem
slug: 01-the-learning-problem
part: "Part 1 — Foundations"
summary: What it means for a model to learn, why training error looks better than reality, and how a held-out set keeps you honest.
minutes: 45
gpu: false
objectives:
  - Compute the empirical risk of a set of predictions for any per-example loss.
  - Build a reproducible train/test split and use it to measure honest error.
  - Fit a polynomial by least squares and explain why training error flatters it.
  - Estimate squared bias, variance, and noise with a simulation, and explain what each one means.
exercise: 01-the-learning-problem
---

# The learning problem

Imagine studying for an exam by memorising last year's paper. You would score perfectly on that paper. On the real exam, with new questions, you might do badly. Machine learning has the same trap. A model that fits the data you gave it is not automatically a model that works on new data.

Every method in this course, from a straight-line fit to a large neural network, tries to answer one question. Given a limited set of examples, how do you pick a model that does well on examples it has never seen? Fitting the examples you have is the easy part. Doing well on new ones is the goal.

In this lesson you learn the words the rest of the course uses for this problem. You measure a model's error, you see why error on the training data is too optimistic, and you split that error into three understandable parts. You also write the code for each idea.

## What you already know

- An average of many random draws gets closer to the true mean as you add more draws.
- The variance of a set of numbers is the average squared distance from their mean.
- A polynomial like $1 - 2x + 0.5x^2$ is a weighted sum of powers of $x$.
- In numpy, `a @ b` is matrix multiplication and `a.mean(axis=0)` averages down the rows.
- Python functions can be passed around as values, such as `lambda p, t: (p - t) ** 2`.

## The supervised setup

In supervised learning you get a set of examples. Each example is an input $x$ and the correct answer $y$, called the target or label. You might have a house's size as $x$ and its price as $y$. You want a function that takes a new $x$ and predicts its $y$.

This course writes $n$ for the number of examples. Each input can have $d$ features (numbers describing it). This lesson uses one feature per example, so each $x_i$ is a single number. The prediction for example $i$ is written $\hat{y}_i$, read "y hat".

There is one important assumption. The examples are **i.i.d.**, short for independent and identically distributed. Identically distributed means every example comes from the same source, like cards dealt from the same deck. Independent means one example tells you nothing about how the next was chosen. When this assumption fails, for example because the world changes after you collect your data, every guarantee in this lesson stops applying. Your code still produces numbers, but they no longer mean what you think.

### Loss: scoring one prediction

A **loss function** scores a single prediction against its target. Small is good. The two most common losses are the following:

- **Squared loss**: $(\hat{y} - y)^2$. It is used for predicting numbers.
- **Zero-one loss**: 1 if the prediction is wrong, 0 if it is right. It is used for predicting categories.

In numpy, each loss works on whole arrays at once:

```python
squared_error = lambda p, t: (p - t) ** 2
zero_one = lambda p, t: (p != t).astype(float)
```

## Risk and empirical risk

The **risk** of a model is its average loss over every example it could ever meet. That is the number you truly care about. You can never compute it, because you never see every possible example.

What you can compute is the average loss over the examples you have. This is the **empirical risk**:

$$
\hat{R} = \frac{1}{n} \sum_{i=1}^{n} \ell(\hat{y}_i, y_i)
$$

Read it aloud piece by piece:

- $\ell(\hat{y}_i, y_i)$ is the loss for example $i$: how wrong prediction $\hat{y}_i$ is compared to target $y_i$.
- $\sum_{i=1}^{n}$ adds up that loss over all $n$ examples.
- $\frac{1}{n}$ divides by $n$, so the whole thing is an average, not a total.

In numpy, it is one line, because the loss already returns one number per example:

```python
risk = np.mean(loss(predictions, targets))
```

### Worked example

Take four predictions and four targets:

| $i$ | $\hat{y}_i$ | $y_i$ | $\hat{y}_i - y_i$ | squared loss |
|---|---|---|---|---|
| 1 | 1.0 | 1.5 | -0.5 | 0.25 |
| 2 | 2.0 | 2.5 | -0.5 | 0.25 |
| 3 | 3.0 | 2.0 | 1.0 | 1.00 |
| 4 | 4.0 | 5.0 | -1.0 | 1.00 |

The sum of the losses is $0.25 + 0.25 + 1 + 1 = 2.5$. Dividing by $n = 4$ gives an empirical risk of $0.625$.

With zero-one loss, predictions `[1, 0, 1]` against targets `[1, 1, 0]` are right once and wrong twice. The empirical risk is $2/3 \approx 0.667$.

If the model was chosen without looking at these examples, this average is a fair estimate of the true risk. On average over many samples it lands exactly on the risk, and it gets closer as $n$ grows. The next section shows what goes wrong when the model was chosen by looking at them.

> [!KEY]
> Empirical risk is the average loss on the examples you have. It stands in for the true risk, which you can never compute.

## Training error is too optimistic

Training a model means searching for the model with the lowest empirical risk on your training data. This idea is called **empirical risk minimisation**. Least squares, logistic regression, and neural network training all work this way.

Here is the catch. The winning model was picked because it scored well on those exact examples. Some of that good score is real skill. Some of it is luck: the model happens to fit the random quirks of this particular sample. On new data those quirks are different, so the luck does not carry over.

The result, stated in words: on average, a trained model's error on its own training data is lower than its true error. The difference is called the **generalisation gap**. It grows as the model becomes more flexible, because a flexible model has more ways to bend around the quirks. It shrinks as you add more training data.

> [!KEY]
> Training error flatters the model. The more flexible the model, the wider the gap between its training error and its true error.

Think back to the exam. Grading your memorised answers on last year's paper tells you nothing about the new paper. You need questions you have not seen.

## Held-out data keeps you honest

The fix is to set some data aside before training. You split your examples into two groups:

- A **training set**, which you use to fit the model.
- A **test set**, also called a held-out set, which you use only to measure error.

The model never saw the test set, so from the test set's point of view the model is fixed. The test error is then a fair, unbiased estimate of the true risk. Unbiased means it is right on average, neither too high nor too low.

> [!KEY]
> Test error is a fair estimate of true error only because the model never saw the test examples during training.

This costs you two things:

- A small test set gives a noisy estimate. It is right on average but can be far off on any one split.
- Every example you put in the test set is one fewer example to train on, so the model gets a little worse.

To split fairly, you shuffle the positions $0, 1, \dots, n-1$ into a random order. The first chunk of that order becomes the test set and the rest becomes the training set. You shuffle positions rather than the arrays themselves, so the same shuffle picks matching rows of `x` and `y`. A fixed random seed makes the split the same every time you run it:

```python
rng = np.random.default_rng(seed)
order = rng.permutation(n)   # a shuffled array of 0..n-1
```

With $n = 20$ and a test fraction of $0.25$, the test set has $\lfloor 0.25 \times 20 \rfloor = 5$ examples. The symbol $\lfloor \cdot \rfloor$ means round down to a whole number. The training set gets the other 15.

## A small model family: polynomials

To see these effects you need models of different flexibility. Polynomials are a good choice. A degree-1 polynomial is a straight line. A degree-9 polynomial can wiggle a lot.

A polynomial of degree $p$ is

$$
\hat{y} = w_0 + w_1 x + w_2 x^2 + \dots + w_p x^p
$$

Read it aloud: take $x$, raise it to each power from 0 to $p$, multiply each power by its weight $w_j$, and add the results. The weights $w_0, \dots, w_p$ are the numbers you learn. The term $x^0$ is always 1, so $w_0$ plays the role of the bias.

The trick is to build a table of powers first. Each row is one example and column $j$ holds $x^j$. This table is the **design matrix**, here called `phi`, with shape `(n, p + 1)`. Then every prediction at once is a matrix product:

```python
phi = x[:, None] ** np.arange(p + 1)[None, :]   # shape (n, p + 1)
y_hat = phi @ w                                  # shape (n,)
```

> [!TIP]
> Print `phi.shape` right after you build it. It should be `(n, p + 1)`. A width of `p` means `np.arange(p)` dropped the highest power.

For example, with $x = [2, -1]$ and $p = 2$:

$$
\phi = \begin{bmatrix} 1 & 2 & 4 \\ 1 & -1 & 1 \end{bmatrix}
$$

Row one is $[2^0, 2^1, 2^2]$ and row two is $[(-1)^0, (-1)^1, (-1)^2]$.

**Least squares** picks the weights $w$ that make the training squared error as small as possible. numpy solves this for you with `np.linalg.lstsq(phi, y, rcond=None)`, which returns the weights as the first item of a tuple. Lesson 2 explains how that solve works inside.

## Bias, variance, and noise

Squared error on a new example can be split into three parts. Each part has a clear meaning, and together they explain why a model that is too simple and a model that is too flexible both do badly.

Picture a darts player. Imagine you could train your model many times, each time on a fresh random training set, and each time ask it to predict at the same input $x$. You would get a cloud of predictions, like darts on a board.

- **Bias** is how far the centre of the cloud is from the bullseye. A too-simple model, like a straight line fit to a curve, misses in the same direction every time.
- **Variance** is how spread out the cloud is. A too-flexible model chases noise, so it changes a lot from one training set to the next.
- **Noise** is randomness in the targets themselves. Even a perfect model cannot predict it.

Suppose the targets come from $y = f(x) + \varepsilon$. Here $f$ is the true function and $\varepsilon$ ("epsilon") is random noise with mean 0 and standard deviation $\sigma$ ("sigma"). Write $\hat{f}(x)$ for the prediction of a model trained on one random training set, and $\bar{f}(x)$ ("f bar") for the average of those predictions over many training sets. Then the expected squared error at $x$ is

$$
\text{expected error} = \big(\bar{f}(x) - f(x)\big)^2 + \text{average of } \big(\hat{f}(x) - \bar{f}(x)\big)^2 + \sigma^2
$$

Read it aloud piece by piece:

- $\big(\bar{f}(x) - f(x)\big)^2$ is the **squared bias**: the gap between the average prediction and the truth, squared.
- The average of $\big(\hat{f}(x) - \bar{f}(x)\big)^2$ is the **variance**: how far each prediction sits from the average prediction, squared and averaged.
- $\sigma^2$ is the **noise**: the variance of the label noise. No model can remove it.

All three parts are zero or positive. Making the model simpler usually raises bias and lowers variance. Making it more flexible does the reverse. Noise stays the same whatever you do.

> [!KEY]
> Expected squared error is squared bias plus variance plus noise. A too-simple model has high bias, a too-flexible model has high variance, and no model removes the noise.

You cannot train on infinitely many training sets, but you can simulate many. This is a **Monte Carlo** estimate: repeat a random experiment many times and average. Store one row of predictions per trial, then:

```python
mean_pred = predictions.mean(axis=0)            # f bar, shape (q,)
bias_sq = (mean_pred - true_function(x_query)) ** 2
variance = predictions.var(axis=0)              # divides by the number of trials
noise = np.full_like(x_query, noise_std ** 2)
```

> [!WARNING]
> `predictions` has one row per trial and one column per query point. Reduce with `axis=0`, down the trials. The result should have shape `(q,)`, one entry per query point.

### Worked example

Say the true value at one point is $f(x) = 1.0$, the noise standard deviation is $\sigma = 0.2$, and four trials give predictions $0.8, 1.2, 1.0, 1.4$.

1. Average prediction: $(0.8 + 1.2 + 1.0 + 1.4) / 4 = 4.4 / 4 = 1.1$.
2. Squared bias: $(1.1 - 1.0)^2 = 0.01$.
3. Distances from the average: $-0.3, 0.1, -0.1, 0.3$. Squared: $0.09, 0.01, 0.01, 0.09$. Their sum is $0.2$, and dividing by 4 gives a variance of $0.05$.
4. Noise: $0.2^2 = 0.04$.
5. Expected error: $0.01 + 0.05 + 0.04 = 0.10$.

The variance here divides by the number of trials, 4, which is what `np.var` does by default.

> [!RECAP]
> - Empirical risk is `np.mean` of the per-example losses, returned as a `float`.
> - Training error is optimistic, so you measure honest error on a held-out test set.
> - A fair split shuffles positions once with a seeded generator and indexes `x` and `y` with the same positions.
> - The design matrix has column $j$ equal to $x^j$, shape `(n, p + 1)`, and least squares finds the weights.
> - Bias-variance: one row of predictions per trial, then squared bias and variance down `axis=0`, plus noise $\sigma^2$.

## Build it step by step

Build the functions in this order. It matches the order the checks test them.

1. **`empirical_risk(predictions, targets, loss)`**. It takes two arrays of shape `(n,)` and a loss function, and returns a single Python `float`. Call `loss` on the two arrays to get one loss per example, then take `np.mean`. The common mistake is returning a numpy scalar or 0-d array instead of a `float`; wrap the result in `float(...)`.

2. **`polynomial_features(x, degree)`**. It takes `x` of shape `(n,)` and returns a matrix of shape `(n, degree + 1)` whose column `j` is `x ** j`. Use `np.arange(degree + 1)` for the powers and broadcasting (`x[:, None]` against a row of powers). The common mistake is `np.arange(degree)`, which drops the highest power.

3. **`fit_polynomial(x, y, degree)`**. It takes `x` and `y` of shape `(n,)` and returns weights of shape `(degree + 1,)`. Build the design matrix with your step 2 function and pass it to `np.linalg.lstsq(phi, y, rcond=None)`. The common mistake is returning the whole tuple from `lstsq`; you need only its first item.

4. **`train_test_split(x, y, test_size, seed)`**. It returns `(x_train, y_train, x_test, y_test)`. Compute the test count with `int(np.floor(test_size * n))`, make `np.random.default_rng(seed).permutation(n)`, take the first chunk as test positions and the rest as training positions, and index both `x` and `y` with them. The common mistake is shuffling `x` and `y` separately, which breaks the pairing between inputs and targets.

5. **`bias_variance_decomposition(true_function, noise_std, x_query, degree, n_train, n_trials, x_domain, seed)`**. It returns three arrays `(bias_sq, variance, noise)`, each of shape `(q,)`, where `q` is the length of `x_query`. The shape of the code is as follows:

   ```python
   rng = np.random.default_rng(seed)
   predictions = np.empty((n_trials, len(x_query)))
   for trial in range(n_trials):
       # draw n_train inputs uniformly from x_domain with rng.uniform
       # make targets: true_function(x_train) + noise_std * rng.normal(...)
       # fit a polynomial, then store its predictions at x_query in this row
       ...
   # reduce down axis 0 for bias_sq and variance; noise is noise_std ** 2
   ```

   The common mistake is averaging over the wrong axis. Each row is one trial, so average down `axis=0` to get one number per query point. Another is using $\sigma$ instead of $\sigma^2$ for the noise term.

## Cost and scaling

In plain words, everything here grows in a straight line with the number of examples, except the least-squares fit, which also grows with the square of the number of weights. The simulation repeats the fit once per trial, so it is the slowest part.

The table uses $n$ for examples, $p$ for the polynomial degree, $T$ for trials, and $q$ for query points.

| Step | Work grows with | Notes |
|---|---|---|
| Empirical risk | $n$ | One loss per example, then an average. |
| Train/test split | $n$ | One shuffle of positions. |
| Polynomial features | $n \times p$ | One entry per example per power. |
| Least-squares fit | $n \times p^2$ | Small for the degrees used here. |
| Bias-variance simulation | $T$ times one fit | The predictions array uses $T \times q$ numbers of memory. |

Doubling $T$ doubles the run time. It makes the estimates only about 1.4 times more precise, because simulation error shrinks with the square root of the number of trials.

## Where this breaks

**The data changes after you train.** Every fairness result here assumes new data comes from the same source as the training data. If customers, sensors, or seasons change, your test error no longer predicts real error. Monitor performance after deployment and retrain on recent data.

**You tune on the test set.** If you try many models and keep the one with the best test score, the test set has become part of the search. Its score is now optimistic, just like training error. Use a separate validation set for choosing models, and touch the test set once at the end.

**The test set is tiny.** A test error from ten examples is fair on average but can be wildly off. With little data, use cross-validation, covered in lesson 7, which averages over several splits.

**A few huge errors dominate.** If the loss sometimes takes enormous values, the average needs many more examples before it settles. Look at the spread of per-example losses, not only their mean.

**High-degree polynomials become unstable.** Powers like $x^{25}$ are tiny or huge, so the fit can swing wildly between training points. That is exactly the high-variance behaviour this lesson describes.

## The exercise

You build the four tools of the learning problem and one experiment. The tools are `empirical_risk`, `train_test_split`, `polynomial_features`, and `fit_polynomial`. The experiment is `bias_variance_decomposition`, which trains many polynomials on fresh random training sets and returns the squared bias, variance, and noise at each query point.

The checks first test each tool on its own. They confirm that `empirical_risk` averages squared and zero-one losses, that `polynomial_features` builds the right table of powers, that `fit_polynomial` recovers an exact polynomial when there is no noise, and that `train_test_split` uses every example exactly once and gives the same split for the same seed. Then they test the lessons of the chapter: training error is lower than held-out error for a flexible model, bias plus variance plus noise matches an independent measurement of the total error, the noise term equals `noise_std ** 2`, and raising the degree lowers bias while raising variance.
