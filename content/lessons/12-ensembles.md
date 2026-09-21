---
title: Bagging, forests, and boosting
slug: 12-ensembles
part: "Part 2 — Classical machine learning"
summary: The variance identity behind bagging, the decorrelation random forests add, and boosting as gradient descent in function space.
minutes: 45
gpu: false
objectives:
  - State the variance of an average of correlated estimators, and see why decorrelation, not more trees, is the lever bagging has left.
  - State the in-bag fraction of the bootstrap and explain out-of-bag error as a held-out estimate that costs nothing extra.
  - State gradient boosting as stagewise functional gradient descent, and see it reduce to fitting residuals under squared error.
  - State AdaBoost's weight update as the exponential-loss instance of the same framework.
prereqs:
  - 11-decision-trees
exercise: 12-ensembles
---

# Bagging, forests, and boosting

A single decision tree has low bias and high variance: it can represent an arbitrarily fine partition of feature space, but a small change to the training sample can flip which split the greedy search takes at the root, rearranging every split beneath it. Two families of methods turn a collection of such unstable estimators into one accurate predictor, by opposite routes. Bagging averages many trees fit independently, trading variance for none of the bias, and works only as well as the trees disagree with each other. Boosting fits trees sequentially, each one correcting the ensemble built so far, trading bias for a controlled amount of variance.

This lesson derives the variance identity that governs bagging, explains why random forests add feature subsampling on top of it, and derives boosting as gradient descent carried out in the space of functions rather than the space of parameters.

## What you already know

- A decision tree grown to purity drives training error to zero and is the high-variance estimator this lesson stabilises.
- $\operatorname{Var}(aX+bY) = a^2\operatorname{Var}(X)+b^2\operatorname{Var}(Y)+2ab\operatorname{Cov}(X,Y)$ for any two random variables.
- The gradient of a scalar loss with respect to a vector of predictions has one entry per prediction.
- $(1-1/n)^n \to e^{-1}$ as $n\to\infty$.

## The bagging variance identity

Let $\hat f_1,\ldots,\hat f_B$ be estimators fit on $B$ independent resamples, each with variance $\sigma^2$ and pairwise correlation $\rho$. Bagging predicts with the average $\bar f=\frac1B\sum_b \hat f_b$. Expanding its variance and regrouping terms by their dependence on $B$ gives

$$
\operatorname{Var}(\bar f) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2 .
$$

The second term is what averaging kills, vanishing as $B\to\infty$. The first term does not depend on $B$ at all: however many trees you average, the ensemble's variance cannot fall below $\rho\sigma^2$. Bagging alone cannot make correlated trees behave like independent ones. The lever left to pull is $\rho$ itself. Lowering the correlation between trees lowers the floor, which is what random forests do.

### The bootstrap and the in-bag fraction

Each $\hat f_b$ is fit on a bootstrap resample: $n$ draws with replacement from the $n$ training points. A point is excluded from one draw with probability $1-1/n$, and from all $n$ draws with probability $(1-1/n)^n \to e^{-1}$ as $n\to\infty$. A bootstrap sample therefore includes a given point with probability converging to $1-e^{-1}\approx0.632$, and leaves it out, out-of-bag, with probability converging to $e^{-1}\approx0.368$. Both fractions are close to their limits by $n$ in the low hundreds.

### Out-of-bag error

Every point is out-of-bag for roughly a third of the trees, by construction. Averaging the predictions of only those trees for a given point gives a held-out prediction, without holding anything out of training. Averaging that error over all $n$ points gives the **out-of-bag (OOB) error**, an estimate of generalisation error that costs nothing beyond the trees you were fitting anyway, and that tracks $k$-fold cross-validation closely once $B$ is large.

### Random forests: decorrelation by feature subsampling

A random forest restricts each split search, at every node of every tree, to a random subset of $m<d$ features rather than all $d$. When two features are both strongly predictive, ordinary bagged trees split on the stronger one near the root of nearly every tree, producing highly correlated trees regardless of which bootstrap sample each saw. Denying most trees access to the stronger feature at a given split forces some to rely on the second-best feature instead, which lowers $\rho$ directly, at the cost of a slightly worse fit for each individual tree, a bias-variance trade paid for in $\rho$ rather than in $\sigma^2$.

## Boosting as functional gradient descent

Boosting builds an additive model $F_M(x)=\sum_{m=1}^M \eta\,h_m(x)$ one term at a time, where each $h_m$ is a weak learner and $\eta$ is a shrinkage factor. Treat the vector of predictions $F=(F(x_1),\ldots,F(x_n))$ as the variable of a generic problem: minimise $L(F)=\sum_i \ell(y_i,F(x_i))$ over all of $\mathbb R^n$, setting aside for a moment that $F$ has to come from a sum of weak learners.

Ordinary gradient descent would move every $F(x_i)$ independently, which is not reachable, because nothing constrains the result to generalise to unseen $x$. Boosting's move is to fit a weak learner $h_m$, typically a shallow tree, to the negative gradient

$$
g_i = -\left.\frac{\partial\ell(y_i,F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}}
$$

by least squares, treating $g$ as a regression target. $h_m$ is the best approximation, within the weak-learner class, to the direction unconstrained gradient descent would take, and $F_m=F_{m-1}+\eta h_m$ steps along it. This is the same algorithm as gradient descent, with projection onto a hypothesis class standing in for the unconstrained step.

### Squared error reduces to fitting residuals

Take $\ell(y,F)=\tfrac12(y-F)^2$. The negative gradient is $g_i=y_i-F_{m-1}(x_i)$, the current residual: gradient boosting under squared error fits every weak learner to the residual left so far, with no further transformation. Because the fitted stump's value on each leaf is the mean of $g$ over that leaf, the residual $g-h$ is orthogonal to $h$: $\sum_i g_i h_i=\sum_i h_i^2$, the same normal-equations orthogonality that made ordinary least squares a projection in lesson 2. That identity pins the ideal step size at exactly $\gamma=1$: the stump fit already found the loss-minimising step, which is why textbook gradient boosting for squared error is described as fitting residuals with no mention of a line search. One round of shrinkage $\eta$ then reduces the loss by

$$
L_m = L_{m-1} - \eta\Big(1-\tfrac{\eta}{2}\Big)\sum_i h_i^2 ,
$$

positive for any $\eta\in(0,2)$, so training loss falls every round unless the fitted stump is identically zero.

### AdaBoost: the exponential-loss instance

Take $y\in\{-1,+1\}$ and $\ell(y,F)=e^{-yF}$, so the current per-point weight is $w_i=e^{-y_iF_{m-1}(x_i)}$ and the weak learner outputs $h(x)\in\{-1,+1\}$. Splitting the weighted loss by whether $h_m$ agrees with $y_i$ and solving for the optimal step size gives

$$
\gamma_m = \frac12\log\frac{1-\epsilon_m}{\epsilon_m}, \qquad \epsilon_m = \frac{W_w}{W_c+W_w} ,
$$

the weighted misclassification rate, where $W_c,W_w$ are the weight mass on correctly and incorrectly classified points. Choosing $h_m$ to minimise the loss at $\gamma_m$ is equivalent to choosing the classifier that minimises $\epsilon_m$. The resulting reweighting rule,

$$
w_i \leftarrow w_i\, e^{-\gamma_m y_i h_m(x_i)} ,
$$

scales a correctly classified point's weight down by $e^{-\gamma_m}<1$ and a misclassified point's weight up by $e^{\gamma_m}>1$, the up-weighting of mistakes AdaBoost is usually introduced through, recovered here as one instance of the functional gradient framework rather than a standalone rule.

### Shrinkage and subsampling

$\eta<1$ trades rounds for stability: the closed-form loss decrease above shows every round makes less progress at a smaller $\eta$, so a given training loss needs proportionally more rounds. Subsampling a fraction of the rows before fitting each round's weak learner, stochastic gradient boosting, adds the same kind of decorrelation between successive learners that feature subsampling adds between the trees of a random forest, at the cost of a noisier gradient estimate.

## Cost and scaling

| Operation | Cost | Notes |
|---|---|---|
| One bootstrap resample | $O(n)$ | $n$ draws from a discrete uniform distribution. |
| Fitting $B$ bagged trees, depth-limited to $O(\log n)$ | $O(Bnd \log n)$ | Independent of each other; embarrassingly parallel. |
| Out-of-bag error, given the $B$ fitted trees | $O(Bn)$ | One prediction per tree per training point. |
| One boosting round, depth-limited weak learner | $O(nd \log n)$ | Same split search as a single tree, on the current residual. |
| $M$ boosting rounds | $O(Mnd \log n)$ | Strictly sequential: round $m$ needs $F_{m-1}$. |

Bagging parallelises across trees; boosting does not, because every round's target depends on the previous round's fit. The sequential dependency is the price of correcting errors rather than merely averaging over them.

## Where this breaks

**Bagging plateaus.** The variance floor $\rho\sigma^2$ means adding trees past the point where $\rho$ stops falling wastes computation without reducing error.

**Boosting overfits when unregularised.** Fitting the residual exactly, with $\eta=1$ and unlimited rounds, drives training loss to zero and can drive generalisation error up well before that, because later rounds fit successively smaller, more sample-specific patterns. Shrinkage, row subsampling, and early stopping against a validation set are the standard controls.

**AdaBoost is sensitive to label noise.** The exponential loss grows without bound as a prediction moves further from the wrong side of the margin, so a single mislabelled point can dominate the weight distribution after a few rounds. Losses that grow linearly, or that cap their growth, are the usual response.

**Neither method fixes a systematically biased weak learner.** Averaging correlated or independent copies of a biased estimator returns something with the same bias. Boosting can lower bias, because each round targets the current residual, but only by increasing variance round over round, which is why it needs shrinkage where bagging does not.

## The exercise

You implement the mechanics behind both families. Write `bootstrap_indices`, which draws one bootstrap resample, and `bagging_variance(rho, sigma2, B)`, the closed-form identity derived above. Write `fit_bagged_stumps`, which fits an ensemble of one-split regression trees to independent bootstrap resamples, and `out_of_bag_error`, which scores that ensemble using only the trees for which each point was held out. Finally write `gradient_boost(X, y, loss_grad, n_rounds, learning_rate)`, stagewise functional gradient descent with depth-limited trees as the weak learner, where `loss_grad(y, F)` returns the negative gradient, the regression target for the next round, given the current predictions `F`.

The checks confirm `bagging_variance` against its two extremes and against a Monte Carlo simulation of correlated Gaussian estimators, that the empirical in-bag fraction of `bootstrap_indices` converges to $1-1/e$, that `out_of_bag_error` tracks a genuinely held-out test error, that the first round of `gradient_boost` under squared error targets the raw labels themselves, that training loss decreases monotonically round over round, and that the loss reduction of a single round matches the closed-form $\eta(1-\eta/2)\sum_i h_i^2$ identity exactly across several learning rates.
