---
title: "Likelihood, MAP, and the Bayesian view"
slug: 011-likelihood-map-and-bayesian-regression
part: "Part 1 — Foundations"
summary: "Squared loss and log loss are likelihoods. Ridge and lasso are MAP. The predictive variance grows away from the data."
minutes: 75
gpu: false
objectives:
  - Derive squared loss from Gaussian noise and log loss from Bernoulli noise by maximum likelihood.
  - Show that ridge is the MAP estimate under a Gaussian prior and lasso is the MAP estimate under a Laplace prior, with lambda equal to the noise variance over the prior variance.
  - Compute the posterior mean and covariance of Bayesian linear regression, and check that the mean is the ridge solution.
  - Explain why the predictive variance grows as an input moves away from the training sizes, and compute that variance.
prereqs:
  - 006-ridge-regression
  - 007-lasso-and-fista
  - 008-sigmoid-and-log-loss
  - 010-exponential-families-and-glms
exercise: 011-likelihood-map-and-bayesian-regression
---

# Likelihood, MAP, and the Bayesian view

You have two houses, and you have already subtracted the price of a studio so the numbers stay small. A studio, size 0, sold at price 0. A house of size 2 sold at price 2. A buyer asks what a house of size 6 should cost, and how sure you are.

The straight line through the two sales is price = size, so it answers 6 and misses the sales by zero. A miss of zero is not a reason to trust that answer at size 6. Two sales cannot tell you whether a slope of 1 is the market or a slope the noise happened to draw. Ridge pulls the weights down by charging a fee, and it still returns one number. The buyer asked for a spread.

This chapter puts a probability on the noise and a probability on the weights. Squared loss and log loss fall out of the noise. The ridge fee and the lasso fee fall out of the weights. The spread is what remains once you keep every weight the data still allows, instead of only the single best one.

## What you already know

- Half the sum of squared misses scores a line, and the best weights solve $X^\top X\, w = X^\top y$.
- Ridge adds $\tfrac{\lambda}{2}\lVert w\rVert^2$ and solves $(X^\top X + \lambda I)\, w = X^\top y$. See [ridge regression](/lessons/006-ridge-regression).
- The lasso adds $\lambda\lVert w\rVert_1$. When $X$ is the identity, the answer is soft thresholding. See [the lasso](/lessons/007-lasso-and-fista).
- A click model pays $\operatorname{softplus}(z) - yz$, the log loss. See [from a score to a probability](/lessons/008-sigmoid-and-log-loss).
- A bell curve is fixed by a mean and a variance, and independent houses multiply their probabilities.

## Two houses, and the band you are building

Figure 1 is the picture this chapter builds. The dashed line is least squares, price = size. The orange line is the shrunk posterior mean, and the band is one predictive standard deviation.

```figure
<svg viewBox="0 0 640 262" role="img" aria-label="Left: posterior mean and a one-standard-deviation band for two houses, wider at size 6 than between the houses. Right: predictive variance, about 1.45 at size 0 and 9.09 at size 6.">
<text x="48" y="16" class="fig-bold">Price and one standard deviation</text>
<line x1="48" y1="210" x2="300" y2="210" class="stroke-muted"/>
<line x1="48" y1="210" x2="48" y2="32" class="stroke-muted"/>
<line x1="48" y1="182.1" x2="300" y2="182.1" class="stroke-grid"/><text x="42" y="186.1" text-anchor="end" class="fig-small">0</text><line x1="48" y1="147.2" x2="300" y2="147.2" class="stroke-grid"/><text x="42" y="151.2" text-anchor="end" class="fig-small">2</text><line x1="48" y1="112.3" x2="300" y2="112.3" class="stroke-grid"/><text x="42" y="116.3" text-anchor="end" class="fig-small">4</text><line x1="48" y1="77.4" x2="300" y2="77.4" class="stroke-grid"/><text x="42" y="81.4" text-anchor="end" class="fig-small">6</text>
<line x1="62.8" y1="210.0" x2="62.8" y2="214.0" class="stroke-muted"/><text x="62.8" y="226.0" text-anchor="middle" class="fig-small">0</text><line x1="136.9" y1="210.0" x2="136.9" y2="214.0" class="stroke-muted"/><text x="136.9" y="226.0" text-anchor="middle" class="fig-small">2</text><line x1="285.2" y1="210.0" x2="285.2" y2="214.0" class="stroke-muted"/><text x="285.2" y="226.0" text-anchor="middle" class="fig-small">6</text>
<text x="174" y="246" text-anchor="middle" class="fig-small">size</text>
<polygon points="48.0,161.6 51.1,160.9 54.3,160.1 57.5,159.3 60.6,158.5 63.8,157.6 66.9,156.7 70.0,155.8 73.2,154.9 76.3,153.9 79.5,152.9 82.7,151.9 85.8,150.9 88.9,149.8 92.1,148.7 95.2,147.6 98.4,146.4 101.5,145.2 104.7,144.0 107.8,142.7 111.0,141.4 114.2,140.1 117.3,138.8 120.4,137.5 123.6,136.1 126.7,134.7 129.9,133.2 133.1,131.8 136.2,130.3 139.3,128.9 142.5,127.4 145.7,125.8 148.8,124.3 151.9,122.7 155.1,121.2 158.2,119.6 161.4,118.0 164.6,116.4 167.7,114.8 170.8,113.2 174.0,111.5 177.1,109.9 180.3,108.2 183.4,106.6 186.6,104.9 189.8,103.2 192.9,101.5 196.0,99.8 199.2,98.1 202.3,96.4 205.5,94.7 208.6,93.0 211.8,91.3 214.9,89.6 218.1,87.8 221.2,86.1 224.4,84.3 227.6,82.6 230.7,80.8 233.9,79.1 237.0,77.3 240.1,75.6 243.3,73.8 246.4,72.0 249.6,70.3 252.8,68.5 255.9,66.7 259.0,65.0 262.2,63.2 265.4,61.4 268.5,59.6 271.6,57.8 274.8,56.0 277.9,54.3 281.1,52.5 284.2,50.7 287.4,48.9 290.6,47.1 293.7,45.3 296.9,43.5 300.0,41.7 300.0,153.7 296.9,154.0 293.7,154.4 290.6,154.8 287.4,155.1 284.2,155.5 281.1,155.8 277.9,156.2 274.8,156.6 271.6,156.9 268.5,157.3 265.4,157.7 262.2,158.1 259.0,158.4 255.9,158.8 252.8,159.2 249.6,159.6 246.4,160.0 243.3,160.4 240.1,160.8 237.0,161.2 233.9,161.6 230.7,162.0 227.6,162.4 224.4,162.8 221.2,163.2 218.1,163.6 214.9,164.1 211.8,164.5 208.6,164.9 205.5,165.4 202.3,165.8 199.2,166.3 196.0,166.7 192.9,167.2 189.8,167.7 186.6,168.1 183.4,168.6 180.3,169.1 177.1,169.6 174.0,170.1 170.8,170.6 167.7,171.2 164.6,171.7 161.4,172.3 158.2,172.8 155.1,173.4 151.9,174.0 148.8,174.6 145.7,175.2 142.5,175.9 139.3,176.5 136.2,177.2 133.1,177.9 129.9,178.6 126.7,179.4 123.6,180.1 120.4,180.9 117.3,181.7 114.2,182.5 111.0,183.4 107.8,184.2 104.7,185.2 101.5,186.1 98.4,187.0 95.2,188.0 92.1,189.1 88.9,190.1 85.8,191.2 82.7,192.3 79.5,193.4 76.3,194.6 73.2,195.8 70.0,197.0 66.9,198.3 63.8,199.6 60.6,200.9 57.5,202.2 54.3,203.6 51.1,205.0 48.0,206.4" class="fill-soft-b"/>
<polyline points="48.0,189.1 51.1,187.6 54.3,186.1 57.5,184.6 60.6,183.1 63.8,181.6 66.9,180.2 70.0,178.7 73.2,177.2 76.3,175.7 79.5,174.2 82.7,172.7 85.8,171.3 88.9,169.8 92.1,168.3 95.2,166.8 98.4,165.3 101.5,163.8 104.7,162.4 107.8,160.9 111.0,159.4 114.2,157.9 117.3,156.4 120.4,154.9 123.6,153.5 126.7,152.0 129.9,150.5 133.1,149.0 136.2,147.5 139.3,146.0 142.5,144.6 145.7,143.1 148.8,141.6 151.9,140.1 155.1,138.6 158.2,137.1 161.4,135.7 164.6,134.2 167.7,132.7 170.8,131.2 174.0,129.7 177.1,128.2 180.3,126.8 183.4,125.3 186.6,123.8 189.8,122.3 192.9,120.8 196.0,119.3 199.2,117.9 202.3,116.4 205.5,114.9 208.6,113.4 211.8,111.9 214.9,110.4 218.1,109.0 221.2,107.5 224.4,106.0 227.6,104.5 230.7,103.0 233.9,101.5 237.0,100.1 240.1,98.6 243.3,97.1 246.4,95.6 249.6,94.1 252.8,92.6 255.9,91.2 259.0,89.7 262.2,88.2 265.4,86.7 268.5,85.2 271.6,83.7 274.8,82.3 277.9,80.8 281.1,79.3 284.2,77.8 287.4,76.3 290.6,74.8 293.7,73.4 296.9,71.9 300.0,70.4" fill="none" class="stroke-muted dashed"/>
<polyline points="48.0,184.0 51.1,182.9 54.3,181.8 57.5,180.7 60.6,179.7 63.8,178.6 66.9,177.5 70.0,176.4 73.2,175.4 76.3,174.3 79.5,173.2 82.7,172.1 85.8,171.0 88.9,170.0 92.1,168.9 95.2,167.8 98.4,166.7 101.5,165.6 104.7,164.6 107.8,163.5 111.0,162.4 114.2,161.3 117.3,160.2 120.4,159.2 123.6,158.1 126.7,157.0 129.9,155.9 133.1,154.9 136.2,153.8 139.3,152.7 142.5,151.6 145.7,150.5 148.8,149.5 151.9,148.4 155.1,147.3 158.2,146.2 161.4,145.1 164.6,144.1 167.7,143.0 170.8,141.9 174.0,140.8 177.1,139.8 180.3,138.7 183.4,137.6 186.6,136.5 189.8,135.4 192.9,134.4 196.0,133.3 199.2,132.2 202.3,131.1 205.5,130.0 208.6,129.0 211.8,127.9 214.9,126.8 218.1,125.7 221.2,124.6 224.4,123.6 227.6,122.5 230.7,121.4 233.9,120.3 237.0,119.3 240.1,118.2 243.3,117.1 246.4,116.0 249.6,114.9 252.8,113.9 255.9,112.8 259.0,111.7 262.2,110.6 265.4,109.5 268.5,108.5 271.6,107.4 274.8,106.3 277.9,105.2 281.1,104.2 284.2,103.1 287.4,102.0 290.6,100.9 293.7,99.8 296.9,98.8 300.0,97.7" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="62.8" cy="182.1" r="5" class="fill-a"/>
<circle cx="136.9" cy="147.2" r="5" class="fill-a"/>
<text x="352" y="16" class="fig-bold">Predictive variance</text>
<line x1="352" y1="210" x2="608" y2="210" class="stroke-muted"/>
<line x1="352" y1="210" x2="352" y2="32" class="stroke-muted"/>
<line x1="352" y1="210.0" x2="608" y2="210.0" class="stroke-grid"/><text x="346" y="214.0" text-anchor="end" class="fig-small">0</text><line x1="352" y1="150.7" x2="608" y2="150.7" class="stroke-grid"/><text x="346" y="154.7" text-anchor="end" class="fig-small">4</text><line x1="352" y1="91.3" x2="608" y2="91.3" class="stroke-grid"/><text x="346" y="95.3" text-anchor="end" class="fig-small">8</text><line x1="352" y1="32.0" x2="608" y2="32.0" class="stroke-grid"/><text x="346" y="36.0" text-anchor="end" class="fig-small">12</text>
<line x1="367.1" y1="210.0" x2="367.1" y2="214.0" class="stroke-muted"/><text x="367.1" y="226.0" text-anchor="middle" class="fig-small">0</text><line x1="442.4" y1="210.0" x2="442.4" y2="214.0" class="stroke-muted"/><text x="442.4" y="226.0" text-anchor="middle" class="fig-small">2</text><line x1="592.9" y1="210.0" x2="592.9" y2="214.0" class="stroke-muted"/><text x="592.9" y="226.0" text-anchor="middle" class="fig-small">6</text>
<text x="480" y="246" text-anchor="middle" class="fig-small">size</text>
<polyline points="352.0,185.6 355.2,186.3 358.4,187.0 361.6,187.6 364.8,188.1 368.0,188.6 371.2,189.0 374.4,189.3 377.6,189.6 380.8,189.9 384.0,190.0 387.2,190.2 390.4,190.2 393.6,190.2 396.8,190.2 400.0,190.0 403.2,189.9 406.4,189.6 409.6,189.4 412.8,189.0 416.0,188.6 419.2,188.1 422.4,187.6 425.6,187.0 428.8,186.4 432.0,185.7 435.2,184.9 438.4,184.1 441.6,183.2 444.8,182.3 448.0,181.3 451.2,180.3 454.4,179.2 457.6,178.0 460.8,176.8 464.0,175.5 467.2,174.1 470.4,172.7 473.6,171.3 476.8,169.8 480.0,168.2 483.2,166.6 486.4,164.9 489.6,163.1 492.8,161.3 496.0,159.4 499.2,157.5 502.4,155.5 505.6,153.5 508.8,151.4 512.0,149.2 515.2,147.0 518.4,144.7 521.6,142.4 524.8,140.0 528.0,137.6 531.2,135.0 534.4,132.5 537.6,129.8 540.8,127.2 544.0,124.4 547.2,121.6 550.4,118.7 553.6,115.8 556.8,112.8 560.0,109.8 563.2,106.7 566.4,103.6 569.6,100.4 572.8,97.1 576.0,93.8 579.2,90.4 582.4,86.9 585.6,83.4 588.8,79.8 592.0,76.2 595.2,72.5 598.4,68.8 601.6,65.0 604.8,61.2 608.0,57.2" fill="none" class="stroke-c" stroke-width="2.5"/>
<circle cx="367.1" cy="188.4" r="4" class="fill-c"/>
<circle cx="392.2" cy="190.2" r="4" class="fill-c"/>
<circle cx="442.4" cy="183.0" r="4" class="fill-c"/>
<circle cx="592.9" cy="75.2" r="4" class="fill-c"/>
<text x="514.9" y="79.2" text-anchor="end" class="fig-small">9.09</text>
<text x="400.2" y="170" class="fig-small">1.33</text>
</svg>
Figure 1. The dashed line fits the two sales exactly. The orange line is the posterior mean, and the band is one predictive standard deviation. The variance is smallest between the houses, at size 2/3, and much larger at size 6.
```

## Squared loss is Gaussian noise

Suppose each sale is the line's prediction plus noise. Write one house as a row $x$, so the studio is $x = (1, 0)$ and the size-2 house is $x = (1, 2)$. The 1 is an intercept column. The prediction is $x^\top w$. The noise is a bell curve of mean 0 and variance $\sigma^2$ (sigma squared), the same variance for every house, and each house is an independent draw. Independent and identically distributed, *i.i.d.* for short, means the houses do not share their noise and they all use the same rule.

The probability of one label is then a bell curve centred on the prediction:

$$
p(y \mid x, w) = (2\pi\sigma^2)^{-1/2} \exp\left(-\frac{(y - x^\top w)^2}{2\sigma^2}\right)
$$

Read it as a bell of height set by $\sigma^2$, centred at $x^\top w$. The *likelihood* is this probability read as a function of the weights, with the labels held fixed. Independent houses multiply, and the log of a product is a sum, so the log likelihood of the list is

$$
\ell(w) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\lVert Xw - y\rVert^2
$$

The first term does not depend on $w$. The second term is squared error, divided by $2\sigma^2$. Maximising $\ell(w)$ is the same as minimising $\lVert Xw - y\rVert^2$, because $1/(2\sigma^2)$ is a positive constant once you treat the noise variance as known. That is why least squares is the *maximum likelihood* estimate for Gaussian noise. The negative log likelihood, which you minimise, is

$$
\operatorname{NLL}(w) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\lVert Xw - y\rVert^2
$$

On the two houses the least-squares weights are $w = (0, 1)$ and the misses are 0. With $\sigma^2 = 1$ the negative log likelihood is $\log(2\pi) \approx 1.838$.

The ridge weights below are $w = (2/11,\ 8/11)$. The predictions are $2/11$ and $18/11$, so the misses are $2/11$ and $-4/11$, and the squared length of the misses is $20/121$. The negative log likelihood is $\log(2\pi) + 10/121 \approx 1.921$, worse than the perfect fit. Likelihood on its own still wants the line that misses by zero. Compute that number as follows:

```python
resid = X @ w - y
nll = 0.5 * n * np.log(2 * np.pi * sigma2) + 0.5 * np.dot(resid, resid) / sigma2
```

> [!KEY]
> With Gaussian noise of known variance, the weights that maximise the likelihood are the least-squares weights. Squared loss is that likelihood, up to a positive scale and an added constant.

## A coin gives you the log loss

A click is not a bell. It is a coin: the label $y$ is 1 or 0, and a score $z = x^\top w$ sets the probability $p$ of heads through the sigmoid, as in the log-loss chapter. The probability of the label you actually saw is $p$ when $y = 1$ and $1 - p$ when $y = 0$. One expression covers both:

$$
p(y \mid x, w) = p^{\,y}(1 - p)^{1 - y}
$$

The log is $y\log p + (1 - y)\log(1 - p)$. Maximising it is the same as minimising the negative, and that negative is the log loss. The softplus form you already code is the same quantity, as the next two cases show.

Take $y = 1$. The negative log is $-\log p$. With $p = 1/(1 + e^{-z})$ that equals $\log(1 + e^{-z})$. And $\operatorname{softplus}(z) - z = \log(1 + e^{z}) - z = \log(1 + e^{-z})$. The two sides agree, so the negative log likelihood is $\operatorname{softplus}(z) - yz$ when $y = 1$.

Take $y = 0$. The negative log is $-\log(1 - p)$. Here $1 - p = 1/(1 + e^{z})$, so $-\log(1 - p) = \log(1 + e^{z}) = \operatorname{softplus}(z)$, which is $\operatorname{softplus}(z) - yz$ at $y = 0$.

At a score of 0 the sigmoid is $1/2$, and a click has negative log likelihood $\log 2$. The softplus form gives $\operatorname{softplus}(0) - 0 = \log 2$. Same number. Log loss is maximum likelihood for a coin, in the same way squared loss is maximum likelihood for a bell.

> [!KEY]
> Bernoulli noise makes the negative log likelihood equal to $\operatorname{softplus}(z) - yz$. Log loss is that likelihood.

## A Gaussian prior is ridge

Likelihood answers "which weights make the sales likely?" It does not answer "which weights were likely before you saw a sale?" A *prior* is that second probability. Take each weight as an independent bell of mean 0 and variance $\tau^2$ (tau squared). Wide $\tau^2$ means you are willing to see a large weight. Narrow $\tau^2$ means you expect weights near 0.

The log of that prior, for $d$ weights, is $-\tfrac{d}{2}\log(2\pi\tau^2) - \lVert w\rVert^2/(2\tau^2)$. The *posterior* is the probability of the weights after the sales. Bayes' rule says it is proportional to likelihood times prior. The marginal probability of the sales does not depend on $w$, so the weights that maximise the posterior are the weights that maximise likelihood times prior. That point is the *maximum a posteriori* estimate, MAP for short.

Minimising the negative log of likelihood times prior gives

$$
J(w) = \operatorname{NLL}(w) + \frac{d}{2}\log(2\pi\tau^2) + \frac{1}{2\tau^2}\lVert w\rVert^2
$$

The two log terms do not move the minimum. What remains as a function of $w$ is squared error over $2\sigma^2$, plus the squared weights over $2\tau^2$. Multiply by the positive number $\sigma^2$ and the minimum stays put:

$$
\frac{1}{2}\lVert Xw - y\rVert^2 + \frac{\sigma^2}{2\tau^2}\lVert w\rVert^2
$$

Set that next to the ridge score from the ridge chapter, $\tfrac12\lVert Xw - y\rVert^2 + \tfrac{\lambda}{2}\lVert w\rVert^2$. The two scores have the same minimum when

$$
\lambda = \frac{\sigma^2}{\tau^2}
$$

Lambda is the noise variance divided by the prior variance. Noisy sales, or a prior that insists on small weights, both raise the fee. Clean sales, or a prior that allows large weights, both lower it.

The gradient of $J$ makes the same point. Differentiate the negative log likelihood and the prior term:

$$
\nabla J(w) = \frac{1}{\sigma^2} X^\top(Xw - y) + \frac{1}{\tau^2} w
$$

Set the gradient to 0 and multiply through by $\sigma^2$. You get $(X^\top X + \lambda I)\, w = X^\top y$ with $\lambda = \sigma^2/\tau^2$, which is the ridge solve. The data pulls with $X^\top(Xw - y)$. The prior pulls with $w/\tau^2$, back towards 0.

On the two houses, $X^\top X = \begin{bmatrix} 2 & 2 \\ 2 & 4 \end{bmatrix}$ and $X^\top y = (2,\ 4)$. With $\sigma^2 = \tau^2 = 1$ you have $\lambda = 1$, so the system is

$$
\begin{bmatrix} 3 & 2 \\ 2 & 5 \end{bmatrix} w = \begin{bmatrix} 2 \\ 4 \end{bmatrix}
$$

The solution is $w = (2/11,\ 8/11) \approx (0.182,\ 0.727)$. Least squares was $(0,\ 1)$. The intercept has moved off 0, and the slope has moved from 1 down to $8/11$, because the prior pays for every unit of slope.

Both log terms contribute $2\log(2\pi) \approx 3.676$. With $\sigma^2 = \tau^2 = 1$, the score at three weights is as follows:

| Weights | Role | Added to $2\log(2\pi)$ | $J(w)$ |
|---|---|---|---|
| $(0,\ 1)$ | Least squares, misses of 0 | $1/2$ | 4.176 |
| $(2/11,\ 8/11)$ | MAP, the ridge solution | $44/121$ | 4.039 |
| $(0,\ 0)$ | Predict 0 everywhere | $2$ | 5.676 |

The perfect fit is no longer the best. Its likelihood term is better by $10/121$, and its prior term is worse by $3/22$, so the posterior prefers the ridge weights. The score and its gradient are as follows:

```python
prior = 0.5 * d * np.log(2 * np.pi * tau2) + 0.5 * np.dot(w, w) / tau2
score = nll + prior
grad = (X.T @ resid) / sigma2 + w / tau2
```

> [!KEY]
> Ridge is MAP with a Gaussian prior of variance $\tau^2$. The fee is $\lambda = \sigma^2/\tau^2$, noise variance over prior variance, and the ridge solve is where the gradient of the negative log posterior is 0.

> [!TIP]
> You still choose $\lambda$ by a held-out score when you do not know $\sigma^2$ and $\tau^2$. This chapter says what the number you chose means.

## A Laplace prior is the lasso

A Gaussian prior pulls a small weight only a little, because the pull is the weight itself. The lasso wants a pull that stays constant all the way to 0, so a weak weight lands on exactly 0. The prior that does that is a Laplace bell on each weight, with width $b > 0$:

$$
p(w_j) = \frac{1}{2b}\exp\left(-\frac{|w_j|}{b}\right)
$$

Independent weights add their negative logs, and the piece that depends on $w$ is $\lVert w\rVert_1 / b$. The MAP objective, dropping terms that do not depend on $w$, is squared error over $2\sigma^2$ plus $\lVert w\rVert_1 / b$. Multiply by $\sigma^2$:

$$
\frac{1}{2}\lVert Xw - y\rVert^2 + \frac{\sigma^2}{b}\lVert w\rVert_1
$$

The lasso score in the lasso chapter is $\tfrac12\lVert Xw - y\rVert^2 + \lambda\lVert w\rVert_1$, with no one-half on the fee. The two agree when $\lambda = \sigma^2/b$.

One weight checks the match. Take $X = 1$, $y = 3$, $\sigma^2 = 1$, and $b = 1$, so $\lambda = 1$. The MAP score is $\tfrac12(w - 3)^2 + |w|$. For $w > 0$ the slope is $(w - 3) + 1$, which is 0 at $w = 2$. Soft thresholding $S_1(3)$ from the lasso chapter is also 2.

A Laplace prior gives you the lasso peak and no closed Gaussian cloud around it. When you need the spread, keep the Gaussian prior, and accept that it does not write exact zeros.

## Keep the whole posterior

MAP returns the peak. The buyer asked how sure you are, which is a question about the cloud, not the peak. With Gaussian noise and a Gaussian prior the posterior is itself a bell, and both its centre and its spread have a formula.

The centre is the ridge solution you already have, $\mu = (X^\top X + \lambda I)^{-1} X^\top y$ with $\lambda = \sigma^2/\tau^2$. The covariance, the spread of the cloud, is

$$
\Sigma = \sigma^2\left(X^\top X + \frac{\sigma^2}{\tau^2} I\right)^{-1}
$$

Read $\Sigma$ as $\sigma^2$ times the inverse of the ridge matrix. A larger $\lambda$, or more rows, makes $\Sigma$ smaller. More rows also pull $\mu$ towards least squares, because the likelihood grows and the prior stays put.

On the two houses with $\sigma^2 = \tau^2 = 1$,

$$
\Sigma = \frac{1}{11}\begin{bmatrix} 5 & -2 \\ -2 & 3 \end{bmatrix}
$$

The trace, the sum of the diagonal, is $8/11 \approx 0.727$. See the same two houses a second time, four rows instead of two. Then $X^\top X$ doubles, $\lambda$ is still 1, and

$$
\Sigma = \begin{bmatrix} 5 & 4 \\ 4 & 9 \end{bmatrix}^{-1}
$$

with trace $14/29 \approx 0.483$. The mean moves from $(2/11,\ 8/11)$ to $(4/29,\ 24/29) \approx (0.138,\ 0.828)$, closer to least squares at $(0,\ 1)$. Figure 2 draws one standard deviation of each cloud against the prior circle.

```figure
<svg viewBox="0 0 640 268" role="img" aria-label="One-standard-deviation ellipses in weight space. A wide prior circle sits on the origin. Two houses shrink it to a smaller ellipse, and repeating those houses shrinks it again towards least squares at intercept 0 and slope 1.">
<text x="58" y="18" class="fig-bold">Weights, one standard deviation</text>
<line x1="58" y1="228" x2="292" y2="228" class="stroke-muted"/>
<line x1="58" y1="228" x2="58" y2="36" class="stroke-muted"/>
<line x1="58" y1="199.2" x2="292" y2="199.2" class="stroke-grid"/><text x="52" y="203.2" text-anchor="end" class="fig-small">-1</text><line x1="58" y1="135.2" x2="292" y2="135.2" class="stroke-grid"/><text x="52" y="139.2" text-anchor="end" class="fig-small">0</text><line x1="58" y1="71.2" x2="292" y2="71.2" class="stroke-grid"/><text x="52" y="75.2" text-anchor="end" class="fig-small">1</text>
<line x1="93.1" y1="228.0" x2="93.1" y2="232.0" class="stroke-muted"/><text x="93.1" y="244.0" text-anchor="middle" class="fig-small">-1</text><line x1="171.1" y1="228.0" x2="171.1" y2="232.0" class="stroke-muted"/><text x="171.1" y="244.0" text-anchor="middle" class="fig-small">0</text><line x1="249.1" y1="228.0" x2="249.1" y2="232.0" class="stroke-muted"/><text x="249.1" y="244.0" text-anchor="middle" class="fig-small">1</text>
<text x="175" y="260" text-anchor="middle" class="fig-small">intercept w0</text>
<text x="8" y="132" class="fig-small">w1</text>
<polyline points="249.1,135.2 248.9,130.7 248.3,126.2 247.4,121.7 246.0,117.4 244.3,113.1 242.2,108.9 239.8,104.8 237.0,100.9 233.9,97.2 230.5,93.7 226.7,90.3 222.7,87.2 218.5,84.4 214.0,81.8 209.3,79.4 204.4,77.3 199.4,75.5 194.2,74.1 188.8,72.9 183.4,72.0 178.0,71.4 172.5,71.2 167.0,71.3 161.5,71.7 156.1,72.4 150.7,73.4 145.4,74.8 140.3,76.4 135.3,78.3 130.5,80.5 125.9,83.0 121.6,85.8 117.4,88.8 113.6,92.0 110.0,95.4 106.7,99.1 103.8,102.9 101.2,106.9 98.9,111.0 97.0,115.2 95.5,119.5 94.3,124.0 93.5,128.4 93.1,132.9 93.1,137.5 93.5,142.0 94.3,146.4 95.5,150.9 97.0,155.2 98.9,159.4 101.2,163.5 103.8,167.5 106.7,171.3 110.0,175.0 113.6,178.4 117.4,181.6 121.6,184.6 125.9,187.4 130.5,189.9 135.3,192.1 140.3,194.0 145.4,195.6 150.7,197.0 156.1,198.0 161.5,198.7 167.0,199.1 172.5,199.2 178.0,199.0 183.4,198.4 188.8,197.5 194.2,196.3 199.4,194.9 204.4,193.1 209.3,191.0 214.0,188.6 218.5,186.0 222.7,183.2 226.7,180.1 230.5,176.7 233.9,173.2 237.0,169.5 239.8,165.6 242.2,161.5 244.3,157.3 246.0,153.0 247.4,148.7 248.3,144.2 248.9,139.7 249.1,135.2" fill="none" class="stroke-muted dashed"/>
<polygon points="168.9,110.5 165.4,108.6 162.0,106.7 158.7,104.6 155.6,102.5 152.6,100.4 149.8,98.1 147.1,95.8 144.7,93.5 142.4,91.2 140.4,88.8 138.6,86.5 137.0,84.1 135.6,81.8 134.5,79.5 133.7,77.2 133.1,75.1 132.8,72.9 132.7,70.9 132.9,68.9 133.4,67.1 134.1,65.3 135.0,63.7 136.3,62.2 137.7,60.8 139.4,59.6 141.4,58.5 143.5,57.6 145.9,56.8 148.4,56.1 151.2,55.7 154.1,55.4 157.1,55.2 160.3,55.3 163.7,55.5 167.1,55.8 170.6,56.4 174.2,57.0 177.9,57.9 181.6,58.9 185.3,60.0 189.0,61.3 192.7,62.7 196.3,64.3 199.9,66.0 203.5,67.8 206.9,69.7 210.2,71.6 213.4,73.7 216.5,75.9 219.4,78.1 222.1,80.3 224.7,82.6 227.0,85.0 229.2,87.3 231.1,89.7 232.8,92.0 234.3,94.4 235.5,96.7 236.5,98.9 237.2,101.2 237.7,103.3 237.9,105.4 237.8,107.4 237.5,109.3 236.9,111.1 236.0,112.8 234.9,114.4 233.6,115.8 232.0,117.1 230.2,118.3 228.2,119.3 225.9,120.2 223.4,120.9 220.8,121.4 218.0,121.8 215.0,122.0 211.8,122.1 208.6,122.0 205.2,121.7 201.7,121.2 198.1,120.6 194.5,119.9 190.8,118.9 187.1,117.9 183.4,116.7 179.7,115.3 176.1,113.8 172.4,112.2 168.9,110.5" class="fill-soft-b stroke-b"/>
<polygon points="169.8,98.3 166.8,96.8 164.0,95.2 161.2,93.5 158.6,91.8 156.1,90.0 153.7,88.2 151.4,86.3 149.3,84.5 147.3,82.6 145.6,80.7 144.0,78.9 142.6,77.0 141.4,75.2 140.3,73.4 139.5,71.7 139.0,70.0 138.6,68.3 138.4,66.8 138.5,65.3 138.7,63.9 139.2,62.6 139.9,61.4 140.8,60.3 141.9,59.3 143.2,58.4 144.7,57.6 146.4,57.0 148.3,56.4 150.3,56.1 152.5,55.8 154.9,55.7 157.3,55.7 159.9,55.8 162.6,56.1 165.4,56.5 168.3,57.0 171.2,57.7 174.2,58.4 177.3,59.3 180.3,60.4 183.4,61.5 186.5,62.7 189.5,64.0 192.5,65.4 195.4,66.9 198.3,68.5 201.1,70.1 203.8,71.8 206.4,73.6 208.9,75.4 211.2,77.2 213.4,79.1 215.4,80.9 217.3,82.8 219.0,84.7 220.5,86.5 221.8,88.4 222.9,90.2 223.8,91.9 224.5,93.7 225.0,95.3 225.3,96.9 225.3,98.5 225.1,99.9 224.8,101.3 224.2,102.5 223.4,103.7 222.4,104.7 221.2,105.7 219.7,106.5 218.1,107.2 216.4,107.8 214.4,108.2 212.3,108.6 210.1,108.7 207.6,108.8 205.1,108.7 202.5,108.5 199.7,108.2 196.9,107.7 194.0,107.1 191.0,106.4 188.0,105.6 184.9,104.6 181.9,103.6 178.8,102.4 175.7,101.1 172.7,99.8 169.8,98.3" class="fill-soft-c stroke-c"/>
<circle cx="171.1" cy="135.2" r="3.5" class="fill-ink"/>
<circle cx="171.1" cy="71.2" r="5" class="fill-a"/>
<circle cx="185.3" cy="88.7" r="4" class="fill-b"/>
<rect x="330" y="48" width="12" height="12" class="fill-soft-b stroke-b"/>
<text x="350" y="59" class="fig-small">two houses</text>
<rect x="330" y="74" width="12" height="12" class="fill-soft-c stroke-c"/>
<text x="350" y="85" class="fig-small">same houses again</text>
<line x1="330" y1="108" x2="342" y2="108" class="stroke-muted dashed"/>
<text x="350" y="112" class="fig-small">prior, standard deviation 1</text>
<circle cx="336" cy="132" r="4" class="fill-a"/>
<text x="350" y="136" class="fig-small">least squares, (0, 1)</text>
</svg>
Figure 2. The prior is the dashed circle. Two houses shrink it to the orange ellipse, centred at the ridge weights. Seeing those houses again shrinks it to the green ellipse and slides the centre towards least squares. In this example the least-squares weights have length 1, so they sit on the prior circle.
```

Solve for the mean and the covariance together, against the same Gram matrix:

```python
lam = sigma2 / tau2
gram = X.T @ X + lam * np.eye(d)
mean = np.linalg.solve(gram, X.T @ y)
cov = sigma2 * np.linalg.solve(gram, np.eye(d))
```

> [!WARNING]
> Swapping $\sigma^2$ and $\tau^2$ puts $\tau^2/\sigma^2$ on the diagonal and multiplies the inverse by $\tau^2$. Both the fee and the spread come out wrong, and the two mistakes agree with each other, so a gradient of 0 does not catch them.

> [!KEY]
> The posterior is Gaussian. Its mean is the ridge solution at $\lambda = \sigma^2/\tau^2$. Its covariance is $\sigma^2$ times the inverse of $X^\top X + \lambda I$, and that covariance shrinks as rows arrive.

## The band widens away from the houses

A house of size $t$ that you have not fitted has row $x = (1,\ t)$. The weights are still a cloud, so $x^\top w$ is a cloud too. Add that sale's own noise, a bell of variance $\sigma^2$ independent of the weights. The predictive distribution is a bell with mean $x^\top \mu$ and variance

$$
v(x) = \sigma^2 + x^\top \Sigma x
$$

The first term is noise you would have even if you knew the weights. The second term is how much the line itself still wobbles at this $x$. A small error in the slope, multiplied by a large size, moves the price by a lot. That is why the band opens as you leave the houses.

It is not smallest on a training point. For a straight line the two sales pin the line together, so the height is most certain between them. Differentiate $x^\top \Sigma x$ in the size $t$ and the minimum on these houses is at $t = 2/3$, where $v = 4/3$. At the studio, $x = (1,\ 0)$, you get $x^\top \Sigma x = 5/11$ and $v = 1 + 5/11 = 16/11$. At size 2, $v = 20/11$. At size 6, $x^\top \Sigma x = 89/11$ and $v = 100/11$. The right-hand plot of Figure 1 is this curve. Forgetting $\sigma^2$ leaves $5/11$ at the studio instead of $16/11$, a spread that claims the noise went away.

The mean at size 6 is $50/11 \approx 4.55$, against the least-squares answer of 6, and the standard deviation is $\sqrt{100/11} \approx 3.02$. The point answer 6 sits inside the band. Compute the two numbers as follows:

```python
pred_mean = float(x @ mean)
pred_var = float(sigma2 + x @ cov @ x)
```

> [!KEY]
> The predictive variance is the noise variance plus $x^\top \Sigma x$. The second term grows as $x$ moves away from the sizes you fitted, so a far house is a wide bell even when the fit looks perfect on the sales you have.

## Cost and scaling

The posterior costs one ridge solve. Form the $d \times d$ Gram matrix, then solve against $X^\top y$ and against the identity, about $d^3$ arithmetic, once. Each further house is a dot product and a quadratic form, about $d^2$, and the covariance stores $d^2$ numbers.

## Where this breaks

- **The noise is not a bell, or the mean is not a line.** This covariance formula stops applying. A Laplace prior still gives the lasso as MAP, and it does not give you $\Sigma$.
- **$\sigma^2$ or $\tau^2$ is far off.** Then $\lambda = \sigma^2/\tau^2$ is the wrong fee and the band has the wrong width. A held-out score still picks $\lambda$, and the formula says which belief that $\lambda$ encodes.
- **The prior is centred somewhere other than 0.** The pull is towards that centre. The ridge solve in the earlier chapter is the centre-at-0 case.
- **A price cannot be negative.** Far from the sales the mean can leave the range you care about, and the predictive bell still assigns probability there.

> [!RECAP]
> - Gaussian noise makes maximum likelihood equal to least squares. Bernoulli noise makes it equal to log loss.
> - A Gaussian prior makes MAP equal to ridge with $\lambda = \sigma^2/\tau^2$. A Laplace prior makes MAP equal to the lasso with $\lambda = \sigma^2/b$.
> - The posterior mean is that ridge solution. The covariance is $\sigma^2(X^\top X + \lambda I)^{-1}$, and it shrinks as rows arrive.
> - The predictive variance is $\sigma^2 + x^\top \Sigma x$. On the two houses it is $16/11$ at size 0 and $100/11$ at size 6.
> - Swapping $\sigma^2$ with $\tau^2$, or dropping $\sigma^2$ from the predictive variance, both give a confident wrong spread.

## Build it step by step

Use the two houses as your scratch example: rows $(1, 0)$ and $(1, 2)$, labels $0$ and $2$, with $\sigma^2 = \tau^2 = 1$. Build six functions, in this order.

1. `neg_log_likelihood_gaussian(X, y, w, sigma2)` returns one float, $\tfrac{n}{2}\log(2\pi\sigma^2) + \tfrac{1}{2\sigma^2}\lVert Xw - y\rVert^2$. At $w = (0, 1)$ that is $\log(2\pi)$. The common mistake is to drop $\sigma^2$ under the squared misses, or to average over rows.

2. `map_objective(X, y, w, sigma2, tau2)` returns one float. Add $\tfrac{d}{2}\log(2\pi\tau^2) + \tfrac{1}{2\tau^2}\lVert w\rVert^2$ to the negative log likelihood. At the MAP weights the value is $2\log(2\pi) + 44/121$. The common mistake is to return the negative log likelihood alone.

3. `map_gradient(X, y, w, sigma2, tau2)` returns shape `(d,)`, equal to $X^\top(Xw - y)/\sigma^2 + w/\tau^2$. A central finite difference of `map_objective` has to match it. The common mistake is the ridge gradient $X^\top(Xw - y) + \lambda w$, which is this gradient times $\sigma^2$.

4. `ridge_weights(X, y, lam)` returns shape `(d,)` from one linear solve. Pass $\lambda = \sigma^2/\tau^2$. On the two houses with $\lambda = 1$ the weights are $(2/11,\ 8/11)$. The common mistake is an inverse where a solve belongs. A finite difference of `map_objective` is 0 at this answer.

5. `posterior_linreg(X, y, sigma2, tau2)` returns the mean, shape `(d,)`, and the covariance, shape `(d, d)`. The mean equals `ridge_weights` at $\lambda = \sigma^2/\tau^2$. The covariance is $\sigma^2$ times the inverse of $X^\top X + \lambda I$, from a solve against the identity. The common mistake is to swap $\sigma^2$ and $\tau^2$.

6. `predictive(x_new, mean, cov, sigma2)` returns two floats, $x^\top \mu$ and $\sigma^2 + x^\top \Sigma x$. The studio row $(1, 0)$ has variance $16/11$, and $(1, 6)$ has variance $100/11$. The common mistake is to return $x^\top \Sigma x$ and drop the noise.

## The exercise

You implement `neg_log_likelihood_gaussian`, `map_objective`, `map_gradient`, `ridge_weights`, `posterior_linreg`, and `predictive`. The checks walk from the negative log likelihood on the two houses, through the MAP score and a finite-difference gradient, to ridge weights where that gradient is 0, a posterior that shrinks when the houses repeat, and a predictive variance that is larger at size 6 than at the studio.

## Further reading

The lectures below are the ones this chapter follows:

- For the probabilistic reading of linear regression and of logistic regression, see [CS229 lecture 3, The concept of underfitting and overfitting](https://see.stanford.edu/Course/CS229/42).
- For the Bernoulli and Gaussian examples inside an exponential family, see [CS229 lecture 4, Newton's method](https://see.stanford.edu/Course/CS229/49).
- For the step from a prior to a MAP estimate and on to a penalty, see [CS229 lecture 11, Bayesian statistics and regularization](https://see.stanford.edu/Course/CS229/36).
- For a Bayesian reading of what training data rules out, see the 6.S898 class [Bayesian analysis of learning and generalisation](https://phillipi.github.io/6.s898/) (Jeremy Bernstein, 28 September 2023).
- For maximum likelihood as a training objective, see [CS236 lecture 4, Maximum likelihood learning](https://deepgenerativemodels.github.io/assets/slides/cs236_lecture4.pdf) (Stefano Ermon).
