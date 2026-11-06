---
title: One loss and one gradient for every family
slug: 020-one-loss-one-gradient
part: "Part 1 — Foundations"
summary: Let a linear score drive the dial, and one loss and one gradient, prediction minus truth, fit temperatures, coins, and counts.
minutes: 40
gpu: false
objectives:
  - Build a model by setting the dial equal to a linear score and predicting the mean A'(dial).
  - Derive the loss A(eta) - y eta as the negative log of the probability of the label, and recover squared error and the log loss from it.
  - Derive the gradient X^T(mu - y) in two lines and check it against finite differences.
  - Fit a Poisson model by gradient descent and read the result as predictions minus counts.
prereqs:
  - 016-log-loss
  - 017-gradient-descent-for-logistic
  - 019-one-dial-three-distributions
exercise: 020-one-loss-one-gradient
---

# One loss and one gradient for every family

Squared error fits temperatures, and the log loss fits clicks. The two losses look unrelated, and so do their gradients, so it seems each kind of label forces you to invent a loss and then do the calculus again. A third kind of label makes the problem concrete. A shop counts how many people arrive in each of three hours: 1, 1, and 2, at hours 0, 1, and 2. You want a rule that predicts the count for any hour.

A straight line is the first idea, and it fails in a way you can see at once. The last chapter fitted a line to the falling counts 3, 1, and 0 and found it predicts −1.67 arrivals at hour 3. A count can never be negative. You need a rule whose prediction stays positive however the weights come out.

This chapter shows that one loss and one gradient serve every family from the previous chapter. The key move is small. Let a linear score drive the dial instead of the prediction, then score the model by how surprised it is by the true labels.

## What you already know

- A family has a dial $\eta$, a log-partition $A(\eta)$, a mean $A'(\eta)$, and a variance $A''(\eta)$, from the chapter "One dial, three distributions".
- A linear score for a row of features $x$ with weights $w$ is $x \cdot w$.
- The log loss for a coin is $\operatorname{softplus}(z) - yz$, and its gradient is $X^\top(p - y)$.
- A gradient is the list of slopes, one per weight, and walking against it lowers the loss.
- A variance is never negative, so $A$ bends upward.

## A line drives the dial

Keep the line, but move it. Instead of predicting the count with $w_0 + w_1 x$, use that score as the dial and let the family turn it into a mean:

$$
\eta = Xw, \qquad \mu = A'(\eta)
$$

Why the dial and not the mean? The dial can be any real number, so every choice of weights is valid, and $A'$ then maps it into the range the mean must stay in. No weights can produce a negative count. Here $X$ is the table of features with one row per example, $w$ holds the weights, and $\mu$ is the vector of predicted means. A model built this way is a *generalised linear model*, or GLM: linear in the dial, then bent by $A'$. For the Poisson, $\mu = e^{\eta}$, which is positive for every $\eta$. For a coin, $A'$ is the sigmoid, and you recover logistic regression.

The shop has the rows $(1, 0)$, $(1, 1)$, and $(1, 2)$, where the first entry multiplies the intercept $w_0$ and the second is the hour. At weights $w = (0, 0)$ every dial reading is 0, so every predicted mean is $e^{0} = 1$. Figure 1 shows the counts and the flat curve that results.

```figure
<svg viewBox="0 0 600 246" role="img" aria-label="Three hourly counts, the flat curve at zero weights, and the fitted curve with its residuals"><line x1="60.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-muted" />
<line x1="60.0" y1="20.0" x2="60.0" y2="200.0" class="stroke-muted" />
<text x="143.3" y="218.0" text-anchor="middle" class="fig-small">hour 0</text>
<text x="143.3" y="236.0" text-anchor="middle" class="fig-small">μ − y = −0.14</text>
<text x="310.0" y="218.0" text-anchor="middle" class="fig-small">hour 1</text>
<text x="310.0" y="236.0" text-anchor="middle" class="fig-small">μ − y = +0.27</text>
<text x="476.7" y="218.0" text-anchor="middle" class="fig-small">hour 2</text>
<text x="476.7" y="236.0" text-anchor="middle" class="fig-small">μ − y = −0.14</text>
<text x="52.0" y="204.0" text-anchor="end" class="fig-small">0</text>
<text x="52.0" y="144.0" text-anchor="end" class="fig-small">1</text>
<text x="52.0" y="84.0" text-anchor="end" class="fig-small">2</text>
<text x="52.0" y="24.0" text-anchor="end" class="fig-small">3</text>
<text x="60.0" y="12.0" text-anchor="start" class="fig-small">arrivals</text>
<line x1="60.0" y1="140.0" x2="560.0" y2="140.0" class="stroke-muted dashed" />
<polyline points="60.0,157.2 66.3,156.5 72.7,155.9 79.0,155.3 85.3,154.6 91.6,153.9 98.0,153.3 104.3,152.6 110.6,151.9 117.0,151.2 123.3,150.4 129.6,149.7 135.9,149.0 142.3,148.2 148.6,147.5 154.9,146.7 161.3,145.9 167.6,145.1 173.9,144.3 180.3,143.5 186.6,142.7 192.9,141.8 199.2,141.0 205.6,140.1 211.9,139.2 218.2,138.3 224.6,137.4 230.9,136.5 237.2,135.6 243.5,134.6 249.9,133.7 256.2,132.7 262.5,131.7 268.9,130.7 275.2,129.7 281.5,128.6 287.8,127.6 294.2,126.5 300.5,125.4 306.8,124.3 313.2,123.2 319.5,122.1 325.8,121.0 332.2,119.8 338.5,118.6 344.8,117.4 351.1,116.2 357.5,115.0 363.8,113.7 370.1,112.5 376.5,111.2 382.8,109.9 389.1,108.6 395.4,107.2 401.8,105.8 408.1,104.5 414.4,103.1 420.8,101.6 427.1,100.2 433.4,98.7 439.7,97.2 446.1,95.7 452.4,94.2 458.7,92.6 465.1,91.1 471.4,89.5 477.7,87.8 484.1,86.2 490.4,84.5 496.7,82.8 503.0,81.1 509.4,79.3 515.7,77.6 522.0,75.8 528.4,73.9 534.7,72.1 541.0,70.2 547.3,68.3 553.7,66.4 560.0,64.4" fill="none" class="stroke-b" stroke-width="2.5" />
<line x1="143.3" y1="140.0" x2="143.3" y2="148.1" class="stroke-bad" stroke-width="3"/>
<line x1="310.0" y1="140.0" x2="310.0" y2="123.8" class="stroke-bad" stroke-width="3"/>
<line x1="476.7" y1="80.0" x2="476.7" y2="88.1" class="stroke-bad" stroke-width="3"/>
<circle cx="143.3" cy="140.0" r="6" class="fill-a"/>
<circle cx="310.0" cy="140.0" r="6" class="fill-a"/>
<circle cx="476.7" cy="80.0" r="6" class="fill-a"/>
<text x="360.0" y="158.0" text-anchor="start" class="fig-small">flat prediction at w = (0, 0)</text>
<text x="551.7" y="44.0" text-anchor="end" class="fig-small fig-bold">fitted curve e^(w0 + w1 x)</text></svg>
Figure 1. The three counts, the flat prediction at zero weights, and the curve at the best weights found later in the chapter. The red segments are the residuals μ − y.
```

## Score the model by how surprised it is

A model that gives the real label high probability is a good one. So score each row by the probability the model assigned to what happened, and take the negative log so that bigger means worse. From the shared formula in the previous chapter, the probability of label $y$ at dial $\eta$ is $h(y)\exp(\eta y - A(\eta))$, so its negative log is

$$
-\log p(y \mid \eta) = A(\eta) - y\,\eta - \log h(y)
$$

Rows are independent, so the total surprise is the sum over rows. The last term, $\log h(y)$, depends only on the label, not on the weights. Changing the weights cannot change it, so it cannot change which weights are best. Drop it, and the loss is

$$
L(w) = \sum_{i=1}^{n}\bigl[A(\eta_i) - y_i\,\eta_i\bigr], \qquad \eta = Xw
$$

Read it aloud: for every row, add the log-partition of its dial reading and subtract the label times that reading. Each family turns it into a loss you know, or one that is easy to read.

| Family | $A(\eta) - y\eta$ | Which loss is it |
|---|---|---|
| Gaussian | $\tfrac12\eta^2 - y\eta = \tfrac12(\eta - y)^2 - \tfrac12 y^2$ | Half the squared error, plus a constant |
| Bernoulli | $\operatorname{softplus}(\eta) - y\eta$ | The log loss from the earlier chapter |
| Poisson | $e^{\eta} - y\eta$ | New: punishes a dial that is too high exponentially, and one that is too low only linearly |

One loss covers all three, and squared error is its Gaussian case.

At $w = (0, 0)$ the shop rows each score $A(0) - y \cdot 0 = e^{0} = 1$, so the loss is $1 + 1 + 1 = 3$.

> [!WARNING]
> This loss drops $\log h(y)$, so it is not the true negative log-likelihood. For the Poisson, $\log h(y) = -\log y!$, so the true negative log-likelihood is this loss plus $\sum_i \log y_i!$. Summing `-poisson.logpmf` from `scipy.stats` at the shop counts and zero weights gives $3 + \log 2 = 3.693$. The shift does not move the best weights.

## The gradient is prediction minus truth

Differentiate one row. The weights enter only through $\eta_i = x_i \cdot w$, so $\partial \eta_i / \partial w = x_i$. The first piece gives $\partial A(\eta_i)/\partial w = A'(\eta_i)\, x_i = \mu_i\, x_i$, because the slope of $A$ is the mean. The second piece gives $\partial (y_i \eta_i)/\partial w = y_i\, x_i$. Subtract:

$$
\nabla L(w) = \sum_i (\mu_i - y_i)\, x_i = X^{\top}(\mu - y)
$$

Read it aloud: each weight is pushed by the prediction minus the truth, multiplied by how large its feature was. This is the same shape as the gradient you used for coins, and it holds for every family because the slope of $A$ is the mean. In numpy:

```python
mu = mean_from_natural(family, X @ w)
grad = X.T @ (mu - y)   # shape (d,)
```

Check it on the shop. The residuals $\mu - y$ at zero weights are $1 - 1 = 0$, $1 - 1 = 0$, and $1 - 2 = -1$. The first weight collects their sum, $-1$. The second weights each by the hour, so it gets $2 \times (-1) = -2$. The gradient is $[-1, -2]$.

```figure
<svg viewBox="0 0 600 246" role="img" aria-label="Loss of the three-row Poisson example as the second weight changes, with the tangent at zero"><line x1="70.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-muted" />
<line x1="70.0" y1="20.0" x2="70.0" y2="200.0" class="stroke-muted" />
<text x="100.6" y="218.0" text-anchor="middle" class="fig-small">−0.5</text>
<text x="253.7" y="218.0" text-anchor="middle" class="fig-small">0</text>
<text x="406.9" y="218.0" text-anchor="middle" class="fig-small">0.5</text>
<text x="560.0" y="218.0" text-anchor="middle" class="fig-small">1</text>
<text x="62.0" y="181.5" text-anchor="end" class="fig-small">3</text>
<text x="62.0" y="136.5" text-anchor="end" class="fig-small">4</text>
<text x="62.0" y="91.5" text-anchor="end" class="fig-small">5</text>
<text x="62.0" y="46.5" text-anchor="end" class="fig-small">6</text>
<text x="315.0" y="238.0" text-anchor="middle" class="fig-small">w1, with w0 held at 0</text>
<polyline points="70.0,94.2 74.1,96.6 78.2,98.9 82.4,101.2 86.5,103.4 90.6,105.7 94.7,108.0 98.8,110.2 102.9,112.4 107.1,114.6 111.2,116.8 115.3,118.9 119.4,121.1 123.5,123.2 127.6,125.3 131.8,127.4 135.9,129.4 140.0,131.5 144.1,133.5 148.2,135.5 152.4,137.5 156.5,139.4 160.6,141.4 164.7,143.3 168.8,145.2 172.9,147.0 177.1,148.9 181.2,150.7 185.3,152.4 189.4,154.2 193.5,155.9 197.6,157.6 201.8,159.3 205.9,160.9 210.0,162.5 214.1,164.1 218.2,165.6 222.4,167.2 226.5,168.6 230.6,170.1 234.7,171.5 238.8,172.9 242.9,174.2 247.1,175.5 251.2,176.7 255.3,178.0 259.4,179.1 263.5,180.3 267.6,181.3 271.8,182.4 275.9,183.4 280.0,184.3 284.1,185.2 288.2,186.1 292.4,186.9 296.5,187.7 300.6,188.4 304.7,189.0 308.8,189.6 312.9,190.2 317.1,190.6 321.2,191.1 325.3,191.4 329.4,191.7 333.5,192.0 337.6,192.1 341.8,192.2 345.9,192.3 350.0,192.2 354.1,192.1 358.2,191.9 362.4,191.7 366.5,191.3 370.6,190.9 374.7,190.4 378.8,189.8 382.9,189.2 387.1,188.4 391.2,187.6 395.3,186.6 399.4,185.6 403.5,184.5 407.6,183.2 411.8,181.9 415.9,180.5 420.0,178.9 424.1,177.3 428.2,175.5 432.4,173.6 436.5,171.6 440.6,169.5 444.7,167.2 448.8,164.9 452.9,162.4 457.1,159.7 461.2,156.9 465.3,154.0 469.4,150.9 473.5,147.7 477.6,144.3 481.8,140.8 485.9,137.1 490.0,133.2 494.1,129.2 498.2,125.0 502.4,120.6 506.5,116.1 510.6,111.3 514.7,106.4 518.8,101.2 522.9,95.9 527.1,90.3 531.2,84.5 535.3,78.5 539.4,72.3 543.5,65.9 547.6,59.2 551.8,52.3 555.9,45.1 560.0,37.7" fill="none" class="stroke-b" stroke-width="2.5" />
<polyline points="131.2,141.5 315.0,195.5" fill="none" class="stroke-a dashed" stroke-width="2.5" />
<circle cx="253.7" cy="177.5" r="6" class="fill-a"/>
<text x="284.4" y="47.0" text-anchor="start" class="fig-small fig-bold">blue: loss 3, slope −2 at w1 = 0</text>
<circle cx="345.9" cy="192.3" r="5" class="fill-c"/>
<text x="284.4" y="65.0" text-anchor="start" class="fig-small">green: bottom of this slice, w1 = 0.30</text></svg>
Figure 2. Hold the intercept at 0 and move only the second weight. The loss is 3 at zero, and the dashed tangent has slope −2, the second gradient entry. A negative slope says to raise this weight.
```

> [!WARNING]
> Writing $y - \mu$ flips the gradient to $[1, 2]$, and every step then walks uphill.

## Why every fit has one bottom

The loss for a single row is $A(\eta_i)$ minus a term that is a straight line in the weights. A straight line does not bend. $A$ bends upward, because its bend is a variance and a variance is never negative. Adding up bowls gives a bowl. So the loss has no false dips to get stuck in.

Two conditions keep the bottom unique. The columns of $X$ must carry independent information, because otherwise two weight settings give the same predictions. For coins, the labels must not split perfectly, because then the loss keeps falling as the weights grow. Figure 3 shows the shape of $A(\eta) - y\eta$ at $y = 1$ for each family.

```figure
<svg viewBox="0 0 610 228" role="img" aria-label="The loss for one row with label 1 against the dial for the Gaussian, Bernoulli, and Poisson families"><text x="107.0" y="26.0" text-anchor="middle" class="fig-bold fig-small">Gaussian, y = 1</text>
<line x1="25.0" y1="185.0" x2="190.0" y2="185.0" class="stroke-muted" />
<text x="25.0" y="202.0" text-anchor="middle" class="fig-small">−2</text>
<text x="91.0" y="202.0" text-anchor="middle" class="fig-small">0</text>
<text x="157.0" y="202.0" text-anchor="middle" class="fig-small">2</text>
<text x="107.0" y="220.0" text-anchor="middle" class="fig-small">η</text>
<polyline points="25.0,60.6 27.1,65.4 29.2,70.2 31.3,74.9 33.4,79.4 35.4,83.9 37.5,88.2 39.6,92.5 41.7,96.6 43.8,100.7 45.9,104.6 48.0,108.4 50.1,112.1 52.2,115.8 54.2,119.3 56.3,122.7 58.4,126.0 60.5,129.2 62.6,132.3 64.7,135.3 66.8,138.2 68.9,141.0 70.9,143.7 73.0,146.3 75.1,148.8 77.2,151.2 79.3,153.4 81.4,155.6 83.5,157.7 85.6,159.6 87.7,161.5 89.7,163.3 91.8,164.9 93.9,166.5 96.0,167.9 98.1,169.2 100.2,170.5 102.3,171.6 104.4,172.6 106.5,173.6 108.5,174.4 110.6,175.1 112.7,175.7 114.8,176.2 116.9,176.6 119.0,176.9 121.1,177.1 123.2,177.2 125.3,177.2 127.3,177.1 129.4,176.9 131.5,176.5 133.6,176.1 135.7,175.6 137.8,175.0 139.9,174.2 142.0,173.4 144.1,172.4 146.1,171.4 148.2,170.2 150.3,169.0 152.4,167.6 154.5,166.2 156.6,164.6 158.7,162.9 160.8,161.1 162.8,159.3 164.9,157.3 167.0,155.2 169.1,153.0 171.2,150.7 173.3,148.3 175.4,145.8 177.5,143.2 179.6,140.5 181.6,137.7 183.7,134.7 185.8,131.7 187.9,128.6 190.0,125.4" fill="none" class="stroke-b" stroke-width="2.5" />
<text x="302.0" y="26.0" text-anchor="middle" class="fig-bold fig-small">Bernoulli, y = 1</text>
<line x1="220.0" y1="185.0" x2="385.0" y2="185.0" class="stroke-muted" />
<text x="220.0" y="202.0" text-anchor="middle" class="fig-small">−2</text>
<text x="286.0" y="202.0" text-anchor="middle" class="fig-small">0</text>
<text x="352.0" y="202.0" text-anchor="middle" class="fig-small">2</text>
<text x="302.0" y="220.0" text-anchor="middle" class="fig-small">η</text>
<polyline points="220.0,59.2 222.1,62.0 224.2,64.9 226.3,67.7 228.4,70.5 230.4,73.3 232.5,76.1 234.6,78.8 236.7,81.5 238.8,84.2 240.9,86.8 243.0,89.4 245.1,91.9 247.2,94.5 249.2,97.0 251.3,99.4 253.4,101.8 255.5,104.2 257.6,106.5 259.7,108.8 261.8,111.1 263.9,113.2 265.9,115.4 268.0,117.5 270.1,119.5 272.2,121.6 274.3,123.5 276.4,125.4 278.5,127.3 280.6,129.1 282.7,130.8 284.7,132.5 286.8,134.2 288.9,135.7 291.0,137.3 293.1,138.8 295.2,140.2 297.3,141.6 299.4,142.9 301.5,144.2 303.5,145.5 305.6,146.7 307.7,147.8 309.8,148.9 311.9,150.0 314.0,151.0 316.1,151.9 318.2,152.8 320.3,153.7 322.3,154.6 324.4,155.4 326.5,156.1 328.6,156.8 330.7,157.5 332.8,158.2 334.9,158.8 337.0,159.4 339.1,160.0 341.1,160.5 343.2,161.0 345.3,161.5 347.4,161.9 349.5,162.4 351.6,162.8 353.7,163.2 355.8,163.5 357.8,163.9 359.9,164.2 362.0,164.5 364.1,164.8 366.2,165.1 368.3,165.3 370.4,165.6 372.5,165.8 374.6,166.0 376.6,166.2 378.7,166.4 380.8,166.6 382.9,166.8 385.0,166.9" fill="none" class="stroke-b" stroke-width="2.5" />
<text x="497.0" y="26.0" text-anchor="middle" class="fig-bold fig-small">Poisson, y = 1</text>
<line x1="415.0" y1="185.0" x2="580.0" y2="185.0" class="stroke-muted" />
<text x="415.0" y="202.0" text-anchor="middle" class="fig-small">−2</text>
<text x="488.3" y="202.0" text-anchor="middle" class="fig-small">0</text>
<text x="561.7" y="202.0" text-anchor="middle" class="fig-small">2</text>
<text x="497.0" y="220.0" text-anchor="middle" class="fig-small">η</text>
<polyline points="415.0,136.8 417.1,137.3 419.2,137.9 421.3,138.5 423.4,139.0 425.4,139.6 427.5,140.1 429.6,140.6 431.7,141.2 433.8,141.7 435.9,142.2 438.0,142.7 440.1,143.2 442.2,143.7 444.2,144.1 446.3,144.6 448.4,145.0 450.5,145.5 452.6,145.9 454.7,146.3 456.8,146.7 458.9,147.1 460.9,147.4 463.0,147.8 465.1,148.1 467.2,148.4 469.3,148.7 471.4,148.9 473.5,149.2 475.6,149.4 477.7,149.6 479.7,149.7 481.8,149.8 483.9,149.9 486.0,150.0 488.1,150.0 490.2,150.0 492.3,149.9 494.4,149.8 496.5,149.7 498.5,149.5 500.6,149.3 502.7,149.0 504.8,148.6 506.9,148.2 509.0,147.7 511.1,147.2 513.2,146.6 515.3,145.9 517.3,145.2 519.4,144.3 521.5,143.4 523.6,142.4 525.7,141.2 527.8,140.0 529.9,138.7 532.0,137.2 534.1,135.6 536.1,133.9 538.2,132.1 540.3,130.1 542.4,127.9 544.5,125.6 546.6,123.1 548.7,120.4 550.8,117.5 552.8,114.4 554.9,111.1 557.0,107.6 559.1,103.8 561.2,99.7 563.3,95.4 565.4,90.8 567.5,85.9 569.6,80.6 571.6,75.0 573.7,69.0 575.8,62.7 577.9,55.9 580.0,48.7" fill="none" class="stroke-b" stroke-width="2.5" />
<circle cx="124.0" cy="177.2" r="5" class="fill-c"/>
<text x="124.0" y="167.2" text-anchor="middle" class="fig-small">μ = y = 1</text>
<text x="381.0" y="126.0" text-anchor="end" class="fig-small">keeps falling</text>
<circle cx="488.3" cy="150.0" r="5" class="fill-c"/>
<text x="488.3" y="170.0" text-anchor="middle" class="fig-small">μ = y = 1</text></svg>
Figure 3. Each family gives a smooth bowl in the dial. The Gaussian and Poisson bottoms sit where the predicted mean equals the label. The Bernoulli never reaches μ = 1, so it keeps falling toward 0 instead.
```

## Walk downhill

Take small steps against the gradient: $w \leftarrow w - \text{lr} \cdot \nabla L(w)$, where lr is the step size. From zero weights on the shop, one step of 0.05 gives $w = 0 - 0.05 \times [-1, -2] = [0.05, 0.1]$. The loss falls from 3 to 2.797. After 100 steps the weights are $[-0.1415, 0.3819]$ with loss 2.65952. After 500 they reach $[-0.1451, 0.3842]$ and loss 2.65951, the bottom of the bowl.

At the bottom the gradient is 0, which is a statement about the predictions. The means are 0.865, 1.270, and 1.865, so the residuals $\mu - y$ are $-0.135$, $+0.270$, and $-0.135$. They sum to 0, and weighting them by the hour also gives 0. The fit has balanced its errors, and the curve in Figure 1 is that result. The first balance has a plain reading. The predictions add up to $0.865 + 1.270 + 1.865 = 4$, the same as the counts $1 + 1 + 2$, so the model gets the total right even though it misses each hour a little.

For the Gaussian family, this loop is gradient descent on half the squared error, so it lands on the least-squares line. One loop covers both.

> [!KEY]
> Setting the dial to a linear score turns a family into a model. Its loss is $A(\eta) - y\eta$, the negative log of the probability of the label with the constant dropped, and its gradient is $X^\top(\mu - y)$ for every family. That answers the opening question: squared error and the log loss are two cases of one loss, and a count fits the same way.

> [!RECAP]
> Hold these ideas before you write code:
>
> - The dial is $\eta = Xw$ and the prediction is $\mu = A'(\eta)$, which keeps counts positive and probabilities inside $(0, 1)$.
> - The loss is $\sum[A(\eta) - y\eta]$, which is half squared error for the Gaussian and the log loss for the Bernoulli.
> - The gradient is $X^\top(\mu - y)$, because the slope of $A$ is the mean.
> - At zero weights on the shop data the loss is 3 and the gradient is $[-1, -2]$.
> - Walking downhill finds the single bottom of the bowl.

## Build it step by step

The starter already holds `log_partition`, `mean_from_natural`, and `variance_from_natural` from the previous chapter. You write three more functions.

1. `glm_loss(family, w, X, y)` returns one number. Compute `eta = X @ w` and return `np.sum(log_partition(family, eta) - y * eta)`. The common mistake is averaging instead of summing, which changes the scale the checks expect.
2. `glm_gradient(family, w, X, y)` returns an array of shape `(d,)`. Compute `mu` from `eta`, then return `X.T @ (mu - y)`. The common mistake is the sign, `y - mu`.
3. `gradient_descent_glm(family, X, y, lr, steps)` starts at zero weights and returns `(w, history)`. Record the loss at zero weights first, then once after each step, so `history` has `steps + 1` entries. The common mistake is recording before the update only, which leaves out the final loss.

The update is one line:

```python
w = w - lr * glm_gradient(family, w, X, y)
```

## Where this breaks

- **Overdispersion.** A Poisson model forces the variance to equal the mean. Counts that spread wider than that give weights that look sharper than the data justify.
- **Gaussian noise scale.** The Gaussian here has variance 1. The best weights do not depend on the true noise scale, but any measure of uncertainty in them does.
- **A perfect split of coin labels.** The loss has no bottom, so the weights grow forever, and gradient descent never stops improving.
- **Huge dial readings.** $e^{\eta}$ overflows past about 709, so a Poisson fit with wild weights returns infinity.

## The exercise

You write `glm_loss`, `glm_gradient`, and `gradient_descent_glm`. The checks confirm that the loss is 3 at zero weights on the shop data, that the Gaussian loss is half the squared error plus a constant, and that the Bernoulli loss matches the log loss. They confirm that the gradient is $[-1, -2]$ on the shop data and matches finite differences for all three families. They also check that gradient descent lowers the loss at every step, and that on a Gaussian problem it approaches the least-squares answer. The first check passes as soon as `glm_loss` is right.
