---
title: "From Gaussian classes to logistic regression and naive Bayes"
slug: 017-gda-logistic-and-naive-bayes
part: "Part 2 — Classical models"
summary: See why two Gaussian classes with one shared covariance give a logistic regression, then drop the covariance with naive Bayes.
minutes: 75
gpu: false
objectives:
  - Derive the log odds of two Gaussian classes with a shared covariance and show that the quadratic terms cancel.
  - Compute the equivalent logistic weights and bias from means, covariance, and priors, and say when fitting the classes beats fitting the boundary.
  - Explain what independence given the class assumes, and what it throws away.
  - Fit Gaussian naive Bayes with per-class variances and a floor, and a word-count model with Laplace smoothing.
  - Score a document by summing count times log word probability, and say why an unseen word must not veto a class.
prereqs:
  - 008-sigmoid-and-log-loss
  - 009-fitting-logistic-regression
  - 016-bayes-rule-and-gaussian-classes
exercise: 017-gda-logistic-and-naive-bayes
---

# From Gaussian classes to logistic regression and naive Bayes

You have two ways to separate two kinds of parcel. One is to fit a bell cloud to each kind and ask which cloud an unseen parcel belongs to, as [Gaussian discriminant analysis](/lessons/016-bayes-rule-and-gaussian-classes) does. The other is to fit a line directly with [logistic regression](/lessons/009-fitting-logistic-regression). Draw the boundaries and they look alike: both are straight. That is odd, because the first method never draws a line. It only describes clouds.

The first half of this chapter explains the coincidence, which is not one. When two Gaussian clouds share a shape, the rule that the cloud method follows is a logistic regression, down to the formula. The only difference is how the weights are found: logistic regression walks downhill, and the cloud method reads them off the averages.

The cloud method has a cost, though. Suppose the "parcels" are emails described by ten thousand word counts. The shared covariance has $10^8$ numbers, about 800 megabytes, to be estimated from a few thousand emails. The second half stops asking how every pair of words moves together and asks about one word at a time. That shortcut, naive Bayes, is wrong in a specific way, and the chapter shows why it still classifies well.

## What you already know

- A logistic regression scores a point with $w^{\top}x + b$ and turns the score into a probability with the sigmoid, as in [sigmoid and log loss](/lessons/008-sigmoid-and-log-loss). A score of 0 means a probability of 0.5, so the boundary is where the score is 0.
- GDA fits a mean per class, one shared covariance $\Sigma$, and class priors, and its posterior is the prior times the density, divided by the total.
- A log turns a product into a sum, and the log of a ratio is a difference of logs.
- A Gaussian is fitted by averaging, and a covariance matrix records how features move together.
- `log_sum_exp` turns log scores into probabilities.

## Two routes to the same line

Take the four points from the last chapter: class 0 at 1 and 3, class 1 at 5 and 7. GDA gives two bells centred at 2 and 6, both with variance 1 and equal priors. By symmetry they have equal height at their midpoint, $x = 4$, so the boundary is there.

```figure
<svg viewBox="0 0 620 380" role="img" aria-label="Top: two bells centred at 2 and 6 with the decision boundary at x equals 4. Bottom: the straight line 4x minus 16, the log odds, crossing zero at the same x">
<text x="60" y="18" class="fig-bold">Two bells, equal priors</text>
<line x1="60" y1="150" x2="580" y2="150" class="stroke-muted"/>
<polygon points="60.0,150.0 60.0,135.6 63.2,134.1 66.5,132.5 69.8,130.8 73.0,128.9 76.2,127.0 79.5,124.9 82.8,122.7 86.0,120.4 89.2,118.0 92.5,115.5 95.8,112.8 99.0,110.1 102.2,107.2 105.5,104.3 108.8,101.3 112.0,98.2 115.2,95.1 118.5,91.9 121.8,88.7 125.0,85.5 128.2,82.3 131.5,79.0 134.8,75.9 138.0,72.7 141.2,69.7 144.5,66.7 147.8,63.9 151.0,61.1 154.2,58.5 157.5,56.1 160.8,53.9 164.0,51.8 167.2,49.9 170.5,48.3 173.8,46.9 177.0,45.7 180.2,44.8 183.5,44.1 186.8,43.7 190.0,43.6 193.3,43.7 196.5,44.1 199.8,44.8 203.0,45.7 206.2,46.9 209.5,48.3 212.8,49.9 216.0,51.8 219.2,53.9 222.5,56.1 225.8,58.5 229.0,61.1 232.3,63.9 235.5,66.7 238.8,69.7 242.0,72.7 245.2,75.9 248.5,79.0 251.8,82.3 255.0,85.5 258.2,88.7 261.5,91.9 264.8,95.1 268.0,98.2 271.2,101.3 274.5,104.3 277.8,107.2 281.0,110.1 284.2,112.8 287.5,115.5 290.8,118.0 294.0,120.4 297.2,122.7 300.5,124.9 303.8,127.0 307.0,128.9 310.2,130.8 313.5,132.5 316.8,134.1 320.0,135.6 323.2,137.0 326.5,138.3 329.8,139.5 333.0,140.5 336.2,141.5 339.5,142.4 342.8,143.3 346.0,144.0 349.2,144.7 352.5,145.3 355.8,145.9 359.0,146.4 362.2,146.8 365.5,147.2 368.8,147.6 372.0,147.9 375.3,148.2 378.5,148.4 381.8,148.6 385.0,148.8 388.3,149.0 391.5,149.1 394.8,149.3 398.0,149.4 401.2,149.5 404.5,149.5 407.8,149.6 411.0,149.7 414.2,149.7 417.5,149.8 420.8,149.8 424.0,149.8 427.2,149.9 430.5,149.9 433.8,149.9 437.0,149.9 440.3,149.9 443.5,149.9 446.8,150.0 450.0,150.0 453.3,150.0 456.5,150.0 459.8,150.0 463.0,150.0 466.2,150.0 469.5,150.0 472.8,150.0 476.0,150.0 479.2,150.0 482.5,150.0 485.8,150.0 489.0,150.0 492.2,150.0 495.5,150.0 498.8,150.0 502.0,150.0 505.3,150.0 508.5,150.0 511.8,150.0 515.0,150.0 518.2,150.0 521.5,150.0 524.8,150.0 528.0,150.0 531.2,150.0 534.5,150.0 537.8,150.0 541.0,150.0 544.2,150.0 547.5,150.0 550.8,150.0 554.0,150.0 557.2,150.0 560.5,150.0 563.8,150.0 567.0,150.0 570.2,150.0 573.5,150.0 576.8,150.0 580.0,150.0 580.0,150.0" class="fill-soft-a"/><polyline points="60.0,135.6 63.2,134.1 66.5,132.5 69.8,130.8 73.0,128.9 76.2,127.0 79.5,124.9 82.8,122.7 86.0,120.4 89.2,118.0 92.5,115.5 95.8,112.8 99.0,110.1 102.2,107.2 105.5,104.3 108.8,101.3 112.0,98.2 115.2,95.1 118.5,91.9 121.8,88.7 125.0,85.5 128.2,82.3 131.5,79.0 134.8,75.9 138.0,72.7 141.2,69.7 144.5,66.7 147.8,63.9 151.0,61.1 154.2,58.5 157.5,56.1 160.8,53.9 164.0,51.8 167.2,49.9 170.5,48.3 173.8,46.9 177.0,45.7 180.2,44.8 183.5,44.1 186.8,43.7 190.0,43.6 193.3,43.7 196.5,44.1 199.8,44.8 203.0,45.7 206.2,46.9 209.5,48.3 212.8,49.9 216.0,51.8 219.2,53.9 222.5,56.1 225.8,58.5 229.0,61.1 232.3,63.9 235.5,66.7 238.8,69.7 242.0,72.7 245.2,75.9 248.5,79.0 251.8,82.3 255.0,85.5 258.2,88.7 261.5,91.9 264.8,95.1 268.0,98.2 271.2,101.3 274.5,104.3 277.8,107.2 281.0,110.1 284.2,112.8 287.5,115.5 290.8,118.0 294.0,120.4 297.2,122.7 300.5,124.9 303.8,127.0 307.0,128.9 310.2,130.8 313.5,132.5 316.8,134.1 320.0,135.6 323.2,137.0 326.5,138.3 329.8,139.5 333.0,140.5 336.2,141.5 339.5,142.4 342.8,143.3 346.0,144.0 349.2,144.7 352.5,145.3 355.8,145.9 359.0,146.4 362.2,146.8 365.5,147.2 368.8,147.6 372.0,147.9 375.3,148.2 378.5,148.4 381.8,148.6 385.0,148.8 388.3,149.0 391.5,149.1 394.8,149.3 398.0,149.4 401.2,149.5 404.5,149.5 407.8,149.6 411.0,149.7 414.2,149.7 417.5,149.8 420.8,149.8 424.0,149.8 427.2,149.9 430.5,149.9 433.8,149.9 437.0,149.9 440.3,149.9 443.5,149.9 446.8,150.0 450.0,150.0 453.3,150.0 456.5,150.0 459.8,150.0 463.0,150.0 466.2,150.0 469.5,150.0 472.8,150.0 476.0,150.0 479.2,150.0 482.5,150.0 485.8,150.0 489.0,150.0 492.2,150.0 495.5,150.0 498.8,150.0 502.0,150.0 505.3,150.0 508.5,150.0 511.8,150.0 515.0,150.0 518.2,150.0 521.5,150.0 524.8,150.0 528.0,150.0 531.2,150.0 534.5,150.0 537.8,150.0 541.0,150.0 544.2,150.0 547.5,150.0 550.8,150.0 554.0,150.0 557.2,150.0 560.5,150.0 563.8,150.0 567.0,150.0 570.2,150.0 573.5,150.0 576.8,150.0 580.0,150.0" fill="none" class="stroke-a"/>
<polygon points="60.0,150.0 60.0,150.0 63.2,150.0 66.5,150.0 69.8,150.0 73.0,150.0 76.2,150.0 79.5,150.0 82.8,150.0 86.0,150.0 89.2,150.0 92.5,150.0 95.8,150.0 99.0,150.0 102.2,150.0 105.5,150.0 108.8,150.0 112.0,150.0 115.2,150.0 118.5,150.0 121.8,150.0 125.0,150.0 128.2,150.0 131.5,150.0 134.8,150.0 138.0,150.0 141.2,150.0 144.5,150.0 147.8,150.0 151.0,150.0 154.2,150.0 157.5,150.0 160.8,150.0 164.0,150.0 167.2,150.0 170.5,150.0 173.8,150.0 177.0,150.0 180.2,150.0 183.5,150.0 186.8,150.0 190.0,150.0 193.3,150.0 196.5,149.9 199.8,149.9 203.0,149.9 206.2,149.9 209.5,149.9 212.8,149.9 216.0,149.8 219.2,149.8 222.5,149.8 225.8,149.7 229.0,149.7 232.3,149.6 235.5,149.5 238.8,149.5 242.0,149.4 245.2,149.3 248.5,149.1 251.8,149.0 255.0,148.8 258.2,148.6 261.5,148.4 264.8,148.2 268.0,147.9 271.2,147.6 274.5,147.2 277.8,146.8 281.0,146.4 284.2,145.9 287.5,145.3 290.8,144.7 294.0,144.0 297.2,143.3 300.5,142.4 303.8,141.5 307.0,140.5 310.2,139.5 313.5,138.3 316.8,137.0 320.0,135.6 323.2,134.1 326.5,132.5 329.8,130.8 333.0,128.9 336.2,127.0 339.5,124.9 342.8,122.7 346.0,120.4 349.2,118.0 352.5,115.5 355.8,112.8 359.0,110.1 362.2,107.2 365.5,104.3 368.8,101.3 372.0,98.2 375.3,95.1 378.5,91.9 381.8,88.7 385.0,85.5 388.3,82.3 391.5,79.0 394.8,75.9 398.0,72.7 401.2,69.7 404.5,66.7 407.8,63.9 411.0,61.1 414.2,58.5 417.5,56.1 420.8,53.9 424.0,51.8 427.2,49.9 430.5,48.3 433.8,46.9 437.0,45.7 440.3,44.8 443.5,44.1 446.8,43.7 450.0,43.6 453.3,43.7 456.5,44.1 459.8,44.8 463.0,45.7 466.2,46.9 469.5,48.3 472.8,49.9 476.0,51.8 479.2,53.9 482.5,56.1 485.8,58.5 489.0,61.1 492.2,63.9 495.5,66.7 498.8,69.7 502.0,72.7 505.3,75.9 508.5,79.0 511.8,82.3 515.0,85.5 518.2,88.7 521.5,91.9 524.8,95.1 528.0,98.2 531.2,101.3 534.5,104.3 537.8,107.2 541.0,110.1 544.2,112.8 547.5,115.5 550.8,118.0 554.0,120.4 557.2,122.7 560.5,124.9 563.8,127.0 567.0,128.9 570.2,130.8 573.5,132.5 576.8,134.1 580.0,135.6 580.0,150.0" class="fill-soft-b"/><polyline points="60.0,150.0 63.2,150.0 66.5,150.0 69.8,150.0 73.0,150.0 76.2,150.0 79.5,150.0 82.8,150.0 86.0,150.0 89.2,150.0 92.5,150.0 95.8,150.0 99.0,150.0 102.2,150.0 105.5,150.0 108.8,150.0 112.0,150.0 115.2,150.0 118.5,150.0 121.8,150.0 125.0,150.0 128.2,150.0 131.5,150.0 134.8,150.0 138.0,150.0 141.2,150.0 144.5,150.0 147.8,150.0 151.0,150.0 154.2,150.0 157.5,150.0 160.8,150.0 164.0,150.0 167.2,150.0 170.5,150.0 173.8,150.0 177.0,150.0 180.2,150.0 183.5,150.0 186.8,150.0 190.0,150.0 193.3,150.0 196.5,149.9 199.8,149.9 203.0,149.9 206.2,149.9 209.5,149.9 212.8,149.9 216.0,149.8 219.2,149.8 222.5,149.8 225.8,149.7 229.0,149.7 232.3,149.6 235.5,149.5 238.8,149.5 242.0,149.4 245.2,149.3 248.5,149.1 251.8,149.0 255.0,148.8 258.2,148.6 261.5,148.4 264.8,148.2 268.0,147.9 271.2,147.6 274.5,147.2 277.8,146.8 281.0,146.4 284.2,145.9 287.5,145.3 290.8,144.7 294.0,144.0 297.2,143.3 300.5,142.4 303.8,141.5 307.0,140.5 310.2,139.5 313.5,138.3 316.8,137.0 320.0,135.6 323.2,134.1 326.5,132.5 329.8,130.8 333.0,128.9 336.2,127.0 339.5,124.9 342.8,122.7 346.0,120.4 349.2,118.0 352.5,115.5 355.8,112.8 359.0,110.1 362.2,107.2 365.5,104.3 368.8,101.3 372.0,98.2 375.3,95.1 378.5,91.9 381.8,88.7 385.0,85.5 388.3,82.3 391.5,79.0 394.8,75.9 398.0,72.7 401.2,69.7 404.5,66.7 407.8,63.9 411.0,61.1 414.2,58.5 417.5,56.1 420.8,53.9 424.0,51.8 427.2,49.9 430.5,48.3 433.8,46.9 437.0,45.7 440.3,44.8 443.5,44.1 446.8,43.7 450.0,43.6 453.3,43.7 456.5,44.1 459.8,44.8 463.0,45.7 466.2,46.9 469.5,48.3 472.8,49.9 476.0,51.8 479.2,53.9 482.5,56.1 485.8,58.5 489.0,61.1 492.2,63.9 495.5,66.7 498.8,69.7 502.0,72.7 505.3,75.9 508.5,79.0 511.8,82.3 515.0,85.5 518.2,88.7 521.5,91.9 524.8,95.1 528.0,98.2 531.2,101.3 534.5,104.3 537.8,107.2 541.0,110.1 544.2,112.8 547.5,115.5 550.8,118.0 554.0,120.4 557.2,122.7 560.5,124.9 563.8,127.0 567.0,128.9 570.2,130.8 573.5,132.5 576.8,134.1 580.0,135.6" fill="none" class="stroke-b"/>
<text x="190.0" y="33.3" text-anchor="middle" class="fig-small">class 0</text>
<text x="450.0" y="33.3" text-anchor="middle" class="fig-small">class 1</text>
<text x="60" y="190" class="fig-bold">Log odds 4x - 16</text>
<line x1="60" y1="270.0" x2="580" y2="270.0" class="stroke-muted"/>
<line x1="60.0" y1="267.0" x2="60.0" y2="273.0" class="stroke-muted"/>
<text x="60.0" y="362" text-anchor="middle" class="fig-small">0</text>
<line x1="125.0" y1="267.0" x2="125.0" y2="273.0" class="stroke-muted"/>
<text x="125.0" y="362" text-anchor="middle" class="fig-small">1</text>
<line x1="190.0" y1="267.0" x2="190.0" y2="273.0" class="stroke-muted"/>
<text x="190.0" y="362" text-anchor="middle" class="fig-small">2</text>
<line x1="255.0" y1="267.0" x2="255.0" y2="273.0" class="stroke-muted"/>
<text x="255.0" y="362" text-anchor="middle" class="fig-small">3</text>
<line x1="320.0" y1="267.0" x2="320.0" y2="273.0" class="stroke-muted"/>
<text x="320.0" y="362" text-anchor="middle" class="fig-small">4</text>
<line x1="385.0" y1="267.0" x2="385.0" y2="273.0" class="stroke-muted"/>
<text x="385.0" y="362" text-anchor="middle" class="fig-small">5</text>
<line x1="450.0" y1="267.0" x2="450.0" y2="273.0" class="stroke-muted"/>
<text x="450.0" y="362" text-anchor="middle" class="fig-small">6</text>
<line x1="515.0" y1="267.0" x2="515.0" y2="273.0" class="stroke-muted"/>
<text x="515.0" y="362" text-anchor="middle" class="fig-small">7</text>
<line x1="580.0" y1="267.0" x2="580.0" y2="273.0" class="stroke-muted"/>
<text x="580.0" y="362" text-anchor="middle" class="fig-small">8</text>
<text x="320" y="378" text-anchor="middle" class="fig-small">feature x</text>
<polyline points="60.0,340.0 580.0,200.0" fill="none" class="stroke-b" stroke-width="2.5"/>
<line x1="320.0" y1="30" x2="320.0" y2="270.0" class="stroke-ink dashed"/>
<circle cx="320.0" cy="270.0" r="5" class="fill-ink"/>
<text x="328.0" y="46" class="fig-bold">boundary x = 4</text>
<circle cx="255.0" cy="287.5" r="4" class="fill-b"/>
<text x="281.0" y="317.5" class="fig-small">x = 3: score -4, probability 0.018</text>
</svg>
Figure 1. Top: the two fitted bells. Bottom: the log odds of class 1 against class 0, which is a straight line. It crosses zero at x = 4, exactly where the two bells have equal height, so the boundary from the bells and the boundary from the line are the same cut.
```

The top of Figure 1 shows the bells. The bottom shows a different quantity, the score $4x - 16$, which is the log odds of class 1 against class 0. It is a straight line, and it crosses zero at $x = 4$, the same place. The next sections show why that line exists and where 4 and $-16$ come from.

## The log odds

Compare the classes through the *odds* of class 1 against class 0, the ratio $p(1 \mid x) / p(0 \mid x)$. Bayes' rule gives both posteriors the same denominator, the total $p(x)$, and a ratio makes it cancel. The log of the ratio is the *log odds*, and it is exactly what a sigmoid turns into a probability: if the log odds is $z$, then $p(1 \mid x) = e^{z}/(1 + e^{z}) = \sigma(z)$. Write it out from Bayes' rule:

$$
\log\frac{p(1 \mid x)}{p(0 \mid x)} = \log\frac{\pi_1}{\pi_0} + \log p(x \mid y = 1) - \log p(x \mid y = 0).
$$

Substitute the Gaussian log density. The constant $-\tfrac d2\log 2\pi$ and the term $-\tfrac12\log\det\Sigma$ are the same for both classes, because the covariance is shared, so they cancel in the difference. What remains is the distance term. Expand it for a class with mean $\mu$:

$$
-\tfrac12 (x-\mu)^{\top}\Sigma^{-1}(x-\mu) = -\tfrac12 x^{\top}\Sigma^{-1}x + \mu^{\top}\Sigma^{-1}x - \tfrac12\mu^{\top}\Sigma^{-1}\mu.
$$

The first piece, $-\tfrac12 x^{\top}\Sigma^{-1}x$, is quadratic in $x$, the only part that would make a curve. It is the same for both classes, so it cancels too. Subtracting the class 0 version from the class 1 version leaves a score that is linear in $x$:

$$
\log\frac{p(1 \mid x)}{p(0 \mid x)} = w^{\top}x + b, \qquad
w = \Sigma^{-1}(\mu_1 - \mu_0), \qquad
b = \log\frac{\pi_1}{\pi_0} - \tfrac12(\mu_0 + \mu_1)^{\top}w.
$$

So $p(1 \mid x) = \sigma(w^{\top}x + b)$, which is the model logistic regression fits. A shared covariance makes the quadratic pieces cancel, and a straight score is what is left.

## Read the weights aloud

The weights $w$ are the difference between the class means, re-weighted by the inverse shape. The difference of means points from class 0 to class 1, the direction in which moving the point is the most evidence for class 1. The inverse shape shrinks directions where the cloud is wide, because a point is often far along them by chance, and boosts directions where it is narrow.

The bias is the prior plus a midpoint term. Without priors, $b = -w^{\top}m$ with $m$ the midpoint of the two means, so the score is $w^{\top}(x - m)$ and the boundary passes through the midpoint. When class 1 is rarer, $\log(\pi_1/\pi_0)$ is negative, so the score drops and class 1 needs stronger evidence.

On the four points, $\mu_1 - \mu_0 = 4$ and $\Sigma = 1$, so $w = 4$. The midpoint is 4, so $b = 0 - 4 \times 4 = -16$. At $x = 3$ the score is $4 \times 3 - 16 = -4$, and $\sigma(-4) = 0.018$, the class 1 probability that GDA found by comparing densities. At $x = 4$ the score is 0 and the probability is 0.5.

In one dimension the shape only rescales. In two dimensions it turns the weights. Take clouds that lean up and to the right, with $\Sigma = \begin{bmatrix} 1.5 & 1.2 \\ 1.2 & 1.5 \end{bmatrix}$ and class means that differ by $(3, 0)$. The inverse shape turns that difference into $w = \Sigma^{-1}(3, 0) = (5.56, -4.44)$, which points down and to the right, not along the means. The reason is the lean: a high second feature usually means the point sits far along the lean, which also raises the first feature, so part of a high first feature is noise. The weights give the second feature a negative sign to subtract that part out.

> [!KEY]
> With two classes and one shared covariance, the quadratic terms cancel in the log odds, so GDA's boundary is a straight logistic-regression line, with weights $\Sigma^{-1}(\mu_1 - \mu_0)$ that come from averages and not from descent. If each class has its own covariance, the quadratic pieces differ and the boundary curves. That model is *quadratic discriminant analysis*.

## When fitting the clouds pays

Both methods produce $\sigma(w^{\top}x + b)$, and they pick $w$ and $b$ by different goals. Logistic regression maximises how well the line predicts the labels, which is why it needs a loop. GDA maximises how well the clouds explain the points and the labels together, which is why it needs only averages and one solve. Why fit the clouds if the line is all you need? With the Gaussian assumption true, the closed form wastes less data. With it false, the closed form is fragile. Figure 2 measures both on simulated three-feature Gaussian data.

```figure
<svg viewBox="0 0 620 330" role="img" aria-label="Two panels of test accuracy against training size for GDA and logistic regression. Left, clean Gaussian data. Right, the same with three far outliers added">
<text x="60" y="30" class="fig-bold">Clean Gaussian data</text>
<line x1="60" y1="260" x2="300" y2="260" class="stroke-muted"/><line x1="60" y1="260" x2="60" y2="60" class="stroke-muted"/>
<line x1="60" y1="260.0" x2="300" y2="260.0" class="stroke-grid"/>
<text x="54" y="264.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="60" y1="220.0" x2="300" y2="220.0" class="stroke-grid"/>
<text x="54" y="224.0" text-anchor="end" class="fig-small">0.6</text>
<line x1="60" y1="180.0" x2="300" y2="180.0" class="stroke-grid"/>
<text x="54" y="184.0" text-anchor="end" class="fig-small">0.7</text>
<line x1="60" y1="140.0" x2="300" y2="140.0" class="stroke-grid"/>
<text x="54" y="144.0" text-anchor="end" class="fig-small">0.8</text>
<line x1="60" y1="100.0" x2="300" y2="100.0" class="stroke-grid"/>
<text x="54" y="104.0" text-anchor="end" class="fig-small">0.9</text>
<line x1="60" y1="60.0" x2="300" y2="60.0" class="stroke-grid"/>
<text x="54" y="64.0" text-anchor="end" class="fig-small">1.0</text>
<text x="60.0" y="278" text-anchor="middle" class="fig-small">10</text>
<text x="108.0" y="278" text-anchor="middle" class="fig-small">20</text>
<text x="156.0" y="278" text-anchor="middle" class="fig-small">40</text>
<text x="204.0" y="278" text-anchor="middle" class="fig-small">80</text>
<text x="252.0" y="278" text-anchor="middle" class="fig-small">160</text>
<text x="300.0" y="278" text-anchor="middle" class="fig-small">320</text>
<text x="180" y="300" text-anchor="middle" class="fig-small">training rows (log scale)</text>
<polyline points="60.0,96.8 108.0,84.7 156.0,80.7 204.0,79.3 252.0,78.4 300.0,77.9" fill="none" class="stroke-b" stroke-width="2.5"/>
<polyline points="60.0,101.3 108.0,90.3 156.0,84.6 204.0,81.4 252.0,79.4 300.0,78.5" fill="none" class="stroke-a" stroke-width="2.5"/>
<circle cx="60.0" cy="96.8" r="3.5" class="fill-b"/><circle cx="60.0" cy="101.3" r="3.5" class="fill-a"/>
<circle cx="108.0" cy="84.7" r="3.5" class="fill-b"/><circle cx="108.0" cy="90.3" r="3.5" class="fill-a"/>
<circle cx="156.0" cy="80.7" r="3.5" class="fill-b"/><circle cx="156.0" cy="84.6" r="3.5" class="fill-a"/>
<circle cx="204.0" cy="79.3" r="3.5" class="fill-b"/><circle cx="204.0" cy="81.4" r="3.5" class="fill-a"/>
<circle cx="252.0" cy="78.4" r="3.5" class="fill-b"/><circle cx="252.0" cy="79.4" r="3.5" class="fill-a"/>
<circle cx="300.0" cy="77.9" r="3.5" class="fill-b"/><circle cx="300.0" cy="78.5" r="3.5" class="fill-a"/>
<text x="360" y="30" class="fig-bold">Plus three far outliers</text>
<line x1="360" y1="260" x2="600" y2="260" class="stroke-muted"/><line x1="360" y1="260" x2="360" y2="60" class="stroke-muted"/>
<line x1="360" y1="260.0" x2="600" y2="260.0" class="stroke-grid"/>
<line x1="360" y1="220.0" x2="600" y2="220.0" class="stroke-grid"/>
<line x1="360" y1="180.0" x2="600" y2="180.0" class="stroke-grid"/>
<line x1="360" y1="140.0" x2="600" y2="140.0" class="stroke-grid"/>
<line x1="360" y1="100.0" x2="600" y2="100.0" class="stroke-grid"/>
<line x1="360" y1="60.0" x2="600" y2="60.0" class="stroke-grid"/>
<text x="360.0" y="278" text-anchor="middle" class="fig-small">10</text>
<text x="408.0" y="278" text-anchor="middle" class="fig-small">20</text>
<text x="456.0" y="278" text-anchor="middle" class="fig-small">40</text>
<text x="504.0" y="278" text-anchor="middle" class="fig-small">80</text>
<text x="552.0" y="278" text-anchor="middle" class="fig-small">160</text>
<text x="600.0" y="278" text-anchor="middle" class="fig-small">320</text>
<text x="480" y="300" text-anchor="middle" class="fig-small">training rows (log scale)</text>
<polyline points="360.0,191.4 408.0,173.6 456.0,159.4 504.0,144.9 552.0,127.2 600.0,105.5" fill="none" class="stroke-b" stroke-width="2.5"/>
<polyline points="360.0,101.3 408.0,90.3 456.0,84.6 504.0,81.4 552.0,79.4 600.0,78.5" fill="none" class="stroke-a" stroke-width="2.5"/>
<circle cx="360.0" cy="191.4" r="3.5" class="fill-b"/><circle cx="360.0" cy="101.3" r="3.5" class="fill-a"/>
<circle cx="408.0" cy="173.6" r="3.5" class="fill-b"/><circle cx="408.0" cy="90.3" r="3.5" class="fill-a"/>
<circle cx="456.0" cy="159.4" r="3.5" class="fill-b"/><circle cx="456.0" cy="84.6" r="3.5" class="fill-a"/>
<circle cx="504.0" cy="144.9" r="3.5" class="fill-b"/><circle cx="504.0" cy="81.4" r="3.5" class="fill-a"/>
<circle cx="552.0" cy="127.2" r="3.5" class="fill-b"/><circle cx="552.0" cy="79.4" r="3.5" class="fill-a"/>
<circle cx="600.0" cy="105.5" r="3.5" class="fill-b"/><circle cx="600.0" cy="78.5" r="3.5" class="fill-a"/>
<text x="64" y="52" class="fig-small">test accuracy</text>
<circle cx="90" cy="215" r="4" class="fill-b"/><text x="100" y="219" class="fig-small">GDA</text>
<circle cx="90" cy="235" r="4" class="fill-a"/><text x="100" y="239" class="fig-small">logistic regression</text>
<text x="120" y="320" class="fig-small fig-bold" fill="none"></text>
</svg>
Figure 2. Orange is GDA and blue is logistic regression, averaged over 400 random training sets per size. On clean data GDA is ahead at every size and the gap closes as rows grow: GDA at 20 rows (0.938) matches logistic regression at 40 (0.939). With three class 1 points 20 units beyond their class mean, GDA drops to 0.67 at 10 rows and 0.89 at 320, while logistic regression does not change.
```

On clean data, GDA is ahead at every size, by about one percentage point with 10 to 20 rows, and the gap closes as the rows grow. GDA at 20 rows scores 0.938, about what logistic regression scores at 40 rows. The gain is modest, and it is real only because the data really are Gaussian with a shared shape.

On the right, three class 1 points sit 20 units beyond their class mean, on the far side from class 0. Logistic regression does not move at all. GDA falls to 0.67 at 10 rows and still reaches only 0.89 at 320. GDA's means and covariance are averages over every point, so a far point drags the mean and swells the covariance, which tilts the line. A logistic gradient for each point is $p - y$, the prediction minus the label, so a point that the model already classifies correctly and confidently contributes almost nothing. A far point on the wrong side would hurt logistic regression too, so the figure shows what each method reads from the data, not that one is always safer. GDA trusts the Gaussian shape, so it needs fewer rows but is hurt by outliers, and logistic regression trusts only the labels.

## The bill for a covariance

GDA's shared covariance has $d^2$ numbers. For $d = 10{,}000$ word counts that is $10^8$ numbers estimated from far fewer emails, so the estimate cannot be trusted and the matrix cannot even be factored. The way out is to stop asking how every pair of words moves together. It costs one number per word and class, where the full model needed $10^8$.

To feed an email to a classifier, throw away the order and keep the counts. This is a *bag of words*: the email becomes a row of $V$ counts, where $V$ is the vocabulary size. Figure 3 shows a toy version with six words and four documents, two spam and two ham.

```figure
<svg viewBox="0 0 620 440" role="img" aria-label="A grid of word counts for four toy documents, then per-class word probabilities before and after adding one to every count">
<text x="10" y="22" class="fig-bold">Word counts per document</text>
<text x="160.0" y="46" text-anchor="middle" class="fig-small">free</text>
<text x="220.0" y="46" text-anchor="middle" class="fig-small">money</text>
<text x="280.0" y="46" text-anchor="middle" class="fig-small">win</text>
<text x="340.0" y="46" text-anchor="middle" class="fig-small">meeting</text>
<text x="400.0" y="46" text-anchor="middle" class="fig-small">report</text>
<text x="460.0" y="46" text-anchor="middle" class="fig-small">lunch</text>
<text x="120" y="71.0" text-anchor="end" class="fig-small">spam 1</text>
<rect x="132" y="56" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="160.0" y="71.0" text-anchor="middle" class="fig-bold">2</text>
<rect x="192" y="56" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="220.0" y="71.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="252" y="56" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="280.0" y="71.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="312" y="56" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="340.0" y="71.0" text-anchor="middle" class="fig-small">0</text>
<rect x="372" y="56" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="400.0" y="71.0" text-anchor="middle" class="fig-small">0</text>
<rect x="432" y="56" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="460.0" y="71.0" text-anchor="middle" class="fig-small">0</text>
<text x="120" y="97.0" text-anchor="end" class="fig-small">spam 2</text>
<rect x="132" y="82" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="160.0" y="97.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="192" y="82" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="220.0" y="97.0" text-anchor="middle" class="fig-bold">2</text>
<rect x="252" y="82" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="280.0" y="97.0" text-anchor="middle" class="fig-small">0</text>
<rect x="312" y="82" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="340.0" y="97.0" text-anchor="middle" class="fig-small">0</text>
<rect x="372" y="82" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="400.0" y="97.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="432" y="82" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="460.0" y="97.0" text-anchor="middle" class="fig-small">0</text>
<text x="120" y="123.0" text-anchor="end" class="fig-small">ham 1</text>
<rect x="132" y="108" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="160.0" y="123.0" text-anchor="middle" class="fig-small">0</text>
<rect x="192" y="108" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="220.0" y="123.0" text-anchor="middle" class="fig-small">0</text>
<rect x="252" y="108" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="280.0" y="123.0" text-anchor="middle" class="fig-small">0</text>
<rect x="312" y="108" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="340.0" y="123.0" text-anchor="middle" class="fig-bold">2</text>
<rect x="372" y="108" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="400.0" y="123.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="432" y="108" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="460.0" y="123.0" text-anchor="middle" class="fig-bold">1</text>
<text x="120" y="149.0" text-anchor="end" class="fig-small">ham 2</text>
<rect x="132" y="134" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="160.0" y="149.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="192" y="134" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="220.0" y="149.0" text-anchor="middle" class="fig-small">0</text>
<rect x="252" y="134" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="280.0" y="149.0" text-anchor="middle" class="fig-small">0</text>
<rect x="312" y="134" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="340.0" y="149.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="372" y="134" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="400.0" y="149.0" text-anchor="middle" class="fig-bold">2</text>
<rect x="432" y="134" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="460.0" y="149.0" text-anchor="middle" class="fig-small">0</text>
<text x="120" y="183.0" text-anchor="end" class="fig-small">spam total</text>
<rect x="132" y="168" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="160.0" y="183.0" text-anchor="middle" class="fig-bold">3</text>
<rect x="192" y="168" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="220.0" y="183.0" text-anchor="middle" class="fig-bold">3</text>
<rect x="252" y="168" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="280.0" y="183.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="312" y="168" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="340.0" y="183.0" text-anchor="middle" class="fig-small">0</text>
<rect x="372" y="168" width="56" height="22" rx="3" class="fill-soft-b"/>
<text x="400.0" y="183.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="432" y="168" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="460.0" y="183.0" text-anchor="middle" class="fig-small">0</text>
<text x="120" y="209.0" text-anchor="end" class="fig-small">ham total</text>
<rect x="132" y="194" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="160.0" y="209.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="192" y="194" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="220.0" y="209.0" text-anchor="middle" class="fig-small">0</text>
<rect x="252" y="194" width="56" height="22" rx="3" class="fill-sunk"/>
<text x="280.0" y="209.0" text-anchor="middle" class="fig-small">0</text>
<rect x="312" y="194" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="340.0" y="209.0" text-anchor="middle" class="fig-bold">3</text>
<rect x="372" y="194" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="400.0" y="209.0" text-anchor="middle" class="fig-bold">3</text>
<rect x="432" y="194" width="56" height="22" rx="3" class="fill-soft-a"/>
<text x="460.0" y="209.0" text-anchor="middle" class="fig-bold">1</text>
<text x="496" y="183.0" class="fig-small">= 8</text><text x="496" y="209.0" class="fig-small">= 8</text>
<text x="30" y="252" class="fig-bold">spam: word probability</text>
<line x1="30" y1="380" x2="290" y2="380" class="stroke-muted"/>
<rect x="34.0" y="286.2" width="17" height="93.8" class="fill-soft-b stroke-b"/>
<rect x="53.0" y="308.6" width="17" height="71.4" class="fill-b"/>
<text x="52.0" y="394" text-anchor="middle" class="fig-small">free</text>
<rect x="77.3" y="286.2" width="17" height="93.8" class="fill-soft-b stroke-b"/>
<rect x="96.3" y="308.6" width="17" height="71.4" class="fill-b"/>
<text x="95.3" y="394" text-anchor="middle" class="fig-small">money</text>
<rect x="120.7" y="348.8" width="17" height="31.2" class="fill-soft-b stroke-b"/>
<rect x="139.7" y="344.3" width="17" height="35.7" class="fill-b"/>
<text x="138.7" y="394" text-anchor="middle" class="fig-small">win</text>
<rect x="164.0" y="380.0" width="17" height="0.0" class="fill-soft-b stroke-b"/>
<rect x="183.0" y="362.1" width="17" height="17.9" class="fill-b"/>
<text x="182.0" y="394" text-anchor="middle" class="fig-small">meeting</text>
<text x="172.0" y="374" text-anchor="middle" class="fig-small fig-bold">0</text>
<rect x="207.3" y="348.8" width="17" height="31.2" class="fill-soft-b stroke-b"/>
<rect x="226.3" y="344.3" width="17" height="35.7" class="fill-b"/>
<text x="225.3" y="394" text-anchor="middle" class="fig-small">report</text>
<rect x="250.7" y="380.0" width="17" height="0.0" class="fill-soft-b stroke-b"/>
<rect x="269.7" y="362.1" width="17" height="17.9" class="fill-b"/>
<text x="268.7" y="394" text-anchor="middle" class="fig-small">lunch</text>
<text x="258.7" y="374" text-anchor="middle" class="fig-small fig-bold">0</text>
<text x="330" y="252" class="fig-bold">ham: word probability</text>
<line x1="330" y1="380" x2="590" y2="380" class="stroke-muted"/>
<rect x="334.0" y="348.8" width="17" height="31.2" class="fill-soft-a stroke-a"/>
<rect x="353.0" y="344.3" width="17" height="35.7" class="fill-a"/>
<text x="352.0" y="394" text-anchor="middle" class="fig-small">free</text>
<rect x="377.3" y="380.0" width="17" height="0.0" class="fill-soft-a stroke-a"/>
<rect x="396.3" y="362.1" width="17" height="17.9" class="fill-a"/>
<text x="395.3" y="394" text-anchor="middle" class="fig-small">money</text>
<text x="385.3" y="374" text-anchor="middle" class="fig-small fig-bold">0</text>
<rect x="420.7" y="380.0" width="17" height="0.0" class="fill-soft-a stroke-a"/>
<rect x="439.7" y="362.1" width="17" height="17.9" class="fill-a"/>
<text x="438.7" y="394" text-anchor="middle" class="fig-small">win</text>
<text x="428.7" y="374" text-anchor="middle" class="fig-small fig-bold">0</text>
<rect x="464.0" y="286.2" width="17" height="93.8" class="fill-soft-a stroke-a"/>
<rect x="483.0" y="308.6" width="17" height="71.4" class="fill-a"/>
<text x="482.0" y="394" text-anchor="middle" class="fig-small">meeting</text>
<rect x="507.3" y="286.2" width="17" height="93.8" class="fill-soft-a stroke-a"/>
<rect x="526.3" y="308.6" width="17" height="71.4" class="fill-a"/>
<text x="525.3" y="394" text-anchor="middle" class="fig-small">report</text>
<rect x="550.7" y="348.8" width="17" height="31.2" class="fill-soft-a stroke-a"/>
<rect x="569.7" y="344.3" width="17" height="35.7" class="fill-a"/>
<text x="568.7" y="394" text-anchor="middle" class="fig-small">lunch</text>
<text x="30" y="416" class="fig-small">Outlined bars: counts divided by the class total (alpha = 0). Solid bars: add 1 to every count first (alpha = 1).</text>
</svg>
Figure 3. Four toy documents and their word counts, with the counts summed per class. The outlined bars show raw frequencies, where spam never uses meeting or lunch and ham never uses money or win, so those bars are exactly zero. The solid bars add one to every count, so no word has probability zero.
```

Bayes' rule needs the density of a document given its class, and estimating that for every combination of words is hopeless. *Naive Bayes* makes it easy with one assumption: once you know the class, the words are independent. If you are told an email is spam, whether it contains "free" says nothing about whether it contains "money". Under that assumption, the density of the whole document is the product of one small probability per word, and its log is a sum.

The assumption is naive because it is false. Suppose "free" and "money" always appear together. "Free" appears in half of spam and a tenth of ham, so it is evidence of 5 to 1 for spam. "Money" appears exactly when "free" does, so it adds no further information, but naive Bayes multiplies in another 5 to 1 and claims 25 to 1. With equal priors it reports spam at 0.96, where the truth is 0.83.

The error is in the confidence, not the choice. Both words point toward spam, so the double count makes the winner more certain and rarely changes which class wins. That is why naive Bayes classifies well even when its probabilities are too sharp. Figure 4 shows the same assumption as a shape.

```figure
<svg viewBox="0 0 620 340" role="img" aria-label="A tilted ellipse of correlated points with an axis-aligned ellipse inside it, and a corner point the independent model finds ordinary">
<polygon points="370.0,74.0 369.8,69.6 369.0,65.7 367.8,62.1 366.2,59.0 364.0,56.3 361.4,54.1 358.4,52.4 354.9,51.1 351.0,50.3 346.6,50.0 341.9,50.2 336.8,50.8 331.4,52.0 325.7,53.6 319.6,55.7 313.3,58.2 306.7,61.2 299.8,64.6 292.8,68.5 285.6,72.7 278.3,77.4 270.8,82.4 263.3,87.8 255.7,93.5 248.1,99.5 240.5,105.8 232.9,112.4 225.4,119.2 218.0,126.2 210.8,133.4 203.6,140.7 196.7,148.1 190.0,155.6 183.5,163.2 177.3,170.8 171.4,178.5 165.8,186.0 160.6,193.5 155.7,201.0 151.2,208.2 147.0,215.4 143.3,222.3 140.1,229.1 137.2,235.6 134.9,241.8 132.9,247.8 131.5,253.4 130.5,258.7 130.1,263.7 130.1,268.2 130.5,272.4 131.5,276.2 132.9,279.5 134.9,282.4 137.2,284.8 140.1,286.8 143.3,288.3 147.0,289.4 151.2,289.9 155.7,290.0 160.6,289.6 165.8,288.7 171.4,287.3 177.3,285.4 183.5,283.1 190.0,280.4 196.7,277.1 203.6,273.5 210.8,269.4 218.0,265.0 225.4,260.1 232.9,254.9 240.5,249.4 248.1,243.5 255.7,237.4 263.3,230.9 270.8,224.2 278.3,217.3 285.6,210.2 292.8,203.0 299.8,195.6 306.7,188.1 313.3,180.6 319.6,173.0 325.7,165.3 331.4,157.8 336.8,150.2 341.9,142.7 346.6,135.4 351.0,128.2 354.9,121.1 358.4,114.3 361.4,107.6 364.0,101.3 366.2,95.2 367.8,89.4 369.0,83.9 369.8,78.8 370.0,74.0" fill="none" class="stroke-a" stroke-width="2"/>
<polygon points="370.0,170.0 369.8,162.4 369.0,154.8 367.8,147.3 366.2,139.9 364.0,132.6 361.4,125.4 358.4,118.4 354.9,111.7 351.0,105.1 346.6,98.9 341.9,92.9 336.8,87.2 331.4,81.8 325.7,76.9 319.6,72.3 313.3,68.0 306.7,64.2 299.8,60.8 292.8,57.9 285.6,55.4 278.3,53.4 270.8,51.8 263.3,50.7 255.7,50.1 248.1,50.0 240.5,50.4 232.9,51.2 225.4,52.5 218.0,54.3 210.8,56.6 203.6,59.3 196.7,62.5 190.0,66.1 183.5,70.1 177.3,74.5 171.4,79.3 165.8,84.5 160.6,90.0 155.7,95.8 151.2,102.0 147.0,108.4 143.3,115.0 140.1,121.9 137.2,129.0 134.9,136.2 132.9,143.6 131.5,151.0 130.5,158.6 130.1,166.2 130.1,173.8 130.5,181.4 131.5,189.0 132.9,196.4 134.9,203.8 137.2,211.0 140.1,218.1 143.3,225.0 147.0,231.6 151.2,238.0 155.7,244.2 160.6,250.0 165.8,255.5 171.4,260.7 177.3,265.5 183.5,269.9 190.0,273.9 196.7,277.5 203.6,280.7 210.8,283.4 218.0,285.7 225.4,287.5 232.9,288.8 240.5,289.6 248.1,290.0 255.7,289.9 263.3,289.3 270.8,288.2 278.3,286.6 285.6,284.6 292.8,282.1 299.8,279.2 306.7,275.8 313.3,272.0 319.6,267.7 325.7,263.1 331.4,258.2 336.8,252.8 341.9,247.1 346.6,241.1 351.0,234.9 354.9,228.3 358.4,221.6 361.4,214.6 364.0,207.4 366.2,200.1 367.8,192.7 369.0,185.2 369.8,177.6 370.0,170.0" fill="none" class="stroke-b dashed" stroke-width="2"/>
<circle cx="249.2" cy="190.7" r="3" class="fill-a"/>
<circle cx="319.8" cy="192.8" r="3" class="fill-a"/>
<circle cx="125.8" cy="250.7" r="3" class="fill-a"/>
<circle cx="253.8" cy="136.8" r="3" class="fill-a"/>
<circle cx="244.5" cy="196.5" r="3" class="fill-a"/>
<circle cx="200.2" cy="231.5" r="3" class="fill-a"/>
<circle cx="283.7" cy="166.3" r="3" class="fill-a"/>
<circle cx="226.0" cy="197.8" r="3" class="fill-a"/>
<circle cx="230.5" cy="212.6" r="3" class="fill-a"/>
<circle cx="259.7" cy="194.1" r="3" class="fill-a"/>
<circle cx="198.5" cy="214.3" r="3" class="fill-a"/>
<circle cx="223.4" cy="181.0" r="3" class="fill-a"/>
<circle cx="292.7" cy="97.6" r="3" class="fill-a"/>
<circle cx="164.0" cy="318.2" r="3" class="fill-a"/>
<circle cx="377.0" cy="100.1" r="3" class="fill-a"/>
<circle cx="199.7" cy="215.5" r="3" class="fill-a"/>
<circle cx="174.9" cy="217.7" r="3" class="fill-a"/>
<circle cx="232.6" cy="176.6" r="3" class="fill-a"/>
<circle cx="243.2" cy="143.9" r="3" class="fill-a"/>
<circle cx="322.3" cy="113.7" r="3" class="fill-a"/>
<circle cx="202.0" cy="149.6" r="3" class="fill-a"/>
<circle cx="314.0" cy="147.0" r="3" class="fill-a"/>
<circle cx="263.7" cy="119.5" r="3" class="fill-a"/>
<circle cx="238.2" cy="131.5" r="3" class="fill-a"/>
<circle cx="335.2" cy="42.0" r="3" class="fill-a"/>
<circle cx="218.0" cy="255.8" r="3" class="fill-a"/>
<circle cx="218.2" cy="155.7" r="3" class="fill-a"/>
<circle cx="226.0" cy="156.0" r="3" class="fill-a"/>
<circle cx="109.6" cy="300.0" r="3" class="fill-a"/>
<circle cx="280.6" cy="168.7" r="3" class="fill-a"/>
<circle cx="216.8" cy="210.6" r="3" class="fill-a"/>
<circle cx="262.2" cy="161.8" r="3" class="fill-a"/>
<circle cx="233.2" cy="227.2" r="3" class="fill-a"/>
<circle cx="383.3" cy="129.1" r="3" class="fill-a"/>
<circle cx="180.4" cy="236.8" r="3" class="fill-a"/>
<circle cx="164.2" cy="256.1" r="3" class="fill-a"/>
<circle cx="283.2" cy="118.7" r="3" class="fill-a"/>
<circle cx="278.2" cy="189.4" r="3" class="fill-a"/>
<circle cx="266.9" cy="86.1" r="3" class="fill-a"/>
<circle cx="221.9" cy="182.3" r="3" class="fill-a"/>
<circle cx="278.8" cy="167.3" r="3" class="fill-a"/>
<circle cx="201.2" cy="222.7" r="3" class="fill-a"/>
<circle cx="295.8" cy="129.3" r="3" class="fill-a"/>
<circle cx="297.9" cy="114.8" r="3" class="fill-a"/>
<circle cx="201.6" cy="250.1" r="3" class="fill-a"/>
<circle cx="181.4" cy="264.0" r="3" class="fill-a"/>
<circle cx="257.1" cy="194.6" r="3" class="fill-a"/>
<circle cx="261.8" cy="156.5" r="3" class="fill-a"/>
<circle cx="288.1" cy="158.6" r="3" class="fill-a"/>
<circle cx="304.2" cy="147.0" r="3" class="fill-a"/>
<circle cx="344.4" cy="85.6" r="3" class="fill-a"/>
<circle cx="209.9" cy="175.6" r="3" class="fill-a"/>
<circle cx="166.5" cy="160.2" r="3" class="fill-a"/>
<circle cx="240.5" cy="196.8" r="3" class="fill-a"/>
<circle cx="163.3" cy="296.4" r="3" class="fill-a"/>
<circle cx="213.0" cy="243.2" r="3" class="fill-a"/>
<circle cx="267.2" cy="227.5" r="3" class="fill-a"/>
<circle cx="201.8" cy="224.2" r="3" class="fill-a"/>
<circle cx="273.2" cy="83.1" r="3" class="fill-a"/>
<circle cx="205.6" cy="212.7" r="3" class="fill-a"/>
<path d="M328.0,240.0 L335.0,254.0 L321.0,254.0 z" class="fill-ink"/>
<text x="342.0" y="253.0" class="fig-bold">corner point</text>
<text x="470" y="40" class="fig-bold">One class, two features</text>
<line x1="470" y1="58" x2="500" y2="58" class="stroke-a" stroke-width="2"/><text x="508" y="62" class="fig-small">true cloud</text>
<line x1="470" y1="78" x2="500" y2="78" class="stroke-b dashed" stroke-width="2"/><text x="508" y="82" class="fig-small">independent model</text>
<text x="470" y="120" class="fig-small">Corner point,</text><text x="470" y="136" class="fig-small">squared distance:</text>
<text x="470" y="156" class="fig-small">independent model: 3.4</text>
<text x="470" y="172" class="fig-small">true cloud: 16.9</text>
</svg>
Figure 4. Two features that rise and fall together (correlation 0.8) make a tilted cloud. Independence replaces it with an upright ellipse (here a circle) that has the same widths. The corner point is rare in the true cloud, at squared distance 16.9, but sits inside the independent model's circle at 3.4, because that model has forgotten the tilt. Both outlines mark squared distance 4 from the centre in their own model.
```

Two features that rise and fall together make a tilted cloud. Ignoring how they move together loses the tilt and keeps only each feature's own width, which is an upright ellipse. The corner point sits where one feature is high and the other low, so it is rare in the real cloud, but the independent model finds it ordinary. This is the same double counting from the point's side: each feature alone looks unremarkable, and only the pair is unusual.

## Gaussian naive Bayes: one bell per feature

For measured features, give each class and each feature its own one-dimensional bell. The class mean is the average of that feature over the class, and the variance is the average squared gap from that mean, over that class's own rows:

$$
\mu_{kj} = \frac{1}{n_k}\sum_{i:\, y_i = k} x_{ij}, \qquad
\sigma^2_{kj} = \frac{1}{n_k}\sum_{i:\, y_i = k} (x_{ij} - \mu_{kj})^2.
$$

Read it aloud: for class $k$ and feature $j$, the centre is the class average, and the width is the average squared gap from that centre. Take the class with values 1 and 3 on one feature. The mean is 2, the gaps are $-1$ and $+1$, and the squared gaps add to 2. Divide by the class count 2 to get variance 1. Dividing by all four rows would give 0.5, which is wrong here. Naive Bayes keeps a separate variance for every class, so each class divides by its own count, unlike the pooled covariance of GDA.

A feature that never changes within a class has variance zero, and the density would divide by zero. So you set a *floor* under every variance. Choose it relative to the other variances, because a floor as tiny as $10^{-9}$ makes any change in that feature look impossible, which is another veto. To classify, add up one log density per feature, and add the log prior:

$$
\log p(x \mid y = k) = \sum_{j=1}^{d} \log \mathcal N(x_j;\, \mu_{kj},\, \sigma^2_{kj}).
$$

## Word counts and smoothing

For counts, give each class a probability for each word, written $\theta_{kj}$ (the Greek letter theta): the chance that a word drawn at random from a class-$k$ document is word $j$. The natural estimate is the share of the class's words that are word $j$. In the toy set, spam uses 8 words in total and "free" 3 times, so $\theta = 3/8$. Scoring a document takes the sum, over words, of the count times the log probability, plus the log prior. A word seen three times votes three times, because each appearance is a separate draw.

Here is the trap. Spam never uses "meeting" in the toy set, so its estimated probability is 0, and ham never uses "money". Take the document "free free money meeting". Spam would score $\log 0 = -\infty$ because of "meeting", and ham $-\infty$ because of "money". One unseen word vetoes a class, whatever the other words say, and here both classes are vetoed, so there is nothing to compare.

Adding a tiny $\epsilon$ to only the zeros does not help, because $\log\epsilon$ is a huge negative number, so a single unseen word still decides the document alone. The better fix says something true: eight words of evidence are not enough to call a word impossible. *Laplace smoothing* pretends you saw every word an extra $\alpha$ times:

$$
\theta_{kj} = \frac{c_{kj} + \alpha}{\sum_{j'} c_{kj'} + \alpha V}.
$$

Here $c_{kj}$ is the count of word $j$ in class $k$, and the denominator adds $\alpha$ once per word, so the probabilities still add to 1. With three words and $\alpha = 1$, counts $[2, 0, 1]$ become $[3, 1, 2]$, which add to 6, so $\theta = [1/2, 1/6, 1/3]$. In the toy set, spam's total of 8 plus 6 makes 14, so "meeting" gets $1/14$ and "free" drops from $3/8$ to $4/14$. A bigger $\alpha$ trusts the counts less, and $\alpha = 1$ is the usual start, which you can tune with cross-validation.

Now score "free free money meeting", with counts $[2, 1, 0, 1, 0, 0]$. Spam's score is $2\log\tfrac{4}{14} + \log\tfrac{4}{14} + \log\tfrac{1}{14} + \log 0.5 = -7.090$, and ham's is $-8.477$. They differ by $\log 4$, so spam is four times more likely than ham. The log-sum-exp is $-6.867$, so the posteriors are $e^{-0.223} = 0.80$ and $e^{-1.609} = 0.20$. In numpy, with `X` of counts shaped `(n, V)`:

```python
joint = X @ log_theta.T + log_priors          # shape (n, K)
log_post = joint - log_sum_exp(joint, axis=1)[:, None]
```

> [!WARNING]
> Without smoothing, one unseen word sends a class to $-\infty$. Smooth the counts of each class after summing its rows, so $\alpha$ is added once per word and not once per document.

> [!KEY]
> Pretending features are independent given the class turns the class density into a product of per-feature terms, so you store one number per feature and class instead of a covariance matrix. The pretence makes the probabilities too confident, but the ranking is often still right. Laplace smoothing keeps an unseen word from vetoing a class.

> [!RECAP]
> - The log odds of two Gaussian classes is $\log(\pi_1/\pi_0) + \log p(x \mid 1) - \log p(x \mid 0)$. A shared covariance cancels the quadratic term, so the posterior is $\sigma(w^{\top}x + b)$ with $w = \Sigma^{-1}(\mu_1 - \mu_0)$.
> - $b = \log(\pi_1/\pi_0) - \tfrac12(\mu_0 + \mu_1)^{\top}w$. GDA needs fewer rows when its assumption holds, and outliers damage it more than they damage logistic regression.
> - Naive Bayes assumes independence given the class, so the log density is a sum over features. It is overconfident when features are correlated, because correlated evidence is counted twice.
> - Gaussian naive Bayes fits a mean and a variance per class and feature, dividing by the class count, with a floor under every variance.
> - Word models give each class a probability per word, add $\alpha$ to every count, and score by count times log probability.

## Build it step by step

The helpers from the [previous chapter](/lessons/016-bayes-rule-and-gaussian-classes), `sigmoid`, `gaussian_nb_log_posterior`, and the other fitted-model functions are already in `starter.py`. You write six functions, in the order the checks test them.

1. **`gda_to_logistic(means, cov, priors)`**. Takes `means` of shape `(2, d)`, `cov` `(d, d)`, and `priors` `(2,)`, and returns `w` of shape `(d,)` and `b` as a Python float. Use `np.linalg.solve(cov, means[1] - means[0])` rather than an inverse. The usual mistake is reversing the means, which flips the sign of `w`.
2. **`log_odds(X, means, cov, priors)`**. Takes points `X` of shape `(n, d)` and returns `(n,)`: column 1 of `gda_log_posterior` minus column 0. The usual mistake is dividing the posteriors and forgetting the log.
3. **`decision_boundary_1d(means, var, priors)`**. Takes two one-dimensional means and a shared variance, and returns the $x$ where the log odds is zero. In one dimension $w = (\mu_1 - \mu_0)/\sigma^2$, and the boundary is $x = -b/w$. The usual mistake is passing a standard deviation where the variance goes.
4. **`fit_naive_bayes_gaussian(X, y, var_floor)`**. Takes `X` of shape `(n, d)` and integer labels. Returns `means`, `variances` (both `(K, d)`), and `priors` `(K,)`. Use `X[y == k].var(axis=0)` and `np.maximum` with the floor. The usual mistake is `ddof=1`, which divides by the class count minus one.
5. **`fit_naive_bayes_multinomial(X, y, alpha)`**. Takes counts `X` of shape `(n, V)`. Returns `log_theta` `(K, V)` and `log_priors` `(K,)`. Sum the class rows, add `alpha`, divide by the smoothed total, and take the log. The usual mistake is smoothing each row before summing.
6. **`multinomial_nb_log_posterior(X, log_theta, log_priors)`**. Returns `(n, K)`. Compute `X @ log_theta.T + log_priors`, then subtract the row log-sum-exp as a column. The usual mistake is passing the priors where the log priors go.

## Cost and scaling

Times use $n$ rows, $d$ features, $K$ classes, and a vocabulary of $V$ words:

| Model | Fit time | Memory |
|---|---|---|
| GDA with a shared covariance | grows like $n d^2$ | $Kd + d^2$ |
| Gaussian naive Bayes | grows like $n d$ | $2Kd$ |
| Word-count naive Bayes | grows like the nonzero counts | $KV$ |

At $V = 10{,}000$, a covariance holds $10^8$ numbers, and a two-class word model holds $2 \times 10^4$.

## Where this breaks

**A wrong shape assumption.** If the classes have different shapes, the cancellation fails and the true boundary curves. A straight GDA boundary is then biased, and logistic regression, which only assumes the line, will usually match or beat it with enough rows.

**Outliers, skew, and copies.** A few far points move GDA's averages and naive Bayes's means and variances, so check the largest Mahalanobis distances before you trust a fit. A skewed measurement breaks the Gaussian model, so take a log or use a count model. A duplicated column is counted twice, as with "free" and "money", and the probabilities get sharper, so drop near-duplicates first.

## The exercise

You implement `gda_to_logistic`, `log_odds`, `decision_boundary_1d`, `fit_naive_bayes_gaussian`, `fit_naive_bayes_multinomial`, and `multinomial_nb_log_posterior`. The starter's closing programs read a logistic regression off the four-point example and a tilted pair of clouds, then fit the toy spam set and two Gaussian clouds.

The checks confirm $w = 4$ and $b = -16$ on the hand example, that $\sigma(w^{\top}x + b)$ reproduces GDA's class 1 probability with unequal priors, that $w$ equals $\Sigma^{-1}$ times the mean difference, and that the boundary sits at $x = 4$ for equal priors and moves toward the rarer class otherwise. They check the Gaussian fit, including variance 1 on the hand example and a floor under a constant feature, the smoothed word probabilities, and that "free free money meeting" scores 0.80 for spam. They also check that a repeated word votes twice and that a 90/10 prior shifts the odds by $\log 9$.
