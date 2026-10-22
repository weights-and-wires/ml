---
title: Why two Gaussians give a straight boundary
slug: 030-gda-is-logistic-regression
part: "Part 2 — Classical models"
summary: Two Gaussian classes that share a covariance have a log odds that is a straight line, so GDA is a logistic regression with closed-form weights.
minutes: 40
gpu: false
objectives:
  - Derive the log odds of two Gaussian classes with a shared covariance and show that the quadratic terms cancel.
  - Compute the equivalent logistic weights and bias from means, covariance, and priors, and read what they mean.
  - Explain when fitting the classes beats fitting the boundary directly, and when it loses.
prereqs:
  - 017-sigmoid-and-probability
  - 019-gradient-descent-for-logistic
  - 028-bayes-rule-classifier
  - 029-gaussian-discriminant-analysis
exercise: 030-gda-is-logistic-regression
---

# Why two Gaussians give a straight boundary

You have two ways to separate two kinds of parcel. One is to fit a bell cloud to each kind and ask which cloud an unseen parcel belongs to, as the previous chapter did. The other is to fit a line directly with logistic regression. Draw the boundaries and they look alike: both are straight. That is odd, because the first method never draws a line. It only describes clouds.

This chapter explains the coincidence, which is not one. When two Gaussian clouds share a shape, the rule that the cloud method follows is a logistic regression, down to the formula. The only difference is how the weights are found. Logistic regression walks downhill to find them. The cloud method reads them off the averages.

Seeing this settles a practical question. If both give the same kind of line, you can choose between them by how they are fitted, how much data they need, and what breaks them. The last section shows pictures of exactly that.

## What you already know

- A logistic regression scores a point with $w^{\top}x + b$ and turns the score into a probability with the sigmoid, as in *From a score to a probability*.
- A score of 0 means a probability of 0.5, so the boundary is where the score is 0.
- GDA fits a mean per class, one shared covariance $\Sigma$, and class priors, as in *Fit a Gaussian cloud to each class*.
- GDA's posterior is the prior times the density, divided by the total, from *Classify by asking who made this*.
- The log of a ratio is a difference of logs, and $e^{a}e^{-b} = e^{a-b}$.

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

The top of Figure 1 shows the bells. The bottom shows a different quantity, the score $4x - 16$, which is the log odds of class 1 against class 0. It is a straight line, and it crosses zero at $x = 4$, the same place. The rest of the chapter shows why that line exists and where 4 and $-16$ come from.

## The log odds

Why compare through the *odds* of class 1 against class 0, the ratio $p(1 \mid x) / p(0 \mid x)$? Because Bayes' rule gives both posteriors the same denominator, the total $p(x)$, and a ratio makes it cancel. The log of the ratio is the *log odds*, and it is exactly what a sigmoid turns into a probability: if the log odds is $z$, then $p(1 \mid x) = e^{z}/(1 + e^{z}) = \sigma(z)$.

Write out the log odds from Bayes' rule, taking the log of prior times density for each class:

$$
\log\frac{p(1 \mid x)}{p(0 \mid x)} = \log\frac{\pi_1}{\pi_0} + \log p(x \mid y = 1) - \log p(x \mid y = 0).
$$

Substitute the Gaussian log density from the previous chapter. The constant $-\tfrac d2\log 2\pi$ and the term $-\tfrac12\log\det\Sigma$ are the same for both classes, because the covariance is shared, so they cancel in the difference. What remains is the distance term. Expand it for a class with mean $\mu$:

$$
-\tfrac12 (x-\mu)^{\top}\Sigma^{-1}(x-\mu) = -\tfrac12 x^{\top}\Sigma^{-1}x + \mu^{\top}\Sigma^{-1}x - \tfrac12\mu^{\top}\Sigma^{-1}\mu.
$$

The first piece, $-\tfrac12 x^{\top}\Sigma^{-1}x$, is quadratic in $x$, the only part that would make a curve. It is the same for both classes, because $\Sigma$ is shared, so it cancels too. Subtracting the class 0 version from the class 1 version leaves a score that is linear in $x$:

$$
\log\frac{p(1 \mid x)}{p(0 \mid x)} = w^{\top}x + b, \qquad
w = \Sigma^{-1}(\mu_1 - \mu_0), \qquad
b = \log\frac{\pi_1}{\pi_0} - \tfrac12(\mu_0 + \mu_1)^{\top}w.
$$

So $p(1 \mid x) = \sigma(w^{\top}x + b)$, which is the model logistic regression fits. This is the whole result: a shared covariance makes the quadratic pieces cancel, and a straight score is what is left.

## Read the weights aloud

The weights $w$ are the difference between the class means, re-weighted by the inverse shape. The difference of means points from class 0 to class 1, which is the direction in which moving the point is the most evidence for class 1. The inverse shape then shrinks directions where the cloud is wide, because a point is often far along them by chance, and boosts directions where it is narrow.

The bias is the prior plus a midpoint term. Without priors, $b = -w^{\top}m$ with $m$ the midpoint of the two means, so the score is $w^{\top}(x - m)$ and the boundary passes through the midpoint. A log prior ratio shifts it. When class 1 is rarer, $\log(\pi_1/\pi_0)$ is negative, so the score drops and class 1 needs stronger evidence, as in the earlier chapter's slide of the boundary.

On the four points, $\mu_1 - \mu_0 = 4$ and $\Sigma = 1$, so $w = 4$. The midpoint is 4, so $b = 0 - 4 \times 4 = -16$. At $x = 3$ the score is $4 \times 3 - 16 = -4$, and $\sigma(-4) = 0.018$, which is the probability of class 1 that the last chapter found by comparing densities. At $x = 4$ the score is 0 and the probability is 0.5.

## Why $w$ is not the mean difference

In one dimension, the shape only rescales. In two dimensions it turns the weights.

```figure
<svg viewBox="0 0 620 360" role="img" aria-label="Two classes of points, the GDA decision line, the weight vector w and the mean difference arrow, which point in different directions">
<defs><marker id="c030-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c030-arrow2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<defs><clipPath id="c030-clip"><rect x="20" y="20" width="580" height="300"/></clipPath></defs>
<g clip-path="url(#c030-clip)">
<line x1="7.8" y1="418.0" x2="375.4" y2="6.3" class="stroke-ink" stroke-width="2"/>
<line x1="7.2" y1="416.4" x2="368.5" y2="-0.9" class="stroke-muted dashed" stroke-width="2"/>
</g>
<circle cx="144.6" cy="169.7" r="4" class="fill-a"/>
<circle cx="180.5" cy="176.2" r="4" class="fill-a"/>
<circle cx="192.0" cy="168.4" r="4" class="fill-a"/>
<circle cx="122.9" cy="154.3" r="4" class="fill-a"/>
<circle cx="187.4" cy="159.7" r="4" class="fill-a"/>
<circle cx="117.5" cy="194.8" r="4" class="fill-a"/>
<circle cx="160.9" cy="197.2" r="4" class="fill-a"/>
<circle cx="139.2" cy="161.0" r="4" class="fill-a"/>
<circle cx="230.0" cy="111.3" r="4" class="fill-a"/>
<circle cx="274.6" cy="96.4" r="4" class="fill-a"/>
<circle cx="252.6" cy="80.8" r="4" class="fill-a"/>
<circle cx="212.9" cy="102.4" r="4" class="fill-a"/>
<circle cx="145.0" cy="186.7" r="4" class="fill-a"/>
<circle cx="294.1" cy="50.1" r="4" class="fill-a"/>
<circle cx="150.6" cy="170.4" r="4" class="fill-a"/>
<circle cx="240.3" cy="101.7" r="4" class="fill-a"/>
<circle cx="216.7" cy="137.1" r="4" class="fill-a"/>
<circle cx="107.7" cy="246.1" r="4" class="fill-a"/>
<circle cx="136.0" cy="157.5" r="4" class="fill-a"/>
<circle cx="183.2" cy="145.8" r="4" class="fill-a"/>
<circle cx="281.0" cy="179.8" r="4" class="fill-b"/>
<circle cx="352.1" cy="108.2" r="4" class="fill-b"/>
<circle cx="242.9" cy="275.2" r="4" class="fill-b"/>
<circle cx="239.9" cy="218.8" r="4" class="fill-b"/>
<circle cx="286.6" cy="105.1" r="4" class="fill-b"/>
<circle cx="268.6" cy="237.1" r="4" class="fill-b"/>
<circle cx="273.7" cy="168.7" r="4" class="fill-b"/>
<circle cx="285.9" cy="152.7" r="4" class="fill-b"/>
<circle cx="279.7" cy="159.6" r="4" class="fill-b"/>
<circle cx="223.2" cy="263.9" r="4" class="fill-b"/>
<circle cx="285.4" cy="194.1" r="4" class="fill-b"/>
<circle cx="302.3" cy="203.0" r="4" class="fill-b"/>
<circle cx="322.5" cy="147.5" r="4" class="fill-b"/>
<circle cx="219.6" cy="202.6" r="4" class="fill-b"/>
<circle cx="372.9" cy="118.4" r="4" class="fill-b"/>
<circle cx="288.9" cy="245.1" r="4" class="fill-b"/>
<circle cx="314.5" cy="152.0" r="4" class="fill-b"/>
<circle cx="208.5" cy="229.9" r="4" class="fill-b"/>
<circle cx="312.1" cy="164.1" r="4" class="fill-b"/>
<circle cx="274.2" cy="134.5" r="4" class="fill-b"/>
<path d="M177.4,148.4 L191.4,148.4 M184.4,141.4 L184.4,155.4" class="stroke-ink" stroke-width="3"/>
<path d="M274.7,183.0 L288.7,183.0 M281.7,176.0 L281.7,190.0" class="stroke-ink" stroke-width="3"/>
<circle cx="233.1" cy="165.7" r="4" class="fill-ink"/>
<line x1="233.1" y1="165.7" x2="315.4" y2="195.0" class="stroke-muted" stroke-width="2.5" marker-end="url(#c030-arrow2)"/>
<line x1="233.1" y1="165.7" x2="298.3" y2="223.9" class="stroke-b" stroke-width="2.5" marker-end="url(#c030-arrow)"/>
<text x="470" y="60" class="fig-bold">Legend</text>
<line x1="470" y1="78" x2="500" y2="78" class="stroke-ink" stroke-width="2"/><text x="508" y="82" class="fig-small">GDA boundary</text>
<line x1="470" y1="98" x2="500" y2="98" class="stroke-muted dashed" stroke-width="2"/><text x="508" y="102" class="fig-small">logistic fit</text>
<line x1="470" y1="118" x2="500" y2="118" class="stroke-muted" stroke-width="2.5"/><text x="508" y="122" class="fig-small">mean difference</text>
<line x1="470" y1="138" x2="500" y2="138" class="stroke-b" stroke-width="2.5"/><text x="508" y="142" class="fig-small">weights w</text>
<text x="470" y="168" class="fig-small">Crosses: class means.</text>
</svg>
Figure 2. The solid line is GDA's boundary, and the arrow w stands square to it. The grey arrow is the difference between the two class means, drawn from the midpoint between them. The two arrows differ because the clouds lean: the inverse of Sigma tips the weights away from the means' direction. The dashed line is a logistic regression fitted to the same 40 points, and it lands close by. Everything is estimated from those 40 points, so the arrows differ slightly from the exact values (3, 0) and (5.56, -4.44) in the text.
```

Figure 2 shows two classes whose clouds lean up and to the right, with $\Sigma = \begin{bmatrix} 1.5 & 1.2 \\ 1.2 & 1.5 \end{bmatrix}$. The class means differ by $(3, 0)$, shown by the grey arrow. The inverse shape turns it into $w = \Sigma^{-1}(3, 0) = (5.56, -4.44)$, which points down and to the right, the orange arrow, and the boundary stands square to it. The reason is the lean. A high second feature usually means the point sits far along the lean, which also raises the first feature, so part of a high first feature is noise. The weights give the second feature a negative sign to subtract that part out.

## Same family, different fit

Both methods produce $\sigma(w^{\top}x + b)$, and they pick $w$ and $b$ by different goals. Logistic regression maximises how well the line predicts the labels, which is why it needs a loop. GDA maximises how well the clouds explain the points and the labels together, which is why it needs only averages and one solve. The dashed line in Figure 2 is a logistic regression fitted by gradient descent to the same points, and it lands close to GDA's line.

The shared covariance is what keeps the boundary straight. If each class has its own covariance, the quadratic pieces differ and no longer cancel, so the boundary curves. That model is *quadratic discriminant analysis*, and it needs more data to fit the extra shapes.

## When fitting the clouds pays

Why would you ever fit the clouds if the line is all you need? Two reasons, one for each side. With the Gaussian assumption true, the closed form wastes less data. With it false, the closed form is fragile. Figure 3 measures both on simulated three-feature Gaussian data.

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
Figure 3. Orange is GDA and blue is logistic regression, averaged over 400 random training sets per size. On clean data GDA is ahead at every size and the gap closes as rows grow: GDA at 20 rows (0.938) matches logistic regression at 40 (0.939). With three class 1 points 20 units beyond their class mean, GDA drops to 0.67 at 10 rows and 0.89 at 320, while logistic regression does not change.
```

On clean data, GDA is ahead at every size, by about one percentage point with 10 to 20 rows, and the gap closes as the rows grow. GDA at 20 rows scores 0.938, about what logistic regression scores at 40 rows. The gain is modest, and it is real only because the data really are Gaussian with a shared shape.

On the right, three class 1 points sit 20 units beyond their class mean, on the far side from class 0. Logistic regression does not move at all, and its curve is the same as on the left. GDA falls to 0.67 at 10 rows and still reaches only 0.89 at 320. The reason is in what each method uses. GDA's means and covariance are averages over every point, so a far point drags the mean and swells the covariance, which tilts the line. A logistic gradient for each point is $p - y$, the prediction minus the label, so a point that the model already classifies correctly and confidently contributes almost nothing. These outliers are of that kind. A far point on the wrong side would hurt logistic regression too, so the figure shows what each method reads from the data, not that one is always safer.

> [!KEY]
> Back to the opening question. With two classes and one shared covariance, the quadratic terms cancel in the log odds, so GDA's boundary is a straight logistic-regression line, with weights $\Sigma^{-1}(\mu_1 - \mu_0)$ that come from averages and not from descent. The two differ in what they trust: GDA trusts the Gaussian shape, so it needs fewer rows but is hurt by outliers, and logistic regression trusts only the labels.

> [!RECAP]
> - The log odds of two Gaussian classes is $\log(\pi_1/\pi_0) + \log p(x \mid 1) - \log p(x \mid 0)$, and the evidence $p(x)$ cancels.
> - A shared covariance cancels the quadratic term, so the log odds is $w^{\top}x + b$ and the posterior is a sigmoid.
> - $w = \Sigma^{-1}(\mu_1 - \mu_0)$ and $b = \log(\pi_1/\pi_0) - \tfrac12(\mu_0 + \mu_1)^{\top}w$.
> - Separate covariances leave a quadratic term and a curved boundary.
> - GDA needs fewer rows when its assumption holds, and outliers damage it more than they damage logistic regression.

## Build it step by step

The earlier functions and `sigmoid` are already in `starter.py`. You write three more, in this order.

1. **`gda_to_logistic(means, cov, priors)`**. Takes `means` of shape `(2, d)`, `cov` `(d, d)`, and `priors` `(2,)`, and returns `w` of shape `(d,)` and `b` as a Python float. Use `np.linalg.solve(cov, means[1] - means[0])` rather than an inverse. The usual mistake is reversing the means, which flips the sign of `w`.
2. **`log_odds(X, means, cov, priors)`**. Takes points `X` of shape `(n, d)` and returns `(n,)`: column 1 of `gda_log_posterior` minus column 0. The usual mistake is dividing the posteriors and forgetting the log.
3. **`decision_boundary_1d(means, var, priors)`**. Takes two one-dimensional means and a shared variance, and returns the $x$ where the log odds is zero. In one dimension $w = (\mu_1 - \mu_0)/\sigma^2$, and the boundary is $x = -b/w$. The usual mistake is passing a standard deviation where the variance goes.

## Where this breaks

**A wrong shape assumption.** If the classes have different shapes, the cancellation fails and the true boundary curves. A straight GDA boundary is then biased, and logistic regression, which only assumes the line, will usually match or beat it with enough rows.

**Outliers.** As Figure 3 shows, a few far points move GDA's averages. Check the largest Mahalanobis distances before you trust the fit.

## The exercise

You implement `gda_to_logistic`, `log_odds`, and `decision_boundary_1d`. The checks confirm $w = 4$ and $b = -16$ on the hand example, and that $\sigma(w^{\top}x + b)$ reproduces GDA's class 1 probability when the priors are unequal. They check that $w$ equals $\Sigma^{-1}$ times the mean difference and not the plain difference, that `log_odds` is a straight line in $x$, and that the boundary sits at $x = 4$ for equal priors and moves toward the rarer class otherwise.
