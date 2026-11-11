---
title: "Gaussian mixtures and EM"
slug: 031-gaussian-mixtures-and-em
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: Give every point a share in every Gaussian blob, refit the blobs from the shares, and alternate. The likelihood never falls.
minutes: 80
gpu: false
objectives:
  - Describe a Gaussian mixture as a weighted die that picks a blob, followed by a draw from that blob's bell.
  - Compute a point's responsibilities with Bayes' rule, and compute the log density of a Gaussian with a full covariance.
  - Explain why the likelihood is handled in logs, and compute a row-wise log-sum-exp that survives scores near -1000.
  - Refit a mixture's weights, means, and covariances as share-weighted versions of the ordinary mean and covariance.
  - Explain why one E-step followed by one M-step can never lower the log-likelihood, and why tiny equal round blobs turn EM into k-means.
prereqs:
  - 017-bayes-rule-and-gaussian-classes
  - 030-k-means
  - 029-power-iteration-and-probabilistic-pca
exercise: 031-gaussian-mixtures-and-em
---

# Gaussian mixtures and EM

A customer lives exactly on the boundary between two depots' areas. k-means must send them to one depot, and it does, with full confidence, because a point's group is always an integer. Nudge the customer's address by a metre and the answer flips. Yet nothing in the data says which depot is right. The honest answer is "about half and half".

Hard assignment costs you in a second way. A customer on the boundary pulls the mean of the group they were forced into exactly as hard as a customer standing next to the depot, and the depots drift. Also, k-means measures plain distance, so it describes every group as a round ball, even when the real group is a stretched and tilted cloud.

This chapter gives each point a share in every group, and lets every group have its own shape. The method is a Gaussian mixture. First you compute the shares, which is the E-step. Then you ask how a blob should move if you trust those shares, which is the M-step, and why alternating the two can never make the fit worse. Along the way you see that k-means is what the method becomes when the blobs shrink.

## What you already know

- A generative classifier scores each class by the prior times how typical the point is for that class, then rescales the scores to add up to 1 ([Bayes' rule and Gaussian classes](/lessons/017-bayes-rule-and-gaussian-classes)).
- A Gaussian is described by a mean and a covariance, where the covariance averages the products of the gaps around the mean.
- Taking logs turns a product into a sum. A probability density is not a probability: it is a height, and only comparisons between heights matter here.
- [k-means](/lessons/030-k-means) moves each centre to the mean of its group, and k-means++ seeds good starting centres.
- EM is a guess-and-refit loop whose log-likelihood rises every round, as in [probabilistic PCA](/lessons/029-power-iteration-and-probabilistic-pca).

## The honest answer is a split

Imagine the data came from a game with two steps. First, roll a weighted die to choose a blob. Second, draw a point from that blob's bell curve. You see the points, never the rolls. Clustering is the job of guessing the rolls. The roll is hidden, so statisticians call it a **latent** choice.

Figure 1 shows the game in one dimension. The blobs have means $0$ and $2$, the same width, and weight $0.5$ each, so the die is fair. The dashed bells are the weighted blobs. The solid curve is their sum, the **mixture**. A bell says "points are common near the centre and rare far away", which is how a cluster looks, and it has two knobs you can fit: a centre and a spread.

```figure
<svg viewBox="0 0 600 295" role="img" aria-label="Two bell curves with means 0 and 2, drawn dashed, and the solid curve that is their sum, with bars at x = 0 and x = 1 showing each blob's share">
<defs></defs>
<line x1="55.0" y1="240.0" x2="545.0" y2="240.0" class="stroke-muted" />
<line x1="55.0" y1="40.0" x2="55.0" y2="240.0" class="stroke-muted" />
<line x1="136.7" y1="240.0" x2="136.7" y2="244.0" class="stroke-muted" />
<text x="136.7" y="257.0" class="fig-small" text-anchor="middle" >-2</text>
<line x1="191.1" y1="240.0" x2="191.1" y2="244.0" class="stroke-muted" />
<text x="191.1" y="257.0" class="fig-small" text-anchor="middle" >-1</text>
<line x1="245.6" y1="240.0" x2="245.6" y2="244.0" class="stroke-muted" />
<text x="245.6" y="257.0" class="fig-small" text-anchor="middle" >0</text>
<line x1="300.0" y1="240.0" x2="300.0" y2="244.0" class="stroke-muted" />
<text x="300.0" y="257.0" class="fig-small" text-anchor="middle" >1</text>
<line x1="354.4" y1="240.0" x2="354.4" y2="244.0" class="stroke-muted" />
<text x="354.4" y="257.0" class="fig-small" text-anchor="middle" >2</text>
<line x1="408.9" y1="240.0" x2="408.9" y2="244.0" class="stroke-muted" />
<text x="408.9" y="257.0" class="fig-small" text-anchor="middle" >3</text>
<line x1="463.3" y1="240.0" x2="463.3" y2="244.0" class="stroke-muted" />
<text x="463.3" y="257.0" class="fig-small" text-anchor="middle" >4</text>
<line x1="55.0" y1="240.0" x2="545.0" y2="240.0" class="stroke-grid" />
<line x1="51.0" y1="240.0" x2="55.0" y2="240.0" class="stroke-muted" />
<text x="48.0" y="244.0" class="fig-small" text-anchor="end" >0.0</text>
<line x1="55.0" y1="195.6" x2="545.0" y2="195.6" class="stroke-grid" />
<line x1="51.0" y1="195.6" x2="55.0" y2="195.6" class="stroke-muted" />
<text x="48.0" y="199.6" class="fig-small" text-anchor="end" >0.1</text>
<line x1="55.0" y1="151.1" x2="545.0" y2="151.1" class="stroke-grid" />
<line x1="51.0" y1="151.1" x2="55.0" y2="151.1" class="stroke-muted" />
<text x="48.0" y="155.1" class="fig-small" text-anchor="end" >0.2</text>
<line x1="55.0" y1="106.7" x2="545.0" y2="106.7" class="stroke-grid" />
<line x1="51.0" y1="106.7" x2="55.0" y2="106.7" class="stroke-muted" />
<text x="48.0" y="110.7" class="fig-small" text-anchor="end" >0.3</text>
<line x1="55.0" y1="62.2" x2="545.0" y2="62.2" class="stroke-grid" />
<line x1="51.0" y1="62.2" x2="55.0" y2="62.2" class="stroke-muted" />
<text x="48.0" y="66.2" class="fig-small" text-anchor="end" >0.4</text>
<text x="300.0" y="24.0" class="fig-bold" text-anchor="middle" >Two blobs with weight 0.5 each, and the density of their mixture</text>
<polyline points="55.0,239.8 57.7,239.8 60.4,239.7 63.2,239.7 65.9,239.6 68.6,239.5 71.3,239.5 74.1,239.4 76.8,239.3 79.5,239.2 82.2,239.0 84.9,238.9 87.7,238.7 90.4,238.5 93.1,238.2 95.8,238.0 98.6,237.7 101.3,237.4 104.0,237.0 106.7,236.6 109.4,236.1 112.2,235.6 114.9,235.0 117.6,234.4 120.3,233.7 123.1,232.9 125.8,232.1 128.5,231.2 131.2,230.2 133.9,229.1 136.7,228.0 139.4,226.7 142.1,225.4 144.8,223.9 147.6,222.4 150.3,220.7 153.0,219.0 155.7,217.2 158.4,215.2 161.2,213.2 163.9,211.0 166.6,208.8 169.3,206.5 172.1,204.0 174.8,201.5 177.5,199.0 180.2,196.3 182.9,193.6 185.7,190.9 188.4,188.1 191.1,185.2 193.8,182.4 196.6,179.5 199.3,176.7 202.0,173.9 204.7,171.1 207.4,168.3 210.2,165.6 212.9,162.9 215.6,160.4 218.3,157.9 221.1,155.5 223.8,153.2 226.5,151.0 229.2,149.0 231.9,147.0 234.7,145.2 237.4,143.5 240.1,142.0 242.8,140.6 245.6,139.3 248.3,138.2 251.0,137.2 253.7,136.3 256.4,135.6 259.2,134.9 261.9,134.3 264.6,133.9 267.3,133.5 270.1,133.2 272.8,133.0 275.5,132.8 278.2,132.7 280.9,132.6 283.7,132.5 286.4,132.5 289.1,132.5 291.8,132.5 294.6,132.5 297.3,132.5 300.0,132.5 302.7,132.5 305.4,132.5 308.2,132.5 310.9,132.5 313.6,132.5 316.3,132.5 319.1,132.6 321.8,132.7 324.5,132.8 327.2,133.0 329.9,133.2 332.7,133.5 335.4,133.9 338.1,134.3 340.8,134.9 343.6,135.6 346.3,136.3 349.0,137.2 351.7,138.2 354.4,139.3 357.2,140.6 359.9,142.0 362.6,143.5 365.3,145.2 368.1,147.0 370.8,149.0 373.5,151.0 376.2,153.2 378.9,155.5 381.7,157.9 384.4,160.4 387.1,162.9 389.8,165.6 392.6,168.3 395.3,171.1 398.0,173.9 400.7,176.7 403.4,179.5 406.2,182.4 408.9,185.2 411.6,188.1 414.3,190.9 417.1,193.6 419.8,196.3 422.5,199.0 425.2,201.5 427.9,204.0 430.7,206.5 433.4,208.8 436.1,211.0 438.8,213.2 441.6,215.2 444.3,217.2 447.0,219.0 449.7,220.7 452.4,222.4 455.2,223.9 457.9,225.4 460.6,226.7 463.3,228.0 466.1,229.1 468.8,230.2 471.5,231.2 474.2,232.1 476.9,232.9 479.7,233.7 482.4,234.4 485.1,235.0 487.8,235.6 490.6,236.1 493.3,236.6 496.0,237.0 498.7,237.4 501.4,237.7 504.2,238.0 506.9,238.2 509.6,238.5 512.3,238.7 515.1,238.9 517.8,239.0 520.5,239.2 523.2,239.3 525.9,239.4 528.7,239.5 531.4,239.5 534.1,239.6 536.8,239.7 539.6,239.7 542.3,239.8 545.0,239.8" fill="none" class="stroke-ink" stroke-width="2.5"/>
<polyline points="55.0,239.8 57.7,239.8 60.4,239.7 63.2,239.7 65.9,239.6 68.6,239.5 71.3,239.5 74.1,239.4 76.8,239.3 79.5,239.2 82.2,239.0 84.9,238.9 87.7,238.7 90.4,238.5 93.1,238.2 95.8,238.0 98.6,237.7 101.3,237.4 104.0,237.0 106.7,236.6 109.4,236.1 112.2,235.6 114.9,235.0 117.6,234.4 120.3,233.7 123.1,232.9 125.8,232.1 128.5,231.2 131.2,230.2 133.9,229.2 136.7,228.0 139.4,226.8 142.1,225.4 144.8,224.0 147.6,222.5 150.3,220.8 153.0,219.1 155.7,217.3 158.4,215.4 161.2,213.3 163.9,211.2 166.6,209.0 169.3,206.7 172.1,204.4 174.8,201.9 177.5,199.4 180.2,196.8 182.9,194.2 185.7,191.6 188.4,188.9 191.1,186.2 193.8,183.5 196.6,180.9 199.3,178.2 202.0,175.6 204.7,173.1 207.4,170.6 210.2,168.2 212.9,166.0 215.6,163.8 218.3,161.8 221.1,159.9 223.8,158.2 226.5,156.6 229.2,155.2 231.9,154.1 234.7,153.1 237.4,152.3 240.1,151.8 242.8,151.5 245.6,151.3 248.3,151.5 251.0,151.8 253.7,152.3 256.4,153.1 259.2,154.1 261.9,155.2 264.6,156.6 267.3,158.2 270.1,159.9 272.8,161.8 275.5,163.8 278.2,166.0 280.9,168.2 283.7,170.6 286.4,173.1 289.1,175.6 291.8,178.2 294.6,180.9 297.3,183.5 300.0,186.2 302.7,188.9 305.4,191.6 308.2,194.2 310.9,196.8 313.6,199.4 316.3,201.9 319.1,204.4 321.8,206.7 324.5,209.0 327.2,211.2 329.9,213.3 332.7,215.4 335.4,217.3 338.1,219.1 340.8,220.8 343.6,222.5 346.3,224.0 349.0,225.4 351.7,226.8 354.4,228.0 357.2,229.2 359.9,230.2 362.6,231.2 365.3,232.1 368.1,232.9 370.8,233.7 373.5,234.4 376.2,235.0 378.9,235.6 381.7,236.1 384.4,236.6 387.1,237.0 389.8,237.4 392.6,237.7 395.3,238.0 398.0,238.2 400.7,238.5 403.4,238.7 406.2,238.9 408.9,239.0 411.6,239.2 414.3,239.3 417.1,239.4 419.8,239.5 422.5,239.5 425.2,239.6 427.9,239.7 430.7,239.7 433.4,239.8 436.1,239.8 438.8,239.8 441.6,239.9 444.3,239.9 447.0,239.9 449.7,239.9 452.4,239.9 455.2,239.9 457.9,240.0 460.6,240.0 463.3,240.0 466.1,240.0 468.8,240.0 471.5,240.0 474.2,240.0 476.9,240.0 479.7,240.0 482.4,240.0 485.1,240.0 487.8,240.0 490.6,240.0 493.3,240.0 496.0,240.0 498.7,240.0 501.4,240.0 504.2,240.0 506.9,240.0 509.6,240.0 512.3,240.0 515.1,240.0 517.8,240.0 520.5,240.0 523.2,240.0 525.9,240.0 528.7,240.0 531.4,240.0 534.1,240.0 536.8,240.0 539.6,240.0 542.3,240.0 545.0,240.0" fill="none" class="stroke-a" stroke-width="2" stroke-dasharray="6 4"/>
<polyline points="55.0,240.0 57.7,240.0 60.4,240.0 63.2,240.0 65.9,240.0 68.6,240.0 71.3,240.0 74.1,240.0 76.8,240.0 79.5,240.0 82.2,240.0 84.9,240.0 87.7,240.0 90.4,240.0 93.1,240.0 95.8,240.0 98.6,240.0 101.3,240.0 104.0,240.0 106.7,240.0 109.4,240.0 112.2,240.0 114.9,240.0 117.6,240.0 120.3,240.0 123.1,240.0 125.8,240.0 128.5,240.0 131.2,240.0 133.9,240.0 136.7,240.0 139.4,240.0 142.1,240.0 144.8,239.9 147.6,239.9 150.3,239.9 153.0,239.9 155.7,239.9 158.4,239.9 161.2,239.8 163.9,239.8 166.6,239.8 169.3,239.7 172.1,239.7 174.8,239.6 177.5,239.5 180.2,239.5 182.9,239.4 185.7,239.3 188.4,239.2 191.1,239.0 193.8,238.9 196.6,238.7 199.3,238.5 202.0,238.2 204.7,238.0 207.4,237.7 210.2,237.4 212.9,237.0 215.6,236.6 218.3,236.1 221.1,235.6 223.8,235.0 226.5,234.4 229.2,233.7 231.9,232.9 234.7,232.1 237.4,231.2 240.1,230.2 242.8,229.2 245.6,228.0 248.3,226.8 251.0,225.4 253.7,224.0 256.4,222.5 259.2,220.8 261.9,219.1 264.6,217.3 267.3,215.4 270.1,213.3 272.8,211.2 275.5,209.0 278.2,206.7 280.9,204.4 283.7,201.9 286.4,199.4 289.1,196.8 291.8,194.2 294.6,191.6 297.3,188.9 300.0,186.2 302.7,183.5 305.4,180.9 308.2,178.2 310.9,175.6 313.6,173.1 316.3,170.6 319.1,168.2 321.8,166.0 324.5,163.8 327.2,161.8 329.9,159.9 332.7,158.2 335.4,156.6 338.1,155.2 340.8,154.1 343.6,153.1 346.3,152.3 349.0,151.8 351.7,151.5 354.4,151.3 357.2,151.5 359.9,151.8 362.6,152.3 365.3,153.1 368.1,154.1 370.8,155.2 373.5,156.6 376.2,158.2 378.9,159.9 381.7,161.8 384.4,163.8 387.1,166.0 389.8,168.2 392.6,170.6 395.3,173.1 398.0,175.6 400.7,178.2 403.4,180.9 406.2,183.5 408.9,186.2 411.6,188.9 414.3,191.6 417.1,194.2 419.8,196.8 422.5,199.4 425.2,201.9 427.9,204.4 430.7,206.7 433.4,209.0 436.1,211.2 438.8,213.3 441.6,215.4 444.3,217.3 447.0,219.1 449.7,220.8 452.4,222.5 455.2,224.0 457.9,225.4 460.6,226.8 463.3,228.0 466.1,229.2 468.8,230.2 471.5,231.2 474.2,232.1 476.9,232.9 479.7,233.7 482.4,234.4 485.1,235.0 487.8,235.6 490.6,236.1 493.3,236.6 496.0,237.0 498.7,237.4 501.4,237.7 504.2,238.0 506.9,238.2 509.6,238.5 512.3,238.7 515.1,238.9 517.8,239.0 520.5,239.2 523.2,239.3 525.9,239.4 528.7,239.5 531.4,239.5 534.1,239.6 536.8,239.7 539.6,239.7 542.3,239.8 545.0,239.8" fill="none" class="stroke-c" stroke-width="2" stroke-dasharray="6 4"/>
<text x="142.1" y="177.8" class="fig-bold" text-anchor="middle" dy="-4">blob 0</text>
<text x="474.2" y="177.8" class="fig-bold" text-anchor="middle" dy="-4">blob 1</text>
<text x="354.4" y="120.0" class="fig-bold" text-anchor="middle" >mixture</text>
<line x1="245.6" y1="240.0" x2="245.6" y2="44.0" class="stroke-muted dashed" />
<rect x="231.6" y="151.3" width="12.0" height="88.7" class="fill-a" />
<rect x="247.6" y="228.0" width="12.0" height="12.0" class="fill-c" />
<line x1="300.0" y1="240.0" x2="300.0" y2="44.0" class="stroke-muted dashed" />
<rect x="286.0" y="186.2" width="12.0" height="53.8" class="fill-a" />
<rect x="302.0" y="186.2" width="12.0" height="53.8" class="fill-c" />
<rect x="212.0" y="107.0" width="56.0" height="19.0" class="fill-paper"/><text x="240.0" y="121.0" class="fig-small" text-anchor="middle" >88 / 12</text>
<rect x="272.0" y="107.0" width="56.0" height="19.0" class="fill-paper"/><text x="300.0" y="121.0" class="fig-small" text-anchor="middle" >50 / 50</text>
<text x="63.2" y="62.2" class="fig-small" text-anchor="start" >bars: weight times bell height,</text>
<text x="63.2" y="79.2" class="fig-small" text-anchor="start" >blue is blob 0, green is blob 1.</text>
<text x="63.2" y="96.2" class="fig-small" text-anchor="start" >labels: the shares in percent</text>
<text x="300.0" y="285.0" class="fig-small" text-anchor="middle" >position x</text>
</svg>
Figure 1. The solid curve is the density of the mixture, the sum of the two weighted bells. At x = 0 the blue bar is much taller, so blob 0 takes 88 percent. At x = 1 the bars are equal, and the honest answer is 50 and 50.
```

Now ask who made a given point. At $x=0$ the blue bar is tall and the green bar is short, so blob 0 probably made it. At $x=1$ the bars are equal. The **responsibility** of blob $j$ for point $i$ is the probability that blob $j$ rolled it. It is Bayes' rule, the same rule as in the generative classifier, with the labels hidden.

Take the points $x=0$ and $x=2$. At $x=0$, blob 0's bell has height $0.399$ and blob 1's has height $0.054$. Multiply each by the weight $0.5$ to get $0.1995$ and $0.0270$. Their total is $0.2265$, so the shares are $0.1995/0.2265=0.8808$ and $0.0270/0.2265=0.1192$. By symmetry, the point at $2$ has the shares swapped. The responsibilities are the weighted heights, rescaled to add to 1:

$$
r_{ij} = \frac{\pi_j \, \mathrm{N}(x_i; \mu_j, \Sigma_j)}{\sum_{l=1}^{k} \pi_l \, \mathrm{N}(x_i; \mu_l, \Sigma_l)} .
$$

Read it aloud. $\pi_j$ is blob $j$'s weight, the chance the die picks it. $\mathrm{N}(x_i;\mu_j,\Sigma_j)$ is the height of its bell at the point. The numerator is how much blob $j$ explains the point, and the denominator adds up all the blobs, so each row of $r$ adds to 1. The sum in the denominator is $0.2265$, the density the mixture assigns to $x=0$.

## Blobs with a shape

In two dimensions a blob needs more than a spread. Its **covariance** matrix $\Sigma$ is a small table. Its diagonal holds how far the blob spreads along each axis, and its off-diagonal entry says whether the two coordinates rise together. A positive off-diagonal entry tilts the blob toward the upper right. The level curve of a bell is an ellipse, and $\Sigma$ sets its stretch and its tilt. Figure 2 shows three blobs, one of them tilted, and the shares they give to 54 sample points.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Left, 54 points from three blobs with one-sigma ellipses, one tilted. Right, the same points coloured by responsibility, with ringed points that belong to no single blob">
<defs></defs>
<rect x="20.0" y="40.0" width="270.0" height="240.0" class="fill-sunk stroke-grid" />
<text x="155.0" y="26.0" class="fig-bold" text-anchor="middle" >Three blobs and their one-sigma ellipses</text>
<rect x="310.0" y="40.0" width="270.0" height="240.0" class="fill-sunk stroke-grid" />
<text x="445.0" y="26.0" class="fig-bold" text-anchor="middle" >Each point coloured by its responsibilities</text>
<polyline points="123.6,212.5 123.5,210.1 123.1,207.8 122.5,205.5 121.6,203.3 120.6,201.2 119.3,199.2 117.8,197.4 116.1,195.7 114.3,194.2 112.3,192.9 110.2,191.9 108.0,191.0 105.7,190.4 103.4,190.0 101.0,189.9 98.6,190.0 96.3,190.4 94.0,191.0 91.8,191.9 89.7,192.9 87.7,194.2 85.9,195.7 84.2,197.4 82.7,199.2 81.4,201.2 80.4,203.3 79.5,205.5 78.9,207.8 78.5,210.1 78.4,212.5 78.5,214.9 78.9,217.2 79.5,219.5 80.4,221.7 81.4,223.8 82.7,225.8 84.2,227.6 85.9,229.3 87.7,230.8 89.7,232.1 91.8,233.1 94.0,234.0 96.3,234.6 98.6,235.0 101.0,235.1 103.4,235.0 105.7,234.6 108.0,234.0 110.2,233.1 112.3,232.1 114.3,230.8 116.1,229.3 117.8,227.6 119.3,225.8 120.6,223.8 121.6,221.7 122.5,219.5 123.1,217.2 123.5,214.9 123.6,212.5" fill="none" class="stroke-a" stroke-width="2"/>
<polyline points="413.6,212.5 413.5,210.1 413.1,207.8 412.5,205.5 411.6,203.3 410.6,201.2 409.3,199.2 407.8,197.4 406.1,195.7 404.3,194.2 402.3,192.9 400.2,191.9 398.0,191.0 395.7,190.4 393.4,190.0 391.0,189.9 388.6,190.0 386.3,190.4 384.0,191.0 381.8,191.9 379.7,192.9 377.7,194.2 375.9,195.7 374.2,197.4 372.7,199.2 371.4,201.2 370.4,203.3 369.5,205.5 368.9,207.8 368.5,210.1 368.4,212.5 368.5,214.9 368.9,217.2 369.5,219.5 370.4,221.7 371.4,223.8 372.7,225.8 374.2,227.6 375.9,229.3 377.7,230.8 379.7,232.1 381.8,233.1 384.0,234.0 386.3,234.6 388.6,235.0 391.0,235.1 393.4,235.0 395.7,234.6 398.0,234.0 400.2,233.1 402.3,232.1 404.3,230.8 406.1,229.3 407.8,227.6 409.3,225.8 410.6,223.8 411.6,221.7 412.5,219.5 413.1,217.2 413.5,214.9 413.6,212.5" fill="none" class="stroke-a" stroke-width="2" stroke-dasharray="5 4"/>
<polyline points="188.4,204.0 185.4,206.3 182.5,208.6 179.5,210.7 176.6,212.6 173.7,214.4 170.9,216.0 168.3,217.3 165.7,218.4 163.4,219.3 161.3,220.0 159.4,220.3 157.7,220.4 156.3,220.3 155.2,219.9 154.4,219.2 153.9,218.3 153.7,217.1 153.8,215.7 154.2,214.1 154.9,212.3 155.9,210.3 157.2,208.2 158.8,205.9 160.6,203.6 162.7,201.1 165.0,198.6 167.4,196.1 170.0,193.6 172.8,191.1 175.6,188.6 178.6,186.3 181.5,184.0 184.5,181.9 187.4,180.0 190.3,178.2 193.1,176.6 195.7,175.3 198.3,174.2 200.6,173.3 202.7,172.6 204.6,172.3 206.3,172.2 207.7,172.3 208.8,172.7 209.6,173.4 210.1,174.3 210.3,175.5 210.2,176.9 209.8,178.5 209.1,180.3 208.1,182.3 206.8,184.4 205.2,186.7 203.4,189.0 201.3,191.5 199.0,194.0 196.6,196.5 194.0,199.0 191.2,201.5 188.4,204.0" fill="none" class="stroke-c" stroke-width="2"/>
<polyline points="478.4,204.0 475.4,206.3 472.5,208.6 469.5,210.7 466.6,212.6 463.7,214.4 460.9,216.0 458.3,217.3 455.7,218.4 453.4,219.3 451.3,220.0 449.4,220.3 447.7,220.4 446.3,220.3 445.2,219.9 444.4,219.2 443.9,218.3 443.7,217.1 443.8,215.7 444.2,214.1 444.9,212.3 445.9,210.3 447.2,208.2 448.8,205.9 450.6,203.6 452.7,201.1 455.0,198.6 457.4,196.1 460.0,193.6 462.8,191.1 465.6,188.6 468.6,186.3 471.5,184.0 474.5,181.9 477.4,180.0 480.3,178.2 483.1,176.6 485.7,175.3 488.3,174.2 490.6,173.3 492.7,172.6 494.6,172.3 496.3,172.2 497.7,172.3 498.8,172.7 499.6,173.4 500.1,174.3 500.3,175.5 500.2,176.9 499.8,178.5 499.1,180.3 498.1,182.3 496.8,184.4 495.2,186.7 493.4,189.0 491.3,191.5 489.0,194.0 486.6,196.5 484.0,199.0 481.2,201.5 478.4,204.0" fill="none" class="stroke-c" stroke-width="2" stroke-dasharray="5 4"/>
<polyline points="155.2,128.8 155.1,126.8 154.8,124.8 154.3,122.9 153.5,121.0 152.6,119.3 151.5,117.6 150.3,116.0 148.9,114.6 147.3,113.4 145.6,112.3 143.9,111.4 142.0,110.6 140.1,110.1 138.1,109.8 136.1,109.7 134.1,109.8 132.1,110.1 130.2,110.6 128.3,111.4 126.6,112.3 124.9,113.4 123.3,114.6 121.9,116.0 120.7,117.6 119.6,119.3 118.7,121.0 117.9,122.9 117.4,124.8 117.1,126.8 117.0,128.8 117.1,130.8 117.4,132.8 117.9,134.7 118.7,136.6 119.6,138.3 120.7,140.0 121.9,141.6 123.3,143.0 124.9,144.2 126.6,145.3 128.3,146.2 130.2,147.0 132.1,147.5 134.1,147.8 136.1,147.9 138.1,147.8 140.1,147.5 142.0,147.0 143.9,146.2 145.6,145.3 147.3,144.2 148.9,143.0 150.3,141.6 151.5,140.0 152.6,138.3 153.5,136.6 154.3,134.7 154.8,132.8 155.1,130.8 155.2,128.8" fill="none" class="stroke-d" stroke-width="2"/>
<polyline points="445.2,128.8 445.1,126.8 444.8,124.8 444.3,122.9 443.5,121.0 442.6,119.3 441.5,117.6 440.3,116.0 438.9,114.6 437.3,113.4 435.6,112.3 433.9,111.4 432.0,110.6 430.1,110.1 428.1,109.8 426.1,109.7 424.1,109.8 422.1,110.1 420.2,110.6 418.3,111.4 416.6,112.3 414.9,113.4 413.3,114.6 411.9,116.0 410.7,117.6 409.6,119.3 408.7,121.0 407.9,122.9 407.4,124.8 407.1,126.8 407.0,128.8 407.1,130.8 407.4,132.8 407.9,134.7 408.7,136.6 409.6,138.3 410.7,140.0 411.9,141.6 413.3,143.0 414.9,144.2 416.6,145.3 418.3,146.2 420.2,147.0 422.1,147.5 424.1,147.8 426.1,147.9 428.1,147.8 430.1,147.5 432.0,147.0 433.9,146.2 435.6,145.3 437.3,144.2 438.9,143.0 440.3,141.6 441.5,140.0 442.6,138.3 443.5,136.6 444.3,134.7 444.8,132.8 445.1,130.8 445.2,128.8" fill="none" class="stroke-d" stroke-width="2" stroke-dasharray="5 4"/>
<circle cx="82.9" cy="242.4" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="95.4" cy="203.0" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="126.7" cy="210.0" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="88.5" cy="230.2" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="117.9" cy="175.6" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="107.2" cy="240.4" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="79.4" cy="176.4" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="105.6" cy="251.6" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="99.1" cy="238.8" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="86.8" cy="223.5" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="84.9" cy="200.0" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="99.6" cy="225.8" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="110.3" cy="193.8" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="63.9" cy="218.3" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="78.8" cy="216.4" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="71.9" cy="212.0" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="100.1" cy="219.4" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="77.3" cy="221.4" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="220.7" cy="181.7" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="182.9" cy="209.9" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="145.1" cy="217.6" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="235.4" cy="148.5" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="212.2" cy="170.8" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="193.4" cy="212.5" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="190.1" cy="192.9" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="163.0" cy="227.4" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="168.6" cy="221.6" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="196.5" cy="195.2" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="138.4" cy="225.1" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="110.8" cy="247.1" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="153.3" cy="209.2" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="199.2" cy="182.9" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="147.3" cy="230.3" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="176.9" cy="200.8" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="167.5" cy="213.0" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="184.7" cy="191.0" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="138.1" cy="145.3" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="153.2" cy="153.6" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="113.2" cy="153.3" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="154.6" cy="135.7" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="117.6" cy="150.5" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="144.1" cy="148.9" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="111.8" cy="117.1" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="113.3" cy="135.0" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="136.0" cy="137.3" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="135.1" cy="103.2" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="126.2" cy="152.8" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="101.0" cy="132.7" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="129.4" cy="123.7" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="127.2" cy="137.9" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="122.3" cy="138.7" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="139.2" cy="136.1" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="138.0" cy="92.5" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="145.2" cy="158.9" r="4.5" class="fill-ink" opacity="0.55"/>
<circle cx="372.9" cy="242.4" r="5.5" class="fill-a" opacity="1.00"/>
<circle cx="385.4" cy="203.0" r="5.5" class="fill-a" opacity="1.00"/>
<circle cx="416.7" cy="210.0" r="5.5" class="fill-a" opacity="0.96"/>
<circle cx="416.7" cy="210.0" r="5.5" class="fill-c" opacity="0.04"/>
<circle cx="378.5" cy="230.2" r="5.5" class="fill-a" opacity="1.00"/>
<circle cx="407.9" cy="175.6" r="5.5" class="fill-a" opacity="0.82"/>
<circle cx="407.9" cy="175.6" r="5.5" class="fill-d" opacity="0.18"/>
<circle cx="407.9" cy="175.6" r="8" class="stroke-ink" fill="none" stroke-width="1.5"/>
<circle cx="397.2" cy="240.4" r="5.5" class="fill-a" opacity="0.94"/>
<circle cx="397.2" cy="240.4" r="5.5" class="fill-c" opacity="0.06"/>
<circle cx="369.4" cy="176.4" r="5.5" class="fill-a" opacity="1.00"/>
<circle cx="395.6" cy="251.6" r="5.5" class="fill-a" opacity="0.85"/>
<circle cx="395.6" cy="251.6" r="5.5" class="fill-c" opacity="0.15"/>
<circle cx="395.6" cy="251.6" r="8" class="stroke-ink" fill="none" stroke-width="1.5"/>
<circle cx="389.1" cy="238.8" r="5.5" class="fill-a" opacity="0.99"/>
<circle cx="389.1" cy="238.8" r="5.5" class="fill-c" opacity="0.01"/>
<circle cx="376.8" cy="223.5" r="5.5" class="fill-a" opacity="1.00"/>
<circle cx="374.9" cy="200.0" r="5.5" class="fill-a" opacity="1.00"/>
<circle cx="389.6" cy="225.8" r="5.5" class="fill-a" opacity="1.00"/>
<circle cx="400.3" cy="193.8" r="5.5" class="fill-a" opacity="1.00"/>
<circle cx="353.9" cy="218.3" r="5.5" class="fill-a" opacity="1.00"/>
<circle cx="368.8" cy="216.4" r="5.5" class="fill-a" opacity="1.00"/>
<circle cx="361.9" cy="212.0" r="5.5" class="fill-a" opacity="1.00"/>
<circle cx="390.1" cy="219.4" r="5.5" class="fill-a" opacity="1.00"/>
<circle cx="367.3" cy="221.4" r="5.5" class="fill-a" opacity="1.00"/>
<circle cx="510.7" cy="181.7" r="5.5" class="fill-c" opacity="1.00"/>
<circle cx="472.9" cy="209.9" r="5.5" class="fill-c" opacity="1.00"/>
<circle cx="435.1" cy="217.6" r="5.5" class="fill-a" opacity="0.21"/>
<circle cx="435.1" cy="217.6" r="5.5" class="fill-c" opacity="0.79"/>
<circle cx="435.1" cy="217.6" r="8" class="stroke-ink" fill="none" stroke-width="1.5"/>
<circle cx="525.4" cy="148.5" r="5.5" class="fill-c" opacity="1.00"/>
<circle cx="502.2" cy="170.8" r="5.5" class="fill-c" opacity="1.00"/>
<circle cx="483.4" cy="212.5" r="5.5" class="fill-c" opacity="1.00"/>
<circle cx="480.1" cy="192.9" r="5.5" class="fill-c" opacity="1.00"/>
<circle cx="453.0" cy="227.4" r="5.5" class="fill-a" opacity="0.04"/>
<circle cx="453.0" cy="227.4" r="5.5" class="fill-c" opacity="0.96"/>
<circle cx="458.6" cy="221.6" r="5.5" class="fill-a" opacity="0.02"/>
<circle cx="458.6" cy="221.6" r="5.5" class="fill-c" opacity="0.98"/>
<circle cx="486.5" cy="195.2" r="5.5" class="fill-c" opacity="1.00"/>
<circle cx="428.4" cy="225.1" r="5.5" class="fill-a" opacity="0.34"/>
<circle cx="428.4" cy="225.1" r="5.5" class="fill-c" opacity="0.66"/>
<circle cx="428.4" cy="225.1" r="8" class="stroke-ink" fill="none" stroke-width="1.5"/>
<circle cx="400.8" cy="247.1" r="5.5" class="fill-a" opacity="0.82"/>
<circle cx="400.8" cy="247.1" r="5.5" class="fill-c" opacity="0.18"/>
<circle cx="400.8" cy="247.1" r="8" class="stroke-ink" fill="none" stroke-width="1.5"/>
<circle cx="443.3" cy="209.2" r="5.5" class="fill-a" opacity="0.09"/>
<circle cx="443.3" cy="209.2" r="5.5" class="fill-c" opacity="0.91"/>
<circle cx="489.2" cy="182.9" r="5.5" class="fill-c" opacity="1.00"/>
<circle cx="437.3" cy="230.3" r="5.5" class="fill-a" opacity="0.15"/>
<circle cx="437.3" cy="230.3" r="5.5" class="fill-c" opacity="0.85"/>
<circle cx="437.3" cy="230.3" r="8" class="stroke-ink" fill="none" stroke-width="1.5"/>
<circle cx="466.9" cy="200.8" r="5.5" class="fill-c" opacity="1.00"/>
<circle cx="457.5" cy="213.0" r="5.5" class="fill-a" opacity="0.01"/>
<circle cx="457.5" cy="213.0" r="5.5" class="fill-c" opacity="0.99"/>
<circle cx="474.7" cy="191.0" r="5.5" class="fill-c" opacity="1.00"/>
<circle cx="428.1" cy="145.3" r="5.5" class="fill-d" opacity="1.00"/>
<circle cx="443.2" cy="153.6" r="5.5" class="fill-d" opacity="0.99"/>
<circle cx="403.2" cy="153.3" r="5.5" class="fill-a" opacity="0.09"/>
<circle cx="403.2" cy="153.3" r="5.5" class="fill-d" opacity="0.91"/>
<circle cx="444.6" cy="135.7" r="5.5" class="fill-d" opacity="1.00"/>
<circle cx="407.6" cy="150.5" r="5.5" class="fill-a" opacity="0.04"/>
<circle cx="407.6" cy="150.5" r="5.5" class="fill-d" opacity="0.96"/>
<circle cx="434.1" cy="148.9" r="5.5" class="fill-d" opacity="1.00"/>
<circle cx="401.8" cy="117.1" r="5.5" class="fill-d" opacity="1.00"/>
<circle cx="403.3" cy="135.0" r="5.5" class="fill-d" opacity="1.00"/>
<circle cx="426.0" cy="137.3" r="5.5" class="fill-d" opacity="1.00"/>
<circle cx="425.1" cy="103.2" r="5.5" class="fill-d" opacity="1.00"/>
<circle cx="416.2" cy="152.8" r="5.5" class="fill-a" opacity="0.03"/>
<circle cx="416.2" cy="152.8" r="5.5" class="fill-d" opacity="0.97"/>
<circle cx="391.0" cy="132.7" r="5.5" class="fill-d" opacity="0.99"/>
<circle cx="419.4" cy="123.7" r="5.5" class="fill-d" opacity="1.00"/>
<circle cx="417.2" cy="137.9" r="5.5" class="fill-d" opacity="1.00"/>
<circle cx="412.3" cy="138.7" r="5.5" class="fill-d" opacity="1.00"/>
<circle cx="429.2" cy="136.1" r="5.5" class="fill-d" opacity="1.00"/>
<circle cx="428.0" cy="92.5" r="5.5" class="fill-d" opacity="1.00"/>
<circle cx="435.2" cy="158.9" r="5.5" class="fill-a" opacity="0.02"/>
<circle cx="435.2" cy="158.9" r="5.5" class="fill-d" opacity="0.98"/>
<line x1="385.0" y1="206.5" x2="397.0" y2="218.5" class="stroke-ink" stroke-width="2"/><line x1="385.0" y1="218.5" x2="397.0" y2="206.5" class="stroke-ink" stroke-width="2"/>
<line x1="466.0" y1="190.3" x2="478.0" y2="202.3" class="stroke-ink" stroke-width="2"/><line x1="466.0" y1="202.3" x2="478.0" y2="190.3" class="stroke-ink" stroke-width="2"/>
<line x1="420.1" y1="122.8" x2="432.1" y2="134.8" class="stroke-ink" stroke-width="2"/><line x1="420.1" y1="134.8" x2="432.1" y2="122.8" class="stroke-ink" stroke-width="2"/>
<text x="155.0" y="302.0" class="fig-small" text-anchor="middle" >the tilted blob has a tilted ellipse</text>
<text x="445.0" y="302.0" class="fig-small" text-anchor="middle" >ringed: no blob has 90% (6 of 54 points)</text>
</svg>
Figure 2. The ellipse is where the blob's density falls to a fixed fraction of its peak, and a tilt in the ellipse is a tilt in the covariance. Points deep inside one ellipse are a solid colour. Points where two ellipses overlap blend, and a ring marks them.
```

The distance that matters for a tilted blob is not plain distance but distance measured in units of the blob's own spread. Take $\Sigma=\begin{bmatrix}2&1.2\\1.2&1\end{bmatrix}$ and two points around the centre, $(1,1)$ and $(1,-1)$. Both are at plain squared distance 2. But $(1,1)$ lies along the blob's long axis, and $(1,-1)$ lies across it. Measured with $\Sigma^{-1}$, the squared distances are 1.07 and 9.64. The second point is nine times farther in the blob's own units, so the blob gives it a far lower height. The log of the height is

$$
\log \mathrm{N}(x; \mu, \Sigma) = -\tfrac{1}{2}\left( d \log(2\pi) + \log\det\Sigma + (x-\mu)^\top \Sigma^{-1} (x-\mu) \right) .
$$

Read it aloud: take the squared distance measured with $\Sigma^{-1}$, add $\log\det\Sigma$ (which says a wider blob spreads its probability thinner, so its peak is lower), add $d\log(2\pi)$, and multiply by $-\tfrac12$. For the two points above the log heights are -2.084 and -6.369. A one-dimensional bell with variance 1 has log height $-\tfrac12\log(2\pi)\approx -0.919$ at its peak. In numpy, with `slogdet` for the log of the determinant:

```python
gap = X - mean
_, logdet = np.linalg.slogdet(cov)
squared = np.sum((gap @ np.linalg.inv(cov)) * gap, axis=1)
log_density = -0.5 * (d * np.log(2 * np.pi) + logdet + squared)
```

## Why logs, and why the sum is trouble

How well does the model explain all $n$ points? Multiply the density of every point. A product of hundreds of heights below 1 is a tiny number that a computer rounds to zero, so take its log, which turns the product into a sum. This is the **log-likelihood**:

$$
\log L = \sum_{i=1}^{n} \log \sum_{j=1}^{k} \pi_j \, \mathrm{N}(x_i; \mu_j, \Sigma_j) .
$$

The log sits outside a sum over blobs, and the log of a sum does not split into pieces. That is why there is no formula for the best $\pi$, $\mu$, and $\Sigma$ in one step, and why the method alternates instead. For the two points, $\log L=\log 0.2265+\log 0.2265=-2.970$.

The same log creates a second problem when you compute the shares. Work with the log scores $s_{ij}=\log\pi_j+\log\mathrm{N}(x_i;\mu_j,\Sigma_j)$. To get the shares you must undo the log, and that is where a computer fails. Try two blobs far from a point, with scores $-1000$ and $-1001$. In floating point `np.exp(-1000.0)` is `0.0`. Both terms vanish, the row sums to 0, and the share is $0/0$, which is `nan`.

The fix is to subtract each row's largest score $m_i$ before exponentiating. The shares do not change, because $m_i$ cancels from the top and the bottom, and the largest shifted term is exactly $e^0=1$, so the row can never vanish:

$$
\log \sum_{j} e^{s_{ij}} = m_i + \log \sum_{j} e^{s_{ij} - m_i} .
$$

For the scores $-1000$ and $-1001$ this gives $-1000+\log(1+e^{-1})=$ -999.6867. This is the **log-sum-exp**. In numpy, with the row maximum kept as a column:

```python
m = s.max(axis=1, keepdims=True)
log_px = m[:, 0] + np.log(np.exp(s - m).sum(axis=1))
resp = np.exp(s - log_px[:, None])
```

> [!TIP]
> Keep the maximum with `keepdims=True`, so it has shape `(n, 1)`. A bare `(n,)` vector broadcasts across columns instead of rows, and when $n$ equals $k$ nothing complains.

> [!KEY]
> A Gaussian mixture says each point came from a blob chosen by a weighted die. The E-step computes, for each point, the probability that each blob made it: weight times height, rescaled so the row adds to 1, with logs and log-sum-exp keeping the numbers safe.

## Refit with weighted statistics

For the two points at $0$ and $2$ the E-step said that point $0$ belongs 88 percent to blob 0 and 12 percent to blob 1, and the point at $2$ the other way round. If you trust those shares for a moment, how should blob 0 move? Its old mean was $0$. The point at $2$ owns only 12 percent of it, so blob 0 should shift a little toward $2$, not all the way and not at all.

Here is the whole idea. Compute the ordinary count, mean, and covariance of the points that belong to a blob, but let each point count in proportion to its share. A point with share $0.12$ counts as $0.12$ of a point. The **soft count** of blob $j$ is $N_j=\sum_i r_{ij}$, the effective number of points the blob owns. For blob 0 in the two-point example, $N_0=0.8808+0.1192=1$. The three refits are then:

$$
\pi_j = \frac{N_j}{n}, \qquad
\mu_j = \frac{1}{N_j} \sum_{i=1}^{n} r_{ij}\, x_i, \qquad
\Sigma_j = \frac{1}{N_j} \sum_{i=1}^{n} r_{ij}\, (x_i - \mu_j)(x_i - \mu_j)^\top + \epsilon I .
$$

Read them aloud. The weight is the blob's share of all $n$ points. The mean is the weighted average of the points. The covariance is the weighted average of the products of the gaps from that new mean, plus a tiny $\epsilon I$ on the diagonal that you meet in "Where this breaks". If every share were exactly 0 or 1, $N_j$ would be the size of a group, $\mu_j$ would be that group's plain mean, and $\Sigma_j$ its plain covariance. For the means, that is exactly the k-means update.

Check it on the two points. The new mean of blob 0 is $(0.8808\cdot 0+0.1192\cdot 2)/1=0.2384$. The squared gaps from it are $0.0568$ and $3.1032$. Weighted, they are $0.8808\cdot0.0568=0.0501$ and $0.1192\cdot3.1032=0.3699$, which add to $0.4200$, so the new variance is $0.42$ and the weight is $1/2$. By symmetry blob 1 moves to $1.7616$ with the same variance. Each blob moved a little toward the other point, which it owns only a small share of. It also narrowed from $1$ to $0.42$, because blob 0 now describes mostly the one point at $0$ plus a small slice of the point at $2$, which is a tighter bunch than a blob of variance $1$ assumed.

In numpy, for the means:

```python
counts = responsibilities.sum(axis=0)              # N_j, shape (k,)
means = (responsibilities.T @ X) / counts[:, None]
```

> [!WARNING]
> The covariance must be measured from the new mean $\mu_j$, the one you just computed in this step. Gaps measured from any other centre, such as the old mean the E-step used or the plain average of all points, give a covariance that is too wide, because the weighted mean is the centre that makes the weighted squared gaps smallest.

## Alternate, and watch the likelihood

The **EM loop** runs the E-step, then the M-step, then the E-step again. The E stands for expectation, the average over the hidden choices that the shares describe, and the M for maximisation, because the refit picks the best blobs for those shares. Start with three equal round blobs, with means at the k-means++ seeds, every covariance equal to the average variance of the data, and weights $1/3$. Figure 3 shows 60 points from three clusters under this loop. Each frame prints $\log L$.

```figure
<svg viewBox="0 0 600 530" role="img" aria-label="Four frames of EM on 60 points from three clusters: round equal ellipses at the start, then ellipses that move and reshape to cover each cluster, with log L rising from -292.7 to -215.8">
<defs><clipPath id="c025-p0"><rect x="20" y="40" width="270" height="222"/></clipPath><clipPath id="c025-p1"><rect x="310" y="40" width="270" height="222"/></clipPath><clipPath id="c025-p3"><rect x="20" y="295" width="270" height="222"/></clipPath><clipPath id="c025-p10"><rect x="310" y="295" width="270" height="222"/></clipPath></defs>
<rect x="20.0" y="40.0" width="270.0" height="222.0" class="fill-sunk stroke-grid" />
<text x="20.0" y="32.0" class="fig-bold" text-anchor="start" >Start</text>
<text x="290.0" y="32.0" class="fig-small" text-anchor="end" >log L = -292.7</text>
<g clip-path="url(#c025-p0)">
<circle cx="121.4" cy="253.3" r="4" class="fill-c" opacity="0.53"/>
<circle cx="121.4" cy="253.3" r="4" class="fill-d" opacity="0.46"/>
<circle cx="89.6" cy="214.2" r="4" class="fill-a" opacity="0.02"/>
<circle cx="89.6" cy="214.2" r="4" class="fill-c" opacity="0.26"/>
<circle cx="89.6" cy="214.2" r="4" class="fill-d" opacity="0.73"/>
<circle cx="72.5" cy="207.3" r="4" class="fill-a" opacity="0.02"/>
<circle cx="72.5" cy="207.3" r="4" class="fill-c" opacity="0.17"/>
<circle cx="72.5" cy="207.3" r="4" class="fill-d" opacity="0.81"/>
<circle cx="41.7" cy="207.6" r="4" class="fill-c" opacity="0.08"/>
<circle cx="41.7" cy="207.6" r="4" class="fill-d" opacity="0.91"/>
<circle cx="64.4" cy="137.8" r="4" class="fill-a" opacity="0.09"/>
<circle cx="64.4" cy="137.8" r="4" class="fill-c" opacity="0.07"/>
<circle cx="64.4" cy="137.8" r="4" class="fill-d" opacity="0.83"/>
<circle cx="85.8" cy="210.0" r="4" class="fill-a" opacity="0.02"/>
<circle cx="85.8" cy="210.0" r="4" class="fill-c" opacity="0.23"/>
<circle cx="85.8" cy="210.0" r="4" class="fill-d" opacity="0.75"/>
<circle cx="75.8" cy="216.2" r="4" class="fill-a" opacity="0.01"/>
<circle cx="75.8" cy="216.2" r="4" class="fill-c" opacity="0.20"/>
<circle cx="75.8" cy="216.2" r="4" class="fill-d" opacity="0.79"/>
<circle cx="60.6" cy="210.8" r="4" class="fill-a" opacity="0.01"/>
<circle cx="60.6" cy="210.8" r="4" class="fill-c" opacity="0.13"/>
<circle cx="60.6" cy="210.8" r="4" class="fill-d" opacity="0.85"/>
<circle cx="90.8" cy="207.8" r="4" class="fill-a" opacity="0.02"/>
<circle cx="90.8" cy="207.8" r="4" class="fill-c" opacity="0.25"/>
<circle cx="90.8" cy="207.8" r="4" class="fill-d" opacity="0.73"/>
<circle cx="100.2" cy="207.0" r="4" class="fill-a" opacity="0.03"/>
<circle cx="100.2" cy="207.0" r="4" class="fill-c" opacity="0.30"/>
<circle cx="100.2" cy="207.0" r="4" class="fill-d" opacity="0.68"/>
<circle cx="81.8" cy="172.7" r="4" class="fill-a" opacity="0.05"/>
<circle cx="81.8" cy="172.7" r="4" class="fill-c" opacity="0.16"/>
<circle cx="81.8" cy="172.7" r="4" class="fill-d" opacity="0.79"/>
<circle cx="92.1" cy="213.0" r="4" class="fill-a" opacity="0.02"/>
<circle cx="92.1" cy="213.0" r="4" class="fill-c" opacity="0.27"/>
<circle cx="92.1" cy="213.0" r="4" class="fill-d" opacity="0.71"/>
<circle cx="77.8" cy="192.5" r="4" class="fill-a" opacity="0.03"/>
<circle cx="77.8" cy="192.5" r="4" class="fill-c" opacity="0.17"/>
<circle cx="77.8" cy="192.5" r="4" class="fill-d" opacity="0.80"/>
<circle cx="119.4" cy="208.4" r="4" class="fill-a" opacity="0.04"/>
<circle cx="119.4" cy="208.4" r="4" class="fill-c" opacity="0.41"/>
<circle cx="119.4" cy="208.4" r="4" class="fill-d" opacity="0.55"/>
<circle cx="76.6" cy="183.4" r="4" class="fill-a" opacity="0.03"/>
<circle cx="76.6" cy="183.4" r="4" class="fill-c" opacity="0.15"/>
<circle cx="76.6" cy="183.4" r="4" class="fill-d" opacity="0.81"/>
<circle cx="64.0" cy="208.8" r="4" class="fill-a" opacity="0.01"/>
<circle cx="64.0" cy="208.8" r="4" class="fill-c" opacity="0.14"/>
<circle cx="64.0" cy="208.8" r="4" class="fill-d" opacity="0.85"/>
<circle cx="98.7" cy="191.7" r="4" class="fill-a" opacity="0.04"/>
<circle cx="98.7" cy="191.7" r="4" class="fill-c" opacity="0.26"/>
<circle cx="98.7" cy="191.7" r="4" class="fill-d" opacity="0.70"/>
<circle cx="83.2" cy="189.9" r="4" class="fill-a" opacity="0.03"/>
<circle cx="83.2" cy="189.9" r="4" class="fill-c" opacity="0.19"/>
<circle cx="83.2" cy="189.9" r="4" class="fill-d" opacity="0.78"/>
<circle cx="25.8" cy="183.0" r="4" class="fill-a" opacity="0.01"/>
<circle cx="25.8" cy="183.0" r="4" class="fill-c" opacity="0.05"/>
<circle cx="25.8" cy="183.0" r="4" class="fill-d" opacity="0.94"/>
<circle cx="62.5" cy="235.9" r="4" class="fill-c" opacity="0.17"/>
<circle cx="62.5" cy="235.9" r="4" class="fill-d" opacity="0.82"/>
<circle cx="234.1" cy="189.3" r="4" class="fill-a" opacity="0.14"/>
<circle cx="234.1" cy="189.3" r="4" class="fill-c" opacity="0.80"/>
<circle cx="234.1" cy="189.3" r="4" class="fill-d" opacity="0.06"/>
<circle cx="219.9" cy="224.2" r="4" class="fill-a" opacity="0.04"/>
<circle cx="219.9" cy="224.2" r="4" class="fill-c" opacity="0.89"/>
<circle cx="219.9" cy="224.2" r="4" class="fill-d" opacity="0.07"/>
<circle cx="229.1" cy="204.1" r="4" class="fill-a" opacity="0.09"/>
<circle cx="229.1" cy="204.1" r="4" class="fill-c" opacity="0.85"/>
<circle cx="229.1" cy="204.1" r="4" class="fill-d" opacity="0.06"/>
<circle cx="256.2" cy="188.4" r="4" class="fill-a" opacity="0.15"/>
<circle cx="256.2" cy="188.4" r="4" class="fill-c" opacity="0.82"/>
<circle cx="256.2" cy="188.4" r="4" class="fill-d" opacity="0.03"/>
<circle cx="232.4" cy="181.3" r="4" class="fill-a" opacity="0.18"/>
<circle cx="232.4" cy="181.3" r="4" class="fill-c" opacity="0.75"/>
<circle cx="232.4" cy="181.3" r="4" class="fill-d" opacity="0.06"/>
<circle cx="224.6" cy="221.3" r="4" class="fill-a" opacity="0.05"/>
<circle cx="224.6" cy="221.3" r="4" class="fill-c" opacity="0.89"/>
<circle cx="224.6" cy="221.3" r="4" class="fill-d" opacity="0.06"/>
<circle cx="240.1" cy="191.7" r="4" class="fill-a" opacity="0.13"/>
<circle cx="240.1" cy="191.7" r="4" class="fill-c" opacity="0.82"/>
<circle cx="240.1" cy="191.7" r="4" class="fill-d" opacity="0.05"/>
<circle cx="224.4" cy="218.5" r="4" class="fill-a" opacity="0.05"/>
<circle cx="224.4" cy="218.5" r="4" class="fill-c" opacity="0.88"/>
<circle cx="224.4" cy="218.5" r="4" class="fill-d" opacity="0.06"/>
<circle cx="233.1" cy="252.1" r="4" class="fill-a" opacity="0.02"/>
<circle cx="233.1" cy="252.1" r="4" class="fill-c" opacity="0.94"/>
<circle cx="233.1" cy="252.1" r="4" class="fill-d" opacity="0.04"/>
<circle cx="242.2" cy="193.4" r="4" class="fill-a" opacity="0.13"/>
<circle cx="242.2" cy="193.4" r="4" class="fill-c" opacity="0.83"/>
<circle cx="242.2" cy="193.4" r="4" class="fill-d" opacity="0.05"/>
<circle cx="196.5" cy="201.9" r="4" class="fill-a" opacity="0.09"/>
<circle cx="196.5" cy="201.9" r="4" class="fill-c" opacity="0.77"/>
<circle cx="196.5" cy="201.9" r="4" class="fill-d" opacity="0.14"/>
<circle cx="209.7" cy="188.2" r="4" class="fill-a" opacity="0.14"/>
<circle cx="209.7" cy="188.2" r="4" class="fill-c" opacity="0.75"/>
<circle cx="209.7" cy="188.2" r="4" class="fill-d" opacity="0.11"/>
<circle cx="188.7" cy="221.0" r="4" class="fill-a" opacity="0.04"/>
<circle cx="188.7" cy="221.0" r="4" class="fill-c" opacity="0.80"/>
<circle cx="188.7" cy="221.0" r="4" class="fill-d" opacity="0.15"/>
<circle cx="242.6" cy="180.4" r="4" class="fill-a" opacity="0.19"/>
<circle cx="242.6" cy="180.4" r="4" class="fill-c" opacity="0.76"/>
<circle cx="242.6" cy="180.4" r="4" class="fill-d" opacity="0.05"/>
<circle cx="186.3" cy="212.9" r="4" class="fill-a" opacity="0.06"/>
<circle cx="186.3" cy="212.9" r="4" class="fill-c" opacity="0.77"/>
<circle cx="186.3" cy="212.9" r="4" class="fill-d" opacity="0.17"/>
<circle cx="235.1" cy="215.1" r="4" class="fill-a" opacity="0.06"/>
<circle cx="235.1" cy="215.1" r="4" class="fill-c" opacity="0.89"/>
<circle cx="235.1" cy="215.1" r="4" class="fill-d" opacity="0.05"/>
<circle cx="259.9" cy="226.5" r="4" class="fill-a" opacity="0.04"/>
<circle cx="259.9" cy="226.5" r="4" class="fill-c" opacity="0.93"/>
<circle cx="259.9" cy="226.5" r="4" class="fill-d" opacity="0.02"/>
<circle cx="235.6" cy="223.7" r="4" class="fill-a" opacity="0.05"/>
<circle cx="235.6" cy="223.7" r="4" class="fill-c" opacity="0.91"/>
<circle cx="235.6" cy="223.7" r="4" class="fill-d" opacity="0.05"/>
<circle cx="256.2" cy="203.5" r="4" class="fill-a" opacity="0.09"/>
<circle cx="256.2" cy="203.5" r="4" class="fill-c" opacity="0.88"/>
<circle cx="256.2" cy="203.5" r="4" class="fill-d" opacity="0.03"/>
<circle cx="221.3" cy="236.8" r="4" class="fill-a" opacity="0.03"/>
<circle cx="221.3" cy="236.8" r="4" class="fill-c" opacity="0.91"/>
<circle cx="221.3" cy="236.8" r="4" class="fill-d" opacity="0.06"/>
<circle cx="188.0" cy="65.6" r="4" class="fill-a" opacity="0.91"/>
<circle cx="188.0" cy="65.6" r="4" class="fill-c" opacity="0.05"/>
<circle cx="188.0" cy="65.6" r="4" class="fill-d" opacity="0.04"/>
<circle cx="169.8" cy="58.0" r="4" class="fill-a" opacity="0.91"/>
<circle cx="169.8" cy="58.0" r="4" class="fill-c" opacity="0.04"/>
<circle cx="169.8" cy="58.0" r="4" class="fill-d" opacity="0.05"/>
<circle cx="161.9" cy="92.9" r="4" class="fill-a" opacity="0.75"/>
<circle cx="161.9" cy="92.9" r="4" class="fill-c" opacity="0.11"/>
<circle cx="161.9" cy="92.9" r="4" class="fill-d" opacity="0.14"/>
<circle cx="139.3" cy="96.1" r="4" class="fill-a" opacity="0.65"/>
<circle cx="139.3" cy="96.1" r="4" class="fill-c" opacity="0.11"/>
<circle cx="139.3" cy="96.1" r="4" class="fill-d" opacity="0.24"/>
<circle cx="181.9" cy="109.0" r="4" class="fill-a" opacity="0.69"/>
<circle cx="181.9" cy="109.0" r="4" class="fill-c" opacity="0.19"/>
<circle cx="181.9" cy="109.0" r="4" class="fill-d" opacity="0.12"/>
<circle cx="143.3" cy="86.7" r="4" class="fill-a" opacity="0.73"/>
<circle cx="143.3" cy="86.7" r="4" class="fill-c" opacity="0.09"/>
<circle cx="143.3" cy="86.7" r="4" class="fill-d" opacity="0.19"/>
<circle cx="159.4" cy="69.1" r="4" class="fill-a" opacity="0.86"/>
<circle cx="159.4" cy="69.1" r="4" class="fill-c" opacity="0.05"/>
<circle cx="159.4" cy="69.1" r="4" class="fill-d" opacity="0.09"/>
<circle cx="130.5" cy="114.3" r="4" class="fill-a" opacity="0.47"/>
<circle cx="130.5" cy="114.3" r="4" class="fill-c" opacity="0.16"/>
<circle cx="130.5" cy="114.3" r="4" class="fill-d" opacity="0.37"/>
<circle cx="154.9" cy="56.5" r="4" class="fill-a" opacity="0.89"/>
<circle cx="154.9" cy="56.5" r="4" class="fill-c" opacity="0.04"/>
<circle cx="154.9" cy="56.5" r="4" class="fill-d" opacity="0.07"/>
<circle cx="169.9" cy="76.1" r="4" class="fill-a" opacity="0.85"/>
<circle cx="169.9" cy="76.1" r="4" class="fill-c" opacity="0.07"/>
<circle cx="169.9" cy="76.1" r="4" class="fill-d" opacity="0.08"/>
<circle cx="148.8" cy="74.6" r="4" class="fill-a" opacity="0.81"/>
<circle cx="148.8" cy="74.6" r="4" class="fill-c" opacity="0.06"/>
<circle cx="148.8" cy="74.6" r="4" class="fill-d" opacity="0.13"/>
<circle cx="150.2" cy="64.3" r="4" class="fill-a" opacity="0.86"/>
<circle cx="150.2" cy="64.3" r="4" class="fill-c" opacity="0.05"/>
<circle cx="150.2" cy="64.3" r="4" class="fill-d" opacity="0.10"/>
<circle cx="139.4" cy="77.7" r="4" class="fill-a" opacity="0.76"/>
<circle cx="139.4" cy="77.7" r="4" class="fill-c" opacity="0.07"/>
<circle cx="139.4" cy="77.7" r="4" class="fill-d" opacity="0.17"/>
<circle cx="152.8" cy="69.7" r="4" class="fill-a" opacity="0.84"/>
<circle cx="152.8" cy="69.7" r="4" class="fill-c" opacity="0.05"/>
<circle cx="152.8" cy="69.7" r="4" class="fill-d" opacity="0.10"/>
<circle cx="159.4" cy="30.3" r="4" class="fill-a" opacity="0.95"/>
<circle cx="159.4" cy="30.3" r="4" class="fill-c" opacity="0.01"/>
<circle cx="159.4" cy="30.3" r="4" class="fill-d" opacity="0.03"/>
<circle cx="184.4" cy="51.0" r="4" class="fill-a" opacity="0.94"/>
<circle cx="184.4" cy="51.0" r="4" class="fill-c" opacity="0.03"/>
<circle cx="184.4" cy="51.0" r="4" class="fill-d" opacity="0.03"/>
<circle cx="115.0" cy="87.0" r="4" class="fill-a" opacity="0.60"/>
<circle cx="115.0" cy="87.0" r="4" class="fill-c" opacity="0.07"/>
<circle cx="115.0" cy="87.0" r="4" class="fill-d" opacity="0.33"/>
<circle cx="143.0" cy="69.9" r="4" class="fill-a" opacity="0.82"/>
<circle cx="143.0" cy="69.9" r="4" class="fill-c" opacity="0.05"/>
<circle cx="143.0" cy="69.9" r="4" class="fill-d" opacity="0.13"/>
<circle cx="110.2" cy="57.3" r="4" class="fill-a" opacity="0.76"/>
<circle cx="110.2" cy="57.3" r="4" class="fill-c" opacity="0.03"/>
<circle cx="110.2" cy="57.3" r="4" class="fill-d" opacity="0.21"/>
<circle cx="176.0" cy="105.9" r="4" class="fill-a" opacity="0.70"/>
<circle cx="176.0" cy="105.9" r="4" class="fill-c" opacity="0.17"/>
<circle cx="176.0" cy="105.9" r="4" class="fill-d" opacity="0.13"/>
<polyline points="252.7,65.6 252.3,58.8 251.3,52.1 249.5,45.6 247.1,39.3 244.0,33.3 240.3,27.6 236.1,22.3 231.3,17.5 226.0,13.3 220.4,9.6 214.3,6.5 208.0,4.1 201.5,2.3 194.8,1.3 188.0,0.9 181.3,1.3 174.6,2.3 168.0,4.1 161.7,6.5 155.7,9.6 150.0,13.3 144.8,17.5 140.0,22.3 135.7,27.6 132.0,33.3 129.0,39.3 126.5,45.6 124.8,52.1 123.7,58.8 123.4,65.6 123.7,72.3 124.8,79.0 126.5,85.6 129.0,91.9 132.0,97.9 135.7,103.6 140.0,108.8 144.8,113.6 150.0,117.9 155.7,121.6 161.7,124.6 168.0,127.1 174.6,128.8 181.3,129.9 188.0,130.2 194.8,129.9 201.5,128.8 208.0,127.1 214.3,124.6 220.4,121.6 226.0,117.9 231.3,113.6 236.1,108.8 240.3,103.6 244.0,97.9 247.1,91.9 249.5,85.6 251.3,79.0 252.3,72.3 252.7,65.6" fill="none" class="stroke-a" stroke-width="2.5"/>
<line x1="183.0" y1="60.6" x2="193.0" y2="70.6" class="stroke-ink" stroke-width="2"/><line x1="183.0" y1="70.6" x2="193.0" y2="60.6" class="stroke-ink" stroke-width="2"/>
<polyline points="253.3,221.0 253.0,214.3 251.9,207.6 250.2,201.1 247.8,194.8 244.7,188.7 241.0,183.0 236.7,177.8 232.0,173.0 226.7,168.7 221.0,165.1 215.0,162.0 208.7,159.6 202.1,157.8 195.5,156.7 188.7,156.4 181.9,156.7 175.3,157.8 168.7,159.6 162.4,162.0 156.4,165.1 150.7,168.7 145.4,173.0 140.6,177.8 136.4,183.0 132.7,188.7 129.6,194.8 127.2,201.1 125.5,207.6 124.4,214.3 124.0,221.0 124.4,227.8 125.5,234.5 127.2,241.0 129.6,247.3 132.7,253.4 136.4,259.1 140.6,264.3 145.4,269.1 150.7,273.4 156.4,277.0 162.4,280.1 168.7,282.5 175.3,284.3 181.9,285.3 188.7,285.7 195.5,285.3 202.1,284.3 208.7,282.5 215.0,280.1 221.0,277.0 226.7,273.4 232.0,269.1 236.7,264.3 241.0,259.1 244.7,253.4 247.8,247.3 250.2,241.0 251.9,234.5 253.0,227.8 253.3,221.0" fill="none" class="stroke-c" stroke-width="2.5"/>
<line x1="183.7" y1="216.0" x2="193.7" y2="226.0" class="stroke-ink" stroke-width="2"/><line x1="183.7" y1="226.0" x2="193.7" y2="216.0" class="stroke-ink" stroke-width="2"/>
<polyline points="141.2,183.4 140.9,176.7 139.8,170.0 138.1,163.4 135.6,157.1 132.6,151.1 128.9,145.4 124.6,140.1 119.8,135.4 114.6,131.1 108.9,127.4 102.9,124.3 96.6,121.9 90.0,120.2 83.3,119.1 76.6,118.8 69.8,119.1 63.1,120.2 56.6,121.9 50.3,124.3 44.3,127.4 38.6,131.1 33.3,135.4 28.5,140.1 24.3,145.4 20.6,151.1 17.5,157.1 15.1,163.4 13.3,170.0 12.3,176.7 11.9,183.4 12.3,190.2 13.3,196.9 15.1,203.4 17.5,209.7 20.6,215.7 24.3,221.4 28.5,226.7 33.3,231.5 38.6,235.7 44.3,239.4 50.3,242.5 56.6,244.9 63.1,246.7 69.8,247.7 76.6,248.1 83.3,247.7 90.0,246.7 96.6,244.9 102.9,242.5 108.9,239.4 114.6,235.7 119.8,231.5 124.6,226.7 128.9,221.4 132.6,215.7 135.6,209.7 138.1,203.4 139.8,196.9 140.9,190.2 141.2,183.4" fill="none" class="stroke-d" stroke-width="2.5"/>
<line x1="71.6" y1="178.4" x2="81.6" y2="188.4" class="stroke-ink" stroke-width="2"/><line x1="71.6" y1="188.4" x2="81.6" y2="178.4" class="stroke-ink" stroke-width="2"/>
</g>
<rect x="310.0" y="40.0" width="270.0" height="222.0" class="fill-sunk stroke-grid" />
<text x="310.0" y="32.0" class="fig-bold" text-anchor="start" >After 1 round</text>
<text x="580.0" y="32.0" class="fig-small" text-anchor="end" >log L = -259.9</text>
<g clip-path="url(#c025-p1)">
<circle cx="411.4" cy="253.3" r="4" class="fill-c" opacity="0.38"/>
<circle cx="411.4" cy="253.3" r="4" class="fill-d" opacity="0.62"/>
<circle cx="379.6" cy="214.2" r="4" class="fill-c" opacity="0.16"/>
<circle cx="379.6" cy="214.2" r="4" class="fill-d" opacity="0.84"/>
<circle cx="362.5" cy="207.3" r="4" class="fill-c" opacity="0.11"/>
<circle cx="362.5" cy="207.3" r="4" class="fill-d" opacity="0.89"/>
<circle cx="331.7" cy="207.6" r="4" class="fill-c" opacity="0.05"/>
<circle cx="331.7" cy="207.6" r="4" class="fill-d" opacity="0.95"/>
<circle cx="354.4" cy="137.8" r="4" class="fill-c" opacity="0.08"/>
<circle cx="354.4" cy="137.8" r="4" class="fill-d" opacity="0.92"/>
<circle cx="375.8" cy="210.0" r="4" class="fill-c" opacity="0.15"/>
<circle cx="375.8" cy="210.0" r="4" class="fill-d" opacity="0.85"/>
<circle cx="365.8" cy="216.2" r="4" class="fill-c" opacity="0.11"/>
<circle cx="365.8" cy="216.2" r="4" class="fill-d" opacity="0.89"/>
<circle cx="350.6" cy="210.8" r="4" class="fill-c" opacity="0.08"/>
<circle cx="350.6" cy="210.8" r="4" class="fill-d" opacity="0.92"/>
<circle cx="380.8" cy="207.8" r="4" class="fill-c" opacity="0.17"/>
<circle cx="380.8" cy="207.8" r="4" class="fill-d" opacity="0.83"/>
<circle cx="390.2" cy="207.0" r="4" class="fill-c" opacity="0.23"/>
<circle cx="390.2" cy="207.0" r="4" class="fill-d" opacity="0.77"/>
<circle cx="371.8" cy="172.7" r="4" class="fill-c" opacity="0.13"/>
<circle cx="371.8" cy="172.7" r="4" class="fill-d" opacity="0.86"/>
<circle cx="382.1" cy="213.0" r="4" class="fill-c" opacity="0.18"/>
<circle cx="382.1" cy="213.0" r="4" class="fill-d" opacity="0.82"/>
<circle cx="367.8" cy="192.5" r="4" class="fill-c" opacity="0.13"/>
<circle cx="367.8" cy="192.5" r="4" class="fill-d" opacity="0.87"/>
<circle cx="409.4" cy="208.4" r="4" class="fill-c" opacity="0.37"/>
<circle cx="409.4" cy="208.4" r="4" class="fill-d" opacity="0.62"/>
<circle cx="366.6" cy="183.4" r="4" class="fill-c" opacity="0.12"/>
<circle cx="366.6" cy="183.4" r="4" class="fill-d" opacity="0.88"/>
<circle cx="354.0" cy="208.8" r="4" class="fill-c" opacity="0.08"/>
<circle cx="354.0" cy="208.8" r="4" class="fill-d" opacity="0.92"/>
<circle cx="388.7" cy="191.7" r="4" class="fill-c" opacity="0.21"/>
<circle cx="388.7" cy="191.7" r="4" class="fill-d" opacity="0.79"/>
<circle cx="373.2" cy="189.9" r="4" class="fill-c" opacity="0.14"/>
<circle cx="373.2" cy="189.9" r="4" class="fill-d" opacity="0.86"/>
<circle cx="315.8" cy="183.0" r="4" class="fill-c" opacity="0.05"/>
<circle cx="315.8" cy="183.0" r="4" class="fill-d" opacity="0.95"/>
<circle cx="352.5" cy="235.9" r="4" class="fill-c" opacity="0.06"/>
<circle cx="352.5" cy="235.9" r="4" class="fill-d" opacity="0.94"/>
<circle cx="524.1" cy="189.3" r="4" class="fill-a" opacity="0.04"/>
<circle cx="524.1" cy="189.3" r="4" class="fill-c" opacity="0.95"/>
<circle cx="509.9" cy="224.2" r="4" class="fill-a" opacity="0.01"/>
<circle cx="509.9" cy="224.2" r="4" class="fill-c" opacity="0.98"/>
<circle cx="519.1" cy="204.1" r="4" class="fill-a" opacity="0.03"/>
<circle cx="519.1" cy="204.1" r="4" class="fill-c" opacity="0.97"/>
<circle cx="546.2" cy="188.4" r="4" class="fill-a" opacity="0.02"/>
<circle cx="546.2" cy="188.4" r="4" class="fill-c" opacity="0.98"/>
<circle cx="522.4" cy="181.3" r="4" class="fill-a" opacity="0.07"/>
<circle cx="522.4" cy="181.3" r="4" class="fill-c" opacity="0.92"/>
<circle cx="522.4" cy="181.3" r="4" class="fill-d" opacity="0.01"/>
<circle cx="514.6" cy="221.3" r="4" class="fill-a" opacity="0.01"/>
<circle cx="514.6" cy="221.3" r="4" class="fill-c" opacity="0.98"/>
<circle cx="530.1" cy="191.7" r="4" class="fill-a" opacity="0.03"/>
<circle cx="530.1" cy="191.7" r="4" class="fill-c" opacity="0.96"/>
<circle cx="514.4" cy="218.5" r="4" class="fill-a" opacity="0.02"/>
<circle cx="514.4" cy="218.5" r="4" class="fill-c" opacity="0.98"/>
<circle cx="523.1" cy="252.1" r="4" class="fill-c" opacity="0.99"/>
<circle cx="532.2" cy="193.4" r="4" class="fill-a" opacity="0.03"/>
<circle cx="532.2" cy="193.4" r="4" class="fill-c" opacity="0.97"/>
<circle cx="486.5" cy="201.9" r="4" class="fill-a" opacity="0.05"/>
<circle cx="486.5" cy="201.9" r="4" class="fill-c" opacity="0.91"/>
<circle cx="486.5" cy="201.9" r="4" class="fill-d" opacity="0.04"/>
<circle cx="499.7" cy="188.2" r="4" class="fill-a" opacity="0.09"/>
<circle cx="499.7" cy="188.2" r="4" class="fill-c" opacity="0.88"/>
<circle cx="499.7" cy="188.2" r="4" class="fill-d" opacity="0.03"/>
<circle cx="478.7" cy="221.0" r="4" class="fill-a" opacity="0.02"/>
<circle cx="478.7" cy="221.0" r="4" class="fill-c" opacity="0.93"/>
<circle cx="478.7" cy="221.0" r="4" class="fill-d" opacity="0.05"/>
<circle cx="532.6" cy="180.4" r="4" class="fill-a" opacity="0.05"/>
<circle cx="532.6" cy="180.4" r="4" class="fill-c" opacity="0.94"/>
<circle cx="476.3" cy="212.9" r="4" class="fill-a" opacity="0.03"/>
<circle cx="476.3" cy="212.9" r="4" class="fill-c" opacity="0.91"/>
<circle cx="476.3" cy="212.9" r="4" class="fill-d" opacity="0.06"/>
<circle cx="525.1" cy="215.1" r="4" class="fill-a" opacity="0.01"/>
<circle cx="525.1" cy="215.1" r="4" class="fill-c" opacity="0.98"/>
<circle cx="549.9" cy="226.5" r="4" class="fill-c" opacity="1.00"/>
<circle cx="525.6" cy="223.7" r="4" class="fill-c" opacity="0.99"/>
<circle cx="546.2" cy="203.5" r="4" class="fill-c" opacity="0.99"/>
<circle cx="511.3" cy="236.8" r="4" class="fill-c" opacity="0.99"/>
<circle cx="478.0" cy="65.6" r="4" class="fill-a" opacity="0.98"/>
<circle cx="478.0" cy="65.6" r="4" class="fill-d" opacity="0.02"/>
<circle cx="459.8" cy="58.0" r="4" class="fill-a" opacity="0.98"/>
<circle cx="459.8" cy="58.0" r="4" class="fill-d" opacity="0.02"/>
<circle cx="451.9" cy="92.9" r="4" class="fill-a" opacity="0.93"/>
<circle cx="451.9" cy="92.9" r="4" class="fill-d" opacity="0.06"/>
<circle cx="429.3" cy="96.1" r="4" class="fill-a" opacity="0.86"/>
<circle cx="429.3" cy="96.1" r="4" class="fill-c" opacity="0.01"/>
<circle cx="429.3" cy="96.1" r="4" class="fill-d" opacity="0.13"/>
<circle cx="471.9" cy="109.0" r="4" class="fill-a" opacity="0.89"/>
<circle cx="471.9" cy="109.0" r="4" class="fill-c" opacity="0.04"/>
<circle cx="471.9" cy="109.0" r="4" class="fill-d" opacity="0.07"/>
<circle cx="433.3" cy="86.7" r="4" class="fill-a" opacity="0.92"/>
<circle cx="433.3" cy="86.7" r="4" class="fill-d" opacity="0.07"/>
<circle cx="449.4" cy="69.1" r="4" class="fill-a" opacity="0.97"/>
<circle cx="449.4" cy="69.1" r="4" class="fill-d" opacity="0.03"/>
<circle cx="420.5" cy="114.3" r="4" class="fill-a" opacity="0.61"/>
<circle cx="420.5" cy="114.3" r="4" class="fill-c" opacity="0.05"/>
<circle cx="420.5" cy="114.3" r="4" class="fill-d" opacity="0.33"/>
<circle cx="444.9" cy="56.5" r="4" class="fill-a" opacity="0.98"/>
<circle cx="444.9" cy="56.5" r="4" class="fill-d" opacity="0.02"/>
<circle cx="459.9" cy="76.1" r="4" class="fill-a" opacity="0.97"/>
<circle cx="459.9" cy="76.1" r="4" class="fill-d" opacity="0.03"/>
<circle cx="438.8" cy="74.6" r="4" class="fill-a" opacity="0.96"/>
<circle cx="438.8" cy="74.6" r="4" class="fill-d" opacity="0.04"/>
<circle cx="440.2" cy="64.3" r="4" class="fill-a" opacity="0.97"/>
<circle cx="440.2" cy="64.3" r="4" class="fill-d" opacity="0.03"/>
<circle cx="429.4" cy="77.7" r="4" class="fill-a" opacity="0.94"/>
<circle cx="429.4" cy="77.7" r="4" class="fill-d" opacity="0.06"/>
<circle cx="442.8" cy="69.7" r="4" class="fill-a" opacity="0.97"/>
<circle cx="442.8" cy="69.7" r="4" class="fill-d" opacity="0.03"/>
<circle cx="449.4" cy="30.3" r="4" class="fill-a" opacity="0.99"/>
<circle cx="474.4" cy="51.0" r="4" class="fill-a" opacity="0.98"/>
<circle cx="474.4" cy="51.0" r="4" class="fill-d" opacity="0.01"/>
<circle cx="405.0" cy="87.0" r="4" class="fill-a" opacity="0.81"/>
<circle cx="405.0" cy="87.0" r="4" class="fill-d" opacity="0.19"/>
<circle cx="433.0" cy="69.9" r="4" class="fill-a" opacity="0.96"/>
<circle cx="433.0" cy="69.9" r="4" class="fill-d" opacity="0.04"/>
<circle cx="400.2" cy="57.3" r="4" class="fill-a" opacity="0.96"/>
<circle cx="400.2" cy="57.3" r="4" class="fill-d" opacity="0.04"/>
<circle cx="466.0" cy="105.9" r="4" class="fill-a" opacity="0.90"/>
<circle cx="466.0" cy="105.9" r="4" class="fill-c" opacity="0.03"/>
<circle cx="466.0" cy="105.9" r="4" class="fill-d" opacity="0.08"/>
<polyline points="424.8,100.7 422.8,96.0 421.2,91.3 419.8,86.5 418.8,81.8 418.1,77.1 417.8,72.5 417.9,68.2 418.3,64.0 419.1,60.1 420.2,56.5 421.7,53.3 423.5,50.5 425.6,48.1 427.9,46.1 430.5,44.6 433.4,43.6 436.4,43.0 439.6,43.0 442.9,43.4 446.3,44.4 449.8,45.8 453.3,47.7 456.7,50.1 460.1,52.9 463.4,56.0 466.5,59.6 469.5,63.4 472.3,67.5 474.8,71.9 477.1,76.4 479.1,81.0 480.8,85.8 482.1,90.6 483.2,95.3 483.8,100.0 484.1,104.5 484.1,108.9 483.6,113.1 482.9,117.0 481.7,120.5 480.3,123.7 478.5,126.6 476.4,129.0 474.0,131.0 471.4,132.5 468.6,133.5 465.5,134.1 462.3,134.1 459.0,133.6 455.6,132.7 452.1,131.2 448.7,129.3 445.2,127.0 441.8,124.2 438.6,121.0 435.4,117.5 432.4,113.7 429.7,109.6 427.1,105.2 424.8,100.7" fill="none" class="stroke-a" stroke-width="2.5"/>
<line x1="446.0" y1="83.5" x2="456.0" y2="93.5" class="stroke-ink" stroke-width="2"/><line x1="446.0" y1="93.5" x2="456.0" y2="83.5" class="stroke-ink" stroke-width="2"/>
<polyline points="482.6,236.9 476.5,235.7 470.6,234.1 464.9,232.2 459.5,230.0 454.3,227.5 449.5,224.6 445.2,221.5 441.3,218.2 438.0,214.8 435.2,211.1 432.9,207.4 431.3,203.6 430.3,199.8 429.9,196.0 430.2,192.3 431.1,188.7 432.7,185.2 434.8,181.9 437.5,178.8 440.8,176.0 444.6,173.4 448.9,171.2 453.7,169.3 458.8,167.7 464.2,166.5 469.9,165.7 475.7,165.3 481.8,165.3 487.9,165.7 494.0,166.5 500.0,167.7 505.9,169.2 511.6,171.1 517.1,173.4 522.2,175.9 527.0,178.7 531.4,181.8 535.2,185.1 538.6,188.6 541.4,192.2 543.6,195.9 545.3,199.7 546.2,203.5 546.6,207.3 546.3,211.1 545.4,214.7 543.9,218.2 541.7,221.5 539.0,224.6 535.7,227.4 531.9,230.0 527.6,232.2 522.9,234.1 517.8,235.7 512.4,236.8 506.7,237.6 500.8,238.0 494.8,238.0 488.7,237.7 482.6,236.9" fill="none" class="stroke-c" stroke-width="2.5"/>
<line x1="483.3" y1="196.7" x2="493.3" y2="206.7" class="stroke-ink" stroke-width="2"/><line x1="483.3" y1="206.7" x2="493.3" y2="196.7" class="stroke-ink" stroke-width="2"/>
<polyline points="413.3,212.9 409.2,216.6 404.9,220.0 400.4,223.0 395.7,225.6 390.9,227.7 386.1,229.4 381.2,230.6 376.4,231.2 371.7,231.4 367.1,231.1 362.8,230.3 358.7,228.9 354.9,227.1 351.4,224.9 348.4,222.1 345.7,219.0 343.5,215.6 341.7,211.7 340.4,207.6 339.6,203.3 339.3,198.7 339.5,194.0 340.3,189.2 341.5,184.4 343.2,179.6 345.4,174.8 348.0,170.1 351.1,165.6 354.5,161.3 358.2,157.3 362.3,153.6 366.6,150.2 371.1,147.2 375.8,144.6 380.6,142.5 385.5,140.8 390.3,139.7 395.1,139.0 399.8,138.8 404.4,139.1 408.7,140.0 412.8,141.3 416.6,143.1 420.1,145.4 423.2,148.1 425.8,151.2 428.1,154.7 429.8,158.5 431.1,162.6 431.9,166.9 432.2,171.5 432.0,176.2 431.2,181.0 430.0,185.8 428.3,190.7 426.1,195.4 423.5,200.1 420.5,204.6 417.1,208.9 413.3,212.9" fill="none" class="stroke-d" stroke-width="2.5"/>
<line x1="380.8" y1="180.1" x2="390.8" y2="190.1" class="stroke-ink" stroke-width="2"/><line x1="380.8" y1="190.1" x2="390.8" y2="180.1" class="stroke-ink" stroke-width="2"/>
</g>
<rect x="20.0" y="295.0" width="270.0" height="222.0" class="fill-sunk stroke-grid" />
<text x="20.0" y="287.0" class="fig-bold" text-anchor="start" >After 3 rounds</text>
<text x="290.0" y="287.0" class="fig-small" text-anchor="end" >log L = -222.7</text>
<g clip-path="url(#c025-p3)">
<circle cx="121.4" cy="508.3" r="4" class="fill-c" opacity="0.16"/>
<circle cx="121.4" cy="508.3" r="4" class="fill-d" opacity="0.84"/>
<circle cx="89.6" cy="469.2" r="4" class="fill-d" opacity="1.00"/>
<circle cx="72.5" cy="462.3" r="4" class="fill-d" opacity="1.00"/>
<circle cx="41.7" cy="462.6" r="4" class="fill-d" opacity="1.00"/>
<circle cx="64.4" cy="392.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="85.8" cy="465.0" r="4" class="fill-d" opacity="1.00"/>
<circle cx="75.8" cy="471.2" r="4" class="fill-d" opacity="1.00"/>
<circle cx="60.6" cy="465.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="90.8" cy="462.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="100.2" cy="462.0" r="4" class="fill-d" opacity="0.99"/>
<circle cx="81.8" cy="427.7" r="4" class="fill-d" opacity="1.00"/>
<circle cx="92.1" cy="468.0" r="4" class="fill-d" opacity="1.00"/>
<circle cx="77.8" cy="447.5" r="4" class="fill-d" opacity="1.00"/>
<circle cx="119.4" cy="463.4" r="4" class="fill-c" opacity="0.12"/>
<circle cx="119.4" cy="463.4" r="4" class="fill-d" opacity="0.88"/>
<circle cx="76.6" cy="438.4" r="4" class="fill-d" opacity="1.00"/>
<circle cx="64.0" cy="463.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="98.7" cy="446.7" r="4" class="fill-d" opacity="1.00"/>
<circle cx="83.2" cy="444.9" r="4" class="fill-d" opacity="1.00"/>
<circle cx="25.8" cy="438.0" r="4" class="fill-d" opacity="1.00"/>
<circle cx="62.5" cy="490.9" r="4" class="fill-d" opacity="1.00"/>
<circle cx="234.1" cy="444.3" r="4" class="fill-c" opacity="1.00"/>
<circle cx="219.9" cy="479.2" r="4" class="fill-c" opacity="1.00"/>
<circle cx="229.1" cy="459.1" r="4" class="fill-c" opacity="1.00"/>
<circle cx="256.2" cy="443.4" r="4" class="fill-c" opacity="1.00"/>
<circle cx="232.4" cy="436.3" r="4" class="fill-c" opacity="1.00"/>
<circle cx="224.6" cy="476.3" r="4" class="fill-c" opacity="1.00"/>
<circle cx="240.1" cy="446.7" r="4" class="fill-c" opacity="1.00"/>
<circle cx="224.4" cy="473.5" r="4" class="fill-c" opacity="1.00"/>
<circle cx="233.1" cy="507.1" r="4" class="fill-c" opacity="1.00"/>
<circle cx="242.2" cy="448.4" r="4" class="fill-c" opacity="1.00"/>
<circle cx="196.5" cy="456.9" r="4" class="fill-c" opacity="1.00"/>
<circle cx="209.7" cy="443.2" r="4" class="fill-c" opacity="1.00"/>
<circle cx="188.7" cy="476.0" r="4" class="fill-c" opacity="1.00"/>
<circle cx="242.6" cy="435.4" r="4" class="fill-c" opacity="1.00"/>
<circle cx="186.3" cy="467.9" r="4" class="fill-c" opacity="1.00"/>
<circle cx="235.1" cy="470.1" r="4" class="fill-c" opacity="1.00"/>
<circle cx="259.9" cy="481.5" r="4" class="fill-c" opacity="1.00"/>
<circle cx="235.6" cy="478.7" r="4" class="fill-c" opacity="1.00"/>
<circle cx="256.2" cy="458.5" r="4" class="fill-c" opacity="1.00"/>
<circle cx="221.3" cy="491.8" r="4" class="fill-c" opacity="1.00"/>
<circle cx="188.0" cy="320.6" r="4" class="fill-a" opacity="1.00"/>
<circle cx="169.8" cy="313.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="161.9" cy="347.9" r="4" class="fill-a" opacity="1.00"/>
<circle cx="139.3" cy="351.1" r="4" class="fill-a" opacity="1.00"/>
<circle cx="181.9" cy="364.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="143.3" cy="341.7" r="4" class="fill-a" opacity="1.00"/>
<circle cx="159.4" cy="324.1" r="4" class="fill-a" opacity="1.00"/>
<circle cx="130.5" cy="369.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="154.9" cy="311.5" r="4" class="fill-a" opacity="1.00"/>
<circle cx="169.9" cy="331.1" r="4" class="fill-a" opacity="1.00"/>
<circle cx="148.8" cy="329.6" r="4" class="fill-a" opacity="1.00"/>
<circle cx="150.2" cy="319.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="139.4" cy="332.7" r="4" class="fill-a" opacity="1.00"/>
<circle cx="152.8" cy="324.7" r="4" class="fill-a" opacity="1.00"/>
<circle cx="159.4" cy="285.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="184.4" cy="306.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="115.0" cy="342.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="143.0" cy="324.9" r="4" class="fill-a" opacity="1.00"/>
<circle cx="110.2" cy="312.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="176.0" cy="360.9" r="4" class="fill-a" opacity="1.00"/>
<polyline points="166.7,345.6 164.9,346.9 163.0,348.0 160.9,349.0 158.9,349.7 156.7,350.2 154.6,350.5 152.4,350.6 150.2,350.4 148.1,350.1 146.1,349.5 144.1,348.7 142.3,347.7 140.6,346.5 139.0,345.1 137.7,343.5 136.4,341.9 135.4,340.0 134.6,338.1 134.0,336.1 133.7,334.0 133.5,331.9 133.6,329.8 133.9,327.7 134.4,325.5 135.2,323.5 136.2,321.5 137.3,319.6 138.6,317.9 140.2,316.2 141.8,314.7 143.6,313.4 145.6,312.3 147.6,311.4 149.7,310.6 151.8,310.1 154.0,309.8 156.1,309.8 158.3,309.9 160.4,310.3 162.4,310.9 164.4,311.7 166.2,312.7 167.9,313.9 169.5,315.3 170.9,316.8 172.1,318.5 173.1,320.3 173.9,322.2 174.5,324.2 174.8,326.3 175.0,328.4 174.9,330.6 174.6,332.7 174.1,334.8 173.3,336.8 172.4,338.8 171.2,340.7 169.9,342.5 168.3,344.1 166.7,345.6" fill="none" class="stroke-a" stroke-width="2.5"/>
<line x1="149.3" y1="325.2" x2="159.3" y2="335.2" class="stroke-ink" stroke-width="2"/><line x1="149.3" y1="335.2" x2="159.3" y2="325.2" class="stroke-ink" stroke-width="2"/>
<polyline points="221.4,483.2 217.4,483.5 213.4,483.6 209.5,483.4 205.8,483.1 202.1,482.5 198.7,481.8 195.5,480.8 192.5,479.7 189.9,478.4 187.5,476.9 185.6,475.3 184.0,473.6 182.7,471.8 181.9,469.9 181.5,467.9 181.5,465.9 182.0,463.9 182.8,461.9 184.1,459.9 185.7,458.0 187.7,456.1 190.0,454.3 192.7,452.6 195.7,451.1 198.9,449.6 202.4,448.4 206.0,447.3 209.8,446.4 213.7,445.7 217.7,445.2 221.7,445.0 225.6,444.9 229.5,445.0 233.3,445.4 236.9,445.9 240.4,446.7 243.6,447.6 246.5,448.8 249.2,450.1 251.5,451.5 253.5,453.1 255.1,454.8 256.3,456.6 257.1,458.5 257.5,460.5 257.5,462.5 257.1,464.5 256.2,466.6 255.0,468.5 253.4,470.5 251.4,472.4 249.0,474.2 246.3,475.8 243.4,477.4 240.1,478.8 236.7,480.0 233.0,481.1 229.2,482.0 225.3,482.7 221.4,483.2" fill="none" class="stroke-c" stroke-width="2.5"/>
<line x1="214.5" y1="459.2" x2="224.5" y2="469.2" class="stroke-ink" stroke-width="2"/><line x1="214.5" y1="469.2" x2="224.5" y2="459.2" class="stroke-ink" stroke-width="2"/>
<polyline points="57.7,462.9 56.9,460.4 56.2,457.7 55.8,455.0 55.7,452.4 55.8,449.7 56.1,447.1 56.7,444.7 57.5,442.3 58.6,440.0 59.8,437.9 61.3,436.1 62.9,434.4 64.7,432.9 66.7,431.7 68.8,430.7 71.0,430.0 73.2,429.6 75.6,429.5 77.9,429.6 80.3,430.0 82.6,430.7 84.9,431.7 87.1,432.9 89.2,434.3 91.2,436.0 93.0,437.9 94.7,440.0 96.2,442.2 97.5,444.6 98.6,447.1 99.5,449.7 100.1,452.3 100.5,455.0 100.7,457.7 100.6,460.3 100.3,462.9 99.7,465.4 98.9,467.8 97.8,470.0 96.6,472.1 95.1,474.0 93.4,475.7 91.6,477.1 89.7,478.3 87.6,479.3 85.4,480.0 83.1,480.4 80.8,480.6 78.5,480.4 76.1,480.0 73.8,479.3 71.5,478.4 69.3,477.2 67.2,475.7 65.2,474.0 63.3,472.1 61.7,470.1 60.1,467.8 58.8,465.4 57.7,462.9" fill="none" class="stroke-d" stroke-width="2.5"/>
<line x1="73.2" y1="450.0" x2="83.2" y2="460.0" class="stroke-ink" stroke-width="2"/><line x1="73.2" y1="460.0" x2="83.2" y2="450.0" class="stroke-ink" stroke-width="2"/>
</g>
<rect x="310.0" y="295.0" width="270.0" height="222.0" class="fill-sunk stroke-grid" />
<text x="310.0" y="287.0" class="fig-bold" text-anchor="start" >After 10 rounds</text>
<text x="580.0" y="287.0" class="fig-small" text-anchor="end" >log L = -215.8</text>
<g clip-path="url(#c025-p10)">
<circle cx="411.4" cy="508.3" r="4" class="fill-d" opacity="1.00"/>
<circle cx="379.6" cy="469.2" r="4" class="fill-d" opacity="1.00"/>
<circle cx="362.5" cy="462.3" r="4" class="fill-d" opacity="1.00"/>
<circle cx="331.7" cy="462.6" r="4" class="fill-d" opacity="1.00"/>
<circle cx="354.4" cy="392.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="375.8" cy="465.0" r="4" class="fill-d" opacity="1.00"/>
<circle cx="365.8" cy="471.2" r="4" class="fill-d" opacity="1.00"/>
<circle cx="350.6" cy="465.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="380.8" cy="462.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="390.2" cy="462.0" r="4" class="fill-d" opacity="1.00"/>
<circle cx="371.8" cy="427.7" r="4" class="fill-d" opacity="1.00"/>
<circle cx="382.1" cy="468.0" r="4" class="fill-d" opacity="1.00"/>
<circle cx="367.8" cy="447.5" r="4" class="fill-d" opacity="1.00"/>
<circle cx="409.4" cy="463.4" r="4" class="fill-d" opacity="1.00"/>
<circle cx="366.6" cy="438.4" r="4" class="fill-d" opacity="1.00"/>
<circle cx="354.0" cy="463.8" r="4" class="fill-d" opacity="1.00"/>
<circle cx="388.7" cy="446.7" r="4" class="fill-d" opacity="1.00"/>
<circle cx="373.2" cy="444.9" r="4" class="fill-d" opacity="1.00"/>
<circle cx="315.8" cy="438.0" r="4" class="fill-d" opacity="1.00"/>
<circle cx="352.5" cy="490.9" r="4" class="fill-d" opacity="1.00"/>
<circle cx="524.1" cy="444.3" r="4" class="fill-c" opacity="1.00"/>
<circle cx="509.9" cy="479.2" r="4" class="fill-c" opacity="1.00"/>
<circle cx="519.1" cy="459.1" r="4" class="fill-c" opacity="1.00"/>
<circle cx="546.2" cy="443.4" r="4" class="fill-c" opacity="1.00"/>
<circle cx="522.4" cy="436.3" r="4" class="fill-c" opacity="1.00"/>
<circle cx="514.6" cy="476.3" r="4" class="fill-c" opacity="1.00"/>
<circle cx="530.1" cy="446.7" r="4" class="fill-c" opacity="1.00"/>
<circle cx="514.4" cy="473.5" r="4" class="fill-c" opacity="1.00"/>
<circle cx="523.1" cy="507.1" r="4" class="fill-c" opacity="1.00"/>
<circle cx="532.2" cy="448.4" r="4" class="fill-c" opacity="1.00"/>
<circle cx="486.5" cy="456.9" r="4" class="fill-c" opacity="1.00"/>
<circle cx="499.7" cy="443.2" r="4" class="fill-c" opacity="1.00"/>
<circle cx="478.7" cy="476.0" r="4" class="fill-c" opacity="1.00"/>
<circle cx="532.6" cy="435.4" r="4" class="fill-c" opacity="1.00"/>
<circle cx="476.3" cy="467.9" r="4" class="fill-c" opacity="1.00"/>
<circle cx="525.1" cy="470.1" r="4" class="fill-c" opacity="1.00"/>
<circle cx="549.9" cy="481.5" r="4" class="fill-c" opacity="1.00"/>
<circle cx="525.6" cy="478.7" r="4" class="fill-c" opacity="1.00"/>
<circle cx="546.2" cy="458.5" r="4" class="fill-c" opacity="1.00"/>
<circle cx="511.3" cy="491.8" r="4" class="fill-c" opacity="1.00"/>
<circle cx="478.0" cy="320.6" r="4" class="fill-a" opacity="1.00"/>
<circle cx="459.8" cy="313.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="451.9" cy="347.9" r="4" class="fill-a" opacity="1.00"/>
<circle cx="429.3" cy="351.1" r="4" class="fill-a" opacity="1.00"/>
<circle cx="471.9" cy="364.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="433.3" cy="341.7" r="4" class="fill-a" opacity="1.00"/>
<circle cx="449.4" cy="324.1" r="4" class="fill-a" opacity="1.00"/>
<circle cx="420.5" cy="369.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="444.9" cy="311.5" r="4" class="fill-a" opacity="1.00"/>
<circle cx="459.9" cy="331.1" r="4" class="fill-a" opacity="1.00"/>
<circle cx="438.8" cy="329.6" r="4" class="fill-a" opacity="1.00"/>
<circle cx="440.2" cy="319.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="429.4" cy="332.7" r="4" class="fill-a" opacity="1.00"/>
<circle cx="442.8" cy="324.7" r="4" class="fill-a" opacity="1.00"/>
<circle cx="449.4" cy="285.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="474.4" cy="306.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="405.0" cy="342.0" r="4" class="fill-a" opacity="1.00"/>
<circle cx="433.0" cy="324.9" r="4" class="fill-a" opacity="1.00"/>
<circle cx="400.2" cy="312.3" r="4" class="fill-a" opacity="1.00"/>
<circle cx="466.0" cy="360.9" r="4" class="fill-a" opacity="1.00"/>
<polyline points="457.2,345.2 455.5,346.6 453.6,347.9 451.6,349.0 449.6,349.9 447.4,350.5 445.3,351.0 443.1,351.2 440.9,351.2 438.8,351.0 436.7,350.6 434.7,349.9 432.8,349.0 431.0,348.0 429.4,346.7 427.9,345.3 426.6,343.7 425.5,341.9 424.6,340.0 423.9,338.1 423.4,336.0 423.1,333.9 423.1,331.8 423.3,329.6 423.7,327.4 424.4,325.3 425.2,323.3 426.3,321.3 427.6,319.4 429.0,317.7 430.6,316.1 432.3,314.6 434.2,313.3 436.2,312.2 438.3,311.4 440.4,310.7 442.5,310.2 444.7,310.0 446.9,310.0 449.0,310.2 451.1,310.7 453.1,311.3 455.0,312.2 456.8,313.3 458.4,314.5 459.9,316.0 461.2,317.6 462.3,319.3 463.2,321.2 463.9,323.1 464.4,325.2 464.7,327.3 464.7,329.5 464.5,331.6 464.1,333.8 463.4,335.9 462.6,337.9 461.5,339.9 460.3,341.8 458.8,343.5 457.2,345.2" fill="none" class="stroke-a" stroke-width="2.5"/>
<line x1="438.9" y1="325.6" x2="448.9" y2="335.6" class="stroke-ink" stroke-width="2"/><line x1="438.9" y1="335.6" x2="448.9" y2="325.6" class="stroke-ink" stroke-width="2"/>
<polyline points="528.9,478.0 527.1,479.3 525.2,480.3 523.1,481.2 521.1,481.9 519.0,482.4 516.9,482.7 514.8,482.8 512.7,482.7 510.7,482.3 508.8,481.8 507.0,481.1 505.3,480.2 503.8,479.0 502.4,477.8 501.2,476.3 500.2,474.8 499.4,473.1 498.8,471.3 498.4,469.4 498.3,467.5 498.3,465.5 498.6,463.5 499.1,461.5 499.8,459.5 500.7,457.6 501.8,455.8 503.1,454.0 504.5,452.3 506.1,450.8 507.9,449.4 509.7,448.1 511.7,447.1 513.7,446.2 515.7,445.5 517.8,445.0 520.0,444.7 522.0,444.6 524.1,444.7 526.1,445.1 528.0,445.6 529.8,446.3 531.5,447.3 533.0,448.4 534.4,449.7 535.6,451.1 536.6,452.6 537.4,454.3 538.0,456.1 538.4,458.0 538.5,459.9 538.5,461.9 538.2,463.9 537.7,465.9 537.0,467.9 536.1,469.8 535.0,471.7 533.7,473.4 532.3,475.1 530.7,476.6 528.9,478.0" fill="none" class="stroke-c" stroke-width="2.5"/>
<line x1="513.4" y1="458.7" x2="523.4" y2="468.7" class="stroke-ink" stroke-width="2"/><line x1="513.4" y1="468.7" x2="523.4" y2="458.7" class="stroke-ink" stroke-width="2"/>
<polyline points="355.7,470.2 353.9,468.1 352.2,465.9 350.8,463.6 349.6,461.2 348.5,458.8 347.7,456.4 347.2,454.0 346.9,451.7 346.8,449.4 346.9,447.2 347.4,445.1 348.0,443.1 348.9,441.3 350.0,439.7 351.3,438.3 352.8,437.1 354.5,436.1 356.4,435.3 358.4,434.8 360.5,434.6 362.7,434.6 365.0,434.8 367.3,435.3 369.7,436.1 372.0,437.0 374.3,438.2 376.6,439.6 378.8,441.3 380.9,443.0 382.8,445.0 384.6,447.1 386.2,449.3 387.7,451.6 388.9,453.9 389.9,456.3 390.7,458.7 391.3,461.1 391.6,463.5 391.7,465.8 391.5,468.0 391.1,470.1 390.4,472.0 389.6,473.8 388.5,475.4 387.1,476.9 385.6,478.1 383.9,479.1 382.1,479.8 380.1,480.3 377.9,480.6 375.7,480.6 373.5,480.3 371.1,479.8 368.8,479.1 366.4,478.1 364.1,476.9 361.9,475.5 359.7,473.9 357.6,472.1 355.7,470.2" fill="none" class="stroke-d" stroke-width="2.5"/>
<line x1="364.2" y1="452.6" x2="374.2" y2="462.6" class="stroke-ink" stroke-width="2"/><line x1="364.2" y1="462.6" x2="374.2" y2="452.6" class="stroke-ink" stroke-width="2"/>
</g>
</svg>
Figure 3. EM from a deliberately plain start: three equal round blobs. Each round moves the blobs onto the clusters and shrinks them to fit, and log L climbs from -292.7 to -215.8. Ellipses are one-sigma contours, crosses are the blob means, and points are coloured by their responsibilities.
```

At the start the blobs are fat circles in the wrong places and $\log L=$ -292.7. After one round it is -259.9, after three -222.7, and after ten -215.8, when the ellipses sit on the clusters and stop moving. Each round the shares sharpen, which lets the blobs fit tighter, which sharpens the shares again.

Look at Figure 3 and you may suspect the numbers always climb. They do, and here is why. The M-step improves a different score than $\log L$: a **bound** that sits underneath it. Picture $\log L$ as a hill over the parameters. The E-step builds a second curve, the bound, that lies under the hill everywhere and touches it exactly at the current parameters. The M-step climbs to the top of the bound, which is at least as high as where it started. The hill is at least as high as the bound at that new point. So the hill is higher than before. Figure 4 shows this with the two-point numbers.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="A curve for log L and a lower curve for the bound that touches it at the old parameter; the M-step moves to the top of the bound, where log L is higher still">
<defs></defs>
<line x1="190.0" y1="250.0" x2="570.0" y2="250.0" class="stroke-muted" />
<line x1="190.0" y1="40.0" x2="190.0" y2="250.0" class="stroke-muted" />
<line x1="190.0" y1="236.9" x2="570.0" y2="236.9" class="stroke-grid" />
<line x1="186.0" y1="236.9" x2="190.0" y2="236.9" class="stroke-muted" />
<text x="183.0" y="240.9" class="fig-small" text-anchor="end" >-3.5</text>
<line x1="190.0" y1="171.2" x2="570.0" y2="171.2" class="stroke-grid" />
<line x1="186.0" y1="171.2" x2="190.0" y2="171.2" class="stroke-muted" />
<text x="183.0" y="175.2" class="fig-small" text-anchor="end" >-3.0</text>
<line x1="190.0" y1="105.6" x2="570.0" y2="105.6" class="stroke-grid" />
<line x1="186.0" y1="105.6" x2="190.0" y2="105.6" class="stroke-muted" />
<text x="183.0" y="109.6" class="fig-small" text-anchor="end" >-2.5</text>
<line x1="190.0" y1="40.0" x2="570.0" y2="40.0" class="stroke-grid" />
<line x1="186.0" y1="40.0" x2="190.0" y2="40.0" class="stroke-muted" />
<text x="183.0" y="44.0" class="fig-small" text-anchor="end" >-2.0</text>
<text x="380.0" y="24.0" class="fig-bold" text-anchor="middle" >log L and the bound under it, over one parameter</text>
<text x="380.0" y="274.0" class="fig-small" text-anchor="middle" >the parameter (schematic shapes, true heights)</text>
<polyline points="190.0,229.0 192.1,226.7 194.2,224.4 196.3,222.1 198.4,219.9 200.6,217.7 202.7,215.4 204.8,213.3 206.9,211.1 209.0,208.9 211.1,206.8 213.2,204.6 215.3,202.5 217.4,200.4 219.6,198.4 221.7,196.3 223.8,194.3 225.9,192.2 228.0,190.2 230.1,188.2 232.2,186.2 234.3,184.3 236.4,182.3 238.6,180.4 240.7,178.5 242.8,176.6 244.9,174.7 247.0,172.8 249.1,171.0 251.2,169.2 253.3,167.4 255.4,165.6 257.6,163.8 259.7,162.0 261.8,160.3 263.9,158.5 266.0,156.8 268.1,155.1 270.2,153.4 272.3,151.8 274.4,150.1 276.6,148.5 278.7,146.9 280.8,145.3 282.9,143.7 285.0,142.1 287.1,140.6 289.2,139.0 291.3,137.5 293.4,136.0 295.6,134.5 297.7,133.1 299.8,131.6 301.9,130.2 304.0,128.8 306.1,127.4 308.2,126.0 310.3,124.6 312.4,123.2 314.6,121.9 316.7,120.6 318.8,119.3 320.9,118.0 323.0,116.7 325.1,115.5 327.2,114.2 329.3,113.0 331.4,111.8 333.6,110.6 335.7,109.5 337.8,108.3 339.9,107.2 342.0,106.0 344.1,104.9 346.2,103.8 348.3,102.8 350.4,101.7 352.6,100.7 354.7,99.7 356.8,98.7 358.9,97.7 361.0,96.7 363.1,95.7 365.2,94.8 367.3,93.9 369.4,93.0 371.6,92.1 373.7,91.2 375.8,90.3 377.9,89.5 380.0,88.7 382.1,87.9 384.2,87.1 386.3,86.3 388.4,85.5 390.6,84.8 392.7,84.1 394.8,83.4 396.9,82.7 399.0,82.0 401.1,81.3 403.2,80.7 405.3,80.1 407.4,79.4 409.6,78.8 411.7,78.3 413.8,77.7 415.9,77.2 418.0,76.6 420.1,76.1 422.2,75.6 424.3,75.1 426.4,74.7 428.6,74.2 430.7,73.8 432.8,73.4 434.9,73.0 437.0,72.6 439.1,72.3 441.2,71.9 443.3,71.6 445.4,71.3 447.6,71.0 449.7,70.7 451.8,70.4 453.9,70.2 456.0,69.9 458.1,69.7 460.2,69.5 462.3,69.3 464.4,69.2 466.6,69.0 468.7,68.9 470.8,68.8 472.9,68.7 475.0,68.6 477.1,68.5 479.2,68.5 481.3,68.4 483.4,68.4 485.6,68.4 487.7,68.4 489.8,68.5 491.9,68.5 494.0,68.6 496.1,68.7 498.2,68.8 500.3,68.9 502.4,69.0 504.6,69.2 506.7,69.3 508.8,69.5 510.9,69.7 513.0,69.9 515.1,70.1 517.2,70.4 519.3,70.7 521.4,70.9 523.6,71.2 525.7,71.5 527.8,71.9 529.9,72.2 532.0,72.6 534.1,73.0 536.2,73.4 538.3,73.8 540.4,74.2 542.6,74.6 544.7,75.1 546.8,75.6 548.9,76.1 551.0,76.6 553.1,77.1 555.2,77.6 557.3,78.2 559.4,78.8 561.6,79.4 563.7,80.0 565.8,80.6 567.9,81.3 570.0,81.9" fill="none" class="stroke-ink" stroke-width="2.5"/>
<polyline points="190.0,237.8 192.1,234.9 194.2,232.1 196.3,229.3 198.4,226.5 200.6,223.8 202.7,221.1 204.8,218.4 206.9,215.8 209.0,213.2 211.1,210.7 213.2,208.2 215.3,205.7 217.4,203.3 219.6,200.9 221.7,198.5 223.8,196.2 225.9,193.9 228.0,191.6 230.1,189.4 232.2,187.2 234.3,185.1 236.4,183.0 238.6,180.9 240.7,178.8 242.8,176.8 244.9,174.9 247.0,172.9 249.1,171.0 251.2,169.2 253.3,167.4 255.4,165.6 257.6,163.8 259.7,162.1 261.8,160.4 263.9,158.8 266.0,157.2 268.1,155.6 270.2,154.1 272.3,152.6 274.4,151.1 276.6,149.7 278.7,148.3 280.8,146.9 282.9,145.6 285.0,144.3 287.1,143.1 289.2,141.9 291.3,140.7 293.4,139.6 295.6,138.5 297.7,137.4 299.8,136.4 301.9,135.4 304.0,134.4 306.1,133.5 308.2,132.6 310.3,131.8 312.4,130.9 314.6,130.2 316.7,129.4 318.8,128.7 320.9,128.1 323.0,127.4 325.1,126.8 327.2,126.3 329.3,125.7 331.4,125.3 333.6,124.8 335.7,124.4 337.8,124.0 339.9,123.7 342.0,123.4 344.1,123.1 346.2,122.9 348.3,122.7 350.4,122.5 352.6,122.4 354.7,122.3 356.8,122.2 358.9,122.2 361.0,122.2 363.1,122.3 365.2,122.4 367.3,122.5 369.4,122.7 371.6,122.9 373.7,123.1 375.8,123.4 377.9,123.7 380.0,124.0 382.1,124.4 384.2,124.8 386.3,125.3 388.4,125.7 390.6,126.3 392.7,126.8 394.8,127.4 396.9,128.1 399.0,128.7 401.1,129.4 403.2,130.2 405.3,130.9 407.4,131.8 409.6,132.6 411.7,133.5 413.8,134.4 415.9,135.4 418.0,136.4 420.1,137.4 422.2,138.5 424.3,139.6 426.4,140.7 428.6,141.9 430.7,143.1 432.8,144.3 434.9,145.6 437.0,146.9 439.1,148.3 441.2,149.7 443.3,151.1 445.4,152.6 447.6,154.1 449.7,155.6 451.8,157.2 453.9,158.8 456.0,160.4 458.1,162.1 460.2,163.8 462.3,165.6 464.4,167.4 466.6,169.2 468.7,171.0 470.8,172.9 472.9,174.9 475.0,176.8 477.1,178.8 479.2,180.9 481.3,183.0 483.4,185.1 485.6,187.2 487.7,189.4 489.8,191.6 491.9,193.9 494.0,196.2 496.1,198.5 498.2,200.9 500.3,203.3 502.4,205.7 504.6,208.2 506.7,210.7 508.8,213.2 510.9,215.8 513.0,218.4 515.1,221.1 517.2,223.8 519.3,226.5 521.4,229.3 523.6,232.1 525.7,234.9 527.8,237.8 529.9,240.7 532.0,243.6 534.1,246.6" fill="none" class="stroke-b" stroke-width="2.5"/>
<text x="522.5" y="59.1" class="fig-bold" text-anchor="start" >log L</text>
<text x="525.9" y="221.1" class="fig-bold" text-anchor="start" >the bound</text>
<line x1="358.9" y1="122.2" x2="358.9" y2="97.7" class="stroke-muted" stroke-width="2"/>
<circle cx="253.3" cy="167.4" r="6" class="fill-ink" />
<circle cx="358.9" cy="122.2" r="6" class="fill-b" />
<circle cx="358.9" cy="97.7" r="6" class="fill-ink" />
<text x="348.9" y="89.7" class="fig-small" text-anchor="end" >new log L = -2.439</text>
<line x1="253.3" y1="175.0" x2="253.3" y2="222.0" class="stroke-muted" />
<text x="262.0" y="196.0" class="fig-small" text-anchor="start" >old parameter:</text>
<text x="262.0" y="212.0" class="fig-small" text-anchor="start" >log L = -2.970,</text>
<text x="262.0" y="228.0" class="fig-small" text-anchor="start" >the bound touches it</text>
<text x="372.0" y="190.0" class="fig-small" text-anchor="start" >bound after the</text>
<text x="372.0" y="206.0" class="fig-small" text-anchor="start" >M-step: -2.626</text>
<line x1="364.9" y1="130.2" x2="364.9" y2="210.0" class="stroke-muted" />
</svg>
Figure 4. The E-step builds the orange bound so that it touches log L at the old parameter. The M-step climbs to the top of the bound, from -2.970 to -2.626. Log L at that new parameter, -2.439, is at least as high as the bound, so log L cannot have fallen.
```

Now the formula, as a summary of the picture. For shares $r$,

$$
\text{bound}(r) = \sum_{i=1}^{n} \sum_{j=1}^{k} r_{ij} \left( \log\pi_j + \log \mathrm{N}(x_i; \mu_j, \Sigma_j) - \log r_{ij} \right) .
$$

Read it aloud. For each point and blob, take the log score from the E-step, subtract the log of the share, and weight by the share. The difference $\log L-\text{bound}$ is the **gap**. It is never negative, because the shares you plug in are a guess about the hidden choices, and no guess can beat the true posterior shares. It is exactly 0 when the shares are the posterior shares the E-step computes. The standard names are ELBO for the bound and KL divergence for the gap. That is why the E-step makes the bound touch. The chain is then:

$$
\log L(\text{new}) \ \ge\ \text{bound}(\text{new}) \ \ge\ \text{bound}(\text{old}) \ =\ \log L(\text{old}).
$$

The first step holds because the gap is not negative. The second holds because the weighted mean and covariance are the values that make the bound as large as possible for fixed shares (with $\epsilon=0$, and a tiny $\epsilon$ only changes the last digits). The last holds because the E-step made the gap 0.

Check the numbers. At the E-step each point gives $-1.308-0.177=-1.485$, so the bound is $-2.970$, equal to $\log L$. After the M-step, with the same shares, the bound is $-2.626$. The true $\log L$ at the new blobs is $-2.439$, above the bound by a gap of $0.187$. The next E-step resets the gap to 0 at $-2.439$. In numpy, build the table of log scores and weight it:

```python
log_r = np.log(r, out=np.zeros_like(r), where=r > 0)
bound = np.sum(r * (scores - log_r))
```

> [!WARNING]
> A share can be exactly 0 when a blob is far away, and $0\cdot\log 0$ is `nan` in numpy. Take the log of $r$ only where $r>0$, as in the code above, since such a term contributes 0.

> [!KEY]
> One E-step then one M-step never lowers $\log L$. A drop in the history means a bug, most often a covariance measured from the wrong centre.

## Tiny blobs give k-means

Give every blob equal weight and the covariance $\sigma^2 I$. The share of a blob then depends on the gap in squared distances divided by $2\sigma^2$. Take a point at $x=1$ with blobs at $0$ and $4$. The squared distances are $1$ and $9$, so the score gap is $(9-1)/(2\sigma^2)=4/\sigma^2$, and the far blob's share is $1/(1+e^{4/\sigma^2})$.

| $\sigma^2$ | Score gap | Far blob's share |
|---|---|---|
| 1 | 4 | 0.018 |
| 0.1 | 40 | 4.2e-18 |
| 0.01 | 400 | 1.9e-174 |
| $10^{-6}$ | 4,000,000 | 0 (underflows) |

As $\sigma^2$ shrinks, any gap in distance becomes an enormous gap in score, so the nearest blob takes all the weight. With hard shares the M-step's mean is a plain group mean, which is the k-means update. On the 60 points, take the Lloyd centres, give them covariances $10^{-6}I$, and run the E-step: the nearest-centre assignment is reproduced for 60 of 60 points, and the M-step returns the Lloyd centres. Started from the same three centres, EM's $\log L$ climbs from -292.7 to -215.8 and Lloyd's objective $J$ falls from 198 to 88.5, and neither ever reverses. They are the same idea at two settings: guess the hidden assignment, then refit.

> [!RECAP]
> - A point's responsibilities are its weighted heights under each blob, rescaled to add to 1. The log density is $-\tfrac12(d\log 2\pi + \log\det\Sigma + \text{squared distance with }\Sigma^{-1})$, with `slogdet` for the determinant.
> - Subtract each row's maximum before the exponential, and add it back after the log. $\log L$ is the sum over points of the log of each point's row total.
> - The M-step is the ordinary weight, mean, and covariance with each point counted by its share, using the new mean in the covariance.
> - The bound sums share times (log score minus log share). It touches $\log L$ at the posterior shares, so $\log L$ never falls.
> - Start EM from k-means++ means, equal round covariances, and equal weights, and run the E-step first. Equal blobs with $\sigma^2\to 0$ give k-means.

## Build it step by step

The k-means helpers, including `kmeans_plusplus_init`, are already in the file. You write six functions, each built from the ones before.

1. **`log_gaussian(X, mean, cov)`** takes `(n, d)`, `(d,)`, and `(d, d)` and returns the log heights, shape `(n,)`. Subtract the mean, get `logdet` from `slogdet`, and compute the squared distance row by row with a product by the inverse and a sum over axis 1. The usual mistake is `gap @ inv @ gap.T`, which builds an `(n, n)` matrix of every pair when you want only the diagonal.
2. **`logsumexp_rows(s)`** takes scores of shape `(n, k)` and returns `(n,)`. Subtract the row maximum, exponentiate, sum, take the log, and add the maximum back. Do not modify `s`. The usual mistake is to exponentiate first.
3. **`gmm_e_step(X, means, covs, weights)`** returns the responsibilities `(n, k)` and the log-likelihood as a Python float. Fill an `(n, k)` score table with `np.log(weights[j]) + log_gaussian(...)`, call `logsumexp_rows`, and subtract the totals from the scores before the exponential. The usual mistake is to put `weights[j]` in front of the log density without taking its log.
4. **`gmm_m_step(X, responsibilities, reg_covar)`** returns `means` `(k, d)`, `covs` `(k, d, d)`, and `weights` `(k,)`. Compute `counts = responsibilities.sum(axis=0)`, then the weights, the means, and for each blob the weighted outer-product sum around its new mean, divided by its count, plus `reg_covar * np.eye(d)`. The usual mistake is dividing by $n$ instead of $N_j$.
5. **`gmm_elbo(X, responsibilities, means, covs, weights)`** returns the bound as a Python float. Build the `(n, k)` table `log(weights[j]) + log_gaussian(...)`, subtract $\log r$ taken only where $r>0$, multiply by $r$, and sum. The usual mistake is `np.log(r)` on a zero share.
6. **`fit_gmm(X, k, iters, rng, reg_covar)`** returns `(means, covs, weights, responsibilities, history)`. Seed the means with `kmeans_plusplus_init`, give every covariance `np.var(X, axis=0).mean() * np.eye(d)`, and use weights of `1 / k`. Run the E-step and record $\log L$, then repeat `iters` times: M-step, E-step, record. The history has `iters + 1` entries. The usual mistake is running the M-step before any E-step, when the shares do not exist yet.

## Cost and scaling

Each E-step needs an inverse and a log determinant per blob, and a product with the inverse for each point. A round of EM costs about $n k d^2$ operations for the shares and the covariances, plus $k d^3$ for the inverses, against $nkd$ for k-means assignment. The memory holds the `(n, k)` shares and $k$ covariances of $d^2$ numbers each. Diagonal covariances cut the covariance memory to $d$ numbers per blob.

## Where this breaks

- **A tilted blob needs many numbers.** A covariance matrix has about $d^2/2$ free entries, so it grows with the square of the dimension, and with few points per blob the estimate is noisy. Diagonal covariances are a cheaper compromise.
- **A far blob underflows without logs.** That is the failure of the naive exponential, and it appears as soon as the data have a few dimensions.
- **A blob can collapse onto one point.** In one dimension, a blob centred on a data point with variance $\sigma^2$ has height $1/\sqrt{2\pi\sigma^2}$ at that point, which is about 399 at $\sigma^2=10^{-6}$ and about 398,942 at $10^{-12}$. The log-likelihood goes to infinity with a useless blob. The $\epsilon I$ in the M-step, `reg_covar`, stops it.
- **EM can start slowly or stop at a bump.** Like Lloyd, it only climbs, so a poor start can take many rounds or end at a worse fit. Seed with k-means++ and keep the run with the best $\log L$.
- **You choose the number of blobs.** More blobs always raise the training $\log L$. Score a held-out set instead, and keep the $k$ where it stops improving.
- **Scales matter.** A column in grams next to one in kilometres dominates every distance. Standardise the columns first.

## The exercise

You build `log_gaussian`, `logsumexp_rows`, `gmm_e_step`, `gmm_m_step`, `gmm_elbo`, and `fit_gmm`. The checks confirm the log height $-0.919$ at the peak of a standard normal, match a direct formula for a tilted two-dimensional covariance, and confirm a diagonal covariance equals the sum of one-dimensional log heights. They push `logsumexp_rows` to scores near $-1000$, and check that `gmm_e_step` reproduces the shares $0.8808$ and $0.1192$ with $\log L=-2.970$, obeys Bayes' rule with unequal weights, and stays finite for blobs 1000 units apart. They reproduce the two-point refit, with means $0.2384$ and $1.7616$ and variance $0.42$, and compare the weighted mean and covariance with numpy's weighted statistics. They confirm the bound equals $\log L$ at the posterior shares, sits below it for other shares, and stays finite when a share is 0. Finally they confirm that `fit_gmm` returns a history of length `iters + 1` that never falls, and recovers the three planted blobs.
