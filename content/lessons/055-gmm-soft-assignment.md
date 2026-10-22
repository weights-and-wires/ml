---
title: Gaussian mixtures and soft assignment
slug: 055-gmm-soft-assignment
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: When blobs overlap, a point cannot honestly belong to one cluster. A Gaussian mixture computes how likely each blob is to have produced each point.
minutes: 45
gpu: false
objectives:
  - Describe a Gaussian mixture as a weighted die that picks a blob, followed by a draw from that blob's bell.
  - Compute a point's responsibilities with Bayes' rule, by hand for two blobs and in code for any number.
  - Read a covariance matrix as the shape of an ellipse, and compute the log density of a Gaussian with it.
  - Explain why the likelihood is handled in logs, and compute a row-wise log-sum-exp that survives scores near -1000.
prereqs:
  - 028-bayes-rule-classifier
  - 029-gaussian-discriminant-analysis
  - 054-kmeans-plusplus-seeding
exercise: 055-gmm-soft-assignment
---

# Gaussian mixtures and soft assignment

A customer lives exactly on the boundary between two depots' areas. k-means must send them to one depot, and it does, with full confidence, because a point's group is always an integer. Nudge the customer's address by a metre and the answer flips. Yet nothing in the data says which depot is right. The honest answer is "about half and half".

Hard assignment costs you in a second way. A customer on the boundary pulls the mean of the group they were forced into exactly as hard as a customer standing next to the depot, and the depots drift. Also, k-means measures plain distance, so it describes every group as a round ball, even when the real group is a stretched and tilted cloud.

This chapter gives each point a share in every group, and lets every group have its own shape. The method is a Gaussian mixture. Its first step, the one that computes the shares, is called the E-step. The next chapter explains the name.

## What you already know

Five ideas carry this chapter:

- A generative classifier scores each class by the prior times how typical the point is for that class, then rescales the scores to add up to 1 (see "Classify by asking who made this").
- A Gaussian is described by a mean and a covariance, where the covariance averages the products of the gaps around the mean (see "Fit a Gaussian cloud to each class").
- Taking logs turns a product into a sum. A one-dimensional bell with variance 1 (the average squared gap from its mean) has height $1/\sqrt{2\pi}$ at its peak, so its log height there is $-\tfrac12\log(2\pi)\approx -0.919$.
- `X` has shape `(n, d)`, one row per point.
- A probability density is not a probability. It is a height, and only comparisons between heights matter here.

## The honest answer is a split

Imagine the data came from a game with two steps. First, roll a weighted die to choose a blob. Second, draw a point from that blob's bell curve. You see the points, never the rolls. Clustering is the job of guessing the rolls. The roll is hidden, so statisticians call it a **latent** choice.

Figure 1 shows the game in one dimension. The blobs have means $0$ and $2$, the same width, and weight $0.5$ each, so the die is fair. The dashed bells are the weighted blobs. The solid curve is their sum, the **mixture**.

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

Why bells? A bell says "points are common near the centre and rare far away", which is how a cluster looks, and it has two knobs you can fit: a centre and a spread. The **Gaussian** is the standard bell-shaped curve with those two knobs, and it is the standard choice for that reason.

Now ask who made a given point. At $x=0$ the blue bar is tall and the green bar is short, so blob 0 probably made it. At $x=1$ the bars are equal. The **responsibility** of blob $j$ for point $i$ is the probability that blob $j$ rolled it. It is Bayes' rule, the same rule as in the generative classifier, with the labels hidden.

### A worked example, two points

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

Read it aloud: take the squared distance measured with $\Sigma^{-1}$, add $\log\det\Sigma$ (which says a wider blob spreads its probability thinner, so its peak is lower), add $d\log(2\pi)$, and multiply by $-\tfrac12$. For the two points above the log heights are -2.084 and -6.369. In numpy, with `slogdet` for the log of the determinant:

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

The log sits outside a sum over blobs, and the log of a sum does not split into pieces. That is why there is no formula for the best $\pi$, $\mu$, and $\Sigma$ in one step. The next chapter gets around it by alternating, which is the idea of EM.

For the two points, $\log L=\log 0.2265+\log 0.2265=-2.970$.

## Log-sum-exp, and why you need it

Work with the log scores $s_{ij}=\log\pi_j+\log\mathrm{N}(x_i;\mu_j,\Sigma_j)$. To get the shares you must undo the log, and that is where a computer fails. Try two blobs far from a point, with scores $-1000$ and $-1001$. In floating point `np.exp(-1000.0)` is `0.0`. Both terms vanish, the row sums to 0, and the share is $0/0$, which is `nan`. The right panel of Figure 3 shows the failure.

```figure
<svg viewBox="0 0 600 295" role="img" aria-label="Left, the share of blob 1 as a point slides from 0 to 4, sharpening as the variance shrinks. Right, exp of minus 1000 underflows to zero while shifted scores stay finite">
<defs></defs>
<line x1="55.0" y1="235.0" x2="325.0" y2="235.0" class="stroke-muted" />
<line x1="55.0" y1="45.0" x2="55.0" y2="235.0" class="stroke-muted" />
<line x1="100.0" y1="235.0" x2="100.0" y2="239.0" class="stroke-muted" />
<text x="100.0" y="252.0" class="fig-small" text-anchor="middle" >0</text>
<line x1="145.0" y1="235.0" x2="145.0" y2="239.0" class="stroke-muted" />
<text x="145.0" y="252.0" class="fig-small" text-anchor="middle" >1</text>
<line x1="190.0" y1="235.0" x2="190.0" y2="239.0" class="stroke-muted" />
<text x="190.0" y="252.0" class="fig-small" text-anchor="middle" >2</text>
<line x1="235.0" y1="235.0" x2="235.0" y2="239.0" class="stroke-muted" />
<text x="235.0" y="252.0" class="fig-small" text-anchor="middle" >3</text>
<line x1="280.0" y1="235.0" x2="280.0" y2="239.0" class="stroke-muted" />
<text x="280.0" y="252.0" class="fig-small" text-anchor="middle" >4</text>
<line x1="55.0" y1="235.0" x2="325.0" y2="235.0" class="stroke-grid" />
<line x1="51.0" y1="235.0" x2="55.0" y2="235.0" class="stroke-muted" />
<text x="48.0" y="239.0" class="fig-small" text-anchor="end" >0</text>
<line x1="55.0" y1="187.5" x2="325.0" y2="187.5" class="stroke-grid" />
<line x1="51.0" y1="187.5" x2="55.0" y2="187.5" class="stroke-muted" />
<text x="48.0" y="191.5" class="fig-small" text-anchor="end" >0.25</text>
<line x1="55.0" y1="140.0" x2="325.0" y2="140.0" class="stroke-grid" />
<line x1="51.0" y1="140.0" x2="55.0" y2="140.0" class="stroke-muted" />
<text x="48.0" y="144.0" class="fig-small" text-anchor="end" >0.5</text>
<line x1="55.0" y1="92.5" x2="325.0" y2="92.5" class="stroke-grid" />
<line x1="51.0" y1="92.5" x2="55.0" y2="92.5" class="stroke-muted" />
<text x="48.0" y="96.5" class="fig-small" text-anchor="end" >0.75</text>
<line x1="55.0" y1="45.0" x2="325.0" y2="45.0" class="stroke-grid" />
<line x1="51.0" y1="45.0" x2="55.0" y2="45.0" class="stroke-muted" />
<text x="48.0" y="49.0" class="fig-small" text-anchor="end" >1</text>
<text x="190.0" y="27.0" class="fig-bold" text-anchor="middle" >Share of blob 1 at position x</text>
<text x="190.0" y="269.0" class="fig-small" text-anchor="middle" >x, between blob 0 at 0 and blob 1 at 4</text>
<polyline points="55.0,235.0 56.1,235.0 57.2,235.0 58.4,235.0 59.5,235.0 60.6,235.0 61.8,235.0 62.9,235.0 64.0,235.0 65.1,235.0 66.2,235.0 67.4,235.0 68.5,235.0 69.6,235.0 70.8,235.0 71.9,235.0 73.0,235.0 74.1,235.0 75.2,235.0 76.4,235.0 77.5,235.0 78.6,235.0 79.8,235.0 80.9,235.0 82.0,235.0 83.1,235.0 84.2,235.0 85.4,235.0 86.5,235.0 87.6,235.0 88.8,235.0 89.9,235.0 91.0,235.0 92.1,235.0 93.2,235.0 94.4,235.0 95.5,235.0 96.6,235.0 97.8,234.9 98.9,234.9 100.0,234.9 101.1,234.9 102.2,234.9 103.4,234.9 104.5,234.9 105.6,234.9 106.8,234.9 107.9,234.9 109.0,234.9 110.1,234.8 111.2,234.8 112.4,234.8 113.5,234.8 114.6,234.8 115.8,234.7 116.9,234.7 118.0,234.7 119.1,234.7 120.3,234.6 121.4,234.6 122.5,234.5 123.6,234.5 124.8,234.4 125.9,234.4 127.0,234.3 128.1,234.2 129.2,234.1 130.4,234.1 131.5,234.0 132.6,233.8 133.8,233.7 134.9,233.6 136.0,233.4 137.1,233.3 138.2,233.1 139.4,232.9 140.5,232.7 141.6,232.5 142.8,232.2 143.9,231.9 145.0,231.6 146.1,231.2 147.2,230.8 148.4,230.4 149.5,229.9 150.6,229.4 151.8,228.9 152.9,228.2 154.0,227.6 155.1,226.8 156.2,226.0 157.4,225.1 158.5,224.1 159.6,223.0 160.8,221.9 161.9,220.6 163.0,219.2 164.1,217.7 165.2,216.0 166.4,214.3 167.5,212.4 168.6,210.3 169.8,208.0 170.9,205.7 172.0,203.1 173.1,200.3 174.2,197.4 175.4,194.3 176.5,191.0 177.6,187.5 178.8,183.9 179.9,180.1 181.0,176.1 182.1,172.0 183.2,167.7 184.4,163.3 185.5,158.8 186.6,154.1 187.8,149.5 188.9,144.7 190.0,140.0 191.1,135.3 192.3,130.5 193.4,125.9 194.5,121.2 195.6,116.7 196.8,112.3 197.9,108.0 199.0,103.9 200.1,99.9 201.2,96.1 202.4,92.5 203.5,89.0 204.6,85.7 205.8,82.6 206.9,79.7 208.0,76.9 209.1,74.3 210.3,72.0 211.4,69.7 212.5,67.6 213.6,65.7 214.8,64.0 215.9,62.3 217.0,60.8 218.1,59.4 219.3,58.1 220.4,57.0 221.5,55.9 222.6,54.9 223.8,54.0 224.9,53.2 226.0,52.4 227.1,51.8 228.2,51.1 229.4,50.6 230.5,50.1 231.6,49.6 232.8,49.2 233.9,48.8 235.0,48.4 236.1,48.1 237.2,47.8 238.4,47.5 239.5,47.3 240.6,47.1 241.8,46.9 242.9,46.7 244.0,46.6 245.1,46.4 246.2,46.3 247.4,46.2 248.5,46.0 249.6,45.9 250.8,45.9 251.9,45.8 253.0,45.7 254.1,45.6 255.2,45.6 256.4,45.5 257.5,45.5 258.6,45.4 259.8,45.4 260.9,45.3 262.0,45.3 263.1,45.3 264.2,45.3 265.4,45.2 266.5,45.2 267.6,45.2 268.8,45.2 269.9,45.2 271.0,45.1 272.1,45.1 273.2,45.1 274.4,45.1 275.5,45.1 276.6,45.1 277.8,45.1 278.9,45.1 280.0,45.1 281.1,45.1 282.2,45.1 283.4,45.0 284.5,45.0 285.6,45.0 286.8,45.0 287.9,45.0 289.0,45.0 290.1,45.0 291.2,45.0 292.4,45.0 293.5,45.0 294.6,45.0 295.8,45.0 296.9,45.0 298.0,45.0 299.1,45.0 300.2,45.0 301.4,45.0 302.5,45.0 303.6,45.0 304.8,45.0 305.9,45.0 307.0,45.0 308.1,45.0 309.2,45.0 310.4,45.0 311.5,45.0 312.6,45.0 313.8,45.0 314.9,45.0 316.0,45.0 317.1,45.0 318.2,45.0 319.4,45.0 320.5,45.0 321.6,45.0 322.8,45.0 323.9,45.0 325.0,45.0" fill="none" class="stroke-a" stroke-width="2.5"/>
<polyline points="55.0,235.0 56.1,235.0 57.2,235.0 58.4,235.0 59.5,235.0 60.6,235.0 61.8,235.0 62.9,235.0 64.0,235.0 65.1,235.0 66.2,235.0 67.4,235.0 68.5,235.0 69.6,235.0 70.8,235.0 71.9,235.0 73.0,235.0 74.1,235.0 75.2,235.0 76.4,235.0 77.5,235.0 78.6,235.0 79.8,235.0 80.9,235.0 82.0,235.0 83.1,235.0 84.2,235.0 85.4,235.0 86.5,235.0 87.6,235.0 88.8,235.0 89.9,235.0 91.0,235.0 92.1,235.0 93.2,235.0 94.4,235.0 95.5,235.0 96.6,235.0 97.8,235.0 98.9,235.0 100.0,235.0 101.1,235.0 102.2,235.0 103.4,235.0 104.5,235.0 105.6,235.0 106.8,235.0 107.9,235.0 109.0,235.0 110.1,235.0 111.2,235.0 112.4,235.0 113.5,235.0 114.6,235.0 115.8,235.0 116.9,235.0 118.0,235.0 119.1,235.0 120.3,235.0 121.4,235.0 122.5,235.0 123.6,235.0 124.8,235.0 125.9,235.0 127.0,235.0 128.1,235.0 129.2,235.0 130.4,235.0 131.5,235.0 132.6,235.0 133.8,235.0 134.9,235.0 136.0,235.0 137.1,235.0 138.2,235.0 139.4,235.0 140.5,235.0 141.6,235.0 142.8,235.0 143.9,234.9 145.0,234.9 146.1,234.9 147.2,234.9 148.4,234.9 149.5,234.9 150.6,234.8 151.8,234.8 152.9,234.7 154.0,234.7 155.1,234.6 156.2,234.5 157.4,234.4 158.5,234.3 159.6,234.1 160.8,234.0 161.9,233.7 163.0,233.4 164.1,233.1 165.2,232.7 166.4,232.2 167.5,231.6 168.6,230.8 169.8,229.9 170.9,228.9 172.0,227.6 173.1,226.0 174.2,224.1 175.4,221.9 176.5,219.2 177.6,216.0 178.8,212.4 179.9,208.0 181.0,203.1 182.1,197.4 183.2,191.0 184.4,183.9 185.5,176.1 186.6,167.7 187.8,158.8 188.9,149.5 190.0,140.0 191.1,130.5 192.3,121.2 193.4,112.3 194.5,103.9 195.6,96.1 196.8,89.0 197.9,82.6 199.0,76.9 200.1,72.0 201.2,67.6 202.4,64.0 203.5,60.8 204.6,58.1 205.8,55.9 206.9,54.0 208.0,52.4 209.1,51.1 210.3,50.1 211.4,49.2 212.5,48.4 213.6,47.8 214.8,47.3 215.9,46.9 217.0,46.6 218.1,46.3 219.3,46.0 220.4,45.9 221.5,45.7 222.6,45.6 223.8,45.5 224.9,45.4 226.0,45.3 227.1,45.3 228.2,45.2 229.4,45.2 230.5,45.1 231.6,45.1 232.8,45.1 233.9,45.1 235.0,45.1 236.1,45.1 237.2,45.0 238.4,45.0 239.5,45.0 240.6,45.0 241.8,45.0 242.9,45.0 244.0,45.0 245.1,45.0 246.2,45.0 247.4,45.0 248.5,45.0 249.6,45.0 250.8,45.0 251.9,45.0 253.0,45.0 254.1,45.0 255.2,45.0 256.4,45.0 257.5,45.0 258.6,45.0 259.8,45.0 260.9,45.0 262.0,45.0 263.1,45.0 264.2,45.0 265.4,45.0 266.5,45.0 267.6,45.0 268.8,45.0 269.9,45.0 271.0,45.0 272.1,45.0 273.2,45.0 274.4,45.0 275.5,45.0 276.6,45.0 277.8,45.0 278.9,45.0 280.0,45.0 281.1,45.0 282.2,45.0 283.4,45.0 284.5,45.0 285.6,45.0 286.8,45.0 287.9,45.0 289.0,45.0 290.1,45.0 291.2,45.0 292.4,45.0 293.5,45.0 294.6,45.0 295.8,45.0 296.9,45.0 298.0,45.0 299.1,45.0 300.2,45.0 301.4,45.0 302.5,45.0 303.6,45.0 304.8,45.0 305.9,45.0 307.0,45.0 308.1,45.0 309.2,45.0 310.4,45.0 311.5,45.0 312.6,45.0 313.8,45.0 314.9,45.0 316.0,45.0 317.1,45.0 318.2,45.0 319.4,45.0 320.5,45.0 321.6,45.0 322.8,45.0 323.9,45.0 325.0,45.0" fill="none" class="stroke-c" stroke-width="2.5"/>
<polyline points="55.0,235.0 56.1,235.0 57.2,235.0 58.4,235.0 59.5,235.0 60.6,235.0 61.8,235.0 62.9,235.0 64.0,235.0 65.1,235.0 66.2,235.0 67.4,235.0 68.5,235.0 69.6,235.0 70.8,235.0 71.9,235.0 73.0,235.0 74.1,235.0 75.2,235.0 76.4,235.0 77.5,235.0 78.6,235.0 79.8,235.0 80.9,235.0 82.0,235.0 83.1,235.0 84.2,235.0 85.4,235.0 86.5,235.0 87.6,235.0 88.8,235.0 89.9,235.0 91.0,235.0 92.1,235.0 93.2,235.0 94.4,235.0 95.5,235.0 96.6,235.0 97.8,235.0 98.9,235.0 100.0,235.0 101.1,235.0 102.2,235.0 103.4,235.0 104.5,235.0 105.6,235.0 106.8,235.0 107.9,235.0 109.0,235.0 110.1,235.0 111.2,235.0 112.4,235.0 113.5,235.0 114.6,235.0 115.8,235.0 116.9,235.0 118.0,235.0 119.1,235.0 120.3,235.0 121.4,235.0 122.5,235.0 123.6,235.0 124.8,235.0 125.9,235.0 127.0,235.0 128.1,235.0 129.2,235.0 130.4,235.0 131.5,235.0 132.6,235.0 133.8,235.0 134.9,235.0 136.0,235.0 137.1,235.0 138.2,235.0 139.4,235.0 140.5,235.0 141.6,235.0 142.8,235.0 143.9,235.0 145.0,235.0 146.1,235.0 147.2,235.0 148.4,235.0 149.5,235.0 150.6,235.0 151.8,235.0 152.9,235.0 154.0,235.0 155.1,235.0 156.2,235.0 157.4,235.0 158.5,235.0 159.6,235.0 160.8,235.0 161.9,235.0 163.0,235.0 164.1,235.0 165.2,235.0 166.4,235.0 167.5,235.0 168.6,235.0 169.8,235.0 170.9,235.0 172.0,235.0 173.1,235.0 174.2,235.0 175.4,235.0 176.5,235.0 177.6,235.0 178.8,235.0 179.9,235.0 181.0,234.9 182.1,234.8 183.2,234.5 184.4,233.7 185.5,231.6 186.6,226.0 187.8,212.4 188.9,183.9 190.0,140.0 191.1,96.1 192.3,67.6 193.4,54.0 194.5,48.4 195.6,46.3 196.8,45.5 197.9,45.2 199.0,45.1 200.1,45.0 201.2,45.0 202.4,45.0 203.5,45.0 204.6,45.0 205.8,45.0 206.9,45.0 208.0,45.0 209.1,45.0 210.3,45.0 211.4,45.0 212.5,45.0 213.6,45.0 214.8,45.0 215.9,45.0 217.0,45.0 218.1,45.0 219.3,45.0 220.4,45.0 221.5,45.0 222.6,45.0 223.8,45.0 224.9,45.0 226.0,45.0 227.1,45.0 228.2,45.0 229.4,45.0 230.5,45.0 231.6,45.0 232.8,45.0 233.9,45.0 235.0,45.0 236.1,45.0 237.2,45.0 238.4,45.0 239.5,45.0 240.6,45.0 241.8,45.0 242.9,45.0 244.0,45.0 245.1,45.0 246.2,45.0 247.4,45.0 248.5,45.0 249.6,45.0 250.8,45.0 251.9,45.0 253.0,45.0 254.1,45.0 255.2,45.0 256.4,45.0 257.5,45.0 258.6,45.0 259.8,45.0 260.9,45.0 262.0,45.0 263.1,45.0 264.2,45.0 265.4,45.0 266.5,45.0 267.6,45.0 268.8,45.0 269.9,45.0 271.0,45.0 272.1,45.0 273.2,45.0 274.4,45.0 275.5,45.0 276.6,45.0 277.8,45.0 278.9,45.0 280.0,45.0 281.1,45.0 282.2,45.0 283.4,45.0 284.5,45.0 285.6,45.0 286.8,45.0 287.9,45.0 289.0,45.0 290.1,45.0 291.2,45.0 292.4,45.0 293.5,45.0 294.6,45.0 295.8,45.0 296.9,45.0 298.0,45.0 299.1,45.0 300.2,45.0 301.4,45.0 302.5,45.0 303.6,45.0 304.8,45.0 305.9,45.0 307.0,45.0 308.1,45.0 309.2,45.0 310.4,45.0 311.5,45.0 312.6,45.0 313.8,45.0 314.9,45.0 316.0,45.0 317.1,45.0 318.2,45.0 319.4,45.0 320.5,45.0 321.6,45.0 322.8,45.0 323.9,45.0 325.0,45.0" fill="none" class="stroke-b" stroke-width="2.5"/>
<line x1="235.0" y1="159.0" x2="257.5" y2="159.0" class="stroke-a" stroke-width="2.5"/>
<text x="262.0" y="163.0" class="fig-small" text-anchor="start" >variance 1</text>
<line x1="235.0" y1="178.0" x2="257.5" y2="178.0" class="stroke-c" stroke-width="2.5"/>
<text x="262.0" y="182.0" class="fig-small" text-anchor="start" >variance 0.5</text>
<line x1="235.0" y1="197.0" x2="257.5" y2="197.0" class="stroke-b" stroke-width="2.5"/>
<text x="262.0" y="201.0" class="fig-small" text-anchor="start" >variance 0.1</text>
<circle cx="190.0" cy="140.0" r="4" class="fill-ink" />
<text x="475.0" y="27.0" class="fig-bold" text-anchor="middle" >Scores -1000 and -1001</text>
<line x1="390.0" y1="235.0" x2="580.0" y2="235.0" class="stroke-muted" />
<line x1="390.0" y1="45.0" x2="390.0" y2="235.0" class="stroke-muted" />
<rect x="405.0" y="235.0" width="28.0" height="1.5" class="fill-bad" />
<text x="419.0" y="229.0" class="fig-small" text-anchor="middle" >0</text>
<rect x="439.0" y="235.0" width="28.0" height="1.5" class="fill-bad" />
<text x="453.0" y="229.0" class="fig-small" text-anchor="middle" >0</text>
<text x="436.0" y="251.0" class="fig-small" text-anchor="middle" >naive</text>
<rect x="505.0" y="45.0" width="28.0" height="190.0" class="fill-c" />
<text x="519.0" y="39.0" class="fig-small" text-anchor="middle" >1</text>
<rect x="539.0" y="165.1" width="28.0" height="69.9" class="fill-c" />
<text x="553.0" y="159.1" class="fig-small" text-anchor="middle" >0.368</text>
<text x="536.0" y="251.0" class="fig-small" text-anchor="middle" >shifted</text>
</svg>
Figure 3. Left: the share is 50 percent halfway between the means and sharpens into a step as the variance shrinks. Right: the naive exponentials are both 0, so the row total is 0 and the share is 0 divided by 0. After subtracting the larger score the biggest term is 1.
```

The fix is to subtract each row's largest score $m_i$ before exponentiating. The shares do not change, because $m_i$ cancels from the top and the bottom, and the largest shifted term is exactly $e^0=1$, so the row can never vanish:

$$
\log \sum_{j} e^{s_{ij}} = m_i + \log \sum_{j} e^{s_{ij} - m_i} .
$$

For the scores $-1000$ and $-1001$ this gives $-1000+\log(1+e^{-1})=$ -999.6867. In numpy, with the row maximum kept as a column:

```python
m = s.max(axis=1, keepdims=True)
log_px = m[:, 0] + np.log(np.exp(s - m).sum(axis=1))
resp = np.exp(s - log_px[:, None])
```

> [!TIP]
> Keep the maximum with `keepdims=True`, so it has shape `(n, 1)`. A bare `(n,)` vector broadcasts across columns instead of rows, and when $n$ equals $k$ nothing complains.

The left panel of Figure 3 shows what the shares look like as a point slides between blobs at $0$ and $4$. At variance 1 the switch is gradual, and it sharpens toward a step as the variance shrinks. Hold on to that picture. It returns in the next chapter.

> [!KEY]
> A Gaussian mixture says each point came from a blob chosen by a weighted die. The E-step computes, for each point, the probability that each blob made it: weight times height, rescaled so the row adds to 1, with logs and log-sum-exp keeping the numbers safe.

> [!RECAP]
> Before you build, keep these points in hand:
> - A point's responsibilities are its weighted heights under each blob, rescaled to add to 1.
> - The log density of a Gaussian is $-\tfrac12(d\log 2\pi + \log\det\Sigma + \text{squared distance with }\Sigma^{-1})$.
> - Compute $\log\det\Sigma$ with `slogdet`, and never with the determinant itself.
> - Subtract each row's maximum before the exponential, and add it back after the log.
> - $\log L$ is the sum over points of the log of each point's row total.

## Build it step by step

You write three functions, each built from the one before.

1. **`log_gaussian(X, mean, cov)`** takes `(n, d)`, `(d,)`, and `(d, d)` and returns the log heights, shape `(n,)`. Subtract the mean, get `logdet` from `slogdet`, and compute the squared distance row by row with a product by the inverse and a sum over axis 1. The usual mistake is `gap @ inv @ gap.T`, which builds an `(n, n)` matrix of every pair when you want only the diagonal.

    ```python
    def log_gaussian(X, mean, cov):
        # -0.5 * (d*log(2 pi) + logdet + squared distance with inv(cov))
    ```

2. **`logsumexp_rows(s)`** takes scores of shape `(n, k)` and returns `(n,)`. Subtract the row maximum, exponentiate, sum, take the log, and add the maximum back. The usual mistake is to exponentiate first.

    ```python
    def logsumexp_rows(s):
        # m + log(sum(exp(s - m))) per row; do not modify s
    ```

3. **`gmm_e_step(X, means, covs, weights)`** returns the responsibilities `(n, k)` and the log-likelihood as a Python float. Fill an `(n, k)` score table with `np.log(weights[j]) + log_gaussian(...)`, call `logsumexp_rows`, and subtract the totals from the scores before the exponential. The usual mistake is to put `weights[j]` in front of the log density without taking its log.

    ```python
    def gmm_e_step(X, means, covs, weights):
        # scores[:, j] = log weight + log density; resp = exp(scores - totals)
    ```

## Where this breaks

- **A tilted blob needs many numbers.** A covariance matrix has about $d^2/2$ free entries, so it grows with the square of the dimension, and with few points per blob the estimate is noisy. Diagonal covariances are a cheaper compromise.
- **The E-step costs more than k-means.** Each blob needs an inverse and a log determinant, and each point needs a product with the inverse. It costs about $n k d^2$ operations, plus $k d^3$ for the inverses, against $nkd$ for assignment.
- **A far blob underflows without logs.** That is the failure in Figure 3, and it appears as soon as the data have a few dimensions.

## The exercise

You build `log_gaussian`, `logsumexp_rows`, and `gmm_e_step`. The checks confirm the log height $-0.919$ at the peak of a standard normal, match a direct formula for a tilted two-dimensional covariance, and confirm a diagonal covariance equals the sum of one-dimensional log heights. They push `logsumexp_rows` to scores near $-1000$, and check that `gmm_e_step` reproduces the chapter's shares $0.8808$ and $0.1192$ with $\log L=-2.970$, obeys Bayes' rule with unequal weights, and stays finite for blobs 1000 units apart.
