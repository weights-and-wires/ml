---
title: "Learning-rate schedules: warmup and cosine decay"
slug: 068-lr-schedules
part: "Part 4 — Neural networks"
summary: Raise the learning rate in a straight line, then lower it along half a cosine, so training starts gently and ends settled.
minutes: 35
gpu: false
objectives:
  - Explain why a learning rate that suits the middle of training suits neither its start nor its end.
  - Compute a linear warmup and a cosine decay by hand, and check four anchor points.
  - Explain why cosine decay stays high early, falls fastest in the middle, and flattens at zero.
  - Run Adam with a schedule and measure how much lower it settles than a constant rate.
prereqs:
  - 065-stochastic-gradients
  - 066-adam
  - 067-adamw
exercise: 068-lr-schedules
---

# Learning-rate schedules: warmup and cosine decay

You train with Adam at a learning rate that works well in the middle of the run. The loss curve has two ugly ends. In the first few steps the loss shoots up before it starts to fall, and in a deep network that first jump can run all the way to `nan`. Late in the run the loss stops falling and wiggles on a plateau, however long you wait.

Lowering the rate cures the wiggle and makes the middle of the run crawl. Raising it speeds the middle and makes the first jump worse. So why does the same learning rate suit neither the beginning nor the end of training?

Because the two ends have different problems, and each needs a different rate. This chapter builds a schedule that changes the rate over time: small at first, large in the middle, and falling to zero at the end.

## What you already know

Five ideas carry this chapter:

- Adam's step is about one learning rate long in every coordinate, whatever the gradient size (see "Adam: a step size for every weight").
- Adam's averages start at zero. At step 1, $\hat{v}$ rests on a single batch, and its window is 1,000 steps.
- A mini-batch gradient is noisy, and the noise keeps the loss above a floor that a smaller learning rate lowers (see "Noisy gradients: stochastic gradient descent and mini-batches").
- Adam's decay is multiplied by the learning rate, so it moves with the schedule (see "AdamW: decaying weights the right way").
- The cosine function equals 1 at angle 0, 0 at angle $\pi/2$, and $-1$ at angle $\pi$.

## Give each end its own rate

At step 1, Adam has seen one batch of eight rows. It has no idea which gradient entries are real signal and which are noise, but it still moves every weight by a full learning rate. If the weights already sit near a good region, that is a kick of full size in a direction chosen by eight rows. Early steps should be short until the averages fill with evidence.

At the end, the problem flips. A batch gradient is noisy even at the bottom, so each step jiggles the weights by roughly the learning rate. The loss then hovers at a floor that grows with the rate. To settle, you must make the steps shorter.

> [!KEY]
> Early on, Adam's averages rest on a few noisy batches, so full-size steps are risky. Late in the run, noise keeps the loss above a floor that shorter steps lower. Each end wants a smaller rate than the middle.

So the rate should rise from near zero, hold a good value for the middle, and fall back to zero. A **learning-rate schedule** is a rule that gives the rate for each step number. Figure 1 shows the one this chapter builds: a straight ramp up, then a smooth fall.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Learning rate against step: linear warmup to 0.01 over 100 steps, then cosine decay to zero at step 1000">
<defs></defs>
<line x1="75.0" y1="260.0" x2="575.0" y2="260.0" class="stroke-muted"/>
<line x1="75.0" y1="30.0" x2="75.0" y2="260.0" class="stroke-muted"/>
<line x1="75.0" y1="260.0" x2="75.0" y2="265.0" class="stroke-muted"/>
<text x="75.0" y="279.0" class="fig-small" text-anchor="middle">0</text>
<line x1="125.0" y1="260.0" x2="125.0" y2="265.0" class="stroke-muted"/>
<text x="125.0" y="279.0" class="fig-small" text-anchor="middle">100</text>
<line x1="237.5" y1="260.0" x2="237.5" y2="265.0" class="stroke-muted"/>
<text x="237.5" y="279.0" class="fig-small" text-anchor="middle">325</text>
<line x1="350.0" y1="260.0" x2="350.0" y2="265.0" class="stroke-muted"/>
<text x="350.0" y="279.0" class="fig-small" text-anchor="middle">550</text>
<line x1="575.0" y1="260.0" x2="575.0" y2="265.0" class="stroke-muted"/>
<text x="575.0" y="279.0" class="fig-small" text-anchor="middle">1000</text>
<line x1="75.0" y1="260.0" x2="575.0" y2="260.0" class="stroke-grid"/>
<line x1="70.0" y1="260.0" x2="75.0" y2="260.0" class="stroke-muted"/>
<text x="66.0" y="264.0" class="fig-small" text-anchor="end">0</text>
<line x1="75.0" y1="160.0" x2="575.0" y2="160.0" class="stroke-grid"/>
<line x1="70.0" y1="160.0" x2="75.0" y2="160.0" class="stroke-muted"/>
<text x="66.0" y="164.0" class="fig-small" text-anchor="end">0.005</text>
<line x1="75.0" y1="60.0" x2="575.0" y2="60.0" class="stroke-grid"/>
<line x1="70.0" y1="60.0" x2="75.0" y2="60.0" class="stroke-muted"/>
<text x="66.0" y="64.0" class="fig-small" text-anchor="end">0.01</text>
<text x="325.0" y="298.0" class="fig-small" text-anchor="middle">step</text>
<polyline points="75.0,260.0 77.5,250.0 80.0,240.0 82.5,230.0 85.0,220.0 87.5,210.0 90.0,200.0 92.5,190.0 95.0,180.0 97.5,170.0 100.0,160.0 102.5,150.0 105.0,140.0 107.5,130.0 110.0,120.0 112.5,110.0 115.0,100.0 117.5,90.0 120.0,80.0 122.5,70.0 125.0,60.0 127.5,60.0 130.0,60.1 132.5,60.1 135.0,60.2 137.5,60.4 140.0,60.5 142.5,60.7 145.0,61.0 147.5,61.2 150.0,61.5 152.5,61.8 155.0,62.2 157.5,62.6 160.0,63.0 162.5,63.4 165.0,63.9 167.5,64.4 170.0,64.9 172.5,65.4 175.0,66.0 177.5,66.6 180.0,67.3 182.5,67.9 185.0,68.6 187.5,69.4 190.0,70.1 192.5,70.9 195.0,71.7 197.5,72.5 200.0,73.4 202.5,74.3 205.0,75.2 207.5,76.1 210.0,77.1 212.5,78.1 215.0,79.1 217.5,80.1 220.0,81.2 222.5,82.3 225.0,83.4 227.5,84.5 230.0,85.7 232.5,86.9 235.0,88.1 237.5,89.3 240.0,90.5 242.5,91.8 245.0,93.1 247.5,94.4 250.0,95.7 252.5,97.1 255.0,98.4 257.5,99.8 260.0,101.2 262.5,102.6 265.0,104.1 267.5,105.5 270.0,107.0 272.5,108.5 275.0,110.0 277.5,111.5 280.0,113.1 282.5,114.6 285.0,116.2 287.5,117.7 290.0,119.3 292.5,120.9 295.0,122.5 297.5,124.2 300.0,125.8 302.5,127.4 305.0,129.1 307.5,130.8 310.0,132.4 312.5,134.1 315.0,135.8 317.5,137.5 320.0,139.2 322.5,140.9 325.0,142.6 327.5,144.4 330.0,146.1 332.5,147.8 335.0,149.5 337.5,151.3 340.0,153.0 342.5,154.8 345.0,156.5 347.5,158.3 350.0,160.0 352.5,161.7 355.0,163.5 357.5,165.2 360.0,167.0 362.5,168.7 365.0,170.5 367.5,172.2 370.0,173.9 372.5,175.6 375.0,177.4 377.5,179.1 380.0,180.8 382.5,182.5 385.0,184.2 387.5,185.9 390.0,187.6 392.5,189.2 395.0,190.9 397.5,192.6 400.0,194.2 402.5,195.8 405.0,197.5 407.5,199.1 410.0,200.7 412.5,202.3 415.0,203.8 417.5,205.4 420.0,206.9 422.5,208.5 425.0,210.0 427.5,211.5 430.0,213.0 432.5,214.5 435.0,215.9 437.5,217.4 440.0,218.8 442.5,220.2 445.0,221.6 447.5,222.9 450.0,224.3 452.5,225.6 455.0,226.9 457.5,228.2 460.0,229.5 462.5,230.7 465.0,231.9 467.5,233.1 470.0,234.3 472.5,235.5 475.0,236.6 477.5,237.7 480.0,238.8 482.5,239.9 485.0,240.9 487.5,241.9 490.0,242.9 492.5,243.9 495.0,244.8 497.5,245.7 500.0,246.6 502.5,247.5 505.0,248.3 507.5,249.1 510.0,249.9 512.5,250.6 515.0,251.4 517.5,252.1 520.0,252.7 522.5,253.4 525.0,254.0 527.5,254.6 530.0,255.1 532.5,255.6 535.0,256.1 537.5,256.6 540.0,257.0 542.5,257.4 545.0,257.8 547.5,258.2 550.0,258.5 552.5,258.8 555.0,259.0 557.5,259.3 560.0,259.5 562.5,259.6 565.0,259.8 567.5,259.9 570.0,259.9 572.5,260.0 575.0,260.0" fill="none" class="stroke-a"/>
<circle cx="75.0" cy="260.0" r="4.5" class="fill-b"/>
<text x="87.0" y="250.0" class="fig-small" text-anchor="start">0</text>
<circle cx="100.0" cy="160.0" r="4.5" class="fill-b"/>
<text x="110.0" y="178.0" class="fig-small" text-anchor="start">0.005</text>
<circle cx="125.0" cy="60.0" r="4.5" class="fill-b"/>
<text x="137.0" y="50.0" class="fig-small" text-anchor="start">0.01</text>
<circle cx="237.5" cy="89.3" r="4.5" class="fill-b"/>
<text x="249.5" y="81.3" class="fig-small" text-anchor="start">0.0085355</text>
<circle cx="350.0" cy="160.0" r="4.5" class="fill-b"/>
<text x="362.0" y="152.0" class="fig-small" text-anchor="start">0.005</text>
<circle cx="575.0" cy="260.0" r="4.5" class="fill-b"/>
<text x="567.0" y="248.0" class="fig-small" text-anchor="end">0</text>
</svg>
Figure 1. The exact schedule from the worked example, drawn from computed values. The rate climbs in a straight line from 0 to 0.01 over the first 100 steps, then follows half a cosine down to 0 at step 1000. The orange dots are the steps in the table: 0, 50, 100, 325, 550, and 1000.
```

## Ramp up in a straight line

**Linear warmup** raises the rate from 0 to a chosen peak over the first `warmup` steps, in a straight line. The rate at step $s$ is the peak times the fraction of warmup done:

$$
\text{lr}(s) = \text{peak}\cdot\frac{s}{\text{warmup}} \qquad \text{for } s < \text{warmup}
$$

Read it aloud: halfway through warmup, use half the peak. Take 100 warmup steps and a peak of 0.01. Step 0 gives 0, step 50 gives $0.01 \times 50/100 = 0.005$, and step 100 reaches 0.01. In numpy:

```python
lr = peak * step / warmup
```

Why a straight line rather than a jump? A jump from 0 to the peak is no warmup at all. The line gives the averages time to fill. With $\beta_1 = 0.9$ the average gradient is a fair summary within about 10 steps. The corrected $\hat{v}$ is an average of the $t$ squared gradients seen so far, so it rests on one batch at step 1, on 20 batches at step 20, and on 100 batches at the end of this chapter's example warmup. A warmup of tens to hundreds of steps lets both settle before the rate gets large.

## Fall along half a cosine

After warmup, let $p$ be how far you are through the decay phase, from 0 at the end of warmup to 1 at the last step:

$$
p = \frac{s - \text{warmup}}{\text{total} - \text{warmup}}, \qquad
\text{lr}(s) = \text{peak}\cdot\tfrac{1}{2}\big(1 + \cos(\pi p)\big)
$$

Read it aloud: the factor $\tfrac12(1 + \cos \pi p)$ starts at 1 when $p = 0$, passes $\tfrac12$ when $p = \tfrac12$, and reaches 0 when $p = 1$. It is half of a cosine wave, lifted so that it runs from 1 down to 0. Multiply it by the peak and you have the rate. With 1,000 total steps these rates hold:

| Step | Phase | Calculation | Rate |
|---|---|---|---|
| 0 | warmup | $0.01 \times 0/100$ | 0 |
| 50 | warmup | $0.01 \times 50/100$ | 0.005 |
| 100 | decay, $p = 0$ | $0.01 \times \tfrac12 \times (1 + 1)$ | 0.01 |
| 325 | decay, $p = 0.25$ | $0.01 \times \tfrac12 \times (1 + 0.7071)$ | 0.0085355 |
| 550 | decay, $p = 0.5$ | $0.01 \times \tfrac12 \times (1 + 0)$ | 0.005 |
| 1000 | decay, $p = 1$ | $0.01 \times \tfrac12 \times (1 - 1)$ | 0 |

In numpy, clamp $p$ so that steps past the end stay at 0:

```python
p = min(max((step - warmup) / (total - warmup), 0.0), 1.0)
lr = peak * 0.5 * (1.0 + np.cos(np.pi * p))
```

Why cosine and not a straight line down? Figure 2 compares them. Cosine stays near the peak early, when the weights are still moving and a high rate pays off. It falls fastest in the middle. It flattens as it nears 0, so the last steps are tiny and the weights can settle. A straight line gives up rate too early. Take the point 90 percent of the way through the decay: cosine has fallen to $\tfrac12(1 + \cos 0.9\pi) = 0.024$ of the peak, while a straight line still holds 0.1. Early, at $p = 0.25$, cosine keeps 85 percent of the peak and the line 75 percent. The choice is partly empirical: many shapes work, and cosine became popular because it works well and has no extra setting to tune.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Cosine decay, linear decay, and a constant rate after warmup">
<defs></defs>
<line x1="75.0" y1="260.0" x2="575.0" y2="260.0" class="stroke-muted"/>
<line x1="75.0" y1="30.0" x2="75.0" y2="260.0" class="stroke-muted"/>
<line x1="75.0" y1="260.0" x2="75.0" y2="265.0" class="stroke-muted"/>
<text x="75.0" y="279.0" class="fig-small" text-anchor="middle">0</text>
<line x1="125.0" y1="260.0" x2="125.0" y2="265.0" class="stroke-muted"/>
<text x="125.0" y="279.0" class="fig-small" text-anchor="middle">100</text>
<line x1="237.5" y1="260.0" x2="237.5" y2="265.0" class="stroke-muted"/>
<text x="237.5" y="279.0" class="fig-small" text-anchor="middle">325</text>
<line x1="350.0" y1="260.0" x2="350.0" y2="265.0" class="stroke-muted"/>
<text x="350.0" y="279.0" class="fig-small" text-anchor="middle">550</text>
<line x1="575.0" y1="260.0" x2="575.0" y2="265.0" class="stroke-muted"/>
<text x="575.0" y="279.0" class="fig-small" text-anchor="middle">1000</text>
<line x1="75.0" y1="260.0" x2="575.0" y2="260.0" class="stroke-grid"/>
<line x1="70.0" y1="260.0" x2="75.0" y2="260.0" class="stroke-muted"/>
<text x="66.0" y="264.0" class="fig-small" text-anchor="end">0</text>
<line x1="75.0" y1="160.0" x2="575.0" y2="160.0" class="stroke-grid"/>
<line x1="70.0" y1="160.0" x2="75.0" y2="160.0" class="stroke-muted"/>
<text x="66.0" y="164.0" class="fig-small" text-anchor="end">0.005</text>
<line x1="75.0" y1="60.0" x2="575.0" y2="60.0" class="stroke-grid"/>
<line x1="70.0" y1="60.0" x2="75.0" y2="60.0" class="stroke-muted"/>
<text x="66.0" y="64.0" class="fig-small" text-anchor="end">0.01</text>
<text x="325.0" y="298.0" class="fig-small" text-anchor="middle">step</text>
<polyline points="75.0,260.0 77.5,250.0 80.0,240.0 82.5,230.0 85.0,220.0 87.5,210.0 90.0,200.0 92.5,190.0 95.0,180.0 97.5,170.0 100.0,160.0 102.5,150.0 105.0,140.0 107.5,130.0 110.0,120.0 112.5,110.0 115.0,100.0 117.5,90.0 120.0,80.0 122.5,70.0 125.0,60.0 127.5,60.0 130.0,60.0 132.5,60.0 135.0,60.0 137.5,60.0 140.0,60.0 142.5,60.0 145.0,60.0 147.5,60.0 150.0,60.0 152.5,60.0 155.0,60.0 157.5,60.0 160.0,60.0 162.5,60.0 165.0,60.0 167.5,60.0 170.0,60.0 172.5,60.0 175.0,60.0 177.5,60.0 180.0,60.0 182.5,60.0 185.0,60.0 187.5,60.0 190.0,60.0 192.5,60.0 195.0,60.0 197.5,60.0 200.0,60.0 202.5,60.0 205.0,60.0 207.5,60.0 210.0,60.0 212.5,60.0 215.0,60.0 217.5,60.0 220.0,60.0 222.5,60.0 225.0,60.0 227.5,60.0 230.0,60.0 232.5,60.0 235.0,60.0 237.5,60.0 240.0,60.0 242.5,60.0 245.0,60.0 247.5,60.0 250.0,60.0 252.5,60.0 255.0,60.0 257.5,60.0 260.0,60.0 262.5,60.0 265.0,60.0 267.5,60.0 270.0,60.0 272.5,60.0 275.0,60.0 277.5,60.0 280.0,60.0 282.5,60.0 285.0,60.0 287.5,60.0 290.0,60.0 292.5,60.0 295.0,60.0 297.5,60.0 300.0,60.0 302.5,60.0 305.0,60.0 307.5,60.0 310.0,60.0 312.5,60.0 315.0,60.0 317.5,60.0 320.0,60.0 322.5,60.0 325.0,60.0 327.5,60.0 330.0,60.0 332.5,60.0 335.0,60.0 337.5,60.0 340.0,60.0 342.5,60.0 345.0,60.0 347.5,60.0 350.0,60.0 352.5,60.0 355.0,60.0 357.5,60.0 360.0,60.0 362.5,60.0 365.0,60.0 367.5,60.0 370.0,60.0 372.5,60.0 375.0,60.0 377.5,60.0 380.0,60.0 382.5,60.0 385.0,60.0 387.5,60.0 390.0,60.0 392.5,60.0 395.0,60.0 397.5,60.0 400.0,60.0 402.5,60.0 405.0,60.0 407.5,60.0 410.0,60.0 412.5,60.0 415.0,60.0 417.5,60.0 420.0,60.0 422.5,60.0 425.0,60.0 427.5,60.0 430.0,60.0 432.5,60.0 435.0,60.0 437.5,60.0 440.0,60.0 442.5,60.0 445.0,60.0 447.5,60.0 450.0,60.0 452.5,60.0 455.0,60.0 457.5,60.0 460.0,60.0 462.5,60.0 465.0,60.0 467.5,60.0 470.0,60.0 472.5,60.0 475.0,60.0 477.5,60.0 480.0,60.0 482.5,60.0 485.0,60.0 487.5,60.0 490.0,60.0 492.5,60.0 495.0,60.0 497.5,60.0 500.0,60.0 502.5,60.0 505.0,60.0 507.5,60.0 510.0,60.0 512.5,60.0 515.0,60.0 517.5,60.0 520.0,60.0 522.5,60.0 525.0,60.0 527.5,60.0 530.0,60.0 532.5,60.0 535.0,60.0 537.5,60.0 540.0,60.0 542.5,60.0 545.0,60.0 547.5,60.0 550.0,60.0 552.5,60.0 555.0,60.0 557.5,60.0 560.0,60.0 562.5,60.0 565.0,60.0 567.5,60.0 570.0,60.0 572.5,60.0 575.0,60.0" fill="none" class="stroke-muted dashed"/>
<polyline points="75.0,260.0 77.5,250.0 80.0,240.0 82.5,230.0 85.0,220.0 87.5,210.0 90.0,200.0 92.5,190.0 95.0,180.0 97.5,170.0 100.0,160.0 102.5,150.0 105.0,140.0 107.5,130.0 110.0,120.0 112.5,110.0 115.0,100.0 117.5,90.0 120.0,80.0 122.5,70.0 125.0,60.0 127.5,61.1 130.0,62.2 132.5,63.3 135.0,64.4 137.5,65.6 140.0,66.7 142.5,67.8 145.0,68.9 147.5,70.0 150.0,71.1 152.5,72.2 155.0,73.3 157.5,74.4 160.0,75.6 162.5,76.7 165.0,77.8 167.5,78.9 170.0,80.0 172.5,81.1 175.0,82.2 177.5,83.3 180.0,84.4 182.5,85.6 185.0,86.7 187.5,87.8 190.0,88.9 192.5,90.0 195.0,91.1 197.5,92.2 200.0,93.3 202.5,94.4 205.0,95.6 207.5,96.7 210.0,97.8 212.5,98.9 215.0,100.0 217.5,101.1 220.0,102.2 222.5,103.3 225.0,104.4 227.5,105.6 230.0,106.7 232.5,107.8 235.0,108.9 237.5,110.0 240.0,111.1 242.5,112.2 245.0,113.3 247.5,114.4 250.0,115.6 252.5,116.7 255.0,117.8 257.5,118.9 260.0,120.0 262.5,121.1 265.0,122.2 267.5,123.3 270.0,124.4 272.5,125.6 275.0,126.7 277.5,127.8 280.0,128.9 282.5,130.0 285.0,131.1 287.5,132.2 290.0,133.3 292.5,134.4 295.0,135.6 297.5,136.7 300.0,137.8 302.5,138.9 305.0,140.0 307.5,141.1 310.0,142.2 312.5,143.3 315.0,144.4 317.5,145.6 320.0,146.7 322.5,147.8 325.0,148.9 327.5,150.0 330.0,151.1 332.5,152.2 335.0,153.3 337.5,154.4 340.0,155.6 342.5,156.7 345.0,157.8 347.5,158.9 350.0,160.0 352.5,161.1 355.0,162.2 357.5,163.3 360.0,164.4 362.5,165.6 365.0,166.7 367.5,167.8 370.0,168.9 372.5,170.0 375.0,171.1 377.5,172.2 380.0,173.3 382.5,174.4 385.0,175.6 387.5,176.7 390.0,177.8 392.5,178.9 395.0,180.0 397.5,181.1 400.0,182.2 402.5,183.3 405.0,184.4 407.5,185.6 410.0,186.7 412.5,187.8 415.0,188.9 417.5,190.0 420.0,191.1 422.5,192.2 425.0,193.3 427.5,194.4 430.0,195.6 432.5,196.7 435.0,197.8 437.5,198.9 440.0,200.0 442.5,201.1 445.0,202.2 447.5,203.3 450.0,204.4 452.5,205.6 455.0,206.7 457.5,207.8 460.0,208.9 462.5,210.0 465.0,211.1 467.5,212.2 470.0,213.3 472.5,214.4 475.0,215.6 477.5,216.7 480.0,217.8 482.5,218.9 485.0,220.0 487.5,221.1 490.0,222.2 492.5,223.3 495.0,224.4 497.5,225.6 500.0,226.7 502.5,227.8 505.0,228.9 507.5,230.0 510.0,231.1 512.5,232.2 515.0,233.3 517.5,234.4 520.0,235.6 522.5,236.7 525.0,237.8 527.5,238.9 530.0,240.0 532.5,241.1 535.0,242.2 537.5,243.3 540.0,244.4 542.5,245.6 545.0,246.7 547.5,247.8 550.0,248.9 552.5,250.0 555.0,251.1 557.5,252.2 560.0,253.3 562.5,254.4 565.0,255.6 567.5,256.7 570.0,257.8 572.5,258.9 575.0,260.0" fill="none" class="stroke-d"/>
<polyline points="75.0,260.0 77.5,250.0 80.0,240.0 82.5,230.0 85.0,220.0 87.5,210.0 90.0,200.0 92.5,190.0 95.0,180.0 97.5,170.0 100.0,160.0 102.5,150.0 105.0,140.0 107.5,130.0 110.0,120.0 112.5,110.0 115.0,100.0 117.5,90.0 120.0,80.0 122.5,70.0 125.0,60.0 127.5,60.0 130.0,60.1 132.5,60.1 135.0,60.2 137.5,60.4 140.0,60.5 142.5,60.7 145.0,61.0 147.5,61.2 150.0,61.5 152.5,61.8 155.0,62.2 157.5,62.6 160.0,63.0 162.5,63.4 165.0,63.9 167.5,64.4 170.0,64.9 172.5,65.4 175.0,66.0 177.5,66.6 180.0,67.3 182.5,67.9 185.0,68.6 187.5,69.4 190.0,70.1 192.5,70.9 195.0,71.7 197.5,72.5 200.0,73.4 202.5,74.3 205.0,75.2 207.5,76.1 210.0,77.1 212.5,78.1 215.0,79.1 217.5,80.1 220.0,81.2 222.5,82.3 225.0,83.4 227.5,84.5 230.0,85.7 232.5,86.9 235.0,88.1 237.5,89.3 240.0,90.5 242.5,91.8 245.0,93.1 247.5,94.4 250.0,95.7 252.5,97.1 255.0,98.4 257.5,99.8 260.0,101.2 262.5,102.6 265.0,104.1 267.5,105.5 270.0,107.0 272.5,108.5 275.0,110.0 277.5,111.5 280.0,113.1 282.5,114.6 285.0,116.2 287.5,117.7 290.0,119.3 292.5,120.9 295.0,122.5 297.5,124.2 300.0,125.8 302.5,127.4 305.0,129.1 307.5,130.8 310.0,132.4 312.5,134.1 315.0,135.8 317.5,137.5 320.0,139.2 322.5,140.9 325.0,142.6 327.5,144.4 330.0,146.1 332.5,147.8 335.0,149.5 337.5,151.3 340.0,153.0 342.5,154.8 345.0,156.5 347.5,158.3 350.0,160.0 352.5,161.7 355.0,163.5 357.5,165.2 360.0,167.0 362.5,168.7 365.0,170.5 367.5,172.2 370.0,173.9 372.5,175.6 375.0,177.4 377.5,179.1 380.0,180.8 382.5,182.5 385.0,184.2 387.5,185.9 390.0,187.6 392.5,189.2 395.0,190.9 397.5,192.6 400.0,194.2 402.5,195.8 405.0,197.5 407.5,199.1 410.0,200.7 412.5,202.3 415.0,203.8 417.5,205.4 420.0,206.9 422.5,208.5 425.0,210.0 427.5,211.5 430.0,213.0 432.5,214.5 435.0,215.9 437.5,217.4 440.0,218.8 442.5,220.2 445.0,221.6 447.5,222.9 450.0,224.3 452.5,225.6 455.0,226.9 457.5,228.2 460.0,229.5 462.5,230.7 465.0,231.9 467.5,233.1 470.0,234.3 472.5,235.5 475.0,236.6 477.5,237.7 480.0,238.8 482.5,239.9 485.0,240.9 487.5,241.9 490.0,242.9 492.5,243.9 495.0,244.8 497.5,245.7 500.0,246.6 502.5,247.5 505.0,248.3 507.5,249.1 510.0,249.9 512.5,250.6 515.0,251.4 517.5,252.1 520.0,252.7 522.5,253.4 525.0,254.0 527.5,254.6 530.0,255.1 532.5,255.6 535.0,256.1 537.5,256.6 540.0,257.0 542.5,257.4 545.0,257.8 547.5,258.2 550.0,258.5 552.5,258.8 555.0,259.0 557.5,259.3 560.0,259.5 562.5,259.6 565.0,259.8 567.5,259.9 570.0,259.9 572.5,260.0 575.0,260.0" fill="none" class="stroke-a"/>
<line x1="90.0" y1="310.0" x2="120.0" y2="310.0" class="stroke-a"/>
<text x="126.0" y="314.0" class="fig-small" text-anchor="start">cosine</text>
<line x1="240.0" y1="310.0" x2="270.0" y2="310.0" class="stroke-d"/>
<text x="276.0" y="314.0" class="fig-small" text-anchor="start">linear</text>
<line x1="390.0" y1="310.0" x2="420.0" y2="310.0" class="stroke-muted dashed"/>
<text x="426.0" y="314.0" class="fig-small" text-anchor="start">constant</text>
</svg>
Figure 2. Three ways to treat the rate after warmup, all with peak 0.01. Cosine (blue) stays near the peak early: at step 325 it is 0.00854, while linear decay (purple) is already at 0.0075. Cosine falls fastest in the middle and flattens as it reaches 0. A constant rate (dashed) never settles.
```

> [!TIP]
> Check four points before you trust a schedule: step 0 gives 0, the end of warmup gives the peak, the middle of the decay gives half the peak, and the last step gives 0.

If `warmup >= total`, there is no decay phase, so dividing by `total - warmup` would divide by zero or a negative number. In that case rise through warmup and hold the peak afterwards.

## Watch both ends improve

Run Adam with peak rate 0.1 on mini-batches of eight rows from the valley of an earlier chapter, once with a constant rate and once with 20 warmup steps followed by cosine decay over 400 steps. Figure 3 shows both runs. At the end of the run, the constant rate hovers at an excess loss of 0.147, averaged over 10 runs, and the schedule reaches 0.003. That is about fifty times lower with the same peak rate and no extra data.

```figure
<svg viewBox="0 0 600 360" role="img" aria-label="Excess loss for Adam with a constant rate and with warmup plus cosine decay, over the whole run and over the first 40 steps">
<defs></defs>
<line x1="60.0" y1="260.0" x2="300.0" y2="260.0" class="stroke-muted"/>
<line x1="60.0" y1="40.0" x2="60.0" y2="260.0" class="stroke-muted"/>
<line x1="60.0" y1="260.0" x2="60.0" y2="265.0" class="stroke-muted"/>
<text x="60.0" y="279.0" class="fig-small" text-anchor="middle">0</text>
<line x1="180.0" y1="260.0" x2="180.0" y2="265.0" class="stroke-muted"/>
<text x="180.0" y="279.0" class="fig-small" text-anchor="middle">200</text>
<line x1="300.0" y1="260.0" x2="300.0" y2="265.0" class="stroke-muted"/>
<text x="300.0" y="279.0" class="fig-small" text-anchor="middle">400</text>
<line x1="60.0" y1="235.6" x2="300.0" y2="235.6" class="stroke-grid"/>
<line x1="55.0" y1="235.6" x2="60.0" y2="235.6" class="stroke-muted"/>
<text x="51.0" y="239.6" class="fig-small" text-anchor="end">1e-3</text>
<line x1="60.0" y1="186.7" x2="300.0" y2="186.7" class="stroke-grid"/>
<line x1="55.0" y1="186.7" x2="60.0" y2="186.7" class="stroke-muted"/>
<text x="51.0" y="190.7" class="fig-small" text-anchor="end">1e-2</text>
<line x1="60.0" y1="137.8" x2="300.0" y2="137.8" class="stroke-grid"/>
<line x1="55.0" y1="137.8" x2="60.0" y2="137.8" class="stroke-muted"/>
<text x="51.0" y="141.8" class="fig-small" text-anchor="end">1e-1</text>
<line x1="60.0" y1="88.9" x2="300.0" y2="88.9" class="stroke-grid"/>
<line x1="55.0" y1="88.9" x2="60.0" y2="88.9" class="stroke-muted"/>
<text x="51.0" y="92.9" class="fig-small" text-anchor="end">1</text>
<text x="180.0" y="298.0" class="fig-small" text-anchor="middle">step</text>
<polyline points="60.0,57.8 60.6,63.1 61.2,71.0 61.8,75.0 62.4,75.2 63.0,74.0 63.6,74.0 64.2,76.2 64.8,79.2 65.4,83.7 66.0,88.4 66.6,92.3 67.2,94.7 67.8,95.3 68.4,96.2 69.0,97.1 69.6,98.2 70.2,100.4 70.8,104.0 71.4,109.0 72.0,114.7 72.6,120.8 73.2,124.4 73.8,125.6 74.4,124.4 75.0,121.9 75.6,120.6 76.2,120.2 76.8,123.0 77.4,126.9 78.0,132.1 78.6,137.1 79.2,141.3 79.8,145.2 80.4,143.2 81.0,141.8 81.6,139.3 82.2,137.5 82.8,136.7 83.4,135.8 84.0,136.3 84.6,138.9 85.2,141.6 85.8,147.1 86.4,151.0 87.0,155.8 87.6,157.2 88.2,153.2 88.8,148.4 89.4,146.5 90.0,144.6 90.6,144.7 91.2,146.2 91.8,148.8 92.4,152.4 93.0,156.9 93.6,159.8 94.2,161.7 94.8,163.4 95.4,160.8 96.0,156.1 96.6,156.2 97.2,155.6 97.8,153.8 98.4,154.8 99.0,154.9 99.6,158.4 100.2,165.0 100.8,169.9 101.4,169.1 102.0,164.3 102.6,157.8 103.2,155.1 103.8,155.7 104.4,159.2 105.0,172.7 105.6,183.7 106.2,184.3 106.8,167.5 107.4,156.3 108.0,147.1 108.6,141.0 109.2,140.8 109.8,150.9 110.4,167.9 111.0,172.0 111.6,165.5 112.2,155.0 112.8,147.9 113.4,143.0 114.0,141.4 114.6,150.7 115.2,161.8 115.8,158.9 116.4,150.5 117.0,143.1 117.6,138.5 118.2,136.2 118.8,142.0 119.4,150.9 120.0,159.2 120.6,156.7 121.2,146.6 121.8,145.1 122.4,145.6 123.0,151.3 123.6,155.7 124.2,162.4 124.8,171.1 125.4,171.9 126.0,166.6 126.6,158.4 127.2,154.0 127.8,152.0 128.4,153.6 129.0,156.1 129.6,155.2 130.2,149.4 130.8,142.6 131.4,136.3 132.0,133.9 132.6,135.3 133.2,141.1 133.8,149.5 134.4,161.8 135.0,166.5 135.6,163.3 136.2,157.8 136.8,151.6 137.4,151.1 138.0,160.0 138.6,170.8 139.2,173.8 139.8,160.3 140.4,147.0 141.0,142.6 141.6,140.3 142.2,142.4 142.8,152.5 143.4,161.6 144.0,162.8 144.6,152.0 145.2,139.9 145.8,132.9 146.4,131.5 147.0,133.1 147.6,134.8 148.2,136.5 148.8,138.5 149.4,141.4 150.0,139.7 150.6,138.7 151.2,141.8 151.8,145.2 152.4,146.7 153.0,144.2 153.6,140.0 154.2,139.3 154.8,140.0 155.4,140.5 156.0,140.3 156.6,145.3 157.2,151.1 157.8,153.5 158.4,148.6 159.0,146.0 159.6,144.6 160.2,149.4 160.8,153.1 161.4,149.3 162.0,143.8 162.6,140.8 163.2,146.3 163.8,156.4 164.4,159.1 165.0,152.2 165.6,147.3 166.2,143.4 166.8,139.6 167.4,139.1 168.0,142.1 168.6,143.2 169.2,146.3 169.8,149.0 170.4,154.7 171.0,160.2 171.6,161.1 172.2,156.1 172.8,157.6 173.4,159.5 174.0,164.0 174.6,160.7 175.2,154.5 175.8,148.4 176.4,147.9 177.0,149.4 177.6,153.4 178.2,154.7 178.8,155.9 179.4,148.2 180.0,142.7 180.6,138.9 181.2,139.7 181.8,145.6 182.4,154.7 183.0,157.0 183.6,148.5 184.2,137.8 184.8,131.1 185.4,128.9 186.0,132.9 186.6,140.6 187.2,149.1 187.8,155.0 188.4,150.7 189.0,141.7 189.6,137.0 190.2,136.7 190.8,145.2 191.4,155.5 192.0,156.0 192.6,150.1 193.2,143.9 193.8,141.1 194.4,144.7 195.0,151.3 195.6,149.1 196.2,147.3 196.8,147.1 197.4,146.8 198.0,143.0 198.6,142.7 199.2,147.9 199.8,152.4 200.4,150.8 201.0,146.5 201.6,141.5 202.2,139.0 202.8,138.9 203.4,147.2 204.0,158.3 204.6,159.3 205.2,149.0 205.8,138.1 206.4,132.9 207.0,132.7 207.6,133.5 208.2,136.4 208.8,146.7 209.4,155.0 210.0,149.5 210.6,140.5 211.2,139.9 211.8,141.8 212.4,143.9 213.0,150.0 213.6,156.5 214.2,154.6 214.8,149.7 215.4,141.8 216.0,139.0 216.6,142.0 217.2,155.9 217.8,158.6 218.4,152.3 219.0,147.9 219.6,143.7 220.2,139.5 220.8,137.0 221.4,144.3 222.0,156.1 222.6,177.3 223.2,172.1 223.8,157.5 224.4,147.1 225.0,139.4 225.6,135.2 226.2,135.7 226.8,138.7 227.4,147.7 228.0,160.1 228.6,167.0 229.2,156.6 229.8,149.1 230.4,145.0 231.0,148.1 231.6,152.9 232.2,160.1 232.8,166.0 233.4,161.6 234.0,154.8 234.6,146.0 235.2,144.9 235.8,145.6 236.4,149.4 237.0,147.1 237.6,144.5 238.2,140.8 238.8,150.1 239.4,161.0 240.0,165.8 240.6,158.4 241.2,153.3 241.8,152.4 242.4,154.0 243.0,161.9 243.6,162.3 244.2,161.5 244.8,157.6 245.4,150.3 246.0,145.5 246.6,145.8 247.2,150.8 247.8,158.8 248.4,160.1 249.0,154.0 249.6,145.5 250.2,141.9 250.8,144.1 251.4,149.3 252.0,153.4 252.6,156.3 253.2,154.3 253.8,152.1 254.4,151.1 255.0,149.0 255.6,148.6 256.2,147.0 256.8,143.8 257.4,143.3 258.0,147.4 258.6,157.8 259.2,160.7 259.8,159.9 260.4,154.3 261.0,152.1 261.6,152.0 262.2,154.6 262.8,153.7 263.4,154.9 264.0,158.3 264.6,156.4 265.2,151.5 265.8,148.7 266.4,150.8 267.0,151.0 267.6,150.8 268.2,149.7 268.8,146.5 269.4,141.7 270.0,138.7 270.6,140.3 271.2,144.1 271.8,149.7 272.4,157.1 273.0,161.8 273.6,156.1 274.2,147.4 274.8,142.5 275.4,140.3 276.0,143.2 276.6,147.4 277.2,145.9 277.8,142.1 278.4,137.9 279.0,137.6 279.6,134.9 280.2,132.2 280.8,131.5 281.4,130.8 282.0,132.8 282.6,133.1 283.2,137.4 283.8,144.7 284.4,150.7 285.0,141.5 285.6,128.8 286.2,121.2 286.8,116.6 287.4,119.7 288.0,129.0 288.6,139.4 289.2,138.6 289.8,125.6 290.4,113.9 291.0,109.6 291.6,118.5 292.2,132.6 292.8,137.1 293.4,129.1 294.0,119.0 294.6,110.7 295.2,105.6 295.8,107.8 296.4,119.1 297.0,132.8 297.6,143.6 298.2,138.2 298.8,130.0 299.4,126.4 300.0,123.8" fill="none" class="stroke-b"/>
<polyline points="60.0,57.8 60.6,57.8 61.2,58.3 61.8,59.3 62.4,60.8 63.0,62.7 63.6,65.1 64.2,67.8 64.8,70.8 65.4,73.8 66.0,76.4 66.6,78.2 67.2,79.0 67.8,79.0 68.4,78.6 69.0,78.4 69.6,78.3 70.2,78.8 70.8,80.8 71.4,83.4 72.0,86.3 72.6,90.3 73.2,95.0 73.8,99.5 74.4,103.3 75.0,106.0 75.6,107.6 76.2,108.4 76.8,109.4 77.4,111.0 78.0,113.4 78.6,116.5 79.2,120.6 79.8,126.2 80.4,131.9 81.0,137.0 81.6,141.5 82.2,143.0 82.8,141.9 83.4,141.5 84.0,142.4 84.6,144.8 85.2,147.1 85.8,149.9 86.4,154.9 87.0,158.3 87.6,158.8 88.2,155.5 88.8,153.2 89.4,150.5 90.0,146.8 90.6,144.5 91.2,143.4 91.8,142.8 92.4,145.0 93.0,148.9 93.6,153.9 94.2,158.7 94.8,162.0 95.4,161.7 96.0,156.1 96.6,152.0 97.2,149.5 97.8,148.4 98.4,148.5 99.0,148.8 99.6,150.7 100.2,156.3 100.8,161.4 101.4,161.0 102.0,159.1 102.6,157.1 103.2,155.3 103.8,154.3 104.4,153.9 105.0,160.6 105.6,168.7 106.2,178.4 106.8,178.9 107.4,172.9 108.0,164.9 108.6,158.4 109.2,155.1 109.8,157.0 110.4,161.5 111.0,165.7 111.6,172.5 112.2,179.9 112.8,178.1 113.4,173.4 114.0,165.0 114.6,164.7 115.2,164.7 115.8,162.0 116.4,158.8 117.0,155.9 117.6,154.6 118.2,153.0 118.8,158.2 119.4,163.9 120.0,169.3 120.6,166.9 121.2,162.4 121.8,157.4 122.4,153.8 123.0,153.9 123.6,156.0 124.2,161.1 124.8,167.7 125.4,173.2 126.0,171.5 126.6,172.2 127.2,168.9 127.8,163.7 128.4,159.5 129.0,155.5 129.6,152.2 130.2,151.7 130.8,150.9 131.4,147.9 132.0,143.9 132.6,142.2 133.2,143.1 133.8,143.4 134.4,147.8 135.0,151.0 135.6,153.3 136.2,153.2 136.8,153.7 137.4,154.7 138.0,162.6 138.6,169.3 139.2,170.7 139.8,161.9 140.4,152.2 141.0,148.8 141.6,145.8 142.2,146.7 142.8,155.1 143.4,164.7 144.0,170.9 144.6,166.0 145.2,155.0 145.8,144.7 146.4,139.2 147.0,137.0 147.6,137.3 148.2,139.1 148.8,142.6 149.4,147.9 150.0,153.1 150.6,157.2 151.2,163.1 151.8,167.9 152.4,169.2 153.0,169.1 153.6,166.5 154.2,166.9 154.8,169.5 155.4,172.7 156.0,172.8 156.6,170.0 157.2,169.7 157.8,170.6 158.4,170.5 159.0,173.8 159.6,174.4 160.2,175.4 160.8,174.4 161.4,172.7 162.0,169.0 162.6,169.3 163.2,170.6 163.8,172.5 164.4,172.7 165.0,167.3 165.6,165.4 166.2,161.3 166.8,158.9 167.4,158.0 168.0,158.0 168.6,154.6 169.2,153.0 169.8,151.1 170.4,153.5 171.0,156.1 171.6,158.7 172.2,161.4 172.8,163.8 173.4,165.7 174.0,163.3 174.6,158.7 175.2,155.5 175.8,155.3 176.4,154.4 177.0,155.0 177.6,157.9 178.2,159.9 178.8,162.3 179.4,164.6 180.0,168.1 180.6,168.2 181.2,168.2 181.8,166.7 182.4,168.0 183.0,171.6 183.6,174.7 184.2,174.2 184.8,173.8 185.4,175.2 186.0,179.6 186.6,178.9 187.2,177.0 187.8,173.7 188.4,171.0 189.0,167.5 189.6,165.9 190.2,166.5 190.8,174.0 191.4,181.5 192.0,190.1 192.6,194.2 193.2,190.9 193.8,186.4 194.4,179.7 195.0,176.2 195.6,172.8 196.2,170.9 196.8,171.8 197.4,179.4 198.0,187.1 198.6,194.2 199.2,197.7 199.8,194.6 200.4,188.5 201.0,182.2 201.6,175.8 202.2,172.1 202.8,167.9 203.4,164.3 204.0,161.0 204.6,160.0 205.2,161.2 205.8,164.1 206.4,166.2 207.0,168.5 207.6,172.2 208.2,172.7 208.8,173.6 209.4,173.8 210.0,171.9 210.6,168.2 211.2,165.5 211.8,164.1 212.4,163.8 213.0,167.7 213.6,173.0 214.2,179.7 214.8,190.7 215.4,203.2 216.0,210.7 216.6,205.1 217.2,196.1 217.8,186.4 218.4,177.8 219.0,173.2 219.6,170.6 220.2,169.3 220.8,169.9 221.4,172.8 222.0,176.9 222.6,180.9 223.2,185.8 223.8,190.6 224.4,194.2 225.0,195.4 225.6,194.3 226.2,193.2 226.8,188.3 227.4,184.5 228.0,180.9 228.6,177.6 229.2,173.4 229.8,171.1 230.4,170.3 231.0,171.4 231.6,174.3 232.2,176.5 232.8,177.7 233.4,178.8 234.0,180.4 234.6,182.2 235.2,185.5 235.8,188.5 236.4,192.4 237.0,193.1 237.6,193.5 238.2,191.3 238.8,187.9 239.4,184.7 240.0,182.2 240.6,180.8 241.2,179.3 241.8,177.6 242.4,176.0 243.0,175.0 243.6,174.3 244.2,175.3 244.8,176.7 245.4,177.9 246.0,179.4 246.6,179.9 247.2,181.1 247.8,183.1 248.4,184.8 249.0,185.5 249.6,186.8 250.2,188.1 250.8,188.1 251.4,188.4 252.0,188.5 252.6,188.3 253.2,188.9 253.8,188.8 254.4,189.2 255.0,191.0 255.6,192.4 256.2,193.4 256.8,194.6 257.4,195.5 258.0,196.4 258.6,197.1 259.2,198.3 259.8,199.3 260.4,200.6 261.0,202.1 261.6,203.7 262.2,205.5 262.8,207.2 263.4,208.4 264.0,208.8 264.6,209.4 265.2,209.4 265.8,208.8 266.4,208.1 267.0,207.1 267.6,206.5 268.2,206.4 268.8,206.7 269.4,206.4 270.0,206.5 270.6,206.8 271.2,206.8 271.8,207.1 272.4,207.1 273.0,206.8 273.6,206.6 274.2,206.6 274.8,206.3 275.4,206.2 276.0,206.3 276.6,206.3 277.2,206.0 277.8,205.9 278.4,206.4 279.0,206.8 279.6,207.3 280.2,207.8 280.8,208.3 281.4,208.9 282.0,209.6 282.6,210.3 283.2,210.8 283.8,211.4 284.4,211.9 285.0,212.4 285.6,212.9 286.2,213.6 286.8,214.0 287.4,214.4 288.0,214.9 288.6,215.4 289.2,215.8 289.8,216.2 290.4,216.7 291.0,217.2 291.6,217.6 292.2,218.1 292.8,218.4 293.4,218.8 294.0,219.1 294.6,219.3 295.2,219.5 295.8,219.6 296.4,219.7 297.0,219.8 297.6,219.9 298.2,219.9 298.8,219.9 299.4,219.9 300.0,219.9" fill="none" class="stroke-a"/>
<text x="60.0" y="34.0" class="fig-small" text-anchor="start">excess loss</text>
<text x="180.0" y="14.0" class="fig-bold" text-anchor="middle">whole run, start far from the bottom</text>
<line x1="350.0" y1="260.0" x2="580.0" y2="260.0" class="stroke-muted"/>
<line x1="350.0" y1="40.0" x2="350.0" y2="260.0" class="stroke-muted"/>
<line x1="350.0" y1="260.0" x2="350.0" y2="265.0" class="stroke-muted"/>
<text x="350.0" y="279.0" class="fig-small" text-anchor="middle">0</text>
<line x1="407.5" y1="260.0" x2="407.5" y2="265.0" class="stroke-muted"/>
<text x="407.5" y="279.0" class="fig-small" text-anchor="middle">10</text>
<line x1="465.0" y1="260.0" x2="465.0" y2="265.0" class="stroke-muted"/>
<text x="465.0" y="279.0" class="fig-small" text-anchor="middle">20</text>
<line x1="522.5" y1="260.0" x2="522.5" y2="265.0" class="stroke-muted"/>
<text x="522.5" y="279.0" class="fig-small" text-anchor="middle">30</text>
<line x1="580.0" y1="260.0" x2="580.0" y2="265.0" class="stroke-muted"/>
<text x="580.0" y="279.0" class="fig-small" text-anchor="middle">40</text>
<line x1="350.0" y1="205.0" x2="580.0" y2="205.0" class="stroke-grid"/>
<line x1="345.0" y1="205.0" x2="350.0" y2="205.0" class="stroke-muted"/>
<text x="341.0" y="209.0" class="fig-small" text-anchor="end">1e-2</text>
<line x1="350.0" y1="150.0" x2="580.0" y2="150.0" class="stroke-grid"/>
<line x1="345.0" y1="150.0" x2="350.0" y2="150.0" class="stroke-muted"/>
<text x="341.0" y="154.0" class="fig-small" text-anchor="end">1e-1</text>
<line x1="350.0" y1="95.0" x2="580.0" y2="95.0" class="stroke-grid"/>
<line x1="345.0" y1="95.0" x2="350.0" y2="95.0" class="stroke-muted"/>
<text x="341.0" y="99.0" class="fig-small" text-anchor="end">1</text>
<text x="465.0" y="298.0" class="fig-small" text-anchor="middle">step</text>
<polyline points="350.0,205.0 355.8,113.6 361.5,148.6 367.2,163.0 373.0,136.2 378.8,127.7 384.5,125.1 390.2,133.0 396.0,141.0 401.8,153.4 407.5,151.8 413.2,140.1 419.0,134.9 424.8,133.1 430.5,138.2 436.2,142.7 442.0,145.7 447.8,146.8 453.5,142.0 459.2,137.7 465.0,134.7 470.8,134.2 476.5,138.7 482.2,147.8 488.0,154.4 493.8,153.2 499.5,151.5 505.2,146.1 511.0,145.3 516.8,147.1 522.5,148.5 528.2,151.4 534.0,151.8 539.8,152.8 545.5,149.1 551.2,146.5 557.0,143.6 562.8,144.8 568.5,146.6 574.2,154.0 580.0,159.6" fill="none" class="stroke-b"/>
<polyline points="350.0,205.0 355.8,205.0 361.5,209.0 367.2,220.1 373.0,238.0 378.8,219.4 384.5,194.0 390.2,174.0 396.0,161.7 401.8,162.5 407.5,167.2 413.2,171.7 419.0,181.6 424.8,176.0 430.5,164.5 436.2,153.3 442.0,145.1 447.8,145.1 453.5,148.5 459.2,152.4 465.0,151.5 470.8,146.4 476.5,150.0 482.2,161.6 488.0,165.5 493.8,157.6 499.5,151.7 505.2,145.1 511.0,148.1 516.8,152.4 522.5,158.0 528.2,162.1 534.0,157.9 539.8,152.4 545.5,143.6 551.2,139.4 557.0,141.1 562.8,147.2 568.5,153.6 574.2,158.3 580.0,155.3" fill="none" class="stroke-a"/>
<text x="350.0" y="34.0" class="fig-small" text-anchor="start">excess loss</text>
<text x="465.0" y="14.0" class="fig-bold" text-anchor="middle">first 40 steps, start near the bottom</text>
<line x1="130.0" y1="345.0" x2="160.0" y2="345.0" class="stroke-b"/>
<text x="166.0" y="349.0" class="fig-small" text-anchor="start">constant rate 0.1</text>
<line x1="330.0" y1="345.0" x2="360.0" y2="345.0" class="stroke-a"/>
<text x="366.0" y="349.0" class="fig-small" text-anchor="start">warmup 20, then cosine</text>
</svg>
Figure 3. Adam with peak rate 0.1 on mini-batches of eight rows from the valley of an earlier chapter, with each curve the average excess loss of 10 runs. Left: the constant rate (orange) hovers on a plateau and never settles, while the schedule (blue) drops below it once the decay lowers the rate. Over the last 50 steps the constant rate averages 0.147 and the schedule 0.0030. Right: from a start 0.014 from the bottom, the first full-size step kicks the loss up to 0.46 on average (orange), while warmup keeps it below 0.07 over the first 10 steps (blue). Once warmup ends at step 20 the rate is 0.1, and the loss swings again until the decay lowers the rate.
```

The start is a smaller problem, but it is real, and you can see why in the right panel. There the walk starts only 0.014 from the bottom along the steep axis, which is an excess loss of 0.01. Adam's first step is one full learning rate in every coordinate (see "Adam: a step size for every weight"), so the steep weight lurches by 0.1, about seven times its distance from the bottom, and overshoots whichever way the first batch points. Averaged over 10 runs, the loss jumps from 0.01 to 0.46. With warmup the first step has rate 0, and the loss stays below 0.07 for the first 10 steps. The left panel starts far from the bottom, so no kick shows: the early loss is high anyway. Warmup protects a good start, and decay improves the end.

> [!KEY]
> The learning rate should change with training. Small steps early give Adam's averages time to fill, a large rate in the middle makes progress, and a rate that falls to zero at the end lets the noisy weights settle.

> [!RECAP]
> Hold these five points before you write code:
>
> - Linear warmup gives `peak * step / warmup` while `step < warmup`, and the peak afterwards.
> - The decay fraction is `0.5 * (1 + cos(pi * p))`, with `p` clamped to `[0, 1]`.
> - `p` runs from 0 at the end of warmup to 1 at the last step.
> - The four checks are 0, the peak, half the peak, and 0.
> - With `warmup >= total`, skip the decay and hold the peak after warmup.

## Build it step by step

The starter gives you `make_valley`, `adam_step`, and `run_with_schedule`. The last one calls `schedule_fn(k)` for each step $k$, passes that rate to the step function, and returns the final weights, the history, and the list of rates. You write three functions, in the order the checks test them:

1. **`linear_warmup(step, warmup, peak)`.** Return `peak * step / warmup` for `step < warmup`, and `peak` otherwise, including when `warmup` is 0. Raise `ValueError` for a negative `step`. The usual mistake is dividing by `warmup` when it is 0.
2. **`cosine_decay_fraction(p)`.** Clamp `p` to `[0, 1]` and return `0.5 * (1 + np.cos(np.pi * p))`. The usual mistake is skipping the clamp, so that steps past the end bounce back up.
3. **`cosine_schedule(step, warmup, total, peak)`.** Use `linear_warmup` while `step < warmup`. Otherwise compute `p` and return `peak * cosine_decay_fraction(p)`. When `warmup >= total`, return `linear_warmup(step, warmup, peak)`. The usual mistake is measuring `p` from step 0 instead of from the end of warmup.

## Where this breaks

Each of these has a direct remedy:

- A cosine schedule needs the total number of steps in advance. If you stop early, the rate has not decayed. Fix the total before you start.
- Zero is not always the best ending. Many runs decay to a small fraction of the peak, such as one tenth, so late steps still make some progress.
- A warmup that is too short still allows an early jump. Lengthen it, or lower the peak.
- A peak that is too high wastes the middle of the run in noise. Lower it, and check the loss curve against a constant rate.

## The exercise

You implement `linear_warmup`, `cosine_decay_fraction`, and `cosine_schedule`. The starter supplies `run_with_schedule`, which runs a step function with the schedule's rate at each step.

The checks cover the warmup line and its edge cases, the cosine factor at $p = 0, 0.25, 0.5, 1$, and the table's rates at steps 0, 50, 100, 325, 550, and 1000. They confirm that steps past the end stay at 0, that the rate never rises during decay, and that `warmup >= total` does not divide by zero. Finally they run Adam on noisy batches and check that the scheduled run ends much lower than a constant rate.
