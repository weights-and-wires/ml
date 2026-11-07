---
title: "SGD, Adam, and learning-rate schedules"
slug: 029-sgd-adam-and-schedules
part: "Part 4 — Neural networks"
summary: Take cheap steps from a few rows, give every weight its own step size with Adam, decay weights with AdamW, and shape the rate over time.
minutes: 80
gpu: false
objectives:
  - Compute a mini-batch gradient and explain why it equals the full gradient on average but sets a floor on the loss.
  - Take SGD and momentum steps, and explain why momentum adds up gradients that agree and cancels gradients that flip.
  - Build an exponential moving average, correct it for starting at zero, and take an Adam step.
  - Show why an L2 penalty inside Adam shrinks weights unevenly, and fix it with AdamW.
  - Compute a linear warmup and a cosine decay, and measure how much lower Adam settles with them.
prereqs:
  - 005-gradient-descent-and-momentum
  - 006-ridge-regression
  - 028-softmax-and-whole-network-gradients
exercise: 029-sgd-adam-and-schedules
---

# SGD, Adam, and learning-rate schedules

You fit a model to 200 rows of data, and the loss is the long narrow valley from earlier in the course: it bends 1 unit along one direction and 100 units along the other. The exact gradient needs every row. With 200 million rows, a normal size for a network, one step costs a full pass over the data, and you need hundreds of steps. The obvious shortcut is to compute the gradient from eight rows. That gradient is wrong, so why does it still lead downhill, and what does its error do to the walk?

Two more problems follow. A bias might see a gradient of 3 while a weight on a huge feature sees 3000, and one learning rate fits at most one of them. The loss curve also has two ugly ends: it jumps in the first few steps, and it wiggles on a plateau at the end.

This chapter builds one training recipe, in the order the problems appear: mini-batches for cheap steps, momentum for faster ones, Adam for a ruler per weight, AdamW for a weight penalty that respects that ruler, and a schedule that changes the rate over time.

## What you already know

Three ideas carry this chapter:

- A gradient step moves the weights against the slope, the learning rate must stay below $2/L$ on a curvature $L$, and momentum remembers the last step to speed up along a long valley (see [gradient descent and momentum](/lessons/005-gradient-descent-and-momentum)).
- A ridge penalty charges a fee on the squared size of the weights, and its gradient pulls each weight toward zero (see [ridge regression](/lessons/006-ridge-regression)).
- Backpropagation returns the gradient of the loss for every weight (see [whole-network gradients](/lessons/028-softmax-and-whole-network-gradients)).

## Use a few rows per step

Here is the valley as a real fitting problem. The data has 200 rows and two input columns, with spread 1 and spread 10. The loss is half the mean squared miss, with curvature 1 along $w_1$ and $10 \times 10 = 100$ along $w_2$. The bottom sits at $(1.04,\ 0.09)$, and you start at $(-1,\ 0.3)$. Subtract the bottom's loss and call the rest the **excess loss**: it is 4.33 at the start. With the exact gradient, the steep axis needs $\text{lr} < 0.02$, and at $\text{lr} = 0.0095$ the flat axis shrinks by only 0.9905 per step, so cutting its distance to a hundredth takes 482 steps.

The loss is an average over rows, so the gradient is too. Row $i$ has its own gradient $g_i$, which says how the loss on that row alone changes when you nudge the weights. A **mini-batch** is a small set of rows, picked at random. Its gradient averages only those rows, and a step with it is **stochastic gradient descent** (SGD), where stochastic means random.

Take one weight and four rows whose gradients are 2, −1, 4, and 3. The full gradient is $(2 - 1 + 4 + 3)/4 = 2$. Draw batches of two rows. There are six pairs, and their averages are 0.5, 3, 2.5, 1.5, 1, and 3.5. Each batch is off by as much as 1.5, yet the six averages themselves average 2. A batch gradient is wrong by a random amount with no preferred direction, which statisticians call **unbiased**. For a batch $B$ of $b$ rows:

$$
g_B = \frac{1}{b} \sum_{i \in B} g_i
$$

Read it aloud: add the gradients of the chosen rows and divide by how many you chose. For least squares, row $i$'s gradient is the row times its own miss, $x_i\,(x_i \cdot w - y_i)$. In numpy:

```python
resid = X[idx] @ w - y[idx]
grad = X[idx].T @ resid / len(idx)
```

At the bottom the exact gradient is zero, but a batch gradient is not, because eight rows disagree about where the bottom is. Each step still moves by the learning rate times that disagreement, so the walk hovers around the bottom. Why not shrink the learning rate until the hovering vanishes? Averaged over 20 runs, a rate of 0.0025 puts the floor about eight times lower than 0.0095, but reaching an excess loss of 0.5 takes 290 steps instead of 77. A large rate jitters, and a small rate crawls.

> [!KEY]
> A mini-batch gradient is right on average and wrong at each step. The wrongness keeps the walk from settling, and a smaller learning rate lowers that floor at the cost of speed.

## Let a ball add the steps up

A rolling ball does not restart at every bump. It keeps its speed, so pushes in the same direction add up and pushes that flip cancel. That is **momentum**. It keeps a velocity $v$, a running sum of past gradients with weights $1, \beta, \beta^2, \ldots$, and steps along it. Storing the velocity gives the same steps as the gradient-plus-last-step form from [gradient descent and momentum](/lessons/005-gradient-descent-and-momentum), and it is easier to carry between calls:

$$
v \leftarrow \beta\, v + g, \qquad w \leftarrow w - \text{lr}\cdot v
$$

Read it aloud: keep a fraction $\beta$ of the old velocity, add the fresh gradient, and step along the result. Take $\beta = 0.9$. A steady gradient of 2 and a gradient that flips between +2 and −2 build very different velocities:

| Step | Steady gradient 2 | Gradient +2, −2, +2, ... |
|---|---|---|
| 1 | 2.0 | 2.0 |
| 2 | 3.8 | −0.2 |
| 3 | 5.42 | 1.82 |
| 4 | 6.878 | −0.362 |
| many | 20.0 | about ±1.05 |

The steady gradient climbs to $2/(1 - 0.9) = 20$, ten times its size. The flipping gradient stays near 1.05, because each push undoes most of the last one. In numpy:

```python
v = beta * state.get("v", np.zeros_like(w)) + grad
new_w = w - lr * v
```

Batch noise acts like the flipping gradient: it partly cancels in $v$, while the true slope keeps its sign and adds up. Independent noise of size $\sigma$ grows in $v$ only to $\sigma/\sqrt{1 - \beta^2} = 2.3\,\sigma$, while the slope grows tenfold.

Why not raise the plain learning rate tenfold, to 0.095? On the steep axis a plain step multiplies the distance by $1 - 100 \times 0.095 = -8.5$, so the walk explodes. Momentum takes long steps only where the gradient keeps its sign, which is the flat floor, and on the steep axis the flips cancel in $v$. It stays stable while $\text{lr} \times \text{curvature} < 2(1 + \beta) = 3.8$, which is $\text{lr} < 0.038$ here, against 0.02 for a plain step. The price is that steady steps are $1/(1 - \beta) = 10$ times longer than the learning rate suggests, so you lower the rate when you raise $\beta$.

Race plain SGD at 0.0095 against momentum at $\text{lr} = 0.0025$ with $\beta = 0.9$. Averaged over 20 runs with batches of eight rows, SGD reaches an excess loss of 0.5 in 77 steps and momentum in 37. Figure 1 shows one run of each. Momentum's floor is higher, about 0.13 against 0.07, because its steps are longer. Give it SGD's effective step, with $\text{lr} = 0.00095$, and it needs 82 steps but its floor falls to 0.03. Momentum does not remove the trade-off between jitter and speed.

```figure
<svg viewBox="0 0 600 350" role="img" aria-label="Excess loss against step for SGD and for momentum on mini-batches of eight rows">
<defs></defs>
<line x1="70.0" y1="270.0" x2="570.0" y2="270.0" class="stroke-muted"/>
<line x1="70.0" y1="20.0" x2="70.0" y2="270.0" class="stroke-muted"/>
<line x1="70.0" y1="270.0" x2="70.0" y2="275.0" class="stroke-muted"/>
<text x="70.0" y="289.0" class="fig-small" text-anchor="middle">0</text>
<line x1="195.0" y1="270.0" x2="195.0" y2="275.0" class="stroke-muted"/>
<text x="195.0" y="289.0" class="fig-small" text-anchor="middle">50</text>
<line x1="320.0" y1="270.0" x2="320.0" y2="275.0" class="stroke-muted"/>
<text x="320.0" y="289.0" class="fig-small" text-anchor="middle">100</text>
<line x1="445.0" y1="270.0" x2="445.0" y2="275.0" class="stroke-muted"/>
<text x="445.0" y="289.0" class="fig-small" text-anchor="middle">150</text>
<line x1="570.0" y1="270.0" x2="570.0" y2="275.0" class="stroke-muted"/>
<text x="570.0" y="289.0" class="fig-small" text-anchor="middle">200</text>
<line x1="70.0" y1="270.0" x2="570.0" y2="270.0" class="stroke-grid"/>
<line x1="65.0" y1="270.0" x2="70.0" y2="270.0" class="stroke-muted"/>
<text x="61.0" y="274.0" class="fig-small" text-anchor="end">1e-2</text>
<line x1="70.0" y1="186.7" x2="570.0" y2="186.7" class="stroke-grid"/>
<line x1="65.0" y1="186.7" x2="70.0" y2="186.7" class="stroke-muted"/>
<text x="61.0" y="190.7" class="fig-small" text-anchor="end">1e-1</text>
<line x1="70.0" y1="103.3" x2="570.0" y2="103.3" class="stroke-grid"/>
<line x1="65.0" y1="103.3" x2="70.0" y2="103.3" class="stroke-muted"/>
<text x="61.0" y="107.3" class="fig-small" text-anchor="end">1</text>
<text x="320.0" y="308.0" class="fig-small" text-anchor="middle">step number</text>
<text x="70.0" y="10.0" class="fig-small" text-anchor="start">excess loss (log scale)</text>
<polyline points="70.0,50.3 72.5,66.6 75.0,74.2 77.5,63.4 80.0,73.5 82.5,78.6 85.0,76.7 87.5,79.8 90.0,73.0 92.5,69.0 95.0,79.5 97.5,82.6 100.0,80.4 102.5,82.5 105.0,84.0 107.5,81.3 110.0,78.9 112.5,83.5 115.0,76.2 117.5,75.7 120.0,90.8 122.5,90.4 125.0,91.4 127.5,91.9 130.0,92.4 132.5,88.9 135.0,92.9 137.5,92.6 140.0,93.8 142.5,91.2 145.0,93.5 147.5,93.5 150.0,99.0 152.5,74.9 155.0,99.8 157.5,101.0 160.0,99.2 162.5,100.6 165.0,101.3 167.5,101.9 170.0,100.9 172.5,106.0 175.0,106.4 177.5,106.6 180.0,105.3 182.5,106.9 185.0,105.8 187.5,99.2 190.0,102.7 192.5,103.9 195.0,103.6 197.5,107.4 200.0,108.6 202.5,108.9 205.0,101.9 207.5,113.1 210.0,107.7 212.5,114.1 215.0,112.8 217.5,88.2 220.0,101.0 222.5,106.4 225.0,111.5 227.5,119.2 230.0,111.7 232.5,122.1 235.0,124.0 237.5,102.6 240.0,122.3 242.5,120.7 245.0,125.3 247.5,128.7 250.0,127.6 252.5,117.2 255.0,119.5 257.5,120.8 260.0,126.1 262.5,116.2 265.0,135.6 267.5,132.5 270.0,136.6 272.5,128.6 275.0,125.1 277.5,136.5 280.0,139.2 282.5,135.9 285.0,137.1 287.5,142.8 290.0,143.2 292.5,140.2 295.0,137.9 297.5,142.4 300.0,139.5 302.5,146.2 305.0,106.8 307.5,148.0 310.0,149.4 312.5,151.1 315.0,141.9 317.5,147.1 320.0,142.6 322.5,142.6 325.0,152.1 327.5,152.2 330.0,151.2 332.5,151.7 335.0,136.4 337.5,156.0 340.0,155.8 342.5,156.7 345.0,153.0 347.5,102.7 350.0,141.3 352.5,140.1 355.0,150.2 357.5,142.7 360.0,160.0 362.5,133.4 365.0,153.3 367.5,151.9 370.0,159.6 372.5,158.6 375.0,151.1 377.5,141.9 380.0,163.4 382.5,162.9 385.0,162.6 387.5,150.8 390.0,165.2 392.5,167.6 395.0,157.1 397.5,161.0 400.0,162.5 402.5,146.5 405.0,140.4 407.5,169.1 410.0,139.9 412.5,103.3 415.0,124.1 417.5,141.4 420.0,169.6 422.5,165.4 425.0,167.6 427.5,157.0 430.0,172.3 432.5,135.0 435.0,115.0 437.5,145.6 440.0,175.3 442.5,163.8 445.0,153.5 447.5,174.5 450.0,172.2 452.5,103.2 455.0,161.1 457.5,174.0 460.0,179.8 462.5,175.8 465.0,180.7 467.5,155.8 470.0,179.4 472.5,179.5 475.0,169.4 477.5,184.7 480.0,171.3 482.5,184.5 485.0,178.3 487.5,174.4 490.0,188.4 492.5,184.8 495.0,168.2 497.5,168.4 500.0,189.5 502.5,187.6 505.0,176.7 507.5,188.5 510.0,125.9 512.5,176.4 515.0,172.9 517.5,149.8 520.0,194.0 522.5,135.7 525.0,188.1 527.5,173.2 530.0,142.4 532.5,169.2 535.0,161.9 537.5,169.9 540.0,144.8 542.5,140.3 545.0,156.4 547.5,201.2 550.0,172.6 552.5,145.0 555.0,197.4 557.5,189.9 560.0,163.6 562.5,168.2 565.0,191.9 567.5,154.4 570.0,139.3" fill="none" class="stroke-b"/>
<polyline points="70.0,50.3 72.5,54.7 75.0,62.3 77.5,69.4 80.0,75.6 82.5,78.0 85.0,74.1 87.5,69.0 90.0,63.0 92.5,69.8 95.0,78.6 97.5,82.5 100.0,75.6 102.5,66.4 105.0,64.3 107.5,65.7 110.0,69.3 112.5,82.6 115.0,93.3 117.5,88.7 120.0,80.4 122.5,79.6 125.0,87.7 127.5,100.0 130.0,104.0 132.5,91.5 135.0,80.9 137.5,74.6 140.0,82.2 142.5,98.0 145.0,116.3 147.5,115.2 150.0,103.6 152.5,107.7 155.0,115.9 157.5,126.9 160.0,131.6 162.5,129.3 165.0,122.5 167.5,125.0 170.0,130.7 172.5,139.6 175.0,146.8 177.5,145.5 180.0,133.8 182.5,134.0 185.0,135.7 187.5,137.9 190.0,140.7 192.5,148.0 195.0,161.4 197.5,165.6 200.0,166.0 202.5,163.9 205.0,158.8 207.5,160.8 210.0,159.6 212.5,167.6 215.0,178.3 217.5,180.5 220.0,187.6 222.5,183.7 225.0,177.3 227.5,194.9 230.0,201.0 232.5,193.2 235.0,183.1 237.5,204.1 240.0,213.9 242.5,215.2 245.0,208.8 247.5,212.9 250.0,220.7 252.5,226.1 255.0,242.5 257.5,246.4 260.0,241.7 262.5,250.6 265.0,263.7 267.5,261.9 270.0,243.3 272.5,202.2 275.0,180.9 277.5,198.4 280.0,270.0 282.5,208.9 285.0,194.7 287.5,187.8 290.0,203.3 292.5,214.5 295.0,219.1 297.5,267.2 300.0,270.0 302.5,218.2 305.0,181.2 307.5,166.8 310.0,163.4 312.5,171.1 315.0,222.1 317.5,270.0 320.0,233.2 322.5,208.5 325.0,196.1 327.5,250.2 330.0,270.0 332.5,221.4 335.0,230.1 337.5,257.7 340.0,270.0 342.5,270.0 345.0,215.1 347.5,270.0 350.0,239.5 352.5,177.2 355.0,157.2 357.5,157.7 360.0,169.5 362.5,239.5 365.0,263.9 367.5,187.8 370.0,188.1 372.5,222.8 375.0,270.0 377.5,270.0 380.0,258.6 382.5,241.9 385.0,269.7 387.5,270.0 390.0,197.0 392.5,170.2 395.0,194.3 397.5,234.4 400.0,270.0 402.5,227.7 405.0,200.8 407.5,218.3 410.0,214.2 412.5,270.0 415.0,237.7 417.5,186.7 420.0,169.7 422.5,175.5 425.0,193.0 427.5,222.9 430.0,270.0 432.5,270.0 435.0,222.7 437.5,174.0 440.0,158.7 442.5,168.2 445.0,192.2 447.5,270.0 450.0,212.9 452.5,243.8 455.0,270.0 457.5,270.0 460.0,233.6 462.5,195.8 465.0,183.7 467.5,224.5 470.0,270.0 472.5,240.2 475.0,211.1 477.5,218.3 480.0,260.6 482.5,270.0 485.0,228.9 487.5,230.1 490.0,270.0 492.5,270.0 495.0,248.4 497.5,205.5 500.0,196.3 502.5,192.3 505.0,188.6 507.5,211.8 510.0,250.3 512.5,176.0 515.0,137.5 517.5,119.6 520.0,122.8 522.5,132.2 525.0,160.0 527.5,200.3 530.0,255.1 532.5,157.3 535.0,141.4 537.5,145.2 540.0,234.0 542.5,233.8 545.0,173.6 547.5,154.4 550.0,144.2 552.5,132.6 555.0,142.6 557.5,169.7 560.0,270.0 562.5,193.9 565.0,149.8 567.5,134.4 570.0,163.0" fill="none" class="stroke-a"/>
<line x1="70.0" y1="128.4" x2="570.0" y2="128.4" class="stroke-muted dashed"/>
<circle cx="247.5" cy="128.4" r="5" class="fill-b"/>
<circle cx="160.0" cy="128.4" r="5" class="fill-a"/>
<text x="253.5" y="102.4" class="fig-small" text-anchor="start">SGD: step 71</text>
<line x1="160.0" y1="136.0" x2="160.0" y2="196.0" class="stroke-muted"/>
<text x="80.0" y="212.0" class="fig-small" text-anchor="start">momentum: step 36</text>
<line x1="110.0" y1="334.0" x2="140.0" y2="334.0" class="stroke-b"/>
<text x="146.0" y="338.0" class="fig-small" text-anchor="start">SGD, learning rate 0.0095</text>
<line x1="340.0" y1="334.0" x2="370.0" y2="334.0" class="stroke-a"/>
<text x="376.0" y="338.0" class="fig-small" text-anchor="start">momentum, 0.0025, beta 0.9</text>
</svg>
Figure 1. One run of each, with the same kind of batch. Momentum reaches the target excess loss of 0.5 (dashed line) at step 36 in this run, SGD at step 71. Averaged over 20 runs the counts are 37 and 77. Momentum's steps are ten times longer than its learning rate suggests, so it moves along the floor faster than any stable SGD could. The price is a larger jitter at the end.
```

## Give each weight its own ruler

Take a learning rate of 0.1. A bias with gradient 3 moves 0.3, which is sensible, but a weight with gradient 3000 moves 300. Lower the rate to suit the second weight and the first moves 0.0003 per step. The cure is to measure each gradient in units of its own typical size, so 3 and 3000 both read "about 1". Why not divide each gradient by its own absolute value? A weight whose gradient flips between +3 and −3 would move a full step each time, although it has no direction. A better ruler divides the *average* gradient by the *typical size* of the gradient: a steady slope gives a long step, and noise that flips sign gives a short one.

An **exponential moving average** keeps a running value and, at each step, blends in the newest number. With decay $\beta$:

$$
\text{avg} \leftarrow \beta\,\text{avg} + (1 - \beta)\,x
$$

Read it aloud: keep a fraction $\beta$ of the old average and take $1 - \beta$ of the fresh number. The number $k$ steps ago has weight $(1 - \beta)\beta^k$, so the average remembers roughly the last $1/(1 - \beta)$ numbers, which is its **window**.

Adam keeps two of these for every weight. The first, $m$, averages the gradient $g$, so it is the typical direction, with $\beta_1 = 0.9$ and a window of 10 steps. The second, $v$, averages $g^2$ entry by entry, so $\sqrt{v}$ is the typical size, a root-mean-square. Squaring stops +3 and −3 from cancelling. It uses $\beta_2 = 0.999$, a window of 1,000 steps, because a size needs a longer memory than a direction.

Both averages start at 0, and 0 is a bad guess. With a steady gradient of 3, one step gives $m = 0.3$, because the unseen numbers count as zeros. For a constant gradient $g$ the raw average is exactly $(1 - \beta_1^{\,t})\,g$. **Bias correction** divides that fraction out. Count $t$ from 1, the number of updates so far including this one:

$$
\hat{m} = \frac{m}{1 - \beta_1^{\,t}}, \qquad \hat{v} = \frac{v}{1 - \beta_2^{\,t}}
$$

Read it aloud: divide each average by the share of the window that has filled so far. For the gradient 3 at $t = 1$, $\hat{m} = 0.3 / 0.1 = 3$ and $\hat{v} = 0.009 / 0.001 = 9$. Later the divisor drifts toward 1 and the correction fades.

> [!WARNING]
> Start $t$ at 1, not 0. At $t = 0$ the divisor $1 - \beta_1^{\,0}$ is zero, and the first step divides by zero. Also store the raw $m$ and $v$, never the corrected copies, or the next step corrects an average that is already corrected.

Divide the corrected direction by the corrected size. A small $\epsilon$, such as $10^{-8}$, keeps the denominator off zero for a weight whose gradient has always been zero:

$$
w \leftarrow w - \text{lr}\cdot\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

Read it aloud: the learning rate times the average direction measured in typical sizes. At $t = 1$ with $\text{lr} = 0.1$, the gradient 3 gives $\hat{m} = 3$ and $\sqrt{\hat{v}} = 3$, so the step is $0.1 \times 3/3 = 0.1$. The gradient 3000 gives $\hat{m} = 3000$ and $\sqrt{\hat{v}} = 3000$, so the step is again 0.1, where plain SGD moved 0.3 and 300. At $t = 1$ the ratio is exactly $g/|g| = \pm 1$, so Adam's first step is one full learning rate in every coordinate. Figure 2 shows the same effect in the valley, and the step in numpy is:

```python
new_w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
```

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Length of the first step on a flat and a steep coordinate for SGD and for Adam">
<defs></defs>
<line x1="80.0" y1="250.0" x2="570.0" y2="250.0" class="stroke-muted"/>
<line x1="80.0" y1="40.0" x2="80.0" y2="250.0" class="stroke-muted"/>
<line x1="80.0" y1="250.0" x2="570.0" y2="250.0" class="stroke-grid"/>
<line x1="75.0" y1="250.0" x2="80.0" y2="250.0" class="stroke-muted"/>
<text x="71.0" y="254.0" class="fig-small" text-anchor="end">1e-3</text>
<line x1="80.0" y1="186.4" x2="570.0" y2="186.4" class="stroke-grid"/>
<line x1="75.0" y1="186.4" x2="80.0" y2="186.4" class="stroke-muted"/>
<text x="71.0" y="190.4" class="fig-small" text-anchor="end">1e-2</text>
<line x1="80.0" y1="122.7" x2="570.0" y2="122.7" class="stroke-grid"/>
<line x1="75.0" y1="122.7" x2="80.0" y2="122.7" class="stroke-muted"/>
<text x="71.0" y="126.7" class="fig-small" text-anchor="end">1e-1</text>
<line x1="80.0" y1="59.1" x2="570.0" y2="59.1" class="stroke-grid"/>
<line x1="75.0" y1="59.1" x2="80.0" y2="59.1" class="stroke-muted"/>
<text x="71.0" y="63.1" class="fig-small" text-anchor="end">1</text>
<text x="80.0" y="30.0" class="fig-small" text-anchor="start">length of the first step (log scale)</text>
<rect x="111.2" y="187.8" width="60.0" height="62.2" class="fill-b"/>
<text x="141.2" y="181.8" class="fig-small" text-anchor="middle">0.0095</text>
<rect x="233.8" y="60.5" width="60.0" height="189.5" class="fill-b"/>
<text x="263.8" y="54.5" class="fig-small" text-anchor="middle">0.95</text>
<rect x="356.2" y="141.9" width="60.0" height="108.1" class="fill-a"/>
<text x="386.2" y="135.9" class="fig-small" text-anchor="middle">0.05</text>
<rect x="478.8" y="141.9" width="60.0" height="108.1" class="fill-a"/>
<text x="508.8" y="135.9" class="fig-small" text-anchor="middle">0.05</text>
<text x="141.2" y="268.0" class="fig-small" text-anchor="middle">flat axis</text>
<text x="263.8" y="268.0" class="fig-small" text-anchor="middle">steep axis</text>
<text x="386.2" y="268.0" class="fig-small" text-anchor="middle">flat axis</text>
<text x="508.8" y="268.0" class="fig-small" text-anchor="middle">steep axis</text>
<text x="202.5" y="288.0" class="fig-bold" text-anchor="middle">SGD, learning rate 0.0095</text>
<text x="447.5" y="288.0" class="fig-bold" text-anchor="middle">Adam, learning rate 0.05</text>
</svg>
Figure 2. At the point (1, 1) of the valley the gradients are 1 on the flat axis and 100 on the steep axis. SGD's first steps differ by a factor of 100, since a step is learning rate times gradient. Adam divides each average gradient by its own typical size, so both steps are the learning rate, 0.05. Bars start at 1e-3.
```

On a valley with curvatures 1 and 10,000, settling below $10^{-4}$ takes SGD 21,837 steps, momentum 2,630, and Adam 167.

> [!KEY]
> Adam keeps an average gradient and an average squared gradient for every weight, corrects both for starting at zero, and divides one by the other. Each weight then moves about one learning rate per step, whatever its gradient size.

## Decay the weights after Adam

**Weight decay** takes a small fraction of every weight away at each step. With learning rate $\eta$ and decay rate $\lambda$, $w \leftarrow w - \eta\,\lambda\,w$. With $w = 1$, $\eta = 0.1$, and $\lambda = 0.01$ the weight loses 0.001, and every weight loses the same fraction. That rule is what an L2 penalty $\tfrac{\lambda}{2}\lVert w\rVert^2$ produces in plain SGD, because the penalty's gradient is $\lambda w$. With $g = 0.5$, the penalised gradient 0.51 gives $1 - 0.051 = 0.949$, and the plain step followed by the decay gives $0.95 - 0.001 = 0.949$ too.

Adam does something to the gradient: it divides by $\sqrt{\hat{v}}$. Put the penalty into the gradient and the sum $g + \lambda w$ goes through that division. Take a quiet weight with typical gradient size 0.01 and a noisy weight with typical size 10, both equal to 1. The pull $\lambda w = 0.01$ is as big as the quiet weight's whole gradient, so Adam takes a long step, but it is a thousandth of the noisy weight's gradient, so Adam sees almost nothing. Freeze both weights at 1, feed Adam 200 noisy gradients of those sizes, and compare the next step with and without decay. Decay should shrink each weight by 0.001. With the penalty inside Adam, the quiet weight shrinks by 0.074, about 74 times too much, and the noisy weight by 0.00010, about ten times too little.

The fix keeps the decay out of the pipeline. Run Adam on the raw gradient, so the averages see only the loss, and then subtract the decay:

$$
w_{\text{AdamW}} = w_{\text{Adam}} - \eta\,\lambda\,w
$$

Read it aloud: take Adam's result and remove the learning rate times the decay rate times the weight you started the step with. This is **AdamW**, and with $\lambda = 0$ it is Adam. In numpy:

```python
new_w, new_state = adam_step(w, grad, state, lr, beta1, beta2, eps)
new_w = new_w - lr * weight_decay * w
```

In the frozen test, both weights shrink by exactly 0.001 under AdamW, because the averages never see the decay.

> [!KEY]
> Adam divides everything in the gradient by the weight's typical gradient size, including an L2 pull. AdamW subtracts $\eta\lambda w$ after the Adam step instead, so every weight shrinks by the same fraction.

## Change the rate over time

At step 1, Adam has seen one batch of eight rows, yet it moves every weight by a full learning rate. Early steps should be short until the averages fill. At the end the problem flips: batch noise keeps the loss at a floor that grows with the rate, and only shorter steps settle it.

So the rate should rise from near zero, hold a good value in the middle, and fall back to zero. A **learning-rate schedule** is a rule that gives the rate for each step number. Figure 3 shows the one this chapter builds.

**Linear warmup** raises the rate from 0 to a chosen peak over the first `warmup` steps, in a straight line:

$$
\text{lr}(s) = \text{peak}\cdot\frac{s}{\text{warmup}} \qquad \text{for } s < \text{warmup}
$$

Read it aloud: halfway through warmup, use half the peak. With 100 warmup steps and a peak of 0.01, step 50 gives $0.01 \times 50/100 = 0.005$ and step 100 reaches 0.01.

A warmup of tens to hundreds of steps lets Adam's averages settle before the rate gets large.

After warmup, let $p$ be how far you are through the decay phase, from 0 at the end of warmup to 1 at the last step:

$$
p = \frac{s - \text{warmup}}{\text{total} - \text{warmup}}, \qquad
\text{lr}(s) = \text{peak}\cdot\tfrac{1}{2}\big(1 + \cos(\pi p)\big)
$$

Read it aloud: the factor $\tfrac12(1 + \cos \pi p)$ is half of a cosine wave, lifted so that it runs from 1 at $p = 0$ through $\tfrac12$ at $p = \tfrac12$ down to 0 at $p = 1$. With 1,000 total steps these rates hold:

| Step | Phase | Rate |
|---|---|---|
| 0 | warmup | 0 |
| 50 | warmup | 0.005 |
| 100 | decay, $p = 0$ | 0.01 |
| 325 | decay, $p = 0.25$, factor $\tfrac12(1 + 0.7071)$ | 0.0085355 |
| 550 | decay, $p = 0.5$ | 0.005 |
| 1000 | decay, $p = 1$ | 0 |

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
Figure 3. The exact schedule from the worked example, drawn from computed values. The rate climbs in a straight line from 0 to 0.01 over the first 100 steps, then follows half a cosine down to 0 at step 1000. The orange dots are the steps in the table: 0, 50, 100, 325, 550, and 1000.
```

Clamp $p$ to $[0, 1]$ so that later steps stay at 0.

Why cosine and not a straight line down? Cosine stays near the peak early, when a high rate pays off, falls fastest in the middle, and flattens near 0, so the last steps are tiny and the weights settle. At $p = 0.9$ cosine has fallen to 0.024 of the peak, while a line still holds 0.1. Many shapes work, and cosine has no extra setting to tune.

> [!TIP]
> Check four points before you trust a schedule: step 0 gives 0, the end of warmup gives the peak, the middle of the decay gives half the peak, and the last step gives 0.

If `warmup >= total`, there is no decay phase, so rise through warmup and hold the peak instead of dividing by `total - warmup`.

Run Adam with peak rate 0.1 on batches of eight rows from the same valley, once with a constant rate and once with 20 warmup steps and cosine decay over 400 steps. Figure 4 shows both runs. Over the last 50 steps the constant rate hovers at an excess loss of 0.147, averaged over 10 runs, and the schedule reaches 0.003, about fifty times lower. In the right panel the walk starts only 0.014 from the bottom along the steep axis, yet Adam's first step lurches by 0.1 and the loss jumps from 0.01 to 0.46. With warmup the loss stays below 0.07 for the first 10 steps.

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
Figure 4. Adam with peak rate 0.1 on mini-batches of eight rows from the same valley, with each curve the average excess loss of 10 runs. Left: the constant rate (orange) hovers on a plateau and never settles, while the schedule (blue) drops below it once the decay lowers the rate. Over the last 50 steps the constant rate averages 0.147 and the schedule 0.0030. Right: from a start 0.014 from the bottom, the first full-size step kicks the loss up to 0.46 on average (orange), while warmup keeps it below 0.07 over the first 10 steps (blue). Once warmup ends at step 20 the rate is 0.1, and the loss swings again until the decay lowers the rate.
```

> [!KEY]
> Mini-batch gradients are cheap but noisy, and momentum adds up the agreeing part. Adam gives every weight about one learning rate per step, AdamW decays every weight by the same fraction, and a warmup-plus-cosine schedule starts gently and ends settled.

> [!RECAP]
> Hold these points before you write code:
>
> - A mini-batch gradient divides by `len(idx)`, and momentum keeps `v = beta * v + grad`, with no $(1 - \beta)$ on the gradient.
> - `ema_update(prev, x, beta)` is `beta * prev + (1 - beta) * x`, and `bias_correct(avg, beta, t)` divides by `1 - beta**t`, with `t` counted from 1.
> - Adam's step is `lr * m_hat / (sqrt(v_hat) + eps)`, and the state holds the raw `m`, the raw `v`, and `t`. AdamW runs Adam on the raw gradient, then subtracts `lr * weight_decay * w` using the incoming `w`.
> - The schedule is `peak * step / warmup` in warmup, then `peak * 0.5 * (1 + cos(pi * p))` with `p` clamped to `[0, 1]`.

## Build it step by step

You write ten functions, in the order the checks test them:

1. **`batch_gradient(X, y, w, idx)`.** It takes `X` of shape `(n, d)`, `y` of shape `(n,)`, `w` of shape `(d,)`, and an integer array `idx`. It returns the `(d,)` gradient of half the mean squared miss over those rows, from `X[idx]`, `y[idx]`, and a transposed matrix product. The usual mistake is dividing by `n`.
2. **`sgd_step(w, grad, state, lr)`.** It returns `(new_w, new_state)`. Subtract `lr * grad` and hand back `state` unchanged. The usual mistake is returning only `new_w`.
3. **`momentum_step(w, grad, state, lr, beta)`.** A missing `"v"` in `state` means zeros. Set `v = beta * v + grad`, subtract `lr * v`, and return `{"v": v}`. The usual mistake is multiplying `grad` by `(1 - beta)`.
4. **`ema_update(prev, x, beta)`** returns `beta * prev + (1 - beta) * x`. The usual mistake is swapping the two weights.
5. **`bias_correct(avg, beta, t)`** returns `avg / (1 - beta**t)` and raises `ValueError` when `t < 1`. The usual mistake is counting $t$ from 0.
6. **`adam_step(w, grad, state, lr, beta1, beta2, eps)`** reads `m`, `v`, and `t` from `state`, with zeros and 0 as defaults, and adds 1 to `t`. Use `ema_update` for $m$ and for $v$ with `grad**2`, correct both, and return `(new_w, {"m": m, "v": v, "t": t})`. The usual mistake is storing the corrected averages.
7. **`adamw_step(..., weight_decay)`** calls `adam_step` on the raw `grad`, then subtracts `lr * weight_decay * w` with the incoming `w`. The usual mistake is adding `weight_decay * w` to `grad`, which is the coupled version.
8. **`linear_warmup(step, warmup, peak)`.** Return `peak * step / warmup` for `step < warmup`, and `peak` otherwise, including when `warmup` is 0. Raise `ValueError` for a negative `step`. The usual mistake is dividing by a `warmup` of 0.
9. **`cosine_decay_fraction(p)`.** Clamp `p` to `[0, 1]` and return `0.5 * (1 + np.cos(np.pi * p))`. The usual mistake is skipping the clamp, so late steps bounce back up.
10. **`cosine_schedule(step, warmup, total, peak)`.** Use `linear_warmup` while `step < warmup`. Otherwise compute `p` and return `peak * cosine_decay_fraction(p)`. When `warmup >= total`, return `linear_warmup(step, warmup, peak)`. The usual mistake is measuring `p` from step 0 instead of from the end of warmup.

## Where this breaks

Each of these has a direct remedy:

- Batches that are not random, such as data sorted by class, give biased gradients. Shuffle the rows each epoch.
- A learning rate over the stability limit sends the loss to infinity. Cut it by three to ten, and lower it when you raise $\beta$.
- Some libraries name the coupled L2 penalty `weight_decay` on a plain Adam, so read the documentation before trusting the name.

## The exercise

You implement the ten functions above. The starter supplies `make_valley`, which builds the 200-row valley with curvatures 1 and 100, and `run_with_schedule`.

The checks compare your batch gradient with finite differences, then test SGD's stability limit and momentum's tenfold steady-state step. They verify the moving average, Adam's first step for gradients 3 and 3000, and that AdamW shrinks a quiet and a noisy weight equally. They check the schedule's table, race momentum against SGD, and run Adam with and without the schedule.
