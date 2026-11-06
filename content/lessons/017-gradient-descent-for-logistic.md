---
title: "Learning the weights: gradient and descent"
slug: 017-gradient-descent-for-logistic
part: "Part 1 — Foundations"
summary: "The log loss has slope X^T(p − y): each weight is blamed by its feature times the error, and descent follows that slope."
minutes: 40
gpu: false
objectives:
  - Derive the gradient X^T(p − y) by blaming each weight for the errors on the rows where its feature was on.
  - Take one gradient step by hand, and fit the model with a loop that stops when the slope is flat.
  - Explain how too small and too large a learning rate each fail, with numbers.
  - Explain why perfectly separable data makes the weights grow without limit, and how a ridge fee stops it.
prereqs:
  - 008-walking-downhill
  - 009-step-size-and-long-valleys
  - 011-ridge-penalty-and-closed-form
  - 016-log-loss
exercise: 017-gradient-descent-for-logistic
---

# Learning the weights: gradient and descent

The last chapter gave you a number that scores any weights. For your two emails, weights $(0.5, -1)$ score 0.6755. A lower number is better, but which way should each weight move to lower it, and by how much? And what happens when the data can be split perfectly: does the model ever say "good enough"?

Without a direction you could only guess at weights and keep the best guess, which is hopeless once there are hundreds of weights. The slope of the loss gives the direction for every weight at once. It also gives a surprise: on a perfectly separable inbox the slope never reaches zero, so a naive fit never stops.

You will compute the slope by hand, take one step, build the loop that repeats it, and then see what perfect data does to it.

## What you already know

- The log loss is $L(w) = \sum_i [\operatorname{softplus}(z_i) - y_i z_i]$ with $z = Xw$ (see How wrong is a probability? The log loss).
- The gradient lists the slope of the loss for every weight, and descent steps against it (see Walking downhill: slope, gradient, and one step).
- A step of size `lr` against the gradient is safe only below a limit set by the curvature (see How big a step? Step size and long valleys).
- For linear regression with the loss $\tfrac12\sum_i(\text{prediction}_i - y_i)^2$, the gradient is $X^\top(\text{prediction} - y)$.
- The ridge fee $\tfrac{\lambda}{2}\lVert w\rVert^2$ penalises large weights (see Ridge: a fee on large weights).

## Blame each weight

Start from one row. Its loss $\operatorname{softplus}(z) - yz$ depends on the weights only through the score $z$. So you need two slopes and the chain rule: how the loss moves when $z$ moves, and how $z$ moves when a weight moves.

The first is easy. The slope of $\ln(1 + e^{z})$ is $e^{z}/(1+e^{z})$, which is $\sigma(z) = p$. The slope of $-yz$ is $-y$. So the loss of one row changes by $p - y$ per unit of score. That is the *error*: the probability minus the truth. The second slope is the feature itself, since $z = \sum_j w_j x_j$ moves by $x_j$ when $w_j$ moves by 1.

Multiply them and add over rows: the slope for weight $j$ is the sum over emails of (error) times (feature $j$). Weight $j$ is blamed for each row in proportion to how wrong that row was and how much its own feature was switched on. A feature that was 0 is not to blame. This is the same shape as linear regression's $X^\top(\text{prediction} - y)$. The sigmoid's slope cancels against the log, which leaves the plain error.

For the two emails the errors are $p - y = (0.6225 - 1,\ 0.1824 - 0) = (-0.3775,\ +0.1824)$. Figure 1 adds the products down each column.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Gradient as blame: each email's error times its feature value, summed down each column, gives the gradient for the bias and the replies weight">
<defs><marker id="c019-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="20.0" y="28.0" class="fig-bold" text-anchor="start">Email</text>
<text x="130.0" y="28.0" class="fig-bold" text-anchor="start">error p − y</text>
<text x="215.0" y="28.0" class="fig-bold" text-anchor="start">bias feature × error</text>
<text x="390.0" y="28.0" class="fig-bold" text-anchor="start">replies feature × error</text>
<line x1="20.0" y1="38.0" x2="580.0" y2="38.0" class="stroke-grid" />
<text x="20.0" y="70.0" text-anchor="start">A (spam)</text>
<text x="130.0" y="70.0" class="fig-mono" text-anchor="start">-0.3775</text>
<text x="215.0" y="70.0" class="fig-small fig-mono" text-anchor="start">1 × (-0.3775) = -0.3775</text>
<text x="390.0" y="70.0" class="fig-small fig-mono" text-anchor="start">0 × (-0.3775) = 0</text>
<text x="20.0" y="114.0" text-anchor="start">B (not spam)</text>
<text x="130.0" y="114.0" class="fig-mono" text-anchor="start">+0.1824</text>
<text x="215.0" y="114.0" class="fig-small fig-mono" text-anchor="start">1 × (+0.1824) = +0.1824</text>
<text x="390.0" y="114.0" class="fig-small fig-mono" text-anchor="start">2 × (+0.1824) = +0.3649</text>
<line x1="20.0" y1="132.0" x2="580.0" y2="132.0" class="stroke-ink" />
<text x="20.0" y="158.0" class="fig-bold" text-anchor="start">Sum = gradient</text>
<text x="215.0" y="158.0" class="fig-bold fig-mono" text-anchor="start">-0.1951</text>
<text x="390.0" y="158.0" class="fig-bold fig-mono" text-anchor="start">+0.3649</text>
<line x1="228.0" y1="240.0" x2="228.0" y2="204.0" class="stroke-ink" marker-end="url(#c019-arrow)"/>
<text x="242.0" y="220.0" class="fig-small" text-anchor="start">bias goes up:</text>
<text x="242.0" y="236.0" class="fig-small" text-anchor="start">0.5 → 0.695</text>
<line x1="380.0" y1="204.0" x2="380.0" y2="240.0" class="stroke-ink" marker-end="url(#c019-arrow)"/>
<text x="394.0" y="220.0" class="fig-small" text-anchor="start">replies weight goes down:</text>
<text x="394.0" y="236.0" class="fig-small" text-anchor="start">−1 → −1.365</text>
<text x="20.0" y="220.0" class="fig-small" text-anchor="start">A negative gradient means raising</text>
<text x="20.0" y="236.0" class="fig-small" text-anchor="start">the weight lowers the loss.</text>
</svg>
Figure 1. The gradient entry for a weight sums error × feature down its column. Notice that the replies weight is blamed only for email B, the only row where that feature is not 0.
```

As a formula, with the ridge fee's slope $\lambda w$ added on:

$$
\nabla L(w) = X^{\top}\bigl(\sigma(Xw) - y\bigr) + \lambda w
$$

Read it aloud: compute the probability for each row, subtract the label to get each row's error, then let each weight collect error times feature through $X^\top$. For the emails it returns $[-0.1951,\ 0.3649]$. In numpy:

```python
grad = X.T @ (sigmoid(X @ w) - y) + ridge * w
```

## Take one step

A negative entry means the loss falls when the weight rises, so descent raises the bias and lowers the replies weight. That is sensible. Both emails share the bias, and A's error is bigger than B's, so the bias rises. Only B has replies, and B was over-predicted, so the replies weight falls. With a step size `lr = 1`:

$$
w \leftarrow w - \text{lr}\cdot\nabla L(w) = (0.5,\,-1) - (-0.1951,\ 0.3649) = (0.6951,\ -1.3649)
$$

The probabilities become 0.667 for A (up from 0.622) and 0.116 for B (down from 0.182), and the loss falls from 0.6755 to 0.528. One step, in the right direction, for every weight at once.

## Choose the step size

A step of 1 worked here, but a step size is a trade. To see it, use a larger inbox of 60 emails with the same two weights. Zero weights give the loss $60 \ln 2 = 41.59$, and the best weights give 28.80.

- **Too small.** With `lr = 0.001`, the loss reaches 29.29 after 80 steps, then crawls, needing 576 steps to get within 0.01 of the minimum.
- **About right.** With `lr = 0.02`, it is within 0.01 of the minimum after 27 steps.
- **Too large.** With `lr = 0.05`, the first step makes the loss jump from 41.6 up to 61.7, and the loss keeps bouncing without settling.

The limit from the step-size chapter explains the last case. Near the bottom, the largest curvature of this loss is about 42, so steps above $2/42 = 0.048$ overshoot the bottom by more than they started from.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Loss against step number for three learning rates on a 60-email inbox: 0.001 crawls, 0.02 settles quickly, 0.05 overshoots and never settles">
<line x1="70.0" y1="244.7" x2="570.0" y2="244.7" class="stroke-grid" />
<text x="62.0" y="248.7" class="fig-small" text-anchor="end">30</text>
<line x1="70.0" y1="181.6" x2="570.0" y2="181.6" class="stroke-grid" />
<text x="62.0" y="185.6" class="fig-small" text-anchor="end">40</text>
<line x1="70.0" y1="118.4" x2="570.0" y2="118.4" class="stroke-grid" />
<text x="62.0" y="122.4" class="fig-small" text-anchor="end">50</text>
<line x1="70.0" y1="55.3" x2="570.0" y2="55.3" class="stroke-grid" />
<text x="62.0" y="59.3" class="fig-small" text-anchor="end">60</text>
<line x1="70.0" y1="252.3" x2="570.0" y2="252.3" class="stroke-muted dashed" />
<text x="570.0" y="268.3" class="fig-small" text-anchor="end">minimum 28.80</text>
<line x1="70.0" y1="270.0" x2="570.0" y2="270.0" class="stroke-muted" />
<line x1="70.0" y1="270.0" x2="70.0" y2="275.0" class="stroke-muted" />
<text x="70.0" y="290.0" class="fig-small" text-anchor="middle">0</text>
<line x1="195.0" y1="270.0" x2="195.0" y2="275.0" class="stroke-muted" />
<text x="195.0" y="290.0" class="fig-small" text-anchor="middle">20</text>
<line x1="320.0" y1="270.0" x2="320.0" y2="275.0" class="stroke-muted" />
<text x="320.0" y="290.0" class="fig-small" text-anchor="middle">40</text>
<line x1="445.0" y1="270.0" x2="445.0" y2="275.0" class="stroke-muted" />
<text x="445.0" y="290.0" class="fig-small" text-anchor="middle">60</text>
<line x1="570.0" y1="270.0" x2="570.0" y2="275.0" class="stroke-muted" />
<text x="570.0" y="290.0" class="fig-small" text-anchor="middle">80</text>
<text x="320.0" y="310.0" text-anchor="middle">step</text>
<text x="62.0" y="22.0" text-anchor="end">loss</text>
<polyline points="70.0,171.5 76.2,187.6 82.5,199.7 88.8,209.0 95.0,216.1 101.2,221.6 107.5,226.0 113.8,229.5 120.0,232.3 126.2,234.6 132.5,236.5 138.8,238.0 145.0,239.3 151.2,240.4 157.5,241.4 163.8,242.2 170.0,242.9 176.2,243.5 182.5,244.0 188.8,244.4 195.0,244.8 201.2,245.1 207.5,245.4 213.8,245.7 220.0,245.9 226.2,246.2 232.5,246.4 238.8,246.5 245.0,246.7 251.2,246.8 257.5,247.0 263.8,247.1 270.0,247.2 276.2,247.3 282.5,247.4 288.8,247.5 295.0,247.5 301.2,247.6 307.5,247.7 313.8,247.7 320.0,247.8 326.2,247.9 332.5,247.9 338.8,248.0 345.0,248.0 351.2,248.1 357.5,248.1 363.8,248.2 370.0,248.2 376.2,248.3 382.5,248.3 388.8,248.3 395.0,248.4 401.2,248.4 407.5,248.4 413.8,248.5 420.0,248.5 426.2,248.5 432.5,248.6 438.8,248.6 445.0,248.6 451.2,248.7 457.5,248.7 463.8,248.7 470.0,248.8 476.2,248.8 482.5,248.8 488.8,248.9 495.0,248.9 501.2,248.9 507.5,248.9 513.8,249.0 520.0,249.0 526.2,249.0 532.5,249.1 538.8,249.1 545.0,249.1 551.2,249.1 557.5,249.2 563.8,249.2 570.0,249.2" class="stroke-a" fill="none"/>
<polyline points="70.0,171.5 76.2,221.8 82.5,242.6 88.8,249.0 95.0,249.9 101.2,250.3 107.5,250.6 113.8,250.8 120.0,251.1 126.2,251.2 132.5,251.4 138.8,251.5 145.0,251.7 151.2,251.7 157.5,251.8 163.8,251.9 170.0,252.0 176.2,252.0 182.5,252.0 188.8,252.1 195.0,252.1 201.2,252.1 207.5,252.2 213.8,252.2 220.0,252.2 226.2,252.2 232.5,252.2 238.8,252.2 245.0,252.2 251.2,252.2 257.5,252.3 263.8,252.3 270.0,252.3 276.2,252.3 282.5,252.3 288.8,252.3 295.0,252.3 301.2,252.3 307.5,252.3 313.8,252.3 320.0,252.3 326.2,252.3 332.5,252.3 338.8,252.3 345.0,252.3 351.2,252.3 357.5,252.3 363.8,252.3 370.0,252.3 376.2,252.3 382.5,252.3 388.8,252.3 395.0,252.3 401.2,252.3 407.5,252.3 413.8,252.3 420.0,252.3 426.2,252.3 432.5,252.3 438.8,252.3 445.0,252.3 451.2,252.3 457.5,252.3 463.8,252.3 470.0,252.3 476.2,252.3 482.5,252.3 488.8,252.3 495.0,252.3 501.2,252.3 507.5,252.3 513.8,252.3 520.0,252.3 526.2,252.3 532.5,252.3 538.8,252.3 545.0,252.3 551.2,252.3 557.5,252.3 563.8,252.3 570.0,252.3" class="stroke-c" fill="none"/>
<polyline points="70.0,171.5 76.2,44.4 82.5,179.0 88.8,249.8 95.0,250.5 101.2,249.6 107.5,250.4 113.8,249.4 120.0,250.3 126.2,249.2 132.5,250.3 138.8,249.1 145.0,250.2 151.2,249.1 157.5,250.2 163.8,249.0 170.0,250.2 176.2,249.0 182.5,250.2 188.8,249.0 195.0,250.2 201.2,249.0 207.5,250.2 213.8,249.0 220.0,250.2 226.2,249.0 232.5,250.2 238.8,249.0 245.0,250.2 251.2,249.0 257.5,250.2 263.8,249.0 270.0,250.2 276.2,249.0 282.5,250.2 288.8,249.0 295.0,250.2 301.2,249.0 307.5,250.2 313.8,249.0 320.0,250.2 326.2,249.0 332.5,250.2 338.8,249.0 345.0,250.2 351.2,249.0 357.5,250.2 363.8,249.0 370.0,250.2 376.2,249.0 382.5,250.2 388.8,249.0 395.0,250.2 401.2,249.0 407.5,250.2 413.8,249.0 420.0,250.2 426.2,249.0 432.5,250.2 438.8,249.0 445.0,250.2 451.2,249.0 457.5,250.2 463.8,249.0 470.0,250.2 476.2,249.0 482.5,250.2 488.8,249.0 495.0,250.2 501.2,249.0 507.5,250.2 513.8,249.0 520.0,250.2 526.2,249.0 532.5,250.2 538.8,249.0 545.0,250.2 551.2,249.0 557.5,250.2 563.8,249.0 570.0,250.2" class="stroke-bad" fill="none"/>
<line x1="220.0" y1="46.0" x2="245.0" y2="46.0" class="stroke-a" />
<text x="254.4" y="50.0" class="fig-small" text-anchor="start">lr 0.001: slow, still above the minimum at step 80</text>
<line x1="220.0" y1="66.0" x2="245.0" y2="66.0" class="stroke-c" />
<text x="254.4" y="70.0" class="fig-small" text-anchor="start">lr 0.02: within 0.01 of the minimum by step 27</text>
<line x1="220.0" y1="86.0" x2="245.0" y2="86.0" class="stroke-bad" />
<text x="254.4" y="90.0" class="fig-small" text-anchor="start">lr 0.05: overshoots and never settles</text>
</svg>
Figure 2. The same loss and the same starting weights, with three step sizes. Notice that the green curve falls fast and flattens at the dashed minimum, the blue curve is still sliding down slowly at step 80, and the red curve overshoots to 61.7 on its first step and keeps wobbling above the minimum.
```

## Stop when the slope is flat

A loop needs a stopping rule. At the bottom of the loss the gradient is 0, so stop when its length falls below a small `tol`. The loop starts at zero weights, records the loss before each step so that the first entry is the loss at zero weights, and stops before stepping if the gradient is already small:

```python
for _ in range(max_iter):
    history.append(log_loss(w, X, y, ridge))
    g = gradient(w, X, y, ridge)
    if np.linalg.norm(g) < tol:
        break
    w = w - lr * g
```

On the 60-email inbox, `lr = 0.02` stops after 185 passes through the loop. The accuracy of the fitted model is 0.767. That beats always answering "not spam", which scores 0.717, but only by five points, which is why you compare accuracy against that baseline.

## Let a perfect split grow the weights

Your two emails are already perfectly separable: weights $(1, -1)$ give A a score of 1 and B a score of $-1$, so both are right. Along the line $w = t\,(1, -1)$, the score of A is $t$ and the score of B is $-t$, and the total loss is $2\ln(1 + e^{-t})$.

Larger $t$ is always better. At $t = 1$ the loss is 0.627, at 5 it is 0.0134, and at 10 it is 0.00009. Figure 3 shows why: a larger $t$ does not move the boundary, which stays at one reply, but it makes the probabilities more extreme. Each doubling makes every row more confident and every loss term smaller. The loss approaches 0 but never reaches it, since the sigmoid never returns exactly 1, so the gradient never reaches 0 either.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Probability of spam against number of replies for weights t times (1, -1) with t = 1, 5, 10: the curve steepens through the same boundary at 1 reply while email A at 0 replies and email B at 2 replies are both classified correctly">
<line x1="70.0" y1="240.0" x2="570.0" y2="240.0" class="stroke-grid" />
<text x="62.0" y="244.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="140.0" x2="570.0" y2="140.0" class="stroke-grid" />
<text x="62.0" y="144.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="40.0" x2="570.0" y2="40.0" class="stroke-grid" />
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">1</text>
<line x1="320.0" y1="240.0" x2="320.0" y2="40.0" class="stroke-grid dashed" />
<polyline points="70.0,76.5 72.5,76.9 75.0,77.4 77.5,77.9 80.1,78.3 82.6,78.8 85.1,79.3 87.6,79.7 90.1,80.2 92.6,80.7 95.1,81.2 97.6,81.7 100.2,82.2 102.7,82.7 105.2,83.2 107.7,83.7 110.2,84.2 112.7,84.8 115.2,85.3 117.7,85.8 120.3,86.3 122.8,86.9 125.3,87.4 127.8,88.0 130.3,88.5 132.8,89.1 135.3,89.6 137.8,90.2 140.4,90.8 142.9,91.4 145.4,91.9 147.9,92.5 150.4,93.1 152.9,93.7 155.4,94.3 157.9,94.9 160.5,95.5 163.0,96.1 165.5,96.7 168.0,97.3 170.5,97.9 173.0,98.6 175.5,99.2 178.0,99.8 180.6,100.4 183.1,101.1 185.6,101.7 188.1,102.4 190.6,103.0 193.1,103.7 195.6,104.3 198.1,105.0 200.7,105.7 203.2,106.3 205.7,107.0 208.2,107.7 210.7,108.3 213.2,109.0 215.7,109.7 218.2,110.4 220.8,111.1 223.3,111.8 225.8,112.5 228.3,113.2 230.8,113.9 233.3,114.6 235.8,115.3 238.3,116.0 240.9,116.7 243.4,117.4 245.9,118.1 248.4,118.8 250.9,119.6 253.4,120.3 255.9,121.0 258.4,121.7 261.0,122.5 263.5,123.2 266.0,123.9 268.5,124.7 271.0,125.4 273.5,126.1 276.0,126.9 278.5,127.6 281.1,128.4 283.6,129.1 286.1,129.9 288.6,130.6 291.1,131.4 293.6,132.1 296.1,132.9 298.6,133.6 301.2,134.4 303.7,135.1 306.2,135.9 308.7,136.6 311.2,137.4 313.7,138.1 316.2,138.9 318.7,139.6 321.3,140.4 323.8,141.1 326.3,141.9 328.8,142.6 331.3,143.4 333.8,144.1 336.3,144.9 338.8,145.6 341.4,146.4 343.9,147.1 346.4,147.9 348.9,148.6 351.4,149.4 353.9,150.1 356.4,150.9 358.9,151.6 361.5,152.4 364.0,153.1 366.5,153.9 369.0,154.6 371.5,155.3 374.0,156.1 376.5,156.8 379.0,157.5 381.6,158.3 384.1,159.0 386.6,159.7 389.1,160.4 391.6,161.2 394.1,161.9 396.6,162.6 399.1,163.3 401.7,164.0 404.2,164.7 406.7,165.4 409.2,166.1 411.7,166.8 414.2,167.5 416.7,168.2 419.2,168.9 421.8,169.6 424.3,170.3 426.8,171.0 429.3,171.7 431.8,172.3 434.3,173.0 436.8,173.7 439.3,174.3 441.9,175.0 444.4,175.7 446.9,176.3 449.4,177.0 451.9,177.6 454.4,178.3 456.9,178.9 459.4,179.6 462.0,180.2 464.5,180.8 467.0,181.4 469.5,182.1 472.0,182.7 474.5,183.3 477.0,183.9 479.5,184.5 482.1,185.1 484.6,185.7 487.1,186.3 489.6,186.9 492.1,187.5 494.6,188.1 497.1,188.6 499.6,189.2 502.2,189.8 504.7,190.4 507.2,190.9 509.7,191.5 512.2,192.0 514.7,192.6 517.2,193.1 519.7,193.7 522.3,194.2 524.8,194.7 527.3,195.2 529.8,195.8 532.3,196.3 534.8,196.8 537.3,197.3 539.8,197.8 542.4,198.3 544.9,198.8 547.4,199.3 549.9,199.8 552.4,200.3 554.9,200.7 557.4,201.2 559.9,201.7 562.5,202.1 565.0,202.6 567.5,203.1 570.0,203.5" class="stroke-a" fill="none"/>
<polyline points="70.0,40.1 72.5,40.1 75.0,40.1 77.5,40.1 80.1,40.1 82.6,40.2 85.1,40.2 87.6,40.2 90.1,40.2 92.6,40.2 95.1,40.2 97.6,40.3 100.2,40.3 102.7,40.3 105.2,40.3 107.7,40.3 110.2,40.4 112.7,40.4 115.2,40.4 117.7,40.5 120.3,40.5 122.8,40.5 125.3,40.6 127.8,40.6 130.3,40.7 132.8,40.7 135.3,40.8 137.8,40.8 140.4,40.9 142.9,41.0 145.4,41.1 147.9,41.1 150.4,41.2 152.9,41.3 155.4,41.4 157.9,41.5 160.5,41.7 163.0,41.8 165.5,41.9 168.0,42.1 170.5,42.2 173.0,42.4 175.5,42.6 178.0,42.8 180.6,43.0 183.1,43.2 185.6,43.5 188.1,43.8 190.6,44.0 193.1,44.3 195.6,44.7 198.1,45.0 200.7,45.4 203.2,45.8 205.7,46.3 208.2,46.8 210.7,47.3 213.2,47.8 215.7,48.4 218.2,49.0 220.8,49.7 223.3,50.4 225.8,51.2 228.3,52.0 230.8,52.9 233.3,53.8 235.8,54.8 238.3,55.9 240.9,57.0 243.4,58.2 245.9,59.5 248.4,60.9 250.9,62.4 253.4,63.9 255.9,65.5 258.4,67.3 261.0,69.1 263.5,71.0 266.0,73.0 268.5,75.2 271.0,77.4 273.5,79.7 276.0,82.2 278.5,84.8 281.1,87.4 283.6,90.2 286.1,93.1 288.6,96.1 291.1,99.2 293.6,102.4 296.1,105.7 298.6,109.0 301.2,112.5 303.7,116.0 306.2,119.6 308.7,123.2 311.2,126.9 313.7,130.6 316.2,134.4 318.7,138.1 321.3,141.9 323.8,145.6 326.3,149.4 328.8,153.1 331.3,156.8 333.8,160.4 336.3,164.0 338.8,167.5 341.4,171.0 343.9,174.3 346.4,177.6 348.9,180.8 351.4,183.9 353.9,186.9 356.4,189.8 358.9,192.6 361.5,195.2 364.0,197.8 366.5,200.3 369.0,202.6 371.5,204.8 374.0,207.0 376.5,209.0 379.0,210.9 381.6,212.7 384.1,214.5 386.6,216.1 389.1,217.6 391.6,219.1 394.1,220.5 396.6,221.8 399.1,223.0 401.7,224.1 404.2,225.2 406.7,226.2 409.2,227.1 411.7,228.0 414.2,228.8 416.7,229.6 419.2,230.3 421.8,231.0 424.3,231.6 426.8,232.2 429.3,232.7 431.8,233.2 434.3,233.7 436.8,234.2 439.3,234.6 441.9,235.0 444.4,235.3 446.9,235.7 449.4,236.0 451.9,236.2 454.4,236.5 456.9,236.8 459.4,237.0 462.0,237.2 464.5,237.4 467.0,237.6 469.5,237.8 472.0,237.9 474.5,238.1 477.0,238.2 479.5,238.3 482.1,238.5 484.6,238.6 487.1,238.7 489.6,238.8 492.1,238.9 494.6,238.9 497.1,239.0 499.6,239.1 502.2,239.2 504.7,239.2 507.2,239.3 509.7,239.3 512.2,239.4 514.7,239.4 517.2,239.5 519.7,239.5 522.3,239.5 524.8,239.6 527.3,239.6 529.8,239.6 532.3,239.7 534.8,239.7 537.3,239.7 539.8,239.7 542.4,239.7 544.9,239.8 547.4,239.8 549.9,239.8 552.4,239.8 554.9,239.8 557.4,239.8 559.9,239.9 562.5,239.9 565.0,239.9 567.5,239.9 570.0,239.9" class="stroke-d" fill="none"/>
<polyline points="70.0,40.0 72.5,40.0 75.0,40.0 77.5,40.0 80.1,40.0 82.6,40.0 85.1,40.0 87.6,40.0 90.1,40.0 92.6,40.0 95.1,40.0 97.6,40.0 100.2,40.0 102.7,40.0 105.2,40.0 107.7,40.0 110.2,40.0 112.7,40.0 115.2,40.0 117.7,40.0 120.3,40.0 122.8,40.0 125.3,40.0 127.8,40.0 130.3,40.0 132.8,40.0 135.3,40.0 137.8,40.0 140.4,40.0 142.9,40.0 145.4,40.0 147.9,40.0 150.4,40.0 152.9,40.0 155.4,40.0 157.9,40.0 160.5,40.0 163.0,40.0 165.5,40.0 168.0,40.0 170.5,40.0 173.0,40.0 175.5,40.0 178.0,40.0 180.6,40.0 183.1,40.1 185.6,40.1 188.1,40.1 190.6,40.1 193.1,40.1 195.6,40.1 198.1,40.1 200.7,40.2 203.2,40.2 205.7,40.2 208.2,40.2 210.7,40.3 213.2,40.3 215.7,40.4 218.2,40.4 220.8,40.5 223.3,40.6 225.8,40.7 228.3,40.8 230.8,40.9 233.3,41.1 235.8,41.3 238.3,41.5 240.9,41.7 243.4,42.0 245.9,42.3 248.4,42.7 250.9,43.1 253.4,43.6 255.9,44.2 258.4,44.9 261.0,45.6 263.5,46.5 266.0,47.5 268.5,48.7 271.0,50.0 273.5,51.6 276.0,53.3 278.5,55.3 281.1,57.6 283.6,60.2 286.1,63.1 288.6,66.4 291.1,70.0 293.6,74.1 296.1,78.6 298.6,83.5 301.2,88.8 303.7,94.6 306.2,100.8 308.7,107.3 311.2,114.2 313.7,121.4 316.2,128.7 318.7,136.2 321.3,143.8 323.8,151.3 326.3,158.6 328.8,165.8 331.3,172.7 333.8,179.2 336.3,185.4 338.8,191.2 341.4,196.5 343.9,201.4 346.4,205.9 348.9,210.0 351.4,213.6 353.9,216.9 356.4,219.8 358.9,222.4 361.5,224.7 364.0,226.7 366.5,228.4 369.0,230.0 371.5,231.3 374.0,232.5 376.5,233.5 379.0,234.4 381.6,235.1 384.1,235.8 386.6,236.4 389.1,236.9 391.6,237.3 394.1,237.7 396.6,238.0 399.1,238.3 401.7,238.5 404.2,238.7 406.7,238.9 409.2,239.1 411.7,239.2 414.2,239.3 416.7,239.4 419.2,239.5 421.8,239.6 424.3,239.6 426.8,239.7 429.3,239.7 431.8,239.8 434.3,239.8 436.8,239.8 439.3,239.8 441.9,239.9 444.4,239.9 446.9,239.9 449.4,239.9 451.9,239.9 454.4,239.9 456.9,239.9 459.4,240.0 462.0,240.0 464.5,240.0 467.0,240.0 469.5,240.0 472.0,240.0 474.5,240.0 477.0,240.0 479.5,240.0 482.1,240.0 484.6,240.0 487.1,240.0 489.6,240.0 492.1,240.0 494.6,240.0 497.1,240.0 499.6,240.0 502.2,240.0 504.7,240.0 507.2,240.0 509.7,240.0 512.2,240.0 514.7,240.0 517.2,240.0 519.7,240.0 522.3,240.0 524.8,240.0 527.3,240.0 529.8,240.0 532.3,240.0 534.8,240.0 537.3,240.0 539.8,240.0 542.4,240.0 544.9,240.0 547.4,240.0 549.9,240.0 552.4,240.0 554.9,240.0 557.4,240.0 559.9,240.0 562.5,240.0 565.0,240.0 567.5,240.0 570.0,240.0" class="stroke-b" fill="none"/>
<circle cx="153.3" cy="40.0" r="6" class="fill-ink"/>
<circle cx="486.7" cy="240.0" r="6" class="fill-ink"/>
<text x="163.3" y="56.0" class="fig-small" text-anchor="start">A: spam</text>
<text x="476.7" y="230.0" class="fig-small" text-anchor="end">B: not spam</text>
<line x1="153.3" y1="240.0" x2="153.3" y2="245.0" class="stroke-muted" />
<text x="153.3" y="260.0" class="fig-small" text-anchor="middle">0</text>
<line x1="320.0" y1="240.0" x2="320.0" y2="245.0" class="stroke-muted" />
<text x="320.0" y="260.0" class="fig-small" text-anchor="middle">1</text>
<line x1="486.7" y1="240.0" x2="486.7" y2="245.0" class="stroke-muted" />
<text x="486.7" y="260.0" class="fig-small" text-anchor="middle">2</text>
<text x="320.0" y="280.0" text-anchor="middle">earlier replies to the sender</text>
<text x="14.0" y="26.0" text-anchor="start">probability of spam</text>
<line x1="445.0" y1="66.0" x2="470.0" y2="66.0" class="stroke-a" />
<text x="478.3" y="70.0" class="fig-small" text-anchor="start">t = 1</text>
<line x1="445.0" y1="86.0" x2="470.0" y2="86.0" class="stroke-d" />
<text x="478.3" y="90.0" class="fig-small" text-anchor="start">t = 5</text>
<line x1="445.0" y1="106.0" x2="470.0" y2="106.0" class="stroke-b" />
<text x="478.3" y="110.0" class="fig-small" text-anchor="start">t = 10</text>
</svg>
Figure 3. Larger weights sharpen the sigmoid around the same boundary. Notice that both emails are already classified correctly at t = 1, yet pushing t higher keeps shrinking the loss.
```

Gradient descent with `lr = 0.5` on these two emails shows the result. The length of the weight vector is 2.06 after 10 steps, 5.28 after 100, 8.60 after 1000, and 10.16 after 3000, while the loss falls from 1.386 at zero weights to 0.377, 0.042, 0.0040, and 0.0013. The weights keep growing, roughly with the logarithm of the number of steps.

The fix is a leash. The ridge fee, with $\lambda = 1$, adds $\tfrac12\lVert w\rVert^2$, which along this line is $t^2$. The leash grows faster than the loss shrinks, so the sum $2\ln(1+e^{-t}) + t^2$ has a bottom, at $t = 0.40$ with a total of 1.19. When the weights are free to leave the line, the ridge fit lands at $(0.176, -0.560)$ with loss 1.11.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Loss along the line w = t(1, -1): the unpenalised loss 2 ln(1 + e^-t) falls towards 0 forever, while adding the ridge fee t squared makes a curve with a minimum near t = 0.40">
<line x1="70.0" y1="260.0" x2="570.0" y2="260.0" class="stroke-grid" />
<text x="62.0" y="264.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="202.5" x2="570.0" y2="202.5" class="stroke-grid" />
<text x="62.0" y="206.5" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="145.0" x2="570.0" y2="145.0" class="stroke-grid" />
<text x="62.0" y="149.0" class="fig-small" text-anchor="end">2</text>
<line x1="70.0" y1="87.5" x2="570.0" y2="87.5" class="stroke-grid" />
<text x="62.0" y="91.5" class="fig-small" text-anchor="end">3</text>
<line x1="70.0" y1="30.0" x2="570.0" y2="30.0" class="stroke-grid" />
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">4</text>
<polyline points="70.0,180.3 72.5,181.1 75.0,182.0 77.5,182.7 80.1,183.5 82.6,184.2 85.1,184.9 87.6,185.6 90.1,186.2 92.6,186.8 95.1,187.3 97.6,187.8 100.2,188.3 102.7,188.8 105.2,189.2 107.7,189.6 110.2,190.0 112.7,190.3 115.2,190.6 117.7,190.9 120.3,191.1 122.8,191.3 125.3,191.5 127.8,191.6 130.3,191.7 132.8,191.8 135.3,191.8 137.8,191.8 140.4,191.8 142.9,191.7 145.4,191.6 147.9,191.5 150.4,191.3 152.9,191.1 155.4,190.9 157.9,190.7 160.5,190.4 163.0,190.1 165.5,189.7 168.0,189.3 170.5,188.9 173.0,188.5 175.5,188.0 178.0,187.5 180.6,186.9 183.1,186.3 185.6,185.7 188.1,185.1 190.6,184.4 193.1,183.7 195.6,183.0 198.1,182.2 200.7,181.4 203.2,180.6 205.7,179.7 208.2,178.8 210.7,177.9 213.2,176.9 215.7,175.9 218.2,174.9 220.8,173.9 223.3,172.8 225.8,171.7 228.3,170.5 230.8,169.3 233.3,168.1 235.8,166.9 238.3,165.6 240.9,164.3 243.4,163.0 245.9,161.6 248.4,160.2 250.9,158.8 253.4,157.3 255.9,155.8 258.4,154.3 261.0,152.8 263.5,151.2 266.0,149.6 268.5,147.9 271.0,146.2 273.5,144.5 276.0,142.8 278.5,141.0 281.1,139.2 283.6,137.4 286.1,135.5 288.6,133.7 291.1,131.7 293.6,129.8 296.1,127.8 298.6,125.8 301.2,123.7 303.7,121.7 306.2,119.6 308.7,117.4 311.2,115.3 313.7,113.1 316.2,110.9 318.7,108.6 321.3,106.3 323.8,104.0 326.3,101.7 328.8,99.3 331.3,96.9 333.8,94.4 336.3,92.0 338.8,89.5 341.4,87.0 343.9,84.4 346.4,81.8 348.9,79.2 351.4,76.6 353.9,73.9 356.4,71.2 358.9,68.5 361.5,65.7 364.0,62.9 366.5,60.1 369.0,57.3 371.5,54.4 374.0,51.5 376.5,48.5 379.0,45.6 381.6,42.6 384.1,39.6 386.6,36.5 389.1,33.4 391.6,30.3" class="stroke-d" fill="none"/>
<polyline points="70.0,180.3 72.5,181.2 75.0,182.0 77.5,182.9 80.1,183.7 82.6,184.5 85.1,185.4 87.6,186.2 90.1,187.0 92.6,187.8 95.1,188.6 97.6,189.4 100.2,190.2 102.7,191.0 105.2,191.8 107.7,192.6 110.2,193.3 112.7,194.1 115.2,194.8 117.7,195.6 120.3,196.3 122.8,197.1 125.3,197.8 127.8,198.5 130.3,199.2 132.8,199.9 135.3,200.6 137.8,201.3 140.4,202.0 142.9,202.7 145.4,203.4 147.9,204.0 150.4,204.7 152.9,205.4 155.4,206.0 157.9,206.7 160.5,207.3 163.0,207.9 165.5,208.6 168.0,209.2 170.5,209.8 173.0,210.4 175.5,211.0 178.0,211.6 180.6,212.2 183.1,212.8 185.6,213.4 188.1,214.0 190.6,214.5 193.1,215.1 195.6,215.6 198.1,216.2 200.7,216.7 203.2,217.3 205.7,217.8 208.2,218.4 210.7,218.9 213.2,219.4 215.7,219.9 218.2,220.4 220.8,220.9 223.3,221.4 225.8,221.9 228.3,222.4 230.8,222.9 233.3,223.3 235.8,223.8 238.3,224.3 240.9,224.7 243.4,225.2 245.9,225.7 248.4,226.1 250.9,226.5 253.4,227.0 255.9,227.4 258.4,227.8 261.0,228.2 263.5,228.7 266.0,229.1 268.5,229.5 271.0,229.9 273.5,230.3 276.0,230.7 278.5,231.1 281.1,231.4 283.6,231.8 286.1,232.2 288.6,232.6 291.1,232.9 293.6,233.3 296.1,233.7 298.6,234.0 301.2,234.4 303.7,234.7 306.2,235.0 308.7,235.4 311.2,235.7 313.7,236.0 316.2,236.4 318.7,236.7 321.3,237.0 323.8,237.3 326.3,237.6 328.8,237.9 331.3,238.2 333.8,238.5 336.3,238.8 338.8,239.1 341.4,239.4 343.9,239.7 346.4,240.0 348.9,240.2 351.4,240.5 353.9,240.8 356.4,241.0 358.9,241.3 361.5,241.6 364.0,241.8 366.5,242.1 369.0,242.3 371.5,242.6 374.0,242.8 376.5,243.0 379.0,243.3 381.6,243.5 384.1,243.7 386.6,244.0 389.1,244.2 391.6,244.4 394.1,244.6 396.6,244.8 399.1,245.1 401.7,245.3 404.2,245.5 406.7,245.7 409.2,245.9 411.7,246.1 414.2,246.3 416.7,246.5 419.2,246.7 421.8,246.8 424.3,247.0 426.8,247.2 429.3,247.4 431.8,247.6 434.3,247.8 436.8,247.9 439.3,248.1 441.9,248.3 444.4,248.4 446.9,248.6 449.4,248.8 451.9,248.9 454.4,249.1 456.9,249.2 459.4,249.4 462.0,249.5 464.5,249.7 467.0,249.8 469.5,250.0 472.0,250.1 474.5,250.3 477.0,250.4 479.5,250.5 482.1,250.7 484.6,250.8 487.1,250.9 489.6,251.1 492.1,251.2 494.6,251.3 497.1,251.5 499.6,251.6 502.2,251.7 504.7,251.8 507.2,251.9 509.7,252.1 512.2,252.2 514.7,252.3 517.2,252.4 519.7,252.5 522.3,252.6 524.8,252.7 527.3,252.8 529.8,252.9 532.3,253.0 534.8,253.1 537.3,253.2 539.8,253.3 542.4,253.4 544.9,253.5 547.4,253.6 549.9,253.7 552.4,253.8 554.9,253.9 557.4,254.0 559.9,254.1 562.5,254.2 565.0,254.2 567.5,254.3 570.0,254.4" class="stroke-b" fill="none"/>
<circle cx="136.8" cy="191.8" r="5" class="fill-d"/>
<line x1="136.8" y1="185.8" x2="90.0" y2="151.0" class="stroke-muted" />
<text x="78.3" y="145.0" class="fig-small" text-anchor="start">minimum 1.19 at t = 0.40</text>
<text x="336.7" y="212.9" class="fig-small" text-anchor="start">loss alone: 2 ln(1 + exp(−t))</text>
<text x="306.7" y="64.5" class="fig-small" text-anchor="end">loss + fee (λ = 1)</text>
<line x1="70.0" y1="260.0" x2="70.0" y2="265.0" class="stroke-muted" />
<text x="70.0" y="280.0" class="fig-small" text-anchor="middle">0</text>
<line x1="236.7" y1="260.0" x2="236.7" y2="265.0" class="stroke-muted" />
<text x="236.7" y="280.0" class="fig-small" text-anchor="middle">1</text>
<line x1="403.3" y1="260.0" x2="403.3" y2="265.0" class="stroke-muted" />
<text x="403.3" y="280.0" class="fig-small" text-anchor="middle">2</text>
<line x1="570.0" y1="260.0" x2="570.0" y2="265.0" class="stroke-muted" />
<text x="570.0" y="280.0" class="fig-small" text-anchor="middle">3</text>
<text x="320.0" y="300.0" text-anchor="middle">distance t along the line w = t(1, −1)</text>
<text x="62.0" y="22.0" text-anchor="end">loss</text>
</svg>
Figure 4. Without a fee the loss only falls. Notice that adding t² turns the falling curve into a bowl with a minimum at t = 0.40.
```

> [!KEY]
> The slope of the log loss for weight $j$ is the sum over rows of error $(p - y)$ times feature $x_j$. Walking against it fits the model, at a pace set by the learning rate. On a perfect split the slope never reaches 0, so the weights grow forever, until a ridge fee puts a bottom under the loss.

> [!RECAP]
> - Each row's loss changes by $p - y$ per unit of score, so the gradient is $X^\top(p - y) + \lambda w$.
> - A step is $w \leftarrow w - \text{lr}\cdot g$. Too small a `lr` crawls, and above $2/L$ the loss rises.
> - The loop records the loss before it steps, so `history[0]` is the loss at zero weights, and it stops when the gradient is shorter than `tol`.
> - On separable data the unpenalised weights grow without limit.
> - A ridge fee gives the loss a finite minimum.

## Build it step by step

Build three functions. The sigmoid, softplus, log loss, and `predict_proba` from the earlier chapters are provided.

1. `gradient(w, X, y, ridge)` takes weights `(d,)`, a matrix `(n, d)`, labels `(n,)`, and returns shape `(d,)`. Use `X.T @ (sigmoid(X @ w) - y) + ridge * w`. The common mistake is `y - p`, which sends every step uphill.

2. `fit_gradient_descent(X, y, lr, max_iter, ridge, tol)` starts at zeros and returns the weights and the loss history. Record the loss, compute the gradient, break if its norm is below `tol`, then step. The common mistake is recording after the update, so the first entry is not the loss at zero weights.

3. `train_accuracy(w, X, y)` returns the fraction of rows whose predicted label equals the true label, with labels from `predict_proba(w, X) >= 0.5`. The common mistake is comparing probabilities with labels directly.

## Where this breaks

- **A perfect split.** The weights keep growing after the held-out score has stopped improving. Add a ridge fee, or stop early.
- **A rare class.** If 99 of 100 emails are not spam, a model that always says "no" is 99 percent accurate. Compare `train_accuracy` against that baseline, and check each class separately.

## The exercise

You implement `gradient`, `fit_gradient_descent`, and `train_accuracy`. The checks compare `gradient` with finite differences of the loss and with the hand result $[-0.1951,\ 0.3649]$, and confirm that a small step against it lowers the loss. They then check that `history[0]` equals $n\ln 2$, that the loop stops on `tol`, and that the fit reaches a flat gradient. The last two checks confirm that accuracy beats the baseline, and that on a perfectly separable inbox the unpenalised weights keep growing while the ridge weights stay small.
