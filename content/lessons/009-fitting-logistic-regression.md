---
title: "Fitting logistic regression: gradient descent and Newton's method"
slug: 009-fitting-logistic-regression
part: "Part 1 — Foundations"
summary: "The log loss has slope X^T(p − y) and bend X^T D X, so descent follows the slope and Newton divides it by the bend."
minutes: 75
gpu: false
objectives:
  - Derive the gradient X^T(p − y) by blaming each weight for the errors on the rows where its feature was on.
  - Take a gradient step by hand, and say how too small and too large a learning rate each fail.
  - Explain why perfectly separable data makes the weights grow without limit, and how a ridge fee stops it.
  - Build the Hessian X^T D X + λI, and take a Newton step by solving H δ = g.
  - Say what Newton costs per step, and what happens on separable data and on duplicate columns.
prereqs:
  - 005-gradient-descent-and-momentum
  - 006-ridge-regression
  - 008-sigmoid-and-log-loss
exercise: 009-fitting-logistic-regression
---

# Fitting logistic regression: gradient descent and Newton's method

The last chapter gave you a number that scores any weights. For your two emails, weights $(0.5, -1)$ score 0.6755. A lower number is better, but which way should each weight move to lower it, and by how much? And what happens when the data can be split perfectly: does the model ever say "good enough"?

Without a direction you could only guess at weights and keep the best guess, which is hopeless once there are hundreds of weights. The slope of the loss gives the direction for every weight at once. But a slope does not say how far to go. On a 60-email inbox, gradient descent needs a step size that you pick by hand: too large and the loss bounces forever, too small and it crawls through 185 passes. Could the loss itself tell you how far to go?

It can, because the loss carries more than a slope. It also carries a bend, which is how fast the slope itself changes. This chapter computes the slope by hand and builds the descent loop. It then measures the bend, divides it out with Newton's method, and fits the same inbox in 7 passes. On the way you meet a surprise: on a perfectly separable inbox the slope never reaches zero.

## What you already know

- The log loss is $L(w) = \sum_i [\operatorname{softplus}(z_i) - y_i z_i]$ with $z = Xw$, and the sigmoid turns a score into a probability $p$ (see [From a score to a probability, and how wrong it is](/lessons/008-sigmoid-and-log-loss)).
- The gradient lists the slope of the loss for every weight, and descent steps against it. A step of size `lr` is safe only below a limit of $2/L$, where $L$ is the largest curvature (see [Gradient descent and momentum](/lessons/005-gradient-descent-and-momentum)).
- For linear regression with the loss $\tfrac12\sum_i(\text{prediction}_i - y_i)^2$, the gradient is $X^\top(\text{prediction} - y)$.
- The ridge fee $\tfrac{\lambda}{2}\lVert w\rVert^2$ penalises large weights (see [Ridge regression](/lessons/006-ridge-regression)).
- `np.linalg.solve(A, b)` finds $x$ with $Ax = b$.

## Blame each weight

Start from one row. Its loss $\operatorname{softplus}(z) - yz$ depends on the weights only through the score $z$. So you need two slopes and the chain rule: how the loss moves when $z$ moves, and how $z$ moves when a weight moves.

The first is easy. The slope of $\ln(1 + e^{z})$ is $e^{z}/(1+e^{z})$, which is $\sigma(z) = p$. The slope of $-yz$ is $-y$. So the loss of one row changes by $p - y$ per unit of score. That is the *error*: the probability minus the truth. The second slope is the feature itself, since $z = \sum_j w_j x_j$ moves by $x_j$ when $w_j$ moves by 1.

Multiply them and add over rows: the slope for weight $j$ is the sum over emails of (error) times (feature $j$). Weight $j$ is blamed for each row in proportion to how wrong that row was and how much its own feature was switched on. A feature that was 0 is not to blame. This is the same shape as linear regression's $X^\top(\text{prediction} - y)$, because the sigmoid's slope cancels against the log and leaves the plain error.

For the two emails the errors are $p - y = (0.6225 - 1,\ 0.1824 - 0) = (-0.3775,\ +0.1824)$. Figure 1 adds the products down each column.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Gradient as blame: each email's error times its feature value, summed down each column, gives the gradient for the bias and the replies weight">
<defs><marker id="c09-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
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
<line x1="228.0" y1="240.0" x2="228.0" y2="204.0" class="stroke-ink" marker-end="url(#c09-arrow)"/>
<text x="242.0" y="220.0" class="fig-small" text-anchor="start">bias goes up:</text>
<text x="242.0" y="236.0" class="fig-small" text-anchor="start">0.5 → 0.695</text>
<line x1="380.0" y1="204.0" x2="380.0" y2="240.0" class="stroke-ink" marker-end="url(#c09-arrow)"/>
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

## Take a step, and choose its size

A negative entry means the loss falls when the weight rises, so descent raises the bias and lowers the replies weight. That is sensible. Both emails share the bias, and A's error is bigger than B's, so the bias rises. Only B has replies, and B was over-predicted, so the replies weight falls. With a step size `lr = 1`:

$$
w \leftarrow w - \text{lr}\cdot\nabla L(w) = (0.5,\,-1) - (-0.1951,\ 0.3649) = (0.6951,\ -1.3649)
$$

The probabilities become 0.667 for A (up from 0.622) and 0.116 for B (down from 0.182), and the loss falls from 0.6755 to 0.528. One step, in the right direction, for every weight at once.

A step of 1 worked here, but a step size is a trade. To see it, use a larger inbox of 60 emails with the same two weights. Zero weights give the loss $60 \ln 2 = 41.59$, and the best weights give 28.80.

- **Too small.** With `lr = 0.001`, the loss reaches 29.29 after 80 steps, then crawls, needing 576 steps to get within 0.01 of the minimum.
- **About right.** With `lr = 0.02`, it is within 0.01 of the minimum after 27 steps.
- **Too large.** With `lr = 0.05`, the first step makes the loss jump from 41.6 up to 61.7, and the loss keeps bouncing without settling.

The limit from [Gradient descent and momentum](/lessons/005-gradient-descent-and-momentum) explains the last case. Near the bottom, the largest curvature of this loss is about 42, so steps above $2/42 = 0.048$ overshoot the bottom by more than they started from.

A loop needs a stopping rule too. At the bottom of the loss the gradient is 0, so stop when its length falls below a small `tol`. The loop starts at zero weights, records the loss before each step so that the first entry is the loss at zero weights, and stops before stepping if the gradient is already small:

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

Larger $t$ is always better. At $t = 1$ the loss is 0.627, at 5 it is 0.0134, and at 10 it is 0.00009. A larger $t$ does not move the boundary, which stays at one reply, but it makes the probabilities more extreme. Each doubling makes every row more confident and every loss term smaller. The loss approaches 0 but never reaches it, since the sigmoid never returns exactly 1, so the gradient never reaches 0 either.

Gradient descent with `lr = 0.5` on these two emails shows the result. The length of the weight vector is 2.06 after 10 steps, 5.28 after 100, 8.60 after 1000, and 10.16 after 3000, while the loss falls from 1.386 at zero weights to 0.377, 0.042, 0.0040, and 0.0013. The weights keep growing, roughly with the logarithm of the number of steps.

The fix is a leash. The ridge fee, with $\lambda = 1$, adds $\tfrac12\lVert w\rVert^2$, which along this line is $t^2$. The leash grows faster than the loss shrinks, so the sum $2\ln(1+e^{-t}) + t^2$ has a bottom, at $t = 0.40$ with a total of 1.19. When the weights are free to leave the line, the ridge fit lands at $(0.176, -0.560)$ with loss 1.11. Figure 2 shows the bowl that the fee builds.

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
Figure 2. Without a fee the loss only falls. Notice that adding t² turns the falling curve into a bowl with a minimum at t = 0.40.
```

> [!KEY]
> The slope of the log loss for weight $j$ is the sum over rows of error $(p - y)$ times feature $x_j$. Walking against it fits the model, at a pace set by the learning rate. On a perfect split the slope never reaches 0, so the weights grow forever, until a ridge fee puts a bottom under the loss.

## Feel the bend

Every number in the step-size list came from you, and the loss itself said nothing about them. To let the loss speak, look at one row. The slope of its loss with respect to its score $z$ is $p - y$. How fast does that slope change as $z$ changes? Only $p$ moves, and the sigmoid's own slope is $\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr) = p(1-p)$. So the row's bend is $p(1 - p)$.

This number has a shape you can read. It is largest, $0.5 \times 0.5 = 0.25$, when $p = 0.5$, and it falls towards 0 as $p$ nears 0 or 1. A confident row barely bends the loss, because moving its score a little changes its probability a little. An undecided row bends it the most. For the two emails, A has $p = 0.6225$ and a bend of $0.6225 \times 0.3775 = 0.2350$, and B has $p = 0.1824$ and a bend of $0.1824 \times 0.8176 = 0.1491$.

Weights, not scores, are what you move, so collect the row bends into a matrix. The *Hessian* is the table of second derivatives: its entry in row $j$, column $k$ says how the slope for weight $j$ changes when weight $k$ moves. A row with features $x$ has slope $(p - y)\,x_j$ for weight $j$. When weight $k$ moves by 1 the score moves by $x_k$, so $p$ changes by $p(1-p)\,x_k$, and the slope for weight $j$ changes by $p(1-p)\,x_j x_k$. That is the row's contribution to the entry. Adding over rows,

$$
H = X^{\top} D X + \lambda I, \qquad D = \operatorname{diag}\bigl(p_1(1-p_1),\ \ldots,\ p_n(1-p_n)\bigr)
$$

Read it aloud: scale each row of $X$ by its bend, multiply by $X^\top$, and add $\lambda$ down the diagonal. For the emails at weights $(0.5, -1)$ the entries are $0.2350 + 0.1491 = 0.3842$, then $2 \times 0.1491 = 0.2983$, then $4 \times 0.1491 = 0.5966$. In numpy, `X * s[:, None]` scales every row without building the $n \times n$ matrix $D$:

```python
s = p * (1.0 - p)
H = (X * s[:, None]).T @ X + ridge * np.eye(d)
```

Every bend $p(1-p)$ is at least 0, so no direction bends downward (a Hessian with this property is called positive semidefinite). That makes the loss a bowl, which means a zero gradient is a bottom and not a trap. Two honest limits apply: the bowl can be flat in a direction that the data cannot tell apart, and for perfectly separable data the bottom sits infinitely far away, as you saw above. Confident rows also lose their bend, which is why the Hessian shrinks as the separable weights grow.

## Fit a parabola and jump

Use the bend. Stand at one weight and know three numbers: the loss, the slope, and the bend. A parabola with those three numbers has an easy bottom, at the current weight minus slope divided by bend. Jump there.

Figure 3 draws this for the replies weight of the inbox, with the bias held fixed at its best value, 0.46. At a weight of 0 the loss is 49.2, the slope is 66.1, and the bend is 115. So the jump is $66.1 / 115 = 0.57$ to the left, landing at $-0.57$ with loss 29.6. A gradient step of 0.005 moves only to $-0.33$ with loss 33.8. The true minimum is at $-0.76$.

```figure
<svg viewBox="0 0 600 360" role="img" aria-label="Newton in one dimension: a parabola matched to the loss at the current weight, with an arrow to its bottom, next to a gradient step that lands short">
<defs><marker id="c09-arrow-3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="70.0" y1="196.4" x2="570.0" y2="196.4" class="stroke-grid" />
<text x="62.0" y="200.4" class="fig-small" text-anchor="end">30</text>
<line x1="70.0" y1="144.4" x2="570.0" y2="144.4" class="stroke-grid" />
<text x="62.0" y="148.4" class="fig-small" text-anchor="end">40</text>
<line x1="70.0" y1="92.4" x2="570.0" y2="92.4" class="stroke-grid" />
<text x="62.0" y="96.4" class="fig-small" text-anchor="end">50</text>
<line x1="70.0" y1="40.4" x2="570.0" y2="40.4" class="stroke-grid" />
<text x="62.0" y="44.4" class="fig-small" text-anchor="end">60</text>
<line x1="70.0" y1="290.0" x2="570.0" y2="290.0" class="stroke-muted" />
<line x1="125.6" y1="290.0" x2="125.6" y2="295.0" class="stroke-muted" />
<text x="125.6" y="310.0" class="fig-small" text-anchor="middle">-1.2</text>
<line x1="236.7" y1="290.0" x2="236.7" y2="295.0" class="stroke-muted" />
<text x="236.7" y="310.0" class="fig-small" text-anchor="middle">-0.8</text>
<line x1="347.8" y1="290.0" x2="347.8" y2="295.0" class="stroke-muted" />
<text x="347.8" y="310.0" class="fig-small" text-anchor="middle">-0.4</text>
<line x1="458.9" y1="290.0" x2="458.9" y2="295.0" class="stroke-muted" />
<text x="458.9" y="310.0" class="fig-small" text-anchor="middle">0</text>
<line x1="570.0" y1="290.0" x2="570.0" y2="295.0" class="stroke-muted" />
<text x="570.0" y="310.0" class="fig-small" text-anchor="middle">0.4</text>
<text x="320.0" y="332.0" text-anchor="middle">weight on replies (bias held at its best value)</text>
<text x="62.0" y="22.0" text-anchor="end">loss</text>
<polyline points="95.0,31.8 97.8,36.2 100.6,40.5 103.3,44.8 106.1,49.0 108.9,53.2 111.7,57.2 114.4,61.3 117.2,65.3 120.0,69.2 122.8,73.0 125.6,76.8 128.3,80.6 131.1,84.2 133.9,87.8 136.7,91.4 139.4,94.9 142.2,98.3 145.0,101.7 147.8,105.0 150.6,108.3 153.3,111.5 156.1,114.6 158.9,117.7 161.7,120.7 164.4,123.6 167.2,126.5 170.0,129.4 172.8,132.2 175.6,134.9 178.3,137.5 181.1,140.1 183.9,142.7 186.7,145.1 189.4,147.5 192.2,149.9 195.0,152.2 197.8,154.4 200.6,156.6 203.3,158.7 206.1,160.8 208.9,162.8 211.7,164.7 214.4,166.6 217.2,168.4 220.0,170.1 222.8,171.8 225.6,173.5 228.3,175.0 231.1,176.6 233.9,178.0 236.7,179.4 239.4,180.7 242.2,182.0 245.0,183.2 247.8,184.4 250.6,185.5 253.3,186.5 256.1,187.5 258.9,188.4 261.7,189.2 264.4,190.0 267.2,190.8 270.0,191.5 272.8,192.1 275.6,192.6 278.3,193.1 281.1,193.5 283.9,193.9 286.7,194.2 289.4,194.5 292.2,194.7 295.0,194.8 297.8,194.9 300.6,194.9 303.3,194.9 306.1,194.7 308.9,194.6 311.7,194.4 314.4,194.1 317.2,193.7 320.0,193.3 322.8,192.9 325.6,192.3 328.3,191.7 331.1,191.1 333.9,190.4 336.7,189.6 339.4,188.8 342.2,187.9 345.0,187.0 347.8,185.9 350.6,184.9 353.3,183.8 356.1,182.6 358.9,181.3 361.7,180.0 364.4,178.6 367.2,177.2 370.0,175.7 372.8,174.2 375.6,172.6 378.3,170.9 381.1,169.2 383.9,167.4 386.7,165.6 389.4,163.6 392.2,161.7 395.0,159.7 397.8,157.6 400.6,155.4 403.3,153.2 406.1,150.9 408.9,148.6 411.7,146.2 414.4,143.8 417.2,141.3 420.0,138.7 422.8,136.1 425.6,133.4 428.3,130.6 431.1,127.8 433.9,125.0 436.7,122.0 439.4,119.1 442.2,116.0 445.0,112.9 447.8,109.7 450.6,106.5 453.3,103.2 456.1,99.9 458.9,96.5 461.7,93.0 464.4,89.5 467.2,85.9 470.0,82.2 472.8,78.5 475.6,74.8 478.3,70.9 481.1,67.0 483.9,63.1 486.7,59.1 489.4,55.0 492.2,50.9 495.0,46.7 497.8,42.4 500.6,38.1 503.3,33.8" class="stroke-d dashed" fill="none"/>
<polyline points="70.0,178.5 72.8,179.1 75.6,179.7 78.3,180.3 81.1,180.9 83.9,181.5 86.7,182.1 89.4,182.7 92.2,183.3 95.0,183.8 97.8,184.4 100.6,185.0 103.3,185.5 106.1,186.1 108.9,186.6 111.7,187.1 114.4,187.7 117.2,188.2 120.0,188.7 122.8,189.2 125.6,189.7 128.3,190.2 131.1,190.7 133.9,191.2 136.7,191.7 139.4,192.1 142.2,192.6 145.0,193.1 147.8,193.5 150.6,194.0 153.3,194.4 156.1,194.8 158.9,195.2 161.7,195.6 164.4,196.0 167.2,196.4 170.0,196.8 172.8,197.2 175.6,197.5 178.3,197.9 181.1,198.2 183.9,198.6 186.7,198.9 189.4,199.2 192.2,199.5 195.0,199.8 197.8,200.0 200.6,200.3 203.3,200.6 206.1,200.8 208.9,201.0 211.7,201.2 214.4,201.4 217.2,201.6 220.0,201.8 222.8,201.9 225.6,202.1 228.3,202.2 231.1,202.3 233.9,202.4 236.7,202.5 239.4,202.5 242.2,202.6 245.0,202.6 247.8,202.6 250.6,202.6 253.3,202.6 256.1,202.5 258.9,202.4 261.7,202.3 264.4,202.2 267.2,202.1 270.0,201.9 272.8,201.7 275.6,201.5 278.3,201.3 281.1,201.1 283.9,200.8 286.7,200.5 289.4,200.1 292.2,199.8 295.0,199.4 297.8,198.9 300.6,198.5 303.3,198.0 306.1,197.5 308.9,196.9 311.7,196.3 314.4,195.7 317.2,195.1 320.0,194.4 322.8,193.7 325.6,192.9 328.3,192.1 331.1,191.3 333.9,190.4 336.7,189.5 339.4,188.5 342.2,187.5 345.0,186.5 347.8,185.4 350.6,184.2 353.3,183.0 356.1,181.8 358.9,180.5 361.7,179.2 364.4,177.8 367.2,176.4 370.0,174.9 372.8,173.4 375.6,171.8 378.3,170.2 381.1,168.5 383.9,166.7 386.7,164.9 389.4,163.0 392.2,161.1 395.0,159.1 397.8,157.1 400.6,155.0 403.3,152.8 406.1,150.6 408.9,148.3 411.7,145.9 414.4,143.5 417.2,141.1 420.0,138.5 422.8,135.9 425.6,133.3 428.3,130.6 431.1,127.8 433.9,124.9 436.7,122.0 439.4,119.0 442.2,116.0 445.0,112.9 447.8,109.7 450.6,106.5 453.3,103.2 456.1,99.9 458.9,96.5 461.7,93.0 464.4,89.5 467.2,85.9 470.0,82.2 472.8,78.5 475.6,74.8 478.3,71.0 481.1,67.1 483.9,63.2 486.7,59.2 489.4,55.2 492.2,51.1 495.0,46.9 497.8,42.8 500.6,38.5 503.3,34.3" class="stroke-b" fill="none"/>
<circle cx="458.9" cy="96.5" r="6" class="fill-a"/>
<text x="448.9" y="90.5" class="fig-small" text-anchor="end">start: loss 49.2, slope 66.1, bend 115</text>
<circle cx="299.8" cy="194.9" r="5" class="fill-d"/>
<circle cx="299.8" cy="198.6" r="5" class="fill-c"/>
<circle cx="247.2" cy="202.6" r="4" class="fill-ink"/>
<text x="239.2" y="192.6" class="fig-small" text-anchor="end">true minimum</text>
<line x1="458.9" y1="258.8" x2="303.8" y2="258.8" class="stroke-ink" marker-end="url(#c09-arrow-3)"/>
<text x="295.8" y="250.8" class="fig-small" text-anchor="start">Newton jump: to -0.57</text>
<line x1="458.9" y1="230.2" x2="371.1" y2="230.2" class="stroke-ink" marker-end="url(#c09-arrow-3)"/>
<text x="363.1" y="222.2" class="fig-small" text-anchor="start">gradient step (lr 0.005): to -0.33</text>
<line x1="299.8" y1="194.9" x2="299.8" y2="290.0" class="stroke-grid dashed" />
<text x="208.9" y="45.6" class="fig-small" text-anchor="start">dashed: parabola with the same</text>
<text x="208.9" y="61.6" class="fig-small" text-anchor="start">height, slope, and bend</text>
</svg>
Figure 3. The dashed parabola shares the solid curve's height, slope, and bend at the start. Notice that the Newton jump goes to the parabola's bottom and lands near the true minimum, and that the real curve flattens faster than the parabola, so one jump does not land exactly.
```

Read the jump as gradient descent with a step size of $1/\text{bend}$ chosen for you at every point. Where the loss is flat the step is long, and where it bends sharply the step is short. This is why a single fixed step size cannot match it.

For many weights, the division becomes a linear solve. Find $\delta$ with $H\delta = g$, then subtract it:

$$
w \leftarrow w - \delta, \qquad H\delta = g, \quad g = \nabla L(w)
$$

Check it on the emails at $(0.5, -1)$, where $g = (-0.1951,\ 0.3649)$. Solving gives $\delta = (-1.6065,\ 1.4148)$. The first row checks out: $0.3842 \times (-1.6065) + 0.2983 \times 1.4148 = -0.1952$, equal to $g_1$ up to rounding. The updated weights are $(2.1065, -2.4148)$ and the loss is 0.178, against 0.528 for the gradient step of size 1.

From zero weights the arithmetic is even cleaner. Every $p$ is 0.5, so every bend is 0.25, the gradient is $(0, 1)$, and $H = \begin{bmatrix}0.5 & 0.5\\ 0.5 & 1\end{bmatrix}$. The solution is $\delta = (-2, 2)$, so the weights become $(2, -2)$ and the loss falls from 1.386 to 0.254. A gradient step of size 1 only reaches $(0, -1)$, with loss 0.820. In numpy:

```python
delta = np.linalg.solve(H, g)
w = w - delta
```

> [!TIP]
> Use `solve` and not `np.linalg.inv(H) @ g`. It is cheaper, and when $H$ is nearly flat in some direction the inverse amplifies rounding noise.

## Watch it converge

The Newton loop is the descent loop with a different step: record the loss, compute the gradient, stop if it is short, build $H$, and subtract $\delta$. On the 60-email inbox it stops after 7 passes, against 185 for gradient descent. The loss above its minimum goes 12.8, 1.01, 0.057, 0.0003, $1.3 \times 10^{-8}$, and then rounding noise. Near the bottom the loss is almost a parabola, so each jump roughly doubles the number of correct digits. Figure 4 plots the gap on a log scale.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Distance of the loss above its minimum on a log scale against step number: Newton is below 1e-13 after 5 steps while gradient descent is near 1e-7 after 100">
<line x1="70.0" y1="70.7" x2="570.0" y2="70.7" class="stroke-grid" />
<text x="62.0" y="74.7" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="132.0" x2="570.0" y2="132.0" class="stroke-grid" />
<text x="62.0" y="136.0" class="fig-small" text-anchor="end">1e-4</text>
<line x1="70.0" y1="193.3" x2="570.0" y2="193.3" class="stroke-grid" />
<text x="62.0" y="197.3" class="fig-small" text-anchor="end">1e-8</text>
<line x1="70.0" y1="254.7" x2="570.0" y2="254.7" class="stroke-grid" />
<text x="62.0" y="258.7" class="fig-small" text-anchor="end">1e-12</text>
<line x1="70.0" y1="270.0" x2="570.0" y2="270.0" class="stroke-muted" />
<line x1="70.0" y1="270.0" x2="70.0" y2="275.0" class="stroke-muted" />
<text x="70.0" y="290.0" class="fig-small" text-anchor="middle">0</text>
<line x1="170.0" y1="270.0" x2="170.0" y2="275.0" class="stroke-muted" />
<text x="170.0" y="290.0" class="fig-small" text-anchor="middle">20</text>
<line x1="270.0" y1="270.0" x2="270.0" y2="275.0" class="stroke-muted" />
<text x="270.0" y="290.0" class="fig-small" text-anchor="middle">40</text>
<line x1="370.0" y1="270.0" x2="370.0" y2="275.0" class="stroke-muted" />
<text x="370.0" y="290.0" class="fig-small" text-anchor="middle">60</text>
<line x1="470.0" y1="270.0" x2="470.0" y2="275.0" class="stroke-muted" />
<text x="470.0" y="290.0" class="fig-small" text-anchor="middle">80</text>
<line x1="570.0" y1="270.0" x2="570.0" y2="275.0" class="stroke-muted" />
<text x="570.0" y="290.0" class="fig-small" text-anchor="middle">100</text>
<text x="320.0" y="310.0" text-anchor="middle">step</text>
<text x="70.0" y="22.0" text-anchor="start">loss above its minimum</text>
<polyline points="70.0,53.7 75.0,60.2 80.0,67.8 85.0,75.0 90.0,77.2 95.0,78.3 100.0,79.4 105.0,80.5 110.0,81.6 115.0,82.7 120.0,83.8 125.0,84.9 130.0,86.0 135.0,87.0 140.0,88.1 145.0,89.2 150.0,90.3 155.0,91.4 160.0,92.5 165.0,93.5 170.0,94.6 175.0,95.7 180.0,96.8 185.0,97.8 190.0,98.9 195.0,100.0 200.0,101.0 205.0,102.1 210.0,103.2 215.0,104.2 220.0,105.3 225.0,106.4 230.0,107.4 235.0,108.5 240.0,109.6 245.0,110.6 250.0,111.7 255.0,112.7 260.0,113.8 265.0,114.9 270.0,115.9 275.0,117.0 280.0,118.1 285.0,119.1 290.0,120.2 295.0,121.2 300.0,122.3 305.0,123.4 310.0,124.4 315.0,125.5 320.0,126.5 325.0,127.6 330.0,128.7 335.0,129.7 340.0,130.8 345.0,131.8 350.0,132.9 355.0,133.9 360.0,135.0 365.0,136.1 370.0,137.1 375.0,138.2 380.0,139.2 385.0,140.3 390.0,141.4 395.0,142.4 400.0,143.5 405.0,144.5 410.0,145.6 415.0,146.6 420.0,147.7 425.0,148.8 430.0,149.8 435.0,150.9 440.0,151.9 445.0,153.0 450.0,154.1 455.0,155.1 460.0,156.2 465.0,157.2 470.0,158.3 475.0,159.3 480.0,160.4 485.0,161.5 490.0,162.5 495.0,163.6 500.0,164.6 505.0,165.7 510.0,166.7 515.0,167.8 520.0,168.9 525.0,169.9 530.0,171.0 535.0,172.0 540.0,173.1 545.0,174.1 550.0,175.2 555.0,176.3 560.0,177.3 565.0,178.4 570.0,179.4" class="stroke-a" fill="none"/>
<polyline points="70.0,53.7 75.0,70.6 80.0,89.8 85.0,124.2 90.0,191.6 95.0,270.0 100.0,270.0" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="53.7" r="3.5" class="fill-b"/>
<circle cx="75.0" cy="70.6" r="3.5" class="fill-b"/>
<circle cx="80.0" cy="89.8" r="3.5" class="fill-b"/>
<circle cx="85.0" cy="124.2" r="3.5" class="fill-b"/>
<circle cx="90.0" cy="191.6" r="3.5" class="fill-b"/>
<circle cx="95.0" cy="270.0" r="3.5" class="fill-b"/>
<circle cx="100.0" cy="270.0" r="3.5" class="fill-b"/>
<text x="110.0" y="124.2" class="fig-small" text-anchor="start">Newton: 7 passes</text>
<text x="320.0" y="116.5" class="fig-small" text-anchor="start">gradient descent (lr 0.02): 185 passes in all</text>
</svg>
Figure 4. The loss above its minimum, on a log scale, for the same inbox. Notice that Newton's seven dots fall off a cliff while gradient descent, which has no sense of the bend, slides down a gentle slope. Values below 1e-13 are drawn at that floor.
```

This is not a guarantee for every start. The parabola is only a local picture, so far from the bottom a full jump can overshoot. From a replies weight of $-1.5$, the bend is 7.1, so the jump is $12.5 / 7.1 = 1.77$ to the right and lands at $+0.27$, where the loss is 70.7, more than twice the 34.6 it started at. Starting from zero weights, as `fit_newton` does, this inbox behaves, but the loss falling every step is a property of this data and not a law.

Newton also runs on the perfect split, and it behaves differently from descent. The two emails have no bottom, yet every Newton step adds about 1 to the length $t$ along $(1, -1)$, because on a loss that decays like $e^{-t}$ the slope and the bend are equal, so slope divided by bend is 1. It stops after 19 passes at weights $(15.3, -15.8)$, not because it found a bottom but because the gradient fell below $10^{-10}$. Giving the loss a ridge fee $\lambda = 1$ puts a bottom back: Newton reaches $(0.176, -0.560)$, with loss 1.110, in 5 passes.

The default `ridge=1e-8` in `fit_newton` does something else. It does not stop growth, since $10^{-8}$ is far too small to act as a leash. It keeps the solve valid. If two columns of $X$ are copies of each other, $H$ has no inverse, and `np.linalg.solve` fails with a singular matrix error. Adding $10^{-8}$ on the diagonal makes it invertible, and the fit finishes in 6 passes.

> [!KEY]
> Each row bends the loss by $p(1-p)$, so confident rows barely bend it and the Hessian $X^\top DX + \lambda I$ is never negative. Newton fits a parabola at the current weights and jumps to its bottom, which is gradient descent with a step size picked for you from the bend. The loss itself therefore tells you how far to go, and the inbox takes 7 passes instead of 185.

> [!RECAP]
> - The gradient is $X^\top(p - y) + \lambda w$, and a step is $w \leftarrow w - \text{lr}\cdot g$. Too small a `lr` crawls, and above $2/L$ the loss rises.
> - The loop records the loss before it steps, so `history[0]` is the loss at zero weights, and it stops when the gradient is shorter than `tol`.
> - On separable data the unpenalised weights grow without limit, and a ridge fee gives the loss a finite minimum.
> - A row's bend is $p(1-p)$, at most 0.25 and near 0 when the row is sure. The Hessian is $X^\top DX + \lambda I$, with shape `(d, d)`, symmetric, and never bending downward.
> - A Newton step solves $H\delta = g$ and subtracts $\delta$. Use `solve`, not an inverse.

## Build it step by step

Build six functions, in this order. The sigmoid, softplus, log loss, and `predict_proba` from the last chapter are provided.

1. `gradient(w, X, y, ridge)` takes weights `(d,)`, a matrix `(n, d)`, labels `(n,)`, and returns shape `(d,)`. Use `X.T @ (sigmoid(X @ w) - y) + ridge * w`. The common mistake is `y - p`, which sends every step uphill.

2. `hessian(w, X, y, ridge)` returns shape `(d, d)`. Compute `p = sigmoid(X @ w)`, scale the rows of `X` by `p * (1 - p)`, multiply by `X`, and add `ridge * np.eye(d)`. The common mistake is `X * s` without `[:, None]`, which scales the columns.

3. `fit_gradient_descent(X, y, lr, max_iter, ridge, tol)` starts at zeros and returns the weights and the loss history. Record the loss, compute the gradient, break if its norm is below `tol`, then step. The common mistake is recording after the update, so the first entry is not the loss at zero weights.

4. `newton_step(w, X, y, ridge)` returns $\delta$, the vector to subtract, as `np.linalg.solve(H, g)`. It does not subtract anything itself.

5. `fit_newton(X, y, max_iter, ridge, tol)` starts at zeros and returns the weights and the loss history, with the same recording rule as gradient descent. Its default `ridge` is `1e-8`. The common mistake is to drop that default, so a duplicate column makes the solve fail.

6. `train_accuracy(w, X, y)` returns the fraction of rows whose predicted label equals the true label, with labels from `predict_proba(w, X) >= 0.5`. The common mistake is comparing probabilities with labels directly.

## Cost and scaling

Few steps do not mean a cheap fit, because each Newton step does more work:

| Piece | Work per step | Memory |
|---|---|---|
| Gradient | about $nd$ | $d$ numbers |
| Hessian | about $nd^2$ | $d^2$ numbers |
| Solve $H\delta = g$ | about $d^3$ | reuses $H$ |

With a handful of weights, seven expensive steps beat hundreds of cheap ones. With 10,000 weights, $H$ holds 100 million numbers and the solve takes about $10^{12}$ operations, so large models use gradient descent, or L-BFGS, which builds a cheap running estimate of the bend from the last few gradients instead of storing $H$.

## Where this breaks

- **A perfect split.** Descent keeps growing the weights after the held-out score has stopped improving, and Newton marches off without a bottom. Use a real ridge fee, or stop early.
- **Overshoot.** A full Newton step far from the bottom can raise the loss. Production code shrinks the step until the loss falls.
- **A rare class.** If 99 of 100 emails are not spam, a model that always says "no" is 99 percent accurate. Compare `train_accuracy` against that baseline, and check each class separately.
- **Many weights.** The cost of $H$ and the solve grows with $d^2$ and $d^3$. Use gradient descent or L-BFGS.

A logistic regression on a network's hidden states is a probe; see [interpretability](/lessons/083-interpretability).

## The exercise

You implement `gradient`, `hessian`, `fit_gradient_descent`, `newton_step`, `fit_newton`, and `train_accuracy`. The checks compare `gradient` with finite differences of the loss and with the hand result $[-0.1951,\ 0.3649]$, and confirm that a small step against it lowers the loss. They compare `hessian` with finite differences of `gradient`, with the hand values above, and with the symmetry and positive-semidefinite rules. They check that `history[0]` equals $n\ln 2$ and that the descent loop stops on `tol`, and that `newton_step` solves $H\delta = g$ and reproduces the step $(-2, 2)$. They run `fit_newton` against gradient descent on the same seeded data and check that it finishes in under ten passes at a stationary point. The last checks confirm that accuracy beats the baseline, that on separable data the unpenalised descent weights keep growing while the ridge weights stay small, and that Newton handles a perfect split with and without a real ridge fee and a matrix with a duplicate column.
