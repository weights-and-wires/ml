---
title: "Convex optimisation and Lagrange duality: bounds, certificates, and gaps"
slug: 019-convex-duality
part: "Part 2 — Classical models"
summary: Price each broken rule with a multiplier, get a lower bound on the best score, and use the gap between the two to prove you are done.
minutes: 75
gpu: false
objectives:
  - Say when a local minimum is the global minimum, using the chord test.
  - Build the Lagrangian and show that the dual function is a lower bound on the best score.
  - State Slater's condition and the four KKT conditions, and check them on a small problem.
  - Compute the dual of a linear program, an equality-constrained quadratic, and ridge regression.
  - Turn a lasso residual into a feasible dual point, and use the duality gap as a stopping test.
prereqs:
  - 006-ridge-regression
  - 007-lasso-and-fista
  - 018-gda-logistic-and-naive-bayes
exercise: 019-convex-duality
---

# Convex optimisation and Lagrange duality: bounds, certificates, and gaps

A bakery must bake at least one tray of bread, and baking $x$ trays costs $x^2$. The cheapest plan is to bake exactly one tray, for a cost of 1. Suppose a problem has 40 unknowns and 20 rules, and a downhill walk reaches 93.9. Is that the best? A flat loss curve does not say, because a curve also flattens on a plateau far from the bottom.

Lagrange duality answers with a number you can compute. Every rule gets a price, and each choice of prices gives a *lower bound*: a score that no allowed plan can beat. A good plan from above and good prices from below squeeze the best score between them. When they meet, you have a proof that you are done, and when they are close, you know how close.

This chapter builds the tool on the smallest example that shows all of it. Then it uses the tool on four problems: a linear program, a quadratic with equality rules, ridge regression, and the lasso. The [support vector machine](/lessons/020-support-vector-machines) chapter that follows uses the same tool to describe the widest road by pushes, so this chapter derives the tool in general and leaves the SVM to that one.

## What you already know

- A minimum of a smooth function is where the slope is zero, as in [gradient descent and momentum](/lessons/005-gradient-descent-and-momentum).
- Ridge regression solves $(X^\top X + \lambda I)w = X^\top y$, as in [ridge regression](/lessons/006-ridge-regression).
- The lasso charges a fee $\lambda$ on absolute weights, and [FISTA](/lessons/007-lasso-and-fista) fits it by a gradient step and then soft thresholding.
- At the lasso answer, every column's correlation with the residual is at most $\lambda$ in size.

## When a local bottom is the bottom

Gradient descent stops where the slope is zero, a *local* minimum: nothing nearby is lower. A global minimum is the lowest point anywhere, and for a general function the two differ. Figure 1 shows a bowl on the left and a curve with two dips of different depths on the right.

```figure
<svg viewBox="0 0 600 240" role="img" aria-label="Left: a bowl with its chord above it. Right: a curve with two dips, where the chord dips below the curve.">
<line x1="30" y1="190.0" x2="260.0" y2="190.0" class="stroke-muted"/>
<text x="30" y="28" class="fig-bold">Convex: chord above the curve</text>
<polyline points="30.0,57.8 32.9,63.6 35.8,69.3 38.7,74.8 41.6,80.2 44.6,85.4 47.5,90.5 50.4,95.5 53.3,100.2 56.2,104.9 59.1,109.4 62.0,113.7 64.9,117.9 67.8,121.9 70.8,125.8 73.7,129.6 76.6,133.2 79.5,136.6 82.4,139.9 85.3,143.0 88.2,146.0 91.1,148.9 94.1,151.5 97.0,154.1 99.9,156.5 102.8,158.7 105.7,160.8 108.6,162.8 111.5,164.6 114.4,166.2 117.3,167.7 120.3,169.0 123.2,170.2 126.1,171.3 129.0,172.2 131.9,172.9 134.8,173.5 137.7,174.0 140.6,174.3 143.5,174.4 146.5,174.4 149.4,174.3 152.3,174.0 155.2,173.5 158.1,172.9 161.0,172.2 163.9,171.3 166.8,170.2 169.7,169.0 172.7,167.7 175.6,166.2 178.5,164.6 181.4,162.8 184.3,160.8 187.2,158.7 190.1,156.5 193.0,154.1 195.9,151.5 198.9,148.9 201.8,146.0 204.7,143.0 207.6,139.9 210.5,136.6 213.4,133.2 216.3,129.6 219.2,125.8 222.2,121.9 225.1,117.9 228.0,113.7 230.9,109.4 233.8,104.9 236.7,100.2 239.6,95.5 242.5,90.5 245.4,85.4 248.4,80.2 251.3,74.8 254.2,69.3 257.1,63.6 260.0,57.8" fill="none" class="stroke-a" stroke-width="2"/>
<line x1="53.0" y1="99.8" x2="237.0" y2="99.8" class="stroke-b dashed" stroke-width="2"/>
<circle cx="53.0" cy="99.8" r="4.5" class="fill-b"/>
<circle cx="237.0" cy="99.8" r="4.5" class="fill-b"/>
<circle cx="145.0" cy="174.4" r="5" class="fill-c"/>
<text x="145.0" y="194.4" class="fig-small" text-anchor="middle">min 0.0</text>
<line x1="330" y1="190.0" x2="560.0" y2="190.0" class="stroke-muted"/>
<text x="330" y="28" class="fig-bold">Not convex: two dips</text>
<polyline points="330.0,123.0 332.9,132.2 335.8,140.4 338.7,147.6 341.6,153.8 344.6,159.2 347.5,163.8 350.4,167.6 353.3,170.7 356.2,173.2 359.1,175.1 362.0,176.4 364.9,177.2 367.8,177.6 370.8,177.6 373.7,177.2 376.6,176.4 379.5,175.4 382.4,174.1 385.3,172.7 388.2,171.0 391.1,169.2 394.1,167.3 397.0,165.3 399.9,163.2 402.8,161.1 405.7,159.0 408.6,156.9 411.5,154.9 414.4,152.9 417.3,151.0 420.3,149.2 423.2,147.4 426.1,145.8 429.0,144.4 431.9,143.1 434.8,141.9 437.7,140.9 440.6,140.1 443.5,139.4 446.5,138.9 449.4,138.5 452.3,138.3 455.2,138.3 458.1,138.4 461.0,138.7 463.9,139.1 466.8,139.7 469.7,140.4 472.7,141.2 475.6,142.1 478.5,143.0 481.4,144.0 484.3,145.1 487.2,146.1 490.1,147.2 493.0,148.2 495.9,149.2 498.9,150.1 501.8,150.9 504.7,151.5 507.6,152.0 510.5,152.2 513.4,152.2 516.3,151.9 519.2,151.3 522.2,150.3 525.1,148.9 528.0,147.0 530.9,144.7 533.8,141.8 536.7,138.3 539.6,134.1 542.5,129.3 545.4,123.6 548.4,117.2 551.3,109.9 554.2,101.7 557.1,92.5 560.0,82.2" fill="none" class="stroke-a" stroke-width="2"/>
<line x1="358.8" y1="174.9" x2="531.2" y2="144.3" class="stroke-b dashed" stroke-width="2"/>
<circle cx="358.8" cy="174.9" r="4.5" class="fill-b"/>
<circle cx="531.2" cy="144.3" r="4.5" class="fill-b"/>
<circle cx="369.0" cy="177.6" r="5" class="fill-c"/>
<text x="369.0" y="197.6" class="fig-small" text-anchor="middle">-1.51</text>
<circle cx="511.9" cy="152.2" r="5" class="fill-c"/>
<text x="511.9" y="172.2" class="fig-small" text-anchor="middle">-0.52</text>
</svg>
Figure 1. Left: the chord between two points on a bowl lies above the curve, which is the convex test. Right: with two dips, the chord drops below the curve, and the dips have different depths (-1.51 and -0.52), so a downhill walk can stop in the shallow one.
```

Pick any two points on a curve and draw the straight segment between them, called a *chord*. On the bowl the chord always lies on or above the curve. A function with that property is *convex*. In symbols, for any points $x$, $y$ and any weight $t$ between 0 and 1:

$$
f\big(t x + (1 - t) y\big) \le t f(x) + (1 - t) f(y).
$$

Read it aloud: the function at a blend of two points is at most the blend of the function values. For $f(x) = x^2$ with $x = -1$, $y = 2$, and $t = 0.5$, the left side is $f(0.5) = 0.25$ and the right side is $0.5(1) + 0.5(4) = 2.5$.

Convexity makes a local bottom a global one. Suppose a convex $f$ had a local minimum at $a$ and a lower point $b$. Points on the chord just next to $a$ have chord height below $f(a)$, and the curve sits at or below the chord there, so points arbitrarily near $a$ are lower than $a$. That contradicts $a$ being a local minimum.

> [!KEY]
> If the score is convex, each inequality rule $g_i(x) \le 0$ has a convex $g_i$, and each equality rule $h_j(x) = 0$ has a straight-line (affine) $h_j$, then every local minimum is a global minimum. Every problem in this chapter has that shape, and the bounds below are only tight because it does.

## Price a broken rule: the Lagrangian

Return to the bakery. Write the rule so that "allowed" means "at most zero": $g(x) = 1 - x \le 0$. The problem is to minimise $f(x) = x^2$ subject to $g(x) \le 0$.

Drop the rule and charge a price $\lambda \ge 0$ for each unit of violation instead. Baking $x = 0.5$ breaks the rule by $g = 0.5$, so a price of 2 adds a fee of 1. The *Lagrangian* adds the priced rule to the score:

$$
L(x, \lambda) = x^2 + \lambda\,(1 - x).
$$

When the plan is allowed, $g(x) \le 0$ and the fee is zero or negative, so $L(x, \lambda) \le f(x)$. When the plan breaks the rule, the fee is positive. An equality rule $h(x) = 0$ gets a price $\nu$ of either sign, because breaking it in either direction is a violation. With several rules, give each its own price:

$$
L(x, \lambda, \nu) = f(x) + \sum_i \lambda_i\, g_i(x) + \sum_j \nu_j\, h_j(x), \qquad \lambda_i \ge 0.
$$

Read it aloud: the score plus a price times the size of each rule. The inequality prices must be nonnegative, because a negative price would pay you to break a rule, and then $L$ would no longer sit below $f$ on allowed plans.

```python
def lagrangian(x, lam, nu, f, gs, hs):
    return f(x) + sum(l * g(x) for l, g in zip(lam, gs)) + sum(n * h(x) for n, h in zip(nu, hs))
```

At $\lambda = 2$ the plan $x = 1$ scores $L = 1 + 2(0) = 1$, and $x = 3$ scores $L = 9 + 2(-2) = 5$.

## The dual function is a lower bound

Fix the prices and let $x$ roam freely, with no rules. The lowest value of the priced score is the *dual function*:

$$
g(\lambda, \nu) = \min_x L(x, \lambda, \nu).
$$

For the bakery, $L(x, \lambda) = x^2 + \lambda - \lambda x$ has slope $2x - \lambda$, which is zero at $x = \lambda/2$. Substituting gives $g(\lambda) = \lambda - \lambda^2/4$.

| Price $\lambda$ | Lowest $x$ | Dual value $g(\lambda)$ |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0.5 | 0.75 |
| 2 | 1 | 1 |
| 3 | 1.5 | 0.75 |
| 4 | 2 | 0 |

Every entry is at most 1, the true best cost. That is *weak duality*, and the argument takes three steps. Take any allowed $x$ and any prices with $\lambda \ge 0$. Then $L(x, \lambda, \nu) \le f(x)$, because $g_i(x) \le 0$ and $h_j(x) = 0$. The dual function is the minimum of $L$ over all $x$, so $g(\lambda, \nu) \le L(x, \lambda, \nu) \le f(x)$ for every allowed $x$. So $g$ is at most the best allowed cost $p^*$.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Left: four curves of x squared plus lambda times one minus x, each with its lowest point below the primal optimum 1. Right: the dual function lambda minus lambda squared over 4, peaking at 1 when lambda is 2.">
<line x1="40.0" y1="210.0" x2="280.0" y2="210.0" class="stroke-muted"/>
<line x1="40.0" y1="163.6" x2="280.0" y2="163.6" class="stroke-ink dashed"/>
<text x="276.0" y="157.6" class="fig-small" text-anchor="end">p* = 1</text>
<text x="40.0" y="22" class="fig-bold">Lagrangian in x, for four prices</text>
<polyline points="40.0,186.8 43.5,188.3 47.0,189.6 50.4,190.8 53.9,191.8 57.4,192.7 60.9,193.4 64.3,193.9 67.8,194.3 71.3,194.5 74.8,194.5 78.3,194.4 81.7,194.2 85.2,193.8 88.7,193.2 92.2,192.4 95.7,191.5 99.1,190.5 102.6,189.3 106.1,187.9 109.6,186.4 113.0,184.7 116.5,182.8 120.0,180.8 123.5,178.6 127.0,176.3 130.4,173.8 133.9,171.2 137.4,168.4 140.9,165.4 144.3,162.3 147.8,159.0 151.3,155.6 154.8,152.0 158.3,148.2 161.7,144.3 165.2,140.2 168.7,136.0 172.2,131.6 175.7,127.0 179.1,122.3 182.6,117.4 186.1,112.4 189.6,107.2 193.0,101.8 196.5,96.3 200.0,90.7 203.5,84.8 207.0,78.8 210.4,72.7 213.9,66.4 217.4,59.9 220.9,53.3 224.3,46.5" fill="none" class="stroke-muted" stroke-width="2"/>
<circle cx="74.3" cy="194.5" r="4.5" class="fill-muted"/>
<polyline points="40.0,140.5 43.5,143.5 47.0,146.4 50.4,149.1 53.9,151.7 57.4,154.1 60.9,156.4 64.3,158.5 67.8,160.5 71.3,162.2 74.8,163.9 78.3,165.3 81.7,166.6 85.2,167.8 88.7,168.8 92.2,169.6 95.7,170.3 99.1,170.8 102.6,171.1 106.1,171.3 109.6,171.4 113.0,171.2 116.5,170.9 120.0,170.5 123.5,169.9 127.0,169.1 130.4,168.2 133.9,167.1 137.4,165.9 140.9,164.5 144.3,162.9 147.8,161.2 151.3,159.4 154.8,157.3 158.3,155.1 161.7,152.8 165.2,150.3 168.7,147.6 172.2,144.8 175.7,141.8 179.1,138.6 182.6,135.3 186.1,131.9 189.6,128.2 193.0,124.5 196.5,120.5 200.0,116.4 203.5,112.2 207.0,107.7 210.4,103.2 213.9,98.4 217.4,93.5 220.9,88.5 224.3,83.3 227.8,77.9 231.3,72.3 234.8,66.7 238.3,60.8 241.7,54.8 245.2,48.6 248.7,42.3" fill="none" class="stroke-a" stroke-width="2"/>
<circle cx="108.6" cy="171.4" r="4.5" class="fill-a"/>
<polyline points="40.0,94.1 43.5,98.7 47.0,103.2 50.4,107.5 53.9,111.6 57.4,115.6 60.9,119.4 64.3,123.1 67.8,126.6 71.3,130.0 74.8,133.2 78.3,136.2 81.7,139.1 85.2,141.8 88.7,144.4 92.2,146.8 95.7,149.0 99.1,151.1 102.6,153.0 106.1,154.7 109.6,156.4 113.0,157.8 116.5,159.1 120.0,160.2 123.5,161.2 127.0,162.0 130.4,162.6 133.9,163.1 137.4,163.4 140.9,163.6 144.3,163.6 147.8,163.5 151.3,163.2 154.8,162.7 158.3,162.1 161.7,161.3 165.2,160.3 168.7,159.2 172.2,158.0 175.7,156.6 179.1,155.0 182.6,153.2 186.1,151.4 189.6,149.3 193.0,147.1 196.5,144.7 200.0,142.2 203.5,139.5 207.0,136.6 210.4,133.6 213.9,130.4 217.4,127.1 220.9,123.6 224.3,120.0 227.8,116.2 231.3,112.2 234.8,108.1 238.3,103.8 241.7,99.4 245.2,94.8 248.7,90.0 252.2,85.1 255.7,80.0 259.1,74.8 262.6,69.4 266.1,63.8 269.6,58.1 273.0,52.2 276.5,46.2 280.0,40.0" fill="none" class="stroke-b" stroke-width="2"/>
<circle cx="142.9" cy="163.6" r="4.5" class="fill-b"/>
<polyline points="40.0,47.7 43.5,53.9 47.0,60.0 50.4,65.8 53.9,71.5 57.4,77.1 60.9,82.5 64.3,87.7 67.8,92.8 71.3,97.7 74.8,102.5 78.3,107.1 81.7,111.5 85.2,115.8 88.7,119.9 92.2,123.9 95.7,127.7 99.1,131.4 102.6,134.8 106.1,138.2 109.6,141.3 113.0,144.4 116.5,147.2 120.0,149.9 123.5,152.4 127.0,154.8 130.4,157.0 133.9,159.1 137.4,161.0 140.9,162.7 144.3,164.3 147.8,165.7 151.3,167.0 154.8,168.1 158.3,169.0 161.7,169.8 165.2,170.4 168.7,170.9 172.2,171.2 175.7,171.3 179.1,171.3 182.6,171.2 186.1,170.8 189.6,170.3 193.0,169.7 196.5,168.9 200.0,167.9 203.5,166.8 207.0,165.5 210.4,164.1 213.9,162.5 217.4,160.7 220.9,158.8 224.3,156.7 227.8,154.5 231.3,152.1 234.8,149.5 238.3,146.8 241.7,143.9 245.2,140.9 248.7,137.7 252.2,134.4 255.7,130.8 259.1,127.2 262.6,123.3 266.1,119.4 269.6,115.2 273.0,110.9 276.5,106.4 280.0,101.8" fill="none" class="stroke-d" stroke-width="2"/>
<circle cx="177.1" cy="171.4" r="4.5" class="fill-d"/>
<text x="74.3" y="212.5" class="fig-small" text-anchor="middle">0.00</text>
<text x="108.6" y="189.4" class="fig-small" text-anchor="middle">0.75</text>
<text x="142.9" y="181.6" class="fig-small" text-anchor="middle">1.00</text>
<text x="177.1" y="189.4" class="fig-small" text-anchor="middle">0.75</text>
<text x="160" y="238.0" class="fig-small" text-anchor="middle">x   (curves for price 0, 1, 2, 3)</text>
<text x="340.0" y="22" class="fig-bold">Dual function g(price)</text>
<line x1="340.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-muted"/>
<line x1="340.0" y1="163.6" x2="570.0" y2="163.6" class="stroke-ink dashed"/>
<polyline points="340.0,194.5 343.9,192.5 347.8,190.5 351.7,188.6 355.6,186.7 359.5,185.0 363.4,183.3 367.3,181.6 371.2,180.1 375.1,178.6 379.0,177.1 382.9,175.8 386.8,174.5 390.7,173.3 394.6,172.2 398.5,171.1 402.4,170.1 406.3,169.2 410.2,168.3 414.1,167.6 418.0,166.8 421.9,166.2 425.8,165.6 429.7,165.1 433.6,164.7 437.5,164.4 441.4,164.1 445.3,163.9 449.2,163.7 453.1,163.6 456.9,163.6 460.8,163.7 464.7,163.9 468.6,164.1 472.5,164.4 476.4,164.7 480.3,165.1 484.2,165.6 488.1,166.2 492.0,166.8 495.9,167.6 499.8,168.3 503.7,169.2 507.6,170.1 511.5,171.1 515.4,172.2 519.3,173.3 523.2,174.5 527.1,175.8 531.0,177.1 534.9,178.6 538.8,180.1 542.7,181.6 546.6,183.3 550.5,185.0 554.4,186.7 558.3,188.6 562.2,190.5 566.1,192.5 570.0,194.5" fill="none" class="stroke-c" stroke-width="2"/>
<circle cx="340.0" cy="194.5" r="4.5" class="fill-muted"/>
<circle cx="397.5" cy="171.4" r="4.5" class="fill-a"/>
<circle cx="455.0" cy="163.6" r="4.5" class="fill-b"/>
<circle cx="512.5" cy="171.4" r="4.5" class="fill-d"/>
<text x="340.0" y="226.0" class="fig-small" text-anchor="middle">0</text>
<text x="397.5" y="226.0" class="fig-small" text-anchor="middle">1</text>
<text x="455.0" y="226.0" class="fig-small" text-anchor="middle">2</text>
<text x="512.5" y="226.0" class="fig-small" text-anchor="middle">3</text>
<text x="570.0" y="226.0" class="fig-small" text-anchor="middle">4</text>
<text x="455.0" y="155.6" class="fig-small" text-anchor="middle">gap 0 at price 2</text>
<text x="455" y="242.0" class="fig-small" text-anchor="middle">price lambda</text>
</svg>
Figure 2. Left: the priced score x² + λ(1 − x) for four prices. Each curve's lowest point (a dot) sits at or below p* = 1. Right: those heights, plotted against the price, trace the dual function, which peaks at exactly p* when the price is 2.
```

Each price gives a curve with a bottom that is a lower bound, and you pick the price that raises the bottom as high as it goes. That search is the *dual problem*: maximise $g(\lambda, \nu)$ over $\lambda \ge 0$ and all $\nu$. Its best value $d^*$ satisfies $d^* \le p^*$, and the difference $p^* - d^*$ is the *duality gap*. The bakery's dual peaks at $\lambda = 2$, where $g = 1 = p^*$, so the gap is zero. The dual function is a minimum of straight lines in the prices, so it is concave, and the dual problem is easy to solve.

> [!KEY]
> For any prices with $\lambda \ge 0$, the dual function is a lower bound on the best score. Any allowed plan is an upper bound. The best score sits between them, and the gap between your two numbers measures how much room is left.

## When the bound is tight: strong duality and Slater's condition

*Strong duality* is the statement that the gap is zero, $d^* = p^*$. It is not automatic: on a problem that is not convex, the gap can stay positive however you choose the prices. For a convex problem, one mild condition is enough. *Slater's condition* says some point is strictly inside every inequality rule, so $g_i(x) < 0$ for all $i$, while the equality rules hold. The bakery has one: $x = 2$ gives $g = -1$. Under Slater's condition a convex problem has $d^* = p^*$, and some prices reach the bound. (A rule that is itself affine only needs an allowed point, not a strictly allowed one.)

Slater's condition can fail on a convex problem. Minimise $x$ subject to $x^2 \le 0$. The only allowed point is $x = 0$, so $p^* = 0$, and no point is strictly inside. The dual function is $g(\lambda) = \min_x (x + \lambda x^2) = -1/(4\lambda)$. Larger prices get closer to 0 but never reach it, so no price achieves the bound.

## The KKT conditions: a certificate of optimality

Suppose a friend hands you a plan $x$ and prices $\lambda$ and $\nu$ and claims the plan is the best. You can check the claim without solving anything, using four conditions named after Karush, Kuhn, and Tucker. The last column fills in the bakery's answer $x = 1$, $\lambda = 2$.

| Condition | What it says | Bakery check |
|---|---|---|
| Primal feasibility | $g_i(x) \le 0$ and $h_j(x) = 0$ | $1 - 1 = 0 \le 0$ |
| Dual feasibility | $\lambda_i \ge 0$ | $2 \ge 0$ |
| Complementary slackness | $\lambda_i\, g_i(x) = 0$ for each $i$ | $2 \cdot 0 = 0$ |
| Stationarity | the slope of $L$ in $x$ is zero | $2x - \lambda = 0$ |

Read the third row aloud: a price and its rule can't both be loose. A rule with slack must have a zero price, and a positive price means the rule is tight. The SVM chapter uses this row to sort houses into groups.

Four conditions certify the answer because $L(\cdot, \lambda, \nu)$ is convex in $x$, and stationarity says its slope is zero at your $x$, so your $x$ is its global minimum. Then $g(\lambda, \nu) = L(x, \lambda, \nu)$. Feasibility and complementary slackness make every priced term zero, so $L(x, \lambda, \nu) = f(x)$. The lower bound equals the cost of an allowed plan, so the gap is zero.

> [!KEY]
> For a convex problem, a plan and prices that satisfy the four KKT conditions are optimal. When Slater's condition holds, the reverse is true too: every optimal plan has prices that satisfy them. The conditions are a checklist you can run on any answer, however you found it.

## Three worked problems

Each problem follows the same steps: write the Lagrangian, minimise over $x$ to get the dual function, then maximise over the prices.

### A small linear program

A linear program minimises a straight-line score over straight-line rules. Take $c = (2, 3)$ with the rules $x_1 + x_2 \ge 4$, $x_1 + 3x_2 \ge 6$, and $x \ge 0$. The corner $(3, 1)$ meets both rules with equality and costs 9. The corners $(6, 0)$ and $(0, 4)$ cost 12 each.

Give the two rules prices $y_1, y_2 \ge 0$ and the bound $x \ge 0$ a price $s \ge 0$. The Lagrangian $c^\top x + y^\top(b - Ax) - s^\top x$ regroups as $b^\top y + (c - A^\top y - s)^\top x$. If any entry of $c - A^\top y - s$ were nonzero, you could push $x$ to infinity and send $L$ to $-\infty$. So the dual function is $b^\top y$ when $A^\top y \le c$ and $-\infty$ otherwise, and the dual problem is another linear program:

$$
\max\; b^\top y \quad\text{subject to}\quad A^\top y \le c,\; y \ge 0.
$$

Here you maximise $4y_1 + 6y_2$ subject to $y_1 + y_2 \le 2$ and $y_1 + 3y_2 \le 3$. The prices $y = (2, 0)$ are allowed and give the bound 8. The prices $y = (1.5, 0.5)$ give $6 + 3 = 9$, which equals the cost of $(3, 1)$, so both are optimal. The KKT conditions agree: both rules are tight at $(3, 1)$, and both entries of $x$ are positive, so both dual constraints must be tight, and they are.

### A quadratic with equality rules

Take $\min\; \tfrac12 x^\top Q x + c^\top x$ subject to $Ax = b$, with $Q$ positive definite. The Lagrangian is $\tfrac12 x^\top Q x + c^\top x + \nu^\top(Ax - b)$, with slope $Qx + c + A^\top\nu$, which is zero at $x(\nu) = -Q^{-1}(c + A^\top\nu)$. Substituting gives the dual function:

$$
g(\nu) = -\tfrac12\,(c + A^\top\nu)^\top Q^{-1}(c + A^\top\nu) - b^\top\nu.
$$

Read it aloud: minus half a squared length, measured with $Q^{-1}$, minus the price times $b$. It is a concave quadratic in $\nu$, so its maximum is a linear solve: $A Q^{-1}A^\top\, \nu^* = -(b + A Q^{-1} c)$.

Take the smallest case: minimise $\tfrac12(x_1^2 + x_2^2)$ subject to $x_1 + x_2 = 1$. Here $Q = I$, $c = 0$, $A = [1\;\,1]$, $b = 1$, so $g(\nu) = -\nu^2 - \nu$. Prices of $-1.5, -1, -0.5, 0$ give $-0.75, 0, 0.25, 0$, so the best is $\nu^* = -0.5$ with $g = 0.25$. The plan $x(\nu^*) = (0.5, 0.5)$ costs $\tfrac12(0.25 + 0.25) = 0.25$. The bound and the cost meet.

### Ridge regression and its dual

Ridge minimises $\tfrac12\lVert y - Xw\rVert^2 + \tfrac12\lambda\lVert w\rVert^2$. Name the residual to get a constrained problem: minimise $\tfrac12\lVert r\rVert^2 + \tfrac12\lambda\lVert w\rVert^2$ subject to $r = y - Xw$. From here on, $\lambda$ is the ridge fee from the earlier chapters, and the price on the equality rule is $\alpha$, one entry per house:

$$
L = \tfrac12\lVert r\rVert^2 + \tfrac12\lambda\lVert w\rVert^2 + \alpha^\top(y - Xw - r).
$$

The slope in $r$ is $r - \alpha$, so $r = \alpha$: the price on each house is its residual. The slope in $w$ is $\lambda w - X^\top\alpha$, so $w = X^\top\alpha/\lambda$. Substituting both gives $g(\alpha) = \alpha^\top y - \tfrac12\alpha^\top(I + XX^\top/\lambda)\alpha$. Set its slope to zero, write $K = XX^\top$ and $a = \alpha/\lambda$, and you get $a = (K + \lambda I)^{-1}y$ and $w = X^\top a$. Read it aloud: the ridge weights are the rows of $X$ added up, each scaled by a number $a_i$ from an $n$ by $n$ solve.

```python
K = X @ X.T
a = np.linalg.solve(K + lam * np.eye(len(y)), y)
w = X.T @ a
```

Check it on two houses: $X = [[1], [2]]$, $y = [1, 2]$, $\lambda = 1$. Then $K + I = [[2, 2], [2, 5]]$ has determinant 6, the solve gives $a = (1/6,\; 1/3)$, and $w = \tfrac16 + \tfrac23 = \tfrac56$. The primal formula $x^\top y/(x^\top x + \lambda) = 5/6$ agrees, and the residual $(1/6, 1/3)$ equals $a$, as $r = \alpha$ says when $\lambda = 1$.

The dual solves a problem of size $n$ (houses) and the primal one of size $d$ (features), so the dual is cheaper when $d \gg n$. Because $K$ holds only dot products between houses, you can swap in any kernel matrix, and that swap is [kernel ridge regression](/lessons/022-kernel-ridge-and-random-features).

## The gap as a stopping test: the lasso

The lasso minimises $P(w) = \tfrac12\lVert Xw - y\rVert^2 + \lambda\lVert w\rVert_1$, and [FISTA](/lessons/007-lasso-and-fista) walks $w$ toward the answer. When should it stop?

Name the residual as with ridge: minimise $\tfrac12\lVert r\rVert^2 + \lambda\lVert w\rVert_1$ subject to $r = y - Xw$, with a price $\theta$ on the equality. Minimising over $r$ gives $r = \theta$. The pieces in $w$ are $\lambda\lVert w\rVert_1 - (X^\top\theta)^\top w$, which is zero when every entry of $X^\top\theta$ is at most $\lambda$ in size and falls to $-\infty$ otherwise. So the dual function is

$$
D(\theta) = \tfrac12\lVert y\rVert^2 - \tfrac12\lVert y - \theta\rVert^2 \quad\text{for}\quad \lVert X^\top\theta\rVert_\infty \le \lambda,
$$

where $\lVert\cdot\rVert_\infty$ is the largest absolute entry. Weak duality says $D(\theta) \le P(w^*)$ for every $\theta$ in that set.

You need a feasible $\theta$ from the current weights. The natural guess is the residual $r = y - Xw$, because at the answer the price equals the residual. Along the way, though, $X^\top r$ can exceed $\lambda$, so $r$ may be outside the allowed set, and then $D(r)$ is not a lower bound. Shrink it into the set:

$$
\theta = r \cdot \min\!\left(1,\; \frac{\lambda}{\lVert X^\top r\rVert_\infty}\right), \qquad \text{gap}(w) = P(w) - D(\theta).
$$

Read it aloud: shrink the residual until its largest correlation with a column is $\lambda$, then measure how far $P$ sits above the bound. At the lasso answer the KKT rule gives $\lVert X^\top r\rVert_\infty \le \lambda$, so $\theta = r$ and the gap is zero.

Try $X = [[1], [2]]$, $y = [1, 2]$, $\lambda = 1$, whose answer is $w = 0.8$.

| Weight $w$ | $P(w)$ | Scale | Dual value | Gap | Gap without rescaling |
|---|---|---|---|---|---|
| 0 | 2.5 | 0.2 | 0.9 | 1.6 | 0 |
| 0.4 | 1.3 | 0.333 | 0.9 | 0.4 | -0.8 |
| 0.8 | 0.9 | 1 | 0.9 | 0 | 0 |

At $w = 0.4$ the residual is $(0.6, 1.2)$, so $X^\top r = 3$ and the scale is $1/3$. Skip the rescaling and you get $-0.8$, an impossible gap. At $w = 0$ the unscaled gap is 0, which falsely claims the answer is found.

> [!WARNING]
> Don't plug the raw residual into $D$. It is feasible only when $\lVert X^\top r\rVert_\infty \le \lambda$, which away from the answer it often isn't.

The gap is an honest stopping rule. Since $D(\theta) \le P(w^*)$, we have $P(w) - P(w^*) \le \text{gap}(w)$, so a gap under $10^{-3}$ proves the objective is within $10^{-3}$ of the best, and you never needed the best value to know it. Figure 3 shows FISTA on 100 houses and 20 features, 4 of which matter, with $\lambda = 10$. The gap starts near 830, falls to 7.4 by step 10 and 0.47 by step 20, and crosses $10^{-3}$ at step 41. The true suboptimality, found with a long run you wouldn't have in practice, stays below it. The gap costs one product with $X^\top$.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Duality gap and true suboptimality of the lasso objective over 60 FISTA steps on a log scale; the gap stays above the suboptimality and both fall.">
<text x="60.0" y="22" class="fig-bold">Lasso duality gap over 60 FISTA steps (log scale)</text>
<line x1="60.0" y1="210.0" x2="560.0" y2="210.0" class="stroke-grid"/>
<text x="54.0" y="214.0" class="fig-small" text-anchor="end">1e-6</text>
<line x1="60.0" y1="153.3" x2="560.0" y2="153.3" class="stroke-grid"/>
<text x="54.0" y="157.3" class="fig-small" text-anchor="end">1e-3</text>
<line x1="60.0" y1="96.7" x2="560.0" y2="96.7" class="stroke-grid"/>
<text x="54.0" y="100.7" class="fig-small" text-anchor="end">1e0</text>
<line x1="60.0" y1="40.0" x2="560.0" y2="40.0" class="stroke-grid"/>
<text x="54.0" y="44.0" class="fig-small" text-anchor="end">1e3</text>
<line x1="60.0" y1="210.0" x2="560.0" y2="210.0" class="stroke-muted"/>
<text x="60.0" y="226.0" class="fig-small" text-anchor="middle">0</text>
<text x="143.3" y="226.0" class="fig-small" text-anchor="middle">10</text>
<text x="226.7" y="226.0" class="fig-small" text-anchor="middle">20</text>
<text x="310.0" y="226.0" class="fig-small" text-anchor="middle">30</text>
<text x="393.3" y="226.0" class="fig-small" text-anchor="middle">40</text>
<text x="476.7" y="226.0" class="fig-small" text-anchor="middle">50</text>
<text x="560.0" y="226.0" class="fig-small" text-anchor="middle">60</text>
<text x="310" y="244.0" class="fig-small" text-anchor="middle">FISTA step</text>
<polyline points="60.0,41.6 68.3,52.6 76.7,58.3 85.0,64.6 93.3,67.5 101.7,69.4 110.0,74.3 118.3,71.2 126.7,72.6 135.0,76.7 143.3,80.3 151.7,88.5 160.0,88.9 168.3,88.4 176.7,91.7 185.0,93.4 193.3,98.9 201.7,102.1 210.0,98.6 218.3,98.9 226.7,102.9 235.0,113.1 243.3,116.2 251.7,113.0 260.0,112.3 268.3,115.5 276.7,124.1 285.0,124.9 293.3,120.0 301.7,120.7 310.0,124.9 318.3,136.2 326.7,138.5 335.0,132.9 343.3,133.4 351.7,137.6 360.0,148.8 368.3,145.5 376.7,140.8 385.0,141.4 393.3,145.6 401.7,156.7 410.0,158.1 418.3,153.5 426.7,154.3 435.0,158.5 443.3,170.1 451.7,166.2 460.0,161.1 468.3,161.5 476.7,165.4 485.0,175.6 493.3,178.9 501.7,173.9 510.0,174.4 518.3,178.2 526.7,188.5 535.0,186.9 543.3,180.8 551.7,181.0 560.0,184.5" fill="none" class="stroke-b" stroke-width="2"/>
<polyline points="60.0,41.8 68.3,54.0 76.7,61.1 85.0,69.1 93.3,79.4 101.7,94.6 110.0,118.8 118.3,108.8 126.7,108.8 135.0,115.3 143.3,128.0 151.7,145.8 160.0,140.0 168.3,138.3 176.7,142.8 185.0,153.2 193.3,171.3 201.7,167.7 210.0,163.8 218.3,166.9 226.7,176.3 235.0,196.7 243.3,193.2 251.7,187.0 260.0,189.3 268.3,198.3 276.7,210.0 285.0,210.0 293.3,208.9 301.7,210.0 310.0,210.0 318.3,210.0 326.7,210.0 335.0,210.0 343.3,210.0 351.7,210.0 360.0,210.0 368.3,210.0 376.7,210.0 385.0,210.0 393.3,210.0 401.7,210.0 410.0,210.0 418.3,210.0 426.7,210.0 435.0,210.0 443.3,210.0 451.7,210.0 460.0,210.0 468.3,210.0 476.7,210.0 485.0,210.0 493.3,210.0 501.7,210.0 510.0,210.0 518.3,210.0 526.7,210.0 535.0,210.0 543.3,210.0 551.7,210.0 560.0,210.0" fill="none" class="stroke-a" stroke-width="2"/>
<text x="343.3" y="125.4" class="fig-small">gap (computable)</text>
<text x="426.7" y="228.0" class="fig-small">true suboptimality (unknown in practice)</text>
</svg>
Figure 3. The lasso duality gap over 60 FISTA steps, on a log scale. The gap needs only the weights and the data, and it stays above the true suboptimality, so a small gap proves the weights are close to the best. The suboptimality line is cut off at 1e-6, the smallest value drawn.
```

> [!KEY]
> Duality turns "the loss stopped moving" into a proof. A feasible dual point is a lower bound, the weights are an upper bound, and the gap says how much improvement is left.

> [!RECAP]
> - A convex score over a convex allowed set has no shallow dips: every local minimum is a global one.
> - The Lagrangian adds a price times each broken rule, with $\lambda \ge 0$ on inequalities and any sign on equalities. Minimising it over $x$ gives the dual function, a lower bound on the best score (weak duality).
> - Under Slater's condition (a strictly allowed point) a convex problem has a gap of zero, and the four KKT conditions certify an optimal plan and prices.
> - The LP dual, the equality-constrained quadratic dual, and the ridge dual $a = (K + \lambda I)^{-1}y$ all come from the same three steps.
> - For the lasso, rescale the residual into $\lVert X^\top\theta\rVert_\infty \le \lambda$ and the gap $P(w) - D(\theta)$ is a non-negative stopping test.

## Build it step by step

The exercise has six functions.

1. **`lagrangian(x, lam, nu, f, gs, hs)`** takes a point, the inequality prices, the equality prices, the score `f`, the inequality functions `gs` (at most zero), and the equality functions `hs` (equal to zero). It returns the priced score. The usual mistake is a flipped rule sign.
2. **`kkt_residuals(Q, c, G, h, x, lam)`** takes $\min\tfrac12 x^\top Qx + c^\top x$ subject to $Gx \le h$, a candidate `x`, and prices `lam`. It returns a dict of four breaches: `stationarity` is $\lVert Qx + c + G^\top\lambda\rVert_\infty$, `primal` is the largest positive entry of $Gx - h$ (0 if none), `dual` is the largest positive entry of $-\lambda$, and `complementary` is the largest $|\lambda_i(Gx - h)_i|$. People forget the absolute value on the last one.
3. **`qp_dual_function(Q, c, A, b, nu)`** returns $g(\nu)$. Use `np.linalg.solve` for $Q^{-1}(c + A^\top\nu)$.
4. **`qp_dual(Q, c, A, b)`** returns `(nu_star, d_star, x_star)`: the best price, the best bound, and the plan $-Q^{-1}(c + A^\top\nu^*)$. The slip here is the sign of $b + AQ^{-1}c$.
5. **`ridge_dual(X, y, lam)`** returns `(a, w)` with $a = (K + \lambda I)^{-1}y$ and $w = X^\top a$. Forgetting the $\lambda I$ gives a singular matrix whenever there are more houses than features.
6. **`lasso_duality_gap(X, y, w, lam)`** returns the gap as a float: residual, scale $\min(1, \lambda/\lVert X^\top r\rVert_\infty)$, dual value, primal value. Guard against dividing by zero when $X^\top r = 0$.

## Where this breaks

The bound is tight only for convex problems that meet Slater's condition, and the $x^2 \le 0$ example shows what failure looks like. A small gap also says the objective is close, not that the weights are: with strongly correlated columns, different weights can score almost the same, so check the support of $w$ as well.

## The exercise

You implement `lagrangian`, `kkt_residuals`, `qp_dual_function`, `qp_dual`, `ridge_dual`, and `lasso_duality_gap`. The checks confirm the bakery's priced scores (1 and 5), that minimising `lagrangian` over a grid reproduces $\lambda - \lambda^2/4$, and that `kkt_residuals` is zero at the answer and positive for a wrong point or a negative price. They confirm that `qp_dual` returns $\nu^* = -0.5$ and $d^* = 0.25$, matches an independent primal optimum on a random problem, and bounds it from below at 50 random prices. `ridge_dual` must give $w = 5/6$ and match the primal ridge solution on a wide design. `lasso_duality_gap` must give 1.6, 0.4, and 0 on the table's three weights, stay non-negative at random weights, be near zero at a long-run answer, and shrink over a FISTA run. The $-0.8$ at $w = 0.4$ catches a gap that skips the rescaling.

For the Lagrangian applied to a real model, continue with the section "From road to pushes: the Lagrangian" in [support vector machines](/lessons/020-support-vector-machines), which uses these tools to derive the SVM dual.

## Further reading

- Boyd and Vandenberghe, [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/), chapter 5, for weak and strong duality, Slater's condition, and the KKT conditions in full generality. The authors host the book free online.
- [Stanford EE364A](https://see.stanford.edu/Course/EE364A), the convex optimisation course taught from that book, for the lecture on the Lagrangian, the dual function, and Slater's condition (lecture 8 on the course page's list).
- [Stanford CS229](https://cs229.stanford.edu/), the machine-learning course whose notes derive the SVM dual from the same Lagrangian. The course home page prints no lecture numbers.
