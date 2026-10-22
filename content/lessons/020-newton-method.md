---
title: Curvature and Newton's method
slug: 020-newton-method
part: "Part 1 — Foundations"
summary: "Each row bends the loss by p(1 − p), and dividing the gradient by that bend gives Newton steps that finish in a handful of passes."
minutes: 40
gpu: false
objectives:
  - Explain curvature as the slope of the sigmoid, and why confident rows barely bend the loss.
  - Build the Hessian X^T D X + λI, and check it against finite differences of the gradient.
  - Take a Newton step by solving H δ = g, and explain it as a parabola fitted at the current weights.
  - Say what Newton costs per step, and what happens on separable data and on duplicate columns.
prereqs:
  - 010-how-big-a-step
  - 018-log-loss
  - 019-gradient-descent-for-logistic
exercise: 020-newton-method
---

# Curvature and Newton's method

Gradient descent fit the 60-email inbox in 185 passes, and only because you picked a step size of 0.02. At 0.05 the loss bounced forever, and at 0.001 it crawled. Every one of those numbers came from you, and the loss itself said nothing about them. Could the loss tell you how far to go?

It can, because the loss carries more than a slope. It also carries a bend: how fast the slope itself changes. A steep slope on a road that barely bends deserves a long stride, and the same slope on a sharp curve deserves a short one. Newton's method measures the bend at every step and divides it out.

This chapter shows where the bend comes from in logistic regression, builds it into a matrix, and uses it to fit the same inbox in 7 passes.

## What you already know

- The gradient is $X^\top(p - y) + \lambda w$, and a step moves the weights against it (see Learning the weights: gradient and descent).
- A step size above $2/L$, where $L$ is the largest curvature, makes the loss rise (see How big a step: curvature and the 2/L limit).
- The sigmoid $\sigma(z)$ turns a score into a probability $p$ (see From a score to a probability).
- The loss is a sum of one term per row (see How wrong is a probability? The log loss).
- `np.linalg.solve(A, b)` finds $x$ with $Ax = b$.

## Feel the bend

Look at one row. The slope of its loss with respect to its score $z$ is $p - y$, from the last chapter. How fast does that slope change as $z$ changes? Only $p$ moves, and the sigmoid's own slope is $\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr) = p(1-p)$. So the row's bend is $p(1 - p)$.

This number has a shape you can read. It is largest, $0.5 \times 0.5 = 0.25$, when $p = 0.5$, and it falls towards 0 as $p$ nears 0 or 1. A confident row barely bends the loss, because moving its score a little changes its probability a little. An undecided row bends the loss the most. For the two emails, A has $p = 0.6225$ and a bend of $0.6225 \times 0.3775 = 0.2350$, and B has $p = 0.1824$ and a bend of $0.1824 \times 0.8176 = 0.1491$.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="The curvature weight p(1-p) against p, with its peak of 0.25 at p = 0.5 and the two lesson emails marked, next to bars showing how it shrinks as the weights grow on the separable emails">
<line x1="60.0" y1="250.0" x2="330.0" y2="250.0" class="stroke-grid" />
<text x="52.0" y="254.0" class="fig-small" text-anchor="end">0</text>
<line x1="60.0" y1="175.0" x2="330.0" y2="175.0" class="stroke-grid" />
<text x="52.0" y="179.0" class="fig-small" text-anchor="end">0.1</text>
<line x1="60.0" y1="100.0" x2="330.0" y2="100.0" class="stroke-grid" />
<text x="52.0" y="104.0" class="fig-small" text-anchor="end">0.2</text>
<line x1="60.0" y1="250.0" x2="330.0" y2="250.0" class="stroke-muted" />
<line x1="60.0" y1="250.0" x2="60.0" y2="255.0" class="stroke-muted" />
<text x="60.0" y="270.0" class="fig-small" text-anchor="middle">0</text>
<line x1="195.0" y1="250.0" x2="195.0" y2="255.0" class="stroke-muted" />
<text x="195.0" y="270.0" class="fig-small" text-anchor="middle">0.5</text>
<line x1="330.0" y1="250.0" x2="330.0" y2="255.0" class="stroke-muted" />
<text x="330.0" y="270.0" class="fig-small" text-anchor="middle">1</text>
<polyline points="60.0,250.0 62.7,242.6 65.4,235.3 68.1,228.2 70.8,221.2 73.5,214.4 76.2,207.7 78.9,201.2 81.6,194.8 84.3,188.6 87.0,182.5 89.7,176.6 92.4,170.8 95.1,165.2 97.8,159.7 100.5,154.4 103.2,149.2 105.9,144.2 108.6,139.3 111.3,134.6 114.0,130.0 116.7,125.6 119.4,121.3 122.1,117.2 124.8,113.2 127.5,109.4 130.2,105.7 132.9,102.2 135.6,98.8 138.3,95.6 141.0,92.5 143.7,89.6 146.4,86.8 149.1,84.2 151.8,81.7 154.5,79.4 157.2,77.2 159.9,75.2 162.6,73.3 165.3,71.6 168.0,70.0 170.7,68.6 173.4,67.3 176.1,66.2 178.8,65.2 181.5,64.4 184.2,63.7 186.9,63.2 189.6,62.8 192.3,62.6 195.0,62.5 197.7,62.6 200.4,62.8 203.1,63.2 205.8,63.7 208.5,64.4 211.2,65.2 213.9,66.2 216.6,67.3 219.3,68.6 222.0,70.0 224.7,71.6 227.4,73.3 230.1,75.2 232.8,77.2 235.5,79.4 238.2,81.7 240.9,84.2 243.6,86.8 246.3,89.6 249.0,92.5 251.7,95.6 254.4,98.8 257.1,102.2 259.8,105.7 262.5,109.4 265.2,113.2 267.9,117.2 270.6,121.3 273.3,125.6 276.0,130.0 278.7,134.6 281.4,139.3 284.1,144.2 286.8,149.2 289.5,154.4 292.2,159.7 294.9,165.2 297.6,170.8 300.3,176.6 303.0,182.5 305.7,188.6 308.4,194.8 311.1,201.2 313.8,207.7 316.5,214.4 319.2,221.2 321.9,228.2 324.6,235.3 327.3,242.6 330.0,250.0" class="stroke-b" fill="none"/>
<circle cx="195.0" cy="62.5" r="5" class="fill-b"/>
<text x="195.0" y="52.5" class="fig-small" text-anchor="middle">peak 0.25</text>
<circle cx="228.1" cy="73.8" r="5" class="fill-a"/>
<text x="238.1" y="63.8" class="fig-small" text-anchor="start">A: 0.2350</text>
<circle cx="109.2" cy="138.2" r="5" class="fill-a"/>
<text x="105.2" y="126.2" class="fig-small" text-anchor="end">B: 0.1491</text>
<text x="195.0" y="290.0" text-anchor="middle">probability p</text>
<text x="24.0" y="26.0" text-anchor="start">curvature weight p(1 − p)</text>
<text x="370.0" y="26.0" text-anchor="start">Weights of rows A and B as t grows</text>
<rect x="370.0" y="80.0" width="18.0" height="170.0" class="fill-a" />
<rect x="392.0" y="80.0" width="18.0" height="170.0" class="fill-d" />
<text x="390.0" y="270.0" class="fig-small" text-anchor="middle">t = 0</text>
<text x="390.0" y="72.0" class="fig-small" text-anchor="middle">0.250</text>
<rect x="440.0" y="116.3" width="18.0" height="133.7" class="fill-a" />
<rect x="462.0" y="116.3" width="18.0" height="133.7" class="fill-d" />
<text x="460.0" y="270.0" class="fig-small" text-anchor="middle">t = 1</text>
<text x="460.0" y="108.3" class="fig-small" text-anchor="middle">0.197</text>
<rect x="510.0" y="245.5" width="18.0" height="4.5" class="fill-a" />
<rect x="532.0" y="245.5" width="18.0" height="4.5" class="fill-d" />
<text x="530.0" y="270.0" class="fig-small" text-anchor="middle">t = 5</text>
<text x="530.0" y="237.5" class="fig-small" text-anchor="middle">0.007</text>
<line x1="360.0" y1="250.0" x2="580.0" y2="250.0" class="stroke-muted" />
<rect x="370.0" y="290.0" width="10.0" height="10.0" class="fill-a" />
<text x="386.0" y="299.0" class="fig-small" text-anchor="start">row A</text>
<rect x="440.0" y="290.0" width="10.0" height="10.0" class="fill-d" />
<text x="456.0" y="299.0" class="fig-small" text-anchor="start">row B</text>
</svg>
Figure 1. The bend of a row is highest at p = 0.5 and shrinks as the row gets sure. Notice in the bars how the two emails lose their bend as the weights along (1, −1) grow: at t = 5 each weighs only 0.007.
```

Weights, not scores, are what you move, so collect the row bends into a matrix. The *Hessian* is the table of second derivatives: its entry in row $j$, column $k$ says how the slope for weight $j$ changes when weight $k$ moves. A row with features $x$ has slope $(p - y)\,x_j$ for weight $j$. When weight $k$ moves by 1 the score moves by $x_k$, so $p$ changes by $p(1-p)\,x_k$, and the slope for weight $j$ changes by $p(1-p)\,x_j x_k$. That is the row's contribution to the entry. Adding over rows, and adding $\lambda$ on the diagonal for the ridge fee:

$$
H = X^{\top} D X + \lambda I, \qquad D = \operatorname{diag}\bigl(p_1(1-p_1),\ \ldots,\ p_n(1-p_n)\bigr)
$$

Read it aloud: scale each row of $X$ by its bend, multiply by $X^\top$, and add $\lambda$ down the diagonal. For the emails at weights $(0.5, -1)$ the entries are $0.2350 + 0.1491 = 0.3842$, then $2 \times 0.1491 = 0.2983$, then $4 \times 0.1491 = 0.5966$. In numpy, `X * s[:, None]` scales every row without building the $n \times n$ matrix $D$:

```python
s = p * (1.0 - p)
H = (X * s[:, None]).T @ X + ridge * np.eye(d)
```

Every bend $p(1-p)$ is at least 0, so no direction bends downward (a Hessian with this property is called positive semidefinite). That makes the loss a bowl, which means a zero gradient is a bottom and not a trap. Two honest limits apply: the bowl can be flat in a direction that the data cannot tell apart, and for perfectly separable data the bottom sits infinitely far away, as the last chapter showed.

## Fit a parabola and jump

Use the bend. Stand at one weight and know three numbers: the loss, the slope, and the bend. A parabola with those three numbers has an easy bottom, at the current weight minus slope divided by bend. Jump there.

Figure 2 draws this for the replies weight of the inbox, with the bias held fixed at its best value, 0.46. At a weight of 0 the loss is 49.2, the slope is 66.1, and the bend is 115. So the jump is $66.1 / 115 = 0.57$ to the left, landing at $-0.57$ with loss 29.6. A gradient step of 0.005 moves only to $-0.33$ with loss 33.8. The true minimum is at $-0.76$.

```figure
<svg viewBox="0 0 600 360" role="img" aria-label="Newton in one dimension: a parabola matched to the loss at the current weight, with an arrow to its bottom, next to a gradient step that lands short">
<defs><marker id="c020-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
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
<line x1="458.9" y1="258.8" x2="303.8" y2="258.8" class="stroke-ink" marker-end="url(#c020-arrow)"/>
<text x="295.8" y="250.8" class="fig-small" text-anchor="start">Newton jump: to -0.57</text>
<line x1="458.9" y1="230.2" x2="371.1" y2="230.2" class="stroke-ink" marker-end="url(#c020-arrow)"/>
<text x="363.1" y="222.2" class="fig-small" text-anchor="start">gradient step (lr 0.005): to -0.33</text>
<line x1="299.8" y1="194.9" x2="299.8" y2="290.0" class="stroke-grid dashed" />
<text x="208.9" y="45.6" class="fig-small" text-anchor="start">dashed: parabola with the same</text>
<text x="208.9" y="61.6" class="fig-small" text-anchor="start">height, slope, and bend</text>
</svg>
Figure 2. The dashed parabola shares the solid curve's height, slope, and bend at the start. Notice that the Newton jump goes to the parabola's bottom and lands near the true minimum, and that the real curve flattens faster than the parabola, so one jump does not land exactly.
```

Read the jump as gradient descent with a step size of $1/\text{bend}$ chosen for you at every point. Where the loss is flat the step is long, and where it bends sharply the step is short. This is why a single fixed step size, as in the last chapter, cannot match it.

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

Use `solve` and not `np.linalg.inv(H) @ g`: it is cheaper, and when $H$ is nearly flat in some direction the inverse amplifies rounding noise.

## Watch it converge

The loop is the descent loop with a different step: record the loss, compute the gradient, stop if it is short, build $H$, and subtract $\delta$. On the 60-email inbox it stops after 7 passes, against 185 for gradient descent. The loss above its minimum goes 12.8, 1.01, 0.057, 0.0003, $1.3 \times 10^{-8}$, and then rounding noise. Near the bottom the loss is almost a parabola, so each jump roughly doubles the number of correct digits.

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
Figure 3. The loss above its minimum, on a log scale, for the same inbox. Notice that Newton's seven dots fall off a cliff while gradient descent, which has no sense of the bend, slides down a gentle slope. Values below 1e-13 are drawn at that floor.
```

This is not a guarantee for every start. The parabola is only a local picture, so far from the bottom a full jump can overshoot. From a replies weight of $-1.5$, the bend is 7.1, so the jump is $12.5 / 7.1 = 1.77$ to the right and lands at $+0.27$, where the loss is 70.7, more than twice the 34.6 it started at. Starting from zero weights, as `fit_newton` does, this inbox behaves, but the loss falling every step is a property of this data and not a law.

## Pay for the bend

Few steps do not mean a cheap fit, because each step does more work:

| Piece | Work per step | Memory |
|---|---|---|
| Gradient | about $nd$ | $d$ numbers |
| Hessian | about $nd^2$ | $d^2$ numbers |
| Solve $H\delta = g$ | about $d^3$ | reuses $H$ |

With a handful of weights, seven expensive steps beat hundreds of cheap ones. With 10,000 weights, $H$ holds 100 million numbers and the solve takes about $10^{12}$ operations, so large models use gradient descent, or L-BFGS, which builds a cheap running estimate of the bend from the last few gradients instead of storing $H$.

## Run Newton on a perfect split

The two emails are perfectly separable, so the unpenalised loss has no bottom. Newton still runs, and every step adds about 1 to the length $t$ along $(1, -1)$, because on a loss that decays like $e^{-t}$ the slope and the bend are equal, so slope divided by bend is 1. It stops after 19 passes at weights $(15.3, -15.8)$, not because it found a bottom but because the gradient fell below $10^{-10}$. Giving the loss a ridge fee $\lambda = 1$ puts a bottom back: Newton reaches $(0.176, -0.560)$, with loss 1.110, in 5 passes.

The default `ridge=1e-8` in `fit_newton` does something else. It does not stop growth, since $10^{-8}$ is far too small to act as a leash. It keeps the solve valid. If two columns of $X$ are copies of each other, $H$ has no inverse, and `np.linalg.solve` fails with a singular matrix error. Adding $10^{-8}$ on the diagonal makes it invertible, and the fit finishes in 6 passes.

> [!KEY]
> Each row bends the loss by $p(1-p)$, so confident rows barely bend it and the Hessian $X^\top DX + \lambda I$ is never negative. Newton fits a parabola at the current weights and jumps to its bottom, which is gradient descent with a step size picked for you from the bend. The loss itself therefore tells you how far to go, and the inbox takes 7 passes instead of 185.

> [!RECAP]
> - A row's bend is $p(1-p)$, at most 0.25 and near 0 when the row is sure.
> - The Hessian is $X^\top DX + \lambda I$, with shape `(d, d)`, symmetric, and never bending downward.
> - A Newton step solves $H\delta = g$ and subtracts $\delta$. Use `solve`, not an inverse.
> - A step costs about $nd^2 + d^3$, so Newton suits few weights, and a far-off full step can overshoot.
> - On separable data the unpenalised loss has no bottom. The default `ridge=1e-8` only keeps the solve valid.

## Build it step by step

Build three functions. The sigmoid, softplus, log loss, gradient, and gradient descent from the earlier chapters are provided.

1. `hessian(w, X, y, ridge)` returns shape `(d, d)`. Compute `p = sigmoid(X @ w)`, scale the rows of `X` by `p * (1 - p)`, multiply by `X`, and add `ridge * np.eye(d)`. The common mistake is `X * s` without `[:, None]`, which scales the columns.

2. `newton_step(w, X, y, ridge)` returns $\delta$, the vector to subtract, as `np.linalg.solve(H, g)`. It does not subtract anything itself.

3. `fit_newton(X, y, max_iter, ridge, tol)` starts at zeros and returns the weights and the loss history, with the same recording rule as gradient descent. Its default `ridge` is `1e-8`. The common mistake is to record the loss after the update, so that `history[0]` is not the loss at zero weights.

## Where this breaks

- **Overshoot.** A full step far from the bottom can raise the loss. Production code shrinks the step until the loss falls.
- **A perfect split.** Newton marches off without a bottom. Use a real ridge fee.
- **Many weights.** The cost of $H$ and the solve grows with $d^2$ and $d^3$. Use gradient descent or L-BFGS.

## The exercise

You implement `hessian`, `newton_step`, and `fit_newton`. The checks compare `hessian` with finite differences of `gradient`, with the hand values above, and with the symmetry and positive-semidefinite rules. They confirm that `newton_step` solves $H\delta = g$ and reproduces the step $(-2, 2)$ from zero weights. They run `fit_newton` against gradient descent on the same seeded data, check that it finishes in under ten passes at a stationary point, and test a perfect split with and without a real ridge fee, and a matrix with a duplicate column.
