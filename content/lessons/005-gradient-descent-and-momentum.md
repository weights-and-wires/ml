---
title: "Gradient descent, step size, and momentum"
slug: 005-gradient-descent-and-momentum
part: "Part 1 — Foundations"
summary: Walk to the bottom of a loss by stepping against the slope, find the longest safe step, and use momentum to cross long valleys faster.
minutes: 75
gpu: false
objectives:
  - Compute the loss and gradient of a two-weight bowl, and run gradient descent along it.
  - Explain why a step multiplies the distance to the bottom by 1 minus the step size times the curvature on each axis, and find the longest safe step, 2/L.
  - Compute the condition number L/mu and count the steps a long valley needs, which grow in proportion to kappa.
  - Write the heavy-ball and Nesterov updates, and explain why a remembered step cuts the count from about kappa to about its square root.
prereqs:
  - 003-least-squares-and-projection
  - 004-stable-least-squares
exercise: 005-gradient-descent-and-momentum
---

# Gradient descent, step size, and momentum

You stand on a hillside in thick fog. You cannot see the valley, but you can feel the ground tilt under your feet. If the ground tilts down to your left, you step left, feel the tilt again, and repeat. You never see the bottom, yet you reach it.

Training a model is the same walk. The height of the hill is the loss, and your position is the list of weights. A formula can sometimes jump to the bottom, as it did when you fit a line by minimising squared misses. So why would anyone walk? Because most losses have no such formula: you can score a logistic model or a neural network at any point, and feel its tilt, but nothing says where the bottom is.

Walking raises two questions. How long should each step be? Try 0.3 where 0.1 worked and the walk flies apart. And why can even a safe step be painfully slow? The same safe rule reaches the bottom in 66 steps on one bowl and 688 on another. This chapter builds the walk, finds the longest safe step, explains the slowdown, and then teaches the walk to remember its last step, as a rolling ball does.

## What you already know

- A loss turns a guess into a height, and lower is better.
- The squared-miss loss of a line is a bowl in the weights, and the best weights sit at the bottom.
- The slope of a curve at a point says how fast the height changes when you move right. At the bottom of a smooth bowl the slope is 0.
- `np.linalg.eigvalsh(A)` returns the eigenvalues of a symmetric matrix in ascending order.

## Feel the slope on a curve

Start with one weight. Let the loss be $(w-3)^2$, a bowl whose bottom is at $w=3$. Pretend you cannot see that. You stand at $w=0$, where the height is 9.

Your only tool is the slope, which here is $2(w-3)$. At $w=0$ it is $-6$. A negative slope means the height falls as $w$ grows, so the bottom is to your right. At $w=3.5$ the slope is $+1$, so the height rises to the right and you must go left. Figure 1 draws the slope at three points.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="The curve (w - 3) squared with the slope and the step direction at three points">
<defs><marker id="c005-slope-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<line x1="50.0" y1="230.0" x2="570.0" y2="230.0" class="stroke-muted" />
<line x1="50.0" y1="230.0" x2="50.0" y2="235.0" class="stroke-muted" />
<text x="50.0" y="250.0" class="fig-small" text-anchor="middle">-1</text>
<line x1="115.0" y1="230.0" x2="115.0" y2="235.0" class="stroke-muted" />
<text x="115.0" y="250.0" class="fig-small" text-anchor="middle">0</text>
<line x1="180.0" y1="230.0" x2="180.0" y2="235.0" class="stroke-muted" />
<text x="180.0" y="250.0" class="fig-small" text-anchor="middle">1</text>
<line x1="245.0" y1="230.0" x2="245.0" y2="235.0" class="stroke-muted" />
<text x="245.0" y="250.0" class="fig-small" text-anchor="middle">2</text>
<line x1="310.0" y1="230.0" x2="310.0" y2="235.0" class="stroke-muted" />
<text x="310.0" y="250.0" class="fig-small" text-anchor="middle">3</text>
<line x1="375.0" y1="230.0" x2="375.0" y2="235.0" class="stroke-muted" />
<text x="375.0" y="250.0" class="fig-small" text-anchor="middle">4</text>
<line x1="440.0" y1="230.0" x2="440.0" y2="235.0" class="stroke-muted" />
<text x="440.0" y="250.0" class="fig-small" text-anchor="middle">5</text>
<line x1="505.0" y1="230.0" x2="505.0" y2="235.0" class="stroke-muted" />
<text x="505.0" y="250.0" class="fig-small" text-anchor="middle">6</text>
<line x1="570.0" y1="230.0" x2="570.0" y2="235.0" class="stroke-muted" />
<text x="570.0" y="250.0" class="fig-small" text-anchor="middle">7</text>
<text x="574.0" y="264.0" class="fig-small" text-anchor="start"></text>
<polyline points="50.0,40.0 56.5,49.4 63.0,58.5 69.5,67.4 76.0,76.1 82.5,84.5 89.0,92.7 95.5,100.7 102.0,108.4 108.5,115.9 115.0,123.1 121.5,130.1 128.0,136.9 134.5,143.4 141.0,149.7 147.5,155.8 154.0,161.6 160.5,167.2 167.0,172.5 173.5,177.6 180.0,182.5 186.5,187.1 193.0,191.5 199.5,195.7 206.0,199.6 212.5,203.3 219.0,206.7 225.5,209.9 232.0,212.9 238.5,215.6 245.0,218.1 251.5,220.4 258.0,222.4 264.5,224.2 271.0,225.7 277.5,227.0 284.0,228.1 290.5,228.9 297.0,229.5 303.5,229.9 310.0,230.0 316.5,229.9 323.0,229.5 329.5,228.9 336.0,228.1 342.5,227.0 349.0,225.7 355.5,224.2 362.0,222.4 368.5,220.4 375.0,218.1 381.5,215.6 388.0,212.9 394.5,209.9 401.0,206.7 407.5,203.3 414.0,199.6 420.5,195.7 427.0,191.5 433.5,187.1 440.0,182.5 446.5,177.6 453.0,172.5 459.5,167.2 466.0,161.6 472.5,155.8 479.0,149.7 485.5,143.4 492.0,136.9 498.5,130.1 505.0,123.1 511.5,115.9 518.0,108.4 524.5,100.7 531.0,92.7 537.5,84.5 544.0,76.1 550.5,67.4 557.0,58.5 563.5,49.4 570.0,40.0" fill="none" class="stroke-a" />
<line x1="56.5" y1="59.0" x2="173.5" y2="187.2" class="stroke-muted dashed" />
<circle cx="115.0" cy="123.1" r="5" class="fill-b"/>
<line x1="115.0" y1="272.0" x2="212.5" y2="272.0" class="stroke-b" marker-end="url(#c005-slope-arrow-b)"/>
<circle cx="115.0" cy="272.0" r="3" class="fill-b"/>
<text x="115.0" y="296.0" class="fig-small" text-anchor="middle">w = 0: slope -6</text>
<line x1="154.0" y1="171.2" x2="271.0" y2="235.3" class="stroke-muted dashed" />
<circle cx="212.5" cy="203.3" r="5" class="fill-b"/>
<line x1="212.5" y1="272.0" x2="261.2" y2="272.0" class="stroke-b" marker-end="url(#c005-slope-arrow-b)"/>
<circle cx="212.5" cy="272.0" r="3" class="fill-b"/>
<text x="212.5" y="296.0" class="fig-small" text-anchor="middle">w = 1.5: slope -3</text>
<line x1="284.0" y1="237.7" x2="401.0" y2="216.3" class="stroke-muted dashed" />
<circle cx="342.5" cy="227.0" r="5" class="fill-b"/>
<line x1="342.5" y1="272.0" x2="326.2" y2="272.0" class="stroke-b" marker-end="url(#c005-slope-arrow-b)"/>
<circle cx="342.5" cy="272.0" r="3" class="fill-b"/>
<text x="342.5" y="296.0" class="fig-small" text-anchor="middle">w = 3.5: slope +1</text>
<text x="310.0" y="30.0" class="fig-small" text-anchor="middle">height = (w - 3)²</text>
<text x="586.0" y="250.0" class="fig-small" text-anchor="start">w</text>
<text x="20.0" y="276.0" class="fig-small" text-anchor="start">step</text>
</svg>
Figure 1. Dashed lines are the slope at three points. Where the slope is negative the curve falls to the right, so the orange step (one quarter of the slope, taken backwards, drawn below the axis) points right. Where the slope is positive it points left. Either way the step moves toward the bottom at w = 3.
```

The sign of the slope tells you the direction, and you walk the opposite way. A steep slope means you are far from the bottom, so you step by a fraction of the slope. That fraction is the step size, written $\alpha$ (alpha), and it is also called the learning rate.

Take $\alpha=0.25$ from $w=0$. The slope is $-6$, so you move to $0-0.25\times(-6)=1.5$. The slope there is $-3$, so you move to $2.25$, then $2.625$, then $2.8125$. Each step is half the size of the one before, because the slope shrinks as you approach the bottom. In symbols:

$$
w_{\text{next}} = w - \alpha \cdot \text{slope}(w)
$$

Read it aloud as "move from $w$ against the slope, by a fraction $\alpha$ of it." Add the slope instead of subtracting it and you climb the hill.

On this curve you could set the slope to 0 and solve: $2(w-3)=0$ gives $w=3$. For a bowl with $d$ weights the same idea becomes a linear system that costs about $d^3$ multiplications, while one slope costs about $d^2$. For a loss that is not a bowl, such as logistic loss or a neural network, the slope conditions have no formula at all. Zoom in on any smooth loss near its minimum and it looks like a bowl, so the last part of every walk looks like the walk you learn here.

## A bowl with a steep side and a flat side

Real models have many weights, so give the bowl two, $w_1$ and $w_2$. The trouble starts when one weight matters more than the other, such as a weight on a length in millimetres against one in metres. Say $w_2$ is ten times as sensitive, with the ten as a made-up stand-in. The height is half of $w_1^2+10w_2^2$, minus $w_1+10w_2$. At $(0,0)$ it is 0. At $(1,1)$ it is $0.5\times 11-11=-5.5$. At $(0,1)$ it is $5-10=-5$.

Where is the bottom? Rewrite the height as

$$
f(w) = \tfrac12 (w_1-1)^2 + 5\,(w_2-1)^2 - 5.5
$$

Expanding it gives the original. Each weight has its own one-weight bowl, and the lowest height, $-5.5$, is at $(1,1)$. Call the distance of a coordinate from its bottom value its *miss*. At $(0,0)$ both misses are 1. The $w_2$ bowl is ten times as sharply curved as the $w_1$ bowl, so $w_2$ is the *steep axis* and $w_1$ is the *flat axis*.

The matrix form, which is the shape you code, is

$$
f(w) = \tfrac12\, w^\top A\, w - b^\top w
$$

Here $A$ has 1 and 10 on its diagonal and 0 elsewhere, and $b=(1,10)$. Read it aloud as "half of $w$ times $A$ times $w$, minus $b$ dotted with $w$." The bottom is where $Aw=b$, which is $(1,1)$. In numpy:

```python
loss = 0.5 * (w @ A @ w) - b @ w  # one number
```

Least squares has this shape too, with $A=X^\top X$ and $b=X^\top y$. A realistic $A$ has nonzero off-diagonal entries, which tilts the bowl.

### Read the slope as a vector

With two weights you feel two slopes, one per axis. To feel the slope along $w_2$ at $(0,0)$, compare the heights at $(0,1)$ and $(0,-1)$, which are $-5$ and $15$: the slope is $(-5-15)/2=-10$. Along $w_1$ the heights at $(1,0)$ and $(-1,0)$ are $-0.5$ and $1.5$, so the slope is $-1$.

The two slopes together form the *gradient*, written $\nabla f$. At $(0,0)$ it is $(-1,-10)$. It points uphill, and it is the steepest way up, so stepping against it is the steepest way down. For this bowl it has a formula:

$$
\nabla f(w) = A\,w - b
$$

At $(0,0)$ it gives $-b=(-1,-10)$, which matches the nudges. In numpy:

```python
grad = A @ w - b  # shape (d,)
```


> [!KEY]
> The gradient is the vector of slopes, one per weight. It points uphill, so you step against it, and for the bowl it is $Aw-b$.

> [!TIP]
> To check any gradient, nudge one weight at a time: `(f(w + h * e) - f(w - h * e)) / (2 * h)`. Here `e` is 0 except for a 1 in one entry, and `h` is about `1e-6`. The exercise checks your gradient this way.

## Take downhill steps

Use $\alpha=0.1$ from $(0,0)$ and subtract $0.1$ times the gradient $(-1,-10)$. You land at $(0.1,1.0)$, where the height is $-5.095$. The gradient there is $(-0.9,0)$, so the next step lands at $(0.19,1.0)$ with height $-5.172$, then $(0.271,1.0)$.

The steep coordinate $w_2$ reached its bottom value in one step: its slope is 10 times its miss, and $0.1\times10=1$, so one step covers exactly the miss. The flat coordinate $w_1$ creeps: its miss goes $1,\,0.9,\,0.81,\,0.729$. In symbols, with $w_k$ for the point after $k$ steps:

$$
w_{k+1} = w_k - \alpha\, \nabla f(w_k)
$$

Read it aloud as "the next point is the current point, minus $\alpha$ times the gradient there." This walk is *gradient descent*. In numpy:

```python
w = w - step * grad(w)  # one step, shape (d,)
```

> [!WARNING]
> The exercise stores the whole path, so row 0 must be the start `w0` and the path must have `iters + 1` rows. Work on a copy of `w0`, or you change the caller's array.

## How big a step? One axis at a time

Why not take a huge step and arrive sooner? With 0.3 in place of 0.1, the loss goes from 0 to 14.7 to 74.6 to 314.6, and the walk flies apart. Look at one coordinate. On the steep axis the slope is $10(w_2-1)$, which is 10 times the miss, and a step moves you by $\alpha$ times that slope, backwards. So the next miss is the old miss minus $10\alpha$ times the old miss:

$$
\text{miss}_{\text{next}} = (1-10\,\alpha)\;\text{miss}
$$

Read it aloud as "each step multiplies the miss by one minus ten times the step size." On the flat axis the same argument gives $1-\alpha$. The matrix $A$ is diagonal, so the axes do not mix.

At $\alpha=0.3$ the steep factor is $1-3=-2$, so the misses go $1,\,-2,\,4,\,-8$, while the flat factor $0.7$ shrinks its misses. One axis converges and the other explodes, and the explosion wins. Figure 2 shows four step sizes on the steep axis.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Distance from the bottom along the steep axis over four steps for four step sizes">
<defs></defs>
<line x1="70.0" y1="250.0" x2="330.0" y2="250.0" class="stroke-muted" />
<line x1="70.0" y1="30.0" x2="70.0" y2="250.0" class="stroke-grid" />
<line x1="70.0" y1="140.0" x2="330.0" y2="140.0" class="stroke-muted dashed" />
<line x1="70.0" y1="250.0" x2="70.0" y2="255.0" class="stroke-muted" />
<text x="70.0" y="270.0" class="fig-small" text-anchor="middle">0</text>
<line x1="156.7" y1="250.0" x2="156.7" y2="255.0" class="stroke-muted" />
<text x="156.7" y="270.0" class="fig-small" text-anchor="middle">1</text>
<line x1="243.3" y1="250.0" x2="243.3" y2="255.0" class="stroke-muted" />
<text x="243.3" y="270.0" class="fig-small" text-anchor="middle">2</text>
<line x1="330.0" y1="250.0" x2="330.0" y2="255.0" class="stroke-muted" />
<text x="330.0" y="270.0" class="fig-small" text-anchor="middle">3</text>
<text x="62.0" y="241.8" class="fig-small" text-anchor="end">-8</text>
<line x1="70.0" y1="237.8" x2="330.0" y2="237.8" class="stroke-grid" />
<text x="62.0" y="192.9" class="fig-small" text-anchor="end">-4</text>
<line x1="70.0" y1="188.9" x2="330.0" y2="188.9" class="stroke-grid" />
<text x="62.0" y="144.0" class="fig-small" text-anchor="end">0</text>
<text x="62.0" y="95.1" class="fig-small" text-anchor="end">4</text>
<line x1="70.0" y1="91.1" x2="330.0" y2="91.1" class="stroke-grid" />
<text x="62.0" y="46.2" class="fig-small" text-anchor="end">8</text>
<line x1="70.0" y1="42.2" x2="330.0" y2="42.2" class="stroke-grid" />
<text x="200.0" y="292.0" class="fig-small" text-anchor="middle">step number</text>
<text x="20.0" y="20.0" class="fig-small" text-anchor="start">signed distance from the bottom, steep axis</text>
<polyline points="70.0,127.8 156.7,133.9 243.3,136.9 330.0,138.5" fill="none" class="stroke-c" />
<circle cx="70.0" cy="127.8" r="3.5" class="fill-c"/>
<circle cx="156.7" cy="133.9" r="3.5" class="fill-c"/>
<circle cx="243.3" cy="136.9" r="3.5" class="fill-c"/>
<circle cx="330.0" cy="138.5" r="3.5" class="fill-c"/>
<line x1="360.0" y1="80.0" x2="386.0" y2="80.0" class="stroke-c" />
<circle cx="373.0" cy="80.0" r="3.5" class="fill-c"/>
<text x="394.0" y="84.0" class="fig-small" text-anchor="start">step 0.05: creeps</text>
<polyline points="70.0,127.8 156.7,140.0 243.3,140.0 330.0,140.0" fill="none" class="stroke-a" />
<circle cx="70.0" cy="127.8" r="3.5" class="fill-a"/>
<circle cx="156.7" cy="140.0" r="3.5" class="fill-a"/>
<circle cx="243.3" cy="140.0" r="3.5" class="fill-a"/>
<circle cx="330.0" cy="140.0" r="3.5" class="fill-a"/>
<line x1="360.0" y1="110.0" x2="386.0" y2="110.0" class="stroke-a" />
<circle cx="373.0" cy="110.0" r="3.5" class="fill-a"/>
<text x="394.0" y="114.0" class="fig-small" text-anchor="start">step 0.1: lands at 0</text>
<polyline points="70.0,127.8 156.7,149.8 243.3,132.2 330.0,146.3" fill="none" class="stroke-d" />
<circle cx="70.0" cy="127.8" r="3.5" class="fill-d"/>
<circle cx="156.7" cy="149.8" r="3.5" class="fill-d"/>
<circle cx="243.3" cy="132.2" r="3.5" class="fill-d"/>
<circle cx="330.0" cy="146.3" r="3.5" class="fill-d"/>
<line x1="360.0" y1="140.0" x2="386.0" y2="140.0" class="stroke-d" />
<circle cx="373.0" cy="140.0" r="3.5" class="fill-d"/>
<text x="394.0" y="144.0" class="fig-small" text-anchor="start">step 0.18: overshoots, settles</text>
<polyline points="70.0,127.8 156.7,164.4 243.3,91.1 330.0,237.8" fill="none" class="stroke-bad" />
<circle cx="70.0" cy="127.8" r="3.5" class="fill-bad"/>
<circle cx="156.7" cy="164.4" r="3.5" class="fill-bad"/>
<circle cx="243.3" cy="91.1" r="3.5" class="fill-bad"/>
<circle cx="330.0" cy="237.8" r="3.5" class="fill-bad"/>
<line x1="360.0" y1="170.0" x2="386.0" y2="170.0" class="stroke-bad" />
<circle cx="373.0" cy="170.0" r="3.5" class="fill-bad"/>
<text x="394.0" y="174.0" class="fig-small" text-anchor="start">step 0.3: swings double</text>
</svg>
Figure 2. The same start, distance 1 from the bottom, with four step sizes. A step of 0.05 creeps (green). A step of 0.1 lands exactly on the bottom (blue). A step of 0.18 overshoots and flips sides each time, but the swings shrink (purple). A step of 0.3 overshoots by more than it started with, so the swings double: 1, -2, 4, -8 (red).
```

The number that matters on each axis is its *curvature*, how fast the slope itself changes: the steep slope grows by 10 for every unit you move, and the flat slope by 1. For a general symmetric $A$, the curvatures along the bowl's own axes, which tilt with $A$, are its *eigenvalues*, written $\lambda$ (lambda). The largest is $L$ and the smallest is $\mu$ (mu). Here $L=10$ and $\mu=1$.

The factor for one axis is $1-\alpha\lambda$. On the steep axis:

- **Below $1/L=0.1$**: the factor is between 0 and 1, so you creep toward the bottom. At $1/L$ it is 0, and you land on the bottom value in one step.
- **Between $1/L$ and $2/L$**: the factor is between $-1$ and 0, so you overshoot to the other side, but the size still shrinks. At $2/L$ it is $-1$, and you swing between two points forever.
- **Above $2/L$**: the size of the factor exceeds 1, so the misses grow.

The flat factor $1-\alpha\mu$ stays between 0.8 and 1 for every step below 0.2, so the steep axis blows up first. Plain gradient descent on a bowl therefore converges from every start when

$$
0 < \alpha < \frac{2}{L}
$$

Read it aloud as "the step must be positive and below two over the largest curvature." The common safe choice is $1/L$. In numpy, the factors for all axes at once are:

```python
factors = np.abs(1 - step * eigenvalues)  # one per axis; any entry above 1 grows
```

> [!KEY]
> On each axis a step multiplies your distance from the bottom by $1-\alpha\lambda$. The axis with the largest curvature $L$ blows up first, so the step must stay below $2/L$, and $1/L$ is the usual safe choice.

## The slow axis sets the pace

With step $1/L$ the steep axis has factor 0 and vanishes after one step. Everything after that is the flat axis, with factor $1-\mu/L$, which is $0.9$ on the running bowl.

To compare walks, count the steps that cut the miss to one thousandth of its start, beginning along the flat axis only, at $(0,1)$. The miss after $k$ steps is $0.9^k$, and $0.9^{66}\approx0.00096$ is the first power under 0.001: 66 steps. Next, stretch the bowl to $A=\text{diag}(1,100)$ and $b=(1,100)$, with the same bottom $(1,1)$. Here $L=100$, the safe step is $0.01$, and the flat factor is $0.99$. The first power of $0.99$ under 0.001 is $0.99^{688}$: 688 steps.

A longer step, slightly below $2/L$, only changes the flat factor from $1-\mu/L$ to about $1-2\mu/L$. That roughly halves the count and does not close a gap of ten times, because the steep walls stop you from pushing harder. Figure 3 puts a round bowl beside the $\kappa=100$ valley.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Two bowls viewed in the same window: round contours and a long thin valley, each with its walk">
<defs><marker id="c005-valley-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><clipPath id="c005-valley-clip0"><rect x="20" y="5" width="270" height="270"/></clipPath><clipPath id="c005-valley-clip1"><rect x="310" y="5" width="270" height="270"/></clipPath></defs>
<rect x="20" y="5" width="270" height="270" class="stroke-grid" fill="none"/>
<g clip-path="url(#c005-valley-clip0)">
<ellipse cx="155.0" cy="140.0" rx="42.2" ry="42.2" class="stroke-grid" fill="none"/>
<ellipse cx="155.0" cy="140.0" rx="84.4" ry="84.4" class="stroke-grid" fill="none"/>
<ellipse cx="155.0" cy="140.0" rx="119.3" ry="119.3" class="stroke-a" fill="none"/>
</g>
<line x1="70.6" y1="224.4" x2="149.1" y2="145.9" class="stroke-b" marker-end="url(#c005-valley-arrow-b)"/>
<circle cx="70.6" cy="224.4" r="4" class="fill-b"/>
<circle cx="155.0" cy="140.0" r="4" class="fill-ink"/>
<text x="155.0" y="297.0" class="fig-small" text-anchor="middle">circle (condition number 1)</text>
<text x="163.0" y="132.0" class="fig-small" text-anchor="start">bottom, one step</text>
<text x="64.6" y="240.4" class="fig-small" text-anchor="end">start</text>
<rect x="310" y="5" width="270" height="270" class="stroke-grid" fill="none"/>
<g clip-path="url(#c005-valley-clip1)">
<ellipse cx="445.0" cy="140.0" rx="848.0" ry="84.8" class="stroke-a" fill="none"/>
<ellipse cx="445.0" cy="140.0" rx="533.6" ry="53.4" class="stroke-grid" fill="none"/>
<ellipse cx="445.0" cy="140.0" rx="266.8" ry="26.7" class="stroke-grid" fill="none"/>
<ellipse cx="445.0" cy="140.0" rx="119.3" ry="11.9" class="stroke-grid" fill="none"/>
</g>
<polyline points="360.6,224.4 361.5,140.0" fill="none" class="stroke-b" />
<polyline points="361.5,140.0 362.3,140.0 363.1,140.0 363.9,140.0 364.8,140.0 365.6,140.0 366.4,140.0 367.1,140.0 367.9,140.0 368.7,140.0 369.5,140.0 370.2,140.0 371.0,140.0 371.7,140.0 372.4,140.0 373.2,140.0 373.9,140.0 374.6,140.0 375.3,140.0 376.0,140.0 376.7,140.0 377.4,140.0 378.0,140.0 378.7,140.0 379.4,140.0 380.0,140.0 380.7,140.0 381.3,140.0 382.0,140.0 382.6,140.0 383.2,140.0 383.8,140.0 384.4,140.0 385.0,140.0 385.6,140.0 386.2,140.0 386.8,140.0 387.4,140.0 388.0,140.0 388.6,140.0 389.1,140.0 389.7,140.0 390.2,140.0 390.8,140.0 391.3,140.0 391.9,140.0 392.4,140.0 392.9,140.0 393.4,140.0 394.0,140.0 394.5,140.0 395.0,140.0 395.5,140.0 396.0,140.0 396.5,140.0 396.9,140.0 397.4,140.0 397.9,140.0 398.4,140.0 398.8,140.0 399.3,140.0 399.8,140.0 400.2,140.0 400.7,140.0 401.1,140.0 401.5,140.0 402.0,140.0 402.4,140.0 402.8,140.0 403.2,140.0 403.7,140.0 404.1,140.0 404.5,140.0 404.9,140.0 405.3,140.0 405.7,140.0 406.1,140.0 406.5,140.0 406.9,140.0 407.2,140.0 407.6,140.0 408.0,140.0 408.4,140.0 408.7,140.0 409.1,140.0 409.4,140.0 409.8,140.0 410.2,140.0 410.5,140.0 410.9,140.0 411.2,140.0 411.5,140.0 411.9,140.0 412.2,140.0 412.5,140.0 412.8,140.0 413.2,140.0 413.5,140.0 413.8,140.0 414.1,140.0 414.4,140.0 414.7,140.0 415.0,140.0 415.3,140.0 415.6,140.0 415.9,140.0 416.2,140.0 416.5,140.0 416.8,140.0 417.1,140.0 417.3,140.0 417.6,140.0 417.9,140.0 418.2,140.0 418.4,140.0 418.7,140.0 419.0,140.0 419.2,140.0 419.5,140.0 419.7,140.0 420.0,140.0 420.2,140.0 420.5,140.0 420.7,140.0 421.0,140.0 421.2,140.0 421.5,140.0 421.7,140.0 421.9,140.0 422.2,140.0 422.4,140.0 422.6,140.0 422.8,140.0 423.1,140.0 423.3,140.0 423.5,140.0 423.7,140.0 423.9,140.0 424.1,140.0 424.3,140.0 424.5,140.0 424.8,140.0 425.0,140.0 425.2,140.0 425.4,140.0 425.5,140.0 425.7,140.0 425.9,140.0 426.1,140.0 426.3,140.0 426.5,140.0 426.7,140.0 426.9,140.0 427.1,140.0 427.2,140.0 427.4,140.0 427.6,140.0 427.8,140.0 427.9,140.0 428.1,140.0 428.3,140.0 428.4,140.0 428.6,140.0 428.8,140.0 428.9,140.0 429.1,140.0 429.2,140.0 429.4,140.0 429.6,140.0 429.7,140.0 429.9,140.0 430.0,140.0 430.2,140.0 430.3,140.0 430.5,140.0 430.6,140.0 430.8,140.0 430.9,140.0 431.0,140.0 431.2,140.0 431.3,140.0 431.5,140.0 431.6,140.0 431.7,140.0 431.9,140.0 432.0,140.0 432.1,140.0 432.2,140.0 432.4,140.0 432.5,140.0 432.6,140.0 432.7,140.0 432.9,140.0 433.0,140.0 433.1,140.0 433.2,140.0 433.3,140.0 433.5,140.0 433.6,140.0 433.7,140.0 433.8,140.0 433.9,140.0 434.0,140.0 434.1,140.0 434.2,140.0 434.4,140.0 434.5,140.0 434.6,140.0 434.7,140.0 434.8,140.0 434.9,140.0 435.0,140.0 435.1,140.0 435.2,140.0 435.3,140.0 435.4,140.0 435.5,140.0 435.6,140.0 435.7,140.0 435.8,140.0 435.8,140.0 435.9,140.0 436.0,140.0 436.1,140.0 436.2,140.0 436.3,140.0 436.4,140.0 436.5,140.0 436.6,140.0 436.6,140.0 436.7,140.0 436.8,140.0 436.9,140.0 437.0,140.0 437.0,140.0 437.1,140.0 437.2,140.0 437.3,140.0 437.4,140.0 437.4,140.0 437.5,140.0 437.6,140.0 437.7,140.0 437.7,140.0 437.8,140.0 437.9,140.0 438.0,140.0 438.0,140.0 438.1,140.0 438.2,140.0 438.2,140.0 438.3,140.0 438.4,140.0 438.4,140.0 438.5,140.0 438.6,140.0 438.6,140.0 438.7,140.0 438.8,140.0 438.8,140.0 438.9,140.0 438.9,140.0 439.0,140.0 439.1,140.0 439.1,140.0 439.2,140.0 439.2,140.0 439.3,140.0 439.3,140.0 439.4,140.0 439.5,140.0 439.5,140.0 439.6,140.0 439.6,140.0 439.7,140.0 439.7,140.0 439.8,140.0 439.8,140.0 439.9,140.0 439.9,140.0 440.0,140.0 440.0,140.0 440.1,140.0 440.1,140.0 440.2,140.0 440.2,140.0 440.3,140.0 440.3,140.0 440.4,140.0 440.4,140.0 440.5,140.0 440.5,140.0 440.6,140.0 440.6,140.0 440.6,140.0 440.7,140.0 440.7,140.0 440.8,140.0 440.8,140.0 440.9,140.0 440.9,140.0 440.9,140.0 441.0,140.0 441.0,140.0 441.1,140.0 441.1,140.0 441.1,140.0 441.2,140.0 441.2,140.0 441.3,140.0 441.3,140.0 441.3,140.0 441.4,140.0 441.4,140.0 441.4,140.0 441.5,140.0 441.5,140.0 441.5,140.0 441.6,140.0 441.6,140.0 441.6,140.0 441.7,140.0 441.7,140.0 441.7,140.0 441.8,140.0 441.8,140.0 441.8,140.0 441.9,140.0 441.9,140.0 441.9,140.0 442.0,140.0 442.0,140.0 442.0,140.0 442.1,140.0 442.1,140.0 442.1,140.0 442.1,140.0 442.2,140.0 442.2,140.0 442.2,140.0 442.3,140.0 442.3,140.0 442.3,140.0 442.3,140.0 442.4,140.0 442.4,140.0 442.4,140.0 442.4,140.0 442.5,140.0 442.5,140.0 442.5,140.0 442.5,140.0 442.6,140.0 442.6,140.0 442.6,140.0 442.6,140.0 442.7,140.0 442.7,140.0 442.7,140.0 442.7,140.0 442.8,140.0 442.8,140.0 442.8,140.0 442.8,140.0 442.8,140.0 442.9,140.0 442.9,140.0 442.9,140.0 442.9,140.0 443.0,140.0 443.0,140.0 443.0,140.0 443.0,140.0 443.0,140.0 443.1,140.0 443.1,140.0 443.1,140.0 443.1,140.0 443.1,140.0 443.1,140.0 443.2,140.0 443.2,140.0 443.2,140.0 443.2,140.0 443.2,140.0 443.3,140.0 443.3,140.0 443.3,140.0 443.3,140.0 443.3,140.0 443.3,140.0 443.4,140.0 443.4,140.0 443.4,140.0 443.4,140.0 443.4,140.0 443.4,140.0 443.5,140.0 443.5,140.0 443.5,140.0 443.5,140.0 443.5,140.0 443.5,140.0 443.5,140.0 443.6,140.0 443.6,140.0 443.6,140.0 443.6,140.0 443.6,140.0 443.6,140.0 443.6,140.0 443.7,140.0 443.7,140.0 443.7,140.0 443.7,140.0 443.7,140.0 443.7,140.0 443.7,140.0 443.7,140.0 443.8,140.0 443.8,140.0 443.8,140.0 443.8,140.0 443.8,140.0 443.8,140.0 443.8,140.0 443.8,140.0 443.9,140.0 443.9,140.0 443.9,140.0 443.9,140.0 443.9,140.0 443.9,140.0 443.9,140.0 443.9,140.0 443.9,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.0,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.1,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.2,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.3,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.4,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.5,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.6,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.7,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.8,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0 444.9,140.0" fill="none" class="stroke-b" />
<circle cx="361.5" cy="140.0" r="3.5" class="fill-b"/>
<circle cx="414.1" cy="140.0" r="3.5" class="fill-b"/>
<circle cx="436.6" cy="140.0" r="3.5" class="fill-b"/>
<circle cx="444.2" cy="140.0" r="3.5" class="fill-b"/>
<circle cx="360.6" cy="224.4" r="4" class="fill-b"/>
<circle cx="445.0" cy="140.0" r="4" class="fill-ink"/>
<text x="445.0" y="297.0" class="fig-small" text-anchor="middle">long valley (condition number 100)</text>
<text x="414.1" y="116.0" class="fig-small" text-anchor="middle">step 100</text>
<text x="436.6" y="170.0" class="fig-small" text-anchor="middle">step 230</text>
<text x="353.5" y="130.0" class="fig-small" text-anchor="end">step 1</text>
<text x="354.6" y="240.4" class="fig-small" text-anchor="end">start</text>
</svg>
Figure 3. Both panels show the same window of the plane, with the same start (0, 0) and the same bottom (1, 1). On the left the contours are circles and one step reaches the bottom. On the right the contours are so stretched that, in this window, they look like parallel lines: the first step climbs the steep wall to the floor, and then the walk crawls along the floor. Step 100 is still 0.37 short of the bottom along the floor, step 230 is still 0.1 short, and step 460 is almost on top of it.
```

The ratio of the largest to the smallest curvature is the *condition number*, written $\kappa$ (kappa):

$$
\kappa = \frac{L}{\mu}
$$

Read it aloud as "the sharpest curvature divided by the flattest." The two bowls have $\kappa=10$ and $\kappa=100$, and with step $1/L$ the flat factor is $1-1/\kappa$. With $\kappa=1$ the lines of equal height are circles, and one step reaches the bottom, because every direction points at it. A larger $\kappa$ stretches them into ellipses whose long side is $\sqrt{\kappa}$ times the short side.

After $k$ steps the flat miss is $(1-1/\kappa)^k$ times the start, so cutting it by a factor $F$ takes about $\kappa\ln F$ steps. For $F=1000$ that is about $6.9\,\kappa$: 69 for $\kappa=10$, close to the exact 66, and 691 for $\kappa=100$, close to 688. The count is proportional to $\kappa$.

Real bowls have large condition numbers. A house-price fit with a size column near 1000 and a bedrooms column near 3 has $\kappa$ of at least $(1000/3)^2\approx110{,}000$. The cure is to *standardise*: subtract each column's mean and divide by its standard deviation, which brings $\kappa$ near 1 for unrelated columns. For least squares, $A=X^\top X$ squares the condition number of $X$, the same squaring that lost digits in [stable least squares](/lessons/004-stable-least-squares).

## A ball keeps its speed

On the valley floor the walk moves right, right, right, each step about the same size as the last, and never notices. A rolling ball keeps going the way it was moving, so it gathers speed along the floor, and the bounce off one wall cancels part of the bounce off the other.

Add a fraction of the last step to the slope step. The fraction is the *momentum* $\beta$ (beta). Take the running bowl with $\alpha=0.1$ and the sample value $\beta=0.5$. The first step has no last step to remember, so it is the plain one, to $(0.1,\,1.0)$. The slope there is $(-0.9,\,0)$, so the push is $(0.09,\,0)$. Half of the last step $(0.1,\,1.0)$ is $(0.05,\,0.5)$, and the step is $(0.14,\,0.5)$, which lands you at $(0.24,\,1.5)$. Plain descent would land at $(0.19,\,1.0)$. One more step lands at $(0.386,\,1.25)$.

On the flat axis the steps are $0.1,\,0.14,\,0.146$, growing, because each push adds to a step already pointing the right way. On the steep axis the coordinate went $1.0,\,1.5,\,1.25$: it overshot, and the push came back and cancelled part of the speed. In symbols, with $w_{k-1}$ for the point before $w_k$:

$$
w_{k+1} = w_k - \alpha\,\nabla f(w_k) + \beta\,(w_k - w_{k-1})
$$

Read it aloud as "step against the slope, then keep a fraction $\beta$ of the last step." This is the *heavy-ball* method. Treat $w_{-1}$ as $w_0$. At $\beta=0$ it is plain descent. In numpy:

```python
w_next = w - step * grad(w) + momentum * (w - w_prev)
```

The flat axis gains because, when every step points the same way, the remembered steps add up: $1+\beta+\beta^2+\cdots=1/(1-\beta)$, so with $\beta=0.5$ the same push acts as a step twice as long. On the steep axis the steps alternate in sign, so the remembered part opposes the next push and the swings shrink.

> [!WARNING]
> Compute `w_next` from the old `w` and `w_prev` first, and only then set `w_prev, w = w, w_next`. If you update `w_prev` first, the momentum term is 0.

### The best settings for a bowl

On one axis with curvature $\lambda$ and miss $m$, the heavy-ball rule regroups as

$$
m_{\text{next}} = (1+\beta-\alpha\lambda)\,m - \beta\,m_{\text{prev}}
$$

Read it aloud as "the next miss is a multiple of the old miss, minus $\beta$ times the miss before." When $\alpha\lambda$ lies between $(1-\sqrt\beta)^2$ and $(1+\sqrt\beta)^2$, the misses shrink by exactly $\sqrt\beta$ per step, with a little swinging, whatever $\lambda$ is. So choose $\alpha$ and $\beta$ so that this window exactly covers every axis: $\alpha\mu=(1-\sqrt\beta)^2$ and $\alpha L=(1+\sqrt\beta)^2$. Solving the two equations gives

$$
\alpha = \frac{4}{(\sqrt{L}+\sqrt{\mu})^2},\qquad \beta = \left(\frac{\sqrt{L}-\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}}\right)^2
$$

For $L=10$ and $\mu=1$ this gives $\alpha\approx0.231$ and $\beta\approx0.270$. The step is above $2/L=0.2$, which does not break the earlier rule, because that rule is for plain descent: with momentum a bowl is stable while $\alpha\lambda<2(1+\beta)$ on every axis. In numpy:

```python
root_L, root_mu = np.sqrt(L), np.sqrt(mu)
step = 4 / (root_L + root_mu) ** 2
momentum = ((root_L - root_mu) / (root_L + root_mu)) ** 2
```

With these settings every axis shrinks by $\sqrt\beta=(\sqrt\kappa-1)/(\sqrt\kappa+1)$ per step, about $1-2/\sqrt{\kappa}$ on a stretched bowl, in place of plain descent's $1-1/\kappa$. The count falls from about $\kappa$ to about $\sqrt{\kappa}$.

### Look ahead, then measure

Heavy ball measures the slope where you stand. The ball already knows momentum will carry it forward, so it can glance ahead and measure the slope there. That is *Nesterov's method*.

From $(0.1,\,1.0)$, with last step $(0.1,\,1.0)$ and $\beta=0.5$, the look-ahead point is $y=(0.15,\,1.5)$. The slope at $y$ is $(-0.85,\,5)$. Step against it from $y$ with $\alpha=0.1$ and you land at $(0.235,\,1.0)$, where heavy ball landed at $(0.24,\,1.5)$. The look-ahead point sits up the steep wall, where the slope is large and points back down, so Nesterov brakes before the overshoot rather than after it. In symbols:

$$
y_k = w_k + \beta\,(w_k - w_{k-1}),\qquad w_{k+1} = y_k - \alpha\,\nabla f(y_k)
$$

Read it aloud as "move ahead by the momentum, then step against the slope measured there." Again $w_{-1}=w_0$. In numpy:

```python
y = w + momentum * (w - w_prev)
w_next = y - step * grad(y)
```

Take the slope at `w` and not at `y`, and you get the heavy-ball step. A usual choice is $\alpha=1/L$ and $\beta=(\sqrt{L}-\sqrt{\mu})/(\sqrt{L}+\sqrt{\mu})$, which is about $0.519$ for the running bowl. This $\beta$ is not squared. Off a bowl, Nesterov's step count has a guarantee for every smooth convex loss, while heavy ball's is proved for bowls only.

### Count the payoff

Take $\kappa=100$, so $\sqrt{\kappa}=10$, and count the steps to bring the miss to one millionth of its start (Figure 4). Plain descent at step $1/L$ needs 1341. Heavy ball with the best settings needs 93, about 14 times fewer, and Nesterov needs 155, about 9 times fewer. Read $\sqrt{\kappa}$ as the scale, not the exact ratio.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Distance from the bottom on a log scale for gradient descent, heavy ball, and Nesterov on a valley with condition number 100">
<defs></defs>
<line x1="70.0" y1="270.0" x2="560.0" y2="270.0" class="stroke-muted" />
<line x1="70.0" y1="30.0" x2="70.0" y2="270.0" class="stroke-grid" />
<line x1="70.0" y1="270.0" x2="70.0" y2="275.0" class="stroke-muted" />
<text x="70.0" y="290.0" class="fig-small" text-anchor="middle">0</text>
<line x1="192.5" y1="270.0" x2="192.5" y2="275.0" class="stroke-muted" />
<text x="192.5" y="290.0" class="fig-small" text-anchor="middle">100</text>
<line x1="315.0" y1="270.0" x2="315.0" y2="275.0" class="stroke-muted" />
<text x="315.0" y="290.0" class="fig-small" text-anchor="middle">200</text>
<line x1="437.5" y1="270.0" x2="437.5" y2="275.0" class="stroke-muted" />
<text x="437.5" y="290.0" class="fig-small" text-anchor="middle">300</text>
<line x1="560.0" y1="270.0" x2="560.0" y2="275.0" class="stroke-muted" />
<text x="560.0" y="290.0" class="fig-small" text-anchor="middle">400</text>
<line x1="70.0" y1="45.0" x2="560.0" y2="45.0" class="stroke-grid" />
<text x="62.0" y="49.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="82.5" x2="560.0" y2="82.5" class="stroke-grid" />
<text x="62.0" y="86.5" class="fig-small" text-anchor="end">10^-1</text>
<line x1="70.0" y1="120.0" x2="560.0" y2="120.0" class="stroke-grid" />
<text x="62.0" y="124.0" class="fig-small" text-anchor="end">10^-2</text>
<line x1="70.0" y1="157.5" x2="560.0" y2="157.5" class="stroke-grid" />
<text x="62.0" y="161.5" class="fig-small" text-anchor="end">10^-3</text>
<line x1="70.0" y1="195.0" x2="560.0" y2="195.0" class="stroke-grid" />
<text x="62.0" y="199.0" class="fig-small" text-anchor="end">10^-4</text>
<line x1="70.0" y1="232.5" x2="560.0" y2="232.5" class="stroke-grid" />
<text x="62.0" y="236.5" class="fig-small" text-anchor="end">10^-5</text>
<line x1="70.0" y1="270.0" x2="560.0" y2="270.0" class="stroke-grid" />
<text x="62.0" y="274.0" class="fig-small" text-anchor="end">10^-6</text>
<text x="315.0" y="312.0" class="fig-small" text-anchor="middle">step number</text>
<polyline points="70.0,45.0 71.2,50.8 72.5,51.0 73.7,51.1 74.9,51.3 76.1,51.5 77.3,51.6 78.6,51.8 79.8,52.0 81.0,52.1 82.2,52.3 83.5,52.4 84.7,52.6 85.9,52.8 87.2,52.9 88.4,53.1 89.6,53.3 90.8,53.4 92.0,53.6 93.3,53.8 94.5,53.9 95.7,54.1 97.0,54.2 98.2,54.4 99.4,54.6 100.6,54.7 101.8,54.9 103.1,55.1 104.3,55.2 105.5,55.4 106.8,55.6 108.0,55.7 109.2,55.9 110.4,56.0 111.7,56.2 112.9,56.4 114.1,56.5 115.3,56.7 116.5,56.9 117.8,57.0 119.0,57.2 120.2,57.4 121.4,57.5 122.7,57.7 123.9,57.8 125.1,58.0 126.3,58.2 127.6,58.3 128.8,58.5 130.0,58.7 131.2,58.8 132.5,59.0 133.7,59.2 134.9,59.3 136.2,59.5 137.4,59.6 138.6,59.8 139.8,60.0 141.1,60.1 142.3,60.3 143.5,60.5 144.7,60.6 145.9,60.8 147.2,61.0 148.4,61.1 149.6,61.3 150.9,61.4 152.1,61.6 153.3,61.8 154.5,61.9 155.8,62.1 157.0,62.3 158.2,62.4 159.4,62.6 160.7,62.8 161.9,62.9 163.1,63.1 164.3,63.2 165.6,63.4 166.8,63.6 168.0,63.7 169.2,63.9 170.4,64.1 171.7,64.2 172.9,64.4 174.1,64.6 175.3,64.7 176.6,64.9 177.8,65.0 179.0,65.2 180.2,65.4 181.5,65.5 182.7,65.7 183.9,65.9 185.1,66.0 186.4,66.2 187.6,66.4 188.8,66.5 190.1,66.7 191.3,66.8 192.5,67.0 193.7,67.2 194.9,67.3 196.2,67.5 197.4,67.7 198.6,67.8 199.8,68.0 201.1,68.2 202.3,68.3 203.5,68.5 204.8,68.6 206.0,68.8 207.2,69.0 208.4,69.1 209.6,69.3 210.9,69.5 212.1,69.6 213.3,69.8 214.5,70.0 215.8,70.1 217.0,70.3 218.2,70.4 219.4,70.6 220.7,70.8 221.9,70.9 223.1,71.1 224.3,71.3 225.6,71.4 226.8,71.6 228.0,71.8 229.2,71.9 230.5,72.1 231.7,72.3 232.9,72.4 234.2,72.6 235.4,72.7 236.6,72.9 237.8,73.1 239.0,73.2 240.3,73.4 241.5,73.6 242.7,73.7 243.9,73.9 245.2,74.1 246.4,74.2 247.6,74.4 248.8,74.5 250.1,74.7 251.3,74.9 252.5,75.0 253.8,75.2 255.0,75.4 256.2,75.5 257.4,75.7 258.6,75.9 259.9,76.0 261.1,76.2 262.3,76.3 263.6,76.5 264.8,76.7 266.0,76.8 267.2,77.0 268.5,77.2 269.7,77.3 270.9,77.5 272.1,77.7 273.4,77.8 274.6,78.0 275.8,78.1 277.0,78.3 278.2,78.5 279.5,78.6 280.7,78.8 281.9,79.0 283.1,79.1 284.4,79.3 285.6,79.5 286.8,79.6 288.1,79.8 289.3,79.9 290.5,80.1 291.7,80.3 293.0,80.4 294.2,80.6 295.4,80.8 296.6,80.9 297.9,81.1 299.1,81.3 300.3,81.4 301.5,81.6 302.8,81.7 304.0,81.9 305.2,82.1 306.4,82.2 307.6,82.4 308.9,82.6 310.1,82.7 311.3,82.9 312.6,83.1 313.8,83.2 315.0,83.4 316.2,83.5 317.4,83.7 318.7,83.9 319.9,84.0 321.1,84.2 322.4,84.4 323.6,84.5 324.8,84.7 326.0,84.9 327.2,85.0 328.5,85.2 329.7,85.3 330.9,85.5 332.2,85.7 333.4,85.8 334.6,86.0 335.8,86.2 337.1,86.3 338.3,86.5 339.5,86.7 340.7,86.8 342.0,87.0 343.2,87.1 344.4,87.3 345.6,87.5 346.8,87.6 348.1,87.8 349.3,88.0 350.5,88.1 351.8,88.3 353.0,88.5 354.2,88.6 355.4,88.8 356.6,88.9 357.9,89.1 359.1,89.3 360.3,89.4 361.6,89.6 362.8,89.8 364.0,89.9 365.2,90.1 366.4,90.3 367.7,90.4 368.9,90.6 370.1,90.7 371.4,90.9 372.6,91.1 373.8,91.2 375.0,91.4 376.2,91.6 377.5,91.7 378.7,91.9 379.9,92.1 381.1,92.2 382.4,92.4 383.6,92.5 384.8,92.7 386.1,92.9 387.3,93.0 388.5,93.2 389.7,93.4 390.9,93.5 392.2,93.7 393.4,93.9 394.6,94.0 395.9,94.2 397.1,94.3 398.3,94.5 399.5,94.7 400.8,94.8 402.0,95.0 403.2,95.2 404.4,95.3 405.7,95.5 406.9,95.7 408.1,95.8 409.3,96.0 410.5,96.1 411.8,96.3 413.0,96.5 414.2,96.6 415.4,96.8 416.7,97.0 417.9,97.1 419.1,97.3 420.3,97.5 421.6,97.6 422.8,97.8 424.0,97.9 425.2,98.1 426.5,98.3 427.7,98.4 428.9,98.6 430.1,98.8 431.4,98.9 432.6,99.1 433.8,99.3 435.1,99.4 436.3,99.6 437.5,99.7 438.7,99.9 439.9,100.1 441.2,100.2 442.4,100.4 443.6,100.6 444.9,100.7 446.1,100.9 447.3,101.1 448.5,101.2 449.8,101.4 451.0,101.5 452.2,101.7 453.4,101.9 454.7,102.0 455.9,102.2 457.1,102.4 458.3,102.5 459.6,102.7 460.8,102.9 462.0,103.0 463.2,103.2 464.5,103.3 465.7,103.5 466.9,103.7 468.1,103.8 469.3,104.0 470.6,104.2 471.8,104.3 473.0,104.5 474.2,104.7 475.5,104.8 476.7,105.0 477.9,105.1 479.1,105.3 480.4,105.5 481.6,105.6 482.8,105.8 484.1,106.0 485.3,106.1 486.5,106.3 487.7,106.5 488.9,106.6 490.2,106.8 491.4,107.0 492.6,107.1 493.9,107.3 495.1,107.4 496.3,107.6 497.5,107.8 498.8,107.9 500.0,108.1 501.2,108.3 502.4,108.4 503.6,108.6 504.9,108.8 506.1,108.9 507.3,109.1 508.6,109.2 509.8,109.4 511.0,109.6 512.2,109.7 513.5,109.9 514.7,110.1 515.9,110.2 517.1,110.4 518.4,110.6 519.6,110.7 520.8,110.9 522.0,111.0 523.2,111.2 524.5,111.4 525.7,111.5 526.9,111.7 528.2,111.9 529.4,112.0 530.6,112.2 531.8,112.4 533.0,112.5 534.3,112.7 535.5,112.8 536.7,113.0 538.0,113.2 539.2,113.3 540.4,113.5 541.6,113.7 542.8,113.8 544.1,114.0 545.3,114.2 546.5,114.3 547.8,114.5 549.0,114.6 550.2,114.8 551.4,115.0 552.6,115.1 553.9,115.3 555.1,115.5 556.3,115.6 557.5,115.8 558.8,116.0 560.0,116.1" fill="none" class="stroke-muted" />
<line x1="300.0" y1="160.0" x2="330.0" y2="160.0" class="stroke-muted" />
<text x="338.0" y="164.0" class="fig-small" text-anchor="start">gradient descent: 1341 steps (off the plot)</text>
<polyline points="70.0,45.0 71.2,50.8 72.5,51.1 73.7,51.5 74.9,52.0 76.1,52.6 77.3,53.3 78.6,54.0 79.8,54.8 81.0,55.6 82.2,56.5 83.5,57.4 84.7,58.4 85.9,59.4 87.2,60.4 88.4,61.5 89.6,62.5 90.8,63.6 92.0,64.8 93.3,65.9 94.5,67.1 95.7,68.3 97.0,69.5 98.2,70.7 99.4,71.9 100.6,73.1 101.8,74.4 103.1,75.7 104.3,76.9 105.5,78.2 106.8,79.5 108.0,80.9 109.2,82.2 110.4,83.5 111.7,84.9 112.9,86.2 114.1,87.6 115.3,88.9 116.5,90.3 117.8,91.7 119.0,93.1 120.2,94.5 121.4,95.9 122.7,97.3 123.9,98.7 125.1,100.1 126.3,101.5 127.6,102.9 128.8,104.4 130.0,105.8 131.2,107.3 132.5,108.7 133.7,110.2 134.9,111.6 136.2,113.1 137.4,114.5 138.6,116.0 139.8,117.5 141.1,118.9 142.3,120.4 143.5,121.9 144.7,123.4 145.9,124.9 147.2,126.4 148.4,127.9 149.6,129.4 150.9,130.9 152.1,132.4 153.3,133.9 154.5,135.4 155.8,136.9 157.0,138.4 158.2,139.9 159.4,141.4 160.7,143.0 161.9,144.5 163.1,146.0 164.3,147.5 165.6,149.1 166.8,150.6 168.0,152.1 169.2,153.7 170.4,155.2 171.7,156.7 172.9,158.3 174.1,159.8 175.3,161.4 176.6,162.9 177.8,164.5 179.0,166.0 180.2,167.6 181.5,169.1 182.7,170.7 183.9,172.2 185.1,173.8 186.4,175.4 187.6,176.9 188.8,178.5 190.1,180.0 191.3,181.6 192.5,183.2 193.7,184.8 194.9,186.3 196.2,187.9 197.4,189.5 198.6,191.0 199.8,192.6 201.1,194.2 202.3,195.8 203.5,197.3 204.8,198.9 206.0,200.5 207.2,202.1 208.4,203.7 209.6,205.3 210.9,206.8 212.1,208.4 213.3,210.0 214.5,211.6 215.8,213.2 217.0,214.8 218.2,216.4 219.4,218.0 220.7,219.6 221.9,221.2 223.1,222.7 224.3,224.3 225.6,225.9 226.8,227.5 228.0,229.1 229.2,230.7 230.5,232.3 231.7,233.9 232.9,235.5 234.2,237.1 235.4,238.7 236.6,240.3 237.8,241.9 239.0,243.6 240.3,245.2 241.5,246.8 242.7,248.4 243.9,250.0 245.2,251.6 246.4,253.2 247.6,254.8 248.8,256.4 250.1,258.0 251.3,259.6 252.5,261.3 253.8,262.9 255.0,264.5 256.2,266.1 257.4,267.7 258.6,269.3 259.9,270.0" fill="none" class="stroke-d" />
<line x1="300.0" y1="180.0" x2="330.0" y2="180.0" class="stroke-d" />
<text x="338.0" y="184.0" class="fig-small" text-anchor="start">Nesterov: 155 steps</text>
<polyline points="70.0,45.0 71.2,35.7 72.5,31.5 73.7,29.6 74.9,29.0 76.1,29.1 77.3,29.7 78.6,30.6 79.8,31.9 81.0,33.4 82.2,35.0 83.5,36.8 84.7,38.8 85.9,40.8 87.2,42.9 88.4,45.1 89.6,47.4 90.8,49.7 92.0,52.0 93.3,54.5 94.5,56.9 95.7,59.4 97.0,61.9 98.2,64.5 99.4,67.1 100.6,69.7 101.8,72.4 103.1,75.0 104.3,77.7 105.5,80.4 106.8,83.2 108.0,85.9 109.2,88.7 110.4,91.4 111.7,94.2 112.9,97.0 114.1,99.8 115.3,102.7 116.5,105.5 117.8,108.4 119.0,111.2 120.2,114.1 121.4,117.0 122.7,119.9 123.9,122.8 125.1,125.7 126.3,128.6 127.6,131.5 128.8,134.4 130.0,137.4 131.2,140.3 132.5,143.3 133.7,146.2 134.9,149.2 136.2,152.2 137.4,155.1 138.6,158.1 139.8,161.1 141.1,164.1 142.3,167.1 143.5,170.1 144.7,173.1 145.9,176.1 147.2,179.1 148.4,182.1 149.6,185.1 150.9,188.1 152.1,191.2 153.3,194.2 154.5,197.2 155.8,200.3 157.0,203.3 158.2,206.3 159.4,209.4 160.7,212.4 161.9,215.5 163.1,218.5 164.3,221.6 165.6,224.7 166.8,227.7 168.0,230.8 169.2,233.9 170.4,236.9 171.7,240.0 172.9,243.1 174.1,246.2 175.3,249.2 176.6,252.3 177.8,255.4 179.0,258.5 180.2,261.6 181.5,264.7 182.7,267.7 183.9,270.0" fill="none" class="stroke-b" />
<line x1="300.0" y1="200.0" x2="330.0" y2="200.0" class="stroke-b" />
<text x="338.0" y="204.0" class="fig-small" text-anchor="start">heavy ball: 93 steps</text>
<text x="20.0" y="20.0" class="fig-small" text-anchor="start">distance to the bottom, relative to the start</text>
</svg>
Figure 4. Same valley (condition number 100), same start, three walks, each stopped when its distance reaches one millionth of the start. Gradient descent (grey) is still above 10^-2 after 400 steps and needs 1341. Heavy ball (orange) needs 93: it first swings out, then falls fastest. Nesterov (purple) needs 155 and does not swing out as far.
```

> [!KEY]
> You can walk to the bottom of a loss without seeing it, because the gradient tells you which way is down. The step must stay below $2/L$, and the flat axis then needs about $\kappa$ steps. A remembered step makes consistent steps add up and alternating steps cancel, which cuts the count to about $\sqrt{\kappa}$ without a bigger step on the walls.

> [!RECAP]
> - The gradient is $Aw-b$ for the bowl, and a step is $w-\alpha\nabla f(w)$.
> - On each axis a step multiplies the miss by $1-\alpha\lambda$, so plain descent needs $\alpha<2/L$.
> - At step $1/L$ the flat axis shrinks by $1-1/\kappa$ per step, so the count grows in proportion to $\kappa=L/\mu$.
> - Heavy ball adds $\beta(w_k-w_{k-1})$ to the step, and Nesterov measures the slope at $y=w+\beta(w-w_{\text{prev}})$. Both take about $\sqrt{\kappa}$ steps.

## Build it step by step

The checks call the functions in this order:

1. **`quadratic_loss_and_grad(A, b, w)`** takes `A` of shape `(d, d)` and `b` and `w` of shape `(d,)`, and returns a float loss and a gradient of shape `(d,)`. Dropping the 0.5, or the minus sign on `b`, fails the hand values.
2. **`gradient_descent(grad, w0, step, iters)`** takes a function `grad` that you call as `grad(w)`, and returns the path of shape `(iters + 1, d)`. Allocate `np.empty((iters + 1, len(w0)))`, set row 0 to `w0`, and fill each later row from the one before, working on a copy of `w0`.
3. **`eigenvalue_range(A)`** returns `(mu, L)` as two floats, the first and last of `np.linalg.eigvalsh(A)`. Use `eigvalsh`, not `eigvals`, because it returns real numbers for a symmetric matrix.
4. **`condition_number(A)`** returns `L / mu`. Dividing the other way gives a number below 1.
5. **`axis_factors(A, step)`** returns `np.abs(1 - step * eigenvalues)`, one entry per eigenvalue in ascending order. At step 0.3 on the running bowl you get `[0.7, 2.0]`.
6. **`steps_to_tolerance(path, w_star, tol)`** returns the first row index whose error is below `tol` times the starting error, or `-1`. Take `errors = np.linalg.norm(path - w_star, axis=1)`, one length per row, then `np.nonzero(errors < tol * errors[0])[0]`. Row 0 never counts.
7. **`heavy_ball(grad, w0, step, momentum, iters)`** returns a path of the same shape. Keep two copies of `w0`, named `w` and `w_prev`, and update `w_prev` only after you compute `w_next`, or the momentum term is 0.
8. **`nesterov(grad, w0, step, momentum, iters)`** returns the path. Use the same two copies, and take the gradient at the look-ahead point `y`. Calling `grad(w)` is the heavy-ball rule, and the hand check catches it.
9. **`best_heavy_ball_params(mu, L)`** returns the tuple `(step, momentum)` as floats. Forgetting the final square on `momentum` gives 0.519 where 0.270 is right.

## Where this breaks

**You do not know $L$.** You rarely compute eigenvalues of a large loss. Halve the step whenever the loss rises, which is the $2/L$ rule applied by hand, and read a loss that falls fast and then crawls as a large $\kappa$.

**The loss is not a bowl.** The curvature changes from place to place, so $\kappa$ is a local number, and the walk ends in the nearest valley, which need not be the lowest.

**Too much momentum.** With $\beta$ near 1 the ball overshoots again and again, and momentum raises the longest safe step to $2(1+\beta)/L$ without removing it. Tune $\beta$ by watching the loss, with 0.9 a usual starting point.

## The exercise

You build `quadratic_loss_and_grad`, `gradient_descent`, `eigenvalue_range`, `condition_number`, `axis_factors`, `steps_to_tolerance`, `heavy_ball`, `nesterov`, and `best_heavy_ball_params`.

The checks compare the loss, gradient, and descent path with hand values and a finite difference, the curvature functions with known values on a tilted matrix, and `axis_factors` with hand values. A step of $1.9/L$ must shrink the error and $2.1/L$ must grow it, and the step counts must be 1 for a circle, 66 for $\kappa=10$, and 688 for $\kappa=100$. The momentum checks compare `heavy_ball` and `nesterov` with hand steps and with plain descent at $\beta=0$, and `best_heavy_ball_params` with $(0.231,0.270)$. The last check confirms that both methods need far fewer steps than plain descent when $\kappa=100$.
