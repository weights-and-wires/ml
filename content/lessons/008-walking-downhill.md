---
title: "Walking downhill: slope, gradient, and one step"
slug: 008-walking-downhill
part: "Part 1 — Foundations"
summary: Feel the slope under your feet, step against it, and walk to the bottom of a bowl you cannot solve for directly.
minutes: 40
gpu: false
objectives:
  - Read the sign of a slope as a direction, and step against it on a curve.
  - Compute the loss and the gradient of a two-weight bowl, and check the gradient with a small nudge.
  - Run gradient descent and watch the loss fall at every step.
  - Explain why people walk to the bottom of a loss instead of solving for it.
prereqs:
  - 005-least-squares-line
  - 007-stable-least-squares
exercise: 008-walking-downhill
---

# Walking downhill: slope, gradient, and one step

You stand on a hillside in thick fog. You cannot see the valley, but you can feel the ground tilt under your feet. If the ground tilts down to your left, you step left, feel the tilt again, and repeat. You never see the bottom, yet you reach it.

Training a model is the same walk. The height of the hill is the loss, and your position is the list of weights. A formula can sometimes jump to the bottom, as it did when you fit a line by minimising squared misses. So why would anyone walk?

Because most losses have no such formula. A logistic model and a neural network have a loss you can score at any point, and a tilt you can feel at any point, but nothing that says where the bottom is. This chapter teaches the walk. The next chapters teach how long each step is allowed to be and how to walk faster.

## What you already know

Five ideas carry this chapter:

- A loss turns a guess into a height, and lower is better (see "Scoring a guess: loss and empirical risk").
- The squared-miss loss of a line is a bowl in the weights, and the best weights sit at the bottom (see "Fit a line by minimising squared misses").
- The slope of a curve at a point says how fast the height changes when you move right. At the bottom of a smooth bowl the slope is 0.
- A weight is a number you can adjust, and there is one weight per feature.
- `A @ w` multiplies a matrix by a vector in numpy.

## Feel the slope on a curve

Start with one weight. Let the loss be $(w-3)^2$, a bowl whose bottom is at $w=3$. Pretend you cannot see that. You stand at $w=0$, where the height is 9.

Your only tool is the slope, which here is $2(w-3)$. At $w=0$ it is $-6$. A negative slope means the height falls as $w$ grows, so the bottom is to your right. At $w=1.5$ the slope is $-3$, and at $w=3.5$ it is $+1$, which means the height rises to the right, so you must go left.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="The curve (w - 3) squared with the slope and the step direction at three points">
<defs><marker id="c009-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
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
<line x1="115.0" y1="272.0" x2="212.5" y2="272.0" class="stroke-b" marker-end="url(#c009-arrow-b)"/>
<circle cx="115.0" cy="272.0" r="3" class="fill-b"/>
<text x="115.0" y="296.0" class="fig-small" text-anchor="middle">w = 0: slope -6</text>
<line x1="154.0" y1="171.2" x2="271.0" y2="235.3" class="stroke-muted dashed" />
<circle cx="212.5" cy="203.3" r="5" class="fill-b"/>
<line x1="212.5" y1="272.0" x2="261.2" y2="272.0" class="stroke-b" marker-end="url(#c009-arrow-b)"/>
<circle cx="212.5" cy="272.0" r="3" class="fill-b"/>
<text x="212.5" y="296.0" class="fig-small" text-anchor="middle">w = 1.5: slope -3</text>
<line x1="284.0" y1="237.7" x2="401.0" y2="216.3" class="stroke-muted dashed" />
<circle cx="342.5" cy="227.0" r="5" class="fill-b"/>
<line x1="342.5" y1="272.0" x2="326.2" y2="272.0" class="stroke-b" marker-end="url(#c009-arrow-b)"/>
<circle cx="342.5" cy="272.0" r="3" class="fill-b"/>
<text x="342.5" y="296.0" class="fig-small" text-anchor="middle">w = 3.5: slope +1</text>
<text x="310.0" y="30.0" class="fig-small" text-anchor="middle">height = (w - 3)²</text>
<text x="586.0" y="250.0" class="fig-small" text-anchor="start">w</text>
<text x="20.0" y="276.0" class="fig-small" text-anchor="start">step</text>
</svg>
Figure 1. Dashed lines are the slope at three points. Where the slope is negative the curve falls to the right, so the orange step (one quarter of the slope, taken backwards, drawn below the axis) points right. Where the slope is positive it points left. Either way the step moves toward the bottom at w = 3.
```

In words, the sign of the slope tells you the direction, and you walk the opposite way. The size of the slope tells you how steep it is, and a steep slope means you are far from the bottom. So you step by a fraction of the slope. That fraction is the step size, written $\alpha$ (alpha), and it is also called the learning rate.

Take $\alpha=0.25$ from $w=0$. The slope is $-6$, so you move to $0-0.25\times(-6)=1.5$. The slope there is $-3$, so you move to $2.25$. Then you reach $2.625$, then $2.8125$. Each step is half the size of the one before, because the slope shrinks as you approach the bottom, which means the walk slows down by itself.

In symbols:

$$
w_{\text{next}} = w - \alpha \cdot \text{slope}(w)
$$

Read it aloud as "move from $w$ against the slope, by a fraction $\alpha$ of it." In numpy it is one line:

```python
w = w - step * slope  # slope is 2 * (w - 3) in this example
```

Add the slope instead of subtracting it and you climb the hill, so the loss grows at every step.

## Why walk when you can solve

On this curve you could solve the problem by setting the slope to 0: $2(w-3)=0$ gives $w=3$. Why not do the same for a loss with a thousand weights?

For a bowl, you can. The slope conditions become a linear system, and solving it costs about $d^3$ multiplications for $d$ weights. That is a billion for $d=1000$. One slope costs about $d^2$, a million, because it is a single matrix-vector product. For a loss that is not a bowl, such as logistic loss or a neural network, the slope conditions have no formula at all, and walking is the only option.

There is a second reason to study a bowl. Zoom in on any smooth loss near its minimum and it looks like a bowl. So the last part of every walk looks like the walk you learn here.

## Two weights make a steep side and a flat side

Real models have many weights, so give the bowl two, $w_1$ and $w_2$. You need two because the interesting trouble starts when one weight matters more than the other. Say $w_2$ is ten times as sensitive, the way a weight on a length measured in millimetres reacts more strongly to a nudge than a weight on the same length in metres. The height is then half of $w_1^2+10w_2^2$, minus $w_1+10w_2$. The 10 is a made-up stand-in for that imbalance, chosen to keep the arithmetic small. At $(0,0)$ every term is 0, so the height is 0. At $(1,1)$ the half is $0.5\times 11=5.5$ and you subtract 11, so the height is $-5.5$. At $(0,1)$ it is $5-10=-5$.

Where is the bottom? Rewrite the height as

$$
f(w) = \tfrac12 (w_1-1)^2 + 5\,(w_2-1)^2 - 5.5
$$

Expand it and you get the original. With that form you can read the bowl: each weight has its own one-weight bowl. The lowest height is $-5.5$ at $(1,1)$, and the walk will be measured against that point. Call the distance of one coordinate from its bottom value its *miss*. At $(0,0)$ both misses are 1.

The two bowls differ in how sharply they curve. The $w_2$ bowl is ten times as curved as the $w_1$ bowl, so $w_2$ is the *steep axis* and $w_1$ is the *flat axis*. Figure 2 draws lines of equal height. They are tight around $w_2$ and stretched along $w_1$.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Contour ellipses of the two-weight bowl with the first steps of the walk">
<defs></defs>
<line x1="20.0" y1="230.0" x2="580.0" y2="230.0" class="stroke-muted" />
<line x1="60.0" y1="230.0" x2="60.0" y2="235.0" class="stroke-muted" />
<text x="60.0" y="248.0" class="fig-small" text-anchor="middle">-2</text>
<line x1="140.0" y1="230.0" x2="140.0" y2="235.0" class="stroke-muted" />
<text x="140.0" y="248.0" class="fig-small" text-anchor="middle">-1</text>
<line x1="220.0" y1="230.0" x2="220.0" y2="235.0" class="stroke-muted" />
<text x="220.0" y="248.0" class="fig-small" text-anchor="middle">0</text>
<line x1="300.0" y1="230.0" x2="300.0" y2="235.0" class="stroke-muted" />
<text x="300.0" y="248.0" class="fig-small" text-anchor="middle">1</text>
<line x1="380.0" y1="230.0" x2="380.0" y2="235.0" class="stroke-muted" />
<text x="380.0" y="248.0" class="fig-small" text-anchor="middle">2</text>
<line x1="460.0" y1="230.0" x2="460.0" y2="235.0" class="stroke-muted" />
<text x="460.0" y="248.0" class="fig-small" text-anchor="middle">3</text>
<line x1="540.0" y1="230.0" x2="540.0" y2="235.0" class="stroke-muted" />
<text x="540.0" y="248.0" class="fig-small" text-anchor="middle">4</text>
<ellipse cx="300.0" cy="125.0" rx="265.3" ry="84.7" class="stroke-a" fill="none"/>
<ellipse cx="300.0" cy="125.0" rx="196.0" ry="62.6" class="stroke-grid" fill="none"/>
<ellipse cx="300.0" cy="125.0" rx="123.9" ry="39.6" class="stroke-grid" fill="none"/>
<ellipse cx="300.0" cy="125.0" rx="62.0" ry="19.8" class="stroke-grid" fill="none"/>
<polyline points="220.0,205.8 228.0,125.0 235.2,125.0 241.7,125.0 247.5,125.0 252.8,125.0" fill="none" class="stroke-b" />
<circle cx="220.0" cy="205.8" r="3" class="fill-b"/>
<circle cx="228.0" cy="125.0" r="3" class="fill-b"/>
<circle cx="235.2" cy="125.0" r="3" class="fill-b"/>
<circle cx="241.7" cy="125.0" r="3" class="fill-b"/>
<circle cx="247.5" cy="125.0" r="3" class="fill-b"/>
<circle cx="252.8" cy="125.0" r="3" class="fill-b"/>
<circle cx="300.0" cy="125.0" r="4" class="fill-ink"/>
<text x="308.0" y="117.0" class="fig-small" text-anchor="start">bottom (1, 1)</text>
<text x="212.0" y="219.8" class="fig-small" text-anchor="end">start (0, 0)</text>
<text x="229.6" y="113.0" class="fig-small" text-anchor="start">walk</text>
<text x="24.0" y="30.0" class="fig-small" text-anchor="start">up: w2 grows (steep axis)</text>
<text x="576.0" y="222.0" class="fig-small" text-anchor="end">right: w1 grows (flat axis)</text>
</svg>
Figure 2. Each ellipse is a set of points with the same height, and the lowest height sits at the centre (1, 1). The ellipses are tight along w2 and stretched along w1, so the bowl is steep in w2 and flat in w1. The orange walk jumps up the steep wall in one step, then creeps along the floor toward the centre.
```

The same bowl in matrix form, which is the shape you code, is

$$
f(w) = \tfrac12\, w^\top A\, w - b^\top w
$$

Here $A$ is the matrix with 1 and 10 on its diagonal and 0 elsewhere, and $b=(1,10)$. Read it aloud as "half of $w$ times $A$ times $w$, minus $b$ dotted with $w$." The bottom is where $Aw=b$, which is $(1,1)$. In numpy:

```python
loss = 0.5 * (w @ A @ w) - b @ w  # one number
```

Least squares has this shape too, with $A=X^\top X$ and $b=X^\top y$, up to a constant that does not move the bottom. A realistic $A$ has nonzero off-diagonal entries, which tilts the ellipses. The diagonal $A$ here keeps the arithmetic small.

## Read the slope as a vector

With two weights you feel two slopes, one per axis. To feel the slope along $w_2$ at $(0,0)$, nudge $w_2$ by 1 in each direction and compare. The heights at $(0,1)$ and $(0,-1)$ are $-5$ and $5-(-10)=15$, so the slope is $(-5-15)/2=-10$. For $w_1$, the heights at $(1,0)$ and $(-1,0)$ are $-0.5$ and $1.5$, so the slope is $(-0.5-1.5)/2=-1$.

The two slopes together form the *gradient*, written $\nabla f$ and read "grad f". At $(0,0)$ it is $(-1,-10)$. Both entries are negative, so the height falls as either weight grows, and the gradient points uphill, toward smaller weights. Walking against it is walking downhill. It is also the steepest way down: for a small step, the change in height is the gradient dotted with the step, and among steps of equal length that dot product is most negative when the step points opposite the gradient.

For this bowl the gradient has a formula:

$$
\nabla f(w) = A\,w - b
$$

At $(0,0)$ it gives $-b=(-1,-10)$, which matches the nudges. Check a second point so it is not a coincidence. At $(1,0)$ the formula gives $(1,0)-(1,10)=(0,-10)$, and nudging gives $w_1$ slope $(0-0)/2=0$ and $w_2$ slope $(-5.5-14.5)/2=-10$. In numpy:

```python
grad = A @ w - b  # shape (d,)
```

Drop the minus sign on $b$ and the gradient at $(0,0)$ becomes $(1,10)$, so you walk uphill.

> [!KEY]
> The gradient is the vector of slopes, one per weight. It points uphill, so you step against it, and for the bowl it is $Aw-b$.

> [!TIP]
> To check any gradient, nudge one weight at a time: `(f(w + h * e) - f(w - h * e)) / (2 * h)`. Here `e` is 0 except for a 1 in one entry, and `h` is about `1e-6`. You write this as `numerical_gradient` in the exercise.

## Take downhill steps

Use $\alpha=0.1$ from $(0,0)$ and subtract $0.1$ times the gradient $(-1,-10)$. You land at $(0.1,1.0)$, where the height is $-5.095$, lower than 0. The gradient there is $(0.1-1,\,10-10)=(-0.9,0)$, so the next step lands at $(0.19,1.0)$ with height $-5.172$. Then $(0.271,1.0)$ with height $-5.234$.

Look at the two coordinates. The steep coordinate $w_2$ reached its bottom value 1 in one step. That is not luck. Its slope is 10 times its miss, and the step size 0.1 times 10 is exactly 1, so one step covers exactly the miss. The next chapter turns this into a rule. The flat coordinate $w_1$ creeps: its miss goes $1,\,0.9,\,0.81,\,0.729$. The walk is gradient descent, and the height never rises, because each step goes downhill.

In symbols, with $w_k$ for the point after $k$ steps:

$$
w_{k+1} = w_k - \alpha\, \nabla f(w_k)
$$

Read it aloud as "the next point is the current point, minus $\alpha$ times the gradient there." In numpy:

```python
w = w - step * grad(w)  # one step, shape (d,)
```

> [!WARNING]
> Gradient descent stores the whole path in the exercise, so row 0 must be the start `w0`, and the path has `iters + 1` rows. A path with `iters` rows fails the shape check.

> [!KEY]
> You can walk to the bottom of a loss without seeing it, because the gradient tells you which way is down, and stepping against it lowers the height. You walk instead of solving because a solve costs about $d^3$ and most losses have no formula, while a gradient costs about $d^2$ and always exists.

> [!RECAP]
> - The slope of a curve points uphill, so you step against it.
> - The gradient is the vector of slopes, one per weight. For the bowl it is $Aw-b$.
> - A step is $w-\alpha\nabla f(w)$, and it lowers the height when $\alpha$ is small enough.
> - A nudge in each direction checks any gradient.
> - A path has `iters + 1` rows, with row 0 equal to the start.

## Build it step by step

The checks call the functions in this order:

1. **`quadratic_loss_and_grad(A, b, w)`** takes `A` of shape `(d, d)` and `b` and `w` of shape `(d,)`. It returns a float loss and a gradient of shape `(d,)`:

   ```python
   # loss = 0.5 * (w @ A @ w) - b @ w, wrapped in float(...)
   # grad = A @ w - b
   ```

   Dropping the 0.5, or the minus sign on `b`, fails the hand values.

2. **`numerical_gradient(f, w, h=1e-6)`** takes a function `f` that maps a vector to a float, and returns an array of the same shape as `w`. Nudge one entry at a time:

   ```python
   # for each i: e = zeros_like(w); e[i] = 1
   # grad[i] = (f(w + h * e) - f(w - h * e)) / (2 * h)
   ```

   Change `w` in place and you corrupt the point for later entries, so build a fresh `w + h * e` each time.

3. **`gradient_descent(grad, w0, step, iters)`** takes a function `grad` that you call as `grad(w)`, and returns the path of shape `(iters + 1, d)`:

   ```python
   path = np.empty((iters + 1, len(w0)))
   # row 0 is w0, each later row is the previous row minus step * grad(previous row)
   ```

   Writing into `w0` changes the caller's array, so work on a copy.

## Where this breaks

**The step is too big.** The loss climbs or the path explodes. The next chapter, "How big a step? Step size and long valleys," shows exactly how big is too big.

**The step is too small.** The walk works but crawls, as the flat coordinate did. The chapter after that explains why.

**The loss is not a bowl.** On a bumpy loss, such as a neural network's, the walk ends in whichever valley is closest, and that valley need not be the lowest one.

## The exercise

You build `quadratic_loss_and_grad`, `numerical_gradient`, and `gradient_descent`.

The checks first compare the loss with the hand values $0$, $-5.5$, and $-5$ and the gradient with $(-1,-10)$. Then they compare your gradient with a finite difference at a random point, and your `numerical_gradient` with known gradients of a non-quadratic function. The last checks run `gradient_descent` and compare the path with the hand path $(0,0)$, $(0.1,1.0)$, $(0.19,1.0)$, verify that the loss falls at every step with step $1/10$, and verify that the path ends near $(1,1)$.
