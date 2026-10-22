---
title: "Momentum: a heavy ball and a look-ahead"
slug: 012-momentum
part: "Part 1 — Foundations"
summary: Keep part of your last step to speed up along the floor and damp the bounce, so steps fall from about kappa to about its square root.
minutes: 40
gpu: false
objectives:
  - Write the heavy-ball update and explain why a remembered step speeds up the flat axis and damps the steep one.
  - Compute the best step and momentum for a bowl with known curvatures, and explain why the step may exceed 2/L.
  - Write Nesterov's look-ahead update and say what it measures differently.
  - Measure how many fewer steps momentum needs on a long valley.
prereqs:
  - 009-walking-downhill
  - 010-how-big-a-step
  - 011-long-valleys
exercise: 012-momentum
---

# Momentum: a heavy ball and a look-ahead

The valley floor is nearly flat, and you are crawling along it. You cannot take a bigger step, because the steep walls would throw you across the valley and the walk would blow up. Yet look at what the walk does on the floor: it moves right, right, right, each step about the same size as the last. It repeats the same move hundreds of times and never notices.

Can the walk use what it just did? A ball rolling down a valley does. It keeps going in the direction it was already moving, so it gathers speed along the floor, and it does not swing wildly at the walls, because the bounce from one wall cancels part of the bounce from the other.

This chapter turns that picture into one extra term in the update, and then into a second version that glances ahead before it measures the slope.

## What you already know

Five ideas carry this chapter:

- A step moves from $w$ to $w-\alpha\nabla f(w)$ (see "Walking downhill: slope, gradient, and one step").
- On each axis a step multiplies the miss by $1-\alpha\lambda$, and plain descent needs $\alpha<2/L$ (see "How big a step: curvature and the 2/L limit").
- The condition number is $\kappa=L/\mu$, and plain descent needs about $\kappa$ times more steps for a stretched bowl (see "Condition number: why long valleys are slow").
- On the running bowl, the flat coordinate moves $0.1,\,0.09,\,0.081,\,\dots$ at step $0.1$: the same direction every time.
- A step is a difference between two points: the last step is $w_k-w_{k-1}$.

## A ball keeps its speed

Add a fraction of the last step to the slope step. The fraction is the *momentum* $\beta$ (beta, said "bay-ta"). The ball's velocity is the last step, friction is the part of it you drop, and the slope is a push.

Take the running bowl, with step $\alpha=0.1$ and $\beta=0.5$. These are sample values, not tuned ones. At the start there is no last step, so the first step is the plain one, to $(0.1,\,1.0)$. At that point the slope is $(-0.9,\,0)$, so the push is $(0.09,\,0)$. Half of the last step $(0.1,\,1.0)$ is $(0.05,\,0.5)$. Add them, and the step is $(0.14,\,0.5)$, which lands you at $(0.24,\,1.5)$. Plain descent would land at $(0.19,\,1.0)$.

Follow one more step. The slope at $(0.24,\,1.5)$ is $(-0.76,\,5)$, so the push is $(0.076,\,-0.5)$. Half of the last step $(0.14,\,0.5)$ is $(0.07,\,0.25)$. The step is $(0.146,\,-0.25)$, and you land at $(0.386,\,1.25)$.

Look at the two axes. On the flat axis the steps are $0.1,\,0.14,\,0.146$, growing, because each push adds to a step already pointing the right way. On the steep axis the coordinate went $1.0,\,1.5,\,1.25$: it overshot the floor, and the push came back and cancelled part of the speed. Figure 1 follows both axes for 30 steps.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Each coordinate minus its bottom value for heavy ball and for plain descent">
<defs></defs>
<line x1="70.0" y1="270.0" x2="560.0" y2="270.0" class="stroke-muted" />
<line x1="70.0" y1="30.0" x2="70.0" y2="270.0" class="stroke-grid" />
<line x1="70.0" y1="123.3" x2="560.0" y2="123.3" class="stroke-muted dashed" />
<line x1="70.0" y1="270.0" x2="70.0" y2="275.0" class="stroke-muted" />
<text x="70.0" y="290.0" class="fig-small" text-anchor="middle">0</text>
<line x1="151.7" y1="270.0" x2="151.7" y2="275.0" class="stroke-muted" />
<text x="151.7" y="290.0" class="fig-small" text-anchor="middle">5</text>
<line x1="233.3" y1="270.0" x2="233.3" y2="275.0" class="stroke-muted" />
<text x="233.3" y="290.0" class="fig-small" text-anchor="middle">10</text>
<line x1="315.0" y1="270.0" x2="315.0" y2="275.0" class="stroke-muted" />
<text x="315.0" y="290.0" class="fig-small" text-anchor="middle">15</text>
<line x1="396.7" y1="270.0" x2="396.7" y2="275.0" class="stroke-muted" />
<text x="396.7" y="290.0" class="fig-small" text-anchor="middle">20</text>
<line x1="478.3" y1="270.0" x2="478.3" y2="275.0" class="stroke-muted" />
<text x="478.3" y="290.0" class="fig-small" text-anchor="middle">25</text>
<line x1="560.0" y1="270.0" x2="560.0" y2="275.0" class="stroke-muted" />
<text x="560.0" y="290.0" class="fig-small" text-anchor="middle">30</text>
<text x="62.0" y="260.7" class="fig-small" text-anchor="end">-1</text>
<text x="62.0" y="194.0" class="fig-small" text-anchor="end">-0.5</text>
<text x="62.0" y="127.3" class="fig-small" text-anchor="end">0</text>
<text x="62.0" y="60.7" class="fig-small" text-anchor="end">0.5</text>
<text x="315.0" y="312.0" class="fig-small" text-anchor="middle">step number</text>
<polyline points="70.0,256.7 86.3,243.3 102.7,231.3 119.0,220.5 135.3,210.8 151.7,202.1 168.0,194.2 184.3,187.1 200.7,180.7 217.0,175.0 233.3,169.8 249.7,165.2 266.0,161.0 282.3,157.2 298.7,153.8 315.0,150.8 331.3,148.0 347.7,145.6 364.0,143.3 380.3,141.3 396.7,139.5 413.0,137.9 429.3,136.5 445.7,135.2 462.0,134.0 478.3,132.9 494.7,131.9 511.0,131.1 527.3,130.3 543.7,129.6 560.0,129.0" fill="none" class="stroke-a dashed" />
<polyline points="70.0,256.7 86.3,243.3 102.7,224.7 119.0,205.2 135.3,187.3 151.7,171.9 168.0,159.4 184.3,149.5 200.7,142.0 217.0,136.3 233.3,132.2 249.7,129.3 266.0,127.2 282.3,125.8 298.7,124.8 315.0,124.2 331.3,123.8 347.7,123.5 364.0,123.4 380.3,123.3 396.7,123.3 413.0,123.3 429.3,123.3 445.7,123.3 462.0,123.3 478.3,123.3 494.7,123.3 511.0,123.3 527.3,123.3 543.7,123.3 560.0,123.3" fill="none" class="stroke-a" />
<polyline points="70.0,256.7 86.3,123.3 102.7,56.7 119.0,90.0 135.3,140.0 151.7,148.3 168.0,127.5 184.3,112.9 200.7,116.0 217.0,124.9 233.3,127.8 249.7,124.8 266.0,121.8 282.3,121.9 298.7,123.3 315.0,124.1 331.3,123.7 347.7,123.1 364.0,123.1 380.3,123.3 396.7,123.4 413.0,123.4 429.3,123.3 445.7,123.3 462.0,123.3 478.3,123.4 494.7,123.3 511.0,123.3 527.3,123.3 543.7,123.3 560.0,123.3" fill="none" class="stroke-b" />
<circle cx="70.0" cy="256.7" r="2.5" class="fill-b"/>
<circle cx="86.3" cy="123.3" r="2.5" class="fill-b"/>
<circle cx="102.7" cy="56.7" r="2.5" class="fill-b"/>
<circle cx="119.0" cy="90.0" r="2.5" class="fill-b"/>
<circle cx="135.3" cy="140.0" r="2.5" class="fill-b"/>
<circle cx="151.7" cy="148.3" r="2.5" class="fill-b"/>
<circle cx="168.0" cy="127.5" r="2.5" class="fill-b"/>
<circle cx="184.3" cy="112.9" r="2.5" class="fill-b"/>
<circle cx="200.7" cy="116.0" r="2.5" class="fill-b"/>
<circle cx="217.0" cy="124.9" r="2.5" class="fill-b"/>
<circle cx="233.3" cy="127.8" r="2.5" class="fill-b"/>
<circle cx="249.7" cy="124.8" r="2.5" class="fill-b"/>
<circle cx="266.0" cy="121.8" r="2.5" class="fill-b"/>
<circle cx="282.3" cy="121.9" r="2.5" class="fill-b"/>
<circle cx="298.7" cy="123.3" r="2.5" class="fill-b"/>
<circle cx="315.0" cy="124.1" r="2.5" class="fill-b"/>
<circle cx="331.3" cy="123.7" r="2.5" class="fill-b"/>
<circle cx="347.7" cy="123.1" r="2.5" class="fill-b"/>
<circle cx="364.0" cy="123.1" r="2.5" class="fill-b"/>
<circle cx="380.3" cy="123.3" r="2.5" class="fill-b"/>
<circle cx="396.7" cy="123.4" r="2.5" class="fill-b"/>
<circle cx="413.0" cy="123.4" r="2.5" class="fill-b"/>
<circle cx="429.3" cy="123.3" r="2.5" class="fill-b"/>
<circle cx="445.7" cy="123.3" r="2.5" class="fill-b"/>
<circle cx="462.0" cy="123.3" r="2.5" class="fill-b"/>
<circle cx="478.3" cy="123.4" r="2.5" class="fill-b"/>
<circle cx="494.7" cy="123.3" r="2.5" class="fill-b"/>
<circle cx="511.0" cy="123.3" r="2.5" class="fill-b"/>
<circle cx="527.3" cy="123.3" r="2.5" class="fill-b"/>
<circle cx="543.7" cy="123.3" r="2.5" class="fill-b"/>
<circle cx="560.0" cy="123.3" r="2.5" class="fill-b"/>
<line x1="250.0" y1="46.0" x2="280.0" y2="46.0" class="stroke-b" />
<text x="288.0" y="50.0" class="fig-small" text-anchor="start">steep axis, heavy ball (plain descent sits on 0 after step 1)</text>
<line x1="250.0" y1="66.0" x2="280.0" y2="66.0" class="stroke-a" />
<text x="288.0" y="70.0" class="fig-small" text-anchor="start">flat axis, heavy ball</text>
<line x1="250.0" y1="86.0" x2="280.0" y2="86.0" class="stroke-a dashed" />
<text x="288.0" y="90.0" class="fig-small" text-anchor="start">flat axis, plain descent</text>
<text x="20.0" y="20.0" class="fig-small" text-anchor="start">coordinate minus its bottom value</text>
</svg>
Figure 1. Each coordinate minus its bottom value, with step 0.1 and momentum 0.5, over 30 steps. On the flat axis (blue) heavy ball closes the gap faster than plain descent (dashed). On the steep axis (orange) it overshoots at step 2, comes back, and the swings shrink, because each return step cancels part of the stored momentum.
```

In symbols, with $w_{k-1}$ for the point before $w_k$:

$$
w_{k+1} = w_k - \alpha\,\nabla f(w_k) + \beta\,(w_k - w_{k-1})
$$

Read it aloud as "step against the slope, then keep a fraction $\beta$ of the last step." This is the *heavy-ball* method. Treat $w_{-1}$ as $w_0$, so the first step is a plain one. In numpy:

```python
w_next = w - step * grad(w) + momentum * (w - w_prev)
```

What does $\beta$ do at the extremes? At $\beta=0$ the extra term vanishes and you have plain descent. Near $\beta=1$ the ball has almost no friction, so it keeps almost all of its speed and overshoots the bottom again and again.

There is a reason the flat axis gains. When every step points the same way, the remembered steps add up: $1+\beta+\beta^2+\cdots=1/(1-\beta)$. With $\beta=0.5$, the same push acts as a step twice as long. On the steep axis the steps alternate in sign, so the remembered part opposes the next push and the swings shrink.

> [!KEY]
> Heavy ball keeps a fraction of the last step. Steps that agree add up, and steps that alternate cancel.

> [!WARNING]
> Compute `w_next` from the old `w` and `w_prev` first, and only then set `w_prev, w = w, w_next`. If you update `w_prev` first, the momentum term is 0.

## The best settings for a bowl

You can pick $\alpha$ and $\beta$ by hand, but for a bowl there is a best pair, and the reason is a short argument. On one axis with curvature $\lambda$, write $m$ for the miss. The heavy-ball rule says $m_{\text{next}}=(1-\alpha\lambda)\,m+\beta\,(m-m_{\text{prev}})$, which regroups as

$$
m_{\text{next}} = (1+\beta-\alpha\lambda)\,m - \beta\,m_{\text{prev}}
$$

Read it aloud as "the new miss is a multiple of the old miss, minus $\beta$ times the miss before." When $\alpha\lambda$ lies between $(1-\sqrt\beta)^2$ and $(1+\sqrt\beta)^2$, the misses shrink by exactly $\sqrt\beta$ per step, with a little swinging, and the shrink does not depend on $\lambda$. So choose $\alpha$ and $\beta$ so that this window just covers every axis: $\alpha\mu=(1-\sqrt\beta)^2$ and $\alpha L=(1+\sqrt\beta)^2$. Then the flat axis and the steep axis shrink at the same rate, and neither one holds the other back. Solving the two equations for $\alpha$ and $\beta$ gives

$$
\alpha = \frac{4}{(\sqrt{L}+\sqrt{\mu})^2},\qquad \beta = \left(\frac{\sqrt{L}-\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}}\right)^2
$$

Read them aloud as "four over the squared sum of the square roots" and "the squared ratio of their difference to their sum." For $L=10$ and $\mu=1$: $\sqrt{10}\approx3.162$, the sum with 1 is $4.162$, so $\alpha\approx0.231$. The difference is $2.162$, and $(2.162/4.162)^2\approx0.270$ is $\beta$. In numpy:

```python
root_L, root_mu = np.sqrt(L), np.sqrt(mu)
step = 4 / (root_L + root_mu) ** 2
momentum = ((root_L - root_mu) / (root_L + root_mu)) ** 2
```

The step $0.231$ is above $2/L=0.2$. That does not break the earlier rule, because the rule is for plain descent. With momentum, a bowl is stable while $\alpha\lambda<2(1+\beta)$ on every axis, so here the limit is $2\times1.27/10\approx0.254$. Momentum raises the longest safe step.

With these settings every axis shrinks by $\sqrt\beta=(\sqrt\kappa-1)/(\sqrt\kappa+1)$ per step, about $1-2/\sqrt{\kappa}$ on a stretched bowl, in place of plain descent's $1-1/\kappa$. The number of steps falls from about $\kappa$ to about $\sqrt{\kappa}$. For $\kappa=10$, $\sqrt{0.270}\approx0.52$ per step, against 0.9.

## Look ahead, then measure

Heavy ball measures the slope where you stand, then adds the momentum. The ball can do better. It already knows momentum will carry it forward, so it can glance ahead to where it is headed and measure the slope there.

That is *Nesterov's method*. From $(0.1,\,1.0)$, with the last step $(0.1,\,1.0)$ and $\beta=0.5$, the look-ahead point is $y=(0.1,\,1.0)+(0.05,\,0.5)=(0.15,\,1.5)$. The slope at $y$ is $(0.15-1,\,15-10)=(-0.85,\,5)$. Step against it from $y$ with $\alpha=0.1$ and you land at $(0.15+0.085,\,1.5-0.5)=(0.235,\,1.0)$. Heavy ball, with the same numbers, landed at $(0.24,\,1.5)$.

```figure
<svg viewBox="0 0 600 360" role="img" aria-label="From the point (0.1, 1.0), heavy ball and Nesterov take different second steps">
<defs><marker id="c012-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c012-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c012-arrow-d" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-d"/></marker></defs>
<line x1="200.0" y1="330.0" x2="352.0" y2="330.0" class="stroke-muted" />
<line x1="200.0" y1="330.0" x2="200.0" y2="7.0" class="stroke-grid" />
<line x1="200.0" y1="330.0" x2="200.0" y2="335.0" class="stroke-muted" />
<text x="200.0" y="350.0" class="fig-small" text-anchor="middle">0</text>
<line x1="238.0" y1="330.0" x2="238.0" y2="335.0" class="stroke-muted" />
<text x="238.0" y="350.0" class="fig-small" text-anchor="middle">0.1</text>
<line x1="276.0" y1="330.0" x2="276.0" y2="335.0" class="stroke-muted" />
<text x="276.0" y="350.0" class="fig-small" text-anchor="middle">0.2</text>
<line x1="314.0" y1="330.0" x2="314.0" y2="335.0" class="stroke-muted" />
<text x="314.0" y="350.0" class="fig-small" text-anchor="middle">0.3</text>
<line x1="352.0" y1="330.0" x2="352.0" y2="335.0" class="stroke-muted" />
<text x="352.0" y="350.0" class="fig-small" text-anchor="middle">0.4</text>
<text x="366.0" y="350.0" class="fig-small" text-anchor="start">w1</text>
<line x1="195.0" y1="254.0" x2="200.0" y2="254.0" class="stroke-muted" />
<text x="190.0" y="258.0" class="fig-small" text-anchor="end">1.0</text>
<line x1="195.0" y1="64.0" x2="200.0" y2="64.0" class="stroke-muted" />
<text x="190.0" y="68.0" class="fig-small" text-anchor="end">1.5</text>
<text x="190.0" y="20.0" class="fig-small" text-anchor="end">w2</text>
<line x1="238.0" y1="254.0" x2="257.0" y2="64.0" class="stroke-d" marker-end="url(#c012-arrow-d)"/>
<line x1="257.0" y1="64.0" x2="289.3" y2="254.0" class="stroke-a" marker-end="url(#c012-arrow-a)"/>
<line x1="238.0" y1="254.0" x2="272.2" y2="254.0" class="stroke-b" marker-end="url(#c012-arrow-b)"/>
<line x1="272.2" y1="254.0" x2="291.2" y2="64.0" class="stroke-b" marker-end="url(#c012-arrow-b)"/>
<circle cx="238.0" cy="254.0" r="5" class="fill-ink"/>
<circle cx="257.0" cy="64.0" r="5" class="fill-d"/>
<circle cx="289.3" cy="254.0" r="5" class="fill-a"/>
<circle cx="291.2" cy="64.0" r="5" class="fill-b"/>
<text x="208.0" y="278.0" class="fig-small" text-anchor="start">(0.1, 1.0) you are here</text>
<text x="247.0" y="52.0" class="fig-small" text-anchor="end">(0.15, 1.5) look-ahead</text>
<text x="363.4" y="68.0" class="fig-small" text-anchor="start">orange: heavy ball lands at (0.24, 1.5)</text>
<text x="363.4" y="258.0" class="fig-small" text-anchor="start">blue: Nesterov lands at (0.235, 1.0)</text>
<text x="363.4" y="155.0" class="fig-small" text-anchor="start">purple: momentum carries you ahead</text>
<text x="363.4" y="171.0" class="fig-small" text-anchor="start">blue: slope measured at the look-ahead</text>
</svg>
Figure 2. Both methods start from the same point and carry the same momentum, (0.05, 0.5). Heavy ball (orange) measures the slope where you stand, which points almost straight along the floor, so it ends up high at (0.24, 1.5). Nesterov (purple, then blue) first moves to the look-ahead point and measures the slope there. That slope points steeply back down, so the step undoes the overshoot and lands at (0.235, 1.0).
```

The look-ahead point sits up the steep wall, where the slope is large and points back down. So Nesterov brakes before the overshoot rather than after it. In symbols:

$$
y_k = w_k + \beta\,(w_k - w_{k-1}),\qquad w_{k+1} = y_k - \alpha\,\nabla f(y_k)
$$

Read it aloud as "move ahead by the momentum, then step against the slope measured there." Again $w_{-1}=w_0$. In numpy:

```python
y = w + momentum * (w - w_prev)
w_next = y - step * grad(y)
```

Take the slope at `w` and not at `y`, and you get the heavy-ball step. A usual choice for Nesterov is $\alpha=1/L$ and $\beta=(\sqrt{L}-\sqrt{\mu})/(\sqrt{L}+\sqrt{\mu})$, which for the running bowl is about $0.519$. This $\beta$ is not squared.

On a bowl, both methods need about $\sqrt{\kappa}$ steps. They differ off a bowl: Nesterov's step count has a guarantee for every smooth convex loss, and heavy ball's guarantee is proved for bowls only.

## Count the payoff

Take a bowl with $\kappa=100$, so $\sqrt{\kappa}=10$, and measure the steps to bring the miss to one millionth of its start. Plain descent at step $1/L$ needs 1341 steps. Heavy ball with the best settings needs 93 steps, about 14 times fewer. Nesterov needs 155, about 9 times fewer. Neither ratio is exactly $\sqrt{\kappa}$, so read $\sqrt{\kappa}$ as the scale.

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
Figure 3. Same valley (condition number 100), same start, three walks, each stopped when its distance reaches one millionth of the start. Gradient descent (grey) is still above 10^-2 after 400 steps and needs 1341. Heavy ball (orange) needs 93: it first swings out, then falls fastest. Nesterov (purple) needs 155 and does not swing out as far.
```

Momentum costs one stored previous point, and each step still needs one gradient. It trades a little memory for a lot of steps.

> [!KEY]
> A walk that remembers its last step speeds up along the flat floor, because consistent steps add up, and damps the bounce between steep walls, because alternating steps cancel. On a long valley that cuts the steps from about $\kappa$ to about $\sqrt{\kappa}$, so the crawl on the floor gets faster without a bigger step on the walls.

> [!RECAP]
> - Heavy ball adds $\beta(w_k-w_{k-1})$ to the slope step. Treat $w_{-1}$ as $w_0$.
> - The remembered steps add up along the floor and cancel across the walls.
> - The best settings are $\alpha=4/(\sqrt{L}+\sqrt{\mu})^2$ and the squared ratio for $\beta$. With momentum the step may exceed $2/L$.
> - Nesterov measures the slope at $y=w+\beta(w-w_{\text{prev}})$.
> - Both take about $\sqrt{\kappa}$ steps on a bowl.

## Build it step by step

The starter includes `gradient_descent`, which you wrote in an earlier chapter. The checks call the functions you write in this order:

1. **`heavy_ball(grad, w0, step, momentum, iters)`** returns the path of shape `(iters + 1, d)`. Keep two copies of `w0`, named `w` and `w_prev`:

   ```python
   w_next = w - step * grad(w) + momentum * (w - w_prev)
   # then: w_prev, w = w, w_next
   ```

   Updating `w_prev` first makes the momentum term 0 and turns the method into plain descent.

2. **`nesterov(grad, w0, step, momentum, iters)`** returns the path. Use the same two copies, plus a look-ahead point:

   ```python
   y = w + momentum * (w - w_prev)
   w_next = y - step * grad(y)
   ```

   Calling `grad(w)` in place of `grad(y)` is the heavy-ball rule, and the hand check catches it.

3. **`best_heavy_ball_params(mu, L)`** returns the tuple `(step, momentum)` from the two formulas, as floats:

   ```python
   # step = 4 / (sqrt(L) + sqrt(mu)) ** 2
   # momentum = ((sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu))) ** 2
   ```

   Forgetting the final square on `momentum` gives 0.519 where 0.270 is right.

## Where this breaks

**Too much momentum.** With $\beta$ near 1 the ball overshoots again and again, and the loss swings. If the loss oscillates, lower $\beta$.

**Unknown curvatures.** The best settings need $\mu$ and $L$. For a real loss you tune $\beta$ by watching the loss, with 0.9 a usual starting point.

**A step that is too big still blows up.** Momentum raises the limit to $2(1+\beta)/L$. It does not remove it.

**The loss is not a bowl.** On a bumpy loss, the ball can roll through a small dip, which is often what you want, but it can also roll past a good valley.

## The exercise

You build `heavy_ball`, `nesterov`, and `best_heavy_ball_params`. The starter supplies `gradient_descent`.

The checks first run `heavy_ball` on the running bowl and compare it with the hand path $(0,0)$, $(0.1,1.0)$, $(0.24,1.5)$, $(0.386,1.25)$ and with the exact recursion on a tilted matrix. They run `nesterov` and compare it with the hand step to $(0.235,1.0)$. They verify that both methods with $\beta=0$ equal plain descent, and that `best_heavy_ball_params` gives $(0.231,0.270)$ for $L=10$ and $\mu=1$. A further check confirms that with those settings the flat axis and the steep axis both shrink by about $\sqrt\beta$ per step. The last checks measure the speedup on a bowl with $\kappa=100$: heavy ball must reach one millionth of the starting miss in far fewer steps than plain descent, and so must Nesterov.
