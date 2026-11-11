---
title: "Approximation theory: how many units does a curve cost?"
slug: 035-approximation-theory
part: "Part 4 — Neural networks"
summary: How the error of a ReLU interpolant falls with the number of pieces, why grids die in high dimensions, and why depth can be exponentially cheaper.
minutes: 70
gpu: false
objectives:
  - Explain why the error of a piecewise-linear interpolant falls as the square of the knot spacing, and compute the pieces needed for a target error.
  - Count the cells a grid needs in d dimensions and explain the curse of dimensionality.
  - Build a sawtooth with 2^k pieces from k folds of a two-unit layer, and compare its unit count with a one-hidden-layer network.
  - State what Barron's dimension-free rate promises, and say why a network existing does not mean training finds it.
prereqs:
  - 034-multilayer-perceptrons
exercise: 035-approximation-theory
---

# Approximation theory: how many units does a curve cost?

A client asks for a network that draws a curve to within 0.001 at every point. The universal approximation theorem from the last chapter says a one-hidden-layer network can do that, and it stops there. It does not say whether the network needs 10 units, 10,000, or 10 billion. The budget decides whether the plan is a quick afternoon or a dead end, so "a network exists" is not yet an answer.

This chapter turns "can represent" into "how many units". You will measure how fast the error of a ReLU network falls as you add pieces. You will see why the same recipe becomes hopeless when the input has many coordinates, and why stacking layers can break the pattern. Along the way you will meet the two limits that no counting argument removes: a theorem about what exists is silent about what training finds.

## What you already know

- A ReLU unit is a hinge, flat until its reading crosses 0 and then rising at slope 1. A hidden layer of $m$ hinges makes at most $m+1$ straight pieces from one input.
- One hinge per segment builds the piecewise-linear interpolant of any knots, as in [Hinges reach any curve](/lessons/034-multilayer-perceptrons).
- The tent map folds $[0,1]$ onto itself, and $k$ folds make $2^k$ pieces from $2k$ units.
- The test error of a model is not its training error, and the gap grows with the model's freedom, as in [learning theory](/lessons/015-learning-theory).
- Gradient descent follows the slope downhill and can stop in a poor place, as in [gradient descent and momentum](/lessons/005-gradient-descent-and-momentum).

## The error of a chord

Draw a smooth curve and pick a few knots on it. The interpolant joins neighbouring knots with straight **chords**. Between two knots the curve bows away from its chord, and the gap is the error at that spot. Figure 1 shows $\sin x$ on $[0, \pi]$ with four segments. The gap is zero at every knot and largest near the middle of a segment.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="The sine curve on 0 to pi, four straight chords joining five points on it, and a red bar marking the largest gap, 0.070, between the curve and a chord.">
<path d="M60.0,190.0 L64.2,186.0 L68.4,182.1 L72.6,178.1 L76.8,174.2 L81.0,170.3 L85.2,166.3 L89.4,162.4 L93.6,158.6 L97.8,154.7 L102.0,150.9 L106.2,147.0 L110.4,143.3 L114.6,139.5 L118.8,135.8 L123.0,132.1 L127.2,128.5 L131.4,124.9 L135.6,121.4 L139.8,117.9 L144.0,114.4 L148.2,111.0 L152.4,107.7 L156.6,104.4 L160.8,101.2 L165.0,98.0 L169.2,94.9 L173.4,91.9 L177.6,88.9 L181.8,86.1 L186.1,83.2 L190.3,80.5 L194.5,77.8 L198.7,75.2 L202.9,72.7 L207.1,70.3 L211.3,68.0 L215.5,65.7 L219.7,63.5 L223.9,61.4 L228.1,59.4 L232.3,57.5 L236.5,55.7 L240.7,54.0 L244.9,52.4 L249.1,50.9 L253.3,49.4 L257.5,48.1 L261.7,46.9 L265.9,45.7 L270.1,44.7 L274.3,43.8 L278.5,42.9 L282.7,42.2 L286.9,41.6 L291.1,41.1 L295.3,40.6 L299.5,40.3 L303.7,40.1 L307.9,40.0 L312.1,40.0 L316.3,40.1 L320.5,40.3 L324.7,40.6 L328.9,41.1 L333.1,41.6 L337.3,42.2 L341.5,42.9 L345.7,43.8 L349.9,44.7 L354.1,45.7 L358.3,46.9 L362.5,48.1 L366.7,49.4 L370.9,50.9 L375.1,52.4 L379.3,54.0 L383.5,55.7 L387.7,57.5 L391.9,59.4 L396.1,61.4 L400.3,63.5 L404.5,65.7 L408.7,68.0 L412.9,70.3 L417.1,72.7 L421.3,75.2 L425.5,77.8 L429.7,80.5 L433.9,83.2 L438.2,86.1 L442.4,88.9 L446.6,91.9 L450.8,94.9 L455.0,98.0 L459.2,101.2 L463.4,104.4 L467.6,107.7 L471.8,111.0 L476.0,114.4 L480.2,117.9 L484.4,121.4 L488.6,124.9 L492.8,128.5 L497.0,132.1 L501.2,135.8 L505.4,139.5 L509.6,143.3 L513.8,147.0 L518.0,150.9 L522.2,154.7 L526.4,158.6 L530.6,162.4 L534.8,166.3 L539.0,170.3 L543.2,174.2 L547.4,178.1 L551.6,182.1 L555.8,186.0 L560.0,190.0" fill="none" class="stroke-a" stroke-width="2.4"/>
<path d="M60.0,190.0 L185.0,83.9 L310.0,40.0 L435.0,83.9 L560.0,190.0" fill="none" class="stroke-b" stroke-width="2.4"/>
<circle cx="60.0" cy="190.0" r="3.5" class="fill-ink"/>
<circle cx="185.0" cy="83.9" r="3.5" class="fill-ink"/>
<circle cx="310.0" cy="40.0" r="3.5" class="fill-ink"/>
<circle cx="435.0" cy="83.9" r="3.5" class="fill-ink"/>
<circle cx="560.0" cy="190.0" r="3.5" class="fill-ink"/>
<line x1="247.5" y1="51.4" x2="247.5" y2="62.0" class="stroke-bad" stroke-width="2.4"/>
<line x1="60" y1="190" x2="560" y2="190" class="stroke-muted"/>
<text x="60" y="210" text-anchor="middle" class="fig-small">0</text>
<text x="560" y="210" text-anchor="middle" class="fig-small">π</text>
<text x="310" y="26" text-anchor="middle" class="fig-bold fill-a">sin x</text>
<text x="470" y="96" text-anchor="start" class="fig-bold fill-b">4 chords</text>
<text x="247" y="34" text-anchor="middle" class="fig-small">gap 0.070</text>
</svg>
Figure 1. A chord lies below a bowed curve, and the gap is zero at the knots and largest between them. The red bar is that largest gap, 0.070, in the second segment, and it sits below the bound of 0.077 derived in this section.
```

How big is the gap? Over one segment of width $h$ the curve $g$ and its chord agree at both ends, and between them the curve bends at a rate set by its second derivative $g''$, the **curvature**. If the curvature were a constant $M$, the gap would be an arch with height $\tfrac{M}{2}\, u\,(h-u)$ at distance $u$ from the left knot. The arch peaks at the middle, $u = h/2$:

$$
\text{largest gap} \;\le\; \frac{M}{2}\cdot\frac{h}{2}\cdot\frac{h}{2} \;=\; \frac{h^2}{8}\, M, \qquad M = \max |g''| .
$$

Read it aloud: "the error is at most one eighth of the curvature times the spacing squared." The square matters. The spacing enters twice because a short segment is both short and straighter, since the curve has less distance in which to bow.

Check it by hand with $g(x) = x^2$ on $[0, 1]$, which has $g'' = 2$. Use two segments, so $h = 0.5$ and the knots are $0$, $0.5$, $1$ with heights $0$, $0.25$, $1$. At $x = 0.25$ the chord is halfway between 0 and 0.25, so it reads $0.125$, while the curve reads $0.25^2 = 0.0625$. The gap is $0.0625$. The bound gives $\tfrac{0.5^2}{8}\cdot 2 = 0.0625$. For a parabola the bound is exact, because its curvature really is constant.

For $\sin x$ the curvature is at most $M = 1$. With four segments $h = \pi/4 = 0.785$, so the bound is $0.0771$, and the measured largest gap is $0.0704$. The bound holds and is close, because $\sin$ bends hardest near the peak and less near the knots.

> [!KEY]
> A chord misses a smooth curve by at most $\tfrac{h^2}{8} M$, where $h$ is the knot spacing and $M$ is the largest curvature. Flat stretches cost almost nothing, and sharp bends cost the most.

## Halve the spacing, quarter the error

Because the error carries $h^2$, halving the spacing divides it by four. Figure 2 plots the measured largest error of $\sin x$ on $[0, \pi]$ for 2, 4, 8, up to 256 segments. The error falls from $0.2105$ to $0.0704$, $0.0188$, $0.00479$, and on to $0.000019$. The ratio between neighbours climbs from $2.99$ toward $4.00$, because the bound is tight only once the segments are short enough for the curvature to look constant.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Log-log plot of the largest interpolation error of sine against the number of segments, from 2 to 256. The points fall along a straight line of slope about minus 2, drawn dashed for comparison.">
<line x1="70" y1="30" x2="70" y2="190" class="stroke-muted"/>
<line x1="70" y1="190" x2="560" y2="190" class="stroke-muted"/>
<path d="M70.0,51.7 L140.0,66.9 L210.0,85.2 L280.0,104.2 L350.0,123.4 L420.0,142.7 L490.0,161.9 L560.0,181.2" fill="none" class="stroke-b" stroke-width="2"/>
<circle cx="70.0" cy="51.7" r="3.5" class="fill-b"/>  <!-- n=2 err=0.21051 -->
<circle cx="140.0" cy="66.9" r="3.5" class="fill-b"/>  <!-- n=4 err=0.07038 -->
<circle cx="210.0" cy="85.2" r="3.5" class="fill-b"/>  <!-- n=8 err=0.01885 -->
<circle cx="280.0" cy="104.2" r="3.5" class="fill-b"/>  <!-- n=16 err=0.00479 -->
<circle cx="350.0" cy="123.4" r="3.5" class="fill-b"/>  <!-- n=32 err=0.00120 -->
<circle cx="420.0" cy="142.7" r="3.5" class="fill-b"/>  <!-- n=64 err=0.00030 -->
<circle cx="490.0" cy="161.9" r="3.5" class="fill-b"/>  <!-- n=128 err=0.00008 -->
<circle cx="560.0" cy="181.2" r="3.5" class="fill-b"/>  <!-- n=256 err=0.00002 -->
<line x1="70.0" y1="51.7" x2="560.0" y2="186.5" class="stroke-muted dashed"/>
<text x="70.0" y="206" text-anchor="middle" class="fig-small">2</text>
<text x="140.0" y="206" text-anchor="middle" class="fig-small">4</text>
<text x="210.0" y="206" text-anchor="middle" class="fig-small">8</text>
<text x="280.0" y="206" text-anchor="middle" class="fig-small">16</text>
<text x="350.0" y="206" text-anchor="middle" class="fig-small">32</text>
<text x="420.0" y="206" text-anchor="middle" class="fig-small">64</text>
<text x="490.0" y="206" text-anchor="middle" class="fig-small">128</text>
<text x="560.0" y="206" text-anchor="middle" class="fig-small">256</text>
<text x="62" y="194.0" text-anchor="end" class="fig-small">1e-5</text>
<text x="62" y="162.0" text-anchor="end" class="fig-small">1e-4</text>
<text x="62" y="130.0" text-anchor="end" class="fig-small">1e-3</text>
<text x="62" y="98.0" text-anchor="end" class="fig-small">1e-2</text>
<text x="62" y="66.0" text-anchor="end" class="fig-small">1e-1</text>
<text x="62" y="34.0" text-anchor="end" class="fig-small">1</text>
<text x="315" y="226" text-anchor="middle" class="fig-small">segments n (log scale)</text>
<text x="300" y="62" text-anchor="start" class="fig-small">dashed: slope -2</text>
<text x="300" y="80" text-anchor="start" class="fig-small fill-b">fitted slope -1.95</text>
</svg>
Figure 2. On log-log axes the error falls along a straight line, and the line is parallel to the dashed slope of -2. Each doubling of the segments moves the point down by a factor of 4.
```

A straight line on log-log axes is a power law. If the error is $c\,n^{-p}$ for $n$ segments, then taking logs gives $\log(\text{error}) = \log c - p\,\log n$, a line whose slope is $-p$. Fitting a line to the eight points of Figure 2 gives a slope of $-1.945$. It falls just short of $-2$ because the two coarsest points are not yet in the regime where the bound is tight.

> [!WARNING]
> Fit the slope to the logarithms of both columns. A line fitted to the raw errors against the raw counts gives a slope of $-0.0003$ for these same points. That number says nothing about the power law, because the largest errors dominate the fit.

Turn the bound around to find a budget. For error at most $\varepsilon$, you need $\tfrac{h^2}{8} M \le \varepsilon$, so $h \le \sqrt{8\varepsilon / M}$. A curve over a length $L$ then needs

$$
n \;=\; \frac{L}{h} \;=\; L\sqrt{\frac{M}{8\varepsilon}} \quad \text{segments} .
$$

Read it aloud: "the number of pieces grows like one over the square root of the error." Each extra digit of accuracy, a tenfold smaller $\varepsilon$, costs a factor of $\sqrt{10} \approx 3.2$ in pieces. Take $\sin x$ on $[0,\pi]$, so $L = \pi$ and $M = 1$. For $\varepsilon = 0.001$ the formula gives $\pi\sqrt{125} = 35.1$, so 36 segments. A real interpolant with 36 segments has largest error $0.00095$, which is under the target. For $\varepsilon = 0.0001$ you need 112 segments, and for $\varepsilon = 0.01$ you need 12.

One hinge builds one segment, so those counts are also hidden-unit counts. The client's 0.001 is cheap here: a few dozen units.

```python
import numpy as np

def pieces_for_error(eps, curvature, length=1.0):
    # n = L * sqrt(M / (8 eps)), rounded up to a whole piece
    return int(np.ceil(length * np.sqrt(curvature / (8 * eps)) - 1e-9))
```

## The curse of dimensionality

The client's next request has ten inputs. The same plan applies: lay a grid over the input cube, and fit a flat piece in each cell. The error in each cell is governed by the spacing along each axis in the same way, so you still need $n = L\sqrt{M/(8\varepsilon)}$ divisions on every one of the $d$ axes. The number of cells multiplies across axes:

$$
\text{cells} \;=\; n^d \;=\; \left(L\sqrt{\frac{M}{8\varepsilon}}\right)^{d} \;\propto\; \varepsilon^{-d/2} .
$$

Read it aloud: "the exponent in the cost is half the dimension." The shape of this result is firm. The constant in front changes with how you measure the error and how many axes share the curvature, but the exponent does not. Table 1 uses $L = 1$ and $M = 8$ so that $n = 1/\sqrt{\varepsilon}$, which makes the numbers easy to check. At $\varepsilon = 0.01$ that is 10 divisions per axis.

| Dimension $d$ | Cells at $\varepsilon = 0.01$ | Cells at $\varepsilon = 0.0001$ |
|---|---|---|
| 1 | 10 | 100 |
| 2 | 100 | 10,000 |
| 3 | 1,000 | 1,000,000 |
| 5 | 100,000 | 10,000,000,000 |
| 10 | 10,000,000,000 | $10^{20}$ |

Ten billion cells, each with its own number to learn, for an error of 0.01 in ten dimensions. This is the **curse of dimensionality**: a method that treats every region of the input separately pays for a number of regions that grows exponentially with the dimension. Halving the spacing quarters the error in every dimension, but it also multiplies the cells by $2^d$, which is 1,024 at $d = 10$.

Two cautions keep this honest. First, the count is for a method that fits each cell on its own. A network with $m$ units in $d$ dimensions can cut the input into far more than $m$ regions, so the unit count and the cell count differ. Second, the curse is a statement about functions that are only smooth. A function that depends on three of its ten inputs, or on one direction in the input, needs far less. The curse bites worst when you know nothing about the target except its curvature.

> [!KEY]
> For a target that is only known to be smooth, a grid of pieces needs about $\varepsilon^{-d/2}$ cells. One input is cheap, and ten inputs are out of reach.

## Depth can be exponentially cheaper

The previous section counted the cost of building a curve one hinge at a time. Depth offers a different way to buy pieces, and the [multilayer perceptron chapter](/lessons/034-multilayer-perceptrons) showed it. Write the tent map as two hinges on $[0,1]$:

$$
T(x) = 2\operatorname{relu}(x) - 4\operatorname{relu}\!\left(x - \tfrac12\right) .
$$

Feed the output of $T$ into another copy of $T$, $k$ times, and the result is the **sawtooth** $S_k(x) = T(T(\cdots T(x)))$. It has $2^k$ straight pieces, and each fold costs two units, so the whole network holds $2k$ hidden units. Take $x = 0, 0.125, \dots, 1$. For $k = 2$ the heights are $0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0$, which is four pieces from four units.

What would a single hidden layer need for the same curve? Its $m$ units give $m$ hinges, and each hinge adds at most one bend, so it makes at most $m+1$ pieces. A sawtooth with $2^k$ pieces therefore needs $m \ge 2^k - 1$ units in one layer. Figure 3 compares the two counts on a log scale. At $k = 4$ the deep network uses 8 units and the shallow one 15. At $k = 10$ it is 20 units against 1,023, and at $k = 20$ it is 40 against 1,048,575.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Line chart on a log scale of hidden units against the number of folds k from 1 to 12. The deep network line rises slowly from 2 to 24 units, while the one-hidden-layer line rises from 1 to 4095 units.">
<line x1="60" y1="30" x2="60" y2="190" class="stroke-muted"/>
<line x1="60" y1="190" x2="560" y2="190" class="stroke-muted"/>
<path d="M60.0,176.7 L105.5,163.3 L150.9,155.5 L196.4,150.0 L241.8,145.7 L287.3,142.2 L332.7,139.2 L378.2,136.7 L423.6,134.4 L469.1,132.4 L514.5,130.5 L560.0,128.9" fill="none" class="stroke-a" stroke-width="2.4"/>
<path d="M60.0,190.0 L105.5,168.9 L150.9,152.6 L196.4,137.9 L241.8,123.9 L287.3,110.3 L332.7,96.8 L378.2,83.4 L423.6,70.0 L469.1,56.7 L514.5,43.3 L560.0,30.0" fill="none" class="stroke-b" stroke-width="2.4"/>
<circle cx="60.0" cy="176.7" r="3" class="fill-a"/><circle cx="60.0" cy="190.0" r="3" class="fill-b"/>
<text x="60.0" y="206" text-anchor="middle" class="fig-small">1</text>
<circle cx="105.5" cy="163.3" r="3" class="fill-a"/><circle cx="105.5" cy="168.9" r="3" class="fill-b"/>
<text x="105.5" y="206" text-anchor="middle" class="fig-small">2</text>
<circle cx="150.9" cy="155.5" r="3" class="fill-a"/><circle cx="150.9" cy="152.6" r="3" class="fill-b"/>
<text x="150.9" y="206" text-anchor="middle" class="fig-small">3</text>
<circle cx="196.4" cy="150.0" r="3" class="fill-a"/><circle cx="196.4" cy="137.9" r="3" class="fill-b"/>
<text x="196.4" y="206" text-anchor="middle" class="fig-small">4</text>
<circle cx="241.8" cy="145.7" r="3" class="fill-a"/><circle cx="241.8" cy="123.9" r="3" class="fill-b"/>
<text x="241.8" y="206" text-anchor="middle" class="fig-small">5</text>
<circle cx="287.3" cy="142.2" r="3" class="fill-a"/><circle cx="287.3" cy="110.3" r="3" class="fill-b"/>
<text x="287.3" y="206" text-anchor="middle" class="fig-small">6</text>
<circle cx="332.7" cy="139.2" r="3" class="fill-a"/><circle cx="332.7" cy="96.8" r="3" class="fill-b"/>
<text x="332.7" y="206" text-anchor="middle" class="fig-small">7</text>
<circle cx="378.2" cy="136.7" r="3" class="fill-a"/><circle cx="378.2" cy="83.4" r="3" class="fill-b"/>
<text x="378.2" y="206" text-anchor="middle" class="fig-small">8</text>
<circle cx="423.6" cy="134.4" r="3" class="fill-a"/><circle cx="423.6" cy="70.0" r="3" class="fill-b"/>
<text x="423.6" y="206" text-anchor="middle" class="fig-small">9</text>
<circle cx="469.1" cy="132.4" r="3" class="fill-a"/><circle cx="469.1" cy="56.7" r="3" class="fill-b"/>
<text x="469.1" y="206" text-anchor="middle" class="fig-small">10</text>
<circle cx="514.5" cy="130.5" r="3" class="fill-a"/><circle cx="514.5" cy="43.3" r="3" class="fill-b"/>
<text x="514.5" y="206" text-anchor="middle" class="fig-small">11</text>
<circle cx="560.0" cy="128.9" r="3" class="fill-a"/><circle cx="560.0" cy="30.0" r="3" class="fill-b"/>
<text x="560.0" y="206" text-anchor="middle" class="fig-small">12</text>
<text x="52" y="194.0" text-anchor="end" class="fig-small">1</text>
<text x="52" y="149.7" text-anchor="end" class="fig-small">10</text>
<text x="52" y="105.4" text-anchor="end" class="fig-small">100</text>
<text x="52" y="61.1" text-anchor="end" class="fig-small">1000</text>
<text x="310" y="226" text-anchor="middle" class="fig-small">folds k (the sawtooth has 2^k pieces)</text>
<text x="90" y="50" text-anchor="start" class="fig-bold fill-b">one hidden layer: 2^k - 1</text>
<text x="404" y="168" text-anchor="start" class="fig-bold fill-a">deep: 2k</text>
</svg>
Figure 3. Hidden units needed for a sawtooth with 2^k pieces, on a log scale. The deep count grows by 2 per fold and stays low, and the one-hidden-layer count doubles with each fold and climbs a straight line on this chart. At k = 12 it is 24 units against 4,095.
```

This is a **depth separation**: a function that a deep network builds cheaply and a shallow one cannot. Telgarsky (2016) proved a stronger version. A network with few layers and few units cannot only fail to match the sawtooth exactly. It must also miss it by a large error, on average over the interval, when it has far fewer than $2^k$ pieces. So the separation survives approximation, and a shallow net cannot get close with a handful of extra units.

The mechanism is the one from the last chapter: a layer can fold what the previous layer made. A shallow layer is a sum of hinges over the whole input, so every new hinge adds one bend. A deep layer reads the zigzag the earlier layer made, so a single hinge in layer two bends once in each of the zigzag's stretches.

> [!WARNING]
> The tent must map $[0,1]$ onto $[0,1]$. A tent scaled to peak at 2, such as $4\min(x, 1-x)$, sends the first fold's output into $[0,2]$. The two-hinge formula for $T$ is a fold only on $[0,1]$, so above 1 it keeps falling and the second fold's output goes negative. The piece count falls off $2^k$, and the values leave $[0,1]$. Check the maximum of $S_k$ as well as its piece count.

Count the pieces of a sawtooth numerically by looking at slopes. Sample $f$ on a fine grid, compute the slope of each small step, and count how often the slope changes. The pieces are the changes plus one:

```python
def count_pieces(f, xs, tol=1e-6):
    slopes = np.diff(f(xs)) / np.diff(xs)
    # a piece boundary is any step where the slope jumps
    changes = np.abs(np.diff(slopes)) > tol
    return 1 + int(changes.sum())
```

For $S_k$ on 4097 evenly spaced points the counter returns $2, 4, 8, \dots, 256$ for $k = 1, \dots, 8$. The grid must contain the bends. The sawtooth's bends sit at multiples of $1/2^k$, and $4097 = 2^{12} + 1$ points include all of them for $k \le 12$. A bend that falls between two grid points looks like a short extra piece, so the count would be too high.

> [!KEY]
> A hinge adds one bend, and a fold multiplies the bends already there. That is why depth can be exponentially cheaper than width for functions built by repeated folding.

## What the theorems leave out

Two results sit on the edge of this chapter, and both come with caveats worth carrying with you.

The first rescues high dimensions for some targets. Barron (1993) showed that if a function's Fourier transform has a finite first moment, a quantity written $C_f$ that grows with how quickly the function wiggles, then a one-hidden-layer network with $m$ sigmoid units reaches a squared error of order $C_f^2/m$ on a ball, in any dimension. The rate in $m$ does not depend on $d$. Compare that with the grid, where reaching squared error $\varepsilon$ costs $\varepsilon^{-d/2}$ cells. The catch lives in $C_f$. For some smooth functions it grows quickly with $d$, so the constant can absorb the benefit, and the condition asks for more smoothness as $d$ grows. It describes functions that have a compact description in a Fourier sense. It does not describe every smooth function.

The second caveat is bigger. Every result in this chapter says that a network with a given number of units exists. None says that training finds it.

> [!WARNING]
> Representability is not learnability. Finding the weights is an optimisation problem, and training rarely finds sawtooth weights even though they exist. See [SGD, Adam, and learning-rate schedules](/lessons/038-sgd-adam-and-schedules). Generalising from a finite sample is a separate question, and [learning theory](/lessons/015-learning-theory) is where it is answered. A network big enough to fit the training points can still miss the points it never saw.

> [!RECAP]
> - A piecewise-linear interpolant with knots $h$ apart misses a smooth curve by at most $\tfrac{h^2}{8}M$, so halving $h$ quarters the error.
> - Reaching error $\varepsilon$ in one dimension takes about $L\sqrt{M/(8\varepsilon)}$ pieces, so each digit of accuracy costs a factor of about 3.2.
> - A grid in $d$ dimensions needs about $\varepsilon^{-d/2}$ cells, which is out of reach at ten inputs.
> - $k$ folds of the two-unit tent map make $2^k$ pieces from $2k$ units, while one hidden layer needs $2^k - 1$ units.
> - Barron's rate is dimension-free in the number of units, but a constant hides the dimension, and no theorem here says training finds the weights.

## Build it step by step

You build the pieces in the order the checks test them. Every function takes plain numpy arrays or numbers.

1. **`tent(x)`** takes an array of numbers in $[0,1]$ and returns $2\operatorname{relu}(x) - 4\operatorname{relu}(x - 0.5)$, built with `np.maximum` so it is the two-unit layer. The mistake people most often make is a tent that peaks at 2, which looks right on one fold and fails on the second.
2. **`sawtooth(x, k)`** applies `tent` $k$ times and returns an array of the same shape. With $k = 0$ it returns `x`. Loop, and feed each output to the next fold.
3. **`count_pieces(f, xs, tol=1e-6)`** takes a function and a sorted grid, and returns the number of straight pieces as an `int`. Use `np.diff` for the slopes, and `np.diff` again for the changes. People forget the `+ 1` that turns boundaries into pieces.
4. **`deep_units_used(k)`** and **`shallow_units_needed(k)`** take a number of folds and return `2 * k` and `2 ** k - 1`. These are exact integers, and the second is one less than the piece count.
5. **`interp_error(f, n_knots, a=0.0, b=1.0)`** takes a vectorised function and a count of evenly spaced knots on $[a, b]$, builds the chord interpolant with `np.interp`, and returns the largest absolute gap on a fine grid of a few thousand points. A coarse check grid hides the error in the middle of each segment, so test at many more points than knots.
6. **`loglog_slope(ns, errs)`** returns the slope of the best line through $(\log n, \log \text{error})$, using `np.polyfit(..., 1)`. The mistake is to fit the raw values. A slope near zero from your own call is the sign.
7. **`pieces_for_error(eps, curvature, length=1.0)`** returns $\lceil L\sqrt{M/(8\varepsilon)}\rceil$ as an `int`. Floating-point error can push an exact answer such as $10.0000000001$ up to 11, so subtract a tiny tolerance before rounding up.
8. **`grid_points_needed(eps, d, curvature=1.0)`** returns the cells for a $d$-dimensional grid, which is `pieces_for_error(eps, curvature) ** d`. It returns a Python `int`, so $10^{10}$ does not overflow.

## Where this breaks

The counting here assumes a target with bounded curvature. A target with a kink, such as $|x - 0.3|$, has infinite curvature at the kink, and the $h^2$ rate drops to $h$ unless a knot lands on the bend. Interpolation with evenly spaced knots does not know where the bends are, so adaptive knots, which cluster near the sharp parts, do better.

The sawtooth is the worst case for a shallow network and a convenient case for a deep one. Real targets rarely look like a perfect fold, so treat depth separation as proof that depth can help a lot, and not as a measurement of how much it helps on your data.

## The exercise

You implement the eight functions above in `starter.py`. The checks compare `tent` and `sawtooth` with hand values and require every output to stay inside $[0,1]$, count the pieces of a line, a tent, and a hand-built zigzag, and confirm that the sawtooth has $2^k$ pieces for $k = 1$ to 8. They compare the unit counts with $2k$ and $2^k - 1$, measure the interpolation error of $x^2$ and $\sin x$ against the $\tfrac{h^2}{8}M$ bound, and fit the log-log slope on an exact power law and on the measured errors of $\sin x$. The last checks compute the pieces for a target error and the cells of a $d$-dimensional grid.

## Further reading

- The [6.S898 Deep Learning, Fall 2023 schedule](https://phillipi.github.io/6.s898/) lists "Approximation theory" on Thursday 9/14, and the [MIT OCW 6.7960 Fall 2024 notes](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/lists/lecture-notes/) title the topic "Lecture 3: Approximation Theory".
- Matus Telgarsky (2016), "Benefits of depth in neural networks", proves the depth separation for the sawtooth, including the approximation version.
- Andrew Barron (1993), "Universal approximation bounds for superpositions of a sigmoidal function", *IEEE Transactions on Information Theory* 39(3), gives the dimension-free rate.
- George Cybenko (1989), "Approximation by superpositions of a sigmoidal function", *Mathematics of Control, Signals and Systems* 2, proves that one hidden layer is enough.
