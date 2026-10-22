---
title: "The widest road and the hinge loss"
slug: 032-widest-road-hinge-loss
part: "Part 2 — Classical models"
summary: Among all lines that separate two groups, pick the one with the widest empty road, and train it by following the slope of the hinge loss.
minutes: 40
gpu: false
objectives:
  - Explain why the widest empty road is the safest of the lines that separate two classes.
  - Compute the margin z = y(w·x + b), and say why the road's half-width is 1/‖w‖.
  - Compare the 0/1, squared, logistic, and hinge losses, and say what C trades off.
  - Write the hinge objective and its subgradient, and check the slope with finite differences.
  - Train a linear SVM by subgradient descent.
prereqs:
  - 013-ridge-penalty-and-closed-form
  - 018-log-loss
  - 019-gradient-descent-for-logistic
  - 031-naive-bayes
exercise: 032-widest-road-hinge-loss
---

# The widest road and the hinge loss

Fifteen lines separate these two groups of houses perfectly, so which one should you build the road along? The map in Figure 1 shows twelve houses. Circles sit in the west and squares in the east. The town wants a road that keeps every house on its own side, and it wants a computer to find the best one.

Every one of those lines makes zero mistakes on the houses you can see, so mistakes cannot choose between them. A line that hugs the circles is a poor choice. A house built later a little east of the circles, or a surveyor's error of a few metres, drops it on the wrong side. You want the line with the most room to spare on both sides.

Logistic regression does not settle this question for you. On separable data its weights grow forever, and the line it heads towards is the one this chapter finds, but only slowly and with no stopping point. Here you write the goal down directly: a number for "room to spare", a loss that cares only about points near the road, and a gradient descent loop that trains it.

## What you already know

These five ideas carry the chapter:

- A linear classifier scores a point with $w^{\top}x + b$ and predicts the sign of the score. The weights $w$ tilt the line, and $b$ slides it.
- Gradient descent steps against the slope of a loss, and the step size sets how far you move.
- $\lVert w \rVert$ is the length of $w$. The vector $[3, 4]$ has length 5.
- The ridge penalty $\tfrac12\lVert w \rVert^2$ keeps weights small, as in "Ridge: a fee on large weights".
- The log loss from "How wrong is a probability? The log loss" never reaches zero, so separable data pushes logistic weights to grow without limit.

## Many lines, one road

Look at Figure 1. Picture a road laid along any separating line and widened until it touches a house on each side. A line that hugs the circles gives a skinny road, because the road runs out of room almost at once. The orange line gives the fattest road, and only three houses, the ringed ones, touch its edges. Move any other house a little and the road stays put. Because the closest houses decide the road, the machine only has to care about points near the boundary.

```figure
<svg viewBox="0 0 600 352" role="img" aria-label="Twelve houses in two groups, fifteen thin lines that all separate them, and one wide shaded road with dashed edges and three ringed houses touching the edges">
<rect x="30" y="12" width="540" height="324" class="fill-paper stroke-grid"/>
<line x1="379.2" y1="336" x2="185.8" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="570" y1="86.8" x2="30" y2="267.1" class="stroke-muted" stroke-width="1"/>
<line x1="397.6" y1="336" x2="180" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="209.7" y1="336" x2="301" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="343.2" y1="336" x2="340.2" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="164.4" y1="336" x2="426.6" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="570" y1="105.6" x2="30" y2="185.1" class="stroke-muted" stroke-width="1"/>
<line x1="394.9" y1="336" x2="186.9" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="161.3" y1="336" x2="310.1" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="570" y1="179.3" x2="30" y2="151.3" class="stroke-muted" stroke-width="1"/>
<line x1="570" y1="37.3" x2="30" y2="333.7" class="stroke-muted" stroke-width="1"/>
<line x1="570" y1="31" x2="30" y2="285.5" class="stroke-muted" stroke-width="1"/>
<line x1="570" y1="161.7" x2="30" y2="191.3" class="stroke-muted" stroke-width="1"/>
<line x1="36.8" y1="336" x2="484.1" y2="12" class="stroke-muted" stroke-width="1"/>
<line x1="65.4" y1="336" x2="405.5" y2="12" class="stroke-muted" stroke-width="1"/>
<polygon points="42.3,336 278.4,336 514,12 277.9,12" class="fill-soft-b"/>
<line x1="160.3" y1="336" x2="396" y2="12" class="stroke-b"/>
<line x1="278.4" y1="336" x2="514" y2="12" class="stroke-b dashed"/>
<line x1="42.3" y1="336" x2="277.9" y2="12" class="stroke-b dashed"/>
<circle cx="192" cy="93" r="5.5" class="fill-a"/>
<circle cx="175.8" cy="152.4" r="5.5" class="fill-a"/><circle cx="175.8" cy="152.4" r="10" class="stroke-ink"/>
<circle cx="165" cy="157.8" r="5.5" class="fill-a"/>
<circle cx="197.4" cy="44.4" r="5.5" class="fill-a"/>
<circle cx="165" cy="141.6" r="5.5" class="fill-a"/>
<circle cx="219" cy="93" r="5.5" class="fill-a"/><circle cx="219" cy="93" r="10" class="stroke-ink"/>
<rect x="461.9" y="244.1" width="11" height="11" class="fill-c"/>
<rect x="456.5" y="168.5" width="11" height="11" class="fill-c"/>
<rect x="386.3" y="222.5" width="11" height="11" class="fill-c"/>
<rect x="353.9" y="265.7" width="11" height="11" class="fill-c"/>
<rect x="359.3" y="211.7" width="11" height="11" class="fill-c"/><circle cx="364.8" cy="217.2" r="10" class="stroke-ink"/>
<rect x="386.3" y="190.1" width="11" height="11" class="fill-c"/>
<text x="111" y="260.4" class="fig-small">z = −1 edge</text>
<text x="478.2" y="87.6" class="fig-small">z = +1 edge</text>
<text x="321.6" y="30.9" class="fig-bold" text-anchor="middle">widest road</text>
<text x="310.8" y="319.8" class="fig-small">15 thin separating lines</text>
<text x="300" y="346" class="fig-small" text-anchor="middle">east (100 m)</text>
</svg>
Figure 1. Every thin grey line splits the circles from the squares, so training error cannot choose between them. The shaded road is the widest empty strip. Its centre line stays farthest from the nearest house on each side, and only the three ringed houses touch its edges.
```

To compute this, you need a number for each house that says how safely it sits on its own side.

## One number for "right side"

The score $w^{\top}x + b$ is positive on one side of the line and negative on the other. A circle house should be on the negative side and a square house on the positive side. So multiply the score by the label $y$, which is $-1$ for circles and $+1$ for squares. The product is positive when the house is on its correct side, for both classes:

$$
z_i = y_i(w^{\top}x_i + b) .
$$

Read it aloud: take the score, and flip its sign for circles, so that "positive" always means "correct". This chapter calls $z_i$ the *margin* of house $i$. The road's half-width, a distance on the map, is a separate quantity that you meet in a moment.

Take $w = [1, 0]$, $b = 0$, and three points. The point $[2, 0]$ with label $+1$ scores 2, so $z = 2$. The point $[0.5, 0]$ with label $+1$ scores 0.5, so $z = 0.5$. The point $[-1, 0]$ with label $-1$ scores $-1$, and $(-1)\times(-1) = 1$, so $z = 1$. In numpy this is one line, `z = y * (X @ w + b)`. A common mistake is to use labels $0$ and $1$. The third point then scores $-1$ with label 0, and the product is 0 instead of 1.

## Why the edges sit at z = 1

Why does the chapter keep asking for $z \ge 1$ and not $z \ge 0$? Because the scale of $w$ is free. Doubling $w$ and $b$ doubles every $z$ but leaves the line untouched. With $z \ge 0$ as the rule, nothing would stop you from making every $z$ as large as you like. So you fix the scale: demand $z \ge 1$ for every house, and let the closest houses sit at exactly $z = 1$.

The width follows. The distance from a point to the line is $\lvert w^{\top}x + b \rvert / \lVert w \rVert$, so a house with $z = 1$ sits at distance $1/\lVert w \rVert$. With $w = [1, 0]$ the edge is 1 unit from the line. Halve $w$ to $[0.5, 0]$, and the same line has $z = 1$ at $x = 2$, which is 2 units away, exactly $1/0.5$.

> [!KEY]
> The road's half-width is $1/\lVert w \rVert$. A wide road means a small $w$, which is the "keep the weights small" idea from ridge.

So the goal becomes: make $\lVert w \rVert$ as small as you can, while every house keeps $z \ge 1$. The chapter minimises $\tfrac12\lVert w \rVert^2$ instead of $\lVert w \rVert$. Both are smallest at the same $w$, and the square has a plain slope, $w$ itself, which you need for descent.

## Why not count the mistakes?

You need a loss that charges a house for falling short of $z = 1$, so that gradient descent can act on it. The obvious first try is to count mistakes: charge 1 when $z \le 0$ and 0 otherwise. Figure 2 shows why this fails. The count is flat almost everywhere. Nudge $w$ slightly and no house changes sides, so the slope is zero and gradient descent has nowhere to go.

Squared loss, $(1 - z)^2$, has a slope, but it charges a house with $z = 3$ a value of 4. That house is safely correct, and the loss still drags the line towards it. The logistic loss $\ln(1 + e^{-z})$ also has a slope, but it never reaches zero. A house with $z = 3$ still costs 0.049, so every house, however far away, keeps tugging on the line.

```figure
<svg viewBox="0 0 600 318" role="img" aria-label="Three loss curves against z: the 0/1 loss, the logistic loss, and the hinge loss, with the hinge flat at zero beyond z equals 1">
<rect x="50" y="16" width="520" height="250" class="fill-paper stroke-grid"/>
<line x1="50" y1="266" x2="50" y2="16" class="stroke-grid"/>
<text x="50" y="291" class="fig-small" text-anchor="middle">−2</text>
<line x1="154" y1="266" x2="154" y2="16" class="stroke-grid"/>
<text x="154" y="291" class="fig-small" text-anchor="middle">−1</text>
<line x1="258" y1="266" x2="258" y2="16" class="stroke-grid"/>
<text x="258" y="291" class="fig-small" text-anchor="middle">0</text>
<line x1="362" y1="266" x2="362" y2="16" class="stroke-grid"/>
<text x="362" y="291" class="fig-small" text-anchor="middle">1</text>
<line x1="466" y1="266" x2="466" y2="16" class="stroke-grid"/>
<text x="466" y="291" class="fig-small" text-anchor="middle">2</text>
<line x1="570" y1="266" x2="570" y2="16" class="stroke-grid"/>
<text x="570" y="291" class="fig-small" text-anchor="middle">3</text>
<line x1="50" y1="266" x2="570" y2="266" class="stroke-grid"/>
<text x="42.7" y="270.2" class="fig-small" text-anchor="end">0</text>
<line x1="50" y1="182.7" x2="570" y2="182.7" class="stroke-grid"/>
<text x="42.7" y="186.8" class="fig-small" text-anchor="end">1</text>
<line x1="50" y1="99.3" x2="570" y2="99.3" class="stroke-grid"/>
<text x="42.7" y="103.5" class="fig-small" text-anchor="end">2</text>
<line x1="50" y1="16" x2="570" y2="16" class="stroke-grid"/>
<text x="42.7" y="20.2" class="fig-small" text-anchor="end">3</text>
<line x1="50" y1="182.7" x2="258" y2="182.7" class="stroke-muted"/>
<line x1="258" y1="266" x2="570" y2="266" class="stroke-muted"/>
<polyline points="50,88.8 55.2,92.4 60.4,96.1 65.6,99.7 70.8,103.3 76,106.8 81.2,110.4 86.4,113.9 91.6,117.3 96.8,120.8 102,124.2 107.2,127.6 112.4,131 117.6,134.3 122.8,137.6 128,140.8 133.2,144.1 138.4,147.2 143.6,150.4 148.8,153.5 154,156.6 159.2,159.6 164.4,162.6 169.6,165.5 174.8,168.4 180,171.3 185.2,174.1 190.4,176.8 195.6,179.5 200.8,182.2 206,184.8 211.2,187.4 216.4,189.9 221.6,192.4 226.8,194.8 232,197.2 237.2,199.5 242.4,201.8 247.6,204 252.8,206.1 258,208.2 263.2,210.3 268.4,212.3 273.6,214.3 278.8,216.2 284,218 289.2,219.8 294.4,221.6 299.6,223.2 304.8,224.9 310,226.5 315.2,228 320.4,229.5 325.6,231 330.8,232.4 336,233.8 341.2,235.1 346.4,236.3 351.6,237.6 356.8,238.8 362,239.9 367.2,241 372.4,242.1 377.6,243.1 382.8,244.1 388,245 393.2,245.9 398.4,246.8 403.6,247.6 408.8,248.4 414,249.2 419.2,250 424.4,250.7 429.6,251.4 434.8,252 440,252.6 445.2,253.3 450.4,253.8 455.6,254.4 460.8,254.9 466,255.4 471.2,255.9 476.4,256.4 481.6,256.8 486.8,257.2 492,257.6 497.2,258 502.4,258.4 507.6,258.8 512.8,259.1 518,259.4 523.2,259.7 528.4,260 533.6,260.3 538.8,260.6 544,260.8 549.2,261.1 554.4,261.3 559.6,261.5 564.8,261.7 570,262" class="stroke-d"/>
<polyline points="50,16 55.2,20.2 60.4,24.3 65.6,28.5 70.8,32.7 76,36.8 81.2,41 86.4,45.2 91.6,49.3 96.8,53.5 102,57.7 107.2,61.8 112.4,66 117.6,70.2 122.8,74.3 128,78.5 133.2,82.7 138.4,86.8 143.6,91 148.8,95.2 154,99.3 159.2,103.5 164.4,107.7 169.6,111.8 174.8,116 180,120.2 185.2,124.3 190.4,128.5 195.6,132.7 200.8,136.8 206,141 211.2,145.2 216.4,149.3 221.6,153.5 226.8,157.7 232,161.8 237.2,166 242.4,170.2 247.6,174.3 252.8,178.5 258,182.7 263.2,186.8 268.4,191 273.6,195.2 278.8,199.3 284,203.5 289.2,207.7 294.4,211.8 299.6,216 304.8,220.2 310,224.3 315.2,228.5 320.4,232.7 325.6,236.8 330.8,241 336,245.2 341.2,249.3 346.4,253.5 351.6,257.7 356.8,261.8 362,266 367.2,266 372.4,266 377.6,266 382.8,266 388,266 393.2,266 398.4,266 403.6,266 408.8,266 414,266 419.2,266 424.4,266 429.6,266 434.8,266 440,266 445.2,266 450.4,266 455.6,266 460.8,266 466,266 471.2,266 476.4,266 481.6,266 486.8,266 492,266 497.2,266 502.4,266 507.6,266 512.8,266 518,266 523.2,266 528.4,266 533.6,266 538.8,266 544,266 549.2,266 554.4,266 559.6,266 564.8,266 570,266" class="stroke-b"/>
<circle cx="466" cy="266" r="5" class="fill-b"/>
<circle cx="310" cy="224.3" r="5" class="fill-b"/>
<circle cx="362" cy="266" r="5" class="fill-b"/>
<circle cx="392" cy="40" r="5" class="fill-b"/><text x="402" y="44" class="fig-small">point 1: z = 2, charge 0</text>
<circle cx="392" cy="60" r="5" class="fill-b"/><text x="402" y="64" class="fig-small">point 2: z = 0.5, charge 0.5</text>
<circle cx="392" cy="80" r="5" class="fill-b"/><text x="402" y="84" class="fig-small">point 3: z = 1, charge 0</text>
<text x="60.4" y="174.3" class="fig-small">0/1 loss</text>
<text x="133.2" y="36.8" class="fig-bold">hinge</text>
<text x="60.4" y="145.2" class="fig-small">logistic</text>
<text x="310" y="309" text-anchor="middle">z = y (w·x + b)</text>
</svg>
Figure 2. The 0/1 loss is flat everywhere except one jump, so it gives no slope to follow. The logistic loss stays above zero forever, so a far-away point keeps pulling. The hinge falls in a straight line and reaches exactly 0 at z = 1, which is why only points on or inside the road matter. Its three orange dots are the worked example, listed in the key.
```

You want a loss that is exactly zero once a house is safely outside the road. That is the *hinge loss*, the shortfall from the edge:

$$
\max(0,\; 1 - z) .
$$

For the three example points, the charges are $\max(0, 1-2) = 0$, $\max(0, 1-0.5) = 0.5$, and $\max(0, 1-1) = 0$. Only the house inside the road pays.

## When a house does not fit: slack and C

A road with $z \ge 1$ for every house may not exist. One house in the wrong place makes the demand impossible. So let a house fall short, and charge for the shortfall. The shortfall of house $i$, $\max(0, 1 - z_i)$, is its *slack*. It is how far the house pokes into the road, or across it. A price $C$ per unit of slack gives the soft-margin objective:

$$
J(w, b) = \frac{1}{2}\lVert w \rVert^2 + C\sum_{i=1}^{n}\max(0,\, 1 - z_i) .
$$

Read it aloud: a fee for a narrow road, plus $C$ for every unit that houses poke into it. The first term wants a small $w$. The second wants every $z$ to reach 1. For the three example points with $C = 1$, the first term is $\tfrac12(1) = 0.5$ and the slack total is $0 + 0.5 + 0 = 0.5$, so $J = 1$. In numpy:

```python
J = 0.5 * w @ w + C * np.sum(np.maximum(0.0, 1.0 - z))
```

> [!WARNING]
> The hinge term is a sum over houses, not a mean. With `np.mean` the three example points give $0.5 + 0.5/3 \approx 0.667$ instead of 1. The hinge part, and its part of the slope, come out $n$ times too small, and the $\lVert w \rVert^2$ term stays the same.

Figure 3 shows what $C$ does. The same nine points include one circle that sits among the squares. With $C = 0.1$, poking is cheap. The road stays wide, with half-width 2.02, and the outlier ends on the wrong side at a small price. With $C = 10$, poking is a hundred times dearer. The road tilts and shrinks to half-width 0.47, and the outlier sits exactly on its edge.

```figure
<svg viewBox="0 0 600 226" role="img" aria-label="Two panels with the same nine points and one outlier. With a small C the road is wide and the outlier is ignored. With a large C the road is narrow and tilted to fit the outlier">
<rect x="10" y="12" width="280" height="175" class="fill-paper stroke-grid"/>
<polygon points="104.1,187 248.8,187 209.9,12 65.2,12" class="fill-soft-b"/>
<line x1="176.4" y1="187" x2="137.6" y2="12" class="stroke-b"/>
<line x1="248.8" y1="187" x2="209.9" y2="12" class="stroke-b dashed" stroke-width="1.5"/>
<line x1="104.1" y1="187" x2="65.2" y2="12" class="stroke-b dashed" stroke-width="1.5"/>
<circle cx="45" cy="47" r="5" class="fill-a"/>
<circle cx="62.5" cy="117" r="5" class="fill-a"/>
<circle cx="80" cy="78.5" r="5" class="fill-a"/>
<circle cx="94" cy="141.5" r="5" class="fill-a"/>
<rect x="215" y="52.5" width="10" height="10" class="fill-c"/>
<rect x="232.5" y="115.5" width="10" height="10" class="fill-c"/>
<rect x="208" y="84" width="10" height="10" class="fill-c"/>
<rect x="250" y="143.5" width="10" height="10" class="fill-c"/>
<circle cx="181.5" cy="99.5" r="5" class="fill-a"/><circle cx="181.5" cy="99.5" r="9.5" class="stroke-ink"/>

<text x="150" y="210" class="fig-small" text-anchor="middle">C = 0.1: ‖w‖ = 0.50, half-width 1/‖w‖ = 2.02</text>
<rect x="310" y="12" width="280" height="175" class="fill-paper stroke-grid"/>
<polygon points="510.7,187 545.7,187 487.3,12 452.3,12" class="fill-soft-b"/>
<line x1="528.2" y1="187" x2="469.8" y2="12" class="stroke-b"/>
<line x1="545.7" y1="187" x2="487.3" y2="12" class="stroke-b dashed" stroke-width="1.5"/>
<line x1="510.7" y1="187" x2="452.3" y2="12" class="stroke-b dashed" stroke-width="1.5"/>
<circle cx="345" cy="47" r="5" class="fill-a"/>
<circle cx="362.5" cy="117" r="5" class="fill-a"/>
<circle cx="380" cy="78.5" r="5" class="fill-a"/>
<circle cx="394" cy="141.5" r="5" class="fill-a"/>
<rect x="515" y="52.5" width="10" height="10" class="fill-c"/>
<rect x="532.5" y="115.5" width="10" height="10" class="fill-c"/>
<rect x="508" y="84" width="10" height="10" class="fill-c"/>
<rect x="550" y="143.5" width="10" height="10" class="fill-c"/>
<circle cx="481.5" cy="99.5" r="5" class="fill-a"/><circle cx="481.5" cy="99.5" r="9.5" class="stroke-ink"/>

<text x="450" y="210" class="fig-small" text-anchor="middle">C = 10: ‖w‖ = 2.11, half-width 1/‖w‖ = 0.47</text>
</svg>
Figure 3. Same points, two prices. On the left, poking into the road is cheap, so the road stays wide and the ringed outlier ends up on the wrong side at a small price. On the right, every unit of poking costs a hundred times more, so the road tilts and shrinks until the outlier sits exactly on its edge.
```

> [!KEY]
> $C$ is the exchange rate between a wide road and slack. A small $C$ buys width and tolerates outliers. A large $C$ buys fit and lets one odd house bend the road.

## Follow the slope

The hinge has a corner at $z = 1$. Left of it the slope is $-1$, and right of it the slope is 0. At the corner, any value from $-1$ to 0 is a valid slope, and each is called a *subgradient*. Pick 0, so a house at exactly $z = 1$ counts as inactive. A house with $z < 1$ is *active*. For an active house the charge is $1 - y_i(w^{\top}x_i + b)$, whose slope in $w$ is $-y_i x_i$ and in $b$ is $-y_i$. Inactive houses add nothing:

$$
\frac{\partial J}{\partial w} = w - C\sum_{i:\, z_i < 1} y_i x_i, \qquad
\frac{\partial J}{\partial b} = -C\sum_{i:\, z_i < 1} y_i .
$$

Read it aloud: the road pulls $w$ towards zero, and each active house pushes the road away from itself. In the example only the second point is active. The slope in $w$ is $[1, 0] - 1\cdot[0.5, 0] = [0.5, 0]$, and the slope in $b$ is $-1$. In numpy, `grad_w = w - C * (y[z < 1] @ X[z < 1])`. A common mistake is `z <= 1`. The third point then joins, and the slope in $w$ becomes $[-0.5, 0]$.

This slope is already enough to train. Start at $w = 0$, $b = 0$, where every house has $z = 0$ and pays 1, so $J = 12$ for the twelve houses with $C = 1$. Step against the slope with a step size that shrinks, $\text{lr}/(1 + \text{lr}\cdot t)$ at step $t$. After 3000 steps, $J$ falls to 0.162, the road's half-width is 1.76, and the smallest margin is slightly above 1. The best possible values are $J = 0.160$ and half-width 1.77. So subgradient descent gives you a working linear SVM, and the next chapters ask a different question about the same road, because that question makes curved boundaries possible.

> [!KEY]
> Among the lines that separate your houses, take the one with the widest empty road. In numbers: keep $z_i \ge 1$ with the smallest $\lVert w \rVert$, charge slack with the hinge, and descend. Only houses on or inside the road push, so a few houses decide the line, which answers the opening question.

## Where this breaks

**Unscaled features.** With coordinates around 5 and 3, $b$ moves slowly, because shifting the line a little changes every score. Centre your features before you train.

**A large C.** Each active house contributes $C$ to the slope, so a step of 0.3 that suits $C = 1$ overshoots for $C = 100$. Lower the first step as you raise $C$.

**Noisy labels.** A large $C$ lets one mislabelled house bend the road, as in Figure 3. You choose $C$ by holding data back, as in "Honest scores: training error, test error, and holding data back".

> [!RECAP]
> - $z_i = y_i(w^{\top}x_i + b)$ is positive when house $i$ is on its correct side, and 1 means it rests on the road's edge.
> - The road's half-width is $1/\lVert w \rVert$, so a wide road is a small $w$.
> - The hinge $\max(0, 1-z)$ is exactly zero beyond the edge, and $C$ prices each unit of slack.
> - Only houses with $z < 1$ enter the subgradient, and the hinge is a sum, not a mean.

## Build it step by step

You write three functions, in the order the checks test them.

1. **`functional_margins(w, b, X, y)`**. `w` is `(d,)`, `X` is `(n, d)`, and `y` is `(n,)` with labels in $\{-1, +1\}$. It returns the `(n,)` array of margins. Use one matrix product and one elementwise product. The common mistake is labels in $\{0, 1\}$.

2. **`hinge_loss_and_subgradient(w, b, X, y, C)`**. It returns `(loss, grad_w, grad_b)`. Reuse `functional_margins`, build the mask with `z < 1`, and apply it to rows of `X` and entries of `y`. The shape of the code is:

```python
z = functional_margins(w, b, X, y)   # shape (n,)
active = z < 1.0                     # strictly less than
grad_w = w - C * (y[active] @ X[active])   # shape (d,)
```

   The common mistakes are `z <= 1` and `np.mean`.

3. **`fit_subgradient(X, y, C, lr, steps)`**. Start from zeros, call the second function in a loop, and subtract both slopes scaled by `lr / (1 + lr * t)`. It returns `(w, b)`. The common mistake is to update `w` first and then compute the slope for `b` from the changed `w`. Compute both slopes before either update.

## The exercise

You build `functional_margins`, `hinge_loss_and_subgradient`, and `fit_subgradient`. The checks confirm margins of 2, 0.5, and 1 on the worked example, then a loss of exactly 1 and a slope of `[0.5, 0]` and $-1$, which catches a mean and a `<=`. The slope is also compared with finite differences away from the corner. Finally the checks train on the twelve houses of Figure 1 and ask for every house on its correct side, an objective within 5 percent of the best value, and a road within 10 percent of the best width.
