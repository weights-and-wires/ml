---
title: "Principal components: compress and know what you lost"
slug: 028-principal-components
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: The SVD of centred data lists every direction of spread. Keeping the top k loses the squared singular values you drop; whitening evens the rest.
minutes: 80
gpu: false
objectives:
  - Centre a cloud, turn a line into one score per point (a shadow), and compute the variance of those scores.
  - Explain why the line with the largest score variance is also the line that loses the least.
  - Read components, scores, and explained variances from the SVD of the centred data, and from the covariance table.
  - Rebuild data from k components, and show that the error equals the sum of the squared singular values you dropped.
  - Pick k from the explained variance ratios or an error budget, and standardise columns when units differ.
  - Whiten data with PCA and ZCA, and say what `eps` does for tiny eigenvalues.
prereqs:
  - 003-least-squares-and-projection
  - 027-gradient-boosting
  - 004-stable-least-squares
exercise: 028-principal-components
---

# Principal components: compress and know what you lost

Every model so far had an answer to learn from. This chapter has none. You hold a table of points and want to describe it with fewer numbers. Picture thousands of dust grains lying in a thin pancake of flour on a tilted tray. Each grain has three coordinates, but the pancake is almost flat, so two numbers (where on the pancake) say nearly everything. Fewer numbers means smaller files, plots you can look at, less noise, and faster models.

Which two numbers? Dropping a coordinate fails: on a tilted tray every axis cuts across the pancake, so dropping one throws away real information. You want the pancake's own sheet. And once you squeeze, a second question follows: the rebuilt rows are not the originals, so how wrong are they, and could a different choice of directions have done better?

You build the sheet one line at a time. First you find the best single line, and see that "keep the points spread out" and "lose as little of them as possible" pick the same line. Then one call to the SVD hands you every direction in order, with no search. Finally you rebuild the data from the top few directions and learn the exact size of the error before you rebuild anything.

## What you already know

- A row of a table is a point, and `X.mean(axis=0)` averages down each column.
- The dot product of a point with a unit vector (a vector of length 1) is how far along that vector the point reaches. The chapter on [least squares and projection](/lessons/003-least-squares-and-projection) drew it as a shadow.
- Variance is the average of squared gaps from the mean, and in a right triangle the two short sides squared add to the long side squared.
- The chapter on [stable least squares](/lessons/004-stable-least-squares) split a matrix as $X = U S V^{\top}$, with orthonormal columns in $U$ and $V$ and singular values $s_1 \ge s_2 \ge \cdots \ge 0$ on the diagonal of $S$.

## Centre the cloud

The running example has three points with two readings each: $(2, 0)$, $(0, 2)$, and $(4, 4)$. Spread means how far points sit from the middle of the cloud. The column means are $(2+0+4)/3 = 2$ and $(0+2+4)/3 = 2$, so the middle is $(2, 2)$. Subtract it from every row and the points become $(0, -2)$, $(-2, 0)$, and $(2, 2)$. That shift is called **centring**. It moves the middle of the cloud to the origin, so a line through the origin is a line through the middle.

Without it, the measurement goes wrong. Along $(1,1)/\sqrt{2}$ the raw squared readings are 2, 2, and 32, summing to 36, mostly the cloud's distance from the origin. After centring, the sum is 12.

```python
mean = X.mean(axis=0)
Xc = X - mean
```

## A score is a shadow

Shine a light at right angles onto a line through the middle, and each point casts a shadow on it. Its **score** is the signed distance of that shadow from the middle. For a unit vector $u$ pointing along the line, the score of a point is its dot product with $u$. Figure 1 shows two lines through the middle of the example.

```figure
<svg viewBox="0 0 600 316" role="img" aria-label="Three points with their shadows on two lines through the mean: line u gets long shadows and short gaps, line w the reverse">
<rect x="10.0" y="12.0" width="270.0" height="270.0" class="stroke-grid" fill="none"/>
<line x1="52.2" y1="282.0" x2="52.2" y2="12.0" class="stroke-grid" />
<line x1="10.0" y1="239.8" x2="280.0" y2="239.8" class="stroke-grid" />
<line x1="94.4" y1="282.0" x2="94.4" y2="12.0" class="stroke-grid" />
<line x1="10.0" y1="197.6" x2="280.0" y2="197.6" class="stroke-grid" />
<line x1="136.6" y1="282.0" x2="136.6" y2="12.0" class="stroke-grid" />
<line x1="10.0" y1="155.4" x2="280.0" y2="155.4" class="stroke-grid" />
<line x1="178.8" y1="282.0" x2="178.8" y2="12.0" class="stroke-grid" />
<line x1="10.0" y1="113.3" x2="280.0" y2="113.3" class="stroke-grid" />
<line x1="220.9" y1="282.0" x2="220.9" y2="12.0" class="stroke-grid" />
<line x1="10.0" y1="71.1" x2="280.0" y2="71.1" class="stroke-grid" />
<line x1="263.1" y1="282.0" x2="263.1" y2="12.0" class="stroke-grid" />
<line x1="10.0" y1="28.9" x2="280.0" y2="28.9" class="stroke-grid" />
<clipPath id="c022-clip0"><rect x="10" y="12" width="270" height="270"/></clipPath>
<g clip-path="url(#c022-clip0)">
<line x1="17.2" y1="274.8" x2="255.9" y2="36.1" class="stroke-b" stroke-width="2.5"/>
<line x1="136.6" y1="239.8" x2="94.4" y2="197.6" class="stroke-muted dashed" />
<line x1="52.2" y1="155.4" x2="94.4" y2="197.6" class="stroke-muted dashed" />
<line x1="220.9" y1="71.1" x2="220.9" y2="71.1" class="stroke-muted dashed" />
</g>
<circle cx="94.4" cy="197.6" r="4.5" class="fill-b"/>
<circle cx="94.4" cy="197.6" r="4.5" class="fill-b"/>
<circle cx="220.9" cy="71.1" r="4.5" class="fill-b"/>
<circle cx="136.6" cy="239.8" r="5.5" class="fill-a"/>
<circle cx="52.2" cy="155.4" r="5.5" class="fill-a"/>
<circle cx="220.9" cy="71.1" r="5.5" class="fill-a"/>
<circle cx="136.6" cy="155.4" r="4" class="fill-ink"/>
<text x="145.0" y="304.0" class="fig-bold" text-anchor="middle">line u: kept 12, lost 4</text>
<text x="128.6" y="257.8" class="fig-small" text-anchor="end">(2, 0)</text>
<text x="48.2" y="145.4" class="fig-small" text-anchor="end">(0, 2)</text>
<text x="214.9" y="59.1" class="fig-small" text-anchor="end">(4, 4)</text>
<text x="144.6" y="177.4" class="fig-small" text-anchor="start">mean (2, 2)</text>
<text x="86.0" y="214.0" class="fig-small" text-anchor="end">score −1.41</text>
<text x="226.9" y="89.1" class="fig-small" text-anchor="start">score 2.83</text>
<rect x="310.0" y="12.0" width="270.0" height="270.0" class="stroke-grid" fill="none"/>
<line x1="352.2" y1="282.0" x2="352.2" y2="12.0" class="stroke-grid" />
<line x1="310.0" y1="239.8" x2="580.0" y2="239.8" class="stroke-grid" />
<line x1="394.4" y1="282.0" x2="394.4" y2="12.0" class="stroke-grid" />
<line x1="310.0" y1="197.6" x2="580.0" y2="197.6" class="stroke-grid" />
<line x1="436.6" y1="282.0" x2="436.6" y2="12.0" class="stroke-grid" />
<line x1="310.0" y1="155.4" x2="580.0" y2="155.4" class="stroke-grid" />
<line x1="478.8" y1="282.0" x2="478.8" y2="12.0" class="stroke-grid" />
<line x1="310.0" y1="113.3" x2="580.0" y2="113.3" class="stroke-grid" />
<line x1="520.9" y1="282.0" x2="520.9" y2="12.0" class="stroke-grid" />
<line x1="310.0" y1="71.1" x2="580.0" y2="71.1" class="stroke-grid" />
<line x1="563.1" y1="282.0" x2="563.1" y2="12.0" class="stroke-grid" />
<line x1="310.0" y1="28.9" x2="580.0" y2="28.9" class="stroke-grid" />
<clipPath id="c022-clip1"><rect x="310" y="12" width="270" height="270"/></clipPath>
<g clip-path="url(#c022-clip1)">
<line x1="317.2" y1="36.1" x2="555.9" y2="274.8" class="stroke-b" stroke-width="2.5"/>
<line x1="436.6" y1="239.8" x2="478.8" y2="197.6" class="stroke-muted dashed" />
<line x1="352.2" y1="155.4" x2="394.4" y2="113.3" class="stroke-muted dashed" />
<line x1="520.9" y1="71.1" x2="436.6" y2="155.4" class="stroke-muted dashed" />
</g>
<circle cx="478.8" cy="197.6" r="4.5" class="fill-b"/>
<circle cx="394.4" cy="113.3" r="4.5" class="fill-b"/>
<circle cx="436.6" cy="155.4" r="4.5" class="fill-b"/>
<circle cx="436.6" cy="239.8" r="5.5" class="fill-a"/>
<circle cx="352.2" cy="155.4" r="5.5" class="fill-a"/>
<circle cx="520.9" cy="71.1" r="5.5" class="fill-a"/>
<circle cx="436.6" cy="155.4" r="4" class="fill-ink"/>
<text x="445.0" y="304.0" class="fig-bold" text-anchor="middle">line w: kept 4, lost 12</text>
<text x="446.6" y="235.8" class="fig-small" text-anchor="start">(2, 0)</text>
<text x="348.2" y="145.4" class="fig-small" text-anchor="end">(0, 2)</text>
<text x="512.9" y="61.1" class="fig-small" text-anchor="end">(4, 4)</text>
<text x="446.6" y="147.4" class="fig-small" text-anchor="start">mean, score 0</text>
<text x="486.8" y="215.6" class="fig-small" text-anchor="start">score 1.41</text>
<text x="386.4" y="105.3" class="fig-small" text-anchor="end">score −1.41</text>
</svg>
Figure 1. The same three points and two lines through the middle (dark dot). The orange dots are the shadows, and the dashed segments are the gaps. On line u the shadows are far apart and the gaps are short (the two lower points share one shadow, and (4, 4) sits on its own shadow). On line w the shadows bunch up and the gaps are long.
```

Take $u = (1, 1)/\sqrt{2}$. The scores of the three centred points are $-2/\sqrt{2} = -1.41$, $-1.41$, and $4/\sqrt{2} = 2.83$. Take $w = (1, -1)/\sqrt{2}$, at right angles to $u$. The scores are $1.41$, $-1.41$, and $0$. Line $u$ spreads the shadows far apart, and line $w$ bunches them together.

The **variance** of the scores puts a number on that. The scores average to zero because the data is centred, so the variance is the sum of squared scores over $n-1$. For $u$ the squares are 2, 2, and 8, which sum to 12, and $12/2 = 6$. For $w$ they are 2, 2, and 0, so the variance is $4/2 = 2$. You divide by $n-1$ because the mean came from the same points, which makes the gaps look a little small.

$$
\operatorname{var}(u) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i \cdot u)^2
$$

Read it aloud as "square each score on $u$, add them up, and divide by $n-1$":

```python
scores = Xc @ u                 # shape (n,), one shadow per point
variance = scores @ scores / (n - 1)
```

Sweep every direction to see which line wins. A line at angle $\theta$ has unit vector $(\cos\theta, \sin\theta)$, so the centred points have scores $-2\sin\theta$, $-2\cos\theta$, and $2\cos\theta + 2\sin\theta$, and the variance works out to $4 + 2\sin 2\theta$. It peaks at 6 when $\theta = 45^\circ$ and dips to 2 at $135^\circ$, so the best and worst lines sit at right angles. Dropping a coordinate keeps the line at $0^\circ$ or $90^\circ$, where the variance is only 4, because the cloud leans.

> [!KEY]
> The **first principal component** is the direction whose scores have the largest variance. On the example it is $(1,1)/\sqrt{2}$, with variance 6.

## Shadows or gaps, the same goal

A second goal sounds different: lose as little of each point as possible. When you replace a point by its shadow, the **gap** is the distance from the point to the line. Each centred point, its shadow, and its gap form a right triangle, with the point's distance from the middle as the long side. Pythagoras gives

$$
\underbrace{\lVert x_i \rVert^2}_{\text{squared length}} = \underbrace{(x_i \cdot u)^2}_{\text{shadow squared}} + \underbrace{\lVert x_i - (x_i \cdot u)\,u \rVert^2}_{\text{gap squared}} .
$$

Read it aloud as "the squared distance from the middle is the squared shadow plus the squared gap". On the example the squared lengths are 4, 4, and 8, which total 16. For line $u$ the squared shadows are 2, 2, and 8, so 12 are kept, and the squared gaps are 2, 2, and 0, so 4 are lost. The total stays 16 whatever line you choose, so a line that keeps more must lose less. Line $w$ confirms it: it keeps 4 and loses 12.

## Let the SVD do the searching

With 100 readings per row, a direction needs 99 angles, and a grid of 180 steps per angle has about $10^{223}$ points. You do not have to search.

Multiply the centred data matrix $X_c$ by a unit direction $u$ and you get the scores. Their sum of squares is $(n-1)$ times the variance along $u$, so "how far does $X_c$ stretch $u$?" and "how spread out are the shadows on $u$?" are the same question. Feed every unit direction through $X_c$ and the outputs trace an ellipse, which Figure 2 draws, and the long axis is the direction of greatest spread. The SVD finds the axes. The stretchiest direction is $v_1$, with stretch $s_1$. Among directions at right angles to $v_1$, the stretchiest is $v_2$, with stretch $s_2$, and so on down the list.

```figure
<svg viewBox="0 0 600 296" role="img" aria-label="A unit circle of input directions on the left, and on the right the ellipse that the data matrix turns it into, with axis lengths 3.46 and 2.00">
<defs><marker id="c022-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c022-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker></defs>
<circle cx="120.0" cy="150.0" r="85" class="stroke-muted" fill="none"/>
<line x1="0.0" y1="150.0" x2="240.0" y2="150.0" class="stroke-grid" />
<line x1="120.0" y1="30.0" x2="120.0" y2="270.0" class="stroke-grid" />
<line x1="120.0" y1="150.0" x2="180.1" y2="89.9" class="stroke-b" stroke-width="2.5" marker-end="url(#c022-arrow)"/>
<text x="188.1" y="83.9" class="fig-bold" text-anchor="start">v1</text>
<line x1="120.0" y1="150.0" x2="59.9" y2="89.9" class="stroke-b" stroke-width="2.5" marker-end="url(#c022-arrow)"/>
<text x="51.9" y="83.9" class="fig-bold" text-anchor="end">v2</text>
<line x1="120.0" y1="150.0" x2="155.9" y2="73.0" class="stroke-a" stroke-width="2" marker-end="url(#c022-arrow-a)"/>
<circle cx="155.9" cy="73.0" r="4.5" class="fill-a"/>
<text x="166.0" y="60.0" class="fig-small" text-anchor="start">any unit u</text>
<text x="120.0" y="282.0" class="fig-bold" text-anchor="middle">inputs: every unit direction</text>
<polyline points="568.6,150.0 568.4,145.8 567.8,141.6 566.9,137.5 565.5,133.4 563.8,129.3 561.8,125.3 559.4,121.3 556.6,117.5 553.5,113.7 550.0,110.0 546.2,106.4 542.1,103.0 537.7,99.7 533.0,96.5 528.0,93.4 522.7,90.5 517.2,87.8 511.4,85.3 505.5,82.9 499.3,80.7 492.9,78.7 486.4,76.9 479.7,75.3 472.8,73.9 465.9,72.7 458.8,71.7 451.7,71.0 444.5,70.4 437.3,70.1 430.0,70.0 422.7,70.1 415.5,70.4 408.3,71.0 401.2,71.7 394.1,72.7 387.2,73.9 380.3,75.3 373.6,76.9 367.1,78.7 360.7,80.7 354.5,82.9 348.6,85.3 342.8,87.8 337.3,90.5 332.0,93.4 327.0,96.5 322.3,99.7 317.9,103.0 313.8,106.4 310.0,110.0 306.5,113.7 303.4,117.5 300.6,121.3 298.2,125.3 296.2,129.3 294.5,133.4 293.1,137.5 292.2,141.6 291.6,145.8 291.4,150.0 291.6,154.2 292.2,158.4 293.1,162.5 294.5,166.6 296.2,170.7 298.2,174.7 300.6,178.7 303.4,182.5 306.5,186.3 310.0,190.0 313.8,193.6 317.9,197.0 322.3,200.3 327.0,203.5 332.0,206.6 337.3,209.5 342.8,212.2 348.6,214.7 354.5,217.1 360.7,219.3 367.1,221.3 373.6,223.1 380.3,224.7 387.2,226.1 394.1,227.3 401.2,228.3 408.3,229.0 415.5,229.6 422.7,229.9 430.0,230.0 437.3,229.9 444.5,229.6 451.7,229.0 458.8,228.3 465.9,227.3 472.8,226.1 479.7,224.7 486.4,223.1 492.9,221.3 499.3,219.3 505.5,217.1 511.4,214.7 517.2,212.2 522.7,209.5 528.0,206.6 533.0,203.5 537.7,200.3 542.1,197.0 546.2,193.6 550.0,190.0 553.5,186.3 556.6,182.5 559.4,178.7 561.8,174.7 563.8,170.7 565.5,166.6 566.9,162.5 567.8,158.4 568.4,154.2 568.6,150.0" fill="none" class="stroke-muted"/>
<line x1="270.0" y1="150.0" x2="590.0" y2="150.0" class="stroke-grid" />
<line x1="430.0" y1="30.0" x2="430.0" y2="270.0" class="stroke-grid" />
<line x1="430.0" y1="150.0" x2="567.6" y2="150.0" class="stroke-b" stroke-width="2.5" marker-end="url(#c022-arrow)"/>
<line x1="430.0" y1="150.0" x2="430.0" y2="71.0" class="stroke-b" stroke-width="2.5" marker-end="url(#c022-arrow)"/>
<text x="562.6" y="170.0" class="fig-bold" text-anchor="end">s1 = 3.46</text>
<text x="440.0" y="58.0" class="fig-bold" text-anchor="start">s2 = 2.00</text>
<line x1="430.0" y1="150.0" x2="560.2" y2="122.6" class="stroke-a" stroke-width="2"/>
<circle cx="560.2" cy="122.6" r="4.5" class="fill-a"/>
<text x="584.0" y="102.0" class="fig-small" text-anchor="middle">Xc u</text>
<text x="430.0" y="282.0" class="fig-bold" text-anchor="middle">outputs: the scores Xc u</text>
<text x="248.0" y="140.0" class="fig-bold" text-anchor="middle">Xc</text>
<line x1="222.0" y1="150.0" x2="274.0" y2="150.0" class="stroke-muted" marker-end="url(#c022-arrow-a)"/>
</svg>
Figure 2. Left: every unit direction u, with the two special directions v1 and v2 at right angles. Right: the data matrix turns that circle into an ellipse, whose long and short axes have lengths s1 and s2. The directions v1 and v2 are exactly the ones that become the axes, so the ellipse is longest where the spread is largest.
```

On the example the stretches are $s_1 = \sqrt{12} = 3.46$ and $s_2 = 2$. Squared and divided by $n-1 = 2$ they give 6 and 2, the variances you found by sweeping.

$$
X_c = U S V^{\top}, \qquad \lambda_i = \frac{s_i^2}{n-1}
$$

Read it aloud as "the data splits into a rotation, a stretch, and a rotation, and direction $i$ holds its stretch squared over $n-1$". The directions are the columns of $V$, so numpy hands them back as the rows of `Vt`:

```python
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
components = Vt[:k].T                   # shape (d, k)
scores = Xc @ components                # shape (n, k)
explained_variance = s[:k] ** 2 / (n - 1)
```

On the example, numpy returns the first row of `Vt` as $(-0.71, -0.71)$, not $(0.71, 0.71)$. Both describe the same line, and the sign carries no meaning, so compare directions with `abs(u @ v)` equal to 1.

> [!WARNING]
> `np.linalg.svd` returns $V^{\top}$, not $V$. The directions are the rows of `Vt`, so the top $k$ are `Vt[:k]`. The columns of `Vt` mean nothing.

The SVD is not the only route. The covariance table $C = X_c^{\top} X_c/(n-1)$ holds each column's variance on the diagonal and how each pair of columns move together off it. For the example its rows are $(4, 2)$ and $(2, 4)$, and the variance along any unit $u$ is $u^{\top} C u$. For $u = (1,1)/\sqrt{2}$ that is $\tfrac{1}{2}(4 + 2 + 2 + 4) = 6$, the number you found by hand.

Substitute $X_c = U S V^{\top}$ into $X_c^{\top} X_c$. The middle $U^{\top} U$ is the identity, because the columns of $U$ are orthonormal, and what remains is $X_c^{\top} X_c = V S^2 V^{\top}$. An **eigenvector** of a matrix is a direction it only stretches, never turns, and the stretch factor is the **eigenvalue**. So the columns of $V$ are the eigenvectors of $C$, with eigenvalues $s_i^2/(n-1)$. On the example, $Cv = \lambda v$ needs $(4 - \lambda)^2 - 4 = 0$, so $\lambda = 6$ or $2$, and for $\lambda = 6$ the unit vector is $(1,1)/\sqrt{2}$, the peak of the sweep.

Why call the SVD at all, when `np.linalg.eigh` handles $C$? Forming $X_c^{\top} X_c$ squares every singular value, so stretches of 1 and $10^{-9}$ become 1 and $10^{-18}$, and double precision keeps only about 16 digits. `eigh` also returns eigenvalues in ascending order, so reverse them:

```python
C = Xc.T @ Xc / (n - 1)
eigenvalues, eigenvectors = np.linalg.eigh(C)    # smallest first
eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
```

## How many directions to keep

Each direction holds a share of the total variance, and the shares add to 1:

$$
r_i = \frac{s_i^2}{\sum_j s_j^2} .
$$

The $n-1$ cancels, so the example has $r = (12/16,\ 4/16) = (0.75,\ 0.25)$. There are $\min(n, d)$ shares, and when $n \le d$ the last is zero, because $n$ centred points span at most $n-1$ directions.

```python
ratios = s ** 2 / np.sum(s ** 2)
```

Take six-dimensional data ($n = 300$) built from two strong directions plus a little noise. Two variances tower (8.96 and 2.34) and four are tiny (about 0.04 each). The cumulative shares are 78% after one and 99% after two. A common habit is to keep the smallest $k$ whose cumulative share reaches 90%, as `int(np.searchsorted(np.cumsum(ratios), 0.90) + 1)`. The 90% is a habit, not a theorem: use a higher target when you need to reconstruct, and look for the elbow in a bar plot of the variances (a **scree plot**).

## Rebuild from k directions

To rebuild a point from $k$ scores, put each score back on its direction and add the mean:

$$
\hat{X} = \bar{x} + Z V_k^{\top}
$$

Read it aloud as "the rebuilt table is the mean, plus each point's scores times the $k$ directions". $Z$ is the $n \times k$ table of scores and $V_k$ is the $d \times k$ table whose columns are the kept directions. The result has rank $k$, which means every row is a mix of the same $k$ directions plus the mean.

On the example with $k = 1$, the scores $-1.41$, $-1.41$, and $2.83$ rebuild $(1,1)$, $(1,1)$, and $(4,4)$. The squared errors are $(2-1)^2 + (0-1)^2 = 2$, then 2, then 0, so the total error is 4. In numpy:

```python
X_hat = mean + scores @ components.T    # shape (n, d)
error = np.sum((X - X_hat) ** 2)
```

The usual mistake is to leave the mean out. Without it the rebuilt points sit around the origin instead of around the cloud.

You did not need to rebuild to know that the error was 4. The singular values $s = (3.46, 2)$ already said it: $s_2^2 = 4$. Here is why. Because the directions $v_1, \dots, v_d$ sit at right angles, Pythagoras works in $d$ dimensions too: a point's squared length is the sum of its squared shadows on all $d$ directions. Adding over points, the total of the squared lengths is $s_1^2 + s_2^2 + \cdots$, because each squared stretch is the total squared length of the shadows on one direction. The kept part is the shadows on the $k$ kept directions, which add up to $s_1^2 + \cdots + s_k^2$. Whatever is left is lost:

$$
\sum_{\text{all entries}} (X - \hat{X})^2 = \sum_{i > k} s_i^2 .
$$

Read it aloud as "the total squared error of the rebuild is the sum of the squared singular values you dropped", for the singular values of $X_c$. On the example, $12 + 4 = 16$ is the total, 12 is kept, and 4 is lost.

On the six-dimensional data, the rebuild error falls as you keep more directions, as Figure 3 shows. Each orange dot is the sum of the dropped $s_i^2$, and each sits on top of the bar from the actual rebuild. Keeping two directions leaves an error of 47.3 out of a total of 3424.1, so the share you kept is $1 - 47.3/3424.1 = 0.986$, the cumulative share from the scree. In numpy, the singular values give the error with no rebuild, so you can choose $k$ for an error budget:

```python
s = np.linalg.svd(Xc, compute_uv=False)
error_at_k = np.sum(s[k:] ** 2)
```

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Bars of the reconstruction error for k from 0 to 6, with orange dots at the sum of the dropped squared singular values sitting on top of each bar">
<line x1="80.0" y1="240.0" x2="550.0" y2="240.0" class="stroke-grid" />
<text x="72.0" y="244.0" class="fig-small" text-anchor="end">0</text>
<line x1="80.0" y1="180.3" x2="550.0" y2="180.3" class="stroke-grid" />
<text x="72.0" y="184.3" class="fig-small" text-anchor="end">1000</text>
<line x1="80.0" y1="120.6" x2="550.0" y2="120.6" class="stroke-grid" />
<text x="72.0" y="124.6" class="fig-small" text-anchor="end">2000</text>
<line x1="80.0" y1="60.8" x2="550.0" y2="60.8" class="stroke-grid" />
<text x="72.0" y="64.8" class="fig-small" text-anchor="end">3000</text>
<rect x="93.4" y="35.5" width="40.3" height="204.5" class="fill-a"/>
<circle cx="113.6" cy="35.5" r="5.5" class="fill-b"/>
<text x="113.6" y="25.5" class="fig-small" text-anchor="middle">3424</text>
<text x="113.6" y="258.0" class="fig-small" text-anchor="middle">0</text>
<rect x="160.6" y="195.5" width="40.3" height="44.5" class="fill-a"/>
<circle cx="180.7" cy="195.5" r="5.5" class="fill-b"/>
<text x="180.7" y="185.5" class="fig-small" text-anchor="middle">746</text>
<text x="180.7" y="258.0" class="fig-small" text-anchor="middle">1</text>
<rect x="227.7" y="237.2" width="40.3" height="2.8" class="fill-a"/>
<circle cx="247.9" cy="237.2" r="5.5" class="fill-b"/>
<text x="247.9" y="227.2" class="fig-small" text-anchor="middle">47</text>
<text x="247.9" y="258.0" class="fig-small" text-anchor="middle">2</text>
<rect x="294.9" y="238.0" width="40.3" height="2.0" class="fill-a"/>
<circle cx="315.0" cy="238.0" r="5.5" class="fill-b"/>
<text x="315.0" y="228.0" class="fig-small" text-anchor="middle">34</text>
<text x="315.0" y="258.0" class="fig-small" text-anchor="middle">3</text>
<rect x="362.0" y="238.7" width="40.3" height="1.3" class="fill-a"/>
<circle cx="382.1" cy="238.7" r="5.5" class="fill-b"/>
<text x="382.1" y="228.7" class="fig-small" text-anchor="middle">21</text>
<text x="382.1" y="258.0" class="fig-small" text-anchor="middle">4</text>
<rect x="429.1" y="239.4" width="40.3" height="0.6" class="fill-a"/>
<circle cx="449.3" cy="239.4" r="5.5" class="fill-b"/>
<text x="449.3" y="229.4" class="fig-small" text-anchor="middle">10</text>
<text x="449.3" y="258.0" class="fig-small" text-anchor="middle">5</text>
<rect x="496.3" y="240.0" width="40.3" height="0.0" class="fill-a"/>
<circle cx="516.4" cy="240.0" r="5.5" class="fill-b"/>
<text x="516.4" y="230.0" class="fig-small" text-anchor="middle">0</text>
<text x="516.4" y="258.0" class="fig-small" text-anchor="middle">6</text>
<text x="315.0" y="276.0" text-anchor="middle">k, the number of directions kept</text>
<text transform="rotate(-90 14 130)" x="14.0" y="130.0" text-anchor="middle">total squared error</text>
<rect x="330.0" y="34.0" width="14.0" height="14.0" class="fill-a"/>
<text x="350.0" y="46.0" class="fig-small" text-anchor="start">error of the rebuild</text>
<circle cx="337.0" cy="68.0" r="6" class="fill-b"/>
<text x="350.0" y="72.0" class="fig-small" text-anchor="start">sum of dropped s² values</text>
</svg>
Figure 3. The error of the rebuild (blue bars) for each number of kept directions k on generated six-dimensional data. The orange dots are the sum of the dropped squared singular values, and each one sits on top of its bar. Two directions already cut the error from 3424 to 47.
```

Could a different pair of directions lose less? The **Eckart-Young theorem** says no: the sheet spanned by the top $k$ directions keeps the most, so it loses the least, and no matrix of rank $k$ is closer to the centred data. To see why, ask how much of each $v_i$ a sheet contains, a weight $w_i$ between 0 and 1, with the weights adding to $k$. The sheet keeps $w_1 s_1^2 + w_2 s_2^2 + \cdots$, which is largest when the weight goes to the $k$ biggest stretches.

Random two-direction sheets in the six-dimensional space all lose more than 47.3.

> [!KEY]
> You never search for the best direction. The SVD of the centred data lists all of them, in order, and $s_i^2/(n-1)$ is the variance each one holds. Keeping the top $k$ loses exactly $\sum_{i>k} s_i^2$, and no other $k$ directions lose less, so you know the error before you rebuild and know that no other choice could have been better.

## Standardise when units differ

PCA measures spread, and spread depends on units. Write the example's second reading in hundredths: $(2, 0)$, $(0, 200)$, and $(4, 400)$. Only the ruler changed. The centred rows are $(0, -200)$, $(-2, 0)$, and $(2, 200)$, the variances are 4 and 40000, and the first direction is $(0.005, 1.000)$, which is the second column and nothing else.

The fix is to **standardise**: centre each column, then divide it by its standard deviation, which is the square root of its variance. The columns are then $(0, -1, 1)$ and $(-1, 0, 1)$, with variance 1 each, and the first direction returns to $(0.71, 0.71)$ with variances 1.5 and 0.5.

```python
Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
```

Standardise when the columns are in different units, or when their sizes are arbitrary. Do not standardise when the columns share a unit and the size is the signal, such as pixel brightnesses. Dividing there would give quiet, noisy pixels the same weight as informative ones.

## Whiten when every direction should count the same

Standardising fixes the column scales but leaves the lean: in the example the two columns still move together. **Whitening** makes the covariance the identity, so every direction has variance 1 and no two move together. Picture the tilted ellipse squeezed along its long axis and stretched along its short one until it is a circle.

In words: rotate onto the principal directions, then divide each score by the square root of its variance. The square root is right because scaling a score by $c$ scales its variance by $c^2$. On the example the scores on $u$ and $w$ are $(-1.41, 1.41)$, $(-1.41, -1.41)$, and $(2.83, 0)$, with variances 6 and 2. Dividing by $\sqrt{6} = 2.45$ and $\sqrt{2} = 1.41$ gives $(-0.58, 1.00)$, $(-0.58, -1.00)$, and $(1.15, 0)$, and both columns now have variance 1. This is **PCA whitening**, and numpy may flip a column's sign.

Write $C = U \Lambda U^{\top}$, with eigenvectors in $U$ and eigenvalues $\lambda_i$ on the diagonal of $\Lambda$. For a centred row $x$, the whitened row is $z = Wx$, with

$$
W_{\text{PCA}} = (\Lambda + \epsilon I)^{-1/2} U^{\top}, \qquad W_{\text{ZCA}} = U (\Lambda + \epsilon I)^{-1/2} U^{\top} .
$$

Read it aloud as "rotate onto the directions, divide each score by $\sqrt{\lambda_i + \epsilon}$, and for **ZCA** whitening rotate back". Rotating back leaves each output axis along an input axis, so $W_{\text{ZCA}}$ is symmetric and the whitened row stays as close to the original as any whitened row can. Kessy, Lewin, and Strimmer prove it is the unique whitening matrix that minimises the squared distance to the input (Section 6.1, Proposition 1). On the example, $W_{\text{ZCA}}$ has rows $(0.558, -0.149)$ and $(-0.149, 0.558)$, the points become $(0.30, -1.12)$, $(-1.12, 0.30)$, and $(0.82, 0.82)$, and their squared distance from the centred points is 4.55, against 9.07 for PCA whitening.

```python
lam, U = np.linalg.eigh(Xc.T @ Xc / (n - 1))
W = U.T / np.sqrt(lam + eps)[:, None]    # PCA; use U @ W for ZCA
Z = Xc @ W.T
```

The $\epsilon$ guards tiny eigenvalues. A direction with variance $10^{-12}$ is noise, and dividing by its square root multiplies it by $10^{6}$. With $\epsilon = 10^{-5}$ no gain exceeds about 316, and whitened variances become $\lambda_i/(\lambda_i + \epsilon)$. Whitening reappears in [normalisation layers](/lessons/040-normalisation-layers), which rescale activations but do not decorrelate them, and in [autoencoders and VAEs](/lessons/074-autoencoders-and-vaes), whose latent prior has identity covariance.

> [!RECAP]
> - Centre first. A score is a shadow, `Xc @ u`, and its variance is `scores @ scores / (n - 1)`.
> - Squared length equals shadow squared plus gap squared, so most kept means least lost.
> - Components are `Vt[:k].T`, and variances are `s[:k] ** 2 / (n - 1)`. The covariance table has the same directions.
> - Shares are $s_i^2/\sum_j s_j^2$. Rebuild with `mean + scores @ components.T`, and the error is `np.sum(s[k:] ** 2)`.
> - Standardise columns when units differ, because spread depends on the ruler.
> - Whiten by dividing each score by $\sqrt{\lambda_i + \epsilon}$. ZCA rotates back, so it stays closest to the input.

## Build it step by step

`sweep_directions` is written for you, because it only calls `variance_along` in a loop. You write ten functions, in this order.

1. **`center(X)`** takes an `(n, d)` array and returns `(Xc, mean)`, shapes `(n, d)` and `(d,)`. The usual mistake is `axis=1`, which averages each row.
2. **`variance_along(Xc, u)`** takes centred data and a unit vector, and returns one float: the scores `Xc @ u`, squared, summed, and divided by `n - 1`. The usual mistake is dividing by `n`.
3. **`shadow_split(Xc, u)`** returns `(kept, lost)`, both summed over all points. Compute `kept` from the scores and `lost` from the leftover vectors `Xc - np.outer(scores, u)`, not as total minus kept. The usual mistake is to forget the sum over points.
4. **`pca_svd(X, k)`** returns `components` `(d, k)`, `scores` `(n, k)`, and `explained_variance` `(k,)`. Centre, call `np.linalg.svd(Xc, full_matrices=False)`. The usual mistake is `Vt[:, :k]`, which slices columns of $V^{\top}$.
5. **`covariance_eigh(X, k)`** returns the top-$k$ eigenvalues `(k,)` and eigenvectors `(d, k)` of the sample covariance, largest first. Use `np.linalg.eigh` and flip the order. The usual mistake is to leave them ascending.
6. **`explained_variance_ratio(X)`** returns all `min(n, d)` shares, largest first, from `np.linalg.svd(Xc, compute_uv=False)`. The usual mistake is to keep only the top $k$, so the shares no longer add to 1.
7. **`reconstruct(mean, components, scores)`** takes a mean `(d,)`, components `(d, k)`, and scores `(n, k)`, and returns the rebuilt table `(n, d)`. The usual mistake is leaving out the mean.
8. **`reconstruction_error(X, k)`** returns the total squared error of the rank-$k$ rebuild as a float. Run `pca_svd`, rebuild with your `reconstruct`, and add up the squared differences from `X`. The usual mistake is to shortcut with the singular-value formula, which makes the check against that formula circular.
9. **`standardise(X)`** returns the centred table with each column divided by its standard deviation (`ddof=1`), shape `(n, d)`. A column that is constant has standard deviation 0, so leave its entries at 0 rather than dividing. The usual mistake is dividing by the variance instead of the standard deviation.
10. **`whiten(X, eps=1e-5, method="pca")`** returns `(Z, W)`, shapes `(n, d)` and `(d, d)`. Centre, get the eigenvalues and eigenvectors of the covariance with your `covariance_eigh`, and build `W` from `1 / np.sqrt(lam + eps)`. The usual mistake is dividing by the eigenvalue instead of its square root.

## Cost and scaling

Two routes to the directions cost different amounts for $n$ rows and $d$ columns:

| Route | How the work grows | Notes |
|---|---|---|
| SVD of $X_c$ | About $n\,d\,\min(n, d)$ | The default, and the most accurate. |
| Eigenvectors of $X_c^{\top}X_c$ | About $n\,d^2 + d^3$ | Forming the product squares the data and loses digits. |

The chapter on [power iteration and probabilistic PCA](/lessons/029-power-iteration-and-probabilistic-pca) shows how to get only the top few directions for a fraction of this cost.

## Where this breaks

Figure 4 shows three failures.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Three failures of PCA: a far-off point tilting the first line, a curved cloud cut by a straight line, and a column in large units taking over the first direction">
<defs></defs>
<rect x="15.0" y="20.0" width="175.0" height="175.0" class="stroke-grid" fill="none"/>
<clipPath id="c022-ca"><rect x="15" y="20" width="175" height="175"/></clipPath><g clip-path="url(#c022-ca)">
<line x1="-135.9" y1="173.3" x2="341.0" y2="154.9" class="stroke-a" stroke-width="2.5"/>
<line x1="-92.4" y1="297.0" x2="299.8" y2="25.0" class="stroke-b" stroke-width="2.5"/>
</g>
<circle cx="110.7" cy="171.0" r="2.8" class="fill-a"/>
<circle cx="122.1" cy="167.5" r="2.8" class="fill-a"/>
<circle cx="110.4" cy="163.6" r="2.8" class="fill-a"/>
<circle cx="71.4" cy="169.2" r="2.8" class="fill-a"/>
<circle cx="124.1" cy="163.8" r="2.8" class="fill-a"/>
<circle cx="113.2" cy="162.6" r="2.8" class="fill-a"/>
<circle cx="89.7" cy="163.0" r="2.8" class="fill-a"/>
<circle cx="116.4" cy="166.4" r="2.8" class="fill-a"/>
<circle cx="111.2" cy="159.4" r="2.8" class="fill-a"/>
<circle cx="109.5" cy="157.5" r="2.8" class="fill-a"/>
<circle cx="103.2" cy="161.1" r="2.8" class="fill-a"/>
<circle cx="115.5" cy="168.4" r="2.8" class="fill-a"/>
<circle cx="84.9" cy="158.5" r="2.8" class="fill-a"/>
<circle cx="98.6" cy="166.4" r="2.8" class="fill-a"/>
<circle cx="91.0" cy="157.6" r="2.8" class="fill-a"/>
<circle cx="116.8" cy="170.0" r="2.8" class="fill-a"/>
<circle cx="103.4" cy="157.4" r="2.8" class="fill-a"/>
<circle cx="95.5" cy="163.3" r="2.8" class="fill-a"/>
<circle cx="83.8" cy="171.1" r="2.8" class="fill-a"/>
<circle cx="96.4" cy="165.2" r="2.8" class="fill-a"/>
<circle cx="102.7" cy="162.8" r="2.8" class="fill-a"/>
<circle cx="95.9" cy="161.4" r="2.8" class="fill-a"/>
<circle cx="133.4" cy="169.4" r="2.8" class="fill-a"/>
<circle cx="126.5" cy="170.2" r="2.8" class="fill-a"/>
<circle cx="37.8" cy="161.9" r="2.8" class="fill-a"/>
<circle cx="57.4" cy="166.2" r="2.8" class="fill-a"/>
<circle cx="98.3" cy="161.7" r="2.8" class="fill-a"/>
<circle cx="92.4" cy="158.3" r="2.8" class="fill-a"/>
<circle cx="107.6" cy="173.7" r="2.8" class="fill-a"/>
<circle cx="107.7" cy="161.6" r="2.8" class="fill-a"/>
<circle cx="153.0" cy="155.4" r="2.8" class="fill-a"/>
<circle cx="76.0" cy="165.1" r="2.8" class="fill-a"/>
<circle cx="93.5" cy="168.3" r="2.8" class="fill-a"/>
<circle cx="151.2" cy="158.4" r="2.8" class="fill-a"/>
<circle cx="117.9" cy="161.6" r="2.8" class="fill-a"/>
<circle cx="118.3" cy="157.5" r="2.8" class="fill-a"/>
<circle cx="90.2" cy="165.4" r="2.8" class="fill-a"/>
<circle cx="63.2" cy="172.6" r="2.8" class="fill-a"/>
<circle cx="106.5" cy="163.9" r="2.8" class="fill-a"/>
<circle cx="105.1" cy="166.0" r="2.8" class="fill-a"/>
<circle cx="150.2" cy="35.9" r="5" class="fill-bad"/>
<text x="142.2" y="39.9" class="fig-small" text-anchor="end">outlier</text>
<text x="102.0" y="214.0" class="fig-bold" text-anchor="middle">one far point</text>
<text x="102.0" y="232.0" class="fig-small" text-anchor="middle">blue 2°, orange 35°</text>
<rect x="215.0" y="20.0" width="175.0" height="175.0" class="stroke-grid" fill="none"/>
<clipPath id="c022-cb"><rect x="215" y="20" width="175" height="175"/></clipPath><g clip-path="url(#c022-cb)">
<line x1="138.7" y1="114.2" x2="466.8" y2="123.6" class="stroke-b" stroke-width="2.5"/>
</g>
<circle cx="357.2" cy="164.4" r="3" class="fill-a"/>
<circle cx="355.4" cy="161.0" r="3" class="fill-a"/>
<circle cx="357.7" cy="144.7" r="3" class="fill-a"/>
<circle cx="352.2" cy="138.0" r="3" class="fill-a"/>
<circle cx="350.8" cy="133.9" r="3" class="fill-a"/>
<circle cx="349.0" cy="126.5" r="3" class="fill-a"/>
<circle cx="343.5" cy="121.3" r="3" class="fill-a"/>
<circle cx="341.0" cy="113.8" r="3" class="fill-a"/>
<circle cx="336.9" cy="109.7" r="3" class="fill-a"/>
<circle cx="331.7" cy="105.8" r="3" class="fill-a"/>
<circle cx="328.0" cy="99.3" r="3" class="fill-a"/>
<circle cx="322.1" cy="95.6" r="3" class="fill-a"/>
<circle cx="314.6" cy="92.3" r="3" class="fill-a"/>
<circle cx="314.2" cy="96.4" r="3" class="fill-a"/>
<circle cx="302.4" cy="95.3" r="3" class="fill-a"/>
<circle cx="301.1" cy="91.4" r="3" class="fill-a"/>
<circle cx="296.0" cy="90.8" r="3" class="fill-a"/>
<circle cx="289.1" cy="93.5" r="3" class="fill-a"/>
<circle cx="283.2" cy="94.5" r="3" class="fill-a"/>
<circle cx="276.5" cy="100.7" r="3" class="fill-a"/>
<circle cx="273.9" cy="99.4" r="3" class="fill-a"/>
<circle cx="267.7" cy="110.8" r="3" class="fill-a"/>
<circle cx="263.9" cy="111.2" r="3" class="fill-a"/>
<circle cx="260.6" cy="116.2" r="3" class="fill-a"/>
<circle cx="254.5" cy="123.1" r="3" class="fill-a"/>
<circle cx="256.4" cy="136.0" r="3" class="fill-a"/>
<circle cx="252.5" cy="136.9" r="3" class="fill-a"/>
<circle cx="250.4" cy="145.0" r="3" class="fill-a"/>
<circle cx="252.8" cy="154.6" r="3" class="fill-a"/>
<circle cx="247.6" cy="165.0" r="3" class="fill-a"/>
<text x="302.0" y="214.0" class="fig-bold" text-anchor="middle">a curved cloud</text>
<text x="302.0" y="232.0" class="fig-small" text-anchor="middle">a line cuts across it</text>
<rect x="415.0" y="20.0" width="175.0" height="175.0" class="stroke-grid" fill="none"/>
<rect x="435.0" y="174.5" width="22.0" height="0.5" class="fill-a"/>
<text x="446.0" y="169.5" class="fig-small" text-anchor="middle">0.01</text>
<rect x="463.0" y="85.0" width="22.0" height="90.0" class="fill-b"/>
<text x="474.0" y="80.0" class="fig-small" text-anchor="middle">1.00</text>
<text x="460.0" y="189.0" class="fig-small" text-anchor="middle">raw</text>
<rect x="515.0" y="111.4" width="22.0" height="63.6" class="fill-a"/>
<text x="526.0" y="106.4" class="fig-small" text-anchor="middle">0.71</text>
<rect x="543.0" y="111.4" width="22.0" height="63.6" class="fill-b"/>
<text x="554.0" y="106.4" class="fig-small" text-anchor="middle">0.71</text>
<text x="540.0" y="189.0" class="fig-small" text-anchor="middle">standardised</text>
<text x="435.0" y="36.0" class="fig-small" text-anchor="start">weight of each column</text>
<text x="435.0" y="52.0" class="fig-small" text-anchor="start">in the first direction</text>
<text x="502.0" y="214.0" class="fig-bold" text-anchor="middle">mixed units</text>
<text x="502.0" y="232.0" class="fig-small" text-anchor="middle">blue: column 1, orange: column 2</text>
</svg>
Figure 4. Left: one far-off point tilts the first line from 2° (blue) to 35° (orange). Middle: a straight line cannot follow an arc. Right: when one column is in much larger units, it takes over the first direction, and standardising restores an even split.
```

- A far-off point drags a direction, because variance squares distance. Remove it.
- A curved cloud, such as an arc, has no flat sheet that fits. Kernel PCA handles curves.
- When two singular values are nearly equal, which of the two directions you keep is unstable, though the plane they span is fine.
- The 90% rule ignores your task. The dropped directions might hold the signal.

## The exercise

You write `center`, `variance_along`, `shadow_split`, `pca_svd`, `covariance_eigh`, `explained_variance_ratio`, `reconstruct`, `reconstruction_error`, `standardise`, and `whiten`. The checks confirm the hand numbers (variance 6 and 2, kept 12 and lost 4, shares 0.75 and 0.25, rebuilt points $(1,1)$, $(1,1)$, $(4,4)$), that the components are orthonormal and no random direction beats the first, that the eigenvectors satisfy $Cv = \lambda v$, that the rebuild error equals $\sum_{i>k} s_i^2$ and no random sheet beats it, and that standardising removes the effect of units. The last two checks confirm the `whiten` hand numbers, that both methods give zero mean and identity covariance, that ZCA is symmetric and closer to the input, and that `eps` caps the gain on a flat direction.

## Further reading

- For where PCA sits in the course, see the Autumn 2018 [CS229 syllabus](https://cs229.stanford.edu/syllabus-autumn2018.html), Lecture 12 class notes "Principal Components Analysis" and Lecture 15 "Principal Component Analysis. Independent Component Analysis."
- For the proof that ZCA is the closest whitening, see Kessy, Lewin, and Strimmer, [Optimal Whitening and Decorrelation](https://arxiv.org/html/1512.00809), Section 6.1 (ZCA-Mahalanobis) and Section 6.3 (PCA whitening).
