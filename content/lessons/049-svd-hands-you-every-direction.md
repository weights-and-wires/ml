---
title: "The SVD hands you every direction at once"
slug: 049-svd-hands-you-every-direction
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: The SVD of centred data lists every direction in order of spread, and each squared singular value says how much variance a direction holds.
minutes: 40
gpu: false
objectives:
  - Explain why the SVD of the centred data gives the directions of greatest spread, in order.
  - Read the components, scores, and explained variances from a call to the SVD.
  - Show that the covariance matrix has the same directions, and compute them with an eigendecomposition.
  - Pick the number of components to keep from the explained variance ratios.
prereqs:
  - 008-qr-and-svd
  - 048-spread-shadows-first-direction
exercise: 049-svd-hands-you-every-direction
---

# The SVD hands you every direction at once

In two dimensions you found the best direction by sweeping an angle through half a turn and picking the peak. Suppose each row has 100 readings. A direction in 100 dimensions needs 99 angles, so a grid with 180 steps per angle has about $10^{223}$ points. You cannot try them all, and you still want the second best direction, and the third.

This chapter shows that you do not have to search. One call to the SVD of the centred data returns every direction, sorted by how much spread it holds, along with the amount.

The reason is a picture: a data matrix turns a circle of directions into an ellipse, and the long axis of that ellipse is the direction of greatest spread. The rest of the chapter makes that picture exact and then reads the answer off it.

## What you already know

Five ideas carry this chapter:

- Centre the data, so the middle of the cloud is the origin. A score is the dot product of a point with a unit vector, and the variance of the scores is `scores @ scores / (n - 1)`.
- On the running example $(2,0)$, $(0,2)$, $(4,4)$, the variance is 6 along $(1,1)/\sqrt{2}$ and 2 along $(1,-1)/\sqrt{2}$, and the covariance table is $C = X_c^{\top}X_c/(n-1)$ with rows $(4,2)$ and $(2,4)$.
- **Orthonormal** vectors have length 1 and meet at right angles.
- "Stable fits with QR and the SVD" split a matrix as $X = U S V^{\top}$, with orthonormal columns in $U$ and $V$ and the singular values $s_1 \ge s_2 \ge \cdots \ge 0$ on the diagonal of $S$.
- An **eigenvector** of a matrix is a direction the matrix only stretches, never turns. The stretch factor is the eigenvalue.

## Stretch is spread

Take the centred data matrix $X_c$ and multiply it by a unit direction $u$. The result $X_c u$ is the list of scores, one shadow per point. Its squared length is the sum of the squared scores, which is $(n-1)$ times the variance along $u$. So the question "how far does $X_c$ stretch $u$?" and the question "how spread out are the shadows on $u$?" are the same question.

Feed every unit direction through $X_c$ and the outputs trace an ellipse. Figure 1 draws it. The SVD finds its axes. Why it finds them is the job of the section "The covariance table finds the same directions", so for now take the picture on trust. The stretchiest direction is $v_1$, and the stretch is $s_1$. Among directions at right angles to $v_1$, the stretchiest is $v_2$, with stretch $s_2$, and so on down the list.

```figure
<svg viewBox="0 0 600 296" role="img" aria-label="A unit circle of input directions on the left, and on the right the ellipse that the data matrix turns it into, with axis lengths 3.46 and 2.00">
<defs><marker id="c049-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c049-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker></defs>
<circle cx="120.0" cy="150.0" r="85" class="stroke-muted" fill="none"/>
<line x1="0.0" y1="150.0" x2="240.0" y2="150.0" class="stroke-grid" />
<line x1="120.0" y1="30.0" x2="120.0" y2="270.0" class="stroke-grid" />
<line x1="120.0" y1="150.0" x2="180.1" y2="89.9" class="stroke-b" stroke-width="2.5" marker-end="url(#c049-arrow)"/>
<text x="188.1" y="83.9" class="fig-bold" text-anchor="start">v1</text>
<line x1="120.0" y1="150.0" x2="59.9" y2="89.9" class="stroke-b" stroke-width="2.5" marker-end="url(#c049-arrow)"/>
<text x="51.9" y="83.9" class="fig-bold" text-anchor="end">v2</text>
<line x1="120.0" y1="150.0" x2="155.9" y2="73.0" class="stroke-a" stroke-width="2" marker-end="url(#c049-arrow-a)"/>
<circle cx="155.9" cy="73.0" r="4.5" class="fill-a"/>
<text x="166.0" y="60.0" class="fig-small" text-anchor="start">any unit u</text>
<text x="120.0" y="282.0" class="fig-bold" text-anchor="middle">inputs: every unit direction</text>
<polyline points="568.6,150.0 568.4,145.8 567.8,141.6 566.9,137.5 565.5,133.4 563.8,129.3 561.8,125.3 559.4,121.3 556.6,117.5 553.5,113.7 550.0,110.0 546.2,106.4 542.1,103.0 537.7,99.7 533.0,96.5 528.0,93.4 522.7,90.5 517.2,87.8 511.4,85.3 505.5,82.9 499.3,80.7 492.9,78.7 486.4,76.9 479.7,75.3 472.8,73.9 465.9,72.7 458.8,71.7 451.7,71.0 444.5,70.4 437.3,70.1 430.0,70.0 422.7,70.1 415.5,70.4 408.3,71.0 401.2,71.7 394.1,72.7 387.2,73.9 380.3,75.3 373.6,76.9 367.1,78.7 360.7,80.7 354.5,82.9 348.6,85.3 342.8,87.8 337.3,90.5 332.0,93.4 327.0,96.5 322.3,99.7 317.9,103.0 313.8,106.4 310.0,110.0 306.5,113.7 303.4,117.5 300.6,121.3 298.2,125.3 296.2,129.3 294.5,133.4 293.1,137.5 292.2,141.6 291.6,145.8 291.4,150.0 291.6,154.2 292.2,158.4 293.1,162.5 294.5,166.6 296.2,170.7 298.2,174.7 300.6,178.7 303.4,182.5 306.5,186.3 310.0,190.0 313.8,193.6 317.9,197.0 322.3,200.3 327.0,203.5 332.0,206.6 337.3,209.5 342.8,212.2 348.6,214.7 354.5,217.1 360.7,219.3 367.1,221.3 373.6,223.1 380.3,224.7 387.2,226.1 394.1,227.3 401.2,228.3 408.3,229.0 415.5,229.6 422.7,229.9 430.0,230.0 437.3,229.9 444.5,229.6 451.7,229.0 458.8,228.3 465.9,227.3 472.8,226.1 479.7,224.7 486.4,223.1 492.9,221.3 499.3,219.3 505.5,217.1 511.4,214.7 517.2,212.2 522.7,209.5 528.0,206.6 533.0,203.5 537.7,200.3 542.1,197.0 546.2,193.6 550.0,190.0 553.5,186.3 556.6,182.5 559.4,178.7 561.8,174.7 563.8,170.7 565.5,166.6 566.9,162.5 567.8,158.4 568.4,154.2 568.6,150.0" fill="none" class="stroke-muted"/>
<line x1="270.0" y1="150.0" x2="590.0" y2="150.0" class="stroke-grid" />
<line x1="430.0" y1="30.0" x2="430.0" y2="270.0" class="stroke-grid" />
<line x1="430.0" y1="150.0" x2="567.6" y2="150.0" class="stroke-b" stroke-width="2.5" marker-end="url(#c049-arrow)"/>
<line x1="430.0" y1="150.0" x2="430.0" y2="71.0" class="stroke-b" stroke-width="2.5" marker-end="url(#c049-arrow)"/>
<text x="562.6" y="170.0" class="fig-bold" text-anchor="end">s1 = 3.46</text>
<text x="440.0" y="58.0" class="fig-bold" text-anchor="start">s2 = 2.00</text>
<line x1="430.0" y1="150.0" x2="560.2" y2="122.6" class="stroke-a" stroke-width="2"/>
<circle cx="560.2" cy="122.6" r="4.5" class="fill-a"/>
<text x="584.0" y="102.0" class="fig-small" text-anchor="middle">Xc u</text>
<text x="430.0" y="282.0" class="fig-bold" text-anchor="middle">outputs: the scores Xc u</text>
<text x="248.0" y="140.0" class="fig-bold" text-anchor="middle">Xc</text>
<line x1="222.0" y1="150.0" x2="274.0" y2="150.0" class="stroke-muted" marker-end="url(#c049-arrow-a)"/>
</svg>
Figure 1. Left: every unit direction u, with the two special directions v1 and v2 at right angles. Right: the data matrix turns that circle into an ellipse, whose long and short axes have lengths s1 and s2. The directions v1 and v2 are exactly the ones that become the axes, so the ellipse is longest where the spread is largest.
```

On the example the stretches are $s_1 = \sqrt{12} = 3.46$ and $s_2 = 2$. Squared and divided by $n-1 = 2$ they give 6 and 2, the same variances you found by sweeping. Each squared singular value tells you how much variance its direction holds, once you divide by $n-1$.

$$
X_c = U S V^{\top}, \qquad \lambda_i = \frac{s_i^2}{n-1}
$$

Read it aloud as "the centred data splits into a rotation, a stretch, and a rotation, and the variance along the $i$-th direction is its stretch squared over $n-1$". The directions are the columns of $V$, so numpy hands them back as the rows of `Vt`. The scores of all points on the first $k$ directions are $X_c$ times those columns:

```python
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
components = Vt[:k].T                   # shape (d, k)
scores = Xc @ components                # shape (n, k)
explained_variance = s[:k] ** 2 / (n - 1)
```

On the example, numpy returns the first row of `Vt` as $(-0.71, -0.71)$, not $(0.71, 0.71)$. Both describe the same line, and the sign carries no meaning.

> [!WARNING]
> `np.linalg.svd` returns $V^{\top}$, not $V$. The directions are the rows of `Vt`, so the top $k$ are `Vt[:k]`. The columns of `Vt` mean nothing.

> [!TIP]
> Two components can differ only by a sign. To compare them, check that `abs(u @ v)` is 1, rather than that `u` equals `v`.

## The covariance table finds the same directions

The SVD is not the only route to these directions. Substitute $X_c = U S V^{\top}$ into $X_c^{\top} X_c$. The middle $U^{\top} U$ is the identity, because the columns of $U$ are orthonormal, and what remains is

$$
X_c^{\top} X_c = V S^2 V^{\top} .
$$

Read it aloud as "the product of the data with itself is a matrix whose eigenvectors are the columns of $V$, with eigenvalues $s_i^2$". Divide by $n-1$ and you have the covariance table $C$, whose eigenvectors are the directions of $V$ and whose eigenvalues are $s_i^2/(n-1)$.

Check it on the example. A direction $v$ is an eigenvector of $C$ with eigenvalue $\lambda$ when $Cv = \lambda v$, which means $(C - \lambda I)v = 0$. A nonzero $v$ can satisfy that only if $C - \lambda I$ squashes some direction flat, and that happens only when its determinant is zero:

$$
(4 - \lambda)^2 - 4 = 0 .
$$

So $\lambda = 6$ or $\lambda = 2$, the two variances again. For $\lambda = 6$ the equation $(C - 6I)v = 0$ has rows $(-2, 2)$ and $(2, -2)$, which say $v_1 = v_2$. The unit vector is $(1,1)/\sqrt{2}$, the peak of the sweep.

The eigenvector is the peak for a reason you can see in one line. Write any unit direction as $u = a\,v_1 + b\,v_2$ with $a^2 + b^2 = 1$. Its variance is $u^{\top} C u = 6a^2 + 2b^2$, a weighted average of 6 and 2. An average with weights that add to 1 is largest when all the weight sits on the larger number, which is $a = 1$. Figure 2 puts the three routes side by side.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Three panels, a sweep of angles, a covariance eigenvector, and the SVD, each ending on the same arrow at 45 degrees">
<defs><marker id="c049-arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<line x1="15.0" y1="120.0" x2="185.0" y2="120.0" class="stroke-grid" />
<line x1="100.0" y1="35.0" x2="100.0" y2="205.0" class="stroke-grid" />
<line x1="100.0" y1="120.0" x2="157.2" y2="120.0" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="42.8" y2="120.0" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="161.7" y2="103.5" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="38.3" y2="136.5" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="159.3" y2="85.8" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="40.7" y2="154.2" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="149.5" y2="70.5" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="50.5" y2="169.5" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="134.2" y2="60.7" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="65.8" y2="179.3" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="116.5" y2="58.3" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="83.5" y2="181.7" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="100.0" y2="62.8" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="100.0" y2="177.2" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="87.2" y2="72.2" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="112.8" y2="167.8" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="78.5" y2="82.7" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="121.5" y2="157.3" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="71.4" y2="91.4" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="128.6" y2="148.6" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="62.7" y2="98.5" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="137.3" y2="141.5" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="52.2" y2="107.2" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="147.8" y2="132.8" class="stroke-muted" />
<line x1="100.0" y1="120.0" x2="149.5" y2="70.5" class="stroke-b" stroke-width="3" marker-end="url(#c049-arrow2)"/>
<text x="100.0" y="226.0" class="fig-bold" text-anchor="middle">1. Sweep the angle</text>
<text x="100.0" y="246.0" class="fig-small" text-anchor="middle">variance peaks at 45°</text>
<text x="100.0" y="264.0" class="fig-small" text-anchor="middle">peak value 6</text>
<line x1="215.0" y1="120.0" x2="385.0" y2="120.0" class="stroke-grid" />
<line x1="300.0" y1="35.0" x2="300.0" y2="205.0" class="stroke-grid" />
<polyline points="349.5,70.5 347.4,68.6 345.0,67.0 342.5,65.6 339.7,64.5 336.7,63.7 333.6,63.2 330.3,62.9 326.8,62.9 323.2,63.2 319.5,63.7 315.8,64.5 311.9,65.6 308.0,67.0 304.0,68.6 300.0,70.5 296.0,72.6 292.0,75.0 288.1,77.5 284.2,80.3 280.5,83.3 276.8,86.4 273.2,89.7 269.7,93.2 266.4,96.8 263.3,100.5 260.3,104.2 257.5,108.1 255.0,112.0 252.6,116.0 250.5,120.0 248.6,124.0 247.0,128.0 245.6,131.9 244.5,135.8 243.7,139.5 243.2,143.2 242.9,146.8 242.9,150.3 243.2,153.6 243.7,156.7 244.5,159.7 245.6,162.5 247.0,165.0 248.6,167.4 250.5,169.5 252.6,171.4 255.0,173.0 257.5,174.4 260.3,175.5 263.3,176.3 266.4,176.8 269.7,177.1 273.2,177.1 276.8,176.8 280.5,176.3 284.2,175.5 288.1,174.4 292.0,173.0 296.0,171.4 300.0,169.5 304.0,167.4 308.0,165.0 311.9,162.5 315.8,159.7 319.5,156.7 323.2,153.6 326.8,150.3 330.3,146.8 333.6,143.2 336.7,139.5 339.7,135.8 342.5,131.9 345.0,128.0 347.4,124.0 349.5,120.0 351.4,116.0 353.0,112.0 354.4,108.1 355.5,104.2 356.3,100.5 356.8,96.8 357.1,93.2 357.1,89.7 356.8,86.4 356.3,83.3 355.5,80.3 354.4,77.5 353.0,75.0 351.4,72.6 349.5,70.5" fill="none" class="stroke-muted"/>
<line x1="300.0" y1="120.0" x2="349.5" y2="70.5" class="stroke-b" stroke-width="3" marker-end="url(#c049-arrow2)"/>
<text x="300.0" y="226.0" class="fig-bold" text-anchor="middle">2. Eigenvector of C</text>
<text x="300.0" y="246.0" class="fig-small" text-anchor="middle">C v = 6 v, v = (0.71, 0.71)</text>
<text x="300.0" y="264.0" class="fig-small" text-anchor="middle">eigenvalue 6</text>
<line x1="415.0" y1="120.0" x2="585.0" y2="120.0" class="stroke-grid" />
<line x1="500.0" y1="35.0" x2="500.0" y2="205.0" class="stroke-grid" />
<circle cx="500.0" cy="160.0" r="4.5" class="fill-a"/>
<circle cx="460.0" cy="120.0" r="4.5" class="fill-a"/>
<circle cx="540.0" cy="80.0" r="4.5" class="fill-a"/>
<line x1="500.0" y1="120.0" x2="549.5" y2="70.5" class="stroke-b" stroke-width="3" marker-end="url(#c049-arrow2)"/>
<text x="500.0" y="226.0" class="fig-bold" text-anchor="middle">3. SVD of Xc</text>
<text x="500.0" y="246.0" class="fig-small" text-anchor="middle">first row of Vt, up to sign</text>
<text x="500.0" y="264.0" class="fig-small" text-anchor="middle">s1² / (n − 1) = 12 / 2 = 6</text>
</svg>
Figure 2. Three ways to find the first direction of the three-point example all end on the same arrow. The sweep, the eigenvector of the covariance table C, and the SVD agree, and each reports the number 6.
```

Why call the SVD at all, when `np.linalg.eigh` handles $C$? It does, and for the top directions of well-behaved data the answers agree. But forming $X_c^{\top} X_c$ squares every singular value. Two directions with stretches 1 and $10^{-9}$ become 1 and $10^{-18}$ in $C$, and double precision keeps only about 16 digits, so the small one drowns in rounding error. The SVD works on $X_c$ directly and never squares. In numpy, `eigh` also returns eigenvalues in ascending order, so reverse them:

```python
C = Xc.T @ Xc / (n - 1)
eigenvalues, eigenvectors = np.linalg.eigh(C)    # smallest first
eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
```

## How many directions to keep

The directions come with a budget. Each one holds a share of the total variance, and the shares add to 1:

$$
r_i = \frac{s_i^2}{\sum_j s_j^2} .
$$

Read it aloud as "the share of direction $i$ is its squared stretch divided by every squared stretch added together". The $n-1$ cancels. On the example $r = (12/16,\ 4/16) = (0.75,\ 0.25)$. There are $\min(n, d)$ singular values in all, and when $n \le d$ the last is zero, because $n$ centred points span at most $n-1$ directions.

```python
ratios = s ** 2 / np.sum(s ** 2)
```

Figure 3 shows a typical picture, called a **scree plot**, for six-dimensional data built from two strong directions plus a little noise. Two bars tower and four are tiny. The cumulative shares are 78% after one and 99% after two, so two directions describe the data. A common habit is to keep the smallest $k$ whose cumulative share reaches 90%:

```python
k = int(np.searchsorted(np.cumsum(ratios), 0.90) + 1)
```

The 90% is a habit, not a theorem. Use a higher target when you compress for later reconstruction, and look for the elbow in the bars when the share creeps up slowly.

```figure
<svg viewBox="0 0 600 285" role="img" aria-label="Bars of the variance of each of six components, two tall and four tiny, with a cumulative line crossing 90 percent at the second">
<line x1="80.0" y1="230.0" x2="520.0" y2="230.0" class="stroke-grid" />
<text x="72.0" y="234.0" class="fig-small" text-anchor="end">0</text>
<line x1="80.0" y1="190.0" x2="520.0" y2="190.0" class="stroke-grid" />
<text x="72.0" y="194.0" class="fig-small" text-anchor="end">2</text>
<line x1="80.0" y1="150.0" x2="520.0" y2="150.0" class="stroke-grid" />
<text x="72.0" y="154.0" class="fig-small" text-anchor="end">4</text>
<line x1="80.0" y1="110.0" x2="520.0" y2="110.0" class="stroke-grid" />
<text x="72.0" y="114.0" class="fig-small" text-anchor="end">6</text>
<line x1="80.0" y1="70.0" x2="520.0" y2="70.0" class="stroke-grid" />
<text x="72.0" y="74.0" class="fig-small" text-anchor="end">8</text>
<line x1="80.0" y1="30.0" x2="520.0" y2="30.0" class="stroke-grid" />
<text x="72.0" y="34.0" class="fig-small" text-anchor="end">10</text>
<text x="528.0" y="234.0" class="fig-small" text-anchor="start">0%</text>
<text x="528.0" y="134.0" class="fig-small" text-anchor="start">50%</text>
<text x="528.0" y="34.0" class="fig-small" text-anchor="start">100%</text>
<rect x="94.7" y="50.9" width="44.0" height="179.1" class="fill-a"/>
<text x="116.7" y="64.0" class="fig-small fill-paper" text-anchor="middle">8.96</text>
<text x="116.7" y="248.0" class="fig-small" text-anchor="middle">1</text>
<rect x="168.0" y="183.3" width="44.0" height="46.7" class="fill-a"/>
<text x="190.0" y="177.3" class="fig-small" text-anchor="middle">2.34</text>
<text x="190.0" y="248.0" class="fig-small" text-anchor="middle">2</text>
<rect x="241.3" y="229.1" width="44.0" height="0.9" class="fill-a"/>
<text x="263.3" y="223.1" class="fig-small" text-anchor="middle">0.04</text>
<text x="263.3" y="248.0" class="fig-small" text-anchor="middle">3</text>
<rect x="314.7" y="229.2" width="44.0" height="0.8" class="fill-a"/>
<text x="336.7" y="223.2" class="fig-small" text-anchor="middle">0.04</text>
<text x="336.7" y="248.0" class="fig-small" text-anchor="middle">4</text>
<rect x="388.0" y="229.2" width="44.0" height="0.8" class="fill-a"/>
<text x="410.0" y="223.2" class="fig-small" text-anchor="middle">0.04</text>
<text x="410.0" y="248.0" class="fig-small" text-anchor="middle">5</text>
<rect x="461.3" y="229.4" width="44.0" height="0.6" class="fill-a"/>
<text x="483.3" y="223.4" class="fig-small" text-anchor="middle">0.03</text>
<text x="483.3" y="248.0" class="fig-small" text-anchor="middle">6</text>
<polyline points="116.7,73.6 190.0,32.8 263.3,32.0 336.7,31.3 410.0,30.6 483.3,30.0" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="116.7" cy="73.6" r="4.5" class="fill-b"/>
<circle cx="190.0" cy="32.8" r="4.5" class="fill-b"/>
<circle cx="263.3" cy="32.0" r="4.5" class="fill-b"/>
<circle cx="336.7" cy="31.3" r="4.5" class="fill-b"/>
<circle cx="410.0" cy="30.6" r="4.5" class="fill-b"/>
<circle cx="483.3" cy="30.0" r="4.5" class="fill-b"/>
<line x1="80.0" y1="50.0" x2="520.0" y2="50.0" class="stroke-b dashed" />
<text x="516.0" y="68.0" class="fig-small" text-anchor="end">90% of the variance</text>
<text x="126.7" y="91.6" class="fig-bold" text-anchor="start">78%</text>
<text x="200.0" y="22.8" class="fig-bold" text-anchor="start">99%</text>
<text x="300.0" y="272.0" text-anchor="middle">component number</text>
<text transform="rotate(-90 14 130)" x="14.0" y="130.0" text-anchor="middle">variance of each component</text>
</svg>
Figure 3. The variance of each of six components on generated data, with a cumulative line. Two bars tower over four small ones, and the line passes 90% at the second bar, so keep two.
```

> [!KEY]
> You never search for the best direction. The SVD of the centred data lists all of them, in order, and $s_i^2/(n-1)$ is the variance each one holds. The eigenvectors of the covariance table give the same directions, but the SVD avoids squaring the data.

> [!RECAP]
> - The data matrix stretches the direction $v_i$ by $s_i$, and a stretch is the spread of the shadows.
> - Components are `Vt[:k].T`, scores are `Xc @ components`, and variances are `s[:k] ** 2 / (n - 1)`.
> - $X_c^{\top}X_c = VS^2V^{\top}$, so the covariance table has the same directions, with eigenvalues $s_i^2/(n-1)$.
> - `eigh` returns ascending order, and its directions can differ from the SVD's by a sign.
> - Ratios are $s_i^2/\sum_j s_j^2$, and you keep enough directions to reach a target such as 90%.

## Build it step by step

`center` is already written for you in the starter, from "Spread, shadows, and the first direction". Build three functions, in this order.

1. **`pca_svd(X, k)`** takes `(n, d)` data and returns `components` with shape `(d, k)`, `scores` with shape `(n, k)`, and `explained_variance` with shape `(k,)`. Centre first, call `np.linalg.svd(Xc, full_matrices=False)`, and use `Vt[:k].T`. The usual mistake is `Vt[:, :k]`, which slices columns of $V^{\top}$. The shape of the code:

    ```python
    def pca_svd(X, k):
        # Centre, take the SVD, keep the top k rows of Vt.
    ```

2. **`covariance_eigh(X, k)`** returns the top-$k$ eigenvalues `(k,)` and eigenvectors `(d, k)` of the sample covariance, largest first. Use `np.linalg.eigh` and flip the order. The usual mistake is to leave them in ascending order. The shape of the code:

    ```python
    def covariance_eigh(X, k):
        # Covariance of the centred data, eigh, reverse, keep k.
    ```

3. **`explained_variance_ratio(X)`** returns all `min(n, d)` shares, largest first. Call `np.linalg.svd(Xc, compute_uv=False)` and divide the squares by their sum. The usual mistake is to keep only the top $k$, so the shares no longer add to 1. The shape of the code:

    ```python
    def explained_variance_ratio(X):
        # s ** 2 / sum(s ** 2), using compute_uv=False.
    ```

## Where this breaks

- If two eigenvalues are nearly equal, any direction in their plane holds almost the same variance, so the individual directions are unstable. The plane is stable, and the directions in it are not.
- A cloud with no spread in some directions gives zero singular values, so some of the $\min(n, d)$ shares are zero.
- The 90% rule says nothing about whether the dropped directions matter to your task. They might hold the signal.

## The exercise

You write `pca_svd`, `covariance_eigh`, and `explained_variance_ratio`. The checks confirm the hand numbers on the three-point example (variances 6 and 2, shares 0.75 and 0.25), that the components are orthonormal and the scores equal $X_c$ times them, and that no random direction beats the first component. They also check that the eigenvectors from `covariance_eigh` satisfy $Cv = \lambda v$, match `pca_svd` up to sign on data with a clear gap, and that the variances add up to the trace of $C$.
