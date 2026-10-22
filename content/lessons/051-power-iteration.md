---
title: "Finding the top eigenvector with one repeated product"
slug: 051-power-iteration
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: Multiply a vector by a symmetric matrix and rescale, again and again, and it turns toward the top eigenvector at a speed set by the gap.
minutes: 40
gpu: false
objectives:
  - Explain why repeated multiplication turns a vector toward the eigenvector with the largest eigenvalue.
  - Run power iteration by hand on a two-by-two matrix, rescaling at every step, and read the eigenvalue from the Rayleigh quotient.
  - Predict how many steps you need from the ratio of the top two eigenvalues.
  - Find a second direction by deflation, and state when power iteration fails.
prereqs:
  - 049-svd-hands-you-every-direction
  - 050-compress-and-know-what-you-lost
exercise: 051-power-iteration
---

# Finding the top eigenvector with one repeated product

Suppose your table has 50,000 columns. Its covariance table has $50{,}000^2 = 2.5$ billion entries, which is 20 GB in double precision, and a full eigendecomposition takes about $50{,}000^3 \approx 10^{14}$ steps. You cannot afford either, but you still want the top direction.

There is a way out. Multiplying the covariance table by one vector is cheap, because $X_c^{\top}(X_c v)$ never builds the table at all. So the question is whether the top direction can come from nothing but such products, and how many you need.

It can. Multiply a vector by the matrix, then multiply the result by it again, and keep going. The vector turns toward the top eigenvector by itself. This chapter shows why, how fast, and what can go wrong.

## What you already know

Four ideas carry this chapter:

- An eigenvector of a matrix $A$ is a direction that $A$ only stretches. The stretch factor is the eigenvalue, so $Av = \lambda v$ (see "The SVD hands you every direction at once").
- A symmetric matrix such as a covariance table has eigenvectors at right angles to each other, so any vector is a mix of them.
- The variance along a unit direction $u$ is $u^{\top} C u$, a weighted average of the eigenvalues.
- The norm of a vector is its length, `np.linalg.norm(v)`.

## Stretch, and the vector turns

Take the matrix $A$ with rows $(2, 0)$ and $(0, 1)$. Its eigenvectors are the two axes, with eigenvalues 2 and 1. It stretches the x direction twice as much as the y direction. Start from $v = (1, 1)/\sqrt{2} = (0.71, 0.71)$ and multiply: $Av = (1.41, 0.71)$. The x part doubled and the y part stayed, so the vector leans toward the x axis. Multiply again and it leans further.

Why not look for the best direction by sweeping an angle, as in "Spread, shadows, and the first direction"? In 50,000 dimensions the sweep is hopeless. Why not call an eigen-solver? That is the $10^{14}$ steps above. Repeated multiplication costs one product per step.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="The vector (0.71, 0.71) swinging toward the x axis in three steps, each step closing about half of the remaining gap">
<defs><marker id="c051-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c051-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker></defs>
<polyline points="350.0,285.0 349.9,279.5 349.8,274.0 349.5,268.5 349.2,263.0 348.7,257.5 348.2,252.1 347.5,246.6 346.8,241.1 345.9,235.7 345.0,230.3 343.9,224.9 342.8,219.5 341.5,214.1 340.2,208.8 338.7,203.5 337.2,198.2 335.5,193.0 333.8,187.7 332.0,182.6 330.0,177.4 328.0,172.3 325.9,167.2 323.7,162.2 321.4,157.2 319.0,152.2 316.5,147.3 314.0,142.4 311.3,137.6 308.6,132.9 305.7,128.1 302.8,123.5 299.8,118.9 296.7,114.3 293.5,109.8 290.3,105.4 286.9,101.0 283.5,96.7 280.0,92.4 276.5,88.2 272.8,84.1 269.1,80.1 265.3,76.1 261.4,72.2 257.5,68.3 253.5,64.6 249.4,60.9 245.3,57.3 241.1,53.7 236.8,50.2 232.5,46.9 228.1,43.6 223.6,40.3 219.1,37.2 214.5,34.1 209.9,31.2 205.2,28.3 200.5,25.5 195.7,22.7 190.8,20.1" fill="none" class="stroke-muted"/>
<line x1="50.0" y1="285.0" x2="370.0" y2="285.0" class="stroke-grid" />
<line x1="50.0" y1="285.0" x2="50.0" y2="0.0" class="stroke-grid" />
<line x1="50.0" y1="285.0" x2="259.0" y2="76.0" class="stroke-b" stroke-width="2.5" marker-end="url(#c051-arrow)"/>
<circle cx="262.1" cy="72.9" r="4" class="fill-b"/>
<text x="272.1" y="76.9" class="fig-small" text-anchor="start">start: (0.7071, 0.7071), 45.0°</text>
<line x1="50.0" y1="285.0" x2="314.3" y2="152.8" class="stroke-b" stroke-width="2.5" marker-end="url(#c051-arrow)"/>
<circle cx="318.3" cy="150.8" r="4" class="fill-b"/>
<text x="328.3" y="154.8" class="fig-small" text-anchor="start">step 1: (0.8944, 0.4472), 26.6°</text>
<line x1="50.0" y1="285.0" x2="336.7" y2="213.3" class="stroke-b" stroke-width="2.5" marker-end="url(#c051-arrow)"/>
<circle cx="341.0" cy="212.2" r="4" class="fill-b"/>
<text x="351.0" y="216.2" class="fig-small" text-anchor="start">step 2: (0.9701, 0.2425), 14.0°</text>
<line x1="50.0" y1="285.0" x2="343.2" y2="248.3" class="stroke-b" stroke-width="2.5" marker-end="url(#c051-arrow)"/>
<circle cx="347.7" cy="247.8" r="4" class="fill-b"/>
<text x="357.7" y="251.8" class="fig-small" text-anchor="start">step 3: (0.9923, 0.1240), 7.1°</text>
<circle cx="350.0" cy="285.0" r="5" class="fill-ink"/>
<text x="344.0" y="305.0" class="fig-bold" text-anchor="end">top eigenvector (1, 0)</text>
</svg>
Figure 1. Starting from (0.71, 0.71), each multiplication by the matrix with eigenvalues 2 and 1 stretches the x part more than the y part, so the vector swings toward the x axis. The angle roughly halves each step, because the ratio of the two eigenvalues is one half.
```

Figure 1 follows the vector for three steps. It swings toward the x axis, a little less each time.

## The mixture argument

Here is why it works. Write the start as a mix of the two eigenvectors, $v_0 = c_1 e_1 + c_2 e_2$, where $e_1$ is the top eigenvector. A product by $A$ stretches each part by its own eigenvalue, so after $t$ products

$$
A^t v_0 = c_1 \lambda_1^t\, e_1 + c_2 \lambda_2^t\, e_2 .
$$

Read it aloud as "each part of the mix has been stretched by its eigenvalue, once for every step". On the example, $\lambda_1 = 2$ and $\lambda_2 = 1$, and the start has equal parts $c_1 = c_2$. After $t$ steps the parts are in the ratio $2^t : 1$, which is $1:1$, $2:1$, $4:1$, $16:1$, and $256:1$ at $t = 0, 1, 2, 4, 8$. The share of the second part falls by a factor $\lambda_2 / \lambda_1 = 1/2$ each step. Figure 2 shows the mix turning almost pure.

```figure
<svg viewBox="0 0 600 285" role="img" aria-label="Pairs of bars for t equal 0, 1, 2, 4, and 8: the part of the vector along the top eigenvector grows toward 1 while the other part shrinks toward 0">
<text x="70.0" y="20.0" class="fig-bold" text-anchor="start">the vector after t steps, scaled to length 1</text>
<line x1="70.0" y1="240.0" x2="550.0" y2="240.0" class="stroke-grid" />
<text x="62.0" y="244.0" class="fig-small" text-anchor="end">0.0</text>
<line x1="70.0" y1="160.0" x2="550.0" y2="160.0" class="stroke-grid" />
<text x="62.0" y="164.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="80.0" x2="550.0" y2="80.0" class="stroke-grid" />
<text x="62.0" y="84.0" class="fig-small" text-anchor="end">1.0</text>
<rect x="81.5" y="126.9" width="32.6" height="113.1" class="fill-b"/>
<rect x="118.2" y="126.9" width="32.6" height="113.1" class="fill-a"/>
<text x="97.8" y="121.9" class="fig-small" text-anchor="middle">0.71</text>
<text x="134.5" y="121.9" class="fig-small" text-anchor="middle">0.71</text>
<text x="118.0" y="258.0" class="fig-small" text-anchor="middle">t = 0</text>
<text x="118.0" y="276.0" class="fig-small" text-anchor="middle">1 : 1</text>
<rect x="177.5" y="96.9" width="32.6" height="143.1" class="fill-b"/>
<rect x="214.2" y="168.4" width="32.6" height="71.6" class="fill-a"/>
<text x="193.8" y="91.9" class="fig-small" text-anchor="middle">0.89</text>
<text x="230.5" y="163.4" class="fig-small" text-anchor="middle">0.45</text>
<text x="214.0" y="258.0" class="fig-small" text-anchor="middle">t = 1</text>
<text x="214.0" y="276.0" class="fig-small" text-anchor="middle">2 : 1</text>
<rect x="273.5" y="84.8" width="32.6" height="155.2" class="fill-b"/>
<rect x="310.2" y="201.2" width="32.6" height="38.8" class="fill-a"/>
<text x="289.8" y="79.8" class="fig-small" text-anchor="middle">0.97</text>
<text x="326.5" y="196.2" class="fig-small" text-anchor="middle">0.24</text>
<text x="310.0" y="258.0" class="fig-small" text-anchor="middle">t = 2</text>
<text x="310.0" y="276.0" class="fig-small" text-anchor="middle">4 : 1</text>
<rect x="369.5" y="80.3" width="32.6" height="159.7" class="fill-b"/>
<rect x="406.2" y="230.0" width="32.6" height="10.0" class="fill-a"/>
<text x="385.8" y="75.3" class="fig-small" text-anchor="middle">1.00</text>
<text x="422.5" y="225.0" class="fig-small" text-anchor="middle">0.06</text>
<text x="406.0" y="258.0" class="fig-small" text-anchor="middle">t = 4</text>
<text x="406.0" y="276.0" class="fig-small" text-anchor="middle">16 : 1</text>
<rect x="465.5" y="80.0" width="32.6" height="160.0" class="fill-b"/>
<rect x="502.2" y="239.4" width="32.6" height="1.0" class="fill-a"/>
<text x="481.8" y="75.0" class="fig-small" text-anchor="middle">1.00</text>
<text x="518.5" y="234.4" class="fig-small" text-anchor="middle">0.00</text>
<text x="502.0" y="258.0" class="fig-small" text-anchor="middle">t = 8</text>
<text x="502.0" y="276.0" class="fig-small" text-anchor="middle">256 : 1</text>
<rect x="70.0" y="34.0" width="12.0" height="12.0" class="fill-b"/>
<text x="88.0" y="45.0" class="fig-small" text-anchor="start">part along the top eigenvector</text>
<rect x="320.0" y="34.0" width="12.0" height="12.0" class="fill-a"/>
<text x="338.0" y="45.0" class="fig-small" text-anchor="start">part along the other eigenvector</text>
</svg>
Figure 2. Write the start as an equal mix of the two eigenvectors. After t steps the mix is 2^t parts of the top eigenvector to 1 part of the other. Scaled to length 1, the top part tends to 1 and the other tends to 0.
```

Two conditions are hiding in that argument. If $c_1 = 0$, the top part is missing and no amount of stretching creates it. And the part with the largest $|\lambda|$ wins, so a big negative eigenvalue would beat a smaller positive one. A covariance table has no negative eigenvalues, so the second condition is harmless there.

## Rescale at every step

The stretching has a cost. The length of $A^t v_0$ grows like $\lambda_1^t$, and $2^{1024}$ is already too big for double precision, so after a thousand steps the numbers overflow. Only the direction matters, so divide by the length after every product. Rescaling does not change the direction.

$$
v \leftarrow \frac{A v}{\lVert A v \rVert}
$$

Read it aloud as "multiply by $A$, then divide by the length of the result so that $v$ stays a unit vector". Here are the first two steps on the example:

| Step | $Av$ | Length of $Av$ | Next $v$ |
|---|---|---|---|
| 1 | $(1.4142,\ 0.7071)$ | $\sqrt{2.5} = 1.5811$ | $(0.8944,\ 0.4472)$ |
| 2 | $(1.7889,\ 0.4472)$ | $\sqrt{3.4} = 1.8439$ | $(0.9701,\ 0.2425)$ |

The ratio of y to x is $1$, then $1/2$, then $1/4$. It halves because $\lambda_2/\lambda_1 = 1/2$. In numpy:

```python
v = A @ v
v = v / np.linalg.norm(v)
```

## Read off the eigenvalue

The vector settles on the top eigenvector. The eigenvalue comes from the **Rayleigh quotient**, $v^{\top} A v$ for a unit vector $v$. At an eigenvector, $v^{\top} A v = \lambda\, v^{\top} v = \lambda$. Away from one, $v$ is a mix with squared weights $a_i^2$ that add to 1, and the quotient is the weighted average $\sum_i \lambda_i a_i^2$. This is the same weighted average as the variance along a direction in the previous chapters, so for a covariance table the Rayleigh quotient is exactly the variance along $v$, the number you are trying to maximise.

On the example, $v = (0.8944, 0.4472)$ gives $2 \cdot 0.8 + 1 \cdot 0.2 = 1.8$. The next $v = (0.9701, 0.2425)$ gives $33/17 = 1.9412$. The true value is 2. For a vector that is not unit length, divide by $v^{\top} v$:

```python
eigenvalue = v @ A @ v / (v @ v)
```

## How many steps

The direction error shrinks by about $\lambda_2/\lambda_1$ per step. To measure it, use the sine of the angle between $v$ and the true eigenvector, which is $\sqrt{1 - (v \cdot e_1)^2}$. Cutting that error by a factor of 1000 takes $\ln(0.001)/\ln(\lambda_2/\lambda_1)$ steps:

| Ratio $\lambda_2/\lambda_1$ | Steps for a 1000-fold cut |
|---|---|
| 0.5 | 10 |
| 0.8 | 31 |
| 0.95 | 135 |

A small gap between the top two eigenvalues means a slow crawl. The eigenvalue estimate converges faster than the direction, because the Rayleigh error grows with the squared sine of the angle, so it shrinks by the square of the ratio each step. Figure 3 shows both.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Error against iteration on a log scale: three straight lines for ratios 0.5, 0.8, and 0.95, and a dashed line for the eigenvalue estimate that falls twice as fast">
<line x1="60.0" y1="25.0" x2="440.0" y2="25.0" class="stroke-grid" />
<text x="52.0" y="29.0" class="fig-small" text-anchor="end">1</text>
<line x1="60.0" y1="68.0" x2="440.0" y2="68.0" class="stroke-grid" />
<text x="52.0" y="72.0" class="fig-small" text-anchor="end">1e-2</text>
<line x1="60.0" y1="111.0" x2="440.0" y2="111.0" class="stroke-grid" />
<text x="52.0" y="115.0" class="fig-small" text-anchor="end">1e-4</text>
<line x1="60.0" y1="154.0" x2="440.0" y2="154.0" class="stroke-grid" />
<text x="52.0" y="158.0" class="fig-small" text-anchor="end">1e-6</text>
<line x1="60.0" y1="197.0" x2="440.0" y2="197.0" class="stroke-grid" />
<text x="52.0" y="201.0" class="fig-small" text-anchor="end">1e-8</text>
<line x1="60.0" y1="240.0" x2="440.0" y2="240.0" class="stroke-grid" />
<text x="52.0" y="244.0" class="fig-small" text-anchor="end">1e-10</text>
<text x="60.0" y="258.0" class="fig-small" text-anchor="middle">0</text>
<text x="186.7" y="258.0" class="fig-small" text-anchor="middle">10</text>
<text x="313.3" y="258.0" class="fig-small" text-anchor="middle">20</text>
<text x="440.0" y="258.0" class="fig-small" text-anchor="middle">30</text>
<polyline points="60.0,28.2 72.7,32.5 85.3,38.2 98.0,44.5 110.7,50.9 123.3,57.4 136.0,63.8 148.7,70.3 161.3,76.8 174.0,83.2 186.7,89.7 199.3,96.2 212.0,102.7 224.7,109.1 237.3,115.6 250.0,122.1 262.7,128.6 275.3,135.0 288.0,141.5 300.7,148.0 313.3,154.4 326.0,160.9 338.7,167.4 351.3,173.9 364.0,180.3 376.7,186.8 389.3,193.3 402.0,199.7 414.7,206.2 427.3,212.7 440.0,219.2" fill="none" class="stroke-a" stroke-width="2.5"/>
<polyline points="60.0,28.2 72.7,29.4 85.3,30.8 98.0,32.3 110.7,34.1 123.3,35.9 136.0,37.8 148.7,39.8 161.3,41.8 174.0,43.8 186.7,45.9 199.3,48.0 212.0,50.0 224.7,52.1 237.3,54.2 250.0,56.3 262.7,58.3 275.3,60.4 288.0,62.5 300.7,64.6 313.3,66.7 326.0,68.8 338.7,70.8 351.3,72.9 364.0,75.0 376.7,77.1 389.3,79.2 402.0,81.3 414.7,83.3 427.3,85.4 440.0,87.5" fill="none" class="stroke-c" stroke-width="2.5"/>
<polyline points="60.0,46.5 72.7,48.8 85.3,51.6 98.0,54.7 110.7,58.1 123.3,61.8 136.0,65.7 148.7,69.6 161.3,73.6 174.0,77.7 186.7,81.8 199.3,85.9 212.0,90.1 224.7,94.2 237.3,98.4 250.0,102.5 262.7,106.7 275.3,110.9 288.0,115.0 300.7,119.2 313.3,123.4 326.0,127.5 338.7,131.7 351.3,135.9 364.0,140.0 376.7,144.2 389.3,148.4 402.0,152.5 414.7,156.7 427.3,160.9 440.0,165.0" fill="none" class="stroke-c dashed" stroke-width="2.5"/>
<polyline points="60.0,28.2 72.7,28.5 85.3,28.7 98.0,29.0 110.7,29.3 123.3,29.6 136.0,29.9 148.7,30.2 161.3,30.5 174.0,30.9 186.7,31.2 199.3,31.6 212.0,31.9 224.7,32.3 237.3,32.7 250.0,33.1 262.7,33.5 275.3,33.9 288.0,34.3 300.7,34.7 313.3,35.1 326.0,35.6 338.7,36.0 351.3,36.4 364.0,36.9 376.7,37.3 389.3,37.8 402.0,38.2 414.7,38.7 427.3,39.1 440.0,39.6" fill="none" class="stroke-d" stroke-width="2.5"/>
<text x="448.0" y="223.2" class="fig-small" text-anchor="start">ratio 0.5</text>
<text x="448.0" y="91.5" class="fig-small" text-anchor="start">ratio 0.8</text>
<text x="448.0" y="43.6" class="fig-small" text-anchor="start">ratio 0.95</text>
<text x="448.0" y="169.0" class="fig-small" text-anchor="start">ratio 0.8, eigenvalue</text>
<text x="448.0" y="185.0" class="fig-small" text-anchor="start">estimate (dashed)</text>
<text x="250.0" y="276.0" text-anchor="middle">iteration t</text>
<text transform="rotate(-90 14 130)" x="14.0" y="130.0" text-anchor="middle">error (log scale)</text>
</svg>
Figure 3. Error on a log scale for three ratios of the top two eigenvalues. Each is a straight line, and the closer the ratio is to 1 (the smaller the gap), the shallower the slope. The dashed line is the error in the eigenvalue estimate for ratio 0.8, which falls twice as steeply as the direction error.
```

## Get the second direction by deflation

To find the next direction, remove the first. **Deflation** subtracts the part of $A$ that the first eigenvector accounts for:

$$
A' = A - \lambda_1\, v_1 v_1^{\top}
$$

Read it aloud as "take $A$ and subtract its stretch along $v_1$". The matrix $A'$ stretches $v_1$ by 0 and every direction at right angles to $v_1$ exactly as before, so its top eigenvalue is $\lambda_2$. Run power iteration on $A'$ to get the second direction.

On the three-point example from the earlier chapters, $C$ has rows $(4, 2)$ and $(2, 4)$, with $\lambda_1 = 6$ and $v_1 = (1,1)/\sqrt{2}$. Then $\lambda_1 v_1 v_1^{\top}$ has all four entries equal to 3, and $A'$ has rows $(1, -1)$ and $(-1, 1)$. Its top eigenvalue is 2, with direction $(1,-1)/\sqrt{2}$, which is the second component you found before.

```python
A_deflated = A - eigenvalue * np.outer(v, v)
```

> [!KEY]
> You can find the top direction with nothing but products: multiply, rescale, repeat. The largest eigenvalue's part grows fastest, so the vector turns toward its eigenvector, at a speed set by $\lambda_2/\lambda_1$. The Rayleigh quotient reads the eigenvalue, and deflation uncovers the next direction.

> [!RECAP]
> - Each product stretches each eigenvector part by its eigenvalue, so the largest one takes over.
> - Rescale to length 1 at every step, or the numbers overflow.
> - The Rayleigh quotient `v @ A @ v / (v @ v)` reads the eigenvalue, and for a covariance table it is the variance along `v`.
> - The direction error shrinks by $\lambda_2/\lambda_1$ per step, and the eigenvalue error by its square.
> - Deflation, `A - eigenvalue * np.outer(v, v)`, exposes the next direction.

## Build it step by step

`center` and `pca_svd` are written for you in the starter, so the last check can compare your power iteration with the SVD. Build three functions, in this order.

1. **`rayleigh_quotient(A, v)`** takes a symmetric matrix `(d, d)` and a vector `(d,)`, and returns the float `v @ A @ v / (v @ v)`. The usual mistake is to forget the division when `v` is not unit length. The shape of the code:

    ```python
    def rayleigh_quotient(A, v):
        # v @ A @ v, divided by v @ v.
    ```

2. **`power_iteration(A, iters)`** returns `(v, eigenvalue)`. Start from `np.ones(d) / np.sqrt(d)`, multiply and rescale `iters` times, and finish with your `rayleigh_quotient`. The usual mistake is to rescale only after the loop, which overflows for large `iters`. The shape of the code:

    ```python
    def power_iteration(A, iters):
        # Multiply by A and divide by the length, every step.
    ```

3. **`deflate(A, v, eigenvalue)`** returns `A - eigenvalue * np.outer(v, v)`, shape `(d, d)`, for a unit vector `v`. The usual mistake is `v @ v`, which is a number, in place of `np.outer(v, v)`, which is a matrix. The shape of the code:

    ```python
    def deflate(A, v, eigenvalue):
        # Subtract the part of A along v.
    ```

## Cost and scaling

One step costs about $d^2$ operations for a dense $A$, and about the number of nonzero entries for a sparse one. For a covariance table, you can skip building it: `Xc.T @ (Xc @ v) / (n - 1)` costs about $2nd$ per step and needs no $d \times d$ storage. Power iteration for $t$ steps costs $t$ times that, against about $d^3$ for a full eigendecomposition.

## Where this breaks

- If the start has no part along the top eigenvector, in exact arithmetic it never finds it. The fixed start `np.ones(d) / np.sqrt(d)` is orthogonal to the top eigenvector for some matrices.
- Deflation inherits any error in $v_1$ and $\lambda_1$. A little of the first direction survives in $A'$, and it can pull the second run off course, so run the first search to a tight tolerance.
- If $\lambda_1 = \lambda_2$, nothing prefers one direction over the other, so the result depends on the start. If they are close, convergence crawls.
- A large negative eigenvalue wins by size, and the vector flips sign at every step. The Rayleigh quotient still reports the negative eigenvalue, because the quotient does not care about the sign of $v$.

## The exercise

You write `rayleigh_quotient`, `power_iteration`, and `deflate`. The checks confirm the hand numbers: a quotient of 2 at an eigenvector and 1.5 at $(1,1)/\sqrt{2}$, and steps 1 and 2 of the trace to $(0.8944, 0.4472)$ with 1.8, and $(0.9701, 0.2425)$ with $33/17$. They run thousands of steps to confirm there is no overflow, measure the direction error over steps 10 to 30 against $\lambda_2/\lambda_1$ and the eigenvalue error against its square, check that deflation leaves the second eigenvalue on top, and confirm that power iteration on the covariance matches the first two SVD components up to sign.
