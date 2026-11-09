---
title: "Power iteration and probabilistic PCA"
slug: 024-power-iteration-and-probabilistic-pca
part: "Part 3 — Trees, ensembles, and unsupervised learning"
summary: Find the top eigenvector from repeated products alone, then add Gaussian fuzz to PCA and fit it by guess-and-refit EM.
minutes: 75
gpu: false
objectives:
  - Explain why repeated multiplication turns a vector toward the eigenvector with the largest eigenvalue, and read the eigenvalue from the Rayleigh quotient.
  - Predict how many steps power iteration needs from the ratio of the top two eigenvalues, and find a second direction by deflation.
  - Describe probabilistic PCA as a hidden coordinate pushed through a matrix W, plus Gaussian fuzz.
  - Run one E-step and one M-step by hand on a two-point example, and compute the log-likelihood with a log-determinant and a solve.
  - Explain why the fitted W is known only up to a rotation, and compare the fitted plane with the PCA plane.
prereqs:
  - 023-principal-components
exercise: 024-power-iteration-and-probabilistic-pca
---

# Power iteration and probabilistic PCA

Suppose your table has 50,000 columns. Its covariance table has $50{,}000^2 = 2.5$ billion entries, which is 20 GB in double precision, and a full eigendecomposition takes about $50{,}000^3 \approx 10^{14}$ steps. You cannot afford either, but you still want the top direction. Multiplying the covariance table by one vector is cheap, because $X_c^{\top}(X_c v)$ never builds the table at all. So can the top direction come from nothing but such products?

It can. Multiply a vector by the matrix, then multiply the result by it again, and the vector turns toward the top eigenvector by itself. The first half of this chapter shows why, how fast, and what can go wrong.

The second half asks a different question about the same sheet. PCA draws a perfectly flat sheet through the data, but real data never lies exactly on it: dust grains on a flour pancake each wobble a little off it. If the sheet is the whole truth, any real table has probability zero. You put the wobble into the model, which gives every table a likelihood, and then you fit it with a loop that guesses the hidden positions on the sheet and refits from the guesses.

## What you already know

- An eigenvector of a matrix $A$ is a direction that $A$ only stretches. The stretch factor is the eigenvalue, so $Av = \lambda v$, as in [principal components](/lessons/023-principal-components).
- A symmetric matrix such as a covariance table has eigenvectors at right angles to each other, so any vector is a mix of them. The variance along a unit direction $u$ is $u^{\top} C u$, a weighted average of the eigenvalues.
- PCA finds a flat sheet, spanned by the top directions of the centred data, with the SVD.
- A **Gaussian** is a bell-shaped spread described by a mean and a variance. The log-likelihood is the log of the probability the model gave to the data you saw, and larger is better ([log loss](/lessons/008-sigmoid-and-log-loss)).

## Stretch, and the vector turns

Take the matrix $A$ with rows $(2, 0)$ and $(0, 1)$. Its eigenvectors are the two axes, with eigenvalues 2 and 1. It stretches the x direction twice as much as the y direction. Start from $v = (1, 1)/\sqrt{2} = (0.71, 0.71)$ and multiply: $Av = (1.41, 0.71)$. The x part doubled and the y part stayed, so the vector leans toward the x axis. Multiply again and it leans further, as Figure 1 shows.

Why not sweep an angle, as in [principal components](/lessons/023-principal-components)? In 50,000 dimensions the sweep is hopeless. Why not call an eigen-solver? That is the $10^{14}$ steps above. Repeated multiplication costs one product per step.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="The vector (0.71, 0.71) swinging toward the x axis in three steps, each step closing about half of the remaining gap">
<defs><marker id="c023-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c023-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker></defs>
<polyline points="350.0,285.0 349.9,279.5 349.8,274.0 349.5,268.5 349.2,263.0 348.7,257.5 348.2,252.1 347.5,246.6 346.8,241.1 345.9,235.7 345.0,230.3 343.9,224.9 342.8,219.5 341.5,214.1 340.2,208.8 338.7,203.5 337.2,198.2 335.5,193.0 333.8,187.7 332.0,182.6 330.0,177.4 328.0,172.3 325.9,167.2 323.7,162.2 321.4,157.2 319.0,152.2 316.5,147.3 314.0,142.4 311.3,137.6 308.6,132.9 305.7,128.1 302.8,123.5 299.8,118.9 296.7,114.3 293.5,109.8 290.3,105.4 286.9,101.0 283.5,96.7 280.0,92.4 276.5,88.2 272.8,84.1 269.1,80.1 265.3,76.1 261.4,72.2 257.5,68.3 253.5,64.6 249.4,60.9 245.3,57.3 241.1,53.7 236.8,50.2 232.5,46.9 228.1,43.6 223.6,40.3 219.1,37.2 214.5,34.1 209.9,31.2 205.2,28.3 200.5,25.5 195.7,22.7 190.8,20.1" fill="none" class="stroke-muted"/>
<line x1="50.0" y1="285.0" x2="370.0" y2="285.0" class="stroke-grid" />
<line x1="50.0" y1="285.0" x2="50.0" y2="0.0" class="stroke-grid" />
<line x1="50.0" y1="285.0" x2="259.0" y2="76.0" class="stroke-b" stroke-width="2.5" marker-end="url(#c023-arrow)"/>
<circle cx="262.1" cy="72.9" r="4" class="fill-b"/>
<text x="272.1" y="76.9" class="fig-small" text-anchor="start">start: (0.7071, 0.7071), 45.0°</text>
<line x1="50.0" y1="285.0" x2="314.3" y2="152.8" class="stroke-b" stroke-width="2.5" marker-end="url(#c023-arrow)"/>
<circle cx="318.3" cy="150.8" r="4" class="fill-b"/>
<text x="328.3" y="154.8" class="fig-small" text-anchor="start">step 1: (0.8944, 0.4472), 26.6°</text>
<line x1="50.0" y1="285.0" x2="336.7" y2="213.3" class="stroke-b" stroke-width="2.5" marker-end="url(#c023-arrow)"/>
<circle cx="341.0" cy="212.2" r="4" class="fill-b"/>
<text x="351.0" y="216.2" class="fig-small" text-anchor="start">step 2: (0.9701, 0.2425), 14.0°</text>
<line x1="50.0" y1="285.0" x2="343.2" y2="248.3" class="stroke-b" stroke-width="2.5" marker-end="url(#c023-arrow)"/>
<circle cx="347.7" cy="247.8" r="4" class="fill-b"/>
<text x="357.7" y="251.8" class="fig-small" text-anchor="start">step 3: (0.9923, 0.1240), 7.1°</text>
<circle cx="350.0" cy="285.0" r="5" class="fill-ink"/>
<text x="344.0" y="305.0" class="fig-bold" text-anchor="end">top eigenvector (1, 0)</text>
</svg>
Figure 1. Starting from (0.71, 0.71), each multiplication by the matrix with eigenvalues 2 and 1 stretches the x part more than the y part, so the vector swings toward the x axis. The angle roughly halves each step, because the ratio of the two eigenvalues is one half.
```

Here is why it works. Write the start as a mix of the two eigenvectors, $v_0 = c_1 e_1 + c_2 e_2$, where $e_1$ is the top eigenvector. A product by $A$ stretches each part by its own eigenvalue, so after $t$ products

$$
A^t v_0 = c_1 \lambda_1^t\, e_1 + c_2 \lambda_2^t\, e_2 .
$$

Read it aloud as "each part of the mix has been stretched by its eigenvalue, once for every step". On the example the start has equal parts, so after $t$ steps the parts are in the ratio $2^t : 1$, which is $1:1$, $2:1$, $4:1$, $16:1$, and $256:1$ at $t = 0, 1, 2, 4, 8$. The share of the second part falls by a factor $\lambda_2 / \lambda_1 = 1/2$ each step.

Two conditions are hiding in that argument. If $c_1 = 0$, the top part is missing and no amount of stretching creates it. And the part with the largest $|\lambda|$ wins, so a big negative eigenvalue would beat a smaller positive one. A covariance table has no negative eigenvalues, so the second condition is harmless there.

## Rescale, and read off the eigenvalue

The stretching has a cost. The length of $A^t v_0$ grows like $\lambda_1^t$, and $2^{1024}$ is already too big for double precision, so after a thousand steps the numbers overflow. Only the direction matters, so divide by the length after every product. Rescaling does not change the direction.

$$
v \leftarrow \frac{A v}{\lVert A v \rVert}
$$

Read it aloud as "multiply by $A$, then divide by the length of the result so that $v$ stays a unit vector". Here are the first two steps on the example:

| Step | $Av$ | Length of $Av$ | Next $v$ |
|---|---|---|---|
| 1 | $(1.4142,\ 0.7071)$ | $\sqrt{2.5} = 1.5811$ | $(0.8944,\ 0.4472)$ |
| 2 | $(1.7889,\ 0.4472)$ | $\sqrt{3.4} = 1.8439$ | $(0.9701,\ 0.2425)$ |

The ratio of y to x is $1$, then $1/2$, then $1/4$. In numpy:

```python
v = A @ v
v = v / np.linalg.norm(v)
```

The vector settles on the top eigenvector, and the eigenvalue comes from the **Rayleigh quotient**, $v^{\top} A v$ for a unit vector $v$. At an eigenvector, $v^{\top} A v = \lambda\, v^{\top} v = \lambda$. Away from one, $v$ is a mix with squared weights $a_i^2$ that add to 1, and the quotient is the weighted average $\sum_i \lambda_i a_i^2$. For a covariance table that is exactly the variance along $v$, the number you are trying to maximise. On the example, $v = (0.8944, 0.4472)$ gives $2 \cdot 0.8 + 1 \cdot 0.2 = 1.8$. The next $v = (0.9701, 0.2425)$ gives $33/17 = 1.9412$. The true value is 2. For a vector that is not unit length, divide by $v^{\top} v$:

```python
eigenvalue = v @ A @ v / (v @ v)
```

## How many steps, and the next direction

The direction error shrinks by about $\lambda_2/\lambda_1$ per step. To measure it, use the sine of the angle between $v$ and the true eigenvector, which is $\sqrt{1 - (v \cdot e_1)^2}$. Cutting that error by a factor of 1000 takes $\ln(0.001)/\ln(\lambda_2/\lambda_1)$ steps:

| Ratio $\lambda_2/\lambda_1$ | Steps for a 1000-fold cut |
|---|---|
| 0.5 | 10 |
| 0.8 | 31 |
| 0.95 | 135 |

A small gap between the top two eigenvalues means a slow crawl. The eigenvalue estimate converges faster than the direction, because the Rayleigh error grows with the squared sine of the angle, so it shrinks by the square of the ratio each step. Figure 2 shows both.

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
Figure 2. Error on a log scale for three ratios of the top two eigenvalues. Each is a straight line, and the closer the ratio is to 1 (the smaller the gap), the shallower the slope. The dashed line is the error in the eigenvalue estimate for ratio 0.8, which falls twice as steeply as the direction error.
```

To find the next direction, remove the first. **Deflation** subtracts the part of $A$ that the first eigenvector accounts for:

$$
A' = A - \lambda_1\, v_1 v_1^{\top}
$$

Read it aloud as "take $A$ and subtract its stretch along $v_1$". The matrix $A'$ stretches $v_1$ by 0 and every direction at right angles to $v_1$ exactly as before, so its top eigenvalue is $\lambda_2$. On the three-point example from [principal components](/lessons/023-principal-components), $C$ has rows $(4, 2)$ and $(2, 4)$, with $\lambda_1 = 6$ and $v_1 = (1,1)/\sqrt{2}$. Then $\lambda_1 v_1 v_1^{\top}$ has all four entries equal to 3, and $A'$ has rows $(1, -1)$ and $(-1, 1)$. Its top eigenvalue is 2, with direction $(1,-1)/\sqrt{2}$, the second component you found before.

```python
A_deflated = A - eigenvalue * np.outer(v, v)
```

> [!KEY]
> You can find the top direction with nothing but products: multiply, rescale, repeat. The largest eigenvalue's part grows fastest, so the vector turns toward its eigenvector, at a speed set by $\lambda_2/\lambda_1$. The Rayleigh quotient reads the eigenvalue, and deflation uncovers the next direction.

## A sheet plus fuzz

Now the second question. Here is the story the model tells. Nature first picks a hidden coordinate $z$ for each point, its position on the sheet. It draws $z$ from the standard bell: centred on 0, with variance 1 in each coordinate. A **hidden** variable (also called a latent variable) is one you never observe. Then a matrix $W$ pushes $z$ into the full space, and a little Gaussian fuzz, with variance $\sigma^2$ in every direction, pushes the point off the sheet:

$$
x = W z + \text{fuzz}.
$$

Figure 3 draws it for one hidden coordinate. $W$ is $d \times k$, so $W z$ is a point on a $k$-dimensional sheet, and the observed $x$ lands near it. $W$ says where the sheet is, and $\sigma^2$ says how thick it is.

```figure
<svg viewBox="0 0 600 340" role="img" aria-label="A tilted line of points W z, each with a dashed circle of fuzz around it and one observed point that sits off the line">
<defs><marker id="c023-m-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c023-m-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker></defs>
<line x1="22.7" y1="308.7" x2="577.3" y2="31.3" class="stroke-b" stroke-width="2.5"/>
<text x="571.3" y="21.3" class="fig-small" text-anchor="end">the sheet: all points W z</text>
<ellipse cx="140.0" cy="250.0" rx="33.3" ry="33.3" class="stroke-muted dashed" fill="none"/>
<line x1="140.0" y1="250.0" x2="117.4" y2="236.8" class="stroke-ink" stroke-width="1.5"/>
<ellipse cx="332.0" cy="154.0" rx="33.3" ry="33.3" class="stroke-muted dashed" fill="none"/>
<line x1="332.0" y1="154.0" x2="319.3" y2="190.5" class="stroke-ink" stroke-width="1.5"/>
<ellipse cx="438.7" cy="100.7" rx="33.3" ry="33.3" class="stroke-muted dashed" fill="none"/>
<line x1="438.7" y1="100.7" x2="462.3" y2="112.1" class="stroke-ink" stroke-width="1.5"/>
<circle cx="140.0" cy="250.0" r="4.5" class="fill-b"/>
<circle cx="117.4" cy="236.8" r="5.5" class="fill-a"/>
<circle cx="332.0" cy="154.0" r="4.5" class="fill-b"/>
<circle cx="319.3" cy="190.5" r="5.5" class="fill-a"/>
<circle cx="438.7" cy="100.7" r="4.5" class="fill-b"/>
<circle cx="462.3" cy="112.1" r="5.5" class="fill-a"/>
<text x="140.0" y="310.0" class="fig-bold" text-anchor="middle">z = −1.5</text>
<text x="332.0" y="214.0" class="fig-bold" text-anchor="middle">z = 0.3</text>
<text x="438.7" y="160.7" class="fig-bold" text-anchor="middle">z = 1.3</text>
<text x="474.3" y="130.1" class="fig-small" text-anchor="start">observed x</text>
<text x="410.0" y="72.0" class="fig-small" text-anchor="end">W z on the sheet</text>
<text x="30.0" y="40.0" class="fig-small" text-anchor="start">dashed circle: fuzz of size σ</text>
</svg>
Figure 3. The model has two parts. The hidden coordinate z picks a spot on the sheet (orange dots on the line), and fuzz of size σ pushes the observed point (blue) off the sheet. A fitted model has to explain the blue points without ever seeing the orange ones.
```

The model gives every table a probability, so you can compare fits by how likely each makes the data. Under the model, the centred points are Gaussian with covariance table

$$
C = W W^{\top} + \sigma^2 I .
$$

Read it aloud as "the spread of the sheet, plus the same fuzz in every direction". The first term comes from $z$ having variance 1, so $Wz$ has covariance $WW^{\top}$. The second is the fuzz, which adds its own variance independently. The best fit is known in closed form: the columns of $W$ span the top $k$ PCA directions, and $\sigma^2$ is the average of the variances you dropped, computed with divisor $n$. $W$ itself is not unique, because a rotation $R$ of the hidden coordinates gives $(WR)(WR)^{\top} = WW^{\top}$ and the same model. So you compare the plane that $W$ spans with the PCA plane, not $W$ with the components.

If the closed form exists, why fit it by a loop? Because the loop is the standard way to fit models with hidden quantities, and this is the smallest place to see it.

## Guess, then refit

Fitting has a chicken-and-egg problem. If you knew each point's hidden $z$, fitting $W$ would be an ordinary regression of $x$ on $z$, and $\sigma^2$ would be the average squared leftover. If you knew $W$ and $\sigma^2$, you could guess each $z$. You know neither.

So start with a guess for $W$ and $\sigma^2$, then alternate. A guess is soft: for each point you keep a bell-shaped belief about $z$ rather than one value. This is the idea behind [k-means](/lessons/025-k-means) with soft guesses in place of hard ones, and [Gaussian mixtures](/lessons/026-gaussian-mixtures-and-em) develop it in general. The method is **expectation-maximisation**, or EM. An **expectation** is a weighted average, so the E-step computes the average of what you need from the hidden values, and the M-step picks the model that maximises the likelihood given those averages.

**The E-step.** Given $W$ and $\sigma^2$, define the small $k \times k$ matrix $M = W^{\top} W + \sigma^2 I$. Then, for centred data $X_c$ with $n$ rows:

$$
E_z = X_c W M^{-1}, \qquad E_{zz} = n\,\sigma^2 M^{-1} + E_z^{\top} E_z .
$$

Read it aloud. $E_z$ is the best guess of each point's hidden coordinates: the point's score on $W$, shrunk by $M^{-1}$. The shrinking has two causes. Part of the point's offset is fuzz and not sheet, and the standard bell says that large $z$ values are rare, so the guess leans toward 0. $E_{zz}$ is the sum, over points, of "guess times guess" plus $\sigma^2 M^{-1}$ for each point, which is the blur that remains in a guess even when $W$ is right.

Try the two-point table with rows $(2, 0)$ and $(-2, 0)$, which is already centred. Start with $W = (1, 0)^{\top}$ and $\sigma^2 = 1$. Then $M = 1 + 1 = 2$. The guesses are $E_z = (2\cdot 1/2,\ -2 \cdot 1/2) = (1, -1)$, so the point at 2 gets the guess 1 and not 2, and $E_{zz} = 2 \cdot 1 \cdot \tfrac{1}{2} + (1 + 1) = 3$.

```python
M_inv = np.linalg.inv(W.T @ W + sigma2 * np.eye(k))
Ez = Xc @ W @ M_inv                      # shape (n, k)
Ezz_sum = n * sigma2 * M_inv + Ez.T @ Ez  # shape (k, k)
```

**The M-step.** If the guesses were the true $z$, the updated $W$ would come from the regression formula $X_c^{\top} Z (Z^{\top}Z)^{-1}$. Swap in the expected values:

$$
W_{\text{new}} = X_c^{\top} E_z\, E_{zz}^{-1} .
$$

Then $\sigma^2$ is the average squared leftover per entry. Expanding the squared leftover of each point and averaging over the $n \cdot d$ entries gives

$$
\sigma^2_{\text{new}} = \frac{1}{nd}\Big( \sum X_c^2 \;-\; 2 \sum \big(E_z \odot X_c W_{\text{new}}\big) \;+\; \operatorname{tr}\big(E_{zz}\, W_{\text{new}}^{\top} W_{\text{new}}\big) \Big) .
$$

Read it aloud as "the squared data, minus twice the match between the guesses and the data, plus the squared fit". $\odot$ multiplies entry by entry, and $\operatorname{tr}$, the **trace**, adds the diagonal of a square matrix. On the two-point table, $X_c^{\top} E_z = (4, 0)$, so $W_{\text{new}} = (4/3, 0)$. The squared data sum to 8. Then $X_c W_{\text{new}} = (8/3, -8/3)$, so the twice-sum is $32/3$, and the trace term is $3 \cdot 16/9 = 16/3$. The numerator is $8 - 32/3 + 16/3 = 8/3$. Divide by $nd = 4$ and $\sigma^2_{\text{new}} = 2/3$.

```python
W_new = (Xc.T @ Ez) @ np.linalg.inv(Ezz_sum)
cross = 2 * np.sum(Ez * (Xc @ W_new))
sigma2_new = (np.sum(Xc**2) - cross + np.trace(Ezz_sum @ W_new.T @ W_new)) / (n * d)
```

> [!WARNING]
> Take the cross term from `W_new`, not the old `W`. The leftover is measured from the updated fit, so every term of the $\sigma^2$ formula uses `W_new`.

## Watching the likelihood

After each round, record the log-likelihood of the centred data under the updated $(W, \sigma^2)$. With $S = X_c^{\top} X_c / n$ and $C = WW^{\top} + \sigma^2 I$:

$$
\ell = -\frac{n}{2}\Big( d \log(2\pi) + \log\det C + \operatorname{tr}(C^{-1} S) \Big) .
$$

Read it aloud. $d\log(2\pi)$ is a constant. The **determinant** $\det C$ measures how much volume $C$ spreads over, so $\log \det C$ grows when the model is wide, and the trace term grows when the data sit where $C$ says they should not. A good model is wide enough to cover the data and no wider. In numpy, `np.linalg.slogdet` returns the log of the determinant without overflow, and `np.linalg.solve(C, S)` gives $C^{-1}S$ without forming the inverse.

On the two-point table before the round, $C = \operatorname{diag}(2, 1)$, so $\log\det C = 0.693$, the trace term is $4/2 = 2$, and $d\log 2\pi = 3.676$. They sum to 6.369, and since $n/2 = 1$, $\ell = -6.37$. After the round, $C = \operatorname{diag}(22/9,\ 2/3)$ with $\det C = 44/27$, so $\log\det C = 0.488$ and the trace term is $18/11 = 1.636$. They sum with 3.676 to 5.800, and $\ell = -5.80$. The likelihood rose.

It must rise every round, and that is useful. The E-step builds a lower bound on the likelihood that touches it at the current $(W, \sigma^2)$. The M-step lifts that bound. The true likelihood sits above the bound, so it cannot fall. If the recorded value ever drops, the code has a bug.

On the six-dimensional data from [principal components](/lessons/023-principal-components) ($n = 300$, two strong directions plus noise), Figure 4 shows what happens. The fitted plane matches the PCA plane to within $10^{-10}$ after five rounds. After 300 rounds, $\sigma^2 = 0.03938$, the same as the average of the four dropped variances.

```figure
<svg viewBox="0 0 600 322" role="img" aria-label="Three stacked curves over three hundred EM rounds: the log-likelihood rising and flattening, the distance between the fitted plane and the PCA plane collapsing, and the noise variance settling on its closed-form value">
<line x1="80.0" y1="81.7" x2="520.0" y2="81.7" class="stroke-grid" />
<text x="72.0" y="85.7" class="fig-small" text-anchor="end">-2500</text>
<line x1="80.0" y1="64.4" x2="520.0" y2="64.4" class="stroke-grid" />
<text x="72.0" y="68.4" class="fig-small" text-anchor="end">-2000</text>
<line x1="80.0" y1="47.2" x2="520.0" y2="47.2" class="stroke-grid" />
<text x="72.0" y="51.2" class="fig-small" text-anchor="end">-1500</text>
<line x1="80.0" y1="30.0" x2="520.0" y2="30.0" class="stroke-grid" />
<text x="72.0" y="34.0" class="fig-small" text-anchor="end">-1000</text>
<text x="80.0" y="24.0" class="fig-small" text-anchor="start">log-likelihood</text>
<polyline points="80.0,91.4 81.5,78.8 82.9,66.6 84.4,51.4 85.9,40.6 87.4,35.8 88.8,34.5 90.3,34.2 91.8,34.1 93.2,34.0 94.7,33.9 96.2,33.8 97.7,33.7 99.1,33.7 100.6,33.6 102.1,33.6 103.5,33.5 105.0,33.5 106.5,33.4 108.0,33.4 109.4,33.3 110.9,33.3 112.4,33.3 113.8,33.2 115.3,33.2 116.8,33.2 118.3,33.1 119.7,33.1 121.2,33.1 122.7,33.1 124.1,33.0 125.6,33.0 127.1,33.0 128.6,33.0 130.0,33.0 131.5,32.9 133.0,32.9 134.4,32.9 135.9,32.9 137.4,32.9 138.9,32.9 140.3,32.8 141.8,32.8 143.3,32.8 144.7,32.8 146.2,32.8 147.7,32.8 149.2,32.8 150.6,32.8 152.1,32.8 153.6,32.7 155.1,32.7 156.5,32.7 158.0,32.7 159.5,32.7 160.9,32.7 162.4,32.7 163.9,32.7 165.4,32.7 166.8,32.7 168.3,32.7 169.8,32.7 171.2,32.6 172.7,32.6 174.2,32.6 175.7,32.6 177.1,32.6 178.6,32.6 180.1,32.6 181.5,32.6 183.0,32.6 184.5,32.6 186.0,32.6 187.4,32.6 188.9,32.6 190.4,32.6 191.8,32.6 193.3,32.6 194.8,32.6 196.3,32.6 197.7,32.6 199.2,32.5 200.7,32.5 202.1,32.5 203.6,32.5 205.1,32.5 206.6,32.5 208.0,32.5 209.5,32.5 211.0,32.5 212.4,32.5 213.9,32.5 215.4,32.5 216.9,32.5 218.3,32.5 219.8,32.5 221.3,32.5 222.7,32.5 224.2,32.5 225.7,32.5 227.2,32.5 228.6,32.5 230.1,32.5 231.6,32.5 233.0,32.5 234.5,32.5 236.0,32.5 237.5,32.5 238.9,32.5 240.4,32.5 241.9,32.5 243.3,32.5 244.8,32.5 246.3,32.5 247.8,32.5 249.2,32.5 250.7,32.5 252.2,32.4 253.6,32.4 255.1,32.4 256.6,32.4 258.1,32.4 259.5,32.4 261.0,32.4 262.5,32.4 263.9,32.4 265.4,32.4 266.9,32.4 268.4,32.4 269.8,32.4 271.3,32.4 272.8,32.4 274.2,32.4 275.7,32.4 277.2,32.4 278.7,32.4 280.1,32.4 281.6,32.4 283.1,32.4 284.5,32.4 286.0,32.4 287.5,32.4 289.0,32.4 290.4,32.4 291.9,32.4 293.4,32.4 294.8,32.4 296.3,32.4 297.8,32.4 299.3,32.4 300.7,32.4 302.2,32.4 303.7,32.4 305.2,32.4 306.6,32.4 308.1,32.4 309.6,32.4 311.0,32.4 312.5,32.4 314.0,32.4 315.5,32.4 316.9,32.4 318.4,32.4 319.9,32.4 321.3,32.4 322.8,32.4 324.3,32.4 325.8,32.4 327.2,32.4 328.7,32.4 330.2,32.4 331.6,32.4 333.1,32.4 334.6,32.4 336.1,32.4 337.5,32.4 339.0,32.4 340.5,32.4 341.9,32.4 343.4,32.4 344.9,32.4 346.4,32.4 347.8,32.4 349.3,32.4 350.8,32.4 352.2,32.4 353.7,32.4 355.2,32.4 356.7,32.4 358.1,32.4 359.6,32.4 361.1,32.4 362.5,32.4 364.0,32.4 365.5,32.4 367.0,32.4 368.4,32.4 369.9,32.4 371.4,32.4 372.8,32.4 374.3,32.4 375.8,32.4 377.3,32.4 378.7,32.4 380.2,32.4 381.7,32.4 383.1,32.4 384.6,32.4 386.1,32.4 387.6,32.4 389.0,32.4 390.5,32.4 392.0,32.4 393.4,32.4 394.9,32.4 396.4,32.4 397.9,32.4 399.3,32.4 400.8,32.4 402.3,32.4 403.7,32.4 405.2,32.4 406.7,32.4 408.2,32.4 409.6,32.4 411.1,32.4 412.6,32.4 414.0,32.4 415.5,32.4 417.0,32.4 418.5,32.4 419.9,32.4 421.4,32.4 422.9,32.4 424.3,32.4 425.8,32.4 427.3,32.4 428.8,32.4 430.2,32.4 431.7,32.4 433.2,32.4 434.6,32.4 436.1,32.4 437.6,32.4 439.1,32.4 440.5,32.4 442.0,32.4 443.5,32.4 444.9,32.4 446.4,32.4 447.9,32.4 449.4,32.4 450.8,32.4 452.3,32.4 453.8,32.4 455.3,32.4 456.7,32.4 458.2,32.4 459.7,32.4 461.1,32.4 462.6,32.4 464.1,32.4 465.6,32.4 467.0,32.4 468.5,32.4 470.0,32.4 471.4,32.4 472.9,32.4 474.4,32.4 475.9,32.4 477.3,32.4 478.8,32.4 480.3,32.4 481.7,32.4 483.2,32.4 484.7,32.4 486.2,32.4 487.6,32.4 489.1,32.4 490.6,32.4 492.0,32.4 493.5,32.4 495.0,32.4 496.5,32.4 497.9,32.4 499.4,32.4 500.9,32.4 502.3,32.4 503.8,32.4 505.3,32.4 506.8,32.4 508.2,32.4 509.7,32.4 511.2,32.4 512.6,32.4 514.1,32.4 515.6,32.4 517.1,32.4 518.5,32.4 520.0,32.4" fill="none" class="stroke-b" stroke-width="2.5"/>
<line x1="80.0" y1="125.0" x2="520.0" y2="125.0" class="stroke-grid" />
<text x="72.0" y="129.0" class="fig-small" text-anchor="end">1</text>
<line x1="80.0" y1="142.7" x2="520.0" y2="142.7" class="stroke-grid" />
<text x="72.0" y="146.7" class="fig-small" text-anchor="end">1e-4</text>
<line x1="80.0" y1="160.4" x2="520.0" y2="160.4" class="stroke-grid" />
<text x="72.0" y="164.4" class="fig-small" text-anchor="end">1e-8</text>
<line x1="80.0" y1="178.1" x2="520.0" y2="178.1" class="stroke-grid" />
<text x="72.0" y="182.1" class="fig-small" text-anchor="end">1e-12</text>
<text x="80.0" y="119.0" class="fig-small" text-anchor="start">distance between planes</text>
<polyline points="80.0,140.1 81.5,147.9 82.9,155.7 84.4,163.5 85.9,171.2 87.4,179.0 88.8,186.4 90.3,187.0 91.8,187.0 93.2,187.0 94.7,187.0 96.2,187.0 97.7,187.0 99.1,187.0 100.6,187.0 102.1,187.0 103.5,187.0 105.0,187.0 106.5,187.0 108.0,187.0 109.4,187.0 110.9,187.0 112.4,187.0 113.8,187.0 115.3,187.0 116.8,187.0 118.3,187.0 119.7,187.0 121.2,187.0 122.7,187.0 124.1,187.0 125.6,187.0 127.1,187.0 128.6,187.0 130.0,187.0 131.5,187.0 133.0,187.0 134.4,187.0 135.9,187.0 137.4,187.0 138.9,187.0 140.3,187.0 141.8,187.0 143.3,187.0 144.7,187.0 146.2,187.0 147.7,187.0 149.2,187.0 150.6,187.0 152.1,187.0 153.6,187.0 155.1,187.0 156.5,187.0 158.0,187.0 159.5,187.0 160.9,187.0 162.4,187.0 163.9,187.0 165.4,187.0 166.8,187.0 168.3,187.0 169.8,187.0 171.2,187.0 172.7,187.0 174.2,187.0 175.7,187.0 177.1,187.0 178.6,187.0 180.1,187.0 181.5,187.0 183.0,187.0 184.5,187.0 186.0,187.0 187.4,187.0 188.9,187.0 190.4,187.0 191.8,187.0 193.3,187.0 194.8,187.0 196.3,187.0 197.7,187.0 199.2,187.0 200.7,187.0 202.1,187.0 203.6,187.0 205.1,187.0 206.6,187.0 208.0,187.0 209.5,187.0 211.0,187.0 212.4,187.0 213.9,187.0 215.4,187.0 216.9,187.0 218.3,187.0 219.8,187.0 221.3,187.0 222.7,187.0 224.2,187.0 225.7,187.0 227.2,187.0 228.6,187.0 230.1,187.0 231.6,187.0 233.0,187.0 234.5,187.0 236.0,187.0 237.5,187.0 238.9,187.0 240.4,187.0 241.9,187.0 243.3,187.0 244.8,187.0 246.3,187.0 247.8,187.0 249.2,187.0 250.7,187.0 252.2,187.0 253.6,187.0 255.1,187.0 256.6,187.0 258.1,187.0 259.5,187.0 261.0,187.0 262.5,187.0 263.9,187.0 265.4,187.0 266.9,187.0 268.4,187.0 269.8,187.0 271.3,187.0 272.8,187.0 274.2,187.0 275.7,187.0 277.2,187.0 278.7,187.0 280.1,187.0 281.6,187.0 283.1,187.0 284.5,187.0 286.0,187.0 287.5,187.0 289.0,187.0 290.4,187.0 291.9,187.0 293.4,187.0 294.8,187.0 296.3,187.0 297.8,187.0 299.3,187.0 300.7,187.0 302.2,187.0 303.7,187.0 305.2,187.0 306.6,187.0 308.1,187.0 309.6,187.0 311.0,187.0 312.5,187.0 314.0,187.0 315.5,187.0 316.9,187.0 318.4,187.0 319.9,187.0 321.3,187.0 322.8,187.0 324.3,187.0 325.8,187.0 327.2,187.0 328.7,187.0 330.2,187.0 331.6,187.0 333.1,187.0 334.6,187.0 336.1,187.0 337.5,187.0 339.0,187.0 340.5,187.0 341.9,187.0 343.4,187.0 344.9,187.0 346.4,187.0 347.8,187.0 349.3,187.0 350.8,187.0 352.2,187.0 353.7,187.0 355.2,187.0 356.7,187.0 358.1,187.0 359.6,187.0 361.1,187.0 362.5,187.0 364.0,187.0 365.5,187.0 367.0,187.0 368.4,187.0 369.9,187.0 371.4,187.0 372.8,187.0 374.3,187.0 375.8,187.0 377.3,187.0 378.7,187.0 380.2,187.0 381.7,187.0 383.1,187.0 384.6,187.0 386.1,187.0 387.6,187.0 389.0,187.0 390.5,187.0 392.0,187.0 393.4,187.0 394.9,187.0 396.4,187.0 397.9,187.0 399.3,187.0 400.8,187.0 402.3,187.0 403.7,187.0 405.2,187.0 406.7,187.0 408.2,187.0 409.6,187.0 411.1,187.0 412.6,187.0 414.0,187.0 415.5,187.0 417.0,187.0 418.5,187.0 419.9,187.0 421.4,187.0 422.9,187.0 424.3,187.0 425.8,187.0 427.3,187.0 428.8,187.0 430.2,187.0 431.7,187.0 433.2,187.0 434.6,187.0 436.1,187.0 437.6,187.0 439.1,187.0 440.5,187.0 442.0,187.0 443.5,187.0 444.9,187.0 446.4,187.0 447.9,187.0 449.4,187.0 450.8,187.0 452.3,187.0 453.8,187.0 455.3,187.0 456.7,187.0 458.2,187.0 459.7,187.0 461.1,187.0 462.6,187.0 464.1,187.0 465.6,187.0 467.0,187.0 468.5,187.0 470.0,187.0 471.4,187.0 472.9,187.0 474.4,187.0 475.9,187.0 477.3,187.0 478.8,187.0 480.3,187.0 481.7,187.0 483.2,187.0 484.7,187.0 486.2,187.0 487.6,187.0 489.1,187.0 490.6,187.0 492.0,187.0 493.5,187.0 495.0,187.0 496.5,187.0 497.9,187.0 499.4,187.0 500.9,187.0 502.3,187.0 503.8,187.0 505.3,187.0 506.8,187.0 508.2,187.0 509.7,187.0 511.2,187.0 512.6,187.0 514.1,187.0 515.6,187.0 517.1,187.0 518.5,187.0 520.0,187.0" fill="none" class="stroke-a" stroke-width="2.5"/>
<line x1="80.0" y1="282.0" x2="520.0" y2="282.0" class="stroke-grid" />
<text x="72.0" y="286.0" class="fig-small" text-anchor="end">0.0</text>
<line x1="80.0" y1="262.6" x2="520.0" y2="262.6" class="stroke-grid" />
<text x="72.0" y="266.6" class="fig-small" text-anchor="end">0.5</text>
<line x1="80.0" y1="243.2" x2="520.0" y2="243.2" class="stroke-grid" />
<text x="72.0" y="247.2" class="fig-small" text-anchor="end">1.0</text>
<line x1="80.0" y1="223.9" x2="520.0" y2="223.9" class="stroke-grid" />
<text x="72.0" y="227.9" class="fig-small" text-anchor="end">1.5</text>
<text x="80.0" y="214.0" class="fig-small" text-anchor="start">noise variance σ²</text>
<line x1="80.0" y1="280.5" x2="520.0" y2="280.5" class="stroke-c dashed" />
<text x="514.0" y="274.5" class="fig-small" text-anchor="end">closed form 0.039</text>
<polyline points="80.0,224.5 81.5,253.4 82.9,265.4 84.4,274.5 85.9,278.3 87.4,279.7 88.8,280.2 90.3,280.4 91.8,280.4 93.2,280.5 94.7,280.5 96.2,280.5 97.7,280.5 99.1,280.5 100.6,280.5 102.1,280.5 103.5,280.5 105.0,280.5 106.5,280.5 108.0,280.5 109.4,280.5 110.9,280.5 112.4,280.5 113.8,280.5 115.3,280.5 116.8,280.5 118.3,280.5 119.7,280.5 121.2,280.5 122.7,280.5 124.1,280.5 125.6,280.5 127.1,280.5 128.6,280.5 130.0,280.5 131.5,280.5 133.0,280.5 134.4,280.5 135.9,280.5 137.4,280.5 138.9,280.5 140.3,280.5 141.8,280.5 143.3,280.5 144.7,280.5 146.2,280.5 147.7,280.5 149.2,280.5 150.6,280.5 152.1,280.5 153.6,280.5 155.1,280.5 156.5,280.5 158.0,280.5 159.5,280.5 160.9,280.5 162.4,280.5 163.9,280.5 165.4,280.5 166.8,280.5 168.3,280.5 169.8,280.5 171.2,280.5 172.7,280.5 174.2,280.5 175.7,280.5 177.1,280.5 178.6,280.5 180.1,280.5 181.5,280.5 183.0,280.5 184.5,280.5 186.0,280.5 187.4,280.5 188.9,280.5 190.4,280.5 191.8,280.5 193.3,280.5 194.8,280.5 196.3,280.5 197.7,280.5 199.2,280.5 200.7,280.5 202.1,280.5 203.6,280.5 205.1,280.5 206.6,280.5 208.0,280.5 209.5,280.5 211.0,280.5 212.4,280.5 213.9,280.5 215.4,280.5 216.9,280.5 218.3,280.5 219.8,280.5 221.3,280.5 222.7,280.5 224.2,280.5 225.7,280.5 227.2,280.5 228.6,280.5 230.1,280.5 231.6,280.5 233.0,280.5 234.5,280.5 236.0,280.5 237.5,280.5 238.9,280.5 240.4,280.5 241.9,280.5 243.3,280.5 244.8,280.5 246.3,280.5 247.8,280.5 249.2,280.5 250.7,280.5 252.2,280.5 253.6,280.5 255.1,280.5 256.6,280.5 258.1,280.5 259.5,280.5 261.0,280.5 262.5,280.5 263.9,280.5 265.4,280.5 266.9,280.5 268.4,280.5 269.8,280.5 271.3,280.5 272.8,280.5 274.2,280.5 275.7,280.5 277.2,280.5 278.7,280.5 280.1,280.5 281.6,280.5 283.1,280.5 284.5,280.5 286.0,280.5 287.5,280.5 289.0,280.5 290.4,280.5 291.9,280.5 293.4,280.5 294.8,280.5 296.3,280.5 297.8,280.5 299.3,280.5 300.7,280.5 302.2,280.5 303.7,280.5 305.2,280.5 306.6,280.5 308.1,280.5 309.6,280.5 311.0,280.5 312.5,280.5 314.0,280.5 315.5,280.5 316.9,280.5 318.4,280.5 319.9,280.5 321.3,280.5 322.8,280.5 324.3,280.5 325.8,280.5 327.2,280.5 328.7,280.5 330.2,280.5 331.6,280.5 333.1,280.5 334.6,280.5 336.1,280.5 337.5,280.5 339.0,280.5 340.5,280.5 341.9,280.5 343.4,280.5 344.9,280.5 346.4,280.5 347.8,280.5 349.3,280.5 350.8,280.5 352.2,280.5 353.7,280.5 355.2,280.5 356.7,280.5 358.1,280.5 359.6,280.5 361.1,280.5 362.5,280.5 364.0,280.5 365.5,280.5 367.0,280.5 368.4,280.5 369.9,280.5 371.4,280.5 372.8,280.5 374.3,280.5 375.8,280.5 377.3,280.5 378.7,280.5 380.2,280.5 381.7,280.5 383.1,280.5 384.6,280.5 386.1,280.5 387.6,280.5 389.0,280.5 390.5,280.5 392.0,280.5 393.4,280.5 394.9,280.5 396.4,280.5 397.9,280.5 399.3,280.5 400.8,280.5 402.3,280.5 403.7,280.5 405.2,280.5 406.7,280.5 408.2,280.5 409.6,280.5 411.1,280.5 412.6,280.5 414.0,280.5 415.5,280.5 417.0,280.5 418.5,280.5 419.9,280.5 421.4,280.5 422.9,280.5 424.3,280.5 425.8,280.5 427.3,280.5 428.8,280.5 430.2,280.5 431.7,280.5 433.2,280.5 434.6,280.5 436.1,280.5 437.6,280.5 439.1,280.5 440.5,280.5 442.0,280.5 443.5,280.5 444.9,280.5 446.4,280.5 447.9,280.5 449.4,280.5 450.8,280.5 452.3,280.5 453.8,280.5 455.3,280.5 456.7,280.5 458.2,280.5 459.7,280.5 461.1,280.5 462.6,280.5 464.1,280.5 465.6,280.5 467.0,280.5 468.5,280.5 470.0,280.5 471.4,280.5 472.9,280.5 474.4,280.5 475.9,280.5 477.3,280.5 478.8,280.5 480.3,280.5 481.7,280.5 483.2,280.5 484.7,280.5 486.2,280.5 487.6,280.5 489.1,280.5 490.6,280.5 492.0,280.5 493.5,280.5 495.0,280.5 496.5,280.5 497.9,280.5 499.4,280.5 500.9,280.5 502.3,280.5 503.8,280.5 505.3,280.5 506.8,280.5 508.2,280.5 509.7,280.5 511.2,280.5 512.6,280.5 514.1,280.5 515.6,280.5 517.1,280.5 518.5,280.5 520.0,280.5" fill="none" class="stroke-c" stroke-width="2.5"/>
<text x="80.0" y="300.0" class="fig-small" text-anchor="middle">1</text>
<text x="225.7" y="300.0" class="fig-small" text-anchor="middle">100</text>
<text x="372.8" y="300.0" class="fig-small" text-anchor="middle">200</text>
<text x="520.0" y="300.0" class="fig-small" text-anchor="middle">300</text>
<text x="300.0" y="319.0" text-anchor="middle">EM round</text>
</svg>
Figure 4. EM on the six-dimensional data from the principal components chapter. The fitted plane lands on the PCA plane within five rounds (the flat bottom is rounding error), and σ² settles on the average of the dropped variances almost at once. The log-likelihood climbs steeply, then creeps up for a few hundred rounds while the columns of W grow to their final lengths (too small a change to see at this scale), and it never falls.
```

There are three ways to find the same plane:

| Route | What it gives | When it earns its place |
|---|---|---|
| SVD | Every direction at once, exactly | The default, when the table fits in memory |
| Power iteration | One direction per run, from products only | Huge or sparse tables |
| EM for probabilistic PCA | The plane, the fuzz level $\sigma^2$, and a likelihood | When you want a model, not only a plane |

> [!KEY]
> PCA can carry fuzz inside it. The model says each point is a hidden coordinate on a sheet, pushed through $W$, plus Gaussian fuzz of variance $\sigma^2$. You cannot see the coordinates, so EM guesses them (E-step) and refits $W$ and $\sigma^2$ from the guesses (M-step), and the log-likelihood rises every round until the plane matches the one that power iteration and the SVD find.

> [!RECAP]
> - Each product stretches each eigenvector part by its eigenvalue, so the largest one takes over. Rescale to length 1 at every step, or the numbers overflow.
> - The Rayleigh quotient `v @ A @ v / (v @ v)` reads the eigenvalue. The direction error shrinks by $\lambda_2/\lambda_1$ per step, and deflation, `A - eigenvalue * np.outer(v, v)`, exposes the next direction.
> - The model is $x = Wz + \text{fuzz}$, with covariance $C = WW^{\top} + \sigma^2 I$.
> - The E-step computes $E_z$ and $E_{zz}$, and the M-step computes $W_{\text{new}}$, then $\sigma^2$ from $W_{\text{new}}$.
> - Record $\ell$ with `slogdet` and `solve`. It never falls, so a drop means a bug. $W$ is known only up to a rotation, so compare planes.

## Build it step by step

`center` and `pca_svd` are written for you, and so is the loop `ppca_em`, which calls your last three functions. Read `ppca_em` after you finish. Build six functions, in this order.

1. **`rayleigh_quotient(A, v)`** takes a symmetric matrix `(d, d)` and a vector `(d,)`, and returns the float `v @ A @ v / (v @ v)`. The usual mistake is to forget the division when `v` is not unit length.
2. **`power_iteration(A, iters)`** returns `(v, eigenvalue)`. Start from `np.ones(d) / np.sqrt(d)`, multiply and rescale `iters` times, and finish with your `rayleigh_quotient`. The usual mistake is to rescale only after the loop, which overflows for large `iters`.
3. **`deflate(A, v, eigenvalue)`** returns `A - eigenvalue * np.outer(v, v)`, shape `(d, d)`, for a unit vector `v`. The usual mistake is `v @ v`, which is a number, in place of `np.outer(v, v)`, which is a matrix.
4. **`ppca_e_step(Xc, W, sigma2)`** takes centred data `(n, d)`, `W` `(d, k)`, and a float, and returns `(Ez, Ezz_sum)` with shapes `(n, k)` and `(k, k)`. Build $M$, invert it once, and use two matrix products. The usual mistake is to leave the $n\sigma^2M^{-1}$ term out of `Ezz_sum`.
5. **`ppca_m_step(Xc, Ez, Ezz_sum)`** returns `(W_new, sigma2_new)` with shapes `(d, k)` and a float. Compute `W_new` first, then the three terms of $\sigma^2$ from `W_new`. Clip `sigma2_new` at `1e-8`. The usual mistake is to use the old $W$ in the cross term.
6. **`ppca_loglik(Xc, W, sigma2)`** returns $\ell$ as a float. Build $C$ and $S$, then use `slogdet` and `solve`. The usual mistake is to drop the factor $n/2$.

## Cost and scaling

One power-iteration step costs about $d^2$ operations for a dense $A$, and about the number of nonzero entries for a sparse one. For a covariance table, skip building it: `Xc.T @ (Xc @ v) / (n - 1)` costs about $2nd$ per step and needs no $d \times d$ storage. Running $t$ steps costs $t$ times that, against about $d^3$ for a full eigendecomposition. Each EM round costs about $n\,d\,k$, because the only matrices that get inverted are $k \times k$.

## Where this breaks

- If the start has no part along the top eigenvector, in exact arithmetic power iteration never finds it. The fixed start `np.ones(d) / np.sqrt(d)` is orthogonal to the top eigenvector for some matrices.
- Deflation inherits any error in $v_1$ and $\lambda_1$. A little of the first direction survives in $A'$, so run the first search to a tight tolerance.
- If $\lambda_1 = \lambda_2$, nothing prefers one direction over the other, so the result depends on the start. If they are close, convergence crawls.
- If the data lie exactly on a sheet, $\sigma^2$ heads to 0 and $C$ becomes singular. The clip at `1e-8` keeps it finite.
- In EM the plane settles in a few rounds, but the lengths of the columns of $W$ can take hundreds more. Judge convergence by the log-likelihood flattening, not by $W$ stopping.

## The exercise

You write `rayleigh_quotient`, `power_iteration`, `deflate`, `ppca_e_step`, `ppca_m_step`, and `ppca_loglik`. The checks confirm the hand numbers: a quotient of 2 at an eigenvector and 1.5 at $(1,1)/\sqrt{2}$, and steps 1 and 2 of the trace to $(0.8944, 0.4472)$ with 1.8, and $(0.9701, 0.2425)$ with $33/17$. They run thousands of steps to confirm there is no overflow, measure the direction error against $\lambda_2/\lambda_1$ and the eigenvalue error against its square, and check that deflation plus power iteration matches the first two SVD components up to sign. For the model they confirm $E_z = (1, -1)$ and $E_{zz} = 3$, then $W_{\text{new}} = 4/3$ and $\sigma^2_{\text{new}} = 2/3$, then log-likelihoods of $-6.37$ and $-5.80$, against a point-by-point posterior and a direct Gaussian density. Finally they run `ppca_em` and confirm that the log-likelihood never decreases, that the fitted plane matches the PCA plane, and that $\sigma^2$ matches the average of the dropped variances.
