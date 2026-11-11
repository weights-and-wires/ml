---
title: "Gaussian processes: kernel ridge with error bars"
slug: 023-gaussian-processes
part: "Part 2 — Classical models"
summary: Treat a curve as one random draw, condition it on noisy points to get a mean and an error bar, and pick the length scale by the evidence.
minutes: 85
gpu: false
objectives:
  - Describe a Gaussian process as a distribution over functions whose covariance is a kernel, and draw functions from it with a Cholesky factor and a little jitter.
  - Condition the prior on noisy data to get the posterior mean and variance, and say why the variance depends on where the points are and not on their targets.
  - Show that the posterior mean is the kernel ridge prediction with lambda equal to the noise variance, and that a linear kernel gives Bayesian linear regression.
  - Compute the log marginal likelihood, read its fit and simplicity terms, and choose a length scale with it.
  - State the n-cubed cost of the exact method and where it stops being practical.
prereqs:
  - 011-likelihood-map-and-bayesian-regression
  - 021-kernels-and-validity
  - 022-kernel-ridge-and-random-features
exercise: 023-gaussian-processes
---

# Gaussian processes: kernel ridge with error bars

You have six noisy measurements of a curve, and the kernel ridge fit from the previous chapter passes through the middle of them. A query arrives at $x = 4.5$, in the gap between two clusters of points, and then another at $x = 8$, far to the right of everything. Kernel ridge gives a number for each, and the two numbers look equally confident. They are not equally trustworthy. The first sits between measurements that disagree, and the second has no measurement within reach.

Two questions follow. How sure is the fit at each input? And how do you pick the bump width without refitting on many folds? A *Gaussian process* answers both with one extra idea: treat the unknown curve as a random draw, and ask what the draw looks like once you have seen the data.

This chapter draws random curves from a kernel, conditions them on data to get a mean and an error bar, checks that the mean is the kernel ridge prediction you already know, and scores a bump width by the probability the model gave to your data.

## What you already know

- A kernel scores how alike two inputs are, and the Gram matrix $K$ holds the score for every pair ([kernels](/lessons/021-kernels-and-validity)).
- Kernel ridge regression predicts $k(x)^{\top}(K + \lambda I)^{-1}y$, where $k(x)$ lists the similarities from $x$ to the training points ([kernel ridge](/lessons/022-kernel-ridge-and-random-features)).
- A Cholesky factor writes a positive definite matrix as $LL^{\top}$, and two triangular solves replace an inverse.
- Under Gaussian noise and a Gaussian prior on the weights, ridge regression is the MAP estimate with $\lambda = \sigma^2/\tau^2$ ([likelihood and the Bayesian view](/lessons/011-likelihood-map-and-bayesian-regression)).
- A bell curve is fixed by a mean and a variance, and a vector of jointly Gaussian numbers is fixed by a mean vector and a covariance matrix.

## A function is a long vector of numbers

Pick 50 input points on a line. A function, looked at only at those points, is a list of 50 heights. A random function is then a random vector with 50 entries. You already know one way to describe a random vector: give it a mean and a covariance matrix.

A Gaussian process does exactly that, and it lets the kernel write the covariance. Take the mean to be zero. Set the covariance between the heights at two inputs $x$ and $x'$ to $k(x, x')$. Two inputs that are similar, which means a kernel value near 1, then have heights that move together. Two inputs that are far apart, with a kernel value near 0, have heights that do not influence each other. This is the whole definition: for any list of inputs, the heights are jointly Gaussian with zero mean and covariance $K$.

The RBF kernel with a length scale $\ell$ is

$$
k(x, x') = \exp\!\left(-\frac{(x - x')^2}{2\ell^2}\right).
$$

Read it aloud: "similarity is 1 at distance zero and falls to $e^{-1/2} \approx 0.61$ at one length scale." The $\gamma$ of the last chapter is $1/(2\ell^2)$. The length scale is easier to think about, because it is a distance in the units of $x$.

For two inputs one length scale apart, the covariance matrix is

$$
K = \begin{bmatrix} 1 & 0.6065 \\ 0.6065 & 1 \end{bmatrix}.
$$

Each height has variance 1, and the two heights have correlation 0.6065. Learn the first height and you already know something about the second. That is how the process encodes smoothness without writing a formula for it.

## Draw functions from the prior

To see what a kernel believes before any data arrives, draw from this distribution. A random vector with covariance $K$ comes from a Cholesky factor. Write $K = LL^{\top}$, draw $z$ from independent standard normals, and set $f = Lz$. The covariance of $f$ is $L\,\mathbb{E}[zz^{\top}]\,L^{\top} = LL^{\top} = K$, because the standard normals have covariance $I$.

Work the two-input case by hand. The factor is

$$
L = \begin{bmatrix} 1 & 0 \\ 0.6065 & 0.7951 \end{bmatrix},
$$

where $0.7951 = \sqrt{1 - 0.6065^2}$. Take $z = (0.5, -1.0)$. Then $f = Lz = (0.5,\ 0.6065 \cdot 0.5 - 0.7951) = (0.5,\ -0.4918)$. The second height keeps 0.6065 of the first and adds fresh randomness for the rest, which is what a correlation of 0.6065 means.

Figure 1 draws three functions at 80 inputs on $[0, 6]$, once with a short length scale and once with a long one, from the same standard normals.

```figure
<svg viewBox="0 0 600 220" role="img" aria-label="Three random functions drawn from a Gaussian process prior with a short length scale on the left and a long one on the right.">
<line x1="46" y1="176" x2="285" y2="176" class="stroke-muted"/><line x1="46" y1="24" x2="46" y2="176" class="stroke-muted"/><line x1="46" y1="100.0" x2="285" y2="100.0" class="stroke-grid"/><text x="46.0" y="191" text-anchor="middle" class="fig-small">0</text><text x="125.7" y="191" text-anchor="middle" class="fig-small">2</text><text x="205.3" y="191" text-anchor="middle" class="fig-small">4</text><text x="285.0" y="191" text-anchor="middle" class="fig-small">6</text><text x="40" y="154.7" text-anchor="end" class="fig-small">-2</text><text x="40" y="104.0" text-anchor="end" class="fig-small">0</text><text x="40" y="53.3" text-anchor="end" class="fig-small">2</text><text x="165.5" y="207" text-anchor="middle" class="fig-small">x</text>
<path d="M46.0,96.8 L49.0,96.4 L52.1,94.4 L55.1,91.8 L58.1,89.7 L61.1,89.6 L64.2,92.3 L67.2,98.2 L70.2,106.5 L73.2,115.4 L76.3,123.4 L79.3,128.8 L82.3,130.8 L85.3,129.5 L88.4,125.8 L91.4,120.5 L94.4,114.5 L97.4,108.3 L100.5,102.0 L103.5,95.9 L106.5,90.1 L109.5,85.2 L112.6,81.8 L115.6,80.4 L118.6,81.3 L121.6,84.1 L124.7,88.5 L127.7,93.6 L130.7,99.1 L133.7,104.6 L136.8,109.7 L139.8,114.3 L142.8,118.4 L145.8,121.8 L148.9,124.4 L151.9,126.3 L154.9,127.7 L157.9,129.1 L161.0,131.0 L164.0,133.8 L167.0,137.4 L170.0,141.2 L173.1,144.3 L176.1,146.1 L179.1,146.0 L182.1,144.3 L185.2,141.5 L188.2,138.4 L191.2,135.8 L194.2,134.1 L197.3,133.2 L200.3,132.8 L203.3,132.4 L206.3,131.7 L209.4,130.6 L212.4,129.4 L215.4,128.5 L218.4,128.2 L221.5,128.6 L224.5,129.0 L227.5,128.9 L230.5,127.3 L233.6,123.8 L236.6,118.3 L239.6,111.4 L242.6,104.0 L245.7,97.2 L248.7,91.6 L251.7,87.5 L254.7,84.9 L257.8,83.4 L260.8,82.7 L263.8,82.8 L266.8,83.5 L269.9,84.5 L272.9,85.4 L275.9,85.7 L278.9,84.8 L282.0,82.6 L285.0,79.6" fill="none" class="stroke-a" stroke-width="2.0"/>
<path d="M46.0,103.3 L49.0,105.8 L52.1,106.9 L55.1,106.6 L58.1,105.5 L61.1,104.2 L64.2,103.2 L67.2,102.8 L70.2,103.1 L73.2,103.8 L76.3,104.9 L79.3,106.1 L82.3,107.1 L85.3,108.1 L88.4,109.0 L91.4,109.8 L94.4,110.4 L97.4,110.5 L100.5,109.6 L103.5,107.3 L106.5,103.5 L109.5,98.6 L112.6,93.4 L115.6,88.7 L118.6,85.5 L121.6,84.2 L124.7,84.8 L127.7,86.7 L130.7,89.1 L133.7,91.3 L136.8,92.6 L139.8,92.8 L142.8,91.7 L145.8,89.7 L148.9,87.1 L151.9,84.5 L154.9,82.4 L157.9,81.5 L161.0,82.2 L164.0,84.4 L167.0,88.1 L170.0,92.6 L173.1,97.1 L176.1,100.9 L179.1,103.3 L182.1,103.9 L185.2,102.8 L188.2,100.0 L191.2,95.9 L194.2,90.6 L197.3,84.7 L200.3,78.8 L203.3,73.6 L206.3,69.9 L209.4,68.3 L212.4,69.0 L215.4,71.7 L218.4,75.8 L221.5,80.2 L224.5,84.1 L227.5,87.0 L230.5,88.8 L233.6,90.1 L236.6,91.6 L239.6,94.0 L242.6,97.8 L245.7,103.0 L248.7,109.4 L251.7,116.4 L254.7,123.3 L257.8,129.7 L260.8,134.9 L263.8,138.6 L266.8,140.3 L269.9,139.9 L272.9,137.2 L275.9,132.5 L278.9,126.1 L282.0,118.8 L285.0,111.4" fill="none" class="stroke-b" stroke-width="2.0"/>
<path d="M46.0,83.8 L49.0,82.3 L52.1,82.5 L55.1,84.1 L58.1,87.0 L61.1,91.1 L64.2,96.2 L67.2,101.9 L70.2,107.9 L73.2,113.5 L76.3,118.2 L79.3,121.7 L82.3,123.6 L85.3,123.7 L88.4,122.0 L91.4,118.4 L94.4,112.8 L97.4,105.4 L100.5,96.2 L103.5,85.9 L106.5,75.1 L109.5,64.9 L112.6,56.1 L115.6,49.5 L118.6,45.3 L121.6,43.3 L124.7,42.9 L127.7,43.4 L130.7,44.0 L133.7,44.3 L136.8,44.4 L139.8,44.4 L142.8,44.8 L145.8,45.9 L148.9,47.9 L151.9,50.6 L154.9,53.8 L157.9,57.3 L161.0,60.5 L164.0,63.3 L167.0,65.3 L170.0,66.5 L173.1,66.8 L176.1,66.5 L179.1,66.1 L182.1,65.9 L185.2,66.6 L188.2,68.3 L191.2,70.8 L194.2,73.9 L197.3,77.1 L200.3,80.3 L203.3,83.5 L206.3,87.1 L209.4,91.5 L212.4,97.0 L215.4,103.5 L218.4,110.4 L221.5,116.8 L224.5,121.9 L227.5,125.0 L230.5,125.9 L233.6,124.7 L236.6,122.1 L239.6,118.7 L242.6,115.2 L245.7,112.2 L248.7,110.1 L251.7,109.1 L254.7,109.2 L257.8,110.1 L260.8,111.3 L263.8,112.0 L266.8,111.9 L269.9,110.9 L272.9,109.4 L275.9,108.2 L278.9,108.2 L282.0,109.8 L285.0,113.2" fill="none" class="stroke-c" stroke-width="2.0"/>
<text x="166" y="14" text-anchor="middle" class="fig-bold">length scale 0.4</text>
<line x1="336" y1="176" x2="575" y2="176" class="stroke-muted"/><line x1="336" y1="24" x2="336" y2="176" class="stroke-muted"/><line x1="336" y1="100.0" x2="575" y2="100.0" class="stroke-grid"/><text x="336.0" y="191" text-anchor="middle" class="fig-small">0</text><text x="415.7" y="191" text-anchor="middle" class="fig-small">2</text><text x="495.3" y="191" text-anchor="middle" class="fig-small">4</text><text x="575.0" y="191" text-anchor="middle" class="fig-small">6</text><text x="330" y="154.7" text-anchor="end" class="fig-small">-2</text><text x="330" y="104.0" text-anchor="end" class="fig-small">0</text><text x="330" y="53.3" text-anchor="end" class="fig-small">2</text><text x="455.5" y="207" text-anchor="middle" class="fig-small">x</text>
<path d="M336.0,96.8 L339.0,96.7 L342.1,96.4 L345.1,96.1 L348.1,95.7 L351.1,95.3 L354.2,94.9 L357.2,94.4 L360.2,94.0 L363.2,93.6 L366.3,93.3 L369.3,93.1 L372.3,93.0 L375.3,93.0 L378.4,93.0 L381.4,93.2 L384.4,93.5 L387.4,94.0 L390.5,94.5 L393.5,95.0 L396.5,95.7 L399.5,96.4 L402.6,97.2 L405.6,98.0 L408.6,98.8 L411.6,99.6 L414.7,100.4 L417.7,101.1 L420.7,101.9 L423.7,102.5 L426.8,103.1 L429.8,103.7 L432.8,104.1 L435.8,104.5 L438.9,104.9 L441.9,105.1 L444.9,105.4 L447.9,105.6 L451.0,105.8 L454.0,106.0 L457.0,106.1 L460.0,106.3 L463.1,106.6 L466.1,106.9 L469.1,107.3 L472.1,107.7 L475.2,108.2 L478.2,108.9 L481.2,109.6 L484.2,110.5 L487.3,111.5 L490.3,112.6 L493.3,113.9 L496.3,115.3 L499.4,116.7 L502.4,118.3 L505.4,120.0 L508.4,121.7 L511.5,123.5 L514.5,125.4 L517.5,127.3 L520.5,129.2 L523.6,131.1 L526.6,133.0 L529.6,134.8 L532.6,136.6 L535.7,138.3 L538.7,139.9 L541.7,141.4 L544.7,142.7 L547.8,143.9 L550.8,144.9 L553.8,145.8 L556.8,146.4 L559.9,146.9 L562.9,147.2 L565.9,147.2 L568.9,147.0 L572.0,146.6 L575.0,146.0" fill="none" class="stroke-a" stroke-width="2.0"/>
<path d="M336.0,103.3 L339.0,104.0 L342.1,104.6 L345.1,105.1 L348.1,105.5 L351.1,105.8 L354.2,106.0 L357.2,106.2 L360.2,106.2 L363.2,106.2 L366.3,106.2 L369.3,106.0 L372.3,105.9 L375.3,105.7 L378.4,105.5 L381.4,105.3 L384.4,105.1 L387.4,104.9 L390.5,104.7 L393.5,104.5 L396.5,104.3 L399.5,104.1 L402.6,103.9 L405.6,103.7 L408.6,103.6 L411.6,103.4 L414.7,103.2 L417.7,103.1 L420.7,102.9 L423.7,102.7 L426.8,102.5 L429.8,102.2 L432.8,102.0 L435.8,101.6 L438.9,101.3 L441.9,100.9 L444.9,100.5 L447.9,100.0 L451.0,99.5 L454.0,98.9 L457.0,98.3 L460.0,97.7 L463.1,97.0 L466.1,96.3 L469.1,95.6 L472.1,94.9 L475.2,94.1 L478.2,93.4 L481.2,92.6 L484.2,91.9 L487.3,91.1 L490.3,90.4 L493.3,89.7 L496.3,88.9 L499.4,88.3 L502.4,87.6 L505.4,86.9 L508.4,86.3 L511.5,85.7 L514.5,85.1 L517.5,84.5 L520.5,84.0 L523.6,83.4 L526.6,82.9 L529.6,82.4 L532.6,81.9 L535.7,81.5 L538.7,81.1 L541.7,80.7 L544.7,80.4 L547.8,80.1 L550.8,79.9 L553.8,79.8 L556.8,79.7 L559.9,79.8 L562.9,79.9 L565.9,80.1 L568.9,80.5 L572.0,80.9 L575.0,81.5" fill="none" class="stroke-b" stroke-width="2.0"/>
<path d="M336.0,83.8 L339.0,83.3 L342.1,83.0 L345.1,82.8 L348.1,82.7 L351.1,82.7 L354.2,82.9 L357.2,83.2 L360.2,83.6 L363.2,84.2 L366.3,85.0 L369.3,85.8 L372.3,86.8 L375.3,87.9 L378.4,89.1 L381.4,90.4 L384.4,91.8 L387.4,93.2 L390.5,94.6 L393.5,96.0 L396.5,97.4 L399.5,98.7 L402.6,99.9 L405.6,101.0 L408.6,102.0 L411.6,102.8 L414.7,103.4 L417.7,103.8 L420.7,103.9 L423.7,103.9 L426.8,103.5 L429.8,102.9 L432.8,102.0 L435.8,100.9 L438.9,99.5 L441.9,97.8 L444.9,95.9 L447.9,93.7 L451.0,91.3 L454.0,88.7 L457.0,86.0 L460.0,83.1 L463.1,80.1 L466.1,77.0 L469.1,73.8 L472.1,70.6 L475.2,67.4 L478.2,64.2 L481.2,61.1 L484.2,58.0 L487.3,55.1 L490.3,52.3 L493.3,49.7 L496.3,47.3 L499.4,45.0 L502.4,43.0 L505.4,41.2 L508.4,39.6 L511.5,38.3 L514.5,37.3 L517.5,36.5 L520.5,36.0 L523.6,35.8 L526.6,35.9 L529.6,36.2 L532.6,36.8 L535.7,37.7 L538.7,38.8 L541.7,40.2 L544.7,41.8 L547.8,43.6 L550.8,45.6 L553.8,47.8 L556.8,50.2 L559.9,52.7 L562.9,55.3 L565.9,58.1 L568.9,60.9 L572.0,63.9 L575.0,66.8" fill="none" class="stroke-c" stroke-width="2.0"/>
<text x="456" y="14" text-anchor="middle" class="fig-bold">length scale 1.5</text>
</svg>
Figure 1. Three random functions from the prior, with length scale 0.4 on the left and 1.5 on the right. The same standard normals drive both panels. Notice that the short length scale gives curves that cross zero three or four times, and the long one gives curves that cross once or twice: the length scale sets how fast a function is allowed to change.
```

The curves are what the kernel considers plausible, and choosing the kernel is choosing that set of shapes.

A fine grid has a practical problem. Neighbouring inputs are almost perfectly correlated, so the Gram matrix is almost singular. At 50 inputs on $[0, 6]$ with $\ell = 0.707$, the smallest eigenvalue is about $-2 \times 10^{-15}$, a rounding error on a value that is mathematically positive. `np.linalg.cholesky` still raises an error. The remedy is *jitter*: add a tiny multiple of the identity, such as $10^{-8}I$, before factoring. It changes the covariance by far less than you can see.

```python
L = np.linalg.cholesky(kernel(X, X) + 1e-8 * np.eye(m))
z = rng.normal(size=(n, m))        # one row of standard normals per function
f = z @ L.T                        # row i is L @ z[i]
```

> [!WARNING]
> With one function per row, the product is `z @ L.T`, not `z @ L`. The second form has covariance $L^{\top}L$, which is not $K$ unless $L$ is symmetric. It still looks like smooth random output, so the mistake passes a glance and fails a covariance check.

> [!KEY]
> A Gaussian process is a distribution over functions. Any finite list of inputs gets a Gaussian vector of heights whose covariance is the kernel matrix. You draw from it with a Cholesky factor, plus jitter when the matrix is nearly singular.

## Condition on data: the posterior

Give the process some noisy measurements. Assume each target is the function value plus independent noise of variance $\sigma^2$, so $y_i = f(x_i) + \varepsilon_i$. The target vector $y$ and the function values $f_*$ at some query inputs are then jointly Gaussian. Their covariance blocks come from the kernel, with $\sigma^2 I$ added to the block for $y$ because of the noise.

A fact about Gaussians does the rest: if you learn part of a jointly Gaussian vector, the rest is still Gaussian, with a shifted mean and a smaller covariance. For one query $x_*$, write $k_*$ for the vector of similarities from $x_*$ to the training points. The result is

$$
\begin{aligned}
\mu(x_*) &= k_*^{\top}(K + \sigma^2 I)^{-1}\, y, \\
v(x_*) &= k(x_*, x_*) - k_*^{\top}(K + \sigma^2 I)^{-1} k_* .
\end{aligned}
$$

Read the first line aloud: "weight the targets by how similar the query is to each training point, after correcting for how similar the training points are to each other." Read the second: "start from the prior variance, and subtract what the data told you." The subtracted term is never negative, so the data can only reduce the variance. The variance is for the function $f$ at the query. A fresh noisy measurement there has variance $v(x_*) + \sigma^2$.

Work the two-point example. The inputs are $0$ and $1$, the kernel has $\ell = 1$, the targets are $y = (1, 2)$, and the noise is $\sigma^2 = 1$. Then $K + I = \begin{bmatrix} 2 & 0.6065 \\ 0.6065 & 2 \end{bmatrix}$.

Query at the training input $x = 0$, where $k_* = (1,\ 0.6065)$. Solving $(K + I)w = k_*$ gives $w = (0.4494,\ 0.1670)$, so $k_*^{\top}w = 0.5506$ and the variance is $1 - 0.5506 = 0.4494$. The mean is $0.7833$, a little below the target 1, because the large noise lets the fit shrink toward zero.

Query at the midpoint $x = 0.5$, where $k_* = (0.8825,\ 0.8825)$. The mean is $1.0157$ and the variance is $0.4024$. 

The last chapter's $1.449$ at a training point was for a noisy target: the $0.4494$ here plus the noise variance 1.

Figure 2 draws the posterior for the six measurements of the last chapter's Figure 1, with $\sigma^2 = 0.1$ and $\gamma = 1$, which is $\ell = 0.707$. The orange line is the mean, and the band is two standard deviations of the function.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Six training points, the Gaussian process posterior mean, and a band of two standard deviations that pinches near the data and widens away from it.">
<line x1="50" y1="220" x2="570" y2="220" class="stroke-muted"/><line x1="50" y1="20" x2="50" y2="220" class="stroke-muted"/><line x1="50" y1="120.0" x2="570" y2="120.0" class="stroke-grid"/><text x="107.8" y="235" text-anchor="middle" class="fig-small">0</text><text x="223.3" y="235" text-anchor="middle" class="fig-small">2</text><text x="338.9" y="235" text-anchor="middle" class="fig-small">4</text><text x="454.4" y="235" text-anchor="middle" class="fig-small">6</text><text x="570.0" y="235" text-anchor="middle" class="fig-small">8</text><text x="44" y="200.9" text-anchor="end" class="fig-small">-2</text><text x="44" y="162.5" text-anchor="end" class="fig-small">-1</text><text x="44" y="124.0" text-anchor="end" class="fig-small">0</text><text x="44" y="85.5" text-anchor="end" class="fig-small">1</text><text x="44" y="47.1" text-anchor="end" class="fig-small">2</text><text x="310.0" y="251" text-anchor="middle" class="fig-small">x</text>
<polygon points="50.0,42.3 52.9,42.2 55.8,42.2 58.7,42.1 61.6,42.1 64.4,42.2 67.3,42.3 70.2,42.5 73.1,42.7 76.0,43.1 78.9,43.6 81.8,44.2 84.7,45.0 87.6,46.0 90.4,47.1 93.3,48.5 96.2,50.1 99.1,51.9 102.0,54.0 104.9,56.3 107.8,58.8 110.7,61.6 113.6,64.5 116.4,67.6 119.3,70.8 122.2,73.9 125.1,77.0 128.0,79.8 130.9,82.0 133.8,83.6 136.7,84.1 139.6,83.6 142.4,82.0 145.3,79.6 148.2,76.5 151.1,73.2 154.0,69.6 156.9,66.0 159.8,62.4 162.7,58.9 165.6,55.6 168.4,52.6 171.3,49.7 174.2,47.1 177.1,44.8 180.0,42.8 182.9,41.1 185.8,39.7 188.7,38.7 191.6,38.0 194.4,37.6 197.3,37.5 200.2,37.9 203.1,38.5 206.0,39.5 208.9,40.9 211.8,42.6 214.7,44.6 217.6,47.0 220.4,49.5 223.3,52.4 226.2,55.3 229.1,58.4 232.0,61.5 234.9,64.5 237.8,67.3 240.7,69.9 243.6,72.1 246.4,74.0 249.3,75.6 252.2,77.0 255.1,78.4 258.0,79.9 260.9,81.5 263.8,83.4 266.7,85.5 269.6,87.8 272.4,90.4 275.3,93.2 278.2,96.1 281.1,99.1 284.0,102.0 286.9,104.9 289.8,107.7 292.7,110.2 295.6,112.5 298.4,114.5 301.3,116.3 304.2,117.8 307.1,119.0 310.0,120.0 312.9,120.7 315.8,121.2 318.7,121.4 321.6,121.3 324.4,120.7 327.3,119.7 330.2,118.2 333.1,116.3 336.0,113.8 338.9,110.9 341.8,107.7 344.7,104.3 347.6,100.6 350.4,96.8 353.3,93.1 356.2,89.4 359.1,85.8 362.0,82.5 364.9,79.3 367.8,76.5 370.7,73.9 373.6,71.7 376.4,69.9 379.3,68.4 382.2,67.3 385.1,66.5 388.0,66.2 390.9,66.1 393.8,66.5 396.7,67.1 399.6,68.1 402.4,69.3 405.3,70.9 408.2,72.6 411.1,74.6 414.0,76.7 416.9,79.1 419.8,81.5 422.7,84.0 425.6,86.7 428.4,89.3 431.3,92.0 434.2,94.6 437.1,97.2 440.0,99.6 442.9,101.9 445.8,103.8 448.7,105.3 451.6,106.1 454.4,106.0 457.3,104.8 460.2,102.5 463.1,99.3 466.0,95.5 468.9,91.4 471.8,87.0 474.7,82.6 477.6,78.2 480.4,74.0 483.3,69.9 486.2,66.0 489.1,62.4 492.0,59.1 494.9,56.0 497.8,53.3 500.7,50.8 503.6,48.6 506.4,46.7 509.3,45.1 512.2,43.8 515.1,42.7 518.0,41.8 520.9,41.1 523.8,40.6 526.7,40.2 529.6,40.0 532.4,39.9 535.3,39.9 538.2,40.0 541.1,40.1 544.0,40.3 546.9,40.5 549.8,40.7 552.7,40.9 555.6,41.2 558.4,41.4 561.3,41.6 564.2,41.8 567.1,42.0 570.0,42.1 570.0,195.9 567.1,195.6 564.2,195.4 561.3,195.1 558.4,194.8 555.6,194.4 552.7,194.0 549.8,193.5 546.9,193.0 544.0,192.3 541.1,191.6 538.2,190.8 535.3,189.9 532.4,188.9 529.6,187.8 526.7,186.6 523.8,185.2 520.9,183.8 518.0,182.2 515.1,180.4 512.2,178.6 509.3,176.6 506.4,174.5 503.6,172.2 500.7,169.9 497.8,167.5 494.9,165.0 492.0,162.4 489.1,159.8 486.2,157.2 483.3,154.5 480.4,151.9 477.6,149.3 474.7,146.9 471.8,144.5 468.9,142.4 466.0,140.6 463.1,139.3 460.2,138.6 457.3,138.9 454.4,140.4 451.6,142.8 448.7,146.1 445.8,150.0 442.9,154.1 440.0,158.4 437.1,162.8 434.2,167.0 431.3,171.1 428.4,175.0 425.6,178.6 422.7,182.1 419.8,185.2 416.9,188.0 414.0,190.5 411.1,192.7 408.2,194.6 405.3,196.1 402.4,197.4 399.6,198.3 396.7,198.9 393.8,199.3 390.9,199.3 388.0,199.1 385.1,198.6 382.2,197.9 379.3,196.9 376.4,195.7 373.6,194.2 370.7,192.6 367.8,190.7 364.9,188.7 362.0,186.6 359.1,184.3 356.2,182.0 353.3,179.6 350.4,177.2 347.6,174.8 344.7,172.6 341.8,170.5 338.9,168.5 336.0,166.9 333.1,165.4 330.2,164.3 327.3,163.5 324.4,162.9 321.6,162.4 318.7,162.1 315.8,161.7 312.9,161.2 310.0,160.6 307.1,159.9 304.2,159.0 301.3,157.9 298.4,156.7 295.6,155.4 292.7,154.0 289.8,152.5 286.9,150.9 284.0,149.1 281.1,147.3 278.2,145.3 275.3,143.2 272.4,140.8 269.6,138.3 266.7,135.7 263.8,133.0 260.9,130.1 258.0,127.4 255.1,124.7 252.2,122.3 249.3,120.4 246.4,119.0 243.6,118.3 240.7,118.4 237.8,119.3 234.9,120.8 232.0,123.0 229.1,125.7 226.2,128.8 223.3,132.1 220.4,135.6 217.6,139.1 214.7,142.6 211.8,146.1 208.9,149.3 206.0,152.4 203.1,155.1 200.2,157.6 197.3,159.7 194.4,161.4 191.6,162.7 188.7,163.6 185.8,164.0 182.9,164.0 180.0,163.6 177.1,162.7 174.2,161.4 171.3,159.7 168.4,157.6 165.6,155.1 162.7,152.4 159.8,149.4 156.9,146.2 154.0,143.0 151.1,139.8 148.2,136.7 145.3,134.0 142.4,131.9 139.6,130.7 136.7,130.6 133.8,131.6 130.9,133.7 128.0,136.5 125.1,139.9 122.2,143.7 119.3,147.6 116.4,151.6 113.6,155.6 110.7,159.4 107.8,163.1 104.9,166.6 102.0,169.9 99.1,173.0 96.2,175.9 93.3,178.5 90.4,180.8 87.6,183.0 84.7,184.9 81.8,186.6 78.9,188.1 76.0,189.4 73.1,190.5 70.2,191.5 67.3,192.4 64.4,193.1 61.6,193.7 58.7,194.2 55.8,194.7 52.9,195.1 50.0,195.4" class="fill-soft-b"/>
<path d="M50.0,118.8 L52.9,118.6 L55.8,118.4 L58.7,118.2 L61.6,117.9 L64.4,117.6 L67.3,117.3 L70.2,117.0 L73.1,116.6 L76.0,116.2 L78.9,115.8 L81.8,115.4 L84.7,114.9 L87.6,114.5 L90.4,114.0 L93.3,113.5 L96.2,113.0 L99.1,112.5 L102.0,112.0 L104.9,111.5 L107.8,111.0 L110.7,110.5 L113.6,110.0 L116.4,109.6 L119.3,109.2 L122.2,108.8 L125.1,108.5 L128.0,108.1 L130.9,107.8 L133.8,107.6 L136.7,107.4 L139.6,107.1 L142.4,107.0 L145.3,106.8 L148.2,106.6 L151.1,106.5 L154.0,106.3 L156.9,106.1 L159.8,105.9 L162.7,105.7 L165.6,105.4 L168.4,105.1 L171.3,104.7 L174.2,104.3 L177.1,103.8 L180.0,103.2 L182.9,102.6 L185.8,101.9 L188.7,101.1 L191.6,100.3 L194.4,99.5 L197.3,98.6 L200.2,97.7 L203.1,96.8 L206.0,96.0 L208.9,95.1 L211.8,94.3 L214.7,93.6 L217.6,93.0 L220.4,92.6 L223.3,92.2 L226.2,92.0 L229.1,92.1 L232.0,92.3 L234.9,92.7 L237.8,93.3 L240.7,94.1 L243.6,95.2 L246.4,96.5 L249.3,98.0 L252.2,99.7 L255.1,101.6 L258.0,103.6 L260.9,105.8 L263.8,108.2 L266.7,110.6 L269.6,113.1 L272.4,115.6 L275.3,118.2 L278.2,120.7 L281.1,123.2 L284.0,125.6 L286.9,127.9 L289.8,130.1 L292.7,132.1 L295.6,133.9 L298.4,135.6 L301.3,137.1 L304.2,138.4 L307.1,139.4 L310.0,140.3 L312.9,141.0 L315.8,141.5 L318.7,141.7 L321.6,141.8 L324.4,141.8 L327.3,141.6 L330.2,141.3 L333.1,140.9 L336.0,140.3 L338.9,139.7 L341.8,139.1 L344.7,138.4 L347.6,137.7 L350.4,137.0 L353.3,136.3 L356.2,135.7 L359.1,135.1 L362.0,134.5 L364.9,134.0 L367.8,133.6 L370.7,133.2 L373.6,133.0 L376.4,132.8 L379.3,132.6 L382.2,132.6 L385.1,132.6 L388.0,132.6 L390.9,132.7 L393.8,132.9 L396.7,133.0 L399.6,133.2 L402.4,133.4 L405.3,133.5 L408.2,133.6 L411.1,133.6 L414.0,133.6 L416.9,133.5 L419.8,133.3 L422.7,133.0 L425.6,132.7 L428.4,132.2 L431.3,131.5 L434.2,130.8 L437.1,130.0 L440.0,129.0 L442.9,128.0 L445.8,126.9 L448.7,125.7 L451.6,124.5 L454.4,123.2 L457.3,121.9 L460.2,120.6 L463.1,119.3 L466.0,118.1 L468.9,116.9 L471.8,115.8 L474.7,114.7 L477.6,113.8 L480.4,112.9 L483.3,112.2 L486.2,111.6 L489.1,111.1 L492.0,110.8 L494.9,110.5 L497.8,110.4 L500.7,110.4 L503.6,110.4 L506.4,110.6 L509.3,110.8 L512.2,111.2 L515.1,111.5 L518.0,112.0 L520.9,112.4 L523.8,112.9 L526.7,113.4 L529.6,113.9 L532.4,114.4 L535.3,114.9 L538.2,115.4 L541.1,115.9 L544.0,116.3 L546.9,116.7 L549.8,117.1 L552.7,117.5 L555.6,117.8 L558.4,118.1 L561.3,118.4 L564.2,118.6 L567.1,118.8 L570.0,119.0" fill="none" class="stroke-b" stroke-width="2.2"/>
<circle cx="137.2" cy="106.2" r="4.5" class="fill-a"/>
<circle cx="244.7" cy="91.9" r="4.5" class="fill-a"/>
<circle cx="293.2" cy="135.0" r="4.5" class="fill-a"/>
<circle cx="328.5" cy="142.3" r="4.5" class="fill-a"/>
<circle cx="450.4" cy="131.5" r="4.5" class="fill-a"/>
<circle cx="462.0" cy="113.5" r="4.5" class="fill-a"/>
</svg>
Figure 2. The posterior mean and a band of two standard deviations for the six points. The band is narrowest at the clusters, where the standard deviation is 0.25 to 0.30, and widens to a variance of 0.75 in the gap near x = 4.9. At x = 8 the variance is 0.999, so the band is back at the prior's width. Notice that the band never closes to zero at a data point, because the noise keeps the data from pinning the function down.
```

Check the numbers against the picture. At the measured input $x = 0.51$ the variance is $0.091$, under the noise level 0.1. In the gap, at $x = 4.5$, it is $0.552$. At $x = 8$ it is $0.999$, almost the prior variance of 1, and the mean has headed back toward zero, the prior mean.

One feature of the formulas is easy to miss. The variance $v(x_*)$ contains no $y$. It depends only on where the training inputs sit, and on the kernel and noise, so you could compute the error bars before collecting a single target. The targets shape the mean, and the positions shape the uncertainty.

At a training point with other training points far away, $K$ is close to the identity and the variance reduces to $1 - 1/(1 + \sigma^2) = \sigma^2/(1 + \sigma^2)$, which is $0.0909$ for $\sigma^2 = 0.1$. The variance at a training point is therefore at most the noise level. Overlapping points pool their information and can go lower, as the $0.062$ at $x = 6.13$ shows.

The code mirrors the formulas. Solve against one Cholesky factor, and reuse it for the variance:

```python
L = np.linalg.cholesky(K + noise * np.eye(n))
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
Ks = kernel(Xs, X)                        # (m, n): queries on the rows
mean = Ks @ alpha
V = np.linalg.solve(L, Ks.T)              # (n, m), so (V**2).sum(0) is k*^T A^-1 k*
var = np.diag(kernel(Xs, Xs)) - (V**2).sum(axis=0)
```

The column sums of $V^2$ equal $k_*^{\top}(K + \sigma^2 I)^{-1}k_*$, because $(K + \sigma^2 I)^{-1} = L^{-\top}L^{-1}$.

> [!WARNING]
> The `var` above is the variance of the function. A fresh noisy target at the same input has variance `var + noise`. Mixing the two gives an error band that is too narrow, or a mismatch with another source of exactly the noise variance.

> [!KEY]
> Conditioning the prior on noisy data gives a Gaussian at every query. Its mean weights the targets by similarity. Its variance is the prior variance minus what the data explains, so it is large away from the data and depends on the inputs and not the targets.

## The mean is kernel ridge, and a linear kernel is Bayesian regression

Look again at the mean: $\mu(x_*) = k_*^{\top}(K + \sigma^2 I)^{-1}y$. The kernel ridge prediction from the last chapter is $k_*^{\top}(K + \lambda I)^{-1}y$. They are the same expression when $\lambda = \sigma^2$. So the ridge fee has a meaning: it is the noise variance, in the units of the kernel. A noisy measurement is one the model must not chase, and a larger $\sigma^2$ tells it so. On the two-point example, the mean at $x = 0.5$ is $1.0157$ from either route.

A Gaussian process therefore adds nothing to the point predictions of kernel ridge. It adds the variance, and a principled way to choose the kernel's parameters.

The link goes one step further, back to the Bayesian regression of the likelihood chapter. Take the linear kernel $k(x, x') = \phi(x)^{\top}\phi(x')$ for a feature vector $\phi$. A process with this kernel is a linear model $f(x) = w^{\top}\phi(x)$ with $w \sim N(0, I)$, because the covariance of two such heights is $\phi(x)^{\top}\mathbb{E}[ww^{\top}]\phi(x') = \phi(x)^{\top}\phi(x')$. Condition it on data and you get the posterior of Bayesian linear regression.

Check it on the two houses of that chapter. The features are $(1, \text{size})$, so $k(s, s') = 1 + s s'$, and the sizes are $0$ and $2$ with prices $0$ and $2$. Take $\sigma^2 = 1$. Then $K = \begin{bmatrix} 1 & 1 \\ 1 & 5 \end{bmatrix}$, $K + I = \begin{bmatrix} 2 & 1 \\ 1 & 6 \end{bmatrix}$, and $(K + I)^{-1}y = (-2/11,\ 4/11)$. For a house of size 6, $k_* = (1,\ 13)$. The mean is $-2/11 + 52/11 = 50/11 \approx 4.55$, and the variance of the function is $37 - 318/11 = 89/11 \approx 8.09$. Add the noise variance 1 to get $100/11 \approx 9.09$. The likelihood chapter found the same $50/11$ and $100/11$ from the weights' posterior. The two views agree, and they differ only in which matrix they solve: a $d \times d$ system in weight space, or an $n \times n$ system in function space.

> [!TIP]
> Use the weight-space form with few features and many points, and the function-space form when the kernel has no finite feature list, as the RBF kernel does. Ridge regression on $D$ random features from the last chapter is an approximate Gaussian process with a $D \times D$ solve.

## Choose a length scale by the evidence

The length scale $\ell$ and the noise $\sigma^2$ shape every answer, and you have to choose them. Cross-validation works but refits many times. A Gaussian process offers a different score, the probability the model assigns to the data you saw.

Under the model, $y$ is a Gaussian vector with mean zero and covariance $K + \sigma^2 I$. Its log density is the *log marginal likelihood*, also called the evidence, because it averages over every function in the prior and not over one fitted function:

$$
\log p(y) = -\tfrac12\, y^{\top}(K + \sigma^2 I)^{-1} y \;-\; \tfrac12 \log\det(K + \sigma^2 I) \;-\; \tfrac{n}{2}\log 2\pi .
$$

Read it aloud: "a data-fit term, a simplicity term, and a constant." The first term is large and negative when the data are far from where the prior puts its mass. The second term is the log of the volume of the prior's covariance. A prior that allows many shapes spreads its probability thinly, so its volume is large and this term is very negative. A prior that allows few shapes has a small volume, and this term is high.

Work the two-point example with $\ell = 1$ and $\sigma^2 = 1$. The matrix $A = K + I$ has determinant $3.632$, so $y^{\top}A^{-1}y = 2.0852$ and $\log\det A = 1.2898$. The three terms are $-1.0426$, $-0.6449$, and $-0.5\cdot 2\cdot\log 2\pi = -1.8379$, and they sum to $-3.5254$.

The two terms pull in opposite directions, and that tension picks the length scale. On the six measurements, with $\sigma^2 = 0.1$, Figure 3 plots them against $\ell$.

```figure
<svg viewBox="0 0 600 236" role="img" aria-label="Left: the data-fit term falls and the simplicity term rises as the length scale grows. Right: their sum, the log marginal likelihood, peaks near a length scale of 1.5.">
<line x1="50" y1="190" x2="270" y2="190" class="stroke-muted"/><line x1="50" y1="24" x2="50" y2="190" class="stroke-muted"/><line x1="50" y1="93.5" x2="270" y2="93.5" class="stroke-grid"/><text x="44" y="174.7" text-anchor="end" class="fig-small">-4</text><text x="44" y="136.1" text-anchor="end" class="fig-small">-2</text><text x="44" y="97.5" text-anchor="end" class="fig-small">0</text><text x="44" y="58.9" text-anchor="end" class="fig-small">2</text>
<text x="50.0" y="205" text-anchor="middle" class="fig-small">0.1</text>
<text x="115.5" y="205" text-anchor="middle" class="fig-small">0.3</text>
<text x="187.3" y="205" text-anchor="middle" class="fig-small">1</text>
<text x="252.8" y="205" text-anchor="middle" class="fig-small">3</text>
<text x="160.0" y="221" text-anchor="middle" class="fig-small">length scale</text>
<line x1="345" y1="190" x2="565" y2="190" class="stroke-muted"/><line x1="345" y1="24" x2="345" y2="190" class="stroke-muted"/><text x="339" y="194.0" text-anchor="end" class="fig-small">-7</text><text x="339" y="147.9" text-anchor="end" class="fig-small">-6.5</text><text x="339" y="101.8" text-anchor="end" class="fig-small">-6</text><text x="339" y="55.7" text-anchor="end" class="fig-small">-5.5</text>
<text x="345.0" y="205" text-anchor="middle" class="fig-small">0.1</text>
<text x="410.5" y="205" text-anchor="middle" class="fig-small">0.3</text>
<text x="482.3" y="205" text-anchor="middle" class="fig-small">1</text>
<text x="547.8" y="205" text-anchor="middle" class="fig-small">3</text>
<text x="455.0" y="221" text-anchor="middle" class="fig-small">length scale</text>
<path d="M50.0,104.8 L51.8,104.8 L53.7,104.8 L55.5,104.8 L57.4,104.8 L59.2,104.9 L61.1,104.9 L62.9,104.9 L64.8,105.0 L66.6,105.0 L68.5,105.0 L70.3,105.1 L72.2,105.1 L74.0,105.2 L75.9,105.2 L77.7,105.3 L79.6,105.3 L81.4,105.4 L83.3,105.5 L85.1,105.5 L87.0,105.6 L88.8,105.7 L90.7,105.7 L92.5,105.8 L94.4,105.9 L96.2,106.0 L98.1,106.0 L99.9,106.1 L101.8,106.2 L103.6,106.3 L105.5,106.4 L107.3,106.4 L109.2,106.5 L111.0,106.6 L112.9,106.7 L114.7,106.8 L116.6,106.9 L118.4,107.0 L120.3,107.1 L122.1,107.3 L123.9,107.4 L125.8,107.5 L127.6,107.7 L129.5,107.8 L131.3,107.9 L133.2,108.1 L135.0,108.3 L136.9,108.4 L138.7,108.6 L140.6,108.8 L142.4,109.0 L144.3,109.2 L146.1,109.5 L148.0,109.7 L149.8,109.9 L151.7,110.2 L153.5,110.5 L155.4,110.7 L157.2,111.0 L159.1,111.3 L160.9,111.7 L162.8,112.0 L164.6,112.3 L166.5,112.7 L168.3,113.1 L170.2,113.4 L172.0,113.8 L173.9,114.3 L175.7,114.7 L177.6,115.1 L179.4,115.6 L181.3,116.0 L183.1,116.5 L185.0,117.0 L186.8,117.5 L188.7,118.1 L190.5,118.6 L192.4,119.2 L194.2,119.8 L196.1,120.4 L197.9,121.0 L199.7,121.7 L201.6,122.5 L203.4,123.2 L205.3,124.1 L207.1,125.0 L209.0,126.0 L210.8,127.1 L212.7,128.3 L214.5,129.6 L216.4,131.1 L218.2,132.6 L220.1,134.3 L221.9,136.2 L223.8,138.1 L225.6,140.2 L227.5,142.4 L229.3,144.6 L231.2,147.0 L233.0,149.3 L234.9,151.7 L236.7,154.1 L238.6,156.5 L240.4,158.8 L242.3,161.0 L244.1,163.1 L246.0,165.2 L247.8,167.1 L249.7,169.0 L251.5,170.7 L253.4,172.4 L255.2,173.9 L257.1,175.4 L258.9,176.8 L260.8,178.1 L262.6,179.3 L264.5,180.5 L266.3,181.6 L268.2,182.7 L270.0,183.7" fill="none" class="stroke-a" stroke-width="2.2"/>
<path d="M50.0,98.9 L51.8,98.8 L53.7,98.8 L55.5,98.7 L57.4,98.6 L59.2,98.6 L61.1,98.5 L62.9,98.4 L64.8,98.3 L66.6,98.2 L68.5,98.0 L70.3,97.9 L72.2,97.7 L74.0,97.6 L75.9,97.4 L77.7,97.2 L79.6,97.0 L81.4,96.8 L83.3,96.6 L85.1,96.3 L87.0,96.1 L88.8,95.9 L90.7,95.6 L92.5,95.3 L94.4,95.1 L96.2,94.8 L98.1,94.5 L99.9,94.2 L101.8,93.9 L103.6,93.6 L105.5,93.3 L107.3,93.0 L109.2,92.7 L111.0,92.4 L112.9,92.1 L114.7,91.7 L116.6,91.4 L118.4,91.1 L120.3,90.7 L122.1,90.4 L123.9,90.0 L125.8,89.6 L127.6,89.2 L129.5,88.8 L131.3,88.4 L133.2,88.0 L135.0,87.6 L136.9,87.1 L138.7,86.7 L140.6,86.2 L142.4,85.7 L144.3,85.2 L146.1,84.6 L148.0,84.1 L149.8,83.5 L151.7,82.9 L153.5,82.3 L155.4,81.7 L157.2,81.0 L159.1,80.4 L160.9,79.7 L162.8,79.0 L164.6,78.3 L166.5,77.6 L168.3,76.9 L170.2,76.1 L172.0,75.4 L173.9,74.6 L175.7,73.9 L177.6,73.1 L179.4,72.3 L181.3,71.5 L183.1,70.7 L185.0,69.9 L186.8,69.1 L188.7,68.3 L190.5,67.5 L192.4,66.7 L194.2,65.8 L196.1,65.0 L197.9,64.1 L199.7,63.2 L201.6,62.4 L203.4,61.5 L205.3,60.6 L207.1,59.7 L209.0,58.8 L210.8,57.8 L212.7,56.9 L214.5,55.9 L216.4,55.0 L218.2,54.0 L220.1,53.0 L221.9,52.0 L223.8,51.0 L225.6,50.0 L227.5,49.0 L229.3,48.0 L231.2,47.0 L233.0,46.0 L234.9,45.1 L236.7,44.1 L238.6,43.1 L240.4,42.2 L242.3,41.2 L244.1,40.3 L246.0,39.4 L247.8,38.5 L249.7,37.5 L251.5,36.6 L253.4,35.8 L255.2,34.9 L257.1,34.0 L258.9,33.1 L260.8,32.2 L262.6,31.4 L264.5,30.5 L266.3,29.7 L268.2,28.9 L270.0,28.0" fill="none" class="stroke-c" stroke-width="2.2"/>
<text x="56" y="47" class="fig-small">simplicity</text>
<text x="174" y="175" class="fig-small">data fit</text>
<path d="M345.0,132.4 L346.8,132.3 L348.7,132.2 L350.5,132.0 L352.4,131.8 L354.2,131.6 L356.1,131.4 L357.9,131.0 L359.8,130.7 L361.6,130.3 L363.5,129.9 L365.3,129.4 L367.2,128.9 L369.0,128.3 L370.9,127.7 L372.7,127.0 L374.6,126.4 L376.4,125.6 L378.3,124.9 L380.1,124.1 L382.0,123.3 L383.8,122.4 L385.7,121.5 L387.5,120.6 L389.4,119.7 L391.2,118.7 L393.1,117.8 L394.9,116.8 L396.8,115.7 L398.6,114.7 L400.5,113.7 L402.3,112.6 L404.2,111.5 L406.0,110.4 L407.9,109.4 L409.7,108.3 L411.6,107.1 L413.4,106.0 L415.3,104.9 L417.1,103.8 L418.9,102.6 L420.8,101.5 L422.6,100.3 L424.5,99.1 L426.3,97.9 L428.2,96.6 L430.0,95.3 L431.9,94.0 L433.7,92.7 L435.6,91.3 L437.4,89.9 L439.3,88.4 L441.1,86.9 L443.0,85.4 L444.8,83.9 L446.7,82.3 L448.5,80.6 L450.4,79.0 L452.2,77.3 L454.1,75.6 L455.9,73.9 L457.8,72.2 L459.6,70.5 L461.5,68.7 L463.3,67.0 L465.2,65.3 L467.0,63.7 L468.9,62.0 L470.7,60.4 L472.6,58.8 L474.4,57.2 L476.3,55.7 L478.1,54.2 L480.0,52.8 L481.8,51.4 L483.7,50.0 L485.5,48.7 L487.4,47.4 L489.2,46.3 L491.1,45.2 L492.9,44.2 L494.7,43.4 L496.6,42.7 L498.4,42.2 L500.3,42.0 L502.1,42.1 L504.0,42.5 L505.8,43.3 L507.7,44.5 L509.5,46.3 L511.4,48.5 L513.2,51.4 L515.1,54.8 L516.9,58.8 L518.8,63.3 L520.6,68.4 L522.5,74.1 L524.3,80.1 L526.2,86.5 L528.0,93.1 L529.9,99.8 L531.7,106.5 L533.6,113.2 L535.4,119.6 L537.3,125.7 L539.1,131.5 L541.0,136.9 L542.8,141.8 L544.7,146.3 L546.5,150.3 L548.4,153.9 L550.2,157.1 L552.1,159.9 L553.9,162.3 L555.8,164.4 L557.6,166.2 L559.5,167.7 L561.3,169.0 L563.2,170.2 L565.0,171.2" fill="none" class="stroke-b" stroke-width="2.2"/>
<circle cx="500.3" cy="42.0" r="4.5" class="fill-b"/>
<text x="500.3" y="32.0" text-anchor="middle" class="fig-small">best</text>
</svg>
Figure 3. Left: as the length scale grows, the data-fit term falls from -0.58 at 0.1 to -4.67 at 4, and the simplicity term rises from -0.28 to 3.39. Right: the sum plus the constant peaks near a length scale of 1.4, with a value of -5.40. Notice that the data-fit term alone prefers the shortest length scale, and only the simplicity term stops it.
```

Read the table behind it:

| Length scale | Data fit | Simplicity | Log evidence |
|---|---|---|---|
| 0.1 | -0.584 | -0.278 | -6.376 |
| 0.707 | -0.997 | 0.827 | -5.683 |
| 1.5 | -1.764 | 1.864 | -5.414 |
| 4.0 | -4.673 | 3.391 | -6.796 |

A very short length scale explains every point, but its prior would have explained almost anything, so it earns little credit. A very long one allows only slow trends and misses the wiggle in the data. The best length scale is about $1.4$, wider than the $0.707$ of Figure 2.

To search by gradient and not by grid, you need the slope of $\log p(y)$. Work in $\log\ell$, which keeps $\ell$ positive and treats a doubling the same at every scale. Write $A = K + \sigma^2 I$ and $\alpha = A^{-1}y$. For the RBF kernel, $\partial K/\partial \log\ell$ is $K$ times the squared distances divided by $\ell^2$, entry by entry. The slope is

$$
\frac{\partial \log p(y)}{\partial \log \ell} = \tfrac12\operatorname{tr}\!\big((\alpha\alpha^{\top} - A^{-1})\,\partial K/\partial \log \ell\big).
$$

Read it aloud: "how much the data's fit improves minus how much the volume grows, summed over every entry of the kernel." You can compute the trace as the sum of an entrywise product, because both matrices are symmetric.

```python
D2 = squared_distances(X, X)
dK = K * D2 / length_scale**2
A_inv = np.linalg.inv(K + noise * np.eye(n))
alpha = A_inv @ y
slope = 0.5 * np.sum((np.outer(alpha, alpha) - A_inv) * dK)
```

> [!TIP]
> Check this slope against `finite_difference` of the log marginal likelihood in $\log\ell$, as the exercise does. A missing term in $\alpha\alpha^{\top} - A^{-1}$ gives a slope with the right sign at some length scales and the wrong sign at others.

> [!KEY]
> A Gaussian process turns kernel ridge into a distribution: the mean you had, an error bar that grows away from the data, and a log marginal likelihood, a data-fit term plus a simplicity term, that picks the length scale without a held-out set.

> [!RECAP]
> - A Gaussian process says that any finite list of heights is Gaussian with zero mean and covariance $K$ from the kernel. Draw from it with $f = Lz$ and $K + 10^{-8}I = LL^{\top}$.
> - Conditioning on noisy data gives a mean $k_*^{\top}(K + \sigma^2 I)^{-1}y$ and a variance $k(x_*, x_*) - k_*^{\top}(K + \sigma^2 I)^{-1}k_*$. The variance is for the function, so add $\sigma^2$ for a fresh target.
> - The variance depends on the inputs and not on the targets. At a training point it is at most $\sigma^2$ and, for spread-out points, $\sigma^2/(1 + \sigma^2)$.
> - The mean equals kernel ridge with $\lambda = \sigma^2$. A linear kernel gives Bayesian linear regression.
> - $\log p(y) = -\tfrac12 y^{\top}A^{-1}y - \tfrac12\log\det A - \tfrac n2\log 2\pi$ with $A = K + \sigma^2 I$. Its slope in $\log\ell$ is $\tfrac12\operatorname{tr}((\alpha\alpha^{\top} - A^{-1})\,\partial K/\partial\log\ell)$.

## Build it step by step

You build six functions, in the order the checks test them. Every one that takes a kernel gets a function that maps two point sets to a Gram matrix.

1. **`rbf_kernel(X1, X2, length_scale)`**. `X1` is `(n1, d)` and `X2` is `(n2, d)`. Returns the `(n1, n2)` matrix $\exp(-\lVert x - x'\rVert^2 / (2\ell^2))$. Clip tiny negative squared distances to 0. The usual mistake is to write $\ell^2$ where $2\ell^2$ belongs, which makes every similarity too small.
2. **`gp_prior_samples(X, kernel, n, rng)`**. Factor `kernel(X, X) + 1e-8 * np.eye(m)`, draw one `(n, m)` array of standard normals from `rng`, and return `z @ L.T` with shape `(n, m)`. The usual mistake is `z @ L`.
3. **`gp_posterior(X, y, Xs, kernel, noise)`**. Returns the mean and the variance of the function at each row of `Xs`, both of shape `(m,)`. Factor `K + noise * I` once. The mean is `Ks @ alpha`. For the variance, solve `L V = Ks.T` and subtract `(V**2).sum(axis=0)` from the prior variances `np.diag(kernel(Xs, Xs))`. The usual mistake is to return the prior variance and forget the subtraction, which gives a variance of 1 everywhere.
4. **`log_marginal_likelihood(X, y, length_scale, noise)`**. Returns a float. Take the log determinant from the factor as `2 * sum(log(diag(L)))`, so half of it is `sum(log(diag(L)))`. The usual mistake is to drop the log-determinant term, which keeps the data-fit term and rewards the shortest length scale.
5. **`lml_grad_log_length_scale(X, y, length_scale, noise)`**. Returns the slope of step 4 with respect to $\log\ell$. Build `dK = K * D2 / length_scale**2`, then the entrywise sum from the formula. The usual mistake is to differentiate with respect to $\ell$, which is off by a factor of $\ell$.
6. **`best_length_scale(X, y, noise, grid)`**. Evaluates step 4 at every entry of `grid` and returns the entry with the highest value, as a Python float. The usual mistake is `np.argmin` in place of `np.argmax`.

## Cost and scaling

The exact method solves one $n \times n$ system. Forming $K$ takes about $n^2 d$ operations and $n^2$ numbers of memory, and the Cholesky factorisation takes about $n^3/3$ operations. After that, each posterior mean costs $O(n)$ per query and each variance costs $O(n^2)$, because it needs a triangular solve.

| Training points | Memory for $K$ in float64 | Flops for Cholesky |
|---|---|---|
| 1,000 | 0.008 GB | $3.3 \times 10^{8}$ |
| 10,000 | 0.80 GB | $3.3 \times 10^{11}$ |
| 100,000 | 80 GB | $3.3 \times 10^{14}$ |

Doubling $n$ makes the factorisation eight times slower, and every candidate length scale needs its own. Exact Gaussian processes are comfortable up to a few thousand points and strained at tens of thousands.

## Where this breaks

**Too many points.** Past about $10^4$ points, memory and cubic time dominate. Approximations replace the kernel with a low-rank one. The random features of the last chapter are one, and the Rasmussen and Williams book devotes a chapter to others.

**A wrong kernel.** The error bars are only as honest as the prior. An RBF kernel assumes a very smooth function, so near a kink the bands are too narrow. Choose the kernel for the shape you expect, and compare kernels by their evidence.

**Evidence is not a guarantee.** Maximising it over a few hyperparameters is stable. Over many it can overfit, and its surface can have several peaks, so start from several length scales.

**Class labels.** The likelihood is not Gaussian, so the posterior has no closed form and the method needs an approximation outside this chapter.

## The exercise

You build `rbf_kernel`, `gp_prior_samples`, `gp_posterior`, `log_marginal_likelihood`, `lml_grad_log_length_scale`, and `best_length_scale`. The checks compare prior draws with $K$ and run a grid that fails without jitter. They compare the posterior with the hand example, with kernel ridge at $\lambda = \sigma^2$, and with the variance rules. They match the log marginal likelihood to the hand value and to an independent Gaussian density, check its slope against a finite difference, and confirm that the best length scale on data drawn with $\ell = 1$ lands near 1.

## Further reading

For the regression formulas and the evidence, see [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/) (Rasmussen and Williams), chapter 2, Regression, and chapter 5, Model Selection and Adaptation of Hyperparameters. For methods that scale past a few thousand points, see chapter 8, Approximation Methods for Large Datasets, of the same book.

For a short derivation of the posterior, see [Gaussian processes](https://cs229.stanford.edu/section/cs229-gaussian_processes.pdf) (Chuong B. Do, updated by Honglak Lee), the CS229 handout, which the [CS229 course page](https://see.stanford.edu/Course/CS229) lists without a lecture number.

For a Bayesian view of what training data rules out, see the 6.S898 class [Bayesian analysis of learning and generalisation](https://phillipi.github.io/6.s898/) (28 September 2023). It is background, because the course page lists Gaussian processes only as optional reading for a later session.
