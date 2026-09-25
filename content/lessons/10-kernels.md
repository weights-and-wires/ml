---
title: Kernels and the representer theorem
slug: 10-kernels
part: "Part 2 — Classical machine learning"
summary: Build kernels, check they are valid, fit kernel ridge regression in closed form, and approximate it with random features.
minutes: 60
gpu: false
objectives:
  - Compute RBF and polynomial kernel matrices between two sets of points without a loop.
  - Check whether a matrix is a valid kernel matrix using its eigenvalues.
  - Build kernel ridge regression by solving one linear system, and use it to predict.
  - Build random Fourier features and watch their error shrink as you add more.
  - Explain the representer theorem and check it numerically.
prereqs:
  - 09-support-vector-machines
  - 04-regularisation
exercise: 10-kernels
---

# Kernels and the representer theorem

A linear model can only draw straight boundaries and fit straight lines. One fix is to invent new features, such as squares and products of the originals, and fit a linear model on those. That works, but the number of new features explodes fast.

The last lesson showed that an SVM only ever needs inner products between points. A **kernel** is a function that gives you the inner product of two points in some richer feature space, without ever building those features. Swap every inner product for a kernel, and a linear method learns curved patterns at almost no extra cost.

This lesson builds two common kernels, a test for whether a function is a valid kernel, and kernel ridge regression. It then shows why the best kernel model always has a simple form, and how to approximate a kernel with a fixed number of random features when your data gets large.

## What you already know

- An inner product $x^{\top}x'$ measures how much two vectors point the same way.
- Ridge regression adds $\lambda\lVert w\rVert^2$ to the squared error to keep weights small.
- A symmetric matrix has real eigenvalues, and `np.linalg.eigvalsh` computes them.
- Solving $A v = y$ with a Cholesky factor is faster and more accurate than forming $A^{-1}$.
- The SVM's dual and its predictions use the data only through the Gram matrix $K$.

## What a kernel is

A kernel $k(x, x')$ takes two points and returns a number that says how similar they are. The **Gram matrix**, or kernel matrix, holds that number for every pair: $K_{ij} = k(x_i, x_j)$. For two different point sets, of sizes $n_1$ and $n_2$, the matrix is $n_1 \times n_2$.

### The RBF kernel

The radial basis function (RBF) kernel, also called the Gaussian kernel, says two points are similar when they are close:

$$
k(x, x') = \exp\left(-\gamma\,\lVert x - x'\rVert^2\right) .
$$

Read it aloud: take the squared distance between the points, multiply by $-\gamma$, and exponentiate. Identical points give 1. Far-apart points give almost 0. The width setting $\gamma > 0$ controls how quickly similarity fades with distance.

Looping over every pair is slow. Instead, expand the squared distance: $\lVert x - x'\rVert^2 = \lVert x\rVert^2 + \lVert x'\rVert^2 - 2x^{\top}x'$. The last term for all pairs is one matrix product:

```python
sq = (X1**2).sum(axis=1)[:, None] + (X2**2).sum(axis=1)[None, :] - 2.0 * X1 @ X2.T
K = np.exp(-gamma * np.maximum(sq, 0.0))   # clip tiny negatives from rounding
```

> [!TIP]
> Check the result before you move on. `rbf_kernel(X, X, gamma)` must have shape `(n, n)`, be symmetric, and have values within rounding of 1 on its diagonal.

### The polynomial kernel

The polynomial kernel raises a shifted inner product to a power:

$$
k(x, x') = (x^{\top}x' + c)^{p} .
$$

Read it aloud: take the inner product, add a constant $c$ (`coef0`), and raise the result to the power $p$ (`degree`). It acts like a linear model on all products of up to $p$ features.

```python
K = (X1 @ X2.T + coef0) ** degree
```

### A worked example

Take the points $x_1 = [0, 0]$ and $x_2 = [1, 0]$ with $\gamma = 0.5$. The squared distances are 0 on the diagonal and 1 off it. So

$$
K = \begin{bmatrix} 1 & e^{-0.5} \\ e^{-0.5} & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0.6065 \\ 0.6065 & 1 \end{bmatrix} .
$$

For the polynomial kernel with $x = [1, 2]$, $x' = [3, 1]$, $c = 1$, and $p = 2$: the inner product is $3 + 2 = 5$, plus 1 gives 6, and $6^2 = 36$.

## When a function is a valid kernel

Not every similarity function is a kernel. For $k$ to be a real inner product in some feature space, every Gram matrix it builds must be **positive semidefinite (PSD)**. A symmetric matrix is PSD when all of its eigenvalues are zero or positive. This result is called Mercer's condition.

Why does it matter? A negative eigenvalue means some combination of points has a negative squared length, which no real feature space allows. Methods like kernel ridge regression and SVMs can then fail or return nonsense.

> [!KEY]
> A function is a valid kernel only if every Gram matrix it builds is positive semidefinite, with no eigenvalue below zero.

In code, check three things: the matrix is square, it equals its own transpose, and its smallest eigenvalue is not below a small negative tolerance:

```python
np.allclose(K, K.T, atol=tol) and np.linalg.eigvalsh(K).min() >= -tol
```

For example, $\begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$ is symmetric but has eigenvalues $-1$ and $3$. It is not PSD, so no kernel can produce it. The RBF and polynomial kernels always produce PSD matrices.

## Kernel ridge regression

Ridge regression fits weights $w$. Its kernel version fits one coefficient $\alpha_i$ per training point instead. The prediction at a new point $x$ is a weighted sum of its similarity to every training point:

$$
\hat{y}(x) = \sum_{i=1}^{n}\alpha_i\,k(x, x_i) .
$$

Think of each training point as a lamp. Its brightness at $x$ is $k(x, x_i)$, and $\alpha_i$ is how much you trust that lamp. The coefficients come from one linear system:

$$
(K + \lambda I)\,\alpha = y .
$$

Read it aloud: take the $n \times n$ training Gram matrix, add $\lambda$ to every diagonal entry, and solve for $\alpha$ against the targets $y$. The $\lambda I$ term plays the same role as in ridge regression. It keeps the solution stable and the function smooth. With $\lambda > 0$ and a PSD $K$, the matrix $K + \lambda I$ is always invertible.

> [!KEY]
> Kernel ridge regression fits one $\alpha_i$ per training point by solving $(K + \lambda I)\alpha = y$. You never compute a weight vector.

Solve it with a Cholesky factor $L$, where $L L^{\top} = K + \lambda I$. Solve $L z = y$, then $L^{\top}\alpha = z$:

```python
L = np.linalg.cholesky(K + lam * np.eye(n))
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
```

To predict, build `K_test` with shape `(n_test, n_train)`, the kernel between each query point and each training point. Then multiply: `K_test @ alpha`.

> [!WARNING]
> Keep the argument order straight. `K_test` has one row per query point and one column per training point, so build it as `rbf_kernel(X_test, X_train, gamma)`.

With the linear kernel $k(x, x') = x^{\top}x'$, kernel ridge regression gives exactly the same predictions as ordinary ridge regression without a bias. The exercise checks this.

### A worked example

Use the RBF Gram matrix from before, with targets $y = [1, 2]$ and $\lambda = 1$. Then

$$
K + \lambda I = \begin{bmatrix} 2 & 0.6065 \\ 0.6065 & 2 \end{bmatrix} .
$$

Its determinant is $2 \cdot 2 - 0.6065^2 = 3.6321$. Solving gives $\alpha = [0.2167, 0.9343]$. The training predictions are $K\alpha = [0.7833, 1.0657]$. They are pulled toward zero from the targets, which is the ridge penalty at work.

## Why the answer always has this form

You might wonder whether a better function exists that is not a weighted sum of kernels at the training points. The **representer theorem** says no. For a loss on the training points plus a penalty on the function's size, the best function is always $\sum_i \alpha_i k(\cdot, x_i)$ over the training points only.

The reason, in words: any extra piece of the function that is not built from training points has zero value at every training point. It cannot lower the loss. But it does add to the size penalty. So adding it only makes the objective worse.

> [!KEY]
> The best function is a weighted sum of kernels at the training points. Anything outside them cannot lower the training loss, and it makes the size penalty bigger.

You can check this with numbers. Describe a candidate function by coefficients over some set of points, possibly more than the training set. The objective is

$$
J = \sum_{i=1}^{n}\left(y_i - (Kc)_i\right)^2 + \lambda\, c^{\top}Kc .
$$

Read it aloud: $c$ is the vector of coefficients over all the points in $K$. The first term is the squared error, using only the first $n$ rows of $Kc$, which are the training points. The second term is $\lambda$ times the function's squared size, $c^{\top}Kc$. The error is a sum, not a mean.

```python
pred = (K @ coeffs)[:n]
J = np.sum((y - pred) ** 2) + lam * coeffs @ K @ coeffs
```

Adding a component that lives outside the training points leaves the first term alone and makes the second one bigger. The last check in the exercise does exactly that.

## Random Fourier features

Kernel ridge regression needs the full $n \times n$ Gram matrix and a solve that grows with $n^3$. For a million points, that is out of reach. **Random Fourier features** swap the kernel for an explicit list of $D$ features, where you choose $D$. A plain linear model on those features then acts almost like the RBF kernel.

The recipe:

1. Draw a $D \times d$ matrix $\Omega$ of random frequencies. Each entry is normal with mean 0 and variance $2\gamma$.
2. Draw $D$ random phase shifts $\phi$, each uniform between 0 and $2\pi$.
3. Build the features for every row of $X$:

$$
Z = \sqrt{\frac{2}{D}}\,\cos\left(X\Omega^{\top} + \phi\right) .
$$

Read it aloud: project the data onto $D$ random directions, shift each by a random phase, take the cosine, and scale by $\sqrt{2/D}$. The result $Z$ has shape $(n, D)$. On average over the random draws, $Z Z^{\top}$ equals the RBF Gram matrix exactly. With finite $D$, the typical error shrinks like $1/\sqrt{D}$. So four times as many features halve the error.

> [!KEY]
> Random Fourier features swap the $n \times n$ Gram matrix for an $n \times D$ feature matrix $Z$, with $Z Z^{\top} \approx K$. You buy accuracy with a larger $D$.

```python
omega = rng.normal(size=(D, d)) * np.sqrt(2.0 * gamma)
phase = rng.uniform(0.0, 2.0 * np.pi, size=D)
Z = np.sqrt(2.0 / D) * np.cos(X @ omega.T + phase[None, :])
```

Draw `omega` first and `phase` second, from the `rng` you are given. Use them for every row of `X`.

> [!RECAP]
> - Both kernels start from one `X1 @ X2.T`. The RBF kernel adds the row sums of squares and exponentiates, and the polynomial kernel adds `coef0` and raises to `degree`.
> - A valid Gram matrix is square, symmetric, and has no eigenvalue below `-tol`.
> - Kernel ridge solves $(K + \lambda I)\alpha = y$ with a Cholesky factor and predicts with `K_test @ alpha`.
> - Random features are $\sqrt{2/D}\cos(X\Omega^{\top} + \phi)$, with entries of $\Omega$ of standard deviation $\sqrt{2\gamma}$, drawn before $\phi$.
> - The representer objective is the summed squared error on the first $n$ rows of $Kc$, plus $\lambda\, c^{\top}Kc$.

## Build it step by step

Work through the functions in the order the checks test them.

1. **`rbf_kernel(X1, X2, gamma)`.** Takes `(n1, d)` and `(n2, d)` arrays. Returns `(n1, n2)`. Use row sums of squares with `[:, None]` and `[None, :]`, one `X1 @ X2.T`, then `np.maximum` and `np.exp`. The common mistake is getting the broadcasting wrong, so you add two vectors of the same length instead of making a matrix.
2. **`polynomial_kernel(X1, X2, degree, coef0)`.** Returns `(n1, n2)`. One line with `@`, `+`, and `**`. The common mistake is raising element-wise powers of `X1` before the product.
3. **`is_psd(K, tol)`.** Returns a Python `bool`. Return `False` if the matrix is not square or not symmetric within `tol`. Otherwise compare `np.linalg.eigvalsh(K).min()` against `-tol`. The common mistake is using `np.linalg.eig`, which can return complex numbers, or demanding eigenvalues strictly above zero.
4. **`kernel_ridge_fit(K, y, lam)`.** Returns `alpha` of shape `(n,)`. Factor `K + lam * np.eye(n)` with `np.linalg.cholesky`, then two solves. The common mistake is adding `lam` to every entry instead of only the diagonal.
5. **`kernel_ridge_predict(K_test, alpha)`.** Returns `(n_test,)`. One matrix-vector product. The common mistake is passing the transpose, of shape `(n_train, n_test)`.
6. **`random_fourier_features(X, D, gamma, rng)`.** Returns `(n, D)`. Draw `omega`, then `phase`, then apply the formula. The common mistake is scaling `omega` by `2 * gamma` instead of `np.sqrt(2 * gamma)`. The variance is $2\gamma$, so the standard deviation is its square root.
7. **`representer_check(K, y, lam, coeffs)`.** Returns a float. `K` is `(N, N)` with `N >= len(y)`. Keep the first `len(y)` entries of `K @ coeffs`, sum the squared errors, and add `lam * coeffs @ K @ coeffs`. The common mistake is using the mean squared error or slicing `coeffs` instead of the predictions.

## Cost and scaling

Everything exact about kernels runs through the $n \times n$ Gram matrix. The table uses $n$ training points, $d$ features, and $D$ random features.

| Method | Fit time | Memory | Predict one point |
|---|---|---|---|
| Exact kernel ridge | $n^2 d$ to build $K$, plus $n^3$ to solve | $n^2$ | grows with $n d$ |
| Random Fourier features plus ridge | grows with $n D d + n D^2$ | $n D$ | grows with $D d$ |

Doubling $n$ makes the exact solve eight times slower and the Gram matrix four times bigger. At $n = 100{,}000$, $K$ alone needs about 80 GB. Random features grow only linearly in $n$. You pick $D$ to trade accuracy for speed.

## Where this breaks

**A bad width $\gamma$.** A very large $\gamma$ makes every point similar only to itself, so the model memorises the training set. A very small $\gamma$ makes all points look alike, so the model is almost constant. Choose $\gamma$ and $\lambda$ together with cross-validation.

**Unscaled features.** The RBF kernel uses distances. A feature measured in thousands swamps one measured in fractions. Standardise every feature first.

**Too much data.** The exact method stops fitting in memory at around $10^5$ points. Use random Fourier features, or Nyström approximation, which builds features from a random subset of the training points.

**Invalid kernels.** A home-made similarity score that fails `is_psd` can make the Cholesky step fail or give meaningless answers. Test it before you use it.

## The exercise

You build a small kernel toolkit. Implement `rbf_kernel` and `polynomial_kernel` to return Gram matrices, and `is_psd` to test validity. Implement `kernel_ridge_fit` and `kernel_ridge_predict` to fit and use kernel ridge regression. Implement `random_fourier_features` to approximate the RBF kernel with explicit features, and `representer_check` to evaluate the regularised objective for any set of coefficients.

The checks compare both kernels with hand-built examples and confirm that their Gram matrices are PSD, while `is_psd` rejects a matrix that is not. They confirm that `kernel_ridge_fit` solves $(K + \lambda I)\alpha = y$ and that a linear kernel reproduces ordinary ridge regression. They check that random features approximate the RBF Gram matrix with error that shrinks as $D$ grows. The last check adds a component from outside the training points and confirms that `representer_check` reports a strictly larger objective every time.
