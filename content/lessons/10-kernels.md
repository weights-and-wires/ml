---
title: Kernels and the representer theorem
slug: 10-kernels
part: "Part 2 — Classical machine learning"
summary: Build kernels, fit kernel ridge regression, check the representer theorem, and approximate a kernel.
minutes: 60
gpu: false
objectives:
  - Compute RBF and polynomial kernel matrices between two sets of points.
  - Test whether a matrix is a valid kernel matrix from its eigenvalues.
  - Fit kernel ridge regression by solving one linear system, and use it to predict.
  - Build random Fourier features and watch the error shrink as you add more.
  - Explain the representer theorem and check it on a small matrix.
prereqs:
  - 09-support-vector-machines
  - 04-regularisation
exercise: 10-kernels
---

# Kernels and the representer theorem

A straight line cannot follow a curve. You can add squares and products of the original features, but ten features already give 45 pairwise products.

An SVM only needs inner products. A kernel hands you the inner product in a richer space, without building those features.

You will build two kernels and test that a matrix can be a kernel. You will fit kernel ridge regression, check the representer theorem, and approximate a kernel with random features.

## What you already know

These five ideas are the ones this lesson uses:

- An inner product $x^{\top}x'$ is large when two vectors point the same way. $[1, 0]$ with itself is 1.
- Ridge regression adds $\lambda\lVert w\rVert^2$ to squared error. With $\lambda = 1$ the penalty equals the squared length of $w$.
- A symmetric matrix has real eigenvalues, and `np.linalg.eigvalsh` returns them.
- Solving $Av = y$ with a Cholesky factor is more accurate than forming $A^{-1}$.
- The SVM dual uses the data only through the Gram matrix $K$.

## Measure similarity with a kernel

A kernel $k(x, x')$ scores two points. The *Gram matrix* has shape $(n_1, n_2)$ for two sets.

Take $[0, 0]$ and $[1, 0]$ with width $\gamma = 0.5$. Squared distances are 0 on the diagonal and 1 off it. Then $e^{-0.5 \times 0} = 1$ and $e^{-0.5} \approx 0.6065$.

That similarity is the *radial basis function* kernel, also called the Gaussian kernel:

$$
k(x, x') = \exp\left(-\gamma\,\lVert x - x'\rVert^2\right) .
$$

Identical points give 1, and far points give nearly 0. These two points return 1 on the diagonal and 0.6065 off it.

Expand the squared distance as $\lVert x\rVert^2 + \lVert x'\rVert^2 - 2x^{\top}x'$, then exponentiate:

```python
sq = (X1**2).sum(1)[:, None] + (X2**2).sum(1)[None, :] - 2 * X1 @ X2.T
K = np.exp(-gamma * np.maximum(sq, 0))  # shape (n1, n2)
```

The usual mistake is adding two length-$n$ vectors instead of broadcasting a column against a row.

> [!TIP]
> `rbf_kernel(X, X, gamma)` is symmetric, with diagonal entries within rounding of 1.

The polynomial kernel raises a shifted inner product to a power. Take $[1, 2]$ and $[3, 1]$, with $c = 1$ and $p = 2$. The inner product is $3 + 2 = 5$. Add 1 to get 6. Then $6^2 = 36$.

$$
k(x, x') = (x^{\top}x' + c)^{p} .
$$

$c$ is `coef0` and $p$ is `degree`. The same pair returns 36. Numpy is `(X1 @ X2.T + coef0) ** degree`.

The usual mistake is powering the entries of `X1` before the product.

## Reject a matrix that is not a kernel

Not every similarity is a kernel. Every Gram matrix must be *positive semidefinite*: symmetric, with no negative eigenvalue. This is Mercer's condition.

Take $\begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$. The determinant condition $(1 - \lambda)^2 - 4 = 0$ gives $\lambda^2 - 2\lambda - 3 = 0$. Factor it as $(\lambda - 3)(\lambda + 1) = 0$. The eigenvalues are 3 and $-1$.

A negative eigenvalue means some combination of points has a negative squared length. No real feature space allows that. Kernel ridge and an SVM can then fail.

> [!KEY]
> A kernel's Gram matrix is positive semidefinite. No eigenvalue sits below zero.

Require a square matrix, agreement with its transpose within `tol`, and smallest eigenvalue at least `-tol`. Numpy is `np.linalg.eigvalsh(K).min() >= -tol`.

The usual mistake is `np.linalg.eig`, which can return complex values, or demanding eigenvalues strictly above 0. A zero eigenvalue is allowed.

The RBF and polynomial kernels always return a positive semidefinite Gram matrix.

## Fit ridge regression with a kernel

Predict a new $x$ as a weighted sum of similarities to the training points:

$$
\hat{y}(x) = \sum_{i=1}^{n}\alpha_i\,k(x, x_i) .
$$

The coefficients solve one linear system:

$$
(K + \lambda I)\,\alpha = y .
$$

$K$ is the $n \times n$ training Gram matrix. Adding $\lambda$ on the diagonal is the ridge penalty. For $\lambda > 0$ and positive semidefinite $K$, the system has one solution.

Use the RBF matrix above, $y = [1, 2]$, and $\lambda = 1$. Then

$$
K + \lambda I = \begin{bmatrix} 2 & 0.6065 \\ 0.6065 & 2 \end{bmatrix}.
$$

The determinant is $4 - 0.6065^2 \approx 3.6321$. Solving gives $\alpha \approx [0.2167, 0.9343]$. Training predictions are $K\alpha \approx [0.7833, 1.0657]$, pulled toward 0 from $[1, 2]$.

Factor $L L^{\top} = K + \lambda I$, solve $Lz = y$, then $L^{\top}\alpha = z$:

```python
L = np.linalg.cholesky(K + lam * np.eye(n))  # shape (n, n)
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))  # shape (n,)
```

The usual mistake is adding `lam` to every entry, not only the diagonal.

> [!KEY]
> Kernel ridge fits one $\alpha_i$ per training point by solving $(K + \lambda I)\alpha = y$.

Predict with a matrix of shape `(n_test, n_train)`, built as the kernel from query rows to training rows. Then `K_test @ alpha`.

> [!WARNING]
> `K_test` has one row per query and one column per training point. Build it as `rbf_kernel(X_test, X_train, gamma)`.

The linear kernel $k = x^{\top}x'$ matches ridge regression with no bias. Points 1 and 2, targets 1 and 2, and $\lambda = 1$ predict $5/6$ and $5/3$ either way.

## Check that extra points cannot help

A piece that is zero on the training points cannot cut the loss, and it still adds to the penalty. The *representer theorem* says the best function uses kernels at the training points only.

Score a candidate by coefficients $c$ over a Gram matrix that may include extra points:

$$
J = \sum_{i=1}^{n}\left(y_i - (Kc)_i\right)^2 + \lambda\, c^{\top}Kc .
$$

The sum uses only the first $n$ entries of $Kc$. The second term is $\lambda$ times the squared size. The error is a sum, not a mean.

Take $K = I_2$, one training target $y = 2$, and $\lambda = 1$. The training block says $(1 + 1)\alpha = 2$, so $\alpha = 1$. With $c = [1, 0]$, the prediction is 1, the squared error is 1, the size is 1, and $J = 2$.

With $c = [1, 1]$ the prediction is still 1, the size is 2, and $J = 3$. The extra 1 is $\lambda t^2$ times the extra point's kernel value, here $1 \times 1^2 \times 1$. Slice the predictions like this:

```python
pred = (K @ coeffs)[:n]  # shape (n,)
```

The usual mistake is a mean squared error, or slicing `coeffs` instead of the predictions.

> [!KEY]
> The best function uses kernels at the training points only. An extra piece raises the penalty and not the fit.

## Replace the kernel with random features

Exact kernel ridge stores an $n \times n$ matrix and solves a system that grows like $n^3$. *Random Fourier features* swap the RBF kernel for $D$ explicit columns.

Draw a $D \times d$ matrix $\Omega$ with each entry normal, mean 0, variance $2\gamma$. Draw $D$ phases uniformly from 0 to $2\pi$, after $\Omega$. Then

$$
Z = \sqrt{\frac{2}{D}}\,\cos\left(X\Omega^{\top} + \phi\right) .
$$

$Z$ has shape $(n, D)$. On average over the draws, $ZZ^{\top}$ equals the RBF Gram matrix. The typical error shrinks like $1/\sqrt{D}$.

One point at 0, $D = 2$, and $\gamma = 0.5$ has frequency standard deviation 1. A draw $\Omega = [0, 0]$ and phases $0$, $\pi/2$ gives $\sqrt{2/2}\cos 0 = 1$ and $\cos(\pi/2) = 0$. The sum of squares is 1, matching $k(0, 0)$. Draw the frequencies like this:

```python
omega = rng.normal(size=(D, d)) * np.sqrt(2.0 * gamma)  # draw this first
Z = np.sqrt(2.0 / D) * np.cos(X @ omega.T + phase)
```

The usual mistake is scaling `omega` by `2 * gamma` instead of `np.sqrt(2 * gamma)`. At $\gamma = 2$ the standard deviation is 2, not 4.

> [!KEY]
> Random features replace the $n \times n$ Gram matrix with an $n \times D$ matrix $Z$, and $ZZ^{\top}$ approximates $K$.

Keep these checks in mind:

> [!RECAP]
> - The RBF kernel exponentiates a matrix of squared distances. The polynomial kernel powers `X1 @ X2.T + coef0`.
> - A Gram matrix is square, symmetric, and has no eigenvalue below `-tol`.
> - Kernel ridge solves $(K + \lambda I)\alpha = y$ and predicts with `K_test @ alpha`.
> - Random features are $\sqrt{2/D}\cos(X\Omega^{\top} + \phi)$, with $\Omega$ drawn before the phases.
> - The representer objective sums squared errors on the first $n$ rows of $Kc$, then adds $\lambda\, c^{\top}Kc$.

## Build it step by step

Build the seven functions in the order the checks test them:

1. **`rbf_kernel(X1, X2, gamma)`**. Takes `(n1, d)` and `(n2, d)`, returns `(n1, n2)`. Use squared row norms, `X1 @ X2.T`, `np.maximum`, and `np.exp`. The common mistake is adding two vectors. Distances look like this:

```python
row_sq = (X1**2).sum(1)[:, None]  # shape (n1, 1)
col_sq = (X2**2).sum(1)[None, :]  # shape (1, n2)
```

2. **`polynomial_kernel(X1, X2, degree, coef0)`**. Returns `(n1, n2)`. One product, one add, one power. The common mistake is powering `X1` first. The product has this shape:

```python
dots = X1 @ X2.T  # shape (n1, n2)
```

3. **`is_psd(K, tol)`**. Returns a Python `bool`. Return `False` if `K` is not square or not symmetric within `tol`. Otherwise compare `np.linalg.eigvalsh(K).min()` with `-tol`. The common mistake is `np.linalg.eig`. The test uses this vector:

```python
evals = np.linalg.eigvalsh(K)  # shape (n,)
```

4. **`kernel_ridge_fit(K, y, lam)`**. `K` is `(n, n)` and `y` is `(n,)`. Returns `alpha` of shape `(n,)`. Factor `K + lam * np.eye(n)` with `np.linalg.cholesky`, then two solves. The common mistake is adding `lam` everywhere. The factor has this shape:

```python
L = np.linalg.cholesky(K + lam * np.eye(n))  # shape (n, n)
```

5. **`kernel_ridge_predict(K_test, alpha)`**. `K_test` is `(n_test, n_train)`. Returns `(n_test,)`. The common mistake is a transpose of shape `(n_train, n_test)`. The product is one line:

```python
y_hat = K_test @ alpha  # shape (n_test,)
```

6. **`random_fourier_features(X, D, gamma, rng)`**. `X` is `(n, d)` and the return is `(n, D)`. Draw `omega` shaped `(D, d)` before `phase` shaped `(D,)`. The common mistake is scaling by `2 * gamma`. Features look like this:

```python
omega = rng.normal(size=(D, d)) * np.sqrt(2.0 * gamma)  # shape (D, d)
phase = rng.uniform(0.0, 2.0 * np.pi, size=D)           # shape (D,)
```

7. **`representer_check(K, y, lam, coeffs)`**. `K` is `(N, N)`, `coeffs` is `(N,)`, and `N >= len(y)`. Returns one float: summed squared error on `(K @ coeffs)[:n]`, plus `lam * coeffs @ K @ coeffs`. The common mistake is a mean, or slicing `coeffs`. The slice has this shape:

```python
pred = (K @ coeffs)[: len(y)]  # shape (n,)
```

## Cost and scaling

Exact kernels run through the $n \times n$ Gram matrix. The methods compare as follows, with $D$ random features:

| Method | Fit time | Memory | One prediction |
|---|---|---|---|
| Exact kernel ridge | $n^2 d$ to build $K$, then $n^3$ to solve | $n^2$ | grows like $n d$ |
| Random features plus ridge | grows like $n D d + n D^2$ | $n D$ | grows like $D d$ |

Doubling $n$ makes the exact solve eight times slower. At $n = 100000$, $K$ needs about 80 GB. Random features grow linearly in $n$, and you pick $D$.

## Where this breaks

**A bad width $\gamma$.** Very large $\gamma$ fits each training point alone. Very small $\gamma$ makes a nearly constant fit. Choose $\gamma$ and $\lambda$ with cross-validation.

**Unscaled features.** The RBF kernel uses distances. A feature in the thousands swamps a feature in fractions. Standardise every feature first.

**Too many rows.** The exact method stops fitting in memory near $10^5$ points. Use random Fourier features, or a Nyström approximation, which builds features from a random subset of the training points.

**An invalid kernel.** A home-made similarity that fails `is_psd` can make Cholesky fail. Test the matrix before you solve.

## The exercise

You implement `rbf_kernel`, `polynomial_kernel`, `is_psd`, `kernel_ridge_fit`, `kernel_ridge_predict`, `random_fourier_features`, and `representer_check`.

The checks match both kernels to hand-built examples and confirm that their Gram matrices are positive semidefinite, while `is_psd` rejects a matrix that is not. They confirm that `kernel_ridge_fit` solves $(K + \lambda I)\alpha = y$, and that a linear kernel matches ordinary ridge regression. They check that random features approach the RBF Gram matrix as $D$ grows. The last check adds a component from outside the training points and confirms that `representer_check` reports a strictly larger objective.
