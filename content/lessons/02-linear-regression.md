---
title: Linear regression
slug: 02-linear-regression
part: "Part 1 — Foundations"
summary: Fit a straight line (or plane) to data by least squares, three ways, and see why the textbook formula loses accuracy.
minutes: 60
gpu: false
objectives:
  - Compute least squares weights with the normal equations, QR, and the SVD.
  - Check a fit by confirming the leftover errors are at right angles to every feature column.
  - Explain why forming X^T X loses about twice as many digits as QR or the SVD.
  - Build the smallest-size solution when some features repeat information from others.
prereqs:
  - 01-the-learning-problem
exercise: 02-linear-regression
---

# Linear regression

Linear regression draws the best straight line through a cloud of points. With more than one input feature, the line becomes a flat plane, but the idea is the same. You pick weights so that a weighted sum of the features lands as close as possible to each target.

"As close as possible" has a precise meaning here. You square each error and add the squares up, then choose the weights that make that total smallest. This is called least squares, and it has an exact answer you can compute in one go, without any trial and error.

The catch is that the formula printed in most textbooks is a poor way to compute that answer on a real computer. It loses accuracy when features are similar to each other. In this lesson, you build that formula anyway, then two better methods, and you measure the difference.

## What you already know

- A model makes a prediction from features, and a loss measures how wrong the prediction is (lesson 1).
- A matrix times a vector is a weighted sum of the matrix's columns.
- The transpose $X^\top$ swaps rows and columns of $X$.
- Solving $Aw = c$ for $w$ is a standard numpy call, `np.linalg.solve(A, c)`.
- A derivative equal to zero marks the bottom of a bowl-shaped function.

## The model and the loss

Think of predicting a flat's rent from its size and its distance to the station. Each flat is one example. Each measured property is one feature. You guess the rent as a weighted sum of the features.

You have $n$ examples and $d$ features. Stack them into the data matrix $X$, which has shape $n \times d$: one row per example, one column per feature. The targets go into a vector $y$ of length $n$. The weights $w$ form a vector of length $d$.

The predictions for every example at once are

$$\hat{y} = Xw$$

Read it aloud: for each row of $X$, multiply each feature by its weight and add them up. That sum is the prediction $\hat{y}$ for that row.

```python
y_hat = X @ w        # shape (n,)
```

This lesson has no separate bias $b$. To get one, add a column of ones to $X$. The weight on that column plays the role of $b$.

The loss is half the sum of squared errors:

$$L(w) = \tfrac{1}{2} \lVert Xw - y \rVert^2$$

Read it aloud: take the error vector $Xw - y$, one error per example. Square each error and add them. The symbol $\lVert v \rVert$ is the norm, or length, of a vector $v$, so $\lVert v \rVert^2$ is the sum of its squared entries. The $\tfrac{1}{2}$ only tidies up the derivative.

```python
loss = 0.5 * np.sum((X @ w - y) ** 2)
```

## The normal equations

The loss is a bowl in weight space. Its lowest point is where the gradient is zero. The gradient is the vector of partial derivatives, one per weight, and it points uphill. For this loss the gradient is $X^\top(Xw - y)$. Setting it to zero gives the normal equations:

$$X^\top X \, w = X^\top y$$

Read it aloud: on the left, $X^\top X$ is a $d \times d$ matrix, sometimes called the Gram matrix. Each entry measures how much two features overlap. On the right, $X^\top y$ is a length-$d$ vector that measures how much each feature lines up with the targets. The best $w$ solves this square system.

```python
w = np.linalg.solve(X.T @ X, X.T @ y)
```

The textbook writes the answer as $w = (X^\top X)^{-1} X^\top y$. Never compute an explicit inverse. A solve is faster and more accurate.

### The residual is at right angles to the features

The residual is the leftover error, $r = y - Xw$. The normal equations say $X^\top r = 0$. In words, the residual has zero dot product with every feature column. Two vectors with zero dot product are orthogonal: they meet at a right angle.

This gives you a free check on any solver. Compute $X^\top(y - Xw)$ and look at its largest absolute entry. At the least squares answer, it is zero up to rounding.

```python
check = np.abs(X.T @ (y - X @ w)).max()
```

Geometrically, $Xw$ is the point in the space of all possible predictions that sits closest to $y$. Closest means the line from it to $y$ drops straight down, at a right angle. This is called a projection.

## A worked example by hand

Fit a line $\hat{y} = w_1 + w_2 x$ to three points: $(1, 1)$, $(2, 2)$, and $(3, 2)$. The first column of ones gives the intercept $w_1$.

The data matrix and targets are:

$$X = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}, \quad y = \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}$$

The two sides of the normal equations are:

$$X^\top X = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}, \quad X^\top y = \begin{bmatrix} 5 \\ 11 \end{bmatrix}$$

The determinant of $X^\top X$ is $3 \cdot 14 - 6 \cdot 6 = 6$. Solving gives $w_1 = (14 \cdot 5 - 6 \cdot 11)/6 = 4/6 \approx 0.6667$ and $w_2 = (3 \cdot 11 - 6 \cdot 5)/6 = 3/6 = 0.5$.

The predictions are $\hat{y} = (1.1667, 1.6667, 2.1667)$. The residuals are $r = (-0.1667, 0.3333, -0.1667)$. Check them: they add to $0$, and $1(-0.1667) + 2(0.3333) + 3(-0.1667) = 0$. So $X^\top r = 0$, as promised.

The singular values of this $X$ are about $4.079$ and $0.600$, so its condition number (defined below) is about $6.79$.

## Why the textbook formula loses digits

The condition number of $X$ measures how close its columns are to repeating each other. It is the ratio of the largest to the smallest singular value, written $\kappa = \sigma_{\max}/\sigma_{\min}$. Singular values are defined in the SVD section below. For now, think of them as the stretch factors of $X$ along its main directions.

```python
s = np.linalg.svd(X, compute_uv=False)   # sorted largest first
kappa = s[0] / s[-1]
```

A rule of thumb: solving a problem with condition number $10^k$ loses about $k$ decimal digits. A computer double carries about 16 digits.

Here is the problem. The condition number of $X^\top X$ is $\kappa^2$, the square of the condition number of $X$. So the normal equations lose about $2k$ digits, while a method that works on $X$ directly loses only $k$. In the worked example, $\kappa(X) \approx 6.79$ and $\kappa(X^\top X) \approx 46.1$. If $\kappa = 10^7$, the normal equations keep about 2 digits and a good method keeps about 9.

The two methods below never form $X^\top X$.

## Solve with QR

The QR factorisation splits $X$ into $X = QR$. Here $Q$ has shape $n \times d$ and orthonormal columns: each column has length 1, and every pair is at right angles, so $Q^\top Q = I$. $R$ is $d \times d$ and upper triangular, meaning every entry below the diagonal is zero.

Substitute $X = QR$ into the normal equations and cancel. You get

$$R \, w = Q^\top y$$

Read it aloud: rotate the targets into the $Q$ coordinates, then solve a triangular system. A triangular system is solved one unknown at a time from the bottom row up. This is called back substitution.

```python
Q, R = np.linalg.qr(X, mode="reduced")
w = solve_triangular(R, Q.T @ y, lower=False)   # from scipy.linalg
```

## Solve with the SVD

The singular value decomposition (SVD) splits $X$ into $X = U \Sigma V^\top$. $U$ is $n \times r$ with orthonormal columns. $\Sigma$ is a diagonal matrix of singular values $s_1 \geq s_2 \geq \dots \geq 0$. $V^\top$ is $r \times d$ with orthonormal rows. numpy returns $s$ as a vector and $V^\top$ as `vt`.

The least squares weights are

$$w = V \, \Sigma^{-1} \, U^\top y$$

Read it aloud from right to left: rotate $y$ into the $U$ coordinates, divide each coordinate by its singular value, then rotate back with $V$.

```python
U, s, vt = np.linalg.svd(X, full_matrices=False)
w = vt.T @ ((U.T @ y) / s)
```

### When features repeat each other

Sometimes one column of $X$ is a combination of others. Say one column holds a length in metres and another the same length in centimetres. Then many different $w$ give exactly the same predictions, because you can shift weight between the two columns. $X$ is called rank deficient, and at least one singular value is zero.

Dividing by a zero singular value blows up. The fix is to drop every singular value at or below a small tolerance, `rcond * s.max()`, and keep only the rest. The result is the minimum-norm solution: of all the $w$ that fit equally well, it is the shortest one. It puts no weight at all in the directions that don't change the predictions.

```python
keep = s > rcond * s[0]
w = vt[keep].T @ ((U[:, keep].T @ y) / s[keep])
```

## Build it step by step

The checks test the functions in this order. Each one takes `X` with shape `(n, d)` and, where needed, `y` with shape `(n,)`.

1. **`condition_number(X)`** returns a float. Call `np.linalg.svd(X, compute_uv=False)` to get the singular values, sorted largest first, and divide the first by the last. Return `float("inf")` when the smallest is exactly zero. Common mistake: dividing the last by the first, which gives a number below 1.
2. **`residual_orthogonality(X, y, w)`** returns a float, the largest absolute entry of $X^\top(y - Xw)$. Use `X @ w`, `X.T @ ...`, `np.abs`, and `.max()`. Common mistake: returning the array instead of one number.
3. **`fit_qr(X, y)`** returns `w` with shape `(d,)`. Call `np.linalg.qr(X, mode="reduced")`, then `solve_triangular(R, Q.T @ y, lower=False)`. Common mistake: using the full QR, which makes `Q` shape `(n, n)` and breaks the shapes.
4. **`fit_normal_equations(X, y)`** returns `w` with shape `(d,)`. Form `X.T @ X` and `X.T @ y`. Factor the Gram matrix with `np.linalg.cholesky`, which returns a lower-triangular `L` with `L @ L.T` equal to the Gram matrix. Then do two triangular solves: first `L z = X^T y` with `lower=True`, then `L^T w = z` with `lower=False`. Common mistake: forgetting the transpose in the second solve.
5. **`fit_svd(X, y, rcond)`** returns `w` with shape `(d,)`. Call `np.linalg.svd(X, full_matrices=False)`. Build a boolean mask `keep` for singular values above `rcond * s[0]`, and use only those columns of `U`, entries of `s`, and rows of `vt`. Common mistake: dividing by every singular value, which puts huge weights in the directions that don't matter.

Don't call `np.linalg.lstsq` or `np.linalg.pinv` in the three solvers. They solve the whole problem for you, and a check rejects them.

## Cost and scaling

All three methods cost about the same when $n$ is much bigger than $d$. The table compares them.

| Method | Rough work | Digits lost | Handles repeated features |
|---|---|---|---|
| Normal equations | $nd^2$ to form $X^\top X$, then $d^3$ | about $2k$ | No, Cholesky fails |
| QR | about $2nd^2$ | about $k$ | No, $R$ has a zero on its diagonal |
| SVD | a few times $nd^2$ | about $k$ | Yes |

Here $k$ is the number of digits in the condition number. Memory is dominated by storing $X$ itself, $n \times d$ numbers. Doubling the number of examples roughly doubles the time. Doubling the number of features roughly quadruples it.

## Where this breaks

- **Features on wildly different scales.** A feature in millimetres next to one in kilometres inflates the condition number. Standardise each column to mean 0 and standard deviation 1 before fitting.
- **Nearly repeated features.** The weights become huge and swing wildly with small changes in the data. The SVD with a tolerance helps. Ridge regression, in lesson 4, is the standard fix.
- **More features than examples.** With $d > n$ there are infinitely many perfect fits. The minimum-norm solution picks one, but it may not predict new data well.
- **Outliers.** Squaring errors makes one far-off point pull the whole line toward it. Look at your residuals, and consider a loss that grows more slowly.
- **A curved relationship.** A straight line can't fit a curve. Add features such as $x^2$, and the method still works, because it is linear in the weights.

## The exercise

You build five functions: `condition_number`, `residual_orthogonality`, `fit_qr`, `fit_normal_equations`, and `fit_svd`. The three solvers must give the same answer on a well-behaved problem, and each must leave a residual at right angles to the columns of `X`.

The checks then push harder. `fit_svd` must return the minimum-norm solution when two columns repeat information from others. On a matrix with condition number $10^7$, `fit_qr` and `fit_svd` must keep about nine digits, while `fit_normal_equations` must visibly lose more. A final check confirms that none of the solvers calls `lstsq` or `pinv`.
