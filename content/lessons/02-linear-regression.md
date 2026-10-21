---
title: Linear regression
slug: 02-linear-regression
part: "Part 1 — Foundations"
summary: Fit a line by least squares three ways, and see why the textbook formula loses correct digits.
minutes: 60
gpu: false
objectives:
  - Fit least squares weights with the normal equations, with QR, and with the SVD.
  - Check a fit by testing that leftover errors meet every column at a right angle.
  - Explain why building X transpose X loses about twice as many correct digits.
  - Build the shortest weight vector when one column repeats another.
prereqs:
  - 01-the-learning-problem
exercise: 02-linear-regression
---

# Linear regression

Three points sit on a page: (1, 1), (2, 2), and (3, 2).

No straight line hits all three. The line that guesses $x$ itself misses the third point by $3-2=1$, and that square is 1. You add the other squares and hunt for the smallest total.

The usual formula throws away correct digits when two columns look alike. You build it, then two safer methods, and you compare the digits.

## What you already know

Five ideas from lesson 1 are enough to start:

- A miss of 0.5 squares to 0.25. Lesson 1 calls that score a loss.
- Lesson 1 added losses 0.25, 0.25, 1, and 1 to get 2.5, then divided by 4 to get 0.625. That average is the empirical risk.
- A weighted sum such as `1*1 + 2*0.5` equals 2. numpy writes a stack of those sums as `X @ w`.
- The transpose of `[[1, 2], [3, 4]]` is `[[1, 3], [2, 4]]`. Rows become columns.
- The bowl `(w - 3) ** 2` is lowest at $w=3$, where the slope is 0.

## Score one line by hand

Use the line that guesses $x$. At $x=1$ and at $x=2$ the square is 0. At $x=3$ the miss is 1 and the square is 1.

Half of that total is $0.5\times(0+0+1)=0.5$. Stack the points into a table $X$ with a column of ones, and put the targets in $y$:

$$
X=\begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix},\quad y=\begin{bmatrix}1\\2\\2\end{bmatrix}
$$

The weights are $w_1=0$ and $w_2=1$. The three rows give $1\cdot0+1\cdot1=1$, then 2, then 3. The squares still add to 1, and half of that is 0.5.

This is linear regression. The guesses are $\hat{y}=Xw$, where $\hat{y}$ (y-hat) is the vector of guesses. The loss is half the sum of squared misses.

In symbols:

$$
L(w)=\tfrac{1}{2}\lVert Xw-y\rVert^2
$$

Here $\lVert v\rVert$ is the length of a vector $v$, so $\lVert v\rVert^2$ is the sum of squared entries. Plug in the error vector $(0,0,1)$. The sum of squares is 1, and half of it is 0.5 again.

Keep the half-sum, because the half tidies the derivative, while dividing by the number of rows would report $1/3$. The column of ones carries the intercept. Sum the squares as follows:

```python
loss = 0.5 * np.sum((X @ w - y) ** 2)  # guesses have shape (n,)
```

Dividing by $n$ reports $1/3$, so the slope no longer matches this loss.

## Solve the 2 by 2 system

The transpose $X^\top$ is $\begin{bmatrix}1&1&1\\1&2&3\end{bmatrix}$. Rows of $X$ became columns.

The top-left of $X^\top X$ is $1+1+1=3$. The top-right is $1\cdot1+1\cdot2+1\cdot3=6$. The bottom-right is $1+4+9=14$.

The targets give $X^\top y$. That is $1+2+2=5$ and $1+4+6=11$.

You solve $Aw=c$ with this 2 by 2 matrix $A$ and this right-hand side $c$:

$$
\begin{bmatrix}3&6\\6&14\end{bmatrix}\begin{bmatrix}w_1\\w_2\end{bmatrix}=\begin{bmatrix}5\\11\end{bmatrix}
$$

The determinant is $3\cdot14-6\cdot6=42-36=6$. Then $w_1=(14\cdot5-6\cdot11)/6=(70-66)/6=2/3$. And $w_2=(3\cdot11-6\cdot5)/6=(33-30)/6=1/2$.

Check the product. $3\cdot(2/3)+6\cdot(1/2)=2+3=5$, and $6\cdot(2/3)+14\cdot(1/2)=4+7=11$.

These are the normal equations. $X^\top X$ is $d\times d$, with $d=2$ columns, and it is the Gram matrix.

In symbols:

$$
X^\top X\,w=X^\top y
$$

The left side times $(2/3,1/2)$ is $(5,11)$, which is the right side.

The slope of the loss is $X^\top(Xw-y)$, one entry per weight. At $(2/3,1/2)$ that slope is the zero vector, the bottom of the bowl. At the trial weights $(0,1)$ the errors are $(0,0,1)$, and $X^\top$ of that is $(1,3)$, not zero.

The textbook writes $w=(X^\top X)^{-1}X^\top y$. Do not form that inverse. A solve is the same arithmetic and keeps more digits.

The exercise uses a Cholesky factor $L$, a lower triangle with $L L^\top$ equal to the Gram matrix. Two substitutions, through $L$ and then through $L^\top$, return $2/3$ and $1/2$. Solve it as follows:

```python
w = np.linalg.solve(X.T @ X, X.T @ y)  # shape (d,)
```

Forming the inverse, or forgetting $L^\top$ on the second substitution, gives a different vector.

## Check the leftovers

The best guesses are $2/3+1/2=7/6$, $2/3+1=5/3$, and $2/3+3/2=13/6$. In decimals that is about 1.1667, 1.6667, and 2.1667.

The leftovers are $1-7/6=-1/6$, $2-5/3=1/3$, and $2-13/6=-1/6$. Call this vector $r=y-Xw$, the residual.

Dot $r$ with the column of ones: $-1/6+1/3-1/6=0$. Dot $r$ with $(1,2,3)$: $-1/6+2/3-3/6=0$.

Those zeros are $X^\top r=0$. The residual meets every column at a right angle.

In symbols:

$$
X^\top(y-Xw)=0
$$

Plug in the two dots you computed. Both are 0.

The numpy check is one number, the largest absolute entry:

```python
check = np.abs(X.T @ (y - X @ w)).max()
```

Returning the vector fails the check, which wants one float. $Xw$ is the closest point to $y$ in the columns of $X$. That is a projection.

> [!KEY]
> Least squares projects $y$ onto the columns of $X$. The residual meets every column at a right angle, so $X^\top(y-Xw)=0$.

> [!TIP]
> Compare that check with `np.abs(X.T @ y).max()`. A result many orders smaller means the fit is right.

## Read the two stretch numbers

The stretches of $X$ are the square roots of the stretches of $X^\top X$. Those two numbers add to $3+14=17$ and multiply to $3\cdot14-6\cdot6=6$. They solve $t^2-17t+6=0$.

The discriminant is $17^2-24=289-24=265$, and $\sqrt{265}\approx16.279$. The roots are $(17+16.279)/2\approx16.639$ and $(17-16.279)/2\approx0.361$.

Then $\sqrt{16.639}\approx4.079$ and $\sqrt{0.361}\approx0.600$. These are the singular values of $X$, written $\sigma$ (sigma, said "sig-ma"). The large one is $\sigma_{\max}\approx4.079$ and the small one is $\sigma_{\min}\approx0.600$.

numpy returns them largest first:

```python
s = np.linalg.svd(X, compute_uv=False)
```

Square roots of the diagonal entries 3 and 14 are not the singular values.

## Count the digits you lose

Divide 4.079 by 0.600 and you get about 6.79. This ratio is the condition number, written $\kappa$ (kappa, said "kap-pa").

In symbols:

$$
\kappa=\sigma_{\max}/\sigma_{\min}
$$

Plug in 4.079 and 0.600 and you get about 6.79 again. The condition number of $X^\top X$ is $\kappa^2$, about $6.79^2\approx46.1$.

A condition number of $10^k$ loses about $k$ digits, and a double carries about 16. So $\kappa=10^7$ keeps about 9 digits on $X$, while $X^\top X$, with condition $10^{14}$, keeps about 2. That is about twice as many lost digits.

> [!KEY]
> Forming $X^\top X$ squares the condition number. The normal equations lose about twice as many digits as QR or the SVD.

Compute the ratio as follows:

```python
kappa = s[0] / s[-1]
```

Dividing the small stretch by the large one returns about 0.15. A condition number is at least 1, and a zero stretch must return `float("inf")`.

## Split the matrix with QR

QR writes $X=QR$. Classical steps on this $X$ go as follows.

The first column $(1,1,1)$ has length $\sqrt{3}\approx1.732$. Divide by that length. The second column $(1,2,3)$ dotted with that unit column is $6/\sqrt{3}\approx3.464$, which pulls off the vector $(2,2,2)$.

The remainder is $(-1,0,1)$, with length $\sqrt{2}\approx1.414$. Then $Q^\top y$ is about $(2.887,0.707)$, and $R$ is about $\begin{bmatrix}1.732&3.464\\0&1.414\end{bmatrix}$.

Back-substitution: $1.414\,w_2=0.707$, so $w_2=0.5$. Then $1.732\,w_1+3.464\cdot0.5=2.887$, so $w_1=2/3$. Same weights.

$Q$ is $n\times d$, here $3\times2$, with $Q^\top Q=I$. Each column has length 1, and columns meet at right angles. $R$ is $d\times d$ and upper triangular, so entries under the diagonal are 0.

In symbols:

$$
Rw=Q^\top y
$$

numpy may flip a sign in a column of $Q$ and the matching row of $R$. The product $QR$, and the weights $(2/3,1/2)$, stay the same. Factor $X$ as follows:

```python
Q, R = np.linalg.qr(X, mode="reduced")  # Q is (n, d), R is (d, d)
w = solve_triangular(R, Q.T @ y, lower=False)
```

`solve_triangular` comes from `scipy.linalg`. Full QR makes $Q$ shape `(n, n)` and the product no longer lines up.

## Divide by the stretch numbers

The SVD splits $X$ into $X=U\Sigma V^\top$. The columns of $U$, and the rows of $V^\top$, have length 1 and meet at right angles. $\Sigma$ (capital sigma) holds 4.079 and 0.600, and numpy calls $V^\top$ the matrix `vt`.

For this $X$, rounded to three decimals, $U^\top y$ is about $(-2.963,0.246)$. Divide by the stretches: $-2.963/4.079\approx-0.726$ and $0.246/0.600=0.410$.

Rotate back with $V$. The first weight is about $(-0.403)(-0.726)+(0.915)(0.410)\approx0.67$, and the second is about $(-0.915)(-0.726)+(-0.403)(0.410)\approx0.50$. Rounding aside, that is $(2/3,1/2)$ again.

In symbols:

$$
w=V\,\Sigma^{-1}\,U^\top y
$$

Compute that product as follows:

```python
U, s, vt = np.linalg.svd(X, full_matrices=False)
w = vt.T @ ((U.T @ y) / s)
```

> [!WARNING]
> numpy's `vt` is already $V^\top$. You get $V$ from `vt.T`. Treating `vt` as $V$ returns the wrong weights.

Sometimes one column repeats another. Take columns $(1,2,3)$ and $(2,4,6)$, with targets $(1,1,1)$. Any weights $(a,b)$ with $a+2b=3/7$ fit equally well, and the shortest pair is $(3/35,6/35)$, about $(0.086,0.171)$.

The second singular value is about 0. Dividing by it blows the weights up to around $10^{14}$.

Drop every singular value at or below `rcond * s[0]`. What remains is the shortest of the equally good weight vectors, the minimum-norm solution. The mask is as follows:

```python
keep = s > rcond * s[0]
w = vt[keep].T @ ((U[:, keep].T @ y) / s[keep])
```

Dividing by every singular value, including the zero, is the blow-up to $10^{14}$.

> [!KEY]
> When one column repeats another, many weight vectors fit equally well. Cutting the tiny singular values picks the shortest one.

> [!RECAP]
> - The loss is half the sum of squared residuals of `X @ w`.
> - The best $w$ solves $X^\top Xw=X^\top y$, and the residual is orthogonal to every column.
> - The condition number is `s[0] / s[-1]`, and $X^\top X$ squares it.
> - QR solves $Rw=Q^\top y$. The SVD uses $w=V\Sigma^{-1}U^\top y$.
> - For a repeated column, keep singular values above `rcond * s[0]`.

## Build it step by step

The checks call the functions in this order. Do not call `lstsq` or `pinv`.

1. **`condition_number(X)`** takes $X$ of shape `(n, d)` and returns one float, $\sigma_{\max}/\sigma_{\min}$:

   ```python
   s = np.linalg.svd(X, compute_uv=False)  # largest first
   # Return s[0] / s[-1], or float("inf") if s[-1] is 0.
   ```

   Dividing the last singular value by the first returns a number below 1.

2. **`residual_orthogonality(X, y, w)`** returns one float, the largest absolute entry of $X^\top(y-Xw)$:

   ```python
   # abs of X.T @ (y - X @ w), then .max(), then float.
   ```

   Returning the vector, instead of one number, fails the check.

3. **`fit_qr(X, y)`** returns `w` of shape `(d,)`:

   ```python
   Q, R = np.linalg.qr(X, mode="reduced")
   # solve_triangular(R, Q.T @ y, lower=False)
   ```

   Full QR makes `Q` shape `(n, n)`.

4. **`fit_normal_equations(X, y)`** returns `w` of shape `(d,)` through Cholesky:

   ```python
   L = np.linalg.cholesky(X.T @ X)  # L @ L.T is the Gram matrix
   # Solve L z = X.T @ y with lower=True.
   # Solve L.T w = z with lower=False.
   ```

   The second solve without `L.T` returns the wrong weights.

5. **`fit_svd(X, y, rcond)`** returns `w` of shape `(d,)`, the minimum-norm solution:

   ```python
   U, s, vt = np.linalg.svd(X, full_matrices=False)
   keep = s > rcond * s[0]
   # vt[keep].T times (U[:, keep].T @ y) / s[keep]
   ```

   Dividing by every singular value puts huge weights on repeated columns.

## Cost and scaling

All three methods cost about the same when you have many more rows than columns. Memory is mostly the table $X$ itself.

Doubling the rows roughly doubles the time. Doubling the columns roughly quadruples it.

Here $n$ is the number of rows, $d$ is the number of columns, and $k$ is the number of digits in the condition number. The work compares as follows:

|Method|Rough work|Digits lost|Repeated columns|
|---|---|---|---|
|Normal equations|$nd^2$ then $d^3$|about $2k$|No, Cholesky fails|
|QR|about $2nd^2$|about $k$|No, a zero on the diagonal of $R$|
|SVD|a few times $nd^2$|about $k$|Yes|

## Where this breaks

**Columns on wild scales.** Millimetres beside kilometres inflate the condition number. Standardise each column to mean 0 and standard deviation 1.

**Nearly repeated columns.** Weights swing when the data twitch. Cut tiny singular values, or use ridge regression from lesson 4.

**More columns than rows.** Many weight vectors fit perfectly. The shortest one can still miss unseen rows.

**One far point.** Squaring pulls the line toward that point. Plot the residuals, and try a loss that grows more slowly.

**A curved cloud.** A straight line misses a curve. Add a column such as $x^2$. The fit stays linear in the weights.

## The exercise

You build `condition_number`, `residual_orthogonality`, `fit_qr`, `fit_normal_equations`, and `fit_svd`. On a well-behaved problem the three solvers must agree, and each must leave a residual orthogonal to the columns of `X`.

`condition_number` must report the large singular value divided by the small one. `residual_orthogonality` must be the largest absolute entry of $X^\top(y-Xw)$. `fit_svd` must return the minimum-norm weights when columns repeat.

At condition number $10^7$, QR and the SVD must keep about nine digits, and the normal equations must lose more. None of the solvers may call `lstsq` or `pinv`.
