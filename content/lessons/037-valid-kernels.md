---
title: What makes a similarity a valid kernel
slug: 037-valid-kernels
part: "Part 2 — Classical models"
summary: A similarity is a valid kernel when it equals dot products of real features, which means its Gram matrix has no negative eigenvalue.
minutes: 35
gpu: false
objectives:
  - Explain why a valid kernel must give every combination of points a non-negative squared length.
  - Find a combination that exposes an invalid similarity, and compute its squared length by hand.
  - Test a Gram matrix with its smallest eigenvalue, using a tolerance, and say why eigvalsh is the right tool.
  - State which RBF and polynomial settings are valid and show a polynomial setting that is not.
prereqs:
  - 033-dual-multipliers-kernels
  - 036-kernels-as-similarity
exercise: 037-valid-kernels
---

# What makes a similarity a valid kernel

A colleague proposes a similarity for two products: score 1 when the products are identical and 2 when they differ. It is easy to compute and it sounds harmless. You write it as a table for two products, with 1 on the diagonal and 2 off it, and hand it to a kernel method.

Something is wrong with that table. It says each product is *more* similar to the other one than to itself. For a dot product that cannot happen, and the methods in this part of the course lean on the table behaving like dot products. Handed a table that does not, a solver can fail with an error, or worse, return a number that means nothing.

This chapter answers one question: when can you trust a similarity score in a kernel method? The answer has a picture, a one-line test, and a short computation.

## What you already know

- A kernel $k(x, x')$ is a dot product of feature lists, $\varphi(x)^{\top}\varphi(x')$ ("Kernels: a similarity score that hides a bigger feature space").
- A Gram matrix holds $k$ for every pair of points in one set.
- Ridge regression adds a fee $\lambda$ on the diagonal to keep a system solvable ("Ridge: a fee on large weights").
- The SVM dual uses the data only through the Gram matrix ("Pushes from each point: the dual and kernels").
- A symmetric matrix has real eigenvalues, and `np.linalg.eigvalsh` returns them in ascending order.

## The picture: features are real points

If $k(x, x') = \varphi(x)^{\top}\varphi(x')$ for some real feature lists, then each point has a position in feature space, and distances between positions are real. The squared distance between two positions follows from expanding the dot product:

$$
\lVert \varphi(x) - \varphi(x')\rVert^2 = k(x, x) + k(x', x') - 2\,k(x, x') .
$$

Read it aloud: "the squared distance is each point's score with itself, minus twice their score with each other." Check it with the simplest real features, $\varphi(x_1) = (1, 0)$ and $\varphi(x_2) = (0, 1)$. Their Gram matrix is the identity: $1 + 1 - 2\cdot 0 = 2$, and the arrow between the points has squared length 2.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Left: the arrow from x2 to x1 for K equal to the identity has squared length 2. Right: the invented matrix would need squared length minus 2.">
<defs><marker id="c037-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="70" y1="170" x2="200" y2="170" class="stroke-grid"/><line x1="70" y1="170" x2="70" y2="40" class="stroke-grid"/><line x1="70" y1="170" x2="160" y2="170" class="stroke-a" stroke-width="2.2" marker-end="url(#c037-arrow)"/><line x1="70" y1="170" x2="70" y2="80" class="stroke-a" stroke-width="2.2" marker-end="url(#c037-arrow)"/><line x1="70" y1="84" x2="157" y2="167" class="stroke-b dashed" stroke-width="2.2" marker-end="url(#c037-arrow)"/><text x="156.0" y="188.0" text-anchor="start" class="fig-bold">x1</text><text x="42.0" y="84.0" text-anchor="start" class="fig-bold">x2</text><text x="132.0" y="119.0" text-anchor="start" class="fig-small">x1 − x2</text><text x="40.0" y="24.0" text-anchor="start" class="fig-bold">Real features, K = I</text><text x="40.0" y="214.0" text-anchor="start" class="fig-mono">1 + 1 − 2·0 = 2</text><text x="40.0" y="234.0" text-anchor="start" class="fig-small">squared length 2: fine</text><text x="330.0" y="24.0" text-anchor="start" class="fig-bold">Invented K = [[1, 2], [2, 1]]</text><text x="330.0" y="70.0" text-anchor="start" class="fig-mono">k(x1,x1) = 1</text><text x="330.0" y="92.0" text-anchor="start" class="fig-mono">k(x2,x2) = 1</text><text x="330.0" y="114.0" text-anchor="start" class="fig-mono">k(x1,x2) = 2</text><text x="330.0" y="150.0" text-anchor="start" class="fig-mono fig-bold">1 + 1 − 2·2 = −2</text><rect x="322" y="162" width="250" height="34" rx="4" class="fill-soft-d"/><text x="330.0" y="184.0" text-anchor="start" class="fig-small">squared length −2: no such arrow</text><line x1="322" y1="134" x2="560" y2="134" class="stroke-bad"/>
</svg>
Figure 1. Left: real features give an arrow between the two points, and its squared length is a real, non-negative number. Right: the invented table would need a squared length of minus 2. Notice that the arithmetic runs without complaint and still describes a picture that cannot be drawn.
```

Try your colleague's table, $K = \begin{bmatrix}1 & 2\\ 2 & 1\end{bmatrix}$. The same formula gives $1 + 1 - 2\cdot 2 = -2$. A squared length of $-2$ is impossible, because a square of a real number is never negative. No real feature lists produce this table, so it is not a kernel.

## Any combination of points has a length

The arrow from one point to the other is one case of a general idea. Pick any weights $v_1, \ldots, v_n$ and form the combination $\sum_i v_i\,\varphi(x_i)$. The weights $(1, -1)$ give the arrow above. This combination is a real vector in feature space, so its squared length must be at least 0.

Stack the feature lists as the rows of a matrix $\Phi$, so that $K = \Phi\Phi^{\top}$. The combination is $\Phi^{\top}v$, and its squared length works out to the quantity $v^{\top}Kv$:

$$
v^{\top}Kv = v^{\top}\Phi\Phi^{\top}v = \lVert \Phi^{\top}v\rVert^2 \ge 0 .
$$

Read it aloud: "the weighted sum of all pairwise scores, using weights $v$ on both sides, is the squared length of a real vector." That is why *every* Gram matrix of a real feature map passes. A symmetric matrix with $v^{\top}Kv \ge 0$ for every $v$ is called *positive semidefinite*. For your colleague's table with $v = (1, -1)$:

$$
v^{\top}Kv = 1\cdot 1 + 1\cdot 1 - 2\cdot 2 = -2 .
$$

One weight vector that gives a negative number is enough to reject the table. That vector is the witness.

## From weights to eigenvalues

You cannot try every $v$. A symmetric matrix has a set of special directions, its eigenvectors, and along each one $v^{\top}Kv$ equals the eigenvalue times $\lVert v\rVert^2$. Every other $v$ is a mix of those directions, so $v^{\top}Kv$ cannot go negative unless one eigenvalue is negative. That turns "for every $v$" into one check: the smallest eigenvalue is at least 0.

For the table above, $\lambda^2 - 2\lambda - 3 = (\lambda - 3)(\lambda + 1) = 0$ gives eigenvalues 3 and $-1$. The eigenvector for $-1$ points along $(1, -1)$, the witness we found. Figure 2 compares those eigenvalues with the six eigenvalues of an RBF Gram matrix on six random points, which all sit above zero.

```figure
<svg viewBox="0 0 600 205" role="img" aria-label="Eigenvalues of an invented similarity matrix, one of them negative, and of an RBF Gram matrix, all positive.">
<line x1="266.7" y1="34" x2="266.7" y2="140" class="stroke-ink"/><line x1="120.0" y1="70" x2="560.0" y2="70" class="stroke-muted"/><circle cx="193.3" cy="70" r="6" class="fill-bad"/><circle cx="486.7" cy="70" r="6" class="fill-a"/><text x="8.0" y="74.0" text-anchor="start" class="fig-small">invented K</text><line x1="120.0" y1="120" x2="560.0" y2="120" class="stroke-muted"/><circle cx="272.9" cy="120" r="6" class="fill-a"/><circle cx="280.2" cy="120" r="6" class="fill-a"/><circle cx="326.9" cy="120" r="6" class="fill-a"/><circle cx="338.0" cy="120" r="6" class="fill-a"/><circle cx="361.9" cy="120" r="6" class="fill-a"/><circle cx="460.0" cy="120" r="6" class="fill-a"/><text x="8.0" y="124.0" text-anchor="start" class="fig-small">RBF Gram, 6 points</text><text x="120.0" y="164.0" text-anchor="middle" class="fig-small">-2</text><text x="193.3" y="164.0" text-anchor="middle" class="fig-small">-1</text><text x="266.7" y="164.0" text-anchor="middle" class="fig-small">0</text><text x="340.0" y="164.0" text-anchor="middle" class="fig-small">1</text><text x="413.3" y="164.0" text-anchor="middle" class="fig-small">2</text><text x="486.7" y="164.0" text-anchor="middle" class="fig-small">3</text><text x="560.0" y="164.0" text-anchor="middle" class="fig-small">4</text><text x="272.7" y="30.0" text-anchor="start" class="fig-small">zero</text><text x="193.3" y="52.0" text-anchor="middle" class="fig-bold">−1</text><text x="486.7" y="52.0" text-anchor="middle" class="fig-bold">3</text><text x="330.0" y="194.0" text-anchor="middle" class="fig-small">eigenvalues of each matrix</text>
</svg>
Figure 2. Eigenvalues on a number line. The invented table has one eigenvalue below zero, drawn in red. The RBF Gram matrix keeps all six above it. Notice that a valid matrix can sit close to zero, as 0.085 does here, without crossing.
```

The statement also runs the other way, and this is where the name Mercer's condition comes from: a similarity whose Gram matrices are positive semidefinite for every finite set of points is a kernel, with some feature lists behind it. This course uses the result and skips its proof.

## Which kernels are valid?

The RBF kernel with $\gamma > 0$ is valid for every set of points. The previous chapter wrote it as one scale factor per point times a weighted sum of polynomial kernels with positive weights, which is a dot product of real features at every step. The polynomial kernel $(x^{\top}x' + c)^p$ is valid when $c \ge 0$ and $p$ is a whole number of at least 1.

The condition on $c$ is real, and a negative $c$ is where the construction breaks. Take degree 2: $(x^{\top}x' + c)^2 = (x^{\top}x')^2 + 2c\,x^{\top}x' + c^2$. The middle term $2c\,x^{\top}x'$ is a dot product of the features $\sqrt{2c}\,x$ and $\sqrt{2c}\,x'$, and for a negative $c$ that square root is not a real number. That shows the usual recipe fails, not that the Gram matrix must fail, so Figure 3 checks it numerically on 8 random points in 3-D. It plots the smallest eigenvalue divided by the largest, which puts the three cases on one scale.

```figure
<svg viewBox="0 0 600 232" role="img" aria-label="Smallest eigenvalue divided by the largest for a degree-2 polynomial Gram matrix with coef0 of 1, 0, and minus 2.">
<line x1="60.0" y1="180.0" x2="560.0" y2="180.0" class="stroke-muted"/><line x1="60.0" y1="20.0" x2="60.0" y2="180.0" class="stroke-muted"/><text x="54.0" y="166.2" text-anchor="end" class="fig-small">-0.6</text><text x="54.0" y="130.7" text-anchor="end" class="fig-small">-0.4</text><text x="54.0" y="95.1" text-anchor="end" class="fig-small">-0.2</text><text x="54.0" y="59.6" text-anchor="end" class="fig-small">0</text><line x1="60.0" y1="55.6" x2="560.0" y2="55.6" class="stroke-ink"/><rect x="108.3" y="53.6" width="70" height="2.0" class="fill-a"/><text x="143.3" y="45.6" text-anchor="middle" class="fig-small">0.00085</text><text x="143.3" y="214.0" text-anchor="middle" class="fig-small">coef0 = 1</text><rect x="275.0" y="55.6" width="70" height="2.0" class="fill-a"/><text x="310.0" y="73.6" text-anchor="middle" class="fig-small">−2.6e−17</text><text x="310.0" y="214.0" text-anchor="middle" class="fig-small">coef0 = 0</text><rect x="441.7" y="55.6" width="70" height="102.7" class="fill-bad"/><text x="476.7" y="174.3" text-anchor="middle" class="fig-small">−0.58</text><text x="476.7" y="214.0" text-anchor="middle" class="fig-small">coef0 = −2</text>
</svg>
Figure 3. The smallest eigenvalue of a degree-2 polynomial Gram matrix, divided by the largest. With coef0 of 1 it is small and positive. With 0 it is zero up to rounding. With minus 2 it is clearly negative (the raw eigenvalue is about -36), so the matrix is not a valid kernel matrix.
```

The zero in the middle is not a mistake. With $c = 0$ the degree-2 kernel in 3-D has only 6 features, so 8 points cannot all be independent, and two eigenvalues are zero. Rounding leaves the smallest at about $-1.4 \times 10^{-15}$ (the $-2.6\times10^{-17}$ in Figure 3 is that number divided by the largest eigenvalue, about 55), which is why a test needs a tolerance and why a zero eigenvalue must be allowed.

## Testing a matrix in numpy

Three conditions define a usable Gram matrix: it is square, it agrees with its transpose within a tolerance `tol`, and its smallest eigenvalue is at least `-tol`. The symmetry test matters because $k(x, x') = k(x', x)$ must hold, and because `eigvalsh` reads only one triangle of the matrix and would silently ignore a mismatch in the other. Use `np.linalg.eigvalsh`, which is written for symmetric matrices and returns real eigenvalues in ascending order. The general `np.linalg.eig` can return complex numbers for a matrix with a small asymmetry.

```python
evals = np.linalg.eigvalsh(K)      # real, ascending
ok = evals[0] >= -tol              # zero is allowed
w = v @ K @ v                      # squared length of one combination
```

## What goes wrong downstream

Suppose you build a Gram matrix that fails the test and pass it to kernel ridge regression, which the next chapter introduces. The solver works with $K + \lambda I$. Add a small $\lambda = 0.5$ to the table above and the eigenvalues become $3.5$ and $-0.5$. A Cholesky factorisation of this matrix fails, because it requires every pivot to be positive. Raise $\lambda$ above 1 and the numbers go through, but the result no longer corresponds to a ridge fit in any feature space.

> [!KEY]
> A similarity is a valid kernel when it behaves like dot products of real features, which means every combination $v$ of points has $v^{\top}Kv \ge 0$, and that holds exactly when the Gram matrix has no negative eigenvalue. It answers the opening question: your colleague's table fails because the combination $(1, -1)$ has squared length $-2$.

> [!RECAP]
> - A valid kernel gives every point a real position in feature space, so the squared length of every combination $\sum_i v_i\varphi(x_i)$ is at least 0.
> - That length is $v^{\top}Kv$, so a Gram matrix must satisfy $v^{\top}Kv \ge 0$ for all $v$.
> - For a symmetric matrix this is the same as having no negative eigenvalue, which `np.linalg.eigvalsh` reports.
> - The RBF kernel with $\gamma > 0$ and the polynomial kernel with $c \ge 0$ are valid. A negative $c$ can break the property.
> - A zero eigenvalue is allowed, and rounding means you test against `-tol`.

## Build it step by step

The starter already contains `rbf_kernel` and `polynomial_kernel` from the previous chapter, so you can build Gram matrices to test. Build three functions.

1. **`min_eigenvalue(K)`**. Takes a symmetric `(n, n)` matrix and returns its smallest eigenvalue as a Python `float`. Use `np.linalg.eigvalsh(K)[0]`. The usual mistake is `np.linalg.eig`, whose eigenvalues are not sorted and may be complex.
2. **`combination_squared_length(K, v)`**. Takes `K` of shape `(n, n)` and `v` of shape `(n,)`, and returns the scalar `v @ K @ v` as a `float`. The usual mistake is `(v * K).sum()`, which spreads `v` across the columns only and gives a different number.
3. **`is_psd(K, tol)`**. Returns a Python `bool`. Return `False` for a non-square matrix, then `False` when `K` and `K.T` differ by more than `tol` in any entry, then compare `min_eigenvalue(K)` with `-tol`. The usual mistake is demanding a strictly positive eigenvalue, which rejects duplicated points.

Check the symmetry test with an absolute tolerance only, so that `tol` means what its name says:

```python
symmetric = np.allclose(K, K.T, atol=tol, rtol=0.0)
```

## Where this breaks

**Large entries.** The eigenvalue rounding error grows with the size of the entries. A polynomial Gram matrix with entries near $10^6$ can show an eigenvalue of $-10^{-9}$ even when it is valid. Scale the features first, or compare against `tol` times the largest eigenvalue.

**A tolerance that hides a real problem.** A `tol` of 1 would accept the table above, whose smallest eigenvalue is $-1$. Keep `tol` near $10^{-8}$ for matrices with entries of order 1.

## The exercise

You implement `min_eigenvalue`, `combination_squared_length`, and `is_psd`. The checks confirm the eigenvalues of small hand-built matrices, confirm that $v = (1, -1)$ gives $-2$ on the invented table and equals $\lVert\Phi^{\top}v\rVert^2$ when $K = \Phi\Phi^{\top}$, and confirm that `is_psd` accepts RBF and polynomial Gram matrices, including a matrix with a zero eigenvalue. They also confirm that `is_psd` rejects the invented table, an asymmetric matrix, a non-square matrix, and a polynomial Gram matrix with `coef0` of $-2$. One check confirms that the symmetry test uses an absolute tolerance, so a matrix with entries near $10^6$ cannot hide a mismatch of $10^{-3}$. The last check confirms that the eigenvector for a negative eigenvalue is a witness with a negative squared length.
