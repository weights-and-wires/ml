---
title: "Check a fit with right angles"
slug: 006-residuals-and-projection
part: "Part 1 — Foundations"
summary: Test a least-squares fit with one number, the dot products of the leftover errors with the columns, and see why zero means the best fit.
minutes: 40
gpu: false
objectives:
  - Compute the leftover errors of a fit and dot them with every column of the table.
  - Explain why dot products of zero mean that no nudge to any weight can lower the loss.
  - Describe the fitted values as the shadow of the targets on the sheet the columns span.
  - Turn the check into one number and into a yes-or-no answer with a sensible tolerance.
prereqs:
  - 005-least-squares-line
exercise: 006-residuals-and-projection
---

# Check a fit with right angles

A colleague sends you weights for the three houses from the previous chapter and says they are the best line. You have a solver, so you could refit and compare. But on a real problem with thousands of rows and dozens of weights, you want a test you can run on any weights, from any source, that says "nobody can do better" without a plot and without a second fit.

Trying nudges does not work, because a model with 50 weights can be nudged in infinitely many directions. You need a single number that is zero exactly when no nudge helps.

This chapter builds that number from a picture. The best weights place the fitted prices at the foot of a perpendicular dropped from the real prices, and the check is a test for that right angle.

## What you already know

- The table $X$ has one row per house and one column per ingredient. The guesses are `X @ w`.
- The loss is half the sum of squared misses, and its slope along every weight is `X.T @ (X @ w - y)`.
- At the bottom of the bowl every slope is zero. That gave the normal equations $X^\top X w = X^\top y$.
- The best weights for the three houses are $w=(2/3,1/2)$.
- A dot product multiplies two vectors entry by entry and adds the results.

## Name the leftovers

Last chapter you called the guess minus the truth the miss. Here it is easier to work with the opposite sign. The *residual* is what the model fails to explain:

$$
r = y - Xw
$$

Read it aloud: the true prices minus the guessed prices. The slope of the loss is $X^\top(Xw-y)=-X^\top r$. It is zero exactly when $X^\top r$ is zero, so the slope check and the residual check are the same check with a flipped sign.

Take the best weights $(2/3,1/2)$. The guessed prices, which this chapter calls the *fitted values*, are $2/3+1/2=7/6$, then $2/3+1=5/3$, then $2/3+3/2=13/6$. The true prices are $(1,2,2)$, so the residual is $(-1/6,\;1/3,\;-1/6)$.

Now dot the residual with each column of $X$. With the column of ones: $-1/6+1/3-1/6=0$. With the column of sizes $(1,2,3)$: $-1/6+2/3-3/6=0$. Both are zero.

Try the line $y=x$, with weights $(0,1)$. The fitted values are $(1,2,3)$ and the residual is $(0,0,-1)$. Dotting with the ones gives $-1$, and dotting with the sizes gives $-3$. Neither is zero. These are the slope from the previous chapter with its sign flipped, and they say which way the loss still falls.

## Why zero dot products mean best

A dot product of zero has a geometric meaning: the two vectors meet at a right angle. The word for this is *orthogonal*. So $X^\top r=0$ says the residual is orthogonal to every column of $X$.

Think about what the columns can reach. Any weights $w$ give a guess $Xw$, which is a mix of the columns: $w_1$ copies of the ones column plus $w_2$ copies of the sizes column. The set of all such mixes is the *column space*. For two columns in three-dimensional space it is a flat sheet through the origin, so you can draw it. With more columns it is a flat slice of higher dimension, and the argument below does not change. Figure 1 shows the sheet, the target vector $y=(1,2,2)$ floating above it, and the fitted values $Xw$ on the sheet.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="The vector y above the flat sheet made by the two columns of X. Its shadow on the sheet is Xw, and the leftover r drops straight down to the sheet at a right angle.">
<defs><marker id="c006-arrow-md" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-d"/></marker><marker id="c006-arrow-ma" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-a"/></marker><marker id="c006-arrow-mr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-bad"/></marker></defs>
<path d="M-2.3 198.2 L262.9 198.2 L360.2 126.6 L95.0 126.6 Z" class="fill-soft-a stroke-muted"/>
<line x1="50.0" y1="175.0" x2="140.1" y2="175.0" class="stroke-d" marker-end="url(#c006-arrow-md)"/>
<text x="136.1" y="191.0" class="fig-small" text-anchor="end">(1,1,1)</text>
<line x1="50.0" y1="175.0" x2="270.6" y2="145.2" class="stroke-d" marker-end="url(#c006-arrow-md)"/>
<text x="276.6" y="139.2" class="fig-small" text-anchor="start">(1,2,3)</text>
<line x1="50.0" y1="175.0" x2="220.3" y2="160.1" class="stroke-b dashed"/>
<line x1="50.0" y1="175.0" x2="220.3" y2="54.0" class="stroke-a" marker-end="url(#c006-arrow-ma)"/>
<line x1="220.3" y1="160.1" x2="220.3" y2="54.0" class="stroke-bad" marker-end="url(#c006-arrow-mr)"/>
<circle cx="220.3" cy="160.1" r="4.5" class="fill-b"/>
<circle cx="50.0" cy="175.0" r="3.5" class="fill-ink"/>
<text x="230.3" y="50.0" class="fig-bold">y = (1, 2, 2)</text>
<text x="232.3" y="180.1" class="fig-bold">Xw = (7/6, 5/3, 13/6)</text>
<text x="230.3" y="109.0" class="fig-bold">r = y - Xw</text>
<text x="60.0" y="193.0" class="fig-small">origin</text>
<text x="560.0" y="215.0" class="fig-small" text-anchor="end">the sheet: every</text>
<text x="560.0" y="231.0" class="fig-small" text-anchor="end">combination of the columns</text>
<path d="M211.3 160.1 L211.3 151.1 L220.3 151.1" class="stroke-ink"/>
</svg>
Figure 1. The sheet holds every guess the line model can make. The target y sits above it. The fitted values are the point of the sheet directly beneath y, and the residual r (red) drops straight down. Notice that the red arrow is perpendicular to the sheet, which is why it is the shortest way down. The vertical direction is stretched so the short arrow stays visible.
```

Picking the best weights means picking the point of the sheet closest to $y$. The closest point on a flat sheet to something above it is the foot of the perpendicular, as when you drop a plumb line from a ceiling. Here is the reason. Take any other point $p$ on the sheet. The residual is orthogonal to the sheet, so it is orthogonal to the segment from the fitted values to $p$. Pythagoras then gives $\lVert y-p\rVert^2=\lVert r\rVert^2+\lVert Xw-p\rVert^2$, which is larger than $\lVert r\rVert^2$ unless $p$ is the foot.

This also answers the nudge question directly. Change the weights by a small step $\delta$, so the fitted values move to $Xw+X\delta$, another point of the sheet. The same Pythagoras argument says the new squared residual is $\lVert r\rVert^2+\lVert X\delta\rVert^2$. You can add something that is zero or positive, so no nudge in any direction lowers the loss.

Check the Pythagoras step with real numbers. Take $p=(1,2,3)$, the guess from weights $(0,1)$. Then $\lVert y-p\rVert^2=1$. The residual has squared length $1/6$, and the segment from $(7/6,5/3,13/6)$ to $p$ is $(1/6,-1/3,-5/6)$, with squared length $30/36=5/6$. Indeed $1/6+5/6=1$.

> [!KEY]
> The fitted values are the *projection* of $y$ onto the column space: the point of the sheet closest to $y$, found by dropping a perpendicular. The residual is orthogonal to every column, which is the same sentence as "every slope is zero".

## See a good and a bad fit side by side

Figure 2 draws the weights $(2/3,1/2)$ next to the weights $(0,1)$.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Two copies of the sheet picture. Left: the best fit, with the leftover straight down and both dot products zero. Right: the weights (0, 1), where the leftover slants and the dot products are minus 1 and minus 3.">
<defs><marker id="c006-arrow-wd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-d"/></marker><marker id="c006-arrow-wa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-a"/></marker><marker id="c006-arrow-wr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-bad"/></marker></defs>
<path d="M-15.2 182.8 L188.8 182.8 L263.6 127.7 L59.6 127.7 Z" class="fill-soft-a stroke-muted"/>
<line x1="25.0" y1="165.0" x2="94.3" y2="165.0" class="stroke-d" marker-end="url(#c006-arrow-wd)"/>
<line x1="25.0" y1="165.0" x2="194.7" y2="142.1" class="stroke-d" marker-end="url(#c006-arrow-wd)"/>
<line x1="25.0" y1="165.0" x2="156.0" y2="153.5" class="stroke-b dashed"/>
<line x1="25.0" y1="165.0" x2="156.0" y2="71.9" class="stroke-a" marker-end="url(#c006-arrow-wa)"/>
<line x1="156.0" y1="153.5" x2="156.0" y2="71.9" class="stroke-bad" marker-end="url(#c006-arrow-wr)"/>
<circle cx="156.0" cy="153.5" r="4.5" class="fill-b"/>
<circle cx="25.0" cy="165.0" r="3.5" class="fill-ink"/>
<text x="25.0" y="22.0" class="fig-bold">Best weights (2/3, 1/2)</text>
<text x="25.0" y="196.0" class="fig-small">residual (-1/6, 1/3, -1/6)</text>
<text x="25.0" y="214.0" class="fig-small">dot products with columns: 0 and 0</text>
<text x="164.0" y="169.5" class="fig-small">Xw</text>
<text x="150.0" y="65.9" class="fig-small" text-anchor="end">y</text>
<text x="148.0" y="116.7" class="fig-bold" text-anchor="end">r</text>
<path d="M284.8 182.8 L488.8 182.8 L563.6 127.7 L359.6 127.7 Z" class="fill-soft-a stroke-muted"/>
<line x1="325.0" y1="165.0" x2="394.3" y2="165.0" class="stroke-d" marker-end="url(#c006-arrow-wd)"/>
<line x1="325.0" y1="165.0" x2="494.7" y2="142.1" class="stroke-d" marker-end="url(#c006-arrow-wd)"/>
<line x1="325.0" y1="165.0" x2="494.7" y2="142.1" class="stroke-b dashed"/>
<line x1="325.0" y1="165.0" x2="456.0" y2="71.9" class="stroke-a" marker-end="url(#c006-arrow-wa)"/>
<line x1="494.7" y1="142.1" x2="456.0" y2="71.9" class="stroke-bad" marker-end="url(#c006-arrow-wr)"/>
<circle cx="494.7" cy="142.1" r="4.5" class="fill-b"/>
<circle cx="325.0" cy="165.0" r="3.5" class="fill-ink"/>
<text x="325.0" y="22.0" class="fig-bold">Weights (0, 1)</text>
<text x="325.0" y="196.0" class="fig-small">residual (0, 0, -1)</text>
<text x="325.0" y="214.0" class="fig-small">dot products with columns: -1 and -3</text>
<text x="502.7" y="158.1" class="fig-small">Xw</text>
<text x="450.0" y="65.9" class="fig-small" text-anchor="end">y</text>
<text x="483.4" y="111.0" class="fig-bold" text-anchor="start">r</text>
</svg>
Figure 2. Left: the best fit, where the residual is straight down and both dot products are zero. Right: the weights (0, 1) land at a different point of the sheet. The residual slants, is longer, and has dot products of -1 and -3. Notice that the slant is the signal: part of the residual still points along the sheet, and that is the part the dot products pick up.
```

The slanted residual has a part that lies along the sheet. That part is what the dot products measure, and it is also the direction you could still move to shrink the error. Figure 3 collects the two dot products for three candidate weight vectors.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Bar chart of the two entries of X transpose r for three candidate weight vectors: minus 1 and minus 3, 2 and 5, and 0 and 0.">
<line x1="60.0" y1="230.0" x2="540.0" y2="230.0" class="stroke-grid"/>
<text x="52.0" y="234.0" class="fig-small" text-anchor="end">-4</text>
<line x1="60.0" y1="188.0" x2="540.0" y2="188.0" class="stroke-grid"/>
<text x="52.0" y="192.0" class="fig-small" text-anchor="end">-2</text>
<line x1="60.0" y1="104.0" x2="540.0" y2="104.0" class="stroke-grid"/>
<text x="52.0" y="108.0" class="fig-small" text-anchor="end">2</text>
<line x1="60.0" y1="62.0" x2="540.0" y2="62.0" class="stroke-grid"/>
<text x="52.0" y="66.0" class="fig-small" text-anchor="end">4</text>
<line x1="60.0" y1="20.0" x2="540.0" y2="20.0" class="stroke-grid"/>
<text x="52.0" y="24.0" class="fig-small" text-anchor="end">6</text>
<line x1="60.0" y1="146.0" x2="540.0" y2="146.0" class="stroke-ink"/>
<text x="52.0" y="150.0" class="fig-small" text-anchor="end">0</text>
<rect x="88.8" y="146.0" width="48.0" height="21.0" class="fill-a"/>
<text x="112.8" y="183.0" class="fig-small" text-anchor="middle">-1</text>
<rect x="143.2" y="146.0" width="48.0" height="63.0" class="fill-b"/>
<text x="167.2" y="225.0" class="fig-small" text-anchor="middle">-3</text>
<text x="140.0" y="250.0" class="fig-small" text-anchor="middle">w = (0, 1)</text>
<rect x="248.8" y="104.0" width="48.0" height="42.0" class="fill-a"/>
<text x="272.8" y="98.0" class="fig-small" text-anchor="middle">2</text>
<rect x="303.2" y="41.0" width="48.0" height="105.0" class="fill-b"/>
<text x="327.2" y="35.0" class="fig-small" text-anchor="middle">5</text>
<text x="300.0" y="250.0" class="fig-small" text-anchor="middle">w = (1, 0)</text>
<rect x="408.8" y="144.8" width="48.0" height="2.5" class="fill-a"/>
<text x="432.8" y="140.0" class="fig-small" text-anchor="middle">0</text>
<rect x="463.2" y="144.8" width="48.0" height="2.5" class="fill-b"/>
<text x="487.2" y="140.0" class="fig-small" text-anchor="middle">0</text>
<text x="460.0" y="250.0" class="fig-small" text-anchor="middle">w = (2/3, 1/2)</text>
<rect x="70.0" y="14" width="10" height="10" class="fill-a"/>
<text x="86.0" y="24.0" class="fig-small">entry 1: column (1,1,1) dotted with r</text>
<rect x="70.0" y="32" width="10" height="10" class="fill-b"/>
<text x="86.0" y="42.0" class="fig-small">entry 2: column (1,2,3) dotted with r</text>
</svg>
Figure 3. The two entries of X transpose r for three weight vectors. Only the best weights make both entries zero. Notice that the entries grow as the fit gets worse.
```

## Write the check as one number

A check needs to return a single number. Take the largest absolute value among the entries of $X^\top r$. This is the *infinity norm*, and it answers "how far from perpendicular is the worst column?". In symbols:

$$
\text{check}(w)=\max_j\,\bigl|\,(X^\top(y-Xw))_j\,\bigr|
$$

Read it aloud: dot the residual with each column, take absolute values, and keep the biggest. For $(0,1)$ the answer is 3. For $(1,0)$ the fitted values are $(1,1,1)$, the residual is $(0,1,1)$, and the dot products are $(2,5)$, so the answer is 5. For $(2/3,1/2)$ it is 0. Here it is in numpy:

```python
r = y - X @ w  # shape (n,)
check = float(np.abs(X.T @ r).max())
```

A computer never reports exactly 0. The best weights produce numbers such as $10^{-16}$, because of rounding. And "small" depends on the units: a check value of 0.001 means one thing when the entries of $X^\top y$ are 10 and another when they are $10^9$. So compare the check with the natural scale, `np.abs(X.T @ y).max()`:

```python
is_best = check <= tol * np.abs(X.T @ y).max()
```

> [!TIP]
> A relative tolerance of `1e-8` passes a correct solver on well-behaved data. A value near `1e-3` on that scale means the weights are off, not merely rounded.

## Why this is the normal equations again

Expand $X^\top(y-Xw)=0$ into $X^\top y - X^\top X w=0$. Move the second term across and you get $X^\top X w = X^\top y$, which is the normal equations from the previous chapter. The two are one fact in two outfits. The slope outfit says "the ground is flat". The geometry outfit says "the residual stands at a right angle to the sheet". Having both lets you check an answer you did not compute yourself.

One condition keeps this honest. If one column repeats another, the sheet has fewer dimensions than the table has columns. Many weight vectors then reach the same foot point, and all of them pass the check. The check says "this is a best fit", not "this is the only one". The chapter Stable fits with QR and the SVD takes that case on.

> [!RECAP]
> - The residual is $r=y-Xw$, the true values minus the fitted values.
> - The best fit has $X^\top r=0$: the residual is orthogonal to every column.
> - The fitted values are the projection of $y$ onto the sheet the columns span. The perpendicular is the shortest way down.
> - As one number, take `np.abs(X.T @ r).max()`, and compare it with `np.abs(X.T @ y).max()` to decide.

## Build it step by step

The starter already contains `fit_normal_equations`, which you wrote in the chapter Fit a line by minimising squared misses. You write three new functions.

1. **`residual_orthogonality(X, y, w)`** returns one float:

   ```python
   r = y - X @ w
   # abs of X.T @ r, then max, then float
   ```

   Returning the whole vector instead of one number fails the check.

2. **`project(X, y)`** returns the fitted values, shape `(n,)`, using the solver you already have:

   ```python
   w = fit_normal_equations(X, y)
   # return X @ w
   ```

   Returning `w` instead of `X @ w` gives the weights, not the shadow.

3. **`is_best_fit(X, y, w, tol)`** returns `True` when the check is at most `tol` times the scale of $X^\top y$:

   ```python
   scale = np.abs(X.T @ y).max()
   # compare residual_orthogonality(...) with tol * scale
   ```

   Comparing with `tol` alone gives different answers when you multiply all the prices by a million.

## Where this breaks

**Repeated columns.** Many weight vectors pass the check, as described earlier.

**A tolerance that is too tight.** A tolerance of $10^{-16}$ rejects correct answers because of rounding. Use a relative tolerance of about $10^{-8}$.

**Zero targets.** If $y$ is all zeros, then $X^\top y=0$ and a relative test compares with zero. Use an absolute tolerance in that case.

## The exercise

You implement `residual_orthogonality`, `project`, and `is_best_fit`. The checks use the three houses, where the answers are 3, 5, and 0 for the three candidate weights, and seeded random tables. They confirm that the projected residual is orthogonal to every column, that $\lVert y\rVert^2=\lVert Xw\rVert^2+\lVert r\rVert^2$, that no other point of the sheet is closer to $y$, and that `is_best_fit` agrees when the prices are scaled by a million.
