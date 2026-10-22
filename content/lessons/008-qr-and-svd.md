---
title: "Stable fits with QR and the SVD"
slug: 008-qr-and-svd
part: "Part 1 — Foundations"
summary: Solve least squares without forming X transpose X, by splitting X into rotations and easy pieces, and pick the shortest weights when columns repeat.
minutes: 45
gpu: false
objectives:
  - Build orthonormal columns from the columns of X by hand, and solve a triangular system for the weights.
  - Explain why working with the orthonormal factor keeps the condition number at kappa instead of kappa squared.
  - Describe the SVD as rotate, stretch, rotate, and solve by undoing each move.
  - Pick the minimum-norm weights when one column repeats another, and count the rank with a cut-off.
prereqs:
  - 007-lost-digits
exercise: 008-qr-and-svd
---

# Stable fits with QR and the SVD

The previous chapter ended with a culprit: forming $X^\top X$ squares the condition number. So can you solve least squares without ever forming it? And what should you do when one column repeats another, as when a house size appears once in units of 100 square metres and again in units of 50, so that the answer is not even unique?

Both questions have the same answer: stop working with the raw columns. Replace $X$ by pieces that are easy to undo and that do not stretch anything. This chapter gives you two ways to do that. The first, called QR, rebuilds the columns from perpendicular ones. The second, the SVD, turns $X$ into rotate, stretch, rotate. You solve three houses with each by hand, then handle the repeated column.

## What you already know

- Forming $X^\top X$ squares the condition number $\kappa=\sigma_{\max}/\sigma_{\min}$, so the normal equations lose about twice as many digits.
- The residual of the best fit is orthogonal to every column, and the fitted values are the projection of $y$ onto the sheet the columns span.
- A triangular system is solved one unknown at a time.
- Two vectors are orthogonal when their dot product is zero.
- A weight vector that adds nothing to the fit can be added freely, and it only makes the vector longer.

## Rebuild the columns from perpendicular ones

Take the three houses again: columns $a=(1,1,1)$ and $b=(1,2,3)$. They are not perpendicular, which is part of why $X^\top X$ is awkward. Build perpendicular columns that span the same sheet, one at a time.

Start with $a$. Its length is $\sqrt3\approx1.732$, so the unit-length version is $q_1=a/\sqrt3$. Next, remove from $b$ the part that points along $q_1$. That part has size $q_1\cdot b=6/\sqrt3\approx3.464$, which is the vector $3.464\,q_1=(2,2,2)$. What is left is $b-(2,2,2)=(-1,0,1)$. It is perpendicular to $a$ by construction. Its length is $\sqrt2\approx1.414$, so $q_2=(-1,0,1)/\sqrt2$.

Figure 1 draws this in the sheet. The vectors $q_1$ and $q_2$ are *orthonormal*: perpendicular, and each of length 1. Every column of $X$ is a recipe using them. Column 1 is $1.732\,q_1$. Column 2 is $3.464\,q_1+1.414\,q_2$. Collect the recipe numbers in a table $R$, so that $X=QR$, where the columns of $Q$ are $q_1$ and $q_2$:

$$
X=QR,\qquad R=\begin{bmatrix}1.732&3.464\\0&1.414\end{bmatrix}
$$

```figure
<svg viewBox="0 0 600 275" role="img" aria-label="Left: the two columns of X drawn in the sheet, with two perpendicular unit arrows q1 and q2 rebuilt from them, and the numbers 1.732, 3.464, and 1.414 that make up R. Right: the triangular system solved from the bottom row.">
<defs>
<marker id="c008-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-ink"/></marker>
<marker id="c008-arrowa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-a"/></marker>
<marker id="c008-arrowb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-b"/></marker>
<marker id="c008-arrowc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-c"/></marker>
</defs>
<line x1="40.0" y1="200.0" x2="300.0" y2="200.0" class="stroke-grid"/>
<line x1="40.0" y1="200.0" x2="40.0" y2="30.0" class="stroke-grid"/>
<text x="40.0" y="18.0" class="fig-bold">The sheet, drawn flat</text>

<line x1="40.0" y1="200.0" x2="213.2" y2="129.3" class="stroke-d" marker-end="url(#c008-arrow)"/>
<line x1="40.0" y1="200.0" x2="126.6" y2="200.0" class="stroke-a" marker-end="url(#c008-arrowa)"/>
<line x1="213.2" y1="200.0" x2="213.2" y2="129.3" class="stroke-muted dashed"/>
<line x1="40.0" y1="214.0" x2="90.0" y2="214.0" class="stroke-c" marker-end="url(#c008-arrowc)"/>
<text x="96.0" y="218.0" class="fig-small">q1</text>
<line x1="26.0" y1="200.0" x2="26.0" y2="150.0" class="stroke-c" marker-end="url(#c008-arrowc)"/>
<text x="20.0" y="175.0" class="fig-small" text-anchor="end">q2</text>

<text x="165.0" y="190.0" class="fig-small" text-anchor="middle">col 1 = 1.732 q1</text>
<text x="213.2" y="216.0" class="fig-small" text-anchor="middle">3.464</text>
<text x="207.2" y="121.3" class="fig-small" text-anchor="end">col 2 = 3.464 q1 + 1.414 q2</text>
<text x="219.2" y="168.6" class="fig-small">1.414</text>

<text x="40.0" y="252.0" class="fig-small">q1, q2: perpendicular, length 1. R holds these</text>
<text x="40.0" y="268.0" class="fig-small">numbers: [[1.732, 3.464], [0, 1.414]]</text>
<line x1="330.0" y1="35.0" x2="330.0" y2="235.0" class="stroke-grid"/>
<text x="345.0" y="18.0" class="fig-bold">Solve R w = Q^T y from the bottom</text>
<text x="345.0" y="50.0" class="fig-small fig-mono">1.732 w1 + 3.464 w2 = 2.887</text>
<text x="345.0" y="70.0" class="fig-small fig-mono">            1.414 w2 = 0.707</text>
<text x="345.0" y="90.0" class="fig-small fig-mono"></text>
<text x="345.0" y="110.0" class="fig-small fig-mono">bottom row: w2 = 0.707 / 1.414 = 0.5</text>
<text x="345.0" y="130.0" class="fig-small fig-mono"></text>
<text x="345.0" y="150.0" class="fig-small fig-mono">top row:</text>
<text x="345.0" y="170.0" class="fig-small fig-mono">w1 = (2.887 - 3.464 x 0.5) / 1.732</text>
<text x="345.0" y="190.0" class="fig-small fig-mono">   = 0.667</text>
<text x="345.0" y="252.0" class="fig-small">Each row brings in one new unknown.</text>
</svg>
Figure 1. Left: column 2 splits into a part along q1 (3.464) and a part along q2 (1.414). Those two numbers and the length of column 1 form R. Right: because R is triangular, you solve it from the bottom row up. Notice that nothing here multiplies X by its own transpose.
```

$R$ is upper triangular: zeros below the diagonal, because column 1 uses only $q_1$. Now solve. Substitute $X=QR$ into the normal equations $X^\top Xw=X^\top y$. Because $Q^\top Q=I$ (perpendicular unit columns), they become $R^\top Rw=R^\top Q^\top y$, and, when no column repeats, you can cancel $R^\top$, leaving

$$
R\,w=Q^\top y
$$

Read it aloud: dot $y$ with each perpendicular direction, then solve the triangular system. Here is the picture behind it, from the chapter Check a fit with right angles. Dotting $y$ with each perpendicular direction gives the shadow's coordinates, so the shadow is $QQ^\top y$. The fitted values are $Xw=QRw$, and setting them equal to the shadow gives $Rw=Q^\top y$, the same system. For the houses, $Q^\top y=(5/\sqrt3,\,1/\sqrt2)\approx(2.887,\,0.707)$. The bottom row gives $1.414\,w_2=0.707$, so $w_2=0.5$. The top row gives $1.732\,w_1+3.464\cdot0.5=2.887$, so $w_1=0.667$. These are the weights $(2/3,1/2)$ again.

Why is this safer? Multiplying by $Q$ only rotates, so it does not change any length. Then $R$ has the same singular values as $X$, and the system $Rw=Q^\top y$ has condition number $\kappa$, not $\kappa^2$. At $\kappa=10^7$ you keep about nine digits instead of two.

In numpy, the one trap is the shape of $Q$. Ask for the reduced factorisation, where $Q$ is $n\times d$:

```python
Q, R = np.linalg.qr(X, mode="reduced")  # Q is (n, d), R is (d, d)
w = solve_triangular(R, Q.T @ y, lower=False)
```

numpy may flip the signs of a column of $Q$ and the matching row of $R$. The product $QR$ and the weights stay the same.

## Rotate, stretch, rotate

QR needs full-rank columns: a repeated column puts a zero on the diagonal of $R$. The SVD (singular value decomposition) handles every case. It says every table splits as

$$
X=U\,\Sigma\,V^\top
$$

Read it aloud: first rotate the weights with $V^\top$, then stretch each axis by its singular value in $\Sigma$, then rotate into the sheet with $U$. Figure 2 shows both directions. Top row: the unit circle becomes the ellipse of the previous chapter. Bottom row: to solve, undo the moves in reverse order.

```figure
<svg viewBox="0 0 600 345" role="img" aria-label="Top row: the unit circle is rotated, stretched along its axes by 4.08 and 0.60, and rotated again to give the ellipse. Bottom row: solving undoes the three moves in reverse order: rotate back, divide by the stretches, rotate back. A red stop sign warns that dividing by a stretch near zero blows up.">
<defs>
<marker id="c008-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-ink"/></marker>
</defs>
<text x="15.0" y="15.0" class="fig-bold">Fit: what X does</text>
<line x1="18.0" y1="75.0" x2="142.0" y2="75.0" class="stroke-grid"/>
<line x1="80.0" y1="35.0" x2="80.0" y2="115.0" class="stroke-grid"/>
<path d="M93.0 75.0 L93.0 74.3 L92.9 73.6 L92.8 73.0 L92.7 72.3 L92.6 71.6 L92.4 71.0 L92.1 70.3 L91.9 69.7 L91.6 69.1 L91.3 68.5 L90.9 67.9 L90.5 67.4 L90.1 66.8 L89.7 66.3 L89.2 65.8 L88.7 65.3 L88.2 64.9 L87.6 64.5 L87.1 64.1 L86.5 63.7 L85.9 63.4 L85.3 63.1 L84.7 62.9 L84.0 62.6 L83.4 62.4 L82.7 62.3 L82.0 62.2 L81.4 62.1 L80.7 62.0 L80.0 62.0 L79.3 62.0 L78.6 62.1 L78.0 62.2 L77.3 62.3 L76.6 62.4 L76.0 62.6 L75.3 62.9 L74.7 63.1 L74.1 63.4 L73.5 63.7 L72.9 64.1 L72.4 64.5 L71.8 64.9 L71.3 65.3 L70.8 65.8 L70.3 66.3 L69.9 66.8 L69.5 67.4 L69.1 67.9 L68.7 68.5 L68.4 69.1 L68.1 69.7 L67.9 70.3 L67.6 71.0 L67.4 71.6 L67.3 72.3 L67.2 73.0 L67.1 73.6 L67.0 74.3 L67.0 75.0 L67.0 75.7 L67.1 76.4 L67.2 77.0 L67.3 77.7 L67.4 78.4 L67.6 79.0 L67.9 79.7 L68.1 80.3 L68.4 80.9 L68.7 81.5 L69.1 82.1 L69.5 82.6 L69.9 83.2 L70.3 83.7 L70.8 84.2 L71.3 84.7 L71.8 85.1 L72.4 85.5 L72.9 85.9 L73.5 86.3 L74.1 86.6 L74.7 86.9 L75.3 87.1 L76.0 87.4 L76.6 87.6 L77.3 87.7 L78.0 87.8 L78.6 87.9 L79.3 88.0 L80.0 88.0 L80.7 88.0 L81.4 87.9 L82.0 87.8 L82.7 87.7 L83.4 87.6 L84.0 87.4 L84.7 87.1 L85.3 86.9 L85.9 86.6 L86.5 86.3 L87.1 85.9 L87.6 85.5 L88.2 85.1 L88.7 84.7 L89.2 84.2 L89.7 83.7 L90.1 83.2 L90.5 82.6 L90.9 82.1 L91.3 81.5 L91.6 80.9 L91.9 80.3 L92.1 79.7 L92.4 79.0 L92.6 78.4 L92.7 77.7 L92.8 77.0 L92.9 76.4 L93.0 75.7 L93.0 75.0 Z" class="fill-soft-b stroke-b"/>
<line x1="80.0" y1="75.0" x2="93.0" y2="75.0" class="stroke-a"/>
<line x1="80.0" y1="75.0" x2="80.0" y2="62.0" class="stroke-c"/>
<text x="80.0" y="129.0" class="fig-small" text-anchor="middle">unit circle</text>
<line x1="146.0" y1="75.0" x2="166.0" y2="75.0" class="stroke-ink" marker-end="url(#c008-arrow)"/>
<line x1="168.0" y1="75.0" x2="292.0" y2="75.0" class="stroke-grid"/>
<line x1="230.0" y1="35.0" x2="230.0" y2="115.0" class="stroke-grid"/>
<path d="M224.8 63.1 L224.1 63.4 L223.6 63.7 L223.0 64.1 L222.4 64.4 L221.9 64.9 L221.3 65.3 L220.8 65.8 L220.4 66.3 L219.9 66.8 L219.5 67.3 L219.1 67.9 L218.8 68.4 L218.4 69.0 L218.1 69.7 L217.9 70.3 L217.7 70.9 L217.5 71.6 L217.3 72.2 L217.2 72.9 L217.1 73.6 L217.0 74.3 L217.0 74.9 L217.0 75.6 L217.1 76.3 L217.2 77.0 L217.3 77.6 L217.4 78.3 L217.6 79.0 L217.8 79.6 L218.1 80.2 L218.4 80.9 L218.7 81.4 L219.1 82.0 L219.4 82.6 L219.9 83.1 L220.3 83.7 L220.8 84.2 L221.3 84.6 L221.8 85.1 L222.3 85.5 L222.9 85.9 L223.4 86.2 L224.0 86.6 L224.7 86.9 L225.3 87.1 L225.9 87.3 L226.6 87.5 L227.2 87.7 L227.9 87.8 L228.6 87.9 L229.3 88.0 L229.9 88.0 L230.6 88.0 L231.3 87.9 L232.0 87.8 L232.6 87.7 L233.3 87.6 L234.0 87.4 L234.6 87.2 L235.2 86.9 L235.9 86.6 L236.4 86.3 L237.0 85.9 L237.6 85.6 L238.1 85.1 L238.7 84.7 L239.2 84.2 L239.6 83.7 L240.1 83.2 L240.5 82.7 L240.9 82.1 L241.2 81.6 L241.6 81.0 L241.9 80.3 L242.1 79.7 L242.3 79.1 L242.5 78.4 L242.7 77.8 L242.8 77.1 L242.9 76.4 L243.0 75.7 L243.0 75.1 L243.0 74.4 L242.9 73.7 L242.8 73.0 L242.7 72.4 L242.6 71.7 L242.4 71.0 L242.2 70.4 L241.9 69.8 L241.6 69.1 L241.3 68.6 L240.9 68.0 L240.6 67.4 L240.1 66.9 L239.7 66.3 L239.2 65.8 L238.7 65.4 L238.2 64.9 L237.7 64.5 L237.1 64.1 L236.6 63.8 L236.0 63.4 L235.3 63.1 L234.7 62.9 L234.1 62.7 L233.4 62.5 L232.8 62.3 L232.1 62.2 L231.4 62.1 L230.7 62.0 L230.1 62.0 L229.4 62.0 L228.7 62.1 L228.0 62.2 L227.4 62.3 L226.7 62.4 L226.0 62.6 L225.4 62.8 L224.8 63.1 Z" class="fill-soft-b stroke-b"/>
<line x1="230.0" y1="75.0" x2="224.8" y2="63.1" class="stroke-a"/>
<line x1="230.0" y1="75.0" x2="218.1" y2="80.2" class="stroke-c"/>
<text x="230.0" y="129.0" class="fig-small" text-anchor="middle">rotate (V^T)</text>
<line x1="296.0" y1="75.0" x2="316.0" y2="75.0" class="stroke-ink" marker-end="url(#c008-arrow)"/>
<line x1="318.0" y1="75.0" x2="442.0" y2="75.0" class="stroke-grid"/>
<line x1="380.0" y1="35.0" x2="380.0" y2="115.0" class="stroke-grid"/>
<path d="M358.6 67.9 L356.1 68.0 L353.7 68.2 L351.3 68.4 L349.0 68.7 L346.8 68.9 L344.7 69.2 L342.7 69.5 L340.8 69.8 L338.9 70.1 L337.2 70.4 L335.7 70.7 L334.2 71.1 L332.9 71.4 L331.7 71.8 L330.6 72.2 L329.6 72.6 L328.8 72.9 L328.2 73.3 L327.7 73.7 L327.3 74.1 L327.1 74.6 L327.0 75.0 L327.0 75.4 L327.2 75.8 L327.6 76.2 L328.1 76.6 L328.7 77.0 L329.5 77.4 L330.4 77.8 L331.5 78.1 L332.6 78.5 L334.0 78.9 L335.4 79.2 L337.0 79.6 L338.6 79.9 L340.4 80.2 L342.3 80.5 L344.3 80.8 L346.4 81.0 L348.6 81.3 L350.9 81.5 L353.3 81.7 L355.7 81.9 L358.2 82.1 L360.8 82.3 L363.4 82.4 L366.0 82.5 L368.7 82.6 L371.5 82.7 L374.2 82.8 L377.0 82.8 L379.8 82.8 L382.5 82.8 L385.3 82.8 L388.1 82.7 L390.8 82.6 L393.5 82.5 L396.2 82.4 L398.8 82.3 L401.4 82.1 L403.9 82.0 L406.3 81.8 L408.7 81.6 L411.0 81.3 L413.2 81.1 L415.3 80.8 L417.3 80.5 L419.2 80.2 L421.1 79.9 L422.8 79.6 L424.3 79.3 L425.8 78.9 L427.1 78.6 L428.3 78.2 L429.4 77.8 L430.4 77.4 L431.2 77.1 L431.8 76.7 L432.3 76.3 L432.7 75.9 L432.9 75.4 L433.0 75.0 L433.0 74.6 L432.8 74.2 L432.4 73.8 L431.9 73.4 L431.3 73.0 L430.5 72.6 L429.6 72.2 L428.5 71.9 L427.4 71.5 L426.0 71.1 L424.6 70.8 L423.0 70.4 L421.4 70.1 L419.6 69.8 L417.7 69.5 L415.7 69.2 L413.6 69.0 L411.4 68.7 L409.1 68.5 L406.7 68.3 L404.3 68.1 L401.8 67.9 L399.2 67.7 L396.6 67.6 L394.0 67.5 L391.3 67.4 L388.5 67.3 L385.8 67.2 L383.0 67.2 L380.2 67.2 L377.5 67.2 L374.7 67.2 L371.9 67.3 L369.2 67.4 L366.5 67.5 L363.8 67.6 L361.2 67.7 L358.6 67.9 Z" class="fill-soft-b stroke-b"/>
<line x1="380.0" y1="75.0" x2="358.6" y2="67.9" class="stroke-a"/>
<line x1="380.0" y1="75.0" x2="331.5" y2="78.1" class="stroke-c"/>
<text x="380.0" y="129.0" class="fig-small" text-anchor="middle">stretch (sigma)</text>
<line x1="446.0" y1="75.0" x2="466.0" y2="75.0" class="stroke-ink" marker-end="url(#c008-arrow)"/>
<line x1="468.0" y1="75.0" x2="592.0" y2="75.0" class="stroke-grid"/>
<line x1="530.0" y1="35.0" x2="530.0" y2="115.0" class="stroke-grid"/>
<path d="M552.5 75.0 L554.8 74.0 L557.1 73.1 L559.3 72.1 L561.4 71.2 L563.4 70.2 L565.3 69.3 L567.2 68.4 L568.9 67.5 L570.5 66.7 L572.0 65.8 L573.4 65.0 L574.7 64.2 L575.8 63.4 L576.9 62.7 L577.8 62.0 L578.5 61.3 L579.2 60.7 L579.7 60.1 L580.0 59.6 L580.3 59.1 L580.3 58.6 L580.3 58.2 L580.1 57.8 L579.8 57.5 L579.3 57.2 L578.7 57.0 L578.0 56.8 L577.1 56.7 L576.2 56.6 L575.0 56.6 L573.8 56.6 L572.4 56.7 L571.0 56.8 L569.4 57.0 L567.7 57.2 L565.9 57.5 L564.0 57.8 L562.0 58.2 L559.9 58.6 L557.7 59.1 L555.5 59.6 L553.2 60.1 L550.8 60.7 L548.4 61.3 L545.9 62.0 L543.4 62.7 L540.8 63.4 L538.3 64.2 L535.6 65.0 L533.0 65.8 L530.4 66.7 L527.7 67.5 L525.1 68.4 L522.5 69.3 L519.9 70.2 L517.3 71.2 L514.8 72.1 L512.3 73.1 L509.9 74.0 L507.5 75.0 L505.2 76.0 L502.9 76.9 L500.7 77.9 L498.6 78.8 L496.6 79.8 L494.7 80.7 L492.8 81.6 L491.1 82.5 L489.5 83.3 L488.0 84.2 L486.6 85.0 L485.3 85.8 L484.2 86.6 L483.1 87.3 L482.2 88.0 L481.5 88.7 L480.8 89.3 L480.3 89.9 L480.0 90.4 L479.7 90.9 L479.7 91.4 L479.7 91.8 L479.9 92.2 L480.2 92.5 L480.7 92.8 L481.3 93.0 L482.0 93.2 L482.9 93.3 L483.8 93.4 L485.0 93.4 L486.2 93.4 L487.6 93.3 L489.0 93.2 L490.6 93.0 L492.3 92.8 L494.1 92.5 L496.0 92.2 L498.0 91.8 L500.1 91.4 L502.3 90.9 L504.5 90.4 L506.8 89.9 L509.2 89.3 L511.6 88.7 L514.1 88.0 L516.6 87.3 L519.2 86.6 L521.7 85.8 L524.4 85.0 L527.0 84.2 L529.6 83.3 L532.3 82.5 L534.9 81.6 L537.5 80.7 L540.1 79.8 L542.7 78.8 L545.2 77.9 L547.7 76.9 L550.1 76.0 L552.5 75.0 Z" class="fill-soft-b stroke-b"/>
<line x1="530.0" y1="75.0" x2="552.5" y2="75.0" class="stroke-a"/>
<line x1="530.0" y1="75.0" x2="575.0" y2="56.6" class="stroke-c"/>
<text x="530.0" y="129.0" class="fig-small" text-anchor="middle">rotate (U)</text>
<text x="15.0" y="150.0" class="fig-bold">Solve: undo it, last move first</text>
<line x1="18.0" y1="210.0" x2="142.0" y2="210.0" class="stroke-grid"/>
<line x1="80.0" y1="170.0" x2="80.0" y2="250.0" class="stroke-grid"/>
<path d="M102.5 210.0 L104.8 209.0 L107.1 208.1 L109.3 207.1 L111.4 206.2 L113.4 205.2 L115.3 204.3 L117.2 203.4 L118.9 202.5 L120.5 201.7 L122.0 200.8 L123.4 200.0 L124.7 199.2 L125.8 198.4 L126.9 197.7 L127.8 197.0 L128.5 196.3 L129.2 195.7 L129.7 195.1 L130.0 194.6 L130.3 194.1 L130.3 193.6 L130.3 193.2 L130.1 192.8 L129.8 192.5 L129.3 192.2 L128.7 192.0 L128.0 191.8 L127.1 191.7 L126.2 191.6 L125.0 191.6 L123.8 191.6 L122.4 191.7 L121.0 191.8 L119.4 192.0 L117.7 192.2 L115.9 192.5 L114.0 192.8 L112.0 193.2 L109.9 193.6 L107.7 194.1 L105.5 194.6 L103.2 195.1 L100.8 195.7 L98.4 196.3 L95.9 197.0 L93.4 197.7 L90.8 198.4 L88.3 199.2 L85.6 200.0 L83.0 200.8 L80.4 201.7 L77.7 202.5 L75.1 203.4 L72.5 204.3 L69.9 205.2 L67.3 206.2 L64.8 207.1 L62.3 208.1 L59.9 209.0 L57.5 210.0 L55.2 211.0 L52.9 211.9 L50.7 212.9 L48.6 213.8 L46.6 214.8 L44.7 215.7 L42.8 216.6 L41.1 217.5 L39.5 218.3 L38.0 219.2 L36.6 220.0 L35.3 220.8 L34.2 221.6 L33.1 222.3 L32.2 223.0 L31.5 223.7 L30.8 224.3 L30.3 224.9 L30.0 225.4 L29.7 225.9 L29.7 226.4 L29.7 226.8 L29.9 227.2 L30.2 227.5 L30.7 227.8 L31.3 228.0 L32.0 228.2 L32.9 228.3 L33.8 228.4 L35.0 228.4 L36.2 228.4 L37.6 228.3 L39.0 228.2 L40.6 228.0 L42.3 227.8 L44.1 227.5 L46.0 227.2 L48.0 226.8 L50.1 226.4 L52.3 225.9 L54.5 225.4 L56.8 224.9 L59.2 224.3 L61.6 223.7 L64.1 223.0 L66.6 222.3 L69.2 221.6 L71.7 220.8 L74.4 220.0 L77.0 219.2 L79.6 218.3 L82.3 217.5 L84.9 216.6 L87.5 215.7 L90.1 214.8 L92.7 213.8 L95.2 212.9 L97.7 211.9 L100.1 211.0 L102.5 210.0 Z" class="fill-soft-b stroke-b"/>
<line x1="80.0" y1="210.0" x2="102.5" y2="210.0" class="stroke-a"/>
<line x1="80.0" y1="210.0" x2="125.0" y2="191.6" class="stroke-c"/>
<text x="80.0" y="264.0" class="fig-small" text-anchor="middle">the ellipse</text>
<line x1="146.0" y1="210.0" x2="166.0" y2="210.0" class="stroke-ink" marker-end="url(#c008-arrow)"/>
<line x1="168.0" y1="210.0" x2="292.0" y2="210.0" class="stroke-grid"/>
<line x1="230.0" y1="170.0" x2="230.0" y2="250.0" class="stroke-grid"/>
<path d="M208.6 202.9 L206.1 203.0 L203.7 203.2 L201.3 203.4 L199.0 203.7 L196.8 203.9 L194.7 204.2 L192.7 204.5 L190.8 204.8 L188.9 205.1 L187.2 205.4 L185.7 205.7 L184.2 206.1 L182.9 206.4 L181.7 206.8 L180.6 207.2 L179.6 207.6 L178.8 207.9 L178.2 208.3 L177.7 208.7 L177.3 209.1 L177.1 209.6 L177.0 210.0 L177.0 210.4 L177.2 210.8 L177.6 211.2 L178.1 211.6 L178.7 212.0 L179.5 212.4 L180.4 212.8 L181.5 213.1 L182.6 213.5 L184.0 213.9 L185.4 214.2 L187.0 214.6 L188.6 214.9 L190.4 215.2 L192.3 215.5 L194.3 215.8 L196.4 216.0 L198.6 216.3 L200.9 216.5 L203.3 216.7 L205.7 216.9 L208.2 217.1 L210.8 217.3 L213.4 217.4 L216.0 217.5 L218.7 217.6 L221.5 217.7 L224.2 217.8 L227.0 217.8 L229.8 217.8 L232.5 217.8 L235.3 217.8 L238.1 217.7 L240.8 217.6 L243.5 217.5 L246.2 217.4 L248.8 217.3 L251.4 217.1 L253.9 217.0 L256.3 216.8 L258.7 216.6 L261.0 216.3 L263.2 216.1 L265.3 215.8 L267.3 215.5 L269.2 215.2 L271.1 214.9 L272.8 214.6 L274.3 214.3 L275.8 213.9 L277.1 213.6 L278.3 213.2 L279.4 212.8 L280.4 212.4 L281.2 212.1 L281.8 211.7 L282.3 211.3 L282.7 210.9 L282.9 210.4 L283.0 210.0 L283.0 209.6 L282.8 209.2 L282.4 208.8 L281.9 208.4 L281.3 208.0 L280.5 207.6 L279.6 207.2 L278.5 206.9 L277.4 206.5 L276.0 206.1 L274.6 205.8 L273.0 205.4 L271.4 205.1 L269.6 204.8 L267.7 204.5 L265.7 204.2 L263.6 204.0 L261.4 203.7 L259.1 203.5 L256.7 203.3 L254.3 203.1 L251.8 202.9 L249.2 202.7 L246.6 202.6 L244.0 202.5 L241.3 202.4 L238.5 202.3 L235.8 202.2 L233.0 202.2 L230.2 202.2 L227.5 202.2 L224.7 202.2 L221.9 202.3 L219.2 202.4 L216.5 202.5 L213.8 202.6 L211.2 202.7 L208.6 202.9 Z" class="fill-soft-b stroke-b"/>
<line x1="230.0" y1="210.0" x2="208.6" y2="202.9" class="stroke-a"/>
<line x1="230.0" y1="210.0" x2="181.5" y2="213.1" class="stroke-c"/>
<text x="230.0" y="264.0" class="fig-small" text-anchor="middle">rotate back (U^T)</text>
<line x1="296.0" y1="210.0" x2="316.0" y2="210.0" class="stroke-ink" marker-end="url(#c008-arrow)"/>
<line x1="318.0" y1="210.0" x2="442.0" y2="210.0" class="stroke-grid"/>
<line x1="380.0" y1="170.0" x2="380.0" y2="250.0" class="stroke-grid"/>
<path d="M374.8 198.1 L374.1 198.4 L373.6 198.7 L373.0 199.1 L372.4 199.4 L371.9 199.9 L371.3 200.3 L370.8 200.8 L370.4 201.3 L369.9 201.8 L369.5 202.3 L369.1 202.9 L368.8 203.4 L368.4 204.0 L368.1 204.7 L367.9 205.3 L367.7 205.9 L367.5 206.6 L367.3 207.2 L367.2 207.9 L367.1 208.6 L367.0 209.3 L367.0 209.9 L367.0 210.6 L367.1 211.3 L367.2 212.0 L367.3 212.6 L367.4 213.3 L367.6 214.0 L367.8 214.6 L368.1 215.2 L368.4 215.9 L368.7 216.4 L369.1 217.0 L369.4 217.6 L369.9 218.1 L370.3 218.7 L370.8 219.2 L371.3 219.6 L371.8 220.1 L372.3 220.5 L372.9 220.9 L373.4 221.2 L374.0 221.6 L374.7 221.9 L375.3 222.1 L375.9 222.3 L376.6 222.5 L377.2 222.7 L377.9 222.8 L378.6 222.9 L379.3 223.0 L379.9 223.0 L380.6 223.0 L381.3 222.9 L382.0 222.8 L382.6 222.7 L383.3 222.6 L384.0 222.4 L384.6 222.2 L385.2 221.9 L385.9 221.6 L386.4 221.3 L387.0 220.9 L387.6 220.6 L388.1 220.1 L388.7 219.7 L389.2 219.2 L389.6 218.7 L390.1 218.2 L390.5 217.7 L390.9 217.1 L391.2 216.6 L391.6 216.0 L391.9 215.3 L392.1 214.7 L392.3 214.1 L392.5 213.4 L392.7 212.8 L392.8 212.1 L392.9 211.4 L393.0 210.7 L393.0 210.1 L393.0 209.4 L392.9 208.7 L392.8 208.0 L392.7 207.4 L392.6 206.7 L392.4 206.0 L392.2 205.4 L391.9 204.8 L391.6 204.1 L391.3 203.6 L390.9 203.0 L390.6 202.4 L390.1 201.9 L389.7 201.3 L389.2 200.8 L388.7 200.4 L388.2 199.9 L387.7 199.5 L387.1 199.1 L386.6 198.8 L386.0 198.4 L385.3 198.1 L384.7 197.9 L384.1 197.7 L383.4 197.5 L382.8 197.3 L382.1 197.2 L381.4 197.1 L380.7 197.0 L380.1 197.0 L379.4 197.0 L378.7 197.1 L378.0 197.2 L377.4 197.3 L376.7 197.4 L376.0 197.6 L375.4 197.8 L374.8 198.1 Z" class="fill-soft-b stroke-b"/>
<line x1="380.0" y1="210.0" x2="374.8" y2="198.1" class="stroke-a"/>
<line x1="380.0" y1="210.0" x2="368.1" y2="215.2" class="stroke-c"/>
<text x="380.0" y="264.0" class="fig-small" text-anchor="middle">divide by sigma</text>
<line x1="446.0" y1="210.0" x2="466.0" y2="210.0" class="stroke-ink" marker-end="url(#c008-arrow)"/>
<line x1="468.0" y1="210.0" x2="592.0" y2="210.0" class="stroke-grid"/>
<line x1="530.0" y1="170.0" x2="530.0" y2="250.0" class="stroke-grid"/>
<path d="M543.0 210.0 L543.0 209.3 L542.9 208.6 L542.8 208.0 L542.7 207.3 L542.6 206.6 L542.4 206.0 L542.1 205.3 L541.9 204.7 L541.6 204.1 L541.3 203.5 L540.9 202.9 L540.5 202.4 L540.1 201.8 L539.7 201.3 L539.2 200.8 L538.7 200.3 L538.2 199.9 L537.6 199.5 L537.1 199.1 L536.5 198.7 L535.9 198.4 L535.3 198.1 L534.7 197.9 L534.0 197.6 L533.4 197.4 L532.7 197.3 L532.0 197.2 L531.4 197.1 L530.7 197.0 L530.0 197.0 L529.3 197.0 L528.6 197.1 L528.0 197.2 L527.3 197.3 L526.6 197.4 L526.0 197.6 L525.3 197.9 L524.7 198.1 L524.1 198.4 L523.5 198.7 L522.9 199.1 L522.4 199.5 L521.8 199.9 L521.3 200.3 L520.8 200.8 L520.3 201.3 L519.9 201.8 L519.5 202.4 L519.1 202.9 L518.7 203.5 L518.4 204.1 L518.1 204.7 L517.9 205.3 L517.6 206.0 L517.4 206.6 L517.3 207.3 L517.2 208.0 L517.1 208.6 L517.0 209.3 L517.0 210.0 L517.0 210.7 L517.1 211.4 L517.2 212.0 L517.3 212.7 L517.4 213.4 L517.6 214.0 L517.9 214.7 L518.1 215.3 L518.4 215.9 L518.7 216.5 L519.1 217.1 L519.5 217.6 L519.9 218.2 L520.3 218.7 L520.8 219.2 L521.3 219.7 L521.8 220.1 L522.4 220.5 L522.9 220.9 L523.5 221.3 L524.1 221.6 L524.7 221.9 L525.3 222.1 L526.0 222.4 L526.6 222.6 L527.3 222.7 L528.0 222.8 L528.6 222.9 L529.3 223.0 L530.0 223.0 L530.7 223.0 L531.4 222.9 L532.0 222.8 L532.7 222.7 L533.4 222.6 L534.0 222.4 L534.7 222.1 L535.3 221.9 L535.9 221.6 L536.5 221.3 L537.1 220.9 L537.6 220.5 L538.2 220.1 L538.7 219.7 L539.2 219.2 L539.7 218.7 L540.1 218.2 L540.5 217.6 L540.9 217.1 L541.3 216.5 L541.6 215.9 L541.9 215.3 L542.1 214.7 L542.4 214.0 L542.6 213.4 L542.7 212.7 L542.8 212.0 L542.9 211.4 L543.0 210.7 L543.0 210.0 Z" class="fill-soft-b stroke-b"/>
<line x1="530.0" y1="210.0" x2="543.0" y2="210.0" class="stroke-a"/>
<line x1="530.0" y1="210.0" x2="530.0" y2="197.0" class="stroke-c"/>
<text x="530.0" y="264.0" class="fig-small" text-anchor="middle">rotate back (V)</text>
<path d="M57.0 330.0 L50.0 337.0 L40.0 337.0 L33.0 330.0 L33.0 320.0 L40.0 313.0 L50.0 313.0 L57.0 320.0 Z" class="fill-bad"/>
<text x="69.0" y="322.0" class="fig-small">If a stretch is near 0, dividing by it</text>
<text x="69.0" y="338.0" class="fig-small">blows that axis up. Cut it off instead.</text>
</svg>
Figure 2. Top: rotate, stretch, rotate turns the circle into the ellipse. Bottom: solving rotates back, divides each axis by its stretch, and rotates back. Notice that the divide step is the only one that can amplify anything, and it amplifies most along the axis with the smallest stretch.
```

In symbols the undo is

$$
w=V\,\Sigma^{-1}\,U^\top y
$$

For the houses, to four digits: $\sigma=(4.0791,\,0.6005)$ and $U^\top y=(-2.9619,\,0.2455)$. Divide each by its stretch to get $(-0.7261,\,0.4089)$. Then $V=\begin{bmatrix}-0.4027&0.9153\\-0.9153&-0.4027\end{bmatrix}$ rotates back, giving $w_1=(-0.4027)(-0.7261)+0.9153\cdot0.4089\approx0.6667$ and $w_2=(-0.9153)(-0.7261)+(-0.4027)(0.4089)\approx0.5000$. The signs of $U$ and $V$ can differ on your machine, in matching pairs, and the weights come out the same.

```python
U, s, vt = np.linalg.svd(X, full_matrices=False)
w = vt.T @ ((U.T @ y) / s)
```

> [!WARNING]
> numpy returns `vt`, which is already $V^\top$. You get $V$ from `vt.T`. Using `vt` as $V$ gives the wrong weights.

## Pick the shortest weights when a column repeats

Now the repeated column. Use the columns $(1,2,3)$ and $(2,4,6)$ with prices $y=(1,1,1)$. The second column is twice the first, so any weights $(a,b)$ give the guess $(a+2b)\,(1,2,3)$. The best guess is the projection of $y$ onto $(1,2,3)$, which is $t=6/14=3/7$ times $(1,2,3)$. So every pair with $a+2b=3/7$ fits equally well. That is the line in Figure 3.

```figure
<svg viewBox="0 0 600 285" role="img" aria-label="The plane of weights (a, b) for two repeated columns. Every point on the line a + 2b = 3/7 fits equally well. The point of the line nearest the origin, (3/35, 6/35), is the shortest weight vector. Dividing by a zero stretch sends the weights far away along the same line.">
<defs>
<marker id="c008-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-bad"/></marker>
</defs>
<line x1="70.0" y1="165.0" x2="430.0" y2="165.0" class="stroke-muted"/>
<line x1="150.0" y1="225.0" x2="150.0" y2="25.0" class="stroke-muted"/>
<text x="430.0" y="181.0" class="fig-small" text-anchor="end">a</text>
<text x="156.0" y="35.0" class="fig-small">b</text>
<text x="230.0" y="181.0" class="fig-small" text-anchor="middle">0.2</text>
<text x="310.0" y="181.0" class="fig-small" text-anchor="middle">0.4</text>
<text x="390.0" y="181.0" class="fig-small" text-anchor="middle">0.6</text>
<text x="144.0" y="209.0" class="fig-small" text-anchor="end">-0.1</text>
<text x="144.0" y="129.0" class="fig-small" text-anchor="end">0.1</text>
<text x="144.0" y="89.0" class="fig-small" text-anchor="end">0.2</text>
<text x="144.0" y="49.0" class="fig-small" text-anchor="end">0.3</text>
<line x1="90.0" y1="49.3" x2="430.0" y2="219.3" class="stroke-b"/>
<line x1="150.0" y1="165.0" x2="184.3" y2="96.4" class="stroke-c dashed"/>
<circle cx="184.3" cy="96.4" r="5.5" class="fill-c"/>
<circle cx="150.0" cy="165.0" r="3.5" class="fill-ink"/>
<path d="M180.7 103.6 L187.9 107.2 L191.4 100.0" class="stroke-ink"/>
<text x="196.3" y="84.4" class="fig-bold fig-small">(3/35, 6/35): shortest</text>
<text x="270.0" y="45.0" class="fig-small">every weight pair on this line,</text>
<text x="270.0" y="61.0" class="fig-small">a + 2b = 3/7, fits equally well</text>
<line x1="350.0" y1="179.3" x2="430.0" y2="219.3" class="stroke-bad" marker-end="url(#c008-arrow)"/>
<text x="334.0" y="211.0" class="fig-small" text-anchor="end">dividing by sigma2 = 0</text>
<text x="334.0" y="227.0" class="fig-small" text-anchor="end">runs off this way, about 1e15 away</text>

</svg>
Figure 3. The orange line holds every weight pair that fits equally well. The green point is the one nearest the origin, reached by a right angle from the origin. Notice that the red arrow, from dividing by a zero stretch, travels along the same line.
```

Which pair do you report? Moving along the line changes the weights but not the fit. That movement is the direction $(2,-1)$, which the matrix sends to zero. The set of all directions a matrix sends to zero is its *null space*. It adds length and no accuracy. So pick the point with none of it, the *minimum-norm* solution, the shortest weight vector. It is the foot of the perpendicular from the origin: $(3/7)/5\cdot(1,2)=(3/35,\,6/35)\approx(0.086,\,0.171)$.

The SVD finds it. The second singular value is 0 in exact arithmetic, and about $10^{-16}$ times the first on a computer. Dividing by it multiplies a bit of rounding noise by $10^{15}$ or more, and the weights run off along the line. The cure is a cut-off: treat any singular value at or below `rcond * s[0]` as zero, and drop that axis.

```python
keep = s > rcond * s[0]  # rcond is 1e-12 in the exercise
w = vt[keep].T @ ((U[:, keep].T @ y) / s[keep])
```

The kept singular values count the *rank*, the number of directions the columns really use. Here the rank is 1. This gives $(0.0857,\,0.1714)$, which is the shortest answer.

> [!KEY]
> Instead of forming $X^\top X$, split $X$ into pieces that do not stretch: QR gives a triangle to back-substitute, and the SVD gives a rotate, divide, rotate. Both keep the condition number at $\kappa$. When columns repeat, the SVD with a cut-off returns the shortest of the equally good weights.

> [!RECAP]
> - QR writes $X=QR$ with orthonormal $Q$ and triangular $R$. Solve $Rw=Q^\top y$ by back-substitution.
> - The SVD writes $X=U\Sigma V^\top$. Solve with $w=V\Sigma^{-1}U^\top y$, and drop singular values at or below `rcond * s[0]`.
> - Both avoid $X^\top X$, so they keep about twice as many digits as the normal equations.
> - A repeated column makes many weight vectors fit equally well. The cut-off picks the shortest one.

## Build it step by step

The checks call the functions in this order. They scan your source and reject `lstsq`, `pinv`, and `X.T @ X`.

1. **`fit_qr(X, y)`** returns `w` of shape `(d,)`:

   ```python
   Q, R = np.linalg.qr(X, mode="reduced")
   # solve_triangular(R, Q.T @ y, lower=False)
   ```

   Full mode gives `Q` shape `(n, n)`, and then the product no longer lines up, so ask for `mode="reduced"`.

2. **`fit_svd(X, y, rcond)`** returns the minimum-norm `w` of shape `(d,)`:

   ```python
   U, s, vt = np.linalg.svd(X, full_matrices=False)
   keep = s > rcond * s[0]
   # vt[keep].T times (U[:, keep].T @ y) / s[keep]
   ```

   Dividing by every singular value, including the near-zero one, makes the weights huge.

3. **`rank(X, rcond)`** returns an integer: how many singular values exceed `rcond * s[0]`:

   ```python
   s = np.linalg.svd(X, compute_uv=False)
   # count the entries of s above rcond * s[0]
   ```

   An all-zero matrix has `s[0] == 0`, so it must return 0.

## Cost and scaling

When the table has many more rows $n$ than columns $d$, all three routes cost about $nd^2$ operations, which means the costs are of the same order. The following table compares them:

|Method|Rough work|Digits lost|Repeated columns|
|---|---|---|---|
|Normal equations|$nd^2$ then $d^3$|about $2k$|No, Cholesky fails|
|QR|about $2nd^2$|about $k$|No, a zero on the diagonal of $R$|
|SVD|a few times $nd^2$|about $k$|Yes|

Doubling $n$ doubles the time. Doubling $d$ at least quadruples it, and more once the $d^3$ term matters.

## Where this breaks

**More columns than rows.** Many weight vectors fit the training rows exactly, and the shortest one can still miss unseen rows. Later chapters on penalties address this.

**Choosing the cut-off.** If `rcond` is too small, a singular value that is really noise survives, and the weights blow up. If it is too large, a real direction is thrown away and the fit gets worse. The default of $10^{-12}$ suits well-scaled tables. Look at the singular values for a sudden drop of many orders of magnitude, and put the cut-off inside that gap.

**Near-repeats that are not exact.** A singular value of $10^{-6}$ is a real signal for the SVD, so it is kept, and dividing by it still amplifies noise a millionfold. The cut-off helps only when the gap is wide. The chapter Ridge: a fee on large weights tackles the gradual case.

## The exercise

You implement `fit_qr`, `fit_svd`, and `rank`. The checks confirm that both solvers return $(2/3,1/2)$ for the three houses and agree with a reference solve on random tables. They also confirm that `fit_svd` returns $(3/35,6/35)$ for the repeated columns and has no part in the null space on a larger rank-deficient table, and that `rank` counts correctly. At $\kappa=10^7$, QR and the SVD must keep about nine digits. The checks scan the source of each function and reject `lstsq`, `pinv`, and `X.T @ X`.
