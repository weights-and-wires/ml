---
title: "Least squares as a bowl and a right angle"
slug: 003-least-squares-and-projection
part: "Part 1 — Foundations"
summary: Find the best line by setting the slope of the squared-miss loss to zero, then check any fit with one right-angle test.
minutes: 70
gpu: false
objectives:
  - Build the table X, with a column of ones, and write the loss as half the sum of squared misses.
  - Compute the slope of the loss, X transpose times the misses, and check it against finite differences.
  - Derive the normal equations by setting both slopes to zero, and solve them.
  - Explain why dot products of the leftover errors with every column equal to zero mean that no nudge can lower the loss.
  - Describe the fitted values as the shadow of the targets on the sheet the columns span, and turn the check into one number.
prereqs:
  - 002-curves-and-bias-variance
exercise: 003-least-squares-and-projection
---

# Least squares as a bowl and a right angle

You have three houses. Sizes are 1, 2, and 3 in units of 100 square metres, and the prices are 1, 2, and 2 lakhs. You want one straight line that predicts price from size, so you can price a fourth house you have not seen.

The trouble is that no straight line passes through all three points. Every line you draw misses at least one house. Without a rule for what "best" means, two people can draw two different lines and both can claim to be right. Guessing weights by hand does not scale either, because a real model has dozens of weights and you cannot try them all. In the chapter on [fitting a curve](/lessons/002-curves-and-bias-variance) you handed a table of columns to `np.linalg.lstsq`. Here you open that solver and see where its answer comes from.

A second problem follows. A colleague sends you weights and says they are the best line. On a problem with thousands of rows, you want a test you can run on any weights, from any source, that says "nobody can do better" without a plot and without a second fit. This chapter gives you the rule for "best", a picture of why it has exactly one answer, a short calculation that finds it, and a single number that checks it.

## What you already know

- A loss is a number that says how bad one guess is, and the [empirical risk](/lessons/001-loss-and-honest-scores) averages the losses of many rows into one score.
- A weight is a number the model multiplies by an input. You choose weights, and the data judge them.
- A line has a slope and a starting height, so picking a line means picking two weights.
- The transpose of a matrix swaps its rows and columns: the transpose of `[[1, 2], [3, 4]]` is `[[1, 3], [2, 4]]`.
- A dot product multiplies two vectors entry by entry and adds the results. A slope tells you which way a curve rises, and at the bottom of a smooth bowl the slope is 0.

## Measure a line by its misses

Look at Figure 1. The left panel draws the line that predicts price equal to size. It passes through the first two houses and misses the third by 1, because it predicts 3 and the house costs 2. The right panel draws a different line, one that touches none of the houses and stays close to all of them.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Three points with two candidate lines. Left: the line y equals x, with squares drawn on the misses. Right: the best line, with much smaller squares.">
<defs></defs>
<line x1="30.0" y1="220.0" x2="275.0" y2="220.0" class="stroke-muted"/>
<line x1="30.0" y1="220.0" x2="30.0" y2="50.0" class="stroke-muted"/>
<line x1="91.2" y1="220.0" x2="91.2" y2="220.0" class="stroke-muted"/>
<text x="91.2" y="236.0" class="fig-small" text-anchor="middle">1</text>
<line x1="152.5" y1="220.0" x2="152.5" y2="220.0" class="stroke-muted"/>
<text x="152.5" y="236.0" class="fig-small" text-anchor="middle">2</text>
<line x1="213.8" y1="220.0" x2="213.8" y2="220.0" class="stroke-muted"/>
<text x="213.8" y="236.0" class="fig-small" text-anchor="middle">3</text>
<text x="16.0" y="167.3" class="fig-small" text-anchor="middle">1</text>
<text x="16.0" y="110.7" class="fig-small" text-anchor="middle">2</text>
<text x="30.0" y="26.0" class="fig-bold">Line y = x</text>
<text x="30.0" y="256.0" class="fig-small">squares add to 1, loss 0.5</text>
<text x="275.0" y="236.0" class="fig-small" text-anchor="end">size</text>
<text x="36.0" y="52.0" class="fig-small">price</text>
<rect x="213.8" y="50.0" width="56.7" height="56.7" class="fill-soft-b stroke-b"/>
<line x1="213.8" y1="106.7" x2="213.8" y2="50.0" class="stroke-bad dashed"/>
<path d="M30.0 220.0 L275.0 -6.7" class="stroke-b"/>
<circle cx="91.2" cy="163.3" r="5" class="fill-a"/>
<circle cx="152.5" cy="106.7" r="5" class="fill-a"/>
<circle cx="213.8" cy="106.7" r="5" class="fill-a"/>
<line x1="325.0" y1="220.0" x2="570.0" y2="220.0" class="stroke-muted"/>
<line x1="325.0" y1="220.0" x2="325.0" y2="50.0" class="stroke-muted"/>
<line x1="386.2" y1="220.0" x2="386.2" y2="220.0" class="stroke-muted"/>
<text x="386.2" y="236.0" class="fig-small" text-anchor="middle">1</text>
<line x1="447.5" y1="220.0" x2="447.5" y2="220.0" class="stroke-muted"/>
<text x="447.5" y="236.0" class="fig-small" text-anchor="middle">2</text>
<line x1="508.8" y1="220.0" x2="508.8" y2="220.0" class="stroke-muted"/>
<text x="508.8" y="236.0" class="fig-small" text-anchor="middle">3</text>
<text x="311.0" y="167.3" class="fig-small" text-anchor="middle">1</text>
<text x="311.0" y="110.7" class="fig-small" text-anchor="middle">2</text>
<text x="325.0" y="26.0" class="fig-bold">Best line y = 2/3 + x/2</text>
<text x="325.0" y="256.0" class="fig-small">squares add to 1/6, loss 0.083</text>
<text x="570.0" y="236.0" class="fig-small" text-anchor="end">size</text>
<text x="331.0" y="52.0" class="fig-small">price</text>
<rect x="386.2" y="153.9" width="9.4" height="9.4" class="fill-soft-b stroke-b"/>
<line x1="386.2" y1="163.3" x2="386.2" y2="153.9" class="stroke-bad dashed"/>
<rect x="447.5" y="106.7" width="18.9" height="18.9" class="fill-soft-b stroke-b"/>
<line x1="447.5" y1="106.7" x2="447.5" y2="125.6" class="stroke-bad dashed"/>
<rect x="508.8" y="97.2" width="9.4" height="9.4" class="fill-soft-b stroke-b"/>
<line x1="508.8" y1="106.7" x2="508.8" y2="97.2" class="stroke-bad dashed"/>
<path d="M325.0 182.2 L570.0 68.9" class="stroke-b"/>
<circle cx="386.2" cy="163.3" r="5" class="fill-a"/>
<circle cx="447.5" cy="106.7" r="5" class="fill-a"/>
<circle cx="508.8" cy="106.7" r="5" class="fill-a"/>
</svg>
Figure 1. Left: the line y = x hits two points and misses one by a full unit. Right: the best line touches none of the points, yet its squares are far smaller. Notice that the dashed misses are vertical, because price is the thing you predict.
```

A *miss* is the guess minus the true price, measured straight up or down. You need one number for the whole line, so you combine the misses. Adding them as they are fails, because a miss of +1 and a miss of −1 cancel and a terrible line scores 0. You can add the absolute values, or you can add the squares.

Squares win for two reasons. First, a miss of 2 costs 4 while a miss of 1 costs 1, so one big error hurts more than two small ones, which is usually what you want. Second, squares make a smooth bowl with a bottom you can find with a slope calculation. Absolute values make a curve with a corner at the bottom, where no single slope exists, so you lose the clean equation this chapter ends with. That is the practical reason squares are the default.

## Turn a line into a table

A line says price equals a starting height plus a slope times the size. Call the starting height $w_1$ and the slope $w_2$. The guess for a house of size $x$ is $w_1\cdot 1 + w_2\cdot x$, a weighted sum of two ingredients: the number 1 and the size.

So give every house a row with the ingredients, a 1 and its size. Stack the rows into a table $X$. The 1s are there so that $w_1$ has something to multiply, which is how a starting height, called the *intercept*, fits into the same weighted sum as everything else:

$$
X=\begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix},\qquad y=\begin{bmatrix}1\\2\\2\end{bmatrix}
$$

The guesses for all houses at once are $Xw$. The line $y = x$ has $w=(0,1)$, so the guesses are 1, 2, and 3, and the misses $Xw-y$ are $(0,0,1)$. In numpy this is one line:

```python
guesses = X @ w  # shape (n,): one guess per house
```

## Write the loss

Add the squared misses, and put a half in front:

$$
L(w)=\tfrac12\,\lVert Xw-y\rVert^2
$$

Read it aloud: take the vector of misses, square each entry, add them up, and halve. The symbol $\lVert v\rVert$ is the length of the vector $v$, so $\lVert v\rVert^2$ is the sum of squared entries. For $w=(0,1)$ the misses are $(0,0,1)$, the squares add to 1, and $L=0.5$. For the best line, $w=(2/3,1/2)$, the misses are $(1/6,-1/3,1/6)$, the squares add to $1/36+4/36+1/36=1/6$, and $L=1/12\approx0.083$.

Why the half? A square has a derivative with a 2 in front, and the half cancels it, so the slope you meet in a moment has no stray 2. Why not divide by the number of rows, as the empirical risk does? You can. Dividing by a constant scales the whole bowl, but the bottom stays at the same weights.

```python
loss = 0.5 * np.sum((X @ w - y) ** 2)  # one float
```

## See the loss as a bowl

Every pair of weights $(w_1,w_2)$ has a loss. Plot the loss as a height over the plane of weights and you get a bowl. Figure 2 shows its contour map, like a hiking map, where each ring joins weights with the same loss.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Contour map of the loss over two weights, with the starting point (0, 1), the slope arrow there, and the minimum at (2/3, 1/2).">
<defs>
<marker id="c003-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-bad"/></marker>
<clipPath id="c003-clip"><rect x="50.0" y="20.0" width="360.0" height="250.0"/></clipPath>
</defs>
<rect x="50" y="20" width="360" height="250" class="fill-sunk"/>
<g clip-path="url(#c003-clip)">
<path d="M188.8 147.8 L188.9 147.6 L189.0 147.5 L189.2 147.4 L189.4 147.3 L189.6 147.2 L189.8 147.2 L190.1 147.1 L190.4 147.1 L190.8 147.1 L191.1 147.1 L191.5 147.1 L192.0 147.1 L192.4 147.2 L192.9 147.3 L193.5 147.3 L194.0 147.4 L194.6 147.6 L195.2 147.7 L195.8 147.8 L196.5 148.0 L197.2 148.2 L197.9 148.3 L198.6 148.5 L199.4 148.8 L200.2 149.0 L201.0 149.2 L201.8 149.5 L202.6 149.8 L203.4 150.0 L204.3 150.3 L205.2 150.6 L206.1 151.0 L207.0 151.3 L207.9 151.6 L208.8 152.0 L209.8 152.3 L210.7 152.7 L211.7 153.1 L212.6 153.4 L213.6 153.8 L214.6 154.2 L215.5 154.6 L216.5 155.1 L217.5 155.5 L218.5 155.9 L219.4 156.3 L220.4 156.8 L221.4 157.2 L222.3 157.6 L223.3 158.1 L224.2 158.5 L225.1 159.0 L226.1 159.4 L227.0 159.9 L227.9 160.3 L228.8 160.8 L229.6 161.2 L230.5 161.7 L231.3 162.1 L232.1 162.6 L232.9 163.0 L233.7 163.4 L234.5 163.9 L235.2 164.3 L235.9 164.7 L236.6 165.1 L237.3 165.6 L237.9 166.0 L238.6 166.4 L239.1 166.7 L239.7 167.1 L240.2 167.5 L240.7 167.9 L241.2 168.2 L241.7 168.6 L242.1 168.9 L242.5 169.2 L242.8 169.5 L243.1 169.8 L243.4 170.1 L243.7 170.4 L243.9 170.6 L244.1 170.9 L244.3 171.1 L244.4 171.3 L244.5 171.6 L244.5 171.7 L244.6 171.9 L244.5 172.1 L244.5 172.2 L244.4 172.4 L244.3 172.5 L244.2 172.6 L244.0 172.7 L243.8 172.8 L243.5 172.8 L243.2 172.9 L242.9 172.9 L242.6 172.9 L242.2 172.9 L241.8 172.9 L241.4 172.9 L240.9 172.8 L240.4 172.7 L239.9 172.7 L239.3 172.6 L238.7 172.4 L238.1 172.3 L237.5 172.2 L236.8 172.0 L236.1 171.8 L235.4 171.7 L234.7 171.5 L233.9 171.2 L233.2 171.0 L232.4 170.8 L231.6 170.5 L230.7 170.2 L229.9 170.0 L229.0 169.7 L228.1 169.4 L227.2 169.0 L226.3 168.7 L225.4 168.4 L224.5 168.0 L223.6 167.7 L222.6 167.3 L221.7 166.9 L220.7 166.6 L219.7 166.2 L218.8 165.8 L217.8 165.4 L216.8 164.9 L215.8 164.5 L214.9 164.1 L213.9 163.7 L212.9 163.2 L212.0 162.8 L211.0 162.4 L210.1 161.9 L209.1 161.5 L208.2 161.0 L207.3 160.6 L206.4 160.1 L205.5 159.7 L204.6 159.2 L203.7 158.8 L202.8 158.3 L202.0 157.9 L201.2 157.4 L200.4 157.0 L199.6 156.6 L198.9 156.1 L198.1 155.7 L197.4 155.3 L196.7 154.9 L196.0 154.4 L195.4 154.0 L194.8 153.6 L194.2 153.3 L193.6 152.9 L193.1 152.5 L192.6 152.1 L192.1 151.8 L191.7 151.4 L191.2 151.1 L190.9 150.8 L190.5 150.5 L190.2 150.2 L189.9 149.9 L189.6 149.6 L189.4 149.4 L189.2 149.1 L189.1 148.9 L188.9 148.7 L188.9 148.4 L188.8 148.3 L188.8 148.1 L188.8 147.9 L188.8 147.8" class="stroke-a"/>
<path d="M128.7 121.3 L128.9 120.9 L129.3 120.5 L129.7 120.1 L130.3 119.9 L131.0 119.6 L131.8 119.4 L132.7 119.3 L133.6 119.2 L134.7 119.2 L135.9 119.2 L137.2 119.3 L138.6 119.4 L140.1 119.5 L141.6 119.7 L143.3 120.0 L145.1 120.3 L146.9 120.7 L148.8 121.1 L150.8 121.5 L152.9 122.0 L155.1 122.6 L157.3 123.2 L159.6 123.8 L162.0 124.5 L164.5 125.2 L167.0 126.0 L169.5 126.8 L172.2 127.6 L174.9 128.5 L177.6 129.4 L180.4 130.4 L183.2 131.4 L186.1 132.4 L189.0 133.5 L191.9 134.6 L194.9 135.7 L197.9 136.9 L200.9 138.1 L203.9 139.3 L207.0 140.5 L210.1 141.8 L213.1 143.1 L216.2 144.4 L219.3 145.7 L222.4 147.0 L225.4 148.4 L228.5 149.8 L231.5 151.2 L234.6 152.6 L237.6 154.0 L240.5 155.4 L243.5 156.8 L246.4 158.2 L249.3 159.6 L252.1 161.1 L254.9 162.5 L257.7 163.9 L260.4 165.3 L263.0 166.7 L265.6 168.1 L268.1 169.5 L270.6 170.9 L273.0 172.3 L275.3 173.6 L277.6 175.0 L279.8 176.3 L281.9 177.6 L283.9 178.8 L285.9 180.1 L287.7 181.3 L289.5 182.5 L291.2 183.7 L292.8 184.9 L294.3 186.0 L295.7 187.1 L297.0 188.1 L298.3 189.1 L299.4 190.1 L300.4 191.0 L301.3 191.9 L302.1 192.8 L302.8 193.6 L303.4 194.4 L303.9 195.2 L304.3 195.9 L304.6 196.5 L304.8 197.1 L304.9 197.7 L304.8 198.2 L304.7 198.7 L304.4 199.1 L304.1 199.5 L303.6 199.9 L303.0 200.1 L302.3 200.4 L301.6 200.6 L300.7 200.7 L299.7 200.8 L298.6 200.8 L297.4 200.8 L296.1 200.7 L294.7 200.6 L293.3 200.5 L291.7 200.3 L290.0 200.0 L288.3 199.7 L286.4 199.3 L284.5 198.9 L282.5 198.5 L280.4 198.0 L278.3 197.4 L276.0 196.8 L273.7 196.2 L271.3 195.5 L268.9 194.8 L266.4 194.0 L263.8 193.2 L261.2 192.4 L258.5 191.5 L255.7 190.6 L253.0 189.6 L250.1 188.6 L247.3 187.6 L244.4 186.5 L241.4 185.4 L238.4 184.3 L235.4 183.1 L232.4 181.9 L229.4 180.7 L226.3 179.5 L223.3 178.2 L220.2 176.9 L217.1 175.6 L214.0 174.3 L211.0 173.0 L207.9 171.6 L204.8 170.2 L201.8 168.8 L198.8 167.4 L195.8 166.0 L192.8 164.6 L189.8 163.2 L186.9 161.8 L184.1 160.4 L181.2 158.9 L178.4 157.5 L175.7 156.1 L173.0 154.7 L170.3 153.3 L167.7 151.9 L165.2 150.5 L162.7 149.1 L160.3 147.7 L158.0 146.4 L155.7 145.0 L153.5 143.7 L151.4 142.4 L149.4 141.2 L147.5 139.9 L145.6 138.7 L143.8 137.5 L142.1 136.3 L140.5 135.1 L139.0 134.0 L137.6 132.9 L136.3 131.9 L135.1 130.9 L133.9 129.9 L132.9 129.0 L132.0 128.1 L131.2 127.2 L130.5 126.4 L129.9 125.6 L129.4 124.8 L129.0 124.1 L128.7 123.5 L128.5 122.9 L128.5 122.3 L128.5 121.8 L128.7 121.3" class="stroke-a"/>
<path d="M77.5 98.8 L77.9 98.1 L78.5 97.5 L79.2 97.0 L80.1 96.5 L81.2 96.2 L82.4 95.9 L83.8 95.6 L85.4 95.5 L87.1 95.5 L89.0 95.5 L91.0 95.6 L93.2 95.7 L95.5 96.0 L98.0 96.3 L100.7 96.7 L103.4 97.2 L106.3 97.8 L109.4 98.4 L112.6 99.2 L115.9 99.9 L119.3 100.8 L122.8 101.7 L126.5 102.7 L130.3 103.8 L134.1 105.0 L138.1 106.2 L142.2 107.4 L146.3 108.8 L150.6 110.2 L154.9 111.7 L159.3 113.2 L163.8 114.8 L168.3 116.4 L172.9 118.1 L177.5 119.8 L182.2 121.6 L187.0 123.4 L191.7 125.3 L196.5 127.2 L201.4 129.2 L206.2 131.2 L211.1 133.2 L215.9 135.3 L220.8 137.4 L225.7 139.5 L230.5 141.7 L235.4 143.8 L240.2 146.0 L245.0 148.2 L249.7 150.5 L254.4 152.7 L259.1 154.9 L263.7 157.2 L268.2 159.4 L272.7 161.7 L277.1 163.9 L281.5 166.2 L285.8 168.4 L290.0 170.7 L294.0 172.9 L298.0 175.1 L302.0 177.2 L305.7 179.4 L309.4 181.5 L313.0 183.7 L316.5 185.7 L319.8 187.8 L323.0 189.8 L326.1 191.8 L329.1 193.7 L331.9 195.6 L334.5 197.5 L337.1 199.3 L339.4 201.1 L341.7 202.8 L343.8 204.4 L345.7 206.0 L347.5 207.6 L349.1 209.1 L350.5 210.5 L351.8 211.9 L352.9 213.2 L353.9 214.4 L354.7 215.6 L355.3 216.7 L355.7 217.8 L356.0 218.7 L356.1 219.6 L356.0 220.5 L355.8 221.2 L355.4 221.9 L354.9 222.5 L354.1 223.0 L353.2 223.5 L352.1 223.8 L350.9 224.1 L349.5 224.4 L347.9 224.5 L346.2 224.5 L344.3 224.5 L342.3 224.4 L340.1 224.3 L337.8 224.0 L335.3 223.7 L332.7 223.3 L329.9 222.8 L327.0 222.2 L323.9 221.6 L320.8 220.8 L317.5 220.1 L314.0 219.2 L310.5 218.3 L306.8 217.3 L303.1 216.2 L299.2 215.0 L295.2 213.8 L291.2 212.6 L287.0 211.2 L282.8 209.8 L278.4 208.3 L274.0 206.8 L269.6 205.2 L265.0 203.6 L260.4 201.9 L255.8 200.2 L251.1 198.4 L246.4 196.6 L241.6 194.7 L236.8 192.8 L232.0 190.8 L227.1 188.8 L222.3 186.8 L217.4 184.7 L212.5 182.6 L207.7 180.5 L202.8 178.3 L198.0 176.2 L193.2 174.0 L188.4 171.8 L183.6 169.5 L178.9 167.3 L174.3 165.1 L169.6 162.8 L165.1 160.6 L160.6 158.3 L156.2 156.1 L151.8 153.8 L147.6 151.6 L143.4 149.3 L139.3 147.1 L135.3 144.9 L131.4 142.8 L127.6 140.6 L123.9 138.5 L120.3 136.3 L116.9 134.3 L113.5 132.2 L110.3 130.2 L107.2 128.2 L104.3 126.3 L101.5 124.4 L98.8 122.5 L96.3 120.7 L93.9 118.9 L91.7 117.2 L89.6 115.6 L87.6 114.0 L85.9 112.4 L84.3 110.9 L82.8 109.5 L81.5 108.1 L80.4 106.8 L79.5 105.6 L78.7 104.4 L78.1 103.3 L77.6 102.2 L77.3 101.3 L77.2 100.4 L77.3 99.5 L77.5 98.8" class="stroke-a"/>
<path d="M10.3 69.2 L10.9 68.2 L11.7 67.3 L12.8 66.5 L14.1 65.9 L15.7 65.3 L17.6 64.9 L19.6 64.6 L22.0 64.3 L24.5 64.3 L27.3 64.3 L30.3 64.4 L33.6 64.7 L37.0 65.1 L40.7 65.6 L44.6 66.2 L48.7 66.9 L53.0 67.7 L57.5 68.7 L62.3 69.7 L67.1 70.9 L72.2 72.2 L77.5 73.6 L82.9 75.1 L88.5 76.7 L94.2 78.4 L100.1 80.2 L106.2 82.1 L112.3 84.0 L118.6 86.1 L125.0 88.3 L131.6 90.5 L138.2 92.9 L144.9 95.3 L151.7 97.8 L158.6 100.4 L165.6 103.0 L172.6 105.8 L179.7 108.6 L186.8 111.4 L194.0 114.3 L201.2 117.3 L208.4 120.3 L215.6 123.4 L222.8 126.5 L230.0 129.6 L237.2 132.8 L244.4 136.0 L251.5 139.3 L258.6 142.5 L265.7 145.8 L272.7 149.2 L279.6 152.5 L286.4 155.8 L293.2 159.2 L299.8 162.5 L306.4 165.8 L312.8 169.2 L319.2 172.5 L325.4 175.8 L331.4 179.1 L337.4 182.3 L343.2 185.6 L348.8 188.8 L354.3 192.0 L359.6 195.1 L364.7 198.2 L369.7 201.2 L374.4 204.2 L379.0 207.1 L383.4 210.0 L387.5 212.8 L391.5 215.6 L395.3 218.3 L398.8 220.9 L402.1 223.4 L405.2 225.9 L408.0 228.3 L410.7 230.6 L413.0 232.8 L415.2 234.9 L417.1 237.0 L418.8 238.9 L420.2 240.7 L421.3 242.5 L422.2 244.1 L422.9 245.7 L423.3 247.1 L423.5 248.5 L423.4 249.7 L423.1 250.8 L422.5 251.8 L421.6 252.7 L420.5 253.5 L419.2 254.1 L417.6 254.7 L415.8 255.1 L413.7 255.4 L411.4 255.7 L408.8 255.7 L406.0 255.7 L403.0 255.6 L399.8 255.3 L396.3 254.9 L392.6 254.4 L388.7 253.8 L384.6 253.1 L380.3 252.3 L375.8 251.3 L371.1 250.3 L366.2 249.1 L361.1 247.8 L355.9 246.4 L350.4 244.9 L344.8 243.3 L339.1 241.6 L333.2 239.8 L327.2 237.9 L321.0 236.0 L314.7 233.9 L308.3 231.7 L301.8 229.5 L295.1 227.1 L288.4 224.7 L281.6 222.2 L274.7 219.6 L267.7 217.0 L260.7 214.2 L253.6 211.4 L246.5 208.6 L239.3 205.7 L232.2 202.7 L224.9 199.7 L217.7 196.6 L210.5 193.5 L203.3 190.4 L196.1 187.2 L188.9 184.0 L181.8 180.7 L174.7 177.5 L167.7 174.2 L160.7 170.8 L153.8 167.5 L146.9 164.2 L140.2 160.8 L133.5 157.5 L127.0 154.2 L120.5 150.8 L114.2 147.5 L108.0 144.2 L101.9 140.9 L96.0 137.7 L90.2 134.4 L84.5 131.2 L79.1 128.0 L73.8 124.9 L68.6 121.8 L63.7 118.8 L58.9 115.8 L54.3 112.9 L50.0 110.0 L45.8 107.2 L41.8 104.4 L38.1 101.7 L34.6 99.1 L31.2 96.6 L28.2 94.1 L25.3 91.7 L22.7 89.4 L20.3 87.2 L18.1 85.1 L16.2 83.0 L14.6 81.1 L13.2 79.3 L12.0 77.5 L11.1 75.9 L10.4 74.3 L10.0 72.9 L9.8 71.5 L9.9 70.3 L10.3 69.2" class="stroke-a"/>
<path d="M-81.8 28.7 L-80.9 27.3 L-79.7 26.0 L-78.1 24.8 L-76.2 23.9 L-73.9 23.1 L-71.2 22.4 L-68.2 22.0 L-64.9 21.7 L-61.2 21.6 L-57.2 21.6 L-52.8 21.8 L-48.1 22.2 L-43.1 22.7 L-37.8 23.5 L-32.1 24.3 L-26.2 25.4 L-20.0 26.6 L-13.4 28.0 L-6.6 29.5 L0.5 31.2 L7.8 33.0 L15.4 35.0 L23.3 37.2 L31.3 39.5 L39.6 42.0 L48.2 44.6 L56.9 47.3 L65.8 50.2 L74.9 53.2 L84.2 56.3 L93.6 59.6 L103.2 63.0 L112.9 66.5 L122.8 70.1 L132.8 73.8 L142.8 77.6 L153.0 81.6 L163.2 85.6 L173.5 89.7 L183.9 93.9 L194.3 98.2 L204.7 102.6 L215.1 107.0 L225.6 111.5 L236.0 116.1 L246.4 120.7 L256.8 125.3 L267.1 130.0 L277.3 134.8 L287.5 139.5 L297.6 144.3 L307.6 149.1 L317.5 154.0 L327.3 158.8 L336.9 163.6 L346.4 168.4 L355.7 173.3 L364.9 178.1 L373.8 182.8 L382.6 187.6 L391.2 192.3 L399.6 197.0 L407.7 201.6 L415.6 206.2 L423.3 210.7 L430.7 215.2 L437.9 219.6 L444.8 223.9 L451.4 228.2 L457.7 232.3 L463.7 236.4 L469.5 240.4 L474.9 244.3 L480.0 248.1 L484.8 251.7 L489.3 255.3 L493.4 258.7 L497.2 262.1 L500.6 265.3 L503.7 268.3 L506.5 271.3 L508.9 274.1 L510.9 276.8 L512.6 279.3 L513.9 281.7 L514.9 283.9 L515.5 286.0 L515.7 287.9 L515.6 289.7 L515.1 291.3 L514.3 292.7 L513.0 294.0 L511.5 295.2 L509.5 296.1 L507.2 296.9 L504.6 297.6 L501.6 298.0 L498.2 298.3 L494.5 298.4 L490.5 298.4 L486.1 298.2 L481.5 297.8 L476.4 297.3 L471.1 296.5 L465.5 295.7 L459.5 294.6 L453.3 293.4 L446.8 292.0 L439.9 290.5 L432.9 288.8 L425.5 287.0 L417.9 285.0 L410.1 282.8 L402.0 280.5 L393.7 278.0 L385.2 275.4 L376.5 272.7 L367.5 269.8 L358.4 266.8 L349.2 263.7 L339.7 260.4 L330.1 257.0 L320.4 253.5 L310.5 249.9 L300.6 246.2 L290.5 242.4 L280.4 238.4 L270.1 234.4 L259.8 230.3 L249.5 226.1 L239.1 221.8 L228.6 217.4 L218.2 213.0 L207.8 208.5 L197.3 203.9 L186.9 199.3 L176.6 194.7 L166.2 190.0 L156.0 185.2 L145.8 180.5 L135.7 175.7 L125.7 170.9 L115.8 166.0 L106.1 161.2 L96.4 156.4 L86.9 151.6 L77.6 146.7 L68.5 141.9 L59.5 137.2 L50.7 132.4 L42.1 127.7 L33.8 123.0 L25.6 118.4 L17.7 113.8 L10.0 109.3 L2.6 104.8 L-4.6 100.4 L-11.4 96.1 L-18.1 91.8 L-24.4 87.7 L-30.4 83.6 L-36.1 79.6 L-41.6 75.7 L-46.7 71.9 L-51.5 68.3 L-55.9 64.7 L-60.0 61.3 L-63.8 57.9 L-67.3 54.7 L-70.4 51.7 L-73.1 48.7 L-75.5 45.9 L-77.6 43.2 L-79.3 40.7 L-80.6 38.3 L-81.6 36.1 L-82.2 34.0 L-82.4 32.1 L-82.3 30.3 L-81.8 28.7" class="stroke-a"/>
<path d="M-210.0 -27.7 L-208.7 -29.8 L-207.0 -31.6 L-204.7 -33.2 L-202.0 -34.6 L-198.7 -35.7 L-194.9 -36.6 L-190.6 -37.3 L-185.8 -37.7 L-180.5 -37.9 L-174.8 -37.8 L-168.5 -37.5 L-161.8 -37.0 L-154.7 -36.2 L-147.1 -35.2 L-139.0 -33.9 L-130.5 -32.4 L-121.6 -30.7 L-112.2 -28.7 L-102.5 -26.6 L-92.4 -24.1 L-81.9 -21.5 L-71.0 -18.6 L-59.8 -15.5 L-48.3 -12.3 L-36.4 -8.7 L-24.2 -5.0 L-11.7 -1.1 L1.0 3.0 L14.0 7.3 L27.3 11.8 L40.8 16.4 L54.5 21.3 L68.4 26.3 L82.5 31.5 L96.7 36.8 L111.1 42.3 L125.6 47.9 L140.3 53.7 L155.0 59.5 L169.8 65.6 L184.6 71.7 L199.5 77.9 L214.5 84.3 L229.4 90.7 L244.3 97.2 L259.2 103.8 L274.0 110.4 L288.7 117.2 L303.4 123.9 L318.0 130.7 L332.4 137.6 L346.7 144.5 L360.8 151.4 L374.8 158.3 L388.5 165.2 L402.1 172.1 L415.4 179.0 L428.5 185.8 L441.4 192.7 L453.9 199.4 L466.2 206.2 L478.1 212.9 L489.8 219.5 L501.1 226.1 L512.1 232.5 L522.7 238.9 L532.9 245.2 L542.8 251.4 L552.2 257.4 L561.2 263.4 L569.9 269.2 L578.1 274.9 L585.8 280.5 L593.1 285.9 L600.0 291.1 L606.3 296.2 L612.2 301.2 L617.7 305.9 L622.6 310.5 L627.0 314.9 L631.0 319.1 L634.4 323.1 L637.3 326.9 L639.7 330.5 L641.6 333.9 L643.0 337.1 L643.8 340.1 L644.2 342.8 L644.0 345.4 L643.3 347.7 L642.1 349.8 L640.3 351.6 L638.1 353.2 L635.3 354.6 L632.0 355.7 L628.2 356.6 L623.9 357.3 L619.2 357.7 L613.9 357.9 L608.1 357.8 L601.9 357.5 L595.2 357.0 L588.0 356.2 L580.4 355.2 L572.3 353.9 L563.8 352.4 L554.9 350.7 L545.6 348.7 L535.8 346.6 L525.7 344.1 L515.2 341.5 L504.4 338.6 L493.2 335.5 L481.6 332.3 L469.7 328.7 L457.6 325.0 L445.1 321.1 L432.3 317.0 L419.3 312.7 L406.1 308.2 L392.6 303.6 L378.9 298.7 L365.0 293.7 L350.9 288.5 L336.6 283.2 L322.2 277.7 L307.7 272.1 L293.1 266.3 L278.3 260.5 L263.5 254.4 L248.7 248.3 L233.8 242.1 L218.9 235.7 L203.9 229.3 L189.0 222.8 L174.2 216.2 L159.3 209.6 L144.6 202.8 L129.9 196.1 L115.4 189.3 L100.9 182.4 L86.6 175.5 L72.5 168.6 L58.6 161.7 L44.8 154.8 L31.2 147.9 L17.9 141.0 L4.8 134.2 L-8.0 127.3 L-20.6 120.6 L-32.8 113.8 L-44.8 107.1 L-56.5 100.5 L-67.8 93.9 L-78.7 87.5 L-89.3 81.1 L-99.6 74.8 L-109.4 68.6 L-118.9 62.6 L-127.9 56.6 L-136.5 50.8 L-144.7 45.1 L-152.5 39.5 L-159.8 34.1 L-166.6 28.9 L-173.0 23.8 L-178.9 18.8 L-184.3 14.1 L-189.2 9.5 L-193.7 5.1 L-197.6 0.9 L-201.1 -3.1 L-204.0 -6.9 L-206.4 -10.5 L-208.3 -13.9 L-209.7 -17.1 L-210.5 -20.1 L-210.9 -22.8 L-210.7 -25.4 L-210.0 -27.7" class="stroke-a"/>
</g>
<text x="230.0" y="300.0" class="fig-small" text-anchor="middle">w1 (height of the line at size 0)</text>
<text x="42.0" y="145.0" class="fig-small" text-anchor="end">w2</text>
<text x="50.0" y="284.0" class="fig-small" text-anchor="middle">-1</text>
<text x="150.0" y="284.0" class="fig-small" text-anchor="middle">0</text>
<text x="250.0" y="284.0" class="fig-small" text-anchor="middle">1</text>
<text x="350.0" y="284.0" class="fig-small" text-anchor="middle">2</text>
<text x="46.0" y="214.0" class="fig-small" text-anchor="end">0</text>
<text x="46.0" y="114.0" class="fig-small" text-anchor="end">1</text>
<line x1="150.0" y1="110.0" x2="162.0" y2="74.0" class="stroke-bad" marker-end="url(#c003-arrow)"/>
<circle cx="150.0" cy="110.0" r="5" class="fill-b"/>
<rect x="52.0" y="101.0" width="92" height="18" class="fill-sunk"/>
<text x="140.0" y="114.0" class="fig-small" text-anchor="end">(0, 1): loss 0.5</text>
<text x="160.0" y="40.0" class="fig-small"></text>
<rect x="166.0" y="62.0" width="78" height="18" class="fill-sunk"/>
<text x="170.0" y="76.0" class="fig-small">slope (1, 3)</text>
<circle cx="216.7" cy="160.0" r="5" class="fill-c"/>
<rect x="223.7" y="181.0" width="150" height="18" class="fill-sunk"/>
<text x="226.7" y="194.0" class="fig-small">(2/3, 1/2): loss 0.083</text>
<text x="420.0" y="34.0" class="fig-small">rings are loss</text>
<text x="420.0" y="50.0" class="fig-small">levels</text>
<text x="420.0" y="66.0" class="fig-small">0.1, 0.25, 0.5,</text>
<text x="420.0" y="82.0" class="fig-small">1, 2, 4 from</text>
<text x="420.0" y="98.0" class="fig-small">inside out</text>
</svg>
Figure 2. Rings of equal loss for the three houses. The start (0, 1) has loss 0.5, and the red arrow, drawn shorter than its true length, points uphill. Notice that the rings shrink toward one point, the minimum at (2/3, 1/2), and that the bowl is long and narrow: many pairs of weights are almost as good as the best.
```

The best line is the lowest point of the bowl. You cannot see the bowl for a model with fifty weights, so you need a way to find the bottom without looking. At the bottom of a smooth bowl the ground is flat in every direction. The *slope* of the loss along each weight is zero there, and a point where both slopes are zero is the bottom, because the bowl has no other flat spot. Each miss is a straight-line function of the weights, and squaring a straight line gives a trough with one low point. Adding troughs gives another trough, so the whole loss is one smooth bowl with no side dips to get stuck in.

## Find the slope of the loss

Let $m_i = w_1 + w_2 x_i - y_i$ be the miss on house $i$. The loss is $\tfrac12\sum m_i^2$. Nudge $w_1$ by a small amount. Each miss moves by that same amount. A square $\tfrac12 m^2$ changes by about $m$ times the change in $m$, because the 2 from squaring cancels the half. So house $i$ changes the loss by $m_i$ times the nudge, and the slope along $w_1$ is $\sum m_i$. Nudge $w_2$ instead. Each miss moves by $x_i$ times the nudge, so the slope along $w_2$ is $\sum m_i x_i$. A big house ($x_i$ large) pulls harder on $w_2$, which is why its miss counts for more.

Collect both slopes in one vector. This vector of slopes is the *gradient*. It points uphill, and its length says how steep the ground is. Written with the table, the gradient is each column of $X$ dotted with the misses:

$$
\nabla L(w)=X^\top(Xw-y)
$$

Read it aloud: multiply the transpose of the table by the vector of misses. Entry one adds up the misses. Entry two adds up the misses weighted by size. At $w=(0,1)$ the misses are $(0,0,1)$, so the gradient is $(1,3)$. That is the red arrow in Figure 2. It points up and to the right, away from the minimum, so stepping the other way goes downhill. The chapter on [walking downhill](/lessons/005-gradient-descent-and-momentum) turns that observation into an algorithm.

```python
gradient = X.T @ (X @ w - y)  # shape (d,)
```

## Set the slope to zero

At the bottom both slopes are zero, so $X^\top(Xw-y)=0$. Move the $y$ term across:

$$
X^\top X\,w = X^\top y
$$

These are the *normal equations*. They are not a recipe handed down. They are the sentence "the slope is zero in every direction", written in matrices. Compute the two pieces for the houses. $X^\top X$ holds the dot products of the columns: $1+1+1=3$, $1+2+3=6$, and $1+4+9=14$. $X^\top y$ holds each column dotted with $y$: $1+2+2=5$ and $1+4+6=11$. So you must solve two equations in two unknowns:

$$
\begin{bmatrix}3&6\\6&14\end{bmatrix}\begin{bmatrix}w_1\\w_2\end{bmatrix}=\begin{bmatrix}5\\11\end{bmatrix}
$$

The determinant is $3\cdot14-6\cdot6=6$. Then $w_1=(14\cdot5-6\cdot11)/6=4/6=2/3$ and $w_2=(3\cdot11-6\cdot5)/6=3/6=1/2$. Check the first equation: $3\cdot\tfrac23+6\cdot\tfrac12=2+3=5$. Check the second: $6\cdot\tfrac23+14\cdot\tfrac12=4+7=11$. The best line is price $=2/3+\tfrac12\times$ size, and a house of size 4 gets a predicted price of $2/3+2\approx2.67$ lakhs.

This system has exactly one answer when no column is a copy of another. When one is, many weights fit equally well, and the chapter on [QR and the SVD](/lessons/004-stable-least-squares) deals with that. Ask numpy to solve the system, not to invert the matrix:

```python
w = np.linalg.solve(X.T @ X, X.T @ y)  # shape (d,)
```

Why not compute the inverse $(X^\top X)^{-1}$ and multiply? In exact arithmetic you get the same weights. On a computer, a solve does fewer operations and avoids one extra round of rounding, so it is faster and a little more accurate.

> [!KEY]
> A line is "best" when it makes the sum of squared vertical misses as small as possible. The loss is a bowl in the weights, its slope is $X^\top(Xw-y)$, and setting that slope to zero gives the normal equations, whose solution is the bottom of the bowl.

## Name the leftovers

Turn to your colleague's weights. You could refit and compare, but trying nudges does not work, because a model with 50 weights can be nudged in infinitely many directions. You need a single number that is zero exactly when no nudge helps.

Here it is easier to work with the opposite sign from the miss. The *residual* is what the model fails to explain:

$$
r = y - Xw
$$

Read it aloud: the true prices minus the guessed prices. The slope of the loss is $X^\top(Xw-y)=-X^\top r$, so it is zero exactly when $X^\top r$ is zero. The slope check and the residual check are the same check with a flipped sign.

Take the best weights $(2/3,1/2)$. The guessed prices, which this chapter also calls the *fitted values*, are $2/3+1/2=7/6$, then $2/3+1=5/3$, then $2/3+3/2=13/6$. The true prices are $(1,2,2)$, so the residual is $(-1/6,\;1/3,\;-1/6)$. Dot the residual with each column of $X$. With the column of ones: $-1/6+1/3-1/6=0$. With the column of sizes $(1,2,3)$: $-1/6+2/3-3/6=0$. Both are zero.

Try the line $y=x$, with weights $(0,1)$. The fitted values are $(1,2,3)$ and the residual is $(0,0,-1)$. Dotting with the ones gives $-1$, and dotting with the sizes gives $-3$. Neither is zero, and they say which way the loss still falls.

## Why zero dot products mean best

A dot product of zero has a geometric meaning: the two vectors meet at a right angle, which the word *orthogonal* names. So $X^\top r=0$ says the residual is orthogonal to every column of $X$.

Think about what the columns can reach. Any weights $w$ give a guess $Xw$, which is a mix of the columns: $w_1$ copies of the ones column plus $w_2$ copies of the sizes column. The set of all such mixes is the *column space*. For two columns in three-dimensional space it is a flat sheet through the origin, so you can draw it. With more columns it is a flat slice of higher dimension, and the argument below does not change. Figure 3 shows the sheet, the target vector $y=(1,2,2)$ floating above it, and the fitted values $Xw$ on the sheet.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="The vector y above the flat sheet made by the two columns of X. Its shadow on the sheet is Xw, and the leftover r drops straight down to the sheet at a right angle.">
<defs><marker id="c003-arrow-md" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-d"/></marker><marker id="c003-arrow-ma" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-a"/></marker><marker id="c003-arrow-mr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-bad"/></marker></defs>
<path d="M-2.3 198.2 L262.9 198.2 L360.2 126.6 L95.0 126.6 Z" class="fill-soft-a stroke-muted"/>
<line x1="50.0" y1="175.0" x2="140.1" y2="175.0" class="stroke-d" marker-end="url(#c003-arrow-md)"/>
<text x="136.1" y="191.0" class="fig-small" text-anchor="end">(1,1,1)</text>
<line x1="50.0" y1="175.0" x2="270.6" y2="145.2" class="stroke-d" marker-end="url(#c003-arrow-md)"/>
<text x="276.6" y="139.2" class="fig-small" text-anchor="start">(1,2,3)</text>
<line x1="50.0" y1="175.0" x2="220.3" y2="160.1" class="stroke-b dashed"/>
<line x1="50.0" y1="175.0" x2="220.3" y2="54.0" class="stroke-a" marker-end="url(#c003-arrow-ma)"/>
<line x1="220.3" y1="160.1" x2="220.3" y2="54.0" class="stroke-bad" marker-end="url(#c003-arrow-mr)"/>
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
Figure 3. The sheet holds every guess the line model can make. The target y sits above it. The fitted values are the point of the sheet directly beneath y, and the residual r (red) drops straight down. Notice that the red arrow is perpendicular to the sheet, which is why it is the shortest way down. The vertical direction is stretched so the short arrow stays visible.
```

Picking the best weights means picking the point of the sheet closest to $y$. The closest point on a flat sheet to something above it is the foot of the perpendicular, as when you drop a plumb line from a ceiling. Here is the reason. Take any other point $p$ on the sheet. The residual is orthogonal to the sheet, so it is orthogonal to the segment from the fitted values to $p$. Pythagoras then gives $\lVert y-p\rVert^2=\lVert r\rVert^2+\lVert Xw-p\rVert^2$, which is larger than $\lVert r\rVert^2$ unless $p$ is the foot.

This also answers the nudge question directly. Change the weights by a small step $\delta$, so the fitted values move to $Xw+X\delta$, another point of the sheet. The same argument says the new squared residual is $\lVert r\rVert^2+\lVert X\delta\rVert^2$. You add something that is zero or positive, so no nudge in any direction lowers the loss.

Check the Pythagoras step with real numbers. Take $p=(1,2,3)$, the guess from weights $(0,1)$. Then $\lVert y-p\rVert^2=1$. The residual has squared length $1/6$, and the segment from $(7/6,5/3,13/6)$ to $p$ is $(1/6,-1/3,-5/6)$, with squared length $30/36=5/6$. Indeed $1/6+5/6=1$.

> [!KEY]
> The fitted values are the *projection* of $y$ onto the column space: the point of the sheet closest to $y$, found by dropping a perpendicular. The residual is orthogonal to every column, which is the same sentence as "every slope is zero".

Figure 4 draws the weights $(2/3,1/2)$ next to the weights $(0,1)$.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Two copies of the sheet picture. Left: the best fit, with the leftover straight down and both dot products zero. Right: the weights (0, 1), where the leftover slants and the dot products are minus 1 and minus 3.">
<defs><marker id="c003-arrow-wd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-d"/></marker><marker id="c003-arrow-wa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-a"/></marker><marker id="c003-arrow-wr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-bad"/></marker></defs>
<path d="M-15.2 182.8 L188.8 182.8 L263.6 127.7 L59.6 127.7 Z" class="fill-soft-a stroke-muted"/>
<line x1="25.0" y1="165.0" x2="94.3" y2="165.0" class="stroke-d" marker-end="url(#c003-arrow-wd)"/>
<line x1="25.0" y1="165.0" x2="194.7" y2="142.1" class="stroke-d" marker-end="url(#c003-arrow-wd)"/>
<line x1="25.0" y1="165.0" x2="156.0" y2="153.5" class="stroke-b dashed"/>
<line x1="25.0" y1="165.0" x2="156.0" y2="71.9" class="stroke-a" marker-end="url(#c003-arrow-wa)"/>
<line x1="156.0" y1="153.5" x2="156.0" y2="71.9" class="stroke-bad" marker-end="url(#c003-arrow-wr)"/>
<circle cx="156.0" cy="153.5" r="4.5" class="fill-b"/>
<circle cx="25.0" cy="165.0" r="3.5" class="fill-ink"/>
<text x="25.0" y="22.0" class="fig-bold">Best weights (2/3, 1/2)</text>
<text x="25.0" y="196.0" class="fig-small">residual (-1/6, 1/3, -1/6)</text>
<text x="25.0" y="214.0" class="fig-small">dot products with columns: 0 and 0</text>
<text x="164.0" y="169.5" class="fig-small">Xw</text>
<text x="150.0" y="65.9" class="fig-small" text-anchor="end">y</text>
<text x="148.0" y="116.7" class="fig-bold" text-anchor="end">r</text>
<path d="M284.8 182.8 L488.8 182.8 L563.6 127.7 L359.6 127.7 Z" class="fill-soft-a stroke-muted"/>
<line x1="325.0" y1="165.0" x2="394.3" y2="165.0" class="stroke-d" marker-end="url(#c003-arrow-wd)"/>
<line x1="325.0" y1="165.0" x2="494.7" y2="142.1" class="stroke-d" marker-end="url(#c003-arrow-wd)"/>
<line x1="325.0" y1="165.0" x2="494.7" y2="142.1" class="stroke-b dashed"/>
<line x1="325.0" y1="165.0" x2="456.0" y2="71.9" class="stroke-a" marker-end="url(#c003-arrow-wa)"/>
<line x1="494.7" y1="142.1" x2="456.0" y2="71.9" class="stroke-bad" marker-end="url(#c003-arrow-wr)"/>
<circle cx="494.7" cy="142.1" r="4.5" class="fill-b"/>
<circle cx="325.0" cy="165.0" r="3.5" class="fill-ink"/>
<text x="325.0" y="22.0" class="fig-bold">Weights (0, 1)</text>
<text x="325.0" y="196.0" class="fig-small">residual (0, 0, -1)</text>
<text x="325.0" y="214.0" class="fig-small">dot products with columns: -1 and -3</text>
<text x="502.7" y="158.1" class="fig-small">Xw</text>
<text x="450.0" y="65.9" class="fig-small" text-anchor="end">y</text>
<text x="483.4" y="111.0" class="fig-bold" text-anchor="start">r</text>
</svg>
Figure 4. Left: the best fit, where the residual is straight down and both dot products are zero. Right: the weights (0, 1) land at a different point of the sheet. The residual slants, is longer, and has dot products of -1 and -3. Notice that the slant is the signal: part of the residual still points along the sheet, and that is the part the dot products pick up.
```

The slanted residual has a part that lies along the sheet. That part is what the dot products measure, and it is also the direction you could still move to shrink the error.

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

Expand $X^\top(y-Xw)=0$ into $X^\top y - X^\top X w=0$ and you get the normal equations again. The two are one fact in two outfits. The slope outfit says "the ground is flat". The geometry outfit says "the residual stands at a right angle to the sheet". Having both lets you check an answer you did not compute yourself.

> [!KEY]
> Least squares has one answer in three descriptions: the bottom of the loss bowl, the solution of $X^\top X w = X^\top y$, and the foot of the perpendicular from $y$ to the column space. The single number $\max|X^\top(y-Xw)|$ tells you whether any weights, from any source, are that answer.

> [!RECAP]
> - The table $X$ has one row per house and a column of ones, so the guesses are `X @ w`. The loss is `0.5 * np.sum((X @ w - y) ** 2)`, and the half does not move the bottom.
> - The slope of the loss is `X.T @ (X @ w - y)`. Setting it to zero gives $X^\top X w = X^\top y$. For the three houses, $w=(2/3,1/2)$.
> - The residual is $r=y-Xw$. The best fit has $X^\top r=0$: the residual is orthogonal to every column.
> - The fitted values are the projection of $y$ onto the sheet the columns span.
> - As one number, take `np.abs(X.T @ r).max()`, and compare it with `np.abs(X.T @ y).max()` to decide.

## Build it step by step

The starter gives you `add_ones_column`, which builds $X$ from a one-dimensional array of sizes. You write six functions, in the order the checks test them.

1. **`loss(X, y, w)`** takes $X$ of shape `(n, d)`, `y` of shape `(n,)`, and `w` of shape `(d,)`. It returns one float, half the sum of squared misses:

   ```python
   misses = X @ w - y  # guess minus truth
   # square, add, halve
   ```

   Forgetting the half is the common mistake.

2. **`gradient(X, y, w)`** returns a vector of shape `(d,)`. The check compares it with finite differences, which nudge each weight by a tiny amount and measure how the loss moves:

   ```python
   misses = X @ w - y
   # return the transpose of X times the misses
   ```

   A sign slip, `y - X @ w`, points downhill and fails the check.

3. **`fit_normal_equations(X, y)`** returns `w` of shape `(d,)`:

   ```python
   # np.linalg.solve(X.T @ X, X.T @ y)
   ```

   Calling `np.linalg.inv` also passes these checks on small tables. Use `solve` anyway, to build the habit before the chapter on [QR and the SVD](/lessons/004-stable-least-squares) shows the cost.

4. **`residual_orthogonality(X, y, w)`** returns one float:

   ```python
   r = y - X @ w
   # abs of X.T @ r, then max, then float
   ```

   Returning the whole vector instead of one number fails the check.

5. **`project(X, y)`** returns the fitted values, shape `(n,)`, using the solver you already wrote:

   ```python
   w = fit_normal_equations(X, y)
   # return X @ w
   ```

   Returning `w` instead of `X @ w` gives the weights, not the shadow.

6. **`is_best_fit(X, y, w, tol)`** returns `True` when the check is at most `tol` times the scale of $X^\top y$:

   ```python
   scale = np.abs(X.T @ y).max()
   # compare residual_orthogonality(...) with tol * scale
   ```

   Comparing with `tol` alone gives different answers when you multiply all the prices by a million.

## Where this breaks

**One wild house.** Squaring multiplies a big miss by itself, so one house priced far off the others drags the line toward it. Plot the misses and look for one that dwarfs the rest. A loss that grows more slowly than the square, such as the absolute miss, reduces the pull.

**A curved cloud.** A straight line cannot follow a bend. Add a column such as $x^2$ to the table. The fit stays linear in the weights, so every formula in this chapter still applies.

**Columns on wild scales or repeated columns.** If one column holds millimetres and another kilometres, the bowl becomes extremely long and thin, and the solve loses accuracy, so standardise each column first. If one column repeats another, many weight vectors reach the same foot point and all of them pass the right-angle check. The check says "this is a best fit", not "this is the only one". The chapter on [QR and the SVD](/lessons/004-stable-least-squares) takes both cases on.

**A tolerance that is too tight, or a zero target.** A tolerance of $10^{-16}$ rejects correct answers because of rounding, so use a relative tolerance of about $10^{-8}$. If $y$ is all zeros, then $X^\top y=0$ and a relative test compares with zero, so use an absolute tolerance in that case.

## The exercise

You implement `loss`, `gradient`, `fit_normal_equations`, `residual_orthogonality`, `project`, and `is_best_fit`. The first checks use the three houses and seeded random tables. They confirm that `loss` equals half the sum of squared misses, that `gradient` matches finite differences at several random weight vectors, and that `fit_normal_equations` returns the hand-solved $(2/3,1/2)$ and weights where the gradient is zero.

The remaining checks cover the right-angle test. `residual_orthogonality` must give 3, 5, and 0 for the three candidate weights, and `project` must return the fitted values 7/6, 5/3, and 13/6. The projected residual must be orthogonal to every column, with $\lVert y\rVert^2=\lVert Xw\rVert^2+\lVert r\rVert^2$, and no other point of the sheet may be closer to $y$. Finally, `is_best_fit` must accept the best weights, reject nudged ones, and agree when the prices are scaled by a million.
