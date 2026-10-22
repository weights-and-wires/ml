---
title: Leave-one-out error from a single ridge fit
slug: 025-leave-one-out-shortcut
part: "Part 2 — Classical models"
summary: A ridge residual flatters the model because the point helped train it, and dividing by one minus the point's leverage gives the honest leave-one-out residual.
minutes: 40
gpu: false
objectives:
  - Explain why a point's ordinary residual is smaller than its honest leave-one-out residual.
  - Build the hat matrix and read each point's leverage from its diagonal.
  - Compute the leave-one-out error of ridge regression from a single fit.
  - Say when the shortcut works, when the divisor misbehaves, and what it costs.
prereqs:
  - 006-residuals-and-projection
  - 013-ridge-penalty-and-closed-form
  - 024-cross-validation
exercise: 025-leave-one-out-shortcut
---

# Leave-one-out error from a single ridge fit

You have 60 houses and a ridge model, and you want the most honest score you can get. In the previous chapter you cut the houses into folds. The most generous version of that idea gives every house its own fold. You hide one house, fit on the other 59, and measure the miss on the hidden house. You repeat that for each of the 60 houses and average the squared misses. This is *leave-one-out cross-validation*. It trains on almost all your data each time, so it is the least pessimistic of the fold schemes.

The cost is the problem. Sixty houses mean sixty fits, and a million houses mean a million fits. If each fit takes a second, you wait eleven days to score one penalty, and you have twenty penalties to try.

This chapter shows that for ridge regression you do not have to refit at all. One fit on all the houses already contains the 60 honest misses, and a small correction recovers them exactly.

## What you already know

- In leave-one-out cross-validation, each point is its own fold, so $k = n$ and there are $n$ fits.
- A residual is the target minus the prediction, so a house priced at 5 and predicted at 4.5 has residual 0.5.
- Ridge regression fits $w = (X^{\top}X + \lambda I)^{-1}X^{\top}y$, and the prediction for each house is its row of $X$ times $w$.
- A matrix times a vector gives a list of weighted sums, so each output is a recipe that mixes the entries of the input.
- The diagonal of a matrix is the list of entries in row $i$, column $i$.

## The ordinary residual flatters the model

Figure 1 shows six houses, at floor areas 0.5, 1, 1.5, 2, 2.5, and 5.5, and a ridge line through the origin with penalty $\lambda = 1$. The far-right house, at floor area 5.5, is what makes this picture useful. Fitted on all six, the line has slope 0.70, and it misses the far house by 0.54. Remove that point and refit on the other five. The slope drops to 0.50, and that line misses the far point by 1.64, three times as much.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="A line through six points, fitted with and without the far-right point; the far point pulls the line toward itself">
<line x1="70" y1="250" x2="560" y2="250" class="stroke-grid"/><text x="60" y="254" text-anchor="end" class="fig-small">0</text>
<line x1="70" y1="206" x2="560" y2="206" class="stroke-grid"/><text x="60" y="210" text-anchor="end" class="fig-small">1</text>
<line x1="70" y1="162" x2="560" y2="162" class="stroke-grid"/><text x="60" y="166" text-anchor="end" class="fig-small">2</text>
<line x1="70" y1="118" x2="560" y2="118" class="stroke-grid"/><text x="60" y="122" text-anchor="end" class="fig-small">3</text>
<line x1="70" y1="74" x2="560" y2="74" class="stroke-grid"/><text x="60" y="78" text-anchor="end" class="fig-small">4</text>
<line x1="70" y1="30" x2="560" y2="30" class="stroke-grid"/><text x="60" y="34" text-anchor="end" class="fig-small">5</text>
<line x1="70" y1="250" x2="70" y2="254" class="stroke-muted"/><text x="70" y="268" text-anchor="middle" class="fig-small">0</text>
<line x1="130" y1="250" x2="130" y2="254" class="stroke-muted"/><text x="130" y="268" text-anchor="middle" class="fig-small">1</text>
<line x1="190" y1="250" x2="190" y2="254" class="stroke-muted"/><text x="190" y="268" text-anchor="middle" class="fig-small">2</text>
<line x1="250" y1="250" x2="250" y2="254" class="stroke-muted"/><text x="250" y="268" text-anchor="middle" class="fig-small">3</text>
<line x1="310" y1="250" x2="310" y2="254" class="stroke-muted"/><text x="310" y="268" text-anchor="middle" class="fig-small">4</text>
<line x1="370" y1="250" x2="370" y2="254" class="stroke-muted"/><text x="370" y="268" text-anchor="middle" class="fig-small">5</text>
<line x1="430" y1="250" x2="430" y2="254" class="stroke-muted"/><text x="430" y="268" text-anchor="middle" class="fig-small">6</text>
<line x1="70" y1="250" x2="560" y2="250" class="stroke-muted"/><line x1="70" y1="250" x2="70" y2="30" class="stroke-muted"/>
<text x="250" y="288" text-anchor="middle">floor area</text><text x="16" y="140" transform="rotate(-90 16 140)" text-anchor="middle">price</text>
<line x1="70" y1="250" x2="442.0" y2="58.43377777777772" class="stroke-b" stroke-width="2.5"/>
<line x1="70" y1="250" x2="442.0" y2="113.13762711864408" class="stroke-d" stroke-width="2.5" stroke-dasharray="6 4"/>
<circle cx="100.0" cy="241.2" r="5.5" class="fill-a"/>
<circle cx="130.0" cy="197.2" r="5.5" class="fill-a"/>
<circle cx="160.0" cy="228.0" r="5.5" class="fill-a"/>
<circle cx="190.0" cy="192.8" r="5.5" class="fill-a"/>
<circle cx="220.0" cy="201.6" r="5.5" class="fill-a"/>
<circle cx="400.0" cy="56.4" r="5.5" class="fill-a"/>
<line x1="394.0" y1="58.0" x2="394.0" y2="82.0" class="stroke-b" stroke-width="4"/>
<line x1="408.0" y1="58.0" x2="408.0" y2="125.0" class="stroke-d" stroke-width="4"/>
<text x="452.0" y="76.0" class="fig-small">miss 0.54 with the point</text>
<text x="452.0" y="130.0" class="fig-small">miss 1.64 without it</text>
<line x1="86" y1="42" x2="118" y2="42" class="stroke-b" stroke-width="2.5"/><text x="126" y="46" class="fig-small">fit on all six points: slope 0.70</text>
<line x1="86" y1="62" x2="118" y2="62" class="stroke-d" stroke-width="2.5" stroke-dasharray="6 4"/><text x="126" y="66" class="fig-small">fit without the far point: slope 0.50</text>
</svg>
Figure 1. The far point pulls the line toward itself. Fitted with the point, the line misses it by 0.54 (orange). Fitted without it, the line misses it by 1.64 (purple, dashed), which is the honest leave-one-out miss.
```

The reason is plain once you see it. The fit is chosen to make squared misses small, and the far point has a loud voice because it sits so far out that tilting the line moves the prediction there a lot. So the line tilts toward it. The miss at that point then looks small, although the model would have missed by far more if it had never seen the point.

A point near the middle has a quieter voice. Tilting the line barely changes the prediction at the middle, so the line ignores that point's wishes and its residual stays close to its honest value. The gap between the ordinary residual and the honest residual therefore depends on how strongly each point pulls on its own prediction. Any correction has to measure that pull.

## Measure how much a point pulls on itself

Ridge predictions are linear in the targets: every fitted value is a weighted sum of all $n$ targets. Collect the weights in an $n$-by-$n$ matrix $H$, so the fitted values are $\hat{y} = Hy$. People call $H$ the *hat matrix*, because it puts a hat on $y$. Row $i$ lists how much each target contributes to the fit at point $i$.

$$
H = X (X^{\top}X + \lambda I)^{-1} X^{\top}
$$

Read it aloud: take the ridge weights, $(X^{\top}X + \lambda I)^{-1}X^{\top}y$, and multiply by $X$ to get predictions, and everything in front of $y$ is $H$. In numpy it is `X @ np.linalg.inv(X.T @ X + lam * np.eye(d)) @ X.T`.

The diagonal entry $h_{ii}$ is the weight of target $i$ in its own fitted value. Call it the *leverage* of point $i$. It sits between two extremes:

- If $h_{ii} = 0$, the point has no say in its own prediction. Its ordinary residual is already honest, because the fit would be identical without it.
- If $h_{ii}$ is close to 1, the fit passes almost exactly through the point. Its ordinary residual is close to 0 whatever the target is, which tells you nothing about how the model does on unseen data.

For the inputs $1, 2, 3$ with $\lambda = 1$ there is one column, and $X^{\top}X + \lambda = 1 + 4 + 9 + 1 = 15$. Entry $(i, j)$ of $H$ is $x_i x_j / 15$, so the leverages are $1/15$, $4/15$, and $9/15$. The point with the largest input has the most pull, as in Figure 1.

## Undo the pull, and check by hand

The ordinary residual is too small by a factor that depends on the pull. The fix divides it by one minus the leverage:

$$
y_i - \hat{y}_i^{(-i)} = \frac{r_i}{1 - h_{ii}}, \qquad r = y - Hy
$$

Here $\hat{y}_i^{(-i)}$ is the prediction for point $i$ from a fit that never saw point $i$. Read it aloud: the honest miss equals the ordinary miss, blown up by $1/(1 - h_{ii})$. When $h_{ii}$ is 0 nothing changes. When $h_{ii}$ is close to 1 the divisor is tiny and the nearly zero residual is blown back up, which is what you want, because a model that has memorised a point says nothing about its honest miss.

Test it on the targets $1, 2, 2$, with inputs $1, 2, 3$ and $\lambda = 1$. The weight is $w = 11/15$, since $X^{\top}y = 1 + 4 + 6 = 11$. The ordinary residuals are $4/15$, $8/15$, and $-1/5$. Dividing each by one minus its leverage and refitting without the point gives the same answers:

| Point | Residual $r_i$ | Leverage | $r_i / (1 - h_{ii})$ | Weight without the point | Honest miss from refitting |
|---|---|---|---|---|---|
| 1 | $4/15$ | $1/15$ | $2/7$ | $10/14$ | $1 - 5/7 = 2/7$ |
| 2 | $8/15$ | $4/15$ | $8/11$ | $7/11$ | $2 - 14/11 = 8/11$ |
| 3 | $-1/5$ | $9/15$ | $-1/2$ | $5/6$ | $2 - 5/2 = -1/2$ |

The two right-hand columns agree in every row, and the last point, with the most leverage, is corrected the most. The mean of the squared honest misses is $(4/49 + 64/121 + 1/4)/3 = 0.2869$, while the same average over the ordinary residuals is only $0.1319$.

Why does the division work? Here is the argument in words. Suppose you replace the target $y_i$ by the prediction that the fit without point $i$ makes there. That fit already agrees with the replaced point perfectly, so adding the point changes nothing, and the fit stays the same. Because ridge is linear, replacing $y_i$ by something else shifts the prediction at point $i$ by $h_{ii}$ times the change. Write $\hat{y}_i$ for the ordinary prediction and $\hat{y}_i^{(-i)}$ for the honest one. The linearity fact says the new prediction at point $i$ is $\hat{y}_i + h_{ii}(\hat{y}_i^{(-i)} - y_i)$. The stays-the-same fact says it is also $\hat{y}_i^{(-i)}$. Setting the two equal and solving for $y_i - \hat{y}_i^{(-i)}$ gives $(y_i - \hat{y}_i)/(1 - h_{ii})$, which is the formula above.

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="Leverage, ordinary residual, and leave-one-out residual for three points; the leave-one-out residual is larger in size at every point, and most where leverage is highest">
<text x="125.0" y="30" text-anchor="middle" class="fig-bold">leverage h</text>
<line x1="40" y1="230.0" x2="210" y2="230.0" class="stroke-muted"/>
<rect x="54" y="218.0" width="36" height="12.0" class="fill-d"/>
<text x="72.0" y="212.0" text-anchor="middle" class="fig-small">1/15</text>
<text x="72.0" y="250" text-anchor="middle" class="fig-small">x = 1</text>
<rect x="106" y="182.0" width="36" height="48.0" class="fill-d"/>
<text x="124.0" y="176.0" text-anchor="middle" class="fig-small">4/15</text>
<text x="124.0" y="250" text-anchor="middle" class="fig-small">x = 2</text>
<rect x="158" y="122.0" width="36" height="108.0" class="fill-d"/>
<text x="176.0" y="116.0" text-anchor="middle" class="fig-small">9/15</text>
<text x="176.0" y="250" text-anchor="middle" class="fig-small">x = 3</text>
<text x="430.0" y="30" text-anchor="middle" class="fig-bold">residual, in-sample (blue) and leave-one-out (orange)</text>
<line x1="260" y1="152.9" x2="600" y2="152.9" class="stroke-muted"/>
<rect x="274" y="118.6" width="40" height="34.3" class="fill-a"/>
<text x="294" y="112.6" text-anchor="middle" class="fig-small">4/15</text>
<rect x="318" y="116.1" width="40" height="36.7" class="fill-b"/>
<text x="338" y="110.1" text-anchor="middle" class="fig-small">2/7</text>
<text x="316" y="250" text-anchor="middle" class="fig-small">x = 1</text>
<rect x="384" y="84.3" width="40" height="68.6" class="fill-a"/>
<text x="404" y="78.3" text-anchor="middle" class="fig-small">8/15</text>
<rect x="428" y="59.4" width="40" height="93.5" class="fill-b"/>
<text x="448" y="53.4" text-anchor="middle" class="fig-small">8/11</text>
<text x="426" y="250" text-anchor="middle" class="fig-small">x = 2</text>
<rect x="494" y="152.9" width="40" height="25.7" class="fill-a"/>
<text x="514" y="193.6" text-anchor="middle" class="fig-small">-1/5</text>
<rect x="538" y="152.9" width="40" height="64.3" class="fill-b"/>
<text x="558" y="232.1" text-anchor="middle" class="fig-small">-1/2</text>
<text x="536" y="250" text-anchor="middle" class="fig-small">x = 3</text>
<text x="310" y="280" text-anchor="middle" class="fig-small">the higher the leverage, the more the in-sample residual is shrunk</text>
</svg>
Figure 2. The leverages, ordinary residuals, and leave-one-out residuals of the hand example. Every honest residual is larger in size than the ordinary one, and the gap grows with leverage.
```

In numpy the shortcut is three lines, with `lam * np.eye(d)` as the term people forget:

```python
H = hat_matrix(X, lam)                  # shape (n, n)
residual = y - H @ y                    # ordinary residuals, shape (n,)
honest = residual / (1.0 - np.diag(H))  # leave-one-out residuals
```

## When the divisor misbehaves

The divisor $1 - h_{ii}$ must stay away from 0, and the penalty is what guarantees that. Figure 3 uses 8 points with 10 features each. With a penalty near 0, the model has more freedom than data and passes through every point, so every leverage is 1 and the formula divides $0$ by $0$. As the penalty grows, every leverage falls toward 0, and a point far from the others keeps its leverage longest. With a positive penalty every leverage is strictly below 1, so the shortcut is safe.

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="Leverage of three points as the ridge penalty grows: each starts at 1 when there are more features than points and falls toward 0">
<line x1="70" y1="240.0" x2="570" y2="240.0" class="stroke-grid"/><text x="58" y="244.0" text-anchor="end" class="fig-small">0</text>
<line x1="70" y1="187.5" x2="570" y2="187.5" class="stroke-grid"/><text x="58" y="191.5" text-anchor="end" class="fig-small">0.25</text>
<line x1="70" y1="135.0" x2="570" y2="135.0" class="stroke-grid"/><text x="58" y="139.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="70" y1="82.5" x2="570" y2="82.5" class="stroke-grid"/><text x="58" y="86.5" text-anchor="end" class="fig-small">0.75</text>
<line x1="70" y1="30.0" x2="570" y2="30.0" class="stroke-grid"/><text x="58" y="34.0" text-anchor="end" class="fig-small">1</text>
<line x1="70.0" y1="240" x2="70.0" y2="244" class="stroke-muted"/><text x="70.0" y="258" text-anchor="middle" class="fig-small">0.001</text>
<line x1="153.3" y1="240" x2="153.3" y2="244" class="stroke-muted"/><text x="153.3" y="258" text-anchor="middle" class="fig-small">0.01</text>
<line x1="236.7" y1="240" x2="236.7" y2="244" class="stroke-muted"/><text x="236.7" y="258" text-anchor="middle" class="fig-small">0.1</text>
<line x1="320.0" y1="240" x2="320.0" y2="244" class="stroke-muted"/><text x="320.0" y="258" text-anchor="middle" class="fig-small">1</text>
<line x1="403.3" y1="240" x2="403.3" y2="244" class="stroke-muted"/><text x="403.3" y="258" text-anchor="middle" class="fig-small">10</text>
<line x1="486.7" y1="240" x2="486.7" y2="244" class="stroke-muted"/><text x="486.7" y="258" text-anchor="middle" class="fig-small">100</text>
<line x1="570.0" y1="240" x2="570.0" y2="244" class="stroke-muted"/><text x="570.0" y="258" text-anchor="middle" class="fig-small">1000</text>
<line x1="70" y1="240" x2="570" y2="240" class="stroke-muted"/><line x1="70" y1="30" x2="70" y2="240" class="stroke-muted"/>
<text x="320" y="282" text-anchor="middle">ridge penalty (log scale)</text><text x="14" y="135" transform="rotate(-90 14 135)" text-anchor="middle">leverage h</text>
<polyline points="70.0,30.0 78.5,30.0 86.9,30.0 95.4,30.0 103.9,30.0 112.4,30.0 120.8,30.0 129.3,30.0 137.8,30.0 146.3,30.0 154.7,30.0 163.2,30.0 171.7,30.0 180.2,30.1 188.6,30.1 197.1,30.1 205.6,30.1 214.1,30.1 222.5,30.2 231.0,30.2 239.5,30.3 248.0,30.3 256.4,30.4 264.9,30.5 273.4,30.6 281.9,30.8 290.3,31.0 298.8,31.2 307.3,31.5 315.8,31.9 324.2,32.3 332.7,32.9 341.2,33.5 349.7,34.4 358.1,35.4 366.6,36.7 375.1,38.3 383.6,40.2 392.0,42.5 400.5,45.4 409.0,48.8 417.5,53.0 425.9,57.9 434.4,63.7 442.9,70.5 451.4,78.4 459.8,87.3 468.3,97.2 476.8,108.0 485.3,119.5 493.7,131.5 502.2,143.5 510.7,155.4 519.2,166.8 527.6,177.5 536.1,187.2 544.6,195.9 553.1,203.4 561.5,210.0 570.0,215.5" fill="none" class="stroke-b" stroke-width="2.5"/>
<polyline points="70.0,30.1 78.5,30.2 86.9,30.2 95.4,30.2 103.9,30.3 112.4,30.4 120.8,30.5 129.3,30.6 137.8,30.8 146.3,31.0 154.7,31.2 163.2,31.6 171.7,32.0 180.2,32.5 188.6,33.1 197.1,33.9 205.6,34.9 214.1,36.2 222.5,37.7 231.0,39.6 239.5,41.9 248.0,44.7 256.4,48.1 264.9,52.2 273.4,57.1 281.9,62.8 290.3,69.4 298.8,76.8 307.3,85.2 315.8,94.4 324.2,104.4 332.7,114.8 341.2,125.7 349.7,136.6 358.1,147.5 366.6,158.1 375.1,168.2 383.6,177.7 392.0,186.4 400.5,194.3 409.0,201.3 417.5,207.5 425.9,212.9 434.4,217.6 442.9,221.6 451.4,224.9 459.8,227.7 468.3,230.0 476.8,231.9 485.3,233.5 493.7,234.8 502.2,235.8 510.7,236.7 519.2,237.3 527.6,237.9 536.1,238.3 544.6,238.7 553.1,238.9 561.5,239.2 570.0,239.3" fill="none" class="stroke-a" stroke-width="2.5"/>
<polyline points="70.0,30.1 78.5,30.1 86.9,30.2 95.4,30.2 103.9,30.3 112.4,30.3 120.8,30.4 129.3,30.5 137.8,30.7 146.3,30.8 154.7,31.0 163.2,31.3 171.7,31.6 180.2,32.0 188.6,32.6 197.1,33.2 205.6,34.0 214.1,35.0 222.5,36.2 231.0,37.7 239.5,39.4 248.0,41.5 256.4,44.1 264.9,47.1 273.4,50.6 281.9,54.7 290.3,59.5 298.8,64.9 307.3,71.0 315.8,77.9 324.2,85.4 332.7,93.6 341.2,102.3 349.7,111.5 358.1,121.0 366.6,130.6 375.1,140.2 383.6,149.7 392.0,159.0 400.5,167.8 409.0,176.2 417.5,184.0 425.9,191.3 434.4,198.0 442.9,204.1 451.4,209.5 459.8,214.4 468.3,218.6 476.8,222.3 485.3,225.4 493.7,228.0 502.2,230.3 510.7,232.1 519.2,233.6 527.6,234.9 536.1,235.9 544.6,236.7 553.1,237.4 561.5,237.9 570.0,238.3" fill="none" class="stroke-a" stroke-width="2.5"/>
<line x1="90" y1="166" x2="122" y2="166" class="stroke-b" stroke-width="2.5"/><text x="130" y="170" class="fig-small">point far from the rest</text>
<line x1="90" y1="186" x2="122" y2="186" class="stroke-a" stroke-width="2.5"/><text x="130" y="190" class="fig-small">two typical points</text>
</svg>
Figure 3. Leverage against the ridge penalty for 8 points and 10 features. At a tiny penalty every leverage is 1, and the divisor is 0. A larger penalty pulls each leverage down, and the point far from the rest falls last.
```

The leverages also add up to something meaningful. Their sum is the effective number of directions the fit uses, called the degrees of freedom, as you saw in the chapter "Ridge for every lambda: the SVD path and degrees of freedom". In Figure 3 that sum is 8 at a tiny penalty, one per point, and it shrinks as the penalty grows.

> [!KEY]
> A point helps train the model that is later scored on it, so its ordinary residual is too small, and more so when the point has high leverage. For ridge, dividing each residual by one minus its leverage gives the exact leave-one-out residual, with a single fit instead of $n$.

> [!RECAP]
> - The hat matrix $H$ maps the targets to the fitted values, and $h_{ii}$ is the leverage of point $i$.
> - The honest miss at point $i$ is $r_i / (1 - h_{ii})$, which equals refitting without that point.
> - A positive penalty keeps every leverage below 1, so the divisor never reaches 0.
> - The leverages add up to the degrees of freedom.

## Build it step by step

You write three functions. A fourth, `loo_by_refitting`, is already in the file: it refits $n$ times, so you can compare your shortcut with the slow way.

1. **`hat_matrix(X, lam)`**. Takes `X` of shape `(n, d)` and a penalty. Returns `H` of shape `(n, n)`. The common mistake is leaving out `lam * np.eye(d)`. At $\lambda = 1$ the hand example would then divide by 14 instead of 15.

   ```python
   gram = X.T @ X + lam * np.eye(d)   # shape (d, d)
   H = X @ np.linalg.inv(gram) @ X.T  # shape (n, n)
   ```

2. **`leverages(X, lam)`**. Returns the diagonal of `H`, shape `(n,)`. The common mistake is calling `np.diag` on the wrong matrix, such as the `(d, d)` Gram matrix instead of the `(n, n)` hat matrix.

3. **`loo_linear_smoother(X, y, lam)`**. Takes `y` of shape `(n,)`. Returns one float, the mean of the squared honest residuals. The common mistake is to square before dividing, or to divide by `leverages` instead of `1 - leverages`.

   ```python
   residual = y - H @ y
   honest = residual / (1.0 - np.diag(H))
   ```

## Cost and scaling

One ridge fit on $n$ points with $d$ features costs about $n d^2$ work. The methods compare as follows:

| Method | Work | Memory |
|---|---|---|
| Leave-one-out by refitting | $n$ fits, about $n^2 d^2$ | one training set |
| Shortcut with the full $H$ | one fit plus $H$, about $n d^2 + n^2 d$ | $n^2$ entries |
| Shortcut with the diagonal only | one fit, about $n d^2$ | $n$ numbers |

Forming the full $H$ costs more than the fit itself when $n$ is much larger than $d$. You only need the diagonal, and `np.einsum("ij,jk,ik->i", X, inverse, X)` computes it from the inverse in about $n d^2$ work without building the $n$-by-$n$ matrix. The exercise builds the full matrix, because the matrix is what you can inspect.

## Where this breaks

**Models that are not linear in the targets.** The shortcut relies on every fitted value being a weighted sum of the targets. That holds for ridge and other linear smoothers. It fails for trees and neural networks, where you refit or use k-fold.

**Preprocessing on all rows.** If you centre or scale the features with statistics from all $n$ rows, the held-out point has already leaked into the fit, and the shortcut inherits the leak.

**A leverage near 1.** With no penalty and more features than points, the divisor is 0. Use a positive penalty.

## The exercise

You implement `hat_matrix`, `leverages`, and `loo_linear_smoother`. The starter's closing program runs the hand example and then compares your shortcut with `loo_by_refitting` at three penalties on 200 random points.

The checks confirm that `hat_matrix` is symmetric and that `H @ y` reproduces the ridge fitted values, and that with no penalty applying the hat twice changes nothing. They check that the leverages for inputs 1, 2, 3 are $1/15$, $4/15$, and $9/15$, that every leverage lies in $[0, 1)$ with a sum equal to $\sum s^2/(s^2 + \lambda)$, and that `loo_linear_smoother` returns 0.2869 on the hand example and matches $n$ refits at three penalties. The last check confirms that the honest error far exceeds the in-sample error when there are 20 features and 40 points.
