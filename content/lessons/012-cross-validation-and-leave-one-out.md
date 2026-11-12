---
title: "Cross-validation and the leave-one-out shortcut"
slug: 012-cross-validation-and-leave-one-out
part: "Part 2 — Classical models"
summary: Score a model only on houses it never saw, with k-fold cross-validation, then get the leave-one-out score of ridge from a single fit.
minutes: 70
gpu: false
objectives:
  - Explain why training error always prefers the most flexible model, and why one hold-out split is a noisy substitute.
  - Build k-fold cross-validation so every row is scored once by a model that never trained on it, and pick a ridge penalty with it.
  - Build the hat matrix and read each point's leverage from its diagonal.
  - Compute the leave-one-out error of ridge from one fit by dividing each residual by one minus its leverage.
  - Say when the shortcut works, when the divisor misbehaves, and what it saves.
prereqs:
  - 001-loss-and-honest-scores
  - 006-ridge-regression
  - 003-least-squares-and-projection
exercise: 012-cross-validation-and-leave-one-out
---

# Cross-validation and the leave-one-out shortcut

You have 60 houses. For each one you know 30 measurements, such as floor area, rooms, and age, and you know the sale price. You want a ridge model that predicts the price of a house it has not seen. Ridge has one dial, the penalty $\lambda$, and you have 20 candidate settings. Which one do you ship?

The obvious move is to fit all 20 and keep the one with the lowest error on your 60 houses. That fails every time, because the winner is always the smallest penalty. A model that is allowed to bend freely bends toward the noise in the houses it was fitted on, and its training error looks wonderful. The score tells you how well the model remembers, not how well it will predict.

This chapter builds the fix in two moves. First you hide some houses from the model, score it on those, and rotate which houses are hidden so that none of your precious 60 is wasted. That is k-fold cross-validation. Then you push the idea to its limit, one house per fold, and find that for ridge you can get all 60 honest misses from a single fit.

## What you already know

- A model scored on the data it was fitted on gets a flattering score, which is why you hold data back to score it. See [honest scores](/lessons/001-loss-and-honest-scores).
- Mean squared error, or MSE, is the average of the squared gaps between prediction and truth, so gaps of 1 and 3 give $(1 + 9)/2 = 5$.
- Ridge regression charges a fee $\lambda$ for large weights and fits $w = (X^{\top}X + \lambda I)^{-1}X^{\top}y$. A bigger $\lambda$ makes the model stiffer. See [ridge regression](/lessons/006-ridge-regression).
- A residual is the target minus the prediction, so a house priced at 5 and predicted at 4.5 has residual 0.5.
- A fold is one group of rows, and a seeded random generator gives the same shuffle every time you run the code.

## Training error picks the wrong end

Figure 1 shows what happens on the 60 houses. The blue line is the error of each ridge model on the houses it was fitted on. It climbs as the penalty grows, so it always prefers a penalty near 0. The green dashed line is the error on 5000 fresh houses from the same source, which you would never have in practice. It falls, bottoms out between penalties of 8 and 15, and climbs again.

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="Training error rises with the ridge penalty while error on unseen houses falls then rises, and cross-validation follows the unseen-house curve">
<line x1="70" y1="240.0" x2="570" y2="240.0" class="stroke-grid"/>
<text x="58" y="244.0" text-anchor="end" class="fig-small">0</text>
<line x1="70" y1="178.2" x2="570" y2="178.2" class="stroke-grid"/>
<text x="58" y="182.2" text-anchor="end" class="fig-small">1</text>
<line x1="70" y1="116.5" x2="570" y2="116.5" class="stroke-grid"/>
<text x="58" y="120.5" text-anchor="end" class="fig-small">2</text>
<line x1="70" y1="54.7" x2="570" y2="54.7" class="stroke-grid"/>
<text x="58" y="58.7" text-anchor="end" class="fig-small">3</text>
<line x1="70.0" y1="240" x2="70.0" y2="244" class="stroke-muted"/><text x="70.0" y="258" text-anchor="middle" class="fig-small">0.01</text>
<line x1="170.0" y1="240" x2="170.0" y2="244" class="stroke-muted"/><text x="170.0" y="258" text-anchor="middle" class="fig-small">0.1</text>
<line x1="270.0" y1="240" x2="270.0" y2="244" class="stroke-muted"/><text x="270.0" y="258" text-anchor="middle" class="fig-small">1</text>
<line x1="370.0" y1="240" x2="370.0" y2="244" class="stroke-muted"/><text x="370.0" y="258" text-anchor="middle" class="fig-small">10</text>
<line x1="470.0" y1="240" x2="470.0" y2="244" class="stroke-muted"/><text x="470.0" y="258" text-anchor="middle" class="fig-small">100</text>
<line x1="570.0" y1="240" x2="570.0" y2="244" class="stroke-muted"/><text x="570.0" y="258" text-anchor="middle" class="fig-small">1000</text>
<line x1="70" y1="240" x2="570" y2="240" class="stroke-muted"/><line x1="70" y1="30" x2="70" y2="240" class="stroke-muted"/>
<text x="320" y="282" text-anchor="middle">ridge penalty (log scale)</text><text x="14" y="135" transform="rotate(-90 14 135)" text-anchor="middle">squared error</text>
<polyline points="70.0,128.4 96.3,128.5 122.6,128.8 148.9,129.3 175.3,130.1 201.6,131.6 227.9,133.9 254.2,137.5 280.5,142.2 306.8,147.6 333.2,152.4 359.5,155.2 385.8,155.2 412.1,152.6 438.4,148.1 464.7,141.7 491.1,131.8 517.4,114.6 543.7,85.1 570.0,39.4" fill="none" class="stroke-c" stroke-width="2.5" stroke-dasharray="6 4"/>
<polyline points="70.0,143.3 96.3,143.6 122.6,144.1 148.9,144.9 175.3,146.3 201.6,148.6 227.9,152.0 254.2,156.4 280.5,161.3 306.8,165.6 333.2,168.3 359.5,169.1 385.8,168.1 412.1,165.9 438.4,162.8 464.7,158.4 491.1,151.0 517.4,137.5 543.7,114.4 570.0,80.8" fill="none" class="stroke-b" stroke-width="2.5" />
<polyline points="70.0,221.2 96.3,221.2 122.6,221.2 148.9,221.2 175.3,221.2 201.6,221.2 227.9,221.1 254.2,221.0 280.5,220.5 306.8,219.5 333.2,217.4 359.5,213.7 385.8,208.4 412.1,201.8 438.4,194.8 464.7,188.0 491.1,180.5 517.4,169.7 543.7,151.3 570.0,121.3" fill="none" class="stroke-a" stroke-width="2.5" />
<circle cx="359.5" cy="169.1" r="6" class="fill-b"/>
<text x="359.5" y="191.1" text-anchor="middle" class="fig-bold">CV picks 7.8</text>
<circle cx="70.0" cy="221.2" r="6" class="fill-a"/>
<text x="82" y="209.2" class="fig-bold">training error picks 0.01</text>
<line x1="300" y1="40" x2="332" y2="40" class="stroke-a" stroke-width="2.5" /><text x="340" y="44" class="fig-small">training error (the 60 houses)</text>
<line x1="300" y1="58" x2="332" y2="58" class="stroke-b" stroke-width="2.5" /><text x="340" y="62" class="fig-small">5-fold cross-validation (the 60 houses)</text>
<line x1="300" y1="76" x2="332" y2="76" class="stroke-c" stroke-width="2.5" stroke-dasharray="6 4"/><text x="340" y="80" class="fig-small">error on 5000 unseen houses</text>
</svg>
Figure 1. Training error (blue) is lowest at the smallest penalty, but error on unseen houses (green, dashed) is lowest between 8 and 15. Five-fold cross-validation (orange) uses only the 60 houses and still finds the bottom of the green bowl.
```

Look at the left edge. At a penalty of 0.01 the training error is 0.30, and the error on unseen houses is 1.81, six times larger. At a penalty near 8 the training error is worse, 0.43, but the fresh error is only 1.37. The model with the better training score is the worse predictor. Training error cannot see overfitting, because overfitting is exactly the habit of doing well on the houses you trained on.

So you need a score from houses the model did not train on. The orange line is that score, and the next sections show how to compute it.

## Why not hold out 20% once?

The simplest honest score is a hold-out split. Shuffle the 60 houses, set 12 aside, fit on the other 48, and score on the 12. Its weakness is that the score depends on which 12 houses you happened to set aside. Repeat the experiment with five different shuffles and the five splits choose penalties of 2.3, 14, 48, 162, and 545. A range that wide is not a measurement of the penalty. It is a measurement of which houses landed in the 12.

A hold-out split also wastes the 12 houses, a fifth of everything you know. The cure for both problems is many splits, with every house playing both roles.

## Let every house sit out once

Picture the 60 houses cut into five piles of 12, called *folds*. In round 1, fold 1 is hidden. You fit on the other four folds and score on fold 1. In round 2, fold 2 is hidden, and so on. After five rounds every house has been hidden exactly once, so every house has been scored by a model that never saw it. This is *k-fold cross-validation*, with $k = 5$.

Here is a case small enough to do by hand. Six houses have prices 3, 5, 4, 8, 6, and 9, in hundreds of thousands. The model is as plain as possible: it predicts the average training price. Suppose the shuffle gives three folds of two houses: houses 1 and 4, houses 2 and 6, and houses 3 and 5.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Three rounds of cross-validation on six houses: each round holds out a different pair of houses">
<text x="98" y="30" text-anchor="end" class="fig-small">house</text><text x="98" y="48" text-anchor="end" class="fig-small">price</text>
<text x="139.0" y="30" text-anchor="middle" class="fig-small">1</text><text x="139.0" y="48" text-anchor="middle" class="fig-bold">3</text>
<text x="197.0" y="30" text-anchor="middle" class="fig-small">2</text><text x="197.0" y="48" text-anchor="middle" class="fig-bold">5</text>
<text x="255.0" y="30" text-anchor="middle" class="fig-small">3</text><text x="255.0" y="48" text-anchor="middle" class="fig-bold">4</text>
<text x="313.0" y="30" text-anchor="middle" class="fig-small">4</text><text x="313.0" y="48" text-anchor="middle" class="fig-bold">8</text>
<text x="371.0" y="30" text-anchor="middle" class="fig-small">5</text><text x="371.0" y="48" text-anchor="middle" class="fig-bold">6</text>
<text x="429.0" y="30" text-anchor="middle" class="fig-small">6</text><text x="429.0" y="48" text-anchor="middle" class="fig-bold">9</text>
<text x="488" y="48" class="fig-small">held-out MSE</text>
<text x="98" y="88.0" text-anchor="end">round 1</text>
<rect x="112" y="62" width="54" height="44" rx="4" class="fill-soft-b stroke-b" stroke-width="2.5"/>
<text x="139.0" y="88.0" text-anchor="middle" class="fig-bold">held out</text>
<rect x="170" y="62" width="54" height="44" rx="4" class="fill-soft-a"/>
<text x="197.0" y="88.0" text-anchor="middle" class="fig-small">train</text>
<rect x="228" y="62" width="54" height="44" rx="4" class="fill-soft-a"/>
<text x="255.0" y="88.0" text-anchor="middle" class="fig-small">train</text>
<rect x="286" y="62" width="54" height="44" rx="4" class="fill-soft-b stroke-b" stroke-width="2.5"/>
<text x="313.0" y="88.0" text-anchor="middle" class="fig-bold">held out</text>
<rect x="344" y="62" width="54" height="44" rx="4" class="fill-soft-a"/>
<text x="371.0" y="88.0" text-anchor="middle" class="fig-small">train</text>
<rect x="402" y="62" width="54" height="44" rx="4" class="fill-soft-a"/>
<text x="429.0" y="88.0" text-anchor="middle" class="fig-small">train</text>
<text x="488" y="88.0" class="fig-mono">6.5</text>
<text x="98" y="140.0" text-anchor="end">round 2</text>
<rect x="112" y="114" width="54" height="44" rx="4" class="fill-soft-a"/>
<text x="139.0" y="140.0" text-anchor="middle" class="fig-small">train</text>
<rect x="170" y="114" width="54" height="44" rx="4" class="fill-soft-b stroke-b" stroke-width="2.5"/>
<text x="197.0" y="140.0" text-anchor="middle" class="fig-bold">held out</text>
<rect x="228" y="114" width="54" height="44" rx="4" class="fill-soft-a"/>
<text x="255.0" y="140.0" text-anchor="middle" class="fig-small">train</text>
<rect x="286" y="114" width="54" height="44" rx="4" class="fill-soft-a"/>
<text x="313.0" y="140.0" text-anchor="middle" class="fig-small">train</text>
<rect x="344" y="114" width="54" height="44" rx="4" class="fill-soft-a"/>
<text x="371.0" y="140.0" text-anchor="middle" class="fig-small">train</text>
<rect x="402" y="114" width="54" height="44" rx="4" class="fill-soft-b stroke-b" stroke-width="2.5"/>
<text x="429.0" y="140.0" text-anchor="middle" class="fig-bold">held out</text>
<text x="488" y="140.0" class="fig-mono">7.0625</text>
<text x="98" y="192.0" text-anchor="end">round 3</text>
<rect x="112" y="166" width="54" height="44" rx="4" class="fill-soft-a"/>
<text x="139.0" y="192.0" text-anchor="middle" class="fig-small">train</text>
<rect x="170" y="166" width="54" height="44" rx="4" class="fill-soft-a"/>
<text x="197.0" y="192.0" text-anchor="middle" class="fig-small">train</text>
<rect x="228" y="166" width="54" height="44" rx="4" class="fill-soft-b stroke-b" stroke-width="2.5"/>
<text x="255.0" y="192.0" text-anchor="middle" class="fig-bold">held out</text>
<rect x="286" y="166" width="54" height="44" rx="4" class="fill-soft-a"/>
<text x="313.0" y="192.0" text-anchor="middle" class="fig-small">train</text>
<rect x="344" y="166" width="54" height="44" rx="4" class="fill-soft-b stroke-b" stroke-width="2.5"/>
<text x="371.0" y="192.0" text-anchor="middle" class="fig-bold">held out</text>
<rect x="402" y="166" width="54" height="44" rx="4" class="fill-soft-a"/>
<text x="429.0" y="192.0" text-anchor="middle" class="fig-small">train</text>
<text x="488" y="192.0" class="fig-mono">2.5625</text>
<text x="488" y="236.0" class="fig-bold fig-mono">avg 5.375</text>
<text x="110" y="236.0" class="fig-small">each column is held out in exactly one round</text>
</svg>
Figure 2. Each round hides a different pair of houses (orange) and trains on the other four (blue). Every column is orange exactly once, so every price is predicted once by a model that never saw it.
```

Take the three rounds in Figure 2 one at a time:

- Round 1 hides prices 3 and 8. The training prices are 5, 4, 6, and 9, with average 6. The gaps are $-3$ and $2$, so the squared gaps are 9 and 4 and the MSE is 6.5.
- Round 2 hides prices 5 and 9. The training average is $(3 + 4 + 8 + 6)/4 = 5.25$. The squared gaps are 0.0625 and 14.0625, so the MSE is 7.0625.
- Round 3 hides prices 4 and 6. The training average is $(3 + 5 + 8 + 9)/4 = 6.25$. The squared gaps are 5.0625 and 0.0625, so the MSE is 2.5625.

The cross-validation score is the average of the three, $(6.5 + 7.0625 + 2.5625)/3 = 5.375$. Compare that with the training error of the same model fitted on all six houses, which predicts 5.83 and scores 4.47. The honest score is higher, as it should be, because the training score had seen the answers.

As a formula, with $\text{MSE}_f$ the error on fold $f$ of the model that never saw fold $f$:

$$
\text{CV} = \frac{1}{k}\sum_{f=1}^{k} \text{MSE}_f
$$

Read it aloud: for each fold, train on the rest and measure the squared error on the fold, then average the $k$ results. The hand example gives $5.375$. When folds have unequal sizes, this plain average weights a small fold as much as a large one, so it differs slightly from the average over all held-out houses. For equal folds the two agree.

## Every penalty gets the same folds

To choose a penalty you compute this score once per candidate, and the candidate with the lowest average wins. The one rule that catches people is that every candidate must be scored on the same folds. The shuffle itself adds noise. At the best penalty, five-fold cross-validation returns anything from 0.85 to 1.22 across 60 different shuffles, and that spread is as large as the gap between neighbouring penalties. If each penalty drew its own folds, the comparison would mix the quality of the penalty with the luck of its draw. A fixed seed inside the scoring function removes the luck, so every penalty faces the same exam.

> [!WARNING]
> Create the random generator inside `cross_val_score`, with a fixed seed, instead of taking it from the caller. Otherwise two penalties see different folds, and the comparison measures luck.

## Choosing k

Each round trains on $(k-1)/k$ of the rows. At $k = 5$ that is 48 of the 60 houses, a little fewer than the 60 the final model will use. A model with less data does a little worse, so the cross-validation score runs a bit high, or *pessimistic*. A larger $k$ shrinks that gap, since $k = 10$ trains on 54 houses, but it costs $k$ fits instead of 5. Five or ten folds are the usual compromise. In Figure 1 the five-fold minimum is 1.15 and the unseen-house error at the same penalty is 1.37. The minimum of 20 noisy estimates leans toward luck, and the [next chapter](/lessons/014-learning-curves-and-winners-curse) shows how large that lean can be.

> [!KEY]
> Training error always prefers the most flexible model, so it cannot choose a penalty. Cross-validation scores every house once, with a model that never saw it, and the penalty with the lowest average is the one to ship. Score every penalty on the same folds.

## The extreme: one house per fold

Now push $k$ all the way to $n$. Hide one house, fit on the other 59, and measure the miss on the hidden house. Repeat for each of the 60 houses and average the squared misses. This is *leave-one-out cross-validation*. It trains on almost all your data each time, so it is the least pessimistic of the fold schemes.

The cost is the problem. Sixty houses mean sixty fits, and a million houses mean a million fits. If each fit takes a second, you wait eleven days to score one penalty, and you have twenty penalties to try. For ridge you do not have to refit at all. One fit on all the houses already contains the 60 honest misses, and a small correction recovers them exactly.

### The ordinary residual flatters the model

Figure 3 shows six houses, at floor areas 0.5, 1, 1.5, 2, 2.5, and 5.5, and a ridge line through the origin with penalty $\lambda = 1$. Fitted on all six, the line has slope 0.70, and it misses the far house by 0.54. Remove that point and refit on the other five. The slope drops to 0.50, and that line misses the far point by 1.64, three times as much.

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
Figure 3. The far point pulls the line toward itself. Fitted with the point, the line misses it by 0.54 (orange). Fitted without it, the line misses it by 1.64 (purple, dashed), which is the honest leave-one-out miss.
```

The reason is plain once you see it. The fit is chosen to make squared misses small, and the far point has a loud voice because it sits so far out that tilting the line moves the prediction there a lot. So the line tilts toward it, and the miss at that point looks small, although the model would have missed by far more if it had never seen the point. A point near the middle has a quieter voice, so its residual stays close to its honest value. The gap between the ordinary residual and the honest one therefore depends on how strongly each point pulls on its own prediction. Any correction has to measure that pull.

### Measure how much a point pulls on itself

Ridge predictions are linear in the targets: every fitted value is a weighted sum of all $n$ targets. Collect the weights in an $n$-by-$n$ matrix $H$, so the fitted values are $\hat{y} = Hy$. People call $H$ the *hat matrix*, because it puts a hat on $y$. Row $i$ lists how much each target contributes to the fit at point $i$.

$$
H = X (X^{\top}X + \lambda I)^{-1} X^{\top}
$$

Read it aloud: take the ridge weights, $(X^{\top}X + \lambda I)^{-1}X^{\top}y$, and multiply by $X$ to get predictions, and everything in front of $y$ is $H$. In numpy it is `X @ np.linalg.inv(X.T @ X + lam * np.eye(d)) @ X.T`.

The diagonal entry $h_{ii}$ is the weight of target $i$ in its own fitted value. Call it the *leverage* of point $i$. It sits between two extremes:

- If $h_{ii} = 0$, the point has no say in its own prediction. Its ordinary residual is already honest, because the fit would be identical without it.
- If $h_{ii}$ is close to 1, the fit passes almost exactly through the point. Its ordinary residual is close to 0 whatever the target is, which tells you nothing about how the model does on unseen data.

For the inputs $1, 2, 3$ with $\lambda = 1$ there is one column, and $X^{\top}X + \lambda = 1 + 4 + 9 + 1 = 15$. Entry $(i, j)$ of $H$ is $x_i x_j / 15$, so the leverages are $1/15$, $4/15$, and $9/15$. The point with the largest input has the most pull, as in Figure 3.

### Undo the pull, and check by hand

The ordinary residual is too small by a factor that depends on the pull. The fix divides it by one minus the leverage:

$$
y_i - \hat{y}_i^{(-i)} = \frac{r_i}{1 - h_{ii}}, \qquad r = y - Hy
$$

Here $\hat{y}_i^{(-i)}$ is the prediction for point $i$ from a fit that never saw point $i$. Read it aloud: the honest miss equals the ordinary miss, blown up by $1/(1 - h_{ii})$. When $h_{ii}$ is 0 nothing changes. When $h_{ii}$ is close to 1 the divisor is tiny and the nearly zero residual is blown back up, which is what you want, because a model that has memorised a point says nothing about its honest miss.

Test it on the targets $1, 2, 2$, with inputs $1, 2, 3$ and $\lambda = 1$. The weight is $w = 11/15$, since $X^{\top}y = 1 + 4 + 6 = 11$. The ordinary residuals are $4/15$, $8/15$, and $-1/5$. Dividing each by one minus its leverage and refitting without the point give the same answers:

| Point | Residual $r_i$ | Leverage | $r_i / (1 - h_{ii})$ | Weight without the point | Honest miss from refitting |
|---|---|---|---|---|---|
| 1 | $4/15$ | $1/15$ | $2/7$ | $10/14$ | $1 - 5/7 = 2/7$ |
| 2 | $8/15$ | $4/15$ | $8/11$ | $7/11$ | $2 - 14/11 = 8/11$ |
| 3 | $-1/5$ | $9/15$ | $-1/2$ | $5/6$ | $2 - 5/2 = -1/2$ |

The two right-hand columns agree in every row, and the last point, with the most leverage, is corrected the most. The mean of the squared honest misses is $(4/49 + 64/121 + 1/4)/3 = 0.2869$, while the same average over the ordinary residuals is only $0.1319$.

Why does the division work? Suppose you replace the target $y_i$ by the prediction that the fit without point $i$ makes there. That fit already agrees with the replaced point perfectly, so adding the point changes nothing, and the fit stays the same. Because ridge is linear, replacing $y_i$ by something else shifts the prediction at point $i$ by $h_{ii}$ times the change. Write $\hat{y}_i$ for the ordinary prediction and $\hat{y}_i^{(-i)}$ for the honest one. Linearity says the new prediction at point $i$ is $\hat{y}_i + h_{ii}(\hat{y}_i^{(-i)} - y_i)$. The stays-the-same fact says it is also $\hat{y}_i^{(-i)}$. Setting the two equal and solving for $y_i - \hat{y}_i^{(-i)}$ gives $(y_i - \hat{y}_i)/(1 - h_{ii})$, which is the formula above.

In numpy the shortcut is three lines, with `lam * np.eye(d)` as the term people forget:

```python
H = hat_matrix(X, lam)                  # shape (n, n)
residual = y - H @ y                    # ordinary residuals, shape (n,)
honest = residual / (1.0 - np.diag(H))  # leave-one-out residuals
```

> [!WARNING]
> With no penalty and more features than points, the model passes through every point, every leverage is 1, and the formula divides 0 by 0. A positive penalty keeps every leverage strictly below 1, so the shortcut stays safe. The leverages also add up to the degrees of freedom you met in [ridge regression](/lessons/006-ridge-regression), which is 8 for 8 points at a tiny penalty and shrinks as the penalty grows.

> [!KEY]
> A point helps train the model that is later scored on it, so its ordinary residual is too small, and more so when the point has high leverage. For ridge, dividing each residual by one minus its leverage gives the exact leave-one-out residual, with one fit instead of $n$.

> [!RECAP]
> - Training error always prefers the smallest penalty, and one hold-out split depends on which houses land in it.
> - In k-fold cross-validation each fold sits out once, the score is the average held-out MSE, and every penalty is scored on the same folds.
> - The hat matrix $H$ maps the targets to the fitted values, and $h_{ii}$ is the leverage of point $i$.
> - The honest miss at point $i$ is $r_i / (1 - h_{ii})$, which equals refitting without that point.

## Build it step by step

You write six functions, in the order the checks test them. A seventh, `loo_by_refitting`, is already in the file: it refits $n$ times, so you can compare your shortcut with the slow way.

1. **`k_fold_indices(n, k, rng)`**. Takes the number of rows, the number of folds, and a seeded generator. Returns a list of `k` sorted integer arrays that partition `0` to `n - 1`, with sizes that differ by at most one. You need `rng.permutation`, `np.array_split`, and `np.sort`. The common mistake is `np.split`, which raises an error when `k` does not divide `n`.

   ```python
   chunks = np.array_split(rng.permutation(n), k)  # k arrays
   ```

2. **`cross_val_score(fit, predict, X, y, k)`**. Takes `X` of shape `(n, d)`, `y` of shape `(n,)`, and two callables, `fit(X_train, y_train)` and `predict(model, X_test)`. Returns the `k` held-out MSE values, shape `(k,)`. Create `np.random.default_rng(0)` inside the function. The common mistake is to take the generator as an argument, so two penalties see different folds.

   ```python
   model = fit(X[train], y[train])
   mse = np.mean((predict(model, X[test]) - y[test]) ** 2)
   ```

3. **`best_lambda_by_cv(X, y, lambdas, k)`**. Fits ridge with a different penalty for each entry of `lambdas`, scores each with `cross_val_score`, and averages the `k` values. Returns the best penalty and the array of `len(lambdas)` means. The common mistake is the loop variable: a lambda that refers to `lam` directly sees only the last penalty once the loop ends. Bind it as a default argument.

   ```python
   fit = lambda Xt, yt, lam=lam: np.linalg.solve(Xt.T @ Xt + lam * np.eye(d), Xt.T @ yt)
   means[j] = cross_val_score(fit, lambda w, Xs: Xs @ w, X, y, k).mean()
   ```

4. **`hat_matrix(X, lam)`**. Takes `X` of shape `(n, d)` and a penalty. Returns `H` of shape `(n, n)`. The common mistake is leaving out `lam * np.eye(d)`. At $\lambda = 1$ the hand example would then divide by 14 instead of 15.

5. **`leverages(X, lam)`**. Returns the diagonal of `H`, shape `(n,)`. The common mistake is calling `np.diag` on the wrong matrix, such as the `(d, d)` Gram matrix instead of the `(n, n)` hat matrix.

6. **`loo_linear_smoother(X, y, lam)`**. Takes `y` of shape `(n,)`. Returns one float, the mean of the squared honest residuals. The common mistake is to square before dividing, or to divide by `leverages` instead of `1 - leverages`.


## Cost and scaling

Cross-validation with $k$ folds costs $k$ fits, each on $(k-1)/k$ of the data. Leave-one-out by refitting costs $n$ fits, about $n^2 d^2$ work in total. The shortcut costs one fit, about $n d^2$ work, plus the $n$-by-$n$ hat matrix if you build it, which costs more than the fit when $n$ is much larger than $d$. You only need the diagonal, and `np.einsum("ij,jk,ik->i", X, inverse, X)` computes it from the inverse without building the matrix. The exercise builds the full matrix, because the matrix is what you can inspect.

## Where this breaks

**Rows that are not exchangeable.** Shuffling assumes any house could equally have landed in any fold. Prices over time, or several scans of one patient, break that. A random split trains on the future and tests on the past, or puts a patient on both sides, and the score is too good. Split a series forward in time, and keep each patient in one fold.

**Reusing the score.** The lowest of 20 cross-validation scores is itself a choice made with the data. Keep a final test set for one use, after you choose.

**Models that are not linear in the targets.** The shortcut relies on every fitted value being a weighted sum of the targets. That holds for ridge and other linear smoothers. It fails for trees and neural networks, where you refit or use k-fold.

**Preprocessing on all rows.** Centring or scaling with statistics from all $n$ rows leaks the held-out point into the fit, and both methods inherit the leak.

Preprocessing fitted on all rows leaks into every fold; [feature pipelines and leakage](/lessons/013-feature-pipelines-and-leakage) fits it inside each fold.

## The exercise

You implement `k_fold_indices`, `cross_val_score`, `best_lambda_by_cv`, `hat_matrix`, `leverages`, and `loo_linear_smoother`. The starter's closing program prints the penalty with the lowest training error beside the penalty with the lowest five-fold error on the 60 houses, then compares your shortcut with `loo_by_refitting`.

The checks confirm that the folds cover every row once, that `cross_val_score` repeats from call to call and never leaks a held-out row into training, and that the penalty from `best_lambda_by_cv` beats both $\lambda = 0.01$ and $\lambda = 1000$ on 5000 fresh houses. The last three confirm that the leverages for inputs 1, 2, 3 are $1/15$, $4/15$, and $9/15$, and that `loo_linear_smoother` returns 0.2869 on the hand example and matches $n$ refits at three penalties.
