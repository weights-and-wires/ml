---
title: Why training error lies, and how k-fold cross-validation fixes it
slug: 022-cross-validation
part: "Part 2 — Classical models"
summary: Score a model only on houses it never saw by letting each fold of the data sit out once, then use those scores to choose a ridge penalty.
minutes: 40
gpu: false
objectives:
  - Explain why training error always prefers the most flexible model, and why a single hold-out split is a noisy substitute.
  - Build k-fold cross-validation so every row is scored once by a model that never trained on it.
  - Pick a ridge penalty by comparing every candidate on the same folds.
  - Say what choosing k trades away, and why every candidate must share one set of folds.
prereqs:
  - 002-honest-scores
  - 011-ridge-penalty-and-closed-form
exercise: 022-cross-validation
---

# Why training error lies, and how k-fold cross-validation fixes it

You have 60 houses. For each one you know 30 measurements, such as floor area, rooms, and age, and you know the sale price. You want a ridge model that predicts the price of a house it has not seen. Ridge has one dial, the penalty $\lambda$, and you have 20 candidate settings. Which one do you ship?

The obvious move is to fit all 20 and keep the one with the lowest error on your 60 houses. That fails every time, because the winner is always the smallest penalty. A model that is allowed to bend freely bends toward the noise in the houses it was fitted on, and its training error looks wonderful. The score tells you how well the model remembers, not how well it will predict.

This chapter builds the fix. You hide some houses from the model, score it on those, and rotate which houses are hidden so that none of your precious 60 is wasted.

## What you already know

- A model scored on the data it was fitted on gets a flattering score, which is why you hold data back to score it.
- Mean squared error, or MSE, is the average of the squared gaps between prediction and truth, so gaps of 1 and 3 give $(1 + 9)/2 = 5$.
- Ridge regression charges a fee $\lambda$ for large weights and fits $w = (X^{\top}X + \lambda I)^{-1}X^{\top}y$. A bigger $\lambda$ makes the model stiffer.
- A fold is one group of rows, and a seeded random generator gives the same shuffle every time you run the code.
- A model that fits its training houses perfectly can still be far off on the next house.

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

So you need a score from houses the model did not train on. The orange line is that score, and the rest of the chapter is how to compute it.

## Why not hold out 20% once?

The simplest honest score is a hold-out split. Shuffle the 60 houses, set 12 aside, fit on the other 48, and score on the 12. This works, and it is what you did in the chapter on honest scores. Its weakness is that the score depends on which 12 houses you happened to set aside.

Figure 2 repeats that experiment five times, with five different shuffles. Each grey curve is one split's held-out error across the 20 penalties, and the dot marks the penalty that split would choose. The five splits choose penalties of 2.3, 14, 48, 162, and 545. A range that wide is not a measurement of the penalty. It is a measurement of which houses landed in the 12.

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="Five hold-out curves with five different minima, and the smoother five-fold average">
<line x1="70" y1="240.0" x2="570" y2="240.0" class="stroke-grid"/><text x="58" y="244.0" text-anchor="end" class="fig-small">0</text>
<line x1="70" y1="178.2" x2="570" y2="178.2" class="stroke-grid"/><text x="58" y="182.2" text-anchor="end" class="fig-small">1</text>
<line x1="70" y1="116.5" x2="570" y2="116.5" class="stroke-grid"/><text x="58" y="120.5" text-anchor="end" class="fig-small">2</text>
<line x1="70" y1="54.7" x2="570" y2="54.7" class="stroke-grid"/><text x="58" y="58.7" text-anchor="end" class="fig-small">3</text>
<line x1="70.0" y1="240" x2="70.0" y2="244" class="stroke-muted"/><text x="70.0" y="258" text-anchor="middle" class="fig-small">0.01</text>
<line x1="170.0" y1="240" x2="170.0" y2="244" class="stroke-muted"/><text x="170.0" y="258" text-anchor="middle" class="fig-small">0.1</text>
<line x1="270.0" y1="240" x2="270.0" y2="244" class="stroke-muted"/><text x="270.0" y="258" text-anchor="middle" class="fig-small">1</text>
<line x1="370.0" y1="240" x2="370.0" y2="244" class="stroke-muted"/><text x="370.0" y="258" text-anchor="middle" class="fig-small">10</text>
<line x1="470.0" y1="240" x2="470.0" y2="244" class="stroke-muted"/><text x="470.0" y="258" text-anchor="middle" class="fig-small">100</text>
<line x1="570.0" y1="240" x2="570.0" y2="244" class="stroke-muted"/><text x="570.0" y="258" text-anchor="middle" class="fig-small">1000</text>
<line x1="70" y1="240" x2="570" y2="240" class="stroke-muted"/><line x1="70" y1="30" x2="70" y2="240" class="stroke-muted"/>
<text x="320" y="282" text-anchor="middle">ridge penalty (log scale)</text><text x="14" y="135" transform="rotate(-90 14 135)" text-anchor="middle">held-out squared error</text>
<polyline points="70.0,158.2 96.3,158.4 122.6,158.8 148.9,159.4 175.3,160.4 201.6,162.0 227.9,164.2 254.2,166.7 280.5,168.6 306.8,169.0 333.2,167.0 359.5,162.7 385.8,156.8 412.1,150.2 438.4,143.2 464.7,134.7 491.1,122.1 517.4,100.2 543.7,64.4 570.0,33.1" fill="none" class="stroke-muted" stroke-width="2.5" />
<circle cx="306.8" cy="169.0" r="5" class="fill-muted"/>
<polyline points="70.0,160.6 96.3,160.6 122.6,160.7 148.9,160.7 175.3,160.7 201.6,160.5 227.9,159.9 254.2,158.5 280.5,156.4 306.8,154.0 333.2,152.1 359.5,151.4 385.8,152.3 412.1,155.1 438.4,160.3 464.7,168.2 491.1,178.1 517.4,187.9 543.7,193.7 570.0,192.0" fill="none" class="stroke-muted" stroke-width="2.5" />
<circle cx="543.7" cy="193.7" r="5" class="fill-muted"/>
<polyline points="70.0,160.7 96.3,160.7 122.6,160.9 148.9,161.1 175.3,161.5 201.6,162.2 227.9,163.6 254.2,166.0 280.5,169.6 306.8,173.8 333.2,177.7 359.5,180.3 385.8,181.3 412.1,181.0 438.4,180.2 464.7,179.5 491.1,178.7 517.4,176.6 543.7,171.4 570.0,162.4" fill="none" class="stroke-muted" stroke-width="2.5" />
<circle cx="385.8" cy="181.3" r="5" class="fill-muted"/>
<polyline points="70.0,129.3 96.3,129.2 122.6,128.9 148.9,128.5 175.3,127.7 201.6,126.4 227.9,124.6 254.2,122.3 280.5,120.1 306.8,118.7 333.2,118.3 359.5,118.8 385.8,120.5 412.1,124.3 438.4,130.3 464.7,137.9 491.1,144.2 517.4,143.6 543.7,129.4 570.0,98.5" fill="none" class="stroke-muted" stroke-width="2.5" />
<circle cx="491.1" cy="144.2" r="5" class="fill-muted"/>
<polyline points="70.0,159.8 96.3,159.9 122.6,160.2 148.9,160.7 175.3,161.6 201.6,163.0 227.9,165.3 254.2,168.7 280.5,173.5 306.8,179.6 333.2,186.5 359.5,193.3 385.8,198.9 412.1,202.8 438.4,204.8 464.7,204.5 491.1,201.0 517.4,191.3 543.7,170.5 570.0,134.8" fill="none" class="stroke-muted" stroke-width="2.5" />
<circle cx="438.4" cy="204.8" r="5" class="fill-muted"/>
<polyline points="70.0,143.3 96.3,143.6 122.6,144.1 148.9,144.9 175.3,146.3 201.6,148.6 227.9,152.0 254.2,156.4 280.5,161.3 306.8,165.6 333.2,168.3 359.5,169.1 385.8,168.1 412.1,165.9 438.4,162.8 464.7,158.4 491.1,151.0 517.4,137.5 543.7,114.4 570.0,80.8" fill="none" class="stroke-b" stroke-width="2.5" />
<circle cx="359.5" cy="169.1" r="6" class="fill-b"/>
<line x1="90" y1="40" x2="122" y2="40" class="stroke-muted" stroke-width="2.5"/><text x="130" y="44" class="fig-small">one 12-house hold-out split (dot: its best penalty)</text>
<line x1="90" y1="58" x2="122" y2="58" class="stroke-b" stroke-width="2.5"/><text x="130" y="62" class="fig-small">5-fold cross-validation average</text>
</svg>
Figure 2. Five different 12-house hold-out splits (grey) pick penalties from 2.3 to 545, because each curve is dominated by which few houses it holds out. The orange five-fold average is steadier and picks 7.8.
```

A hold-out split also wastes data in a second way. The 12 houses never help train the model you ship, and with only 60 houses that is a fifth of everything you know. Both problems have the same cure: use many splits, and let every house play both roles.

## Let every house sit out once

Picture the 60 houses cut into five piles of 12, called *folds*. In round 1, fold 1 is hidden. You fit on the other four folds and score on fold 1. In round 2, fold 2 is hidden, and so on. After five rounds every house has been hidden exactly once, so every house has been scored by a model that never saw it. This is *k-fold cross-validation*, with $k = 5$.

Here is a case small enough to do by hand. Six houses have prices 3, 5, 4, 8, 6, and 9, in hundreds of thousands. The model is as plain as possible: it predicts the average training price. Suppose the shuffle gives three folds of two houses, houses 1 and 4, houses 2 and 6, and houses 3 and 5.

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
Figure 3. Each round hides a different pair of houses (orange) and trains on the other four (blue). Every column is orange exactly once, so every price is predicted once by a model that never saw it.
```

Look at the three rounds in Figure 3 one at a time:

- Round 1 hides prices 3 and 8. The training prices are 5, 4, 6, and 9, with average 6. The gaps are $-3$ and $2$, so the squared gaps are 9 and 4 and the MSE is 6.5.
- Round 2 hides prices 5 and 9. The training average is $(3 + 4 + 8 + 6)/4 = 5.25$. The squared gaps are 0.0625 and 14.0625, so the MSE is 7.0625.
- Round 3 hides prices 4 and 6. The training average is $(3 + 5 + 8 + 9)/4 = 6.25$. The squared gaps are 5.0625 and 0.0625, so the MSE is 2.5625.

The cross-validation score is the average of the three, $(6.5 + 7.0625 + 2.5625)/3 = 5.375$. Compare that with the training error of the same model fitted on all six houses, which predicts 5.83 and scores 4.47. The honest score is higher, as it should be, because the training score had seen the answers.

As a formula, with $\text{MSE}_f$ the error on fold $f$ of the model that never saw fold $f$:

$$
\text{CV} = \frac{1}{k}\sum_{f=1}^{k} \text{MSE}_f
$$

Read it aloud: for each fold, train on the rest and measure the squared error on the fold, then average the $k$ results. The hand example gives $5.375$. When folds have unequal sizes, this plain average of fold scores weights a small fold as much as a large one, so it differs slightly from the average over all held-out houses. For equal folds the two agree.

In numpy the folds come from a shuffle and a cut, and each round is a fit and a score:

```python
order = rng.permutation(n)               # shuffle the row numbers once
folds = np.array_split(order, k)         # k pieces; sizes differ by at most one
train = np.setdiff1d(np.arange(n), test) # every row not in the held-out fold
mse = np.mean((predict(model, X[test]) - y[test]) ** 2)
```

## Every penalty gets the same folds

To choose a penalty you compute this score once per candidate, and the candidate with the lowest average wins. The one rule that catches people is that every candidate must be scored on the same folds. The shuffle itself adds noise. At the best penalty, five-fold cross-validation returns anything from 0.85 to 1.22 across 60 different shuffles, depending only on which shuffle you use, and that spread is as large as the gap between neighbouring penalties. If each penalty drew its own folds, the comparison would mix the quality of the penalty with the luck of its draw. A fixed seed inside the scoring function removes the luck, so every penalty faces the same exam.

This is why `cross_val_score` creates its own generator with a fixed seed instead of taking one from the caller.

## Choosing k

Each round trains on $(k-1)/k$ of the rows. At $k = 5$ that is 48 of the 60 houses, a little fewer than the 60 the final model will use. A model with less data does a little worse, so the cross-validation score runs a bit high, or *pessimistic*. A larger $k$ shrinks that gap, since $k = 10$ trains on 54 houses, but it costs $k$ fits instead of 5. At the extreme $k = n$, every house is its own fold, which is called leave-one-out. For ridge it has a shortcut that needs a single fit, and the next chapter builds it.

Five or ten folds are the usual compromise. In Figure 1 the five-fold minimum is 1.15 and the unseen-house error at the same penalty is 1.37. Neither number is exact, because 60 houses give a noisy estimate, and the minimum of 20 noisy estimates leans toward luck. A later chapter, on why the best score among many settings is too good, shows how large that lean can be.

> [!KEY]
> Training error always prefers the most flexible model, so it cannot choose a penalty. Cross-validation scores every house once, with a model that never saw it, and the penalty with the lowest average is the one to ship. Score every penalty on the same folds, or the comparison measures luck.

> [!RECAP]
> - Training error falls as flexibility grows, so it always picks the smallest penalty.
> - A single hold-out split depends on which houses land in it, and it wastes those houses.
> - In k-fold cross-validation each fold sits out once, and the score is the average held-out MSE.
> - Every penalty must be scored on the same folds, so the shuffle seed is fixed inside the scorer.

## Build it step by step

You write three functions, in the order the checks test them.

1. **`k_fold_indices(n, k, rng)`**. Takes the number of rows, the number of folds, and a seeded generator. Returns a list of `k` sorted integer arrays that partition `0` to `n - 1`, with sizes that differ by at most one. You need `rng.permutation`, `np.array_split`, and `np.sort`. The common mistake is `np.split`, which raises an error when `k` does not divide `n`.

   ```python
   order = rng.permutation(n)        # shape (n,)
   chunks = np.array_split(order, k) # k arrays
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

## Where this breaks

**Rows that are not exchangeable.** Shuffling assumes any house could equally have landed in any fold. Prices over time, or several scans of one patient, break that. A random split trains on the future and tests on the past, or puts a patient on both sides, and the score is too good. Split a series forward in time, and keep each patient in one fold.

**Reusing the score.** The lowest of 20 cross-validation scores is itself a choice made with the data. Keep a final test set that you use once, after you have chosen.

**Tiny data.** With a few dozen rows, a fold holds a handful of houses, and one odd house moves the average. Prefer more folds, or a repeated scheme, over trusting one shuffle.

## The exercise

You implement `k_fold_indices`, `cross_val_score`, and `best_lambda_by_cv`. The starter file ends with a short program that runs your code on the 60 houses and prints the penalty with the lowest training error beside the penalty with the lowest five-fold error.

The checks confirm that the folds cover every row once and have sizes 3, 2, and 2 for seven rows in three folds. They confirm that `cross_val_score` returns `k` finite scores that repeat from call to call, that a model predicting 0 scores the mean of $y^2$ because every row is scored once, that no held-out row leaks into training, and that a noise-free linear fit scores 0. The last two checks confirm that `best_lambda_by_cv` returns one mean per penalty, and that its choice beats both $\lambda = 0.01$ and $\lambda = 1000$ on 5000 fresh houses.
