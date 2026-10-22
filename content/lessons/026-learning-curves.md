---
title: Reading a learning curve
slug: 026-learning-curves
part: "Part 2 — Classical models"
summary: Plot training and validation error against the number of training houses to learn whether more data or a different model will lower your error.
minutes: 40
gpu: false
objectives:
  - Explain why training error rises and validation error falls as the training set grows.
  - Build a learning curve from one fixed validation set and nested training sets.
  - Read a curve to decide between more data, a richer model, and a stronger penalty.
  - Turn the reading into a rule that returns underfit, overfit, or good.
prereqs:
  - 004-bias-variance-noise
  - 024-cross-validation
exercise: 026-learning-curves
---

# Reading a learning curve

You have 60 houses with one measurement each, the floor area, and a price. You fit a model, and its error on houses it has not seen is 0.49. That is too high for what you need. Two remedies come to mind. You could collect ten times as many houses, which takes months and costs real money. Or you could change the model, which takes an afternoon. If you pick the wrong one, you spend the months and then see the same 0.49.

You need a way to ask, before you spend anything, what more data would do. You cannot buy the data to find out, but you can ask a cheaper question: how does the score change when the model trains on 10 houses, then 20, then 45? The answer is a *learning curve*. Its shape tells you whether the model is starved for data or limited by its own design.

## What you already know

- Training error is the average loss on the houses a model was fitted on, and validation error is the average loss on houses it never saw.
- A model with few weights can only draw simple shapes, so it misses a curved pattern, which is called underfitting.
- A model with many weights can bend through the noise in its training houses, which is called overfitting.
- Noise is the part of a price that no rule can predict, so even a perfect model misses by that much on average.
- A seeded shuffle gives the same order every time, so a split can be repeated exactly.

## Two models, two shapes

The prices in this chapter follow a curve, the sine of the floor area, plus noise. The noise has a squared size of 0.25, so no model can score below 0.25. We fit two models: a straight line, which has 2 weights, and a degree 8 polynomial, which has 9 weights. Figure 1 trains each on the first 10 houses, then 14, and so on up to 45, and records both errors at every size.

```figure
<svg viewBox="0 0 640 310" role="img" aria-label="Two learning curves. For a straight line the training and validation errors meet at a high level. For a degree 8 curve they start far apart and close slowly.">
<line x1="46" y1="230.0" x2="296" y2="230.0" class="stroke-grid"/><text x="40" y="234.0" text-anchor="end" class="fig-small">0</text>
<line x1="46" y1="185.0" x2="296" y2="185.0" class="stroke-grid"/><text x="40" y="189.0" text-anchor="end" class="fig-small">0.25</text>
<line x1="46" y1="140.0" x2="296" y2="140.0" class="stroke-grid"/><text x="40" y="144.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="46" y1="95.0" x2="296" y2="95.0" class="stroke-grid"/><text x="40" y="99.0" text-anchor="end" class="fig-small">0.75</text>
<line x1="46" y1="50.0" x2="296" y2="50.0" class="stroke-grid"/><text x="40" y="54.0" text-anchor="end" class="fig-small">1</text>
<line x1="53.0" y1="230" x2="53.0" y2="234" class="stroke-muted"/><text x="53.0" y="247" text-anchor="middle" class="fig-small">10</text>
<line x1="120.4" y1="230" x2="120.4" y2="234" class="stroke-muted"/><text x="120.4" y="247" text-anchor="middle" class="fig-small">20</text>
<line x1="187.9" y1="230" x2="187.9" y2="234" class="stroke-muted"/><text x="187.9" y="247" text-anchor="middle" class="fig-small">30</text>
<line x1="289.0" y1="230" x2="289.0" y2="234" class="stroke-muted"/><text x="289.0" y="247" text-anchor="middle" class="fig-small">45</text>
<line x1="46" y1="230" x2="296" y2="230" class="stroke-muted"/><line x1="46" y1="42" x2="46" y2="230" class="stroke-muted"/>
<line x1="46" y1="185.0" x2="296" y2="185.0" class="stroke-muted" stroke-dasharray="5 4"/>
<text x="171.0" y="30" text-anchor="middle" class="fig-bold">straight line: 2 weights</text>
<polyline points="53.0,162.1 80.0,158.4 106.9,152.3 147.4,154.1 187.9,155.1 241.8,152.2 289.0,151.7" fill="none" class="stroke-a" stroke-width="2.5"/>
<circle cx="53.0" cy="162.1" r="3.5" class="fill-a"/>
<circle cx="80.0" cy="158.4" r="3.5" class="fill-a"/>
<circle cx="106.9" cy="152.3" r="3.5" class="fill-a"/>
<circle cx="147.4" cy="154.1" r="3.5" class="fill-a"/>
<circle cx="187.9" cy="155.1" r="3.5" class="fill-a"/>
<circle cx="241.8" cy="152.2" r="3.5" class="fill-a"/>
<circle cx="289.0" cy="151.7" r="3.5" class="fill-a"/>
<polyline points="53.0,125.1 80.0,135.8 106.9,140.0 147.4,139.5 187.9,141.5 241.8,142.3 289.0,142.1" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="53.0" cy="125.1" r="3.5" class="fill-b"/>
<circle cx="80.0" cy="135.8" r="3.5" class="fill-b"/>
<circle cx="106.9" cy="140.0" r="3.5" class="fill-b"/>
<circle cx="147.4" cy="139.5" r="3.5" class="fill-b"/>
<circle cx="187.9" cy="141.5" r="3.5" class="fill-b"/>
<circle cx="241.8" cy="142.3" r="3.5" class="fill-b"/>
<circle cx="289.0" cy="142.1" r="3.5" class="fill-b"/>
<text x="285.0" y="133.1" text-anchor="end" class="fig-small">validation 0.49</text>
<text x="285.0" y="170.7" text-anchor="end" class="fig-small">training 0.44</text>
<line x1="366" y1="230.0" x2="616" y2="230.0" class="stroke-grid"/><text x="360" y="234.0" text-anchor="end" class="fig-small">0</text>
<line x1="366" y1="185.0" x2="616" y2="185.0" class="stroke-grid"/><text x="360" y="189.0" text-anchor="end" class="fig-small">0.25</text>
<line x1="366" y1="140.0" x2="616" y2="140.0" class="stroke-grid"/><text x="360" y="144.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="366" y1="95.0" x2="616" y2="95.0" class="stroke-grid"/><text x="360" y="99.0" text-anchor="end" class="fig-small">0.75</text>
<line x1="366" y1="50.0" x2="616" y2="50.0" class="stroke-grid"/><text x="360" y="54.0" text-anchor="end" class="fig-small">1</text>
<line x1="373.0" y1="230" x2="373.0" y2="234" class="stroke-muted"/><text x="373.0" y="247" text-anchor="middle" class="fig-small">10</text>
<line x1="440.4" y1="230" x2="440.4" y2="234" class="stroke-muted"/><text x="440.4" y="247" text-anchor="middle" class="fig-small">20</text>
<line x1="507.9" y1="230" x2="507.9" y2="234" class="stroke-muted"/><text x="507.9" y="247" text-anchor="middle" class="fig-small">30</text>
<line x1="609.0" y1="230" x2="609.0" y2="234" class="stroke-muted"/><text x="609.0" y="247" text-anchor="middle" class="fig-small">45</text>
<line x1="366" y1="230" x2="616" y2="230" class="stroke-muted"/><line x1="366" y1="42" x2="366" y2="230" class="stroke-muted"/>
<line x1="366" y1="185.0" x2="616" y2="185.0" class="stroke-muted" stroke-dasharray="5 4"/>
<text x="491.0" y="30" text-anchor="middle" class="fig-bold">degree 8 curve: 9 weights</text>
<polyline points="373.0,210.9 400.0,204.6 426.9,197.7 467.4,195.5 507.9,193.6 561.8,190.7 609.0,189.9" fill="none" class="stroke-a" stroke-width="2.5"/>
<circle cx="373.0" cy="210.9" r="3.5" class="fill-a"/>
<circle cx="400.0" cy="204.6" r="3.5" class="fill-a"/>
<circle cx="426.9" cy="197.7" r="3.5" class="fill-a"/>
<circle cx="467.4" cy="195.5" r="3.5" class="fill-a"/>
<circle cx="507.9" cy="193.6" r="3.5" class="fill-a"/>
<circle cx="561.8" cy="190.7" r="3.5" class="fill-a"/>
<circle cx="609.0" cy="189.9" r="3.5" class="fill-a"/>
<polyline points="373.0,88.6 400.0,114.6 426.9,145.9 467.4,158.4 507.9,164.2 561.8,171.5 609.0,171.2" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="373.0" cy="88.6" r="3.5" class="fill-b"/>
<circle cx="400.0" cy="114.6" r="3.5" class="fill-b"/>
<circle cx="426.9" cy="145.9" r="3.5" class="fill-b"/>
<circle cx="467.4" cy="158.4" r="3.5" class="fill-b"/>
<circle cx="507.9" cy="164.2" r="3.5" class="fill-b"/>
<circle cx="561.8" cy="171.5" r="3.5" class="fill-b"/>
<circle cx="609.0" cy="171.2" r="3.5" class="fill-b"/>
<text x="605.0" y="162.2" text-anchor="end" class="fig-small">validation 0.33</text>
<text x="605.0" y="208.9" text-anchor="end" class="fig-small">training 0.22</text>
<text x="171" y="270" text-anchor="middle">houses used for training</text><text x="491" y="270" text-anchor="middle">houses used for training</text>
<text x="320" y="296" text-anchor="middle" class="fig-small">squared error; dashed line: the noise floor of 0.25 that no model can remove</text>
<text x="12" y="140" transform="rotate(-90 12 140)" text-anchor="middle">squared error</text>
</svg>
Figure 1. Learning curves for the two models. The straight line's curves sit close together at about 0.45, well above the noise floor, and more houses do not move them. The degree 8 curve starts with a huge gap, and both curves slide toward each other as houses are added.
```

Each curve is the median of 60 different shuffles of the same 60 houses. A single shuffle gives a wiggly line, because a validation set of 15 houses is small, and the median removes most of that wiggle.

Both panels share a pattern. Training error rises as houses are added, and validation error falls. Here is why. Ten houses are easy to fit, and a flexible model can pass close to every one, so the training error starts low. With 45 houses the model cannot please them all, and the training error climbs. Validation error moves the other way. A model fitted on 10 houses has chased the quirks of those 10, so it does badly on fresh houses, and each extra house averages out some of that chasing. The two curves therefore squeeze toward the same level, which is the best this model can do.

## A tiny curve by hand

You can watch the squeeze with the plainest model there is: predict the average training price. Set aside two validation houses with prices 5 and 9. Shuffle the rest once, which gives prices 3, 7, 4, and 8 in that order. Train on the first 1, then the first 2, then all 4:

| Training houses | Prediction | Training error | Validation error |
|---|---|---|---|
| 1 (price 3) | 3 | 0 | $(4 + 36)/2 = 20$ |
| 2 (prices 3, 7) | 5 | $(4 + 4)/2 = 4$ | $(0 + 16)/2 = 8$ |
| 4 (prices 3, 7, 4, 8) | 5.5 | $(6.25 + 2.25 + 2.25 + 6.25)/4 = 4.25$ | $(0.25 + 12.25)/2 = 6.25$ |

Training error climbs from 0 to 4 to 4.25. A single house is fitted perfectly, because the average of one price is that price. Validation error falls from 20 to 8 to 6.25, and the two columns head toward the same value, the spread of the prices around their average. A model that predicts one number cannot do better than that spread, and the curves show where it lies.

## Read the two shapes

Return to Figure 1 and ask what each panel says about the remedies.

The straight line's curves have already met. Training error is 0.44 and validation error is 0.49, and neither moves as the houses grow from 18 to 45. More data cannot help this model, because it has already seen enough to find its best line, and its best line is still wrong about a curved pattern. The curves sit 0.2 above the noise floor of 0.25, and that distance is the price of drawing a straight line through a curve. The cure is a richer model or better features, such as the square of the floor area.

The degree 8 curve has a different problem. Training error is 0.22, which is below the noise floor, so the model is fitting noise. Validation error is 0.33, so the gap is 0.10. The curves are still closing in on each other as the houses grow, which means more houses would help, and so would a stronger penalty, since both attack the model's habit of chasing noise. A richer model would not help, because this one already has plenty of flexibility. Notice that this model is still better than the straight line, 0.33 against 0.49. A wide gap says what would help this model, not that it is the worse model.

One caution belongs here. In real work you do not know the noise floor of 0.25. A high curve that has flattened tells you that this model has hit its limit, but not whether a richer model would do better, or whether most of the remaining error is noise. The learning curve narrows the choice, and a trial of the richer model settles it.

## Build the curve fairly

A learning curve is only trustworthy when the only thing that changes from one point to the next is the amount of training data. Figure 2 shows the two rules that make that true.

```figure
<svg viewBox="0 0 620 270" role="img" aria-label="One shuffled row of 60 houses: 15 held out for validation, then growing prefixes of the remaining 45 used for training">
<text x="70" y="26" class="fig-small">shuffle once: 60 houses in a row</text>
<text x="60" y="68" text-anchor="end" class="fig-small">train on 10</text>
<rect x="70" y="52" width="7" height="26" class="fill-d"/>
<rect x="78" y="52" width="7" height="26" class="fill-d"/>
<rect x="86" y="52" width="7" height="26" class="fill-d"/>
<rect x="94" y="52" width="7" height="26" class="fill-d"/>
<rect x="102" y="52" width="7" height="26" class="fill-d"/>
<rect x="110" y="52" width="7" height="26" class="fill-d"/>
<rect x="118" y="52" width="7" height="26" class="fill-d"/>
<rect x="126" y="52" width="7" height="26" class="fill-d"/>
<rect x="134" y="52" width="7" height="26" class="fill-d"/>
<rect x="142" y="52" width="7" height="26" class="fill-d"/>
<rect x="150" y="52" width="7" height="26" class="fill-d"/>
<rect x="158" y="52" width="7" height="26" class="fill-d"/>
<rect x="166" y="52" width="7" height="26" class="fill-d"/>
<rect x="174" y="52" width="7" height="26" class="fill-d"/>
<rect x="182" y="52" width="7" height="26" class="fill-d"/>
<rect x="190" y="52" width="7" height="26" class="fill-a"/>
<rect x="198" y="52" width="7" height="26" class="fill-a"/>
<rect x="206" y="52" width="7" height="26" class="fill-a"/>
<rect x="214" y="52" width="7" height="26" class="fill-a"/>
<rect x="222" y="52" width="7" height="26" class="fill-a"/>
<rect x="230" y="52" width="7" height="26" class="fill-a"/>
<rect x="238" y="52" width="7" height="26" class="fill-a"/>
<rect x="246" y="52" width="7" height="26" class="fill-a"/>
<rect x="254" y="52" width="7" height="26" class="fill-a"/>
<rect x="262" y="52" width="7" height="26" class="fill-a"/>
<rect x="270" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="278" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="286" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="294" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="302" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="310" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="318" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="326" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="334" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="342" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="350" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="358" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="366" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="374" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="382" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="390" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="398" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="406" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="414" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="422" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="430" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="438" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="446" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="454" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="462" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="470" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="478" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="486" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="494" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="502" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="510" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="518" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="526" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="534" y="52" width="7" height="26" class="fill-soft-a"/>
<rect x="542" y="52" width="7" height="26" class="fill-soft-a"/>
<text x="60" y="120" text-anchor="end" class="fig-small">train on 18</text>
<rect x="70" y="104" width="7" height="26" class="fill-d"/>
<rect x="78" y="104" width="7" height="26" class="fill-d"/>
<rect x="86" y="104" width="7" height="26" class="fill-d"/>
<rect x="94" y="104" width="7" height="26" class="fill-d"/>
<rect x="102" y="104" width="7" height="26" class="fill-d"/>
<rect x="110" y="104" width="7" height="26" class="fill-d"/>
<rect x="118" y="104" width="7" height="26" class="fill-d"/>
<rect x="126" y="104" width="7" height="26" class="fill-d"/>
<rect x="134" y="104" width="7" height="26" class="fill-d"/>
<rect x="142" y="104" width="7" height="26" class="fill-d"/>
<rect x="150" y="104" width="7" height="26" class="fill-d"/>
<rect x="158" y="104" width="7" height="26" class="fill-d"/>
<rect x="166" y="104" width="7" height="26" class="fill-d"/>
<rect x="174" y="104" width="7" height="26" class="fill-d"/>
<rect x="182" y="104" width="7" height="26" class="fill-d"/>
<rect x="190" y="104" width="7" height="26" class="fill-a"/>
<rect x="198" y="104" width="7" height="26" class="fill-a"/>
<rect x="206" y="104" width="7" height="26" class="fill-a"/>
<rect x="214" y="104" width="7" height="26" class="fill-a"/>
<rect x="222" y="104" width="7" height="26" class="fill-a"/>
<rect x="230" y="104" width="7" height="26" class="fill-a"/>
<rect x="238" y="104" width="7" height="26" class="fill-a"/>
<rect x="246" y="104" width="7" height="26" class="fill-a"/>
<rect x="254" y="104" width="7" height="26" class="fill-a"/>
<rect x="262" y="104" width="7" height="26" class="fill-a"/>
<rect x="270" y="104" width="7" height="26" class="fill-a"/>
<rect x="278" y="104" width="7" height="26" class="fill-a"/>
<rect x="286" y="104" width="7" height="26" class="fill-a"/>
<rect x="294" y="104" width="7" height="26" class="fill-a"/>
<rect x="302" y="104" width="7" height="26" class="fill-a"/>
<rect x="310" y="104" width="7" height="26" class="fill-a"/>
<rect x="318" y="104" width="7" height="26" class="fill-a"/>
<rect x="326" y="104" width="7" height="26" class="fill-a"/>
<rect x="334" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="342" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="350" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="358" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="366" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="374" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="382" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="390" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="398" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="406" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="414" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="422" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="430" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="438" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="446" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="454" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="462" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="470" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="478" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="486" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="494" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="502" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="510" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="518" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="526" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="534" y="104" width="7" height="26" class="fill-soft-a"/>
<rect x="542" y="104" width="7" height="26" class="fill-soft-a"/>
<text x="60" y="172" text-anchor="end" class="fig-small">train on 30</text>
<rect x="70" y="156" width="7" height="26" class="fill-d"/>
<rect x="78" y="156" width="7" height="26" class="fill-d"/>
<rect x="86" y="156" width="7" height="26" class="fill-d"/>
<rect x="94" y="156" width="7" height="26" class="fill-d"/>
<rect x="102" y="156" width="7" height="26" class="fill-d"/>
<rect x="110" y="156" width="7" height="26" class="fill-d"/>
<rect x="118" y="156" width="7" height="26" class="fill-d"/>
<rect x="126" y="156" width="7" height="26" class="fill-d"/>
<rect x="134" y="156" width="7" height="26" class="fill-d"/>
<rect x="142" y="156" width="7" height="26" class="fill-d"/>
<rect x="150" y="156" width="7" height="26" class="fill-d"/>
<rect x="158" y="156" width="7" height="26" class="fill-d"/>
<rect x="166" y="156" width="7" height="26" class="fill-d"/>
<rect x="174" y="156" width="7" height="26" class="fill-d"/>
<rect x="182" y="156" width="7" height="26" class="fill-d"/>
<rect x="190" y="156" width="7" height="26" class="fill-a"/>
<rect x="198" y="156" width="7" height="26" class="fill-a"/>
<rect x="206" y="156" width="7" height="26" class="fill-a"/>
<rect x="214" y="156" width="7" height="26" class="fill-a"/>
<rect x="222" y="156" width="7" height="26" class="fill-a"/>
<rect x="230" y="156" width="7" height="26" class="fill-a"/>
<rect x="238" y="156" width="7" height="26" class="fill-a"/>
<rect x="246" y="156" width="7" height="26" class="fill-a"/>
<rect x="254" y="156" width="7" height="26" class="fill-a"/>
<rect x="262" y="156" width="7" height="26" class="fill-a"/>
<rect x="270" y="156" width="7" height="26" class="fill-a"/>
<rect x="278" y="156" width="7" height="26" class="fill-a"/>
<rect x="286" y="156" width="7" height="26" class="fill-a"/>
<rect x="294" y="156" width="7" height="26" class="fill-a"/>
<rect x="302" y="156" width="7" height="26" class="fill-a"/>
<rect x="310" y="156" width="7" height="26" class="fill-a"/>
<rect x="318" y="156" width="7" height="26" class="fill-a"/>
<rect x="326" y="156" width="7" height="26" class="fill-a"/>
<rect x="334" y="156" width="7" height="26" class="fill-a"/>
<rect x="342" y="156" width="7" height="26" class="fill-a"/>
<rect x="350" y="156" width="7" height="26" class="fill-a"/>
<rect x="358" y="156" width="7" height="26" class="fill-a"/>
<rect x="366" y="156" width="7" height="26" class="fill-a"/>
<rect x="374" y="156" width="7" height="26" class="fill-a"/>
<rect x="382" y="156" width="7" height="26" class="fill-a"/>
<rect x="390" y="156" width="7" height="26" class="fill-a"/>
<rect x="398" y="156" width="7" height="26" class="fill-a"/>
<rect x="406" y="156" width="7" height="26" class="fill-a"/>
<rect x="414" y="156" width="7" height="26" class="fill-a"/>
<rect x="422" y="156" width="7" height="26" class="fill-a"/>
<rect x="430" y="156" width="7" height="26" class="fill-soft-a"/>
<rect x="438" y="156" width="7" height="26" class="fill-soft-a"/>
<rect x="446" y="156" width="7" height="26" class="fill-soft-a"/>
<rect x="454" y="156" width="7" height="26" class="fill-soft-a"/>
<rect x="462" y="156" width="7" height="26" class="fill-soft-a"/>
<rect x="470" y="156" width="7" height="26" class="fill-soft-a"/>
<rect x="478" y="156" width="7" height="26" class="fill-soft-a"/>
<rect x="486" y="156" width="7" height="26" class="fill-soft-a"/>
<rect x="494" y="156" width="7" height="26" class="fill-soft-a"/>
<rect x="502" y="156" width="7" height="26" class="fill-soft-a"/>
<rect x="510" y="156" width="7" height="26" class="fill-soft-a"/>
<rect x="518" y="156" width="7" height="26" class="fill-soft-a"/>
<rect x="526" y="156" width="7" height="26" class="fill-soft-a"/>
<rect x="534" y="156" width="7" height="26" class="fill-soft-a"/>
<rect x="542" y="156" width="7" height="26" class="fill-soft-a"/>
<text x="60" y="224" text-anchor="end" class="fig-small">train on 45</text>
<rect x="70" y="208" width="7" height="26" class="fill-d"/>
<rect x="78" y="208" width="7" height="26" class="fill-d"/>
<rect x="86" y="208" width="7" height="26" class="fill-d"/>
<rect x="94" y="208" width="7" height="26" class="fill-d"/>
<rect x="102" y="208" width="7" height="26" class="fill-d"/>
<rect x="110" y="208" width="7" height="26" class="fill-d"/>
<rect x="118" y="208" width="7" height="26" class="fill-d"/>
<rect x="126" y="208" width="7" height="26" class="fill-d"/>
<rect x="134" y="208" width="7" height="26" class="fill-d"/>
<rect x="142" y="208" width="7" height="26" class="fill-d"/>
<rect x="150" y="208" width="7" height="26" class="fill-d"/>
<rect x="158" y="208" width="7" height="26" class="fill-d"/>
<rect x="166" y="208" width="7" height="26" class="fill-d"/>
<rect x="174" y="208" width="7" height="26" class="fill-d"/>
<rect x="182" y="208" width="7" height="26" class="fill-d"/>
<rect x="190" y="208" width="7" height="26" class="fill-a"/>
<rect x="198" y="208" width="7" height="26" class="fill-a"/>
<rect x="206" y="208" width="7" height="26" class="fill-a"/>
<rect x="214" y="208" width="7" height="26" class="fill-a"/>
<rect x="222" y="208" width="7" height="26" class="fill-a"/>
<rect x="230" y="208" width="7" height="26" class="fill-a"/>
<rect x="238" y="208" width="7" height="26" class="fill-a"/>
<rect x="246" y="208" width="7" height="26" class="fill-a"/>
<rect x="254" y="208" width="7" height="26" class="fill-a"/>
<rect x="262" y="208" width="7" height="26" class="fill-a"/>
<rect x="270" y="208" width="7" height="26" class="fill-a"/>
<rect x="278" y="208" width="7" height="26" class="fill-a"/>
<rect x="286" y="208" width="7" height="26" class="fill-a"/>
<rect x="294" y="208" width="7" height="26" class="fill-a"/>
<rect x="302" y="208" width="7" height="26" class="fill-a"/>
<rect x="310" y="208" width="7" height="26" class="fill-a"/>
<rect x="318" y="208" width="7" height="26" class="fill-a"/>
<rect x="326" y="208" width="7" height="26" class="fill-a"/>
<rect x="334" y="208" width="7" height="26" class="fill-a"/>
<rect x="342" y="208" width="7" height="26" class="fill-a"/>
<rect x="350" y="208" width="7" height="26" class="fill-a"/>
<rect x="358" y="208" width="7" height="26" class="fill-a"/>
<rect x="366" y="208" width="7" height="26" class="fill-a"/>
<rect x="374" y="208" width="7" height="26" class="fill-a"/>
<rect x="382" y="208" width="7" height="26" class="fill-a"/>
<rect x="390" y="208" width="7" height="26" class="fill-a"/>
<rect x="398" y="208" width="7" height="26" class="fill-a"/>
<rect x="406" y="208" width="7" height="26" class="fill-a"/>
<rect x="414" y="208" width="7" height="26" class="fill-a"/>
<rect x="422" y="208" width="7" height="26" class="fill-a"/>
<rect x="430" y="208" width="7" height="26" class="fill-a"/>
<rect x="438" y="208" width="7" height="26" class="fill-a"/>
<rect x="446" y="208" width="7" height="26" class="fill-a"/>
<rect x="454" y="208" width="7" height="26" class="fill-a"/>
<rect x="462" y="208" width="7" height="26" class="fill-a"/>
<rect x="470" y="208" width="7" height="26" class="fill-a"/>
<rect x="478" y="208" width="7" height="26" class="fill-a"/>
<rect x="486" y="208" width="7" height="26" class="fill-a"/>
<rect x="494" y="208" width="7" height="26" class="fill-a"/>
<rect x="502" y="208" width="7" height="26" class="fill-a"/>
<rect x="510" y="208" width="7" height="26" class="fill-a"/>
<rect x="518" y="208" width="7" height="26" class="fill-a"/>
<rect x="526" y="208" width="7" height="26" class="fill-a"/>
<rect x="534" y="208" width="7" height="26" class="fill-a"/>
<rect x="542" y="208" width="7" height="26" class="fill-a"/>
<text x="130.0" y="262" text-anchor="middle" class="fig-small">validation (15)</text>
<text x="370.0" y="262" text-anchor="middle" class="fig-small">the pool (45): training sets are its first 10, 18, 30, 45</text>
</svg>
Figure 2. Shuffle the 60 houses once. The first 15 (purple) are the validation set for every size. Each training set (blue) is the first 10, 18, 30, or 45 houses of the rest, so every set contains the one before it.
```

First, the validation set is fixed. If each size drew its own validation houses, a change in the score might come from easier houses and not from more data. Second, the training sets nest. If each size drew a fresh random sample, a size-30 set might lack houses a size-18 set had, and again the score would mix more data with a different draw. Nested sets mean the curve moves only because houses were added.

In numpy, one shuffle does both jobs. The first `n_val` rows of the shuffle are validation, and each training set is a prefix of the remaining rows, called the pool:

```python
order = np.random.default_rng(seed).permutation(n)
val_idx, pool = order[:n_val], order[n_val:]
rows = pool[:size]            # training set for this size
```

## Turn the reading into a rule

Two numbers carry most of the reading: the validation error at the largest size, and the gap, which is validation error minus training error at that size. Figure 3 places the two models on a map of those numbers.

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="A map of gap against validation error with three regions: good, underfit, and overfit, and the two running models placed on it">
<rect x="80.0" y="140.0" width="200.0" height="100.0" class="fill-soft-c"/>
<rect x="80.0" y="40" width="200.0" height="100.0" class="fill-soft-a"/>
<rect x="280.0" y="40" width="300.0" height="200" class="fill-soft-b"/>
<line x1="80" y1="240" x2="580" y2="240" class="stroke-muted"/><line x1="80" y1="40" x2="80" y2="240" class="stroke-muted"/>
<line x1="80.0" y1="240" x2="80.0" y2="244" class="stroke-muted"/><text x="80.0" y="258" text-anchor="middle" class="fig-small">0</text>
<line x1="205.0" y1="240" x2="205.0" y2="244" class="stroke-muted"/><text x="205.0" y="258" text-anchor="middle" class="fig-small">0.05</text>
<line x1="330.0" y1="240" x2="330.0" y2="244" class="stroke-muted"/><text x="330.0" y="258" text-anchor="middle" class="fig-small">0.1</text>
<line x1="455.0" y1="240" x2="455.0" y2="244" class="stroke-muted"/><text x="455.0" y="258" text-anchor="middle" class="fig-small">0.15</text>
<line x1="580.0" y1="240" x2="580.0" y2="244" class="stroke-muted"/><text x="580.0" y="258" text-anchor="middle" class="fig-small">0.2</text>
<text x="72" y="244.0" text-anchor="end" class="fig-small">0</text>
<text x="72" y="194.0" text-anchor="end" class="fig-small">0.2</text>
<text x="72" y="144.0" text-anchor="end" class="fig-small">0.4</text>
<text x="72" y="94.0" text-anchor="end" class="fig-small">0.6</text>
<text x="72" y="44.0" text-anchor="end" class="fig-small">0.8</text>
<text x="330.0" y="282" text-anchor="middle">gap: validation minus training error, at the largest size</text><text x="14" y="140" transform="rotate(-90 14 140)" text-anchor="middle">validation error</text>
<text x="180.0" y="188.0" text-anchor="middle" class="fig-bold">good</text><text x="180.0" y="204.0" text-anchor="middle" class="fig-small">stop tuning</text>
<text x="86.0" y="58" class="fig-bold">underfit</text><text x="86.0" y="74" class="fig-small">richer model</text><text x="86.0" y="89" class="fig-small">or features</text>
<text x="430.0" y="58" class="fig-bold">overfit</text><text x="430.0" y="74" class="fig-small">more data, a stronger</text><text x="430.0" y="89" class="fig-small">penalty, or fewer features</text>
<circle cx="212.6" cy="117.9" r="6" class="fill-ink"/><text x="202.6" y="121.9" text-anchor="end" class="fig-small fig-bold">straight line</text>
<circle cx="339.7" cy="158.3" r="6" class="fill-ink"/><text x="349.7" y="162.3" text-anchor="start" class="fig-small fig-bold">degree 8 curve</text>
<line x1="280.0" y1="40" x2="280.0" y2="240" class="stroke-ink" stroke-dasharray="5 4"/><line x1="80" y1="140.0" x2="280.0" y2="140.0" class="stroke-ink" stroke-dasharray="5 4"/>
</svg>
Figure 3. A wide gap means overfit, whatever the height. A small gap with a high validation error means underfit. A small gap with a low error means good. The two dashed lines are thresholds you choose: 0.08 for the gap and 0.4 for the error you can live with.
```

The straight line sits at a gap of 0.05 and an error of 0.49, in the underfit region. The degree 8 curve sits at a gap of 0.10 and an error of 0.33, in the overfit region. The thresholds belong to you. The error threshold is the score you need for the job, and the gap threshold is how much overfitting you are willing to leave. The rule checks the gap first, because a wide gap means the model is chasing noise and more data or a penalty will help, whatever the height.

> [!KEY]
> A learning curve shows what changes when you add houses. If training and validation error have met above the error you need, more data will not help, so change the model. If they are far apart, the model is chasing noise, so add data, add a penalty, or give it fewer features.

> [!RECAP]
> - Training error rises and validation error falls as the training set grows, and they squeeze toward the best this model can do.
> - Curves that meet at a high level mean the model is too simple, so more data will not help.
> - A wide gap means the model chases noise, and more data or a stronger penalty narrows it.
> - Use one fixed validation set and nested training sets, so only the amount of data changes.

## Build it step by step

You write three functions, in the order the checks test them.

1. **`validation_split(n, val_fraction, seed)`**. Takes the number of rows, the fraction to hold out, and a seed. Returns `(val_idx, pool_idx)`, two integer arrays that together cover `0` to `n - 1` once. The validation set has `int(round(val_fraction * n))` rows. The common mistake is to build the generator without the seed, so the split changes on every call.

   ```python
   order = np.random.default_rng(seed).permutation(n)
   n_val = int(round(val_fraction * n))
   ```

2. **`learning_curve(fit, predict, X, y, train_sizes, val_fraction, seed)`**. Calls `validation_split` once, then loops over the sizes. Returns `(train_errors, val_errors)`, each of shape `(len(train_sizes),)`. The common mistake is a fresh shuffle inside the loop, which breaks the nesting.

   ```python
   rows = pool_idx[:size]
   model = fit(X[rows], y[rows])
   train_err = np.mean((predict(model, X[rows]) - y[rows]) ** 2)
   ```

3. **`diagnose_curve(train_errors, val_errors, high_threshold, gap_threshold)`**. Reads the last entry of each curve and returns `'overfit'`, `'underfit'`, or `'good'`. Compute `gap = val_errors[-1] - train_errors[-1]`, test the gap first, then the height. The common mistake is the reverse order, which calls a wide-gap curve with a high error `'underfit'`.

## Where this breaks

**A small validation set.** The validation set here has 15 houses, so one odd house moves the curve. Average over several shuffles, as Figure 1 does, or use cross-validation to score each size.

**Unknown noise floor.** The curve tells you a model has stopped improving, not how much room remains. Try the richer model before you decide the rest is noise.

**A curve that is still falling.** If validation error is still dropping at your largest size, more data will help. If it has been flat for the last few sizes, it will not.

## The exercise

You implement `validation_split`, `learning_curve`, and `diagnose_curve`. The starter file ends with a short program that builds the 60 houses, takes the median curve over 60 shuffles for the straight line and the degree 8 curve, and prints a verdict for each.

The checks confirm that `validation_split` returns the right sizes, covers every row once, and repeats for one seed. They confirm that `learning_curve` returns one finite error per size, that training sets nest, that every size is scored on the same validation rows, and that on linear data the training error starts near 0 and the validation error falls. The last two checks confirm that `diagnose_curve` gives the right verdict on four hand-built curves, and that it calls the straight line `'underfit'` and the degree 8 curve `'overfit'` on the chapter's houses.
