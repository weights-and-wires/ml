---
title: Why the best score among many settings is too good
slug: 025-winners-curse-and-flexibility
part: "Part 2 — Classical models"
summary: The best of many scores is partly luck, so score the choice on data it never saw, and measure how well a set of models fits noise to size the discount.
minutes: 45
gpu: false
objectives:
  - Explain why the lowest of many noisy scores is too optimistic, and simulate how much.
  - Build nested cross-validation to score a penalty choice on data that never influenced it.
  - State the finite-class bound in plain words and say what grows its discount with the number of settings.
  - Estimate how well a set of models fits random signs, and say what a value near 1 means.
prereqs:
  - 022-cross-validation
  - 024-learning-curves
exercise: 025-winners-curse-and-flexibility
---

# Why the best score among many settings is too good

You tried 20 ridge penalties on your 60 houses and scored each with five-fold cross-validation. The best one reaches 1.15. You are about to write "expected error 1.15" in your report. Is that what you will see on houses you have not met?

It is not. The score 1.15 is the lowest of 20 scores, and you picked it because it was the lowest. Each score carries some luck from the folds that happened to be drawn, and by choosing the minimum you chose the penalty whose luck was best. If you report 1.15, you promise more than the model delivers.

This chapter shows why the best of many is too good, how to get an honest number, and how to reason about how much to discount. The discount grows with how many settings you tried and with how well your set of models could fit pure noise.

## What you already know

- Cross-validation scores a model on every fold it did not train on, and the average of those scores is your estimate of its error on unseen houses.
- An estimate from 60 houses is not the truth. Draw different folds and you get a different number.
- A ridge penalty is a setting you choose by comparing scores.
- If you make several tries and each one fails with chance 0.1, the chance that at least one fails is at most the sum of the chances.
- Examples are *i.i.d.* when each is drawn independently from the same source, as with repeated rolls of one die.

## The winner's curse

Suppose all 20 penalties are equally good, with a true error of exactly 1.00. Each estimate is fair on its own, 1.00 plus some noise with a spread of 0.1. Figure 1 shows one try. The 20 dots scatter around the true error, and the lowest, model 13, reads 0.77. You keep it, and you report 0.77, although its true error is 1.00.

```figure
<svg viewBox="0 0 640 310" role="img" aria-label="Twenty equally good models get noisy scores, and the lowest one sits well below the true error; repeated many times, the best of twenty scores is almost always below the truth">
<text x="180.0" y="28" text-anchor="middle" class="fig-bold">one try: 20 models, one score each</text>
<line x1="60" y1="208.3" x2="300" y2="208.3" class="stroke-grid"/><text x="54" y="212.3" text-anchor="end" class="fig-small">0.8</text>
<line x1="60" y1="145.0" x2="300" y2="145.0" class="stroke-grid"/><text x="54" y="149.0" text-anchor="end" class="fig-small">1.0</text>
<line x1="60" y1="81.7" x2="300" y2="81.7" class="stroke-grid"/><text x="54" y="85.7" text-anchor="end" class="fig-small">1.2</text>
<line x1="60" y1="145.0" x2="300" y2="145.0" class="stroke-ink" stroke-dasharray="6 4"/>
<line x1="60" y1="240" x2="300" y2="240" class="stroke-muted"/><line x1="60" y1="42" x2="60" y2="240" class="stroke-muted"/>
<circle cx="70.0" cy="141.0" r="4.5" class="fill-a"/>
<circle cx="81.6" cy="149.2" r="4.5" class="fill-a"/>
<circle cx="93.2" cy="124.7" r="4.5" class="fill-a"/>
<circle cx="104.7" cy="141.7" r="4.5" class="fill-a"/>
<circle cx="116.3" cy="162.0" r="4.5" class="fill-a"/>
<circle cx="127.9" cy="133.5" r="4.5" class="fill-a"/>
<circle cx="139.5" cy="103.7" r="4.5" class="fill-a"/>
<circle cx="151.1" cy="115.0" r="4.5" class="fill-a"/>
<circle cx="162.6" cy="167.3" r="4.5" class="fill-a"/>
<circle cx="174.2" cy="185.1" r="4.5" class="fill-a"/>
<circle cx="185.8" cy="164.7" r="4.5" class="fill-a"/>
<circle cx="197.4" cy="143.7" r="4.5" class="fill-a"/>
<circle cx="208.9" cy="218.6" r="4.5" class="fill-a"/>
<circle cx="220.5" cy="151.9" r="4.5" class="fill-a"/>
<circle cx="232.1" cy="184.5" r="4.5" class="fill-a"/>
<circle cx="243.7" cy="168.2" r="4.5" class="fill-a"/>
<circle cx="255.3" cy="162.2" r="4.5" class="fill-a"/>
<circle cx="266.8" cy="155.0" r="4.5" class="fill-a"/>
<circle cx="278.4" cy="132.0" r="4.5" class="fill-a"/>
<circle cx="290.0" cy="112.0" r="4.5" class="fill-a"/>
<circle cx="208.9" cy="218.6" r="9" fill="none" class="stroke-b" stroke-width="2.5"/>
<text x="222.9" y="236.6" class="fig-bold">winner 0.77</text>
<text x="68" y="59.5" class="fig-small">dashed: true error 1.00, the same for all 20</text>
<text x="180.0" y="292" text-anchor="middle">model number</text>
<text x="14" y="150" transform="rotate(-90 14 150)" text-anchor="middle">estimated error</text>
<text x="490.0" y="28" text-anchor="middle" class="fig-bold">20,000 tries</text>
<rect x="370.0" y="239.9" width="6.5" height="0.1" class="fill-soft-a"/>
<rect x="377.5" y="240.0" width="6.5" height="0.0" class="fill-soft-a"/>
<rect x="385.0" y="239.6" width="6.5" height="0.4" class="fill-soft-a"/>
<rect x="392.5" y="239.5" width="6.5" height="0.5" class="fill-soft-a"/>
<rect x="400.0" y="238.4" width="6.5" height="1.6" class="fill-soft-a"/>
<rect x="407.5" y="237.0" width="6.5" height="3.0" class="fill-soft-a"/>
<rect x="415.0" y="233.9" width="6.5" height="6.1" class="fill-soft-a"/>
<rect x="422.5" y="231.1" width="6.5" height="8.9" class="fill-soft-a"/>
<rect x="430.0" y="224.6" width="6.5" height="15.4" class="fill-soft-a"/>
<rect x="437.5" y="213.8" width="6.5" height="26.2" class="fill-soft-a"/>
<rect x="445.0" y="201.1" width="6.5" height="38.9" class="fill-soft-a"/>
<rect x="452.5" y="186.8" width="6.5" height="53.2" class="fill-soft-a"/>
<rect x="460.0" y="173.9" width="6.5" height="66.1" class="fill-soft-a"/>
<rect x="467.5" y="165.6" width="6.5" height="74.4" class="fill-soft-a"/>
<rect x="475.0" y="152.3" width="6.5" height="87.7" class="fill-soft-a"/>
<rect x="482.5" y="145.4" width="6.5" height="94.6" class="fill-soft-a"/>
<rect x="490.0" y="144.0" width="6.5" height="96.0" class="fill-soft-a"/>
<rect x="497.5" y="150.2" width="6.5" height="89.8" class="fill-soft-a"/>
<rect x="505.0" y="166.4" width="6.5" height="73.6" class="fill-soft-a"/>
<rect x="512.5" y="176.4" width="6.5" height="63.6" class="fill-soft-a"/>
<rect x="520.0" y="187.7" width="6.5" height="52.3" class="fill-soft-a"/>
<rect x="527.5" y="202.6" width="6.5" height="37.4" class="fill-soft-a"/>
<rect x="535.0" y="214.3" width="6.5" height="25.7" class="fill-soft-a"/>
<rect x="542.5" y="222.8" width="6.5" height="17.2" class="fill-soft-a"/>
<rect x="550.0" y="230.0" width="6.5" height="10.0" class="fill-soft-a"/>
<rect x="557.5" y="235.0" width="6.5" height="5.0" class="fill-soft-a"/>
<rect x="565.0" y="236.5" width="6.5" height="3.5" class="fill-soft-a"/>
<rect x="572.5" y="239.1" width="6.5" height="0.9" class="fill-soft-a"/>
<rect x="580.0" y="239.5" width="6.5" height="0.5" class="fill-soft-a"/>
<rect x="587.5" y="239.7" width="6.5" height="0.3" class="fill-soft-a"/>
<rect x="595.0" y="239.8" width="6.5" height="0.2" class="fill-soft-a"/>
<rect x="602.5" y="239.9" width="6.5" height="0.1" class="fill-soft-a"/>
<rect x="370.0" y="238.2" width="6.5" height="1.8" class="fill-b" opacity="0.85"/>
<rect x="377.5" y="237.1" width="6.5" height="2.9" class="fill-b" opacity="0.85"/>
<rect x="385.0" y="232.5" width="6.5" height="7.5" class="fill-b" opacity="0.85"/>
<rect x="392.5" y="224.7" width="6.5" height="15.3" class="fill-b" opacity="0.85"/>
<rect x="400.0" y="211.2" width="6.5" height="28.8" class="fill-b" opacity="0.85"/>
<rect x="407.5" y="182.7" width="6.5" height="57.3" class="fill-b" opacity="0.85"/>
<rect x="415.0" y="144.4" width="6.5" height="95.6" class="fill-b" opacity="0.85"/>
<rect x="422.5" y="95.8" width="6.5" height="144.2" class="fill-b" opacity="0.85"/>
<rect x="430.0" y="60.0" width="6.5" height="180.0" class="fill-b" opacity="0.85"/>
<rect x="437.5" y="64.2" width="6.5" height="175.8" class="fill-b" opacity="0.85"/>
<rect x="445.0" y="101.0" width="6.5" height="139.0" class="fill-b" opacity="0.85"/>
<rect x="452.5" y="166.4" width="6.5" height="73.6" class="fill-b" opacity="0.85"/>
<rect x="460.0" y="214.7" width="6.5" height="25.3" class="fill-b" opacity="0.85"/>
<rect x="467.5" y="234.9" width="6.5" height="5.1" class="fill-b" opacity="0.85"/>
<rect x="475.0" y="239.5" width="6.5" height="0.5" class="fill-b" opacity="0.85"/>
<rect x="482.5" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="490.0" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="497.5" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="505.0" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="512.5" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="520.0" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="527.5" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="535.0" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="542.5" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="550.0" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="557.5" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="565.0" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="572.5" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="580.0" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="587.5" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="595.0" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<rect x="602.5" y="240.0" width="6.5" height="0.0" class="fill-b" opacity="0.85"/>
<line x1="370" y1="240" x2="610" y2="240" class="stroke-muted"/>
<line x1="370.0" y1="240" x2="370.0" y2="244" class="stroke-muted"/><text x="370.0" y="258" text-anchor="middle" class="fig-small">0.6</text>
<line x1="430.0" y1="240" x2="430.0" y2="244" class="stroke-muted"/><text x="430.0" y="258" text-anchor="middle" class="fig-small">0.8</text>
<line x1="490.0" y1="240" x2="490.0" y2="244" class="stroke-muted"/><text x="490.0" y="258" text-anchor="middle" class="fig-small">1.0</text>
<line x1="550.0" y1="240" x2="550.0" y2="244" class="stroke-muted"/><text x="550.0" y="258" text-anchor="middle" class="fig-small">1.2</text>
<line x1="610.0" y1="240" x2="610.0" y2="244" class="stroke-muted"/><text x="610.0" y="258" text-anchor="middle" class="fig-small">1.4</text>
<line x1="490.0" y1="50" x2="490.0" y2="240" class="stroke-ink" stroke-dasharray="6 4"/>
<line x1="433.9" y1="50" x2="433.9" y2="240" class="stroke-b" stroke-width="2"/>
<text x="496.0" y="58" class="fig-small">truth 1.00</text>
<text x="427.9" y="74" text-anchor="end" class="fig-small fig-bold">best of 20: mean 0.81</text>
<text x="574.0" y="160" text-anchor="middle" class="fig-small">one model</text><text x="574.0" y="174" text-anchor="middle" class="fig-small">(light blue)</text>
<text x="490.0" y="292" text-anchor="middle">estimated error of the model you keep</text>
</svg>
Figure 1. Left: 20 equal models, each with a noisy estimate. The lowest dot reads 0.77, far below the true error. Right: 20,000 repeats. A single model's estimate (light blue) centres on 1.00, but the best of 20 (orange) centres on 0.81.
```

The right panel repeats the experiment 20,000 times. One model's estimate is centred on the truth, so it is fair. The best of 20 is centred on 0.81, about 0.19 too low, and it fell below 1.00 in essentially every repeat. The reason is that selecting on a score uses the luck up. A dot ends up lowest when its noise was negative, so the winner's noise is not an average draw. It is the most favourable of 20.

The size of the effect depends on how many you try and how noisy each estimate is, and not on the true error. With noise 0.1 the expected gap is:

| Models tried | 1 | 2 | 5 | 20 | 100 |
|---|---|---|---|---|---|
| Winner looks this much too good | 0.00 | 0.06 | 0.12 | 0.19 | 0.25 |

A single model has no curse, because you did not choose it from a crowd. Twenty models give about 1.9 noise spreads, and 100 give about 2.5. The gap grows slowly, so trying more settings costs more, but not much more. In practice the models are not equal, and the truly best penalty is better than the rest. The lean toward luck still applies on top of that.

## Score the choice on data it never saw

The cure is to score on houses that played no part in choosing. A held-out test set does this, but it costs data you may not have. *Nested cross-validation* does it with the data you have, by treating the choice as part of the model. Think of it as an exam with practice exams. The practice exams choose your study plan. The real exam then scores the plan, and you never use it to choose.

Figure 2 shows one round. An outer fold is hidden and used only for scoring. On the other 48 houses, an inner cross-validation tries all 20 penalties and picks the one with the lowest average. You refit with that penalty on all 48 houses and score the hidden outer fold. Then you repeat for each outer fold.

```figure
<svg viewBox="0 0 620 320" role="img" aria-label="Nested cross-validation: one outer round holds out a fold for scoring only, while four inner rounds on the remaining houses choose the penalty">
<text x="60" y="24" class="fig-small">outer round 1 of 5: the 60 houses in 5 blocks of 12</text>
<rect x="62" y="34" width="92" height="34" rx="4" class="fill-soft-b stroke-b" stroke-width="2.5"/>
<rect x="158" y="34" width="92" height="34" rx="4" class="fill-soft-a"/>
<rect x="254" y="34" width="92" height="34" rx="4" class="fill-soft-a"/>
<rect x="350" y="34" width="92" height="34" rx="4" class="fill-soft-a"/>
<rect x="446" y="34" width="92" height="34" rx="4" class="fill-soft-a"/>
<text x="108.0" y="56" text-anchor="middle" class="fig-bold fig-small">only scores</text>
<text x="204.0" y="56" text-anchor="middle" class="fig-small">outer training</text>
<text x="300.0" y="56" text-anchor="middle" class="fig-small">outer training</text>
<text x="396.0" y="56" text-anchor="middle" class="fig-small">outer training</text>
<text x="492.0" y="56" text-anchor="middle" class="fig-small">outer training</text>
<text x="348" y="94" text-anchor="middle" class="fig-small">inside the outer training houses (48): 4 inner rounds, each hides one block and tries all 20 penalties</text>
<text x="52" y="126" text-anchor="end" class="fig-small">inner 1</text>
<rect x="158" y="106" width="92" height="30" rx="4" class="fill-soft-c stroke-c" stroke-width="2.5"/>
<text x="204.0" y="125" text-anchor="middle" class="fig-small fig-bold">inner test</text>
<rect x="254" y="106" width="92" height="30" rx="4" class="fill-soft-a"/>
<text x="300.0" y="125" text-anchor="middle" class="fig-small ">train</text>
<rect x="350" y="106" width="92" height="30" rx="4" class="fill-soft-a"/>
<text x="396.0" y="125" text-anchor="middle" class="fig-small ">train</text>
<rect x="446" y="106" width="92" height="30" rx="4" class="fill-soft-a"/>
<text x="492.0" y="125" text-anchor="middle" class="fig-small ">train</text>
<text x="52" y="164" text-anchor="end" class="fig-small">inner 2</text>
<rect x="158" y="144" width="92" height="30" rx="4" class="fill-soft-a"/>
<text x="204.0" y="163" text-anchor="middle" class="fig-small ">train</text>
<rect x="254" y="144" width="92" height="30" rx="4" class="fill-soft-c stroke-c" stroke-width="2.5"/>
<text x="300.0" y="163" text-anchor="middle" class="fig-small fig-bold">inner test</text>
<rect x="350" y="144" width="92" height="30" rx="4" class="fill-soft-a"/>
<text x="396.0" y="163" text-anchor="middle" class="fig-small ">train</text>
<rect x="446" y="144" width="92" height="30" rx="4" class="fill-soft-a"/>
<text x="492.0" y="163" text-anchor="middle" class="fig-small ">train</text>
<text x="52" y="202" text-anchor="end" class="fig-small">inner 3</text>
<rect x="158" y="182" width="92" height="30" rx="4" class="fill-soft-a"/>
<text x="204.0" y="201" text-anchor="middle" class="fig-small ">train</text>
<rect x="254" y="182" width="92" height="30" rx="4" class="fill-soft-a"/>
<text x="300.0" y="201" text-anchor="middle" class="fig-small ">train</text>
<rect x="350" y="182" width="92" height="30" rx="4" class="fill-soft-c stroke-c" stroke-width="2.5"/>
<text x="396.0" y="201" text-anchor="middle" class="fig-small fig-bold">inner test</text>
<rect x="446" y="182" width="92" height="30" rx="4" class="fill-soft-a"/>
<text x="492.0" y="201" text-anchor="middle" class="fig-small ">train</text>
<text x="52" y="240" text-anchor="end" class="fig-small">inner 4</text>
<rect x="158" y="220" width="92" height="30" rx="4" class="fill-soft-a"/>
<text x="204.0" y="239" text-anchor="middle" class="fig-small ">train</text>
<rect x="254" y="220" width="92" height="30" rx="4" class="fill-soft-a"/>
<text x="300.0" y="239" text-anchor="middle" class="fig-small ">train</text>
<rect x="350" y="220" width="92" height="30" rx="4" class="fill-soft-a"/>
<text x="396.0" y="239" text-anchor="middle" class="fig-small ">train</text>
<rect x="446" y="220" width="92" height="30" rx="4" class="fill-soft-c stroke-c" stroke-width="2.5"/>
<text x="492.0" y="239" text-anchor="middle" class="fig-small fig-bold">inner test</text>
<rect x="62" y="106" width="92" height="144" rx="4" class="fill-sunk"/>
<text x="108.0" y="170" text-anchor="middle" class="fig-small">not touched</text><text x="108.0" y="186" text-anchor="middle" class="fig-small">while choosing</text>
<text x="60" y="276" class="fig-small">lowest inner average picks the penalty: 26.4. Refit on all 48, then score the outer block: 1.45.</text>
<text x="60" y="298" class="fig-small">Repeat for each outer block. The five outer scores average to the honest number, 1.27.</text>
</svg>
Figure 2. One outer round. The orange block is never used to choose the penalty, only to score the choice. The four inner rounds work on the blue houses alone.
```

What you score is the recipe, "choose the penalty by cross-validation and then fit", and not any one penalty. On the 60 houses, the five outer rounds pick penalties of 26.4, 2.3, 7.8, 7.8, and 14.4, and score 1.45, 1.14, 0.69, 0.95, and 2.13. Those penalties differ, which is fine, because every outer fold was scored by a choice that never saw it. The average of the five scores is 1.27. That is the honest number, against the 1.15 you got by reusing the data that chose. On 5000 unseen houses the penalty 7.8 scores 1.37, so 1.27 sits closer to the truth than 1.15 does.

In numpy the shape is a loop inside a loop. The inner part reuses `cross_val_score` on the outer training rows alone:

```python
for test in k_fold_indices(n, outer_k, rng):
    train = np.setdiff1d(np.arange(n), test)
    means = [cross_val_score(fit_with(lam), predict, X[train], y[train], inner_k).mean()
             for lam in lambdas]                       # inner loop: choose
    w = ridge(X[train], y[train], lambdas[np.argmin(means)])
    outer_scores.append(np.mean((X[test] @ w - y[test]) ** 2))  # score
```

This costs outer times inner times settings fits, here $5 \times 4 \times 20 = 400$ plus 5.

## How big is the discount?

Nested cross-validation gives you the number. The next question is why the discount grows with the number of settings, and by how much. For a loss between 0 and 1, such as the fraction of wrong answers, a short argument answers it.

First, one model's training average is close to its true average, because an average of $n$ independent values rarely strays far from its mean. The chance of straying by more than $t$ is at most $2e^{-2nt^2}$. Second, you try $M$ models, and you want all of them close at once. By the sum rule from the list above, which is called the union bound, the chance that at least one strays is at most $M$ times larger, $2Me^{-2nt^2}$. Set that to a failure chance $\delta$ that you choose, and solve for $t$:

$$
\text{gap} \le \sqrt{\frac{\log(2M/\delta)}{2n}}
$$

Read it aloud: with probability at least $1 - \delta$, every one of the $M$ models has a training average within this gap of its true average. The $n$ sits under the square root because more examples make each average steadier, and $M$ sits inside the log because the sum rule multiplies the failure chance by $M$, which solving for $t$ turns into a log. The log is the natural logarithm, and the examples must be i.i.d. For $M = 2$, $n = 4$, and $\delta = 0.5$, the log of $2 \cdot 2 / 0.5 = 8$ is 2.079, divided by $2n = 8$ is 0.260, and the square root is 0.510. For $M = 100$, $n = 1000$, and $\delta = 0.05$ it is 0.064, and four times the data halves it. In numpy it is `np.sqrt(np.log(2 * M / delta) / (2 * n))`.

The logarithm is the good news in Figure 3. Going from 1 model to 20 raises the gap from 0.18 to 0.24 at $n = 60$ and $\delta = 0.05$, and going to 1000 raises it only to 0.30. More settings cost more, but slowly.

```figure
<svg viewBox="0 0 640 310" role="img" aria-label="Left: the finite-class bound grows slowly with the number of models. Right: the noise-fitting score rises from about 0 for one model to 1 for every sign pattern.">
<text x="180.0" y="28" text-anchor="middle" class="fig-bold">finite bound, 60 examples</text>
<line x1="60" y1="240.0" x2="300" y2="240.0" class="stroke-grid"/><text x="54" y="244.0" text-anchor="end" class="fig-small">0.0</text>
<line x1="60" y1="202.0" x2="300" y2="202.0" class="stroke-grid"/><text x="54" y="206.0" text-anchor="end" class="fig-small">0.1</text>
<line x1="60" y1="164.0" x2="300" y2="164.0" class="stroke-grid"/><text x="54" y="168.0" text-anchor="end" class="fig-small">0.2</text>
<line x1="60" y1="126.0" x2="300" y2="126.0" class="stroke-grid"/><text x="54" y="130.0" text-anchor="end" class="fig-small">0.3</text>
<line x1="60" y1="88.0" x2="300" y2="88.0" class="stroke-grid"/><text x="54" y="92.0" text-anchor="end" class="fig-small">0.4</text>
<line x1="60" y1="50.0" x2="300" y2="50.0" class="stroke-grid"/><text x="54" y="54.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="60.0" y1="240" x2="60.0" y2="244" class="stroke-muted"/><text x="60.0" y="258" text-anchor="middle" class="fig-small">1</text>
<line x1="140.0" y1="240" x2="140.0" y2="244" class="stroke-muted"/><text x="140.0" y="258" text-anchor="middle" class="fig-small">10²</text>
<line x1="220.0" y1="240" x2="220.0" y2="244" class="stroke-muted"/><text x="220.0" y="258" text-anchor="middle" class="fig-small">10⁴</text>
<line x1="300.0" y1="240" x2="300.0" y2="244" class="stroke-muted"/><text x="300.0" y="258" text-anchor="middle" class="fig-small">10⁶</text>
<line x1="60" y1="240" x2="300" y2="240" class="stroke-muted"/><line x1="60" y1="42" x2="60" y2="240" class="stroke-muted"/>
<polyline points="60.0,173.4 64.1,171.3 68.1,169.3 72.2,167.3 76.3,165.4 80.3,163.5 84.4,161.7 88.5,159.9 92.5,158.2 96.6,156.5 100.7,154.8 104.7,153.2 108.8,151.6 112.9,150.0 116.9,148.4 121.0,146.9 125.1,145.4 129.2,143.9 133.2,142.5 137.3,141.0 141.4,139.6 145.4,138.2 149.5,136.9 153.6,135.5 157.6,134.2 161.7,132.8 165.8,131.5 169.8,130.2 173.9,129.0 178.0,127.7 182.0,126.5 186.1,125.2 190.2,124.0 194.2,122.8 198.3,121.6 202.4,120.4 206.4,119.2 210.5,118.1 214.6,116.9 218.6,115.8 222.7,114.7 226.8,113.5 230.8,112.4 234.9,111.3 239.0,110.2 243.1,109.2 247.1,108.1 251.2,107.0 255.3,106.0 259.3,104.9 263.4,103.9 267.5,102.9 271.5,101.8 275.6,100.8 279.7,99.8 283.7,98.8 287.8,97.8 291.9,96.8 295.9,95.8 300.0,94.9" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="60.0" cy="173.4" r="4.5" class="fill-b"/>
<text x="68.0" y="193.4" class="fig-small">M=1: 0.18</text>
<circle cx="112.0" cy="150.3" r="4.5" class="fill-b"/>
<text x="120.0" y="170.3" class="fig-small">M=20: 0.24</text>
<circle cx="180.0" cy="127.1" r="4.5" class="fill-b"/>
<text x="188.0" y="147.1" class="fig-small">M=1000: 0.30</text>
<text x="180.0" y="292" text-anchor="middle">number of models M (log scale)</text>
<text x="14" y="150" transform="rotate(-90 14 150)" text-anchor="middle">largest possible gap</text>
<text x="500" y="28" text-anchor="middle" class="fig-bold">how well the set fits random signs</text>
<line x1="380" y1="240.0" x2="620" y2="240.0" class="stroke-grid"/><text x="374" y="244.0" text-anchor="end" class="fig-small">0.0</text>
<line x1="380" y1="145.0" x2="620" y2="145.0" class="stroke-grid"/><text x="374" y="149.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="380" y1="50.0" x2="620" y2="50.0" class="stroke-grid"/><text x="374" y="54.0" text-anchor="end" class="fig-small">1.0</text>
<rect x="392" y="239.8" width="44" height="0.2" class="fill-d"/>
<text x="414" y="233.8" text-anchor="middle" class="fig-small">0.00</text>
<text x="414" y="258" text-anchor="middle" class="fig-small">1</text><text x="414" y="272" text-anchor="middle" class="fig-small">model</text>
<rect x="452" y="195.4" width="44" height="44.6" class="fill-d"/>
<text x="474" y="189.4" text-anchor="middle" class="fig-small">0.23</text>
<text x="474" y="258" text-anchor="middle" class="fig-small">10</text><text x="474" y="272" text-anchor="middle" class="fig-small">models</text>
<rect x="512" y="155.1" width="44" height="84.9" class="fill-d"/>
<text x="534" y="149.1" text-anchor="middle" class="fig-small">0.45</text>
<text x="534" y="258" text-anchor="middle" class="fig-small">300</text><text x="534" y="272" text-anchor="middle" class="fig-small">models</text>
<rect x="572" y="50.0" width="44" height="190.0" class="fill-d"/>
<text x="594" y="44.0" text-anchor="middle" class="fig-small">1.00</text>
<text x="594" y="258" text-anchor="middle" class="fig-small">all</text><text x="594" y="272" text-anchor="middle" class="fig-small">128 patterns</text>
<line x1="380" y1="240" x2="620" y2="240" class="stroke-muted"/>
<text x="500" y="298" text-anchor="middle" class="fig-small">first three: 40 examples; last: 7 examples</text>
</svg>
Figure 3. Left: the finite-class bound at 60 examples grows slowly with the number of models. Right: how well a set of models fits random signs, from about 0 for one model to 1 for every sign pattern.
```

Two honest limits apply. The bound needs a loss between 0 and 1, and the squared error of a house price is not. Cross-validation has no such condition, which is why it is the working tool, and the bound is the intuition for why the discount exists. The bound is also loose: at $n = 60$ and 20 models it allows a gap of 0.24, which is large for a loss that cannot exceed 1.

## When there are infinitely many models

A ridge penalty is a continuous knob, so there are infinitely many models, and $\log M$ is meaningless. The fix is to stop counting models and to measure how well the set could fit pure noise. Give each of the $n$ examples a random sign, $+1$ or $-1$. These signs are noise, with no pattern to learn. Score each model by how well its outputs line up with the signs, and keep the best model. A set that can line up with almost any noise is a set whose winner can be lucky by a lot.

Take two models on two examples, model 1 with outputs $[1, 1]$ and model 2 with outputs $[1, -1]$. Each model's score is the average of its output times the sign:

| Signs | Model 1 | Model 2 | Best |
|---|---|---|---|
| $[-1, -1]$ | $-1$ | $0$ | $0$ |
| $[-1, 1]$ | $0$ | $-1$ | $0$ |
| $[1, -1]$ | $0$ | $1$ | $1$ |
| $[1, 1]$ | $1$ | $0$ | $1$ |

The best column averages to $(0 + 0 + 1 + 1)/4 = 0.5$. Averaged over all sign patterns, that is the *empirical Rademacher complexity*. Row $j$ of $F$ holds model $j$'s output on each example, and for the bound below those entries are each model's loss on each example, which is why the exercise names the matrix `losses_matrix`:

$$
\hat{R} = \text{average over random } \sigma \text{ of } \; \max_{j} \frac{1}{n} \sum_{i=1}^{n} \sigma_i F_{ji}
$$

Read it aloud: for each random sign vector $\sigma$, dot every model with it, divide by $n$, keep the best, and average over many draws. One model scores near 0, because its dot with random signs is as often negative as positive. A set that contains every sign pattern scores exactly 1, because one member always matches the noise. Figure 3 shows 1 model at 0.00, 10 at 0.23, 300 at 0.45, and all 128 patterns on 7 examples at 1.00. In numpy it is `(F @ signs.T).max(axis=0).mean() / n`, with `signs` of shape `(num_draws, n)`.

This score has a payoff. For a loss between 0 and 1, with probability at least $1 - \delta$:

$$
\text{true error} \le \text{training error} + 2\hat{R} + 3\sqrt{\frac{\log(2/\delta)}{2n}}
$$

The discount is twice the noise-fitting score plus a small term that shrinks with $n$. When $\hat{R}$ is near 1, the discount exceeds 2 and the bound says nothing for a loss that cannot exceed 1.

> [!KEY]
> The best of 20 scores is too good, because choosing the lowest picks the luckiest. Score the choice on houses it never saw, with nested cross-validation, and expect the discount to grow with the number of settings you try and with how well your models could fit noise.

> [!RECAP]
> - The lowest of many noisy scores sits below the truth, by about 1.9 noise spreads for 20 equal models.
> - Nested cross-validation chooses the penalty inside each outer training set and scores it on the outer fold.
> - The finite-class bound grows as the log of the number of models, and needs a loss between 0 and 1.
> - The random-sign score is near 0 for one model and 1 for a set that matches every sign pattern.

## Build it step by step

You write three functions. The file also provides `k_fold_indices` and `cross_val_score` from the cross-validation chapter, already implemented, and `finite_class_bound`, the formula above.

1. **`best_of_m_gap(true_error, noise_sd, m, trials, rng)`**. Simulates `trials` repeats of choosing the lowest of `m` equal models. Returns one float: the average of `true_error` minus the lowest estimate. The common mistake is to take `max` and measure the wrong tail.

   ```python
   estimates = true_error + noise_sd * rng.standard_normal((trials, m))
   winner = estimates.min(axis=1)     # shape (trials,)
   ```

2. **`empirical_rademacher(losses_matrix, rng, num_draws)`**. Takes `losses_matrix` of shape `(num_models, n)`. Returns one float. The common mistake is `max(axis=1)`, which keeps the best draw for each model and not the best model for each draw.

   ```python
   signs = rng.choice(np.array([-1.0, 1.0]), size=(num_draws, n))
   scores = losses_matrix @ signs.T   # shape (num_models, num_draws)
   ```

3. **`nested_cv_score(X, y, lambdas, outer_k, inner_k)`**. Builds the outer folds with `k_fold_indices(n, outer_k, np.random.default_rng(0))`. For each, chooses the penalty with `cross_val_score` on the outer training rows, refits, and scores the outer fold. Returns the mean of the outer fold errors. The common mistake is to choose the penalty with the whole data set, which puts the outer fold back into the choice.

## Where this breaks

**Tiny data.** With a few dozen houses, nested folds are small, and the honest score is itself noisy. Compare fewer settings.

**Reusing the honest score.** If you adjust the recipe after seeing the nested score, you have started a fresh search, and the nested score is no longer honest.

**Loose bounds.** Neither bound is tight, and both need a loss between 0 and 1. Use them to see why the discount grows, not to quote a number.

## The exercise

You implement `best_of_m_gap`, `empirical_rademacher`, and `nested_cv_score`. The starter file ends with a program that prints the winner's gap for 1, 2, 5, 20, and 100 models, the noise-fitting score for 1 and 300 models, and the lucky and honest scores on the 60 houses.

The checks confirm that `best_of_m_gap` is 0 for one model and about 0.187 for 20, and that it grows with `m`, doubles when the noise doubles, and ignores the true error. They confirm that `empirical_rademacher` is near 0 for one model, grows for 300, gives 0.5 on the two-model example, and gives 1 for all 128 sign patterns. They confirm that `nested_cv_score` equals plain cross-validation for a single penalty and finds the exact penalty on noise-free data. The last check confirms that on pure noise the nested score is clearly higher than the lowest of 20 scores.
