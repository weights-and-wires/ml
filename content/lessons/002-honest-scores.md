---
title: "Honest scores: training error, test error, and holding data back"
slug: 002-honest-scores
part: "Part 1 — Foundations"
summary: A rule that has seen the answers scores too well, so you hold some flats back, fit on the rest, and score on the flats it never saw.
minutes: 35
gpu: false
objectives:
  - Explain why a rule that memorises its training flats scores zero error and still fails on flats it has not seen.
  - Split a dataset into a training set and a test set with a seeded shuffle that keeps each size paired with its rent.
  - Compute the generalisation gap, the test risk minus the training risk, for a rule.
  - Name two ways a held-out score can still mislead, and say how a single split adds noise.
prereqs:
  - 001-scoring-a-guess
exercise: 002-honest-scores
---

# Honest scores: training error, test error, and holding data back

Four flats were enough to compare two slopes. Next, you keep a folder of 20 flats with their rents, and a friend proposes a rule: to guess the rent of a flat, find the flat in your folder with the closest size and copy its rent. You score the rule on the folder with the average squared miss from the previous chapter, "Scoring a guess: loss and empirical risk". The score is exactly 0. Every guess is exactly right.

Is the rule perfect? It cannot be. The rule never learned anything about rents. It looked each answer up in the list it was given. Show it a flat that is not in the folder, and it copies the rent of a neighbour that may not resemble the unseen flat at all.

This chapter explains why the folder score lies, and how to get a score you can trust: hold some flats back before you fit, and score the rule only on those.

## What you already know

You already use these ideas:

- The empirical risk is the average loss over the flats you score.
- A squared loss of 0 means a perfect guess, and bigger numbers mean bigger misses.
- A seed makes a random generator repeat the same numbers, so a result can be reproduced.
- A list of positions, such as `[4, 19, 6]`, can pick several entries from an array at once.
- Rounding down to a whole number is called taking the floor.

## A rule that memorises

Four real flats are too few to show what goes wrong, and you cannot know the true pattern behind real rents. So from this chapter on, the running example is a made-up market where you do know it. Measure flat size on a scale from 0 (the smallest flat in the folder) to 1 (the largest). Measure rent as how many hundreds of pounds the flat sits above or below the typical rent. In this market the rent follows a smooth wave in the size, plus a random wobble from the landlord's mood. The wave is the pattern worth learning, and the wobble is chance that a new flat will not repeat. Figure 1 shows 20 such flats.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Twenty flats: a memorising rule passes through every training flat and misses the held-out flats">
<line x1="50.0" y1="290.0" x2="580.0" y2="290.0" class="stroke-muted" />
<line x1="50.0" y1="30.0" x2="50.0" y2="290.0" class="stroke-muted" />
<line x1="50.0" y1="236.5" x2="580.0" y2="236.5" class="stroke-grid" />
<text x="42.0" y="240.5" text-anchor="end" class="fig-small">−1</text>
<line x1="50.0" y1="160.0" x2="580.0" y2="160.0" class="stroke-grid" />
<text x="42.0" y="164.0" text-anchor="end" class="fig-small">0</text>
<line x1="50.0" y1="83.5" x2="580.0" y2="83.5" class="stroke-grid" />
<text x="42.0" y="87.5" text-anchor="end" class="fig-small">1</text>
<text x="50.0" y="308.0" text-anchor="middle" class="fig-small">0</text>
<text x="315.0" y="308.0" text-anchor="middle" class="fig-small">0.5</text>
<text x="580.0" y="308.0" text-anchor="middle" class="fig-small">1</text>
<text x="315.0" y="325.0" text-anchor="middle" class="fig-small">flat size (0 = smallest in the folder, 1 = largest)</text>
<text x="8.0" y="22.0" class="fig-small">rent, above or below typical (£100)</text>
<polyline points="50.0,163.5 55.1,163.5 55.1,144.0 100.9,144.0 100.9,83.6 168.0,83.6 168.0,52.8 200.9,52.8 200.9,69.2 272.9,69.2 272.9,182.7 337.5,182.7 337.5,191.2 362.9,191.2 362.9,220.9 412.1,220.9 412.1,257.0 436.7,257.0 436.7,230.9 459.6,230.9 459.6,253.2 493.4,253.2 493.4,223.4 506.0,223.4 506.0,232.9 520.6,232.9 520.6,197.7 539.7,197.7 539.7,185.3 580.0,185.3" fill="none" class="stroke-b" />
<line x1="481.0" y1="209.8" x2="481.0" y2="253.2" class="stroke-bad" />
<line x1="274.0" y1="90.0" x2="274.0" y2="182.7" class="stroke-bad" />
<line x1="371.5" y1="224.5" x2="371.5" y2="220.9" class="stroke-bad" />
<line x1="71.7" y1="155.8" x2="71.7" y2="144.0" class="stroke-bad" />
<line x1="67.8" y1="131.6" x2="67.8" y2="144.0" class="stroke-bad" />
<circle cx="507.5" cy="232.9" r="5" class="fill-a"/>
<circle cx="58.8" cy="144.0" r="5" class="fill-a"/>
<circle cx="51.5" cy="163.5" r="5" class="fill-a"/>
<circle cx="482.4" cy="253.2" r="5" class="fill-a"/>
<circle cx="338.1" cy="191.2" r="5" class="fill-a"/>
<circle cx="387.6" cy="220.9" r="5" class="fill-a"/>
<circle cx="504.4" cy="223.4" r="5" class="fill-a"/>
<circle cx="436.6" cy="257.0" r="5" class="fill-a"/>
<circle cx="533.8" cy="197.7" r="5" class="fill-a"/>
<circle cx="208.8" cy="69.2" r="5" class="fill-a"/>
<circle cx="337.0" cy="182.7" r="5" class="fill-a"/>
<circle cx="436.7" cy="230.9" r="5" class="fill-a"/>
<circle cx="545.6" cy="185.3" r="5" class="fill-a"/>
<circle cx="193.0" cy="52.8" r="5" class="fill-a"/>
<circle cx="143.1" cy="83.6" r="5" class="fill-a"/>
<circle cx="481.0" cy="209.8" r="5.5" class="fill-paper stroke-a"/>
<circle cx="274.0" cy="90.0" r="5.5" class="fill-paper stroke-a"/>
<circle cx="371.5" cy="224.5" r="5.5" class="fill-paper stroke-a"/>
<circle cx="71.7" cy="155.8" r="5.5" class="fill-paper stroke-a"/>
<circle cx="67.8" cy="131.6" r="5.5" class="fill-paper stroke-a"/>
<circle cx="400.0" cy="50.0" r="5" class="fill-a"/>
<text x="410.0" y="54.0" class="fig-small">15 training flats</text>
<circle cx="400.0" cy="70.0" r="5.5" class="fill-paper stroke-a"/>
<text x="410.0" y="74.0" class="fig-small">5 held-out flats</text>
<line x1="392.0" y1="90.0" x2="408.0" y2="90.0" class="stroke-b" />
<text x="414.0" y="94.0" class="fig-small">memorising rule</text>
<line x1="400.0" y1="106.0" x2="400.0" y2="114.0" class="stroke-bad" />
<text x="414.0" y="114.0" class="fig-small">held-out miss</text>
</svg>
Figure 1. Twenty flats and a memorising rule. The orange staircase hits every blue training dot, so the training error is 0. Notice the red segments: the rule misses the hollow held-out flats, one of them by more than 1 (the segment at size 0.42 runs along a step edge).
```

The memorising rule is the orange staircase in Figure 1. For any size, it returns the rent of the closest training flat. It passes through every blue dot, so its score on those flats is 0.

Take a tiny case with real numbers. Two training flats have sizes 0.2 and 0.6 and rents 1.0 and −0.5. A third flat, which the rule has not seen, has size 0.35 and rent 0.7. The closest training size is 0.2, because $|0.35-0.2|=0.15$ and $|0.35-0.6|=0.25$. The rule guesses 1.0. The miss is 0.3 and the squared loss is 0.09. On its own two flats the rule scores 0. On the third flat it scores 0.09. The gap between those two scores is the lie.

Figure 1 shows the same thing at scale. The five hollow dots are flats the rule never saw. Their squared losses are 0.322, 1.467, 0.002, 0.024, and 0.026. The average is $1.842/5=0.368$. A rule that scores 0 on 15 flats and 0.368 on 5 others has not learned the market.

You may know the exam version: memorise last year's paper and score 100, then score much less on another paper. The 100 measures your memory, not your understanding.

## Why the folder score flatters

A *training set* is the flats you fit the rule to. The rule has seen their answers, so its score on them is the *training error*. The score on flats it has not seen is the *test error*.

Why is the training error too low? Every rent in the folder is the wave plus a wobble. The wobble is luck: the landlord of flat 7 happened to be in a good mood. A rule that is free to bend will bend toward that luck, because doing so lowers its score on the folder. The memoriser takes this to the limit. It copies each wobble as if it were a fact about the market. But the next flat arrives with its own wobble, so the copied luck does not help and sometimes hurts. The training error counts the luck as skill, and the test error does not.

This matters because searching for the lowest training score would pick the memoriser every time. Picking the rule with the lowest average loss on the training data is called *empirical risk minimisation*, and it is a sensible way to fit. The trouble is only in what you report afterwards. The score you minimised cannot also be the score you trust, because the rule was tuned against it.

Why not score on every flat you own, to use all the data? Because the fit already used all of them. You need flats the fit never touched.

## Hold some flats back

Set aside a fraction of the flats before you fit anything. With $n=20$ flats and a fraction 0.25, you get $0.25\times20=5$ test flats and 15 training flats. You fit on the 15 and score on the 5. Those 5 play the role of next year's flats.

Why shuffle first, and not take the last 5 rows? If the folder is sorted by size, the last 5 flats are the five biggest, and the rule would be tested on a kind of flat it never trained on. A random shuffle makes the test flats look like the training flats on average. The rule is then tested on the same kind of flat it was fit to, but on different individual flats.

Shuffle positions, not the two arrays separately. The permutation is a list of positions from 0 to 19 in random order. Use the same list on both the sizes and the rents, and each size keeps its own rent. Two independent shuffles would pair 30 square metres with somebody else's rent. Figure 2 shows one permutation, with the five test flats first.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Twenty flats in their original order, then shuffled by a seeded permutation into five test flats and fifteen training flats">
<text x="20.0" y="22.0" class="fig-bold">Original order (flat numbers 0 to 19)</text>
<text x="20.0" y="232.0" class="fig-bold">After the shuffle: 5 test flats, then 15 training flats</text>
<rect x="20.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="34.0" y="53.0" text-anchor="middle" class="fig-small">0</text>
<rect x="48.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="62.0" y="53.0" text-anchor="middle" class="fig-small">1</text>
<rect x="76.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="90.0" y="53.0" text-anchor="middle" class="fig-small">2</text>
<rect x="104.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="118.0" y="53.0" text-anchor="middle" class="fig-small">3</text>
<rect x="132.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="146.0" y="53.0" text-anchor="middle" class="fig-small">4</text>
<rect x="160.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="174.0" y="53.0" text-anchor="middle" class="fig-small">5</text>
<rect x="188.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="202.0" y="53.0" text-anchor="middle" class="fig-small">6</text>
<rect x="216.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="230.0" y="53.0" text-anchor="middle" class="fig-small">7</text>
<rect x="244.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="258.0" y="53.0" text-anchor="middle" class="fig-small">8</text>
<rect x="272.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="286.0" y="53.0" text-anchor="middle" class="fig-small">9</text>
<rect x="300.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="314.0" y="53.0" text-anchor="middle" class="fig-small">10</text>
<rect x="328.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="342.0" y="53.0" text-anchor="middle" class="fig-small">11</text>
<rect x="356.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="370.0" y="53.0" text-anchor="middle" class="fig-small">12</text>
<rect x="384.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="398.0" y="53.0" text-anchor="middle" class="fig-small">13</text>
<rect x="412.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="426.0" y="53.0" text-anchor="middle" class="fig-small">14</text>
<rect x="440.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="454.0" y="53.0" text-anchor="middle" class="fig-small">15</text>
<rect x="468.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="482.0" y="53.0" text-anchor="middle" class="fig-small">16</text>
<rect x="496.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="510.0" y="53.0" text-anchor="middle" class="fig-small">17</text>
<rect x="524.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="538.0" y="53.0" text-anchor="middle" class="fig-small">18</text>
<rect x="552.0" y="36.0" width="28.0" height="26.0" class="fill-sunk stroke-muted" />
<text x="566.0" y="53.0" text-anchor="middle" class="fig-small">19</text>
<line x1="146.0" y1="62.0" x2="34.0" y2="170.0" class="stroke-b" />
<line x1="566.0" y1="62.0" x2="62.0" y2="170.0" class="stroke-b" />
<line x1="202.0" y1="62.0" x2="90.0" y2="170.0" class="stroke-b" />
<line x1="90.0" y1="62.0" x2="118.0" y2="170.0" class="stroke-b" />
<line x1="398.0" y1="62.0" x2="146.0" y2="170.0" class="stroke-b" />
<line x1="482.0" y1="62.0" x2="174.0" y2="170.0" class="stroke-grid" />
<line x1="118.0" y1="62.0" x2="202.0" y2="170.0" class="stroke-grid" />
<line x1="342.0" y1="62.0" x2="230.0" y2="170.0" class="stroke-grid" />
<line x1="314.0" y1="62.0" x2="258.0" y2="170.0" class="stroke-grid" />
<line x1="258.0" y1="62.0" x2="286.0" y2="170.0" class="stroke-grid" />
<line x1="34.0" y1="62.0" x2="314.0" y2="170.0" class="stroke-grid" />
<line x1="370.0" y1="62.0" x2="342.0" y2="170.0" class="stroke-grid" />
<line x1="230.0" y1="62.0" x2="370.0" y2="170.0" class="stroke-grid" />
<line x1="174.0" y1="62.0" x2="398.0" y2="170.0" class="stroke-grid" />
<line x1="538.0" y1="62.0" x2="426.0" y2="170.0" class="stroke-grid" />
<line x1="510.0" y1="62.0" x2="454.0" y2="170.0" class="stroke-grid" />
<line x1="426.0" y1="62.0" x2="482.0" y2="170.0" class="stroke-grid" />
<line x1="286.0" y1="62.0" x2="510.0" y2="170.0" class="stroke-grid" />
<line x1="62.0" y1="62.0" x2="538.0" y2="170.0" class="stroke-grid" />
<line x1="454.0" y1="62.0" x2="566.0" y2="170.0" class="stroke-grid" />
<rect x="20.0" y="170.0" width="28.0" height="26.0" class="fill-soft-b stroke-b" />
<text x="34.0" y="187.0" text-anchor="middle" class="fig-small">4</text>
<rect x="48.0" y="170.0" width="28.0" height="26.0" class="fill-soft-b stroke-b" />
<text x="62.0" y="187.0" text-anchor="middle" class="fig-small">19</text>
<rect x="76.0" y="170.0" width="28.0" height="26.0" class="fill-soft-b stroke-b" />
<text x="90.0" y="187.0" text-anchor="middle" class="fig-small">6</text>
<rect x="104.0" y="170.0" width="28.0" height="26.0" class="fill-soft-b stroke-b" />
<text x="118.0" y="187.0" text-anchor="middle" class="fig-small">2</text>
<rect x="132.0" y="170.0" width="28.0" height="26.0" class="fill-soft-b stroke-b" />
<text x="146.0" y="187.0" text-anchor="middle" class="fig-small">13</text>
<rect x="160.0" y="170.0" width="28.0" height="26.0" class="fill-soft-a stroke-a" />
<text x="174.0" y="187.0" text-anchor="middle" class="fig-small">16</text>
<rect x="188.0" y="170.0" width="28.0" height="26.0" class="fill-soft-a stroke-a" />
<text x="202.0" y="187.0" text-anchor="middle" class="fig-small">3</text>
<rect x="216.0" y="170.0" width="28.0" height="26.0" class="fill-soft-a stroke-a" />
<text x="230.0" y="187.0" text-anchor="middle" class="fig-small">11</text>
<rect x="244.0" y="170.0" width="28.0" height="26.0" class="fill-soft-a stroke-a" />
<text x="258.0" y="187.0" text-anchor="middle" class="fig-small">10</text>
<rect x="272.0" y="170.0" width="28.0" height="26.0" class="fill-soft-a stroke-a" />
<text x="286.0" y="187.0" text-anchor="middle" class="fig-small">8</text>
<rect x="300.0" y="170.0" width="28.0" height="26.0" class="fill-soft-a stroke-a" />
<text x="314.0" y="187.0" text-anchor="middle" class="fig-small">0</text>
<rect x="328.0" y="170.0" width="28.0" height="26.0" class="fill-soft-a stroke-a" />
<text x="342.0" y="187.0" text-anchor="middle" class="fig-small">12</text>
<rect x="356.0" y="170.0" width="28.0" height="26.0" class="fill-soft-a stroke-a" />
<text x="370.0" y="187.0" text-anchor="middle" class="fig-small">7</text>
<rect x="384.0" y="170.0" width="28.0" height="26.0" class="fill-soft-a stroke-a" />
<text x="398.0" y="187.0" text-anchor="middle" class="fig-small">5</text>
<rect x="412.0" y="170.0" width="28.0" height="26.0" class="fill-soft-a stroke-a" />
<text x="426.0" y="187.0" text-anchor="middle" class="fig-small">18</text>
<rect x="440.0" y="170.0" width="28.0" height="26.0" class="fill-soft-a stroke-a" />
<text x="454.0" y="187.0" text-anchor="middle" class="fig-small">17</text>
<rect x="468.0" y="170.0" width="28.0" height="26.0" class="fill-soft-a stroke-a" />
<text x="482.0" y="187.0" text-anchor="middle" class="fig-small">14</text>
<rect x="496.0" y="170.0" width="28.0" height="26.0" class="fill-soft-a stroke-a" />
<text x="510.0" y="187.0" text-anchor="middle" class="fig-small">9</text>
<rect x="524.0" y="170.0" width="28.0" height="26.0" class="fill-soft-a stroke-a" />
<text x="538.0" y="187.0" text-anchor="middle" class="fig-small">1</text>
<rect x="552.0" y="170.0" width="28.0" height="26.0" class="fill-soft-a stroke-a" />
<text x="566.0" y="187.0" text-anchor="middle" class="fig-small">15</text>
<text x="90.0" y="212.0" text-anchor="middle" class="fig-small">test</text>
<text x="370.0" y="212.0" text-anchor="middle" class="fig-small">train</text>
</svg>
Figure 2. One seeded shuffle of the flat numbers. Every flat moves as a pair, with its size and rent together. Notice that the test flats are scattered across the original order, not the last five rows.
```

A seed fixes the shuffle, so a colleague can reproduce your split. The first `n_test` positions of the permutation are the test flats, and the rest are the training flats:

```python
order = np.random.default_rng(seed).permutation(n)   # positions 0..n-1 in random order
test_idx, train_idx = order[:n_test], order[n_test:]
x_train, y_train = x[train_idx], y[train_idx]        # the same index list for x and y
```

The count rounds down: $\lfloor0.25\times20\rfloor=5$. Beware of floating-point arithmetic. In Python, `0.29 * 100` is 28.999999999999996, so a plain `int(...)` gives 28 test flats when the answer is 29. Add a tiny nudge before you truncate, as in `int(test_size * n + 1e-9)`.

> [!KEY]
> A held-out score is honest because the rule never saw those flats while it was being fit. The training score measures memory, and the test score measures what the rule does with a flat it has not met.

## One split is noisy

The held-out score is a fair estimate, but it comes from only five flats. Split the same 20 flats with 30 different seeds and score the memoriser on each test set. Figure 3 shows the results. The training risk is 0 on every split. The test risk ranges from 0.028 to 0.385, with a mean of 0.131. The split in Figure 1 happens to be one of the unlucky ones, at 0.368.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Held-out risk of the memorising rule on thirty different random splits, all above the training risk of zero">
<line x1="60.0" y1="220.0" x2="570.0" y2="220.0" class="stroke-muted" />
<line x1="60.0" y1="40.0" x2="60.0" y2="220.0" class="stroke-grid" />
<text x="60.0" y="238.0" text-anchor="middle" class="fig-small">0.0</text>
<line x1="173.3" y1="40.0" x2="173.3" y2="220.0" class="stroke-grid" />
<text x="173.3" y="238.0" text-anchor="middle" class="fig-small">0.1</text>
<line x1="286.7" y1="40.0" x2="286.7" y2="220.0" class="stroke-grid" />
<text x="286.7" y="238.0" text-anchor="middle" class="fig-small">0.2</text>
<line x1="400.0" y1="40.0" x2="400.0" y2="220.0" class="stroke-grid" />
<text x="400.0" y="238.0" text-anchor="middle" class="fig-small">0.3</text>
<line x1="513.3" y1="40.0" x2="513.3" y2="220.0" class="stroke-grid" />
<text x="513.3" y="238.0" text-anchor="middle" class="fig-small">0.4</text>
<text x="315.0" y="262.0" text-anchor="middle" class="fig-small">held-out risk (mean squared loss), one dot per seed</text>
<circle cx="91.4" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="113.4" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="116.4" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="118.0" cy="186.0" r="5.5" class="fill-b"/>
<circle cx="123.8" cy="173.0" r="5.5" class="fill-b"/>
<circle cx="127.8" cy="160.0" r="5.5" class="fill-b"/>
<circle cx="130.1" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="138.4" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="138.8" cy="186.0" r="5.5" class="fill-b"/>
<circle cx="143.1" cy="173.0" r="5.5" class="fill-b"/>
<circle cx="157.0" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="159.4" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="160.5" cy="186.0" r="5.5" class="fill-b"/>
<circle cx="162.3" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="169.6" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="173.0" cy="186.0" r="5.5" class="fill-b"/>
<circle cx="183.5" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="186.3" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="190.9" cy="186.0" r="5.5" class="fill-b"/>
<circle cx="193.1" cy="173.0" r="5.5" class="fill-b"/>
<circle cx="205.8" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="216.4" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="219.0" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="224.0" cy="186.0" r="5.5" class="fill-b"/>
<circle cx="255.2" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="258.1" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="455.7" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="477.5" cy="212.0" r="5.5" class="fill-b"/>
<circle cx="481.3" cy="199.0" r="5.5" class="fill-b"/>
<circle cx="496.3" cy="212.0" r="5.5" class="fill-b"/>
<line x1="60.0" y1="40.0" x2="60.0" y2="220.0" class="stroke-a" />
<text x="68.0" y="56.0" class="fig-small">training risk: 0 on every split</text>
<line x1="208.9" y1="90.0" x2="208.9" y2="220.0" class="stroke-c dashed" />
<text x="214.9" y="104.0" class="fig-small">mean 0.131</text>
</svg>
Figure 3. The same rule scored on thirty random splits of the same flats. Notice that the training risk sits at 0 on every split, to the left of the whole cloud, and that the held-out risk jumps around.
```

Two lessons follow. A single test score has noise because a few flats carry it, so do not read much into small differences. A bigger test set steadies the score, but every flat you hold back is a flat the fit cannot learn from, so there is a trade-off. Also, touch the test set once. If you try ten rules and keep the one with the best test score, you have tuned to the test flats, and the score flatters again.

The difference between the two risks has a name. The *generalisation gap* is the test risk minus the training risk. For the split in Figure 1 it is $0.368-0=0.368$.

## When even a held-out score misleads

A test score estimates future error only when the test flats resemble the flats you will meet. Statisticians call this condition *i.i.d.*, short for independent and identically distributed. It has two parts, and each one fails in a concrete way.

*Identically distributed* means every flat comes from the same market. If your folder holds London flats and you will rent in Leeds, the test flats come from the wrong market.

*Independent* means one flat does not give away another. Suppose the folder holds ten adverts for the same building. A shuffle can put one advert in the training set and its twin in the test set. The memoriser copies the exact rent and scores brilliantly, and the score says nothing about another building.

## Memorise and measure

The memorising rule is a nearest-neighbour lookup. Compare the size of each flat you want to guess to every training size, and take the closest:

```python
d = np.abs(x_new[:, None] - x_train[None, :])   # shape (m, n): distance from each queried flat to each training flat
nearest = np.argmin(d, axis=1)                  # closest training flat for each queried flat
guesses = y_train[nearest]
```

The gap function chains the three tools: split, memorise, and score with `empirical_risk` from the previous chapter.

> [!RECAP]
> - The training error is the score on flats the rule was fit to. It is too low on average.
> - Hold flats back first, fit on the rest, and score on the held-out flats.
> - Shuffle positions once with a seed, and index sizes and rents with the same positions.
> - The generalisation gap is the test risk minus the training risk.
> - One split is noisy, and the score misleads when flats are not independent or not from the same market.

## Build it step by step

The starter file provides `squared_loss` and `empirical_risk`, which you wrote in the previous chapter. You write three functions, in the order the checks test them.

1. **`train_test_split(x, y, test_size, seed)`** returns `(x_train, y_train, x_test, y_test)`. It takes `int(test_size * n + 1e-9)` test flats:

   ```python
   order = np.random.default_rng(seed).permutation(len(x))
   # The first n_test positions are the test flats. The rest are training flats.
   ```

   Do not shuffle `x` and `y` separately. That breaks every pair.

2. **`memoriser_predict(x_train, y_train, x_new)`** returns an array of shape `(m,)`, one rent per size you ask about:

   ```python
   d = np.abs(x_new[:, None] - x_train[None, :])   # (m, n)
   # argmin along axis 1 gives the nearest training flat for each queried flat.
   ```

   `axis=0` finds the nearest queried flat for each training flat, which is the wrong question.

3. **`generalisation_gap(x, y, test_size, seed)`** splits, memorises, and returns `(train_risk, test_risk, gap)` as Python floats, using squared loss:

   ```python
   x_train, y_train, x_test, y_test = train_test_split(x, y, test_size, seed)
   # Score the memoriser on the training flats, then on the test flats.
   ```

   Scoring only on the test flats would leave out the training risk, which must be 0.

## Where this breaks

**Rents go stale.** If rents change after you collect them, later flats come from a different market, and the test score understates the error you meet. Refit on recent rents.

**You pick the best test score.** Choosing among many rules by their test score tunes the rule to the test flats. Keep a separate set of flats for choosing between rules, and use the test set once.

**The test set is tiny.** Ten test flats give a fair average and a wild single score, as Figure 3 shows. The chapter "Why training error lies, and how k-fold cross-validation fixes it" shows a way to use every flat for testing without cheating.

**Neighbouring rows leak.** Adverts for the same building, or the same person measured twice, split across train and test make the score optimistic. Split by building, not by row.

## The exercise

You build `train_test_split`, `memoriser_predict`, and `generalisation_gap`. The split checks that every flat lands in exactly one set, that each size stays with its rent, that the same seed repeats the split and different seeds differ, and that the test count rounds the way the chapter describes.

The memoriser must copy the rent of the nearest training size, and it must score exactly 0 on its own training flats. The gap check compares your numbers with an independent computation on seeded flats, and confirms that the test risk is above the training risk.
