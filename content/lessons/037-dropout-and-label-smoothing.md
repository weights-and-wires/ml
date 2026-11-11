---
title: "Dropout and label smoothing: keep a network from trusting itself"
slug: 037-dropout-and-label-smoothing
part: "Part 4 — Neural networks"
summary: Switch hidden units off at random so none can lean on a partner, then soften the one-hot target so scores stop growing without bound.
minutes: 70
gpu: false
objectives:
  - Explain how units that always appear together memorise the training set, and how random removal stops it.
  - Compute inverted dropout by hand, say why the survivors are divided by 1 - p, and write its backward pass.
  - Explain why a loss against one-hot targets keeps pushing scores upward, and what that costs on a wrong or mislabelled example.
  - Build the smoothed target, its loss with a max-shifted log-softmax, and its gradient.
  - Compute where the smoothed loss is lowest, and the score gap that corresponds to it.
prereqs:
  - 031-backpropagation
  - 032-softmax-and-whole-network-gradients
  - 036-weight-decay-and-early-stopping
exercise: 037-dropout-and-label-smoothing
---

# Dropout and label smoothing: keep a network from trusting itself

A hidden layer has four units. Suppose that on your training set, unit 2 tends to overshoot, and unit 3 learns to correct it by pushing the other way. Together the pair gets the training examples right. Neither unit is useful alone: unit 3 has learned a correction that only makes sense when unit 2 is present and overshooting in its usual way. On an unseen example, unit 2 does something different, and unit 3's correction makes things worse.

This is **co-adaptation**: units become tuned to each other instead of to the data. Nothing in the loss complains, because on the training set everyone is always present. A second kind of self-trust sits at the output. A classifier looks at photos of cats, dogs, and foxes and, after enough training, says "100 percent dog" on every dog in the training set. Then it meets a fox that looks like a dog and says "100 percent dog" again. It is wrong and completely sure of itself, and it earns that habit honestly, because the loss rewards every extra bit of certainty.

This chapter gives one remedy for each. **Dropout** attacks co-adaptation by switching units off at random while you train. **Label smoothing** attacks overconfidence by asking the model for 93 percent instead of 100, using only a change to the target.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Three copies of a small network with four inputs, six hidden units, and two outputs. In each copy a different random set of hidden units is dropped, drawn as dashed circles with dashed edges.">
<line x1="40" y1="60" x2="105" y2="44" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="100" x2="105" y2="44" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="140" x2="105" y2="44" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="180" x2="105" y2="44" class="stroke-muted" stroke-width="0.8"/>
<line x1="105" y1="44" x2="170" y2="100" class="stroke-muted" stroke-width="0.8"/>
<line x1="105" y1="44" x2="170" y2="140" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="60" x2="105" y2="76" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="100" x2="105" y2="76" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="140" x2="105" y2="76" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="180" x2="105" y2="76" class="stroke-muted" stroke-width="0.8"/>
<line x1="105" y1="76" x2="170" y2="100" class="stroke-muted" stroke-width="0.8"/>
<line x1="105" y1="76" x2="170" y2="140" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="60" x2="105" y2="108" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="40" y1="100" x2="105" y2="108" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="40" y1="140" x2="105" y2="108" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="40" y1="180" x2="105" y2="108" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="105" y1="108" x2="170" y2="100" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="105" y1="108" x2="170" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="40" y1="60" x2="105" y2="140" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="100" x2="105" y2="140" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="140" x2="105" y2="140" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="180" x2="105" y2="140" class="stroke-muted" stroke-width="0.8"/>
<line x1="105" y1="140" x2="170" y2="100" class="stroke-muted" stroke-width="0.8"/>
<line x1="105" y1="140" x2="170" y2="140" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="60" x2="105" y2="172" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="100" x2="105" y2="172" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="140" x2="105" y2="172" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="180" x2="105" y2="172" class="stroke-muted" stroke-width="0.8"/>
<line x1="105" y1="172" x2="170" y2="100" class="stroke-muted" stroke-width="0.8"/>
<line x1="105" y1="172" x2="170" y2="140" class="stroke-muted" stroke-width="0.8"/>
<line x1="40" y1="60" x2="105" y2="204" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="40" y1="100" x2="105" y2="204" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="40" y1="140" x2="105" y2="204" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="40" y1="180" x2="105" y2="204" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="105" y1="204" x2="170" y2="100" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="105" y1="204" x2="170" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<circle cx="40" cy="60" r="7" class="fill-a"/>
<circle cx="40" cy="100" r="7" class="fill-a"/>
<circle cx="40" cy="140" r="7" class="fill-a"/>
<circle cx="40" cy="180" r="7" class="fill-a"/>
<circle cx="170" cy="100" r="7" class="fill-d"/>
<circle cx="170" cy="140" r="7" class="fill-d"/>
<circle cx="105" cy="44" r="8" class="fill-b"/>
<circle cx="105" cy="76" r="8" class="fill-b"/>
<circle cx="105" cy="108" r="8" class="fill-paper stroke-muted dashed"/>
<circle cx="105" cy="140" r="8" class="fill-b"/>
<circle cx="105" cy="172" r="8" class="fill-b"/>
<circle cx="105" cy="204" r="8" class="fill-paper stroke-muted dashed"/>
<text x="105.0" y="24.0" text-anchor="middle" class="fig-bold">Step 1: mask 110110</text>
<text x="105.0" y="246.0" text-anchor="middle" class="fig-small">4 of 6 units on</text>
<line x1="235" y1="60" x2="300" y2="44" class="stroke-muted" stroke-width="0.8"/>
<line x1="235" y1="100" x2="300" y2="44" class="stroke-muted" stroke-width="0.8"/>
<line x1="235" y1="140" x2="300" y2="44" class="stroke-muted" stroke-width="0.8"/>
<line x1="235" y1="180" x2="300" y2="44" class="stroke-muted" stroke-width="0.8"/>
<line x1="300" y1="44" x2="365" y2="100" class="stroke-muted" stroke-width="0.8"/>
<line x1="300" y1="44" x2="365" y2="140" class="stroke-muted" stroke-width="0.8"/>
<line x1="235" y1="60" x2="300" y2="76" class="stroke-muted" stroke-width="0.8"/>
<line x1="235" y1="100" x2="300" y2="76" class="stroke-muted" stroke-width="0.8"/>
<line x1="235" y1="140" x2="300" y2="76" class="stroke-muted" stroke-width="0.8"/>
<line x1="235" y1="180" x2="300" y2="76" class="stroke-muted" stroke-width="0.8"/>
<line x1="300" y1="76" x2="365" y2="100" class="stroke-muted" stroke-width="0.8"/>
<line x1="300" y1="76" x2="365" y2="140" class="stroke-muted" stroke-width="0.8"/>
<line x1="235" y1="60" x2="300" y2="108" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="235" y1="100" x2="300" y2="108" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="235" y1="140" x2="300" y2="108" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="235" y1="180" x2="300" y2="108" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="300" y1="108" x2="365" y2="100" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="300" y1="108" x2="365" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="235" y1="60" x2="300" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="235" y1="100" x2="300" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="235" y1="140" x2="300" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="235" y1="180" x2="300" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="300" y1="140" x2="365" y2="100" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="300" y1="140" x2="365" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="235" y1="60" x2="300" y2="172" class="stroke-muted" stroke-width="0.8"/>
<line x1="235" y1="100" x2="300" y2="172" class="stroke-muted" stroke-width="0.8"/>
<line x1="235" y1="140" x2="300" y2="172" class="stroke-muted" stroke-width="0.8"/>
<line x1="235" y1="180" x2="300" y2="172" class="stroke-muted" stroke-width="0.8"/>
<line x1="300" y1="172" x2="365" y2="100" class="stroke-muted" stroke-width="0.8"/>
<line x1="300" y1="172" x2="365" y2="140" class="stroke-muted" stroke-width="0.8"/>
<line x1="235" y1="60" x2="300" y2="204" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="235" y1="100" x2="300" y2="204" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="235" y1="140" x2="300" y2="204" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="235" y1="180" x2="300" y2="204" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="300" y1="204" x2="365" y2="100" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="300" y1="204" x2="365" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<circle cx="235" cy="60" r="7" class="fill-a"/>
<circle cx="235" cy="100" r="7" class="fill-a"/>
<circle cx="235" cy="140" r="7" class="fill-a"/>
<circle cx="235" cy="180" r="7" class="fill-a"/>
<circle cx="365" cy="100" r="7" class="fill-d"/>
<circle cx="365" cy="140" r="7" class="fill-d"/>
<circle cx="300" cy="44" r="8" class="fill-b"/>
<circle cx="300" cy="76" r="8" class="fill-b"/>
<circle cx="300" cy="108" r="8" class="fill-paper stroke-muted dashed"/>
<circle cx="300" cy="140" r="8" class="fill-paper stroke-muted dashed"/>
<circle cx="300" cy="172" r="8" class="fill-b"/>
<circle cx="300" cy="204" r="8" class="fill-paper stroke-muted dashed"/>
<text x="300.0" y="24.0" text-anchor="middle" class="fig-bold">Step 2: mask 110010</text>
<text x="300.0" y="246.0" text-anchor="middle" class="fig-small">3 of 6 units on</text>
<line x1="430" y1="60" x2="495" y2="44" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="430" y1="100" x2="495" y2="44" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="430" y1="140" x2="495" y2="44" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="430" y1="180" x2="495" y2="44" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="495" y1="44" x2="560" y2="100" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="495" y1="44" x2="560" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="430" y1="60" x2="495" y2="76" class="stroke-muted" stroke-width="0.8"/>
<line x1="430" y1="100" x2="495" y2="76" class="stroke-muted" stroke-width="0.8"/>
<line x1="430" y1="140" x2="495" y2="76" class="stroke-muted" stroke-width="0.8"/>
<line x1="430" y1="180" x2="495" y2="76" class="stroke-muted" stroke-width="0.8"/>
<line x1="495" y1="76" x2="560" y2="100" class="stroke-muted" stroke-width="0.8"/>
<line x1="495" y1="76" x2="560" y2="140" class="stroke-muted" stroke-width="0.8"/>
<line x1="430" y1="60" x2="495" y2="108" class="stroke-muted" stroke-width="0.8"/>
<line x1="430" y1="100" x2="495" y2="108" class="stroke-muted" stroke-width="0.8"/>
<line x1="430" y1="140" x2="495" y2="108" class="stroke-muted" stroke-width="0.8"/>
<line x1="430" y1="180" x2="495" y2="108" class="stroke-muted" stroke-width="0.8"/>
<line x1="495" y1="108" x2="560" y2="100" class="stroke-muted" stroke-width="0.8"/>
<line x1="495" y1="108" x2="560" y2="140" class="stroke-muted" stroke-width="0.8"/>
<line x1="430" y1="60" x2="495" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="430" y1="100" x2="495" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="430" y1="140" x2="495" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="430" y1="180" x2="495" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="495" y1="140" x2="560" y2="100" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="495" y1="140" x2="560" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="430" y1="60" x2="495" y2="172" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="430" y1="100" x2="495" y2="172" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="430" y1="140" x2="495" y2="172" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="430" y1="180" x2="495" y2="172" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="495" y1="172" x2="560" y2="100" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="495" y1="172" x2="560" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="430" y1="60" x2="495" y2="204" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="430" y1="100" x2="495" y2="204" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="430" y1="140" x2="495" y2="204" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="430" y1="180" x2="495" y2="204" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="495" y1="204" x2="560" y2="100" class="stroke-grid dashed" stroke-width="0.8"/>
<line x1="495" y1="204" x2="560" y2="140" class="stroke-grid dashed" stroke-width="0.8"/>
<circle cx="430" cy="60" r="7" class="fill-a"/>
<circle cx="430" cy="100" r="7" class="fill-a"/>
<circle cx="430" cy="140" r="7" class="fill-a"/>
<circle cx="430" cy="180" r="7" class="fill-a"/>
<circle cx="560" cy="100" r="7" class="fill-d"/>
<circle cx="560" cy="140" r="7" class="fill-d"/>
<circle cx="495" cy="44" r="8" class="fill-paper stroke-muted dashed"/>
<circle cx="495" cy="76" r="8" class="fill-b"/>
<circle cx="495" cy="108" r="8" class="fill-b"/>
<circle cx="495" cy="140" r="8" class="fill-paper stroke-muted dashed"/>
<circle cx="495" cy="172" r="8" class="fill-paper stroke-muted dashed"/>
<circle cx="495" cy="204" r="8" class="fill-paper stroke-muted dashed"/>
<text x="495.0" y="24.0" text-anchor="middle" class="fig-bold">Step 3: mask 011000</text>
<text x="495.0" y="246.0" text-anchor="middle" class="fig-small">2 of 6 units on</text>
</svg>
Figure 1. Three training steps, each with a different random mask on the six hidden units (orange: kept, dashed: dropped, p = 0.5). Every step trains a different thinned network, and all of them share the same weights. A unit cannot rely on a given neighbour, because that neighbour is often missing.
```

## What you already know

- Overfitting means a low training error and a high error on unseen data.
- Multiplying an array by another elementwise has a simple backward rule: the slope of $a \cdot b$ with respect to $a$ is $b$ ([backpropagation](/lessons/031-backpropagation)).
- Softmax turns a row of scores, called **logits**, into probabilities that sum to 1. The gradient of softmax followed by cross-entropy is the probabilities minus the target ([softmax and cross-entropy](/lessons/032-softmax-and-whole-network-gradients)).
- Subtracting the largest score from every score before exponentiating leaves the softmax unchanged and stops large scores from overflowing.
- [Weight decay and early stopping](/lessons/036-weight-decay-and-early-stopping) shrink weights. The two methods here do something else: dropout adds noise, and label smoothing changes the target.

## A crowd of thinned networks

On every training step, switch off each hidden unit at random with probability $p$. No unit can then count on any particular partner, because that partner vanishes on many steps. Look at Figure 1. Each panel is the same network with a different set of hidden units switched off. A layer of 4 units has $2^4 = 16$ ways to keep or drop its units, and a layer of 6 has 64. Each training step picks one at random and trains that **thinned network**, and the thinned networks share their weights.

Another way to see it: training one network with dropout trains a whole crowd of thinned networks at once. At test time you use the full layer, which behaves approximately like the average of the crowd. It is an approximation, because averaging 64 networks and running one network with scaled-down outputs are not identical when the layers are nonlinear. But each unit has been forced to be useful alongside many different sets of partners, so it generalises better.

You might ask why not use fewer units. A smaller layer cannot form as many partnerships, but it also has less room to represent what the task needs. Dropout keeps the full width at test time and teaches each unit to carry its own weight.

## A worked example, and the scale factor

Take a layer's four outputs $x = [2, 4, 6, 8]$ and drop each with probability $p = 0.5$. Suppose the random mask comes out $[1, 0, 1, 1]$, where 1 means kept. The masked output is $[2, 0, 6, 8]$, and it sums to 16. The layer sums to 20 without dropout.

That gap matters. Over all 16 masks, each unit is kept in exactly 8, so the masked sum averages $20/2 = 10$. Figure 2 shows 200 random masks: the sums cluster around 10 while test time delivers 20. The next layer would see inputs about half as large in training as at test time, and its weights would be tuned for the wrong scale.

```figure
<svg viewBox="0 0 600 246" role="img" aria-label="Two histograms of the layer sum over 200 random masks. Without scaling the bars cluster around 10, away from the test-time value of 20. With division by one minus p the bars cluster around 20, on the test-time value.">
<line x1="50.0" y1="194.0" x2="280.0" y2="194.0" class="stroke-muted"/>
<line x1="50.0" y1="194.0" x2="50.0" y2="44.0" class="stroke-muted"/>
<line x1="60.5" y1="194.0" x2="60.5" y2="198.0" class="stroke-muted"/>
<text x="60.5" y="211.0" text-anchor="middle" class="fig-small">0</text>
<line x1="112.7" y1="194.0" x2="112.7" y2="198.0" class="stroke-muted"/>
<text x="112.7" y="211.0" text-anchor="middle" class="fig-small">10</text>
<line x1="165.0" y1="194.0" x2="165.0" y2="198.0" class="stroke-muted"/>
<text x="165.0" y="211.0" text-anchor="middle" class="fig-small">20</text>
<line x1="217.3" y1="194.0" x2="217.3" y2="198.0" class="stroke-muted"/>
<text x="217.3" y="211.0" text-anchor="middle" class="fig-small">30</text>
<line x1="269.5" y1="194.0" x2="269.5" y2="198.0" class="stroke-muted"/>
<text x="269.5" y="211.0" text-anchor="middle" class="fig-small">40</text>
<line x1="46.0" y1="194.0" x2="50.0" y2="194.0" class="stroke-muted"/>
<text x="43.0" y="198.0" text-anchor="end" class="fig-small">0</text>
<line x1="46.0" y1="119.0" x2="50.0" y2="119.0" class="stroke-muted"/>
<text x="43.0" y="123.0" text-anchor="end" class="fig-small">25</text>
<line x1="46.0" y1="44.0" x2="50.0" y2="44.0" class="stroke-muted"/>
<text x="43.0" y="48.0" text-anchor="end" class="fig-small">50</text>
<text x="165.0" y="228.0" text-anchor="middle" class="fig-small">layer sum</text>
<rect x="56.3" y="164.0" width="8.4" height="30.0" class="fill-soft-a stroke-a"/>
<rect x="66.7" y="161.0" width="8.4" height="33.0" class="fill-soft-a stroke-a"/>
<rect x="77.2" y="152.0" width="8.4" height="42.0" class="fill-soft-a stroke-a"/>
<rect x="87.6" y="125.0" width="8.4" height="69.0" class="fill-soft-a stroke-a"/>
<rect x="98.1" y="116.0" width="8.4" height="78.0" class="fill-soft-a stroke-a"/>
<rect x="108.5" y="92.0" width="8.4" height="102.0" class="fill-soft-a stroke-a"/>
<rect x="119.0" y="125.0" width="8.4" height="69.0" class="fill-soft-a stroke-a"/>
<rect x="129.5" y="95.0" width="8.4" height="99.0" class="fill-soft-a stroke-a"/>
<rect x="139.9" y="167.0" width="8.4" height="27.0" class="fill-soft-a stroke-a"/>
<rect x="150.4" y="158.0" width="8.4" height="36.0" class="fill-soft-a stroke-a"/>
<rect x="160.8" y="179.0" width="8.4" height="15.0" class="fill-soft-a stroke-a"/>
<line x1="165.0" y1="194.0" x2="165.0" y2="44.0" class="stroke-ink" stroke-width="1.4"/>
<line x1="111.7" y1="194.0" x2="111.7" y2="44.0" class="stroke-b dashed" stroke-width="2"/>
<text x="165.0" y="24.0" text-anchor="middle" class="fig-bold">No scaling</text>
<text x="169.0" y="50.0" text-anchor="start" class="fig-small">test time: 20</text>
<text x="107.7" y="74.0" text-anchor="end" class="fig-small fig-bold">mean 9.8</text>
<line x1="340.0" y1="194.0" x2="570.0" y2="194.0" class="stroke-muted"/>
<line x1="340.0" y1="194.0" x2="340.0" y2="44.0" class="stroke-muted"/>
<line x1="350.5" y1="194.0" x2="350.5" y2="198.0" class="stroke-muted"/>
<text x="350.5" y="211.0" text-anchor="middle" class="fig-small">0</text>
<line x1="402.7" y1="194.0" x2="402.7" y2="198.0" class="stroke-muted"/>
<text x="402.7" y="211.0" text-anchor="middle" class="fig-small">10</text>
<line x1="455.0" y1="194.0" x2="455.0" y2="198.0" class="stroke-muted"/>
<text x="455.0" y="211.0" text-anchor="middle" class="fig-small">20</text>
<line x1="507.3" y1="194.0" x2="507.3" y2="198.0" class="stroke-muted"/>
<text x="507.3" y="211.0" text-anchor="middle" class="fig-small">30</text>
<line x1="559.5" y1="194.0" x2="559.5" y2="198.0" class="stroke-muted"/>
<text x="559.5" y="211.0" text-anchor="middle" class="fig-small">40</text>
<line x1="336.0" y1="194.0" x2="340.0" y2="194.0" class="stroke-muted"/>
<text x="333.0" y="198.0" text-anchor="end" class="fig-small">0</text>
<line x1="336.0" y1="119.0" x2="340.0" y2="119.0" class="stroke-muted"/>
<text x="333.0" y="123.0" text-anchor="end" class="fig-small">25</text>
<line x1="336.0" y1="44.0" x2="340.0" y2="44.0" class="stroke-muted"/>
<text x="333.0" y="48.0" text-anchor="end" class="fig-small">50</text>
<text x="455.0" y="228.0" text-anchor="middle" class="fig-small">layer sum</text>
<rect x="342.1" y="164.0" width="16.7" height="30.0" class="fill-soft-a stroke-a"/>
<rect x="363.0" y="161.0" width="16.7" height="33.0" class="fill-soft-a stroke-a"/>
<rect x="383.9" y="152.0" width="16.7" height="42.0" class="fill-soft-a stroke-a"/>
<rect x="404.8" y="125.0" width="16.7" height="69.0" class="fill-soft-a stroke-a"/>
<rect x="425.7" y="116.0" width="16.7" height="78.0" class="fill-soft-a stroke-a"/>
<rect x="446.6" y="92.0" width="16.7" height="102.0" class="fill-soft-a stroke-a"/>
<rect x="467.5" y="125.0" width="16.7" height="69.0" class="fill-soft-a stroke-a"/>
<rect x="488.5" y="95.0" width="16.7" height="99.0" class="fill-soft-a stroke-a"/>
<rect x="509.4" y="167.0" width="16.7" height="27.0" class="fill-soft-a stroke-a"/>
<rect x="530.3" y="158.0" width="16.7" height="36.0" class="fill-soft-a stroke-a"/>
<rect x="551.2" y="179.0" width="16.7" height="15.0" class="fill-soft-a stroke-a"/>
<line x1="455.0" y1="194.0" x2="455.0" y2="44.0" class="stroke-ink" stroke-width="1.4"/>
<line x1="453.0" y1="194.0" x2="453.0" y2="44.0" class="stroke-b dashed" stroke-width="2"/>
<text x="455.0" y="24.0" text-anchor="middle" class="fig-bold">Divide by 1 − p</text>
<text x="459.0" y="50.0" text-anchor="start" class="fig-small">test time: 20</text>
<text x="449.0" y="74.0" text-anchor="end" class="fig-small fig-bold">mean 19.6</text>
</svg>
Figure 2. The sum of the layer's four outputs for x = [2, 4, 6, 8] over 200 random masks at p = 0.5. Without scaling, the sums average 9.8, half of the 20 the layer produces at test time. Dividing by 1 − p moves the average to 19.6, so training and test time agree on average.
```

The fix is to divide the survivors by the chance of surviving, $1 - p$. Here $[2, 0, 6, 8] / 0.5 = [4, 0, 12, 16]$. One draw gives a sum of 32, more than 20, but the average over masks is 20. Check one unit: with $x = 2$, the output is 4 half the time and 0 half the time, which averages 2. Each unit survives with probability $1-p$ and is then multiplied by $1/(1-p)$, so its average output is $(1-p) \cdot x/(1-p) = x$.

$$
\text{out} = \frac{x \cdot \text{mask}}{1 - p}
$$

Read it aloud: "keep a unit with probability $1 - p$, and if you keep it, scale it up by the reciprocal of that chance." The mask is a 0/1 array the same shape as $x$, and $p$ is the drop probability. In numpy:

```python
mask = (rng.random(x.shape) < 1.0 - p).astype(float)
out = x * mask / (1.0 - p)
```

A unit is kept when its random draw falls below $1 - p$. The mask depends only on the shape and the random generator, not on the values in `x`.

Why not leave training alone and multiply the outputs by $1 - p$ at test time? The original dropout paper did exactly that, and it works. The cost is that the model you ship is not the model you trained, so every place that runs it must apply the scaling. Moving the division into training, which is called **inverted dropout**, makes the test-time pass plain: with `training=False` you return `x` and a mask of ones.

## The backward pass

The forward pass is an elementwise multiply by the array $\text{mask}/(1-p)$. The slope of a product with respect to one factor is the other factor, so the backward pass multiplies the incoming gradient by the same array:

$$
dx = \frac{dout \cdot \text{mask}}{1 - p}
$$

Read it aloud: "a unit that was dropped did not affect the loss on this step, so it gets no gradient. A unit that was kept passes its gradient back, scaled by the same $1/(1-p)$ it was scaled by going forward." With $dout = [1, 1, 1, 1]$ and the mask $[1, 0, 1, 1]$, the result is $dx = [2, 0, 2, 2]$.

The mask is random, which might make the gradient seem undefined. It is not, because you hold the mask fixed between the forward and backward pass. For that one step the layer is an ordinary multiply by a fixed array. This is also why `dropout_forward` returns the mask: the backward pass needs the same one.

```python
dx = dout * mask / (1.0 - p)
```

> [!KEY]
> Dropout trains a crowd of thinned networks so that no unit can lean on a particular partner. Dividing the survivors by $1 - p$ keeps the layer's average output equal to its test-time output, so test time needs no change.

## Why certainty is expensive

The output side has its own problem. Suppose the scores for a row are $[g, 0, 0]$, so the right class leads each wrong class by a gap $g$. Its probability is $e^g / (e^g + 2)$. Suppose the label on some example is wrong, and the true class is the second one. The loss is $\ln(e^g + 2)$, which is close to $g$ once $g$ is large. At a gap of 8 that costs 8.0, and at a gap of 20 it costs 20.0.

With a one-hot target, the loss is zero only when the right class has probability 1, which needs an infinite gap between scores. So training keeps raising the scores for as long as it runs, and a model pushed toward ever larger gaps pays a large price on every example it gets wrong. You might ask why not cap the scores at some maximum. A cap fights the optimiser: the gradient still says "go higher", and the cap throws that push away. A better fix moves the target itself, so the gradient falls to zero by its own accord at a sensible gap.

## Softening the target

Take scores $z = [2, 0, 0]$, right class 1, $K = 3$ classes, and a smoothing amount $\epsilon = 0.1$. Softmax uses $e^2 = 7.389$, so the probabilities are $[7.389, 1, 1] / 9.389 = [0.7870, 0.1065, 0.1065]$.

The ordinary target is $[1, 0, 0]$. **Label smoothing** takes a share $\epsilon$ of the target and spreads it evenly over all $K$ classes. The right class keeps $1 - \epsilon$ plus its even share $\epsilon / K$, so it gets $0.9 + 0.0333 = 0.9333$. Each wrong class gets $\epsilon/K = 0.0333$. The smoothed target is $[0.9333, 0.0333, 0.0333]$, and Figure 3 shows it beside the other two distributions.

```figure
<svg viewBox="0 0 600 282" role="img" aria-label="Grouped bars for three classes. The one-hot target is 1, 0, 0. The smoothed target is 0.933, 0.033, 0.033. The softmax output for scores 2, 0, 0 is 0.787, 0.107, 0.107.">
<line x1="60.0" y1="206.0" x2="560.0" y2="206.0" class="stroke-muted"/>
<line x1="60.0" y1="206.0" x2="60.0" y2="36.0" class="stroke-muted"/>
<line x1="56.0" y1="206.0" x2="60.0" y2="206.0" class="stroke-muted"/>
<text x="53.0" y="210.0" text-anchor="end" class="fig-small">0</text>
<line x1="56.0" y1="130.1" x2="60.0" y2="130.1" class="stroke-muted"/>
<text x="53.0" y="134.1" text-anchor="end" class="fig-small">0.5</text>
<line x1="56.0" y1="54.2" x2="60.0" y2="54.2" class="stroke-muted"/>
<text x="53.0" y="58.2" text-anchor="end" class="fig-small">1</text>
<text x="143.3" y="224.0" text-anchor="middle" class="fig-small">class 1</text>
<rect x="83.3" y="54.2" width="37.0" height="151.8" class="fill-soft-a stroke-a"/>
<text x="101.8" y="49.2" text-anchor="middle" class="fig-small">1.000</text>
<rect x="123.3" y="64.3" width="37.0" height="141.7" class="fill-soft-c stroke-c"/>
<text x="141.8" y="59.3" text-anchor="middle" class="fig-small">0.933</text>
<rect x="163.3" y="86.5" width="37.0" height="119.5" class="fill-soft-b stroke-b"/>
<text x="181.8" y="81.5" text-anchor="middle" class="fig-small">0.787</text>
<text x="310.0" y="224.0" text-anchor="middle" class="fig-small">class 2</text>
<rect x="250.0" y="206.0" width="37.0" height="0.0" class="fill-soft-a stroke-a"/>
<text x="268.5" y="201.0" text-anchor="middle" class="fig-small">0.000</text>
<rect x="290.0" y="200.9" width="37.0" height="5.1" class="fill-soft-c stroke-c"/>
<text x="308.5" y="195.9" text-anchor="middle" class="fig-small">0.033</text>
<rect x="330.0" y="189.8" width="37.0" height="16.2" class="fill-soft-b stroke-b"/>
<text x="348.5" y="184.8" text-anchor="middle" class="fig-small">0.107</text>
<text x="476.7" y="224.0" text-anchor="middle" class="fig-small">class 3</text>
<rect x="416.7" y="206.0" width="37.0" height="0.0" class="fill-soft-a stroke-a"/>
<text x="435.2" y="201.0" text-anchor="middle" class="fig-small">0.000</text>
<rect x="456.7" y="200.9" width="37.0" height="5.1" class="fill-soft-c stroke-c"/>
<text x="475.2" y="195.9" text-anchor="middle" class="fig-small">0.033</text>
<rect x="496.7" y="189.8" width="37.0" height="16.2" class="fill-soft-b stroke-b"/>
<text x="515.2" y="184.8" text-anchor="middle" class="fig-small">0.107</text>
<rect x="330" y="238" width="0" height="0" class="fill-soft-a stroke-a"/>
<rect x="330" y="238" width="0" height="0" class="fill-soft-c stroke-c"/>
<rect x="330" y="238" width="0" height="0" class="fill-soft-b stroke-b"/>
<rect x="60" y="248" width="14" height="14" class="fill-soft-a stroke-a"/>
<text x="80.0" y="260.0" text-anchor="start" class="fig-small">one-hot target</text>
<rect x="215" y="248" width="14" height="14" class="fill-soft-c stroke-c"/>
<text x="235.0" y="260.0" text-anchor="start" class="fig-small">smoothed target</text>
<rect x="370" y="248" width="14" height="14" class="fill-soft-b stroke-b"/>
<text x="390.0" y="260.0" text-anchor="start" class="fig-small">softmax of [2, 0, 0]</text>
</svg>
Figure 3. Three distributions over three classes. The smoothed target [0.933, 0.033, 0.033] is the one-hot target pulled slightly toward uniform. The softmax output for scores [2, 0, 0] is [0.787, 0.107, 0.107], and training moves it toward the smoothed target, not the one-hot one.
```

$$
y^{\text{smooth}} = (1 - \epsilon)\,\text{onehot} + \frac{\epsilon}{K}
$$

Read it aloud: "scale the one-hot row down to $1 - \epsilon$, then add $\epsilon/K$ to every class." The row still sums to 1, because $(1 - \epsilon) + K \cdot \epsilon/K = 1$. In numpy, with `targets` an integer array of class indices:

```python
y_smooth = (1.0 - eps) * np.eye(K)[targets] + eps / K
```

## The loss, and why to subtract the max

The loss on one row is cross-entropy against the smoothed target, and the loss on a batch is the mean over rows:

$$
L = -\frac{1}{N} \sum_{\text{rows}} \sum_{c=1}^{K} y^{\text{smooth}}_c \log p_c
$$

Read it aloud: "for each row, weight each class's log probability by its target share, flip the sign, and then average over the rows." On the worked row, $-(0.9333 \ln 0.7870 + 0.0333 \ln 0.1065 + 0.0333 \ln 0.1065) = 0.3729$. With $\epsilon = 0$ it would be $-\ln 0.7870 = 0.2395$. The smoothed loss is higher on this row, because the model is less sure than the smoothed target asks.

You need $\log p$ for every class, and computing $p$ first and taking its log fails when $p$ underflows to 0. Compute the log directly instead. Shift the scores by their row maximum, then subtract the log of the sum of exponentials:

```python
shifted = logits - logits.max(axis=1, keepdims=True)
log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
```

The shift changes nothing in the result, because the same constant cancels in the softmax, and it keeps every exponent at most 0. Try scores $[800, 0, 0]$. Without the shift, $e^{800}$ overflows. With it the loss is finite, and it equals $2 \times 0.0333 \times 800 = 53.3$: the model is certain, and the smoothed target punishes certainty heavily. With $\epsilon = 0$ the same row costs 0.

> [!TIP]
> The exercise gives you `log_softmax`, already written. Use it for every loss in this chapter rather than taking the log of a probability.

## The gradient, and where the minimum sits

The gradient with respect to the scores keeps the shape you already know from softmax and cross-entropy:

$$
\frac{\partial L}{\partial z} = \frac{p - y^{\text{smooth}}}{N}
$$

Read it aloud: "predicted probabilities minus the smoothed target, divided by the number of rows." For softmax followed by cross-entropy, the gradient with respect to score $c$ is $p_c \sum_k y_k - y_c$. The smoothed target sums to 1, so the sum drops out and leaves $p_c - y_c$. The division by $N$ appears because the loss is a mean over rows. On the worked row, with $N = 1$, the gradient is $[-0.1463, 0.0732, 0.0732]$: it raises the right score and lowers the others, as it should.

```python
grad = (p - y_smooth) / N
```

The gradient is zero when $p = y^{\text{smooth}}$. Training stops pushing once the model gives the right class 0.9333 and each other class 0.0333, which is a gap of $\ln(0.9333 / 0.0333) = \ln 28 = 3.33$ between the right score and each wrong one. Figure 4 plots the loss against that gap. With the one-hot target it keeps falling. With the smoothed target it is lowest at 3.33, where it equals 0.291, and it rises again beyond that. The lowest value is not 0 because the target itself is not certain: 0.291 is the entropy of the smoothed target, the loss no model can beat. At a gap of 8, the loss is 0.534.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Loss against the score gap. The one-hot curve falls toward zero as the gap grows. The smoothed curve reaches its lowest value of 0.291 at a gap of 3.33 and then rises, reaching 0.534 at a gap of 8.">
<defs><marker id="c033-arrow-1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<line x1="60.0" y1="226.0" x2="560.0" y2="226.0" class="stroke-muted"/>
<line x1="60.0" y1="226.0" x2="60.0" y2="36.0" class="stroke-muted"/>
<line x1="60.0" y1="226.0" x2="60.0" y2="230.0" class="stroke-muted"/>
<text x="60.0" y="243.0" text-anchor="middle" class="fig-small">0</text>
<line x1="185.0" y1="226.0" x2="185.0" y2="230.0" class="stroke-muted"/>
<text x="185.0" y="243.0" text-anchor="middle" class="fig-small">2</text>
<line x1="310.0" y1="226.0" x2="310.0" y2="230.0" class="stroke-muted"/>
<text x="310.0" y="243.0" text-anchor="middle" class="fig-small">4</text>
<line x1="435.0" y1="226.0" x2="435.0" y2="230.0" class="stroke-muted"/>
<text x="435.0" y="243.0" text-anchor="middle" class="fig-small">6</text>
<line x1="560.0" y1="226.0" x2="560.0" y2="230.0" class="stroke-muted"/>
<text x="560.0" y="243.0" text-anchor="middle" class="fig-small">8</text>
<line x1="56.0" y1="226.0" x2="60.0" y2="226.0" class="stroke-muted"/>
<text x="53.0" y="230.0" text-anchor="end" class="fig-small">0</text>
<line x1="56.0" y1="150.0" x2="60.0" y2="150.0" class="stroke-muted"/>
<text x="53.0" y="154.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="56.0" y1="74.0" x2="60.0" y2="74.0" class="stroke-muted"/>
<text x="53.0" y="78.0" text-anchor="end" class="fig-small">1</text>
<text x="310.0" y="260.0" text-anchor="middle" class="fig-small">gap between the right score and each wrong score</text>
<text x="60.0" y="28.0" text-anchor="start" class="fig-small">loss</text>
<path d="M60.0,59.0 L61.7,61.7 L63.3,64.4 L65.0,67.0 L66.7,69.7 L68.4,72.3 L70.0,74.8 L71.7,77.4 L73.4,79.9 L75.1,82.4 L76.7,84.9 L78.4,87.3 L80.1,89.7 L81.7,92.1 L83.4,94.5 L85.1,96.9 L86.8,99.2 L88.4,101.5 L90.1,103.7 L91.8,106.0 L93.4,108.2 L95.1,110.3 L96.8,112.5 L98.5,114.6 L100.1,116.7 L101.8,118.8 L103.5,120.8 L105.2,122.8 L106.8,124.8 L108.5,126.8 L110.2,128.7 L111.8,130.6 L113.5,132.5 L115.2,134.4 L116.9,136.2 L118.5,138.0 L120.2,139.8 L121.9,141.5 L123.5,143.3 L125.2,144.9 L126.9,146.6 L128.6,148.3 L130.2,149.9 L131.9,151.5 L133.6,153.0 L135.3,154.6 L136.9,156.1 L138.6,157.6 L140.3,159.0 L141.9,160.5 L143.6,161.9 L145.3,163.3 L147.0,164.6 L148.6,166.0 L150.3,167.3 L152.0,168.6 L153.6,169.8 L155.3,171.1 L157.0,172.3 L158.7,173.5 L160.3,174.7 L162.0,175.8 L163.7,177.0 L165.4,178.1 L167.0,179.2 L168.7,180.2 L170.4,181.3 L172.0,182.3 L173.7,183.3 L175.4,184.3 L177.1,185.3 L178.7,186.2 L180.4,187.1 L182.1,188.0 L183.7,188.9 L185.4,189.8 L187.1,190.7 L188.8,191.5 L190.4,192.3 L192.1,193.1 L193.8,193.9 L195.5,194.7 L197.1,195.4 L198.8,196.1 L200.5,196.9 L202.1,197.6 L203.8,198.2 L205.5,198.9 L207.2,199.6 L208.8,200.2 L210.5,200.8 L212.2,201.5 L213.8,202.1 L215.5,202.6 L217.2,203.2 L218.9,203.8 L220.5,204.3 L222.2,204.9 L223.9,205.4 L225.6,205.9 L227.2,206.4 L228.9,206.9 L230.6,207.3 L232.2,207.8 L233.9,208.3 L235.6,208.7 L237.3,209.1 L238.9,209.6 L240.6,210.0 L242.3,210.4 L243.9,210.8 L245.6,211.2 L247.3,211.5 L249.0,211.9 L250.6,212.2 L252.3,212.6 L254.0,212.9 L255.7,213.3 L257.3,213.6 L259.0,213.9 L260.7,214.2 L262.3,214.5 L264.0,214.8 L265.7,215.1 L267.4,215.4 L269.0,215.6 L270.7,215.9 L272.4,216.2 L274.0,216.4 L275.7,216.7 L277.4,216.9 L279.1,217.1 L280.7,217.4 L282.4,217.6 L284.1,217.8 L285.8,218.0 L287.4,218.2 L289.1,218.4 L290.8,218.6 L292.4,218.8 L294.1,219.0 L295.8,219.2 L297.5,219.3 L299.1,219.5 L300.8,219.7 L302.5,219.8 L304.1,220.0 L305.8,220.2 L307.5,220.3 L309.2,220.5 L310.8,220.6 L312.5,220.7 L314.2,220.9 L315.9,221.0 L317.5,221.1 L319.2,221.3 L320.9,221.4 L322.5,221.5 L324.2,221.6 L325.9,221.7 L327.6,221.9 L329.2,222.0 L330.9,222.1 L332.6,222.2 L334.2,222.3 L335.9,222.4 L337.6,222.5 L339.3,222.6 L340.9,222.6 L342.6,222.7 L344.3,222.8 L346.0,222.9 L347.6,223.0 L349.3,223.1 L351.0,223.1 L352.6,223.2 L354.3,223.3 L356.0,223.4 L357.7,223.4 L359.3,223.5 L361.0,223.6 L362.7,223.6 L364.3,223.7 L366.0,223.7 L367.7,223.8 L369.4,223.9 L371.0,223.9 L372.7,224.0 L374.4,224.0 L376.1,224.1 L377.7,224.1 L379.4,224.2 L381.1,224.2 L382.7,224.3 L384.4,224.3 L386.1,224.4 L387.8,224.4 L389.4,224.4 L391.1,224.5 L392.8,224.5 L394.4,224.6 L396.1,224.6 L397.8,224.6 L399.5,224.7 L401.1,224.7 L402.8,224.7 L404.5,224.8 L406.2,224.8 L407.8,224.8 L409.5,224.9 L411.2,224.9 L412.8,224.9 L414.5,225.0 L416.2,225.0 L417.9,225.0 L419.5,225.0 L421.2,225.1 L422.9,225.1 L424.5,225.1 L426.2,225.1 L427.9,225.2 L429.6,225.2 L431.2,225.2 L432.9,225.2 L434.6,225.2 L436.3,225.3 L437.9,225.3 L439.6,225.3 L441.3,225.3 L442.9,225.3 L444.6,225.4 L446.3,225.4 L448.0,225.4 L449.6,225.4 L451.3,225.4 L453.0,225.4 L454.6,225.5 L456.3,225.5 L458.0,225.5 L459.7,225.5 L461.3,225.5 L463.0,225.5 L464.7,225.5 L466.4,225.5 L468.0,225.6 L469.7,225.6 L471.4,225.6 L473.0,225.6 L474.7,225.6 L476.4,225.6 L478.1,225.6 L479.7,225.6 L481.4,225.6 L483.1,225.7 L484.7,225.7 L486.4,225.7 L488.1,225.7 L489.8,225.7 L491.4,225.7 L493.1,225.7 L494.8,225.7 L496.5,225.7 L498.1,225.7 L499.8,225.7 L501.5,225.7 L503.1,225.7 L504.8,225.8 L506.5,225.8 L508.2,225.8 L509.8,225.8 L511.5,225.8 L513.2,225.8 L514.8,225.8 L516.5,225.8 L518.2,225.8 L519.9,225.8 L521.5,225.8 L523.2,225.8 L524.9,225.8 L526.6,225.8 L528.2,225.8 L529.9,225.8 L531.6,225.8 L533.2,225.8 L534.9,225.8 L536.6,225.9 L538.3,225.9 L539.9,225.9 L541.6,225.9 L543.3,225.9 L544.9,225.9 L546.6,225.9 L548.3,225.9 L550.0,225.9 L551.6,225.9 L553.3,225.9 L555.0,225.9 L556.7,225.9 L558.3,225.9 L560.0,225.9" fill="none" class="stroke-a" stroke-width="2.4"/>
<path d="M60.0,59.0 L61.7,61.4 L63.3,63.8 L65.0,66.2 L66.7,68.6 L68.4,70.9 L70.0,73.2 L71.7,75.5 L73.4,77.7 L75.1,80.0 L76.7,82.2 L78.4,84.3 L80.1,86.5 L81.7,88.6 L83.4,90.7 L85.1,92.8 L86.8,94.8 L88.4,96.8 L90.1,98.8 L91.8,100.8 L93.4,102.7 L95.1,104.6 L96.8,106.5 L98.5,108.4 L100.1,110.2 L101.8,112.0 L103.5,113.8 L105.2,115.5 L106.8,117.2 L108.5,118.9 L110.2,120.6 L111.8,122.2 L113.5,123.9 L115.2,125.4 L116.9,127.0 L118.5,128.5 L120.2,130.0 L121.9,131.5 L123.5,132.9 L125.2,134.4 L126.9,135.8 L128.6,137.1 L130.2,138.5 L131.9,139.8 L133.6,141.1 L135.3,142.4 L136.9,143.6 L138.6,144.8 L140.3,146.0 L141.9,147.2 L143.6,148.3 L145.3,149.4 L147.0,150.5 L148.6,151.6 L150.3,152.6 L152.0,153.7 L153.6,154.7 L155.3,155.6 L157.0,156.6 L158.7,157.5 L160.3,158.4 L162.0,159.3 L163.7,160.2 L165.4,161.0 L167.0,161.8 L168.7,162.6 L170.4,163.4 L172.0,164.1 L173.7,164.9 L175.4,165.6 L177.1,166.3 L178.7,167.0 L180.4,167.6 L182.1,168.3 L183.7,168.9 L185.4,169.5 L187.1,170.1 L188.8,170.6 L190.4,171.2 L192.1,171.7 L193.8,172.2 L195.5,172.7 L197.1,173.2 L198.8,173.6 L200.5,174.1 L202.1,174.5 L203.8,174.9 L205.5,175.3 L207.2,175.7 L208.8,176.1 L210.5,176.4 L212.2,176.8 L213.8,177.1 L215.5,177.4 L217.2,177.7 L218.9,178.0 L220.5,178.3 L222.2,178.6 L223.9,178.8 L225.6,179.0 L227.2,179.3 L228.9,179.5 L230.6,179.7 L232.2,179.9 L233.9,180.1 L235.6,180.2 L237.3,180.4 L238.9,180.6 L240.6,180.7 L242.3,180.8 L243.9,180.9 L245.6,181.1 L247.3,181.2 L249.0,181.3 L250.6,181.3 L252.3,181.4 L254.0,181.5 L255.7,181.5 L257.3,181.6 L259.0,181.6 L260.7,181.7 L262.3,181.7 L264.0,181.7 L265.7,181.7 L267.4,181.7 L269.0,181.7 L270.7,181.7 L272.4,181.7 L274.0,181.7 L275.7,181.7 L277.4,181.7 L279.1,181.6 L280.7,181.6 L282.4,181.5 L284.1,181.5 L285.8,181.4 L287.4,181.3 L289.1,181.3 L290.8,181.2 L292.4,181.1 L294.1,181.0 L295.8,180.9 L297.5,180.8 L299.1,180.7 L300.8,180.6 L302.5,180.5 L304.1,180.4 L305.8,180.3 L307.5,180.2 L309.2,180.1 L310.8,179.9 L312.5,179.8 L314.2,179.7 L315.9,179.5 L317.5,179.4 L319.2,179.2 L320.9,179.1 L322.5,178.9 L324.2,178.8 L325.9,178.6 L327.6,178.5 L329.2,178.3 L330.9,178.1 L332.6,178.0 L334.2,177.8 L335.9,177.6 L337.6,177.5 L339.3,177.3 L340.9,177.1 L342.6,176.9 L344.3,176.7 L346.0,176.5 L347.6,176.3 L349.3,176.2 L351.0,176.0 L352.6,175.8 L354.3,175.6 L356.0,175.4 L357.7,175.2 L359.3,175.0 L361.0,174.8 L362.7,174.5 L364.3,174.3 L366.0,174.1 L367.7,173.9 L369.4,173.7 L371.0,173.5 L372.7,173.3 L374.4,173.1 L376.1,172.8 L377.7,172.6 L379.4,172.4 L381.1,172.2 L382.7,171.9 L384.4,171.7 L386.1,171.5 L387.8,171.3 L389.4,171.0 L391.1,170.8 L392.8,170.6 L394.4,170.3 L396.1,170.1 L397.8,169.9 L399.5,169.6 L401.1,169.4 L402.8,169.2 L404.5,168.9 L406.2,168.7 L407.8,168.4 L409.5,168.2 L411.2,168.0 L412.8,167.7 L414.5,167.5 L416.2,167.2 L417.9,167.0 L419.5,166.7 L421.2,166.5 L422.9,166.3 L424.5,166.0 L426.2,165.8 L427.9,165.5 L429.6,165.3 L431.2,165.0 L432.9,164.8 L434.6,164.5 L436.3,164.3 L437.9,164.0 L439.6,163.8 L441.3,163.5 L442.9,163.2 L444.6,163.0 L446.3,162.7 L448.0,162.5 L449.6,162.2 L451.3,162.0 L453.0,161.7 L454.6,161.5 L456.3,161.2 L458.0,161.0 L459.7,160.7 L461.3,160.4 L463.0,160.2 L464.7,159.9 L466.4,159.7 L468.0,159.4 L469.7,159.1 L471.4,158.9 L473.0,158.6 L474.7,158.4 L476.4,158.1 L478.1,157.8 L479.7,157.6 L481.4,157.3 L483.1,157.1 L484.7,156.8 L486.4,156.5 L488.1,156.3 L489.8,156.0 L491.4,155.7 L493.1,155.5 L494.8,155.2 L496.5,155.0 L498.1,154.7 L499.8,154.4 L501.5,154.2 L503.1,153.9 L504.8,153.6 L506.5,153.4 L508.2,153.1 L509.8,152.8 L511.5,152.6 L513.2,152.3 L514.8,152.0 L516.5,151.8 L518.2,151.5 L519.9,151.2 L521.5,151.0 L523.2,150.7 L524.9,150.4 L526.6,150.2 L528.2,149.9 L529.9,149.6 L531.6,149.4 L533.2,149.1 L534.9,148.8 L536.6,148.6 L538.3,148.3 L539.9,148.0 L541.6,147.8 L543.3,147.5 L544.9,147.2 L546.6,147.0 L548.3,146.7 L550.0,146.4 L551.6,146.2 L553.3,145.9 L555.0,145.6 L556.7,145.4 L558.3,145.1 L560.0,144.8" fill="none" class="stroke-b" stroke-width="2.4"/>
<line x1="268.3" y1="226.0" x2="268.3" y2="181.7" class="stroke-muted dashed" stroke-width="1.2"/>
<circle cx="268.3" cy="181.7" r="5.5" class="fill-b"/>
<text x="276.3" y="203.2" text-anchor="start" class="fig-bold">best gap 3.33</text>
<line x1="85.0" y1="46.6" x2="178.8" y2="46.6" class="stroke-ink" stroke-width="1.8" marker-end="url(#c033-arrow-1)"/>
<text x="188.1" y="50.6" text-anchor="start" class="fig-small">descent widens the gap</text>
<line x1="535.0" y1="107.4" x2="441.2" y2="107.4" class="stroke-ink" stroke-width="1.8" marker-end="url(#c033-arrow-1)"/>
<text x="435.0" y="111.4" text-anchor="end" class="fig-small">descent narrows it</text>
<text x="560.0" y="217.9" text-anchor="end" class="fig-small">one-hot target</text>
<text x="560.0" y="136.8" text-anchor="end" class="fig-small">smoothed target</text>
</svg>
Figure 4. Loss against the gap between the right score and each wrong score, for scores [gap, 0, 0]. With a one-hot target the loss keeps falling as the gap grows. With the smoothed target it is lowest at a gap of ln 28 = 3.33, where the loss is 0.291. Past it, the gradient reverses, so the scores stop growing.
```

This answers the opening problem. The model is asked for 93 percent, and it is penalised for going past that. The scores stop growing, and a mislabelled example costs $\ln(28 + 2) = 3.40$ at the best gap, instead of 8.0 or more at the gaps the one-hot loss keeps driving toward.

> [!KEY]
> Replacing the one-hot target with a softened one moves the lowest point of the loss to a finite confidence, $1 - \epsilon + \epsilon/K$ for the right class. The gradient $p - y^{\text{smooth}}$ then vanishes at a finite gap of scores, so training stops asking for 100 percent.

## How the tools fit together

Weight decay and early stopping share a story: both pull weights toward zero. Dropout and label smoothing tell different stories. Dropout injects noise into the units, so none can rely on its partners. Label smoothing changes the target, so the loss is lowest at a finite confidence. They are not four versions of one idea, so you can combine them, and you choose each strength on held-out data.

> [!KEY]
> Dropout and label smoothing keep a network from trusting itself. Dropout stops a unit from leaning on a partner that might vanish, and label smoothing stops the output from reaching for 100 percent. Both cost almost nothing to add, and you tune $p$ and $\epsilon$ on held-out data.

> [!RECAP]
> Hold these points before you write code:
>
> - Dropout switches each unit off with probability $p$ on every training step. A unit is kept when its draw is below $1 - p$.
> - Divide the survivors by $1 - p$, so the average output equals the input. At test time, return `x` and a mask of ones. The backward pass multiplies `dout` by the same mask and by $1/(1-p)$.
> - The smoothed target is $(1 - \epsilon)$ times the one-hot row plus $\epsilon/K$. Each row sums to 1.
> - The loss is the mean over rows of $-\sum_c y^{\text{smooth}}_c \log p_c$, computed with a max-shifted log-softmax.
> - The gradient is $(p - y^{\text{smooth}})/N$, and the loss is lowest where $p$ equals the smoothed target, a finite gap of scores.

## Build it step by step

The starter gives you `log_softmax`, the max-shifted log-softmax from above, already written. The checks call your functions in this order:

1. **`dropout_mask(shape, p, rng)`.** Take a shape, the drop probability, and a `numpy.random.Generator`. Return a float array of 0s and 1s with that shape. Draw from `rng.random(shape)` and compare it with $1 - p$. The usual mistake is to keep units when the draw is below $p$, which drops them with probability $1-p$.
2. **`dropout_forward(x, p, rng, training)`.** Return `(out, mask)`. If `training` is false, return `x` and `np.ones_like(x)`. Otherwise build the mask with `dropout_mask`, then scale. The usual mistake is to divide by `p` instead of `1 - p`. At $p = 0.5$ the two are equal, so the checks also use $p = 0.25$ to catch it.
3. **`dropout_backward(dout, mask, p)`.** Return an array the same shape as `dout`. The usual mistake is to skip the $1/(1-p)$ on the way back, or to divide by `p`; the checks use $p = 0.25$ here too.
4. **`smooth_targets(targets, num_classes, eps)`.** Take integer class indices of shape `(N,)`. Return the `(N, K)` float matrix of smoothed targets. Index into `np.eye(num_classes)`. The usual mistake is to add `eps / K` before scaling, so the row no longer sums to 1.
5. **`label_smoothing_loss(logits, targets, eps)`.** Return the mean over rows as a float, using `smooth_targets` and `log_softmax`. The usual mistake is to sum over rows instead of averaging.
6. **`label_smoothing_grad(logits, targets, eps)`.** Return an `(N, K)` array. Get the probabilities from `np.exp(log_softmax(logits))`. The usual mistake is to leave out the division by `N`.

## Cost and scaling

Dropout costs one random draw and one multiply per activation, and a stored mask the size of the activations so the backward pass can reuse it. The larger cost is indirect: because each step trains only part of the network, training usually needs more steps to reach the same training loss. Label smoothing costs a few operations per score and no extra memory beyond the target matrix.

## Where this breaks

- Dropout changes the spread of a layer's outputs between training and test. A batch norm layer after it collects its running averages during training, with the noise present, and then sees clean inputs at test time. The mismatch can hurt, so many convolutional networks skip dropout, or place it after the last normalisation layer.
- A drop rate near 1 leaves almost nothing switched on, so the signal is gone and training stalls. Rates between 0.1 and 0.5 are typical. With too little noise, dropout does nothing.
- Label smoothing makes the model's probabilities stop meaning what they say. A calibrated model is right about 80 percent of the time when it says 80 percent, but after smoothing the top probability is pulled toward $1 - \epsilon + \epsilon/K$ whatever the evidence. Skip smoothing when you need calibrated probabilities, or fix them afterwards.
- A large $\epsilon$ blurs the target so much that the classes look alike, and the model underfits. Small values such as 0.1 are common.

## The exercise

You implement `dropout_mask`, `dropout_forward`, `dropout_backward`, `smooth_targets`, `label_smoothing_loss`, and `label_smoothing_grad`.

The checks verify that the mask holds only 0s and 1s, keeps about a $1-p$ share of units, and repeats when you reseed the generator. They check that `dropout_forward` returns `x` unchanged at test time, builds its mask with `dropout_mask`, scales the survivors, and averages to `x` over many draws. They run your backward pass on the worked example and compare it with a finite difference with the mask held fixed. Then they verify the smoothed target on the worked example and the loss on the worked row, that it is a mean over rows, and that it stays finite for scores of 800. They compare the gradient with a finite difference of your loss, and check that it vanishes where the probabilities equal the smoothed target. The last check scans the loss over score gaps and confirms that its lowest point is near $\ln 28$.
