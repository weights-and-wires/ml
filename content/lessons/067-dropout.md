---
title: "Dropout: train a crowd of thinned networks"
slug: 067-dropout
part: "Part 4 — Neural networks"
summary: Switch hidden units off at random during training so none can rely on a partner, and scale the survivors so test time needs no change.
minutes: 40
gpu: false
objectives:
  - Explain how units that always appear together memorise the training set, and how random removal stops it.
  - Compute inverted dropout by hand, and say why the survivors are divided by 1 - p.
  - Show with an exact average that the expected output of the layer equals its input.
  - Write the backward pass, and say why it reuses the forward pass's mask.
prereqs:
  - 055-backprop-array-operations
  - 065-weight-decay
  - 066-early-stopping-ridge
exercise: 067-dropout
---

# Dropout: train a crowd of thinned networks

A hidden layer has four units. Suppose that on your training set, unit 2 tends to overshoot, and unit 3 learns to correct it by pushing the other way. Together the pair gets the training examples right. Neither unit is useful alone: unit 3 has learned a correction that only makes sense when unit 2 is present and overshooting in its usual way. On a new example, unit 2 does something different, and unit 3's correction now makes things worse.

This is **co-adaptation**: units become tuned to each other instead of to the data. The more units a layer has, the more such partnerships it can form, and each one can memorise a few training examples. Nothing in the loss complains, because on the training set everyone is always present. The question is: if a unit can count on a particular neighbour always being there, what happens when that neighbour is missing?

You can ask the question during training, before it matters. On every training step, switch off each hidden unit at random with probability $p$. No unit can then count on any particular partner, because that partner will vanish on many steps. This is **dropout**.

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

- Overfitting means a low training error and a high error on new data ("Honest scores: training error, test error, and holding data back").
- A layer's activations are the numbers its units output, one per unit.
- Multiplying an array by another elementwise has a simple backward rule: the slope of $a \cdot b$ with respect to $a$ is $b$ ("The backward rule for each array operation").
- The mean of a random quantity is its value averaged over many draws.
- Weight decay and early stopping shrink weights. Dropout does something else: it adds noise.

## A crowd of thinned networks

Look at Figure 1. Each panel is the same network with a different set of hidden units switched off. A layer of 4 units has $2^4 = 16$ ways to keep or drop its units, and a layer of 6 has 64. Each training step picks one at random and trains that **thinned network**. The thinned networks share their weights, so a unit that is kept in two different masks is trained by both.

A second way to see it: training one network with dropout is training a whole crowd of thinned networks at once. At test time you use the full layer, which behaves approximately like the average of the crowd. It is an approximation, because averaging 64 networks and running one network with scaled-down outputs are not identical for a network with nonlinear layers. But each unit has been forced to be useful alongside many different sets of partners, so it cannot lean on any one of them, and it generalises better.

You might ask why not use fewer units. A smaller layer cannot form as many partnerships, but it also has less room to represent what the task needs. Dropout keeps the full width at test time, and teaches each unit to carry its own weight.

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

The fix is to divide the survivors by the chance of surviving, $1 - p$. Here $[2, 0, 6, 8] / 0.5 = [4, 0, 12, 16]$. One draw now gives a sum of 32, more than 20, but the average over masks is 20. Check one unit: with $x = 2$, the output is 4 half the time and 0 half the time, which averages 2. Each unit survives with probability $1-p$ and then is multiplied by $1/(1-p)$, so its average output is $(1-p) \cdot x/(1-p) = x$.

$$
\text{out} = \frac{x \cdot \text{mask}}{1 - p}
$$

Read it aloud: "keep a unit with probability $1 - p$, and if you keep it, scale it up by the reciprocal of that chance." The mask is a 0/1 array the same shape as $x$. Here $p$ is the drop probability. In numpy:

```python
mask = (rng.random(x.shape) < 1.0 - p).astype(float)
out = x * mask / (1.0 - p)
```

A unit is kept when its random draw falls below $1 - p$. The mask depends only on the shape and the random generator, not on the values in `x`.

Why not leave training alone and multiply the outputs by $1 - p$ at test time? The original dropout paper did exactly that, and it works: the test-time layer then matches the training average. The cost is that the model you ship is not the model you trained, so every place that runs it must apply the scaling. Moving the division into training, which is called **inverted dropout**, makes the test-time pass plain: with `training=False` you return `x` and a mask of ones.

> [!KEY]
> Dropout trains a crowd of thinned networks so that no unit can lean on a particular partner. Dividing the survivors by $1 - p$ keeps the layer's average output equal to its test-time output, so test time needs no change.

## The backward pass

The forward pass is an elementwise multiply by the array $\text{mask}/(1-p)$. The slope of a product with respect to one factor is the other factor, so the backward pass multiplies the incoming gradient by the same array:

$$
dx = \frac{dout \cdot \text{mask}}{1 - p}
$$

Read it aloud: "a unit that was dropped did not affect the loss on this step, so it gets no gradient. A unit that was kept passes its gradient back, scaled by the same $1/(1-p)$ it was scaled by going forward." Figure 3 draws it with $dout = [1, 1, 1, 1]$, giving $dx = [2, 0, 2, 2]$.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="A forward row x equals 2, 4, 6, 8 becomes out equals 4, 0, 12, 16 by multiplying by the mask 1, 0, 1, 1 and dividing by 0.5. A backward row dout equals 1, 1, 1, 1 becomes dx equals 2, 0, 2, 2 by the same mask and scale.">
<defs><marker id="c075-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="20.0" y="22.0" text-anchor="start" class="fig-bold">Forward</text>
<text x="20.0" y="198.0" text-anchor="start" class="fig-bold">Backward</text>
<rect x="180" y="24" width="60" height="32" rx="6" class="fill-soft-a stroke-a"/>
<text x="210.0" y="45.0" text-anchor="middle" class="fig-bold">2</text>
<rect x="270" y="24" width="60" height="32" rx="6" class="fill-paper stroke-muted dashed"/>
<text x="300.0" y="45.0" text-anchor="middle" class="fig-bold">4</text>
<rect x="360" y="24" width="60" height="32" rx="6" class="fill-soft-a stroke-a"/>
<text x="390.0" y="45.0" text-anchor="middle" class="fig-bold">6</text>
<rect x="450" y="24" width="60" height="32" rx="6" class="fill-soft-a stroke-a"/>
<text x="480.0" y="45.0" text-anchor="middle" class="fig-bold">8</text>
<rect x="180" y="114" width="60" height="32" rx="6" class="fill-soft-b stroke-b"/>
<text x="210.0" y="135.0" text-anchor="middle" class="fig-bold">4</text>
<rect x="270" y="114" width="60" height="32" rx="6" class="fill-paper stroke-muted dashed"/>
<text x="300.0" y="135.0" text-anchor="middle" class="fig-bold">0</text>
<rect x="360" y="114" width="60" height="32" rx="6" class="fill-soft-b stroke-b"/>
<text x="390.0" y="135.0" text-anchor="middle" class="fig-bold">12</text>
<rect x="450" y="114" width="60" height="32" rx="6" class="fill-soft-b stroke-b"/>
<text x="480.0" y="135.0" text-anchor="middle" class="fig-bold">16</text>
<rect x="180" y="199" width="60" height="32" rx="6" class="fill-soft-a stroke-a"/>
<text x="210.0" y="220.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="270" y="199" width="60" height="32" rx="6" class="fill-paper stroke-muted dashed"/>
<text x="300.0" y="220.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="360" y="199" width="60" height="32" rx="6" class="fill-soft-a stroke-a"/>
<text x="390.0" y="220.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="450" y="199" width="60" height="32" rx="6" class="fill-soft-a stroke-a"/>
<text x="480.0" y="220.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="180" y="289" width="60" height="32" rx="6" class="fill-soft-b stroke-b"/>
<text x="210.0" y="310.0" text-anchor="middle" class="fig-bold">2</text>
<rect x="270" y="289" width="60" height="32" rx="6" class="fill-paper stroke-muted dashed"/>
<text x="300.0" y="310.0" text-anchor="middle" class="fig-bold">0</text>
<rect x="360" y="289" width="60" height="32" rx="6" class="fill-soft-b stroke-b"/>
<text x="390.0" y="310.0" text-anchor="middle" class="fig-bold">2</text>
<rect x="450" y="289" width="60" height="32" rx="6" class="fill-soft-b stroke-b"/>
<text x="480.0" y="310.0" text-anchor="middle" class="fig-bold">2</text>
<text x="20.0" y="45.0" text-anchor="start" class="fig-small">x</text>
<text x="20.0" y="135.0" text-anchor="start" class="fig-small">out</text>
<text x="20.0" y="220.0" text-anchor="start" class="fig-small">dout</text>
<text x="20.0" y="310.0" text-anchor="start" class="fig-small">dx</text>
<line x1="210" y1="56" x2="210" y2="114" class="stroke-muted" stroke-width="1.6" marker-end="url(#c075-arrow)"/>
<line x1="210" y1="231" x2="210" y2="289" class="stroke-muted" stroke-width="1.6" marker-end="url(#c075-arrow)"/>
<line x1="300" y1="56" x2="300" y2="114" class="stroke-grid dashed" stroke-width="1.6"/>
<line x1="300" y1="231" x2="300" y2="289" class="stroke-grid dashed" stroke-width="1.6"/>
<line x1="390" y1="56" x2="390" y2="114" class="stroke-muted" stroke-width="1.6" marker-end="url(#c075-arrow)"/>
<line x1="390" y1="231" x2="390" y2="289" class="stroke-muted" stroke-width="1.6" marker-end="url(#c075-arrow)"/>
<line x1="480" y1="56" x2="480" y2="114" class="stroke-muted" stroke-width="1.6" marker-end="url(#c075-arrow)"/>
<line x1="480" y1="231" x2="480" y2="289" class="stroke-muted" stroke-width="1.6" marker-end="url(#c075-arrow)"/>
<text x="120.0" y="88.0" text-anchor="middle" class="fig-small">× mask, ÷ 0.5</text>
<text x="120.0" y="262.0" text-anchor="middle" class="fig-small">× same mask, ÷ 0.5</text>
<text x="308.0" y="88.0" text-anchor="start" class="fig-small">dropped: 0</text>
<text x="308.0" y="262.0" text-anchor="start" class="fig-small">dropped: 0</text>
<text x="540.0" y="88.0" text-anchor="start" class="fig-small"></text>
</svg>
Figure 3. The mask [1, 0, 1, 1] from the forward pass is reused on the way back. The dropped unit passes nothing forward and receives no gradient. Each kept unit is scaled by 1 / (1 − p) = 2 in both directions, so dout = [1, 1, 1, 1] becomes dx = [2, 0, 2, 2].
```

The mask is random, which might make the gradient seem undefined. It is not, because you hold the mask fixed between the forward and backward pass. For that one step the layer is an ordinary multiply by a fixed array. This is also why `dropout_forward` returns the mask: the backward pass needs the same one.

```python
dx = dout * mask / (1.0 - p)
```

> [!RECAP]
> Hold these points before you write code:
>
> - Dropout switches each unit off with probability $p$ on every training step, so units cannot rely on particular partners.
> - Divide the survivors by $1 - p$, so the average output equals the input. At test time, return `x` and a mask of ones.
> - The backward pass multiplies `dout` by the same mask and by $1/(1-p)$.
> - $p$ is the drop probability. A unit is kept when its draw is below $1 - p$.

## Build it step by step

The checks call the functions in this order:

1. **`dropout_mask(shape, p, rng)`.** Take a shape, the drop probability, and a `numpy.random.Generator`. Return a float array of 0s and 1s with that shape. Draw from `rng.random(shape)` and compare it with $1 - p$. The usual mistake is to keep units when the draw is below $p$, which drops them with probability $1-p$.
2. **`dropout_forward(x, p, rng, training)`.** Return `(out, mask)`. If `training` is false, return `x` and `np.ones_like(x)`. Otherwise build the mask with `dropout_mask`, then scale. The usual mistake is to divide by `p` instead of `1 - p`. At $p = 0.5$ the two are equal, so the checks also use $p = 0.25$ to catch it.
3. **`dropout_backward(dout, mask, p)`.** Return an array the same shape as `dout`. The usual mistake is to skip the $1/(1-p)$ on the way back, or to divide by `p`; the checks use $p = 0.25$ here too.

## Cost and scaling

Dropout costs one random draw and one multiply per activation, and a stored mask the size of the activations so the backward pass can reuse it. The larger cost is indirect: because each step trains only part of the network, training usually needs more steps to reach the same training loss.

## Where this breaks

- Dropout changes the spread of a layer's outputs between training and test. A batch norm layer after it collects its running averages during training, with the noise present, and then sees clean inputs at test time. The mismatch can hurt, so many convolutional networks skip dropout, or place it after the last normalisation layer.
- A drop rate near 1 leaves almost nothing switched on, so the signal is gone and training stalls. Rates between 0.1 and 0.5 are typical.
- With too little noise to matter, dropout does nothing. With too much, the network underfits. As with weight decay, choose $p$ on held-out data.

## The exercise

You implement `dropout_mask`, `dropout_forward`, and `dropout_backward`.

The checks verify that the mask holds only 0s and 1s and keeps about a $1-p$ share of units, and that it repeats when you reseed the generator. They check that `dropout_forward` returns `x` unchanged at test time, builds its mask with `dropout_mask`, and scales the survivors. They check that the average output over many draws matches `x`. The last checks run your backward pass on the worked example, and compare it with a finite difference with the mask held fixed.
