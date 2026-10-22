---
title: Why a deep network's signal shrinks or explodes
slug: 069-signal-size-through-layers
part: "Part 4 — Neural networks"
summary: A linear layer multiplies the signal's mean square by n_in times the weight variance, and fifty layers compound any miss from 1.
minutes: 35
gpu: false
objectives:
  - Explain why one linear layer multiplies the mean square of its signal by n_in times the weight variance.
  - Predict the signal's size after fifty layers from one layer's gain, and say why a small miss from 1 is fatal.
  - Explain why no fixed weight scale, such as 0.01, works for every layer width.
  - Build a simulator that measures signal size through a random stack and compare it with your prediction.
prereqs:
  - 058-forward-pass-and-gelu
  - 064-backprop-mlp-and-gradient-check
  - 068-lr-schedules
exercise: 069-signal-size-through-layers
---

# Why a deep network's signal shrinks or explodes

You build a network of 50 linear layers, each 128 units wide, and fill every weight with a random number from a bell curve with standard deviation 0.1. That looks small and safe. You feed in one input whose entries are about the size of 1 and read the last layer. The numbers there are around 900.

Now change the standard deviation to 0.07, which looks just as safe. The last layer's numbers are around 0.00002. Whatever input you feed in, the output is close to zero. A network like that cannot learn, because the output carries no trace of the input, and the gradient on the way back shrinks in the same way, so the early layers barely move.

Where did the size go, and what decides whether it survives? This chapter answers by counting. One layer multiplies the signal's size by a number you can compute from the layer's shape and the weight scale, and fifty layers multiply by that number fifty times.

## What you already know

Four ideas carry this chapter:

- A linear layer multiplies a vector of inputs by a weight matrix (see "Run a network forward, with a sharp or smooth bend").
- Training fills the weights with random numbers first, and gradient descent changes them afterward.
- The gradient travels back through the same layers, multiplied by each layer's weights (see "Putting it together: gradients for a whole network").
- A bell-curve (normal) draw with mean 0 and standard deviation $s$ is as often negative as positive, and its square averages to $s^2$.

## Watch a signal survive, die, or blow up

Think of a photocopier. Copy a page, copy the copy, and repeat 50 times. If each copy is 10% too dark, the 50th is $1.1^{50} \approx 117$ times too dark and the page is a black square. If each is 10% too light, the 50th is $0.9^{50} \approx 0.005$ as dark, almost blank. Only a copier set to exactly 100% leaves the page readable. Each layer of a network is a copier, and the weights set its dial.

To measure "how dark" a layer's signal is, use its **mean square**: square every number in the layer and average them. For the two numbers $3$ and $-4$ the mean square is $(9 + 16)/2 = 12.5$. The plain average would let positive and negative numbers cancel, and the mean square cannot be fooled that way. This chapter writes $q$ for the mean square of a layer.

Figure 1 shows $q$ for three dial settings, using the simulator you build in the exercise. The vertical axis is logarithmic, so a constant factor per layer draws a straight line.

```figure
<svg viewBox="0 0 600 350" role="img" aria-label="Mean square against layer number on a log axis for gains 0.7, 1.0, and 1.4">
<line x1="70.0" y1="300.0" x2="470.0" y2="300.0" class="stroke-muted"/>
<line x1="70.0" y1="300.0" x2="70.0" y2="30.0" class="stroke-grid"/>
<line x1="70.0" y1="284.1" x2="470.0" y2="284.1" class="stroke-grid"/>
<text x="62.0" y="288.1" class="fig-small" text-anchor="end">10⁻⁸</text>
<line x1="70.0" y1="220.6" x2="470.0" y2="220.6" class="stroke-grid"/>
<text x="62.0" y="224.6" class="fig-small" text-anchor="end">10⁻⁴</text>
<line x1="70.0" y1="157.1" x2="470.0" y2="157.1" class="stroke-grid"/>
<text x="62.0" y="161.1" class="fig-small" text-anchor="end">10⁰</text>
<line x1="70.0" y1="93.5" x2="470.0" y2="93.5" class="stroke-grid"/>
<text x="62.0" y="97.5" class="fig-small" text-anchor="end">10⁴</text>
<line x1="70.0" y1="30.0" x2="470.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">10⁸</text>
<line x1="70.0" y1="300.0" x2="70.0" y2="305.0" class="stroke-muted"/>
<text x="70.0" y="320.0" class="fig-small" text-anchor="middle">0</text>
<line x1="150.0" y1="300.0" x2="150.0" y2="305.0" class="stroke-muted"/>
<text x="150.0" y="320.0" class="fig-small" text-anchor="middle">10</text>
<line x1="230.0" y1="300.0" x2="230.0" y2="305.0" class="stroke-muted"/>
<text x="230.0" y="320.0" class="fig-small" text-anchor="middle">20</text>
<line x1="310.0" y1="300.0" x2="310.0" y2="305.0" class="stroke-muted"/>
<text x="310.0" y="320.0" class="fig-small" text-anchor="middle">30</text>
<line x1="390.0" y1="300.0" x2="390.0" y2="305.0" class="stroke-muted"/>
<text x="390.0" y="320.0" class="fig-small" text-anchor="middle">40</text>
<line x1="470.0" y1="300.0" x2="470.0" y2="305.0" class="stroke-muted"/>
<text x="470.0" y="320.0" class="fig-small" text-anchor="middle">50</text>
<text x="270.0" y="342.0" class="fig-small" text-anchor="middle">layer number</text>
<text x="62.0" y="20.0" class="fig-small" text-anchor="start">mean square (log scale)</text>
<line x1="70.0" y1="157.1" x2="470.0" y2="157.1" class="stroke-muted dashed"/>
<polyline points="70.0,157.1 78.0,159.5 86.0,162.0 94.0,164.4 102.0,166.9 110.0,169.4 118.0,171.8 126.0,174.3 134.0,176.7 142.0,179.2 150.0,181.7 158.0,184.1 166.0,186.6 174.0,189.0 182.0,191.5 190.0,194.0 198.0,196.4 206.0,198.9 214.0,201.3 222.0,203.8 230.0,206.3 238.0,208.7 246.0,211.2 254.0,213.6 262.0,216.1 270.0,218.6 278.0,221.0 286.0,223.5 294.0,225.9 302.0,228.4 310.0,230.9 318.0,233.3 326.0,235.8 334.0,238.2 342.0,240.7 350.0,243.2 358.0,245.6 366.0,248.1 374.0,250.5 382.0,253.0 390.0,255.5 398.0,257.9 406.0,260.4 414.0,262.8 422.0,265.3 430.0,267.8 438.0,270.2 446.0,272.7 454.0,275.1 462.0,277.6 470.0,280.1" fill="none" class="stroke-a"/>
<circle cx="70.0" cy="157.1" r="3.5" class="fill-a"/>
<circle cx="110.0" cy="169.1" r="3.5" class="fill-a"/>
<circle cx="150.0" cy="181.3" r="3.5" class="fill-a"/>
<circle cx="190.0" cy="193.8" r="3.5" class="fill-a"/>
<circle cx="230.0" cy="205.9" r="3.5" class="fill-a"/>
<circle cx="270.0" cy="215.4" r="3.5" class="fill-a"/>
<circle cx="310.0" cy="225.9" r="3.5" class="fill-a"/>
<circle cx="350.0" cy="238.0" r="3.5" class="fill-a"/>
<circle cx="390.0" cy="248.6" r="3.5" class="fill-a"/>
<circle cx="430.0" cy="261.3" r="3.5" class="fill-a"/>
<circle cx="470.0" cy="271.8" r="3.5" class="fill-a"/>
<polyline points="70.0,157.1 78.0,157.1 86.0,157.1 94.0,157.1 102.0,157.1 110.0,157.1 118.0,157.1 126.0,157.1 134.0,157.1 142.0,157.1 150.0,157.1 158.0,157.1 166.0,157.1 174.0,157.1 182.0,157.1 190.0,157.1 198.0,157.1 206.0,157.1 214.0,157.1 222.0,157.1 230.0,157.1 238.0,157.1 246.0,157.1 254.0,157.1 262.0,157.1 270.0,157.1 278.0,157.1 286.0,157.1 294.0,157.1 302.0,157.1 310.0,157.1 318.0,157.1 326.0,157.1 334.0,157.1 342.0,157.1 350.0,157.1 358.0,157.1 366.0,157.1 374.0,157.1 382.0,157.1 390.0,157.1 398.0,157.1 406.0,157.1 414.0,157.1 422.0,157.1 430.0,157.1 438.0,157.1 446.0,157.1 454.0,157.1 462.0,157.1 470.0,157.1" fill="none" class="stroke-c"/>
<circle cx="70.0" cy="157.1" r="3.5" class="fill-c"/>
<circle cx="110.0" cy="156.8" r="3.5" class="fill-c"/>
<circle cx="150.0" cy="156.7" r="3.5" class="fill-c"/>
<circle cx="190.0" cy="156.9" r="3.5" class="fill-c"/>
<circle cx="230.0" cy="156.7" r="3.5" class="fill-c"/>
<circle cx="270.0" cy="153.9" r="3.5" class="fill-c"/>
<circle cx="310.0" cy="152.1" r="3.5" class="fill-c"/>
<circle cx="350.0" cy="151.8" r="3.5" class="fill-c"/>
<circle cx="390.0" cy="150.2" r="3.5" class="fill-c"/>
<circle cx="430.0" cy="150.6" r="3.5" class="fill-c"/>
<circle cx="470.0" cy="148.8" r="3.5" class="fill-c"/>
<polyline points="70.0,157.1 78.0,154.7 86.0,152.4 94.0,150.1 102.0,147.8 110.0,145.5 118.0,143.1 126.0,140.8 134.0,138.5 142.0,136.2 150.0,133.9 158.0,131.5 166.0,129.2 174.0,126.9 182.0,124.6 190.0,122.2 198.0,119.9 206.0,117.6 214.0,115.3 222.0,113.0 230.0,110.6 238.0,108.3 246.0,106.0 254.0,103.7 262.0,101.4 270.0,99.0 278.0,96.7 286.0,94.4 294.0,92.1 302.0,89.8 310.0,87.4 318.0,85.1 326.0,82.8 334.0,80.5 342.0,78.1 350.0,75.8 358.0,73.5 366.0,71.2 374.0,68.9 382.0,66.5 390.0,64.2 398.0,61.9 406.0,59.6 414.0,57.3 422.0,54.9 430.0,52.6 438.0,50.3 446.0,48.0 454.0,45.7 462.0,43.3 470.0,41.0" fill="none" class="stroke-b"/>
<circle cx="70.0" cy="157.1" r="3.5" class="fill-b"/>
<circle cx="110.0" cy="145.2" r="3.5" class="fill-b"/>
<circle cx="150.0" cy="133.5" r="3.5" class="fill-b"/>
<circle cx="190.0" cy="122.1" r="3.5" class="fill-b"/>
<circle cx="230.0" cy="110.3" r="3.5" class="fill-b"/>
<circle cx="270.0" cy="95.8" r="3.5" class="fill-b"/>
<circle cx="310.0" cy="82.5" r="3.5" class="fill-b"/>
<circle cx="350.0" cy="70.6" r="3.5" class="fill-b"/>
<circle cx="390.0" cy="57.4" r="3.5" class="fill-b"/>
<circle cx="430.0" cy="46.1" r="3.5" class="fill-b"/>
<circle cx="470.0" cy="32.7" r="3.5" class="fill-b"/>
<text x="480.0" y="284.1" class="fig-small" text-anchor="start">0.7⁵⁰ ≈ 1.8×10⁻⁸</text>
<text x="480.0" y="45.0" class="fig-small" text-anchor="start">1.4⁵⁰ ≈ 2.0×10⁷</text>
<text x="480.0" y="161.1" class="fig-small" text-anchor="start">1.0: line at 1</text>
<text x="480.0" y="144.8" class="fig-small" text-anchor="start">dot at 3.3</text>
<text x="80.0" y="38.0" class="fig-small" text-anchor="start">line: predicted g^layers.  dots: simulated, width 128</text>
</svg>
Figure 1. Mean square of the signal in a 50-layer linear network of width 128, for three layer gains. Lines are the prediction (gain to the power of the layer number). Dots are simulations with real random weights. A constant gain is a straight line on a log axis, so a gain of 0.7 falls to about one part in 56 million and a gain of 1.4 grows to about 20 million. The three runs reuse the same random numbers, so at layer 50 every dot sits about 3.3 times above its line. That gap is the wobble of one random draw, and it is small next to a factor of about 10 million either way. The dots follow the lines' slopes.
```

The line that falls and the line that rises are the same mistake in opposite directions. Both are off from 1 by a modest factor at each layer, and fifty layers turn that modest factor into a factor of $10^{-8}$ or $10^{7}$. The factor itself comes from one layer, so that is where to look next.

## Where one layer's factor comes from

Each output of a linear layer is a weighted sum of the layer's inputs:

$$
\text{output} = w_1 h_1 + w_2 h_2 + \dots + w_n h_n
$$

Here $h_1, \dots, h_n$ are the $n$ inputs and $w_1, \dots, w_n$ are the weights, drawn at random before training. Start with $n = 2$ and the inputs $h = (1, 2)$. Their mean square is $(1 + 4)/2 = 2.5$. Two draws of the weights give outputs of $0.126 \cdot 1 + (-0.132) \cdot 2 = -0.138$ and $0.640 \cdot 1 + 0.105 \cdot 2 = 0.850$, with squares $0.019$ and $0.723$. Average the square over 100,000 draws with standard deviation 1 and you get 4.98, which is close to 5.

Why 5? Expand the square of one output:

$$
(w_1 h_1 + w_2 h_2)^2 = w_1^2 h_1^2 + w_2^2 h_2^2 + 2\, w_1 w_2\, h_1 h_2
$$

The last term mixes two different weights. Each weight is as likely to be negative as positive and is drawn separately from the other, so their product is as often negative as positive, and it averages to zero. The first two terms do not cancel, because squares are never negative. Each $w^2$ averages to $s^2$, so the average of the whole square is $s^2 (h_1^2 + h_2^2) = 1 \cdot (1 + 4) = 5$.

The same argument holds for $n$ inputs. The cross terms average to zero, and what is left is $s^2$ times the sum of the squared inputs. A sum of squares is $n$ times the mean square, so, averaged over the weights, the output's mean square is

$$
q_{\text{out}} = n_{\text{in}}\, s^2 \, q_{\text{in}}
$$

Read it aloud as "the output's size is the input's size, times the number of inputs, times the weight variance." The number $n_{\text{in}}$ counts the inputs to one output unit, and it is also called the layer's **fan-in**. The product $n_{\text{in}} s^2$ is the layer's **gain**, the dial setting on the photocopier. Figure 2 shows the gain at work for $s = 1$ and inputs of mean square 1. The spread of the output grows with $n$, and the measured mean squares land on 2, 4, and 8.

```figure
<svg viewBox="0 0 600 352" role="img" aria-label="Three panels: one output as a sum of 2, 4, or 8 weighted inputs, and a histogram of the output that widens as n grows">
<circle cx="35.0" cy="75.0" r="5" class="fill-a"/>
<line x1="42.0" y1="75.0" x2="158.0" y2="100.0" class="stroke-muted"/>
<circle cx="35.0" cy="125.0" r="5" class="fill-a"/>
<line x1="42.0" y1="125.0" x2="158.0" y2="100.0" class="stroke-muted"/>
<circle cx="165.0" cy="100.0" r="8" class="fill-b"/>
<text x="35.0" y="38.0" class="fig-small" text-anchor="start">n = 2 inputs</text>
<line x1="20.0" y1="300.0" x2="190.0" y2="300.0" class="stroke-muted"/>
<rect x="20.5" y="300.0" width="6.1" height="0.0" class="fill-b"/>
<rect x="27.6" y="300.0" width="6.1" height="0.0" class="fill-b"/>
<rect x="34.7" y="300.0" width="6.1" height="0.0" class="fill-b"/>
<rect x="41.8" y="299.9" width="6.1" height="0.1" class="fill-b"/>
<rect x="48.8" y="299.9" width="6.1" height="0.1" class="fill-b"/>
<rect x="55.9" y="299.7" width="6.1" height="0.3" class="fill-b"/>
<rect x="63.0" y="299.1" width="6.1" height="0.9" class="fill-b"/>
<rect x="70.1" y="298.1" width="6.1" height="1.9" class="fill-b"/>
<rect x="77.2" y="293.5" width="6.1" height="6.5" class="fill-b"/>
<rect x="84.2" y="283.4" width="6.1" height="16.6" class="fill-b"/>
<rect x="91.3" y="253.2" width="6.1" height="46.8" class="fill-b"/>
<rect x="98.4" y="170.5" width="6.1" height="129.5" class="fill-b"/>
<rect x="105.5" y="170.0" width="6.1" height="130.0" class="fill-b"/>
<rect x="112.6" y="253.2" width="6.1" height="46.8" class="fill-b"/>
<rect x="119.7" y="283.3" width="6.1" height="16.7" class="fill-b"/>
<rect x="126.8" y="293.6" width="6.1" height="6.4" class="fill-b"/>
<rect x="133.8" y="298.0" width="6.1" height="2.0" class="fill-b"/>
<rect x="140.9" y="299.4" width="6.1" height="0.6" class="fill-b"/>
<rect x="148.0" y="299.6" width="6.1" height="0.4" class="fill-b"/>
<rect x="155.1" y="299.9" width="6.1" height="0.1" class="fill-b"/>
<rect x="162.2" y="299.9" width="6.1" height="0.1" class="fill-b"/>
<rect x="169.2" y="300.0" width="6.1" height="0.0" class="fill-b"/>
<rect x="176.3" y="300.0" width="6.1" height="0.0" class="fill-b"/>
<rect x="183.4" y="300.0" width="6.1" height="0.0" class="fill-b"/>
<text x="105.0" y="316.0" class="fig-small" text-anchor="middle">output</text>
<text x="105.0" y="164.0" class="fig-small fig-bold" text-anchor="middle">mean square 1.94</text>
<line x1="105.0" y1="300.0" x2="105.0" y2="304.0" class="stroke-muted"/>
<circle cx="230.0" cy="62.5" r="5" class="fill-a"/>
<line x1="237.0" y1="62.5" x2="353.0" y2="100.0" class="stroke-muted"/>
<circle cx="230.0" cy="87.5" r="5" class="fill-a"/>
<line x1="237.0" y1="87.5" x2="353.0" y2="100.0" class="stroke-muted"/>
<circle cx="230.0" cy="112.5" r="5" class="fill-a"/>
<line x1="237.0" y1="112.5" x2="353.0" y2="100.0" class="stroke-muted"/>
<circle cx="230.0" cy="137.5" r="5" class="fill-a"/>
<line x1="237.0" y1="137.5" x2="353.0" y2="100.0" class="stroke-muted"/>
<circle cx="360.0" cy="100.0" r="8" class="fill-b"/>
<text x="230.0" y="38.0" class="fig-small" text-anchor="start">n = 4 inputs</text>
<line x1="215.0" y1="300.0" x2="385.0" y2="300.0" class="stroke-muted"/>
<rect x="215.5" y="300.0" width="6.1" height="0.0" class="fill-b"/>
<rect x="222.6" y="300.0" width="6.1" height="0.0" class="fill-b"/>
<rect x="229.7" y="300.0" width="6.1" height="0.0" class="fill-b"/>
<rect x="236.8" y="299.9" width="6.1" height="0.1" class="fill-b"/>
<rect x="243.8" y="299.5" width="6.1" height="0.5" class="fill-b"/>
<rect x="250.9" y="298.8" width="6.1" height="1.2" class="fill-b"/>
<rect x="258.0" y="297.2" width="6.1" height="2.8" class="fill-b"/>
<rect x="265.1" y="293.0" width="6.1" height="7.0" class="fill-b"/>
<rect x="272.2" y="285.6" width="6.1" height="14.4" class="fill-b"/>
<rect x="279.2" y="270.2" width="6.1" height="29.8" class="fill-b"/>
<rect x="286.3" y="243.8" width="6.1" height="56.2" class="fill-b"/>
<rect x="293.4" y="209.9" width="6.1" height="90.1" class="fill-b"/>
<rect x="300.5" y="209.1" width="6.1" height="90.9" class="fill-b"/>
<rect x="307.6" y="241.1" width="6.1" height="58.9" class="fill-b"/>
<rect x="314.7" y="271.0" width="6.1" height="29.0" class="fill-b"/>
<rect x="321.8" y="286.0" width="6.1" height="14.0" class="fill-b"/>
<rect x="328.8" y="293.9" width="6.1" height="6.1" class="fill-b"/>
<rect x="335.9" y="297.5" width="6.1" height="2.5" class="fill-b"/>
<rect x="343.0" y="298.7" width="6.1" height="1.3" class="fill-b"/>
<rect x="350.1" y="299.5" width="6.1" height="0.5" class="fill-b"/>
<rect x="357.2" y="299.7" width="6.1" height="0.3" class="fill-b"/>
<rect x="364.2" y="299.9" width="6.1" height="0.1" class="fill-b"/>
<rect x="371.3" y="300.0" width="6.1" height="0.0" class="fill-b"/>
<rect x="378.4" y="300.0" width="6.1" height="0.0" class="fill-b"/>
<text x="300.0" y="316.0" class="fig-small" text-anchor="middle">output</text>
<text x="300.0" y="164.0" class="fig-small fig-bold" text-anchor="middle">mean square 4.02</text>
<line x1="300.0" y1="300.0" x2="300.0" y2="304.0" class="stroke-muted"/>
<circle cx="425.0" cy="56.2" r="5" class="fill-a"/>
<line x1="432.0" y1="56.2" x2="548.0" y2="100.0" class="stroke-muted"/>
<circle cx="425.0" cy="68.8" r="5" class="fill-a"/>
<line x1="432.0" y1="68.8" x2="548.0" y2="100.0" class="stroke-muted"/>
<circle cx="425.0" cy="81.2" r="5" class="fill-a"/>
<line x1="432.0" y1="81.2" x2="548.0" y2="100.0" class="stroke-muted"/>
<circle cx="425.0" cy="93.8" r="5" class="fill-a"/>
<line x1="432.0" y1="93.8" x2="548.0" y2="100.0" class="stroke-muted"/>
<circle cx="425.0" cy="106.2" r="5" class="fill-a"/>
<line x1="432.0" y1="106.2" x2="548.0" y2="100.0" class="stroke-muted"/>
<circle cx="425.0" cy="118.8" r="5" class="fill-a"/>
<line x1="432.0" y1="118.8" x2="548.0" y2="100.0" class="stroke-muted"/>
<circle cx="425.0" cy="131.2" r="5" class="fill-a"/>
<line x1="432.0" y1="131.2" x2="548.0" y2="100.0" class="stroke-muted"/>
<circle cx="425.0" cy="143.8" r="5" class="fill-a"/>
<line x1="432.0" y1="143.8" x2="548.0" y2="100.0" class="stroke-muted"/>
<circle cx="555.0" cy="100.0" r="8" class="fill-b"/>
<text x="425.0" y="38.0" class="fig-small" text-anchor="start">n = 8 inputs</text>
<line x1="410.0" y1="300.0" x2="580.0" y2="300.0" class="stroke-muted"/>
<rect x="410.5" y="299.8" width="6.1" height="0.2" class="fill-b"/>
<rect x="417.6" y="299.8" width="6.1" height="0.2" class="fill-b"/>
<rect x="424.7" y="299.5" width="6.1" height="0.5" class="fill-b"/>
<rect x="431.8" y="299.1" width="6.1" height="0.9" class="fill-b"/>
<rect x="438.8" y="298.2" width="6.1" height="1.8" class="fill-b"/>
<rect x="445.9" y="295.8" width="6.1" height="4.2" class="fill-b"/>
<rect x="453.0" y="292.5" width="6.1" height="7.5" class="fill-b"/>
<rect x="460.1" y="286.1" width="6.1" height="13.9" class="fill-b"/>
<rect x="467.2" y="276.2" width="6.1" height="23.8" class="fill-b"/>
<rect x="474.2" y="264.0" width="6.1" height="36.0" class="fill-b"/>
<rect x="481.3" y="247.8" width="6.1" height="52.2" class="fill-b"/>
<rect x="488.4" y="238.3" width="6.1" height="61.7" class="fill-b"/>
<rect x="495.5" y="237.6" width="6.1" height="62.4" class="fill-b"/>
<rect x="502.6" y="250.0" width="6.1" height="50.0" class="fill-b"/>
<rect x="509.7" y="262.1" width="6.1" height="37.9" class="fill-b"/>
<rect x="516.8" y="277.7" width="6.1" height="22.3" class="fill-b"/>
<rect x="523.8" y="286.0" width="6.1" height="14.0" class="fill-b"/>
<rect x="530.9" y="292.0" width="6.1" height="8.0" class="fill-b"/>
<rect x="538.0" y="295.8" width="6.1" height="4.2" class="fill-b"/>
<rect x="545.1" y="298.0" width="6.1" height="2.0" class="fill-b"/>
<rect x="552.2" y="299.0" width="6.1" height="1.0" class="fill-b"/>
<rect x="559.2" y="299.4" width="6.1" height="0.6" class="fill-b"/>
<rect x="566.3" y="299.8" width="6.1" height="0.2" class="fill-b"/>
<rect x="573.4" y="299.6" width="6.1" height="0.4" class="fill-b"/>
<text x="495.0" y="316.0" class="fig-small" text-anchor="middle">output</text>
<text x="495.0" y="164.0" class="fig-small fig-bold" text-anchor="middle">mean square 7.95</text>
<line x1="495.0" y1="300.0" x2="495.0" y2="304.0" class="stroke-muted"/>
<text x="300.0" y="346.0" class="fig-small" text-anchor="middle">each panel: 20,000 draws, weights s = 1, inputs of mean square 1, same horizontal range −12 to 12</text>
</svg>
Figure 2. One output is a sum of n products of a weight and an input. Each panel shows 20,000 random draws of that output. The spread grows with n, and the measured mean square (1.94, 4.02, 7.95) lands close to n times the weight variance times the input's mean square, which is n here.
```

Two conditions make the formula true, and they are worth knowing, because they are what breaks it. The weights need mean 0, and each weight must be drawn separately from the others and from the input. If every weight were 1, the output for $h = (1, 2)$ would be $3$ and its square $9$, not 5, because the cross term would add up instead of cancel. Random initialisation meets both conditions. In code the formula is two lines:

```python
gain = n_in * scale ** 2        # the factor one layer applies to q
q_out = gain * q_in             # on average, over the random weights
```

One drawn matrix is a single sample of that average, so a single network wobbles around it, as the dots in Figure 1 do.

## Why not just pick a small number

The obvious fix is to choose one safe weight scale, such as 0.01, and use it everywhere. Figure 3 shows why that fails. The gain is $n_{\text{in}} s^2$, so it depends on the width $n_{\text{in}}$ as well as on $s$.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Gain against layer width on log axes for weight scales 1, 0.1, 0.01, and one over the square root of the width">
<line x1="70.0" y1="280.0" x2="470.0" y2="280.0" class="stroke-muted"/>
<line x1="70.0" y1="280.0" x2="70.0" y2="30.0" class="stroke-grid"/>
<line x1="70.0" y1="280.0" x2="470.0" y2="280.0" class="stroke-grid"/>
<text x="62.0" y="284.0" class="fig-small" text-anchor="end">10⁻⁴</text>
<line x1="70.0" y1="217.5" x2="470.0" y2="217.5" class="stroke-grid"/>
<text x="62.0" y="221.5" class="fig-small" text-anchor="end">10⁻²</text>
<line x1="70.0" y1="155.0" x2="470.0" y2="155.0" class="stroke-grid"/>
<text x="62.0" y="159.0" class="fig-small" text-anchor="end">10⁰</text>
<line x1="70.0" y1="92.5" x2="470.0" y2="92.5" class="stroke-grid"/>
<text x="62.0" y="96.5" class="fig-small" text-anchor="end">10²</text>
<line x1="70.0" y1="30.0" x2="470.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">10⁴</text>
<line x1="70.0" y1="280.0" x2="70.0" y2="285.0" class="stroke-muted"/>
<text x="70.0" y="300.0" class="fig-small" text-anchor="middle">1</text>
<line x1="170.0" y1="280.0" x2="170.0" y2="285.0" class="stroke-muted"/>
<text x="170.0" y="300.0" class="fig-small" text-anchor="middle">10</text>
<line x1="270.0" y1="280.0" x2="270.0" y2="285.0" class="stroke-muted"/>
<text x="270.0" y="300.0" class="fig-small" text-anchor="middle">100</text>
<line x1="370.0" y1="280.0" x2="370.0" y2="285.0" class="stroke-muted"/>
<text x="370.0" y="300.0" class="fig-small" text-anchor="middle">1,000</text>
<line x1="470.0" y1="280.0" x2="470.0" y2="285.0" class="stroke-muted"/>
<text x="470.0" y="300.0" class="fig-small" text-anchor="middle">10,000</text>
<text x="270.0" y="322.0" class="fig-small" text-anchor="middle">layer width n_in</text>
<text x="62.0" y="20.0" class="fig-small" text-anchor="start">gain n_in·s²</text>
<line x1="70.0" y1="155.0" x2="470.0" y2="155.0" class="stroke-c dashed"/>
<polyline points="70.0,155.0 470.0,30.0" fill="none" class="stroke-b"/>
<text x="480.0" y="34.0" class="fig-small" text-anchor="start">s = 1</text>
<circle cx="280.7" cy="89.1" r="4.5" class="fill-b"/>
<polyline points="70.0,217.5 470.0,92.5" fill="none" class="stroke-a"/>
<text x="480.0" y="96.5" class="fig-small" text-anchor="start">s = 0.1</text>
<circle cx="280.7" cy="151.6" r="4.5" class="fill-a"/>
<polyline points="70.0,280.0 470.0,155.0" fill="none" class="stroke-d"/>
<text x="480.0" y="159.0" class="fig-small" text-anchor="start">s = 0.01</text>
<circle cx="280.7" cy="214.1" r="4.5" class="fill-d"/>
<text x="462.0" y="149.0" class="fig-small" text-anchor="end">gain 1 (the goal)</text>
<text x="274.7" y="79.1" class="fig-small" text-anchor="end">128</text>
<text x="288.7" y="169.6" class="fig-small" text-anchor="start">1.28 at width 128</text>
<text x="274.7" y="206.1" class="fig-small" text-anchor="end">0.0128</text>
<text x="480.0" y="34.0" class="fig-small" text-anchor="middle"></text>
</svg>
Figure 3. The gain n_in·s² for three fixed weight scales, as the layer gets wider. Each line crosses the gain-1 level (green dashed) at exactly one width: s = 1 at width 1, s = 0.1 at width 100, s = 0.01 at width 10,000. The dots mark width 128. A scale that is right for one width is wrong for every other.
```

With $s = 0.01$ the gain is $0.0001\, n_{\text{in}}$. At width 128 that is $0.0128$, so each layer removes 99% of the size, and two layers leave about one part in 6,000. The scale 0.01 is exactly right for a layer of width 10,000 and wrong everywhere else. The scale $s = 1$ is exactly right for width 1 and gives a gain of 128 at width 128. Any fixed number is right for one width.

The scale has to follow the width. For gain 1 you need $s^2 = 1/n_{\text{in}}$, which is $s = 1/\sqrt{128} \approx 0.0884$ at width 128. Now the opening makes sense. The standard deviation 0.1 is 13% above 0.0884 and gives a gain of $128 \cdot 0.01 = 1.28$. The standard deviation 0.07 is 21% below it and gives a gain of $128 \cdot 0.0049 = 0.627$. A miss of 13% to 21% in $s$ looks tiny, and it is fatal, for the reason in the next section.

## From one layer to fifty

Each layer multiplies $q$ by the gain $g$, so after $L$ layers

$$
q_L = g^L \, q_0
$$

Read it aloud as "start with the input's size, then apply the gain once per layer." Here $q_0$ is the input's mean square and $q_L$ is the mean square after $L$ layers. The table lists $g^{L}$ for four gains:

| Gain per layer | After 10 layers | After 50 layers |
|---|---|---|
| 0.7 | 0.028 | $1.8 \times 10^{-8}$ |
| 0.9 | 0.35 | 0.0052 |
| 1.1 | 2.6 | 117 |
| 1.4 | 29 | $2.0 \times 10^{7}$ |

A gain of 0.7 takes only 39 layers to reach one part in a million. You can get that number from a logarithm, because you want the smallest $L$ with $g^L < 10^{-6}$, so $L > \ln(10^{-6}) / \ln(0.7) = 38.7$:

```python
depth = math.ceil(math.log(floor) / math.log(gain))   # for gain < 1
```

The opening numbers match these predictions. With $s = 0.1$ the simulation gave a mean square of $7.6 \times 10^5$, against a predicted $2.3 \times 10^5$, so typical entries grew from about 1 to about 870. With $s = 0.07$ it gave $2.5 \times 10^{-10}$, against $7.4 \times 10^{-11}$. Both sit about 3.3 times above the prediction, which is the wobble of one random draw, and it is small next to a factor of $10^{5}$ to $10^{10}$.

The backward pass has the same problem. The gradient at a layer's inputs is the transposed weight matrix times the gradient at its outputs. Each entry of it is a sum with one product for every output unit of the layer, so the same argument applies, with the number of outputs $n_{\text{out}}$ in place of $n_{\text{in}}$. For a square layer the two counts agree, so the gradient shrinks or grows with the same gain. When the two counts differ, the chapter "Xavier and He initialisation" shows what to do.

> [!KEY]
> One linear layer multiplies the mean square of its signal by $n_{\text{in}} s^2$, and fifty layers multiply by it fifty times. That is where the signal went: a gain of 0.7 per layer leaves one part in 56 million, and a gain of 1.4 gives twenty million. What decides survival is whether $n_{\text{in}} s^2$ is close to 1, so the weight scale must be chosen from the layer's width.

> [!RECAP]
> - The mean square $q$ of a layer measures the size of its signal, because squares do not cancel.
> - For weights with mean 0 drawn separately from each other, one layer gives $q_{\text{out}} = n_{\text{in}} s^2 q_{\text{in}}$, because the cross terms average to zero.
> - The gain $n_{\text{in}} s^2$ depends on the width, so no single weight scale suits every layer.
> - After $L$ layers the size is $g^L$ times the start, so a gain a little off 1 fails at depth 50.
> - The gradient obeys the same rule going backward, with $n_{\text{out}}$ counting the terms.

## Build it step by step

The checks call the functions in this order:

1. **`layer_gain(n_in, scale)`** takes the fan-in and the weight standard deviation and returns a float. It is one line:

   ```python
   return float(n_in * scale ** 2)
   ```

   The usual mistake is to forget that `scale` is a standard deviation. The gain uses its square.

2. **`depth_gain(gain, depth)`** returns the factor after `depth` identical layers, which is `gain ** depth`. The usual mistake is `gain * depth`, which grows in a straight line and misses the compounding.

3. **`forward_variance(depth, widths, scale, activation, num_trials, seed)`** simulates the stack and returns `depth + 1` mean squares. Start with `rng = np.random.default_rng(seed)` and draw the input `h` of shape `(num_trials, widths[0])`. Record `np.mean(h ** 2)`. Then, for each layer, draw `W` of shape `(widths[l], widths[l + 1])` with `rng.normal(scale=scale, ...)`, replace `h` with `h @ W`, and record the mean square again. When `activation` is `"relu"`, apply `np.maximum(h, 0.0)` after the multiply. The chapter "Xavier and He initialisation" explains what that does. The usual mistake is to apply ReLU to the input before the first layer.

## Where this breaks

The prediction is an average over random weights, so it has limits:

- A single network wobbles around the prediction, and the wobble compounds over many layers. A gain of exactly 1 ended at 3.3 after 50 layers in Figure 1, not at 1.
- Weights that are not centred on 0, or that are correlated, add their cross terms back in, and the formula under-counts.
- A nonlinearity changes the size as well. ReLU, the activation you meet in the next chapter, removes part of the signal at each layer, so a gain of 1 is not the right target there.
- After training starts, the weights move away from their initial values, so the gain at step 10,000 is not the gain you set. The chapter "Normalising activations: batch norm versus layer norm" covers a layer that restores the scale on every pass.

## The exercise

You write `layer_gain`, `depth_gain`, and `forward_variance`. The checks verify the gain formula with exact numbers, and the compounding with $0.7^{50}$ and $1.4^{50}$. They check that `forward_variance` returns `depth + 1` values that start at the input's mean square, and that one linear layer multiplies it by $n_{\text{in}} s^2$. The last two checks run ten layers of width 256 with a gain of 0.8 and a gain of 1.3, and compare the measured size with your own prediction within a factor of 2.
