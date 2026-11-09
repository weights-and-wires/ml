---
title: "Signal size through layers, and Xavier and He initialisation"
slug: 032-signal-size-and-initialisation
part: "Part 4 — Neural networks"
summary: Count how a layer scales its signal, see why fifty layers compound any miss, and set the weight scale with Xavier or He so the gain is 1.
minutes: 70
gpu: false
objectives:
  - Explain why one linear layer multiplies the mean square of its signal by n_in times the weight variance.
  - Predict the signal's size after fifty layers from one layer's gain, and say why no fixed weight scale works for every width.
  - Explain why the forward pass counts a layer's inputs and the backward pass counts its outputs, and derive the Xavier scale.
  - Show why a ReLU removes half of the mean square of symmetric numbers, and derive the He scale.
  - Confirm on a fifty-layer ReLU network that He keeps the signal and Xavier loses half per layer.
prereqs:
  - 028-multilayer-perceptrons
  - 030-softmax-and-whole-network-gradients
  - 031-sgd-adam-and-schedules
exercise: 032-signal-size-and-initialisation
---

# Signal size through layers, and Xavier and He initialisation

You build a network of 50 linear layers, each 128 units wide, and fill every weight with a random number from a bell curve with standard deviation 0.1. That looks small and safe. You feed in one input whose entries are about the size of 1 and read the last layer. The numbers there are around 900. Change the standard deviation to 0.07, which looks equally safe, and the last layer's numbers are around 0.00002. A network like that cannot learn, because the output carries no trace of the input, and the gradient shrinks the same way on its way back.

Where did the size go, and what decides whether it survives? This chapter answers by counting. One layer multiplies the signal's size by a number you can compute from the layer's shape and the weight scale, and fifty layers multiply by it fifty times. Counting also exposes two follow-up questions. How do you choose the scale when a layer has a different number of inputs and outputs? And why does a network of ReLUs need a different answer from a network of tanh units?

## What you already know

Three ideas carry this chapter:

- A linear layer multiplies a vector of inputs by a weight matrix, and ReLU replaces every negative number with 0 (see [multilayer perceptrons](/lessons/028-multilayer-perceptrons)).
- The gradient travels back through the same layers, multiplied by the transposed weights (see [whole-network gradients](/lessons/030-softmax-and-whole-network-gradients)).
- Training fills the weights with random numbers first. A bell-curve (normal) draw with mean 0 and standard deviation $s$ is as often negative as positive, and its square averages to $s^2$.

## Watch a signal survive, die, or blow up

Think of a photocopier. Copy a page, copy the copy, and repeat 50 times. If each copy is 10% too dark, the 50th is $1.1^{50} \approx 117$ times too dark and the page is a black square. If each is 10% too light, the 50th is $0.9^{50} \approx 0.005$ as dark, almost blank. Only a copier set to exactly 100% leaves the page readable. Each layer of a network is a copier, and the weights set its dial.

To measure how dark a layer's signal is, use its **mean square**: square every number in the layer and average them. For $3$ and $-4$ it is $(9 + 16)/2 = 12.5$. A plain average would let positive and negative numbers cancel, and the mean square cannot be fooled that way. This chapter writes $q$ for the mean square of a layer.

Figure 1 shows $q$ for three dial settings, using the simulator you build in the exercise. The vertical axis is logarithmic, so a constant factor per layer draws a straight line. The line that falls and the line that rises are the same mistake in opposite directions: both are off from 1 by a modest factor per layer, and fifty layers turn that into a factor of $10^{-8}$ or $10^{7}$.

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

## Where one layer's factor comes from

Each output of a linear layer is a weighted sum of the layer's inputs: $\text{output} = w_1 h_1 + \dots + w_n h_n$, where $h_1, \dots, h_n$ are the $n$ inputs and $w_1, \dots, w_n$ are weights drawn at random before training. Start with $n = 2$ and inputs $h = (1, 2)$, whose mean square is $(1 + 4)/2 = 2.5$. Two draws of the weights give outputs $0.126 \cdot 1 + (-0.132) \cdot 2 = -0.138$ and $0.640 \cdot 1 + 0.105 \cdot 2 = 0.850$, with squares $0.019$ and $0.723$. Averaged over 100,000 draws with standard deviation 1, the square comes to 4.98, close to 5.

Why 5? Expand the square of one output:

$$
(w_1 h_1 + w_2 h_2)^2 = w_1^2 h_1^2 + w_2^2 h_2^2 + 2\, w_1 w_2\, h_1 h_2
$$

The last term mixes two different weights. Each is as likely to be negative as positive and is drawn separately from the other, so their product averages to zero. The first two terms do not cancel, because squares are never negative, and each $w^2$ averages to $s^2$. The average of the whole square is $s^2 (h_1^2 + h_2^2) = 1 \cdot (1 + 4) = 5$.

The same argument holds for $n$ inputs. The cross terms vanish, and what is left is $s^2$ times the sum of the squared inputs, which is $n$ times their mean square. So, averaged over the weights,

$$
q_{\text{out}} = n_{\text{in}}\, s^2 \, q_{\text{in}}
$$

Read it aloud: the output's size is the input's size, times the number of inputs, times the weight variance. The number $n_{\text{in}}$ counts the inputs to one output unit and is the layer's **fan-in**. The product $n_{\text{in}} s^2$ is the layer's **gain**, the dial setting on the photocopier. In code:

```python
gain = n_in * scale ** 2        # the factor one layer applies to q
q_out = gain * q_in             # on average, over the random weights
```

Two conditions make this true, and they are what breaks it. The weights need mean 0, and each must be drawn separately from the others and from the input. If every weight were 1, the output for $h = (1, 2)$ would be $3$ and its square $9$, not 5, because the cross term would add up instead of cancel. One drawn matrix is a single sample of the average, so one network wobbles around the prediction, as the dots in Figure 1 do.

## From one layer to fifty

Each layer multiplies $q$ by the gain $g$, so after $L$ layers $q_L = g^L q_0$. Read it aloud: start with the input's size and apply the gain once per layer. The table lists $g^{L}$ for four gains:

| Gain per layer | After 10 layers | After 50 layers |
|---|---|---|
| 0.7 | 0.028 | $1.8 \times 10^{-8}$ |
| 0.9 | 0.35 | 0.0052 |
| 1.1 | 2.6 | 117 |
| 1.4 | 29 | $2.0 \times 10^{7}$ |

The opening numbers match. With $s = 0.1$ the gain is $128 \cdot 0.01 = 1.28$, and the simulation gave a mean square of $7.6 \times 10^5$ against a predicted $2.3 \times 10^5$, so typical entries grew from about 1 to about 870. With $s = 0.07$ the gain is $128 \cdot 0.0049 = 0.627$, and it gave $2.5 \times 10^{-10}$ against $7.4 \times 10^{-11}$. Both sit about 3.3 times above the prediction, which is the wobble of one random draw, and it is small next to a factor of $10^{5}$ to $10^{10}$.

The obvious fix is to choose one safe weight scale, such as 0.01, and use it everywhere. Figure 2 shows why that fails. The gain $n_{\text{in}} s^2$ depends on the width $n_{\text{in}}$ as well as on $s$. With $s = 0.01$ and width 128 the gain is $0.0128$, so each layer removes 99% of the size. The scale 0.01 is exactly right for width 10,000 and wrong everywhere else, and any fixed number is right for one width. For gain 1 you need $s^2 = 1/n_{\text{in}}$, which is $s = 1/\sqrt{128} \approx 0.0884$ at width 128. The standard deviation 0.1 misses by 13% and 0.07 by 21%, and both are fatal at depth 50.

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
Figure 2. The gain n_in·s² for three fixed weight scales, as the layer gets wider. Each line crosses the gain-1 level (green dashed) at exactly one width: s = 1 at width 1, s = 0.1 at width 100, s = 0.01 at width 10,000. The dots mark width 128. A scale that is right for one width is wrong for every other.
```

> [!KEY]
> One linear layer multiplies the mean square of its signal by $n_{\text{in}} s^2$, and $L$ layers multiply by it $L$ times. A gain of 0.7 leaves one part in 56 million after fifty layers, and a gain of 1.4 gives twenty million. Survival needs the gain close to 1, so the weight scale must follow the layer's width.

## Forward counts inputs, backward counts outputs

Going forward, one output collects a product from each of the $n_{\text{in}}$ inputs. Going backward, one input's gradient collects a product from each of the $n_{\text{out}}$ outputs it feeds, because the gradient is multiplied by the transposed weights. By the argument above, the forward pass multiplies the mean square by $n_{\text{in}} s^2$, and the backward pass multiplies the gradient's mean square by $n_{\text{out}} s^2$. The number of outputs is the **fan-out**.

When the two counts differ, no single scale suits both. Take a layer that narrows the network from 256 units to 64. The forward pass wants $s^2 = 1/256$ and the backward pass wants $s^2 = 1/64$. Pick the forward value and the gradient shrinks by a factor of 4 at that layer. Pick the backward value and the signal grows by 4. The simple compromise sets the average of the two gains to 1:

$$
\frac{n_{\text{in}} s^2 + n_{\text{out}} s^2}{2} = 1
\quad\Rightarrow\quad
s_{\text{Xavier}} = \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}
$$

Read it aloud: set the weight variance to two over the sum of the fan-in and the fan-out. This is **Xavier initialisation**, also called Glorot. For the narrow layer, $s^2 = 2/320 = 1/160$, so $s \approx 0.0791$, the forward gain is $256/160 = 1.6$, and the backward gain is $64/160 = 0.4$. Neither is 1, and their average is. When $n_{\text{in}} = n_{\text{out}} = 128$, both gains are exactly 1 and $s = 0.0884$. In code:

```python
s = np.sqrt(2.0 / (n_in + n_out))   # a standard deviation, not a variance
```

Xavier suits tanh. For small inputs tanh is nearly the identity, since $\tanh(0.5) = 0.462$, so a gain of 1 passes the signal on almost unchanged. Larger inputs get squeezed, since $\tanh(2) = 0.964$, so the match is close and not perfect. In a width-128, 50-layer tanh network at Xavier's scale, the last mean square is 0.017, a gentle loss spread over fifty layers.

## What a ReLU throws away

Put a ReLU after every layer and a gain of exactly 1 is still too small. After 50 layers of width 128 at Xavier's scale, the simulator reports a mean square near $3 \times 10^{-16}$. Something in the ReLU eats the signal.

Take four pre-activations, the numbers right before the ReLU: $-2, -1, 1, 2$. Their mean square is $(4 + 1 + 1 + 4)/4 = 2.5$. After the ReLU they are $0, 0, 1, 2$, and the mean square is $(0 + 0 + 1 + 4)/4 = 1.25$. Exactly half survives. That is not luck. A pre-activation in a freshly initialised layer is a sum of weights times inputs, and the weights are as likely to be negative as positive, so a value is as likely to be $-a$ as $+a$. The square of $-a$ equals the square of $+a$, so the negative values hold exactly half of the squared size, and the ReLU removes them. Figure 3 shows this as an area: weight every value by its square, and the area under the curve is the mean square. A sample of 100,000 standard normal numbers went from a mean square of 1.000 to 0.499.

```figure
<svg viewBox="0 0 620 310" role="img" aria-label="Two curves of squared size, before and after a ReLU. The negative half of the area is removed.">
<polygon points="20.0,240.0 20.0,234.3 21.8,233.4 23.6,232.4 25.4,231.3 27.1,230.0 28.9,228.6 30.7,227.0 32.5,225.3 34.3,223.3 36.1,221.2 37.9,218.8 39.6,216.2 41.4,213.4 43.2,210.3 45.0,207.0 46.8,203.5 48.6,199.6 50.4,195.6 52.1,191.2 53.9,186.6 55.7,181.8 57.5,176.7 59.3,171.5 61.1,166.0 62.9,160.4 64.6,154.6 66.4,148.8 68.2,142.9 70.0,137.0 71.8,131.1 73.6,125.3 75.4,119.6 77.1,114.2 78.9,109.0 80.7,104.1 82.5,99.6 84.3,95.6 86.1,92.1 87.9,89.1 89.6,86.8 91.4,85.2 93.2,84.3 95.0,84.1 96.8,84.7 98.6,86.1 100.4,88.4 102.1,91.4 103.9,95.3 105.7,100.0 107.5,105.4 109.3,111.5 111.1,118.2 112.9,125.5 114.6,133.3 116.4,141.5 118.2,150.0 120.0,158.7 121.8,167.5 123.6,176.3 125.4,184.9 127.1,193.2 128.9,201.2 130.7,208.7 132.5,215.6 134.3,221.8 136.1,227.2 137.9,231.7 139.6,235.3 141.4,237.9 143.2,239.5 145.0,240.0 145.0,240.0" class="fill-soft-b"/>
<polygon points="145.0,240.0 145.0,240.0 146.8,239.5 148.6,237.9 150.4,235.3 152.1,231.7 153.9,227.2 155.7,221.8 157.5,215.6 159.3,208.7 161.1,201.2 162.9,193.2 164.6,184.9 166.4,176.3 168.2,167.5 170.0,158.7 171.8,150.0 173.6,141.5 175.4,133.3 177.1,125.5 178.9,118.2 180.7,111.5 182.5,105.4 184.3,100.0 186.1,95.3 187.9,91.4 189.6,88.4 191.4,86.1 193.2,84.7 195.0,84.1 196.8,84.3 198.6,85.2 200.4,86.8 202.1,89.1 203.9,92.1 205.7,95.6 207.5,99.6 209.3,104.1 211.1,109.0 212.9,114.2 214.6,119.6 216.4,125.3 218.2,131.1 220.0,137.0 221.8,142.9 223.6,148.8 225.4,154.6 227.1,160.4 228.9,166.0 230.7,171.5 232.5,176.7 234.3,181.8 236.1,186.6 237.9,191.2 239.6,195.6 241.4,199.6 243.2,203.5 245.0,207.0 246.8,210.3 248.6,213.4 250.4,216.2 252.1,218.8 253.9,221.2 255.7,223.3 257.5,225.3 259.3,227.0 261.1,228.6 262.9,230.0 264.6,231.3 266.4,232.4 268.2,233.4 270.0,234.3 270.0,240.0" class="fill-soft-a"/>
<polyline points="20.0,234.3 21.8,233.4 23.6,232.4 25.4,231.3 27.1,230.0 28.9,228.6 30.7,227.0 32.5,225.3 34.3,223.3 36.1,221.2 37.9,218.8 39.6,216.2 41.4,213.4 43.2,210.3 45.0,207.0 46.8,203.5 48.6,199.6 50.4,195.6 52.1,191.2 53.9,186.6 55.7,181.8 57.5,176.7 59.3,171.5 61.1,166.0 62.9,160.4 64.6,154.6 66.4,148.8 68.2,142.9 70.0,137.0 71.8,131.1 73.6,125.3 75.4,119.6 77.1,114.2 78.9,109.0 80.7,104.1 82.5,99.6 84.3,95.6 86.1,92.1 87.9,89.1 89.6,86.8 91.4,85.2 93.2,84.3 95.0,84.1 96.8,84.7 98.6,86.1 100.4,88.4 102.1,91.4 103.9,95.3 105.7,100.0 107.5,105.4 109.3,111.5 111.1,118.2 112.9,125.5 114.6,133.3 116.4,141.5 118.2,150.0 120.0,158.7 121.8,167.5 123.6,176.3 125.4,184.9 127.1,193.2 128.9,201.2 130.7,208.7 132.5,215.6 134.3,221.8 136.1,227.2 137.9,231.7 139.6,235.3 141.4,237.9 143.2,239.5 145.0,240.0" fill="none" class="stroke-b"/>
<polyline points="145.0,240.0 146.8,239.5 148.6,237.9 150.4,235.3 152.1,231.7 153.9,227.2 155.7,221.8 157.5,215.6 159.3,208.7 161.1,201.2 162.9,193.2 164.6,184.9 166.4,176.3 168.2,167.5 170.0,158.7 171.8,150.0 173.6,141.5 175.4,133.3 177.1,125.5 178.9,118.2 180.7,111.5 182.5,105.4 184.3,100.0 186.1,95.3 187.9,91.4 189.6,88.4 191.4,86.1 193.2,84.7 195.0,84.1 196.8,84.3 198.6,85.2 200.4,86.8 202.1,89.1 203.9,92.1 205.7,95.6 207.5,99.6 209.3,104.1 211.1,109.0 212.9,114.2 214.6,119.6 216.4,125.3 218.2,131.1 220.0,137.0 221.8,142.9 223.6,148.8 225.4,154.6 227.1,160.4 228.9,166.0 230.7,171.5 232.5,176.7 234.3,181.8 236.1,186.6 237.9,191.2 239.6,195.6 241.4,199.6 243.2,203.5 245.0,207.0 246.8,210.3 248.6,213.4 250.4,216.2 252.1,218.8 253.9,221.2 255.7,223.3 257.5,225.3 259.3,227.0 261.1,228.6 262.9,230.0 264.6,231.3 266.4,232.4 268.2,233.4 270.0,234.3" fill="none" class="stroke-a"/>
<line x1="20.0" y1="240.0" x2="270.0" y2="240.0" class="stroke-muted"/>
<line x1="73.6" y1="240.0" x2="73.6" y2="245.0" class="stroke-muted"/>
<text x="73.6" y="260.0" class="fig-small" text-anchor="middle">-2</text>
<line x1="145.0" y1="240.0" x2="145.0" y2="245.0" class="stroke-muted"/>
<text x="145.0" y="260.0" class="fig-small" text-anchor="middle">0</text>
<line x1="216.4" y1="240.0" x2="216.4" y2="245.0" class="stroke-muted"/>
<text x="216.4" y="260.0" class="fig-small" text-anchor="middle">2</text>
<text x="145.0" y="280.0" class="fig-small" text-anchor="middle">pre-activation value x</text>
<polygon points="455.0,240.0 455.0,240.0 456.8,239.5 458.6,237.9 460.4,235.3 462.1,231.7 463.9,227.2 465.7,221.8 467.5,215.6 469.3,208.7 471.1,201.2 472.9,193.2 474.6,184.9 476.4,176.3 478.2,167.5 480.0,158.7 481.8,150.0 483.6,141.5 485.4,133.3 487.1,125.5 488.9,118.2 490.7,111.5 492.5,105.4 494.3,100.0 496.1,95.3 497.9,91.4 499.6,88.4 501.4,86.1 503.2,84.7 505.0,84.1 506.8,84.3 508.6,85.2 510.4,86.8 512.1,89.1 513.9,92.1 515.7,95.6 517.5,99.6 519.3,104.1 521.1,109.0 522.9,114.2 524.6,119.6 526.4,125.3 528.2,131.1 530.0,137.0 531.8,142.9 533.6,148.8 535.4,154.6 537.1,160.4 538.9,166.0 540.7,171.5 542.5,176.7 544.3,181.8 546.1,186.6 547.9,191.2 549.6,195.6 551.4,199.6 553.2,203.5 555.0,207.0 556.8,210.3 558.6,213.4 560.4,216.2 562.1,218.8 563.9,221.2 565.7,223.3 567.5,225.3 569.3,227.0 571.1,228.6 572.9,230.0 574.6,231.3 576.4,232.4 578.2,233.4 580.0,234.3 580.0,240.0" class="fill-soft-a"/>
<polyline points="330.0,234.3 331.8,233.4 333.6,232.4 335.4,231.3 337.1,230.0 338.9,228.6 340.7,227.0 342.5,225.3 344.3,223.3 346.1,221.2 347.9,218.8 349.6,216.2 351.4,213.4 353.2,210.3 355.0,207.0 356.8,203.5 358.6,199.6 360.4,195.6 362.1,191.2 363.9,186.6 365.7,181.8 367.5,176.7 369.3,171.5 371.1,166.0 372.9,160.4 374.6,154.6 376.4,148.8 378.2,142.9 380.0,137.0 381.8,131.1 383.6,125.3 385.4,119.6 387.1,114.2 388.9,109.0 390.7,104.1 392.5,99.6 394.3,95.6 396.1,92.1 397.9,89.1 399.6,86.8 401.4,85.2 403.2,84.3 405.0,84.1 406.8,84.7 408.6,86.1 410.4,88.4 412.1,91.4 413.9,95.3 415.7,100.0 417.5,105.4 419.3,111.5 421.1,118.2 422.9,125.5 424.6,133.3 426.4,141.5 428.2,150.0 430.0,158.7 431.8,167.5 433.6,176.3 435.4,184.9 437.1,193.2 438.9,201.2 440.7,208.7 442.5,215.6 444.3,221.8 446.1,227.2 447.9,231.7 449.6,235.3 451.4,237.9 453.2,239.5 455.0,240.0" fill="none" class="stroke-b dashed"/>
<polyline points="455.0,240.0 456.8,239.5 458.6,237.9 460.4,235.3 462.1,231.7 463.9,227.2 465.7,221.8 467.5,215.6 469.3,208.7 471.1,201.2 472.9,193.2 474.6,184.9 476.4,176.3 478.2,167.5 480.0,158.7 481.8,150.0 483.6,141.5 485.4,133.3 487.1,125.5 488.9,118.2 490.7,111.5 492.5,105.4 494.3,100.0 496.1,95.3 497.9,91.4 499.6,88.4 501.4,86.1 503.2,84.7 505.0,84.1 506.8,84.3 508.6,85.2 510.4,86.8 512.1,89.1 513.9,92.1 515.7,95.6 517.5,99.6 519.3,104.1 521.1,109.0 522.9,114.2 524.6,119.6 526.4,125.3 528.2,131.1 530.0,137.0 531.8,142.9 533.6,148.8 535.4,154.6 537.1,160.4 538.9,166.0 540.7,171.5 542.5,176.7 544.3,181.8 546.1,186.6 547.9,191.2 549.6,195.6 551.4,199.6 553.2,203.5 555.0,207.0 556.8,210.3 558.6,213.4 560.4,216.2 562.1,218.8 563.9,221.2 565.7,223.3 567.5,225.3 569.3,227.0 571.1,228.6 572.9,230.0 574.6,231.3 576.4,232.4 578.2,233.4 580.0,234.3" fill="none" class="stroke-a"/>
<line x1="330.0" y1="240.0" x2="455.0" y2="240.0" class="stroke-b"/>
<line x1="330.0" y1="240.0" x2="580.0" y2="240.0" class="stroke-muted"/>
<line x1="383.6" y1="240.0" x2="383.6" y2="245.0" class="stroke-muted"/>
<text x="383.6" y="260.0" class="fig-small" text-anchor="middle">-2</text>
<line x1="455.0" y1="240.0" x2="455.0" y2="245.0" class="stroke-muted"/>
<text x="455.0" y="260.0" class="fig-small" text-anchor="middle">0</text>
<line x1="526.4" y1="240.0" x2="526.4" y2="245.0" class="stroke-muted"/>
<text x="526.4" y="260.0" class="fig-small" text-anchor="middle">2</text>
<text x="455.0" y="280.0" class="fig-small" text-anchor="middle">value after ReLU</text>
<text x="150.0" y="28.0" class="fig-bold" text-anchor="middle">before ReLU: area 1.00</text>
<text x="460.0" y="28.0" class="fig-bold" text-anchor="middle">after ReLU: area 0.50</text>
<text x="150.0" y="46.0" class="fig-small" text-anchor="middle">height = x² · p(x)</text>
<text x="460.0" y="46.0" class="fig-small" text-anchor="middle">negative side removed</text>
<text x="95.0" y="215.0" class="fig-bold" text-anchor="middle">0.50</text>
<text x="195.0" y="215.0" class="fig-bold" text-anchor="middle">0.50</text>
<text x="505.0" y="215.0" class="fig-bold" text-anchor="middle">0.50</text>
<text x="405.0" y="215.0" class="fig-bold" text-anchor="middle">0</text>
</svg>
Figure 3. The height of each curve is x² times how often the value x occurs, so the area under it is the mean square. Before the ReLU (left) the curve is symmetric, so the negative half (orange) holds exactly half of the area. The ReLU turns every negative value into 0 and removes that half (right). In a sample of 100,000 standard normal numbers the mean square fell from 1.000 to 0.499.
```

Next, count the whole layer. The linear part multiplies $q$ by $n_{\text{in}} s^2$, and the ReLU halves it, so one ReLU layer multiplies $q$ by $\tfrac{1}{2} n_{\text{in}} s^2$. With Xavier's scale on a square layer that is $\tfrac12$ per layer, and $0.5^{50} \approx 9 \times 10^{-16}$ matches the simulator's $3 \times 10^{-16}$. The chapter tracks the mean square and not the variance because the numbers after a ReLU are never negative, so they are not centred. The next layer only needs their mean square, because the cross terms vanish whatever the inputs' average is.

If the ReLU halves the mean square, double the gain of the linear layer so that the two cancel. You want $\tfrac12 n_{\text{in}} s^2 = 1$, which gives

$$
s_{\text{He}} = \sqrt{\frac{2}{n_{\text{in}}}}
$$

Read it aloud: set the weight variance to two over the fan-in. This is **He initialisation**, after Kaiming He. For width 128, $s = \sqrt{2/128} = 0.125$, against Xavier's $0.0884$, a ratio of $\sqrt{2}$ because the variance doubles. In code:

```python
s = np.sqrt(2.0 / n_in)
```

Figure 4 puts both scales through the same 50-layer ReLU network. With He's scale the size wanders but has no trend: it was 0.46 at layer 10 and 0.36 at layer 50 in the seed-0 run, and three other seeds end at 0.16, 1.6, and 1.2. With Xavier's scale it falls in a straight line on the log axis.

```figure
<svg viewBox="0 0 620 350" role="img" aria-label="Mean square against layer number on a log axis for a 50-layer ReLU network under He and Xavier scales">
<line x1="70.0" y1="300.0" x2="470.0" y2="300.0" class="stroke-muted"/>
<line x1="70.0" y1="300.0" x2="70.0" y2="30.0" class="stroke-grid"/>
<line x1="70.0" y1="285.8" x2="470.0" y2="285.8" class="stroke-grid"/>
<text x="62.0" y="289.8" class="fig-small" text-anchor="end">10⁻¹⁶</text>
<line x1="70.0" y1="228.9" x2="470.0" y2="228.9" class="stroke-grid"/>
<text x="62.0" y="232.9" class="fig-small" text-anchor="end">10⁻¹²</text>
<line x1="70.0" y1="172.1" x2="470.0" y2="172.1" class="stroke-grid"/>
<text x="62.0" y="176.1" class="fig-small" text-anchor="end">10⁻⁸</text>
<line x1="70.0" y1="115.3" x2="470.0" y2="115.3" class="stroke-grid"/>
<text x="62.0" y="119.3" class="fig-small" text-anchor="end">10⁻⁴</text>
<line x1="70.0" y1="58.4" x2="470.0" y2="58.4" class="stroke-grid"/>
<text x="62.0" y="62.4" class="fig-small" text-anchor="end">10⁰</text>
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
<polyline points="70.0,58.5 78.0,58.5 86.0,58.7 94.0,59.6 102.0,60.5 110.0,61.4 118.0,60.8 126.0,61.5 134.0,64.2 142.0,63.8 150.0,63.5 158.0,62.6 166.0,62.6 174.0,62.2 182.0,62.4 190.0,61.0 198.0,60.4 206.0,61.3 214.0,59.9 222.0,59.8 230.0,60.4 238.0,61.3 246.0,60.6 254.0,61.7 262.0,61.4 270.0,62.8 278.0,63.1 286.0,63.7 294.0,63.9 302.0,61.7 310.0,61.6 318.0,63.9 326.0,63.4 334.0,62.7 342.0,63.8 350.0,63.1 358.0,60.6 366.0,59.7 374.0,57.3 382.0,57.2 390.0,56.1 398.0,57.0 406.0,57.2 414.0,57.5 422.0,58.0 430.0,57.0 438.0,56.8 446.0,56.2 454.0,53.4 462.0,55.6 470.0,57.4" fill="none" class="stroke-a dashed"/>
<polyline points="70.0,58.4 78.0,58.5 86.0,59.6 94.0,60.4 102.0,60.5 110.0,59.3 118.0,58.0 126.0,59.6 134.0,60.0 142.0,61.3 150.0,61.9 158.0,61.3 166.0,59.8 174.0,60.5 182.0,59.6 190.0,58.0 198.0,57.2 206.0,57.6 214.0,57.7 222.0,57.7 230.0,58.4 238.0,56.8 246.0,55.7 254.0,55.7 262.0,56.3 270.0,55.6 278.0,57.4 286.0,56.9 294.0,55.7 302.0,54.6 310.0,54.9 318.0,55.8 326.0,54.7 334.0,54.3 342.0,54.2 350.0,55.8 358.0,56.0 366.0,53.3 374.0,54.1 382.0,54.1 390.0,54.5 398.0,53.0 406.0,53.5 414.0,56.5 422.0,55.0 430.0,55.1 438.0,57.7 446.0,55.5 454.0,56.0 462.0,56.1 470.0,55.4" fill="none" class="stroke-a dashed"/>
<polyline points="70.0,58.5 78.0,58.4 86.0,58.6 94.0,59.2 102.0,58.6 110.0,57.0 118.0,56.8 126.0,56.6 134.0,56.4 142.0,57.1 150.0,56.9 158.0,57.2 166.0,56.4 174.0,57.3 182.0,57.7 190.0,56.8 198.0,58.4 206.0,58.2 214.0,58.9 222.0,62.5 230.0,61.9 238.0,61.0 246.0,62.5 254.0,62.3 262.0,61.6 270.0,60.5 278.0,60.7 286.0,65.6 294.0,65.4 302.0,65.8 310.0,64.1 318.0,63.1 326.0,61.4 334.0,59.5 342.0,60.3 350.0,60.5 358.0,60.0 366.0,60.5 374.0,60.2 382.0,61.4 390.0,62.2 398.0,64.8 406.0,65.1 414.0,65.2 422.0,65.6 430.0,67.3 438.0,67.4 446.0,67.2 454.0,68.3 462.0,68.0 470.0,69.6" fill="none" class="stroke-a dashed"/>
<polyline points="70.0,58.4 78.0,58.4 86.0,58.6 94.0,58.4 102.0,58.5 110.0,59.5 118.0,60.2 126.0,61.2 134.0,62.5 142.0,62.6 150.0,63.2 158.0,61.8 166.0,60.7 174.0,61.9 182.0,61.3 190.0,60.7 198.0,62.0 206.0,61.1 214.0,59.4 222.0,62.0 230.0,61.0 238.0,60.3 246.0,58.9 254.0,59.3 262.0,59.1 270.0,58.7 278.0,58.7 286.0,60.8 294.0,61.2 302.0,62.1 310.0,62.7 318.0,61.6 326.0,60.9 334.0,60.9 342.0,62.1 350.0,61.1 358.0,61.2 366.0,61.6 374.0,62.8 382.0,63.1 390.0,65.0 398.0,65.1 406.0,65.2 414.0,64.4 422.0,63.3 430.0,63.9 438.0,61.2 446.0,63.7 454.0,64.7 462.0,64.3 470.0,64.7" fill="none" class="stroke-a"/>
<polyline points="70.0,58.5 78.0,62.8 86.0,67.3 94.0,72.5 102.0,77.6 110.0,82.8 118.0,86.5 126.0,91.4 134.0,98.4 142.0,102.3 150.0,106.3 158.0,109.6 166.0,113.9 174.0,117.8 182.0,122.3 190.0,125.1 198.0,128.9 206.0,134.0 214.0,136.9 222.0,141.1 230.0,146.0 238.0,151.2 246.0,154.7 254.0,160.0 262.0,164.1 270.0,169.8 278.0,174.3 286.0,179.2 294.0,183.7 302.0,185.8 310.0,190.0 318.0,196.5 326.0,200.3 334.0,203.9 342.0,209.3 350.0,212.8 358.0,214.6 366.0,218.0 374.0,219.9 382.0,224.1 390.0,227.2 398.0,232.4 406.0,236.9 414.0,241.5 422.0,246.2 430.0,249.5 438.0,253.6 446.0,257.2 454.0,258.8 462.0,265.2 470.0,271.3" fill="none" class="stroke-b dashed"/>
<polyline points="70.0,58.4 78.0,62.8 86.0,68.2 94.0,73.2 102.0,77.7 110.0,80.6 118.0,83.7 126.0,89.6 134.0,94.2 142.0,99.8 150.0,104.7 158.0,108.3 166.0,111.2 174.0,116.2 182.0,119.5 190.0,122.2 198.0,125.6 206.0,130.3 214.0,134.7 222.0,139.0 230.0,143.9 238.0,146.7 246.0,149.8 254.0,154.1 262.0,159.0 270.0,162.5 278.0,168.6 286.0,172.4 294.0,175.5 302.0,178.7 310.0,183.2 318.0,188.4 326.0,191.6 334.0,195.4 342.0,199.7 350.0,205.5 358.0,210.0 366.0,211.5 374.0,216.7 382.0,221.0 390.0,225.6 398.0,228.4 406.0,233.2 414.0,240.4 422.0,243.3 430.0,247.6 438.0,254.5 446.0,256.6 454.0,261.3 462.0,265.7 470.0,269.3" fill="none" class="stroke-b dashed"/>
<polyline points="70.0,58.5 78.0,62.6 86.0,67.2 94.0,72.0 102.0,75.7 110.0,78.4 118.0,82.5 126.0,86.6 134.0,90.6 142.0,95.6 150.0,99.7 158.0,104.3 166.0,107.8 174.0,112.9 182.0,117.6 190.0,121.0 198.0,126.8 206.0,130.9 214.0,135.9 222.0,143.8 230.0,147.5 238.0,150.8 246.0,156.6 254.0,160.7 262.0,164.3 270.0,167.5 278.0,171.9 286.0,181.1 294.0,185.2 302.0,189.8 310.0,192.4 318.0,195.7 326.0,198.3 334.0,200.7 342.0,205.7 350.0,210.2 358.0,214.0 366.0,218.8 374.0,222.7 382.0,228.2 390.0,233.3 398.0,240.2 406.0,244.7 414.0,249.1 422.0,253.8 430.0,259.8 438.0,264.2 446.0,268.3 454.0,273.7 462.0,277.6 470.0,283.5" fill="none" class="stroke-b dashed"/>
<polyline points="70.0,58.4 78.0,62.7 86.0,67.2 94.0,71.3 102.0,75.6 110.0,80.9 118.0,85.9 126.0,91.1 134.0,96.7 142.0,101.1 150.0,106.0 158.0,108.8 166.0,112.1 174.0,117.6 182.0,121.2 190.0,124.8 198.0,130.5 206.0,133.8 214.0,136.4 222.0,143.3 230.0,146.6 238.0,150.1 246.0,153.1 254.0,157.7 262.0,161.8 270.0,165.6 278.0,169.9 286.0,176.3 294.0,181.0 302.0,186.1 310.0,191.1 318.0,194.2 326.0,197.8 334.0,202.1 342.0,207.6 350.0,210.9 358.0,215.2 366.0,219.9 374.0,225.4 382.0,230.0 390.0,236.1 398.0,240.5 406.0,244.9 414.0,248.3 422.0,251.5 430.0,256.4 438.0,258.0 446.0,264.8 454.0,270.0 462.0,273.9 470.0,278.6" fill="none" class="stroke-b"/>
<text x="480.0" y="68.7" class="fig-bold" text-anchor="start">He</text>
<text x="480.0" y="282.6" class="fig-bold" text-anchor="start">Xavier: 3×10⁻¹⁶</text>
<text x="80.0" y="38.0" class="fig-small" text-anchor="start">solid: seed 0.  dashed: seeds 1 to 3</text>
<text x="300.0" y="149.9" class="fig-small" text-anchor="start">0.5 per layer</text>
</svg>
Figure 4. Mean square through 50 ReLU layers of width 128, from the exercise's own simulator. With He's scale (blue) the size wanders, staying between about 0.15 and 2.5, and shows no trend. With Xavier's scale (orange) it loses about half per layer, in a straight line on this log axis, and ends near 3×10⁻¹⁶, close to the 0.5⁵⁰ ≈ 9×10⁻¹⁶ that halving predicts.
```

Why does He use only the fan-in? It protects the forward signal, which every later layer reads. After the ReLU's mask removes half of the backward gradient, its gain through a layer is about $n_{\text{out}} s^2 / 2 = n_{\text{out}} / n_{\text{in}}$. A simulation of the 256-to-64 layer at He's scale gave 0.248, against $64/256 = 0.25$. That is harmless when neighbouring layers have similar width, and a fan-out variant of He protects the backward pass when it matters.

He is not meant for tanh. With He's scale in a tanh network the pre-activations start with a mean square of 2.0 and tanh squeezes them, so the size settles near 0.3 and not at 1. About 7% of the units end up with $|h| > 0.9$, where tanh's slope is below 0.19, and those units pass almost no gradient.

> [!KEY]
> A layer's forward pass counts its fan-in and its backward pass counts its fan-out, so Xavier averages the two and sets $s^2 = 2/(n_{\text{in}} + n_{\text{out}})$. A ReLU removes half of the mean square, so He doubles the weight variance to $2/n_{\text{in}}$. Use Xavier for tanh-like activations and He for ReLU. A gain of exactly 1 is the right target only when the activation keeps the size.

> [!RECAP]
> - For weights with mean 0 drawn separately from each other, one layer gives $q_{\text{out}} = n_{\text{in}} s^2 q_{\text{in}}$, because the cross terms average to zero. After $L$ layers the size is $g^L$ times the start.
> - The gain depends on the width, so no single weight scale suits every layer.
> - Backward, one input sums $n_{\text{out}}$ products, so the two gains are $n_{\text{in}} s^2$ and $n_{\text{out}} s^2$. Xavier sets their average to 1.
> - A ReLU removes exactly half of the mean square of symmetric numbers, so He uses $s = \sqrt{2/n_{\text{in}}}$ and a ReLU layer has gain 1.

## Build it step by step

You write six functions, in the order the checks test them:

1. **`layer_gain(n_in, scale)`** takes the fan-in and the weight standard deviation and returns `float(n_in * scale ** 2)`. The usual mistake is to forget that `scale` is a standard deviation, so the gain uses its square.
2. **`depth_gain(gain, depth)`** returns the factor after `depth` identical layers, which is `gain ** depth`. The usual mistake is `gain * depth`, which grows in a straight line and misses the compounding.
3. **`forward_variance(depth, widths, scale, activation, num_trials, seed)`** simulates the stack and returns `depth + 1` mean squares. Start with `rng = np.random.default_rng(seed)` and draw the input `h` of shape `(num_trials, widths[0])`. Record `np.mean(h ** 2)`. Then, for each layer, draw `W` of shape `(widths[l], widths[l + 1])` with `rng.normal(scale=scale, ...)`, replace `h` with `h @ W`, and record again. When `activation` is `"relu"`, apply `np.maximum(h, 0.0)` after the multiply. The usual mistake is to apply ReLU to the input before the first layer.
4. **`xavier_scale(n_in, n_out)`** returns `np.sqrt(2.0 / (n_in + n_out))` as a float. The usual mistake is to return $2/(n_{\text{in}} + n_{\text{out}})$, the variance, and skip the square root.
5. **`relu_mean_square_ratio(preact)`** returns the mean square after a ReLU divided by the mean square before it, for an array of any shape. Use `np.maximum(preact, 0.0)` and `np.mean(... ** 2)`. The usual mistake is to use `np.var`, which subtracts the mean, when you want the mean of the squares.
6. **`he_scale(n_in)`** returns `np.sqrt(2.0 / n_in)` as a float. The usual mistake is to copy Xavier and include `n_out`.

## Where this breaks

- A single network wobbles around the prediction, and the wobble compounds over many layers. A gain of exactly 1 ended at 3.3 after 50 layers in Figure 1, not at 1.
- Weights that are not centred on 0, or that are correlated, add their cross terms back in, and the formula under-counts. The one-half rule for ReLU also needs numbers that are symmetric around 0, so it holds at initialisation and only roughly after training.
- After training starts, the weights move away from their initial values, so the gain at step 10,000 is not the gain you set. [Normalisation layers](/lessons/033-normalisation-layers) restore the scale on every pass.

## The exercise

You write `layer_gain`, `depth_gain`, `forward_variance`, `xavier_scale`, `relu_mean_square_ratio`, and `he_scale`.

The checks verify the gain formula with exact numbers, the compounding with $0.7^{50}$ and $1.4^{50}$, and that `forward_variance` returns `depth + 1` values that start at the input's mean square. They run ten layers of width 256 with gains of 0.8 and 1.3 and compare the measured size with your prediction within a factor of 2. They test Xavier's scale on the 256-to-64 layer, compute the ReLU ratio by hand on $-2, -1, 1, 2$ and on random numbers, and verify He's scale. The last check runs fifty ReLU layers of width 128: He must keep the signal near its starting size, and Xavier must lose about half per layer.
