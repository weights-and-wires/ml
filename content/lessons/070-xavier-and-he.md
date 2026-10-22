---
title: Xavier and He initialisation
slug: 070-xavier-and-he
part: "Part 4 — Neural networks"
summary: Set each layer's weight scale from its fan-in and fan-out so the gain is 1, and double the weight variance in front of a ReLU.
minutes: 35
gpu: false
objectives:
  - Explain why the forward pass counts a layer's inputs and the backward pass counts its outputs, and derive the Xavier scale from the two.
  - Show why a ReLU removes half of the mean square of symmetric numbers, and derive the He scale from that.
  - Say which of Xavier and He suits tanh and which suits ReLU, and why.
  - Confirm on a fifty-layer ReLU network that He keeps the signal and Xavier loses half per layer.
prereqs:
  - 058-forward-pass-and-gelu
  - 064-backprop-mlp-and-gradient-check
  - 069-signal-size-through-layers
exercise: 070-xavier-and-he
---

# Xavier and He initialisation

Last chapter you found that a layer's gain is $n_{\text{in}} s^2$, and that a network survives fifty layers only when that gain is close to 1. So for a layer with 256 inputs you set $s^2 = 1/256$, and the forward signal is safe. Then you add a layer that narrows the network from 256 units to 64, and a question you had not met appears: the gradient flowing back through that layer sums over 64 terms, not 256, so it sees a different gain.

A second problem hides behind the first. Put a ReLU after every layer and a gain of exactly 1 is still too small. After 50 layers of width 128 the simulator reports a mean square near $3 \times 10^{-16}$. Something in the ReLU eats the signal.

This chapter answers both. How do you choose $s$ when a layer's number of inputs and number of outputs differ, and why does a network of ReLUs need a different answer from a network of tanh units?

## What you already know

Four ideas carry this chapter:

- A linear layer multiplies the mean square $q$ of its signal by the gain $n_{\text{in}} s^2$ (see "Why a deep network's signal shrinks or explodes").
- The number of inputs to one unit is its **fan-in**, $n_{\text{in}}$. The number of units an input feeds is its **fan-out**, $n_{\text{out}}$.
- The gradient at a layer's inputs is the transposed weight matrix times the gradient at its outputs (see "Putting it together: gradients for a whole network").
- ReLU replaces every negative number with 0 and keeps the positive ones (see "Run a network forward, with a sharp or smooth bend").

## Forward asks for one number, backward asks for another

Picture the layer as a set of input dots joined to output dots, as in Figure 1. Going forward, one output collects a product from each of the $n_{\text{in}}$ inputs. Going backward, one input's gradient collects a product from each of the $n_{\text{out}}$ outputs it feeds, because the gradient is multiplied by the transposed weights.

```figure
<svg viewBox="0 0 620 330" role="img" aria-label="A layer with 8 inputs and 2 outputs. Forward, one output sums 8 products. Backward, one input collects 2 products.">
<defs><marker id="c070-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c070-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<line x1="36.0" y1="60.0" x2="202.0" y2="88.0" class="stroke-grid"/>
<line x1="36.0" y1="60.0" x2="202.0" y2="200.0" class="stroke-grid"/>
<line x1="36.0" y1="88.0" x2="202.0" y2="88.0" class="stroke-grid"/>
<line x1="36.0" y1="88.0" x2="202.0" y2="200.0" class="stroke-grid"/>
<line x1="36.0" y1="116.0" x2="202.0" y2="88.0" class="stroke-grid"/>
<line x1="36.0" y1="116.0" x2="202.0" y2="200.0" class="stroke-grid"/>
<line x1="36.0" y1="144.0" x2="202.0" y2="88.0" class="stroke-grid"/>
<line x1="36.0" y1="144.0" x2="202.0" y2="200.0" class="stroke-grid"/>
<line x1="36.0" y1="172.0" x2="202.0" y2="88.0" class="stroke-grid"/>
<line x1="36.0" y1="172.0" x2="202.0" y2="200.0" class="stroke-grid"/>
<line x1="36.0" y1="200.0" x2="202.0" y2="88.0" class="stroke-grid"/>
<line x1="36.0" y1="200.0" x2="202.0" y2="200.0" class="stroke-grid"/>
<line x1="36.0" y1="228.0" x2="202.0" y2="88.0" class="stroke-grid"/>
<line x1="36.0" y1="228.0" x2="202.0" y2="200.0" class="stroke-grid"/>
<line x1="36.0" y1="256.0" x2="202.0" y2="88.0" class="stroke-grid"/>
<line x1="36.0" y1="256.0" x2="202.0" y2="200.0" class="stroke-grid"/>
<line x1="36.0" y1="60.0" x2="200.0" y2="88.0" class="stroke-a" marker-end="url(#c070-arrow-a)"/>
<line x1="36.0" y1="88.0" x2="200.0" y2="88.0" class="stroke-a" marker-end="url(#c070-arrow-a)"/>
<line x1="36.0" y1="116.0" x2="200.0" y2="88.0" class="stroke-a" marker-end="url(#c070-arrow-a)"/>
<line x1="36.0" y1="144.0" x2="200.0" y2="88.0" class="stroke-a" marker-end="url(#c070-arrow-a)"/>
<line x1="36.0" y1="172.0" x2="200.0" y2="88.0" class="stroke-a" marker-end="url(#c070-arrow-a)"/>
<line x1="36.0" y1="200.0" x2="200.0" y2="88.0" class="stroke-a" marker-end="url(#c070-arrow-a)"/>
<line x1="36.0" y1="228.0" x2="200.0" y2="88.0" class="stroke-a" marker-end="url(#c070-arrow-a)"/>
<line x1="36.0" y1="256.0" x2="200.0" y2="88.0" class="stroke-a" marker-end="url(#c070-arrow-a)"/>
<circle cx="30.0" cy="60.0" r="6" class="fill-muted"/>
<circle cx="30.0" cy="88.0" r="6" class="fill-muted"/>
<circle cx="30.0" cy="116.0" r="6" class="fill-muted"/>
<circle cx="30.0" cy="144.0" r="6" class="fill-muted"/>
<circle cx="30.0" cy="172.0" r="6" class="fill-muted"/>
<circle cx="30.0" cy="200.0" r="6" class="fill-muted"/>
<circle cx="30.0" cy="228.0" r="6" class="fill-muted"/>
<circle cx="30.0" cy="256.0" r="6" class="fill-muted"/>
<circle cx="210.0" cy="88.0" r="8" class="fill-a"/>
<circle cx="210.0" cy="200.0" r="8" class="fill-muted"/>
<circle cx="30.0" cy="60.0" r="6" class="fill-a"/>
<circle cx="30.0" cy="88.0" r="6" class="fill-a"/>
<circle cx="30.0" cy="116.0" r="6" class="fill-a"/>
<circle cx="30.0" cy="144.0" r="6" class="fill-a"/>
<circle cx="30.0" cy="172.0" r="6" class="fill-a"/>
<circle cx="30.0" cy="200.0" r="6" class="fill-a"/>
<circle cx="30.0" cy="228.0" r="6" class="fill-a"/>
<circle cx="30.0" cy="256.0" r="6" class="fill-a"/>
<circle cx="210.0" cy="88.0" r="8" class="fill-b"/>
<circle cx="210.0" cy="200.0" r="8" class="fill-muted"/>
<line x1="346.0" y1="60.0" x2="512.0" y2="88.0" class="stroke-grid"/>
<line x1="346.0" y1="60.0" x2="512.0" y2="200.0" class="stroke-grid"/>
<line x1="346.0" y1="88.0" x2="512.0" y2="88.0" class="stroke-grid"/>
<line x1="346.0" y1="88.0" x2="512.0" y2="200.0" class="stroke-grid"/>
<line x1="346.0" y1="116.0" x2="512.0" y2="88.0" class="stroke-grid"/>
<line x1="346.0" y1="116.0" x2="512.0" y2="200.0" class="stroke-grid"/>
<line x1="346.0" y1="144.0" x2="512.0" y2="88.0" class="stroke-grid"/>
<line x1="346.0" y1="144.0" x2="512.0" y2="200.0" class="stroke-grid"/>
<line x1="346.0" y1="172.0" x2="512.0" y2="88.0" class="stroke-grid"/>
<line x1="346.0" y1="172.0" x2="512.0" y2="200.0" class="stroke-grid"/>
<line x1="346.0" y1="200.0" x2="512.0" y2="88.0" class="stroke-grid"/>
<line x1="346.0" y1="200.0" x2="512.0" y2="200.0" class="stroke-grid"/>
<line x1="346.0" y1="228.0" x2="512.0" y2="88.0" class="stroke-grid"/>
<line x1="346.0" y1="228.0" x2="512.0" y2="200.0" class="stroke-grid"/>
<line x1="346.0" y1="256.0" x2="512.0" y2="88.0" class="stroke-grid"/>
<line x1="346.0" y1="256.0" x2="512.0" y2="200.0" class="stroke-grid"/>
<line x1="510.0" y1="88.0" x2="348.0" y2="144.0" class="stroke-b" marker-end="url(#c070-arrow-b)"/>
<line x1="510.0" y1="200.0" x2="348.0" y2="144.0" class="stroke-b" marker-end="url(#c070-arrow-b)"/>
<circle cx="340.0" cy="60.0" r="6" class="fill-muted"/>
<circle cx="340.0" cy="88.0" r="6" class="fill-muted"/>
<circle cx="340.0" cy="116.0" r="6" class="fill-muted"/>
<circle cx="340.0" cy="144.0" r="6" class="fill-b"/>
<circle cx="340.0" cy="172.0" r="6" class="fill-muted"/>
<circle cx="340.0" cy="200.0" r="6" class="fill-muted"/>
<circle cx="340.0" cy="228.0" r="6" class="fill-muted"/>
<circle cx="340.0" cy="256.0" r="6" class="fill-muted"/>
<circle cx="520.0" cy="88.0" r="8" class="fill-muted"/>
<circle cx="520.0" cy="200.0" r="8" class="fill-muted"/>
<text x="120.0" y="26.0" class="fig-bold" text-anchor="middle">Forward: one output sums 8 products</text>
<text x="430.0" y="26.0" class="fig-bold" text-anchor="middle">Backward: one input collects 2 products</text>
<text x="120.0" y="285.0" class="fig-small" text-anchor="middle">needs n_in·s² = 1</text>
<text x="430.0" y="285.0" class="fig-small" text-anchor="middle">needs n_out·s² = 1</text>
<text x="310.0" y="318.0" class="fig-small" text-anchor="middle">Xavier compromise: average the two gains, so s² = 2/(n_in + n_out)</text>
<text x="30.0" y="48.0" class="fig-small" text-anchor="start">inputs (n_in = 8)</text>
<text x="240.0" y="48.0" class="fig-small" text-anchor="end">outputs (n_out = 2)</text>
</svg>
Figure 1. The same layer seen from both ends. Going forward (left), each output unit sums one product for every input, so n_in terms. Going backward (right), each input unit's gradient sums one product for every output it feeds, so n_out terms. The ratio 8 to 2 matches a layer with 256 inputs and 64 outputs.
```

By the argument of the previous chapter, the forward pass multiplies the mean square by $n_{\text{in}} s^2$, and the backward pass multiplies the gradient's mean square by $n_{\text{out}} s^2$. For the 256-to-64 layer, the forward pass wants $s^2 = 1/256 \approx 0.0039$ and the backward pass wants $s^2 = 1/64 \approx 0.0156$. No single $s$ satisfies both. If you pick the forward value, the gradient shrinks by a factor of 4 at that layer. If you pick the backward value, the signal grows by 4.

The simple compromise is to ask for the two gains to average to 1:

$$
\frac{n_{\text{in}} s^2 + n_{\text{out}} s^2}{2} = 1
\quad\Rightarrow\quad
s_{\text{Xavier}} = \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}
$$

Read it aloud as "set the weight variance to two over the sum of the fan-in and the fan-out." This is **Xavier initialisation**, also called Glorot. The 2 in the numerator comes from taking an average of two counts. For the narrow layer, $s^2 = 2/320 = 1/160$, so $s \approx 0.0791$, the forward gain is $256/160 = 1.6$, and the backward gain is $64/160 = 0.4$. A simulation of that layer gives 1.60 and 0.40. Neither is exactly 1, and the average of the two is. When $n_{\text{in}} = n_{\text{out}} = 128$, both gains are exactly 1, and $s = \sqrt{2/256} = 0.0884$. In code:

```python
s = np.sqrt(2.0 / (n_in + n_out))   # a standard deviation, not a variance
```

Xavier was designed for activations like tanh, and the reason is the shape of the curve. For small inputs, tanh is nearly the identity: $\tanh(0.1) = 0.0997$ and $\tanh(0.5) = 0.462$. A gain of 1 then passes the signal on almost unchanged, so gain 1 is the right target. Larger inputs get squeezed, since $\tanh(2) = 0.964$, so the match is close and not perfect. In a width-128, 50-layer tanh network with Xavier's scale, the mean square after the last layer is $0.017$, which is a loss of about 60 times. That is a real drift, because the squeezing takes a little off at every layer. It is a gentle loss, spread over fifty layers, and it is nothing like the $10^{-16}$ that the same scale gives a ReLU network, where each layer cuts the size in half.

## What a ReLU throws away

Take four pre-activations, the numbers just before the ReLU: $-2, -1, 1, 2$. Their mean square is $(4 + 1 + 1 + 4)/4 = 2.5$. After the ReLU they are $0, 0, 1, 2$, and the mean square is $(0 + 0 + 1 + 4)/4 = 1.25$. Exactly half survives.

That is not luck. The numbers feeding a ReLU in a freshly initialised layer are sums of weights times inputs, and the weights are as likely to be negative as positive. So a pre-activation is as likely to be $-a$ as $+a$. The square of $-a$ equals the square of $+a$, so the negative values hold exactly half of the squared size, and the ReLU removes them. Figure 2 shows this as an area. Weight every value by its square, and the area under the curve is the mean square.

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
Figure 2. The height of each curve is x² times how often the value x occurs, so the area under it is the mean square. Before the ReLU (left) the curve is symmetric, so the negative half (orange) holds exactly half of the area. The ReLU turns every negative value into 0 and removes that half (right). In a sample of 100,000 standard normal numbers the mean square fell from 1.000 to 0.499.
```

The rule is exact for symmetric numbers: $q_{\text{after}} = \tfrac{1}{2} q_{\text{before}}$. A sample of 100,000 standard normal numbers gave a mean square of 1.000 before and 0.499 after.

Now count the whole layer. The linear part multiplies $q$ by $n_{\text{in}} s^2$, and the ReLU halves it, so one ReLU layer multiplies $q$ by $\tfrac{1}{2} n_{\text{in}} s^2$. With Xavier's scale on a square layer, that is $\tfrac12$ per layer. After 50 layers, $0.5^{50} \approx 9 \times 10^{-16}$, and the simulator, which uses random weights, gave $3 \times 10^{-16}$.

One detail explains why the chapter tracks mean square and not variance. After a ReLU the numbers are never negative, so their average is above zero and they are not centred. The formula for the next layer only needs their mean square, as the previous chapter showed: the cross terms vanish because the weights have mean 0, whatever the inputs' average is. So the mean square is the right quantity after a ReLU as well.

## He: put back what the ReLU removed

If the ReLU halves the mean square, double the gain of the linear layer so that the two cancel. You want $\tfrac12 n_{\text{in}} s^2 = 1$, which gives $n_{\text{in}} s^2 = 2$:

$$
s_{\text{He}} = \sqrt{\frac{2}{n_{\text{in}}}}
$$

Read it aloud as "set the weight variance to two over the fan-in." This is **He initialisation**, after Kaiming He. For width 128, $s = \sqrt{2/128} = 0.125$, compared with Xavier's $0.0884$. The ratio is $\sqrt{2}$, because the weight variance is doubled. In code:

```python
s = np.sqrt(2.0 / n_in)
```

Figure 3 puts the two scales through the same 50-layer ReLU network. With He's scale the size wanders but has no trend. It was 0.46 at layer 10 and 0.36 at layer 50 in the seed-0 run, and the other three seeds end at 0.16, 1.6, and 1.2. With Xavier's scale it falls in a straight line.

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
Figure 3. Mean square through 50 ReLU layers of width 128, from the exercise's own simulator. With He's scale (blue) the size wanders, staying between about 0.15 and 2.5, and shows no trend. With Xavier's scale (orange) it loses about half per layer, in a straight line on this log axis, and ends near 3×10⁻¹⁶, close to the 0.5⁵⁰ ≈ 9×10⁻¹⁶ that halving predicts.
```

Why does He use only the fan-in? It protects the forward signal, which every later layer reads. The backward gradient is a different matter. After the ReLU's mask removes half of it, its gain through a layer is about $n_{\text{out}} s^2 / 2 = n_{\text{out}} / n_{\text{in}}$. A simulation of the 256-to-64 layer with He's scale gave 0.248, against $64/256 = 0.25$. That is off by the ratio of the widths, and it is harmless when neighbouring layers have similar width. There is a fan-out variant of He that protects the backward pass instead, and you can pick it when it matters.

He is also not meant for tanh, and the failure is milder than a doubling. With He's scale in a tanh network, the pre-activations start with a mean square of 2.0 and tanh squeezes them, so the size settles at a mean square near 0.3 and not at 1. About 7% of the units end up with $|h| > 0.9$, where tanh's slope is below 0.19. Those units pass almost no gradient. You can check this yourself:

```python
h = rng.normal(size=(300, 128))
for _ in range(50):
    h = np.tanh(h @ rng.normal(scale=s, size=(128, 128)))
print(np.mean(h ** 2), np.mean(np.abs(h) > 0.9))
```

> [!KEY]
> A layer's forward pass counts its fan-in and its backward pass counts its fan-out, so Xavier averages the two and sets $s^2 = 2/(n_{\text{in}} + n_{\text{out}})$. A ReLU removes half of the mean square, so He doubles the weight variance to $2/n_{\text{in}}$. Use Xavier for tanh-like activations and He for ReLU, and a gain of exactly 1 is the right target only when the activation keeps the size.

> [!RECAP]
> - Forward, one output sums $n_{\text{in}}$ products. Backward, one input sums $n_{\text{out}}$ products, so the two gains are $n_{\text{in}} s^2$ and $n_{\text{out}} s^2$.
> - Xavier sets their average to 1: $s = \sqrt{2/(n_{\text{in}} + n_{\text{out}})}$.
> - A ReLU removes exactly half of the mean square of symmetric numbers, so with Xavier a ReLU network loses half per layer.
> - He doubles the weight variance, $s = \sqrt{2/n_{\text{in}}}$, so a ReLU layer has gain 1.
> - He protects the forward signal. The backward gain is about $n_{\text{out}}/n_{\text{in}}$.

## Build it step by step

The starter includes `layer_gain` and `forward_variance`, which you wrote in the previous chapter. The checks call the functions you write in this order:

1. **`xavier_scale(n_in, n_out)`** returns the standard deviation as a float:

   ```python
   np.sqrt(2.0 / (n_in + n_out))
   ```

   The usual mistake is to return $2/(n_{\text{in}} + n_{\text{out}})$, the variance, and skip the square root.

2. **`relu_mean_square_ratio(preact)`** returns the mean square after a ReLU divided by the mean square before it, for an array of any shape:

   ```python
   after = np.maximum(preact, 0.0)
   ```

   Divide `np.mean(after ** 2)` by `np.mean(preact ** 2)`. The usual mistake is to use `np.var`, which subtracts the mean, when you want the mean of the squares.

3. **`he_scale(n_in)`** returns `np.sqrt(2.0 / n_in)` as a float. The usual mistake is to copy Xavier and include `n_out`.

## Where this breaks

- The one-half rule needs numbers that are symmetric around 0. Biases, and weights that have been trained, break the symmetry, so it holds at initialisation and only roughly afterward.
- The gain is an average over random weights, so one network wobbles. In Figure 3, He's size ranges over a factor of several by layer 50.
- A tanh network is not harmed as much as a ReLU network by a wrong scale, because tanh flattens large inputs. It is harmed in a different way: saturated units pass almost no gradient.
- Very deep networks still drift, because the scale is set once and the weights move afterward. The chapter "Normalising activations: batch norm versus layer norm" puts the scale back on every pass.

## The exercise

You write `xavier_scale`, `relu_mean_square_ratio`, and `he_scale`. The starter supplies `layer_gain` and `forward_variance`.

The checks verify that Xavier's scale gives the formula and that the forward and backward gains of the 256-to-64 layer average to 1. They compute the ReLU ratio by hand on $-2, -1, 1, 2$, and on random numbers at two scales, where it must come out near one half. They verify He's scale and that half the gain of a He layer is 1. The last two checks run fifty ReLU layers of width 128: He must keep the signal near its starting size, and Xavier must lose about half per layer.
