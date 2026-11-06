---
title: Run a network forward, with a sharp or smooth bend
slug: 052-forward-pass-and-gelu
part: "Part 4 — Neural networks"
summary: Trace numbers through a network layer by layer, and learn GELU, the smooth bend, with its slope checked by finite differences.
minutes: 40
gpu: false
objectives:
  - Trace an input through a network by hand, naming the pre-activation and the activation at every unit.
  - Write the forward pass for a list of layers, using the shape (out, in) for each weight matrix.
  - Compute GELU from its tanh formula and explain why it is a smoothed version of ReLU.
  - Derive the GELU slope with the product rule and check it against finite differences.
prereqs:
  - 051-why-layers-need-a-bend
exercise: 052-forward-pass-and-gelu
---

# Run a network forward, with a sharp or smooth bend

A small network sits in front of you with three hidden units. You feed it the readings $[2, -3]$. What numbers sit at each stage on the way to the answer, and can you predict the final value before you run any code?

If you cannot, you are trusting a function you cannot debug. When a network returns nonsense, the first question is which layer produced the first wrong number, and you can only answer it if you know what the right numbers look like. You also cannot read a network written as one line of numpy, such as `h = relu(h @ W.T + b)` in a loop, without a picture of what that line does to each entry.

This chapter gives you that picture. It has one repeated move, mix and then bend, and you will trace it by hand before you write it as a loop. You will also meet a second kind of bend, a smooth one called GELU, so that your loop can switch between a sharp bend and a smooth one.

## What you already know

- A unit is one row of a weight matrix, and a layer is `x @ W.T + b` with `W` of shape `(out, in)` ("Why a stack of layers needs a bend").
- ReLU keeps positive numbers and cuts negative ones to 0, and its slope is 1 or 0.
- Without a bend between them, affine layers merge into one.
- A finite difference checks a slope: `(f(x + h) - f(x - h)) / (2 * h)`.
- The chain rule multiplies slopes along a chain of functions.

## Trace one input by hand

Figure 1 shows the network. It has two inputs, three hidden units, and one output. I chose the weights so each unit asks a question you can read off.

```figure
<svg viewBox="0 0 600 318" role="img" aria-label="A network with two inputs 2 and minus 3, three hidden units whose values before ReLU are 2, minus 3 and minus 2 and after ReLU are 2, 0 and 0, and one output 2.5.">

<line x1="114.0" y1="105.0" x2="276.0" y2="68.0" class="stroke-a" stroke-width="1.8"/>
<line x1="114.0" y1="105.0" x2="276.0" y2="156.0" class="stroke-muted dashed" stroke-width="1"/>
<line x1="114.0" y1="105.0" x2="276.0" y2="244.0" class="stroke-a" stroke-width="1.8"/>
<line x1="114.0" y1="195.0" x2="276.0" y2="68.0" class="stroke-muted dashed" stroke-width="1"/>
<line x1="114.0" y1="195.0" x2="276.0" y2="156.0" class="stroke-a" stroke-width="1.8"/>
<line x1="114.0" y1="195.0" x2="276.0" y2="244.0" class="stroke-a" stroke-width="1.8"/>
<line x1="324.0" y1="68.0" x2="486.0" y2="150.0" class="stroke-a" stroke-width="1.8"/>
<line x1="324.0" y1="156.0" x2="486.0" y2="150.0" class="stroke-a" stroke-width="1.8"/>
<line x1="324.0" y1="244.0" x2="486.0" y2="150.0" class="stroke-a" stroke-width="1.8"/>
<circle cx="90.0" cy="105.0" r="24" class="fill-soft-a stroke-a"/>
<text x="90.0" y="110.0" text-anchor="middle" class="fig-bold">2</text>
<text x="90.0" y="145.0" text-anchor="middle" class="fig-small">x1</text>
<circle cx="90.0" cy="195.0" r="24" class="fill-soft-a stroke-a"/>
<text x="90.0" y="200.0" text-anchor="middle" class="fig-bold">−3</text>
<text x="90.0" y="235.0" text-anchor="middle" class="fig-small">x2</text>
<circle cx="300.0" cy="68.0" r="24" class="fill-soft-b stroke-b"/>
<text x="300.0" y="73.0" text-anchor="middle" class="fig-bold">2</text>
<text x="330.0" y="46.0" text-anchor="start" class="fig-small">z = 2</text>
<circle cx="300.0" cy="156.0" r="24" class="fill-sunk stroke-muted"/>
<text x="300.0" y="161.0" text-anchor="middle" class="fig-bold">0</text>
<text x="330.0" y="134.0" text-anchor="start" class="fig-small">z = −3</text>
<line x1="283.2" y1="139.2" x2="316.8" y2="172.8" class="stroke-bad" stroke-width="2"/>
<circle cx="300.0" cy="244.0" r="24" class="fill-sunk stroke-muted"/>
<text x="300.0" y="249.0" text-anchor="middle" class="fig-bold">0</text>
<text x="330.0" y="278.0" text-anchor="start" class="fig-small">z = −2</text>
<line x1="283.2" y1="227.2" x2="316.8" y2="260.8" class="stroke-bad" stroke-width="2"/>
<text x="294.0" y="286.0" text-anchor="end" class="fig-small">unit 3 has constant −1</text>
<circle cx="510.0" cy="150.0" r="24" class="fill-soft-d stroke-d"/>
<text x="510.0" y="155.0" text-anchor="middle" class="fig-bold">2.5</text>
<text x="510.0" y="192.0" text-anchor="middle" class="fig-small">output</text>
<text x="510.0" y="114.0" text-anchor="middle" class="fig-small">+ bias 0.5</text>
<text x="90.0" y="22.0" text-anchor="middle" class="fig-bold">inputs</text>
<text x="300.0" y="22.0" text-anchor="middle" class="fig-bold">hidden layer (3 units)</text>
<text x="510.0" y="22.0" text-anchor="middle" class="fig-bold">output layer</text>
<text x="300.0" y="306.0" text-anchor="middle" class="fig-small">z = value before the bend; number inside the circle = value after ReLU</text>
</svg>
Figure 1. The running example, with every number retraced. Solid blue lines are weights equal to 1 and dashed grey lines are weights equal to 0. The value z is each unit's reading before the bend. ReLU keeps 2 and cuts the other two to 0 (crossed out), so the output unit adds 2 + 0 + 0 and the bias 0.5 to get 2.5.
```

- Unit 1 has weights $(1, 0)$ and constant 0, so it passes the first reading through. It reads $2$.
- Unit 2 has weights $(0, 1)$ and constant 0, so it passes the second reading through. It reads $-3$.
- Unit 3 has weights $(1, 1)$ and constant $-1$, so it asks whether the sum of the readings beats 1. It reads $2 + (-3) - 1 = -2$.

Each unit's raw reading, before any bend, is its **pre-activation**. Here the pre-activations are $[2, -3, -2]$. The bend turns them into the **activations**: ReLU gives $[2, 0, 0]$. The hidden layer is called hidden because you never observe its values directly, only the output. The output unit has weights $(1, 1, 1)$, constant $0.5$, and no bend, so it adds up what survived and shifts by $0.5$. That gives $2 + 0 + 0 + 0.5 = 2.5$.

Running the numbers from the input to the output, layer by layer, is the **forward pass**. Nothing about it is special to this network: every layer takes the last layer's activations, forms pre-activations, and applies its bend. That is why a network of thousands of units is still a short loop.

## The same move as code

Write the layer once with two conventions side by side. For a single input vector the math is

$$
h_{\text{out}} = \sigma(W h_{\text{in}} + b) ,
$$

where $\sigma$ is the bend applied to each entry. Read it aloud: "multiply by the weights, add the constants, then bend every entry." Numpy code usually holds data as rows, so the same step is

```python
z = h @ W.T + b
h = activation(z)
```

Both forms give the same numbers. In the example the shapes are $(3,)$ after the first layer and $(1,)$ after the second. The common mistake is to write `h @ W` and leave out the transpose. When `W` is `(3, 2)` and `h` is `(2,)`, numpy raises a shape error, so you notice at once. When `W` is square, it runs and returns the wrong numbers silently, because it reads the columns of `W` as units instead of the rows.

> [!TIP]
> Print `z.shape` after each layer. You want `(3,)` and then `(1,)`. Any other shape means `W` faces the wrong way.

## Soften the corner with GELU

ReLU has two small problems. At 0 its slope jumps from 0 to 1, a corner. And every negative reading is cut to exactly 0, so a unit that reads slightly negative contributes nothing, and a nudge to its reading changes nothing. Many modern networks use a smooth cousin that rounds the corner and lets a little of the negative side through.

It is called the **Gaussian error linear unit**, GELU for short. Its idea: multiply $x$ by the probability that a standard bell-curve random number falls below $x$. That probability is near 0 for very negative $x$, so $x$ is switched off, and near 1 for large $x$, so $x$ passes. In between the switch is gradual. The exact version needs the error function, so most code uses a cheaper stand-in built from **tanh**, a smooth S-shaped function that rises from $-1$ to $1$ and passes through 0:

$$
\operatorname{gelu}(x) = \tfrac{1}{2}\, x \left(1 + \tanh(u)\right), \qquad
u = \sqrt{2/\pi}\,\left(x + 0.044715\, x^3\right) .
$$

Read it aloud: "$u$ is a scaled cubic curve in $x$. Squash it with tanh, shift the result from the range $(-1, 1)$ up to $(0, 2)$, halve it to get a switch between 0 and 1, and multiply by $x$." The two constants are fitted so that the tanh switch tracks the exact bell-curve probability closely. Neither constant needs memorising.

Work $x = 1$. First $\sqrt{2/\pi} = 0.797885$ and $1 + 0.044715 = 1.044715$, so $u = 0.833562$. Then $\tanh(u) = 0.682384$, the switch is $\tfrac12(1 + 0.682384) = 0.841192$, and $\operatorname{gelu}(1) = 0.841192$. The exact bell-curve version gives $0.841345$, so the stand-in is close. At $x = -1$ the same steps give $u = -0.833562$, switch $0.158808$, and $\operatorname{gelu}(-1) = -0.158808$, a small dip below 0 that ReLU never makes. Figure 2 plots both curves. The dip is deepest near $x = -0.75$, at about $-0.17$.

In numpy:

```python
u = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
return 0.5 * x * (1 + np.tanh(u))
```

## The slope of GELU

Every later chapter that trains a network needs the slope of each bend, because the slope says how a nudge to a reading changes the activation. For ReLU it is a switch, on or off. For GELU it is a dial, and you compute it with two rules. The **product rule** says the slope of $a(x)\,s(x)$ is $a'(x)\,s(x) + a(x)\,s'(x)$. The chain rule handles the tanh and the cubic.

Write $\operatorname{gelu}(x) = x \cdot s(x)$ with $s(x) = \tfrac12(1 + t)$ and $t = \tanh(u)$. The slope of $\tanh$ is $1 - t^2$, and $u$ has slope $\sqrt{2/\pi}\,(1 + 3 \cdot 0.044715\, x^2)$. Then

$$
\operatorname{gelu}'(x) = \tfrac{1}{2}(1 + t) + \tfrac{1}{2}\, x\, (1 - t^2)\, \sqrt{2/\pi}\,\left(1 + 3 \cdot 0.044715\, x^2\right) .
$$

Read it aloud: "the switch itself, plus $x$ times how fast the switch is changing." At $x = 1$ the first term is the switch, $0.841192$. For the second, $1 - t^2 = 0.534352$ and the slope of $u$ is $0.797885 \times 1.134145 = 0.904917$, so the term is $0.5 \times 0.534352 \times 0.904917 = 0.241772$. The sum is $1.082964$.

Slightly above 1 may surprise you. Figure 2, right, shows the reason: past the dip the curve rises a little faster than a straight line to catch up with $x$, then settles back toward a slope of 1. Left of about $-0.75$ the slope is slightly negative, because the curve is dipping toward its minimum near $-0.17$, and far on the left it returns to 0, as it does for ReLU. ReLU never has a negative slope.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Left: ReLU in grey and GELU in orange on the range minus 4 to 4; GELU dips just below zero for negative inputs and matches ReLU far from zero. Right: ReLU's slope is a step from 0 to 1, while GELU's slope rises smoothly, passing 0.5 at zero and peaking slightly above 1 near x equal to 1.4.">

<line x1="50.0" y1="181.0" x2="290.0" y2="181.0" class="stroke-muted"/>
<text x="50.0" y="232.0" text-anchor="middle" class="fig-small">−4</text>
<text x="110.0" y="232.0" text-anchor="middle" class="fig-small">−2</text>
<text x="170.0" y="232.0" text-anchor="middle" class="fig-small">0</text>
<text x="230.0" y="232.0" text-anchor="middle" class="fig-small">2</text>
<text x="290.0" y="232.0" text-anchor="middle" class="fig-small">4</text>
<line x1="340.0" y1="190.7" x2="570.0" y2="190.7" class="stroke-muted"/>
<text x="340.0" y="206.7" text-anchor="middle" class="fig-small">−4</text>
<text x="397.5" y="206.7" text-anchor="middle" class="fig-small">−2</text>
<text x="455.0" y="206.7" text-anchor="middle" class="fig-small">0</text>
<text x="512.5" y="206.7" text-anchor="middle" class="fig-small">2</text>
<text x="570.0" y="206.7" text-anchor="middle" class="fig-small">4</text>
<text x="46.0" y="185.0" text-anchor="end" class="fig-small">0</text>
<text x="46.0" y="151.0" text-anchor="end" class="fig-small">1</text>
<text x="46.0" y="117.0" text-anchor="end" class="fig-small">2</text>
<text x="46.0" y="83.0" text-anchor="end" class="fig-small">3</text>
<text x="46.0" y="219.0" text-anchor="end" class="fig-small">−1</text>
<text x="336.0" y="194.7" text-anchor="end" class="fig-small">0</text>
<text x="336.0" y="73.3" text-anchor="end" class="fig-small">1</text>
<text x="170.0" y="22.0" text-anchor="middle" class="fig-bold">ReLU and GELU</text>
<text x="455.0" y="22.0" text-anchor="middle" class="fig-bold">their slopes</text>
<path d="M50.0,181.0 L50.6,181.0 L51.2,181.0 L51.8,181.0 L52.4,181.0 L53.0,181.0 L53.6,181.0 L54.2,181.0 L54.8,181.0 L55.4,181.0 L56.0,181.0 L56.6,181.0 L57.2,181.0 L57.8,181.0 L58.4,181.0 L59.0,181.0 L59.6,181.0 L60.2,181.0 L60.8,181.0 L61.4,181.0 L62.0,181.0 L62.6,181.0 L63.2,181.0 L63.8,181.0 L64.4,181.0 L65.0,181.0 L65.6,181.0 L66.2,181.0 L66.8,181.0 L67.4,181.0 L68.0,181.0 L68.6,181.0 L69.2,181.0 L69.8,181.0 L70.4,181.0 L71.0,181.0 L71.6,181.0 L72.2,181.0 L72.8,181.0 L73.4,181.0 L74.0,181.0 L74.6,181.0 L75.2,181.0 L75.8,181.0 L76.4,181.0 L77.0,181.0 L77.6,181.0 L78.2,181.0 L78.8,181.0 L79.4,181.0 L80.0,181.0 L80.6,181.0 L81.2,181.0 L81.8,181.0 L82.4,181.0 L83.0,181.0 L83.6,181.0 L84.2,181.0 L84.8,181.0 L85.4,181.0 L86.0,181.0 L86.6,181.0 L87.2,181.0 L87.8,181.0 L88.4,181.0 L89.0,181.0 L89.6,181.0 L90.2,181.0 L90.8,181.0 L91.4,181.0 L92.0,181.0 L92.6,181.0 L93.2,181.0 L93.8,181.0 L94.4,181.0 L95.0,181.0 L95.6,181.0 L96.2,181.0 L96.8,181.0 L97.4,181.0 L98.0,181.0 L98.6,181.0 L99.2,181.0 L99.8,181.0 L100.4,181.0 L101.0,181.0 L101.6,181.0 L102.2,181.0 L102.8,181.0 L103.4,181.0 L104.0,181.0 L104.6,181.0 L105.2,181.0 L105.8,181.0 L106.4,181.0 L107.0,181.0 L107.6,181.0 L108.2,181.0 L108.8,181.0 L109.4,181.0 L110.0,181.0 L110.6,181.0 L111.2,181.0 L111.8,181.0 L112.4,181.0 L113.0,181.0 L113.6,181.0 L114.2,181.0 L114.8,181.0 L115.4,181.0 L116.0,181.0 L116.6,181.0 L117.2,181.0 L117.8,181.0 L118.4,181.0 L119.0,181.0 L119.6,181.0 L120.2,181.0 L120.8,181.0 L121.4,181.0 L122.0,181.0 L122.6,181.0 L123.2,181.0 L123.8,181.0 L124.4,181.0 L125.0,181.0 L125.6,181.0 L126.2,181.0 L126.8,181.0 L127.4,181.0 L128.0,181.0 L128.6,181.0 L129.2,181.0 L129.8,181.0 L130.4,181.0 L131.0,181.0 L131.6,181.0 L132.2,181.0 L132.8,181.0 L133.4,181.0 L134.0,181.0 L134.6,181.0 L135.2,181.0 L135.8,181.0 L136.4,181.0 L137.0,181.0 L137.6,181.0 L138.2,181.0 L138.8,181.0 L139.4,181.0 L140.0,181.0 L140.6,181.0 L141.2,181.0 L141.8,181.0 L142.4,181.0 L143.0,181.0 L143.6,181.0 L144.2,181.0 L144.8,181.0 L145.4,181.0 L146.0,181.0 L146.6,181.0 L147.2,181.0 L147.8,181.0 L148.4,181.0 L149.0,181.0 L149.6,181.0 L150.2,181.0 L150.8,181.0 L151.4,181.0 L152.0,181.0 L152.6,181.0 L153.2,181.0 L153.8,181.0 L154.4,181.0 L155.0,181.0 L155.6,181.0 L156.2,181.0 L156.8,181.0 L157.4,181.0 L158.0,181.0 L158.6,181.0 L159.2,181.0 L159.8,181.0 L160.4,181.0 L161.0,181.0 L161.6,181.0 L162.2,181.0 L162.8,181.0 L163.4,181.0 L164.0,181.0 L164.6,181.0 L165.2,181.0 L165.8,181.0 L166.4,181.0 L167.0,181.0 L167.6,181.0 L168.2,181.0 L168.8,181.0 L169.4,181.0 L170.0,181.0 L170.6,180.3 L171.2,179.6 L171.8,179.0 L172.4,178.3 L173.0,177.6 L173.6,176.9 L174.2,176.2 L174.8,175.6 L175.4,174.9 L176.0,174.2 L176.6,173.5 L177.2,172.8 L177.8,172.2 L178.4,171.5 L179.0,170.8 L179.6,170.1 L180.2,169.4 L180.8,168.8 L181.4,168.1 L182.0,167.4 L182.6,166.7 L183.2,166.0 L183.8,165.4 L184.4,164.7 L185.0,164.0 L185.6,163.3 L186.2,162.6 L186.8,162.0 L187.4,161.3 L188.0,160.6 L188.6,159.9 L189.2,159.2 L189.8,158.6 L190.4,157.9 L191.0,157.2 L191.6,156.5 L192.2,155.8 L192.8,155.2 L193.4,154.5 L194.0,153.8 L194.6,153.1 L195.2,152.4 L195.8,151.8 L196.4,151.1 L197.0,150.4 L197.6,149.7 L198.2,149.0 L198.8,148.4 L199.4,147.7 L200.0,147.0 L200.6,146.3 L201.2,145.6 L201.8,145.0 L202.4,144.3 L203.0,143.6 L203.6,142.9 L204.2,142.2 L204.8,141.6 L205.4,140.9 L206.0,140.2 L206.6,139.5 L207.2,138.8 L207.8,138.2 L208.4,137.5 L209.0,136.8 L209.6,136.1 L210.2,135.4 L210.8,134.8 L211.4,134.1 L212.0,133.4 L212.6,132.7 L213.2,132.0 L213.8,131.4 L214.4,130.7 L215.0,130.0 L215.6,129.3 L216.2,128.6 L216.8,128.0 L217.4,127.3 L218.0,126.6 L218.6,125.9 L219.2,125.2 L219.8,124.6 L220.4,123.9 L221.0,123.2 L221.6,122.5 L222.2,121.8 L222.8,121.2 L223.4,120.5 L224.0,119.8 L224.6,119.1 L225.2,118.4 L225.8,117.8 L226.4,117.1 L227.0,116.4 L227.6,115.7 L228.2,115.0 L228.8,114.4 L229.4,113.7 L230.0,113.0 L230.6,112.3 L231.2,111.6 L231.8,111.0 L232.4,110.3 L233.0,109.6 L233.6,108.9 L234.2,108.2 L234.8,107.6 L235.4,106.9 L236.0,106.2 L236.6,105.5 L237.2,104.8 L237.8,104.2 L238.4,103.5 L239.0,102.8 L239.6,102.1 L240.2,101.4 L240.8,100.8 L241.4,100.1 L242.0,99.4 L242.6,98.7 L243.2,98.0 L243.8,97.4 L244.4,96.7 L245.0,96.0 L245.6,95.3 L246.2,94.6 L246.8,94.0 L247.4,93.3 L248.0,92.6 L248.6,91.9 L249.2,91.2 L249.8,90.6 L250.4,89.9 L251.0,89.2 L251.6,88.5 L252.2,87.8 L252.8,87.2 L253.4,86.5 L254.0,85.8 L254.6,85.1 L255.2,84.4 L255.8,83.8 L256.4,83.1 L257.0,82.4 L257.6,81.7 L258.2,81.0 L258.8,80.4 L259.4,79.7 L260.0,79.0 L260.6,78.3 L261.2,77.6 L261.8,77.0 L262.4,76.3 L263.0,75.6 L263.6,74.9 L264.2,74.2 L264.8,73.6 L265.4,72.9 L266.0,72.2 L266.6,71.5 L267.2,70.8 L267.8,70.2 L268.4,69.5 L269.0,68.8 L269.6,68.1 L270.2,67.4 L270.8,66.8 L271.4,66.1 L272.0,65.4 L272.6,64.7 L273.2,64.0 L273.8,63.4 L274.4,62.7 L275.0,62.0 L275.6,61.3 L276.2,60.6 L276.8,60.0 L277.4,59.3 L278.0,58.6 L278.6,57.9 L279.2,57.2 L279.8,56.6 L280.4,55.9 L281.0,55.2 L281.6,54.5 L282.2,53.8 L282.8,53.2 L283.4,52.5 L284.0,51.8 L284.6,51.1 L285.2,50.4 L285.8,49.8 L286.4,49.1 L287.0,48.4 L287.6,47.7 L288.2,47.0 L288.8,46.4 L289.4,45.7 L290.0,45.0" fill="none" class="stroke-muted" stroke-width="2"/>
<path d="M50.0,181.0 L50.6,181.0 L51.2,181.0 L51.8,181.0 L52.4,181.0 L53.0,181.0 L53.6,181.0 L54.2,181.0 L54.8,181.0 L55.4,181.0 L56.0,181.0 L56.6,181.0 L57.2,181.0 L57.8,181.0 L58.4,181.0 L59.0,181.0 L59.6,181.0 L60.2,181.0 L60.8,181.0 L61.4,181.0 L62.0,181.0 L62.6,181.0 L63.2,181.0 L63.8,181.0 L64.4,181.0 L65.0,181.0 L65.6,181.0 L66.2,181.0 L66.8,181.0 L67.4,181.0 L68.0,181.0 L68.6,181.0 L69.2,181.0 L69.8,181.0 L70.4,181.0 L71.0,181.0 L71.6,181.0 L72.2,181.1 L72.8,181.1 L73.4,181.1 L74.0,181.1 L74.6,181.1 L75.2,181.1 L75.8,181.1 L76.4,181.1 L77.0,181.1 L77.6,181.1 L78.2,181.1 L78.8,181.1 L79.4,181.1 L80.0,181.1 L80.6,181.1 L81.2,181.1 L81.8,181.1 L82.4,181.2 L83.0,181.2 L83.6,181.2 L84.2,181.2 L84.8,181.2 L85.4,181.2 L86.0,181.2 L86.6,181.2 L87.2,181.3 L87.8,181.3 L88.4,181.3 L89.0,181.3 L89.6,181.3 L90.2,181.3 L90.8,181.4 L91.4,181.4 L92.0,181.4 L92.6,181.4 L93.2,181.4 L93.8,181.5 L94.4,181.5 L95.0,181.5 L95.6,181.5 L96.2,181.6 L96.8,181.6 L97.4,181.6 L98.0,181.7 L98.6,181.7 L99.2,181.7 L99.8,181.8 L100.4,181.8 L101.0,181.8 L101.6,181.9 L102.2,181.9 L102.8,181.9 L103.4,182.0 L104.0,182.0 L104.6,182.1 L105.2,182.1 L105.8,182.2 L106.4,182.2 L107.0,182.3 L107.6,182.3 L108.2,182.4 L108.8,182.4 L109.4,182.5 L110.0,182.5 L110.6,182.6 L111.2,182.7 L111.8,182.7 L112.4,182.8 L113.0,182.9 L113.6,182.9 L114.2,183.0 L114.8,183.1 L115.4,183.1 L116.0,183.2 L116.6,183.3 L117.2,183.3 L117.8,183.4 L118.4,183.5 L119.0,183.6 L119.6,183.7 L120.2,183.7 L120.8,183.8 L121.4,183.9 L122.0,184.0 L122.6,184.1 L123.2,184.2 L123.8,184.2 L124.4,184.3 L125.0,184.4 L125.6,184.5 L126.2,184.6 L126.8,184.7 L127.4,184.8 L128.0,184.9 L128.6,184.9 L129.2,185.0 L129.8,185.1 L130.4,185.2 L131.0,185.3 L131.6,185.4 L132.2,185.5 L132.8,185.5 L133.4,185.6 L134.0,185.7 L134.6,185.8 L135.2,185.9 L135.8,185.9 L136.4,186.0 L137.0,186.1 L137.6,186.1 L138.2,186.2 L138.8,186.3 L139.4,186.3 L140.0,186.4 L140.6,186.5 L141.2,186.5 L141.8,186.6 L142.4,186.6 L143.0,186.6 L143.6,186.7 L144.2,186.7 L144.8,186.7 L145.4,186.7 L146.0,186.8 L146.6,186.8 L147.2,186.8 L147.8,186.8 L148.4,186.8 L149.0,186.8 L149.6,186.7 L150.2,186.7 L150.8,186.7 L151.4,186.6 L152.0,186.6 L152.6,186.5 L153.2,186.5 L153.8,186.4 L154.4,186.3 L155.0,186.2 L155.6,186.2 L156.2,186.0 L156.8,185.9 L157.4,185.8 L158.0,185.7 L158.6,185.5 L159.2,185.4 L159.8,185.2 L160.4,185.1 L161.0,184.9 L161.6,184.7 L162.2,184.5 L162.8,184.3 L163.4,184.1 L164.0,183.9 L164.6,183.6 L165.2,183.4 L165.8,183.1 L166.4,182.8 L167.0,182.6 L167.6,182.3 L168.2,182.0 L168.8,181.7 L169.4,181.3 L170.0,181.0 L170.6,180.7 L171.2,180.3 L171.8,179.9 L172.4,179.6 L173.0,179.2 L173.6,178.8 L174.2,178.4 L174.8,177.9 L175.4,177.5 L176.0,177.1 L176.6,176.6 L177.2,176.1 L177.8,175.7 L178.4,175.2 L179.0,174.7 L179.6,174.2 L180.2,173.7 L180.8,173.2 L181.4,172.6 L182.0,172.1 L182.6,171.5 L183.2,171.0 L183.8,170.4 L184.4,169.8 L185.0,169.2 L185.6,168.7 L186.2,168.0 L186.8,167.4 L187.4,166.8 L188.0,166.2 L188.6,165.6 L189.2,164.9 L189.8,164.3 L190.4,163.6 L191.0,163.0 L191.6,162.3 L192.2,161.6 L192.8,160.9 L193.4,160.3 L194.0,159.6 L194.6,158.9 L195.2,158.2 L195.8,157.5 L196.4,156.8 L197.0,156.0 L197.6,155.3 L198.2,154.6 L198.8,153.9 L199.4,153.1 L200.0,152.4 L200.6,151.7 L201.2,150.9 L201.8,150.2 L202.4,149.4 L203.0,148.7 L203.6,147.9 L204.2,147.2 L204.8,146.4 L205.4,145.7 L206.0,144.9 L206.6,144.1 L207.2,143.4 L207.8,142.6 L208.4,141.9 L209.0,141.1 L209.6,140.3 L210.2,139.6 L210.8,138.8 L211.4,138.0 L212.0,137.3 L212.6,136.5 L213.2,135.7 L213.8,134.9 L214.4,134.2 L215.0,133.4 L215.6,132.6 L216.2,131.9 L216.8,131.1 L217.4,130.4 L218.0,129.6 L218.6,128.8 L219.2,128.1 L219.8,127.3 L220.4,126.5 L221.0,125.8 L221.6,125.0 L222.2,124.3 L222.8,123.5 L223.4,122.8 L224.0,122.0 L224.6,121.2 L225.2,120.5 L225.8,119.7 L226.4,119.0 L227.0,118.3 L227.6,117.5 L228.2,116.8 L228.8,116.0 L229.4,115.3 L230.0,114.5 L230.6,113.8 L231.2,113.1 L231.8,112.3 L232.4,111.6 L233.0,110.9 L233.6,110.1 L234.2,109.4 L234.8,108.7 L235.4,108.0 L236.0,107.2 L236.6,106.5 L237.2,105.8 L237.8,105.1 L238.4,104.3 L239.0,103.6 L239.6,102.9 L240.2,102.2 L240.8,101.5 L241.4,100.8 L242.0,100.1 L242.6,99.3 L243.2,98.6 L243.8,97.9 L244.4,97.2 L245.0,96.5 L245.6,95.8 L246.2,95.1 L246.8,94.4 L247.4,93.7 L248.0,93.0 L248.6,92.3 L249.2,91.6 L249.8,90.9 L250.4,90.2 L251.0,89.5 L251.6,88.8 L252.2,88.1 L252.8,87.4 L253.4,86.7 L254.0,86.0 L254.6,85.3 L255.2,84.6 L255.8,84.0 L256.4,83.3 L257.0,82.6 L257.6,81.9 L258.2,81.2 L258.8,80.5 L259.4,79.8 L260.0,79.1 L260.6,78.4 L261.2,77.7 L261.8,77.1 L262.4,76.4 L263.0,75.7 L263.6,75.0 L264.2,74.3 L264.8,73.6 L265.4,72.9 L266.0,72.3 L266.6,71.6 L267.2,70.9 L267.8,70.2 L268.4,69.5 L269.0,68.8 L269.6,68.2 L270.2,67.5 L270.8,66.8 L271.4,66.1 L272.0,65.4 L272.6,64.7 L273.2,64.1 L273.8,63.4 L274.4,62.7 L275.0,62.0 L275.6,61.3 L276.2,60.7 L276.8,60.0 L277.4,59.3 L278.0,58.6 L278.6,57.9 L279.2,57.3 L279.8,56.6 L280.4,55.9 L281.0,55.2 L281.6,54.5 L282.2,53.8 L282.8,53.2 L283.4,52.5 L284.0,51.8 L284.6,51.1 L285.2,50.4 L285.8,49.8 L286.4,49.1 L287.0,48.4 L287.6,47.7 L288.2,47.0 L288.8,46.4 L289.4,45.7 L290.0,45.0" fill="none" class="stroke-b" stroke-width="2.2"/>
<circle cx="200.0" cy="152.4" r="4.5" class="fill-ink"/>
<text x="192.0" y="144.4" text-anchor="end" class="fig-small">x = 1: 0.841</text>
<circle cx="140.0" cy="186.4" r="4.5" class="fill-ink"/>
<text x="134.0" y="208.4" text-anchor="middle" class="fig-small">x = −1: −0.159</text>
<text x="86.0" y="62.6" text-anchor="start" class="fig-small">ReLU (grey)</text>
<text x="86.0" y="83.0" text-anchor="start" class="fig-small">GELU (orange)</text>
<path d="M340.0,190.7 L455.0,190.7" fill="none" class="stroke-muted" stroke-width="2"/>
<path d="M455.0,69.3 L570.0,69.3" fill="none" class="stroke-muted" stroke-width="2"/>
<path d="M340.0,190.8 L340.6,190.8 L341.1,190.8 L341.7,190.8 L342.3,190.8 L342.9,190.8 L343.4,190.8 L344.0,190.8 L344.6,190.8 L345.2,190.8 L345.8,190.8 L346.3,190.8 L346.9,190.8 L347.5,190.8 L348.1,190.8 L348.6,190.9 L349.2,190.9 L349.8,190.9 L350.4,190.9 L350.9,190.9 L351.5,190.9 L352.1,190.9 L352.6,191.0 L353.2,191.0 L353.8,191.0 L354.4,191.0 L354.9,191.0 L355.5,191.1 L356.1,191.1 L356.7,191.1 L357.2,191.1 L357.8,191.2 L358.4,191.2 L359.0,191.2 L359.6,191.3 L360.1,191.3 L360.7,191.3 L361.3,191.4 L361.9,191.4 L362.4,191.5 L363.0,191.5 L363.6,191.6 L364.1,191.6 L364.7,191.7 L365.3,191.7 L365.9,191.8 L366.4,191.8 L367.0,191.9 L367.6,192.0 L368.2,192.0 L368.8,192.1 L369.3,192.2 L369.9,192.3 L370.5,192.4 L371.1,192.5 L371.6,192.6 L372.2,192.6 L372.8,192.7 L373.4,192.9 L373.9,193.0 L374.5,193.1 L375.1,193.2 L375.6,193.3 L376.2,193.4 L376.8,193.6 L377.4,193.7 L377.9,193.8 L378.5,194.0 L379.1,194.1 L379.7,194.3 L380.2,194.5 L380.8,194.6 L381.4,194.8 L382.0,195.0 L382.6,195.1 L383.1,195.3 L383.7,195.5 L384.3,195.7 L384.9,195.9 L385.4,196.1 L386.0,196.3 L386.6,196.5 L387.1,196.7 L387.7,197.0 L388.3,197.2 L388.9,197.4 L389.4,197.6 L390.0,197.9 L390.6,198.1 L391.2,198.4 L391.8,198.6 L392.3,198.8 L392.9,199.1 L393.5,199.4 L394.1,199.6 L394.6,199.9 L395.2,200.1 L395.8,200.4 L396.4,200.6 L396.9,200.9 L397.5,201.2 L398.1,201.4 L398.6,201.7 L399.2,202.0 L399.8,202.2 L400.4,202.5 L400.9,202.7 L401.5,203.0 L402.1,203.2 L402.7,203.5 L403.2,203.7 L403.8,203.9 L404.4,204.2 L405.0,204.4 L405.6,204.6 L406.1,204.8 L406.7,205.0 L407.3,205.2 L407.9,205.3 L408.4,205.5 L409.0,205.7 L409.6,205.8 L410.1,205.9 L410.7,206.0 L411.3,206.1 L411.9,206.2 L412.4,206.3 L413.0,206.3 L413.6,206.4 L414.2,206.4 L414.8,206.4 L415.3,206.3 L415.9,206.3 L416.5,206.2 L417.1,206.1 L417.6,206.0 L418.2,205.9 L418.8,205.7 L419.4,205.5 L419.9,205.3 L420.5,205.0 L421.1,204.7 L421.6,204.4 L422.2,204.1 L422.8,203.7 L423.4,203.3 L423.9,202.9 L424.5,202.4 L425.1,201.9 L425.7,201.4 L426.2,200.8 L426.8,200.2 L427.4,199.5 L428.0,198.9 L428.6,198.2 L429.1,197.4 L429.7,196.6 L430.3,195.8 L430.9,194.9 L431.4,194.0 L432.0,193.1 L432.6,192.1 L433.1,191.1 L433.7,190.1 L434.3,189.0 L434.9,187.8 L435.4,186.7 L436.0,185.5 L436.6,184.2 L437.2,183.0 L437.8,181.7 L438.3,180.3 L438.9,178.9 L439.5,177.5 L440.1,176.1 L440.6,174.6 L441.2,173.1 L441.8,171.6 L442.4,170.0 L442.9,168.4 L443.5,166.8 L444.1,165.1 L444.6,163.4 L445.2,161.7 L445.8,160.0 L446.4,158.2 L446.9,156.4 L447.5,154.6 L448.1,152.8 L448.7,151.0 L449.2,149.1 L449.8,147.3 L450.4,145.4 L451.0,143.5 L451.6,141.6 L452.1,139.7 L452.7,137.7 L453.3,135.8 L453.9,133.9 L454.4,131.9 L455.0,130.0 L455.6,128.1 L456.1,126.1 L456.7,124.2 L457.3,122.3 L457.9,120.3 L458.4,118.4 L459.0,116.5 L459.6,114.6 L460.2,112.7 L460.8,110.9 L461.3,109.0 L461.9,107.2 L462.5,105.4 L463.1,103.6 L463.6,101.8 L464.2,100.0 L464.8,98.3 L465.4,96.6 L465.9,94.9 L466.5,93.2 L467.1,91.6 L467.6,90.0 L468.2,88.4 L468.8,86.9 L469.4,85.4 L470.0,83.9 L470.5,82.5 L471.1,81.1 L471.7,79.7 L472.2,78.3 L472.8,77.0 L473.4,75.8 L474.0,74.5 L474.5,73.3 L475.1,72.2 L475.7,71.0 L476.3,69.9 L476.9,68.9 L477.4,67.9 L478.0,66.9 L478.6,66.0 L479.1,65.1 L479.7,64.2 L480.3,63.4 L480.9,62.6 L481.4,61.8 L482.0,61.1 L482.6,60.5 L483.2,59.8 L483.8,59.2 L484.3,58.6 L484.9,58.1 L485.5,57.6 L486.1,57.1 L486.6,56.7 L487.2,56.3 L487.8,55.9 L488.4,55.6 L488.9,55.3 L489.5,55.0 L490.1,54.7 L490.6,54.5 L491.2,54.3 L491.8,54.1 L492.4,54.0 L493.0,53.9 L493.5,53.8 L494.1,53.7 L494.7,53.7 L495.2,53.6 L495.8,53.6 L496.4,53.6 L497.0,53.7 L497.6,53.7 L498.1,53.8 L498.7,53.9 L499.3,54.0 L499.9,54.1 L500.4,54.2 L501.0,54.3 L501.6,54.5 L502.1,54.7 L502.7,54.8 L503.3,55.0 L503.9,55.2 L504.4,55.4 L505.0,55.6 L505.6,55.8 L506.2,56.1 L506.8,56.3 L507.3,56.5 L507.9,56.8 L508.5,57.0 L509.0,57.3 L509.6,57.5 L510.2,57.8 L510.8,58.0 L511.4,58.3 L511.9,58.6 L512.5,58.8 L513.1,59.1 L513.6,59.4 L514.2,59.6 L514.8,59.9 L515.4,60.1 L516.0,60.4 L516.5,60.6 L517.1,60.9 L517.7,61.2 L518.2,61.4 L518.8,61.6 L519.4,61.9 L520.0,62.1 L520.5,62.4 L521.1,62.6 L521.7,62.8 L522.3,63.0 L522.9,63.3 L523.4,63.5 L524.0,63.7 L524.6,63.9 L525.1,64.1 L525.7,64.3 L526.3,64.5 L526.9,64.7 L527.5,64.9 L528.0,65.0 L528.6,65.2 L529.2,65.4 L529.8,65.5 L530.3,65.7 L530.9,65.9 L531.5,66.0 L532.0,66.2 L532.6,66.3 L533.2,66.4 L533.8,66.6 L534.4,66.7 L534.9,66.8 L535.5,66.9 L536.1,67.0 L536.6,67.1 L537.2,67.3 L537.8,67.4 L538.4,67.4 L539.0,67.5 L539.5,67.6 L540.1,67.7 L540.7,67.8 L541.2,67.9 L541.8,68.0 L542.4,68.0 L543.0,68.1 L543.5,68.2 L544.1,68.2 L544.7,68.3 L545.3,68.3 L545.9,68.4 L546.4,68.4 L547.0,68.5 L547.6,68.5 L548.1,68.6 L548.7,68.6 L549.3,68.7 L549.9,68.7 L550.5,68.7 L551.0,68.8 L551.6,68.8 L552.2,68.8 L552.8,68.9 L553.3,68.9 L553.9,68.9 L554.5,68.9 L555.0,69.0 L555.6,69.0 L556.2,69.0 L556.8,69.0 L557.4,69.0 L557.9,69.1 L558.5,69.1 L559.1,69.1 L559.6,69.1 L560.2,69.1 L560.8,69.1 L561.4,69.1 L562.0,69.2 L562.5,69.2 L563.1,69.2 L563.7,69.2 L564.2,69.2 L564.8,69.2 L565.4,69.2 L566.0,69.2 L566.5,69.2 L567.1,69.2 L567.7,69.2 L568.3,69.2 L568.9,69.2 L569.4,69.2 L570.0,69.2" fill="none" class="stroke-b" stroke-width="2.2"/>
<circle cx="483.8" cy="59.2" r="4.5" class="fill-ink"/>
<text x="491.8" y="45.2" text-anchor="middle" class="fig-small">x = 1: 1.083</text>
<text x="443.5" y="134.0" text-anchor="end" class="fig-small">0.5 at 0</text>
<text x="489.5" y="148.2" text-anchor="start" class="fig-small">grey: ReLU, a step</text>
<text x="489.5" y="166.4" text-anchor="start" class="fig-small">orange: GELU, a dial</text>
</svg>
Figure 2. Left: GELU follows ReLU far from 0 and rounds the corner, with a small dip below 0 for negative inputs. Right: ReLU's slope is a switch, 0 then 1, while GELU's slope is a dial that reaches 0.5 at x = 0 and overshoots 1 (peak about 1.13 near x = 1.4) before settling back toward 1.
```

> [!KEY]
> A network is one move repeated: form each unit's pre-activation as a weighted sum, then bend it into an activation, and feed the activations to the next layer. The bend can be sharp (ReLU) or smooth (GELU), and in both cases it has a slope you can compute and check.

> [!RECAP]
> - A forward pass turns the input into pre-activations, then activations, one layer at a time.
> - For row data a layer is `h @ W.T + b` with `W` shaped `(out, in)`, and the usual mistake is dropping the transpose.
> - GELU is `0.5 * x * (1 + tanh(u))` with `u = sqrt(2/pi) * (x + 0.044715 * x**3)`.
> - Its slope has two terms, both built from one `t = tanh(u)`.
> - An activation name picks the bend: `"relu"`, `"gelu"`, or `"identity"` for none.

## Build it step by step

`relu` and `relu_grad` from the last chapter are already in the starter. Build three functions.

1. **`gelu(x)`** takes an array of any shape and returns the tanh version, in the same shape. Compute `u` once and reuse it. A common mistake is dropping the cubic term, which gives numbers that are close but fail the check at 1. Write it like this:

    ```python
    def gelu(x):
        # 0.5 * x * (1 + tanh(u)), with u from x and x**3.
    ```

2. **`gelu_grad(x)`** returns the slope, in the same shape. Compute `t = np.tanh(u)` once and use it in both terms. The checks compare against finite differences. A common mistake is forgetting the `3 *` in the slope of `u`. Write it like this:

    ```python
    def gelu_grad(x):
        # 0.5 * (1 + t) plus x times the switch's slope.
    ```

3. **`mlp_forward(params, x)`** takes a list of layers and a vector of shape `(in,)`, and returns the output vector. Each layer is a dict with keys `"W"`, `"b"`, and `"activation"`. Loop over the layers: form `z`, then pick the bend by name. `"identity"` returns `z` unchanged, and an unknown name raises `ValueError`. Write it like this:

    ```python
    def mlp_forward(params, x):
        # For each layer: z = h @ W.T + b, then the named bend.
    ```

## Where this breaks

GELU costs more than ReLU: the tanh and the cubic take several operations per entry, and a later chapter counts them. A hand-written forward pass on a single vector is also slow for large data, so real code runs a whole batch of rows through one matrix product at a time.

## The exercise

You implement `gelu`, `gelu_grad`, and `mlp_forward`. The checks compare `gelu` with the values at $-1$, $0$, and $1$, and with its limits at $\pm 8$. They compare `gelu_grad` with finite differences, and with $1.082964$ at $x = 1$. They run `mlp_forward` on the hand example, which must return $[2.5]$, on a network with a GELU layer and rectangular weights, and on an unknown activation name.
