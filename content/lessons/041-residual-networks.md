---
title: "Residual networks: the shortcut, the gradient, and the bottleneck"
slug: 041-residual-networks
part: "Part 5 — Vision and sequences"
summary: A block that adds a correction to its input trains at depth, keeps a fifty-layer gradient alive, and stays cheap with a bottleneck.
minutes: 75
gpu: false
objectives:
  - Explain why a 56-layer plain network can train worse than a 20-layer one, and how adding the input back makes "do nothing" the default.
  - Compute a residual block's forward and backward passes by hand, and verify them against finite differences.
  - Explain why a plus sign sends the gradient down both paths, and why the ReLU mask acts as a switch on the branch only.
  - Measure how much gradient survives fifty plain and fifty residual layers, and say why the scalar picture overshoots.
  - Count the multiply-adds of a convolution and of a bottleneck block, and show the bottleneck is about 17 times cheaper.
prereqs:
  - 029-multilayer-perceptrons
  - 030-backpropagation
  - 031-softmax-and-whole-network-gradients
  - 033-signal-size-and-initialisation
  - 039-convolution-and-receptive-field
  - 040-im2col-and-convolution-backward
exercise: 041-residual-networks
---

# Residual networks: the shortcut, the gradient, and the bottleneck

You train a network with 20 layers and record its error on the training set. Then you stack 36 more layers on top, keep the data and the optimiser the same, and train again. The 56-layer network ends with a higher training error than the 20-layer one. This result, measured on the CIFAR-10 image set, led to residual networks.

It is not overfitting, because the training error itself is worse. It is also strange, because the deep network could copy the 20 trained layers and let the 36 extra layers pass their input along unchanged. A solution at least as good exists, and gradient descent fails to find it.

This chapter has three parts. One change makes "pass the input along unchanged" the easy default: add the block's input back onto its output. You build that block forward and backward, measure how much gradient survives fifty layers with and without it, and count what depth costs, which a bottleneck keeps affordable.

## What you already know

- A linear layer turns each row of `x` into `x @ W.T + b`, and a plain layer with small random weights shrinks its signal toward zero ([the multilayer perceptron](/lessons/029-multilayer-perceptrons), [signal size](/lessons/033-signal-size-and-initialisation)).
- The chain rule multiplies slopes along a path, and a number that reaches the loss by several paths collects the slope of each ([backpropagation](/lessons/030-backpropagation)).
- A gradient check compares a computed gradient with a finite-difference estimate ([whole-network gradients](/lessons/031-softmax-and-whole-network-gradients)).
- A convolution slides a small filter over a picture, and one filter produces one output channel ([convolution](/lessons/039-convolution-and-receptive-field)).

## Why an ordinary layer struggles to do nothing

Picture a chain of 36 people, each handed a page and asked to hand it on. In a plain network each person redraws the page from memory, so "do nothing" needs every person to be exact. A plain block's output comes only from its weights, so handing the input along needs one precise setting of every weight. A block with a ReLU in the middle has no identity matrix at all, because the ReLU wipes out negative entries, and small random weights output numbers close to zero, the opposite of the input.

Change the picture. Hand the page along untouched, and let each person attach a sticky note with a correction. If nobody writes a note, the page arrives as it started, and writing no note is easy, because zero is where small weights already sit. Figure 1 draws both versions. The page is $x$, the sticky note is $F(x)$, and the result is $y$.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="A plain block with three steps, above a residual block whose input also travels around those steps to a plus sign">
<defs><marker id="c036-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<defs><marker id="c036-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker></defs>
<text x="20" y="22" class="fig-bold">Plain block</text>
<text x="22" y="65" class="fig-mono">x</text>
<line x1="34.0" y1="60.0" x2="68.0" y2="60.0" class="stroke-ink" marker-end="url(#c036-arrow)"/>
<rect x="70.0" y="40.0" width="100.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/><text x="120.0" y="64.5" text-anchor="middle">linear</text>
<line x1="170.0" y1="60.0" x2="188.0" y2="60.0" class="stroke-ink" marker-end="url(#c036-arrow)"/>
<rect x="190.0" y="40.0" width="100.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/><text x="240.0" y="64.5" text-anchor="middle">ReLU</text>
<line x1="290.0" y1="60.0" x2="308.0" y2="60.0" class="stroke-ink" marker-end="url(#c036-arrow)"/>
<rect x="310.0" y="40.0" width="100.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/><text x="360.0" y="64.5" text-anchor="middle">linear</text>
<line x1="410.0" y1="60.0" x2="500.0" y2="60.0" class="stroke-ink" marker-end="url(#c036-arrow)"/>
<text x="508" y="65" class="fig-mono">y = F(x)</text>
<text x="20" y="108" class="fig-small">Zero weights give y = 0: the input is gone.</text>
<text x="20" y="150" class="fig-bold">Residual block</text>
<text x="22" y="213" class="fig-mono">x</text>
<line x1="34" y1="208" x2="68" y2="208" class="stroke-ink" marker-end="url(#c036-arrow)"/>
<rect x="70.0" y="188.0" width="100.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/><text x="120.0" y="212.5" text-anchor="middle">linear</text>
<line x1="170.0" y1="208.0" x2="188.0" y2="208.0" class="stroke-ink" marker-end="url(#c036-arrow)"/>
<rect x="190.0" y="188.0" width="100.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/><text x="240.0" y="212.5" text-anchor="middle">ReLU</text>
<line x1="290.0" y1="208.0" x2="308.0" y2="208.0" class="stroke-ink" marker-end="url(#c036-arrow)"/>
<rect x="310.0" y="188.0" width="100.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/><text x="360.0" y="212.5" text-anchor="middle">linear</text>
<line x1="410.0" y1="208.0" x2="474.0" y2="208.0" class="stroke-ink" marker-end="url(#c036-arrow)"/>
<circle cx="490" cy="208" r="15" class="fill-paper stroke-ink"/>
<text x="490" y="214" text-anchor="middle" class="fig-bold">+</text>
<line x1="505.0" y1="208.0" x2="556.0" y2="208.0" class="stroke-ink" marker-end="url(#c036-arrow)"/>
<text x="560" y="213" class="fig-mono">y</text>
<path d="M48,208 L48,162 L490,162 L490,192" fill="none" class="stroke-a" marker-end="url(#c036-arrow-a)"/>
<text x="269" y="156" text-anchor="middle" class="fig-small">shortcut: x travels unchanged</text>
<text x="240" y="250" text-anchor="middle" class="fig-small">branch F(x): the correction</text>
<text x="20" y="286" class="fig-small">Zero weights in the last layer give F(x) = 0, so y = x. No ReLU after the plus sign.</text>
</svg>
Figure 1. The plain block (top) has one path, so a zero last layer erases the input. The residual block (bottom) adds a second path, the shortcut in blue, that carries x around the three steps to the plus sign. Notice that nothing sits between the plus sign and y.
```

The block in the lower half is a **residual block**. Its output is the input plus a correction:

$$
y = x + F(x)
$$

Read it aloud: the block's output $y$ is its input $x$ plus whatever $F$ computes from $x$. $F$ is the **residual branch**, the layers that write the sticky note. The arc that carries $x$ around them to the plus sign is the **shortcut**.

### Work one block forward

Take two examples with two features each, one example per row:

$$
x = \begin{bmatrix} 1 & 2 \\ -1 & 3 \end{bmatrix}
$$

This pair is the running example for the chapter. The branch has two linear layers with a ReLU between them. The first has $W_1$ equal to the 2 by 2 identity and $b_1 = [0, -1]$. The second has $W_2 = \begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}$ and $b_2 = [0.1, 0]$. The steps are:

1. The first layer gives $h_1 = x W_1^{\top} + b_1 = \begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix}$, because the identity leaves $x$ alone and $b_1$ subtracts 1 from the second column.
2. The ReLU turns the $-1$ into 0, so $a_1 = \begin{bmatrix} 1 & 1 \\ 0 & 2 \end{bmatrix}$.
3. The second layer gives $h_2 = a_1 W_2^{\top} + b_2 = \begin{bmatrix} 2.1 & 0.5 \\ 0.1 & 1.0 \end{bmatrix}$. This is $F(x)$, the correction.
4. The addition gives $y = x + h_2 = \begin{bmatrix} 3.1 & 2.5 \\ -0.9 & 4.0 \end{bmatrix}$.

Example 2 keeps its negative entry, $-0.9$, because the block applies no ReLU after the plus sign. A ReLU there would clip negative entries, so even a zero correction would no longer return $x$.

Zero the last layer, with $W_2 = 0$ and $b_2 = 0$. Then $h_2$ is all zeros, and $y = x$ exactly, while a plain block with the same zeros returns $F(x) = 0$ and loses the input. With small random weights (standard deviation 0.05, width 8) the residual block moves no input entry by more than 0.06, while the plain block's output is near zero. In numpy:

```python
h1 = x @ W1.T + b1
a1 = np.maximum(h1, 0.0)
h2 = a1 @ W2.T + b2
y = x + h2
```

Replace each layer of the 56-layer network with a residual block whose last layer starts at zero. Every one of the 36 extra blocks starts as the identity, so training only has to find corrections that help. An identity start is a good starting point, not a guarantee, because training moves the correction away from zero.

> [!KEY]
> The deep network failed because "do nothing" is a precise setting for a plain block, and small random weights miss it. A residual block turns "do nothing" into a zero correction, which small weights already land near.

## A plus sign splits the gradient in two

Now the backward pass. The loss reports, for every number in the output, how much it would like that number to change: the gradient `dY`. Training needs how each weight and bias should change, and what message goes to the layer below. If you follow only the branch, you lose the shortcut's contribution, and that message is wrong.

Start with one number. Let $x = 2$ and the branch be $f(x) = 0.5 x^2$, so $y = x + f(x) = 4$. Nudge $x$ to 2.001 and $y$ becomes about 4.003, three times the nudge. The slope is $1 + f'(x) = 1 + x = 3$: one part comes through the shortcut, and two through the branch. As a formula:

$$
\frac{dy}{dx} = 1 + f'(x)
$$

Read it aloud: the output's slope with respect to the input is 1, from the shortcut, plus the branch's own slope. When the message $dY$ arrives, each path multiplies it by that path's slope. The shortcut multiplies by 1, so it returns $dY$ unchanged, and the branch multiplies by its own slopes. Figure 2 shows this on the running example, with $dY$ all ones, meaning "raise every output number a little".

```figure
<svg viewBox="0 0 620 360" role="img" aria-label="Gradient flowing backward through a residual block: it arrives at the plus sign, splits into a shortcut path that carries it unchanged and a branch path through W2, the ReLU mask, and W1, then adds back together at the input">
<defs><marker id="c036-back-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-d"/></marker></defs>
<circle cx="530" cy="150" r="15" class="fill-paper stroke-ink"/><text x="530" y="156" text-anchor="middle" class="fig-bold">+</text>
<circle cx="90" cy="150" r="15" class="fill-paper stroke-ink"/><text x="90" y="156" text-anchor="middle" class="fig-bold">+</text>
<text x="595" y="140" text-anchor="middle" class="fig-bold">dY</text>
<line x1="610" y1="150" x2="546" y2="150" class="stroke-d" marker-end="url(#c036-back-arrow)"/>
<path d="M530,134 L530,70 L90,70 L90,133" fill="none" class="stroke-d" stroke-width="2.5" marker-end="url(#c036-back-arrow)"/>
<text x="310" y="60" text-anchor="middle" class="fig-bold">shortcut: dY passes unchanged</text>
<text x="310" y="86" text-anchor="middle" class="fig-small fig-mono">[[1, 1], [1, 1]]</text>
<line x1="530" y1="166" x2="530" y2="230" class="stroke-d"/>
<line x1="530" y1="230" x2="516" y2="230" class="stroke-d" marker-end="url(#c036-back-arrow)"/>
<rect x="430.0" y="211.0" width="84.0" height="38.0" rx="6" class="fill-soft-b stroke-b"/><text x="472.0" y="234.5" text-anchor="middle">× W2</text>
<line x1="430" y1="230" x2="396" y2="230" class="stroke-d" marker-end="url(#c036-back-arrow)"/>
<rect x="310.0" y="211.0" width="84.0" height="38.0" rx="6" class="fill-soft-b stroke-b"/><text x="352.0" y="234.5" text-anchor="middle">× mask</text>
<line x1="310" y1="230" x2="276" y2="230" class="stroke-d" marker-end="url(#c036-back-arrow)"/>
<rect x="190.0" y="211.0" width="84.0" height="38.0" rx="6" class="fill-soft-b stroke-b"/><text x="232.0" y="234.5" text-anchor="middle">× W1</text>
<path d="M190,230 L90,230 L90,167" fill="none" class="stroke-d" marker-end="url(#c036-back-arrow)"/>
<text x="522" y="264" text-anchor="middle" class="fig-small">dh2 = dY</text>
<text x="412" y="264" text-anchor="middle" class="fig-small">da1</text>
<text x="412.0" y="280" text-anchor="middle" class="fig-mono fig-small">2  0.5</text>
<text x="412.0" y="295" text-anchor="middle" class="fig-mono fig-small">2  0.5</text>
<text x="292" y="264" text-anchor="middle" class="fig-small">dh1</text>
<text x="292.0" y="280" text-anchor="middle" class="fig-mono fig-small">2  0.5</text>
<text x="292.0" y="295" text-anchor="middle" class="fig-mono fig-small">0  0.5</text>
<text x="150" y="264" text-anchor="middle" class="fig-small">branch term</text>
<text x="150.0" y="280" text-anchor="middle" class="fig-mono fig-small">2  0.5</text>
<text x="150.0" y="295" text-anchor="middle" class="fig-mono fig-small">0  0.5</text>
<text x="120" y="146" class="fig-bold">dx = dY + branch term</text>
<text x="120" y="166" text-anchor="start" class="fig-mono fig-small">3  1.5</text>
<text x="120" y="181" text-anchor="start" class="fig-mono fig-small">1  1.5</text>
</svg>
Figure 2. The backward pass of the running example, with dY all ones. The gradient splits at the plus sign. The shortcut carries it to the input unchanged, and the branch carries it through W2, the mask, and W1 as a smaller term. The two meet at the input and add to dx.
```

### Walk the branch backward

The plus sign gives the branch the whole message, so $dh_2 = dY$. The entry $W_2[i, j]$ multiplies $a_1[n, j]$ on its way into $h_2[n, i]$ for every example $n$, so its gradient adds up $dh_2[n, i] \cdot a_1[n, j]$ over the examples. That is one entry of a matrix product, so $dW_2 = dh_2^{\top} a_1 = \begin{bmatrix} 1 & 3 \\ 1 & 3 \end{bmatrix}$. The entry 3 is $1 \cdot 1 + 1 \cdot 2$, the second column of $a_1$ summed over both examples. The bias is added to every row, so $db_2 = [2, 2]$. The message passes down through the weights, so $da_1 = dh_2 W_2 = \begin{bmatrix} 2 & 0.5 \\ 2 & 0.5 \end{bmatrix}$.

The ReLU is a switch. Its slope is 1 where the unit is on and 0 where it is off, because nudging an off unit's input changes nothing at its output. The mask $[h_1 > 0]$ holds a 1 for each unit that is on:

$$
dh_1 = da_1 \odot [h_1 > 0]
$$

The symbol $\odot$ multiplies entry by entry. The one negative $h_1$ is in example 2, feature 1, so $dh_1 = \begin{bmatrix} 2 & 0.5 \\ 0 & 0.5 \end{bmatrix}$. The first layer repeats the same two moves: $dW_1 = dh_1^{\top} x = \begin{bmatrix} 2 & 4 \\ 0 & 2.5 \end{bmatrix}$ and $db_1 = [2, 1]$. Its message to the input is $dh_1 W_1 = dh_1$, because $W_1$ is the identity. Last, the two paths meet at the input:

$$
dx = dY + dh_1 W_1
$$

On the example, $dx = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} + \begin{bmatrix} 2 & 0.5 \\ 0 & 0.5 \end{bmatrix} = \begin{bmatrix} 3 & 1.5 \\ 1 & 1.5 \end{bmatrix}$. The ones from the shortcut sit inside every entry of $dx$, and no switch can turn them off. As numpy:

```python
dW2 = dY.T @ a1
da1 = dY @ W2
dh1 = da1 * (h1 > 0)
dx = dY + dh1 @ W1
```

> [!WARNING]
> The forward pass computes `x @ W.T`, but the backward pass computes `dY @ W`, with no transpose. With square weights both run, and only one is right, so reason from the forward pass.

To check this, pick the loss $L = \sum (dY \odot y)$, whose gradient with respect to $y$ is exactly `dY`. Nudge each entry of $x$ and of every parameter up and down, divide the change in $L$ by twice the nudge, and compare with your backward pass. Keep $h_1$ away from zero, because the ReLU has a corner there.

## How much gradient survives fifty layers

Take a network with 50 layers. How much of the gradient arrives at the first one? If almost none, that layer never learns: at $10^{-50}$ of its starting size, the update is smaller than the smallest positive 32-bit float, about $10^{-45}$, so it rounds to exactly zero.

Think of the chain of people again, passing along the gradient. In a plain line each person repeats a tenth of the message, so after fifty people one part in $10^{50}$ is left. In a shortcut line each person repeats the whole message and adds a note worth a tenth. A plain layer with slope $J$ multiplies the gradient by $J$, and a residual layer has slope $1 + J$, the plus-sign rule from the last section. Through fifty layers the factors multiply:

$$
\text{plain: } J^{50}, \qquad \text{residual: } (1 + J)^{50}
$$

With $J = 0.1$, the plain product is $10^{-50}$ and the residual product is $1.1^{50}$, about 117. A gradient that grows by 117 is not healthy either, so something in this picture is too simple.

### Why the notes do not all point the same way

The scalar picture assumes each note points along the message. A real gradient is a vector, and a weight matrix turns it as well as scaling it. To isolate the turning, give every layer 0.1 times an **orthogonal matrix**, which rotates or flips a vector and never changes its length. A plain layer then shortens the gradient by exactly 0.1 every time, matching the scalar picture.

A residual layer computes $g + gW$ on the way back. The note $gW$ is 0.1 times as long as $g$ but points at some angle $\theta$ to it. The law of cosines gives the length after one layer:

$$
|g + gW|^2 = |g|^2 \left( 1 + 2 \cdot 0.1 \cos\theta + 0.1^2 \right)
$$

When the note points along $g$, $\cos\theta = 1$ and the factor is $1.1^2$, the scalar picture. At a right angle, $\cos\theta = 0$ and the length becomes $\sqrt{1.01} \approx 1.005$ times as long.

In 16 numbers a random direction typically has a cosine of about $\pm 0.25$ with $g$, one over the square root of 16. A note leaning toward $g$ gives a factor of about 1.03, and one leaning away gives about 0.98, so the length wanders a few percent above and below 1 and the changes largely cancel. Averaged over random directions, the cosine term cancels and the squared length grows by only $0.1^2 = 0.01$ per layer, so a typical length over 50 layers is about $1.01^{25} \approx 1.28$. That is the scale of the measured values below, not 117.

### Fifty layers, measured

The exercise builds a stack of 16-number layers, each 0.1 times an orthogonal matrix, starts from a gradient of length 1, and walks the layers in reverse with `g = g @ W` (plain) or `g = g + g @ W` (residual). Figure 3 plots the length that arrives for every depth from 0 to 50.

```figure
<svg viewBox="0 0 640 350" role="img" aria-label="Log-scale plot of gradient length against depth from 0 to 50: the plain stack falls in a straight line to 1e-50, the scalar shortcut picture rises to 117, and the simulated matrix shortcut stack stays close to 1">
<line x1="70" y1="300" x2="430" y2="300" class="stroke-muted"/><line x1="70" y1="300" x2="70" y2="45.0" class="stroke-muted"/>
<line x1="70" y1="300.0" x2="430" y2="300.0" class="stroke-grid"/><text x="62" y="304.0" text-anchor="end" class="fig-small">1e-50</text>
<line x1="70" y1="251.9" x2="430" y2="251.9" class="stroke-grid"/><text x="62" y="255.9" text-anchor="end" class="fig-small">1e-40</text>
<line x1="70" y1="203.8" x2="430" y2="203.8" class="stroke-grid"/><text x="62" y="207.8" text-anchor="end" class="fig-small">1e-30</text>
<line x1="70" y1="155.7" x2="430" y2="155.7" class="stroke-grid"/><text x="62" y="159.7" text-anchor="end" class="fig-small">1e-20</text>
<line x1="70" y1="107.5" x2="430" y2="107.5" class="stroke-grid"/><text x="62" y="111.5" text-anchor="end" class="fig-small">1e-10</text>
<line x1="70" y1="59.4" x2="430" y2="59.4" class="stroke-grid"/><text x="62" y="63.4" text-anchor="end" class="fig-small">1e0</text>
<line x1="70.0" y1="300" x2="70.0" y2="305" class="stroke-muted"/><text x="70.0" y="320" text-anchor="middle" class="fig-small">0</text>
<line x1="142.0" y1="300" x2="142.0" y2="305" class="stroke-muted"/><text x="142.0" y="320" text-anchor="middle" class="fig-small">10</text>
<line x1="214.0" y1="300" x2="214.0" y2="305" class="stroke-muted"/><text x="214.0" y="320" text-anchor="middle" class="fig-small">20</text>
<line x1="286.0" y1="300" x2="286.0" y2="305" class="stroke-muted"/><text x="286.0" y="320" text-anchor="middle" class="fig-small">30</text>
<line x1="358.0" y1="300" x2="358.0" y2="305" class="stroke-muted"/><text x="358.0" y="320" text-anchor="middle" class="fig-small">40</text>
<line x1="430.0" y1="300" x2="430.0" y2="305" class="stroke-muted"/><text x="430.0" y="320" text-anchor="middle" class="fig-small">50</text>
<text x="250.0" y="342" text-anchor="middle" class="fig-small">depth (number of layers)</text>
<text x="70" y="33" class="fig-small">gradient length at the input (log scale)</text>
<polyline points="70.0,59.4 77.2,64.2 84.4,69.1 91.6,73.9 98.8,78.7 106.0,83.5 113.2,88.3 120.4,93.1 127.6,97.9 134.8,102.7 142.0,107.5 149.2,112.4 156.4,117.2 163.6,122.0 170.8,126.8 178.0,131.6 185.2,136.4 192.4,141.2 199.6,146.0 206.8,150.8 214.0,155.7 221.2,160.5 228.4,165.3 235.6,170.1 242.8,174.9 250.0,179.7 257.2,184.5 264.4,189.3 271.6,194.2 278.8,199.0 286.0,203.8 293.2,208.6 300.4,213.4 307.6,218.2 314.8,223.0 322.0,227.8 329.2,232.6 336.4,237.5 343.6,242.3 350.8,247.1 358.0,251.9 365.2,256.7 372.4,261.5 379.6,266.3 386.8,271.1 394.0,275.9 401.2,280.8 408.4,285.6 415.6,290.4 422.8,295.2 430.0,300.0" fill="none" class="stroke-b" stroke-width="2.5"/>
<polyline points="70.0,59.4 77.2,59.2 84.4,59.0 91.6,58.8 98.8,58.6 106.0,58.4 113.2,58.2 120.4,58.0 127.6,57.8 134.8,57.6 142.0,57.4 149.2,57.2 156.4,57.0 163.6,56.8 170.8,56.6 178.0,56.4 185.2,56.2 192.4,56.0 199.6,55.8 206.8,55.7 214.0,55.5 221.2,55.3 228.4,55.1 235.6,54.9 242.8,54.7 250.0,54.5 257.2,54.3 264.4,54.1 271.6,53.9 278.8,53.7 286.0,53.5 293.2,53.3 300.4,53.1 307.6,52.9 314.8,52.7 322.0,52.5 329.2,52.3 336.4,52.1 343.6,51.9 350.8,51.7 358.0,51.5 365.2,51.3 372.4,51.1 379.6,50.9 386.8,50.7 394.0,50.5 401.2,50.3 408.4,50.1 415.6,49.9 422.8,49.7 430.0,49.5" fill="none" class="stroke-muted dashed" stroke-width="2"/>
<polyline points="70.0,59.4 77.2,59.4 84.4,59.4 91.6,59.4 98.8,59.5 106.0,59.4 113.2,59.5 120.4,59.5 127.6,59.6 134.8,59.6 142.0,59.6 149.2,59.7 156.4,59.6 163.6,59.6 170.8,59.6 178.0,59.7 185.2,59.6 192.4,59.5 199.6,59.4 206.8,59.3 214.0,59.3 221.2,59.4 228.4,59.4 235.6,59.5 242.8,59.5 250.0,59.4 257.2,59.3 264.4,59.3 271.6,59.2 278.8,59.2 286.0,59.2 293.2,59.2 300.4,59.1 307.6,59.2 314.8,59.1 322.0,59.1 329.2,59.1 336.4,59.0 343.6,59.0 350.8,59.0 358.0,59.0 365.2,59.1 372.4,59.1 379.6,59.2 386.8,59.1 394.0,59.2 401.2,59.3 408.4,59.2 415.6,59.1 422.8,59.1 430.0,59.1" fill="none" class="stroke-a" stroke-width="2.5"/>
<circle cx="142.0" cy="107.5" r="4.5" class="fill-b"/>
<circle cx="142.0" cy="59.6" r="4.5" class="fill-a"/>
<circle cx="430.0" cy="300.0" r="4.5" class="fill-b"/>
<circle cx="430.0" cy="59.1" r="4.5" class="fill-a"/>
<text x="444" y="27" class="fig-small">1.1^50 = 117: the scalar</text>
<text x="444" y="41" class="fig-small">picture (dashed)</text>
<text x="444" y="81" class="fig-small fig-bold">shortcut, simulated</text>
<text x="444" y="97" class="fig-small">depth 10: 0.90</text>
<text x="444" y="113" class="fig-small">depth 50: 1.15</text>
<text x="444" y="266" class="fig-small fig-bold">plain</text>
<text x="444" y="282" class="fig-small">depth 10: 1e-10</text>
<text x="444" y="298" class="fig-small">depth 50: 1e-50</text>
</svg>
Figure 3. Gradient length at the input against depth, on a log axis, each depth a separate stack with width 16 and weight scale 0.1 (seed 0). The plain stack is a straight line down, exactly 0.1 per layer. The scalar picture (dashed) would grow to 117. The simulated shortcut stack hovers near 1, and its wiggles are too small to see on this axis.
```

The numbers at two depths are:

| Depth | Residual length | Plain length |
|---|---|---|
| 10 | 0.90 | $10^{-10}$ |
| 50 | 1.15 | $10^{-50}$ |

The plain column is exact. The residual column depends on the random matrices: for 300 seeds at depth 50, nine runs in ten land between 0.9 and 1.7, with a median of 1.22, so read 1.15 as one typical value and not as a law. What does not depend on the seed is the plain stack's collapse and the shortcut stack's length staying within a small factor of 1.

> [!KEY]
> A plain layer multiplies the gradient by its weights, so fifty small factors multiply to nothing. A residual layer adds the gradient back to itself, so the length stays near one. The shortcut is not better because the gradient grows. It is better because the gradient is not forced to shrink.

## What depth costs, and how a bottleneck cuts it

A block of two 3×3 convolutions on 256 channels, applied to a 14×14 picture, takes 231,211,008 multiply-adds, so a hundred such blocks take over 23 billion for one picture's forward pass. Fewer channels would be cheaper, but the channels are how a layer holds what it has seen at each position. The question is how to keep the wide path between blocks and make each block cheap.

A **bottleneck** block packs, works, and unpacks. A cheap 1×1 convolution shrinks 256 channels to 64, the 3×3 convolution runs on those 64, and a 1×1 grows back to 256. Figure 4 draws the widths to scale. The shortcut skips all three steps at full width, so the plus sign still adds arrays of the same shape.

```figure
<svg viewBox="0 0 640 340" role="img" aria-label="Channel widths through a bottleneck block: 256 channels in, 64 after a one-by-one convolution, 64 after a three-by-three convolution, 256 after a one-by-one convolution, with the shortcut passing over the top at full width">
<defs><marker id="c036-f4-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<defs><marker id="c036-f4-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker></defs>
<rect x="60" y="119.6" width="44" height="140.8" class="fill-soft-a stroke-a"/>
<text x="82.0" y="282" text-anchor="middle" class="fig-bold">256</text>
<text x="82.0" y="300" text-anchor="middle" class="fig-small">input</text>
<rect x="200" y="172.4" width="44" height="35.2" class="fill-soft-a stroke-a"/>
<text x="222.0" y="282" text-anchor="middle" class="fig-bold">64</text>
<text x="222.0" y="300" text-anchor="middle" class="fig-small">after 1×1</text>
<rect x="340" y="172.4" width="44" height="35.2" class="fill-soft-a stroke-a"/>
<text x="362.0" y="282" text-anchor="middle" class="fig-bold">64</text>
<text x="362.0" y="300" text-anchor="middle" class="fig-small">after 3×3</text>
<rect x="480" y="119.6" width="44" height="140.8" class="fill-soft-a stroke-a"/>
<text x="502.0" y="282" text-anchor="middle" class="fig-bold">256</text>
<text x="502.0" y="300" text-anchor="middle" class="fig-small">after 1×1</text>
<line x1="110" y1="190" x2="192" y2="190" class="stroke-ink" marker-end="url(#c036-f4-arrow)"/>
<text x="151" y="180" text-anchor="middle" class="fig-small">1×1 conv: shrink</text>
<line x1="250" y1="190" x2="332" y2="190" class="stroke-ink" marker-end="url(#c036-f4-arrow)"/>
<text x="291" y="180" text-anchor="middle" class="fig-small">3×3 conv: work</text>
<line x1="390" y1="190" x2="472" y2="190" class="stroke-ink" marker-end="url(#c036-f4-arrow)"/>
<text x="431" y="180" text-anchor="middle" class="fig-small">1×1 conv: grow</text>
<path d="M82.0,119.6 L82.0,95.6 L585,95.6 L585,174" fill="none" class="stroke-a" stroke-width="2.5" marker-end="url(#c036-f4-arrow-a)"/>
<text x="322" y="88" text-anchor="middle" class="fig-small">shortcut at full width: 256 channels, no cost</text>
<line x1="524" y1="190" x2="568" y2="190" class="stroke-ink" marker-end="url(#c036-f4-arrow)"/>
<circle cx="585" cy="190" r="15" class="fill-paper stroke-ink"/><text x="585" y="196" text-anchor="middle" class="fig-bold">+</text>
</svg>
Figure 4. The channel widths of one bottleneck block, with heights drawn in proportion to the channel counts. The block shrinks 256 channels to 64 with a cheap 1×1 convolution, does its 3×3 work at that narrow width, and grows back to 256. The shortcut skips all of it at full width, so the plus sign adds two arrays of the same shape.
```

Squeezing is safe because the first 1×1 convolution is learned, so it picks which 64 mixtures of the channels are worth keeping, and because the shortcut still carries all 256 channels past the block, so the branch only writes a correction.

### What a convolution costs

Take a convolution that keeps the image the same size, with stride 1, and ignore biases. Each output number reads a window of `k × k` pixels in all $c_{\text{in}}$ input channels, which is $k^2 c_{\text{in}}$ products, and there are $H \cdot W \cdot c_{\text{out}}$ output numbers. Multiply the two counts:

$$
\text{cost} = H \, W \, c_{\text{in}} \, c_{\text{out}} \, k^2
$$

Read it aloud: the cost is the number of output numbers times the number of products each one needs. On a 2×2 picture with 4 channels in and 4 out, a 1×1 convolution costs $2 \cdot 2 \cdot 4 \cdot 4 \cdot 1 = 64$ and a 3×3 convolution costs $2 \cdot 2 \cdot 4 \cdot 4 \cdot 9 = 576$. A 1×1 convolution has no factor of 9 because it mixes the channels at each pixel and never looks at a neighbour. Cost is also a product of both channel counts, so halving every channel divides it by four. A bottleneck uses both facts. In numpy:

```python
flops = h * w * c_in * c_out * k * k
```

### One bottleneck, counted

The block has a 1×1 from $c_{\text{in}}$ to $c_{\text{mid}}$, a 3×3 from $c_{\text{mid}}$ to $c_{\text{mid}}$, and a 1×1 from $c_{\text{mid}}$ to $c_{\text{out}}$. The shortcut is a plain addition and is not in the count:

$$
\text{flops} = H W \left( c_{\text{in}} c_{\text{mid}} + 9\, c_{\text{mid}}^2 + c_{\text{mid}} c_{\text{out}} \right)
$$

With $c_{\text{in}} = c_{\text{out}} = 256$, $c_{\text{mid}} = 64$, at 14×14, the shrink costs $256 \times 64 = 16{,}384$ per pixel, the 3×3 costs $9 \times 64 \times 64 = 36{,}864$, and the grow costs $64 \times 256 = 16{,}384$. That is $69{,}632$ per pixel, and over $14 \times 14 = 196$ pixels the block costs $13{,}647{,}872$.

Two full-width 3×3 convolutions cost $2 \times 9 \times 256 \times 256 = 1{,}179{,}648$ per pixel, which is $231{,}211{,}008$ over the picture, so the ratio is about 16.9. It does not depend on the picture size when the middle is a quarter of the outer width $C$: the bottleneck costs $\tfrac{17}{16} C^2$ per pixel against $18 C^2$, and $18 / (17/16) = 288/17 \approx 16.9$.

> [!KEY]
> A convolution's cost is the product of its channel counts, so shrinking the channels with a cheap 1×1 makes the 3×3 work cost a quarter of the width squared. A bottleneck keeps the wide path between blocks and spends about 17 times fewer multiply-adds, which is how a network affords fifty or a hundred blocks.

> [!RECAP]
> Hold these points before you build:
>
> - A residual block computes $y = x + F(x)$ with no ReLU after the sum, so a zero correction returns the input.
> - The plus sign sends `dY` to the branch and to the shortcut, so `dx = dY + dh1 @ W1`. Forward multiplies by `W.T`, and backward by `W`.
> - A plain layer gives $J^L$ over $L$ layers and a residual layer $(1 + J)^L$, but the note points at an angle, so the real length stays near 1.
> - A convolution costs $H W c_{\text{in}} c_{\text{out}} k^2$, and a bottleneck about 16.9 times less at a quarter width.

## Build it step by step

The checks call the functions in this order:

1. **`residual_block_forward(x, params)`** takes `x` of shape `(n, d)` and `params` holding `W1`, `b1`, `W2`, `b2`. It returns `(y, cache)`, with `cache` the tuple `(x, W1, W2, h1, a1)` in that order. The usual mistake is a ReLU after the sum.
2. **`identity_gap(x, params, residual)`** returns the largest absolute difference between the block's output and `x`, as a `float`. When `residual` is false the output is a plain block's `F(x)`, which is `y - x` for your forward pass.
3. **`residual_block_backward(dY, cache)`** returns `(dx, grads)` with keys `W1`, `b1`, `W2`, `b2`, each shaped like its parameter. Use `.T @`, `.sum(axis=0)` for the biases, and the mask `h1 > 0`. The usual mistake is to return only the branch term and forget to add `dY`.
4. **`orthogonal_matrix(d, rng)`.** Return a `(d, d)` matrix `Q` with `Q.T @ Q` equal to the identity. Take `q, r = np.linalg.qr(rng.normal(size=(d, d)))` and multiply the columns of `q` by `np.sign(np.diag(r))`. The usual mistake is to return the Gaussian matrix, which does not keep lengths.
5. **`scalar_gradient_factor(J, depth, residual)`.** Return `J ** depth` for a plain stack and `(1 + J) ** depth` for a residual stack, as a `float`. The usual mistake is to return `1 + J ** depth`.
6. **`gradient_norm_through_stack(depth, residual)`.** Use width 16, scale 0.1, and `np.random.default_rng(0)`. Build `depth` layers, each 0.1 times `orthogonal_matrix(16, rng)`. Start from `np.ones(16) / np.sqrt(16)`, walk the layers in reverse, and return `float(np.linalg.norm(g))`. The usual mistake is a Gaussian matrix, so the plain shrink is not exactly 0.1 per layer.
7. **`conv_flops(c_in, c_out, k, h, w)`.** Return the multiply-add count of one convolution as an `int`. The usual mistake is to leave out the kernel area, which makes every convolution look like a 1×1.
8. **`bottleneck_flops(c_in, c_mid, c_out, h, w)`.** Return an `int`. Call `conv_flops` three times: a 1×1 from `c_in` to `c_mid`, a 3×3 from `c_mid` to `c_mid`, and a 1×1 from `c_mid` to `c_out`. The usual mistake is to use `k = 1` for all three.
9. **`bottleneck_savings(c, h, w)`.** Return a `float`: the cost of two 3×3 convolutions at `c` channels divided by `bottleneck_flops(c, c // 4, c, h, w)`. The usual mistake is to divide the other way, which gives about 0.06 and not about 17.

## Where this breaks

- A block that changes the channel count cannot add its input to its output, so it puts a learned 1×1 convolution on the shortcut, a **projection shortcut**. Memory grows in a straight line with depth, because backpropagation keeps every block's activations.
- A shortcut keeps the gradient alive, but it does not fix a badly scaled branch. With weight scale 0.5 in place of 0.1, the same 50-layer residual stack has a median length near 150 over 200 seeds, because the notes are no longer small.
- Shrinking gradients are one mechanism, not the whole explanation. The authors of the original residual-network paper reported that plain networks with normalisation layers, whose signals were healthy, still trained worse at depth.
- Real blocks put normalisation inside the branch, which this chapter leaves out. [The transformer block](/lessons/055-the-transformer-block) builds that version.
- A multiply-add count is a guide and not a stopwatch, because running time also depends on how much data a layer moves.

## The exercise

You build `residual_block_forward`, `identity_gap`, `residual_block_backward`, `orthogonal_matrix`, `scalar_gradient_factor`, `gradient_norm_through_stack`, `conv_flops`, `bottleneck_flops`, and `bottleneck_savings`. The checks match the forward pass and every gradient to the worked numbers above, check that a zero branch returns the input, and compare gradients with finite differences. They confirm that the plain stack's length equals $0.1^{\text{depth}}$ and that the residual stack at depth 50 stays between 0.1 and 10, far below the scalar prediction. Last, they match `conv_flops` and `bottleneck_flops` on hand-computed cases, and confirm a ratio near 16.9 at several widths and picture sizes.
