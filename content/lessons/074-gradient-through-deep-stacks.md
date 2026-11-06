---
title: Why the shortcut keeps a fifty-layer gradient alive
slug: 074-gradient-through-deep-stacks
part: "Part 5 — Vision and sequences"
summary: Fifty plain layers shrink the gradient to nothing; a residual layer multiplies by one plus a small term, so the length stays near one.
minutes: 40
gpu: false
objectives:
  - Explain why a gradient that passes through fifty plain layers shrinks like 0.1 to the power 50.
  - Derive that a residual layer multiplies the gradient by one plus the branch's slope, and say why the scalar version of this overshoots.
  - Explain why a vector's length stays near one when a short, sideways note is added to it fifty times.
  - Build a stack of orthogonal layers and measure the gradient that reaches the input with and without the shortcut.
prereqs:
  - 061-signal-size-through-layers
  - 073-the-residual-block
exercise: 074-gradient-through-deep-stacks
---

# Why the shortcut keeps a fifty-layer gradient alive

You train a network with 50 layers. On every step the loss sends a gradient back, and that gradient must travel through 50 layers to reach the first one. How much of it arrives?

If the answer is almost none, the first layers never learn. The weights of layer 1 change by the learning rate times the gradient that reaches it. When that gradient is $10^{-50}$ of its starting size, the change is smaller than the smallest positive number a 32-bit float can store, about $10^{-45}$, so the update rounds to exactly zero and the layer stays where it started. This is one way a deep plain network fails to train.

This chapter counts how much of the gradient survives. For a plain stack, it is a long product of small factors and it collapses. For a residual stack, each factor is one plus a small term, and the product behaves very differently. You build both stacks and measure them.

## What you already know

Four ideas carry this chapter:

- A layer's backward pass multiplies the gradient by its weights, so a gradient through many layers is multiplied by many weight matrices (see "Why a deep network's signal shrinks or explodes").
- A residual block's input gradient is `dY` plus the branch's term (see "Gradients through a plus sign: the residual backward pass").
- The length of a vector, its **norm**, is the square root of the sum of its squared entries.
- A product of many numbers below one shrinks toward zero.

## Fifty people in a line

Think of the chain of people from the chapter "Why deeper networks got worse, and the residual block", this time passing along the gradient. In a plain line, each person hears the message and repeats a tenth of it to the next. After one person, a tenth of the message is left. After two, a hundredth. After fifty, one part in $10^{50}$. In a shortcut line, each person repeats the whole message and adds a note worth a tenth. Figure 1 draws five people of each line.

```figure
<svg viewBox="0 0 640 300" role="img" aria-label="Bar charts of how much of a message is left after each of five people: with a plain chain it falls to a tenth, then a hundredth, while with a shortcut chain in the scalar picture it grows slowly">
<text x="30" y="26" class="fig-bold">Plain line</text><text x="30" y="44" class="fig-small">each person repeats a tenth</text>
<line x1="30" y1="250" x2="300" y2="250" class="stroke-muted"/>
<rect x="40" y="130.00" width="36" height="120.00" class="fill-b"/>
<text x="58" y="124.0" text-anchor="middle" class="fig-small">1</text>
<text x="58" y="268" text-anchor="middle" class="fig-small">0</text>
<rect x="92" y="238.00" width="36" height="12.00" class="fill-b"/>
<text x="110" y="232.0" text-anchor="middle" class="fig-small">0.1</text>
<text x="110" y="268" text-anchor="middle" class="fig-small">1</text>
<rect x="144" y="248.80" width="36" height="1.20" class="fill-b"/>
<text x="162" y="242.8" text-anchor="middle" class="fig-small">0.01</text>
<text x="162" y="268" text-anchor="middle" class="fig-small">2</text>
<rect x="196" y="249.88" width="36" height="1.20" class="fill-b"/>
<text x="214" y="242.8" text-anchor="middle" class="fig-small">0.001</text>
<text x="214" y="268" text-anchor="middle" class="fig-small">3</text>
<rect x="248" y="249.99" width="36" height="1.20" class="fill-b"/>
<text x="266" y="242.8" text-anchor="middle" class="fig-small">1e-4</text>
<text x="266" y="268" text-anchor="middle" class="fig-small">4</text>
<text x="165" y="288" text-anchor="middle" class="fig-small">people who have passed it on</text>
<text x="340" y="26" class="fig-bold">Shortcut line, scalar picture</text><text x="340" y="44" class="fig-small">each repeats it all, plus a tenth</text>
<line x1="340" y1="250" x2="610" y2="250" class="stroke-muted"/>
<rect x="350" y="130.00" width="36" height="120.00" class="fill-a"/>
<text x="368" y="124.0" text-anchor="middle" class="fig-small">1</text>
<text x="368" y="268" text-anchor="middle" class="fig-small">0</text>
<rect x="402" y="118.00" width="36" height="132.00" class="fill-a"/>
<text x="420" y="112.0" text-anchor="middle" class="fig-small">1.1</text>
<text x="420" y="268" text-anchor="middle" class="fig-small">1</text>
<rect x="454" y="104.80" width="36" height="145.20" class="fill-a"/>
<text x="472" y="98.8" text-anchor="middle" class="fig-small">1.21</text>
<text x="472" y="268" text-anchor="middle" class="fig-small">2</text>
<rect x="506" y="90.28" width="36" height="159.72" class="fill-a"/>
<text x="524" y="84.3" text-anchor="middle" class="fig-small">1.331</text>
<text x="524" y="268" text-anchor="middle" class="fig-small">3</text>
<rect x="558" y="74.31" width="36" height="175.69" class="fill-a"/>
<text x="576" y="68.3" text-anchor="middle" class="fig-small">1.4641</text>
<text x="576" y="268" text-anchor="middle" class="fig-small">4</text>
<text x="475" y="288" text-anchor="middle" class="fig-small">people who have passed it on</text>
</svg>
Figure 1. Fifty-person chain, drawn for five people, with the scalar picture. On the plain line a tenth survives each hand-off, so the message is a hundredth after two people and almost gone after four. On the shortcut line each person passes it on whole and adds a tenth, so the amount grows. That growth is too optimistic, as the next section shows.
```

The numbers come from the chain rule. Suppose a plain layer multiplies the gradient by a number $J$, its slope. A residual layer computes $y = x + F(x)$, whose slope is $1 + J$, the same plus-sign rule you used in the previous chapter. Through fifty layers the factors multiply:

$$
\text{plain: } J^{50}, \qquad \text{residual: } (1 + J)^{50}
$$

Read it aloud: a plain stack multiplies fifty copies of $J$, and a residual stack multiplies fifty copies of one plus $J$. With $J = 0.1$, the plain product is $10^{-50}$, as in the telephone line. The residual product is $1.1^{50}$, about 117.

That second number should make you suspicious. A gradient that grows by a factor of 117 is not healthy either. If it were the true answer, a residual stack would swap one disaster for another. The mistake is in the picture, not in the arithmetic, and the next section finds it.

## Why the notes do not all point the same way

The scalar picture assumes that each layer's note points in the same direction as the message. A gradient in a real layer is a list of numbers, a vector, and the note is the message multiplied by a weight matrix. Multiplying by a matrix turns a vector as well as scaling it, so the note usually points somewhere else.

To count this, give every layer a weight of the simplest kind. A Gaussian random matrix would also rotate the gradient, but it would stretch it by an amount that depends on luck and on the width, which hides the effect we want to see. So use the kind that rotates and nothing else: 0.1 times an **orthogonal matrix**, a matrix that rotates or flips a vector and never changes its length. A plain layer then shortens the gradient by exactly 0.1 every time, because the orthogonal part keeps the length and the 0.1 scales it. That matches the scalar picture, so for the plain stack the scalar and matrix stories agree.

For the residual layer, the layer computes $g + gW$ on the way back. The note $gW$ has length 0.1 times the length of $g$, but it points at some angle $\theta$ to $g$. The law of cosines gives the length after one layer:

$$
|g + gW|^2 = |g|^2 \left( 1 + 2 \cdot 0.1 \cos\theta + 0.1^2 \right)
$$

Read it aloud: the squared length after the layer is the old one times one, plus twice the note's relative length times the cosine of the angle between them, plus the note's relative length squared. When the note points along $g$, $\cos\theta = 1$ and the factor is $1.1^2$, which is the scalar picture. When the note is at a right angle, $\cos\theta = 0$ and the length barely changes: it becomes $\sqrt{1.01} \approx 1.005$ times as long.

In 16 numbers, a random direction typically has a cosine of about $\pm 0.25$ with $g$, which is one over the square root of 16. Figure 2 draws three cases to scale. A note leaning toward $g$ gives a factor of about 1.03, and a note leaning away gives about 0.98. The factor therefore wanders a few percent above and below 1 from layer to layer, and the changes largely cancel. There is no steady push toward zero, and no steady push toward 117. Averaged over random directions, the cosine term cancels and the squared length grows by only the $0.1^2 = 0.01$ term per layer, so over 50 layers a typical length is about $1.01^{25} \approx 1.28$. That is the scale of the measured values below, not 117.

```figure
<svg viewBox="0 0 640 300" role="img" aria-label="Three panels showing a vector g of length one with a short added vector of length 0.1: pointing along g the sum is 1.10 long, leaning toward g at cosine 0.25 the sum is 1.03 long, and leaning away at cosine minus 0.25 the sum is 0.98 long">
<defs><marker id="c083-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker></defs>
<defs><marker id="c083-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<defs><marker id="c083-arrow-c" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-c"/></marker></defs>
<text x="25" y="28" class="fig-bold">Scalar picture:</text><text x="25" y="46" class="fig-small">the note points along g</text>
<polyline points="147.9,84.0 151.4,89.2 154.7,94.6 157.7,100.1 160.6,105.8 163.1,111.6 165.5,117.4 167.6,123.4 169.4,129.4 171.0,135.6 172.3,141.7 173.4,148.0 174.2,154.2 174.7,160.5 175.0,166.8 175.0,173.2 174.7,179.5 174.2,185.8 173.4,192.0 172.3,198.3 171.0,204.4 169.4,210.6" fill="none" class="stroke-muted dashed"/>
<line x1="25" y1="170" x2="190.0" y2="170.0" class="stroke-c" stroke-width="6"/>
<line x1="25" y1="170" x2="173" y2="170" class="stroke-a" stroke-width="2.5" marker-end="url(#c083-arrow-a)"/>
<line x1="175" y1="170" x2="190.0" y2="170.0" class="stroke-b" stroke-width="2.5" marker-end="url(#c083-arrow-b)"/>
<text x="100.0" y="194" text-anchor="middle" class="fig-small">g (length 1)</text>
<text x="100.0" y="238" text-anchor="middle" class="fig-bold">g + g@W: 1.10</text>
<text x="100.0" y="256" text-anchor="middle" class="fig-small">outside the dashed arc</text>
<text x="235" y="28" class="fig-bold">Matrix, in 16 numbers:</text><text x="235" y="46" class="fig-small">the note leans toward g, cos = 0.25</text>
<polyline points="357.9,84.0 361.4,89.2 364.7,94.6 367.7,100.1 370.6,105.8 373.1,111.6 375.5,117.4 377.6,123.4 379.4,129.4 381.0,135.6 382.3,141.7 383.4,148.0 384.2,154.2 384.7,160.5 385.0,166.8 385.0,173.2 384.7,179.5 384.2,185.8 383.4,192.0 382.3,198.3 381.0,204.4 379.4,210.6" fill="none" class="stroke-muted dashed"/>
<line x1="235" y1="170" x2="388.8" y2="155.5" class="stroke-c" stroke-width="6"/>
<line x1="235" y1="170" x2="383" y2="170" class="stroke-a" stroke-width="2.5" marker-end="url(#c083-arrow-a)"/>
<line x1="385" y1="170" x2="388.8" y2="155.5" class="stroke-b" stroke-width="2.5" marker-end="url(#c083-arrow-b)"/>
<text x="310.0" y="194" text-anchor="middle" class="fig-small">g (length 1)</text>
<text x="310.0" y="238" text-anchor="middle" class="fig-bold">g + g@W: 1.03</text>
<text x="310.0" y="256" text-anchor="middle" class="fig-small">outside the dashed arc</text>
<text x="445" y="28" class="fig-bold">Matrix, in 16 numbers:</text><text x="445" y="46" class="fig-small">the note leans away, cos = -0.25</text>
<polyline points="567.9,84.0 571.4,89.2 574.7,94.6 577.7,100.1 580.6,105.8 583.1,111.6 585.5,117.4 587.6,123.4 589.4,129.4 591.0,135.6 592.3,141.7 593.4,148.0 594.2,154.2 594.7,160.5 595.0,166.8 595.0,173.2 594.7,179.5 594.2,185.8 593.4,192.0 592.3,198.3 591.0,204.4 589.4,210.6" fill="none" class="stroke-muted dashed"/>
<line x1="445" y1="170" x2="591.2" y2="155.5" class="stroke-c" stroke-width="6"/>
<line x1="445" y1="170" x2="593" y2="170" class="stroke-a" stroke-width="2.5" marker-end="url(#c083-arrow-a)"/>
<line x1="595" y1="170" x2="591.2" y2="155.5" class="stroke-b" stroke-width="2.5" marker-end="url(#c083-arrow-b)"/>
<text x="520.0" y="194" text-anchor="middle" class="fig-small">g (length 1)</text>
<text x="520.0" y="238" text-anchor="middle" class="fig-bold">g + g@W: 0.98</text>
<text x="520.0" y="256" text-anchor="middle" class="fig-small">inside the dashed arc</text>
</svg>
Figure 2. One layer's effect on a gradient g of length 1, with true lengths. The added vector g@W is 0.1 long. In the scalar picture it points along g, so the sum is 1.10. In 16 numbers it is usually at an angle, so the sum is within a few percent of 1, above or below. The dashed arc is length 1: in the middle panel the green tip pokes just outside it, in the right panel just inside. Blue is g, orange is the note g@W, green is the sum.
```

## Fifty layers, measured

Measure it. The exercise builds a stack of 16-number layers, each 0.1 times an orthogonal matrix, with seed 0. It starts from a gradient of length 1, `np.ones(d) / np.sqrt(d)`, walks the layers in reverse, and reports the length that arrives. Each layer uses `g = g @ W` in a plain stack and `g = g + g @ W` in a residual stack. Figure 3 plots the result for every depth from 0 to 50. The backward rule is the one from the previous chapter in row form: the forward pass multiplies by `W.T`, so the backward pass multiplies by `W`.

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

The numbers at the two depths in the figure are:

| Depth | Residual length | Plain length |
|---|---|---|
| 10 | 0.90 | $10^{-10}$ |
| 50 | 1.15 | $10^{-50}$ |

The plain column is exact, since each layer shortens the length by exactly 0.1. The residual column depends on the random matrices: for 300 different seeds at depth 50, nine runs in ten land between 0.9 and 1.7. The median over those seeds is 1.22, and seed 0 gives 1.15, so read 1.15 as one typical value and not as a law. What does not depend on the seed is the plain stack's collapse, and the fact that the shortcut stack's length stays within a small factor of 1 instead of falling by fifty orders of magnitude.

> [!KEY]
> A plain layer multiplies the gradient by its weights, so fifty small factors multiply to nothing. A residual layer adds the gradient back to itself, so the length stays near one. The shortcut keeps a fifty-layer gradient alive, which answers the opening question: nearly all of it arrives.

## Be honest about what this shows

Three cautions keep the result in proportion:

- The shortcut stack is not better because its gradient grows. It is better because the gradient is not forced to shrink.
- The residual picture here uses small, orthogonal weights. A branch with large weights can still make the gradient explode, as the list in "Where this breaks" shows.
- Shrinking gradients are one mechanism, not the whole explanation. The authors of the original residual-network paper reported that plain networks with normalisation layers, whose signals were healthy, still trained worse at depth.

> [!RECAP]
> Hold these points before you write code:
>
> - A plain layer multiplies the gradient by $J$, so $L$ layers give $J^L$. A residual layer multiplies by $1 + J$.
> - The scalar picture overshoots because the note points along the message. A real layer's note points at an angle, and the length after the layer follows the law of cosines.
> - An orthogonal matrix keeps a vector's length, so 0.1 times one shortens a gradient by exactly 0.1.
> - Use `g @ W` for a plain layer and `g + g @ W` for a residual layer, walking the layers in reverse.
> - At depth 50 the residual length is near 1 and the plain length is $10^{-50}$.

## Build it step by step

The checks call the functions in this order:

1. **`orthogonal_matrix(d, rng)`.** `d` is the size and `rng` is a `np.random.Generator`. Return a `(d, d)` matrix `Q` with `Q.T @ Q` equal to the identity. Draw `rng.normal(size=(d, d))`, take `q, r = np.linalg.qr(a)`, and multiply the columns of `q` by `np.sign(np.diag(r))`. That sign fix makes the matrix evenly random. The usual mistake is to skip the QR and return the Gaussian matrix, which does not keep lengths.
2. **`scalar_gradient_factor(J, depth, residual)`.** Return `J ** depth` for a plain stack and `(1 + J) ** depth` for a residual stack, as a `float`. This is the toy model in the first section. The usual mistake is to return `1 + J ** depth`.
3. **`gradient_norm_through_stack(depth, residual)`.** Use width 16, scale 0.1, and `np.random.default_rng(0)`. Build `depth` layers, each 0.1 times `orthogonal_matrix(16, rng)`. Start from `np.ones(16) / np.sqrt(16)`, walk the layers in reverse, and return `float(np.linalg.norm(g))`. The usual mistake is a Gaussian matrix in place of an orthogonal one, which makes the plain stack's shrink differ from exactly 0.1 per layer.

## Where this breaks

Each of these limits has a plain consequence:

- A shortcut keeps the gradient alive, but it does not fix a badly scaled branch. The branch still needs a sensible starting scale and a normalisation layer.
- A branch that starts at zero makes the whole network an identity at step zero, the same idea as the zero correction in the first chapter on residual blocks. A zero starting scale for the last normalisation layer in each branch is one way to do that. This chapter's exercise has no normalisation layers, so this is a pointer to "Stacking it up: residuals, layer norm, and the pre-norm block" and not something you build here.
- The 0.1 scale matters. With weight scale 0.5 in place of 0.1, the same 50-layer residual stack has a median length near 150 over 200 seeds, because the notes are no longer small.

## The exercise

You build `orthogonal_matrix`, `scalar_gradient_factor`, and `gradient_norm_through_stack`. The checks confirm that your matrices keep lengths and are not the same on two draws. They match the toy factors for plain and residual stacks. The plain stack's length at depths 10 and 50 must equal $0.1^{\text{depth}}$ to many digits, which catches a Gaussian matrix. The residual stack at depth 50 must stay between 0.1 and 10, and the gap between the two must widen with depth.
