---
title: Walking slopes backwards through a calculation
slug: 054-backprop-the-graph-and-reverse-order
part: "Part 4 — Neural networks"
summary: One backward walk through a recorded calculation returns the slope of the loss for every input, if each node finishes before it forwards.
minutes: 40
gpu: false
objectives:
  - Explain why one backward walk gives every slope while nudging each input needs a rerun per input.
  - Compute the slopes of a small calculation by hand, passing each operation's local slope backwards.
  - Explain why a value used twice collects the sum of its slopes, and why a node must finish before it forwards.
  - Check any slope against a two-sided nudge.
prereqs:
  - 052-forward-pass-and-gelu
  - 053-width-depth-and-cost
exercise: 054-backprop-the-graph-and-reverse-order
---

# Walking slopes backwards through a calculation

You have a network with a million weights and a loss that scores its mistakes. To train it, you need to know for every weight whether the loss goes up or down when the weight grows, and by how much. That number is the slope of the loss with respect to the weight, and the list of all of them is the **gradient**.

The obvious way to get a slope is to measure it. Nudge one weight up a little, rerun the network, and see how much the loss moved. That costs two runs for one weight, because you want to nudge up and down, so a million weights cost two million runs for a single training step. Writing the slopes by hand as formulas does not scale either: the formula for a two-layer network fills a page and changes with the network.

This chapter shows that one walk backwards through the calculation hands you every slope at once, for about the price of a few forward runs. The question to keep in mind: how can one backward pass produce all the slopes, and how do you keep it correct when a value is used in two places?

## What you already know

- A slope says how much an output moves for a small nudge of an input. Nudge $a$ by 0.01, see the output move by 0.03, and the slope is 3.
- A forward pass is a chain of small steps, each computing a value from earlier ones (see *Run a network forward, with a sharp or smooth bend*).
- You can count what a forward pass costs (see *Count the parameters, flops, and memory of a network*).
- Slopes along a chain multiply: if $u$ moves 3 per unit of $a$, and $c$ moves 2 per unit of $u$, then $c$ moves 6 per unit of $a$.

## Draw the calculation as a graph

Take a calculation small enough to hold in your head. Let $a = 3$ and $b = 2$, multiply them to get $p = a \cdot b = 6$, then add $a$ again to get $c = p + a = 9$. This is the running example.

Figure 1 draws it as a **graph**: each value is a **node**, and an arrow runs from each input to the value computed from it. The inputs of a step are that node's **children** — the name is odd, because the arrows run from child to result, but it is the word the code uses. Recording the steps as you run them, so you can walk them again later, is sometimes called keeping a **tape**. The tape and the graph are the same record.

```figure
<svg viewBox="0 0 600 322" role="img" aria-label="The graph of c equals a times b plus a, with forward values in blue and backward slopes in orange.">
<defs><marker id="c061-arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker><marker id="c061-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<rect x="38.0" y="88.0" width="84.0" height="44.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="80.0" y="107.0" text-anchor="middle" class="fig-bold">a</text>
<text x="80.0" y="124.0" text-anchor="middle" class="fig-small fill-a">value 3</text>
<rect x="38.0" y="228.0" width="84.0" height="44.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="80.0" y="247.0" text-anchor="middle" class="fig-bold">b</text>
<text x="80.0" y="264.0" text-anchor="middle" class="fig-small fill-a">value 2</text>
<rect x="248.0" y="158.0" width="84.0" height="44.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="290.0" y="177.0" text-anchor="middle" class="fig-bold">p</text>
<text x="290.0" y="194.0" text-anchor="middle" class="fig-small fill-a">value 6</text>
<rect x="468.0" y="158.0" width="84.0" height="44.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="510.0" y="177.0" text-anchor="middle" class="fig-bold">c</text>
<text x="510.0" y="194.0" text-anchor="middle" class="fig-small fill-a">value 9</text>
<line x1="122.8" y1="131.6" x2="242.8" y2="171.6" class="stroke-muted" marker-end="url(#c061-arrow-muted)"/>
<line x1="247.2" y1="158.4" x2="127.2" y2="118.4" class="stroke-b dashed" marker-end="url(#c061-arrow-b)"/>
<text x="190.4" y="132.9" text-anchor="middle" class="fig-bold fill-b">2</text>
<line x1="127.2" y1="241.6" x2="247.2" y2="201.6" class="stroke-muted" marker-end="url(#c061-arrow-muted)"/>
<line x1="242.8" y1="188.4" x2="122.8" y2="228.4" class="stroke-b dashed" marker-end="url(#c061-arrow-b)"/>
<text x="179.6" y="202.9" text-anchor="middle" class="fig-bold fill-b">3</text>
<line x1="335.0" y1="187.0" x2="465.0" y2="187.0" class="stroke-muted" marker-end="url(#c061-arrow-muted)"/>
<line x1="465.0" y1="173.0" x2="335.0" y2="173.0" class="stroke-b dashed" marker-end="url(#c061-arrow-b)"/>
<text x="400.0" y="168.0" text-anchor="middle" class="fig-bold fill-b">1</text>
<path d="M90,85.0 C90,30 500,30 500,155.0" class="stroke-muted" marker-end="url(#c061-arrow-muted)"/>
<path d="M520,155.0 C520,50 110,50 110,85.0" class="stroke-b dashed" marker-end="url(#c061-arrow-b)"/>
<text x="300.0" y="48.0" text-anchor="middle" class="fig-bold fill-b">1</text>
<text x="300.0" y="24.0" text-anchor="middle" class="fig-small fill-muted">a feeds the sum as well</text>
<text x="510.0" y="222.0" text-anchor="middle" class="fig-small fill-b">slope 1</text>
<text x="290.0" y="222.0" text-anchor="middle" class="fig-small fill-b">slope 1</text>
<text x="80.0" y="292.0" text-anchor="middle" class="fig-small fill-b">slope 3</text>
<text x="70.0" y="152.0" text-anchor="middle" class="fig-small fill-b">slope 2 + 1 = 3</text>
<text x="300.0" y="314.0" text-anchor="middle" class="fig-small fill-muted">grey arrows: values going forward; orange dashed arrows: slopes coming back</text>
</svg>
Figure 1. The graph of c = a·b + a at a = 3, b = 2. The input a has two outgoing arrows, so two orange slopes come back to it, 2 through the product and 1 through the sum. They add to 3.
```

Notice that $a$ has two arrows leaving it, because it feeds both the product and the final sum. That detail matters in a moment.

## Why each step needs only its own slope

Why should a walk backwards work at all? Because each step only needs to know how its own output reacts to its own inputs. Nudge $a$ by a small amount $h$ in $p = a \cdot b$. The output moves by $b \cdot h$, so the local slope of $p$ in $a$ is $b$, which is 2. Nudge $b$ instead and $p$ moves by $a \cdot h$, so the local slope in $b$ is 3. That is why a product hands each input the *other* input: the nudge is multiplied by whatever the input is multiplied with. For a sum $p + a$, a nudge of $h$ in either input moves the result by exactly $h$, so each local slope is 1.

Pass the blame backwards. The output $c$ moves 1 per unit of itself, so start with a slope of 1 at $c$. Write $\bar{c}$ for the slope of the final output with respect to $c$, and likewise $\bar{p}$, $\bar{a}$, $\bar{b}$. Then the walk is:

1. At the sum, $\bar{p}$ gets $1 \times 1 = 1$ and one share of $\bar{a}$ gets $1 \times 1 = 1$.
2. At the product, $\bar{p} = 1$ is multiplied by each local slope. Input $a$ gets $1 \times b = 2$ and input $b$ gets $1 \times a = 3$.

Because the product used $a$ and the sum used $a$, node $a$ holds two shares, 1 and 2.

## A value used twice collects both slopes

Do the two shares add, or does the second replace the first? They add. Nudge $a$ by $h$ and watch what happens. Through the product, $c$ moves by $2h$. Through the sum, $c$ moves by $1h$. Both happen at the same time, because the same $a$ feeds both steps, so $c$ moves by $3h$ in total. The slope of $c$ in $a$ is $2 + 1 = 3$: a slope is the sum over every route from the node to the output.

Expanding confirms it: with $b = 2$, $c = 2a + a = 3a$, so the slope in $a$ is 3, and $c = 3b + 3$ gives slope 3 in $b$.

Here are the two rules, read aloud. For $z = x + y$: the slope at $x$ gains $\bar{z}$, and the slope at $y$ gains $\bar{z}$. For $z = x \cdot y$: the slope at $x$ gains $\bar{z} \cdot y$, and the slope at $y$ gains $\bar{z} \cdot x$.

$$
z = x + y:\quad \bar{x} \mathrel{+}= \bar{z},\ \ \bar{y} \mathrel{+}= \bar{z}
\qquad\qquad
z = x \cdot y:\quad \bar{x} \mathrel{+}= \bar{z}\, y,\ \ \bar{y} \mathrel{+}= \bar{z}\, x .
$$

The symbol $\mathrel{+}=$ means "add onto what is already stored", which is the whole point. In numpy it is one line per input:

```python
self.grad += out.grad * other.value
other.grad += out.grad * self.value
```

The classic trap is a product of a value with itself. At $a = 3$, each factor of $a \cdot a$ contributes $1 \times 3 = 3$, and the true slope $2a$ is $3 + 3 = 6$. Write `=` instead of `+=` and the second contribution erases the first, leaving 3.

> [!WARNING]
> Every backward rule uses `+=`. A node that feeds two places must receive two slopes and keep both.

## Finish a node before it forwards

There is a second trap, about order. A node forwards its slope to its own inputs when its backward rule runs. If that happens before all of the node's users have sent their shares, the node forwards an incomplete number, and no later step can repair it.

Figure 2 shows this on a diamond. Hang the earlier idea under an extra input: $x = 1$ and $a = 2x = 2$. Then $b = a \cdot a = 4$, $c = a + a = 4$, and $d = b + c = 8$. Because $d = a^2 + 2a$, the slope of $d$ in $a$ is $2a + 2 = 6$, and in $x$ it is $6 \times 2 = 12$.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="The graph x to a, a to b and c, b and c to d, with the slope walk drawn in the right order and in a wrong order.">
<defs><marker id="c061-arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<rect x="32.0" y="53.0" width="56.0" height="34.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="60.0" y="75.0" text-anchor="middle" class="fig-bold">x</text>
<rect x="162.0" y="53.0" width="56.0" height="34.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="190.0" y="75.0" text-anchor="middle" class="fig-bold">a</text>
<rect x="302.0" y="19.0" width="56.0" height="34.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="330.0" y="41.0" text-anchor="middle" class="fig-bold">b</text>
<rect x="302.0" y="87.0" width="56.0" height="34.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="330.0" y="109.0" text-anchor="middle" class="fig-bold">c</text>
<rect x="442.0" y="53.0" width="56.0" height="34.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="470.0" y="75.0" text-anchor="middle" class="fig-bold">d</text>
<text x="60.0" y="118.0" text-anchor="middle" class="fig-small fill-a">x = 1</text>
<text x="190.0" y="118.0" text-anchor="middle" class="fig-small fill-a">a = 2x = 2</text>
<text x="420.0" y="16.0" text-anchor="end" class="fig-small fill-a">b = a·a = 4</text>
<text x="420.0" y="140.0" text-anchor="end" class="fig-small fill-a">c = a + a = 4</text>
<text x="520.0" y="118.0" text-anchor="middle" class="fig-small fill-a">d = b + c = 8</text>
<line x1="91.0" y1="70.0" x2="159.0" y2="70.0" class="stroke-muted" marker-end="url(#c061-arrow-muted)"/>
<line x1="221.0" y1="62.5" x2="299.0" y2="43.5" class="stroke-muted" marker-end="url(#c061-arrow-muted)"/>
<line x1="221.0" y1="77.5" x2="299.0" y2="96.5" class="stroke-muted" marker-end="url(#c061-arrow-muted)"/>
<line x1="361.0" y1="43.5" x2="439.0" y2="62.5" class="stroke-muted" marker-end="url(#c061-arrow-muted)"/>
<line x1="361.0" y1="96.5" x2="439.0" y2="77.5" class="stroke-muted" marker-end="url(#c061-arrow-muted)"/>
<text x="10.0" y="162.0" text-anchor="start" class="fig-bold">Right order: d, c, b, a</text>
<rect x="10.0" y="170.0" width="134.0" height="46.0" rx="6" class="fill-soft-c stroke-muted"/>
<text x="22.0" y="188.0" text-anchor="start" class="fig-bold">d</text>
<text x="22.0" y="206.0" text-anchor="start" class="fig-small fill-b">sends 1 to b and c</text>
<line x1="145.0" y1="193.0" x2="158.0" y2="193.0" class="stroke-muted" marker-end="url(#c061-arrow-muted)"/>
<rect x="160.0" y="170.0" width="134.0" height="46.0" rx="6" class="fill-soft-c stroke-muted"/>
<text x="172.0" y="188.0" text-anchor="start" class="fig-bold">c</text>
<text x="172.0" y="206.0" text-anchor="start" class="fig-small fill-b">a holds 2</text>
<line x1="295.0" y1="193.0" x2="308.0" y2="193.0" class="stroke-muted" marker-end="url(#c061-arrow-muted)"/>
<rect x="310.0" y="170.0" width="134.0" height="46.0" rx="6" class="fill-soft-c stroke-muted"/>
<text x="322.0" y="188.0" text-anchor="start" class="fig-bold">b</text>
<text x="322.0" y="206.0" text-anchor="start" class="fig-small fill-b">a holds 6</text>
<line x1="445.0" y1="193.0" x2="458.0" y2="193.0" class="stroke-muted" marker-end="url(#c061-arrow-muted)"/>
<rect x="460.0" y="170.0" width="134.0" height="46.0" rx="6" class="fill-soft-c stroke-muted"/>
<text x="472.0" y="188.0" text-anchor="start" class="fig-bold">a</text>
<text x="472.0" y="206.0" text-anchor="start" class="fig-small fill-b">x gets 6·2 = 12</text>
<text x="10.0" y="242.0" text-anchor="start" class="fig-bold">Wrong order: d, b, a, c</text>
<rect x="10.0" y="250.0" width="134.0" height="46.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="22.0" y="268.0" text-anchor="start" class="fig-bold">d</text>
<text x="22.0" y="286.0" text-anchor="start" class="fig-small fill-b">sends 1 to b and c</text>
<line x1="145.0" y1="273.0" x2="158.0" y2="273.0" class="stroke-muted" marker-end="url(#c061-arrow-muted)"/>
<rect x="160.0" y="250.0" width="134.0" height="46.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="172.0" y="268.0" text-anchor="start" class="fig-bold">b</text>
<text x="172.0" y="286.0" text-anchor="start" class="fig-small fill-b">a holds 4</text>
<line x1="295.0" y1="273.0" x2="308.0" y2="273.0" class="stroke-muted" marker-end="url(#c061-arrow-muted)"/>
<rect x="310.0" y="250.0" width="134.0" height="46.0" rx="6" class="fill-sunk stroke-bad"/>
<text x="322.0" y="268.0" text-anchor="start" class="fig-bold">a</text>
<text x="322.0" y="286.0" text-anchor="start" class="fig-small fill-bad">x gets 4·2 = 8</text>
<line x1="445.0" y1="273.0" x2="458.0" y2="273.0" class="stroke-muted" marker-end="url(#c061-arrow-muted)"/>
<rect x="460.0" y="250.0" width="134.0" height="46.0" rx="6" class="fill-sunk stroke-bad"/>
<text x="472.0" y="268.0" text-anchor="start" class="fig-bold">c</text>
<text x="472.0" y="286.0" text-anchor="start" class="fig-small fill-b">a holds 6, too late</text>
<text x="300.0" y="322.0" text-anchor="middle" class="fig-small fill-muted">same graph, same rules; only the order of the walk differs</text>
</svg>
Figure 2. The graph from x to d, walked twice. Each chip is one node running its backward rule, with what it hands on. In the wrong order, a forwards its slope to x while it holds only 4, so x ends at 8 instead of 12, and nothing corrects it later.
```

In the top row the walk goes $d$, $c$, $b$, $a$. When $a$ runs, both $b$ and $c$ have written into it, so $a$ holds 6 and sends 12 to $x$. In the bottom row $a$ runs before $c$, holds only the 4 from $b$, and sends 8 to $x$. The missing 2 arrives afterwards, with nobody left to forward it.

The fix is to run the nodes in an order where every user of a node comes before the node. You already have such an order for free: the forward pass. In the forward direction each node is computed after its inputs, so each node comes before all its users. Reverse that list and every user comes first. A list where each node comes after its children is called a **topological order**. Build one with a depth-first walk that visits a node's children first, then appends the node, and skips any node already visited:

```python
def visit(node):
    if id(node) in seen:
        return
    seen.add(id(node))
    for child in node._children:
        visit(child)
    order.append(node)
```

Then set the output's slope to 1 (ones, for an array), and call each node's backward rule in `reversed(order)`. Without the visited set, a node used twice would run twice and double-count its slope.

> [!KEY]
> One backward walk gives every slope because each step needs only its own local slope, a node used twice collects the sum of its slopes, and the reverse of the forward order guarantees a node is complete before it forwards.

## Why backward and not forward

You could also push slopes forward from an input: "if $a$ moves by $h$, how much does each later value move?" One such pass answers for *one* input and *every* later value. You would need a pass per weight, which is no better than nudging. Backward is the opposite shape: one pass answers for *one* output, the loss, and *every* input. A training run has one loss and a million weights, so the backward direction wins.

The price is memory. The rule for a product needs the other input's value, and ReLU's rule needs to know which inputs were positive, so the forward values must be kept until the backward walk uses them. *Putting it together: gradients for a whole network* counts this cost.

## Check a slope with two nudges

How do you know your slopes are right? Measure one. Nudge the input up and down, and divide the change in the output by the width of the nudge. At $a = 3$, $b = 2$, $c(a) = 2a + a$ with $h = 0.001$: $c(3.001) = 9.003$ and $c(2.999) = 8.997$. The gap is 0.006 and the width is 0.002, so the slope is 3. It matches the walk.

That is a **central difference**. In symbols, with a small $h$,

$$
\text{slope} \approx \frac{f(x + h) - f(x - h)}{2h} .
$$

Read it aloud: take the output a little right of $x$, minus the output a little left of it, divided by the distance between the two points. Why both sides? Figure 3 shows that a one-sided nudge tilts the line, because the curve bends away from the tangent; the two-sided line is balanced, so the first-order error cancels. The one-sided error shrinks like $h$, the two-sided like $h^2$. Shrinking $h$ forever does not help, because below about $10^{-8}$ rounding dominates. A value near $10^{-5}$ is a good default, and on a well-scaled function it agrees with the true slope to roughly nine digits.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="The curve a cubed with a one-sided secant that tilts steeply and a two-sided secant much closer to the tangent at a equal to 2.">

<line x1="60.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-muted"/>
<line x1="60.0" y1="250.0" x2="60.0" y2="20.0" class="stroke-muted"/>
<line x1="143.3" y1="250.0" x2="143.3" y2="254.0" class="stroke-muted"/>
<text x="143.3" y="268.0" text-anchor="middle" class="fig-small">1</text>
<line x1="310.0" y1="250.0" x2="310.0" y2="254.0" class="stroke-muted"/>
<text x="310.0" y="268.0" text-anchor="middle" class="fig-small">2</text>
<line x1="476.7" y1="250.0" x2="476.7" y2="254.0" class="stroke-muted"/>
<text x="476.7" y="268.0" text-anchor="middle" class="fig-small">3</text>
<line x1="56.0" y1="227.0" x2="60.0" y2="227.0" class="stroke-muted"/>
<text x="52.0" y="231.0" text-anchor="end" class="fig-small">0</text>
<line x1="60.0" y1="227.0" x2="560.0" y2="227.0" class="stroke-grid"/>
<line x1="56.0" y1="169.5" x2="60.0" y2="169.5" class="stroke-muted"/>
<text x="52.0" y="173.5" text-anchor="end" class="fig-small">10</text>
<line x1="60.0" y1="169.5" x2="560.0" y2="169.5" class="stroke-grid"/>
<line x1="56.0" y1="112.0" x2="60.0" y2="112.0" class="stroke-muted"/>
<text x="52.0" y="116.0" text-anchor="end" class="fig-small">20</text>
<line x1="60.0" y1="112.0" x2="560.0" y2="112.0" class="stroke-grid"/>
<line x1="56.0" y1="54.5" x2="60.0" y2="54.5" class="stroke-muted"/>
<text x="52.0" y="58.5" text-anchor="end" class="fig-small">30</text>
<line x1="60.0" y1="54.5" x2="560.0" y2="54.5" class="stroke-grid"/>
<path d="M60.0,226.3 L65.8,226.1 L71.7,225.9 L77.5,225.7 L83.3,225.5 L89.2,225.2 L95.0,224.9 L100.8,224.6 L106.7,224.3 L112.5,223.9 L118.3,223.5 L124.2,223.0 L130.0,222.5 L135.8,222.0 L141.7,221.4 L147.5,220.8 L153.3,220.2 L159.2,219.5 L165.0,218.7 L170.8,217.9 L176.7,217.1 L182.5,216.2 L188.3,215.2 L194.2,214.2 L200.0,213.2 L205.8,212.1 L211.7,210.9 L217.5,209.7 L223.3,208.4 L229.2,207.0 L235.0,205.6 L240.8,204.1 L246.7,202.6 L252.5,200.9 L258.3,199.2 L264.2,197.5 L270.0,195.7 L275.8,193.7 L281.7,191.8 L287.5,189.7 L293.3,187.6 L299.2,185.3 L305.0,183.0 L310.8,180.7 L316.7,178.2 L322.5,175.6 L328.3,173.0 L334.2,170.3 L340.0,167.4 L345.8,164.5 L351.7,161.5 L357.5,158.4 L363.3,155.2 L369.2,151.9 L375.0,148.5 L380.8,145.0 L386.7,141.4 L392.5,137.7 L398.3,133.9 L404.2,130.0 L410.0,125.9 L415.8,121.8 L421.7,117.6 L427.5,113.2 L433.3,108.7 L439.2,104.1 L445.0,99.4 L450.8,94.6 L456.7,89.6 L462.5,84.6 L468.3,79.4 L474.2,74.1 L480.0,68.6 L485.8,63.1 L491.7,57.4 L497.5,51.5 L503.3,45.6 L509.2,39.5 L515.0,33.2 L520.8,26.9 L526.7,20.4" class="stroke-a"/>
<text x="532.7" y="24.4" text-anchor="start" class="fig-bold fill-a">a³</text>
<path d="M176.7,236.2 L460.0,118.9" class="stroke-muted dashed"/>
<path d="M260.0,213.8 L485.0,66.3" class="stroke-bad"/>
<path d="M110.0,236.2 L485.0,68.0" class="stroke-c"/>
<circle cx="143.3" cy="221.2" r="4.5" class="fill-ink"/>
<circle cx="310.0" cy="181.0" r="4.5" class="fill-ink"/>
<circle cx="476.7" cy="71.8" r="4.5" class="fill-ink"/>
<text x="134.0" y="206.0" text-anchor="end" class="fig-small">a − h</text>
<text x="302.0" y="173.0" text-anchor="end" class="fig-small">a = 2</text>
<text x="466.7" y="73.8" text-anchor="end" class="fig-small">a + h</text>
<line x1="100.0" y1="40.0" x2="130.0" y2="40.0" class="stroke-muted dashed"/>
<text x="138.0" y="44.0" text-anchor="start" class="fig-small">true slope: 12</text>
<line x1="100.0" y1="60.0" x2="130.0" y2="60.0" class="stroke-bad"/>
<text x="138.0" y="64.0" text-anchor="start" class="fig-small">one-sided: slope 19</text>
<line x1="100.0" y1="80.0" x2="130.0" y2="80.0" class="stroke-c"/>
<text x="138.0" y="84.0" text-anchor="start" class="fig-small">two-sided: slope 13</text>
<text x="310.0" y="288.0" text-anchor="middle" class="fig-small">a (nudge size h = 1, exaggerated so the tilt is visible)</text>
</svg>
Figure 3. Two estimates of the slope of a³ at a = 2, using a huge nudge of h = 1. Stepping only to the right gives 19, far above the true 12. Stepping to both sides gives 13, because the errors from the two sides mostly cancel. Shrink h and the two-sided error falls like h², the one-sided like h.
```

This check costs two runs per input, so it is a test, never a training method. Use it to confirm every rule you write.

> [!RECAP]
> - A slope for every input comes from one backward walk, not from one rerun per input.
> - Each step passes slope using only its local slopes: a sum passes the slope through, a product hands each input the other input times the slope.
> - A value used twice collects the sum of the slopes from every route, so use `+=`.
> - Run the nodes in reverse of the forward order, so every node is complete before it forwards.
> - A central difference checks any slope with two runs.

## Build it step by step

You write three functions. The `Tensor` class stores a `value`, a `grad` (the slope, starting at zero), and its `_children`. `Tensor.__add__` is provided as a model.

1. **`central_difference(f, x, h=1e-5)`** takes a function of one number and a point, and returns the two-sided slope. It is one line. The common mistake is dividing by `h` instead of `2 * h`.

    ```python
    def central_difference(f, x, h=1e-5):
        # (f(x + h) - f(x - h)) / (2h)
    ```

2. **`Tensor.__mul__(other)`** builds the product with children `(self, other)` and a backward rule like the one for `+`. Each input gains the output slope times the other value. The common mistake is `=` instead of `+=`, which makes `a * a` return 3.

    ```python
    def __mul__(self, other):
        # out = Tensor(self.value * other.value, children=(self, other))
        # backward: self.grad += ..., other.grad += ...
    ```

3. **`Tensor.backward()`** builds the topological order with a depth-first walk and a visited set, sets the output's `grad` to ones, then calls `_backward_fn` on each node in `reversed(order)`. The common mistake is skipping the visited set, which runs a shared node twice.

    ```python
    def backward(self):
        # order = [] ; visit children first, then append
        # self.grad = ones ; for node in reversed(order): node._backward_fn()
    ```

## The exercise

You implement `central_difference`, `Tensor.__mul__`, and `Tensor.backward`. `Tensor.__init__` and `Tensor.__add__` are provided.

The checks run the two-sided slope on $3a$ and $a^2$ and compare it with a one-sided nudge. They check the product rule, the running example $c = a \cdot b + a$ (slope 3 in both inputs), `a * a` at 3 (slope 6), the diamond at 2 (slope 6), and the diamond hung under $x$ (slope 12, which comes out right only when the order is right). A final check compares the walk with `central_difference` on a mixed graph.
