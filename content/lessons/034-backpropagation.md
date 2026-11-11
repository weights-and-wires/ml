---
title: Backpropagation
slug: 034-backpropagation
part: "Part 4 — Neural networks"
summary: Walk slopes backwards through a recorded calculation, then derive the backward rule of each array operation from which inputs fed each output.
minutes: 75
gpu: false
objectives:
  - Explain why one backward walk gives every slope while nudging each input needs a rerun per input.
  - Compute the slopes of a small calculation by hand, and explain why a value used twice sums its slopes and a node must finish before it forwards.
  - Explain why a bias copied onto many rows collects the sum of the slopes of its copies.
  - Derive the matrix-product backward rule from one output entry, and say why the transposes appear.
  - Use the rule that a gradient has the shape of its value to catch a wrong rule, and check any slope against a two-sided nudge.
prereqs:
  - 033-multilayer-perceptrons
exercise: 034-backpropagation
---

# Backpropagation

You have a network with a million weights and a loss that scores its mistakes. To train it, you need to know for every weight whether the loss goes up or down when the weight grows, and by how much. That number is the slope of the loss with respect to the weight, and the list of all of them is the **gradient**.

The obvious way to get a slope is to measure it. Nudge one weight up a little, rerun the network, and see how much the loss moved. That costs two runs for one weight, because you want to nudge up and down, so a million weights cost two million runs for a single training step. Writing the slopes by hand as formulas does not scale either: the formula for a two-layer network fills a page and changes with the network.

This chapter shows that one walk backwards through the calculation hands you every slope at once, for about the price of a few forward runs. First you learn the walk on single numbers, including the two traps that break it: a value used twice, and the order of the steps. Then you extend it to arrays, where a real layer multiplies a batch by a weight matrix, adds a bias to every row, and bends with a ReLU.

## What you already know

- A slope says how much an output moves for a small nudge of an input. Nudge $a$ by 0.01, see the output move by 0.03, and the slope is 3.
- Slopes along a chain multiply: if $u$ moves 3 per unit of $a$, and $c$ moves 2 per unit of $u$, then $c$ moves 6 per unit of $a$.
- A forward pass is a chain of small steps, each computing a value from earlier ones ([multilayer perceptrons](/lessons/033-multilayer-perceptrons)).
- A matrix product `A @ B` multiplies shapes `(n, k)` and `(k, m)` to give `(n, m)`, and numpy broadcasting stretches a bias of shape `(m,)` across every row of an `(n, m)` matrix.

## Draw the calculation as a graph

Take a calculation small enough to hold in your head. Let $a = 3$ and $b = 2$, multiply them to get $p = a \cdot b = 6$, then add $a$ again to get $c = p + a = 9$.

Figure 1 draws it as a **graph**: each value is a **node**, and an arrow runs from each input to the value computed from it. The inputs of a step are that node's **children**, which is the word the code uses even though the arrows run from child to result. Recording the steps as you run them, so you can walk them again later, is sometimes called keeping a **tape**. The tape and the graph are the same record.

```figure
<svg viewBox="0 0 600 322" role="img" aria-label="The graph of c equals a times b plus a, with forward values in blue and backward slopes in orange.">
<defs><marker id="c027-arrow1-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker><marker id="c027-arrow1-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
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
<line x1="122.8" y1="131.6" x2="242.8" y2="171.6" class="stroke-muted" marker-end="url(#c027-arrow1-muted)"/>
<line x1="247.2" y1="158.4" x2="127.2" y2="118.4" class="stroke-b dashed" marker-end="url(#c027-arrow1-b)"/>
<text x="190.4" y="132.9" text-anchor="middle" class="fig-bold fill-b">2</text>
<line x1="127.2" y1="241.6" x2="247.2" y2="201.6" class="stroke-muted" marker-end="url(#c027-arrow1-muted)"/>
<line x1="242.8" y1="188.4" x2="122.8" y2="228.4" class="stroke-b dashed" marker-end="url(#c027-arrow1-b)"/>
<text x="179.6" y="202.9" text-anchor="middle" class="fig-bold fill-b">3</text>
<line x1="335.0" y1="187.0" x2="465.0" y2="187.0" class="stroke-muted" marker-end="url(#c027-arrow1-muted)"/>
<line x1="465.0" y1="173.0" x2="335.0" y2="173.0" class="stroke-b dashed" marker-end="url(#c027-arrow1-b)"/>
<text x="400.0" y="168.0" text-anchor="middle" class="fig-bold fill-b">1</text>
<path d="M90,85.0 C90,30 500,30 500,155.0" class="stroke-muted" marker-end="url(#c027-arrow1-muted)"/>
<path d="M520,155.0 C520,50 110,50 110,85.0" class="stroke-b dashed" marker-end="url(#c027-arrow1-b)"/>
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

A walk backwards works because each step only needs to know how its own output reacts to its own inputs. Nudge $a$ by a small amount $h$ in $p = a \cdot b$. The output moves by $b \cdot h$, so the local slope of $p$ in $a$ is $b$, which is 2. Nudge $b$ instead and $p$ moves by $a \cdot h$, so the local slope in $b$ is 3. A product hands each input the *other* input, because the nudge is multiplied by whatever the input is multiplied with. For a sum $p + a$, a nudge of $h$ in either input moves the result by exactly $h$, so each local slope is 1.

Pass the blame backwards. The output $c$ moves 1 per unit of itself, so start with a slope of 1 at $c$. Write $\bar{c}$ for the slope of the final output with respect to $c$, and likewise $\bar{p}$, $\bar{a}$, $\bar{b}$. At the sum, $\bar{p}$ gets $1 \times 1 = 1$ and one share of $\bar{a}$ gets $1 \times 1 = 1$. At the product, $\bar{p} = 1$ is multiplied by each local slope: input $a$ gets $1 \times b = 2$ and input $b$ gets $1 \times a = 3$. Because the product used $a$ and the sum used $a$, node $a$ holds two shares, 1 and 2.

## A value used twice collects both slopes

Do the two shares add, or does the second replace the first? They add. Nudge $a$ by $h$. Through the product, $c$ moves by $2h$. Through the sum, $c$ moves by $1h$. Both happen at once, because the same $a$ feeds both steps, so $c$ moves by $3h$ in total. A slope is the sum over every route from the node to the output. Expanding confirms it: with $b = 2$, $c = 2a + a = 3a$, so the slope in $a$ is 3.

The two rules, read aloud: for $z = x + y$, the slope at $x$ and at $y$ each gain $\bar{z}$. For $z = x \cdot y$, the slope at $x$ gains $\bar{z} \cdot y$ and the slope at $y$ gains $\bar{z} \cdot x$.

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

The second trap is order. A node forwards its slope to its own inputs when its backward rule runs. If that happens before all of the node's users have sent their shares, the node forwards an incomplete number, and no later step can repair it.

Figure 2 shows this on a diamond. Hang the earlier idea under an extra input: $x = 1$ and $a = 2x = 2$. Then $b = a \cdot a = 4$, $c = a + a = 4$, and $d = b + c = 8$. Because $d = a^2 + 2a$, the slope of $d$ in $a$ is $2a + 2 = 6$, and in $x$ it is $6 \times 2 = 12$.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="The graph x to a, a to b and c, b and c to d, with the slope walk drawn in the right order and in a wrong order.">
<defs><marker id="c027-arrow2-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
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
<line x1="91.0" y1="70.0" x2="159.0" y2="70.0" class="stroke-muted" marker-end="url(#c027-arrow2-muted)"/>
<line x1="221.0" y1="62.5" x2="299.0" y2="43.5" class="stroke-muted" marker-end="url(#c027-arrow2-muted)"/>
<line x1="221.0" y1="77.5" x2="299.0" y2="96.5" class="stroke-muted" marker-end="url(#c027-arrow2-muted)"/>
<line x1="361.0" y1="43.5" x2="439.0" y2="62.5" class="stroke-muted" marker-end="url(#c027-arrow2-muted)"/>
<line x1="361.0" y1="96.5" x2="439.0" y2="77.5" class="stroke-muted" marker-end="url(#c027-arrow2-muted)"/>
<text x="10.0" y="162.0" text-anchor="start" class="fig-bold">Right order: d, c, b, a</text>
<rect x="10.0" y="170.0" width="134.0" height="46.0" rx="6" class="fill-soft-c stroke-muted"/>
<text x="22.0" y="188.0" text-anchor="start" class="fig-bold">d</text>
<text x="22.0" y="206.0" text-anchor="start" class="fig-small fill-b">sends 1 to b and c</text>
<line x1="145.0" y1="193.0" x2="158.0" y2="193.0" class="stroke-muted" marker-end="url(#c027-arrow2-muted)"/>
<rect x="160.0" y="170.0" width="134.0" height="46.0" rx="6" class="fill-soft-c stroke-muted"/>
<text x="172.0" y="188.0" text-anchor="start" class="fig-bold">c</text>
<text x="172.0" y="206.0" text-anchor="start" class="fig-small fill-b">a holds 2</text>
<line x1="295.0" y1="193.0" x2="308.0" y2="193.0" class="stroke-muted" marker-end="url(#c027-arrow2-muted)"/>
<rect x="310.0" y="170.0" width="134.0" height="46.0" rx="6" class="fill-soft-c stroke-muted"/>
<text x="322.0" y="188.0" text-anchor="start" class="fig-bold">b</text>
<text x="322.0" y="206.0" text-anchor="start" class="fig-small fill-b">a holds 6</text>
<line x1="445.0" y1="193.0" x2="458.0" y2="193.0" class="stroke-muted" marker-end="url(#c027-arrow2-muted)"/>
<rect x="460.0" y="170.0" width="134.0" height="46.0" rx="6" class="fill-soft-c stroke-muted"/>
<text x="472.0" y="188.0" text-anchor="start" class="fig-bold">a</text>
<text x="472.0" y="206.0" text-anchor="start" class="fig-small fill-b">x gets 6·2 = 12</text>
<text x="10.0" y="242.0" text-anchor="start" class="fig-bold">Wrong order: d, b, a, c</text>
<rect x="10.0" y="250.0" width="134.0" height="46.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="22.0" y="268.0" text-anchor="start" class="fig-bold">d</text>
<text x="22.0" y="286.0" text-anchor="start" class="fig-small fill-b">sends 1 to b and c</text>
<line x1="145.0" y1="273.0" x2="158.0" y2="273.0" class="stroke-muted" marker-end="url(#c027-arrow2-muted)"/>
<rect x="160.0" y="250.0" width="134.0" height="46.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="172.0" y="268.0" text-anchor="start" class="fig-bold">b</text>
<text x="172.0" y="286.0" text-anchor="start" class="fig-small fill-b">a holds 4</text>
<line x1="295.0" y1="273.0" x2="308.0" y2="273.0" class="stroke-muted" marker-end="url(#c027-arrow2-muted)"/>
<rect x="310.0" y="250.0" width="134.0" height="46.0" rx="6" class="fill-sunk stroke-bad"/>
<text x="322.0" y="268.0" text-anchor="start" class="fig-bold">a</text>
<text x="322.0" y="286.0" text-anchor="start" class="fig-small fill-bad">x gets 4·2 = 8</text>
<line x1="445.0" y1="273.0" x2="458.0" y2="273.0" class="stroke-muted" marker-end="url(#c027-arrow2-muted)"/>
<rect x="460.0" y="250.0" width="134.0" height="46.0" rx="6" class="fill-sunk stroke-bad"/>
<text x="472.0" y="268.0" text-anchor="start" class="fig-bold">c</text>
<text x="472.0" y="286.0" text-anchor="start" class="fig-small fill-b">a holds 6, too late</text>
<text x="300.0" y="322.0" text-anchor="middle" class="fig-small fill-muted">same graph, same rules; only the order of the walk differs</text>
</svg>
Figure 2. The graph from x to d, walked twice. Each chip is one node running its backward rule, with what it hands on. In the wrong order, a forwards its slope to x while it holds only 4, so x ends at 8 instead of 12, and nothing corrects it later.
```

In the top row the walk goes $d$, $c$, $b$, $a$. When $a$ runs, both $b$ and $c$ have written into it, so $a$ holds 6 and sends 12 to $x$. In the bottom row $a$ runs before $c$, holds only the 4 from $b$, and sends 8 to $x$. The missing 2 arrives afterwards, with nobody left to forward it.

The fix is to run the nodes in an order where every user of a node comes before the node. The forward pass already gives one: each node is computed after its inputs, so reversing that list puts every user first. A list where each node comes after its children is a **topological order**. Build one with a depth-first walk that visits a node's children first, then appends the node, and skips any node already visited:

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

## Why backward, and how to check it

You could also push slopes forward from an input: "if $a$ moves by $h$, how much does each later value move?" One such pass answers for *one* input and *every* later value, so you need a pass per weight. Backward is the opposite shape: one pass answers for *one* output, the loss, and *every* input. A training run has one loss and a million weights, so backward wins. The price is memory, because the rule for a product needs the other input's value, so the forward values must be kept until the backward walk uses them.

To check a slope, measure it. Nudge the input up and down, and divide the change in the output by the width of the nudge. At $a = 3$, $b = 2$, $c(a) = 2a + a$ with $h = 0.001$: $c(3.001) = 9.003$ and $c(2.999) = 8.997$. The gap is 0.006 and the width is 0.002, so the slope is 3, matching the walk. This is a **central difference**:

$$
\text{slope} \approx \frac{f(x + h) - f(x - h)}{2h} .
$$

Read it aloud: the output a little right of $x$, minus the output a little left of it, divided by the distance between the two points. Both sides matter because a one-sided nudge tilts the line when the curve bends: the one-sided error shrinks like $h$, the two-sided like $h^2$. Shrinking $h$ forever does not help, because below about $10^{-8}$ rounding dominates, and $10^{-5}$ is a good default. This check costs two runs per input, so it is a test, never a training method.

## A slope for an array has the shape of the array

The loss is one number, but the value you are asking about is often an array. The slope of the loss with respect to an array has one entry per entry of the array: entry $(i, j)$ says how much the loss moves per unit nudge of that one entry. So the slope of a $3 \times 2$ weight matrix is itself a $3 \times 2$ array.

> [!KEY]
> A gradient has exactly the shape of its value. If a backward rule hands back the wrong shape, the rule is wrong.

Every array rule then falls out of the question you already asked about single numbers: which inputs fed this output, and with what weight? Suppose the slope arriving at the output of a bias add is $\bar{Z} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, the same shape as the output.

## A bias is a value used on every row

Adding a bias $b = [10, -30]$, of shape $(2,)$, to a $2 \times 2$ matrix looks like one operation, but numpy copies the bias onto each row first. The bias is one value used twice, exactly like $a$ above, so it collects the sum of its slopes. In Figure 3, each copy receives the slope of its own row, $[1, 2]$ and $[3, 4]$, and the bias gets their sum, $[4, 6]$. Shape $(2,)$, the shape of the bias.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="A bias of two numbers copied onto two rows going forward, and the two row slopes adding back into one bias slope going backward.">
<defs><marker id="c027-arrow3-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker><marker id="c027-arrow3-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="10.0" y="22.0" text-anchor="start" class="fig-bold">Forward: one bias, copied onto every row</text>
<rect x="40.0" y="86.0" width="40" height="28" class="fill-soft-a stroke-muted"/>
<text x="60.0" y="104.0" text-anchor="middle" class="fig-small">10</text>
<rect x="80.0" y="86.0" width="40" height="28" class="fill-soft-a stroke-muted"/>
<text x="100.0" y="104.0" text-anchor="middle" class="fig-small">-30</text>
<text x="80.0" y="134.0" text-anchor="middle" class="fig-small">bias b</text>
<rect x="330.0" y="50.0" width="40" height="28" class="fill-soft-a stroke-muted"/>
<text x="350.0" y="68.0" text-anchor="middle" class="fig-small">10</text>
<rect x="370.0" y="50.0" width="40" height="28" class="fill-soft-a stroke-muted"/>
<text x="390.0" y="68.0" text-anchor="middle" class="fig-small">-30</text>
<text x="430.0" y="68.0" text-anchor="start" class="fig-small">copy for row 0</text>
<path d="M124,100 C210,100 240,64 324,64" class="stroke-muted dashed" marker-end="url(#c027-arrow3-muted)"/>
<rect x="330.0" y="122.0" width="40" height="28" class="fill-soft-a stroke-muted"/>
<text x="350.0" y="140.0" text-anchor="middle" class="fig-small">10</text>
<rect x="370.0" y="122.0" width="40" height="28" class="fill-soft-a stroke-muted"/>
<text x="390.0" y="140.0" text-anchor="middle" class="fig-small">-30</text>
<text x="430.0" y="140.0" text-anchor="start" class="fig-small">copy for row 1</text>
<path d="M124,100 C210,100 240,136 324,136" class="stroke-muted dashed" marker-end="url(#c027-arrow3-muted)"/>
<text x="10.0" y="200.0" text-anchor="start" class="fig-bold">Backward: each copy sends its slope back, and the slopes add</text>
<rect x="330.0" y="222.0" width="40" height="28" class="fill-soft-b stroke-muted"/>
<text x="350.0" y="240.0" text-anchor="middle" class="fig-small">1</text>
<rect x="370.0" y="222.0" width="40" height="28" class="fill-soft-b stroke-muted"/>
<text x="390.0" y="240.0" text-anchor="middle" class="fig-small">2</text>
<text x="430.0" y="240.0" text-anchor="start" class="fig-small fill-b">slope at row 0</text>
<path d="M324,236 C240,236 210,246 126,246" class="stroke-b dashed" marker-end="url(#c027-arrow3-b)"/>
<rect x="330.0" y="262.0" width="40" height="28" class="fill-soft-b stroke-muted"/>
<text x="350.0" y="280.0" text-anchor="middle" class="fig-small">3</text>
<rect x="370.0" y="262.0" width="40" height="28" class="fill-soft-b stroke-muted"/>
<text x="390.0" y="280.0" text-anchor="middle" class="fig-small">4</text>
<text x="430.0" y="280.0" text-anchor="start" class="fig-small fill-b">slope at row 1</text>
<path d="M324,276 C240,276 210,246 126,246" class="stroke-b dashed" marker-end="url(#c027-arrow3-b)"/>
<rect x="40.0" y="232.0" width="40" height="28" class="fill-soft-b stroke-muted"/>
<text x="60.0" y="250.0" text-anchor="middle" class="fig-small">4</text>
<rect x="80.0" y="232.0" width="40" height="28" class="fill-soft-b stroke-muted"/>
<text x="100.0" y="250.0" text-anchor="middle" class="fig-small">6</text>
<text x="80.0" y="280.0" text-anchor="middle" class="fig-small fill-b">slope at b = [1+3, 2+4]</text>
</svg>
Figure 3. A bias used on two rows is one value used twice. Going forward it is copied; going backward each copy sends back its own slope, and the bias receives their sum, one number per column.
```

The helper that undoes the copying is `unbroadcast(grad, shape)`. It sums over the axes numpy added or stretched, until the slope has the shape of the original value:

```python
while grad.ndim > len(shape):
    grad = grad.sum(axis=0)
for axis, size in enumerate(shape):
    if size == 1 and grad.shape[axis] != 1:
        grad = grad.sum(axis=axis, keepdims=True)
```

Every `+=` in an add or a multiply that might broadcast goes through `unbroadcast` first. Without it, the bias would receive a $2 \times 2$ slope on a value of shape $(2,)$.

## A matrix product: ask who fed each output

Next, the transposes. A product $Z = A B$ has $Z_{ij} = \sum_k A_{ik} B_{kj}$: output entry $(i, j)$ is row $i$ of $A$ dotted with column $j$ of $B$.

Take one input entry, $A_{ik}$. It helped build every entry in row $i$ of $Z$, one for each column $j$, and in $Z_{ij}$ it is multiplied by $B_{kj}$. Nudge $A_{ik}$ by $h$ and $Z_{ij}$ moves by $B_{kj} h$. Each $Z_{ij}$ then moves the loss by $\bar{Z}_{ij}$ per unit, so by the sum rule,

$$
\bar{A}_{ik} = \sum_j \bar{Z}_{ij}\, B_{kj} .
$$

Read it aloud: the slope at $A_{ik}$ is row $i$ of $\bar{Z}$ dotted with row $k$ of $B$. A row of $\bar{Z}$ dotted with a *row* of $B$ is a row of $\bar{Z}$ times a *column* of $B^\top$, which is why the transpose appears. The same argument for $B_{kj}$, which fed every row $i$ of $Z$ with weight $A_{ik}$, gives $\bar{B}_{kj} = \sum_i A_{ik}\, \bar{Z}_{ij}$. Together:

$$
\bar{A} \mathrel{+}= \bar{Z}\, B^\top, \qquad \bar{B} \mathrel{+}= A^\top \bar{Z} .
$$

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Matrix product grids: A times B gives Z, then Z-slope times B transposed gives A-slope, with one entry highlighted in each row.">

<text x="10.0" y="22.0" text-anchor="start" class="fig-bold">Forward: Z = A B</text>
<rect x="20.0" y="50.0" width="34" height="26" class="fill-soft-a stroke-muted"/>
<text x="37.0" y="67.0" text-anchor="middle" class="fig-small">1</text>
<rect x="54.0" y="50.0" width="34" height="26" class="fill-soft-a stroke-muted"/>
<text x="71.0" y="67.0" text-anchor="middle" class="fig-small">2</text>
<rect x="88.0" y="50.0" width="34" height="26" class="fill-soft-a stroke-muted"/>
<text x="105.0" y="67.0" text-anchor="middle" class="fig-small">3</text>
<rect x="20.0" y="76.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="37.0" y="93.0" text-anchor="middle" class="fig-small">4</text>
<rect x="54.0" y="76.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="71.0" y="93.0" text-anchor="middle" class="fig-small">5</text>
<rect x="88.0" y="76.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="105.0" y="93.0" text-anchor="middle" class="fig-small">6</text>
<text x="71.0" y="42.0" text-anchor="middle" class="fig-small">A (2×3)</text>
<text x="142.0" y="76.0" text-anchor="middle">×</text>
<rect x="162.0" y="50.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="179.0" y="67.0" text-anchor="middle" class="fig-small">1</text>
<rect x="196.0" y="50.0" width="34" height="26" class="fill-soft-c stroke-muted"/>
<text x="213.0" y="67.0" text-anchor="middle" class="fig-small">0</text>
<rect x="162.0" y="76.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="179.0" y="93.0" text-anchor="middle" class="fig-small">2</text>
<rect x="196.0" y="76.0" width="34" height="26" class="fill-soft-c stroke-muted"/>
<text x="213.0" y="93.0" text-anchor="middle" class="fig-small">1</text>
<rect x="162.0" y="102.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="179.0" y="119.0" text-anchor="middle" class="fig-small">0</text>
<rect x="196.0" y="102.0" width="34" height="26" class="fill-soft-c stroke-muted"/>
<text x="213.0" y="119.0" text-anchor="middle" class="fig-small">3</text>
<text x="196.0" y="42.0" text-anchor="middle" class="fig-small">B (3×2)</text>
<text x="250.0" y="76.0" text-anchor="middle">=</text>
<rect x="270.0" y="50.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="287.0" y="67.0" text-anchor="middle" class="fig-small">5</text>
<rect x="304.0" y="50.0" width="34" height="26" class="fill-soft-b stroke-muted"/>
<text x="321.0" y="67.0" text-anchor="middle" class="fig-small">11</text>
<rect x="270.0" y="76.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="287.0" y="93.0" text-anchor="middle" class="fig-small">14</text>
<rect x="304.0" y="76.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="321.0" y="93.0" text-anchor="middle" class="fig-small">23</text>
<text x="304.0" y="42.0" text-anchor="middle" class="fig-small">Z (2×2)</text>
<text x="358.0" y="58.0" text-anchor="start" class="fig-small">Z[0,1] = row 0 of A</text>
<text x="358.0" y="74.0" text-anchor="start" class="fig-small">· column 1 of B</text>
<text x="358.0" y="90.0" text-anchor="start" class="fig-small fill-a">= 0 + 2 + 9 = 11</text>
<text x="10.0" y="170.0" text-anchor="start" class="fig-bold">Backward: slope of A = (slope of Z) × Bᵀ</text>
<rect x="20.0" y="200.0" width="34" height="26" class="fill-soft-b stroke-muted"/>
<text x="37.0" y="217.0" text-anchor="middle" class="fig-small">1</text>
<rect x="54.0" y="200.0" width="34" height="26" class="fill-soft-b stroke-muted"/>
<text x="71.0" y="217.0" text-anchor="middle" class="fig-small">2</text>
<rect x="20.0" y="226.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="37.0" y="243.0" text-anchor="middle" class="fig-small">3</text>
<rect x="54.0" y="226.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="71.0" y="243.0" text-anchor="middle" class="fig-small">4</text>
<text x="54.0" y="192.0" text-anchor="middle" class="fig-small">Z slope (2×2)</text>
<text x="108.0" y="219.0" text-anchor="middle">×</text>
<rect x="128.0" y="200.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="145.0" y="217.0" text-anchor="middle" class="fig-small">1</text>
<rect x="162.0" y="200.0" width="34" height="26" class="fill-soft-c stroke-muted"/>
<text x="179.0" y="217.0" text-anchor="middle" class="fig-small">2</text>
<rect x="196.0" y="200.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="213.0" y="217.0" text-anchor="middle" class="fig-small">0</text>
<rect x="128.0" y="226.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="145.0" y="243.0" text-anchor="middle" class="fig-small">0</text>
<rect x="162.0" y="226.0" width="34" height="26" class="fill-soft-c stroke-muted"/>
<text x="179.0" y="243.0" text-anchor="middle" class="fig-small">1</text>
<rect x="196.0" y="226.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="213.0" y="243.0" text-anchor="middle" class="fig-small">3</text>
<text x="179.0" y="192.0" text-anchor="middle" class="fig-small">Bᵀ (2×3)</text>
<text x="250.0" y="219.0" text-anchor="middle">=</text>
<rect x="270.0" y="200.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="287.0" y="217.0" text-anchor="middle" class="fig-small">1</text>
<rect x="304.0" y="200.0" width="34" height="26" class="fill-soft-a stroke-muted"/>
<text x="321.0" y="217.0" text-anchor="middle" class="fig-small">4</text>
<rect x="338.0" y="200.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="355.0" y="217.0" text-anchor="middle" class="fig-small">6</text>
<rect x="270.0" y="226.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="287.0" y="243.0" text-anchor="middle" class="fig-small">3</text>
<rect x="304.0" y="226.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="321.0" y="243.0" text-anchor="middle" class="fig-small">10</text>
<rect x="338.0" y="226.0" width="34" height="26" class="fill-sunk stroke-muted"/>
<text x="355.0" y="243.0" text-anchor="middle" class="fig-small">12</text>
<text x="321.0" y="192.0" text-anchor="middle" class="fig-small">A slope (2×3)</text>
<text x="386.0" y="208.0" text-anchor="start" class="fig-small">A slope[0,1] = row 0 of Z slope</text>
<text x="386.0" y="224.0" text-anchor="start" class="fig-small fill-b">· row 1 of B = 1·2 + 2·1 = 4</text>
<text x="300.0" y="300.0" text-anchor="middle" class="fig-small fill-muted">Each entry of the A slope is a row of the Z slope dotted with a row of B.</text>
</svg>
Figure 4. A product of a 2×3 matrix and a 3×2 matrix. Forward, the highlighted output entry uses one row of A and one column of B. Backward, the highlighted slope for A uses the same row of B, dotted with a row of the output slope, which is why Bᵀ appears.
```

Check it on numbers, with $A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$, $B = \begin{bmatrix} 1 & 0 \\ 2 & 1 \\ 0 & 3 \end{bmatrix}$, and $\bar{Z} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$. Then $Z = \begin{bmatrix} 5 & 11 \\ 14 & 23 \end{bmatrix}$. Entry $\bar{A}_{01}$ is row 0 of $\bar{Z}$, $[1, 2]$, dotted with row 1 of $B$, $[2, 1]$, giving $2 + 2 = 4$. The full answer is $\bar{A} = \begin{bmatrix} 1 & 4 & 6 \\ 3 & 10 & 12 \end{bmatrix}$ and $\bar{B} = \begin{bmatrix} 13 & 18 \\ 17 & 24 \\ 21 & 30 \end{bmatrix}$, with the shapes of $A$ and $B$. In numpy:

```python
self.grad += out.grad @ other.value.T
other.grad += self.value.T @ out.grad
```

The shape rule pays off here. Without the transpose, a $2 \times 2$ times a $3 \times 2$ does not multiply, and numpy raises an error. But the shapes force the transposes only when the matrices are not square, so test with a non-square product.

## ReLU and sum: gates and spreading

ReLU computes $\max(x, 0)$ one entry at a time. A positive input passes through unchanged, so its local slope is 1. A negative input is replaced by 0 whatever you do to it nearby, so its local slope is 0. Take $x = [-1, 2, 0.5, -3]$ with incoming slope $[5, 7, 1, 2]$. Entry 0 meets a closed gate and stops, so the outgoing slope is $[0, 7, 1, 0]$. A sum, $z = \sum_i x_i$, has no gates: nudge any entry by $h$ and the sum moves by $h$, so every entry receives the whole incoming slope.

$$
z = \operatorname{relu}(x):\quad \bar{x} \mathrel{+}= \bar{z} \odot [x > 0]
\qquad\qquad
z = \textstyle\sum_i x_i:\quad \bar{x} \mathrel{+}= \bar{z} \cdot \mathbf{1} .
$$

The symbol $\odot$ is entry-by-entry multiplication, and $[x > 0]$ is 1 where the test holds and 0 elsewhere. In numpy:

```python
self.grad += out.grad * (self.value > 0)
```

At exactly 0 the ReLU has a corner, so it has no single slope there. The engine picks 0, and real activations almost never land on exactly 0.

## Run a whole layer backwards

Look at what each rule used. The matrix product needed both inputs, the ReLU needed which inputs were positive, and the sum and add needed nothing. So the forward pass must keep its values until the backward walk uses them, which is why memory grows with depth.

Put the pieces together on $A$ and $B$ as above, a bias $b = [10, -30]$, and the layer $\operatorname{relu}(AB + b)$. Forward, $AB + b = \begin{bmatrix} 15 & -19 \\ 24 & -7 \end{bmatrix}$, and the ReLU gives $\begin{bmatrix} 15 & 0 \\ 24 & 0 \end{bmatrix}$. With the slope arriving at the ReLU output $\bar{Z} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ again, the second column sits behind closed gates, so the slope passing back is $\begin{bmatrix} 1 & 0 \\ 3 & 0 \end{bmatrix}$. The bias collects its column sums, $[4, 0]$. The product hands back $\bar{A} = \begin{bmatrix} 1 & 2 & 0 \\ 3 & 6 & 0 \end{bmatrix}$ and $\bar{B} = \begin{bmatrix} 13 & 0 \\ 17 & 0 \\ 21 & 0 \end{bmatrix}$. The second column of $B$ gets nothing, because every example blocked it. A weight whose unit is always off learns nothing from that batch.

> [!KEY]
> Backpropagation is one backward walk in reverse topological order. Every rule comes from asking which inputs fed each output entry: a copied value sums the slopes of its copies, a product sums over the entries it touched, a gate passes or blocks, and the result always has the shape of the value.

> [!RECAP]
> - A slope for every input comes from one backward walk, with each step passing slope using only its local slopes, and `+=` for a value used twice.
> - Run the nodes in reverse of the forward order, so every node is complete before it forwards.
> - A central difference, `(f(x + h) - f(x - h)) / (2 * h)`, checks any slope with two runs.
> - A gradient has the shape of its value, and a broadcast value collects the slopes of its copies through `unbroadcast`.
> - For `Z = A @ B`, $\bar{A} = \bar{Z} B^\top$ and $\bar{B} = A^\top \bar{Z}$. ReLU multiplies the slope by 1 where the input is positive and 0 elsewhere.

## Build it step by step

You write six functions. The `Tensor` class stores a `value`, a `grad` (the slope, starting at zero), and its `_children`. `Tensor.__init__` and `Tensor.__add__` are provided, and `__add__` is the model for the rules you write. Once you write `unbroadcast`, edit `__add__` and `__mul__` so every contribution goes through it.

1. **`central_difference(f, x, h=1e-5)`** takes a function of one number and a point, and returns the two-sided slope. It is one line. The common mistake is dividing by `h` instead of `2 * h`.

2. **`Tensor.__mul__(other)`** builds the product with children `(self, other)` and a backward rule like the one for `+`. Each input gains the output slope times the other value. The common mistake is `=` instead of `+=`, which makes `a * a` return 3.

3. **`Tensor.backward()`** builds the topological order with a depth-first walk and a visited set, sets the output's `grad` to ones, then calls `_backward_fn` on each node in `reversed(order)`. The common mistake is skipping the visited set, which runs a shared node twice.

4. **`unbroadcast(grad, shape)`** takes a slope array and the shape of the original value, and returns an array of exactly that shape. Sum leading axes away first, then sum with `keepdims=True` over each axis where `shape` has size 1. The common mistake is forgetting `keepdims=True`, which drops an axis and breaks a `(4, 1)` value.

5. **`Tensor.__matmul__(other)`** builds `out` from `self.value @ other.value` with children `(self, other)`. Its backward adds `out.grad @ other.value.T` to `self.grad` and `self.value.T @ out.grad` to `other.grad`. The common mistake is putting the transpose on the wrong factor, which fails loudly only on non-square shapes.

6. **`Tensor.relu()`** returns `np.maximum(self.value, 0)` with one child. Its backward adds `out.grad * (self.value > 0)`. The common mistake is `=` instead of `+=`, which breaks a tensor that feeds two places.

## Cost and scaling

The backward walk visits each node once, so its cost is the same order as the forward pass: about two forward passes for a matrix product, because the layer does two products backward for the one it did forward. The memory is the stored forward values: both inputs of every product and the sign of every ReLU input, so it grows with depth times batch size. The central-difference check costs two forward runs per parameter, which is why it is a test and not a way to train.

## The exercise

You implement `central_difference`, `Tensor.__mul__`, `Tensor.backward`, `unbroadcast`, `Tensor.__matmul__`, and `Tensor.relu`. The checks start with `central_difference` on $3a$, $a^2$, and $a^3$, compare it with a one-sided nudge, and check the product rule. They run the graph $c = a \cdot b + a$ (slope 3 in both inputs), `a * a` at 3 (slope 6), the diamond (slope 6) and the diamond hung under $x$ (slope 12, which comes out right only when the order is right), and a mixed graph against `central_difference`. Then they run `unbroadcast` on the $[[1, 2], [3, 4]]$ example, a broadcast bias and multiply, the $2 \times 3$ times $3 \times 2$ example with its exact slopes, a non-square product, the ReLU gates, and a whole layer, $\operatorname{relu}(XW + b)$, whose $W$ and $b$ slopes must match finite differences.
