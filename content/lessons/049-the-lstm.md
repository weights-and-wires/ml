---
title: The LSTM, a belt with three gates
slug: 049-the-lstm
part: "Part 5 — Vision and sequences"
summary: A plain recurrent net loses the gradient to a product of small factors. An LSTM carries memory on a gated belt, with three gates to erase, write, and show.
minutes: 75
gpu: false
objectives:
  - Work a chain by hand for a plain net with factor 0.45 and for a belt with a gate of 0.9, and explain why a gate on a belt keeps the gradient alive.
  - Explain why a memory belt needs separate decisions to erase, to write, and to show.
  - Compute a gate as the sigmoid of x W plus h U plus b, and say why gates use the sigmoid and the candidate uses tanh.
  - Work one LSTM unit by hand, and write the forward step of an LSTM cell in numpy with the right shapes.
prereqs:
  - 031-multilayer-perceptrons
  - 047-recurrent-cells-and-bptt
  - 048-vanishing-gradients-and-clipping
exercise: 049-the-lstm
---

# The LSTM, a belt with three gates

Imagine a net reading a review one word at a time. At step 2 the review says "I bought this for my daughter". At step 200, after a long list of complaints about the shipping, it ends "so I would not buy this for ___ again", and the right word ("her") depends on what the net read at step 2. To learn that, the net needs a training signal that travels from step 200 back to step 2 and says "the word you read there mattered".

A plain recurrent net cannot carry that signal. By the time it has travelled back 200 steps, it is smaller than the smallest number a computer can usefully store, so the weights that handled step 2 never hear about the mistake. [The previous chapter](/lessons/048-vanishing-gradients-and-clipping) showed this with matrices. This chapter shows it with one number per step, and then changes the path so the number stays close to one.

The change is a belt that carries the memory straight along, with a gate on it, a number between 0 and 1 that the net chooses. A belt that only keeps and adds is not enough, though, because the net also has to decide what to erase, what to write, and when to show the memory. The answer is the LSTM, short for long short-term memory, which puts three gates on the belt. You build the running product that measures how much gradient a belt keeps, a single gate, and one full forward step of the cell.

## What you already know

- A recurrent cell reuses one set of weights at every step and passes a hidden state forward, and backpropagation through time multiplies the gradient by a local slope at each link.
- Over many steps those factors multiply, so factors below one make the gradient vanish, and clipping cannot lengthen a gradient that has faded.
- The sigmoid squashes any number into the range from 0 to 1, with 0 going to 0.5. The tanh squashes any number into the range from −1 to 1, and its slope is never above 1.
- A layer multiplies a batch `x`, with one row per example, by a weight matrix and adds a bias.
- The symbol $\odot$ means multiplying two arrays entry by entry, which is `*` in numpy.

## One factor per step

Picture the net as a chain of steps. The loss sits at step $t$, and you want to know how much it cares about the hidden state $k$ steps earlier. Each link passes the gradient back with a multiplier, because of the chain rule, so the gradient that reaches $k$ steps back is the starting gradient times one factor per link.

What is that factor for a plain net? Take one hidden unit with $h_t = \tanh(w\,h_{t-1} + \dots)$. The slope of $h_t$ with respect to $h_{t-1}$ is $w$ times the slope of tanh. The tanh slope is at most 1 and is smaller when the unit is far from zero. As an illustration, take a typical slope of 0.5 and a recurrent weight of 0.9, so the factor is $0.5 \times 0.9 = 0.45$. This is an illustrative choice, not a law, but any factor below one behaves the same way.

Figure 1 draws two chains of five steps with the same starting gradient of 1 at step $t$. In the top chain each link multiplies by 0.45.

```figure
<svg viewBox="0 0 600 345" role="img" aria-label="Two chains of five steps. Going back from the loss, a plain net multiplies the gradient by 0.45 at every link and a belt multiplies it by 0.9. Bars show the gradient left at each step.">
<defs><marker id="c039-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker><marker id="c039-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="20" y="22" class="fig-bold">Plain net: every link multiplies the gradient by 0.45</text>
<circle cx="80" cy="62" r="22" class="fill-soft-a stroke-a"/>
<text x="80" y="67" text-anchor="middle" class="fig-small">step t</text>
<rect x="64" y="85" width="32" height="50" class="fill-bad"/>
<line x1="56" y1="135" x2="104" y2="135" class="stroke-muted"/>
<text x="80" y="153" text-anchor="middle" class="fig-mono">1</text>
<circle cx="190" cy="62" r="22" class="fill-soft-a stroke-a"/>
<text x="190" y="67" text-anchor="middle" class="fig-small">t-1</text>
<rect x="174" y="112.5" width="32" height="22.5" class="fill-bad"/>
<line x1="166" y1="135" x2="214" y2="135" class="stroke-muted"/>
<text x="190" y="153" text-anchor="middle" class="fig-mono">0.45</text>
<line x1="104" y1="62" x2="166" y2="62" class="stroke-b" marker-end="url(#c039-arrow-b)"/>
<text x="135" y="52" text-anchor="middle" class="fig-small">x0.45</text>
<circle cx="300" cy="62" r="22" class="fill-soft-a stroke-a"/>
<text x="300" y="67" text-anchor="middle" class="fig-small">t-2</text>
<rect x="284" y="124.9" width="32" height="10.1" class="fill-bad"/>
<line x1="276" y1="135" x2="324" y2="135" class="stroke-muted"/>
<text x="300" y="153" text-anchor="middle" class="fig-mono">0.20</text>
<line x1="214" y1="62" x2="276" y2="62" class="stroke-b" marker-end="url(#c039-arrow-b)"/>
<text x="245" y="52" text-anchor="middle" class="fig-small">x0.45</text>
<circle cx="410" cy="62" r="22" class="fill-soft-a stroke-a"/>
<text x="410" y="67" text-anchor="middle" class="fig-small">t-3</text>
<rect x="394" y="130.4" width="32" height="4.6" class="fill-bad"/>
<line x1="386" y1="135" x2="434" y2="135" class="stroke-muted"/>
<text x="410" y="153" text-anchor="middle" class="fig-mono">0.09</text>
<line x1="324" y1="62" x2="386" y2="62" class="stroke-b" marker-end="url(#c039-arrow-b)"/>
<text x="355" y="52" text-anchor="middle" class="fig-small">x0.45</text>
<circle cx="520" cy="62" r="22" class="fill-soft-a stroke-a"/>
<text x="520" y="67" text-anchor="middle" class="fig-small">t-4</text>
<rect x="504" y="132.9" width="32" height="2.1" class="fill-bad"/>
<line x1="496" y1="135" x2="544" y2="135" class="stroke-muted"/>
<text x="520" y="153" text-anchor="middle" class="fig-mono">0.04</text>
<line x1="434" y1="62" x2="496" y2="62" class="stroke-b" marker-end="url(#c039-arrow-b)"/>
<text x="465" y="52" text-anchor="middle" class="fig-small">x0.45</text>
<text x="20" y="197" class="fig-bold">Belt: every link multiplies the gradient by a gate of 0.9</text>
<circle cx="80" cy="237" r="22" class="fill-soft-a stroke-a"/>
<text x="80" y="242" text-anchor="middle" class="fig-small">step t</text>
<rect x="64" y="260" width="32" height="50" class="fill-c"/>
<line x1="56" y1="310" x2="104" y2="310" class="stroke-muted"/>
<text x="80" y="328" text-anchor="middle" class="fig-mono">1</text>
<circle cx="190" cy="237" r="22" class="fill-soft-a stroke-a"/>
<text x="190" y="242" text-anchor="middle" class="fig-small">t-1</text>
<rect x="174" y="265" width="32" height="45" class="fill-c"/>
<line x1="166" y1="310" x2="214" y2="310" class="stroke-muted"/>
<text x="190" y="328" text-anchor="middle" class="fig-mono">0.90</text>
<line x1="104" y1="237" x2="166" y2="237" class="stroke-b" marker-end="url(#c039-arrow-b)"/>
<text x="135" y="227" text-anchor="middle" class="fig-small">x0.9</text>
<circle cx="300" cy="237" r="22" class="fill-soft-a stroke-a"/>
<text x="300" y="242" text-anchor="middle" class="fig-small">t-2</text>
<rect x="284" y="269.5" width="32" height="40.5" class="fill-c"/>
<line x1="276" y1="310" x2="324" y2="310" class="stroke-muted"/>
<text x="300" y="328" text-anchor="middle" class="fig-mono">0.81</text>
<line x1="214" y1="237" x2="276" y2="237" class="stroke-b" marker-end="url(#c039-arrow-b)"/>
<text x="245" y="227" text-anchor="middle" class="fig-small">x0.9</text>
<circle cx="410" cy="237" r="22" class="fill-soft-a stroke-a"/>
<text x="410" y="242" text-anchor="middle" class="fig-small">t-3</text>
<rect x="394" y="273.6" width="32" height="36.5" class="fill-c"/>
<line x1="386" y1="310" x2="434" y2="310" class="stroke-muted"/>
<text x="410" y="328" text-anchor="middle" class="fig-mono">0.73</text>
<line x1="324" y1="237" x2="386" y2="237" class="stroke-b" marker-end="url(#c039-arrow-b)"/>
<text x="355" y="227" text-anchor="middle" class="fig-small">x0.9</text>
<circle cx="520" cy="237" r="22" class="fill-soft-a stroke-a"/>
<text x="520" y="242" text-anchor="middle" class="fig-small">t-4</text>
<rect x="504" y="277.2" width="32" height="32.8" class="fill-c"/>
<line x1="496" y1="310" x2="544" y2="310" class="stroke-muted"/>
<text x="520" y="328" text-anchor="middle" class="fig-mono">0.66</text>
<line x1="434" y1="237" x2="496" y2="237" class="stroke-b" marker-end="url(#c039-arrow-b)"/>
<text x="465" y="227" text-anchor="middle" class="fig-small">x0.9</text>
</svg>
Figure 1. The gradient flows from step t back to earlier steps, and each bar shows how much is left on arrival. The plain chain loses more than half at every link and is down to 0.04 after four. Notice that the belt chain, with a factor of 0.9, still holds 0.66 after the same four links.
```

Read the top chain from left to right: 1, then $0.45$, then $0.2025$, then 0.0911, then 0.0410. After four links 4 percent of the gradient is left. After 10 links it is $0.45^{10} = 0.00034$, and after 200 it is about $4 \times 10^{-70}$. In general, after $k$ links the share left is

$$
\text{share left after } k \text{ steps} = \text{factor}^{k}
$$

Read it aloud: the share of the gradient that survives $k$ links is the per-link factor multiplied by itself $k$ times. A number below one, multiplied by itself over and over, heads to zero. In numpy, `factor ** np.arange(1, steps + 1)` gives the share left after every number of links.

You might think the fix is to make the recurrent weight bigger until the factor is 1. A factor of exactly 1 is a knife edge, because the tanh slope changes with the input and no single weight keeps the factor at 1. Push the weight up until the typical factor is 1, and the steps where the slope is larger have a factor above 1. A factor of 1.1 compounds to $1.1^{200} \approx 1.9 \times 10^{8}$ over 200 steps, so you trade a fading gradient for an exploding one.

A second idea is to stop redrawing the memory. Let the memory be a number $c$ and add each input to it: $c_t = c_{t-1} + x_t$. The gradient along this path has a factor of exactly 1, so it survives. But the memory never lets go of anything. Every input is added forever, so it grows without limit, and the net cannot clear a value once it stops being useful. The fix keeps the unchanged path and adds the missing control: a number that decides, at each step, how much of the old memory to keep.

## A belt with a gate on it

Call the memory the **belt**, and give it a **gate**: a number $f_t$ between 0 and 1 that multiplies the old memory before the fresh content is added.

$$
c_t = f_t \cdot c_{t-1} + (\text{fresh content})
$$

Read it aloud: the memory at step $t$ is a fraction $f_t$ of the old memory, plus whatever fresh content this step writes. A gate of 1 keeps everything, a gate of 0 wipes the belt, and a gate of 0.5 halves it. The net computes $f_t$ from what it reads, so it can keep the memory through the shipping complaints and clear it when a fresh sentence begins.

Next, ask what the gradient does across one link of the belt. The fresh content does not depend on $c_{t-1}$, so the only place $c_{t-1}$ enters $c_t$ is through the product $f_t \cdot c_{t-1}$, whose slope with respect to $c_{t-1}$ is $f_t$. That is the whole factor, with no weight matrix and no tanh slope. Three gates of 0.9 in a row give $0.9$, then $0.81$, then $0.729$. The belt still holds 73 percent after three links, where the plain net held 9 percent. When the gates differ from step to step, the share left after $k$ links is their product:

$$
f_1 \cdot f_2 \cdots f_k
$$

In numpy, a running product along the time axis is `np.cumprod(forget_gates, axis=0)`. It keeps every row, where `np.prod` keeps only the last. The path is named the **constant error carousel**: the error, meaning the gradient, stays roughly constant as it rides round and round the belt.

> [!WARNING]
> The product above is the gradient along the belt only. The gates are computed from the previous hidden state, which depends on the previous memory, so a second, smaller path exists. Gating slows the fade a great deal, but does not remove it.

How far can a gradient reach? Use the rule from the previous chapter: with a per-step factor below one, the share left falls under a threshold after $t > \ln(\text{threshold})/\ln(\text{factor})$ steps. With a threshold of 0.01, a factor of 0.45 gives 5.77, so 6 steps. A gate of 0.9 takes 44 steps, a gate of 0.99 takes 459, and a gate of 0.995 takes 919. The plain net hears from 6 steps back, and the belt can hear from hundreds. The exact gate value matters a great deal: a gate of 0.995 keeps 0.37 of the gradient at step 200, while 0.9 keeps almost nothing.

> [!KEY]
> The gradient through many steps is a product with one factor per step, so factors below one make it vanish. A belt with a gate has only the gate as its factor, so a gate near one lets the gradient survive. That answers the opening question: change the path so the memory is multiplied by a learned number near one, not by a weight matrix and a tanh slope.

## Store, hold, read

A belt that always keeps what it has and always adds what arrives has no way to decide anything. Picture a net that sees a number at step 1, then two steps of noise, and must report the number at step 4. Such a belt holds the number together with all of the noise, and shows that mixture at every step, whether anyone asked or not.

To store a number cleanly, someone has to make three separate decisions. Before writing, the old content must be erased. Once the number is stored, the noise must be refused. And only at step 4 should the memory be shown to the rest of the network. The net makes those decisions itself, with three gates, one per decision. Figure 2 follows one memory cell through the task. The cell starts with a leftover value of 0.7. Each panel shows the gate settings, the cell value $c$ as a green bar, and the value $h$ that the cell shows to the rest of the network as an orange bar.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Five panels showing a stored number: a leftover cell value of 0.7 is wiped and replaced by 1, held for two steps while noise arrives, and shown at the last step as 0.76">
<rect x="12" y="14" width="108" height="262" rx="8" class="fill-sunk stroke-grid"/>
<text x="66" y="38" text-anchor="middle" class="fig-bold">start</text>
<text x="24" y="64" class="fig-small">leftover</text>
<text x="24" y="82" class="fig-small">memory</text>
<line x1="22" y1="250" x2="110" y2="250" class="stroke-muted"/>
<rect x="28" y="190.5" width="30" height="59.5" class="fill-c"/>
<text x="43" y="184.5" text-anchor="middle" class="fig-small">0.70</text>
<text x="43" y="266" text-anchor="middle" class="fig-small">cell c</text>
<rect x="72" y="250" width="30" height="0.5" class="fill-b"/>
<text x="87" y="244" text-anchor="middle" class="fig-small">0.00</text>
<text x="87" y="266" text-anchor="middle" class="fig-small">shown h</text>
<rect x="128" y="14" width="108" height="262" rx="8" class="fill-sunk stroke-grid"/>
<text x="182" y="38" text-anchor="middle" class="fig-bold">step 1: write</text>
<text x="140" y="64" class="fig-small">forget f = 0</text>
<text x="140" y="82" class="fig-small">input  i = 1</text>
<text x="140" y="100" class="fig-small">output o = 0</text>
<text x="140" y="118" class="fig-small">cand. g = 1.0</text>
<line x1="138" y1="250" x2="226" y2="250" class="stroke-muted"/>
<rect x="144" y="165" width="30" height="85" class="fill-c"/>
<text x="159" y="159" text-anchor="middle" class="fig-small">1.00</text>
<text x="159" y="266" text-anchor="middle" class="fig-small">cell c</text>
<rect x="188" y="250" width="30" height="0.5" class="fill-b"/>
<text x="203" y="244" text-anchor="middle" class="fig-small">0.00</text>
<text x="203" y="266" text-anchor="middle" class="fig-small">shown h</text>
<rect x="244" y="14" width="108" height="262" rx="8" class="fill-sunk stroke-grid"/>
<text x="298" y="38" text-anchor="middle" class="fig-bold">step 2: hold</text>
<text x="256" y="64" class="fig-small">forget f = 1</text>
<text x="256" y="82" class="fig-small">input  i = 0</text>
<text x="256" y="100" class="fig-small">output o = 0</text>
<text x="256" y="118" class="fig-small">cand. g = 0.8</text>
<line x1="254" y1="250" x2="342" y2="250" class="stroke-muted"/>
<rect x="260" y="165" width="30" height="85" class="fill-c"/>
<text x="275" y="159" text-anchor="middle" class="fig-small">1.00</text>
<text x="275" y="266" text-anchor="middle" class="fig-small">cell c</text>
<rect x="304" y="250" width="30" height="0.5" class="fill-b"/>
<text x="319" y="244" text-anchor="middle" class="fig-small">0.00</text>
<text x="319" y="266" text-anchor="middle" class="fig-small">shown h</text>
<rect x="360" y="14" width="108" height="262" rx="8" class="fill-sunk stroke-grid"/>
<text x="414" y="38" text-anchor="middle" class="fig-bold">step 3: hold</text>
<text x="372" y="64" class="fig-small">forget f = 1</text>
<text x="372" y="82" class="fig-small">input  i = 0</text>
<text x="372" y="100" class="fig-small">output o = 0</text>
<text x="372" y="118" class="fig-small">cand. g = -0.6</text>
<line x1="370" y1="250" x2="458" y2="250" class="stroke-muted"/>
<rect x="376" y="165" width="30" height="85" class="fill-c"/>
<text x="391" y="159" text-anchor="middle" class="fig-small">1.00</text>
<text x="391" y="266" text-anchor="middle" class="fig-small">cell c</text>
<rect x="420" y="250" width="30" height="0.5" class="fill-b"/>
<text x="435" y="244" text-anchor="middle" class="fig-small">0.00</text>
<text x="435" y="266" text-anchor="middle" class="fig-small">shown h</text>
<rect x="476" y="14" width="108" height="262" rx="8" class="fill-sunk stroke-grid"/>
<text x="530" y="38" text-anchor="middle" class="fig-bold">step 4: read</text>
<text x="488" y="64" class="fig-small">forget f = 1</text>
<text x="488" y="82" class="fig-small">input  i = 0</text>
<text x="488" y="100" class="fig-small">output o = 1</text>
<text x="488" y="118" class="fig-small">cand. g = 0.4</text>
<line x1="486" y1="250" x2="574" y2="250" class="stroke-muted"/>
<rect x="492" y="165" width="30" height="85" class="fill-c"/>
<text x="507" y="159" text-anchor="middle" class="fig-small">1.00</text>
<text x="507" y="266" text-anchor="middle" class="fig-small">cell c</text>
<rect x="536" y="185.3" width="30" height="64.7" class="fill-b"/>
<text x="551" y="179.3" text-anchor="middle" class="fig-small">0.76</text>
<text x="551" y="266" text-anchor="middle" class="fig-small">shown h</text>
</svg>
Figure 2. A cell wipes its leftover 0.7 and writes a 1 at step 1, holds it through two steps of noise (candidates 0.8 and −0.6), and shows it at step 4. Notice that the green bar stays at 1.00 while the noise arrives, because the input gate is 0, and that the orange bar is 0 until the output gate opens.
```

Read the panels as three jobs. At step 1 the forget gate is 0, which wipes the 0.7, and the input gate is 1, so the candidate 1 is written. At steps 2 and 3 the forget gate is 1, which keeps the cell, and the input gate is 0, which refuses the noise. At step 4 the output gate opens, and the cell shows $\tanh(1) = 0.76$.

Why three gates and not one? A single gate on the belt could keep or clear the memory, but whatever arrives would always be written in full, noise included. And without an output gate the memory is shown at every step, so the stored number would disturb steps 2 and 3, when the net should be quiet. The **forget gate** says what to erase, the **input gate** says what to write, and the **output gate** says what to show.

## A gate is a learned fraction

A gate is a faucet whose opening the net chooses. For one unit it is a number between 0 and 1 that multiplies something: 0 blocks it, 1 passes it, and 0.5 passes half. The net computes the number from the two things it can see at step $t$, the input $x$ that just arrived and the previous shown value $h$, the **hidden state**, which summarises what the cell has done so far. Each gate sends both through a weighted sum and then a sigmoid, so the result always lands between 0 and 1.

For one unit, take $x = 1$, $h = 0$, input weight 1, hidden weight 0, and bias 1. The sum is $1 \times 1 + 0 \times 0 + 1 = 2$, and $\sigma(2) = 1 / (1 + e^{-2}) = 0.8808$. That gate lets 88 percent through. As a formula, with $\sigma$ for the sigmoid:

$$
\text{gate} = \sigma\left(x W + h U + b\right)
$$

Read it aloud: take the input times its weights, add the previous shown value times its weights, add a bias, and squash the total into the range 0 to 1. In numpy it is one line, `sigmoid(x @ W + h @ U + b)`.

To see what a gate does by default, suppose the weighted parts add to 0. A bias of 0 gives $\sigma(0) = 0.5$, a bias of 1 gives 0.73, and a bias of 2 gives 0.88. Figure 3 marks these on the curve and shows what share of the belt each one keeps.

```figure
<svg viewBox="0 0 620 290" role="img" aria-label="The sigmoid curve from minus six to six with three marked points, bias 0 giving 0.50, bias 1 giving 0.73, and bias 2 giving 0.88, and three bars showing the share of the belt each one keeps">
<line x1="50" y1="220" x2="380" y2="220" class="stroke-grid"/>
<text x="42" y="224" text-anchor="end" class="fig-small">0</text>
<line x1="50" y1="130" x2="380" y2="130" class="stroke-grid"/>
<text x="42" y="134" text-anchor="end" class="fig-small">0.5</text>
<line x1="50" y1="40" x2="380" y2="40" class="stroke-grid"/>
<text x="42" y="44" text-anchor="end" class="fig-small">1</text>
<line x1="50" y1="220" x2="50" y2="225" class="stroke-muted"/>
<text x="50" y="240" text-anchor="middle" class="fig-small">-6</text>
<line x1="132.5" y1="220" x2="132.5" y2="225" class="stroke-muted"/>
<text x="132.5" y="240" text-anchor="middle" class="fig-small">-3</text>
<line x1="215" y1="220" x2="215" y2="225" class="stroke-muted"/>
<text x="215" y="240" text-anchor="middle" class="fig-small">0</text>
<line x1="297.5" y1="220" x2="297.5" y2="225" class="stroke-muted"/>
<text x="297.5" y="240" text-anchor="middle" class="fig-small">3</text>
<line x1="380" y1="220" x2="380" y2="225" class="stroke-muted"/>
<text x="380" y="240" text-anchor="middle" class="fig-small">6</text>
<line x1="50" y1="220" x2="380" y2="220" class="stroke-muted"/>
<line x1="215" y1="40" x2="215" y2="220" class="stroke-grid"/>
<polyline points="50,219.6 52.7,219.5 55.5,219.5 58.2,219.4 61,219.3 63.8,219.3 66.5,219.2 69.2,219.1 72,219 74.8,218.9 77.5,218.8 80.2,218.7 83,218.5 85.8,218.4 88.5,218.2 91.2,218 94,217.8 96.8,217.6 99.5,217.3 102.2,217.1 105,216.8 107.8,216.4 110.5,216.1 113.2,215.7 116,215.2 118.8,214.7 121.5,214.2 124.3,213.6 127,213 129.8,212.2 132.5,211.5 135.2,210.6 138,209.7 140.8,208.7 143.5,207.6 146.2,206.3 149,205 151.8,203.6 154.5,202 157.2,200.4 160,198.5 162.8,196.6 165.5,194.5 168.2,192.2 171,189.8 173.8,187.2 176.5,184.4 179.2,181.5 182,178.3 184.8,175 187.5,171.6 190.2,168 193,164.2 195.8,160.3 198.5,156.2 201.2,152 204,147.8 206.8,143.4 209.5,139 212.3,134.5 215,130 217.7,125.5 220.5,121 223.2,116.6 226,112.2 228.8,108 231.5,103.8 234.2,99.7 237,95.8 239.8,92 242.5,88.4 245.2,85 248,81.7 250.8,78.5 253.5,75.6 256.2,72.8 259,70.2 261.8,67.8 264.5,65.5 267.2,63.4 270,61.5 272.8,59.6 275.5,58 278.2,56.4 281,55 283.8,53.7 286.5,52.4 289.2,51.3 292,50.3 294.8,49.4 297.5,48.5 300.2,47.8 303,47 305.8,46.4 308.5,45.8 311.2,45.3 314,44.8 316.7,44.3 319.5,43.9 322.2,43.6 325,43.2 327.8,42.9 330.5,42.7 333.3,42.4 336,42.2 338.8,42 341.5,41.8 344.2,41.6 347,41.5 349.8,41.3 352.5,41.2 355.2,41.1 358,41 360.8,40.9 363.5,40.8 366.2,40.7 369,40.7 371.7,40.6 374.5,40.5 377.2,40.5 380,40.4" fill="none" class="stroke-a" stroke-width="2.5"/>
<text x="215" y="262" text-anchor="middle" class="fig-small">pre-activation: x W + h U + b  (here with x W + h U = 0, so it equals the bias b)</text>
<text x="42" y="26" class="fig-small">gate value</text>
<circle cx="215" cy="130" r="5" class="fill-b"/>
<line x1="215" y1="130" x2="50" y2="130" class="stroke-b dashed"/>
<text x="223" y="146" class="fig-small">b = 0: 0.50</text>
<circle cx="242.5" cy="88.4" r="5" class="fill-b"/>
<line x1="242.5" y1="88.4" x2="50" y2="88.4" class="stroke-b dashed"/>
<text x="250.5" y="104.4" class="fig-small">b = 1: 0.73</text>
<circle cx="270" cy="61.5" r="5" class="fill-b"/>
<line x1="270" y1="61.5" x2="50" y2="61.5" class="stroke-b dashed"/>
<text x="278" y="81.5" class="fig-small">b = 2: 0.88</text>
<text x="430" y="40" class="fig-bold">share of the belt kept</text>
<text x="430" y="70" class="fig-small">b = 0</text>
<rect x="430" y="78" width="160" height="22" rx="3" class="fill-sunk stroke-grid"/>
<rect x="430" y="78" width="80" height="22" rx="3" class="fill-c"/>
<text x="590" y="74" text-anchor="end" class="fig-small">0.50</text>
<text x="430" y="132" class="fig-small">b = 1</text>
<rect x="430" y="140" width="160" height="22" rx="3" class="fill-sunk stroke-grid"/>
<rect x="430" y="140" width="117" height="22" rx="3" class="fill-c"/>
<text x="590" y="136" text-anchor="end" class="fig-small">0.73</text>
<text x="430" y="194" class="fig-small">b = 2</text>
<rect x="430" y="202" width="160" height="22" rx="3" class="fill-sunk stroke-grid"/>
<rect x="430" y="202" width="140.9" height="22" rx="3" class="fill-c"/>
<text x="590" y="198" text-anchor="end" class="fig-small">0.88</text>
</svg>
Figure 3. The sigmoid, with the gate values for bias 0, 1, and 2 when the weighted parts add to 0. Notice that the bias slides the gate toward open: at 0 the belt loses half of its memory every step, and at 2 it keeps 0.88.
```

This is why the forget bias starts at 1 or 2 in practice. At a bias of 0, a fresh network halves its memory at every step, and $0.5^{10} = 0.001$ of it remains after 10 steps, which is no better than a plain net. A bias of 2 starts the belt mostly open, with $0.88^{10} = 0.28$ left after 10 steps, so the net begins with a memory and learns when to close the gate.

## Candidate and cell

The gates only scale things. The content they scale is the **candidate**, $g$, a proposal for what to write. It has the same form as a gate, a weighted sum of $x$ and $h$, but it uses tanh instead of the sigmoid. A gate is a fraction to multiply by, so it must lie between 0 and 1. Content needs a sign, since a memory must be able to hold a negative value or subtract from a positive one, and with a sigmoid candidate the cell could only ever add positive numbers.

The memory itself is the **cell state** $c$, which is the belt. The cell state and the shown value $h$ are different on purpose: $c$ is private, long-term memory that is never squashed on its way forward, and $h$ is the public, squashed view of it that the next layer and the next step's gates see.

## One step with real numbers

Take one unit. The old cell is $c = 0.4$, the candidate is $g = 0.6$, and the gates are $f = 0.9$ for forget, $i = 0.2$ for input, and $o = 0.5$ for output. Figure 4 follows the numbers.

```figure
<svg viewBox="0 0 620 250" role="img" aria-label="One LSTM unit with numbers on the arrows: the old cell 0.4 times the forget gate 0.9 gives 0.36, the candidate 0.6 times the input gate 0.2 gives 0.12, they add to 0.48, tanh gives 0.4462, and the output gate 0.5 gives a shown value of 0.2231">
<defs><marker id="c039-fig4-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker><marker id="c039-fig4-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<rect x="10" y="80" width="80" height="40" rx="6" class="fill-soft-a stroke-a"/>
<text x="50" y="105" text-anchor="middle">c = 0.4</text>
<line x1="90" y1="100" x2="123" y2="100" class="stroke-ink" marker-end="url(#c039-fig4-arrow)"/>
<circle cx="140" cy="100" r="15" class="fill-paper stroke-ink"/>
<text x="140" y="105" text-anchor="middle" class="fig-bold">×</text>
<text x="140" y="48" text-anchor="middle" class="fig-small">forget gate</text>
<text x="140" y="64" text-anchor="middle" class="fig-bold">f = 0.9</text>
<line x1="140" y1="70" x2="140" y2="83" class="stroke-b" marker-end="url(#c039-fig4-arrow-b)"/>
<line x1="155" y1="100" x2="223" y2="100" class="stroke-ink" marker-end="url(#c039-fig4-arrow)"/>
<text x="190" y="92" text-anchor="middle" class="fig-mono">0.36</text>
<circle cx="240" cy="100" r="15" class="fill-paper stroke-ink"/>
<text x="240" y="105" text-anchor="middle" class="fig-bold">+</text>
<rect x="90" y="165" width="80" height="30" rx="6" class="fill-soft-c stroke-c"/>
<text x="130" y="185" text-anchor="middle">g = 0.6</text>
<text x="130" y="212" text-anchor="middle" class="fig-small">candidate</text>
<line x1="170" y1="180" x2="223" y2="180" class="stroke-ink" marker-end="url(#c039-fig4-arrow)"/>
<circle cx="240" cy="180" r="15" class="fill-paper stroke-ink"/>
<text x="240" y="185" text-anchor="middle" class="fig-bold">×</text>
<line x1="240" y1="165" x2="240" y2="117" class="stroke-ink" marker-end="url(#c039-fig4-arrow)"/>
<text x="254" y="146" class="fig-mono">0.12</text>
<text x="240" y="232" text-anchor="middle" class="fig-bold">input gate  i = 0.2</text>
<line x1="240" y1="218" x2="240" y2="197" class="stroke-b" marker-end="url(#c039-fig4-arrow-b)"/>
<line x1="255" y1="100" x2="303" y2="100" class="stroke-ink" marker-end="url(#c039-fig4-arrow)"/>
<rect x="305" y="80" width="90" height="40" rx="6" class="fill-soft-d stroke-d"/>
<text x="350" y="105" text-anchor="middle">c_new = 0.48</text>
<line x1="395" y1="100" x2="610" y2="100" class="stroke-ink" marker-end="url(#c039-fig4-arrow)"/>
<text x="500" y="92" text-anchor="middle" class="fig-small">to the next step</text>
<line x1="350" y1="120" x2="350" y2="163" class="stroke-ink" marker-end="url(#c039-fig4-arrow)"/>
<rect x="305" y="165" width="90" height="30" rx="6" class="fill-soft-a stroke-a"/>
<text x="350" y="185" text-anchor="middle">tanh</text>
<line x1="395" y1="180" x2="443" y2="180" class="stroke-ink" marker-end="url(#c039-fig4-arrow)"/>
<text x="428" y="160" text-anchor="middle" class="fig-mono">0.4462</text>
<circle cx="460" cy="180" r="15" class="fill-paper stroke-ink"/>
<text x="460" y="185" text-anchor="middle" class="fig-bold">×</text>
<text x="470" y="232" text-anchor="middle" class="fig-bold">output gate  o = 0.5</text>
<line x1="460" y1="218" x2="460" y2="197" class="stroke-b" marker-end="url(#c039-fig4-arrow-b)"/>
<line x1="475" y1="180" x2="520" y2="180" class="stroke-ink" marker-end="url(#c039-fig4-arrow)"/>
<rect x="522" y="165" width="90" height="30" rx="6" class="fill-soft-b stroke-b"/>
<text x="567" y="185" text-anchor="middle">h = 0.2231</text>
</svg>
Figure 4. One LSTM unit with its numbers. The forget gate keeps 0.36 of the old cell, the input gate lets in 0.12 of the candidate, and they add to the new cell 0.48. The output gate then scales tanh of the cell to give the shown value 0.2231.
```

Keep most of the old cell: $0.9 \times 0.4 = 0.36$. Let in a little of the candidate: $0.2 \times 0.6 = 0.12$. The new cell is $0.36 + 0.12 = 0.48$. Squash it, $\tanh(0.48) = 0.4462$, and show half of it, $0.5 \times 0.4462 = 0.2231$. In symbols, for a whole vector of units at once:

$$
c_{\text{new}} = f \odot c + i \odot g, \qquad
h_{\text{new}} = o \odot \tanh(c_{\text{new}})
$$

Read it aloud: the new cell is the kept share of the old cell plus the written share of the candidate, and the shown value is the output gate times the squashed cell. The four quantities come from the same inputs:

$$
\begin{aligned}
f &= \sigma(x W_f + h U_f + b_f) && \text{forget gate} \\
i &= \sigma(x W_i + h U_i + b_i) && \text{input gate} \\
o &= \sigma(x W_o + h U_o + b_o) && \text{output gate} \\
g &= \tanh(x W_g + h U_g + b_g) && \text{candidate}
\end{aligned}
$$

Each line has its own weights and its own bias. In numpy:

```python
c_new = f * c + i * g
h_new = o * np.tanh(c_new)
```

The most common mistake is putting tanh on a gate or a sigmoid on the candidate. For a batch of $n$ examples, `x` is `(n, d_x)`, `h` and `c` are `(n, d_h)`, each `W` is `(d_x, d_h)`, each `U` is `(d_h, d_h)`, each `b` is `(d_h,)`, and each gate is `(n, d_h)`.

> [!WARNING]
> The feed-forward chapters wrote `x @ W.T`, with `W` shaped `(out, in)`. The LSTM weights here are shaped `(in, out)`, so you write `x @ W` with no transpose. A transpose runs without an error whenever the input and hidden widths match, and gives wrong numbers.

> [!KEY]
> Who decides when to erase, write, and show? Three gates, each a learned fraction between 0 and 1 computed from the input and the last shown value. The forget gate scales the old cell, the input gate scales the candidate, and the output gate scales what the cell shows, while the belt itself stays a plain sum with no weight matrix on it.

> [!RECAP]
> - A plain net multiplies the gradient by a factor per step, so with factor 0.45 it keeps $0.45^k$ of it. A belt multiplies the memory by a gate, and across $k$ links the share left is the product of the gates, which `np.cumprod` computes along time.
> - A gate is $\sigma(xW + hU + b)$, a fraction between 0 and 1 for every hidden unit, with `W` shaped `(d_x, d_h)` and no transpose.
> - Forget erases old memory, input writes the candidate, and output decides how much of the cell to show.
> - The candidate uses tanh because content needs a sign. The gates use the sigmoid because a fraction cannot be negative.
> - A forget bias of 1 or 2 starts the belt mostly open.

## Build it step by step

The checks test five functions in this order. The stable `_sigmoid` is provided.

1. **`gradient_after_steps(factor, steps)`.** Takes a factor and a number of steps, and returns an array of length `steps` whose entry $t - 1$ is `factor ** t`. Build the exponents with `np.arange(1, steps + 1)`. The most common mistake is starting the exponents at 0, which makes the array begin with 1 and end one step short.
2. **`forget_gate_gradient_product(forget_gates)`.** Takes gates of shape `(T, hidden_size)` and returns the same shape, where row $t$ holds the product of rows 0 through $t$. Use `np.cumprod` along axis 0. The most common mistake is `np.prod`, which returns one row.
3. **`gate(x, h, W, U, b)`.** Takes the input `(n, d_x)`, the previous shown value `(n, d_h)`, and one gate's weights, and returns an array of shape `(n, d_h)` with every entry between 0 and 1. Use `x @ W + h @ U + b` and the provided `_sigmoid`. The most common mistake is transposing `W`.
4. **`carry_cell(c, steps, f, i, g)`.** Applies `c = f * c + i * g` `steps` times with the gates held fixed, and returns the final cell. A loop is the clearest form. The most common mistake is reading `c` from the wrong place: the update uses the cell from the previous pass of the loop, not the starting one.
5. **`lstm_cell_forward(x, h, c, params)`.** Returns `(h_new, c_new, cache)`. Build `f`, `i`, and `o` with `gate`, build `g` with tanh, then apply the two update lines. Store `x`, `h`, `c`, `f`, `i`, `o`, `g`, `c_new`, `tanh_c_new`, and `params` in the cache. The most common mistake is forgetting `tanh_c_new`, which the backward pass needs.

## Where this breaks

A belt with gates near one slows the fade without removing it. A gate of 0.99 keeps $0.99^{1000} = 4.3 \times 10^{-5}$ of the gradient over 1,000 steps, so very long memories are still hard, and nothing forces the gates to stay near one while the net learns. A gate sees only `x` and `h`, never the cell `c` itself, so a unit cannot open its output gate because of a value stored in its own cell, except through what it showed earlier. An open input gate with candidates of one sign also lets the cell grow without limit, which is why practitioners still clip gradients. [Attention](/lessons/051-attention-as-soft-lookup) gives the net a direct path to every earlier step so that distance stops mattering.

## The exercise

You build `gradient_after_steps`, `forget_gate_gradient_product`, `gate`, `carry_cell`, and `lstm_cell_forward`. The checks compare the plain-net curve with repeated multiplication, match the running product to a loop, and require that a belt of gates at 0.995 keeps more than 0.3 of the gradient at step 200 while a plain net at 0.45 keeps under $10^{-60}$. They confirm that a gate stays strictly between 0 and 1 and gives 0.8808 on the worked example, that a forget gate of 1 and an input gate of 0 hold a cell exactly while a forget gate of 0 replaces it, that the forward step gives a new cell of 0.48 and a shown value of 0.2231 on the worked numbers, and that it matches the gate equations on random weights.
