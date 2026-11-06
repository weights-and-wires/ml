---
title: Why a recurrent net forgets, and the belt that fixes it
slug: 080-why-rnns-forget-the-belt
part: "Part 5 — Vision and sequences"
summary: A gradient through many steps is a product of one factor per step, so a path whose factor is a gate near one lets it survive where a plain net loses it.
minutes: 35
gpu: false
objectives:
  - Explain why the gradient through many recurrent steps is a product with one factor per step, and why factors below one make it vanish.
  - Work a four-step chain by hand for a plain net with factor 0.45 and for a belt with a gate of 0.9.
  - Say why a bigger recurrent weight or adding the input to the memory does not fix the problem, and why a gate on a belt does.
  - Compute how many steps a gradient survives for any per-step factor, and the running product of forget gates with np.cumprod.
prereqs:
  - 077-backprop-through-time
  - 078-vanishing-and-exploding-gradients
  - 079-truncation-and-clipping
exercise: 080-why-rnns-forget-the-belt
---

# Why a recurrent net forgets, and the belt that fixes it

Imagine a net reading a review one word at a time. At step 2 the review says "I bought this for my daughter". At step 200, after a long list of complaints about the shipping, it ends "so I would not buy this for ___ again", and the right word ("her") depends on what the net read at step 2. To learn that, the net needs a training signal that travels from step 200 back to step 2 and says "the word you read there mattered".

A plain recurrent net cannot carry that signal. By the time it has travelled back 200 steps, it is smaller than the smallest number a computer can usefully store, so the weights that handled step 2 never hear about the mistake at step 200. The net can learn from the last few words and almost nothing earlier. The chapter "Why gradients fade or blow up" showed this with matrices. This chapter shows it with one number per step, and then changes the path so the number stays close to one.

The change is a belt that carries the memory straight along, with a gate on it, a number between 0 and 1 that the net chooses. You build the running product that measures how much gradient a belt keeps, and a function that tells you how far a gradient can reach. The next chapter opens up the gates themselves.

## What you already know

Five ideas carry this chapter:

- A recurrent cell reuses one set of weights at every step and passes a hidden state forward (see "A network that reads one step at a time").
- The gradient is the slope of the loss, which says how much the loss changes when a number is nudged.
- Backpropagation through time sends the gradient back one step at a time, multiplying by a local slope at each link (see "Walk the chain backward: backpropagation through time").
- The tanh squashes any number into the range from −1 to 1, and its slope is never above 1.
- Clipping shortens a gradient that is too long, but cannot lengthen one that has faded (see "Taming long sequences: truncation and clipping").

## One factor per step

Picture the net as a chain of steps. The loss sits at step $t$, and you want to know how much it cares about the hidden state $k$ steps earlier. Each link in the chain passes the gradient back with a multiplier, because of the chain rule: the slope across two links is the slope across the first times the slope across the second. So the gradient that reaches $k$ steps back is the starting gradient times one factor per link.

What is that factor for a plain net? Take one hidden unit with $h_t = \tanh(w\,h_{t-1} + \dots)$. The slope of $h_t$ with respect to $h_{t-1}$ is $w$ times the slope of tanh. The tanh slope is at most 1 and is smaller when the unit is far from zero. As an illustration, take a typical slope of 0.5 and a recurrent weight of 0.9, so the factor is $0.5 \times 0.9 = 0.45$. This is an illustrative choice, not a law, but any factor below one behaves the same way.

Figure 1 draws two chains of five steps with the same starting gradient of 1 at step $t$. In the top chain each link multiplies by 0.45.

```figure
<svg viewBox="0 0 600 345" role="img" aria-label="Two chains of five steps. Going back from the loss, a plain net multiplies the gradient by 0.45 at every link and a belt multiplies it by 0.9. Bars show the gradient left at each step.">
<defs><marker id="c089-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker><marker id="c089-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
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
<line x1="104" y1="62" x2="166" y2="62" class="stroke-b" marker-end="url(#c089-arrow-b)"/>
<text x="135" y="52" text-anchor="middle" class="fig-small">x0.45</text>
<circle cx="300" cy="62" r="22" class="fill-soft-a stroke-a"/>
<text x="300" y="67" text-anchor="middle" class="fig-small">t-2</text>
<rect x="284" y="124.9" width="32" height="10.1" class="fill-bad"/>
<line x1="276" y1="135" x2="324" y2="135" class="stroke-muted"/>
<text x="300" y="153" text-anchor="middle" class="fig-mono">0.20</text>
<line x1="214" y1="62" x2="276" y2="62" class="stroke-b" marker-end="url(#c089-arrow-b)"/>
<text x="245" y="52" text-anchor="middle" class="fig-small">x0.45</text>
<circle cx="410" cy="62" r="22" class="fill-soft-a stroke-a"/>
<text x="410" y="67" text-anchor="middle" class="fig-small">t-3</text>
<rect x="394" y="130.4" width="32" height="4.6" class="fill-bad"/>
<line x1="386" y1="135" x2="434" y2="135" class="stroke-muted"/>
<text x="410" y="153" text-anchor="middle" class="fig-mono">0.09</text>
<line x1="324" y1="62" x2="386" y2="62" class="stroke-b" marker-end="url(#c089-arrow-b)"/>
<text x="355" y="52" text-anchor="middle" class="fig-small">x0.45</text>
<circle cx="520" cy="62" r="22" class="fill-soft-a stroke-a"/>
<text x="520" y="67" text-anchor="middle" class="fig-small">t-4</text>
<rect x="504" y="132.9" width="32" height="2.1" class="fill-bad"/>
<line x1="496" y1="135" x2="544" y2="135" class="stroke-muted"/>
<text x="520" y="153" text-anchor="middle" class="fig-mono">0.04</text>
<line x1="434" y1="62" x2="496" y2="62" class="stroke-b" marker-end="url(#c089-arrow-b)"/>
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
<line x1="104" y1="237" x2="166" y2="237" class="stroke-b" marker-end="url(#c089-arrow-b)"/>
<text x="135" y="227" text-anchor="middle" class="fig-small">x0.9</text>
<circle cx="300" cy="237" r="22" class="fill-soft-a stroke-a"/>
<text x="300" y="242" text-anchor="middle" class="fig-small">t-2</text>
<rect x="284" y="269.5" width="32" height="40.5" class="fill-c"/>
<line x1="276" y1="310" x2="324" y2="310" class="stroke-muted"/>
<text x="300" y="328" text-anchor="middle" class="fig-mono">0.81</text>
<line x1="214" y1="237" x2="276" y2="237" class="stroke-b" marker-end="url(#c089-arrow-b)"/>
<text x="245" y="227" text-anchor="middle" class="fig-small">x0.9</text>
<circle cx="410" cy="237" r="22" class="fill-soft-a stroke-a"/>
<text x="410" y="242" text-anchor="middle" class="fig-small">t-3</text>
<rect x="394" y="273.6" width="32" height="36.5" class="fill-c"/>
<line x1="386" y1="310" x2="434" y2="310" class="stroke-muted"/>
<text x="410" y="328" text-anchor="middle" class="fig-mono">0.73</text>
<line x1="324" y1="237" x2="386" y2="237" class="stroke-b" marker-end="url(#c089-arrow-b)"/>
<text x="355" y="227" text-anchor="middle" class="fig-small">x0.9</text>
<circle cx="520" cy="237" r="22" class="fill-soft-a stroke-a"/>
<text x="520" y="242" text-anchor="middle" class="fig-small">t-4</text>
<rect x="504" y="277.2" width="32" height="32.8" class="fill-c"/>
<line x1="496" y1="310" x2="544" y2="310" class="stroke-muted"/>
<text x="520" y="328" text-anchor="middle" class="fig-mono">0.66</text>
<line x1="434" y1="237" x2="496" y2="237" class="stroke-b" marker-end="url(#c089-arrow-b)"/>
<text x="465" y="227" text-anchor="middle" class="fig-small">x0.9</text>
</svg>
Figure 1. The gradient flows from step t back to earlier steps, and each bar shows how much is left on arrival. The plain chain loses more than half at every link and is down to 0.04 after four. Notice that the belt chain, with a factor of 0.9, still holds 0.66 after the same four links.
```

Read the top chain from left to right: 1, then $1 \times 0.45 = 0.45$, then $0.45 \times 0.45 = 0.2025$, then 0.0911, then 0.0410. After four links 4 percent of the gradient is left. After 10 links it is $0.45^{10} = 0.00034$, and after 200 it is about $4 \times 10^{-70}$. In general, after $k$ links the share left is

$$
\text{share left after } k \text{ steps} = \text{factor}^{k}
$$

Read it aloud: the share of the gradient that survives $k$ links is the per-link factor multiplied by itself $k$ times. A number below one, multiplied by itself over and over, heads to zero. A number above one heads to infinity, which is the explosion that the earlier chapter on clipping dealt with. In numpy, `factor ** np.arange(1, steps + 1)` gives the share left after every number of links.

## Why not just make the weight bigger

If 0.45 is too small, raise the recurrent weight until the factor is 1. The trouble is that a factor of exactly 1 is a knife edge. The tanh slope changes with the input, so the factor is sometimes 0.2 and sometimes 0.9, and no single weight keeps it at 1. Push the weight up until the typical factor is 1, and the steps where the slope is larger have a factor above 1. A factor of 1.1 compounds to $1.1^{200} \approx 1.9 \times 10^{8}$ over 200 steps, so you trade a fading gradient for an exploding one. The same weight also has to compute with the state and remember it, so tuning it for one job damages the other.

A second idea is to stop redrawing the memory. Let the memory be a number $c$ and add each input to it: $c_t = c_{t-1} + x_t$. The gradient along this path has a factor of exactly 1, so it survives. But the memory never lets go of anything. Every input is added forever, so it grows without limit, and the net cannot clear a value once it stops being useful.

The fix keeps the first idea, an unchanged path for the memory, and adds the missing control: a number that decides, at each step, how much of the old memory to keep.

## A belt with a gate on it

Call the memory the **belt**, and give it a **gate**: a number $f_t$ between 0 and 1 that multiplies the old memory before the fresh content is added.

$$
c_t = f_t \cdot c_{t-1} + (\text{fresh content})
$$

Read it aloud: the memory at step $t$ is a fraction $f_t$ of the old memory, plus whatever fresh content this step writes. A gate of 1 keeps everything, a gate of 0 wipes the belt, and a gate of 0.5 halves it. The net computes $f_t$ from what it reads, so it can keep the memory through the shipping complaints and clear it when a fresh sentence begins. Figure 2 sets the two cells side by side.

```figure
<svg viewBox="0 0 620 305" role="img" aria-label="A plain recurrent cell, where the state passes through a weight matrix and tanh at every step, above a cell with a belt, where the memory passes through only a gate of 0.9 at every step">
<defs><marker id="c089-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker><marker id="c089-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="20" y="22" class="fig-bold">Plain cell: the state is redrawn at every step</text>
<circle cx="60" cy="80" r="20" class="fill-soft-a stroke-a"/>
<text x="60" y="85" text-anchor="middle" class="fig-small">h1</text>
<circle cx="200" cy="80" r="20" class="fill-soft-a stroke-a"/>
<text x="200" y="85" text-anchor="middle" class="fig-small">h2</text>
<line x1="80" y1="74" x2="180" y2="74" class="stroke-ink" marker-end="url(#c089-arrow)"/>
<rect x="92" y="36" width="76" height="24" rx="5" class="fill-soft-b stroke-b"/>
<text x="130" y="53" text-anchor="middle" class="fig-small">W, tanh</text>
<line x1="180" y1="100" x2="80" y2="100" class="stroke-b" marker-end="url(#c089-arrow-b)"/>
<text x="130" y="120" text-anchor="middle" class="fig-small">x0.45</text>
<circle cx="340" cy="80" r="20" class="fill-soft-a stroke-a"/>
<text x="340" y="85" text-anchor="middle" class="fig-small">h3</text>
<line x1="220" y1="74" x2="320" y2="74" class="stroke-ink" marker-end="url(#c089-arrow)"/>
<rect x="232" y="36" width="76" height="24" rx="5" class="fill-soft-b stroke-b"/>
<text x="270" y="53" text-anchor="middle" class="fig-small">W, tanh</text>
<line x1="320" y1="100" x2="220" y2="100" class="stroke-b" marker-end="url(#c089-arrow-b)"/>
<text x="270" y="120" text-anchor="middle" class="fig-small">x0.45</text>
<circle cx="480" cy="80" r="20" class="fill-soft-a stroke-a"/>
<text x="480" y="85" text-anchor="middle" class="fig-small">h4</text>
<line x1="360" y1="74" x2="460" y2="74" class="stroke-ink" marker-end="url(#c089-arrow)"/>
<rect x="372" y="36" width="76" height="24" rx="5" class="fill-soft-b stroke-b"/>
<text x="410" y="53" text-anchor="middle" class="fig-small">W, tanh</text>
<line x1="460" y1="100" x2="360" y2="100" class="stroke-b" marker-end="url(#c089-arrow-b)"/>
<text x="410" y="120" text-anchor="middle" class="fig-small">x0.45</text>
<text x="515" y="70" class="fig-small">every link</text>
<text x="515" y="88" class="fig-small">shrinks the</text>
<text x="515" y="106" class="fig-small">gradient</text>
<text x="20" y="175" class="fig-bold">Cell with a belt: the memory passes through a gate</text>
<line x1="30" y1="235" x2="590" y2="235" class="stroke-c" marker-end="url(#c089-arrow)" stroke-width="3"/>
<text x="34" y="222" class="fig-small">belt (memory)</text>
<circle cx="150" cy="235" r="18" class="fill-paper stroke-ink"/>
<text x="150" y="240" text-anchor="middle" class="fig-small">0.9</text>
<text x="150" y="205" text-anchor="middle" class="fig-small">gate</text>
<circle cx="290" cy="235" r="18" class="fill-paper stroke-ink"/>
<text x="290" y="240" text-anchor="middle" class="fig-small">0.9</text>
<text x="290" y="205" text-anchor="middle" class="fig-small">gate</text>
<circle cx="430" cy="235" r="18" class="fill-paper stroke-ink"/>
<text x="430" y="240" text-anchor="middle" class="fig-small">0.9</text>
<text x="430" y="205" text-anchor="middle" class="fig-small">gate</text>
<line x1="570" y1="268" x2="448" y2="268" class="stroke-b" marker-end="url(#c089-arrow-b)"/>
<text x="509" y="288" text-anchor="middle" class="fig-small">x0.9</text>
<line x1="412" y1="268" x2="308" y2="268" class="stroke-b" marker-end="url(#c089-arrow-b)"/>
<text x="360" y="288" text-anchor="middle" class="fig-small">x0.9</text>
<line x1="272" y1="268" x2="168" y2="268" class="stroke-b" marker-end="url(#c089-arrow-b)"/>
<text x="220" y="288" text-anchor="middle" class="fig-small">x0.9</text>
<text x="34" y="272" class="fig-small">gradient</text>
<text x="34" y="288" class="fig-small">flows back</text>
</svg>
Figure 2. The same backward trip through three links in two cells. In the plain cell the state passes through a weight matrix and tanh, which multiplies the gradient by 0.45 each time. In the cell with a belt the memory passes through only a gate, here 0.9, so the gradient is multiplied by the gate and nothing else.
```

Next, ask what the gradient does across one link of the belt. The fresh content does not depend on $c_{t-1}$, so the only place $c_{t-1}$ enters $c_t$ is through the product $f_t \cdot c_{t-1}$. The slope of $f_t \cdot c_{t-1}$ with respect to $c_{t-1}$ is $f_t$. That is the whole factor, with no weight matrix and no tanh slope.

Three gates of 0.9 in a row give $0.9$, then $0.9 \times 0.9 = 0.81$, then $0.81 \times 0.9 = 0.729$. The belt still holds 73 percent after three links, where the plain net held 9 percent. When the gates differ from step to step, the share left after $k$ links is their product:

$$
f_1 \cdot f_2 \cdots f_k
$$

In numpy, a running product along the time axis is `np.cumprod(forget_gates, axis=0)`. It keeps every row, where `np.prod` keeps only the last. The path is named the **constant error carousel**: the error, meaning the gradient, stays roughly constant as it rides round and round the belt.

> [!WARNING]
> The product above is the gradient along the belt only. The gates are computed from the previous hidden state, which depends on the previous memory, so a second, smaller path exists. Gating slows the fade a great deal, but does not remove it.

## How far can a gradient reach

Figure 3 plots the share left against the number of steps for four factors, on a log scale, where equal steps down mean equal factors of ten.

```figure
<svg viewBox="0 0 620 295" role="img" aria-label="Share of gradient left against steps back on a log scale, for a plain net with factor 0.45 and gates of 0.9, 0.99, and 0.995">
<line x1="70" y1="40" x2="460" y2="40" class="stroke-grid"/>
<text x="62" y="44" text-anchor="end" class="fig-small">1</text>
<line x1="70" y1="82" x2="460" y2="82" class="stroke-grid"/>
<text x="62" y="86" text-anchor="end" class="fig-small">1e-2</text>
<line x1="70" y1="124" x2="460" y2="124" class="stroke-grid"/>
<text x="62" y="128" text-anchor="end" class="fig-small">1e-4</text>
<line x1="70" y1="166" x2="460" y2="166" class="stroke-grid"/>
<text x="62" y="170" text-anchor="end" class="fig-small">1e-6</text>
<line x1="70" y1="208" x2="460" y2="208" class="stroke-grid"/>
<text x="62" y="212" text-anchor="end" class="fig-small">1e-8</text>
<line x1="70" y1="250" x2="460" y2="250" class="stroke-grid"/>
<text x="62" y="254" text-anchor="end" class="fig-small">1e-10</text>
<line x1="70" y1="250" x2="70" y2="255" class="stroke-muted"/>
<text x="70" y="270" text-anchor="middle" class="fig-small">0</text>
<line x1="167.5" y1="250" x2="167.5" y2="255" class="stroke-muted"/>
<text x="167.5" y="270" text-anchor="middle" class="fig-small">50</text>
<line x1="265" y1="250" x2="265" y2="255" class="stroke-muted"/>
<text x="265" y="270" text-anchor="middle" class="fig-small">100</text>
<line x1="362.5" y1="250" x2="362.5" y2="255" class="stroke-muted"/>
<text x="362.5" y="270" text-anchor="middle" class="fig-small">150</text>
<line x1="460" y1="250" x2="460" y2="255" class="stroke-muted"/>
<text x="460" y="270" text-anchor="middle" class="fig-small">200</text>
<line x1="70" y1="250" x2="460" y2="250" class="stroke-muted"/>
<line x1="70" y1="40" x2="70" y2="250" class="stroke-muted"/>
<text x="265" y="290" text-anchor="middle" class="fig-small">steps back</text>
<text x="62" y="26" class="fig-small">share of gradient left (log scale)</text>
<polyline points="70,40 72,47.3 73.9,54.6 75.8,61.8 77.8,69.1 79.8,76.4 81.7,83.7 83.7,91 85.6,98.3 87.5,105.5 89.5,112.8 91.5,120.1 93.4,127.4 95.3,134.7 97.3,142 99.2,149.2 101.2,156.5 103.2,163.8 105.1,171.1 107,178.4 109,185.7 111,192.9 112.9,200.2 114.8,207.5 116.8,214.8 118.8,222.1 120.7,229.3 122.7,236.6 124.6,243.9" fill="none" class="stroke-bad" stroke-width="2.5"/>
<polyline points="70,40 72,41 73.9,41.9 75.8,42.9 77.8,43.8 79.8,44.8 81.7,45.8 83.7,46.7 85.6,47.7 87.5,48.6 89.5,49.6 91.5,50.6 93.4,51.5 95.3,52.5 97.3,53.5 99.2,54.4 101.2,55.4 103.2,56.3 105.1,57.3 107,58.3 109,59.2 111,60.2 112.9,61.1 114.8,62.1 116.8,63.1 118.8,64 120.7,65 122.7,65.9 124.6,66.9 126.5,67.9 128.5,68.8 130.4,69.8 132.4,70.7 134.3,71.7 136.3,72.7 138.2,73.6 140.2,74.6 142.2,75.6 144.1,76.5 146.1,77.5 148,78.4 149.9,79.4 151.9,80.4 153.8,81.3 155.8,82.3 157.8,83.2 159.7,84.2 161.7,85.2 163.6,86.1 165.6,87.1 167.5,88 169.4,89 171.4,90 173.3,90.9 175.3,91.9 177.2,92.8 179.2,93.8 181.2,94.8 183.1,95.7 185.1,96.7 187,97.7 188.9,98.6 190.9,99.6 192.8,100.5 194.8,101.5 196.8,102.5 198.7,103.4 200.7,104.4 202.6,105.3 204.6,106.3 206.5,107.3 208.4,108.2 210.4,109.2 212.3,110.1 214.3,111.1 216.2,112.1 218.2,113 220.2,114 222.1,115 224.1,115.9 226,116.9 227.9,117.8 229.9,118.8 231.8,119.8 233.8,120.7 235.8,121.7 237.7,122.6 239.7,123.6 241.6,124.6 243.6,125.5 245.5,126.5 247.4,127.4 249.4,128.4 251.3,129.4 253.3,130.3 255.2,131.3 257.2,132.2 259.1,133.2 261.1,134.2 263.1,135.1 265,136.1 266.9,137.1 268.9,138 270.9,139 272.8,139.9 274.8,140.9 276.7,141.9 278.6,142.8 280.6,143.8 282.6,144.7 284.5,145.7 286.4,146.7 288.4,147.6 290.4,148.6 292.3,149.5 294.2,150.5 296.2,151.5 298.1,152.4 300.1,153.4 302.1,154.3 304,155.3 305.9,156.3 307.9,157.2 309.9,158.2 311.8,159.2 313.8,160.1 315.7,161.1 317.6,162 319.6,163 321.6,164 323.5,164.9 325.4,165.9 327.4,166.8 329.4,167.8 331.3,168.8 333.2,169.7 335.2,170.7 337.1,171.6 339.1,172.6 341.1,173.6 343,174.5 344.9,175.5 346.9,176.4 348.9,177.4 350.8,178.4 352.8,179.3 354.7,180.3 356.6,181.3 358.6,182.2 360.6,183.2 362.5,184.1 364.4,185.1 366.4,186.1 368.4,187 370.3,188 372.2,188.9 374.2,189.9 376.1,190.9 378.1,191.8 380.1,192.8 382,193.7 383.9,194.7 385.9,195.7 387.9,196.6 389.8,197.6 391.8,198.5 393.7,199.5 395.6,200.5 397.6,201.4 399.6,202.4 401.5,203.4 403.4,204.3 405.4,205.3 407.4,206.2 409.3,207.2 411.2,208.2 413.2,209.1 415.1,210.1 417.1,211 419.1,212 421,213 422.9,213.9 424.9,214.9 426.9,215.8 428.8,216.8 430.8,217.8 432.7,218.7 434.6,219.7 436.6,220.7 438.6,221.6 440.5,222.6 442.4,223.5 444.4,224.5 446.4,225.5 448.3,226.4 450.2,227.4 452.2,228.3 454.1,229.3 456.1,230.3 458.1,231.2 460,232.2" fill="none" class="stroke-b" stroke-width="2.5"/>
<polyline points="70,40 72,40.1 73.9,40.2 75.8,40.3 77.8,40.4 79.8,40.5 81.7,40.5 83.7,40.6 85.6,40.7 87.5,40.8 89.5,40.9 91.5,41 93.4,41.1 95.3,41.2 97.3,41.3 99.2,41.4 101.2,41.5 103.2,41.6 105.1,41.6 107,41.7 109,41.8 111,41.9 112.9,42 114.8,42.1 116.8,42.2 118.8,42.3 120.7,42.4 122.7,42.5 124.6,42.6 126.5,42.7 128.5,42.7 130.4,42.8 132.4,42.9 134.3,43 136.3,43.1 138.2,43.2 140.2,43.3 142.2,43.4 144.1,43.5 146.1,43.6 148,43.7 149.9,43.8 151.9,43.8 153.8,43.9 155.8,44 157.8,44.1 159.7,44.2 161.7,44.3 163.6,44.4 165.6,44.5 167.5,44.6 169.4,44.7 171.4,44.8 173.3,44.9 175.3,44.9 177.2,45 179.2,45.1 181.2,45.2 183.1,45.3 185.1,45.4 187,45.5 188.9,45.6 190.9,45.7 192.8,45.8 194.8,45.9 196.8,46 198.7,46 200.7,46.1 202.6,46.2 204.6,46.3 206.5,46.4 208.4,46.5 210.4,46.6 212.3,46.7 214.3,46.8 216.2,46.9 218.2,47 220.2,47.1 222.1,47.1 224.1,47.2 226,47.3 227.9,47.4 229.9,47.5 231.8,47.6 233.8,47.7 235.8,47.8 237.7,47.9 239.7,48 241.6,48.1 243.6,48.2 245.5,48.2 247.4,48.3 249.4,48.4 251.3,48.5 253.3,48.6 255.2,48.7 257.2,48.8 259.1,48.9 261.1,49 263.1,49.1 265,49.2 266.9,49.3 268.9,49.3 270.9,49.4 272.8,49.5 274.8,49.6 276.7,49.7 278.6,49.8 280.6,49.9 282.6,50 284.5,50.1 286.4,50.2 288.4,50.3 290.4,50.4 292.3,50.4 294.2,50.5 296.2,50.6 298.1,50.7 300.1,50.8 302.1,50.9 304,51 305.9,51.1 307.9,51.2 309.9,51.3 311.8,51.4 313.8,51.5 315.7,51.5 317.6,51.6 319.6,51.7 321.6,51.8 323.5,51.9 325.4,52 327.4,52.1 329.4,52.2 331.3,52.3 333.2,52.4 335.2,52.5 337.1,52.6 339.1,52.6 341.1,52.7 343,52.8 344.9,52.9 346.9,53 348.9,53.1 350.8,53.2 352.8,53.3 354.7,53.4 356.6,53.5 358.6,53.6 360.6,53.7 362.5,53.7 364.4,53.8 366.4,53.9 368.4,54 370.3,54.1 372.2,54.2 374.2,54.3 376.1,54.4 378.1,54.5 380.1,54.6 382,54.7 383.9,54.8 385.9,54.8 387.9,54.9 389.8,55 391.8,55.1 393.7,55.2 395.6,55.3 397.6,55.4 399.6,55.5 401.5,55.6 403.4,55.7 405.4,55.8 407.4,55.9 409.3,55.9 411.2,56 413.2,56.1 415.1,56.2 417.1,56.3 419.1,56.4 421,56.5 422.9,56.6 424.9,56.7 426.9,56.8 428.8,56.9 430.8,57 432.7,57 434.6,57.1 436.6,57.2 438.6,57.3 440.5,57.4 442.4,57.5 444.4,57.6 446.4,57.7 448.3,57.8 450.2,57.9 452.2,58 454.1,58.1 456.1,58.1 458.1,58.2 460,58.3" fill="none" class="stroke-a" stroke-width="2.5"/>
<polyline points="70,40 72,40 73.9,40.1 75.8,40.1 77.8,40.2 79.8,40.2 81.7,40.3 83.7,40.3 85.6,40.4 87.5,40.4 89.5,40.5 91.5,40.5 93.4,40.5 95.3,40.6 97.3,40.6 99.2,40.7 101.2,40.7 103.2,40.8 105.1,40.8 107,40.9 109,40.9 111,41 112.9,41 114.8,41.1 116.8,41.1 118.8,41.1 120.7,41.2 122.7,41.2 124.6,41.3 126.5,41.3 128.5,41.4 130.4,41.4 132.4,41.5 134.3,41.5 136.3,41.6 138.2,41.6 140.2,41.6 142.2,41.7 144.1,41.7 146.1,41.8 148,41.8 149.9,41.9 151.9,41.9 153.8,42 155.8,42 157.8,42.1 159.7,42.1 161.7,42.1 163.6,42.2 165.6,42.2 167.5,42.3 169.4,42.3 171.4,42.4 173.3,42.4 175.3,42.5 177.2,42.5 179.2,42.6 181.2,42.6 183.1,42.7 185.1,42.7 187,42.7 188.9,42.8 190.9,42.8 192.8,42.9 194.8,42.9 196.8,43 198.7,43 200.7,43.1 202.6,43.1 204.6,43.2 206.5,43.2 208.4,43.2 210.4,43.3 212.3,43.3 214.3,43.4 216.2,43.4 218.2,43.5 220.2,43.5 222.1,43.6 224.1,43.6 226,43.7 227.9,43.7 229.9,43.7 231.8,43.8 233.8,43.8 235.8,43.9 237.7,43.9 239.7,44 241.6,44 243.6,44.1 245.5,44.1 247.4,44.2 249.4,44.2 251.3,44.3 253.3,44.3 255.2,44.3 257.2,44.4 259.1,44.4 261.1,44.5 263.1,44.5 265,44.6 266.9,44.6 268.9,44.7 270.9,44.7 272.8,44.8 274.8,44.8 276.7,44.8 278.6,44.9 280.6,44.9 282.6,45 284.5,45 286.4,45.1 288.4,45.1 290.4,45.2 292.3,45.2 294.2,45.3 296.2,45.3 298.1,45.3 300.1,45.4 302.1,45.4 304,45.5 305.9,45.5 307.9,45.6 309.9,45.6 311.8,45.7 313.8,45.7 315.7,45.8 317.6,45.8 319.6,45.9 321.6,45.9 323.5,45.9 325.4,46 327.4,46 329.4,46.1 331.3,46.1 333.2,46.2 335.2,46.2 337.1,46.3 339.1,46.3 341.1,46.4 343,46.4 344.9,46.4 346.9,46.5 348.9,46.5 350.8,46.6 352.8,46.6 354.7,46.7 356.6,46.7 358.6,46.8 360.6,46.8 362.5,46.9 364.4,46.9 366.4,46.9 368.4,47 370.3,47 372.2,47.1 374.2,47.1 376.1,47.2 378.1,47.2 380.1,47.3 382,47.3 383.9,47.4 385.9,47.4 387.9,47.5 389.8,47.5 391.8,47.5 393.7,47.6 395.6,47.6 397.6,47.7 399.6,47.7 401.5,47.8 403.4,47.8 405.4,47.9 407.4,47.9 409.3,48 411.2,48 413.2,48 415.1,48.1 417.1,48.1 419.1,48.2 421,48.2 422.9,48.3 424.9,48.3 426.9,48.4 428.8,48.4 430.8,48.5 432.7,48.5 434.6,48.5 436.6,48.6 438.6,48.6 440.5,48.7 442.4,48.7 444.4,48.8 446.4,48.8 448.3,48.9 450.2,48.9 452.2,49 454.1,49 456.1,49.1 458.1,49.1 460,49.1" fill="none" class="stroke-c" stroke-width="2.5"/>
<circle cx="126.5" cy="250" r="4" class="fill-bad"/>
<text x="136.6" y="242" class="fig-small">plain 0.45 leaves the chart at step 29</text>
<text x="468" y="43.1" class="fig-small">gate 0.995: 0.37</text>
<text x="468" y="70.3" class="fig-small">gate 0.99: 0.13</text>
<text x="468" y="236.2" class="fig-small">gate 0.9: 7e-10</text>
</svg>
Figure 3. Share of the gradient left against steps back, for a plain net at 0.45 and gates of 0.9, 0.99, and 0.995. The plain net leaves the chart after 29 steps, while a gate of 0.995 still keeps 0.37 at step 200. Notice how much the exact gate value matters: 0.9 keeps almost nothing at 200, and 0.995 keeps over a third.
```

To say it as a number, pick a threshold, such as "less than 1 percent is left", and count the steps it takes to get there. You want the smallest $t$ with $\text{factor}^t < \text{threshold}$. Take logs of both sides. The log of a number below one is negative, so dividing by it flips the inequality:

$$
t > \frac{\ln(\text{threshold})}{\ln(\text{factor})}
$$

Read it aloud: the number of steps is the log of the threshold divided by the log of the factor, rounded up to the next whole number. With a factor of 0.45 and a threshold of 0.01, the ratio is 5.77, so the answer is 6 steps. A gate of 0.9 takes 44 steps, a gate of 0.99 takes 459, and a gate of 0.995 takes 919. The plain net hears from 6 steps back, and the belt can hear from hundreds.

> [!KEY]
> The gradient through many steps is a product with one factor per step, so factors below one make it vanish. A belt with a gate has only the gate as its factor, so a gate near one lets the gradient survive, which is the answer to the opening question: change the path so the memory is multiplied by a learned number near one, not by a weight matrix and a tanh slope.

> [!RECAP]
> - A gradient across $k$ links is the product of one factor per link, so a plain net with factor 0.45 keeps $0.45^k$ of it.
> - Bigger weights trade fading for exploding, and adding inputs to the memory never lets it forget.
> - A belt multiplies the memory by a gate between 0 and 1, and the slope across one belt link is that gate.
> - Across $k$ links the share left is the product of the gates, which `np.cumprod` computes along time.
> - The steps a gradient survives are $\ln(\text{threshold}) / \ln(\text{factor})$, rounded up.

## Build it step by step

The checks test three functions in this order.

1. **`gradient_after_steps(factor, steps)`.** Takes a factor and a number of steps, and returns an array of length `steps` whose entry $t - 1$ is `factor ** t`. Build the exponents with `np.arange(1, steps + 1)`. The most common mistake is starting the exponents at 0, which makes the array begin with 1 and end one step short.
2. **`steps_until_below(factor, threshold)`.** Takes a factor and a threshold, both strictly between 0 and 1, and returns the smallest whole $t$ with `factor ** t < threshold` as an `int`. Use `np.log` on both numbers, divide, and take the floor plus one. The most common mistake is `np.ceil`, which returns `t` instead of `t + 1` when the ratio is a whole number.
3. **`forget_gate_gradient_product(forget_gates)`.** Takes gates of shape `(T, hidden_size)` and returns the same shape, where row $t$ holds the product of rows 0 through $t$. Use `np.cumprod` along axis 0. The most common mistake is `np.prod`, which returns one row.

## Where this breaks

A belt with gates near one slows the fade without removing it. A gate of 0.99 keeps $0.99^{1000} = 4.3 \times 10^{-5}$ of the gradient over 1,000 steps, so very long memories are still hard. The gates must also stay near one while the net learns, and nothing forces them to. The next chapters build the cell that learns those gates, and a later chapter, "Why one vector is not enough: attention as a soft lookup", gives the net a direct path to every earlier step so that distance stops mattering.

## The exercise

You build `gradient_after_steps`, `steps_until_below`, and `forget_gate_gradient_product`. The checks compare the plain-net curve with repeated multiplication, confirm that `steps_until_below` returns the first step that crosses the threshold, match the running product to a loop, and require that a belt of gates at 0.995 keeps more than 0.3 of the gradient at step 200 while a plain net at 0.45 keeps under $10^{-60}$.
