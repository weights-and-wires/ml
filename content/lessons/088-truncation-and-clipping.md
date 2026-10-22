---
title: "Taming long sequences: truncation and clipping"
slug: 088-truncation-and-clipping
part: "Part 5 — Vision and sequences"
summary: Cut the backward pass into windows to save memory, and cap a gradient's length by its global norm without turning it.
minutes: 40
gpu: false
objectives:
  - Explain why training on a very long sequence needs truncation, and what truncation costs.
  - Say why a state is carried across a chunk boundary even though the gradient is not.
  - Explain why clipping by one shared factor keeps a gradient's direction, and why per-array clipping does not.
  - Write the chunk boundaries, a clip by global norm, and a truncated backward pass.
prereqs:
  - 086-backprop-through-time
  - 087-vanishing-and-exploding-gradients
exercise: 088-truncation-and-clipping
---

# Taming long sequences: truncation and clipping

Your sequence has 10,000 steps, and one huge gradient has sent your weights to `nan`. You face two problems at once. The backward pass needs every state the forward pass produced, so memory grows with the length. And a single enormous gradient can undo thousands of good updates in one step.

You could chop the sequence into short pieces and train on each one separately, but that throws away the memory the model is meant to build, because every piece would start from a blank state. You could lower the learning rate until nothing explodes, but then everything trains slowly, including the steps that were fine. This chapter gives each problem a smaller, sharper fix: cut the gradient but keep the state, and cap the length of a gradient but keep its direction.

## What you already know

You already use these ideas:

- The backward loop adds each step's own loss to a carry from later steps, and passes the carry back through the cell.
- The backward pass needs every state from the forward pass, so training memory grows with $n T d_h$.
- A gradient over many weights can be treated as one long vector, whose norm is its length: the square root of the sum of the squares of its entries.
- Over many steps the carry is a product of factors, so it can shrink to nothing or grow without limit.
- A step along the negative gradient goes downhill, and the learning rate scales how far.

## Cut the gradient, keep the state

Truncated backpropagation through time keeps the forward pass whole and cuts only the backward pass. Pick a window size $k$. Split the $T$ steps into chunks of $k$. Run the backward pass on each chunk alone, treating the state that entered the chunk as a plain constant. Figure 1 shows six steps with $k = 2$.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Two rows of six cells. In full backpropagation the gradient arrows cross every boundary. In truncated backpropagation with chunks of two, they stop at the chunk edges while forward arrows continue">
<defs><marker id="c088-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c088-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="20" y="24" class="fig-bold">Full backpropagation: the gradient crosses every boundary</text>
<rect x="40" y="40" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="72" y="65" text-anchor="middle" class="fig-small">step 1</text>
<line x1="104" y1="60" x2="132" y2="60" class="stroke-a" marker-end="url(#c088-arrow-a)"/>
<rect x="132" y="40" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="164" y="65" text-anchor="middle" class="fig-small">step 2</text>
<line x1="196" y1="60" x2="224" y2="60" class="stroke-a" marker-end="url(#c088-arrow-a)"/>
<rect x="224" y="40" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="256" y="65" text-anchor="middle" class="fig-small">step 3</text>
<line x1="288" y1="60" x2="316" y2="60" class="stroke-a" marker-end="url(#c088-arrow-a)"/>
<rect x="316" y="40" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="348" y="65" text-anchor="middle" class="fig-small">step 4</text>
<line x1="380" y1="60" x2="408" y2="60" class="stroke-a" marker-end="url(#c088-arrow-a)"/>
<rect x="408" y="40" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="440" y="65" text-anchor="middle" class="fig-small">step 5</text>
<line x1="472" y1="60" x2="500" y2="60" class="stroke-a" marker-end="url(#c088-arrow-a)"/>
<rect x="500" y="40" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="532" y="65" text-anchor="middle" class="fig-small">step 6</text>
<line x1="516" y1="112" x2="456" y2="112" class="stroke-b" marker-end="url(#c088-arrow-b)"/>
<line x1="424" y1="112" x2="364" y2="112" class="stroke-b" marker-end="url(#c088-arrow-b)"/>
<line x1="332" y1="112" x2="272" y2="112" class="stroke-b" marker-end="url(#c088-arrow-b)"/>
<line x1="240" y1="112" x2="180" y2="112" class="stroke-b" marker-end="url(#c088-arrow-b)"/>
<line x1="148" y1="112" x2="88" y2="112" class="stroke-b" marker-end="url(#c088-arrow-b)"/>
<text x="20" y="170" class="fig-bold">Truncated, k = 2: the gradient stops at each chunk edge</text>
<rect x="32" y="182" width="172" height="112" class="fill-soft-a"/>
<rect x="216" y="182" width="172" height="112" class="fill-soft-c"/>
<rect x="400" y="182" width="172" height="112" class="fill-soft-d"/>
<rect x="40" y="192" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="72" y="217" text-anchor="middle" class="fig-small">step 1</text>
<line x1="104" y1="212" x2="132" y2="212" class="stroke-a" marker-end="url(#c088-arrow-a)"/>
<rect x="132" y="192" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="164" y="217" text-anchor="middle" class="fig-small">step 2</text>
<line x1="196" y1="212" x2="224" y2="212" class="stroke-a" marker-end="url(#c088-arrow-a)"/>
<rect x="224" y="192" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="256" y="217" text-anchor="middle" class="fig-small">step 3</text>
<line x1="288" y1="212" x2="316" y2="212" class="stroke-a" marker-end="url(#c088-arrow-a)"/>
<rect x="316" y="192" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="348" y="217" text-anchor="middle" class="fig-small">step 4</text>
<line x1="380" y1="212" x2="408" y2="212" class="stroke-a" marker-end="url(#c088-arrow-a)"/>
<rect x="408" y="192" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="440" y="217" text-anchor="middle" class="fig-small">step 5</text>
<line x1="472" y1="212" x2="500" y2="212" class="stroke-a" marker-end="url(#c088-arrow-a)"/>
<rect x="500" y="192" width="64" height="40" rx="5" class="fill-paper stroke-ink"/>
<text x="532" y="217" text-anchor="middle" class="fig-small">step 6</text>
<line x1="148" y1="264" x2="88" y2="264" class="stroke-b" marker-end="url(#c088-arrow-b)"/>
<line x1="332" y1="264" x2="272" y2="264" class="stroke-b" marker-end="url(#c088-arrow-b)"/>
<line x1="516" y1="264" x2="456" y2="264" class="stroke-b" marker-end="url(#c088-arrow-b)"/>
<line x1="204" y1="258" x2="216" y2="270" class="stroke-bad"/>
<line x1="204" y1="270" x2="216" y2="258" class="stroke-bad"/>
<text x="210" y="288" text-anchor="middle" class="fig-small fill-bad">cut</text>
<line x1="388" y1="258" x2="400" y2="270" class="stroke-bad"/>
<line x1="388" y1="270" x2="400" y2="258" class="stroke-bad"/>
<text x="394" y="288" text-anchor="middle" class="fig-small fill-bad">cut</text>
<text x="114" y="175" text-anchor="middle" class="fig-small"></text>
<text x="298" y="175" text-anchor="middle" class="fig-small"></text>
<text x="482" y="175" text-anchor="middle" class="fig-small"></text>
</svg>
Figure 1. Blue arrows are the forward pass, orange arrows are gradients, and the three tinted blocks are chunks. Forward values still run through all six steps, so the state keeps its memory, but a gradient starting at step 6 stops at the edge between steps 4 and 5, and never reaches steps 1 to 4.
```

The forward arrows in blue still run through all six steps. The orange gradient arrows stop at each chunk edge. Two things follow. The model still carries a memory forward through all six steps, because the state entering each chunk is the true state from the step before it, not a fresh $h_0$. But no gradient crosses a boundary, so the model gets no signal about how an earlier chunk shaped a later one.

Use the running example once more, with $h_0 = 0$, readings $1, 0.5, 0, 0, 0, 0$, and a loss that pulls on the last state with strength 1. Full backpropagation gives a gradient on $h_1$ of $0.1529$ and on the readings $dx_1 = 0.0601$, and the weights get $dW_{xh} = 0.2158$, $dW_{hh} = 0.9626$, and $db = 2.8532$. With $k = 2$, only the last chunk, steps 5 and 6, sees the loss. Its gradient reaches back two steps and stops, so $dx_1$ through $dx_4$ are exactly 0, $dW_{hh} = 0.4710$, $db = 1.6815$, and $dW_{xh} = 0$, because the last two readings are zero and the only readings that mattered are out of reach.

That last number is the price. A dependency longer than $k$ steps cannot be learned, because no gradient ever connects its two ends. The spike at step 1 in the opening of the previous chapter would be invisible with $k = 2$. Pick $k$ longer than the dependencies you care about, and shorter than your memory budget allows.

The saving is in memory. If you run the forward pass chunk by chunk and drop each chunk's activations after its backward pass, you keep only $n k d_h$ numbers instead of $n T d_h$. With $n = 32$, $d_h = 256$, and 8 bytes per number, 10,000 steps needs about 655 MB for the states alone, and a window of $k = 100$ needs about 6.6 MB. Time does not grow, since each step is still visited once forward and once backward. The exercise runs the forward pass once, over the whole sequence, to keep the code short. A memory-saving loop interleaves the two passes, and the numbers it produces are the same.

```python
h_in = h_seq[start - 1] if start > 0 else h0   # true state entering the chunk
dx_c, dW_xh_c, dW_hh_c, db_c, _ = bptt(dH[start:end], x[start:end], h_in, W_xh, W_hh, b)
```

The usual mistake is to restart every chunk from `h0`, which changes the forward values as well as the gradient. The last output of `bptt`, the gradient on the chunk's starting state, is the one that would cross the boundary, so it is dropped.

## A step that is too long

The second problem is different. Sometimes the gradient is correct and still enormous. Figure 2 shows a loss with a bowl in the middle, $0.05\,w^2$, and steep walls beyond $|w| = 1$, where the loss climbs as $5(|w|-1)^2$ on top. Recurrent networks can have loss surfaces with walls like this, because the same weight is multiplied into the output many times.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="A loss curve with a bowl in the middle and steep walls on both sides. From w = 1.5 an unclipped step lands at w = -3.65 on the far wall, and a clipped step lands at w = 0.5 inside the bowl">
<defs><marker id="c088-arrow-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-bad"/></marker><marker id="c088-arrow-c" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-c"/></marker></defs>
<polyline points="60,41 61.7,43.7 63.4,46.3 65.1,49 66.8,51.6 68.5,54.2 70.2,56.8 71.9,59.4 73.6,61.9 75.4,64.4 77.1,67 78.8,69.4 80.5,71.9 82.2,74.3 83.9,76.8 85.6,79.2 87.3,81.5 89,83.9 90.7,86.2 92.4,88.5 94.1,90.8 95.8,93.1 97.5,95.4 99.2,97.6 100.9,99.8 102.6,102 104.3,104.2 106.1,106.3 107.8,108.4 109.5,110.5 111.2,112.6 112.9,114.7 114.6,116.7 116.3,118.8 118,120.8 119.7,122.7 121.4,124.7 123.1,126.6 124.8,128.5 126.5,130.4 128.2,132.3 129.9,134.2 131.6,136 133.3,137.8 135.1,139.6 136.8,141.4 138.5,143.1 140.2,144.8 141.9,146.5 143.6,148.2 145.3,149.9 147,151.5 148.7,153.2 150.4,154.8 152.1,156.3 153.8,157.9 155.5,159.4 157.2,160.9 158.9,162.4 160.6,163.9 162.3,165.4 164,166.8 165.8,168.2 167.5,169.6 169.2,171 170.9,172.3 172.6,173.6 174.3,174.9 176,176.2 177.7,177.5 179.4,178.7 181.1,180 182.8,181.2 184.5,182.3 186.2,183.5 187.9,184.6 189.6,185.7 191.3,186.8 193,187.9 194.7,189 196.5,190 198.2,191 199.9,192 201.6,193 203.3,193.9 205,194.8 206.7,195.8 208.4,196.6 210.1,197.5 211.8,198.3 213.5,199.2 215.2,200 216.9,200.7 218.6,201.5 220.3,202.2 222,203 223.7,203.7 225.5,204.3 227.2,205 228.9,205.6 230.6,206.2 232.3,206.8 234,207.4 235.7,207.9 237.4,208.5 239.1,209 240.8,209.5 242.5,209.9 244.2,210.4 245.9,210.8 247.6,211.2 249.3,211.6 251,211.9 252.7,212.3 254.4,212.6 256.2,212.9 257.9,213.2 259.6,213.4 261.3,213.6 263,213.8 264.7,214 266.4,214.2 268.1,214.4 269.8,214.5 271.5,214.6 273.2,214.7 274.9,214.7 276.6,214.8 278.3,214.8 280,214.8 281.7,214.8 283.4,214.8 285.2,214.8 286.9,214.9 288.6,214.9 290.3,214.9 292,214.9 293.7,214.9 295.4,214.9 297.1,214.9 298.8,214.9 300.5,214.9 302.2,214.9 303.9,214.9 305.6,214.9 307.3,214.9 309,214.9 310.7,214.9 312.4,214.9 314.1,215 315.9,215 317.6,215 319.3,215 321,215 322.7,215 324.4,215 326.1,215 327.8,215 329.5,215 331.2,215 332.9,215 334.6,215 336.3,215 338,215 339.7,215 341.4,215 343.1,215 344.8,215 346.6,215 348.3,215 350,215 351.7,215 353.4,215 355.1,215 356.8,215 358.5,215 360.2,215 361.9,215 363.6,215 365.3,215 367,215 368.7,215 370.4,215 372.1,215 373.8,215 375.6,215 377.3,215 379,215 380.7,215 382.4,215 384.1,215 385.8,215 387.5,215 389.2,214.9 390.9,214.9 392.6,214.9 394.3,214.9 396,214.9 397.7,214.9 399.4,214.9 401.1,214.9 402.8,214.9 404.5,214.9 406.3,214.9 408,214.9 409.7,214.9 411.4,214.9 413.1,214.9 414.8,214.9 416.5,214.8 418.2,214.8 419.9,214.8 421.6,214.8 423.3,214.8 425,214.8 426.7,214.8 428.4,214.7 430.1,214.7 431.8,214.6 433.5,214.5 435.3,214.3 437,214.2 438.7,214 440.4,213.8 442.1,213.6 443.8,213.3 445.5,213.1 447.2,212.8 448.9,212.5 450.6,212.2 452.3,211.8 454,211.5 455.7,211.1 457.4,210.7 459.1,210.2 460.8,209.8 462.5,209.3 464.2,208.8 466,208.3 467.7,207.8 469.4,207.2 471.1,206.7 472.8,206.1 474.5,205.4 476.2,204.8 477.9,204.1 479.6,203.5 481.3,202.8 483,202 484.7,201.3 486.4,200.5 488.1,199.7 489.8,198.9 491.5,198.1 493.2,197.3 494.9,196.4 496.7,195.5 498.4,194.6 500.1,193.7 501.8,192.7 503.5,191.7 505.2,190.7 506.9,189.7 508.6,188.7 510.3,187.6 512,186.5 513.7,185.4 515.4,184.3 517.1,183.2 518.8,182 520.5,180.8 522.2,179.6 523.9,178.4 525.7,177.1 527.4,175.9 529.1,174.6 530.8,173.3 532.5,171.9 534.2,170.6 535.9,169.2 537.6,167.8 539.3,166.4 541,165 542.7,163.5 544.4,162 546.1,160.5 547.8,159 549.5,157.4 551.2,155.9 552.9,154.3 554.6,152.7 556.4,151.1 558.1,149.4 559.8,147.7 561.5,146.1 563.2,144.3 564.9,142.6 566.6,140.9 568.3,139.1 570,137.3" class="stroke-ink"/>
<line x1="60" y1="222" x2="570" y2="222" class="stroke-muted"/>
<text x="315" y="254" text-anchor="middle">weight w</text>
<text x="20" y="24" class="fig-bold">loss</text>
<text x="60" y="240" text-anchor="middle" class="fig-small">-4</text>
<text x="205.7" y="240" text-anchor="middle" class="fig-small">-2</text>
<text x="351.4" y="240" text-anchor="middle" class="fig-small">0</text>
<text x="497.1" y="240" text-anchor="middle" class="fig-small">2</text>
<circle cx="460.7" cy="209.8" r="5" class="fill-ink"/>
<circle cx="85.5" cy="79" r="5" class="fill-bad"/>
<circle cx="387.9" cy="215" r="5" class="fill-c"/>
<text x="468.7" y="199.8" class="fig-small">start, loss 1.36</text>
<text x="93.5" y="75" class="fig-small fill-bad">loss 35.8</text>
<text x="381.9" y="181" text-anchor="end" class="fig-small fill-c">loss 0.0125</text>
<line x1="460.7" y1="205" x2="89.5" y2="205" class="stroke-bad" marker-end="url(#c088-arrow-bad)"/>
<text x="273.1" y="198" text-anchor="middle" class="fig-small fill-bad">unclipped step: 5.15</text>
<line x1="460.7" y1="190" x2="386.9" y2="190" class="stroke-c" marker-end="url(#c088-arrow-c)"/>
<text x="424.3" y="170" text-anchor="middle" class="fig-small fill-c">clipped: 1</text>
</svg>
Figure 2. The loss is 0.05w² inside the bowl and climbs steeply beyond |w| = 1. The gradient at w = 1.5 is 5.15, so a plain step with learning rate 1 flies across the bowl to the far wall, where the loss is 35.8. Capping the step at length 1 lands at w = 0.5, where the loss is 0.0125.
```

Start at $w = 1.5$, where the loss is 1.36 and the gradient is $0.1 \times 1.5 + 10 \times 0.5 = 5.15$. The gradient points the right way, toward the bowl, but with a learning rate of 1 the step has length 5.15. That jumps over the bowl to $w = -3.65$, high on the far wall, where the loss is 35.8. The gradient there is $-26.9$, so the next step lands near $w = 23$, where the loss is about 2,494. Each step is bigger than the last, which is how weights end up at `nan`.

You might say: lower the learning rate. That would tame this wall, but it would also shrink every ordinary step in the flat parts of training, and you cannot know in advance how steep the next wall is. A better rule is to leave small gradients alone and shorten only the ones that are too long. Cap the step at length 1, and the same start lands at $w = 0.5$, inside the bowl, with loss 0.0125. The direction was never the problem, only the length.

## Clip by the global norm

A model has many weight arrays, so the gradient is a list of arrays. Treat the whole list as one long vector and call its length $G$, the global norm. If $G$ is above a limit, multiply every array by the same factor so that the length lands exactly on the limit:

$$
g \leftarrow \min\left(1,\ \frac{\text{max\_norm}}{G}\right) g
$$

Read it aloud: if the gradient is shorter than the limit, the factor is 1 and nothing changes, and if it is longer, the factor is the limit divided by the length, which shrinks it to exactly the limit. Two gradient arrays $[3]$ and $[4]$ have $G = \sqrt{9 + 16} = 5$. With a limit of 1, the factor is $0.2$, giving $[0.6]$ and $[0.8]$. In numpy:

```python
G = np.sqrt(sum(np.sum(g ** 2) for g in grads))
scale = min(1.0, max_norm / G) if G > 0 else 1.0
```

The guard on `G > 0` keeps an all-zero gradient from dividing by zero. Figure 3 explains why the factor is shared.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="A gradient arrow of length 5 in the direction (3, 4), clipped on the left by one shared factor so its direction is kept, and clipped on the right array by array, which turns it">
<defs><marker id="c088-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c088-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c088-arrow-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-bad"/></marker></defs>
<text x="30" y="28" class="fig-bold">Global clip, limit 1</text>
<line x1="50" y1="265" x2="213.2" y2="265" class="stroke-muted"/>
<line x1="50" y1="265" x2="50" y2="53.8" class="stroke-muted"/>
<path d="M98,265 A48,48 0 0 0 50,217" class="stroke-muted dashed"/>
<line x1="50" y1="265" x2="194" y2="73" class="stroke-a" marker-end="url(#c088-arrow-a)"/>
<text x="202" y="87" class="fig-small">(3, 4)</text>
<text x="202" y="101" class="fig-small">length 5</text>
<line x1="50" y1="265" x2="78.8" y2="226.6" class="stroke-b" marker-end="url(#c088-arrow-b)"/>
<text x="116" y="243" class="fig-small fill-b">(0.6, 0.8)</text>
<text x="116" y="257" class="fig-small">length 1</text>
<text x="320" y="28" class="fig-bold">Clip each array alone, limit 1</text>
<line x1="340" y1="265" x2="503.2" y2="265" class="stroke-muted"/>
<line x1="340" y1="265" x2="340" y2="53.8" class="stroke-muted"/>
<path d="M388,265 A48,48 0 0 0 340,217" class="stroke-muted dashed"/>
<line x1="340" y1="265" x2="484" y2="73" class="stroke-a dashed" marker-end="url(#c088-arrow-a)"/>
<text x="492" y="87" class="fig-small">(3, 4)</text>
<line x1="340" y1="265" x2="388" y2="217" class="stroke-bad" marker-end="url(#c088-arrow-bad)"/>
<text x="410" y="243" class="fig-small fill-bad">(1, 1)</text>
<text x="410" y="257" class="fig-small">points at 45°, not 53°</text>
</svg>
Figure 3. The dashed arc is part of a circle of radius 1. On the left, one shared factor 1/5 shortens (3, 4) to (0.6, 0.8) along the same line. On the right, clipping [3] and [4] separately gives (1, 1), which has length 1.41, not 1, and points in a different direction.
```

With one shared factor, the clipped gradient is the original one made shorter, so it still points the same way and still goes downhill. Clip each array by its own norm, and the arrays are scaled by different amounts, so the combined direction turns. In the figure, $[3]$ and $[4]$ clipped separately both become 1, giving $(1, 1)$, which points at 45 degrees instead of 53 degrees, and which is not even the right length. That is a different update from the one the gradient asked for.

Clipping has a price too. When it triggers, the step is shorter than the true gradient step, so the update is biased toward caution. And it only ever shrinks: a gradient of $10^{-10}$ passes through with a factor of 1, which means clipping does nothing for the vanishing gradients of the last chapter. The loud failure has a cure here, and the silent one needs a different structure, which the next chapter builds.

> [!KEY]
> Truncation cuts the gradient at chunk edges but carries the state across them, so memory stays small and long-range memory still exists, at the cost of never learning links longer than $k$. Clipping caps a gradient's length with one shared factor, so it keeps the direction and stops the jump to `nan`. Together they answer the opening: train long sequences in windows, and cap the steps.

> [!RECAP]
> - Truncated backpropagation runs the backward pass on chunks of $k$ steps, each starting from the true state that entered it.
> - No gradient crosses a chunk boundary, so a dependency longer than $k$ cannot be learned.
> - Memory falls from $n T d_h$ to $n k d_h$ if the forward pass is also run chunk by chunk.
> - Clipping multiplies every array by $\min(1, \text{max\_norm}/G)$, with $G$ the global norm, so the direction is kept.
> - Clipping cures explosion but not vanishing, because it only ever shrinks a gradient.

## Build it step by step

The checks call the functions in this order. The forward pass and `bptt` from earlier chapters are written for you.

1. **`chunk_bounds(T, k)`.** Return a list of `(start, end)` pairs, with `end` exclusive, covering `0` to `T` in order. Use `range(0, T, k)` and `min(start + k, T)`. The usual mistake is a last chunk that runs past `T`.
2. **`clip_by_global_norm(grads, max_norm)`.** Return a separate list with every array times one shared scale, computed from the norm of all the arrays together. Guard a zero norm. The usual mistake is clipping each array by its own norm.
3. **`truncated_bptt(dH, x, h0, W_xh, W_hh, b, k)`.** Run `rnn_forward` once. For each pair from `chunk_bounds`, call `bptt` on that slice with the true state that entered it, copy `dx` into the slice, and add the weight gradients. Return `(dx, dW_xh, dW_hh, db)`. The usual mistake is passing `h0` to every chunk.

## Cost and scaling

| Quantity | Full BPTT | Truncated, window $k$ |
|---|---|---|
| Time for forward and backward | proportional to $T$ | proportional to $T$ |
| Memory for states | grows with $n T d_h$ | grows with $n k d_h$ |
| Longest dependency that can be learned | $T$ | $k$ |

Clipping costs one pass over the gradients to compute the norm, which is small beside the backward pass.

## Where this breaks

- A window shorter than the dependency you need hides it completely, and nothing in the loss tells you so. Test $k$ on a task where you know how far back the answer lies.
- A clip limit that is too low shortens every step, and training slows down. Watch how often clipping triggers. If it triggers on most steps, the limit is cutting the signal, not only the spikes.
- Clipping does not fix fading, and truncation makes fading worse, since the gradient has fewer steps in which to live.

## The exercise

You build `chunk_bounds`, `clip_by_global_norm`, and `truncated_bptt`. The checks run the chunk boundaries on 6 steps, 7 steps, and the case $k \ge T$. They clip $[3]$ and $[4]$ to $[0.6]$ and $[0.8]$, leave a short gradient alone, and confirm that a long one lands on the limit with the same direction. For `truncated_bptt`, they compare to full `bptt` when $k = T$, and to finite differences of a loss in which each chunk's starting state is held constant. Last, they confirm that a loss on the final step sends no gradient into earlier chunks.
