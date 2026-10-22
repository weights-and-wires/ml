---
title: "The feed-forward network and the gate: SwiGLU"
slug: 103-gated-feed-forward
part: "Part 6 — Transformers"
summary: Attention mixes tokens and a per-token network computes with the result; SwiGLU lets one learned branch gate another at the same weight budget.
minutes: 40
gpu: false
objectives:
  - Explain the division of labour in a transformer layer, where attention moves information between tokens and the feed-forward network computes on each token alone.
  - Compute SiLU by hand, and say how it differs from ReLU.
  - Compute a SwiGLU output by hand for a two-unit example, and say what the gate branch and the up branch each contribute.
  - Choose the hidden width of a gated network so that it has the same number of weights as a plain network, and count both.
prereqs:
  - 058-forward-pass-and-gelu
  - 102-rotary-embeddings
exercise: 103-gated-feed-forward
---

# The feed-forward network and the gate: SwiGLU

The last two chapters gave attention a sense of position, first with sinusoidal clocks and then with rotary embeddings. After attention, the word "bit" in "dog bit man" holds a blend of what "dog" and "man" had to say. What happens to that blend? The token still has to do something with it: decide that this is a biting event, that the subject is a dog, that the object is a man. Attention only gathers and averages. Its weights depend on the tokens, but the values it combines are plain vectors that get added up. A stack of nothing but attention layers would keep passing averaged values along and never compute a feature from a token's own vector.

So each token needs a small network of its own. What is the simplest per-token network, and how do you make it more expressive without making it bigger? This chapter answers in two steps. First, a plain network of two matrices with a bend between them. Second, a gated version called SwiGLU, which spends the same number of weights in a different way and tends to do better.

## What you already know

You already have these ideas:

- A layer multiplies a vector by a weight matrix, and a nonlinear function between layers is what lets a network compute more than one matrix could.
- ReLU sets negative numbers to zero and leaves positive numbers alone.
- The sigmoid, $\sigma(z) = 1/(1 + e^{-z})$, squashes any number into the range 0 to 1.
- A shape $(n, m)$ times a shape $(m, k)$ gives $(n, k)$, and numpy's `@` also works on a leading batch axis.
- Multiplying two vectors entry by entry, written $\odot$, scales each entry of one by the matching entry of the other.

## Attention talks, the feed-forward network thinks

A transformer layer splits the work in two. Attention is communication: each token reads from the other tokens. The feed-forward network is computation: each token takes what it collected and transforms it, alone. "Alone" matters. The same weights are applied to every token's row, and no row sees another, so this step never mixes tokens. The mixing stays in attention.

The plain version has two matrices. $W_1$ has shape $(d, d_{\text{ff}})$ and expands a token from $d$ numbers to $d_{\text{ff}}$ numbers. $W_2$ has shape $(d_{\text{ff}}, d)$ and compresses it back. In between sits ReLU:

$$
\operatorname{FFN}(x) = \operatorname{ReLU}(x W_1)\, W_2
$$

Read it aloud: expand the token, switch off the negative entries, and compress back. The width $d_{\text{ff}}$ is called the feed-forward width, and it is usually about $4d$. The expansion gives the network room to hold many separate features before the compression picks among them. Each entry of $xW_1$ is a hidden unit: one weighted sum of the token's numbers.

The weakness shows when you look at one hidden unit. It computes $\operatorname{ReLU}(w \cdot x)$, a single curve of a single projection. That one number has to do two jobs: say how strongly a feature is present, and say whether to pass anything on. A gate separates the two jobs. Before building it, swap the ReLU for a smoother bend.

## SiLU, a smooth ReLU

ReLU has a corner at 0, and it throws away every negative input. SiLU keeps a small negative response and has no corner. Its definition is

$$
\operatorname{silu}(z) = z \cdot \sigma(z)
$$

Read it aloud: multiply the number by the sigmoid of the number, so the sigmoid acts as a volume knob that runs from 0 to 1. For $z = 1$, the sigmoid is $1/(1 + e^{-1}) = 0.7311$, so $\operatorname{silu}(1) = 0.7311$. For $z = -1$, the sigmoid is $0.2689$, so $\operatorname{silu}(-1) = -0.2689$. For a large positive $z$ the knob is fully open and $\operatorname{silu}(z) \approx z$. For a large negative $z$ the knob is shut and $\operatorname{silu}(z) \approx 0$. Figure 1 draws the curve.

```figure
<svg viewBox="0 0 600 332" role="img" aria-label="ReLU, sigmoid, and SiLU curves from minus 4 to 4, on a wide view and zoomed near zero">
<text x="160.0" y="24" text-anchor="middle" class="fig-bold">Wide view</text>
<line x1="40" y1="230.0" x2="280" y2="230.0" class="stroke-grid"/>
<text x="34" y="234.0" text-anchor="end" class="fig-small">-1</text>
<line x1="40" y1="194.0" x2="280" y2="194.0" class="stroke-grid"/>
<text x="34" y="198.0" text-anchor="end" class="fig-small">0</text>
<line x1="40" y1="158.0" x2="280" y2="158.0" class="stroke-grid"/>
<text x="34" y="162.0" text-anchor="end" class="fig-small">1</text>
<line x1="40" y1="122.0" x2="280" y2="122.0" class="stroke-grid"/>
<text x="34" y="126.0" text-anchor="end" class="fig-small">2</text>
<line x1="40" y1="86.0" x2="280" y2="86.0" class="stroke-grid"/>
<text x="34" y="90.0" text-anchor="end" class="fig-small">3</text>
<line x1="40" y1="50.0" x2="280" y2="50.0" class="stroke-grid"/>
<text x="34" y="54.0" text-anchor="end" class="fig-small">4</text>
<line x1="160.0" y1="50" x2="160.0" y2="230" class="stroke-muted"/>
<text x="40.0" y="246" text-anchor="middle" class="fig-small">-4</text>
<text x="100.0" y="246" text-anchor="middle" class="fig-small">-2</text>
<text x="160.0" y="246" text-anchor="middle" class="fig-small">0</text>
<text x="220.0" y="246" text-anchor="middle" class="fig-small">2</text>
<text x="280.0" y="246" text-anchor="middle" class="fig-small">4</text>
<polyline points="40.0,194.0 41.5,194.0 43.0,194.0 44.5,194.0 46.0,194.0 47.5,194.0 49.0,194.0 50.5,194.0 52.0,194.0 53.5,194.0 55.0,194.0 56.5,194.0 58.0,194.0 59.5,194.0 61.0,194.0 62.5,194.0 64.0,194.0 65.5,194.0 67.0,194.0 68.5,194.0 70.0,194.0 71.5,194.0 73.0,194.0 74.5,194.0 76.0,194.0 77.5,194.0 79.0,194.0 80.5,194.0 82.0,194.0 83.5,194.0 85.0,194.0 86.5,194.0 88.0,194.0 89.5,194.0 91.0,194.0 92.5,194.0 94.0,194.0 95.5,194.0 97.0,194.0 98.5,194.0 100.0,194.0 101.5,194.0 103.0,194.0 104.5,194.0 106.0,194.0 107.5,194.0 109.0,194.0 110.5,194.0 112.0,194.0 113.5,194.0 115.0,194.0 116.5,194.0 118.0,194.0 119.5,194.0 121.0,194.0 122.5,194.0 124.0,194.0 125.5,194.0 127.0,194.0 128.5,194.0 130.0,194.0 131.5,194.0 133.0,194.0 134.5,194.0 136.0,194.0 137.5,194.0 139.0,194.0 140.5,194.0 142.0,194.0 143.5,194.0 145.0,194.0 146.5,194.0 148.0,194.0 149.5,194.0 151.0,194.0 152.5,194.0 154.0,194.0 155.5,194.0 157.0,194.0 158.5,194.0 160.0,194.0 161.5,192.2 163.0,190.4 164.5,188.6 166.0,186.8 167.5,185.0 169.0,183.2 170.5,181.4 172.0,179.6 173.5,177.8 175.0,176.0 176.5,174.2 178.0,172.4 179.5,170.6 181.0,168.8 182.5,167.0 184.0,165.2 185.5,163.4 187.0,161.6 188.5,159.8 190.0,158.0 191.5,156.2 193.0,154.4 194.5,152.6 196.0,150.8 197.5,149.0 199.0,147.2 200.5,145.4 202.0,143.6 203.5,141.8 205.0,140.0 206.5,138.2 208.0,136.4 209.5,134.6 211.0,132.8 212.5,131.0 214.0,129.2 215.5,127.4 217.0,125.6 218.5,123.8 220.0,122.0 221.5,120.2 223.0,118.4 224.5,116.6 226.0,114.8 227.5,113.0 229.0,111.2 230.5,109.4 232.0,107.6 233.5,105.8 235.0,104.0 236.5,102.2 238.0,100.4 239.5,98.6 241.0,96.8 242.5,95.0 244.0,93.2 245.5,91.4 247.0,89.6 248.5,87.8 250.0,86.0 251.5,84.2 253.0,82.4 254.5,80.6 256.0,78.8 257.5,77.0 259.0,75.2 260.5,73.4 262.0,71.6 263.5,69.8 265.0,68.0 266.5,66.2 268.0,64.4 269.5,62.6 271.0,60.8 272.5,59.0 274.0,57.2 275.5,55.4 277.0,53.6 278.5,51.8 280.0,50.0" class="stroke-muted dashed"/>
<polyline points="40.0,196.6 41.5,196.7 43.0,196.8 44.5,196.9 46.0,197.0 47.5,197.1 49.0,197.2 50.5,197.3 52.0,197.4 53.5,197.6 55.0,197.7 56.5,197.8 58.0,198.0 59.5,198.1 61.0,198.2 62.5,198.4 64.0,198.5 65.5,198.7 67.0,198.8 68.5,199.0 70.0,199.1 71.5,199.3 73.0,199.4 74.5,199.6 76.0,199.8 77.5,199.9 79.0,200.1 80.5,200.3 82.0,200.5 83.5,200.6 85.0,200.8 86.5,201.0 88.0,201.2 89.5,201.4 91.0,201.5 92.5,201.7 94.0,201.9 95.5,202.1 97.0,202.2 98.5,202.4 100.0,202.6 101.5,202.7 103.0,202.9 104.5,203.0 106.0,203.2 107.5,203.3 109.0,203.5 110.5,203.6 112.0,203.7 113.5,203.8 115.0,203.9 116.5,203.9 118.0,204.0 119.5,204.0 121.0,204.0 122.5,204.0 124.0,204.0 125.5,204.0 127.0,203.9 128.5,203.8 130.0,203.7 131.5,203.5 133.0,203.4 134.5,203.2 136.0,202.9 137.5,202.7 139.0,202.4 140.5,202.0 142.0,201.7 143.5,201.2 145.0,200.8 146.5,200.3 148.0,199.8 149.5,199.2 151.0,198.6 152.5,197.9 154.0,197.2 155.5,196.5 157.0,195.7 158.5,194.9 160.0,194.0 161.5,193.1 163.0,192.1 164.5,191.1 166.0,190.0 167.5,188.9 169.0,187.8 170.5,186.6 172.0,185.4 173.5,184.1 175.0,182.8 176.5,181.4 178.0,180.1 179.5,178.6 181.0,177.2 182.5,175.7 184.0,174.1 185.5,172.6 187.0,171.0 188.5,169.3 190.0,167.7 191.5,166.0 193.0,164.3 194.5,162.6 196.0,160.8 197.5,159.0 199.0,157.2 200.5,155.4 202.0,153.6 203.5,151.7 205.0,149.9 206.5,148.0 208.0,146.1 209.5,144.2 211.0,142.3 212.5,140.3 214.0,138.4 215.5,136.4 217.0,134.5 218.5,132.5 220.0,130.6 221.5,128.6 223.0,126.6 224.5,124.7 226.0,122.7 227.5,120.7 229.0,118.7 230.5,116.8 232.0,114.8 233.5,112.8 235.0,110.8 236.5,108.8 238.0,106.9 239.5,104.9 241.0,102.9 242.5,100.9 244.0,99.0 245.5,97.0 247.0,95.0 248.5,93.1 250.0,91.1 251.5,89.2 253.0,87.2 254.5,85.3 256.0,83.3 257.5,81.4 259.0,79.4 260.5,77.5 262.0,75.6 263.5,73.6 265.0,71.7 266.5,69.8 268.0,67.8 269.5,65.9 271.0,64.0 272.5,62.1 274.0,60.2 275.5,58.3 277.0,56.4 278.5,54.5 280.0,52.6" class="stroke-b"/>
<text x="160" y="258" text-anchor="middle" class="fig-small">z</text>
<text x="455.0" y="24" text-anchor="middle" class="fig-bold">Zoom near zero</text>
<line x1="340" y1="230.0" x2="570" y2="230.0" class="stroke-grid"/>
<text x="334" y="234.0" text-anchor="end" class="fig-small">-0.5</text>
<line x1="340" y1="177.1" x2="570" y2="177.1" class="stroke-grid"/>
<text x="334" y="181.1" text-anchor="end" class="fig-small">0</text>
<line x1="340" y1="124.1" x2="570" y2="124.1" class="stroke-grid"/>
<text x="334" y="128.1" text-anchor="end" class="fig-small">0.5</text>
<line x1="340" y1="71.2" x2="570" y2="71.2" class="stroke-grid"/>
<text x="334" y="75.2" text-anchor="end" class="fig-small">1</text>
<line x1="455.0" y1="50" x2="455.0" y2="230" class="stroke-muted"/>
<text x="340.0" y="246" text-anchor="middle" class="fig-small">-4</text>
<text x="397.5" y="246" text-anchor="middle" class="fig-small">-2</text>
<text x="455.0" y="246" text-anchor="middle" class="fig-small">0</text>
<text x="512.5" y="246" text-anchor="middle" class="fig-small">2</text>
<text x="570.0" y="246" text-anchor="middle" class="fig-small">4</text>
<polyline points="340.0,177.1 341.4,177.1 342.9,177.1 344.3,177.1 345.8,177.1 347.2,177.1 348.6,177.1 350.1,177.1 351.5,177.1 352.9,177.1 354.4,177.1 355.8,177.1 357.2,177.1 358.7,177.1 360.1,177.1 361.6,177.1 363.0,177.1 364.4,177.1 365.9,177.1 367.3,177.1 368.8,177.1 370.2,177.1 371.6,177.1 373.1,177.1 374.5,177.1 375.9,177.1 377.4,177.1 378.8,177.1 380.2,177.1 381.7,177.1 383.1,177.1 384.6,177.1 386.0,177.1 387.4,177.1 388.9,177.1 390.3,177.1 391.8,177.1 393.2,177.1 394.6,177.1 396.1,177.1 397.5,177.1 398.9,177.1 400.4,177.1 401.8,177.1 403.2,177.1 404.7,177.1 406.1,177.1 407.6,177.1 409.0,177.1 410.4,177.1 411.9,177.1 413.3,177.1 414.8,177.1 416.2,177.1 417.6,177.1 419.1,177.1 420.5,177.1 421.9,177.1 423.4,177.1 424.8,177.1 426.2,177.1 427.7,177.1 429.1,177.1 430.6,177.1 432.0,177.1 433.4,177.1 434.9,177.1 436.3,177.1 437.8,177.1 439.2,177.1 440.6,177.1 442.1,177.1 443.5,177.1 444.9,177.1 446.4,177.1 447.8,177.1 449.2,177.1 450.7,177.1 452.1,177.1 453.6,177.1 455.0,177.1 456.4,171.8 457.9,166.5 459.3,161.2 460.8,155.9 462.2,150.6 463.6,145.3 465.1,140.0 466.5,134.7 467.9,129.4 469.4,124.1 470.8,118.8 472.2,113.5 473.7,108.2 475.1,102.9 476.6,97.6 478.0,92.4 479.4,87.1 480.9,81.8 482.3,76.5 483.8,71.2 485.2,65.9 486.6,60.6 488.1,55.3" class="stroke-muted dashed"/>
<polyline points="340.0,175.2 341.4,175.1 342.9,175.0 344.3,174.9 345.8,174.7 347.2,174.6 348.6,174.5 350.1,174.4 351.5,174.2 352.9,174.1 354.4,174.0 355.8,173.8 357.2,173.6 358.7,173.5 360.1,173.3 361.6,173.1 363.0,172.9 364.4,172.7 365.9,172.5 367.3,172.3 368.8,172.0 370.2,171.8 371.6,171.5 373.1,171.3 374.5,171.0 375.9,170.7 377.4,170.4 378.8,170.1 380.2,169.7 381.7,169.4 383.1,169.0 384.6,168.6 386.0,168.3 387.4,167.8 388.9,167.4 390.3,167.0 391.8,166.5 393.2,166.0 394.6,165.5 396.1,165.0 397.5,164.4 398.9,163.9 400.4,163.3 401.8,162.7 403.2,162.0 404.7,161.4 406.1,160.7 407.6,160.0 409.0,159.3 410.4,158.5 411.9,157.7 413.3,156.9 414.8,156.1 416.2,155.3 417.6,154.4 419.1,153.5 420.5,152.5 421.9,151.6 423.4,150.6 424.8,149.6 426.2,148.6 427.7,147.5 429.1,146.5 430.6,145.4 432.0,144.2 433.4,143.1 434.9,141.9 436.3,140.7 437.8,139.5 439.2,138.3 440.6,137.1 442.1,135.8 443.5,134.6 444.9,133.3 446.4,132.0 447.8,130.7 449.2,129.4 450.7,128.1 452.1,126.8 453.6,125.4 455.0,124.1 456.4,122.8 457.9,121.5 459.3,120.2 460.8,118.8 462.2,117.5 463.6,116.2 465.1,114.9 466.5,113.7 467.9,112.4 469.4,111.2 470.8,109.9 472.2,108.7 473.7,107.5 475.1,106.3 476.6,105.1 478.0,104.0 479.4,102.9 480.9,101.8 482.3,100.7 483.8,99.7 485.2,98.6 486.6,97.6 488.1,96.6 489.5,95.7 490.9,94.8 492.4,93.9 493.8,93.0 495.2,92.1 496.7,91.3 498.1,90.5 499.6,89.7 501.0,89.0 502.4,88.2 503.9,87.5 505.3,86.9 506.8,86.2 508.2,85.6 509.6,85.0 511.1,84.4 512.5,83.8 513.9,83.3 515.4,82.7 516.8,82.2 518.2,81.7 519.7,81.3 521.1,80.8 522.6,80.4 524.0,80.0 525.4,79.6 526.9,79.2 528.3,78.8 529.8,78.5 531.2,78.2 532.6,77.8 534.1,77.5 535.5,77.2 536.9,77.0 538.4,76.7 539.8,76.4 541.2,76.2 542.7,76.0 544.1,75.7 545.6,75.5 547.0,75.3 548.4,75.1 549.9,74.9 551.3,74.8 552.8,74.6 554.2,74.4 555.6,74.3 557.1,74.1 558.5,74.0 559.9,73.9 561.4,73.7 562.8,73.6 564.2,73.5 565.7,73.4 567.1,73.3 568.6,73.2 570.0,73.1" class="stroke-d"/>
<polyline points="340.0,184.7 341.4,185.0 342.9,185.3 344.3,185.6 345.8,185.9 347.2,186.2 348.6,186.5 350.1,186.8 351.5,187.2 352.9,187.6 354.4,187.9 355.8,188.3 357.2,188.7 358.7,189.1 360.1,189.5 361.6,189.9 363.0,190.3 364.4,190.8 365.9,191.2 367.3,191.7 368.8,192.1 370.2,192.6 371.6,193.1 373.1,193.6 374.5,194.1 375.9,194.6 377.4,195.1 378.8,195.6 380.2,196.1 381.7,196.6 383.1,197.1 384.6,197.7 386.0,198.2 387.4,198.7 388.9,199.2 390.3,199.8 391.8,200.3 393.2,200.8 394.6,201.3 396.1,201.8 397.5,202.3 398.9,202.8 400.4,203.2 401.8,203.7 403.2,204.1 404.7,204.5 406.1,204.9 407.6,205.2 409.0,205.5 410.4,205.8 411.9,206.0 413.3,206.2 414.8,206.4 416.2,206.5 417.6,206.5 419.1,206.5 420.5,206.5 421.9,206.3 423.4,206.1 424.8,205.9 426.2,205.5 427.7,205.1 429.1,204.6 430.6,204.0 432.0,203.3 433.4,202.5 434.9,201.7 436.3,200.7 437.8,199.6 439.2,198.4 440.6,197.0 442.1,195.6 443.5,194.1 444.9,192.4 446.4,190.6 447.8,188.6 449.2,186.6 450.7,184.4 452.1,182.1 453.6,179.6 455.0,177.1 456.4,174.3 457.9,171.5 459.3,168.5 460.8,165.4 462.2,162.2 463.6,158.8 465.1,155.3 466.5,151.7 467.9,148.0 469.4,144.1 470.8,140.1 472.2,136.0 473.7,131.8 475.1,127.5 476.6,123.1 478.0,118.6 479.4,114.0 480.9,109.3 482.3,104.5 483.8,99.7 485.2,94.7 486.6,89.7 488.1,84.6 489.5,79.4 490.9,74.2 492.4,68.9 493.8,63.5 495.2,58.1 496.7,52.7" class="stroke-b"/>
<circle cx="483.8" cy="99.7" r="4" class="fill-b"/>
<text x="493.8" y="132.6" class="fig-small">SiLU(1) = 0.7311</text>
<circle cx="418.2" cy="206.5" r="4" class="fill-b"/>
<text x="426.2" y="222.5" class="fig-small">lowest: −0.278</text>
<text x="455" y="258" text-anchor="middle" class="fig-small">z</text>
<line x1="120" y1="288" x2="146" y2="288" class="stroke-muted dashed"/><text x="152" y="292" class="fig-small">ReLU</text>
<line x1="230" y1="288" x2="256" y2="288" class="stroke-b"/><text x="262" y="292" class="fig-small">SiLU</text>
<line x1="340" y1="288" x2="366" y2="288" class="stroke-d"/><text x="372" y="292" class="fig-small">sigmoid (right panel only)</text>
<text x="300" y="318" text-anchor="middle" class="fig-small">SiLU(z) = z × sigmoid(z). ReLU(z) = max(0, z). The right panel zooms in on the same curves.</text>
</svg>
Figure 1. ReLU (dashed) and SiLU, with the sigmoid in the zoomed panel. Notice that SiLU follows ReLU far from zero, but near zero it is smooth, and it dips slightly below zero (to about −0.278) before returning to 0.
```

## The gate

A gated network uses two projections of the same token instead of one. The gate branch computes $xW_{\text{gate}}$ and passes it through SiLU. The up branch computes $xW_{\text{up}}$ and applies no bend. The hidden vector is their entry-by-entry product, and a third matrix $W_{\text{down}}$ compresses it back:

$$
\operatorname{SwiGLU}(x) = \big(\operatorname{silu}(x W_{\text{gate}}) \odot (x W_{\text{up}})\big)\, W_{\text{down}}
$$

Read it aloud: one branch decides how much of each hidden unit passes, the other branch supplies what passes, and the product is what goes on to $W_{\text{down}}$. $W_{\text{gate}}$ and $W_{\text{up}}$ both have shape $(d, d_h)$, and $W_{\text{down}}$ has shape $(d_h, d)$. The letters $d_h$ name the hidden width of this gated network, and you meet the reason it differs from $d_{\text{ff}}$ below. SiLU goes on the gate branch only.

Work one token by hand, with $d = 2$ and a hidden width of 2. Take $x = [1, 2]$ and these matrices, written as rows:

$$
W_{\text{gate}} = \begin{bmatrix} -3 & 1 \\ 0 & 1 \end{bmatrix}, \quad
W_{\text{up}} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
W_{\text{down}} = \begin{bmatrix} 1 & 0 \\ 1 & -1 \end{bmatrix}
$$

The gate branch gives $xW_{\text{gate}} = [1 \times (-3) + 2 \times 0,\ 1 \times 1 + 2 \times 1] = [-3, 3]$. SiLU turns that into $[-0.142, 2.858]$, because $\operatorname{silu}(-3) = -3 \times 0.0474 = -0.142$ and $\operatorname{silu}(3) = 3 \times 0.9526 = 2.858$. The gate has nearly shut unit 1 and opened unit 2. The up branch gives $xW_{\text{up}} = [1, 2]$. Multiply entry by entry to get the hidden vector $[-0.142 \times 1,\ 2.858 \times 2] = [-0.142, 5.715]$. Finally $W_{\text{down}}$ gives $[-0.142 + 5.715,\ -5.715] = [5.573, -5.715]$. Figure 2 follows these numbers through the network.

```figure
<svg viewBox="0 0 600 372" role="img" aria-label="Data flow through a gated feed-forward network for the token [1, 2]: a gate branch through SiLU and an up branch multiply entry by entry, then a down projection gives the output">
<defs><marker id="c103-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="205.0" y="20.0" width="190" height="32" rx="6" class="stroke-ink fill-paper"/>
<text x="300" y="40" text-anchor="middle" class="fig-bold fig-small">token x = [1, 2]</text>
<rect x="55.0" y="92.0" width="230" height="40" rx="6" class="stroke-ink fill-paper"/>
<text x="170" y="108" text-anchor="middle" class="fig-bold fig-small">gate: x W_gate</text>
<text x="170" y="124" text-anchor="middle" class="fig-small">[−3, 3]</text>
<rect x="315.0" y="92.0" width="230" height="40" rx="6" class="stroke-ink fill-paper"/>
<text x="430" y="108" text-anchor="middle" class="fig-bold fig-small">up: x W_up</text>
<text x="430" y="124" text-anchor="middle" class="fig-small">[1, 2]</text>
<line x1="270" y1="52" x2="200" y2="90" class="stroke-ink" marker-end="url(#c103-arrow)"/>
<line x1="330" y1="52" x2="400" y2="90" class="stroke-ink" marker-end="url(#c103-arrow)"/>
<rect x="55.0" y="170.0" width="230" height="40" rx="6" class="stroke-ink fill-paper"/>
<text x="170" y="186" text-anchor="middle" class="fig-bold fig-small">SiLU of the gate</text>
<text x="170" y="202" text-anchor="middle" class="fig-small">[−0.142, 2.858]</text>
<line x1="170" y1="132" x2="170" y2="168" class="stroke-ink" marker-end="url(#c103-arrow)"/>
<text x="292" y="186" class="fig-small">unit 1: nearly shut</text><text x="292" y="202" class="fig-small">unit 2: open</text>
<circle cx="300" cy="252" r="15" class="stroke-ink fill-paper"/><text x="300" y="258" text-anchor="middle" class="fig-bold">×</text>
<polyline points="170,210 170,252 283,252" class="stroke-ink" marker-end="url(#c103-arrow)"/>
<polyline points="430,132 430,252 317,252" class="stroke-ink" marker-end="url(#c103-arrow)"/>
<text x="320" y="282" class="fig-small">hidden = [−0.142, 5.715]</text>
<line x1="300" y1="267" x2="300" y2="296" class="stroke-ink" marker-end="url(#c103-arrow)"/>
<rect x="215.0" y="299.0" width="170" height="30" rx="6" class="stroke-ink fill-paper"/>
<text x="300" y="318" text-anchor="middle" class="fig-bold fig-small">× W_down</text>
<text x="300" y="362" text-anchor="middle" class="fig-bold fig-small">output = [5.573, −5.715]</text>
<line x1="300" y1="329" x2="300" y2="346" class="stroke-ink" marker-end="url(#c103-arrow)"/>
</svg>
Figure 2. The token [1, 2] through a SwiGLU network with two hidden units. Notice that the gate branch nearly shuts unit 1 (−0.142) and opens unit 2 (2.858), so unit 2 passes the up value 2 magnified 2.858 times, while unit 1 passes its up value 1 scaled down to −0.142.
```

The gate depends on the token, not only on the weights. Feed the same network $x = [-1, 2]$. The gate branch gives $[3, 1]$, and SiLU turns that into $[2.858, 0.731]$. The up branch gives $[-1, 2]$. The hidden vector is $[-2.858, 1.462]$. Now unit 1 is wide open and unit 2 is partly open. Different tokens switch different hidden units on, with different strengths, and the up branch decides what flows through each unit that is on. The gate values are not capped at 1: a gate of 2.858 magnifies the up value, and a gate near 0 mutes it. A product of two projections can also express "feature A and feature B are both present" directly, which a single ReLU of one weighted sum has to approximate.

Does the gate really help? Experiments that compare gated and plain feed-forward networks of the same size find that the gated ones reach a lower loss. The "how much" and "what" picture is a way to think about that result. It is not a proof.

In numpy, with `z * sigmoid(z)` written as a stable expression:

```python
gate = x @ W_gate
silu = gate * 0.5 * (1.0 + np.tanh(gate / 2.0))   # gate * sigmoid(gate), safe for large negative gate
hidden = silu * (x @ W_up)
out = hidden @ W_down
```

The expression `0.5 * (1 + tanh(z / 2))` equals the sigmoid, and it never overflows. The more obvious `1 / (1 + np.exp(-z))` overflows for a gate value near $-800$, because $e^{800}$ is larger than a float can hold.

## Same budget, three matrices

A plain network holds two matrices of $d \times d_{\text{ff}}$ weights, so $2\,d\,d_{\text{ff}}$ in all. SwiGLU holds three of $d \times d_h$, so $3\,d\,d_h$. If you used the same width for both, the gated network would have 50 percent more weights, and a comparison would mix "gating helps" with "a bigger network helps". To compare fairly, set the two counts equal. That gives

$$
3\,d\,d_h = 2\,d\,d_{\text{ff}} \quad\Longrightarrow\quad d_h = \tfrac{2}{3}\, d_{\text{ff}}
$$

Read it aloud: with three matrices instead of two, each matrix gets two-thirds of the width. A check with $d = 2$ and $d_{\text{ff}} = 6$: the plain network holds $2 \times 2 \times 6 = 24$ weights and the gated one holds $3 \times 2 \times 4 = 24$. When $d_{\text{ff}}$ is not a multiple of 3, you round. At $d = 64$ and $d_{\text{ff}} = 256$, two-thirds is 170.67, so you take $d_h = 171$. Then the plain network holds $32768$ weights and the gated one holds $3 \times 64 \times 171 = 32832$, a little over 0.2 percent more. Figure 3 compares the counts, including the 49152 weights you would get by keeping the width at 256. This chapter leaves out bias vectors, which most gated networks do not use.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Bars for the weights in a feed-forward network at width 64: 32768 for the plain network, 32832 for SwiGLU with hidden width 171, and 49152 for SwiGLU with hidden width 256">
<line x1="60" y1="250" x2="560" y2="250" class="stroke-muted"/>
<rect x="105" y="140.0" width="90" height="110.0" class="fill-a"/>
<text x="150" y="132.0" text-anchor="middle" class="fig-bold">32,768</text>
<text x="150" y="270" text-anchor="middle" class="fig-small fig-bold">Plain network</text>
<text x="150" y="286" text-anchor="middle" class="fig-small">2 matrices, d_ff = 256</text>
<rect x="255" y="139.8" width="90" height="110.2" class="fill-c"/>
<text x="300" y="131.8" text-anchor="middle" class="fig-bold">32,832</text>
<text x="300" y="270" text-anchor="middle" class="fig-small fig-bold">SwiGLU, d_h = 171</text>
<text x="300" y="286" text-anchor="middle" class="fig-small">3 matrices, two-thirds width</text>
<rect x="405" y="85.0" width="90" height="165.0" class="fill-bad"/>
<text x="450" y="77.0" text-anchor="middle" class="fig-bold">49,152</text>
<text x="450" y="270" text-anchor="middle" class="fig-small fig-bold">SwiGLU, d_h = 256</text>
<text x="450" y="286" text-anchor="middle" class="fig-small">3 matrices, same width</text>
<line x1="60" y1="140.0" x2="560" y2="140.0" class="stroke-ink dashed"/>
<text x="598" y="134.0" text-anchor="end" class="fig-small">plain budget</text>
<text x="450" y="59.0" text-anchor="middle" class="fig-small">+50 percent</text>
<text x="300" y="22" text-anchor="middle" class="fig-bold">Weights in one feed-forward network, d = 64</text>
</svg>
Figure 3. Weights in the feed-forward part at d = 64. Notice that two-thirds width puts SwiGLU within 0.2 percent of the plain network, while keeping the width at 256 adds 50 percent more weights.
```

> [!KEY]
> Attention moves information between tokens, and the feed-forward network computes on each token alone. SwiGLU computes with a gate: one branch decides how much of each hidden unit passes and the other supplies what passes, and a hidden width of two-thirds of $d_{\text{ff}}$ keeps the weight count equal to a plain network. The expressiveness comes from how the weights are used, not from having more of them.

## Where this breaks

**SiLU belongs on the gate only.** Putting it on the up branch as well changes the function. The output should double when the up branch doubles, and with SiLU on both branches it does not. A network trained with one definition gives wrong answers with the other.

**The two widths are easy to confuse.** The hidden width $d_h$ of the feed-forward network is not the width $d_{\text{head}}$ of one attention head. They are separate numbers that happen to look alike.

**The two-thirds rule is rounded in practice.** Many models round $d_h$ up to a multiple of 64 or 256, so that matrix multiplies run efficiently. The count then drifts a little from the exact budget, which is harmless once you know it is there.

> [!RECAP]
> Carry these facts into the exercise:
> - The feed-forward network works on each token's row alone. Attention does the mixing.
> - `silu(z) = z * sigmoid(z)`, and a stable sigmoid avoids `np.exp` of a large positive number.
> - `swiglu(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down`, with SiLU on the gate only.
> - A plain network holds `2 * d * d_ff` weights. A gated one holds `3 * d * d_h` with `d_h = round(2 * d_ff / 3)`.

## Build it step by step

You write three functions, in the order the checks test them.

1. **`silu(z)`** takes an array of any shape and returns `z * sigmoid(z)` with the same shape. Write the sigmoid as `0.5 * (1 + np.tanh(z / 2))`. The common mistake is `1 / (1 + np.exp(-z))`, which overflows for large negative `z`, and the check turns that overflow into a failure.
2. **`swiglu(x, W_gate, W_up, W_down)`** maps `(..., d)` to `(..., d)`. Compute the two products, apply your `silu` to the gate product, multiply entry by entry, and finish with `@ W_down`. The common mistake is SiLU on the up branch too.
3. **`ffn_parameter_count(d_model, d_ff, gated)`** returns an `int`. For a plain network it is `2 * d_model * d_ff`. For a gated one, compute `d_h = round(2 * d_ff / 3)` and return `3 * d_model * d_h`. The common mistake is to use `d_ff` as the gated width, which gives 50 percent too many.

## The exercise

You write `silu`, `swiglu`, and `ffn_parameter_count`. The checks confirm `silu(0) = 0` and `silu(1) = 0.7311`, that large positive inputs return nearly `z`, and that large negative inputs return a finite number near 0 without overflow. They also check the identity $\operatorname{silu}(z) - \operatorname{silu}(-z) = z$. The `swiglu` checks compare it with the two-unit hand example and with a direct computation, confirm that a zero gate gives zero output, that doubling `W_up` doubles the output, and that a leading batch axis works. The last checks confirm the counts 24 and 24 at $d = 2$ and $d_{\text{ff}} = 6$, and that the gated count at $d = 64$ and $d_{\text{ff}} = 256$ is within 1 percent of the plain count.
