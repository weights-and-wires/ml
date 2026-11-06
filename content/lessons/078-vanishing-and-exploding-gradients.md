---
title: Why gradients fade or blow up
slug: 078-vanishing-and-exploding-gradients
part: "Part 5 — Vision and sequences"
summary: The gradient to a state many steps back is a product of one factor per step, so it shrinks or grows like a power, set by the largest eigenvalue.
minutes: 40
gpu: false
objectives:
  - Explain why a gradient that crosses many steps behaves like compound interest.
  - Say why a matrix power is governed by its largest absolute eigenvalue, and compute that number, including for complex eigenvalues.
  - Predict the size of a gradient after T steps with the rule (kappa times rho) to the power T, and state when the rule is exact and when it is only a guide.
  - Compare the prediction with a gradient measured by your own backward pass.
prereqs:
  - 046-power-iteration
  - 074-gradient-through-deep-stacks
  - 077-backprop-through-time
exercise: 078-vanishing-and-exploding-gradients
---

# Why gradients fade or blow up

You train the sensor model on runs of 50 readings. The label depends on the first reading: a spike at step 1 means the machine will overheat at step 50. The model learns to use the last few readings well, and it ignores the spike at step 1, even though the spike decides the answer. Training for longer does not help. Where did the learning signal go?

The weights learn from the gradient that reaches them. The gradient for step 1 has to travel back through 49 cells to get there, and the previous chapter showed that it gets multiplied at every cell. A product of 49 numbers can be extremely small, and then the weights hear almost nothing about step 1, even though the forward pass carries the information fine. It can also be extremely large, which breaks training in a different and louder way.

This chapter explains why, in one picture and one rule, and then measures it with the backward pass you wrote.

## What you already know

You already use these ideas:

- In a backward pass through time, the carry is multiplied by a tanh slope and by $W_{hh}$ at every step.
- The tanh slope $1 - h^2$ is between 0 and 1.
- An eigenvector of a square matrix $W$ is a direction $v$ that $W$ only stretches: $Wv = \lambda v$ for a number $\lambda$, its eigenvalue.
- Repeating a multiplication by a matrix, as in power iteration, lines a vector up with one direction.
- In a deep stack of plain layers, many factors below one shrink the gradient toward zero.

## One factor per step

Start with the simplest case, where the state is a single number and the weight is a single number $w$. Ignore the tanh slope for a moment. Each step back multiplies the carry by $w$, so after $T$ steps the gradient is $w^T$ times its starting size. That is compound interest. With $w = 0.9$, thirty steps leave $0.9^{30} = 0.042$ of the signal. With $w = 1.1$, thirty steps give $1.1^{30} = 17.4$. With $w = 1$ nothing changes. A weight 10% away from one, in either direction, is enough to wreck a long chain.

Next, put the tanh slope back in. In the running example of the last chapter, each step back multiplied the gradient by $0.8$ for $W_{hh}$ and by a slope near $0.6$. That is about $0.48$ per step, so a gradient 30 steps back is $0.48^{30} \approx 3 \times 10^{-10}$ of the gradient at the end. The step-1 spike in the opening is out of reach.

For a real cell, the state is a vector and $W_{hh}$ is a matrix, so the question becomes what repeated multiplication by a matrix does. Figure 1 gives the answer first and the explanation follows. It plots the size of the gradient that reaches a state $t$ steps back, for three recurrent matrices that differ only in their spectral radius $\rho$, a number defined below. The lines are a rule; the dots are gradients measured with `bptt` from the previous chapter.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Gradient size against steps back on a log axis: lines for rho 0.9, 1.0, and 1.05 with measured dots on top">
<defs></defs>
<line x1="80" y1="260" x2="520" y2="260" class="stroke-grid"/>
<text x="72" y="264" text-anchor="end" class="fig-small">1e-5</text>
<line x1="80" y1="233.8" x2="520" y2="233.8" class="stroke-grid"/>
<line x1="80" y1="207.5" x2="520" y2="207.5" class="stroke-grid"/>
<text x="72" y="211.5" text-anchor="end" class="fig-small">1e-3</text>
<line x1="80" y1="181.2" x2="520" y2="181.2" class="stroke-grid"/>
<line x1="80" y1="155" x2="520" y2="155" class="stroke-grid"/>
<text x="72" y="159" text-anchor="end" class="fig-small">1e-1</text>
<line x1="80" y1="128.8" x2="520" y2="128.8" class="stroke-grid"/>
<line x1="80" y1="102.5" x2="520" y2="102.5" class="stroke-grid"/>
<text x="72" y="106.5" text-anchor="end" class="fig-small">1e1</text>
<line x1="80" y1="76.2" x2="520" y2="76.2" class="stroke-grid"/>
<line x1="80" y1="50" x2="520" y2="50" class="stroke-grid"/>
<text x="72" y="54" text-anchor="end" class="fig-small">1e3</text>
<text x="72" y="132.8" text-anchor="end" class="fig-small">1</text>
<text x="80" y="282" text-anchor="middle" class="fig-small">0</text>
<text x="168" y="282" text-anchor="middle" class="fig-small">20</text>
<text x="256" y="282" text-anchor="middle" class="fig-small">40</text>
<text x="344" y="282" text-anchor="middle" class="fig-small">60</text>
<text x="432" y="282" text-anchor="middle" class="fig-small">80</text>
<text x="520" y="282" text-anchor="middle" class="fig-small">100</text>
<text x="300" y="302" text-anchor="middle">steps back</text>
<text x="20" y="28" class="fig-bold">gradient size relative to the last step</text>
<polyline points="80,128.8 84.4,130 88.8,131.2 93.2,132.4 97.6,133.6 102,134.8 106.4,136 110.8,137.2 115.2,138.4 119.6,139.6 124,140.8 128.4,142 132.8,143.2 137.2,144.4 141.6,145.6 146,146.8 150.4,148 154.8,149.2 159.2,150.4 163.6,151.6 168,152.8 172.4,154 176.8,155.2 181.2,156.4 185.6,157.6 190,158.8 194.4,160 198.8,161.2 203.2,162.4 207.6,163.6 212,164.8 216.4,166 220.8,167.2 225.2,168.4 229.6,169.6 234,170.8 238.4,172 242.8,173.2 247.2,174.4 251.6,175.6 256,176.8 260.4,178 264.8,179.2 269.2,180.4 273.6,181.6 278,182.8 282.4,184 286.8,185.2 291.2,186.4 295.6,187.6 300,188.8 304.4,190 308.8,191.2 313.2,192.4 317.6,193.6 322,194.8 326.4,196 330.8,197.2 335.2,198.4 339.6,199.6 344,200.8 348.4,202 352.8,203.2 357.2,204.4 361.6,205.6 366,206.8 370.4,208 374.8,209.2 379.2,210.4 383.6,211.6 388,212.8 392.4,214 396.8,215.2 401.2,216.4 405.6,217.6 410,218.8 414.4,220 418.8,221.2 423.2,222.4 427.6,223.6 432,224.8 436.4,226 440.8,227.2 445.2,228.4 449.6,229.6 454,230.8 458.4,232 462.8,233.2 467.2,234.4 471.6,235.7 476,236.9 480.4,238.1 484.8,239.3 489.2,240.5 493.6,241.7 498,242.9 502.4,244.1 506.8,245.3 511.2,246.5 515.6,247.7 520,248.9" class="stroke-a"/>
<circle cx="124" cy="140.8" r="3.5" class="fill-a"/>
<circle cx="168" cy="152.8" r="3.5" class="fill-a"/>
<circle cx="212" cy="164.8" r="3.5" class="fill-a"/>
<circle cx="256" cy="176.8" r="3.5" class="fill-a"/>
<circle cx="300" cy="188.9" r="3.5" class="fill-a"/>
<circle cx="344" cy="200.9" r="3.5" class="fill-a"/>
<circle cx="388" cy="212.9" r="3.5" class="fill-a"/>
<circle cx="432" cy="224.9" r="3.5" class="fill-a"/>
<circle cx="476" cy="236.9" r="3.5" class="fill-a"/>
<circle cx="520" cy="248.9" r="3.5" class="fill-a"/>
<text x="528" y="252.9" class="fig-small">ρ = 0.9</text>
<polyline points="80,128.8 84.4,128.8 88.8,128.8 93.2,128.8 97.6,128.8 102,128.8 106.4,128.8 110.8,128.8 115.2,128.8 119.6,128.8 124,128.8 128.4,128.8 132.8,128.8 137.2,128.8 141.6,128.8 146,128.8 150.4,128.8 154.8,128.8 159.2,128.8 163.6,128.8 168,128.8 172.4,128.8 176.8,128.8 181.2,128.8 185.6,128.8 190,128.8 194.4,128.8 198.8,128.8 203.2,128.8 207.6,128.8 212,128.8 216.4,128.8 220.8,128.8 225.2,128.8 229.6,128.8 234,128.8 238.4,128.8 242.8,128.8 247.2,128.8 251.6,128.8 256,128.8 260.4,128.8 264.8,128.8 269.2,128.8 273.6,128.8 278,128.8 282.4,128.8 286.8,128.8 291.2,128.8 295.6,128.8 300,128.8 304.4,128.8 308.8,128.8 313.2,128.8 317.6,128.8 322,128.8 326.4,128.8 330.8,128.8 335.2,128.8 339.6,128.8 344,128.8 348.4,128.8 352.8,128.8 357.2,128.8 361.6,128.8 366,128.8 370.4,128.8 374.8,128.8 379.2,128.8 383.6,128.8 388,128.8 392.4,128.8 396.8,128.8 401.2,128.8 405.6,128.8 410,128.8 414.4,128.8 418.8,128.8 423.2,128.8 427.6,128.8 432,128.8 436.4,128.8 440.8,128.8 445.2,128.8 449.6,128.8 454,128.8 458.4,128.8 462.8,128.8 467.2,128.8 471.6,128.8 476,128.8 480.4,128.8 484.8,128.8 489.2,128.8 493.6,128.8 498,128.8 502.4,128.8 506.8,128.8 511.2,128.8 515.6,128.8 520,128.8" class="stroke-muted"/>
<circle cx="124" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="168" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="212" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="256" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="300" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="344" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="388" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="432" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="476" cy="128.8" r="3.5" class="fill-ink"/>
<circle cx="520" cy="128.8" r="3.5" class="fill-ink"/>
<text x="528" y="132.8" class="fig-small">ρ = 1</text>
<polyline points="80,128.8 84.4,128.2 88.8,127.6 93.2,127.1 97.6,126.5 102,126 106.4,125.4 110.8,124.9 115.2,124.3 119.6,123.7 124,123.2 128.4,122.6 132.8,122.1 137.2,121.5 141.6,121 146,120.4 150.4,119.9 154.8,119.3 159.2,118.7 163.6,118.2 168,117.6 172.4,117.1 176.8,116.5 181.2,116 185.6,115.4 190,114.8 194.4,114.3 198.8,113.7 203.2,113.2 207.6,112.6 212,112.1 216.4,111.5 220.8,111 225.2,110.4 229.6,109.8 234,109.3 238.4,108.7 242.8,108.2 247.2,107.6 251.6,107.1 256,106.5 260.4,105.9 264.8,105.4 269.2,104.8 273.6,104.3 278,103.7 282.4,103.2 286.8,102.6 291.2,102.1 295.6,101.5 300,100.9 304.4,100.4 308.8,99.8 313.2,99.3 317.6,98.7 322,98.2 326.4,97.6 330.8,97 335.2,96.5 339.6,95.9 344,95.4 348.4,94.8 352.8,94.3 357.2,93.7 361.6,93.2 366,92.6 370.4,92 374.8,91.5 379.2,90.9 383.6,90.4 388,89.8 392.4,89.3 396.8,88.7 401.2,88.1 405.6,87.6 410,87 414.4,86.5 418.8,85.9 423.2,85.4 427.6,84.8 432,84.3 436.4,83.7 440.8,83.1 445.2,82.6 449.6,82 454,81.5 458.4,80.9 462.8,80.4 467.2,79.8 471.6,79.2 476,78.7 480.4,78.1 484.8,77.6 489.2,77 493.6,76.5 498,75.9 502.4,75.4 506.8,74.8 511.2,74.2 515.6,73.7 520,73.1" class="stroke-b"/>
<circle cx="124" cy="123.2" r="3.5" class="fill-b"/>
<circle cx="168" cy="117.7" r="3.5" class="fill-b"/>
<circle cx="212" cy="112.1" r="3.5" class="fill-b"/>
<circle cx="256" cy="106.6" r="3.5" class="fill-b"/>
<circle cx="300" cy="101" r="3.5" class="fill-b"/>
<circle cx="344" cy="95.5" r="3.5" class="fill-b"/>
<circle cx="388" cy="89.9" r="3.5" class="fill-b"/>
<circle cx="432" cy="84.5" r="3.5" class="fill-b"/>
<circle cx="476" cy="79.3" r="3.5" class="fill-b"/>
<circle cx="520" cy="74.5" r="3.5" class="fill-b"/>
<text x="528" y="77.1" class="fig-small">ρ = 1.05</text>
</svg>
Figure 1. Lines are the prediction ρ^t and dots are gradients measured with the backward pass from the previous chapter, on matrices with that spectral radius. Notice that the dots sit on or just under the lines (tanh flattens a little once the state has grown): after 100 steps the ρ = 0.9 signal is down to 0.00003 and the ρ = 1.05 signal is up to about 117, against a prediction of 131.5.
```

The ρ = 0.9 line dives, the ρ = 1 line stays flat, and the ρ = 1.05 line climbs, and the measured dots lie on the lines. The rest of the chapter says what $\rho$ is and why it controls the slope of each line.

## Why an eigenvalue appears

Take the matrix $W = \mathrm{diag}(0.5, 0.9)$ and the vector $(1, 1)$. Multiplying by $W$ halves the first entry and takes 90% of the second, so after $t$ steps the vector is $(0.5^t, 0.9^t)$. After five steps that is $(0.031, 0.590)$. The first part has almost vanished, and the second part is still most of the vector. Figure 2 draws this.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Repeatedly applying diag(0.5, 0.9) to the vector (1, 1) squeezes it toward the 0.9 direction">
<defs></defs>
<text x="20" y="28" class="fig-bold">Repeated multiplication by W = diag(0.5, 0.9)</text>
<line x1="70" y1="250" x2="270" y2="250" class="stroke-muted"/>
<line x1="70" y1="250" x2="70" y2="50" class="stroke-muted"/>
<text x="272" y="266" text-anchor="end" class="fig-small">0.5 direction</text>
<text x="78" y="48" class="fig-small">0.9 direction</text>
<polyline points="260,60 165,79 117.5,96.1 93.8,111.5 81.9,125.3 75.9,137.8" class="stroke-b"/>
<circle cx="260" cy="60" r="4" class="fill-b"/>
<circle cx="165" cy="79" r="4" class="fill-b"/>
<circle cx="117.5" cy="96.1" r="4" class="fill-b"/>
<circle cx="93.8" cy="111.5" r="4" class="fill-b"/>
<circle cx="81.9" cy="125.3" r="4" class="fill-b"/>
<circle cx="75.9" cy="137.8" r="4" class="fill-b"/>
<text x="264" y="48" text-anchor="end" class="fig-small">t=0: (1, 1)</text>
<text x="159" y="67" text-anchor="middle" class="fig-small">t=1: (0.5, 0.9)</text>
<text x="125.5" y="112.1" class="fig-small">t=2</text>
<text x="101.8" y="127.5" class="fig-small">t=3</text>
<text x="89.9" y="143.3" class="fig-small">t=4</text>
<text x="83.9" y="159.8" class="fig-small">t=5</text>
<text x="460" y="28" text-anchor="middle" class="fig-bold">Size of each component</text>
<line x1="355" y1="250" x2="590" y2="250" class="stroke-muted"/>
<rect x="360" y="80" width="14" height="170" class="fill-a"/>
<rect x="376" y="80" width="14" height="170" class="fill-b"/>
<text x="375" y="266" text-anchor="middle" class="fig-small">0</text>
<rect x="398" y="165" width="14" height="85" class="fill-a"/>
<rect x="414" y="97" width="14" height="153" class="fill-b"/>
<text x="413" y="266" text-anchor="middle" class="fig-small">1</text>
<rect x="436" y="207.5" width="14" height="42.5" class="fill-a"/>
<rect x="452" y="112.3" width="14" height="137.7" class="fill-b"/>
<text x="451" y="266" text-anchor="middle" class="fig-small">2</text>
<rect x="474" y="228.8" width="14" height="21.2" class="fill-a"/>
<rect x="490" y="126.1" width="14" height="123.9" class="fill-b"/>
<text x="489" y="266" text-anchor="middle" class="fig-small">3</text>
<rect x="512" y="239.4" width="14" height="10.6" class="fill-a"/>
<rect x="528" y="138.5" width="14" height="111.5" class="fill-b"/>
<text x="527" y="266" text-anchor="middle" class="fig-small">4</text>
<rect x="550" y="244.7" width="14" height="5.3" class="fill-a"/>
<rect x="566" y="149.6" width="14" height="100.4" class="fill-b"/>
<text x="565" y="266" text-anchor="middle" class="fig-small">5</text>
<text x="475" y="286" text-anchor="middle" class="fig-small">steps t</text>
<rect x="470" y="44" width="12" height="12" class="fill-a"/>
<text x="486" y="55" class="fig-small">0.5 direction</text>
<rect x="470" y="62" width="12" height="12" class="fill-b"/>
<text x="486" y="73" class="fig-small">0.9 direction</text>
</svg>
Figure 2. Each multiplication by W keeps 0.9 of the vertical part and only 0.5 of the horizontal part. After five steps the horizontal part is 0.03 and the vertical part is 0.59, so the direction that survives longest, the one with the largest eigenvalue, decides what is left.
```

The reason is that along an eigenvector, multiplying by $W$ is the same as multiplying by a number. If $Wv = \lambda v$, then $W^2 v = \lambda W v = \lambda^2 v$, and $W^T v = \lambda^T v$. A diagonal matrix has the axes as its eigenvectors, so each axis is shrunk by its own power. A general vector is a mix of eigenvectors, so after many steps the part with the largest $|\lambda|$ outgrows the rest. That is why one number, the largest absolute eigenvalue, tells you the growth rate. It is called the **spectral radius**:

$$
\rho(W) = \max_i |\lambda_i|
$$

Read it aloud: take every eigenvalue, measure its distance from zero, and keep the biggest. Here $\rho = 0.9$. The distance matters because the eigenvalues of a real matrix can be complex. A rotation by 90 degrees has no real eigenvalue, its eigenvalues are $i$ and $-i$, and each has absolute value 1, which is right, because a rotation neither shrinks nor stretches a vector.

```python
rho = np.max(np.abs(np.linalg.eigvals(W)))
```

The usual mistake is leaving out `np.abs`, so a large negative or complex eigenvalue is missed.

## The rule, and when to trust it

Put the two factors together. Each step back multiplies the gradient by the tanh slope, which we summarise as one number $\kappa$ between 0 and 1, and by the matrix, which acts like $\rho$. After $T$ steps, the gradient has size about

$$
\left(\kappa\,\rho(W_{hh})\right)^T
$$

Read it aloud: multiply the typical slope by the spectral radius to get a rate per step, then raise the rate to the number of steps. If the rate is below one, the gradient vanishes. If it is above one, the gradient can explode.

Be careful about what this rule promises. It is exact when $W_{hh}$ is a rotation scaled by $\rho$, as in Figure 1, and when the slope is the same at every step. That is why the dots land on the lines: in that run the state stays tiny, so $\kappa$ is 1. For a general matrix the rule gives the long-run rate, and over a short stretch the gradient can grow first and shrink later. The slope also changes at every step, because the state changes.

The two directions are not equally reliable. With $\kappa \le 1$, a spectral radius below one is a guarantee that the gradient vanishes. A spectral radius above one is only a warning. Take the same $\rho = 1.05$ matrix, but start the state with entries of size about 1 instead of 0.001. Over four seeded runs the gradient after 100 steps came out between 0.1 and 0.7, not 131.5. Large states sit on the flat parts of tanh, which pushes $\kappa$ far below one and cancels the growth. Explosions happen where the state is small enough for the slope to stay near one, or where the weights are large enough to win against the squash.

## Why not set the spectral radius to exactly one?

You can start training with $\rho = 1$, for example with an orthogonal matrix, and in Figure 1 that line stays flat. The trouble is that the flat line assumed a tiny state. In the same four seeded runs with a state of size about 1, the gradient after 100 steps of the $\rho = 1$ matrix came out between 0.03 and 0.16, because $\kappa$ was below one at most steps. Training also moves the weights away from $\rho = 1$ as soon as it starts. Nothing in a plain cell holds the rate at one, so the chapter "Why a recurrent net forgets, and the belt that fixes it" builds a path whose factor is held near one on purpose.

## How far back can the signal reach?

Turn the rate into a distance. Call a signal too faint to learn from when it falls below a fraction, say 1%, of the signal at the end. The smallest $T$ with $(\kappa\rho)^T < 0.01$ comes from taking logarithms. Because $\log(\kappa\rho)$ is negative when the rate is below one, dividing flips the inequality, so $T > \log(0.01)/\log(\kappa\rho)$. With $\kappa\rho = 0.9$ that gives $T = 44$, since $0.9^{44} = 0.0097$. With the running example's rate of 0.48, it is 7 steps. Figure 3 shows how fast the gap opens in both directions.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Bars on a log scale: after 10, 50, and 100 steps the signal shrinks to 0.349, 0.005, and 0.000027 at 0.9 per step, and grows to 1.63, 11.5, and 131.5 at 1.05 per step">
<defs></defs>
<text x="20" y="26" class="fig-bold">how much of the end-of-sequence signal reaches the step that far back</text>
<text x="322" y="50" text-anchor="end" class="fig-small fill-a">0.9 per step</text>
<text x="338" y="50" class="fig-small fill-b">1.05 per step</text>
<line x1="330" y1="56" x2="330" y2="232" class="stroke-muted"/>
<text x="20" y="100">10 steps back</text>
<rect x="312.6" y="80" width="17.4" height="28" class="fill-a"/>
<rect x="330" y="80" width="8.1" height="28" class="fill-b"/>
<text x="312.6" y="75" text-anchor="end" class="fig-small">0.349</text>
<text x="338.1" y="75" class="fig-small">1.63</text>
<text x="20" y="162">50 steps back</text>
<rect x="243.1" y="142" width="86.9" height="28" class="fill-a"/>
<rect x="330" y="142" width="40.3" height="28" class="fill-b"/>
<text x="243.1" y="137" text-anchor="end" class="fig-small">0.0052</text>
<text x="370.3" y="137" class="fig-small">11.5</text>
<text x="20" y="224">100 steps back</text>
<rect x="156.1" y="204" width="173.9" height="28" class="fill-a"/>
<rect x="330" y="204" width="80.5" height="28" class="fill-b"/>
<text x="156.1" y="199" text-anchor="end" class="fig-small">2.7e-5</text>
<text x="410.5" y="199" class="fig-small">131.5</text>
<text x="330" y="252" text-anchor="middle" class="fig-small">1 = no change; each bar's length is its number of powers of ten</text>
</svg>
Figure 3. Compound interest, measured in powers of ten. A factor of 0.9 per step sounds mild, yet after 100 steps the signal is about 37,000 times smaller; a factor of 1.05 sounds mild too, and it multiplies the signal by 131.5.
```

Fading and blowing up feel different in practice. An exploding gradient is loud: the weights jump and the loss turns into `nan`. A vanishing gradient is silent: the loss flattens at a mediocre value, and the model looks like it has finished learning. The next chapter has a cure for the loud failure, and the section after it says why that cure cannot fix the silent one.

> [!KEY]
> The gradient to a state $T$ steps back is a product of one factor per step, so it behaves like $(\kappa\rho)^T$. With a rate below one, the step-1 spike in the opening has almost no voice in training, and that is where the signal went.

> [!RECAP]
> - The carry is multiplied once per step, so a gradient crossing $T$ steps is a $T$-fold product.
> - Along an eigenvector, a matrix acts as the number $\lambda$, so the largest absolute eigenvalue $\rho$ sets the long-run rate.
> - The rule $(\kappa\rho)^T$ is exact for a scaled rotation and a constant slope, and is a guide otherwise.
> - $\rho < 1$ with $\kappa \le 1$ guarantees vanishing. $\rho > 1$ warns of explosion but does not guarantee it.
> - $T > \log(\text{threshold}) / \log(\kappa\rho)$ gives the steps until the signal drops below a threshold.

## Build it step by step

The checks call the functions in this order. The forward pass and `bptt` from the earlier chapters are already written for you, so you can compare your prediction with a measured gradient.

1. **`spectral_radius(W)`.** `W` is a square matrix. Return the largest absolute eigenvalue as a float. Use `np.linalg.eigvals`, `np.abs`, and `np.max`. The usual mistake is forgetting `np.abs`, which fails on the rotation.
2. **`gradient_scale_over_time(W, kappa, T)`.** Return `(kappa * rho) ** T` as a float, with `rho` from your first function. The usual mistake is applying the power to $\rho$ alone and then multiplying by $\kappa$.
3. **`steps_until_below(rho, kappa, threshold)`.** Return the smallest whole number $T$ with $(\kappa\rho)^T <$ threshold, or `None` when $\kappa\rho$ is 1 or more. Use logarithms, and round up to the next whole number. The usual mistake is an off-by-one when the answer is exactly a whole number, as with $0.5^2 = 0.25$ and a threshold of 0.25, where two steps is not below.

## Cost and scaling

The prediction costs one eigenvalue computation, which grows with the cube of the hidden size $d_h$, and is paid once per weight matrix, not once per step. The measurement costs one backward pass.

## Where this breaks

- The rule treats the slope $\kappa$ as one number, while the real slope changes at every step. Treat the result as an order of magnitude.
- For a matrix that is not a scaled rotation, the gradient can grow for a while before it decays, so the early part of a curve can sit above the rule.
- Fixing the explosion does not fix the fade. Both come from the same product, but their cures differ.

## The exercise

You build `spectral_radius`, `gradient_scale_over_time`, and `steps_until_below`. The checks use a diagonal matrix, a rotation whose eigenvalues are complex, and a random symmetric matrix compared against an independent computation. They test `steps_until_below` against counting one step at a time, including the case that returns `None`. The last check runs your prediction beside the backward pass at $\rho = 0.9$ and $\rho = 1.05$ and requires the two to agree within a factor of two.
