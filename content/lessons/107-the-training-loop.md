---
title: "One training step and the loop around it"
slug: 107-the-training-loop
part: "Part 6 — Transformers"
summary: Give every weight its own step with AdamW, then wire loss, backward, clip, schedule, and update into one training step.
minutes: 45
gpu: true
objectives:
  - Explain why one learning rate suits weights with very different gradient sizes badly, and how dividing by a running size fixes it.
  - Compute one AdamW update by hand, including bias correction and decoupled weight decay.
  - Order the five parts of a training step and say why the clip comes before the update.
  - Build the update and a full train_step, and check them against hand-computed values.
prereqs:
  - 066-adam
  - 067-adamw
  - 105-next-token-loss
  - 106-safe-steps
exercise: 107-the-training-loop
---

# One training step and the loop around it

Two weights sit in the same model. The gradient for one is near 0.001, and the gradient for the other is near 100. Take a learning rate of 0.01 and plain gradient descent. The first weight moves 0.00001 per step, which is frozen. The second moves 1 per step, which is a leap. Neither rate is right for both.

You cannot tune a rate per weight by hand, because a transformer has millions of weights and their gradient sizes change as training goes on. So how can one learning rate be fair to every weight at once? And once it is, what does it take to put the loss, the clip, the schedule, and the optimiser together into one training step?

This chapter recalls the answer, AdamW, in the first half, and spends the second half on the wiring. You write the update for one tensor, then the step that calls everything you have built so far.

## What you already know

Five ideas carry this chapter:

- Adam keeps two running averages for each weight: $m$ for the gradient and $v$ for its square, and steps by $m$ over the square root of $v$ (see "Adam: a step size for every weight").
- Both averages start at zero, so each is divided by $1 - \beta^t$, where $t$ counts updates from 1.
- AdamW shrinks each weight by a fixed fraction after Adam's division, so the shrink does not depend on the gradient size (see "AdamW: decaying weights the right way").
- The loss, from "How a language model is scored: next-token loss and perplexity", is one number that `backward()` turns into a gradient for every weight.
- The schedule and the clip are from "Taking safe steps: warmup, cosine decay, and gradient clipping".

## The step in five boxes

Figure 1 is the whole chapter in one picture. A training step takes the weights and a batch, and returns updated weights.

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="One training step as five boxes in a row: forward, loss, backward, clip, update, with the learning rate and the optimiser state feeding the update">
<defs><marker id="c107-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="18" y="60" width="104" height="66" rx="8" class="fill-sunk stroke-muted"/>
<text x="70" y="86" text-anchor="middle" class="fig-bold">1  forward</text>
<text x="70" y="108" text-anchor="middle" class="fig-small">model_fn</text>
<text x="70" y="146" text-anchor="middle" class="fig-small">logits (B, T, V)</text>
<line x1="124" y1="93" x2="140" y2="93" class="stroke-ink" marker-end="url(#c107-arrow)"/>
<rect x="142" y="60" width="104" height="66" rx="8" class="fill-soft-c stroke-c"/>
<text x="194" y="86" text-anchor="middle" class="fig-bold">2  loss</text>
<text x="194" y="108" text-anchor="middle" class="fig-small">causal_lm_loss</text>
<text x="194" y="146" text-anchor="middle" class="fig-small">one number</text>
<line x1="248" y1="93" x2="264" y2="93" class="stroke-ink" marker-end="url(#c107-arrow)"/>
<rect x="266" y="60" width="104" height="66" rx="8" class="fill-sunk stroke-muted"/>
<text x="318" y="86" text-anchor="middle" class="fig-bold">3  backward</text>
<text x="318" y="108" text-anchor="middle" class="fig-small">loss.backward()</text>
<text x="318" y="146" text-anchor="middle" class="fig-small">a gradient per weight</text>
<line x1="372" y1="93" x2="388" y2="93" class="stroke-ink" marker-end="url(#c107-arrow)"/>
<rect x="390" y="60" width="104" height="66" rx="8" class="fill-soft-c stroke-c"/>
<text x="442" y="86" text-anchor="middle" class="fig-bold">4  clip</text>
<text x="442" y="108" text-anchor="middle" class="fig-small">clip_global_norm</text>
<text x="442" y="146" text-anchor="middle" class="fig-small">norm at most the limit</text>
<line x1="496" y1="93" x2="512" y2="93" class="stroke-ink" marker-end="url(#c107-arrow)"/>
<rect x="514" y="60" width="104" height="66" rx="8" class="fill-soft-b stroke-b"/>
<text x="566" y="86" text-anchor="middle" class="fig-bold">5  update</text>
<text x="566" y="108" text-anchor="middle" class="fig-small">adamw_update</text>
<text x="566" y="146" text-anchor="middle" class="fig-small">updated weights</text>
<text x="70" y="44" text-anchor="middle" class="fig-small">params and a batch</text>
<line x1="70" y1="50" x2="70" y2="58" class="stroke-ink" marker-end="url(#c107-arrow)"/>
<rect x="364" y="184" width="150" height="40" rx="8" class="fill-soft-c stroke-c"/>
<text x="439" y="202" text-anchor="middle" class="fig-small">learning rate for this step</text>
<text x="439" y="217" text-anchor="middle" class="fig-small">warmup_cosine_lr</text>
<line x1="514" y1="204" x2="552" y2="164" class="stroke-ink" marker-end="url(#c107-arrow)"/>
<rect x="508" y="236" width="124" height="40" rx="8" class="fill-soft-a stroke-a"/>
<text x="570" y="254" text-anchor="middle" class="fig-small">opt_state</text>
<text x="570" y="269" text-anchor="middle" class="fig-small">step, m, v</text>
<line x1="570" y1="234" x2="570" y2="166" class="stroke-ink" marker-end="url(#c107-arrow)"/>
<rect x="18" y="246" width="16" height="12" rx="2" class="fill-soft-c stroke-c"/>
<text x="42" y="256" text-anchor="start" class="fig-small">written in earlier chapters</text>
<rect x="18" y="266" width="16" height="12" rx="2" class="fill-soft-b stroke-b"/>
<text x="42" y="276" text-anchor="start" class="fig-small">you write it in this chapter</text>
<rect x="18" y="286" width="16" height="12" rx="2" class="fill-sunk stroke-muted"/>
<text x="42" y="296" text-anchor="start" class="fig-small">the model and autograd</text>
</svg>
Figure 1. One training step is five boxes in a row. The weights and a batch go in, and updated weights come out. The green boxes are functions you wrote earlier, and the orange one is the function you write in this chapter. The clip sits before the update so the optimiser never sees an oversized gradient.
```

The boxes run in an order that has reasons. The forward pass turns inputs into logits, and the loss turns logits and targets into one number. The backward pass fills a gradient for every weight. Only then can you clip, because clipping needs the whole gradient to measure its length. The clip comes before the update because Adam stores averages of whatever gradient it is given. If an oversized gradient reached it, the spike would sit in $m$ and $v$ for many steps. Clipped first, it enters at a safe size.

The learning rate comes from the schedule. The loop counts updates from 1, and the schedule is asked about the update about to be applied, so the first update uses a rate of peak divided by warmup rather than 0. Last, the update returns fresh weight tensors. Autograd must not record the update as part of the next step's graph, and a fresh tensor starts with no stored gradient, so you never have to zero the old ones.

## Why one rate is unfair, and the fix

Figure 2 shows the opening example over six steps.

```figure
<svg viewBox="0 0 620 280" role="img" aria-label="Step length per step for two weights, one with gradients near 0.001 and one near 100, under plain descent and under Adam">
<text x="180" y="48" text-anchor="middle" class="fig-bold">plain descent, rate 0.01</text>
<text x="180" y="66" text-anchor="middle" class="fig-small">step length = rate × gradient</text>
<line x1="56" y1="215" x2="60" y2="215" class="stroke-muted"/>
<text x="52" y="219" text-anchor="end" class="fig-small">0</text>
<line x1="56" y1="171.7" x2="60" y2="171.7" class="stroke-muted"/>
<text x="52" y="175.7" text-anchor="end" class="fig-small">0.4</text>
<line x1="60" y1="171.7" x2="310" y2="171.7" class="stroke-grid"/>
<line x1="56" y1="128.3" x2="60" y2="128.3" class="stroke-muted"/>
<text x="52" y="132.3" text-anchor="end" class="fig-small">0.8</text>
<line x1="60" y1="128.3" x2="310" y2="128.3" class="stroke-grid"/>
<line x1="56" y1="85" x2="60" y2="85" class="stroke-muted"/>
<text x="52" y="89" text-anchor="end" class="fig-small">1.2</text>
<line x1="60" y1="85" x2="310" y2="85" class="stroke-grid"/>
<line x1="60" y1="215" x2="310" y2="215" class="stroke-muted"/>
<line x1="60" y1="215" x2="60" y2="79" class="stroke-muted"/>
<rect x="72" y="215" width="14" height="0" class="fill-a"/>
<rect x="88" y="95.8" width="14" height="119.2" class="fill-d"/>
<text x="87" y="231" text-anchor="middle" class="fig-small">1</text>
<rect x="112" y="215" width="14" height="0" class="fill-a"/>
<rect x="128" y="117.5" width="14" height="97.5" class="fill-d"/>
<text x="127" y="231" text-anchor="middle" class="fig-small">2</text>
<rect x="152" y="215" width="14" height="0" class="fill-a"/>
<rect x="168" y="101.2" width="14" height="113.8" class="fill-d"/>
<text x="167" y="231" text-anchor="middle" class="fig-small">3</text>
<rect x="192" y="215" width="14" height="0" class="fill-a"/>
<rect x="208" y="112.1" width="14" height="102.9" class="fill-d"/>
<text x="207" y="231" text-anchor="middle" class="fig-small">4</text>
<rect x="232" y="215" width="14" height="0" class="fill-a"/>
<rect x="248" y="106.7" width="14" height="108.3" class="fill-d"/>
<text x="247" y="231" text-anchor="middle" class="fig-small">5</text>
<rect x="272" y="215" width="14" height="0" class="fill-a"/>
<rect x="288" y="104.5" width="14" height="110.5" class="fill-d"/>
<text x="287" y="231" text-anchor="middle" class="fig-small">6</text>
<text x="185" y="251" text-anchor="middle">step</text>
<text x="470" y="48" text-anchor="middle" class="fig-bold">Adam, rate 0.01</text>
<text x="470" y="66" text-anchor="middle" class="fig-small">step length = rate × m̂ / √v̂</text>
<line x1="346" y1="215" x2="350" y2="215" class="stroke-muted"/>
<text x="342" y="219" text-anchor="end" class="fig-small">0</text>
<line x1="346" y1="171.7" x2="350" y2="171.7" class="stroke-muted"/>
<text x="342" y="175.7" text-anchor="end" class="fig-small">0.004</text>
<line x1="350" y1="171.7" x2="600" y2="171.7" class="stroke-grid"/>
<line x1="346" y1="128.3" x2="350" y2="128.3" class="stroke-muted"/>
<text x="342" y="132.3" text-anchor="end" class="fig-small">0.008</text>
<line x1="350" y1="128.3" x2="600" y2="128.3" class="stroke-grid"/>
<line x1="346" y1="85" x2="350" y2="85" class="stroke-muted"/>
<text x="342" y="89" text-anchor="end" class="fig-small">0.012</text>
<line x1="350" y1="85" x2="600" y2="85" class="stroke-grid"/>
<line x1="350" y1="215" x2="600" y2="215" class="stroke-muted"/>
<line x1="350" y1="215" x2="350" y2="79" class="stroke-muted"/>
<rect x="362" y="106.7" width="14" height="108.3" class="fill-a"/>
<rect x="378" y="106.7" width="14" height="108.3" class="fill-d"/>
<text x="377" y="231" text-anchor="middle" class="fig-small">1</text>
<rect x="402" y="109.4" width="14" height="105.6" class="fill-a"/>
<rect x="418" y="107.5" width="14" height="107.5" class="fill-d"/>
<text x="417" y="231" text-anchor="middle" class="fig-small">2</text>
<rect x="442" y="108.2" width="14" height="106.8" class="fill-a"/>
<rect x="458" y="107.1" width="14" height="107.9" class="fill-d"/>
<text x="457" y="231" text-anchor="middle" class="fig-small">3</text>
<rect x="482" y="108.4" width="14" height="106.6" class="fill-a"/>
<rect x="498" y="107.2" width="14" height="107.8" class="fill-d"/>
<text x="497" y="231" text-anchor="middle" class="fig-small">4</text>
<rect x="522" y="108" width="14" height="107" class="fill-a"/>
<rect x="538" y="107.1" width="14" height="107.9" class="fill-d"/>
<text x="537" y="231" text-anchor="middle" class="fig-small">5</text>
<rect x="562" y="107.5" width="14" height="107.5" class="fill-a"/>
<rect x="578" y="106.9" width="14" height="108.1" class="fill-d"/>
<text x="577" y="231" text-anchor="middle" class="fig-small">6</text>
<text x="475" y="251" text-anchor="middle">step</text>
<rect x="72" y="10" width="10" height="10" class="fill-a"/>
<text x="86" y="19" text-anchor="start" class="fig-small">weight A, gradient about 0.001</text>
<rect x="362" y="10" width="10" height="10" class="fill-d"/>
<text x="376" y="19" text-anchor="start" class="fig-small">weight B, gradient about 100</text>
<text x="185" y="273" text-anchor="middle" class="fig-small">weight A moves about 0.00001 per step, too short to draw</text>
</svg>
Figure 2. Step lengths of two weights over six steps. Under plain descent, the weight with gradients near 100 moves about 1 per step and the one near 0.001 moves about 0.00001, too little to draw. Under Adam, dividing by each weight's own typical size puts both at about 0.01, the learning rate. The two panels use different vertical scales.
```

Plain descent makes the step length proportional to the gradient, so a gradient 100,000 times larger gives a step 100,000 times longer. Adam measures each weight in its own units. The average $m$ says which way and how hard the gradient pushes, and $\sqrt{v}$ is the typical size of the gradient, a root-mean-square. Their ratio is near $+1$ when the gradient keeps pushing the same way, near $-1$ when it keeps pushing the other way, and near 0 when it flips sign at random. Every weight with a steady slope therefore moves about one learning rate per step, whatever its gradient size. Why not divide each gradient by its own absolute value? A weight whose gradient flips between $+3$ and $-3$ would then jump a full step back and forth, although it has no direction. The average in the numerator cancels that noise.

Take $\theta = 1$, $g = 0.5$ at the first update, with learning rate 0.1, $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay $\lambda = 0.1$, and both averages at 0. Here $\theta$ is the weight. The averages become $m = 0.1 \times 0.5 = 0.05$ and $v = 0.05 \times 0.25 = 0.0125$.

Those are too small, and Figure 3 shows why.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="The raw running average of a constant gradient 0.5 with beta 0.9 climbing from 0.05 toward 0.5, against the corrected average that equals 0.5 from step 1">
<line x1="70" y1="240" x2="70" y2="245" class="stroke-muted"/>
<text x="70" y="260" text-anchor="middle" class="fig-small">1</text>
<line x1="133.4" y1="240" x2="133.4" y2="245" class="stroke-muted"/>
<text x="133.4" y="260" text-anchor="middle" class="fig-small">5</text>
<line x1="212.8" y1="240" x2="212.8" y2="245" class="stroke-muted"/>
<text x="212.8" y="260" text-anchor="middle" class="fig-small">10</text>
<line x1="292.1" y1="240" x2="292.1" y2="245" class="stroke-muted"/>
<text x="292.1" y="260" text-anchor="middle" class="fig-small">15</text>
<line x1="371.4" y1="240" x2="371.4" y2="245" class="stroke-muted"/>
<text x="371.4" y="260" text-anchor="middle" class="fig-small">20</text>
<line x1="450.7" y1="240" x2="450.7" y2="245" class="stroke-muted"/>
<text x="450.7" y="260" text-anchor="middle" class="fig-small">25</text>
<line x1="530" y1="240" x2="530" y2="245" class="stroke-muted"/>
<text x="530" y="260" text-anchor="middle" class="fig-small">30</text>
<line x1="65" y1="240" x2="70" y2="240" class="stroke-muted"/>
<text x="61" y="244" text-anchor="end" class="fig-small">0</text>
<line x1="65" y1="205" x2="70" y2="205" class="stroke-muted"/>
<text x="61" y="209" text-anchor="end" class="fig-small">0.1</text>
<line x1="70" y1="205" x2="530" y2="205" class="stroke-grid"/>
<line x1="65" y1="170" x2="70" y2="170" class="stroke-muted"/>
<text x="61" y="174" text-anchor="end" class="fig-small">0.2</text>
<line x1="70" y1="170" x2="530" y2="170" class="stroke-grid"/>
<line x1="65" y1="135" x2="70" y2="135" class="stroke-muted"/>
<text x="61" y="139" text-anchor="end" class="fig-small">0.3</text>
<line x1="70" y1="135" x2="530" y2="135" class="stroke-grid"/>
<line x1="65" y1="100" x2="70" y2="100" class="stroke-muted"/>
<text x="61" y="104" text-anchor="end" class="fig-small">0.4</text>
<line x1="70" y1="100" x2="530" y2="100" class="stroke-grid"/>
<line x1="65" y1="65" x2="70" y2="65" class="stroke-muted"/>
<text x="61" y="69" text-anchor="end" class="fig-small">0.5</text>
<line x1="70" y1="65" x2="530" y2="65" class="stroke-grid"/>
<line x1="65" y1="30" x2="70" y2="30" class="stroke-muted"/>
<text x="61" y="34" text-anchor="end" class="fig-small">0.6</text>
<line x1="70" y1="30" x2="530" y2="30" class="stroke-grid"/>
<line x1="70" y1="240" x2="530" y2="240" class="stroke-muted"/>
<line x1="70" y1="240" x2="70" y2="24" class="stroke-muted"/>
<line x1="70" y1="65" x2="530" y2="65" class="stroke-ink dashed"/>
<text x="530" y="57" text-anchor="end" class="fig-small">true gradient, 0.5</text>

<path d="M70,222.5 L85.9,206.8 L101.7,192.6 L117.6,179.8 L133.4,168.3 L149.3,158 L165.2,148.7 L181,140.3 L196.9,132.8 L212.8,126 L228.6,119.9 L244.5,114.4 L260.3,109.5 L276.2,105 L292.1,101 L307.9,97.4 L323.8,94.2 L339.7,91.3 L355.5,88.6 L371.4,86.3 L387.2,84.1 L403.1,82.2 L419,80.5 L434.8,79 L450.7,77.6 L466.6,76.3 L482.4,75.2 L498.3,74.2 L514.1,73.2 L530,72.4" fill="none" class="stroke-b" stroke-width="3"/>
<circle cx="70" cy="222.5" r="3" class="fill-b"/>
<circle cx="85.9" cy="206.8" r="3" class="fill-b"/>
<circle cx="101.7" cy="192.6" r="3" class="fill-b"/>
<circle cx="117.6" cy="179.8" r="3" class="fill-b"/>
<circle cx="133.4" cy="168.3" r="3" class="fill-b"/>
<circle cx="149.3" cy="158" r="3" class="fill-b"/>
<circle cx="165.2" cy="148.7" r="3" class="fill-b"/>
<circle cx="181" cy="140.3" r="3" class="fill-b"/>
<circle cx="196.9" cy="132.8" r="3" class="fill-b"/>
<circle cx="212.8" cy="126" r="3" class="fill-b"/>
<circle cx="228.6" cy="119.9" r="3" class="fill-b"/>
<circle cx="244.5" cy="114.4" r="3" class="fill-b"/>
<circle cx="260.3" cy="109.5" r="3" class="fill-b"/>
<circle cx="276.2" cy="105" r="3" class="fill-b"/>
<circle cx="292.1" cy="101" r="3" class="fill-b"/>
<circle cx="307.9" cy="97.4" r="3" class="fill-b"/>
<circle cx="323.8" cy="94.2" r="3" class="fill-b"/>
<circle cx="339.7" cy="91.3" r="3" class="fill-b"/>
<circle cx="355.5" cy="88.6" r="3" class="fill-b"/>
<circle cx="371.4" cy="86.3" r="3" class="fill-b"/>
<circle cx="387.2" cy="84.1" r="3" class="fill-b"/>
<circle cx="403.1" cy="82.2" r="3" class="fill-b"/>
<circle cx="419" cy="80.5" r="3" class="fill-b"/>
<circle cx="434.8" cy="79" r="3" class="fill-b"/>
<circle cx="450.7" cy="77.6" r="3" class="fill-b"/>
<circle cx="466.6" cy="76.3" r="3" class="fill-b"/>
<circle cx="482.4" cy="75.2" r="3" class="fill-b"/>
<circle cx="498.3" cy="74.2" r="3" class="fill-b"/>
<circle cx="514.1" cy="73.2" r="3" class="fill-b"/>
<circle cx="530" cy="72.4" r="3" class="fill-b"/>
<circle cx="70" cy="65" r="5" class="fill-a"/>
<circle cx="70" cy="222.5" r="5" class="fill-b"/>
<text x="82" y="226.5" text-anchor="start" class="fig-small">m = 0.05, ten times too small</text>
<text x="82" y="87" text-anchor="start" class="fig-small">m̂ = 0.5 from the first step</text>
<text x="276.2" y="124.5" text-anchor="start" class="fig-small">raw m climbs toward 0.5</text>
<text x="300" y="284" text-anchor="middle">step t</text>
<text transform="rotate(-90 22 135)" x="22" y="135" text-anchor="middle">average of the gradient</text>
</svg>
Figure 3. A constant gradient of 0.5 with β1 = 0.9. The raw average m starts at zero, so after one step it holds only 0.05, a tenth of the truth, and it takes dozens of steps to catch up. Dividing by 1 − 0.9^t gives m̂, which equals 0.5 at every step. The fix is large at first and fades as t grows.
```

An average that starts at zero has seen one number and a window full of zeros. With a steady gradient of 0.5, the raw $m$ is exactly $0.5 \times (1 - 0.9^t)$: 0.05 at $t = 1$, 0.095 at $t = 2$, and 0.33 at $t = 10$. Dividing by $1 - 0.9^t$ repairs it. The factor is 10 at $t = 1$, 5.3 at $t = 2$, 1.5 at $t = 10$, and 1.005 at $t = 50$, so it fades as the average fills. The same fix applies to $v$ with $\beta_2$. Here $\hat{m} = 0.05/0.1 = 0.5$ and $\hat{v} = 0.0125/0.05 = 0.25$, so $\sqrt{\hat{v}} = 0.5$ and the ratio is exactly 1: a full learning rate in the direction of the gradient. At $t = 0$ the divisor $1 - \beta^0$ is zero, which is why counting starts at 1.

The decay is separate. Each weight loses the fraction $\text{lr} \times \lambda = 0.1 \times 0.1 = 0.01$ of itself, and then the Adam step is subtracted:

$$
\theta \leftarrow \theta\,(1 - \text{lr}\cdot\lambda) - \text{lr}\,\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

Read it aloud: shrink the weight by a small fraction, then move it one learning rate against the normalised gradient. The small number $\epsilon$, $10^{-8}$, keeps the denominator away from zero for a weight that has had no gradient yet. For the example, $1 \times 0.99 - 0.1 \times 1 = 0.89$. With $\beta_2 = 0.95$, $v$ remembers about $1/(1 - 0.95) = 20$ steps. That is shorter than the 1,000 of the default, so it forgets a spike sooner, and it is common in language-model training. In code, for one weight tensor:

```python
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad**2
m_hat = m / (1 - beta1**step)
v_hat = v / (1 - beta2**step)
theta = theta * (1 - lr * weight_decay) - lr * m_hat / (v_hat**0.5 + eps)
```

Weight decay has one rule of thumb. Matrices get it, and biases and layer-norm gains do not. A gain starts at 1 and scales a whole sublayer's output, so decaying it toward 0 would mute the sublayer. These are vectors, with one dimension, so the rule in code is: no decay for any tensor with fewer than two dimensions.

## Wire the step

With `adamw_update` written, `train_step` is a short script over the same five boxes:

```python
logits = model_fn(params, batch["inputs"])
loss = causal_lm_loss(logits, batch["targets"])
loss.backward()
clip_global_norm([params[n].grad for n in params], config["max_grad_norm"])
step = opt_state["step"] + 1
lr = warmup_cosine_lr(step, warmup, total, peak, floor)
# then, inside torch.no_grad(), call adamw_update on every name
```

Clip all gradients together as one list, so there is one global norm for the whole model, not one per tensor. Store each result as `updated.detach().requires_grad_()`, a fresh leaf. Return the updated weights, an updated `opt_state` with the step and the two averages, and the loss as a float.

Adam changes what a clip does. Multiplying every gradient of a run by the same constant leaves $\hat{m}/\sqrt{\hat{v}}$ unchanged, and so every step. A clip therefore matters when it treats one batch differently from its neighbours by shrinking a spike, which is the case you built it for.

One more detail from the demo. Its token table `params["embed"]` reads tokens in, and its transpose scores tokens out. Using one matrix for both is called weight tying. It saves $V \times d$ weights, 38.4 million for a vocabulary of 50,000 and a width of 768, and it keeps the two uses of a token consistent. Both uses add into one gradient.

> [!KEY]
> One learning rate is fair to every weight when each weight's gradient is measured in its own units: Adam divides the averaged gradient by its typical size, so every weight with a steady slope moves about one learning rate per step. A training step is then five boxes in order: forward, loss, backward, clip, update. The clip comes before the update so a spike never reaches the averages, and the schedule is asked about the update about to be applied, counted from 1.

> [!RECAP]
> - $m$ averages the gradient and $v$ averages its square. The ratio $\hat{m}/\sqrt{\hat{v}}$ is near $\pm 1$, so every weight moves about one learning rate per step.
> - Divide by $1 - \beta^t$ to repair averages that start at zero. Count $t$ from 1.
> - Decay shrinks the weight by $\text{lr} \times \lambda$ outside the division. Biases and layer-norm gains get none.
> - A training step runs forward, loss, backward, clip (all gradients together), then update with the rate for step $t$.
> - Return fresh leaf tensors and a state holding the step and both averages.

## Build it step by step

The checks test two functions, in this order. The file already holds the functions from the earlier chapters.

1. **`adamw_update(theta, grad, m, v, step, lr, beta1, beta2, eps, weight_decay)`.** Return `(new_theta, new_m, new_v)` for one tensor. Update $m$ and $v$ from `grad`, correct them with `step`, then combine the shrink and the Adam step. It must work for numpy arrays and torch tensors. Raise `ValueError` when `step` is below 1. The usual mistake is to correct with `beta1 ** step` while counting from 0.
2. **`train_step(model_fn, params, batch, opt_state, config)`.** Run the five boxes. Read `peak_lr`, `floor_lr`, `warmup_steps`, `total_steps`, and `max_grad_norm` from `config`, with defaults `beta1` 0.9, `beta2` 0.95, `eps` 1e-8, and `weight_decay` 0. Use decay 0 for names in `config["no_decay"]`. Do the update inside `torch.no_grad()`. Return `(new_params, new_opt_state, loss)`, where `new_opt_state["step"]` is the old step plus one and `loss` is a float. The usual mistake is to clip each `.grad` on its own instead of as one list.

## Where this breaks

- Decaying a layer-norm gain weakens its sublayer. Keep gains and biases in `no_decay`.
- Counting the step from 0 breaks twice: the bias correction divides by zero, and the schedule hands the first update a rate of 0.
- Forgetting `torch.no_grad()` around the update records the update in the graph, which wastes memory and ties the next step to this one.
- Reusing the old tensors would add this step's gradient to the last one, because `backward()` adds into `.grad`. Fresh leaves avoid it.

## The exercise

You write `adamw_update` and `train_step`. The checks match the worked update of 0.89, the second step of 0.7811, the shrink alone when the gradient is zero, and the rule that every weight moves one learning rate whatever its gradient size. They check that step 0 is refused, that `train_step` matches an independent computation, that the stored averages hold the clipped gradient, that the returned weights are fresh leaves, and that twenty steps push the loss well below $\ln V$. Run trains a tiny transformer on a line of text and prints a sample.
