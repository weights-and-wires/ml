---
title: "The training step, and what a run costs"
slug: 053-training-loop-and-cost
part: "Part 6 — Transformers"
summary: Give every weight its own step with AdamW, wire one training step, then estimate a run's flops and memory and keep tiny gradients in 16 bits.
minutes: 75
gpu: true
objectives:
  - Explain why one learning rate suits weights with very different gradient sizes badly, and how dividing by a running size fixes it.
  - Compute one AdamW update by hand, including bias correction and decoupled weight decay.
  - Order the five parts of a training step, and say why the clip comes before the update.
  - Estimate a run's flops with the 6ND rule and its memory as four copies of the model.
  - Explain with a number line why fp16 loses small gradients while bf16 keeps them, and why loss scaling restores a gradient exactly.
prereqs:
  - 031-sgd-adam-and-schedules
  - 052-next-token-loss-and-safe-steps
  - 028-multilayer-perceptrons
exercise: 053-training-loop-and-cost
---

# The training step, and what a run costs

Two weights sit in the same model. The gradient for one is near 0.001, and the gradient for the other is near 100. Take a learning rate of 0.01 and plain gradient descent. The first weight moves 0.00001 per step, which is frozen. The second moves 1 per step, which is a leap. Neither rate is right for both. You cannot tune a rate per weight by hand, because a transformer has millions of weights and their gradient sizes change as training goes on. So how can one learning rate be fair to every weight, and what does it take to wire the loss, the clip, the schedule, and the optimiser into one training step?

Once you can take a step, a second question arrives before you start a run. You have 10 billion tokens of text, a model with 100 million weights, and one GPU. How long will the run take? Will the model fit in the GPU's memory? And why do some people who train in 16-bit numbers for speed find that their gradients vanished and no error was raised?

The first half of this chapter builds the update for one tensor and the step that calls everything you have built so far. The second half counts that step: its flops, its memory, and what a number format does to a small gradient. Doing that arithmetic first saves you from launching a run that would take a month or crash on the first step.

## What you already know

- Adam keeps two running averages for each weight, $m$ for the gradient and $v$ for its square, and AdamW shrinks each weight by a fixed fraction on top (see [Adam and AdamW](/lessons/031-sgd-adam-and-schedules)).
- The loss is one number that `backward()` turns into a gradient for every weight, and the schedule and the clip keep the step safe (see [next-token loss and safe steps](/lessons/052-next-token-loss-and-safe-steps)).
- A flop is one floating-point operation, and a matrix of $a \times b$ weights multiplies a vector of $b$ numbers in $a \times b$ multiplications and about as many additions (see [width, depth, and cost](/lessons/028-multilayer-perceptrons)).
- A gradient is a slope, and slopes near a good solution are often tiny.

## The step in five boxes

Figure 1 is the first half of the chapter in one picture. A training step takes the weights and a batch, and returns updated weights.

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="One training step as five boxes in a row: forward, loss, backward, clip, update, with the learning rate and the optimiser state feeding the update">
<defs><marker id="c48-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="18" y="60" width="104" height="66" rx="8" class="fill-sunk stroke-muted"/>
<text x="70" y="86" text-anchor="middle" class="fig-bold">1  forward</text>
<text x="70" y="108" text-anchor="middle" class="fig-small">model_fn</text>
<text x="70" y="146" text-anchor="middle" class="fig-small">logits (B, T, V)</text>
<line x1="124" y1="93" x2="140" y2="93" class="stroke-ink" marker-end="url(#c48-arrow)"/>
<rect x="142" y="60" width="104" height="66" rx="8" class="fill-soft-c stroke-c"/>
<text x="194" y="86" text-anchor="middle" class="fig-bold">2  loss</text>
<text x="194" y="108" text-anchor="middle" class="fig-small">causal_lm_loss</text>
<text x="194" y="146" text-anchor="middle" class="fig-small">one number</text>
<line x1="248" y1="93" x2="264" y2="93" class="stroke-ink" marker-end="url(#c48-arrow)"/>
<rect x="266" y="60" width="104" height="66" rx="8" class="fill-sunk stroke-muted"/>
<text x="318" y="86" text-anchor="middle" class="fig-bold">3  backward</text>
<text x="318" y="108" text-anchor="middle" class="fig-small">loss.backward()</text>
<text x="318" y="146" text-anchor="middle" class="fig-small">a gradient per weight</text>
<line x1="372" y1="93" x2="388" y2="93" class="stroke-ink" marker-end="url(#c48-arrow)"/>
<rect x="390" y="60" width="104" height="66" rx="8" class="fill-soft-c stroke-c"/>
<text x="442" y="86" text-anchor="middle" class="fig-bold">4  clip</text>
<text x="442" y="108" text-anchor="middle" class="fig-small">clip_global_norm</text>
<text x="442" y="146" text-anchor="middle" class="fig-small">norm at most the limit</text>
<line x1="496" y1="93" x2="512" y2="93" class="stroke-ink" marker-end="url(#c48-arrow)"/>
<rect x="514" y="60" width="104" height="66" rx="8" class="fill-soft-b stroke-b"/>
<text x="566" y="86" text-anchor="middle" class="fig-bold">5  update</text>
<text x="566" y="108" text-anchor="middle" class="fig-small">adamw_update</text>
<text x="566" y="146" text-anchor="middle" class="fig-small">updated weights</text>
<text x="70" y="44" text-anchor="middle" class="fig-small">params and a batch</text>
<line x1="70" y1="50" x2="70" y2="58" class="stroke-ink" marker-end="url(#c48-arrow)"/>
<rect x="364" y="184" width="150" height="40" rx="8" class="fill-soft-c stroke-c"/>
<text x="439" y="202" text-anchor="middle" class="fig-small">learning rate for this step</text>
<text x="439" y="217" text-anchor="middle" class="fig-small">warmup_cosine_lr</text>
<line x1="514" y1="204" x2="552" y2="164" class="stroke-ink" marker-end="url(#c48-arrow)"/>
<rect x="508" y="236" width="124" height="40" rx="8" class="fill-soft-a stroke-a"/>
<text x="570" y="254" text-anchor="middle" class="fig-small">opt_state</text>
<text x="570" y="269" text-anchor="middle" class="fig-small">step, m, v</text>
<line x1="570" y1="234" x2="570" y2="166" class="stroke-ink" marker-end="url(#c48-arrow)"/>
<rect x="18" y="246" width="16" height="12" rx="2" class="fill-soft-c stroke-c"/>
<text x="42" y="256" text-anchor="start" class="fig-small">written in earlier chapters</text>
<rect x="18" y="266" width="16" height="12" rx="2" class="fill-soft-b stroke-b"/>
<text x="42" y="276" text-anchor="start" class="fig-small">you write it in this chapter</text>
<rect x="18" y="286" width="16" height="12" rx="2" class="fill-sunk stroke-muted"/>
<text x="42" y="296" text-anchor="start" class="fig-small">the model and autograd</text>
</svg>
Figure 1. One training step is five boxes in a row. The weights and a batch go in, and updated weights come out. The green boxes are functions you wrote earlier, and the orange one is the function you write in this chapter. The clip sits before the update so the optimiser never sees an oversized gradient.
```

The boxes run in an order that has reasons. The forward pass turns inputs into logits, and the loss turns logits and targets into one number. The backward pass fills a gradient for every weight. Only then can you clip, because clipping needs the whole gradient to measure its length. The clip comes before the update because Adam stores averages of whatever gradient it is given. An oversized gradient would sit in $m$ and $v$ for many steps, and clipped first, it enters at a safe size.

The learning rate comes from the schedule. The loop counts updates from 1, and the schedule is asked about the update about to be applied, so the first update uses a rate of peak divided by warmup rather than 0. Last, the update returns fresh weight tensors, so autograd does not record the update in the next step's graph and you never have to zero old gradients.

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

Plain descent makes the step length proportional to the gradient, so a gradient 100,000 times larger gives a step 100,000 times longer. Adam measures each weight in its own units. The average $m$ says which way and how hard the gradient pushes, and $\sqrt{v}$ is the typical size of the gradient, a root-mean-square. Their ratio is near $+1$ when the gradient keeps pushing the same way, near $-1$ when it keeps pushing the other way, and near 0 when it flips sign at random. Every weight with a steady slope therefore moves about one learning rate per step, whatever its gradient size.

Why not divide each gradient by its own absolute value? A weight whose gradient flips between $+3$ and $-3$ would then jump a full step back and forth, although it has no direction. The average in the numerator cancels that noise.

Take weight $\theta = 1$ and gradient $g = 0.5$ at the first update, with learning rate 0.1, $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay $\lambda = 0.1$, and both averages at 0. The averages become $m = 0.1 \times 0.5 = 0.05$ and $v = 0.05 \times 0.25 = 0.0125$. Those are too small: an average that starts at zero has seen one number and a window full of zeros. With a steady gradient of 0.5, the raw $m$ is exactly $0.5 \times (1 - 0.9^t)$, which is 0.05 at $t = 1$, 0.095 at $t = 2$, and 0.33 at $t = 10$.

Dividing by $1 - \beta^t$ repairs it. The factor is 10 at $t = 1$, 5.3 at $t = 2$, 1.5 at $t = 10$, and 1.005 at $t = 50$, so it fades as the average fills. Here $\hat{m} = 0.05/0.1 = 0.5$ and $\hat{v} = 0.0125/0.05 = 0.25$, so $\sqrt{\hat{v}} = 0.5$ and the ratio is exactly 1: a full learning rate in the direction of the gradient. At $t = 0$ the divisor is zero, which is why counting starts at 1.

The decay is separate. Each weight loses the fraction $\text{lr} \times \lambda = 0.1 \times 0.1 = 0.01$ of itself, and then the Adam step is subtracted:

$$
\theta \leftarrow \theta\,(1 - \text{lr}\cdot\lambda) - \text{lr}\,\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

Read it aloud: shrink the weight by a small fraction, then move it one learning rate against the normalised gradient. The small number $\epsilon$, $10^{-8}$, keeps the denominator away from zero for a weight that has had no gradient yet. For the example, $1 \times 0.99 - 0.1 \times 1 = 0.89$. With $\beta_2 = 0.95$, $v$ remembers about $1/(1 - 0.95) = 20$ steps, shorter than the 1,000 of the default, so it forgets a spike sooner. In code, for one weight tensor:

```python
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad**2
m_hat = m / (1 - beta1**step)
v_hat = v / (1 - beta2**step)
theta = theta * (1 - lr * weight_decay) - lr * m_hat / (v_hat**0.5 + eps)
```

Weight decay has one rule of thumb. Matrices get it, and biases and layer-norm gains do not. A gain starts at 1 and scales a whole sublayer's output, so decaying it toward 0 would mute the sublayer. These are vectors, so the rule in code is: no decay for any tensor with fewer than two dimensions.

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

> [!KEY]
> One learning rate is fair to every weight when each weight's gradient is measured in its own units: Adam divides the averaged gradient by its typical size, so every weight with a steady slope moves about one learning rate per step. A training step is then five boxes in order, forward, loss, backward, clip, update. The clip comes before the update so a spike never reaches the averages, and the schedule is asked about the update about to be applied, counted from 1.

## Count the work

Count what one step costs. Start with one token and one weight. The weight multiplies a number that arrives, and the product is added into a running total: one multiplication and one addition, two flops. A model with $N$ weights does this for every weight, so the forward pass costs about $2N$ flops per token.

The backward pass has twice the work, for a reason you can see on a single weight. Every weight sits between an input $a$ and an output, and the loss sends back a gradient $dy$ for that output. Two things are needed. The gradient for the input $a$ is $w \cdot dy$, and it has to be passed further back. The gradient for the weight is $a \cdot dy$, and it is used to update the weight. Each is one multiplication and one addition per weight, so each costs about $2N$, and backward costs $4N$ in all. Figure 3 draws the three jobs.

```figure
<svg viewBox="0 0 620 280" role="img" aria-label="Bars showing forward cost 2N, backward cost 4N split into two halves, and the total 6N flops per token">
<text x="118" y="86" text-anchor="end" class="fig-bold">forward</text>
<rect x="130" y="60" width="160" height="40" class="fill-soft-a stroke-a"/>
<text x="210" y="85" text-anchor="middle" class="fig-bold">2N</text>
<text x="118" y="146" text-anchor="end" class="fig-bold">backward</text>
<rect x="130" y="120" width="160" height="40" class="fill-soft-b stroke-b"/>
<text x="210" y="137" text-anchor="middle" class="fig-bold">2N</text>
<text x="210" y="153" text-anchor="middle" class="fig-small">gradient for the inputs</text>
<rect x="290" y="120" width="160" height="40" class="fill-soft-c stroke-c"/>
<text x="370" y="137" text-anchor="middle" class="fig-bold">2N</text>
<text x="370" y="153" text-anchor="middle" class="fig-small">gradient for the weights</text>
<text x="118" y="216" text-anchor="end" class="fig-bold">one token</text>
<rect x="130" y="190" width="160" height="40" class="fill-soft-a stroke-a"/>
<text x="210" y="215" text-anchor="middle" class="fig-bold">2N</text>
<rect x="290" y="190" width="160" height="40" class="fill-soft-b stroke-b"/>
<text x="370" y="215" text-anchor="middle" class="fig-bold">2N</text>
<rect x="450" y="190" width="160" height="40" class="fill-soft-c stroke-c"/>
<text x="530" y="215" text-anchor="middle" class="fig-bold">2N</text>
<text x="370" y="254" text-anchor="middle" class="fig-small">6N flops in all: three jobs the size of one forward pass</text>
<text x="370" y="30" text-anchor="middle" class="fig-small">flops for one token, N weights (drawn for N = 1 million, so 2N = 2 million)</text>
</svg>
Figure 3. The flops one token costs for a model with N weights. The forward pass costs 2N. The backward pass has to produce a gradient for the inputs of every layer, to keep sending back, and a gradient for the weights, to update them, and each costs about as much as the forward pass. The total is 6N, three forward-sized jobs.
```

One token costs $2N + 4N = 6N$ flops, so a run over $D$ tokens costs:

$$
C \approx 6\,N\,D
$$

Read it aloud: six flops, times the number of weights, times the number of tokens. As a check, take a toy model with a million weights and ten thousand tokens: $6 \times 10^6 \times 10^4 = 6 \times 10^{10}$ flops. In code it is one line, `6 * params * tokens`.

Return to the opening question. A model with $10^8$ weights on $10^{10}$ tokens costs $6 \times 10^{8} \times 10^{10} = 6 \times 10^{18}$ flops. Suppose your GPU sustains $10^{14}$ flops per second, a round number for a datacenter GPU doing 16-bit matrix work at a reasonable share of its peak. Check the figure for yours. The run takes $6 \times 10^{18} / 10^{14} = 6 \times 10^{4}$ seconds, about 16.7 hours. A model ten times larger, 1 billion weights, trained on 20 billion tokens, costs $6 \times 10^9 \times 2 \times 10^{10} = 1.2 \times 10^{20}$ flops, which is $1.2 \times 10^{6}$ seconds, about 14 days on the same GPU.

The rule is an estimate. It ignores the attention score table and the softmax, which add a little for long sequences, and it counts every weight although an embedding lookup does no multiplication. The clip and the update touch each weight a few times per step, not per token, so they are a rounding error next to $6N$ per token.

## Count the memory

Time is half the budget. The other half is memory, and the weights are only one of the things you must keep. During training you hold four copies of the model's shape:

| Copy | Why you need it |
|---|---|
| Weights | The model itself. |
| Gradients | One per weight, filled by the backward pass. |
| $m$ | Adam's average of the gradient. |
| $v$ | Adam's average of the squared gradient. |

That is $4N$ numbers. If every number takes 4 bytes, as in fp32, the memory is $16N$ bytes. For $10^8$ weights that is 1.6 GB, which fits. For $10^9$ weights it is 16 GB, and the GPU still needs room for the activations, the intermediate values the backward pass reads, which grow with the batch size and the sequence length. Plain gradient descent needs only two copies, weights and gradients.

Halving every number to 2 bytes does not halve the bill in practice, because common recipes keep an fp32 copy of the weights as well. The next sections explain why 16 bits is worth the trouble, and what it risks.

> [!KEY]
> Before a run, estimate it. Compute is about $6ND$ flops, a forward pass plus a backward pass that costs two forward-sized jobs. Memory is four copies of the model, weights, gradients, $m$, and $v$, which is $16N$ bytes in fp32, plus activations. For 100 million weights on 10 billion tokens, that is about 17 hours and 1.6 GB.

## Look inside a number

A floating-point number is scientific notation in base 2: a sign, a size, and some digits, as in $1.5 \times 2^{k}$. The **exponent** is the $k$. Its bits decide how large and how small a number can be, which is the range. The **mantissa** is the $1.5$. Its bits decide how many digits you keep, which is the precision. Three formats matter here:

| Format | Exponent bits | Mantissa bits | Largest | Smallest normal | Gap between neighbours |
|---|---|---|---|---|---|
| fp32 | 8 | 23 | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ | about $10^{-7}$ of the value |
| fp16 | 5 | 10 | 65,504 | $6.1 \times 10^{-5}$ | about $10^{-3}$ of the value |
| bf16 | 8 | 7 | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ | about $8 \times 10^{-3}$ of the value |

The last column is the coarseness of the grid of numbers the format can hold. In fp16, the next number above 1000 is about 1 higher, so a gradient is kept to only about three digits.

Below $6.1 \times 10^{-5}$, fp16 keeps going with a thin strip of subnormal numbers, which have lower precision, down to $6 \times 10^{-8}$. So a gradient of $10^{-6}$ is stored as $1.013 \times 10^{-6}$, about 1.3 percent off. A gradient of $10^{-7}$ is stored as $1.19 \times 10^{-7}$, 19 percent off. A gradient of $10^{-8}$, below half of the smallest subnormal, rounds to exactly 0. Precision collapses first, and then the number disappears with no error. bf16 trades the other way: its 8 exponent bits give it fp32's range, so $10^{-8}$ survives, but with only 7 mantissa bits it keeps about two to three digits.

## Rescue small gradients by scaling the loss

Why not stay in fp32? Because 16-bit numbers halve the memory traffic and run faster on GPUs built for them. So fp16 users look for a way to keep the gradients in range. The trick is to multiply the loss by a number $s$ before the backward pass, as Figure 4 shows.

```figure
<svg viewBox="0 0 620 275" role="img" aria-label="A gradient of 1e-8 lies left of the fp16 range, multiplying by 1024 moves it inside the range, and dividing by 1024 afterwards restores it">
<defs><marker id="c48-scale-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="50" y1="110" x2="50" y2="210" class="stroke-grid"/>
<line x1="50" y1="210" x2="50" y2="216" class="stroke-muted"/>
<text x="50" y="232" text-anchor="middle" class="fig-small">1e-10</text>
<line x1="116.2" y1="110" x2="116.2" y2="210" class="stroke-grid"/>
<line x1="116.2" y1="210" x2="116.2" y2="216" class="stroke-muted"/>
<text x="116.2" y="232" text-anchor="middle" class="fig-small">1e-8</text>
<line x1="182.5" y1="110" x2="182.5" y2="210" class="stroke-grid"/>
<line x1="182.5" y1="210" x2="182.5" y2="216" class="stroke-muted"/>
<text x="182.5" y="232" text-anchor="middle" class="fig-small">1e-6</text>
<line x1="248.8" y1="110" x2="248.8" y2="210" class="stroke-grid"/>
<line x1="248.8" y1="210" x2="248.8" y2="216" class="stroke-muted"/>
<text x="248.8" y="232" text-anchor="middle" class="fig-small">1e-4</text>
<line x1="315" y1="110" x2="315" y2="210" class="stroke-grid"/>
<line x1="315" y1="210" x2="315" y2="216" class="stroke-muted"/>
<text x="315" y="232" text-anchor="middle" class="fig-small">1e-2</text>
<line x1="381.2" y1="110" x2="381.2" y2="210" class="stroke-grid"/>
<line x1="381.2" y1="210" x2="381.2" y2="216" class="stroke-muted"/>
<text x="381.2" y="232" text-anchor="middle" class="fig-small">1</text>
<line x1="447.5" y1="110" x2="447.5" y2="210" class="stroke-grid"/>
<line x1="447.5" y1="210" x2="447.5" y2="216" class="stroke-muted"/>
<text x="447.5" y="232" text-anchor="middle" class="fig-small">1e2</text>
<line x1="513.8" y1="110" x2="513.8" y2="210" class="stroke-grid"/>
<line x1="513.8" y1="210" x2="513.8" y2="216" class="stroke-muted"/>
<text x="513.8" y="232" text-anchor="middle" class="fig-small">1e4</text>
<line x1="580" y1="110" x2="580" y2="210" class="stroke-grid"/>
<line x1="580" y1="210" x2="580" y2="216" class="stroke-muted"/>
<text x="580" y="232" text-anchor="middle" class="fig-small">1e6</text>
<line x1="50" y1="210" x2="580" y2="210" class="stroke-muted"/>
<rect x="241.6" y="150" width="299.1" height="40" class="fill-soft-b stroke-b"/>
<rect x="141.9" y="160" width="99.7" height="20" class="fill-sunk stroke-muted"/>
<text x="381.2" y="175" text-anchor="middle" class="fig-bold">fp16 can hold these</text>
<circle cx="116.2" cy="170" r="6" class="fill-bad"/>
<text x="138.2" y="124" text-anchor="end" class="fig-small">gradient 1e-8</text>
<text x="138.2" y="138" text-anchor="end" class="fig-small">stored as 0</text>
<circle cx="216" cy="170" r="6" class="fill-a"/>
<text x="194" y="124" text-anchor="start" class="fig-small">× 1024 = 1.02e-5</text>
<text x="194" y="138" text-anchor="start" class="fig-small">stored as 1.0252e-5</text>
<path d="M124.2,100 L208,100" fill="none" class="stroke-ink" stroke-width="1.5" marker-end="url(#c48-scale-arrow)"/>
<text x="206.1" y="92" text-anchor="middle" class="fig-small">multiply the loss by 1024</text>

<text x="315" y="262" text-anchor="middle" class="fig-small">Then divide the gradient by 1024 in fp32: 1.0012e-8, within 0.2 percent of 1e-8.</text>
</svg>
Figure 4. Loss scaling moves a gradient across the axis without changing its meaning. In fp16, 1e-8 rounds to zero. Multiplying the loss by 1024 multiplies every gradient by 1024, which lifts 1e-8 to about 1e-5, where fp16 can store it. Dividing by 1024 in fp32 afterwards gives back 1.0012e-8.
```

This is harmless because of the chain rule. The slope of $s \times L$ with respect to a weight $w$ is $s$ times the slope of $L$, so every gradient in the model is multiplied by exactly $s$ and nothing else changes:

$$
\frac{\partial (s\,L)}{\partial w} = s\,\frac{\partial L}{\partial w}
$$

Read it aloud: scale the loss by $s$ and every gradient scales by $s$. Take $s = 1024$. A gradient of $10^{-8}$ becomes $1.024 \times 10^{-5}$, which fp16 stores as $1.0252 \times 10^{-5}$, a rounding error of 0.1 percent. After the backward pass, convert the gradients to fp32 and divide by 1024 to get $1.0012 \times 10^{-8}$, within 0.2 percent of the truth. Without scaling, it was 0. The simulation in code is two lines:

```python
scaled = fp16_roundtrip(true_grad * scale)
recovered = scaled / scale
```

The division has to happen in fp32. If you divided in fp16, you would push the value back out of range.

A scale that is too large fails the other way. A gradient of 1.0 at a scale of $10^5$ becomes $10^5$, which exceeds fp16's largest number, 65,504, and turns into infinity. Training libraries watch for infinities, skip that step, and lower the scale, which is called dynamic loss scaling. bf16 needs none of this, because its range already holds the small gradients.

> [!WARNING]
> fp16 gives no error when a small gradient rounds to zero. The loss curve stops improving. Scale the loss, or use bf16, before you trust a 16-bit run.

> [!KEY]
> The number format decides whether tiny gradients survive the backward pass of the step you built. fp16 has a small range and loses them unless you scale the loss, because scaling multiplies every gradient by the same $s$ and you divide it out in fp32. bf16 has fp32's range and needs no scaling.

> [!RECAP]
> - $m$ averages the gradient and $v$ averages its square. The ratio $\hat{m}/\sqrt{\hat{v}}$ is near $\pm 1$, so every weight moves about one learning rate per step. Divide by $1 - \beta^t$ and count $t$ from 1.
> - Decay shrinks the weight by $\text{lr} \times \lambda$ outside the division. Biases and layer-norm gains get none.
> - A training step runs forward, loss, backward, clip (all gradients together), then update with the rate for step $t$. It returns fresh leaf tensors and a state holding the step and both averages.
> - Forward costs $2N$ flops per token, backward costs $4N$, and a run costs $6ND$. Training holds four copies of the model, $16N$ bytes in fp32.
> - fp16 loses gradients below about $10^{-8}$, so multiply the loss by $s$ and divide the gradients by $s$ in fp32.

## Build it step by step

The checks test five functions, in this order. The file already holds the functions from the earlier chapters and `fp16_roundtrip`, which casts a number to fp16 and back so you can see what survives.

1. **`adamw_update(theta, grad, m, v, step, lr, beta1, beta2, eps, weight_decay)`.** Return `(new_theta, new_m, new_v)` for one tensor. Update $m$ and $v$ from `grad`, correct them with `step`, then combine the shrink and the Adam step. It must work for numpy arrays and torch tensors. Raise `ValueError` when `step` is below 1. The usual mistake is to correct with `beta1 ** step` while counting from 0.
2. **`train_step(model_fn, params, batch, opt_state, config)`.** Run the five boxes. Read `peak_lr`, `floor_lr`, `warmup_steps`, `total_steps`, and `max_grad_norm` from `config`, with defaults `beta1` 0.9, `beta2` 0.95, `eps` 1e-8, and `weight_decay` 0. Use decay 0 for names in `config["no_decay"]`. Do the update inside `torch.no_grad()`. Return `(new_params, new_opt_state, loss)`, where `new_opt_state["step"]` is the old step plus one and `loss` is a float. The usual mistake is to clip each `.grad` on its own instead of as one list.
3. **`training_flops(params, tokens)`.** Return `6 * params * tokens`. The usual mistake is to count only the forward pass, which reports $2ND$.
4. **`training_memory_bytes(params, bytes_per_number=4, copies=4)`.** Return the bytes for `copies` numbers per weight. The defaults describe fp32 weights, gradients, $m$, and $v$, so a million weights need 16 million bytes.
5. **`scaled_grad_survives(true_grad, scale)`.** Multiply `true_grad` by `scale`, pass the result through `fp16_roundtrip`, divide by `scale`, and return the recovered gradient. It should work on a number or an array. A scale of 1 loses $10^{-8}$, and a scale that is too large returns infinity.

## Where this breaks

- Decaying a layer-norm gain weakens its sublayer. Keep gains and biases in `no_decay`.
- Counting the step from 0 breaks twice: the bias correction divides by zero, and the schedule hands the first update a rate of 0.
- Forgetting `torch.no_grad()` around the update records the update in the graph, which wastes memory and ties the next step to this one. Reusing the old tensors would also add this step's gradient to the last one, because `backward()` adds into `.grad`. Fresh leaves avoid both.
- A loss scale that is too large overflows to infinity. The step must be skipped, not applied.
- The $6ND$ rule and the $16N$-byte rule are estimates. They leave out attention's extra flops, the activations, and any extra fp32 copies.

## The exercise

You write `adamw_update`, `train_step`, `training_flops`, `training_memory_bytes`, and `scaled_grad_survives`. The checks match the worked update of 0.89 and the second step of 0.7811, the shrink alone when the gradient is zero, and the rule that every weight moves one learning rate whatever its gradient size. They check that step 0 is refused, that `train_step` matches an independent computation with clipped averages and fresh leaves, and that twenty steps push the loss well below $\ln V$. They then compare the flops with $6ND$ and the memory with 16 bytes per weight, and check that a gradient of $10^{-8}$ vanishes at scale 1, returns within 1 percent at scale 1024, and overflows to infinity when the scale is too large. Run trains a tiny transformer on a line of text, prints a sample, and prints the opening estimate with a loss-scaling table.
