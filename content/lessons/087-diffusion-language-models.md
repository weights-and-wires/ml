---
title: "Diffusion language models: mask, then unmask"
slug: 087-diffusion-language-models
part: "Part 8 — Generative models"
summary: "Hide tokens at a random rate, train a masked predictor with a 1/t weight, and sample by unmasking the most confident slots first."
minutes: 70
gpu: false
objectives:
  - Draw the forward process of a masked diffusion model, and say how it differs from Gaussian noise and from masked-token training at a fixed rate.
  - Compute the masked-diffusion loss by hand, with its 1/t weight and its normalisation by sequence length, and check its gradient against finite differences.
  - Run the unmasking schedule and a low-confidence sampler on a table of logits, and say why it beats random remasking in principle.
prereqs:
  - 065-encoders-and-cross-attention
  - 075-decoding-and-sampling
  - 086-diffusion-models
exercise: 087-diffusion-language-models
---

# Diffusion language models: mask, then unmask

You write a sentence one word at a time, left to right, and each word is a commitment: the fifth word can't change once you start the sixth. A language model from the chapter on [decoding and sampling](/lessons/075-decoding-and-sampling) works the same way. It also has a cost you can't avoid. A 500-token answer takes 500 passes through the network, one per token, in a fixed order.

Suppose you wrote the sentence the way you edit it instead. You start with a page of blanks, fill in the words you are sure about first, and let those words guide the blanks that are left. The [diffusion chapter](/lessons/086-diffusion-models) did this for points on a plane: noise them, learn to undo the noise, and walk backwards. Text is a different material. Words are discrete, and adding half a unit of Gaussian noise to a word id means nothing. This chapter swaps the noise for something that fits text: you hide tokens behind a mask symbol.

The model in this chapter is a masked diffusion model, and the reference is LLaDA ([Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)). You build three pieces: the forward process that hides tokens, the loss that trains a model to reveal them, and the sampler that starts from a fully hidden page and fills it in.

## What you already know

- A softmax turns scores into probabilities, and cross-entropy is the negative log of the probability of the right answer.
- A masked language model, as in BERT, hides some tokens and predicts them from both sides, as in the chapter on [encoders and cross-attention](/lessons/065-encoders-and-cross-attention).
- A causal language model predicts the next token and samples one token per pass.
- A diffusion model corrupts data along a schedule indexed by a time $t$, trains a network to undo one step, and samples by walking from noise back to data.
- An expectation is an average over the random draws, and the average of an indicator is the probability of its event.

## The forward process: hide each token with probability t

Draw one number $t$ between 0 and 1. For every token in the sequence, flip a coin that lands "hide" with probability $t$, independently of the other tokens, and replace a hidden token with a special mask token written `[MASK]`. At $t = 0$ nothing is hidden. At $t = 1$ every token is hidden. LLaDA draws $t$ uniformly from 0 to 1, once per training sequence (its Algorithm 1), so one sequence is hidden at one rate and the next at another.

Figure 1 shows the process on an eight-word sentence. It uses one uniform draw per position and hides a position whenever its draw is below $t$. That detail matters for the picture: a position hidden at $t = 0.25$ is hidden at every larger $t$ as well. The mask is *absorbing*. Once a token has turned into `[MASK]`, no later step brings it back or turns it into a different word.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="One sentence of eight words masked at five levels of t; masked positions stay masked as t grows">
  <text x="10" y="42" class="fig-small">t = 0</text>
  <rect x="90" y="20" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="119.0" y="41.0" text-anchor="middle" class="fig-small">the</text>
  <rect x="152" y="20" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="181.0" y="41.0" text-anchor="middle" class="fig-small">cat</text>
  <rect x="214" y="20" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="243.0" y="41.0" text-anchor="middle" class="fig-small">sat</text>
  <rect x="276" y="20" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="305.0" y="41.0" text-anchor="middle" class="fig-small">on</text>
  <rect x="338" y="20" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="367.0" y="41.0" text-anchor="middle" class="fig-small">the</text>
  <rect x="400" y="20" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="429.0" y="41.0" text-anchor="middle" class="fig-small">mat</text>
  <rect x="462" y="20" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="491.0" y="41.0" text-anchor="middle" class="fig-small">all</text>
  <rect x="524" y="20" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="553.0" y="41.0" text-anchor="middle" class="fig-small">day</text>
  <text x="10" y="86" class="fig-small">t = 0.25</text>
  <rect x="90" y="64" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="119.0" y="85.0" text-anchor="middle" class="fig-small">the</text>
  <rect x="152" y="64" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="181.0" y="85.0" text-anchor="middle" class="fig-small">cat</text>
  <rect x="214" y="64" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="243.0" y="85.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="276" y="64" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="305.0" y="85.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="338" y="64" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="367.0" y="85.0" text-anchor="middle" class="fig-small">the</text>
  <rect x="400" y="64" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="429.0" y="85.0" text-anchor="middle" class="fig-small">mat</text>
  <rect x="462" y="64" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="491.0" y="85.0" text-anchor="middle" class="fig-small">all</text>
  <rect x="524" y="64" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="553.0" y="85.0" text-anchor="middle" class="fig-small">day</text>
  <text x="10" y="130" class="fig-small">t = 0.5</text>
  <rect x="90" y="108" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="119.0" y="129.0" text-anchor="middle" class="fig-small">the</text>
  <rect x="152" y="108" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="181.0" y="129.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="214" y="108" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="243.0" y="129.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="276" y="108" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="305.0" y="129.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="338" y="108" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="367.0" y="129.0" text-anchor="middle" class="fig-small">the</text>
  <rect x="400" y="108" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="429.0" y="129.0" text-anchor="middle" class="fig-small">mat</text>
  <rect x="462" y="108" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="491.0" y="129.0" text-anchor="middle" class="fig-small">all</text>
  <rect x="524" y="108" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="553.0" y="129.0" text-anchor="middle" class="fig-small">day</text>
  <text x="10" y="174" class="fig-small">t = 0.75</text>
  <rect x="90" y="152" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="119.0" y="173.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="152" y="152" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="181.0" y="173.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="214" y="152" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="243.0" y="173.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="276" y="152" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="305.0" y="173.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="338" y="152" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="367.0" y="173.0" text-anchor="middle" class="fig-small">the</text>
  <rect x="400" y="152" width="58" height="34" class="fill-soft-a stroke-grid"/>
  <text x="429.0" y="173.0" text-anchor="middle" class="fig-small">mat</text>
  <rect x="462" y="152" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="491.0" y="173.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="524" y="152" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="553.0" y="173.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <text x="10" y="218" class="fig-small">t = 1</text>
  <rect x="90" y="196" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="119.0" y="217.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="152" y="196" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="181.0" y="217.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="214" y="196" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="243.0" y="217.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="276" y="196" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="305.0" y="217.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="338" y="196" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="367.0" y="217.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="400" y="196" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="429.0" y="217.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="462" y="196" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="491.0" y="217.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="524" y="196" width="58" height="34" class="fill-sunk stroke-grid"/>
  <text x="553.0" y="217.0" text-anchor="middle" class="fill-muted fig-small">M</text>
</svg>
Figure 1. One sentence hidden at five levels of t, from seeded draws. Notice that the grey slots only grow: each row hides every slot that the row above it hides. The t = 0.5 row hides 3 of 8 slots, not 4, because each slot is its own coin flip.
```

### How this differs from Gaussian diffusion

The chapter on [diffusion models](/lessons/086-diffusion-models) adds Gaussian noise to a point. The point blurs gradually: at every time the point still exists, only less clearly, and the network predicts the noise that was added. Here the corruption is all or nothing for each token. A token is intact or it is gone, and a token that is gone leaves no trace of which word it was. The network therefore predicts the clean token itself, as a probability over the vocabulary, rather than a noise vector.

Both processes end at a state that holds no information about the data: a standard normal cloud for Gaussian diffusion and a page of `[MASK]` for masked diffusion. Both walk back from that state in small steps.

### How this differs from BERT

BERT-style training also hides tokens and predicts them. Two things differ. BERT hides a fixed share, 15 percent, so the network practises only on mostly complete sentences. It never sees a page that is 90 percent hidden, so you can't start from a blank page and sample from it. Masked diffusion draws the rate from 0 to 1, so the network practises at every level, including the fully blank page. The second difference is the loss weight, which the next section derives. The fixed 15 percent has no weight and no link to the likelihood of the data.

> [!KEY]
> The forward process hides each token independently with probability $t$, with $t$ drawn uniformly from 0 to 1. The mask is absorbing, so the hidden set only grows with $t$. Because $t$ reaches 1, the network learns to predict from nothing, and that is what makes sampling from a blank page possible.

## The loss: a masked language model with a 1/t weight

Take a clean sequence $x_0$ of $L$ tokens, draw $t$, and hide tokens to get $x_t$. The network $p_\theta$ reads the whole partly hidden sequence $x_t$ and returns, for every position, a probability over the vocabulary. It reads both sides of every slot, so it has no causal mask. The training signal is the cross-entropy of the clean token at each hidden position, and nothing at the positions you left visible: the model can copy those, so scoring them teaches nothing.

LLaDA's Equation 3 adds one factor. Read it first in words: sum the cross-entropy over the hidden positions, divide by $t$, and average over the random draws.

$$
\mathcal{L}(\theta) = -\,\mathbb{E}_{t,\,x_0,\,x_t}\left[\frac{1}{t}\sum_{i=1}^{L}\mathbf{1}[x_t^i = \texttt{M}]\,\log p_\theta(x_0^i \mid x_t)\right]
$$

Here $\mathbf{1}[x_t^i = \texttt{M}]$ is 1 when position $i$ is hidden and 0 otherwise, and $\texttt{M}$ stands for the mask token. The sum runs over all $L$ positions, and the indicator keeps the hidden ones. Equation 3 has no division by $L$. LLaDA's Algorithm 1 divides by $t \cdot L$, which turns the sum into a per-token average, and the exercise does the same. This rescales each sequence's loss by a constant, so it doesn't change which model minimises the loss on a fixed length.

### Why divide by t

At rate $t$ about $tL$ positions are hidden, so an unweighted sum has about $tL$ terms. A run at $t = 0.1$ would carry a tenth of the signal of a run at $t = 1$. The weight repairs that. Each position is hidden with probability $t$, and the average of its indicator is $t$, so the factor $1/t$ cancels it. In expectation over the mask, every position contributes its loss once, whatever the rate. All mask levels then count equally, instead of the heavily masked levels dominating.

### Why this bounds the likelihood

The loss is more than a heuristic. LLaDA's Equation 4 states that it is an upper bound on the negative log-likelihood of the data:

$$
-\,\mathbb{E}_{x_0}\left[\log p_\theta(x_0)\right] \;\le\; \mathcal{L}(\theta)
$$

Here $p_\theta(x_0)$ is the probability that the model, run as a generator, assigns to the clean sequence. A lower loss then pushes the negative log-likelihood down, the same quantity a causal language model minimises, so the two kinds of model can be compared on that scale. The paper credits the proof to earlier work on masked diffusion by Shi and co-authors, Sahoo and co-authors, and Ou and co-authors, and restates the bound as Equation 13 in its appendix. This chapter does not reprove it. It is a bound rather than an equality, so the loss value is not the exact negative log-likelihood. It never falls below it, and that is the guarantee a training loss needs.

### A loss worked by hand

Take $L = 4$ clean tokens $x_0 = (2, 0, 1, 0)$ over a vocabulary of 3, and $t = 0.5$. Positions 2 and 4 (1-indexed) are hidden. The network returns these probabilities for the clean token at each position:

| Position | Hidden? | Clean token | Probability of the clean token |
|---|---|---|---|
| 1 | no | 2 | 0.5 |
| 2 | yes | 0 | 0.5 |
| 3 | no | 1 | 0.8 |
| 4 | yes | 0 | 0.25 |

Only rows 2 and 4 count. Their cross-entropies are $-\ln 0.5 = 0.6931$ and $-\ln 0.25 = 1.3863$, and they sum to 2.0794. The weight and the length normalisation divide by $t \cdot L = 0.5 \times 4 = 2$, so the loss is 1.0397.

An ordinary language-model loss averages over all four positions and gives 0.7489. The two numbers differ because the ordinary loss rewards the model for the visible tokens, and the hidden ones carry the training signal. The script `scripts/figures/087-diffusion-language-models.py` prints both values.

### The gradient

The loss is a sum of cross-entropies on softmax outputs, so the gradient with respect to the logits keeps the familiar shape. For a hidden position $i$ with logits $z_i$ and clean token $y_i$, the row of the gradient is

$$
\frac{1}{t\,L}\left(\operatorname{softmax}(z_i) - \operatorname{onehot}(y_i)\right)
$$

The row is the predicted probabilities minus the one-hot target, scaled by $1/(tL)$. In numpy it is one line per row. Rows for visible positions are zero, because those positions didn't enter the loss. The exercise checks this gradient against `finite_difference`.

> [!KEY]
> Masked diffusion trains a masked language model at every mask rate, with the cross-entropy on hidden positions only and a $1/t$ weight. The weight makes each position count once in expectation, and the result bounds the negative log-likelihood from above (LLaDA, Equations 3 and 4).

> [!WARNING]
> Don't average the cross-entropy over all positions. The visible positions have a loss near zero or an arbitrary value that depends on how the model treats visible input. Averaging over them dilutes the signal by a factor that changes with $t$, and drops the connection to Equation 4.

## The sampler: start blank, unmask the sure slots first

Sampling runs the process backwards. You choose the length $L$ and the number of steps $N$ in advance. The state starts as $L$ copies of `[MASK]`. A step takes the state at time $t$ to time $s = t - 1/N$ in three moves:

1. Run the network once on the partly hidden sequence, and predict a token for every hidden slot.
2. Treat every one of those predictions as provisional.
3. Hide some of the predicted slots again, so that the share of hidden slots matches time $s$.

The reverse process of a masked diffusion model works out as follows. Of the slots hidden at time $t$, each stays hidden at time $s$ with probability $s/t$. LLaDA's Section 2.4 describes this as remasking a fraction $s/t$ of the predicted tokens. That is where the schedule comes from.

### Random versus low-confidence remasking

Two rules choose which slots go back under the mask.

- **Random remasking** (Algorithm 4) hides each predicted slot with probability $s/t$, independent of what the network predicted. It follows the reverse process directly.
- **Low-confidence remasking** (Algorithm 5) takes the largest softmax probability at each hidden slot as its confidence and keeps the most confident predictions. It hides the least confident ones. LLaDA adopts this rule, and credits the idea to MaskGIT.

The reason to prefer low-confidence is practical. A slot where the network is nearly sure is probably right, and filling it early gives every other slot more context. A slot where the network is unsure is better left until more neighbours are filled in. Random remasking commits an unsure slot as readily as a sure one, and a wrong word committed early stays wrong, because the network never revisits a visible token.

Low-confidence remasking has a deterministic count. After step $k$ of $N$, $s = 1 - k/N$, and the number of slots left unmasked is

$$
n_{\text{un}} = \lfloor L\,(1 - s) \rfloor = \lfloor L\,k / N \rfloor
$$

That is Algorithm 5's $n_{\text{un}} = \lfloor L(1-s) \rfloor$. The printed pseudocode follows it with a line that reads as hiding the $n_{\text{un}}$ lowest-confidence positions. Read literally, that would hide a growing number of tokens at each step. This chapter follows the surrounding text and Section 2.4: $n_{\text{un}}$ is the number of positions that stay filled, and the rest go back to `[MASK]`. Slots that were filled at an earlier step count as confidence 1 in the algorithm, so they stay filled.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Five rows of eight tokens for each sampler: low-confidence fills the high-confidence slots first, random fills slots in arbitrary order">
  <text x="160" y="18" text-anchor="middle" class="fig-bold">low-confidence</text>
  <text x="56" y="46" text-anchor="end" class="fig-small">conf</text>
  <text x="72" y="46" text-anchor="middle" class="fig-small">0.87</text>
  <text x="97" y="46" text-anchor="middle" class="fig-small">0.98</text>
  <text x="122" y="46" text-anchor="middle" class="fig-small">0.71</text>
  <text x="147" y="46" text-anchor="middle" class="fig-small">0.95</text>
  <text x="172" y="46" text-anchor="middle" class="fig-small">0.48</text>
  <text x="197" y="46" text-anchor="middle" class="fig-small">0.80</text>
  <text x="222" y="46" text-anchor="middle" class="fig-small">0.60</text>
  <text x="247" y="46" text-anchor="middle" class="fig-small">0.35</text>
  <text x="56" y="80" text-anchor="end" class="fig-small">step 0</text>
  <rect x="60" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="72.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="85" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="97.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="110" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="122.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="135" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="147.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="160" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="172.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="185" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="197.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="210" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="222.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="235" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="247.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <text x="56" y="120" text-anchor="end" class="fig-small">step 1</text>
  <rect x="60" y="96" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="72.0" y="117.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="85" y="96" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="97.0" y="117.0" text-anchor="middle" class="fig-small">2</text>
  <rect x="110" y="96" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="122.0" y="117.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="135" y="96" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="147.0" y="117.0" text-anchor="middle" class="fig-small">3</text>
  <rect x="160" y="96" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="172.0" y="117.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="185" y="96" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="197.0" y="117.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="210" y="96" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="222.0" y="117.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="235" y="96" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="247.0" y="117.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <text x="56" y="160" text-anchor="end" class="fig-small">step 2</text>
  <rect x="60" y="136" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="72.0" y="157.0" text-anchor="middle" class="fig-small">0</text>
  <rect x="85" y="136" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="97.0" y="157.0" text-anchor="middle" class="fig-small">2</text>
  <rect x="110" y="136" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="122.0" y="157.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="135" y="136" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="147.0" y="157.0" text-anchor="middle" class="fig-small">3</text>
  <rect x="160" y="136" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="172.0" y="157.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="185" y="136" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="197.0" y="157.0" text-anchor="middle" class="fig-small">2</text>
  <rect x="210" y="136" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="222.0" y="157.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="235" y="136" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="247.0" y="157.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <text x="56" y="200" text-anchor="end" class="fig-small">step 3</text>
  <rect x="60" y="176" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="72.0" y="197.0" text-anchor="middle" class="fig-small">0</text>
  <rect x="85" y="176" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="97.0" y="197.0" text-anchor="middle" class="fig-small">2</text>
  <rect x="110" y="176" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="122.0" y="197.0" text-anchor="middle" class="fig-small">1</text>
  <rect x="135" y="176" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="147.0" y="197.0" text-anchor="middle" class="fig-small">3</text>
  <rect x="160" y="176" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="172.0" y="197.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="185" y="176" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="197.0" y="197.0" text-anchor="middle" class="fig-small">2</text>
  <rect x="210" y="176" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="222.0" y="197.0" text-anchor="middle" class="fig-small">1</text>
  <rect x="235" y="176" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="247.0" y="197.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <text x="56" y="240" text-anchor="end" class="fig-small">step 4</text>
  <rect x="60" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="72.0" y="237.0" text-anchor="middle" class="fig-small">0</text>
  <rect x="85" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="97.0" y="237.0" text-anchor="middle" class="fig-small">2</text>
  <rect x="110" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="122.0" y="237.0" text-anchor="middle" class="fig-small">1</text>
  <rect x="135" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="147.0" y="237.0" text-anchor="middle" class="fig-small">3</text>
  <rect x="160" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="172.0" y="237.0" text-anchor="middle" class="fig-small">0</text>
  <rect x="185" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="197.0" y="237.0" text-anchor="middle" class="fig-small">2</text>
  <rect x="210" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="222.0" y="237.0" text-anchor="middle" class="fig-small">1</text>
  <rect x="235" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="247.0" y="237.0" text-anchor="middle" class="fig-small">3</text>
  <text x="440" y="18" text-anchor="middle" class="fig-bold">random</text>
  <text x="336" y="46" text-anchor="end" class="fig-small">conf</text>
  <text x="352" y="46" text-anchor="middle" class="fig-small">0.87</text>
  <text x="377" y="46" text-anchor="middle" class="fig-small">0.98</text>
  <text x="402" y="46" text-anchor="middle" class="fig-small">0.71</text>
  <text x="427" y="46" text-anchor="middle" class="fig-small">0.95</text>
  <text x="452" y="46" text-anchor="middle" class="fig-small">0.48</text>
  <text x="477" y="46" text-anchor="middle" class="fig-small">0.80</text>
  <text x="502" y="46" text-anchor="middle" class="fig-small">0.60</text>
  <text x="527" y="46" text-anchor="middle" class="fig-small">0.35</text>
  <text x="336" y="80" text-anchor="end" class="fig-small">step 0</text>
  <rect x="340" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="352.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="365" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="377.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="390" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="402.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="415" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="427.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="440" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="452.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="465" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="477.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="490" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="502.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="515" y="56" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="527.0" y="77.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <text x="336" y="120" text-anchor="end" class="fig-small">step 1</text>
  <rect x="340" y="96" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="352.0" y="117.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="365" y="96" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="377.0" y="117.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="390" y="96" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="402.0" y="117.0" text-anchor="middle" class="fig-small">1</text>
  <rect x="415" y="96" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="427.0" y="117.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="440" y="96" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="452.0" y="117.0" text-anchor="middle" class="fig-small">0</text>
  <rect x="465" y="96" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="477.0" y="117.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="490" y="96" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="502.0" y="117.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="515" y="96" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="527.0" y="117.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <text x="336" y="160" text-anchor="end" class="fig-small">step 2</text>
  <rect x="340" y="136" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="352.0" y="157.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="365" y="136" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="377.0" y="157.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="390" y="136" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="402.0" y="157.0" text-anchor="middle" class="fig-small">1</text>
  <rect x="415" y="136" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="427.0" y="157.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="440" y="136" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="452.0" y="157.0" text-anchor="middle" class="fig-small">0</text>
  <rect x="465" y="136" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="477.0" y="157.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="490" y="136" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="502.0" y="157.0" text-anchor="middle" class="fig-small">1</text>
  <rect x="515" y="136" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="527.0" y="157.0" text-anchor="middle" class="fig-small">3</text>
  <text x="336" y="200" text-anchor="end" class="fig-small">step 3</text>
  <rect x="340" y="176" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="352.0" y="197.0" text-anchor="middle" class="fig-small">0</text>
  <rect x="365" y="176" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="377.0" y="197.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="390" y="176" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="402.0" y="197.0" text-anchor="middle" class="fig-small">1</text>
  <rect x="415" y="176" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="427.0" y="197.0" text-anchor="middle" class="fig-small">3</text>
  <rect x="440" y="176" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="452.0" y="197.0" text-anchor="middle" class="fig-small">0</text>
  <rect x="465" y="176" width="24" height="34" class="fill-sunk stroke-grid"/>
  <text x="477.0" y="197.0" text-anchor="middle" class="fill-muted fig-small">M</text>
  <rect x="490" y="176" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="502.0" y="197.0" text-anchor="middle" class="fig-small">1</text>
  <rect x="515" y="176" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="527.0" y="197.0" text-anchor="middle" class="fig-small">3</text>
  <text x="336" y="240" text-anchor="end" class="fig-small">step 4</text>
  <rect x="340" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="352.0" y="237.0" text-anchor="middle" class="fig-small">0</text>
  <rect x="365" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="377.0" y="237.0" text-anchor="middle" class="fig-small">2</text>
  <rect x="390" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="402.0" y="237.0" text-anchor="middle" class="fig-small">1</text>
  <rect x="415" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="427.0" y="237.0" text-anchor="middle" class="fig-small">3</text>
  <rect x="440" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="452.0" y="237.0" text-anchor="middle" class="fig-small">0</text>
  <rect x="465" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="477.0" y="237.0" text-anchor="middle" class="fig-small">2</text>
  <rect x="490" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="502.0" y="237.0" text-anchor="middle" class="fig-small">1</text>
  <rect x="515" y="216" width="24" height="34" class="fill-soft-a stroke-grid"/>
  <text x="527.0" y="237.0" text-anchor="middle" class="fig-small">3</text>
</svg>
Figure 2. Both samplers on the same table of logits, 8 slots and 4 steps. The conf row is the largest softmax probability in each slot. Notice that low-confidence fills slots 1 and 3 first, which have confidences 0.98 and 0.95, while random fills slots 3 and 5 first.
```

### One sampler run by hand

Take $L = 8$ and $N = 4$. The schedule is $\lfloor 8k/4 \rfloor = 2, 4, 6, 8$, so every step fills two more slots. The toy model is a fixed table of logits, with one row per slot and the same rows at every step. The largest probability in each slot, from slot 1 to slot 8, is 0.87, 0.98, 0.71, 0.95, 0.48, 0.80, 0.60, and 0.36.

- **Step 1.** Everything is hidden, and the target is 2 unmasked slots. The two highest confidences are 0.98 in slot 2 and 0.95 in slot 4, and their tokens are 2 and 3.
- **Step 2.** The target is 4 unmasked, with 2 filled already, so 2 more. Of the hidden slots, slot 1 (0.87) and slot 6 (0.80) are the most confident, and their tokens are 0 and 2.
- **Step 3.** The target is 6. Slot 3 (0.71) and slot 7 (0.60) fill in, with tokens 1 and 1.
- **Step 4.** The remaining slots are 5 (0.48) and 8 (0.36), with tokens 0 and 3, and the page is full.

The final sequence is $(0, 2, 1, 3, 0, 2, 1, 3)$. A fixed table doesn't change as slots fill, so both samplers end on the same sequence and differ only in order. A trained network changes its predictions as context arrives, and that is where the order starts to matter. The toy table shows the mechanism and the bookkeeping, not the quality gap.

When $L$ isn't a multiple of $N$, the floor does the rounding. At $L = 10$ and $N = 4$ the schedule is 2, 5, 7, 10, so the steps fill 2, 3, 2, and 3 slots.

> [!KEY]
> A sampler step predicts every hidden slot, then hides the least confident predictions again, so the count of filled slots follows $\lfloor Lk/N \rfloor$. Low-confidence remasking commits the sure slots first and lets them serve as context for the rest.

> [!TIP]
> To debug a sampler, print the unfilled count after every step. It must follow $\lfloor Lk/N \rfloor$ exactly. A count that jumps ahead or lags means the step is filling slots that were already filled, or filling none.

## Cost and scaling

A causal model needs $L$ passes for $L$ tokens. A masked diffusion sampler needs $N$ passes, and you choose $N$. Setting $N = L$ fills one slot per pass, and setting $N$ below $L$ fills several slots per pass, each chosen without seeing the others, so quality can drop.

Each pass runs the network over the whole length $L$, with no causal mask. A causal model can cache the keys and values of tokens it already wrote and process one token per pass. The key-value cache from the chapter on [decoding and sampling](/lessons/075-decoding-and-sampling) doesn't carry over directly, because a filled slot can change its neighbours' predictions at the next pass. Cost grows with $N$ times the cost of one full-length pass.

## Where this breaks

- **Length is fixed up front.** You choose $L$ before the first step. A causal model stops when it writes an end token, and a masked diffusion model must be told how many slots to fill.
- **Committed tokens stay.** Low-confidence remasking decides which predictions to keep in the current step, and a kept token isn't revisited. A wrong early commitment can't be repaired by later steps.
- **Few steps fill many slots at once.** Slots filled in the same pass are chosen independently, so two slots that need to agree, such as the two halves of a repeated phrase, can disagree.
- **A heavily weighted small $t$.** The $1/t$ weight is large at small $t$, so the loss has high variance from the near-complete sequences. Training draws many values of $t$ to average it out.

## Build it step by step

The vocabulary has $V$ tokens with ids 0 to $V-1$, and the mask id is $V$. `softmax` is given.

1. `forward_mask(tokens, t, rng, mask_id)` draws one uniform number per position from `rng`, hides the positions where the draw is below `t`, and returns `(noisy, mask)`. `noisy` holds `mask_id` at hidden positions and the original token elsewhere, and `mask` is a boolean array. At $t = 0$ nothing is hidden, and at $t = 1$ everything is. The mistake people most often make is to hide a fixed count of tokens, which makes the slots dependent instead of independent coin flips.

2. `masked_diffusion_loss(logits, targets, mask, t)` takes `logits` of shape `(L, V)`. It takes the log of the softmax, picks the log probability of each target with `logp[np.arange(L), targets]`, multiplies by `mask`, sums, and divides by `t * L` with a minus sign. The worked example returns 1.0397. Averaging over all positions returns 0.7489 on it, and dropping the $1/t$ returns 0.5199.

3. `masked_diffusion_loss_grad(logits, targets, mask, t)` returns an array of shape `(L, V)`. Each row is `mask[:, None] * (softmax - one_hot) / (t * L)`. Rows for visible positions are zero. A central difference of `masked_diffusion_loss` matches it.

4. `num_unmasked_after(length, step, n_steps)` returns `(length * step) // n_steps`, in integer arithmetic. For $L = 10$ and $N = 4$ it returns 2, 5, 7, and 10. Rounding to the nearest integer returns 2, 5, 8, and 10, which fills too many slots at step 3.

5. `low_confidence_remask_step(x, logits, mask_id, n_unmasked)` computes the argmax token and the largest probability at every position. It lists the hidden positions, sorts them by confidence from largest to smallest with a stable sort, and fills the first `n_unmasked - (count already unmasked)` of them. Visible positions keep their tokens, even when the model predicts another token there. Sorting ascending and filling the first slots remasks the most confident tokens, which gets the order backwards.

6. `sample(logit_fn, length, n_steps, mask_id)` starts from `np.full(length, mask_id)`, calls `logit_fn(x)` at every step, applies the step with `num_unmasked_after(length, k, n_steps)`, and returns the final array with a list of $N + 1$ snapshots. On the 8-slot table the snapshots hold 0, 2, 4, 6, and 8 filled slots.

> [!RECAP]
> - The forward process hides each token independently with probability $t$, and $t$ is uniform on $[0, 1]$. The mask is absorbing, and at $t = 1$ the sequence is blank.
> - The loss is the cross-entropy on hidden positions only, weighted by $1/t$ (LLaDA Equation 3, divided by $L$ in its Algorithm 1). It bounds the negative log-likelihood from above (Equation 4).
> - On the worked example the loss is 1.0397, and an average over all positions gives 0.7489.
> - The sampler starts blank, predicts every hidden slot, and keeps $\lfloor Lk/N \rfloor$ slots filled after step $k$.
> - Low-confidence remasking keeps the most confident predictions. Random remasking hides slots without looking at confidence.

## The exercise

You implement `forward_mask`, `masked_diffusion_loss`, `masked_diffusion_loss_grad`, `num_unmasked_after`, `low_confidence_remask_step`, and `sample`. The checks verify the masking rate and independence on 20000 tokens, the hand-computed loss, the $1/t$ scaling, the gradient against `finite_difference`, the schedule, and the full unmasking history of the 8-slot table. A loss that averages over all positions fails the hand-computed value with 0.7489 where 1.0397 is expected, and a sampler that fills the least confident slots first fails the step-1 check.

## Further reading

- [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) (the LLaDA paper). The forward process is in Section 2.1, the loss is Equation 3, the bound is Equation 4, the algorithms are 1 for pretraining and 4 and 5 for the two remasking rules. The paper page at [papers.ifkash.dev](https://papers.ifkash.dev/llada/) walks through it.
- [Diffusion models: add noise, then walk it back](/lessons/086-diffusion-models) is the Gaussian counterpart, with the same forward-then-reverse structure.
- [Encoders and cross-attention](/lessons/065-encoders-and-cross-attention) covers masked-token training at a fixed rate.
- [Decoding and sampling](/lessons/075-decoding-and-sampling) covers the one-token-per-pass sampler that this chapter replaces.
