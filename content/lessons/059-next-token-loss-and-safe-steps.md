---
title: "Training a language model: next-token loss, schedules, and clipping"
slug: 059-next-token-loss-and-safe-steps
part: "Part 6 — Transformers"
summary: Score a model by minus the log of the probability it gave the true next token, then keep each step safe with a schedule and a clip.
minutes: 70
gpu: false
objectives:
  - Explain why accuracy cannot train a language model and why minus the log of the true token's probability can.
  - Compute the next-token loss by hand, without overflow, and read it as perplexity.
  - Turn a text into inputs and shifted targets, and explain why the causal mask makes one pass give many honest lessons.
  - Compute the warmup-then-cosine learning-rate schedule by hand.
  - Compute the global norm of a list of gradient arrays and scale them to a limit, keeping their direction.
prereqs:
  - 005-gradient-descent-and-momentum
  - 033-softmax-and-whole-network-gradients
  - 053-scaled-attention-and-heads
  - 057-the-transformer-block
  - 058-counts-to-a-language-model
  - 034-sgd-adam-and-schedules
  - 048-vanishing-gradients-and-clipping
exercise: 059-next-token-loss-and-safe-steps
---

# Training a language model: next-token loss, schedules, and clipping

You have built a stack of transformer blocks. Give it the text "to be or not to" and it answers with a probability for every word in its vocabulary. The true next word is "be". One model puts 0.99 on "be". Another puts 0.01. Both saw the same text and the same answer, and you need one number that tells them apart, because training works by pushing a number down with gradient descent.

The obvious number is accuracy: count how often the most probable word is the right one. Accuracy cannot guide training. A model that says 0.51 and one that says 0.99 both count as right, and when you nudge a weight a little, the winner stays the winner, so accuracy stays flat and its gradient is zero almost everywhere. The number you want reacts to every small change in the probability of the true word, and punishes a confident wrong answer much harder than an unsure one.

Once you have the number, a second problem appears. Your model trains fine for 2,000 steps, the loss falls smoothly, and then one odd batch arrives, the loss jumps to 40, and it never comes back. This chapter builds the number, shows how one pass over a text produces many scores at once, and then adds two controls that keep every step safe: a schedule that changes the learning rate as training goes on, and a clip that shortens any gradient that is too long.

## What you already know

Six ideas carry this chapter:

- A smoothed count table is already a next-word model. Its mean negative log likelihood is the score this chapter uses, and a uniform table has perplexity equal to the vocabulary size (see [from counts to a language model](/lessons/058-counts-to-a-language-model)).
- A softmax turns a list of scores, called logits, into positive numbers that add to 1 (see [softmax and cross-entropy](/lessons/033-softmax-and-whole-network-gradients)). A token is one piece of text, and in this chapter it is a whole word.
- A causal mask stops a position from looking at later tokens. A stack of blocks ends with one vector per position (see [the transformer block](/lessons/057-the-transformer-block)), and one more matrix turns each vector into logits.
- Training walks downhill: a step is the learning rate times a direction taken from the gradient, so a bigger rate means a longer step (see [gradient descent](/lessons/005-gradient-descent-and-momentum)).
- A mini-batch gradient is right on average but noisy at each step.
- The norm of a vector is its length, the square root of the sum of its squared entries.

## Surprise is the cost

Start with a picture. Figure 1 plots a cost against the probability $p$ that the model gave to the true token.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="The cost curve minus log p against the probability given to the true token, with squared error dashed for comparison">
<defs><marker id="c047-nll-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="70" y1="280" x2="70" y2="285" class="stroke-muted"/>
<text x="70" y="300" text-anchor="middle" class="fig-small">0</text>
<line x1="192.5" y1="280" x2="192.5" y2="285" class="stroke-muted"/>
<text x="192.5" y="300" text-anchor="middle" class="fig-small">0.25</text>
<line x1="315" y1="280" x2="315" y2="285" class="stroke-muted"/>
<text x="315" y="300" text-anchor="middle" class="fig-small">0.5</text>
<line x1="437.5" y1="280" x2="437.5" y2="285" class="stroke-muted"/>
<text x="437.5" y="300" text-anchor="middle" class="fig-small">0.75</text>
<line x1="560" y1="280" x2="560" y2="285" class="stroke-muted"/>
<text x="560" y="300" text-anchor="middle" class="fig-small">1</text>
<line x1="65" y1="280" x2="70" y2="280" class="stroke-muted"/>
<text x="61" y="284" text-anchor="end" class="fig-small">0</text>
<line x1="65" y1="230" x2="70" y2="230" class="stroke-muted"/>
<text x="61" y="234" text-anchor="end" class="fig-small">1</text>
<line x1="70" y1="230" x2="560" y2="230" class="stroke-grid"/>
<line x1="65" y1="180" x2="70" y2="180" class="stroke-muted"/>
<text x="61" y="184" text-anchor="end" class="fig-small">2</text>
<line x1="70" y1="180" x2="560" y2="180" class="stroke-grid"/>
<line x1="65" y1="130" x2="70" y2="130" class="stroke-muted"/>
<text x="61" y="134" text-anchor="end" class="fig-small">3</text>
<line x1="70" y1="130" x2="560" y2="130" class="stroke-grid"/>
<line x1="65" y1="80" x2="70" y2="80" class="stroke-muted"/>
<text x="61" y="84" text-anchor="end" class="fig-small">4</text>
<line x1="70" y1="80" x2="560" y2="80" class="stroke-grid"/>
<line x1="65" y1="30" x2="70" y2="30" class="stroke-muted"/>
<text x="61" y="34" text-anchor="end" class="fig-small">5</text>
<line x1="70" y1="30" x2="560" y2="30" class="stroke-grid"/>
<line x1="70" y1="280" x2="560" y2="280" class="stroke-muted"/>
<line x1="70" y1="280" x2="70" y2="22" class="stroke-muted"/>
<path d="M73.9,38.6 L76.4,62.7 L78.8,78.9 L81.2,91.1 L83.6,100.9 L86.1,109.1 L88.5,116.2 L90.9,122.3 L93.4,127.8 L95.8,132.8 L98.2,137.3 L100.7,141.4 L103.1,145.2 L105.5,148.8 L107.9,152.1 L110.4,155.2 L112.8,158.1 L115.2,160.9 L117.7,163.5 L120.1,166 L122.5,168.3 L125,170.6 L127.4,172.8 L129.8,174.8 L132.2,176.8 L134.7,178.8 L137.1,180.6 L139.5,182.4 L142,184.1 L144.4,185.8 L146.8,187.4 L149.3,188.9 L151.7,190.4 L154.1,191.9 L156.6,193.3 L159,194.7 L161.4,196 L163.8,197.4 L166.3,198.6 L168.7,199.9 L171.1,201.1 L173.6,202.3 L176,203.5 L178.4,204.6 L180.9,205.7 L183.3,206.8 L185.7,207.8 L188.1,208.9 L190.6,209.9 L193,210.9 L195.4,211.9 L197.9,212.8 L200.3,213.8 L202.7,214.7 L205.2,215.6 L207.6,216.5 L210,217.4 L212.5,218.2 L214.9,219.1 L217.3,219.9 L219.7,220.7 L222.2,221.5 L224.6,222.3 L227,223.1 L229.5,223.9 L231.9,224.6 L234.3,225.4 L236.8,226.1 L239.2,226.8 L241.6,227.5 L244,228.2 L246.5,228.9 L248.9,229.6 L251.3,230.3 L253.8,231 L256.2,231.6 L258.6,232.3 L261.1,232.9 L263.5,233.5 L265.9,234.2 L268.4,234.8 L270.8,235.4 L273.2,236 L275.6,236.6 L278.1,237.2 L280.5,237.8 L282.9,238.3 L285.4,238.9 L287.8,239.5 L290.2,240 L292.7,240.6 L295.1,241.1 L297.5,241.6 L299.9,242.2 L302.4,242.7 L304.8,243.2 L307.2,243.7 L309.7,244.2 L312.1,244.7 L314.5,245.2 L317,245.7 L319.4,246.2 L321.8,246.7 L324.3,247.2 L326.7,247.7 L329.1,248.1 L331.5,248.6 L334,249.1 L336.4,249.5 L338.8,250 L341.3,250.4 L343.7,250.9 L346.1,251.3 L348.6,251.8 L351,252.2 L353.4,252.6 L355.8,253.1 L358.3,253.5 L360.7,253.9 L363.1,254.3 L365.6,254.7 L368,255.1 L370.4,255.5 L372.9,255.9 L375.3,256.3 L377.7,256.7 L380.2,257.1 L382.6,257.5 L385,257.9 L387.4,258.3 L389.9,258.7 L392.3,259.1 L394.7,259.4 L397.2,259.8 L399.6,260.2 L402,260.5 L404.5,260.9 L406.9,261.3 L409.3,261.6 L411.7,262 L414.2,262.3 L416.6,262.7 L419,263 L421.5,263.4 L423.9,263.7 L426.3,264.1 L428.8,264.4 L431.2,264.7 L433.6,265.1 L436,265.4 L438.5,265.7 L440.9,266.1 L443.3,266.4 L445.8,266.7 L448.2,267.1 L450.6,267.4 L453.1,267.7 L455.5,268 L457.9,268.3 L460.4,268.6 L462.8,268.9 L465.2,269.3 L467.6,269.6 L470.1,269.9 L472.5,270.2 L474.9,270.5 L477.4,270.8 L479.8,271.1 L482.2,271.4 L484.7,271.7 L487.1,271.9 L489.5,272.2 L491.9,272.5 L494.4,272.8 L496.8,273.1 L499.2,273.4 L501.7,273.7 L504.1,273.9 L506.5,274.2 L509,274.5 L511.4,274.8 L513.8,275.1 L516.3,275.3 L518.7,275.6 L521.1,275.9 L523.5,276.1 L526,276.4 L528.4,276.7 L530.8,276.9 L533.3,277.2 L535.7,277.5 L538.1,277.7 L540.6,278 L543,278.2 L545.4,278.5 L547.8,278.7 L550.3,279 L552.7,279.3 L555.1,279.5 L557.6,279.8 L560,280" fill="none" class="stroke-b" stroke-width="3"/>
<path d="M73.9,230.8 L76.4,231.3 L78.8,231.8 L81.2,232.3 L83.6,232.7 L86.1,233.2 L88.5,233.7 L90.9,234.2 L93.4,234.7 L95.8,235.1 L98.2,235.6 L100.7,236.1 L103.1,236.5 L105.5,237 L107.9,237.4 L110.4,237.9 L112.8,238.4 L115.2,238.8 L117.7,239.3 L120.1,239.7 L122.5,240.1 L125,240.6 L127.4,241 L129.8,241.5 L132.2,241.9 L134.7,242.3 L137.1,242.8 L139.5,243.2 L142,243.6 L144.4,244 L146.8,244.5 L149.3,244.9 L151.7,245.3 L154.1,245.7 L156.6,246.1 L159,246.5 L161.4,246.9 L163.8,247.3 L166.3,247.7 L168.7,248.1 L171.1,248.5 L173.6,248.9 L176,249.3 L178.4,249.7 L180.9,250.1 L183.3,250.4 L185.7,250.8 L188.1,251.2 L190.6,251.6 L193,252 L195.4,252.3 L197.9,252.7 L200.3,253.1 L202.7,253.4 L205.2,253.8 L207.6,254.1 L210,254.5 L212.5,254.8 L214.9,255.2 L217.3,255.5 L219.7,255.9 L222.2,256.2 L224.6,256.6 L227,256.9 L229.5,257.2 L231.9,257.6 L234.3,257.9 L236.8,258.2 L239.2,258.6 L241.6,258.9 L244,259.2 L246.5,259.5 L248.9,259.8 L251.3,260.2 L253.8,260.5 L256.2,260.8 L258.6,261.1 L261.1,261.4 L263.5,261.7 L265.9,262 L268.4,262.3 L270.8,262.6 L273.2,262.9 L275.6,263.2 L278.1,263.4 L280.5,263.7 L282.9,264 L285.4,264.3 L287.8,264.6 L290.2,264.8 L292.7,265.1 L295.1,265.4 L297.5,265.7 L299.9,265.9 L302.4,266.2 L304.8,266.4 L307.2,266.7 L309.7,267 L312.1,267.2 L314.5,267.5 L317,267.7 L319.4,267.9 L321.8,268.2 L324.3,268.4 L326.7,268.7 L329.1,268.9 L331.5,269.1 L334,269.4 L336.4,269.6 L338.8,269.8 L341.3,270 L343.7,270.3 L346.1,270.5 L348.6,270.7 L351,270.9 L353.4,271.1 L355.8,271.3 L358.3,271.5 L360.7,271.7 L363.1,271.9 L365.6,272.1 L368,272.3 L370.4,272.5 L372.9,272.7 L375.3,272.9 L377.7,273.1 L380.2,273.3 L382.6,273.4 L385,273.6 L387.4,273.8 L389.9,274 L392.3,274.1 L394.7,274.3 L397.2,274.5 L399.6,274.6 L402,274.8 L404.5,275 L406.9,275.1 L409.3,275.3 L411.7,275.4 L414.2,275.6 L416.6,275.7 L419,275.9 L421.5,276 L423.9,276.1 L426.3,276.3 L428.8,276.4 L431.2,276.5 L433.6,276.7 L436,276.8 L438.5,276.9 L440.9,277 L443.3,277.2 L445.8,277.3 L448.2,277.4 L450.6,277.5 L453.1,277.6 L455.5,277.7 L457.9,277.8 L460.4,277.9 L462.8,278 L465.2,278.1 L467.6,278.2 L470.1,278.3 L472.5,278.4 L474.9,278.5 L477.4,278.6 L479.8,278.7 L482.2,278.7 L484.7,278.8 L487.1,278.9 L489.5,279 L491.9,279 L494.4,279.1 L496.8,279.2 L499.2,279.2 L501.7,279.3 L504.1,279.3 L506.5,279.4 L509,279.5 L511.4,279.5 L513.8,279.6 L516.3,279.6 L518.7,279.6 L521.1,279.7 L523.5,279.7 L526,279.8 L528.4,279.8 L530.8,279.8 L533.3,279.9 L535.7,279.9 L538.1,279.9 L540.6,279.9 L543,279.9 L545.4,280 L547.8,280 L550.3,280 L552.7,280 L555.1,280 L557.6,280 L560,280" fill="none" class="stroke-muted dashed" stroke-width="2"/>
<text x="76" y="258" text-anchor="start" class="fig-small">squared error</text>
<text x="76" y="272" text-anchor="start" class="fig-small">(1 − p)²</text>
<circle cx="560" cy="280" r="5" class="fill-a"/>
<text x="565" y="250" text-anchor="end" class="fig-small">p = 1: cost 0</text>
<circle cx="455.6" cy="268" r="5" class="fill-a"/>
<text x="420" y="214" text-anchor="start" class="fig-small">p = 0.787: cost 0.2395</text>
<line x1="455.6" y1="262" x2="445" y2="222" class="stroke-muted"/>
<circle cx="315" cy="245.3" r="5" class="fill-a"/>
<text x="352" y="176" text-anchor="start" class="fig-small">p = 0.5: cost 0.69</text>
<line x1="315" y1="239.3" x2="350" y2="182" class="stroke-muted"/>
<circle cx="119" cy="164.9" r="5" class="fill-a"/>
<text x="133" y="158.9" text-anchor="start" class="fig-small">p = 0.1: cost 2.30</text>
<circle cx="74.9" cy="49.7" r="5" class="fill-a"/>
<text x="88.9" y="53.7" text-anchor="start" class="fig-small">p = 0.01: cost 4.61</text>
<text x="315" y="324" text-anchor="middle">probability the model gave the true token, p</text>
<text transform="rotate(-90 22 155)" x="22" y="155" text-anchor="middle">cost</text>
</svg>
Figure 1. The cost of a guess is minus the natural log of the probability the model gave the true token. It is 0 at certainty and climbs steeply as p falls toward 0. The dashed squared error (1 − p)² stays below 1 however wrong the model is, so it barely separates 0.5 from 0.01.
```

Read it as surprise. A model that gave the true token probability 1 was not surprised, so the cost is 0. A model that gave it 0.5 was a little surprised, and one that gave it 0.01 was very surprised. The curve is $-\ln p$, where $\ln$ is the natural logarithm, so the cost is in units called nats.

Why not the squared error $(1 - p)^2$ that you used for regression? Look at the dashed curve. It never rises above 1, so a model that gave the true word 0.01 pays 0.98 and one that gave it 0.5 pays 0.25, barely four times as much. Its slope is also small where the model is most wrong. The slope of $-\ln p$ is $-1/p$, which is $-100$ at $p = 0.01$, so a confident mistake gets a strong push. The log has a second reason: the probability a model gives to a whole text is the product of the probabilities of its tokens, and the log of a product is a sum, so the cost of a text is a sum of per-token costs.

## Score two positions by hand

Shrink the vocabulary to three words: "be", "or", and "to". Take the text "to be or". It holds two lessons. After "to" the model must say "be", and after "to be" it must say "or". Suppose the model's logits, in the order be, or, to, are $[2, 0, 0]$ at the first position and $[0, 3, 0]$ at the second.

Exponentiate each logit and divide by the total. At the first position, $e^2 = 7.389$ and the other two give 1 each, so the total is 9.389 and the probabilities are 0.787, 0.107, and 0.107. At the second, $e^3 = 20.086$ and the total is 22.086, so the probabilities are 0.045, 0.909, and 0.045. The cost at each position is minus the log of the true token's probability. At the first position that is $-\ln 0.787 = 0.2395$, and at the second $-\ln 0.909 = 0.0949$. The loss is their mean, $(0.2395 + 0.0949)/2 = 0.1672$. As a formula, with $N$ positions and $p_i[\text{target}_i]$ the probability position $i$ gave to its true token:

$$
L = -\frac{1}{N}\sum_{i=1}^{N} \ln p_i[\text{target}_i]
$$

Read it aloud: for every position, take the log of the probability given to the right token, average those, and flip the sign. You average rather than add, so that a longer text does not look worse only because it has more positions. In torch, with `logits` of shape `(..., V)` and integer `targets` of shape `(...)`:

```python
logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
cost = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
loss = cost.mean()
```

The first line computes every log probability, the second uses `gather` to pick each row's target entry, and the third averages. The gradient is easy to read too. For one position, the gradient with respect to the logits is the probabilities minus a one-hot vector that is 1 at the true token. At the first position that is $(0.787 - 1,\ 0.107,\ 0.107) = (-0.213,\ 0.107,\ 0.107)$: gradient descent raises the true logit and lowers the others, and the push fades to nothing as $p$ approaches 1. The mean over $N$ positions divides each position's gradient by $N$. The exercise checks this against finite differences.

### Why subtract the largest logit

The softmax needs $e^{z}$ for each logit $z$. A float cannot hold $e^{1000}$, which comes out as infinity, and infinity divided by infinity is not a number. Logits can grow to the hundreds during training, so the plain formula fails exactly when a model is confident. The fix uses one fact: $e^{z} = e^{m} e^{z - m}$ for any $m$. Pull $e^{m}$ out of the sum, take the log, and choose $m$ to be the largest logit:

$$
\ln \sum_j e^{z_j} = m + \ln \sum_j e^{z_j - m}
$$

Read it aloud: the log of the total equals the largest logit plus the log of a total in which the largest term is $e^0 = 1$. Every term is then between 0 and 1, so nothing overflows. Take $z = [1000, 999]$ and $m = 1000$. The shifted total is $1 + e^{-1} = 1.3679$, its log is 0.3133, and the answer is 1000.3133. That is `torch.logsumexp`. The log probability is then $z - \ln\sum e^{z}$, which is why the code above never forms a probability at all.

## One pass, many lessons

Training does not feed the model one question at a time. It feeds a window of text and asks for a prediction at every position at once, as in Figure 2.

```figure
<svg viewBox="0 0 620 280" role="img" aria-label="Teacher forcing: the text to be or gives inputs to and be, targets be and or, and a two by two causal mask that blocks each position from seeing its own answer">
<defs><marker id="c047-shift-arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="20" y="46" text-anchor="start" class="fig-bold">text</text>
<rect x="90" y="28" width="60" height="34" rx="6" class="fill-soft-a stroke-a"/>
<text x="120" y="50" text-anchor="middle">to</text>
<rect x="170" y="28" width="60" height="34" rx="6" class="fill-soft-a stroke-a"/>
<text x="200" y="50" text-anchor="middle">be</text>
<rect x="250" y="28" width="60" height="34" rx="6" class="fill-soft-a stroke-a"/>
<text x="280" y="50" text-anchor="middle">or</text>
<text x="20" y="126" text-anchor="start" class="fig-bold">inputs</text>
<rect x="90" y="108" width="60" height="34" rx="6" class="fill-soft-a stroke-a"/>
<text x="120" y="130" text-anchor="middle">to</text>
<rect x="170" y="108" width="60" height="34" rx="6" class="fill-soft-a stroke-a"/>
<text x="200" y="130" text-anchor="middle">be</text>
<text x="20" y="226" text-anchor="start" class="fig-bold">targets</text>
<rect x="90" y="208" width="60" height="34" rx="6" class="fill-soft-c stroke-c"/>
<text x="120" y="230" text-anchor="middle">be</text>
<rect x="170" y="208" width="60" height="34" rx="6" class="fill-soft-c stroke-c"/>
<text x="200" y="230" text-anchor="middle">or</text>
<line x1="120" y1="62" x2="120" y2="106" class="stroke-muted"/>
<line x1="200" y1="62" x2="120" y2="206" class="stroke-muted dashed"/>
<line x1="134" y1="144" x2="134" y2="204" class="stroke-ink" marker-end="url(#c047-shift-arrow2)"/>
<line x1="200" y1="62" x2="200" y2="106" class="stroke-muted"/>
<line x1="280" y1="62" x2="200" y2="206" class="stroke-muted dashed"/>
<line x1="214" y1="144" x2="214" y2="204" class="stroke-ink" marker-end="url(#c047-shift-arrow2)"/>
<text x="250" y="266" text-anchor="middle" class="fig-small">each token predicts the one after it</text>
<text x="490" y="48" text-anchor="middle" class="fig-bold">causal mask</text>
<text x="460" y="64" text-anchor="middle" class="fig-small">to</text>
<text x="520" y="64" text-anchor="middle" class="fig-small">be</text>
<text x="422" y="104" text-anchor="end" class="fig-small">position 1</text>
<text x="422" y="164" text-anchor="end" class="fig-small">position 2</text>
<rect x="430" y="70" width="60" height="60" class="fill-soft-c stroke-c"/>
<text x="460" y="105" text-anchor="middle" class="fig-small">sees</text>
<rect x="490" y="70" width="60" height="60" class="fill-sunk stroke-muted"/>
<text x="520" y="105" text-anchor="middle" class="fig-small">blocked</text>
<rect x="430" y="130" width="60" height="60" class="fill-soft-c stroke-c"/>
<text x="460" y="165" text-anchor="middle" class="fig-small">sees</text>
<rect x="490" y="130" width="60" height="60" class="fill-soft-c stroke-c"/>
<text x="520" y="165" text-anchor="middle" class="fig-small">sees</text>
<text x="490" y="218" text-anchor="middle" class="fig-small">position 1 cannot read “be”,</text>
<text x="490" y="236" text-anchor="middle" class="fig-small">its own answer</text>
</svg>
Figure 2. The text “to be or” gives two lessons from one pass. The targets are the inputs moved one place left, so each position predicts the token after it. The causal mask on the right lets position 1 see only “to”, so it cannot copy its answer, “be”, from the next input.
```

The inputs are the tokens from the start of the window to the second to last, and the targets are the same tokens moved one place left, so each target is the token after its input. A window of $T + 1$ tokens gives $T$ lessons from a single forward pass. Feeding the true text, rather than the model's own earlier guesses, is called **teacher forcing**: the teacher supplies the correct previous words, so one early mistake does not wreck every later position.

Why are those $T$ lessons honest? Position 1 of "to be or" has a target "be", and "be" is the input at position 2. Without a mask, position 1 could read its answer from the next input and copy it, and the loss would fall to zero while the model learned nothing. The causal mask removes that path. In code, with `block_size` the number of lessons:

```python
inputs = tokens[start : start + block_size]
targets = tokens[start + 1 : start + block_size + 1]
```

Slicing past the end of an array quietly returns something shorter, so check the window fits.

## Read the loss: ln V and perplexity

A fresh model has small random weights, so its logits sit near zero. A softmax of equal numbers gives every word the probability $1/V$, where $V$ is the vocabulary size. The cost of the true word is then $-\ln(1/V) = \ln V$, whichever word is true: with 11 words, $\ln 11 = 2.398$, and with 50,000 words it is 10.82. A first loss close to $\ln V$ suggests that tokens, logits, and loss are wired correctly.

A loss of 2.398 means little on its own. A better unit comes from a model that picks uniformly among $k$ words. It gives the true word $1/k$, so its loss is $\ln k$. Turn that around: $e^{\text{loss}}$ equals $k$. So define the **perplexity** as $e^{L}$, the number of equally likely words that would produce the same loss:

| Model | Loss | Perplexity |
|---|---|---|
| Certain of the right word | 0 | 1 |
| Untrained, 11 words | 2.398 | 11 |

The worked loss of 0.1672 gives $e^{0.1672} = 1.18$: the model behaves as if it were choosing between about 1.2 words. In code it is `math.exp(float(loss))`.

> [!KEY]
> A language model is scored by minus the log of the probability it gave to the true next token, averaged over every position. A confident right answer costs almost nothing (0.99 costs 0.01), a confident wrong one costs a lot (0.01 costs 4.6), and the cost changes with every small nudge. An untrained model scores $\ln V$, and perplexity, $e^{L}$, reads the loss as a count of choices.

## Start gently, finish gently

Now the odd batch. Training is walking downhill in fog. At each step the gradient tells you the slope under your feet, and you move a distance set by the learning rate. The slope you feel at one spot is only a guess about the ground a few steps away. Two things can go wrong: the learning rate can be wrong for the stage of training, and a single gradient can be far longer than any other. The schedule handles the first. Figure 3 shows the one this chapter uses: the rate climbs in a straight line, then glides down to a floor.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="The warmup then cosine learning rate schedule with four marked points, against a constant rate and a linear decay">
<line x1="70" y1="260" x2="70" y2="265" class="stroke-muted"/>
<text x="70" y="280" text-anchor="middle" class="fig-small">0</text>
<line x1="162" y1="260" x2="162" y2="265" class="stroke-muted"/>
<text x="162" y="280" text-anchor="middle" class="fig-small">10</text>
<line x1="254" y1="260" x2="254" y2="265" class="stroke-muted"/>
<text x="254" y="280" text-anchor="middle" class="fig-small">20</text>
<line x1="346" y1="260" x2="346" y2="265" class="stroke-muted"/>
<text x="346" y="280" text-anchor="middle" class="fig-small">30</text>
<line x1="438" y1="260" x2="438" y2="265" class="stroke-muted"/>
<text x="438" y="280" text-anchor="middle" class="fig-small">40</text>
<line x1="530" y1="260" x2="530" y2="265" class="stroke-muted"/>
<text x="530" y="280" text-anchor="middle" class="fig-small">50</text>
<line x1="65" y1="260" x2="70" y2="260" class="stroke-muted"/>
<text x="61" y="264" text-anchor="end" class="fig-small">0</text>
<line x1="65" y1="207.7" x2="70" y2="207.7" class="stroke-muted"/>
<text x="61" y="211.7" text-anchor="end" class="fig-small">0.5</text>
<line x1="70" y1="207.7" x2="530" y2="207.7" class="stroke-grid"/>
<line x1="65" y1="155.5" x2="70" y2="155.5" class="stroke-muted"/>
<text x="61" y="159.5" text-anchor="end" class="fig-small">1</text>
<line x1="70" y1="155.5" x2="530" y2="155.5" class="stroke-grid"/>
<line x1="65" y1="103.2" x2="70" y2="103.2" class="stroke-muted"/>
<text x="61" y="107.2" text-anchor="end" class="fig-small">1.5</text>
<line x1="70" y1="103.2" x2="530" y2="103.2" class="stroke-grid"/>
<line x1="65" y1="50.9" x2="70" y2="50.9" class="stroke-muted"/>
<text x="61" y="54.9" text-anchor="end" class="fig-small">2</text>
<line x1="70" y1="50.9" x2="530" y2="50.9" class="stroke-grid"/>
<line x1="70" y1="260" x2="530" y2="260" class="stroke-muted"/>
<line x1="70" y1="260" x2="70" y2="24" class="stroke-muted"/>
<line x1="162" y1="50.9" x2="530" y2="50.9" class="stroke-muted dashed"/>
<line x1="162" y1="50.9" x2="530" y2="249.5" class="stroke-d dashed"/>
<path d="M70,260 L72.3,254.8 L74.6,249.5 L76.9,244.3 L79.2,239.1 L81.5,233.9 L83.8,228.6 L86.1,223.4 L88.4,218.2 L90.7,213 L93,207.7 L95.3,202.5 L97.6,197.3 L99.9,192 L102.2,186.8 L104.5,181.6 L106.8,176.4 L109.1,171.1 L111.4,165.9 L113.7,160.7 L116,155.5 L118.3,150.2 L120.6,145 L122.9,139.8 L125.2,134.5 L127.5,129.3 L129.8,124.1 L132.1,118.9 L134.4,113.6 L136.7,108.4 L139,103.2 L141.3,98 L143.6,92.7 L145.9,87.5 L148.2,82.3 L150.5,77 L152.8,71.8 L155.1,66.6 L157.4,61.4 L159.7,56.1 L162,50.9 L164.3,50.9 L166.6,51 L168.9,51.1 L171.2,51.2 L173.5,51.4 L175.8,51.6 L178.1,51.8 L180.4,52.1 L182.7,52.5 L185,52.8 L187.3,53.2 L189.6,53.7 L191.9,54.1 L194.2,54.6 L196.5,55.2 L198.8,55.8 L201.1,56.4 L203.4,57 L205.7,57.7 L208,58.5 L210.3,59.2 L212.6,60 L214.9,60.9 L217.2,61.7 L219.5,62.6 L221.8,63.6 L224.1,64.5 L226.4,65.5 L228.7,66.6 L231,67.6 L233.3,68.7 L235.6,69.9 L237.9,71 L240.2,72.2 L242.5,73.5 L244.8,74.7 L247.1,76 L249.4,77.3 L251.7,78.6 L254,80 L256.3,81.4 L258.6,82.8 L260.9,84.3 L263.2,85.7 L265.5,87.2 L267.8,88.7 L270.1,90.3 L272.4,91.8 L274.7,93.4 L277,95 L279.3,96.7 L281.6,98.3 L283.9,100 L286.2,101.7 L288.5,103.4 L290.8,105.1 L293.1,106.9 L295.4,108.6 L297.7,110.4 L300,112.2 L302.3,114 L304.6,115.9 L306.9,117.7 L309.2,119.5 L311.5,121.4 L313.8,123.3 L316.1,125.2 L318.4,127 L320.7,128.9 L323,130.9 L325.3,132.8 L327.6,134.7 L329.9,136.6 L332.2,138.6 L334.5,140.5 L336.8,142.4 L339.1,144.4 L341.4,146.3 L343.7,148.3 L346,150.2 L348.3,152.2 L350.6,154.1 L352.9,156.1 L355.2,158 L357.5,160 L359.8,161.9 L362.1,163.8 L364.4,165.8 L366.7,167.7 L369,169.6 L371.3,171.5 L373.6,173.4 L375.9,175.3 L378.2,177.2 L380.5,179.1 L382.8,180.9 L385.1,182.8 L387.4,184.6 L389.7,186.4 L392,188.2 L394.3,190 L396.6,191.8 L398.9,193.6 L401.2,195.3 L403.5,197 L405.8,198.8 L408.1,200.4 L410.4,202.1 L412.7,203.8 L415,205.4 L417.3,207 L419.6,208.6 L421.9,210.2 L424.2,211.7 L426.5,213.2 L428.8,214.7 L431.1,216.2 L433.4,217.6 L435.7,219.1 L438,220.5 L440.3,221.8 L442.6,223.2 L444.9,224.5 L447.2,225.7 L449.5,227 L451.8,228.2 L454.1,229.4 L456.4,230.6 L458.7,231.7 L461,232.8 L463.3,233.9 L465.6,234.9 L467.9,235.9 L470.2,236.9 L472.5,237.8 L474.8,238.7 L477.1,239.6 L479.4,240.4 L481.7,241.2 L484,242 L486.3,242.7 L488.6,243.4 L490.9,244.1 L493.2,244.7 L495.5,245.3 L497.8,245.8 L500.1,246.3 L502.4,246.8 L504.7,247.2 L507,247.6 L509.3,248 L511.6,248.3 L513.9,248.6 L516.2,248.9 L518.5,249.1 L520.8,249.2 L523.1,249.4 L525.4,249.5 L527.7,249.5 L530,249.5" fill="none" class="stroke-b" stroke-width="3"/>
<circle cx="116" cy="155.5" r="5" class="fill-a"/>
<circle cx="162" cy="50.9" r="5" class="fill-a"/>
<circle cx="346" cy="150.2" r="5" class="fill-a"/>
<circle cx="530" cy="249.5" r="5" class="fill-a"/>
<text x="108" y="145.5" text-anchor="end" class="fig-small">(5, 1.0)</text>
<text x="162" y="38.9" text-anchor="middle" class="fig-small">(10, 2.0)</text>
<text x="334" y="174.2" text-anchor="end" class="fig-small">(30, 1.05)</text>
<text x="540" y="253.5" text-anchor="start" class="fig-small">(50, 0.1)</text>
<text x="327.6" y="42.9" text-anchor="start" class="fig-small">constant rate</text>
<text x="392.8" y="158.1" text-anchor="start" class="fig-small">linear decay</text>
<text x="97.6" y="68.9" text-anchor="start" class="fig-small">warmup</text>
<text x="300" y="304" text-anchor="middle">step</text>
<text transform="rotate(-90 22 145)" x="22" y="145" text-anchor="middle">learning rate</text>
</svg>
Figure 3. The schedule from warmup_cosine_lr(step, 10, 50, 2, 0.1), drawn from computed values. The rate climbs in a straight line to 2 over 10 steps, then glides to 0.1 along half a cosine. The two pieces meet at step 10 with no jump, and the curve is flat at the peak and at the floor. The dashed lines are a constant rate and a straight-line decay, which agrees with the cosine at step 30.
```

Why start small? At step 0 the weights are random, so the first gradients come from a model that knows nothing. Adam, the optimiser you will wire in next, also has no history yet: its running averages have seen one or two batches. A big step would trust a guess that has no evidence behind it. It is like a driver pulling away on ice: ease in, then speed up.

Why not keep the rate small forever? Because the middle of training needs long steps to cover ground. Why decay at the end? Each batch gives a slightly different gradient, so every step is the true downhill direction plus noise. A large step amplifies that noise, and in a narrow valley the path bounces from wall to wall and never settles. A small step amplifies it less, so the path comes to rest near the bottom. The schedule takes bold steps while the ground is far away and small steps when you are close.

Why a cosine? It is smooth and flat at both ends: it holds near the peak for a while, falls fastest in the middle, and eases onto the floor. A straight line works too, as the dashed line in Figure 3 shows, but a sudden drop jolts the run. The floor stays above zero, often at about a tenth of the peak, so a little learning continues at the end.

Take a warmup of 10 steps, a total of 50, a peak of 2, and a floor of 0.1. At step 5 the rate is $2 \times 5 / 10 = 1.0$. At step 10 warmup is done and the rate is the peak, 2. At step 30 you are halfway through the glide, because $(30 - 10)/(50 - 10) = 0.5$. The cosine of $\pi \times 0.5$ is 0, so the rate is $0.1 + 0.5 \times (2 - 0.1) \times (1 + 0) = 1.05$. At step 50, progress is 1, the cosine of $\pi$ is $-1$, and the rate lands on the floor, 0.1. Those are the four dots in Figure 3. As a formula, with $s$ the step and progress $= (s - \text{warmup})/(\text{total} - \text{warmup})$:

$$
\text{lr}(s) =
\begin{cases}
\text{peak}\cdot \dfrac{s}{\text{warmup}} & s < \text{warmup} \\[1.5ex]
\text{floor} + \tfrac{1}{2}(\text{peak} - \text{floor})\big(1 + \cos(\pi \cdot \text{progress})\big) & s \ge \text{warmup}
\end{cases}
$$

Read it aloud: before the end of warmup, take a share of the peak equal to the fraction of warmup done. After it, start at the peak, where the cosine is 1, and slide to the floor, where the cosine is $-1$. In numpy:

```python
if step < warmup:
    return peak * step / warmup
progress = (step - warmup) / (total - warmup)
return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))
```

The function only maps a number to a rate. When `train_step` calls it in [the training loop](/lessons/060-training-loop-and-cost), it passes the count of the update about to be applied, starting at 1, so the first update uses a rate of peak divided by warmup and never 0.

## Cap the length of a step

The schedule cannot protect you from one bad batch, because it does not look at the gradient. A rare batch, such as an odd document, can produce a gradient far longer than usual. With plain gradient descent, that gradient times the learning rate is a step that is equally too long. Adam divides by a running size, so one step stays bounded, but the spike still enters its running averages and keeps pushing for many steps afterwards. Either way, you want a limit on what one batch can say.

Think of a hiker who is told never to take a step longer than one metre, however steep the slope looks. The limit goes on the gradient's length, not on each of its numbers, because the direction is the useful part. Figure 4 shows the difference.

```figure
<svg viewBox="0 0 640 290" role="img" aria-label="Gradient arrow of length 5 at 53 degrees, the same arrow shrunk by a global scale factor to length 2.5, and an entry-wise clip that bends it to 45 degrees">
<defs><marker id="c047-clip-arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<defs><marker id="c047-clip-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker></defs>
<defs><marker id="c047-clip-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<defs><marker id="c047-clip-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-bad"/></marker></defs>
<circle cx="200" cy="250" r="4" class="fill-ink"/>
<path d="M300,250 L300,247.4 L299.9,244.8 L299.7,242.2 L299.5,239.5 L299.1,236.9 L298.8,234.4 L298.3,231.8 L297.8,229.2 L297.2,226.7 L296.6,224.1 L295.9,221.6 L295.1,219.1 L294.3,216.6 L293.4,214.2 L292.4,211.7 L291.4,209.3 L290.3,206.9 L289.1,204.6 L287.9,202.3 L286.6,200 L285.3,197.8 L283.9,195.5 L282.4,193.4 L280.9,191.2 L279.3,189.1 L277.7,187.1 L276,185.1 L274.3,183.1 L272.5,181.2 L270.7,179.3 L268.8,177.5 L266.9,175.7 L264.9,174 L262.9,172.3 L260.9,170.7 L258.8,169.1 L256.6,167.6 L254.5,166.1 L252.2,164.7 L250,163.4 L247.7,162.1 L245.4,160.9 L243.1,159.7 L240.7,158.6 L238.3,157.6 L235.8,156.6 L233.4,155.7 L230.9,154.9 L228.4,154.1 L225.9,153.4 L223.3,152.8 L220.8,152.2 L218.2,151.7 L215.6,151.2 L213.1,150.9 L210.5,150.5 L207.8,150.3 L205.2,150.1 L202.6,150 L200,150" fill="none" class="stroke-muted dashed"/>
<text x="306" y="226.1" text-anchor="start" class="fig-small">limit 2.5</text>
<line x1="200" y1="250" x2="317" y2="94" class="stroke-a" marker-end="url(#c047-clip-a)" stroke-width="3"/>
<text x="328" y="94" text-anchor="start" class="fig-small">gradient (3, 4), length 5</text>
<line x1="200" y1="250" x2="297" y2="153" class="stroke-bad" marker-end="url(#c047-clip-bad)" stroke-width="3"/>
<text x="310" y="154" text-anchor="start" class="fig-small">each entry clipped to 2.5: (2.5, 2.5)</text>
<line x1="200" y1="250" x2="258" y2="172" class="stroke-b" marker-end="url(#c047-clip-b)" stroke-width="3"/>
<text x="248" y="174" text-anchor="end" class="fig-small">global scale (1.5, 2)</text>
<text x="214" y="236" text-anchor="start" class="fig-small"></text>
</svg>
Figure 4. One gradient, drawn with its first entry across and its second entry up, and two ways to limit it. Multiplying both entries by 0.5 shrinks the arrow to the circle of radius 2.5 and keeps its direction, 53.1 degrees. Clipping each entry to 2.5 turns it to 45 degrees, a different direction, and the result still has length 3.54, which is over the limit.
```

The **global norm** is the length of all the gradients taken together. A model has hundreds of arrays, such as one matrix per layer, but only one point in weight space, so treat every number in every array as one long vector and measure its length. Take two arrays, $[3, 4]$ and $[0]$. The squares sum to $9 + 16 + 0 = 25$ and the norm is 5. Set a limit of 2.5. The gradient is twice too long, so multiply every entry of every array by $2.5/5 = 0.5$. The array $[3, 4]$ becomes $[1.5, 2]$, which has length 2.5 and points the same way. With a limit of 10 the gradient is already short enough, and nothing changes. Clipping each entry to 2.5 instead gives $[2.5, 2.5]$. That arrow points at 45 degrees instead of 53.1, so the step heads a different way, and its length is still 3.54. It fixed neither problem.

As a formula, with $g$ the global norm:

$$
g = \sqrt{\sum_{\text{all entries}} \text{grad}^2}, \qquad
\text{grad} \leftarrow \text{grad} \cdot \min\!\Big(1, \frac{\text{max\_norm}}{g}\Big)
$$

Read it aloud: measure the total length, and if it is over the limit, shrink every entry by the limit divided by the length. The `min` with 1 means a short gradient is left alone. In numpy:

```python
total = math.sqrt(sum(float((g**2).sum()) for g in grads))
if total > max_norm:
    for g in grads:
        g *= max_norm / total
```

The function also returns `total`, the norm from before the clip, so you can log it. A healthy log starts high, falls, and wanders around a steady value, such as 1.9, 1.3, 0.9, 0.7, 0.6, 0.7. A spike is one number far above its neighbours: 0.6, 0.7, 31.0, 0.6. With a limit of 1.0, the clip shrinks that step by a factor of 31 and the run carries on.

> [!WARNING]
> Scale in place with `g *= scale`. The line `g = g * scale` builds another array and leaves the caller's gradients unchanged, so nothing is clipped.

> [!KEY]
> A safe step has a rate suited to the stage of training and a bounded length. The schedule sets the rate: small while the weights are random and the averages are empty, large through the middle, small again to settle. The clip limits the length of any one gradient by scaling all of it together, which keeps its direction. Together they answer the second question: one odd batch cannot write an arbitrarily long gradient into the update, and no stage of training runs at a rate meant for another.

## Where this breaks

**A first loss far above $\ln V$** means the logits start too large, so fix the output matrix scale before you train. **A loss that plunges toward 0 within a few steps on real text** means the model can see its answer, so check the causal mask and the shift of the targets. Perplexity also depends on the vocabulary, so two models with different tokenizers do not compare directly.

**A clip set near the typical norm** acts on every step, so every update has the same length and the gradient's size is lost. Choose a limit that catches spikes and leaves ordinary steps alone. A limit of 1.0 is a common choice for transformers. If the norm is over the limit on every step, raise the limit or lower the learning rate.

**A norm that is `nan` or infinite** is not fixed by scaling: `nan > max_norm` is false, so the clip leaves it alone, and the weights turn to `nan`. Skip the step instead. And the schedule needs the true number of steps: if `total` is larger than the run you actually finish, the rate never reaches its floor.

> [!RECAP]
> - The cost of one position is $-\ln p$, where $p$ is the probability given to the true token. The loss is the mean over positions, and perplexity is $e^{L}$. An untrained model scores $\ln V$.
> - Compute log probabilities as logits minus `logsumexp`, so large logits never overflow.
> - Inputs are a window of tokens, and targets are the same window moved one place left. The causal mask keeps each position from seeing its answer.
> - The learning rate ramps up in a straight line over warmup, then glides down along half a cosine to a floor above zero.
> - The global norm treats every entry of every gradient array as one long vector. Clipping multiplies every array by the same factor, `max_norm / norm`, only when the norm is over the limit, in place, and returns the pre-clip norm.

## Build it step by step

The checks test six functions. The file also holds `causal_lm_loss`, already written, which is the mean of your `per_token_loss`, and `lr_trace`, which lists the schedule at evenly spaced steps so you can see its shape.

1. **`per_token_loss(logits, targets)`.** Logits have shape `(..., V)` and targets have shape `(...)`. Return the cost of every position, shape `(...)`, with `torch.logsumexp` and `gather`. Keep `dim=-1` and `keepdim=True` on the `logsumexp` so the subtraction lines up. The usual mistake is to build the softmax first, which overflows on large logits.
2. **`perplexity(loss)`.** Return `math.exp(float(loss))`. The usual mistake is to report the loss itself.
3. **`make_lm_batch(tokens, block_size, start)`.** Return the pair `(inputs, targets)`, each of length `block_size`, with targets one place to the left of inputs. Raise `ValueError` when the window would run past the end of `tokens`, because a slice would silently return fewer targets.
4. **`warmup_cosine_lr(step, warmup, total, peak, floor)`.** Return a float. Below `warmup`, return `peak * step / warmup`. From `warmup` on, compute progress as `(step - warmup) / (total - warmup)` and apply the cosine. The usual mistake is dividing by `total` alone, which makes the glide miss the floor.
5. **`global_norm(grads)`.** Take a list of numpy arrays or torch tensors of any shapes. Return the square root of the sum of every squared entry, as a float. Sum the per-array totals with `float(...)` before the square root. Change nothing.
6. **`clip_global_norm(grads, max_norm)`.** Call `global_norm`, and when the total is over `max_norm`, multiply every array by `max_norm / total` in place with `g *= scale`. Return the norm from before the clip. The usual mistake is `g = g * scale`.

## The exercise

You write `per_token_loss`, `perplexity`, `make_lm_batch`, `warmup_cosine_lr`, `global_norm`, and `clip_global_norm`. The checks match the hand costs of 0.2395 and 0.0949, stay finite on logits of 1000, test the gradient against finite differences, and confirm that uniform logits give $\ln V$ and perplexity $V$. They check the shifted windows, the schedule at the four worked points, and the global norm on mixed shapes. They confirm that clipping scales to exactly the limit, keeps the direction, works in place, and leaves a short gradient unchanged. Run trains a bigram table on a line of text, prints the loss falling from $\ln V$, and prints the schedule as a bar chart.
