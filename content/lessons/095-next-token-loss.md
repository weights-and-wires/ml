---
title: "How a language model is scored: next-token loss and perplexity"
slug: 095-next-token-loss
part: "Part 6 — Transformers"
summary: Score a language model by minus the log of the probability it gave the true next token, then read that score as perplexity.
minutes: 35
gpu: false
objectives:
  - Explain why accuracy cannot train a language model and why minus the log of the true token's probability can.
  - Compute the next-token loss by hand on two positions, and read its formula aloud.
  - Turn a text into inputs and shifted targets, and explain why the causal mask makes one pass give many honest lessons.
  - Compute the loss without overflow, check that an untrained model scores the log of the vocabulary size, and read the loss as perplexity.
prereqs:
  - 008-walking-downhill
  - 056-softmax-cross-entropy-gradient
  - 089-heads-and-causal-mask
  - 094-the-residual-block
exercise: 095-next-token-loss
---

# How a language model is scored: next-token loss and perplexity

You have built a stack of transformer blocks. Give it the text "to be or not to" and it answers with a probability for every word in its vocabulary. The true next word is "be". One model puts 0.99 on "be". Another puts 0.01. Both saw the same text and the same answer, and you need one number that tells them apart, because training works by pushing a number down with gradient descent.

The obvious number is accuracy: count how often the most probable word is the right one. Accuracy cannot guide training. A model that says 0.51 and one that says 0.99 both count as right, so accuracy sees no difference between them. When you nudge a weight a little, the probabilities move a little but the winner stays the winner, so accuracy stays flat and its gradient is zero almost everywhere. The number you want reacts to every small change in the probability of the true word, and punishes a confident wrong answer much harder than an unsure one.

This chapter builds that number. It then shows how one pass over a text produces many such scores at once, and ends with a way to read the number in plain words.

## What you already know

Five ideas carry this chapter:

- A softmax turns a list of scores, called logits, into positive numbers that add to 1 (see "Softmax and cross-entropy: a loss with a gradient you can read").
- A token is one piece of text. In this chapter a token is a whole word.
- Training walks downhill: it moves each weight against the gradient of one number, the loss (see "Walking downhill: slope, gradient, and one step").
- A causal mask stops a position from looking at later tokens (see "Many heads and the causal mask").
- A stack of blocks ends with one vector per position (see "Stacking it up: residuals, layer norm, and the pre-norm block"), and one more matrix turns each vector into logits.

## Surprise is the cost

Start with a picture. Figure 1 plots a cost against the probability $p$ that the model gave to the true token.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="The cost curve minus log p against the probability given to the true token, with squared error dashed for comparison">
<defs><marker id="c105-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
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

Read it as surprise. A model that gave the true token probability 1 was not surprised, so the cost is 0. A model that gave it 0.5 was a little surprised, and one that gave it 0.01 was very surprised. A model that gave it 0 would be infinitely surprised, and the curve runs off the top of the page. The curve is $-\ln p$, where $\ln$ is the natural logarithm, so the cost is in units called nats.

Why not the squared error $(1 - p)^2$ that you used for regression? Look at the dashed curve. It never rises above 1, so a model that gave the true word 0.01 pays 0.98 and one that gave it 0.5 pays 0.25, barely four times as much. Its slope is also small where the model is most wrong, at most 2 in size. A model that is badly wrong would learn slowly. The slope of $-\ln p$ is $-1/p$, which is $-100$ at $p = 0.01$, so a confident mistake gets a strong push.

The log has a second reason. The probability a model gives to a whole text is the product of the probabilities of its tokens, and the log of a product is a sum, so the cost of a text is a sum of per-token costs.

## Score two positions by hand

Shrink the vocabulary to three words: "be", "or", and "to". Take the text "to be or". It holds two lessons. After "to" the model must say "be", and after "to be" it must say "or". Suppose the model's logits, in the order be, or, to, are $[2, 0, 0]$ at the first position and $[0, 3, 0]$ at the second.

Exponentiate each logit and divide by the total. At the first position, $e^2 = 7.389$ and the other two give 1 each, so the total is 9.389 and the probabilities are 0.787, 0.107, and 0.107. At the second, $e^3 = 20.086$ and the total is 22.086, so the probabilities are 0.045, 0.909, and 0.045. Figure 2 draws them with the true token dark.

```figure
<svg viewBox="0 0 620 280" role="img" aria-label="Softmax probabilities for the two worked positions and for an untrained model over eleven words, with the true next token highlighted">
<text x="117" y="40" text-anchor="middle" class="fig-bold">after “to”</text>
<rect x="30" y="79.8" width="50" height="110.2" class="fill-b stroke-b"/>
<text x="55" y="73.8" text-anchor="middle" class="fig-small">0.787</text>
<text x="55" y="206" text-anchor="middle" class="fig-small">be</text>
<rect x="92" y="175.1" width="50" height="14.9" class="fill-soft-b stroke-b"/>
<text x="117" y="169.1" text-anchor="middle" class="fig-small">0.107</text>
<text x="117" y="206" text-anchor="middle" class="fig-small">or</text>
<rect x="154" y="175.1" width="50" height="14.9" class="fill-soft-b stroke-b"/>
<text x="179" y="169.1" text-anchor="middle" class="fig-small">0.107</text>
<text x="179" y="206" text-anchor="middle" class="fig-small">to</text>
<line x1="24" y1="190" x2="210" y2="190" class="stroke-muted"/>
<text x="117" y="232" text-anchor="middle" class="fig-small">target be: loss 0.2395</text>
<text x="327" y="40" text-anchor="middle" class="fig-bold">after “to be”</text>
<rect x="240" y="183.7" width="50" height="6.3" class="fill-soft-b stroke-b"/>
<text x="265" y="177.7" text-anchor="middle" class="fig-small">0.045</text>
<text x="265" y="206" text-anchor="middle" class="fig-small">be</text>
<rect x="302" y="62.7" width="50" height="127.3" class="fill-b stroke-b"/>
<text x="327" y="56.7" text-anchor="middle" class="fig-small">0.909</text>
<text x="327" y="206" text-anchor="middle" class="fig-small">or</text>
<rect x="364" y="183.7" width="50" height="6.3" class="fill-soft-b stroke-b"/>
<text x="389" y="177.7" text-anchor="middle" class="fig-small">0.045</text>
<text x="389" y="206" text-anchor="middle" class="fig-small">to</text>
<line x1="234" y1="190" x2="420" y2="190" class="stroke-muted"/>
<text x="327" y="232" text-anchor="middle" class="fig-small">target or: loss 0.0949</text>
<text x="515" y="40" text-anchor="middle" class="fig-bold">untrained, 11 words</text>
<rect x="445" y="177.3" width="10" height="12.7" class="fill-soft-b stroke-b"/>
<rect x="458" y="177.3" width="10" height="12.7" class="fill-soft-b stroke-b"/>
<rect x="471" y="177.3" width="10" height="12.7" class="fill-soft-b stroke-b"/>
<rect x="484" y="177.3" width="10" height="12.7" class="fill-b stroke-b"/>
<rect x="497" y="177.3" width="10" height="12.7" class="fill-soft-b stroke-b"/>
<rect x="510" y="177.3" width="10" height="12.7" class="fill-soft-b stroke-b"/>
<rect x="523" y="177.3" width="10" height="12.7" class="fill-soft-b stroke-b"/>
<rect x="536" y="177.3" width="10" height="12.7" class="fill-soft-b stroke-b"/>
<rect x="549" y="177.3" width="10" height="12.7" class="fill-soft-b stroke-b"/>
<rect x="562" y="177.3" width="10" height="12.7" class="fill-soft-b stroke-b"/>
<rect x="575" y="177.3" width="10" height="12.7" class="fill-soft-b stroke-b"/>
<line x1="439" y1="190" x2="591" y2="190" class="stroke-muted"/>
<text x="515" y="232" text-anchor="middle" class="fig-small">loss ln 11 = 2.398</text>
<text x="515" y="171.3" text-anchor="middle" class="fig-small">0.091</text>
<text x="515" y="250" text-anchor="middle" class="fig-small">perplexity 11</text>
<text x="310" y="272" text-anchor="middle" class="fig-small">the dark bar is the true next token</text>
</svg>
Figure 2. The model's probabilities at the two positions of “to be or”, and a flat model over 11 words. The loss reads only the height of the dark bar: a tall bar means a small loss, and a flat model pays ln 11 whichever word is true.
```

The cost at each position is minus the log of the dark bar. At the first position that is $-\ln 0.787 = 0.2395$, and at the second $-\ln 0.909 = 0.0949$. The loss is their mean, $(0.2395 + 0.0949)/2 = 0.1672$. The other bars matter only through the total, which shrinks the dark bar when they grow.

As a formula, with $N$ positions and $p_i[\text{target}_i]$ the probability position $i$ gave to its true token:

$$
L = -\frac{1}{N}\sum_{i=1}^{N} \ln p_i[\text{target}_i]
$$

Read it aloud: for every position, take the log of the probability given to the right token, average those, and flip the sign. You average rather than add, so that a longer text does not look worse only because it has more positions. In torch, with `logits` of shape `(..., V)` and integer `targets` of shape `(...)`:

```python
logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
cost = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
loss = cost.mean()
```

The first line computes every log probability, the second uses `gather` to pick each row's target entry, and the third averages. The next section explains `logsumexp`.

This loss also has a gradient you can read, from "Softmax and cross-entropy: a loss with a gradient you can read". For one position, the gradient with respect to the logits is the probabilities minus a one-hot vector that is 1 at the true token. At the first position that is $(0.787 - 1,\ 0.107,\ 0.107) = (-0.213,\ 0.107,\ 0.107)$: gradient descent raises the true logit and lowers the others, and the push fades to nothing as $p$ approaches 1. The mean over $N$ positions divides each position's gradient by $N$. The exercise checks this against finite differences.

## Why subtract the largest logit

The softmax needs $e^{z}$ for each logit $z$. A float cannot hold $e^{1000}$, which comes out as infinity, and infinity divided by infinity is not a number. Logits can grow to the hundreds during training, so the plain formula fails exactly when a model is confident.

The fix uses one fact: $e^{z} = e^{m} e^{z - m}$ for any $m$. Pull $e^{m}$ out of the sum, take the log, and choose $m$ to be the largest logit:

$$
\ln \sum_j e^{z_j} = m + \ln \sum_j e^{z_j - m}
$$

Read it aloud: the log of the total equals the largest logit plus the log of a total in which the largest term is $e^0 = 1$. Every term is then between 0 and 1, so nothing overflows. Take $z = [1000, 999]$ and $m = 1000$. The shifted total is $1 + e^{-1} = 1.3679$, its log is 0.3133, and the answer is 1000.3133. That is `torch.logsumexp`. The log probability is then $z - \ln\sum e^{z}$, which is why the code above never forms a probability at all.

## One pass, many lessons

Training does not feed the model one question at a time. It feeds a window of text and asks for a prediction at every position at once, as in Figure 3.

```figure
<svg viewBox="0 0 620 280" role="img" aria-label="Teacher forcing: the text to be or gives inputs to and be, targets be and or, and a two by two causal mask that blocks each position from seeing its own answer">
<defs><marker id="c105-arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
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
<line x1="134" y1="144" x2="134" y2="204" class="stroke-ink" marker-end="url(#c105-arrow2)"/>
<line x1="200" y1="62" x2="200" y2="106" class="stroke-muted"/>
<line x1="280" y1="62" x2="200" y2="206" class="stroke-muted dashed"/>
<line x1="214" y1="144" x2="214" y2="204" class="stroke-ink" marker-end="url(#c105-arrow2)"/>
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
Figure 3. The text “to be or” gives two lessons from one pass. The targets are the inputs moved one place left, so each position predicts the token after it. The causal mask on the right lets position 1 see only “to”, so it cannot copy its answer, “be”, from the next input.
```

The inputs are the tokens from the start of the window to the second to last, and the targets are the same tokens moved one place left, so each target is the token after its input. A window of $T + 1$ tokens gives $T$ lessons from a single forward pass, instead of $T$ separate passes. Feeding the true text, rather than the model's own earlier guesses, is called **teacher forcing**: the teacher supplies the correct previous words, so one early mistake does not wreck every later position.

Why are those $T$ lessons honest? Position 1 of "to be or" has a target "be", and "be" is the input at position 2. Without a mask, position 1 could read its answer from the next input and copy it, and the loss would fall to zero while the model learned nothing. The causal mask removes that path. The output at each position depends only on the tokens up to and including its own. In code, with `block_size` the number of lessons:

```python
inputs = tokens[start : start + block_size]
targets = tokens[start + 1 : start + block_size + 1]
```

Slicing past the end of an array quietly returns something shorter, so a good version checks the window fits.

## An untrained model scores ln V

A fresh model has small random weights, so its logits sit near zero. A softmax of equal numbers gives every word the probability $1/V$, where $V$ is the vocabulary size. The cost of the true word is then $-\ln(1/V) = \ln V$, whichever word is true. That is the third panel of Figure 2: with 11 words, $\ln 11 = 2.398$. With 50,000 words it is 10.82.

This is a check you can run before spending any compute. A first loss close to $\ln V$ suggests that tokens, logits, and loss are wired correctly.

## Perplexity: how many choices

A loss of 2.398 means little on its own. A better unit comes from a model that picks uniformly among $k$ words. It gives the true word $1/k$, so its loss is $\ln k$. Turn that around: $e^{\text{loss}}$ equals $k$. So define the **perplexity** as $e^{L}$, the number of equally likely words that would produce the same loss:

| Model | Loss | Perplexity |
|---|---|---|
| Certain of the right word | 0 | 1 |
| Torn between 2 words | 0.693 | 2 |
| Torn between 4 words | 1.386 | 4 |
| Untrained, 11 words | 2.398 | 11 |

The worked loss of 0.1672 gives $e^{0.1672} = 1.18$: the model behaves as if it were choosing between about 1.2 words. In code it is `math.exp(float(loss))`.

> [!KEY]
> A language model is scored by minus the log of the probability it gave to the true next token, averaged over every position. A confident right answer costs almost nothing (0.99 costs 0.01), and a confident wrong one costs a lot (0.01 costs 4.6), and the cost changes with every small nudge, which accuracy cannot do. An untrained model scores $\ln V$, and perplexity, $e^{L}$, reads the loss as a count of choices.

> [!RECAP]
> - The cost of one position is $-\ln p$, where $p$ is the probability given to the true token. The loss is the mean over positions.
> - Compute log probabilities as logits minus `logsumexp`, so large logits never overflow.
> - Inputs are a window of tokens, and targets are the same window moved one place left. The causal mask keeps each position from seeing its answer.
> - An untrained model scores $\ln V$ and perplexity $V$.
> - Perplexity is $e^{L}$, the number of equally likely choices that would give the same loss.

## Build it step by step

The checks test three functions. They check `per_token_loss` first, then `perplexity`, then `make_lm_batch`. The file also holds `causal_lm_loss`, already written, which is the mean of your `per_token_loss`.

1. **`per_token_loss(logits, targets)`.** Logits have shape `(..., V)` and targets have shape `(...)`. Return the cost of every position, shape `(...)`, with `torch.logsumexp` and `gather`. Keep `dim=-1` and `keepdim=True` on the `logsumexp` so the subtraction lines up. The usual mistake is to build the softmax first, which overflows on large logits.
2. **`perplexity(loss)`.** Return `math.exp(float(loss))`. The usual mistake is to report the loss itself.
3. **`make_lm_batch(tokens, block_size, start)`.** Return the pair `(inputs, targets)`, each of length `block_size`, with targets one place to the left of inputs. Raise `ValueError` when the window would run past the end of `tokens`, because a slice would silently return fewer targets.

## Where this breaks

- A first loss far above $\ln V$ means the logits start too large, from an output matrix with too big an initial scale or a missing normalisation. Fix that before you train.
- A loss that plunges toward 0 within a few steps on real text means the model can see its answer. Check the causal mask and the shift of the targets.
- Perplexity depends on the vocabulary. Two models with different tokenizers have different per-token losses, so their perplexities do not compare directly.

## The exercise

You write `per_token_loss`, `perplexity`, and `make_lm_batch`. The checks match the hand costs of 0.2395 and 0.0949, stay finite on logits of 1000, test the gradient against finite differences, and confirm that uniform logits give $\ln V$ and perplexity $V$. They also check the shifted windows and the error at the end of the data. Run trains a bigram table on a line of text and prints the loss falling from $\ln V$.
