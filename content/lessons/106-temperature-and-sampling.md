---
title: "From scores to a token: temperature and sampling"
slug: 106-temperature-and-sampling
part: "Part 7 — Large language models"
summary: Sharpen or flatten the model's scores with temperature, cut the junk tail with top-k or top-p, then draw one token.
minutes: 40
gpu: false
objectives:
  - Explain why always taking the top score gives repetitive text, and why drawing from the raw probabilities lets junk through.
  - Compute a temperature softmax by hand, and explain why the largest score has to be subtracted first.
  - Build top-k and top-p filtering, and explain why the token that crosses p is kept and why p adapts where k does not.
  - Assemble temperature, filtering, and a seeded draw into one function that picks the next token.
prereqs:
  - 088-scaled-attention-scores
  - 095-next-token-loss
  - 105-decoding-and-the-kv-cache
exercise: 106-temperature-and-sampling
---

# From scores to a token: temperature and sampling

The last layer of a language model hands you one score for every token in its vocabulary. Take a toy vocabulary of five tokens with scores `[0.5, -2.0, 3.5, 3.4, -1.0]`. Token 2 scores 3.5 and token 3 scores 3.4. Which one do you write next? Should it always be token 2, and if not, how do you let the output vary without ever writing a nonsense word?

Always taking the top score looks safe, but it has two costs. The same prompt gives the same text every time. And a gap of 0.1 between the top two scores is a coin flip that greedy decoding settles the same way every time. Drawing at random from the model's own probabilities fixes that, but it opens a different hole. A real vocabulary holds tens of thousands of tokens, and the junk ones each have a tiny probability that adds up. Suppose 10,000 junk tokens each have a probability of 0.00001. Together they have 10 percent, so one draw in ten is nonsense.

This chapter builds the whole step in four moves. You turn scores into probabilities with a temperature that sharpens or flattens them, cut the junk tail with top-k or top-p, and draw one token. The model's job ends at the scores, and everything here decides what to do with them.

## What you already know

- A softmax turns scores into positive numbers that add up to 1, as in "Letting tokens look at each other: scores and softmax weights".
- Subtracting the same number from every score leaves a softmax unchanged, and keeps `np.exp` from overflowing.
- A language model gives a probability to each possible next token, as in "How a language model is scored: next-token loss and perplexity".
- A token you pick is fed back in, so one choice shapes every choice after it.
- Drawing with probabilities $p$ picks token $i$ with probability $p_i$.

## Why not always take the top score?

Greedy decoding returns the index of the largest score, `int(np.argmax(logits))`, which is 2 for our scores. It has no randomness, which is useful when you want the same answer every time. For writing text it fails in two ways. It is dull, because every run of the same prompt is identical. It also tends to loop, because a phrase that was the most likely continuation once looks even more likely after it appears again, so the top choice repeats it.

The alternative is to treat the probabilities as the model's real opinion and sample from them. Subtract the largest score 3.5 to get `[-3.0, -5.5, 0, -0.1, -4.5]`. The exponentials are `[0.0498, 0.0041, 1, 0.9048, 0.0111]`, which add up to 1.9698. Dividing gives probabilities `[0.0253, 0.0021, 0.5077, 0.4594, 0.0056]`. Token 2 gets 51 percent and token 3 gets 46 percent, so both appear. Now you want a dial that says how bold the draw should be.

## Temperature: a dial on commitment

Divide every score by a number $\tau$ (tau) before the softmax. Why does that act as a dial? The softmax exponentiates the scores, and exponentiating magnifies gaps between them. Dividing by $\tau$ shrinks all gaps when $\tau$ is above 1 and stretches them when $\tau$ is below 1. Stretched gaps make the leader dominate. Shrunken gaps make everything look alike.

```figure
<svg viewBox="0 0 600 252" role="img" aria-label="Three bar charts of the same five scores at temperature 0.5, 1, and 2: the junk tail shrinks when cold and grows when hot">
<text x="106" y="20" text-anchor="middle" class="fig-bold">τ = 0.5, sharper</text>
<line x1="14" y1="176" x2="198" y2="176" class="stroke-muted"/>
<line x1="14" y1="138" x2="198" y2="138" class="stroke-muted dashed"/>
<rect x="22" y="175.7" width="28" height="0.3" rx="2" class="fill-muted"/>
<text x="36" y="170.7" text-anchor="middle" class="fig-small">0.001</text>
<text x="36" y="192" text-anchor="middle" class="fig-small">0</text>
<rect x="57" y="176" width="28" height="0" rx="2" class="fill-muted"/>
<text x="71" y="171" text-anchor="middle" class="fig-small">0.000</text>
<text x="71" y="192" text-anchor="middle" class="fig-small">1</text>
<rect x="92" y="71.7" width="28" height="104.3" rx="2" class="fill-b"/>
<text x="106" y="66.7" text-anchor="middle" class="fig-small">0.549</text>
<text x="106" y="192" text-anchor="middle" class="fig-small">2</text>
<rect x="127" y="90.6" width="28" height="85.4" rx="2" class="fill-b"/>
<text x="141" y="85.6" text-anchor="middle" class="fig-small">0.450</text>
<text x="141" y="192" text-anchor="middle" class="fig-small">3</text>
<rect x="162" y="176" width="28" height="0" rx="2" class="fill-muted"/>
<text x="176" y="171" text-anchor="middle" class="fig-small">0.000</text>
<text x="176" y="192" text-anchor="middle" class="fig-small">4</text>
<text x="20" y="134" class="fig-small">0.2</text>
<text x="302" y="20" text-anchor="middle" class="fig-bold">τ = 1</text>
<line x1="210" y1="176" x2="394" y2="176" class="stroke-muted"/>
<line x1="210" y1="138" x2="394" y2="138" class="stroke-muted dashed"/>
<rect x="218" y="171.2" width="28" height="4.8" rx="2" class="fill-muted"/>
<text x="232" y="166.2" text-anchor="middle" class="fig-small">0.025</text>
<text x="232" y="192" text-anchor="middle" class="fig-small">0</text>
<rect x="253" y="175.6" width="28" height="0.4" rx="2" class="fill-muted"/>
<text x="267" y="170.6" text-anchor="middle" class="fig-small">0.002</text>
<text x="267" y="192" text-anchor="middle" class="fig-small">1</text>
<rect x="288" y="79.5" width="28" height="96.5" rx="2" class="fill-b"/>
<text x="302" y="74.5" text-anchor="middle" class="fig-small">0.508</text>
<text x="302" y="192" text-anchor="middle" class="fig-small">2</text>
<rect x="323" y="88.7" width="28" height="87.3" rx="2" class="fill-b"/>
<text x="337" y="83.7" text-anchor="middle" class="fig-small">0.459</text>
<text x="337" y="192" text-anchor="middle" class="fig-small">3</text>
<rect x="358" y="174.9" width="28" height="1.1" rx="2" class="fill-muted"/>
<text x="372" y="169.9" text-anchor="middle" class="fig-small">0.006</text>
<text x="372" y="192" text-anchor="middle" class="fig-small">4</text>
<text x="216" y="134" class="fig-small">0.2</text>
<text x="498" y="20" text-anchor="middle" class="fig-bold">τ = 2, flatter</text>
<line x1="406" y1="176" x2="590" y2="176" class="stroke-muted"/>
<line x1="406" y1="138" x2="590" y2="138" class="stroke-muted dashed"/>
<rect x="414" y="157.9" width="28" height="18.1" rx="2" class="fill-muted"/>
<text x="428" y="152.9" text-anchor="middle" class="fig-small">0.095</text>
<text x="428" y="192" text-anchor="middle" class="fig-small">0</text>
<rect x="449" y="170.8" width="28" height="5.2" rx="2" class="fill-muted"/>
<text x="463" y="165.8" text-anchor="middle" class="fig-small">0.027</text>
<text x="463" y="192" text-anchor="middle" class="fig-small">1</text>
<rect x="484" y="94.9" width="28" height="81.1" rx="2" class="fill-b"/>
<text x="498" y="89.9" text-anchor="middle" class="fig-small">0.427</text>
<text x="498" y="192" text-anchor="middle" class="fig-small">2</text>
<rect x="519" y="98.9" width="28" height="77.1" rx="2" class="fill-b"/>
<text x="533" y="93.9" text-anchor="middle" class="fig-small">0.406</text>
<text x="533" y="192" text-anchor="middle" class="fig-small">3</text>
<rect x="554" y="167.5" width="28" height="8.5" rx="2" class="fill-muted"/>
<text x="568" y="162.5" text-anchor="middle" class="fig-small">0.045</text>
<text x="568" y="192" text-anchor="middle" class="fig-small">4</text>
<text x="412" y="134" class="fig-small">0.2</text>
<text x="300" y="218" text-anchor="middle" class="fig-small">token id under each bar; dashed line is 0.2, where all five tokens would tie</text>
<rect x="150" y="232" width="10" height="10" rx="2" class="fill-b"/>
<text x="166" y="241" class="fig-small">the two leaders</text>
<rect x="300" y="232" width="10" height="10" rx="2" class="fill-muted"/>
<text x="316" y="241" class="fig-small">the junk tail</text>
</svg>
Figure 1. The same scores at three temperatures. Cold (left), the two leaders take almost everything and the tail all but vanishes, yet the near tie between tokens 2 and 3 stays a near tie. Hot (right), the tail grows to 0.167 in total. Temperature reshapes the whole distribution, but a gap of 0.1 only decides the draw at very low temperatures.
```

Figure 1 shows the same five scores at three temperatures. Cold, on the left, the junk tokens 0, 1, and 4 together get only 0.0014. Hot, on the right, they get 0.1675. Notice also what does not change. Tokens 2 and 3 stay close in all three panels, because a gap of 0.1 only matters once $\tau$ is about as small as the gap. Their ratio is $e^{0.1/\tau}$, which is 1.22 at $\tau = 0.5$ and would reach $e^{1} = 2.7$ only at $\tau = 0.1$. Temperature mainly controls how much weight the tail gets, and only a very cold setting turns a near tie into a clear win.

For $\tau = 2$ the work is short. Dividing the scores by 2 gives `[0.25, -1, 1.75, 1.7, -0.5]`. Subtract the largest, 1.75, to get `[-1.5, -2.75, 0, -0.05, -2.25]`. The exponentials are `[0.2231, 0.0639, 1, 0.9512, 0.1054]`, which add up to 2.3437, so the probabilities are `[0.0952, 0.0273, 0.4267, 0.4059, 0.0450]`.

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

Read it aloud: divide each score by $\tau$, exponentiate, and take its share of the total. The two ends of the dial are worth knowing. As $\tau$ shrinks toward 0, the leader's share goes to 1, which is greedy decoding. As $\tau$ grows, every share goes to $1/5 = 0.2$, which is a uniform draw. Because dividing by zero fails, the code defines $\tau = 0$ to mean greedy. In numpy:

```python
s = logits / temperature
probs = np.exp(s - s.max())
probs = probs / probs.sum()
```

> [!WARNING]
> Keep the subtraction of `s.max()`. At $\tau = 0.001$ the largest scaled score is 3500, and `np.exp(3500)` overflows to `inf`, which turns the result into `nan`.

## The tail problem and the first filter

Temperature shrinks the tail but never removes it. At any temperature above 0 every junk token keeps some probability, and a long tail of them still adds up. A filter removes the tail for good: keep the tokens that are plausible, set the rest to zero, and divide by what remains so the probabilities add up to 1 again.

The simplest filter is top-k. It keeps the $k$ largest probabilities. At $\tau = 2$ with $k = 2$, keep tokens 2 and 3, which hold 0.4267 and 0.4059. They total 0.8326, so after dividing, token 2 gets 0.5125, token 3 gets 0.4875, and tokens 0, 1, and 4 get 0.

Why not just use top-k everywhere? Because $k$ is a headcount that ignores the shape of the distribution. Figure 2 shows two distributions side by side.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Two bar charts of ten sorted probabilities, one with a single dominant token and one nearly flat, with the tokens kept by top-k 3 and top-p 0.9 marked">
<text x="154" y="20" text-anchor="middle" class="fig-bold">Confident: one clear answer</text>
<line x1="14" y1="130" x2="294" y2="130" class="stroke-muted"/>
<rect x="18" y="40" width="22" height="90" rx="2" class="fill-b"/>
<text x="29" y="144" text-anchor="middle" class="fig-small">.95</text>
<rect x="46" y="128.1" width="22" height="1.9" rx="2" class="fill-muted"/>
<text x="57" y="144" text-anchor="middle" class="fig-small">.02</text>
<rect x="74" y="129.1" width="22" height="0.9" rx="2" class="fill-muted"/>
<text x="85" y="144" text-anchor="middle" class="fig-small">.01</text>
<rect x="102" y="129.3" width="22" height="0.7" rx="2" class="fill-muted"/>
<text x="113" y="144" text-anchor="middle" class="fig-small">.007</text>
<rect x="130" y="129.6" width="22" height="0.4" rx="2" class="fill-muted"/>
<text x="141" y="144" text-anchor="middle" class="fig-small">.004</text>
<rect x="158" y="129.7" width="22" height="0.3" rx="2" class="fill-muted"/>
<text x="169" y="144" text-anchor="middle" class="fig-small">.003</text>
<rect x="186" y="129.8" width="22" height="0.2" rx="2" class="fill-muted"/>
<text x="197" y="144" text-anchor="middle" class="fig-small">.002</text>
<rect x="214" y="129.8" width="22" height="0.2" rx="2" class="fill-muted"/>
<text x="225" y="144" text-anchor="middle" class="fig-small">.002</text>
<rect x="242" y="129.9" width="22" height="0.1" rx="2" class="fill-muted"/>
<text x="253" y="144" text-anchor="middle" class="fig-small">.001</text>
<rect x="270" y="129.9" width="22" height="0.1" rx="2" class="fill-muted"/>
<text x="281" y="144" text-anchor="middle" class="fig-small">.001</text>
<line x1="18" y1="160" x2="96" y2="160" class="stroke-d" stroke-width="3"/>
<text x="18" y="178" class="fig-small">top-k = 3 keeps 0.98 of the mass</text>
<line x1="18" y1="192" x2="40" y2="192" class="stroke-bad" stroke-width="3"/>
<text x="18" y="210" class="fig-small">top-p = 0.9 keeps 1 token (0.95)</text>
<text x="454" y="20" text-anchor="middle" class="fig-bold">Unsure: many fair answers</text>
<line x1="314" y1="130" x2="594" y2="130" class="stroke-muted"/>
<rect x="318" y="40" width="22" height="90" rx="2" class="fill-b"/>
<text x="329" y="144" text-anchor="middle" class="fig-small">.17</text>
<rect x="346" y="50.6" width="22" height="79.4" rx="2" class="fill-b"/>
<text x="357" y="144" text-anchor="middle" class="fig-small">.15</text>
<rect x="374" y="55.9" width="22" height="74.1" rx="2" class="fill-b"/>
<text x="385" y="144" text-anchor="middle" class="fig-small">.14</text>
<rect x="402" y="66.5" width="22" height="63.5" rx="2" class="fill-b"/>
<text x="413" y="144" text-anchor="middle" class="fig-small">.12</text>
<rect x="430" y="77.1" width="22" height="52.9" rx="2" class="fill-b"/>
<text x="441" y="144" text-anchor="middle" class="fig-small">.10</text>
<rect x="458" y="82.4" width="22" height="47.6" rx="2" class="fill-b"/>
<text x="469" y="144" text-anchor="middle" class="fig-small">.09</text>
<rect x="486" y="87.6" width="22" height="42.4" rx="2" class="fill-b"/>
<text x="497" y="144" text-anchor="middle" class="fig-small">.08</text>
<rect x="514" y="98.2" width="22" height="31.8" rx="2" class="fill-b"/>
<text x="525" y="144" text-anchor="middle" class="fig-small">.06</text>
<rect x="542" y="103.5" width="22" height="26.5" rx="2" class="fill-muted"/>
<text x="553" y="144" text-anchor="middle" class="fig-small">.05</text>
<rect x="570" y="108.8" width="22" height="21.2" rx="2" class="fill-muted"/>
<text x="581" y="144" text-anchor="middle" class="fig-small">.04</text>
<line x1="318" y1="160" x2="396" y2="160" class="stroke-d" stroke-width="3"/>
<text x="318" y="178" class="fig-small">top-k = 3 keeps 0.46 of the mass</text>
<line x1="318" y1="192" x2="536" y2="192" class="stroke-bad" stroke-width="3"/>
<text x="318" y="210" class="fig-small">top-p = 0.9 keeps 8 tokens (0.91)</text>
<text x="300" y="252" text-anchor="middle" class="fig-small">Sorted probabilities of ten tokens. Orange bars are the ones top-p keeps. Each panel is scaled so its tallest bar fills the same height.</text>
</svg>
Figure 2. A fixed k does not look at the shape. On the confident distribution it keeps two extra tokens that add 3 percent. On the unsure distribution it keeps three tokens and drops more than half the mass. A fixed p keeps 1 token on the first and 8 on the second.
```

When the model is confident, $k = 3$ keeps two tokens it barely believes in. When the model is unsure, $k = 3$ cuts away more than half the mass, and many fair answers vanish. We want the number of survivors to adapt.

## Top-p: keep the smallest set that reaches p

Top-p, also called nucleus sampling, sorts the tokens from most to least likely and keeps the shortest prefix whose total reaches $p$. Confident distributions reach $p$ after a token or two. Flat ones need many. Take $\tau = 2$ and $p = 0.9$. Sorted, the probabilities are 0.4267 (token 2), 0.4059 (token 3), 0.0952 (token 0), 0.0450 (token 4), and 0.0273 (token 1). The running totals are 0.4267, 0.8325, 0.9278, 0.9727, and 1.0.

```figure
<svg viewBox="0 0 620 264" role="img" aria-label="Sorted bars of the temperature 2 probabilities with a running-total line crossing 0.9 at the third bar">
<line x1="70" y1="190" x2="510" y2="190" class="stroke-grid"/>
<text x="62" y="194" text-anchor="end" class="fig-small">0.0</text>
<line x1="70" y1="115" x2="510" y2="115" class="stroke-grid"/>
<text x="62" y="119" text-anchor="end" class="fig-small">0.5</text>
<line x1="70" y1="40" x2="510" y2="40" class="stroke-grid"/>
<text x="62" y="44" text-anchor="end" class="fig-small">1.0</text>
<rect x="90" y="126" width="48" height="64" rx="2" class="fill-b"/>
<text x="114" y="206" text-anchor="middle" class="fig-small">token 2</text>
<text x="114" y="222" text-anchor="middle" class="fig-small">0.4267</text>
<rect x="174" y="129.1" width="48" height="60.9" rx="2" class="fill-b"/>
<text x="198" y="206" text-anchor="middle" class="fig-small">token 3</text>
<text x="198" y="222" text-anchor="middle" class="fig-small">0.4059</text>
<rect x="258" y="175.7" width="48" height="14.3" rx="2" class="fill-b"/>
<text x="282" y="206" text-anchor="middle" class="fig-small">token 0</text>
<text x="282" y="222" text-anchor="middle" class="fig-small">0.0952</text>
<rect x="342" y="183.3" width="48" height="6.7" rx="2" class="fill-muted"/>
<text x="366" y="206" text-anchor="middle" class="fig-small">token 4</text>
<text x="366" y="222" text-anchor="middle" class="fig-small">0.0450</text>
<rect x="426" y="185.9" width="48" height="4.1" rx="2" class="fill-muted"/>
<text x="450" y="206" text-anchor="middle" class="fig-small">token 1</text>
<text x="450" y="222" text-anchor="middle" class="fig-small">0.0273</text>
<polyline points="114,126 198,65.1 282,50.8 366,44.1 450,40" fill="none" class="stroke-d" stroke-width="2"/>
<circle cx="114" cy="126" r="4" class="fill-d"/>
<text x="120" y="114" text-anchor="end" class="fig-small fig-bold">0.4267</text>
<circle cx="198" cy="65.1" r="4" class="fill-d"/>
<text x="206" y="83.1" class="fig-small fig-bold">0.8325</text>
<circle cx="282" cy="50.8" r="4" class="fill-d"/>
<text x="282" y="41.8" text-anchor="middle" class="fig-small fig-bold">0.9278</text>
<circle cx="366" cy="44.1" r="4" class="fill-d"/>
<text x="366" y="35.1" text-anchor="middle" class="fig-small fig-bold">0.9727</text>
<circle cx="450" cy="40" r="4" class="fill-d"/>
<text x="450" y="31" text-anchor="middle" class="fig-small fig-bold">1.0000</text>
<line x1="70" y1="55" x2="510" y2="55" class="stroke-bad dashed"/>
<text x="514" y="59" class="fig-small">p = 0.9</text>
<line x1="324" y1="16" x2="324" y2="194" class="stroke-muted dashed"/>
<text x="318" y="14" text-anchor="end" class="fig-small">kept</text>
<text x="330" y="14" class="fig-small">dropped</text>
<text x="14" y="100" class="fig-small">mass</text>
<text x="300" y="254" text-anchor="middle" class="fig-small">bars: probability of each token, sorted. line: running total of the bars.</text>
</svg>
Figure 3. Sort the probabilities, then add them from the top. The running total first reaches 0.9 at the third token (0.9278), so that token is kept and everything after it is dropped. Tokens 0, 2, and 3 survive.
```

Figure 3 draws the sort and the running total.

The total first reaches 0.9 at the third token, so you keep tokens 2, 3, and 0 and drop the rest. Dividing by 0.9278 gives `[0.1026, 0, 0.4599, 0.4375, 0]`. The token that crosses $p$ must be kept. If you stopped just before it, the kept mass would be below $p$, and when the leader alone is already above $p$, as with $p = 0.3$ here, you would keep nothing at all. In numpy this is `np.argsort(-probs)`, then `np.cumsum`, then `np.searchsorted(cumulative, p, side="left")`, which returns the position of the first total that reaches $p$.

There is one trap. Rounding can leave a running total just under the target. Ten tokens of probability 0.1 add to 0.9999999999999999, so `p = 1.0` makes `np.searchsorted` return 10, one past the last index. Clamp the position to `len(probs) - 1`.

> [!KEY]
> Greedy decoding is dull, and drawing from the raw probabilities lets the junk tail through. Temperature controls how much weight the tail gets, and a filter removes it. Top-p keeps the smallest set that reaches $p$, so the number of survivors adapts to how sure the model is, which a fixed $k$ cannot do.

## The whole step, in order

Put the pieces in a fixed order. Start from the logits. Optionally apply a repetition penalty. Divide by the temperature and take the softmax. Apply top-k, then top-p on the renormalised result. Draw one token with `rng.choice`, using the generator you were given so that a seed reproduces the text.

The repetition penalty addresses the looping from earlier. For every token already written, divide its score by $r$ if the score is positive and multiply it by $r$ if it is negative. With $r = 1.3$, token 2's score 3.5 becomes $3.5 / 1.3 = 2.6923$, and a score of $-2.0$ becomes $-2.6$. Both move down. Dividing a negative score by 1.3 would move it up toward zero, which would reward the repeat. You can add it as a step before the temperature. It is not graded here.

```python
probs = apply_temperature(logits, temperature)   # sharpen or flatten
probs = filter_probs(probs, top_k, top_p)        # cut the tail
token = int(rng.choice(len(probs), p=probs))     # one draw
```

> [!RECAP]
> Before you build:
> - Temperature divides the logits by $\tau$ before a stable softmax. A temperature of 0 means greedy decoding.
> - Top-k keeps the $k$ largest probabilities and renormalises.
> - Top-p keeps the smallest sorted prefix whose total reaches $p$, including the token that crosses it, and clamps the cutoff to the last index.
> - Apply top-k first and top-p second, then draw with the generator you were given.

## Build it step by step

1. **`apply_temperature(logits, temperature)`.** It takes a vector of scores and a positive number, and returns a vector of probabilities of the same length. Divide, subtract the maximum, exponentiate, and divide by the sum. Check it against the hand numbers above: temperature 2 gives `[0.0952, 0.0273, 0.4267, 0.4059, 0.0450]`. The usual mistake is to skip the maximum.
2. **`filter_probs(probs, top_k=None, top_p=None)`.** It returns a copy and never changes its input. For top-k, find the $k$ largest with `np.argsort(-probs)`, zero the others, and divide by the kept total. For top-p, sort the same way, take `np.cumsum`, find the cutoff with `np.searchsorted`, clamp it, keep through it, and renormalise. Run top-k first. The usual mistake is to stop before the crossing token.
3. **`sample_token(logits, rng, temperature=1.0, top_k=None, top_p=None)`.** If the temperature is 0 return `int(np.argmax(logits))`. Otherwise chain `apply_temperature`, `filter_probs`, and `rng.choice(len(probs), p=probs)`. Return a plain `int`. The usual mistake is to call `np.random.choice`, which ignores the seed you were handed.

## Where this breaks

None of these dials repairs a model that is wrong. A low temperature or a small $p$ pulls the output toward greedy, and loops come back. A high temperature with no filter lets the tail through. The settings also interact, because the temperature runs first: a hotter distribution is flatter, so the same $p$ keeps more tokens. A repetition penalty that is too strong pushes the model away from words it needs to repeat, such as a name.

## The exercise

You write `apply_temperature`, `filter_probs`, and `sample_token`. The checks match the hand-worked probabilities at temperatures 0.5, 1, and 2, stay finite at 0.001, and confirm that a hotter temperature lowers the top probability and raises the tail. They check top-k on `[0.4, 0.3, 0.2, 0.1]`, top-p on an unsorted vector, the crossing token, and `p = 1.0` on ten equal probabilities. They check that `sample_token` at temperature 0 is greedy, never returns a token the filter removed, and, over 10,000 seeded draws, matches the filtered distribution.
