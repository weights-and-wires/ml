---
title: "Speculative decoding: guess fast, verify once"
slug: 107-speculative-decoding
part: "Part 7 — Large language models"
summary: A small model drafts several tokens, the large model checks them in one pass, and an accept-or-replace rule keeps the output exactly the large model's.
minutes: 40
gpu: false
objectives:
  - Explain why a large model can check several drafted tokens for about the cost of writing one.
  - Compute the acceptance chance and the replacement distribution for a drafted token by hand.
  - Show with p = [0.7, 0.2, 0.1] and q = [0.1, 0.1, 0.8] that accept-or-replace gives exactly p.
  - Estimate the expected tokens per target pass from the acceptance rate and the number of drafted tokens.
prereqs:
  - 105-decoding-and-the-kv-cache
  - 106-temperature-and-sampling
exercise: 107-speculative-decoding
---

# Speculative decoding: guess fast, verify once

The big model spends most of every step waiting. Reading its weights takes 7 milliseconds, and the arithmetic it does with them takes 0.045 milliseconds, so the chip is idle more than 99 percent of the time. Suppose a much smaller model, one that costs a tenth as much per step, could guess the next few tokens. Could the big model then check all the guesses at once, for the price of one step? And if it did, would the final text still be what the big model alone would have written, or only something close?

Both parts have a good answer, and the second one is surprising. The check is cheap because the big model is memory bound, so reading three extra positions rides along with a read it was making anyway. The text stays exactly the big model's because of a small rule for deciding which guesses to keep. This chapter builds the story first, then the rule, then a proof by hand, and last the speedup you can expect.

## What you already know

- Writing one token waits on reading the weights from memory, as in "Why decoding waits on memory, and the KV cache".
- A pass over several tokens at once costs almost the same as a pass over one while the chip is memory bound.
- A causal mask lets one pass compute the output at every position of a sequence, as in "Many heads and the causal mask".
- A model gives a probability distribution over the next token, and sampling draws from it, as in "From scores to a token: temperature and sampling".
- A draw with probabilities $q$ picks token $x$ with probability $q(x)$.

## The picture: a student drafts, a teacher checks

Call the big model the target, with next-token distribution $p$, and the small model the draft, with distribution $q$. One round goes like this. The draft writes $k$ tokens one after another, say $k = 3$. Then the target reads the last token written plus all three drafted tokens in a single pass. Earlier tokens are already in its cache from "The KV cache: remember, don't recompute", so only these four positions are new. Thanks to the causal mask, that one pass gives the target's own distribution at every position: after the last written token, after the first draft token, after the second, and after the third. Each of those distributions is trustworthy only if every draft token before it is kept, because it was computed assuming the draft text came before it. That is why the check runs left to right and stops at the first rejection.

```figure
<svg viewBox="0 0 600 244" role="img" aria-label="Timeline: the target alone writes five tokens in five long steps; draft-then-verify writes three tokens in about 1.3 target steps">
<defs><marker id="c120-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="60" y="36" class="fig-bold">Target model alone: one slow step per token</text>
<rect x="60" y="46" width="96" height="38" rx="2" class="fill-soft-a stroke-a"/>
<text x="70" y="70">the</text>
<rect x="160" y="46" width="96" height="38" rx="2" class="fill-soft-a stroke-a"/>
<text x="164" y="70">cat</text>
<rect x="260" y="46" width="96" height="38" rx="2" class="fill-soft-a stroke-a"/>
<text x="270" y="70">slept</text>
<rect x="360" y="46" width="96" height="38" rx="2" class="fill-soft-a stroke-a"/>
<text x="370" y="70">on</text>
<rect x="460" y="46" width="96" height="38" rx="2" class="fill-soft-a stroke-a"/>
<text x="470" y="70">the</text>
<text x="60" y="118" class="fig-bold">Draft three tokens, then check them in one target pass</text>
<rect x="60" y="134" width="8" height="24" rx="2" class="fill-soft-d stroke-d"/>
<rect x="70" y="134" width="8" height="24" rx="2" class="fill-soft-d stroke-d"/>
<rect x="80" y="134" width="8" height="24" rx="2" class="fill-soft-d stroke-d"/>
<text x="74" y="176" text-anchor="middle" class="fig-small">draft</text>
<rect x="90" y="126" width="100" height="38" rx="2" class="fill-soft-b stroke-b"/>
<text x="140" y="142" text-anchor="middle" class="fig-small">target checks</text>
<text x="140" y="157" text-anchor="middle" class="fig-small">all three</text>
<line x1="204" y1="144" x2="238" y2="144" class="stroke-muted" marker-end="url(#c120-arrow)"/>
<rect x="244" y="128" width="40" height="34" rx="2" class="fill-c"/>
<text x="264" y="150" text-anchor="middle" class="fig-small fig-bold">the</text>
<rect x="288" y="128" width="40" height="34" rx="2" class="fill-c"/>
<text x="308" y="150" text-anchor="middle" class="fig-small fig-bold">cat</text>
<rect x="332" y="128" width="40" height="34" rx="2" class="fill-b"/>
<text x="352" y="150" text-anchor="middle" class="fig-small fig-bold">slept</text>
<text x="244" y="182" class="fig-small">kept, kept, replaced: 3 tokens</text>
<line x1="190" y1="48" x2="190" y2="198" class="stroke-muted dashed"/>
<text x="196" y="210" class="fig-small">At this moment the target alone has written about 1.3 tokens</text>
<text x="60" y="232" class="fig-small">Assumes one draft step costs a tenth of one target step.</text>
</svg>
Figure 1. Top, the target writes one token per slow step. Bottom, three cheap draft steps and one target pass produce three tokens: two guesses accepted and the third replaced by the target's own pick. The dashed line marks the same moment on both rows.
```

Why is that pass cheap? Four tokens share one read of the weights, so the arithmetic intensity is 4 flops per byte instead of 1. The ridge point of the chip is 156, so four tokens cost the same time as one. Now compare the two models position by position, from left to right. Keep each draft token the target agrees with. At the first one it rejects, write the target's own pick instead, and throw away every draft token after it, because those were written after a token that no longer exists. If all three are kept, the pass hands over a bonus token for free, from the last position. So a round always gives at least one token and at most $k + 1$.

## Why not keep whatever the draft says?

"Agrees with" needs a careful definition, because the obvious ones are wrong. If you accept every draft token, the output follows $q$, the small model's opinion, and you have only built a slower small model. If you accept a draft token only when it is also the target's most likely token, you get the right answer for greedy decoding. For sampling it is too strict: a token the target would draw 20 percent of the time is never accepted from the draft, so the mix of tokens generally no longer follows $p$, and the variety that temperature and top-p provided is distorted. What we want is stronger: the tokens that come out must be distributed exactly as if we had sampled from $p$ directly.

## The rule, by hand

Take a target $p = [0.7, 0.2, 0.1]$ and a draft $q = [0.1, 0.1, 0.8]$. The target likes token 0 and the draft likes token 2, so this is a deliberately bad draft. The draft proposes token $x$ with probability $q(x)$. We accept it with probability $\min(1, p(x)/q(x))$. Follow each case:

- $x = 0$ comes up with probability 0.1. The ratio $p/q$ is $0.7/0.1 = 7$, so it is accepted for sure. It adds 0.1 to the output for token 0.
- $x = 1$ comes up with probability 0.1. The ratio is 2, so it is accepted for sure. It adds 0.1 to token 1.
- $x = 2$ comes up with probability 0.8. The ratio is $0.1/0.8 = 0.125$, so it is accepted 12.5 percent of the time. That adds $0.8 \times 0.125 = 0.1$ to token 2, and the other $0.8 \times 0.875 = 0.7$ is rejected.

The accepted mass is $0.1 + 0.1 + 0.1 = 0.3$, so 70 percent of rounds end in a rejection. Rejected rounds need a replacement from somewhere. We take it from what the target wants and the draft did not supply: $\max(p - q, 0) = [0.6, 0.1, 0]$. It adds up to 0.7, and dividing by 0.7 gives the replacement distribution $[0.857, 0.143, 0]$. Multiply by the rejection chance 0.7 to get $[0.6, 0.1, 0]$. Now add the two sources, accepted and replaced:

$$
[0.1,\ 0.1,\ 0.1] + [0.6,\ 0.1,\ 0] = [0.7,\ 0.2,\ 0.1] = p
$$

```figure
<svg viewBox="0 0 600 312" role="img" aria-label="For each of three tokens, a pair of bars for the target p and the draft q with their overlap in green, the target's leftover and the draft's excess">
<line x1="30" y1="210" x2="450" y2="210" class="stroke-muted"/>
<rect x="54" y="190" width="42" height="20" rx="0" class="fill-c"/>
<rect x="54" y="70" width="42" height="120" rx="0" class="fill-soft-b stroke-b"/>
<rect x="104" y="190" width="42" height="20" rx="0" class="fill-c"/>
<text x="75" y="64" text-anchor="middle" class="fig-bold">0.7</text>
<text x="125" y="184" text-anchor="middle" class="fig-bold">0.1</text>
<text x="75" y="226" text-anchor="middle" class="fig-small">p</text>
<text x="125" y="226" text-anchor="middle" class="fig-small">q</text>
<text x="100" y="244" text-anchor="middle" class="fig-bold">token 0</text>
<text x="100" y="262" text-anchor="middle" class="fig-small">min 0.1, left 0.6</text>
<rect x="194" y="190" width="42" height="20" rx="0" class="fill-c"/>
<rect x="194" y="170" width="42" height="20" rx="0" class="fill-soft-b stroke-b"/>
<rect x="244" y="190" width="42" height="20" rx="0" class="fill-c"/>
<text x="215" y="164" text-anchor="middle" class="fig-bold">0.2</text>
<text x="265" y="184" text-anchor="middle" class="fig-bold">0.1</text>
<text x="215" y="226" text-anchor="middle" class="fig-small">p</text>
<text x="265" y="226" text-anchor="middle" class="fig-small">q</text>
<text x="240" y="244" text-anchor="middle" class="fig-bold">token 1</text>
<text x="240" y="262" text-anchor="middle" class="fig-small">min 0.1, left 0.1</text>
<rect x="334" y="190" width="42" height="20" rx="0" class="fill-c"/>
<rect x="384" y="190" width="42" height="20" rx="0" class="fill-c"/>
<rect x="384" y="50" width="42" height="140" rx="0" class="fill-sunk stroke-muted dashed"/>
<text x="355" y="184" text-anchor="middle" class="fig-bold">0.1</text>
<text x="405" y="44" text-anchor="middle" class="fig-bold">0.8</text>
<text x="355" y="226" text-anchor="middle" class="fig-small">p</text>
<text x="405" y="226" text-anchor="middle" class="fig-small">q</text>
<text x="380" y="244" text-anchor="middle" class="fig-bold">token 2</text>
<text x="380" y="262" text-anchor="middle" class="fig-small">min 0.1, left 0.0</text>
<rect x="462" y="40" width="14" height="14" rx="0" class="fill-c"/>
<text x="482" y="52" class="fig-small">overlap: min(p, q)</text>
<rect x="462" y="66" width="14" height="14" rx="0" class="fill-soft-b stroke-b"/>
<text x="482" y="78" class="fig-small">target leftover</text>
<rect x="462" y="92" width="14" height="14" rx="0" class="fill-sunk stroke-muted dashed"/>
<text x="482" y="104" class="fig-small">draft excess</text>
<text x="300" y="284" text-anchor="middle" class="fig-small">Overlap: 0.1 + 0.1 + 0.1 = 0.3, the acceptance rate.</text>
<text x="300" y="302" text-anchor="middle" class="fig-small">Target leftover: 0.6 + 0.1 + 0 = 0.7, the rejection chance.</text>
</svg>
Figure 2. Green is the part of each token that both models agree on, and it adds up to 0.3. The orange leftover is what the target wants and the draft did not supply, and it adds up to 0.7. Green plus orange gives p for every token.
```

The output is exactly $p$, even though the draft was badly wrong. Figure 2 shows why. For every token, the green overlap $\min(p, q)$ is the part the draft supplies and the target accepts, and the orange leftover $\max(p - q, 0)$ is the part the replacement supplies.

## Why it always works

The hand example is an instance of four steps that hold for any $p$ and $q$.

1. The draft proposes $x$ with probability $q(x)$ and it is accepted with probability $\min(1, p(x)/q(x))$, so $x$ comes out by acceptance with probability $q(x) \cdot \min(1, p(x)/q(x)) = \min(p(x), q(x))$.
2. The target still wants $p(x) - \min(p(x), q(x)) = \max(p(x) - q(x), 0)$ more of $x$.
3. The chance of rejection is $1 - \sum_x \min(p(x), q(x))$. Since $p = \min(p, q) + \max(p - q, 0)$ for every token, summing over all tokens gives $1 = \sum \min(p, q) + \sum \max(p - q, 0)$, so the rejection chance equals $\sum_x \max(p(x) - q(x), 0)$. That is exactly the total of the leftover, which is what we divide by to normalise it.
4. So rejection times the normalised leftover is $\max(p(x) - q(x), 0)$ with nothing left over, and adding it to step 1 gives $p(x)$.

In code, the rule is a few lines. Use the generator you are given for both draws:

```python
accept = rng.uniform() < min(1.0, p[x] / q[x])
leftover = np.maximum(p - q, 0.0)
replacement = leftover / leftover.sum()
```

Two conditions keep it safe. The draft token was drawn from $q$, so $q(x) > 0$ and the ratio is defined. And if $p$ equals $q$ the leftover is all zeros, but then no token is ever rejected, so you only need a fallback, such as returning $p$, so that you never divide by zero. The overlap $\sum \min(p, q)$ is the acceptance rate $\alpha$, which is 0.3 here.

## How much faster?

Suppose each draft token is accepted with the same chance $\alpha$, independent of the others. A round always yields one token. It yields a second only if the first draft token was kept, which has probability $\alpha$, a third with probability $\alpha^2$, and so on up to $\alpha^k$. Add them up:

$$
\text{tokens per pass} = 1 + \alpha + \alpha^2 + \dots + \alpha^k = \frac{1 - \alpha^{k+1}}{1 - \alpha}
$$

Read it aloud: one guaranteed token, plus the chance of each further one surviving. As $\alpha$ approaches 1 it approaches $k + 1$, and at exactly $\alpha = 1$ every term is 1, so the sum is $k + 1$ without the fraction. In numpy, it is `(1 - alpha**(k + 1)) / (1 - alpha)`. With $k = 3$ and $\alpha = 0.8$ you get $1 + 0.8 + 0.64 + 0.512 = 2.952$. With $\alpha = 0.3$ you get 1.417. Figure 3 plots the same formula for $k = 2$, $4$, and $8$.

```figure
<svg viewBox="0 0 620 296" role="img" aria-label="Expected tokens per target pass against acceptance rate for k equal to 2, 4, and 8: starting at 1 and bending up toward k plus 1">
<line x1="70" y1="252" x2="560" y2="252" class="stroke-grid"/>
<text x="62" y="256" text-anchor="end" class="fig-small">1</text>
<line x1="70" y1="224" x2="560" y2="224" class="stroke-grid"/>
<text x="62" y="228" text-anchor="end" class="fig-small">2</text>
<line x1="70" y1="196" x2="560" y2="196" class="stroke-grid"/>
<text x="62" y="200" text-anchor="end" class="fig-small">3</text>
<line x1="70" y1="168" x2="560" y2="168" class="stroke-grid"/>
<text x="62" y="172" text-anchor="end" class="fig-small">4</text>
<line x1="70" y1="140" x2="560" y2="140" class="stroke-grid"/>
<text x="62" y="144" text-anchor="end" class="fig-small">5</text>
<line x1="70" y1="112" x2="560" y2="112" class="stroke-grid"/>
<text x="62" y="116" text-anchor="end" class="fig-small">6</text>
<line x1="70" y1="84" x2="560" y2="84" class="stroke-grid"/>
<text x="62" y="88" text-anchor="end" class="fig-small">7</text>
<line x1="70" y1="56" x2="560" y2="56" class="stroke-grid"/>
<text x="62" y="60" text-anchor="end" class="fig-small">8</text>
<line x1="70" y1="28" x2="560" y2="28" class="stroke-grid"/>
<text x="62" y="32" text-anchor="end" class="fig-small">9</text>
<text x="70" y="268" text-anchor="middle" class="fig-small">0.0</text>
<text x="168" y="268" text-anchor="middle" class="fig-small">0.2</text>
<text x="266" y="268" text-anchor="middle" class="fig-small">0.4</text>
<text x="364" y="268" text-anchor="middle" class="fig-small">0.6</text>
<text x="462" y="268" text-anchor="middle" class="fig-small">0.8</text>
<text x="560" y="268" text-anchor="middle" class="fig-small">1.0</text>
<line x1="70" y1="252" x2="560" y2="252" class="stroke-muted"/>
<polyline points="70,252 74.9,251.7 79.8,251.4 84.7,251.1 89.6,250.8 94.5,250.5 99.4,250.2 104.3,249.9 109.2,249.6 114.1,249.3 119,248.9 123.9,248.6 128.8,248.2 133.7,247.9 138.6,247.5 143.5,247.2 148.4,246.8 153.3,246.4 158.2,246.1 163.1,245.7 168,245.3 172.9,244.9 177.8,244.5 182.7,244.1 187.6,243.7 192.5,243.2 197.4,242.8 202.3,242.4 207.2,242 212.1,241.5 217,241.1 221.9,240.6 226.8,240.2 231.7,239.7 236.6,239.2 241.5,238.8 246.4,238.3 251.3,237.8 256.2,237.3 261.1,236.8 266,236.3 270.9,235.8 275.8,235.3 280.7,234.8 285.6,234.3 290.5,233.7 295.4,233.2 300.3,232.7 305.2,232.1 310.1,231.6 315,231 319.9,230.4 324.8,229.9 329.7,229.3 334.6,228.7 339.5,228.1 344.4,227.5 349.3,226.9 354.2,226.3 359.1,225.7 364,225.1 368.9,224.5 373.8,223.9 378.7,223.2 383.6,222.6 388.5,222 393.4,221.3 398.3,220.7 403.2,220 408.1,219.3 413,218.7 417.9,218 422.8,217.3 427.7,216.6 432.6,215.9 437.5,215.2 442.4,214.5 447.3,213.8 452.2,213.1 457.1,212.4 462,211.7 466.9,210.9 471.8,210.2 476.7,209.5 481.6,208.7 486.5,208 491.4,207.2 496.3,206.4 501.2,205.7 506.1,204.9 511,204.1 515.9,203.3 520.8,202.5 525.7,201.7 530.6,200.9 535.5,200.1 540.4,199.3 545.3,198.5 550.2,197.7 555.1,196.8 560,196" fill="none" class="stroke-a" stroke-width="3"/>
<text x="566" y="200" class="fig-small fig-bold">k = 2</text>
<polyline points="70,252 74.9,251.7 79.8,251.4 84.7,251.1 89.6,250.8 94.5,250.5 99.4,250.2 104.3,249.9 109.2,249.6 114.1,249.2 119,248.9 123.9,248.5 128.8,248.2 133.7,247.8 138.6,247.4 143.5,247.1 148.4,246.7 153.3,246.3 158.2,245.9 163.1,245.4 168,245 172.9,244.6 177.8,244.1 182.7,243.7 187.6,243.2 192.5,242.7 197.4,242.2 202.3,241.7 207.2,241.2 212.1,240.6 217,240.1 221.9,239.5 226.8,239 231.7,238.4 236.6,237.8 241.5,237.1 246.4,236.5 251.3,235.9 256.2,235.2 261.1,234.5 266,233.8 270.9,233.1 275.8,232.4 280.7,231.6 285.6,230.8 290.5,230 295.4,229.2 300.3,228.4 305.2,227.5 310.1,226.6 315,225.8 319.9,224.8 324.8,223.9 329.7,222.9 334.6,221.9 339.5,220.9 344.4,219.9 349.3,218.8 354.2,217.7 359.1,216.6 364,215.4 368.9,214.3 373.8,213.1 378.7,211.8 383.6,210.6 388.5,209.3 393.4,208 398.3,206.6 403.2,205.2 408.1,203.8 413,202.4 417.9,200.9 422.8,199.3 427.7,197.8 432.6,196.2 437.5,194.6 442.4,192.9 447.3,191.2 452.2,189.5 457.1,187.7 462,185.9 466.9,184 471.8,182.1 476.7,180.2 481.6,178.2 486.5,176.2 491.4,174.1 496.3,172 501.2,169.8 506.1,167.6 511,165.3 515.9,163 520.8,160.7 525.7,158.3 530.6,155.8 535.5,153.3 540.4,150.8 545.3,148.2 550.2,145.5 555.1,142.8 560,140" fill="none" class="stroke-b" stroke-width="3"/>
<text x="566" y="144" class="fig-small fig-bold">k = 4</text>
<polyline points="70,252 74.9,251.7 79.8,251.4 84.7,251.1 89.6,250.8 94.5,250.5 99.4,250.2 104.3,249.9 109.2,249.6 114.1,249.2 119,248.9 123.9,248.5 128.8,248.2 133.7,247.8 138.6,247.4 143.5,247.1 148.4,246.7 153.3,246.3 158.2,245.9 163.1,245.4 168,245 172.9,244.6 177.8,244.1 182.7,243.6 187.6,243.2 192.5,242.7 197.4,242.2 202.3,241.6 207.2,241.1 212.1,240.6 217,240 221.9,239.4 226.8,238.8 231.7,238.2 236.6,237.6 241.5,236.9 246.4,236.3 251.3,235.6 256.2,234.8 261.1,234.1 266,233.3 270.9,232.6 275.8,231.7 280.7,230.9 285.6,230 290.5,229.1 295.4,228.2 300.3,227.2 305.2,226.2 310.1,225.2 315,224.1 319.9,223 324.8,221.8 329.7,220.6 334.6,219.4 339.5,218.1 344.4,216.7 349.3,215.3 354.2,213.8 359.1,212.3 364,210.7 368.9,209 373.8,207.3 378.7,205.5 383.6,203.6 388.5,201.7 393.4,199.6 398.3,197.5 403.2,195.2 408.1,192.9 413,190.4 417.9,187.9 422.8,185.2 427.7,182.4 432.6,179.5 437.5,176.4 442.4,173.2 447.3,169.8 452.2,166.3 457.1,162.6 462,158.8 466.9,154.8 471.8,150.5 476.7,146.1 481.6,141.4 486.5,136.6 491.4,131.5 496.3,126.1 501.2,120.5 506.1,114.6 511,108.5 515.9,102 520.8,95.3 525.7,88.2 530.6,80.7 535.5,72.9 540.4,64.8 545.3,56.2 550.2,47.2 555.1,37.8 560,28" fill="none" class="stroke-c" stroke-width="3"/>
<text x="566" y="32" class="fig-small fig-bold">k = 8</text>
<line x1="217" y1="240.1" x2="217" y2="252" class="stroke-ink dashed"/>
<circle cx="217" cy="240.1" r="5" class="fill-ink"/>
<line x1="462" y1="185.9" x2="462" y2="252" class="stroke-ink dashed"/>
<circle cx="462" cy="185.9" r="5" class="fill-ink"/>
<circle cx="86" cy="36.4" r="5" class="fill-ink"/>
<text x="98" y="40.4" class="fig-small">alpha 0.3, k = 4: 1.43 tokens per pass</text>
<circle cx="86" cy="61.6" r="5" class="fill-ink"/>
<text x="98" y="65.6" class="fig-small">alpha 0.8, k = 4: 3.36 tokens per pass</text>
<text x="315" y="290" text-anchor="middle">acceptance rate alpha</text>
<text x="70" y="18" class="fig-small">expected tokens per target pass</text>
</svg>
Figure 3. A draft that is never accepted (alpha near 0) gives 1 token per pass, so nothing is gained. A draft that is usually accepted approaches k + 1. More drafted tokens only pay when alpha is high. The two dots mark k = 4 at alpha 0.3 and 0.8.
```

Tokens per pass is not yet speed, because the draft also takes time. If a draft step costs 0.1 of a target step, a round with $k = 3$ takes $1 + 3 \times 0.1 = 1.3$ target steps. At $\alpha = 0.8$ the speedup is $2.952 / 1.3 = 2.27$. At $\alpha = 0.3$ it is $1.417 / 1.3 = 1.09$, which is barely worth the trouble.

> [!KEY]
> A big model that waits on memory can check several drafted tokens for about the price of one, so a cheap draft can pay off. Accepting a token with probability $\min(1, p/q)$ and replacing a rejected one from $\max(p - q, 0)$ makes the output follow $p$ exactly, whatever the draft is. A bad draft only costs speed, never correctness.

> [!RECAP]
> Before you build:
> - A draft token $x$ is kept with probability $\min(1, p(x)/q(x))$, and the chance that a token drawn from $q$ is kept is $\sum \min(p, q)$.
> - A rejected token is replaced by a draw from $\max(p - q, 0)$ divided by its total, with $p$ as the fallback when the total is zero.
> - Both draws use the `rng` you are given, never `np.random`.
> - The expected tokens per target pass are $(1 - \alpha^{k+1}) / (1 - \alpha)$, which tends to $k + 1$ as $\alpha$ approaches 1.

## Build it step by step

1. **`expected_accept_rate(p, q)`.** It takes two distributions of the same length and returns a float: the sum of the elementwise minimum. Check it on the worked example, where it gives 0.3. The usual mistake is to take the maximum, or to forget the sum.
2. **`residual_distribution(p, q)`.** It returns `np.maximum(p - q, 0)` divided by its sum, as an array that adds to 1. When the sum is 0, return a copy of `p`. On the worked example it gives `[0.857, 0.143, 0]`. The usual mistake is to forget the division.
3. **`speculative_accept(p, q, draft_token, rng)`.** Draw one number with `rng.uniform()` and compare it with `min(1, p[draft_token] / q[draft_token])`. If it is smaller, return `int(draft_token)`. Otherwise draw a replacement with `rng.choice(len(p), p=residual)`. It returns a plain `int`. The usual mistake is to draw the replacement from `p` instead of the leftover.

## Where this breaks

The speedup depends on three things. First, the chip must be memory bound. At a large batch, decode is already near the ridge point, so checking extra positions is no longer free, and drafting can slow the system down. Second, the draft must agree with the target often. At $\alpha$ near 0 you get about one token per round and pay for the draft as well. Third, $k$ cannot grow forever, because the $j$-th draft token survives with chance about $\alpha^j$, so late tokens are mostly wasted work. The draft also has to share the target's vocabulary, so that $p$ and $q$ describe the same tokens.

## The exercise

You write `expected_accept_rate`, `residual_distribution`, and `speculative_accept`. The checks confirm that the acceptance rate of the worked example is 0.3 and equals one minus the leftover mass for random distributions, and that the replacement distribution is `[6/7, 1/7, 0]`, falling back to $p$ when $p$ equals $q$. They check that a token with $p \ge q$ is always kept. They check that, given draft token 2, the output is `[0.75, 0.125, 0.125]`, which tells a correct rule from one that samples $p$ after a rejection. They check that drafts drawn from $q$ give outputs distributed as $p$ over 20,000 seeded rounds.
