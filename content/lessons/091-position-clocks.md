---
title: "Where is each token? Sinusoidal position clocks"
slug: 091-position-clocks
part: "Part 6 — Transformers"
summary: Attention ignores word order, so you stamp each position with a fingerprint of sine and cosine clock hands that turn at different speeds.
minutes: 40
gpu: false
objectives:
  - Show with numbers that attention gives the same outputs, shuffled, when you shuffle its input rows.
  - Explain why integer positions, a single frequency, and a learned table each fall short.
  - Build the sinusoidal encoding from clock hands that turn at different speeds, and compute it by hand.
  - Show that moving k positions is one fixed turn of every hand, and that the similarity of two positions depends only on their distance.
prereqs:
  - 088-scaled-attention-scores
  - 090-attention-backward-and-cost
exercise: 091-position-clocks
---

# Where is each token? Sinusoidal position clocks

"Dog bit man" and "man bit dog" use the same three words and mean opposite things. Self-attention, the mechanism you built over the last few chapters, cannot tell the two sentences apart. It gives each token a blend of the other tokens' vectors, and the blend weights come from dot products between the vectors. Nothing in that recipe looks at where a word stands.

That is a real problem. Without order, a model that reads "the dog bit the man" cannot know which noun did the biting, so it cannot answer "who was bitten?" The model needs a signal for position. This chapter builds the standard first one: a fingerprint for each position, made of sine and cosine "clock hands" that turn at different speeds, which you add to each token's vector.

The next chapter builds a second signal, rotary embeddings, and that is the one the transformer block in this course uses. The clock picture from this chapter carries over to it directly.

## What you already know

You already have these ideas:

- A token is a vector of numbers. Attention scores two tokens by the dot product of their vectors, then turns the scores into weights with a softmax.
- A token's attention output is the weighted average of the value vectors.
- For an angle $a$ in radians, the point $(\sin a, \cos a)$ lies on a circle of radius 1. A full turn is $2\pi \approx 6.28$ radians.
- A dot product is large when two vectors point the same way.
- Adding two vectors of the same length adds them entry by entry.

## Attention cannot see order

Figure 1 shows the failure with three tokens. Give each token a vector of two numbers: dog is $[1, 0]$, bit is $[0, 1]$, and man is $[1, 1]$. To keep the arithmetic small, let each token's own vector serve as its query, its key, and its value, and skip the division by $\sqrt{d}$ from the scaling chapter. Scaling changes the numbers but not the point.

```figure
<svg viewBox="0 0 600 296" role="img" aria-label="Attention weights for dog bit man and for man bit dog: the dog row holds the same weights in a new order and gives the same output">
<text x="164" y="26" text-anchor="middle" class="fig-bold">dog bit man</text>
<text x="108.0" y="58" text-anchor="middle" class="fig-small">dog</text>
<text x="164.0" y="58" text-anchor="middle" class="fig-small">bit</text>
<text x="220.0" y="58" text-anchor="middle" class="fig-small">man</text>
<text x="72" y="102.0" text-anchor="end" class="fig-small">dog</text>
<rect x="80" y="70" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.63"/>
<text x="108.0" y="102.0" text-anchor="middle" class="fig-small fig-mono">0.42</text>
<rect x="136" y="70" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.23"/>
<text x="164.0" y="102.0" text-anchor="middle" class="fig-small fig-mono">0.16</text>
<rect x="192" y="70" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.63"/>
<text x="220.0" y="102.0" text-anchor="middle" class="fig-small fig-mono">0.42</text>
<text x="72" y="158.0" text-anchor="end" class="fig-small">bit</text>
<rect x="80" y="126" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.23"/>
<text x="108.0" y="158.0" text-anchor="middle" class="fig-small fig-mono">0.16</text>
<rect x="136" y="126" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.63"/>
<text x="164.0" y="158.0" text-anchor="middle" class="fig-small fig-mono">0.42</text>
<rect x="192" y="126" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.63"/>
<text x="220.0" y="158.0" text-anchor="middle" class="fig-small fig-mono">0.42</text>
<text x="72" y="214.0" text-anchor="end" class="fig-small">man</text>
<rect x="80" y="182" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.32"/>
<text x="108.0" y="214.0" text-anchor="middle" class="fig-small fig-mono">0.21</text>
<rect x="136" y="182" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.32"/>
<text x="164.0" y="214.0" text-anchor="middle" class="fig-small fig-mono">0.21</text>
<rect x="192" y="182" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.86"/>
<text x="220.0" y="214.0" text-anchor="middle" class="fig-small fig-mono">0.58</text>
<rect x="80" y="70" width="168" height="56" class="stroke-b"/>
<text x="164" y="264" text-anchor="middle" class="fig-small">output for dog: [0.8446, 0.5777]</text>
<text x="484" y="26" text-anchor="middle" class="fig-bold">man bit dog</text>
<text x="428.0" y="58" text-anchor="middle" class="fig-small">man</text>
<text x="484.0" y="58" text-anchor="middle" class="fig-small">bit</text>
<text x="540.0" y="58" text-anchor="middle" class="fig-small">dog</text>
<text x="392" y="102.0" text-anchor="end" class="fig-small">man</text>
<rect x="400" y="70" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.86"/>
<text x="428.0" y="102.0" text-anchor="middle" class="fig-small fig-mono">0.58</text>
<rect x="456" y="70" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.32"/>
<text x="484.0" y="102.0" text-anchor="middle" class="fig-small fig-mono">0.21</text>
<rect x="512" y="70" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.32"/>
<text x="540.0" y="102.0" text-anchor="middle" class="fig-small fig-mono">0.21</text>
<text x="392" y="158.0" text-anchor="end" class="fig-small">bit</text>
<rect x="400" y="126" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.63"/>
<text x="428.0" y="158.0" text-anchor="middle" class="fig-small fig-mono">0.42</text>
<rect x="456" y="126" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.63"/>
<text x="484.0" y="158.0" text-anchor="middle" class="fig-small fig-mono">0.42</text>
<rect x="512" y="126" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.23"/>
<text x="540.0" y="158.0" text-anchor="middle" class="fig-small fig-mono">0.16</text>
<text x="392" y="214.0" text-anchor="end" class="fig-small">dog</text>
<rect x="400" y="182" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.63"/>
<text x="428.0" y="214.0" text-anchor="middle" class="fig-small fig-mono">0.42</text>
<rect x="456" y="182" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.23"/>
<text x="484.0" y="214.0" text-anchor="middle" class="fig-small fig-mono">0.16</text>
<rect x="512" y="182" width="56" height="56" class="fill-a stroke-grid" fill-opacity="0.63"/>
<text x="540.0" y="214.0" text-anchor="middle" class="fig-small fig-mono">0.42</text>
<rect x="400" y="182" width="168" height="56" class="stroke-b"/>
<text x="484" y="264" text-anchor="middle" class="fig-small">output for dog: [0.8446, 0.5777]</text>
<line x1="262" y1="154" x2="340" y2="154" class="stroke-muted" marker-end="url(#c101-arrow)"/>
<text x="301" y="144" text-anchor="middle" class="fig-small">swap man and dog</text>
<defs><marker id="c101-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="300" y="286" text-anchor="middle" class="fig-small">Orange outline: the dog row. Darker blue means more weight.</text>
</svg>
Figure 1. Attention weights for the same three tokens in two orders. The orange row is dog. Notice that it holds the same weights on the same words and gives the same output, so nothing in the output records the order.
```

The score between two tokens is their dot product. For dog the three scores are 1 (with dog), 0 (with bit), and 1 (with man). A softmax turns the scores $[1, 0, 1]$ into weights: $e^1 = 2.718$ and $e^0 = 1$, so the weights are $2.718 / 6.437 = 0.4223$, $1 / 6.437 = 0.1554$, and $0.4223$. The output for dog is the weighted average of the three vectors:

$$
0.4223\,[1,0] + 0.1554\,[0,1] + 0.4223\,[1,1] = [0.8446,\ 0.5777]
$$

Now read the sentence as "man bit dog". The rows and columns of the weight table are reordered, but the dog row holds the same three numbers attached to the same three words. The output for dog is still $[0.8446, 0.5777]$. Every token's output is unchanged and only appears in a different place. In words: shuffle the input rows and the output rows shuffle the same way. This property is called permutation equivariance, and it holds because each score depends only on the two vectors being compared, and an average does not care in which order you add the terms.

So the output for "dog" carries no record of whether dog came first or last. The model cannot tell the subject from the object.

## Why not add the position number?

The obvious fix is to add the position $p$ to every entry of each token's vector, so that "dog at position 3" becomes dog plus 3. Three versions of this idea fail, and each failure points at what the right answer needs.

- **Counting up grows without bound.** Position 500 adds 500 to entries that are about 1 in size, so the position drowns the word.
- **One clock hand repeats.** Use $\sin p$ and $\cos p$ as the fingerprint. The pair turns by one radian per position and returns to its start every 6.28 positions, so position 44 reads $[0.0177, 0.9998]$ and position 0 reads $[0, 1]$. They are nearly the same fingerprint, 44 steps apart.
- **A learned table has a hard limit.** You can store one trainable vector for each position. Several real models do, and it works within the lengths seen in training. A table with 2048 rows has nothing to say about position 2049.

So the fingerprint must stay small, must be different at every position, and must not need a table. A set of clocks does all three.

## A clock for each position

Think of a wall clock. The seconds hand returns to the top every 60 ticks, so on its own it cannot tell 10:00:05 from 10:01:05. The minute hand tells those apart, but it barely moves between two ticks. The hour hand separates morning from evening. Read together, the three hands give each moment a unique reading.

Position encoding uses the same idea. A hand is a point on a circle. For an angle $a$ its tip sits at $(\sin a, \cos a)$, which is two numbers. The hand for position $p$ turns to the angle $p\,\omega$, where $\omega$ (omega) is the hand's speed in radians per position. A fast hand has $\omega = 1$ and turns about 57 degrees per position. A slow hand has $\omega = 0.01$. Figure 2 draws four hands at positions 0 through 6.

```figure
<svg viewBox="0 0 600 340" role="img" aria-label="Clock hands for four frequencies at positions 0 to 6: the fast hand spins, the slowest barely moves">
<text x="159.0" y="30" text-anchor="middle" class="fig-small">p = 0</text>
<text x="221.0" y="30" text-anchor="middle" class="fig-small">p = 1</text>
<text x="283.0" y="30" text-anchor="middle" class="fig-small">p = 2</text>
<text x="345.0" y="30" text-anchor="middle" class="fig-small">p = 3</text>
<text x="407.0" y="30" text-anchor="middle" class="fig-small">p = 4</text>
<text x="469.0" y="30" text-anchor="middle" class="fig-small">p = 5</text>
<text x="531.0" y="30" text-anchor="middle" class="fig-small">p = 6</text>
<text x="14" y="81.0" class="fig-mono">ω = 1</text>
<text x="14" y="97.0" class="fig-small">fast hand</text>
<circle cx="159.0" cy="83.0" r="24" class="stroke-muted"/>
<line x1="159.0" y1="59.0" x2="159.0" y2="63.0" class="stroke-muted"/>
<line x1="159.0" y1="83.0" x2="159.0" y2="62.0" class="stroke-b"/>
<circle cx="159.0" cy="83.0" r="2.5" class="fill-ink"/>
<circle cx="221.0" cy="83.0" r="24" class="stroke-muted"/>
<line x1="221.0" y1="59.0" x2="221.0" y2="63.0" class="stroke-muted"/>
<line x1="221.0" y1="83.0" x2="238.7" y2="71.7" class="stroke-b"/>
<circle cx="221.0" cy="83.0" r="2.5" class="fill-ink"/>
<circle cx="283.0" cy="83.0" r="24" class="stroke-muted"/>
<line x1="283.0" y1="59.0" x2="283.0" y2="63.0" class="stroke-muted"/>
<line x1="283.0" y1="83.0" x2="302.1" y2="91.7" class="stroke-b"/>
<circle cx="283.0" cy="83.0" r="2.5" class="fill-ink"/>
<circle cx="345.0" cy="83.0" r="24" class="stroke-muted"/>
<line x1="345.0" y1="59.0" x2="345.0" y2="63.0" class="stroke-muted"/>
<line x1="345.0" y1="83.0" x2="348.0" y2="103.8" class="stroke-b"/>
<circle cx="345.0" cy="83.0" r="2.5" class="fill-ink"/>
<circle cx="407.0" cy="83.0" r="24" class="stroke-muted"/>
<line x1="407.0" y1="59.0" x2="407.0" y2="63.0" class="stroke-muted"/>
<line x1="407.0" y1="83.0" x2="391.1" y2="96.7" class="stroke-b"/>
<circle cx="407.0" cy="83.0" r="2.5" class="fill-ink"/>
<circle cx="469.0" cy="83.0" r="24" class="stroke-muted"/>
<line x1="469.0" y1="59.0" x2="469.0" y2="63.0" class="stroke-muted"/>
<line x1="469.0" y1="83.0" x2="448.9" y2="77.0" class="stroke-b"/>
<circle cx="469.0" cy="83.0" r="2.5" class="fill-ink"/>
<circle cx="531.0" cy="83.0" r="24" class="stroke-muted"/>
<line x1="531.0" y1="59.0" x2="531.0" y2="63.0" class="stroke-muted"/>
<line x1="531.0" y1="83.0" x2="525.1" y2="62.8" class="stroke-b"/>
<circle cx="531.0" cy="83.0" r="2.5" class="fill-ink"/>
<text x="14" y="147.0" class="fig-mono">ω = 0.1</text>
<text x="14" y="163.0" class="fig-small"></text>
<circle cx="159.0" cy="149.0" r="24" class="stroke-muted"/>
<line x1="159.0" y1="125.0" x2="159.0" y2="129.0" class="stroke-muted"/>
<line x1="159.0" y1="149.0" x2="159.0" y2="128.0" class="stroke-b"/>
<circle cx="159.0" cy="149.0" r="2.5" class="fill-ink"/>
<circle cx="221.0" cy="149.0" r="24" class="stroke-muted"/>
<line x1="221.0" y1="125.0" x2="221.0" y2="129.0" class="stroke-muted"/>
<line x1="221.0" y1="149.0" x2="223.1" y2="128.1" class="stroke-b"/>
<circle cx="221.0" cy="149.0" r="2.5" class="fill-ink"/>
<circle cx="283.0" cy="149.0" r="24" class="stroke-muted"/>
<line x1="283.0" y1="125.0" x2="283.0" y2="129.0" class="stroke-muted"/>
<line x1="283.0" y1="149.0" x2="287.2" y2="128.4" class="stroke-b"/>
<circle cx="283.0" cy="149.0" r="2.5" class="fill-ink"/>
<circle cx="345.0" cy="149.0" r="24" class="stroke-muted"/>
<line x1="345.0" y1="125.0" x2="345.0" y2="129.0" class="stroke-muted"/>
<line x1="345.0" y1="149.0" x2="351.2" y2="128.9" class="stroke-b"/>
<circle cx="345.0" cy="149.0" r="2.5" class="fill-ink"/>
<circle cx="407.0" cy="149.0" r="24" class="stroke-muted"/>
<line x1="407.0" y1="125.0" x2="407.0" y2="129.0" class="stroke-muted"/>
<line x1="407.0" y1="149.0" x2="415.2" y2="129.7" class="stroke-b"/>
<circle cx="407.0" cy="149.0" r="2.5" class="fill-ink"/>
<circle cx="469.0" cy="149.0" r="24" class="stroke-muted"/>
<line x1="469.0" y1="125.0" x2="469.0" y2="129.0" class="stroke-muted"/>
<line x1="469.0" y1="149.0" x2="479.1" y2="130.6" class="stroke-b"/>
<circle cx="469.0" cy="149.0" r="2.5" class="fill-ink"/>
<circle cx="531.0" cy="149.0" r="24" class="stroke-muted"/>
<line x1="531.0" y1="125.0" x2="531.0" y2="129.0" class="stroke-muted"/>
<line x1="531.0" y1="149.0" x2="542.9" y2="131.7" class="stroke-b"/>
<circle cx="531.0" cy="149.0" r="2.5" class="fill-ink"/>
<text x="14" y="213.0" class="fig-mono">ω = 0.01</text>
<text x="14" y="229.0" class="fig-small"></text>
<circle cx="159.0" cy="215.0" r="24" class="stroke-muted"/>
<line x1="159.0" y1="191.0" x2="159.0" y2="195.0" class="stroke-muted"/>
<line x1="159.0" y1="215.0" x2="159.0" y2="194.0" class="stroke-b"/>
<circle cx="159.0" cy="215.0" r="2.5" class="fill-ink"/>
<circle cx="221.0" cy="215.0" r="24" class="stroke-muted"/>
<line x1="221.0" y1="191.0" x2="221.0" y2="195.0" class="stroke-muted"/>
<line x1="221.0" y1="215.0" x2="221.2" y2="194.0" class="stroke-b"/>
<circle cx="221.0" cy="215.0" r="2.5" class="fill-ink"/>
<circle cx="283.0" cy="215.0" r="24" class="stroke-muted"/>
<line x1="283.0" y1="191.0" x2="283.0" y2="195.0" class="stroke-muted"/>
<line x1="283.0" y1="215.0" x2="283.4" y2="194.0" class="stroke-b"/>
<circle cx="283.0" cy="215.0" r="2.5" class="fill-ink"/>
<circle cx="345.0" cy="215.0" r="24" class="stroke-muted"/>
<line x1="345.0" y1="191.0" x2="345.0" y2="195.0" class="stroke-muted"/>
<line x1="345.0" y1="215.0" x2="345.6" y2="194.0" class="stroke-b"/>
<circle cx="345.0" cy="215.0" r="2.5" class="fill-ink"/>
<circle cx="407.0" cy="215.0" r="24" class="stroke-muted"/>
<line x1="407.0" y1="191.0" x2="407.0" y2="195.0" class="stroke-muted"/>
<line x1="407.0" y1="215.0" x2="407.8" y2="194.0" class="stroke-b"/>
<circle cx="407.0" cy="215.0" r="2.5" class="fill-ink"/>
<circle cx="469.0" cy="215.0" r="24" class="stroke-muted"/>
<line x1="469.0" y1="191.0" x2="469.0" y2="195.0" class="stroke-muted"/>
<line x1="469.0" y1="215.0" x2="470.0" y2="194.0" class="stroke-b"/>
<circle cx="469.0" cy="215.0" r="2.5" class="fill-ink"/>
<circle cx="531.0" cy="215.0" r="24" class="stroke-muted"/>
<line x1="531.0" y1="191.0" x2="531.0" y2="195.0" class="stroke-muted"/>
<line x1="531.0" y1="215.0" x2="532.3" y2="194.0" class="stroke-b"/>
<circle cx="531.0" cy="215.0" r="2.5" class="fill-ink"/>
<text x="14" y="279.0" class="fig-mono">ω = 0.001</text>
<text x="14" y="295.0" class="fig-small">slowest hand</text>
<circle cx="159.0" cy="281.0" r="24" class="stroke-muted"/>
<line x1="159.0" y1="257.0" x2="159.0" y2="261.0" class="stroke-muted"/>
<line x1="159.0" y1="281.0" x2="159.0" y2="260.0" class="stroke-b"/>
<circle cx="159.0" cy="281.0" r="2.5" class="fill-ink"/>
<circle cx="221.0" cy="281.0" r="24" class="stroke-muted"/>
<line x1="221.0" y1="257.0" x2="221.0" y2="261.0" class="stroke-muted"/>
<line x1="221.0" y1="281.0" x2="221.0" y2="260.0" class="stroke-b"/>
<circle cx="221.0" cy="281.0" r="2.5" class="fill-ink"/>
<circle cx="283.0" cy="281.0" r="24" class="stroke-muted"/>
<line x1="283.0" y1="257.0" x2="283.0" y2="261.0" class="stroke-muted"/>
<line x1="283.0" y1="281.0" x2="283.0" y2="260.0" class="stroke-b"/>
<circle cx="283.0" cy="281.0" r="2.5" class="fill-ink"/>
<circle cx="345.0" cy="281.0" r="24" class="stroke-muted"/>
<line x1="345.0" y1="257.0" x2="345.0" y2="261.0" class="stroke-muted"/>
<line x1="345.0" y1="281.0" x2="345.1" y2="260.0" class="stroke-b"/>
<circle cx="345.0" cy="281.0" r="2.5" class="fill-ink"/>
<circle cx="407.0" cy="281.0" r="24" class="stroke-muted"/>
<line x1="407.0" y1="257.0" x2="407.0" y2="261.0" class="stroke-muted"/>
<line x1="407.0" y1="281.0" x2="407.1" y2="260.0" class="stroke-b"/>
<circle cx="407.0" cy="281.0" r="2.5" class="fill-ink"/>
<circle cx="469.0" cy="281.0" r="24" class="stroke-muted"/>
<line x1="469.0" y1="257.0" x2="469.0" y2="261.0" class="stroke-muted"/>
<line x1="469.0" y1="281.0" x2="469.1" y2="260.0" class="stroke-b"/>
<circle cx="469.0" cy="281.0" r="2.5" class="fill-ink"/>
<circle cx="531.0" cy="281.0" r="24" class="stroke-muted"/>
<line x1="531.0" y1="257.0" x2="531.0" y2="261.0" class="stroke-muted"/>
<line x1="531.0" y1="281.0" x2="531.1" y2="260.0" class="stroke-b"/>
<circle cx="531.0" cy="281.0" r="2.5" class="fill-ink"/>
<text x="300" y="336" text-anchor="middle" class="fig-small">Each hand points to (sin pω, cos pω), measured clockwise from 12 o’clock. Here d = 8.</text>
</svg>
Figure 2. Four clock hands at positions 0 to 6. Notice that the fast hand turns ten times faster than the next, while the slowest hand barely leaves 12 o'clock. Together the four readings differ at every position.
```

Take two hands, a fast one with $\omega = 1$ and a slow one with $\omega = 0.01$, so the encoding has width 4. At position 0 every angle is 0, so the row is $[\sin 0, \cos 0, \sin 0, \cos 0] = [0, 1, 0, 1]$. At position 1 the row is $[\sin 1, \cos 1, \sin 0.01, \cos 0.01] = [0.8415, 0.5403, 0.0100, 0.99995]$. At position 2 it is $[0.9093, -0.4161, 0.0200, 0.9998]$. The three rows differ, and every entry stays between $-1$ and $1$, so the fingerprint never swamps the word.

For a width of $d$ entries there are $d/2$ hands. Hand $i$, for $i = 0, 1, \dots, d/2 - 1$, has speed $\omega_i$:

$$
\omega_i = 10000^{-2i/d}, \qquad
\mathrm{PE}(p)_{2i} = \sin(p\,\omega_i), \qquad
\mathrm{PE}(p)_{2i+1} = \cos(p\,\omega_i)
$$

Read it aloud. Hand 0 turns at speed 1, and each later hand turns slower, down to about 1/10000 for the last one. The encoding $\mathrm{PE}(p)$ of position $p$ lists each hand's sine and cosine side by side, with the sine in an even column and the cosine in the odd column next to it. At $d = 8$ the speeds are 1, 0.1, 0.01, and 0.001, which are the four rows of Figure 2. The base 10000 makes the slowest hand need about $2\pi \times 10000 \approx 63000$ positions to turn once.

In numpy, the speeds, the angles for every position and hand, and the interleaved columns take four lines:

```python
omega = 10000.0 ** (-2 * np.arange(d // 2) / d)      # (d/2,) one speed per hand
angles = np.arange(T)[:, None] * omega[None, :]      # (T, d/2) angle for each position and hand
pe = np.empty((T, d))
pe[:, 0::2], pe[:, 1::2] = np.sin(angles), np.cos(angles)
```

To use the fingerprint, add row $p$ of `pe` to the vector of the token at position $p$. Go back to the three tokens and use a single hand with $\omega = 1$, so the fingerprint is $[\sin p, \cos p]$. In "dog bit man" the three input vectors become $[1, 1]$, $[0.8415, 1.5403]$, and $[1.9093, 0.5839]$. In "man bit dog" they are $[1, 2]$, $[0.8415, 1.5403]$, and $[1.9093, -0.4161]$. Run attention on each and the output for dog is $[1.3063, 1.0268]$ in the first sentence and $[1.8022, -0.1768]$ in the second. The two sentences give different answers.

## Why a shift is a fixed turn

Any fingerprint that is different at every position would pass the three tests above. Why sines and cosines? Because a good fingerprint should make "three tokens later" a single operation that a network learns once and then reuses at every position. Sine and cosine pairs have exactly that property, and the reason is the angle-addition rules:

$$
\sin(a+b) = \sin a \cos b + \cos a \sin b, \qquad
\cos(a+b) = \cos a \cos b - \sin a \sin b
$$

Let $a = p\,\omega$ be a hand's angle at position $p$, and let $b = k\,\omega$ be the extra turn from $k$ more positions. The two rules say that the hand's reading at position $p + k$ comes from its reading at $p$ by a matrix:

$$
\begin{bmatrix} \sin((p+k)\omega) \\ \cos((p+k)\omega) \end{bmatrix}
=
\begin{bmatrix} \cos k\omega & \sin k\omega \\ -\sin k\omega & \cos k\omega \end{bmatrix}
\begin{bmatrix} \sin p\omega \\ \cos p\omega \end{bmatrix}
$$

Multiply out the first row to check: $\sin p\omega \cos k\omega + \cos p\omega \sin k\omega$ is the first rule. The matrix depends on $k$ and $\omega$ but not on $p$. So one matrix per offset $k$ works at every position, and it turns each hand by $k\omega$. Stack the matrices for all hands on the diagonal of a $d \times d$ matrix $R_k$ and you get $\mathrm{PE}(p+k) = R_k\,\mathrm{PE}(p)$ for every $p$.

A check with one hand at $\omega = 1$ and $k = 1$: the matrix is $\begin{bmatrix} 0.5403 & 0.8415 \\ -0.8415 & 0.5403 \end{bmatrix}$ and $\mathrm{PE}(1) = [0.8415, 0.5403]$. The first output entry is $0.5403 \times 0.8415 + 0.8415 \times 0.5403 = 0.9093$, which is $\sin 2$. The second is $-0.8415 \times 0.8415 + 0.5403 \times 0.5403 = -0.4161$, which is $\cos 2$. The result is $\mathrm{PE}(2)$.

The same rules give a second fact. The dot product of two fingerprints adds up one term per hand, and $\sin a \sin b + \cos a \cos b = \cos(a - b)$, so

$$
\mathrm{PE}(p) \cdot \mathrm{PE}(q) = \sum_{i=0}^{d/2-1} \cos\big((p - q)\,\omega_i\big)
$$

The similarity of two positions depends only on the distance between them. At width 4, positions 1 and 2 give $\cos 1 + \cos 0.01 = 1.5403$, and positions 3 and 4 give the same $1.5403$. In `numpy`, that is `pe @ pe.T`, and Figure 3 plots it.

```figure
<svg viewBox="0 0 600 336" role="img" aria-label="Left: the 32 by 32 sinusoidal encoding matrix as a heatmap. Right: the similarity of two positions against their distance, with points from three different starting positions all on one curve">
<text x="140.0" y="22" text-anchor="middle" class="fig-bold">Encoding matrix</text>
<rect x="36.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="42.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="49.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="55.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="62.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="68.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="75.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="81.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="88.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="94.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="101.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="107.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="114.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="120.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="127.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="133.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="140.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="146.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="153.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="159.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="172.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="185.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="198.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="211.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="40.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.84"/>
<rect x="42.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.54"/>
<rect x="49.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.53"/>
<rect x="55.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.85"/>
<rect x="62.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.31"/>
<rect x="68.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.95"/>
<rect x="75.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.18"/>
<rect x="81.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="88.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.10"/>
<rect x="94.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="101.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.06"/>
<rect x="107.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="114.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="120.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="127.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="133.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="140.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="146.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="153.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="159.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="172.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="185.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="198.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="211.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="46.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.91"/>
<rect x="42.5" y="53.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.42"/>
<rect x="49.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.90"/>
<rect x="55.5" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.43"/>
<rect x="62.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.59"/>
<rect x="68.5" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.81"/>
<rect x="75.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.35"/>
<rect x="81.5" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.94"/>
<rect x="88.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.20"/>
<rect x="94.5" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="101.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.11"/>
<rect x="107.5" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="114.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.06"/>
<rect x="120.5" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="127.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.04"/>
<rect x="133.5" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="140.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="146.5" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="153.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="159.5" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="172.5" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="185.5" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="198.5" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="211.5" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="53.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.14"/>
<rect x="42.5" y="59.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.99"/>
<rect x="49.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="55.5" y="59.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.12"/>
<rect x="62.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.81"/>
<rect x="68.5" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.58"/>
<rect x="75.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.51"/>
<rect x="81.5" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.86"/>
<rect x="88.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.30"/>
<rect x="94.5" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.96"/>
<rect x="101.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.17"/>
<rect x="107.5" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="114.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.09"/>
<rect x="120.5" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="127.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.05"/>
<rect x="133.5" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="140.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="146.5" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="153.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="159.5" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="172.5" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="185.5" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="198.5" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="211.5" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="59.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="66.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.76"/>
<rect x="42.5" y="66.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.65"/>
<rect x="49.0" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.78"/>
<rect x="55.5" y="66.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.63"/>
<rect x="62.0" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.95"/>
<rect x="68.5" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.30"/>
<rect x="75.0" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.65"/>
<rect x="81.5" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.76"/>
<rect x="88.0" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.39"/>
<rect x="94.5" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.92"/>
<rect x="101.0" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.22"/>
<rect x="107.5" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.97"/>
<rect x="114.0" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.13"/>
<rect x="120.5" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="127.0" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.07"/>
<rect x="133.5" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="140.0" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.04"/>
<rect x="146.5" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="153.0" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="159.5" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="172.5" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="185.5" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="198.5" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="211.5" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="66.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="72.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.96"/>
<rect x="42.5" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.28"/>
<rect x="49.0" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.32"/>
<rect x="55.5" y="72.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.95"/>
<rect x="62.0" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="68.5" y="72.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.01"/>
<rect x="75.0" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.78"/>
<rect x="81.5" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.63"/>
<rect x="88.0" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.48"/>
<rect x="94.5" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.88"/>
<rect x="101.0" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.28"/>
<rect x="107.5" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.96"/>
<rect x="114.0" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.16"/>
<rect x="120.5" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="127.0" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.09"/>
<rect x="133.5" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="140.0" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.05"/>
<rect x="146.5" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="153.0" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="159.5" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="172.5" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="185.5" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="198.5" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="211.5" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="72.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="79.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.28"/>
<rect x="42.5" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.96"/>
<rect x="49.0" y="79.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.23"/>
<rect x="55.5" y="79.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.97"/>
<rect x="62.0" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.95"/>
<rect x="68.5" y="79.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.32"/>
<rect x="75.0" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.88"/>
<rect x="81.5" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.48"/>
<rect x="88.0" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.56"/>
<rect x="94.5" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.83"/>
<rect x="101.0" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.33"/>
<rect x="107.5" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.94"/>
<rect x="114.0" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.19"/>
<rect x="120.5" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="127.0" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.11"/>
<rect x="133.5" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="140.0" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.06"/>
<rect x="146.5" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="153.0" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="159.5" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="172.5" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="185.5" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="198.5" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="211.5" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="79.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.66"/>
<rect x="42.5" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.75"/>
<rect x="49.0" y="85.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.71"/>
<rect x="55.5" y="85.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.70"/>
<rect x="62.0" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.80"/>
<rect x="68.5" y="85.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.60"/>
<rect x="75.0" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.95"/>
<rect x="81.5" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.32"/>
<rect x="88.0" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.64"/>
<rect x="94.5" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.76"/>
<rect x="101.0" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.38"/>
<rect x="107.5" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.92"/>
<rect x="114.0" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.22"/>
<rect x="120.5" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="127.0" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.12"/>
<rect x="133.5" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="140.0" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.07"/>
<rect x="146.5" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="153.0" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.04"/>
<rect x="159.5" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="172.5" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="185.5" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="198.5" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="211.5" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="85.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="42.5" y="92.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.15"/>
<rect x="49.0" y="92.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.98"/>
<rect x="55.5" y="92.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.21"/>
<rect x="62.0" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.57"/>
<rect x="68.5" y="92.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.82"/>
<rect x="75.0" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="81.5" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.15"/>
<rect x="88.0" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.72"/>
<rect x="94.5" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.70"/>
<rect x="101.0" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.43"/>
<rect x="107.5" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.90"/>
<rect x="114.0" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.25"/>
<rect x="120.5" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.97"/>
<rect x="127.0" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.14"/>
<rect x="133.5" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="140.0" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.08"/>
<rect x="146.5" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="153.0" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.04"/>
<rect x="159.5" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="172.5" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="185.5" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="198.5" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="211.5" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="92.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.41"/>
<rect x="42.5" y="98.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.91"/>
<rect x="49.0" y="98.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.94"/>
<rect x="55.5" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.34"/>
<rect x="62.0" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.29"/>
<rect x="68.5" y="98.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.96"/>
<rect x="75.0" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="81.5" y="98.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.03"/>
<rect x="88.0" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.78"/>
<rect x="94.5" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.62"/>
<rect x="101.0" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.48"/>
<rect x="107.5" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.87"/>
<rect x="114.0" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.28"/>
<rect x="120.5" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.96"/>
<rect x="127.0" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.16"/>
<rect x="133.5" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="140.0" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.09"/>
<rect x="146.5" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="153.0" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.05"/>
<rect x="159.5" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="172.5" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="185.5" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="198.5" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="98.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="105.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.54"/>
<rect x="42.5" y="105.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.84"/>
<rect x="49.0" y="105.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.61"/>
<rect x="55.5" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.79"/>
<rect x="62.0" y="105.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.02"/>
<rect x="68.5" y="105.0" width="6.5" height="6.5" class="fill-b" fill-opacity="1.00"/>
<rect x="75.0" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="81.5" y="105.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.21"/>
<rect x="88.0" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.84"/>
<rect x="94.5" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.54"/>
<rect x="101.0" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.53"/>
<rect x="107.5" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.85"/>
<rect x="114.0" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.31"/>
<rect x="120.5" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.95"/>
<rect x="127.0" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.18"/>
<rect x="133.5" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="140.0" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.10"/>
<rect x="146.5" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="153.0" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.06"/>
<rect x="159.5" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="172.5" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="185.5" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="198.5" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="105.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="111.5" width="6.5" height="6.5" class="fill-b" fill-opacity="1.00"/>
<rect x="42.5" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="49.0" y="111.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.10"/>
<rect x="55.5" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="62.0" y="111.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.33"/>
<rect x="68.5" y="111.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.94"/>
<rect x="75.0" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.93"/>
<rect x="81.5" y="111.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.38"/>
<rect x="88.0" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.89"/>
<rect x="94.5" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.45"/>
<rect x="101.0" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.58"/>
<rect x="107.5" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.81"/>
<rect x="114.0" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.34"/>
<rect x="120.5" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.94"/>
<rect x="127.0" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.19"/>
<rect x="133.5" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="140.0" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.11"/>
<rect x="146.5" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="153.0" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.06"/>
<rect x="159.5" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="172.5" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="185.5" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="198.5" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="111.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="118.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.54"/>
<rect x="42.5" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.84"/>
<rect x="49.0" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.45"/>
<rect x="55.5" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.89"/>
<rect x="62.0" y="118.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.61"/>
<rect x="68.5" y="118.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.79"/>
<rect x="75.0" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.85"/>
<rect x="81.5" y="118.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.53"/>
<rect x="88.0" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.93"/>
<rect x="94.5" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.36"/>
<rect x="101.0" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.62"/>
<rect x="107.5" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.78"/>
<rect x="114.0" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.37"/>
<rect x="120.5" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.93"/>
<rect x="127.0" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.21"/>
<rect x="133.5" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="140.0" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.12"/>
<rect x="146.5" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="153.0" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.07"/>
<rect x="159.5" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.04"/>
<rect x="172.5" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="185.5" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="198.5" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="118.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.42"/>
<rect x="42.5" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.91"/>
<rect x="49.0" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.86"/>
<rect x="55.5" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.52"/>
<rect x="62.0" y="124.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.82"/>
<rect x="68.5" y="124.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.57"/>
<rect x="75.0" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.74"/>
<rect x="81.5" y="124.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.68"/>
<rect x="88.0" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.96"/>
<rect x="94.5" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.27"/>
<rect x="101.0" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.67"/>
<rect x="107.5" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.74"/>
<rect x="114.0" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.40"/>
<rect x="120.5" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.92"/>
<rect x="127.0" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.23"/>
<rect x="133.5" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.97"/>
<rect x="140.0" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.13"/>
<rect x="146.5" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="153.0" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.07"/>
<rect x="159.5" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.04"/>
<rect x="172.5" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="185.5" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="198.5" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="124.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="42.5" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.14"/>
<rect x="49.0" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="55.5" y="131.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.02"/>
<rect x="62.0" y="131.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.96"/>
<rect x="68.5" y="131.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.28"/>
<rect x="75.0" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.61"/>
<rect x="81.5" y="131.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.79"/>
<rect x="88.0" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="94.5" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.17"/>
<rect x="101.0" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.71"/>
<rect x="107.5" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.71"/>
<rect x="114.0" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.43"/>
<rect x="120.5" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.90"/>
<rect x="127.0" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.25"/>
<rect x="133.5" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.97"/>
<rect x="140.0" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.14"/>
<rect x="146.5" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="153.0" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.08"/>
<rect x="159.5" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.04"/>
<rect x="172.5" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="185.5" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="198.5" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="131.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.65"/>
<rect x="42.5" y="137.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.76"/>
<rect x="49.0" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.84"/>
<rect x="55.5" y="137.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.55"/>
<rect x="62.0" y="137.5" width="6.5" height="6.5" class="fill-b" fill-opacity="1.00"/>
<rect x="68.5" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="75.0" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.46"/>
<rect x="81.5" y="137.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.89"/>
<rect x="88.0" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="94.5" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.07"/>
<rect x="101.0" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.75"/>
<rect x="107.5" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.66"/>
<rect x="114.0" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.46"/>
<rect x="120.5" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.89"/>
<rect x="127.0" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.26"/>
<rect x="133.5" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.96"/>
<rect x="140.0" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.15"/>
<rect x="146.5" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="153.0" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.08"/>
<rect x="159.5" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.05"/>
<rect x="172.5" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="185.5" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="198.5" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="224.5" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="137.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="144.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.29"/>
<rect x="42.5" y="144.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.96"/>
<rect x="49.0" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.41"/>
<rect x="55.5" y="144.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.91"/>
<rect x="62.0" y="144.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.94"/>
<rect x="68.5" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.34"/>
<rect x="75.0" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.29"/>
<rect x="81.5" y="144.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.96"/>
<rect x="88.0" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="94.5" y="144.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.03"/>
<rect x="101.0" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.78"/>
<rect x="107.5" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.62"/>
<rect x="114.0" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.48"/>
<rect x="120.5" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.87"/>
<rect x="127.0" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.28"/>
<rect x="133.5" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.96"/>
<rect x="140.0" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.16"/>
<rect x="146.5" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="153.0" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.09"/>
<rect x="159.5" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.05"/>
<rect x="172.5" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="185.5" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="198.5" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="144.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="150.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.96"/>
<rect x="42.5" y="150.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.28"/>
<rect x="49.0" y="150.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.13"/>
<rect x="55.5" y="150.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.99"/>
<rect x="62.0" y="150.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.79"/>
<rect x="68.5" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.62"/>
<rect x="75.0" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.12"/>
<rect x="81.5" y="150.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.99"/>
<rect x="88.0" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="94.5" y="150.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.13"/>
<rect x="101.0" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.82"/>
<rect x="107.5" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.58"/>
<rect x="114.0" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.51"/>
<rect x="120.5" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.86"/>
<rect x="127.0" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.30"/>
<rect x="133.5" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.95"/>
<rect x="140.0" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.17"/>
<rect x="146.5" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="153.0" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.10"/>
<rect x="159.5" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="166.0" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.05"/>
<rect x="172.5" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="185.5" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="198.5" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="150.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="157.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.75"/>
<rect x="42.5" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.66"/>
<rect x="49.0" y="157.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.64"/>
<rect x="55.5" y="157.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.77"/>
<rect x="62.0" y="157.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.56"/>
<rect x="68.5" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.83"/>
<rect x="75.0" y="157.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.06"/>
<rect x="81.5" y="157.0" width="6.5" height="6.5" class="fill-b" fill-opacity="1.00"/>
<rect x="88.0" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.97"/>
<rect x="94.5" y="157.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.23"/>
<rect x="101.0" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.85"/>
<rect x="107.5" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.53"/>
<rect x="114.0" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.54"/>
<rect x="120.5" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.84"/>
<rect x="127.0" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.31"/>
<rect x="133.5" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.95"/>
<rect x="140.0" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.18"/>
<rect x="146.5" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="153.0" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.10"/>
<rect x="159.5" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="166.0" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.06"/>
<rect x="172.5" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="185.5" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="198.5" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="157.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.15"/>
<rect x="42.5" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="49.0" y="163.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.95"/>
<rect x="55.5" y="163.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.31"/>
<rect x="62.0" y="163.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.27"/>
<rect x="68.5" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.96"/>
<rect x="75.0" y="163.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.23"/>
<rect x="81.5" y="163.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.97"/>
<rect x="88.0" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.95"/>
<rect x="94.5" y="163.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.32"/>
<rect x="101.0" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.88"/>
<rect x="107.5" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.48"/>
<rect x="114.0" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.57"/>
<rect x="120.5" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.82"/>
<rect x="127.0" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.33"/>
<rect x="133.5" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.94"/>
<rect x="140.0" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.19"/>
<rect x="146.5" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="153.0" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.11"/>
<rect x="159.5" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="166.0" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.06"/>
<rect x="172.5" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="185.5" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="198.5" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="163.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.91"/>
<rect x="42.5" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.41"/>
<rect x="49.0" y="170.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.97"/>
<rect x="55.5" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.25"/>
<rect x="62.0" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.04"/>
<rect x="68.5" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="75.0" y="170.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.40"/>
<rect x="81.5" y="170.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.92"/>
<rect x="88.0" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.91"/>
<rect x="94.5" y="170.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.42"/>
<rect x="101.0" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.90"/>
<rect x="107.5" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.43"/>
<rect x="114.0" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.59"/>
<rect x="120.5" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.81"/>
<rect x="127.0" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.35"/>
<rect x="133.5" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.94"/>
<rect x="140.0" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.20"/>
<rect x="146.5" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="153.0" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.11"/>
<rect x="159.5" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="166.0" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.06"/>
<rect x="172.5" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.04"/>
<rect x="185.5" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="198.5" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="170.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.84"/>
<rect x="42.5" y="176.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.55"/>
<rect x="49.0" y="176.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.69"/>
<rect x="55.5" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.73"/>
<rect x="62.0" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.35"/>
<rect x="68.5" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.94"/>
<rect x="75.0" y="176.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.56"/>
<rect x="81.5" y="176.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.83"/>
<rect x="88.0" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.86"/>
<rect x="94.5" y="176.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.50"/>
<rect x="101.0" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.92"/>
<rect x="107.5" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.38"/>
<rect x="114.0" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.62"/>
<rect x="120.5" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.79"/>
<rect x="127.0" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.36"/>
<rect x="133.5" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.93"/>
<rect x="140.0" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.21"/>
<rect x="146.5" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="153.0" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.12"/>
<rect x="159.5" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="166.0" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.07"/>
<rect x="172.5" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.04"/>
<rect x="185.5" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="198.5" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="176.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="183.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.01"/>
<rect x="42.5" y="183.0" width="6.5" height="6.5" class="fill-b" fill-opacity="1.00"/>
<rect x="49.0" y="183.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.19"/>
<rect x="55.5" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="62.0" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.62"/>
<rect x="68.5" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.78"/>
<rect x="75.0" y="183.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.70"/>
<rect x="81.5" y="183.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.72"/>
<rect x="88.0" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.81"/>
<rect x="94.5" y="183.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.59"/>
<rect x="101.0" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.94"/>
<rect x="107.5" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.33"/>
<rect x="114.0" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.64"/>
<rect x="120.5" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.77"/>
<rect x="127.0" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.38"/>
<rect x="133.5" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.92"/>
<rect x="140.0" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.22"/>
<rect x="146.5" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="153.0" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.12"/>
<rect x="159.5" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="166.0" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.07"/>
<rect x="172.5" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.04"/>
<rect x="185.5" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="198.5" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="183.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="189.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.85"/>
<rect x="42.5" y="189.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.53"/>
<rect x="49.0" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.36"/>
<rect x="55.5" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.93"/>
<rect x="62.0" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.84"/>
<rect x="68.5" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.55"/>
<rect x="75.0" y="189.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.81"/>
<rect x="81.5" y="189.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.58"/>
<rect x="88.0" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.75"/>
<rect x="94.5" y="189.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.67"/>
<rect x="101.0" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.96"/>
<rect x="107.5" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.27"/>
<rect x="114.0" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.66"/>
<rect x="120.5" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.75"/>
<rect x="127.0" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.40"/>
<rect x="133.5" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.92"/>
<rect x="140.0" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.23"/>
<rect x="146.5" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.97"/>
<rect x="153.0" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.13"/>
<rect x="159.5" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="166.0" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.07"/>
<rect x="172.5" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.04"/>
<rect x="185.5" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="198.5" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="189.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="196.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.91"/>
<rect x="42.5" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.42"/>
<rect x="49.0" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.80"/>
<rect x="55.5" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.60"/>
<rect x="62.0" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.97"/>
<rect x="68.5" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.26"/>
<rect x="75.0" y="196.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.90"/>
<rect x="81.5" y="196.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.43"/>
<rect x="88.0" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.68"/>
<rect x="94.5" y="196.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.74"/>
<rect x="101.0" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="107.5" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.22"/>
<rect x="114.0" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.69"/>
<rect x="120.5" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.73"/>
<rect x="127.0" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.41"/>
<rect x="133.5" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.91"/>
<rect x="140.0" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.24"/>
<rect x="146.5" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.97"/>
<rect x="153.0" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.13"/>
<rect x="159.5" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="166.0" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.08"/>
<rect x="172.5" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.04"/>
<rect x="185.5" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="198.5" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="196.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="202.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.13"/>
<rect x="42.5" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="49.0" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="55.5" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.08"/>
<rect x="62.0" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="68.5" y="202.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.05"/>
<rect x="75.0" y="202.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.96"/>
<rect x="81.5" y="202.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.26"/>
<rect x="88.0" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.60"/>
<rect x="94.5" y="202.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.80"/>
<rect x="101.0" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="107.5" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.16"/>
<rect x="114.0" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.71"/>
<rect x="120.5" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.70"/>
<rect x="127.0" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.43"/>
<rect x="133.5" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.90"/>
<rect x="140.0" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.25"/>
<rect x="146.5" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.97"/>
<rect x="153.0" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.14"/>
<rect x="159.5" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="166.0" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.08"/>
<rect x="172.5" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.04"/>
<rect x="185.5" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="198.5" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="202.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.76"/>
<rect x="42.5" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.65"/>
<rect x="49.0" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.89"/>
<rect x="55.5" y="209.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.47"/>
<rect x="62.0" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.93"/>
<rect x="68.5" y="209.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.36"/>
<rect x="75.0" y="209.0" width="6.5" height="6.5" class="fill-b" fill-opacity="1.00"/>
<rect x="81.5" y="209.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.09"/>
<rect x="88.0" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.52"/>
<rect x="94.5" y="209.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.86"/>
<rect x="101.0" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="107.5" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.11"/>
<rect x="114.0" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.73"/>
<rect x="120.5" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.68"/>
<rect x="127.0" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.45"/>
<rect x="133.5" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.90"/>
<rect x="140.0" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.26"/>
<rect x="146.5" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.97"/>
<rect x="153.0" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.15"/>
<rect x="159.5" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="166.0" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.08"/>
<rect x="172.5" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.05"/>
<rect x="185.5" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="198.5" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="211.5" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="209.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.96"/>
<rect x="42.5" y="215.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.29"/>
<rect x="49.0" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.50"/>
<rect x="55.5" y="215.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.87"/>
<rect x="62.0" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.77"/>
<rect x="68.5" y="215.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.63"/>
<rect x="75.0" y="215.5" width="6.5" height="6.5" class="fill-b" fill-opacity="1.00"/>
<rect x="81.5" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.09"/>
<rect x="88.0" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.43"/>
<rect x="94.5" y="215.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.90"/>
<rect x="101.0" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="107.5" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.05"/>
<rect x="114.0" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.75"/>
<rect x="120.5" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.66"/>
<rect x="127.0" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.46"/>
<rect x="133.5" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.89"/>
<rect x="140.0" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.27"/>
<rect x="146.5" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.96"/>
<rect x="153.0" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.15"/>
<rect x="159.5" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="166.0" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.09"/>
<rect x="172.5" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.05"/>
<rect x="185.5" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="198.5" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="211.5" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="215.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.27"/>
<rect x="42.5" y="222.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.96"/>
<rect x="49.0" y="222.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.04"/>
<rect x="55.5" y="222.0" width="6.5" height="6.5" class="fill-b" fill-opacity="1.00"/>
<rect x="62.0" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.54"/>
<rect x="68.5" y="222.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.84"/>
<rect x="75.0" y="222.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.96"/>
<rect x="81.5" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.26"/>
<rect x="88.0" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.33"/>
<rect x="94.5" y="222.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.94"/>
<rect x="101.0" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="107.5" y="222.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.00"/>
<rect x="114.0" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.77"/>
<rect x="120.5" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.63"/>
<rect x="127.0" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.48"/>
<rect x="133.5" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.88"/>
<rect x="140.0" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.28"/>
<rect x="146.5" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.96"/>
<rect x="153.0" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.16"/>
<rect x="159.5" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="166.0" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.09"/>
<rect x="172.5" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.05"/>
<rect x="185.5" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="198.5" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="211.5" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.00"/>
<rect x="237.5" y="222.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="228.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.66"/>
<rect x="42.5" y="228.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.75"/>
<rect x="49.0" y="228.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.56"/>
<rect x="55.5" y="228.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.83"/>
<rect x="62.0" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.25"/>
<rect x="68.5" y="228.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.97"/>
<rect x="75.0" y="228.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.90"/>
<rect x="81.5" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.43"/>
<rect x="88.0" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.24"/>
<rect x="94.5" y="228.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.97"/>
<rect x="101.0" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="107.5" y="228.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.06"/>
<rect x="114.0" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.79"/>
<rect x="120.5" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.61"/>
<rect x="127.0" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.49"/>
<rect x="133.5" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.87"/>
<rect x="140.0" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.29"/>
<rect x="146.5" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.96"/>
<rect x="153.0" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.16"/>
<rect x="159.5" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="166.0" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.09"/>
<rect x="172.5" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.05"/>
<rect x="185.5" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="198.5" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="211.5" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="237.5" y="228.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="235.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.99"/>
<rect x="42.5" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.15"/>
<rect x="49.0" y="235.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.92"/>
<rect x="55.5" y="235.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.40"/>
<rect x="62.0" y="235.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.06"/>
<rect x="68.5" y="235.0" width="6.5" height="6.5" class="fill-b" fill-opacity="1.00"/>
<rect x="75.0" y="235.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.81"/>
<rect x="81.5" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.58"/>
<rect x="88.0" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.14"/>
<rect x="94.5" y="235.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.99"/>
<rect x="101.0" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="107.5" y="235.0" width="6.5" height="6.5" class="fill-b" fill-opacity="0.12"/>
<rect x="114.0" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.81"/>
<rect x="120.5" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.58"/>
<rect x="127.0" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.51"/>
<rect x="133.5" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.86"/>
<rect x="140.0" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.30"/>
<rect x="146.5" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.96"/>
<rect x="153.0" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.17"/>
<rect x="159.5" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="166.0" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.09"/>
<rect x="172.5" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.05"/>
<rect x="185.5" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="198.5" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="211.5" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="237.5" y="235.0" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36.0" y="241.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.40"/>
<rect x="42.5" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.91"/>
<rect x="49.0" y="241.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.99"/>
<rect x="55.5" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.15"/>
<rect x="62.0" y="241.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.37"/>
<rect x="68.5" y="241.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.93"/>
<rect x="75.0" y="241.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.70"/>
<rect x="81.5" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.72"/>
<rect x="88.0" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.04"/>
<rect x="94.5" y="241.5" width="6.5" height="6.5" class="fill-b" fill-opacity="1.00"/>
<rect x="101.0" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.99"/>
<rect x="107.5" y="241.5" width="6.5" height="6.5" class="fill-b" fill-opacity="0.17"/>
<rect x="114.0" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.83"/>
<rect x="120.5" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.56"/>
<rect x="127.0" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.52"/>
<rect x="133.5" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.85"/>
<rect x="140.0" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.31"/>
<rect x="146.5" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.95"/>
<rect x="153.0" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.17"/>
<rect x="159.5" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.98"/>
<rect x="166.0" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.10"/>
<rect x="172.5" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="179.0" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.06"/>
<rect x="185.5" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="192.0" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.03"/>
<rect x="198.5" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="205.0" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.02"/>
<rect x="211.5" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="218.0" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="224.5" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="231.0" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="0.01"/>
<rect x="237.5" y="241.5" width="6.5" height="6.5" class="fill-a" fill-opacity="1.00"/>
<rect x="36" y="40" width="208.0" height="208.0" class="stroke-muted"/>
<text x="140.0" y="264.0" text-anchor="middle" class="fig-small">columns: fast hands left, slow right</text>
<text x="140.0" y="280.0" text-anchor="middle" class="fig-small">rows: p = 0 at top, 31 at bottom</text>
<text x="460.0" y="22" text-anchor="middle" class="fig-bold">Similarity against distance</text>
<line x1="350" y1="252.0" x2="570" y2="252.0" class="stroke-grid"/>
<text x="344" y="256.0" text-anchor="end" class="fig-small">0</text>
<line x1="350" y1="156.0" x2="570" y2="156.0" class="stroke-grid"/>
<text x="344" y="160.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="350" y1="60.0" x2="570" y2="60.0" class="stroke-grid"/>
<text x="344" y="64.0" text-anchor="end" class="fig-small">1</text>
<polyline points="350.0,60.0 351.8,60.5 353.5,62.2 355.3,64.8 357.1,68.2 358.9,72.4 360.6,77.1 362.4,82.1 364.2,87.2 366.0,92.2 367.7,96.9 369.5,101.1 371.3,104.7 373.1,107.6 374.8,109.8 376.6,111.3 378.4,112.1 380.2,112.3 381.9,112.0 383.7,111.4 385.5,110.7 387.3,109.9 389.0,109.4 390.8,109.1 392.6,109.3 394.4,109.9 396.1,111.1 397.9,112.8 399.7,114.9 401.5,117.4 403.2,120.1 405.0,122.9 406.8,125.7 408.5,128.3 410.3,130.5 412.1,132.3 413.9,133.4 415.6,133.9 417.4,133.7 419.2,132.8 421.0,131.3 422.7,129.3 424.5,126.9 426.3,124.3 428.1,121.7 429.8,119.2 431.6,117.1 433.4,115.5 435.2,114.6 436.9,114.5 438.7,115.2 440.5,116.7 442.3,119.1 444.0,122.2 445.8,126.0 447.6,130.2 449.4,134.7 451.1,139.2 452.9,143.6 454.7,147.6 456.5,151.1 458.2,153.9 460.0,155.8 461.8,156.8 463.5,156.9 465.3,156.0 467.1,154.3 468.9,151.9 470.6,148.8 472.4,145.4 474.2,141.7 476.0,138.0 477.7,134.4 479.5,131.3 481.3,128.6 483.1,126.5 484.8,125.1 486.6,124.5 488.4,124.5 490.2,125.1 491.9,126.3 493.7,128.0 495.5,129.9 497.3,131.9 499.0,133.8 500.8,135.6 502.6,137.1 504.4,138.1 506.1,138.7 507.9,138.8 509.7,138.5 511.5,137.8 513.2,136.8 515.0,135.7 516.8,134.5 518.5,133.5 520.3,132.8 522.1,132.6 523.9,133.0 525.6,134.1 527.4,135.9 529.2,138.4 531.0,141.6 532.7,145.5 534.5,149.9 536.3,154.6 538.1,159.4 539.8,164.2 541.6,168.7 543.4,172.8 545.2,176.2 546.9,178.7 548.7,180.4 550.5,181.0 552.3,180.6 554.0,179.1 555.8,176.7 557.6,173.5 559.4,169.6 561.1,165.2 562.9,160.6 564.7,156.0 566.5,151.5 568.2,147.3 570.0,143.7" class="stroke-ink"/>
<circle cx="350.0" cy="60.0" r="3" class="fill-a"/>
<circle cx="357.1" cy="68.2" r="3" class="fill-a"/>
<circle cx="364.2" cy="87.2" r="3" class="fill-a"/>
<circle cx="371.3" cy="104.7" r="3" class="fill-a"/>
<circle cx="378.4" cy="112.1" r="3" class="fill-a"/>
<circle cx="385.5" cy="110.7" r="3" class="fill-a"/>
<circle cx="392.6" cy="109.3" r="3" class="fill-a"/>
<circle cx="399.7" cy="114.9" r="3" class="fill-a"/>
<circle cx="406.8" cy="125.7" r="3" class="fill-a"/>
<circle cx="413.9" cy="133.4" r="3" class="fill-a"/>
<circle cx="421.0" cy="131.3" r="3" class="fill-a"/>
<circle cx="428.1" cy="121.7" r="3" class="fill-a"/>
<circle cx="435.2" cy="114.6" r="3" class="fill-a"/>
<circle cx="442.3" cy="119.1" r="3" class="fill-a"/>
<circle cx="449.4" cy="134.7" r="3" class="fill-a"/>
<circle cx="456.5" cy="151.1" r="3" class="fill-a"/>
<circle cx="463.5" cy="156.9" r="3" class="fill-a"/>
<circle cx="470.6" cy="148.8" r="3" class="fill-a"/>
<circle cx="477.7" cy="134.4" r="3" class="fill-a"/>
<circle cx="484.8" cy="125.1" r="3" class="fill-a"/>
<circle cx="491.9" cy="126.3" r="3" class="fill-a"/>
<circle cx="499.0" cy="133.8" r="3" class="fill-a"/>
<circle cx="506.1" cy="138.7" r="3" class="fill-a"/>
<circle cx="513.2" cy="136.8" r="3" class="fill-a"/>
<circle cx="520.3" cy="132.8" r="3" class="fill-a"/>
<circle cx="527.4" cy="135.9" r="3" class="fill-a"/>
<circle cx="534.5" cy="149.9" r="3" class="fill-a"/>
<circle cx="541.6" cy="168.7" r="3" class="fill-a"/>
<circle cx="548.7" cy="180.4" r="3" class="fill-a"/>
<circle cx="555.8" cy="176.7" r="3" class="fill-a"/>
<circle cx="562.9" cy="160.6" r="3" class="fill-a"/>
<circle cx="570.0" cy="143.7" r="3" class="fill-a"/>
<circle cx="350.0" cy="60.0" r="3" class="fill-b"/>
<circle cx="357.1" cy="68.2" r="3" class="fill-b"/>
<circle cx="364.2" cy="87.2" r="3" class="fill-b"/>
<circle cx="371.3" cy="104.7" r="3" class="fill-b"/>
<circle cx="378.4" cy="112.1" r="3" class="fill-b"/>
<circle cx="385.5" cy="110.7" r="3" class="fill-b"/>
<circle cx="392.6" cy="109.3" r="3" class="fill-b"/>
<circle cx="399.7" cy="114.9" r="3" class="fill-b"/>
<circle cx="406.8" cy="125.7" r="3" class="fill-b"/>
<circle cx="413.9" cy="133.4" r="3" class="fill-b"/>
<circle cx="421.0" cy="131.3" r="3" class="fill-b"/>
<circle cx="428.1" cy="121.7" r="3" class="fill-b"/>
<circle cx="435.2" cy="114.6" r="3" class="fill-b"/>
<circle cx="442.3" cy="119.1" r="3" class="fill-b"/>
<circle cx="449.4" cy="134.7" r="3" class="fill-b"/>
<circle cx="456.5" cy="151.1" r="3" class="fill-b"/>
<circle cx="463.5" cy="156.9" r="3" class="fill-b"/>
<circle cx="470.6" cy="148.8" r="3" class="fill-b"/>
<circle cx="477.7" cy="134.4" r="3" class="fill-b"/>
<circle cx="484.8" cy="125.1" r="3" class="fill-b"/>
<circle cx="491.9" cy="126.3" r="3" class="fill-b"/>
<circle cx="499.0" cy="133.8" r="3" class="fill-b"/>
<circle cx="506.1" cy="138.7" r="3" class="fill-b"/>
<circle cx="350.0" cy="60.0" r="3" class="fill-c"/>
<circle cx="357.1" cy="68.2" r="3" class="fill-c"/>
<circle cx="364.2" cy="87.2" r="3" class="fill-c"/>
<circle cx="371.3" cy="104.7" r="3" class="fill-c"/>
<circle cx="378.4" cy="112.1" r="3" class="fill-c"/>
<circle cx="385.5" cy="110.7" r="3" class="fill-c"/>
<circle cx="392.6" cy="109.3" r="3" class="fill-c"/>
<circle cx="399.7" cy="114.9" r="3" class="fill-c"/>
<circle cx="406.8" cy="125.7" r="3" class="fill-c"/>
<circle cx="413.9" cy="133.4" r="3" class="fill-c"/>
<circle cx="421.0" cy="131.3" r="3" class="fill-c"/>
<circle cx="428.1" cy="121.7" r="3" class="fill-c"/>
<circle cx="435.2" cy="114.6" r="3" class="fill-c"/>
<circle cx="442.3" cy="119.1" r="3" class="fill-c"/>
<text x="350.0" y="268" text-anchor="middle" class="fig-small">0</text>
<text x="421.0" y="268" text-anchor="middle" class="fig-small">10</text>
<text x="491.9" y="268" text-anchor="middle" class="fig-small">20</text>
<text x="562.9" y="268" text-anchor="middle" class="fig-small">30</text>
<text x="460.0" y="284" text-anchor="middle" class="fig-small">distance |p − q|</text>
<circle cx="372" cy="320" r="3" class="fill-a"/><text x="380" y="324" class="fig-small">p = 0</text>
<circle cx="432" cy="320" r="3" class="fill-b"/><text x="440" y="324" class="fig-small">p = 9</text>
<circle cx="492" cy="320" r="3" class="fill-c"/><text x="500" y="324" class="fig-small">p = 18</text>
</svg>
Figure 3. Left: the encoding for 32 positions and 32 columns, blue for positive and orange for negative values. Right: the similarity of two positions against their distance, divided by d/2. Notice that dots from three different starting positions land on one curve, so only the distance matters, and nearby positions are the most similar.
```

> [!KEY]
> Attention cannot see order, so you add to each token a fingerprint made of clock hands that turn at different speeds. Fast hands separate neighbours and slow hands separate distant positions. Because moving $k$ positions turns every hand by a fixed amount, "$k$ tokens later" is the same linear operation everywhere, and "dog bit man" no longer looks like "man bit dog".

## Where this breaks

**Adding mixes position into the word.** The fingerprint lands in the same coordinates that hold the word's meaning, so later layers have to learn to pull the two apart. The next chapter rotates the queries and keys instead, so the attention score itself depends on distance. The transformer block in this course uses that method, so you will not see the sinusoidal encoding inside the finished block. The two are alternatives.

**The formula gives a value at any position, but the network may not know what to do with it.** A table has no row for a position it never trained on. The sinusoid always returns a vector, yet a network that has only seen short sequences has still never met the angles of long ones.

> [!WARNING]
> This chapter puts the sine in an even column and the cosine in the odd column next to it. Some libraries put all the sines first and all the cosines after. Both carry the same information, but a network trained with one layout reads the other as noise. Match the layout of the model you load.

Learned tables and ALiBi, which subtracts a penalty from the attention score that grows with distance and stores no vectors at all, are two more ways to supply position.

> [!RECAP]
> Carry these facts into the exercise:
> - Shuffling the input rows of attention shuffles its output rows the same way, so attention alone has no sense of order.
> - Hand $i$ turns at speed $\omega_i = 10000^{-2i/d}$, and its reading at position $p$ is $(\sin p\omega_i, \cos p\omega_i)$ in columns $2i$ and $2i+1$.
> - Moving $k$ positions turns hand $i$ by $k\,\omega_i$, with a matrix that does not depend on $p$.
> - The dot product of two fingerprints depends only on the distance between their positions.

## Build it step by step

You write three functions, in the order the checks test them.

1. **`sinusoidal_encoding(T, d)`** returns an array of shape `(T, d)`. Build `omega` with `np.arange(d // 2)`, take the outer product with the positions, and write `np.sin` into columns `0::2` and `np.cos` into columns `1::2`. The common mistake is to drop the 2 from the exponent, which stops the speeds at 0.01 instead of 1/10000.
2. **`offset_rotation(k, d)`** returns the `(d, d)` matrix $R_k$ with `R @ pe[p] == pe[p + k]`. Start from `np.zeros((d, d))`, loop over the hands, and fill the block at rows and columns `2*i` and `2*i + 1` with `[[cos, sin], [-sin, cos]]` for the angle `k * omega[i]`. The common mistake is to transpose the block, which moves backwards by `k`.
3. **`add_positions(x, pe)`** takes token vectors of shape `(T, d)` and a position table with at least `T` rows, and returns `x + pe[:T]`. The common mistake is `x += pe[:T]`, which changes the caller's array.

## The exercise

You write `sinusoidal_encoding`, `offset_rotation`, and `add_positions`. The checks compare your encoding with the hand values for positions 0 and 1 at width 4, and confirm that the similarity matrix `pe @ pe.T` has diagonal $d/2$ and depends only on the distance between positions. They apply your `offset_rotation` matrix to the encoding at several positions and compare with the encoding of the shifted position, and confirm that turning by $a$ and then by $b$ equals turning by $a + b$. The last checks confirm that `add_positions` leaves its input alone and gives two identical tokens different vectors.
