---
title: "Rotating queries and keys: rotary position embeddings"
slug: 092-rotary-embeddings
part: "Part 6 — Transformers"
summary: Turn each query and key by an angle that grows with its position, and the attention score depends only on how far apart the two tokens are.
minutes: 40
gpu: false
objectives:
  - Explain why adding a position vector tells the attention score about distance only indirectly.
  - Show, with a rotation matrix, that rotating a query by m and a key by n makes their dot product depend on n minus m.
  - Build rotary embeddings from many two-entry planes that turn at different speeds, using the rotate-half layout.
  - Apply the rotation to queries and keys only, and say why the values stay untouched.
prereqs:
  - 088-scaled-attention-scores
  - 091-position-clocks
exercise: 092-rotary-embeddings
---

# Rotating queries and keys: rotary position embeddings

Take the word "bit" in "dog bit man" and ask it to find its subject, the noun right before it. For that to work, the attention score between "bit" and "dog" must know that the two sit one place apart. It must also give the same answer when the sentence starts at the top of a page and when it starts fifty words in.

The previous chapter, "Where is each token? Sinusoidal position clocks", added a position fingerprint to each token's vector before the query and key matrices touch it. That carries the position in, but only indirectly. After the matrices, the score is a tangle of word terms, position terms, and mixed terms, and nothing guarantees that the distance between the tokens appears in it as a clean number. The network has to learn to untangle it.

Can you put position into the query and the key themselves, so the score is already a function of the distance? Yes, with a rotation: turn each query and key by an angle proportional to its position. This chapter shows why that works, builds it, and explains how it fits a real attention head. The transformer block in this course uses this method.

## What you already know

You already have these ideas:

- An attention score is the dot product of a query and a key, and a softmax turns the scores into weights.
- A point at angle $a$ on a circle of radius 1 is $(\cos a, \sin a)$, and a clock hand with speed $\omega$ reaches angle $p\,\omega$ at position $p$, as in the previous chapter.
- A dot product equals the two lengths times the cosine of the angle between the vectors.
- Turning a point $[x, y]$ by an angle $a$ gives $[x\cos a - y\sin a,\ x\sin a + y\cos a]$ and keeps its length.
- Turning by $a$ and then by $b$ is one turn by $a + b$, and turning by $-a$ undoes a turn by $a$.

## Turn the arrow by its position

Picture a query and a key as arrows in a plane. Turn the query by $m\,\omega$, where $m$ is its position. Turn the key by $n\,\omega$, where $n$ is its position. The angle between the two arrows has changed by $n\,\omega - m\,\omega = (n - m)\,\omega$, which is a function of the distance and nothing else. A dot product is the two lengths times the cosine of the angle between them, so the lengths still carry the content and the angle carries the distance. Figure 1 draws two cases.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="A query and a key drawn as arrows on a circle, turned by their positions. Moving both by two positions leaves the angle between them unchanged">
<defs><marker id="c102-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c102-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c102-arrow-m" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="150" y="22" text-anchor="middle" class="fig-bold">q at position 1, k at position 3</text>
<circle cx="150" cy="145" r="92" class="stroke-grid"/>
<line x1="52" y1="145" x2="248" y2="145" class="stroke-grid"/>
<line x1="150" y1="47" x2="150" y2="243" class="stroke-grid"/>
<line x1="150" y1="145" x2="242.0" y2="145.0" class="stroke-muted dashed"/>
<text x="246.0" y="139" class="fig-small">start [1, 0]</text>
<line x1="150" y1="145" x2="197.5" y2="71.0" class="stroke-a" marker-end="url(#c102-arrow-a)"/>
<line x1="150" y1="145" x2="62.9" y2="132.6" class="stroke-b" marker-end="url(#c102-arrow-b)"/>
<text x="203.3" y="59.8" class="fig-bold">q</text>
<text x="41.1" y="134.0" class="fig-bold">k</text>
<path d="M170.5,113.0 A38,38 0 0 0 112.4,139.6" class="stroke-ink"/>
<text x="112.7" y="98.1" class="fig-small">gap 2</text>
<text x="150" y="271" text-anchor="middle" class="fig-mono">score = cos 2 = -0.4161</text>
<text x="450" y="22" text-anchor="middle" class="fig-bold">q at position 3, k at position 5</text>
<circle cx="450" cy="145" r="92" class="stroke-grid"/>
<line x1="352" y1="145" x2="548" y2="145" class="stroke-grid"/>
<line x1="450" y1="47" x2="450" y2="243" class="stroke-grid"/>
<line x1="450" y1="145" x2="542.0" y2="145.0" class="stroke-muted dashed"/>
<text x="546.0" y="139" class="fig-small">start [1, 0]</text>
<line x1="450" y1="145" x2="362.9" y2="132.6" class="stroke-a" marker-end="url(#c102-arrow-a)"/>
<line x1="450" y1="145" x2="475.0" y2="229.4" class="stroke-b" marker-end="url(#c102-arrow-b)"/>
<text x="341.1" y="134.0" class="fig-bold">q</text>
<text x="476.1" y="250.6" class="fig-bold">k</text>
<path d="M412.4,139.6 A38,38 0 0 0 460.8,181.4" class="stroke-ink"/>
<text x="399.4" y="191.4" class="fig-small">gap 2</text>
<text x="450" y="271" text-anchor="middle" class="fig-mono">score = cos 2 = -0.4161</text>
</svg>
Figure 1. A query (blue) and a key (orange) turned by their positions. Notice that the angle between them is the gap between the positions, so moving both tokens by two places leaves the angle, and the score, unchanged.
```

Shift both tokens by the same amount, say two places later, and both arrows turn by the same extra angle. The angle between them does not change, so the score does not change. That is the whole idea.

A worked example makes it concrete. Take speed $\omega = 1$ and let the query and key both start as $[1, 0]$. Turned to positions $m$ and $n$, they become $[\cos m, \sin m]$ and $[\cos n, \sin n]$. Their dot product is $\cos m \cos n + \sin m \sin n = \cos(n - m)$.

- Query at position 0 and key at position 1: the query stays $[1, 0]$, the key is $[0.5403, 0.8415]$, and the score is $0.5403$.
- Query at position 1 and key at position 2: the query is $[0.5403, 0.8415]$ and the key is $[-0.4161, 0.9093]$. The score is $0.5403 \times (-0.4161) + 0.8415 \times 0.9093 = -0.2248 + 0.7651 = 0.5403$.

The two pairs sit one place apart, and both score $\cos 1 = 0.5403$. In Figure 1 the gap is 2, and the score is $\cos 2 = -0.4161$, negative because the arrows are more than a right angle apart.

The reason fits in one line. Write $R(a)$ for the turn by angle $a$, the matrix $\begin{bmatrix} \cos a & -\sin a \\ \sin a & \cos a \end{bmatrix}$. Undoing the turn by $m\omega$ and then turning by $n\omega$ leaves a net turn of $(n - m)\omega$, so $R(m\omega)^\top R(n\omega) = R((n - m)\omega)$. Then

$$
\big(R(m\omega)\,q\big)\cdot\big(R(n\omega)\,k\big)
= q^\top R(m\omega)^\top R(n\omega)\,k
= q^\top R\big((n - m)\omega\big)\,k
$$

Read it aloud: the score between the turned query and the turned key equals the score between the plain query and a key turned by the distance. The positions $m$ and $n$ only ever appear as $n - m$.

## From one plane to many

A real attention head has a query of $d$ entries, for example 8 or 64, not 2. Split the entries into $d/2$ pairs. Each pair is its own plane, and each plane gets its own speed, $\omega_j = \theta^{-2j/d}$ for pair $j$. The Greek letter theta, $\theta$, is the base, and its usual value is 10000. The reason for different speeds is the one from the clocks: fast planes tell neighbours apart and slow planes tell distant tokens apart. Figure 2 turns one 8-entry vector to position 5.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Four planes, one for each pair of entries of a width 8 vector, each showing an arrow turned by position 5 at its own speed">
<defs><marker id="c102-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c102-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c102-arrow-m" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<text x="82" y="28" text-anchor="middle" class="fig-bold">pair (0, 4)</text>
<text x="82" y="46" text-anchor="middle" class="fig-small">speed ω = 1</text>
<circle cx="82" cy="120" r="54" class="stroke-grid"/>
<line x1="24" y1="120" x2="140" y2="120" class="stroke-grid"/>
<line x1="82" y1="62" x2="82" y2="178" class="stroke-grid"/>
<line x1="82" y1="120" x2="134.0" y2="120.0" class="stroke-muted dashed"/>
<line x1="82" y1="120" x2="96.8" y2="169.9" class="stroke-b" marker-end="url(#c102-arrow-b)"/>
<path d="M104.0,120.0 A22,22 0 1 0 88.2,141.1" class="stroke-ink"/>
<text x="82" y="196" text-anchor="middle" class="fig-small">turned 5 rad</text>
<text x="227" y="28" text-anchor="middle" class="fig-bold">pair (1, 5)</text>
<text x="227" y="46" text-anchor="middle" class="fig-small">speed ω = 0.1</text>
<circle cx="227" cy="120" r="54" class="stroke-grid"/>
<line x1="169" y1="120" x2="285" y2="120" class="stroke-grid"/>
<line x1="227" y1="62" x2="227" y2="178" class="stroke-grid"/>
<line x1="227" y1="120" x2="279.0" y2="120.0" class="stroke-muted dashed"/>
<line x1="227" y1="120" x2="272.6" y2="95.1" class="stroke-b" marker-end="url(#c102-arrow-b)"/>
<path d="M249.0,120.0 A22,22 0 0 0 246.3,109.5" class="stroke-ink"/>
<text x="227" y="196" text-anchor="middle" class="fig-small">turned 0.5 rad</text>
<text x="372" y="28" text-anchor="middle" class="fig-bold">pair (2, 6)</text>
<text x="372" y="46" text-anchor="middle" class="fig-small">speed ω = 0.01</text>
<circle cx="372" cy="120" r="54" class="stroke-grid"/>
<line x1="314" y1="120" x2="430" y2="120" class="stroke-grid"/>
<line x1="372" y1="62" x2="372" y2="178" class="stroke-grid"/>
<line x1="372" y1="120" x2="424.0" y2="120.0" class="stroke-muted dashed"/>
<line x1="372" y1="120" x2="423.9" y2="117.4" class="stroke-b" marker-end="url(#c102-arrow-b)"/>
<text x="372" y="196" text-anchor="middle" class="fig-small">turned 0.05 rad</text>
<text x="517" y="28" text-anchor="middle" class="fig-bold">pair (3, 7)</text>
<text x="517" y="46" text-anchor="middle" class="fig-small">speed ω = 0.001</text>
<circle cx="517" cy="120" r="54" class="stroke-grid"/>
<line x1="459" y1="120" x2="575" y2="120" class="stroke-grid"/>
<line x1="517" y1="62" x2="517" y2="178" class="stroke-grid"/>
<line x1="517" y1="120" x2="569.0" y2="120.0" class="stroke-muted dashed"/>
<line x1="517" y1="120" x2="569.0" y2="119.7" class="stroke-b" marker-end="url(#c102-arrow-b)"/>
<text x="517" y="196" text-anchor="middle" class="fig-small">turned 0.005 rad</text>
<text x="300" y="250" text-anchor="middle" class="fig-small">Dashed: the pair before turning. Orange: the pair after position 5 turns it by 5ω.</text>
</svg>
Figure 2. One 8-entry vector at position 5, drawn as four planes. Notice that one position turns every plane, at four different speeds: the first pair spins almost a full turn, while the last barely moves.
```

The full score is the sum of the pairs' dot products. Each pair's term depends only on $n - m$, so the sum does too.

Which entries form a pair? Any fixed choice works, because the argument above does not care. The layout used here, called rotate-half, pairs entry $j$ with entry $j + d/2$. At $d = 8$ the pairs are $(0, 4)$, $(1, 5)$, $(2, 6)$, and $(3, 7)$, with speeds 1, 0.1, 0.01, and 0.001. This layout turns every pair with two slices and one concatenation, where pairing neighbours would need strided indexing. It differs from the previous chapter's layout, which put sine and cosine in neighbouring columns, so do not mix the two.

## The formula and the code

Split a vector $x$ into its first half $x_1$ and its second half $x_2$, so pair $j$ is $(x_1[j], x_2[j])$. Let $\cos$ and $\sin$ hold $\cos(p\omega_j)$ and $\sin(p\omega_j)$ for the token's position $p$, with each speed stored twice so that the tables match the width of $x$. Then

$$
\operatorname{RoPE}(x) = x \odot \cos + \operatorname{rotate\_half}(x) \odot \sin,
\qquad \operatorname{rotate\_half}(x) = [-x_2,\; x_1]
$$

The symbol $\odot$ means multiply entry by entry. Read it aloud: for pair $(a, b)$ the first entry becomes $a\cos - b\sin$ and the second becomes $a\sin + b\cos$, which is exactly the turn from the Figure 1 picture. Check it at width 2 with $x = [1, 0]$ at position 1: $x \odot \cos = [0.5403, 0]$ and $\operatorname{rotate\_half}(x) \odot \sin = [0, 0.8415]$, so the result is $[0.5403, 0.8415]$, the key from the worked example. In numpy:

```python
freqs = theta ** (-2 * np.arange(d // 2) / d)        # (d/2,) one speed per pair
angles = np.arange(T)[:, None] * freqs[None, :]      # (T, d/2) angle for each position and pair
cos = np.concatenate([np.cos(angles)] * 2, axis=-1)  # (T, d), each speed stored twice
sin = np.concatenate([np.sin(angles)] * 2, axis=-1)

half = x.shape[-1] // 2
rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
out = x * cos + rot * sin
```

The tables have shape `(T, d)`. They broadcast against a single vector, a `(T, d)` sequence, or a `(heads, T, d)` stack, so one call turns every head at once.

Apply the rotation to queries and keys, and leave the values alone. The score is the only place where distance should matter. Values carry the content that the weighted average blends into the output, and rotating them would turn that content by an amount that depends on the absolute position, with nothing later to cancel it.

Figure 3 tests the claim on random vectors. For one random query and key of width 16, the curve is the score as the offset $n - m$ changes. The dots come from the real rotation at three different absolute positions: $m = 20$, 50, and 120. Every dot lands on the curve.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="The rotated query-key score for random vectors against the offset n minus m, with points from three different absolute positions all on one curve">
<line x1="70" y1="240.0" x2="560" y2="240.0" class="stroke-grid"/>
<text x="62" y="244.0" text-anchor="end" class="fig-small">-5</text>
<line x1="70" y1="217.8" x2="560" y2="217.8" class="stroke-grid"/>
<text x="62" y="221.8" text-anchor="end" class="fig-small">-4</text>
<line x1="70" y1="195.6" x2="560" y2="195.6" class="stroke-grid"/>
<text x="62" y="199.6" text-anchor="end" class="fig-small">-3</text>
<line x1="70" y1="173.3" x2="560" y2="173.3" class="stroke-grid"/>
<text x="62" y="177.3" text-anchor="end" class="fig-small">-2</text>
<line x1="70" y1="151.1" x2="560" y2="151.1" class="stroke-grid"/>
<text x="62" y="155.1" text-anchor="end" class="fig-small">-1</text>
<line x1="70" y1="128.9" x2="560" y2="128.9" class="stroke-grid"/>
<text x="62" y="132.9" text-anchor="end" class="fig-small">0</text>
<line x1="70" y1="106.7" x2="560" y2="106.7" class="stroke-grid"/>
<text x="62" y="110.7" text-anchor="end" class="fig-small">1</text>
<line x1="70" y1="84.4" x2="560" y2="84.4" class="stroke-grid"/>
<text x="62" y="88.4" text-anchor="end" class="fig-small">2</text>
<line x1="70" y1="62.2" x2="560" y2="62.2" class="stroke-grid"/>
<text x="62" y="66.2" text-anchor="end" class="fig-small">3</text>
<line x1="70" y1="40.0" x2="560" y2="40.0" class="stroke-grid"/>
<text x="62" y="44.0" text-anchor="end" class="fig-small">4</text>
<line x1="315.0" y1="40" x2="315.0" y2="240" class="stroke-grid dashed"/>
<polyline points="70.0,122.8 73.8,126.9 77.7,129.6 81.5,130.9 85.3,130.9 89.1,129.9 93.0,128.1 96.8,125.8 100.6,123.5 104.5,121.4 108.3,119.9 112.1,119.4 115.9,120.1 119.8,122.0 123.6,125.5 127.4,130.3 131.2,136.4 135.1,143.7 138.9,151.7 142.7,160.2 146.6,168.8 150.4,177.0 154.2,184.6 158.0,191.0 161.9,196.0 165.7,199.3 169.5,200.8 173.4,200.3 177.2,198.0 181.0,194.0 184.8,188.4 188.7,181.7 192.5,174.0 196.3,165.9 200.2,157.7 204.0,149.8 207.8,142.6 211.6,136.4 215.5,131.4 219.3,127.7 223.1,125.4 227.0,124.4 230.8,124.6 234.6,125.8 238.4,127.6 242.3,129.8 246.1,131.9 249.9,133.6 253.8,134.5 257.6,134.4 261.4,133.0 265.2,130.3 269.1,126.0 272.9,120.4 276.7,113.5 280.5,105.5 284.4,96.9 288.2,87.8 292.0,78.8 295.9,70.2 299.7,62.5 303.5,56.0 307.3,51.0 311.2,47.7 315.0,46.4 318.8,46.9 322.7,49.3 326.5,53.4 330.3,59.0 334.1,65.8 338.0,73.3 341.8,81.2 345.6,89.2 349.5,96.7 353.3,103.5 357.1,109.4 360.9,114.0 364.8,117.3 368.6,119.3 372.4,120.1 376.2,119.9 380.1,118.8 383.9,117.3 387.7,115.6 391.6,114.2 395.4,113.3 399.2,113.4 403.0,114.6 406.9,117.3 410.7,121.4 414.5,127.1 418.4,134.2 422.2,142.6 426.0,152.0 429.8,162.1 433.7,172.6 437.5,182.9 441.3,192.7 445.2,201.6 449.0,209.2 452.8,215.3 456.6,219.7 460.5,222.1 464.3,222.7 468.1,221.5 472.0,218.6 475.8,214.4 479.6,209.1 483.4,203.2 487.3,197.0 491.1,190.9 494.9,185.3 498.8,180.4 502.6,176.7 506.4,174.1 510.2,172.8 514.1,172.8 517.9,173.9 521.7,176.0 525.5,178.8 529.4,181.8 533.2,184.9 537.0,187.6 540.9,189.5 544.7,190.3 548.5,189.9 552.3,187.9 556.2,184.4 560.0,179.3" class="stroke-ink"/>
<circle cx="70.0" cy="122.8" r="6.5" class="stroke-a"/>
<circle cx="85.3" cy="130.9" r="6.5" class="stroke-a"/>
<circle cx="100.6" cy="123.5" r="6.5" class="stroke-a"/>
<circle cx="115.9" cy="120.1" r="6.5" class="stroke-a"/>
<circle cx="131.2" cy="136.4" r="6.5" class="stroke-a"/>
<circle cx="146.6" cy="168.8" r="6.5" class="stroke-a"/>
<circle cx="161.9" cy="196.0" r="6.5" class="stroke-a"/>
<circle cx="177.2" cy="198.0" r="6.5" class="stroke-a"/>
<circle cx="192.5" cy="174.0" r="6.5" class="stroke-a"/>
<circle cx="207.8" cy="142.6" r="6.5" class="stroke-a"/>
<circle cx="223.1" cy="125.4" r="6.5" class="stroke-a"/>
<circle cx="238.4" cy="127.6" r="6.5" class="stroke-a"/>
<circle cx="253.8" cy="134.5" r="6.5" class="stroke-a"/>
<circle cx="269.1" cy="126.0" r="6.5" class="stroke-a"/>
<circle cx="284.4" cy="96.9" r="6.5" class="stroke-a"/>
<circle cx="299.7" cy="62.5" r="6.5" class="stroke-a"/>
<circle cx="315.0" cy="46.4" r="6.5" class="stroke-a"/>
<circle cx="330.3" cy="59.0" r="6.5" class="stroke-a"/>
<circle cx="345.6" cy="89.2" r="6.5" class="stroke-a"/>
<circle cx="360.9" cy="114.0" r="6.5" class="stroke-a"/>
<circle cx="376.2" cy="119.9" r="6.5" class="stroke-a"/>
<circle cx="391.6" cy="114.2" r="6.5" class="stroke-a"/>
<circle cx="406.9" cy="117.3" r="6.5" class="stroke-a"/>
<circle cx="422.2" cy="142.6" r="6.5" class="stroke-a"/>
<circle cx="437.5" cy="182.9" r="6.5" class="stroke-a"/>
<circle cx="452.8" cy="215.3" r="6.5" class="stroke-a"/>
<circle cx="468.1" cy="221.5" r="6.5" class="stroke-a"/>
<circle cx="483.4" cy="203.2" r="6.5" class="stroke-a"/>
<circle cx="498.8" cy="180.4" r="6.5" class="stroke-a"/>
<circle cx="514.1" cy="172.8" r="6.5" class="stroke-a"/>
<circle cx="529.4" cy="181.8" r="6.5" class="stroke-a"/>
<circle cx="544.7" cy="190.3" r="6.5" class="stroke-a"/>
<circle cx="560.0" cy="179.3" r="6.5" class="stroke-a"/>
<circle cx="70.0" cy="122.8" r="4.5" class="stroke-b"/>
<circle cx="85.3" cy="130.9" r="4.5" class="stroke-b"/>
<circle cx="100.6" cy="123.5" r="4.5" class="stroke-b"/>
<circle cx="115.9" cy="120.1" r="4.5" class="stroke-b"/>
<circle cx="131.2" cy="136.4" r="4.5" class="stroke-b"/>
<circle cx="146.6" cy="168.8" r="4.5" class="stroke-b"/>
<circle cx="161.9" cy="196.0" r="4.5" class="stroke-b"/>
<circle cx="177.2" cy="198.0" r="4.5" class="stroke-b"/>
<circle cx="192.5" cy="174.0" r="4.5" class="stroke-b"/>
<circle cx="207.8" cy="142.6" r="4.5" class="stroke-b"/>
<circle cx="223.1" cy="125.4" r="4.5" class="stroke-b"/>
<circle cx="238.4" cy="127.6" r="4.5" class="stroke-b"/>
<circle cx="253.8" cy="134.5" r="4.5" class="stroke-b"/>
<circle cx="269.1" cy="126.0" r="4.5" class="stroke-b"/>
<circle cx="284.4" cy="96.9" r="4.5" class="stroke-b"/>
<circle cx="299.7" cy="62.5" r="4.5" class="stroke-b"/>
<circle cx="315.0" cy="46.4" r="4.5" class="stroke-b"/>
<circle cx="330.3" cy="59.0" r="4.5" class="stroke-b"/>
<circle cx="345.6" cy="89.2" r="4.5" class="stroke-b"/>
<circle cx="360.9" cy="114.0" r="4.5" class="stroke-b"/>
<circle cx="376.2" cy="119.9" r="4.5" class="stroke-b"/>
<circle cx="391.6" cy="114.2" r="4.5" class="stroke-b"/>
<circle cx="406.9" cy="117.3" r="4.5" class="stroke-b"/>
<circle cx="422.2" cy="142.6" r="4.5" class="stroke-b"/>
<circle cx="437.5" cy="182.9" r="4.5" class="stroke-b"/>
<circle cx="452.8" cy="215.3" r="4.5" class="stroke-b"/>
<circle cx="468.1" cy="221.5" r="4.5" class="stroke-b"/>
<circle cx="483.4" cy="203.2" r="4.5" class="stroke-b"/>
<circle cx="498.8" cy="180.4" r="4.5" class="stroke-b"/>
<circle cx="514.1" cy="172.8" r="4.5" class="stroke-b"/>
<circle cx="529.4" cy="181.8" r="4.5" class="stroke-b"/>
<circle cx="544.7" cy="190.3" r="4.5" class="stroke-b"/>
<circle cx="560.0" cy="179.3" r="4.5" class="stroke-b"/>
<circle cx="70.0" cy="122.8" r="2.2" class="fill-c"/>
<circle cx="85.3" cy="130.9" r="2.2" class="fill-c"/>
<circle cx="100.6" cy="123.5" r="2.2" class="fill-c"/>
<circle cx="115.9" cy="120.1" r="2.2" class="fill-c"/>
<circle cx="131.2" cy="136.4" r="2.2" class="fill-c"/>
<circle cx="146.6" cy="168.8" r="2.2" class="fill-c"/>
<circle cx="161.9" cy="196.0" r="2.2" class="fill-c"/>
<circle cx="177.2" cy="198.0" r="2.2" class="fill-c"/>
<circle cx="192.5" cy="174.0" r="2.2" class="fill-c"/>
<circle cx="207.8" cy="142.6" r="2.2" class="fill-c"/>
<circle cx="223.1" cy="125.4" r="2.2" class="fill-c"/>
<circle cx="238.4" cy="127.6" r="2.2" class="fill-c"/>
<circle cx="253.8" cy="134.5" r="2.2" class="fill-c"/>
<circle cx="269.1" cy="126.0" r="2.2" class="fill-c"/>
<circle cx="284.4" cy="96.9" r="2.2" class="fill-c"/>
<circle cx="299.7" cy="62.5" r="2.2" class="fill-c"/>
<circle cx="315.0" cy="46.4" r="2.2" class="fill-c"/>
<circle cx="330.3" cy="59.0" r="2.2" class="fill-c"/>
<circle cx="345.6" cy="89.2" r="2.2" class="fill-c"/>
<circle cx="360.9" cy="114.0" r="2.2" class="fill-c"/>
<circle cx="376.2" cy="119.9" r="2.2" class="fill-c"/>
<circle cx="391.6" cy="114.2" r="2.2" class="fill-c"/>
<circle cx="406.9" cy="117.3" r="2.2" class="fill-c"/>
<circle cx="422.2" cy="142.6" r="2.2" class="fill-c"/>
<circle cx="437.5" cy="182.9" r="2.2" class="fill-c"/>
<circle cx="452.8" cy="215.3" r="2.2" class="fill-c"/>
<circle cx="468.1" cy="221.5" r="2.2" class="fill-c"/>
<circle cx="483.4" cy="203.2" r="2.2" class="fill-c"/>
<circle cx="498.8" cy="180.4" r="2.2" class="fill-c"/>
<circle cx="514.1" cy="172.8" r="2.2" class="fill-c"/>
<circle cx="529.4" cy="181.8" r="2.2" class="fill-c"/>
<circle cx="544.7" cy="190.3" r="2.2" class="fill-c"/>
<circle cx="560.0" cy="179.3" r="2.2" class="fill-c"/>
<text x="70.0" y="258" text-anchor="middle" class="fig-small">-16</text>
<text x="192.5" y="258" text-anchor="middle" class="fig-small">-8</text>
<text x="315.0" y="258" text-anchor="middle" class="fig-small">0</text>
<text x="437.5" y="258" text-anchor="middle" class="fig-small">8</text>
<text x="560.0" y="258" text-anchor="middle" class="fig-small">16</text>
<text x="315.0" y="276" text-anchor="middle" class="fig-small">offset n − m</text>
<text x="14" y="26" class="fig-small">score q·k</text>
<circle cx="190" cy="288" r="6.5" class="stroke-a"/><text x="202" y="292" class="fig-small">m = 20</text>
<circle cx="300" cy="288" r="4.5" class="stroke-b"/><text x="312" y="292" class="fig-small">m = 50</text>
<circle cx="410" cy="288" r="2.2" class="fill-c"/><text x="422" y="292" class="fig-small">m = 120</text>
</svg>
Figure 3. The score for a random query and key at width 16, against the offset n − m. Notice that the markers from three very different absolute positions (m = 20, 50, and 120) all land on one curve, so the score depends on the offset alone.
```

> [!KEY]
> Turn each query and each key by an angle proportional to its position, and their dot product depends on how far apart the tokens are, not on where they sit. The word "bit" gets the same score for the noun one place before it at the top of a page or fifty words in.

## Where this breaks

**The sign convention does not matter, except when you load a model.** Using $[x_2, -x_1]$ in place of $[-x_2, x_1]$ turns every vector the other way. Queries and keys both turn the other way, so the score still depends only on $n - m$. The convention only has to match the one a pretrained model used, and it must be the same for queries and keys.

**Long sequences show angles the model never saw.** A fast pair wraps around many times in a short text, so all its angles are familiar. A slow pair does not. The slowest pair of the width-8 example turns only $2048 \times 0.001 = 2.05$ radians over 2048 positions. A model trained on 2048 positions has never seen that pair at 8 radians, which is where position 8000 puts it. People handle this by scaling positions down so long sequences land inside the trained range, usually followed by a little more training.

**Rotation is one route to position.** ALiBi takes a different route. It stores nothing, rotates nothing, and subtracts from each score a penalty that grows with the distance.

> [!RECAP]
> Carry these facts into the exercise:
> - Turning a query by $m\omega$ and a key by $n\omega$ makes their dot product depend on $n - m$, because $R(m\omega)^\top R(n\omega) = R((n - m)\omega)$.
> - A head of width $d$ holds $d/2$ planes. Entry $j$ pairs with entry $j + d/2$, and pair $j$ turns at speed $\theta^{-2j/d}$.
> - `rope(x) = x * cos + rotate_half(x) * sin`, with `rotate_half(x) = [-x2, x1]` and each speed stored in both halves of the tables.
> - Rotate queries and keys, never values.

## Build it step by step

You write three functions, in the order the checks test them.

1. **`rope_tables(T, d, theta=10000.0)`** returns a pair `(cos, sin)`, each of shape `(T, d)`. Compute the speeds `theta ** (-2 * np.arange(d // 2) / d)`, take the outer product with `np.arange(T)`, and concatenate the cosine of it with itself along the last axis. The common mistake is to return `(T, d // 2)` tables, which do not line up with `x`.
2. **`apply_rope(x, cos, sin)`** takes `x` with last axis `d` and tables that broadcast against it, and returns the same shape. Build `rotate_half` with `x[..., :half]` so that any leading shape works. The common mistake is `[x2, -x1]`, which turns the other way. The checks use the convention above.
3. **`rope_score(q, k, m, n, theta=10000.0)`** takes two vectors of shape `(d,)` and returns the dot product of `q` turned to position `m` and `k` turned to position `n`, as a Python `float`. Build the tables once with `rope_tables` and pick rows `m` and `n`. The common mistake is to build tables of length `max(m, n)`, which is one row short.

## The exercise

You write `rope_tables`, `apply_rope`, and `rope_score`. The checks confirm the shapes of the tables, that $\cos^2 + \sin^2 = 1$, that row 0 is the identity turn, and that each speed appears in both halves. They compare `apply_rope` with an explicit two-by-two turn of every pair, check that it keeps lengths, and check that it works on a vector, a sequence, and a stack of heads. The last checks confirm that `rope_score` equals $\cos(n - m)$ in the width-2 example, equals the plain dot product when $m = n$, and returns the same value for every pair of positions with the same offset.
