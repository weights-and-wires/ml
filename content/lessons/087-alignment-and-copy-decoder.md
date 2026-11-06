---
title: "Where the decoder looks: the alignment matrix and a copy decoder"
slug: 087-alignment-and-copy-decoder
part: "Part 5 — Vision and sequences"
summary: Stack the attention weights of every decoder step into an alignment matrix, and hand-set a scorer so each step reads exactly the right position.
minutes: 40
gpu: false
objectives:
  - Describe the decoder loop with attention, and how it differs from reading one fixed summary vector.
  - Read an alignment matrix, and say what the identity matrix means for a copy task.
  - Explain why the hand-built copy decoder works, using scores of -10 and -20, and why flipping the sign of v breaks it.
  - Build the alignment matrix, an order-blind summary, and the copy decoder in numpy.
prereqs:
  - 084-attention-as-soft-lookup
  - 085-additive-attention
  - 086-attention-gradients
exercise: 087-alignment-and-copy-decoder
---

# Where the decoder looks: the alignment matrix and a copy decoder

At output step 3, which source word is the decoder reading? One call to attention answers that for one step: its weights say how much each source position counts. Collect the weights from every step and you have the answer for the whole output, in one table.

That table has one row per output step and one column per source position, so it is a `T_y` by `T_x` matrix, the **alignment matrix**. Row $t$ is where the decoder looked while it wrote token $t$. You can read it like a picture, and it needs no labels. A trained translation model never receives a list of which source word goes with which output word, yet its matrix often shows those pairs.

In this chapter you build the matrix, and then you ask a sharper question: can you set the scorer by hand, with no training, so that every step reads exactly the right position? The task is copying. The output must repeat the source in order, so step $t$ should read position $t$, and the ideal matrix is the identity. Building that by hand shows what the weights have to look like for attention to be right, and closes the loop with the order-blind summary from "Why one vector is not enough: attention as a soft lookup".

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Top: an encoder squeezes the source into one narrow vector for the decoder. Bottom: at each step the decoder sends a query to an attention box that reads all encoder states and returns a context">
<defs><marker id="c096-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker><marker id="c096-arrow-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="14.0" y="20.0" class="fig-bold">One summary vector</text>
<rect x="14.0" y="28.0" width="90.0" height="22.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="59.0" y="44.0" text-anchor="middle" class="fig-small">state 1</text>
<line x1="104.0" y1="39.0" x2="226.0" y2="62.0" class="stroke-ink" marker-end="url(#c096-arrow-ink)"/>
<rect x="14.0" y="53.0" width="90.0" height="22.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="59.0" y="69.0" text-anchor="middle" class="fig-small">state 2</text>
<line x1="104.0" y1="64.0" x2="226.0" y2="62.0" class="stroke-ink" marker-end="url(#c096-arrow-ink)"/>
<rect x="14.0" y="78.0" width="90.0" height="22.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="59.0" y="94.0" text-anchor="middle" class="fig-small">state 3</text>
<line x1="104.0" y1="89.0" x2="226.0" y2="62.0" class="stroke-ink" marker-end="url(#c096-arrow-ink)"/>
<rect x="228.0" y="40.0" width="24.0" height="46.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="240.0" y="102.0" text-anchor="middle" class="fig-small">one vector</text>
<line x1="252.0" y1="63.0" x2="300.0" y2="63.0" class="stroke-ink" marker-end="url(#c096-arrow-ink)"/>
<rect x="302.0" y="42.0" width="190.0" height="42.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="397.0" y="68.0" text-anchor="middle" class="fig-small">decoder writes every token</text>
<line x1="14.0" y1="108.0" x2="586.0" y2="108.0" class="stroke-grid"/>
<text x="14.0" y="134.0" class="fig-bold">Attention: a fresh look at every step</text>
<rect x="14.0" y="148.0" width="90.0" height="32.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="59.0" y="169.0" text-anchor="middle" class="fig-small">state 1</text>
<line x1="104.0" y1="164.0" x2="256.0" y2="200.0" class="stroke-ink" marker-end="url(#c096-arrow-ink)"/>
<rect x="14.0" y="188.0" width="90.0" height="32.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="59.0" y="209.0" text-anchor="middle" class="fig-small">state 2</text>
<line x1="104.0" y1="204.0" x2="256.0" y2="200.0" class="stroke-ink" marker-end="url(#c096-arrow-ink)"/>
<rect x="14.0" y="228.0" width="90.0" height="32.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="59.0" y="249.0" text-anchor="middle" class="fig-small">state 3</text>
<line x1="104.0" y1="244.0" x2="256.0" y2="200.0" class="stroke-ink" marker-end="url(#c096-arrow-ink)"/>
<text x="130.0" y="138.0" class="fig-small"></text>
<rect x="258.0" y="170.0" width="120.0" height="60.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="318.0" y="195.0" text-anchor="middle" class="fig-bold">attention</text>
<text x="318.0" y="214.0" text-anchor="middle" class="fig-small">weights + blend</text>
<rect x="440.0" y="170.0" width="140.0" height="60.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="510.0" y="195.0" text-anchor="middle" class="fig-bold">decoder step t</text>
<text x="510.0" y="214.0" text-anchor="middle" class="fig-small">writes token t</text>
<line x1="440.0" y1="190.0" x2="380.0" y2="190.0" class="stroke-b" marker-end="url(#c096-arrow-b)"/>
<text x="410.0" y="182.0" text-anchor="middle" class="fig-small fig-bold">query</text>
<line x1="380.0" y1="214.0" x2="438.0" y2="214.0" class="stroke-ink" marker-end="url(#c096-arrow-ink)"/>
<text x="410.0" y="232.0" text-anchor="middle" class="fig-small fig-bold">context</text>
<text x="380.0" y="262.0" text-anchor="middle" class="fig-small">The query changes at every step, so the context does too.</text>
</svg>
Figure 1. With one summary vector (top) every output token reads the same numbers. With attention (bottom) the decoder builds a different query at each step and gets back a different context. Notice that the arrows from the encoder states are the same at every step, while the query and the context are not.
```

## What you already know

Five ideas carry this chapter:

- Attention is score, softmax, blend, and `attend(scores, values)` returns the context and the weights (see "Why one vector is not enough: attention as a soft lookup").
- The additive scorer is `tanh(W1 @ query + keys @ W2.T) @ v`, and the sign of `v` decides whether more active hidden entries raise or lower a score (see "Additive attention: let a small network score the match").
- A one-hot row has a single 1 and zeros elsewhere, and `np.eye(n)[tokens]` stacks one row per token.
- `np.argmax` returns the index of the largest entry of a vector.
- A tanh of a large number, such as $\tanh(10)$, is within $10^{-8}$ of 1.

## The decoder loop

With one summary vector, the decoder reads the same numbers at every step. With attention, the loop at step $t$ has four moves. The decoder's current state becomes a query. Attention scores that query against every encoder state and returns a context. The decoder uses the context to write token $t$. Its updated state becomes the query for step $t+1$. Figure 1 draws it. The encoder states stay fixed, so the same keys and values are read at every step, and what changes is the query, which is why the context changes.

The exercise leaves out the recurrent decoder, the part that updates a state from step to step, and hands you the queries directly. That isolates the look-back so you can test it on its own.

## The alignment matrix

Stack the weight vector from each step, one per row:

$$
A_{t,i} = \text{weight on source position } i \text{ at output step } t
$$

Each row sums to 1, because each row is a softmax. A cell near 1 means step $t$ read almost only position $i$. For a copy task the target is the identity: step 1 reads position 1, step 2 reads position 2, and so on. For translation the dark cells would form word pairs, and they would cross wherever the two languages order words differently, as in the sketch on the right of Figure 2, where "la maison blanche" reads "the white house" in a different order.

Be clear about what is learned here and what is not. In this chapter you build the weights by hand. A trained model finds them with the backward pass from "Backpropagation through attention", which adjusts $W_1$, $W_2$, and $v$ until the weights put the mass where the loss wants it.

```figure
<svg viewBox="0 0 600 316" role="img" aria-label="Left: a five by five alignment matrix from the copy decoder, dark on the diagonal. Right: a three by three sketch of a translation alignment with a crossing">
<text x="14.0" y="22.0" class="fig-bold">Copy decoder, computed</text>
<text x="172.0" y="50.0" text-anchor="middle" class="fig-small">source position (token)</text>
<text x="84.0" y="66.0" text-anchor="middle" class="fig-small">1 (3)</text>
<text x="128.0" y="66.0" text-anchor="middle" class="fig-small">2 (0)</text>
<text x="172.0" y="66.0" text-anchor="middle" class="fig-small">3 (4)</text>
<text x="216.0" y="66.0" text-anchor="middle" class="fig-small">4 (1)</text>
<text x="260.0" y="66.0" text-anchor="middle" class="fig-small">5 (2)</text>
<text x="56.0" y="96.0" text-anchor="end" class="fig-small">step 1</text>
<rect x="62.0" y="70.0" width="44.0" height="44.0" class="fill-a stroke-paper"/>
<text x="84.0" y="96.0" text-anchor="middle" class="fig-small fig-mono fill-paper">1.000</text>
<rect x="106.0" y="70.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="128.0" y="96.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="150.0" y="70.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="172.0" y="96.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="194.0" y="70.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="216.0" y="96.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="238.0" y="70.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="260.0" y="96.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<text x="56.0" y="140.0" text-anchor="end" class="fig-small">step 2</text>
<rect x="62.0" y="114.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="84.0" y="140.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="106.0" y="114.0" width="44.0" height="44.0" class="fill-a stroke-paper"/>
<text x="128.0" y="140.0" text-anchor="middle" class="fig-small fig-mono fill-paper">1.000</text>
<rect x="150.0" y="114.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="172.0" y="140.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="194.0" y="114.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="216.0" y="140.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="238.0" y="114.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="260.0" y="140.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<text x="56.0" y="184.0" text-anchor="end" class="fig-small">step 3</text>
<rect x="62.0" y="158.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="84.0" y="184.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="106.0" y="158.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="128.0" y="184.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="150.0" y="158.0" width="44.0" height="44.0" class="fill-a stroke-paper"/>
<text x="172.0" y="184.0" text-anchor="middle" class="fig-small fig-mono fill-paper">1.000</text>
<rect x="194.0" y="158.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="216.0" y="184.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="238.0" y="158.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="260.0" y="184.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<text x="56.0" y="228.0" text-anchor="end" class="fig-small">step 4</text>
<rect x="62.0" y="202.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="84.0" y="228.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="106.0" y="202.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="128.0" y="228.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="150.0" y="202.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="172.0" y="228.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="194.0" y="202.0" width="44.0" height="44.0" class="fill-a stroke-paper"/>
<text x="216.0" y="228.0" text-anchor="middle" class="fig-small fig-mono fill-paper">1.000</text>
<rect x="238.0" y="202.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="260.0" y="228.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<text x="56.0" y="272.0" text-anchor="end" class="fig-small">step 5</text>
<rect x="62.0" y="246.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="84.0" y="272.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="106.0" y="246.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="128.0" y="272.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="150.0" y="246.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="172.0" y="272.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="194.0" y="246.0" width="44.0" height="44.0" class="fill-soft-a stroke-paper"/>
<text x="216.0" y="272.0" text-anchor="middle" class="fig-small fig-mono">0.00</text>
<rect x="238.0" y="246.0" width="44.0" height="44.0" class="fill-a stroke-paper"/>
<text x="260.0" y="272.0" text-anchor="middle" class="fig-small fig-mono fill-paper">1.000</text>
<text x="172.0" y="312.0" text-anchor="middle" class="fig-small">cells below 0.00005 show as 0.00</text>
<text x="340.0" y="22.0" class="fig-bold">Translation, sketched</text>
<text x="406.0" y="90.0" text-anchor="middle" class="fig-small">the</text>
<text x="458.0" y="90.0" text-anchor="middle" class="fig-small">white</text>
<text x="510.0" y="90.0" text-anchor="middle" class="fig-small">house</text>
<text x="374.0" y="130.0" text-anchor="end" class="fig-small">la</text>
<rect x="380.0" y="100.0" width="52.0" height="52.0" class="fill-b stroke-paper"/>
<rect x="432.0" y="100.0" width="52.0" height="52.0" class="fill-soft-b stroke-paper"/>
<rect x="484.0" y="100.0" width="52.0" height="52.0" class="fill-soft-b stroke-paper"/>
<text x="374.0" y="182.0" text-anchor="end" class="fig-small">maison</text>
<rect x="380.0" y="152.0" width="52.0" height="52.0" class="fill-soft-b stroke-paper"/>
<rect x="432.0" y="152.0" width="52.0" height="52.0" class="fill-soft-b stroke-paper"/>
<rect x="484.0" y="152.0" width="52.0" height="52.0" class="fill-b stroke-paper"/>
<text x="374.0" y="234.0" text-anchor="end" class="fig-small">blanche</text>
<rect x="380.0" y="204.0" width="52.0" height="52.0" class="fill-soft-b stroke-paper"/>
<rect x="432.0" y="204.0" width="52.0" height="52.0" class="fill-b stroke-paper"/>
<rect x="484.0" y="204.0" width="52.0" height="52.0" class="fill-soft-b stroke-paper"/>
<text x="458.0" y="278.0" text-anchor="middle" class="fig-small">not model output: an illustration</text>
<text x="458.0" y="298.0" text-anchor="middle" class="fig-small">the highlighted cells cross where the</text>
<text x="458.0" y="314.0" text-anchor="middle" class="fig-small">word order differs</text>
</svg>
Figure 2. Each row is the attention weights of one decoder step. In the copy decoder (left, computed by decode_with_attention) every step looks almost entirely at the same position, so the matrix is nearly the identity. Notice the crossing in the sketch: a translation reads positions out of order.
```

## Setting the scorer by hand

You need every query $t$ to give a high score to key $t$ and low scores to the rest, using the additive scorer from the last two chapters. Here is a design with reasons for each choice.

- **Queries and keys are $10\, e_t$.** Position $t$ gets a direction of its own, the $t$-th basis vector $e_t$ (a 1 in slot $t$ and zeros elsewhere), scaled by 10. Query $t$ and key $t$ point the same way, and every other pair is perpendicular.
- **$W_1$ and $W_2$ are the identity.** The sum for query $t$ and key $i$ is then $u = 10\,e_t + 10\,e_i$.
- **The values are one-hot rows of the source tokens.** A context that is nearly one of those rows tells you which token was read.

Trace the sum. When $i = t$, the two terms land in the same slot, so $u = 20\,e_t$, which is one active entry. When $i \ne t$, they land in different slots, so $u$ has two active entries of 10. The scale of 10 is what makes this clean, because $\tanh(10)$ and $\tanh(20)$ are both within $10^{-8}$ of 1. Every active entry of $\tanh(u)$ is exactly 1 for practical purposes, so $\tanh(u)$ counts the active entries.

The vector $v$ then turns that count into a score. With $v = -10$ on every entry, the score is $-10$ times the number of active entries. The matching key has one, so it scores $-10$. Every other key has two, so it scores $-20$. A gap of 10 between scores sends a weight of $e^{-10} \approx 4.5 \times 10^{-5}$ to each wrong position. With five source tokens, four wrong positions hold $1.8 \times 10^{-4}$ in total, and the right position gets $0.99982$. Figure 3 shows step 3.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Two bar panels for decoder step 3: with v equal to minus 10, the matching position gets nearly all the weight; with v equal to plus 10, it gets almost none">
<text x="20.0" y="24.0" class="fig-bold">v = -10 on every entry</text>
<text x="20.0" y="44.0" class="fig-small">scores:</text>
<line x1="20.0" y1="200.0" x2="286.0" y2="200.0" class="stroke-muted"/>
<line x1="30.0" y1="200.0" x2="68.0" y2="200.0" class="stroke-b"/>
<text x="49.0" y="194.0" text-anchor="middle" class="fig-small fig-mono">4.5e-5</text>
<text x="49.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">-20</text>
<text x="49.0" y="218.0" text-anchor="middle" class="fig-small">pos 1</text>
<line x1="82.0" y1="200.0" x2="120.0" y2="200.0" class="stroke-b"/>
<text x="101.0" y="194.0" text-anchor="middle" class="fig-small fig-mono">4.5e-5</text>
<text x="101.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">-20</text>
<text x="101.0" y="218.0" text-anchor="middle" class="fig-small">pos 2</text>
<rect x="134.0" y="95.0" width="38.0" height="105.0" class="fill-b"/>
<text x="153.0" y="89.0" text-anchor="middle" class="fig-small fig-mono">0.99982</text>
<text x="153.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">-10</text>
<text x="153.0" y="218.0" text-anchor="middle" class="fig-small">pos 3</text>
<line x1="186.0" y1="200.0" x2="224.0" y2="200.0" class="stroke-b"/>
<text x="205.0" y="194.0" text-anchor="middle" class="fig-small fig-mono">4.5e-5</text>
<text x="205.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">-20</text>
<text x="205.0" y="218.0" text-anchor="middle" class="fig-small">pos 4</text>
<line x1="238.0" y1="200.0" x2="276.0" y2="200.0" class="stroke-b"/>
<text x="257.0" y="194.0" text-anchor="middle" class="fig-small fig-mono">4.5e-5</text>
<text x="257.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">-20</text>
<text x="257.0" y="218.0" text-anchor="middle" class="fig-small">pos 5</text>
<text x="320.0" y="24.0" class="fig-bold">v = +10 on every entry</text>
<text x="320.0" y="44.0" class="fig-small">scores:</text>
<line x1="320.0" y1="200.0" x2="586.0" y2="200.0" class="stroke-muted"/>
<rect x="330.0" y="173.8" width="38.0" height="26.2" class="fill-soft-b stroke-b"/>
<text x="349.0" y="167.8" text-anchor="middle" class="fig-small fig-mono">0.25</text>
<text x="349.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">20</text>
<text x="349.0" y="218.0" text-anchor="middle" class="fig-small">pos 1</text>
<rect x="382.0" y="173.8" width="38.0" height="26.2" class="fill-soft-b stroke-b"/>
<text x="401.0" y="167.8" text-anchor="middle" class="fig-small fig-mono">0.25</text>
<text x="401.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">20</text>
<text x="401.0" y="218.0" text-anchor="middle" class="fig-small">pos 2</text>
<line x1="434.0" y1="200.0" x2="472.0" y2="200.0" class="stroke-b"/>
<text x="453.0" y="194.0" text-anchor="middle" class="fig-small fig-mono">1.1e-5</text>
<text x="453.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">10</text>
<text x="453.0" y="218.0" text-anchor="middle" class="fig-small">pos 3</text>
<rect x="486.0" y="173.8" width="38.0" height="26.2" class="fill-soft-b stroke-b"/>
<text x="505.0" y="167.8" text-anchor="middle" class="fig-small fig-mono">0.25</text>
<text x="505.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">20</text>
<text x="505.0" y="218.0" text-anchor="middle" class="fig-small">pos 4</text>
<rect x="538.0" y="173.8" width="38.0" height="26.2" class="fill-soft-b stroke-b"/>
<text x="557.0" y="167.8" text-anchor="middle" class="fig-small fig-mono">0.25</text>
<text x="557.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">20</text>
<text x="557.0" y="218.0" text-anchor="middle" class="fig-small">pos 5</text>
<text x="14.0" y="250.0" class="fig-small">Step 3 matches position 3 (darker bar). With v = -10 a match activates one entry and scores -10;</text>
<text x="14.0" y="268.0" class="fig-small">the others activate two entries and score -20. With v = +10 the order flips, so the match loses.</text>
</svg>
Figure 3. Same queries, same keys, opposite sign of v. Notice how few numbers decide the outcome: scores of -10 against -20 put 0.99982 on the right position, and the flipped sign sends the right position's weight down to 1e-5.
```

The sign of $v$ matters. Flip it to $+10$, and the match scores $+10$ while every other key scores $+20$, which is the same "active entries" count in the other direction. The non-matching positions win. The diagonal falls to about $1.1 \times 10^{-5}$, and each of the four other positions gets about $0.25$.

> [!WARNING]
> A positive `v` does not give a weaker copy decoder, it gives the opposite one. With `v_sign=+1`, the decoded tokens no longer equal the source.

The decoder reads the token off the context with `np.argmax(context)`. Because the context is nearly the one-hot row of the token at position $t$, the argmax is that token. In the exercise, `decode_with_attention([3, 0, 4, 1, 2], 5)` returns `[3, 0, 4, 1, 2]`.

## Closing the loop with the opening

The order-blind summary from the first chapter is the mean of the one-hot rows:

```python
one_hot = np.eye(vocab_size)[tokens]     # (T, vocab_size)
summary = one_hot.mean(axis=0)           # (vocab_size,)
```

For `[0, 1, 2]` and `[2, 1, 0]` it is $[1/3, 1/3, 1/3]$ both times, so a decoder reading it cannot recover the order. The attention decoder reads the same two sources and returns `[0, 1, 2]` and `[2, 1, 0]`. Order is kept because it lives in which position each step reads, and the weights say that out loud.

Be fair about what this shows. The hand-built queries $10\,e_t$ already name the position to read at each step, so the decoder had the order handed to it. What the comparison proves is that attention can carry order through, because the encoder keeps one vector per position. It does not prove that a decoder finds the right queries by itself. Training has to teach the decoder to produce them from its own state, and the backward pass from the last chapter is how.

> [!KEY]
> A decoder that attends does not need the encoder to squeeze the source into one vector. Each step gets its own query, reads its own blend of the encoder states, and the alignment matrix shows where it looked. In the copy task, scores of $-10$ against $-20$ give a nearly perfect identity matrix, and the order that an average throws away is still there to read, as long as the queries ask for it.

> [!RECAP]
> Before you build:
> - The alignment matrix has one row per decoder step, each row a softmax, so each row sums to 1.
> - `alignment_matrix` calls `additive_attention` once per query and stacks the contexts and the weights.
> - The copy decoder uses queries and keys `10 * np.eye(T)`, identity `W1` and `W2`, one-hot values, and `v = -10 * np.ones(T)`.
> - A matching key activates one hidden entry and scores `-10`, and the others activate two and score `-20`. Flipping the sign of `v` reverses the winner.
> - `mean_summary` is the mean of the one-hot rows, so it is the same for every ordering of the tokens.

## Build it step by step

The starter already holds `softmax`, `attend`, `additive_scores`, and `additive_attention`. The checks test these functions in order:

1. **`alignment_matrix(queries, keys, values, W1, W2, v)`.** Takes `queries` of shape `(T_y, d_s)` and returns `(contexts, A)` of shapes `(T_y, d_v)` and `(T_y, T_x)`. Loop over the queries, call `additive_attention`, and stack the results with `np.stack`. The usual mistake is passing the whole `queries` array in one call, which treats it as a single query.
2. **`mean_summary(source_tokens, vocab_size)`.** Returns the mean of the one-hot rows, shape `(vocab_size,)`. The usual mistake is averaging over axis 1 instead of axis 0.
3. **`decode_with_attention(source_tokens, vocab_size, v_sign=-1.0)`.** Builds the hand-set parameters described above, calls `alignment_matrix`, and reads each token with `np.argmax`. It returns `(decoded, alignment)`. The usual mistake is a positive `v`.

## Cost and scaling

The total work grows with `T_x * T_y`, because every output step looks at every source position. Double both lengths and the work goes up four times. The per-step work depends on the scorer, and the table counts the multiply-adds that produce the scores. The softmax and the blend add about `T_x * d_v` more at every step, whichever scorer you use. For additive attention, `W2 @ keys` costs `T_x * h * d_h` and does not depend on the decoder's state, so you can compute it once and reuse it at every step. That is why the first row has no `d_h` factor on the `T_x` term.

| Scorer | Work per decoder step |
|---|---|
| Additive, with `W2 @ keys` computed once | `h * d_s + T_x * h` |
| Additive, recomputing `W2 @ keys` | `h * d_s + T_x * h * (d_h + 1)` |
| Dot with a matrix | `d_s * d_h + T_x * d_h` |
| Dot only | `T_x * d_h` |

## Where this breaks

Attention fixes what the decoder can see, and it still fails in familiar ways:

- The decoder can read the same source word at many steps, and the output repeats a phrase. A fix called coverage keeps a running total of the weight each position has received and feeds it to the scorer, so the decoder can learn to move on.
- The encoder still reads one step at a time, and attention does not speed that chain up. The next part replaces it with attention itself.
- Attention only blends what the encoder states hold. A detail the encoder dropped does not come back.
- The work grows with both lengths, so a long input and a long output together get expensive.

## The exercise

You build `alignment_matrix`, `mean_summary`, and `decode_with_attention`. The checks confirm that every row of the alignment matrix sums to 1, that it equals a stack of single `additive_attention` calls, and that `mean_summary` is the same for `[0, 1, 2]` and `[2, 1, 0]`. They also require that `decode_with_attention` copies `[3, 0, 4, 1, 2]` exactly with an alignment matrix within 0.01 of the identity, that `v_sign=+1.0` pushes the diagonal below 0.01, and that the attention decoder tells the two orderings apart while the mean summary cannot.
