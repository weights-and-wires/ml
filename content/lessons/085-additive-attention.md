---
title: "Additive attention: let a small network score the match"
slug: 085-additive-attention
part: "Part 5 — Vision and sequences"
summary: Score each key with a tiny learned network, so the model learns its own idea of what counts as a match instead of using a fixed dot product.
minutes: 40
gpu: false
objectives:
  - Explain what the dot product cannot do as a scorer, and what a small learned scorer adds.
  - Compute additive attention scores and weights by hand, and say why the non-matching key can win when v is positive.
  - Explain why the bend in tanh is what lets the query affect the weights, and why additive attention matches a dot product only for the one query its parameters were built from.
  - Build the vectorised additive scorer and the attention on top of it in numpy.
prereqs:
  - 052-forward-pass-and-gelu
  - 084-attention-as-soft-lookup
exercise: 085-additive-attention
---

# Additive attention: let a small network score the match

Last chapter you scored each source position with a dot product. That worked on a toy case, but it asks a lot of the vectors. The decoder state and the encoder state have to share a width. Worse, the dot product measures one fixed kind of match, which is "do these two vectors point the same way". A decoder writing a verb may need the subject of the source sentence, a word whose vector is nothing like the decoder's own state. A fixed measure of similarity has no way to say "this other kind of key is the one I want".

You could put a matrix between the query and the key, as you did last chapter. That helps, because the matrix can re-weigh the coordinates, but the score is still one product of the two vectors, so it can only say how much two coordinates move together. This chapter replaces the dot product with a tiny network that takes a query and a key and returns a score. The network is learned, so the model decides for itself what a good match looks like.

This scorer is called additive attention, from the paper by Bahdanau and colleagues. The dot-product scorer with a matrix comes from the paper by Luong and colleagues, which is why you see the names "Bahdanau attention" and "Luong attention" in older writing. The rest of the machinery, the softmax and the blend, does not change.

```figure
<svg viewBox="0 0 640 250" role="img" aria-label="A scoring network: the query goes through W1, key 2 goes through W2, the results are added, passed through tanh, and dotted with v to give one score">
<defs><marker id="c094-arrow-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="14.0" y="24.0" class="fig-bold">How key 2 is scored in the worked example</text>
<rect x="10.0" y="50.0" width="90.0" height="40.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="55.0" y="67.0" text-anchor="middle" class="fig-small">query</text>
<text x="55.0" y="83.0" text-anchor="middle" class="fig-small fig-mono">[1, 0]</text>
<rect x="10.0" y="150.0" width="90.0" height="40.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="55.0" y="167.0" text-anchor="middle" class="fig-small">key 2</text>
<text x="55.0" y="183.0" text-anchor="middle" class="fig-small fig-mono">[0, 1]</text>
<line x1="100.0" y1="70.0" x2="133.0" y2="70.0" class="stroke-ink" marker-end="url(#c094-arrow-ink)"/>
<rect x="135.0" y="50.0" width="72.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="171.0" y="67.0" text-anchor="middle" class="fig-bold">W1</text>
<text x="171.0" y="83.0" text-anchor="middle" class="fig-small">identity</text>
<line x1="100.0" y1="170.0" x2="133.0" y2="170.0" class="stroke-ink" marker-end="url(#c094-arrow-ink)"/>
<rect x="135.0" y="150.0" width="72.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="171.0" y="167.0" text-anchor="middle" class="fig-bold">W2</text>
<text x="171.0" y="183.0" text-anchor="middle" class="fig-small">identity</text>
<path d="M207,70 L272,70 L272,104" fill="none" class="stroke-ink" marker-end="url(#c094-arrow-ink)"/>
<path d="M207,170 L272,170 L272,136" fill="none" class="stroke-ink" marker-end="url(#c094-arrow-ink)"/>
<text x="240.0" y="62.0" text-anchor="middle" class="fig-small fig-mono">[1, 0]</text>
<text x="240.0" y="190.0" text-anchor="middle" class="fig-small fig-mono">[0, 1]</text>
<circle cx="272.0" cy="120.0" r="14" class="fill-paper stroke-ink"/>
<text x="272.0" y="126.0" text-anchor="middle" class="fig-bold">+</text>
<line x1="286.0" y1="120.0" x2="340.0" y2="120.0" class="stroke-ink" marker-end="url(#c094-arrow-ink)"/>
<text x="310.0" y="111.0" text-anchor="middle" class="fig-small fig-mono">[1, 1]</text>
<rect x="342.0" y="100.0" width="70.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="377.0" y="125.0" text-anchor="middle" class="fig-bold">tanh</text>
<text x="404.0" y="160.0" text-anchor="end" class="fig-small fig-mono">[0.7616, 0.7616]</text>
<line x1="412.0" y1="120.0" x2="444.0" y2="120.0" class="stroke-ink" marker-end="url(#c094-arrow-ink)"/>
<rect x="446.0" y="100.0" width="74.0" height="40.0" rx="6" class="fill-soft-d stroke-d"/>
<text x="483.0" y="117.0" text-anchor="middle" class="fig-small">dot with v</text>
<text x="483.0" y="133.0" text-anchor="middle" class="fig-small fig-mono">v = [1, 1]</text>
<line x1="520.0" y1="120.0" x2="548.0" y2="120.0" class="stroke-ink" marker-end="url(#c094-arrow-ink)"/>
<text x="552.0" y="116.0" class="fig-small">score</text>
<text x="552.0" y="134.0" class="fig-bold fig-mono">1.5232</text>
<text x="14.0" y="228.0" class="fig-small">Key 1 goes through the same boxes: [1, 0] + [1, 0] = [2, 0], tanh gives [0.9640, 0], score 0.9640.</text>
</svg>
Figure 1. The scorer is a one-hidden-layer network whose input is a query and a key. Notice that the same W1, W2, and v score every key, and only the key changes between runs.
```

## What you already know

Five ideas carry this chapter:

- Attention is score, softmax, blend: `attend(scores, values)` returns a weighted blend of the values (see "Why one vector is not enough: attention as a soft lookup").
- A linear layer turns a vector into `W @ x`, and a bend such as tanh turns a stack of them into a network (see "Run a network forward, with a sharp or smooth bend").
- The function tanh squeezes any number into the range from $-1$ to $1$, is close to a straight line near 0, and flattens out for large inputs.
- A `(T, d)` array times a `(d,)` vector gives one number per row.
- Adding a `(d,)` vector to a `(T, d)` array adds it to every row.

## Why a network can score better than a dot product

Think of the scorer as a small judge. You hand it a query and one key, and it returns a number. Figure 1 shows the judge on key 2 of the running example. It has four parts, and each has a reason:

1. **W1 projects the query** into a space of width $h$. The width $h$ is your choice, the size of the judge's hidden layer.
2. **W2 projects the key** into the same space. Both then live in one space, even if the query and the key started with different widths.
3. **Add them, then apply tanh.** The sum puts the query and the key in one vector, and the bend lets the judge respond to combinations a straight sum cannot, such as a large first entry together with a small second one.
4. **Dot with $v$** collapses the $h$ numbers into one score. The vector $v$ says how much each hidden unit counts, and its sign says whether a unit raises or lowers the score.

Adding $W_1 q$ and $W_2 k$ is the same as one ordinary layer applied to the query and the key written one after the other. So this scorer is a one-hidden-layer network, with tanh as its bend and $v$ as its output layer. Training can set $W_1$, $W_2$, and $v$, so the match is learned rather than assumed.

Why keep the tanh? Take it out and the score becomes $v \cdot W_1 q + v \cdot W_2 k_i$. The first term is the same for every key, and a softmax ignores any amount added to all scores equally, so the query would have no effect on the weights at all. The bend is the only place where the query and the key meet, and it is what makes the score depend on both together.

## Work one case by hand

Reuse the running example. The query is $[1, 0]$, the keys are $[1, 0]$ and $[0, 1]$, and the values are $[10, 0]$ and $[0, 10]$. Let $W_1$ and $W_2$ be the identity, and let $v = [1, 1]$.

1. **Key 1.** The sum is $[1, 0] + [1, 0] = [2, 0]$. Then $\tanh(2) = 0.9640$ and $\tanh(0) = 0$, so the score is $0.9640 + 0 = 0.9640$.
2. **Key 2.** The sum is $[1, 0] + [0, 1] = [1, 1]$. Then $\tanh(1) = 0.7616$ twice, so the score is $0.7616 + 0.7616 = 1.5232$.
3. **Weights.** Subtract the largest score to get $-0.5592$ and $0$. The exponentials are $0.5717$ and $1$, which sum to $1.5717$, so the weights are $0.3637$ and $0.6363$.
4. **Context.** $0.3637 \cdot [10, 0] + 0.6363 \cdot [0, 10] = [3.637, 6.363]$.

The key that matches the query lost, which looks like a bug. It is not. With a positive $v$, this judge adds up how much each hidden unit is active. Key 1 puts one big number into the sum, and tanh flattens it, so $\tanh(2) = 0.9640$ is less than twice $\tanh(1) = 0.7616$. Key 2 spreads its input over two units, and two moderate numbers add up to more than one saturated number. Figure 2 shows the curve. With a positive $v$ this judge rewards how many units are active, and overlap with the query is not part of it.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Left: the tanh curve with its values at 1 and 2 marked. Right: stacked bars comparing key 1, one entry of 0.9640, with key 2, two entries of 0.7616">
<text x="14.0" y="24.0" class="fig-bold">tanh(x) bends flat</text>
<line x1="40.0" y1="130.0" x2="360.0" y2="130.0" class="stroke-muted"/>
<line x1="200.0" y1="40.0" x2="200.0" y2="220.0" class="stroke-muted"/>
<line x1="44.0" y1="126.0" x2="44.0" y2="134.0" class="stroke-muted"/>
<text x="44.0" y="150.0" text-anchor="middle" class="fig-small">-3</text>
<line x1="96.0" y1="126.0" x2="96.0" y2="134.0" class="stroke-muted"/>
<text x="96.0" y="150.0" text-anchor="middle" class="fig-small">-2</text>
<line x1="148.0" y1="126.0" x2="148.0" y2="134.0" class="stroke-muted"/>
<text x="148.0" y="150.0" text-anchor="middle" class="fig-small">-1</text>
<line x1="252.0" y1="126.0" x2="252.0" y2="134.0" class="stroke-muted"/>
<text x="252.0" y="150.0" text-anchor="middle" class="fig-small">1</text>
<line x1="304.0" y1="126.0" x2="304.0" y2="134.0" class="stroke-muted"/>
<text x="304.0" y="150.0" text-anchor="middle" class="fig-small">2</text>
<line x1="356.0" y1="126.0" x2="356.0" y2="134.0" class="stroke-muted"/>
<text x="356.0" y="150.0" text-anchor="middle" class="fig-small">3</text>
<line x1="196.0" y1="220.0" x2="204.0" y2="220.0" class="stroke-muted"/>
<text x="190.0" y="224.0" text-anchor="end" class="fig-small">-1</text>
<line x1="196.0" y1="40.0" x2="204.0" y2="40.0" class="stroke-muted"/>
<text x="190.0" y="44.0" text-anchor="end" class="fig-small">1</text>
<polyline points="44.0,219.6 46.6,219.5 49.2,219.5 51.8,219.4 54.4,219.3 57.0,219.3 59.6,219.2 62.2,219.1 64.8,219.0 67.4,218.9 70.0,218.8 72.6,218.7 75.2,218.5 77.8,218.4 80.4,218.2 83.0,218.0 85.6,217.8 88.2,217.6 90.8,217.3 93.4,217.1 96.0,216.8 98.6,216.4 101.2,216.1 103.8,215.7 106.4,215.2 109.0,214.7 111.6,214.2 114.2,213.6 116.8,213.0 119.4,212.2 122.0,211.5 124.6,210.6 127.2,209.7 129.8,208.7 132.4,207.6 135.0,206.3 137.6,205.0 140.2,203.6 142.8,202.0 145.4,200.4 148.0,198.5 150.6,196.6 153.2,194.5 155.8,192.2 158.4,189.8 161.0,187.2 163.6,184.4 166.2,181.5 168.8,178.3 171.4,175.0 174.0,171.6 176.6,168.0 179.2,164.2 181.8,160.3 184.4,156.2 187.0,152.0 189.6,147.8 192.2,143.4 194.8,139.0 197.4,134.5 200.0,130.0 202.6,125.5 205.2,121.0 207.8,116.6 210.4,112.2 213.0,108.0 215.6,103.8 218.2,99.7 220.8,95.8 223.4,92.0 226.0,88.4 228.6,85.0 231.2,81.7 233.8,78.5 236.4,75.6 239.0,72.8 241.6,70.2 244.2,67.8 246.8,65.5 249.4,63.4 252.0,61.5 254.6,59.6 257.2,58.0 259.8,56.4 262.4,55.0 265.0,53.7 267.6,52.4 270.2,51.3 272.8,50.3 275.4,49.4 278.0,48.5 280.6,47.8 283.2,47.0 285.8,46.4 288.4,45.8 291.0,45.3 293.6,44.8 296.2,44.3 298.8,43.9 301.4,43.6 304.0,43.2 306.6,42.9 309.2,42.7 311.8,42.4 314.4,42.2 317.0,42.0 319.6,41.8 322.2,41.6 324.8,41.5 327.4,41.3 330.0,41.2 332.6,41.1 335.2,41.0 337.8,40.9 340.4,40.8 343.0,40.7 345.6,40.7 348.2,40.6 350.8,40.5 353.4,40.5 356.0,40.4" fill="none" class="stroke-b"/>
<line x1="252.0" y1="130.0" x2="252.0" y2="61.5" class="stroke-a dashed"/>
<circle cx="252.0" cy="61.5" r="5" class="fill-a"/>
<line x1="304.0" y1="130.0" x2="304.0" y2="43.2" class="stroke-a dashed"/>
<circle cx="304.0" cy="43.2" r="5" class="fill-a"/>
<text x="362.0" y="176.0" text-anchor="end" class="fig-small">blue dots: tanh(1) = 0.7616</text>
<text x="362.0" y="194.0" text-anchor="end" class="fig-small">tanh(2) = 0.9640</text>
<line x1="395.0" y1="250.0" x2="590.0" y2="250.0" class="stroke-muted"/>
<rect x="410.0" y="163.2" width="56.0" height="86.8" class="fill-a"/>
<text x="438.0" y="155.2" text-anchor="middle" class="fig-small fig-mono">0.9640</text>
<rect x="512.0" y="181.5" width="56.0" height="68.5" class="fill-b"/>
<rect x="512.0" y="112.9" width="56.0" height="68.5" class="fill-soft-b stroke-b"/>
<text x="540.0" y="104.9" text-anchor="middle" class="fig-small fig-mono">1.5232</text>
<text x="438.0" y="268.0" text-anchor="middle" class="fig-small">key 1</text>
<text x="438.0" y="284.0" text-anchor="middle" class="fig-small fig-mono">u = [2, 0]</text>
<text x="540.0" y="268.0" text-anchor="middle" class="fig-small">key 2</text>
<text x="540.0" y="284.0" text-anchor="middle" class="fig-small fig-mono">u = [1, 1]</text>
<text x="395.0" y="24.0" class="fig-bold">score with v = [1, 1]</text>
</svg>
Figure 2. One big entry (tanh of 2) adds less than two moderate ones (tanh of 1 twice), because the curve flattens. Notice that this is why the non-matching key 2 scores higher when v is positive.
```

The sign of $v$ decides what counts as a match. Set $v = [-1, -1]$ and the scores become $-0.9640$ and $-1.5232$, so the weights swap to $0.6363$ and $0.3637$, and key 1 wins. Do not read this as a rule that a negative $v$ finds the matching key. Take the keys $[1, 0]$ and $[-1, 0]$ with the same query and $v = [-1, -1]$. The second key gives a sum of $[0, 0]$ and a score of 0, which beats $-0.9640$, so the key that points opposite to the query wins. What counts as a match depends on $W_1$, $W_2$, and $v$ together, and training chooses them.

## The formula, read aloud

With $q$ for the query and $k_i$ for key $i$, the score is:

$$
z_i = v^{\top} \tanh\left(W_1 q + W_2 k_i\right)
$$

Read it aloud: project the query with $W_1$, project key $i$ with $W_2$, add the two results, bend every entry with tanh, and take a dot product with $v$ to get one number. The shapes are $W_1$: `(h, d_s)`, $W_2$: `(h, d_h)`, and $v$: `(h,)`. Everything after the score is as before: softmax the scores to get weights, then blend the values.

To score every key at once, add the one projected query to every row of the projected keys:

```python
u = W1 @ query + keys @ W2.T      # (h,) added to every row of (T, h)
scores = np.tanh(u) @ v           # (T,)
```

> [!TIP]
> Write the shape of each result in a comment, as above. A shape error in attention almost always means that `W1` and `W2` were swapped, or that a `.T` was dropped. `W1` goes with the query.

## Can it imitate a dot product?

For one query, it can. The function tanh is close to a straight line near 0, so $\tanh(x) \approx x$ for small $x$. Choose $W_1 = W_2 = c\,I$ for a small number $c$, and $v = q / c$. Each hidden entry is then $\tanh(c(q_j + k_j))$, and dividing by $c$ through $v$ undoes the shrink:

$$
z_i \approx q \cdot (q + k_i) = q \cdot q + q \cdot k_i
$$

The first term is the same for every key, and a softmax ignores any amount added to all scores equally. So the weights are those of the plain dot product. Figure 3 shows this on five random keys. At $c = 0.001$ the bars agree to five decimal places, and at $c = 1$ the saturating tanh gives a different pattern.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Left: attention weights for two keys under v = [1, 1] and v = [-1, -1]. Right: for five keys, weights from a dot product, from additive attention with a tiny c, and with c = 1">
<text x="14.0" y="24.0" class="fig-bold">Flip v, flip the winner</text>
<line x1="14.0" y1="235.0" x2="250.0" y2="235.0" class="stroke-muted"/>
<rect x="30.0" y="187.7" width="38.0" height="47.3" class="fill-b"/>
<text x="49.0" y="181.7" text-anchor="middle" class="fig-small fig-mono">0.3637</text>
<text x="49.0" y="251.0" text-anchor="middle" class="fig-small">key 1</text>
<rect x="74.0" y="152.3" width="38.0" height="82.7" class="fill-soft-b stroke-b"/>
<text x="93.0" y="146.3" text-anchor="middle" class="fig-small fig-mono">0.6363</text>
<text x="93.0" y="251.0" text-anchor="middle" class="fig-small">key 2</text>
<text x="71.0" y="273.0" text-anchor="middle" class="fig-mono fig-small">v = [1, 1]</text>
<rect x="142.0" y="152.3" width="38.0" height="82.7" class="fill-b"/>
<text x="161.0" y="146.3" text-anchor="middle" class="fig-small fig-mono">0.6363</text>
<text x="161.0" y="251.0" text-anchor="middle" class="fig-small">key 1</text>
<rect x="186.0" y="187.7" width="38.0" height="47.3" class="fill-soft-b stroke-b"/>
<text x="205.0" y="181.7" text-anchor="middle" class="fig-small fig-mono">0.3637</text>
<text x="205.0" y="251.0" text-anchor="middle" class="fig-small">key 2</text>
<text x="183.0" y="273.0" text-anchor="middle" class="fig-mono fig-small">v = [-1, -1]</text>
<text x="285.0" y="24.0" class="fig-bold">Imitating a dot product, for one query</text>
<line x1="285.0" y1="235.0" x2="590.0" y2="235.0" class="stroke-muted"/>
<rect x="292.0" y="182.6" width="15.0" height="52.4" class="fill-a"/>
<rect x="309.0" y="182.6" width="15.0" height="52.4" class="fill-b"/>
<rect x="326.0" y="177.0" width="15.0" height="58.0" class="fill-soft-b stroke-b"/>
<text x="317.0" y="251.0" text-anchor="middle" class="fig-small">key 1</text>
<rect x="352.0" y="220.8" width="15.0" height="14.2" class="fill-a"/>
<rect x="369.0" y="220.8" width="15.0" height="14.2" class="fill-b"/>
<rect x="386.0" y="210.1" width="15.0" height="24.9" class="fill-soft-b stroke-b"/>
<text x="377.0" y="251.0" text-anchor="middle" class="fig-small">key 2</text>
<rect x="412.0" y="120.4" width="15.0" height="114.6" class="fill-a"/>
<rect x="429.0" y="120.4" width="15.0" height="114.6" class="fill-b"/>
<rect x="446.0" y="177.9" width="15.0" height="57.1" class="fill-soft-b stroke-b"/>
<text x="437.0" y="251.0" text-anchor="middle" class="fig-small">key 3</text>
<rect x="472.0" y="225.7" width="15.0" height="9.3" class="fill-a"/>
<rect x="489.0" y="225.7" width="15.0" height="9.3" class="fill-b"/>
<rect x="506.0" y="197.3" width="15.0" height="37.7" class="fill-soft-b stroke-b"/>
<text x="497.0" y="251.0" text-anchor="middle" class="fig-small">key 4</text>
<rect x="532.0" y="195.5" width="15.0" height="39.5" class="fill-a"/>
<rect x="549.0" y="195.5" width="15.0" height="39.5" class="fill-b"/>
<rect x="566.0" y="182.6" width="15.0" height="52.4" class="fill-soft-b stroke-b"/>
<text x="557.0" y="251.0" text-anchor="middle" class="fig-small">key 5</text>
<rect x="292.0" y="41.0" width="12.0" height="12.0" class="fill-a"/>
<text x="310.0" y="52.0" class="fig-small">dot product</text>
<rect x="292.0" y="59.0" width="12.0" height="12.0" class="fill-b"/>
<text x="310.0" y="70.0" class="fig-small">additive, c = 0.001</text>
<rect x="292.0" y="77.0" width="12.0" height="12.0" class="fill-soft-b stroke-b"/>
<text x="310.0" y="88.0" class="fig-small">additive, c = 1</text>
<text x="14.0" y="290.0" class="fig-small">Right: random keys with query [1.5, 0.5]. The orange and blue bars match for c = 0.001.</text>
</svg>
Figure 3. Left: with v positive the second key wins, and with v negative the first does. Right: when the hidden layer is shrunk so tanh is nearly straight, the additive weights (orange) land on the dot-product weights (blue) for the query the parameters were built from, and at c = 1 they do not.
```

Read the construction carefully, though. The vector $v = q / c$ was built from one particular query, and $v$ is a learned parameter that stays fixed while the query changes from step to step. Feed the same parameters a different query $q'$ and the straight-line regime gives $z_i \approx q \cdot q' + q \cdot k_i$. The weights still follow the old $q$ and ignore $q'$. In a test with random vectors, the weights for $q'$ matched those for $q$ to $10^{-7}$ and missed the dot-product weights for $q'$ by 0.59. This is the same fact as the one in the last section: with no bend, the query cannot steer the weights. So the construction shows that the two scorers can agree for one query, and it does not show that additive attention contains the dot product. To follow a changing query, a trained scorer has to work in the bending part of tanh.

Additive attention costs more learned numbers than a dot product, as the table shows:

| Scorer | Score for key $i$ | Learned numbers |
|---|---|---|
| Dot product | $q \cdot k_i$ | none |
| Dot with a matrix | $q^{\top} W k_i$ | `d_s * d_h` |
| Additive | $v^{\top} \tanh(W_1 q + W_2 k_i)$ | `h * (d_s + d_h + 1)` |

> [!KEY]
> Attention is always score, softmax, blend, and only the scorer changes. A dot product assumes that "similar" means "a match". Additive attention lets $W_1$, $W_2$, and $v$ learn what a match is, and it gets its query-and-key interaction from the bend in tanh, at the cost of a small network and a tanh for every key.

> [!RECAP]
> Before you build:
> - Additive attention scores a key with `v @ tanh(W1 @ query + W2 @ key)`, a one-hidden-layer network on the query and the key.
> - Vectorised, it is `np.tanh(W1 @ query + keys @ W2.T) @ v`, and `W1` goes with the query.
> - A positive `v` can make the non-matching key win because tanh saturates, and the sign of `v` decides what a match means, so no sign rule holds in general.
> - Tiny `W1 = W2 = c * I` with `v = query / c` gives the weights of a plain dot product for that one query only, because without the bend the query cannot steer the weights.
> - The softmax and the blend are the `attend` function from the last chapter.

## Build it step by step

The starter already holds `softmax`, `attend`, and `multiplicative_attention` from the last chapter. The checks test these functions in order:

1. **`additive_scores(query, keys, W1, W2, v)`.** Takes `query` `(d_s,)`, `keys` `(T, d_h)`, `W1` `(h, d_s)`, `W2` `(h, d_h)`, and `v` `(h,)`, and returns `(T,)` scores. Build `u` with `W1 @ query + keys @ W2.T`, apply `np.tanh`, then `@ v`. The usual mistake is swapping `W1` and `W2`.
2. **`additive_attention(query, keys, values, W1, W2, v)`.** Returns `(context, weights)`. Call `additive_scores`, then pass the scores to `attend`. The usual mistake is writing the softmax again by hand instead of reusing `attend`.
3. **`emulate_dot_params(query, c)`.** Returns `(W1, W2, v)` equal to `(c * I, c * I, query / c)`, with `I` the identity of the query's width. The usual mistake is forgetting to divide `v` by `c`, which leaves every score tiny and every weight nearly equal.

## Where this breaks

Additive attention scores each key with a tanh layer, so it costs more than a dot product, which is one matrix product with no bend. Its tanh also saturates, as Figure 2 shows, so large pre-activations pass a small slope backward, a fact the next chapter measures. And because the sign of $v$ is learned, a freshly started scorer can prefer the wrong key at first. Training moves it, which is why the scorer is learned and not fixed.

## The exercise

You build `additive_scores`, `additive_attention`, and `emulate_dot_params`. The checks confirm that the scores on the worked example are $0.9640$ and $1.5232$, and that the vectorised scores equal a loop over keys with different widths for the query and the keys. They check that the weights are $0.3637$ and $0.6363$, that flipping $v$ swaps them, and that the opposite key $[-1, 0]$ can win. They also require that `emulate_dot_params` has the right shapes and that its additive weights match the plain dot-product weights to within $10^{-4}$. A last check confirms the caveat from the chapter: with those parameters, a different query gets the same weights as the original one, so they no longer follow its dot product.
