---
title: "Attention as a soft lookup, with a dot product or a learned scorer"
slug: 056-attention-as-soft-lookup
part: "Part 5 — Vision and sequences"
summary: A decoder scores every source position, turns the scores into weights with softmax, and reads a blend. Score with a dot product or a small network.
minutes: 70
gpu: false
objectives:
  - Explain why a single summary vector cannot serve a decoder that needs different parts of the input at different steps.
  - Compute attention by hand as a score, a softmax, and a weighted blend of values, and say why softmax is used instead of picking the best match.
  - Score keys with a dot product, with a matrix between the query and the key, and with a small additive network.
  - Explain why the bend in tanh lets the query steer the weights, and why additive attention matches a dot product only for the one query its parameters were built from.
  - Build a stable softmax, the blend, and the three scorers in numpy.
prereqs:
  - 037-softmax-and-whole-network-gradients
  - 052-recurrent-cells-and-bptt
  - 055-lstm-backward-and-the-gru
  - 034-multilayer-perceptrons
exercise: 056-attention-as-soft-lookup
---

# Attention as a soft lookup, with a dot product or a learned scorer

A translation model has two halves. The encoder reads the source sentence. The decoder writes the translation one word at a time. In the plainest design, the encoder hands the decoder a single vector, and the decoder sees nothing else.

Make that vector as simple as possible. Give each word a one-hot row, which is a row of zeros with a single 1 marking which word it is, and average the rows. The sentence "dog bites man" becomes the vector $[1/3, 1/3, 1/3]$. So does "man bites dog". The words are the same and the meanings are opposite, yet the vector is identical, so no decoder behind it could tell them apart.

A real encoder, such as the one in [recurrent cells and backpropagation through time](/lessons/052-recurrent-cells-and-bptt), keeps some order, but it has the same problem in a milder form. Its summary holds a fixed number of values, whether the source has five words or five hundred, and the decoder needs different parts of the source at different steps.

The fix is to stop forcing the encoder to choose. Keep one vector per source position, and let the decoder look back at all of them and pull out what it needs at each step. That look-back is called attention. In this chapter you build its core with a dot-product match, then replace the dot product with a small network that learns what a match is. The next chapter sends gradients through it and reads the weights as a picture of where the decoder looked.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Two sentences with the same three words in different order are each turned into one-hot rows, and both collapse to the same averaged vector">
<defs><marker id="c041-arrow-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="14.0" y="32.0" class="fig-bold">dog bites man</text>
<rect x="14.0" y="40.0" width="70.0" height="46.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="49.0" y="58.0" text-anchor="middle">dog</text>
<text x="49.0" y="76.0" text-anchor="middle" class="fig-small fig-mono">[1,0,0]</text>
<rect x="94.0" y="40.0" width="70.0" height="46.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="129.0" y="58.0" text-anchor="middle">bites</text>
<text x="129.0" y="76.0" text-anchor="middle" class="fig-small fig-mono">[0,1,0]</text>
<rect x="174.0" y="40.0" width="70.0" height="46.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="209.0" y="58.0" text-anchor="middle">man</text>
<text x="209.0" y="76.0" text-anchor="middle" class="fig-small fig-mono">[0,0,1]</text>
<line x1="254.0" y1="63.0" x2="296.0" y2="63.0" class="stroke-ink" marker-end="url(#c041-arrow-ink)"/>
<rect x="298.0" y="40.0" width="150.0" height="46.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="373.0" y="58.0" text-anchor="middle" class="fig-small">mean of the rows</text>
<text x="373.0" y="76.0" text-anchor="middle" class="fig-small fig-mono">[0.33, 0.33, 0.33]</text>
<line x1="448.0" y1="63.0" x2="486.0" y2="63.0" class="stroke-ink" marker-end="url(#c041-arrow-ink)"/>
<rect x="488.0" y="40.0" width="98.0" height="46.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="537.0" y="68.0" text-anchor="middle">decoder</text>
<text x="14.0" y="142.0" class="fig-bold">man bites dog</text>
<rect x="14.0" y="150.0" width="70.0" height="46.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="49.0" y="168.0" text-anchor="middle">man</text>
<text x="49.0" y="186.0" text-anchor="middle" class="fig-small fig-mono">[0,0,1]</text>
<rect x="94.0" y="150.0" width="70.0" height="46.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="129.0" y="168.0" text-anchor="middle">bites</text>
<text x="129.0" y="186.0" text-anchor="middle" class="fig-small fig-mono">[0,1,0]</text>
<rect x="174.0" y="150.0" width="70.0" height="46.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="209.0" y="168.0" text-anchor="middle">dog</text>
<text x="209.0" y="186.0" text-anchor="middle" class="fig-small fig-mono">[1,0,0]</text>
<line x1="254.0" y1="173.0" x2="296.0" y2="173.0" class="stroke-ink" marker-end="url(#c041-arrow-ink)"/>
<rect x="298.0" y="150.0" width="150.0" height="46.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="373.0" y="168.0" text-anchor="middle" class="fig-small">mean of the rows</text>
<text x="373.0" y="186.0" text-anchor="middle" class="fig-small fig-mono">[0.33, 0.33, 0.33]</text>
<line x1="448.0" y1="173.0" x2="486.0" y2="173.0" class="stroke-ink" marker-end="url(#c041-arrow-ink)"/>
<rect x="488.0" y="150.0" width="98.0" height="46.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="537.0" y="178.0" text-anchor="middle">decoder</text>
<text x="373.0" y="122.0" text-anchor="middle" class="fig-bold">identical</text>
<text x="14.0" y="232.0" class="fig-small">The decoder receives the same three numbers in both cases, so it cannot say which word came first.</text>
</svg>
Figure 1. Averaging one-hot rows keeps which words appeared and throws away where they stood. Notice that both rows end in the same orange box, so no decoder behind it can tell the two sentences apart.
```

## What you already know

Five ideas carry this chapter:

- A dot product multiplies matching entries of two vectors and adds them. It is large when the vectors point the same way and zero when they are perpendicular.
- A softmax turns a list of scores into positive weights that sum to 1 (see [softmax and cross-entropy](/lessons/037-softmax-and-whole-network-gradients)).
- A weighted average multiplies each vector by its weight and adds the results. A `(T, d)` array times a `(d,)` vector gives one dot product per row.
- An encoder turns each source position into a vector, its hidden state.
- A linear layer computes `W @ x`, and a bend such as tanh turns a stack of them into a network (see [the multilayer perceptron](/lessons/034-multilayer-perceptrons)). The function tanh squeezes any number into the range from $-1$ to $1$, is close to a straight line near 0, and flattens out for large inputs.

## A lookup that returns a blend

Picture a library desk. You hand over a request. Each book on the shelf has a label on its spine, and the pages inside are what you take home. A strict librarian compares your request to the labels, finds the one that matches exactly, and hands you that single book. A Python dictionary works like this: `books["cooking"]` returns one value for the key that equals the request.

Attention changes one thing. The librarian hands you a blend of every book, taking most from the spines that match your request best and a little from the rest. Three names fit this picture:

- The **query** is the request. In a translator it is the decoder's state at the current step, which encodes what it needs next.
- The **keys** are the spine labels. You compare the query against each key to decide how well it matches.
- The **values** are the pages. You take away a blend of them.

In a translator, the keys and the values are usually the same thing, the encoder's state at each source position. They get two names because a model can match on one view of a position and read another.

Work one case with two source positions. The query is $[1, 0]$. The keys are $[1, 0]$ and $[0, 1]$. The values are $[10, 0]$ and $[0, 10]$. There are three steps, and Figure 2 draws all of them.

1. **Score.** The dot product with key 1 is $1 \cdot 1 + 0 \cdot 0 = 1$. With key 2 it is $0$. A high score means a good match.
2. **Weight.** The softmax needs $e^1 = 2.718$ and $e^0 = 1$. These sum to $3.718$, so the weights are $2.718 / 3.718 = 0.7311$ and $1 / 3.718 = 0.2689$.
3. **Blend.** The result is $0.7311 \cdot [10, 0] + 0.2689 \cdot [0, 10] = [7.311, 2.689]$. This vector is the **context**.

A hard lookup would return $[10, 0]$. The soft lookup returns mostly that, with 27% of the other value mixed in.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Left: scores 1 and 0 for two keys become weights 0.7311 and 0.2689. Right: the two values and the context between them on a line">
<text x="14.0" y="22.0" class="fig-bold">query = [1, 0]</text>
<text x="14.0" y="52.0" class="fig-small">scores (query · key)</text>
<text x="14.0" y="76.0">key 1</text>
<rect x="70.0" y="62.0" width="110.0" height="20.0" class="fill-a"/>
<text x="188.0" y="77.0" class="fig-mono">1.0</text>
<text x="14.0" y="106.0">key 2</text>
<line x1="70.0" y1="92.0" x2="70.0" y2="112.0" class="stroke-a"/>
<text x="78.0" y="107.0" class="fig-mono">0.0</text>
<text x="14.0" y="152.0" class="fig-small">weights (softmax of the scores)</text>
<text x="14.0" y="176.0">key 1</text>
<rect x="70.0" y="162.0" width="131.6" height="20.0" class="fill-b"/>
<text x="209.6" y="177.0" class="fig-mono">0.7311</text>
<text x="14.0" y="206.0">key 2</text>
<rect x="70.0" y="192.0" width="48.4" height="20.0" class="fill-b"/>
<text x="126.4" y="207.0" class="fig-mono">0.2689</text>
<text x="14.0" y="240.0" class="fig-small">keys [1, 0] and [0, 1]</text>
<text x="14.0" y="258.0" class="fig-small">values [10, 0] and [0, 10]</text>
<line x1="350.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-muted"/>
<line x1="350.0" y1="250.0" x2="350.0" y2="40.0" class="stroke-muted"/>
<line x1="350.0" y1="250.0" x2="350.0" y2="255.0" class="stroke-muted"/>
<text x="350.0" y="270.0" text-anchor="middle" class="fig-small">0</text>
<line x1="350.0" y1="250.0" x2="345.0" y2="250.0" class="stroke-muted"/>
<text x="340.0" y="254.0" text-anchor="end" class="fig-small">0</text>
<line x1="450.0" y1="250.0" x2="450.0" y2="255.0" class="stroke-muted"/>
<text x="450.0" y="270.0" text-anchor="middle" class="fig-small">5</text>
<line x1="350.0" y1="150.0" x2="345.0" y2="150.0" class="stroke-muted"/>
<text x="340.0" y="154.0" text-anchor="end" class="fig-small">5</text>
<line x1="550.0" y1="250.0" x2="550.0" y2="255.0" class="stroke-muted"/>
<text x="550.0" y="270.0" text-anchor="middle" class="fig-small">10</text>
<line x1="350.0" y1="50.0" x2="345.0" y2="50.0" class="stroke-muted"/>
<text x="340.0" y="54.0" text-anchor="end" class="fig-small">10</text>
<line x1="550.0" y1="250.0" x2="350.0" y2="50.0" class="stroke-muted dashed"/>
<circle cx="550.0" cy="250.0" r="6" class="fill-a"/>
<circle cx="350.0" cy="50.0" r="6" class="fill-a"/>
<circle cx="496.2" cy="196.2" r="7" class="fill-b"/>
<text x="578.0" y="238.0" text-anchor="end">value 1</text>
<text x="578.0" y="224.0" text-anchor="end" class="fig-small">(10, 0)</text>
<text x="362.0" y="54.0">value 2 (0, 10)</text>
<text x="482.0" y="200.0" text-anchor="end" class="fig-bold">context</text>
<text x="482.0" y="216.0" text-anchor="end" class="fig-bold">(7.311, 2.689)</text>
<text x="505.0" y="288.0" text-anchor="middle" class="fig-small">first entry</text>
<text x="350.0" y="24.0" text-anchor="middle" class="fig-small">second entry</text>
</svg>
Figure 2. Scores 1 and 0 become weights 0.7311 and 0.2689, and the context lands 73% of the way from value 2 to value 1. Notice it sits on the dashed line between the two values, never outside it.
```

## Why not pick the best match?

Taking the key with the top score looks simpler. It fails for two reasons.

The first is training. Picking the largest score changes in jumps. Nudge a score a little and the pick usually stays the same, so the slope is zero and gradient descent learns nothing. Or the pick flips completely, and the slope is undefined. Softmax weights move smoothly when the scores move, so a loss can tell the scorer which way to shift.

The second is that a decoder sometimes needs two positions at once, such as a noun and its adjective. A blend can mix them, and a single pick cannot.

A blend is not too blurry, because the model controls the sharpness by how far apart it puts the scores. Scores $[1, 0]$ give weights $0.731$ and $0.269$, a soft blend. Scores $[3, 0]$ give $0.953$ and $0.047$. Scores $[10, 0]$ put 99.995% of the weight on one position, which is nearly a hard pick, and it still has a slope.

## The formula, read aloud

Write $z_i$ for the score of source position $i$, $a_i$ for its weight, $q$ for the query, $k_i$ for key $i$, and $V_i$ for value row $i$. The whole mechanism is:

$$
z_i = q \cdot k_i, \qquad
a_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad
c = \sum_i a_i V_i
$$

Read it aloud: the score of position $i$ is the dot product of the query with key $i$. Its weight is the exponential of that score divided by the sum of all the exponentials, which makes every weight positive and the weights sum to 1. The context $c$ is each value row scaled by its weight, then added. With scores 1 and 0 you get $[7.311, 2.689]$ again.

Why the exponential, and not each score divided by the total? Scores can be negative, and dividing by a total that is zero or negative gives nonsense weights. The exponential turns any score into a positive number, and it keeps the order, so a higher score always gets a higher weight.

Subtract the largest score before you exponentiate. The weights do not change, because every term in the top and the bottom of the fraction gets multiplied by the same $e^{-\text{max}}$. The scores $[101, 100]$ give exactly the weights of $[1, 0]$, and without the subtraction, $e^{800}$ overflows. In numpy:

```python
exp = np.exp(z - z.max())
weights = exp / exp.sum()
context = weights @ values      # (T,) @ (T, d_v) -> (d_v,)
```

Because the weights are non-negative and sum to 1, the context is a convex combination of the values, which means a blend that cannot leave the range of its inputs. In Figure 2 the context sits on the dashed line between the two values.

> [!KEY]
> Attention is score, softmax, blend. A decoder that gets one vector per source position can put weight on "dog" at one step and on "man" at the next, so the order that a one-vector summary loses is rebuilt from which positions the decoder asks for.

## Scoring with a dot product, and with a matrix

A key that points where the query points scores high, so the dot product is a reasonable measure of a match. It also grows with the length of the vectors, a fact that matters in a [later chapter](/lessons/058-scaled-attention-and-heads).

The plain dot product needs the query and the keys to have the same width, and it treats every coordinate the same way. A decoder state and an encoder state often differ in both width and meaning. So you can put a matrix $W$, shaped `(d_s, d_h)`, between them:

$$
z_i = q^{\top} W k_i
$$

Read it aloud: first $W$ turns key $i$ into something that lives in the query's space, then you take the dot product. The entries of $W$ decide which coordinate of the query should line up with which coordinate of the key, and training can learn them. With $W$ equal to the identity, you recover the plain dot product. For all keys at once, compute `W.T @ query` first, which has shape `(d_h,)`, then take one dot product per key:

```python
scores = keys @ query                 # no matrix
scores = keys @ (W.T @ query)         # with W of shape (d_s, d_h)
```

> [!WARNING]
> Test for a missing matrix with `W is None`. Writing `if W:` raises an error on a numpy array, because an array with many entries has no single true or false value.

## When one fixed measure is not enough

Even with a matrix, the score is one product of the query and the key, so it can only say how much two coordinates move together. A decoder writing a verb may need the subject of the source sentence, a word whose vector is nothing like the decoder's own state. A fixed measure of similarity has no way to say "this other kind of key is the one I want".

So replace the dot product with a tiny network that takes a query and a key and returns a score. The network is learned, so the model decides for itself what a good match looks like. This scorer is called additive attention, from the paper by Bahdanau and colleagues. The dot product with a matrix comes from the paper by Luong and colleagues. The softmax and the blend do not change.

Think of the scorer as a small judge. You hand it a query and one key, and it returns a number. Figure 3 shows the judge on key 2 of the running example. It has four parts, and each has a reason:

1. **W1 projects the query** into a space of width $h$. The width $h$ is your choice, the size of the judge's hidden layer.
2. **W2 projects the key** into the same space. Both then live in one space, even if the query and the key started with different widths.
3. **Add them, then apply tanh.** The sum puts the query and the key in one vector, and the bend lets the judge respond to combinations a straight sum cannot, such as a large first entry together with a small second one.
4. **Dot with $v$** collapses the $h$ numbers into one score. The vector $v$ says how much each hidden unit counts, and its sign says whether a unit raises or lowers the score.

Adding $W_1 q$ and $W_2 k$ is the same as one ordinary layer applied to the query and the key written one after the other. So this scorer is a one-hidden-layer network, with tanh as its bend and $v$ as its output layer. Training can set $W_1$, $W_2$, and $v$, so the match is learned rather than assumed.

```figure
<svg viewBox="0 0 640 250" role="img" aria-label="A scoring network: the query goes through W1, key 2 goes through W2, the results are added, passed through tanh, and dotted with v to give one score">
<defs><marker id="c041-arrow-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="14.0" y="24.0" class="fig-bold">How key 2 is scored in the worked example</text>
<rect x="10.0" y="50.0" width="90.0" height="40.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="55.0" y="67.0" text-anchor="middle" class="fig-small">query</text>
<text x="55.0" y="83.0" text-anchor="middle" class="fig-small fig-mono">[1, 0]</text>
<rect x="10.0" y="150.0" width="90.0" height="40.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="55.0" y="167.0" text-anchor="middle" class="fig-small">key 2</text>
<text x="55.0" y="183.0" text-anchor="middle" class="fig-small fig-mono">[0, 1]</text>
<line x1="100.0" y1="70.0" x2="133.0" y2="70.0" class="stroke-ink" marker-end="url(#c041-arrow-ink)"/>
<rect x="135.0" y="50.0" width="72.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="171.0" y="67.0" text-anchor="middle" class="fig-bold">W1</text>
<text x="171.0" y="83.0" text-anchor="middle" class="fig-small">identity</text>
<line x1="100.0" y1="170.0" x2="133.0" y2="170.0" class="stroke-ink" marker-end="url(#c041-arrow-ink)"/>
<rect x="135.0" y="150.0" width="72.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="171.0" y="167.0" text-anchor="middle" class="fig-bold">W2</text>
<text x="171.0" y="183.0" text-anchor="middle" class="fig-small">identity</text>
<path d="M207,70 L272,70 L272,104" fill="none" class="stroke-ink" marker-end="url(#c041-arrow-ink)"/>
<path d="M207,170 L272,170 L272,136" fill="none" class="stroke-ink" marker-end="url(#c041-arrow-ink)"/>
<text x="240.0" y="62.0" text-anchor="middle" class="fig-small fig-mono">[1, 0]</text>
<text x="240.0" y="190.0" text-anchor="middle" class="fig-small fig-mono">[0, 1]</text>
<circle cx="272.0" cy="120.0" r="14" class="fill-paper stroke-ink"/>
<text x="272.0" y="126.0" text-anchor="middle" class="fig-bold">+</text>
<line x1="286.0" y1="120.0" x2="340.0" y2="120.0" class="stroke-ink" marker-end="url(#c041-arrow-ink)"/>
<text x="310.0" y="111.0" text-anchor="middle" class="fig-small fig-mono">[1, 1]</text>
<rect x="342.0" y="100.0" width="70.0" height="40.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="377.0" y="125.0" text-anchor="middle" class="fig-bold">tanh</text>
<text x="404.0" y="160.0" text-anchor="end" class="fig-small fig-mono">[0.7616, 0.7616]</text>
<line x1="412.0" y1="120.0" x2="444.0" y2="120.0" class="stroke-ink" marker-end="url(#c041-arrow-ink)"/>
<rect x="446.0" y="100.0" width="74.0" height="40.0" rx="6" class="fill-soft-d stroke-d"/>
<text x="483.0" y="117.0" text-anchor="middle" class="fig-small">dot with v</text>
<text x="483.0" y="133.0" text-anchor="middle" class="fig-small fig-mono">v = [1, 1]</text>
<line x1="520.0" y1="120.0" x2="548.0" y2="120.0" class="stroke-ink" marker-end="url(#c041-arrow-ink)"/>
<text x="552.0" y="116.0" class="fig-small">score</text>
<text x="552.0" y="134.0" class="fig-bold fig-mono">1.5232</text>
<text x="14.0" y="228.0" class="fig-small">Key 1 goes through the same boxes: [1, 0] + [1, 0] = [2, 0], tanh gives [0.9640, 0], score 0.9640.</text>
</svg>
Figure 3. The scorer is a one-hidden-layer network whose input is a query and a key. Notice that the same W1, W2, and v score every key, and only the key changes between runs.
```

Why keep the tanh? Take it out and the score becomes $v \cdot W_1 q + v \cdot W_2 k_i$. The first term is the same for every key, and a softmax ignores any amount added to all scores equally, so the query would have no effect on the weights at all. The bend is the only place where the query and the key meet.

### Work one case by hand

Reuse the running example. The query is $[1, 0]$, the keys are $[1, 0]$ and $[0, 1]$, and the values are $[10, 0]$ and $[0, 10]$. Let $W_1$ and $W_2$ be the identity, and let $v = [1, 1]$.

1. **Key 1.** The sum is $[1, 0] + [1, 0] = [2, 0]$. Then $\tanh(2) = 0.9640$ and $\tanh(0) = 0$, so the score is $0.9640$.
2. **Key 2.** The sum is $[1, 0] + [0, 1] = [1, 1]$. Then $\tanh(1) = 0.7616$ twice, so the score is $1.5232$.
3. **Weights.** Subtract the largest score to get $-0.5592$ and $0$. The exponentials are $0.5717$ and $1$, which sum to $1.5717$, so the weights are $0.3637$ and $0.6363$.
4. **Context.** $0.3637 \cdot [10, 0] + 0.6363 \cdot [0, 10] = [3.637, 6.363]$.

The key that matches the query lost, which looks like a bug. It is not. With a positive $v$, this judge adds up how much each hidden unit is active. Key 1 puts one big number into the sum, and tanh flattens it, so $\tanh(2) = 0.9640$ is less than twice $\tanh(1) = 0.7616$. Key 2 spreads its input over two units, and two moderate numbers add up to more than one saturated number. Figure 4 shows the curve.

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
Figure 4. One big entry (tanh of 2) adds less than two moderate ones (tanh of 1 twice), because the curve flattens. Notice that this is why the non-matching key 2 scores higher when v is positive.
```

The sign of $v$ decides what counts as a match. Set $v = [-1, -1]$ and the scores become $-0.9640$ and $-1.5232$, so the weights swap to $0.6363$ and $0.3637$, and key 1 wins. Do not read this as a rule that a negative $v$ finds the matching key. Take the keys $[1, 0]$ and $[-1, 0]$ with the same query and $v = [-1, -1]$. The second key gives a sum of $[0, 0]$ and a score of 0, which beats $-0.9640$, so the key that points opposite to the query wins. What counts as a match depends on $W_1$, $W_2$, and $v$ together, and training chooses them.

### The formula, read aloud

With $q$ for the query and $k_i$ for key $i$, the score is:

$$
z_i = v^{\top} \tanh\left(W_1 q + W_2 k_i\right)
$$

Read it aloud: project the query with $W_1$, project key $i$ with $W_2$, add the two results, bend every entry with tanh, and take a dot product with $v$ to get one number. The shapes are $W_1$: `(h, d_s)`, $W_2$: `(h, d_h)`, and $v$: `(h,)`. To score every key at once, add the one projected query to every row of the projected keys:

```python
u = W1 @ query + keys @ W2.T      # (h,) added to every row of (T, h)
scores = np.tanh(u) @ v           # (T,)
```

> [!TIP]
> Write the shape of each result in a comment, as above. A shape error in attention almost always means that `W1` and `W2` were swapped, or that a `.T` was dropped. `W1` goes with the query.

### Can it imitate a dot product?

For one query, it can. The function tanh is close to a straight line near 0, so $\tanh(x) \approx x$ for small $x$. Choose $W_1 = W_2 = c\,I$ for a small number $c$, and $v = q / c$. Each hidden entry is then $\tanh(c(q_j + k_j))$, and dividing by $c$ through $v$ undoes the shrink:

$$
z_i \approx q \cdot (q + k_i) = q \cdot q + q \cdot k_i
$$

The first term is the same for every key, and a softmax ignores any amount added to all scores equally. So the weights are those of the plain dot product. At $c = 0.001$ they agree to within $10^{-4}$, and at $c = 1$ the saturating tanh gives a different pattern.

Read the construction carefully, though. The vector $v = q / c$ was built from one particular query, and $v$ is a learned parameter that stays fixed while the query changes from step to step. Feed the same parameters a different query $q'$ and the straight-line regime gives $z_i \approx q \cdot q' + q \cdot k_i$. The weights still follow the old $q$ and ignore $q'$. In a test with random vectors, the weights for $q'$ matched those for $q$ to $10^{-7}$ and missed the dot-product weights for $q'$ by 0.59. This is the same fact as the one about removing the tanh: with no bend, the query cannot steer the weights. To follow a changing query, a trained scorer has to work in the bending part of tanh.

Additive attention costs more learned numbers than a dot product, as the table shows:

| Scorer | Score for key $i$ | Learned numbers |
|---|---|---|
| Dot product | $q \cdot k_i$ | none |
| Dot with a matrix | $q^{\top} W k_i$ | `d_s * d_h` |
| Additive | $v^{\top} \tanh(W_1 q + W_2 k_i)$ | `h * (d_s + d_h + 1)` |

> [!KEY]
> Only the scorer changes between the two kinds of attention. A dot product assumes that "similar" means "a match". Additive attention lets $W_1$, $W_2$, and $v$ learn what a match is, and it gets its query-and-key interaction from the bend in tanh.

> [!RECAP]
> Before you build:
> - The scores become weights with a stable softmax, and the context is `weights @ values`, a blend that stays inside the range of the values.
> - A matrix `W` between the query and the key lets them differ in width, and `W is None` gives the plain dot product.
> - Additive attention scores a key with `v @ tanh(W1 @ query + W2 @ key)`. Vectorised, it is `np.tanh(W1 @ query + keys @ W2.T) @ v`, and `W1` goes with the query.
> - A positive `v` can make the non-matching key win because tanh saturates, and the sign of `v` decides what a match means, so no sign rule holds in general.
> - Tiny `W1 = W2 = c * I` with `v = query / c` gives the weights of a plain dot product for that one query only, because without the bend the query cannot steer the weights.

## Build it step by step

The checks test the functions in this order:

1. **`softmax(z)`.** Takes scores of shape `(T,)` and returns weights of shape `(T,)`. Subtract `z.max()` before `np.exp`, then divide by the sum. The usual mistake is forgetting the subtraction, which passes small tests and breaks on scores of 800.
2. **`attend(scores, values)`.** Takes scores `(T,)` and values `(T, d_v)`. It returns `(context, weights)`: the softmax of the scores, and `weights @ values`. This is the shared tail of every attention scorer.
3. **`multiplicative_attention(query, keys, values, W=None)`.** Returns `(context, weights)` from `attend`. It builds the scores as `keys @ query` when `W is None` and `keys @ (W.T @ query)` otherwise. The usual mistake is `keys @ (W @ query)`, which passes when `W` is square and symmetric and fails on a non-square one.
4. **`additive_scores(query, keys, W1, W2, v)`.** Takes `query` `(d_s,)`, `keys` `(T, d_h)`, `W1` `(h, d_s)`, `W2` `(h, d_h)`, and `v` `(h,)`, and returns `(T,)` scores. Build `u` with `W1 @ query + keys @ W2.T`, apply `np.tanh`, then `@ v`. The usual mistake is swapping `W1` and `W2`.
5. **`additive_attention(query, keys, values, W1, W2, v)`.** Returns `(context, weights)`. Call `additive_scores`, then pass the scores to `attend`. The usual mistake is writing the softmax again by hand instead of reusing `attend`.
6. **`emulate_dot_params(query, c)`.** Returns `(W1, W2, v)` equal to `(c * I, c * I, query / c)`, with `I` the identity of the query's width. The usual mistake is forgetting to divide `v` by `c`, which leaves every score tiny and every weight nearly equal.

## Where this breaks

Dot products grow with the width of the vectors, so wide vectors give a softmax that is nearly a hard pick, which the later chapter on scaling fixes. Additive attention scores each key with a tanh layer, so it costs more than a dot product, which is one matrix product with no bend. Its tanh also saturates, as Figure 4 shows, so large pre-activations pass a small slope backward, a fact the [next chapter](/lessons/057-attention-gradients-and-alignment) measures. And because the sign of $v$ is learned, a freshly started scorer can prefer the wrong key at first. Training moves it, which is why the scorer is learned and not fixed.

## The exercise

You build `softmax`, `attend`, `multiplicative_attention`, `additive_scores`, `additive_attention`, and `emulate_dot_params`. The checks confirm that `softmax` survives scores of $\pm 800$, that `attend` reproduces the worked context $[7.311, 2.689]$, and that `multiplicative_attention` matches `query @ W @ keys[i]` for a non-square `W`. They confirm the additive scores $0.9640$ and $1.5232$, the weights $0.3637$ and $0.6363$, and that flipping $v$ swaps them. The last three checks cover `emulate_dot_params`: the shapes, additive weights that match the dot-product weights, and the caveat that a different query gets the same weights as the original one.
