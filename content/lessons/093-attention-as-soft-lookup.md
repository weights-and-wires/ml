---
title: "Why one vector is not enough: attention as a soft lookup"
slug: 093-attention-as-soft-lookup
part: "Part 5 — Vision and sequences"
summary: A decoder scores every source position against its current need, turns the scores into weights with softmax, and reads a weighted blend.
minutes: 40
gpu: false
objectives:
  - Explain why a single summary vector cannot serve a decoder that needs different parts of the input at different steps.
  - Compute attention by hand as a score, a softmax, and a weighted blend of values, and say how it differs from a dictionary lookup.
  - Explain why softmax is used instead of picking the best match, and why the blend stays inside the range of the values.
  - Build a stable softmax, the blend, and a dot-product scorer with an optional matrix in numpy.
prereqs:
  - 063-softmax-cross-entropy-gradient
  - 085-recurrent-cell
  - 092-gru-and-parameter-counts
exercise: 093-attention-as-soft-lookup
---

# Why one vector is not enough: attention as a soft lookup

A translation model has two halves. The encoder reads the source sentence. The decoder writes the translation one word at a time. In the plainest design, the encoder hands the decoder a single vector, and the decoder sees nothing else.

Make that vector as simple as possible. Give each word a one-hot row, which is a row of zeros with a single 1 marking which word it is, and average the rows. The sentence "dog bites man" becomes the vector $[1/3, 1/3, 1/3]$. So does "man bites dog". The words are the same and the meanings are opposite, yet the vector is identical, so no decoder behind it could tell them apart.

A real encoder, such as the one in "A network that reads one step at a time", does not average, so it keeps some order. It still has the same problem in a milder form. Its summary holds a fixed number of values, whether the source has five words or five hundred, and the decoder needs different parts of the source at different steps. The first output word needs the first source words, and the fortieth needs something else. One summary has to serve every step.

The fix is to stop forcing the encoder to choose. Keep one vector per source position, and let the decoder look back at all of them and pull out what it needs at each step. That look-back is called attention. In this chapter you build its core, and the next chapters change how the match is scored, send gradients through it, and read the weights as a picture of where the decoder looked.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Two sentences with the same three words in different order are each turned into one-hot rows, and both collapse to the same averaged vector">
<defs><marker id="c093-arrow-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
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
<line x1="254.0" y1="63.0" x2="296.0" y2="63.0" class="stroke-ink" marker-end="url(#c093-arrow-ink)"/>
<rect x="298.0" y="40.0" width="150.0" height="46.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="373.0" y="58.0" text-anchor="middle" class="fig-small">mean of the rows</text>
<text x="373.0" y="76.0" text-anchor="middle" class="fig-small fig-mono">[0.33, 0.33, 0.33]</text>
<line x1="448.0" y1="63.0" x2="486.0" y2="63.0" class="stroke-ink" marker-end="url(#c093-arrow-ink)"/>
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
<line x1="254.0" y1="173.0" x2="296.0" y2="173.0" class="stroke-ink" marker-end="url(#c093-arrow-ink)"/>
<rect x="298.0" y="150.0" width="150.0" height="46.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="373.0" y="168.0" text-anchor="middle" class="fig-small">mean of the rows</text>
<text x="373.0" y="186.0" text-anchor="middle" class="fig-small fig-mono">[0.33, 0.33, 0.33]</text>
<line x1="448.0" y1="173.0" x2="486.0" y2="173.0" class="stroke-ink" marker-end="url(#c093-arrow-ink)"/>
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
- A softmax turns a list of scores into positive weights that sum to 1 (see "Softmax and cross-entropy: a loss with a gradient you can read").
- A weighted average multiplies each vector by its weight and adds the results.
- An encoder turns each source position into a vector, its hidden state (see "A network that reads one step at a time").
- A `(T, d)` array times a `(d,)` vector gives a `(T,)` vector, one dot product per row.

## A lookup that returns a blend

Picture a library desk. You hand over a request. Each book on the shelf has a label on its spine, and the pages inside are what you take home. A strict librarian compares your request to the labels, finds the one that matches exactly, and hands you that single book. A Python dictionary works like this: `books["cooking"]` returns one value for the key that equals the request.

Attention changes one thing. The librarian hands you a blend of every book, taking most from the spines that match your request best and a little from the rest. Three names fit this picture:

- The **query** is the request. In a translator it is the decoder's state at the current step, which encodes what it needs next. The chapter "Where the decoder looks: the alignment matrix and a copy decoder" shows the loop that produces a query at each step.
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

The second is that a decoder sometimes needs two positions at once, such as a word whose translation depends on both a noun and its adjective. A blend can mix them, and a single pick cannot.

You might worry that a blend is too blurry. It is not, because the model controls the sharpness by how far apart it puts the scores. A score gap of 1 gives a soft blend, and a gap of 10 puts 99.995% of the weight on one position, which is nearly a hard pick, yet it still has a slope. Figure 3 shows the effect.

```figure
<svg viewBox="0 0 600 304" role="img" aria-label="Four pairs of bars: the weights from scores 1 and 0, from 101 and 100, from 3 and 0, and from 10 and 0">
<line x1="20.0" y1="220.0" x2="580.0" y2="220.0" class="stroke-muted"/>
<rect x="40.0" y="110.3" width="40.0" height="109.7" class="fill-b"/>
<text x="60.0" y="104.3" text-anchor="middle" class="fig-small fig-mono">0.731</text>
<rect x="86.0" y="179.7" width="40.0" height="40.3" class="fill-soft-b stroke-b"/>
<text x="106.0" y="173.7" text-anchor="middle" class="fig-small fig-mono">0.269</text>
<text x="83.0" y="242.0" text-anchor="middle" class="fig-small">scores</text>
<text x="83.0" y="260.0" text-anchor="middle" class="fig-mono">[1, 0]</text>
<rect x="180.0" y="110.3" width="40.0" height="109.7" class="fill-b"/>
<text x="200.0" y="104.3" text-anchor="middle" class="fig-small fig-mono">0.731</text>
<rect x="226.0" y="179.7" width="40.0" height="40.3" class="fill-soft-b stroke-b"/>
<text x="246.0" y="173.7" text-anchor="middle" class="fig-small fig-mono">0.269</text>
<text x="223.0" y="242.0" text-anchor="middle" class="fig-small">scores</text>
<text x="223.0" y="260.0" text-anchor="middle" class="fig-mono">[101, 100]</text>
<rect x="320.0" y="77.1" width="40.0" height="142.9" class="fill-b"/>
<text x="340.0" y="71.1" text-anchor="middle" class="fig-small fig-mono">0.953</text>
<rect x="366.0" y="212.9" width="40.0" height="7.1" class="fill-soft-b stroke-b"/>
<text x="386.0" y="206.9" text-anchor="middle" class="fig-small fig-mono">0.047</text>
<text x="363.0" y="242.0" text-anchor="middle" class="fig-small">scores</text>
<text x="363.0" y="260.0" text-anchor="middle" class="fig-mono">[3, 0]</text>
<rect x="460.0" y="70.0" width="40.0" height="150.0" class="fill-b"/>
<text x="480.0" y="64.0" text-anchor="middle" class="fig-small fig-mono">1.000</text>
<line x1="506.0" y1="220.0" x2="546.0" y2="220.0" class="stroke-b"/>
<text x="526.0" y="214.0" text-anchor="middle" class="fig-small fig-mono">0.00005</text>
<text x="503.0" y="242.0" text-anchor="middle" class="fig-small">scores</text>
<text x="503.0" y="260.0" text-anchor="middle" class="fig-mono">[10, 0]</text>
<text x="40.0" y="28.0" class="fig-small">Same scores, shifted by 100, give the same weights.</text>
<text x="40.0" y="292.0" class="fig-small">A gap of 1 gives a soft blend. A gap of 10 is almost a hard pick.</text>
</svg>
Figure 3. Each pair of bars is the softmax of the two scores under it. Notice that [1, 0] and [101, 100] match exactly, and that a bigger gap between the scores pushes nearly all the weight onto one position.
```

## The formula, read aloud

Write $z_i$ for the score of source position $i$, $a_i$ for its weight, $q$ for the query, $k_i$ for key $i$, and $V_i$ for value row $i$. The whole mechanism is:

$$
z_i = q \cdot k_i, \qquad
a_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad
c = \sum_i a_i V_i
$$

Read it aloud: the score of position $i$ is the dot product of the query with key $i$. Its weight is the exponential of that score divided by the sum of all the exponentials, which makes every weight positive and the weights sum to 1. The context $c$ is each value row scaled by its weight, then added. With scores 1 and 0 you get $[7.311, 2.689]$ again.

Why the exponential, and not just each score divided by the total? Scores can be negative, and dividing by a total that is zero or negative gives nonsense weights. The exponential turns any score into a positive number, and it keeps the order, so a higher score always gets a higher weight. Subtract the largest score before you exponentiate. The weights do not change, because every term in the top and the bottom of the fraction gets multiplied by the same $e^{-\text{max}}$. The scores $[101, 100]$ give exactly the weights of $[1, 0]$, and without the subtraction, $e^{800}$ overflows. In numpy:

```python
exp = np.exp(z - z.max())
weights = exp / exp.sum()
context = weights @ values      # (T,) @ (T, d_v) -> (d_v,)
```

Because the weights are non-negative and sum to 1, the context is a convex combination of the values, which means a blend that cannot leave the range of its inputs. In Figure 2 the context sits on the dashed line between the two values. Attention cannot invent a number that none of the values contained.

## Why a dot product, and why sometimes a matrix

Multiplying matching entries and adding them gives a large positive number when two vectors point the same way, zero when they are perpendicular, and a negative number when they point opposite ways. A key that points where the query points scores high, so the dot product is a reasonable measure of a match. It also grows with the length of the vectors, a fact that matters in a later chapter.

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

> [!KEY]
> Attention is score, softmax, blend. A decoder that gets one vector per source position can put weight on "dog" at one step and on "man" at the next, so the order the one-vector summary lost is rebuilt from which positions the decoder asks for.

> [!RECAP]
> Before you build:
> - A query is compared with every key, and the match decides how much of each value you take.
> - The scores become weights with a stable softmax, so the weights are positive and sum to 1.
> - The context is `weights @ values`, a blend that stays inside the range of the values.
> - A soft blend is trainable because its weights move smoothly with the scores, and a large score gap makes it nearly a hard pick.
> - A matrix `W` between the query and the key lets them differ in width, and `W is None` gives the plain dot product.

## Build it step by step

The checks test the functions in this order:

1. **`softmax(z)`.** Takes scores of shape `(T,)` and returns weights of shape `(T,)`. Subtract `z.max()` before `np.exp`, then divide by the sum. The usual mistake is forgetting the subtraction, which passes small tests and breaks on scores of 800.
2. **`attend(scores, values)`.** Takes scores `(T,)` and values `(T, d_v)`. It returns `(context, weights)`: the softmax of the scores, and `weights @ values`. This is the shared tail of every attention in the next chapters.
3. **`multiplicative_attention(query, keys, values, W=None)`.** Returns `(context, weights)` from `attend`. It builds the scores as `keys @ query` when `W is None` and `keys @ (W.T @ query)` otherwise. The usual mistake is `keys @ (W @ query)`, which passes when `W` is square and symmetric and fails on a non-square one.

## Where this breaks

Dot products grow with the width of the vectors, so wide vectors give large score gaps and a softmax that is nearly a hard pick. A later chapter divides the scores to fix that. A plain dot product is also a fixed way to measure a match. The next chapter lets a small network learn its own.

## The exercise

You build `softmax`, `attend`, and `multiplicative_attention`. The checks confirm that `softmax([1, 0])` gives $0.7311$ and $0.2689$, that it survives scores of $\pm 800$, and that shifting all scores changes nothing. They also confirm that `attend` reproduces the worked context $[7.311, 2.689]$, equals `weights @ values`, and stays inside the range of the values. They require `multiplicative_attention` to match a hand computation of `query @ W @ keys[i]` with a non-square `W`, and to give the same answer for `W=None` and `W` equal to the identity.
