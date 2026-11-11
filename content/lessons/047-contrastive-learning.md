---
title: "Contrastive learning: InfoNCE, collapse, and CLIP"
slug: 047-contrastive-learning
part: "Part 5 — Vision and sequences"
summary: Learn embeddings without labels by picking each example's other view out of a batch, and see why negatives stop collapse.
minutes: 70
gpu: false
objectives:
  - Explain why a loss that only pulls two views together lets every embedding collapse to one point.
  - Compute InfoNCE by hand as a row-wise softmax over cosine similarities with a temperature, and read the gradient as a pull and a push.
  - Describe how negatives prevent collapse, and measure it with alignment and uniformity.
  - Write CLIP's symmetric image-text loss and say why it is symmetric.
  - Name masked reconstruction as the other self-supervised family and say what it avoids.
prereqs:
  - 033-backpropagation
  - 034-softmax-and-whole-network-gradients
  - 044-residual-networks
exercise: 047-contrastive-learning
---

# Contrastive learning: InfoNCE, collapse, and CLIP

You have a million photos and no labels. Nobody has told you which photos show dogs, which show boats, or which show anything at all. A classifier needs a label for every training example, so it has nothing to learn from. Yet the photos hold plenty of structure, and a network that found it would be a strong starting point for a later task with few labels.

Here is a signal that needs no labels. Make two copies of one photo, one cropped and one with its colours shifted. Both show the same thing, so a good representation maps them to nearby points. Copies of different photos should land far apart. The photo's own identity supplies the label.

The obvious loss asks only for the nearby part, and it fails: the network maps every photo to the same point and scores a perfect zero. This chapter builds the loss that avoids this, InfoNCE, and uses it to train CLIP, which matches images to captions. It ends with the other family of label-free methods, which reconstruct hidden pieces of the input.

## What you already know

- A softmax turns a row of scores into probabilities, and cross-entropy is the negative log of the probability given to the right answer ([softmax and whole-network gradients](/lessons/034-softmax-and-whole-network-gradients)).
- The gradient of softmax cross-entropy with respect to the scores is the prediction minus the truth, and the chain rule carries it back through any layer ([backpropagation](/lessons/033-backpropagation)).
- A vector's length is its norm, and dividing a vector by its norm leaves a unit vector that points the same way.
- The dot product of two unit vectors is the cosine of the angle between them: 1 for the same direction, 0 for perpendicular, and -1 for opposite.
- An augmentation, such as a crop or a colour shift, changes how an example looks and not what it is. [Augmentation, transfer, and ViT](/lessons/045-augmentation-transfer-and-vit) covers the standard recipes.

## Two views, one example

An **encoder** is a network that turns an input into a short vector called an **embedding**. In this chapter the encoder is a single linear layer so that you can see everything it does, but the ideas carry over to any network.

Take a batch of $N$ examples and make two augmented copies of each. The copies are the **views**: view 1 and view 2. Send both through the encoder. Row $i$ of the first embedding matrix $z_1$ and row $i$ of the second matrix $z_2$ come from the same example, so they form a **positive pair**. Row $i$ of $z_1$ and row $j$ of $z_2$ with $j \ne i$ come from different examples. Those are **negatives**. A batch of $N$ examples gives each row $N-1$ negatives for free, because the other examples are already in the batch.

The running example for this chapter has $N = 3$ examples and two-dimensional embeddings, so that every number fits on the page. Each row has length 1, which the next sections explain:

$$
z_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0.6 & 0.8 \end{bmatrix}, \qquad
z_2 = \begin{bmatrix} 0.8 & 0.6 \\ 0.6 & 0.8 \\ 0 & 1 \end{bmatrix}
$$

The positive pairs are row 1 with row 1, row 2 with row 2, and row 3 with row 3. This is a hard batch on purpose: row 2 of $z_1$ is closer to row 3 of $z_2$ than to its own partner.

## Why pulling views together is not enough

The first idea is to minimise the distance between the two views of each example. This loss is called **alignment**: if the pairs coincide, it is 0.

It has a trivial perfect solution. An encoder that outputs the same vector for everything makes every pair coincide, so the loss is exactly 0, and nothing in the loss prefers a useful encoder to a constant one. This failure is called **collapse**, and gradient descent finds it because it is the shortest road to zero.

The missing ingredient is a reason to keep different examples apart, and that reason is the negatives. If the encoder maps everything to one point, each example's partner is no easier to pick out than any other row in the batch, and a loss that rewards picking the partner out charges for it.

## InfoNCE: find the partner in a line-up

Here is the picture. For one example, view 1 is the **query**. The $N$ rows of view 2 are a line-up of candidates, and exactly one of them is the query's partner. Pick it out. This is a classification problem with $N$ classes, where "class $j$" means "the $j$th row of view 2", and the true class of query $i$ is $i$. You already know the loss for a classification problem: softmax followed by cross-entropy.

The scores need a definition. Compare two embeddings by the cosine of the angle between them, which is the dot product of the unit-length versions. Write $n = z / \lVert z \rVert$ for a normalised row, and

$$
s_{ij} = n_{1,i} \cdot n_{2,j}
$$

Read it aloud: $s_{ij}$ is the cosine similarity between query $i$ and candidate $j$. Dividing by the length first means a row's size cannot raise its score. Without this step, an encoder could lower the loss by making all its vectors longer, which says nothing about whether the partners are close.

Divide each similarity by a number $\tau$ called the **temperature**, and apply a softmax along each row. The probability that query $i$ picks candidate $j$ is

$$
p_{ij} = \frac{\exp(s_{ij}/\tau)}{\sum_{k} \exp(s_{ik}/\tau)}
$$

The loss is the mean negative log probability of the right pick, which is the **InfoNCE** loss, named in the paper that introduced contrastive predictive coding:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \log p_{ii}
$$

Read it aloud: for each query, take the probability it gives its own partner, take the negative log, and average over the batch. The loss is 0 when every query is certain of its partner, and it grows as the queries get confused.

### Work the running example

Use $\tau = 0.5$. The similarity grid is $s = z_1 z_2^{\top}$:

$$
s = \begin{bmatrix} 0.8 & 0.6 & 0 \\ 0.6 & 0.8 & 1.0 \\ 0.96 & 1.0 & 0.8 \end{bmatrix}
$$

Dividing by 0.5 doubles every entry, so the logits are 1.6, 1.2, 0 in row 1, then 1.2, 1.6, 2.0, then 1.92, 2.0, 1.6. Figure 1 shows the softmax of each row.

```figure
<svg viewBox="0 0 640 330" role="img" aria-label="A 3 by 3 grid of similarities between three view-1 embeddings and three view-2 embeddings, with the softmax probability in each cell and the loss for each row">
<text x="20" y="28" class="fig-bold">view 2 (candidates)</text>
<text x="160.0" y="58" text-anchor="middle" class="fig-mono">z2[1]</text>
<text x="260.0" y="58" text-anchor="middle" class="fig-mono">z2[2]</text>
<text x="360.0" y="58" text-anchor="middle" class="fig-mono">z2[3]</text>
<text x="100" y="105.0" text-anchor="end" class="fig-mono">z1[1]</text>
<rect x="110" y="70" width="100" height="62" class="fill-soft-b stroke-b"/>
<text x="160.0" y="96" text-anchor="middle" class="fig-bold">p = 0.534</text>
<text x="160.0" y="117" text-anchor="middle" class="fig-small">logit 1.60</text>
<rect x="210" y="70" width="100" height="62" class="fill-soft-a stroke-grid"/>
<text x="260.0" y="96" text-anchor="middle" class="fig-bold">p = 0.358</text>
<text x="260.0" y="117" text-anchor="middle" class="fig-small">logit 1.20</text>
<rect x="310" y="70" width="100" height="62" class="fill-soft-a stroke-grid"/>
<text x="360.0" y="96" text-anchor="middle" class="fig-bold">p = 0.108</text>
<text x="360.0" y="117" text-anchor="middle" class="fig-small">logit 0.00</text>
<text x="426" y="96" class="fig-small">loss 0.627</text>
<text x="426" y="116" class="fig-small fig-mono">-log 0.534</text>
<text x="100" y="167.0" text-anchor="end" class="fig-mono">z1[2]</text>
<rect x="110" y="132" width="100" height="62" class="fill-soft-a stroke-grid"/>
<text x="160.0" y="158" text-anchor="middle" class="fig-bold">p = 0.212</text>
<text x="160.0" y="179" text-anchor="middle" class="fig-small">logit 1.20</text>
<rect x="210" y="132" width="100" height="62" class="fill-soft-b stroke-b"/>
<text x="260.0" y="158" text-anchor="middle" class="fig-bold">p = 0.316</text>
<text x="260.0" y="179" text-anchor="middle" class="fig-small">logit 1.60</text>
<rect x="310" y="132" width="100" height="62" class="fill-soft-a stroke-grid"/>
<text x="360.0" y="158" text-anchor="middle" class="fig-bold">p = 0.472</text>
<text x="360.0" y="179" text-anchor="middle" class="fig-small">logit 2.00</text>
<text x="426" y="158" class="fig-small">loss 1.151</text>
<text x="426" y="178" class="fig-small fig-mono">-log 0.316</text>
<text x="100" y="229.0" text-anchor="end" class="fig-mono">z1[3]</text>
<rect x="110" y="194" width="100" height="62" class="fill-soft-a stroke-grid"/>
<text x="160.0" y="220" text-anchor="middle" class="fig-bold">p = 0.356</text>
<text x="160.0" y="241" text-anchor="middle" class="fig-small">logit 1.92</text>
<rect x="210" y="194" width="100" height="62" class="fill-soft-a stroke-grid"/>
<text x="260.0" y="220" text-anchor="middle" class="fig-bold">p = 0.386</text>
<text x="260.0" y="241" text-anchor="middle" class="fig-small">logit 2.00</text>
<rect x="310" y="194" width="100" height="62" class="fill-soft-b stroke-b"/>
<text x="360.0" y="220" text-anchor="middle" class="fig-bold">p = 0.258</text>
<text x="360.0" y="241" text-anchor="middle" class="fig-small">logit 1.60</text>
<text x="426" y="220" class="fig-small">loss 1.353</text>
<text x="426" y="240" class="fig-small fig-mono">-log 0.258</text>
<text x="110" y="284" class="fig-small">Orange cells hold the positive pair. Each row sums to 1. Mean row loss: 1.044. log 3 = 1.099.</text>
</svg>
Figure 1. The similarity grid for three examples at temperature 0.5. Each row is a three-way classification, and the loss is the negative log of the orange cell. Notice that rows 2 and 3 give the positive less than the largest probability, so those rows cost more than log 3 would suggest.
```

Take row 1. The three exponentials are $e^{1.6} = 4.953$, $e^{1.2} = 3.320$, and $e^{0} = 1$, which sum to 9.273. The positive's probability is $4.953 / 9.273 = 0.534$, and the loss for the row is $-\ln 0.534 = 0.627$. Row 2 is worse: its positive gets only 0.316 because candidate 3 scores higher, so its loss is 1.151. Row 3 gives its positive 0.259, a loss of 1.353. The mean is $L = (0.627 + 1.151 + 1.353)/3 = 1.044$.

A reference point is $\ln 3 = 1.099$, the loss when a query gives all three candidates the same probability, a random guess. This batch scores 1.044, barely better, because two of its three rows prefer a wrong candidate. If every embedding is the same vector, every probability is $1/N$ and the loss is exactly $\ln N$. You will use this as a check.

### Read the loss in numpy

```python
n1, n2 = l2_normalize(z1), l2_normalize(z2)
logits = n1 @ n2.T / tau                    # (N, N): row i = query i
loss = np.mean(logsumexp(logits, axis=1) - np.diag(logits))
```

The matrix product builds all $N^2$ similarities at once. The last line computes $-\log p_{ii}$ as the log of the row's summed exponentials minus the positive's logit, and `logsumexp` subtracts the row maximum first, so it never exponentiates a large number.

> [!WARNING]
> The softmax runs along each row, so it sums over candidates for one query. Taking it down the columns instead answers a different question, "which query goes with this candidate?", and returns a different number whenever the grid is not symmetric. In the running example the two readings give 1.044 and 1.085.

### What the temperature does

The temperature controls how sharply the softmax separates scores. Here is the running example's loss at five temperatures:

| $\tau$ | 0.05 | 0.1 | 0.5 | 2 | 10 |
|---|---|---|---|---|---|
| $L$ | 2.807 | 1.620 | 1.044 | 1.069 | 1.092 |

At a large $\tau$ all logits shrink toward 0, every probability approaches $1/3$, and the loss approaches $\ln 3 = 1.099$. The loss then barely reacts to which candidate is closer. At a small $\tau$ the largest logit dominates its row, so one wrong candidate that scores a little above the partner is punished heavily. A small temperature therefore makes the loss attend to the hardest negatives. Too small, and a few near-duplicate negatives dominate the gradient. Values between 0.05 and 0.5 are typical.

## The gradient is a pull and a push

Training needs $\partial L / \partial s_{ij}$. The softmax cross-entropy rule from [the softmax chapter](/lessons/034-softmax-and-whole-network-gradients) gives the answer for each row, and averaging over $N$ rows and the $1/\tau$ in the logits scale it:

$$
\frac{\partial L}{\partial s_{ij}} = \frac{p_{ij} - \delta_{ij}}{N\tau}
$$

Here $\delta_{ij}$ is 1 when $i = j$ and 0 otherwise. Read it aloud: for the positive, the entry is negative, because $p_{ii} - 1 < 0$, so raising $s_{ii}$ lowers the loss and the positive is pulled closer. For every negative, the entry is $p_{ij}/(N\tau) > 0$, so raising $s_{ij}$ raises the loss and the negative is pushed away. The size of each push is the probability the model wastes on that negative, so the hardest negatives are pushed hardest.

In the running example, row 2 has $p_{22} = 0.316$ and a pull of $(0.316 - 1)/(3 \times 0.5) = -0.456$. It also has a push of $0.472/1.5 = 0.315$ on candidate 3, the negative it likes too much, and a push of 0.141 on candidate 1. The model fixes its worst confusion first.

The embeddings are not the similarities, so the chain rule continues. The similarity grid depends on the unit rows, $\partial L/\partial n_1 = G\, n_2$ and $\partial L / \partial n_2 = G^{\top} n_1$, where $G$ is the matrix of entries above. Then the gradient passes through the normalisation. Dividing $z$ by its length means only the direction of $z$ matters, so the gradient must be perpendicular to $z$, and it shrinks as $z$ gets longer:

$$
\frac{\partial L}{\partial z} = \frac{1}{\lVert z \rVert}\left( g - n\,(n \cdot g) \right), \qquad g = \frac{\partial L}{\partial n}
$$

Read it aloud: take the gradient with respect to the unit row, remove the part of it that points along the row, because stretching a row changes nothing, and divide by the row's length. In numpy:

```python
G = (P - np.eye(N)) / (N * tau)
dn1, dn2 = G @ n2, G.T @ n1
dz1 = (dn1 - n1 * np.sum(n1 * dn1, axis=1, keepdims=True)) / np.linalg.norm(z1, axis=1, keepdims=True)
```

For the running example, $dz_1$ for row 3 is $[0.290, -0.218]$. The exercise checks it against finite differences.

> [!TIP]
> To test the normalisation step on its own, check that each row of $dz$ has a zero dot product with the matching row of $z$. A gradient that forgets the projection fails this test, and finite differences catch it as well.

## What negatives buy: collapse, alignment, and uniformity

Does the negative half of the loss prevent collapse? A small experiment answers it, and the exercise runs it. The data has four clusters and 12 numbers per example. Two numbers hold the cluster, with means $(\pm 3, \pm 3)$ and a little noise. The other ten are nuisance, a random draw with standard deviation 3. The augmentation redraws the nuisance numbers for each view, so the only thing two views always share is the cluster signal. The encoder is a linear map from 12 numbers to 2, and the batch is 64 examples.

Train it twice. In one run the loss is InfoNCE with $\tau = 0.5$ for 100 steps. In the other the loss is alignment alone, for 5000 steps. Figure 2 plots the 2-D test embeddings, normalised to the unit circle.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Two unit circles. Left: embeddings of four classes spread around the circle in four separate groups. Right: all embeddings sit on two opposite points.">
<defs></defs>
<circle cx="150" cy="140" r="100" class="fill-paper stroke-grid"/>
<circle cx="450" cy="140" r="100" class="fill-paper stroke-grid"/>
<circle cx="250" cy="135" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="249" cy="151" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="243" cy="176" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="247" cy="162" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="249" cy="154" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="250" cy="148" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="250" cy="140" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="248" cy="162" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="249" cy="151" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="247" cy="165" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="248" cy="160" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="249" cy="152" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="160" cy="41" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="158" cy="40" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="165" cy="41" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="178" cy="44" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="166" cy="41" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="168" cy="42" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="163" cy="41" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="169" cy="42" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="143" cy="40" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="172" cy="42" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="166" cy="41" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="153" cy="40" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="107" cy="230" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="122" cy="236" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="150" cy="240" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="130" cy="238" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="131" cy="238" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="125" cy="237" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="142" cy="240" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="126" cy="237" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="116" cy="234" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="130" cy="238" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="127" cy="237" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="122" cy="236" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="53" cy="117" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="55" cy="110" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="52" cy="122" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="54" cy="112" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="51" cy="124" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="54" cy="113" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="50" cy="145" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="57" cy="103" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="50" cy="137" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="51" cy="127" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="50" cy="149" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="50" cy="138" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-a" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-b" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="495" cy="229" r="4" class="fill-c" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-d" fill-opacity="0.8"/>
<circle cx="405" cy="51" r="4" class="fill-d" fill-opacity="0.8"/>
<text x="150" y="268" text-anchor="middle" class="fig-bold">InfoNCE, 100 steps</text>
<text x="150" y="288" text-anchor="middle" class="fig-small">k-NN accuracy 1.00, uniformity -1.41</text>
<text x="450" y="268" text-anchor="middle" class="fig-bold">Alignment only, 5000 steps</text>
<text x="450" y="288" text-anchor="middle" class="fig-small">k-NN accuracy 0.48, uniformity -0.70</text>
<circle cx="150" cy="313" r="5" class="fill-a"/><text x="160" y="317" class="fig-small">class 0</text>
<circle cx="250" cy="313" r="5" class="fill-b"/><text x="260" y="317" class="fig-small">class 1</text>
<circle cx="350" cy="313" r="5" class="fill-c"/><text x="360" y="317" class="fig-small">class 2</text>
<circle cx="450" cy="313" r="5" class="fill-d"/><text x="460" y="317" class="fig-small">class 3</text>
<text x="450" y="30" text-anchor="middle" class="fig-small">classes 1 and 3 share one point</text>
<text x="450" y="254" text-anchor="middle" class="fig-small">classes 0 and 2 share the other</text>
</svg>
Figure 2. Test embeddings of a linear encoder, on the unit circle. With negatives (left), the four classes land in four separate groups. With positives only (right), the encoder squashes everything onto a line, so the two points hold two classes each. Notice that the right-hand loss is nearly zero, which is why nothing in it warns you.
```

InfoNCE separates all four clusters into four groups around the circle, and a 5-nearest-neighbour classifier on these embeddings gets all 100 test points right. The alignment-only encoder has an alignment of 0.0000, a perfect score, and a nearest-neighbour accuracy of 0.48, because it has folded the circle onto a line and two classes share each end. A linear map into two dimensions can only collapse partway, to one direction. An encoder with more freedom collapses all the way, to a single point.

Two numbers describe the picture. **Alignment** is the mean squared distance between the unit embeddings of the two views of an example:

$$
\text{alignment} = \frac{1}{N} \sum_{i} \lVert n_{1,i} - n_{2,i} \rVert^2
$$

Low is good, and 0 means the pairs coincide. **Uniformity** measures how evenly the embeddings fill the circle. Take every pair $i < j$, compute a kernel that is 1 for identical points and near 0 for far-apart points, average it, and take the log:

$$
\text{uniformity} = \log \operatorname{mean}_{i<j} \exp\!\left(-t \, \lVert n_i - n_j \rVert^2\right)
$$

Read it aloud: if all points coincide, every kernel value is 1 and uniformity is 0, the largest it can be. Spreading points apart makes the kernel smaller, so uniformity goes down. Use $t = 2$. Two opposite unit vectors are at squared distance 4, so a batch that is half one point and half its opposite scores about $\log(1/2) = -0.69$, and that is what the collapsed encoder reports: $-0.698$. The InfoNCE encoder scores $-1.407$. Wang and Isola showed that InfoNCE, as the batch grows, optimises both: the pull lowers alignment, and the push from the negatives lowers uniformity.

> [!KEY]
> A contrastive loss has two jobs. The pull makes two views of one example agree, and the push, from the negatives, spreads different examples apart. Drop the push and the cheapest solution is a constant encoder, which a pull-only loss scores as perfect.

Methods called BYOL and SimSiam avoid collapse without negatives. They add a small predictor network on one branch and stop the gradient from flowing through the other branch. Why this works is understood only in part, and the exercise uses negatives.

## CLIP: match an image to its caption in both directions

CLIP trains two encoders at once, one for images and one for captions, on pairs scraped from the web. The two "views" are different kinds of thing: image $i$ and its caption. The batch is the same as before, with image embeddings in $z_{\text{img}}$ and caption embeddings in $z_{\text{txt}}$, row $i$ of each from the same pair.

The grid $s = n_{\text{img}} n_{\text{txt}}^{\top}$ can be read two ways. Reading along rows asks, for each image, which caption in the batch is the right one: that is InfoNCE with images as queries. Reading down the columns asks, for each caption, which image is the right one: that is InfoNCE with captions as queries, which is `info_nce(txt, img)`. CLIP averages the two:

$$
L_{\text{CLIP}} = \tfrac{1}{2}\left( L_{\text{img}\to\text{txt}} + L_{\text{txt}\to\text{img}} \right)
$$

Both terms read one grid, so swapping the roles of image and text swaps the two terms, and the sum does not change. The loss is symmetric: `clip_loss(a, b)` equals `clip_loss(b, a)`. A single `info_nce` is not symmetric, and the running example shows how much the two directions differ: 1.0438 reading rows and 1.0855 reading columns. Their mean is $L_{\text{CLIP}} = 1.0646$ on the running example.

Two details matter in practice. CLIP learns the temperature along with the weights, and its batch holds tens of thousands of pairs, because a larger batch supplies more negatives per row. The CPC paper shows that the loss gives a lower bound on the mutual information between the views, how much one view tells you about the other, and that this bound cannot exceed $\log N$, so a larger batch raises the ceiling.

After training, caption embeddings work as a classifier: embed "a photo of a dog" and "a photo of a boat", and label an image with the closest sentence. No dog or boat labels were used.

## Masked reconstruction: the other family

Contrastive methods are one family of self-supervised learning. The other hides part of the input and trains a network to reconstruct it. A masked autoencoder cuts an image into patches, hides most of them (He et al. hide 75 percent), sends the visible patches through an encoder, and asks a small decoder to reproduce the pixels of the hidden patches. The loss is a squared error on the hidden patches.

This family needs no negatives and has no collapse, because the target is the data, and a constant output cannot reproduce it. The price is that the loss rewards getting pixels right, so some capacity goes to detail that a later task may not need. A contrastive loss discards that detail through the augmentations. [The autoencoder chapter](/lessons/069-autoencoders-and-vaes) covers reconstruction in more depth.

> [!KEY]
> You have a way to learn embeddings from unlabelled data. Pick each example's other view out of a batch (InfoNCE), and negatives keep the embeddings spread out, so the encoder cannot collapse. Read the same grid in both directions and you have CLIP. Hide part of the input and reconstruct it, and you have the other family.

> [!RECAP]
> - Two augmented views of one example form a positive pair. The other examples in the batch are its negatives.
> - InfoNCE is a row-wise softmax cross-entropy over cosine similarities divided by a temperature. It equals $\ln N$ when all embeddings are identical.
> - The gradient pulls each positive closer with weight $(1 - p_{ii})/(N\tau)$ and pushes each negative away with weight $p_{ij}/(N\tau)$, then loses the part along each row because of the normalisation.
> - Without negatives, an alignment-only loss is satisfied by a constant encoder, and collapse is what you get. Alignment and uniformity measure the two sides.
> - CLIP's loss averages the row reading and the column reading of one similarity grid, so it is symmetric.

## Build it step by step

1. **`l2_normalize(z)`** takes shape `(N, d)` and returns the same shape with every row of length 1. Use `np.linalg.norm(z, axis=1, keepdims=True)`. The mistake people most often make is to omit `keepdims=True`.
2. **`info_nce(z1, z2, tau)`** returns the mean cross-entropy as a float. Normalise, build `logits = n1 @ n2.T / tau`, and take `logsumexp` of each row minus its diagonal entry. People most often skip the normalisation or take the softmax over columns, and the hand-example check catches both.
3. **`info_nce_grad(z1, z2, tau)`** returns `(dz1, dz2)`. Build `G = (P - I) / (N * tau)`, then `dn1 = G @ n2` and `dn2 = G.T @ n1`, then remove the part along each row and divide by that row's length. The common mistake is to return `dn1` and `dn2`, the gradients with respect to the unit rows.
4. **`clip_loss(img, txt, tau)`** returns the mean of `info_nce(img, txt, tau)` and `info_nce(txt, img, tau)`.
5. **`alignment(z1, z2)`** returns the mean squared distance between unit rows. Normalise first.
6. **`uniformity(z, t=2.0)`** returns the log of the mean of `exp(-t * d2)` over pairs `i < j`. Use `np.triu_indices(N, k=1)`. Including the pairs `i == j` inflates the mean.
7. **`train_encoder(make_views, d_in, d_out, tau, steps, lr, seed)`** returns `W` of shape `(d_in, d_out)`. Each step draws views, embeds with `v @ W`, and updates `W -= lr * (v1.T @ dz1 + v2.T @ dz2)`. People often forget the second view's term.
8. **`knn_accuracy(train_z, train_y, test_z, test_y, k)`** returns the fraction of test rows whose `k` nearest training rows, by cosine similarity, vote for the right label.

## Where this breaks

- **False negatives.** Two different examples in a batch can show the same thing, such as two photos of dogs. The loss pushes them apart anyway. People filter such pairs when labels or captions allow, or accept the noise.
- **Augmentations decide what you learn.** The embedding ignores whatever the augmentations change and keeps whatever they leave alone. If colour is never shifted, the encoder can match partners by colour and learn nothing else.
- **Batch size and cost.** The similarity grid has $N^2$ entries, so the loss costs memory with the square of the batch. A small batch gives few negatives and an easy task, which is why published runs use large batches.

## The exercise

You build the pipeline in numpy. `l2_normalize`, `info_nce`, and `info_nce_grad` form the core, `clip_loss` reuses `info_nce`, `alignment` and `uniformity` are the diagnostics, and `train_encoder` and `knn_accuracy` train and score a linear encoder.

The checks run in that order. They test the hand example, including unnormalised rows and the row-versus-column readings, and the value $\ln N$ for identical embeddings. They check the gradient against finite differences and perpendicularity, check that `clip_loss` is symmetric, and train an encoder for 100 steps on the planted clusters, which must reach a k-NN accuracy of 0.9 while a constant embedding scores near chance.

## Further reading

- The InfoNCE loss and its mutual-information bound: Oord, Li, and Vinyals, ["Representation learning with contrastive predictive coding"](https://arxiv.org/abs/1807.03748), 2018.
- Two augmented views, a normalised embedding, and a temperature, at scale: Chen, Kornblith, Norouzi, and Hinton, ["A simple framework for contrastive learning of visual representations"](https://arxiv.org/abs/2002.05709), 2020.
- The symmetric image-text loss and zero-shot classification: Radford and others, ["Learning transferable visual models from natural language supervision"](https://arxiv.org/abs/2103.00020), 2021.
- Alignment and uniformity: Wang and Isola, ["Understanding contrastive representation learning through alignment and uniformity on the hypersphere"](https://arxiv.org/abs/2005.10242), 2020.
- Masked reconstruction: He and others, ["Masked autoencoders are scalable vision learners"](https://arxiv.org/abs/2111.06377), 2021.
- Collapse prevention without negatives: Grill and others, ["Bootstrap your own latent"](https://arxiv.org/abs/2006.07733), 2020, and Chen and He, ["Exploring simple Siamese representation learning"](https://arxiv.org/abs/2011.10566), 2020.
- Course material: the Stanford CS231n [schedule](https://cs231n.stanford.edu/schedule.html) lists "Lecture 12: Self-supervised Learning". The MIT 6.S898 [schedule](https://phillipi.github.io/6.s898/) lists "Representation learning -- reconstruction-based", "Representation learning -- similarity-based", and "Representation learning -- theory".
