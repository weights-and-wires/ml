---
title: "Graph neural networks: message passing, GCN, and over-smoothing"
slug: 058-graph-neural-networks
part: "Part 5 — Vision and sequences"
summary: Learn on data shaped like a network by letting each node average its neighbours, and see why the node order must not matter.
minutes: 65
gpu: false
objectives:
  - Explain why a plain network on a flattened adjacency matrix breaks when the nodes are renumbered.
  - Define permutation equivariance for node outputs and invariance for graph outputs, and check both numerically.
  - Compute one GCN layer by hand, with self-loops and symmetric normalisation, and write its gradient.
  - Say what sum, mean, and max aggregation cannot tell apart, and what that has to do with the Weisfeiler-Lehman test.
  - Measure over-smoothing as layers stack, and describe attention over neighbours as a masked softmax.
prereqs:
  - 036-backpropagation
  - 046-convolution-and-receptive-field
  - 056-attention-as-soft-lookup
exercise: 058-graph-neural-networks
---

# Graph neural networks: message passing, GCN, and over-smoothing

Four people sit in a friendship network. You know a few facts about each one, such as age and how many posts they read per day, and you know who is friends with whom. Two of them have told you which topic they follow. You want to guess the topic of the other two. A good guess uses both kinds of information: a person's own facts, and the topics their friends follow, because friends tend to follow similar things.

The networks you have built so far do not fit this data. An image is a grid, so a convolution can slide a filter over fixed neighbours. A sentence is a line, so a recurrent cell can walk along it. A friendship network, a molecule, a road map, and a web of citations have no grid and no line. Each item has a different number of neighbours, and nothing says which neighbour comes first.

This chapter builds the standard answer, the graph neural network. Each node collects its neighbours' features, combines them with its own, and passes the result on. You write the layer that does this, check that renumbering the nodes cannot change the answer, watch what goes wrong when you stack many layers, and train a two-layer network to find two communities from six labels.

## What you already know

- A matrix product mixes rows: row $i$ of $A X$ is a weighted sum of the rows of $X$, with weights taken from row $i$ of $A$ ([multilayer perceptrons](/lessons/034-multilayer-perceptrons)).
- Backpropagation sends the gradient of a loss back through a matrix product and a ReLU ([backpropagation](/lessons/036-backpropagation)).
- A convolution gives every position the same weights and mixes it with its neighbours ([convolution and receptive field](/lessons/046-convolution-and-receptive-field)).
- Attention scores a set of items, turns the scores into weights with a softmax, and blends the items ([attention as a soft lookup](/lessons/056-attention-as-soft-lookup)).
- A function is symmetric when swapping its arguments leaves the value unchanged, and a symmetric matrix equals its own transpose.

## A graph is a list of nodes and a table of who touches whom

A **graph** has **nodes**, the items, and **edges**, the links between pairs of nodes. In this chapter edges have no direction: if node 1 is linked to node 2, then node 2 is linked to node 1. Each node carries a feature vector, and the features of all $n$ nodes stack into a matrix $H$ with $n$ rows, one per node.

The links go into an **adjacency matrix** $A$, an $n$ by $n$ table with a 1 in row $i$, column $j$ when nodes $i$ and $j$ are linked and a 0 otherwise. Because the links have no direction, $A$ equals its transpose. The diagonal is 0, because no node is linked to itself. The row sum of $A$ is a node's **degree**, its number of neighbours.

The running example has four nodes. Node 1 is the centre of a star, and nodes 0, 2, and 3 are leaves that touch only the centre:

$$
A = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 1 & 1 \\ 0 & 1 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}, \qquad
H = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 2 & 0 \end{bmatrix}
$$

The degrees are 1, 3, 1, and 1. Each node has two features, so $H$ has shape $(4, 2)$. The star is deliberately lopsided: one node has three neighbours and the others have one, and this difference matters in every section below.

## Why not flatten the table and use a network?

The obvious plan is to flatten $A$ into a vector of $n^2$ numbers, append the features, and feed everything to a multilayer perceptron. This fails, and the reason is the heart of the chapter.

The nodes have no natural order. You numbered the star's centre as node 1, but you could as well have numbered it 3. The graph is the same, yet the matrix changes. Renumbering applies a **permutation** to both the rows and the columns of $A$, and to the rows of $H$. Renumbering the star with the order $[2, 0, 1, 3]$ changes 8 of the 16 entries of $A$, as `scripts/figures/058-graph-neural-networks.py` prints, so a network on the flattened $A$ sees a different input. It would have to learn the same fact for every one of the $n!$ numberings, and its fixed input size could not take a five-node graph after training on four-node graphs.

The fix is to build the symmetry into the layer instead of asking the weights to learn it. Two properties say what the layer must respect:

- A **node-level** output, such as a prediction for each person, must be **permutation equivariant**. Renumbering the input nodes renumbers the output rows the same way. Node 3's prediction is still node 3's prediction, wherever it sits in the list.
- A **graph-level** output, such as one prediction for a whole molecule, must be **permutation invariant**. Renumbering the nodes leaves it unchanged, because the molecule has not changed.

> [!KEY]
> Nodes have no order, so a graph layer must give the same answer under every renumbering. Per-node outputs move with the renumbering (equivariance). One output for the whole graph does not change (invariance). A layer built from the same weights for every node and an order-free combination of neighbours has both properties by construction.

## Message passing: ask your neighbours

Here is the picture. Every node holds a note with its features. In one round, each node reads the notes of its neighbours, combines them into one summary, and then updates its own note using that summary. The reading is the **aggregation** step and the rewriting is the **update** step. Gilmer and colleagues named this pattern **message passing**, and nearly every graph network fits it.

The aggregation must not depend on the order in which the neighbours are read, so it is a sum, a mean, or a max. The update is usually a learned linear map and a ReLU with the same weights at every node, which is a convolution's shared filter again and is why a graph of any size can use the same layer.

Work the aggregation by hand on the star, using only the first feature column, $h = [1, 0, 1, 2]$. Row $i$ of $A$ picks out node $i$'s neighbours, and a node does not read itself.

| Node | Neighbours | Their features | Sum | Mean | Max |
|---|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 0 | 0 |
| 1 | 0, 2, 3 | 1, 1, 2 | 4 | 1.333 | 2 |
| 2 | 1 | 0 | 0 | 0 | 0 |
| 3 | 1 | 0 | 0 | 0 | 0 |

The sum is the product $A h$. Two things are wrong with plain aggregation. A node never reads its own feature, so its own information vanishes after one round. And the sum grows with the degree: a node with a hundred neighbours receives numbers a hundred times larger than a node with one. The next section fixes both.

## The GCN layer: self-loops and symmetric normalisation

Kipf and Welling's graph convolutional network (GCN) fixes the two problems with two small changes to $A$.

First, add a link from every node to itself, which gives $A + I$, where $I$ is the identity matrix. A node's own feature is then part of its sum. This is the **self-loop**. Write $D$ for the diagonal matrix of the row sums of $A + I$, so that $D_{ii}$ is node $i$'s degree plus one.

Second, rescale the entries so that high-degree nodes do not dominate. The simplest rescaling divides each row by its row sum, which turns the sum into a mean. The GCN uses the **symmetric normalisation** instead:

$$
\hat A = D^{-1/2} (A + I) D^{-1/2}, \qquad \hat A_{ij} = \frac{(A + I)_{ij}}{\sqrt{d_i \, d_j}}
$$

Read it aloud: each entry of $A + I$ is divided by the square root of the degree of its row node times the square root of the degree of its column node, where each degree counts the self-loop. The reason for the square roots is fairness to both ends of an edge. A leaf that listens to a hub should not be drowned out by it, and the hub should not be flooded by thousands of leaves. Dividing by $\sqrt{d_i d_j}$ treats the two directions of an edge the same, which is why $\hat A$ is symmetric. Dividing a row by its own sum, $D^{-1}(A + I)$, gives a different weight for each direction and a matrix that is not symmetric.

Compute it for the star. With self-loops the degrees $d$ are 2, 4, 2, and 2. A leaf's self-loop weight is $1/\sqrt{2 \cdot 2} = 0.5$. The centre's is $1/4 = 0.25$. Each edge between the centre and a leaf is $1/\sqrt{2 \cdot 4} = 0.3536$ in both directions:

$$
\hat A = \begin{bmatrix} 0.5 & 0.3536 & 0 & 0 \\ 0.3536 & 0.25 & 0.3536 & 0.3536 \\ 0 & 0.3536 & 0.5 & 0 \\ 0 & 0.3536 & 0 & 0.5 \end{bmatrix}
$$

Row normalisation, $D^{-1}(A + I)$, would give 0.25 from the centre to a leaf and 0.5 the other way, so it is not symmetric. An exercise check catches the difference.

The layer multiplies the normalised neighbour sum by a weight matrix $W$ of shape $(d_{in}, d_{out})$ and bends the result with a ReLU:

$$
H' = \operatorname{ReLU}\!\left(\hat A \, H \, W\right)
$$

Read it aloud: $\hat A H$ replaces each node's features by a weighted mix of its own and its neighbours', $W$ re-mixes the feature columns the same way at every node, and the ReLU keeps the positive part. Take $W = [1, -1]^{\top}$. The mix is

$$
\hat A H = \begin{bmatrix} 0.5 & 0.3536 \\ 1.4142 & 0.6036 \\ 0.5 & 0.8536 \\ 1.0 & 0.3536 \end{bmatrix}
$$

For node 0 the first entry is $0.5 \cdot 1 + 0.3536 \cdot 0 = 0.5$ and the second is $0.5 \cdot 0 + 0.3536 \cdot 1 = 0.3536$. Applying $W$ subtracts the second column from the first, giving $0.1464, 0.8107, -0.3536, 0.6464$, and the ReLU clips the negative entry to 0, so the output is $[0.1464, 0.8107, 0, 0.6464]$. Figure 1 draws the graph with these numbers.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="A four-node star graph. Each edge carries the weight 0.354, each leaf a self-loop weight of 0.5, and the centre 0.25. Under each node are its input features and its layer output after the weight and ReLU.">
<line x1="140" y1="70" x2="300" y2="150" class="stroke-a"/>
<line x1="140" y1="230" x2="300" y2="150" class="stroke-a"/>
<line x1="460" y1="150" x2="300" y2="150" class="stroke-a"/>
<text x="195" y="96" class="fig-small fig-mono" text-anchor="middle">0.354</text>
<text x="195" y="214" class="fig-small fig-mono" text-anchor="middle">0.354</text>
<text x="380" y="140" class="fig-small fig-mono" text-anchor="middle">0.354</text>
<circle cx="140" cy="70" r="26" class="fill-soft-a stroke-a"/>
<circle cx="140" cy="230" r="26" class="fill-soft-a stroke-a"/>
<circle cx="460" cy="150" r="26" class="fill-soft-a stroke-a"/>
<circle cx="300" cy="150" r="30" class="fill-soft-b stroke-b"/>
<text x="140" y="75" class="fig-bold" text-anchor="middle">0</text>
<text x="140" y="235" class="fig-bold" text-anchor="middle">2</text>
<text x="460" y="155" class="fig-bold" text-anchor="middle">3</text>
<text x="300" y="155" class="fig-bold" text-anchor="middle">1</text>
<text x="140" y="32" class="fig-small" text-anchor="middle">self 0.5</text>
<text x="140" y="270" class="fig-small" text-anchor="middle">self 0.5</text>
<text x="460" y="112" class="fig-small" text-anchor="middle">self 0.5</text>
<text x="300" y="112" class="fig-small" text-anchor="middle">self 0.25</text>
<text x="60" y="100" class="fig-small fig-mono">in (1, 0)</text>
<text x="60" y="118" class="fig-small fig-mono">out 0.146</text>
<text x="60" y="260" class="fig-small fig-mono">in (1, 1)</text>
<text x="60" y="278" class="fig-small fig-mono">out 0.000</text>
<text x="420" y="196" class="fig-small fig-mono">in (2, 0)</text>
<text x="420" y="214" class="fig-small fig-mono">out 0.646</text>
<text x="300" y="196" class="fig-small fig-mono" text-anchor="middle">in (0, 1)</text>
<text x="300" y="214" class="fig-small fig-mono" text-anchor="middle">out 0.811</text>
</svg>
Figure 1. The star graph with its normalised weights. The centre has more neighbours, so its self-loop weight (0.25) is smaller than a leaf's (0.5). Leaf 2 is pulled to zero because its pre-activation, 0.5 minus 0.854 for the weighted mix, is negative.
```

In numpy the layer is two lines, and the backward pass is the matrix-product rule from [backpropagation](/lessons/036-backpropagation). Here `d_pre` is `d_out` with zeros wherever the ReLU clipped the value:

```python
pre = A_hat @ H @ W
out = np.maximum(pre, 0.0)
# backward, given d_out of shape (n, d_out)
d_pre = d_out * (pre > 0)
dW = (A_hat @ H).T @ d_pre        # shape (d_in, d_out)
dH = A_hat.T @ d_pre @ W.T        # shape (n, d_in)
```

> [!KEY]
> A GCN layer is one normalised averaging step, $\hat A H$, followed by a shared linear map and a ReLU. The self-loop keeps a node's own feature. The symmetric normalisation stops a node with many neighbours from producing larger numbers than a node with few, and it treats both ends of an edge alike.

### The layer is permutation safe

Renumbering by a permutation matrix $P$ replaces $A$ with $P A P^{\top}$ and $H$ with $P H$, and $\hat A$ becomes $P \hat A P^{\top}$. Then $(P \hat A P^{\top})(P H) W = P \, (\hat A H W)$, because $P^{\top} P$ is the identity, and the ReLU acts on each entry on its own. The output rows move by the same $P$, which is equivariance. The seeded check in `scripts/figures/058-graph-neural-networks.py` finds a largest difference of 0.0 on the star.

## What each aggregation cannot tell apart

Sum, mean, and max all ignore order, but they throw away different information. Take a node whose neighbours all carry a single number, and compare the multisets of neighbour values:

- **Mean cannot tell $\{1, 2\}$ from $\{1, 1, 2, 2\}$.** Both have mean 1.5. Mean forgets how many neighbours there are. The sum sees 3 and 6.
- **Sum cannot tell $\{1, 3\}$ from $\{2, 2\}$.** Both sum to 4. The max sees 3 and 2.
- **Max cannot tell $\{1, 3\}$ from $\{2, 3\}$.** Both have maximum 3. The mean sees 2 and 2.5.

No choice is best everywhere. Sum keeps the count but lets high-degree nodes produce large values, which is the problem the normalisation solved. Mean and max are bounded, but they lose parts of the neighbourhood.

There is a deeper limit that no choice of these three removes. Consider a six-node ring and two separate triangles. In both graphs every node has exactly two neighbours, and if every node starts with the same feature, every node receives the same message in every round. A message-passing layer cannot tell the two graphs apart, however many rounds it runs, even though one graph is connected and the other is not. This is the idea behind the **Weisfeiler-Lehman test**, a classical method that repeatedly relabels each node by the multiset of its neighbours' labels. Xu and colleagues showed that a message-passing network with an injective aggregation, such as the sum followed by a suitable update, can tell apart exactly the graphs that this test can, and no more. The practical lesson is modest: use a sum when counts matter, and expect the power of your layer to be bounded by what a neighbourhood multiset can express.

## Over-smoothing: too many layers wash out the nodes

Each layer averages a node with its neighbours, so stacking $k$ layers lets information travel $k$ edges. That sounds good, since deeper networks see more of the graph. But averaging has a side effect that grows with depth.

Take the weightless version, $H \leftarrow \hat A H$ repeated $k$ times, on a path of six nodes with starting features $[3, -1, 2, -2, 1, -3]$. Mixing the values of connected nodes brings them closer, and a connected graph has no barrier to stop the process. Li, Han, and Wu showed that repeated graph convolution is a form of Laplacian smoothing that pushes the features of a connected component toward one value. This is **over-smoothing**: the nodes become indistinguishable, so a classifier cannot separate them.

One detail needs care. The symmetric normalisation does not push the raw features to the same value. It pushes them toward values proportional to $\sqrt{d_i}$, where $d_i$ is the self-loop degree. So to see the convergence, divide each row by $\sqrt{d_i}$ before measuring. Then define the **spread** as the average over feature columns of the largest value minus the smallest across nodes. Each step of the rescaled process replaces a node's value with a weighted average of its neighbours' values, and an average never leaves the range of the numbers it averages, so the spread cannot rise. The script prints the spread at each depth:

| Layers | 0 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| Spread | 4.243 | 1.061 | 0.587 | 0.179 | 0.017 |

Figure 2 draws the six nodes' rescaled values from depth 0 to depth 12.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Six lines showing the rescaled feature of each node of a six-node path as layers stack from 0 to 12. They start between about minus 2.1 and 2.1 and squeeze toward zero.">
<line x1="60" y1="150" x2="560" y2="150" class="stroke-grid"/>
<line x1="70" y1="20" x2="70" y2="280" class="stroke-grid"/>
<polyline points="70,33.3 110,107.5 150,104.0 190,113.5 230,116.8 270,120.8 310,124.0 350,127.0 390,129.5 430,131.8 470,133.9 510,135.7 550,137.3" class="stroke-a" fill="none"/>
<polyline points="70,86.5 110,160.6 150,133.5 190,141.0 230,140.0 270,141.6 310,142.4 350,143.3 390,144.0 430,144.7 470,145.3 510,145.8 550,146.3" class="stroke-a" fill="none"/>
<polyline points="70,118.2 110,199.5 150,177.1 190,179.9 230,175.1 270,172.8 310,170.1 350,167.9 390,165.9 430,164.1 470,162.5 510,161.1 550,159.9" class="stroke-a" fill="none"/>
<polyline points="70,181.8 110,100.5 150,122.9 190,120.1 230,124.9 270,127.2 310,129.9 350,132.1 390,134.1 430,135.9 470,137.5 510,138.9 550,140.1" class="stroke-b" fill="none"/>
<polyline points="70,213.5 110,139.4 150,166.5 190,159.0 230,160.0 270,158.4 310,157.6 350,156.7 390,156.0 430,155.3 470,154.7 510,154.2 550,153.7" class="stroke-b" fill="none"/>
<polyline points="70,266.7 110,192.5 150,196.0 190,186.5 230,183.2 270,179.2 310,176.0 350,173.0 390,170.5 430,168.2 470,166.1 510,164.3 550,162.7" class="stroke-b" fill="none"/>
<text x="64" y="154" class="fig-small" text-anchor="end">0</text>
<text x="70" y="296" class="fig-small" text-anchor="middle">layer 0</text>
<text x="550" y="296" class="fig-small" text-anchor="middle">layer 12</text>
<text x="76" y="30" class="fig-small">nodes that start positive</text>
<text x="76" y="272" class="fig-small">nodes that start negative</text>
</svg>
Figure 2. Features on a six-node path, divided by the square root of each node's degree, as weightless GCN layers stack. Blue lines start above zero and orange lines start below. They begin far apart and end in a narrow band, so a classifier after many layers has little left to separate the nodes by.
```

The practical rule is to keep GNNs shallow, often two or three layers. Researchers also add residual connections, as in [residual networks](/lessons/048-residual-networks), so that a node keeps a direct copy of its own features.

> [!WARNING]
> More layers does not mean a better graph network. Each layer averages, and averaging many times erases the differences between nodes. If accuracy falls as you add layers while the training loss still falls, over-smoothing is a likely cause.

## Attention over neighbours

The GCN weight on a neighbour, $1/\sqrt{d_i d_j}$, depends only on degrees. It cannot say that one neighbour matters more than another because of what the neighbour contains. [Attention](/lessons/056-attention-as-soft-lookup) can.

Velickovic and colleagues' graph attention network (GAT) scores each neighbour of node $i$ from the two feature vectors, takes a softmax over only the neighbours (and the node itself), and blends the neighbours' transformed features with those weights. The adjacency matrix acts as a mask, the way the causal mask in [scaled attention](/lessons/059-scaled-attention-and-heads) hides future positions: scores for pairs with no edge are set to minus infinity before the softmax, so they get weight zero. The result is the same blend you wrote for a soft lookup, with a neighbourhood as the set of keys.

Read that backward and a Transformer layer is message passing on a complete graph, where every token is linked to every other. The two ideas meet there. A GCN fixes the weights from the graph structure, while attention learns them from the content.

## Graph-level readout

To predict one value for a whole graph, such as whether a molecule is toxic, you need one vector from $n$ node vectors. The **readout** reduces over the node axis with an order-free function: sum, mean, or max. The result is the same under every renumbering, which is invariance. The seeded check finds a difference of $1.1 \times 10^{-16}$, which is floating-point rounding. Sum keeps the size of the graph, mean divides it out, and max records whether any node had a feature present.

## Putting it together: two communities from six labels

You now have every piece for a graph classifier. The exercise builds a two-layer GCN:

$$
\text{logits} = \hat A \; \operatorname{ReLU}(\hat A X W_1) \; W_2
$$

The graph is a **stochastic block model**: 60 nodes in two communities of 30, with a link probability of 0.25 inside a community and 0.02 across. The seeded draw has 239 edges, and 216 stay inside a community. Each node has two weak features, a shift of 0.3 buried in noise of standard deviation 1.

Only six nodes carry labels, three per community. Nearest-class-mean on the features alone gets 61 percent of the 54 unlabelled nodes right, barely above a coin flip. The graph supplies what the features lack: after two layers each node averages over its community, so the noise cancels. The reference solution reaches 100 percent. The loss is softmax cross-entropy over the six labelled rows, and its gradient reaches the other nodes through $\hat A$.

## Cost and scaling

A dense $\hat A$ has $n^2$ entries, so $\hat A H$ costs $n^2 d$ operations. Real graphs are sparse, so a sparse layer costs about $m d$ for $m$ edges. The exercise uses dense matrices because its graphs are small.

## Where this breaks

- **Over-smoothing.** Deep stacks wash the nodes together. Keep the stack shallow or add residual connections.
- **Limited power.** A message-passing layer cannot separate a ring from two triangles. Positional features break such ties.
- **Wrong graph.** The method assumes that linked nodes are similar. If links join opposite classes, averaging mixes the classes.

## Build it step by step

> [!RECAP]
> - A graph's nodes have no order, so a layer must be equivariant for node outputs and invariant for graph outputs.
> - Message passing aggregates the neighbours with an order-free function, then updates each node with shared weights.
> - The GCN layer is $\operatorname{ReLU}(\hat A H W)$ with $\hat A = D^{-1/2}(A + I)D^{-1/2}$, symmetric and with self-loops.
> - Sum, mean, and max each lose different information, and all of them are bounded by the Weisfeiler-Lehman test.
> - Stacking many layers averages the nodes together, which is over-smoothing. Attention learns the neighbour weights from content.

1. **`normalised_adjacency(A)`** takes an `(n, n)` 0/1 matrix and returns an `(n, n)` matrix. Add `np.eye(n)`, take the row sums $d$, and divide entry $(i, j)$ by `sqrt(d[i] * d[j])`. The mistake people most often make is to divide each row by its own sum, which is not symmetric, so the star example in the checks catches it.
2. **`gcn_layer(A_hat, H, W, relu=True)`** returns `A_hat @ H @ W`, passed through `np.maximum(., 0)` when `relu` is true.
3. **`gcn_layer_backward(A_hat, H, W, d_out, relu=True)`** returns `(dH, dW)`. Zero `d_out` where the pre-activation is not positive, then use `(A_hat @ H).T @ d_pre` and `A_hat.T @ d_pre @ W.T`. Forgetting the ReLU mask is the usual mistake, and the finite-difference check fails with the ReLU on.
4. **`message_pass(A, H, agg)`** takes `agg` of `"sum"`, `"mean"`, or `"max"` and returns an `(n, d)` array. Use `np.flatnonzero(A[i])` for the neighbours. A node with no neighbours gets zeros, not an error from an empty max.
5. **`readout(H, kind)`** reduces over axis 0. Axis 1 gives one number per node.
6. **`oversmoothing_curve(A, H, layers)`** returns `layers + 1` spreads. Measure first, then apply `A_hat @ X`, and rescale rows by `1 / sqrt(deg + 1)` before measuring. Without the rescale the curve levels off above zero.
7. **`train_gcn(A, X, y, train_idx, ...)`** runs the two-layer network with gradient descent and returns predicted classes for every node. Send the gradient through the second layer first, then the first, and divide the loss by the number of labelled rows. The starter provides `two_community_graph`.

## The exercise

You build the graph layer in numpy. `normalised_adjacency` and `gcn_layer` form the core, `gcn_layer_backward` gives the gradients, `message_pass` and `readout` are the order-free reductions, `oversmoothing_curve` measures the effect of depth, and `train_gcn` trains the two-layer network on `two_community_graph`.

The checks run in that order. They test the star hand example, symmetry, and the fixed vector $\sqrt{d}$ that a row-normalised matrix lacks. They check the gradients against finite differences with the ReLU on and off, the three aggregations on a graph with an isolated node, and permutation equivariance and invariance. The last two require the spread to fall on a bipartite path, where a version without self-loops fails, and the trained network to reach 90 percent accuracy on the 54 unlabelled nodes.

## Further reading

- The GCN layer: Kipf and Welling, ["Semi-supervised classification with graph convolutional networks"](https://arxiv.org/abs/1609.02907), 2017.
- Message passing as a general framework: Gilmer and others, ["Neural message passing for quantum chemistry"](https://arxiv.org/abs/1704.01212), 2017.
- The limit set by the Weisfeiler-Lehman test: Xu, Hu, Leskovec, and Jegelka, ["How powerful are graph neural networks?"](https://arxiv.org/abs/1810.00826), 2019.
- Attention over neighbours: Velickovic and others, ["Graph attention networks"](https://arxiv.org/abs/1710.10903), 2018.
- Over-smoothing: Li, Han, and Wu, ["Deeper insights into graph convolutional networks for semi-supervised learning"](https://arxiv.org/abs/1801.07606), 2018.
- Course material: the Stanford CS224W [schedule](https://web.stanford.edu/class/cs224w/) lists "3. Graph neural networks" and "6. Theory of GNNs". The MIT OpenCourseWare [6.7960 Fall 2024 lecture notes](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/lists/lecture-notes/) list "Lecture 5: Graph Neural Networks", and the [6.S898 Fall 2023 schedule](https://phillipi.github.io/6.s898/) lists "Architectures: Graphs" on Thursday 9/22 without a lecture number.
