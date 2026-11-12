---
title: "Text embeddings and retrieval: pooling, InfoNCE, and recall@k"
slug: 077-text-embeddings-and-retrieval
part: "Part 7 — Large language models"
summary: Turn a decoder's hidden states into one vector per text, train it with InfoNCE and hard negatives, and score a search with recall@k and nDCG@k.
minutes: 70
gpu: false
objectives:
  - Pool a decoder's hidden states into one unit vector with last-token and mean pooling, and handle padding on either side.
  - Explain why causal attention makes the last token the natural summary of a text.
  - Compute InfoNCE for query-passage pairs by hand with in-batch and hard negatives, and say what the temperature does.
  - Compute recall@k and nDCG@k, with graded relevance, on a small corpus.
  - Describe the E5-Mistral recipe, including why queries carry an instruction and passages do not.
prereqs:
  - 051-contrastive-learning
  - 063-the-transformer-block
  - 076-evaluating-language-models
exercise: 077-text-embeddings-and-retrieval
---

# Text embeddings and retrieval: pooling, InfoNCE, and recall@k

A support team has 40,000 help articles and a customer who types "my card got declined abroad". The best article is titled "Using your account while travelling". The two share no words, so a keyword search ranks it nowhere. A search that compares meaning instead needs one vector per text, placed so that texts about the same thing sit close together. The customer's question becomes a vector, and the answer is the nearest article vectors.

You already have a model that reads text well: a decoder language model. It produces one hidden vector for every token, and none of them is a summary of the whole text, because predicting the next word never asked for one. This chapter shows how to get one vector out of those hidden states, how to train it with the contrastive loss from [contrastive learning](/lessons/051-contrastive-learning), and how to measure whether the search works. The recipe is the one behind E5-Mistral, and you build each piece on tiny arrays.

## What you already know

- InfoNCE is a row-wise softmax over cosine similarities, and it pulls a positive pair together while it pushes the other rows of the batch away ([contrastive learning](/lessons/051-contrastive-learning)).
- Dividing a vector by its length leaves a unit vector, and the dot product of two unit vectors is their cosine.
- A decoder block lets each token attend to the tokens before it and to itself, and never to later ones ([the transformer block](/lessons/063-the-transformer-block)).
- A held-out score needs an error bar ([evaluating language models](/lessons/076-evaluating-language-models)).
- A padded batch holds sequences of different lengths, and a mask marks which positions are real tokens.

## From a text to one vector

Feed a text of $L$ tokens through a decoder. The last layer returns $L$ hidden vectors, one per token, each of dimension $d$. A search index needs exactly one vector per text, so you must collapse $L$ vectors into one. This step is called **pooling**.

**Mean pooling** averages the $L$ vectors. **Last-token pooling** keeps only the vector at the final token. Mean pooling looks fairer, because every token votes, and last-token pooling looks wasteful. For a decoder the wasteful choice is the natural one, and Figure 1 shows why.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="A 5 by 5 causal attention grid: position 1 sees one token and position 5, the last, sees all five">
<text x="120" y="28" class="fig-bold">token that is read (key)</text>
<text x="150" y="42" text-anchor="middle" class="fig-mono">t1</text>
<text x="210" y="42" text-anchor="middle" class="fig-mono">t2</text>
<text x="270" y="42" text-anchor="middle" class="fig-mono">t3</text>
<text x="330" y="42" text-anchor="middle" class="fig-mono">t4</text>
<text x="390" y="42" text-anchor="middle" class="fig-mono">EOS</text>
<text x="110" y="75" text-anchor="end" class="fig-mono">t1</text>
<rect x="120" y="50" width="60" height="40" class="fill-soft-a stroke-grid"/>
<rect x="180" y="50" width="60" height="40" class="fill-sunk stroke-grid"/>
<rect x="240" y="50" width="60" height="40" class="fill-sunk stroke-grid"/>
<rect x="300" y="50" width="60" height="40" class="fill-sunk stroke-grid"/>
<rect x="360" y="50" width="60" height="40" class="fill-sunk stroke-grid"/>
<text x="434" y="75" class="fig-small">sees 1 token</text>
<text x="110" y="115" text-anchor="end" class="fig-mono">t2</text>
<rect x="120" y="90" width="60" height="40" class="fill-soft-a stroke-grid"/>
<rect x="180" y="90" width="60" height="40" class="fill-soft-a stroke-grid"/>
<rect x="240" y="90" width="60" height="40" class="fill-sunk stroke-grid"/>
<rect x="300" y="90" width="60" height="40" class="fill-sunk stroke-grid"/>
<rect x="360" y="90" width="60" height="40" class="fill-sunk stroke-grid"/>
<text x="434" y="115" class="fig-small">sees 2 tokens</text>
<text x="110" y="155" text-anchor="end" class="fig-mono">t3</text>
<rect x="120" y="130" width="60" height="40" class="fill-soft-a stroke-grid"/>
<rect x="180" y="130" width="60" height="40" class="fill-soft-a stroke-grid"/>
<rect x="240" y="130" width="60" height="40" class="fill-soft-a stroke-grid"/>
<rect x="300" y="130" width="60" height="40" class="fill-sunk stroke-grid"/>
<rect x="360" y="130" width="60" height="40" class="fill-sunk stroke-grid"/>
<text x="434" y="155" class="fig-small">sees 3 tokens</text>
<text x="110" y="195" text-anchor="end" class="fig-mono">t4</text>
<rect x="120" y="170" width="60" height="40" class="fill-soft-a stroke-grid"/>
<rect x="180" y="170" width="60" height="40" class="fill-soft-a stroke-grid"/>
<rect x="240" y="170" width="60" height="40" class="fill-soft-a stroke-grid"/>
<rect x="300" y="170" width="60" height="40" class="fill-soft-a stroke-grid"/>
<rect x="360" y="170" width="60" height="40" class="fill-sunk stroke-grid"/>
<text x="434" y="195" class="fig-small">sees 4 tokens</text>
<text x="110" y="235" text-anchor="end" class="fig-mono">EOS</text>
<rect x="120" y="210" width="60" height="40" class="fill-soft-b stroke-b"/>
<rect x="180" y="210" width="60" height="40" class="fill-soft-b stroke-b"/>
<rect x="240" y="210" width="60" height="40" class="fill-soft-b stroke-b"/>
<rect x="300" y="210" width="60" height="40" class="fill-soft-b stroke-b"/>
<rect x="360" y="210" width="60" height="40" class="fill-soft-b stroke-b"/>
<text x="434" y="235" class="fig-small">sees 5 tokens</text>
<text x="20" y="150" class="fig-bold" transform="rotate(-90 20 150)" text-anchor="middle">position that is written</text>
<text x="120" y="280" class="fig-small">Blue cells are allowed. The orange row is the last token, the only one that reads the whole text.</text>
</svg>
Figure 1. Causal attention for a five-token text. Position t reads tokens 1 to t and nothing after. The first token's vector depends on one token, and the last token's vector depends on all five.
```

Under a **causal mask**, token $t$ attends only to tokens $1$ through $t$. The hidden vector at token 1 is a function of token 1 alone, so two texts that begin with the same word start with the same vector, whatever follows. The vector at the last token is the only one that has read the whole text. That makes it the natural place to store a summary, and it makes the average a mixture of vectors that each read part of the text.

The E5-Mistral recipe appends an end-of-sequence token, written `[EOS]`, to every text and uses the last-layer vector at that token as the embedding. Pretraining gave that position the job of predicting what comes next. Contrastive fine-tuning gives it another job: summarise what came before.

> [!KEY]
> A causal decoder lets only its last token read the whole text, so last-token pooling reads the one vector that could hold a summary. Mean pooling averages vectors that each saw a prefix.

### A worked pooling example

Take a text with three real tokens and $d = 2$. Its last-layer vectors are

$$
h_1 = (1, 0), \qquad h_2 = (0, 2), \qquad h_3 = (2, 2)
$$

The mean is $\left(\tfrac{1+0+2}{3}, \tfrac{0+2+2}{3}\right) = (1, 1.333)$. Last-token pooling returns $h_3 = (2, 2)$. Divide by its length $\sqrt{8} = 2.828$ and you get the unit vector $(0.707, 0.707)$.

Batches are padded to one length, so the text sits in a matrix of four rows. Where the padding goes changes which row is "last":

- **Right padding** puts the real tokens first: rows $h_1, h_2, h_3, \text{pad}$. Row $-1$ is padding.
- **Left padding** puts the padding first: rows $\text{pad}, h_1, h_2, h_3$. Row $-1$ is $h_3$.

> [!WARNING]
> Taking row $-1$ works for left-padded batches and silently returns a padding vector for right-padded ones. Make the padding rows hold the value 9 and the mistake gives $(9, 9)$ instead of $(2, 2)$. The same trap hits mean pooling: averaging all four rows gives $(3, 3.25)$, because the padding counts as a token. Always use the mask.

The mask fixes both. For last-token pooling, the index of the last real token is the largest position where the mask is 1. For mean pooling, multiply the vectors by the mask and divide by the number of real tokens, not by the padded length:

$$
e_{\text{last}} = h_{\ell}, \quad \ell = \max\{t : m_t = 1\}, \qquad
e_{\text{mean}} = \frac{1}{\sum_t m_t} \sum_t m_t h_t
$$

Read it aloud: the last-token embedding is the vector at the last position where the mask is 1, and the mean embedding is the masked sum divided by the count of real tokens.

```python
positions = np.arange(T)[None, :]
last = np.where(mask > 0, positions, -1).max(axis=1)   # index of the last real token
e_last = hidden[np.arange(B), last]
e_mean = (hidden * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
```

### Normalise, then compare with cosine

A pooled vector's length depends on the text, and a length is not a meaning. A passage that produces a bigger vector would get bigger dot products with every query. Divide each embedding by its length. For unit vectors the dot product equals the cosine of the angle between them:

$$
\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert} = \hat a \cdot \hat b
$$

Here $\hat a = a / \lVert a \rVert$. A search is then one matrix product: stack the unit passage embeddings into a matrix $D$, and the scores for a query $q$ are $D q$. Rank by score.

## Teach the vectors with InfoNCE

To make closeness mean "the passage answers the query", you fine-tune with a contrastive loss. [Contrastive learning](/lessons/051-contrastive-learning) derives InfoNCE and its gradient, so this section only adapts it from two image views to a query and a passage.

A training example is a **query** $q$ and a **positive passage** $d^+$ that answers it. A batch holds $B$ such pairs. For query $i$, the other $B - 1$ positives belong to other queries, so they answer different questions. Those are the **in-batch negatives**, and they cost nothing. Each example can also carry a **hard negative**: a passage that looks relevant and is not, such as another article about card fees. A hard negative is hard because it sits close to the query, which is where the model still makes mistakes.

Stack all the passages into one matrix of $M$ rows: the $B$ positives first, then the hard negatives. Query $i$ must pick row $i$ out of all $M$. With a similarity $s$ (the cosine) and a temperature $\tau$, the loss is

$$
L = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(s(q_i, d_i)/\tau)}{\sum_{j=1}^{M} \exp(s(q_i, d_j)/\tau)}
$$

Read it aloud: for each query, take the probability that a softmax over every passage gives its own positive, take the negative log, and average over the batch. A hard negative that belongs to another query's example still counts as a negative for this query, so the denominator holds all $M$ passages.

### Work the example

Use two queries and unit vectors in two dimensions, with $\tau = 0.1$:

$$
q_1 = (1, 0), \quad q_2 = (0, 1), \quad
d_1 = (0.8, 0.6), \quad d_2 = (0.6, 0.8), \quad
h_1 = (0.6, -0.8), \quad h_2 = (0.8, -0.6)
$$

Here $d_1$ and $d_2$ are the positives and $h_1$, $h_2$ are the hard negatives. The cosines are dot products, and dividing by 0.1 multiplies them by 10. Query 1 gives the logits $8, 6, 6, 8$ for $d_1, d_2, h_1, h_2$. Query 2 gives $6, 8, -8, -6$. Figure 2 shows the softmax of each row.

```figure
<svg viewBox="0 0 640 300" role="img" aria-label="A 2 by 4 grid of softmax probabilities for two queries over two positives and two hard negatives at temperature 0.1">
<text x="172" y="40" text-anchor="middle" class="fig-small">p1 (positive of q1)</text>
<text x="277" y="58" text-anchor="middle" class="fig-small">p2 (positive of q2)</text>
<text x="382" y="40" text-anchor="middle" class="fig-small">h1 (hard, q1)</text>
<text x="487" y="58" text-anchor="middle" class="fig-small">h2 (hard, q2)</text>
<text x="110" y="110" text-anchor="end" class="fig-mono">q1</text>
<rect x="120" y="70" width="105" height="70" class="fill-soft-b stroke-b"/>
<text x="172" y="100" text-anchor="middle" class="fig-bold">p = 0.440</text>
<text x="172" y="122" text-anchor="middle" class="fig-small">logit 8</text>
<rect x="225" y="70" width="105" height="70" class="fill-soft-a stroke-grid"/>
<text x="277" y="100" text-anchor="middle" class="fig-bold">p = 0.060</text>
<text x="277" y="122" text-anchor="middle" class="fig-small">logit 6</text>
<rect x="330" y="70" width="105" height="70" class="fill-soft-d stroke-grid"/>
<text x="382" y="100" text-anchor="middle" class="fig-bold">p = 0.060</text>
<text x="382" y="122" text-anchor="middle" class="fig-small">logit 6</text>
<rect x="435" y="70" width="105" height="70" class="fill-soft-d stroke-grid"/>
<text x="487" y="100" text-anchor="middle" class="fig-bold">p = 0.440</text>
<text x="487" y="122" text-anchor="middle" class="fig-small">logit 8</text>
<text x="552" y="110" class="fig-small">loss 0.820</text>
<text x="110" y="180" text-anchor="end" class="fig-mono">q2</text>
<rect x="120" y="140" width="105" height="70" class="fill-soft-a stroke-grid"/>
<text x="172" y="170" text-anchor="middle" class="fig-bold">p = 0.119</text>
<text x="172" y="192" text-anchor="middle" class="fig-small">logit 6</text>
<rect x="225" y="140" width="105" height="70" class="fill-soft-b stroke-b"/>
<text x="277" y="170" text-anchor="middle" class="fig-bold">p = 0.881</text>
<text x="277" y="192" text-anchor="middle" class="fig-small">logit 8</text>
<rect x="330" y="140" width="105" height="70" class="fill-soft-d stroke-grid"/>
<text x="382" y="170" text-anchor="middle" class="fig-bold">p = 0.000</text>
<text x="382" y="192" text-anchor="middle" class="fig-small">logit -8</text>
<rect x="435" y="140" width="105" height="70" class="fill-soft-d stroke-grid"/>
<text x="487" y="170" text-anchor="middle" class="fig-bold">p = 0.000</text>
<text x="487" y="192" text-anchor="middle" class="fig-small">logit -6</text>
<text x="552" y="180" class="fig-small">loss 0.127</text>
<text x="120" y="240" class="fig-small">Orange: the positive. Purple: hard negatives. Mean loss 0.4735.</text>
</svg>
Figure 2. Softmax probabilities for two queries over two positives and two hard negatives at temperature 0.1. Query 1 gives its positive 0.440 because hard negative h2 scores as high as the positive does.
```

For query 1, divide every exponential by $e^8$. The four terms become $1,\ e^{-2},\ e^{-2},\ 1$, which are $1, 0.135, 0.135, 1$ and sum to 2.271. The positive's probability is $1 / 2.271 = 0.440$, and the row loss is $-\ln 0.440 = 0.820$. For query 2 the terms are $e^{-2}, 1, e^{-16}, e^{-14}$, which sum to 1.135, so the probability is 0.881 and the loss is 0.127. The mean is $L = (0.820 + 0.127)/2 = 0.474$.

Drop the hard negatives and keep only the two positives. Each row then has the probability $1/(1 + e^{-2}) = 0.881$ for its positive, and $L = 0.127$. The hard negative $h_2$ raised the loss from 0.127 to 0.474 because it looks as good as the right answer to query 1. The model must move $h_2$ away from $q_1$ or move $d_1$ closer.

### The temperature sets how hard a negative bites

Cosines live in $[-1, 1]$, so without a temperature every softmax would be nearly flat. Dividing by a small $\tau$ stretches the scores. A cosine gap of 0.2 becomes a logit gap of 2 at $\tau = 0.1$ and a gap of 10 at $\tau = 0.02$.

Rerun the example at $\tau = 0.02$. Query 1 has logits $40, 30, 30, 40$, so the hard negative $h_2$ ties the positive and the row loss is $\ln 2 = 0.693$. Query 2 has logits $30, 40, -40, -30$, and its loss is about $4.5 \times 10^{-5}$. The mean is 0.347. Without the hard negatives the loss at $\tau = 0.02$ is also $4.5 \times 10^{-5}$, so adding the two hard negatives multiplied it by about 7,600.

A low temperature makes the loss ignore easy negatives and focus on the ones that tie or beat the positive. That helps when the hard negatives are truly wrong, and hurts when one is a relevant passage nobody labelled. The E5-Mistral paper fixes $\tau$ at 0.02.

> [!KEY]
> InfoNCE treats a query's positive as the right answer in a multiple-choice question whose other options are the batch's other passages and the hard negatives. The temperature decides how sharply the loss punishes the options that score close to the answer.

### The gradient

Collect the query vectors into a matrix $Q$ with $B$ rows and the passages into $D$ with $M$ rows. The logits are $Z = Q D^{\top} / \tau$, a $B \times M$ grid. Let $P$ be the row softmax of $Z$ and let $Y$ be the $B \times M$ matrix with a 1 at $[i, i]$ and 0 elsewhere. Then softmax cross-entropy gives $\partial L / \partial Z = (P - Y)/B$, the prediction minus the truth, averaged over the batch. With $G = (P - Y)/(B\tau)$, the chain rule through $Z = QD^{\top}/\tau$ gives

$$
\frac{\partial L}{\partial Q} = G D, \qquad \frac{\partial L}{\partial D} = G^{\top} Q
$$

Read it aloud: a positive's entry in $G$ is negative, which is a pull, and every other entry is positive, which is a push, for the queries and the passages alike. The exercise checks this against finite differences.

## Queries carry an instruction, passages do not

One embedding model serves many tasks, and "answer this question" and "find a near-duplicate" call for different notions of closeness. E5-Mistral writes the task into the query: the query text becomes `Instruct: {task description}` on one line and `Query: {the query}` on the next. The passage is embedded as it is.

The asymmetry has a practical reason. You embed the passages once and store the vectors. If the instruction were part of the passage text, each task would need its own copy of the index. On the query side, only the short query is re-embedded when the task changes.

The paper reports that proprietary LLMs wrote the synthetic training data, and that it fine-tunes Mistral-7B with LoRA and a standard contrastive loss in fewer than 1,000 steps. The paper page [E5-Mistral](https://papers.ifkash.dev/e5-mistral/) walks through the details. Nothing in the loss is special: it is the InfoNCE you computed above, applied to last-token embeddings.

## Score the search: recall@k and nDCG@k

A trained embedder needs a test: a corpus, some queries, and for each query a **relevance grade** per passage, 0 for irrelevant and larger integers for more relevant. Rank the passages by score and ask how good the top of the ranking is. Use one query over six passages. The passages are $A$ to $F$ with grades $0, 3, 0, 2, 1, 0$. The model's cosines are $0.95, 0.90, 0.80, 0.60, 0.30, -0.20$, so it ranks them in the order $A, B, C, D, E, F$. Passage $A$ is topically close and wrong, and it sits at rank 1.

**Recall@k** asks what fraction of the relevant passages appear in the top $k$. Three passages are relevant ($B$, $D$, $E$). The top 3 holds only $B$, so recall@3 is $1/3$. The top 5 holds all three, so recall@5 is $3/3 = 1$. Recall ignores order inside the top $k$ and ignores the grades, so a grade-1 passage counts the same as a grade-3 passage.

**nDCG@k** uses both. The **discounted cumulative gain** adds up the grade of each passage in the top $k$, discounted by its rank, so that a hit at rank 1 counts more than a hit at rank 3:

$$
\mathrm{DCG@}k = \sum_{r=1}^{k} \frac{g_r}{\log_2(r + 1)}
$$

Here $g_r$ is the grade of the passage at rank $r$. The discounts are $1/\log_2 2 = 1$ at rank 1, $1/\log_2 3 = 0.631$ at rank 2, and $1/\log_2 4 = 0.5$ at rank 3. This chapter uses the grade itself as the gain. Some libraries use $2^{g} - 1$ instead, which rewards a top grade more, so state which one you use when you report a number.

The model's top 3 has grades $0, 3, 0$, so $\mathrm{DCG@3} = 0 + 3 \times 0.631 + 0 = 1.893$. That number depends on how large the grades are, so it can't be compared across queries on its own. Divide by the **ideal DCG**, which is the DCG of the best possible ranking, with the grades sorted from largest to smallest. Here the ideal top 3 has grades $3, 2, 1$, so $\mathrm{IDCG@3} = 3 + 2 \times 0.631 + 1 \times 0.5 = 4.762$. The ratio is

$$
\mathrm{nDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k}, \qquad \mathrm{nDCG@3} = \frac{1.893}{4.762} = 0.3975
$$

The ratio lies between 0 and 1, and it equals 1 exactly when the ranking puts the grades in the best possible order. Figure 3 draws the two sums side by side.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Bars of gain times discount at ranks 1 to 3 for the model ranking and for the ideal ranking, with DCG 1.893 and ideal DCG 4.762">
<text x="20" y="28" class="fig-bold">rank</text><text x="170" y="28" class="fig-bold">gain / log2(rank + 1)</text>
<text x="20" y="72" class="fig-mono">1</text>
<text x="52" y="64" class="fig-small">model: doc A, grade 0</text>
<rect x="170" y="50" width="1.0" height="18" class="fill-b"/>
<text x="178.0" y="64" class="fig-small">0.000</text>
<text x="52" y="88" class="fig-small">ideal: grade 3</text>
<rect x="170" y="74" width="210.0" height="18" class="fill-a"/>
<text x="388.0" y="88" class="fig-small">3.000</text>
<text x="20" y="132" class="fig-mono">2</text>
<text x="52" y="124" class="fig-small">model: doc B, grade 3</text>
<rect x="170" y="110" width="132.5" height="18" class="fill-b"/>
<text x="310.5" y="124" class="fig-small">1.893</text>
<text x="52" y="148" class="fig-small">ideal: grade 2</text>
<rect x="170" y="134" width="88.3" height="18" class="fill-a"/>
<text x="266.3" y="148" class="fig-small">1.262</text>
<text x="20" y="192" class="fig-mono">3</text>
<text x="52" y="184" class="fig-small">model: doc C, grade 0</text>
<rect x="170" y="170" width="1.0" height="18" class="fill-b"/>
<text x="178.0" y="184" class="fig-small">0.000</text>
<text x="52" y="208" class="fig-small">ideal: grade 1</text>
<rect x="170" y="194" width="35.0" height="18" class="fill-a"/>
<text x="213.0" y="208" class="fig-small">0.500</text>
<text x="20" y="244" class="fig-bold">DCG@3 = 1.8928, ideal DCG@3 = 4.7619, nDCG@3 = 0.3975</text>
</svg>
Figure 3. Gain divided by the rank discount at ranks 1 to 3, for the model's ranking and for the ideal one. The model wastes rank 1 on a grade-0 passage, which is where the ideal ranking earns the most.
```

> [!WARNING]
> Skipping the division by the ideal DCG is a common slip. The raw DCG of 1.893 grows with the grades, so you can't average it across queries. On the six-passage corpus the right answer is $\mathrm{nDCG@3} = 0.3975$, and any other value means the normaliser is missing or wrong.

> [!TIP]
> Report recall@k and nDCG@k together. Recall tells you whether the answer is in the list a reader sees, and nDCG tells you whether the best answers are near the top.

> [!KEY]
> Recall@k counts how many of the relevant passages the top $k$ finds. nDCG@k also rewards putting the most relevant passages first, and it divides by the ideal DCG so the score runs from 0 to 1.

> [!RECAP]
> - Pooling collapses a decoder's per-token vectors into one. Last-token pooling reads the `[EOS]` vector, which is the only one that has seen the whole text under a causal mask. Mean pooling averages over real tokens only.
> - A mask, not a fixed index, finds the last real token, so either padding side gives the same embedding. Normalise so a dot product is a cosine.
> - InfoNCE for retrieval is a softmax over the batch's positives plus the hard negatives, with a temperature. A low temperature, such as 0.02 in E5-Mistral, makes the loss focus on the negatives that tie the positive.
> - Queries carry a task instruction and passages do not, so one passage index serves every task.
> - Recall@k counts relevant passages found in the top $k$. nDCG@k weights hits by a log discount and divides by the ideal DCG.

## Build it step by step

You write nine functions, and the checks call them in this order.

1. **`last_token_pool(hidden, mask)`** takes `hidden` of shape `(batch, time, d)` and a 0/1 `mask` of shape `(batch, time)`, and returns shape `(batch, d)`. Find the largest position where the mask is 1 with `np.where` and `max(axis=1)`, then index with `hidden[np.arange(batch), last]`. The usual mistake is `hidden[:, -1]`, which fails on right-padded input.
2. **`mean_pool(hidden, mask)`** returns the masked average. Multiply by `mask[:, :, None]`, sum over time, and divide by `mask.sum(axis=1, keepdims=True)`. The usual mistake is to divide by the padded length.
3. **`l2_normalize(x)`** divides each row by its length and returns the same shape. Keep a small floor under the length so a zero row does not produce `nan`.
4. **`embed(hidden, mask, pooling)`** pools with `"last"` or `"mean"` and then normalises, so every row has length 1. The usual mistake is to normalise before pooling.
5. **`info_nce_loss(q, docs, tau)`** takes queries of shape `(B, d)` and a passage matrix of shape `(M, d)` whose first $B$ rows are the positives. Compute the logits `q @ docs.T / tau`, subtract each row's maximum, and return the mean of `logsumexp(row) - row[i]` for row $i$. The usual mistake is to build the softmax over the positives alone, which drops the hard negatives.
6. **`info_nce_grad(q, docs, tau)`** returns `(dq, ddocs)`. Build `G = (P - Y) / (B * tau)` and return `G @ docs` and `G.T @ q`. The usual mistake is to forget the factor $1/B$ or $1/\tau$.
7. **`recall_at_k(scores, relevance, k)`** ranks with `np.argsort(-scores, kind="stable")`, counts how many of the top $k$ have a grade above 0, and divides by the number of passages with a grade above 0. The usual mistake is to divide by $k$.
8. **`ndcg_at_k(scores, relevance, k)`** builds the discount `1 / np.log2(np.arange(2, k + 2))`, computes the DCG of your ranking and of the grades sorted in descending order, and returns the ratio. Return 0 when every grade is 0.
9. **`search(q_emb, doc_emb, k)`** returns the indices of the top $k$ passages per query, best first, from the matrix product `q_emb @ doc_emb.T`.

## Where this breaks

- **Missing end-of-sequence token.** Last-token pooling reads whatever the final token is. If a tokenizer does not append `[EOS]` for you, you pool a different vector than the one the model was trained on. Append the token yourself and check it is there.
- **False negatives.** A mined hard negative can be a relevant passage that no annotator labelled. At a temperature of 0.02, the loss pushes that passage away with force. Filter mined negatives with a stronger scorer, or skip the top-ranked ones.
- **Small batches give weak negatives.** A batch of 4 gives each query 3 in-batch negatives, so embedding recipes favour large batches.
- **A toy corpus flatters the model.** Six passages cannot show how a model behaves on a million. Real benchmarks use many queries and large corpora, and they need error bars like the ones in [evaluating language models](/lessons/076-evaluating-language-models).
- **Two nDCG conventions.** Linear gain and $2^g - 1$ gain give different numbers for the same ranking. Compare numbers only when both sides use the same one.

## The exercise

You write `last_token_pool`, `mean_pool`, `l2_normalize`, `embed`, `info_nce_loss`, `info_nce_grad`, `recall_at_k`, `ndcg_at_k`, and `search`. The checks confirm that pooling gives the same answer for right-padded and left-padded input and that mean pooling ignores padding. They check `info_nce_loss` against the hand example (0.4735 with the hard negatives, 0.1269 without) and `info_nce_grad` against finite differences. They check recall@3 of $1/3$ and nDCG@3 of 0.3975 on the six-passage corpus, and that a perfect ranking scores 1. A final check embeds a seeded toy corpus whose last token carries the topic and confirms that `search` returns every passage of the query's topic, with either padding side.

## Further reading

- For the recipe behind this chapter, see [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368) (Wang et al., 2023), which covers the `[EOS]` pooling, the query instruction, the synthetic data, and the contrastive loss with its temperature. The page [E5-Mistral](https://papers.ifkash.dev/e5-mistral/) on the papers site works through it.
- For the loss itself, including its derivation and the collapse it prevents, see [contrastive learning](/lessons/051-contrastive-learning).
- For a broader view of how embedders are evaluated across tasks and languages, see the benchmark paper [MMTEB](https://arxiv.org/abs/2502.13595) (arXiv 2502.13595).
