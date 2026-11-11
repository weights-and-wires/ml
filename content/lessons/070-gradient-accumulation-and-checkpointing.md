---
title: "Gradient accumulation and activation checkpointing"
slug: 070-gradient-accumulation-and-checkpointing
part: "Part 6 — Transformers"
summary: Fit a bigger batch or a deeper model in the same memory, by adding up micro-batch gradients and by recomputing activations instead of storing them.
minutes: 60
gpu: false
objectives:
  - Split the memory of one training step into weights, gradients, optimizer state, and activations, and say which of them grow with batch size and depth.
  - Accumulate micro-batch gradients into the exact full-batch gradient, including when the micro-batches differ in size.
  - Name the cases where accumulation is not the same as a large batch, such as batch norm statistics and dropout.
  - Store only segment boundaries and recompute the rest in the backward pass, and count the memory saved and the extra forward work.
  - Derive why a segment count near the square root of the number of layers minimises the stored activations.
prereqs:
  - 069-training-loop-and-cost
exercise: 070-gradient-accumulation-and-checkpointing
---

# Gradient accumulation and activation checkpointing

You want to train with a batch of 128 sequences, because that is the batch size your recipe was tuned for. On your GPU, a batch of 32 already uses most of the memory, and a batch of 128 stops with an out-of-memory error on the first step. Shrinking the batch to 32 runs, but it changes the training: the gradient is noisier, and the learning rate that worked at 128 may no longer be right. Is there a way to get the update of a batch of 128 while holding only 32 sequences at a time?

A second wall waits behind the first. Suppose you drop to a batch of 8 and the memory still does not fit, because the model is deep. The backward pass reads the output of every layer from the forward pass, so those outputs must sit in memory until the gradient comes back to them. Doubling the depth doubles that pile. Is there a way to keep fewer of them and still get the exact gradient?

Both questions have the same shape. Memory is the limit, and time is the thing you are willing to spend. This chapter builds the two standard trades. Gradient accumulation spends extra steps of the same size to act like a bigger batch. Activation checkpointing spends one extra forward pass to hold far fewer activations. Both give the gradient you would have computed without the trick, up to rounding, and both are a few lines of code once you see why they are exact.

## What you already know

- A training step has a forward pass, a backward pass, a clip, and an optimizer update (see [the training step](/lessons/069-training-loop-and-cost)).
- AdamW keeps two extra numbers per weight, so a step holds about $16N$ bytes of fp32 numbers for $N$ weights (see [what a run costs](/lessons/069-training-loop-and-cost)).
- The loss is a mean over examples, so its gradient is the mean of the per-example gradients (see [stochastic gradient descent](/lessons/038-sgd-adam-and-schedules)).
- Backpropagation applies the chain rule layer by layer, from the last layer to the first, and each layer needs its own input (see [backpropagation](/lessons/036-backpropagation)).
- Batch norm normalises with the mean and variance of the current batch, and layer norm normalises each example alone (see [normalisation layers](/lessons/040-normalisation-layers)).

## Where the memory goes

Open a training step and count what sits in memory at the moment the backward pass is busiest. There are four kinds of thing.

| Piece | Size | Grows with |
|---|---|---|
| Weights | $4N$ bytes in fp32 | The model. |
| Gradients | $4N$ bytes | The model. |
| Optimizer state, $m$ and $v$ | $8N$ bytes | The model. |
| Activations | Depends on the batch | Batch size, sequence length, and depth. |

The first three rows are the $16N$-byte bill from the last chapter. They do not change when you change the batch. The fourth row is different. An **activation** is the output a layer produced in the forward pass and that the backward pass needs later. The chain rule needs the layer's input to compute its gradient, so every layer's input must be kept until the gradient reaches it.

Take a model with 24 layers and width 1024, which has about $12 \times 24 \times 1024^2 \approx 3.0 \times 10^8$ weights. The first three rows total 4.83 GB. Count activations with the simplest possible rule: keep one array of shape (tokens, width) per layer, in fp32. With sequences of 1,024 tokens:

- A batch of 8 holds 8,192 tokens, and the activations take $8192 \times 1024 \times 24 \times 4 = 0.81$ GB.
- A batch of 32 takes 3.22 GB.
- A batch of 128 takes 12.88 GB.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Three stacked bars of training memory for batch 8, 32, and 128. The weights, gradients, and optimizer state stay at 4.83 gigabytes while the activations grow from 0.81 to 12.88 gigabytes.">
  <text x="30" y="26" class="fig-bold">Memory of one training step (24 layers, width 1024, 1,024 tokens per sequence)</text>
  <line x1="130" y1="50" x2="130" y2="200" class="stroke-muted"/>
  <text x="30" y="92">batch 8</text>
  <rect x="130" y="70" width="25.8" height="34" class="fill-a"/>
  <rect x="155.8" y="70" width="25.8" height="34" class="fill-b"/>
  <rect x="181.6" y="70" width="51.5" height="34" class="fill-c"/>
  <rect x="233.1" y="70" width="17.3" height="34" class="fill-d"/>
  <text x="262" y="92" class="fig-small">5.64 GB</text>
  <text x="30" y="142">batch 32</text>
  <rect x="130" y="120" width="25.8" height="34" class="fill-a"/>
  <rect x="155.8" y="120" width="25.8" height="34" class="fill-b"/>
  <rect x="181.6" y="120" width="51.5" height="34" class="fill-c"/>
  <rect x="233.1" y="120" width="69" height="34" class="fill-d"/>
  <text x="314" y="142" class="fig-small">8.05 GB</text>
  <text x="30" y="192">batch 128</text>
  <rect x="130" y="170" width="25.8" height="34" class="fill-a"/>
  <rect x="155.8" y="170" width="25.8" height="34" class="fill-b"/>
  <rect x="181.6" y="170" width="51.5" height="34" class="fill-c"/>
  <rect x="233.1" y="170" width="276" height="34" class="fill-d"/>
  <text x="520" y="192" class="fig-small">17.72 GB</text>
  <rect x="130" y="226" width="12" height="12" class="fill-a"/><text x="146" y="237" class="fig-small">weights</text>
  <rect x="212" y="226" width="12" height="12" class="fill-b"/><text x="228" y="237" class="fig-small">gradients</text>
  <rect x="302" y="226" width="12" height="12" class="fill-c"/><text x="318" y="237" class="fig-small">m and v</text>
  <rect x="392" y="226" width="12" height="12" class="fill-d"/><text x="408" y="237" class="fig-small">activations</text>
</svg>
Figure 1. The first three blocks are the same in every row. Only the activation block grows with the batch, and at 128 sequences it is larger than everything else combined.
```

The activation block is a floor, not a forecast. A real transformer block keeps several arrays per layer, among them the attention weights, the wide hidden layer of the feed-forward network, and the normalised inputs. The shape of the result still holds. Weights, gradients, and optimizer state are fixed by the model. Activations scale with batch times sequence length times depth, which are the three things you want to make larger.

> [!KEY]
> A training step's memory is a fixed part and a part you can trade. Weights, gradients, and optimizer state cost $16N$ bytes in fp32 whatever the batch is. Activations cost roughly batch times sequence length times width times depth, and both tricks in this chapter shrink that part.

## Accumulate gradients to act like a bigger batch

The loss over a batch of $N$ examples is the mean of the per-example losses, so its gradient is the mean of the per-example gradients:

$$
g = \frac{1}{N} \sum_{i=1}^{N} g_i
$$

Read this aloud: the gradient of the batch is the average of the gradients of its examples. The right side does not care how you group the sum. You can add the examples in one go, or add them in groups and then combine the groups. That is the whole idea. Run the forward and backward pass on 32 examples, keep the gradient, free the activations, run the next 32, and add. After four groups you have summed all 128 per-example gradients, and one optimizer step uses the result.

Do it with numbers small enough to check by hand. One weight $w = 1$, three examples $(x, y) = (1, 3), (2, 2), (3, 0)$, and the loss $\tfrac12 (w x - y)^2$ per example. The per-example gradient is $(w x - y)\,x$:

| Example | $(x, y)$ | $(wx - y)\,x$ |
|---|---|---|
| 1 | (1, 3) | $(1 - 3) \times 1 = -2$ |
| 2 | (2, 2) | $(2 - 2) \times 2 = 0$ |
| 3 | (3, 0) | $(3 - 0) \times 3 = 9$ |

The full-batch gradient is $(-2 + 0 + 9)/3 = 7/3 \approx 2.333$. Split the batch into a micro-batch of one example and a micro-batch of two. Each micro-batch backward pass returns the mean over its own examples: $-2$ for the first, and $(0 + 9)/2 = 4.5$ for the second. Three ways to combine them are tempting.

- **Sum the means.** $-2 + 4.5 = 2.5$. This is too large, and for $k$ equal micro-batches it is exactly $k$ times too large.
- **Average the means.** $(-2 + 4.5)/2 = 1.25$. This is wrong because the second micro-batch holds twice as many examples, so it deserves twice the say.
- **Weight each mean by its share of the examples.** $\tfrac13(-2) + \tfrac23(4.5) = 2.333$. This matches.

The rule is to weight each micro-batch's mean loss and gradient by $n_i / N$, where $n_i$ is the micro-batch size and $N$ is the total:

$$
g = \sum_{i=1}^{k} \frac{n_i}{N}\, \bar g_i
$$

Here $\bar g_i$ is the mean gradient of micro-batch $i$. When all sizes are equal, $n_i / N = 1/k$ and the rule becomes "average the micro-batch means". That is the version people remember, and it fails the moment the last micro-batch is short, which happens whenever the dataset size is not a multiple of the micro-batch size. A language model has the same trap one level down. If the loss is a mean over tokens and sequences are padded or masked, the weight is the number of counted tokens in the micro-batch, not the number of sequences.

In code, the loop adds a weighted gradient into a buffer and takes one optimizer step at the end:

```python
acc = [np.zeros_like(W) for W in params]
for (x, y), w in zip(micro_batches, weights):   # weights = n_i / N
    loss, grads = mean_loss_grads(params, x, y)
    for a, g in zip(acc, grads):
        a += w * g
# one clip and one optimizer step with acc
```

Check the pieces of a real loop against this. The gradient clip from the last chapter acts on the accumulated gradient, after the loop, because a clip on each micro-batch is not the clip of the sum. The learning-rate schedule counts optimizer steps, not micro-batches. And the cost is time: four micro-batches take about four times as long as one, so accumulation makes a step slower in exchange for fitting.

> [!KEY]
> Accumulation is exact because the mean is a sum. Weight each micro-batch's mean by its share $n_i / N$, add, and take one optimizer step. Summing the means is $k$ times too large, and averaging them is wrong whenever the sizes differ.

## Where accumulation breaks

Accumulation reproduces the large batch only when the loss for one example does not depend on the other examples in its batch. Two common layers break that.

**Batch norm.** A batch norm layer subtracts the batch's mean and divides by its standard deviation, so every example's output depends on its batch-mates. Take the six values 1 to 6. As one batch, the mean is 3.5 and the variance is 2.92, so the value 1 becomes $(1 - 3.5)/\sqrt{2.92} = -1.46$. Split into two micro-batches of three, the first has mean 2 and variance 0.67, so the same value 1 becomes $(1 - 2)/\sqrt{0.67} = -1.22$. The forward pass is different, so no rearrangement of the gradients can recover the large-batch gradient. This is one reason transformers use layer norm, which normalises each example on its own and is safe under accumulation.

**Dropout and other randomness.** Dropout draws a random mask per example. Accumulation is still a valid way to train, because each example gets some mask, but a run with 4 micro-batches does not reproduce a run with one big batch bit for bit unless you control the random streams. The same holds for any layer that samples.

A third difference is small but real. Floating-point addition is not associative, so summing in groups gives the same gradient up to rounding, about $10^{-16}$ relative in float64 and a few units in the last place in fp32. The exercise checks to $10^{-10}$ in float64.

> [!WARNING]
> Accumulation does not fix layers that mix examples. With batch norm, a micro-batch of 4 normalises with 4 examples, not 128, so the model trains and evaluates differently from the large-batch recipe. Prefer layer norm, or keep the micro-batch large enough for stable statistics.

## Checkpoint activations to hold fewer of them

Accumulation shrinks the activation block by shrinking the batch. Depth is untouched: a micro-batch of 8 still stores one array for each of the 24 layers. To cut that, change what you store.

The picture is a long road with a rock every few steps. Without checkpointing you remember every step of the walk, so you can walk back. With it, you remember only the rocks. To walk back over a stretch, you start at the rock, replay the walk forward to rebuild the steps in that stretch, and then retrace them. You spend extra walking to save memory.

In words, the method has four parts:

1. Cut the chain of $L$ layers into $k$ **segments**, each a run of consecutive layers.
2. In the forward pass, keep only the input to each segment, the **boundary**, and drop every array inside a segment.
3. In the backward pass, take the segments from last to first. For one segment, run its layers forward again from its boundary to rebuild the inputs inside it, and keep them.
4. Backpropagate through the segment with those inputs, then free them and move to the previous segment.

Work a chain of 4 layers cut into $k = 2$ segments, layers 1 and 2, and layers 3 and 4. Plain backprop keeps four layer inputs, $h_0, h_1, h_2, h_3$. Checkpointing keeps two boundaries, $h_0$ and $h_2$. The backward pass starts with the segment holding layers 3 and 4. It needs $h_2$, which is stored, and $h_3 = f_3(h_2)$, which it rebuilds with one forward call. It holds the two boundaries plus $h_3$, so the peak is three arrays instead of four. Then it does the same for layers 1 and 2, rebuilding $h_1 = f_1(h_0)$. The extra work is two forward calls, one per segment.

The count is $L - k$, and the reason is that the last layer of a segment never needs its output rebuilt. Its backward step needs its input and the gradient arriving from above, and the next segment's boundary already holds its output. A segment of $s$ layers therefore rebuilds $s - 1$ inputs. Summed over segments that is $L - k$ forward calls, which is less than one full extra forward pass and close to it when $k$ is small. If you count every layer's output rebuilt, the total is $L$, and then it is exactly one extra forward pass. This matches the cost reported by Chen and colleagues in [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174): one extra forward pass per mini-batch.

The peak count is the $k$ boundaries, plus the interior inputs of the segment being processed. The first input of a segment is its boundary, which is already counted, so a segment of $s$ layers adds $s - 1$:

$$
\text{peak}(L, k) = k + \left\lceil \frac{L}{k} \right\rceil - 1
$$

Read it aloud: the boundaries you keep, plus the inside of the largest segment, minus the one input that is both. For $L = 4$ and $k = 2$ it gives $2 + 2 - 1 = 3$, matching the walk above.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Bar chart of peak stored layer inputs for 16 layers against the number of segments from 1 to 16. The peak is 16 at one segment, falls to a minimum of 7 at four segments, and rises back to 16 at sixteen segments.">
  <text x="30" y="22" class="fig-bold">Peak stored layer inputs, 16 layers</text>
  <line x1="55" y1="190" x2="520" y2="190" class="stroke-muted"/>
  <line x1="55" y1="40" x2="55" y2="190" class="stroke-muted"/>
  <rect x="70" y="40.0" width="20" height="150.0" class="fill-a"/>
  <rect x="98" y="105.6" width="20" height="84.4" class="fill-a"/>
  <rect x="126" y="115.0" width="20" height="75.0" class="fill-a"/>
  <rect x="154" y="124.4" width="20" height="65.6" class="fill-b"/>
  <rect x="182" y="115.0" width="20" height="75.0" class="fill-a"/>
  <rect x="210" y="115.0" width="20" height="75.0" class="fill-a"/>
  <rect x="238" y="105.6" width="20" height="84.4" class="fill-a"/>
  <rect x="266" y="105.6" width="20" height="84.4" class="fill-a"/>
  <rect x="294" y="96.2" width="20" height="93.8" class="fill-a"/>
  <rect x="322" y="86.9" width="20" height="103.1" class="fill-a"/>
  <rect x="350" y="77.5" width="20" height="112.5" class="fill-a"/>
  <rect x="378" y="68.1" width="20" height="121.9" class="fill-a"/>
  <rect x="406" y="58.8" width="20" height="131.2" class="fill-a"/>
  <rect x="434" y="49.4" width="20" height="140.6" class="fill-a"/>
  <rect x="462" y="40.0" width="20" height="150.0" class="fill-a"/>
  <rect x="490" y="40.0" width="20" height="150.0" class="fill-a"/>
  <text x="66" y="34" class="fig-small">16</text>
  <text x="158" y="118" class="fig-small fig-bold">7</text>
  <text x="486" y="34" class="fig-small">16</text>
  <text x="73" y="206" class="fig-small">1</text>
  <text x="157" y="206" class="fig-small">4</text>
  <text x="493" y="206" class="fig-small">16</text>
  <text x="200" y="228" class="fig-small">number of segments k</text>
  <text x="270" y="60" class="fig-small">no checkpointing stores 16</text>
  <line x1="268" y1="56" x2="248" y2="44" class="stroke-muted dashed"/>
</svg>
Figure 2. The peak is 16 at both ends, where checkpointing saves nothing, and lowest at 4 segments, the square root of 16. The curve is flat near its minimum, so 3 to 6 segments are all within one input of the best.
```

At the left edge, $k = 1$, one segment holds everything, and you rebuild the whole chain at once. At the right edge, every layer is its own segment, so you keep every input and rebuild nothing. The saving lives between.

## Pick the segment count

For large $L$ ignore the ceiling and the $-1$, so the peak is about $k + L/k$. Two forces pull in opposite directions. More segments mean more boundaries to keep, which is the $k$ term. Fewer segments mean longer segments to rebuild at once, which is the $L/k$ term. The sum is smallest where the two terms are equal. Set the slope to zero:

$$
\frac{d}{dk}\left( k + \frac{L}{k} \right) = 1 - \frac{L}{k^2} = 0 \quad\Rightarrow\quad k = \sqrt{L}
$$

At that point both terms equal $\sqrt{L}$, and the peak is $2\sqrt{L}$. For 16 layers that is $k = 4$ and a peak of about 8, and the exact formula gives 7. For 100 layers it is $k = 10$ and a peak of 19, against 100 without checkpointing. This is the sublinear memory in the title of the Chen paper: memory grows as $\sqrt{L}$ instead of $L$.

The price is compute. A forward pass costs about one unit and a backward pass about two, the same ratio as the $2ND$ and $4ND$ split behind the $6ND$ rule from the last chapter. A plain step costs three units. With one extra forward pass it costs four, so a step takes about a third longer. People pay that when it lets them use a larger micro-batch or a deeper model, and skip it when memory is not the limit.

Checkpointing has its own traps. The recomputed forward pass must produce exactly the arrays the original forward pass did, or the gradient is computed at the wrong point. A layer with dropout needs its random state saved before the segment and restored before the rebuild, so the same mask is drawn. A layer that updates running statistics, such as batch norm in training mode, would update them twice if the rebuild runs it again.

> [!WARNING]
> A recomputed segment must be deterministic. If dropout draws a different mask during the rebuild, the backward pass uses activations the loss never saw, and the gradients are wrong with no error raised. Save and restore the random state around every checkpointed segment.

## Where many GPUs come in

Everything in this chapter fits on one device. When the model or the batch outgrows a single GPU, the standard answer is to split the work across several, with data parallelism, sharded optimizer state (ZeRO), and tensor parallelism. Those are about moving and combining tensors between machines, which makes them a systems topic more than a modelling one, and the GPU course owns them. Its chapters on [collective communication](https://gpu.ifkash.dev/c/21-collective-communication) and on [data parallel training and ZeRO-1](https://gpu.ifkash.dev/c/25-ddp-and-zero) build the pieces on real hardware. Accumulation and checkpointing remain useful there, since each device still has its own activations.

> [!RECAP]
> - Training memory is weights, gradients, and optimizer state, which cost $16N$ bytes in fp32, plus activations, which grow with batch, sequence length, and depth.
> - Accumulation weights each micro-batch's mean loss and gradient by $n_i / N$ and takes one optimizer step. Summing the means is $k$ times too large, and averaging them is wrong for unequal sizes.
> - Accumulation fails for layers that mix examples, such as batch norm, and is only reproducible bit for bit when random streams are controlled.
> - Checkpointing keeps $k$ segment boundaries and rebuilds the inside of each segment in the backward pass, which costs $L - k$ extra forward calls and holds $k + \lceil L/k \rceil - 1$ inputs at peak.
> - The peak is smallest near $k = \sqrt{L}$, where it is about $2\sqrt{L}$, and the step takes about a third longer.

## Build it step by step

The checks test six functions in this order. The file already holds the chain model: `layer_forward`, `layer_backward`, `init_chain`, and `mean_loss_grads`, which is plain backprop storing every input.

1. **`activation_bytes(batch, width, layers, bytes_per_number=4)`.** Return the bytes for one array of shape `(batch, width)` per layer. Eight rows, width 4, and 3 layers in fp32 is 384. The usual mistake is to forget the number of layers, which reports one layer's worth.
2. **`micro_batch_weights(sizes)`.** Return each size divided by the total, as a float array. The sizes `[2, 3, 5]` give `[0.2, 0.3, 0.5]`. The usual mistake is integer division, which turns every share into 0.
3. **`accumulate_grads(params, batches)`.** Call `mean_loss_grads` once for each `(x, y)` micro-batch, weight its loss and its gradients by that micro-batch's share, and add them. Return `(loss, grads)`. The usual mistakes are summing without weights, which is $k$ times too large, and using $1/k$ weights, which fails when sizes differ.
4. **`stored_peak(layers, segments)` and `best_segment_count(layers)`.** The first returns $k + \lceil L/k \rceil - 1$. The second returns the count from 1 to $L$ with the smallest peak, taking the smaller count on a tie. For 16 layers it is 4. The usual mistake is to return `round(sqrt(L))` without checking the exact peak, which can miss on ties.
5. **`checkpointed_backward(params, x, y, segments)`.** Cut the layers with `np.array_split(np.arange(L), segments)`. In the forward pass keep each segment's input and nothing else. In the backward pass, for each segment from the last, rebuild its interior inputs with `layer_forward`, call `layer_backward` from the last layer to the first, and free the rebuilt inputs. Return `(loss, grads, stats)` with `peak_stored` and `recomputed_layers`. The usual mistake is to rebuild from the very first input each time, which gives the right gradients and the wrong counts.

## Where this breaks

- Accumulation with batch norm changes the statistics, as the worked numbers showed. Use layer norm or a larger micro-batch.
- A short last micro-batch breaks the $1/k$ rule. Weight by examples, or by counted tokens when the loss ignores padding.
- Clipping each micro-batch gradient is not the same as clipping the sum. Clip once, after the loop.
- Checkpointing with random layers needs saved and restored random state, or the gradients are wrong.
- The saving is small when $L$ is small. At 4 layers the best peak is 3 against 4, and the extra forward pass is not worth it.

## The exercise

You write `activation_bytes`, `micro_batch_weights`, `accumulate_grads`, `stored_peak`, `best_segment_count`, and `checkpointed_backward`. The checks match the worked 384 bytes and the shares 0.2, 0.3, and 0.5. They check the peaks 7, 16, 16, and 6 for the segment formula, and that `best_segment_count` always minimises it and stays under $2\sqrt{L} + 1$ for every $L$ up to 60. They compare `accumulate_grads` with the full-batch gradient to $10^{-10}$ for equal micro-batches and for sizes 1, 2, 5, and 8, with a finite-difference check on one layer. They compare `checkpointed_backward` with plain backprop for 1, 3, 4, and 10 segments. Last, they count the calls to `layer_forward` and confirm that 16 layers in 4 segments make 16 forward calls and 12 rebuilds, with a peak of 7 stored inputs against 16. Run builds a 16-layer chain and prints the gaps, which should be near zero.

## Further reading

- Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin, [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174), 2016. The origin of the segment scheme and the $\sqrt{L}$ rule.
- Stanford CS336, [schedule](http://cs336.stanford.edu/spring2025/), Lectures 7 and 8, "Parallelism". How accumulation fits with data and tensor parallelism.
- Deep Learning Systems, [lecture list](https://dlsyscourse.org/lectures/), Lecture 19, "Training Large Models".
- The GPU course, [collective communication](https://gpu.ifkash.dev/c/21-collective-communication) and [data parallel training and ZeRO-1](https://gpu.ifkash.dev/c/25-ddp-and-zero), for everything that needs more than one device.
