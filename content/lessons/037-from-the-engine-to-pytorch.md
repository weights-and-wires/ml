---
title: From your engine to PyTorch
slug: 037-from-the-engine-to-pytorch
part: "Part 5 — Vision and sequences"
summary: Map requires_grad, backward, and no_grad onto your tensor, then train a module and match an attention blend with einsum.
minutes: 70
gpu: false
objectives:
  - Map requires_grad, .grad, backward, and no_grad onto the tensor you built.
  - Write a module with registered parameters and a handwritten SGD step that zeros the gradient.
  - Batch rows with a seeded shuffle, and write the train loop and the eval loop.
  - Write the attention blend with einsum so it matches the loop from the attention chapter.
  - Name a forgotten zero_grad, train mode left on, an in-place op, and a dtype or device mismatch.
prereqs:
  - 029-backpropagation
  - 031-sgd-adam-and-schedules
exercise: 037-from-the-engine-to-pytorch
---

# From your engine to PyTorch

You trained [a digit reader](/lessons/036p-project-read-digits) by writing every slope yourself. That project stays in numpy, because the point was to see the arithmetic. The convolution and recurrent chapters still ask you to derive a slope when the slope is the idea. Later, [the training step](/lessons/054-training-loop-and-cost) stops writing slopes and calls PyTorch.

PyTorch records the same tape as the tensor in [backpropagation](/lessons/029-backpropagation). It stores a value, it stores a slope, and it walks that slope back through every operation. The names differ, and a few defaults differ with them. This chapter is the map from your names to those calls, plus a training loop small enough to check by hand.

This chapter does not rebuild the engine. It points each piece of your tensor at the call that does the same job, puts those calls inside a module, and fits four points that sit on a straight line.

## What you already know

- A tensor stores a value, a slope, and the inputs it came from. `backward` walks that record from the output ([backpropagation](/lessons/029-backpropagation)).
- A slope that arrives twice is added, so every backward rule uses `+=`.
- One SGD step subtracts the learning rate times the slope ([SGD, Adam, and learning-rate schedules](/lessons/031-sgd-adam-and-schedules)).
- A linear layer multiplies by a weight matrix and adds a bias, and a ReLU bends a hidden layer ([multilayer perceptrons](/lessons/028-multilayer-perceptrons)).
- Softmax turns a row of scores into positive weights that sum to 1 ([softmax and whole-network gradients](/lessons/030-softmax-and-whole-network-gradients)).

## The same walk, under four names

Picture the little graph you already walked. $a = 3$ and $b = 2$ multiply to $6$, then $a$ is added again, and the result is $c = 9$. Two slopes come back to $a$, because $a$ feeds both the product and the sum, and they add. One slope comes back to $b$.

Your `Tensor` keeps a slope, and that slope starts as zeros of the same shape as the value. A torch tensor keeps a slope only when you ask, and until the first walk `.grad` is `None` rather than zero. The first `backward` writes it. Every later `backward` adds.

The four pieces line up as follows:

| Your tensor | PyTorch | What it does |
|---|---|---|
| A value whose slope starts at 0 | `requires_grad=True`, and `.grad` starts as `None` | Marks an input you want a slope for |
| `grad +=` inside each rule | `backward` adds into `.grad` | Walks the tape from a scalar loss |
| A step you left unrecorded, because you stayed in numpy | `torch.no_grad()` | Keeps the weight update off the next tape |
| `backward` sets the output slope to 1 | `loss.backward()` | Starts the walk from the loss |

The same two numbers, in PyTorch, are one product and one sum:

```python
a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
c = a * b + a
c.backward()
```

`c` holds 9. `a.grad` holds 3, which is $b + 1$, the product's slope plus the sum's slope. `b.grad` holds 3, which is $a$. That is the graph from the backpropagation chapter, with the walk done for you.

Run the same lines again without clearing the slopes. `backward` adds a second 3 onto each slope, and both `.grad` attributes hold 6. Your engine did the same thing, because the rule was `+=`.

> [!KEY]
> `backward` adds into `.grad`. The first call writes the slope. The next call keeps the old one and adds another.

## Keep the update off the tape

A training step changes the weight. If you record that change, the next loss depends on the previous one, and the tape ties the steps together. Your engine left the step unrecorded, because you subtracted numpy arrays. PyTorch records every operation on a tensor that requires a slope, so the subtraction must happen in a block that refuses to record.

Take the slopes you have, 3 and 3, and a learning rate of $0.25$. The update subtracts $0.25 \times 3 = 0.75$ from each value. $a$ moves from 3 to 2.25, and $b$ moves from 2 to 1.25.

```python
with torch.no_grad():
    a -= 0.25 * a.grad
    b -= 0.25 * b.grad
```

`torch.no_grad()` is the switch. Inside the block, the subtraction changes the stored number and adds no node. After the block, recording is on again. The slopes themselves are still 3 and 3, because nothing has cleared them.

Drop the block and write `a -= 0.25 * a.grad` on its own. PyTorch raises `a leaf Variable that requires grad is being used in an in-place operation.` A leaf is a tensor you created, rather than a tensor that was computed from other tensors. An in-place operation writes into the same storage. The tape may still need the old value, so the write is refused. The `no_grad` block is what makes the write legal, and what keeps it off the next tape.

## A forgotten zero doubles the next step

Clearing is a separate job from the update. Consider a parameter $p = 1$ whose loss is $p$ itself, so the true slope is 1 on every step, and a learning rate of $0.1$.

The first `backward` writes 1. The step subtracts $0.1 \times 1$, and $p$ moves from 1 to 0.9. If you then set the slope to 0, the second `backward` writes 1 again, the second update is 0.1, and $p$ ends at 0.8.

Skip the zero. The slope is still 1 when the second `backward` runs, so it becomes $1 + 1 = 2$. The second update is $0.1 \times 2 = 0.2$, and $p$ ends at 0.7. The step doubled because the old slope was still in the box.

> [!WARNING]
> Zero `.grad` after you have used it. A step that forgets leaves the slope in place, and the next update is too large by exactly the leftover.

An optimizer's `zero_grad` is this clearing, done for every parameter. The exercise writes the clearing by hand, so you can see it.

## Fit four points, and read the curve

The same step on a line is enough to see a whole loop. The inputs are $0, 1, 2, 3$. The targets are $1, 3, 5, 7$, which is the line $y = 2x + 1$. The model is one weight and one bias, both starting at 0, so every prediction starts at 0.

The misses are $1, 3, 5, 7$. The loss is the mean of their squares:

$$
L = \frac{1}{n}\sum_i (x_i w + b - y_i)^2 .
$$

Read it aloud: each miss is the prediction minus the target, the square forgets the sign, and the mean shares the blame across the four points. $(1 + 9 + 25 + 49) / 4 = 21$.

The slopes follow by the chain rule. The derivative of a square brings down a 2, and the inner derivative in $w$ is $x_i$:

$$
\frac{\partial L}{\partial w} = \frac{2}{n}\sum_i (x_i w + b - y_i)\, x_i, \qquad
\frac{\partial L}{\partial b} = \frac{2}{n}\sum_i (x_i w + b - y_i) .
$$

At $w = 0$ and $b = 0$, the miss is $-y_i$. The weight slope is $\tfrac{1}{2}\big(0 + (-3)(1) + (-5)(2) + (-7)(3)\big) = -17$. The bias slope drops the $x_i$ and is $\tfrac{1}{2}(-1-3-5-7) = -8$.

One SGD step with learning rate $0.05$ subtracts the rate times the slope:

$$
w \leftarrow w - \alpha \frac{\partial L}{\partial w}, \qquad
b \leftarrow b - \alpha \frac{\partial L}{\partial b} .
$$

The weight becomes $0 - 0.05 \times (-17) = 0.85$. The bias becomes $0.4$. Measured after that step, the loss is 7.059. A second step reaches 2.373. Figure 1 draws the 20 steps.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Training loss on four points, falling from 21 toward zero over 20 steps, on a log scale">
  <line x1="72" y1="53.7" x2="576" y2="53.7" class="stroke-grid"/>
  <line x1="72" y1="90.8" x2="576" y2="90.8" class="stroke-grid"/>
  <line x1="72" y1="127.9" x2="576" y2="127.9" class="stroke-grid"/>
  <line x1="72" y1="165.0" x2="576" y2="165.0" class="stroke-grid"/>
  <line x1="72" y1="202.1" x2="576" y2="202.1" class="stroke-grid"/>
  <line x1="72" y1="36" x2="72" y2="228" class="stroke-muted"/>
  <line x1="72" y1="228" x2="576" y2="228" class="stroke-muted"/>
  <polyline fill="none" class="stroke-b" points="72.0,41.7 97.2,59.3 122.4,76.9 147.6,94.4 172.8,112.0 198.0,129.4 223.2,146.8 248.4,163.7 273.6,179.7 298.8,193.4 324.0,203.4 349.2,209.5 374.4,212.7 399.6,214.6 424.8,215.9 450.0,217.0 475.2,218.0 500.4,219.0 525.6,220.0 550.8,220.9 576.0,221.9"/>
  <circle cx="72.0" cy="41.7" r="3.5" class="fill-b"/>
  <circle cx="576.0" cy="221.9" r="3.5" class="fill-b"/>
  <text x="66" y="57" text-anchor="end" class="fig-small">10</text>
  <text x="66" y="94" text-anchor="end" class="fig-small">1</text>
  <text x="66" y="131" text-anchor="end" class="fig-small">0.1</text>
  <text x="66" y="169" text-anchor="end" class="fig-small">0.01</text>
  <text x="66" y="206" text-anchor="end" class="fig-small">0.001</text>
  <text x="84" y="34" class="fig-small">21</text>
  <text x="72" y="248" text-anchor="middle" class="fig-small">0</text>
  <text x="198" y="248" text-anchor="middle" class="fig-small">5</text>
  <text x="324" y="248" text-anchor="middle" class="fig-small">10</text>
  <text x="450" y="248" text-anchor="middle" class="fig-small">15</text>
  <text x="576" y="248" text-anchor="middle" class="fig-small">20</text>
  <text x="324" y="270" text-anchor="middle" class="fig-small">step</text>
  <text x="16" y="20" class="fig-small">loss, log scale</text>
</svg>
Figure 1. Mean squared error on the four points, for 20 full-batch steps at learning rate 0.05. The loss starts at 21 and is 0.000292 after the last step. The weight ends at 2.013 and the bias at 0.972, against a line of weight 2 and bias 1.
```

The loop that draws the figure is the loop you are about to write, with one batch that holds every row. Each step runs the loss, calls `backward`, subtracts, and zeros the slope.

> [!KEY]
> The training step is four lines: the loss, `backward`, a subtraction inside `no_grad`, and a zero slope. Leave any one out and the next step is wrong.

## Register every parameter

A module remembers which tensors are weights. Subclass `nn.Module`, call the parent constructor, and store each weight as an `nn.Parameter`. A plain tensor stored on the object is invisible to `parameters()`, so a later step skips it.

`torch.nn.Linear` stores `weight` with shape `(out_features, in_features)` and `bias` with shape `(out_features,)`. The forward pass is the product you already know, with a transpose so the weight's first axis is the output:

```python
y = x @ weight.T + bias
```

Check it on one row. Let $x = [1, 0, -1]$, let the weight be $[[1, 0, 0], [0, 2, 1]]$, and let the bias be $[0.5, -1]$. Then $x W^{\top} = [1, -1]$, and the bias moves that to $[1.5, -2]$. Swap the weight's axes and this row still has a shape that multiplies, but a batch whose width is the input size fails, or, worse, a square weight runs and returns the wrong numbers.

A multilayer perceptron is a list of those layers. Hide a `ReLU` between them, and leave the last layer linear, so the output can be any real number. Store the layers in an `nn.ModuleList`. A Python list holds the same objects, and `parameters()` does not walk it, so the box looks empty.

The step then reads `model.parameters()` and updates each one. That is the whole point of registration: the loop does not need the names.

## Shuffle, train, and evaluate

A full batch of four rows is a special case. A real epoch walks the rows in groups. Draw the order once per epoch from a `torch.Generator` that you seed yourself:

```python
generator = torch.Generator()
generator.manual_seed(seed)
order = torch.randperm(n, generator=generator)
```

Slice `order` into blocks of `batch_size`. The last block can be shorter. For each block, compute the mean squared error, call `backward`, and hand `model.parameters()` to the step that subtracts and zeros. Add `loss.detach() * rows_in_the_block` into a running total and divide by $n$ at the end. That total is the mean over rows. The mean of the batch means, with the short block weighted the same as a full one, is a different number.

Call `model.train()` at the start of the epoch. Call `model.eval()` when you measure. The two modes exist because some layers change their behaviour. Dropout is the one you have already built by hand. With `torch.manual_seed(0)`, a dropout layer of probability $0.5$ applied to $[1, 1, 1, 1]$ returns $[0, 0, 2, 0]$ in train mode. The kept entry is scaled by $1 / (1 - 0.5) = 2$, which is the inverted dropout from [dropout and label smoothing](/lessons/035-dropout-and-label-smoothing). The same layer in eval mode returns $[1, 1, 1, 1]$. Leave the module in train mode and the measurement randomly deletes coordinates.

The eval pass also belongs inside `torch.no_grad()`. You want a number, and you do not want a tape or a slope on the inputs. On the four points, with the weight and the bias still 0, that number is 21.

> [!TIP]
> Seed a local `Generator`. Calling `torch.manual_seed` inside the epoch resets the global stream and makes the next random draw in the program depend on how many epochs you ran.

## Write the blend with einsum

[Attention as a soft lookup](/lessons/045-attention-as-soft-lookup) scores each source position with a dot product, turns the scores into weights, and blends the values. This chapter does not derive that blend again. The loop, for one query $q$, one key $k_s$, and one value row $v_s$, is:

$$
z_s = q \cdot k_s, \qquad
a_s = \frac{e^{z_s}}{\sum_{s'} e^{z_{s'}}}, \qquad
c = \sum_s a_s v_s .
$$

Read it aloud: the score is a dot product, the weight is a softmax over source positions, and the context is the weighted sum of the values. Subtract the largest score before the exponential, as that chapter does, so a score of 800 does not overflow.

The same query and keys as that chapter make the arithmetic visible. The query is $[1, 0]$, the keys are $[1, 0]$ and $[0, 1]$, and the values are $[10, 0]$ and $[0, 10]$. The scores are $1$ and $0$. The weights are $0.7311$ and $0.2689$. The context is $[7.311, 2.689]$.

A batch repeats the loop for every query. Give the query shape `(B, T, D)`, the keys `(B, S, D)`, and the values `(B, S, D_v)`. The context has shape `(B, T, D_v)`. `einsum` names the axes and says which ones to multiply and which to sum. A matrix product is the small version, and it is worth seeing before the blend:

```python
# a is (2, 2), b is (2, 2), and the result is a @ b
torch.einsum("ij,jk->ik", a, b)
```

The letters on the left of the arrow are the inputs. A letter that disappears on the right is summed. For the scores, `t` is the query, `s` is the source, and `d` is the shared width:

```python
scores = torch.einsum("btd,bsd->bts", q, k)
```

`b` is the batch, and it rides along because it appears on the right. Softmax over `s` (the last axis) turns each query's scores into weights. A second einsum sums the source axis against the values. Softmax over `t` instead is the usual mistake: each source's weights then sum to 1, and the context is a different blend.

> [!KEY]
> The einsum is the attention chapter's loop, with the axes named. It is the same scores, the same softmax, and the same weighted sum.

## Where this breaks

Four bugs show up as a bad loss, a crash, or a measurement that changes between runs.

**The slope is left in the box.** On the four-point line the loss can still fall, so a falling curve does not prove the zero happened. After the step, `.grad` is 0 or empty. In the scalar example the second update is 0.2 and $p$ ends at 0.7.

**The module is left in train mode.** A linear layer ignores the flag, so a test on one can pass while a later dropout layer looks noisy. `eval_mse` has to call `eval` and leave the module there. The dropout row $[0, 0, 2, 0]$ is not $[1, 1, 1, 1]$.

**An in-place write hits a leaf that still requires a slope.** `a -= 0.25 * a.grad`, outside `no_grad`, raises `a leaf Variable that requires grad is being used in an in-place operation.` The same write inside the block is the update you want.

**The dtype or the device does not match.** A `torch.nn.Linear` holds float32 weights. A float64 batch raises `mat1 and mat2 must have the same dtype, but got Double and Float`. A tensor on one device refuses a tensor on another. Adding a CPU tensor to a tensor on the `meta` device, which stores a shape and no values, raises `Tensor on device cpu is not on the expected device meta!` Numpy defaults to float64, so a batch built with numpy and cast carelessly is the usual way to hit the dtype line.

> [!WARNING]
> Cast the batch to the weight's dtype, and put it on the weight's device, before the first product. The crash names both sides.

## Cost and scaling

The tape stores activations until `backward` consumes them. A forward pass inside `no_grad` stores none of them, so the eval pass is cheaper and it leaves no stray slope. One step still costs the layer's two products, plus the tape's bookkeeping.

The einsum and the Python loop build the same score table, with one row per query and one column per source. The loop pays an interpreter step per entry. The einsum launches one product. At sequence length $T$ that table is the $T \times T$ cost the attention chapter already counts.

## Build it step by step

> [!RECAP]
> - `requires_grad` marks a leaf, `.grad` starts empty, and `backward` adds into it.
> - `no_grad` keeps the update off the tape, and the step zeros `.grad` or the next update doubles.
> - `parameters()` sees a tensor only after you register it as an `nn.Parameter`, inside a module or a `ModuleList`.
> - Train mode and eval mode disagree as soon as a layer such as dropout is present.
> - The einsum blend is the attention loop: scores, softmax over the source, weighted sum of the values.

1. **`Linear(in_features, out_features)`** is an `nn.Module`. Register `weight` with shape `(out_features, in_features)` and `bias` with shape `(out_features,)`, both as `nn.Parameter`. `forward(x)` takes `x` of shape `(N, in_features)` and returns `(N, out_features)`, the same numbers as `torch.nn.Linear` with the same weight and bias. The usual mistake is to store `weight` as `(in_features, out_features)`. A square layer hides it.

2. **`MLP(sizes)`** is an `nn.Module`. `sizes` is a list of widths, such as `[3, 4, 2]`. Build one `Linear` between each pair, store them in an `nn.ModuleList`, apply `ReLU` after every layer except the last, and return the last layer's output. `parameters()` must see every weight and every bias. The usual mistake is a Python list, which leaves `parameters()` empty, or a `ReLU` on the output.

3. **`sgd_step(params, lr)`** updates each parameter in place. Inside `torch.no_grad()`, subtract `lr * p.grad` and then zero that gradient. Skip a parameter whose gradient is `None`. Return nothing. The usual mistake is the subtraction without the zero, so the second step on a slope of 1 with learning rate 0.1 moves by 0.2.

4. **`train_epoch(model, x, y, lr, batch_size, seed)`** returns the mean squared error over the rows, as a Python float. `x` has shape `(n, in_features)` and `y` has shape `(n, out_features)`. Call `model.train()`, shuffle with a local `Generator` seeded by `seed`, and include the last short block. The usual mistake is an unweighted mean of the batch losses, which treats a short block as a full one.

5. **`eval_mse(model, x, y)`** returns that same mean on the whole set, as a Python float. Call `model.eval()`, compute the number inside `torch.no_grad()`, and leave the module in eval mode. The usual mistake is to leave `model.training` true.

6. **`einsum_attention(q, k, v)`** takes `q` of shape `(B, T, D)`, `k` of shape `(B, S, D)`, and `v` of shape `(B, S, D_v)`. It returns the context, shape `(B, T, D_v)`. Score with an einsum, take a stable softmax over the source axis, and blend with a second einsum. The usual mistake is softmax over the query axis, or `exp` of a raw score of 800, which overflows to infinity.

## The exercise

You implement `Linear`, `MLP`, `sgd_step`, `train_epoch`, `eval_mse`, and `einsum_attention`. The checks compare `Linear` and `MLP` with `torch.nn.Linear` on outputs and on gradients, to $10^{-6}$. They require one SGD step to move the parameter by the learning rate times the slope and to leave a zero slope, and they reject a second step of 0.2 when 0.1 is the fresh slope. They run a seeded epoch, require the loss to fall, require the weights to match a step that zeros, and require `eval_mse` to return 21 on the four points and to leave the module in eval mode. The last checks match `einsum_attention` to the loop, including the worked context and a score of 800.

## Further reading

These notes and lectures are the ones this chapter follows:

- For `requires_grad`, for `backward` adding into `.grad` (the tutorial's square goes from a slope of 12 to a slope of 24), for `zero_grad`, for `nn.Module` and `nn.Parameter`, and for `DataLoader`, see [CS224N: PyTorch Tutorial (Winter 2021)](https://web.stanford.edu/class/cs224n/materials/CS224N_PyTorch_Tutorial.html) (Dilara Soylu). The [CS224N schedule](https://web.stanford.edu/class/cs224n/index.html) lists that material as a PyTorch tutorial session, not as a numbered lecture.
- For einsum as a way to name the axes of a product, see lecture 2 of [CS336](https://cs336.stanford.edu/), "PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity)". The [lecture 2 trace](https://cs336.stanford.edu/lectures/?trace=lecture_02) writes a batched product as `einsum` with named axes.
- For a PyTorch tutorial session that is not given a lecture number, see the [6.S898 Fall 2023 schedule](https://phillipi.github.io/6.s898/) (Phillip Isola, Sara Beery, and Jeremy Bernstein), which lists "PyTorch Tutorial" on 12 and 13 September. The [6.7960 OpenCourseWare page](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/resources/mit6_7960f24_review_mp4/) titles the same kind of session "PyTorch Tutorial".
- For tensors, `nn.Module`, and a training step in a lab, see Software Lab 1 on the [6.S191 schedule](https://introtodeeplearning.com/) ("Deep Learning in Python; Music Generation") and the notebook [Lab 1: Intro to PyTorch and Music Generation with RNNs](https://github.com/MITDeepLearning/introtodeeplearning/blob/master/lab1/PT_Part1_Intro.ipynb).
