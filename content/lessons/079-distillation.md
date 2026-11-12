---
title: "Distillation: teaching a small model with a big one's probabilities"
slug: 079-distillation
part: "Part 7 — Large language models"
summary: A student learns more from a teacher's softened probabilities than from hard labels, and the T² factor keeps the loss fair across temperatures.
minutes: 70
gpu: false
objectives:
  - Explain why a teacher's soft probabilities carry more information per example than a hard label.
  - Soften a softmax with a temperature and compute it without overflow.
  - Write the distillation loss as T² times KL(teacher || student), mixed with cross-entropy, and derive its gradient.
  - Say why the T² factor keeps gradient sizes level across temperatures, and what the loss becomes as T grows.
  - Match a teacher's intermediate features through a learned projection, and say how language models distill by sequence.
prereqs:
  - 037-softmax-and-whole-network-gradients
  - 075-decoding-and-sampling
  - 078-fine-tuning-and-lora
exercise: 079-distillation
---

# Distillation: teaching a small model with a big one's probabilities

You have a large model that reads handwritten digits well, and you need a small one that runs on a phone. The obvious plan is to train the small model on the same labeled images. It learns, but it ends up worse than the large model, and when you only have a few hundred images it ends up much worse.

Part of the reason is that a label throws information away. The label for a picture of a 7 says "seven" and nothing else. It does not say that this 7 has a flag on its stem that makes it look a little like a 1, or that it looks nothing like a 9. The large model knows these things. They are in the probabilities it puts on the wrong answers.

Distillation trains the small model, the student, to copy the large model's full output, not only its top answer. This chapter shows why that output is worth copying, builds the loss that copies it, and derives the one factor, T², that people forget and then wonder why their student trains badly.

## What you already know

- A softmax turns a row of scores, called logits, into probabilities that sum to 1, as in the chapter on [softmax and whole-network gradients](/lessons/037-softmax-and-whole-network-gradients).
- Cross-entropy against a one-hot label is $-\log$ of the probability on the right class, and its gradient with respect to the logits is the probabilities minus the one-hot label.
- KL divergence measures how far one distribution is from another, and it is never negative.
- Temperature divides the logits before the softmax, as in the chapter on [decoding and sampling](/lessons/075-decoding-and-sampling).
- Fine-tuning starts from a model that already exists, as in the chapter on [LoRA](/lessons/078-fine-tuning-and-lora).

## The picture: what a hard label hides

Take one picture of a 7, a teacher with three output classes (7, 1, and 9), and a student. The teacher's logits are $[6, 3, 0]$. At temperature 1 that is the probabilities $[0.9503, 0.0473, 0.0024]$. Figure 1 draws them on the left, next to a student with logits $[4, 3, 1]$.

```figure
<svg viewBox="0 0 620 280" role="img" aria-label="Teacher and student probabilities for classes 7, 1, and 9, at temperature 1 and temperature 3">
<text x="40" y="24" class="fig-bold">T = 1</text><text x="340" y="24" class="fig-bold">T = 3</text>
<line x1="40" y1="210" x2="280" y2="210" class="stroke-muted"/><line x1="340" y1="210" x2="580" y2="210" class="stroke-muted"/>
<rect x="50" y="96.0" width="28" height="114.0" class="fill-a"/>
<text x="48" y="91.0" class="fig-small">0.950</text>
<rect x="82" y="125.4" width="28" height="84.6" class="fill-b"/>
<text x="80" y="120.4" class="fig-small">0.705</text>
<text x="72" y="232" class="fig-bold">7</text>
<rect x="130" y="204.3" width="28" height="5.7" class="fill-a"/>
<text x="128" y="199.3" class="fig-small">0.048</text>
<rect x="162" y="178.9" width="28" height="31.1" class="fill-b"/>
<text x="160" y="173.9" class="fig-small">0.259</text>
<text x="152" y="232" class="fig-bold">1</text>
<rect x="210" y="209.7" width="28" height="0.3" class="fill-a"/>
<text x="208" y="204.7" class="fig-small">0.003</text>
<rect x="242" y="205.8" width="28" height="4.2" class="fill-b"/>
<text x="240" y="200.8" class="fig-small">0.035</text>
<text x="232" y="232" class="fig-bold">9</text>
<rect x="350" y="130.2" width="28" height="79.8" class="fill-a"/>
<text x="348" y="125.2" class="fig-small">0.665</text>
<rect x="382" y="152.4" width="28" height="57.6" class="fill-b"/>
<text x="380" y="147.4" class="fig-small">0.480</text>
<text x="372" y="232" class="fig-bold">7</text>
<rect x="430" y="180.6" width="28" height="29.4" class="fill-a"/>
<text x="428" y="175.6" class="fig-small">0.245</text>
<rect x="462" y="168.7" width="28" height="41.3" class="fill-b"/>
<text x="460" y="163.7" class="fig-small">0.344</text>
<text x="452" y="232" class="fig-bold">1</text>
<rect x="510" y="199.2" width="28" height="10.8" class="fill-a"/>
<text x="508" y="194.2" class="fig-small">0.090</text>
<rect x="542" y="188.8" width="28" height="21.2" class="fill-b"/>
<text x="540" y="183.8" class="fig-small">0.177</text>
<text x="532" y="232" class="fig-bold">9</text>
<rect x="200" y="252" width="14" height="14" class="fill-a"/><text x="220" y="264">teacher</text>
<rect x="300" y="252" width="14" height="14" class="fill-b"/><text x="320" y="264">student</text>
</svg>
Figure 1. Left: at T = 1 the teacher's wrong-class probabilities are almost invisible. Right: at T = 3 they are large enough to matter, and the gap between the student and the teacher on all three classes is clear.
```

Read the teacher's left panel. It puts 95 percent on "7", about 4.7 percent on "1", and 0.24 percent on "9". The ratio between the two wrong answers is about 20 to 1. That ratio is the teacher saying that this picture is much more like a 1 than a 9. The field calls this information dark knowledge: it is in the output, but at temperature 1 it is nearly invisible to a loss.

Here is why it is invisible. Cross-entropy against the teacher's distribution multiplies each class's log-probability by the teacher's probability. The "1" and "9" terms get weights 0.047 and 0.002, so they barely move the loss, and the student can score well while ignoring them. A hard label is the extreme case, with weights 0 and 0.

The fix is to flatten the teacher's distribution before the student sees it. Figure 1's right panel does that with temperature 3. The wrong classes hold 24 percent and 9 percent, so the loss cares about them.

## The words: temperature, then a loss

Temperature $T$ is a positive number that divides the logits before the softmax:

$$
p_i^{(T)} = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

Read it aloud: score each class, divide the score by $T$, exponentiate, and normalize. At $T = 1$ it is the ordinary softmax. As $T$ grows, every $z_i / T$ moves toward 0, so every probability moves toward $1/K$ for $K$ classes. As $T$ shrinks toward 0, the largest logit takes everything. The teacher's logits $[6, 3, 0]$ divided by 3 are $[2, 1, 0]$, and the softmax of that is $[0.6652, 0.2447, 0.0900]$.

The student and the teacher both use the same $T$. The loss then compares the two softened distributions. The measure to use is the KL divergence from the teacher's distribution to the student's:

$$
\mathrm{KL}(p \,\|\, q) = \sum_i p_i \log \frac{p_i}{q_i}
$$

Read it aloud: for each class, take the log of the ratio of the teacher's probability $p_i$ to the student's $q_i$, weight it by the teacher's probability, and add. It is 0 when the two distributions are equal and positive otherwise. It is not symmetric. $\mathrm{KL}(p \,\|\, q)$ weights the mismatch by the teacher's probabilities, so it asks the student to cover everything the teacher considers likely. Swap the arguments and the student's probabilities do the weighting instead, which changes both the value and the gradient. The teacher goes first.

> [!KEY]
> Distillation asks the student to match the teacher's softened distribution, $\mathrm{KL}(p_t^{(T)} \,\|\, p_s^{(T)})$. The temperature makes the wrong-class probabilities large enough for the loss to see them.

### A worked example

Use the teacher $[6, 3, 0]$ and the student $[4, 3, 1]$ from Figure 1, with true label "7", which is class 0.

At $T = 3$ the teacher is $[0.6652, 0.2447, 0.0900]$. The student's logits divided by 3 are $[1.333, 1, 0.333]$, which gives $[0.4798, 0.3438, 0.1765]$. The KL divergence is

$$
0.6652 \log \tfrac{0.6652}{0.4798} + 0.2447 \log \tfrac{0.2447}{0.3438} + 0.0900 \log \tfrac{0.0900}{0.1765} = 0.2175 - 0.0832 - 0.0606 = 0.0737.
$$

That number is small, and that is a problem, which the next section explains. The hard-label cross-entropy for class 0 at $T = 1$ is $-\log 0.7054 = 0.3490$, where $0.7054$ is the student's ordinary softmax probability on class 0.

## Why the loss needs a T²

Here is the full loss, which mixes the soft term with the ordinary cross-entropy on the hard label $y$:

$$
L = \alpha \, T^2 \, \mathrm{KL}\big(p_t^{(T)} \,\|\, p_s^{(T)}\big) + (1 - \alpha)\, \mathrm{CE}\big(p_s^{(1)}, y\big)
$$

Read it aloud: take the softened KL at temperature $T$, multiply it by $T^2$, weight it by $\alpha$, and add the weight $1 - \alpha$ times the cross-entropy of the student's ordinary softmax against the label. The weight $\alpha$ is between 0 and 1. A larger $\alpha$ trusts the teacher more. The hard term uses temperature 1 because the label is a hard answer. In the example, $T^2 \mathrm{KL} = 9 \times 0.0737 = 0.6633$, so with $\alpha = 0.5$ the loss is $0.5 \times 0.6633 + 0.5 \times 0.3490 = 0.5062$.

Why $T^2$? Why not leave the soft term as it is? Look at its gradient. Write $q = p_s^{(T)}$ and $p = p_t^{(T)}$. The student's logit $z_j$ reaches $\mathrm{KL}$ only through $\log q_i = z_i / T - \log \sum_k e^{z_k / T}$. Differentiating $-\sum_i p_i \log q_i$ with respect to $z_j$ gives $-p_j / T$ from the first part and $q_j / T$ from the log-sum-exp, because the $p_i$ add to 1:

$$
\frac{\partial\, \mathrm{KL}}{\partial z_j} = \frac{q_j - p_j}{T}
$$

The gradient of the KL divergence is the difference of the two softened distributions, divided by $T$. That is the same shape as the cross-entropy gradient, probabilities minus targets, with one extra $1/T$. The $1/T$ is the first reason to worry, and the second is that $q_j - p_j$ also shrinks as $T$ grows, because both distributions flatten toward uniform. Each one costs a factor of about $1/T$, so the soft gradient shrinks like $1/T^2$.

Figure 2 shows this for the worked example. Without a correction, the gradient norm falls from 0.326 at $T = 1$ to 0.0029 at $T = 16$. A student trained at high temperature would barely feel its soft term next to the hard term, and you would have to retune the learning rate or $\alpha$ every time you changed $T$.

```figure
<svg viewBox="0 0 620 250" role="img" aria-label="Gradient norm against temperature, with and without the T squared factor">
<line x1="60" y1="190" x2="540" y2="190" class="stroke-muted"/><line x1="60" y1="30" x2="60" y2="190" class="stroke-muted"/>
<line x1="60" y1="46" x2="540" y2="46" class="stroke-muted dashed"/><text x="400" y="38" class="fig-small">limit 0.720</text>
<text x="54" y="208" class="fig-small">1</text>
<text x="174" y="208" class="fig-small">2</text>
<text x="294" y="208" class="fig-small">4</text>
<text x="414" y="208" class="fig-small">8</text>
<text x="534" y="208" class="fig-small">16</text>
<text x="22" y="194" class="fig-small">0</text>
<text x="22" y="114" class="fig-small">0.4</text>
<text x="22" y="34" class="fig-small">0.8</text>
<polyline points="60,124.9 180,71.1 300,47.2 420,42.8 540,43.4" fill="none" class="stroke-b"/>
<polyline points="60,124.9 180,160.3 300,181.1 420,187.7 540,189.4" fill="none" class="stroke-a"/>
<circle cx="60" cy="124.9" r="4" class="fill-b"/>
<circle cx="180" cy="71.1" r="4" class="fill-b"/>
<circle cx="300" cy="47.2" r="4" class="fill-b"/>
<circle cx="420" cy="42.8" r="4" class="fill-b"/>
<circle cx="540" cy="43.4" r="4" class="fill-b"/>
<circle cx="60" cy="124.9" r="4" class="fill-a"/>
<circle cx="180" cy="160.3" r="4" class="fill-a"/>
<circle cx="300" cy="181.1" r="4" class="fill-a"/>
<circle cx="420" cy="187.7" r="4" class="fill-a"/>
<circle cx="540" cy="189.4" r="4" class="fill-a"/>
<text x="200" y="88" class="fig-bold">with T²</text><text x="200" y="168" class="fig-bold">without T²</text>
<text x="240" y="236">temperature T (log scale)</text>
<text x="22" y="20" class="fig-small">norm</text>
</svg>
Figure 2. The size of the soft gradient for the worked example, with and without the T² factor. Without it the gradient dies as T grows. With it the gradient levels off near 0.72, the value of the logit-matching limit.
```

Multiplying the loss by $T^2$ multiplies its gradient by $T^2$ too, which cancels the shrinking:

$$
\frac{\partial}{\partial z_j}\Big[T^2\,\mathrm{KL}\Big] = T\,\big(p_s^{(T)} - p_t^{(T)}\big)_j
$$

With this factor the norm is 0.595 at $T = 2$, 0.714 at $T = 4$, 0.736 at $T = 8$, and 0.733 at $T = 16$. The gradient stays level, so $\alpha$ and the learning rate keep their meaning as you change the temperature. The full gradient of the loss with respect to the student's logits is

$$
\frac{\partial L}{\partial z_j} = \alpha\, T \big(p_s^{(T)} - p_t^{(T)}\big)_j + (1 - \alpha)\big(p_s^{(1)} - \text{onehot}(y)\big)_j
$$

Read it aloud: for each logit, the soft part pushes the student's softened probability toward the teacher's, and the hard part pushes the student's ordinary probability toward the label. Each part is a difference of two probability vectors, so each sums to zero over the classes, which means a step never changes the overall level of the logits, only how they compare. In numpy the whole thing is two softmaxes, one subtraction each, and a weighted sum.

For one example at $T = 3$ and $\alpha = 1$, the three entries are $[-0.5565, 0.2971, 0.2594]$. The student's top logit should rise, because the teacher gives class 0 more probability than the student does, and the other two should fall.

> [!WARNING]
> Leaving out the $T^2$ factor does not raise an error. The loss still goes down, so you notice only that the student is worse at high temperatures. A loss at $T = 3$ should be 9 times the raw KL, and a gradient check against finite differences cannot find the bug, because the loss and gradient are wrong in the same way. Check the value against a hand-computed number as well.

## What high temperature does

At a high temperature the loss simplifies. For a large $T$, $e^{z_i / T}$ is close to $1 + z_i / T$, so the softened probability is close to

$$
p_i^{(T)} \approx \frac{1}{K}\left(1 + \frac{z_i - \bar z}{T}\right)
$$

where $\bar z$ is the mean of the row's logits and $K$ is the number of classes. The mean appears because the normalizing sum adds $\sum_j z_j / T = K \bar z / T$ to the denominator. Subtracting the mean is called centring, and we write $c_i = z_i - \bar z$.

The two distributions then differ by $(c^{s}_i - c^{t}_i) / (K T)$ in each class, so the gradient with the $T^2$ factor tends to

$$
T\,\big(p_s^{(T)} - p_t^{(T)}\big)_i \;\longrightarrow\; \frac{c^{s}_i - c^{t}_i}{K}
$$

Read it aloud: the student's push is the gap between the centred logits, divided by the number of classes. That is the gradient of $\tfrac{1}{2K}\sum_i (c^{s}_i - c^{t}_i)^2$, a squared error between the logit rows. So in the limit, distillation is logit matching, the student regresses onto the teacher's raw scores, with each row shifted to a mean of 0 because the softmax cannot see a constant added to every logit.

For the worked example the limit is $[-0.5556, 0.1111, 0.4444]$. At $T = 16$ the actual gradient is $[-0.5791, 0.1579, 0.4212]$, close but not equal, and at $T = 3$ it is $[-0.5565, 0.2971, 0.2594]$. That matches Figure 2: the curve sits at the limit's norm, 0.7201, only for large $T$.

Why does this matter in practice? It tells you what temperature is for. A low $T$ copies mainly the teacher's top choice. A very high $T$ copies every logit, including the big negative ones that say nothing useful, so the student spends effort on noise. A moderate temperature passes on the wrong-class ranking without that noise. You tune $T$ on held-out data.

> [!KEY]
> With the $T^2$ factor, the soft gradient stays level as $T$ grows, and in the limit it equals the centred logit gap divided by $K$. Distillation at a low temperature is about the teacher's top choice, and at a very high temperature it is regression on logits.

## Does it help? A small experiment

Take a linear student with 10 features and 3 classes, and give it 30 labeled examples. The labels are random draws from a true probability model, as real labels are noisy, so a label says "class 2" even when the true model gave class 2 only 40 percent of the time. The teacher is a model trained on much more data, which we imitate with the true logits plus a little noise.

Train the student twice, with the same 400 gradient steps, once with $\alpha = 0$ (hard labels only) and once with $\alpha = 1$ at $T = 3$. Then measure how often each student picks the true model's best class on 2000 fresh points. Over five seeds, the hard-label student agrees 73.3 percent of the time and the soft-target student 95.9 percent. The soft student wins on every seed.

The reason is the information per example. A hard label gives one choice among 3, and a noisy one, so the student has to guess the shape of the true model from a single draw. A soft target gives the whole distribution, with the same sample's noise already averaged out by the teacher. Thirty soft targets act like many more hard labels. Hinton and his coauthors argued the same thing about soft targets in the original paper. The script `scripts/figures/079-distillation.py` prints these numbers.

## Matching features, not only outputs

The output is not the only thing a teacher knows. Its hidden layers hold features, the vectors a layer produces before the final scores. FitNets (Romero and coauthors, 2015) ask the student to match a teacher's intermediate layer, which gives the student a signal from the middle of the network, not only the end.

There is one obstacle. The student is usually narrower than the teacher, so its feature vector $h_s$ has $d_s$ entries and the teacher's $h_t$ has $d_t$, and you cannot subtract vectors of different sizes. The fix is a learned linear projection $P$ with shape $(d_s, d_t)$ that maps the student's features to the teacher's width. The loss is a mean squared error:

$$
L_{\text{feat}} = \frac{1}{2n}\,\big\lVert h_s P - h_t \big\rVert_F^2
$$

Read it aloud: project the $n$ student feature rows, subtract the teacher's rows, square every entry, add them up, and divide by $2n$. Let $R = h_s P - h_t$. The gradient with respect to $P$ is

$$
\frac{\partial L_{\text{feat}}}{\partial P} = \frac{1}{n}\, h_s^{\top} R
$$

Use a small example to check it. Take $h_s = \begin{bmatrix}1 & 0\\0 & 2\end{bmatrix}$, $h_t = \begin{bmatrix}1&1&0\\0&0&3\end{bmatrix}$, and $P = \begin{bmatrix}1&0&0\\0&1&1\end{bmatrix}$. Then $h_s P = \begin{bmatrix}1&0&0\\0&2&2\end{bmatrix}$, so $R = \begin{bmatrix}0&-1&0\\0&2&-1\end{bmatrix}$, the squares add to $1 + 4 + 1 = 6$, and the loss is $6/4 = 1.5$.

The projection $P$ is trained along with the student and thrown away afterward, because it only exists to make the shapes agree. Feature loss is usually added to the output loss, not used alone.

## Sequences: distilling a language model

A language model's output is a distribution over the next token at every position. You can apply the loss above at each position, which is word-level distillation. Kim and Rush (2016) proposed a different route for translation, called sequence-level knowledge distillation: run the teacher on the training inputs, keep the sentences it produces, and train the student on those sentences with ordinary cross-entropy, as if they were the labels. The student then learns the teacher's behavior on whole sentences, and you need only the teacher's text, not its logits, which is the only option when the teacher sits behind an API.

## Related methods, and where they live

Quantization stores a model's weights in fewer bits, such as 8-bit integers, and it is a different way to shrink a model from distillation, and the two combine. The inference course owns that topic. Its chapter [Quantization on Ampere](https://qwen.ifkash.dev/c/18-quantization) derives the error bound and builds the kernels.

Pruning removes weights, usually the smallest ones, and then retrains to recover. It makes a model smaller on paper, but unstructured sparsity rarely speeds up dense hardware, so it needs a structure the hardware can use before it saves time.

On-policy distillation for language models, where the student generates its own text and the teacher scores it, belongs to the reinforcement learning course. Start from its [home page](https://rl.ifkash.dev/), because it builds on policy gradients that this course does not cover.

## Build it step by step

> [!RECAP]
> - A hard label hides how a teacher ranks the wrong answers; its soft probabilities carry that ranking.
> - Temperature $T$ divides the logits before the softmax, and a larger $T$ makes the wrong classes visible.
> - The loss is $\alpha T^2 \mathrm{KL}(p_t^{(T)} \| p_s^{(T)}) + (1 - \alpha)\,\mathrm{CE}$, with the teacher first in the KL.
> - The gradient with respect to a student logit is $\alpha T (p_s^{(T)} - p_t^{(T)}) + (1 - \alpha)(p_s - \text{onehot})$, and the $T^2$ keeps it level across temperatures.
> - As $T$ grows the soft gradient tends to the centred logit gap over $K$, which is logit matching.

1. **`soft_targets(logits, T)`** takes logits of shape `(K,)` or `(n, K)` and returns probabilities of the same shape. Divide by `T`, subtract the row maximum, exponentiate, and divide by the sum along the last axis. The usual mistake is to skip the subtraction, which overflows for logits near 1000.
2. **`kl_divergence(p, q)`** returns one divergence per row. Compute `sum(p * (log p - log q))` along the last axis, treating a zero in `p` as contributing 0. The usual mistake is to pass the arguments in the wrong order.
3. **`distill_loss(z_s, z_t, y, T, alpha)`** returns a float, the mean over rows. Build the teacher's and student's soft targets at `T`, take the KL with the teacher first and multiply by `T**2`, then add the hard cross-entropy at temperature 1. The usual mistake is to leave out `T**2`.
4. **`distill_grad(z_s, z_t, y, T, alpha)`** returns shape `(n, K)`. The soft part is `T * (p_s_T - p_t_T)`, the hard part is `p_s - onehot(y)`, and you divide by `n` for the mean. The usual mistake is to use `(p_s_T - p_t_T) / T`, which is the gradient of the KL without the factor.
5. **`logit_match_grad(z_s, z_t)`** returns the limit of the soft gradient for large `T`, with the same shape as the inputs. Subtract each row's mean from both inputs, take the difference, and divide by `K`. The usual mistake is to forget to centre.
6. **`feature_distill_loss(h_s, h_t, P)`** returns a float. Project, subtract, square, sum, and divide by `2 * n`.
7. **`feature_distill_grad(h_s, h_t, P)`** returns shape `(d_s, d_t)`. Compute `h_s.T @ R / n`. The usual mistake is to forget the factor of `1 / n` that the loss has.
8. **`fit_student(X, z_t, y, T, alpha, steps, lr)`** returns `W` of shape `(d, K)` for a linear student `z = X @ W`. Start from zeros and repeat `W -= lr * X.T @ distill_grad(X @ W, z_t, y, T, alpha)`. `distill_grad` already averages over rows, so don't divide by `n` again.

## Where this breaks

A student that is too small cannot copy the teacher, and then a high $\alpha$ gives a student that fits neither the labels nor the teacher well. Mix in the hard term, or use a larger student.

A teacher's mistakes are copied along with its knowledge. Where the teacher is confidently wrong, a hard-label student at least gets some corrective signal from the true label, and a pure soft student gets none. This is one reason to keep $\alpha$ below 1 in practice.

The temperature is a hyperparameter, and a wrong value costs accuracy. Tune it with $\alpha$ on held-out data, and remember that the hard term does not use it.

Last, the teacher's logits must come from exactly the same input the student sees. If you augment the image, run the teacher on the augmented image, not on the original.

> [!TIP]
> Cache the teacher's logits when the inputs are fixed. You then pay for the big model once, and every student epoch reads from memory.

## The exercise

You write the eight functions above. The checks confirm `soft_targets` on the worked numbers and its stability, that a larger temperature raises entropy, and that `kl_divergence` is zero for equal inputs and not symmetric. They check `distill_loss` on the worked example, `distill_grad` against finite differences, and that the gradient stays level at $T = 4$, 8, and 16. They check `logit_match_grad` against the gradient at $T = 1000$, the feature loss and its gradient against the worked example and finite differences, and that `fit_student` with soft targets beats hard labels on a 30-example problem.

## Further reading

- Geoffrey Hinton, Oriol Vinyals, and Jeff Dean, "Distilling the Knowledge in a Neural Network" (2015), arXiv:1503.02531. The source for soft targets, temperature, and the $T^2$ scaling.
- Adriana Romero and coauthors, "FitNets: Hints for Thin Deep Nets" (2015), arXiv:1412.6550. Matching intermediate features through a regressor.
- Yoon Kim and Alexander Rush, "Sequence-Level Knowledge Distillation" (2016), arXiv:1606.07947. Training a student on the teacher's generated sentences.
- The inference course chapter [Quantization on Ampere](https://qwen.ifkash.dev/c/18-quantization), for the bit-level side of shrinking a model.
- The reinforcement learning course, at [rl.ifkash.dev](https://rl.ifkash.dev/), for on-policy distillation of language models.
