---
title: "Beyond the plain CNN: augment, transfer, separate, and patchify"
slug: 041-augmentation-transfer-and-vit
part: "Part 5 — Vision and sequences"
summary: Stretch a small image set with label-preserving edits, reuse a frozen feature extractor, cut a layer's cost, and turn an image into tokens.
minutes: 70
gpu: false
objectives:
  - Explain why a crop, a flip, a brightness change, or a mixup is a regulariser, and when an edit changes the label.
  - Blend two examples and their labels with the same weight.
  - Train a linear head on a frozen feature extractor, and say when to unfreeze it.
  - Count the weights of a depthwise-separable layer and derive its cost ratio against a full layer.
  - Cut an image into patches and show that the patch embedding of a vision transformer is a strided convolution.
prereqs:
  - 038-convolution-and-receptive-field
  - 040-residual-networks
exercise: 041-augmentation-transfer-and-vit
---

# Beyond the plain CNN: augment, transfer, separate, and patchify

You have 500 labelled photos of two kinds of machine part, and you want a classifier. A convolutional network from the last three chapters has millions of weights and 500 examples to fit them. It will memorise the 500 photos, including the angle of the light and the position of the part in the frame, and then fail on the 501st. Three habits attack this from different sides. Make more examples out of the ones you have, by shifting, flipping, and blending them in ways that leave the answer unchanged. Borrow a feature extractor that someone trained on a huge image set, and fit only a small head. Or reshape the network: make each layer cheaper, or cut the image into tiles and treat each tile as a word.

This chapter teaches all three on tiny arrays. The last idea, tiles as words, is how a vision transformer reads a picture. It turns out to be a convolution you already know, with the stride set to the filter size.

## What you already know

- A convolution layer reuses one small filter at every position, so its weight count does not depend on the image size ([convolution](/lessons/038-convolution-and-receptive-field)).
- A stride moves a convolution's window by more than one pixel.
- A max-pool layer sends its gradient back to the position of the maximum, as [the convolution chapter](/lessons/038-convolution-and-receptive-field) shows. This chapter does not repeat it.
- Regularisation means making a model worse at memorising so that it is better on data it has not seen ([dropout and label smoothing](/lessons/035-dropout-and-label-smoothing)).
- A ridge regression has a closed-form solution that solves one linear system.

## Make more data without collecting it

If every training photo of a washer has the washer dead centre, the network can learn "bright ring at pixel (5, 5)" and score perfectly. A washer two pixels to the left then confuses it.

The fix is **data augmentation**: each time the training loop reads an example, it applies a small random edit first, so the network never sees exactly the same array twice. The edit must **preserve the label**, which means a person looking at the edited image would give the same answer. Because the edit is random, the network cannot lean on exact pixel positions, the same job dropout does for units. Augmentation regularises the input instead of the weights.

Three edits cover most image work:

- **Random crop.** Pad the image with zeros on every side, then cut out a window the original size at a random offset. The content shifts by up to the padding, and the rest of the window fills with zeros.
- **Random flip.** With probability one half, reverse the columns. A mirror image of a washer is a washer.
- **Colour jitter.** Change the brightness and contrast a little. With brightness shift $b$ and contrast factor $c$, a pixel $v$ in an image of mean $m$ becomes $\operatorname{clip}(c\,(v - m) + m + b,\ 0,\ 1)$.

Take two pixels, 0.2 and 0.8, so $m = 0.5$. With $c = 1.2$ and $b = 0.1$, they become $1.2 \times (0.2 - 0.5) + 0.6 = 0.24$ and $1.2 \times 0.3 + 0.6 = 0.96$. The gap widens from 0.6 to 0.72: the same scene under a different light.

> [!WARNING]
> An edit preserves the label only for a label that does not depend on what it changes. A horizontal flip turns a "b" into a "d" and a left arrow into a right arrow. A crop that cuts off the one object that defines the class leaves a blank image with the old label. Choose edits by asking what the label depends on, not by copying a recipe.

In code, the crop-and-flip draws three random numbers and slices:

```python
padded = np.pad(x, pad)
top = rng.integers(0, 2 * pad + 1)
left = rng.integers(0, 2 * pad + 1)
out = padded[top:top + H, left:left + W]
if rng.random() < 0.5:
    out = out[:, ::-1]
```

The checks replay your seeded generator, so draw `top`, then `left`, then the flip.

> [!TIP]
> Apply augmentation to training data only. Feed test images as they are.

### Mixup blends the labels too

The edits above leave the label alone. **Mixup** goes the other way. It takes two training examples and blends both the inputs and the labels with the same weight $\lambda$ between 0 and 1:

$$
\tilde{x} = \lambda\, x_i + (1 - \lambda)\, x_j, \qquad \tilde{y} = \lambda\, y_i + (1 - \lambda)\, y_j
$$

Read it aloud: the blended input is $\lambda$ of example $i$ plus the rest of example $j$, and the label gets the same blend. The labels are one-hot vectors, with a 1 in the true class and 0 elsewhere, so a blend such as $[0.75, 0.25]$ says "75 percent class 0, 25 percent class 1". The loss then asks the network to predict those fractions.

Take two examples with $x_i = (1, 0)$, $x_j = (0, 2)$, $y_i = (1, 0)$, and $y_j = (0, 1)$, and $\lambda = 0.75$. The blended input is $0.75 \times (1, 0) + 0.25 \times (0, 2) = (0.75, 0.5)$, and the blended label is $(0.75, 0.25)$. Figure 1 plots the whole line from one example to the other.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Two training points and the mixed points between them on a straight line, with their blended labels">
<line x1="60" y1="230" x2="230" y2="230" class="stroke-muted"/>
<line x1="60" y1="230" x2="60" y2="10" class="stroke-muted"/>
<text x="236" y="234" class="fig-small">x1</text><text x="52" y="6" class="fig-small">x2</text>
<line x1="170.0" y1="230" x2="60" y2="10.0" class="stroke-muted dashed"/>
<circle cx="170.0" cy="230.0" r="6" class="fill-a"/>
<text x="330" y="234.0" class="fig-small fig-mono">lam 1.00: x = (1.00, 0.00), y = (1.00, 0.00)</text>
<line x1="178.0" y1="230.0" x2="322" y2="230.0" class="stroke-grid"/>
<circle cx="142.5" cy="175.0" r="6" class="fill-b"/>
<text x="330" y="179.0" class="fig-small fig-mono">lam 0.75: x = (0.75, 0.50), y = (0.75, 0.25)</text>
<line x1="150.5" y1="175.0" x2="322" y2="175.0" class="stroke-grid"/>
<circle cx="115.0" cy="120.0" r="6" class="fill-b"/>
<text x="330" y="124.0" class="fig-small fig-mono">lam 0.50: x = (0.50, 1.00), y = (0.50, 0.50)</text>
<line x1="123.0" y1="120.0" x2="322" y2="120.0" class="stroke-grid"/>
<circle cx="87.5" cy="65.0" r="6" class="fill-b"/>
<text x="330" y="69.0" class="fig-small fig-mono">lam 0.25: x = (0.25, 1.50), y = (0.25, 0.75)</text>
<line x1="95.5" y1="65.0" x2="322" y2="65.0" class="stroke-grid"/>
<circle cx="60.0" cy="10.0" r="6" class="fill-a"/>
<text x="330" y="14.0" class="fig-small fig-mono">lam 0.00: x = (0.00, 2.00), y = (0.00, 1.00)</text>
<line x1="68.0" y1="10.0" x2="322" y2="10.0" class="stroke-grid"/>
<text x="330" y="24" class="fig-bold">Mixed examples and labels</text>
</svg>
Figure 1. Mixup draws a straight line between two examples. A point 75 percent of the way to the first example gets 75 percent of its label, so the label moves in step with the input.
```

It helps because it pushes the network to behave in a straight line between training points. Halfway between a washer and a bolt, the output should be about half and half, not a confident wrong answer. The mixup paper draws $\lambda$ from a Beta distribution for each batch; the exercise takes $\lambda$ as an argument so your code is deterministic.

> [!WARNING]
> The classic bug is to blend the inputs and forget the labels. The network then sees a half-washer, half-bolt image labelled "washer" with full confidence, which teaches the opposite of what mixup is for.

> [!KEY]
> Augmentation tells the network what your labels do not depend on, so it cannot spend capacity on those details.

## Borrow a feature extractor

Augmentation stretches 500 photos. **Transfer learning** goes further and borrows from a network trained on millions. The early layers of a trained image network learn general things: edges, colour blobs, simple textures. These help on almost any picture, including your machine parts. The late layers are specific to the original task. Yosinski et al. measured this by copying layers between networks: early layers transfer well, and late ones are tied to the original classes.

So you keep the early layers, call them the **feature extractor**, and replace the last layer with a **head** that you train for your own classes. Two stages follow:

1. **Freeze the extractor and train the head.** A frozen weight takes no updates, so you run each image through the extractor once and fit the head on the feature vectors. The head is small, so 500 examples can be enough.
2. **Optionally unfreeze and fine-tune.** Once the head works, you let some or all extractor weights move too, with a learning rate much smaller than the head's, so the pretrained features bend toward your task without being wiped out.

The order matters because the head starts random, so its gradient is large. If the extractor were free too, that gradient would push the trained early layers around to serve a head that knows nothing. Training the head first gives the extractor a sensible gradient when you unfreeze it.

The exercise runs stage 1 on a tiny task. The images are 8 by 8, with noise. Class 0 has a vertical bright bar and class 1 a horizontal one, so both classes have the same total brightness and a brightness threshold scores 0.517 on held-out images, a coin flip. The extractor is a fixed random matrix followed by a ReLU, and the head is a ridge regression on its features. Averaged over three seeds, the held-out accuracy depends on the extractor's width:

| Extractor width | Held-out accuracy | Brightness threshold |
|---|---|---|
| 16 | 0.713 | 0.517 |
| 64 | 0.842 | 0.517 |
| 256 | 0.972 | 0.517 |

No weight in the extractor was trained. Width gives the head more directions to choose from. A pretrained extractor does better, because data shaped its features, but the mechanics are the same: features once, head on top.

The head fits in closed form. Give each example a feature row, append a 1 for the bias, and set the target to $t = 2y - 1$, so class 1 maps to $+1$ and class 0 to $-1$. Ridge regression minimises

$$
\frac{1}{n}\sum_{i}\bigl(f_i^\top w - t_i\bigr)^2 + \lambda \lVert w \rVert^2
$$

where $f_i$ is the feature row of example $i$ with its bias entry and $\lambda$ sets how strongly large weights are penalised. Setting the gradient to zero gives one linear system, $\bigl(\tfrac{1}{n}F^\top F + \lambda I\bigr) w = \tfrac{1}{n}F^\top t$, which `np.linalg.solve` handles. You predict class 1 when the score $f^\top w$ is above 0.

Freezing is bookkeeping in the update step: a frozen parameter keeps its value, and every other one takes the usual step $p \leftarrow p - \eta g$, with $\eta$ the learning rate. That is `apply_update`.

> [!KEY]
> A frozen extractor turns "train a network" into "fit a small linear model on fixed features". Fine-tuning then lets the features adjust, at a small learning rate, after the head has stopped being random.

[Fine-tuning and LoRA](/lessons/059-fine-tuning-and-lora) takes the same idea to language models.

## Make a layer cheaper without making it weaker

The residual chapter cut cost with a 1 by 1 bottleneck. A second cut splits what a convolution does. A full 3 by 3 layer with $c_{in}$ channels in and $c_{out}$ out does two jobs at once with a single filter: it looks at 3 by 3 neighbourhoods in space, and it mixes the channels. A **depthwise-separable convolution** does the jobs in sequence:

1. **Depthwise step.** One 3 by 3 filter per input channel, and each filter sees only its own channel. This looks in space and mixes nothing across channels. It has $c_{in} k^2$ weights.
2. **Pointwise step.** A 1 by 1 convolution from $c_{in}$ channels to $c_{out}$. This mixes channels at each position and looks at no neighbours. It has $c_{in} c_{out}$ weights.

The total is $c_{in}k^2 + c_{in}c_{out}$ against $c_{in}c_{out}k^2$ for the full layer. Divide one by the other:

$$
\frac{c_{in}k^2 + c_{in}c_{out}}{c_{in}c_{out}k^2} = \frac{1}{c_{out}} + \frac{1}{k^2}
$$

Read it aloud: compared with the full layer, the separable one pays $1/c_{out}$ for channel mixing plus $1/k^2$ for spatial filtering. For $k = 3$ and 64 channels in and out, the full layer has $64 \times 64 \times 9 = 36{,}864$ weights, and the separable layer has $64 \times 9 + 64 \times 64 = 576 + 4096 = 4672$. The ratio is $4672 / 36864 = 0.1267$, which equals $1/64 + 1/9 = 0.1267$. The multiply-add count has the same ratio, because both layers produce the same output grid and the count at each position is proportional to the weight count. Figure 2 shows the fraction for four channel counts.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Bars comparing the weights of a full 3 by 3 layer and a depthwise-separable layer at 16, 32, 64, and 128 channels">
<text x="20" y="24" class="fig-bold">Weights of a 3 by 3 layer, c channels in and c out</text>
<text x="20" y="64">c = 16</text>
<rect x="90" y="50" width="300" height="14" class="fill-a"/>
<rect x="90" y="67" width="52.1" height="14" class="fill-b"/>
<text x="398" y="61" class="fig-small">2,304 full</text>
<text x="150.1" y="78" class="fig-small">400 separable, 0.174 of full</text>
<text x="20" y="106">c = 32</text>
<rect x="90" y="92" width="300" height="14" class="fill-a"/>
<rect x="90" y="109" width="42.7" height="14" class="fill-b"/>
<text x="398" y="103" class="fig-small">9,216 full</text>
<text x="140.7" y="120" class="fig-small">1,312 separable, 0.142 of full</text>
<text x="20" y="148">c = 64</text>
<rect x="90" y="134" width="300" height="14" class="fill-a"/>
<rect x="90" y="151" width="38.0" height="14" class="fill-b"/>
<text x="398" y="145" class="fig-small">36,864 full</text>
<text x="136.0" y="162" class="fig-small">4,672 separable, 0.127 of full</text>
<text x="20" y="190">c = 128</text>
<rect x="90" y="176" width="300" height="14" class="fill-a"/>
<rect x="90" y="193" width="35.7" height="14" class="fill-b"/>
<text x="398" y="187" class="fig-small">147,456 full</text>
<text x="133.7" y="204" class="fig-small">17,536 separable, 0.119 of full</text>
</svg>
Figure 2. Every full bar is scaled to the same length, so the orange bar is the fraction left. The fraction falls from 0.174 at 16 channels toward 1/9 = 0.111 as c grows, because the 1/c term vanishes.
```

The ratio cannot drop below $1/k^2$, which is $1/9$ for 3 by 3 filters, so the saving is about eight- to nine-fold, not a hundredfold.

The separable layer is not the same function as a full layer in general. It equals a full layer whose kernel has a special shape: the weight from input channel $c$ to output channel $o$ at spatial offset $(a, b)$ is $pw[o, c] \times dw[c, a, b]$. The spatial pattern for input channel $c$ is the same for every output channel, and only its scale changes. The exercise builds that kernel, runs a plain full convolution with it, and checks that your `depthwise_separable_conv` gives the same output. A full layer can represent more functions, so the saving costs some freedom, and the budget you free buys more layers or channels.

> [!KEY]
> Splitting "look in space" from "mix channels" cuts a 3 by 3 layer's weights to $1/c_{out} + 1/9$ of the full count, and the separable layer equals a full layer with a rank-one structure across channels and space.

## Cut the image into tiles

Convolution slides a small filter over the image. A **vision transformer** (ViT) takes a different route: it cuts the image into non-overlapping square tiles called **patches**, flattens each tile into a vector, and treats the vectors as a sequence of tokens, like words in a sentence. A transformer then reads the sequence the way it reads text, using [attention](/lessons/046-attention-as-soft-lookup) and the [transformer block](/lessons/052-the-transformer-block).

The cutting step is `patchify`. For an image of shape `(C, H, W)` and patch side $p$, it returns one row per patch, in reading order: left to right, then top to bottom. Each row holds the $C p^2$ numbers of one patch. Figure 3 shows it on a 6 by 6 image with $p = 2$, where the pixel numbers 0 to 35 make the order visible.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="A 6 by 6 image cut into nine 2 by 2 patches, numbered left to right and top to bottom">
<text x="20" y="24" class="fig-bold">Image, pixels numbered 0 to 35</text>
<rect x="20" y="40" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="35.0" y="59.5" text-anchor="middle" class="fig-small">0</text>
<rect x="50" y="40" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="65.0" y="59.5" text-anchor="middle" class="fig-small">1</text>
<rect x="80" y="40" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="95.0" y="59.5" text-anchor="middle" class="fig-small">2</text>
<rect x="110" y="40" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="125.0" y="59.5" text-anchor="middle" class="fig-small">3</text>
<rect x="140" y="40" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="155.0" y="59.5" text-anchor="middle" class="fig-small">4</text>
<rect x="170" y="40" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="185.0" y="59.5" text-anchor="middle" class="fig-small">5</text>
<rect x="20" y="70" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="35.0" y="89.5" text-anchor="middle" class="fig-small">6</text>
<rect x="50" y="70" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="65.0" y="89.5" text-anchor="middle" class="fig-small">7</text>
<rect x="80" y="70" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="95.0" y="89.5" text-anchor="middle" class="fig-small">8</text>
<rect x="110" y="70" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="125.0" y="89.5" text-anchor="middle" class="fig-small">9</text>
<rect x="140" y="70" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="155.0" y="89.5" text-anchor="middle" class="fig-small">10</text>
<rect x="170" y="70" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="185.0" y="89.5" text-anchor="middle" class="fig-small">11</text>
<rect x="20" y="100" width="30" height="30" class="fill-soft-d stroke-grid"/>
<text x="35.0" y="119.5" text-anchor="middle" class="fig-small">12</text>
<rect x="50" y="100" width="30" height="30" class="fill-soft-d stroke-grid"/>
<text x="65.0" y="119.5" text-anchor="middle" class="fig-small">13</text>
<rect x="80" y="100" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="95.0" y="119.5" text-anchor="middle" class="fig-small">14</text>
<rect x="110" y="100" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="125.0" y="119.5" text-anchor="middle" class="fig-small">15</text>
<rect x="140" y="100" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="155.0" y="119.5" text-anchor="middle" class="fig-small">16</text>
<rect x="170" y="100" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="185.0" y="119.5" text-anchor="middle" class="fig-small">17</text>
<rect x="20" y="130" width="30" height="30" class="fill-soft-d stroke-grid"/>
<text x="35.0" y="149.5" text-anchor="middle" class="fig-small">18</text>
<rect x="50" y="130" width="30" height="30" class="fill-soft-d stroke-grid"/>
<text x="65.0" y="149.5" text-anchor="middle" class="fig-small">19</text>
<rect x="80" y="130" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="95.0" y="149.5" text-anchor="middle" class="fig-small">20</text>
<rect x="110" y="130" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="125.0" y="149.5" text-anchor="middle" class="fig-small">21</text>
<rect x="140" y="130" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="155.0" y="149.5" text-anchor="middle" class="fig-small">22</text>
<rect x="170" y="130" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="185.0" y="149.5" text-anchor="middle" class="fig-small">23</text>
<rect x="20" y="160" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="35.0" y="179.5" text-anchor="middle" class="fig-small">24</text>
<rect x="50" y="160" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="65.0" y="179.5" text-anchor="middle" class="fig-small">25</text>
<rect x="80" y="160" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="95.0" y="179.5" text-anchor="middle" class="fig-small">26</text>
<rect x="110" y="160" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="125.0" y="179.5" text-anchor="middle" class="fig-small">27</text>
<rect x="140" y="160" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="155.0" y="179.5" text-anchor="middle" class="fig-small">28</text>
<rect x="170" y="160" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="185.0" y="179.5" text-anchor="middle" class="fig-small">29</text>
<rect x="20" y="190" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="35.0" y="209.5" text-anchor="middle" class="fig-small">30</text>
<rect x="50" y="190" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="65.0" y="209.5" text-anchor="middle" class="fig-small">31</text>
<rect x="80" y="190" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="95.0" y="209.5" text-anchor="middle" class="fig-small">32</text>
<rect x="110" y="190" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="125.0" y="209.5" text-anchor="middle" class="fig-small">33</text>
<rect x="140" y="190" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="155.0" y="209.5" text-anchor="middle" class="fig-small">34</text>
<rect x="170" y="190" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="185.0" y="209.5" text-anchor="middle" class="fig-small">35</text>
<rect x="20" y="40" width="60" height="60" class="stroke-ink"/>
<text x="24" y="52" class="fig-bold fig-small">T0</text>
<rect x="80" y="40" width="60" height="60" class="stroke-ink"/>
<text x="84" y="52" class="fig-bold fig-small">T1</text>
<rect x="140" y="40" width="60" height="60" class="stroke-ink"/>
<text x="144" y="52" class="fig-bold fig-small">T2</text>
<rect x="20" y="100" width="60" height="60" class="stroke-ink"/>
<text x="24" y="112" class="fig-bold fig-small">T3</text>
<rect x="80" y="100" width="60" height="60" class="stroke-ink"/>
<text x="84" y="112" class="fig-bold fig-small">T4</text>
<rect x="140" y="100" width="60" height="60" class="stroke-ink"/>
<text x="144" y="112" class="fig-bold fig-small">T5</text>
<rect x="20" y="160" width="60" height="60" class="stroke-ink"/>
<text x="24" y="172" class="fig-bold fig-small">T6</text>
<rect x="80" y="160" width="60" height="60" class="stroke-ink"/>
<text x="84" y="172" class="fig-bold fig-small">T7</text>
<rect x="140" y="160" width="60" height="60" class="stroke-ink"/>
<text x="144" y="172" class="fig-bold fig-small">T8</text>
<text x="260" y="24" class="fig-bold">Tokens, one row per patch</text>
<text x="260" y="48" class="fig-mono fig-small">T0</text>
<text x="300" y="48" class="fig-mono fig-small">[ 0,  1,  6,  7]</text>
<text x="260" y="70" class="fig-mono fig-small">T1</text>
<text x="300" y="70" class="fig-mono fig-small">[ 2,  3,  8,  9]</text>
<text x="260" y="92" class="fig-mono fig-small">T2</text>
<text x="300" y="92" class="fig-mono fig-small">[ 4,  5, 10, 11]</text>
<text x="260" y="114" class="fig-mono fig-small">T3</text>
<text x="300" y="114" class="fig-mono fig-small">[12, 13, 18, 19]</text>
<text x="260" y="136" class="fig-mono fig-small">T4</text>
<text x="300" y="136" class="fig-mono fig-small">[14, 15, 20, 21]</text>
<text x="260" y="158" class="fig-mono fig-small">T5</text>
<text x="300" y="158" class="fig-mono fig-small">[16, 17, 22, 23]</text>
<text x="260" y="180" class="fig-mono fig-small">T6</text>
<text x="300" y="180" class="fig-mono fig-small">[24, 25, 30, 31]</text>
<text x="260" y="202" class="fig-mono fig-small">T7</text>
<text x="300" y="202" class="fig-mono fig-small">[26, 27, 32, 33]</text>
<text x="260" y="224" class="fig-mono fig-small">T8</text>
<text x="300" y="224" class="fig-mono fig-small">[28, 29, 34, 35]</text>
<text x="260" y="248" class="fig-small">Each token is the patch read row by row.</text>
</svg>
Figure 3. A 6 by 6 image with p = 2 becomes 9 tokens of 4 numbers each. Notice that T1 holds pixels 2, 3, 8, and 9, the block directly right of T0, and that the pixel numbers inside a token jump by 6 where the patch changes row.
```

Token T0 holds pixels 0, 1, 6, and 7: row 0 of the patch, then row 1. T1 holds 2, 3, 8, and 9. At 224 by 224 pixels with $p = 16$, you get $(224/16)^2 = 196$ tokens, each a vector of $16 \times 16 \times 3 = 768$ numbers.

Each token then goes through one linear map to the model's working width $D$. This is **patch embedding**: the embedded token is $E\,t + b$, where $t$ is one flattened patch and $E$ has $D$ rows and $C p^2$ columns. For ViT-Base, $D = 768$, so $E$ has $768 \times 768 = 589{,}824$ weights.

Why it equals a strided convolution: a convolution with $D$ filters of size $p \times p \times C$ and stride $p$ puts each filter's window on exactly one patch, and the windows do not overlap because the stride equals the filter size. Each output value is a dot product between one filter and one patch, plus the bias, so row $d$ of $E$ is filter $d$ flattened. The two computations are the same arithmetic:

```python
tokens = patchify(x, p) @ kernel.reshape(D, -1).T + bias
```

The `kernel.reshape(D, -1)` flattens each filter in $(C, p, p)$ order, so `patchify` must flatten each patch in the same order. If it reads each patch with the channel axis last, so $(p, p, C)$, the dot products pair the wrong weights with the wrong pixels. On a one-channel image the two orders agree, so the bug hides until you test a colour image.

> [!WARNING]
> Check `patchify` on an image with at least two channels. A one-channel test passes both orders, and the bug shows up only when the embedding disagrees with the strided convolution.

The model also adds a position embedding, so it knows which tile is where ([position encodings](/lessons/051-position-encodings)), and a learned class token that collects the answer.

> [!KEY]
> Patch embedding is a convolution with filter size equal to stride, so a vision transformer begins with one convolutional layer and then replaces the rest of the sliding with attention between tiles.

> [!RECAP]
> - Augmentation applies a random label-preserving edit each time an example is read, which regularises the input.
> - Mixup blends two inputs and their one-hot labels with the same weight, so the labels move in step with the inputs. Blending only the inputs is a bug.
> - Transfer learning freezes a feature extractor and fits a head, then optionally unfreezes with a small learning rate.
> - A depthwise-separable layer has $c_{in}k^2 + c_{in}c_{out}$ weights, which is $1/c_{out} + 1/k^2$ of a full layer, and it equals a full layer with a factored kernel.
> - Patchify cuts an image into tiles, flattened channel first, and patch embedding of those tiles equals a convolution with stride equal to the filter size.

## Build it step by step

The checks call the functions in this order:

1. **`random_crop_flip(x, rng, pad=2)`** takes `x` of shape `(H, W)` and returns the same shape. Pad with `np.pad`, draw `top`, `left`, and the flip from `rng` in that order, slice, and reverse the columns on a flip. The usual mistake is a different draw order.
2. **`mixup(x, y, lam, perm)`** takes inputs `(N, ...)`, one-hot labels `(N, K)`, a weight, and partner indices. It returns `(x_mix, y_mix)`. The usual mistake is to return `y` unchanged.
3. **`separable_params(c_in, c_out, k)`** returns an integer, $c_{in}k^2 + c_{in}c_{out}$. The usual mistake is to multiply the depthwise term by `c_out`, which counts a full layer.
4. **`depthwise_separable_conv(x, dw, pw)`** takes `x` `(C, H, W)`, `dw` `(C, k, k)`, and `pw` `(C_out, C)`, and returns `(C_out, H - k + 1, W - k + 1)`. Loop over positions, sum the depthwise step over the last two axes only, then apply `np.tensordot(pw, mid, axes=(1, 0))`. The usual mistake is to sum over channels in the depthwise step.
5. **`patchify(x, p)`** takes `(C, H, W)` and returns `(patches, C * p * p)`. Reshape to `(C, H // p, p, W // p, p)`, transpose to `(H // p, W // p, C, p, p)`, and reshape. The usual mistake is to put the channel axis last.
6. **`patch_embed(x, kernel, bias)`** takes `kernel` `(D, C, p, p)` and `bias` `(D,)`, and returns `(patches, D)`. Use `patchify` and one matrix product with `kernel.reshape(D, -1).T`.
7. **`frozen_features(x, W)`** takes images `(N, H, W)` and returns `relu(x_flat @ W)`. The usual mistake is to forget the flatten.
8. **`fit_head(F, y, lam)`** appends a column of ones, sets `t = 2 * y - 1`, solves the ridge system with `np.linalg.solve`, and returns `w` of shape `(D + 1,)`. The usual mistake is to leave out the bias column.
9. **`head_predict(F, w)`** returns integer labels, 1 where the score is above 0.
10. **`apply_update(params, grads, lr, frozen)`** returns a dictionary in which frozen names keep their arrays and the rest take `params[k] - lr * grads[k]`. The usual mistake is to modify `params` in place.

## Where this breaks

- Augmentation can hurt when an edit changes the label. Flipping digits, rotating a "6" into a "9", and cropping the one defining object all teach the wrong answer.
- Mixup blurs the labels, which can hurt when classes are sharply separated. Treat $\lambda$ as a setting to tune.
- Transfer works best when source and target images look alike. Features from everyday photos help less on X-rays, where fine-tuning more layers usually matters.
- A vision transformer assumes less than a convolution, which assumes nearby pixels matter together and a pattern means the same wherever it sits. A transformer learns both from data, so it usually needs more data, stronger augmentation, or a pretrained start.
- A multiply-add count is a guide and not a stopwatch. A depthwise step does little arithmetic per number it reads from memory, so the running-time saving is often smaller than the count suggests. Patch size also trades detail for cost: attention work grows with the square of the token count, so $p = 8$ gives 784 tokens instead of 196 and about sixteen times the work.

## The exercise

You build `random_crop_flip`, `mixup`, `separable_params`, `depthwise_separable_conv`, `patchify`, `patch_embed`, `frozen_features`, `fit_head`, `head_predict`, and `apply_update`. The checks replay your generator against a reference crop and flip, confirm that a bright block survives every shift, and check mixed inputs and labels against the worked numbers. They confirm the separable weight count and ratio, match your separable convolution to a full convolution with a factored kernel, and match your patch embedding to a strided convolution on three channels. They check the ridge gradient with finite differences, confirm that a frozen random extractor plus your head beats a brightness threshold, and verify that a frozen parameter stays unchanged.

## Further reading

- Stanford CS231n, [course schedule](https://cs231n.stanford.edu/schedule.html): Lecture 6, "CNN Architectures", for the architecture families and transfer learning, and Lecture 8, "Attention and Transformers", for the vision transformer.
- Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), 2020. The vision transformer, with patch embedding.
- Zhang et al., [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412), 2017. Blending inputs and labels.
- Howard et al., [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861), 2017. Depthwise-separable convolutions and their cost ratio.
- Yosinski et al., [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792), 2014. Which layers transfer and how fine-tuning helps.
