---
title: "Detection and segmentation: IoU, NMS, average precision, and Dice"
slug: 042-detection-and-segmentation
part: "Part 5 — Vision and sequences"
summary: Score boxes with IoU, prune duplicates with NMS, grade a detector with average precision, and grade a mask with Dice.
minutes: 70
gpu: false
objectives:
  - Compute the intersection over union of two boxes, and explain why a missing clamp gives a positive overlap for boxes that are apart.
  - Write greedy non-maximum suppression, and say why it compares a box only with the boxes it kept.
  - Match detections to ground truth, build a precision-recall curve, and compute all-point average precision by hand.
  - Compute pixel IoU and the Dice coefficient for a mask, and write the soft Dice loss with its gradient.
  - Describe what an anchor is and what a U-Net skip path carries.
prereqs:
  - 014-scoring-classifiers
  - 038-convolution-and-receptive-field
  - 040-residual-networks
exercise: 042-detection-and-segmentation
---

# Detection and segmentation: IoU, NMS, average precision, and Dice

A classifier answers one question about a photo: what is in it? A street camera needs more. It must say that there are three cars and a cyclist, and where each one is. A scan reader at a hospital needs more again. It must say which pixels belong to the tumour, because a surgeon plans around the outline, not the label.

Both jobs need a way to say how close a guess is to the truth, and "right or wrong" does not do it. A box that misses the car by two pixels is nearly right. A detector also produces many guesses for one car, and you must keep one. This chapter builds four tools: an overlap score for boxes, a rule that prunes duplicates, a grade for a whole detector, and a score and a loss for masks. Each has a small trap that gives a plausible wrong number, so you check each against a hand example.

## What you already know

- Precision is the share of your positive calls that are right, and recall is the share of the true positives you found ([scoring classifiers](/lessons/014-scoring-classifiers)).
- A convolution slides a small filter over a picture and keeps the picture's grid shape ([convolution](/lessons/038-convolution-and-receptive-field)).
- A shortcut adds a block's input to its output, so the block only learns a correction ([residual networks](/lessons/040-residual-networks)).
- A gradient check compares a written gradient with a finite-difference estimate.

## Boxes and how much they overlap

Start with the picture in Figure 1. A box is a rectangle given by its top-left corner $(x_1, y_1)$ and its bottom-right corner $(x_2, y_2)$. Here $y$ grows downward, as it does in an image. Box A is $(0, 0, 4, 4)$ and box B is $(2, 2, 6, 6)$.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Left, two overlapping boxes with an intersection of 4 and a union of 28. Right, IoU falls from 1 to 0 as one box slides away along the diagonal.">
<rect x="30.0" y="40.0" width="80.0" height="80.0" class="fill-soft-a stroke-a"/>
<rect x="70.0" y="80.0" width="80.0" height="80.0" class="fill-soft-b stroke-b"/>
<rect x="70" y="80" width="40" height="40" class="fill-c stroke-c"/>
<text x="36.0" y="58.0" class="fig-bold">A</text>
<text x="132.0" y="152.0" class="fig-bold">B</text>
<text x="78" y="105" class="fig-small fill-paper">4</text>
<text x="30" y="212" class="fig-small">intersection 4, union 16 + 16 - 4 = 28</text>
<text x="30" y="232" class="fig-small">IoU = 4 / 28 = 0.143</text>
<line x1="330" y1="180" x2="570" y2="180" class="stroke-muted"/>
<line x1="330" y1="30" x2="330" y2="180" class="stroke-muted"/>
<rect x="342" y="30.0" width="36" height="150.0" class="fill-b"/>
<text x="360.0" y="24.0" text-anchor="middle" class="fig-small">1.00</text>
<text x="360.0" y="198" text-anchor="middle" class="fig-small">0</text>
<rect x="388" y="121.3" width="36" height="58.7" class="fill-b"/>
<text x="406.0" y="115.3" text-anchor="middle" class="fig-small">0.39</text>
<text x="406.0" y="198" text-anchor="middle" class="fig-small">1</text>
<rect x="434" y="158.6" width="36" height="21.4" class="fill-b"/>
<text x="452.0" y="152.6" text-anchor="middle" class="fig-small">0.14</text>
<text x="452.0" y="198" text-anchor="middle" class="fig-small">2</text>
<rect x="480" y="175.2" width="36" height="4.8" class="fill-b"/>
<text x="498.0" y="169.2" text-anchor="middle" class="fig-small">0.03</text>
<text x="498.0" y="198" text-anchor="middle" class="fig-small">3</text>
<rect x="526" y="180.0" width="36" height="0.0" class="fill-b"/>
<text x="544.0" y="174.0" text-anchor="middle" class="fig-small">0.00</text>
<text x="544.0" y="198" text-anchor="middle" class="fig-small">4</text>
<text x="450.0" y="220" text-anchor="middle" class="fig-small">slide B along the diagonal by this many units</text>
<text x="330" y="20" class="fig-small">IoU of A and the slid box</text>
</svg>
Figure 1. Two boxes overlap by 4 of 28 units, an IoU of 0.14. Sliding the second box by a single unit already drops the IoU from 1 to 0.39, so IoU punishes small shifts hard.
```

Why not compare the four coordinates directly? A miss of 3 pixels means very different things for a box 10 pixels wide and for a box 300 pixels wide. You want a score that is the same for a tiny box and a huge one, and falls to zero when the boxes share nothing. The share of the combined area that they have in common does both. It is the **intersection over union**, or IoU.

For the two boxes in Figure 1, the shared part is a square from $(2, 2)$ to $(4, 4)$, so its area is $2 \times 2 = 4$. Each box has area $4 \times 4 = 16$. Adding the two areas counts the shared square twice, so the union is $16 + 16 - 4 = 28$. The IoU is $4 / 28 = 1/7 \approx 0.143$. The bar chart in Figure 1 shows how fast this falls. Sliding box B by one unit along the diagonal leaves an IoU of 0.39, and two units leaves 0.14.

Written as a formula, the overlap along each axis is the smaller right edge minus the larger left edge, and the intersection is the product of the two overlaps:

$$
w = \max(0,\ \min(x_2, x_2') - \max(x_1, x_1')), \qquad h = \max(0,\ \min(y_2, y_2') - \max(y_1, y_1'))
$$

$$
\text{IoU} = \frac{w\,h}{\text{area} + \text{area}' - w\,h}
$$

Read the first line aloud: take the right edge that is more to the left, take the left edge that is more to the right, and subtract. If the boxes overlap on this axis, the result is positive. If they are apart, it is negative, and the $\max(0, \cdot)$ turns it into zero. The second line divides the shared area by the combined area.

> [!WARNING]
> Forget the $\max(0, \cdot)$ and the code runs, passes a test on overlapping boxes, and is wrong. Take box $(0, 0, 2, 2)$ and box $(5, 5, 7, 7)$. Both raw overlaps are $-3$, and $(-3) \times (-3) = +9$, so the "intersection" is a positive 9 for two boxes that are far apart. Clamp each overlap at zero before you multiply.

To score every detection against every ground-truth box, you want all pairs at once. Give the first set shape `(N, 4)` and the second `(M, 4)`. Indexing with `a[:, None, :]` and `b[None, :, :]` lines up every pair, and `np.maximum` and `np.minimum` then broadcast to shape `(N, M, 2)`. The result is an `(N, M)` matrix with no Python loop.

```python
top_left = np.maximum(a[:, None, :2], b[None, :, :2])      # (N, M, 2)
bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])  # (N, M, 2)
wh = np.clip(bottom_right - top_left, 0.0, None)           # clamp each side at 0
inter = wh[..., 0] * wh[..., 1]                            # (N, M)
```

> [!KEY]
> IoU is the shared area divided by the combined area. It ignores the size of the boxes, so one threshold means the same thing for a pedestrian and a bus. Clamp each side's overlap at zero before you multiply.

## Many boxes for one object: non-maximum suppression

A detector does not draw one box per object. It scores a box at thousands of places in the image, and every place that sees most of a car fires. You end up with a cluster of nearly identical boxes on one car, and you want to keep one.

The obvious fix is to keep the top three boxes. This fails because a box on the second car can score below the third duplicate on the first car, so you keep three boxes on one car and none on the other.

The fix that works is greedy. Sort the boxes by score. Keep the best one. Throw away every remaining box that overlaps it by more than a threshold, because those are its duplicates. Then take the best of what is left and repeat. This is **non-maximum suppression** (NMS). A box on the second car has a small IoU with the first car's box, so it survives to become the next kept box.

Work an example with threshold 0.5. Four boxes share the same height, 10, and have these $x$ ranges and scores:

| Box | x range | Score |
|---|---|---|
| 0 | 0 to 10 | 0.9 |
| 1 | 3 to 13 | 0.8 |
| 2 | 6 to 16 | 0.7 |
| 3 | 30 to 40 | 0.6 |

Box 0 has the top score, so keep it. Box 1 overlaps it by $7 \times 10 = 70$ out of a union of $200 - 70 = 130$, an IoU of 0.538. That is above 0.5, so drop box 1. Box 2 overlaps box 0 by $4 \times 10 = 40$ out of $160$, an IoU of 0.25. That is below the threshold, so box 2 is kept. Box 3 is far away and is kept. The result is boxes 0, 2, and 3.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Four boxes drawn as bars along x. Box 0 is kept, box 1 is suppressed, box 2 is kept, and box 3 is kept.">
<text x="20" y="22" class="fig-bold">Boxes share one y range, so each is drawn as a bar along x</text>
<rect x="60" y="40" width="120" height="28" rx="4" class="fill-soft-c stroke-c"/>
<text x="68" y="59" class="fig-mono">box 0  score 0.9</text>
<text x="190" y="59" class="fig-small">keep</text>
<rect x="96" y="90" width="120" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="104" y="109" class="fig-mono">box 1  score 0.8</text>
<text x="226" y="109" class="fig-small">suppressed by 0</text>
<rect x="132" y="140" width="120" height="28" rx="4" class="fill-soft-c stroke-c"/>
<text x="140" y="159" class="fig-mono">box 2  score 0.7</text>
<text x="262" y="159" class="fig-small">keep</text>
<rect x="420" y="190" width="120" height="28" rx="4" class="fill-soft-c stroke-c"/>
<text x="428" y="209" class="fig-mono">box 3  score 0.6</text>
<text x="410" y="209" text-anchor="end" class="fig-small">keep</text>
<text x="20" y="245" class="fig-small">IoU(0,1) = 0.538   IoU(1,2) = 0.538   IoU(0,2) = 0.250   threshold 0.5</text>
</svg>
Figure 2. Box 2 overlaps suppressed box 1 (IoU 0.538) but not kept box 0 (IoU 0.25), so it survives. A version that also compares against suppressed boxes would wrongly drop it.
```

Box 2 is the instructive one. It overlaps box 1 with IoU 0.538, above the threshold, but box 1 is gone. A removed box can't claim anything. If you compare each box against every higher-scored box, suppressed or not, box 2 is dropped because of box 1, and you lose a box that has little to do with any survivor.

> [!WARNING]
> Compare a candidate only with the boxes you have kept. The slip is easy to make with a precomputed IoU matrix, because the row for box 2 holds an entry for box 1 and nothing stops you reading it. Your answer on the four-box example is then `[0, 3]` instead of `[0, 2, 3]`.

A low threshold also removes the real second member of a tight crowd. A high threshold keeps the crowd and some duplicates. The usual choice is 0.5.

The cost is $O(N^2)$ for the IoU matrix, which is cheap for a few hundred detections per image.

> [!KEY]
> NMS sorts by score, keeps the best box, drops every box that overlaps it by more than the threshold, and repeats with what is left. A suppressed box never suppresses another.

## Grading a detector: matching, precision-recall, and average precision

A detector returns a list of boxes, each with a score and an image, and you want one number for how good it is. A classifier would use precision and recall, defined in [scoring classifiers](/lessons/014-scoring-classifiers). They need a "true positive", and for boxes the labels don't give one. You must decide whether each box found an object.

Use IoU for that. Fix a threshold, usually 0.5. Go through the detections from the highest score to the lowest. For each one, look at the ground-truth boxes in its image that no earlier detection has claimed, and take the one with the highest IoU. If that IoU is at least the threshold, the detection is a **true positive** and the ground-truth box is used up. Otherwise it is a **false positive**. If two detections sit on one object, only the higher-scored one is a true positive, because the object was already found. That is why duplicates hurt a detector's grade, and why NMS helps.

A ground-truth box that no detection claims is a false negative. There is no "true negative" count, because no sensible number of empty boxes exists that you could have declined to draw. That is why detection grades use precision and recall, not the ROC curve.

Take the example below. An image holds three objects, at $x$ ranges 0 to 10, 20 to 30, and 40 to 50. The detector returns six boxes. In score order:

| Rank | Score | What it is | IoU with its best free object | Result |
|---|---|---|---|---|
| 1 | 0.95 | shifted 1 unit from object 1 | 0.82 | true positive |
| 2 | 0.90 | shifted from object 2 | 0.68 | true positive |
| 3 | 0.80 | a second box on object 1 | 0.82, but object 1 is used | false positive |
| 4 | 0.70 | clutter, no object | 0 | false positive |
| 5 | 0.60 | shifted from object 3 | 0.67 | true positive |
| 6 | 0.50 | a second box on object 2 | 0, every object is used | false positive |

After each detection, compute precision as hits so far over detections so far, and recall as hits so far over the 3 objects. This gives a precision of 1, 1, 0.667, 0.5, 0.6, 0.5 and a recall of 0.333, 0.667, 0.667, 0.667, 1, 1. Each point is a dot in Figure 3.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Precision against recall for six detections, with the envelope whose area is the average precision of 0.867.">
<line x1="70" y1="210" x2="530" y2="210" class="stroke-muted"/>
<line x1="70" y1="30" x2="70" y2="210" class="stroke-muted"/>
<path d="M 70.0 210.0 L 70.0 30.0 L 223.3 30.0 L 223.3 30.0 L 376.7 30.0 L 376.7 102.0 L 530.0 102.0 L 530.0 210.0 Z" class="fill-soft-c stroke-c"/>
<circle cx="223.3" cy="30.0" r="5" class="fill-a"/>
<text x="231.3" y="22.0" class="fig-small">1</text>
<circle cx="376.7" cy="30.0" r="5" class="fill-a"/>
<text x="384.7" y="22.0" class="fig-small">2</text>
<circle cx="376.7" cy="90.0" r="5" class="fill-a"/>
<text x="384.7" y="106.0" class="fig-small">3</text>
<circle cx="376.7" cy="120.0" r="5" class="fill-a"/>
<text x="384.7" y="136.0" class="fig-small">4</text>
<circle cx="530.0" cy="102.0" r="5" class="fill-a"/>
<text x="538.0" y="94.0" class="fig-small">5</text>
<circle cx="530.0" cy="120.0" r="5" class="fill-a"/>
<text x="538.0" y="136.0" class="fig-small">6</text>
<text x="70.0" y="228" text-anchor="middle" class="fig-small">0</text>
<text x="60" y="214.0" text-anchor="end" class="fig-small">0</text>
<text x="300.0" y="228" text-anchor="middle" class="fig-small">0.5</text>
<text x="60" y="124.0" text-anchor="end" class="fig-small">0.5</text>
<text x="530.0" y="228" text-anchor="middle" class="fig-small">1</text>
<text x="60" y="34.0" text-anchor="end" class="fig-small">1</text>
<text x="300.0" y="250" text-anchor="middle" class="fig-small">recall</text>
<text x="70" y="20" class="fig-small">precision. Green area = AP = 0.867</text>
</svg>
Figure 3. Each dot is the precision and recall after one more detection. Detection 3 (a duplicate) and detection 4 (clutter) pull precision down, but the envelope only keeps the best precision at each recall or higher, so the area is 0.867.
```

A curve is not a number, so you reduce it to its area. The curve zigzags, because a false positive makes precision fall and the next true positive makes it rise. The standard step is to replace each precision by the largest precision at that recall or any higher recall. At recall 1, for example, the best precision any point reaches is 0.6, so the envelope at recall 1 is 0.6, not 0.5. This reads "how precise could you be if you accepted this much recall, or more". The area under the envelope is the **average precision** (AP).

For the example, the envelope is 1 up to recall 0.667 and 0.6 from there to recall 1. The area is $\tfrac13 \cdot 1 + \tfrac13 \cdot 1 + \tfrac13 \cdot 0.6 = 0.867$. In code, pad recall with 0 in front and 1 at the end, pad precision with 0 on both sides, take a running maximum from the right, and add up the rectangles where recall steps up:

$$
\text{AP} = \sum_k (r_k - r_{k-1})\, \hat{p}_k, \qquad \hat{p}_k = \max_{j \ge k} p_j
$$

Read it aloud: each time recall rises from $r_{k-1}$ to $r_k$, add a rectangle as wide as the rise and as tall as the best precision from that point on.

You will see two conventions. This chapter uses **all-point** interpolation, as in the PASCAL VOC challenge from 2010 on, which computes the exact area under the envelope. COCO instead samples the envelope at 101 recall values, $0, 0.01, \dots, 1$, and averages them. On the example this gives 0.8653, close to the exact 0.8667. COCO also averages AP over ten IoU thresholds, from 0.50 to 0.95 in steps of 0.05, which rewards boxes that fit tightly. Finally, **mean average precision** (mAP) is the average of AP over object classes. Say which convention a reported number uses, because the same detector scores differently under each.

> [!KEY]
> A detection is a true positive when it takes a not-yet-claimed ground-truth box at an IoU of at least the threshold, in score order. Average precision is the area under the precision-recall curve after each precision is replaced by the best precision at that recall or higher.

## Masks instead of boxes: pixel IoU, Dice, and a loss

Segmentation labels every pixel. The output is a **mask**, a grid where 1 means "this pixel belongs to the object". You grade it against a true mask.

Pixel accuracy, the share of pixels labelled correctly, fails on small objects. If a tumour covers 1% of a scan, a model that outputs all zeros is 99% accurate and finds nothing. Accuracy rewards the huge background. You want a score that looks only at the object.

Two scores do. Let $A$ be the set of predicted object pixels and $B$ the set of true object pixels. **Pixel IoU** is $|A \cap B| / |A \cup B|$, the same ratio as for boxes, counting pixels. The **Dice coefficient** is

$$
\text{Dice} = \frac{2\,|A \cap B|}{|A| + |B|}
$$

Read it aloud: twice the shared pixels, divided by the total pixels in both masks. Take a 4 by 4 grid. The true mask is the $2 \times 2$ block in rows 1 and 2, columns 1 and 2, so $|B| = 4$. The prediction covers rows 1 and 2 and columns 1 to 3, so $|A| = 6$. The shared pixels number 4. The IoU is $4/6 = 0.667$ and the Dice is $2 \cdot 4 / (4 + 6) = 0.8$. The two always agree on the ranking, because Dice $= 2\,\text{IoU} / (1 + \text{IoU})$. Check: $2 \cdot 0.667 / 1.667 = 0.8$.

Dice is not differentiable as written, because a mask of hard 0s and 1s has no slope. A network outputs a probability $p_i$ for each pixel, between 0 and 1. Replace the set sizes with sums to get the **soft Dice**, with truth $g_i$ per pixel and a small constant $\varepsilon$ that keeps the ratio defined when both masks are empty:

$$
D = \frac{2\sum_i p_i g_i + \varepsilon}{\sum_i p_i + \sum_i g_i + \varepsilon}, \qquad L = 1 - D
$$

Let $N = 2\sum_i p_i g_i + \varepsilon$ be the numerator and $S = \sum_i p_i + \sum_i g_i + \varepsilon$ the denominator. Pixel $i$ changes $N$ by $2 g_i$ per unit of $p_i$ and changes $S$ by 1. The quotient rule gives

$$
\frac{\partial L}{\partial p_i} = -\frac{2 g_i S - N}{S^2}
$$

Read it aloud: for a pixel whose truth is 1, the numerator is $2S - N$, which is always positive because $N$ can't exceed $2S$, so raising $p_i$ lowers the loss. For a pixel whose truth is 0, it is $-N$, so raising $p_i$ raises the loss. That is what you want.

Work one case with four pixels, $p = (0.9, 0.8, 0.2, 0.1)$, $g = (1, 1, 0, 0)$, and $\varepsilon = 0$. Then $N = 2(0.9 + 0.8) = 3.4$ and $S = 2.0 + 2 = 4$, so $D = 0.85$ and $L = 0.15$. The gradient for the two object pixels is $-(2 \cdot 4 - 3.4)/16 = -0.2875$ and for the two background pixels it is $+3.4/16 = +0.2125$. The signs push the first two probabilities up and the last two down.

```python
num = 2.0 * np.sum(p * g) + eps
den = np.sum(p) + np.sum(g) + eps
loss = 1.0 - num / den
grad = -(2.0 * g * den - num) / den**2    # same shape as p
```

> [!KEY]
> Pixel accuracy rewards the background. Dice and pixel IoU score only the object, and soft Dice replaces set sizes with sums of probabilities so a network can train on it. The V-Net paper introduced it for 3-D scans, where the object fills a small share of the volume.

When the true mask is tiny, one pixel moves $S$ and $N$ by a large share, so the gradient is noisy. Many projects add cross-entropy, which has a steady gradient, to the Dice loss.

## Two design ideas: anchors and the U-Net skip path

You build neither of these two ideas here, but the papers rely on both.

An **anchor** is a reference box fixed in advance. At every position of a feature grid, a detector places a few anchors of different sizes and shapes, say a square, a wide box, and a tall box. For each anchor, the network predicts a score and four small offsets that nudge the anchor onto an object. This turns "find boxes anywhere" into "grade a fixed list of candidates", which a convolution can do. IoU appears in training: an anchor counts as a positive example for an object when their IoU is above a threshold, and as background when it is below a lower one. Several anchors overlap one object, so several fire, and that is where the duplicates come from.

A **U-Net** is the standard network for segmentation. An encoder halves the grid again and again, so each cell sees a wider area and knows what is there. A decoder grows the grid back, so that it can label every pixel. Shrinking throws away precise position, which the decoder can't recover from the small grid alone. So the U-Net copies the encoder's feature map at each size and glues it onto the decoder's map of the same size, as shown in Figure 4. This is the **skip path**. The decoder then has both "what" from below and "where" from the skip. It resembles the shortcut in [residual networks](/lessons/040-residual-networks), with one difference: a residual block adds the two maps, and a U-Net concatenates them along the channel axis, so the later layers see both and can weight each.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="A U-shaped network: the encoder shrinks the grid, the decoder grows it back, and dashed skip arrows carry each encoder map across to the decoder map of the same size.">
<defs><marker id="c041-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="20" y="22" class="fig-bold">Encoder: shrink the grid</text>
<text x="360" y="22" class="fig-bold">Decoder: grow it back</text>
<rect x="30" y="40" width="40" height="160" rx="4" class="fill-soft-a stroke-a"/>
<rect x="100" y="70" width="36" height="100" rx="4" class="fill-soft-a stroke-a"/>
<rect x="166" y="95" width="32" height="50" rx="4" class="fill-soft-a stroke-a"/>
<rect x="232" y="108" width="28" height="24" rx="4" class="fill-soft-d stroke-d"/>
<rect x="296" y="95" width="32" height="50" rx="4" class="fill-soft-c stroke-c"/>
<rect x="362" y="70" width="36" height="100" rx="4" class="fill-soft-c stroke-c"/>
<rect x="432" y="40" width="40" height="160" rx="4" class="fill-soft-c stroke-c"/>
<text x="50" y="222" text-anchor="middle" class="fig-small">full</text>
<text x="118" y="222" text-anchor="middle" class="fig-small">1/2</text>
<text x="182" y="222" text-anchor="middle" class="fig-small">1/4</text>
<text x="246" y="222" text-anchor="middle" class="fig-small">1/8</text>
<text x="312" y="222" text-anchor="middle" class="fig-small">1/4</text>
<text x="380" y="222" text-anchor="middle" class="fig-small">1/2</text>
<text x="452" y="222" text-anchor="middle" class="fig-small">full</text>
<path d="M 70 52 L 430 52" class="stroke-b dashed" fill="none" marker-end="url(#c041-arrow)"/>
<path d="M 136 78 L 360 78" class="stroke-b dashed" fill="none" marker-end="url(#c041-arrow)"/>
<path d="M 198 102 L 294 102" class="stroke-b dashed" fill="none" marker-end="url(#c041-arrow)"/>
<text x="490" y="120" class="fig-small">one mask value</text>
<text x="490" y="138" class="fig-small">per pixel</text>
<text x="20" y="250" class="fig-small">Dashed arrows: skip paths. Each copies an encoder map and joins it to the decoder map of the same size.</text>
</svg>
Figure 4. The encoder halves the grid until each cell sees a wide area. The decoder doubles it back. The skip paths carry position detail across, so the output mask has sharp edges.
```

> [!RECAP]
> - IoU is shared area over combined area. Clamp each side's overlap at zero, and compute all pairs with broadcasting.
> - NMS keeps the best box, drops its overlaps above a threshold, and repeats. Compare only with kept boxes.
> - A detection is a true positive when it claims an unclaimed ground-truth box at IoU of at least the threshold, in score order. Duplicates become false positives.
> - All-point AP is the area under the precision envelope. COCO uses 101 recall points.
> - Dice is $2|A \cap B| / (|A| + |B|)$, and soft Dice has the gradient $-(2 g_i S - N)/S^2$.

## Build it step by step

1. **`box_iou(a, b)`** takes boxes of shape `(N, 4)` and `(M, 4)` as $(x_1, y_1, x_2, y_2)$ and returns an `(N, M)` matrix. Use `a[:, None, :]` and `b[None, :, :]` with `np.maximum` and `np.minimum`, and `np.clip(..., 0, None)` on the side lengths. The mistake people most often make is to skip the clamp, which gives a positive overlap for boxes that are apart in both $x$ and $y$.
2. **`nms(boxes, scores, thr)`** returns the kept indices, best score first. Sort with `np.argsort(-scores, kind="stable")`, then keep a box only when its IoU with every kept box is at most `thr`. The mistake people most often make is to compare with suppressed boxes as well.
3. **`match_detections(boxes, scores, image_ids, gt_boxes, gt_image_ids, thr)`** returns `(order, is_tp)`. Walk the detections in score order. A detection may only claim an unclaimed ground-truth box from its own image, and takes the one with the largest IoU. The mistake people most often make is to let two detections claim one ground-truth box, which counts a duplicate as a hit.
4. **`precision_recall(is_tp, n_gt)`** takes the hit flags in score order and the number of ground-truth boxes, and returns precision and recall after each detection. Use `np.cumsum`. Divide recall by `n_gt`, not by the number of detections.
5. **`average_precision(precision, recall)`** returns the all-point area. Pad both arrays, take a running maximum from the right with `np.maximum.accumulate` on the reversed array, and add up the rectangles where recall changes. The mistake people most often make is to skip the envelope and integrate the zigzag.
6. **`mask_iou(pred, gt)`** and **`dice_score(pred, gt)`** take binary masks and return a float. Count with `np.logical_and` and `np.logical_or`. Return 1.0 when both masks are empty, because the division gives NaN.
7. **`soft_dice_loss(p, g, eps)`** returns `(loss, grad)`, with `grad` the same shape as `p`. Compute the numerator and denominator once and reuse them in the gradient. The mistake people most often make is to forget that the denominator also depends on `p`.

## Where this breaks

- **The thresholds are choices.** AP at an IoU of 0.5 hides sloppy boxes, which is why COCO averages from 0.50 to 0.95. NMS at 0.5 drops real objects in a crowd.
- **Dice on an empty true mask.** If the true mask is empty and the prediction is not, hard Dice is 0 whatever the prediction. The $\varepsilon$ in soft Dice keeps the loss finite.
- **Boxes describe thin objects badly.** A diagonal pipe fills a box that is mostly background, which is the reason to use a mask.

## The exercise

You implement `box_iou`, `nms`, `match_detections`, `precision_recall`, `average_precision`, `mask_iou`, `dice_score`, and `soft_dice_loss`. The ten checks run from the smallest piece to the whole: hand values, a double loop, boxes apart in both $x$ and $y$, the four-box NMS example and a brute-force reference, the six-detection AP of 0.8667, Dice against $2\,\text{IoU}/(1 + \text{IoU})$, and the soft Dice gradient against finite differences. The last check runs your pieces together on a seeded detector that fires four times on each object, and requires NMS to raise the AP.

## Further reading

- Stanford CS231n, Lecture 9, "Object Detection, Image Segmentation, Visualizing and Understanding" (Spring 2026 schedule at `cs231n.stanford.edu/schedule.html`).
- MIT 6.S191, Lecture 3, "Deep Computer Vision" (2026 schedule at `introtodeeplearning.com`).
- Everingham, Van Gool, Williams, Winn, and Zisserman, "The PASCAL Visual Object Classes (VOC) Challenge", 2010. It defines the matching rule and the all-point AP used here.
- Lin and coauthors, "Microsoft COCO: Common Objects in Context", 2014. It is the source of the COCO dataset and its AP convention.
- Ronneberger, Fischer, and Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", 2015.
- Milletari, Navab, and Ahmadi, "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation", 2016. It introduced the Dice loss.
