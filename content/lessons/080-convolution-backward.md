---
title: Sending the gradient back through a convolution
slug: 080-convolution-backward
part: "Part 5 — Vision and sequences"
summary: Find the gradient of every weight, bias, and pixel in a convolution, and see why each one is a sum over positions or windows.
minutes: 40
gpu: false
objectives:
  - Explain why a shared weight's gradient adds up over every position and every image in the batch.
  - Explain why a pixel's gradient adds up over every window that contains it, and count those windows.
  - Write col2im as a scatter-add and use it to finish the backward pass of a convolution.
  - Check dx, dw, and db against finite differences, and test col2im against im2col with the adjoint identity.
prereqs:
  - 062-backprop-array-operations
  - 078-output-size-and-receptive-field
  - 079-convolution-as-matrix-multiply
exercise: 080-convolution-backward
---

# Sending the gradient back through a convolution

A small network that finds cat ears gets a photo wrong, and the loss says the output was too low. Training needs to know whom to blame. A 3 by 3 filter slides over the photo and produces a number at every position, so each of its nine weights influenced thousands of outputs. Each pixel was read by up to nine windows, so each pixel influenced several outputs too. How much of the blame goes to each weight and each pixel?

You could ask the question the slow way: nudge one weight, run the forward pass again, and see how the loss moves. A layer with 64 input channels, 64 output channels, and 3 by 3 filters has 36,864 weights, so that costs 36,864 forward passes for one step of training. The fast way is to follow the blame backward through the same steps the forward pass took. This chapter shows how, and the answer is a short rule: where something was shared, the blame adds up.

## What you already know

You already use these ideas:

- For a matrix product $Y = WX$, the gradients are $dW = dY\,X^\top$ and $dX = W^\top dY$, from "The backward rule for each array operation".
- When a value feeds several places, its gradient is the sum of the gradients flowing back from each place.
- The forward pass of a convolution is $Y_{\text{mat}} = W_{\text{mat}}\,\text{col} + b$, from "Turning a convolution into one matrix multiply".
- `im2col` copies each pixel into every window that contains it.
- A finite difference, $(L(x + \epsilon) - L(x - \epsilon)) / 2\epsilon$, estimates a gradient numerically.

## A shared weight collects blame from every position

Use the small example from the previous chapter: the 3 by 3 image holding 1 to 9, a 2 by 2 kernel with rows (1, 2) and (3, 4), and stride 1. The output $Y$ is 2 by 2. Suppose the loss tells you, for each of the four outputs, that raising it by 1 raises the loss by 1. So $dY$ is all ones.

Take the kernel weight in the top-left tap, $w_{00}$. It was used four times: window 1 multiplied it by pixel 1, window 2 by pixel 2, window 3 by pixel 4, and window 4 by pixel 5. Raise $w_{00}$ by a tiny amount $\epsilon$ and the four outputs rise by $\epsilon \cdot 1$, $\epsilon \cdot 2$, $\epsilon \cdot 4$, and $\epsilon \cdot 5$. The loss rises by $\epsilon$ times 1 + 2 + 4 + 5 = 12 (each output counts once, because $dY$ is all ones). So $dw_{00} = 12$. The other three taps give $16$, $24$, and $28$, each a sum over the four windows of the pixel that tap met.

That is the general rule for a weight: its gradient is the sum, over every output position that used it, of the upstream gradient at that position times the pixel it met. With a batch, every image counts too, because the same weight is used on every image. The bias is the easiest case. It is added once to every output of its channel, so raising it by $\epsilon$ raises every output by $\epsilon$, and $db = 4$, the sum of $dY$ over the positions. With several images, the sum also runs over the batch.

## A pixel collects blame from every window that read it

Ask the same question for a pixel. Pixel 1, the top-left corner, sits in one window, at tap $(0, 0)$, so its gradient is $1 \times w_{00} = 1$. The middle pixel, 5, sits in all four windows, and it meets a different tap each time: tap $(1, 1)$ with weight 4, then $(1, 0)$ with weight 3, then $(0, 1)$ with weight 2, then $(0, 0)$ with weight 1. Each window pushes some blame onto it, so the gradient is $4 + 3 + 2 + 1 = 10$. Figure 1 counts the windows that cover each pixel. The whole $dx$ for this example is

$$
dx = \begin{bmatrix} 1 & 3 & 2 \\ 4 & 10 & 6 \\ 3 & 7 & 4 \end{bmatrix}
$$

```figure
<svg viewBox="0 0 600 352" role="img" aria-label="Four 2 by 2 windows on a 3 by 3 image and the number of windows covering each pixel">
<text x="20" y="22" class="fig-bold">Where each of the four windows sits (1 = inside the window)</text>
<rect x="30" y="40" width="22" height="22" class="fill-soft-a stroke-grid"/>
<text x="41" y="55.5" text-anchor="middle" class="fig-small">1</text>
<rect x="52" y="40" width="22" height="22" class="fill-soft-a stroke-grid"/>
<text x="63" y="55.5" text-anchor="middle" class="fig-small">1</text>
<rect x="74" y="40" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="85" y="55.5" text-anchor="middle" class="fig-small">0</text>
<rect x="30" y="62" width="22" height="22" class="fill-soft-a stroke-grid"/>
<text x="41" y="77.5" text-anchor="middle" class="fig-small">1</text>
<rect x="52" y="62" width="22" height="22" class="fill-soft-a stroke-grid"/>
<text x="63" y="77.5" text-anchor="middle" class="fig-small">1</text>
<rect x="74" y="62" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="85" y="77.5" text-anchor="middle" class="fig-small">0</text>
<rect x="30" y="84" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="41" y="99.5" text-anchor="middle" class="fig-small">0</text>
<rect x="52" y="84" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="63" y="99.5" text-anchor="middle" class="fig-small">0</text>
<rect x="74" y="84" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="85" y="99.5" text-anchor="middle" class="fig-small">0</text>
<text x="63" y="122" text-anchor="middle" class="fig-small">window 1</text>
<text x="116" y="79" text-anchor="middle" class="fig-bold">+</text>
<rect x="136" y="40" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="147" y="55.5" text-anchor="middle" class="fig-small">0</text>
<rect x="158" y="40" width="22" height="22" class="fill-soft-b stroke-grid"/>
<text x="169" y="55.5" text-anchor="middle" class="fig-small">1</text>
<rect x="180" y="40" width="22" height="22" class="fill-soft-b stroke-grid"/>
<text x="191" y="55.5" text-anchor="middle" class="fig-small">1</text>
<rect x="136" y="62" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="147" y="77.5" text-anchor="middle" class="fig-small">0</text>
<rect x="158" y="62" width="22" height="22" class="fill-soft-b stroke-grid"/>
<text x="169" y="77.5" text-anchor="middle" class="fig-small">1</text>
<rect x="180" y="62" width="22" height="22" class="fill-soft-b stroke-grid"/>
<text x="191" y="77.5" text-anchor="middle" class="fig-small">1</text>
<rect x="136" y="84" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="147" y="99.5" text-anchor="middle" class="fig-small">0</text>
<rect x="158" y="84" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="169" y="99.5" text-anchor="middle" class="fig-small">0</text>
<rect x="180" y="84" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="191" y="99.5" text-anchor="middle" class="fig-small">0</text>
<text x="169" y="122" text-anchor="middle" class="fig-small">window 2</text>
<text x="222" y="79" text-anchor="middle" class="fig-bold">+</text>
<rect x="242" y="40" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="253" y="55.5" text-anchor="middle" class="fig-small">0</text>
<rect x="264" y="40" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="275" y="55.5" text-anchor="middle" class="fig-small">0</text>
<rect x="286" y="40" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="297" y="55.5" text-anchor="middle" class="fig-small">0</text>
<rect x="242" y="62" width="22" height="22" class="fill-soft-c stroke-grid"/>
<text x="253" y="77.5" text-anchor="middle" class="fig-small">1</text>
<rect x="264" y="62" width="22" height="22" class="fill-soft-c stroke-grid"/>
<text x="275" y="77.5" text-anchor="middle" class="fig-small">1</text>
<rect x="286" y="62" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="297" y="77.5" text-anchor="middle" class="fig-small">0</text>
<rect x="242" y="84" width="22" height="22" class="fill-soft-c stroke-grid"/>
<text x="253" y="99.5" text-anchor="middle" class="fig-small">1</text>
<rect x="264" y="84" width="22" height="22" class="fill-soft-c stroke-grid"/>
<text x="275" y="99.5" text-anchor="middle" class="fig-small">1</text>
<rect x="286" y="84" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="297" y="99.5" text-anchor="middle" class="fig-small">0</text>
<text x="275" y="122" text-anchor="middle" class="fig-small">window 3</text>
<text x="328" y="79" text-anchor="middle" class="fig-bold">+</text>
<rect x="348" y="40" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="359" y="55.5" text-anchor="middle" class="fig-small">0</text>
<rect x="370" y="40" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="381" y="55.5" text-anchor="middle" class="fig-small">0</text>
<rect x="392" y="40" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="403" y="55.5" text-anchor="middle" class="fig-small">0</text>
<rect x="348" y="62" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="359" y="77.5" text-anchor="middle" class="fig-small">0</text>
<rect x="370" y="62" width="22" height="22" class="fill-soft-d stroke-grid"/>
<text x="381" y="77.5" text-anchor="middle" class="fig-small">1</text>
<rect x="392" y="62" width="22" height="22" class="fill-soft-d stroke-grid"/>
<text x="403" y="77.5" text-anchor="middle" class="fig-small">1</text>
<rect x="348" y="84" width="22" height="22" class="fill-paper stroke-grid"/>
<text x="359" y="99.5" text-anchor="middle" class="fig-small">0</text>
<rect x="370" y="84" width="22" height="22" class="fill-soft-d stroke-grid"/>
<text x="381" y="99.5" text-anchor="middle" class="fig-small">1</text>
<rect x="392" y="84" width="22" height="22" class="fill-soft-d stroke-grid"/>
<text x="403" y="99.5" text-anchor="middle" class="fig-small">1</text>
<text x="381" y="122" text-anchor="middle" class="fig-small">window 4</text>
<text x="30" y="160" class="fig-bold">Add the four masks: how many windows cover each pixel</text>
<rect x="30" y="176" width="56" height="56" class="fill-soft-a stroke-grid"/>
<text x="58" y="208.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="86" y="176" width="56" height="56" class="fill-soft-c stroke-grid"/>
<text x="114" y="208.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="142" y="176" width="56" height="56" class="fill-soft-a stroke-grid"/>
<text x="170" y="208.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="30" y="232" width="56" height="56" class="fill-soft-c stroke-grid"/>
<text x="58" y="264.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="86" y="232" width="56" height="56" class="fill-soft-b stroke-grid"/>
<text x="114" y="264.5" text-anchor="middle" class="fig-bold">4</text>
<rect x="142" y="232" width="56" height="56" class="fill-soft-c stroke-grid"/>
<text x="170" y="264.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="30" y="288" width="56" height="56" class="fill-soft-a stroke-grid"/>
<text x="58" y="320.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="86" y="288" width="56" height="56" class="fill-soft-c stroke-grid"/>
<text x="114" y="320.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="142" y="288" width="56" height="56" class="fill-soft-a stroke-grid"/>
<text x="170" y="320.5" text-anchor="middle" class="fig-bold">1</text>
<text x="230" y="200">Corner pixel: 1 window, so its gradient has 1 term.</text>
<text x="230" y="224">Edge pixel: 2 windows, so 2 terms to add.</text>
<text x="230" y="248">Middle pixel: 4 windows, so 4 terms to add.</text>
<text x="230" y="280" class="fig-small">Pixel 5 sits in every window, in a different</text>
<text x="230" y="296" class="fig-small">kernel tap each time (4, 3, 2, 1 for the weights</text>
<text x="230" y="312" class="fig-small">(1, 2, 3, 4)), so its gradient is 4 + 3 + 2 + 1 = 10.</text>
</svg>
Figure 1. Four windows tile a 3 by 3 image with overlap. The counts 1, 2, 1 / 2, 4, 2 / 1, 2, 1 say how many windows read each pixel. A pixel that many windows read receives gradient from each of them, so its gradient is a sum with that many terms.
```

If every kernel weight were 1, $dx$ would equal the coverage counts in Figure 1: 1, 2, 1 on the first row, 2, 4, 2 on the second, and 1, 2, 1 on the third. With real weights, each count becomes a weighted sum.

## One rule in matrix form

The two sums are what the product rule gives for $Y_{\text{mat}} = W_{\text{mat}}\,\text{col} + b$. Let $dY_{\text{mat}}$ be $dY$ reshaped to `(N, C_out, L)`, where $L = H_{\text{out}} W_{\text{out}}$. Then

$$
db = \sum_{n,\,l} dY_{\text{mat}}, \qquad
dW_{\text{mat}} = \sum_{n} dY_{\text{mat}}\;\text{col}^\top, \qquad
d\,\text{col} = W_{\text{mat}}^\top\, dY_{\text{mat}}
$$

Read it aloud. The bias gradient adds $dY$ over images and positions. The weight gradient multiplies the upstream gradient at each position by the patch that was there, which is the matrix product with $\text{col}^\top$, and adds over images. The gradient of the columns sends each output's blame back through the kernel that made it. In numpy, they are three lines:

```python
db = dY_mat.sum(axis=(0, 2))
dw_mat = np.einsum("nol,ncl->oc", dY_mat, col)
dcol = np.einsum("oc,nol->ncl", w_mat, dY_mat)
```

The weight gradient sums over `n` and `l` because those letters are missing from the output of the einsum string, which is the "shared weight" rule written as notation. Reshape `dw_mat` to the shape of `w`. Figure 2 shows how the three products connect.

```figure
<svg viewBox="0 0 600 352" role="img" aria-label="Flow of the backward pass: dY splits into db, dw, and dcol, and dcol returns to image shape through col2im">
<defs>
<marker id="c080-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker>
</defs>
<rect x="4" y="142" width="150" height="56" class="fill-soft-a stroke-a"/>
<text x="14" y="162" class="fig-bold">dY</text>
<text x="14" y="180" class="fig-small">(N, C_out, H_out, W_out)</text>
<rect x="4" y="222" width="150" height="40" class="fill-paper stroke-grid"/>
<text x="14" y="242" class="fig-bold">dY_mat</text>
<text x="14" y="260" class="fig-small">(N, C_out, L)</text>
<line x1="79" y1="198" x2="79" y2="222" class="stroke-ink" marker-end="url(#c080-arrow)"/>
<text x="86" y="214" class="fig-small">reshape</text>
<rect x="190" y="30" width="200" height="56" class="fill-soft-c stroke-c"/>
<text x="200" y="50" class="fig-bold">db = sum of dY_mat</text>
<text x="200" y="68" class="fig-small">over n and l: shape (C_out,)</text>
<rect x="190" y="142" width="200" height="56" class="fill-soft-b stroke-b"/>
<text x="200" y="162" class="fig-bold">dw_mat = dY_mat · colᵀ</text>
<text x="200" y="180" class="fig-small">summed over n: (C_out, C_in·k·k)</text>
<rect x="190" y="254" width="200" height="56" class="fill-soft-d stroke-d"/>
<text x="200" y="274" class="fig-bold">dcol = W_matᵀ · dY_mat</text>
<text x="200" y="292" class="fig-small">shape (N, C_in·k·k, L)</text>
<line x1="154" y1="242" x2="188" y2="58" class="stroke-ink" marker-end="url(#c080-arrow)"/>
<line x1="154" y1="242" x2="188" y2="170" class="stroke-ink" marker-end="url(#c080-arrow)"/>
<line x1="154" y1="242" x2="188" y2="282" class="stroke-ink" marker-end="url(#c080-arrow)"/>
<rect x="436" y="30" width="160" height="56" class="fill-paper stroke-grid"/>
<text x="446" y="50" class="fig-bold">db</text>
<text x="446" y="68" class="fig-small">(C_out,)</text>
<rect x="436" y="142" width="160" height="56" class="fill-paper stroke-grid"/>
<text x="446" y="162" class="fig-bold">dw</text>
<text x="446" y="180" class="fig-small">reshape to (C_out, C_in, k, k)</text>
<rect x="436" y="254" width="160" height="56" class="fill-paper stroke-grid"/>
<text x="446" y="274" class="fig-bold">dx</text>
<text x="446" y="292" class="fig-small">(N, C_in, H, W)</text>
<line x1="390" y1="58" x2="434" y2="58" class="stroke-ink" marker-end="url(#c080-arrow)"/>
<line x1="390" y1="170" x2="434" y2="170" class="stroke-ink" marker-end="url(#c080-arrow)"/>
<line x1="390" y1="282" x2="434" y2="282" class="stroke-ink" marker-end="url(#c080-arrow)"/>
<text x="392" y="276" class="fig-small fig-bold">col2im</text>
<text x="436" y="328" class="fig-small">col2im adds with +=</text>
<text x="190" y="216" class="fig-small">needs col, saved from the forward pass</text>
<text x="190" y="328" class="fig-small">needs the flattened weights W_mat</text>
</svg>
Figure 2. The backward pass of a convolution. The incoming gradient dY (reshaped to dY_mat) feeds three products. Two of them (db and dw) add over the batch and over positions, because weights and biases are shared. The third (dcol) lives in the column layout and returns to image shape through col2im, which adds overlapping entries.
```

The third line leaves one job. $d\,\text{col}$ is the gradient for every copy of every pixel, laid out like `col`, and you need the gradient for each pixel itself. During the forward pass, `im2col` copied each pixel into every window that contains it. Copying forward means adding backward: a pixel that was copied into four columns gets the sum of the four gradients. That step is **col2im**. It takes each kernel offset's slice of $d\,\text{col}$ and adds it into the matching strided slice of a zero image:

```python
dx_padded[:, :, dh:dh + stride * h_out:stride, dw:dw + stride * w_out:stride] += dcol_r[:, :, dh, dw]
```

Figure 3 shows the result on the chapter's example. With $dY$ all ones, each window sends the kernel's own numbers back, and adding the four windows into place produces $dx$ above.

```figure
<svg viewBox="0 0 600 384" role="img" aria-label="Four 2 by 2 patches added into a 3 by 3 canvas with plus-equals, beside the wrong result of overwriting">
<text x="20" y="22" class="fig-bold">With dY all ones, each window sends the kernel's own numbers back</text>
<rect x="30" y="42" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="45" y="61.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="60" y="42" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="75" y="61.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="30" y="72" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="45" y="91.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="60" y="72" width="30" height="30" class="fill-soft-a stroke-grid"/>
<text x="75" y="91.5" text-anchor="middle" class="fig-bold">4</text>
<text x="60" y="118" text-anchor="middle" class="fig-small">window 1</text>
<text x="60" y="132" text-anchor="middle" class="fig-small">corner (0, 0)</text>
<rect x="160" y="42" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="175" y="61.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="190" y="42" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="205" y="61.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="160" y="72" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="175" y="91.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="190" y="72" width="30" height="30" class="fill-soft-b stroke-grid"/>
<text x="205" y="91.5" text-anchor="middle" class="fig-bold">4</text>
<text x="190" y="118" text-anchor="middle" class="fig-small">window 2</text>
<text x="190" y="132" text-anchor="middle" class="fig-small">corner (0, 1)</text>
<rect x="290" y="42" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="305" y="61.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="320" y="42" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="335" y="61.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="290" y="72" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="305" y="91.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="320" y="72" width="30" height="30" class="fill-soft-c stroke-grid"/>
<text x="335" y="91.5" text-anchor="middle" class="fig-bold">4</text>
<text x="320" y="118" text-anchor="middle" class="fig-small">window 3</text>
<text x="320" y="132" text-anchor="middle" class="fig-small">corner (1, 0)</text>
<rect x="420" y="42" width="30" height="30" class="fill-soft-d stroke-grid"/>
<text x="435" y="61.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="450" y="42" width="30" height="30" class="fill-soft-d stroke-grid"/>
<text x="465" y="61.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="420" y="72" width="30" height="30" class="fill-soft-d stroke-grid"/>
<text x="435" y="91.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="450" y="72" width="30" height="30" class="fill-soft-d stroke-grid"/>
<text x="465" y="91.5" text-anchor="middle" class="fig-bold">4</text>
<text x="450" y="118" text-anchor="middle" class="fig-small">window 4</text>
<text x="450" y="132" text-anchor="middle" class="fig-small">corner (1, 1)</text>
<text x="30" y="178" class="fig-bold">Add them into place (+=)</text>
<rect x="30" y="194" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="56" y="224.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="82" y="194" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="108" y="224.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="134" y="194" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="160" y="224.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="30" y="246" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="56" y="276.5" text-anchor="middle" class="fig-bold">4</text>
<rect x="82" y="246" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="108" y="276.5" text-anchor="middle" class="fig-bold">10</text>
<rect x="134" y="246" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="160" y="276.5" text-anchor="middle" class="fig-bold">6</text>
<rect x="30" y="298" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="56" y="328.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="82" y="298" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="108" y="328.5" text-anchor="middle" class="fig-bold">7</text>
<rect x="134" y="298" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="160" y="328.5" text-anchor="middle" class="fig-bold">4</text>
<text x="30" y="368" class="fig-small">this is dx</text>
<text x="330" y="178" class="fig-bold">Overwrite instead (=): last window wins</text>
<rect x="330" y="194" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="356" y="224.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="382" y="194" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="408" y="224.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="434" y="194" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="460" y="224.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="330" y="246" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="356" y="276.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="382" y="246" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="408" y="276.5" text-anchor="middle" class="fig-bold">1</text>
<rect x="434" y="246" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="460" y="276.5" text-anchor="middle" class="fig-bold">2</text>
<rect x="330" y="298" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="356" y="328.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="382" y="298" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="408" y="328.5" text-anchor="middle" class="fig-bold">3</text>
<rect x="434" y="298" width="52" height="52" class="fill-paper stroke-grid"/>
<text x="460" y="328.5" text-anchor="middle" class="fig-bold">4</text>
<rect x="382" y="194" width="52" height="52" class="stroke-bad"/>
<rect x="330" y="246" width="52" height="52" class="stroke-bad"/>
<rect x="382" y="246" width="52" height="52" class="stroke-bad"/>
<rect x="434" y="246" width="52" height="52" class="stroke-bad"/>
<rect x="382" y="298" width="52" height="52" class="stroke-bad"/>
<text x="330" y="368" class="fig-small">red cells are wrong: overlaps lost</text>
</svg>
Figure 3. col2im on the chapter's example. Adding the four patches gives dx = (1, 3, 2 / 4, 10, 6 / 3, 7, 4). Overwriting keeps only the last patch that touched each pixel, so the five pixels read by two or more windows are wrong (red). Notice that the four corner pixels, each read by one window, survive either way.
```

The `+=` is essential. Writing `=` keeps only the last window that touched each pixel, and the red cells in Figure 3 are wrong for exactly that reason. After the loop, cut the padding off the canvas. When the padding is 0, do not write `canvas[:, :, 0:-0]`, because `0:-0` is an empty slice. Return the whole canvas instead.

Mathematicians call this relationship the **adjoint**, or transpose: `im2col` is a linear map from images to columns, and `col2im` is the matrix transpose of that map. The defining test is that moving the map across a dot product changes nothing:

$$
\langle \text{im2col}(x),\, c\rangle = \langle x,\, \text{col2im}(c)\rangle
$$

Read it aloud: take any image $x$ and any array $c$ shaped like `col`. Copy the image into columns and take the dot product with $c$, and you get the same number as taking the dot product of $x$ with the columns scattered back. The exercise tests this identity directly, with random $x$ and $c$, which catches a wrong slice, a wrong stride, or a mishandled padding. It is not the same as an inverse: `col2im(im2col(x))` multiplies each pixel by its coverage count instead of returning $x$.

> [!KEY]
> A shared weight and bias collect gradient from every position and image, so $dW$ and $db$ are sums. A pixel read by several windows collects gradient from each, so $dx$ is a scatter-add of $d\,\text{col}$. Both sums come from one fact: the forward pass reused the same number in many places.

## Trust the gradient only after you check it

A hand derivation can be wrong in a way that still looks right: a missed sum, a transposed axis. The check from "Putting it together: gradients for a whole network" works here. Pick a scalar loss, $L = \sum dY \odot Y$, which has $dL/dY = dY$ by construction. Nudge each entry of $x$, $w$, or $b$ by $\pm\epsilon$, rerun the forward pass, and compare $(L_+ - L_-)/2\epsilon$ with your analytic gradient. That needs one forward pass per entry, which is far too slow for training and fast enough for a 5 by 5 image. The exercise does this for all three gradients, with a stride of 2 and padding of 1 so that overlaps, skipped pixels, and the padding cut all get tested.

> [!RECAP]
> Hold these points before you write code:
>
> - $db$ sums $dY$ over the batch and all positions.
> - $dW_{\text{mat}}$ is $dY_{\text{mat}}\,\text{col}^\top$ summed over the batch, reshaped to the shape of $w$.
> - $d\,\text{col} = W_{\text{mat}}^\top\, dY_{\text{mat}}$, and `col2im` adds each entry back to the pixel it was copied from with `+=`.
> - `col2im` is the transpose of `im2col`, so $\langle\text{im2col}(x), c\rangle = \langle x, \text{col2im}(c)\rangle$.
> - The number of windows covering a pixel is `col2im` applied to ones.

## Build it step by step

The checks call the functions in this order. The starter already contains `conv_output_shape`, `im2col`, and `conv2d_forward`, solved, from the earlier chapters.

1. **`col2im(dcol, x_shape, kernel_size, stride, padding)`.** `dcol` has the shape that `im2col` returns, and `x_shape` is `(N, C, H, W)`. Reshape `dcol` to `(N, C, k, k, H_out, W_out)`, add each offset's slice into a zero canvas of the padded size with `+=`, then cut off the padding. The most common mistake is `=` instead of `+=`, or `padding:-padding` when the padding is 0.
2. **`window_coverage(x_shape, kernel_size, stride, padding)`.** Return an `(H, W)` array with the number of windows that read each pixel. Apply `col2im` to an array of ones with the shape `im2col` would give for a single one-channel image. The most common mistake is returning the padded size, `(H + 2 * padding, W + 2 * padding)`, because the cut-off in `col2im` already removes the padding.
3. **`conv2d_backward(dY, x, w, b, stride, padding)`.** Return `(dx, dw, db)` with the shapes of `x`, `w`, and `b`. Rebuild `col` with `im2col`, form the three gradients above, and call `col2im` for `dx`. The most common mistake is to skip the sum over the batch in `dw`.

## Where this breaks

- A stride larger than the kernel leaves some pixels in no window. Their coverage is 0, so their gradient is 0: the network cannot learn from pixels it never reads. Try `window_coverage((1, 1, 2, 6), 2, 3, 0)`.
- Padding cells take part in the forward pass, but they are not pixels. Their gradient is computed and then thrown away by the cut-off.
- `dcol` is as large as `col`, about $k^2$ times the image. The backward pass needs that memory at the same time as the forward pass's saved `col`, which is why large networks recompute it or build it in pieces.
- Convolutions are equivariant to shifts, as "Sliding one small filter over a picture" showed, but not to rotations or changes of scale. A backward pass that is correct cannot fix a model that has no way to represent those changes.

## The exercise

You implement `col2im`, `window_coverage`, and `conv2d_backward` in `starter.py`. The checks confirm that `col2im` adds overlaps and satisfies the adjoint identity at several strides and paddings, that `window_coverage` returns 1, 2, 1 / 2, 4, 2 / 1, 2, 1 on the example and 0 for pixels no window reads, and that `conv2d_backward` matches finite differences for `dx`, `dw`, and `db`. A last check reproduces the hand example of this chapter.
