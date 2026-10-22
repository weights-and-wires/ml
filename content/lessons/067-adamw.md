---
title: "AdamW: decaying weights the right way"
slug: 067-adamw
part: "Part 4 — Neural networks"
summary: Apply weight decay after Adam's division, not inside the gradient, so every weight shrinks by the same fraction.
minutes: 35
gpu: false
objectives:
  - Explain weight decay as a fixed fraction of each weight removed per step.
  - Show that an L2 penalty and decoupled decay are the same for plain SGD.
  - Explain why an L2 penalty inside Adam shrinks quiet weights too much and noisy weights too little.
  - Take an AdamW step, and measure how much shrink decay alone causes.
prereqs:
  - 013-ridge-penalty-and-closed-form
  - 065-stochastic-gradients
  - 066-adam
exercise: 067-adamw
---

# AdamW: decaying weights the right way

You train a model with Adam and the weights keep growing. A weight that grows large lets the model fit noise in the training data, so you reach for the standard remedy: charge a fee on large weights, an L2 penalty, and let the gradient carry it. The fee worked in plain gradient descent, so you add it to Adam's gradient and run.

Then you inspect the weights. One weight that always saw tiny gradients has collapsed to zero. Another that always saw huge gradients has barely moved. The penalty was the same for both, so why did Adam treat them so differently? Why does adding the penalty to the gradient not do what it did for SGD?

The answer is where the penalty enters Adam's pipeline. This chapter shows where it goes wrong, and the one-line change, called AdamW, that fixes it.

## What you already know

Five ideas carry this chapter:

- A ridge penalty charges a fee on the squared size of the weights, and its gradient pulls each weight toward zero (see "Ridge: a fee on large weights").
- An SGD step subtracts the learning rate times the gradient (see "Noisy gradients: stochastic gradient descent and mini-batches").
- Adam divides an average gradient $\hat{m}$ by a typical size $\sqrt{\hat{v}}$, so every weight moves about one learning rate per step (see "Adam: a step size for every weight").
- In this chapter a weight is **quiet** when its gradients are small, so its $\hat{v}$ is small, and **noisy** when its gradients are large and jumpy. Only the size of the gradient matters here.
- The state holds the raw averages $m$ and $v$, and the step count $t$.

## Shrink every weight by the same fraction

**Weight decay** is a simple rule: at each step, take a small fraction of every weight away. With learning rate $\eta$ and decay rate $\lambda$, each weight loses the fraction $\eta\lambda$ of itself:

$$
w \leftarrow w - \eta\,\lambda\,w
$$

Read it aloud: subtract the learning rate times the decay rate times the weight. Take $w = 1$, $\eta = 0.1$, and $\lambda = 0.01$. The weight loses $0.1 \times 0.01 \times 1 = 0.001$. A weight of 4 loses 0.004, so large weights lose more, and every weight loses the same fraction. That uniformity is the point: decay should not favour any weight.

Where does the rule come from? A penalty $\tfrac{\lambda}{2}\lVert w\rVert^2$ added to the loss has gradient $\lambda w$. Put that into a gradient step and you subtract $\eta\lambda w$. Adding the penalty to the loss and shrinking the weight by a fixed fraction are two views of one idea. A later chapter, "Weight decay: a pull toward zero", studies why the shrinking helps. Here the question is how to apply it.

For plain SGD the two views give the same number. Take $w = 1$, a gradient $g = 0.5$, $\eta = 0.1$, and $\lambda = 0.01$. With the penalty in the gradient, the gradient is $0.5 + 0.01 = 0.51$ and the weight becomes $1 - 0.051 = 0.949$. With decay applied separately, the plain step gives $1 - 0.1 \times 0.5 = 0.95$, and the decay removes $0.001$ more, so the weight is $0.949$. Same answer, because SGD does nothing to the gradient except scale it.

## See where Adam breaks the match

Adam does something to the gradient: it divides by $\sqrt{\hat{v}}$. Put the penalty into the gradient and the sum $g + \lambda w$ goes through the averages and then through that division. The penalty is measured in the weight's own gradient units. Figure 1 shows the pipeline.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Two update pipelines: where the decay term joins when it is added to the gradient and when it is applied after Adam">
<defs><marker id="c067-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c067-arrow-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<text x="300.0" y="20.0" class="fig-bold" text-anchor="middle">Adam with L2 in the gradient</text>
<rect x="2.0" y="34.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="2" y="34" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="54.0" y="65.0" class="fig-small" text-anchor="middle">gradient g</text>
<rect x="125.0" y="34.0" width="104.0" height="54.0" class="fill-soft-b"/>
<rect x="125" y="34" width="104" height="54" fill="none" class="stroke-bad"/>
<text x="177.0" y="57.0" class="fig-small" text-anchor="middle">add decay</text>
<text x="177.0" y="73.0" class="fig-small" text-anchor="middle">g + lambda w</text>
<rect x="248.0" y="34.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="248" y="34" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="300.0" y="57.0" class="fig-small" text-anchor="middle">averages</text>
<text x="300.0" y="73.0" class="fig-small" text-anchor="middle">m and v</text>
<rect x="371.0" y="34.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="371" y="34" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="423.0" y="57.0" class="fig-small" text-anchor="middle">divide</text>
<text x="423.0" y="73.0" class="fig-small" text-anchor="middle">m_hat / sqrt(v_hat)</text>
<rect x="494.0" y="34.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="494" y="34" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="546.0" y="65.0" class="fig-small" text-anchor="middle">new weights</text>
<line x1="106.0" y1="61.0" x2="125.0" y2="61.0" class="stroke-ink" marker-end="url(#c067-arrow-ink)"/>
<line x1="229.0" y1="61.0" x2="248.0" y2="61.0" class="stroke-ink" marker-end="url(#c067-arrow-ink)"/>
<line x1="352.0" y1="61.0" x2="371.0" y2="61.0" class="stroke-ink" marker-end="url(#c067-arrow-ink)"/>
<line x1="475.0" y1="61.0" x2="494.0" y2="61.0" class="stroke-ink" marker-end="url(#c067-arrow-ink)"/>
<text x="300.0" y="110.0" class="fig-small" text-anchor="middle">The decay term goes through the averages and gets divided by sqrt(v_hat)</text>
<text x="300.0" y="156.0" class="fig-bold" text-anchor="middle">AdamW</text>
<rect x="2.0" y="170.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="2" y="170" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="54.0" y="201.0" class="fig-small" text-anchor="middle">gradient g</text>
<rect x="125.0" y="170.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="125" y="170" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="177.0" y="193.0" class="fig-small" text-anchor="middle">averages</text>
<text x="177.0" y="209.0" class="fig-small" text-anchor="middle">m and v</text>
<rect x="248.0" y="170.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="248" y="170" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="300.0" y="193.0" class="fig-small" text-anchor="middle">divide</text>
<text x="300.0" y="209.0" class="fig-small" text-anchor="middle">m_hat / sqrt(v_hat)</text>
<rect x="371.0" y="170.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="371" y="170" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="423.0" y="193.0" class="fig-small" text-anchor="middle">subtract</text>
<text x="423.0" y="209.0" class="fig-small" text-anchor="middle">lr x step</text>
<rect x="494.0" y="170.0" width="104.0" height="54.0" class="fill-sunk"/>
<rect x="494" y="170" width="104" height="54" fill="none" class="stroke-muted"/>
<text x="546.0" y="201.0" class="fig-small" text-anchor="middle">new weights</text>
<line x1="106.0" y1="197.0" x2="125.0" y2="197.0" class="stroke-ink" marker-end="url(#c067-arrow-ink)"/>
<line x1="229.0" y1="197.0" x2="248.0" y2="197.0" class="stroke-ink" marker-end="url(#c067-arrow-ink)"/>
<line x1="352.0" y1="197.0" x2="371.0" y2="197.0" class="stroke-ink" marker-end="url(#c067-arrow-ink)"/>
<line x1="475.0" y1="197.0" x2="494.0" y2="197.0" class="stroke-ink" marker-end="url(#c067-arrow-ink)"/>
<rect x="331.0" y="260.0" width="184.0" height="40.0" class="fill-soft-a"/>
<rect x="331" y="260" width="184" height="40" fill="none" class="stroke-a"/>
<text x="423.0" y="284.0" class="fig-small" text-anchor="middle">decay: lr x lambda x w</text>
<line x1="423.0" y1="260.0" x2="423.0" y2="224.0" class="stroke-a" marker-end="url(#c067-arrow-a)"/>
<text x="300.0" y="322.0" class="fig-small" text-anchor="middle">The decay term skips the averages and the division, and joins at the end</text>
</svg>
Figure 1. Top: with L2 in the gradient, the decay term enters at the start, so Adam treats it as part of the gradient and divides it by the weight's typical gradient size. Bottom: AdamW keeps the decay out of the averages and subtracts lr x lambda x w at the end, using the weight from before the step.
```

Consider a quiet weight, with gradient noise of typical size 0.01, and a noisy weight, with typical size 10. Both equal 1. The pull $\lambda w = 0.01$ is as big as the quiet weight's whole gradient, so Adam sees a strong, steady push and takes a long step. For the noisy weight the same pull is a thousandth of its gradient size, so Adam sees almost nothing.

The numbers follow from one fact: Adam turns a push into a step of $\eta$ times the push, divided by the weight's typical gradient size. For the noisy weight that is $0.1 \times 0.01 / 10 = 0.0001$, ten times less than the intended 0.001. For the quiet weight, adding the pull makes the typical size about $\sqrt{0.01^2 + 0.01^2} = 0.014$, so the step is about $0.1 \times 0.01 / 0.014 = 0.07$, roughly seventy times more.

To measure this, freeze the weights at 1, feed Adam 200 noisy gradients of those two sizes, and compare the next step with decay and without it. The difference is how much shrink decay alone causes. With $\eta = 0.1$ and $\lambda = 0.01$ it should be 0.001 for both. Figure 2 shows what happens.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Shrink of two weights in one Adam step caused by decay alone, coupled to the gradient versus decoupled">
<defs></defs>
<line x1="80.0" y1="250.0" x2="580.0" y2="250.0" class="stroke-muted"/>
<line x1="80.0" y1="40.0" x2="80.0" y2="250.0" class="stroke-muted"/>
<line x1="80.0" y1="233.4" x2="580.0" y2="233.4" class="stroke-grid"/>
<line x1="75.0" y1="233.4" x2="80.0" y2="233.4" class="stroke-muted"/>
<text x="71.0" y="237.4" class="fig-small" text-anchor="end">1e-4</text>
<line x1="80.0" y1="178.2" x2="580.0" y2="178.2" class="stroke-grid"/>
<line x1="75.0" y1="178.2" x2="80.0" y2="178.2" class="stroke-muted"/>
<text x="71.0" y="182.2" class="fig-small" text-anchor="end">1e-3</text>
<line x1="80.0" y1="122.9" x2="580.0" y2="122.9" class="stroke-grid"/>
<line x1="75.0" y1="122.9" x2="80.0" y2="122.9" class="stroke-muted"/>
<text x="71.0" y="126.9" class="fig-small" text-anchor="end">1e-2</text>
<line x1="80.0" y1="67.6" x2="580.0" y2="67.6" class="stroke-grid"/>
<line x1="75.0" y1="67.6" x2="80.0" y2="67.6" class="stroke-muted"/>
<text x="71.0" y="71.6" class="fig-small" text-anchor="end">1e-1</text>
<text x="80.0" y="30.0" class="fig-small" text-anchor="start">shrink caused by decay (log scale)</text>
<rect x="94.5" y="75.0" width="96.0" height="175.0" class="fill-b"/>
<text x="142.5" y="69.0" class="fig-small" text-anchor="middle">0.0735</text>
<rect x="219.5" y="233.5" width="96.0" height="16.5" class="fill-b"/>
<text x="267.5" y="227.5" class="fig-small" text-anchor="middle">0.00010</text>
<rect x="344.5" y="178.2" width="96.0" height="71.8" class="fill-a"/>
<text x="392.5" y="172.2" class="fig-small" text-anchor="middle">0.0010</text>
<rect x="469.5" y="178.2" width="96.0" height="71.8" class="fill-a"/>
<text x="517.5" y="172.2" class="fig-small" text-anchor="middle">0.0010</text>
<text x="142.5" y="268.0" class="fig-small" text-anchor="middle">quiet weight</text>
<text x="267.5" y="268.0" class="fig-small" text-anchor="middle">noisy weight</text>
<text x="392.5" y="268.0" class="fig-small" text-anchor="middle">quiet weight</text>
<text x="517.5" y="268.0" class="fig-small" text-anchor="middle">noisy weight</text>
<text x="205.0" y="288.0" class="fig-bold" text-anchor="middle">Adam with L2 in the gradient</text>
<text x="455.0" y="288.0" class="fig-bold" text-anchor="middle">AdamW</text>
<line x1="80.0" y1="178.2" x2="580.0" y2="178.2" class="stroke-muted dashed"/>
<text x="267.5" y="172.2" class="fig-small" text-anchor="middle">intended: 0.001</text>
</svg>
Figure 2. Both weights equal 1, the learning rate is 0.1, and the decay is 0.01, so decay should shrink each weight by 0.001 (dashed line). The quiet weight sees gradients of typical size 0.01 and the noisy one sees size 10. With L2 inside Adam, the quiet weight shrinks by 0.074, about 74 times too much, and the noisy weight by 0.00010, about ten times too little. AdamW shrinks both by exactly 0.001.
```

Why not just raise or lower $\lambda$ until the average looks right? One number cannot fix two weights that are off by factors of 74 and 10 in opposite directions. The error depends on each weight's gradient size, which changes during training.

> [!KEY]
> Adam divides everything in the gradient by the weight's typical gradient size, including the decay pull. So an L2 penalty inside Adam shrinks quiet weights too much and noisy weights too little.

## Apply the decay after Adam

The fix keeps the decay out of the pipeline. Run Adam on the raw gradient, so the averages see only the loss. Then subtract the decay from the old weight:

$$
w_{\text{new}} = w_{\text{Adam}} - \eta\,\lambda\,w
$$

Read it aloud: take Adam's result and remove the learning rate times the decay rate times the weight you started the step with. This is **AdamW**, where the W stands for weight decay. Every weight loses the fraction $\eta\lambda$, as the opening rule said. With $\lambda = 0$ it is Adam. In numpy:

```python
new_w, new_state = adam_step(w, grad, state, lr, beta1, beta2, eps)
new_w = new_w - lr * weight_decay * w
```

Run the frozen test again on AdamW. Both weights shrink by exactly 0.001, whatever their gradient sizes. The averages never see the decay, so the state is the same as Adam's.

Figure 3 shows the difference over time. Two weights start at 1 and see only noise, one quiet and one noisy, with learning rate 0.01 and decay 1. Decay alone would take a weight from 1 to $0.99^{300} = 0.05$ in 300 steps. Under L2 in the gradient, the quiet weight falls to zero long before that, while the noisy one has only dropped to 0.60. Under AdamW, both fade at about the same pace.

```figure
<svg viewBox="0 0 600 350" role="img" aria-label="Two weights with pure noise gradients under Adam with L2, and under AdamW">
<defs></defs>
<line x1="55.0" y1="250.0" x2="290.0" y2="250.0" class="stroke-muted"/>
<line x1="55.0" y1="40.0" x2="55.0" y2="250.0" class="stroke-muted"/>
<line x1="55.0" y1="250.0" x2="55.0" y2="255.0" class="stroke-muted"/>
<text x="55.0" y="269.0" class="fig-small" text-anchor="middle">0</text>
<line x1="133.3" y1="250.0" x2="133.3" y2="255.0" class="stroke-muted"/>
<text x="133.3" y="269.0" class="fig-small" text-anchor="middle">100</text>
<line x1="211.7" y1="250.0" x2="211.7" y2="255.0" class="stroke-muted"/>
<text x="211.7" y="269.0" class="fig-small" text-anchor="middle">200</text>
<line x1="290.0" y1="250.0" x2="290.0" y2="255.0" class="stroke-muted"/>
<text x="290.0" y="269.0" class="fig-small" text-anchor="middle">300</text>
<line x1="55.0" y1="222.0" x2="290.0" y2="222.0" class="stroke-grid"/>
<line x1="50.0" y1="222.0" x2="55.0" y2="222.0" class="stroke-muted"/>
<text x="46.0" y="226.0" class="fig-small" text-anchor="end">0</text>
<line x1="55.0" y1="152.0" x2="290.0" y2="152.0" class="stroke-grid"/>
<line x1="50.0" y1="152.0" x2="55.0" y2="152.0" class="stroke-muted"/>
<text x="46.0" y="156.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="55.0" y1="82.0" x2="290.0" y2="82.0" class="stroke-grid"/>
<line x1="50.0" y1="82.0" x2="55.0" y2="82.0" class="stroke-muted"/>
<text x="46.0" y="86.0" class="fig-small" text-anchor="end">1</text>
<text x="172.5" y="288.0" class="fig-small" text-anchor="middle">step</text>
<polyline points="55.0,82.0 55.8,83.4 56.6,84.8 57.4,86.2 58.1,87.6 58.9,89.0 59.7,90.4 60.5,91.8 61.3,93.2 62.0,94.6 62.8,95.9 63.6,97.3 64.4,98.7 65.2,100.1 66.0,101.5 66.8,102.8 67.5,104.2 68.3,105.6 69.1,106.9 69.9,108.3 70.7,109.6 71.5,111.0 72.2,112.3 73.0,113.6 73.8,115.0 74.6,116.3 75.4,117.6 76.2,118.9 76.9,120.2 77.7,121.5 78.5,122.8 79.3,124.1 80.1,125.4 80.8,126.6 81.6,127.9 82.4,129.1 83.2,130.4 84.0,131.6 84.8,132.9 85.5,134.1 86.3,135.3 87.1,136.5 87.9,137.7 88.7,138.9 89.5,140.1 90.2,141.2 91.0,142.4 91.8,143.5 92.6,144.7 93.4,145.8 94.2,146.9 95.0,148.0 95.7,149.1 96.5,150.2 97.3,151.3 98.1,152.4 98.9,153.4 99.7,154.5 100.4,155.5 101.2,156.6 102.0,157.6 102.8,158.6 103.6,159.6 104.3,160.6 105.1,161.6 105.9,162.6 106.7,163.5 107.5,164.5 108.3,165.4 109.1,166.4 109.8,167.3 110.6,168.2 111.4,169.1 112.2,170.0 113.0,170.9 113.8,171.8 114.5,172.6 115.3,173.5 116.1,174.3 116.9,175.2 117.7,176.0 118.5,176.8 119.2,177.6 120.0,178.4 120.8,179.2 121.6,180.0 122.4,180.8 123.1,181.5 123.9,182.3 124.7,183.0 125.5,183.8 126.3,184.5 127.1,185.2 127.8,185.9 128.6,186.6 129.4,187.3 130.2,188.0 131.0,188.6 131.8,189.3 132.6,189.9 133.3,190.6 134.1,191.2 134.9,191.8 135.7,192.4 136.5,193.0 137.2,193.6 138.0,194.2 138.8,194.7 139.6,195.3 140.4,195.8 141.2,196.4 141.9,196.9 142.7,197.5 143.5,198.0 144.3,198.5 145.1,199.0 145.9,199.5 146.7,200.0 147.4,200.4 148.2,200.9 149.0,201.4 149.8,201.8 150.6,202.3 151.3,202.7 152.1,203.1 152.9,203.6 153.7,204.0 154.5,204.4 155.3,204.8 156.1,205.2 156.8,205.6 157.6,206.0 158.4,206.4 159.2,206.7 160.0,207.1 160.8,207.4 161.5,207.8 162.3,208.1 163.1,208.5 163.9,208.8 164.7,209.1 165.4,209.4 166.2,209.7 167.0,210.0 167.8,210.3 168.6,210.6 169.4,210.9 170.1,211.1 170.9,211.4 171.7,211.7 172.5,211.9 173.3,212.2 174.1,212.4 174.9,212.7 175.6,212.9 176.4,213.1 177.2,213.4 178.0,213.6 178.8,213.8 179.6,214.0 180.3,214.2 181.1,214.5 181.9,214.7 182.7,214.9 183.5,215.1 184.2,215.2 185.0,215.4 185.8,215.6 186.6,215.8 187.4,216.0 188.2,216.2 188.9,216.3 189.7,216.5 190.5,216.7 191.3,216.8 192.1,217.0 192.9,217.2 193.7,217.3 194.4,217.4 195.2,217.6 196.0,217.7 196.8,217.9 197.6,218.0 198.3,218.1 199.1,218.2 199.9,218.4 200.7,218.5 201.5,218.6 202.3,218.7 203.1,218.8 203.8,218.9 204.6,219.0 205.4,219.1 206.2,219.2 207.0,219.3 207.8,219.3 208.5,219.4 209.3,219.5 210.1,219.6 210.9,219.7 211.7,219.7 212.5,219.8 213.2,219.9 214.0,219.9 214.8,220.0 215.6,220.1 216.4,220.1 217.1,220.2 217.9,220.2 218.7,220.3 219.5,220.3 220.3,220.4 221.1,220.4 221.8,220.5 222.6,220.5 223.4,220.6 224.2,220.6 225.0,220.7 225.8,220.7 226.5,220.8 227.3,220.8 228.1,220.9 228.9,220.9 229.7,221.0 230.5,221.0 231.2,221.1 232.0,221.1 232.8,221.2 233.6,221.2 234.4,221.2 235.2,221.3 236.0,221.3 236.7,221.3 237.5,221.3 238.3,221.4 239.1,221.4 239.9,221.4 240.7,221.4 241.4,221.4 242.2,221.4 243.0,221.4 243.8,221.4 244.6,221.4 245.4,221.4 246.1,221.4 246.9,221.4 247.7,221.4 248.5,221.4 249.3,221.5 250.0,221.5 250.8,221.5 251.6,221.5 252.4,221.5 253.2,221.5 254.0,221.5 254.8,221.5 255.5,221.5 256.3,221.5 257.1,221.6 257.9,221.6 258.7,221.6 259.4,221.6 260.2,221.6 261.0,221.6 261.8,221.7 262.6,221.7 263.4,221.7 264.1,221.7 264.9,221.7 265.7,221.8 266.5,221.8 267.3,221.8 268.1,221.8 268.9,221.8 269.6,221.8 270.4,221.8 271.2,221.9 272.0,221.9 272.8,221.9 273.6,221.9 274.3,221.9 275.1,221.9 275.9,221.9 276.7,221.9 277.5,221.9 278.2,222.0 279.0,222.0 279.8,222.0 280.6,222.0 281.4,222.0 282.2,222.0 282.9,222.1 283.7,222.1 284.5,222.1 285.3,222.1 286.1,222.1 286.9,222.1 287.6,222.1 288.4,222.1 289.2,222.1 290.0,222.1" fill="none" class="stroke-a"/>
<polyline points="55.0,82.0 55.8,80.6 56.6,81.5 57.4,82.6 58.1,83.7 58.9,83.8 59.7,84.0 60.5,84.1 61.3,83.9 62.0,83.7 62.8,83.9 63.6,84.4 64.4,85.0 65.2,85.6 66.0,85.9 66.8,86.2 67.5,86.5 68.3,86.9 69.1,87.4 69.9,87.8 70.7,88.5 71.5,89.3 72.2,90.3 73.0,91.1 73.8,92.1 74.6,93.2 75.4,94.0 76.2,94.8 76.9,95.6 77.7,96.5 78.5,97.2 79.3,97.7 80.1,98.0 80.8,98.3 81.6,98.8 82.4,98.9 83.2,99.0 84.0,99.1 84.8,99.0 85.5,99.1 86.3,99.4 87.1,99.7 87.9,99.7 88.7,99.9 89.5,100.3 90.2,100.8 91.0,101.5 91.8,102.1 92.6,102.7 93.4,103.2 94.2,103.5 95.0,103.9 95.7,104.2 96.5,104.2 97.3,104.3 98.1,104.5 98.9,104.7 99.7,104.9 100.4,105.0 101.2,105.3 102.0,105.6 102.8,106.1 103.6,106.3 104.3,106.4 105.1,106.5 105.9,106.5 106.7,106.6 107.5,106.8 108.3,107.2 109.1,107.7 109.8,108.3 110.6,109.0 111.4,109.6 112.2,110.2 113.0,110.7 113.8,111.3 114.5,111.8 115.3,112.3 116.1,112.5 116.9,112.7 117.7,112.5 118.5,112.3 119.2,112.2 120.0,112.0 120.8,111.7 121.6,111.5 122.4,111.3 123.1,111.3 123.9,111.2 124.7,111.3 125.5,111.4 126.3,111.3 127.1,111.2 127.8,111.2 128.6,111.5 129.4,111.6 130.2,111.5 131.0,111.5 131.8,111.7 132.6,111.8 133.3,112.0 134.1,112.1 134.9,112.4 135.7,112.5 136.5,112.4 137.2,112.6 138.0,113.1 138.8,113.6 139.6,114.2 140.4,114.6 141.2,115.4 141.9,115.9 142.7,116.4 143.5,117.0 144.3,117.6 145.1,118.0 145.9,118.2 146.7,118.6 147.4,118.8 148.2,118.8 149.0,118.7 149.8,118.6 150.6,118.4 151.3,118.3 152.1,118.5 152.9,118.6 153.7,118.7 154.5,118.6 155.3,118.7 156.1,118.8 156.8,119.2 157.6,119.5 158.4,119.8 159.2,120.1 160.0,120.4 160.8,120.6 161.5,120.9 162.3,121.0 163.1,121.1 163.9,121.0 164.7,121.2 165.4,121.4 166.2,121.4 167.0,121.5 167.8,121.5 168.6,121.5 169.4,121.4 170.1,121.4 170.9,121.5 171.7,121.6 172.5,121.8 173.3,122.1 174.1,121.9 174.9,121.7 175.6,121.5 176.4,121.3 177.2,121.1 178.0,120.8 178.8,120.6 179.6,120.3 180.3,120.2 181.1,120.0 181.9,119.9 182.7,119.8 183.5,119.6 184.2,119.5 185.0,119.2 185.8,118.8 186.6,118.5 187.4,118.1 188.2,117.5 188.9,117.0 189.7,116.6 190.5,116.0 191.3,115.5 192.1,115.2 192.9,115.2 193.7,115.0 194.4,114.9 195.2,114.9 196.0,114.9 196.8,115.3 197.6,115.6 198.3,115.7 199.1,115.8 199.9,115.9 200.7,116.0 201.5,116.3 202.3,116.5 203.1,116.8 203.8,117.2 204.6,117.4 205.4,117.6 206.2,117.7 207.0,117.7 207.8,117.8 208.5,117.8 209.3,117.8 210.1,117.8 210.9,117.6 211.7,117.5 212.5,117.6 213.2,117.9 214.0,118.3 214.8,118.9 215.6,119.4 216.4,120.0 217.1,120.2 217.9,120.4 218.7,120.7 219.5,120.9 220.3,121.3 221.1,121.7 221.8,122.0 222.6,122.2 223.4,122.3 224.2,122.3 225.0,122.3 225.8,122.6 226.5,122.9 227.3,123.2 228.1,123.6 228.9,124.3 229.7,125.0 230.5,125.7 231.2,126.4 232.0,127.0 232.8,127.6 233.6,128.0 234.4,128.6 235.2,129.2 236.0,129.5 236.7,129.9 237.5,130.4 238.3,130.8 239.1,131.1 239.9,131.2 240.7,131.6 241.4,132.1 242.2,132.6 243.0,133.1 243.8,133.7 244.6,134.4 245.4,134.8 246.1,135.1 246.9,135.5 247.7,135.8 248.5,136.0 249.3,135.9 250.0,135.9 250.8,136.0 251.6,136.2 252.4,136.2 253.2,136.3 254.0,136.4 254.8,136.5 255.5,136.6 256.3,136.6 257.1,136.5 257.9,136.7 258.7,136.9 259.4,136.9 260.2,137.0 261.0,136.9 261.8,136.9 262.6,137.2 263.4,137.4 264.1,137.9 264.9,138.3 265.7,138.8 266.5,138.9 267.3,139.1 268.1,139.4 268.9,139.4 269.6,139.5 270.4,139.5 271.2,139.5 272.0,139.6 272.8,139.7 273.6,139.6 274.3,139.4 275.1,139.1 275.9,139.2 276.7,139.3 277.5,139.4 278.2,139.5 279.0,139.6 279.8,139.6 280.6,139.7 281.4,139.8 282.2,139.7 282.9,139.6 283.7,139.5 284.5,139.3 285.3,139.1 286.1,139.1 286.9,139.1 287.6,138.8 288.4,138.6 289.2,138.2 290.0,137.9" fill="none" class="stroke-b"/>
<text x="172.0" y="22.0" class="fig-bold" text-anchor="middle">Adam with L2 in the gradient</text>
<line x1="340.0" y1="250.0" x2="575.0" y2="250.0" class="stroke-muted"/>
<line x1="340.0" y1="40.0" x2="340.0" y2="250.0" class="stroke-muted"/>
<line x1="340.0" y1="250.0" x2="340.0" y2="255.0" class="stroke-muted"/>
<text x="340.0" y="269.0" class="fig-small" text-anchor="middle">0</text>
<line x1="418.3" y1="250.0" x2="418.3" y2="255.0" class="stroke-muted"/>
<text x="418.3" y="269.0" class="fig-small" text-anchor="middle">100</text>
<line x1="496.7" y1="250.0" x2="496.7" y2="255.0" class="stroke-muted"/>
<text x="496.7" y="269.0" class="fig-small" text-anchor="middle">200</text>
<line x1="575.0" y1="250.0" x2="575.0" y2="255.0" class="stroke-muted"/>
<text x="575.0" y="269.0" class="fig-small" text-anchor="middle">300</text>
<line x1="340.0" y1="222.0" x2="575.0" y2="222.0" class="stroke-grid"/>
<line x1="335.0" y1="222.0" x2="340.0" y2="222.0" class="stroke-muted"/>
<text x="331.0" y="226.0" class="fig-small" text-anchor="end">0</text>
<line x1="340.0" y1="152.0" x2="575.0" y2="152.0" class="stroke-grid"/>
<line x1="335.0" y1="152.0" x2="340.0" y2="152.0" class="stroke-muted"/>
<text x="331.0" y="156.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="340.0" y1="82.0" x2="575.0" y2="82.0" class="stroke-grid"/>
<line x1="335.0" y1="82.0" x2="340.0" y2="82.0" class="stroke-muted"/>
<text x="331.0" y="86.0" class="fig-small" text-anchor="end">1</text>
<text x="457.5" y="288.0" class="fig-small" text-anchor="middle">step</text>
<polyline points="340.0,82.0 340.8,84.8 341.6,87.4 342.4,88.9 343.1,91.0 343.9,92.5 344.7,93.8 345.5,94.5 346.3,95.0 347.1,95.5 347.8,96.2 348.6,96.9 349.4,97.5 350.2,98.4 351.0,99.2 351.8,99.9 352.5,100.5 353.3,101.1 354.1,101.8 354.9,102.4 355.7,103.3 356.4,103.9 357.2,104.9 358.0,105.9 358.8,107.1 359.6,108.6 360.4,110.2 361.1,111.6 361.9,112.8 362.7,114.1 363.5,115.1 364.3,116.1 365.1,117.3 365.9,118.6 366.6,120.0 367.4,121.5 368.2,122.9 369.0,124.5 369.8,126.2 370.6,127.7 371.3,129.2 372.1,130.6 372.9,131.9 373.7,133.0 374.5,134.1 375.2,135.1 376.0,136.0 376.8,136.8 377.6,137.7 378.4,138.6 379.2,139.3 379.9,140.0 380.7,140.7 381.5,141.6 382.3,142.3 383.1,142.7 383.9,143.1 384.6,143.5 385.4,144.0 386.2,144.8 387.0,145.6 387.8,146.6 388.6,147.6 389.4,148.5 390.1,149.1 390.9,149.7 391.7,150.2 392.5,150.7 393.3,151.2 394.1,151.9 394.8,152.3 395.6,152.8 396.4,153.4 397.2,153.9 398.0,154.3 398.8,154.5 399.5,154.7 400.3,154.9 401.1,155.1 401.9,155.4 402.7,155.5 403.4,155.8 404.2,156.0 405.0,156.6 405.8,157.1 406.6,157.9 407.4,158.9 408.1,159.9 408.9,161.0 409.7,161.8 410.5,162.3 411.3,162.7 412.1,163.3 412.9,163.8 413.6,164.2 414.4,164.7 415.2,165.0 416.0,165.4 416.8,165.7 417.6,166.0 418.3,166.1 419.1,166.2 419.9,166.2 420.7,166.3 421.5,166.5 422.2,166.6 423.0,166.8 423.8,166.9 424.6,167.1 425.4,167.2 426.2,167.3 426.9,167.4 427.7,167.5 428.5,167.5 429.3,167.7 430.1,167.9 430.9,168.0 431.6,168.4 432.4,168.5 433.2,168.5 434.0,168.1 434.8,167.9 435.6,168.0 436.4,168.4 437.1,168.7 437.9,169.0 438.7,169.3 439.5,169.8 440.3,170.4 441.1,170.9 441.8,171.3 442.6,171.6 443.4,171.8 444.2,172.0 445.0,172.1 445.8,172.1 446.5,172.5 447.3,172.8 448.1,173.0 448.9,173.2 449.7,173.3 450.4,173.5 451.2,173.6 452.0,173.4 452.8,173.2 453.6,172.9 454.4,172.5 455.1,172.3 455.9,172.1 456.7,171.9 457.5,171.6 458.3,171.5 459.1,171.6 459.9,171.8 460.6,171.9 461.4,172.1 462.2,172.3 463.0,172.7 463.8,173.2 464.6,173.7 465.3,174.0 466.1,174.3 466.9,174.5 467.7,174.9 468.5,175.3 469.2,175.8 470.0,176.3 470.8,176.8 471.6,177.4 472.4,178.1 473.2,178.8 473.9,179.4 474.7,180.2 475.5,180.9 476.3,181.5 477.1,182.3 477.9,182.9 478.6,183.4 479.4,184.1 480.2,184.7 481.0,185.4 481.8,185.8 482.6,186.4 483.4,186.8 484.1,187.3 484.9,187.6 485.7,187.9 486.5,188.4 487.3,188.7 488.1,189.1 488.8,189.6 489.6,189.9 490.4,190.0 491.2,190.1 492.0,190.0 492.8,189.9 493.5,189.9 494.3,189.9 495.1,190.0 495.9,190.3 496.7,190.7 497.5,191.1 498.2,191.2 499.0,191.2 499.8,191.0 500.6,190.8 501.4,190.7 502.1,190.6 502.9,190.4 503.7,190.2 504.5,190.2 505.3,190.0 506.1,190.1 506.9,190.4 507.6,190.6 508.4,191.0 509.2,191.6 510.0,192.3 510.8,192.9 511.5,193.3 512.3,193.7 513.1,194.2 513.9,194.7 514.7,195.4 515.5,195.9 516.2,196.5 517.0,197.1 517.8,197.8 518.6,198.4 519.4,198.8 520.2,199.2 521.0,199.3 521.7,199.4 522.5,199.6 523.3,199.7 524.1,199.7 524.9,199.4 525.6,199.2 526.4,199.0 527.2,198.8 528.0,198.1 528.8,197.7 529.6,197.2 530.4,196.7 531.1,196.1 531.9,195.9 532.7,195.6 533.5,195.3 534.3,195.3 535.0,195.3 535.8,195.1 536.6,195.2 537.4,195.0 538.2,195.1 539.0,195.2 539.8,195.4 540.5,195.5 541.3,195.6 542.1,195.9 542.9,196.4 543.7,196.7 544.5,197.0 545.2,197.3 546.0,197.6 546.8,198.1 547.6,198.6 548.4,199.2 549.1,199.6 549.9,200.2 550.7,200.5 551.5,200.6 552.3,200.7 553.1,201.0 553.9,201.4 554.6,201.8 555.4,202.2 556.2,202.8 557.0,203.3 557.8,203.7 558.5,203.9 559.3,204.3 560.1,204.6 560.9,204.8 561.7,205.1 562.5,205.5 563.2,206.0 564.0,206.6 564.8,207.3 565.6,207.8 566.4,208.3 567.2,208.7 568.0,209.3 568.7,209.8 569.5,210.3 570.3,210.6 571.1,210.7 571.9,210.6 572.6,210.5 573.4,210.6 574.2,210.5 575.0,210.2" fill="none" class="stroke-a"/>
<polyline points="340.0,82.0 340.8,82.0 341.6,83.3 342.4,85.5 343.1,87.8 343.9,89.1 344.7,90.4 345.5,91.6 346.3,92.5 347.1,93.4 347.8,94.6 348.6,96.3 349.4,97.9 350.2,99.6 351.0,100.9 351.8,102.3 352.5,103.6 353.3,104.9 354.1,106.4 354.9,107.8 355.7,109.4 356.4,111.3 357.2,113.2 358.0,114.9 358.8,116.9 359.6,119.0 360.4,120.7 361.1,122.4 361.9,124.1 362.7,125.8 363.5,127.4 364.3,128.6 365.1,129.8 365.9,130.8 366.6,132.1 367.4,132.9 368.2,133.9 369.0,134.7 369.8,135.3 370.6,136.1 371.3,137.2 372.1,138.2 372.9,138.9 373.7,139.8 374.5,140.9 375.2,142.1 376.0,143.5 376.8,144.8 377.6,146.1 378.4,147.2 379.2,148.2 379.9,149.2 380.7,150.0 381.5,150.7 382.3,151.4 383.1,152.2 383.9,153.0 384.6,153.7 385.4,154.4 386.2,155.2 387.0,156.1 387.8,157.1 388.6,157.8 389.4,158.5 390.1,159.1 390.9,159.6 391.7,160.2 392.5,160.9 393.3,161.8 394.1,162.8 394.8,163.9 395.6,165.0 396.4,166.1 397.2,167.1 398.0,168.1 398.8,169.1 399.5,170.1 400.3,171.0 401.1,171.6 401.9,172.2 402.7,172.4 403.4,172.6 404.2,172.8 405.0,173.0 405.8,173.0 406.6,173.2 407.4,173.4 408.1,173.7 408.9,174.0 409.7,174.5 410.5,174.9 411.3,175.2 412.1,175.5 412.9,175.8 413.6,176.4 414.4,176.9 415.2,177.2 416.0,177.5 416.8,178.0 417.6,178.4 418.3,179.0 419.1,179.4 419.9,179.9 420.7,180.3 421.5,180.6 422.2,181.0 423.0,181.8 423.8,182.7 424.6,183.5 425.4,184.2 426.2,185.3 426.9,186.0 427.7,186.8 428.5,187.7 429.3,188.6 430.1,189.2 430.9,189.6 431.6,190.2 432.4,190.6 433.2,190.9 434.0,191.0 434.8,191.0 435.6,191.1 436.4,191.2 437.1,191.6 437.9,191.9 438.7,192.2 439.5,192.3 440.3,192.6 441.1,192.9 441.8,193.5 442.6,194.0 443.4,194.4 444.2,194.9 445.0,195.4 445.8,195.7 446.5,196.2 447.3,196.5 448.1,196.7 448.9,196.8 449.7,197.2 450.4,197.5 451.2,197.7 452.0,197.9 452.8,198.0 453.6,198.2 454.4,198.2 455.1,198.3 455.9,198.6 456.7,198.8 457.5,199.2 458.3,199.6 459.1,199.5 459.9,199.5 460.6,199.4 461.4,199.3 462.2,199.3 463.0,199.1 463.8,199.0 464.6,198.8 465.3,198.8 466.1,198.8 466.9,198.8 467.7,198.8 468.5,198.8 469.2,198.8 470.0,198.6 470.8,198.4 471.6,198.2 472.4,197.9 473.2,197.5 473.9,197.1 474.7,196.8 475.5,196.4 476.3,196.1 477.1,195.9 477.9,196.1 478.6,196.1 479.4,196.1 480.2,196.2 481.0,196.5 481.8,197.0 482.6,197.4 483.4,197.7 484.1,197.9 484.9,198.2 485.7,198.4 486.5,198.8 487.3,199.2 488.1,199.6 488.8,200.1 489.6,200.4 490.4,200.7 491.2,200.9 492.0,201.1 492.8,201.3 493.5,201.4 494.3,201.5 495.1,201.5 495.9,201.5 496.7,201.5 497.5,201.6 498.2,202.1 499.0,202.6 499.8,203.2 500.6,203.8 501.4,204.6 502.1,204.8 502.9,205.1 503.7,205.4 504.5,205.8 505.3,206.2 506.1,206.6 506.9,207.0 507.6,207.3 508.4,207.4 509.2,207.4 510.0,207.5 510.8,207.8 511.5,208.2 512.3,208.5 513.1,209.0 513.9,209.7 514.7,210.5 515.5,211.2 516.2,211.9 517.0,212.5 517.8,213.1 518.6,213.5 519.4,214.1 520.2,214.7 521.0,214.9 521.7,215.4 522.5,215.8 523.3,216.1 524.1,216.4 524.9,216.5 525.6,216.9 526.4,217.4 527.2,217.8 528.0,218.2 528.8,218.9 529.6,219.4 530.4,219.8 531.1,220.1 531.9,220.3 532.7,220.6 533.5,220.7 534.3,220.5 535.0,220.5 535.8,220.5 536.6,220.6 537.4,220.6 538.2,220.6 539.0,220.6 539.8,220.6 540.5,220.6 541.3,220.6 542.1,220.4 542.9,220.6 543.7,220.7 544.5,220.6 545.2,220.6 546.0,220.5 546.8,220.4 547.6,220.6 548.4,220.8 549.1,221.2 549.9,221.5 550.7,221.9 551.5,222.0 552.3,222.1 553.1,222.2 553.9,222.2 554.6,222.2 555.4,222.2 556.2,222.1 557.0,222.1 557.8,222.1 558.5,221.8 559.3,221.6 560.1,221.3 560.9,221.2 561.7,221.2 562.5,221.3 563.2,221.3 564.0,221.3 564.8,221.2 565.6,221.3 566.4,221.3 567.2,221.2 568.0,220.9 568.7,220.7 569.5,220.5 570.3,220.3 571.1,220.2 571.9,220.1 572.6,219.8 573.4,219.5 574.2,219.1 575.0,218.7" fill="none" class="stroke-b"/>
<text x="457.0" y="22.0" class="fig-bold" text-anchor="middle">AdamW</text>
<line x1="150.0" y1="334.0" x2="180.0" y2="334.0" class="stroke-a"/>
<text x="186.0" y="338.0" class="fig-small" text-anchor="start">quiet weight</text>
<line x1="340.0" y1="334.0" x2="370.0" y2="334.0" class="stroke-b"/>
<text x="376.0" y="338.0" class="fig-small" text-anchor="start">noisy weight</text>
</svg>
Figure 3. Both weights start at 1 and see only noise, with typical gradient size 0.01 (quiet, blue) and 10 (noisy, orange). The learning rate is 0.01 and the decay is 1. With L2 in the gradient (left), the quiet weight is driven to zero by step 300 while the noisy weight has only fallen to 0.60. With AdamW (right), both weights fade at about the same pace, ending at 0.08 and 0.02, close to the 0.05 that decay alone would give. With no decay at all, the weights only wander: they end at 1.22 and 0.84. The curves come from one run of random noise, so the exact endings change with the seed, but the quiet weight always collapses under L2.
```

> [!KEY]
> AdamW subtracts $\eta\lambda w$ after the Adam step, so every weight shrinks by the same fraction, and the averages see only the loss.

> [!RECAP]
> Hold these four points before you write code:
>
> - Weight decay removes the fraction `lr * weight_decay` of each weight per step.
> - In plain SGD an L2 penalty in the gradient and decay after the step are the same.
> - In Adam, `grad + weight_decay * w` goes through the division, so quiet weights shrink too much and noisy weights too little.
> - AdamW calls Adam on the raw gradient, then subtracts `lr * weight_decay * w` using the incoming `w`.

## Build it step by step

The starter gives you `sgd_step` and `adam_step`, written in earlier chapters. You write three functions, in the order the checks test them:

1. **`adam_l2_step(w, grad, state, lr, beta1, beta2, eps, weight_decay)`.** This is the coupled version, kept so that you can compare. Call `adam_step` on `grad + weight_decay * w`. The usual mistake is applying the decay after the step.
2. **`adamw_step(w, grad, state, lr, beta1, beta2, eps, weight_decay)`.** Call `adam_step` on the raw `grad`, then subtract `lr * weight_decay * w` with the incoming `w`. The usual mistake is using the new weight in the decay term.
3. **`decay_amount(step_fn, w, grads, lr, weight_decay)`.** Keep `w` frozen. Run `step_fn(w, g, state, lr=lr, weight_decay=d)` over every row of `grads`, threading the state. Do it once with `d = 0` and once with `d = weight_decay`, take the weights from the last call of each run, and return the first minus the second. The usual mistake is updating `w` between calls, which mixes the decay into later gradients.

## Where this breaks

Each of these has a direct remedy:

- A decay rate that is too large pulls the weights to zero faster than the loss can push back, and the loss stops falling. Lower $\lambda$.
- Decay on every parameter is rarely what you want. Common practice leaves biases and normalisation scales out of the decay.
- The decay is scaled by the learning rate, so a schedule that lowers the learning rate lowers the decay with it.
- Some libraries name their coupled penalty `weight_decay` on a plain Adam. It is the L2 version from this chapter. Read the documentation before trusting the name.

## The exercise

You implement `adam_l2_step`, `adamw_step`, and `decay_amount`. The starter supplies `sgd_step` and `adam_step`.

The checks verify that the L2 version is Adam on the penalised gradient, that AdamW differs from Adam by exactly $\eta\lambda w$, and that AdamW with zero decay is Adam. They verify that plain SGD gives the same answer both ways. Finally they measure the shrink from decay alone, and confirm that it is the same for both weights under AdamW and very uneven under L2.
