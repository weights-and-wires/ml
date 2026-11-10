---
title: Recurrent cells and backpropagation through time
slug: 044-recurrent-cells-and-bptt
part: "Part 5 — Vision and sequences"
summary: Reuse one set of weights at every step with a fixed-size state, then unroll the chain and backpropagate through it.
minutes: 75
gpu: false
objectives:
  - Explain why a fixed window fails on sequences of different lengths, and what a hidden state does instead.
  - Compute two steps of a recurrent cell by hand, and show that its weight count does not depend on the sequence length.
  - Write the cell, the loop that runs it over a sequence, and a readout that gives a loss and a gradient on every state.
  - Explain why a shared weight's gradient is a sum over steps, and why the previous state matters in that sum.
  - Write the gradient of one step and the backward loop, and check them against finite differences.
prereqs:
  - 030-softmax-and-whole-network-gradients
  - 029-backpropagation
  - 038-convolution-and-receptive-field
  - 039-im2col-and-convolution-backward
  - 040-residual-networks
exercise: 044-recurrent-cells-and-bptt
---

# Recurrent cells and backpropagation through time

A sensor on a machine sends one reading every minute. One example is a run of 3 readings, and another is a run of 30. Each run ends with a label, such as "overheated" or "fine". You want one model that reads either run and predicts the label.

A network you have built so far takes a fixed number of inputs. You could pad every run to 30 with zeros, but you would need a longer pad the day a run of 31 arrives. Padding also hides a deeper problem: the network treats step 3 and step 9 as unrelated places, even though a spike means the same thing in both. A person reading the numbers one by one would keep a running impression, such as "it has been warming up", and update it with every reading. A recurrent network does the same with one small vector, the hidden state, and one set of weights reused at every step.

Then the prediction is wrong, and the loss is one number. Three weights were used at every step to produce it. Which should change, and by how much? Nudging each weight and rerunning the sequence needs one pass per weight, which is hopeless for 400 weights. The chain rule seems stuck, because the same weight sits at every step. This chapter builds the cell, and then shows that drawing it once per step turns the problem into ordinary backpropagation.

## What you already know

- A dense layer turns an input vector into an output vector with a weight matrix and a bias, then bends it with a function such as tanh, which squashes any number into the range from −1 to 1.
- A convolution reuses the same weights at every position, so it finds a pattern wherever it appears ([convolution](/lessons/038-convolution-and-receptive-field)).
- Backpropagation starts at the loss and walks backward, multiplying by the local slope at each operation, and a value used in several places collects the sum of the gradients from each place ([backpropagation](/lessons/029-backpropagation)).
- For $z = x W^{\top}$, the gradient on $W$ is the upstream gradient transposed times $x$, and the gradient on $x$ is the upstream gradient times $W$.
- A gradient check nudges a value by a tiny amount and watches the loss ([whole-network gradients](/lessons/030-softmax-and-whole-network-gradients)).

## Keep a summary and update it

Figure 1 shows the idea twice. On the left, one cell reads an input, combines it with the summary so far, and feeds its answer back to itself. On the right, the same picture is unrolled: you copy the cell once per step, and each copy hands its output to the next.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="A recurrent cell drawn as a loop, and the same cell unrolled into four copies that share weights">
<defs><marker id="c037-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c037-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="20" y="28" class="fig-bold">Rolled: one cell, fed back</text>
<rect x="40" y="105" width="80" height="50" rx="6" class="fill-soft-b stroke-b"/>
<text x="80" y="135" text-anchor="middle">cell</text>
<line x1="80" y1="205" x2="80" y2="157" class="stroke-a" marker-end="url(#c037-arrow-a)"/>
<text x="80" y="225" text-anchor="middle" class="fill-a fig-bold">x</text>
<path d="M80,103 V68 H150 V130 H122" class="stroke-b" marker-end="url(#c037-arrow-b)"/>
<text x="160" y="98" class="fill-b fig-bold">h</text>
<text x="160" y="114" class="fig-small">fed back</text>
<text x="250" y="28" class="fig-bold">Unrolled: four copies of the same cell</text>
<rect x="255" y="105" width="58" height="50" rx="6" class="fill-soft-b stroke-b"/>
<text x="284" y="135" text-anchor="middle">cell</text>
<line x1="284" y1="205" x2="284" y2="157" class="stroke-a" marker-end="url(#c037-arrow-a)"/>
<text x="284" y="225" text-anchor="middle" class="fill-a fig-bold">x1</text>
<rect x="337" y="105" width="58" height="50" rx="6" class="fill-soft-b stroke-b"/>
<text x="366" y="135" text-anchor="middle">cell</text>
<line x1="366" y1="205" x2="366" y2="157" class="stroke-a" marker-end="url(#c037-arrow-a)"/>
<text x="366" y="225" text-anchor="middle" class="fill-a fig-bold">x2</text>
<rect x="419" y="105" width="58" height="50" rx="6" class="fill-soft-b stroke-b"/>
<text x="448" y="135" text-anchor="middle">cell</text>
<line x1="448" y1="205" x2="448" y2="157" class="stroke-a" marker-end="url(#c037-arrow-a)"/>
<text x="448" y="225" text-anchor="middle" class="fill-a fig-bold">x3</text>
<rect x="501" y="105" width="58" height="50" rx="6" class="fill-soft-b stroke-b"/>
<text x="530" y="135" text-anchor="middle">cell</text>
<line x1="530" y1="205" x2="530" y2="157" class="stroke-a" marker-end="url(#c037-arrow-a)"/>
<text x="530" y="225" text-anchor="middle" class="fill-a fig-bold">x4</text>
<line x1="221" y1="130" x2="255" y2="130" class="stroke-b" marker-end="url(#c037-arrow-b)"/>
<text x="238" y="121" text-anchor="middle" class="fill-b fig-bold">h0</text>
<line x1="313" y1="130" x2="337" y2="130" class="stroke-b" marker-end="url(#c037-arrow-b)"/>
<text x="325" y="121" text-anchor="middle" class="fill-b fig-bold">h1</text>
<line x1="395" y1="130" x2="419" y2="130" class="stroke-b" marker-end="url(#c037-arrow-b)"/>
<text x="407" y="121" text-anchor="middle" class="fill-b fig-bold">h2</text>
<line x1="477" y1="130" x2="501" y2="130" class="stroke-b" marker-end="url(#c037-arrow-b)"/>
<text x="489" y="121" text-anchor="middle" class="fill-b fig-bold">h3</text>
<line x1="559" y1="130" x2="593" y2="130" class="stroke-b" marker-end="url(#c037-arrow-b)"/>
<text x="576" y="121" text-anchor="middle" class="fill-b fig-bold">h4</text>
<text x="407" y="78" text-anchor="middle" class="fig-small">same W_xh, W_hh, and b in every copy</text>
<line x1="255" y1="85" x2="559" y2="85" class="stroke-muted dashed"/>
</svg>
Figure 1. The loop on the left and the chain on the right are the same network. In the chain, h0 enters the first copy, each copy hands its new state to the next, and every copy uses the same weights, so adding a step adds no parameters.
```

The summary is the **hidden state**, written $h$. It is a vector of fixed size $d_h$, so it can hold only so much, which means the cell must decide what to keep. The state before the first step is $h_0$, usually all zeros. After reading input $x_t$, the cell produces $h_t$ from the input and the previous state.

The simplest cell is a leaky running total, $h_t = 0.8\,h_{t-1} + 0.5\,x_t$. The 0.8 says how much of the old summary survives, and the 0.5 says how much a fresh reading counts. For the readings $x_1 = 1$ and $x_2 = 0.5$ from $h_0 = 0$, you get $h_1 = 0.5$ and $h_2 = 0.8 \times 0.5 + 0.5 \times 0.5 = 0.65$. Why not stop there? A weight of 1 or more on the old state would push the total up without limit, and the total is a straight line in its inputs, so it can only blend readings and cannot decide anything. A real cell wraps the same sum in $\tanh$. The bend keeps every entry of the state between −1 and 1, and it lets the cell respond to a pattern instead of an average.

### The cell, step by step

Keep the weights 0.5 and 0.8, and the readings $x_1 = 1$ and $x_2 = 0.5$. $W_{xh}$ is the weight on the input, and it decides how much each reading matters. $W_{hh}$ is the weight on the previous state, and it decides how much memory carries over. The bias $b$ shifts the sum, and here it is 0.

At step 1 the number inside the squash is $0.5 \times 1 + 0.8 \times 0 + 0 = 0.5$, so $h_1 = \tanh(0.5) = 0.4621$. At step 2 it is $0.5 \times 0.5 + 0.8 \times 0.4621 = 0.6197$, so $h_2 = \tanh(0.6197) = 0.5509$. In symbols, one step is

$$
h_t = \tanh\left(x_t W_{xh}^{\top} + h_{t-1} W_{hh}^{\top} + b\right)
$$

Read it aloud: multiply the input by its weights, multiply the previous state by its weights, add the bias, and squash the total. In numpy, with the rows of every array being examples, it is one line:

```python
h_t = np.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b)
```

$W_{xh}$ has shape `(d_h, d_x)` and $W_{hh}$ has shape `(d_h, d_h)`, written as (output, input) the way a dense layer stores its weights, so each needs a transpose when the rows of `x_t` are examples. With zero input after step 2, the state shrinks by a factor a little under 0.8 per step: the cell remembers, and slowly forgets.

A whole sequence is stored time-major, as an array of shape `(T, n, d_x)` with $T$ steps and $n$ examples. Running the cell over it gives `(T, n, d_h)`, where row 0 is $h_1$. You do not store $h_0$, because you already have it.

### One set of weights for any length

Suppose each reading has $d_x = 8$ features and you want 16 hidden units. A dense layer over a window of 30 readings has $30 \times 8 \times 16 + 16 = 3856$ weights, and a window of 60 roughly doubles that. A recurrent cell has $d_h(d_h + d_x + 1) = 16 \times 25 = 400$ weights: the $16 \times 16$ matrix $W_{hh}$, the $16 \times 8$ matrix $W_{xh}$, and 16 biases. The count is 400 for a run of 3 and 400 for a run of 3,000, because the copies in Figure 1 share everything.

Sharing also makes the model learn faster, because a spike at step 3 and a spike at step 9 go through the same weights. A window can only see so far back. If the signal that decides the label is a spike at step 2 and the window of three covers steps 6, 7, and 8, no weights can use a number the model never sees. The hidden state has read every step, so the information is at least available to it. How much of it survives is a question about the weights, which [vanishing and exploding gradients](/lessons/045-vanishing-gradients-and-clipping) takes up.

### Giving the state a job

To train a cell, you read a result out of the state and score it. Read out the last state through a linear layer, $\hat y = h_T W_{hy}^{\top} + b_y$, and use the squared error

$$
L = \tfrac{1}{2}\sum (\hat y - y)^2
$$

Read it aloud: the prediction is a weighted copy of the final state, and the loss is half the squared gap to the target, added up over every example. Backpropagation needs the gradient of $L$ with respect to each state, called $dH$. The loss reads only $h_T$, so every other state gets 0 from the loss itself and the last one gets $(\hat y - y)\,W_{hy}$.

Try it on the two steps above, with $W_{hy} = 2$, $b_y = 0$, and a target $y = 1$. The prediction is $\hat y = 2 \times 0.5509 = 1.1018$, so the gap is 0.1018 and $L = \tfrac{1}{2} \times 0.1018^2 = 0.0052$. The gradient on the last state is $0.1018 \times 2 = 0.2037$, and on the first state it is 0. If the task scored every step, as in tagging every word of a sentence, `dH` would be nonzero at every step.

```python
err = h_seq[-1] @ W_hy.T + b_y - y
dH = np.zeros_like(h_seq)
dH[-1] = err @ W_hy
```

> [!KEY]
> A recurrent cell is one set of weights applied at every step, carrying a fixed-size state. A run of 3 readings and a run of 30 go through the same model, and the weight count stays the same.

## Unroll the chain and walk it backward

Once the cell is drawn once per step, the recurrent network is a long ordinary network whose layers happen to share weights. Run backpropagation on it, and the chain rule does the rest. Use the running example again, but start from $h_0 = 0.3$ so the starting state has a nonzero value for $W_{hh}$ to multiply. The forward pass gives $h_1 = \tanh(0.5 + 0.8 \times 0.3) = \tanh(0.74) = 0.6291$ and $h_2 = \tanh(0.25 + 0.8 \times 0.6291) = \tanh(0.7533) = 0.6371$.

To keep the numbers clean, pretend the loss pulls on $h_2$ with a strength of 1, so $dH_2 = 1$. Every gradient below is proportional to that strength, so the pattern holds for a real loss. Figure 2 shows the forward pass in blue, and the gradient walking back through it in orange.

```figure
<svg viewBox="0 0 620 285" role="img" aria-label="Forward values flow right in blue along two cells; gradients flow left in orange, multiplied by the tanh slope and by the recurrent weight at each step">
<defs><marker id="c037-f2-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c037-f2-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="20" y="62" class="fig-bold">Forward pass</text>
<rect x="140" y="95" width="80" height="50" rx="6" class="fill-soft-a stroke-a"/>
<text x="180" y="115" text-anchor="middle">step 1</text>
<text x="180" y="133" text-anchor="middle" class="fig-small">x1 = 1</text>
<rect x="380" y="95" width="80" height="50" rx="6" class="fill-soft-a stroke-a"/>
<text x="420" y="115" text-anchor="middle">step 2</text>
<text x="420" y="133" text-anchor="middle" class="fig-small">x2 = 0.5</text>
<line x1="20" y1="120" x2="138" y2="120" class="stroke-a" marker-end="url(#c037-f2-arrow-a)"/>
<text x="80" y="112" text-anchor="middle" class="fig-small">h0 = 0.3</text>
<line x1="222" y1="120" x2="378" y2="120" class="stroke-a" marker-end="url(#c037-f2-arrow-a)"/>
<text x="300" y="112" text-anchor="middle" class="fig-small">h1 = 0.6291</text>
<line x1="462" y1="120" x2="600" y2="120" class="stroke-a" marker-end="url(#c037-f2-arrow-a)"/>
<text x="535" y="112" text-anchor="middle" class="fig-small">h2 = 0.6371</text>
<text x="20" y="188" class="fig-bold">Backward pass</text>
<rect x="528" y="217" width="64" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="560" y="236" text-anchor="middle">1</text>
<text x="560" y="264" text-anchor="middle" class="fig-small">dh2</text>
<rect x="388" y="217" width="64" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="420" y="236" text-anchor="middle">0.5941</text>
<text x="420" y="264" text-anchor="middle" class="fig-small">dz2</text>
<rect x="268" y="217" width="64" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="300" y="236" text-anchor="middle">0.4753</text>
<text x="300" y="264" text-anchor="middle" class="fig-small">dh1</text>
<rect x="148" y="217" width="64" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="180" y="236" text-anchor="middle">0.2871</text>
<text x="180" y="264" text-anchor="middle" class="fig-small">dz1</text>
<rect x="28" y="217" width="64" height="28" rx="4" class="fill-soft-b stroke-b"/>
<text x="60" y="236" text-anchor="middle">0.2297</text>
<text x="60" y="264" text-anchor="middle" class="fig-small">dh0</text>
<line x1="526" y1="231" x2="454" y2="231" class="stroke-b" marker-end="url(#c037-f2-arrow-b)"/>
<text x="490" y="209" text-anchor="middle" class="fig-small fill-b">× 0.5941</text>
<text x="490" y="277" text-anchor="middle" class="fig-small"></text>
<line x1="386" y1="231" x2="334" y2="231" class="stroke-b" marker-end="url(#c037-f2-arrow-b)"/>
<text x="360" y="209" text-anchor="middle" class="fig-small fill-b">× 0.8</text>
<text x="360" y="277" text-anchor="middle" class="fig-small"></text>
<line x1="266" y1="231" x2="214" y2="231" class="stroke-b" marker-end="url(#c037-f2-arrow-b)"/>
<text x="240" y="209" text-anchor="middle" class="fig-small fill-b">× 0.6042</text>
<text x="240" y="277" text-anchor="middle" class="fig-small"></text>
<line x1="146" y1="231" x2="94" y2="231" class="stroke-b" marker-end="url(#c037-f2-arrow-b)"/>
<text x="120" y="209" text-anchor="middle" class="fig-small fill-b">× 0.8</text>
<text x="120" y="277" text-anchor="middle" class="fig-small"></text>
<line x1="180" y1="147" x2="180" y2="215" class="stroke-muted dashed"/>
<line x1="420" y1="147" x2="420" y2="215" class="stroke-muted dashed"/>
</svg>
Figure 2. A pull of 1 on h2 travels back through two cells. Each cell contributes two multiplications: the tanh slope (0.5941 at step 2, 0.6042 at step 1) and the weight 0.8. Notice that the value shrinks at every multiplication, from 1 down to 0.2297 at h0.
```

Two things happen at every cell as the gradient walks back. First, it passes through tanh, which scales it by the slope $1 - h^2$ at that step, because a flat part of tanh passes less of a nudge than a steep part. Second, it passes through the recurrent weight, which scales it by 0.8. The 1 on $h_2$ becomes $0.5941$ at the cell, then $0.4753$ on $h_1$, then $0.2871$, then $0.2297$ on $h_0$.

### Where the loss lives

The gradient at a state has two sources. The first is the loss attached directly to that state, $dH_t$. The second is everything later in the sequence, which reaches $h_t$ by flowing back through the next cell. Call that the **carry**:

$$
dh_t = dH_t + \text{carry}
$$

In the example, $dH_2 = 1$ and the carry into step 2 is 0, because nothing comes after it. At step 1, $dH_1 = 0$ and the carry is $0.4753$, so $dh_1 = 0.4753$ comes only from the future.

### One step backward

Take step 2 alone. The state came out of $h_2 = \tanh(z_2)$, so the gradient on the sum inside tanh is $dz_2 = dh_2 \times (1 - h_2^2) = 0.5941$. You do not need $z_2$, because the slope can be computed from the stored output $h_2$. The sum $z_2 = x_2 W_{xh}^{\top} + h_1 W_{hh}^{\top} + b$ has three terms, so $dz_2$ flows to each. The gradient on a weight is $dz$ times the value that weight multiplied, and the gradient on a value is $dz$ times the weight. Figure 3 lays out the five results.

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="A box diagram of one backward step: the incoming gradient passes through the tanh gate and splits into five outputs">
<defs><marker id="c037-f3-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<rect x="10" y="115" width="110" height="56" rx="6" class="fill-soft-b stroke-b"/>
<text x="65" y="138" text-anchor="middle" class="fig-bold">dh_t</text>
<text x="65" y="156" text-anchor="middle" class="fig-small">(n, d_h)</text>
<text x="65" y="108" text-anchor="middle" class="fig-small">dH[t] + carry</text>
<line x1="122" y1="143" x2="148" y2="143" class="stroke-b" marker-end="url(#c037-f3-arrow-b)"/>
<rect x="150" y="115" width="110" height="56" rx="6" class="fill-soft-d stroke-d"/>
<text x="205" y="138" text-anchor="middle" class="fig-bold">tanh gate</text>
<text x="205" y="156" text-anchor="middle" class="fig-small">× (1 − h_t²)</text>
<line x1="262" y1="143" x2="288" y2="143" class="stroke-b" marker-end="url(#c037-f3-arrow-b)"/>
<rect x="290" y="115" width="90" height="56" rx="6" class="fill-soft-b stroke-b"/>
<text x="335" y="138" text-anchor="middle" class="fig-bold">dz</text>
<text x="335" y="156" text-anchor="middle" class="fig-small">(n, d_h)</text>
<rect x="420" y="14" width="190" height="42" rx="6" class="fill-soft-b stroke-b"/>
<text x="430" y="33">dx_t = dz W_xh</text>
<text x="430" y="49" class="fig-small">(n, d_x)</text>
<line x1="382" y1="143" x2="418" y2="35" class="stroke-b" marker-end="url(#c037-f3-arrow-b)"/>
<rect x="420" y="68" width="190" height="42" rx="6" class="fill-soft-b stroke-b"/>
<text x="430" y="87">dW_xh = dzᵀ x_t</text>
<text x="430" y="103" class="fig-small">(d_h, d_x)</text>
<line x1="382" y1="143" x2="418" y2="89" class="stroke-b" marker-end="url(#c037-f3-arrow-b)"/>
<rect x="420" y="122" width="190" height="42" rx="6" class="fill-soft-b stroke-b"/>
<text x="430" y="141">dW_hh = dzᵀ h_prev</text>
<text x="430" y="157" class="fig-small">(d_h, d_h)</text>
<line x1="382" y1="143" x2="418" y2="143" class="stroke-b" marker-end="url(#c037-f3-arrow-b)"/>
<rect x="420" y="176" width="190" height="42" rx="6" class="fill-soft-b stroke-b"/>
<text x="430" y="195">db = sum of dz over n</text>
<text x="430" y="211" class="fig-small">(d_h,)</text>
<line x1="382" y1="143" x2="418" y2="197" class="stroke-b" marker-end="url(#c037-f3-arrow-b)"/>
<rect x="420" y="230" width="190" height="42" rx="6" class="fill-soft-d stroke-d"/>
<text x="430" y="249">dh_prev = dz W_hh</text>
<text x="430" y="265" class="fig-small">(n, d_h)</text>
<line x1="382" y1="143" x2="418" y2="251" class="stroke-b" marker-end="url(#c037-f3-arrow-b)"/>
<text x="500" y="288" text-anchor="middle" class="fig-small fill-d">dh_prev is the carry for step t − 1</text>
</svg>
Figure 3. One step of the backward loop. The tanh gate turns dh_t into dz, and dz alone produces every other gradient. Four outputs are collected into sums or written to dx, and the purple one, dh_prev, becomes the carry for the previous step.
```

At step 2 these are $dW_{xh} = 0.5941 \times 0.5 = 0.2970$, $dW_{hh} = 0.5941 \times 0.6291 = 0.3738$, $db = 0.5941$, $dx_2 = 0.2970$, and the part sent to the previous state, $dh_1 = 0.5941 \times 0.8 = 0.4753$, which is the carry. For a batch whose rows are examples, the five lines are

```python
dz = dh_t * (1.0 - h_t ** 2)
dx_t, dh_prev = dz @ W_xh, dz @ W_hh
dW_xh, dW_hh, db = dz.T @ x_t, dz.T @ h_prev, dz.sum(axis=0)
```

Here `dz.sum(axis=0)` adds up the rows of the batch, because the bias was added to every example.

### Each copy of a weight gets a nudge

Step 1 has $dh_1 = 0 + 0.4753$. The slope is $1 - 0.6291^2 = 0.6042$, so $dz_1 = 0.4753 \times 0.6042 = 0.2871$. The weights at step 1 get $dW_{xh} = 0.2871 \times 1 = 0.2871$, $dW_{hh} = 0.2871 \times 0.3 = 0.0861$, and $db = 0.2871$. The carry that leaves step 1 is $0.2871 \times 0.8 = 0.2297$, the gradient on $h_0$.

Think of $W_{hh}$ as being photocopied once per step. Each copy was used once and receives its own nudge, and the weight is one object that gets all of them. So its gradient is the sum, $0.0861 + 0.3738 = 0.4599$. In the same way $dW_{xh} = 0.2871 + 0.2970 = 0.5842$ and $db = 0.2871 + 0.5941 = 0.8812$. Figure 4 shows the sums.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Bars showing the gradient of W_hh and W_xh as a sum of one contribution from each step">
<defs></defs>
<text x="160" y="32" text-anchor="middle" class="fig-bold">Gradient of W_hh</text>
<line x1="55" y1="215" x2="275" y2="215" class="stroke-muted"/>
<rect x="60" y="193.5" width="44" height="21.5" class="fill-soft-b stroke-b"/>
<text x="82" y="186.5" text-anchor="middle" class="fig-small">0.0861</text>
<text x="82" y="233" text-anchor="middle" class="fig-small">step 1 copy</text>
<rect x="135" y="121.6" width="44" height="93.4" class="fill-soft-b stroke-b"/>
<text x="157" y="114.6" text-anchor="middle" class="fig-small">0.3738</text>
<text x="157" y="233" text-anchor="middle" class="fig-small">step 2 copy</text>
<rect x="210" y="100" width="44" height="115" class="fill-b"/>
<text x="232" y="93" text-anchor="middle" class="fig-small">0.4599</text>
<text x="232" y="233" text-anchor="middle" class="fig-small">sum</text>
<text x="440" y="32" text-anchor="middle" class="fig-bold">Gradient of W_xh</text>
<line x1="335" y1="215" x2="555" y2="215" class="stroke-muted"/>
<rect x="340" y="143.2" width="44" height="71.8" class="fill-soft-b stroke-b"/>
<text x="362" y="136.2" text-anchor="middle" class="fig-small">0.2871</text>
<text x="362" y="233" text-anchor="middle" class="fig-small">step 1 copy</text>
<rect x="415" y="140.7" width="44" height="74.3" class="fill-soft-b stroke-b"/>
<text x="437" y="133.7" text-anchor="middle" class="fig-small">0.2970</text>
<text x="437" y="233" text-anchor="middle" class="fig-small">step 2 copy</text>
<rect x="490" y="69" width="44" height="146" class="fill-b"/>
<text x="512" y="62" text-anchor="middle" class="fig-small">0.5842</text>
<text x="512" y="233" text-anchor="middle" class="fig-small">sum</text>
<text x="300" y="250" text-anchor="middle" class="fig-small">The step 1 copy of W_hh multiplies the starting state 0.3, so its bar is small but not zero.</text>
</svg>
Figure 4. One shared weight is several copies. Each copy gets its own nudge, and the weight's gradient is their sum: 0.0861 + 0.3738 = 0.4599 for W_hh, and 0.2871 + 0.2970 = 0.5842 for W_xh.
```

The full backward loop repeats two moves from the last step to the first: add $dH_t$ to the carry, take one step backward, add that step's weight gradients to the running sums, and keep the carry it returns. After step 1, the carry is $dh_0$.

$$
dh_t = dH_t + \text{carry}, \qquad
dW = \sum_t dW^{(t)}, \qquad
\text{carry} \leftarrow dz_t\, W_{hh}
$$

Read it aloud: the gradient on each state is its own loss plus what came back from the future, each weight's gradient is the total of its per-step nudges, and the carry is the gradient that goes to the previous state. In code, inside `for t in reversed(range(T))`:

```python
h_prev = h_seq[t - 1] if t > 0 else h0
dx[t], carry, dW_xh_t, dW_hh_t, db_t = step_backward(dH[t] + carry, h_seq[t], h_prev, x[t], W_xh, W_hh)
dW_hh += dW_hh_t
```

> [!WARNING]
> Row `t` of the stored states is $h_{t+1}$, so the state before step `t` is `h_seq[t - 1]`, or `h0` when `t` is 0. Using `h_seq[t]` there gives $dW_{hh} = 0.5592$ in the example instead of $0.4599$, and with $h_0 = 0$ the mistake at step 1 would be hidden.

### Trust the numbers by checking them

Add $10^{-6}$ to $W_{hh}$ and subtract $10^{-6}$ from it, rerun the forward pass each time, and divide the difference in $h_2$ by $2 \times 10^{-6}$. The result is $0.4599$, matching the sum above. The exercise checks every output this way, so if a gradient in the prose and the code disagree, the finite difference tells you which one is wrong.

> [!KEY]
> Backpropagation through time is ordinary backpropagation on the unrolled chain. The gradient on each state is its own loss plus a carry from the future, and each shared weight's gradient is the sum of the nudges from every copy. The loss blames each weight in proportion to how much each copy of it mattered.

> [!RECAP]
> Hold these points before you build:
>
> - The hidden state is a fixed-size summary, updated at every step with $h_t = \tanh(x_t W_{xh}^{\top} + h_{t-1} W_{hh}^{\top} + b)$. The weight count is $d_h(d_h + d_x + 1)$ for any length.
> - Store the states $h_1$ through $h_T$ as `(T, n, d_h)`, and do not store $h_0$. A last-step loss gives a `dH` that is zero except in the last row.
> - The gradient on a state is $dH_t$ plus the carry from later steps. At each step, multiply by $1 - h_t^2$ to get $dz$, then use $dz$ to produce $dx$, the carry $dz\,W_{hh}$, and this step's weight gradients.
> - Each shared weight's gradient is the sum of its per-step gradients, and the previous state at step 0 is $h_0$.

## Build it step by step

The checks call the functions in this order.

1. **`rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b)`.** `x_t` is `(n, d_x)` and `h_prev` is `(n, d_h)`. Return the next state, `(n, d_h)`, with `@`, `.T`, and `np.tanh`. The usual mistake is forgetting a transpose, which fails whenever `d_x` and `d_h` differ.
2. **`rnn_forward(x, h0, W_xh, W_hh, b)`.** `x` is `(T, n, d_x)`. Keep a running `h_prev` that starts at `h0`, call your cell at every step, and store each result in row `t` of an array `(T, n, d_h)`. The usual mistake is storing `h0` in row 0.
3. **`last_step_loss_and_dH(h_seq, W_hy, b_y, y)`.** Compute the prediction from `h_seq[-1]`, return the loss as a float, and return `dH` with the shape of `h_seq`, filled only in the last row. The usual mistake is dropping the factor $W_{hy}$ when sending the gradient back through the readout.
4. **`tanh_backward(dh, h)`.** `h` is the output of tanh. Return `dh * (1 - h**2)`. The usual mistake is applying tanh to `h` a second time.
5. **`step_backward(dh_t, h_t, h_prev, x_t, W_xh, W_hh)`.** Call your tanh gradient to get `dz`, then return `(dx_t, dh_prev, dW_xh, dW_hh, db)`. The usual mistake is a missing transpose in the weight gradients.
6. **`bptt(dH, x, h0, W_xh, W_hh, b)`.** Run `rnn_forward` once, then loop from the last step to the first, with the carry starting at zeros. Pass `dH[t] + carry` to `step_backward`, add the weight gradients into running sums, and copy `dx[t]`. Return `(dx, dW_xh, dW_hh, db, dh0)`, where `dh0` is the final carry. The usual mistake is using `h_seq[t]` as the previous state.

## Cost and scaling

The weights do not grow with $T$, but the work does: one step costs about $2 n d_h (d_h + d_x)$ operations, and a sequence costs $T$ times that. The backward pass visits each step once and does about the same work as the forward pass. It needs every state from the forward pass, so the memory for training grows with $n T d_h$. [Truncation and clipping](/lessons/045-vanishing-gradients-and-clipping) show how to cap that memory.

## Where this breaks

- Step $t$ needs the state from step $t - 1$, so the $T$ steps run one after another, however many processors you have.
- The state has a fixed size, so something must be forgotten, and a reading that arrived long ago can fade to nothing.
- The carry passes through a multiplication at every step, by 0.8 and by a tanh slope in the example, and the pull of 1 on $h_2$ has already become 0.2297 on $h_0$ after two steps. [The next chapter](/lessons/045-vanishing-gradients-and-clipping) asks what that does over many steps.
- If you keep only the last step's weight gradient and forget the sum over steps, the weights train on a fraction of the signal.

## The exercise

You build `rnn_cell_forward`, `rnn_forward`, `last_step_loss_and_dH`, `tanh_backward`, `step_backward`, and `bptt`. The checks match the hand examples: the first two states 0.4621 and 0.5509, the six-step trace, a loss of 0.005185, and $dW_{xh} = 0.5842$, $dW_{hh} = 0.4599$, $db = 0.8812$, and $dh_0 = 0.2297$. They check that reading a prefix gives the first rows of the full run, compare every gradient with finite differences, and check that a loss on every step gives the same gradients as the losses of each step added up.
