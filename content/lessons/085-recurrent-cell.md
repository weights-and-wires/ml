---
title: A network that reads one step at a time
slug: 085-recurrent-cell
part: "Part 5 — Vision and sequences"
summary: Reuse one set of weights at every step and carry a fixed-size state, so one model can read a sequence of any length.
minutes: 40
gpu: false
objectives:
  - Explain why a fixed window fails on sequences of different lengths, and what a hidden state does instead.
  - Compute two steps of a recurrent cell by hand, and say what each of the three weights is for.
  - Count the weights of a recurrent cell and show that the count does not depend on the sequence length.
  - Write the cell, the loop that runs it over a sequence, and a readout that gives a loss and a gradient on every state.
prereqs:
  - 064-backprop-mlp-and-gradient-check
  - 077-sliding-a-filter
  - 084-bottleneck-cost
exercise: 085-recurrent-cell
---

# A network that reads one step at a time

A sensor on a machine sends one reading every minute. One example is a run of 3 readings, and another is a run of 30. Each run ends with a label, such as "overheated" or "fine". You want one model that reads either run and predicts the label.

A network you have built so far takes a fixed number of inputs. A run of 3 numbers and a run of 30 numbers do not fit the same input layer. You could pad every run to 30 with zeros, but you would need a longer pad the day a run of 31 arrives. Padding also hides a deeper problem, which this chapter shows: the network treats step 3 and step 9 as unrelated places, even though a spike means the same thing in both.

A person reading the readings one by one would keep a running impression, such as "it has been warming up", and update it with every reading. A recurrent network does the same with one small vector, called the hidden state, and one set of weights that it reuses at every step.

## What you already know

You already use these ideas:

- A dense layer turns an input vector into an output vector with a weight matrix and a bias, then bends it with a function such as tanh.
- Tanh squashes any number into the range from −1 to 1.
- A convolution reuses the same weights at every position, so it finds a pattern wherever it appears.
- The loss is one number that says how wrong a prediction is, and the gradient says how the loss changes when a value changes.
- In numpy, a matrix of shape `(n, d)` times a matrix of shape `(d, m)` gives shape `(n, m)`.

## Keep a summary and update it

Figure 1 shows the idea twice. On the left, one cell reads an input, combines it with the summary so far, and feeds its answer back to itself. On the right, the same picture is unrolled: you copy the cell once per step, and each copy hands its output to the next.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="A recurrent cell drawn as a loop, and the same cell unrolled into four copies that share weights">
<defs><marker id="c085-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c085-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="20" y="28" class="fig-bold">Rolled: one cell, fed back</text>
<rect x="40" y="105" width="80" height="50" rx="6" class="fill-soft-b stroke-b"/>
<text x="80" y="135" text-anchor="middle">cell</text>
<line x1="80" y1="205" x2="80" y2="157" class="stroke-a" marker-end="url(#c085-arrow-a)"/>
<text x="80" y="225" text-anchor="middle" class="fill-a fig-bold">x</text>
<path d="M80,103 V68 H150 V130 H122" class="stroke-b" marker-end="url(#c085-arrow-b)"/>
<text x="160" y="98" class="fill-b fig-bold">h</text>
<text x="160" y="114" class="fig-small">fed back</text>
<text x="250" y="28" class="fig-bold">Unrolled: four copies of the same cell</text>
<rect x="255" y="105" width="58" height="50" rx="6" class="fill-soft-b stroke-b"/>
<text x="284" y="135" text-anchor="middle">cell</text>
<line x1="284" y1="205" x2="284" y2="157" class="stroke-a" marker-end="url(#c085-arrow-a)"/>
<text x="284" y="225" text-anchor="middle" class="fill-a fig-bold">x1</text>
<rect x="337" y="105" width="58" height="50" rx="6" class="fill-soft-b stroke-b"/>
<text x="366" y="135" text-anchor="middle">cell</text>
<line x1="366" y1="205" x2="366" y2="157" class="stroke-a" marker-end="url(#c085-arrow-a)"/>
<text x="366" y="225" text-anchor="middle" class="fill-a fig-bold">x2</text>
<rect x="419" y="105" width="58" height="50" rx="6" class="fill-soft-b stroke-b"/>
<text x="448" y="135" text-anchor="middle">cell</text>
<line x1="448" y1="205" x2="448" y2="157" class="stroke-a" marker-end="url(#c085-arrow-a)"/>
<text x="448" y="225" text-anchor="middle" class="fill-a fig-bold">x3</text>
<rect x="501" y="105" width="58" height="50" rx="6" class="fill-soft-b stroke-b"/>
<text x="530" y="135" text-anchor="middle">cell</text>
<line x1="530" y1="205" x2="530" y2="157" class="stroke-a" marker-end="url(#c085-arrow-a)"/>
<text x="530" y="225" text-anchor="middle" class="fill-a fig-bold">x4</text>
<line x1="221" y1="130" x2="255" y2="130" class="stroke-b" marker-end="url(#c085-arrow-b)"/>
<text x="238" y="121" text-anchor="middle" class="fill-b fig-bold">h0</text>
<line x1="313" y1="130" x2="337" y2="130" class="stroke-b" marker-end="url(#c085-arrow-b)"/>
<text x="325" y="121" text-anchor="middle" class="fill-b fig-bold">h1</text>
<line x1="395" y1="130" x2="419" y2="130" class="stroke-b" marker-end="url(#c085-arrow-b)"/>
<text x="407" y="121" text-anchor="middle" class="fill-b fig-bold">h2</text>
<line x1="477" y1="130" x2="501" y2="130" class="stroke-b" marker-end="url(#c085-arrow-b)"/>
<text x="489" y="121" text-anchor="middle" class="fill-b fig-bold">h3</text>
<line x1="559" y1="130" x2="593" y2="130" class="stroke-b" marker-end="url(#c085-arrow-b)"/>
<text x="576" y="121" text-anchor="middle" class="fill-b fig-bold">h4</text>
<text x="407" y="78" text-anchor="middle" class="fig-small">same W_xh, W_hh, and b in every copy</text>
<line x1="255" y1="85" x2="559" y2="85" class="stroke-muted dashed"/>
</svg>
Figure 1. The loop on the left and the chain on the right are the same network. In the chain, h0 enters the first copy, each copy hands its new state to the next, and every copy uses the same weights, so adding a step adds no parameters.
```

The summary is the **hidden state**, written $h$. It is a vector of fixed size $d_h$, so it can hold only so much, which means the cell must decide what to keep. The state before the first step is $h_0$, usually all zeros. After reading input $x_t$, the cell produces $h_t$ from two things: the input and the previous state.

The simplest cell you could write is a leaky running total. Take $h_t = 0.8\,h_{t-1} + 0.5\,x_t$. The 0.8 says how much of the old summary survives, and the 0.5 says how much a fresh reading counts. Feed it the readings $x_1 = 1$ and then $x_2 = 0.5$, starting from $h_0 = 0$. You get $h_1 = 0.5$ and $h_2 = 0.8 \times 0.5 + 0.5 \times 0.5 = 0.65$.

You might ask why not stop there. The leaky total has two problems. An input of 1 on every step pushes it up toward 2.5, and a weight of 1 or more on the old state would push it up without limit. It is also a straight line in its inputs, so it can only blend readings and cannot decide anything. A real cell needs one bend, so it wraps the same sum in $\tanh$. The bend keeps every entry of the state between −1 and 1, which means a long run cannot push the state off to infinity, and it lets the cell respond to a pattern instead of an average.

## The cell, step by step

Keep the weights 0.5 and 0.8, and the readings $x_1 = 1$ and $x_2 = 0.5$. Give the three numbers their proper names. $W_{xh}$ is the weight on the input, and it decides how much each reading matters. $W_{hh}$ is the weight on the previous state, and it decides how much memory carries over. The bias $b$ shifts the sum, and here it is 0.

At step 1 the number inside the squash is $0.5 \times 1 + 0.8 \times 0 + 0 = 0.5$, so $h_1 = \tanh(0.5) = 0.4621$. At step 2 it is $0.5 \times 0.5 + 0.8 \times 0.4621 = 0.25 + 0.3697 = 0.6197$, so $h_2 = \tanh(0.6197) = 0.5509$. The state rose from 0.4621 to 0.5509 because the second reading was positive and the first state was still mostly there.

In symbols, one step is

$$
h_t = \tanh\left(x_t W_{xh}^{\top} + h_{t-1} W_{hh}^{\top} + b\right)
$$

Read it aloud: multiply the input by its weights, multiply the previous state by its weights, add the bias, and squash the total. In numpy, with the rows of every array being examples, it is one line:

```python
h_t = np.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b)
```

$W_{xh}$ has shape `(d_h, d_x)` and $W_{hh}$ has shape `(d_h, d_h)`, written as (output, input) the way a dense layer stores its weights. Because each row of `x_t` is one example, you transpose the weights. The usual mistake is `W_xh @ x_t`, which has the wrong shapes for a batch of rows.

Figure 2 follows the same weights over six steps, with the readings $1, 0.5$, and then zeros. The state climbs while readings arrive. Once they stop, the state shrinks by a factor a little under 0.8 per step: the cell is remembering, and slowly forgetting.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Bars for six input values and a line for the hidden state, which rises for two steps and then leaks back toward zero">
<defs></defs>
<line x1="70" y1="220" x2="570" y2="220" class="stroke-grid"/>
<text x="62" y="224" text-anchor="end" class="fig-small">0</text>
<line x1="70" y1="135" x2="570" y2="135" class="stroke-grid"/>
<text x="62" y="139" text-anchor="end" class="fig-small">0.5</text>
<line x1="70" y1="50" x2="570" y2="50" class="stroke-grid"/>
<text x="62" y="54" text-anchor="end" class="fig-small">1</text>
<rect x="102.7" y="50" width="18" height="170" class="fill-soft-a stroke-a"/>
<text x="111.7" y="240" text-anchor="middle" class="fig-small">step 1</text>
<rect x="186" y="135" width="18" height="85" class="fill-soft-a stroke-a"/>
<text x="195" y="240" text-anchor="middle" class="fig-small">step 2</text>
<text x="278.3" y="240" text-anchor="middle" class="fig-small">step 3</text>
<text x="361.7" y="240" text-anchor="middle" class="fig-small">step 4</text>
<text x="445" y="240" text-anchor="middle" class="fig-small">step 5</text>
<text x="528.3" y="240" text-anchor="middle" class="fig-small">step 6</text>
<polyline points="111.7,141.4 195,126.3 278.3,149.6 361.7,165.6 445,177.4 528.3,186.4" class="stroke-b"/>
<circle cx="111.7" cy="141.4" r="4.5" class="fill-b"/>
<circle cx="195" cy="126.3" r="4.5" class="fill-b"/>
<circle cx="278.3" cy="149.6" r="4.5" class="fill-b"/>
<circle cx="361.7" cy="165.6" r="4.5" class="fill-b"/>
<circle cx="445" cy="177.4" r="4.5" class="fill-b"/>
<circle cx="528.3" cy="186.4" r="4.5" class="fill-b"/>
<text x="125.7" y="159.4" class="fig-small">0.4621</text>
<text x="205" y="116.3" class="fig-small">0.5509</text>
<text x="528.3" y="176.4" text-anchor="middle" class="fig-small">0.1977</text>
<rect x="150" y="14" width="12" height="12" class="fill-soft-a stroke-a"/>
<text x="168" y="25" class="fig-small">input x</text>
<circle cx="256" cy="20" r="4.5" class="fill-b"/>
<text x="266" y="25" class="fig-small">hidden state h</text>
</svg>
Figure 2. The running example with weights 0.5 and 0.8, and inputs 1, 0.5, and then zeros. The state jumps up after the two non-zero inputs, then shrinks by a factor a little under 0.8 per step once the input stops: it remembers, and slowly forgets.
```

A whole sequence is stored time-major, as an array of shape `(T, n, d_x)` with $T$ steps and $n$ examples. Running the cell over it gives an array of shape `(T, n, d_h)`, where row 0 is $h_1$. You do not store $h_0$, because you already have it.

## One set of weights for any length

Return to the sensor, and ask what the padded window would cost. Suppose each reading has $d_x = 8$ features and you want 16 hidden units. A dense layer over a window of 30 readings has $30 \times 8 \times 16 + 16 = 3856$ weights, and a window of 60 roughly doubles that. A recurrent cell has $d_h(d_h + d_x + 1) = 16 \times 25 = 400$ weights: the $16 \times 16$ matrix $W_{hh}$, the $16 \times 8$ matrix $W_{xh}$, and 16 biases. The count is 400 for a run of 3 and 400 for a run of 3,000, because the copies in Figure 1 share everything.

Sharing is also what makes the model learn faster. A spike at step 3 and a spike at step 9 go through the same weights, so what the cell learns from one helps with the other. A window with separate weights per position has to learn the spike once per position, the problem that "Sliding one small filter over a picture" solved for images.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="An eight-step sequence read by a window of three steps and by a hidden state; the window cannot see step 2">
<defs><marker id="c085-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="20" y="24" class="fig-bold">A sequence of eight steps. The signal that matters is the spike at step 2.</text>
<rect x="70" y="40" width="56" height="34" class="fill-paper stroke-grid"/>
<text x="98" y="62" text-anchor="middle">0</text>
<text x="98" y="90" text-anchor="middle" class="fig-small">1</text>
<rect x="132" y="40" width="56" height="34" class="fill-soft-a stroke-a"/>
<text x="160" y="62" text-anchor="middle" class="fill-a fig-bold">1</text>
<text x="160" y="90" text-anchor="middle" class="fig-small">2</text>
<rect x="194" y="40" width="56" height="34" class="fill-paper stroke-grid"/>
<text x="222" y="62" text-anchor="middle">0</text>
<text x="222" y="90" text-anchor="middle" class="fig-small">3</text>
<rect x="256" y="40" width="56" height="34" class="fill-paper stroke-grid"/>
<text x="284" y="62" text-anchor="middle">0</text>
<text x="284" y="90" text-anchor="middle" class="fig-small">4</text>
<rect x="318" y="40" width="56" height="34" class="fill-paper stroke-grid"/>
<text x="346" y="62" text-anchor="middle">0</text>
<text x="346" y="90" text-anchor="middle" class="fig-small">5</text>
<rect x="380" y="40" width="56" height="34" class="fill-paper stroke-grid"/>
<text x="408" y="62" text-anchor="middle">0</text>
<text x="408" y="90" text-anchor="middle" class="fig-small">6</text>
<rect x="442" y="40" width="56" height="34" class="fill-paper stroke-grid"/>
<text x="470" y="62" text-anchor="middle">0</text>
<text x="470" y="90" text-anchor="middle" class="fig-small">7</text>
<rect x="504" y="40" width="56" height="34" class="fill-paper stroke-grid"/>
<text x="532" y="62" text-anchor="middle">0</text>
<text x="532" y="90" text-anchor="middle" class="fig-small">8</text>
<text x="20" y="118" class="fig-bold">Fixed window of 3</text>
<rect x="377" y="126" width="186" height="34" class="fill-soft-d stroke-d"/>
<text x="470" y="148" text-anchor="middle">sees steps 6, 7, 8</text>
<text x="70" y="148" class="fig-small">steps 1 to 5: out of view</text>
<text x="20" y="190" class="fig-bold">Hidden state</text>
<rect x="70" y="198" width="38" height="30" class="fill-soft-b stroke-b"/>
<text x="89" y="218" text-anchor="middle" class="fill-b fig-bold">h1</text>
<line x1="108" y1="213" x2="132" y2="213" class="stroke-b" marker-end="url(#c085-arrow-b)"/>
<rect x="132" y="198" width="38" height="30" class="fill-soft-b stroke-b"/>
<text x="151" y="218" text-anchor="middle" class="fill-b fig-bold">h2</text>
<line x1="170" y1="213" x2="194" y2="213" class="stroke-b" marker-end="url(#c085-arrow-b)"/>
<rect x="194" y="198" width="38" height="30" class="fill-soft-b stroke-b"/>
<text x="213" y="218" text-anchor="middle" class="fill-b fig-bold">h3</text>
<line x1="232" y1="213" x2="256" y2="213" class="stroke-b" marker-end="url(#c085-arrow-b)"/>
<rect x="256" y="198" width="38" height="30" class="fill-soft-b stroke-b"/>
<text x="275" y="218" text-anchor="middle" class="fill-b fig-bold">h4</text>
<line x1="294" y1="213" x2="318" y2="213" class="stroke-b" marker-end="url(#c085-arrow-b)"/>
<rect x="318" y="198" width="38" height="30" class="fill-soft-b stroke-b"/>
<text x="337" y="218" text-anchor="middle" class="fill-b fig-bold">h5</text>
<line x1="356" y1="213" x2="380" y2="213" class="stroke-b" marker-end="url(#c085-arrow-b)"/>
<rect x="380" y="198" width="38" height="30" class="fill-soft-b stroke-b"/>
<text x="399" y="218" text-anchor="middle" class="fill-b fig-bold">h6</text>
<line x1="418" y1="213" x2="442" y2="213" class="stroke-b" marker-end="url(#c085-arrow-b)"/>
<rect x="442" y="198" width="38" height="30" class="fill-soft-b stroke-b"/>
<text x="461" y="218" text-anchor="middle" class="fill-b fig-bold">h7</text>
<line x1="480" y1="213" x2="504" y2="213" class="stroke-b" marker-end="url(#c085-arrow-b)"/>
<rect x="504" y="198" width="38" height="30" class="fill-soft-b stroke-b"/>
<text x="523" y="218" text-anchor="middle" class="fill-b fig-bold">h8</text>
<text x="70" y="252">h is updated at every step, so step 2 can leave a trace in it.</text>
<text x="70" y="274" class="fig-small">Whether the trace survives to step 8 depends on the weights, the story of the next chapters.</text>
</svg>
Figure 3. A window of three steps cannot use step 2 when it predicts at step 8, whatever it learns. The hidden state has read every step, so the information is at least available to it.
```

Sharing does not rescue a window, which can only see so far back. In Figure 3, the signal that decides the label is a spike at step 2, but the window of three sees steps 6, 7, and 8. No weights, however well trained, can use a number the model never sees. The hidden state has read every step, so the information is at least available to it. How much of the spike survives to step 8 is a question about the weights, and the later chapters of this part spend their time on it.

## Giving the state a job

So far the state has no purpose. To train a cell, you read a result out of the state and score it. Read out the last state through a linear layer, $\hat y = h_T W_{hy}^{\top} + b_y$, and use the squared error

$$
L = \tfrac{1}{2}\sum (\hat y - y)^2
$$

Read it aloud: the prediction is a weighted copy of the final state, and the loss is half the squared gap to the target, added up over every example. The next chapter needs the gradient of $L$ with respect to each state, which this chapter calls $dH$. Because the loss reads only $h_T$, no other state feeds it directly, so every other state gets 0 from the loss itself and the last one gets $(\hat y - y)\,W_{hy}$. The earlier states still matter, because $h_T$ was built from them, and the next chapter shows how their share of the blame travels back.

Try it on the two steps above, with $W_{hy} = 2$, $b_y = 0$, and a target $y = 1$. The prediction is $\hat y = 2 \times 0.5509 = 1.1018$, so the gap is 0.1018 and $L = \tfrac{1}{2} \times 0.1018^2 = 0.0052$. The gradient on the last state is $0.1018 \times 2 = 0.2037$, and on the first state it is 0. The sign tells you that raising $h_2$ would make the loss larger, so training should lower it.

```python
err = h_seq[-1] @ W_hy.T + b_y - y
dH = np.zeros_like(h_seq)
dH[-1] = err @ W_hy
```

If the task scored every step, as in tagging every word of a sentence, `dH` would be nonzero at every step. The next chapter handles both.

> [!KEY]
> A recurrent cell is one set of weights applied at every step, carrying a fixed-size state. That solves the opening problem: a run of 3 readings and a run of 30 go through the same model, and the weight count stays the same.

> [!WARNING]
> The state has a fixed size, so it cannot hold everything it has read. What it keeps is decided by the weights, not by you.

> [!RECAP]
> - The hidden state is a fixed-size summary, updated at every step from the input and the previous state.
> - One set of weights, $W_{xh}$, $W_{hh}$, and $b$, serves every step, so the weight count is $d_h(d_h + d_x + 1)$ for any length.
> - Tanh keeps each entry of the state between −1 and 1.
> - Store the states $h_1$ through $h_T$ as an array of shape `(T, n, d_h)`, and do not store $h_0$.
> - To train the cell, read out the last state, score it, and keep the gradient on every state, which is zero except at the last step.

## Build it step by step

The checks call the functions in this order.

1. **`rnn_cell_forward(x_t, h_prev, W_xh, W_hh, b)`.** `x_t` is `(n, d_x)` and `h_prev` is `(n, d_h)`. Return the next state, `(n, d_h)`. Use `@`, `.T`, and `np.tanh`. The usual mistake is forgetting a transpose, which fails whenever `d_x` and `d_h` differ.
2. **`rnn_forward(x, h0, W_xh, W_hh, b)`.** `x` is `(T, n, d_x)`. Keep a running `h_prev` that starts at `h0`, call your cell at every step, and store each result in row `t` of an array of shape `(T, n, d_h)`. The usual mistake is storing `h0` in row 0.
3. **`last_step_loss_and_dH(h_seq, W_hy, b_y, y)`.** Compute the prediction from `h_seq[-1]`, return the loss as a float, and return `dH` with the shape of `h_seq`. Start `dH` as zeros and fill only the last row. The usual mistake is dropping the factor $W_{hy}$ when sending the gradient back through the readout.

## Cost and scaling

With $n$ examples, $T$ steps, input size $d_x$, and hidden size $d_h$, the costs grow as follows.

| Quantity | How it grows |
|---|---|
| Weights | $d_h(d_h + d_x + 1)$, the same for any $T$ |
| Work for one step | about $2 n d_h (d_h + d_x)$ operations |
| Work for the sequence | $T$ times one step |
| Memory for the states | grows with $n T d_h$ |

## Where this breaks

- Step $t$ needs the state from step $t - 1$, so the $T$ steps run one after another, however many processors you have.
- The state has a fixed size, so something must be forgotten. A reading that arrived long ago can fade to nothing. The chapter "Why gradients fade or blow up" shows how fast that happens.

## The exercise

You build `rnn_cell_forward`, `rnn_forward`, and `last_step_loss_and_dH`. The checks match the hand example, so the first two states come out as 0.4621 and 0.5509. They also compare the loop against a direct computation, check that reading a prefix of a sequence gives the first rows of the full run, and check `dH` against finite differences, which means nudging each state by a tiny amount and watching the loss.
