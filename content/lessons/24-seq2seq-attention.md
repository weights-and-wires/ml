---
title: Sequence to sequence with attention
slug: 24-seq2seq-attention
part: "Part 4 — Vision and sequences"
summary: "Why squeezing a source sequence into one vector fails, and the additive and multiplicative mechanisms that let a decoder look back."
minutes: 40
gpu: false
objectives:
  - State why a fixed-length encoder bottleneck loses information as the source sequence grows.
  - Derive Bahdanau's additive attention score and the context vector it produces.
  - Derive Luong's multiplicative attention and compare its parameter and flop cost with the additive form.
  - Explain attention as a differentiable dictionary lookup and name the query, key, and value roles.
prereqs:
  - 23-gated-recurrence
exercise: 24-seq2seq-attention
---

# Sequence to sequence with attention

An encoder-decoder built from the cells of the previous lesson reads a source
sequence with one recurrent network and keeps only its final hidden state. It
hands that single vector to a second recurrent network as the entire summary
of everything it must translate, answer, or continue. That final state has
fixed dimension $m$ regardless of whether the source has ten tokens or ten
thousand. The last two lessons were about fighting a *gradient* that decays
with sequence length. This design has a second, purely representational
problem: even a network with a perfect gradient can't recover from a
bottleneck that never had room to store the input in the first place.

A vector in $\mathbb{R}^m$ has a fixed information capacity, and a source
sequence of length $T_x$ drawn from a vocabulary of size $V$ carries up to
$T_x \log_2 V$ bits. Once that exceeds what $m$ floating-point numbers can
distinguish, no encoder, however well trained, can pack the whole sequence
into $h_{T_x}$ without losing some of it. The decoder then has no way to ask
for the piece it lost. The observed result is a length-degradation curve:
translation quality holds up on short sentences and falls off sharply past
some length the bottleneck was sized for. This happens well before either
RNN's gradient has had a chance to vanish. This lesson removes the bottleneck
by giving the decoder access to every encoder state, not only the last one.
It does so through a mechanism that is itself a small, trainable,
differentiable dictionary lookup.

## What you already know

- An encoder RNN or LSTM processes $x_1, \dots, x_{T_x}$ into hidden states
  $h_1, \dots, h_{T_x}$, and a decoder RNN produces $s_1, \dots, s_{T_y}$
  conditioned on a state derived from the encoder.
- The softmax function $\operatorname{softmax}(z)_k = e^{z_k} / \sum_j
  e^{z_j}$ maps a vector to a probability distribution.
- A convex combination $\sum_k \alpha_k v_k$ with $\alpha_k \geq 0$,
  $\sum_k \alpha_k = 1$ lies in the convex hull of the $v_k$.
- The chain rule differentiates a composition by multiplying local Jacobians
  in reverse order.
- $\tanh$ has derivative $1 - \tanh^2(z)$ and saturates toward $\pm 1$ for
  large $|z|$.

## Attention as a soft dictionary lookup

Give the decoder, at every output step $t$, access to the full list of
encoder states $h_1, \dots, h_{T_x}$ instead of a single summary. The
mechanism that decides how much of each $h_i$ to use is a differentiable
analogue of a dictionary lookup. It has a **query** $s_t$ (what the decoder
wants at this step) and a set of **keys** $h_i$ (what each stored item
advertises about itself). It also has a set of **values** $h_i$, or a
separate projection of it (what gets returned once a key matches). A hard lookup would
pick the single best-matching key. Attention replaces that hard choice with a
softmax over the match scores, returning a weighted blend of every value
instead of a single selection. This way the whole operation stays
differentiable, and the network never has to commit to a discrete decision
during training. This vocabulary — query, key, value — carries over unchanged
to the next lesson, where the query, keys, and values all come from the same
sequence.

## Additive attention

Bahdanau's construction scores the query against every key with a small
feedforward network. Project the query and each key into a shared space of
dimension $d$, add, squash, and take a learned dot product:

$$
e_{t,i} = v^{\top} \tanh(W_1 h_i + W_2 s_t), \qquad i = 1, \dots, T_x ,
$$

with $W_1 \in \mathbb{R}^{d \times d_h}$, $W_2 \in \mathbb{R}^{d \times
d_s}$, and $v \in \mathbb{R}^d$. The scalar $e_{t,i}$ is the model's learned
notion of how relevant source position $i$ is to decoder step $t$. Normalize
these scores over the source positions with a softmax,

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T_x} \exp(e_{t,j})} ,
$$

so $\alpha_{t,\cdot}$ is a probability distribution over source positions for
each decoder step separately. The context vector handed to the decoder at
step $t$ is the resulting weighted average of the encoder states,

$$
c_t = \sum_{i=1}^{T_x} \alpha_{t,i}\, h_i .
$$

Every quantity here is a smooth function of $W_1$, $W_2$, $v$, the encoder
states, and the decoder state, so $c_t$ trains by ordinary backpropagation
alongside the rest of the network. The decoder consumes $c_t$ concatenated
with its own state to produce the next output, in place of relying on a
single fixed vector carried across every step.

## Multiplicative attention

Luong's alternative drops the feedforward network and scores with a bilinear
form instead:

$$
e_{t,i} = s_t^{\top} W h_i \qquad \text{(general)}, \qquad\qquad
e_{t,i} = s_t^{\top} h_i \qquad \text{(dot, when } d_s = d_h \text{)} ,
$$

with weights only in the general form, $W \in \mathbb{R}^{d_s \times d_h}$.
Softmax normalization and the context vector are computed exactly as above.
Comparing costs for $T_x$ source positions and state dimension $d$: additive
attention evaluates $T_x$ feedforward passes through a hidden layer of width
$d$, costing $O(T_x d (d_s + d_h))$ flops and holding $d(d_s+d_h) + d$
parameters. The dot-product form costs $O(T_x d)$ flops for $T_x$ inner
products and holds no parameters at all. The general form sits between the
two, at $O(T_x d_s d_h)$ flops and $d_s d_h$ parameters. Dropping the hidden
layer and the nonlinearity is what buys multiplicative attention its speed:
a batch of matrix products runs faster than the same flop count spent on
$T_x$ separate $\tanh$ activations. This is why it became the more common
choice once the extra representational power of the additive form stopped
being worth its cost on most tasks.

Stacking $\alpha_{t,i}$ over every decoder step and source position gives a
$T_y \times T_x$ **alignment matrix**. Nothing forces this matrix to mean
anything: no supervision tells the model which source words correspond to
which output words. In practice, on tasks with a genuine positional or
lexical correspondence between input and output, the trained alignment
matrix traces that correspondence closely enough to read as a soft word
alignment. This is why attention weights became a standard diagnostic for
what a sequence model actually uses.

## What attention does not fix

Attention removes the fixed-length bottleneck: the decoder can reach any
encoder state directly, with no vector required to carry information further
than the softmax and the weighted sum. It does not remove the encoder's
recurrence. Computing $h_1, \dots, h_{T_x}$ still requires $T_x$ sequential
steps, because each $h_i$ depends on $h_{i-1}$ exactly as in the previous two
lessons. Attention fixes what the decoder can access, not how the encoder
computes what it makes available, and that sequential dependency is the wall
the next lesson's architecture removes entirely.

## Cost and scaling

For one decoder step against $T_x$ encoder states, with $d$ the shared
attention dimension:

| Route | Flops per decoder step | Parameters |
|---|---|---|
| Additive | $O(T_x \, d (d_s + d_h))$ | $d(d_s + d_h) + d$ |
| Multiplicative, general | $O(T_x \, d_s d_h)$ | $d_s d_h$ |
| Multiplicative, dot | $O(T_x \, d_h)$ | $0$ |

Over a full decode of $T_y$ steps, the cost multiplies by $T_y$, giving
$O(T_x T_y \, d)$ total attention flops for any of the three variants once
$d$ absorbs the state dimensions. This is the same quadratic-in-sequence-length
shape that reappears, unchanged in kind, when the next lesson lets a
sequence attend to itself.

## Where this breaks

**Coverage.** Nothing in the softmax constrains attention across decoder
steps. A source position can be attended to heavily at step $t$ and just as
heavily again at step $t+3$, with nothing in the objective penalizing it.
The typical symptom is repetition — the decoder emits the same phrase twice
because it kept returning to the same source content. Coverage mechanisms
address this by accumulating a running sum of past attention weights per
source position and feeding it back into the score function. Doing so costs
an extra state the decoder has to carry.

**Still quadratic, still sequential to build.** The $O(T_x T_y)$ attention
cost is cheap next to the bottleneck it replaces, but it's paid new at every
decoder step. The encoder that produces the keys and values is still
built one recurrent step at a time. The next lesson removes that by making
the encoder itself built entirely out of attention.

**No content the encoder never captured.** Attention only redistributes what
$h_1, \dots, h_{T_x}$ already contain. If a source token's relevant context
depended on information the encoder cell failed to carry into $h_i$, no
amount of attending to $h_i$ recovers information that was never encoded
there. That failure is the long-range gradient story from the last lesson.

## The exercise

You implement both attention mechanisms, their backward pass, and a decode
loop. The loop shows what a correctly aligned attention matrix looks like on
a task simple enough to construct by hand. Write `additive_attention(query,
keys, values, W1, W2, v)`, returning the context vector and the attention
weights. Write `multiplicative_attention(query, keys, values, W=None)`, which
uses the general bilinear form when `W` is given and falls back to the dot
product when it's `None`. Write `attention_backward(dcontext, cache)` for the
additive form, returning the gradient with respect to every argument of
`additive_attention`. Finally, write `decode_with_attention(source_tokens,
vocab_size)`, which constructs a fixed set of additive-attention weights by
hand, no learning involved, and runs a short greedy decode over a copy task.
In that task, a decoder at step $t$ attends almost entirely to source
position $t$ and reproduces the token found there.

The checks verify that both attention forms produce non-negative weights
that sum to one and a context that is a convex combination of the values.
They check `attention_backward` against `mlcheck.finite_difference`. They
confirm that additive and multiplicative attention agree once the additive
parameters are scaled and aimed to emulate a dot product. They also run the
copy task to confirm the decoded sequence exactly reproduces the source and
that the resulting alignment matrix sits within a numeric tolerance of the
identity matrix.
