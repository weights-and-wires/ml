---
title: Sequence to sequence with attention
slug: 24-seq2seq-attention
part: "Part 4 — Vision and sequences"
summary: Why compressing a source sequence into one vector fails, and the additive and multiplicative mechanisms that let a decoder look back instead.
minutes: 50
gpu: false
objectives:
  - State the information-theoretic objection to a fixed-length encoder bottleneck and the length-degradation curve it produces.
  - Derive Bahdanau's additive attention score and the context vector it produces.
  - Derive Luong's multiplicative attention and compare its parameter and flop cost with the additive form.
  - Explain attention as a differentiable dictionary lookup and name the query, key, and value roles.
prereqs:
  - 23-gated-recurrence
exercise: 24-seq2seq-attention
---

# Sequence to sequence with attention

An encoder-decoder built from the cells of the previous lesson reads a source
sequence with one recurrent network, keeps only its final hidden state, and
hands that single vector to a second recurrent network as the entire summary
of everything it must translate, answer, or continue. That final state has
fixed dimension $m$ regardless of whether the source sequence has ten tokens
or ten thousand. Every architecture in the last two lessons was built to fight
a *gradient* that decays with sequence length; this design has a second, purely
representational problem: even a network with a perfect, undecayed gradient
cannot recover from a bottleneck that never had room to store the input in the
first place.

The objection is not empirical folklore. A vector in $\mathbb{R}^m$ has a fixed
information capacity, and a source sequence of length $T_x$ drawn from a
vocabulary of size $V$ carries up to $T_x \log_2 V$ bits. Once $T_x \log_2 V$
exceeds what $m$ floating-point numbers can distinguish at the working
precision, no encoder, however well trained, can pack the whole sequence into
$h_{T_x}$ without lossy compression, and the decoder has no way to ask for the
piece it lost. The observed consequence is a length-degradation curve: encoder-
decoder translation quality holds up on short sentences and falls off sharply
past some length the bottleneck was sized for, well before either RNN's
gradient has had a chance to vanish. This lesson removes the bottleneck by
giving the decoder access to every encoder state, not only the last one, through
a mechanism that is itself a small, trainable, differentiable dictionary lookup.

## What you already know

- An encoder RNN or LSTM processes $x_1, \dots, x_{T_x}$ into hidden states
  $h_1, \dots, h_{T_x}$, and a decoder RNN produces $s_1, \dots, s_{T_y}$
  conditioned on an initial state derived from the encoder.
- The softmax function $\operatorname{softmax}(z)_k = e^{z_k} / \sum_j e^{z_j}$
  maps a vector to a probability distribution and is invariant to adding a
  constant to every entry of $z$.
- A convex combination $\sum_k \alpha_k v_k$ with $\alpha_k \geq 0$,
  $\sum_k \alpha_k = 1$ lies in the convex hull of the $v_k$.
- The chain rule differentiates a composition by multiplying local Jacobians in
  reverse order.
- $\tanh$ has derivative $1 - \tanh^2(z)$ and saturates toward $\pm 1$ for large
  $|z|$.

## Attention as a soft dictionary lookup

Give the decoder, at every output step $t$, access to the full list of encoder
states $h_1, \dots, h_{T_x}$ instead of a single summary. The mechanism that
decides how much of each $h_i$ to use is a differentiable analogue of a
dictionary lookup: a **query** $s_t$ (what the decoder wants at this step), a set
of **keys** $h_i$ (what each stored item advertises about itself), and a set of
**values** $h_i$ or a separate projection of it (what gets returned once a key
matches). A hard lookup would pick the single best-matching key; attention
replaces the hard choice with a softmax over the match scores, giving a
weighted blend of every value instead of a single selection, so the whole
operation stays differentiable end to end and the network never has to commit
to a discrete decision during training. This vocabulary — query, key, value —
is introduced here because the next lesson reuses it unchanged when the query,
keys, and values all come from the same sequence.

## Additive attention

Bahdanau's construction scores the query against every key with a small
feedforward network rather than a fixed similarity function. Project the query
and each key into a shared space of dimension $d$, add, squash, and take a
learned dot product:

$$
e_{t,i} = v^{\top} \tanh(W_1 h_i + W_2 s_t), \qquad i = 1, \dots, T_x ,
$$

with $W_1 \in \mathbb{R}^{d \times d_h}$, $W_2 \in \mathbb{R}^{d \times d_s}$,
and $v \in \mathbb{R}^d$. The scalar $e_{t,i}$ is the model's learned notion of
how relevant source position $i$ is to decoder step $t$. Normalise these scores
across the source positions, for a fixed $t$, with a softmax:

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T_x} \exp(e_{t,j})} .
$$

Because the sum runs over $i$ and not over $t$, $\alpha_{t,\cdot}$ is a
probability distribution over source positions for each decoder step
separately — the decoder redistributes its attention independently at every
output. The context vector handed to the decoder at step $t$ is the resulting
weighted average of the encoder states:

$$
c_t = \sum_{i=1}^{T_x} \alpha_{t,i}\, h_i .
$$

Every quantity here is a smooth function of $W_1$, $W_2$, $v$, the encoder
states, and the decoder state, so $c_t$ is differentiable with respect to all
of them and trains by ordinary backpropagation alongside the rest of the
network. The decoder consumes $c_t$ concatenated with its own state to produce
the next output and the next decoder state, in place of relying on a single
fixed vector carried across every step.

## Multiplicative attention

Luong's alternative drops the feedforward network and scores with a bilinear
form instead:

$$
e_{t,i} = s_t^{\top} W h_i \qquad \text{(general)}, \qquad\qquad
e_{t,i} = s_t^{\top} h_i \qquad \text{(dot, when } d_s = d_h \text{)} ,
$$

with weights only in the general form, $W \in \mathbb{R}^{d_s \times d_h}$.
Softmax normalisation and the context vector are computed exactly as above.
Comparing costs for $T_x$ source positions and state dimension $d$: additive
attention evaluates $T_x$ feedforward passes through a hidden layer of width
$d$, costing $O(T_x d (d_s + d_h))$ flops and holding $d(d_s+d_h) + d$
parameters, dominated by $W_1$, $W_2$, and $v$; the dot-product form costs
$O(T_x d)$ flops for $T_x$ inner products and holds no parameters at all; the
general multiplicative form sits between the two, at $O(T_x d_s d_h)$ flops and
$d_s d_h$ parameters for $W$. Dropping the hidden layer and the nonlinearity is
what buys multiplicative attention its speed: on the hardware this course
targets, a batch of matrix products is faster than the same count of flops
spent evaluating $T_x$ separate $\tanh$ activations, so multiplicative
attention became the more common choice once the extra representational power
of the additive form stopped being worth its cost on most tasks.

### Reading the alignment matrix

Stacking $\alpha_{t,i}$ over every decoder step $t$ and source position $i$
gives a $T_y \times T_x$ **alignment matrix**. Nothing forces this matrix to
mean anything: it falls out of training on the end task with no supervision
telling the model which source words correspond to which output words. In
practice, on tasks with a genuine positional or lexical correspondence between
input and output, the alignment matrix trained this way turns out to trace
that correspondence closely enough to be read as a soft word alignment, which
is why attention weights became a standard diagnostic for what a sequence
model is actually using, independent of whatever the final output looks like.

## What attention does not fix

Attention removes the fixed-length bottleneck: the decoder can now reach any
encoder state directly, with no vector required to carry information further
than the softmax and the weighted sum. It does not remove the encoder's
recurrence. Computing $h_1, \dots, h_{T_x}$ still requires $T_x$ sequential
steps of whatever cell encodes the source, because each $h_i$ depends on
$h_{i-1}$ exactly as in the previous two lessons. Attention is a fix for what
the decoder can access, not for how the encoder computes what it makes
available, and the encoder's sequential dependency remains the wall the next
lesson's architecture is built to remove entirely.

## Cost and scaling

For one decoder step against $T_x$ encoder states, with $d$ the shared
attention dimension:

| Route | Flops per decoder step | Parameters |
|---|---|---|
| Additive | $O(T_x \, d (d_s + d_h))$ | $d(d_s + d_h) + d$ |
| Multiplicative, general | $O(T_x \, d_s d_h)$ | $d_s d_h$ |
| Multiplicative, dot | $O(T_x \, d_h)$ | $0$ |

Over a full decode of $T_y$ steps the cost multiplies by $T_y$, giving
$O(T_x T_y \, d)$ total attention flops for any of the three variants once $d$
absorbs the state dimensions — the same quadratic-in-sequence-length shape that
reappears, unchanged in kind, when the next lesson lets a single sequence
attend to itself.

## Where this breaks

**Coverage.** Nothing in the softmax constrains attention across decoder steps:
a source position can be attended to heavily at step $t$ and just as heavily
again at step $t+3$, with nothing in the objective penalising it. The typical
symptom is repetition — the decoder emits the same phrase twice because it kept
returning to the same source content. Coverage mechanisms address this by
accumulating a running sum of past attention weights per source position and
feeding that sum back into the score function, so the model can learn to
penalise attending again to what it has already used, at the cost of an extra
state the decoder has to carry and update.

**Still quadratic, still sequential to build.** The $O(T_x T_y)$ attention cost
is cheap next to the bottleneck it replaces, but it is paid new at every
decoder step, and the encoder that produces the keys and values it attends
over is still built one recurrent step at a time. Neither of the mechanisms in
this lesson removes that; the next lesson does, by making the encoder itself
built entirely out of attention.

**No content the encoder never captured.** Attention only redistributes what
$h_1, \dots, h_{T_x}$ already contain. If a source token's relevant context
depended on information the encoder cell failed to carry into $h_i$ — the usual
long-range gradient story from the last lesson — no amount of attending to
$h_i$ recovers information that was never encoded there.

## The exercise

You implement both attention mechanisms, their backward pass, and a decode
loop that demonstrates what a correctly aligned attention matrix looks like on
a task simple enough to construct by hand. Write `additive_attention(query,
keys, values, W1, W2, v)`, returning the context vector and the attention
weights, and `multiplicative_attention(query, keys, values, W=None)`, which
uses the general bilinear form when `W` is given and falls back to the dot
product when it is `None`. Write `attention_backward(dcontext, cache)` for the
additive form, returning the gradient with respect to every argument of
`additive_attention` given the cache it produces and an upstream gradient on
the context vector. Finally write `decode_with_attention(source_tokens,
vocab_size)`, which constructs a fixed set of additive-attention weights by
hand — no learning involved — and runs a short greedy decode over a copy task:
a decoder that, at step $t$, should attend almost entirely to source position
$t$ and reproduce the token found there.

The checks verify that both attention forms produce non-negative weights that
sum to one and a context that is a convex combination of the values, check
`attention_backward` against `mlcheck.finite_difference`, confirm that additive
and multiplicative attention agree once the additive parameters are scaled and
aimed to emulate a dot product, and run the copy task to confirm the decoded
sequence exactly reproduces the source and that the resulting alignment matrix
sits within a numeric tolerance of the identity matrix.
