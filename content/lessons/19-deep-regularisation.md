---
title: Regularisation in deep networks
slug: 19-deep-regularisation
part: "Part 3 — Neural networks"
summary: Dropout, weight decay, early stopping, and label smoothing, and why three of the four are the same shrinkage idea in disguise.
minutes: 45
gpu: false
objectives:
  - Implement inverted dropout, and explain why it needs no rescaling at test time.
  - Explain why weight decay only changes the effective learning rate once a weight feeds into a scale-invariant normalisation layer.
  - Derive the ridge penalty that a fixed number of gradient-descent iterations corresponds to, in the eigenbasis of a quadratic loss.
  - State what label smoothing changes about the loss function's minimum, and why double descent contradicts the classical capacity story.
prereqs:
  - 18-initialisation-and-norm
exercise: 19-deep-regularisation
---

# Regularisation in deep networks

Classical learning theory counts parameters to measure capacity, and
predicts that a model with more parameters than training examples memorises
the training set and generalises badly. A modern network routinely has far
more parameters than examples and generalises well anyway, so parameter
count was never the right way to think about what regularisation controls.
This lesson covers the four techniques that took over instead: dropout,
weight decay, early stopping, and label smoothing. Three of them turn out to
be the same shrinkage operation reached from different directions, and the
fourth changes what the loss function treats as a perfect answer.

Every result here reduces to something you already know: an expectation over
a random mask, a gradient-descent recursion split by eigenvector, a softmax
gradient. This lesson uses that background to show dropout collapsing to a
ridge penalty you can write down, early stopping collapsing to the same
penalty from a different knob, and why overparameterised networks generalise
well despite having room to memorise the data.

## What you already know

- Ridge regression shrinks each coordinate of the least-squares solution, in
  the eigenbasis of $X^{\top}X$, more for small eigenvalues than large ones
  (lesson 2).
- Weight decay and L2 regularisation give the same update under plain
  gradient descent, and diverge once the gradient is preconditioned
  (lesson 17).
- The softmax cross-entropy gradient with respect to the logits is
  $p - \text{onehot}(y)$, where $p = \operatorname{softmax}(z)$.
- Gradient descent on a quadratic loss updates each eigencoordinate
  independently, shrinking the distance to the optimum by a factor
  $(1-\eta\lambda_i)$ per step, where $\lambda_i$ is that direction's
  curvature.
- Cross-entropy between two distributions is never smaller than the target
  distribution's own entropy, with equality only when the two distributions
  match.

## Dropout

Dropout zeroes each unit of a layer independently with probability $p$
during training, then rescales the survivors. Writing $m_i \sim
\operatorname{Bernoulli}(1-p)$ for each unit and $q = 1-p$, the **inverted**
form computes

$$
y_i = \frac{m_i}{q}\,x_i .
$$

Because $\mathbb E[m_i] = q$, the expected activation under the random mask
equals the input exactly, $\mathbb E[y_i] = x_i$, with no approximation. That
is why inverted dropout needs no rescaling at test time: the test-time
forward pass, which returns $x$ unchanged, already computes what training
does on average for a single unit.

It is not, however, what training does on average through a whole nonlinear
network. Each forward pass samples one of many possible subnetworks, one per
mask, and the ideal test-time prediction would average the network's output
over all of them. Evaluating the network once at the expected input gives a
different answer whenever a nonlinearity sits downstream of the dropout
layer, so the test-time shortcut is exact only up to the first nonlinearity
and an approximation after that.

### Dropout as an adaptive ridge penalty

Apply dropout directly to a linear model's inputs and look at the expected
squared error over the mask distribution. Working through the expectation
term by term shows that dropout on a linear model's inputs is, in
expectation, ordinary least squares plus an extra penalty:

$$
\mathbb E_m\big[(y-w^{\top}\tilde x)^2\big] = (y-w^\top x)^2 + \frac{p}{q}\sum_i w_i^2 x_i^2 .
$$

This penalty is a ridge term that is not uniform across coordinates: each
weight $w_i$ is shrunk in proportion to $x_i^2$, so features that are large
or that vary a lot are penalised harder than features that barely move.
Standard ridge recovers this only when every feature shares the same scale.

## Weight decay under normalisation

Weight decay shrinks $\lVert w \rVert$ directly, and that shrinkage only
changes the function a layer computes when the output actually depends on
$\lVert w \rVert$. Once a weight feeds straight into a normalisation layer,
it stops depending on that scale at all: rescaling $w$ rescales the
pre-normalisation activation, and normalisation divides the scale straight
back out. A gradient step under this scale invariance leaves $\lVert w
\rVert$ nearly unchanged on its own, and it is weight decay's explicit
shrinkage that actually moves $\lVert w \rVert$. Because a smaller $\lVert w
\rVert$ makes the gradient's magnitude larger by the same factor, shrinking
$w$ rotates the direction that matters, $w/\lVert w \rVert$, faster for the
same nominal learning rate. Under a scale-invariant layer, weight decay is
not a penalty on the function computed at all — it is a knob on the
effective learning rate.

## Early stopping as ridge in disguise

Consider a quadratic loss whose Hessian has eigenvalues $\lambda_i$. Running
gradient descent from $w_0 = 0$ moves each eigencoordinate toward the
optimum geometrically, at rate $(1-\eta\lambda_i)$ per step, while ridge
regression on the same problem shrinks that coordinate by a fixed factor
that depends on a penalty $\gamma$. Setting the two shrinkage factors equal
after $k$ steps and solving for $\gamma$ gives

$$
\gamma(\lambda_i,\eta,k) = \lambda_i\,\frac{(1-\eta\lambda_i)^k}{1-(1-\eta\lambda_i)^k} .
$$

At $k=0$, $\gamma \to \infty$: zero steps is the same as an infinitely strong
penalty, and both give $w=0$. As $k \to \infty$, $\gamma \to 0$: training to
convergence is the same as no penalty at all. Stopping at a finite $k$
implements a ridge penalty of exactly this size, direction by direction —
and because $\gamma$ stays large longest for small $\lambda_i$, early
stopping penalises the low-curvature directions hardest, the same
directions gradient descent is slowest to move along in the first place.

## Label smoothing

Cross-entropy against a one-hot target has no finite minimiser in the
logits: pushing the target class's probability toward $1$ keeps lowering the
loss forever, which only stops once the logit gap between the target and
every other class is infinite. Label smoothing replaces the one-hot target
with a softened one, spreading a small amount $\varepsilon$ of probability
evenly across all $K$ classes:

$$
y^{\text{smooth}}_k = (1-\varepsilon)\,\mathbb 1[k=t] + \frac{\varepsilon}{K} .
$$

The same cross-entropy gradient identity, $p - q$, applies with
$q=y^{\text{smooth}}$ in place of the one-hot target. Because cross-entropy
against any fixed target is minimised exactly where the prediction matches
that target, the loss has a finite global minimum at $p_{\text{target}}
= 1-\varepsilon+\varepsilon/K$ instead of at $1$, so training stops rewarding
overconfidence once the model reaches that point.

## Data augmentation as an invariance prior

An augmentation is a transformation under which a label is known not to
change, such as a small rotation or crop for an image. Training on the
transformed input with the original label penalises the model for changing
its output along the direction that transformation moves the input, in the
same way a derivative penalty on that direction would. Augmentation encodes,
as data rather than as a formula, exactly which directions in the input the
task considers irrelevant.

## Double descent

The classical story predicts test risk falls, then rises, as capacity grows
past the point where a model can fit the training set exactly. Overparameterised
networks instead show test risk falling, spiking near that threshold, and
falling again beyond it — a second descent parameter-counting has no room
for.

The resolution is the same one lesson 2 gave for a design matrix with more
features than samples: past the threshold, infinitely many parameter
settings fit the training data exactly, and which one gradient descent finds
is a modelling choice, not a fact about the data. Near the threshold, that
choice is barely constrained, and small noise in the data swings the
selected solution wildly, which is the spike in risk. Well past the
threshold, gradient descent starting from a small initialisation keeps
finding the minimum-norm solution among the many that fit — and a
minimum-norm interpolant, unlike an arbitrary one, is exactly the kind of
solution capacity-counting intuition says should generalise.

## Cost and scaling

| Technique | Extra compute | Extra memory |
|---|---|---|
| Dropout | One random mask per call, $O(n)$ | One mask, $O(n)$ |
| Weight decay | $O(d)$, one multiply-add per parameter | None |
| Label smoothing | $O(NK)$, the cost of the loss itself | None |
| Early stopping | One validation pass per check | A checkpoint's worth of parameters |

None of these four changes the asymptotic cost of training measurably. The
price of regularisation here is almost entirely in tuning $p$, $\lambda$,
$\varepsilon$, or the stopping criterion, not in flops.

## Where this breaks

**Dropout combined with batch normalisation.** Both change behaviour between
training and test, and the two discrepancies compound rather than cancel,
producing a train-test accuracy gap larger than either technique's own gap.

**Weight decay without decoupling.** Everything above assumes the plain-SGD
equivalence with L2. Fold decay into an adaptive optimiser's gradient
instead, and the shrinkage becomes per-parameter and history-dependent,
which lesson 17 already showed can hurt generalisation.

**Early stopping on a noisy validation curve.** The criterion needs the
validation loss, which is a random variable. Stopping at its first local
uptick reacts to noise rather than the underlying trend, and stopping too
early gives up accuracy a slightly longer run would have recovered.

**Label smoothing before distillation.** A teacher trained with label
smoothing produces softened, less informative differences between
non-target class probabilities, which is exactly the signal a student model
needs when distilling from those probabilities. Smoothing a teacher can help
it train while making it a worse teacher.

## The exercise

You implement one representative from each family: dropout's forward and
backward pass, a weight-decay step, label smoothing's loss and gradient, and
the early-stopping-to-ridge correspondence as a formula you can evaluate
directly.

Write `dropout_forward(x, p, rng, training)`, using the inverted
formulation above, and `dropout_backward(dout, mask, p)`. Write
`weight_decay_step(w, grad, lr, weight_decay)`, applying a plain gradient
step with decay folded in exactly as lesson 17 established for SGD. Write
`label_smoothing_loss(logits, targets, eps)` and `label_smoothing_grad(logits,
targets, eps)`, following the softmax-gradient identity above. Write
`early_stopping_equivalent_ridge(eigenvalue, step, iterations)`, returning
$\gamma(\lambda, \eta, k)$ from the derivation above.

The checks confirm dropout's expected activation matches the input over many
independent draws, and that its backward pass matches a finite difference
computed with the mask held fixed. They confirm the label-smoothing gradient
against a finite difference too, and separately confirm that the loss is
lower, and the gradient is zero, at the logits that make $p$ equal to the
smoothed target than at logits pushed toward a one-hot prediction — the
check that would catch an implementation that smooths the loss but leaves
the true minimiser at a one-hot output. The last check is the point of the
lesson: running gradient descent on a diagonal quadratic for $k$ steps and
comparing the result, coordinate by coordinate, to the ridge solution that
`early_stopping_equivalent_ridge`'s $\gamma$ predicts for that same $k$.
