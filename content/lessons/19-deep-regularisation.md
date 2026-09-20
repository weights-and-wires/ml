---
title: Regularisation in deep networks
slug: 19-deep-regularisation
part: "Part 3 — Neural networks"
summary: Dropout, weight decay, early stopping, and label smoothing derived as the same shrinkage idea in four different disguises.
minutes: 50
gpu: false
objectives:
  - Derive inverted dropout's expectation-preservation property, and its equivalence to an adaptive ridge penalty on a linear model.
  - Explain why weight decay only changes the effective learning rate once a weight feeds into a scale-invariant normalisation layer.
  - Derive the ridge penalty that a fixed number of gradient-descent iterations corresponds to, in the eigenbasis of a quadratic loss.
  - State the logit gap label smoothing targets, and why double descent contradicts the classical capacity story.
prereqs:
  - 18-initialisation-and-norm
exercise: 19-deep-regularisation
---

# Regularisation in deep networks

Classical learning theory measures capacity by counting parameters, and
predicts that a model with more parameters than training examples memorises
the training set and generalises badly. A modern network routinely has far
more parameters than examples and generalises well anyway, which means
parameter count was never the right proxy for what regularisation actually
controls. This lesson is about the four techniques that took over instead —
dropout, weight decay, early stopping, and label smoothing — and the fact that
three of them are the same shrinkage operation, derived from different
starting points, and the fourth changes what the loss function considers a
perfect answer.

Each derivation here reduces to algebra you already know: an expectation of a
random mask, a gradient-descent recursion diagonalised in an eigenbasis, a
softmax gradient. What is new is seeing dropout collapse to a ridge penalty
you can write down, seeing early stopping collapse to the same ridge penalty
with a different knob, and seeing why the mechanism that reconciles
overparameterisation with generalisation is the same minimum-norm argument
lesson 2 raised about the pseudoinverse.

## What you already know

- Ridge regression shrinks each coordinate of the least-squares solution in
  the eigenbasis of $X^{\top}X$, by a factor $\sigma_i/(\sigma_i^2+\lambda)$
  in place of OLS's $1/\sigma_i$ (lesson 2).
- Weight decay and L2 regularisation are the same update for plain gradient
  descent, and diverge once the gradient is preconditioned (lesson 17).
- The softmax cross-entropy gradient with respect to the logits is $p -
  \text{onehot}(y)$, where $p=\operatorname{softmax}(z)$.
- Gradient descent on a quadratic loss updates each eigencoordinate
  independently, by a factor $(1-\eta\lambda_i)$ per step, where $\lambda_i$
  is that direction's curvature.
- Gibbs' inequality: for distributions $p,q$, cross-entropy
  $H(q,p) = H(q) + \operatorname{KL}(q\Vert p) \ge H(q)$, with equality iff
  $p=q$.

## Dropout

Dropout zeroes each unit of a layer independently with probability $p$ during
training, and rescales the survivors. Writing $m_i \sim
\operatorname{Bernoulli}(1-p)$ independently for each unit and $q=1-p$, the
**inverted** form computes

$$
y_i = \frac{m_i}{q}\,x_i .
$$

Because $\mathbb E[m_i]=q$, $\mathbb E[y_i] = \frac{q}{q}x_i = x_i$: the
expected activation under the training-time random mask equals the
deterministic input exactly, for every unit, with no approximation. That is
why inverted dropout needs no rescaling at test time — the test-time forward
pass, which returns $x$ unchanged, already computes what training does on
average.

It is not, however, exactly what training does on average through a
*nonlinear* layer. Each forward pass samples one of $2^n$ subnetworks (one per
setting of the mask), and the ideal test-time prediction would average the
network's output over all of them, $\mathbb E_m[f(m\odot x/q)]$. Evaluating
$f$ once at the expected input, $f(\mathbb E_m[m\odot x/q]) = f(x)$, is a
different quantity whenever $f$ is nonlinear — the two coincide for a linear
$f$, by linearity of expectation, and only approximate each other once a
nonlinearity sits between the dropout layer and the output. This is the
approximation the weight-scaling shortcut makes: exact for the first moment
of a linear readout, approximate for everything downstream of a nonlinearity.

### Dropout as an adaptive ridge penalty

Apply dropout directly to the inputs of a linear model, $\tilde x = m\odot
x/q$, and consider the expected squared error over the mask distribution,

$$
\mathbb E_m\big[(y - w^{\top}\tilde x)^2\big] .
$$

Expand the square: $\mathbb E_m[(y-w^\top\tilde x)^2] = y^2 - 2y\,w^\top
\mathbb E[\tilde x] + w^\top \mathbb E[\tilde x\tilde x^\top] w$. The linear
term uses $\mathbb E[\tilde x]=x$, matching plain least squares. For the
quadratic term, independence across coordinates gives $\mathbb E[\tilde
x_i\tilde x_j] = x_ix_j$ for $i\ne j$, while for $i=j$, using
$m_i^2=m_i$ for a Bernoulli variable, $\mathbb E[\tilde x_i^2] =
\mathbb E[m_i]x_i^2/q^2 = x_i^2/q$. So

$$
\mathbb E[\tilde x\tilde x^\top] = xx^\top + \operatorname{diag}\!\Big(x_i^2\Big(\frac1q-1\Big)\Big)
= xx^\top + \frac{p}{q}\operatorname{diag}(x_i^2) ,
$$

and the expected objective is

$$
\mathbb E_m\big[(y-w^\top\tilde x)^2\big] = (y-w^\top x)^2 + \frac{p}{q}\sum_i w_i^2 x_i^2 .
$$

Dropout on a linear model's inputs is, in expectation, ordinary least squares
plus a ridge penalty that is not uniform: each coefficient $w_i$ is shrunk in
proportion to $x_i^2$, so features that are large or that vary a lot in the
data are penalised harder than features that barely move. Standard ridge
recovers a special case of this only when every feature has the same scale.

## Weight decay under normalisation

Weight decay shrinks $\lVert w\rVert$ directly. That shrinkage changes the
function a layer computes only when the layer's output actually depends on
$\lVert w\rVert$ — and it stops depending on it once the layer feeds into a
normalisation layer. If $w$'s output is immediately standardised, the network
computes a function $f$ with $f(cw) = f(w)$ for every $c>0$: rescaling $w$
rescales the pre-normalisation activation, and the normalisation divides that
scale straight back out.

Differentiate $f(cw)=f(w)$ with respect to $c$ at $c=1$: $w^\top\nabla
f(w) = 0$ — the gradient is orthogonal to $w$ under scale invariance, an
instance of Euler's homogeneous-function identity. Differentiate the same
relation with respect to $w$ to get $\nabla f(cw) = \frac1c \nabla f(w)$: the
gradient's magnitude scales as $1/\lVert w\rVert$ for a scale-invariant $f$.
A plain gradient step, being orthogonal to $w$, leaves $\lVert w\rVert$ nearly
unchanged to first order; weight decay's explicit $w\leftarrow w(1-\eta\lambda)$
is what actually drives $\lVert w\rVert$'s evolution here. Shrinking $\lVert
w\rVert$ makes $\lVert\nabla f\rVert$ larger by the same factor, so the next
step rotates $w/\lVert w\rVert$ — the only thing $f$ can see — by more,
for the same nominal $\eta$. Weight decay under a scale-invariant layer is not
a penalty on the function computed at all; it is a knob on $\lVert w\rVert$,
and through $\lVert w\rVert$, on the effective learning rate applied to the
direction that matters.

## Early stopping as ridge in disguise

Consider a quadratic loss $L(w) = \tfrac12(w-w^\star)^\top H(w-w^\star)$, with
$H=Q\Lambda Q^\top$. Gradient descent from $w_0=0$ gives
$w_{k+1}-w^\star = (I-\eta H)(w_k-w^\star)$; in the eigenbasis, writing $c_i$
for the $i$-th eigencoordinate of $w^\star$, each coordinate evolves
independently as $a_{k,i} = c_i\big(1-(1-\eta\lambda_i)^k\big)$, starting at
$a_{0,i}=0$ and approaching $c_i$ as $k\to\infty$ whenever $|1-\eta\lambda_i|<1$.

Ridge regression on the same problem shrinks the same coordinate to
$a_{\text{ridge},i} = \dfrac{\lambda_i}{\lambda_i+\gamma}\,c_i$. Setting the
two equal and solving for $\gamma$ as a function of $k$,

$$
1-(1-\eta\lambda_i)^k = \frac{\lambda_i}{\lambda_i+\gamma}
\quad\Longrightarrow\quad
\gamma(\lambda_i,\eta,k) = \lambda_i\,\frac{(1-\eta\lambda_i)^k}{1-(1-\eta\lambda_i)^k} .
$$

At $k=0$, $\gamma\to\infty$: zero steps is the same as an infinitely strong
penalty, and both give $w=0$. As $k\to\infty$, $\gamma\to0$: training to
convergence is the same as no penalty at all. Stopping at a finite $k$
implements a ridge penalty of exactly this size, direction by direction —
which is also why early stopping penalises the low-curvature directions
hardest: for small $\lambda_i$, $\gamma$ stays large for far longer, so the
directions gradient descent is slowest to move along are the ones it
regularises most.

## Label smoothing

Cross-entropy against a one-hot target has no finite minimiser in the
logits: pushing $p_{\text{target}}\to1$ keeps lowering the loss, which is
answered only by driving the logit gap between the target class and every
other class to infinity. Label smoothing replaces the one-hot target with

$$
y^{\text{smooth}}_k = (1-\varepsilon)\,\mathbb 1[k=t] + \frac{\varepsilon}{K} ,
$$

for $K$ classes. For any target distribution $q$ with $\sum_k q_k=1$, the
cross-entropy gradient with respect to the logits is $\partial(-\sum_k q_k
\log p_k)/\partial z_j = p_j - q_j$ — the same identity you already know for
$q=\text{onehot}(y)$, applied to $q=y^{\text{smooth}}$. By Gibbs' inequality,
$H(y^{\text{smooth}}, p) \ge H(y^{\text{smooth}})$ with equality exactly when
$p=y^{\text{smooth}}$, so the loss has a finite global minimum at
$p_{\text{target}} = 1-\varepsilon+\varepsilon/K$ instead of at $1$. Inverting
the softmax at that point gives the logit gap label smoothing targets: with
the remaining probability spread evenly over the other $K-1$ classes,

$$
z_{\text{target}} - z_{\text{other}} = \log\frac{1-\varepsilon+\varepsilon/K}{\varepsilon/K} ,
$$

a finite number instead of the unbounded gap plain cross-entropy asks for.
Training stops rewarding overconfidence once the model reaches this gap,
rather than only when gradient noise happens to stop it.

## Data augmentation as an invariance prior

An augmentation is a transformation $t$ under which the label is known not to
change — a small rotation, crop, or colour jitter for an image. Training on
$t(x)$ with the original label $y$ is empirical risk minimisation over an
enlarged, correlated dataset, and in the limit of small transformations it
penalises the loss's directional derivative along the tangent of $t$ at $x$: a
model that fits both $x$ and $t(x)$ to the same label has approximately zero
directional derivative of its output along that tangent direction, the same
effect a Tikhonov penalty on that derivative would have if you wrote it down
directly. Augmentation encodes, as data rather than as a penalty term, exactly
which directions in input space the task considers irrelevant.

## Double descent

The classical story predicts test risk falls, then rises, as capacity grows
past the point where the model can fit the training set exactly — the
**interpolation threshold**. Overparameterised networks instead show test risk
falling, rising sharply near the threshold, and falling again beyond it: a
second descent the parameter-counting story has no room for.

The resolution is the same one lesson 2 gave for a design matrix with more
features than samples: past the interpolation threshold, infinitely many
parameter settings fit the training data exactly, and which one gradient
descent finds is a modelling choice, not a consequence of the data. Near the
threshold, that choice is close to unconstrained — the solution set is a
low-dimensional, poorly conditioned sliver, and small noise in the data swings
the selected solution wildly, which is the spike in risk. Well past the
threshold, the solution set is large and gradient descent from a small
initialisation keeps finding the minimum-norm member of it, the same object
$X^+y$ was in lesson 2 — and a minimum-norm interpolant, unlike an arbitrary
one, is exactly the kind of solution capacity-counting intuition says should
generalise.

## Cost and scaling

| Technique | Extra compute | Extra memory |
|---|---|---|
| Dropout | one random mask per call, $O(n)$ | one mask, $O(n)$ |
| Weight decay | $O(d)$, one multiply-add per parameter | none |
| Label smoothing | $O(NK)$, the cost of the loss itself | none |
| Early stopping | one validation pass per check, $O(N_{\text{val}}d)$ | a checkpoint's worth of parameters |

None of the four changes the asymptotic cost of training measurably; the
price of regularisation here is almost entirely in tuning $p$, $\lambda$,
$\varepsilon$, or the stopping criterion, not in flops.

## Where this breaks

**Dropout combined with batch normalisation.** Dropout's train-time scaling
and batch norm's train-time batch statistics both behave differently at test
time, and the two discrepancies compound rather than cancel — a documented
source of a train-test accuracy gap distinct from either technique's gap
alone.

**Weight decay without decoupling, again.** Everything this lesson says about
decay assumes the plain-SGD equivalence with L2; fold it into an adaptive
optimiser's gradient instead of decoupling it, and the shrinkage becomes
per-parameter and history-dependent in the way lesson 17 already showed
degrades generalisation.

**Early stopping on a noisy validation curve.** The criterion needs the
validation loss itself, which is a random variable; stopping at its first
local uptick reacts to noise, not to the underlying trend, and stopping too
early costs exactly the bias a slightly longer run would have removed.

**Label smoothing before distillation.** A teacher trained with label
smoothing produces softened, less informative differences between non-target
class probabilities, which is precisely the signal a student model
distilling from those probabilities needs — smoothing a teacher can make it
train, but make it a worse teacher.

## The exercise

You implement one representative from each family: dropout's forward and
backward pass, a weight-decay step, label smoothing's loss and gradient, and
the early-stopping-to-ridge correspondence as a formula you can evaluate
directly.

Write `dropout_forward(x, p, rng, training)`, using the inverted formulation
above, and `dropout_backward(dout, mask, p)`. Write `weight_decay_step(w,
grad, lr, weight_decay)`, applying a plain gradient step with decay folded in
exactly as lesson 17 established for SGD. Write `label_smoothing_loss(logits,
targets, eps)` and `label_smoothing_grad(logits, targets, eps)`, following the
softmax-gradient identity above. Write `early_stopping_equivalent_ridge(
eigenvalue, step, iterations)`, returning $\gamma(\lambda,\eta,k)$ from the
derivation above.

The checks confirm dropout's expected activation matches the input over many
independent draws, and that its backward pass matches a finite difference
computed with the mask held fixed. They confirm the label-smoothing gradient
against a finite difference too, and separately confirm that the loss is
lower, and the gradient is zero, at the logits that make $p$ equal to the
smoothed target than at logits pushed toward a one-hot prediction — the check
that would catch an implementation that quietly smooths the loss but leaves
the true minimiser at $p=\text{onehot}(y)$. The last check is the point of
the lesson: running gradient descent on a diagonal quadratic for $k$ steps and
comparing the result, coordinate by coordinate, to the ridge solution that
`early_stopping_equivalent_ridge`'s $\gamma$ predicts for that same $k$.
