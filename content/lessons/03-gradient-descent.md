---
title: Gradient descent
slug: 03-gradient-descent
part: "Part 1 — Foundations"
summary: The descent lemma, the fixed-step rate in the condition number, and how momentum improves it from kappa to sqrt(kappa).
minutes: 60
gpu: false
objectives:
  - State smoothness and strong convexity precisely, and derive the descent lemma from smoothness alone.
  - Derive the fixed-step contraction rate for gradient descent on a quadratic, and identify the condition number as the iteration count.
  - Derive the heavy-ball and Nesterov updates, and explain why their rate improves from a kappa dependence to a sqrt(kappa) dependence.
  - State which rate holds for which method, under which assumption, and what changes once the objective is not a quadratic.
exercise: 03-gradient-descent
prereqs:
  - 02-linear-regression
---

# Gradient descent

Lesson 2 solved least squares in closed form, and the closed form does not
generalise: the moment the objective is not a quadratic in a form you can
factor directly — a logistic loss, a deep network, an $\ell_1$ penalty — there
is no normal equation to write down. What remains is the gradient, and the
question this lesson answers is how fast following it gets you to the answer,
and how that answer depends on the shape of the objective around its minimum.

The headline results are stated as bounds most optimisation courses assert and
this one derives: the fixed-step rate for gradient descent, and the two
tricks — heavy-ball momentum and Nesterov acceleration — that improve the
iteration count's dependence on the condition number from linear to square
root. The precision matters, because these bounds are exactly what you tune
against every time you set a learning rate later in this course.

## What you already know

- A twice-differentiable $f$ is convex iff $\nabla^2 f(w) \succeq 0$ everywhere.
- The eigenvalues of a symmetric matrix bound its quadratic form:
  $\lambda_{\min}(M) \lVert v \rVert^2 \leq v^\top M v \leq \lambda_{\max}(M) \lVert v \rVert^2$.
- Least squares is the quadratic $L(w) = \tfrac{1}{2}\lVert Xw - y \rVert^2$
  with gradient $X^\top(Xw - y)$ and Hessian $X^\top X$.
- The Cauchy-Schwarz inequality: $\lvert u^\top v \rvert \leq \lVert u \rVert \lVert v \rVert$.
- A first-order Taylor expansion with a remainder:
  $f(v) = f(w) + \nabla f(w)^\top(v - w) + \int_0^1 \dots$, which the descent
  lemma below turns into a usable inequality.

## Smoothness and strong convexity

Two constants govern every rate in this lesson. $f$ is **$L$-smooth** when its
gradient is $L$-Lipschitz:

$$
\lVert \nabla f(u) - \nabla f(v) \rVert \leq L \lVert u - v \rVert \quad \text{for all } u, v .
$$

For twice-differentiable $f$ this is equivalent to $\nabla^2 f(w) \preceq LI$
everywhere. $f$ is **$\mu$-strongly convex** when

$$
f(v) \geq f(u) + \nabla f(u)^\top (v - u) + \frac{\mu}{2}\lVert v - u \rVert^2 \quad \text{for all } u, v ,
$$

equivalent, for twice-differentiable $f$, to $\nabla^2 f(w) \succeq \mu I$.
Strong convexity is a quantitative sharpening of convexity: it says $f$ lies
above not just its tangent plane but a tangent paraboloid, which is what will
let a contraction argument go through. For the quadratic
$f(w) = \tfrac{1}{2}w^\top A w - b^\top w$ with $A$ symmetric, $\nabla^2 f = A$
everywhere, so $L = \lambda_{\max}(A)$ and $\mu = \lambda_{\min}(A)$ exactly —
a quadratic is the case where these constants are not bounds but the truth.
The ratio $\kappa = L / \mu \geq 1$ is the **condition number** of the
problem; it is the same quantity as $\kappa_2(X)^2$ from lesson 2 when
$A = X^\top X$.

## The descent lemma

$L$-smoothness alone — no convexity needed — bounds how much $f$ can rise
above its linear approximation. Starting from
$f(v) - f(u) = \int_0^1 \nabla f(u + t(v-u))^\top (v - u)\, dt$, add and
subtract $\nabla f(u)^\top(v-u)$ inside the integral:

$$
f(v) - f(u) - \nabla f(u)^\top(v - u) = \int_0^1 \big(\nabla f(u + t(v-u)) - \nabla f(u)\big)^\top (v-u)\, dt .
$$

Bound the integrand with Cauchy-Schwarz and then $L$-smoothness:

$$
\big|\big(\nabla f(u+t(v-u)) - \nabla f(u)\big)^\top(v-u)\big|
\leq \lVert \nabla f(u+t(v-u)) - \nabla f(u) \rVert \, \lVert v - u \rVert
\leq L t \lVert v - u \rVert^2 .
$$

Integrating $Lt\lVert v-u\rVert^2$ over $t \in [0,1]$ gives
$\tfrac{L}{2}\lVert v - u \rVert^2$, so

$$
f(v) \leq f(u) + \nabla f(u)^\top (v - u) + \frac{L}{2} \lVert v - u \rVert^2 . \tag{descent lemma}
$$

Set $v = u - \tfrac{1}{L}\nabla f(u)$, the gradient step with step size
$1/L$. The right side becomes $f(u) - \tfrac{1}{2L}\lVert \nabla f(u)
\rVert^2$, so

$$
f(u - \tfrac{1}{L}\nabla f(u)) \leq f(u) - \frac{1}{2L}\lVert \nabla f(u) \rVert^2 .
$$

Every step of size $1/L$ decreases $f$ by at least a term proportional to the
squared gradient norm — the sole fact that makes gradient descent work at all,
and it needed nothing but smoothness.

## The fixed-step rate on a quadratic

Strong convexity turns the descent guarantee into a rate. For the quadratic
$f(w) = \tfrac{1}{2}w^\top A w - b^\top w$ with minimiser $w^\star = A^{-1}b$
(so $\nabla f(w) = A(w - w^\star)$), a fixed-step update
$w_{k+1} = w_k - \alpha \nabla f(w_k)$ becomes

$$
w_{k+1} - w^\star = w_k - w^\star - \alpha A (w_k - w^\star) = (I - \alpha A)(w_k - w^\star) .
$$

$A$ is symmetric, so diagonalise $A = Q \Lambda Q^\top$ with
$\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_d)$,
$\mu \leq \lambda_i \leq L$. In the rotated coordinates
$z_k = Q^\top(w_k - w^\star)$, each coordinate evolves independently:

$$
(z_{k+1})_i = (1 - \alpha \lambda_i)(z_k)_i \implies (z_k)_i = (1 - \alpha\lambda_i)^k (z_0)_i .
$$

Since $Q$ is orthogonal it preserves norms, so $\lVert w_k - w^\star \rVert =
\lVert z_k \rVert$, and

$$
\lVert w_k - w^\star \rVert \leq \Big(\max_i \lvert 1 - \alpha \lambda_i \rvert\Big)^k \lVert w_0 - w^\star \rVert .
$$

The contraction factor $\max_i \lvert 1 - \alpha\lambda_i \rvert$ is minimised
over $\alpha$ by balancing its value at the two extreme eigenvalues,
$\lvert 1 - \alpha\mu \rvert = \lvert 1 - \alpha L \rvert$, giving
$\alpha^\star = 2/(\mu + L)$ and a factor of $(\kappa - 1)/(\kappa+1)$. The
simpler and more common choice $\alpha = 1/L$ gives contraction factor
$1 - \mu/L$ directly, since then $1 - \alpha L = 0$ and $1 - \alpha\mu = 1 -
\mu/L$ is the largest term in magnitude whenever $\mu \le L$. Either way,

$$
\lVert w_k - w^\star \rVert \leq \left(1 - \frac{\mu}{L}\right)^{k} \lVert w_0 - w^\star \rVert
$$

for step $\alpha = 1/L$. This is linear convergence: the error shrinks by a
constant factor every step. To reach $\lVert w_k - w^\star\rVert \leq
\varepsilon \lVert w_0 - w^\star \rVert$, take logarithms:

$$
k \geq \frac{\log(1/\varepsilon)}{-\log(1 - \mu/L)} \approx \kappa \log(1/\varepsilon) ,
$$

using $-\log(1-x) \geq x$. The condition number is, up to the constant
$\log(1/\varepsilon)$, the iteration count. A well-conditioned quadratic
($\kappa$ near 1) converges in a handful of steps; an ill-conditioned one
needs on the order of $\kappa$ of them. Beyond $\alpha = 2/L$, the factor
$\lvert 1 - \alpha L \rvert$ exceeds 1 and the iterates diverge along the
top-eigenvalue direction, regardless of how small $\mu$ is — smoothness alone
sets the stability boundary.

## Exact line search

Instead of fixing $\alpha$, choose the step that minimises $f$ exactly along
the current gradient direction: $\alpha_k = \operatorname*{arg\,min}_{\alpha}
f(w_k - \alpha \nabla f(w_k))$. For the quadratic, write $g = \nabla f(w_k)$
and expand $f(w_k - \alpha g) = f(w_k) - \alpha g^\top g + \tfrac{1}{2}\alpha^2
g^\top A g$, a quadratic in the scalar $\alpha$ with a positive leading
coefficient. Setting its derivative to zero,
$-g^\top g + \alpha\, g^\top A g = 0$, gives

$$
\alpha_k = \frac{g^\top g}{g^\top A g} .
$$

Exact line search removes the need to know $L$ and $\mu$ in advance, and on a
quadratic it never does worse per-step than the best fixed step — but it does
not remove the $\kappa$ dependence: on the same ill-conditioned quadratic it
zig-zags across the narrow valley one eigen-direction at a time, and the
worst-case rate is again governed by $(\kappa-1)/(\kappa+1)$. Buying a better
step each iteration is not the same as buying a better rate.

## Heavy-ball momentum

Fix the actual problem instead: the rate is $\kappa$ because the update only
uses the current gradient, memoryless. Polyak's heavy-ball method adds inertia:

$$
w_{k+1} = w_k - \alpha \nabla f(w_k) + \beta (w_k - w_{k-1}) .
$$

On the quadratic, in the same rotated coordinates as before, each coordinate
$i$ now obeys a linear recurrence of order two,
$(z_{k+1})_i = (1 - \alpha\lambda_i)(z_k)_i + \beta\big((z_k)_i - (z_{k-1})_i\big)$,
whose characteristic equation is $t^2 - (1 + \beta - \alpha\lambda_i) t + \beta
= 0$. Choosing $\alpha$ and $\beta$ so that this quadratic has a repeated root
for both extreme eigenvalues $\mu$ and $L$ simultaneously — the standard
derivation, omitted here for length — gives the optimal choice

$$
\alpha^\star = \left(\frac{2}{\sqrt{L} + \sqrt{\mu}}\right)^2, \qquad
\beta^\star = \left(\frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}}\right)^2 ,
$$

with both roots of the characteristic equation equal to
$\rho = (\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$ in magnitude at $\lambda = \mu$.
The per-step contraction factor becomes $\rho$ instead of $1 - 1/\kappa
\approx (\kappa-1)/\kappa$, and by the same logarithm argument as before the
iteration count to reach accuracy $\varepsilon$ scales as
$\sqrt{\kappa}\log(1/\varepsilon)$ rather than $\kappa \log(1/\varepsilon)$.
Momentum buys a square root in the condition number, which for $\kappa =
10{,}000$ is the difference between $10{,}000$ steps and $100$.

## Nesterov acceleration

Heavy-ball's guarantee is specific to quadratics; for general smooth strongly
convex functions it is not known to be globally convergent at the accelerated
rate, and can cycle. Nesterov's method achieves the $\sqrt{\kappa}$ rate with
a guarantee that holds for every $L$-smooth $\mu$-strongly convex function, by
evaluating the gradient at an extrapolated point rather than at the current
iterate:

$$
y_k = w_k + \beta (w_k - w_{k-1}), \qquad w_{k+1} = y_k - \alpha \nabla f(y_k) ,
$$

with $\alpha = 1/L$ and $\beta = (\sqrt{L}-\sqrt{\mu})/(\sqrt{L}+\sqrt{\mu})$
in the strongly convex case. The one-line intuition, made rigorous by
Nesterov's estimate-sequence technique, is that $y_k$ looks ahead using the
momentum term before the gradient is taken, so the gradient step corrects for
where the trajectory is heading rather than only where it stands —
gradient descent reacts, Nesterov's method anticipates. The estimate-sequence
proof constructs a sequence of quadratic lower bounds $\phi_k$ on $f$ that
$w_k$ minimises with a controlled gap to $f(w_k)$, shrinking geometrically at
rate $\sqrt{\kappa}$; the construction is the technical heart of the method
and is not repeated here.

## What goes wrong away from quadratics

Every rate above that mentions a single number $\kappa$ used the fact that a
quadratic's Hessian is constant, so $L$ and $\mu$ are the same everywhere.
For a general $L$-smooth, $\mu$-strongly convex function, the same rates hold
for gradient descent and for Nesterov's method — the descent lemma and strong
convexity inequality were stated for general $f$ — but the heavy-ball
analysis above used exact diagonalisation of a fixed $A$ and does not extend;
heavy-ball's global rate on non-quadratic strongly convex objectives is an
open problem in general, and momentum used naively can genuinely overshoot
and oscillate on curved objectives where the local quadratic approximation
changes quickly between iterations. Away from convexity entirely, none of
these rates say anything about reaching a global minimum at all — gradient
descent converges to a stationary point, and the loss landscapes in later
lessons are not convex.

## Cost and scaling

| Method | Per-step cost | Iterations for accuracy $\varepsilon$ | Extra state |
|---|---|---|---|
| Fixed-step gradient descent | One gradient, $O(\text{grad cost})$ | $O(\kappa \log(1/\varepsilon))$ | None |
| Exact line search (quadratic) | One gradient, one matrix-vector product | $O(\kappa \log(1/\varepsilon))$ | None |
| Heavy ball | One gradient | $O(\sqrt{\kappa} \log(1/\varepsilon))$ | One previous iterate |
| Nesterov | One gradient, at an extrapolated point | $O(\sqrt{\kappa} \log(1/\varepsilon))$ | One previous iterate |

The per-step cost of every method here is dominated by one gradient
evaluation; momentum's entire cost is one extra vector of memory and two
scalar multiplies. That is why it is the default rather than an exotic
option: for the price of storing $w_{k-1}$, the iteration count's dependence
on $\kappa$ drops from linear to square root, and $\kappa$ is frequently in
the thousands for the problems this course builds up to.

## Where this breaks

**Overshooting the smoothness constant.** Every rate above assumes $\alpha
\leq 2/L$ for gradient descent, or the matching bound for momentum. Since $L$
is rarely known exactly, practitioners estimate it or use a line search;
guessing too large a step does not degrade gracefully; it diverges.

**Momentum on a noisy or non-quadratic landscape.** Stochastic gradients, the
subject of the optimisers used from lesson 17 onward, mean the heavy-ball and
Nesterov update rules above no longer see a clean gradient at each step, and
momentum can amplify noise along directions where the curvature changes
quickly, which is why practical momentum coefficients are usually far from
the quadratic-optimal $\beta^\star$ derived here.

**Restarting.** Both momentum methods can build up velocity in a direction
that is no longer descending once the local quadratic approximation shifts;
restarting the momentum term to zero when the objective stops decreasing is a
common, simple fix with no clean theoretical rate of its own.

**Ill-conditioning that momentum cannot see.** Heavy-ball's optimal
parameters require knowing $\mu$ and $L$ exactly; get them wrong and the
method can converge slower than plain gradient descent, or fail to converge
at all, because the characteristic-equation roots that gave the clean
$\sqrt{\kappa}$ rate assumed the exact extremal eigenvalues.

## The exercise

You build gradient descent and its two accelerated variants on the one
problem where every rate above is exact rather than a bound: a quadratic with
a prescribed condition number.

Write `quadratic_loss_and_grad(A, b, w)`, returning the value and gradient of
$f(w) = \tfrac{1}{2}w^\top A w - b^\top w$; `gradient_descent(grad, w0, step,
iters)`, returning the full trajectory of iterates; `exact_line_search_step(A,
grad_vec)`, the closed-form optimal step derived above; and `heavy_ball` and
`nesterov`, each taking a fixed step and momentum coefficient and returning
their trajectories.

The checks verify the gradient against `mlcheck.finite_difference`, confirm
that a step above $2/L$ makes `gradient_descent` diverge, and — by starting
the iterates displaced only along the slowest eigen-direction of a diagonal
$A$, where the contraction is exact rather than approximate — confirm the
observed per-step ratio matches $1 - \mu/L$ to near machine precision. A
further pair of checks replays the heavy-ball and Nesterov recursions by hand
and compares them to your trajectories exactly, since both are deterministic
given their inputs. The last check is the point of the lesson: it measures
how many iterations `gradient_descent` and `heavy_ball` each need to reach a
fixed target accuracy on the same ill-conditioned quadratic, and checks that
the ratio is close to $\sqrt{\kappa}$, not $\kappa$.
