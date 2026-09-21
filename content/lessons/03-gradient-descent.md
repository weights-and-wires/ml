---
title: Gradient descent
slug: 03-gradient-descent
part: "Part 1 — Foundations"
summary: The fixed-step rate for gradient descent in the condition number, and how momentum improves it to a square root.
minutes: 55
gpu: false
objectives:
  - State smoothness and strong convexity, and explain what the descent lemma guarantees from smoothness alone.
  - State the fixed-step contraction rate for gradient descent on a quadratic, and identify the condition number as the iteration count.
  - State the heavy-ball and Nesterov updates, and explain why their rate improves from a kappa dependence to a sqrt(kappa) dependence.
  - Say which rate holds for which method, under which assumption, and what changes once the objective is not a quadratic.
exercise: 03-gradient-descent
prereqs:
  - 02-linear-regression
---

# Gradient descent

Lesson 2 solved least squares in closed form, and that closed form does not generalise. Once the objective stops being a quadratic you can factor directly — a logistic loss, a deep network, an $\ell_1$ penalty — there is no normal equation to write down. What remains is the gradient. This lesson answers two questions about following it: how fast does it reach the answer, and how does that depend on the objective's shape near its minimum?

The headline results are two rates. Fixed-step gradient descent needs a number of iterations proportional to the condition number, $\kappa$, to reach a target accuracy. Two variants, heavy-ball momentum and Nesterov acceleration, cut that dependence to $\sqrt{\kappa}$. That difference is not academic: for a condition number of $10{,}000$, it is the gap between $10{,}000$ steps and $100$. These rates are exactly what you tune against every time you set a learning rate later in this course.

## What you already know

- A twice-differentiable $f$ is convex if and only if $\nabla^2 f(w) \succeq 0$ everywhere.
- The eigenvalues of a symmetric matrix bound its quadratic form: $\lambda_{\min}(M) \lVert v \rVert^2 \leq v^\top M v \leq \lambda_{\max}(M) \lVert v \rVert^2$.
- Least squares is the quadratic $L(w) = \tfrac{1}{2}\lVert Xw - y \rVert^2$, with gradient $X^\top(Xw - y)$ and Hessian $X^\top X$.
- The Cauchy-Schwarz inequality: $\lvert u^\top v \rvert \leq \lVert u \rVert \lVert v \rVert$.
- A first-order Taylor expansion approximates $f$ near a point with a linear term plus a remainder that the descent lemma below turns into a usable bound.

## Smoothness and strong convexity

Two constants govern every rate in this lesson. $f$ is **$L$-smooth** when its gradient is $L$-Lipschitz, meaning nearby points have gradients that are close in proportion to $L$:

$$
\lVert \nabla f(u) - \nabla f(v) \rVert \leq L \lVert u - v \rVert \quad \text{for all } u, v .
$$

$f$ is **$\mu$-strongly convex** when it lies above not just its tangent plane, the ordinary convexity condition, but a tangent paraboloid with curvature at least $\mu$ in every direction:

$$
f(v) \geq f(u) + \nabla f(u)^\top (v - u) + \frac{\mu}{2}\lVert v - u \rVert^2 \quad \text{for all } u, v .
$$

For twice-differentiable $f$, the two conditions are equivalent to $\nabla^2 f(w) \preceq LI$ and $\nabla^2 f(w) \succeq \mu I$. For the quadratic $f(w) = \tfrac{1}{2}w^\top A w - b^\top w$ with $A$ symmetric, $\nabla^2 f = A$ everywhere, so $L = \lambda_{\max}(A)$ and $\mu = \lambda_{\min}(A)$ exactly — a quadratic is the case where these constants stop being bounds and become the truth. The ratio $\kappa = L / \mu \geq 1$ is the **condition number** of the problem, the same quantity as $\kappa_2(X)^2$ from lesson 2 when $A = X^\top X$.

## The descent lemma

$L$-smoothness alone, with no convexity required, bounds how far $f$ can rise above its linear approximation:

$$
f(v) \leq f(u) + \nabla f(u)^\top (v - u) + \frac{L}{2} \lVert v - u \rVert^2 . \tag{descent lemma}
$$

It holds because a Lipschitz gradient cannot let $f$ curve away from its tangent line faster than the quadratic $\tfrac{L}{2}\lVert v-u\rVert^2$ allows. Integrating the Lipschitz bound along the segment from $u$ to $v$ gives exactly this term. Set $v = u - \tfrac{1}{L}\nabla f(u)$, the gradient step with step size $1/L$, and the right side becomes $f(u) - \tfrac{1}{2L}\lVert \nabla f(u) \rVert^2$, so

$$
f\Big(u - \tfrac{1}{L}\nabla f(u)\Big) \leq f(u) - \frac{1}{2L}\lVert \nabla f(u) \rVert^2 .
$$

Every step of size $1/L$ decreases $f$ by an amount proportional to the squared gradient norm. That single fact, needing nothing but smoothness, is why gradient descent works at all.

## The fixed-step rate on a quadratic

Strong convexity turns the descent guarantee into a rate. On the quadratic $f(w) = \tfrac{1}{2}w^\top A w - b^\top w$ with minimiser $w^\star = A^{-1}b$, a fixed-step update $w_{k+1} = w_k - \alpha \nabla f(w_k)$ satisfies $w_{k+1} - w^\star = (I - \alpha A)(w_k - w^\star)$. Diagonalise $A = Q \Lambda Q^\top$ and rotate coordinates by $Q$: each coordinate then shrinks independently by a factor $1 - \alpha \lambda_i$, one per eigenvalue $\mu \leq \lambda_i \leq L$, and because $Q$ preserves norms,

$$
\lVert w_k - w^\star \rVert \leq \Big(\max_i \lvert 1 - \alpha \lambda_i \rvert\Big)^k \lVert w_0 - w^\star \rVert .
$$

The step $\alpha = 1/L$ makes the top eigenvalue's factor exactly $0$, leaving $1 - \mu/L$ as the largest remaining term, so

$$
\lVert w_k - w^\star \rVert \leq \left(1 - \frac{\mu}{L}\right)^{k} \lVert w_0 - w^\star \rVert .
$$

This is linear convergence: the error shrinks by a constant factor every step. Reaching accuracy $\varepsilon$ needs

$$
k \geq \frac{\log(1/\varepsilon)}{-\log(1 - \mu/L)} \approx \kappa \log(1/\varepsilon) ,
$$

using $-\log(1-x) \geq x$. Up to that logarithmic factor, the condition number is the iteration count. A well-conditioned quadratic converges in a handful of steps, and an ill-conditioned one needs on the order of $\kappa$ of them. Push $\alpha$ past $2/L$ and the top-eigenvalue factor exceeds $1$ in magnitude, and the iterates diverge regardless of how small $\mu$ is — smoothness alone sets that stability boundary.

## Exact line search

Instead of fixing $\alpha$, choose the step that minimises $f$ exactly along the current gradient direction. For the quadratic, writing $g = \nabla f(w_k)$, that minimisation has a closed form:

$$
\alpha_k = \frac{g^\top g}{g^\top A g} .
$$

It follows from setting the derivative of $f(w_k - \alpha g)$, a one-dimensional quadratic in $\alpha$, to zero. Exact line search removes the need to know $L$ and $\mu$ in advance, but it does not remove the $\kappa$ dependence. On an ill-conditioned quadratic it zig-zags across the narrow valley one eigen-direction at a time. Its worst-case rate is the same $(\kappa-1)/(\kappa+1)$ as the best fixed step. A better step each iteration is not the same as a better rate.

## Heavy-ball momentum

The fixed-step rate is stuck at $\kappa$ because the update only looks at the current gradient. Polyak's heavy-ball method adds inertia from the previous step:

$$
w_{k+1} = w_k - \alpha \nabla f(w_k) + \beta (w_k - w_{k-1}) .
$$

Choosing $\alpha$ and $\beta$ to make this recursion contract as fast as possible at both extreme eigenvalues gives the optimal parameters

$$
\alpha^\star = \left(\frac{2}{\sqrt{L} + \sqrt{\mu}}\right)^2, \qquad
\beta^\star = \left(\frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}}\right)^2 ,
$$

for which the per-step contraction factor is $\rho = (\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$ instead of $1 - 1/\kappa$. By the same logarithm argument as before, reaching accuracy $\varepsilon$ now takes $O(\sqrt{\kappa}\log(1/\varepsilon))$ iterations rather than $O(\kappa \log(1/\varepsilon))$. Momentum buys a square root in the condition number, for the price of remembering one extra vector.

## Nesterov acceleration

Heavy-ball's guarantee is specific to quadratics; for general smooth, strongly convex functions it can cycle instead of converging at the accelerated rate. Nesterov's method reaches the same $\sqrt{\kappa}$ rate with a guarantee that holds for every $L$-smooth, $\mu$-strongly convex function. It does so by evaluating the gradient at an extrapolated point rather than at the current iterate:

$$
y_k = w_k + \beta (w_k - w_{k-1}), \qquad w_{k+1} = y_k - \alpha \nabla f(y_k) ,
$$

with $\alpha = 1/L$ and $\beta = (\sqrt{L}-\sqrt{\mu})/(\sqrt{L}+\sqrt{\mu})$ in the strongly convex case. The one-line intuition is that $y_k$ looks ahead using the momentum term before the gradient is taken, so the step corrects for where the trajectory is heading rather than only where it stands. Gradient descent reacts, and Nesterov's method anticipates. The rigorous argument builds a sequence of quadratic lower bounds on $f$ that shrink geometrically at rate $\sqrt{\kappa}$. It is not repeated here, because nothing in the exercise depends on reconstructing it.

## What goes wrong away from quadratics

Every rate above that mentions one number $\kappa$ relies on a quadratic's constant Hessian, where $L$ and $\mu$ are the same everywhere. Gradient descent and Nesterov's method keep the same rates for a general $L$-smooth, $\mu$-strongly convex function, since the descent lemma and the strong-convexity bound were stated for general $f$. Heavy-ball's analysis used exact diagonalisation of one fixed matrix $A$ and does not extend. Its global rate away from quadratics is an open problem. Naive momentum can overshoot and oscillate once curvature changes quickly between iterations. Away from convexity entirely, none of these rates say anything about reaching a global minimum. Gradient descent converges only to a stationary point, and the loss landscapes later in this course are not convex.

## Cost and scaling

| Method | Per-step cost | Iterations for accuracy $\varepsilon$ | Extra state |
|---|---|---|---|
| Fixed-step gradient descent | One gradient | $O(\kappa \log(1/\varepsilon))$ | None |
| Exact line search (quadratic) | One gradient, one matrix-vector product | $O(\kappa \log(1/\varepsilon))$ | None |
| Heavy ball | One gradient | $O(\sqrt{\kappa} \log(1/\varepsilon))$ | One previous iterate |
| Nesterov | One gradient, at an extrapolated point | $O(\sqrt{\kappa} \log(1/\varepsilon))$ | One previous iterate |

Every method here spends most of its per-step cost on one gradient evaluation. Momentum's entire overhead is one extra vector of memory and two scalar multiplies. That is why it is the default rather than an exotic option. For the price of storing $w_{k-1}$, the iteration count's dependence on $\kappa$ drops from linear to square root, and $\kappa$ is frequently in the thousands for the problems this course builds toward.

## Where this breaks

**Overshooting the smoothness constant.** Every rate above assumes $\alpha \leq 2/L$ for gradient descent, or the matching bound for momentum. $L$ is rarely known exactly, so practitioners estimate it or use a line search; too large a step diverges rather than degrading gracefully.

**Momentum on a noisy or non-quadratic landscape.** Stochastic gradients are the subject of the optimisers used from lesson 17 onward. They mean the heavy-ball and Nesterov updates no longer see a clean gradient at each step. Momentum can amplify noise where curvature changes quickly, which is why practical momentum coefficients usually sit far from the quadratic-optimal $\beta^\star$ derived here.

**Restarting.** Both momentum methods can build up velocity in a direction that stops descending once the local quadratic approximation shifts. Restarting the momentum term to zero when the objective stops decreasing is a common fix, with no clean theoretical rate of its own.

**Ill-conditioning that momentum cannot see.** Heavy-ball's optimal parameters require knowing $\mu$ and $L$ exactly. Get them wrong and the method can converge slower than plain gradient descent, or not at all, because the clean $\sqrt{\kappa}$ rate assumed the exact extremal eigenvalues.

## The exercise

You build gradient descent and its two accelerated variants on the one problem where every rate above is exact rather than a bound: a quadratic with a prescribed condition number.

Write `quadratic_loss_and_grad(A, b, w)`, returning the value and gradient of $f(w) = \tfrac{1}{2}w^\top A w - b^\top w$. Write `gradient_descent(grad, w0, step, iters)`, returning the full trajectory of iterates. Write `exact_line_search_step(A, grad_vec)`, the closed-form optimal step derived above. Write `heavy_ball` and `nesterov`, each taking a fixed step and momentum coefficient and returning their trajectories.

The checks verify the gradient against `mlcheck.finite_difference`. They confirm that a step above $2/L$ makes `gradient_descent` diverge. By starting the iterates displaced only along the slowest eigen-direction of a diagonal $A$, where the contraction is exact rather than approximate, they confirm the observed per-step ratio matches $1 - \mu/L$ to near machine precision. A further pair of checks replays the heavy-ball and Nesterov recursions by hand and compares them to your trajectories exactly, since both are deterministic given their inputs. The last check is the point of the lesson. It measures how many iterations `gradient_descent` and `heavy_ball` each need to reach a fixed target accuracy on the same ill-conditioned quadratic. It also checks that the ratio between them is close to $\sqrt{\kappa}$, not $\kappa$.
