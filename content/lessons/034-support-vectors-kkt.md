---
title: "Which points matter: support vectors and the KKT checklist"
slug: 034-support-vectors-kkt
part: "Part 2 — Classical models"
summary: Every house is far and ignored, resting on an edge, or pushing at the cap, and one number says how badly a set of pushes breaks that picture.
minutes: 40
gpu: false
objectives:
  - Sort every house into one of three groups by the size of its push, and say why each group sits where it does.
  - Define a support vector, and explain why deleting a house with no push leaves the road unchanged.
  - Compute the slack of each house from the pushes alone.
  - Compute the violation of the KKT checklist, and say what a violation of zero does and does not tell you.
prereqs:
  - 032-widest-road-hinge-loss
  - 033-dual-multipliers-kernels
exercise: 034-support-vectors-kkt
---

# Which points matter: support vectors and the KKT checklist

A colleague's solver hands you a list of pushes, one number $\alpha_i$ per house, and a bias $b$. Is this the best road? Training again and comparing would answer the question, but it costs as much as the work you are trying to check. You want a test that looks only at the pushes you were handed. You also want to know which houses decide where the road goes.

Without such a test you cannot tell a finished solver from an unfinished one. The next chapter builds a solver that improves the pushes step by step, and it needs a rule for when to stop. The same test gives that rule.

You use the nine houses of Figure 3 in "The widest road and the hinge loss", with $C = 0.3$, and a six-house example on a line that you can check by hand.

## What you already know

These five ideas carry the chapter:

- A push $\alpha_i$ obeys $0 \le \alpha_i \le C$, and all the pushes balance, $\sum_i \alpha_i y_i = 0$.
- The score of house $i$ is $f_i = \sum_k \alpha_k y_k K_{ik} + b$, and its margin is $z_i = y_i f_i$.
- The road's edges are at $z = 1$. A house with $z < 1$ falls short and pays slack.
- The slack of house $i$ is $\max(0, 1 - z_i)$, and $C$ is the price of one unit of it.
- The two-house example from the last chapter: pushes of $0.5$ at $x = \pm 1$ rebuild the road with $w = 1$.

## Three kinds of houses

Look at Figure 1. Only four of the nine houses push at all. Five sit far from the road, faint, with a push of exactly zero. Two sit on the dashed edges and push gently. Two are capped at the largest push the price $C$ allows, and one of those is across the centre line.

```figure
<svg viewBox="0 0 600 346" role="img" aria-label="Nine houses with a road. Five faint houses far from the road push with zero force, two houses on the edges push a little, and two capped houses inside the road or across it push as hard as allowed">
<defs><marker id="c034-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-d"/></marker></defs>
<rect x="20" y="12" width="560" height="300" class="fill-paper stroke-grid"/>
<polygon points="198.4,312 471.9,312 392,12 118.5,12" class="fill-soft-b"/>
<line x1="335.2" y1="312" x2="255.2" y2="12" class="stroke-b"/>
<line x1="471.9" y1="312" x2="392" y2="12" class="stroke-b dashed"/>
<line x1="198.4" y1="312" x2="118.5" y2="12" class="stroke-b dashed"/>
<circle cx="85.9" cy="80.8" r="5" class="fill-muted"/>
<circle cx="118.8" cy="205.8" r="5" class="fill-muted"/>
<circle cx="151.8" cy="137" r="5" class="fill-a"/>
<circle cx="151.8" cy="137" r="9.5" class="stroke-ink"/>
<line x1="164.4" y1="133.8" x2="176.6" y2="130.7" class="stroke-d" marker-end="url(#c034-arrow)"/>
<circle cx="178.1" cy="249.5" r="5" class="fill-muted"/>
<rect x="410.3" y="94.5" width="10" height="10" class="fill-c"/>
<circle cx="415.3" cy="99.5" r="9.5" class="stroke-ink"/>
<line x1="402.7" y1="102.7" x2="390.5" y2="105.8" class="stroke-d" marker-end="url(#c034-arrow)"/>
<rect x="443.2" y="207" width="10" height="10" class="fill-muted"/>
<rect x="397.1" y="150.8" width="10" height="10" class="fill-c"/>
<circle cx="402.1" cy="155.8" r="9.5" class="stroke-bad"/>
<line x1="389.5" y1="158.9" x2="356.6" y2="167.3" class="stroke-d" marker-end="url(#c034-arrow)"/>
<rect x="476.2" y="257" width="10" height="10" class="fill-muted"/>
<circle cx="342.8" cy="174.5" r="5" class="fill-a"/>
<circle cx="342.8" cy="174.5" r="9.5" class="stroke-bad"/>
<line x1="355.4" y1="171.3" x2="388.4" y2="163" class="stroke-d" marker-end="url(#c034-arrow)"/>
<text x="30" y="336" class="fig-small">faint: α = 0, no push</text>
<text x="190" y="336" class="fig-small">plain ring: 0 &lt; α &lt; C, on an edge</text>
<text x="410" y="336" class="fig-small">red ring: α = C, capped</text>
</svg>
Figure 1. The same nine houses as before, now fitted with C = 0.3. Purple arrows show the pushes, and a longer arrow means a bigger α. Only four houses push at all. The two houses on the dashed edges push gently. The two capped houses push at the maximum, and one of them is across the centre line.
```

Why does each house land in its group? Take the plank picture, where each house is a peg.

- A house strictly outside the road, with $z > 1$, has room to spare. Its demand "stay beyond the edge" is not binding, so there is nothing to push against and its push is 0.
- A house that pushes by some amount below the cap must be touching the plank, which means it rests exactly on its edge, $z = 1$. If it had room, its push would drop to zero. If it were short of the edge, it would pay slack.
- A house short of the edge, with $z < 1$, pays $C$ for each unit of shortfall. That is the most any shortfall can cost, and a push measures how much the house's demand is worth, so its push is the largest allowed, $\alpha_i = C$. A house across the centre line is in this group too.

So three facts hold together: $\alpha = 0$ means $z \ge 1$, $0 < \alpha < C$ means $z = 1$, and $\alpha = C$ means $z \le 1$. A *support vector* is a house with $\alpha_i > 0$, so it is an edge house or a capped house. The road is built only from $w = \sum_i \alpha_i y_i x_i$, and a house with $\alpha_i = 0$ adds nothing to that sum. Delete the five zero-push houses and refit, as in Figure 3, and you get the same $w$ to three decimals. The support vectors are the houses that decide the road, and the rest could be erased.

## One number per house: g

To test a set of pushes, measure how far each house is past its edge:

$$
g_i = y_i f_i - 1 = z_i - 1 .
$$

Read it aloud: $g_i$ is zero on the edge, positive beyond it, and negative inside the road. Each group has a rule on $g$, and a house breaks its rule by the distance to its allowed band, as in Figure 2:

| Group | Meaning | Rule | Violation |
|---|---|---|---|
| $\alpha_i = 0$ | far, no push | $g_i \ge 0$ | $\max(0, -g_i)$ |
| $0 < \alpha_i < C$ | on an edge, free | $g_i = 0$ | $\lvert g_i \rvert$ |
| $\alpha_i = C$ | capped, at or short of the edge | $g_i \le 0$ | $\max(0, g_i)$ |

```figure
<svg viewBox="0 0 600 314" role="img" aria-label="Number lines for the value g. Each group of houses has an allowed band: g at least zero for no push, g exactly zero for a free push, g at most zero for a capped push. At the best pushes every house sits inside its band. With all pushes zero, every house sits at g equal to minus one, one unit outside its band">
<text x="12" y="20" class="fig-bold">The best pushes</text>
<text x="12" y="56" class="fig-small">α = 0</text>
<text x="12" y="70" class="fig-small">needs g ≥ 0</text>
<line x1="81.50000000000003" y1="56" x2="518.5" y2="56" class="stroke-grid"/>
<rect x="300" y="42" width="209" height="28" class="fill-soft-c"/>
<circle cx="426.7" cy="56" r="5" class="fill-ink"/>
<text x="426.7" y="38" text-anchor="middle" class="fig-small">x = −3</text>
<circle cx="363.3" cy="56" r="5" class="fill-ink"/>
<text x="363.3" y="38" text-anchor="middle" class="fig-small">x = 3</text>
<text x="12" y="102" class="fig-small">0 &lt; α &lt; C</text>
<text x="12" y="116" class="fig-small">needs g = 0</text>
<line x1="81.50000000000003" y1="102" x2="518.5" y2="102" class="stroke-grid"/>
<rect x="297" y="88" width="6" height="28" class="fill-b"/>
<circle cx="300" cy="102" r="5" class="fill-ink"/>
<text x="300" y="84" text-anchor="middle" class="fig-small">x = −1 and x = 2</text>
<circle cx="300" cy="102" r="5" class="fill-ink"/>
<text x="12" y="148" class="fig-small">α = C</text>
<text x="12" y="162" class="fig-small">needs g ≤ 0</text>
<line x1="81.50000000000003" y1="148" x2="518.5" y2="148" class="stroke-grid"/>
<rect x="91" y="134" width="209" height="28" class="fill-soft-d"/>
<circle cx="205" cy="148" r="5" class="fill-ink"/>
<text x="213" y="130" text-anchor="end" class="fig-small">x = 0.5</text>
<circle cx="236.7" cy="148" r="5" class="fill-ink"/>
<text x="228.7" y="130" class="fig-small">x = 1</text>
<line x1="81.50000000000003" y1="174" x2="518.5" y2="174" class="stroke-muted"/>
<line x1="110" y1="170" x2="110" y2="178" class="stroke-muted"/>
<text x="110" y="192" text-anchor="middle" class="fig-small">−2</text>
<line x1="205" y1="170" x2="205" y2="178" class="stroke-muted"/>
<text x="205" y="192" text-anchor="middle" class="fig-small">−1</text>
<line x1="300" y1="170" x2="300" y2="178" class="stroke-muted"/>
<text x="300" y="192" text-anchor="middle" class="fig-small">0</text>
<line x1="395" y1="170" x2="395" y2="178" class="stroke-muted"/>
<text x="395" y="192" text-anchor="middle" class="fig-small">1</text>
<line x1="490" y1="170" x2="490" y2="178" class="stroke-muted"/>
<text x="490" y="192" text-anchor="middle" class="fig-small">2</text>
<text x="300" y="208" text-anchor="middle" class="fig-small">g = y f − 1</text>
<text x="12" y="234" class="fig-bold">Someone hands you α = 0 and b = 0</text>
<text x="12" y="270" class="fig-small">α = 0</text>
<text x="12" y="284" class="fig-small">needs g ≥ 0</text>
<line x1="81.50000000000003" y1="270" x2="518.5" y2="270" class="stroke-grid"/>
<rect x="300" y="256" width="209" height="28" class="fill-soft-c"/>
<rect x="205" y="267" width="95" height="6" class="fill-bad"/>
<circle cx="205" cy="270" r="5" class="fill-ink"/>
<text x="205" y="256" text-anchor="middle" class="fig-small">all six houses, g = −1</text>
<text x="252.5" y="300" text-anchor="middle" class="fig-small">violation = 1</text>
</svg>
Figure 2. Each row is one group, with its allowed band shaded. For the best pushes of a six-house example on a line, every dot lands inside its row's band. For α = 0 and b = 0, every house has g = −1, one unit left of the band it needs, and the red bar is the violation. A violation is the distance a house lies outside its band.
```

Take six houses on a line at $x = -3, -1, 0.5, 1, 3, 2$ with labels $-1, -1, -1, +1, +1, +1$, and $C = 1$. Someone hands you $\alpha = [0, \tfrac1{18}, 1, 1, 0, \tfrac1{18}]$ and $b = -\tfrac13$. The balance rule holds, because $-\tfrac1{18} - 1 + 1 + \tfrac1{18} = 0$. The weight is $w = \tfrac1{18} - \tfrac12 + 1 + \tfrac19 = \tfrac23$. So $f = \tfrac23 x - \tfrac13$, which gives $f = [-\tfrac73, -1, 0, \tfrac13, \tfrac53, 1]$ and $g = [\tfrac43, 0, -1, -\tfrac23, \tfrac23, 0]$. Check the groups in order. The two zero-push houses have $g \ge 0$. The two free houses have $g = 0$. The two capped houses have $g \le 0$. No house breaks its rule, so the largest violation is 0.

Next, someone hands you $\alpha = 0$ and $b = 0$. Every score is 0, so every $g_i = -1$. Every house is in the zero group and needs $g \ge 0$, so each violates by 1. The largest violation is 1.

In numpy, with `f = (alpha * y) @ K + b`:

```python
g = y * f - 1.0
groups = margin_groups(alpha, C, tol)   # 0, 1, or 2 for each house
```

Slack comes from the same $g$. It is $\max(0, 1 - z_i) = \max(0, -g_i)$. At the best pushes of the six-house example it is $[0, 0, 1, \tfrac23, 0, 0]$. Only the two capped houses pay slack, which matches the groups: a house with push below $C$ has $g \ge 0$, so it falls short by nothing.

A common mistake is to sort the groups with `alpha == 0`. A solver can store $6\times 10^{-16}$ for a house whose push should be zero, and that house then lands in the free group and gets a large false violation. Compare with a tolerance, `alpha <= tol` and `alpha >= C - tol`. Two different tolerances are in play here. `tol` decides which end of the box a push sits at, and the next chapter's solver has its own, larger tolerance for when to stop.

> [!KEY]
> At the best pushes, every house is in exactly one of three groups: far away and ignored, resting on an edge and holding the road up, or short of the edge and pushing as hard as allowed. The largest violation of those three rules is the test you wanted.

## What the number does not say

The largest violation is a stopping measure, not a distance from the best answer. It tells you that every house is within that amount of its allowed band, and it is zero only when all three rules hold. The six-house check prints a value near $10^{-16}$ rather than a clean 0, which is zero up to rounding.

The checklist has two more rules: the box rule $0 \le \alpha_i \le C$ and the balance rule $\sum_i \alpha_i y_i = 0$. The function in this chapter does not test them. The next chapter's solver keeps both true at every step by construction. All the rules together are the *KKT conditions*, the checklist that the best pushes obey. For this problem, a set of pushes that obeys all of them is the best set, so a violation of zero is a certificate and not only a hint.

The violation also depends on $b$. A wrong $b$ shifts every $g$ and creates a violation even when $\alpha$ is right. The next chapter's solver reads the same test in a per-house form. With $E_i = f_i - y_i$, the error of the score, $y_i E_i = y_i f_i - 1 = g_i$, because $y_i^2 = 1$.

## Where this breaks

**A house at the cap and on the edge at once.** A house can have $\alpha = C$ and $z = 1$ exactly. Its $g$ is 0, which satisfies the capped rule, so it passes the test, yet it sits on an edge as a free house would. The groups overlap there, so treat the boundary between them as soft.

**No free house.** When every push is 0 or $C$, no house sits strictly on an edge, so the edges do not pin $b$. Any $b$ in a range gives the same pushes, and the test passes for each of them.

**A loose tolerance.** If `tol` is too large, a small genuine push counts as zero and the check calls a wrong set of pushes right. Keep it small, near $10^{-8}$ for numbers that come from a careful solver.

> [!RECAP]
> - $\alpha = 0$ means $z \ge 1$, $0 < \alpha < C$ means $z = 1$, and $\alpha = C$ means $z \le 1$.
> - A support vector is a house with $\alpha > 0$, and deleting the others leaves the road unchanged.
> - $g_i = y_i f_i - 1$, and a house's violation is its distance outside its group's band.
> - The largest violation is a stopping measure, not a distance, and the box and balance rules are checked separately.

## Build it step by step

You write three functions, in the order the checks test them. The score function `dual_scores` is already written, from the chapter "Pushes from each point: the dual and kernels".

1. **`margin_groups(alpha, C, tol)`**. `alpha` is `(n,)`. It returns an integer array of shape `(n,)` with 0, 1, or 2. Start from ones and overwrite the two ends with `alpha <= tol` and `alpha >= C - tol`. The common mistake is `alpha == 0`, or using 1 in place of `C`.

2. **`slack_amounts(alpha, y, K, b)`**. It returns the `(n,)` array `max(0, 1 - y * f)`. Use `np.maximum`, not `max`.

3. **`kkt_violation(alpha, y, K, b, C, tol)`**. It returns one float. The shape of the code is:

```python
g = y * dual_scores(alpha, y, K, b) - 1.0     # shape (n,)
groups = margin_groups(alpha, C, tol)         # shape (n,)
```

   Build the three violation arrays, pick one per house with `np.where`, and return the maximum. The common mistake is the wrong sign for the capped group, which needs `max(0, g)`, not `max(0, -g)`.

## The exercise

You build `margin_groups`, `slack_amounts`, and `kkt_violation`. The checks confirm the groups, including a push of $10^{-16}$ and a cap other than 1. They confirm the slack and the violations on the six-house example, 0 at the best pushes and 1 at $\alpha = 0$. They compare `kkt_violation` with a house-by-house loop on random pushes. Finally an independent solver finds the best pushes for the nine houses of Figure 1, and your functions must show a tiny violation there, a visible one when an edge house loses its push, and an unchanged road when the zero-push houses are deleted.
