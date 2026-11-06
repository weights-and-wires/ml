---
title: The representer theorem and random Fourier features
slug: 036-representer-and-random-features
part: "Part 2 — Classical models"
summary: Bumps at the training points are enough, which makes kernel methods cost n squared; random features cut that to linear in n.
minutes: 45
gpu: false
objectives:
  - Explain why a kernel model needs bumps only at the training points, using the split into a part inside and a part outside their span.
  - Compute the objective gap lambda t^2 d^T K d for an extra component, and check it numerically.
  - Build random Fourier features and explain why a cosine of a random projection averages to an RBF bump.
  - Fit ridge regression on random features and say what it costs compared with exact kernel ridge regression.
prereqs:
  - 033-kernels-as-similarity
  - 035-kernel-ridge-regression
exercise: 036-representer-and-random-features
---

# The representer theorem and random Fourier features

Kernel ridge regression puts one bump on each training point. A model could in principle put bumps anywhere: on a grid, on random locations, between the points. Why do the training points suffice? If they do not, kernel ridge regression solved the wrong problem.

There is a second worry, which has nothing to do with correctness. The system you solved has one row per training point, and the kernel matrix has $n^2$ entries. At $n = 100{,}000$ points that matrix needs about 80 GB of memory before you solve anything.

This chapter answers both. First you see why bumps at the training points are enough, which is the *representer theorem*. Then you see how to keep the kernel idea without ever forming the $n \times n$ matrix, by replacing the RBF kernel with a modest number of random features.

## What you already know

- A kernel model predicts $\sum_i \alpha_i k(x, x_i)$, a sum of scaled bumps, and its weights solve $(K + \lambda I)\alpha = y$ ("Kernel ridge regression: a weighted sum of similarities").
- The size fee for such a function is $\lambda\,\alpha^{\top}K\alpha$, the ridge fee $\lambda\lVert w\rVert^2$ written in terms of $\alpha$.
- An RBF bump with width set by $\gamma$ scores 1 at its centre and falls toward 0 with distance ("Kernels: a similarity score that hides a bigger feature space").
- A product of two cosines turns into a sum: $2\cos a\cos b = \cos(a - b) + \cos(a + b)$.
- Ordinary ridge regression on $d$ features solves a $d \times d$ system ("Ridge: a fee on large weights").

## The picture: a wiggle that costs and buys nothing

Take three training points at $x = -1, 0, 1$ and fit them with kernel ridge regression. Call the result the best fit. Build a second function by adding bumps at four *other* centres, $-2, -0.5, 0.5, 2$, with weights chosen so that the extra part is exactly zero at the three training points. Figure 1 shows both.

```figure
<svg viewBox="0 0 600 254" role="img" aria-label="Two curves that agree at three training points and differ between them: the best fit and the best fit plus a wiggle that is zero at the training points.">
<line x1="50.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-muted"/><line x1="50.0" y1="20.0" x2="50.0" y2="210.0" class="stroke-muted"/><line x1="50.0" y1="136.9" x2="570.0" y2="136.9" class="stroke-grid"/><text x="50.0" y="226.0" text-anchor="middle" class="fig-small">-3</text><text x="136.7" y="226.0" text-anchor="middle" class="fig-small">-2</text><text x="223.3" y="226.0" text-anchor="middle" class="fig-small">-1</text><text x="310.0" y="226.0" text-anchor="middle" class="fig-small">0</text><text x="396.7" y="226.0" text-anchor="middle" class="fig-small">1</text><text x="483.3" y="226.0" text-anchor="middle" class="fig-small">2</text><text x="570.0" y="226.0" text-anchor="middle" class="fig-small">3</text><text x="44.0" y="214.0" text-anchor="end" class="fig-small">-1</text><text x="44.0" y="140.9" text-anchor="end" class="fig-small">0</text><text x="44.0" y="67.8" text-anchor="end" class="fig-small">1</text><path d="M50.0,137.1 L51.7,137.2 L53.5,137.2 L55.2,137.2 L57.0,137.2 L58.7,137.2 L60.4,137.3 L62.2,137.3 L63.9,137.3 L65.7,137.3 L67.4,137.4 L69.1,137.4 L70.9,137.4 L72.6,137.5 L74.3,137.5 L76.1,137.5 L77.8,137.6 L79.6,137.6 L81.3,137.7 L83.0,137.7 L84.8,137.8 L86.5,137.8 L88.3,137.9 L90.0,137.9 L91.7,138.0 L93.5,138.1 L95.2,138.1 L97.0,138.2 L98.7,138.2 L100.4,138.3 L102.2,138.4 L103.9,138.5 L105.7,138.5 L107.4,138.6 L109.1,138.7 L110.9,138.8 L112.6,138.9 L114.3,138.9 L116.1,139.0 L117.8,139.1 L119.6,139.2 L121.3,139.3 L123.0,139.4 L124.8,139.4 L126.5,139.5 L128.3,139.6 L130.0,139.7 L131.7,139.8 L133.5,139.8 L135.2,139.9 L137.0,140.0 L138.7,140.0 L140.4,140.1 L142.2,140.1 L143.9,140.2 L145.7,140.2 L147.4,140.3 L149.1,140.3 L150.9,140.3 L152.6,140.3 L154.3,140.3 L156.1,140.3 L157.8,140.2 L159.6,140.2 L161.3,140.1 L163.0,140.0 L164.8,140.0 L166.5,139.8 L168.3,139.7 L170.0,139.6 L171.7,139.4 L173.5,139.2 L175.2,139.0 L177.0,138.7 L178.7,138.5 L180.4,138.2 L182.2,137.9 L183.9,137.5 L185.7,137.2 L187.4,136.8 L189.1,136.3 L190.9,135.9 L192.6,135.4 L194.3,134.9 L196.1,134.3 L197.8,133.7 L199.6,133.1 L201.3,132.4 L203.0,131.7 L204.8,131.0 L206.5,130.3 L208.3,129.5 L210.0,128.6 L211.7,127.8 L213.5,126.9 L215.2,125.9 L217.0,124.9 L218.7,123.9 L220.4,122.9 L222.2,121.8 L223.9,120.7 L225.7,119.6 L227.4,118.4 L229.1,117.3 L230.9,116.0 L232.6,114.8 L234.3,113.5 L236.1,112.3 L237.8,110.9 L239.6,109.6 L241.3,108.3 L243.0,106.9 L244.8,105.6 L246.5,104.2 L248.3,102.8 L250.0,101.4 L251.7,100.1 L253.5,98.7 L255.2,97.3 L257.0,95.9 L258.7,94.6 L260.4,93.2 L262.2,91.9 L263.9,90.6 L265.7,89.3 L267.4,88.0 L269.1,86.8 L270.9,85.5 L272.6,84.4 L274.3,83.2 L276.1,82.1 L277.8,81.0 L279.6,80.0 L281.3,79.0 L283.0,78.1 L284.8,77.2 L286.5,76.4 L288.3,75.7 L290.0,75.0 L291.7,74.3 L293.5,73.7 L295.2,73.2 L297.0,72.8 L298.7,72.4 L300.4,72.0 L302.2,71.8 L303.9,71.6 L305.7,71.5 L307.4,71.5 L309.1,71.5 L310.9,71.6 L312.6,71.8 L314.3,72.0 L316.1,72.3 L317.8,72.7 L319.6,73.1 L321.3,73.7 L323.0,74.2 L324.8,74.9 L326.5,75.6 L328.3,76.4 L330.0,77.2 L331.7,78.1 L333.5,79.1 L335.2,80.1 L337.0,81.1 L338.7,82.2 L340.4,83.4 L342.2,84.6 L343.9,85.8 L345.7,87.1 L347.4,88.4 L349.1,89.7 L350.9,91.1 L352.6,92.5 L354.3,93.9 L356.1,95.3 L357.8,96.8 L359.6,98.3 L361.3,99.7 L363.0,101.2 L364.8,102.7 L366.5,104.2 L368.3,105.7 L370.0,107.2 L371.7,108.7 L373.5,110.2 L375.2,111.6 L377.0,113.1 L378.7,114.5 L380.4,115.9 L382.2,117.3 L383.9,118.6 L385.7,120.0 L387.4,121.3 L389.1,122.6 L390.9,123.8 L392.6,125.0 L394.3,126.2 L396.1,127.4 L397.8,128.5 L399.6,129.5 L401.3,130.6 L403.0,131.6 L404.8,132.5 L406.5,133.4 L408.3,134.3 L410.0,135.2 L411.7,135.9 L413.5,136.7 L415.2,137.4 L417.0,138.1 L418.7,138.7 L420.4,139.3 L422.2,139.9 L423.9,140.4 L425.7,140.9 L427.4,141.3 L429.1,141.7 L430.9,142.1 L432.6,142.4 L434.3,142.7 L436.1,143.0 L437.8,143.3 L439.6,143.5 L441.3,143.7 L443.0,143.8 L444.8,143.9 L446.5,144.0 L448.3,144.1 L450.0,144.2 L451.7,144.2 L453.5,144.2 L455.2,144.2 L457.0,144.2 L458.7,144.2 L460.4,144.1 L462.2,144.0 L463.9,144.0 L465.7,143.9 L467.4,143.8 L469.1,143.6 L470.9,143.5 L472.6,143.4 L474.3,143.3 L476.1,143.1 L477.8,143.0 L479.6,142.8 L481.3,142.6 L483.0,142.5 L484.8,142.3 L486.5,142.1 L488.3,142.0 L490.0,141.8 L491.7,141.6 L493.5,141.5 L495.2,141.3 L497.0,141.1 L498.7,141.0 L500.4,140.8 L502.2,140.6 L503.9,140.5 L505.7,140.3 L507.4,140.2 L509.1,140.0 L510.9,139.9 L512.6,139.7 L514.3,139.6 L516.1,139.5 L517.8,139.3 L519.6,139.2 L521.3,139.1 L523.0,139.0 L524.8,138.9 L526.5,138.8 L528.3,138.7 L530.0,138.6 L531.7,138.5 L533.5,138.4 L535.2,138.3 L537.0,138.2 L538.7,138.1 L540.4,138.1 L542.2,138.0 L543.9,137.9 L545.7,137.9 L547.4,137.8 L549.1,137.7 L550.9,137.7 L552.6,137.6 L554.3,137.6 L556.1,137.5 L557.8,137.5 L559.6,137.5 L561.3,137.4 L563.0,137.4 L564.8,137.3 L566.5,137.3 L568.3,137.3 L570.0,137.3" fill="none" class="stroke-b" stroke-width="2.2"/><path d="M50.0,154.1 L51.7,154.8 L53.5,155.5 L55.2,156.2 L57.0,157.0 L58.7,157.7 L60.4,158.5 L62.2,159.2 L63.9,160.0 L65.7,160.8 L67.4,161.6 L69.1,162.4 L70.9,163.2 L72.6,164.0 L74.3,164.9 L76.1,165.7 L77.8,166.5 L79.6,167.3 L81.3,168.1 L83.0,168.9 L84.8,169.7 L86.5,170.5 L88.3,171.3 L90.0,172.1 L91.7,172.9 L93.5,173.6 L95.2,174.4 L97.0,175.1 L98.7,175.8 L100.4,176.5 L102.2,177.1 L103.9,177.8 L105.7,178.4 L107.4,179.0 L109.1,179.5 L110.9,180.1 L112.6,180.6 L114.3,181.0 L116.1,181.5 L117.8,181.9 L119.6,182.2 L121.3,182.5 L123.0,182.8 L124.8,183.1 L126.5,183.3 L128.3,183.4 L130.0,183.5 L131.7,183.6 L133.5,183.6 L135.2,183.6 L137.0,183.5 L138.7,183.4 L140.4,183.2 L142.2,183.0 L143.9,182.7 L145.7,182.4 L147.4,182.0 L149.1,181.6 L150.9,181.1 L152.6,180.6 L154.3,180.0 L156.1,179.4 L157.8,178.7 L159.6,178.0 L161.3,177.2 L163.0,176.3 L164.8,175.4 L166.5,174.5 L168.3,173.5 L170.0,172.4 L171.7,171.3 L173.5,170.2 L175.2,169.0 L177.0,167.7 L178.7,166.4 L180.4,165.1 L182.2,163.7 L183.9,162.3 L185.7,160.8 L187.4,159.3 L189.1,157.7 L190.9,156.1 L192.6,154.5 L194.3,152.8 L196.1,151.1 L197.8,149.4 L199.6,147.6 L201.3,145.8 L203.0,144.0 L204.8,142.1 L206.5,140.2 L208.3,138.3 L210.0,136.4 L211.7,134.4 L213.5,132.5 L215.2,130.5 L217.0,128.5 L218.7,126.5 L220.4,124.5 L222.2,122.4 L223.9,120.4 L225.7,118.4 L227.4,116.4 L229.1,114.4 L230.9,112.4 L232.6,110.4 L234.3,108.4 L236.1,106.4 L237.8,104.4 L239.6,102.5 L241.3,100.6 L243.0,98.7 L244.8,96.9 L246.5,95.1 L248.3,93.3 L250.0,91.5 L251.7,89.8 L253.5,88.2 L255.2,86.6 L257.0,85.0 L258.7,83.5 L260.4,82.1 L262.2,80.7 L263.9,79.3 L265.7,78.1 L267.4,76.9 L269.1,75.7 L270.9,74.7 L272.6,73.7 L274.3,72.7 L276.1,71.9 L277.8,71.1 L279.6,70.4 L281.3,69.8 L283.0,69.3 L284.8,68.8 L286.5,68.5 L288.3,68.2 L290.0,68.0 L291.7,67.9 L293.5,67.8 L295.2,67.9 L297.0,68.0 L298.7,68.2 L300.4,68.5 L302.2,68.9 L303.9,69.3 L305.7,69.9 L307.4,70.5 L309.1,71.2 L310.9,71.9 L312.6,72.7 L314.3,73.6 L316.1,74.6 L317.8,75.6 L319.6,76.7 L321.3,77.8 L323.0,79.0 L324.8,80.2 L326.5,81.5 L328.3,82.8 L330.0,84.2 L331.7,85.6 L333.5,87.0 L335.2,88.5 L337.0,89.9 L338.7,91.4 L340.4,93.0 L342.2,94.5 L343.9,96.0 L345.7,97.5 L347.4,99.1 L349.1,100.6 L350.9,102.1 L352.6,103.6 L354.3,105.1 L356.1,106.6 L357.8,108.0 L359.6,109.4 L361.3,110.8 L363.0,112.1 L364.8,113.4 L366.5,114.7 L368.3,115.9 L370.0,117.1 L371.7,118.2 L373.5,119.3 L375.2,120.3 L377.0,121.3 L378.7,122.2 L380.4,123.0 L382.2,123.8 L383.9,124.5 L385.7,125.1 L387.4,125.7 L389.1,126.2 L390.9,126.7 L392.6,127.1 L394.3,127.4 L396.1,127.7 L397.8,127.9 L399.6,128.0 L401.3,128.0 L403.0,128.0 L404.8,128.0 L406.5,127.8 L408.3,127.6 L410.0,127.4 L411.7,127.1 L413.5,126.7 L415.2,126.3 L417.0,125.9 L418.7,125.4 L420.4,124.8 L422.2,124.2 L423.9,123.6 L425.7,122.9 L427.4,122.2 L429.1,121.5 L430.9,120.7 L432.6,119.9 L434.3,119.1 L436.1,118.3 L437.8,117.4 L439.6,116.6 L441.3,115.7 L443.0,114.8 L444.8,114.0 L446.5,113.1 L448.3,112.2 L450.0,111.3 L451.7,110.5 L453.5,109.6 L455.2,108.8 L457.0,107.9 L458.7,107.1 L460.4,106.4 L462.2,105.6 L463.9,104.9 L465.7,104.1 L467.4,103.5 L469.1,102.8 L470.9,102.2 L472.6,101.6 L474.3,101.1 L476.1,100.6 L477.8,100.1 L479.6,99.7 L481.3,99.3 L483.0,98.9 L484.8,98.6 L486.5,98.4 L488.3,98.1 L490.0,98.0 L491.7,97.8 L493.5,97.7 L495.2,97.7 L497.0,97.7 L498.7,97.7 L500.4,97.8 L502.2,97.9 L503.9,98.0 L505.7,98.2 L507.4,98.5 L509.1,98.7 L510.9,99.1 L512.6,99.4 L514.3,99.8 L516.1,100.2 L517.8,100.6 L519.6,101.1 L521.3,101.6 L523.0,102.1 L524.8,102.6 L526.5,103.2 L528.3,103.8 L530.0,104.4 L531.7,105.0 L533.5,105.7 L535.2,106.3 L537.0,107.0 L538.7,107.7 L540.4,108.4 L542.2,109.1 L543.9,109.8 L545.7,110.5 L547.4,111.2 L549.1,111.9 L550.9,112.7 L552.6,113.4 L554.3,114.1 L556.1,114.8 L557.8,115.5 L559.6,116.2 L561.3,116.9 L563.0,117.6 L564.8,118.3 L566.5,119.0 L568.3,119.7 L570.0,120.3" fill="none" class="stroke-bad dashed" stroke-width="2.2"/><circle cx="223.3" cy="122.3" r="4.5" class="fill-a"/><circle cx="310.0" cy="63.8" r="4.5" class="fill-a"/><circle cx="396.7" cy="129.6" r="4.5" class="fill-a"/><text x="336.0" y="31.0" text-anchor="start" class="fig-small">red dashed: fit + wiggle, equal at x = −1, 0, 1</text><text x="58.7" y="56.5" text-anchor="start" class="fig-small">orange: best fit, smaller penalty</text><text x="300.0" y="246.0" text-anchor="middle" class="fig-small">x</text>
</svg>
Figure 1. The best fit (orange) and the best fit plus a wiggle (red, dashed). They agree at the three training points, so their squared errors are identical. Notice that the red curve differs only between and beyond the points, where no data constrains it.
```

The two curves score the same on the training data, so the squared error cannot tell them apart. The size fee can. The wiggle is extra function that does nothing for the fit and still counts toward the size, so the penalty only goes up. An optimiser that wants a low objective sets the wiggle to zero. That is the representer theorem in one picture.

## The idea in numbers

Describe a candidate by weights $c$ over a list of centres, which may include extra points beyond the training points. The model is $\sum_j c_j k(x, z_j)$, and its objective is

$$
J(c) = \sum_{i=1}^{n}\left(y_i - (Kc)_i\right)^2 + \lambda\,c^{\top}Kc .
$$

Read it aloud: "squared error at the $n$ training points, plus lambda times the squared size of the function." The matrix $K$ here covers all the centres, and only the first $n$ entries of $Kc$ are the training predictions. The size $c^{\top}Kc$ is the squared length of the function in feature space, by the same step as $\lVert w\rVert^2 = \alpha^{\top}K\alpha$ in the previous chapter.

Work the smallest case. Take two centres with $K = I_2$, so the second centre is unrelated to the first, one training target $y = 2$, and $\lambda = 1$. The best weight on the training point is $\alpha = 1$, since $(1 + 1)\alpha = 2$. With $c = (1, 0)$ the prediction is 1, the squared error is 1, the size is 1, and $J = 2$. Add the extra centre with weight 1, so $c = (1, 1)$. The prediction is still 1, but the size is 2, and $J = 3$. Figure 2 shows these two bars on the left.

```figure
<svg viewBox="0 0 600 236" role="img" aria-label="Left: objective values 2 and 3 for the hand example. Right: the objective is a parabola in t with its lowest point at t equal to 0.">
<text x="115.0" y="22.0" text-anchor="middle" class="fig-small fig-bold">hand example</text><line x1="40.0" y1="184.0" x2="190.0" y2="184.0" class="stroke-ink"/><rect x="55.0" y="100.7" width="45.0" height="83.3" class="fill-b"/><text x="77.5" y="94.7" text-anchor="middle" class="fig-small">J = 2</text><text x="77.5" y="200.0" text-anchor="middle" class="fig-small">c = [1, 0]</text><rect x="130.0" y="59.0" width="45.0" height="125.0" class="fill-bad"/><text x="152.5" y="53.0" text-anchor="middle" class="fig-small">J = 3</text><text x="152.5" y="200.0" text-anchor="middle" class="fig-small">c = [1, 1]</text><text x="425.0" y="22.0" text-anchor="middle" class="fig-small fig-bold">6 training + 10 extra points: J against t</text><line x1="280.0" y1="184.0" x2="570.0" y2="184.0" class="stroke-muted"/><path d="M280.0,64.2 L283.6,68.6 L287.2,72.9 L290.9,77.1 L294.5,81.2 L298.1,85.2 L301.8,89.0 L305.4,92.8 L309.0,96.4 L312.6,100.0 L316.2,103.4 L319.9,106.7 L323.5,109.9 L327.1,113.0 L330.8,115.9 L334.4,118.8 L338.0,121.6 L341.6,124.2 L345.2,126.7 L348.9,129.1 L352.5,131.4 L356.1,133.6 L359.8,135.7 L363.4,137.6 L367.0,139.5 L370.6,141.2 L374.2,142.8 L377.9,144.4 L381.5,145.8 L385.1,147.1 L388.8,148.2 L392.4,149.3 L396.0,150.2 L399.6,151.1 L403.2,151.8 L406.9,152.4 L410.5,152.9 L414.1,153.3 L417.8,153.6 L421.4,153.8 L425.0,153.8 L428.6,153.8 L432.2,153.6 L435.9,153.3 L439.5,152.9 L443.1,152.4 L446.8,151.8 L450.4,151.1 L454.0,150.2 L457.6,149.3 L461.2,148.2 L464.9,147.1 L468.5,145.8 L472.1,144.4 L475.8,142.8 L479.4,141.2 L483.0,139.5 L486.6,137.6 L490.2,135.7 L493.9,133.6 L497.5,131.4 L501.1,129.1 L504.8,126.7 L508.4,124.2 L512.0,121.6 L515.6,118.8 L519.2,115.9 L522.9,113.0 L526.5,109.9 L530.1,106.7 L533.8,103.4 L537.4,100.0 L541.0,96.4 L544.6,92.8 L548.2,89.0 L551.9,85.2 L555.5,81.2 L559.1,77.1 L562.8,72.9 L566.4,68.6 L570.0,64.2" fill="none" class="stroke-b" stroke-width="2.2"/><circle cx="280.0" cy="64.2" r="4" class="fill-a"/><circle cx="316.2" cy="103.4" r="4" class="fill-a"/><circle cx="352.5" cy="131.4" r="4" class="fill-a"/><circle cx="388.8" cy="148.2" r="4" class="fill-a"/><circle cx="425.0" cy="153.8" r="4" class="fill-a"/><circle cx="461.2" cy="148.2" r="4" class="fill-a"/><circle cx="497.5" cy="131.4" r="4" class="fill-a"/><circle cx="533.8" cy="103.4" r="4" class="fill-a"/><circle cx="570.0" cy="64.2" r="4" class="fill-a"/><text x="280.0" y="200.0" text-anchor="middle" class="fig-small">−2</text><text x="352.5" y="200.0" text-anchor="middle" class="fig-small">−1</text><text x="425.0" y="200.0" text-anchor="middle" class="fig-small">0</text><text x="497.5" y="200.0" text-anchor="middle" class="fig-small">1</text><text x="570.0" y="200.0" text-anchor="middle" class="fig-small">2</text><text x="425.0" y="222.0" text-anchor="middle" class="fig-small">t = how much of the extra wiggle you add</text><text x="425.0" y="175.8" text-anchor="middle" class="fig-small">J = 0.87</text><text x="566.0" y="56.2" text-anchor="end" class="fig-small">J = 1.76</text>
</svg>
Figure 2. Left: the hand example. Adding an extra component that changes no prediction raises J from 2 to 3. Right: on a problem with 6 training and 10 extra points, J against the amount t of an extra component is a parabola with its lowest point at t = 0.
```

In general, write $c = (\alpha,\ t\,d)$, where $\alpha$ is the best training-point weight vector and $d$ weights the extra centres. Choose $d$ so that the extra part is zero at every training point, which means $K_{ta}\,d = 0$ for the training-by-extra block $K_{ta}$. Then the training predictions do not change, and the size splits cleanly:

$$
J(c) = J(\alpha, 0) + \lambda\,t^2\, d^{\top}K_{aa}\,d .
$$

Read it aloud: "the objective rises by lambda times $t^2$ times the size of the extra part." The cross term $2t\,\alpha^{\top}K_{ta}d$ vanishes because $K_{ta}d = 0$. In the hand example $d = 1$ and $K_{aa} = 1$, so the gap is $1 \cdot 1^2 \cdot 1 = 1$, matching $3 - 2$. In general $d^{\top}K_{aa}d$ is the squared size of the extra part, which is positive unless that part is the zero function. Figure 2, right, shows the parabola.

Any function with extra centres splits this way. Take its predictions at the $n$ training points and find training-point weights $\alpha$ that reproduce them, which is a linear solve with $K_{tt}$ whenever that block is invertible. The function built from those weights agrees with the original at every training point, so the difference between the two vanishes there. That difference is the leftover. It changes no training prediction and only adds to the size, so the best function has no leftover. The argument leans on the penalty being the function's squared length: a penalty that ignored the leftover would let it stay.

```python
pred = (K @ coeffs)[:n]                      # training predictions only
J = np.sum((y - pred) ** 2) + lam * coeffs @ K @ coeffs
```

## The cost: an n-by-n matrix

The theorem is good news for correctness: $n$ weights are enough, so the system in the previous chapter is the right one. It is bad news for cost, because that system has $n^2$ entries and an $n^3$ solve, as the cost section of that chapter showed.

## Random features: an RBF bump from random cosines

To avoid the matrix, find explicit features $z(x)$ whose dot products approximate the kernel, so $z(x)^{\top}z(x') \approx k(x, x')$. Then ordinary ridge regression on $z$ replaces the kernel solve. For the RBF kernel there is a way to build them from random numbers.

The idea has two steps. First, a cosine of a random projection, averaged over random frequencies, is a Gaussian bump. Picture one cosine $\cos(\omega^{\top}\delta)$ of the difference $\delta = x - x'$. It equals 1 at $\delta = 0$, then ripples up and down, and a fast ripple (large $\omega$) falls away from 1 quickly while a slow one stays near 1. Average many ripples with random speeds and they agree only near $\delta = 0$, so the average starts at 1 and fades with distance, which is the shape of a bump. If $\omega$ is drawn from a normal distribution with variance $2\gamma$ in each coordinate, the average is exactly $\exp(-\gamma\lVert\delta\rVert^2)$. This is where the variance $2\gamma$ comes from: it makes the bump the RBF kernel.

Second, split the difference into separate points, because a dot product needs a function of $x$ times a function of $x'$. The product-to-sum rule turns a product of cosines into $2\cos(\omega^{\top}x + \phi)\cos(\omega^{\top}x' + \phi) = \cos(\omega^{\top}(x - x')) + \cos(\omega^{\top}(x + x') + 2\phi)$. The second term depends on $x + x'$, which is not a similarity, so we want it to vanish. A random phase $\phi$ spread evenly over $[0, 2\pi)$ averages it to zero and leaves $\cos(\omega^{\top}(x - x'))$. The left side is a function of $x$ times a function of $x'$, which is a dot product of one-entry feature lists. So

$$
z(x) = \sqrt{2}\,\cos(\omega^{\top}x + \phi)
$$

has $\mathbb{E}\left[z(x)\,z(x')\right] = k(x, x')$. One draw gives a noisy estimate. Average $D$ independent draws by stacking them into a feature vector of length $D$ and scaling by $1/\sqrt{D}$:

$$
Z = \sqrt{\frac{2}{D}}\,\cos\!\left(X\Omega^{\top} + \phi\right), \qquad Z Z^{\top} \approx K .
$$

Here $\Omega$ is a $D \times d$ matrix of normal draws with variance $2\gamma$, and $\phi$ holds $D$ phases. $Z$ has shape $(n, D)$.

Check it with two one-dimensional points, $x = 0$ and $x' = 1$, at $\gamma = 0.5$. The frequencies have variance 1, and the exact similarity is $e^{-0.5} = 0.6065$. Draw $D = 4$ features with `np.random.default_rng(0)`, drawing $\omega$ and then $\phi$:

| $k$ | $\omega_k$ | $\phi_k$ | $\cos(\phi_k)$ | $\cos(\omega_k + \phi_k)$ | product |
|---|---|---|---|---|---|
| 1 | 0.126 | 5.110 | 0.387 | 0.500 | 0.193 |
| 2 | −0.132 | 5.735 | 0.853 | 0.777 | 0.663 |
| 3 | 0.640 | 3.812 | −0.784 | −0.257 | 0.202 |
| 4 | 0.105 | 4.584 | −0.128 | −0.024 | 0.003 |

The estimate is $\tfrac{2}{4}(0.193 + 0.663 + 0.202 + 0.003) = 0.531$, against 0.6065 exactly. Four features are noisy. The self-similarity of $x = 0$ comes out at $0.755$ and not 1, because a single draw is only right on average. With more draws from the same recipe the estimates are 0.639 at $D = 100$, 0.597 at $D = 2000$, and 0.610 at $D = 50{,}000$.

Averaging $D$ draws shrinks the typical error like $1/\sqrt{D}$, the same way averaging coin flips does. Figure 3 measures it on 10 random points.

```figure
<svg viewBox="0 0 600 236" role="img" aria-label="Mean absolute error between the random-feature Gram matrix and the exact RBF Gram matrix against D, on log-log axes, with a slope minus one half reference.">
<line x1="80" y1="190" x2="550" y2="190" class="stroke-muted"/><line x1="80" y1="20" x2="80" y2="190" class="stroke-muted"/><text x="80.0" y="206.0" text-anchor="middle" class="fig-small">10</text><text x="197.5" y="206.0" text-anchor="middle" class="fig-small">40</text><text x="315.0" y="206.0" text-anchor="middle" class="fig-small">160</text><text x="432.5" y="206.0" text-anchor="middle" class="fig-small">640</text><text x="550.0" y="206.0" text-anchor="middle" class="fig-small">2560</text><text x="74.0" y="24.0" text-anchor="end" class="fig-small">0.4</text><text x="74.0" y="75.2" text-anchor="end" class="fig-small">0.1</text><text x="74.0" y="134.6" text-anchor="end" class="fig-small">0.02</text><text x="74.0" y="194.0" text-anchor="end" class="fig-small">0.004</text><path d="M80.0,40.5 L197.5,66.1 L315.0,91.6 L432.5,117.2 L550.0,142.8" fill="none" class="stroke-muted dashed" stroke-width="2.2"/><path d="M80.0,40.5 L197.5,68.2 L315.0,97.8 L432.5,114.2 L550.0,145.9" fill="none" class="stroke-b" stroke-width="2.2"/><circle cx="80.0" cy="40.5" r="4.5" class="fill-b"/><circle cx="197.5" cy="68.2" r="4.5" class="fill-b"/><circle cx="315.0" cy="97.8" r="4.5" class="fill-b"/><circle cx="432.5" cy="114.2" r="4.5" class="fill-b"/><circle cx="550.0" cy="145.9" r="4.5" class="fill-b"/><text x="315.0" y="67.6" text-anchor="start" class="fig-small">slope −1/2 reference</text><text x="325.0" y="127.8" text-anchor="start" class="fig-small">measured error</text><text x="315.0" y="226.0" text-anchor="middle" class="fig-small">number of random features D (log scale)</text><text x="12.0" y="100.0" text-anchor="start" class="fig-small">mean error</text>
</svg>
Figure 3. Mean absolute error between the random-feature Gram matrix and the exact RBF Gram matrix, against D, averaged over 30 seeds. Notice that the measured line runs roughly parallel to the dashed slope of minus one half: four times as many features cuts the error roughly in half.
```

## Fitting with random features

Once you have $Z$, fit ordinary ridge regression on it. The weights solve a $D \times D$ system, and a prediction is a dot product:

$$
(Z^{\top}Z + \lambda I)\,w = Z^{\top}y, \qquad \hat{y}(x) = z(x)^{\top}w .
$$

Read it aloud: "ridge regression, with the random features as the columns." Because $ZZ^{\top} \approx K$, these predictions approach those of exact kernel ridge regression as $D$ grows.

Test points need the *same* $\Omega$ and $\phi$ as the training points. A fresh draw defines different features, and the learned weights mean nothing for them. Draw from a generator whose state you control, and reuse the draw: the shapes of $\Omega$ and $\phi$ depend only on $D$ and $d$, so a generator with the same seed reproduces them for any number of rows.

```python
omega = rng.normal(size=(D, d)) * np.sqrt(2.0 * gamma)
phase = rng.uniform(0.0, 2.0 * np.pi, size=D)
Z = np.sqrt(2.0 / D) * np.cos(X @ omega.T + phase)
```

> [!KEY]
> Bumps at the training points are enough, because any extra component that vanishes there changes no prediction and only adds to the penalty. That fixes the form of the model at $n$ weights and an $n \times n$ system. Random Fourier features then replace the kernel with $D$ explicit columns, so you fit a $D \times D$ ridge problem and never form $K$.

> [!RECAP]
> - An extra component that vanishes at every training point changes no prediction and adds $\lambda t^2 d^{\top}K_{aa}d$ to the objective, so the best function uses training-point bumps only.
> - The size of a function with weights $c$ is $c^{\top}Kc$, its squared length in feature space.
> - A cosine of a random projection, with frequencies of variance $2\gamma$ and a random phase, averages to the RBF kernel.
> - $Z = \sqrt{2/D}\cos(X\Omega^{\top} + \phi)$ gives $ZZ^{\top} \approx K$, with error shrinking like $1/\sqrt{D}$.
> - Ridge on $Z$ solves a $D \times D$ system, and test points must reuse the same $\Omega$ and $\phi$.

## Build it step by step

The starter provides `rbf_kernel` and `kernel_ridge_fit` from the earlier exercises. Build three functions.

1. **`representer_check(K, y, lam, coeffs)`**. `K` is `(N, N)`, `coeffs` is `(N,)`, and `N` is at least `len(y)`. Return one float: the summed squared error on the first `len(y)` entries of `K @ coeffs`, plus `lam * coeffs @ K @ coeffs`. The usual mistake is a mean in place of a sum, or slicing `coeffs` instead of the predictions.
2. **`random_fourier_features(X, D, gamma, rng)`**. `X` is `(n, d)` and the return is `(n, D)`. Draw `omega` with shape `(D, d)` and standard deviation `np.sqrt(2 * gamma)`, then `phase` with shape `(D,)` from the interval 0 to $2\pi$. The usual mistake is scaling `omega` by `2 * gamma` instead of its square root, which gives variance $4\gamma^2$ in place of $2\gamma$.
3. **`ridge_on_features(Z, y, lam)`**. `Z` is `(n, D)`. Solve `(Z.T @ Z + lam * np.eye(D)) w = Z.T @ y` and return `w` with shape `(D,)`. The usual mistake is solving the $n \times n$ system `Z @ Z.T`, which is the kernel route and costs more when $n$ is larger than $D$.

## Cost and scaling

The two routes compare as follows, with $n$ training points, $d$ inputs, and $D$ random features:

| Method | Build | Solve | Memory | One prediction |
|---|---|---|---|---|
| Exact kernel ridge | $n^2 d$ | $n^3$ | $n^2$ | about $n d$ |
| Random features plus ridge | $n D d$ | $n D^2 + D^3$ | $n D$ | about $D d$ |

Random features grow linearly in $n$ for fixed $D$. At $n = 100{,}000$ and $D = 1{,}000$, the matrix $Z$ holds $10^8$ numbers, about 0.8 GB, against the 80 GB for $K$. The price is accuracy: you choose $D$, and the kernel error falls only like $1/\sqrt{D}$.

## Where this breaks

**Too few features.** The Gram matrix is noisy at small $D$, and the fit inherits the noise. Increase $D$ until predictions stop changing.

**Other kernels.** This construction is for the RBF kernel. A Nyström approximation builds features from a random subset of the training points and works for any kernel.

**Forgetting to share the draw.** Features for test points drawn separately from the training features give predictions with no relation to the fit.

## The exercise

You implement `representer_check`, `random_fourier_features`, and `ridge_on_features`. The checks confirm the hand example, where $J$ is 2 and 3, and that the function sums over training rows only. They add an extra component that vanishes at the training points and confirm that the objective rises by exactly $\lambda t^2 d^{\top}K_{aa}d$. They check that the random features have the right shape and that their Gram matrix approaches the exact RBF Gram matrix as $D$ grows, at about the $1/\sqrt{D}$ rate. They check that two calls with the same seed share one draw. Finally they check that `ridge_on_features` has zero gradient at its answer, confirmed with finite differences, and that its predictions approach exact kernel ridge predictions.
