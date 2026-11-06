---
title: Early stopping is a ridge penalty in disguise
slug: 066-early-stopping-ridge
part: "Part 4 — Neural networks"
summary: Gradient descent from zero learns steep directions first, so stopping after k steps shrinks the flat ones, as a ridge penalty would.
minutes: 40
gpu: false
objectives:
  - Explain from a stretched-bowl picture why gradient descent from zero learns steep directions before flat ones.
  - Derive the weight after k steps on one coordinate, and the ridge penalty that gives the same weight.
  - State when that match holds, and why it needs a step size no larger than one over the curvature.
  - Pick a stopping step from a noisy validation curve, using patience.
prereqs:
  - 009-step-size-and-long-valleys
  - 011-ridge-penalty-and-closed-form
  - 065-weight-decay
exercise: 066-early-stopping-ridge
---

# Early stopping is a ridge penalty in disguise

You train a network and watch two numbers. The training loss falls at every step. The validation loss, measured on data the network never trains on, falls too, bottoms out at step 40, and then climbs while training keeps improving. Figure 1 shows the shape. You stop at step 40 and keep the weights from there.

You did not write a penalty anywhere, yet the result is as if you had. Without it, the later steps would keep fitting noise that is specific to your training set, and the model would get worse on new data. This chapter answers two questions. Why does the validation loss turn up? And why is stopping at step 40 the same kind of act as adding a ridge penalty?

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Line chart on a log scale. Training loss falls steadily from about 4 to 0.014. Validation loss falls to its minimum of 0.118 at step 40 and then rises to 0.40 by step 200.">
<line x1="60.0" y1="226.0" x2="560.0" y2="226.0" class="stroke-muted"/>
<line x1="60.0" y1="226.0" x2="60.0" y2="36.0" class="stroke-muted"/>
<line x1="60.0" y1="226.0" x2="60.0" y2="230.0" class="stroke-muted"/>
<text x="60.0" y="243.0" text-anchor="middle" class="fig-small">0</text>
<line x1="160.0" y1="226.0" x2="160.0" y2="230.0" class="stroke-muted"/>
<text x="160.0" y="243.0" text-anchor="middle" class="fig-small">40</text>
<line x1="310.0" y1="226.0" x2="310.0" y2="230.0" class="stroke-muted"/>
<text x="310.0" y="243.0" text-anchor="middle" class="fig-small">100</text>
<line x1="435.0" y1="226.0" x2="435.0" y2="230.0" class="stroke-muted"/>
<text x="435.0" y="243.0" text-anchor="middle" class="fig-small">150</text>
<line x1="560.0" y1="226.0" x2="560.0" y2="230.0" class="stroke-muted"/>
<text x="560.0" y="243.0" text-anchor="middle" class="fig-small">200</text>
<text x="310.0" y="260.0" text-anchor="middle" class="fig-small">step k</text>
<text x="60.0" y="28.0" text-anchor="start" class="fig-small">loss, log scale</text>
<line x1="60" y1="226.0" x2="560" y2="226.0" class="stroke-grid"/>
<text x="53.0" y="230.0" text-anchor="end" class="fig-small">0.01</text>
<line x1="60" y1="155.6" x2="560" y2="155.6" class="stroke-grid"/>
<text x="53.0" y="159.6" text-anchor="end" class="fig-small">0.1</text>
<line x1="60" y1="85.2" x2="560" y2="85.2" class="stroke-grid"/>
<text x="53.0" y="89.2" text-anchor="end" class="fig-small">1</text>
<path d="M60.0,43.2 L62.5,66.1 L65.0,78.2 L67.5,84.1 L70.0,87.2 L72.5,89.3 L75.0,90.9 L77.5,92.2 L80.0,93.4 L82.5,94.4 L85.0,95.4 L87.5,96.3 L90.0,97.1 L92.5,97.9 L95.0,98.7 L97.5,99.5 L100.0,100.2 L102.5,101.0 L105.0,101.7 L107.5,102.4 L110.0,103.2 L112.5,103.9 L115.0,104.6 L117.5,105.3 L120.0,105.9 L122.5,106.6 L125.0,107.3 L127.5,108.0 L130.0,108.6 L132.5,109.3 L135.0,110.0 L137.5,110.6 L140.0,111.3 L142.5,111.9 L145.0,112.6 L147.5,113.2 L150.0,113.8 L152.5,114.5 L155.0,115.1 L157.5,115.8 L160.0,116.4 L162.5,117.0 L165.0,117.6 L167.5,118.3 L170.0,118.9 L172.5,119.5 L175.0,120.2 L177.5,120.8 L180.0,121.4 L182.5,122.0 L185.0,122.7 L187.5,123.3 L190.0,123.9 L192.5,124.5 L195.0,125.1 L197.5,125.8 L200.0,126.4 L202.5,127.0 L205.0,127.6 L207.5,128.2 L210.0,128.9 L212.5,129.5 L215.0,130.1 L217.5,130.7 L220.0,131.3 L222.5,131.9 L225.0,132.6 L227.5,133.2 L230.0,133.8 L232.5,134.4 L235.0,135.0 L237.5,135.6 L240.0,136.3 L242.5,136.9 L245.0,137.5 L247.5,138.1 L250.0,138.7 L252.5,139.3 L255.0,140.0 L257.5,140.6 L260.0,141.2 L262.5,141.8 L265.0,142.4 L267.5,143.0 L270.0,143.6 L272.5,144.3 L275.0,144.9 L277.5,145.5 L280.0,146.1 L282.5,146.7 L285.0,147.3 L287.5,147.9 L290.0,148.6 L292.5,149.2 L295.0,149.8 L297.5,150.4 L300.0,151.0 L302.5,151.6 L305.0,152.3 L307.5,152.9 L310.0,153.5 L312.5,154.1 L315.0,154.7 L317.5,155.3 L320.0,155.9 L322.5,156.6 L325.0,157.2 L327.5,157.8 L330.0,158.4 L332.5,159.0 L335.0,159.6 L337.5,160.2 L340.0,160.9 L342.5,161.5 L345.0,162.1 L347.5,162.7 L350.0,163.3 L352.5,163.9 L355.0,164.5 L357.5,165.2 L360.0,165.8 L362.5,166.4 L365.0,167.0 L367.5,167.6 L370.0,168.2 L372.5,168.8 L375.0,169.5 L377.5,170.1 L380.0,170.7 L382.5,171.3 L385.0,171.9 L387.5,172.5 L390.0,173.1 L392.5,173.8 L395.0,174.4 L397.5,175.0 L400.0,175.6 L402.5,176.2 L405.0,176.8 L407.5,177.4 L410.0,178.1 L412.5,178.7 L415.0,179.3 L417.5,179.9 L420.0,180.5 L422.5,181.1 L425.0,181.8 L427.5,182.4 L430.0,183.0 L432.5,183.6 L435.0,184.2 L437.5,184.8 L440.0,185.4 L442.5,186.1 L445.0,186.7 L447.5,187.3 L450.0,187.9 L452.5,188.5 L455.0,189.1 L457.5,189.7 L460.0,190.4 L462.5,191.0 L465.0,191.6 L467.5,192.2 L470.0,192.8 L472.5,193.4 L475.0,194.0 L477.5,194.7 L480.0,195.3 L482.5,195.9 L485.0,196.5 L487.5,197.1 L490.0,197.7 L492.5,198.3 L495.0,199.0 L497.5,199.6 L500.0,200.2 L502.5,200.8 L505.0,201.4 L507.5,202.0 L510.0,202.6 L512.5,203.3 L515.0,203.9 L517.5,204.5 L520.0,205.1 L522.5,205.7 L525.0,206.3 L527.5,206.9 L530.0,207.6 L532.5,208.2 L535.0,208.8 L537.5,209.4 L540.0,210.0 L542.5,210.6 L545.0,211.2 L547.5,211.9 L550.0,212.5 L552.5,213.1 L555.0,213.7 L557.5,214.3 L560.0,214.9" fill="none" class="stroke-a" stroke-width="2.4"/>
<path d="M60.0,44.4 L62.5,70.1 L65.0,86.1 L67.5,95.8 L70.0,102.3 L72.5,107.5 L75.0,111.8 L77.5,115.6 L80.0,118.9 L82.5,122.0 L85.0,124.7 L87.5,127.1 L90.0,129.4 L92.5,131.4 L95.0,133.2 L97.5,135.0 L100.0,136.5 L102.5,138.0 L105.0,139.3 L107.5,140.5 L110.0,141.6 L112.5,142.7 L115.0,143.6 L117.5,144.5 L120.0,145.3 L122.5,146.0 L125.0,146.7 L127.5,147.3 L130.0,147.8 L132.5,148.3 L135.0,148.7 L137.5,149.1 L140.0,149.4 L142.5,149.7 L145.0,149.9 L147.5,150.1 L150.0,150.3 L152.5,150.4 L155.0,150.4 L157.5,150.5 L160.0,150.5 L162.5,150.5 L165.0,150.4 L167.5,150.3 L170.0,150.2 L172.5,150.0 L175.0,149.9 L177.5,149.7 L180.0,149.5 L182.5,149.3 L185.0,149.0 L187.5,148.8 L190.0,148.5 L192.5,148.2 L195.0,147.9 L197.5,147.6 L200.0,147.3 L202.5,147.0 L205.0,146.7 L207.5,146.3 L210.0,146.0 L212.5,145.6 L215.0,145.3 L217.5,144.9 L220.0,144.6 L222.5,144.2 L225.0,143.8 L227.5,143.4 L230.0,143.1 L232.5,142.7 L235.0,142.3 L237.5,141.9 L240.0,141.6 L242.5,141.2 L245.0,140.8 L247.5,140.4 L250.0,140.1 L252.5,139.7 L255.0,139.3 L257.5,138.9 L260.0,138.6 L262.5,138.2 L265.0,137.8 L267.5,137.5 L270.0,137.1 L272.5,136.8 L275.0,136.4 L277.5,136.0 L280.0,135.7 L282.5,135.3 L285.0,135.0 L287.5,134.7 L290.0,134.3 L292.5,134.0 L295.0,133.7 L297.5,133.3 L300.0,133.0 L302.5,132.7 L305.0,132.3 L307.5,132.0 L310.0,131.7 L312.5,131.4 L315.0,131.1 L317.5,130.8 L320.0,130.5 L322.5,130.2 L325.0,129.9 L327.5,129.6 L330.0,129.3 L332.5,129.0 L335.0,128.7 L337.5,128.5 L340.0,128.2 L342.5,127.9 L345.0,127.6 L347.5,127.4 L350.0,127.1 L352.5,126.9 L355.0,126.6 L357.5,126.3 L360.0,126.1 L362.5,125.8 L365.0,125.6 L367.5,125.3 L370.0,125.1 L372.5,124.9 L375.0,124.6 L377.5,124.4 L380.0,124.2 L382.5,123.9 L385.0,123.7 L387.5,123.5 L390.0,123.3 L392.5,123.0 L395.0,122.8 L397.5,122.6 L400.0,122.4 L402.5,122.2 L405.0,122.0 L407.5,121.8 L410.0,121.6 L412.5,121.4 L415.0,121.2 L417.5,121.0 L420.0,120.8 L422.5,120.6 L425.0,120.4 L427.5,120.2 L430.0,120.1 L432.5,119.9 L435.0,119.7 L437.5,119.5 L440.0,119.4 L442.5,119.2 L445.0,119.0 L447.5,118.8 L450.0,118.7 L452.5,118.5 L455.0,118.3 L457.5,118.2 L460.0,118.0 L462.5,117.9 L465.0,117.7 L467.5,117.6 L470.0,117.4 L472.5,117.3 L475.0,117.1 L477.5,117.0 L480.0,116.8 L482.5,116.7 L485.0,116.5 L487.5,116.4 L490.0,116.3 L492.5,116.1 L495.0,116.0 L497.5,115.8 L500.0,115.7 L502.5,115.6 L505.0,115.5 L507.5,115.3 L510.0,115.2 L512.5,115.1 L515.0,115.0 L517.5,114.8 L520.0,114.7 L522.5,114.6 L525.0,114.5 L527.5,114.4 L530.0,114.2 L532.5,114.1 L535.0,114.0 L537.5,113.9 L540.0,113.8 L542.5,113.7 L545.0,113.6 L547.5,113.5 L550.0,113.4 L552.5,113.3 L555.0,113.1 L557.5,113.0 L560.0,112.9" fill="none" class="stroke-b" stroke-width="2.4"/>
<line x1="160.0" y1="226.0" x2="160.0" y2="150.5" class="stroke-muted dashed" stroke-width="1.2"/>
<circle cx="160.0" cy="150.5" r="5.5" class="fill-b"/>
<text x="166.0" y="219.1" text-anchor="start" class="fig-bold">best step: 40</text>
<text x="485.0" y="216.5" text-anchor="end" class="fig-small">training loss</text>
<text x="560.0" y="104.9" text-anchor="end" class="fig-small">validation loss</text>
</svg>
Figure 1. Training loss keeps falling, to 0.014 at step 200. Validation loss reaches its lowest value, 0.118, at step 40, then climbs to 0.404. Stopping at the dot keeps the weights that generalise best.
```

## What you already know

- Held-out data scores a model on examples it did not train on, so the validation loss shows how well it generalises.
- A gradient step moves each weight against its slope by a step size $\alpha$.
- Curvature ("How big a step? Step size and long valleys") says how sharply a loss bends along one axis. On one axis, a step multiplies your distance from the bottom by $1 - \alpha \times \text{curvature}$.
- Ridge ("Ridge: a fee on large weights") adds the fee $\tfrac{\lambda}{2}\|w\|^2$ to the loss, which pulls weights toward zero.
- Weight decay ("Weight decay: a pull toward zero") is that same pull applied on every gradient step.

## Why the validation loss turns up

To see this, use the simplest loss with a shape: a bowl. Rotate the weights so that the bowl's axes do not interact. Along each such axis, called a **direction**, the loss is a parabola $\tfrac{1}{2}c\,(w - w^*)^2$. Here $w^*$ is the best weight for that direction on the training data, and the **curvature** $c$ says how sharply the parabola bends. (Linear algebra calls these curvatures eigenvalues, which is why the code says `eigenvalue`.) Gradient descent treats each direction on its own, so a single direction is enough to follow the story.

Figure 1 uses four directions with curvatures 5, 2, 0.5, and 0.1. The true best weight in every direction is 1. The training data gets it wrong by different amounts: the training optimum is $[1.0, 0.7, 0.8, 4.0]$. The flat direction, with curvature 0.1, is the worst. A flat direction is one where changing the weight hardly changes the training loss, so the data hardly pins it down. Each noisy point shifts the best weight by an amount that grows as the curvature shrinks, because a flat bowl barely notices a weight that is off, so noise decides where $w^*$ lands. Here that is 4.0 instead of 1.

Gradient descent starts at zero and learns steep directions first, which are nearly right, and flat directions last, which are far wrong. The flat direction passes its true value of 1 near step 29 and keeps going toward 4.0. Validation loss keeps falling a little longer, while the other directions settle, then turns up at step 40. By step 200 it has risen from 0.118 to 0.404, while the training loss, measured from its lowest value, is down to 0.014.

## A stretched bowl

The word "first" needs a picture. Figure 2 draws a bowl with two directions: a steep one with curvature 10 and a flat one with curvature 0.5, so the bowl is a long narrow valley. The answer $w^*$ sits at 1 on the flat axis and 0.6 on the steep axis, and gradient descent at step size 0.05 starts at the origin. The steep coordinate multiplies its remaining distance by $1 - 0.05 \times 10 = 0.5$ each step, so after 5 steps it has covered 97% of the way. The flat coordinate multiplies its remaining distance by $1 - 0.05 \times 0.5 = 0.975$, so after 5 steps it has covered only 12%.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Contour plot of a stretched bowl with its minimum marked at (1, 0.6). The gradient descent path from the origin climbs the steep direction in a few steps, then crawls along the flat direction toward the minimum, reaching 12 percent of the flat distance at 5 steps, 40 percent at 20 steps and 92 percent at 100 steps.">
<defs><marker id="c074-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<line x1="60.0" y1="228.0" x2="560.0" y2="228.0" class="stroke-muted"/>
<line x1="60.0" y1="228.0" x2="60.0" y2="28.0" class="stroke-muted"/>
<line x1="89.4" y1="228.0" x2="89.4" y2="232.0" class="stroke-muted"/>
<text x="89.4" y="245.0" text-anchor="middle" class="fig-small">0</text>
<line x1="236.5" y1="228.0" x2="236.5" y2="232.0" class="stroke-muted"/>
<text x="236.5" y="245.0" text-anchor="middle" class="fig-small">0.5</text>
<line x1="383.5" y1="228.0" x2="383.5" y2="232.0" class="stroke-muted"/>
<text x="383.5" y="245.0" text-anchor="middle" class="fig-small">1</text>
<line x1="530.6" y1="228.0" x2="530.6" y2="232.0" class="stroke-muted"/>
<text x="530.6" y="245.0" text-anchor="middle" class="fig-small">1.5</text>
<line x1="56.0" y1="216.2" x2="60.0" y2="216.2" class="stroke-muted"/>
<text x="53.0" y="220.2" text-anchor="end" class="fig-small">0</text>
<line x1="56.0" y1="98.6" x2="60.0" y2="98.6" class="stroke-muted"/>
<text x="53.0" y="102.6" text-anchor="end" class="fig-small">0.5</text>
<text x="310.0" y="262.0" text-anchor="middle" class="fig-small">w, flat direction (curvature 0.5)</text>
<text x="60.0" y="20.0" text-anchor="start" class="fig-small">w, steep direction (curvature 10)</text>
<path d="M442.4,75.1 L442.3,74.5 L442.0,73.9 L441.6,73.4 L441.0,72.9 L440.3,72.3 L439.4,71.8 L438.4,71.3 L437.2,70.7 L435.8,70.2 L434.3,69.8 L432.7,69.3 L430.9,68.8 L429.0,68.4 L427.0,68.0 L424.8,67.6 L422.6,67.2 L420.2,66.8 L417.7,66.5 L415.2,66.2 L412.5,65.9 L409.7,65.6 L406.9,65.4 L404.1,65.2 L401.1,65.0 L398.1,64.9 L395.1,64.7 L392.0,64.6 L389.0,64.6 L385.9,64.5 L382.8,64.5 L379.6,64.6 L376.6,64.6 L373.5,64.7 L370.4,64.8 L367.4,64.9 L364.5,65.1 L361.6,65.3 L358.7,65.5 L355.9,65.8 L353.2,66.0 L350.6,66.3 L348.1,66.7 L345.7,67.0 L343.3,67.4 L341.1,67.8 L339.0,68.2 L337.1,68.6 L335.2,69.1 L333.5,69.5 L332.0,70.0 L330.5,70.5 L329.3,71.0 L328.1,71.5 L327.2,72.0 L326.4,72.6 L325.7,73.1 L325.2,73.7 L324.9,74.2 L324.7,74.8 L324.7,75.3 L324.9,75.9 L325.2,76.4 L325.7,77.0 L326.4,77.5 L327.2,78.1 L328.1,78.6 L329.3,79.1 L330.5,79.6 L332.0,80.1 L333.5,80.6 L335.2,81.1 L337.1,81.5 L339.0,81.9 L341.1,82.4 L343.3,82.7 L345.7,83.1 L348.1,83.5 L350.6,83.8 L353.2,84.1 L355.9,84.4 L358.7,84.6 L361.6,84.8 L364.5,85.0 L367.4,85.2 L370.4,85.3 L373.5,85.4 L376.6,85.5 L379.6,85.6 L382.8,85.6 L385.9,85.6 L389.0,85.5 L392.0,85.5 L395.1,85.4 L398.1,85.3 L401.1,85.1 L404.1,84.9 L406.9,84.7 L409.7,84.5 L412.5,84.2 L415.2,83.9 L417.7,83.6 L420.2,83.3 L422.6,82.9 L424.8,82.5 L427.0,82.1 L429.0,81.7 L430.9,81.3 L432.7,80.8 L434.3,80.4 L435.8,79.9 L437.2,79.4 L438.4,78.9 L439.4,78.3 L440.3,77.8 L441.0,77.3 L441.6,76.7 L442.0,76.2 L442.3,75.6 L442.4,75.1" fill="none" class="stroke-grid" stroke-width="1.2"/>
<path d="M501.2,75.1 L501.0,73.9 L500.5,72.8 L499.7,71.7 L498.6,70.6 L497.1,69.6 L495.3,68.5 L493.2,67.5 L490.8,66.4 L488.1,65.4 L485.2,64.5 L481.9,63.5 L478.3,62.6 L474.5,61.7 L470.5,60.9 L466.2,60.1 L461.6,59.3 L456.9,58.6 L451.9,57.9 L446.8,57.3 L441.5,56.7 L436.0,56.2 L430.3,55.8 L424.6,55.3 L418.7,55.0 L412.7,54.7 L406.7,54.4 L400.6,54.2 L394.4,54.1 L388.2,54.0 L382.0,54.0 L375.8,54.1 L369.6,54.2 L363.4,54.3 L357.4,54.5 L351.3,54.8 L345.4,55.1 L339.6,55.5 L333.9,56.0 L328.3,56.5 L322.9,57.0 L317.7,57.6 L312.6,58.3 L307.8,59.0 L303.1,59.7 L298.7,60.5 L294.5,61.3 L290.6,62.2 L286.9,63.1 L283.5,64.0 L280.4,64.9 L277.5,65.9 L275.0,66.9 L272.7,68.0 L270.8,69.0 L269.2,70.1 L267.9,71.2 L266.9,72.3 L266.3,73.4 L265.9,74.5 L265.9,75.6 L266.3,76.7 L266.9,77.8 L267.9,78.9 L269.2,80.0 L270.8,81.1 L272.7,82.1 L275.0,83.2 L277.5,84.2 L280.4,85.2 L283.5,86.1 L286.9,87.1 L290.6,88.0 L294.5,88.8 L298.7,89.6 L303.1,90.4 L307.8,91.2 L312.6,91.9 L317.7,92.5 L322.9,93.1 L328.3,93.6 L333.9,94.1 L339.6,94.6 L345.4,95.0 L351.3,95.3 L357.4,95.6 L363.4,95.8 L369.6,96.0 L375.8,96.1 L382.0,96.1 L388.2,96.1 L394.4,96.0 L400.6,95.9 L406.7,95.7 L412.7,95.4 L418.7,95.1 L424.6,94.8 L430.3,94.4 L436.0,93.9 L441.5,93.4 L446.8,92.8 L451.9,92.2 L456.9,91.5 L461.6,90.8 L466.2,90.0 L470.5,89.2 L474.5,88.4 L478.3,87.5 L481.9,86.6 L485.2,85.7 L488.1,84.7 L490.8,83.7 L493.2,82.7 L495.3,81.6 L497.1,80.6 L498.6,79.5 L499.7,78.4 L500.5,77.3 L501.0,76.2 L501.2,75.1" fill="none" class="stroke-grid" stroke-width="1.2"/>
<path d="M560.0,75.1 L559.8,73.4 L559.0,71.7 L557.8,70.1 L556.1,68.4 L553.9,66.8 L551.2,65.2 L548.1,63.7 L544.5,62.1 L540.4,60.6 L536.0,59.2 L531.1,57.7 L525.7,56.4 L520.0,55.1 L513.9,53.8 L507.5,52.6 L500.7,51.5 L493.6,50.4 L486.1,49.4 L478.4,48.4 L470.4,47.6 L462.2,46.8 L453.7,46.1 L445.1,45.5 L436.3,44.9 L427.3,44.5 L418.2,44.1 L409.1,43.8 L399.8,43.6 L390.5,43.5 L381.2,43.5 L371.9,43.6 L362.6,43.7 L353.4,44.0 L344.3,44.3 L335.2,44.7 L326.3,45.2 L317.6,45.8 L309.1,46.4 L300.7,47.2 L292.6,48.0 L284.8,48.9 L277.2,49.9 L269.9,50.9 L262.9,52.0 L256.3,53.2 L250.0,54.4 L244.1,55.7 L238.6,57.0 L233.5,58.4 L228.8,59.9 L224.5,61.4 L220.7,62.9 L217.4,64.4 L214.4,66.0 L212.0,67.6 L210.1,69.3 L208.6,70.9 L207.6,72.6 L207.1,74.2 L207.1,75.9 L207.6,77.6 L208.6,79.2 L210.1,80.9 L212.0,82.5 L214.4,84.1 L217.4,85.7 L220.7,87.2 L224.5,88.8 L228.8,90.2 L233.5,91.7 L238.6,93.1 L244.1,94.4 L250.0,95.7 L256.3,96.9 L262.9,98.1 L269.9,99.2 L277.2,100.3 L284.8,101.2 L292.6,102.1 L300.7,102.9 L309.1,103.7 L317.6,104.3 L326.3,104.9 L335.2,105.4 L344.3,105.8 L353.4,106.2 L362.6,106.4 L371.9,106.6 L381.2,106.6 L390.5,106.6 L399.8,106.5 L409.1,106.3 L418.2,106.0 L427.3,105.6 L436.3,105.2 L445.1,104.6 L453.7,104.0 L462.2,103.3 L470.4,102.5 L478.4,101.7 L486.1,100.7 L493.6,99.7 L500.7,98.7 L507.5,97.5 L513.9,96.3 L520.0,95.1 L525.7,93.7 L531.1,92.4 L536.0,91.0 L540.4,89.5 L544.5,88.0 L548.1,86.5 L551.2,84.9 L553.9,83.3 L556.1,81.7 L557.8,80.0 L559.0,78.4 L559.8,76.7 L560.0,75.1" fill="none" class="stroke-grid" stroke-width="1.2"/>
<path d="M89.4,216.2 L96.8,145.6 L103.9,110.4 L110.9,92.7 L117.7,83.9 L124.4,79.5 L130.9,77.3 L137.2,76.2 L143.3,75.6 L149.3,75.3 L155.2,75.2 L160.9,75.1 L166.5,75.1 L171.9,75.1 L177.2,75.1 L182.3,75.1 L187.4,75.1 L192.3,75.1 L197.1,75.1 L201.7,75.1 L206.3,75.1 L210.7,75.1 L215.0,75.1 L219.2,75.1 L223.3,75.1 L227.3,75.1 L231.3,75.1 L235.1,75.1 L238.8,75.1 L242.4,75.1 L245.9,75.1 L249.4,75.1 L252.7,75.1 L256.0,75.1 L259.2,75.1 L262.3,75.1 L265.3,75.1 L268.3,75.1 L271.1,75.1 L274.0,75.1 L276.7,75.1 L279.4,75.1 L282.0,75.1 L284.5,75.1 L287.0,75.1 L289.4,75.1 L291.8,75.1 L294.0,75.1 L296.3,75.1 L298.5,75.1 L300.6,75.1 L302.7,75.1 L304.7,75.1 L306.7,75.1 L308.6,75.1 L310.5,75.1 L312.3,75.1 L314.1,75.1 L315.8,75.1 L317.5,75.1 L319.1,75.1 L320.8,75.1 L322.3,75.1 L323.9,75.1 L325.3,75.1 L326.8,75.1 L328.2,75.1 L329.6,75.1 L330.9,75.1 L332.3,75.1 L333.5,75.1 L334.8,75.1 L336.0,75.1 L337.2,75.1 L338.4,75.1 L339.5,75.1 L340.6,75.1 L341.7,75.1 L342.7,75.1 L343.7,75.1 L344.7,75.1 L345.7,75.1 L346.6,75.1 L347.6,75.1 L348.5,75.1 L349.3,75.1 L350.2,75.1 L351.0,75.1 L351.8,75.1 L352.6,75.1 L353.4,75.1 L354.2,75.1 L354.9,75.1 L355.6,75.1 L356.3,75.1 L357.0,75.1 L357.6,75.1 L358.3,75.1 L358.9,75.1 L359.5,75.1 L360.1,75.1" fill="none" class="stroke-b" stroke-width="2.4"/>
<circle cx="124.4" cy="79.5" r="5" class="fill-b"/>
<circle cx="206.3" cy="75.1" r="5" class="fill-b"/>
<circle cx="360.1" cy="75.1" r="5" class="fill-b"/>
<text x="118.4" y="67.5" text-anchor="start" class="fig-bold">5 steps</text>
<text x="206.3" y="63.1" text-anchor="middle" class="fig-bold">20 steps</text>
<text x="360.1" y="63.1" text-anchor="middle" class="fig-bold">100 steps</text>
<circle cx="89.4" cy="216.2" r="4.5" class="fill-ink"/>
<text x="99.4" y="210.2" text-anchor="start" class="fig-small">start (0, 0)</text>
<circle cx="383.5" cy="75.1" r="6" class="fill-a"/>
<text x="392.5" y="93.1" text-anchor="start" class="fig-bold">w*</text>
</svg>
Figure 2. Contours of a stretched bowl whose answer w* is at (1, 0.6), with gradient descent from the origin at step size 0.05. The path rises almost straight up first, then crawls sideways. The steep coordinate is 97% of the way to its answer after 5 steps, while the flat coordinate has covered only 12%. It needs about 100 steps to get most of the way (92%).
```

Stop after 5 steps and you hold a weight that is nearly right in the steep direction and still near zero in the flat one. That is what a pull toward zero does: it wins against a weak data gradient and loses against a strong one. Flat directions have the weakest gradient, so the pull holds them near zero. Early stopping gets the same result without any pull. It never lets the flat directions travel far.

## One direction, worked by hand

Make it exact on one direction. Take curvature $c = 2$, step size $\alpha = 0.1$, and $w^* = 1$. Start at $w = 0$. The slope is $c\,(w - w^*) = 2 \times (0 - 1) = -2$, so one step lands at $0 - 0.1 \times (-2) = 0.2$. The distance from $w^*$ went from 1 to 0.8, which is $1 - \alpha c = 0.8$ times the old distance. That factor is the same at every step, so after $k$ steps the distance is $0.8^k$ times the starting distance:

$$
w_k = \big(1 - (1 - \alpha c)^k\big)\, w^*
$$

Read it aloud: "the weight has covered all but a fraction $(1-\alpha c)^k$ of the way to the answer." With $k = 10$ the weight is $1 - 0.8^{10} = 0.89263$.

Now ridge on the same direction. The loss becomes $\tfrac{1}{2}c\,(w - w^*)^2 + \tfrac{1}{2}\lambda w^2$. Its slope is $c\,(w - w^*) + \lambda w$, which is zero at

$$
w = \frac{c}{c + \lambda}\, w^*
$$

The fraction $c/(c+\lambda)$ is the share of $w^*$ that survives the penalty. It is 1 with no penalty and near 0 with a huge one. To match gradient descent, set the surviving share equal to the share covered, $\frac{c}{c+\lambda} = 1 - b$ with $b = (1-\alpha c)^k$, and solve for $\lambda$:

$$
\lambda = \frac{c\, b}{1 - b}
$$

Read it aloud: "the penalty is the curvature times the fraction still to go, divided by the fraction already done." In numpy:

```python
b = (1.0 - step * eigenvalue) ** iterations
penalty = eigenvalue * b / (1.0 - b)
```

With $c = 2$ and $\alpha = 0.1$, the fraction still to go is $0.8^k$:

| Steps $k$ | Still to go $0.8^k$ | Penalty $\lambda$ | Weight both ways |
|---|---|---|---|
| 0 | 1 | infinite | 0 |
| 1 | 0.8 | $2 \times 0.8 / 0.2 = 8$ | 0.2 |
| 10 | 0.10737 | 0.24058 | 0.89263 |

At $k = 0$ nothing has been learned, which is exactly what an infinitely strong penalty does. As $k$ grows the penalty falls to zero. Figure 3 shows both curves. Fewer steps mean a stronger penalty, so the step count is a dial.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Three panels. Top left: the fraction of the answer reached rises from 0 to nearly 1 over 60 steps. Bottom left: the equivalent ridge penalty is infinite at step 0, 8 at step 1, 0.241 at step 10, and falls toward 0. Right: bars show that after 25 steps curvature 0.1 has reached 22 percent, curvature 1 has reached 93 percent, and curvature 5 has reached 100 percent.">
<text x="50.0" y="22.0" text-anchor="start" class="fig-bold">Fraction of w* reached, 1 − 0.8^k</text>
<line x1="50.0" y1="114.0" x2="350.0" y2="114.0" class="stroke-muted"/>
<line x1="50.0" y1="114.0" x2="50.0" y2="34.0" class="stroke-muted"/>
<line x1="50.0" y1="114.0" x2="50.0" y2="118.0" class="stroke-muted"/>
<text x="50.0" y="131.0" text-anchor="middle" class="fig-small">0</text>
<line x1="150.0" y1="114.0" x2="150.0" y2="118.0" class="stroke-muted"/>
<text x="150.0" y="131.0" text-anchor="middle" class="fig-small">20</text>
<line x1="250.0" y1="114.0" x2="250.0" y2="118.0" class="stroke-muted"/>
<text x="250.0" y="131.0" text-anchor="middle" class="fig-small">40</text>
<line x1="350.0" y1="114.0" x2="350.0" y2="118.0" class="stroke-muted"/>
<text x="350.0" y="131.0" text-anchor="middle" class="fig-small">60</text>
<line x1="46.0" y1="114.0" x2="50.0" y2="114.0" class="stroke-muted"/>
<text x="43.0" y="118.0" text-anchor="end" class="fig-small">0</text>
<line x1="46.0" y1="37.8" x2="50.0" y2="37.8" class="stroke-muted"/>
<text x="43.0" y="41.8" text-anchor="end" class="fig-small">1</text>
<path d="M50.0,114.0 L55.0,98.8 L60.0,86.6 L65.0,76.8 L70.0,69.0 L75.0,62.8 L80.0,57.8 L85.0,53.8 L90.0,50.6 L95.0,48.0 L100.0,46.0 L105.0,44.4 L110.0,43.0 L115.0,42.0 L120.0,41.2 L125.0,40.5 L130.0,40.0 L135.0,39.5 L140.0,39.2 L145.0,38.9 L150.0,38.7 L155.0,38.5 L160.0,38.4 L165.0,38.3 L170.0,38.2 L175.0,38.1 L180.0,38.0 L185.0,38.0 L190.0,38.0 L195.0,37.9 L200.0,37.9 L205.0,37.9 L210.0,37.9 L215.0,37.9 L220.0,37.8 L225.0,37.8 L230.0,37.8 L235.0,37.8 L240.0,37.8 L245.0,37.8 L250.0,37.8 L255.0,37.8 L260.0,37.8 L265.0,37.8 L270.0,37.8 L275.0,37.8 L280.0,37.8 L285.0,37.8 L290.0,37.8 L295.0,37.8 L300.0,37.8 L305.0,37.8 L310.0,37.8 L315.0,37.8 L320.0,37.8 L325.0,37.8 L330.0,37.8 L335.0,37.8 L340.0,37.8 L345.0,37.8 L350.0,37.8" fill="none" class="stroke-a" stroke-width="2.4"/>
<circle cx="50.0" cy="114.0" r="4.5" class="fill-a"/>
<circle cx="55.0" cy="98.8" r="4.5" class="fill-a"/>
<circle cx="100.0" cy="46.0" r="4.5" class="fill-a"/>
<text x="62.0" y="108.8" text-anchor="start" class="fig-small">k=1: 0.2</text>
<text x="108.0" y="62.0" text-anchor="start" class="fig-small">k=10: 0.893</text>
<text x="50.0" y="176.0" text-anchor="start" class="fig-bold">Equivalent ridge penalty</text>
<line x1="50.0" y1="270.0" x2="350.0" y2="270.0" class="stroke-muted"/>
<line x1="50.0" y1="270.0" x2="50.0" y2="190.0" class="stroke-muted"/>
<line x1="50.0" y1="270.0" x2="50.0" y2="274.0" class="stroke-muted"/>
<text x="50.0" y="287.0" text-anchor="middle" class="fig-small">0</text>
<line x1="150.0" y1="270.0" x2="150.0" y2="274.0" class="stroke-muted"/>
<text x="150.0" y="287.0" text-anchor="middle" class="fig-small">20</text>
<line x1="250.0" y1="270.0" x2="250.0" y2="274.0" class="stroke-muted"/>
<text x="250.0" y="287.0" text-anchor="middle" class="fig-small">40</text>
<line x1="350.0" y1="270.0" x2="350.0" y2="274.0" class="stroke-muted"/>
<text x="350.0" y="287.0" text-anchor="middle" class="fig-small">60</text>
<line x1="46.0" y1="270.0" x2="50.0" y2="270.0" class="stroke-muted"/>
<text x="43.0" y="274.0" text-anchor="end" class="fig-small">0</text>
<line x1="46.0" y1="230.0" x2="50.0" y2="230.0" class="stroke-muted"/>
<text x="43.0" y="234.0" text-anchor="end" class="fig-small">5</text>
<line x1="46.0" y1="190.0" x2="50.0" y2="190.0" class="stroke-muted"/>
<text x="43.0" y="194.0" text-anchor="end" class="fig-small">10</text>
<text x="200.0" y="304.0" text-anchor="middle" class="fig-small">step k</text>
<path d="M55.0,206.0 L60.0,241.6 L65.0,253.2 L70.0,258.9 L75.0,262.2 L80.0,264.3 L85.0,265.8 L90.0,266.8 L95.0,267.5 L100.0,268.1 L105.0,268.5 L110.0,268.8 L115.0,269.1 L120.0,269.3 L125.0,269.4 L130.0,269.5 L135.0,269.6 L140.0,269.7 L145.0,269.8 L150.0,269.8 L155.0,269.9 L160.0,269.9 L165.0,269.9 L170.0,269.9 L175.0,269.9 L180.0,270.0 L185.0,270.0 L190.0,270.0 L195.0,270.0 L200.0,270.0 L205.0,270.0 L210.0,270.0 L215.0,270.0 L220.0,270.0 L225.0,270.0 L230.0,270.0 L235.0,270.0 L240.0,270.0 L245.0,270.0 L250.0,270.0 L255.0,270.0 L260.0,270.0 L265.0,270.0 L270.0,270.0 L275.0,270.0 L280.0,270.0 L285.0,270.0 L290.0,270.0 L295.0,270.0 L300.0,270.0 L305.0,270.0 L310.0,270.0 L315.0,270.0 L320.0,270.0 L325.0,270.0 L330.0,270.0 L335.0,270.0 L340.0,270.0 L345.0,270.0 L350.0,270.0" fill="none" class="stroke-b" stroke-width="2.4"/>
<line x1="50.0" y1="196.4" x2="50.0" y2="191.0" class="stroke-b" stroke-width="2.2" marker-end="url(#c074-arrow)"/>
<text x="59.0" y="194.2" text-anchor="start" class="fig-small">k=0: infinite</text>
<circle cx="55.0" cy="206.0" r="4.5" class="fill-b"/>
<text x="63.0" y="210.0" text-anchor="start" class="fig-small">k=1: 8</text>
<circle cx="100.0" cy="268.1" r="4.5" class="fill-b"/>
<text x="108.0" y="260.1" text-anchor="start" class="fig-small">k=10: 0.241</text>
<text x="500.0" y="22.0" text-anchor="middle" class="fig-bold">Reached at k = 25</text>
<line x1="420.0" y1="240.0" x2="580.0" y2="240.0" class="stroke-muted"/>
<line x1="420.0" y1="240.0" x2="420.0" y2="50.0" class="stroke-muted"/>
<line x1="416.0" y1="240.0" x2="420.0" y2="240.0" class="stroke-muted"/>
<text x="413.0" y="244.0" text-anchor="end" class="fig-small">0</text>
<line x1="416.0" y1="157.4" x2="420.0" y2="157.4" class="stroke-muted"/>
<text x="413.0" y="161.4" text-anchor="end" class="fig-small">0.5</text>
<line x1="416.0" y1="74.8" x2="420.0" y2="74.8" class="stroke-muted"/>
<text x="413.0" y="78.8" text-anchor="end" class="fig-small">1</text>
<rect x="430.7" y="203.3" width="32.0" height="36.7" class="fill-soft-c stroke-c"/>
<text x="446.7" y="197.3" text-anchor="middle" class="fig-bold">0.22</text>
<text x="446.7" y="257.0" text-anchor="middle" class="fig-small">c = 0.1</text>
<rect x="484.0" y="86.6" width="32.0" height="153.4" class="fill-soft-c stroke-c"/>
<text x="500.0" y="80.6" text-anchor="middle" class="fig-bold">0.93</text>
<text x="500.0" y="257.0" text-anchor="middle" class="fig-small">c = 1</text>
<rect x="537.3" y="74.8" width="32.0" height="165.2" class="fill-soft-c stroke-c"/>
<text x="553.3" y="68.8" text-anchor="middle" class="fig-bold">1.00</text>
<text x="553.3" y="257.0" text-anchor="middle" class="fig-small">c = 5</text>
</svg>
Figure 3. For curvature 2 and step 0.1, the fraction reached rises as the penalty falls: at k = 0 the penalty is infinite, at k = 1 it is 8, and at k = 10 it is 0.241. Right: after 25 steps a steep direction (c = 5) is fully released, while a flat one (c = 0.1) has moved only 22% of the way, so it still carries a large penalty.
```

The match is direction by direction, and the penalty differs between directions. After 25 steps at $\alpha = 0.1$, a direction with curvature 5 has penalty near $1.5 \times 10^{-7}$, curvature 1 has 0.077, and curvature 0.1 has 0.35. Early stopping is ridge with a small penalty on steep directions and a large one on flat directions.

> [!KEY]
> Gradient descent from zero covers steep directions first, so after $k$ steps each direction is shrunk as if by a ridge penalty, large on flat directions and small on steep ones. Fewer steps mean a larger penalty. That is why stopping where validation loss is lowest works like a penalty you never wrote.

## When the match holds

The formula needs $0 < \alpha c \le 1$. In that range, $1 - \alpha c$ lies between 0 and 1, so $b$ is a fraction that shrinks and $\lambda$ is positive. If $1 < \alpha c < 2$, then $1 - \alpha c$ is negative: each step overshoots $w^*$ and the error flips sign. At odd $k$, $b$ is negative and $\lambda$ comes out negative, a meaningless fee. If $\alpha c > 2$, the error grows at every step and gradient descent diverges, which is the 2/L limit from the earlier chapter. So the function you write raises a `ValueError` outside the valid range. At $\alpha c = 1$, a single step lands on $w^*$ and the penalty is 0.

## Stopping by watching the validation loss

Figure 1 shows the whole curve, but while training you only see it up to the present step. You find the lowest point by remembering the best step so far. A real validation loss is also jumpy, so a single bad check does not mean you have passed the lowest point.

**Patience** is how many checks in a row you wait without a new best before giving up. Take the losses $[5.0, 4.0, 4.5, 3.0, 3.5, 3.6, 3.7]$. With patience 1, you see 4.5 after the best 4.0, give up, and return step 1, missing the 3.0 at step 3. With patience 2 you survive 4.5, find 3.0, then see 3.5 and 3.6 with no improvement and return step 3.

> [!RECAP]
> Hold these points before you write code:
>
> - A direction is one independent axis of the loss, with curvature $c$. Gradient descent from zero gives $w_k = (1 - (1-\alpha c)^k)\, w^*$.
> - Ridge on that direction keeps the share $c/(c+\lambda)$ of $w^*$.
> - Matching the two gives $\lambda = c\,b/(1-b)$ with $b = (1-\alpha c)^k$. It is infinite at $k = 0$.
> - The match holds only for $0 < \alpha c \le 1$.
> - Patience waits through several checks without improvement before it stops.

## Build it step by step

The checks call the functions in this order:

1. **`gd_quadratic_1d(eigenvalue, step, w_star, iterations)`.** Run gradient descent from zero on $\tfrac{1}{2}c\,(w - w^*)^2$ with a loop, and return the final weight as a float. The usual mistake is to start at $w^*$ instead of 0.
2. **`early_stopping_equivalent_ridge(eigenvalue, step, iterations)`.** Return the penalty from the formula as a float. Return `float("inf")` when `iterations` is 0, and raise `ValueError` when `step * eigenvalue` is not in $(0, 1]$. The usual mistake is dividing by zero at 0 steps.
3. **`best_stopping_step(val_losses, patience)`.** Scan the list in order. Count the checks since the last strict improvement, stop when that count reaches `patience`, and return the index of the best loss seen. The usual mistake is to return the index of the global minimum, which a real run cannot know.

## Where this breaks

- A real network's loss is not a quadratic bowl. The picture holds near the starting point, where the loss is close to a bowl, and it gets looser as training moves far away.
- Adam divides each weight's gradient by its typical size, which evens out how fast different weights move. The neat order of steep first and flat last then holds less cleanly.
- You choose the stopping step with validation data, so the validation score at that step is a little too good. Keep a separate test set, as "Why the best score among many settings is too good" explains.

## The exercise

You implement `gd_quadratic_1d`, `early_stopping_equivalent_ridge`, and `best_stopping_step`.

The checks compare your loop with $(1 - (1-\alpha c)^k)\,w^*$, including a step that overshoots. They check the penalty at 0, 1, and 10 steps against the table, and that gradient descent after $k$ steps equals the ridge solution with your penalty. They confirm that the penalty falls as $k$ grows, that steep directions get smaller penalties, and that the function refuses a step above $1/c$. The last two checks run `best_stopping_step` on a V-shaped curve and on a jumpy one, where patience decides which dip you find.
