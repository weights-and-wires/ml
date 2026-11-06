---
title: "Taking safe steps: warmup, cosine decay, and gradient clipping"
slug: 096-safe-steps
part: "Part 6 — Transformers"
summary: Keep every training step safe with a learning rate that ramps up and glides down, and a clip that shortens any step that would be too long.
minutes: 40
gpu: false
objectives:
  - Explain why a transformer run wants a small learning rate at the start and at the end, and compute the warmup-then-cosine schedule by hand.
  - Explain how one unusually large gradient can undo a long run, and why we limit its length instead of its entries.
  - Compute the global norm of a list of gradient arrays and scale them to a limit, keeping their direction.
  - Read a log of gradient norms and tell a healthy run from a spiky one.
prereqs:
  - 060-lr-schedules
  - 079-truncation-and-clipping
  - 095-next-token-loss
exercise: 096-safe-steps
---

# Taking safe steps: warmup, cosine decay, and gradient clipping

Your model trains fine for 2,000 steps. The loss falls smoothly. Then one odd batch arrives, the loss jumps to 40, and it never comes back. A week of compute has become a model you cannot use. Where did the damage come from?

Training is walking downhill in fog. At each step the gradient tells you the slope under your feet, and you move a distance set by the learning rate. The slope you feel at one spot is only a guess about the ground a few steps away. When the ground is smooth the guess is good, and when one batch reports a slope 100 times steeper than usual, a step along it can land you somewhere far away and much worse. Two things can go wrong: the learning rate can be wrong for the stage of training, and a single gradient can be far longer than any other.

This chapter gives you two controls. A schedule changes the learning rate as training goes on. A clip shortens any gradient that is too long, without turning it.

## What you already know

Five ideas carry this chapter:

- A step is the learning rate times a direction taken from the gradient, so a bigger rate means a longer step (see "Walking downhill: slope, gradient, and one step").
- A learning-rate schedule changes the rate over the run, and a linear warmup followed by a cosine decay is a common choice (see "Learning-rate schedules: warmup and cosine decay").
- A mini-batch gradient is right on average but noisy at each step.
- The norm of a vector is its length, the square root of the sum of its squared entries.
- Clipping a gradient by its norm limits how far one step can go (see "Taming long sequences: truncation and clipping").

## Start gently, finish gently

Figure 1 shows the schedule this chapter uses. The rate climbs in a straight line, then glides down to a floor.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="The warmup then cosine learning rate schedule with four marked points, against a constant rate and a linear decay">
<line x1="70" y1="260" x2="70" y2="265" class="stroke-muted"/>
<text x="70" y="280" text-anchor="middle" class="fig-small">0</text>
<line x1="162" y1="260" x2="162" y2="265" class="stroke-muted"/>
<text x="162" y="280" text-anchor="middle" class="fig-small">10</text>
<line x1="254" y1="260" x2="254" y2="265" class="stroke-muted"/>
<text x="254" y="280" text-anchor="middle" class="fig-small">20</text>
<line x1="346" y1="260" x2="346" y2="265" class="stroke-muted"/>
<text x="346" y="280" text-anchor="middle" class="fig-small">30</text>
<line x1="438" y1="260" x2="438" y2="265" class="stroke-muted"/>
<text x="438" y="280" text-anchor="middle" class="fig-small">40</text>
<line x1="530" y1="260" x2="530" y2="265" class="stroke-muted"/>
<text x="530" y="280" text-anchor="middle" class="fig-small">50</text>
<line x1="65" y1="260" x2="70" y2="260" class="stroke-muted"/>
<text x="61" y="264" text-anchor="end" class="fig-small">0</text>
<line x1="65" y1="207.7" x2="70" y2="207.7" class="stroke-muted"/>
<text x="61" y="211.7" text-anchor="end" class="fig-small">0.5</text>
<line x1="70" y1="207.7" x2="530" y2="207.7" class="stroke-grid"/>
<line x1="65" y1="155.5" x2="70" y2="155.5" class="stroke-muted"/>
<text x="61" y="159.5" text-anchor="end" class="fig-small">1</text>
<line x1="70" y1="155.5" x2="530" y2="155.5" class="stroke-grid"/>
<line x1="65" y1="103.2" x2="70" y2="103.2" class="stroke-muted"/>
<text x="61" y="107.2" text-anchor="end" class="fig-small">1.5</text>
<line x1="70" y1="103.2" x2="530" y2="103.2" class="stroke-grid"/>
<line x1="65" y1="50.9" x2="70" y2="50.9" class="stroke-muted"/>
<text x="61" y="54.9" text-anchor="end" class="fig-small">2</text>
<line x1="70" y1="50.9" x2="530" y2="50.9" class="stroke-grid"/>
<line x1="70" y1="260" x2="530" y2="260" class="stroke-muted"/>
<line x1="70" y1="260" x2="70" y2="24" class="stroke-muted"/>
<line x1="162" y1="50.9" x2="530" y2="50.9" class="stroke-muted dashed"/>
<line x1="162" y1="50.9" x2="530" y2="249.5" class="stroke-d dashed"/>
<path d="M70,260 L72.3,254.8 L74.6,249.5 L76.9,244.3 L79.2,239.1 L81.5,233.9 L83.8,228.6 L86.1,223.4 L88.4,218.2 L90.7,213 L93,207.7 L95.3,202.5 L97.6,197.3 L99.9,192 L102.2,186.8 L104.5,181.6 L106.8,176.4 L109.1,171.1 L111.4,165.9 L113.7,160.7 L116,155.5 L118.3,150.2 L120.6,145 L122.9,139.8 L125.2,134.5 L127.5,129.3 L129.8,124.1 L132.1,118.9 L134.4,113.6 L136.7,108.4 L139,103.2 L141.3,98 L143.6,92.7 L145.9,87.5 L148.2,82.3 L150.5,77 L152.8,71.8 L155.1,66.6 L157.4,61.4 L159.7,56.1 L162,50.9 L164.3,50.9 L166.6,51 L168.9,51.1 L171.2,51.2 L173.5,51.4 L175.8,51.6 L178.1,51.8 L180.4,52.1 L182.7,52.5 L185,52.8 L187.3,53.2 L189.6,53.7 L191.9,54.1 L194.2,54.6 L196.5,55.2 L198.8,55.8 L201.1,56.4 L203.4,57 L205.7,57.7 L208,58.5 L210.3,59.2 L212.6,60 L214.9,60.9 L217.2,61.7 L219.5,62.6 L221.8,63.6 L224.1,64.5 L226.4,65.5 L228.7,66.6 L231,67.6 L233.3,68.7 L235.6,69.9 L237.9,71 L240.2,72.2 L242.5,73.5 L244.8,74.7 L247.1,76 L249.4,77.3 L251.7,78.6 L254,80 L256.3,81.4 L258.6,82.8 L260.9,84.3 L263.2,85.7 L265.5,87.2 L267.8,88.7 L270.1,90.3 L272.4,91.8 L274.7,93.4 L277,95 L279.3,96.7 L281.6,98.3 L283.9,100 L286.2,101.7 L288.5,103.4 L290.8,105.1 L293.1,106.9 L295.4,108.6 L297.7,110.4 L300,112.2 L302.3,114 L304.6,115.9 L306.9,117.7 L309.2,119.5 L311.5,121.4 L313.8,123.3 L316.1,125.2 L318.4,127 L320.7,128.9 L323,130.9 L325.3,132.8 L327.6,134.7 L329.9,136.6 L332.2,138.6 L334.5,140.5 L336.8,142.4 L339.1,144.4 L341.4,146.3 L343.7,148.3 L346,150.2 L348.3,152.2 L350.6,154.1 L352.9,156.1 L355.2,158 L357.5,160 L359.8,161.9 L362.1,163.8 L364.4,165.8 L366.7,167.7 L369,169.6 L371.3,171.5 L373.6,173.4 L375.9,175.3 L378.2,177.2 L380.5,179.1 L382.8,180.9 L385.1,182.8 L387.4,184.6 L389.7,186.4 L392,188.2 L394.3,190 L396.6,191.8 L398.9,193.6 L401.2,195.3 L403.5,197 L405.8,198.8 L408.1,200.4 L410.4,202.1 L412.7,203.8 L415,205.4 L417.3,207 L419.6,208.6 L421.9,210.2 L424.2,211.7 L426.5,213.2 L428.8,214.7 L431.1,216.2 L433.4,217.6 L435.7,219.1 L438,220.5 L440.3,221.8 L442.6,223.2 L444.9,224.5 L447.2,225.7 L449.5,227 L451.8,228.2 L454.1,229.4 L456.4,230.6 L458.7,231.7 L461,232.8 L463.3,233.9 L465.6,234.9 L467.9,235.9 L470.2,236.9 L472.5,237.8 L474.8,238.7 L477.1,239.6 L479.4,240.4 L481.7,241.2 L484,242 L486.3,242.7 L488.6,243.4 L490.9,244.1 L493.2,244.7 L495.5,245.3 L497.8,245.8 L500.1,246.3 L502.4,246.8 L504.7,247.2 L507,247.6 L509.3,248 L511.6,248.3 L513.9,248.6 L516.2,248.9 L518.5,249.1 L520.8,249.2 L523.1,249.4 L525.4,249.5 L527.7,249.5 L530,249.5" fill="none" class="stroke-b" stroke-width="3"/>
<circle cx="116" cy="155.5" r="5" class="fill-a"/>
<circle cx="162" cy="50.9" r="5" class="fill-a"/>
<circle cx="346" cy="150.2" r="5" class="fill-a"/>
<circle cx="530" cy="249.5" r="5" class="fill-a"/>
<text x="108" y="145.5" text-anchor="end" class="fig-small">(5, 1.0)</text>
<text x="162" y="38.9" text-anchor="middle" class="fig-small">(10, 2.0)</text>
<text x="334" y="174.2" text-anchor="end" class="fig-small">(30, 1.05)</text>
<text x="540" y="253.5" text-anchor="start" class="fig-small">(50, 0.1)</text>
<text x="327.6" y="42.9" text-anchor="start" class="fig-small">constant rate</text>
<text x="392.8" y="158.1" text-anchor="start" class="fig-small">linear decay</text>
<text x="97.6" y="68.9" text-anchor="start" class="fig-small">warmup</text>
<text x="300" y="304" text-anchor="middle">step</text>
<text transform="rotate(-90 22 145)" x="22" y="145" text-anchor="middle">learning rate</text>
</svg>
Figure 1. The schedule from warmup_cosine_lr(step, 10, 50, 2, 0.1), drawn from computed values. The rate climbs in a straight line to 2 over 10 steps, then glides to 0.1 along half a cosine. The two pieces meet at step 10 with no jump, and the curve is flat at the peak and at the floor. The dashed lines are a constant rate and a straight-line decay, which agrees with the cosine at step 30.
```

Why start small? At step 0 the weights are random, so the first gradients come from a model that knows nothing and they point in rough directions. Adam, the optimiser you will wire in next, also has no history yet: its running averages have seen one or two batches. A big step at that point would trust a guess that has no evidence behind it. It is like a driver pulling away on ice: ease in, then speed up.

Why not keep the rate small forever? Because the middle of training needs long steps to cover ground, and a small rate would make it crawl. Why decay at the end? Look at Figure 2.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="A narrow valley drawn as contour ellipses with two noisy gradient descent paths, one with a constant learning rate and one with a decaying rate">
<defs><marker id="c106-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<path d="M360,112 L359.9,110.7 L359.4,109.4 L358.7,108.1 L357.7,106.8 L356.4,105.6 L354.8,104.4 L353,103.2 L350.9,102.1 L348.5,101 L346,99.9 L343.2,99 L340.1,98.1 L336.9,97.2 L333.6,96.5 L330,95.8 L326.3,95.1 L322.5,94.6 L318.5,94.2 L314.5,93.8 L310.4,93.5 L306.3,93.4 L302.1,93.3 L297.9,93.3 L293.7,93.4 L289.6,93.5 L285.5,93.8 L281.5,94.2 L277.5,94.6 L273.7,95.1 L270,95.8 L266.4,96.5 L263.1,97.2 L259.9,98.1 L256.8,99 L254,99.9 L251.5,101 L249.1,102.1 L247,103.2 L245.2,104.4 L243.6,105.6 L242.3,106.8 L241.3,108.1 L240.6,109.4 L240.1,110.7 L240,112 L240.1,113.3 L240.6,114.6 L241.3,115.9 L242.3,117.2 L243.6,118.4 L245.2,119.6 L247,120.8 L249.1,121.9 L251.5,123 L254,124.1 L256.8,125 L259.9,125.9 L263.1,126.8 L266.4,127.5 L270,128.2 L273.7,128.9 L277.5,129.4 L281.5,129.8 L285.5,130.2 L289.6,130.5 L293.7,130.6 L297.9,130.7 L302.1,130.7 L306.3,130.6 L310.4,130.5 L314.5,130.2 L318.5,129.8 L322.5,129.4 L326.3,128.9 L330,128.2 L333.6,127.5 L336.9,126.8 L340.1,125.9 L343.2,125 L346,124.1 L348.5,123 L350.9,121.9 L353,120.8 L354.8,119.6 L356.4,118.4 L357.7,117.2 L358.7,115.9 L359.4,114.6 L359.9,113.3 L360,112" fill="none" class="stroke-grid"/>
<path d="M420,112 L419.7,109.4 L418.8,106.8 L417.4,104.2 L415.4,101.7 L412.8,99.2 L409.6,96.7 L406,94.4 L401.8,92.1 L397.1,90 L391.9,87.9 L386.3,86 L380.3,84.1 L373.9,82.4 L367.1,80.9 L360,79.5 L352.6,78.3 L345,77.2 L337.1,76.3 L329,75.6 L320.8,75.1 L312.5,74.7 L304.2,74.5 L295.8,74.5 L287.5,74.7 L279.2,75.1 L271,75.6 L262.9,76.3 L255,77.2 L247.4,78.3 L240,79.5 L232.9,80.9 L226.1,82.4 L219.7,84.1 L213.7,86 L208.1,87.9 L202.9,90 L198.2,92.1 L194,94.4 L190.4,96.7 L187.2,99.2 L184.6,101.7 L182.6,104.2 L181.2,106.8 L180.3,109.4 L180,112 L180.3,114.6 L181.2,117.2 L182.6,119.8 L184.6,122.3 L187.2,124.8 L190.4,127.3 L194,129.6 L198.2,131.9 L202.9,134 L208.1,136.1 L213.7,138 L219.7,139.9 L226.1,141.6 L232.9,143.1 L240,144.5 L247.4,145.7 L255,146.8 L262.9,147.7 L271,148.4 L279.2,148.9 L287.5,149.3 L295.8,149.5 L304.2,149.5 L312.5,149.3 L320.8,148.9 L329,148.4 L337.1,147.7 L345,146.8 L352.6,145.7 L360,144.5 L367.1,143.1 L373.9,141.6 L380.3,139.9 L386.3,138 L391.9,136.1 L397.1,134 L401.8,131.9 L406,129.6 L409.6,127.3 L412.8,124.8 L415.4,122.3 L417.4,119.8 L418.8,117.2 L419.7,114.6 L420,112" fill="none" class="stroke-grid"/>
<path d="M489.7,112 L489.3,107.9 L487.9,103.7 L485.6,99.7 L482.4,95.7 L478.3,91.7 L473.3,87.9 L467.5,84.2 L460.9,80.6 L453.5,77.1 L445.3,73.9 L436.5,70.8 L427,67.9 L416.8,65.3 L406.1,62.8 L394.9,60.7 L383.2,58.7 L371.1,57 L358.6,55.6 L345.9,54.5 L332.9,53.6 L319.8,53 L306.6,52.7 L293.4,52.7 L280.2,53 L267.1,53.6 L254.1,54.5 L241.4,55.6 L228.9,57 L216.8,58.7 L205.1,60.7 L193.9,62.8 L183.2,65.3 L173,67.9 L163.5,70.8 L154.7,73.9 L146.5,77.1 L139.1,80.6 L132.5,84.2 L126.7,87.9 L121.7,91.7 L117.6,95.7 L114.4,99.7 L112.1,103.7 L110.7,107.9 L110.3,112 L110.7,116.1 L112.1,120.3 L114.4,124.3 L117.6,128.3 L121.7,132.3 L126.7,136.1 L132.5,139.8 L139.1,143.4 L146.5,146.9 L154.7,150.1 L163.5,153.2 L173,156.1 L183.2,158.7 L193.9,161.2 L205.1,163.3 L216.8,165.3 L228.9,167 L241.4,168.4 L254.1,169.5 L267.1,170.4 L280.2,171 L293.4,171.3 L306.6,171.3 L319.8,171 L332.9,170.4 L345.9,169.5 L358.6,168.4 L371.1,167 L383.2,165.3 L394.9,163.3 L406.1,161.2 L416.8,158.7 L427,156.1 L436.5,153.2 L445.3,150.1 L453.5,146.9 L460.9,143.4 L467.5,139.8 L473.3,136.1 L478.3,132.3 L482.4,128.3 L485.6,124.3 L487.9,120.3 L489.3,116.1 L489.7,112" fill="none" class="stroke-grid"/>
<path d="M568.3,112 L567.7,106.2 L565.7,100.3 L562.5,94.6 L557.9,88.9 L552.1,83.3 L545.1,77.9 L536.9,72.6 L527.6,67.6 L517.1,62.7 L505.6,58.1 L493,53.8 L479.5,49.7 L465.2,45.9 L450,42.5 L434.2,39.4 L417.6,36.6 L400.5,34.3 L382.9,32.3 L364.9,30.6 L346.6,29.4 L328,28.6 L309.4,28.2 L290.6,28.2 L272,28.6 L253.4,29.4 L235.1,30.6 L217.1,32.3 L199.5,34.3 L182.4,36.6 L165.8,39.4 L150,42.5 L134.8,45.9 L120.5,49.7 L107,53.8 L94.4,58.1 L82.9,62.7 L72.4,67.6 L63.1,72.6 L54.9,77.9 L47.9,83.3 L42.1,88.9 L37.5,94.6 L34.3,100.3 L32.3,106.2 L31.7,112 L32.3,117.8 L34.3,123.7 L37.5,129.4 L42.1,135.1 L47.9,140.7 L54.9,146.1 L63.1,151.4 L72.4,156.4 L82.9,161.3 L94.4,165.9 L107,170.2 L120.5,174.3 L134.8,178.1 L150,181.5 L165.8,184.6 L182.4,187.4 L199.5,189.7 L217.1,191.7 L235.1,193.4 L253.4,194.6 L272,195.4 L290.6,195.8 L309.4,195.8 L328,195.4 L346.6,194.6 L364.9,193.4 L382.9,191.7 L400.5,189.7 L417.6,187.4 L434.2,184.6 L450,181.5 L465.2,178.1 L479.5,174.3 L493,170.2 L505.6,165.9 L517.1,161.3 L527.6,156.4 L536.9,151.4 L545.1,146.1 L552.1,140.7 L557.9,135.1 L562.5,129.4 L565.7,123.7 L567.7,117.8 L568.3,112" fill="none" class="stroke-grid"/>
<circle cx="300" cy="112" r="4" class="fill-ink"/>
<text x="300" y="160" text-anchor="middle" class="fig-small">minimum (centre)</text>
<path d="M60,22 L74.3,161.4 L96.6,70.2 L121.7,142.2 L153.3,87.4 L174.6,155.4 L187,76.4 L201.1,134.1 L218.3,92.3 L224.4,125.2 L227,100.5 L234.9,132.2 L238.8,92.9 L246.6,130.5 L241,95.9 L249,131.7 L259.8,94.9 L259,129.3 L263,103.8 L283.8,125.8 L291.3,89.1 L290.6,134.6 L294.3,86.8 L294.8,130.7 L287,103.3 L287.3,126.9 L289.9,93.8 L287.5,130.1 L290.2,92.4 L289.9,108.4 L286.9,118.4 L298.1,107.6 L304,121 L315.7,98.4 L309.7,130.9 L321.5,93.9 L317.2,121.2 L305.8,96.1 L303.1,116.3 L294.4,108.6 L297.2,101.8 L287.5,125.3 L284.4,110.3 L284.1,108.5 L290.6,108.7 L283.5,103.7 L288.8,115.9 L288.7,113.3 L297.4,98.2 L297.7,131.5 L293.5,98.8 L296.1,124.2 L297.9,108.8 L302.9,115.4 L303.2,113.4 L301.5,129.9 L292.5,109.5 L305.4,111.3 L308.4,116.5 L321,117.3 L312.4,98.3" fill="none" class="stroke-bad" stroke-width="1.2"/>
<circle cx="74.3" cy="161.4" r="2.5" class="fill-bad"/>
<circle cx="96.6" cy="70.2" r="2.5" class="fill-bad"/>
<circle cx="121.7" cy="142.2" r="2.5" class="fill-bad"/>
<circle cx="153.3" cy="87.4" r="2.5" class="fill-bad"/>
<circle cx="174.6" cy="155.4" r="2.5" class="fill-bad"/>
<circle cx="187" cy="76.4" r="2.5" class="fill-bad"/>
<circle cx="201.1" cy="134.1" r="2.5" class="fill-bad"/>
<circle cx="218.3" cy="92.3" r="2.5" class="fill-bad"/>
<circle cx="224.4" cy="125.2" r="2.5" class="fill-bad"/>
<circle cx="227" cy="100.5" r="2.5" class="fill-bad"/>
<circle cx="234.9" cy="132.2" r="2.5" class="fill-bad"/>
<circle cx="238.8" cy="92.9" r="2.5" class="fill-bad"/>
<circle cx="246.6" cy="130.5" r="2.5" class="fill-bad"/>
<circle cx="241" cy="95.9" r="2.5" class="fill-bad"/>
<circle cx="249" cy="131.7" r="2.5" class="fill-bad"/>
<circle cx="259.8" cy="94.9" r="2.5" class="fill-bad"/>
<circle cx="259" cy="129.3" r="2.5" class="fill-bad"/>
<circle cx="263" cy="103.8" r="2.5" class="fill-bad"/>
<circle cx="283.8" cy="125.8" r="2.5" class="fill-bad"/>
<circle cx="291.3" cy="89.1" r="2.5" class="fill-bad"/>
<circle cx="290.6" cy="134.6" r="2.5" class="fill-bad"/>
<circle cx="294.3" cy="86.8" r="2.5" class="fill-bad"/>
<circle cx="294.8" cy="130.7" r="2.5" class="fill-bad"/>
<circle cx="287" cy="103.3" r="2.5" class="fill-bad"/>
<circle cx="287.3" cy="126.9" r="2.5" class="fill-bad"/>
<circle cx="289.9" cy="93.8" r="2.5" class="fill-bad"/>
<circle cx="287.5" cy="130.1" r="2.5" class="fill-bad"/>
<circle cx="290.2" cy="92.4" r="2.5" class="fill-bad"/>
<circle cx="289.9" cy="108.4" r="2.5" class="fill-bad"/>
<circle cx="286.9" cy="118.4" r="2.5" class="fill-bad"/>
<circle cx="298.1" cy="107.6" r="2.5" class="fill-bad"/>
<circle cx="304" cy="121" r="2.5" class="fill-bad"/>
<circle cx="315.7" cy="98.4" r="2.5" class="fill-bad"/>
<circle cx="309.7" cy="130.9" r="2.5" class="fill-bad"/>
<circle cx="321.5" cy="93.9" r="2.5" class="fill-bad"/>
<circle cx="317.2" cy="121.2" r="2.5" class="fill-bad"/>
<circle cx="305.8" cy="96.1" r="2.5" class="fill-bad"/>
<circle cx="303.1" cy="116.3" r="2.5" class="fill-bad"/>
<circle cx="294.4" cy="108.6" r="2.5" class="fill-bad"/>
<circle cx="297.2" cy="101.8" r="2.5" class="fill-bad"/>
<circle cx="287.5" cy="125.3" r="2.5" class="fill-bad"/>
<circle cx="284.4" cy="110.3" r="2.5" class="fill-bad"/>
<circle cx="284.1" cy="108.5" r="2.5" class="fill-bad"/>
<circle cx="290.6" cy="108.7" r="2.5" class="fill-bad"/>
<circle cx="283.5" cy="103.7" r="2.5" class="fill-bad"/>
<circle cx="288.8" cy="115.9" r="2.5" class="fill-bad"/>
<circle cx="288.7" cy="113.3" r="2.5" class="fill-bad"/>
<circle cx="297.4" cy="98.2" r="2.5" class="fill-bad"/>
<circle cx="297.7" cy="131.5" r="2.5" class="fill-bad"/>
<circle cx="293.5" cy="98.8" r="2.5" class="fill-bad"/>
<circle cx="296.1" cy="124.2" r="2.5" class="fill-bad"/>
<circle cx="297.9" cy="108.8" r="2.5" class="fill-bad"/>
<circle cx="302.9" cy="115.4" r="2.5" class="fill-bad"/>
<circle cx="303.2" cy="113.4" r="2.5" class="fill-bad"/>
<circle cx="301.5" cy="129.9" r="2.5" class="fill-bad"/>
<circle cx="292.5" cy="109.5" r="2.5" class="fill-bad"/>
<circle cx="305.4" cy="111.3" r="2.5" class="fill-bad"/>
<circle cx="308.4" cy="116.5" r="2.5" class="fill-bad"/>
<circle cx="321" cy="117.3" r="2.5" class="fill-bad"/>
<circle cx="312.4" cy="98.3" r="2.5" class="fill-bad"/>
<path d="M60,22 L74.3,161.4 L96.6,70.3 L121.6,141.9 L153,87.9 L174.1,154.3 L186.4,78.4 L200.3,131.3 L217,95.6 L223,121.6 L225.6,104.2 L233.2,127.8 L237,98.6 L244.3,123.7 L239.5,103.5 L246.6,123.2 L256.2,104.3 L255.8,119.4 L259.4,112.8 L276.6,117.8 L283.1,100 L283.2,120.1 L286.6,103.5 L287.5,114 L282.6,115.4 L283.2,116.9 L285.2,106.8 L284,115.3 L285.9,108.3 L286,101.3 L284.5,113.8 L290.6,112.4 L294.1,114.9 L300.3,109.1 L298.3,115.4 L304,111.2 L303,109.9 L299.2,107.9 L298.4,107.7 L295.6,110.3 L296.5,107.1 L293.6,111.4 L292.5,114.1 L292.2,111.9 L293.5,110.5 L291.8,108.6 L292.8,109.4 L292.7,111.1 L294.2,109.1 L294.3,111.4 L293.7,111.8 L294,112.2 L294.3,112.9 L295,112.9 L295.1,113.1 L295,115 L294.2,115.6 L295.4,114.7 L295.8,114.6 L297.1,115 L296.5,113.6" fill="none" class="stroke-c" stroke-width="2"/>
<circle cx="74.3" cy="161.4" r="2.5" class="fill-c"/>
<circle cx="96.6" cy="70.3" r="2.5" class="fill-c"/>
<circle cx="121.6" cy="141.9" r="2.5" class="fill-c"/>
<circle cx="153" cy="87.9" r="2.5" class="fill-c"/>
<circle cx="174.1" cy="154.3" r="2.5" class="fill-c"/>
<circle cx="186.4" cy="78.4" r="2.5" class="fill-c"/>
<circle cx="200.3" cy="131.3" r="2.5" class="fill-c"/>
<circle cx="217" cy="95.6" r="2.5" class="fill-c"/>
<circle cx="223" cy="121.6" r="2.5" class="fill-c"/>
<circle cx="225.6" cy="104.2" r="2.5" class="fill-c"/>
<circle cx="233.2" cy="127.8" r="2.5" class="fill-c"/>
<circle cx="237" cy="98.6" r="2.5" class="fill-c"/>
<circle cx="244.3" cy="123.7" r="2.5" class="fill-c"/>
<circle cx="239.5" cy="103.5" r="2.5" class="fill-c"/>
<circle cx="246.6" cy="123.2" r="2.5" class="fill-c"/>
<circle cx="256.2" cy="104.3" r="2.5" class="fill-c"/>
<circle cx="255.8" cy="119.4" r="2.5" class="fill-c"/>
<circle cx="259.4" cy="112.8" r="2.5" class="fill-c"/>
<circle cx="276.6" cy="117.8" r="2.5" class="fill-c"/>
<circle cx="283.1" cy="100" r="2.5" class="fill-c"/>
<circle cx="283.2" cy="120.1" r="2.5" class="fill-c"/>
<circle cx="286.6" cy="103.5" r="2.5" class="fill-c"/>
<circle cx="287.5" cy="114" r="2.5" class="fill-c"/>
<circle cx="282.6" cy="115.4" r="2.5" class="fill-c"/>
<circle cx="283.2" cy="116.9" r="2.5" class="fill-c"/>
<circle cx="285.2" cy="106.8" r="2.5" class="fill-c"/>
<circle cx="284" cy="115.3" r="2.5" class="fill-c"/>
<circle cx="285.9" cy="108.3" r="2.5" class="fill-c"/>
<circle cx="286" cy="101.3" r="2.5" class="fill-c"/>
<circle cx="284.5" cy="113.8" r="2.5" class="fill-c"/>
<circle cx="290.6" cy="112.4" r="2.5" class="fill-c"/>
<circle cx="294.1" cy="114.9" r="2.5" class="fill-c"/>
<circle cx="300.3" cy="109.1" r="2.5" class="fill-c"/>
<circle cx="298.3" cy="115.4" r="2.5" class="fill-c"/>
<circle cx="304" cy="111.2" r="2.5" class="fill-c"/>
<circle cx="303" cy="109.9" r="2.5" class="fill-c"/>
<circle cx="299.2" cy="107.9" r="2.5" class="fill-c"/>
<circle cx="298.4" cy="107.7" r="2.5" class="fill-c"/>
<circle cx="295.6" cy="110.3" r="2.5" class="fill-c"/>
<circle cx="296.5" cy="107.1" r="2.5" class="fill-c"/>
<circle cx="293.6" cy="111.4" r="2.5" class="fill-c"/>
<circle cx="292.5" cy="114.1" r="2.5" class="fill-c"/>
<circle cx="292.2" cy="111.9" r="2.5" class="fill-c"/>
<circle cx="293.5" cy="110.5" r="2.5" class="fill-c"/>
<circle cx="291.8" cy="108.6" r="2.5" class="fill-c"/>
<circle cx="292.8" cy="109.4" r="2.5" class="fill-c"/>
<circle cx="292.7" cy="111.1" r="2.5" class="fill-c"/>
<circle cx="294.2" cy="109.1" r="2.5" class="fill-c"/>
<circle cx="294.3" cy="111.4" r="2.5" class="fill-c"/>
<circle cx="293.7" cy="111.8" r="2.5" class="fill-c"/>
<circle cx="294" cy="112.2" r="2.5" class="fill-c"/>
<circle cx="294.3" cy="112.9" r="2.5" class="fill-c"/>
<circle cx="295" cy="112.9" r="2.5" class="fill-c"/>
<circle cx="295.1" cy="113.1" r="2.5" class="fill-c"/>
<circle cx="295" cy="115" r="2.5" class="fill-c"/>
<circle cx="294.2" cy="115.6" r="2.5" class="fill-c"/>
<circle cx="295.4" cy="114.7" r="2.5" class="fill-c"/>
<circle cx="295.8" cy="114.6" r="2.5" class="fill-c"/>
<circle cx="297.1" cy="115" r="2.5" class="fill-c"/>
<circle cx="296.5" cy="113.6" r="2.5" class="fill-c"/>
<circle cx="60" cy="22" r="5" class="fill-ink"/>
<text x="68" y="12" text-anchor="start" class="fig-small">start</text>
<line x1="40" y1="232" x2="70" y2="232" class="stroke-bad" stroke-width="1.5"/>
<text x="78" y="236" text-anchor="start" class="fig-small">constant rate 0.11</text>
<line x1="300" y1="232" x2="330" y2="232" class="stroke-c" stroke-width="2.5"/>
<text x="338" y="236" text-anchor="start" class="fig-small">rate decays from 0.11 to 0.01</text>
</svg>
Figure 2. A narrow valley with two descents on the same noisy gradients, one dot per step. The constant rate keeps bouncing across the valley floor. The decaying rate takes the same bold early steps, then shrinks them and settles. Averaged over the last 15 steps and over 300 random seeds, the loss is 0.20 with the constant rate and 0.017 with the decaying one. The vertical axis is stretched to show the valley.
```

Each batch gives a slightly different gradient, so every step is the true downhill direction plus noise. A large step amplifies that noise, and the path bounces from wall to wall of the valley, never settling. A small step amplifies it less, so the path comes to rest near the bottom. The schedule takes bold steps while the ground is far away and small steps when you are close.

Why a cosine, and not a straight line or a sudden drop? A cosine is smooth and flat at both ends: it holds near the peak for a while, falls fastest in the middle, and eases onto the floor. A straight line works too, and the dashed line in Figure 1 shows how close it is. A sudden drop jolts the run. The floor stays above zero, often at about a tenth of the peak, so a little learning continues at the end.

Take a warmup of 10 steps, a total of 50, a peak of 2, and a floor of 0.1. At step 5 the rate is $2 \times 5 / 10 = 1.0$. At step 10 warmup is done and the rate is the peak, 2. At step 30 you are halfway through the glide, because $(30 - 10)/(50 - 10) = 0.5$. The cosine of $\pi \times 0.5$ is 0, so the rate is $0.1 + 0.5 \times (2 - 0.1) \times (1 + 0) = 1.05$. At step 50, progress is 1, the cosine of $\pi$ is $-1$, and the rate lands on the floor, 0.1. Those are the four dots in Figure 1.

As a formula, with $s$ the step and progress $= (s - \text{warmup})/(\text{total} - \text{warmup})$:

$$
\text{lr}(s) =
\begin{cases}
\text{peak}\cdot \dfrac{s}{\text{warmup}} & s < \text{warmup} \\[1.5ex]
\text{floor} + \tfrac{1}{2}(\text{peak} - \text{floor})\big(1 + \cos(\pi \cdot \text{progress})\big) & s \ge \text{warmup}
\end{cases}
$$

Read it aloud: before the end of warmup, take a share of the peak equal to the fraction of warmup done. After it, start at the peak, where the cosine is 1, and slide to the floor, where the cosine is $-1$. In numpy:

```python
if step < warmup:
    return peak * step / warmup
progress = (step - warmup) / (total - warmup)
return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))
```

The function only maps a number to a rate. When `train_step` calls it in the next chapter, it passes the count of the update about to be applied, starting at 1, so the first update uses a rate of peak divided by warmup and never 0.

## Cap the length of a step

The schedule cannot protect you from one bad batch, because it does not look at the gradient. A rare batch, such as an odd document, can produce a gradient far longer than usual. With plain gradient descent, that gradient times the learning rate is a step that is equally too long. Adam divides by a running size, so one step stays bounded, but the spike still enters its running averages and keeps pushing for many steps afterwards. Either way, you want a limit on what one batch can say.

Think of a hiker who is told never to take a step longer than one metre, however steep the slope looks. The slope can say "go far". The limit says "far is one metre". Applied to a gradient, the limit goes on its length, not on each of its numbers, because a gradient has a direction as well as a length, and the direction is the useful part. Figure 3 shows the difference.

```figure
<svg viewBox="0 0 640 290" role="img" aria-label="Gradient arrow of length 5 at 53 degrees, the same arrow shrunk by a global scale factor to length 2.5, and an entry-wise clip that bends it to 45 degrees">
<defs><marker id="c106-arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<defs><marker id="c106-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker></defs>
<defs><marker id="c106-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<defs><marker id="c106-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-bad"/></marker></defs>
<circle cx="200" cy="250" r="4" class="fill-ink"/>
<path d="M300,250 L300,247.4 L299.9,244.8 L299.7,242.2 L299.5,239.5 L299.1,236.9 L298.8,234.4 L298.3,231.8 L297.8,229.2 L297.2,226.7 L296.6,224.1 L295.9,221.6 L295.1,219.1 L294.3,216.6 L293.4,214.2 L292.4,211.7 L291.4,209.3 L290.3,206.9 L289.1,204.6 L287.9,202.3 L286.6,200 L285.3,197.8 L283.9,195.5 L282.4,193.4 L280.9,191.2 L279.3,189.1 L277.7,187.1 L276,185.1 L274.3,183.1 L272.5,181.2 L270.7,179.3 L268.8,177.5 L266.9,175.7 L264.9,174 L262.9,172.3 L260.9,170.7 L258.8,169.1 L256.6,167.6 L254.5,166.1 L252.2,164.7 L250,163.4 L247.7,162.1 L245.4,160.9 L243.1,159.7 L240.7,158.6 L238.3,157.6 L235.8,156.6 L233.4,155.7 L230.9,154.9 L228.4,154.1 L225.9,153.4 L223.3,152.8 L220.8,152.2 L218.2,151.7 L215.6,151.2 L213.1,150.9 L210.5,150.5 L207.8,150.3 L205.2,150.1 L202.6,150 L200,150" fill="none" class="stroke-muted dashed"/>
<text x="306" y="226.1" text-anchor="start" class="fig-small">limit 2.5</text>
<line x1="200" y1="250" x2="317" y2="94" class="stroke-a" marker-end="url(#c106-a)" stroke-width="3"/>
<text x="328" y="94" text-anchor="start" class="fig-small">gradient (3, 4), length 5</text>
<line x1="200" y1="250" x2="297" y2="153" class="stroke-bad" marker-end="url(#c106-bad)" stroke-width="3"/>
<text x="310" y="154" text-anchor="start" class="fig-small">each entry clipped to 2.5: (2.5, 2.5)</text>
<line x1="200" y1="250" x2="258" y2="172" class="stroke-b" marker-end="url(#c106-b)" stroke-width="3"/>
<text x="248" y="174" text-anchor="end" class="fig-small">global scale (1.5, 2)</text>
<text x="214" y="236" text-anchor="start" class="fig-small"></text>
</svg>
Figure 3. One gradient, drawn with its first entry across and its second entry up, and two ways to limit it. Multiplying both entries by 0.5 shrinks the arrow to the circle of radius 2.5 and keeps its direction, 53.1 degrees. Clipping each entry to 2.5 turns it to 45 degrees, a different direction, and the result still has length 3.54, which is over the limit.
```

The **global norm** is the length of all the gradients taken together. A model has hundreds of arrays, such as one matrix per layer, but only one point in weight space, so treat every number in every array as one long vector and measure its length. Square every entry, add them all, and take the square root. Take two arrays, $[3, 4]$ and $[0]$. The squares sum to $9 + 16 + 0 = 25$ and the norm is 5.

Set a limit of 2.5. The gradient is twice too long, so multiply every entry of every array by $2.5/5 = 0.5$. The array $[3, 4]$ becomes $[1.5, 2]$, which has length 2.5 and points the same way. With a limit of 10 the gradient is already short enough, and nothing changes.

Clipping each entry to 2.5 instead gives $[2.5, 2.5]$. That arrow points at 45 degrees instead of 53.1, so the step heads a different way, and its length is still 3.54. It fixed neither problem.

As a formula, with $g$ the global norm:

$$
g = \sqrt{\sum_{\text{all entries}} \text{grad}^2}, \qquad
\text{grad} \leftarrow \text{grad} \cdot \min\!\Big(1, \frac{\text{max\_norm}}{g}\Big)
$$

Read it aloud: measure the total length, and if it is over the limit, shrink every entry by the limit divided by the length. The `min` with 1 means a short gradient is left alone. In numpy:

```python
total = math.sqrt(sum(float((g**2).sum()) for g in grads))
if total > max_norm:
    for g in grads:
        g *= max_norm / total
```

The function also returns `total`, the norm from before the clip, so you can log it.

> [!WARNING]
> Scale in place with `g *= scale`. The line `g = g * scale` builds another array and leaves the caller's gradients unchanged, so nothing is clipped.

## Read the log of gradient norms

Print the returned norm at every step. A healthy log often starts high, falls, and then wanders around a steady value, such as 1.9, 1.3, 0.9, 0.7, 0.6, 0.6, 0.7. A spike is one number far above its neighbours: 0.6, 0.7, 31.0, 0.6. With a limit of 1.0, the clip shrinks that step by a factor of 31 and the run carries on.

Choose a limit that catches spikes and leaves ordinary steps alone. A limit of 1.0 is a common choice for transformers. If the norm is over the limit on every step, the clip has become a second learning rate. Raise the limit or lower the learning rate.

> [!KEY]
> A safe step has a rate suited to the stage of training and a bounded length. The schedule sets the rate: small while the weights are random and the averages are empty, large through the middle, small again to settle. The clip limits the length of any one gradient by scaling all of it together, which keeps its direction. Together they answer the opening question: one odd batch cannot write an arbitrarily long gradient into the update, and no stage of training runs at a rate meant for another.

> [!RECAP]
> - The learning rate ramps up in a straight line over warmup, then glides down along half a cosine to a floor above zero.
> - Warmup protects the first steps, when weights are random and averages are empty. Decay lets the path settle instead of bouncing.
> - The global norm treats every entry of every gradient array as one long vector.
> - Clipping multiplies every array by the same factor, `max_norm / norm`, only when the norm is over the limit. In place, and the pre-clip norm is returned.
> - Clipping each entry separately bends the direction, and can leave the length over the limit.

## Build it step by step

The checks test three functions, in this order. The file also holds `lr_trace`, already written, which lists the schedule at evenly spaced steps so you can see its shape.

1. **`warmup_cosine_lr(step, warmup, total, peak, floor)`.** Return a float. Below `warmup`, return `peak * step / warmup`. From `warmup` on, compute progress as `(step - warmup) / (total - warmup)` and apply the cosine. The usual mistake is dividing by `total` alone, which makes the glide miss the floor.
2. **`global_norm(grads)`.** Take a list of numpy arrays or torch tensors of any shapes. Return the square root of the sum of every squared entry, as a float. Sum the per-array totals with `float(...)` before the square root. Change nothing.
3. **`clip_global_norm(grads, max_norm)`.** Call `global_norm`, and when the total is over `max_norm`, multiply every array by `max_norm / total` in place with `g *= scale`. Return the norm from before the clip. The usual mistake is `g = g * scale`.

## Where this breaks

- A clip set near the typical norm acts on every step, so every update has the same length and the gradient's size is lost.
- A norm that is `nan` or infinite is not fixed by scaling: `nan > max_norm` is false, so the clip leaves it alone, and the weights turn to `nan`. Skip the step instead.
- The schedule needs the true number of steps. If `total` is larger than the run you actually finish, the rate never reaches its floor.

## The exercise

You write `warmup_cosine_lr`, `global_norm`, and `clip_global_norm`. The checks match the schedule at the four worked points and test that the two pieces meet without a jump. They check the global norm on mixed shapes, and that clipping scales to exactly the limit, keeps the direction, works in place, and leaves a short gradient unchanged. Run prints the schedule as a bar chart.
