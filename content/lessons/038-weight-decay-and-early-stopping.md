---
title: "Weight decay and early stopping: two ways to keep weights small"
slug: 038-weight-decay-and-early-stopping
part: "Part 4 — Neural networks"
summary: Pull every weight toward zero on each step, then see that stopping gradient descent early shrinks weights the same way a ridge penalty does.
minutes: 70
gpu: false
objectives:
  - Explain from a picture why a fit that hugs noisy points needs large weights, and why a pull toward zero prevents it.
  - Derive the weight-decay step from a ridge penalty, and show that a weight settles where the pull and the data balance.
  - Explain why a weight vector that feeds a normalisation layer turns through a larger angle when it is shorter.
  - Show that gradient descent from zero after k steps equals ridge with a penalty that depends on k and on the direction's curvature.
  - Pick a stopping step from a noisy validation curve, using patience.
prereqs:
  - 006-ridge-regression
  - 005-gradient-descent-and-momentum
  - 035-sgd-adam-and-schedules
  - 037-normalisation-layers
exercise: 038-weight-decay-and-early-stopping
---

# Weight decay and early stopping: two ways to keep weights small

You have eight noisy measurements and a model with eight adjustable weights. That is enough freedom to pass through every point exactly. Look at the left panel of Figure 1. The curve does hit all eight points, but between them it swings far below anything the trend supports. To reach every wiggle of noise, the curve needs weights as large as 52, and big weights are what make a curve swing.

Nothing in the training loss objects. The loss only asks how far the curve is from the points, so a model with enough weights uses them to chase noise, and an unseen point lands on the wrong part of the swing. A deep network has millions of weights, so you cannot fix this by picking a smaller model. You need a force that keeps weights small while the network trains. This chapter shows two. The first adds a pull toward zero to every step. The second adds nothing at all: you stop training early, and the stopping turns out to be the same kind of act.

```figure
<svg viewBox="0 0 600 266" role="img" aria-label="Two panels with the same eight noisy points. Left: a curve through every point that swings far below the trend, with weights up to about 52. Right: a smooth curve close to the trend with weights up to about 1.7.">
<line x1="50.0" y1="218.0" x2="280.0" y2="218.0" class="stroke-muted"/>
<line x1="50.0" y1="218.0" x2="50.0" y2="48.0" class="stroke-muted"/>
<line x1="60.5" y1="218.0" x2="60.5" y2="222.0" class="stroke-muted"/>
<text x="60.5" y="235.0" text-anchor="middle" class="fig-small">-1</text>
<line x1="165.0" y1="218.0" x2="165.0" y2="222.0" class="stroke-muted"/>
<text x="165.0" y="235.0" text-anchor="middle" class="fig-small">0</text>
<line x1="269.5" y1="218.0" x2="269.5" y2="222.0" class="stroke-muted"/>
<text x="269.5" y="235.0" text-anchor="middle" class="fig-small">1</text>
<line x1="46.0" y1="201.8" x2="50.0" y2="201.8" class="stroke-muted"/>
<text x="43.0" y="205.8" text-anchor="end" class="fig-small">-2</text>
<line x1="46.0" y1="161.3" x2="50.0" y2="161.3" class="stroke-muted"/>
<text x="43.0" y="165.3" text-anchor="end" class="fig-small">-1</text>
<line x1="46.0" y1="120.9" x2="50.0" y2="120.9" class="stroke-muted"/>
<text x="43.0" y="124.9" text-anchor="end" class="fig-small">0</text>
<line x1="46.0" y1="80.4" x2="50.0" y2="80.4" class="stroke-muted"/>
<text x="43.0" y="84.4" text-anchor="end" class="fig-small">1</text>
<path d="M60.5,155.0 L61.5,155.0 L62.6,155.0 L63.6,155.1 L64.7,155.1 L65.7,155.1 L66.8,155.1 L67.8,155.2 L68.9,155.2 L69.9,155.2 L71.0,155.2 L72.0,155.2 L73.1,155.2 L74.1,155.2 L75.2,155.2 L76.2,155.2 L77.3,155.1 L78.3,155.1 L79.4,155.1 L80.4,155.0 L81.5,155.0 L82.5,154.9 L83.6,154.8 L84.6,154.8 L85.7,154.7 L86.7,154.6 L87.8,154.5 L88.8,154.4 L89.9,154.2 L90.9,154.1 L92.0,154.0 L93.0,153.8 L94.1,153.7 L95.1,153.5 L96.2,153.3 L97.2,153.2 L98.3,153.0 L99.3,152.7 L100.4,152.5 L101.4,152.3 L102.5,152.1 L103.5,151.8 L104.6,151.6 L105.6,151.3 L106.7,151.0 L107.7,150.7 L108.8,150.4 L109.8,150.1 L110.9,149.8 L111.9,149.4 L113.0,149.1 L114.0,148.7 L115.1,148.3 L116.1,148.0 L117.2,147.6 L118.2,147.2 L119.3,146.8 L120.3,146.3 L121.4,145.9 L122.4,145.4 L123.5,145.0 L124.5,144.5 L125.6,144.0 L126.6,143.6 L127.7,143.1 L128.8,142.6 L129.8,142.0 L130.9,141.5 L131.9,141.0 L133.0,140.4 L134.0,139.9 L135.1,139.3 L136.1,138.8 L137.2,138.2 L138.2,137.6 L139.3,137.0 L140.3,136.4 L141.4,135.8 L142.4,135.2 L143.5,134.6 L144.5,133.9 L145.6,133.3 L146.6,132.7 L147.7,132.0 L148.7,131.4 L149.8,130.7 L150.8,130.1 L151.9,129.4 L152.9,128.7 L154.0,128.1 L155.0,127.4 L156.1,126.7 L157.1,126.0 L158.2,125.3 L159.2,124.7 L160.3,124.0 L161.3,123.3 L162.4,122.6 L163.4,121.9 L164.5,121.2 L165.5,120.5 L166.6,119.8 L167.6,119.1 L168.7,118.4 L169.7,117.8 L170.8,117.1 L171.8,116.4 L172.9,115.7 L173.9,115.0 L175.0,114.3 L176.0,113.7 L177.1,113.0 L178.1,112.3 L179.2,111.7 L180.2,111.0 L181.3,110.3 L182.3,109.7 L183.4,109.0 L184.4,108.4 L185.5,107.8 L186.5,107.1 L187.6,106.5 L188.6,105.9 L189.7,105.3 L190.7,104.7 L191.8,104.1 L192.8,103.5 L193.9,103.0 L194.9,102.4 L196.0,101.8 L197.0,101.3 L198.1,100.7 L199.1,100.2 L200.2,99.7 L201.2,99.2 L202.3,98.7 L203.4,98.2 L204.4,97.7 L205.5,97.2 L206.5,96.7 L207.6,96.3 L208.6,95.8 L209.7,95.4 L210.7,95.0 L211.8,94.5 L212.8,94.1 L213.9,93.8 L214.9,93.4 L216.0,93.0 L217.0,92.6 L218.1,92.3 L219.1,92.0 L220.2,91.6 L221.2,91.3 L222.3,91.0 L223.3,90.7 L224.4,90.4 L225.4,90.2 L226.5,89.9 L227.5,89.7 L228.6,89.4 L229.6,89.2 L230.7,89.0 L231.7,88.8 L232.8,88.6 L233.8,88.4 L234.9,88.2 L235.9,88.0 L237.0,87.9 L238.0,87.7 L239.1,87.6 L240.1,87.5 L241.2,87.3 L242.2,87.2 L243.3,87.1 L244.3,87.0 L245.4,87.0 L246.4,86.9 L247.5,86.8 L248.5,86.8 L249.6,86.7 L250.6,86.7 L251.7,86.6 L252.7,86.6 L253.8,86.6 L254.8,86.5 L255.9,86.5 L256.9,86.5 L258.0,86.5 L259.0,86.5 L260.1,86.5 L261.1,86.5 L262.2,86.6 L263.2,86.6 L264.3,86.6 L265.3,86.6 L266.4,86.7 L267.4,86.7 L268.5,86.7 L269.5,86.8" fill="none" class="stroke-muted dashed" stroke-width="1.5"/>
<path d="M60.5,153.4 L61.5,163.7 L62.6,172.3 L63.6,179.5 L64.7,185.5 L65.7,190.2 L66.8,193.8 L67.8,196.5 L68.9,198.3 L69.9,199.3 L71.0,199.7 L72.0,199.4 L73.1,198.7 L74.1,197.4 L75.2,195.8 L76.2,193.9 L77.3,191.7 L78.3,189.2 L79.4,186.6 L80.4,183.9 L81.5,181.1 L82.5,178.3 L83.6,175.4 L84.6,172.6 L85.7,169.8 L86.7,167.1 L87.8,164.4 L88.8,161.9 L89.9,159.4 L90.9,157.1 L92.0,155.0 L93.0,153.0 L94.1,151.1 L95.1,149.4 L96.2,147.9 L97.2,146.6 L98.3,145.4 L99.3,144.4 L100.4,143.5 L101.4,142.8 L102.5,142.3 L103.5,141.9 L104.6,141.7 L105.6,141.6 L106.7,141.6 L107.7,141.8 L108.8,142.1 L109.8,142.4 L110.9,142.9 L111.9,143.5 L113.0,144.1 L114.0,144.8 L115.1,145.6 L116.1,146.4 L117.2,147.2 L118.2,148.1 L119.3,149.0 L120.3,149.9 L121.4,150.8 L122.4,151.6 L123.5,152.5 L124.5,153.3 L125.6,154.1 L126.6,154.9 L127.7,155.6 L128.8,156.3 L129.8,156.8 L130.9,157.4 L131.9,157.8 L133.0,158.1 L134.0,158.4 L135.1,158.6 L136.1,158.7 L137.2,158.7 L138.2,158.6 L139.3,158.4 L140.3,158.1 L141.4,157.7 L142.4,157.1 L143.5,156.5 L144.5,155.8 L145.6,155.0 L146.6,154.0 L147.7,153.0 L148.7,151.9 L149.8,150.7 L150.8,149.3 L151.9,147.9 L152.9,146.5 L154.0,144.9 L155.0,143.2 L156.1,141.5 L157.1,139.7 L158.2,137.9 L159.2,136.0 L160.3,134.1 L161.3,132.1 L162.4,130.0 L163.4,128.0 L164.5,125.9 L165.5,123.8 L166.6,121.7 L167.6,119.6 L168.7,117.5 L169.7,115.4 L170.8,113.3 L171.8,111.2 L172.9,109.1 L173.9,107.1 L175.0,105.2 L176.0,103.2 L177.1,101.4 L178.1,99.6 L179.2,97.8 L180.2,96.1 L181.3,94.5 L182.3,93.0 L183.4,91.6 L184.4,90.2 L185.5,88.9 L186.5,87.8 L187.6,86.7 L188.6,85.7 L189.7,84.8 L190.7,84.0 L191.8,83.3 L192.8,82.8 L193.9,82.3 L194.9,81.9 L196.0,81.6 L197.0,81.5 L198.1,81.4 L199.1,81.4 L200.2,81.5 L201.2,81.7 L202.3,82.0 L203.4,82.4 L204.4,82.8 L205.5,83.3 L206.5,83.9 L207.6,84.5 L208.6,85.2 L209.7,86.0 L210.7,86.7 L211.8,87.5 L212.8,88.4 L213.9,89.2 L214.9,90.0 L216.0,90.9 L217.0,91.7 L218.1,92.5 L219.1,93.3 L220.2,94.0 L221.2,94.7 L222.3,95.3 L223.3,95.9 L224.4,96.4 L225.4,96.8 L226.5,97.1 L227.5,97.3 L228.6,97.3 L229.6,97.3 L230.7,97.2 L231.7,96.9 L232.8,96.5 L233.8,95.9 L234.9,95.2 L235.9,94.4 L237.0,93.4 L238.0,92.2 L239.1,90.9 L240.1,89.5 L241.2,88.0 L242.2,86.3 L243.3,84.5 L244.3,82.6 L245.4,80.5 L246.4,78.4 L247.5,76.3 L248.5,74.1 L249.6,71.8 L250.6,69.6 L251.7,67.4 L252.7,65.2 L253.8,63.2 L254.8,61.3 L255.9,59.5 L256.9,58.0 L258.0,56.8 L259.0,55.8 L260.1,55.3 L261.1,55.2 L262.2,55.7 L263.2,56.7 L264.3,58.5 L265.3,61.0 L266.4,64.3 L267.4,68.6 L268.5,74.0 L269.5,80.5" fill="none" class="stroke-b" stroke-width="2.4"/>
<circle cx="60.5" cy="153.4" r="4.5" class="fill-a"/>
<circle cx="90.3" cy="158.4" r="4.5" class="fill-a"/>
<circle cx="120.2" cy="149.7" r="4.5" class="fill-a"/>
<circle cx="150.1" cy="150.3" r="4.5" class="fill-a"/>
<circle cx="179.9" cy="96.6" r="4.5" class="fill-a"/>
<circle cx="209.8" cy="86.1" r="4.5" class="fill-a"/>
<circle cx="239.7" cy="90.2" r="4.5" class="fill-a"/>
<circle cx="269.5" cy="80.5" r="4.5" class="fill-a"/>
<text x="165.0" y="22.0" text-anchor="middle" class="fig-bold">Exact fit: hugs the points</text>
<text x="165.0" y="38.0" text-anchor="middle" class="fig-small">largest weight 52.3</text>
<line x1="340.0" y1="218.0" x2="570.0" y2="218.0" class="stroke-muted"/>
<line x1="340.0" y1="218.0" x2="340.0" y2="48.0" class="stroke-muted"/>
<line x1="350.5" y1="218.0" x2="350.5" y2="222.0" class="stroke-muted"/>
<text x="350.5" y="235.0" text-anchor="middle" class="fig-small">-1</text>
<line x1="455.0" y1="218.0" x2="455.0" y2="222.0" class="stroke-muted"/>
<text x="455.0" y="235.0" text-anchor="middle" class="fig-small">0</text>
<line x1="559.5" y1="218.0" x2="559.5" y2="222.0" class="stroke-muted"/>
<text x="559.5" y="235.0" text-anchor="middle" class="fig-small">1</text>
<line x1="336.0" y1="201.8" x2="340.0" y2="201.8" class="stroke-muted"/>
<text x="333.0" y="205.8" text-anchor="end" class="fig-small">-2</text>
<line x1="336.0" y1="161.3" x2="340.0" y2="161.3" class="stroke-muted"/>
<text x="333.0" y="165.3" text-anchor="end" class="fig-small">-1</text>
<line x1="336.0" y1="120.9" x2="340.0" y2="120.9" class="stroke-muted"/>
<text x="333.0" y="124.9" text-anchor="end" class="fig-small">0</text>
<line x1="336.0" y1="80.4" x2="340.0" y2="80.4" class="stroke-muted"/>
<text x="333.0" y="84.4" text-anchor="end" class="fig-small">1</text>
<path d="M350.5,155.0 L351.5,155.0 L352.6,155.0 L353.6,155.1 L354.7,155.1 L355.7,155.1 L356.8,155.1 L357.8,155.2 L358.9,155.2 L359.9,155.2 L361.0,155.2 L362.0,155.2 L363.1,155.2 L364.1,155.2 L365.2,155.2 L366.2,155.2 L367.3,155.1 L368.3,155.1 L369.4,155.1 L370.4,155.0 L371.5,155.0 L372.5,154.9 L373.6,154.8 L374.6,154.8 L375.7,154.7 L376.7,154.6 L377.8,154.5 L378.8,154.4 L379.9,154.2 L380.9,154.1 L382.0,154.0 L383.0,153.8 L384.1,153.7 L385.1,153.5 L386.2,153.3 L387.2,153.2 L388.3,153.0 L389.3,152.7 L390.4,152.5 L391.4,152.3 L392.5,152.1 L393.5,151.8 L394.6,151.6 L395.6,151.3 L396.7,151.0 L397.7,150.7 L398.8,150.4 L399.8,150.1 L400.9,149.8 L401.9,149.4 L403.0,149.1 L404.0,148.7 L405.1,148.3 L406.1,148.0 L407.2,147.6 L408.2,147.2 L409.3,146.8 L410.3,146.3 L411.4,145.9 L412.4,145.4 L413.5,145.0 L414.5,144.5 L415.6,144.0 L416.6,143.6 L417.7,143.1 L418.8,142.6 L419.8,142.0 L420.9,141.5 L421.9,141.0 L423.0,140.4 L424.0,139.9 L425.1,139.3 L426.1,138.8 L427.2,138.2 L428.2,137.6 L429.3,137.0 L430.3,136.4 L431.4,135.8 L432.4,135.2 L433.5,134.6 L434.5,133.9 L435.6,133.3 L436.6,132.7 L437.7,132.0 L438.7,131.4 L439.8,130.7 L440.8,130.1 L441.9,129.4 L442.9,128.7 L444.0,128.1 L445.0,127.4 L446.1,126.7 L447.1,126.0 L448.2,125.3 L449.2,124.7 L450.3,124.0 L451.3,123.3 L452.4,122.6 L453.4,121.9 L454.5,121.2 L455.5,120.5 L456.6,119.8 L457.6,119.1 L458.7,118.4 L459.7,117.8 L460.8,117.1 L461.8,116.4 L462.9,115.7 L463.9,115.0 L465.0,114.3 L466.0,113.7 L467.1,113.0 L468.1,112.3 L469.2,111.7 L470.2,111.0 L471.3,110.3 L472.3,109.7 L473.4,109.0 L474.4,108.4 L475.5,107.8 L476.5,107.1 L477.6,106.5 L478.6,105.9 L479.7,105.3 L480.7,104.7 L481.8,104.1 L482.8,103.5 L483.9,103.0 L484.9,102.4 L486.0,101.8 L487.0,101.3 L488.1,100.7 L489.1,100.2 L490.2,99.7 L491.2,99.2 L492.3,98.7 L493.4,98.2 L494.4,97.7 L495.5,97.2 L496.5,96.7 L497.6,96.3 L498.6,95.8 L499.7,95.4 L500.7,95.0 L501.8,94.5 L502.8,94.1 L503.9,93.8 L504.9,93.4 L506.0,93.0 L507.0,92.6 L508.1,92.3 L509.1,92.0 L510.2,91.6 L511.2,91.3 L512.3,91.0 L513.3,90.7 L514.4,90.4 L515.4,90.2 L516.5,89.9 L517.5,89.7 L518.6,89.4 L519.6,89.2 L520.7,89.0 L521.7,88.8 L522.8,88.6 L523.8,88.4 L524.9,88.2 L525.9,88.0 L527.0,87.9 L528.0,87.7 L529.1,87.6 L530.1,87.5 L531.2,87.3 L532.2,87.2 L533.3,87.1 L534.3,87.0 L535.4,87.0 L536.4,86.9 L537.5,86.8 L538.5,86.8 L539.6,86.7 L540.6,86.7 L541.7,86.6 L542.7,86.6 L543.8,86.6 L544.8,86.5 L545.9,86.5 L546.9,86.5 L548.0,86.5 L549.0,86.5 L550.1,86.5 L551.1,86.5 L552.2,86.6 L553.2,86.6 L554.3,86.6 L555.3,86.6 L556.4,86.7 L557.4,86.7 L558.5,86.7 L559.5,86.8" fill="none" class="stroke-muted dashed" stroke-width="1.5"/>
<path d="M350.5,153.4 L351.5,154.1 L352.6,154.8 L353.6,155.4 L354.7,156.0 L355.7,156.6 L356.8,157.1 L357.8,157.6 L358.9,158.0 L359.9,158.4 L361.0,158.8 L362.0,159.1 L363.1,159.4 L364.1,159.7 L365.2,159.9 L366.2,160.1 L367.3,160.3 L368.3,160.4 L369.4,160.6 L370.4,160.6 L371.5,160.7 L372.5,160.7 L373.6,160.8 L374.6,160.7 L375.7,160.7 L376.7,160.6 L377.8,160.5 L378.8,160.4 L379.9,160.3 L380.9,160.1 L382.0,160.0 L383.0,159.8 L384.1,159.6 L385.1,159.3 L386.2,159.1 L387.2,158.8 L388.3,158.5 L389.3,158.2 L390.4,157.9 L391.4,157.6 L392.5,157.2 L393.5,156.8 L394.6,156.5 L395.6,156.1 L396.7,155.7 L397.7,155.2 L398.8,154.8 L399.8,154.4 L400.9,153.9 L401.9,153.4 L403.0,152.9 L404.0,152.5 L405.1,151.9 L406.1,151.4 L407.2,150.9 L408.2,150.4 L409.3,149.8 L410.3,149.3 L411.4,148.7 L412.4,148.2 L413.5,147.6 L414.5,147.0 L415.6,146.4 L416.6,145.8 L417.7,145.2 L418.8,144.6 L419.8,144.0 L420.9,143.4 L421.9,142.8 L423.0,142.1 L424.0,141.5 L425.1,140.9 L426.1,140.2 L427.2,139.6 L428.2,138.9 L429.3,138.2 L430.3,137.6 L431.4,136.9 L432.4,136.2 L433.5,135.6 L434.5,134.9 L435.6,134.2 L436.6,133.5 L437.7,132.9 L438.7,132.2 L439.8,131.5 L440.8,130.8 L441.9,130.1 L442.9,129.4 L444.0,128.7 L445.0,128.0 L446.1,127.3 L447.1,126.6 L448.2,125.9 L449.2,125.2 L450.3,124.5 L451.3,123.8 L452.4,123.1 L453.4,122.4 L454.5,121.7 L455.5,121.0 L456.6,120.3 L457.6,119.6 L458.7,118.9 L459.7,118.2 L460.8,117.5 L461.8,116.9 L462.9,116.2 L463.9,115.5 L465.0,114.8 L466.0,114.1 L467.1,113.4 L468.1,112.7 L469.2,112.1 L470.2,111.4 L471.3,110.7 L472.3,110.0 L473.4,109.4 L474.4,108.7 L475.5,108.1 L476.5,107.4 L477.6,106.7 L478.6,106.1 L479.7,105.5 L480.7,104.8 L481.8,104.2 L482.8,103.6 L483.9,102.9 L484.9,102.3 L486.0,101.7 L487.0,101.1 L488.1,100.5 L489.1,99.9 L490.2,99.3 L491.2,98.7 L492.3,98.2 L493.4,97.6 L494.4,97.0 L495.5,96.5 L496.5,95.9 L497.6,95.4 L498.6,94.9 L499.7,94.4 L500.7,93.8 L501.8,93.3 L502.8,92.8 L503.9,92.3 L504.9,91.9 L506.0,91.4 L507.0,90.9 L508.1,90.5 L509.1,90.1 L510.2,89.6 L511.2,89.2 L512.3,88.8 L513.3,88.4 L514.4,88.0 L515.4,87.6 L516.5,87.3 L517.5,86.9 L518.6,86.5 L519.6,86.2 L520.7,85.9 L521.7,85.6 L522.8,85.3 L523.8,85.0 L524.9,84.7 L525.9,84.4 L527.0,84.2 L528.0,83.9 L529.1,83.7 L530.1,83.5 L531.2,83.3 L532.2,83.1 L533.3,82.9 L534.3,82.7 L535.4,82.6 L536.4,82.4 L537.5,82.3 L538.5,82.1 L539.6,82.0 L540.6,81.9 L541.7,81.8 L542.7,81.7 L543.8,81.6 L544.8,81.6 L545.9,81.5 L546.9,81.5 L548.0,81.4 L549.0,81.4 L550.1,81.4 L551.1,81.4 L552.2,81.4 L553.2,81.3 L554.3,81.4 L555.3,81.4 L556.4,81.4 L557.4,81.4 L558.5,81.4 L559.5,81.4" fill="none" class="stroke-b" stroke-width="2.4"/>
<circle cx="350.5" cy="153.4" r="4.5" class="fill-a"/>
<circle cx="380.3" cy="158.4" r="4.5" class="fill-a"/>
<circle cx="410.2" cy="149.7" r="4.5" class="fill-a"/>
<circle cx="440.1" cy="150.3" r="4.5" class="fill-a"/>
<circle cx="469.9" cy="96.6" r="4.5" class="fill-a"/>
<circle cx="499.8" cy="86.1" r="4.5" class="fill-a"/>
<circle cx="529.7" cy="90.2" r="4.5" class="fill-a"/>
<circle cx="559.5" cy="80.5" r="4.5" class="fill-a"/>
<text x="455.0" y="22.0" text-anchor="middle" class="fig-bold">Same model, weights pulled in</text>
<text x="455.0" y="38.0" text-anchor="middle" class="fig-small">largest weight 1.72</text>
<line x1="50" y1="252" x2="74" y2="252" class="stroke-muted dashed"/>
<text x="80.0" y="256.0" text-anchor="start" class="fig-small">trend that made the data</text>
<circle cx="250" cy="252" r="4.5" class="fill-a"/>
<text x="259.0" y="256.0" text-anchor="start" class="fig-small">8 noisy points</text>
<line x1="355" y1="252" x2="379" y2="252" class="stroke-b"/>
<text x="385.0" y="256.0" text-anchor="start" class="fig-small">fitted curve (8 weights)</text>
</svg>
Figure 1. Both curves use the same eight-weight model and the same eight points. On the left nothing limits the weights, so the curve swings between points (down to -1.9) while the trend stays near -0.8. On the right a pull toward zero keeps the weights small, and the curve stays close to the trend.
```

## What you already know

- A gradient step moves each weight a little against its slope. The size of the move is the learning rate, written $\alpha$.
- [Ridge regression](/lessons/006-ridge-regression) adds a fee on the squared weights to the loss, so large weights cost more. The slope of $\tfrac{1}{2}\lambda w^2$ is $\lambda w$.
- On one axis, a gradient step multiplies your distance from the bottom of a bowl by $1 - \alpha \times \text{curvature}$ ([gradient descent](/lessons/005-gradient-descent-and-momentum)).
- A validation loss, measured on data the network never trains on, shows how well it generalises.
- A [normalisation layer](/lessons/037-normalisation-layers) rescales its inputs, so it cannot see their overall size.

## Why large weights, and what to do about them

The right panel of Figure 1 is the same model, with a small pull toward zero acting on every weight. It shows where gradient steps with that pull settle: the minimum of the squared misses plus a ridge fee with $\lambda = 0.05$. The largest weight drops from 52 to 1.7, the curve stays close to the dashed trend, and its squared error against that trend is about 29 times smaller. It misses the training points a little, and that is the price. A fit that hugs the noise is not better, only closer to the training set.

You might ask why not set a hard ceiling and clip any weight that exceeds it. A ceiling treats a weight at 0.99 times the limit as fine and one at 1.01 times the limit as broken. Weights then pile up at the wall, and the gradient that pushed them there is thrown away. A pull proportional to size has no wall. Big weights feel it most, small weights barely feel it, and the pull is the slope of a penalty you already know.

## A pull toward zero on every step

Picture each weight tied to the origin by a rubber band. The data pulls the weight toward whatever value reduces the loss, and the band pulls it back toward zero. The band stretches further, and pulls harder, the further the weight is from zero.

Take one weight $w = 1$, a data gradient $g = 0.1$, a learning rate $\alpha = 0.1$, and a pull strength $\lambda = 0.01$. The data step removes $\alpha g = 0.01$. The band removes $\alpha \lambda w = 0.1 \times 0.01 \times 1 = 0.001$. The updated weight is $1 - 0.01 - 0.001 = 0.989$.

See where the band comes from. Add the ridge fee $\tfrac{\lambda}{2}\|w\|^2$ to the loss, where $\|w\|^2$ is the sum of the squared weights. Its slope with respect to each weight is $\lambda w$, so a gradient step on the total loss moves by $\alpha(g + \lambda w)$:

$$
w_{\text{new}} = w - \alpha g - \alpha \lambda w
$$

Read it aloud: "take the usual step, then also give up a fraction $\alpha\lambda$ of the weight." The $\tfrac{1}{2}$ in the fee cancels the 2 from differentiating the square. In numpy:

```python
penalty_grad = lam * w           # slope of (lam / 2) * sum(w**2)
new_w = w - lr * (grad + penalty_grad)
```

That is **weight decay**. For plain gradient descent it is exactly a gradient step on the ridge loss, which is why it keeps weights small the same way ridge does.

## What decay does over many steps

One step hardly shows the effect, so follow one weight for 30 steps. First suppose the data stops caring, so $g = 0$. Each step multiplies the weight by $1 - \alpha\lambda$, and after $k$ steps the weight is $w_0 (1 - \alpha\lambda)^k$. With $\alpha = 0.5$ and $\lambda = 0.2$ the factor is $0.9$. The weight is $0.349$ after 10 steps and $0.042$ after 30. Figure 2 shows the curves for decay 0, 0.05, and 0.2.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Line chart of one weight over 30 steps. With decay 0 it stays at 1, with decay 0.05 it falls to about 0.47, with decay 0.2 it falls to about 0.04. A fourth curve, with data pulling toward 0.6, settles at 0.5.">
<line x1="50.0" y1="220.0" x2="550.0" y2="220.0" class="stroke-muted"/>
<line x1="50.0" y1="220.0" x2="50.0" y2="40.0" class="stroke-muted"/>
<line x1="50.0" y1="220.0" x2="50.0" y2="40.0" class="stroke-grid"/>
<line x1="50.0" y1="220.0" x2="50.0" y2="224.0" class="stroke-muted"/>
<text x="50.0" y="237.0" text-anchor="middle" class="fig-small">0</text>
<line x1="216.7" y1="220.0" x2="216.7" y2="40.0" class="stroke-grid"/>
<line x1="216.7" y1="220.0" x2="216.7" y2="224.0" class="stroke-muted"/>
<text x="216.7" y="237.0" text-anchor="middle" class="fig-small">10</text>
<line x1="383.3" y1="220.0" x2="383.3" y2="40.0" class="stroke-grid"/>
<line x1="383.3" y1="220.0" x2="383.3" y2="224.0" class="stroke-muted"/>
<text x="383.3" y="237.0" text-anchor="middle" class="fig-small">20</text>
<line x1="550.0" y1="220.0" x2="550.0" y2="40.0" class="stroke-grid"/>
<line x1="550.0" y1="220.0" x2="550.0" y2="224.0" class="stroke-muted"/>
<text x="550.0" y="237.0" text-anchor="middle" class="fig-small">30</text>
<line x1="50.0" y1="220.0" x2="550.0" y2="220.0" class="stroke-grid"/>
<line x1="46.0" y1="220.0" x2="50.0" y2="220.0" class="stroke-muted"/>
<text x="43.0" y="224.0" text-anchor="end" class="fig-small">0</text>
<line x1="50.0" y1="134.3" x2="550.0" y2="134.3" class="stroke-grid"/>
<line x1="46.0" y1="134.3" x2="50.0" y2="134.3" class="stroke-muted"/>
<text x="43.0" y="138.3" text-anchor="end" class="fig-small">0.5</text>
<line x1="50.0" y1="48.6" x2="550.0" y2="48.6" class="stroke-grid"/>
<line x1="46.0" y1="48.6" x2="50.0" y2="48.6" class="stroke-muted"/>
<text x="43.0" y="52.6" text-anchor="end" class="fig-small">1</text>
<text x="300.0" y="254.0" text-anchor="middle" class="fig-small">step k</text>
<text x="50.0" y="32.0" text-anchor="start" class="fig-small">weight</text>
<path d="M50.0,48.6 L66.7,48.6 L83.3,48.6 L100.0,48.6 L116.7,48.6 L133.3,48.6 L150.0,48.6 L166.7,48.6 L183.3,48.6 L200.0,48.6 L216.7,48.6 L233.3,48.6 L250.0,48.6 L266.7,48.6 L283.3,48.6 L300.0,48.6 L316.7,48.6 L333.3,48.6 L350.0,48.6 L366.7,48.6 L383.3,48.6 L400.0,48.6 L416.7,48.6 L433.3,48.6 L450.0,48.6 L466.7,48.6 L483.3,48.6 L500.0,48.6 L516.7,48.6 L533.3,48.6 L550.0,48.6" fill="none" class="stroke-d" stroke-width="2.4"/>
<path d="M50.0,48.6 L66.7,52.9 L83.3,57.0 L100.0,61.1 L116.7,65.1 L133.3,69.0 L150.0,72.7 L166.7,76.4 L183.3,80.0 L200.0,83.5 L216.7,86.9 L233.3,90.2 L250.0,93.5 L266.7,96.6 L283.3,99.7 L300.0,102.7 L316.7,105.7 L333.3,108.5 L350.0,111.3 L366.7,114.0 L383.3,116.7 L400.0,119.3 L416.7,121.8 L433.3,124.2 L450.0,126.6 L466.7,129.0 L483.3,131.2 L500.0,133.5 L516.7,135.6 L533.3,137.7 L550.0,139.8" fill="none" class="stroke-a" stroke-width="2.4"/>
<path d="M50.0,48.6 L66.7,65.7 L83.3,81.1 L100.0,95.0 L116.7,107.5 L133.3,118.8 L150.0,128.9 L166.7,138.0 L183.3,146.2 L200.0,153.6 L216.7,160.2 L233.3,166.2 L250.0,171.6 L266.7,176.4 L283.3,180.8 L300.0,184.7 L316.7,188.2 L333.3,191.4 L350.0,194.3 L366.7,196.8 L383.3,199.2 L400.0,201.2 L416.7,203.1 L433.3,204.8 L450.0,206.3 L466.7,207.7 L483.3,208.9 L500.0,210.0 L516.7,211.0 L533.3,211.9 L550.0,212.7" fill="none" class="stroke-b" stroke-width="2.4"/>
<path d="M50.0,48.6 L66.7,100.0 L83.3,120.6 L100.0,128.8 L116.7,132.1 L133.3,133.4 L150.0,133.9 L166.7,134.1 L183.3,134.2 L200.0,134.3 L216.7,134.3 L233.3,134.3 L250.0,134.3 L266.7,134.3 L283.3,134.3 L300.0,134.3 L316.7,134.3 L333.3,134.3 L350.0,134.3 L366.7,134.3 L383.3,134.3 L400.0,134.3 L416.7,134.3 L433.3,134.3 L450.0,134.3 L466.7,134.3 L483.3,134.3 L500.0,134.3 L516.7,134.3 L533.3,134.3 L550.0,134.3" fill="none" class="stroke-c" stroke-width="2.4"/>
<line x1="50.0" y1="134.3" x2="550.0" y2="134.3" class="stroke-muted dashed" stroke-width="1.2"/>
<text x="550.0" y="40.9" text-anchor="end" class="fig-small">decay 0: stays at 1</text>
<text x="550.0" y="161.8" text-anchor="end" class="fig-small">decay 0.05</text>
<text x="550.0" y="204.7" text-anchor="end" class="fig-small">decay 0.2</text>
<text x="200.0" y="126.3" text-anchor="start" class="fig-small">data pulls to 0.6, settles at 0.5</text>
</svg>
Figure 2. One weight over 30 steps with learning rate 0.5. With no data gradient the weight shrinks by a fixed fraction each step, so the curve falls steeply at first and flattens. The green curve adds a data gradient that pulls toward 0.6, and decay 0.2 holds it at 0.5 instead.
```

Suppose instead that the data does care. Let it pull the weight toward $0.6$ with slope $g = w - 0.6$. The weight stops moving when the data step and the band step cancel, so $(w - 0.6) + 0.2\,w = 0$, which gives $w = 0.6 / 1.2 = 0.5$. The green curve in Figure 2 settles there. In general, when the data's slope is $w - w^*$, the weight settles at $w^*/(1+\lambda)$, where $w^*$ is where the data alone would put it. The pull always wins a little, by a larger margin when $\lambda$ is larger. A steeper data slope resists the pull more, and the second half of this chapter works out exactly how much.

A weight therefore survives only while the data keeps paying for it. A weight the data ignores decays toward zero, and a weight the data wants stays close to where the data wants it, a little closer to zero. One caution: this equals ridge only for plain gradient descent. Adam divides each weight's gradient by its typical size, which would also divide the pull if you folded it into the gradient. The [AdamW discussion](/lessons/035-sgd-adam-and-schedules) shows why the pull belongs outside that division.

> [!KEY]
> Weight decay shrinks every weight by the fraction $\alpha\lambda$ per step, on top of the data step. A weight with no gradient fades geometrically, and a weight with a gradient settles where the data and the pull balance.

## Weights that feed a normalisation layer

There is one place where decay behaves differently, and a picture explains it. Suppose a weight vector feeds a normalisation layer. That layer rescales its inputs, so multiplying the whole weight vector by 2 leaves the layer's output, and therefore the loss, unchanged. Only the direction of the weight vector matters, not its length.

Because the loss does not change when you slide the vector along its own length, the gradient has no component along that length. It points sideways, at a right angle to the vector. The vector turns by an angle whose tangent is the step length divided by the vector length:

$$
\theta = \arctan\!\left(\frac{\alpha \,\|g\|}{\|w\|}\right)
$$

Read it aloud: "the angle is how big the sideways step is compared with how long the vector is." A step of length 1 on a vector of length 4 turns it through $\arctan(1/4) = 14.0°$. The same step on a vector of length 2 turns it through $\arctan(1/2) = 26.6°$. Shorter weights change direction faster.

Two consequences follow. First, a sideways step makes the vector slightly longer, from length 4 to $\sqrt{4^2 + 1^2} = 4.12$, so without decay these weights drift longer and turn less and less. Decay shrinks them back and keeps them turning. Second, in a real network the gradient is itself larger when the weights are shorter, because a sideways nudge of a given length is a bigger change of direction on a short vector. The effect is stronger than the picture shows. For these weights, decay does not change the function, because their size is invisible to the loss. It changes how quickly the direction moves, which is why people say it acts like a larger learning rate.

> [!WARNING]
> Do not decay biases or the scale and shift of a normalisation layer. Those parameters are few, they are not where a fit hugs noise, and decay on the scale pulls normalised outputs toward zero.

## Stopping early does the same job

Weight decay writes the pull into the update. Suppose instead that you write nothing, and watch two numbers instead. The training loss falls at every step. The validation loss falls too, bottoms out, and then climbs while training keeps improving. In one run it reaches 0.118 at step 40 and climbs to 0.404 by step 200, while the training loss falls to 0.014. You stop at step 40 and keep those weights. You never wrote a penalty, yet the result is as if you had. The rest of this chapter shows why.

To see it, use the simplest loss with a shape: a bowl. Rotate the weights so that the bowl's axes do not interact. Along each such axis, called a **direction**, the loss is a parabola $\tfrac{1}{2}c\,(w - w^*)^2$. Here $w^*$ is the best weight for that direction on the training data, and the **curvature** $c$ says how sharply the parabola bends. (Linear algebra calls these curvatures eigenvalues, which is why the code says `eigenvalue`.) Gradient descent treats each direction on its own, so a single direction is enough to follow the story.

A flat direction, with small $c$, is one where changing the weight hardly changes the training loss, so the data hardly pins it down. Noise then decides where $w^*$ lands. In the run above there are four directions with curvatures 5, 2, 0.5, and 0.1. The true best weight in every direction is 1, but the training optimum is $[1.0, 0.7, 0.8, 4.0]$. The flat direction is the worst, far wrong at 4.0. Gradient descent starts at zero, learns the steep directions first, and reaches the flat one last, so the weights are best just before the flat direction overshoots.

The word "first" needs a picture. Figure 3 draws a bowl with two directions: a steep one with curvature 10 and a flat one with curvature 0.5, so the bowl is a long narrow valley. The answer $w^*$ sits at 1 on the flat axis and 0.6 on the steep axis, and gradient descent at step size 0.05 starts at the origin. The steep coordinate multiplies its remaining distance by $1 - 0.05 \times 10 = 0.5$ each step, so after 5 steps it has covered 97% of the way. The flat coordinate multiplies its remaining distance by $1 - 0.05 \times 0.5 = 0.975$, so after 5 steps it has covered only 12%.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Contour plot of a stretched bowl with its minimum marked at (1, 0.6). The gradient descent path from the origin climbs the steep direction in a few steps, then crawls along the flat direction toward the minimum, reaching 12 percent of the flat distance at 5 steps, 40 percent at 20 steps and 92 percent at 100 steps.">
<defs><marker id="c032-arrow-1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
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
Figure 3. Contours of a stretched bowl whose answer w* is at (1, 0.6), with gradient descent from the origin at step size 0.05. The path rises almost straight up first, then crawls sideways. The steep coordinate is 97% of the way to its answer after 5 steps, while the flat coordinate has covered only 12%. It needs about 100 steps to get most of the way (92%).
```

Stop after 5 steps and you hold a weight that is nearly right in the steep direction and still near zero in the flat one. That is what a pull toward zero does: it wins against a weak data gradient and loses against a strong one. Flat directions have the weakest gradient, so the pull holds them near zero. Early stopping gets the same result without any pull, because it never lets the flat directions travel far.

## One direction, worked by hand

Make it exact on one direction. Take curvature $c = 2$, step size $\alpha = 0.1$, and $w^* = 1$. Start at $w = 0$. The slope is $c\,(w - w^*) = 2 \times (0 - 1) = -2$, so one step lands at $0 - 0.1 \times (-2) = 0.2$. The distance from $w^*$ went from 1 to 0.8, which is $1 - \alpha c = 0.8$ times the old distance. That factor is the same at every step, so after $k$ steps the distance is $0.8^k$ times the starting distance:

$$
w_k = \big(1 - (1 - \alpha c)^k\big)\, w^*
$$

Read it aloud: "the weight has covered all but a fraction $(1-\alpha c)^k$ of the way to the answer." With $k = 10$ the weight is $1 - 0.8^{10} = 0.89263$.

Next, take ridge on the same direction. The loss becomes $\tfrac{1}{2}c\,(w - w^*)^2 + \tfrac{1}{2}\lambda w^2$. Its slope is $c\,(w - w^*) + \lambda w$, which is zero at

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

At $k = 0$ nothing has been learned, which is exactly what an infinitely strong penalty does. As $k$ grows the penalty falls to zero. Figure 4 shows both curves. Fewer steps mean a stronger penalty, so the step count is a dial.

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
<line x1="50.0" y1="196.4" x2="50.0" y2="191.0" class="stroke-b" stroke-width="2.2" marker-end="url(#c032-arrow-1)"/>
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
Figure 4. For curvature 2 and step 0.1, the fraction reached rises as the penalty falls: at k = 0 the penalty is infinite, at k = 1 it is 8, and at k = 10 it is 0.241. Right: after 25 steps a steep direction (c = 5) is fully released, while a flat one (c = 0.1) has moved only 22% of the way, so it still carries a large penalty.
```

The match is direction by direction, and the penalty differs between directions. After 25 steps at $\alpha = 0.1$, a direction with curvature 5 has penalty near $1.5 \times 10^{-7}$, curvature 1 has 0.077, and curvature 0.1 has 0.35. Early stopping is ridge with a small penalty on steep directions and a large one on flat directions.

> [!KEY]
> Gradient descent from zero covers steep directions first, so after $k$ steps each direction is shrunk as if by a ridge penalty, large on flat directions and small on steep ones. Fewer steps mean a larger penalty.

## When the match holds

The formula needs $0 < \alpha c \le 1$. In that range, $1 - \alpha c$ lies between 0 and 1, so $b$ is a fraction that shrinks and $\lambda$ is positive. If $1 < \alpha c < 2$, then $1 - \alpha c$ is negative: each step overshoots $w^*$ and the error flips sign. At odd $k$, $b$ is negative and $\lambda$ comes out negative, a meaningless fee. If $\alpha c > 2$, the error grows at every step and gradient descent diverges, which is the 2/L limit from the [gradient descent chapter](/lessons/005-gradient-descent-and-momentum). So the function you write raises a `ValueError` outside the valid range. At $\alpha c = 1$, a single step lands on $w^*$ and the penalty is 0.

## Stopping by watching the validation loss

You only see the validation curve up to the present step, so you find its lowest point by remembering the best step so far. A real validation loss is also jumpy, so a single bad check does not mean you have passed the lowest point. **Patience** is how many checks in a row you wait without a better loss before giving up. Take the losses $[5.0, 4.0, 4.5, 3.0, 3.5, 3.6, 3.7]$. With patience 1, you see 4.5 after the best 4.0, give up, and return step 1, missing the 3.0 at step 3. With patience 2 you survive 4.5, find 3.0, then see 3.5 and 3.6 with no improvement and return step 3.

> [!KEY]
> Large weights are the symptom of a fit that chases noise. Weight decay cures it by pulling every weight toward zero on each step. Early stopping cures it by never letting flat directions travel far, which is a ridge penalty you never wrote. You pick the stopping step by watching validation loss with some patience.

> [!RECAP]
> Hold these points before you write code:
>
> - The pull is the slope of the ridge fee $\tfrac{\lambda}{2}\|w\|^2$, which is $\lambda w$. One step is $w - \alpha(g + \lambda w)$, and with no gradient $k$ steps leave $w_0(1-\alpha\lambda)^k$.
> - For weights that feed a normalisation layer, the step is sideways, and the angle turned is $\arctan(\alpha\|g\| / \|w\|)$.
> - Gradient descent from zero on one direction gives $w_k = (1 - (1-\alpha c)^k)\, w^*$, and ridge keeps the share $c/(c+\lambda)$ of $w^*$.
> - Matching the two gives $\lambda = c\,b/(1-b)$ with $b = (1-\alpha c)^k$. It is infinite at $k = 0$, and the match holds only for $0 < \alpha c \le 1$.
> - Patience waits through several checks without improvement before it stops.

## Build it step by step

The checks call the functions in this order:

1. **`l2_penalty_grad(w, lam)`.** Take the weights and the pull strength. Return the slope of the ridge fee, with the same shape as `w`. The usual mistake is to forget that the fee has a $\tfrac{1}{2}$, so its slope has no extra factor of 2.
2. **`weight_decay_step(w, grad, lr, weight_decay)`.** Take the weights, the data gradient, the learning rate, and the pull strength. Return the weights after one step, built from your first function. The usual mistake is to leave `lr` out of the pull, so that the decay does not shrink with the learning rate.
3. **`step_angle(w, grad, lr)`.** Take a weight vector and a gradient that is at a right angle to it. Return the angle in radians that one step turns the vector through. Use `np.linalg.norm` for both lengths. The usual mistake is to return degrees, or to use `arcsin` where the picture calls for the tangent.
4. **`gd_quadratic_1d(eigenvalue, step, w_star, iterations)`.** Run gradient descent from zero on $\tfrac{1}{2}c\,(w - w^*)^2$ with a loop, and return the final weight as a float. The usual mistake is to start at $w^*$ instead of 0.
5. **`early_stopping_equivalent_ridge(eigenvalue, step, iterations)`.** Return the penalty from the formula as a float. Return `float("inf")` when `iterations` is 0, and raise `ValueError` when `step * eigenvalue` is not in $(0, 1]$. The usual mistake is dividing by zero at 0 steps.
6. **`best_stopping_step(val_losses, patience)`.** Scan the list in order. Count the checks since the last strict improvement, stop when that count reaches `patience`, and return the index of the best loss seen. The usual mistake is to return the index of the global minimum, which a real run cannot know.

## Where this breaks

- Too much decay makes the weights too small to fit even the real signal. The right strength depends on the optimiser and the model, so search for it on held-out data. Decay controls the size of weights, not how many matter: it shrinks all of them a little and none to exactly zero.
- Adding the pull to the gradient and decoupling it are the same only for plain gradient descent. With Adam, use the decoupled form.
- A real network's loss is not a quadratic bowl. The early-stopping picture holds near the starting point and gets looser as training moves far away. Adam also evens out how fast different weights move, so the order of steep first and flat last holds less cleanly.
- You choose the stopping step with validation data, so the validation score at that step is a little too good. Keep a separate test set, as the discussion of the [winner's curse](/lessons/014-learning-curves-and-winners-curse) explains.

## The exercise

You implement `l2_penalty_grad`, `weight_decay_step`, `step_angle`, `gd_quadratic_1d`, `early_stopping_equivalent_ridge`, and `best_stopping_step`.

The checks compare `l2_penalty_grad` with a finite difference of the ridge fee, and they check your step on the worked example, where $w = 1$ becomes $0.989$. They confirm that your step equals a gradient step on a loss that includes the fee, that a weight with no gradient fades as $(1 - \alpha\lambda)^k$, and that a weight pulled toward $0.6$ settles at $0.5$. Then they compare `step_angle` with an angle measured from the vectors. Next they compare your loop with $(1 - (1-\alpha c)^k)\,w^*$, including a step that overshoots, check the penalty at 0, 1, and 10 steps against the table, and confirm that $k$ steps of gradient descent equal the ridge solution with your penalty. They confirm that the function refuses a step above $1/c$. The last check runs `best_stopping_step` on a V-shaped curve and on a jumpy one, where patience decides which dip you find.
