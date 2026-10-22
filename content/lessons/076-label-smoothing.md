---
title: "Label smoothing: ask for a little less certainty"
slug: 076-label-smoothing
part: "Part 4 — Neural networks"
summary: Soften the one-hot target so the loss is lowest at a finite confidence, and scores stop growing without bound.
minutes: 40
gpu: false
objectives:
  - Explain why a loss against one-hot targets keeps pushing scores upward, and what that costs on a wrong or mislabelled example.
  - Build the smoothed target and the loss against it, using a max-shifted log-softmax.
  - Derive why the gradient is the predicted probabilities minus the smoothed target, divided by the batch size.
  - Compute where the loss is lowest, and the score gap that corresponds to it.
prereqs:
  - 063-softmax-cross-entropy-gradient
  - 075-dropout
exercise: 076-label-smoothing
---

# Label smoothing: ask for a little less certainty

A classifier looks at photos of cats, dogs, and foxes. After enough training it says "100 percent dog" on every dog in the training set. Then it meets a fox that looks like a dog and says "100 percent dog" again. It is wrong and completely sure of itself. A model that says 100 percent on every training example will also say 100 percent when it is wrong, and it earns that habit honestly: the loss rewards it for every extra bit of certainty.

The reward never stops. With a one-hot target, the loss is only zero when the right class has probability 1, which needs an infinite gap between scores. So training keeps raising the scores for as long as it runs. The question is: how do you ask the model for 93 percent instead of 100, using only a change to the target?

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Two loss curves against the probability given to the right class. The one-hot curve falls steadily toward zero at probability 1. The smoothed curve reaches its lowest value, 0.291, at probability 0.9333 and then rises.">
<line x1="60.0" y1="226.0" x2="560.0" y2="226.0" class="stroke-muted"/>
<line x1="60.0" y1="226.0" x2="60.0" y2="36.0" class="stroke-muted"/>
<line x1="60.0" y1="226.0" x2="60.0" y2="230.0" class="stroke-muted"/>
<text x="60.0" y="243.0" text-anchor="middle" class="fig-small">0.5</text>
<line x1="160.0" y1="226.0" x2="160.0" y2="230.0" class="stroke-muted"/>
<text x="160.0" y="243.0" text-anchor="middle" class="fig-small">0.6</text>
<line x1="260.0" y1="226.0" x2="260.0" y2="230.0" class="stroke-muted"/>
<text x="260.0" y="243.0" text-anchor="middle" class="fig-small">0.7</text>
<line x1="360.0" y1="226.0" x2="360.0" y2="230.0" class="stroke-muted"/>
<text x="360.0" y="243.0" text-anchor="middle" class="fig-small">0.8</text>
<line x1="460.0" y1="226.0" x2="460.0" y2="230.0" class="stroke-muted"/>
<text x="460.0" y="243.0" text-anchor="middle" class="fig-small">0.9</text>
<line x1="560.0" y1="226.0" x2="560.0" y2="230.0" class="stroke-muted"/>
<text x="560.0" y="243.0" text-anchor="middle" class="fig-small">1</text>
<line x1="56.0" y1="226.0" x2="60.0" y2="226.0" class="stroke-muted"/>
<text x="53.0" y="230.0" text-anchor="end" class="fig-small">0</text>
<line x1="56.0" y1="131.0" x2="60.0" y2="131.0" class="stroke-muted"/>
<text x="53.0" y="135.0" text-anchor="end" class="fig-small">0.4</text>
<line x1="56.0" y1="36.0" x2="60.0" y2="36.0" class="stroke-muted"/>
<text x="53.0" y="40.0" text-anchor="end" class="fig-small">0.8</text>
<text x="310.0" y="260.0" text-anchor="middle" class="fig-small">probability given to the right class</text>
<text x="60.0" y="28.0" text-anchor="start" class="fig-small">loss for one row</text>
<path d="M60.0,61.4 L61.3,62.0 L62.5,62.6 L63.8,63.2 L65.0,63.7 L66.3,64.3 L67.5,64.9 L68.8,65.5 L70.0,66.1 L71.3,66.7 L72.5,67.3 L73.8,67.8 L75.0,68.4 L76.3,69.0 L77.5,69.6 L78.8,70.1 L80.0,70.7 L81.3,71.3 L82.5,71.8 L83.8,72.4 L85.0,73.0 L86.3,73.5 L87.5,74.1 L88.8,74.7 L90.0,75.2 L91.3,75.8 L92.5,76.4 L93.8,76.9 L95.1,77.5 L96.3,78.0 L97.6,78.6 L98.8,79.1 L100.1,79.7 L101.3,80.2 L102.6,80.8 L103.8,81.3 L105.1,81.9 L106.3,82.4 L107.6,83.0 L108.8,83.5 L110.1,84.0 L111.3,84.6 L112.6,85.1 L113.8,85.7 L115.1,86.2 L116.3,86.7 L117.6,87.3 L118.8,87.8 L120.1,88.3 L121.3,88.9 L122.6,89.4 L123.8,89.9 L125.1,90.4 L126.3,91.0 L127.6,91.5 L128.9,92.0 L130.1,92.5 L131.4,93.1 L132.6,93.6 L133.9,94.1 L135.1,94.6 L136.4,95.1 L137.6,95.6 L138.9,96.2 L140.1,96.7 L141.4,97.2 L142.6,97.7 L143.9,98.2 L145.1,98.7 L146.4,99.2 L147.6,99.7 L148.9,100.2 L150.1,100.7 L151.4,101.2 L152.6,101.7 L153.9,102.2 L155.1,102.7 L156.4,103.2 L157.6,103.7 L158.9,104.2 L160.2,104.7 L161.4,105.2 L162.7,105.7 L163.9,106.2 L165.2,106.7 L166.4,107.2 L167.7,107.7 L168.9,108.2 L170.2,108.7 L171.4,109.2 L172.7,109.6 L173.9,110.1 L175.2,110.6 L176.4,111.1 L177.7,111.6 L178.9,112.1 L180.2,112.5 L181.4,113.0 L182.7,113.5 L183.9,114.0 L185.2,114.4 L186.4,114.9 L187.7,115.4 L188.9,115.9 L190.2,116.3 L191.4,116.8 L192.7,117.3 L194.0,117.8 L195.2,118.2 L196.5,118.7 L197.7,119.2 L199.0,119.6 L200.2,120.1 L201.5,120.5 L202.7,121.0 L204.0,121.5 L205.2,121.9 L206.5,122.4 L207.7,122.9 L209.0,123.3 L210.2,123.8 L211.5,124.2 L212.7,124.7 L214.0,125.1 L215.2,125.6 L216.5,126.0 L217.7,126.5 L219.0,127.0 L220.2,127.4 L221.5,127.9 L222.7,128.3 L224.0,128.7 L225.2,129.2 L226.5,129.6 L227.8,130.1 L229.0,130.5 L230.3,131.0 L231.5,131.4 L232.8,131.9 L234.0,132.3 L235.3,132.7 L236.5,133.2 L237.8,133.6 L239.0,134.1 L240.3,134.5 L241.5,134.9 L242.8,135.4 L244.0,135.8 L245.3,136.2 L246.5,136.7 L247.8,137.1 L249.0,137.5 L250.3,138.0 L251.5,138.4 L252.8,138.8 L254.0,139.3 L255.3,139.7 L256.5,140.1 L257.8,140.5 L259.0,141.0 L260.3,141.4 L261.6,141.8 L262.8,142.2 L264.1,142.7 L265.3,143.1 L266.6,143.5 L267.8,143.9 L269.1,144.3 L270.3,144.8 L271.6,145.2 L272.8,145.6 L274.1,146.0 L275.3,146.4 L276.6,146.8 L277.8,147.3 L279.1,147.7 L280.3,148.1 L281.6,148.5 L282.8,148.9 L284.1,149.3 L285.3,149.7 L286.6,150.1 L287.8,150.6 L289.1,151.0 L290.3,151.4 L291.6,151.8 L292.8,152.2 L294.1,152.6 L295.4,153.0 L296.6,153.4 L297.9,153.8 L299.1,154.2 L300.4,154.6 L301.6,155.0 L302.9,155.4 L304.1,155.8 L305.4,156.2 L306.6,156.6 L307.9,157.0 L309.1,157.4 L310.4,157.8 L311.6,158.2 L312.9,158.6 L314.1,159.0 L315.4,159.4 L316.6,159.8 L317.9,160.2 L319.1,160.6 L320.4,160.9 L321.6,161.3 L322.9,161.7 L324.1,162.1 L325.4,162.5 L326.7,162.9 L327.9,163.3 L329.2,163.7 L330.4,164.1 L331.7,164.4 L332.9,164.8 L334.2,165.2 L335.4,165.6 L336.7,166.0 L337.9,166.4 L339.2,166.7 L340.4,167.1 L341.7,167.5 L342.9,167.9 L344.2,168.3 L345.4,168.6 L346.7,169.0 L347.9,169.4 L349.2,169.8 L350.4,170.1 L351.7,170.5 L352.9,170.9 L354.2,171.3 L355.4,171.6 L356.7,172.0 L357.9,172.4 L359.2,172.8 L360.5,173.1 L361.7,173.5 L363.0,173.9 L364.2,174.2 L365.5,174.6 L366.7,175.0 L368.0,175.4 L369.2,175.7 L370.5,176.1 L371.7,176.5 L373.0,176.8 L374.2,177.2 L375.5,177.6 L376.7,177.9 L378.0,178.3 L379.2,178.6 L380.5,179.0 L381.7,179.4 L383.0,179.7 L384.2,180.1 L385.5,180.5 L386.7,180.8 L388.0,181.2 L389.2,181.5 L390.5,181.9 L391.7,182.2 L393.0,182.6 L394.3,183.0 L395.5,183.3 L396.8,183.7 L398.0,184.0 L399.3,184.4 L400.5,184.7 L401.8,185.1 L403.0,185.4 L404.3,185.8 L405.5,186.1 L406.8,186.5 L408.0,186.8 L409.3,187.2 L410.5,187.5 L411.8,187.9 L413.0,188.2 L414.3,188.6 L415.5,188.9 L416.8,189.3 L418.0,189.6 L419.3,190.0 L420.5,190.3 L421.8,190.7 L423.0,191.0 L424.3,191.4 L425.5,191.7 L426.8,192.1 L428.1,192.4 L429.3,192.7 L430.6,193.1 L431.8,193.4 L433.1,193.8 L434.3,194.1 L435.6,194.4 L436.8,194.8 L438.1,195.1 L439.3,195.5 L440.6,195.8 L441.8,196.1 L443.1,196.5 L444.3,196.8 L445.6,197.1 L446.8,197.5 L448.1,197.8 L449.3,198.1 L450.6,198.5 L451.8,198.8 L453.1,199.1 L454.3,199.5 L455.6,199.8 L456.8,200.1 L458.1,200.5 L459.3,200.8 L460.6,201.1 L461.9,201.5 L463.1,201.8 L464.4,202.1 L465.6,202.5 L466.9,202.8 L468.1,203.1 L469.4,203.4 L470.6,203.8 L471.9,204.1 L473.1,204.4 L474.4,204.7 L475.6,205.1 L476.9,205.4 L478.1,205.7 L479.4,206.0 L480.6,206.4 L481.9,206.7 L483.1,207.0 L484.4,207.3 L485.6,207.6 L486.9,208.0 L488.1,208.3 L489.4,208.6 L490.6,208.9 L491.9,209.2 L493.2,209.6 L494.4,209.9 L495.7,210.2 L496.9,210.5 L498.2,210.8 L499.4,211.2 L500.7,211.5 L501.9,211.8 L503.2,212.1 L504.4,212.4 L505.7,212.7 L506.9,213.0 L508.2,213.4 L509.4,213.7 L510.7,214.0 L511.9,214.3 L513.2,214.6 L514.4,214.9 L515.7,215.2 L516.9,215.5 L518.2,215.9 L519.4,216.2 L520.7,216.5 L521.9,216.8 L523.2,217.1 L524.4,217.4 L525.7,217.7 L527.0,218.0 L528.2,218.3 L529.5,218.6 L530.7,218.9 L532.0,219.2 L533.2,219.6 L534.5,219.9 L535.7,220.2 L537.0,220.5 L538.2,220.8 L539.5,221.1 L540.7,221.4 L542.0,221.7 L543.2,222.0 L544.5,222.3 L545.7,222.6 L547.0,222.9 L548.2,223.2 L549.5,223.5 L550.7,223.8 L552.0,224.1 L553.2,224.4 L554.5,224.7 L555.7,225.0 L557.0,225.3 L558.2,225.6 L559.5,225.9" fill="none" class="stroke-a" stroke-width="2.4"/>
<path d="M60.0,50.4 L61.3,50.9 L62.5,51.4 L63.8,51.9 L65.0,52.5 L66.3,53.0 L67.5,53.5 L68.8,54.0 L70.0,54.5 L71.3,55.0 L72.5,55.5 L73.8,56.0 L75.0,56.5 L76.3,57.0 L77.5,57.5 L78.8,58.0 L80.0,58.5 L81.3,59.0 L82.5,59.4 L83.8,59.9 L85.0,60.4 L86.3,60.9 L87.5,61.4 L88.8,61.9 L90.0,62.4 L91.3,62.8 L92.5,63.3 L93.8,63.8 L95.1,64.3 L96.3,64.7 L97.6,65.2 L98.8,65.7 L100.1,66.2 L101.3,66.6 L102.6,67.1 L103.8,67.6 L105.1,68.0 L106.3,68.5 L107.6,69.0 L108.8,69.4 L110.1,69.9 L111.3,70.3 L112.6,70.8 L113.8,71.3 L115.1,71.7 L116.3,72.2 L117.6,72.6 L118.8,73.1 L120.1,73.5 L121.3,74.0 L122.6,74.4 L123.8,74.9 L125.1,75.3 L126.3,75.8 L127.6,76.2 L128.9,76.7 L130.1,77.1 L131.4,77.5 L132.6,78.0 L133.9,78.4 L135.1,78.8 L136.4,79.3 L137.6,79.7 L138.9,80.2 L140.1,80.6 L141.4,81.0 L142.6,81.4 L143.9,81.9 L145.1,82.3 L146.4,82.7 L147.6,83.1 L148.9,83.6 L150.1,84.0 L151.4,84.4 L152.6,84.8 L153.9,85.3 L155.1,85.7 L156.4,86.1 L157.6,86.5 L158.9,86.9 L160.2,87.3 L161.4,87.7 L162.7,88.2 L163.9,88.6 L165.2,89.0 L166.4,89.4 L167.7,89.8 L168.9,90.2 L170.2,90.6 L171.4,91.0 L172.7,91.4 L173.9,91.8 L175.2,92.2 L176.4,92.6 L177.7,93.0 L178.9,93.4 L180.2,93.8 L181.4,94.2 L182.7,94.6 L183.9,95.0 L185.2,95.4 L186.4,95.8 L187.7,96.1 L188.9,96.5 L190.2,96.9 L191.4,97.3 L192.7,97.7 L194.0,98.1 L195.2,98.5 L196.5,98.8 L197.7,99.2 L199.0,99.6 L200.2,100.0 L201.5,100.4 L202.7,100.7 L204.0,101.1 L205.2,101.5 L206.5,101.9 L207.7,102.2 L209.0,102.6 L210.2,103.0 L211.5,103.3 L212.7,103.7 L214.0,104.1 L215.2,104.5 L216.5,104.8 L217.7,105.2 L219.0,105.5 L220.2,105.9 L221.5,106.3 L222.7,106.6 L224.0,107.0 L225.2,107.3 L226.5,107.7 L227.8,108.1 L229.0,108.4 L230.3,108.8 L231.5,109.1 L232.8,109.5 L234.0,109.8 L235.3,110.2 L236.5,110.5 L237.8,110.9 L239.0,111.2 L240.3,111.6 L241.5,111.9 L242.8,112.3 L244.0,112.6 L245.3,112.9 L246.5,113.3 L247.8,113.6 L249.0,114.0 L250.3,114.3 L251.5,114.6 L252.8,115.0 L254.0,115.3 L255.3,115.7 L256.5,116.0 L257.8,116.3 L259.0,116.6 L260.3,117.0 L261.6,117.3 L262.8,117.6 L264.1,118.0 L265.3,118.3 L266.6,118.6 L267.8,118.9 L269.1,119.3 L270.3,119.6 L271.6,119.9 L272.8,120.2 L274.1,120.6 L275.3,120.9 L276.6,121.2 L277.8,121.5 L279.1,121.8 L280.3,122.1 L281.6,122.4 L282.8,122.8 L284.1,123.1 L285.3,123.4 L286.6,123.7 L287.8,124.0 L289.1,124.3 L290.3,124.6 L291.6,124.9 L292.8,125.2 L294.1,125.5 L295.4,125.8 L296.6,126.1 L297.9,126.4 L299.1,126.7 L300.4,127.0 L301.6,127.3 L302.9,127.6 L304.1,127.9 L305.4,128.2 L306.6,128.5 L307.9,128.8 L309.1,129.1 L310.4,129.4 L311.6,129.7 L312.9,130.0 L314.1,130.3 L315.4,130.5 L316.6,130.8 L317.9,131.1 L319.1,131.4 L320.4,131.7 L321.6,132.0 L322.9,132.2 L324.1,132.5 L325.4,132.8 L326.7,133.1 L327.9,133.4 L329.2,133.6 L330.4,133.9 L331.7,134.2 L332.9,134.5 L334.2,134.7 L335.4,135.0 L336.7,135.3 L337.9,135.5 L339.2,135.8 L340.4,136.1 L341.7,136.3 L342.9,136.6 L344.2,136.9 L345.4,137.1 L346.7,137.4 L347.9,137.6 L349.2,137.9 L350.4,138.2 L351.7,138.4 L352.9,138.7 L354.2,138.9 L355.4,139.2 L356.7,139.4 L357.9,139.7 L359.2,139.9 L360.5,140.2 L361.7,140.4 L363.0,140.7 L364.2,140.9 L365.5,141.1 L366.7,141.4 L368.0,141.6 L369.2,141.9 L370.5,142.1 L371.7,142.3 L373.0,142.6 L374.2,142.8 L375.5,143.1 L376.7,143.3 L378.0,143.5 L379.2,143.7 L380.5,144.0 L381.7,144.2 L383.0,144.4 L384.2,144.6 L385.5,144.9 L386.7,145.1 L388.0,145.3 L389.2,145.5 L390.5,145.8 L391.7,146.0 L393.0,146.2 L394.3,146.4 L395.5,146.6 L396.8,146.8 L398.0,147.0 L399.3,147.2 L400.5,147.4 L401.8,147.7 L403.0,147.9 L404.3,148.1 L405.5,148.3 L406.8,148.5 L408.0,148.7 L409.3,148.8 L410.5,149.0 L411.8,149.2 L413.0,149.4 L414.3,149.6 L415.5,149.8 L416.8,150.0 L418.0,150.2 L419.3,150.4 L420.5,150.5 L421.8,150.7 L423.0,150.9 L424.3,151.1 L425.5,151.2 L426.8,151.4 L428.1,151.6 L429.3,151.8 L430.6,151.9 L431.8,152.1 L433.1,152.3 L434.3,152.4 L435.6,152.6 L436.8,152.7 L438.1,152.9 L439.3,153.0 L440.6,153.2 L441.8,153.3 L443.1,153.5 L444.3,153.6 L445.6,153.8 L446.8,153.9 L448.1,154.0 L449.3,154.2 L450.6,154.3 L451.8,154.4 L453.1,154.6 L454.3,154.7 L455.6,154.8 L456.8,154.9 L458.1,155.0 L459.3,155.2 L460.6,155.3 L461.9,155.4 L463.1,155.5 L464.4,155.6 L465.6,155.7 L466.9,155.8 L468.1,155.9 L469.4,156.0 L470.6,156.0 L471.9,156.1 L473.1,156.2 L474.4,156.3 L475.6,156.3 L476.9,156.4 L478.1,156.5 L479.4,156.5 L480.6,156.6 L481.9,156.6 L483.1,156.7 L484.4,156.7 L485.6,156.7 L486.9,156.8 L488.1,156.8 L489.4,156.8 L490.6,156.8 L491.9,156.9 L493.2,156.9 L494.4,156.9 L495.7,156.8 L496.9,156.8 L498.2,156.8 L499.4,156.8 L500.7,156.7 L501.9,156.7 L503.2,156.7 L504.4,156.6 L505.7,156.5 L506.9,156.4 L508.2,156.4 L509.4,156.3 L510.7,156.2 L511.9,156.0 L513.2,155.9 L514.4,155.8 L515.7,155.6 L516.9,155.5 L518.2,155.3 L519.4,155.1 L520.7,154.9 L521.9,154.7 L523.2,154.4 L524.4,154.2 L525.7,153.9 L527.0,153.6 L528.2,153.3 L529.5,152.9 L530.7,152.5 L532.0,152.1 L533.2,151.7 L534.5,151.2 L535.7,150.7 L537.0,150.2 L538.2,149.6 L539.5,148.9 L540.7,148.2 L542.0,147.4 L543.2,146.6 L544.5,145.6 L545.7,144.6 L547.0,143.4 L548.2,142.1 L549.5,140.6 L550.7,138.8 L552.0,136.8 L553.2,134.4 L554.5,131.4 L555.7,127.6 L557.0,122.4 L558.2,114.1 L559.5,94.6" fill="none" class="stroke-b" stroke-width="2.4"/>
<circle cx="493.3" cy="156.9" r="5.5" class="fill-b"/>
<text x="489.3" y="124.9" text-anchor="end" class="fig-bold">lowest at 0.9333</text>
<text x="400.0" y="218.7" text-anchor="end" class="fig-small">one-hot target: reaches 0 only at 1</text>
<text x="180.0" y="53.8" text-anchor="start" class="fig-small">smoothed target, ε = 0.1</text>
</svg>
Figure 1. Loss for one row against the probability given to the right class, with three classes and the two wrong classes sharing what is left equally. With a one-hot target (blue) the loss keeps falling until the probability is 1. With the smoothed target (orange) it is lowest, 0.291, at 0.9333, and rises again beyond that, so more certainty is no longer rewarded.
```

## What you already know

- Softmax turns a row of scores, called **logits**, into probabilities that sum to 1 ("Softmax and cross-entropy: a loss with a gradient you can read").
- Cross-entropy against a target is $-\sum_c y_c \log p_c$. With a one-hot target it is minus the log of the probability on the right class.
- The gradient of softmax followed by cross-entropy is the probabilities minus the target.
- Subtracting the largest score from every score before exponentiating leaves the softmax unchanged and stops large scores from overflowing.
- Overfitting means doing better on the training set than on new data. Overconfidence is a form of it.

## Why certainty is expensive

Suppose the scores for a row are $[g, 0, 0]$, so the right class leads each wrong class by a gap $g$. Its probability is $e^g / (e^g + 2)$. Now suppose the label on some example is wrong, and the true class is the second one. The loss is $\ln(e^g + 2)$, which is close to $g$ once $g$ is large. At a gap of 8 that costs 8.0, and at a gap of 20 it costs 20.0. A model trained toward ever larger gaps pays a large price on every example it gets wrong, and mislabelled training examples make it pay on the training set too.

You might ask why not cap the scores at some maximum. A cap fights the optimiser: the gradient still says "go higher", and the cap throws that push away. The loss at the cap is not even the lowest the model could reach. A better fix moves the target itself, so the gradient falls to zero by its own accord at a sensible gap.

## Softening the target

Take scores $z = [2, 0, 0]$, right class 1, $K = 3$ classes, and a smoothing amount $\epsilon = 0.1$. Softmax uses $e^2 = 7.389$, so the probabilities are $[7.389, 1, 1] / 9.389 = [0.7870, 0.1065, 0.1065]$.

The ordinary target is $[1, 0, 0]$. **Label smoothing** takes a share $\epsilon$ of the target and spreads it evenly over all $K$ classes. The right class keeps $1 - \epsilon$ plus its even share $\epsilon / K$, so it gets $0.9 + 0.0333 = 0.9333$. Each wrong class gets $\epsilon/K = 0.0333$. The new target is $[0.9333, 0.0333, 0.0333]$, and Figure 2 shows it beside the other two distributions.

```figure
<svg viewBox="0 0 600 282" role="img" aria-label="Grouped bars for three classes. The one-hot target is 1, 0, 0. The smoothed target is 0.933, 0.033, 0.033. The softmax output for scores 2, 0, 0 is 0.787, 0.107, 0.107.">
<line x1="60.0" y1="206.0" x2="560.0" y2="206.0" class="stroke-muted"/>
<line x1="60.0" y1="206.0" x2="60.0" y2="36.0" class="stroke-muted"/>
<line x1="56.0" y1="206.0" x2="60.0" y2="206.0" class="stroke-muted"/>
<text x="53.0" y="210.0" text-anchor="end" class="fig-small">0</text>
<line x1="56.0" y1="130.1" x2="60.0" y2="130.1" class="stroke-muted"/>
<text x="53.0" y="134.1" text-anchor="end" class="fig-small">0.5</text>
<line x1="56.0" y1="54.2" x2="60.0" y2="54.2" class="stroke-muted"/>
<text x="53.0" y="58.2" text-anchor="end" class="fig-small">1</text>
<text x="143.3" y="224.0" text-anchor="middle" class="fig-small">class 1</text>
<rect x="83.3" y="54.2" width="37.0" height="151.8" class="fill-soft-a stroke-a"/>
<text x="101.8" y="49.2" text-anchor="middle" class="fig-small">1.000</text>
<rect x="123.3" y="64.3" width="37.0" height="141.7" class="fill-soft-c stroke-c"/>
<text x="141.8" y="59.3" text-anchor="middle" class="fig-small">0.933</text>
<rect x="163.3" y="86.5" width="37.0" height="119.5" class="fill-soft-b stroke-b"/>
<text x="181.8" y="81.5" text-anchor="middle" class="fig-small">0.787</text>
<text x="310.0" y="224.0" text-anchor="middle" class="fig-small">class 2</text>
<rect x="250.0" y="206.0" width="37.0" height="0.0" class="fill-soft-a stroke-a"/>
<text x="268.5" y="201.0" text-anchor="middle" class="fig-small">0.000</text>
<rect x="290.0" y="200.9" width="37.0" height="5.1" class="fill-soft-c stroke-c"/>
<text x="308.5" y="195.9" text-anchor="middle" class="fig-small">0.033</text>
<rect x="330.0" y="189.8" width="37.0" height="16.2" class="fill-soft-b stroke-b"/>
<text x="348.5" y="184.8" text-anchor="middle" class="fig-small">0.107</text>
<text x="476.7" y="224.0" text-anchor="middle" class="fig-small">class 3</text>
<rect x="416.7" y="206.0" width="37.0" height="0.0" class="fill-soft-a stroke-a"/>
<text x="435.2" y="201.0" text-anchor="middle" class="fig-small">0.000</text>
<rect x="456.7" y="200.9" width="37.0" height="5.1" class="fill-soft-c stroke-c"/>
<text x="475.2" y="195.9" text-anchor="middle" class="fig-small">0.033</text>
<rect x="496.7" y="189.8" width="37.0" height="16.2" class="fill-soft-b stroke-b"/>
<text x="515.2" y="184.8" text-anchor="middle" class="fig-small">0.107</text>
<rect x="330" y="238" width="0" height="0" class="fill-soft-a stroke-a"/>
<rect x="330" y="238" width="0" height="0" class="fill-soft-c stroke-c"/>
<rect x="330" y="238" width="0" height="0" class="fill-soft-b stroke-b"/>
<rect x="60" y="248" width="14" height="14" class="fill-soft-a stroke-a"/>
<text x="80.0" y="260.0" text-anchor="start" class="fig-small">one-hot target</text>
<rect x="215" y="248" width="14" height="14" class="fill-soft-c stroke-c"/>
<text x="235.0" y="260.0" text-anchor="start" class="fig-small">smoothed target</text>
<rect x="370" y="248" width="14" height="14" class="fill-soft-b stroke-b"/>
<text x="390.0" y="260.0" text-anchor="start" class="fig-small">softmax of [2, 0, 0]</text>
</svg>
Figure 2. Three distributions over three classes. The smoothed target [0.933, 0.033, 0.033] is the one-hot target pulled slightly toward uniform. The softmax output for scores [2, 0, 0] is [0.787, 0.107, 0.107], and training moves it toward the smoothed target, not the one-hot one.
```

$$
y^{\text{smooth}} = (1 - \epsilon)\,\text{onehot} + \frac{\epsilon}{K}
$$

Read it aloud: "scale the one-hot row down to $1 - \epsilon$, then add $\epsilon/K$ to every class." The row still sums to 1, because $(1 - \epsilon) + K \cdot \epsilon/K = 1$. In numpy, with `targets` an integer array of class indices:

```python
y_smooth = (1.0 - eps) * np.eye(K)[targets] + eps / K
```

## The loss, and why to subtract the max

The loss on one row is cross-entropy against the smoothed target, and the loss on a batch is the mean over rows:

$$
L = -\frac{1}{N} \sum_{\text{rows}} \sum_{c=1}^{K} y^{\text{smooth}}_c \log p_c
$$

Read it aloud: "for each row, weight each class's log probability by its target share, flip the sign, and then average over the rows." On the worked row, $-(0.9333 \ln 0.7870 + 0.0333 \ln 0.1065 + 0.0333 \ln 0.1065) = 0.3729$. With $\epsilon = 0$ it would be $-\ln 0.7870 = 0.2395$. The smoothed loss is higher on this row, because the model is less sure than the smoothed target asks.

You need $\log p$ for every class, and computing $p$ first and taking its log fails when $p$ underflows to 0. Compute the log directly instead. Shift the scores by their row maximum, then subtract the log of the sum of exponentials:

```python
shifted = logits - logits.max(axis=1, keepdims=True)
log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
```

The shift changes nothing in the result, because the same constant cancels in the softmax, and it keeps every exponent at most 0. Try scores $[800, 0, 0]$. Without the shift, $e^{800}$ overflows. With it the loss is finite, and it equals $2 \times 0.0333 \times 800 = 53.3$: the model is certain, and the smoothed target punishes certainty heavily. With $\epsilon = 0$ the same row costs 0.

## The gradient

The gradient with respect to the scores keeps the shape you already know from the softmax chapter:

$$
\frac{\partial L}{\partial z} = \frac{p - y^{\text{smooth}}}{N}
$$

Read it aloud: "predicted probabilities minus the smoothed target, divided by the number of rows." The reason it has this form is that, for softmax followed by cross-entropy, the gradient with respect to score $c$ is $p_c \sum_k y_k - y_c$. The target sums to 1, and the smoothed target does too, so the sum drops out and leaves $p_c - y_c$. The only change from plain cross-entropy is the target. The division by $N$ appears because the loss is a mean over rows. On the worked row, with $N = 1$, the gradient is $[-0.1463, 0.0732, 0.0732]$: it raises the right score and lowers the others, as it should.

```python
grad = (p - y_smooth) / N
```

## Where the minimum sits

The gradient is zero when $p = y^{\text{smooth}}$. Training stops pushing once the model gives the right class 0.9333 and each other class 0.0333, which is a gap of $\ln(0.9333 / 0.0333) = \ln 28 = 3.33$ between the right score and each wrong one. Figure 3 plots the loss against that gap. With the one-hot target it keeps falling. With the smoothed target it is lowest at 3.33, where it equals 0.291, and it rises again beyond that. The lowest value is not 0 because the target itself is not certain: 0.291 is the entropy of the smoothed target, the loss no model can beat, so nothing is left to gain from more confidence. At a gap of 8, the loss is 0.534. 

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Loss against the score gap. The one-hot curve falls toward zero as the gap grows. The smoothed curve reaches its lowest value of 0.291 at a gap of 3.33 and then rises, reaching 0.534 at a gap of 8.">
<defs><marker id="c076-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>
<line x1="60.0" y1="226.0" x2="560.0" y2="226.0" class="stroke-muted"/>
<line x1="60.0" y1="226.0" x2="60.0" y2="36.0" class="stroke-muted"/>
<line x1="60.0" y1="226.0" x2="60.0" y2="230.0" class="stroke-muted"/>
<text x="60.0" y="243.0" text-anchor="middle" class="fig-small">0</text>
<line x1="185.0" y1="226.0" x2="185.0" y2="230.0" class="stroke-muted"/>
<text x="185.0" y="243.0" text-anchor="middle" class="fig-small">2</text>
<line x1="310.0" y1="226.0" x2="310.0" y2="230.0" class="stroke-muted"/>
<text x="310.0" y="243.0" text-anchor="middle" class="fig-small">4</text>
<line x1="435.0" y1="226.0" x2="435.0" y2="230.0" class="stroke-muted"/>
<text x="435.0" y="243.0" text-anchor="middle" class="fig-small">6</text>
<line x1="560.0" y1="226.0" x2="560.0" y2="230.0" class="stroke-muted"/>
<text x="560.0" y="243.0" text-anchor="middle" class="fig-small">8</text>
<line x1="56.0" y1="226.0" x2="60.0" y2="226.0" class="stroke-muted"/>
<text x="53.0" y="230.0" text-anchor="end" class="fig-small">0</text>
<line x1="56.0" y1="150.0" x2="60.0" y2="150.0" class="stroke-muted"/>
<text x="53.0" y="154.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="56.0" y1="74.0" x2="60.0" y2="74.0" class="stroke-muted"/>
<text x="53.0" y="78.0" text-anchor="end" class="fig-small">1</text>
<text x="310.0" y="260.0" text-anchor="middle" class="fig-small">gap between the right score and each wrong score</text>
<text x="60.0" y="28.0" text-anchor="start" class="fig-small">loss</text>
<path d="M60.0,59.0 L61.7,61.7 L63.3,64.4 L65.0,67.0 L66.7,69.7 L68.4,72.3 L70.0,74.8 L71.7,77.4 L73.4,79.9 L75.1,82.4 L76.7,84.9 L78.4,87.3 L80.1,89.7 L81.7,92.1 L83.4,94.5 L85.1,96.9 L86.8,99.2 L88.4,101.5 L90.1,103.7 L91.8,106.0 L93.4,108.2 L95.1,110.3 L96.8,112.5 L98.5,114.6 L100.1,116.7 L101.8,118.8 L103.5,120.8 L105.2,122.8 L106.8,124.8 L108.5,126.8 L110.2,128.7 L111.8,130.6 L113.5,132.5 L115.2,134.4 L116.9,136.2 L118.5,138.0 L120.2,139.8 L121.9,141.5 L123.5,143.3 L125.2,144.9 L126.9,146.6 L128.6,148.3 L130.2,149.9 L131.9,151.5 L133.6,153.0 L135.3,154.6 L136.9,156.1 L138.6,157.6 L140.3,159.0 L141.9,160.5 L143.6,161.9 L145.3,163.3 L147.0,164.6 L148.6,166.0 L150.3,167.3 L152.0,168.6 L153.6,169.8 L155.3,171.1 L157.0,172.3 L158.7,173.5 L160.3,174.7 L162.0,175.8 L163.7,177.0 L165.4,178.1 L167.0,179.2 L168.7,180.2 L170.4,181.3 L172.0,182.3 L173.7,183.3 L175.4,184.3 L177.1,185.3 L178.7,186.2 L180.4,187.1 L182.1,188.0 L183.7,188.9 L185.4,189.8 L187.1,190.7 L188.8,191.5 L190.4,192.3 L192.1,193.1 L193.8,193.9 L195.5,194.7 L197.1,195.4 L198.8,196.1 L200.5,196.9 L202.1,197.6 L203.8,198.2 L205.5,198.9 L207.2,199.6 L208.8,200.2 L210.5,200.8 L212.2,201.5 L213.8,202.1 L215.5,202.6 L217.2,203.2 L218.9,203.8 L220.5,204.3 L222.2,204.9 L223.9,205.4 L225.6,205.9 L227.2,206.4 L228.9,206.9 L230.6,207.3 L232.2,207.8 L233.9,208.3 L235.6,208.7 L237.3,209.1 L238.9,209.6 L240.6,210.0 L242.3,210.4 L243.9,210.8 L245.6,211.2 L247.3,211.5 L249.0,211.9 L250.6,212.2 L252.3,212.6 L254.0,212.9 L255.7,213.3 L257.3,213.6 L259.0,213.9 L260.7,214.2 L262.3,214.5 L264.0,214.8 L265.7,215.1 L267.4,215.4 L269.0,215.6 L270.7,215.9 L272.4,216.2 L274.0,216.4 L275.7,216.7 L277.4,216.9 L279.1,217.1 L280.7,217.4 L282.4,217.6 L284.1,217.8 L285.8,218.0 L287.4,218.2 L289.1,218.4 L290.8,218.6 L292.4,218.8 L294.1,219.0 L295.8,219.2 L297.5,219.3 L299.1,219.5 L300.8,219.7 L302.5,219.8 L304.1,220.0 L305.8,220.2 L307.5,220.3 L309.2,220.5 L310.8,220.6 L312.5,220.7 L314.2,220.9 L315.9,221.0 L317.5,221.1 L319.2,221.3 L320.9,221.4 L322.5,221.5 L324.2,221.6 L325.9,221.7 L327.6,221.9 L329.2,222.0 L330.9,222.1 L332.6,222.2 L334.2,222.3 L335.9,222.4 L337.6,222.5 L339.3,222.6 L340.9,222.6 L342.6,222.7 L344.3,222.8 L346.0,222.9 L347.6,223.0 L349.3,223.1 L351.0,223.1 L352.6,223.2 L354.3,223.3 L356.0,223.4 L357.7,223.4 L359.3,223.5 L361.0,223.6 L362.7,223.6 L364.3,223.7 L366.0,223.7 L367.7,223.8 L369.4,223.9 L371.0,223.9 L372.7,224.0 L374.4,224.0 L376.1,224.1 L377.7,224.1 L379.4,224.2 L381.1,224.2 L382.7,224.3 L384.4,224.3 L386.1,224.4 L387.8,224.4 L389.4,224.4 L391.1,224.5 L392.8,224.5 L394.4,224.6 L396.1,224.6 L397.8,224.6 L399.5,224.7 L401.1,224.7 L402.8,224.7 L404.5,224.8 L406.2,224.8 L407.8,224.8 L409.5,224.9 L411.2,224.9 L412.8,224.9 L414.5,225.0 L416.2,225.0 L417.9,225.0 L419.5,225.0 L421.2,225.1 L422.9,225.1 L424.5,225.1 L426.2,225.1 L427.9,225.2 L429.6,225.2 L431.2,225.2 L432.9,225.2 L434.6,225.2 L436.3,225.3 L437.9,225.3 L439.6,225.3 L441.3,225.3 L442.9,225.3 L444.6,225.4 L446.3,225.4 L448.0,225.4 L449.6,225.4 L451.3,225.4 L453.0,225.4 L454.6,225.5 L456.3,225.5 L458.0,225.5 L459.7,225.5 L461.3,225.5 L463.0,225.5 L464.7,225.5 L466.4,225.5 L468.0,225.6 L469.7,225.6 L471.4,225.6 L473.0,225.6 L474.7,225.6 L476.4,225.6 L478.1,225.6 L479.7,225.6 L481.4,225.6 L483.1,225.7 L484.7,225.7 L486.4,225.7 L488.1,225.7 L489.8,225.7 L491.4,225.7 L493.1,225.7 L494.8,225.7 L496.5,225.7 L498.1,225.7 L499.8,225.7 L501.5,225.7 L503.1,225.7 L504.8,225.8 L506.5,225.8 L508.2,225.8 L509.8,225.8 L511.5,225.8 L513.2,225.8 L514.8,225.8 L516.5,225.8 L518.2,225.8 L519.9,225.8 L521.5,225.8 L523.2,225.8 L524.9,225.8 L526.6,225.8 L528.2,225.8 L529.9,225.8 L531.6,225.8 L533.2,225.8 L534.9,225.8 L536.6,225.9 L538.3,225.9 L539.9,225.9 L541.6,225.9 L543.3,225.9 L544.9,225.9 L546.6,225.9 L548.3,225.9 L550.0,225.9 L551.6,225.9 L553.3,225.9 L555.0,225.9 L556.7,225.9 L558.3,225.9 L560.0,225.9" fill="none" class="stroke-a" stroke-width="2.4"/>
<path d="M60.0,59.0 L61.7,61.4 L63.3,63.8 L65.0,66.2 L66.7,68.6 L68.4,70.9 L70.0,73.2 L71.7,75.5 L73.4,77.7 L75.1,80.0 L76.7,82.2 L78.4,84.3 L80.1,86.5 L81.7,88.6 L83.4,90.7 L85.1,92.8 L86.8,94.8 L88.4,96.8 L90.1,98.8 L91.8,100.8 L93.4,102.7 L95.1,104.6 L96.8,106.5 L98.5,108.4 L100.1,110.2 L101.8,112.0 L103.5,113.8 L105.2,115.5 L106.8,117.2 L108.5,118.9 L110.2,120.6 L111.8,122.2 L113.5,123.9 L115.2,125.4 L116.9,127.0 L118.5,128.5 L120.2,130.0 L121.9,131.5 L123.5,132.9 L125.2,134.4 L126.9,135.8 L128.6,137.1 L130.2,138.5 L131.9,139.8 L133.6,141.1 L135.3,142.4 L136.9,143.6 L138.6,144.8 L140.3,146.0 L141.9,147.2 L143.6,148.3 L145.3,149.4 L147.0,150.5 L148.6,151.6 L150.3,152.6 L152.0,153.7 L153.6,154.7 L155.3,155.6 L157.0,156.6 L158.7,157.5 L160.3,158.4 L162.0,159.3 L163.7,160.2 L165.4,161.0 L167.0,161.8 L168.7,162.6 L170.4,163.4 L172.0,164.1 L173.7,164.9 L175.4,165.6 L177.1,166.3 L178.7,167.0 L180.4,167.6 L182.1,168.3 L183.7,168.9 L185.4,169.5 L187.1,170.1 L188.8,170.6 L190.4,171.2 L192.1,171.7 L193.8,172.2 L195.5,172.7 L197.1,173.2 L198.8,173.6 L200.5,174.1 L202.1,174.5 L203.8,174.9 L205.5,175.3 L207.2,175.7 L208.8,176.1 L210.5,176.4 L212.2,176.8 L213.8,177.1 L215.5,177.4 L217.2,177.7 L218.9,178.0 L220.5,178.3 L222.2,178.6 L223.9,178.8 L225.6,179.0 L227.2,179.3 L228.9,179.5 L230.6,179.7 L232.2,179.9 L233.9,180.1 L235.6,180.2 L237.3,180.4 L238.9,180.6 L240.6,180.7 L242.3,180.8 L243.9,180.9 L245.6,181.1 L247.3,181.2 L249.0,181.3 L250.6,181.3 L252.3,181.4 L254.0,181.5 L255.7,181.5 L257.3,181.6 L259.0,181.6 L260.7,181.7 L262.3,181.7 L264.0,181.7 L265.7,181.7 L267.4,181.7 L269.0,181.7 L270.7,181.7 L272.4,181.7 L274.0,181.7 L275.7,181.7 L277.4,181.7 L279.1,181.6 L280.7,181.6 L282.4,181.5 L284.1,181.5 L285.8,181.4 L287.4,181.3 L289.1,181.3 L290.8,181.2 L292.4,181.1 L294.1,181.0 L295.8,180.9 L297.5,180.8 L299.1,180.7 L300.8,180.6 L302.5,180.5 L304.1,180.4 L305.8,180.3 L307.5,180.2 L309.2,180.1 L310.8,179.9 L312.5,179.8 L314.2,179.7 L315.9,179.5 L317.5,179.4 L319.2,179.2 L320.9,179.1 L322.5,178.9 L324.2,178.8 L325.9,178.6 L327.6,178.5 L329.2,178.3 L330.9,178.1 L332.6,178.0 L334.2,177.8 L335.9,177.6 L337.6,177.5 L339.3,177.3 L340.9,177.1 L342.6,176.9 L344.3,176.7 L346.0,176.5 L347.6,176.3 L349.3,176.2 L351.0,176.0 L352.6,175.8 L354.3,175.6 L356.0,175.4 L357.7,175.2 L359.3,175.0 L361.0,174.8 L362.7,174.5 L364.3,174.3 L366.0,174.1 L367.7,173.9 L369.4,173.7 L371.0,173.5 L372.7,173.3 L374.4,173.1 L376.1,172.8 L377.7,172.6 L379.4,172.4 L381.1,172.2 L382.7,171.9 L384.4,171.7 L386.1,171.5 L387.8,171.3 L389.4,171.0 L391.1,170.8 L392.8,170.6 L394.4,170.3 L396.1,170.1 L397.8,169.9 L399.5,169.6 L401.1,169.4 L402.8,169.2 L404.5,168.9 L406.2,168.7 L407.8,168.4 L409.5,168.2 L411.2,168.0 L412.8,167.7 L414.5,167.5 L416.2,167.2 L417.9,167.0 L419.5,166.7 L421.2,166.5 L422.9,166.3 L424.5,166.0 L426.2,165.8 L427.9,165.5 L429.6,165.3 L431.2,165.0 L432.9,164.8 L434.6,164.5 L436.3,164.3 L437.9,164.0 L439.6,163.8 L441.3,163.5 L442.9,163.2 L444.6,163.0 L446.3,162.7 L448.0,162.5 L449.6,162.2 L451.3,162.0 L453.0,161.7 L454.6,161.5 L456.3,161.2 L458.0,161.0 L459.7,160.7 L461.3,160.4 L463.0,160.2 L464.7,159.9 L466.4,159.7 L468.0,159.4 L469.7,159.1 L471.4,158.9 L473.0,158.6 L474.7,158.4 L476.4,158.1 L478.1,157.8 L479.7,157.6 L481.4,157.3 L483.1,157.1 L484.7,156.8 L486.4,156.5 L488.1,156.3 L489.8,156.0 L491.4,155.7 L493.1,155.5 L494.8,155.2 L496.5,155.0 L498.1,154.7 L499.8,154.4 L501.5,154.2 L503.1,153.9 L504.8,153.6 L506.5,153.4 L508.2,153.1 L509.8,152.8 L511.5,152.6 L513.2,152.3 L514.8,152.0 L516.5,151.8 L518.2,151.5 L519.9,151.2 L521.5,151.0 L523.2,150.7 L524.9,150.4 L526.6,150.2 L528.2,149.9 L529.9,149.6 L531.6,149.4 L533.2,149.1 L534.9,148.8 L536.6,148.6 L538.3,148.3 L539.9,148.0 L541.6,147.8 L543.3,147.5 L544.9,147.2 L546.6,147.0 L548.3,146.7 L550.0,146.4 L551.6,146.2 L553.3,145.9 L555.0,145.6 L556.7,145.4 L558.3,145.1 L560.0,144.8" fill="none" class="stroke-b" stroke-width="2.4"/>
<line x1="268.3" y1="226.0" x2="268.3" y2="181.7" class="stroke-muted dashed" stroke-width="1.2"/>
<circle cx="268.3" cy="181.7" r="5.5" class="fill-b"/>
<text x="276.3" y="203.2" text-anchor="start" class="fig-bold">best gap 3.33</text>
<line x1="85.0" y1="46.6" x2="178.8" y2="46.6" class="stroke-ink" stroke-width="1.8" marker-end="url(#c076-arrow)"/>
<text x="188.1" y="50.6" text-anchor="start" class="fig-small">descent widens the gap</text>
<line x1="535.0" y1="107.4" x2="441.2" y2="107.4" class="stroke-ink" stroke-width="1.8" marker-end="url(#c076-arrow)"/>
<text x="435.0" y="111.4" text-anchor="end" class="fig-small">descent narrows it</text>
<text x="560.0" y="217.9" text-anchor="end" class="fig-small">one-hot target</text>
<text x="560.0" y="136.8" text-anchor="end" class="fig-small">smoothed target</text>
</svg>
Figure 3. Loss against the gap between the right score and each wrong score, for scores [gap, 0, 0]. With a one-hot target the loss keeps falling as the gap grows. With the smoothed target it is lowest at a gap of ln 28 = 3.33, where the loss is 0.291. Past it, the gradient reverses, so the scores stop growing.
```

This is the answer to the opening question. The model is asked for 93 percent, and it is penalised for going past that. The scores stop growing, and a mislabelled example costs $\ln(28 + 2) = 3.40$ at the best gap, instead of 8.0 or more at the gaps the one-hot loss keeps driving toward.

> [!KEY]
> Replacing the one-hot target with a softened one moves the lowest point of the loss to a finite confidence, $1 - \epsilon + \epsilon/K$ for the right class. The gradient $p - y^{\text{smooth}}$ then vanishes at a finite gap of scores, so training stops asking for 100 percent.

> [!RECAP]
> Hold these points before you write code:
>
> - The smoothed target is $(1 - \epsilon)$ times the one-hot row plus $\epsilon/K$. Each row sums to 1.
> - The loss is the mean over rows of $-\sum_c y^{\text{smooth}}_c \log p_c$, computed with a max-shifted log-softmax.
> - The gradient is $(p - y^{\text{smooth}})/N$ because the target sums to 1 and the loss is a mean.
> - The loss is lowest where $p$ equals the smoothed target, a finite gap of scores.

## Build it step by step

The starter gives you `log_softmax`, the max-shifted log-softmax from above, already written. The checks call your functions in this order:

1. **`smooth_targets(targets, num_classes, eps)`.** Take integer class indices of shape `(N,)`. Return the `(N, K)` float matrix of smoothed targets. Index into `np.eye(num_classes)`. The usual mistake is to add `eps / K` before scaling, so the row no longer sums to 1.
2. **`label_smoothing_loss(logits, targets, eps)`.** Return the mean over rows as a float, using `smooth_targets` and `log_softmax`. The usual mistake is to sum over rows instead of averaging.
3. **`label_smoothing_grad(logits, targets, eps)`.** Return an `(N, K)` array. Get the probabilities from `np.exp(log_softmax(logits))`. The usual mistake is to leave out the division by `N`.

## Cost and scaling

Label smoothing costs a few operations per score and no extra memory beyond the target matrix, so it is the cheapest of the four regularisers in this part.

## Where this breaks

- The model's probabilities stop meaning what they say. A calibrated model is right about 80 percent of the time when it says 80 percent. After smoothing, the model's top probability is pulled toward $1 - \epsilon + \epsilon/K$ whatever the evidence, so its probabilities no longer track how often it is right. Skip smoothing when you need calibrated probabilities, or fix them afterwards.
- A large $\epsilon$ blurs the target so much that the classes look alike, and the model underfits. Small values such as 0.1 are common.
- Smoothing does not help when the labels are already soft or when the model is not overconfident.

## How the four tools fit together

You have now seen four ways to keep a network from hugging its training set. Weight decay and early stopping share a story: both pull weights toward zero, and the early-stopping chapter shows that stopping after $k$ steps acts as a ridge penalty on each direction. Dropout and label smoothing tell different stories. Dropout injects noise into the units, so none can rely on its partners. Label smoothing changes the target, so the loss is lowest at a finite confidence. They are not four versions of one idea, so you can combine them, and you choose each strength on held-out data.

## The exercise

You implement `smooth_targets`, `label_smoothing_loss`, and `label_smoothing_grad`.

The checks verify the smoothed target on the worked example and that every row sums to 1. They check the loss on the worked row, that it is a mean over rows, and that it stays finite for scores of 800. They compare the gradient with a finite difference of your loss and check that it vanishes where the probabilities equal the smoothed target. The last check scans the loss over score gaps and confirms that its lowest point is near $\ln 28$.
