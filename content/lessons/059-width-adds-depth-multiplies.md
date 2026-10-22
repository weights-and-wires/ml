---
title: Width adds bends and depth multiplies them
slug: 059-width-adds-depth-multiplies
part: "Part 4 — Neural networks"
summary: One ReLU unit adds one bend, so width adds pieces one at a time, while stacked layers fold the curve again and again, so pieces double.
minutes: 45
gpu: false
objectives:
  - Explain a ReLU unit as a hinge and say how many straight pieces m units can make.
  - Build the tent map from two ReLU units and show that each extra layer doubles the number of pieces.
  - Say why one hidden layer needs about 2^k units to match a depth-k network.
  - Count the straight pieces of any one-input ReLU network exactly, by tracking a slope and an intercept per interval.
prereqs:
  - 057-why-layers-need-a-bend
  - 058-forward-pass-and-gelu
exercise: 059-width-adds-depth-multiplies
---

# Width adds bends and depth multiplies them

Suppose you want to fit a signal with many bends, such as the load on a power grid that peaks eight times a day. A hidden layer of ReLU units can bend a curve once per unit. How could a network with only a handful of units make hundreds of bends, and how would you count them to check?

If every bend costs one unit, a curve with 256 bends costs about 256 units, and the network grows as fast as the signal wiggles. That is the wide-layer route. This chapter shows a cheaper route: stack layers so each one folds what the last one made, and the bends multiply. You will also learn to count the pieces exactly, so the claim is something you can test.

## What you already know

- A unit is a weighted sum and a constant, followed by a bend, and a forward pass runs the layers in order ("Run a network forward, with a sharp or smooth bend").
- ReLU keeps positive numbers and cuts negative ones to 0.
- Two affine layers with no bend between them merge into one ("Why a stack of layers needs a bend").
- Width is the number of units in a layer, and depth is the number of layers.
- A slope and an intercept describe a straight line, $y = \text{slope}\cdot x + \text{intercept}$.

## A ReLU unit is a hinge

Take one unit that sees a single input $x$, with constant $-t$. Its output $\operatorname{relu}(x - t)$ is flat at 0 until $x$ reaches $t$, then rises at slope 1: a hinge that bends at $t$. Scale it by a weight and the rise changes slope. Add several hinges that bend at different places and the slopes add up, so the sum is a chain of straight pieces joined at the bends, like a wire bent at several joints.

```figure
<svg viewBox="0 0 600 272" role="img" aria-label="Two panels showing the curve sine of pi x on 0 to 1 in blue and a polyline of orange straight pieces approximating it, built from 4 hinge units on the left and 8 on the right, with the largest gap 0.070 and 0.019.">

<line x1="40.0" y1="193.3" x2="270.0" y2="193.3" class="stroke-muted"/>
<path d="M40.0,193.3 L40.6,192.2 L41.2,191.1 L41.7,190.1 L42.3,189.0 L42.9,188.0 L43.5,186.9 L44.0,185.8 L44.6,184.8 L45.2,183.7 L45.8,182.7 L46.3,181.6 L46.9,180.5 L47.5,179.5 L48.1,178.4 L48.6,177.4 L49.2,176.3 L49.8,175.3 L50.4,174.2 L51.0,173.2 L51.5,172.1 L52.1,171.1 L52.7,170.0 L53.3,169.0 L53.8,167.9 L54.4,166.9 L55.0,165.9 L55.6,164.8 L56.1,163.8 L56.7,162.8 L57.3,161.7 L57.9,160.7 L58.4,159.7 L59.0,158.6 L59.6,157.6 L60.2,156.6 L60.8,155.6 L61.3,154.5 L61.9,153.5 L62.5,152.5 L63.1,151.5 L63.6,150.5 L64.2,149.5 L64.8,148.5 L65.4,147.5 L65.9,146.5 L66.5,145.5 L67.1,144.5 L67.7,143.5 L68.2,142.5 L68.8,141.6 L69.4,140.6 L70.0,139.6 L70.6,138.6 L71.1,137.7 L71.7,136.7 L72.3,135.7 L72.9,134.8 L73.4,133.8 L74.0,132.9 L74.6,131.9 L75.2,131.0 L75.7,130.0 L76.3,129.1 L76.9,128.2 L77.5,127.3 L78.0,126.3 L78.6,125.4 L79.2,124.5 L79.8,123.6 L80.4,122.7 L80.9,121.8 L81.5,120.9 L82.1,120.0 L82.7,119.1 L83.2,118.2 L83.8,117.3 L84.4,116.5 L85.0,115.6 L85.5,114.7 L86.1,113.9 L86.7,113.0 L87.3,112.2 L87.8,111.3 L88.4,110.5 L89.0,109.6 L89.6,108.8 L90.2,108.0 L90.7,107.2 L91.3,106.4 L91.9,105.5 L92.5,104.7 L93.0,103.9 L93.6,103.2 L94.2,102.4 L94.8,101.6 L95.3,100.8 L95.9,100.0 L96.5,99.3 L97.1,98.5 L97.6,97.8 L98.2,97.0 L98.8,96.3 L99.4,95.5 L99.9,94.8 L100.5,94.1 L101.1,93.4 L101.7,92.7 L102.3,92.0 L102.8,91.3 L103.4,90.6 L104.0,89.9 L104.6,89.2 L105.1,88.5 L105.7,87.9 L106.3,87.2 L106.9,86.6 L107.4,85.9 L108.0,85.3 L108.6,84.7 L109.2,84.0 L109.7,83.4 L110.3,82.8 L110.9,82.2 L111.5,81.6 L112.1,81.0 L112.6,80.4 L113.2,79.8 L113.8,79.3 L114.4,78.7 L114.9,78.2 L115.5,77.6 L116.1,77.1 L116.7,76.5 L117.2,76.0 L117.8,75.5 L118.4,75.0 L119.0,74.5 L119.5,74.0 L120.1,73.5 L120.7,73.0 L121.3,72.5 L121.9,72.1 L122.4,71.6 L123.0,71.1 L123.6,70.7 L124.2,70.3 L124.7,69.8 L125.3,69.4 L125.9,69.0 L126.5,68.6 L127.0,68.2 L127.6,67.8 L128.2,67.4 L128.8,67.0 L129.3,66.7 L129.9,66.3 L130.5,66.0 L131.1,65.6 L131.7,65.3 L132.2,64.9 L132.8,64.6 L133.4,64.3 L134.0,64.0 L134.5,63.7 L135.1,63.4 L135.7,63.1 L136.3,62.9 L136.8,62.6 L137.4,62.3 L138.0,62.1 L138.6,61.9 L139.1,61.6 L139.7,61.4 L140.3,61.2 L140.9,61.0 L141.5,60.8 L142.0,60.6 L142.6,60.4 L143.2,60.2 L143.8,60.1 L144.3,59.9 L144.9,59.8 L145.5,59.6 L146.1,59.5 L146.6,59.4 L147.2,59.2 L147.8,59.1 L148.4,59.0 L148.9,58.9 L149.5,58.9 L150.1,58.8 L150.7,58.7 L151.3,58.7 L151.8,58.6 L152.4,58.6 L153.0,58.5 L153.6,58.5 L154.1,58.5 L154.7,58.5 L155.3,58.5 L155.9,58.5 L156.4,58.5 L157.0,58.5 L157.6,58.6 L158.2,58.6 L158.7,58.7 L159.3,58.7 L159.9,58.8 L160.5,58.9 L161.1,58.9 L161.6,59.0 L162.2,59.1 L162.8,59.2 L163.4,59.4 L163.9,59.5 L164.5,59.6 L165.1,59.8 L165.7,59.9 L166.2,60.1 L166.8,60.2 L167.4,60.4 L168.0,60.6 L168.5,60.8 L169.1,61.0 L169.7,61.2 L170.3,61.4 L170.9,61.6 L171.4,61.9 L172.0,62.1 L172.6,62.3 L173.2,62.6 L173.7,62.9 L174.3,63.1 L174.9,63.4 L175.5,63.7 L176.0,64.0 L176.6,64.3 L177.2,64.6 L177.8,64.9 L178.3,65.3 L178.9,65.6 L179.5,66.0 L180.1,66.3 L180.7,66.7 L181.2,67.0 L181.8,67.4 L182.4,67.8 L183.0,68.2 L183.5,68.6 L184.1,69.0 L184.7,69.4 L185.3,69.8 L185.8,70.3 L186.4,70.7 L187.0,71.1 L187.6,71.6 L188.1,72.1 L188.7,72.5 L189.3,73.0 L189.9,73.5 L190.5,74.0 L191.0,74.5 L191.6,75.0 L192.2,75.5 L192.8,76.0 L193.3,76.5 L193.9,77.1 L194.5,77.6 L195.1,78.2 L195.6,78.7 L196.2,79.3 L196.8,79.8 L197.4,80.4 L197.9,81.0 L198.5,81.6 L199.1,82.2 L199.7,82.8 L200.3,83.4 L200.8,84.0 L201.4,84.7 L202.0,85.3 L202.6,85.9 L203.1,86.6 L203.7,87.2 L204.3,87.9 L204.9,88.5 L205.4,89.2 L206.0,89.9 L206.6,90.6 L207.2,91.3 L207.7,92.0 L208.3,92.7 L208.9,93.4 L209.5,94.1 L210.1,94.8 L210.6,95.5 L211.2,96.3 L211.8,97.0 L212.4,97.8 L212.9,98.5 L213.5,99.3 L214.1,100.0 L214.7,100.8 L215.2,101.6 L215.8,102.4 L216.4,103.2 L217.0,103.9 L217.5,104.7 L218.1,105.5 L218.7,106.4 L219.3,107.2 L219.8,108.0 L220.4,108.8 L221.0,109.6 L221.6,110.5 L222.2,111.3 L222.7,112.2 L223.3,113.0 L223.9,113.9 L224.5,114.7 L225.0,115.6 L225.6,116.5 L226.2,117.3 L226.8,118.2 L227.3,119.1 L227.9,120.0 L228.5,120.9 L229.1,121.8 L229.6,122.7 L230.2,123.6 L230.8,124.5 L231.4,125.4 L232.0,126.3 L232.5,127.3 L233.1,128.2 L233.7,129.1 L234.3,130.0 L234.8,131.0 L235.4,131.9 L236.0,132.9 L236.6,133.8 L237.1,134.8 L237.7,135.7 L238.3,136.7 L238.9,137.7 L239.4,138.6 L240.0,139.6 L240.6,140.6 L241.2,141.6 L241.8,142.5 L242.3,143.5 L242.9,144.5 L243.5,145.5 L244.1,146.5 L244.6,147.5 L245.2,148.5 L245.8,149.5 L246.4,150.5 L246.9,151.5 L247.5,152.5 L248.1,153.5 L248.7,154.5 L249.2,155.6 L249.8,156.6 L250.4,157.6 L251.0,158.6 L251.6,159.7 L252.1,160.7 L252.7,161.7 L253.3,162.8 L253.9,163.8 L254.4,164.8 L255.0,165.9 L255.6,166.9 L256.2,167.9 L256.7,169.0 L257.3,170.0 L257.9,171.1 L258.5,172.1 L259.0,173.2 L259.6,174.2 L260.2,175.3 L260.8,176.3 L261.4,177.4 L261.9,178.4 L262.5,179.5 L263.1,180.5 L263.7,181.6 L264.2,182.7 L264.8,183.7 L265.4,184.8 L266.0,185.8 L266.5,186.9 L267.1,188.0 L267.7,189.0 L268.3,190.1 L268.8,191.1 L269.4,192.2 L270.0,193.3" fill="none" class="stroke-a" stroke-width="2"/>
<path d="M40.0,193.3 L40.6,192.3 L41.2,191.3 L41.7,190.4 L42.3,189.4 L42.9,188.5 L43.5,187.5 L44.0,186.6 L44.6,185.6 L45.2,184.7 L45.8,183.7 L46.3,182.8 L46.9,181.8 L47.5,180.8 L48.1,179.9 L48.6,178.9 L49.2,178.0 L49.8,177.0 L50.4,176.1 L51.0,175.1 L51.5,174.2 L52.1,173.2 L52.7,172.2 L53.3,171.3 L53.8,170.3 L54.4,169.4 L55.0,168.4 L55.6,167.5 L56.1,166.5 L56.7,165.6 L57.3,164.6 L57.9,163.6 L58.4,162.7 L59.0,161.7 L59.6,160.8 L60.2,159.8 L60.8,158.9 L61.3,157.9 L61.9,157.0 L62.5,156.0 L63.1,155.0 L63.6,154.1 L64.2,153.1 L64.8,152.2 L65.4,151.2 L65.9,150.3 L66.5,149.3 L67.1,148.4 L67.7,147.4 L68.2,146.4 L68.8,145.5 L69.4,144.5 L70.0,143.6 L70.6,142.6 L71.1,141.7 L71.7,140.7 L72.3,139.8 L72.9,138.8 L73.4,137.8 L74.0,136.9 L74.6,135.9 L75.2,135.0 L75.7,134.0 L76.3,133.1 L76.9,132.1 L77.5,131.2 L78.0,130.2 L78.6,129.2 L79.2,128.3 L79.8,127.3 L80.4,126.4 L80.9,125.4 L81.5,124.5 L82.1,123.5 L82.7,122.6 L83.2,121.6 L83.8,120.6 L84.4,119.7 L85.0,118.7 L85.5,117.8 L86.1,116.8 L86.7,115.9 L87.3,114.9 L87.8,114.0 L88.4,113.0 L89.0,112.0 L89.6,111.1 L90.2,110.1 L90.7,109.2 L91.3,108.2 L91.9,107.3 L92.5,106.3 L93.0,105.4 L93.6,104.4 L94.2,103.4 L94.8,102.5 L95.3,101.5 L95.9,100.6 L96.5,99.6 L97.1,98.7 L97.6,97.9 L98.2,97.5 L98.8,97.1 L99.4,96.7 L99.9,96.3 L100.5,95.9 L101.1,95.5 L101.7,95.1 L102.3,94.7 L102.8,94.3 L103.4,93.9 L104.0,93.5 L104.6,93.1 L105.1,92.7 L105.7,92.3 L106.3,91.9 L106.9,91.5 L107.4,91.1 L108.0,90.7 L108.6,90.3 L109.2,89.9 L109.7,89.5 L110.3,89.1 L110.9,88.8 L111.5,88.4 L112.1,88.0 L112.6,87.6 L113.2,87.2 L113.8,86.8 L114.4,86.4 L114.9,86.0 L115.5,85.6 L116.1,85.2 L116.7,84.8 L117.2,84.4 L117.8,84.0 L118.4,83.6 L119.0,83.2 L119.5,82.8 L120.1,82.4 L120.7,82.0 L121.3,81.6 L121.9,81.2 L122.4,80.8 L123.0,80.4 L123.6,80.0 L124.2,79.7 L124.7,79.3 L125.3,78.9 L125.9,78.5 L126.5,78.1 L127.0,77.7 L127.6,77.3 L128.2,76.9 L128.8,76.5 L129.3,76.1 L129.9,75.7 L130.5,75.3 L131.1,74.9 L131.7,74.5 L132.2,74.1 L132.8,73.7 L133.4,73.3 L134.0,72.9 L134.5,72.5 L135.1,72.1 L135.7,71.7 L136.3,71.3 L136.8,70.9 L137.4,70.5 L138.0,70.2 L138.6,69.8 L139.1,69.4 L139.7,69.0 L140.3,68.6 L140.9,68.2 L141.5,67.8 L142.0,67.4 L142.6,67.0 L143.2,66.6 L143.8,66.2 L144.3,65.8 L144.9,65.4 L145.5,65.0 L146.1,64.6 L146.6,64.2 L147.2,63.8 L147.8,63.4 L148.4,63.0 L148.9,62.6 L149.5,62.2 L150.1,61.8 L150.7,61.4 L151.3,61.1 L151.8,60.7 L152.4,60.3 L153.0,59.9 L153.6,59.5 L154.1,59.1 L154.7,58.7 L155.3,58.7 L155.9,59.1 L156.4,59.5 L157.0,59.9 L157.6,60.3 L158.2,60.7 L158.7,61.1 L159.3,61.4 L159.9,61.8 L160.5,62.2 L161.1,62.6 L161.6,63.0 L162.2,63.4 L162.8,63.8 L163.4,64.2 L163.9,64.6 L164.5,65.0 L165.1,65.4 L165.7,65.8 L166.2,66.2 L166.8,66.6 L167.4,67.0 L168.0,67.4 L168.5,67.8 L169.1,68.2 L169.7,68.6 L170.3,69.0 L170.9,69.4 L171.4,69.8 L172.0,70.2 L172.6,70.5 L173.2,70.9 L173.7,71.3 L174.3,71.7 L174.9,72.1 L175.5,72.5 L176.0,72.9 L176.6,73.3 L177.2,73.7 L177.8,74.1 L178.3,74.5 L178.9,74.9 L179.5,75.3 L180.1,75.7 L180.7,76.1 L181.2,76.5 L181.8,76.9 L182.4,77.3 L183.0,77.7 L183.5,78.1 L184.1,78.5 L184.7,78.9 L185.3,79.3 L185.8,79.7 L186.4,80.0 L187.0,80.4 L187.6,80.8 L188.1,81.2 L188.7,81.6 L189.3,82.0 L189.9,82.4 L190.5,82.8 L191.0,83.2 L191.6,83.6 L192.2,84.0 L192.8,84.4 L193.3,84.8 L193.9,85.2 L194.5,85.6 L195.1,86.0 L195.6,86.4 L196.2,86.8 L196.8,87.2 L197.4,87.6 L197.9,88.0 L198.5,88.4 L199.1,88.8 L199.7,89.1 L200.3,89.5 L200.8,89.9 L201.4,90.3 L202.0,90.7 L202.6,91.1 L203.1,91.5 L203.7,91.9 L204.3,92.3 L204.9,92.7 L205.4,93.1 L206.0,93.5 L206.6,93.9 L207.2,94.3 L207.7,94.7 L208.3,95.1 L208.9,95.5 L209.5,95.9 L210.1,96.3 L210.6,96.7 L211.2,97.1 L211.8,97.5 L212.4,97.9 L212.9,98.7 L213.5,99.6 L214.1,100.6 L214.7,101.5 L215.2,102.5 L215.8,103.4 L216.4,104.4 L217.0,105.4 L217.5,106.3 L218.1,107.3 L218.7,108.2 L219.3,109.2 L219.8,110.1 L220.4,111.1 L221.0,112.0 L221.6,113.0 L222.2,114.0 L222.7,114.9 L223.3,115.9 L223.9,116.8 L224.5,117.8 L225.0,118.7 L225.6,119.7 L226.2,120.6 L226.8,121.6 L227.3,122.6 L227.9,123.5 L228.5,124.5 L229.1,125.4 L229.6,126.4 L230.2,127.3 L230.8,128.3 L231.4,129.2 L232.0,130.2 L232.5,131.2 L233.1,132.1 L233.7,133.1 L234.3,134.0 L234.8,135.0 L235.4,135.9 L236.0,136.9 L236.6,137.8 L237.1,138.8 L237.7,139.8 L238.3,140.7 L238.9,141.7 L239.4,142.6 L240.0,143.6 L240.6,144.5 L241.2,145.5 L241.8,146.4 L242.3,147.4 L242.9,148.4 L243.5,149.3 L244.1,150.3 L244.6,151.2 L245.2,152.2 L245.8,153.1 L246.4,154.1 L246.9,155.0 L247.5,156.0 L248.1,157.0 L248.7,157.9 L249.2,158.9 L249.8,159.8 L250.4,160.8 L251.0,161.7 L251.6,162.7 L252.1,163.6 L252.7,164.6 L253.3,165.6 L253.9,166.5 L254.4,167.5 L255.0,168.4 L255.6,169.4 L256.2,170.3 L256.7,171.3 L257.3,172.2 L257.9,173.2 L258.5,174.2 L259.0,175.1 L259.6,176.1 L260.2,177.0 L260.8,178.0 L261.4,178.9 L261.9,179.9 L262.5,180.8 L263.1,181.8 L263.7,182.8 L264.2,183.7 L264.8,184.7 L265.4,185.6 L266.0,186.6 L266.5,187.5 L267.1,188.5 L267.7,189.4 L268.3,190.4 L268.8,191.3 L269.4,192.3 L270.0,193.3" fill="none" class="stroke-b" stroke-width="2.2"/>
<circle cx="97.5" cy="98.0" r="3.5" class="fill-b"/>
<circle cx="155.0" cy="58.5" r="3.5" class="fill-b"/>
<circle cx="212.5" cy="98.0" r="3.5" class="fill-b"/>
<text x="155.0" y="22.0" text-anchor="middle" class="fig-bold">4 units, 4 pieces</text>
<text x="155.0" y="236.0" text-anchor="middle" class="fig-small">largest gap 0.070</text>
<text x="40.0" y="207.3" text-anchor="middle" class="fig-small">0</text>
<text x="270.0" y="207.3" text-anchor="middle" class="fig-small">1</text>
<line x1="330.0" y1="193.3" x2="560.0" y2="193.3" class="stroke-muted"/>
<path d="M330.0,193.3 L330.6,192.2 L331.2,191.1 L331.7,190.1 L332.3,189.0 L332.9,188.0 L333.5,186.9 L334.0,185.8 L334.6,184.8 L335.2,183.7 L335.8,182.7 L336.3,181.6 L336.9,180.5 L337.5,179.5 L338.1,178.4 L338.6,177.4 L339.2,176.3 L339.8,175.3 L340.4,174.2 L341.0,173.2 L341.5,172.1 L342.1,171.1 L342.7,170.0 L343.3,169.0 L343.8,167.9 L344.4,166.9 L345.0,165.9 L345.6,164.8 L346.1,163.8 L346.7,162.8 L347.3,161.7 L347.9,160.7 L348.4,159.7 L349.0,158.6 L349.6,157.6 L350.2,156.6 L350.8,155.6 L351.3,154.5 L351.9,153.5 L352.5,152.5 L353.1,151.5 L353.6,150.5 L354.2,149.5 L354.8,148.5 L355.4,147.5 L355.9,146.5 L356.5,145.5 L357.1,144.5 L357.7,143.5 L358.2,142.5 L358.8,141.6 L359.4,140.6 L360.0,139.6 L360.6,138.6 L361.1,137.7 L361.7,136.7 L362.3,135.7 L362.9,134.8 L363.4,133.8 L364.0,132.9 L364.6,131.9 L365.2,131.0 L365.7,130.0 L366.3,129.1 L366.9,128.2 L367.5,127.3 L368.0,126.3 L368.6,125.4 L369.2,124.5 L369.8,123.6 L370.4,122.7 L370.9,121.8 L371.5,120.9 L372.1,120.0 L372.7,119.1 L373.2,118.2 L373.8,117.3 L374.4,116.5 L375.0,115.6 L375.5,114.7 L376.1,113.9 L376.7,113.0 L377.3,112.2 L377.8,111.3 L378.4,110.5 L379.0,109.6 L379.6,108.8 L380.2,108.0 L380.7,107.2 L381.3,106.4 L381.9,105.5 L382.5,104.7 L383.0,103.9 L383.6,103.2 L384.2,102.4 L384.8,101.6 L385.3,100.8 L385.9,100.0 L386.5,99.3 L387.1,98.5 L387.6,97.8 L388.2,97.0 L388.8,96.3 L389.4,95.5 L389.9,94.8 L390.5,94.1 L391.1,93.4 L391.7,92.7 L392.3,92.0 L392.8,91.3 L393.4,90.6 L394.0,89.9 L394.6,89.2 L395.1,88.5 L395.7,87.9 L396.3,87.2 L396.9,86.6 L397.4,85.9 L398.0,85.3 L398.6,84.7 L399.2,84.0 L399.7,83.4 L400.3,82.8 L400.9,82.2 L401.5,81.6 L402.1,81.0 L402.6,80.4 L403.2,79.8 L403.8,79.3 L404.4,78.7 L404.9,78.2 L405.5,77.6 L406.1,77.1 L406.7,76.5 L407.2,76.0 L407.8,75.5 L408.4,75.0 L409.0,74.5 L409.5,74.0 L410.1,73.5 L410.7,73.0 L411.3,72.5 L411.9,72.1 L412.4,71.6 L413.0,71.1 L413.6,70.7 L414.2,70.3 L414.7,69.8 L415.3,69.4 L415.9,69.0 L416.5,68.6 L417.0,68.2 L417.6,67.8 L418.2,67.4 L418.8,67.0 L419.3,66.7 L419.9,66.3 L420.5,66.0 L421.1,65.6 L421.7,65.3 L422.2,64.9 L422.8,64.6 L423.4,64.3 L424.0,64.0 L424.5,63.7 L425.1,63.4 L425.7,63.1 L426.3,62.9 L426.8,62.6 L427.4,62.3 L428.0,62.1 L428.6,61.9 L429.1,61.6 L429.7,61.4 L430.3,61.2 L430.9,61.0 L431.5,60.8 L432.0,60.6 L432.6,60.4 L433.2,60.2 L433.8,60.1 L434.3,59.9 L434.9,59.8 L435.5,59.6 L436.1,59.5 L436.6,59.4 L437.2,59.2 L437.8,59.1 L438.4,59.0 L438.9,58.9 L439.5,58.9 L440.1,58.8 L440.7,58.7 L441.3,58.7 L441.8,58.6 L442.4,58.6 L443.0,58.5 L443.6,58.5 L444.1,58.5 L444.7,58.5 L445.3,58.5 L445.9,58.5 L446.4,58.5 L447.0,58.5 L447.6,58.6 L448.2,58.6 L448.7,58.7 L449.3,58.7 L449.9,58.8 L450.5,58.9 L451.1,58.9 L451.6,59.0 L452.2,59.1 L452.8,59.2 L453.4,59.4 L453.9,59.5 L454.5,59.6 L455.1,59.8 L455.7,59.9 L456.2,60.1 L456.8,60.2 L457.4,60.4 L458.0,60.6 L458.5,60.8 L459.1,61.0 L459.7,61.2 L460.3,61.4 L460.9,61.6 L461.4,61.9 L462.0,62.1 L462.6,62.3 L463.2,62.6 L463.7,62.9 L464.3,63.1 L464.9,63.4 L465.5,63.7 L466.0,64.0 L466.6,64.3 L467.2,64.6 L467.8,64.9 L468.3,65.3 L468.9,65.6 L469.5,66.0 L470.1,66.3 L470.7,66.7 L471.2,67.0 L471.8,67.4 L472.4,67.8 L473.0,68.2 L473.5,68.6 L474.1,69.0 L474.7,69.4 L475.3,69.8 L475.8,70.3 L476.4,70.7 L477.0,71.1 L477.6,71.6 L478.1,72.1 L478.7,72.5 L479.3,73.0 L479.9,73.5 L480.5,74.0 L481.0,74.5 L481.6,75.0 L482.2,75.5 L482.8,76.0 L483.3,76.5 L483.9,77.1 L484.5,77.6 L485.1,78.2 L485.6,78.7 L486.2,79.3 L486.8,79.8 L487.4,80.4 L487.9,81.0 L488.5,81.6 L489.1,82.2 L489.7,82.8 L490.3,83.4 L490.8,84.0 L491.4,84.7 L492.0,85.3 L492.6,85.9 L493.1,86.6 L493.7,87.2 L494.3,87.9 L494.9,88.5 L495.4,89.2 L496.0,89.9 L496.6,90.6 L497.2,91.3 L497.7,92.0 L498.3,92.7 L498.9,93.4 L499.5,94.1 L500.1,94.8 L500.6,95.5 L501.2,96.3 L501.8,97.0 L502.4,97.8 L502.9,98.5 L503.5,99.3 L504.1,100.0 L504.7,100.8 L505.2,101.6 L505.8,102.4 L506.4,103.2 L507.0,103.9 L507.5,104.7 L508.1,105.5 L508.7,106.4 L509.3,107.2 L509.8,108.0 L510.4,108.8 L511.0,109.6 L511.6,110.5 L512.2,111.3 L512.7,112.2 L513.3,113.0 L513.9,113.9 L514.5,114.7 L515.0,115.6 L515.6,116.5 L516.2,117.3 L516.8,118.2 L517.3,119.1 L517.9,120.0 L518.5,120.9 L519.1,121.8 L519.6,122.7 L520.2,123.6 L520.8,124.5 L521.4,125.4 L522.0,126.3 L522.5,127.3 L523.1,128.2 L523.7,129.1 L524.3,130.0 L524.8,131.0 L525.4,131.9 L526.0,132.9 L526.6,133.8 L527.1,134.8 L527.7,135.7 L528.3,136.7 L528.9,137.7 L529.4,138.6 L530.0,139.6 L530.6,140.6 L531.2,141.6 L531.8,142.5 L532.3,143.5 L532.9,144.5 L533.5,145.5 L534.1,146.5 L534.6,147.5 L535.2,148.5 L535.8,149.5 L536.4,150.5 L536.9,151.5 L537.5,152.5 L538.1,153.5 L538.7,154.5 L539.2,155.6 L539.8,156.6 L540.4,157.6 L541.0,158.6 L541.6,159.7 L542.1,160.7 L542.7,161.7 L543.3,162.8 L543.9,163.8 L544.4,164.8 L545.0,165.9 L545.6,166.9 L546.2,167.9 L546.7,169.0 L547.3,170.0 L547.9,171.1 L548.5,172.1 L549.0,173.2 L549.6,174.2 L550.2,175.3 L550.8,176.3 L551.4,177.4 L551.9,178.4 L552.5,179.5 L553.1,180.5 L553.7,181.6 L554.2,182.7 L554.8,183.7 L555.4,184.8 L556.0,185.8 L556.5,186.9 L557.1,188.0 L557.7,189.0 L558.3,190.1 L558.8,191.1 L559.4,192.2 L560.0,193.3" fill="none" class="stroke-a" stroke-width="2"/>
<path d="M330.0,193.3 L330.6,192.2 L331.2,191.2 L331.7,190.2 L332.3,189.1 L332.9,188.1 L333.5,187.1 L334.0,186.0 L334.6,185.0 L335.2,184.0 L335.8,182.9 L336.3,181.9 L336.9,180.9 L337.5,179.8 L338.1,178.8 L338.6,177.7 L339.2,176.7 L339.8,175.7 L340.4,174.6 L341.0,173.6 L341.5,172.6 L342.1,171.5 L342.7,170.5 L343.3,169.5 L343.8,168.4 L344.4,167.4 L345.0,166.4 L345.6,165.3 L346.1,164.3 L346.7,163.3 L347.3,162.2 L347.9,161.2 L348.4,160.2 L349.0,159.1 L349.6,158.1 L350.2,157.1 L350.8,156.0 L351.3,155.0 L351.9,154.0 L352.5,152.9 L353.1,151.9 L353.6,150.9 L354.2,149.8 L354.8,148.8 L355.4,147.8 L355.9,146.7 L356.5,145.7 L357.1,144.7 L357.7,143.6 L358.2,142.6 L358.8,141.6 L359.4,140.7 L360.0,139.8 L360.6,138.9 L361.1,138.1 L361.7,137.2 L362.3,136.3 L362.9,135.4 L363.4,134.6 L364.0,133.7 L364.6,132.8 L365.2,131.9 L365.7,131.1 L366.3,130.2 L366.9,129.3 L367.5,128.4 L368.0,127.5 L368.6,126.7 L369.2,125.8 L369.8,124.9 L370.4,124.0 L370.9,123.2 L371.5,122.3 L372.1,121.4 L372.7,120.5 L373.2,119.7 L373.8,118.8 L374.4,117.9 L375.0,117.0 L375.5,116.1 L376.1,115.3 L376.7,114.4 L377.3,113.5 L377.8,112.6 L378.4,111.8 L379.0,110.9 L379.6,110.0 L380.2,109.1 L380.7,108.3 L381.3,107.4 L381.9,106.5 L382.5,105.6 L383.0,104.7 L383.6,103.9 L384.2,103.0 L384.8,102.1 L385.3,101.2 L385.9,100.4 L386.5,99.5 L387.1,98.6 L387.6,97.8 L388.2,97.2 L388.8,96.6 L389.4,96.1 L389.9,95.5 L390.5,94.9 L391.1,94.3 L391.7,93.7 L392.3,93.1 L392.8,92.5 L393.4,92.0 L394.0,91.4 L394.6,90.8 L395.1,90.2 L395.7,89.6 L396.3,89.0 L396.9,88.4 L397.4,87.8 L398.0,87.3 L398.6,86.7 L399.2,86.1 L399.7,85.5 L400.3,84.9 L400.9,84.3 L401.5,83.7 L402.1,83.2 L402.6,82.6 L403.2,82.0 L403.8,81.4 L404.4,80.8 L404.9,80.2 L405.5,79.6 L406.1,79.1 L406.7,78.5 L407.2,77.9 L407.8,77.3 L408.4,76.7 L409.0,76.1 L409.5,75.5 L410.1,75.0 L410.7,74.4 L411.3,73.8 L411.9,73.2 L412.4,72.6 L413.0,72.0 L413.6,71.4 L414.2,70.9 L414.7,70.3 L415.3,69.7 L415.9,69.1 L416.5,68.7 L417.0,68.5 L417.6,68.2 L418.2,68.0 L418.8,67.8 L419.3,67.6 L419.9,67.4 L420.5,67.2 L421.1,67.0 L421.7,66.8 L422.2,66.6 L422.8,66.4 L423.4,66.2 L424.0,66.0 L424.5,65.8 L425.1,65.6 L425.7,65.4 L426.3,65.2 L426.8,65.0 L427.4,64.8 L428.0,64.5 L428.6,64.3 L429.1,64.1 L429.7,63.9 L430.3,63.7 L430.9,63.5 L431.5,63.3 L432.0,63.1 L432.6,62.9 L433.2,62.7 L433.8,62.5 L434.3,62.3 L434.9,62.1 L435.5,61.9 L436.1,61.7 L436.6,61.5 L437.2,61.3 L437.8,61.0 L438.4,60.8 L438.9,60.6 L439.5,60.4 L440.1,60.2 L440.7,60.0 L441.3,59.8 L441.8,59.6 L442.4,59.4 L443.0,59.2 L443.6,59.0 L444.1,58.8 L444.7,58.6 L445.3,58.6 L445.9,58.8 L446.4,59.0 L447.0,59.2 L447.6,59.4 L448.2,59.6 L448.7,59.8 L449.3,60.0 L449.9,60.2 L450.5,60.4 L451.1,60.6 L451.6,60.8 L452.2,61.0 L452.8,61.3 L453.4,61.5 L453.9,61.7 L454.5,61.9 L455.1,62.1 L455.7,62.3 L456.2,62.5 L456.8,62.7 L457.4,62.9 L458.0,63.1 L458.5,63.3 L459.1,63.5 L459.7,63.7 L460.3,63.9 L460.9,64.1 L461.4,64.3 L462.0,64.5 L462.6,64.8 L463.2,65.0 L463.7,65.2 L464.3,65.4 L464.9,65.6 L465.5,65.8 L466.0,66.0 L466.6,66.2 L467.2,66.4 L467.8,66.6 L468.3,66.8 L468.9,67.0 L469.5,67.2 L470.1,67.4 L470.7,67.6 L471.2,67.8 L471.8,68.0 L472.4,68.2 L473.0,68.5 L473.5,68.7 L474.1,69.1 L474.7,69.7 L475.3,70.3 L475.8,70.9 L476.4,71.4 L477.0,72.0 L477.6,72.6 L478.1,73.2 L478.7,73.8 L479.3,74.4 L479.9,75.0 L480.5,75.5 L481.0,76.1 L481.6,76.7 L482.2,77.3 L482.8,77.9 L483.3,78.5 L483.9,79.1 L484.5,79.6 L485.1,80.2 L485.6,80.8 L486.2,81.4 L486.8,82.0 L487.4,82.6 L487.9,83.2 L488.5,83.7 L489.1,84.3 L489.7,84.9 L490.3,85.5 L490.8,86.1 L491.4,86.7 L492.0,87.3 L492.6,87.8 L493.1,88.4 L493.7,89.0 L494.3,89.6 L494.9,90.2 L495.4,90.8 L496.0,91.4 L496.6,92.0 L497.2,92.5 L497.7,93.1 L498.3,93.7 L498.9,94.3 L499.5,94.9 L500.1,95.5 L500.6,96.1 L501.2,96.6 L501.8,97.2 L502.4,97.8 L502.9,98.6 L503.5,99.5 L504.1,100.4 L504.7,101.2 L505.2,102.1 L505.8,103.0 L506.4,103.9 L507.0,104.7 L507.5,105.6 L508.1,106.5 L508.7,107.4 L509.3,108.3 L509.8,109.1 L510.4,110.0 L511.0,110.9 L511.6,111.8 L512.2,112.6 L512.7,113.5 L513.3,114.4 L513.9,115.3 L514.5,116.1 L515.0,117.0 L515.6,117.9 L516.2,118.8 L516.8,119.7 L517.3,120.5 L517.9,121.4 L518.5,122.3 L519.1,123.2 L519.6,124.0 L520.2,124.9 L520.8,125.8 L521.4,126.7 L522.0,127.5 L522.5,128.4 L523.1,129.3 L523.7,130.2 L524.3,131.1 L524.8,131.9 L525.4,132.8 L526.0,133.7 L526.6,134.6 L527.1,135.4 L527.7,136.3 L528.3,137.2 L528.9,138.1 L529.4,138.9 L530.0,139.8 L530.6,140.7 L531.2,141.6 L531.8,142.6 L532.3,143.6 L532.9,144.7 L533.5,145.7 L534.1,146.7 L534.6,147.8 L535.2,148.8 L535.8,149.8 L536.4,150.9 L536.9,151.9 L537.5,152.9 L538.1,154.0 L538.7,155.0 L539.2,156.0 L539.8,157.1 L540.4,158.1 L541.0,159.1 L541.6,160.2 L542.1,161.2 L542.7,162.2 L543.3,163.3 L543.9,164.3 L544.4,165.3 L545.0,166.4 L545.6,167.4 L546.2,168.4 L546.7,169.5 L547.3,170.5 L547.9,171.5 L548.5,172.6 L549.0,173.6 L549.6,174.6 L550.2,175.7 L550.8,176.7 L551.4,177.7 L551.9,178.8 L552.5,179.8 L553.1,180.9 L553.7,181.9 L554.2,182.9 L554.8,184.0 L555.4,185.0 L556.0,186.0 L556.5,187.1 L557.1,188.1 L557.7,189.1 L558.3,190.2 L558.8,191.2 L559.4,192.2 L560.0,193.3" fill="none" class="stroke-b" stroke-width="2.2"/>
<circle cx="358.8" cy="141.7" r="3.5" class="fill-b"/>
<circle cx="387.5" cy="98.0" r="3.5" class="fill-b"/>
<circle cx="416.2" cy="68.7" r="3.5" class="fill-b"/>
<circle cx="445.0" cy="58.5" r="3.5" class="fill-b"/>
<circle cx="473.8" cy="68.7" r="3.5" class="fill-b"/>
<circle cx="502.5" cy="98.0" r="3.5" class="fill-b"/>
<circle cx="531.2" cy="141.7" r="3.5" class="fill-b"/>
<text x="445.0" y="22.0" text-anchor="middle" class="fig-bold">8 units, 8 pieces</text>
<text x="445.0" y="236.0" text-anchor="middle" class="fig-small">largest gap 0.019</text>
<text x="330.0" y="207.3" text-anchor="middle" class="fig-small">0</text>
<text x="560.0" y="207.3" text-anchor="middle" class="fig-small">1</text>
<text x="300.0" y="262.0" text-anchor="middle" class="fig-small">blue: the curve, orange: the sum of hinges, dots: where the bends sit</text>
</svg>
Figure 1. The curve sin(πx) approximated by sums of ReLU hinges, with bends at evenly spaced points. Each unit contributes one bend (orange dots). Notice that doubling the units from 4 to 8 cuts the largest gap from 0.070 to 0.019, roughly a quarter.
```

Figure 1 builds $\sin(\pi x)$ this way, with bends spaced evenly. Each unit adds at most one corner, so $m$ units make at most $m+1$ pieces. The reason is that one unit's reading is a straight line in $x$, and a straight line crosses 0 at most once. In the figure the first unit bends at 0, the edge of the range, so $m$ units give $m$ pieces. With 4 pieces the largest gap to the curve is $0.070$, and with 8 it is $0.019$, about a quarter. A smooth curve is close to a many-jointed polyline, and more joints means a closer fit.

This is the heart of the **universal approximation theorem**: one wide hidden layer can come as close as you like to any continuous function on a bounded interval. For a curve whose slope never exceeds $L$, pieces of width $h$ keep the gap under $L h / 2$, so on an interval of length 1 about $L/(2\epsilon)$ pieces reach a gap $\epsilon$. Three honest caveats: this is for a single input, and with many inputs the number of units needed can grow explosively. The theorem says such weights exist, not that training finds them. And it says nothing about how many units you need, which is the question here.

## Fold the line

The sawtooth shows the cheaper route. Start with the **tent map**, which takes $x$ in $[0, 1]$ up to 1 at $x = \tfrac12$ and back down:

$$
T(x) = 2\min(x,\, 1-x) = 2\operatorname{relu}(x) - 4\operatorname{relu}\!\left(x - \tfrac12\right) \quad \text{on } [0,1] .
$$

Read it aloud: "twice the smaller of $x$ and $1-x$, which two hinges make: a rise of slope 2 from the first, then a drop of 4 from the second that turns the slope into $-2$." Check $x = 0.75$: $2(0.75) - 4(0.25) = 0.5$, and $2\cdot\min(0.75, 0.25) = 0.5$.

Now feed $T$'s output into $T$ again. Because $T$ takes $[0,1]$ onto $[0,1]$, the second tent has a full-size input to work on, and it folds each half again. Think of folding a strip of paper in half: it has two layers. Fold the result again and it has four. The table shows $T$ and $T(T(x))$ on an even grid:

| $x$ | 0 | 0.125 | 0.25 | 0.375 | 0.5 | 0.625 | 0.75 | 0.875 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| $T(x)$ | 0 | 0.25 | 0.5 | 0.75 | 1 | 0.75 | 0.5 | 0.25 | 0 |
| $T(T(x))$ | 0 | 0.5 | 1 | 0.5 | 0 | 0.5 | 1 | 0.5 | 0 |

$T$ has one peak and two pieces. $T(T(x))$ has two peaks and four pieces, and Figure 2 shows a third fold with eight. After $k$ folds there are $2^k$ pieces.

```figure
<svg viewBox="0 0 600 302" role="img" aria-label="Three stacked panels showing the tent map and its repeats on the range 0 to 1, with 2, 4 and 8 straight pieces coloured alternately, every peak reaching 1.">

<text x="70.0" y="32.0" text-anchor="start" class="fig-bold">T(x): 2 pieces</text>
<line x1="70.0" y1="94.0" x2="540.0" y2="94.0" class="stroke-muted"/>
<path d="M70.0,94.0 L305.0,40.0" fill="none" class="stroke-a" stroke-width="2.6"/>
<path d="M305.0,40.0 L540.0,94.0" fill="none" class="stroke-b" stroke-width="2.6"/>
<circle cx="70.0" cy="94.0" r="3" class="fill-ink"/>
<circle cx="305.0" cy="40.0" r="3" class="fill-ink"/>
<circle cx="540.0" cy="94.0" r="3" class="fill-ink"/>
<text x="70.0" y="108.0" text-anchor="middle" class="fig-small">0</text>
<text x="540.0" y="108.0" text-anchor="middle" class="fig-small">1</text>
<text x="62.0" y="44.0" text-anchor="end" class="fig-small">1</text>
<text x="70.0" y="126.0" text-anchor="start" class="fig-bold">T(T(x)): 4 pieces</text>
<line x1="70.0" y1="188.0" x2="540.0" y2="188.0" class="stroke-muted"/>
<path d="M70.0,188.0 L187.5,134.0" fill="none" class="stroke-a" stroke-width="2.6"/>
<path d="M187.5,134.0 L305.0,188.0" fill="none" class="stroke-b" stroke-width="2.6"/>
<path d="M305.0,188.0 L422.5,134.0" fill="none" class="stroke-a" stroke-width="2.6"/>
<path d="M422.5,134.0 L540.0,188.0" fill="none" class="stroke-b" stroke-width="2.6"/>
<circle cx="70.0" cy="188.0" r="3" class="fill-ink"/>
<circle cx="187.5" cy="134.0" r="3" class="fill-ink"/>
<circle cx="305.0" cy="188.0" r="3" class="fill-ink"/>
<circle cx="422.5" cy="134.0" r="3" class="fill-ink"/>
<circle cx="540.0" cy="188.0" r="3" class="fill-ink"/>
<text x="70.0" y="202.0" text-anchor="middle" class="fig-small">0</text>
<text x="540.0" y="202.0" text-anchor="middle" class="fig-small">1</text>
<text x="62.0" y="138.0" text-anchor="end" class="fig-small">1</text>
<text x="70.0" y="220.0" text-anchor="start" class="fig-bold">T(T(T(x))): 8 pieces</text>
<line x1="70.0" y1="282.0" x2="540.0" y2="282.0" class="stroke-muted"/>
<path d="M70.0,282.0 L128.8,228.0" fill="none" class="stroke-a" stroke-width="2.6"/>
<path d="M128.8,228.0 L187.5,282.0" fill="none" class="stroke-b" stroke-width="2.6"/>
<path d="M187.5,282.0 L246.2,228.0" fill="none" class="stroke-a" stroke-width="2.6"/>
<path d="M246.2,228.0 L305.0,282.0" fill="none" class="stroke-b" stroke-width="2.6"/>
<path d="M305.0,282.0 L363.8,228.0" fill="none" class="stroke-a" stroke-width="2.6"/>
<path d="M363.8,228.0 L422.5,282.0" fill="none" class="stroke-b" stroke-width="2.6"/>
<path d="M422.5,282.0 L481.2,228.0" fill="none" class="stroke-a" stroke-width="2.6"/>
<path d="M481.2,228.0 L540.0,282.0" fill="none" class="stroke-b" stroke-width="2.6"/>
<circle cx="70.0" cy="282.0" r="3" class="fill-ink"/>
<circle cx="128.8" cy="228.0" r="3" class="fill-ink"/>
<circle cx="187.5" cy="282.0" r="3" class="fill-ink"/>
<circle cx="246.2" cy="228.0" r="3" class="fill-ink"/>
<circle cx="305.0" cy="282.0" r="3" class="fill-ink"/>
<circle cx="363.8" cy="228.0" r="3" class="fill-ink"/>
<circle cx="422.5" cy="282.0" r="3" class="fill-ink"/>
<circle cx="481.2" cy="228.0" r="3" class="fill-ink"/>
<circle cx="540.0" cy="282.0" r="3" class="fill-ink"/>
<text x="70.0" y="296.0" text-anchor="middle" class="fig-small">0</text>
<text x="540.0" y="296.0" text-anchor="middle" class="fig-small">1</text>
<text x="62.0" y="232.0" text-anchor="end" class="fig-small">1</text>
</svg>
Figure 2. Each fold doubles the number of pieces: 2, then 4, then 8. The colours alternate piece by piece. Notice that every peak reaches 1 and every valley reaches 0, so the next fold has a full-height copy of the zigzag to work on in each half.
```

## Build the folded network

Each fold needs two hinges, so a depth-$k$ network uses $2k$ hidden units. Write the hidden values of layer 1 as $h_1 = \operatorname{relu}(x)$ and $h_2 = \operatorname{relu}(x - \tfrac12)$. The tent's output is $y = 2h_1 - 4h_2$. The next layer needs $\operatorname{relu}(y)$ and $\operatorname{relu}(y - \tfrac12)$, but a layer can only take weighted sums of the previous layer's values. So you fuse the combine step into its weights: both rows are $[2, -4]$, and the constants are $[0, -\tfrac12]$. The first layer has rows $[1]$ and $[1]$ with constants $[0, -\tfrac12]$, and the last layer is $[2, -4]$ with no bend. In code:

```python
# Later hidden layer: apply the tent map again to y = 2 h1 - 4 h2.
W = np.array([[2.0, -4.0], [2.0, -4.0]]); b = np.array([0.0, -0.5])
```

A single hidden layer cannot keep up. It has at most $n + 1$ pieces with $n$ units, so $2^k$ pieces need at least $2^k - 1$ units, and $2^k$ units are enough. Figure 3 compares the two. At $k = 6$ the folded network has 64 pieces from 12 units, where one layer needs 64.

```figure
<svg viewBox="0 0 600 236" role="img" aria-label="Grouped bars for k from 1 to 6 comparing the number of hidden units a depth-k network needs, which is 2k, with the 2 to the k units one hidden layer needs for the same number of pieces, which reaches 64 at k equal to 6 while 2k reaches 12.">

<line x1="60.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-muted"/>
<text x="54.0" y="204.0" text-anchor="end" class="fig-small">0</text>
<line x1="60.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-grid"/>
<text x="54.0" y="155.4" text-anchor="end" class="fig-small">20</text>
<line x1="60.0" y1="151.4" x2="560.0" y2="151.4" class="stroke-grid"/>
<text x="54.0" y="106.9" text-anchor="end" class="fig-small">40</text>
<line x1="60.0" y1="102.9" x2="560.0" y2="102.9" class="stroke-grid"/>
<text x="54.0" y="58.3" text-anchor="end" class="fig-small">60</text>
<line x1="60.0" y1="54.3" x2="560.0" y2="54.3" class="stroke-grid"/>
<rect x="103.4" y="195.1" width="26.0" height="4.9" class="fill-b"/>
<rect x="133.4" y="195.1" width="26.0" height="4.9" class="fill-a"/>
<text x="116.4" y="190.1" text-anchor="middle" class="fig-small">2</text>
<text x="146.4" y="190.1" text-anchor="middle" class="fig-small">2</text>
<text x="131.4" y="216.0" text-anchor="middle" class="fig-small">k = 1</text>
<rect x="174.9" y="190.3" width="26.0" height="9.7" class="fill-b"/>
<rect x="204.9" y="190.3" width="26.0" height="9.7" class="fill-a"/>
<text x="187.9" y="185.3" text-anchor="middle" class="fig-small">4</text>
<text x="217.9" y="185.3" text-anchor="middle" class="fig-small">4</text>
<text x="202.9" y="216.0" text-anchor="middle" class="fig-small">k = 2</text>
<rect x="246.3" y="185.4" width="26.0" height="14.6" class="fill-b"/>
<rect x="276.3" y="180.6" width="26.0" height="19.4" class="fill-a"/>
<text x="259.3" y="180.4" text-anchor="middle" class="fig-small">6</text>
<text x="289.3" y="175.6" text-anchor="middle" class="fig-small">8</text>
<text x="274.3" y="216.0" text-anchor="middle" class="fig-small">k = 3</text>
<rect x="317.7" y="180.6" width="26.0" height="19.4" class="fill-b"/>
<rect x="347.7" y="161.1" width="26.0" height="38.9" class="fill-a"/>
<text x="330.7" y="175.6" text-anchor="middle" class="fig-small">8</text>
<text x="360.7" y="156.1" text-anchor="middle" class="fig-small">16</text>
<text x="345.7" y="216.0" text-anchor="middle" class="fig-small">k = 4</text>
<rect x="389.1" y="175.7" width="26.0" height="24.3" class="fill-b"/>
<rect x="419.1" y="122.3" width="26.0" height="77.7" class="fill-a"/>
<text x="402.1" y="170.7" text-anchor="middle" class="fig-small">10</text>
<text x="432.1" y="117.3" text-anchor="middle" class="fig-small">32</text>
<text x="417.1" y="216.0" text-anchor="middle" class="fig-small">k = 5</text>
<rect x="460.6" y="170.9" width="26.0" height="29.1" class="fill-b"/>
<rect x="490.6" y="44.6" width="26.0" height="155.4" class="fill-a"/>
<text x="473.6" y="165.9" text-anchor="middle" class="fig-small">12</text>
<text x="503.6" y="39.6" text-anchor="middle" class="fig-small">64</text>
<text x="488.6" y="216.0" text-anchor="middle" class="fig-small">k = 6</text>
<rect x="80.0" y="16.0" width="12.0" height="12.0" class="fill-b"/>
<text x="98.0" y="26.0" text-anchor="start" class="fig-small">depth k: 2k units</text>
<rect x="250.0" y="16.0" width="12.0" height="12.0" class="fill-a"/>
<text x="268.0" y="26.0" text-anchor="start" class="fig-small">one hidden layer: 2^k units (at least 2^k − 1)</text>
</svg>
Figure 3. Hidden units needed to make the 2^k-piece sawtooth. A network of depth k uses 2k units (orange). A single hidden layer needs at least 2^k − 1 and 2^k are enough (blue). Notice that the two tie at k = 1 and 2, then the gap widens fast: 12 against 64 at k = 6.
```

## Count the pieces exactly

To test the claim you need to count pieces without plotting. The idea is that inside any stretch of the input where no unit changes sign, the whole network is one straight line. So the counter keeps, for every stretch, a slope and an intercept for each value in the current layer. The input itself starts with slope 1 and intercept 0 on $[x_{\min}, x_{\max}]$.

For each layer, on each stretch, do four things:

1. Compute every unit's reading as a line: slope `W @ slopes`, intercept `W @ intercepts + b`.
2. A reading crosses 0 at $x = -\text{intercept}/\text{slope}$. For a ReLU unit, whose output changes character at that point, split the stretch there if the crossing lands strictly inside it. An identity unit has no bend, so it never splits anything.
3. On each part, test the sign at the midpoint. A ReLU unit with a negative midpoint reading is cut to 0, so its slope and intercept become 0.
4. After the last layer, merge neighbours whose final lines are equal.

Run it on one hinge, $\operatorname{relu}(x - 0.5)$ on $[0, 1]$, followed by an output weight of 1. The reading has slope 1 and intercept $-0.5$, so it crosses at $x = 0.5$ and the stretch splits in two. On $[0, 0.5]$ the midpoint reading is $-0.25$, so the unit is cut to slope 0, intercept 0. On $[0.5, 1]$ it stays slope 1, intercept $-0.5$. The two lines differ, so the count is 2. Change the output weight to 0 and both parts give slope 0, intercept 0, so the merge leaves 1 piece: the crossing happened but changed nothing.

Depth multiplies here. In the second layer of the sawtooth, the reading $y - \tfrac12$ crosses 0 once inside each of the two stretches the first layer made, so it splits both and you get four pieces. One hidden layer cannot do this: its readings are lines over the whole input, so each unit splits at most once.

> [!KEY]
> A unit adds one bend, so width adds pieces one at a time. A layer can fold the curve the previous layer made, so $k$ layers of width 2 make $2^k$ pieces from $2k$ units, while one hidden layer needs about $2^k$. Depth reuses work, as a function reuses a helper.

> [!RECAP]
> - A ReLU unit is a hinge, and $m$ units make at most $m+1$ pieces.
> - The tent map is $2\operatorname{relu}(x) - 4\operatorname{relu}(x-\tfrac12)$, and one fold doubles the pieces.
> - Later layers use weights `[[2, -4], [2, -4]]` and constants `[0, -0.5]`, and the first layer reads `x` directly.
> - Counting tracks a slope and an intercept per stretch, splits at interior roots, cuts negative ReLU units to 0, and merges equal neighbours.
> - One hidden layer needs at least $2^k - 1$ units to match depth $k$.

## Build it step by step

`relu`, `gelu`, and `mlp_forward` from the last chapters are already in the starter. Build two functions.

1. **`count_linear_regions_1d(params, x_min, x_max)`** takes a network with one input, ReLU or identity hidden layers, and an identity output layer, and returns the number of pieces as an integer. Keep a list of stretches, each with its left end, right end, slopes, and intercepts. Split only at roots strictly inside the stretch, and skip units whose slope is near 0. The most common mistake is to skip the merge, which over-counts a crossing that changes nothing. Write it like this:

    ```python
    def count_linear_regions_1d(params, x_min, x_max):
        # For each layer: split on interior roots, cut inactive ReLUs, then merge.
    ```

2. **`sawtooth_network(depth)`** returns `depth + 1` layers. The first has `W` of shape `(2, 1)` with both rows `[1]` and constants `[0, -0.5]`. The `depth - 1` middle layers have the fused weights and constants above, all with `"relu"`. The last has `W = [[2, -4]]`, constant `[0]`, and `"identity"`. The usual mistake is an extra layer that combines the two units, which breaks the count `depth + 1`. Write it like this:

    ```python
    def sawtooth_network(depth):
        # depth ReLU layers of width 2, then one identity layer.
    ```

## Where this breaks

The sawtooth network exists, but training will rarely find it. Gradient descent starts from random weights and settles on whatever bends fit the data, so depth makes a function possible, not likely. The counting code also assumes one input; with two or more inputs the pieces are flat patches with straight borders, and counting them is much harder.

## The exercise

You implement `count_linear_regions_1d` and `sawtooth_network`. The checks count a single hinge (2 pieces), an identity-only network (1), a hinge whose crossing lies outside the interval (1), and a hinge with output weight 0 (1, by the merge). They count a one-hidden-layer network with kinks at five points (6), and confirm the sawtooth has $2^k$ pieces for $k = 1$ to 5. They also compare it with the closed form $T$ applied $k$ times, and check its layer shapes.
