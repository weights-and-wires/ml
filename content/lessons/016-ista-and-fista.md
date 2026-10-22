---
title: "ISTA and FISTA: fitting the lasso by stepping and thresholding"
slug: 016-ista-and-fista
part: "Part 1 — Foundations"
summary: Fit the lasso for any design by stepping downhill and then soft-thresholding, speed it up with momentum, and test for the finish with the KKT rules.
minutes: 45
gpu: false
objectives:
  - Explain why you cannot set the lasso's slope to zero, and why plain gradient steps never land on exact zeros.
  - Build ISTA, which steps on the squared error and then soft-thresholds, and FISTA, which adds momentum.
  - Test whether a set of weights is a lasso answer with the KKT rules.
prereqs:
  - 010-how-big-a-step
  - 011-long-valleys
  - 012-momentum
  - 015-lasso-and-soft-thresholding
exercise: 016-ista-and-fista
---

# ISTA and FISTA: fitting the lasso by stepping and thresholding

You have 100 rows and 40 correlated features, and you suspect only 5 of them matter. You want the lasso weights. The previous chapter solved the lasso in one move, but only when $X$ was the identity. For a real data matrix there is no formula, because the penalty has a corner at 0. You need a method that walks to the bottom, and a way to know when you have arrived.

The walk is awkward for two reasons. The squared error ties the weights together, and the corner means weights should land on exactly 0, which an ordinary downhill walk almost never does. This chapter splits the job: step downhill on the smooth squared error, then soft-threshold to deal with the corner.

Figure 1 shows the payoff on a two-weight problem with a long, narrow valley, which is what correlated columns make. Both methods use the same step size, and one of them builds speed along the valley floor and arrives far sooner.

```figure
<svg viewBox="0 0 600 292" role="img" aria-label="Contours of a long narrow valley in the plane of two weights, with the ISTA path creeping along the valley floor and the FISTA path moving along it faster, both towards the best weights.">

<polyline points="301.8,195.5 301.2,194.9 300.7,194.4 300.2,193.9 299.8,193.5 299.4,193.0 299.0,192.6 298.6,192.2 298.2,191.9 297.8,191.5 297.5,191.2 297.1,190.8 296.8,190.5 296.5,190.2 296.2,189.9 295.9,189.6 295.6,189.3 295.3,189.0 295.0,188.7 294.7,188.4 294.4,188.1 294.1,187.9 293.9,187.6 293.6,187.3 293.3,187.0 293.0,186.7 292.7,186.5 292.4,186.2 292.2,185.9 291.9,185.6 291.6,185.3 291.3,185.0 290.9,184.7 290.6,184.4 290.3,184.1 290.0,183.7 289.6,183.4 289.3,183.0 288.9,182.7 288.5,182.3 288.1,181.9 287.7,181.5 287.2,181.0 286.7,180.6 286.2,180.1 285.7,179.5 285.1,179.0 284.5,178.4 283.8,177.7 283.1,177.0 282.3,176.2 281.4,175.4 280.5,174.4 279.4,173.3 278.1,172.1 276.7,170.8 275.1,169.2 273.2,167.3 270.9,165.2 268.2,162.5 264.8,159.3 260.5,155.2 255.0,149.9 247.4,142.8 236.8,132.9 227.7,126.3 223.8,126.7 221.5,129.0 221.0,133.0 222.2,138.2 224.4,144.0 227.1,149.7 230.1,155.1 238.0,163.3 243.8,169.3 248.2,173.8 251.7,177.4 254.6,180.3 256.9,182.6 258.8,184.6 260.5,186.3 261.9,187.7 263.2,189.0 264.3,190.1 265.3,191.1 266.2,192.0 267.0,192.8 267.8,193.6 268.4,194.2 269.1,194.9 269.6,195.5 270.2,196.0 270.7,196.5 271.2,197.0 271.6,197.4 272.1,197.9 272.5,198.3 272.9,198.7 273.2,199.0 273.6,199.4 273.9,199.7 274.3,200.1 274.6,200.4 274.9,200.7 275.2,201.0 275.5,201.3 275.8,201.6 276.1,201.9 276.4,202.2 276.7,202.5 277.0,202.8 277.3,203.0 277.6,203.3 277.8,203.6 278.1,203.9 278.4,204.2 278.7,204.4 279.0,204.7 279.3,205.0 279.6,205.3 279.9,205.6 280.2,205.9 280.5,206.2 280.8,206.5 281.1,206.8 281.4,207.2 281.8,207.5 282.1,207.9 282.5,208.2 282.9,208.6 283.3,209.0 283.7,209.4 284.2,209.9 284.7,210.3 285.2,210.8 285.7,211.4 286.3,211.9 286.9,212.5 287.6,213.2 288.3,213.9 289.1,214.7 290.0,215.5 291.0,216.5 292.0,217.6 293.3,218.8 294.7,220.1 296.3,221.7 298.2,223.6 300.5,225.8 303.2,228.4 306.6,231.6 310.9,235.7 316.5,241.0 324.0,248.2 334.6,258.1" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="341.3,235.8 333.4,227.6 327.6,221.6 323.2,217.1 319.7,213.5 316.9,210.6 314.5,208.3 312.6,206.3 310.9,204.6 309.5,203.2 308.2,201.9 307.1,200.8 306.1,199.8 305.2,198.9 304.4,198.1 303.7,197.3 303.0,196.7 302.4,196.0 301.8,195.5" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="317.8,195.5 316.8,194.4 315.7,193.4 314.8,192.4 313.9,191.5 313.0,190.6 312.2,189.8 311.4,189.0 310.7,188.3 309.9,187.6 309.2,186.9 308.6,186.2 307.9,185.6 307.3,184.9 306.7,184.3 306.0,183.7 305.4,183.1 304.9,182.5 304.3,182.0 303.7,181.4 303.1,180.8 302.6,180.3 302.0,179.7 301.4,179.2 300.9,178.6 300.3,178.0 299.8,177.5 299.2,176.9 298.6,176.3 298.0,175.8 297.4,175.2 296.8,174.6 296.2,174.0 295.6,173.3 294.9,172.7 294.2,172.0 293.5,171.3 292.8,170.6 292.1,169.9 291.3,169.1 290.5,168.3 289.6,167.5 288.7,166.6 287.8,165.7 286.8,164.7 285.7,163.6 284.6,162.5 283.3,161.3 282.0,160.0 280.5,158.5 278.9,157.0 277.2,155.3 275.2,153.4 273.0,151.2 270.6,148.8 267.7,146.1 264.5,142.9 260.7,139.2 256.1,134.8 250.7,129.6 243.9,123.1 235.4,114.9 226.4,107.6 218.5,103.0 210.0,98.5 201.4,95.0 193.9,93.5 189.1,95.4 188.0,101.1 190.5,109.7 195.1,119.4 200.5,128.8 205.8,137.4 210.8,144.9 215.3,151.5 219.3,157.1 222.8,162.0 226.0,166.3 228.8,170.1 232.0,173.7 235.3,177.1 238.2,180.0 240.7,182.5 242.9,184.8 244.9,186.8 246.7,188.6 248.3,190.2 249.8,191.7 251.2,193.0 252.4,194.3 253.6,195.5 254.7,196.5 255.7,197.6 256.6,198.5 257.5,199.4 258.4,200.3 259.2,201.1 260.0,201.9 260.7,202.6 261.5,203.3 262.2,204.0 262.8,204.7 263.5,205.3 264.1,206.0 264.8,206.6 265.4,207.2 266.0,207.8 266.6,208.4 267.1,208.9 267.7,209.5 268.3,210.1 268.8,210.6 269.4,211.2 270.0,211.8 270.5,212.3 271.1,212.9 271.7,213.4 272.2,214.0 272.8,214.6 273.4,215.2 274.0,215.7 274.6,216.3 275.2,217.0 275.9,217.6 276.5,218.2 277.2,218.9 277.9,219.6 278.6,220.3 279.3,221.0 280.1,221.8 280.9,222.6 281.8,223.4 282.7,224.3 283.6,225.2 284.6,226.2 285.7,227.3 286.9,228.4 288.1,229.6 289.4,230.9 290.9,232.4 292.5,233.9 294.2,235.6 296.2,237.5 298.4,239.7 300.9,242.1 303.7,244.8 306.9,248.0 310.7,251.7 315.3,256.1" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="348.0,225.8 343.4,221.1 339.5,217.2 336.1,213.8 333.2,210.9 330.7,208.4 328.5,206.1 326.5,204.1 324.7,202.3 323.1,200.7 321.6,199.2 320.3,197.9 319.0,196.6 317.8,195.5" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="341.4,195.5 339.5,193.6 337.7,191.8 336.1,190.2 334.5,188.6 333.0,187.1 331.6,185.7 330.2,184.4 328.9,183.1 327.7,181.8 326.5,180.6 325.3,179.5 324.2,178.3 323.1,177.2 322.0,176.2 320.9,175.1 319.9,174.1 318.9,173.1 317.9,172.1 316.9,171.1 315.9,170.1 314.9,169.1 313.9,168.2 313.0,167.2 312.0,166.3 311.0,165.3 310.0,164.3 309.0,163.3 308.0,162.3 307.0,161.3 306.0,160.3 304.9,159.3 303.9,158.2 302.8,157.1 301.6,156.0 300.5,154.9 299.3,153.7 298.0,152.5 296.7,151.2 295.4,149.9 294.0,148.5 292.5,147.0 291.0,145.5 289.3,143.9 287.6,142.2 285.7,140.3 283.7,138.4 281.6,136.3 279.2,134.0 276.7,131.5 274.0,128.8 270.9,125.8 267.5,122.5 263.7,118.8 259.5,114.7 254.6,109.9 248.9,104.5 242.3,98.1 234.5,90.5 225.9,82.9 216.6,75.8 205.8,67.7 193.2,58.3 178.3,47.6 161.1,36.0 142.1,24.3" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="107.0,22.8 114.1,40.9 126.5,61.9 139.9,81.5 152.2,98.4 162.9,112.6 172.0,124.4 179.8,134.3 186.5,142.7 192.3,149.9 197.4,156.1 201.8,161.6 205.8,166.4 209.3,170.6 212.4,174.4 215.3,177.9 217.9,181.0 220.3,183.9 222.6,186.6 224.6,189.0 226.6,191.3 228.3,193.4 230.0,195.5 231.9,197.3 233.7,199.1 235.3,200.7 236.9,202.3 238.4,203.8 239.8,205.2 241.2,206.6 242.5,207.8 243.7,209.1 244.9,210.3 246.1,211.5 247.2,212.6 248.3,213.7 249.4,214.7 250.5,215.8 251.5,216.8 252.5,217.8 253.5,218.8 254.5,219.8 255.5,220.8 256.5,221.8 257.5,222.7 258.4,223.7 259.4,224.7 260.4,225.6 261.4,226.6 262.4,227.6 263.4,228.6 264.4,229.6 265.4,230.6 266.5,231.6 267.5,232.7 268.6,233.8 269.8,234.9 270.9,236.0 272.1,237.2 273.4,238.4 274.7,239.7 276.0,241.0 277.4,242.4 278.9,243.9 280.5,245.4 282.1,247.0 283.8,248.7 285.7,250.6 287.7,252.5 289.8,254.6 292.2,256.9 294.7,259.4" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="347.9,202.0 345.6,199.6 343.4,197.5 341.4,195.5" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="349.8,174.6 348.0,172.8 346.2,171.0 344.5,169.3 342.8,167.6 341.1,166.0 339.5,164.4 337.9,162.8 336.4,161.3 334.9,159.7 333.3,158.2 331.8,156.8 330.3,155.3 328.8,153.8 327.4,152.3 325.9,150.8 324.4,149.4 322.9,147.9 321.4,146.4 319.8,144.9 318.3,143.3 316.7,141.8 315.1,140.2 313.4,138.6 311.8,136.9 310.0,135.2 308.3,133.5 306.4,131.7 304.5,129.8 302.6,127.8 300.5,125.8 298.4,123.7 296.1,121.5 293.7,119.1 291.2,116.7 288.5,114.1 285.7,111.3 282.7,108.3 279.4,105.1 275.8,101.6 272.0,97.8 267.8,93.7 263.1,89.1 257.9,84.1 252.2,78.4 245.6,72.1 238.2,64.8 229.6,56.5 220.4,48.7 209.8,39.8 197.4,29.5" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="69.3,38.3 90.1,63.5 107.1,83.8 121.1,100.4 132.9,114.2 142.8,125.8 151.4,135.6 158.8,144.2 165.3,151.6 171.0,158.2 176.2,164.0 180.8,169.3 185.0,174.0 188.8,178.4 192.3,182.3 195.5,186.0 198.5,189.4 201.3,192.5 203.9,195.5 206.4,198.2 208.7,200.8 210.9,203.3 213.0,205.7 215.0,207.9 216.9,210.1 218.8,212.1 220.6,214.1 222.3,216.1 224.0,217.9 225.6,219.7 227.2,221.5 228.7,223.2 230.3,224.9 231.9,226.5 233.5,228.1 235.0,229.6 236.6,231.2 238.1,232.7 239.6,234.2 241.1,235.6 242.6,237.1 244.1,238.6 245.5,240.1 247.0,241.5 248.5,243.0 250.1,244.5 251.6,246.0 253.1,247.6 254.7,249.1 256.3,250.7 258.0,252.3 259.6,254.0 261.4,255.7 263.1,257.4 265.0,259.3" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="349.2,152.6 347.3,150.7 345.4,148.8 343.5,146.9 341.6,145.1 339.8,143.2 337.9,141.4 336.1,139.5 334.2,137.7 332.3,135.8 330.4,133.9 328.5,132.0 326.5,130.1 324.6,128.2 322.5,126.2 320.5,124.2 318.4,122.1 316.2,119.9 314.0,117.7 311.7,115.5 309.3,113.1 306.8,110.7 304.3,108.2 301.6,105.5 298.7,102.7 295.8,99.8 292.6,96.7 289.3,93.4 285.7,89.9 281.9,86.2 277.8,82.1 273.3,77.8 268.5,73.1 263.2,67.9 257.4,62.2 250.9,55.8 243.6,48.8 235.5,40.8 226.3,32.2 216.2,23.4" fill="none" class="stroke-muted" stroke-width="1"/>
<polyline points="32.4,24.6 55.9,51.8 75.1,73.8 90.9,91.9 104.2,106.9 115.5,119.7 125.3,130.6 133.8,140.2 141.3,148.5 148.0,156.0 154.0,162.6 159.4,168.6 164.3,174.0 168.8,179.0 172.9,183.6 176.8,187.8 180.4,191.8 183.7,195.5 186.8,198.9 189.8,202.2 192.6,205.2 195.3,208.2 197.8,210.9 200.3,213.6 202.6,216.2 204.8,218.6 207.0,221.0 209.1,223.3 211.2,225.6 213.2,227.7 215.1,229.9 217.0,232.0 218.9,234.0 220.8,236.0 222.6,238.0 224.4,240.0 226.2,241.9 228.0,243.9 229.8,245.8 231.6,247.7 233.5,249.5 235.4,251.4 237.2,253.2 239.1,255.1 241.0,257.0 242.9,258.9" fill="none" class="stroke-muted" stroke-width="1"/>
<rect x="30.0" y="20.0" width="320.0" height="240.0" fill="none" class="stroke-muted"/>
<text x="34.0" y="276.0" class="fig-small">w1</text>
<text x="24.0" y="32.0" class="fig-small" text-anchor="end">w2</text>
<polyline points="80.0,80.0 129.9,36.1 133.2,39.4 136.5,42.6 139.7,45.8 143.0,49.0 146.2,52.2 149.4,55.4 152.6,58.6 155.8,61.7 158.9,64.9 162.1,68.0 165.2,71.1 168.4,74.2 171.5,77.3 174.6,80.3 177.7,83.4 180.7,86.4 183.8,89.5 186.8,92.5 189.9,95.5 192.9,98.5 195.9,101.4 198.9,104.4 201.9,107.4 204.8,110.3" fill="none" class="stroke-d" stroke-width="6"/>
<polyline points="80.0,80.0 129.9,36.1 133.2,39.4 137.4,43.5 142.5,48.5 148.4,54.4 155.1,61.0 162.6,68.5 170.9,76.7 179.9,85.6 189.6,95.2 200.1,105.6 211.2,116.6 222.9,128.2 230.1,140.4 238.4,148.6 245.4,155.6 251.5,161.6 256.8,166.8 261.4,171.4 265.5,175.4 269.1,179.0 272.4,182.2 275.2,185.1 277.8,187.7 280.2,190.0" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="143.0" cy="49.0" r="4.5" class="fill-d"/>
<text x="132.0" y="53.0" class="fig-small fig-bold" text-anchor="end">5</text>
<circle cx="158.9" cy="64.9" r="4.5" class="fill-d"/>
<text x="147.9" y="68.9" class="fig-small fig-bold" text-anchor="end">10</text>
<circle cx="174.6" cy="80.3" r="4.5" class="fill-d"/>
<text x="163.6" y="84.3" class="fig-small fig-bold" text-anchor="end">15</text>
<circle cx="189.9" cy="95.5" r="4.5" class="fill-d"/>
<text x="178.9" y="99.5" class="fig-small fig-bold" text-anchor="end">20</text>
<circle cx="204.8" cy="110.3" r="4.5" class="fill-d"/>
<text x="193.8" y="114.3" class="fig-small fig-bold" text-anchor="end">25</text>
<circle cx="148.4" cy="54.4" r="4.5" class="fill-b"/>
<text x="159.4" y="52.4" class="fig-small fig-bold" text-anchor="start">5</text>
<circle cx="189.6" cy="95.2" r="4.5" class="fill-b"/>
<text x="200.6" y="93.2" class="fig-small fig-bold" text-anchor="start">10</text>
<circle cx="238.4" cy="148.6" r="4.5" class="fill-b"/>
<text x="249.4" y="146.6" class="fig-small fig-bold" text-anchor="start">15</text>
<circle cx="265.5" cy="175.4" r="4.5" class="fill-b"/>
<text x="276.5" y="173.4" class="fig-small fig-bold" text-anchor="start">20</text>
<circle cx="280.2" cy="190.0" r="4.5" class="fill-b"/>
<text x="291.2" y="188.0" class="fig-small fig-bold" text-anchor="start">25</text>
<circle cx="285.7" cy="195.5" r="5" class="fill-ink"/>
<circle cx="80.0" cy="80.0" r="5.5" class="fill-a"/>
<text x="90.0" y="84.0" class="fig-small">start</text>
<text x="295.7" y="217.5" class="fig-small">best weights</text>
<line x1="372.0" y1="40.0" x2="400.0" y2="40.0" class="stroke-d" stroke-width="3"/>
<text x="408.0" y="44.0" class="fig-small fig-bold">ISTA</text>
<line x1="372.0" y1="64.0" x2="400.0" y2="64.0" class="stroke-b" stroke-width="3"/>
<text x="408.0" y="68.0" class="fig-small fig-bold">FISTA</text>
<text x="372.0" y="100.0" class="fig-small">Numbers are step counts: ISTA on the left</text>
<text x="372.0" y="117.0" class="fig-small">of the path, FISTA on the right.</text>
<text x="372.0" y="134.0" class="fig-small">Gray lines: the score is 0.5, 2, 6, 14,</text>
<text x="372.0" y="151.0" class="fig-small">and 22 above its best value.</text>
<text x="372.0" y="185.0" class="fig-small">ISTA drops into the valley in one step,</text>
<text x="372.0" y="202.0" class="fig-small">then creeps along its floor.</text>
<text x="372.0" y="219.0" class="fig-small">FISTA gathers speed along the floor.</text>
<text x="372.0" y="253.0" class="fig-small">Score above best after 25 steps:</text>
<text x="372.0" y="270.0" class="fig-small">ISTA 1.29, FISTA 0.001</text>
</svg>
Figure 1. Contours of the lasso score for two correlated columns, with the paths of ISTA and FISTA from the same start. Both drop into the valley on the first step and run along the same floor, so the purple ISTA line hides under the orange one. Compare the dots: after 25 steps ISTA is still far from the best weights, while FISTA is almost there.
```

## What you already know

- Soft thresholding $S_t(x) = \mathrm{sign}(x)\max(|x| - t, 0)$ moves $x$ towards 0 by $t$ and stops at 0. You built it in the chapter *Lasso and soft thresholding: why some weights become exactly zero*.
- The lasso score is $\tfrac12\lVert Xw - y\rVert^2 + \lambda\lVert w\rVert_1$.
- A downhill step must stay under $2/L$, where $L$ is the largest curvature, as in *How big a step: curvature and the 2/L limit*. For the squared error, $L$ is the largest eigenvalue of $X^\top X$, which is the square of the largest singular value of $X$.
- A long narrow valley is slow because one step size has to fit its steep walls, as in *Condition number: why long valleys are slow*, and momentum speeds up a walk along it, as in *Momentum: a heavy ball and a look-ahead*.

## Why not set the slope to zero, or walk anyway?

Ridge had a formula because its score is smooth, so you set the slope to zero. The lasso fee $\lambda|w|$ has a corner at 0: its slope is $-\lambda$ on the left and $+\lambda$ on the right, with no single slope in between. You cannot solve "slope equals zero" at a weight that sits on the corner, and sitting there is exactly what you want.

The next idea is to walk anyway, using the slope $\lambda\,\mathrm{sign}(w)$ for the fee. Take $X$ with one column $(1, 2)$, labels $y = (1, 2)$, and $\lambda = 6$. The best weight is 0, because the data's pull at 0 is 5, which is below 6. Start at $w = 0.1$ with step 0.2. The slope of the squared error is $5w - 5$, and the fee adds $+6$ to the right of 0 and $-6$ to the left. At $w = 0.1$ the total slope is $-4.5 + 6 = 1.5$, so the walk goes to $0.1 - 0.2 \times 1.5 = -0.2$. There the slope is $-6 - 6 = -12$, so it jumps to 2.2, then back to $-0.2$, and so on forever. Each step jumps over 0 and never lands on it, so a plain walk bounces around the answer instead of settling on an exact zero.

The method that works treats the two parts differently. It walks on the smooth part, and for the corner it asks a direct question: given where the walk landed, what is the best compromise between staying there and paying the fee? You answered that question for one number in the previous chapter. It is soft thresholding.

## Step downhill, then threshold

One ISTA step takes a plain gradient step on the squared error, then applies soft thresholding. Use the column $(1, 2)$ with labels $(1, 2)$ and $\lambda = 1$, and start at $w = 0$.

At $w = 0$ the misses are $-1$ and $-2$, so the score is $\tfrac12(1 + 4) = 2.5$. The gradient of the squared error is $X^\top(Xw - y) = 1(-1) + 2(-2) = -5$. Here $X^\top X = 5$, so $L = 5$ and the step is $1/L = 0.2$. The gradient step goes to $0 - 0.2 \times (-5) = 1$. Then the fee gets the same step size, so you threshold at $0.2 \times 1 = 0.2$, and $1$ becomes $0.8$. The predictions are $0.8$ and $1.6$, the misses are $-0.2$ and $-0.4$, and the score is $\tfrac12(0.04 + 0.16) + 0.8 = 0.9$. Figure 2 draws this step.

In symbols, with step $\alpha$:

$$
w_{k+1} = S_{\alpha\lambda}\!\big(w_k - \alpha\, X^\top(Xw_k - y)\big)
$$

Read it as: step against the gradient of the squared error, then soft-threshold at the step size times lambda. Each step is the best compromise between two things: staying near the gradient step's landing point, and paying the fee. The threshold is step times lambda because that compromise, with the distance scaled by the step size, is the one-number problem $\tfrac12(z - v)^2 + \alpha\lambda|z|$ from the previous chapter, with $t = \alpha\lambda$. A long step trusts the gradient more, so the fee gets a proportionally larger say. A step of $1/L$ is safe because the squared error is a bowl whose steepest curvature is $L$. For a convex bowl and a step of at most $1/L$, the score never rises from one ISTA step to the next.

```python
grad = X.T @ (X @ w - y)                          # shape (d,)
w = soft_threshold(w - step * grad, step * lam)   # threshold is step * lam, not lam
```

Return to the $\lambda = 6$ case where the plain walk bounced. From $w = 0.1$ the gradient step goes to $0.1 - 0.2 \times (-4.5) = 1.0$, and the threshold is $0.2 \times 6 = 1.2$. Since $1.0 < 1.2$, the weight lands on exactly 0 and stays there.

This also ties back to the previous chapter. When $X$ is the identity, $L = 1$ and the step is 1, so the gradient step $w - (w - y)$ lands on $y$ whatever $w$ was, and the threshold step then gives $S_\lambda(y)$. One ISTA step is the exact lasso solve you built there.

## Know when you have finished: the KKT rules

At the bottom of a smooth bowl the slope is zero. The lasso needs a test that also covers weights on the corner. Write $g = X^\top(Xw - y)$, the gradient of the squared error, and read $-g_i$ as the data's pull on weight $i$. The fee's pull is $-\lambda\,\mathrm{sign}(w_i)$. These two conditions are called the *KKT rules*, after Karush, Kuhn, and Tucker, who found them. Each weight must obey one of them:

- If $w_i \ne 0$, then $g_i + \lambda\,\mathrm{sign}(w_i) = 0$. The data's pull and the fee's pull cancel.
- If $w_i = 0$, then $|g_i| \le \lambda$. The data's pull is too weak to move the weight off 0.

On the column example at $w = 0.8$ we have $g = -1$, so $g + \lambda = 0$. The first rule holds, which confirms that 0.8 is the answer. At $w = 0$ we have $g = -5$, and $5 > 1$ breaks the second rule by $5 - 1 = 4$, so you have not finished. Define the *violation* as the largest breach over all weights. It is 0 at a lasso answer and positive elsewhere, which makes it the test for "am I done?"

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Left: a number line showing one ISTA step from 0 to 1 and then to 0.8 after thresholding, with equal data and fee pulls at 0.8. Right: bars of the FISTA momentum weight for steps 1 to 8, rising from 0 towards 1.">
<defs><marker id="c016-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker><marker id="c016-arrow-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-a"/></marker><marker id="c016-arrow-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<text x="40.0" y="24.0" class="fig-bold">One ISTA step on the one-column example</text>
<line x1="40.0" y1="135.0" x2="280.0" y2="135.0" class="stroke-muted" />
<line x1="58.5" y1="131.0" x2="58.5" y2="139.0" class="stroke-muted" />
<text x="58.5" y="155.0" class="fig-small" text-anchor="middle">0</text>
<line x1="150.8" y1="131.0" x2="150.8" y2="139.0" class="stroke-muted" />
<text x="150.8" y="155.0" class="fig-small" text-anchor="middle">0.5</text>
<line x1="206.2" y1="131.0" x2="206.2" y2="139.0" class="stroke-muted" />
<text x="206.2" y="155.0" class="fig-small" text-anchor="middle">0.8</text>
<line x1="243.1" y1="131.0" x2="243.1" y2="139.0" class="stroke-muted" />
<text x="243.1" y="155.0" class="fig-small" text-anchor="middle">1</text>
<circle cx="58.5" cy="135.0" r="5.5" class="fill-a"/>
<path d="M58.5,125 Q150.8,73 240.1,125" fill="none" class="stroke-ink" stroke-width="1.8" marker-end="url(#c016-arrow)"/>
<text x="150.8" y="85.0" class="fig-small" text-anchor="middle">gradient step: 0 + 0.2 × 5 = 1</text>
<circle cx="243.1" cy="135.0" r="5" class="fill-muted"/>
<path d="M243.1,165 Q224.6,181 209.2,165" fill="none" class="stroke-ink" stroke-width="1.8" marker-end="url(#c016-arrow)"/>
<text x="224.6" y="189.0" class="fig-small" text-anchor="middle">soft-threshold by 0.2</text>
<circle cx="206.2" cy="135.0" r="6" class="fill-b"/>
<text x="40.0" y="220.0" class="fig-small">At 0.8 the two pulls cancel, so the step stops:</text>
<line x1="209.2" y1="236" x2="276.2" y2="236" class="stroke-a" stroke-width="3" marker-end="url(#c016-arrow-a)"/>
<line x1="203.2" y1="236" x2="136.2" y2="236" class="stroke-b" stroke-width="3" marker-end="url(#c016-arrow-b)"/>
<text x="242.2" y="255.0" class="fig-small" text-anchor="middle">data +1</text>
<text x="170.2" y="255.0" class="fig-small" text-anchor="middle">fee -1</text>
<text x="340.0" y="24.0" class="fig-bold">FISTA momentum weight, steps 1 to 8</text>
<line x1="340.0" y1="215.0" x2="570.0" y2="215.0" class="stroke-muted" />
<line x1="340.0" y1="215.0" x2="340.0" y2="50.0" class="stroke-muted" />
<line x1="340.0" y1="215.0" x2="570.0" y2="215.0" class="stroke-grid" />
<line x1="336.0" y1="215.0" x2="340.0" y2="215.0" class="stroke-muted" />
<text x="332.0" y="219.0" class="fig-small" text-anchor="end">0</text>
<line x1="340.0" y1="132.5" x2="570.0" y2="132.5" class="stroke-grid" />
<line x1="336.0" y1="132.5" x2="340.0" y2="132.5" class="stroke-muted" />
<text x="332.0" y="136.5" class="fig-small" text-anchor="end">0.5</text>
<line x1="340.0" y1="50.0" x2="570.0" y2="50.0" class="stroke-grid" />
<line x1="336.0" y1="50.0" x2="340.0" y2="50.0" class="stroke-muted" />
<text x="332.0" y="54.0" class="fig-small" text-anchor="end">1</text>
<rect x="342.4" y="215.0" width="24" height="0.0" class="fill-b"/>
<text x="354.4" y="209.0" class="fig-small" text-anchor="middle">0.00</text>
<text x="354.4" y="231.0" class="fig-small" text-anchor="middle">1</text>
<rect x="371.1" y="168.5" width="24" height="46.5" class="fill-b"/>
<text x="383.1" y="162.5" class="fig-small" text-anchor="middle">0.28</text>
<text x="383.1" y="231.0" class="fig-small" text-anchor="middle">2</text>
<rect x="399.9" y="143.4" width="24" height="71.6" class="fill-b"/>
<text x="411.9" y="137.4" class="fig-small" text-anchor="middle">0.43</text>
<text x="411.9" y="231.0" class="fig-small" text-anchor="middle">3</text>
<rect x="428.6" y="127.4" width="24" height="87.6" class="fill-b"/>
<text x="440.6" y="121.4" class="fig-small" text-anchor="middle">0.53</text>
<text x="440.6" y="231.0" class="fig-small" text-anchor="middle">4</text>
<rect x="457.4" y="116.2" width="24" height="98.8" class="fill-b"/>
<text x="469.4" y="110.2" class="fig-small" text-anchor="middle">0.60</text>
<text x="469.4" y="231.0" class="fig-small" text-anchor="middle">5</text>
<rect x="486.1" y="107.9" width="24" height="107.1" class="fill-b"/>
<text x="498.1" y="101.9" class="fig-small" text-anchor="middle">0.65</text>
<text x="498.1" y="231.0" class="fig-small" text-anchor="middle">6</text>
<rect x="514.9" y="101.5" width="24" height="113.5" class="fill-b"/>
<text x="526.9" y="95.5" class="fig-small" text-anchor="middle">0.69</text>
<text x="526.9" y="231.0" class="fig-small" text-anchor="middle">7</text>
<rect x="543.6" y="96.4" width="24" height="118.6" class="fill-b"/>
<text x="555.6" y="90.4" class="fig-small" text-anchor="middle">0.72</text>
<text x="555.6" y="231.0" class="fig-small" text-anchor="middle">8</text>
<text x="455.0" y="250.0" class="fig-small" text-anchor="middle">step k</text>
</svg>
Figure 2. Left: the one ISTA step on the column example, from 0 to 1 by the gradient step and to 0.8 by thresholding. At 0.8 the data pulls up with strength 1 and the fee pulls down with strength 1, so the KKT rule holds. Right: the FISTA momentum weight for the first eight steps, which the next section explains.
```

```python
g = X.T @ (X @ w - y)
breach = np.where(w != 0, np.abs(g + lam * np.sign(w)), np.maximum(np.abs(g) - lam, 0.0))
```

## Add momentum: FISTA

In Figure 1, ISTA sits on the valley floor after one step, and the floor is nearly flat. The step size is set by the steepest wall, so along the floor each step covers little: the curvatures here are about 78 and 0.34, a ratio of 231, so a step covers about 1/231 of the remaining distance. This is the long-valley problem, and momentum is its cure.

FISTA takes each step from a look-ahead point $z$, found by extending the last move. A counter $t$ starts at 1 and sets how far to extend:

$$
t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}, \qquad z_{k+1} = w_{k+1} + \frac{t_k - 1}{t_{k+1}}\,(w_{k+1} - w_k)
$$

Read it as: take the ISTA step from $z$ to get $w_{k+1}$, then push $z$ past $w_{k+1}$ in the direction you just moved, by a fraction $(t_k - 1)/t_{k+1}$ of that move. The fraction starts at 0 and rises towards 1 (right panel of Figure 2: 0, 0.28, 0.43, 0.53, and so on). So momentum builds as the walk keeps going in a consistent direction. The exact recurrence for $t$ is what makes the $1/k^2$ guarantee below work. Any fraction that rises towards 1 gives the same picture.

The column example converges in one step, so use a smaller step of 0.1 to see momentum work. ISTA moves 0, 0.4, 0.6, 0.7, 0.75. FISTA moves 0, 0.4, 0.6, 0.728, 0.792. The first two steps are equal because the fraction is 0 at step 1. By step 3 the look-ahead has pushed FISTA ahead, towards the answer 0.8. By step 5 it has overshot to 0.813. The score shows the same: FISTA's score goes 0.9129, 0.90016, then 0.90041 at step 5. It rose. ISTA's score cannot rise, but FISTA's can, because momentum overshoots. That is acceptable, and it is why you test the finish with the KKT rules, not with "the score stopped falling".

```python
t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
z = w_new + ((t - 1) / t_new) * (w_new - w)     # uses the old w and the old t
```

On the exercise's 40-feature design, with $\lambda = 3$, Figure 3 compares the two. After 80 steps ISTA's score is still 22.6 above its best, and FISTA's is 0.28 above. The guarantees agree: for a convex score and a step of at most $1/L$, the gap shrinks like $1/k$ for ISTA and $1/k^2$ for FISTA. For a gap ten times smaller, ISTA needs about ten times more steps and FISTA about three times more.

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="Score above its best value against step number on a log scale for ISTA and FISTA on a design with correlated columns. FISTA ends well below ISTA, with a few small bumps.">

<line x1="70.0" y1="220.0" x2="450.0" y2="220.0" class="stroke-muted" />
<line x1="70.0" y1="220.0" x2="70.0" y2="45.0" class="stroke-muted" />
<line x1="70.0" y1="220.0" x2="70.0" y2="224.0" class="stroke-muted" />
<text x="70.0" y="236.0" class="fig-small" text-anchor="middle">0</text>
<line x1="165.0" y1="220.0" x2="165.0" y2="224.0" class="stroke-muted" />
<text x="165.0" y="236.0" class="fig-small" text-anchor="middle">20</text>
<line x1="260.0" y1="220.0" x2="260.0" y2="224.0" class="stroke-muted" />
<text x="260.0" y="236.0" class="fig-small" text-anchor="middle">40</text>
<line x1="355.0" y1="220.0" x2="355.0" y2="224.0" class="stroke-muted" />
<text x="355.0" y="236.0" class="fig-small" text-anchor="middle">60</text>
<line x1="450.0" y1="220.0" x2="450.0" y2="224.0" class="stroke-muted" />
<text x="450.0" y="236.0" class="fig-small" text-anchor="middle">80</text>
<line x1="70.0" y1="220.0" x2="450.0" y2="220.0" class="stroke-grid" />
<line x1="66.0" y1="220.0" x2="70.0" y2="220.0" class="stroke-muted" />
<text x="62.0" y="224.0" class="fig-small" text-anchor="end">0.1</text>
<line x1="70.0" y1="185.0" x2="450.0" y2="185.0" class="stroke-grid" />
<line x1="66.0" y1="185.0" x2="70.0" y2="185.0" class="stroke-muted" />
<text x="62.0" y="189.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="150.0" x2="450.0" y2="150.0" class="stroke-grid" />
<line x1="66.0" y1="150.0" x2="70.0" y2="150.0" class="stroke-muted" />
<text x="62.0" y="154.0" class="fig-small" text-anchor="end">10</text>
<line x1="70.0" y1="115.0" x2="450.0" y2="115.0" class="stroke-grid" />
<line x1="66.0" y1="115.0" x2="70.0" y2="115.0" class="stroke-muted" />
<text x="62.0" y="119.0" class="fig-small" text-anchor="end">100</text>
<line x1="70.0" y1="80.0" x2="450.0" y2="80.0" class="stroke-grid" />
<line x1="66.0" y1="80.0" x2="70.0" y2="80.0" class="stroke-muted" />
<text x="62.0" y="84.0" class="fig-small" text-anchor="end">1000</text>
<line x1="70.0" y1="45.0" x2="450.0" y2="45.0" class="stroke-grid" />
<line x1="66.0" y1="45.0" x2="70.0" y2="45.0" class="stroke-muted" />
<text x="62.0" y="49.0" class="fig-small" text-anchor="end">10000</text>
<text x="260.0" y="252.0" class="fig-small" text-anchor="middle">step</text>
<text x="70.0" y="24.0" class="fig-bold">Score above its best value (log scale)</text>
<polyline points="70.0,74.7 74.8,100.0 79.5,115.1 84.2,122.4 89.0,125.3 93.8,126.6 98.5,127.1 103.2,127.4 108.0,127.7 112.8,127.9 117.5,128.0 122.2,128.2 127.0,128.3 131.8,128.5 136.5,128.7 141.2,128.8 146.0,129.0 150.8,129.1 155.5,129.3 160.2,129.4 165.0,129.6 169.8,129.7 174.5,129.9 179.2,130.0 184.0,130.2 188.8,130.3 193.5,130.5 198.2,130.6 203.0,130.8 207.8,130.9 212.5,131.1 217.2,131.2 222.0,131.4 226.8,131.5 231.5,131.7 236.2,131.8 241.0,131.9 245.8,132.1 250.5,132.2 255.2,132.3 260.0,132.5 264.8,132.6 269.5,132.7 274.2,132.9 279.0,133.0 283.8,133.1 288.5,133.3 293.2,133.4 298.0,133.5 302.8,133.7 307.5,133.8 312.2,133.9 317.0,134.1 321.8,134.2 326.5,134.3 331.2,134.5 336.0,134.6 340.8,134.7 345.5,134.8 350.2,135.0 355.0,135.1 359.8,135.2 364.5,135.4 369.2,135.5 374.0,135.6 378.8,135.7 383.5,135.9 388.2,136.0 393.0,136.1 397.8,136.3 402.5,136.4 407.2,136.5 412.0,136.6 416.8,136.8 421.5,136.9 426.2,137.0 431.0,137.1 435.8,137.2 440.5,137.4 445.2,137.5 450.0,137.6" fill="none" class="stroke-d" stroke-width="3"/>
<polyline points="70.0,74.7 74.8,100.0 79.5,115.1 84.2,124.0 89.0,126.8 93.8,127.4 98.5,127.8 103.2,128.2 108.0,128.6 112.8,129.1 117.5,129.6 122.2,130.1 127.0,130.7 131.8,131.3 136.5,131.9 141.2,132.5 146.0,133.2 150.8,133.9 155.5,134.6 160.2,135.4 165.0,136.2 169.8,137.0 174.5,137.9 179.2,138.7 184.0,139.6 188.8,140.6 193.5,141.5 198.2,142.5 203.0,143.4 207.8,144.4 212.5,145.4 217.2,146.3 222.0,147.3 226.8,148.3 231.5,149.4 236.2,150.5 241.0,151.6 245.8,152.7 250.5,153.8 255.2,155.0 260.0,156.1 264.8,157.2 269.5,158.4 274.2,159.6 279.0,160.9 283.8,162.1 288.5,163.4 293.2,164.7 298.0,166.2 302.8,167.5 307.5,168.6 312.2,169.7 317.0,170.9 321.8,172.1 326.5,173.3 331.2,174.6 336.0,176.2 340.8,178.0 345.5,180.0 350.2,181.8 355.0,183.6 359.8,185.4 364.5,187.3 369.2,189.4 374.0,191.3 378.8,193.5 383.5,196.1 388.2,198.6 393.0,199.2 397.8,199.4 402.5,199.4 407.2,198.8 412.0,198.6 416.8,198.7 421.5,199.3 426.2,200.8 431.0,203.2 435.8,206.8 440.5,209.5 445.2,207.1 450.0,204.1" fill="none" class="stroke-b" stroke-width="3"/>
<text x="460.0" y="131.6" class="fig-small fig-bold">ISTA</text>
<text x="460.0" y="146.6" class="fig-small">22.6 above best</text>
<text x="460.0" y="161.6" class="fig-small">(score 37.95)</text>
<text x="460.0" y="180.1" class="fig-small fig-bold">FISTA</text>
<text x="460.0" y="195.1" class="fig-small">0.28 above best</text>
<text x="460.0" y="210.1" class="fig-small">(score 15.66)</text>
<text x="98.5" y="199.0" class="fig-small">Small bumps near steps 71 and 79:</text>
<text x="98.5" y="209.5" class="fig-small">momentum can make the score tick up.</text>
</svg>
Figure 3. How far each method's score is above its best value, on 40 correlated features with lambda 3. After 80 steps FISTA's score is 15.66 and ISTA's is 37.95, a ratio of 0.41. The small bumps on FISTA's curve come from momentum overshooting.
```

> [!KEY]
> The lasso's corner stops you setting the slope to zero, and a plain walk bounces around 0. ISTA splits the job: step down the smooth squared error, then soft-threshold at step times lambda, which writes exact zeros. The KKT rules say when you are done. FISTA adds a growing look-ahead that speeds up the walk along narrow valleys, at the cost of a score that can tick up.

> [!RECAP]
> - The lasso corner has no single slope, and a plain walk on it bounces around 0. Soft thresholding lands on 0 exactly.
> - ISTA: $w \leftarrow S_{\alpha\lambda}(w - \alpha X^\top(Xw - y))$ with $\alpha = 1/L$. On the column example it goes from 0 to 0.8, with score 0.9.
> - KKT: $g_i + \lambda\,\mathrm{sign}(w_i) = 0$ for nonzero weights, and $|g_i| \le \lambda$ for zero weights. The violation is the largest breach, and it is 0 at the answer.
> - FISTA takes the same step from a look-ahead $z$. The counter $t$ starts at 1, and the history records the score at $w$.
> - Momentum helps most on narrow valleys, and FISTA's score may rise.

## Build it step by step

The starter already has `soft_threshold`, `lasso_objective`, and `lipschitz_constant`, which returns the largest eigenvalue of $X^\top X$. You write three functions, in the order the checks test them.

1. `ista(X, y, lam, step, iters, w0=None)` returns `(w, history)`. `w` has shape `(d,)`, and `history` has length `iters + 1` with the starting score at index 0. Copy `w0`, or use zeros, so you never change the caller's array. The most common mistake is to threshold at `lam`, which turns 0.8 into 0 on the column example. The threshold is `step * lam`.

2. `kkt_violation(X, y, w, lam)` returns a float. Compute `g`, choose between the two breaches with `np.where(w != 0, ..., ...)`, and return the largest. The common mistake is to test only `|g| <= lam`, which accepts a nonzero weight whose pull has not cancelled.

3. `fista(X, y, lam, step, iters, w0=None)` returns `(w, history)` with the same shapes. Start `t` at 1 and `z` at a copy of `w`. Take the gradient at `z`, not at `w`, and record the score at `w`. The common mistake is to overwrite `w` and `t` before you form `z`, so that `w_new - w` becomes 0 and momentum disappears. Form `z` first.

## Where this breaks

- **A step above $1/L$.** The guarantee that ISTA's score never rises needs a step of at most $1/L$, and above $2/L$ the walk can blow up. Compute $L$ from the largest eigenvalue of $X^\top X$. Each step costs two matrix–vector products, about $2nd$ operations, and never forms $X^\top X$.
- **A fee that is too large.** When $\lambda$ is at least $\max|X^\top y|$, the answer is all zeros, and both methods return it. On the column example that threshold is 5.
- **Stopping on the score.** FISTA's score can rise, so "the score stopped falling" is not a safe stopping rule. Stop when the KKT violation is small.

## The exercise

You build `ista`, `kkt_violation`, and `fista`. The checks confirm that one ISTA step on the column example goes from 0 to 0.8 and the score from 2.5 to 0.9, that the score never rises at step $1/L$, and that thresholding writes exact zeros. They test `kkt_violation` on the hand values 0 and 4. For FISTA, they check three steps at step 0.1 against 0.728, that after 80 steps its score is under 0.7 times ISTA's, and that 800 steps end with a tiny KKT violation.
