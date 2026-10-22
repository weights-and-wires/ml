---
title: "Why the textbook formula loses digits"
slug: 007-lost-digits
part: "Part 1 — Foundations"
summary: See why nearly repeated columns make weights wobble, and why solving the normal equations throws away about twice as many digits as the data warrant.
minutes: 40
gpu: false
objectives:
  - Show that a tiny wobble in the targets can swing the weights when two columns look alike.
  - Read singular values as the stretches of a matrix, and the condition number as their ratio.
  - Explain why forming X transpose X squares the condition number, and estimate the digits that survive.
  - Solve the normal equations with a Cholesky factor and two triangular solves.
prereqs:
  - 006-residuals-and-projection
  - 005-least-squares-line
exercise: 007-lost-digits
---

# Why the textbook formula loses digits

A house record lists its height twice: once in metres, and once in centimetres, rounded and with a speck of measurement noise. The second column is almost exactly 100 times the first. You hand both columns to the normal equations and get back two weights. Can you trust any of their digits?

You cannot, and the reason is not a bug. When two columns look alike, the data cannot tell "one unit of the first" from "one unit of the second", so the weights can trade off against each other at almost no cost to the fit. A speck of noise then decides where they land. The textbook route makes things worse, because it multiplies the problem by itself before it solves anything.

This chapter shows the swing with real numbers, gives you a name and a picture for how bad it is, and counts the digits you keep. You then write the fast version of the solver, the Cholesky one, and see what it costs.

## What you already know

- The best weights solve $X^\top X w = X^\top y$, the normal equations.
- The residual $y-Xw$ is orthogonal to every column of $X$.
- Solving a 2 by 2 system divides by its determinant, $ad-bc$.
- A double-precision number carries about 16 correct decimal digits.
- A matrix times a vector gives a mix of the matrix's columns.

## Watch the weights swing

Shrink the situation to three houses. The first column is the size, $(1,2,3)$. The second column measures size a second way, $(1,2,3.1)$, which differs from the first only in the last entry. Take the prices $y=(1,2,3)$. The first column predicts them perfectly, so the best weights are $(1,0)$.

Now wobble one price by 0.03, to $y'=(1,2,3.03)$. A wobble that small is about the size of rounding in the measurements. Solve again. The matrix $X^\top X$, the table of dot products between pairs of columns, is called the *Gram matrix*. Here it is

$$
\begin{bmatrix}14&14.3\\14.3&14.61\end{bmatrix}
$$

and its determinant is $14\cdot14.61-14.3^2=204.54-204.49=0.05$. That is tiny next to entries of size 14, and the solve divides by it. For $y'$ you get $X^\top y'=(14.09,\,14.393)$. Then

$$
w_1=\frac{14.61\cdot14.09-14.3\cdot14.393}{0.05}=\frac{205.8549-205.8199}{0.05}=0.7,
$$

and $w_2=(14\cdot14.393-14.3\cdot14.09)/0.05=(201.502-201.487)/0.05=0.3$. The weights moved from $(1,0)$ to $(0.7,0.3)$, so each weight moved by 0.3, ten times the wobble in $y$. The cause is a subtraction of two numbers about 205 that agree to four digits. Those four digits cancel, and the answer lives in what is left. Keep that count of four in mind: it comes back when you count digits.

Figure 1 draws the loss contours for this problem in the weights.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Two contour maps. Left: nearly repeated columns give a very long thin valley, and a wobble of 0.03 in one price moves the best weights from (1, 0) to (0.7, 0.3), along the valley. Right: perpendicular columns give round contours and the best weights barely move.">
<defs>
<clipPath id="c007-clipL"><rect x="20" y="40" width="264" height="204"/></clipPath>
<clipPath id="c007-clipR"><rect x="320" y="40" width="264" height="204"/></clipPath>
</defs>
<rect x="20" y="40" width="264" height="204" class="fill-sunk"/>
<text x="20.0" y="24.0" class="fig-bold">Nearly repeated columns</text>
<g clip-path="url(#c007-clipL)">
<path d="M105.1 96.5 L105.2 96.5 L105.3 96.6 L105.4 96.7 L105.5 96.8 L105.8 97.0 L106.0 97.2 L106.3 97.5 L106.7 97.8 L107.0 98.1 L107.5 98.5 L107.9 99.0 L108.5 99.5 L109.0 100.0 L109.6 100.5 L110.3 101.1 L110.9 101.8 L111.7 102.5 L112.4 103.2 L113.2 104.0 L114.1 104.8 L115.0 105.6 L115.9 106.5 L116.8 107.4 L117.8 108.3 L118.8 109.3 L119.9 110.3 L121.0 111.4 L122.1 112.5 L123.3 113.6 L124.5 114.7 L125.7 115.9 L127.0 117.1 L128.3 118.4 L129.6 119.6 L130.9 120.9 L132.3 122.3 L133.7 123.6 L135.1 125.0 L136.5 126.4 L138.0 127.8 L139.5 129.3 L141.0 130.7 L142.5 132.2 L144.1 133.7 L145.6 135.2 L147.2 136.8 L148.8 138.3 L150.4 139.9 L152.1 141.5 L153.7 143.1 L155.3 144.7 L157.0 146.3 L158.7 147.9 L160.3 149.6 L162.0 151.2 L163.7 152.9 L165.4 154.5 L167.1 156.2 L168.8 157.8 L170.5 159.5 L172.2 161.2 L173.9 162.8 L175.6 164.5 L177.3 166.1 L179.0 167.8 L180.6 169.4 L182.3 171.1 L184.0 172.7 L185.6 174.3 L187.3 175.9 L188.9 177.5 L190.5 179.1 L192.1 180.7 L193.7 182.3 L195.3 183.8 L196.8 185.4 L198.4 186.9 L199.9 188.4 L201.4 189.9 L202.9 191.3 L204.3 192.7 L205.7 194.2 L207.1 195.5 L208.5 196.9 L209.9 198.3 L211.2 199.6 L212.5 200.9 L213.8 202.1 L215.0 203.3 L216.2 204.5 L217.4 205.7 L218.5 206.8 L219.6 208.0 L220.7 209.0 L221.8 210.1 L222.8 211.1 L223.7 212.0 L224.7 213.0 L225.6 213.9 L226.4 214.7 L227.2 215.6 L228.0 216.3 L228.8 217.1 L229.5 217.8 L230.1 218.5 L230.7 219.1 L231.3 219.7 L231.8 220.2 L232.3 220.7 L232.8 221.2 L233.2 221.6 L233.6 222.0 L233.9 222.3 L234.1 222.6 L234.4 222.9 L234.6 223.1 L234.7 223.3 L234.8 223.4 L234.9 223.5 L234.9 223.5 L234.8 223.5 L234.7 223.4 L234.6 223.3 L234.5 223.2 L234.2 223.0 L234.0 222.8 L233.7 222.5 L233.3 222.2 L233.0 221.9 L232.5 221.5 L232.1 221.0 L231.5 220.5 L231.0 220.0 L230.4 219.5 L229.7 218.9 L229.1 218.2 L228.3 217.5 L227.6 216.8 L226.8 216.0 L225.9 215.2 L225.0 214.4 L224.1 213.5 L223.2 212.6 L222.2 211.7 L221.2 210.7 L220.1 209.7 L219.0 208.6 L217.9 207.5 L216.7 206.4 L215.5 205.3 L214.3 204.1 L213.0 202.9 L211.7 201.6 L210.4 200.4 L209.1 199.1 L207.7 197.7 L206.3 196.4 L204.9 195.0 L203.5 193.6 L202.0 192.2 L200.5 190.7 L199.0 189.3 L197.5 187.8 L195.9 186.3 L194.4 184.8 L192.8 183.2 L191.2 181.7 L189.6 180.1 L187.9 178.5 L186.3 176.9 L184.7 175.3 L183.0 173.7 L181.3 172.1 L179.7 170.4 L178.0 168.8 L176.3 167.1 L174.6 165.5 L172.9 163.8 L171.2 162.2 L169.5 160.5 L167.8 158.8 L166.1 157.2 L164.4 155.5 L162.7 153.9 L161.0 152.2 L159.4 150.6 L157.7 148.9 L156.0 147.3 L154.4 145.7 L152.7 144.1 L151.1 142.5 L149.5 140.9 L147.9 139.3 L146.3 137.7 L144.7 136.2 L143.2 134.6 L141.6 133.1 L140.1 131.6 L138.6 130.1 L137.1 128.7 L135.7 127.3 L134.3 125.8 L132.9 124.5 L131.5 123.1 L130.1 121.7 L128.8 120.4 L127.5 119.1 L126.2 117.9 L125.0 116.7 L123.8 115.5 L122.6 114.3 L121.5 113.2 L120.4 112.0 L119.3 111.0 L118.2 109.9 L117.2 108.9 L116.3 108.0 L115.3 107.0 L114.4 106.1 L113.6 105.3 L112.8 104.4 L112.0 103.7 L111.2 102.9 L110.5 102.2 L109.9 101.5 L109.3 100.9 L108.7 100.3 L108.2 99.8 L107.7 99.3 L107.2 98.8 L106.8 98.4 L106.4 98.0 L106.1 97.7 L105.9 97.4 L105.6 97.1 L105.4 96.9 L105.3 96.7 L105.2 96.6 L105.1 96.5 L105.1 96.5" class="stroke-a"/>
<path d="M-35.1 -40.8 L-35.0 -40.8 L-34.8 -40.6 L-34.4 -40.3 L-33.8 -39.9 L-33.2 -39.3 L-32.3 -38.6 L-31.4 -37.7 L-30.3 -36.7 L-29.1 -35.6 L-27.7 -34.4 L-26.2 -33.0 L-24.6 -31.5 L-22.8 -29.8 L-20.9 -28.0 L-18.9 -26.1 L-16.7 -24.1 L-14.5 -21.9 L-12.1 -19.6 L-9.5 -17.2 L-6.9 -14.7 L-4.1 -12.0 L-1.2 -9.3 L1.8 -6.4 L5.0 -3.4 L8.2 -0.3 L11.6 2.9 L15.0 6.3 L18.6 9.7 L22.3 13.2 L26.1 16.9 L29.9 20.6 L33.9 24.5 L38.0 28.4 L42.1 32.4 L46.4 36.5 L50.7 40.7 L55.1 44.9 L59.6 49.3 L64.2 53.7 L68.8 58.2 L73.5 62.8 L78.3 67.4 L83.1 72.1 L88.0 76.9 L93.0 81.7 L98.0 86.5 L103.0 91.5 L108.1 96.4 L113.2 101.4 L118.4 106.5 L123.6 111.6 L128.9 116.7 L134.2 121.8 L139.5 127.0 L144.8 132.2 L150.1 137.4 L155.5 142.6 L160.8 147.9 L166.2 153.1 L171.6 158.4 L176.9 163.7 L182.3 168.9 L187.7 174.2 L193.0 179.4 L198.3 184.6 L203.6 189.8 L208.9 195.0 L214.2 200.2 L219.4 205.3 L224.6 210.4 L229.8 215.5 L234.9 220.5 L239.9 225.5 L245.0 230.5 L249.9 235.4 L254.9 240.2 L259.7 245.0 L264.5 249.7 L269.3 254.4 L273.9 259.0 L278.5 263.5 L283.0 268.0 L287.5 272.4 L291.8 276.7 L296.1 281.0 L300.3 285.1 L304.4 289.2 L308.4 293.2 L312.3 297.1 L316.2 300.8 L319.9 304.5 L323.5 308.1 L327.0 311.6 L330.4 315.0 L333.7 318.3 L336.9 321.5 L339.9 324.6 L342.9 327.5 L345.7 330.4 L348.4 333.1 L351.0 335.7 L353.5 338.2 L355.8 340.5 L358.0 342.8 L360.1 344.9 L362.1 346.9 L363.9 348.7 L365.6 350.5 L367.1 352.1 L368.5 353.5 L369.8 354.9 L371.0 356.1 L372.0 357.1 L372.8 358.1 L373.6 358.9 L374.2 359.5 L374.6 360.0 L374.9 360.4 L375.1 360.7 L375.1 360.8 L375.0 360.8 L374.8 360.6 L374.4 360.3 L373.8 359.9 L373.2 359.3 L372.3 358.6 L371.4 357.7 L370.3 356.7 L369.1 355.6 L367.7 354.4 L366.2 353.0 L364.6 351.5 L362.8 349.8 L360.9 348.0 L358.9 346.1 L356.7 344.1 L354.5 341.9 L352.1 339.6 L349.5 337.2 L346.9 334.7 L344.1 332.0 L341.2 329.3 L338.2 326.4 L335.0 323.4 L331.8 320.3 L328.4 317.1 L325.0 313.7 L321.4 310.3 L317.7 306.8 L313.9 303.1 L310.1 299.4 L306.1 295.5 L302.0 291.6 L297.9 287.6 L293.6 283.5 L289.3 279.3 L284.9 275.1 L280.4 270.7 L275.8 266.3 L271.2 261.8 L266.5 257.2 L261.7 252.6 L256.9 247.9 L252.0 243.1 L247.0 238.3 L242.0 233.5 L237.0 228.5 L231.9 223.6 L226.8 218.6 L221.6 213.5 L216.4 208.4 L211.1 203.3 L205.8 198.2 L200.5 193.0 L195.2 187.8 L189.9 182.6 L184.5 177.4 L179.2 172.1 L173.8 166.9 L168.4 161.6 L163.1 156.3 L157.7 151.1 L152.3 145.8 L147.0 140.6 L141.7 135.4 L136.4 130.2 L131.1 125.0 L125.8 119.8 L120.6 114.7 L115.4 109.6 L110.2 104.5 L105.1 99.5 L100.1 94.5 L95.0 89.5 L90.1 84.6 L85.1 79.8 L80.3 75.0 L75.5 70.3 L70.7 65.6 L66.1 61.0 L61.5 56.5 L57.0 52.0 L52.5 47.6 L48.2 43.3 L43.9 39.0 L39.7 34.9 L35.6 30.8 L31.6 26.8 L27.7 22.9 L23.8 19.2 L20.1 15.5 L16.5 11.9 L13.0 8.4 L9.6 5.0 L6.3 1.7 L3.1 -1.5 L0.1 -4.6 L-2.9 -7.5 L-5.7 -10.4 L-8.4 -13.1 L-11.0 -15.7 L-13.5 -18.2 L-15.8 -20.5 L-18.0 -22.8 L-20.1 -24.9 L-22.1 -26.9 L-23.9 -28.7 L-25.6 -30.5 L-27.1 -32.1 L-28.5 -33.5 L-29.8 -34.9 L-31.0 -36.1 L-32.0 -37.1 L-32.8 -38.1 L-33.6 -38.9 L-34.2 -39.5 L-34.6 -40.0 L-34.9 -40.4 L-35.1 -40.7 L-35.1 -40.8" class="stroke-a"/>
<path d="M-478.6 -475.0 L-478.3 -474.9 L-477.5 -474.4 L-476.3 -473.4 L-474.6 -472.0 L-472.5 -470.2 L-469.9 -467.9 L-466.9 -465.3 L-463.4 -462.1 L-459.6 -458.6 L-455.3 -454.6 L-450.5 -450.3 L-445.4 -445.4 L-439.8 -440.2 L-433.8 -434.6 L-427.4 -428.6 L-420.6 -422.1 L-413.3 -415.3 L-405.7 -408.1 L-397.7 -400.4 L-389.3 -392.4 L-380.5 -384.0 L-371.3 -375.3 L-361.8 -366.2 L-351.8 -356.7 L-341.6 -346.8 L-331.0 -336.6 L-320.0 -326.1 L-308.7 -315.3 L-297.1 -304.1 L-285.2 -292.6 L-272.9 -280.8 L-260.3 -268.6 L-247.5 -256.2 L-234.4 -243.5 L-220.9 -230.6 L-207.3 -217.3 L-193.3 -203.8 L-179.1 -190.1 L-164.7 -176.1 L-150.0 -161.9 L-135.1 -147.4 L-120.1 -132.8 L-104.8 -117.9 L-89.3 -102.9 L-73.6 -87.7 L-57.8 -72.3 L-41.8 -56.7 L-25.7 -41.0 L-9.5 -25.2 L6.9 -9.2 L23.4 6.8 L40.0 23.0 L56.7 39.3 L73.4 55.7 L90.3 72.1 L107.1 88.6 L124.1 105.1 L141.0 121.7 L158.0 138.3 L175.0 154.9 L191.9 171.6 L208.9 188.2 L225.8 204.8 L242.7 221.3 L259.6 237.9 L276.4 254.3 L293.1 270.7 L309.7 287.1 L326.2 303.3 L342.7 319.4 L359.0 335.5 L375.2 351.4 L391.2 367.2 L407.1 382.8 L422.8 398.3 L438.4 413.6 L453.7 428.8 L468.9 443.7 L483.9 458.5 L498.6 473.1 L513.1 487.4 L527.4 501.6 L541.5 515.5 L555.3 529.1 L568.8 542.5 L582.1 555.7 L595.0 568.5 L607.7 581.1 L620.1 593.4 L632.2 605.4 L643.9 617.1 L655.4 628.5 L666.5 639.5 L677.2 650.3 L687.6 660.7 L697.7 670.7 L707.4 680.4 L716.7 689.8 L725.7 698.7 L734.2 707.4 L742.4 715.6 L750.2 723.5 L757.6 730.9 L764.6 738.0 L771.2 744.7 L777.3 751.0 L783.1 756.8 L788.4 762.3 L793.3 767.4 L797.8 772.0 L801.9 776.2 L805.5 780.0 L808.7 783.4 L811.4 786.3 L813.7 788.9 L815.6 791.0 L817.0 792.6 L818.0 793.8 L818.6 794.6 L818.6 795.0 L818.3 794.9 L817.5 794.4 L816.3 793.4 L814.6 792.0 L812.5 790.2 L809.9 787.9 L806.9 785.3 L803.4 782.1 L799.6 778.6 L795.3 774.6 L790.5 770.3 L785.4 765.4 L779.8 760.2 L773.8 754.6 L767.4 748.6 L760.6 742.1 L753.3 735.3 L745.7 728.1 L737.7 720.4 L729.3 712.4 L720.5 704.0 L711.3 695.3 L701.8 686.2 L691.8 676.7 L681.6 666.8 L671.0 656.6 L660.0 646.1 L648.7 635.3 L637.1 624.1 L625.2 612.6 L612.9 600.8 L600.3 588.6 L587.5 576.2 L574.4 563.5 L560.9 550.6 L547.3 537.3 L533.3 523.8 L519.1 510.1 L504.7 496.1 L490.0 481.9 L475.1 467.4 L460.1 452.8 L444.8 437.9 L429.3 422.9 L413.6 407.7 L397.8 392.3 L381.8 376.7 L365.7 361.0 L349.5 345.2 L333.1 329.2 L316.6 313.2 L300.0 297.0 L283.3 280.7 L266.6 264.3 L249.7 247.9 L232.9 231.4 L215.9 214.9 L199.0 198.3 L182.0 181.7 L165.0 165.1 L148.1 148.4 L131.1 131.8 L114.2 115.2 L97.3 98.7 L80.4 82.1 L63.6 65.7 L46.9 49.3 L30.3 32.9 L13.8 16.7 L-2.7 0.6 L-19.0 -15.5 L-35.2 -31.4 L-51.2 -47.2 L-67.1 -62.8 L-82.8 -78.3 L-98.4 -93.6 L-113.7 -108.8 L-128.9 -123.7 L-143.9 -138.5 L-158.6 -153.1 L-173.1 -167.4 L-187.4 -181.6 L-201.5 -195.5 L-215.3 -209.1 L-228.8 -222.5 L-242.1 -235.7 L-255.0 -248.5 L-267.7 -261.1 L-280.1 -273.4 L-292.2 -285.4 L-303.9 -297.1 L-315.4 -308.5 L-326.5 -319.5 L-337.2 -330.3 L-347.6 -340.7 L-357.7 -350.7 L-367.4 -360.4 L-376.7 -369.8 L-385.7 -378.7 L-394.2 -387.4 L-402.4 -395.6 L-410.2 -403.5 L-417.6 -410.9 L-424.6 -418.0 L-431.2 -424.7 L-437.3 -431.0 L-443.1 -436.8 L-448.4 -442.3 L-453.3 -447.4 L-457.8 -452.0 L-461.9 -456.2 L-465.5 -460.0 L-468.7 -463.4 L-471.4 -466.3 L-473.7 -468.9 L-475.6 -471.0 L-477.0 -472.6 L-478.0 -473.8 L-478.6 -474.6 L-478.6 -475.0" class="stroke-a"/>
</g>
<circle cx="170.0" cy="160.0" r="4.5" class="fill-c"/>
<circle cx="134.0" cy="124.0" r="4.5" class="fill-b"/>
<text x="158.0" y="178.0" class="fig-small" text-anchor="end">best for y: (1, 0)</text>
<text x="148.0" y="118.0" class="fig-small">after the wobble:</text>
<text x="148.0" y="134.0" class="fig-small">(0.7, 0.3)</text>
<text x="152.0" y="262.0" class="fig-small" text-anchor="middle">w1 across, w2 up; same scale in both panels</text>
<rect x="320" y="40" width="264" height="204" class="fill-sunk"/>
<text x="320.0" y="24.0" class="fig-bold">Perpendicular columns</text>
<g clip-path="url(#c007-clipR)">
<path d="M451.4 130.7 L451.7 130.7 L452.0 130.8 L452.3 130.8 L452.7 130.8 L453.0 130.8 L453.3 130.9 L453.6 130.9 L453.9 131.0 L454.2 131.1 L454.5 131.1 L454.8 131.2 L455.1 131.3 L455.4 131.4 L455.7 131.5 L456.0 131.6 L456.3 131.8 L456.6 131.9 L456.8 132.0 L457.1 132.2 L457.4 132.3 L457.7 132.5 L457.9 132.7 L458.2 132.8 L458.5 133.0 L458.7 133.2 L459.0 133.4 L459.2 133.6 L459.4 133.8 L459.7 134.0 L459.9 134.2 L460.1 134.5 L460.3 134.7 L460.5 134.9 L460.7 135.2 L460.9 135.4 L461.1 135.7 L461.3 135.9 L461.5 136.2 L461.6 136.5 L461.8 136.7 L461.9 137.0 L462.1 137.3 L462.2 137.6 L462.4 137.9 L462.5 138.1 L462.6 138.4 L462.7 138.7 L462.8 139.0 L462.9 139.3 L463.0 139.6 L463.1 139.9 L463.1 140.2 L463.2 140.5 L463.3 140.9 L463.3 141.2 L463.3 141.5 L463.4 141.8 L463.4 142.1 L463.4 142.4 L463.4 142.7 L463.4 143.0 L463.4 143.4 L463.4 143.7 L463.3 144.0 L463.3 144.3 L463.3 144.6 L463.2 144.9 L463.1 145.2 L463.1 145.5 L463.0 145.8 L462.9 146.1 L462.8 146.4 L462.7 146.7 L462.6 147.0 L462.5 147.3 L462.4 147.6 L462.2 147.9 L462.1 148.2 L461.9 148.5 L461.8 148.7 L461.6 149.0 L461.5 149.3 L461.3 149.5 L461.1 149.8 L460.9 150.0 L460.7 150.3 L460.5 150.5 L460.3 150.8 L460.1 151.0 L459.9 151.2 L459.7 151.4 L459.4 151.7 L459.2 151.9 L459.0 152.1 L458.7 152.3 L458.5 152.4 L458.2 152.6 L457.9 152.8 L457.7 153.0 L457.4 153.1 L457.1 153.3 L456.8 153.4 L456.6 153.6 L456.3 153.7 L456.0 153.8 L455.7 153.9 L455.4 154.0 L455.1 154.1 L454.8 154.2 L454.5 154.3 L454.2 154.4 L453.9 154.5 L453.6 154.5 L453.3 154.6 L453.0 154.6 L452.7 154.7 L452.3 154.7 L452.0 154.7 L451.7 154.7 L451.4 154.7 L451.1 154.7 L450.8 154.7 L450.5 154.7 L450.1 154.7 L449.8 154.6 L449.5 154.6 L449.2 154.5 L448.9 154.5 L448.6 154.4 L448.3 154.3 L448.0 154.2 L447.7 154.1 L447.4 154.0 L447.1 153.9 L446.8 153.8 L446.5 153.7 L446.2 153.6 L446.0 153.4 L445.7 153.3 L445.4 153.1 L445.1 153.0 L444.9 152.8 L444.6 152.6 L444.3 152.4 L444.1 152.3 L443.8 152.1 L443.6 151.9 L443.4 151.7 L443.1 151.4 L442.9 151.2 L442.7 151.0 L442.5 150.8 L442.3 150.5 L442.1 150.3 L441.9 150.0 L441.7 149.8 L441.5 149.5 L441.3 149.3 L441.2 149.0 L441.0 148.7 L440.9 148.5 L440.7 148.2 L440.6 147.9 L440.4 147.6 L440.3 147.3 L440.2 147.0 L440.1 146.7 L440.0 146.4 L439.9 146.1 L439.8 145.8 L439.7 145.5 L439.7 145.2 L439.6 144.9 L439.5 144.6 L439.5 144.3 L439.5 144.0 L439.4 143.7 L439.4 143.4 L439.4 143.0 L439.4 142.7 L439.4 142.4 L439.4 142.1 L439.4 141.8 L439.5 141.5 L439.5 141.2 L439.5 140.9 L439.6 140.5 L439.7 140.2 L439.7 139.9 L439.8 139.6 L439.9 139.3 L440.0 139.0 L440.1 138.7 L440.2 138.4 L440.3 138.1 L440.4 137.9 L440.6 137.6 L440.7 137.3 L440.9 137.0 L441.0 136.7 L441.2 136.5 L441.3 136.2 L441.5 135.9 L441.7 135.7 L441.9 135.4 L442.1 135.2 L442.3 134.9 L442.5 134.7 L442.7 134.5 L442.9 134.2 L443.1 134.0 L443.4 133.8 L443.6 133.6 L443.8 133.4 L444.1 133.2 L444.3 133.0 L444.6 132.8 L444.9 132.7 L445.1 132.5 L445.4 132.3 L445.7 132.2 L446.0 132.0 L446.2 131.9 L446.5 131.8 L446.8 131.6 L447.1 131.5 L447.4 131.4 L447.7 131.3 L448.0 131.2 L448.3 131.1 L448.6 131.1 L448.9 131.0 L449.2 130.9 L449.5 130.9 L449.8 130.8 L450.1 130.8 L450.5 130.8 L450.8 130.8 L451.1 130.7 L451.4 130.7" class="stroke-a"/>
<path d="M451.4 118.7 L452.0 118.7 L452.7 118.8 L453.3 118.8 L453.9 118.9 L454.5 118.9 L455.2 119.0 L455.8 119.1 L456.4 119.3 L457.0 119.4 L457.6 119.6 L458.2 119.7 L458.8 119.9 L459.4 120.1 L460.0 120.3 L460.6 120.6 L461.2 120.8 L461.7 121.1 L462.3 121.4 L462.9 121.6 L463.4 122.0 L463.9 122.3 L464.5 122.6 L465.0 123.0 L465.5 123.3 L466.0 123.7 L466.5 124.1 L467.0 124.5 L467.5 124.9 L467.9 125.3 L468.4 125.8 L468.8 126.2 L469.2 126.7 L469.6 127.1 L470.1 127.6 L470.4 128.1 L470.8 128.6 L471.2 129.1 L471.5 129.7 L471.9 130.2 L472.2 130.7 L472.5 131.3 L472.8 131.8 L473.1 132.4 L473.3 133.0 L473.6 133.6 L473.8 134.1 L474.0 134.7 L474.2 135.3 L474.4 135.9 L474.6 136.5 L474.7 137.1 L474.9 137.7 L475.0 138.4 L475.1 139.0 L475.2 139.6 L475.3 140.2 L475.3 140.9 L475.4 141.5 L475.4 142.1 L475.4 142.7 L475.4 143.4 L475.4 144.0 L475.3 144.6 L475.3 145.2 L475.2 145.9 L475.1 146.5 L475.0 147.1 L474.9 147.7 L474.7 148.3 L474.6 148.9 L474.4 149.6 L474.2 150.2 L474.0 150.7 L473.8 151.3 L473.6 151.9 L473.3 152.5 L473.1 153.1 L472.8 153.6 L472.5 154.2 L472.2 154.7 L471.9 155.3 L471.5 155.8 L471.2 156.3 L470.8 156.8 L470.4 157.3 L470.1 157.8 L469.6 158.3 L469.2 158.8 L468.8 159.3 L468.4 159.7 L467.9 160.1 L467.5 160.6 L467.0 161.0 L466.5 161.4 L466.0 161.8 L465.5 162.2 L465.0 162.5 L464.5 162.9 L463.9 163.2 L463.4 163.5 L462.9 163.8 L462.3 164.1 L461.7 164.4 L461.2 164.7 L460.6 164.9 L460.0 165.1 L459.4 165.4 L458.8 165.6 L458.2 165.7 L457.6 165.9 L457.0 166.1 L456.4 166.2 L455.8 166.3 L455.2 166.4 L454.5 166.5 L453.9 166.6 L453.3 166.7 L452.7 166.7 L452.0 166.7 L451.4 166.7 L450.8 166.7 L450.1 166.7 L449.5 166.7 L448.9 166.6 L448.3 166.5 L447.6 166.4 L447.0 166.3 L446.4 166.2 L445.8 166.1 L445.2 165.9 L444.6 165.7 L444.0 165.6 L443.4 165.4 L442.8 165.1 L442.2 164.9 L441.6 164.7 L441.1 164.4 L440.5 164.1 L439.9 163.8 L439.4 163.5 L438.9 163.2 L438.3 162.9 L437.8 162.5 L437.3 162.2 L436.8 161.8 L436.3 161.4 L435.8 161.0 L435.3 160.6 L434.9 160.1 L434.4 159.7 L434.0 159.3 L433.6 158.8 L433.2 158.3 L432.7 157.8 L432.4 157.3 L432.0 156.8 L431.6 156.3 L431.3 155.8 L430.9 155.3 L430.6 154.7 L430.3 154.2 L430.0 153.6 L429.7 153.1 L429.5 152.5 L429.2 151.9 L429.0 151.3 L428.8 150.7 L428.6 150.2 L428.4 149.6 L428.2 148.9 L428.1 148.3 L427.9 147.7 L427.8 147.1 L427.7 146.5 L427.6 145.9 L427.5 145.2 L427.5 144.6 L427.4 144.0 L427.4 143.4 L427.4 142.7 L427.4 142.1 L427.4 141.5 L427.5 140.9 L427.5 140.2 L427.6 139.6 L427.7 139.0 L427.8 138.4 L427.9 137.7 L428.1 137.1 L428.2 136.5 L428.4 135.9 L428.6 135.3 L428.8 134.7 L429.0 134.1 L429.2 133.6 L429.5 133.0 L429.7 132.4 L430.0 131.8 L430.3 131.3 L430.6 130.7 L430.9 130.2 L431.3 129.7 L431.6 129.1 L432.0 128.6 L432.4 128.1 L432.7 127.6 L433.2 127.1 L433.6 126.7 L434.0 126.2 L434.4 125.8 L434.9 125.3 L435.3 124.9 L435.8 124.5 L436.3 124.1 L436.8 123.7 L437.3 123.3 L437.8 123.0 L438.3 122.6 L438.9 122.3 L439.4 122.0 L439.9 121.6 L440.5 121.4 L441.1 121.1 L441.6 120.8 L442.2 120.6 L442.8 120.3 L443.4 120.1 L444.0 119.9 L444.6 119.7 L445.2 119.6 L445.8 119.4 L446.4 119.3 L447.0 119.1 L447.6 119.0 L448.3 118.9 L448.9 118.9 L449.5 118.8 L450.1 118.8 L450.8 118.7 L451.4 118.7" class="stroke-a"/>
<path d="M451.4 106.7 L452.3 106.7 L453.3 106.8 L454.2 106.8 L455.2 106.9 L456.1 107.0 L457.0 107.2 L458.0 107.3 L458.9 107.5 L459.8 107.7 L460.7 108.0 L461.6 108.2 L462.5 108.5 L463.4 108.8 L464.3 109.1 L465.2 109.5 L466.0 109.8 L466.9 110.2 L467.7 110.7 L468.6 111.1 L469.4 111.6 L470.2 112.0 L471.0 112.5 L471.8 113.1 L472.6 113.6 L473.3 114.2 L474.1 114.8 L474.8 115.4 L475.5 116.0 L476.2 116.6 L476.9 117.3 L477.5 118.0 L478.2 118.6 L478.8 119.4 L479.4 120.1 L480.0 120.8 L480.5 121.6 L481.1 122.3 L481.6 123.1 L482.1 123.9 L482.6 124.7 L483.0 125.6 L483.5 126.4 L483.9 127.2 L484.3 128.1 L484.7 129.0 L485.0 129.8 L485.3 130.7 L485.6 131.6 L485.9 132.5 L486.2 133.4 L486.4 134.3 L486.6 135.3 L486.8 136.2 L487.0 137.1 L487.1 138.0 L487.2 139.0 L487.3 139.9 L487.4 140.9 L487.4 141.8 L487.4 142.7 L487.4 143.7 L487.4 144.6 L487.3 145.6 L487.2 146.5 L487.1 147.4 L487.0 148.4 L486.8 149.3 L486.6 150.2 L486.4 151.1 L486.2 152.1 L485.9 153.0 L485.6 153.9 L485.3 154.8 L485.0 155.6 L484.7 156.5 L484.3 157.4 L483.9 158.2 L483.5 159.1 L483.0 159.9 L482.6 160.7 L482.1 161.5 L481.6 162.3 L481.1 163.1 L480.5 163.9 L480.0 164.7 L479.4 165.4 L478.8 166.1 L478.2 166.8 L477.5 167.5 L476.9 168.2 L476.2 168.8 L475.5 169.5 L474.8 170.1 L474.1 170.7 L473.3 171.3 L472.6 171.9 L471.8 172.4 L471.0 172.9 L470.2 173.4 L469.4 173.9 L468.6 174.4 L467.7 174.8 L466.9 175.2 L466.0 175.6 L465.2 176.0 L464.3 176.3 L463.4 176.7 L462.5 177.0 L461.6 177.3 L460.7 177.5 L459.8 177.7 L458.9 177.9 L458.0 178.1 L457.0 178.3 L456.1 178.4 L455.2 178.5 L454.2 178.6 L453.3 178.7 L452.3 178.7 L451.4 178.7 L450.5 178.7 L449.5 178.7 L448.6 178.6 L447.6 178.5 L446.7 178.4 L445.8 178.3 L444.8 178.1 L443.9 177.9 L443.0 177.7 L442.1 177.5 L441.2 177.3 L440.3 177.0 L439.4 176.7 L438.5 176.3 L437.6 176.0 L436.8 175.6 L435.9 175.2 L435.1 174.8 L434.2 174.4 L433.4 173.9 L432.6 173.4 L431.8 172.9 L431.0 172.4 L430.2 171.9 L429.5 171.3 L428.7 170.7 L428.0 170.1 L427.3 169.5 L426.6 168.8 L425.9 168.2 L425.3 167.5 L424.6 166.8 L424.0 166.1 L423.4 165.4 L422.8 164.7 L422.3 163.9 L421.7 163.1 L421.2 162.3 L420.7 161.5 L420.2 160.7 L419.8 159.9 L419.3 159.1 L418.9 158.2 L418.5 157.4 L418.1 156.5 L417.8 155.6 L417.5 154.8 L417.2 153.9 L416.9 153.0 L416.6 152.1 L416.4 151.1 L416.2 150.2 L416.0 149.3 L415.8 148.4 L415.7 147.4 L415.6 146.5 L415.5 145.6 L415.4 144.6 L415.4 143.7 L415.4 142.7 L415.4 141.8 L415.4 140.9 L415.5 139.9 L415.6 139.0 L415.7 138.0 L415.8 137.1 L416.0 136.2 L416.2 135.3 L416.4 134.3 L416.6 133.4 L416.9 132.5 L417.2 131.6 L417.5 130.7 L417.8 129.8 L418.1 129.0 L418.5 128.1 L418.9 127.2 L419.3 126.4 L419.8 125.6 L420.2 124.7 L420.7 123.9 L421.2 123.1 L421.7 122.3 L422.3 121.6 L422.8 120.8 L423.4 120.1 L424.0 119.4 L424.6 118.6 L425.3 118.0 L425.9 117.3 L426.6 116.6 L427.3 116.0 L428.0 115.4 L428.7 114.8 L429.5 114.2 L430.2 113.6 L431.0 113.1 L431.8 112.5 L432.6 112.0 L433.4 111.6 L434.2 111.1 L435.1 110.7 L435.9 110.2 L436.8 109.8 L437.6 109.5 L438.5 109.1 L439.4 108.8 L440.3 108.5 L441.2 108.2 L442.1 108.0 L443.0 107.7 L443.9 107.5 L444.8 107.3 L445.8 107.2 L446.7 107.0 L447.6 106.9 L448.6 106.8 L449.5 106.8 L450.5 106.7 L451.4 106.7" class="stroke-a"/>
</g>
<circle cx="451.4" cy="142.7" r="4.5" class="fill-c"/>
<circle cx="452.6" cy="141.3" r="4.5" class="fill-b"/>
<text x="451.4" y="222.7" class="fig-small" text-anchor="middle">the two minima almost coincide</text>
<text x="452.0" y="262.0" class="fig-small" text-anchor="middle">w1 across, w2 up; same scale in both panels</text>
</svg>
Figure 1. Left: for nearly repeated columns the contours are a very long thin valley, and the two best weight vectors sit far apart along it. Right: for perpendicular columns the contours are round, and the same wobble hardly moves the minimum. Notice that the left valley is almost a line, so many weights fit almost equally well.
```

The weights $(0.7,0.3)$ fit the original prices almost as well as $(1,0)$: the loss rises only from 0 to $0.00045$. The data say there is a long strip of nearly equally good weights, so tiny noise slides the answer along the strip.

## Measure the squash

To name this effect, ask what the matrix does to weights. Take every weight vector of length 1, a circle, and multiply by $X$. The result is an ellipse, and the matrix is characterised by how far it stretches in its longest direction and its shortest. These two stretches are the *singular values*, written $\sigma_{\max}$ and $\sigma_{\min}$ ($\sigma$ is the Greek letter sigma).

Figure 2 shows the ellipses for the two matrices of this chapter.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Two ellipses. The matrix of the three houses turns the unit circle into an ellipse with semi-axes 4.08 and 0.60. The nearly repeated columns turn it into an almost flat line with semi-axes 5.35 and 0.04.">
<text x="15.0" y="22.0" class="fig-bold fig-small">Three houses: columns (1,1,1) and (1,2,3)</text>
<line x1="10.0" y1="105.0" x2="290.0" y2="105.0" class="stroke-grid"/>
<line x1="150.0" y1="35.0" x2="150.0" y2="175.0" class="stroke-grid"/>
<path d="M174.0 105.0 L174.0 103.7 L173.9 102.5 L173.7 101.2 L173.5 100.0 L173.2 98.8 L172.8 97.6 L172.4 96.4 L171.9 95.2 L171.4 94.1 L170.8 93.0 L170.1 91.9 L169.4 90.9 L168.7 89.9 L167.8 88.9 L167.0 88.0 L166.1 87.2 L165.1 86.3 L164.1 85.6 L163.1 84.9 L162.0 84.2 L160.9 83.6 L159.8 83.1 L158.6 82.6 L157.4 82.2 L156.2 81.8 L155.0 81.5 L153.8 81.3 L152.5 81.1 L151.3 81.0 L150.0 81.0 L148.7 81.0 L147.5 81.1 L146.2 81.3 L145.0 81.5 L143.8 81.8 L142.6 82.2 L141.4 82.6 L140.2 83.1 L139.1 83.6 L138.0 84.2 L136.9 84.9 L135.9 85.6 L134.9 86.3 L133.9 87.2 L133.0 88.0 L132.2 88.9 L131.3 89.9 L130.6 90.9 L129.9 91.9 L129.2 93.0 L128.6 94.1 L128.1 95.2 L127.6 96.4 L127.2 97.6 L126.8 98.8 L126.5 100.0 L126.3 101.2 L126.1 102.5 L126.0 103.7 L126.0 105.0 L126.0 106.3 L126.1 107.5 L126.3 108.8 L126.5 110.0 L126.8 111.2 L127.2 112.4 L127.6 113.6 L128.1 114.8 L128.6 115.9 L129.2 117.0 L129.9 118.1 L130.6 119.1 L131.3 120.1 L132.2 121.1 L133.0 122.0 L133.9 122.8 L134.9 123.7 L135.9 124.4 L136.9 125.1 L138.0 125.8 L139.1 126.4 L140.2 126.9 L141.4 127.4 L142.6 127.8 L143.8 128.2 L145.0 128.5 L146.2 128.7 L147.5 128.9 L148.7 129.0 L150.0 129.0 L151.3 129.0 L152.5 128.9 L153.8 128.7 L155.0 128.5 L156.2 128.2 L157.4 127.8 L158.6 127.4 L159.8 126.9 L160.9 126.4 L162.0 125.8 L163.1 125.1 L164.1 124.4 L165.1 123.7 L166.1 122.8 L167.0 122.0 L167.8 121.1 L168.7 120.1 L169.4 119.1 L170.1 118.1 L170.8 117.0 L171.4 115.9 L171.9 114.8 L172.4 113.6 L172.8 112.4 L173.2 111.2 L173.5 110.0 L173.7 108.8 L173.9 107.5 L174.0 106.3 L174.0 105.0 Z" class="stroke-muted dashed"/>
<path d="M247.9 105.0 L247.8 104.2 L247.4 103.5 L246.7 102.7 L245.8 102.0 L244.6 101.3 L243.1 100.5 L241.4 99.8 L239.4 99.1 L237.2 98.5 L234.8 97.8 L232.1 97.2 L229.2 96.5 L226.1 95.9 L222.8 95.4 L219.2 94.8 L215.5 94.3 L211.6 93.8 L207.5 93.3 L203.3 92.9 L198.9 92.5 L194.4 92.2 L189.8 91.8 L185.1 91.5 L180.3 91.3 L175.3 91.1 L170.4 90.9 L165.3 90.8 L160.2 90.7 L155.1 90.6 L150.0 90.6 L144.9 90.6 L139.8 90.7 L134.7 90.8 L129.6 90.9 L124.7 91.1 L119.7 91.3 L114.9 91.5 L110.2 91.8 L105.6 92.2 L101.1 92.5 L96.7 92.9 L92.5 93.3 L88.4 93.8 L84.5 94.3 L80.8 94.8 L77.2 95.4 L73.9 95.9 L70.8 96.5 L67.9 97.2 L65.2 97.8 L62.8 98.5 L60.6 99.1 L58.6 99.8 L56.9 100.5 L55.4 101.3 L54.2 102.0 L53.3 102.7 L52.6 103.5 L52.2 104.2 L52.1 105.0 L52.2 105.8 L52.6 106.5 L53.3 107.3 L54.2 108.0 L55.4 108.7 L56.9 109.5 L58.6 110.2 L60.6 110.9 L62.8 111.5 L65.2 112.2 L67.9 112.8 L70.8 113.5 L73.9 114.1 L77.2 114.6 L80.8 115.2 L84.5 115.7 L88.4 116.2 L92.5 116.7 L96.7 117.1 L101.1 117.5 L105.6 117.8 L110.2 118.2 L114.9 118.5 L119.7 118.7 L124.7 118.9 L129.6 119.1 L134.7 119.2 L139.8 119.3 L144.9 119.4 L150.0 119.4 L155.1 119.4 L160.2 119.3 L165.3 119.2 L170.4 119.1 L175.3 118.9 L180.3 118.7 L185.1 118.5 L189.8 118.2 L194.4 117.8 L198.9 117.5 L203.3 117.1 L207.5 116.7 L211.6 116.2 L215.5 115.7 L219.2 115.2 L222.8 114.6 L226.1 114.1 L229.2 113.5 L232.1 112.8 L234.8 112.2 L237.2 111.5 L239.4 110.9 L241.4 110.2 L243.1 109.5 L244.6 108.7 L245.8 108.0 L246.7 107.3 L247.4 106.5 L247.8 105.8 L247.9 105.0 Z" class="fill-soft-b stroke-b"/>

<text x="150.0" y="193.0" class="fig-small" text-anchor="middle">ratio 4.079 / 0.600 = 6.8</text>
<text x="250.0" y="126.0" class="fig-small" text-anchor="end">long 4.079</text>
<line x1="150.0" y1="120.0" x2="150.0" y2="140.0" class="stroke-muted"/>
<text x="150.0" y="153.0" class="fig-small" text-anchor="middle">short 0.600</text>

<text x="315.0" y="22.0" class="fig-bold fig-small">Nearly repeated: (1,2,3) and (1,2,3.1)</text>
<line x1="310.0" y1="105.0" x2="590.0" y2="105.0" class="stroke-grid"/>
<line x1="450.0" y1="35.0" x2="450.0" y2="175.0" class="stroke-grid"/>
<path d="M474.0 105.0 L474.0 103.7 L473.9 102.5 L473.7 101.2 L473.5 100.0 L473.2 98.8 L472.8 97.6 L472.4 96.4 L471.9 95.2 L471.4 94.1 L470.8 93.0 L470.1 91.9 L469.4 90.9 L468.7 89.9 L467.8 88.9 L467.0 88.0 L466.1 87.2 L465.1 86.3 L464.1 85.6 L463.1 84.9 L462.0 84.2 L460.9 83.6 L459.8 83.1 L458.6 82.6 L457.4 82.2 L456.2 81.8 L455.0 81.5 L453.8 81.3 L452.5 81.1 L451.3 81.0 L450.0 81.0 L448.7 81.0 L447.5 81.1 L446.2 81.3 L445.0 81.5 L443.8 81.8 L442.6 82.2 L441.4 82.6 L440.2 83.1 L439.1 83.6 L438.0 84.2 L436.9 84.9 L435.9 85.6 L434.9 86.3 L433.9 87.2 L433.0 88.0 L432.2 88.9 L431.3 89.9 L430.6 90.9 L429.9 91.9 L429.2 93.0 L428.6 94.1 L428.1 95.2 L427.6 96.4 L427.2 97.6 L426.8 98.8 L426.5 100.0 L426.3 101.2 L426.1 102.5 L426.0 103.7 L426.0 105.0 L426.0 106.3 L426.1 107.5 L426.3 108.8 L426.5 110.0 L426.8 111.2 L427.2 112.4 L427.6 113.6 L428.1 114.8 L428.6 115.9 L429.2 117.0 L429.9 118.1 L430.6 119.1 L431.3 120.1 L432.2 121.1 L433.0 122.0 L433.9 122.8 L434.9 123.7 L435.9 124.4 L436.9 125.1 L438.0 125.8 L439.1 126.4 L440.2 126.9 L441.4 127.4 L442.6 127.8 L443.8 128.2 L445.0 128.5 L446.2 128.7 L447.5 128.9 L448.7 129.0 L450.0 129.0 L451.3 129.0 L452.5 128.9 L453.8 128.7 L455.0 128.5 L456.2 128.2 L457.4 127.8 L458.6 127.4 L459.8 126.9 L460.9 126.4 L462.0 125.8 L463.1 125.1 L464.1 124.4 L465.1 123.7 L466.1 122.8 L467.0 122.0 L467.8 121.1 L468.7 120.1 L469.4 119.1 L470.1 118.1 L470.8 117.0 L471.4 115.9 L471.9 114.8 L472.4 113.6 L472.8 112.4 L473.2 111.2 L473.5 110.0 L473.7 108.8 L473.9 107.5 L474.0 106.3 L474.0 105.0 Z" class="stroke-muted dashed"/>
<path d="M578.4 105.0 L578.2 104.9 L577.7 104.9 L576.8 104.8 L575.6 104.8 L574.0 104.7 L572.1 104.7 L569.8 104.6 L567.3 104.6 L564.4 104.5 L561.2 104.5 L557.7 104.5 L553.9 104.4 L549.8 104.4 L545.4 104.3 L540.8 104.3 L535.9 104.3 L530.8 104.2 L525.5 104.2 L519.9 104.2 L514.2 104.1 L508.3 104.1 L502.2 104.1 L496.0 104.1 L489.7 104.0 L483.2 104.0 L476.7 104.0 L470.1 104.0 L463.4 104.0 L456.7 104.0 L450.0 104.0 L443.3 104.0 L436.6 104.0 L429.9 104.0 L423.3 104.0 L416.8 104.0 L410.3 104.0 L404.0 104.1 L397.8 104.1 L391.7 104.1 L385.8 104.1 L380.1 104.2 L374.5 104.2 L369.2 104.2 L364.1 104.3 L359.2 104.3 L354.6 104.3 L350.2 104.4 L346.1 104.4 L342.3 104.5 L338.8 104.5 L335.6 104.5 L332.7 104.6 L330.2 104.6 L327.9 104.7 L326.0 104.7 L324.4 104.8 L323.2 104.8 L322.3 104.9 L321.8 104.9 L321.6 105.0 L321.8 105.1 L322.3 105.1 L323.2 105.2 L324.4 105.2 L326.0 105.3 L327.9 105.3 L330.2 105.4 L332.7 105.4 L335.6 105.5 L338.8 105.5 L342.3 105.5 L346.1 105.6 L350.2 105.6 L354.6 105.7 L359.2 105.7 L364.1 105.7 L369.2 105.8 L374.5 105.8 L380.1 105.8 L385.8 105.9 L391.7 105.9 L397.8 105.9 L404.0 105.9 L410.3 106.0 L416.8 106.0 L423.3 106.0 L429.9 106.0 L436.6 106.0 L443.3 106.0 L450.0 106.0 L456.7 106.0 L463.4 106.0 L470.1 106.0 L476.7 106.0 L483.2 106.0 L489.7 106.0 L496.0 105.9 L502.2 105.9 L508.3 105.9 L514.2 105.9 L519.9 105.8 L525.5 105.8 L530.8 105.8 L535.9 105.7 L540.8 105.7 L545.4 105.7 L549.8 105.6 L553.9 105.6 L557.7 105.5 L561.2 105.5 L564.4 105.5 L567.3 105.4 L569.8 105.4 L572.1 105.3 L574.0 105.3 L575.6 105.2 L576.8 105.2 L577.7 105.1 L578.2 105.1 L578.4 105.0 Z" class="fill-soft-b stroke-b"/>

<text x="450.0" y="193.0" class="fig-small" text-anchor="middle">ratio 5.349 / 0.042 = 128</text>
<text x="580.0" y="126.0" class="fig-small" text-anchor="end">long 5.349</text>
<line x1="450.0" y1="108.0" x2="450.0" y2="140.0" class="stroke-muted"/>
<text x="450.0" y="153.0" class="fig-small" text-anchor="middle">short 0.042</text>

<text x="300.0" y="222.0" class="fig-small" text-anchor="middle">dashed circle: all weight vectors of length 1</text>
</svg>
Figure 2. The unit circle (dashed) turned by each matrix. Three houses give a moderately flat ellipse, with a ratio of 7. Nearly repeated columns give an almost flat one, with a ratio of 128. Notice that the short axis comes from the weight direction that barely changes the guesses.
```

The short axis explains the valley. A weight direction that barely changes the guesses is a direction the data cannot pin down. The ratio of the two stretches is the *condition number*:

$$
\kappa=\frac{\sigma_{\max}}{\sigma_{\min}}
$$

Read it aloud: how many times longer the long axis is than the short one. A wobble of size $\varepsilon$ in $y$ can move the weights by up to $\varepsilon/\sigma_{\min}$. The reason is the short axis: a weight step of length 1 can change the guesses by as little as $\sigma_{\min}$, so to produce a change of $\varepsilon$ in the guesses the weights may need to move $\varepsilon/\sigma_{\min}$. For the nearly repeated columns, $0.03/0.0418\approx0.72$, and the actual move was $\sqrt{0.3^2+0.3^2}\approx0.42$.

You find the stretches by hand from the Gram matrix. Its two eigenvalues, the stretching factors of $X^\top X$, are $\sigma^2$ for each $\sigma$. For the three houses $X^\top X=\begin{bmatrix}3&6\\6&14\end{bmatrix}$. The eigenvalues add to the trace, $3+14=17$, and multiply to the determinant, $3\cdot14-36=6$. For a 2 by 2 matrix the two eigenvalues are the roots of $t^2-(\text{trace})\,t+\text{determinant}=0$, so here they solve $t^2-17t+6=0$ and $t=(17\pm\sqrt{265})/2\approx16.639$ and $0.361$. Square roots give $\sigma_{\max}\approx4.079$ and $\sigma_{\min}\approx0.600$, so $\kappa\approx6.79$. In numpy:

```python
s = np.linalg.svd(X, compute_uv=False)  # largest first
kappa = s[0] / s[-1]
```

Dividing the small value by the large one gives 0.147, and a condition number is never below 1. If $\sigma_{\min}$ is exactly 0, return `float("inf")`.

## Count the digits you lose

Now the textbook formula. It does not solve with $X$. It solves with $X^\top X$. Multiplying by $X$ stretches by the singular values, and multiplying by $X^\top$ stretches again by the same amounts along the same axes. So the singular values of $X^\top X$ are $\sigma_{\max}^2$ and $\sigma_{\min}^2$, and its condition number is $\kappa^2$.

Check the numbers. For the three houses $\kappa^2\approx6.79^2=46.1$, and the Gram matrix $16.639/0.361\approx46.1$. For the nearly repeated columns $\kappa\approx127.9$ and $\kappa^2\approx16\,400$. That is about $10^{4.2}$, the four cancelled digits from the weight swing: squaring the condition number is the cancellation you saw.

A rule of thumb connects $\kappa$ to digits. Each factor of 10 in the condition number lets a rounding error in the data grow by a factor of 10 in the weights, which pushes one more digit of the answer into noise. A problem with condition number $10^k$ can lose about $k$ of the 16 digits a double carries. It is a rough estimate of the worst case, not a guarantee. Solving with $X$ at $\kappa=10^7$ keeps about $16-7=9$ digits. Solving with $X^\top X$ has condition $10^{14}$ and keeps about $16-14=2$.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Digits kept out of 16 against the base-10 logarithm of the condition number. Solving with X loses one digit per power of ten. Solving with X transpose X loses two. At a condition number of 10 to the 7th, 9 digits survive against 2.">
<line x1="55.0" y1="245.0" x2="555.0" y2="245.0" class="stroke-grid"/>
<text x="47.0" y="249.0" class="fig-small" text-anchor="end">0</text>
<line x1="55.0" y1="245.0" x2="55.0" y2="15.0" class="stroke-grid"/>
<text x="55.0" y="261.0" class="fig-small" text-anchor="middle">0</text>
<line x1="55.0" y1="187.5" x2="555.0" y2="187.5" class="stroke-grid"/>
<text x="47.0" y="191.5" class="fig-small" text-anchor="end">4</text>
<line x1="180.0" y1="245.0" x2="180.0" y2="15.0" class="stroke-grid"/>
<text x="180.0" y="261.0" class="fig-small" text-anchor="middle">4</text>
<line x1="55.0" y1="130.0" x2="555.0" y2="130.0" class="stroke-grid"/>
<text x="47.0" y="134.0" class="fig-small" text-anchor="end">8</text>
<line x1="305.0" y1="245.0" x2="305.0" y2="15.0" class="stroke-grid"/>
<text x="305.0" y="261.0" class="fig-small" text-anchor="middle">8</text>
<line x1="55.0" y1="72.5" x2="555.0" y2="72.5" class="stroke-grid"/>
<text x="47.0" y="76.5" class="fig-small" text-anchor="end">12</text>
<line x1="430.0" y1="245.0" x2="430.0" y2="15.0" class="stroke-grid"/>
<text x="430.0" y="261.0" class="fig-small" text-anchor="middle">12</text>
<line x1="55.0" y1="15.0" x2="555.0" y2="15.0" class="stroke-grid"/>
<text x="47.0" y="19.0" class="fig-small" text-anchor="end">16</text>
<line x1="555.0" y1="245.0" x2="555.0" y2="15.0" class="stroke-grid"/>
<text x="555.0" y="261.0" class="fig-small" text-anchor="middle">16</text>
<line x1="55.0" y1="245.0" x2="555.0" y2="245.0" class="stroke-muted"/>
<line x1="55.0" y1="245.0" x2="55.0" y2="15.0" class="stroke-muted"/>
<path d="M55.0 15.0 L70.6 22.2 L86.2 29.4 L101.9 36.6 L117.5 43.8 L133.1 50.9 L148.8 58.1 L164.4 65.3 L180.0 72.5 L195.6 79.7 L211.2 86.9 L226.9 94.1 L242.5 101.2 L258.1 108.4 L273.8 115.6 L289.4 122.8 L305.0 130.0 L320.6 137.2 L336.2 144.4 L351.9 151.6 L367.5 158.8 L383.1 165.9 L398.8 173.1 L414.4 180.3 L430.0 187.5 L445.6 194.7 L461.2 201.9 L476.9 209.1 L492.5 216.2 L508.1 223.4 L523.8 230.6 L539.4 237.8 L555.0 245.0" class="stroke-a"/>
<path d="M55.0 15.0 L70.6 29.4 L86.2 43.8 L101.9 58.1 L117.5 72.5 L133.1 86.9 L148.8 101.2 L164.4 115.6 L180.0 130.0 L195.6 144.4 L211.2 158.8 L226.9 173.1 L242.5 187.5 L258.1 201.9 L273.8 216.2 L289.4 230.6 L305.0 245.0 L320.6 245.0 L336.2 245.0 L351.9 245.0 L367.5 245.0 L383.1 245.0 L398.8 245.0 L414.4 245.0 L430.0 245.0 L445.6 245.0 L461.2 245.0 L476.9 245.0 L492.5 245.0 L508.1 245.0 L523.8 245.0 L539.4 245.0 L555.0 245.0" class="stroke-b"/>
<line x1="273.8" y1="245.0" x2="273.8" y2="15.0" class="stroke-muted dashed"/>
<circle cx="273.8" cy="115.6" r="5" class="fill-a"/>
<circle cx="273.8" cy="216.2" r="5" class="fill-b"/>
<text x="283.8" y="109.6" class="fig-small">9 digits with X</text>
<text x="283.8" y="208.2" class="fig-small">2 digits with X transpose X</text>
<text x="314.4" y="32.2" class="fig-small">blue: solving with X keeps 16 - k</text>
<text x="314.4" y="48.2" class="fig-small">orange: through X transpose X keeps 16 - 2k</text>
<text x="305.0" y="279.0" class="fig-small" text-anchor="middle">k, where the condition number is 10^k</text>
</svg>
Figure 3. Digits kept against the size of the condition number. The orange line falls twice as fast. Notice that the orange line reaches zero at k = 8, where X itself still has about 8 good digits.
```

> [!KEY]
> When two columns look alike, the data cannot pin down their weights, and the condition number $\kappa$ measures how badly. Forming $X^\top X$ squares $\kappa$, so the normal equations lose about twice as many digits as a method that works on $X$ itself.

## Solve the normal equations with Cholesky

The normal equations are still worth knowing, because they are the fastest route when $\kappa$ is small, and you meet them everywhere. A good way to solve them uses the fact that $X^\top X$ is symmetric and, when the columns are not repeated, positive definite: $v^\top X^\top X v=\lVert Xv\rVert^2>0$ for every nonzero $v$. Such a matrix has a *Cholesky factor*, a lower triangular $L$ (zeros above the diagonal) with $LL^\top=X^\top X$. Positive definiteness is what keeps every square root in the recipe below a root of a positive number.

For the houses, the entries are $L_{11}=\sqrt3\approx1.732$, $L_{21}=6/\sqrt3\approx3.464$, and $L_{22}=\sqrt{14-12}=\sqrt2\approx1.414$. Then solve $LL^\top w=X^\top y$ in two easy steps. First solve $Lz=X^\top y=(5,11)$ from the top: $z_1=5/1.732\approx2.887$ and $z_2=(11-3.464\cdot2.887)/1.414\approx0.707$. Then solve $L^\top w=z$ from the bottom: $w_2=0.707/1.414=0.5$ and $w_1=(2.887-3.464\cdot0.5)/1.732\approx0.667$. The weights are $(2/3,1/2)$ again.

A triangular system is easy because each row has one new unknown. Cholesky needs about half the work of a general solve. But it works on the Gram matrix, so it inherits the squared condition number, and it fails, with an error or with garbage, when columns repeat exactly. In code:

```python
L = np.linalg.cholesky(X.T @ X)  # lower triangle, L @ L.T is the Gram matrix
z = solve_triangular(L, X.T @ y, lower=True)
w = solve_triangular(L.T, z, lower=False)
```

`solve_triangular` comes from `scipy.linalg`. Leaving out the transpose in the second solve gives a different vector.

> [!RECAP]
> - Nearly repeated columns make a long thin valley, so a tiny wobble in $y$ moves the weights a lot.
> - The singular values are the stretches of $X$. The condition number is $\kappa=\sigma_{\max}/\sigma_{\min}$.
> - $X^\top X$ has condition number $\kappa^2$, so the normal equations lose about twice as many digits.
> - A Cholesky factor solves them with two triangular solves, and keeps the same accuracy limit.

## Build it step by step

The starter provides `make_ill_conditioned(n, d, kappa, seed)`, which builds a table with a chosen condition number. You write three functions.

1. **`condition_number(X)`** returns one float:

   ```python
   s = np.linalg.svd(X, compute_uv=False)  # largest first
   # s[0] / s[-1], or float("inf") when s[-1] is 0
   ```

   Dividing the last by the first returns a number below 1.

2. **`digits_kept(kappa, squared)`** returns the estimate $\max(0,\,16-k)$ for $k=\log_{10}\kappa$, or $\max(0,\,16-2k)$ when `squared` is true:

   ```python
   k = np.log10(kappa)
   # 16 - k, or 16 - 2 * k, never below 0
   ```

   Forgetting the clamp at 0 returns negative digits.

3. **`fit_normal_equations(X, y)`** returns `w` of shape `(d,)` through a Cholesky factor, with the three lines above. This replaces the `np.linalg.solve` version from the chapter Fit a line by minimising squared misses.

## Where this breaks

**Columns on wild scales.** Metres beside centimetres inflate $\kappa$ even when no column repeats. Standardise each column to mean 0 and standard deviation 1. That cures the scale but not a true near-repeat.

**Nearly repeated columns.** Drop one of them, or add a penalty that pulls weights toward zero. The chapter Ridge: a fee on large weights covers the penalty.

## The exercise

You implement `condition_number`, `digits_kept`, and `fit_normal_equations`. The checks cover a built matrix with a known condition number, the identity, a rank-deficient matrix, and the weights $(0.7,0.3)$ for the nearly repeated houses. A final check builds a table with $\kappa=10^7$ and requires your Cholesky solver to be at least 100 times less accurate than a reference that works on $X$. That is a weaker claim than the seven digits the rule of thumb predicts, so it passes with room to spare. The checks reject source that calls `lstsq` or `pinv`, and require that `fit_normal_equations` calls `np.linalg.cholesky`.
