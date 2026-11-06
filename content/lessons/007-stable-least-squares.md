---
title: "Lost digits, and how QR and the SVD keep them"
slug: 007-stable-least-squares
part: "Part 1 — Foundations"
summary: See why nearly repeated columns make weights wobble and the normal equations lose digits, then solve least squares stably with QR and the SVD.
minutes: 60
gpu: false
objectives:
  - Explain why nearly repeated columns make the weights swing, and measure it with the condition number.
  - Show that forming X transpose X squares the condition number, and estimate the digits that survive.
  - Solve least squares with QR, and with the SVD as rotate, stretch, rotate, without forming X transpose X.
  - Pick the minimum-norm weights when a column repeats another, and count the rank with a cut-off.
prereqs:
  - 006-residuals-and-projection
  - 005-least-squares-line
exercise: 007-stable-least-squares
---

# Lost digits, and how QR and the SVD keep them

A house record lists its height twice: once in metres, and once in centimetres, rounded and with a speck of measurement noise. The second column is almost exactly 100 times the first. You hand both columns to the normal equations and get back two weights. Can you trust any of their digits?

You cannot, and the reason is not a bug. When two columns look alike, the data cannot tell "one unit of the first" from "one unit of the second", so the weights can trade off against each other at almost no cost to the fit. A speck of noise then decides where they land. The textbook route makes things worse, because it multiplies the problem by itself before it solves anything.

This chapter shows the swing with real numbers, gives you a name and a picture for how bad it is, and counts the digits you keep. Then it asks whether you can solve least squares without multiplying the problem by itself, and what to report when a column repeats exactly.

## What you already know

- The best weights solve $X^\top X w = X^\top y$, the normal equations.
- The residual $y-Xw$ is orthogonal to every column of $X$, and the fitted values are the projection of $y$ onto the sheet the columns span.
- Solving a 2 by 2 system divides by its determinant, $ad-bc$.
- A triangular system is solved one unknown at a time.
- A double-precision number carries about 16 correct decimal digits.

## Watch the weights swing

Shrink the situation to three houses. The first column is the size, $(1,2,3)$. The second column measures size a second way, $(1,2,3.1)$, which differs from the first only in the last entry. Take the prices $y=(1,2,3)$. The first column predicts them perfectly, so the best weights are $(1,0)$.

Now wobble one price by 0.03, to $y'=(1,2,3.03)$. A wobble that small is about the size of rounding in the measurements. Solve again. The matrix $X^\top X$, the table of dot products between pairs of columns, is the *Gram matrix*:

$$
\begin{bmatrix}14&14.3\\14.3&14.61\end{bmatrix}
$$

and its determinant is $14\cdot14.61-14.3^2=204.54-204.49=0.05$. That is tiny next to entries of size 14, and the solve divides by it. For $y'$ you get $X^\top y'=(14.09,\,14.393)$. Then

$$
w_1=\frac{14.61\cdot14.09-14.3\cdot14.393}{0.05}=\frac{205.8549-205.8199}{0.05}=0.7,
$$

and $w_2=(14\cdot14.393-14.3\cdot14.09)/0.05=(201.502-201.487)/0.05=0.3$. The weights moved from $(1,0)$ to $(0.7,0.3)$, so each weight moved by 0.3, ten times the wobble in $y$. The cause is a subtraction of two numbers about 205 that agree to four digits. Those four digits cancel, and the answer lives in what is left. 

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

The weights $(0.7,0.3)$ fit the original prices almost as well as $(1,0)$: the loss rises only from 0 to $0.00045$. A long strip of weights fits nearly equally well, so tiny noise slides the answer along it.

## Measure the squash

To name this effect, multiply every weight vector of length 1, a circle, by $X$. The result is an ellipse, and its longest and shortest stretches are the *singular values*, written $\sigma_{\max}$ and $\sigma_{\min}$ ($\sigma$ is the Greek letter sigma). The short axis explains the valley: a weight direction that barely changes the guesses is a direction the data cannot pin down.

The ratio of the two stretches is the *condition number*:

$$
\kappa=\frac{\sigma_{\max}}{\sigma_{\min}}
$$

Read it aloud: how many times longer the long axis is than the short one. A wobble of size $\varepsilon$ in $y$ can move the weights by up to $\varepsilon/\sigma_{\min}$, because a weight step of length 1 can change the guesses by as little as $\sigma_{\min}$. For the nearly repeated columns, $\sigma_{\max}\approx5.349$ and $\sigma_{\min}\approx0.0418$, so $\kappa\approx128$, and the bound is $0.03/0.0418\approx0.72$. The actual move was $\sqrt{0.3^2+0.3^2}\approx0.42$.

You find the stretches by hand from the Gram matrix, whose eigenvalues are $\sigma^2$ for each $\sigma$. For the three houses with columns $(1,1,1)$ and $(1,2,3)$, $X^\top X=\begin{bmatrix}3&6\\6&14\end{bmatrix}$ has trace 17 and determinant 6. Its eigenvalues are the roots of $t^2-17t+6=0$, about $16.639$ and $0.361$. Square roots give $\sigma_{\max}\approx4.079$ and $\sigma_{\min}\approx0.600$, so $\kappa\approx6.79$. In numpy:

```python
s = np.linalg.svd(X, compute_uv=False)  # largest first
kappa = s[0] / s[-1]
```

A condition number is never below 1. If $\sigma_{\min}$ is exactly 0, return `float("inf")`.

## Count the digits you lose

The textbook formula does not solve with $X$. It solves with $X^\top X$. Multiplying by $X$ stretches by the singular values, and multiplying by $X^\top$ stretches again by the same amounts along the same axes. So the singular values of $X^\top X$ are $\sigma_{\max}^2$ and $\sigma_{\min}^2$, and its condition number is $\kappa^2$.

Check the numbers. For the nearly repeated columns $\kappa^2\approx16\,400$, about $10^{4.2}$: the four cancelled digits from the weight swing.

A rule of thumb connects $\kappa$ to digits. Each factor of 10 in the condition number lets a rounding error grow by a factor of 10 in the weights, which pushes one more digit into noise. So a problem with condition number $10^k$ can lose about $k$ of the 16 digits a double carries. It is a rough worst-case estimate, not a guarantee. Solving with $X$ at $\kappa=10^7$ keeps about $16-7=9$ digits. Solving with $X^\top X$ has condition $10^{14}$ and keeps about $16-14=2$.

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
Figure 2. Digits kept against the size of the condition number. The orange line falls twice as fast. Notice that the orange line reaches zero at k = 8, where X itself still has about 8 good digits.
```

> [!KEY]
> When two columns look alike, the data cannot pin down their weights, and the condition number $\kappa$ measures how badly. Forming $X^\top X$ squares $\kappa$, so the normal equations lose about twice as many digits as a method that works on $X$ itself.

The normal equations are still the fastest route when $\kappa$ is small. Solve them with a *Cholesky factor*, a lower triangular $L$ (zeros above the diagonal) with $LL^\top=X^\top X$. It exists because $X^\top X$ is symmetric and, when no column repeats, positive definite. Solve $Lz=X^\top y$ from the top, then $L^\top w=z$ from the bottom. For the houses with $y=(1,2,2)$ this gives $z\approx(2.887,0.707)$ and $w=(2/3,1/2)$.

```python
L = np.linalg.cholesky(X.T @ X)  # lower triangle, L @ L.T is the Gram matrix
z = solve_triangular(L, X.T @ y, lower=True)
w = solve_triangular(L.T, z, lower=False)
```

`solve_triangular` comes from `scipy.linalg`. Cholesky needs about half the work of a general solve, but it works on the Gram matrix, so it inherits the squared condition number, and it fails when columns repeat exactly.

## Rebuild the columns from perpendicular ones

The cure is to stop working with the raw columns and use pieces that are easy to undo and stretch nothing. The first way is *QR*. Take the three houses again: columns $a=(1,1,1)$ and $b=(1,2,3)$ are not perpendicular. Build perpendicular columns that span the same sheet, one at a time.

Start with $a$. Its length is $\sqrt3\approx1.732$, so the unit-length version is $q_1=a/\sqrt3$. Next, remove from $b$ the part that points along $q_1$. That part has size $q_1\cdot b=6/\sqrt3\approx3.464$, which is the vector $3.464\,q_1=(2,2,2)$. What is left is $b-(2,2,2)=(-1,0,1)$, perpendicular to $a$ by construction. Its length is $\sqrt2\approx1.414$, so $q_2=(-1,0,1)/\sqrt2$.

Figure 3 draws this in the sheet. The vectors $q_1$ and $q_2$ are *orthonormal*: perpendicular, and each of length 1. Every column of $X$ is a recipe using them. Column 1 is $1.732\,q_1$. Column 2 is $3.464\,q_1+1.414\,q_2$. Collect the recipe numbers in a table $R$, so that $X=QR$, where the columns of $Q$ are $q_1$ and $q_2$:

$$
X=QR,\qquad R=\begin{bmatrix}1.732&3.464\\0&1.414\end{bmatrix}
$$

```figure
<svg viewBox="0 0 600 275" role="img" aria-label="Left: the two columns of X drawn in the sheet, with two perpendicular unit arrows q1 and q2 rebuilt from them, and the numbers 1.732, 3.464, and 1.414 that make up R. Right: the triangular system solved from the bottom row.">
<defs>
<marker id="c007-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-ink"/></marker>
<marker id="c007-arrowa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-a"/></marker>
<marker id="c007-arrowb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-b"/></marker>
<marker id="c007-arrowc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-c"/></marker>
</defs>
<line x1="40.0" y1="200.0" x2="300.0" y2="200.0" class="stroke-grid"/>
<line x1="40.0" y1="200.0" x2="40.0" y2="30.0" class="stroke-grid"/>
<text x="40.0" y="18.0" class="fig-bold">The sheet, drawn flat</text>

<line x1="40.0" y1="200.0" x2="213.2" y2="129.3" class="stroke-d" marker-end="url(#c007-arrow)"/>
<line x1="40.0" y1="200.0" x2="126.6" y2="200.0" class="stroke-a" marker-end="url(#c007-arrowa)"/>
<line x1="213.2" y1="200.0" x2="213.2" y2="129.3" class="stroke-muted dashed"/>
<line x1="40.0" y1="214.0" x2="90.0" y2="214.0" class="stroke-c" marker-end="url(#c007-arrowc)"/>
<text x="96.0" y="218.0" class="fig-small">q1</text>
<line x1="26.0" y1="200.0" x2="26.0" y2="150.0" class="stroke-c" marker-end="url(#c007-arrowc)"/>
<text x="20.0" y="175.0" class="fig-small" text-anchor="end">q2</text>

<text x="165.0" y="190.0" class="fig-small" text-anchor="middle">col 1 = 1.732 q1</text>
<text x="213.2" y="216.0" class="fig-small" text-anchor="middle">3.464</text>
<text x="207.2" y="121.3" class="fig-small" text-anchor="end">col 2 = 3.464 q1 + 1.414 q2</text>
<text x="219.2" y="168.6" class="fig-small">1.414</text>

<text x="40.0" y="252.0" class="fig-small">q1, q2: perpendicular, length 1. R holds these</text>
<text x="40.0" y="268.0" class="fig-small">numbers: [[1.732, 3.464], [0, 1.414]]</text>
<line x1="330.0" y1="35.0" x2="330.0" y2="235.0" class="stroke-grid"/>
<text x="345.0" y="18.0" class="fig-bold">Solve R w = Q^T y from the bottom</text>
<text x="345.0" y="50.0" class="fig-small fig-mono">1.732 w1 + 3.464 w2 = 2.887</text>
<text x="345.0" y="70.0" class="fig-small fig-mono">            1.414 w2 = 0.707</text>
<text x="345.0" y="90.0" class="fig-small fig-mono"></text>
<text x="345.0" y="110.0" class="fig-small fig-mono">bottom row: w2 = 0.707 / 1.414 = 0.5</text>
<text x="345.0" y="130.0" class="fig-small fig-mono"></text>
<text x="345.0" y="150.0" class="fig-small fig-mono">top row:</text>
<text x="345.0" y="170.0" class="fig-small fig-mono">w1 = (2.887 - 3.464 x 0.5) / 1.732</text>
<text x="345.0" y="190.0" class="fig-small fig-mono">   = 0.667</text>
<text x="345.0" y="252.0" class="fig-small">Each row brings in one new unknown.</text>
</svg>
Figure 3. Left: column 2 splits into a part along q1 (3.464) and a part along q2 (1.414). Those two numbers and the length of column 1 form R. Right: because R is triangular, you solve it from the bottom row up. Notice that nothing here multiplies X by its own transpose.
```

$R$ is upper triangular: zeros below the diagonal, because column 1 uses only $q_1$. Dotting $y$ with each perpendicular direction gives the shadow's coordinates, so the shadow is $QQ^\top y$. The fitted values are $Xw=QRw$, and setting them equal to the shadow gives

$$
R\,w=Q^\top y
$$

Read it aloud: dot $y$ with each perpendicular direction, then solve the triangle. For the houses with $y=(1,2,2)$, $Q^\top y=(5/\sqrt3,\,1/\sqrt2)\approx(2.887,\,0.707)$. The bottom row gives $w_2=0.707/1.414=0.5$. The top row gives $1.732\,w_1+3.464\cdot0.5=2.887$, so $w_1=0.667$. These are the same weights $(2/3,1/2)$, with no $X^\top X$ anywhere.

Why is this safer? Multiplying by $Q$ only rotates, so $R$ has the same singular values as $X$, and $Rw=Q^\top y$ has condition number $\kappa$, not $\kappa^2$. At $\kappa=10^7$ you keep about nine digits instead of two. In numpy, ask for the reduced factorisation, where $Q$ is $n\times d$:

```python
Q, R = np.linalg.qr(X, mode="reduced")  # Q is (n, d), R is (d, d)
w = solve_triangular(R, Q.T @ y, lower=False)
```

## Rotate, stretch, rotate

QR needs full-rank columns: a repeated column puts a zero on the diagonal of $R$. The SVD (singular value decomposition) handles every case. It says every table splits as

$$
X=U\,\Sigma\,V^\top
$$

Read it aloud: rotate the weights with $V^\top$, stretch each axis by its singular value in $\Sigma$, then rotate into the sheet with $U$. In Figure 4, the top row turns the unit circle into the ellipse from the measuring section, and the bottom row undoes the moves in reverse order to solve.

```figure
<svg viewBox="0 0 600 345" role="img" aria-label="Top row: the unit circle is rotated, stretched along its axes by 4.08 and 0.60, and rotated again to give the ellipse. Bottom row: solving undoes the three moves in reverse order: rotate back, divide by the stretches, rotate back. A red stop sign warns that dividing by a stretch near zero blows up.">
<defs>
<marker id="c007-rsr-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="fill-ink"/></marker>
</defs>
<text x="15.0" y="15.0" class="fig-bold">Fit: what X does</text>
<line x1="18.0" y1="75.0" x2="142.0" y2="75.0" class="stroke-grid"/>
<line x1="80.0" y1="35.0" x2="80.0" y2="115.0" class="stroke-grid"/>
<path d="M93.0 75.0 L93.0 74.3 L92.9 73.6 L92.8 73.0 L92.7 72.3 L92.6 71.6 L92.4 71.0 L92.1 70.3 L91.9 69.7 L91.6 69.1 L91.3 68.5 L90.9 67.9 L90.5 67.4 L90.1 66.8 L89.7 66.3 L89.2 65.8 L88.7 65.3 L88.2 64.9 L87.6 64.5 L87.1 64.1 L86.5 63.7 L85.9 63.4 L85.3 63.1 L84.7 62.9 L84.0 62.6 L83.4 62.4 L82.7 62.3 L82.0 62.2 L81.4 62.1 L80.7 62.0 L80.0 62.0 L79.3 62.0 L78.6 62.1 L78.0 62.2 L77.3 62.3 L76.6 62.4 L76.0 62.6 L75.3 62.9 L74.7 63.1 L74.1 63.4 L73.5 63.7 L72.9 64.1 L72.4 64.5 L71.8 64.9 L71.3 65.3 L70.8 65.8 L70.3 66.3 L69.9 66.8 L69.5 67.4 L69.1 67.9 L68.7 68.5 L68.4 69.1 L68.1 69.7 L67.9 70.3 L67.6 71.0 L67.4 71.6 L67.3 72.3 L67.2 73.0 L67.1 73.6 L67.0 74.3 L67.0 75.0 L67.0 75.7 L67.1 76.4 L67.2 77.0 L67.3 77.7 L67.4 78.4 L67.6 79.0 L67.9 79.7 L68.1 80.3 L68.4 80.9 L68.7 81.5 L69.1 82.1 L69.5 82.6 L69.9 83.2 L70.3 83.7 L70.8 84.2 L71.3 84.7 L71.8 85.1 L72.4 85.5 L72.9 85.9 L73.5 86.3 L74.1 86.6 L74.7 86.9 L75.3 87.1 L76.0 87.4 L76.6 87.6 L77.3 87.7 L78.0 87.8 L78.6 87.9 L79.3 88.0 L80.0 88.0 L80.7 88.0 L81.4 87.9 L82.0 87.8 L82.7 87.7 L83.4 87.6 L84.0 87.4 L84.7 87.1 L85.3 86.9 L85.9 86.6 L86.5 86.3 L87.1 85.9 L87.6 85.5 L88.2 85.1 L88.7 84.7 L89.2 84.2 L89.7 83.7 L90.1 83.2 L90.5 82.6 L90.9 82.1 L91.3 81.5 L91.6 80.9 L91.9 80.3 L92.1 79.7 L92.4 79.0 L92.6 78.4 L92.7 77.7 L92.8 77.0 L92.9 76.4 L93.0 75.7 L93.0 75.0 Z" class="fill-soft-b stroke-b"/>
<line x1="80.0" y1="75.0" x2="93.0" y2="75.0" class="stroke-a"/>
<line x1="80.0" y1="75.0" x2="80.0" y2="62.0" class="stroke-c"/>
<text x="80.0" y="129.0" class="fig-small" text-anchor="middle">unit circle</text>
<line x1="146.0" y1="75.0" x2="166.0" y2="75.0" class="stroke-ink" marker-end="url(#c007-rsr-arrow)"/>
<line x1="168.0" y1="75.0" x2="292.0" y2="75.0" class="stroke-grid"/>
<line x1="230.0" y1="35.0" x2="230.0" y2="115.0" class="stroke-grid"/>
<path d="M224.8 63.1 L224.1 63.4 L223.6 63.7 L223.0 64.1 L222.4 64.4 L221.9 64.9 L221.3 65.3 L220.8 65.8 L220.4 66.3 L219.9 66.8 L219.5 67.3 L219.1 67.9 L218.8 68.4 L218.4 69.0 L218.1 69.7 L217.9 70.3 L217.7 70.9 L217.5 71.6 L217.3 72.2 L217.2 72.9 L217.1 73.6 L217.0 74.3 L217.0 74.9 L217.0 75.6 L217.1 76.3 L217.2 77.0 L217.3 77.6 L217.4 78.3 L217.6 79.0 L217.8 79.6 L218.1 80.2 L218.4 80.9 L218.7 81.4 L219.1 82.0 L219.4 82.6 L219.9 83.1 L220.3 83.7 L220.8 84.2 L221.3 84.6 L221.8 85.1 L222.3 85.5 L222.9 85.9 L223.4 86.2 L224.0 86.6 L224.7 86.9 L225.3 87.1 L225.9 87.3 L226.6 87.5 L227.2 87.7 L227.9 87.8 L228.6 87.9 L229.3 88.0 L229.9 88.0 L230.6 88.0 L231.3 87.9 L232.0 87.8 L232.6 87.7 L233.3 87.6 L234.0 87.4 L234.6 87.2 L235.2 86.9 L235.9 86.6 L236.4 86.3 L237.0 85.9 L237.6 85.6 L238.1 85.1 L238.7 84.7 L239.2 84.2 L239.6 83.7 L240.1 83.2 L240.5 82.7 L240.9 82.1 L241.2 81.6 L241.6 81.0 L241.9 80.3 L242.1 79.7 L242.3 79.1 L242.5 78.4 L242.7 77.8 L242.8 77.1 L242.9 76.4 L243.0 75.7 L243.0 75.1 L243.0 74.4 L242.9 73.7 L242.8 73.0 L242.7 72.4 L242.6 71.7 L242.4 71.0 L242.2 70.4 L241.9 69.8 L241.6 69.1 L241.3 68.6 L240.9 68.0 L240.6 67.4 L240.1 66.9 L239.7 66.3 L239.2 65.8 L238.7 65.4 L238.2 64.9 L237.7 64.5 L237.1 64.1 L236.6 63.8 L236.0 63.4 L235.3 63.1 L234.7 62.9 L234.1 62.7 L233.4 62.5 L232.8 62.3 L232.1 62.2 L231.4 62.1 L230.7 62.0 L230.1 62.0 L229.4 62.0 L228.7 62.1 L228.0 62.2 L227.4 62.3 L226.7 62.4 L226.0 62.6 L225.4 62.8 L224.8 63.1 Z" class="fill-soft-b stroke-b"/>
<line x1="230.0" y1="75.0" x2="224.8" y2="63.1" class="stroke-a"/>
<line x1="230.0" y1="75.0" x2="218.1" y2="80.2" class="stroke-c"/>
<text x="230.0" y="129.0" class="fig-small" text-anchor="middle">rotate (V^T)</text>
<line x1="296.0" y1="75.0" x2="316.0" y2="75.0" class="stroke-ink" marker-end="url(#c007-rsr-arrow)"/>
<line x1="318.0" y1="75.0" x2="442.0" y2="75.0" class="stroke-grid"/>
<line x1="380.0" y1="35.0" x2="380.0" y2="115.0" class="stroke-grid"/>
<path d="M358.6 67.9 L356.1 68.0 L353.7 68.2 L351.3 68.4 L349.0 68.7 L346.8 68.9 L344.7 69.2 L342.7 69.5 L340.8 69.8 L338.9 70.1 L337.2 70.4 L335.7 70.7 L334.2 71.1 L332.9 71.4 L331.7 71.8 L330.6 72.2 L329.6 72.6 L328.8 72.9 L328.2 73.3 L327.7 73.7 L327.3 74.1 L327.1 74.6 L327.0 75.0 L327.0 75.4 L327.2 75.8 L327.6 76.2 L328.1 76.6 L328.7 77.0 L329.5 77.4 L330.4 77.8 L331.5 78.1 L332.6 78.5 L334.0 78.9 L335.4 79.2 L337.0 79.6 L338.6 79.9 L340.4 80.2 L342.3 80.5 L344.3 80.8 L346.4 81.0 L348.6 81.3 L350.9 81.5 L353.3 81.7 L355.7 81.9 L358.2 82.1 L360.8 82.3 L363.4 82.4 L366.0 82.5 L368.7 82.6 L371.5 82.7 L374.2 82.8 L377.0 82.8 L379.8 82.8 L382.5 82.8 L385.3 82.8 L388.1 82.7 L390.8 82.6 L393.5 82.5 L396.2 82.4 L398.8 82.3 L401.4 82.1 L403.9 82.0 L406.3 81.8 L408.7 81.6 L411.0 81.3 L413.2 81.1 L415.3 80.8 L417.3 80.5 L419.2 80.2 L421.1 79.9 L422.8 79.6 L424.3 79.3 L425.8 78.9 L427.1 78.6 L428.3 78.2 L429.4 77.8 L430.4 77.4 L431.2 77.1 L431.8 76.7 L432.3 76.3 L432.7 75.9 L432.9 75.4 L433.0 75.0 L433.0 74.6 L432.8 74.2 L432.4 73.8 L431.9 73.4 L431.3 73.0 L430.5 72.6 L429.6 72.2 L428.5 71.9 L427.4 71.5 L426.0 71.1 L424.6 70.8 L423.0 70.4 L421.4 70.1 L419.6 69.8 L417.7 69.5 L415.7 69.2 L413.6 69.0 L411.4 68.7 L409.1 68.5 L406.7 68.3 L404.3 68.1 L401.8 67.9 L399.2 67.7 L396.6 67.6 L394.0 67.5 L391.3 67.4 L388.5 67.3 L385.8 67.2 L383.0 67.2 L380.2 67.2 L377.5 67.2 L374.7 67.2 L371.9 67.3 L369.2 67.4 L366.5 67.5 L363.8 67.6 L361.2 67.7 L358.6 67.9 Z" class="fill-soft-b stroke-b"/>
<line x1="380.0" y1="75.0" x2="358.6" y2="67.9" class="stroke-a"/>
<line x1="380.0" y1="75.0" x2="331.5" y2="78.1" class="stroke-c"/>
<text x="380.0" y="129.0" class="fig-small" text-anchor="middle">stretch (sigma)</text>
<line x1="446.0" y1="75.0" x2="466.0" y2="75.0" class="stroke-ink" marker-end="url(#c007-rsr-arrow)"/>
<line x1="468.0" y1="75.0" x2="592.0" y2="75.0" class="stroke-grid"/>
<line x1="530.0" y1="35.0" x2="530.0" y2="115.0" class="stroke-grid"/>
<path d="M552.5 75.0 L554.8 74.0 L557.1 73.1 L559.3 72.1 L561.4 71.2 L563.4 70.2 L565.3 69.3 L567.2 68.4 L568.9 67.5 L570.5 66.7 L572.0 65.8 L573.4 65.0 L574.7 64.2 L575.8 63.4 L576.9 62.7 L577.8 62.0 L578.5 61.3 L579.2 60.7 L579.7 60.1 L580.0 59.6 L580.3 59.1 L580.3 58.6 L580.3 58.2 L580.1 57.8 L579.8 57.5 L579.3 57.2 L578.7 57.0 L578.0 56.8 L577.1 56.7 L576.2 56.6 L575.0 56.6 L573.8 56.6 L572.4 56.7 L571.0 56.8 L569.4 57.0 L567.7 57.2 L565.9 57.5 L564.0 57.8 L562.0 58.2 L559.9 58.6 L557.7 59.1 L555.5 59.6 L553.2 60.1 L550.8 60.7 L548.4 61.3 L545.9 62.0 L543.4 62.7 L540.8 63.4 L538.3 64.2 L535.6 65.0 L533.0 65.8 L530.4 66.7 L527.7 67.5 L525.1 68.4 L522.5 69.3 L519.9 70.2 L517.3 71.2 L514.8 72.1 L512.3 73.1 L509.9 74.0 L507.5 75.0 L505.2 76.0 L502.9 76.9 L500.7 77.9 L498.6 78.8 L496.6 79.8 L494.7 80.7 L492.8 81.6 L491.1 82.5 L489.5 83.3 L488.0 84.2 L486.6 85.0 L485.3 85.8 L484.2 86.6 L483.1 87.3 L482.2 88.0 L481.5 88.7 L480.8 89.3 L480.3 89.9 L480.0 90.4 L479.7 90.9 L479.7 91.4 L479.7 91.8 L479.9 92.2 L480.2 92.5 L480.7 92.8 L481.3 93.0 L482.0 93.2 L482.9 93.3 L483.8 93.4 L485.0 93.4 L486.2 93.4 L487.6 93.3 L489.0 93.2 L490.6 93.0 L492.3 92.8 L494.1 92.5 L496.0 92.2 L498.0 91.8 L500.1 91.4 L502.3 90.9 L504.5 90.4 L506.8 89.9 L509.2 89.3 L511.6 88.7 L514.1 88.0 L516.6 87.3 L519.2 86.6 L521.7 85.8 L524.4 85.0 L527.0 84.2 L529.6 83.3 L532.3 82.5 L534.9 81.6 L537.5 80.7 L540.1 79.8 L542.7 78.8 L545.2 77.9 L547.7 76.9 L550.1 76.0 L552.5 75.0 Z" class="fill-soft-b stroke-b"/>
<line x1="530.0" y1="75.0" x2="552.5" y2="75.0" class="stroke-a"/>
<line x1="530.0" y1="75.0" x2="575.0" y2="56.6" class="stroke-c"/>
<text x="530.0" y="129.0" class="fig-small" text-anchor="middle">rotate (U)</text>
<text x="15.0" y="150.0" class="fig-bold">Solve: undo it, last move first</text>
<line x1="18.0" y1="210.0" x2="142.0" y2="210.0" class="stroke-grid"/>
<line x1="80.0" y1="170.0" x2="80.0" y2="250.0" class="stroke-grid"/>
<path d="M102.5 210.0 L104.8 209.0 L107.1 208.1 L109.3 207.1 L111.4 206.2 L113.4 205.2 L115.3 204.3 L117.2 203.4 L118.9 202.5 L120.5 201.7 L122.0 200.8 L123.4 200.0 L124.7 199.2 L125.8 198.4 L126.9 197.7 L127.8 197.0 L128.5 196.3 L129.2 195.7 L129.7 195.1 L130.0 194.6 L130.3 194.1 L130.3 193.6 L130.3 193.2 L130.1 192.8 L129.8 192.5 L129.3 192.2 L128.7 192.0 L128.0 191.8 L127.1 191.7 L126.2 191.6 L125.0 191.6 L123.8 191.6 L122.4 191.7 L121.0 191.8 L119.4 192.0 L117.7 192.2 L115.9 192.5 L114.0 192.8 L112.0 193.2 L109.9 193.6 L107.7 194.1 L105.5 194.6 L103.2 195.1 L100.8 195.7 L98.4 196.3 L95.9 197.0 L93.4 197.7 L90.8 198.4 L88.3 199.2 L85.6 200.0 L83.0 200.8 L80.4 201.7 L77.7 202.5 L75.1 203.4 L72.5 204.3 L69.9 205.2 L67.3 206.2 L64.8 207.1 L62.3 208.1 L59.9 209.0 L57.5 210.0 L55.2 211.0 L52.9 211.9 L50.7 212.9 L48.6 213.8 L46.6 214.8 L44.7 215.7 L42.8 216.6 L41.1 217.5 L39.5 218.3 L38.0 219.2 L36.6 220.0 L35.3 220.8 L34.2 221.6 L33.1 222.3 L32.2 223.0 L31.5 223.7 L30.8 224.3 L30.3 224.9 L30.0 225.4 L29.7 225.9 L29.7 226.4 L29.7 226.8 L29.9 227.2 L30.2 227.5 L30.7 227.8 L31.3 228.0 L32.0 228.2 L32.9 228.3 L33.8 228.4 L35.0 228.4 L36.2 228.4 L37.6 228.3 L39.0 228.2 L40.6 228.0 L42.3 227.8 L44.1 227.5 L46.0 227.2 L48.0 226.8 L50.1 226.4 L52.3 225.9 L54.5 225.4 L56.8 224.9 L59.2 224.3 L61.6 223.7 L64.1 223.0 L66.6 222.3 L69.2 221.6 L71.7 220.8 L74.4 220.0 L77.0 219.2 L79.6 218.3 L82.3 217.5 L84.9 216.6 L87.5 215.7 L90.1 214.8 L92.7 213.8 L95.2 212.9 L97.7 211.9 L100.1 211.0 L102.5 210.0 Z" class="fill-soft-b stroke-b"/>
<line x1="80.0" y1="210.0" x2="102.5" y2="210.0" class="stroke-a"/>
<line x1="80.0" y1="210.0" x2="125.0" y2="191.6" class="stroke-c"/>
<text x="80.0" y="264.0" class="fig-small" text-anchor="middle">the ellipse</text>
<line x1="146.0" y1="210.0" x2="166.0" y2="210.0" class="stroke-ink" marker-end="url(#c007-rsr-arrow)"/>
<line x1="168.0" y1="210.0" x2="292.0" y2="210.0" class="stroke-grid"/>
<line x1="230.0" y1="170.0" x2="230.0" y2="250.0" class="stroke-grid"/>
<path d="M208.6 202.9 L206.1 203.0 L203.7 203.2 L201.3 203.4 L199.0 203.7 L196.8 203.9 L194.7 204.2 L192.7 204.5 L190.8 204.8 L188.9 205.1 L187.2 205.4 L185.7 205.7 L184.2 206.1 L182.9 206.4 L181.7 206.8 L180.6 207.2 L179.6 207.6 L178.8 207.9 L178.2 208.3 L177.7 208.7 L177.3 209.1 L177.1 209.6 L177.0 210.0 L177.0 210.4 L177.2 210.8 L177.6 211.2 L178.1 211.6 L178.7 212.0 L179.5 212.4 L180.4 212.8 L181.5 213.1 L182.6 213.5 L184.0 213.9 L185.4 214.2 L187.0 214.6 L188.6 214.9 L190.4 215.2 L192.3 215.5 L194.3 215.8 L196.4 216.0 L198.6 216.3 L200.9 216.5 L203.3 216.7 L205.7 216.9 L208.2 217.1 L210.8 217.3 L213.4 217.4 L216.0 217.5 L218.7 217.6 L221.5 217.7 L224.2 217.8 L227.0 217.8 L229.8 217.8 L232.5 217.8 L235.3 217.8 L238.1 217.7 L240.8 217.6 L243.5 217.5 L246.2 217.4 L248.8 217.3 L251.4 217.1 L253.9 217.0 L256.3 216.8 L258.7 216.6 L261.0 216.3 L263.2 216.1 L265.3 215.8 L267.3 215.5 L269.2 215.2 L271.1 214.9 L272.8 214.6 L274.3 214.3 L275.8 213.9 L277.1 213.6 L278.3 213.2 L279.4 212.8 L280.4 212.4 L281.2 212.1 L281.8 211.7 L282.3 211.3 L282.7 210.9 L282.9 210.4 L283.0 210.0 L283.0 209.6 L282.8 209.2 L282.4 208.8 L281.9 208.4 L281.3 208.0 L280.5 207.6 L279.6 207.2 L278.5 206.9 L277.4 206.5 L276.0 206.1 L274.6 205.8 L273.0 205.4 L271.4 205.1 L269.6 204.8 L267.7 204.5 L265.7 204.2 L263.6 204.0 L261.4 203.7 L259.1 203.5 L256.7 203.3 L254.3 203.1 L251.8 202.9 L249.2 202.7 L246.6 202.6 L244.0 202.5 L241.3 202.4 L238.5 202.3 L235.8 202.2 L233.0 202.2 L230.2 202.2 L227.5 202.2 L224.7 202.2 L221.9 202.3 L219.2 202.4 L216.5 202.5 L213.8 202.6 L211.2 202.7 L208.6 202.9 Z" class="fill-soft-b stroke-b"/>
<line x1="230.0" y1="210.0" x2="208.6" y2="202.9" class="stroke-a"/>
<line x1="230.0" y1="210.0" x2="181.5" y2="213.1" class="stroke-c"/>
<text x="230.0" y="264.0" class="fig-small" text-anchor="middle">rotate back (U^T)</text>
<line x1="296.0" y1="210.0" x2="316.0" y2="210.0" class="stroke-ink" marker-end="url(#c007-rsr-arrow)"/>
<line x1="318.0" y1="210.0" x2="442.0" y2="210.0" class="stroke-grid"/>
<line x1="380.0" y1="170.0" x2="380.0" y2="250.0" class="stroke-grid"/>
<path d="M374.8 198.1 L374.1 198.4 L373.6 198.7 L373.0 199.1 L372.4 199.4 L371.9 199.9 L371.3 200.3 L370.8 200.8 L370.4 201.3 L369.9 201.8 L369.5 202.3 L369.1 202.9 L368.8 203.4 L368.4 204.0 L368.1 204.7 L367.9 205.3 L367.7 205.9 L367.5 206.6 L367.3 207.2 L367.2 207.9 L367.1 208.6 L367.0 209.3 L367.0 209.9 L367.0 210.6 L367.1 211.3 L367.2 212.0 L367.3 212.6 L367.4 213.3 L367.6 214.0 L367.8 214.6 L368.1 215.2 L368.4 215.9 L368.7 216.4 L369.1 217.0 L369.4 217.6 L369.9 218.1 L370.3 218.7 L370.8 219.2 L371.3 219.6 L371.8 220.1 L372.3 220.5 L372.9 220.9 L373.4 221.2 L374.0 221.6 L374.7 221.9 L375.3 222.1 L375.9 222.3 L376.6 222.5 L377.2 222.7 L377.9 222.8 L378.6 222.9 L379.3 223.0 L379.9 223.0 L380.6 223.0 L381.3 222.9 L382.0 222.8 L382.6 222.7 L383.3 222.6 L384.0 222.4 L384.6 222.2 L385.2 221.9 L385.9 221.6 L386.4 221.3 L387.0 220.9 L387.6 220.6 L388.1 220.1 L388.7 219.7 L389.2 219.2 L389.6 218.7 L390.1 218.2 L390.5 217.7 L390.9 217.1 L391.2 216.6 L391.6 216.0 L391.9 215.3 L392.1 214.7 L392.3 214.1 L392.5 213.4 L392.7 212.8 L392.8 212.1 L392.9 211.4 L393.0 210.7 L393.0 210.1 L393.0 209.4 L392.9 208.7 L392.8 208.0 L392.7 207.4 L392.6 206.7 L392.4 206.0 L392.2 205.4 L391.9 204.8 L391.6 204.1 L391.3 203.6 L390.9 203.0 L390.6 202.4 L390.1 201.9 L389.7 201.3 L389.2 200.8 L388.7 200.4 L388.2 199.9 L387.7 199.5 L387.1 199.1 L386.6 198.8 L386.0 198.4 L385.3 198.1 L384.7 197.9 L384.1 197.7 L383.4 197.5 L382.8 197.3 L382.1 197.2 L381.4 197.1 L380.7 197.0 L380.1 197.0 L379.4 197.0 L378.7 197.1 L378.0 197.2 L377.4 197.3 L376.7 197.4 L376.0 197.6 L375.4 197.8 L374.8 198.1 Z" class="fill-soft-b stroke-b"/>
<line x1="380.0" y1="210.0" x2="374.8" y2="198.1" class="stroke-a"/>
<line x1="380.0" y1="210.0" x2="368.1" y2="215.2" class="stroke-c"/>
<text x="380.0" y="264.0" class="fig-small" text-anchor="middle">divide by sigma</text>
<line x1="446.0" y1="210.0" x2="466.0" y2="210.0" class="stroke-ink" marker-end="url(#c007-rsr-arrow)"/>
<line x1="468.0" y1="210.0" x2="592.0" y2="210.0" class="stroke-grid"/>
<line x1="530.0" y1="170.0" x2="530.0" y2="250.0" class="stroke-grid"/>
<path d="M543.0 210.0 L543.0 209.3 L542.9 208.6 L542.8 208.0 L542.7 207.3 L542.6 206.6 L542.4 206.0 L542.1 205.3 L541.9 204.7 L541.6 204.1 L541.3 203.5 L540.9 202.9 L540.5 202.4 L540.1 201.8 L539.7 201.3 L539.2 200.8 L538.7 200.3 L538.2 199.9 L537.6 199.5 L537.1 199.1 L536.5 198.7 L535.9 198.4 L535.3 198.1 L534.7 197.9 L534.0 197.6 L533.4 197.4 L532.7 197.3 L532.0 197.2 L531.4 197.1 L530.7 197.0 L530.0 197.0 L529.3 197.0 L528.6 197.1 L528.0 197.2 L527.3 197.3 L526.6 197.4 L526.0 197.6 L525.3 197.9 L524.7 198.1 L524.1 198.4 L523.5 198.7 L522.9 199.1 L522.4 199.5 L521.8 199.9 L521.3 200.3 L520.8 200.8 L520.3 201.3 L519.9 201.8 L519.5 202.4 L519.1 202.9 L518.7 203.5 L518.4 204.1 L518.1 204.7 L517.9 205.3 L517.6 206.0 L517.4 206.6 L517.3 207.3 L517.2 208.0 L517.1 208.6 L517.0 209.3 L517.0 210.0 L517.0 210.7 L517.1 211.4 L517.2 212.0 L517.3 212.7 L517.4 213.4 L517.6 214.0 L517.9 214.7 L518.1 215.3 L518.4 215.9 L518.7 216.5 L519.1 217.1 L519.5 217.6 L519.9 218.2 L520.3 218.7 L520.8 219.2 L521.3 219.7 L521.8 220.1 L522.4 220.5 L522.9 220.9 L523.5 221.3 L524.1 221.6 L524.7 221.9 L525.3 222.1 L526.0 222.4 L526.6 222.6 L527.3 222.7 L528.0 222.8 L528.6 222.9 L529.3 223.0 L530.0 223.0 L530.7 223.0 L531.4 222.9 L532.0 222.8 L532.7 222.7 L533.4 222.6 L534.0 222.4 L534.7 222.1 L535.3 221.9 L535.9 221.6 L536.5 221.3 L537.1 220.9 L537.6 220.5 L538.2 220.1 L538.7 219.7 L539.2 219.2 L539.7 218.7 L540.1 218.2 L540.5 217.6 L540.9 217.1 L541.3 216.5 L541.6 215.9 L541.9 215.3 L542.1 214.7 L542.4 214.0 L542.6 213.4 L542.7 212.7 L542.8 212.0 L542.9 211.4 L543.0 210.7 L543.0 210.0 Z" class="fill-soft-b stroke-b"/>
<line x1="530.0" y1="210.0" x2="543.0" y2="210.0" class="stroke-a"/>
<line x1="530.0" y1="210.0" x2="530.0" y2="197.0" class="stroke-c"/>
<text x="530.0" y="264.0" class="fig-small" text-anchor="middle">rotate back (V)</text>
<path d="M57.0 330.0 L50.0 337.0 L40.0 337.0 L33.0 330.0 L33.0 320.0 L40.0 313.0 L50.0 313.0 L57.0 320.0 Z" class="fill-bad"/>
<text x="69.0" y="322.0" class="fig-small">If a stretch is near 0, dividing by it</text>
<text x="69.0" y="338.0" class="fig-small">blows that axis up. Cut it off instead.</text>
</svg>
Figure 4. Top: rotate, stretch, rotate turns the circle into the ellipse. Bottom: solving rotates back, divides each axis by its stretch, and rotates back. Notice that the divide step is the only one that can amplify anything, and it amplifies most along the axis with the smallest stretch.
```

In symbols the undo is $w=V\,\Sigma^{-1}\,U^\top y$.

For the houses, to four digits: $U^\top y=(-2.9619,\,0.2455)$. Dividing by $\sigma=(4.0791,\,0.6005)$ gives $(-0.7261,\,0.4089)$. Then $V=\begin{bmatrix}-0.4027&0.9153\\-0.9153&-0.4027\end{bmatrix}$ rotates back to $w\approx(0.6667,\,0.5000)$. The signs of $U$ and $V$ can differ in matching pairs, and the weights stay the same.

```python
U, s, vt = np.linalg.svd(X, full_matrices=False)
w = vt.T @ ((U.T @ y) / s)
```

> [!WARNING]
> numpy returns `vt`, which is already $V^\top$. You get $V$ from `vt.T`. Using `vt` as $V$ gives the wrong weights.

### Pick the shortest weights when a column repeats

Use the columns $(1,2,3)$ and $(2,4,6)$ with prices $y=(1,1,1)$. The second column is twice the first, so any weights $(a,b)$ give the guess $(a+2b)\,(1,2,3)$. The best guess is the projection of $y$ onto $(1,2,3)$, which is $3/7$ times $(1,2,3)$. So every pair with $a+2b=3/7$ fits equally well.

Which pair do you report? Moving along that line, in the direction $(2,-1)$, changes the weights but not the fit. The directions a matrix sends to zero are its *null space*, and moving there adds length and no accuracy. So pick the point with none of it, the *minimum-norm* solution, the shortest weight vector. It is the foot of the perpendicular from the origin: $(3/35,\,6/35)\approx(0.086,\,0.171)$.

The SVD finds it. The second singular value is 0 in exact arithmetic, and about $10^{-16}$ times the first on a computer. Dividing by it multiplies rounding noise by $10^{15}$ or more, and the weights run off along the line. The cure is a cut-off: treat any singular value at or below `rcond * s[0]` as zero, and drop that axis.

```python
keep = s > rcond * s[0]  # rcond is 1e-12 in the exercise
w = vt[keep].T @ ((U[:, keep].T @ y) / s[keep])
```

The kept singular values count the *rank*, the number of directions the columns really use. Here the rank is 1, and the result is $(0.0857,\,0.1714)$.

> [!KEY]
> Instead of forming $X^\top X$, split $X$ into pieces that do not stretch: QR gives a triangle to back-substitute, and the SVD gives a rotate, divide, rotate. Both keep the condition number at $\kappa$. When columns repeat, the SVD with a cut-off returns the shortest of the equally good weights.

> [!RECAP]
> - Nearly repeated columns make a long thin valley, so a tiny wobble in $y$ moves the weights a lot. The condition number is $\kappa=\sigma_{\max}/\sigma_{\min}$.
> - $X^\top X$ has condition number $\kappa^2$, so the normal equations lose about twice as many digits.
> - QR writes $X=QR$ and solves $Rw=Q^\top y$. The SVD writes $X=U\Sigma V^\top$ and solves $w=V\Sigma^{-1}U^\top y$. Both keep $\kappa$.
> - A repeated column makes many weights fit equally well. Dropping singular values at or below `rcond * s[0]` picks the shortest.

## Build it step by step

The starter provides `make_ill_conditioned(n, d, kappa, seed)`, which builds a table with a chosen condition number. You write six functions. The QR and SVD solvers must not contain `X.T @ X`.

1. **`condition_number(X)`** returns one float: `s[0] / s[-1]` from `np.linalg.svd(X, compute_uv=False)`, or `float("inf")` when `s[-1]` is 0. Dividing the last by the first returns a number below 1.
2. **`digits_kept(kappa, squared)`** returns $\max(0,\,16-k)$ for $k=\log_{10}\kappa$, or $\max(0,\,16-2k)$ when `squared` is true. Forgetting the clamp returns negative digits.
3. **`fit_normal_equations(X, y)`** returns `w` of shape `(d,)` through a Cholesky factor and two `solve_triangular` calls, as in the code earlier. The second call needs `L.T`.
4. **`fit_qr(X, y)`** returns `w` of shape `(d,)` from `np.linalg.qr(X, mode="reduced")` and `solve_triangular(R, Q.T @ y, lower=False)`. Full mode gives `Q` shape `(n, n)`, and then the product no longer lines up.
5. **`fit_svd(X, y, rcond)`** returns the minimum-norm `w` of shape `(d,)`. Build `keep = s > rcond * s[0]`, then use `vt[keep]`, `U[:, keep]`, and `s[keep]`. Dividing by a near-zero singular value makes the weights huge.
6. **`rank(X, rcond)`** returns an integer: how many singular values exceed `rcond * s[0]`. An all-zero matrix has `s[0] == 0`, so it must return 0.

## Cost and scaling

When the table has many more rows $n$ than columns $d$, all three routes cost about $nd^2$ operations: Cholesky is the cheapest, QR about twice that, and the SVD a few times that. Only the SVD handles repeated columns: Cholesky fails, and QR puts a zero on the diagonal of $R$.

## Where this breaks

**Columns on wild scales.** Metres beside centimetres inflate $\kappa$ even when no column repeats. Standardise each column to mean 0 and standard deviation 1, which cures the scale but not a true near-repeat.

**Nearly repeated columns.** A singular value of $10^{-6}$ is a real signal for the SVD, so it is kept, and dividing by it still amplifies noise a millionfold. The cut-off helps only when the gap is wide: look at the singular values for a sudden drop of many orders of magnitude, and put the cut-off inside that gap. For the gradual case, drop a column or add a penalty on large weights, which the chapter Ridge: a fee on large weights covers.

## The exercise

You implement `condition_number`, `digits_kept`, `fit_normal_equations`, `fit_qr`, `fit_svd`, and `rank`. The checks cover known condition numbers, the weights $(0.7,0.3)$ for the nearly repeated houses, and $(2/3,1/2)$ from all three solvers for the three houses. They also check that `fit_svd` returns $(3/35,6/35)$ for the repeated columns, and that `rank` counts correctly. At $\kappa=10^7$, your Cholesky solver must be at least 100 times less accurate than a reference that works on $X$, while QR and the SVD keep about nine digits. The checks scan each solver's source and reject `lstsq` and `pinv`.
