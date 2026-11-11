---
title: "Kernels: similarity scores and what makes one valid"
slug: 020-kernels-and-validity
part: "Part 2 — Classical models"
summary: A kernel scores how alike two points are and equals a dot product in a bigger space, and it is valid when its Gram matrix has no negative eigenvalue.
minutes: 80
gpu: false
objectives:
  - Show with a worked example that a polynomial kernel equals a dot product of explicit features.
  - Explain what the RBF kernel measures and what the dial gamma does to it.
  - Compute RBF and polynomial kernel matrices between two sets of points with one matrix product.
  - Explain why a valid kernel must give every combination of points a non-negative squared length.
  - Test a Gram matrix with its smallest eigenvalue, using a tolerance, and name a polynomial setting that fails.
  - Show why the weight that minimises a penalised loss lies in the span of the training features.
  - Centre a Gram matrix, and check that kernel PCA on a linear kernel matches the scores of the centred points.
prereqs:
  - 006-ridge-regression
  - 019-support-vector-machines
exercise: 020-kernels-and-validity
---

# Kernels: similarity scores and what makes one valid

You have eight points that lie, with some noise, along one hump and one dip of a sine curve. You fit a straight line and it misses badly: the line cannot bend, so it cuts across the hump and the dip and follows neither. The model needs curves.

The usual repair is to add features by hand: $x^2$, $x^3$, and products of the original columns. That works for one input and gets expensive fast. With 10 original features, every term up to degree 3 needs 286 columns, and you must choose them before you see the data. This chapter shows a way to get the benefit of those columns without writing them down. The idea is a *kernel*: a score for how alike two points are, which secretly equals a dot product in a much bigger feature space. You met one in [support vector machines](/lessons/019-support-vector-machines), where the table $K$ took the place of the data.

A second question follows. A colleague proposes a similarity for two products: score 1 when the products are identical and 2 when they differ. It is easy to compute, and it says each product is *more* similar to the other one than to itself. Can a kernel method trust that score? The second half of the chapter answers with a picture, a one-line test, and a short computation.

## What you already know

- A model is a weighted sum of features, and a curve is a line in the right features.
- A dot product $x^{\top}x'$ is large and positive when two vectors point the same way, and zero when they are perpendicular.
- The SVM dual touches the data only through dot products between pairs of points, the Gram matrix.
- Ridge regression adds a fee $\lambda$ on the diagonal to keep a system solvable ([ridge regression](/lessons/006-ridge-regression)).
- A symmetric matrix has real eigenvalues, and `np.linalg.eigvalsh` returns them in ascending order.

## The picture: a sum of bumps follows a curve

Look at the left half of Figure 1. The dashed line is the best straight line through the eight points, and its typical miss (root-mean-square) is about 0.63. The faint dotted curve is the sine the points were drawn from. The right half shows the goal: put a smooth bump on each training point and add the bumps up. A bump is tall at its own point and fades with distance, so near a training point the nearest bumps dominate and the curve follows the data. The typical miss falls to about 0.06. [Kernel ridge regression](/lessons/021-kernel-ridge-and-random-features) shows how the bump heights are found. Here you learn the bump itself.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Eight noisy points on a sine curve. A straight line misses the curve; a sum of bumps follows it.">
<line x1="40.0" y1="204.0" x2="280.0" y2="204.0" class="stroke-muted"/><line x1="40.0" y1="34.0" x2="40.0" y2="204.0" class="stroke-muted"/><line x1="40.0" y1="119.0" x2="280.0" y2="119.0" class="stroke-grid"/><text x="50.6" y="220.0" text-anchor="middle" class="fig-small">0</text><text x="121.2" y="220.0" text-anchor="middle" class="fig-small">2</text><text x="191.8" y="220.0" text-anchor="middle" class="fig-small">4</text><text x="262.4" y="220.0" text-anchor="middle" class="fig-small">6</text><text x="34.0" y="176.1" text-anchor="end" class="fig-small">-1</text><text x="34.0" y="123.0" text-anchor="end" class="fig-small">0</text><text x="34.0" y="69.9" text-anchor="end" class="fig-small">1</text><text x="160.0" y="22.0" text-anchor="middle" class="fig-bold">A straight line</text><path d="M40.0,134.7 L41.2,133.0 L42.4,131.2 L43.6,129.4 L44.8,127.6 L46.0,125.8 L47.2,124.0 L48.4,122.2 L49.6,120.4 L50.9,118.6 L52.1,116.8 L53.3,115.0 L54.5,113.2 L55.7,111.4 L56.9,109.6 L58.1,107.8 L59.3,106.0 L60.5,104.3 L61.7,102.5 L62.9,100.8 L64.1,99.1 L65.3,97.5 L66.5,95.8 L67.7,94.2 L68.9,92.6 L70.2,91.0 L71.4,89.5 L72.6,88.0 L73.8,86.6 L75.0,85.1 L76.2,83.8 L77.4,82.4 L78.6,81.1 L79.8,79.9 L81.0,78.7 L82.2,77.5 L83.4,76.4 L84.6,75.3 L85.8,74.3 L87.0,73.4 L88.2,72.5 L89.4,71.6 L90.7,70.8 L91.9,70.1 L93.1,69.4 L94.3,68.8 L95.5,68.2 L96.7,67.7 L97.9,67.3 L99.1,66.9 L100.3,66.6 L101.5,66.3 L102.7,66.1 L103.9,66.0 L105.1,65.9 L106.3,65.9 L107.5,65.9 L108.7,66.0 L109.9,66.2 L111.2,66.4 L112.4,66.7 L113.6,67.1 L114.8,67.5 L116.0,68.0 L117.2,68.5 L118.4,69.1 L119.6,69.8 L120.8,70.5 L122.0,71.2 L123.2,72.1 L124.4,72.9 L125.6,73.9 L126.8,74.8 L128.0,75.9 L129.2,77.0 L130.5,78.1 L131.7,79.3 L132.9,80.5 L134.1,81.8 L135.3,83.1 L136.5,84.5 L137.7,85.9 L138.9,87.3 L140.1,88.8 L141.3,90.3 L142.5,91.8 L143.7,93.4 L144.9,95.0 L146.1,96.6 L147.3,98.3 L148.5,100.0 L149.7,101.7 L151.0,103.4 L152.2,105.2 L153.4,106.9 L154.6,108.7 L155.8,110.5 L157.0,112.3 L158.2,114.1 L159.4,115.9 L160.6,117.7 L161.8,119.5 L163.0,121.3 L164.2,123.1 L165.4,124.9 L166.6,126.7 L167.8,128.5 L169.0,130.3 L170.3,132.1 L171.5,133.8 L172.7,135.6 L173.9,137.3 L175.1,139.0 L176.3,140.6 L177.5,142.3 L178.7,143.9 L179.9,145.5 L181.1,147.1 L182.3,148.6 L183.5,150.1 L184.7,151.5 L185.9,152.9 L187.1,154.3 L188.3,155.7 L189.5,156.9 L190.8,158.2 L192.0,159.4 L193.2,160.6 L194.4,161.7 L195.6,162.7 L196.8,163.7 L198.0,164.7 L199.2,165.6 L200.4,166.4 L201.6,167.2 L202.8,167.9 L204.0,168.6 L205.2,169.2 L206.4,169.8 L207.6,170.3 L208.8,170.7 L210.1,171.1 L211.3,171.4 L212.5,171.7 L213.7,171.9 L214.9,172.0 L216.1,172.1 L217.3,172.1 L218.5,172.1 L219.7,172.0 L220.9,171.8 L222.1,171.5 L223.3,171.3 L224.5,170.9 L225.7,170.5 L226.9,170.0 L228.1,169.5 L229.3,168.9 L230.6,168.2 L231.8,167.5 L233.0,166.7 L234.2,165.9 L235.4,165.0 L236.6,164.1 L237.8,163.1 L239.0,162.1 L240.2,161.0 L241.4,159.8 L242.6,158.6 L243.8,157.4 L245.0,156.1 L246.2,154.8 L247.4,153.5 L248.6,152.1 L249.8,150.6 L251.1,149.1 L252.3,147.6 L253.5,146.1 L254.7,144.5 L255.9,142.9 L257.1,141.3 L258.3,139.6 L259.5,137.9 L260.7,136.2 L261.9,134.5 L263.1,132.7 L264.3,131.0 L265.5,129.2 L266.7,127.4 L267.9,125.6 L269.1,123.8 L270.4,122.0 L271.6,120.2 L272.8,118.4 L274.0,116.6 L275.2,114.7 L276.4,112.9 L277.6,111.1 L278.8,109.4 L280.0,107.6" fill="none" class="stroke-grid" stroke-width="2.2" stroke-dasharray="2 3"/><path d="M40.0,81.6 L41.2,82.0 L42.4,82.4 L43.6,82.8 L44.8,83.2 L46.0,83.6 L47.2,84.0 L48.4,84.4 L49.6,84.8 L50.9,85.2 L52.1,85.6 L53.3,86.0 L54.5,86.4 L55.7,86.8 L56.9,87.2 L58.1,87.6 L59.3,88.0 L60.5,88.4 L61.7,88.8 L62.9,89.1 L64.1,89.5 L65.3,89.9 L66.5,90.3 L67.7,90.7 L68.9,91.1 L70.2,91.5 L71.4,91.9 L72.6,92.3 L73.8,92.7 L75.0,93.1 L76.2,93.5 L77.4,93.9 L78.6,94.3 L79.8,94.7 L81.0,95.1 L82.2,95.5 L83.4,95.9 L84.6,96.3 L85.8,96.7 L87.0,97.1 L88.2,97.5 L89.4,97.9 L90.7,98.3 L91.9,98.7 L93.1,99.1 L94.3,99.5 L95.5,99.9 L96.7,100.3 L97.9,100.7 L99.1,101.1 L100.3,101.5 L101.5,101.9 L102.7,102.2 L103.9,102.6 L105.1,103.0 L106.3,103.4 L107.5,103.8 L108.7,104.2 L109.9,104.6 L111.2,105.0 L112.4,105.4 L113.6,105.8 L114.8,106.2 L116.0,106.6 L117.2,107.0 L118.4,107.4 L119.6,107.8 L120.8,108.2 L122.0,108.6 L123.2,109.0 L124.4,109.4 L125.6,109.8 L126.8,110.2 L128.0,110.6 L129.2,111.0 L130.5,111.4 L131.7,111.8 L132.9,112.2 L134.1,112.6 L135.3,113.0 L136.5,113.4 L137.7,113.8 L138.9,114.2 L140.1,114.6 L141.3,115.0 L142.5,115.4 L143.7,115.7 L144.9,116.1 L146.1,116.5 L147.3,116.9 L148.5,117.3 L149.7,117.7 L151.0,118.1 L152.2,118.5 L153.4,118.9 L154.6,119.3 L155.8,119.7 L157.0,120.1 L158.2,120.5 L159.4,120.9 L160.6,121.3 L161.8,121.7 L163.0,122.1 L164.2,122.5 L165.4,122.9 L166.6,123.3 L167.8,123.7 L169.0,124.1 L170.3,124.5 L171.5,124.9 L172.7,125.3 L173.9,125.7 L175.1,126.1 L176.3,126.5 L177.5,126.9 L178.7,127.3 L179.9,127.7 L181.1,128.1 L182.3,128.5 L183.5,128.8 L184.7,129.2 L185.9,129.6 L187.1,130.0 L188.3,130.4 L189.5,130.8 L190.8,131.2 L192.0,131.6 L193.2,132.0 L194.4,132.4 L195.6,132.8 L196.8,133.2 L198.0,133.6 L199.2,134.0 L200.4,134.4 L201.6,134.8 L202.8,135.2 L204.0,135.6 L205.2,136.0 L206.4,136.4 L207.6,136.8 L208.8,137.2 L210.1,137.6 L211.3,138.0 L212.5,138.4 L213.7,138.8 L214.9,139.2 L216.1,139.6 L217.3,140.0 L218.5,140.4 L219.7,140.8 L220.9,141.2 L222.1,141.6 L223.3,141.9 L224.5,142.3 L225.7,142.7 L226.9,143.1 L228.1,143.5 L229.3,143.9 L230.6,144.3 L231.8,144.7 L233.0,145.1 L234.2,145.5 L235.4,145.9 L236.6,146.3 L237.8,146.7 L239.0,147.1 L240.2,147.5 L241.4,147.9 L242.6,148.3 L243.8,148.7 L245.0,149.1 L246.2,149.5 L247.4,149.9 L248.6,150.3 L249.8,150.7 L251.1,151.1 L252.3,151.5 L253.5,151.9 L254.7,152.3 L255.9,152.7 L257.1,153.1 L258.3,153.5 L259.5,153.9 L260.7,154.3 L261.9,154.7 L263.1,155.1 L264.3,155.4 L265.5,155.8 L266.7,156.2 L267.9,156.6 L269.1,157.0 L270.4,157.4 L271.6,157.8 L272.8,158.2 L274.0,158.6 L275.2,159.0 L276.4,159.4 L277.6,159.8 L278.8,160.2 L280.0,160.6" fill="none" class="stroke-bad dashed" stroke-width="2.2"/><circle cx="44.7" cy="134.7" r="4.5" class="fill-a"/><circle cx="78.6" cy="54.7" r="4.5" class="fill-a"/><circle cx="118.2" cy="67.2" r="4.5" class="fill-a"/><circle cx="146.8" cy="100.3" r="4.5" class="fill-a"/><circle cx="171.6" cy="136.3" r="4.5" class="fill-a"/><circle cx="208.0" cy="175.8" r="4.5" class="fill-a"/><circle cx="240.4" cy="169.2" r="4.5" class="fill-a"/><circle cx="267.5" cy="129.3" r="4.5" class="fill-a"/><line x1="330.0" y1="204.0" x2="570.0" y2="204.0" class="stroke-muted"/><line x1="330.0" y1="34.0" x2="330.0" y2="204.0" class="stroke-muted"/><line x1="330.0" y1="119.0" x2="570.0" y2="119.0" class="stroke-grid"/><text x="340.6" y="220.0" text-anchor="middle" class="fig-small">0</text><text x="411.2" y="220.0" text-anchor="middle" class="fig-small">2</text><text x="481.8" y="220.0" text-anchor="middle" class="fig-small">4</text><text x="552.4" y="220.0" text-anchor="middle" class="fig-small">6</text><text x="324.0" y="176.1" text-anchor="end" class="fig-small">-1</text><text x="324.0" y="123.0" text-anchor="end" class="fig-small">0</text><text x="324.0" y="69.9" text-anchor="end" class="fig-small">1</text><text x="450.0" y="22.0" text-anchor="middle" class="fig-bold">A sum of bumps</text><path d="M330.0,134.7 L331.2,133.0 L332.4,131.2 L333.6,129.4 L334.8,127.6 L336.0,125.8 L337.2,124.0 L338.4,122.2 L339.6,120.4 L340.9,118.6 L342.1,116.8 L343.3,115.0 L344.5,113.2 L345.7,111.4 L346.9,109.6 L348.1,107.8 L349.3,106.0 L350.5,104.3 L351.7,102.5 L352.9,100.8 L354.1,99.1 L355.3,97.5 L356.5,95.8 L357.7,94.2 L358.9,92.6 L360.2,91.0 L361.4,89.5 L362.6,88.0 L363.8,86.6 L365.0,85.1 L366.2,83.8 L367.4,82.4 L368.6,81.1 L369.8,79.9 L371.0,78.7 L372.2,77.5 L373.4,76.4 L374.6,75.3 L375.8,74.3 L377.0,73.4 L378.2,72.5 L379.4,71.6 L380.7,70.8 L381.9,70.1 L383.1,69.4 L384.3,68.8 L385.5,68.2 L386.7,67.7 L387.9,67.3 L389.1,66.9 L390.3,66.6 L391.5,66.3 L392.7,66.1 L393.9,66.0 L395.1,65.9 L396.3,65.9 L397.5,65.9 L398.7,66.0 L399.9,66.2 L401.2,66.4 L402.4,66.7 L403.6,67.1 L404.8,67.5 L406.0,68.0 L407.2,68.5 L408.4,69.1 L409.6,69.8 L410.8,70.5 L412.0,71.2 L413.2,72.1 L414.4,72.9 L415.6,73.9 L416.8,74.8 L418.0,75.9 L419.2,77.0 L420.5,78.1 L421.7,79.3 L422.9,80.5 L424.1,81.8 L425.3,83.1 L426.5,84.5 L427.7,85.9 L428.9,87.3 L430.1,88.8 L431.3,90.3 L432.5,91.8 L433.7,93.4 L434.9,95.0 L436.1,96.6 L437.3,98.3 L438.5,100.0 L439.7,101.7 L441.0,103.4 L442.2,105.2 L443.4,106.9 L444.6,108.7 L445.8,110.5 L447.0,112.3 L448.2,114.1 L449.4,115.9 L450.6,117.7 L451.8,119.5 L453.0,121.3 L454.2,123.1 L455.4,124.9 L456.6,126.7 L457.8,128.5 L459.0,130.3 L460.3,132.1 L461.5,133.8 L462.7,135.6 L463.9,137.3 L465.1,139.0 L466.3,140.6 L467.5,142.3 L468.7,143.9 L469.9,145.5 L471.1,147.1 L472.3,148.6 L473.5,150.1 L474.7,151.5 L475.9,152.9 L477.1,154.3 L478.3,155.7 L479.5,156.9 L480.8,158.2 L482.0,159.4 L483.2,160.6 L484.4,161.7 L485.6,162.7 L486.8,163.7 L488.0,164.7 L489.2,165.6 L490.4,166.4 L491.6,167.2 L492.8,167.9 L494.0,168.6 L495.2,169.2 L496.4,169.8 L497.6,170.3 L498.8,170.7 L500.1,171.1 L501.3,171.4 L502.5,171.7 L503.7,171.9 L504.9,172.0 L506.1,172.1 L507.3,172.1 L508.5,172.1 L509.7,172.0 L510.9,171.8 L512.1,171.5 L513.3,171.3 L514.5,170.9 L515.7,170.5 L516.9,170.0 L518.1,169.5 L519.3,168.9 L520.6,168.2 L521.8,167.5 L523.0,166.7 L524.2,165.9 L525.4,165.0 L526.6,164.1 L527.8,163.1 L529.0,162.1 L530.2,161.0 L531.4,159.8 L532.6,158.6 L533.8,157.4 L535.0,156.1 L536.2,154.8 L537.4,153.5 L538.6,152.1 L539.8,150.6 L541.1,149.1 L542.3,147.6 L543.5,146.1 L544.7,144.5 L545.9,142.9 L547.1,141.3 L548.3,139.6 L549.5,137.9 L550.7,136.2 L551.9,134.5 L553.1,132.7 L554.3,131.0 L555.5,129.2 L556.7,127.4 L557.9,125.6 L559.1,123.8 L560.4,122.0 L561.6,120.2 L562.8,118.4 L564.0,116.6 L565.2,114.7 L566.4,112.9 L567.6,111.1 L568.8,109.4 L570.0,107.6" fill="none" class="stroke-grid" stroke-width="2.2" stroke-dasharray="2 3"/><path d="M330.0,136.6 L331.2,135.3 L332.4,133.9 L333.6,132.4 L334.8,130.7 L336.0,128.9 L337.2,126.9 L338.4,124.8 L339.6,122.5 L340.9,120.1 L342.1,117.7 L343.3,115.1 L344.5,112.4 L345.7,109.7 L346.9,106.9 L348.1,104.0 L349.3,101.1 L350.5,98.2 L351.7,95.3 L352.9,92.4 L354.1,89.5 L355.3,86.6 L356.5,83.8 L357.7,81.1 L358.9,78.4 L360.2,75.8 L361.4,73.4 L362.6,71.0 L363.8,68.8 L365.0,66.7 L366.2,64.7 L367.4,62.9 L368.6,61.2 L369.8,59.7 L371.0,58.3 L372.2,57.1 L373.4,56.0 L374.6,55.1 L375.8,54.3 L377.0,53.7 L378.2,53.3 L379.4,53.0 L380.7,52.8 L381.9,52.7 L383.1,52.8 L384.3,53.0 L385.5,53.2 L386.7,53.6 L387.9,54.1 L389.1,54.7 L390.3,55.3 L391.5,56.0 L392.7,56.8 L393.9,57.6 L395.1,58.5 L396.3,59.4 L397.5,60.3 L398.7,61.3 L399.9,62.3 L401.2,63.4 L402.4,64.4 L403.6,65.5 L404.8,66.6 L406.0,67.7 L407.2,68.9 L408.4,70.0 L409.6,71.2 L410.8,72.3 L412.0,73.5 L413.2,74.7 L414.4,75.9 L415.6,77.1 L416.8,78.4 L418.0,79.6 L419.2,80.9 L420.5,82.2 L421.7,83.5 L422.9,84.8 L424.1,86.2 L425.3,87.5 L426.5,88.9 L427.7,90.3 L428.9,91.7 L430.1,93.2 L431.3,94.7 L432.5,96.2 L433.7,97.7 L434.9,99.2 L436.1,100.7 L437.3,102.3 L438.5,103.9 L439.7,105.5 L441.0,107.1 L442.2,108.7 L443.4,110.3 L444.6,111.9 L445.8,113.6 L447.0,115.2 L448.2,116.9 L449.4,118.5 L450.6,120.1 L451.8,121.8 L453.0,123.4 L454.2,125.1 L455.4,126.7 L456.6,128.3 L457.8,129.9 L459.0,131.5 L460.3,133.1 L461.5,134.7 L462.7,136.3 L463.9,137.8 L465.1,139.4 L466.3,140.9 L467.5,142.4 L468.7,143.9 L469.9,145.4 L471.1,146.9 L472.3,148.3 L473.5,149.7 L474.7,151.1 L475.9,152.5 L477.1,153.9 L478.3,155.2 L479.5,156.5 L480.8,157.8 L482.0,159.1 L483.2,160.3 L484.4,161.5 L485.6,162.7 L486.8,163.8 L488.0,164.9 L489.2,166.0 L490.4,167.0 L491.6,168.0 L492.8,168.9 L494.0,169.8 L495.2,170.7 L496.4,171.4 L497.6,172.2 L498.8,172.9 L500.1,173.5 L501.3,174.1 L502.5,174.6 L503.7,175.0 L504.9,175.4 L506.1,175.7 L507.3,175.9 L508.5,176.1 L509.7,176.2 L510.9,176.2 L512.1,176.2 L513.3,176.0 L514.5,175.8 L515.7,175.5 L516.9,175.1 L518.1,174.6 L519.3,174.1 L520.6,173.5 L521.8,172.8 L523.0,172.0 L524.2,171.1 L525.4,170.1 L526.6,169.1 L527.8,168.0 L529.0,166.8 L530.2,165.6 L531.4,164.3 L532.6,162.9 L533.8,161.5 L535.0,160.0 L536.2,158.5 L537.4,157.0 L538.6,155.4 L539.8,153.7 L541.1,152.1 L542.3,150.4 L543.5,148.8 L544.7,147.1 L545.9,145.4 L547.1,143.7 L548.3,142.1 L549.5,140.4 L550.7,138.8 L551.9,137.3 L553.1,135.7 L554.3,134.2 L555.5,132.8 L556.7,131.4 L557.9,130.0 L559.1,128.8 L560.4,127.5 L561.6,126.4 L562.8,125.3 L564.0,124.3 L565.2,123.3 L566.4,122.4 L567.6,121.6 L568.8,120.8 L570.0,120.2" fill="none" class="stroke-b" stroke-width="2.2"/><circle cx="334.7" cy="134.7" r="4.5" class="fill-a"/><circle cx="368.6" cy="54.7" r="4.5" class="fill-a"/><circle cx="408.2" cy="67.2" r="4.5" class="fill-a"/><circle cx="436.8" cy="100.3" r="4.5" class="fill-a"/><circle cx="461.6" cy="136.3" r="4.5" class="fill-a"/><circle cx="498.0" cy="175.8" r="4.5" class="fill-a"/><circle cx="530.4" cy="169.2" r="4.5" class="fill-a"/><circle cx="557.5" cy="129.3" r="4.5" class="fill-a"/><text x="160.0" y="236.0" text-anchor="middle" class="fig-small">line misses the curve</text><text x="450.0" y="236.0" text-anchor="middle" class="fig-small">curve follows the points</text>
</svg>
Figure 1. Left: no straight line can follow a hump and a dip. Right: the same eight points fitted by a sum of smooth bumps (gamma = 1), one centred on each point. Notice that the orange curve bends where the data bends, and that it needs no hand-made columns.
```

A bump needs a rule for how tall it is at distance $d$ from its centre: 1 for a point on top of the centre, and smaller as points move away. A function that scores two points this way is a kernel. Adding features by hand hides two costs. The first is the count, since a degree-$p$ polynomial in $d$ inputs has on the order of $d^p$ columns. The second is the choice: a Gaussian bump in one dimension is a sum of polynomial terms of *every* degree, so no finite list of columns reproduces it. A kernel avoids both by working with the dot product between two points' features, not with the features themselves.

## A kernel is a dot product you never build

Take two points in the plane, $x = (1, 2)$ and $x' = (3, 1)$. Build six features for each one:

$$
\varphi(x) = \left(1,\ \sqrt{2}\,x_1,\ \sqrt{2}\,x_2,\ x_1^2,\ \sqrt{2}\,x_1 x_2,\ x_2^2\right).
$$

For $x$ this is $(1, 1.41, 2.83, 1, 2.83, 4)$ and for $x'$ it is $(1, 4.24, 1.41, 9, 4.24, 1)$. Multiply the two lists entry by entry and you get $1, 6, 4, 9, 12, 4$, which add to 36. Figure 2 shows both routes.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Two routes to 36: six explicit features per point, or one dot product of the original points.">
<text x="8.0" y="56.0" text-anchor="start" class="fig-small">φ(x), x = (1, 2)</text><rect x="150" y="34" width="52" height="32" rx="4" class="fill-soft-a"/><text x="176.0" y="55.0" text-anchor="middle">1</text><rect x="206" y="34" width="52" height="32" rx="4" class="fill-soft-a"/><text x="232.0" y="55.0" text-anchor="middle">1.41</text><rect x="262" y="34" width="52" height="32" rx="4" class="fill-soft-a"/><text x="288.0" y="55.0" text-anchor="middle">2.83</text><rect x="318" y="34" width="52" height="32" rx="4" class="fill-soft-a"/><text x="344.0" y="55.0" text-anchor="middle">1</text><rect x="374" y="34" width="52" height="32" rx="4" class="fill-soft-a"/><text x="400.0" y="55.0" text-anchor="middle">2.83</text><rect x="430" y="34" width="52" height="32" rx="4" class="fill-soft-a"/><text x="456.0" y="55.0" text-anchor="middle">4</text><text x="8.0" y="100.0" text-anchor="start" class="fig-small">φ(x'), x' = (3, 1)</text><rect x="150" y="78" width="52" height="32" rx="4" class="fill-soft-a"/><text x="176.0" y="99.0" text-anchor="middle">1</text><rect x="206" y="78" width="52" height="32" rx="4" class="fill-soft-a"/><text x="232.0" y="99.0" text-anchor="middle">4.24</text><rect x="262" y="78" width="52" height="32" rx="4" class="fill-soft-a"/><text x="288.0" y="99.0" text-anchor="middle">1.41</text><rect x="318" y="78" width="52" height="32" rx="4" class="fill-soft-a"/><text x="344.0" y="99.0" text-anchor="middle">9</text><rect x="374" y="78" width="52" height="32" rx="4" class="fill-soft-a"/><text x="400.0" y="99.0" text-anchor="middle">4.24</text><rect x="430" y="78" width="52" height="32" rx="4" class="fill-soft-a"/><text x="456.0" y="99.0" text-anchor="middle">1</text><text x="8.0" y="144.0" text-anchor="start" class="fig-small">products</text><rect x="150" y="122" width="52" height="32" rx="4" class="fill-soft-b"/><text x="176.0" y="143.0" text-anchor="middle">1</text><rect x="206" y="122" width="52" height="32" rx="4" class="fill-soft-b"/><text x="232.0" y="143.0" text-anchor="middle">6</text><rect x="262" y="122" width="52" height="32" rx="4" class="fill-soft-b"/><text x="288.0" y="143.0" text-anchor="middle">4</text><rect x="318" y="122" width="52" height="32" rx="4" class="fill-soft-b"/><text x="344.0" y="143.0" text-anchor="middle">9</text><rect x="374" y="122" width="52" height="32" rx="4" class="fill-soft-b"/><text x="400.0" y="143.0" text-anchor="middle">12</text><rect x="430" y="122" width="52" height="32" rx="4" class="fill-soft-b"/><text x="456.0" y="143.0" text-anchor="middle">4</text><text x="490.0" y="143.0" text-anchor="start" class="fig-bold">sum = 36</text><text x="318.0" y="22.0" text-anchor="middle" class="fig-small">six features each</text><rect x="8" y="176" width="566" height="40" rx="4" class="fill-soft-c"/><text x="18.0" y="201.0" text-anchor="start">One dot product of the originals: x·x' = 1·3 + 2·1 = 5, then (5 + 1)² = 36</text>
</svg>
Figure 2. The six-feature route and the one-dot-product route land on the same number, 36. Notice that the bottom route never mentions a square or a product of coordinates.
```

Take the shortcut. The ordinary dot product of the two points is $1 \cdot 3 + 2 \cdot 1 = 5$. Add 1 to get 6, and square it: 36. The routes agree because squaring $(x_1x_1' + x_2x_2' + 1)$ produces exactly the six terms of $\varphi$, and each $\sqrt{2}$ comes from splitting the factor 2 of a cross term between the two points:

$$
\left(x^{\top}x' + 1\right)^2 = \varphi(x)^{\top}\varphi(x') .
$$

Read it aloud: "one plus the dot product, squared, equals the dot product of the six-entry feature lists." A function $k(x, x')$ with this property is a *kernel*, and $\varphi$ is its *feature map*. The shortcut costs one dot product of two-entry points, while the explicit route costs six features per point, and 66 for 10 inputs.

```python
phi = poly_features_degree2(X)      # shape (n, 6)
K_explicit = phi @ phi.T            # six features per point
K_shortcut = (X @ X.T + 1.0) ** 2   # one dot product per pair
```

The *linear kernel* is the plain dot product $x^{\top}x'$. The polynomial kernel $(x^{\top}x' + c)^p$ generalises it, and a positive $c$ mixes in all lower degrees, which is why $c = 1$ produced a constant, linear, and quadratic feature together.

## The best weight lies in the span of the training points

You can store the trained weight as one coefficient per training point. That holds when the loss sees $w$ only through the predictions $w^{\top}\varphi(x_i)$ and you add a penalty $\lambda\lVert w\rVert^2$ with $\lambda > 0$.

Split $w = w_{\parallel} + w_{\perp}$. The piece $w_{\parallel}$ lies in the span of $\varphi(x_1),\ldots,\varphi(x_n)$, and $w_{\perp}$ is at right angles to every one of them. Perpendicular pieces add in squared length:

$$
\lVert w\rVert^2 = \lVert w_{\parallel}\rVert^2 + \lVert w_{\perp}\rVert^2 .
$$

The prediction $w^{\top}\varphi(x_i)$ drops $w_{\perp}$, because $w_{\perp}^{\top}\varphi(x_i) = 0$. The misses do not change, and the penalty grows whenever $w_{\perp}$ is not zero. Every minimizer therefore lies in the span, $w = \sum_i \alpha_i\,\varphi(x_i)$. This is the *representer theorem*. With $\lambda = 0$, a perpendicular piece costs nothing, so a minimizer need not lie in the span.

### Work the split by hand

Take $\varphi(x_1) = (1, 0, 0)$, $\varphi(x_2) = (0, 1, 0)$, labels $4$ and $6$, and $\lambda = 1$. For $w = (a, b, c)$ the objective is the sum of squared misses plus $\lambda$ times the squared length:

$$
J = (4 - a)^2 + (6 - b)^2 + \lambda(a^2 + b^2 + c^2).
$$

The derivative in $a$ is $-2(4 - a) + 2\lambda a$. Set it to zero and divide by 2, which leaves $4 - a = \lambda a$. The same step gives $6 - b = \lambda b$ and $\lambda c = 0$. With $\lambda = 1$ the solution is $(a, b, c) = (2, 3, 0)$, so $w = 2\varphi(x_1) + 3\varphi(x_2)$. The predictions are $2$ and $3$. The squared misses are $(4 - 2)^2 + (6 - 3)^2 = 13$, the squared length is $4 + 9 = 13$, and $J = 26$.

The weight $(2, 3, 4)$ predicts $2$ and $3$ as well, so the misses stay $13$. Its squared length is $4 + 9 + 16 = 29$, and $J = 42$. The extra $16$ is the squared length of $(0, 0, 4)$. Figure 3 shows that piece at right angles to the span.

```figure
<svg viewBox="0 0 600 240" role="img" aria-label="The weight (2, 3, 4) split into a span piece (2, 3, 0) and a perpendicular piece (0, 0, 4).">
  <defs>
    <marker id="c018-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="fill-ink"/>
    </marker>
  </defs>
  <line x1="40" y1="185" x2="560" y2="185" class="stroke-muted"/>
  <line x1="150" y1="185" x2="251" y2="185" class="stroke-a" stroke-width="2.2" marker-end="url(#c018-arrow)"/>
  <line x1="251" y1="185" x2="251" y2="73" class="stroke-c" stroke-width="2.2" marker-end="url(#c018-arrow)"/>
  <line x1="150" y1="185" x2="251" y2="73" class="stroke-b" stroke-width="2.2" marker-end="url(#c018-arrow)"/>
  <path d="M239,185 L239,173 L251,173" fill="none" class="stroke-ink"/>
  <circle cx="150" cy="185" r="3.5" class="fill-ink"/>
  <text x="188" y="208" text-anchor="middle" class="fig-small">in the span, (2, 3, 0)</text>
  <text x="262" y="130" text-anchor="start" class="fig-small">(0, 0, 4)</text>
  <text x="48" y="120" text-anchor="start" class="fig-small">w = (2, 3, 4)</text>
  <text x="430" y="208" text-anchor="middle" class="fig-small">span of the training features</text>
</svg>
Figure 3. The span of the two training features, seen edge-on. The weight (2, 3, 4) and the shadow (2, 3, 0) both predict 2 and 3. Notice that the upright piece has squared length 16, so the objective rises from 26 to 42.
```

[Kernel ridge regression](/lessons/021-kernel-ridge-and-random-features) solves for $\alpha_i$. A new point then scores $\sum_i \alpha_i\, k(x, x_i)$.

> [!KEY]
> The weight that minimises a penalised training loss lies in the span of the training features.

## The RBF kernel: similarity that fades with distance

The *radial basis function* kernel, written RBF and also called the Gaussian kernel, is the bump from Figure 1:

$$
k(x, x') = \exp\!\left(-\gamma\,\lVert x - x'\rVert^2\right) .
$$

Read it aloud: "take the squared distance between the points, multiply by minus gamma, and exponentiate." A point compared with itself scores $e^0 = 1$, and the score falls toward 0 as the distance grows. For $x = (0, 0)$, $x' = (1, 0)$, and $\gamma = 0.5$, the squared distance is 1 and the score is $e^{-0.5} \approx 0.6065$.

The number $\gamma$ is the inverse of a width: a large $\gamma$ gives a narrow bump, and a small $\gamma$ gives a wide one (Figure 4). At $\gamma = 10$, points 0.5 apart score near 0.08 and hardly count as neighbours, while at $\gamma = 0.1$ even points 3 apart score above 0.4.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Three RBF bumps for gamma 0.1, 1, and 10 plotted against distance.">
<line x1="60.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-muted"/><line x1="60.0" y1="20.0" x2="60.0" y2="200.0" class="stroke-muted"/><text x="60.0" y="216.0" text-anchor="middle" class="fig-small">-3</text><text x="143.3" y="216.0" text-anchor="middle" class="fig-small">-2</text><text x="226.7" y="216.0" text-anchor="middle" class="fig-small">-1</text><text x="310.0" y="216.0" text-anchor="middle" class="fig-small">0</text><text x="393.3" y="216.0" text-anchor="middle" class="fig-small">1</text><text x="476.7" y="216.0" text-anchor="middle" class="fig-small">2</text><text x="560.0" y="216.0" text-anchor="middle" class="fig-small">3</text><text x="54.0" y="204.0" text-anchor="end" class="fig-small">0</text><text x="54.0" y="118.3" text-anchor="end" class="fig-small">0.5</text><text x="54.0" y="32.6" text-anchor="end" class="fig-small">1</text><text x="310.0" y="236.0" text-anchor="middle" class="fig-small">distance d from the centre point</text><path d="M60.0,130.3 L61.7,129.5 L63.3,128.6 L65.0,127.8 L66.7,126.9 L68.4,126.1 L70.0,125.2 L71.7,124.3 L73.4,123.5 L75.1,122.6 L76.7,121.7 L78.4,120.8 L80.1,119.9 L81.7,119.0 L83.4,118.2 L85.1,117.3 L86.8,116.4 L88.4,115.5 L90.1,114.6 L91.8,113.7 L93.4,112.7 L95.1,111.8 L96.8,110.9 L98.5,110.0 L100.1,109.1 L101.8,108.2 L103.5,107.2 L105.2,106.3 L106.8,105.4 L108.5,104.5 L110.2,103.5 L111.8,102.6 L113.5,101.7 L115.2,100.8 L116.9,99.8 L118.5,98.9 L120.2,98.0 L121.9,97.0 L123.5,96.1 L125.2,95.2 L126.9,94.2 L128.6,93.3 L130.2,92.4 L131.9,91.4 L133.6,90.5 L135.3,89.6 L136.9,88.6 L138.6,87.7 L140.3,86.8 L141.9,85.9 L143.6,84.9 L145.3,84.0 L147.0,83.1 L148.6,82.2 L150.3,81.3 L152.0,80.4 L153.6,79.4 L155.3,78.5 L157.0,77.6 L158.7,76.7 L160.3,75.8 L162.0,74.9 L163.7,74.1 L165.4,73.2 L167.0,72.3 L168.7,71.4 L170.4,70.5 L172.0,69.7 L173.7,68.8 L175.4,67.9 L177.1,67.1 L178.7,66.2 L180.4,65.4 L182.1,64.6 L183.7,63.7 L185.4,62.9 L187.1,62.1 L188.8,61.3 L190.4,60.5 L192.1,59.7 L193.8,58.9 L195.5,58.1 L197.1,57.3 L198.8,56.5 L200.5,55.8 L202.1,55.0 L203.8,54.3 L205.5,53.5 L207.2,52.8 L208.8,52.1 L210.5,51.3 L212.2,50.6 L213.8,49.9 L215.5,49.3 L217.2,48.6 L218.9,47.9 L220.5,47.2 L222.2,46.6 L223.9,45.9 L225.6,45.3 L227.2,44.7 L228.9,44.1 L230.6,43.5 L232.2,42.9 L233.9,42.3 L235.6,41.7 L237.3,41.1 L238.9,40.6 L240.6,40.1 L242.3,39.5 L243.9,39.0 L245.6,38.5 L247.3,38.0 L249.0,37.5 L250.6,37.1 L252.3,36.6 L254.0,36.1 L255.7,35.7 L257.3,35.3 L259.0,34.9 L260.7,34.5 L262.3,34.1 L264.0,33.7 L265.7,33.4 L267.4,33.0 L269.0,32.7 L270.7,32.3 L272.4,32.0 L274.0,31.7 L275.7,31.4 L277.4,31.2 L279.1,30.9 L280.7,30.7 L282.4,30.4 L284.1,30.2 L285.8,30.0 L287.4,29.8 L289.1,29.6 L290.8,29.5 L292.4,29.3 L294.1,29.2 L295.8,29.1 L297.5,29.0 L299.1,28.9 L300.8,28.8 L302.5,28.7 L304.1,28.7 L305.8,28.6 L307.5,28.6 L309.2,28.6 L310.8,28.6 L312.5,28.6 L314.2,28.6 L315.9,28.7 L317.5,28.7 L319.2,28.8 L320.9,28.9 L322.5,29.0 L324.2,29.1 L325.9,29.2 L327.6,29.3 L329.2,29.5 L330.9,29.6 L332.6,29.8 L334.2,30.0 L335.9,30.2 L337.6,30.4 L339.3,30.7 L340.9,30.9 L342.6,31.2 L344.3,31.4 L346.0,31.7 L347.6,32.0 L349.3,32.3 L351.0,32.7 L352.6,33.0 L354.3,33.4 L356.0,33.7 L357.7,34.1 L359.3,34.5 L361.0,34.9 L362.7,35.3 L364.3,35.7 L366.0,36.1 L367.7,36.6 L369.4,37.1 L371.0,37.5 L372.7,38.0 L374.4,38.5 L376.1,39.0 L377.7,39.5 L379.4,40.1 L381.1,40.6 L382.7,41.1 L384.4,41.7 L386.1,42.3 L387.8,42.9 L389.4,43.5 L391.1,44.1 L392.8,44.7 L394.4,45.3 L396.1,45.9 L397.8,46.6 L399.5,47.2 L401.1,47.9 L402.8,48.6 L404.5,49.3 L406.2,49.9 L407.8,50.6 L409.5,51.3 L411.2,52.1 L412.8,52.8 L414.5,53.5 L416.2,54.3 L417.9,55.0 L419.5,55.8 L421.2,56.5 L422.9,57.3 L424.5,58.1 L426.2,58.9 L427.9,59.7 L429.6,60.5 L431.2,61.3 L432.9,62.1 L434.6,62.9 L436.3,63.7 L437.9,64.6 L439.6,65.4 L441.3,66.2 L442.9,67.1 L444.6,67.9 L446.3,68.8 L448.0,69.7 L449.6,70.5 L451.3,71.4 L453.0,72.3 L454.6,73.2 L456.3,74.1 L458.0,74.9 L459.7,75.8 L461.3,76.7 L463.0,77.6 L464.7,78.5 L466.4,79.4 L468.0,80.4 L469.7,81.3 L471.4,82.2 L473.0,83.1 L474.7,84.0 L476.4,84.9 L478.1,85.9 L479.7,86.8 L481.4,87.7 L483.1,88.6 L484.7,89.6 L486.4,90.5 L488.1,91.4 L489.8,92.4 L491.4,93.3 L493.1,94.2 L494.8,95.2 L496.5,96.1 L498.1,97.0 L499.8,98.0 L501.5,98.9 L503.1,99.8 L504.8,100.8 L506.5,101.7 L508.2,102.6 L509.8,103.5 L511.5,104.5 L513.2,105.4 L514.8,106.3 L516.5,107.2 L518.2,108.2 L519.9,109.1 L521.5,110.0 L523.2,110.9 L524.9,111.8 L526.6,112.7 L528.2,113.7 L529.9,114.6 L531.6,115.5 L533.2,116.4 L534.9,117.3 L536.6,118.2 L538.3,119.0 L539.9,119.9 L541.6,120.8 L543.3,121.7 L544.9,122.6 L546.6,123.5 L548.3,124.3 L550.0,125.2 L551.6,126.1 L553.3,126.9 L555.0,127.8 L556.7,128.6 L558.3,129.5 L560.0,130.3" fill="none" class="stroke-a" stroke-width="2.2"/><path d="M60.0,200.0 L61.7,200.0 L63.3,200.0 L65.0,200.0 L66.7,200.0 L68.4,200.0 L70.0,200.0 L71.7,200.0 L73.4,199.9 L75.1,199.9 L76.7,199.9 L78.4,199.9 L80.1,199.9 L81.7,199.9 L83.4,199.9 L85.1,199.9 L86.8,199.9 L88.4,199.9 L90.1,199.8 L91.8,199.8 L93.4,199.8 L95.1,199.8 L96.8,199.8 L98.5,199.7 L100.1,199.7 L101.8,199.7 L103.5,199.6 L105.2,199.6 L106.8,199.6 L108.5,199.5 L110.2,199.5 L111.8,199.4 L113.5,199.3 L115.2,199.3 L116.9,199.2 L118.5,199.1 L120.2,199.0 L121.9,199.0 L123.5,198.9 L125.2,198.7 L126.9,198.6 L128.6,198.5 L130.2,198.4 L131.9,198.2 L133.6,198.1 L135.3,197.9 L136.9,197.7 L138.6,197.5 L140.3,197.3 L141.9,197.1 L143.6,196.8 L145.3,196.6 L147.0,196.3 L148.6,196.0 L150.3,195.6 L152.0,195.3 L153.6,194.9 L155.3,194.5 L157.0,194.1 L158.7,193.7 L160.3,193.2 L162.0,192.7 L163.7,192.1 L165.4,191.6 L167.0,191.0 L168.7,190.3 L170.4,189.7 L172.0,188.9 L173.7,188.2 L175.4,187.4 L177.1,186.5 L178.7,185.7 L180.4,184.7 L182.1,183.8 L183.7,182.7 L185.4,181.7 L187.1,180.5 L188.8,179.4 L190.4,178.1 L192.1,176.8 L193.8,175.5 L195.5,174.1 L197.1,172.6 L198.8,171.1 L200.5,169.5 L202.1,167.9 L203.8,166.2 L205.5,164.4 L207.2,162.6 L208.8,160.7 L210.5,158.8 L212.2,156.8 L213.8,154.7 L215.5,152.6 L217.2,150.4 L218.9,148.2 L220.5,145.9 L222.2,143.5 L223.9,141.1 L225.6,138.6 L227.2,136.1 L228.9,133.5 L230.6,130.9 L232.2,128.2 L233.9,125.5 L235.6,122.8 L237.3,120.0 L238.9,117.2 L240.6,114.3 L242.3,111.4 L243.9,108.5 L245.6,105.6 L247.3,102.7 L249.0,99.7 L250.6,96.8 L252.3,93.8 L254.0,90.9 L255.7,88.0 L257.3,85.0 L259.0,82.1 L260.7,79.2 L262.3,76.4 L264.0,73.6 L265.7,70.8 L267.4,68.1 L269.0,65.4 L270.7,62.8 L272.4,60.2 L274.0,57.7 L275.7,55.3 L277.4,52.9 L279.1,50.6 L280.7,48.5 L282.4,46.4 L284.1,44.4 L285.8,42.5 L287.4,40.7 L289.1,39.0 L290.8,37.5 L292.4,36.0 L294.1,34.7 L295.8,33.5 L297.5,32.4 L299.1,31.5 L300.8,30.6 L302.5,30.0 L304.1,29.4 L305.8,29.0 L307.5,28.7 L309.2,28.6 L310.8,28.6 L312.5,28.7 L314.2,29.0 L315.9,29.4 L317.5,30.0 L319.2,30.6 L320.9,31.5 L322.5,32.4 L324.2,33.5 L325.9,34.7 L327.6,36.0 L329.2,37.5 L330.9,39.0 L332.6,40.7 L334.2,42.5 L335.9,44.4 L337.6,46.4 L339.3,48.5 L340.9,50.6 L342.6,52.9 L344.3,55.3 L346.0,57.7 L347.6,60.2 L349.3,62.8 L351.0,65.4 L352.6,68.1 L354.3,70.8 L356.0,73.6 L357.7,76.4 L359.3,79.2 L361.0,82.1 L362.7,85.0 L364.3,88.0 L366.0,90.9 L367.7,93.8 L369.4,96.8 L371.0,99.7 L372.7,102.7 L374.4,105.6 L376.1,108.5 L377.7,111.4 L379.4,114.3 L381.1,117.2 L382.7,120.0 L384.4,122.8 L386.1,125.5 L387.8,128.2 L389.4,130.9 L391.1,133.5 L392.8,136.1 L394.4,138.6 L396.1,141.1 L397.8,143.5 L399.5,145.9 L401.1,148.2 L402.8,150.4 L404.5,152.6 L406.2,154.7 L407.8,156.8 L409.5,158.8 L411.2,160.7 L412.8,162.6 L414.5,164.4 L416.2,166.2 L417.9,167.9 L419.5,169.5 L421.2,171.1 L422.9,172.6 L424.5,174.1 L426.2,175.5 L427.9,176.8 L429.6,178.1 L431.2,179.4 L432.9,180.5 L434.6,181.7 L436.3,182.7 L437.9,183.8 L439.6,184.7 L441.3,185.7 L442.9,186.5 L444.6,187.4 L446.3,188.2 L448.0,188.9 L449.6,189.7 L451.3,190.3 L453.0,191.0 L454.6,191.6 L456.3,192.1 L458.0,192.7 L459.7,193.2 L461.3,193.7 L463.0,194.1 L464.7,194.5 L466.4,194.9 L468.0,195.3 L469.7,195.6 L471.4,196.0 L473.0,196.3 L474.7,196.6 L476.4,196.8 L478.1,197.1 L479.7,197.3 L481.4,197.5 L483.1,197.7 L484.7,197.9 L486.4,198.1 L488.1,198.2 L489.8,198.4 L491.4,198.5 L493.1,198.6 L494.8,198.7 L496.5,198.9 L498.1,199.0 L499.8,199.0 L501.5,199.1 L503.1,199.2 L504.8,199.3 L506.5,199.3 L508.2,199.4 L509.8,199.5 L511.5,199.5 L513.2,199.6 L514.8,199.6 L516.5,199.6 L518.2,199.7 L519.9,199.7 L521.5,199.7 L523.2,199.8 L524.9,199.8 L526.6,199.8 L528.2,199.8 L529.9,199.8 L531.6,199.9 L533.2,199.9 L534.9,199.9 L536.6,199.9 L538.3,199.9 L539.9,199.9 L541.6,199.9 L543.3,199.9 L544.9,199.9 L546.6,199.9 L548.3,200.0 L550.0,200.0 L551.6,200.0 L553.3,200.0 L555.0,200.0 L556.7,200.0 L558.3,200.0 L560.0,200.0" fill="none" class="stroke-b" stroke-width="2.2"/><path d="M60.0,200.0 L61.7,200.0 L63.3,200.0 L65.0,200.0 L66.7,200.0 L68.4,200.0 L70.0,200.0 L71.7,200.0 L73.4,200.0 L75.1,200.0 L76.7,200.0 L78.4,200.0 L80.1,200.0 L81.7,200.0 L83.4,200.0 L85.1,200.0 L86.8,200.0 L88.4,200.0 L90.1,200.0 L91.8,200.0 L93.4,200.0 L95.1,200.0 L96.8,200.0 L98.5,200.0 L100.1,200.0 L101.8,200.0 L103.5,200.0 L105.2,200.0 L106.8,200.0 L108.5,200.0 L110.2,200.0 L111.8,200.0 L113.5,200.0 L115.2,200.0 L116.9,200.0 L118.5,200.0 L120.2,200.0 L121.9,200.0 L123.5,200.0 L125.2,200.0 L126.9,200.0 L128.6,200.0 L130.2,200.0 L131.9,200.0 L133.6,200.0 L135.3,200.0 L136.9,200.0 L138.6,200.0 L140.3,200.0 L141.9,200.0 L143.6,200.0 L145.3,200.0 L147.0,200.0 L148.6,200.0 L150.3,200.0 L152.0,200.0 L153.6,200.0 L155.3,200.0 L157.0,200.0 L158.7,200.0 L160.3,200.0 L162.0,200.0 L163.7,200.0 L165.4,200.0 L167.0,200.0 L168.7,200.0 L170.4,200.0 L172.0,200.0 L173.7,200.0 L175.4,200.0 L177.1,200.0 L178.7,200.0 L180.4,200.0 L182.1,200.0 L183.7,200.0 L185.4,200.0 L187.1,200.0 L188.8,200.0 L190.4,200.0 L192.1,200.0 L193.8,200.0 L195.5,200.0 L197.1,200.0 L198.8,200.0 L200.5,200.0 L202.1,200.0 L203.8,200.0 L205.5,200.0 L207.2,200.0 L208.8,200.0 L210.5,200.0 L212.2,200.0 L213.8,200.0 L215.5,200.0 L217.2,200.0 L218.9,200.0 L220.5,200.0 L222.2,200.0 L223.9,200.0 L225.6,200.0 L227.2,200.0 L228.9,200.0 L230.6,200.0 L232.2,200.0 L233.9,200.0 L235.6,199.9 L237.3,199.9 L238.9,199.9 L240.6,199.8 L242.3,199.8 L243.9,199.7 L245.6,199.6 L247.3,199.4 L249.0,199.2 L250.6,198.9 L252.3,198.6 L254.0,198.1 L255.7,197.6 L257.3,196.8 L259.0,196.0 L260.7,194.8 L262.3,193.5 L264.0,191.8 L265.7,189.9 L267.4,187.5 L269.0,184.7 L270.7,181.5 L272.4,177.7 L274.0,173.4 L275.7,168.4 L277.4,162.9 L279.1,156.8 L280.7,150.1 L282.4,142.7 L284.1,134.8 L285.8,126.5 L287.4,117.7 L289.1,108.6 L290.8,99.4 L292.4,90.0 L294.1,80.8 L295.8,71.8 L297.5,63.3 L299.1,55.4 L300.8,48.2 L302.5,42.0 L304.1,36.8 L305.8,32.8 L307.5,30.1 L309.2,28.7 L310.8,28.7 L312.5,30.1 L314.2,32.8 L315.9,36.8 L317.5,42.0 L319.2,48.2 L320.9,55.4 L322.5,63.3 L324.2,71.8 L325.9,80.8 L327.6,90.0 L329.2,99.4 L330.9,108.6 L332.6,117.7 L334.2,126.5 L335.9,134.8 L337.6,142.7 L339.3,150.1 L340.9,156.8 L342.6,162.9 L344.3,168.4 L346.0,173.4 L347.6,177.7 L349.3,181.5 L351.0,184.7 L352.6,187.5 L354.3,189.9 L356.0,191.8 L357.7,193.5 L359.3,194.8 L361.0,196.0 L362.7,196.8 L364.3,197.6 L366.0,198.1 L367.7,198.6 L369.4,198.9 L371.0,199.2 L372.7,199.4 L374.4,199.6 L376.1,199.7 L377.7,199.8 L379.4,199.8 L381.1,199.9 L382.7,199.9 L384.4,199.9 L386.1,200.0 L387.8,200.0 L389.4,200.0 L391.1,200.0 L392.8,200.0 L394.4,200.0 L396.1,200.0 L397.8,200.0 L399.5,200.0 L401.1,200.0 L402.8,200.0 L404.5,200.0 L406.2,200.0 L407.8,200.0 L409.5,200.0 L411.2,200.0 L412.8,200.0 L414.5,200.0 L416.2,200.0 L417.9,200.0 L419.5,200.0 L421.2,200.0 L422.9,200.0 L424.5,200.0 L426.2,200.0 L427.9,200.0 L429.6,200.0 L431.2,200.0 L432.9,200.0 L434.6,200.0 L436.3,200.0 L437.9,200.0 L439.6,200.0 L441.3,200.0 L442.9,200.0 L444.6,200.0 L446.3,200.0 L448.0,200.0 L449.6,200.0 L451.3,200.0 L453.0,200.0 L454.6,200.0 L456.3,200.0 L458.0,200.0 L459.7,200.0 L461.3,200.0 L463.0,200.0 L464.7,200.0 L466.4,200.0 L468.0,200.0 L469.7,200.0 L471.4,200.0 L473.0,200.0 L474.7,200.0 L476.4,200.0 L478.1,200.0 L479.7,200.0 L481.4,200.0 L483.1,200.0 L484.7,200.0 L486.4,200.0 L488.1,200.0 L489.8,200.0 L491.4,200.0 L493.1,200.0 L494.8,200.0 L496.5,200.0 L498.1,200.0 L499.8,200.0 L501.5,200.0 L503.1,200.0 L504.8,200.0 L506.5,200.0 L508.2,200.0 L509.8,200.0 L511.5,200.0 L513.2,200.0 L514.8,200.0 L516.5,200.0 L518.2,200.0 L519.9,200.0 L521.5,200.0 L523.2,200.0 L524.9,200.0 L526.6,200.0 L528.2,200.0 L529.9,200.0 L531.6,200.0 L533.2,200.0 L534.9,200.0 L536.6,200.0 L538.3,200.0 L539.9,200.0 L541.6,200.0 L543.3,200.0 L544.9,200.0 L546.6,200.0 L548.3,200.0 L550.0,200.0 L551.6,200.0 L553.3,200.0 L555.0,200.0 L556.7,200.0 L558.3,200.0 L560.0,200.0" fill="none" class="stroke-c" stroke-width="2.2"/><line x1="439.2" y1="30.3" x2="472.5" y2="30.3" class="stroke-a" stroke-width="2.2"/><text x="478.3" y="34.3" text-anchor="start" class="fig-small">γ = 0.1 (wide)</text><line x1="439.2" y1="43.1" x2="472.5" y2="43.1" class="stroke-b" stroke-width="2.2"/><text x="478.3" y="47.1" text-anchor="start" class="fig-small">γ = 1</text><line x1="439.2" y1="56.0" x2="472.5" y2="56.0" class="stroke-c" stroke-width="2.2"/><text x="478.3" y="60.0" text-anchor="start" class="fig-small">γ = 10 (narrow)</text>
</svg>
Figure 4. The same bump for three values of gamma. Notice that gamma = 10 is narrow: points a distance of 0.5 apart already score near 0.08, so they hardly count as neighbours. At gamma = 0.1 even points 3 apart score above 0.4.
```

Is the RBF kernel also a dot product of features? Yes, with infinitely many. Expand the squared distance as $\lVert x\rVert^2 + \lVert x'\rVert^2 - 2x^{\top}x'$ and the kernel becomes $e^{-\gamma\lVert x\rVert^2}\, e^{-\gamma\lVert x'\rVert^2}\, e^{2\gamma x^{\top}x'}$. The last factor expands as $\sum_p (2\gamma)^p (x^{\top}x')^p / p!$, a weighted sum of polynomial kernels of every degree. That is the endless feature list no hand-built column set could match.

> [!KEY]
> A kernel is a similarity score that equals a dot product of features you never build. You get the curves of the extra columns, whether 45 of them or infinitely many, at the price of one dot product per pair of points.

## Gram matrices in numpy

A *Gram matrix* holds the kernel score for every pair of points in one set. For $n$ points it is an $n \times n$ table, symmetric because $k(x, x') = k(x', x)$, with ones on the diagonal for the RBF kernel. Comparing two sets of $n_1$ and $n_2$ points gives an $n_1 \times n_2$ table. A double loop would run $n_1 n_2$ slow Python iterations, but the squared distance splits into two pieces that depend on one point each and one dot product:

$$
\lVert x - x'\rVert^2 = \lVert x\rVert^2 + \lVert x'\rVert^2 - 2\,x^{\top}x' .
$$

In numpy the first two pieces become a column and a row that broadcast into a table, and `X1 @ X2.T` is the third piece for every pair at once:

```python
sq = (X1**2).sum(1)[:, None] + (X2**2).sum(1)[None, :] - 2 * X1 @ X2.T
K = np.exp(-gamma * np.maximum(sq, 0))   # shape (n1, n2)
```

The `np.maximum` is there for a reason. For two nearly identical points the three terms nearly cancel, and rounding can leave a squared distance of $-10^{-16}$. A negative distance would give a score slightly above 1, so you clip it at zero.

## When is a similarity a valid kernel?

Back to your colleague's table, $K = \begin{bmatrix}1 & 2\\ 2 & 1\end{bmatrix}$. If $k(x, x') = \varphi(x)^{\top}\varphi(x')$ for real feature lists, each point has a position in feature space, and the squared distance between two positions is real. Expanding the dot product gives

$$
\lVert \varphi(x) - \varphi(x')\rVert^2 = k(x, x) + k(x', x') - 2\,k(x, x') .
$$

Read it aloud: "the squared distance is each point's score with itself, minus twice their score with each other." Check it with real features $(1, 0)$ and $(0, 1)$. Their Gram matrix is the identity, and $1 + 1 - 2\cdot 0 = 2$. Your colleague's table gives $1 + 1 - 2\cdot 2 = -2$. A squared length of $-2$ is impossible, because the square of a real number is never negative, so no real feature lists produce this table (Figure 5).

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Left: the arrow from x2 to x1 for K equal to the identity has squared length 2. Right: the invented matrix would need squared length minus 2.">
<defs><marker id="c016-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<line x1="70" y1="170" x2="200" y2="170" class="stroke-grid"/><line x1="70" y1="170" x2="70" y2="40" class="stroke-grid"/><line x1="70" y1="170" x2="160" y2="170" class="stroke-a" stroke-width="2.2" marker-end="url(#c016-arrow)"/><line x1="70" y1="170" x2="70" y2="80" class="stroke-a" stroke-width="2.2" marker-end="url(#c016-arrow)"/><line x1="70" y1="84" x2="157" y2="167" class="stroke-b dashed" stroke-width="2.2" marker-end="url(#c016-arrow)"/><text x="156.0" y="188.0" text-anchor="start" class="fig-bold">x1</text><text x="42.0" y="84.0" text-anchor="start" class="fig-bold">x2</text><text x="132.0" y="119.0" text-anchor="start" class="fig-small">x1 − x2</text><text x="40.0" y="24.0" text-anchor="start" class="fig-bold">Real features, K = I</text><text x="40.0" y="214.0" text-anchor="start" class="fig-mono">1 + 1 − 2·0 = 2</text><text x="40.0" y="234.0" text-anchor="start" class="fig-small">squared length 2: fine</text><text x="330.0" y="24.0" text-anchor="start" class="fig-bold">Invented K = [[1, 2], [2, 1]]</text><text x="330.0" y="70.0" text-anchor="start" class="fig-mono">k(x1,x1) = 1</text><text x="330.0" y="92.0" text-anchor="start" class="fig-mono">k(x2,x2) = 1</text><text x="330.0" y="114.0" text-anchor="start" class="fig-mono">k(x1,x2) = 2</text><text x="330.0" y="150.0" text-anchor="start" class="fig-mono fig-bold">1 + 1 − 2·2 = −2</text><rect x="322" y="162" width="250" height="34" rx="4" class="fill-soft-d"/><text x="330.0" y="184.0" text-anchor="start" class="fig-small">squared length −2: no such arrow</text><line x1="322" y1="134" x2="560" y2="134" class="stroke-bad"/>
</svg>
Figure 5. Left: real features give an arrow between the two points, and its squared length is a real, non-negative number. Right: the invented table would need a squared length of minus 2. Notice that the arithmetic runs without complaint and still describes a picture that cannot be drawn.
```

The arrow between two points is one case of a general idea. Pick any weights $v_1, \ldots, v_n$ and form the combination $\sum_i v_i\,\varphi(x_i)$. The weights $(1, -1)$ give the arrow above. Stack the feature lists as the rows of $\Phi$, so that $K = \Phi\Phi^{\top}$. The combination is $\Phi^{\top}v$, and its squared length is

$$
v^{\top}Kv = v^{\top}\Phi\Phi^{\top}v = \lVert \Phi^{\top}v\rVert^2 \ge 0 .
$$

Read it aloud: "the weighted sum of all pairwise scores, with weights $v$ on both sides, is the squared length of a real vector." So every Gram matrix of a real feature map passes. A symmetric matrix with $v^{\top}Kv \ge 0$ for every $v$ is *positive semidefinite*. For your colleague's table with $v = (1, -1)$, $v^{\top}Kv = 1 + 1 - 2\cdot 2 = -2$. One such vector, the *witness*, is enough to reject the table.

You cannot try every $v$. A symmetric matrix has special directions, its eigenvectors, and along each one $v^{\top}Kv$ equals the eigenvalue times $\lVert v\rVert^2$. Every other $v$ is a mix of those directions, so $v^{\top}Kv$ cannot go negative unless one eigenvalue is negative. That turns "for every $v$" into one check: the smallest eigenvalue is at least 0. For the table above, $\lambda^2 - 2\lambda - 3 = (\lambda - 3)(\lambda + 1) = 0$ gives eigenvalues 3 and $-1$, and the eigenvector for $-1$ points along $(1, -1)$, the witness. The converse also holds, and it is the content of Mercer's condition: a similarity whose Gram matrices are positive semidefinite for every finite set of points is a kernel. This course uses the result and skips its proof.

### Check one valid kernel and one matrix that fails

The points $(1, 0)$ and $(1, 1)$, with the linear kernel, have Gram matrix

$$
\begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix}.
$$

Its characteristic polynomial is $(1 - \lambda)(2 - \lambda) - 1 = \lambda^2 - 3\lambda + 1$. The roots are $(3 - \sqrt{5})/2 \approx 0.382$ and $(3 + \sqrt{5})/2 \approx 2.618$. Both are positive, so this kernel passes the check on these two points.

The colleague's matrix has eigenvalues $3$ and $-1$. The negative one fails the check. A positive largest eigenvalue does not make a matrix valid.

> [!KEY]
> A similarity is a valid kernel when it behaves like dot products of real features: every combination $v$ of points has $v^{\top}Kv \ge 0$, which holds exactly when the Gram matrix has no negative eigenvalue. Your colleague's table fails because the combination $(1, -1)$ has squared length $-2$.

## Which kernels are valid?

The RBF kernel with $\gamma > 0$ is valid for every set of points. Above, it was one scale factor per point times a weighted sum of polynomial kernels with positive weights, a dot product of real features at every step. The polynomial kernel $(x^{\top}x' + c)^p$ is valid when $c \ge 0$ and $p$ is a whole number of at least 1.

The condition on $c$ is real. At degree 2, $(x^{\top}x' + c)^2 = (x^{\top}x')^2 + 2c\,x^{\top}x' + c^2$, and the middle term is a dot product of the features $\sqrt{2c}\,x$ and $\sqrt{2c}\,x'$. For a negative $c$ that square root is not real. That shows the recipe fails, not that the matrix must fail, so check it numerically on 8 random points in 3-D: with $c = 1$ the smallest eigenvalue is about 0.06, and with $c = -2$ it is about $-36$, so the matrix is not a valid kernel matrix.

The case $c = 0$ shows why a test needs a tolerance. A degree-2 kernel in 3-D has only 6 features, so 8 points cannot all be independent, and two eigenvalues are zero. Rounding leaves the smallest at about $-10^{-15}$. A zero eigenvalue is allowed, so the test compares against $-\text{tol}$, not 0. Duplicated points also give a zero eigenvalue, so demanding a strictly positive one is the common mistake.

> [!WARNING]
> Use $c \ge 0$ and a whole-number degree $p \ge 1$. A solver that receives an invalid table can fail with an error or, worse, return a number that means nothing. Adding a small ridge $\lambda = 0.5$ to the colleague's table leaves eigenvalues $3.5$ and $-0.5$, and a Cholesky factorisation fails because it needs every pivot to be positive.

Three conditions define a usable Gram matrix: it is square, it agrees with its transpose within a tolerance `tol`, and its smallest eigenvalue is at least `-tol`. Use `np.linalg.eigvalsh`, which is written for symmetric matrices and returns real eigenvalues in ascending order. The general `np.linalg.eig` can return complex numbers for a matrix with a small asymmetry. Measure the asymmetry with an absolute tolerance, so that `tol` means what its name says:

```python
symmetric = np.allclose(K, K.T, atol=tol, rtol=0.0)
evals = np.linalg.eigvalsh(K)      # real, ascending
ok = symmetric and evals[0] >= -tol
```

## Centre a Gram matrix

[Principal components](/lessons/027-principal-components) computes scores from the centred points. With only a Gram matrix, centre the table instead. Let $H$ be the identity with $1/n$ subtracted from every entry, let $r_i$ be the mean of row $i$ of a symmetric $K$, and let $\bar K$ be the mean of every entry.

$$
(HKH)_{ij} = K_{ij} - r_i - r_j + \bar K.
$$

Read it aloud: "subtract the mean of row $i$ and the mean of row $j$, then add back the mean of the whole table." On a linear kernel, $HKH$ equals the Gram matrix of the centred points. For a unit eigenvector $v$ with eigenvalue $\lambda$, the kernel PCA scores are $\sqrt{\lambda}\, v$.

### Work the centring by hand

The points $(1, 0)$, $(0, 1)$, and $(1, 1)$ have linear-kernel matrix

$$
K = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}.
$$

The row means are $2/3$, $2/3$, and $4/3$, and $\bar K = 8/9$. The top-left entry is $1 - 2/3 - 2/3 + 8/9 = 5/9$, and

$$
HKH = \begin{bmatrix}
5/9 & -4/9 & -1/9 \\
-4/9 & 5/9 & -1/9 \\
-1/9 & -1/9 & 2/9
\end{bmatrix}.
$$

The centred points $(1/3, -2/3)$, $(-2/3, 1/3)$, and $(1/3, 1/3)$ have these dot products. The eigenvalues are $1$, $1/3$, and $0$. The scores for eigenvalue $1$ are $-\sqrt{2}/2$, $\sqrt{2}/2$, and $0$: each centred point dotted with $(-1, 1)/\sqrt{2}$. The first dot product is $(1/3)(-1/\sqrt{2}) + (-2/3)(1/\sqrt{2}) = -1/\sqrt{2}$. On a linear kernel, kernel PCA matches those PCA scores. Subtracting only the row mean leaves the top-left entry at $1 - 2/3 = 1/3$, not $5/9$.

```python
row = K.mean(1)[:, None]
col = K.mean(0)[None, :]
Kc = K - row - col + K.mean()
```

> [!RECAP]
> - A polynomial kernel equals a dot product of explicit features (the worked pair gave 36 both ways), and the RBF kernel falls toward 0 as $\gamma$ grows.
> - A penalised minimizer lies in the span of the training features. The worked split has $J = 26$ for $(2, 3, 0)$ and $J = 42$ for $(2, 3, 4)$.
> - Build an `(n1, n2)` kernel matrix from row norms and one product, and clip squared distances at zero.
> - A valid kernel has no negative eigenvalue: $(3 \pm \sqrt{5})/2$ passes, and $3$ and $-1$ fail. A zero eigenvalue is allowed, so test against `-tol`.
> - `centre_gram` returns $HKH$. Row-mean-only centring leaves the worked top-left entry at $1/3$ instead of $5/9$.

## Build it step by step

You build seven functions, in the order the checks test them.

1. **`rbf_kernel(X1, X2, gamma)`**. Takes `(n1, d)` and `(n2, d)` points and returns `(n1, n2)`. Compute squared row norms as a column and a row, subtract twice `X1 @ X2.T`, clip at zero, and exponentiate. The usual mistake is adding two flat vectors, which gives a length-$n$ result and no error. Use `[:, None]` and `[None, :]`.
2. **`polynomial_kernel(X1, X2, degree, coef0)`**. Returns `(n1, n2)`. One product, one addition, one power: `(X1 @ X2.T + coef0) ** degree`. The usual mistake is raising `X1` to a power before the product.
3. **`poly_features_degree2(X)`**. Takes `(n, 2)` points and returns `(n, 6)` with the columns of $\varphi$ in the order shown above. Slice the two columns of `X` once and use `np.stack`. The usual mistake is dropping a $\sqrt{2}$, which breaks the match with `polynomial_kernel` at degree 2 and `coef0` 1.
4. **`min_eigenvalue(K)`**. Returns the smallest eigenvalue of a symmetric `(n, n)` matrix as a Python `float`: `np.linalg.eigvalsh(K)[0]`. The usual mistake is `np.linalg.eig`, whose eigenvalues are not sorted and may be complex.
5. **`combination_squared_length(K, v)`**. Returns `v @ K @ v` as a `float`. The usual mistake is `(v * K).sum()`, which spreads `v` across the columns only.
6. **`is_psd(K, tol)`**. Returns a Python `bool`. Return `False` for a non-square matrix, then `False` when `K` and `K.T` differ by more than `tol` in any entry, then compare `min_eigenvalue(K)` with `-tol`.
7. **`centre_gram(K)`**. Takes a symmetric `(n, n)` Gram matrix and returns `(n, n)`. Subtract the row means and the column means, then add the mean of every entry. The usual mistake is subtracting the row mean only, which leaves the three-point top-left entry at $1/3$ instead of $5/9$.

## Where this breaks

**Unscaled features.** The RBF kernel uses distances, so a feature measured in thousands swamps one measured in fractions. Standardise every column before you build a kernel matrix.

**A badly chosen $\gamma$.** A very large $\gamma$ makes every point a stranger to every other, so the Gram matrix drifts toward the identity. A very small $\gamma$ makes every point a twin of every other, so it drifts toward all ones. You choose $\gamma$ by holding data back.

**Large entries and loose tolerances.** Eigenvalue rounding error grows with the size of the entries, so a valid polynomial Gram matrix with entries near $10^6$ can show an eigenvalue of $-10^{-9}$. Scale the features first. A `tol` of 1 would accept the colleague's table, whose smallest eigenvalue is $-1$, so keep `tol` near $10^{-8}$ for entries of order 1.

## The exercise

You build `rbf_kernel`, `polynomial_kernel`, `poly_features_degree2`, `min_eigenvalue`, `combination_squared_length`, `is_psd`, and `centre_gram`. The checks compare both kernels with hand-built numbers, confirm that the RBF Gram matrix is symmetric with a unit diagonal and never exceeds 1, and confirm that a larger `gamma` lowers the off-diagonal similarity. The kernel-trick check requires `poly_features_degree2(X1) @ poly_features_degree2(X2).T` to equal `polynomial_kernel(X1, X2, 2, 1.0)` to within $10^{-10}$. Then the checks confirm that $v = (1, -1)$ gives $-2$ on the colleague's table, and that `is_psd` accepts RBF and polynomial matrices, including one with a zero eigenvalue, and rejects the colleague's table, an asymmetric matrix, a non-square matrix, and a polynomial matrix with `coef0` of $-2$. The witness check confirms that the eigenvector of a negative eigenvalue has negative squared length. The `centre_gram` check confirms the centred three-point matrix, with top-left entry $5/9$, and that kernel PCA on a centred linear kernel matches the PCA scores of the same points.

## Further reading

For kernels, the kernel trick, and Mercer's theorem, see [Lecture 8 - Kernels](https://see.stanford.edu/Course/CS229/39) in CS229 (Andrew Ng). Theorem (Mercer) in section 5.4 of the [CS229 lecture notes](https://cs229.stanford.edu/main_notes.pdf) (Tengyu Ma and Andrew Ng) states the same test: every finite Gram matrix is symmetric and positive semidefinite.
