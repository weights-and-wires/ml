---
title: Fit a Gaussian cloud to each class
slug: 027-gaussian-discriminant-analysis
part: "Part 2 — Classical models"
summary: Describe each class by its average point and one shared shape, then score unseen points by distance measured in that shape.
minutes: 40
gpu: false
objectives:
  - Explain a covariance matrix as the shape of a cloud, and a Mahalanobis distance as distance measured in that shape.
  - Fit class means, one shared covariance, and priors by averaging, and say why the covariance divides by n.
  - Compute a Gaussian log density with a Cholesky factor and combine class scores with Bayes' rule.
prereqs:
  - 026-bayes-rule-classifier
exercise: 027-gaussian-discriminant-analysis
---

# Fit a Gaussian cloud to each class

Each parcel is measured two ways, so each one is a point on a page. The parcels from warehouse 0 make a cloud of points, and the parcels from warehouse 1 make another. Both clouds are long and thin, and both lean the same way, because heavier parcels tend to be larger. A parcel arrives that sits between the clouds. Which cloud is it closer to, when "closer" must respect that the clouds are long in one direction and narrow in another?

The previous chapter, *Classify by asking who made this*, showed how to turn "how typical is this point for each class" into a probability. This chapter supplies the missing step: how to fit those bells from labelled data, in many dimensions at once. The method is called *Gaussian discriminant analysis*, or GDA. 

Measuring closeness with a ruler goes wrong here, and you will see exactly how. The fix is to measure distance in units of the cloud's own shape.

## What you already know

- The mean of some numbers is their sum divided by how many there are, so the mean of 1 and 3 is 2.
- A variance is the average squared gap from the mean, and it measures how wide a bell is.
- Bayes' rule scores each class by prior times density, and `log_sum_exp` turns the log scores into probabilities.
- A log density is the log of a bell's height at a point.
- A column times a row is a matrix, and adding up such products over many rows is what `diff.T @ diff` does.

## A cloud has a centre and a shape

To describe one class, you need two things. The first is where its points are centred. The second is how they spread: how wide the cloud is along each direction, and which way it leans.

The centre is the *mean*: add the points and divide by how many there are. The spread is the *covariance matrix*, written $\Sigma$ (the capital Greek letter sigma). On the diagonal it holds the variance of each feature, which says how wide the cloud is along each axis. Off the diagonal it holds how much two features move together, which says how much the cloud leans. A cloud with a positive off-diagonal entry leans up and to the right. A covariance matrix is therefore a shape, and every such shape draws as an ellipse.

```figure
<svg viewBox="0 0 620 370" role="img" aria-label="Scatter of two classes with fitted means and one shared covariance ellipse per class, plus a dashed oversized ellipse made by using the overall mean">
<polygon points="515.5,119.0 515.0,114.8 513.6,111.0 511.4,107.6 508.2,104.5 504.1,101.7 499.1,99.3 493.3,97.3 486.7,95.8 479.2,94.6 471.0,93.8 462.1,93.4 452.4,93.5 442.1,93.9 431.2,94.8 419.6,96.0 407.6,97.7 395.0,99.7 382.1,102.2 368.7,105.0 355.0,108.2 341.1,111.7 326.9,115.6 312.6,119.8 298.1,124.3 283.6,129.0 269.2,134.1 254.8,139.4 240.5,144.9 226.5,150.6 212.6,156.5 199.1,162.6 185.9,168.8 173.2,175.1 160.9,181.5 149.1,187.9 137.8,194.4 127.2,200.8 117.2,207.3 107.9,213.7 99.3,220.0 91.5,226.3 84.4,232.4 78.2,238.3 72.8,244.1 68.3,249.7 64.7,255.1 61.9,260.2 60.1,265.1 59.2,269.6 59.2,273.9 60.1,277.9 61.9,281.5 64.7,284.8 68.3,287.8 72.8,290.3 78.2,292.5 84.4,294.3 91.5,295.7 99.3,296.7 107.9,297.2 117.2,297.4 127.2,297.2 137.8,296.5 149.1,295.5 160.9,294.0 173.2,292.1 185.9,289.9 199.1,287.3 212.6,284.3 226.5,280.9 240.5,277.2 254.8,273.2 269.2,268.8 283.6,264.2 298.1,259.3 312.6,254.1 326.9,248.7 341.1,243.1 355.0,237.2 368.7,231.3 382.1,225.1 395.0,218.9 407.6,212.5 419.6,206.1 431.2,199.7 442.1,193.2 452.4,186.7 462.1,180.3 471.0,173.9 479.2,167.6 486.7,161.5 493.3,155.4 499.1,149.6 504.1,143.9 508.2,138.4 511.4,133.1 513.6,128.1 515.0,123.4 515.5,119.0" fill="none" class="stroke-muted dashed"/>
<polygon points="291.4,153.5 291.2,150.0 290.6,146.8 289.5,143.9 288.0,141.3 286.2,139.1 283.9,137.1 281.2,135.5 278.1,134.3 274.7,133.4 270.9,132.8 266.8,132.6 262.3,132.8 257.6,133.3 252.5,134.1 247.2,135.4 241.6,136.9 235.8,138.8 229.8,141.0 223.7,143.6 217.3,146.4 210.9,149.6 204.4,153.0 197.7,156.8 191.1,160.7 184.4,165.0 177.7,169.4 171.1,174.1 164.5,178.9 158.0,183.9 151.6,189.1 145.3,194.4 139.3,199.8 133.4,205.2 127.7,210.8 122.2,216.4 117.1,222.0 112.1,227.6 107.5,233.1 103.2,238.7 99.3,244.1 95.7,249.5 92.4,254.7 89.5,259.8 87.0,264.8 85.0,269.6 83.3,274.1 82.0,278.5 81.2,282.6 80.7,286.5 80.7,290.1 81.2,293.5 82.0,296.5 83.3,299.3 85.0,301.7 87.0,303.8 89.5,305.6 92.4,307.0 95.7,308.1 99.3,308.8 103.2,309.2 107.5,309.2 112.1,308.9 117.1,308.2 122.2,307.2 127.7,305.8 133.4,304.0 139.3,302.0 145.3,299.6 151.6,296.9 158.0,293.9 164.5,290.6 171.1,287.0 177.7,283.1 184.4,279.0 191.1,274.7 197.7,270.1 204.4,265.4 210.9,260.5 217.3,255.4 223.7,250.2 229.8,244.8 235.8,239.4 241.6,233.9 247.2,228.3 252.5,222.7 257.6,217.1 262.3,211.5 266.8,205.9 270.9,200.5 274.7,195.0 278.1,189.7 281.2,184.6 283.9,179.5 286.2,174.7 288.0,170.0 289.5,165.5 290.6,161.3 291.2,157.3 291.4,153.5" fill="none" class="stroke-a"/>
<polygon points="493.9,102.4 493.6,99.0 493.0,95.8 491.9,92.9 490.5,90.3 488.6,88.0 486.3,86.1 483.6,84.5 480.6,83.2 477.1,82.3 473.3,81.7 469.2,81.5 464.7,81.7 460.0,82.2 454.9,83.1 449.6,84.3 444.0,85.8 438.2,87.7 432.3,90.0 426.1,92.5 419.8,95.4 413.3,98.5 406.8,102.0 400.2,105.7 393.5,109.7 386.8,113.9 380.1,118.3 373.5,123.0 366.9,127.8 360.4,132.9 354.0,138.0 347.8,143.3 341.7,148.7 335.8,154.2 330.1,159.7 324.7,165.3 319.5,170.9 314.6,176.5 310.0,182.1 305.7,187.6 301.7,193.1 298.1,198.4 294.8,203.7 292.0,208.8 289.5,213.7 287.4,218.5 285.7,223.1 284.4,227.4 283.6,231.6 283.2,235.5 283.2,239.1 283.6,242.4 284.4,245.5 285.7,248.2 287.4,250.6 289.5,252.7 292.0,254.5 294.8,255.9 298.1,257.0 301.7,257.8 305.7,258.1 310.0,258.2 314.6,257.8 319.5,257.1 324.7,256.1 330.1,254.7 335.8,253.0 341.7,250.9 347.8,248.5 354.0,245.8 360.4,242.8 366.9,239.5 373.5,235.9 380.1,232.1 386.8,228.0 393.5,223.6 400.2,219.1 406.8,214.3 413.3,209.4 419.8,204.3 426.1,199.1 432.3,193.7 438.2,188.3 444.0,182.8 449.6,177.2 454.9,171.6 460.0,166.0 464.7,160.4 469.2,154.9 473.3,149.4 477.1,144.0 480.6,138.7 483.6,133.5 486.3,128.5 488.6,123.6 490.5,118.9 491.9,114.5 493.0,110.2 493.6,106.2 493.9,102.4" fill="none" class="stroke-b"/>
<circle cx="242.0" cy="217.0" r="4" class="fill-a"/>
<circle cx="179.3" cy="206.5" r="4" class="fill-a"/>
<circle cx="95.7" cy="278.9" r="4" class="fill-a"/>
<circle cx="217.6" cy="217.5" r="4" class="fill-a"/>
<circle cx="96.5" cy="228.3" r="4" class="fill-a"/>
<circle cx="171.9" cy="266.2" r="4" class="fill-a"/>
<circle cx="206.1" cy="147.9" r="4" class="fill-a"/>
<circle cx="184.2" cy="273.4" r="4" class="fill-a"/>
<circle cx="193.6" cy="247.8" r="4" class="fill-a"/>
<circle cx="217.9" cy="207.6" r="4" class="fill-a"/>
<circle cx="206.9" cy="181.6" r="4" class="fill-a"/>
<circle cx="183.3" cy="236.6" r="4" class="fill-a"/>
<circle cx="130.8" cy="229.2" r="4" class="fill-a"/>
<circle cx="279.0" cy="154.5" r="4" class="fill-a"/>
<circle cx="422.6" cy="132.2" r="4" class="fill-b"/>
<circle cx="439.2" cy="113.5" r="4" class="fill-b"/>
<circle cx="364.4" cy="179.8" r="4" class="fill-b"/>
<circle cx="430.4" cy="133.8" r="4" class="fill-b"/>
<circle cx="448.2" cy="151.9" r="4" class="fill-b"/>
<circle cx="360.3" cy="209.3" r="4" class="fill-b"/>
<circle cx="271.2" cy="215.7" r="4" class="fill-b"/>
<circle cx="480.5" cy="74.5" r="4" class="fill-b"/>
<circle cx="427.1" cy="122.1" r="4" class="fill-b"/>
<circle cx="385.6" cy="219.2" r="4" class="fill-b"/>
<circle cx="376.1" cy="169.5" r="4" class="fill-b"/>
<circle cx="314.3" cy="245.9" r="4" class="fill-b"/>
<circle cx="327.3" cy="233.5" r="4" class="fill-b"/>
<circle cx="391.6" cy="177.2" r="4" class="fill-b"/>
<path d="M179.1,220.9 L193.1,220.9 M186.1,213.9 L186.1,227.9" class="stroke-ink" stroke-width="3"/>
<path d="M381.5,169.9 L395.5,169.9 M388.5,162.9 L388.5,176.9" class="stroke-ink" stroke-width="3"/>
<circle cx="287.3" cy="195.4" r="5" fill="none" class="stroke-muted"/>
<path d="M300.0,271.2 L307.0,285.2 L293.0,285.2 z" class="fill-ink"/>
<text x="312.0" y="284.2" class="fig-bold">query point</text>
<text x="62.0" y="165.0" class="fig-bold">class 0</text>
<text x="410.0" y="72.0" class="fig-bold">class 1</text>
<text x="170.0" y="326.0" class="fig-small">overall mean (grey)</text>
<text x="170.0" y="341.0" class="fig-small">and the oversized ellipse it gives</text>
</svg>
Figure 1. Crosses mark the class means, and each solid ellipse holds the points within distance 2 of its mean, measured in units of the shared shape. One shared shape, fitted from the gaps around each class's own mean, wraps both clouds. The dashed grey ellipse comes from subtracting the overall mean instead: it also counts the gap between the two centres as spread, so it is far too large. The triangle is the point the chapter classifies.
```

Figure 1 shows two classes of points, with a cross at each class mean and an ellipse around each. Both ellipses are the same shape, because the model gives all classes one shared covariance. Sharing is a choice, and it has a reason. A covariance of $d$ features has about $d^2$ numbers, and a class with few points cannot pin them all down, so pooling every class's points into one estimate steadies it. The price is that the model cannot describe a round class and a long one. The next chapter shows what sharing buys.

## Why a ruler fails

Back to the parcel between the clouds. It is the triangle in Figure 1. Its straight-line distance to the class 0 mean is 2.46, and to the class 1 mean it is 2.70, so a ruler says class 0. But the triangle's gap from the class 0 mean runs across the clouds' lean, the short way, where a cloud almost never reaches. Its gap from the class 1 mean runs mostly along the lean, where points are common.

```figure
<svg viewBox="0 0 620 330" role="img" aria-label="A tilted ellipse with two points at the same straight-line distance from its centre but different Mahalanobis distances">
<marker id="c029-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker>
<polygon points="326.2,114.0 326.1,112.4 325.7,110.9 325.0,109.6 324.1,108.5 322.9,107.5 321.5,106.7 319.8,106.1 317.8,105.6 315.6,105.2 313.1,105.1 310.5,105.1 307.6,105.3 304.5,105.6 301.1,106.1 297.6,106.8 293.9,107.6 290.0,108.6 286.0,109.8 281.7,111.1 277.4,112.5 272.9,114.1 268.3,115.9 263.6,117.7 258.8,119.8 253.9,121.9 248.9,124.1 243.9,126.5 238.9,129.0 233.8,131.6 228.7,134.2 223.7,137.0 218.6,139.8 213.6,142.7 208.6,145.7 203.7,148.7 198.8,151.8 194.0,154.9 189.4,158.0 184.8,161.2 180.4,164.4 176.1,167.5 172.0,170.7 168.0,173.8 164.2,176.9 160.6,180.0 157.2,183.1 154.0,186.0 151.0,189.0 148.2,191.8 145.6,194.6 143.3,197.3 141.2,200.0 139.4,202.5 137.8,204.9 136.5,207.2 135.4,209.4 134.6,211.5 134.1,213.4 133.8,215.2 133.8,216.8 134.1,218.4 134.6,219.7 135.4,220.9 136.5,222.0 137.8,222.9 139.4,223.6 141.2,224.2 143.3,224.6 145.6,224.9 148.2,224.9 151.0,224.8 154.0,224.6 157.2,224.2 160.6,223.6 164.2,222.8 168.0,221.9 172.0,220.8 176.1,219.6 180.4,218.2 184.8,216.7 189.4,215.0 194.0,213.2 198.8,211.3 203.7,209.2 208.6,207.0 213.6,204.7 218.6,202.3 223.7,199.7 228.7,197.1 233.8,194.4 238.9,191.6 243.9,188.7 248.9,185.8 253.9,182.8 258.8,179.7 263.6,176.6 268.3,173.5 272.9,170.4 277.4,167.2 281.7,164.1 286.0,160.9 290.0,157.8 293.9,154.6 297.6,151.5 301.1,148.5 304.5,145.4 307.6,142.5 310.5,139.6 313.1,136.8 315.6,134.0 317.8,131.3 319.8,128.8 321.5,126.3 322.9,123.9 324.1,121.7 325.0,119.6 325.7,117.6 326.1,115.7 326.2,114.0" fill="none" class="stroke-b"/>
<polygon points="422.5,62.9 422.2,59.8 421.4,56.9 420.1,54.3 418.2,52.0 415.8,50.1 412.9,48.4 409.5,47.1 405.6,46.1 401.2,45.5 396.3,45.2 390.9,45.2 385.1,45.5 378.9,46.2 372.3,47.2 365.2,48.6 357.8,50.2 350.0,52.2 341.9,54.5 333.5,57.1 324.8,60.0 315.8,63.2 306.6,66.7 297.2,70.5 287.5,74.5 277.8,78.8 267.9,83.3 257.9,88.0 247.8,93.0 237.6,98.2 227.5,103.5 217.3,109.0 207.2,114.7 197.1,120.5 187.2,126.4 177.3,132.5 167.6,138.6 158.1,144.8 148.8,151.1 139.7,157.4 130.8,163.7 122.3,170.0 114.0,176.3 106.0,182.6 98.4,188.9 91.2,195.0 84.4,201.1 77.9,207.1 71.9,213.0 66.3,218.7 61.2,224.3 56.6,229.7 52.4,234.9 48.7,240.0 45.6,244.8 42.9,249.4 40.8,253.8 39.2,257.9 38.1,261.8 37.6,265.4 37.6,268.7 38.1,271.7 39.2,274.5 40.8,276.9 42.9,279.0 45.6,280.8 48.7,282.3 52.4,283.4 56.6,284.2 61.2,284.7 66.3,284.9 71.9,284.7 77.9,284.2 84.4,283.3 91.2,282.1 98.4,280.6 106.0,278.8 114.0,276.7 122.3,274.2 130.8,271.5 139.7,268.4 148.8,265.1 158.1,261.4 167.6,257.5 177.3,253.4 187.2,249.0 197.1,244.4 207.2,239.5 217.3,234.5 227.5,229.2 237.6,223.8 247.8,218.2 257.9,212.4 267.9,206.6 277.8,200.6 287.5,194.5 297.2,188.3 306.6,182.1 315.8,175.8 324.8,169.5 333.5,163.1 341.9,156.8 350.0,150.5 357.8,144.2 365.2,138.0 372.3,131.9 378.9,125.9 385.1,120.0 390.9,114.2 396.3,108.5 401.2,103.0 405.6,97.7 409.5,92.5 412.9,87.6 415.8,82.9 418.2,78.4 420.1,74.1 421.4,70.1 422.2,66.4 422.5,62.9" fill="none" class="stroke-b"/>
<circle cx="230" cy="165" r="110" fill="none" class="stroke-muted dashed"/>
<line x1="230" y1="165" x2="325.3" y2="110.0" class="stroke-ink" marker-end="url(#c029-arrow)"/>
<circle cx="325.3" cy="110.0" r="5" class="fill-a"/>
<text x="337.3" y="102.0" class="fig-bold">A</text>
<line x1="230" y1="165" x2="175.0" y2="69.7" class="stroke-ink" marker-end="url(#c029-arrow)"/>
<circle cx="175.0" cy="69.7" r="5" class="fill-a"/>
<text x="153.0" y="61.7" class="fig-bold">B</text>
<circle cx="230" cy="165" r="5" class="fill-ink"/>
<text x="470" y="60" class="fig-bold">Point A</text>
<text x="470" y="78" class="fig-small">straight-line distance 2.0</text>
<text x="470" y="94" class="fig-small">shape distance 1.0</text>
<text x="470" y="130" class="fig-bold">Point B</text>
<text x="470" y="148" class="fig-small">straight-line distance 2.0</text>
<text x="470" y="164" class="fig-small">shape distance 4.0</text>
<text x="470" y="210" class="fig-small">Orange ellipses: shape</text>
<text x="470" y="226" class="fig-small">distance 1 and 2.</text>
<text x="470" y="242" class="fig-small">Grey circle: straight-line</text>
<text x="470" y="258" class="fig-small">distance 2.</text>
</svg>
Figure 2. Points A and B are the same straight-line distance from the centre, but A lies along the cloud's long axis and B across its short one. Measured in units of the cloud's own shape, A is typical and B is far out.
```

Figure 2 isolates the effect. Points A and B are the same straight-line distance, 2.0, from the centre. A sits along the cloud's long axis, where the cloud is wide, so it is an ordinary point. B sits across the short axis, so it is far outside. Measured in units of the cloud's own shape, A is 1.0 away and B is 4.0 away. That is the *Mahalanobis distance*: how many shape units a point is from the centre. In one dimension it is the familiar count of standard deviations, $|x - \mu| / \sigma$. In many dimensions the shape matrix does the dividing:

$$
d_M^2 = (x - \mu)^{\top} \Sigma^{-1} (x - \mu).
$$

Read it aloud: take the gap vector from the mean, scale it by the inverse shape so that wide directions count less and narrow ones count more, and take the squared length. For the triangle, the squared shape distances are 25.8 to class 0 and 6.2 to class 1, so by this measure it belongs to class 1.

## Fit the cloud by averaging

Fitting means choosing the bell that makes the training points most likely, called *maximum likelihood*. It sounds heavy, but for a Gaussian it works out to averaging.

Take four points on a line, with class 0 at 1 and 3, and class 1 at 5 and 7.

- **Priors.** Two of the four points are in each class, so both priors are $2/4 = 0.5$.
- **Means.** Class 0's mean is $(1 + 3)/2 = 2$ and class 1's is $(5 + 7)/2 = 6$. The mean is the best centre because it minimises the total squared gap: any other centre leaves some points farther away than the balance point.
- **Shape.** Each point's gap from its own class mean is $-1, +1, -1, +1$. The squared gaps are all 1, and their sum is 4. Divide by the 4 points to get a variance of 1.

Figure 3 draws the result.

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="Four points on a line, two fitted bells with standard deviation 1 centred at 2 and 6, and the log densities at x equal 3">
<line x1="50" y1="210" x2="570" y2="210" class="stroke-muted"/>
<text x="102.0" y="230" text-anchor="middle" class="fig-small">0</text>
<text x="154.0" y="230" text-anchor="middle" class="fig-small">1</text>
<text x="206.0" y="230" text-anchor="middle" class="fig-small">2</text>
<text x="258.0" y="230" text-anchor="middle" class="fig-small">3</text>
<text x="310.0" y="230" text-anchor="middle" class="fig-small">4</text>
<text x="362.0" y="230" text-anchor="middle" class="fig-small">5</text>
<text x="414.0" y="230" text-anchor="middle" class="fig-small">6</text>
<text x="466.0" y="230" text-anchor="middle" class="fig-small">7</text>
<text x="518.0" y="230" text-anchor="middle" class="fig-small">8</text>
<polygon points="50.0,210.0 50.0,208.3 52.6,208.1 55.2,207.8 57.8,207.4 60.4,207.0 63.0,206.6 65.6,206.1 68.2,205.5 70.8,204.9 73.4,204.2 76.0,203.4 78.6,202.5 81.2,201.5 83.8,200.5 86.4,199.3 89.0,198.0 91.6,196.6 94.2,195.1 96.8,193.4 99.4,191.6 102.0,189.6 104.6,187.5 107.2,185.2 109.8,182.8 112.4,180.2 115.0,177.4 117.6,174.5 120.2,171.4 122.8,168.1 125.4,164.7 128.0,161.1 130.6,157.3 133.2,153.4 135.8,149.4 138.4,145.3 141.0,141.0 143.6,136.6 146.2,132.2 148.8,127.7 151.4,123.2 154.0,118.6 156.6,114.0 159.2,109.5 161.8,105.0 164.4,100.6 167.0,96.2 169.6,92.0 172.2,88.0 174.8,84.1 177.4,80.4 180.0,77.0 182.6,73.8 185.2,70.9 187.8,68.2 190.4,65.9 193.0,63.9 195.6,62.3 198.2,61.0 200.8,60.0 203.4,59.5 206.0,59.3 208.6,59.5 211.2,60.0 213.8,61.0 216.4,62.3 219.0,63.9 221.6,65.9 224.2,68.2 226.8,70.9 229.4,73.8 232.0,77.0 234.6,80.4 237.2,84.1 239.8,88.0 242.4,92.0 245.0,96.2 247.6,100.6 250.2,105.0 252.8,109.5 255.4,114.0 258.0,118.6 260.6,123.2 263.2,127.7 265.8,132.2 268.4,136.6 271.0,141.0 273.6,145.3 276.2,149.4 278.8,153.4 281.4,157.3 284.0,161.1 286.6,164.7 289.2,168.1 291.8,171.4 294.4,174.5 297.0,177.4 299.6,180.2 302.2,182.8 304.8,185.2 307.4,187.5 310.0,189.6 312.6,191.6 315.2,193.4 317.8,195.1 320.4,196.6 323.0,198.0 325.6,199.3 328.2,200.5 330.8,201.5 333.4,202.5 336.0,203.4 338.6,204.2 341.2,204.9 343.8,205.5 346.4,206.1 349.0,206.6 351.6,207.0 354.2,207.4 356.8,207.8 359.4,208.1 362.0,208.3 364.6,208.6 367.2,208.8 369.8,208.9 372.4,209.1 375.0,209.2 377.6,209.3 380.2,209.4 382.8,209.5 385.4,209.6 388.0,209.7 390.6,209.7 393.2,209.8 395.8,209.8 398.4,209.8 401.0,209.9 403.6,209.9 406.2,209.9 408.8,209.9 411.4,209.9 414.0,209.9 416.6,210.0 419.2,210.0 421.8,210.0 424.4,210.0 427.0,210.0 429.6,210.0 432.2,210.0 434.8,210.0 437.4,210.0 440.0,210.0 442.6,210.0 445.2,210.0 447.8,210.0 450.4,210.0 453.0,210.0 455.6,210.0 458.2,210.0 460.8,210.0 463.4,210.0 466.0,210.0 468.6,210.0 471.2,210.0 473.8,210.0 476.4,210.0 479.0,210.0 481.6,210.0 484.2,210.0 486.8,210.0 489.4,210.0 492.0,210.0 494.6,210.0 497.2,210.0 499.8,210.0 502.4,210.0 505.0,210.0 507.6,210.0 510.2,210.0 512.8,210.0 515.4,210.0 518.0,210.0 520.6,210.0 523.2,210.0 525.8,210.0 528.4,210.0 531.0,210.0 533.6,210.0 536.2,210.0 538.8,210.0 541.4,210.0 544.0,210.0 546.6,210.0 549.2,210.0 551.8,210.0 554.4,210.0 557.0,210.0 559.6,210.0 562.2,210.0 564.8,210.0 567.4,210.0 570.0,210.0 570.0,210.0" class="fill-soft-a"/><polyline points="50.0,208.3 52.6,208.1 55.2,207.8 57.8,207.4 60.4,207.0 63.0,206.6 65.6,206.1 68.2,205.5 70.8,204.9 73.4,204.2 76.0,203.4 78.6,202.5 81.2,201.5 83.8,200.5 86.4,199.3 89.0,198.0 91.6,196.6 94.2,195.1 96.8,193.4 99.4,191.6 102.0,189.6 104.6,187.5 107.2,185.2 109.8,182.8 112.4,180.2 115.0,177.4 117.6,174.5 120.2,171.4 122.8,168.1 125.4,164.7 128.0,161.1 130.6,157.3 133.2,153.4 135.8,149.4 138.4,145.3 141.0,141.0 143.6,136.6 146.2,132.2 148.8,127.7 151.4,123.2 154.0,118.6 156.6,114.0 159.2,109.5 161.8,105.0 164.4,100.6 167.0,96.2 169.6,92.0 172.2,88.0 174.8,84.1 177.4,80.4 180.0,77.0 182.6,73.8 185.2,70.9 187.8,68.2 190.4,65.9 193.0,63.9 195.6,62.3 198.2,61.0 200.8,60.0 203.4,59.5 206.0,59.3 208.6,59.5 211.2,60.0 213.8,61.0 216.4,62.3 219.0,63.9 221.6,65.9 224.2,68.2 226.8,70.9 229.4,73.8 232.0,77.0 234.6,80.4 237.2,84.1 239.8,88.0 242.4,92.0 245.0,96.2 247.6,100.6 250.2,105.0 252.8,109.5 255.4,114.0 258.0,118.6 260.6,123.2 263.2,127.7 265.8,132.2 268.4,136.6 271.0,141.0 273.6,145.3 276.2,149.4 278.8,153.4 281.4,157.3 284.0,161.1 286.6,164.7 289.2,168.1 291.8,171.4 294.4,174.5 297.0,177.4 299.6,180.2 302.2,182.8 304.8,185.2 307.4,187.5 310.0,189.6 312.6,191.6 315.2,193.4 317.8,195.1 320.4,196.6 323.0,198.0 325.6,199.3 328.2,200.5 330.8,201.5 333.4,202.5 336.0,203.4 338.6,204.2 341.2,204.9 343.8,205.5 346.4,206.1 349.0,206.6 351.6,207.0 354.2,207.4 356.8,207.8 359.4,208.1 362.0,208.3 364.6,208.6 367.2,208.8 369.8,208.9 372.4,209.1 375.0,209.2 377.6,209.3 380.2,209.4 382.8,209.5 385.4,209.6 388.0,209.7 390.6,209.7 393.2,209.8 395.8,209.8 398.4,209.8 401.0,209.9 403.6,209.9 406.2,209.9 408.8,209.9 411.4,209.9 414.0,209.9 416.6,210.0 419.2,210.0 421.8,210.0 424.4,210.0 427.0,210.0 429.6,210.0 432.2,210.0 434.8,210.0 437.4,210.0 440.0,210.0 442.6,210.0 445.2,210.0 447.8,210.0 450.4,210.0 453.0,210.0 455.6,210.0 458.2,210.0 460.8,210.0 463.4,210.0 466.0,210.0 468.6,210.0 471.2,210.0 473.8,210.0 476.4,210.0 479.0,210.0 481.6,210.0 484.2,210.0 486.8,210.0 489.4,210.0 492.0,210.0 494.6,210.0 497.2,210.0 499.8,210.0 502.4,210.0 505.0,210.0 507.6,210.0 510.2,210.0 512.8,210.0 515.4,210.0 518.0,210.0 520.6,210.0 523.2,210.0 525.8,210.0 528.4,210.0 531.0,210.0 533.6,210.0 536.2,210.0 538.8,210.0 541.4,210.0 544.0,210.0 546.6,210.0 549.2,210.0 551.8,210.0 554.4,210.0 557.0,210.0 559.6,210.0 562.2,210.0 564.8,210.0 567.4,210.0 570.0,210.0" fill="none" class="stroke-a"/>
<polygon points="50.0,210.0 50.0,210.0 52.6,210.0 55.2,210.0 57.8,210.0 60.4,210.0 63.0,210.0 65.6,210.0 68.2,210.0 70.8,210.0 73.4,210.0 76.0,210.0 78.6,210.0 81.2,210.0 83.8,210.0 86.4,210.0 89.0,210.0 91.6,210.0 94.2,210.0 96.8,210.0 99.4,210.0 102.0,210.0 104.6,210.0 107.2,210.0 109.8,210.0 112.4,210.0 115.0,210.0 117.6,210.0 120.2,210.0 122.8,210.0 125.4,210.0 128.0,210.0 130.6,210.0 133.2,210.0 135.8,210.0 138.4,210.0 141.0,210.0 143.6,210.0 146.2,210.0 148.8,210.0 151.4,210.0 154.0,210.0 156.6,210.0 159.2,210.0 161.8,210.0 164.4,210.0 167.0,210.0 169.6,210.0 172.2,210.0 174.8,210.0 177.4,210.0 180.0,210.0 182.6,210.0 185.2,210.0 187.8,210.0 190.4,210.0 193.0,210.0 195.6,210.0 198.2,210.0 200.8,210.0 203.4,210.0 206.0,209.9 208.6,209.9 211.2,209.9 213.8,209.9 216.4,209.9 219.0,209.9 221.6,209.8 224.2,209.8 226.8,209.8 229.4,209.7 232.0,209.7 234.6,209.6 237.2,209.5 239.8,209.4 242.4,209.3 245.0,209.2 247.6,209.1 250.2,208.9 252.8,208.8 255.4,208.6 258.0,208.3 260.6,208.1 263.2,207.8 265.8,207.4 268.4,207.0 271.0,206.6 273.6,206.1 276.2,205.5 278.8,204.9 281.4,204.2 284.0,203.4 286.6,202.5 289.2,201.5 291.8,200.5 294.4,199.3 297.0,198.0 299.6,196.6 302.2,195.1 304.8,193.4 307.4,191.6 310.0,189.6 312.6,187.5 315.2,185.2 317.8,182.8 320.4,180.2 323.0,177.4 325.6,174.5 328.2,171.4 330.8,168.1 333.4,164.7 336.0,161.1 338.6,157.3 341.2,153.4 343.8,149.4 346.4,145.3 349.0,141.0 351.6,136.6 354.2,132.2 356.8,127.7 359.4,123.2 362.0,118.6 364.6,114.0 367.2,109.5 369.8,105.0 372.4,100.6 375.0,96.2 377.6,92.0 380.2,88.0 382.8,84.1 385.4,80.4 388.0,77.0 390.6,73.8 393.2,70.9 395.8,68.2 398.4,65.9 401.0,63.9 403.6,62.3 406.2,61.0 408.8,60.0 411.4,59.5 414.0,59.3 416.6,59.5 419.2,60.0 421.8,61.0 424.4,62.3 427.0,63.9 429.6,65.9 432.2,68.2 434.8,70.9 437.4,73.8 440.0,77.0 442.6,80.4 445.2,84.1 447.8,88.0 450.4,92.0 453.0,96.2 455.6,100.6 458.2,105.0 460.8,109.5 463.4,114.0 466.0,118.6 468.6,123.2 471.2,127.7 473.8,132.2 476.4,136.6 479.0,141.0 481.6,145.3 484.2,149.4 486.8,153.4 489.4,157.3 492.0,161.1 494.6,164.7 497.2,168.1 499.8,171.4 502.4,174.5 505.0,177.4 507.6,180.2 510.2,182.8 512.8,185.2 515.4,187.5 518.0,189.6 520.6,191.6 523.2,193.4 525.8,195.1 528.4,196.6 531.0,198.0 533.6,199.3 536.2,200.5 538.8,201.5 541.4,202.5 544.0,203.4 546.6,204.2 549.2,204.9 551.8,205.5 554.4,206.1 557.0,206.6 559.6,207.0 562.2,207.4 564.8,207.8 567.4,208.1 570.0,208.3 570.0,210.0" class="fill-soft-b"/><polyline points="50.0,210.0 52.6,210.0 55.2,210.0 57.8,210.0 60.4,210.0 63.0,210.0 65.6,210.0 68.2,210.0 70.8,210.0 73.4,210.0 76.0,210.0 78.6,210.0 81.2,210.0 83.8,210.0 86.4,210.0 89.0,210.0 91.6,210.0 94.2,210.0 96.8,210.0 99.4,210.0 102.0,210.0 104.6,210.0 107.2,210.0 109.8,210.0 112.4,210.0 115.0,210.0 117.6,210.0 120.2,210.0 122.8,210.0 125.4,210.0 128.0,210.0 130.6,210.0 133.2,210.0 135.8,210.0 138.4,210.0 141.0,210.0 143.6,210.0 146.2,210.0 148.8,210.0 151.4,210.0 154.0,210.0 156.6,210.0 159.2,210.0 161.8,210.0 164.4,210.0 167.0,210.0 169.6,210.0 172.2,210.0 174.8,210.0 177.4,210.0 180.0,210.0 182.6,210.0 185.2,210.0 187.8,210.0 190.4,210.0 193.0,210.0 195.6,210.0 198.2,210.0 200.8,210.0 203.4,210.0 206.0,209.9 208.6,209.9 211.2,209.9 213.8,209.9 216.4,209.9 219.0,209.9 221.6,209.8 224.2,209.8 226.8,209.8 229.4,209.7 232.0,209.7 234.6,209.6 237.2,209.5 239.8,209.4 242.4,209.3 245.0,209.2 247.6,209.1 250.2,208.9 252.8,208.8 255.4,208.6 258.0,208.3 260.6,208.1 263.2,207.8 265.8,207.4 268.4,207.0 271.0,206.6 273.6,206.1 276.2,205.5 278.8,204.9 281.4,204.2 284.0,203.4 286.6,202.5 289.2,201.5 291.8,200.5 294.4,199.3 297.0,198.0 299.6,196.6 302.2,195.1 304.8,193.4 307.4,191.6 310.0,189.6 312.6,187.5 315.2,185.2 317.8,182.8 320.4,180.2 323.0,177.4 325.6,174.5 328.2,171.4 330.8,168.1 333.4,164.7 336.0,161.1 338.6,157.3 341.2,153.4 343.8,149.4 346.4,145.3 349.0,141.0 351.6,136.6 354.2,132.2 356.8,127.7 359.4,123.2 362.0,118.6 364.6,114.0 367.2,109.5 369.8,105.0 372.4,100.6 375.0,96.2 377.6,92.0 380.2,88.0 382.8,84.1 385.4,80.4 388.0,77.0 390.6,73.8 393.2,70.9 395.8,68.2 398.4,65.9 401.0,63.9 403.6,62.3 406.2,61.0 408.8,60.0 411.4,59.5 414.0,59.3 416.6,59.5 419.2,60.0 421.8,61.0 424.4,62.3 427.0,63.9 429.6,65.9 432.2,68.2 434.8,70.9 437.4,73.8 440.0,77.0 442.6,80.4 445.2,84.1 447.8,88.0 450.4,92.0 453.0,96.2 455.6,100.6 458.2,105.0 460.8,109.5 463.4,114.0 466.0,118.6 468.6,123.2 471.2,127.7 473.8,132.2 476.4,136.6 479.0,141.0 481.6,145.3 484.2,149.4 486.8,153.4 489.4,157.3 492.0,161.1 494.6,164.7 497.2,168.1 499.8,171.4 502.4,174.5 505.0,177.4 507.6,180.2 510.2,182.8 512.8,185.2 515.4,187.5 518.0,189.6 520.6,191.6 523.2,193.4 525.8,195.1 528.4,196.6 531.0,198.0 533.6,199.3 536.2,200.5 538.8,201.5 541.4,202.5 544.0,203.4 546.6,204.2 549.2,204.9 551.8,205.5 554.4,206.1 557.0,206.6 559.6,207.0 562.2,207.4 564.8,207.8 567.4,208.1 570.0,208.3" fill="none" class="stroke-b"/>
<circle cx="154.0" cy="210" r="6" class="fill-a"/>
<circle cx="258.0" cy="210" r="6" class="fill-a"/>
<circle cx="362.0" cy="210" r="6" class="fill-b"/>
<circle cx="466.0" cy="210" r="6" class="fill-b"/>
<line x1="258.0" y1="210" x2="258.0" y2="46" class="stroke-ink dashed"/>
<circle cx="258.0" cy="118.6" r="4" class="fill-ink"/><circle cx="258.0" cy="208.3" r="4" class="fill-ink"/>
<text x="268.0" y="114.6" class="fig-small">class 0</text>
<text x="310" y="252" text-anchor="middle" class="fig-small">At x = 3 (dashed line): class 0 log density -1.4189, class 1 log density -5.4189</text>
<text x="206.0" y="46.9" text-anchor="middle" class="fig-bold">mean 2</text>
<text x="414.0" y="46.9" text-anchor="middle" class="fig-bold">mean 6</text>
<text x="310" y="274" text-anchor="middle">feature x</text>
<text x="310" y="294" text-anchor="middle" class="fig-small">Dots on the axis are the four training points 1, 3, 5, 7.</text>
</svg>
Figure 3. The four points give two bells with standard deviation 1 (variance 1), centred at their class means 2 and 6. At x = 3 the class 0 log density, -1.4189, is four higher than class 1's, -5.4189: one unit from a centre costs 0.5, three units cost 4.5.
```

The variance is an average of squared gaps because a bell that is too narrow gives distant points a huge penalty, and a bell that is too wide wastes its height on empty space. The two pressures balance exactly when the variance equals the average squared gap. In symbols, with $n_k$ points in class $k$ and $n$ points in all,

$$
\pi_k = \frac{n_k}{n}, \qquad
\mu_k = \frac{1}{n_k}\sum_{i:\, y_i = k} x_i, \qquad
\Sigma = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu_{y_i})(x_i - \mu_{y_i})^{\top}.
$$

Read it aloud: the prior is the class's share of the points, the mean is the class's average point, and the shared shape is the average, over every point, of its gap from its own class mean, multiplied by itself as a column times a row. The shape sum divides by $n$, the number of rows, and not by $n - 1$. The $n - 1$ habit from statistics corrects for estimating a larger population from a sample. Here you describe the points in front of you, and the maximum likelihood answer is the plain average. The two differ by a factor $n/(n-1)$, which is $4/3$ for these four points.

> [!WARNING]
> Subtract each point's own class mean, never the overall mean. Figure 1 shows the dashed grey ellipse that results from the overall mean: it counts the gap between the class centres as spread, so the shape is far too large and the model blurs the classes together.

In numpy, one class adds `diff.T @ diff` for `diff = X[y == k] - means[k]`, and you divide the total once by `n`.

## Score a point

A Gaussian density in many dimensions has three ingredients, and each answers a question. The first is the distance: the farther a point is from the centre in shape units, the less likely it is, falling off as $e^{-d_M^2/2}$. The second is the size of the cloud. The density must still add up to 1, so a larger cloud spreads the same total over more room and has a lower peak. The size of a shape is $\det\Sigma$, its determinant, so the peak shrinks by $1/\sqrt{\det\Sigma}$. The third is a constant, $(2\pi)^{-d/2}$, which fixes the total at 1. Take the log and the product becomes a sum:

$$
\log p(x \mid y = k) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log\det\Sigma - \frac{1}{2}(x - \mu_k)^{\top}\Sigma^{-1}(x - \mu_k).
$$

With one shared $\Sigma$, the first two terms are identical for every class, so they cancel when you compare classes. Only the distance term and the prior separate them. That is the picture: GDA picks the class that is near in shape units, tilted by how common the class is.

Use the four points at $x = 3$. The constant is $-\tfrac12\log(2\pi) = -0.9189$ and $\log\det\Sigma = 0$. Class 0 is 1 unit away, so its log density is $-0.9189 - 0.5 = -1.4189$. Class 1 is 3 units away, so $-0.9189 - 4.5 = -5.4189$. Adding $\log 0.5 = -0.6931$ gives log scores $-2.1121$ and $-6.1121$. The log-sum-exp is $-2.0939$, so the posteriors are $e^{-0.0182}$ and $e^{-4.0182}$, which is 0.982 and 0.018. The point at 3 belongs to class 0.

## Factor the shape instead of inverting it

The formula has $\Sigma^{-1}$, but you should not call `np.linalg.inv`. An inverse is slow to build and loses digits when $\Sigma$ is nearly flat in some direction. Instead, factor the shape once as $\Sigma = L L^{\top}$, where $L$ is lower triangular (zeros above the diagonal). This is the *Cholesky factor*. Two facts then do all the work.

First, solving $L z = x - \mu$ for $z$ is cheap, because the triangle lets you solve from the top row down. And $z^{\top}z = (x-\mu)^{\top}\Sigma^{-1}(x-\mu)$, the squared shape distance. Second, the determinant of a triangle is the product of its diagonal, so $\log\det\Sigma = 2\sum_i \log L_{ii}$.

Take $\Sigma = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$. Its factor is $L = \begin{bmatrix} 2 & 0 \\ 1 & \sqrt2 \end{bmatrix}$, since $LL^{\top}$ returns $\Sigma$. For a gap of $(2, 3)$, the first row gives $z_1 = 2/2 = 1$. The second gives $z_2 = (3 - 1\cdot 1)/\sqrt2 = \sqrt2$. So $z^{\top}z = 1 + 2 = 3$, and the inverse route gives the same 3. The log determinant is $2(\log 2 + \log\sqrt2) = 2.079$, which is $\log 8$, and $\det\Sigma = 12 - 4 = 8$. In numpy, `np.linalg.cholesky` returns $L$ and `np.linalg.solve(L, diff)` solves for $z$. A dedicated triangular solver is faster still, but the general solve is fine at this size.

> [!KEY]
> Back to the parcel between the clouds. Fit each class by averaging: its mean is the centre, and one shared covariance, the average squared gap from each point's own class mean, is the shape. Then "closer" means closer in shape units, the Mahalanobis distance, tilted by the prior. A ruler picked the wrong class for the triangle, and the shape distance picks the right one.

> [!RECAP]
> - A class is a Gaussian cloud: a mean for the centre and a covariance for the shape.
> - Maximum likelihood fits a Gaussian by averaging, and the shared covariance divides by $n$, using gaps from each class's own mean.
> - A point's score is its prior, discounted by squared Mahalanobis distance, which is distance in units of the shape.
> - With a shared covariance, the constant and the log determinant cancel across classes.
> - Factor the covariance with Cholesky instead of inverting it.

## Build it step by step

The functions `log_sum_exp` and `log_posterior_from_log_density` from the previous chapter are already in `starter.py`. You write three more, in this order.

1. **`fit_gda(X, y)`**. Takes `X` of shape `(n, d)` and integer labels `y`. Returns `means` `(K, d)`, `cov` `(d, d)`, and `priors` `(K,)`, with `K = int(y.max()) + 1`. Accumulate `diff.T @ diff` for each class into one array and divide once by `n`. The usual mistake is dividing by `n - 1` or by each class's count.
2. **`gaussian_log_density(X, mean, cov)`**. Takes `X` of shape `(n, d)` and returns `(n,)` log densities. Factor `cov` once, solve `np.linalg.solve(L, (X - mean).T)`, and sum the squares down axis 0. The usual mistake is dropping the factor $\tfrac12$ on the distance term. Unlike the constant, it does not cancel across classes.
3. **`gda_log_posterior(X, means, cov, priors)`**. Stacks one log density per class into an `(n, K)` array, and passes it to `log_posterior_from_log_density`. The usual mistake is stacking along the wrong axis, which gives `(K, n)`.

## Where this breaks

**A covariance you cannot factor.** If a feature is constant, or two features are copies, $\Sigma$ has a flat direction and Cholesky fails. With $n$ points in $K$ classes, the pooled covariance has rank at most $n - K$, so too many features for the rows fails the same way. Adding a small multiple of the identity to `cov` repairs it.

## The exercise

You implement `fit_gda`, `gaussian_log_density`, and `gda_log_posterior`. The checks reproduce the four-point example, and compare your estimates with a direct computation. They test `gaussian_log_density` against scipy, and check that equal straight-line distances give different densities along different axes of a stretched cloud. The last two confirm the posterior 0.982 at $x = 3$, and that three unequal classes match an independent calculation.
