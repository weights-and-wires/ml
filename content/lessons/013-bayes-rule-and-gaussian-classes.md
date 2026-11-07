---
title: "Bayes' rule and Gaussian classes"
slug: 013-bayes-rule-and-gaussian-classes
part: "Part 2 — Classical models"
summary: Classify by prior times how typical a point is for each class, in logs, then fit one Gaussian cloud per class by averaging.
minutes: 75
gpu: false
objectives:
  - Explain what a generative classifier models, and what a density is and is not.
  - Compute a posterior by hand from priors and class densities, and say why the prior can flip the winner.
  - Write a numerically stable log-sum-exp and use it to turn log scores into probabilities.
  - Explain a covariance matrix as the shape of a cloud, and a Mahalanobis distance as distance measured in that shape.
  - Fit class means, one shared covariance, and priors by averaging, and score points with a Cholesky factor.
prereqs:
  - 008-sigmoid-and-log-loss
  - 009-fitting-logistic-regression
  - 012-learning-curves-and-winners-curse
exercise: 013-bayes-rule-and-gaussian-classes
---

# Bayes' rule and Gaussian classes

A parcel weighing 3 kg arrives at a depot with no label. Two warehouses could have sent it. Warehouse 0 ships mostly light parcels and warehouse 1 mostly heavy ones, but warehouse 0 ships nine parcels for every one from warehouse 1. Which warehouse sent it, and how sure are you?

If you look only at the weight, 3 kg is a little more typical of the heavy warehouse. If you look only at the traffic, warehouse 0 is the safe guess. Neither fact answers the question alone, so you need a recipe that weighs both and tells you a probability, not only a name.

[Logistic regression](/lessons/009-fitting-logistic-regression) learned a boundary directly. This chapter takes the opposite route. You describe what each warehouse's parcels look like, then ask which description would more plausibly have produced this one. That is a *generative classifier*, because it models how the data are generated. The first half builds the recipe for one weight. The second half measures each parcel in two ways, so each one is a point in a plane, and shows how to fit the descriptions from labelled data when the clouds are long, thin, and tilted.

## What you already know

- A probability lies between 0 and 1, and the probabilities of all the classes add to 1.
- A bell curve has a centre (its mean) and a width (its variance, the square of the standard deviation).
- The sigmoid turns a score into a probability, as in [sigmoid and log loss](/lessons/008-sigmoid-and-log-loss).
- A logarithm turns a product into a sum, so $\log(ab) = \log a + \log b$.
- Logistic regression models the class given the point, $p(y \mid x)$, and never models what the points look like.

## Why describe each class at all

Why not keep learning the boundary directly? Often you should. But a model of each class can flag a point that looks like no class at all, can invent typical points by sampling, and can settle on a good answer from fewer examples when its assumption about the shape is right, as the [next chapter](/lessons/014-gda-logistic-and-naive-bayes) shows. The plan is to learn, for each class, a rule that says how typical a given weight is for that class. That rule is a *density*, and it is easy to misread, so start there.

## The picture: prior times height

Warehouse 0's parcel weights follow a bell curve centred at 1.5 kg, and warehouse 1's follow a bell centred at 4 kg. Both have a standard deviation of 1 kg. The dashed curves in Figure 1 are these two bells.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Two bell-shaped weight curves for warehouses 0 and 1, each shrunk by its prior, with a vertical line at 3 kg">
<line x1="50" y1="230" x2="560" y2="230" class="stroke-muted"/>
<line x1="50.0" y1="230" x2="50.0" y2="236" class="stroke-muted"/><text x="50.0" y="252" text-anchor="middle" class="fig-small">0</text>
<line x1="128.5" y1="230" x2="128.5" y2="236" class="stroke-muted"/><text x="128.5" y="252" text-anchor="middle" class="fig-small">1</text>
<line x1="206.9" y1="230" x2="206.9" y2="236" class="stroke-muted"/><text x="206.9" y="252" text-anchor="middle" class="fig-small">2</text>
<line x1="285.4" y1="230" x2="285.4" y2="236" class="stroke-muted"/><text x="285.4" y="252" text-anchor="middle" class="fig-small">3</text>
<line x1="363.8" y1="230" x2="363.8" y2="236" class="stroke-muted"/><text x="363.8" y="252" text-anchor="middle" class="fig-small">4</text>
<line x1="442.3" y1="230" x2="442.3" y2="236" class="stroke-muted"/><text x="442.3" y="252" text-anchor="middle" class="fig-small">5</text>
<line x1="520.8" y1="230" x2="520.8" y2="236" class="stroke-muted"/><text x="520.8" y="252" text-anchor="middle" class="fig-small">6</text>
<text x="305" y="278" text-anchor="middle">parcel weight (kg)</text>
<polyline points="50.0,173.7 53.9,169.4 57.8,164.9 61.8,160.3 65.7,155.5 69.6,150.6 73.5,145.6 77.5,140.5 81.4,135.3 85.3,130.1 89.2,124.8 93.2,119.5 97.1,114.3 101.0,109.1 104.9,104.0 108.8,99.1 112.8,94.2 116.7,89.6 120.6,85.1 124.5,80.9 128.5,76.9 132.4,73.2 136.3,69.9 140.2,66.9 144.2,64.2 148.1,61.9 152.0,60.0 155.9,58.5 159.8,57.4 163.8,56.8 167.7,56.5 171.6,56.8 175.5,57.4 179.5,58.5 183.4,60.0 187.3,61.9 191.2,64.2 195.2,66.9 199.1,69.9 203.0,73.2 206.9,76.9 210.8,80.9 214.8,85.1 218.7,89.6 222.6,94.2 226.5,99.1 230.5,104.0 234.4,109.1 238.3,114.3 242.2,119.5 246.2,124.8 250.1,130.1 254.0,135.3 257.9,140.5 261.8,145.6 265.8,150.6 269.7,155.5 273.6,160.3 277.5,164.9 281.5,169.4 285.4,173.7 289.3,177.8 293.2,181.8 297.2,185.5 301.1,189.1 305.0,192.5 308.9,195.7 312.8,198.7 316.8,201.5 320.7,204.1 324.6,206.5 328.5,208.8 332.5,210.9 336.4,212.8 340.3,214.6 344.2,216.2 348.2,217.7 352.1,219.0 356.0,220.3 359.9,221.4 363.8,222.4 367.8,223.3 371.7,224.1 375.6,224.8 379.5,225.5 383.5,226.0 387.4,226.6 391.3,227.0 395.2,227.4 399.2,227.8 403.1,228.1 407.0,228.3 410.9,228.6 414.8,228.8 418.8,229.0 422.7,229.1 426.6,229.3 430.5,229.4 434.5,229.5 438.4,229.5 442.3,229.6 446.2,229.7 450.2,229.7 454.1,229.8 458.0,229.8 461.9,229.8 465.8,229.9 469.8,229.9 473.7,229.9 477.6,229.9 481.5,229.9 485.5,230.0 489.4,230.0 493.3,230.0 497.2,230.0 501.2,230.0 505.1,230.0 509.0,230.0 512.9,230.0 516.8,230.0 520.8,230.0 524.7,230.0 528.6,230.0 532.5,230.0 536.5,230.0 540.4,230.0 544.3,230.0 548.2,230.0 552.2,230.0 556.1,230.0 560.0,230.0" fill="none" class="stroke-a dashed"/>
<polyline points="50.0,229.9 53.9,229.9 57.8,229.9 61.8,229.9 65.7,229.9 69.6,229.8 73.5,229.8 77.5,229.8 81.4,229.7 85.3,229.7 89.2,229.6 93.2,229.5 97.1,229.5 101.0,229.4 104.9,229.3 108.8,229.1 112.8,229.0 116.7,228.8 120.6,228.6 124.5,228.3 128.5,228.1 132.4,227.8 136.3,227.4 140.2,227.0 144.2,226.6 148.1,226.0 152.0,225.5 155.9,224.8 159.8,224.1 163.8,223.3 167.7,222.4 171.6,221.4 175.5,220.3 179.5,219.0 183.4,217.7 187.3,216.2 191.2,214.6 195.2,212.8 199.1,210.9 203.0,208.8 206.9,206.5 210.8,204.1 214.8,201.5 218.7,198.7 222.6,195.7 226.5,192.5 230.5,189.1 234.4,185.5 238.3,181.8 242.2,177.8 246.2,173.7 250.1,169.4 254.0,164.9 257.9,160.3 261.8,155.5 265.8,150.6 269.7,145.6 273.6,140.5 277.5,135.3 281.5,130.1 285.4,124.8 289.3,119.5 293.2,114.3 297.2,109.1 301.1,104.0 305.0,99.1 308.9,94.2 312.8,89.6 316.8,85.1 320.7,80.9 324.6,76.9 328.5,73.2 332.5,69.9 336.4,66.9 340.3,64.2 344.2,61.9 348.2,60.0 352.1,58.5 356.0,57.4 359.9,56.8 363.8,56.5 367.8,56.8 371.7,57.4 375.6,58.5 379.5,60.0 383.5,61.9 387.4,64.2 391.3,66.9 395.2,69.9 399.2,73.2 403.1,76.9 407.0,80.9 410.9,85.1 414.8,89.6 418.8,94.2 422.7,99.1 426.6,104.0 430.5,109.1 434.5,114.3 438.4,119.5 442.3,124.8 446.2,130.1 450.2,135.3 454.1,140.5 458.0,145.6 461.9,150.6 465.8,155.5 469.8,160.3 473.7,164.9 477.6,169.4 481.5,173.7 485.5,177.8 489.4,181.8 493.3,185.5 497.2,189.1 501.2,192.5 505.1,195.7 509.0,198.7 512.9,201.5 516.8,204.1 520.8,206.5 524.7,208.8 528.6,210.9 532.5,212.8 536.5,214.6 540.4,216.2 544.3,217.7 548.2,219.0 552.2,220.3 556.1,221.4 560.0,222.4" fill="none" class="stroke-b dashed"/>
<polygon points="50.0,230.0 50.0,179.3 53.9,175.4 57.8,171.4 61.8,167.2 65.7,162.9 69.6,158.5 73.5,154.0 77.5,149.4 81.4,144.8 85.3,140.0 89.2,135.3 93.2,130.6 97.1,125.9 101.0,121.2 104.9,116.6 108.8,112.2 112.8,107.8 116.7,103.6 120.6,99.6 124.5,95.8 128.5,92.2 132.4,88.9 136.3,85.9 140.2,83.2 144.2,80.8 148.1,78.7 152.0,77.0 155.9,75.6 159.8,74.7 163.8,74.1 167.7,73.9 171.6,74.1 175.5,74.7 179.5,75.6 183.4,77.0 187.3,78.7 191.2,80.8 195.2,83.2 199.1,85.9 203.0,88.9 206.9,92.2 210.8,95.8 214.8,99.6 218.7,103.6 222.6,107.8 226.5,112.2 230.5,116.6 234.4,121.2 238.3,125.9 242.2,130.6 246.2,135.3 250.1,140.0 254.0,144.8 257.9,149.4 261.8,154.0 265.8,158.5 269.7,162.9 273.6,167.2 277.5,171.4 281.5,175.4 285.4,179.3 289.3,183.0 293.2,186.6 297.2,190.0 301.1,193.2 305.0,196.2 308.9,199.1 312.8,201.8 316.8,204.3 320.7,206.7 324.6,208.9 328.5,210.9 332.5,212.8 336.4,214.5 340.3,216.1 344.2,217.6 348.2,218.9 352.1,220.1 356.0,221.2 359.9,222.2 363.8,223.1 367.8,224.0 371.7,224.7 375.6,225.3 379.5,225.9 383.5,226.4 387.4,226.9 391.3,227.3 395.2,227.7 399.2,228.0 403.1,228.3 407.0,228.5 410.9,228.7 414.8,228.9 418.8,229.1 422.7,229.2 426.6,229.3 430.5,229.4 434.5,229.5 438.4,229.6 442.3,229.7 446.2,229.7 450.2,229.8 454.1,229.8 458.0,229.8 461.9,229.9 465.8,229.9 469.8,229.9 473.7,229.9 477.6,229.9 481.5,229.9 485.5,230.0 489.4,230.0 493.3,230.0 497.2,230.0 501.2,230.0 505.1,230.0 509.0,230.0 512.9,230.0 516.8,230.0 520.8,230.0 524.7,230.0 528.6,230.0 532.5,230.0 536.5,230.0 540.4,230.0 544.3,230.0 548.2,230.0 552.2,230.0 556.1,230.0 560.0,230.0 560.0,230.0" class="fill-soft-a"/>
<polygon points="50.0,230.0 50.0,230.0 53.9,230.0 57.8,230.0 61.8,230.0 65.7,230.0 69.6,230.0 73.5,230.0 77.5,230.0 81.4,230.0 85.3,230.0 89.2,230.0 93.2,230.0 97.1,229.9 101.0,229.9 104.9,229.9 108.8,229.9 112.8,229.9 116.7,229.9 120.6,229.9 124.5,229.8 128.5,229.8 132.4,229.8 136.3,229.7 140.2,229.7 144.2,229.7 148.1,229.6 152.0,229.5 155.9,229.5 159.8,229.4 163.8,229.3 167.7,229.2 171.6,229.1 175.5,229.0 179.5,228.9 183.4,228.8 187.3,228.6 191.2,228.5 195.2,228.3 199.1,228.1 203.0,227.9 206.9,227.7 210.8,227.4 214.8,227.1 218.7,226.9 222.6,226.6 226.5,226.2 230.5,225.9 234.4,225.6 238.3,225.2 242.2,224.8 246.2,224.4 250.1,223.9 254.0,223.5 257.9,223.0 261.8,222.5 265.8,222.1 269.7,221.6 273.6,221.0 277.5,220.5 281.5,220.0 285.4,219.5 289.3,219.0 293.2,218.4 297.2,217.9 301.1,217.4 305.0,216.9 308.9,216.4 312.8,216.0 316.8,215.5 320.7,215.1 324.6,214.7 328.5,214.3 332.5,214.0 336.4,213.7 340.3,213.4 344.2,213.2 348.2,213.0 352.1,212.8 356.0,212.7 359.9,212.7 363.8,212.7 367.8,212.7 371.7,212.7 375.6,212.8 379.5,213.0 383.5,213.2 387.4,213.4 391.3,213.7 395.2,214.0 399.2,214.3 403.1,214.7 407.0,215.1 410.9,215.5 414.8,216.0 418.8,216.4 422.7,216.9 426.6,217.4 430.5,217.9 434.5,218.4 438.4,219.0 442.3,219.5 446.2,220.0 450.2,220.5 454.1,221.0 458.0,221.6 461.9,222.1 465.8,222.5 469.8,223.0 473.7,223.5 477.6,223.9 481.5,224.4 485.5,224.8 489.4,225.2 493.3,225.6 497.2,225.9 501.2,226.2 505.1,226.6 509.0,226.9 512.9,227.1 516.8,227.4 520.8,227.7 524.7,227.9 528.6,228.1 532.5,228.3 536.5,228.5 540.4,228.6 544.3,228.8 548.2,228.9 552.2,229.0 556.1,229.1 560.0,229.2 560.0,230.0" class="fill-soft-b"/>
<polyline points="50.0,179.3 53.9,175.4 57.8,171.4 61.8,167.2 65.7,162.9 69.6,158.5 73.5,154.0 77.5,149.4 81.4,144.8 85.3,140.0 89.2,135.3 93.2,130.6 97.1,125.9 101.0,121.2 104.9,116.6 108.8,112.2 112.8,107.8 116.7,103.6 120.6,99.6 124.5,95.8 128.5,92.2 132.4,88.9 136.3,85.9 140.2,83.2 144.2,80.8 148.1,78.7 152.0,77.0 155.9,75.6 159.8,74.7 163.8,74.1 167.7,73.9 171.6,74.1 175.5,74.7 179.5,75.6 183.4,77.0 187.3,78.7 191.2,80.8 195.2,83.2 199.1,85.9 203.0,88.9 206.9,92.2 210.8,95.8 214.8,99.6 218.7,103.6 222.6,107.8 226.5,112.2 230.5,116.6 234.4,121.2 238.3,125.9 242.2,130.6 246.2,135.3 250.1,140.0 254.0,144.8 257.9,149.4 261.8,154.0 265.8,158.5 269.7,162.9 273.6,167.2 277.5,171.4 281.5,175.4 285.4,179.3 289.3,183.0 293.2,186.6 297.2,190.0 301.1,193.2 305.0,196.2 308.9,199.1 312.8,201.8 316.8,204.3 320.7,206.7 324.6,208.9 328.5,210.9 332.5,212.8 336.4,214.5 340.3,216.1 344.2,217.6 348.2,218.9 352.1,220.1 356.0,221.2 359.9,222.2 363.8,223.1 367.8,224.0 371.7,224.7 375.6,225.3 379.5,225.9 383.5,226.4 387.4,226.9 391.3,227.3 395.2,227.7 399.2,228.0 403.1,228.3 407.0,228.5 410.9,228.7 414.8,228.9 418.8,229.1 422.7,229.2 426.6,229.3 430.5,229.4 434.5,229.5 438.4,229.6 442.3,229.7 446.2,229.7 450.2,229.8 454.1,229.8 458.0,229.8 461.9,229.9 465.8,229.9 469.8,229.9 473.7,229.9 477.6,229.9 481.5,229.9 485.5,230.0 489.4,230.0 493.3,230.0 497.2,230.0 501.2,230.0 505.1,230.0 509.0,230.0 512.9,230.0 516.8,230.0 520.8,230.0 524.7,230.0 528.6,230.0 532.5,230.0 536.5,230.0 540.4,230.0 544.3,230.0 548.2,230.0 552.2,230.0 556.1,230.0 560.0,230.0" fill="none" class="stroke-a"/>
<polyline points="50.0,230.0 53.9,230.0 57.8,230.0 61.8,230.0 65.7,230.0 69.6,230.0 73.5,230.0 77.5,230.0 81.4,230.0 85.3,230.0 89.2,230.0 93.2,230.0 97.1,229.9 101.0,229.9 104.9,229.9 108.8,229.9 112.8,229.9 116.7,229.9 120.6,229.9 124.5,229.8 128.5,229.8 132.4,229.8 136.3,229.7 140.2,229.7 144.2,229.7 148.1,229.6 152.0,229.5 155.9,229.5 159.8,229.4 163.8,229.3 167.7,229.2 171.6,229.1 175.5,229.0 179.5,228.9 183.4,228.8 187.3,228.6 191.2,228.5 195.2,228.3 199.1,228.1 203.0,227.9 206.9,227.7 210.8,227.4 214.8,227.1 218.7,226.9 222.6,226.6 226.5,226.2 230.5,225.9 234.4,225.6 238.3,225.2 242.2,224.8 246.2,224.4 250.1,223.9 254.0,223.5 257.9,223.0 261.8,222.5 265.8,222.1 269.7,221.6 273.6,221.0 277.5,220.5 281.5,220.0 285.4,219.5 289.3,219.0 293.2,218.4 297.2,217.9 301.1,217.4 305.0,216.9 308.9,216.4 312.8,216.0 316.8,215.5 320.7,215.1 324.6,214.7 328.5,214.3 332.5,214.0 336.4,213.7 340.3,213.4 344.2,213.2 348.2,213.0 352.1,212.8 356.0,212.7 359.9,212.7 363.8,212.7 367.8,212.7 371.7,212.7 375.6,212.8 379.5,213.0 383.5,213.2 387.4,213.4 391.3,213.7 395.2,214.0 399.2,214.3 403.1,214.7 407.0,215.1 410.9,215.5 414.8,216.0 418.8,216.4 422.7,216.9 426.6,217.4 430.5,217.9 434.5,218.4 438.4,219.0 442.3,219.5 446.2,220.0 450.2,220.5 454.1,221.0 458.0,221.6 461.9,222.1 465.8,222.5 469.8,223.0 473.7,223.5 477.6,223.9 481.5,224.4 485.5,224.8 489.4,225.2 493.3,225.6 497.2,225.9 501.2,226.2 505.1,226.6 509.0,226.9 512.9,227.1 516.8,227.4 520.8,227.7 524.7,227.9 528.6,228.1 532.5,228.3 536.5,228.5 540.4,228.6 544.3,228.8 548.2,228.9 552.2,229.0 556.1,229.1 560.0,229.2" fill="none" class="stroke-b"/>
<line x1="285.4" y1="230" x2="285.4" y2="32.2" class="stroke-ink"/>
<text x="291.4" y="34.3" class="fig-bold">3 kg</text>
<circle cx="285.4" cy="173.7" r="4" class="fill-a"/>
<circle cx="285.4" cy="124.8" r="4" class="fill-b"/>
<circle cx="285.4" cy="179.3" r="4" class="fill-a"/>
<circle cx="285.4" cy="219.5" r="4" class="fill-b"/>
<text x="69.6" y="34.3" class="fig-bold">warehouse 0</text>
<text x="69.6" y="50.3" class="fig-small">90% of parcels</text>
<text x="442.3" y="34.3" class="fig-bold">warehouse 1</text>
<text x="442.3" y="50.3" class="fig-small">10% of parcels</text>
<text x="295.4" y="128.8" text-anchor="start" class="fig-small">0.242</text>
<text x="295.4" y="177.7" text-anchor="start" class="fig-small">0.130</text>
<text x="275.4" y="183.3" text-anchor="end" class="fig-small">0.117</text>
<text x="275.4" y="211.5" text-anchor="end" class="fig-small">0.024</text>
</svg>
Figure 1. The height of a curve is a density. Dashed curves are each warehouse's own weight density. Shaded curves are the same densities shrunk by how often each warehouse ships. At 3 kg the heavy warehouse's dashed curve is higher, but after shrinking, warehouse 0's shaded curve is almost five times taller.
```

A *density* is the height of such a curve at a weight. It is not a probability. The chance that a parcel weighs exactly 3.000 kg is zero, because a scale can always read one more decimal. What has meaning is the chance of landing in a thin slice, say 3.00 to 3.01 kg, and that chance is about the height times the slice width. Both warehouses get the same slice, so the ratio of heights is the ratio of chances. That is why you can compare heights at one weight, and why a height can exceed 1: a very narrow bell is tall, and its area is still 1.

A bell with centre $\mu$ and variance $\sigma^2$ has height

$$
\frac{1}{\sqrt{2\pi\sigma^2}} \, e^{-(x-\mu)^2 / (2\sigma^2)}.
$$

Read it aloud: the exponent is zero at the centre, which gives the peak, and it falls off with the squared gap, measured in units of the variance. The factor in front makes the area under the curve equal to 1. At 3 kg, warehouse 0's gap is $3 - 1.5 = 1.5$, so the exponent is $-2.25/2 = -1.125$, and the height is $0.3989 \times e^{-1.125} = 0.130$. Warehouse 1's gap is 1, the exponent is $-0.5$, and the height is $0.3989 \times 0.607 = 0.242$. Judged on weight alone, the heavy warehouse is the better guess, by about 1.9 to 1.

Now bring in the traffic. Nine parcels in ten come from warehouse 0, so before you weigh anything, warehouse 0 has a head start. Shrink each bell by its share of the traffic, which gives the shaded curves. Warehouse 0's shaded height at 3 kg is $0.9 \times 0.130 = 0.117$, and warehouse 1's is $0.1 \times 0.242 = 0.024$. After the shrink the order has flipped, and warehouse 0 is ahead by about 4.8 to 1. The traffic share is the *prior*, the probability of a class before you look at the point. The shrunk height is the prior times the density.

## Rescale the scores into probabilities

The two scores, 0.1166 and 0.0242, do not add to 1, so they are not probabilities yet. Every parcel at 3 kg came from one of the two warehouses, so their shares must add to 1. Divide each score by the total, 0.1408. Warehouse 0 gets $0.1166 / 0.1408 = 0.828$ and warehouse 1 gets $0.0242 / 0.1408 = 0.172$.

You can be about 83 percent sure the parcel came from the busy, light warehouse. The division never changes the winner, because it shrinks both scores by the same factor, so you need it only for an honest probability.

Why not compare the densities and ignore the traffic? Because that rule is wrong whenever the classes are unevenly common. With equal priors, the same two bells give 0.349 for warehouse 0 and 0.651 for warehouse 1, so the answer flips to the heavy warehouse, and a depot that follows that rule sends nine parcels in ten to the wrong place whenever the weights are ambiguous. With equal priors the two shaded curves cross at 2.75 kg, halfway between the means. When warehouse 0 is nine times as busy, the crossing moves to 3.63 kg, and swapping the priors moves it to 1.87 kg. The boundary moves toward the rarer class, because the rare class needs stronger evidence.

For class $k$, write $\pi_k$ for its prior (the Greek letter pi, used as a name for that share) and $p(x \mid y = k)$ for its density at the point $x$. Then

$$
p(y = k \mid x) = \frac{\pi_k \, p(x \mid y = k)}{\sum_j \pi_j \, p(x \mid y = j)} .
$$

Read it aloud: the probability of class $k$ given the point is that class's prior times density, divided by the same quantity added up over every class. The top is the shrunk height, and the bottom is the total, 0.1408 in the example. This is *Bayes' rule*, and the left side, $p(y = k \mid x)$, is the *posterior*, the probability after you have seen the point.

## Work in logs

A real density is often tiny. A thousand-word document multiplies a thousand small numbers, a computer stores numbers only down to about $10^{-308}$, and the product silently becomes 0. The fix is to keep every score as a logarithm, where a product becomes a sum and a tiny number becomes a moderate negative one. The log score for class $k$ is $\log \pi_k + \log p(x \mid y = k)$. In the example, those are $-2.149$ and $-3.722$.

Dividing by the total then needs the log of a sum of exponentials, the *log-sum-exp*:

$$
\log \sum_j e^{a_j}, \qquad \text{where } a_j = \log \pi_j + \log p(x \mid y = j).
$$

Computed directly, it underflows again, because $e^{-1000}$ is 0, so subtract the largest score $m$ first. Since $\sum_j e^{a_j} = e^{m}\sum_j e^{a_j - m}$, you can write

$$
\log \sum_j e^{a_j} = m + \log \sum_j e^{a_j - m}.
$$

The largest term becomes $e^{0} = 1$, so nothing underflows to zero. Take two scores, $-1000$ and $-1001$. Directly, both exponentials are 0. With the trick, $m = -1000$, the sum is $1 + e^{-1} = 1.368$, and the log-sum-exp is $-1000 + \log 1.368 = -999.687$. The posteriors are $1 / 1.368 = 0.731$ and $0.368 / 1.368 = 0.269$.

The log posterior is the log score minus the log-sum-exp, and `np.exp` of it adds to 1 along each row. In numpy, with `log_density` of shape `(n, K)`:

```python
joint = log_density + np.log(priors)             # shape (n, K)
log_post = joint - log_sum_exp(joint, axis=1)[:, None]
```

> [!WARNING]
> Normalise across the classes, which is `axis=1` when each row is one point. Using `axis=0` makes each class's column add to 1 instead, and the posteriors look plausible but are wrong. Check that `np.exp(log_post).sum(axis=1)` is 1 for every row.

> [!KEY]
> A generative classifier scores each warehouse by its prior times how typical 3 kg is for it, so the busy warehouse beats the slightly more typical heavy one, 83 percent to 17 percent. Dividing by the total only turns scores into probabilities, and working in logs with log-sum-exp keeps every step safe from underflow.

## A cloud has a centre and a shape

Now measure each parcel two ways, so each one is a point on a page. The parcels from warehouse 0 make a cloud of points, and the parcels from warehouse 1 make another. Both clouds are long and thin, and both lean the same way, because heavier parcels tend to be larger. A parcel arrives that sits between the clouds. Which cloud is it closer to, when "closer" must respect that the clouds are long in one direction and narrow in another? The method is called *Gaussian discriminant analysis*, or GDA, and it supplies the missing step: how to fit those bells from labelled data, in many dimensions at once.

To describe one class, you need where its points are centred and how they spread. The centre is the *mean*: add the points and divide by how many there are. The spread is the *covariance matrix*, written $\Sigma$ (the capital Greek letter sigma). On the diagonal it holds the variance of each feature, which says how wide the cloud is along each axis. Off the diagonal it holds how much two features move together, which says how much the cloud leans. A covariance matrix is therefore a shape, and every such shape draws as an ellipse.

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
Figure 2. Crosses mark the class means, and each solid ellipse holds the points within distance 2 of its mean, measured in units of the shared shape. One shared shape, fitted from the gaps around each class's own mean, wraps both clouds. The dashed grey ellipse comes from subtracting the overall mean instead: it also counts the gap between the two centres as spread, so it is far too large. The triangle is the point the chapter classifies.
```

Figure 2 shows two classes with a cross at each class mean and an ellipse around each. Both ellipses are the same shape, because the model gives all classes one shared covariance. A covariance of $d$ features has about $d^2$ numbers, and a class with few points cannot pin them all down, so pooling every class's points into one estimate steadies it. The price is that the model cannot describe a round class and a long one.

## Why a ruler fails

Back to the parcel between the clouds, the triangle in Figure 2. Its straight-line distance to the class 0 mean is 2.46, and to the class 1 mean it is 2.70, so a ruler says class 0. But the triangle's gap from the class 0 mean runs across the clouds' lean, the short way, where a cloud almost never reaches. Its gap from the class 1 mean runs mostly along the lean, where points are common.

```figure
<svg viewBox="0 0 620 330" role="img" aria-label="A tilted ellipse with two points at the same straight-line distance from its centre but different Mahalanobis distances">
<marker id="c013-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker>
<polygon points="326.2,114.0 326.1,112.4 325.7,110.9 325.0,109.6 324.1,108.5 322.9,107.5 321.5,106.7 319.8,106.1 317.8,105.6 315.6,105.2 313.1,105.1 310.5,105.1 307.6,105.3 304.5,105.6 301.1,106.1 297.6,106.8 293.9,107.6 290.0,108.6 286.0,109.8 281.7,111.1 277.4,112.5 272.9,114.1 268.3,115.9 263.6,117.7 258.8,119.8 253.9,121.9 248.9,124.1 243.9,126.5 238.9,129.0 233.8,131.6 228.7,134.2 223.7,137.0 218.6,139.8 213.6,142.7 208.6,145.7 203.7,148.7 198.8,151.8 194.0,154.9 189.4,158.0 184.8,161.2 180.4,164.4 176.1,167.5 172.0,170.7 168.0,173.8 164.2,176.9 160.6,180.0 157.2,183.1 154.0,186.0 151.0,189.0 148.2,191.8 145.6,194.6 143.3,197.3 141.2,200.0 139.4,202.5 137.8,204.9 136.5,207.2 135.4,209.4 134.6,211.5 134.1,213.4 133.8,215.2 133.8,216.8 134.1,218.4 134.6,219.7 135.4,220.9 136.5,222.0 137.8,222.9 139.4,223.6 141.2,224.2 143.3,224.6 145.6,224.9 148.2,224.9 151.0,224.8 154.0,224.6 157.2,224.2 160.6,223.6 164.2,222.8 168.0,221.9 172.0,220.8 176.1,219.6 180.4,218.2 184.8,216.7 189.4,215.0 194.0,213.2 198.8,211.3 203.7,209.2 208.6,207.0 213.6,204.7 218.6,202.3 223.7,199.7 228.7,197.1 233.8,194.4 238.9,191.6 243.9,188.7 248.9,185.8 253.9,182.8 258.8,179.7 263.6,176.6 268.3,173.5 272.9,170.4 277.4,167.2 281.7,164.1 286.0,160.9 290.0,157.8 293.9,154.6 297.6,151.5 301.1,148.5 304.5,145.4 307.6,142.5 310.5,139.6 313.1,136.8 315.6,134.0 317.8,131.3 319.8,128.8 321.5,126.3 322.9,123.9 324.1,121.7 325.0,119.6 325.7,117.6 326.1,115.7 326.2,114.0" fill="none" class="stroke-b"/>
<polygon points="422.5,62.9 422.2,59.8 421.4,56.9 420.1,54.3 418.2,52.0 415.8,50.1 412.9,48.4 409.5,47.1 405.6,46.1 401.2,45.5 396.3,45.2 390.9,45.2 385.1,45.5 378.9,46.2 372.3,47.2 365.2,48.6 357.8,50.2 350.0,52.2 341.9,54.5 333.5,57.1 324.8,60.0 315.8,63.2 306.6,66.7 297.2,70.5 287.5,74.5 277.8,78.8 267.9,83.3 257.9,88.0 247.8,93.0 237.6,98.2 227.5,103.5 217.3,109.0 207.2,114.7 197.1,120.5 187.2,126.4 177.3,132.5 167.6,138.6 158.1,144.8 148.8,151.1 139.7,157.4 130.8,163.7 122.3,170.0 114.0,176.3 106.0,182.6 98.4,188.9 91.2,195.0 84.4,201.1 77.9,207.1 71.9,213.0 66.3,218.7 61.2,224.3 56.6,229.7 52.4,234.9 48.7,240.0 45.6,244.8 42.9,249.4 40.8,253.8 39.2,257.9 38.1,261.8 37.6,265.4 37.6,268.7 38.1,271.7 39.2,274.5 40.8,276.9 42.9,279.0 45.6,280.8 48.7,282.3 52.4,283.4 56.6,284.2 61.2,284.7 66.3,284.9 71.9,284.7 77.9,284.2 84.4,283.3 91.2,282.1 98.4,280.6 106.0,278.8 114.0,276.7 122.3,274.2 130.8,271.5 139.7,268.4 148.8,265.1 158.1,261.4 167.6,257.5 177.3,253.4 187.2,249.0 197.1,244.4 207.2,239.5 217.3,234.5 227.5,229.2 237.6,223.8 247.8,218.2 257.9,212.4 267.9,206.6 277.8,200.6 287.5,194.5 297.2,188.3 306.6,182.1 315.8,175.8 324.8,169.5 333.5,163.1 341.9,156.8 350.0,150.5 357.8,144.2 365.2,138.0 372.3,131.9 378.9,125.9 385.1,120.0 390.9,114.2 396.3,108.5 401.2,103.0 405.6,97.7 409.5,92.5 412.9,87.6 415.8,82.9 418.2,78.4 420.1,74.1 421.4,70.1 422.2,66.4 422.5,62.9" fill="none" class="stroke-b"/>
<circle cx="230" cy="165" r="110" fill="none" class="stroke-muted dashed"/>
<line x1="230" y1="165" x2="325.3" y2="110.0" class="stroke-ink" marker-end="url(#c013-arrow)"/>
<circle cx="325.3" cy="110.0" r="5" class="fill-a"/>
<text x="337.3" y="102.0" class="fig-bold">A</text>
<line x1="230" y1="165" x2="175.0" y2="69.7" class="stroke-ink" marker-end="url(#c013-arrow)"/>
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
Figure 3. Points A and B are the same straight-line distance from the centre, but A lies along the cloud's long axis and B across its short one. Measured in units of the cloud's own shape, A is typical and B is far out.
```

Figure 3 isolates the effect. Points A and B are the same straight-line distance, 2.0, from the centre. A sits along the cloud's long axis, where the cloud is wide, so it is an ordinary point. B sits across the short axis, so it is far outside. Measured in units of the cloud's own shape, A is 1.0 away and B is 4.0 away. That is the *Mahalanobis distance*: how many shape units a point is from the centre. In one dimension it is the familiar count of standard deviations, $|x - \mu| / \sigma$. In many dimensions the shape matrix does the dividing:

$$
d_M^2 = (x - \mu)^{\top} \Sigma^{-1} (x - \mu).
$$

Read it aloud: take the gap vector from the mean, scale it by the inverse shape so that wide directions count less and narrow ones count more, and take the squared length. For the triangle, the squared shape distances are 25.8 to class 0 and 6.2 to class 1, so by this measure it belongs to class 1.

## Fit the cloud by averaging

Fitting means choosing the bell that makes the training points most likely, called *maximum likelihood*. For a Gaussian it works out to averaging. Take four points on a line, with class 0 at 1 and 3, and class 1 at 5 and 7.

- **Priors.** Two of the four points are in each class, so both priors are $2/4 = 0.5$.
- **Means.** Class 0's mean is $(1 + 3)/2 = 2$ and class 1's is $(5 + 7)/2 = 6$. The mean minimises the total squared gap.
- **Shape.** Each point's gap from its own class mean is $-1, +1, -1, +1$. The squared gaps are all 1, and their sum is 4. Divide by the 4 points to get a variance of 1.

Figure 4 draws the result.

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
Figure 4. The four points give two bells with standard deviation 1 (variance 1), centred at their class means 2 and 6. At x = 3 the class 0 log density, -1.4189, is four higher than class 1's, -5.4189: one unit from a centre costs 0.5, three units cost 4.5.
```

A bell that is too narrow gives distant points a huge penalty, and one that is too wide wastes its height on empty space, and the two pressures balance exactly when the variance equals the average squared gap. In symbols, with $n_k$ points in class $k$ and $n$ points in all,

$$
\pi_k = \frac{n_k}{n}, \qquad
\mu_k = \frac{1}{n_k}\sum_{i:\, y_i = k} x_i, \qquad
\Sigma = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu_{y_i})(x_i - \mu_{y_i})^{\top}.
$$

Read it aloud: the prior is the class's share of the points, the mean is the class's average point, and the shared shape is the average, over every point, of its gap from its own class mean, multiplied by itself as a column times a row. The shape sum divides by $n$, the number of rows, and not by $n - 1$. The $n - 1$ habit from statistics corrects for estimating a population from a sample, but here the maximum likelihood answer is the plain average. The two differ by a factor $n/(n-1)$, which is $4/3$ for these four points.

> [!WARNING]
> Subtract each point's own class mean, never the overall mean. The overall mean counts the gap between the class centres as spread, so the shape is far too large and the model blurs the classes together. The dashed grey ellipse in Figure 2 shows the damage.

## Score a point

A Gaussian density in many dimensions has three ingredients. The first is the distance: the farther a point is from the centre in shape units, the less likely it is, falling off as $e^{-d_M^2/2}$. The second is the size of the cloud: a larger cloud spreads the same total of 1 over more room, so its peak shrinks by $1/\sqrt{\det\Sigma}$, where $\det\Sigma$, the determinant, measures the size of a shape. The third is a constant, $(2\pi)^{-d/2}$. Take the log and the product becomes a sum:

$$
\log p(x \mid y = k) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log\det\Sigma - \frac{1}{2}(x - \mu_k)^{\top}\Sigma^{-1}(x - \mu_k).
$$

With one shared $\Sigma$, the first two terms are identical for every class, so they cancel when you compare classes. Only the distance term and the prior separate them: GDA picks the class that is near in shape units, tilted by how common the class is.

Use the four points at $x = 3$. The constant is $-\tfrac12\log(2\pi) = -0.9189$ and $\log\det\Sigma = 0$. Class 0 is 1 unit away, so its log density is $-0.9189 - 0.5 = -1.4189$. Class 1 is 3 units away, so $-0.9189 - 4.5 = -5.4189$. Adding $\log 0.5 = -0.6931$ gives log scores $-2.1121$ and $-6.1121$. The log-sum-exp is $-2.0939$, so the posteriors are $e^{-0.0182}$ and $e^{-4.0182}$, which is 0.982 and 0.018. The point at 3 belongs to class 0.

### Factor the shape instead of inverting it

The formula has $\Sigma^{-1}$, but an inverse is slow to build and loses digits when $\Sigma$ is nearly flat in some direction. Instead, factor the shape once as $\Sigma = L L^{\top}$, where $L$ is lower triangular (zeros above the diagonal). This is the *Cholesky factor*. Two facts then do all the work. First, solving $L z = x - \mu$ for $z$ is cheap, because the triangle lets you solve from the top row down, and $z^{\top}z = (x-\mu)^{\top}\Sigma^{-1}(x-\mu)$, the squared shape distance. Second, the determinant of a triangle is the product of its diagonal, so $\log\det\Sigma = 2\sum_i \log L_{ii}$.

Take $\Sigma = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$. Its factor is $L = \begin{bmatrix} 2 & 0 \\ 1 & \sqrt2 \end{bmatrix}$, since $LL^{\top}$ returns $\Sigma$. For a gap of $(2, 3)$, the first row gives $z_1 = 2/2 = 1$ and the second gives $z_2 = (3 - 1\cdot 1)/\sqrt2 = \sqrt2$. So $z^{\top}z = 1 + 2 = 3$, and the inverse route gives the same 3. The log determinant is $2(\log 2 + \log\sqrt2) = 2.079$, which is $\log 8$, and $\det\Sigma = 12 - 4 = 8$. In numpy, `np.linalg.cholesky` returns $L$ and `np.linalg.solve(L, diff)` solves for $z$.

> [!KEY]
> Fit each class by averaging: its mean is the centre, and one shared covariance, the average squared gap from each point's own class mean, is the shape. Then "closer" means closer in shape units, the Mahalanobis distance, tilted by the prior. A ruler picked the wrong class for the triangle, and the shape distance picks the right one.

> [!RECAP]
> - A generative classifier models what each class looks like, and the posterior is proportional to prior times density. Ignoring the prior fails when classes are uneven.
> - A density is a height, not a probability. Dividing by the total only rescales the scores to add to 1.
> - Keep scores as logs, and use log-sum-exp with the maximum subtracted.
> - A class is a Gaussian cloud: maximum likelihood gives the mean and a shared covariance that divides by $n$, using gaps from each class's own mean.
> - A point's score is its prior, discounted by squared Mahalanobis distance. Factor the covariance with Cholesky instead of inverting it.

## Build it step by step

You write six functions, in the order the checks test them. A seventh, `predict_class`, is given.

1. **`normal_log_density_1d(x, mean, var)`**. Takes an array `x` and a scalar mean and variance, and returns the log of the Gaussian height at each point, with the shape of `x`. The log of the bell is $-\tfrac12 \log(2\pi\sigma^2) - (x-\mu)^2 / (2\sigma^2)$. The usual mistake is passing the standard deviation where the variance goes.
2. **`log_sum_exp(a, axis)`**. Takes an array and an axis, and returns the log of the sum of exponentials along that axis, with the axis removed. Subtract `np.max(a, axis=axis, keepdims=True)` first and add it back after the log. The usual mistake is subtracting a single global maximum, which helps one row and starves another.
3. **`log_posterior_from_log_density(log_density, priors)`**. Takes `(n, K)` log densities and `(K,)` priors, and returns `(n, K)` log posteriors. Add `np.log(priors)`, then subtract a row-wise log-sum-exp kept as a column. The usual mistake is the wrong axis.
4. **`fit_gda(X, y)`**. Takes `X` of shape `(n, d)` and integer labels `y`. Returns `means` `(K, d)`, `cov` `(d, d)`, and `priors` `(K,)`, with `K = int(y.max()) + 1`. Accumulate `diff.T @ diff` for each class into one array and divide once by `n`. The usual mistake is dividing by `n - 1` or by each class's count.
5. **`gaussian_log_density(X, mean, cov)`**. Takes `X` of shape `(n, d)` and returns `(n,)` log densities. Factor `cov` once, solve `np.linalg.solve(L, (X - mean).T)`, and sum the squares down axis 0. The usual mistake is dropping the factor $\tfrac12$ on the distance term. Unlike the constant, it does not cancel across classes.
6. **`gda_log_posterior(X, means, cov, priors)`**. Stacks one log density per class into an `(n, K)` array, and passes it to `log_posterior_from_log_density`. The usual mistake is stacking along the wrong axis, which gives `(K, n)`.

## Where this breaks

**A covariance you cannot factor.** If a feature is constant, or two features are copies, $\Sigma$ has a flat direction and Cholesky fails. Too many features for the rows fails the same way. Adding a small multiple of the identity to `cov` repairs it.

## The exercise

You implement `normal_log_density_1d`, `log_sum_exp`, `log_posterior_from_log_density`, `fit_gda`, `gaussian_log_density`, and `gda_log_posterior`. The starter's closing programs run the parcel example at two priors, then the four-point example and two tilted clouds.

The checks compare your one-dimensional density and `log_sum_exp` with scipy, including values near $\pm 1000$, and reproduce the 0.828 and 0.172 posterior and the prior flip at 3 kg. They confirm that `fit_gda` reproduces the four-point example and a three-class pooled covariance, that `gaussian_log_density` matches scipy and gives different densities at equal straight-line distances along different axes, and that `gda_log_posterior` gives 0.982 at $x = 3$ and matches an independent calculation on three unequal classes.
