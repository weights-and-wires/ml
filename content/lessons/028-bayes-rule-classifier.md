---
title: Classify by asking who made this
slug: 028-bayes-rule-classifier
part: "Part 2 — Classical models"
summary: Score each class by its prior times how typical the point is for that class, then rescale to probabilities, in logs so nothing underflows.
minutes: 35
gpu: false
objectives:
  - Explain what a generative classifier models, and what a density is and is not.
  - Compute a posterior by hand from priors and class densities, and say why the prior can flip the winner.
  - Write a numerically stable log-sum-exp and use it to turn log scores into probabilities.
prereqs:
  - 017-sigmoid-and-probability
  - 019-gradient-descent-for-logistic
  - 027-winners-curse-and-flexibility
exercise: 028-bayes-rule-classifier
---

# Classify by asking who made this

A parcel weighing 3 kg arrives at a depot with no label. Two warehouses could have sent it. Warehouse 0 ships mostly light parcels and warehouse 1 mostly heavy ones, but warehouse 0 ships nine parcels for every one from warehouse 1. Which warehouse sent it, and how sure are you?

If you look only at the weight, 3 kg is a little more typical of the heavy warehouse. If you look only at the traffic, warehouse 0 is the safe guess. Neither fact answers the question alone, and a rule that uses only one of them is wrong in a predictable way. You need a recipe that weighs both, and that tells you a probability and not only a name.

The earlier chapters on logistic regression learned a boundary directly: a line, and a sigmoid of the distance from it. This chapter takes the opposite route. You describe what each warehouse's parcels look like, then ask which description would more plausibly have produced this one. That is a *generative classifier*, because it models how the data are generated. The rest of this part builds on it.

## What you already know

- A probability lies between 0 and 1, and the probabilities of all the classes add to 1.
- A bell curve has a centre (its mean) and a width (its variance, the square of the standard deviation).
- The sigmoid turns a score into a probability, as in the chapter *From a score to a probability*.
- A logarithm turns a product into a sum, so $\log(ab) = \log a + \log b$.
- Logistic regression, from *Learning the weights: gradient and descent*, models the class given the point, $p(y \mid x)$, and never models what the points look like.

## Why describe each class at all

Why not keep learning the boundary directly? Often you should. But a model of each class buys things a boundary cannot give you. It can flag a point that looks like no class at all, because every class finds it unlikely. It can invent typical points by sampling from a class. And when its assumption about the shape is right, it can settle on a good answer from fewer examples, which a later chapter in this part shows with a picture.

So the plan is to learn, for each class, a rule that says how typical a given weight is for that class. That rule is a *density*, and it is easy to misread, so start there.

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

A *density* is the height of such a curve at a weight. It is not a probability. The chance that a parcel weighs exactly 3.000 kg is zero, because a scale can always read one more decimal. What has meaning is the chance of landing in a thin slice, say 3.00 to 3.01 kg, and that chance is about the height times the slice width. Both warehouses get the same slice, so the ratio of heights is the ratio of chances. That is why you can compare heights at one weight. It is also why a height can exceed 1: a very narrow bell is tall, and its area is still 1.

Where do the heights come from? A bell with centre $\mu$ and variance $\sigma^2$ has height

$$
\frac{1}{\sqrt{2\pi\sigma^2}} \, e^{-(x-\mu)^2 / (2\sigma^2)}.
$$

Read it aloud: the exponent is zero at the centre, which gives the peak, and it falls off with the squared gap, measured in units of the variance. The factor in front makes the area under the curve equal to 1. At 3 kg, warehouse 0's gap is $3 - 1.5 = 1.5$, so the exponent is $-2.25/2 = -1.125$, and the height is $0.3989 \times e^{-1.125} = 0.130$. Warehouse 1's gap is 1, the exponent is $-0.5$, and the height is $0.3989 \times 0.607 = 0.242$.

So the heights are 0.130 for warehouse 0 and 0.242 for warehouse 1. Judged on weight alone, the heavy warehouse is the better guess, by about 1.9 to 1.

Next, bring in the traffic. Nine parcels in ten come from warehouse 0, so before you weigh anything, warehouse 0 has a head start. Shrink each bell by its share of the traffic, which gives the shaded curves. Warehouse 0's shaded height at 3 kg is $0.9 \times 0.130 = 0.117$. Warehouse 1's is $0.1 \times 0.242 = 0.024$. After the shrink the order has flipped, and warehouse 0 is ahead by about 4.8 to 1.

The traffic share is the *prior*, the probability of a class before you look at the point. The shrunk height is the prior times the density. It says how much of all parcels come from this warehouse and look like this one.

## Rescale the scores into probabilities

Those two scores, 0.1166 and 0.0242, do not add to 1, so they are not probabilities yet. Every parcel at 3 kg came from one of the two warehouses, so their shares must add to 1. Divide each score by the total, 0.1408:

- Warehouse 0: $0.1166 / 0.1408 = 0.828$.
- Warehouse 1: $0.0242 / 0.1408 = 0.172$.

You can be about 83 percent sure the parcel came from the busy, light warehouse. Figure 2 shows what the division did.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Bar charts of the two unnormalised scores and the same bars rescaled to sum to one">
<text x="30" y="30" class="fig-bold">Scores: prior × density</text>
<line x1="30" y1="210" x2="270" y2="210" class="stroke-muted"/>
<rect x="60" y="85.7" width="80" height="124.3" class="fill-a"/>
<text x="100" y="79.7" text-anchor="middle" class="fig-bold">0.1166</text>
<text x="100" y="232" text-anchor="middle">warehouse 0</text>
<rect x="170" y="184.2" width="80" height="25.8" class="fill-b"/>
<text x="210" y="178.2" text-anchor="middle" class="fig-bold">0.0242</text>
<text x="210" y="232" text-anchor="middle">warehouse 1</text>
<text x="330" y="30" class="fig-bold">Divide by the total, 0.1408</text>
<line x1="330" y1="210" x2="570" y2="210" class="stroke-muted"/>
<rect x="360" y="77.5" width="80" height="132.5" class="fill-a"/>
<text x="400" y="71.5" text-anchor="middle" class="fig-bold">0.828</text>
<text x="400" y="232" text-anchor="middle">warehouse 0</text>
<rect x="470" y="182.5" width="80" height="27.5" class="fill-b"/>
<text x="510" y="176.5" text-anchor="middle" class="fig-bold">0.172</text>
<text x="510" y="232" text-anchor="middle">warehouse 1</text>
<text x="300" y="130" text-anchor="middle" class="fig-bold">→</text>
<text x="300" y="258" text-anchor="middle" class="fig-small">same winner, same ratio 4.8 to 1</text>
</svg>
Figure 2. Dividing by the total shrinks both bars by the same factor, so the winner and the ratio between the bars (4.8 to 1) do not change. Only the units change: the right-hand bars add to 1.
```

The division never changes the winner, because it shrinks both scores by the same factor. If you only need a name, the scores are enough. You need the division when you want an honest probability, to say how sure you are, or to compare against another model.

## Why the prior cannot be skipped

Why not compare the densities and ignore the traffic? Because that rule is wrong whenever the classes are unevenly common. With equal priors, the same two bells give 0.349 for warehouse 0 and 0.651 for warehouse 1, so the answer flips to the heavy warehouse. A depot that follows that rule sends nine parcels in ten to the wrong place whenever the weights are ambiguous.

Figure 3 shows the same bells under three priors. The dashed line marks the weight where the two shaded curves cross, the point where the two warehouses are equally likely.

```figure
<svg viewBox="0 0 620 250" role="img" aria-label="Three panels of the same two bells under priors 50 50, 90 10 and 10 90, with the crossing point marked in each">
<text x="15" y="28" class="fig-bold">prior 50 / 50</text>
<line x1="15" y1="190" x2="185" y2="190" class="stroke-muted"/>
<text x="15.0" y="208" text-anchor="middle" class="fig-small">0</text>
<text x="71.7" y="208" text-anchor="middle" class="fig-small">2</text>
<text x="128.3" y="208" text-anchor="middle" class="fig-small">4</text>
<text x="185.0" y="208" text-anchor="middle" class="fig-small">6</text>
<polygon points="15.0,190.0 15.0,167.3 16.9,165.0 18.8,162.6 20.7,160.0 22.6,157.4 24.4,154.7 26.3,151.9 28.2,149.1 30.1,146.2 32.0,143.4 33.9,140.7 35.8,138.0 37.7,135.4 39.6,132.9 41.4,130.5 43.3,128.4 45.2,126.4 47.1,124.7 49.0,123.3 50.9,122.1 52.8,121.1 54.7,120.5 56.6,120.2 58.4,120.2 60.3,120.5 62.2,121.1 64.1,122.1 66.0,123.3 67.9,124.7 69.8,126.4 71.7,128.4 73.6,130.5 75.4,132.9 77.3,135.4 79.2,138.0 81.1,140.7 83.0,143.4 84.9,146.2 86.8,149.1 88.7,151.9 90.6,154.7 92.4,157.4 94.3,160.0 96.2,162.6 98.1,165.0 100.0,167.3 101.9,169.5 103.8,171.6 105.7,173.5 107.6,175.3 109.4,177.0 111.3,178.5 113.2,179.9 115.1,181.2 117.0,182.3 118.9,183.3 120.8,184.2 122.7,185.0 124.6,185.8 126.4,186.4 128.3,186.9 130.2,187.4 132.1,187.8 134.0,188.2 135.9,188.5 137.8,188.7 139.7,189.0 141.6,189.1 143.4,189.3 145.3,189.4 147.2,189.5 149.1,189.6 151.0,189.7 152.9,189.8 154.8,189.8 156.7,189.8 158.6,189.9 160.4,189.9 162.3,189.9 164.2,189.9 166.1,190.0 168.0,190.0 169.9,190.0 171.8,190.0 173.7,190.0 175.6,190.0 177.4,190.0 179.3,190.0 181.2,190.0 183.1,190.0 185.0,190.0 185.0,190.0" class="fill-soft-a"/>
<polygon points="15.0,190.0 15.0,190.0 16.9,190.0 18.8,190.0 20.7,189.9 22.6,189.9 24.4,189.9 26.3,189.9 28.2,189.9 30.1,189.8 32.0,189.8 33.9,189.7 35.8,189.7 37.7,189.6 39.6,189.5 41.4,189.4 43.3,189.2 45.2,189.1 47.1,188.9 49.0,188.6 50.9,188.3 52.8,188.0 54.7,187.6 56.6,187.2 58.4,186.7 60.3,186.1 62.2,185.4 64.1,184.7 66.0,183.8 67.9,182.8 69.8,181.7 71.7,180.6 73.6,179.2 75.4,177.8 77.3,176.2 79.2,174.5 81.1,172.6 83.0,170.6 84.9,168.5 86.8,166.2 88.7,163.8 90.6,161.3 92.4,158.7 94.3,156.0 96.2,153.3 98.1,150.5 100.0,147.7 101.9,144.8 103.8,142.0 105.7,139.3 107.6,136.6 109.4,134.1 111.3,131.7 113.2,129.4 115.1,127.4 117.0,125.6 118.9,124.0 120.8,122.6 122.7,121.6 124.6,120.8 126.4,120.3 128.3,120.2 130.2,120.3 132.1,120.8 134.0,121.6 135.9,122.6 137.8,124.0 139.7,125.6 141.6,127.4 143.4,129.4 145.3,131.7 147.2,134.1 149.1,136.6 151.0,139.3 152.9,142.0 154.8,144.8 156.7,147.7 158.6,150.5 160.4,153.3 162.3,156.0 164.2,158.7 166.1,161.3 168.0,163.8 169.9,166.2 171.8,168.5 173.7,170.6 175.6,172.6 177.4,174.5 179.3,176.2 181.2,177.8 183.1,179.2 185.0,180.6 185.0,190.0" class="fill-soft-b"/>
<polyline points="15.0,167.3 16.9,165.0 18.8,162.6 20.7,160.0 22.6,157.4 24.4,154.7 26.3,151.9 28.2,149.1 30.1,146.2 32.0,143.4 33.9,140.7 35.8,138.0 37.7,135.4 39.6,132.9 41.4,130.5 43.3,128.4 45.2,126.4 47.1,124.7 49.0,123.3 50.9,122.1 52.8,121.1 54.7,120.5 56.6,120.2 58.4,120.2 60.3,120.5 62.2,121.1 64.1,122.1 66.0,123.3 67.9,124.7 69.8,126.4 71.7,128.4 73.6,130.5 75.4,132.9 77.3,135.4 79.2,138.0 81.1,140.7 83.0,143.4 84.9,146.2 86.8,149.1 88.7,151.9 90.6,154.7 92.4,157.4 94.3,160.0 96.2,162.6 98.1,165.0 100.0,167.3 101.9,169.5 103.8,171.6 105.7,173.5 107.6,175.3 109.4,177.0 111.3,178.5 113.2,179.9 115.1,181.2 117.0,182.3 118.9,183.3 120.8,184.2 122.7,185.0 124.6,185.8 126.4,186.4 128.3,186.9 130.2,187.4 132.1,187.8 134.0,188.2 135.9,188.5 137.8,188.7 139.7,189.0 141.6,189.1 143.4,189.3 145.3,189.4 147.2,189.5 149.1,189.6 151.0,189.7 152.9,189.8 154.8,189.8 156.7,189.8 158.6,189.9 160.4,189.9 162.3,189.9 164.2,189.9 166.1,190.0 168.0,190.0 169.9,190.0 171.8,190.0 173.7,190.0 175.6,190.0 177.4,190.0 179.3,190.0 181.2,190.0 183.1,190.0 185.0,190.0" fill="none" class="stroke-a"/>
<polyline points="15.0,190.0 16.9,190.0 18.8,190.0 20.7,189.9 22.6,189.9 24.4,189.9 26.3,189.9 28.2,189.9 30.1,189.8 32.0,189.8 33.9,189.7 35.8,189.7 37.7,189.6 39.6,189.5 41.4,189.4 43.3,189.2 45.2,189.1 47.1,188.9 49.0,188.6 50.9,188.3 52.8,188.0 54.7,187.6 56.6,187.2 58.4,186.7 60.3,186.1 62.2,185.4 64.1,184.7 66.0,183.8 67.9,182.8 69.8,181.7 71.7,180.6 73.6,179.2 75.4,177.8 77.3,176.2 79.2,174.5 81.1,172.6 83.0,170.6 84.9,168.5 86.8,166.2 88.7,163.8 90.6,161.3 92.4,158.7 94.3,156.0 96.2,153.3 98.1,150.5 100.0,147.7 101.9,144.8 103.8,142.0 105.7,139.3 107.6,136.6 109.4,134.1 111.3,131.7 113.2,129.4 115.1,127.4 117.0,125.6 118.9,124.0 120.8,122.6 122.7,121.6 124.6,120.8 126.4,120.3 128.3,120.2 130.2,120.3 132.1,120.8 134.0,121.6 135.9,122.6 137.8,124.0 139.7,125.6 141.6,127.4 143.4,129.4 145.3,131.7 147.2,134.1 149.1,136.6 151.0,139.3 152.9,142.0 154.8,144.8 156.7,147.7 158.6,150.5 160.4,153.3 162.3,156.0 164.2,158.7 166.1,161.3 168.0,163.8 169.9,166.2 171.8,168.5 173.7,170.6 175.6,172.6 177.4,174.5 179.3,176.2 181.2,177.8 183.1,179.2 185.0,180.6" fill="none" class="stroke-b"/>
<line x1="92.9" y1="190" x2="92.9" y2="62" class="stroke-ink dashed"/>
<text x="92.9" y="56" text-anchor="middle" class="fig-bold">x = 2.75</text>
<text x="100" y="232" text-anchor="middle" class="fig-small">weight (kg)</text>
<text x="215" y="28" class="fig-bold">prior 90 / 10</text>
<line x1="215" y1="190" x2="385" y2="190" class="stroke-muted"/>
<text x="215.0" y="208" text-anchor="middle" class="fig-small">0</text>
<text x="271.7" y="208" text-anchor="middle" class="fig-small">2</text>
<text x="328.3" y="208" text-anchor="middle" class="fig-small">4</text>
<text x="385.0" y="208" text-anchor="middle" class="fig-small">6</text>
<polygon points="215.0,190.0 215.0,149.2 216.9,145.0 218.8,140.6 220.7,136.0 222.6,131.3 224.4,126.4 226.3,121.4 228.2,116.3 230.1,111.2 232.0,106.2 233.9,101.2 235.8,96.3 237.7,91.6 239.6,87.2 241.4,83.0 243.3,79.1 245.2,75.6 247.1,72.5 249.0,69.9 250.9,67.7 252.8,66.1 254.7,65.0 256.6,64.4 258.4,64.4 260.3,65.0 262.2,66.1 264.1,67.7 266.0,69.9 267.9,72.5 269.8,75.6 271.7,79.1 273.6,83.0 275.4,87.2 277.3,91.6 279.2,96.3 281.1,101.2 283.0,106.2 284.9,111.2 286.8,116.3 288.7,121.4 290.6,126.4 292.4,131.3 294.3,136.0 296.2,140.6 298.1,145.0 300.0,149.2 301.9,153.2 303.8,156.9 305.7,160.4 307.6,163.6 309.4,166.6 311.3,169.3 313.2,171.8 315.1,174.1 317.0,176.1 318.9,178.0 320.8,179.6 322.7,181.1 324.6,182.4 326.4,183.5 328.3,184.5 330.2,185.3 332.1,186.1 334.0,186.7 335.9,187.3 337.8,187.7 339.7,188.1 341.6,188.5 343.4,188.7 345.3,189.0 347.2,189.2 349.1,189.3 351.0,189.5 352.9,189.6 354.8,189.7 356.7,189.7 358.6,189.8 360.4,189.8 362.3,189.9 364.2,189.9 366.1,189.9 368.0,189.9 369.9,190.0 371.8,190.0 373.7,190.0 375.6,190.0 377.4,190.0 379.3,190.0 381.2,190.0 383.1,190.0 385.0,190.0 385.0,190.0" class="fill-soft-a"/>
<polygon points="215.0,190.0 215.0,190.0 216.9,190.0 218.8,190.0 220.7,190.0 222.6,190.0 224.4,190.0 226.3,190.0 228.2,190.0 230.1,190.0 232.0,190.0 233.9,189.9 235.8,189.9 237.7,189.9 239.6,189.9 241.4,189.9 243.3,189.8 245.2,189.8 247.1,189.8 249.0,189.7 250.9,189.7 252.8,189.6 254.7,189.5 256.6,189.4 258.4,189.3 260.3,189.2 262.2,189.1 264.1,188.9 266.0,188.8 267.9,188.6 269.8,188.3 271.7,188.1 273.6,187.8 275.4,187.6 277.3,187.2 279.2,186.9 281.1,186.5 283.0,186.1 284.9,185.7 286.8,185.2 288.7,184.8 290.6,184.3 292.4,183.7 294.3,183.2 296.2,182.7 298.1,182.1 300.0,181.5 301.9,181.0 303.8,180.4 305.7,179.9 307.6,179.3 309.4,178.8 311.3,178.3 313.2,177.9 315.1,177.5 317.0,177.1 318.9,176.8 320.8,176.5 322.7,176.3 324.6,176.2 326.4,176.1 328.3,176.0 330.2,176.1 332.1,176.2 334.0,176.3 335.9,176.5 337.8,176.8 339.7,177.1 341.6,177.5 343.4,177.9 345.3,178.3 347.2,178.8 349.1,179.3 351.0,179.9 352.9,180.4 354.8,181.0 356.7,181.5 358.6,182.1 360.4,182.7 362.3,183.2 364.2,183.7 366.1,184.3 368.0,184.8 369.9,185.2 371.8,185.7 373.7,186.1 375.6,186.5 377.4,186.9 379.3,187.2 381.2,187.6 383.1,187.8 385.0,188.1 385.0,190.0" class="fill-soft-b"/>
<polyline points="215.0,149.2 216.9,145.0 218.8,140.6 220.7,136.0 222.6,131.3 224.4,126.4 226.3,121.4 228.2,116.3 230.1,111.2 232.0,106.2 233.9,101.2 235.8,96.3 237.7,91.6 239.6,87.2 241.4,83.0 243.3,79.1 245.2,75.6 247.1,72.5 249.0,69.9 250.9,67.7 252.8,66.1 254.7,65.0 256.6,64.4 258.4,64.4 260.3,65.0 262.2,66.1 264.1,67.7 266.0,69.9 267.9,72.5 269.8,75.6 271.7,79.1 273.6,83.0 275.4,87.2 277.3,91.6 279.2,96.3 281.1,101.2 283.0,106.2 284.9,111.2 286.8,116.3 288.7,121.4 290.6,126.4 292.4,131.3 294.3,136.0 296.2,140.6 298.1,145.0 300.0,149.2 301.9,153.2 303.8,156.9 305.7,160.4 307.6,163.6 309.4,166.6 311.3,169.3 313.2,171.8 315.1,174.1 317.0,176.1 318.9,178.0 320.8,179.6 322.7,181.1 324.6,182.4 326.4,183.5 328.3,184.5 330.2,185.3 332.1,186.1 334.0,186.7 335.9,187.3 337.8,187.7 339.7,188.1 341.6,188.5 343.4,188.7 345.3,189.0 347.2,189.2 349.1,189.3 351.0,189.5 352.9,189.6 354.8,189.7 356.7,189.7 358.6,189.8 360.4,189.8 362.3,189.9 364.2,189.9 366.1,189.9 368.0,189.9 369.9,190.0 371.8,190.0 373.7,190.0 375.6,190.0 377.4,190.0 379.3,190.0 381.2,190.0 383.1,190.0 385.0,190.0" fill="none" class="stroke-a"/>
<polyline points="215.0,190.0 216.9,190.0 218.8,190.0 220.7,190.0 222.6,190.0 224.4,190.0 226.3,190.0 228.2,190.0 230.1,190.0 232.0,190.0 233.9,189.9 235.8,189.9 237.7,189.9 239.6,189.9 241.4,189.9 243.3,189.8 245.2,189.8 247.1,189.8 249.0,189.7 250.9,189.7 252.8,189.6 254.7,189.5 256.6,189.4 258.4,189.3 260.3,189.2 262.2,189.1 264.1,188.9 266.0,188.8 267.9,188.6 269.8,188.3 271.7,188.1 273.6,187.8 275.4,187.6 277.3,187.2 279.2,186.9 281.1,186.5 283.0,186.1 284.9,185.7 286.8,185.2 288.7,184.8 290.6,184.3 292.4,183.7 294.3,183.2 296.2,182.7 298.1,182.1 300.0,181.5 301.9,181.0 303.8,180.4 305.7,179.9 307.6,179.3 309.4,178.8 311.3,178.3 313.2,177.9 315.1,177.5 317.0,177.1 318.9,176.8 320.8,176.5 322.7,176.3 324.6,176.2 326.4,176.1 328.3,176.0 330.2,176.1 332.1,176.2 334.0,176.3 335.9,176.5 337.8,176.8 339.7,177.1 341.6,177.5 343.4,177.9 345.3,178.3 347.2,178.8 349.1,179.3 351.0,179.9 352.9,180.4 354.8,181.0 356.7,181.5 358.6,182.1 360.4,182.7 362.3,183.2 364.2,183.7 366.1,184.3 368.0,184.8 369.9,185.2 371.8,185.7 373.7,186.1 375.6,186.5 377.4,186.9 379.3,187.2 381.2,187.6 383.1,187.8 385.0,188.1" fill="none" class="stroke-b"/>
<line x1="317.8" y1="190" x2="317.8" y2="62" class="stroke-ink dashed"/>
<text x="317.8" y="56" text-anchor="middle" class="fig-bold">x = 3.63</text>
<text x="300" y="232" text-anchor="middle" class="fig-small">weight (kg)</text>
<text x="415" y="28" class="fig-bold">prior 10 / 90</text>
<line x1="415" y1="190" x2="585" y2="190" class="stroke-muted"/>
<text x="415.0" y="208" text-anchor="middle" class="fig-small">0</text>
<text x="471.7" y="208" text-anchor="middle" class="fig-small">2</text>
<text x="528.3" y="208" text-anchor="middle" class="fig-small">4</text>
<text x="585.0" y="208" text-anchor="middle" class="fig-small">6</text>
<polygon points="415.0,190.0 415.0,185.5 416.9,185.0 418.8,184.5 420.7,184.0 422.6,183.5 424.4,182.9 426.3,182.4 428.2,181.8 430.1,181.2 432.0,180.7 433.9,180.1 435.8,179.6 437.7,179.1 439.6,178.6 441.4,178.1 443.3,177.7 445.2,177.3 447.1,176.9 449.0,176.7 450.9,176.4 452.8,176.2 454.7,176.1 456.6,176.0 458.4,176.0 460.3,176.1 462.2,176.2 464.1,176.4 466.0,176.7 467.9,176.9 469.8,177.3 471.7,177.7 473.6,178.1 475.4,178.6 477.3,179.1 479.2,179.6 481.1,180.1 483.0,180.7 484.9,181.2 486.8,181.8 488.7,182.4 490.6,182.9 492.4,183.5 494.3,184.0 496.2,184.5 498.1,185.0 500.0,185.5 501.9,185.9 503.8,186.3 505.7,186.7 507.6,187.1 509.4,187.4 511.3,187.7 513.2,188.0 515.1,188.2 517.0,188.5 518.9,188.7 520.8,188.8 522.7,189.0 524.6,189.2 526.4,189.3 528.3,189.4 530.2,189.5 532.1,189.6 534.0,189.6 535.9,189.7 537.8,189.7 539.7,189.8 541.6,189.8 543.4,189.9 545.3,189.9 547.2,189.9 549.1,189.9 551.0,189.9 552.9,190.0 554.8,190.0 556.7,190.0 558.6,190.0 560.4,190.0 562.3,190.0 564.2,190.0 566.1,190.0 568.0,190.0 569.9,190.0 571.8,190.0 573.7,190.0 575.6,190.0 577.4,190.0 579.3,190.0 581.2,190.0 583.1,190.0 585.0,190.0 585.0,190.0" class="fill-soft-a"/>
<polygon points="415.0,190.0 415.0,190.0 416.9,189.9 418.8,189.9 420.7,189.9 422.6,189.9 424.4,189.8 426.3,189.8 428.2,189.8 430.1,189.7 432.0,189.6 433.9,189.5 435.8,189.4 437.7,189.2 439.6,189.1 441.4,188.9 443.3,188.6 445.2,188.3 447.1,187.9 449.0,187.5 450.9,187.0 452.8,186.4 454.7,185.7 456.6,184.9 458.4,184.0 460.3,182.9 462.2,181.7 464.1,180.4 466.0,178.8 467.9,177.1 469.8,175.1 471.7,173.0 473.6,170.6 475.4,168.0 477.3,165.1 479.2,162.0 481.1,158.7 483.0,155.1 484.9,151.2 486.8,147.1 488.7,142.8 490.6,138.3 492.4,133.7 494.3,128.8 496.2,123.9 498.1,118.9 500.0,113.8 501.9,108.7 503.8,103.7 505.7,98.7 507.6,94.0 509.4,89.4 511.3,85.0 513.2,81.0 515.1,77.3 517.0,74.0 518.9,71.1 520.8,68.7 522.7,66.8 524.6,65.4 526.4,64.6 528.3,64.3 530.2,64.6 532.1,65.4 534.0,66.8 535.9,68.7 537.8,71.1 539.7,74.0 541.6,77.3 543.4,81.0 545.3,85.0 547.2,89.4 549.1,94.0 551.0,98.7 552.9,103.7 554.8,108.7 556.7,113.8 558.6,118.9 560.4,123.9 562.3,128.8 564.2,133.7 566.1,138.3 568.0,142.8 569.9,147.1 571.8,151.2 573.7,155.1 575.6,158.7 577.4,162.0 579.3,165.1 581.2,168.0 583.1,170.6 585.0,173.0 585.0,190.0" class="fill-soft-b"/>
<polyline points="415.0,185.5 416.9,185.0 418.8,184.5 420.7,184.0 422.6,183.5 424.4,182.9 426.3,182.4 428.2,181.8 430.1,181.2 432.0,180.7 433.9,180.1 435.8,179.6 437.7,179.1 439.6,178.6 441.4,178.1 443.3,177.7 445.2,177.3 447.1,176.9 449.0,176.7 450.9,176.4 452.8,176.2 454.7,176.1 456.6,176.0 458.4,176.0 460.3,176.1 462.2,176.2 464.1,176.4 466.0,176.7 467.9,176.9 469.8,177.3 471.7,177.7 473.6,178.1 475.4,178.6 477.3,179.1 479.2,179.6 481.1,180.1 483.0,180.7 484.9,181.2 486.8,181.8 488.7,182.4 490.6,182.9 492.4,183.5 494.3,184.0 496.2,184.5 498.1,185.0 500.0,185.5 501.9,185.9 503.8,186.3 505.7,186.7 507.6,187.1 509.4,187.4 511.3,187.7 513.2,188.0 515.1,188.2 517.0,188.5 518.9,188.7 520.8,188.8 522.7,189.0 524.6,189.2 526.4,189.3 528.3,189.4 530.2,189.5 532.1,189.6 534.0,189.6 535.9,189.7 537.8,189.7 539.7,189.8 541.6,189.8 543.4,189.9 545.3,189.9 547.2,189.9 549.1,189.9 551.0,189.9 552.9,190.0 554.8,190.0 556.7,190.0 558.6,190.0 560.4,190.0 562.3,190.0 564.2,190.0 566.1,190.0 568.0,190.0 569.9,190.0 571.8,190.0 573.7,190.0 575.6,190.0 577.4,190.0 579.3,190.0 581.2,190.0 583.1,190.0 585.0,190.0" fill="none" class="stroke-a"/>
<polyline points="415.0,190.0 416.9,189.9 418.8,189.9 420.7,189.9 422.6,189.9 424.4,189.8 426.3,189.8 428.2,189.8 430.1,189.7 432.0,189.6 433.9,189.5 435.8,189.4 437.7,189.2 439.6,189.1 441.4,188.9 443.3,188.6 445.2,188.3 447.1,187.9 449.0,187.5 450.9,187.0 452.8,186.4 454.7,185.7 456.6,184.9 458.4,184.0 460.3,182.9 462.2,181.7 464.1,180.4 466.0,178.8 467.9,177.1 469.8,175.1 471.7,173.0 473.6,170.6 475.4,168.0 477.3,165.1 479.2,162.0 481.1,158.7 483.0,155.1 484.9,151.2 486.8,147.1 488.7,142.8 490.6,138.3 492.4,133.7 494.3,128.8 496.2,123.9 498.1,118.9 500.0,113.8 501.9,108.7 503.8,103.7 505.7,98.7 507.6,94.0 509.4,89.4 511.3,85.0 513.2,81.0 515.1,77.3 517.0,74.0 518.9,71.1 520.8,68.7 522.7,66.8 524.6,65.4 526.4,64.6 528.3,64.3 530.2,64.6 532.1,65.4 534.0,66.8 535.9,68.7 537.8,71.1 539.7,74.0 541.6,77.3 543.4,81.0 545.3,85.0 547.2,89.4 549.1,94.0 551.0,98.7 552.9,103.7 554.8,108.7 556.7,113.8 558.6,118.9 560.4,123.9 562.3,128.8 564.2,133.7 566.1,138.3 568.0,142.8 569.9,147.1 571.8,151.2 573.7,155.1 575.6,158.7 577.4,162.0 579.3,165.1 581.2,168.0 583.1,170.6 585.0,173.0" fill="none" class="stroke-b"/>
<line x1="468.0" y1="190" x2="468.0" y2="62" class="stroke-ink dashed"/>
<text x="468.0" y="56" text-anchor="middle" class="fig-bold">x = 1.87</text>
<text x="500" y="232" text-anchor="middle" class="fig-small">weight (kg)</text>
</svg>
Figure 3. The same two bells under three priors. The dashed line marks where the shaded curves cross, the weight at which the two warehouses are equally likely. It sits at 2.75 kg when the priors are equal, and slides toward the rarer warehouse as the priors tilt.
```

With equal priors the crossing sits at 2.75 kg, halfway between the means. When warehouse 0 is nine times as busy, the crossing moves to 3.63 kg: a parcel must look quite heavy before the rare warehouse becomes the better bet. Swap the priors and it slides to 1.87 kg. The boundary moves toward the rarer class, because the rare class needs stronger evidence.

## The formula

You have watched the recipe work, so the formula is a summary of it. For class $k$, write $\pi_k$ for its prior (the Greek letter pi, used as a name for that share) and $p(x \mid y = k)$ for its density at the point $x$. Then

$$
p(y = k \mid x) = \frac{\pi_k \, p(x \mid y = k)}{\sum_j \pi_j \, p(x \mid y = j)} .
$$

Read it aloud: the probability of class $k$ given the point is that class's prior times density, divided by the same quantity added up over every class. The top is the shrunk height. The bottom is the total, 0.1408 in the example. This is *Bayes' rule*, and the left side, $p(y = k \mid x)$, is the *posterior*, the probability after you have seen the point.

## Work in logs

There is one practical problem. A real density is often tiny. A document with a thousand words multiplies a thousand small numbers, and a computer stores numbers only down to about $10^{-308}$, so the product silently becomes 0 and the division becomes 0 divided by 0. The fix is to keep every score as a logarithm, where a product becomes a sum and a tiny number becomes a moderate negative one. The log score for class $k$ is $\log \pi_k + \log p(x \mid y = k)$. In the example, those are $-2.149$ and $-3.722$.

Dividing by the total then needs the log of a sum of exponentials, which has a name, the *log-sum-exp*:

$$
\log \sum_j e^{a_j}, \qquad \text{where } a_j = \log \pi_j + \log p(x \mid y = j).
$$

Computing it directly brings the underflow back, because $e^{-1000}$ is 0. The trick is to subtract the largest score $m$ first. Since $\sum_j e^{a_j} = e^{m}\sum_j e^{a_j - m}$, you can write

$$
\log \sum_j e^{a_j} = m + \log \sum_j e^{a_j - m}.
$$

The largest term becomes $e^{0} = 1$, so the sum is at least 1 and nothing underflows to zero. Take two scores, $-1000$ and $-1001$. Directly, $e^{-1000}$ and $e^{-1001}$ are both 0. With the trick, $m = -1000$, the sum is $1 + e^{-1} = 1.368$, and the log-sum-exp is $-1000 + \log 1.368 = -999.687$. The posteriors are $1 / 1.368 = 0.731$ and $0.368 / 1.368 = 0.269$.

The log posterior is then the log score minus the log-sum-exp, and `np.exp` of it adds to 1 along each row. In numpy, with `log_density` of shape `(n, K)`:

```python
joint = log_density + np.log(priors)             # shape (n, K)
log_post = joint - log_sum_exp(joint, axis=1)[:, None]
```

> [!WARNING]
> Normalise across the classes, which is `axis=1` when each row is one point. Using `axis=0` makes each class's column add to 1 instead, and the posteriors look plausible but are wrong. Check that `np.exp(log_post).sum(axis=1)` is 1 for every row.

> [!KEY]
> Back to the parcel. A generative classifier scores each warehouse by its prior times how typical 3 kg is for it, so the busy warehouse beats the slightly more typical heavy one, 83 percent to 17 percent. Dividing by the total only turns scores into probabilities, and working in logs with log-sum-exp keeps every step safe from underflow.

> [!RECAP]
> - A generative classifier models what each class looks like, then asks which class would more plausibly have produced the point.
> - A density is a height, not a probability, and heights are comparable at the same point.
> - The posterior is proportional to prior times density, and ignoring the prior fails when classes are uneven.
> - Dividing by the total only rescales the scores to add to 1; it never changes the winner.
> - Keep scores as logs, and use log-sum-exp with the maximum subtracted.

## Build it step by step

You write three small functions, and a fourth, `predict_class`, is given.

1. **`normal_log_density_1d(x, mean, var)`**. Takes an array `x` and a scalar mean and variance, and returns the log of the Gaussian height at each point, with the shape of `x`. The log of the bell is $-\tfrac12 \log(2\pi\sigma^2) - (x-\mu)^2 / (2\sigma^2)$. The usual mistake is passing the standard deviation where the variance goes.
2. **`log_sum_exp(a, axis)`**. Takes an array and an axis, and returns the log of the sum of exponentials along that axis, with the axis removed. Subtract `np.max(a, axis=axis, keepdims=True)` first and add it back after the log. The usual mistake is subtracting a single global maximum, which helps one row and starves another.
3. **`log_posterior_from_log_density(log_density, priors)`**. Takes `(n, K)` log densities and `(K,)` priors, and returns `(n, K)` log posteriors. Add `np.log(priors)`, then subtract a row-wise log-sum-exp kept as a column. The usual mistake is the wrong axis, covered in the warning above.

## The exercise

You implement `normal_log_density_1d`, `log_sum_exp`, and `log_posterior_from_log_density` in `starter.py`. The given `predict_class` takes the largest log posterior in each row.

The checks compare your density with scipy and confirm that a narrow bell has a positive log density. They test `log_sum_exp` along both axes and at values near $\pm 1000$. They reproduce the 0.828 and 0.172 posterior, confirm that equal densities return the prior, and verify that a 90/10 prior flips the winner at 3 kg.
