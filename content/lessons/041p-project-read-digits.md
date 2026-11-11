---
title: "Train a digit reader from scratch"
slug: 041p-project-read-digits
part: "Part 4 — Neural networks"
summary: Build, check, and train a regularised two-hidden-layer network in numpy that reads noisy 10 by 10 digits and beats a linear baseline.
minutes: 180
gpu: false
kind: project
objectives:
  - Assemble initialisation, a forward pass, a smoothed loss, backprop, AdamW, a schedule, dropout, and early stopping into one training loop.
  - Prove the whole network's gradients with a finite-difference check before trusting any training curve.
  - Switch dropout off at evaluation and keep the parameters from the best validation epoch.
  - Judge a network against a linear baseline built from the same code.
prereqs:
  - 033-multilayer-perceptrons
  - 035-softmax-and-whole-network-gradients
  - 036-sgd-adam-and-schedules
  - 037-signal-size-and-initialisation
  - 039-weight-decay-and-early-stopping
  - 040-dropout-and-label-smoothing
  - 041-debugging-and-tuning
exercise: 041p-project-read-digits
---

# Train a digit reader from scratch

A postal sorting startup is putting a camera over a conveyor belt. The camera crops one handwritten digit from each parcel's postcode and hands your code a tiny, 10 by 10 grey image. The crops are bad: the digit sits a pixel off-centre, the pen was thick or thin, and the sensor adds grain. The belt moves, so the reader has to be fast and has to be right on digits it has never seen.

The obvious first try is a linear model: one weight per pixel for each of the ten digits, then a softmax. It is quick to train and it gets most digits right. It still misreads about one digit in ten, and at a postal sorter's volume that is thousands of misrouted parcels a day. The reason is structural: a digit shifted by one pixel lights up a different set of pixels, and no single weight per pixel can recognise "a 4, wherever it landed". A hidden layer can.

Your job is to build the better reader yourself, in numpy, with no framework. Every part of Part 4 has a role: starting weights that keep the signal healthy, a forward and backward pass you can trust, an optimiser and schedule that get to a good answer in a few seconds, and three guards (dropout, label smoothing, early stopping) against memorising 1,500 training images.

## The brief

Deliver a module that does the following:

- Builds a network with two hidden layers and trains it with mini-batch AdamW.
- Passes a finite-difference gradient check on the whole network, with label smoothing and with dropout.
- Stops training early on a validation set and returns the parameters from the best epoch.
- Predicts with dropout off.
- Trains in a few seconds.

The work is judged by the checks in `tests.py`. A good answer reads at least 95.5% of 1,000 unseen test digits correctly, beats a linear softmax baseline trained with the same code by at least five percentage points, and finishes training in under eight seconds. The reference solution reads about 98% and trains in well under a second.

## What you will use

- [A forward pass through stacked layers with a ReLU](/lessons/033-multilayer-perceptrons) gives you the shape of the network.
- [Softmax and cross-entropy](/lessons/035-softmax-and-whole-network-gradients) gives you a loss whose gradient on the scores is a clean subtraction.
- [Backprop through a whole network and the gradient check](/lessons/035-softmax-and-whole-network-gradients) gives you the backward pass and the test that proves it.
- [Mini-batches](/lessons/036-sgd-adam-and-schedules) turn one expensive gradient into many cheap, noisy ones.
- [Adam and AdamW](/lessons/036-sgd-adam-and-schedules) give each weight its own step size and decay the weights without bending the gradient.
- [Warmup and cosine decay](/lessons/036-sgd-adam-and-schedules) set the learning rate at every step.
- [He initialisation](/lessons/037-signal-size-and-initialisation) sets the starting weights so a ReLU network's signal keeps its size.
- [Early stopping](/lessons/039-weight-decay-and-early-stopping) halts training at the best validation loss.
- [Dropout](/lessons/040-dropout-and-label-smoothing) switches hidden units off at random, in training only.
- [Label smoothing](/lessons/040-dropout-and-label-smoothing) asks the network for slightly less certainty.
- [Debugging and tuning](/lessons/041-debugging-and-tuning) gives you the order of checks: predict the untrained loss, memorise a small batch, then read the training and validation losses before you change the model. The update-ratio rule in that chapter is for plain SGD. This project steps with AdamW, so you do not look for a ratio of $10^{-3}$.

## Meet the data

`make_data(seed)` draws every image from code, so nothing is downloaded. It holds ten stroke templates, one per digit, written as a handful of corner points in a unit square. For each image the renderer does four things to a template:

1. Moves every corner by a small random amount, so no two 4s are the same shape.
2. Shifts the whole digit by -1, 0, or +1 pixels in each direction.
3. Draws the strokes with a random thickness between 0.07 and 0.15 of the image width.
4. Adds Gaussian noise with standard deviation 0.25 to every pixel.

Each image is a flat vector of 100 numbers, and each label is an integer from 0 to 9. The function returns three disjoint sets: 1,500 training images, 500 validation images, and 1,000 test images, with the digits balanced in each. Figure 1 shows what the network sees.

```figure
<svg viewBox="0 0 600 205" role="img" aria-label="Two rows of ten-by-ten pixel glyphs: one noisy draw of each digit from 0 to 9, and ten different noisy draws of the digit 7.">
<text x="30" y="20" class="fig-bold">One noisy draw of each digit</text>
<rect x="30" y="30" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M60 30h5v5h-5zM35 35h5v5h-5zM30 40h5v5h-5zM55 40h5v5h-5zM50 45h5v5h-5zM55 45h5v5h-5zM65 45h5v5h-5zM30 50h5v5h-5zM60 50h5v5h-5zM55 55h5v5h-5zM50 60h5v5h-5zM60 60h5v5h-5zM65 60h5v5h-5zM30 75h5v5h-5zM35 75h5v5h-5zM40 75h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M40 30h5v5h-5zM75 30h5v5h-5zM30 45h5v5h-5zM35 45h5v5h-5zM45 50h5v5h-5zM65 50h5v5h-5zM70 50h5v5h-5zM30 55h5v5h-5zM35 55h5v5h-5zM50 55h5v5h-5zM60 65h5v5h-5zM70 65h5v5h-5zM75 65h5v5h-5zM65 75h5v5h-5zM75 75h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M65 30h5v5h-5zM40 35h5v5h-5zM45 35h5v5h-5zM55 35h5v5h-5zM70 35h5v5h-5zM35 40h5v5h-5zM40 40h5v5h-5zM65 40h5v5h-5zM70 40h5v5h-5zM75 45h5v5h-5zM35 50h5v5h-5zM75 50h5v5h-5zM45 55h5v5h-5zM70 55h5v5h-5zM35 60h5v5h-5zM45 60h5v5h-5zM35 65h5v5h-5zM50 65h5v5h-5zM40 70h5v5h-5zM50 70h5v5h-5zM55 70h5v5h-5zM75 70h5v5h-5zM55 75h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M45 30h5v5h-5zM50 30h5v5h-5zM55 30h5v5h-5zM70 30h5v5h-5zM50 35h5v5h-5zM60 35h5v5h-5zM65 35h5v5h-5zM75 35h5v5h-5zM45 40h5v5h-5zM50 40h5v5h-5zM60 40h5v5h-5zM75 40h5v5h-5zM40 45h5v5h-5zM45 45h5v5h-5zM70 45h5v5h-5zM40 50h5v5h-5zM40 55h5v5h-5zM75 55h5v5h-5zM40 60h5v5h-5zM70 60h5v5h-5zM75 60h5v5h-5zM40 65h5v5h-5zM45 65h5v5h-5zM55 65h5v5h-5zM65 65h5v5h-5zM45 70h5v5h-5zM60 70h5v5h-5zM65 70h5v5h-5zM70 70h5v5h-5zM45 75h5v5h-5zM50 75h5v5h-5zM60 75h5v5h-5zM70 75h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="84" y="30" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M89 30h5v5h-5zM99 30h5v5h-5zM94 35h5v5h-5zM124 35h5v5h-5zM84 40h5v5h-5zM109 45h5v5h-5zM129 45h5v5h-5zM124 50h5v5h-5zM104 55h5v5h-5zM124 55h5v5h-5zM129 55h5v5h-5zM99 60h5v5h-5zM104 60h5v5h-5zM119 60h5v5h-5zM124 60h5v5h-5zM129 60h5v5h-5zM99 65h5v5h-5zM119 65h5v5h-5zM84 70h5v5h-5zM99 70h5v5h-5zM109 70h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M119 30h5v5h-5zM104 35h5v5h-5zM99 40h5v5h-5zM84 45h5v5h-5zM99 45h5v5h-5zM109 50h5v5h-5zM109 55h5v5h-5zM109 60h5v5h-5zM89 65h5v5h-5zM109 75h5v5h-5zM114 75h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M109 35h5v5h-5zM114 35h5v5h-5zM104 40h5v5h-5zM109 40h5v5h-5zM114 50h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M114 30h5v5h-5zM114 40h5v5h-5zM104 45h5v5h-5zM114 45h5v5h-5zM114 55h5v5h-5zM114 60h5v5h-5zM109 65h5v5h-5zM114 65h5v5h-5zM114 70h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="138" y="30" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M143 30h5v5h-5zM148 30h5v5h-5zM158 30h5v5h-5zM178 30h5v5h-5zM143 35h5v5h-5zM153 40h5v5h-5zM168 40h5v5h-5zM173 45h5v5h-5zM138 50h5v5h-5zM183 50h5v5h-5zM138 55h5v5h-5zM158 55h5v5h-5zM148 60h5v5h-5zM168 60h5v5h-5zM173 60h5v5h-5zM158 65h5v5h-5zM173 65h5v5h-5zM183 70h5v5h-5zM148 75h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M173 40h5v5h-5zM143 45h5v5h-5zM153 50h5v5h-5zM143 60h5v5h-5zM143 65h5v5h-5zM148 65h5v5h-5zM163 65h5v5h-5zM178 65h5v5h-5zM158 75h5v5h-5zM168 75h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M178 35h5v5h-5zM148 40h5v5h-5zM183 40h5v5h-5zM153 60h5v5h-5zM163 60h5v5h-5zM153 70h5v5h-5zM178 70h5v5h-5zM163 75h5v5h-5zM173 75h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M148 35h5v5h-5zM153 35h5v5h-5zM158 35h5v5h-5zM163 35h5v5h-5zM168 35h5v5h-5zM173 35h5v5h-5zM143 40h5v5h-5zM178 40h5v5h-5zM178 45h5v5h-5zM173 50h5v5h-5zM178 50h5v5h-5zM163 55h5v5h-5zM168 55h5v5h-5zM173 55h5v5h-5zM158 60h5v5h-5zM153 65h5v5h-5zM143 70h5v5h-5zM148 70h5v5h-5zM158 70h5v5h-5zM163 70h5v5h-5zM168 70h5v5h-5zM173 70h5v5h-5zM178 75h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="192" y="30" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M197 30h5v5h-5zM227 30h5v5h-5zM192 35h5v5h-5zM197 40h5v5h-5zM192 45h5v5h-5zM217 45h5v5h-5zM232 50h5v5h-5zM237 50h5v5h-5zM202 55h5v5h-5zM207 55h5v5h-5zM212 55h5v5h-5zM217 55h5v5h-5zM217 60h5v5h-5zM227 65h5v5h-5zM197 70h5v5h-5zM197 75h5v5h-5zM202 75h5v5h-5zM217 75h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M217 30h5v5h-5zM207 35h5v5h-5zM237 35h5v5h-5zM202 40h5v5h-5zM207 40h5v5h-5zM237 45h5v5h-5zM192 60h5v5h-5zM212 65h5v5h-5zM222 65h5v5h-5zM237 70h5v5h-5zM212 75h5v5h-5zM222 75h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M202 30h5v5h-5zM212 30h5v5h-5zM197 35h5v5h-5zM217 35h5v5h-5zM217 40h5v5h-5zM237 40h5v5h-5zM232 45h5v5h-5zM227 50h5v5h-5zM222 55h5v5h-5zM227 55h5v5h-5zM237 55h5v5h-5zM237 60h5v5h-5zM202 65h5v5h-5zM207 65h5v5h-5zM217 65h5v5h-5zM202 70h5v5h-5zM207 70h5v5h-5zM217 70h5v5h-5zM227 70h5v5h-5zM207 75h5v5h-5zM227 75h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M207 30h5v5h-5zM202 35h5v5h-5zM212 35h5v5h-5zM222 35h5v5h-5zM227 35h5v5h-5zM232 35h5v5h-5zM212 40h5v5h-5zM222 40h5v5h-5zM227 40h5v5h-5zM232 40h5v5h-5zM222 45h5v5h-5zM227 45h5v5h-5zM217 50h5v5h-5zM222 50h5v5h-5zM232 55h5v5h-5zM227 60h5v5h-5zM232 60h5v5h-5zM232 65h5v5h-5zM237 65h5v5h-5zM212 70h5v5h-5zM222 70h5v5h-5zM232 70h5v5h-5zM232 75h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="246" y="30" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M266 30h5v5h-5zM271 30h5v5h-5zM251 40h5v5h-5zM256 40h5v5h-5zM261 40h5v5h-5zM281 40h5v5h-5zM286 40h5v5h-5zM246 45h5v5h-5zM271 50h5v5h-5zM291 50h5v5h-5zM266 65h5v5h-5zM271 65h5v5h-5zM286 65h5v5h-5zM271 70h5v5h-5zM286 70h5v5h-5zM266 75h5v5h-5zM286 75h5v5h-5zM291 75h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M281 30h5v5h-5zM246 35h5v5h-5zM266 35h5v5h-5zM256 45h5v5h-5zM261 45h5v5h-5zM266 50h5v5h-5zM246 55h5v5h-5zM246 65h5v5h-5zM256 65h5v5h-5zM261 65h5v5h-5zM281 70h5v5h-5zM276 75h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M281 35h5v5h-5zM266 40h5v5h-5zM266 45h5v5h-5zM251 50h5v5h-5zM286 50h5v5h-5zM256 55h5v5h-5zM261 55h5v5h-5zM271 55h5v5h-5zM291 55h5v5h-5zM246 60h5v5h-5zM251 60h5v5h-5zM261 60h5v5h-5zM276 60h5v5h-5zM286 60h5v5h-5zM291 60h5v5h-5zM281 65h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M276 30h5v5h-5zM271 35h5v5h-5zM276 35h5v5h-5zM271 40h5v5h-5zM276 40h5v5h-5zM271 45h5v5h-5zM276 45h5v5h-5zM281 45h5v5h-5zM256 50h5v5h-5zM261 50h5v5h-5zM276 50h5v5h-5zM281 50h5v5h-5zM251 55h5v5h-5zM266 55h5v5h-5zM276 55h5v5h-5zM281 55h5v5h-5zM286 55h5v5h-5zM256 60h5v5h-5zM266 60h5v5h-5zM271 60h5v5h-5zM281 60h5v5h-5zM276 65h5v5h-5zM276 70h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="300" y="30" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M310 30h5v5h-5zM300 35h5v5h-5zM305 40h5v5h-5zM320 40h5v5h-5zM300 45h5v5h-5zM345 45h5v5h-5zM305 50h5v5h-5zM315 55h5v5h-5zM345 55h5v5h-5zM300 60h5v5h-5zM320 60h5v5h-5zM330 60h5v5h-5zM335 60h5v5h-5zM305 65h5v5h-5zM315 65h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M315 30h5v5h-5zM320 30h5v5h-5zM325 30h5v5h-5zM335 30h5v5h-5zM340 30h5v5h-5zM305 35h5v5h-5zM330 35h5v5h-5zM325 40h5v5h-5zM340 40h5v5h-5zM345 40h5v5h-5zM320 55h5v5h-5zM310 65h5v5h-5zM320 65h5v5h-5zM325 65h5v5h-5zM335 65h5v5h-5zM300 70h5v5h-5zM315 75h5v5h-5zM335 75h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M330 30h5v5h-5zM345 30h5v5h-5zM325 35h5v5h-5zM310 40h5v5h-5zM315 40h5v5h-5zM330 40h5v5h-5zM335 40h5v5h-5zM340 45h5v5h-5zM345 50h5v5h-5zM335 55h5v5h-5zM340 55h5v5h-5zM340 60h5v5h-5zM345 60h5v5h-5zM330 65h5v5h-5zM345 65h5v5h-5zM305 70h5v5h-5zM330 70h5v5h-5zM310 75h5v5h-5zM320 75h5v5h-5zM325 75h5v5h-5zM330 75h5v5h-5zM340 75h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M310 35h5v5h-5zM315 35h5v5h-5zM320 35h5v5h-5zM335 35h5v5h-5zM340 35h5v5h-5zM345 35h5v5h-5zM310 45h5v5h-5zM315 45h5v5h-5zM320 45h5v5h-5zM325 45h5v5h-5zM330 45h5v5h-5zM335 45h5v5h-5zM310 50h5v5h-5zM315 50h5v5h-5zM320 50h5v5h-5zM325 50h5v5h-5zM330 50h5v5h-5zM335 50h5v5h-5zM340 50h5v5h-5zM340 65h5v5h-5zM310 70h5v5h-5zM315 70h5v5h-5zM320 70h5v5h-5zM325 70h5v5h-5zM335 70h5v5h-5zM340 70h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="354" y="30" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M354 30h5v5h-5zM364 30h5v5h-5zM399 30h5v5h-5zM364 35h5v5h-5zM369 35h5v5h-5zM389 35h5v5h-5zM359 45h5v5h-5zM369 45h5v5h-5zM374 45h5v5h-5zM389 50h5v5h-5zM389 55h5v5h-5zM399 55h5v5h-5zM374 60h5v5h-5zM379 60h5v5h-5zM364 65h5v5h-5zM369 65h5v5h-5zM394 65h5v5h-5zM369 75h5v5h-5zM379 75h5v5h-5zM394 75h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M374 30h5v5h-5zM389 30h5v5h-5zM374 35h5v5h-5zM359 40h5v5h-5zM389 60h5v5h-5zM359 65h5v5h-5zM389 65h5v5h-5zM369 70h5v5h-5zM384 70h5v5h-5zM389 70h5v5h-5zM399 70h5v5h-5zM364 75h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M379 30h5v5h-5zM384 35h5v5h-5zM379 40h5v5h-5zM384 50h5v5h-5zM374 55h5v5h-5zM359 60h5v5h-5zM364 60h5v5h-5zM369 60h5v5h-5zM374 65h5v5h-5zM364 70h5v5h-5zM379 70h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M384 30h5v5h-5zM379 35h5v5h-5zM364 40h5v5h-5zM369 40h5v5h-5zM374 40h5v5h-5zM364 45h5v5h-5zM359 50h5v5h-5zM364 50h5v5h-5zM369 50h5v5h-5zM374 50h5v5h-5zM379 50h5v5h-5zM354 55h5v5h-5zM359 55h5v5h-5zM364 55h5v5h-5zM369 55h5v5h-5zM379 55h5v5h-5zM384 55h5v5h-5zM354 60h5v5h-5zM384 60h5v5h-5zM354 65h5v5h-5zM379 65h5v5h-5zM384 65h5v5h-5zM359 70h5v5h-5zM374 70h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="408" y="30" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M413 35h5v5h-5zM423 35h5v5h-5zM433 35h5v5h-5zM408 40h5v5h-5zM453 40h5v5h-5zM413 45h5v5h-5zM418 45h5v5h-5zM423 45h5v5h-5zM433 45h5v5h-5zM408 50h5v5h-5zM428 50h5v5h-5zM418 55h5v5h-5zM433 55h5v5h-5zM418 60h5v5h-5zM428 60h5v5h-5zM448 60h5v5h-5zM423 70h5v5h-5zM438 70h5v5h-5zM408 75h5v5h-5zM423 75h5v5h-5zM453 75h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M433 30h5v5h-5zM418 35h5v5h-5zM428 35h5v5h-5zM438 35h5v5h-5zM438 45h5v5h-5zM413 55h5v5h-5zM433 60h5v5h-5zM413 65h5v5h-5zM428 65h5v5h-5zM428 70h5v5h-5zM413 75h5v5h-5zM418 75h5v5h-5zM433 75h5v5h-5zM443 75h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M443 35h5v5h-5zM413 40h5v5h-5zM423 40h5v5h-5zM438 40h5v5h-5zM443 40h5v5h-5zM438 50h5v5h-5zM448 50h5v5h-5zM443 55h5v5h-5zM438 65h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M448 35h5v5h-5zM418 40h5v5h-5zM428 40h5v5h-5zM433 40h5v5h-5zM448 40h5v5h-5zM443 45h5v5h-5zM448 45h5v5h-5zM443 50h5v5h-5zM438 55h5v5h-5zM438 60h5v5h-5zM443 60h5v5h-5zM433 65h5v5h-5zM433 70h5v5h-5zM428 75h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="462" y="30" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M462 30h5v5h-5zM477 30h5v5h-5zM497 30h5v5h-5zM502 30h5v5h-5zM467 35h5v5h-5zM477 35h5v5h-5zM482 35h5v5h-5zM502 35h5v5h-5zM467 40h5v5h-5zM472 45h5v5h-5zM467 50h5v5h-5zM507 50h5v5h-5zM502 55h5v5h-5zM507 55h5v5h-5zM462 60h5v5h-5zM477 60h5v5h-5zM472 65h5v5h-5zM477 75h5v5h-5zM482 75h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M487 35h5v5h-5zM492 35h5v5h-5zM497 35h5v5h-5zM472 40h5v5h-5zM507 40h5v5h-5zM462 45h5v5h-5zM467 45h5v5h-5zM477 55h5v5h-5zM502 60h5v5h-5zM507 65h5v5h-5zM472 70h5v5h-5zM477 70h5v5h-5zM487 70h5v5h-5zM497 75h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M497 40h5v5h-5zM477 45h5v5h-5zM482 50h5v5h-5zM492 50h5v5h-5zM497 50h5v5h-5zM487 60h5v5h-5zM497 65h5v5h-5zM497 70h5v5h-5zM502 70h5v5h-5zM507 70h5v5h-5zM487 75h5v5h-5zM502 75h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M477 40h5v5h-5zM482 40h5v5h-5zM487 40h5v5h-5zM492 40h5v5h-5zM502 40h5v5h-5zM482 45h5v5h-5zM487 45h5v5h-5zM492 45h5v5h-5zM497 45h5v5h-5zM502 45h5v5h-5zM507 45h5v5h-5zM477 50h5v5h-5zM487 50h5v5h-5zM502 50h5v5h-5zM482 55h5v5h-5zM487 55h5v5h-5zM492 55h5v5h-5zM497 55h5v5h-5zM482 60h5v5h-5zM492 60h5v5h-5zM497 60h5v5h-5zM477 65h5v5h-5zM482 65h5v5h-5zM487 65h5v5h-5zM492 65h5v5h-5zM502 65h5v5h-5zM482 70h5v5h-5zM492 70h5v5h-5zM492 75h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="516" y="30" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M551 30h5v5h-5zM556 30h5v5h-5zM516 35h5v5h-5zM516 40h5v5h-5zM516 50h5v5h-5zM526 50h5v5h-5zM526 60h5v5h-5zM541 60h5v5h-5zM521 65h5v5h-5zM546 65h5v5h-5zM541 70h5v5h-5zM521 75h5v5h-5zM526 75h5v5h-5zM556 75h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M551 35h5v5h-5zM526 45h5v5h-5zM531 45h5v5h-5zM546 50h5v5h-5zM521 55h5v5h-5zM526 55h5v5h-5zM546 60h5v5h-5zM556 70h5v5h-5zM561 70h5v5h-5zM541 75h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M536 30h5v5h-5zM531 35h5v5h-5zM546 35h5v5h-5zM556 35h5v5h-5zM546 40h5v5h-5zM551 40h5v5h-5zM556 40h5v5h-5zM561 40h5v5h-5zM536 45h5v5h-5zM556 45h5v5h-5zM531 50h5v5h-5zM551 50h5v5h-5zM536 55h5v5h-5zM556 55h5v5h-5zM561 55h5v5h-5zM556 60h5v5h-5zM561 75h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M541 30h5v5h-5zM546 30h5v5h-5zM536 35h5v5h-5zM541 35h5v5h-5zM531 40h5v5h-5zM536 40h5v5h-5zM561 45h5v5h-5zM536 50h5v5h-5zM541 50h5v5h-5zM556 50h5v5h-5zM561 50h5v5h-5zM541 55h5v5h-5zM546 55h5v5h-5zM551 55h5v5h-5zM561 60h5v5h-5zM556 65h5v5h-5zM561 65h5v5h-5z" class="fill-a" opacity="1.0"/>
<text x="55" y="98" text-anchor="middle" class="fig-small">0</text>
<text x="109" y="98" text-anchor="middle" class="fig-small">1</text>
<text x="163" y="98" text-anchor="middle" class="fig-small">2</text>
<text x="217" y="98" text-anchor="middle" class="fig-small">3</text>
<text x="271" y="98" text-anchor="middle" class="fig-small">4</text>
<text x="325" y="98" text-anchor="middle" class="fig-small">5</text>
<text x="379" y="98" text-anchor="middle" class="fig-small">6</text>
<text x="433" y="98" text-anchor="middle" class="fig-small">7</text>
<text x="487" y="98" text-anchor="middle" class="fig-small">8</text>
<text x="541" y="98" text-anchor="middle" class="fig-small">9</text>
<text x="30" y="132" class="fig-bold">Ten draws of the digit 7: shift, stroke width, and noise all change</text>
<rect x="30" y="142" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M65 142h5v5h-5zM50 147h5v5h-5zM55 147h5v5h-5zM70 147h5v5h-5zM75 147h5v5h-5zM40 157h5v5h-5zM45 157h5v5h-5zM50 157h5v5h-5zM30 162h5v5h-5zM35 162h5v5h-5zM40 162h5v5h-5zM55 162h5v5h-5zM45 167h5v5h-5zM50 167h5v5h-5zM30 172h5v5h-5zM70 172h5v5h-5zM30 177h5v5h-5zM35 177h5v5h-5zM70 177h5v5h-5zM75 177h5v5h-5zM30 182h5v5h-5zM35 182h5v5h-5zM75 182h5v5h-5zM30 187h5v5h-5zM40 187h5v5h-5zM45 187h5v5h-5zM55 187h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M40 142h5v5h-5zM40 147h5v5h-5zM45 147h5v5h-5zM60 147h5v5h-5zM65 147h5v5h-5zM45 152h5v5h-5zM55 157h5v5h-5zM60 157h5v5h-5zM75 162h5v5h-5zM60 167h5v5h-5zM35 172h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M65 152h5v5h-5zM65 157h5v5h-5zM65 162h5v5h-5zM70 162h5v5h-5zM65 167h5v5h-5zM70 167h5v5h-5zM60 172h5v5h-5zM65 177h5v5h-5zM50 182h5v5h-5zM60 182h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M40 152h5v5h-5zM50 152h5v5h-5zM55 152h5v5h-5zM60 152h5v5h-5zM70 152h5v5h-5zM75 152h5v5h-5zM70 157h5v5h-5zM75 157h5v5h-5zM65 172h5v5h-5zM55 177h5v5h-5zM60 177h5v5h-5zM55 182h5v5h-5zM50 187h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="84" y="142" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M89 142h5v5h-5zM109 142h5v5h-5zM89 157h5v5h-5zM124 157h5v5h-5zM129 162h5v5h-5zM89 167h5v5h-5zM99 167h5v5h-5zM109 167h5v5h-5zM94 172h5v5h-5zM119 172h5v5h-5zM129 172h5v5h-5zM84 177h5v5h-5zM89 177h5v5h-5zM109 177h5v5h-5zM119 177h5v5h-5zM104 182h5v5h-5zM109 187h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M84 147h5v5h-5zM94 147h5v5h-5zM99 147h5v5h-5zM104 147h5v5h-5zM109 147h5v5h-5zM114 147h5v5h-5zM94 162h5v5h-5zM114 167h5v5h-5zM124 167h5v5h-5zM99 187h5v5h-5zM124 187h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M129 142h5v5h-5zM119 152h5v5h-5zM84 157h5v5h-5zM94 157h5v5h-5zM99 157h5v5h-5zM109 162h5v5h-5zM119 162h5v5h-5zM104 167h5v5h-5zM99 172h5v5h-5zM124 172h5v5h-5zM104 177h5v5h-5zM94 182h5v5h-5zM99 182h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M89 147h5v5h-5zM84 152h5v5h-5zM89 152h5v5h-5zM94 152h5v5h-5zM99 152h5v5h-5zM104 152h5v5h-5zM109 152h5v5h-5zM114 152h5v5h-5zM104 157h5v5h-5zM109 157h5v5h-5zM114 157h5v5h-5zM119 157h5v5h-5zM114 162h5v5h-5zM104 172h5v5h-5zM109 172h5v5h-5zM99 177h5v5h-5zM94 187h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="138" y="142" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M173 142h5v5h-5zM183 142h5v5h-5zM183 147h5v5h-5zM138 152h5v5h-5zM183 157h5v5h-5zM143 162h5v5h-5zM153 162h5v5h-5zM183 167h5v5h-5zM143 172h5v5h-5zM173 177h5v5h-5zM183 177h5v5h-5zM158 182h5v5h-5zM143 187h5v5h-5zM148 187h5v5h-5zM153 187h5v5h-5zM168 187h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M153 142h5v5h-5zM178 142h5v5h-5zM143 147h5v5h-5zM178 147h5v5h-5zM163 152h5v5h-5zM158 157h5v5h-5zM178 157h5v5h-5zM183 162h5v5h-5zM153 167h5v5h-5zM148 172h5v5h-5zM163 172h5v5h-5zM158 177h5v5h-5zM178 177h5v5h-5zM148 182h5v5h-5zM173 182h5v5h-5zM178 187h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M138 142h5v5h-5zM158 142h5v5h-5zM168 142h5v5h-5zM153 147h5v5h-5zM153 152h5v5h-5zM163 157h5v5h-5zM173 157h5v5h-5zM163 162h5v5h-5zM168 162h5v5h-5zM158 167h5v5h-5zM168 182h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M143 142h5v5h-5zM148 142h5v5h-5zM163 142h5v5h-5zM138 147h5v5h-5zM148 147h5v5h-5zM158 147h5v5h-5zM163 147h5v5h-5zM168 147h5v5h-5zM173 147h5v5h-5zM148 152h5v5h-5zM158 152h5v5h-5zM168 152h5v5h-5zM173 152h5v5h-5zM178 152h5v5h-5zM168 157h5v5h-5zM158 162h5v5h-5zM163 167h5v5h-5zM168 167h5v5h-5zM153 172h5v5h-5zM158 172h5v5h-5zM148 177h5v5h-5zM153 177h5v5h-5zM143 182h5v5h-5zM153 182h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="192" y="142" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M197 142h5v5h-5zM217 142h5v5h-5zM227 142h5v5h-5zM237 142h5v5h-5zM197 167h5v5h-5zM212 167h5v5h-5zM237 167h5v5h-5zM192 172h5v5h-5zM192 177h5v5h-5zM202 177h5v5h-5zM197 182h5v5h-5zM232 182h5v5h-5zM227 187h5v5h-5zM232 187h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M202 142h5v5h-5zM212 142h5v5h-5zM232 147h5v5h-5zM222 167h5v5h-5zM207 172h5v5h-5zM217 172h5v5h-5zM222 187h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M207 142h5v5h-5zM192 147h5v5h-5zM212 147h5v5h-5zM217 147h5v5h-5zM222 147h5v5h-5zM227 147h5v5h-5zM237 147h5v5h-5zM197 152h5v5h-5zM202 152h5v5h-5zM192 157h5v5h-5zM202 157h5v5h-5zM212 157h5v5h-5zM227 157h5v5h-5zM237 157h5v5h-5zM237 162h5v5h-5zM222 177h5v5h-5zM227 177h5v5h-5zM232 177h5v5h-5zM207 182h5v5h-5zM217 182h5v5h-5zM227 182h5v5h-5zM217 187h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M197 147h5v5h-5zM202 147h5v5h-5zM207 147h5v5h-5zM207 152h5v5h-5zM212 152h5v5h-5zM217 152h5v5h-5zM222 152h5v5h-5zM227 152h5v5h-5zM232 152h5v5h-5zM237 152h5v5h-5zM207 157h5v5h-5zM217 157h5v5h-5zM222 157h5v5h-5zM232 157h5v5h-5zM222 162h5v5h-5zM227 162h5v5h-5zM232 162h5v5h-5zM217 167h5v5h-5zM227 167h5v5h-5zM232 167h5v5h-5zM222 172h5v5h-5zM227 172h5v5h-5zM232 172h5v5h-5zM217 177h5v5h-5zM212 182h5v5h-5zM222 182h5v5h-5zM207 187h5v5h-5zM212 187h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="246" y="142" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M246 142h5v5h-5zM251 152h5v5h-5zM261 152h5v5h-5zM266 152h5v5h-5zM276 152h5v5h-5zM291 157h5v5h-5zM251 167h5v5h-5zM271 167h5v5h-5zM256 172h5v5h-5zM286 172h5v5h-5zM276 182h5v5h-5zM286 182h5v5h-5zM251 187h5v5h-5zM261 187h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M251 142h5v5h-5zM266 147h5v5h-5zM271 152h5v5h-5zM276 157h5v5h-5zM256 162h5v5h-5zM276 162h5v5h-5zM266 172h5v5h-5zM281 172h5v5h-5zM266 177h5v5h-5zM246 187h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M266 142h5v5h-5zM281 142h5v5h-5zM291 142h5v5h-5zM251 147h5v5h-5zM286 147h5v5h-5zM256 152h5v5h-5zM281 152h5v5h-5zM291 152h5v5h-5zM281 157h5v5h-5zM286 162h5v5h-5zM276 167h5v5h-5zM271 172h5v5h-5zM276 177h5v5h-5zM266 182h5v5h-5zM271 182h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M256 142h5v5h-5zM261 142h5v5h-5zM271 142h5v5h-5zM276 142h5v5h-5zM286 142h5v5h-5zM256 147h5v5h-5zM261 147h5v5h-5zM271 147h5v5h-5zM276 147h5v5h-5zM281 147h5v5h-5zM291 147h5v5h-5zM286 152h5v5h-5zM286 157h5v5h-5zM281 162h5v5h-5zM281 167h5v5h-5zM276 172h5v5h-5zM271 177h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="300" y="142" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M305 142h5v5h-5zM315 147h5v5h-5zM325 147h5v5h-5zM300 152h5v5h-5zM325 152h5v5h-5zM345 152h5v5h-5zM310 157h5v5h-5zM300 162h5v5h-5zM310 162h5v5h-5zM300 167h5v5h-5zM320 167h5v5h-5zM310 172h5v5h-5zM335 172h5v5h-5zM310 177h5v5h-5zM310 182h5v5h-5zM315 182h5v5h-5zM325 182h5v5h-5zM330 182h5v5h-5zM340 182h5v5h-5zM305 187h5v5h-5zM320 187h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M320 142h5v5h-5zM340 142h5v5h-5zM335 147h5v5h-5zM305 167h5v5h-5zM335 167h5v5h-5zM330 172h5v5h-5zM340 172h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M325 142h5v5h-5zM330 142h5v5h-5zM345 147h5v5h-5zM330 157h5v5h-5zM335 157h5v5h-5zM340 157h5v5h-5zM325 162h5v5h-5zM330 162h5v5h-5zM335 162h5v5h-5zM320 177h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M310 142h5v5h-5zM315 142h5v5h-5zM335 142h5v5h-5zM345 142h5v5h-5zM340 147h5v5h-5zM335 152h5v5h-5zM340 152h5v5h-5zM325 167h5v5h-5zM330 167h5v5h-5zM320 172h5v5h-5zM325 172h5v5h-5zM325 177h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="354" y="142" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M379 142h5v5h-5zM389 142h5v5h-5zM399 142h5v5h-5zM394 147h5v5h-5zM364 157h5v5h-5zM374 157h5v5h-5zM379 157h5v5h-5zM364 162h5v5h-5zM369 162h5v5h-5zM389 162h5v5h-5zM399 162h5v5h-5zM359 167h5v5h-5zM364 167h5v5h-5zM369 167h5v5h-5zM394 167h5v5h-5zM364 172h5v5h-5zM369 172h5v5h-5zM364 177h5v5h-5zM379 177h5v5h-5zM394 177h5v5h-5zM374 182h5v5h-5zM379 182h5v5h-5zM394 182h5v5h-5zM354 187h5v5h-5zM359 187h5v5h-5zM394 187h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M359 142h5v5h-5zM364 147h5v5h-5zM379 147h5v5h-5zM374 162h5v5h-5zM384 162h5v5h-5zM354 172h5v5h-5zM354 177h5v5h-5zM384 177h5v5h-5zM364 182h5v5h-5zM369 182h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M364 152h5v5h-5zM369 152h5v5h-5zM384 152h5v5h-5zM384 157h5v5h-5zM374 167h5v5h-5zM379 172h5v5h-5zM369 177h5v5h-5zM364 187h5v5h-5zM369 187h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M354 152h5v5h-5zM359 152h5v5h-5zM374 152h5v5h-5zM379 152h5v5h-5zM389 152h5v5h-5zM379 162h5v5h-5zM379 167h5v5h-5zM374 172h5v5h-5zM374 177h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="408" y="142" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M408 147h5v5h-5zM423 147h5v5h-5zM433 147h5v5h-5zM408 152h5v5h-5zM438 152h5v5h-5zM408 157h5v5h-5zM433 157h5v5h-5zM453 162h5v5h-5zM423 167h5v5h-5zM428 167h5v5h-5zM418 172h5v5h-5zM443 172h5v5h-5zM413 177h5v5h-5zM443 177h5v5h-5zM433 182h5v5h-5zM428 187h5v5h-5zM453 187h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M413 147h5v5h-5zM443 147h5v5h-5zM453 152h5v5h-5zM413 157h5v5h-5zM438 157h5v5h-5zM453 157h5v5h-5zM418 167h5v5h-5zM443 167h5v5h-5zM408 177h5v5h-5zM423 177h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M428 147h5v5h-5zM443 152h5v5h-5zM448 157h5v5h-5zM433 162h5v5h-5zM448 162h5v5h-5zM433 167h5v5h-5zM438 167h5v5h-5zM428 172h5v5h-5zM423 182h5v5h-5zM428 182h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M413 142h5v5h-5zM418 142h5v5h-5zM423 142h5v5h-5zM428 142h5v5h-5zM433 142h5v5h-5zM438 142h5v5h-5zM443 142h5v5h-5zM448 142h5v5h-5zM453 142h5v5h-5zM418 147h5v5h-5zM438 147h5v5h-5zM448 147h5v5h-5zM453 147h5v5h-5zM448 152h5v5h-5zM443 157h5v5h-5zM438 162h5v5h-5zM443 162h5v5h-5zM433 172h5v5h-5zM438 172h5v5h-5zM428 177h5v5h-5zM433 177h5v5h-5zM438 177h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="462" y="142" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M477 142h5v5h-5zM492 142h5v5h-5zM462 147h5v5h-5zM467 147h5v5h-5zM502 147h5v5h-5zM462 152h5v5h-5zM477 162h5v5h-5zM482 162h5v5h-5zM492 162h5v5h-5zM492 167h5v5h-5zM462 172h5v5h-5zM477 172h5v5h-5zM487 172h5v5h-5zM467 177h5v5h-5zM487 177h5v5h-5zM467 182h5v5h-5zM497 182h5v5h-5zM462 187h5v5h-5zM472 187h5v5h-5zM502 187h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M482 142h5v5h-5zM482 147h5v5h-5zM487 147h5v5h-5zM497 157h5v5h-5zM507 162h5v5h-5zM502 167h5v5h-5zM482 172h5v5h-5zM502 172h5v5h-5zM507 172h5v5h-5zM477 177h5v5h-5zM502 177h5v5h-5zM472 182h5v5h-5zM477 187h5v5h-5zM492 187h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M497 147h5v5h-5zM482 152h5v5h-5zM492 152h5v5h-5zM497 152h5v5h-5zM507 152h5v5h-5zM472 157h5v5h-5zM477 157h5v5h-5zM482 157h5v5h-5zM487 157h5v5h-5zM507 157h5v5h-5zM497 162h5v5h-5zM502 162h5v5h-5zM507 167h5v5h-5zM497 177h5v5h-5zM487 187h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M472 147h5v5h-5zM477 147h5v5h-5zM492 147h5v5h-5zM507 147h5v5h-5zM467 152h5v5h-5zM472 152h5v5h-5zM477 152h5v5h-5zM487 152h5v5h-5zM502 152h5v5h-5zM467 157h5v5h-5zM492 157h5v5h-5zM502 157h5v5h-5zM497 167h5v5h-5zM492 172h5v5h-5zM497 172h5v5h-5zM492 177h5v5h-5zM482 182h5v5h-5zM487 182h5v5h-5zM492 182h5v5h-5zM482 187h5v5h-5z" class="fill-a" opacity="1.0"/>
<rect x="516" y="142" width="50" height="50" class="fill-sunk stroke-muted"/>
<path d="M516 142h5v5h-5zM516 147h5v5h-5zM536 147h5v5h-5zM526 152h5v5h-5zM541 152h5v5h-5zM516 157h5v5h-5zM521 157h5v5h-5zM536 157h5v5h-5zM541 157h5v5h-5zM561 157h5v5h-5zM531 162h5v5h-5zM556 167h5v5h-5zM526 172h5v5h-5zM521 177h5v5h-5zM561 177h5v5h-5zM521 187h5v5h-5zM536 187h5v5h-5zM546 187h5v5h-5zM556 187h5v5h-5zM561 187h5v5h-5z" class="fill-a" opacity="0.25"/>
<path d="M526 147h5v5h-5zM546 152h5v5h-5zM526 157h5v5h-5zM556 157h5v5h-5zM541 162h5v5h-5zM556 162h5v5h-5zM536 177h5v5h-5zM536 182h5v5h-5zM551 187h5v5h-5z" class="fill-a" opacity="0.5"/>
<path d="M531 142h5v5h-5zM546 142h5v5h-5zM521 147h5v5h-5zM541 147h5v5h-5zM546 147h5v5h-5zM556 147h5v5h-5zM561 147h5v5h-5zM561 152h5v5h-5zM546 157h5v5h-5zM551 157h5v5h-5zM546 162h5v5h-5zM551 162h5v5h-5zM536 172h5v5h-5zM541 172h5v5h-5zM541 177h5v5h-5z" class="fill-a" opacity="0.75"/>
<path d="M526 142h5v5h-5zM536 142h5v5h-5zM541 142h5v5h-5zM551 142h5v5h-5zM556 142h5v5h-5zM561 142h5v5h-5zM531 147h5v5h-5zM551 147h5v5h-5zM551 152h5v5h-5zM556 152h5v5h-5zM541 167h5v5h-5zM546 167h5v5h-5zM551 167h5v5h-5zM546 172h5v5h-5zM551 172h5v5h-5zM546 177h5v5h-5zM551 177h5v5h-5zM541 182h5v5h-5z" class="fill-a" opacity="1.0"/>
</svg>
Figure 1. Top: one draw of each digit. Bottom: ten draws of the digit 7. The planted pattern is the shift, the thickness, and the grain. A pixel that is dark in one 7 can be light in the next, so the identity of a digit lives in the arrangement of strokes, not in any one pixel.
```

The planted difficulty is the shift. A linear model has one weight per pixel and class, so it learns a blurry average of each digit's possible positions. A hidden unit can learn a short stroke detector that fires wherever the stroke lands, and the next layer combines detectors. That is why your result is judged against the linear baseline: the gap between them measures what the hidden layers buy.

## Plan the build

Figure 2 shows the pipeline. Data and starting weights come first. Everything inside the dashed box repeats for every mini-batch. The validation and stopping stages run once per epoch, and the test set is touched once, at the end.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Pipeline: make_data and he_init feed a repeated mini-batch loop of forward with dropout, smoothed cross-entropy, backward and AdamW update, with lr_schedule feeding the update and gradient_check testing backward; each epoch ends with validation and early stopping, and the best parameters go to predict on the test set.">
<defs>
<marker id="p076-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker>
<marker id="p076-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker>
</defs>
<rect x="10" y="14" width="124" height="52" rx="6" class="fill-sunk stroke-a"/>
<text x="72" y="36" text-anchor="middle" class="fig-bold">make_data</text>
<text x="72" y="55" text-anchor="middle" class="fig-small">1500 / 500 / 1000</text>
<rect x="160" y="14" width="124" height="52" rx="6" class="fill-sunk stroke-b"/>
<text x="222" y="36" text-anchor="middle" class="fig-bold">he_init</text>
<text x="222" y="55" text-anchor="middle" class="fig-small">100-64-32-10</text>
<rect x="310" y="14" width="124" height="52" rx="6" class="fill-sunk stroke-muted dashed"/>
<text x="372" y="36" text-anchor="middle" class="fig-bold">gradient_check</text>
<text x="372" y="55" text-anchor="middle" class="fig-small">tiny net, once</text>
<rect x="466" y="14" width="124" height="52" rx="6" class="fill-sunk stroke-b"/>
<text x="528" y="36" text-anchor="middle" class="fig-bold">lr_schedule</text>
<text x="528" y="55" text-anchor="middle" class="fig-small">warmup, cosine</text>
<line x1="136" y1="40" x2="157" y2="40" class="stroke-muted" marker-end="url(#p076-arrow)"/>
<rect x="10" y="94" width="580" height="104" rx="8" class="stroke-muted dashed" fill="none"/>
<text x="20" y="112" class="fig-small fill-muted">every mini-batch of 64</text>
<rect x="20" y="124" width="116" height="58" rx="6" class="fill-sunk stroke-b"/>
<text x="78" y="148" text-anchor="middle" class="fig-bold">forward</text>
<text x="78" y="166" text-anchor="middle" class="fig-small">dropout on</text>
<rect x="164" y="124" width="116" height="58" rx="6" class="fill-sunk stroke-b"/>
<text x="222" y="148" text-anchor="middle" class="fig-bold">smoothed</text>
<text x="222" y="166" text-anchor="middle" class="fig-bold">cross-entropy</text>
<rect x="308" y="124" width="116" height="58" rx="6" class="fill-sunk stroke-b"/>
<text x="366" y="148" text-anchor="middle" class="fig-bold">backward</text>
<text x="366" y="166" text-anchor="middle" class="fig-small">same masks</text>
<rect x="452" y="124" width="116" height="58" rx="6" class="fill-sunk stroke-b"/>
<text x="510" y="148" text-anchor="middle" class="fig-bold">adamw_update</text>
<text x="510" y="166" text-anchor="middle" class="fig-small">no decay on b</text>
<line x1="138" y1="153" x2="161" y2="153" class="stroke-muted" marker-end="url(#p076-arrow)"/>
<line x1="282" y1="153" x2="305" y2="153" class="stroke-muted" marker-end="url(#p076-arrow)"/>
<line x1="426" y1="153" x2="449" y2="153" class="stroke-muted" marker-end="url(#p076-arrow)"/>
<polyline points="222,66 222,82 78,82 78,121" fill="none" class="stroke-muted" marker-end="url(#p076-arrow)"/>
<line x1="372" y1="68" x2="372" y2="121" class="stroke-muted dashed" marker-end="url(#p076-arrow)"/>
<line x1="528" y1="68" x2="528" y2="121" class="stroke-muted" marker-end="url(#p076-arrow)"/>
<rect x="10" y="236" width="150" height="58" rx="6" class="fill-sunk stroke-c"/>
<text x="85" y="260" text-anchor="middle" class="fig-bold">validation loss</text>
<text x="85" y="278" text-anchor="middle" class="fig-small">dropout off</text>
<rect x="206" y="236" width="150" height="58" rx="6" class="fill-sunk stroke-c"/>
<text x="281" y="260" text-anchor="middle" class="fig-bold">early stopping</text>
<text x="281" y="278" text-anchor="middle" class="fig-small">keep the best epoch</text>
<rect x="402" y="236" width="150" height="58" rx="6" class="fill-sunk stroke-a"/>
<text x="477" y="260" text-anchor="middle" class="fig-bold">predict</text>
<text x="477" y="278" text-anchor="middle" class="fig-small">test set, once</text>
<line x1="85" y1="200" x2="85" y2="233" class="stroke-muted" marker-end="url(#p076-arrow)"/>
<text x="95" y="222" class="fig-small fill-muted">each epoch</text>
<line x1="162" y1="265" x2="203" y2="265" class="stroke-muted" marker-end="url(#p076-arrow)"/>
<line x1="358" y1="265" x2="399" y2="265" class="stroke-muted" marker-end="url(#p076-arrow)"/>
</svg>
Figure 2. The pipeline. Orange boxes are the model and its training, blue boxes are data, green boxes are the honest evaluation. The dashed box repeats until early stopping ends it. The gradient check runs once on a small network before any training.
```

The order is not arbitrary. You check the gradients before training, because a training curve that looks fine can hide a wrong gradient that merely trains slowly. You judge the stopping point on validation data, because the training loss only ever goes down. You touch the test set once, at the end, because every look at it while you tune turns it into a second validation set and makes the final number optimistic.

> [!RECAP]
> - He initialisation draws weights with standard deviation $\sqrt{2/n_{in}}$ so a ReLU layer keeps the size of its signal.
> - Inverted dropout multiplies hidden activations by a mask divided by $1-p$ in training, and the backward pass reuses the same mask. At evaluation it does nothing.
> - Label smoothing swaps the one-hot target for $(1-\varepsilon)\,\text{onehot} + \varepsilon/k$, and the gradient on the logits is still $(p - \text{target})/n$.
> - AdamW decays the weights directly, outside the adaptive step, and leaves biases alone.
> - Early stopping watches plain validation loss and returns the weights from the best epoch.

## Build it step by step

Throughout, a network is a list of layers, and each layer is a dict `{"W": ..., "b": ...}` with `W` of shape `(n_in, n_out)`, so a layer computes `h @ W + b`, as in [the gradient chapter](/lessons/035-softmax-and-whole-network-gradients). `make_data` and `adamw_step` are given. You write the rest.

1. **`he_init(layer_sizes, rng)`.** Pair each size with the next, draw `W` from a normal with standard deviation $\sqrt{2/n_{in}}$, and set `b` to zeros. For the sizes `[100, 64, 32, 10]` the three scales are $\sqrt{2/100} = 0.141$, $\sqrt{2/64} = 0.177$, and $\sqrt{2/32} = 0.25$. The usual mistake is using the number of columns instead of rows. `n_in` is the first axis of `W`, because this project stores `W` as `(n_in, n_out)`.

2. **`forward(params, X, p_drop, rng, training)`.** Compute `z = h @ W + b` for each layer. The last layer returns `z` as the logits with no ReLU and no dropout. Every earlier layer applies a ReLU, and in training mode multiplies the result by `(rng.random(a.shape) >= p_drop) / (1 - p_drop)`. With $p = 0.2$ every survivor is multiplied by $1.25$, so the expected value of the layer's output equals its evaluation-time value, as shown in [the dropout chapter](/lessons/040-dropout-and-label-smoothing). Return the logits and a cache with `h_in`, `z`, and `mask` for each layer. The usual mistake is applying dropout to the logits or to the input. Dropout belongs on the hidden activations only.

3. **`smoothed_cross_entropy(logits, y, eps)`.** Compute a log-softmax with the row maximum subtracted, as in [the softmax chapter](/lessons/035-softmax-and-whole-network-gradients). Build the target $(1-\varepsilon)\,\text{onehot}(y) + \varepsilon/k$. For $k = 10$ and $\varepsilon = 0.1$ the correct class gets $0.91$ and each of the other nine gets $0.01$. The loss is the row-wise mean of $-\sum_j t_j \log p_j$, and the gradient on the logits is $(p - t)/n$. The usual mistake is forgetting the division by $n$, which makes the loss a mean and the gradient a sum, and the gradient check catches it.

4. **`backward(params, cache, dlogits)`.** Walk from the last layer to the first. For layer $i$ the weight gradient is `h_in.T @ d` and the bias gradient is `d.sum(axis=0)`. Then pass the slope to the layer before: multiply by `W.T`, then by that layer's dropout mask if it has one, then by `(z > 0)` for the ReLU. The order matters because the forward pass applied the ReLU first and the mask second. The usual mistake is redrawing the mask here. Use the one in the cache.

5. **`loss_and_grads(params, X, y, eps, p_drop, rng, training)`.** Call `forward`, then the loss, then `backward`, and return the loss with the gradients. This is three lines.

6. **`gradient_check(params, X, y, eps, grads, h)`.** Nudge each entry of each array up and down by `h`, recompute the loss with dropout off, and divide the difference by `2h`. Compare each whole array to the backprop gradient with the relative error from [the gradient chapter](/lessons/035-softmax-and-whole-network-gradients). When `grads` is supplied, check those instead of computing them, which lets you confirm that the check can fail. The usual mistake is leaving an entry nudged. Put each entry back right after reading the loss.

7. **`adamw_update(params, grads, states, lr, weight_decay)`.** Call the given `adamw_step` on every array, with `weight_decay` for each `W` and `0.0` for each `b`. Return fresh lists. Decay pulls weights toward zero, which shrinks the function a network can express. A bias shifts an output and carries no such risk, so decaying it only hurts. The usual mistake is mutating `params` in place, which corrupts the early-stopping snapshot.

8. **`lr_schedule(step, warmup, total, peak)`.** Return `peak * step / warmup` during warmup, and afterwards `peak * 0.5 * (1 + cos(pi * progress))` with `progress` running from 0 at the end of warmup to 1 at `total`, as in [the schedule chapter](/lessons/036-sgd-adam-and-schedules). With 24 steps per epoch, 60 epochs, and 5% warmup, `total` is 1,440 and `warmup` is 72. The rate is $1.5 \times 10^{-3}$ at step 36 and again at step 756, for a peak of $3 \times 10^{-3}$.

9. **`predict(params, X)`.** Run `forward` with `training=False` and return the index of the largest logit in each row. The usual mistake is forgetting that evaluation mode is a separate code path, not a smaller `p_drop`.

10. **`train(data, ...)`.** This is the loop. Build the parameters and an empty AdamW state for every array. Each epoch, shuffle with `rng.permutation`, step through mini-batches, get the rate from `lr_schedule`, call `loss_and_grads` with dropout on, and apply `adamw_update`. After each epoch, measure plain validation cross-entropy (`eps=0`) and accuracy with dropout off. When the loss is the lowest so far, copy the parameters. Stop once `patience` epochs in a row have failed to beat the best, and return the best parameters with a history. The usual mistake is returning the final parameters. By then they are `patience` epochs past the best.

Calling `train(data, hidden=(), p_drop=0.0)` builds a network with no hidden layer. That is the linear softmax baseline, produced by the same code, so any difference in the results comes from the hidden layers and not from a different training loop.

## Read your results

Run the file and compare. The numbers below come from the reference solution on `make_data(0)`.

| What you see | Reference value | What it means |
|---|---|---|
| Gradient check, 100-8-10 network, 20 images, $\varepsilon = 0.1$ | $2 \times 10^{-10}$ | Backprop matches finite differences to rounding error. |
| The same check with the first layer's gradient scaled by 1.5 | $0.20$ | The check can fail, so a pass means something. |
| Loss on untrained weights | $2.63$ | Close to $\ln 10 = 2.30$, so the starting network is not saturated. |
| Epochs run, best epoch (counting from 0) | 37 and 26 | Early stopping ended the run 23 epochs before the planned 60. |
| Validation accuracy at the best epoch | $0.988$ | The number used to pick the epoch, not to report. |
| Test accuracy of the network | $0.978$ | The number to tell the postal service. |
| Test accuracy of the linear baseline | $0.896$ | About one digit in ten misread. |
| Training time | $0.16$ s | Well inside a few seconds. |

Over seeds 0 to 4, the network scores between $0.978$ and $0.989$ on its test set, and the linear baseline between $0.896$ and $0.912$. The lead is at least $0.07$ every time, which is why the check asks for $0.05$.

Figure 3 shows the run behind those numbers. The validation loss falls from about $2.2$ to $0.18$ at epoch 26, then wanders slightly above it for 10 epochs, and training stops. The training loss sits higher than the validation loss. That is not a bug: the training loss includes label smoothing, which keeps it above zero, and it is measured with dropout on, while the validation loss is plain cross-entropy with dropout off. Compare train and validation curves only with the same loss and the same mode.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Training loss and validation loss against epoch for the reference run, with the best epoch marked at 26 and training stopped after epoch 36.">
<line x1="60" y1="225.0" x2="560" y2="225.0" class="stroke-grid"/>
<text x="52" y="229.0" text-anchor="end" class="fig-small">0.0</text>
<line x1="60" y1="191.7" x2="560" y2="191.7" class="stroke-grid"/>
<text x="52" y="195.7" text-anchor="end" class="fig-small">0.5</text>
<line x1="60" y1="158.3" x2="560" y2="158.3" class="stroke-grid"/>
<text x="52" y="162.3" text-anchor="end" class="fig-small">1.0</text>
<line x1="60" y1="125.0" x2="560" y2="125.0" class="stroke-grid"/>
<text x="52" y="129.0" text-anchor="end" class="fig-small">1.5</text>
<line x1="60" y1="91.7" x2="560" y2="91.7" class="stroke-grid"/>
<text x="52" y="95.7" text-anchor="end" class="fig-small">2.0</text>
<line x1="60" y1="58.3" x2="560" y2="58.3" class="stroke-grid"/>
<text x="52" y="62.3" text-anchor="end" class="fig-small">2.5</text>
<line x1="60" y1="25.0" x2="560" y2="25.0" class="stroke-grid"/>
<text x="52" y="29.0" text-anchor="end" class="fig-small">3.0</text>
<text x="60.0" y="243" text-anchor="middle" class="fig-small">0</text>
<text x="144.7" y="243" text-anchor="middle" class="fig-small">10</text>
<text x="229.5" y="243" text-anchor="middle" class="fig-small">20</text>
<text x="314.2" y="243" text-anchor="middle" class="fig-small">30</text>
<text x="399.0" y="243" text-anchor="middle" class="fig-small">40</text>
<text x="483.7" y="243" text-anchor="middle" class="fig-small">50</text>
<text x="560.0" y="243" text-anchor="middle" class="fig-small">59</text>
<text x="310" y="263" text-anchor="middle" class="fig-small">epoch (the schedule is planned for 60)</text>
<rect x="365.1" y="25" width="194.9" height="200" class="fill-sunk"/>
<text x="462.5" y="41" text-anchor="middle" class="fig-small">not run</text>
<polyline points="60.0,46.8 68.5,78.8 76.9,98.5 85.4,122.6 93.9,138.3 102.4,146.6 110.8,152.8 119.3,157.9 127.8,160.8 136.3,163.8 144.7,165.9 153.2,167.5 161.7,169.5 170.2,170.4 178.6,172.6 187.1,171.5 195.6,174.4 204.1,174.5 212.5,174.2 221.0,176.0 229.5,175.6 238.0,175.8 246.4,176.1 254.9,175.7 263.4,177.3 271.9,177.7 280.3,177.2 288.8,178.3 297.3,178.0 305.8,178.4 314.2,178.9 322.7,178.7 331.2,179.1 339.7,179.3 348.1,178.6 356.6,179.7 365.1,179.3" fill="none" class="stroke-a"/>
<polyline points="60.0,78.7 68.5,94.6 76.9,132.7 85.4,162.1 93.9,176.2 102.4,186.4 110.8,191.7 119.3,195.6 127.8,198.2 136.3,200.8 144.7,202.8 153.2,204.3 161.7,206.9 170.2,206.2 178.6,205.8 187.1,208.3 195.6,209.8 204.1,210.0 212.5,210.3 221.0,210.5 229.5,210.8 238.0,210.7 246.4,211.2 254.9,211.6 263.4,211.1 271.9,211.1 280.3,212.9 288.8,212.0 297.3,212.2 305.8,212.4 314.2,212.7 322.7,212.5 331.2,212.1 339.7,212.4 348.1,212.7 356.6,212.3 365.1,212.5" fill="none" class="stroke-b"/>
<line x1="280.3" y1="25" x2="280.3" y2="225" class="stroke-c dashed"/>
<circle cx="280.3" cy="212.9" r="5" class="fill-c"/>
<text x="272.3" y="41" text-anchor="end" class="fig-small">best epoch 26</text>
<text x="357.1" y="217" text-anchor="end" class="fig-small">stop after epoch 36</text>
<text x="127.8" y="138.8" class="fig-small fill-a">training loss (smoothed, dropout on)</text>
<text x="161.7" y="196.0" class="fig-small fill-b">validation loss (plain, dropout off)</text>
</svg>
Figure 3. Training loss (blue) and validation loss (orange) for the reference run. The green marker is the best epoch, the one whose parameters `train` returns. The shaded region is the epochs the schedule had planned and early stopping skipped.
```

Here is what a wrong number usually points to.

| Symptom | Likely cause |
|---|---|
| Gradient check error above $10^{-4}$ | A missing transpose, a mask redrawn in `backward`, or a missing division by $n$. Check one layer at a time. |
| Gradient check passes, but test accuracy is far below the linear baseline | Dropout left on at evaluation, or a learning rate that is too high. |
| Accuracy about $0.94$ instead of $0.98$ | Dropout is still on in `predict`. The reference network drops from $0.978$ to $0.938$ when you leave it on. |
| Validation loss far above the minimum in the history, with the returned weights | `train` returned the last epoch's parameters, not the best epoch's. |
| Loss on untrained weights far above $\ln 10$ | Weights drawn with the wrong scale, or `n_in` taken from the wrong axis. |
| Test accuracy much lower than validation accuracy | The test set was looked at while you tuned, or the validation set is too small to be a reliable guide. |

Be honest about what the guards bought. On this data a network with no dropout, no smoothing, and no stopping scores $0.980$, which is as good. Its training loss falls to $0.002$ and it runs all 60 epochs. The data is easy enough that memorising 1,500 images does not hurt accuracy, so the regularisers here cost nothing and save 23 epochs rather than win points. They matter when the data is scarcer or noisier, and you cannot know ahead of time which case you are in, which is why you build them and measure.

> [!KEY]
> The hidden layers, not the regularisers, bought the eight points over the linear baseline. A gradient check, an honest validation split, and dropout switched off at evaluation are what let you trust that number.

## Take it further

- Raise the pixel noise in `render_digit` from 0.25 to 0.6 and compare the network with and without dropout and smoothing. This shows whether the guards matter more when the data is noisier.
- Cut the training set to 200 images and repeat the comparison. This shows how regularisation pays off when there is little data to learn from.
- Replace the ReLU with the GELU from [the forward-pass chapter](/lessons/033-multilayer-perceptrons) and rerun the gradient check. This shows what breaks, and what stays fixed, when you swap an activation.
- Plot the digits the network misreads. This shows which shapes are ambiguous after a one-pixel shift, so you can say what a human operator should review.
- Replace the fixed `patience` with a rule that also stops when validation accuracy has not moved for 15 epochs. This shows how much the stopping rule changes the epoch you return.

## The exercise

You implement `he_init`, `forward`, `smoothed_cross_entropy`, `backward`, `loss_and_grads`, `gradient_check`, `adamw_update`, `lr_schedule`, `predict`, and `train`. `make_data` and `adamw_step` are given. The twelve checks go from the first stage to the last:

- `he_init` has the right shapes, zero biases, and weights with standard deviation $\sqrt{2/n_{in}}$.
- `forward` runs without a generator at evaluation, applies dropout only in training and only to hidden layers, and keeps the average output unchanged over many masks.
- `smoothed_cross_entropy` matches a hand-computed value and a finite-difference gradient, and survives logits of 900.
- `loss_and_grads` matches finite differences for every weight and bias, first without dropout and then with a fixed dropout mask.
- `gradient_check` reports a tiny error for correct gradients, a large error for a broken gradient, and restores every entry it nudges.
- `adamw_update` takes the expected first step, decays weights but not biases, and leaves its inputs unchanged.
- `lr_schedule` warms up linearly, peaks, and decays along a cosine to zero.
- `train` stops exactly `patience` epochs after the best epoch, returns the best epoch's parameters, and follows the schedule. `predict` is deterministic and matches the evaluation-mode logits.
- Two end-to-end checks train on fresh seeds. The first needs at least 95.5% test accuracy in under eight seconds. The second needs the network to beat the linear baseline by five points.
