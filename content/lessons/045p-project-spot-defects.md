---
title: "Spot defects on a production line"
slug: 045p-project-spot-defects
part: "Part 5 — Vision and sequences"
kind: project
summary: Build a small residual CNN in numpy that finds scratches and dents anywhere in a part's photo, and show why a flat network cannot.
minutes: 180
gpu: false
objectives:
  - Assemble im2col convolution, a residual block, global average pooling, and a linear head into one trainable detector.
  - Verify every backward pass against finite differences, from one convolution up to the whole network.
  - Use the receptive field to argue that the network can see a whole scratch.
  - Show that weight sharing beats a flat network with the same parameter count, and that moving a defect barely changes the answer.
prereqs:
  - 037-convolution-and-receptive-field
  - 038-im2col-and-convolution-backward
  - 039-residual-networks
  - 031-sgd-adam-and-schedules
exercise: 045p-project-spot-defects
---

# Spot defects on a production line

A factory camera photographs every part that leaves a press. Each photo is a small greyscale picture, 14 by 14 pixels after the line's preprocessing. About half of the parts you will see in testing carry a defect: a thin scratch, or a shallow dent. A person at the end of the line looks at a screen and pulls the bad parts. The plant wants a program that does the same job, so the person can look at the doubtful cases only.

The difficulty is that a defect can be anywhere. A scratch in the top-left corner and a scratch in the bottom-right corner are the same fault, but they light up completely different pixels. A network that treats the picture as one long list of 196 numbers has to learn the fault once for every place it can occur, and with a thousand training photos it runs out of examples long before it runs out of places. A convolution learns one small detector and slides it everywhere, so a single example anywhere teaches the detector for all positions.

In this project you build that detector from the parts of Part 5, check every piece against a slow or independent answer, and then run the experiment that justifies the whole design: the same data, the same optimiser, and about the same number of parameters, with and without weight sharing.

## The brief

You deliver five things:

- A receptive-field function, and the argument that your network sees a whole scratch.
- A convolution forward pass written as one matrix multiply, and its backward pass.
- A residual block, global average pooling, and a complete network with its backward pass.
- An Adam update, used by the training loop that is already written.
- The measured comparison: your CNN against a flat MLP with a similar number of parameters.

The work is judged by gradients and by results. Every backward pass must match finite differences. The trained CNN must reach at least 90% accuracy on 400 unseen test pictures, and it must lead the MLP by at least 20 points. Shifting every test picture by a few pixels must not drop its accuracy by more than 12 points.

## What you will use

- [Receptive field and output size](/lessons/037-convolution-and-receptive-field) tell you how many input pixels one unit depends on.
- [Convolution as one matrix multiply](/lessons/038-im2col-and-convolution-backward) gives you `im2col` and a fast forward pass.
- [Convolution backward](/lessons/038-im2col-and-convolution-backward) gives you the gradients for the input, the kernels, and the biases.
- [The residual block, forward and backward](/lessons/039-residual-networks) gives you the shortcut, here with convolutions in the branch.
- [Adam](/lessons/031-sgd-adam-and-schedules) is the optimiser that makes the small network train in a few epochs.

## Meet the data

`make_data(seed)` builds the pictures. Every one is 14 by 14 and has three layers, each planted on purpose.

The first layer is a background texture: a smooth set of stripes with a random direction, spacing, phase, and strength. Real metal has grain like this, and it matters because a defect must be told apart from texture, not from a flat grey. The second layer is pixel noise, a Gaussian with standard deviation 0.25 on every pixel, which stands for sensor noise. The third layer exists only in the defective half. It is either a bright scratch, a segment of 7 pixels at a random angle, or a dark dent, a Gaussian blob. The centre falls anywhere that keeps most of the defect inside the picture.

The labels are 1.0 for defective and 0.0 for good, exactly half each. You get 1000 training pictures and 400 test pictures. Figure 1 shows two good parts, a scratch, and a dent.

```figure
<svg viewBox="0 0 600 190" role="img" aria-label="Four 14 by 14 greyscale pictures: two good parts, one with a bright scratch, one with a dark dent">
<rect x="20" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="28" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="36" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="44" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="52" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.49"/>
<rect x="60" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="68" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.33"/>
<rect x="76" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="84" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="92" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="100" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="108" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.72"/>
<rect x="116" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.66"/>
<rect x="124" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="20" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.66"/>
<rect x="28" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.59"/>
<rect x="36" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="44" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="52" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="60" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.39"/>
<rect x="68" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.34"/>
<rect x="76" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.29"/>
<rect x="84" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.31"/>
<rect x="92" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="100" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.59"/>
<rect x="108" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.78"/>
<rect x="116" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="124" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.68"/>
<rect x="20" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.72"/>
<rect x="28" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="36" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="44" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="52" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="60" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.27"/>
<rect x="68" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.37"/>
<rect x="76" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="84" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="92" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="100" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="108" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.66"/>
<rect x="116" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.67"/>
<rect x="124" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="20" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.68"/>
<rect x="28" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.67"/>
<rect x="36" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="44" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="52" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="60" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="68" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="76" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.47"/>
<rect x="84" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="92" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.47"/>
<rect x="100" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="108" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="116" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.76"/>
<rect x="124" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.49"/>
<rect x="20" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="28" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="36" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="44" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.32"/>
<rect x="52" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="60" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.47"/>
<rect x="68" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.47"/>
<rect x="76" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="84" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.47"/>
<rect x="92" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.69"/>
<rect x="100" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="108" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.70"/>
<rect x="116" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="124" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="20" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="28" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="36" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="44" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.35"/>
<rect x="52" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.29"/>
<rect x="60" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.39"/>
<rect x="68" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="76" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="84" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="92" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="100" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.67"/>
<rect x="108" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="116" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.78"/>
<rect x="124" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="20" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="28" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="36" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.59"/>
<rect x="44" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.36"/>
<rect x="52" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="60" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="68" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.47"/>
<rect x="76" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.49"/>
<rect x="84" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="92" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="100" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="108" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.59"/>
<rect x="116" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="124" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="20" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.67"/>
<rect x="28" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="36" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.35"/>
<rect x="44" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.33"/>
<rect x="52" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.28"/>
<rect x="60" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.36"/>
<rect x="68" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="76" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="84" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="92" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="100" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.81"/>
<rect x="108" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="116" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.76"/>
<rect x="124" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="20" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="28" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.72"/>
<rect x="36" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.37"/>
<rect x="44" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="52" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.35"/>
<rect x="60" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.32"/>
<rect x="68" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="76" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.41"/>
<rect x="84" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="92" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.69"/>
<rect x="100" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="108" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="116" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="124" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="20" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="28" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="36" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="44" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.38"/>
<rect x="52" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="60" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="68" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.47"/>
<rect x="76" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="84" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="92" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="100" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="108" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="116" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="124" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="20" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="28" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.32"/>
<rect x="36" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.47"/>
<rect x="44" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.31"/>
<rect x="52" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="60" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.37"/>
<rect x="68" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="76" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="84" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="92" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="100" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="108" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="116" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="124" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="20" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.49"/>
<rect x="28" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.37"/>
<rect x="36" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.29"/>
<rect x="44" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="52" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="60" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.35"/>
<rect x="68" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="76" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="84" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="92" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="100" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.68"/>
<rect x="108" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="116" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="124" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="20" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="28" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="36" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="44" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.31"/>
<rect x="52" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.23"/>
<rect x="60" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.31"/>
<rect x="68" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="76" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="84" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="92" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="100" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.49"/>
<rect x="108" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="116" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="124" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.37"/>
<rect x="20" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="28" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.32"/>
<rect x="36" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="44" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.32"/>
<rect x="52" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.41"/>
<rect x="60" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="68" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="76" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.41"/>
<rect x="84" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="92" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.74"/>
<rect x="100" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="108" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="116" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="124" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.39"/>
<rect x="20" y="34" width="112" height="112" class="stroke-grid"/>
<text x="76" y="24" text-anchor="middle" class="fig-bold">good, label 0</text>
<text x="76" y="164" text-anchor="middle" class="fig-small">image 4</text>
<rect x="168" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.68"/>
<rect x="176" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="184" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.66"/>
<rect x="192" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.76"/>
<rect x="200" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="208" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="216" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="224" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="232" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.35"/>
<rect x="240" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="248" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="256" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="264" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.36"/>
<rect x="272" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.31"/>
<rect x="168" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="176" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="184" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="192" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.47"/>
<rect x="200" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="208" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="216" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="224" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.70"/>
<rect x="232" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="240" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="248" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="256" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.34"/>
<rect x="264" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.32"/>
<rect x="272" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.39"/>
<rect x="168" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="176" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.77"/>
<rect x="184" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.71"/>
<rect x="192" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="200" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="208" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="216" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="224" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="232" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="240" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="248" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="256" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.35"/>
<rect x="264" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="272" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.33"/>
<rect x="168" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="176" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="184" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="192" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="200" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.70"/>
<rect x="208" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="216" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="224" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="232" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="240" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.36"/>
<rect x="248" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.34"/>
<rect x="256" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="264" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="272" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.41"/>
<rect x="168" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="176" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="184" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="192" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="200" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.68"/>
<rect x="208" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="216" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="224" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.76"/>
<rect x="232" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="240" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.39"/>
<rect x="248" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="256" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.32"/>
<rect x="264" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.29"/>
<rect x="272" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="168" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="176" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="184" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="192" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.76"/>
<rect x="200" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="208" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="216" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.79"/>
<rect x="224" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="232" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="240" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="248" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="256" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.38"/>
<rect x="264" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="272" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.38"/>
<rect x="168" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="176" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="184" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="192" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.49"/>
<rect x="200" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.70"/>
<rect x="208" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.68"/>
<rect x="216" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="224" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="232" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.66"/>
<rect x="240" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="248" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.35"/>
<rect x="256" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.37"/>
<rect x="264" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.32"/>
<rect x="272" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.34"/>
<rect x="168" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.59"/>
<rect x="176" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="184" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.68"/>
<rect x="192" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="200" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="208" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="216" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.59"/>
<rect x="224" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="232" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="240" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.95"/>
<rect x="248" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="256" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="264" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.29"/>
<rect x="272" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="168" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.59"/>
<rect x="176" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="184" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="192" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="200" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.75"/>
<rect x="208" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.71"/>
<rect x="216" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.70"/>
<rect x="224" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.74"/>
<rect x="232" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.75"/>
<rect x="240" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.95"/>
<rect x="248" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.73"/>
<rect x="256" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="264" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="272" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="168" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="176" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="184" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="192" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="200" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="208" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.66"/>
<rect x="216" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.73"/>
<rect x="224" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.68"/>
<rect x="232" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.59"/>
<rect x="240" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.87"/>
<rect x="248" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.84"/>
<rect x="256" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="264" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="272" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.33"/>
<rect x="168" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.59"/>
<rect x="176" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="184" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="192" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="200" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="208" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="216" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.71"/>
<rect x="224" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="232" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="240" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="248" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.91"/>
<rect x="256" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.82"/>
<rect x="264" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="272" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.37"/>
<rect x="168" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.36"/>
<rect x="176" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.38"/>
<rect x="184" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="192" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="200" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.68"/>
<rect x="208" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.66"/>
<rect x="216" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="224" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.76"/>
<rect x="232" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.71"/>
<rect x="240" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="248" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.89"/>
<rect x="256" y="122" width="8" height="8" class="fill-ink" fill-opacity="1.00"/>
<rect x="264" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="272" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.41"/>
<rect x="168" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.47"/>
<rect x="176" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="184" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="192" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.67"/>
<rect x="200" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="208" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="216" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="224" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="232" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.66"/>
<rect x="240" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="248" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.70"/>
<rect x="256" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.90"/>
<rect x="264" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="272" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="168" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="176" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.47"/>
<rect x="184" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="192" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="200" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="208" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="216" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="224" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.73"/>
<rect x="232" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.70"/>
<rect x="240" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="248" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="256" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.76"/>
<rect x="264" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.80"/>
<rect x="272" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="168" y="34" width="112" height="112" class="stroke-grid"/>
<text x="224" y="24" text-anchor="middle" class="fig-bold">scratch, label 1</text>
<text x="224" y="164" text-anchor="middle" class="fig-small">image 13</text>
<rect x="316" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="324" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.77"/>
<rect x="332" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.75"/>
<rect x="340" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="348" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="356" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.72"/>
<rect x="364" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="372" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="380" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.69"/>
<rect x="388" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="396" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.36"/>
<rect x="404" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="412" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="420" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="316" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.66"/>
<rect x="324" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.59"/>
<rect x="332" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="340" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="348" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.59"/>
<rect x="356" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="364" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.66"/>
<rect x="372" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="380" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.49"/>
<rect x="388" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.35"/>
<rect x="396" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="404" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="412" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.49"/>
<rect x="420" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="316" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.66"/>
<rect x="324" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="332" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="340" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="348" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="356" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="364" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.32"/>
<rect x="372" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="380" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="388" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.34"/>
<rect x="396" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.41"/>
<rect x="404" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.36"/>
<rect x="412" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.36"/>
<rect x="420" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.38"/>
<rect x="316" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="324" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="332" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="340" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.47"/>
<rect x="348" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.38"/>
<rect x="356" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.30"/>
<rect x="364" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.10"/>
<rect x="372" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.24"/>
<rect x="380" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.24"/>
<rect x="388" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.16"/>
<rect x="396" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.31"/>
<rect x="404" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.31"/>
<rect x="412" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.30"/>
<rect x="420" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.24"/>
<rect x="316" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="324" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="332" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="340" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.49"/>
<rect x="348" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="356" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.14"/>
<rect x="364" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.23"/>
<rect x="372" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.07"/>
<rect x="380" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.11"/>
<rect x="388" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="396" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.34"/>
<rect x="404" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.41"/>
<rect x="412" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.29"/>
<rect x="420" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="316" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="324" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.39"/>
<rect x="332" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="340" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.37"/>
<rect x="348" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.36"/>
<rect x="356" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.25"/>
<rect x="364" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.06"/>
<rect x="372" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.07"/>
<rect x="380" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.17"/>
<rect x="388" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.28"/>
<rect x="396" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.15"/>
<rect x="404" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="412" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.35"/>
<rect x="420" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.29"/>
<rect x="316" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="324" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="332" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.28"/>
<rect x="340" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="348" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.25"/>
<rect x="356" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.17"/>
<rect x="364" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.16"/>
<rect x="372" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.33"/>
<rect x="380" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.25"/>
<rect x="388" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.30"/>
<rect x="396" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.37"/>
<rect x="404" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.36"/>
<rect x="412" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.37"/>
<rect x="420" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.39"/>
<rect x="316" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.41"/>
<rect x="324" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="332" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.27"/>
<rect x="340" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.37"/>
<rect x="348" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="356" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.27"/>
<rect x="364" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.31"/>
<rect x="372" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.27"/>
<rect x="380" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.28"/>
<rect x="388" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="396" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.38"/>
<rect x="404" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="412" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="420" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="316" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.33"/>
<rect x="324" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.31"/>
<rect x="332" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.36"/>
<rect x="340" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.38"/>
<rect x="348" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.33"/>
<rect x="356" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="364" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="372" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="380" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="388" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="396" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="404" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="412" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="420" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="316" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="324" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="332" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="340" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.33"/>
<rect x="348" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.41"/>
<rect x="356" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="364" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="372" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="380" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="388" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="396" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="404" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="412" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="420" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.69"/>
<rect x="316" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.35"/>
<rect x="324" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="332" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="340" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="348" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="356" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="364" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="372" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="380" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="388" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="396" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="404" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="412" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="420" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.71"/>
<rect x="316" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="324" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="332" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.71"/>
<rect x="340" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="348" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.67"/>
<rect x="356" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="364" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="372" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="380" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="388" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.69"/>
<rect x="396" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="404" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="412" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.69"/>
<rect x="420" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.78"/>
<rect x="316" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.49"/>
<rect x="324" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="332" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="340" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="348" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="356" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="364" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.74"/>
<rect x="372" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.72"/>
<rect x="380" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="388" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="396" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="404" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="412" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="420" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.49"/>
<rect x="316" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.57"/>
<rect x="324" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.59"/>
<rect x="332" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="340" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="348" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.70"/>
<rect x="356" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="364" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="372" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.69"/>
<rect x="380" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="388" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="396" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.66"/>
<rect x="404" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="412" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="420" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="316" y="34" width="112" height="112" class="stroke-grid"/>
<text x="372" y="24" text-anchor="middle" class="fig-bold">dent, label 1</text>
<text x="372" y="164" text-anchor="middle" class="fig-small">image 0</text>
<rect x="464" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="472" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.69"/>
<rect x="480" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.67"/>
<rect x="488" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="496" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.73"/>
<rect x="504" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="512" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.29"/>
<rect x="520" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="528" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.31"/>
<rect x="536" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.34"/>
<rect x="544" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="552" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.35"/>
<rect x="560" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.33"/>
<rect x="568" y="34" width="8" height="8" class="fill-ink" fill-opacity="0.49"/>
<rect x="464" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="472" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="480" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.68"/>
<rect x="488" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="496" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="504" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="512" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.59"/>
<rect x="520" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="528" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="536" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="544" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.38"/>
<rect x="552" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.27"/>
<rect x="560" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.34"/>
<rect x="568" y="42" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="464" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.37"/>
<rect x="472" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.69"/>
<rect x="480" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.66"/>
<rect x="488" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="496" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="504" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="512" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="520" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="528" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="536" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="544" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.31"/>
<rect x="552" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.33"/>
<rect x="560" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.38"/>
<rect x="568" y="50" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="464" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.66"/>
<rect x="472" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="480" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="488" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="496" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.67"/>
<rect x="504" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="512" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="520" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.34"/>
<rect x="528" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="536" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="544" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.38"/>
<rect x="552" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.20"/>
<rect x="560" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.49"/>
<rect x="568" y="58" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="464" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="472" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.67"/>
<rect x="480" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="488" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.67"/>
<rect x="496" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.73"/>
<rect x="504" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="512" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.75"/>
<rect x="520" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="528" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="536" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.34"/>
<rect x="544" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.41"/>
<rect x="552" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.27"/>
<rect x="560" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.31"/>
<rect x="568" y="66" width="8" height="8" class="fill-ink" fill-opacity="0.36"/>
<rect x="464" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="472" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.61"/>
<rect x="480" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.71"/>
<rect x="488" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="496" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="504" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="512" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="520" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="528" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="536" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.37"/>
<rect x="544" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.41"/>
<rect x="552" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.36"/>
<rect x="560" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.35"/>
<rect x="568" y="74" width="8" height="8" class="fill-ink" fill-opacity="0.37"/>
<rect x="464" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.68"/>
<rect x="472" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.73"/>
<rect x="480" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="488" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="496" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="504" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.41"/>
<rect x="512" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="520" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="528" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="536" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.27"/>
<rect x="544" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="552" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="560" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="568" y="82" width="8" height="8" class="fill-ink" fill-opacity="0.33"/>
<rect x="464" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="472" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="480" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.76"/>
<rect x="488" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="496" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="504" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.70"/>
<rect x="512" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.74"/>
<rect x="520" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="528" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="536" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.22"/>
<rect x="544" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="552" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.39"/>
<rect x="560" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.22"/>
<rect x="568" y="90" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="464" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="472" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="480" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.59"/>
<rect x="488" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.69"/>
<rect x="496" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.64"/>
<rect x="504" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="512" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="520" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="528" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="536" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.47"/>
<rect x="544" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.29"/>
<rect x="552" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.28"/>
<rect x="560" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="568" y="98" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="464" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="472" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.69"/>
<rect x="480" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.59"/>
<rect x="488" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.68"/>
<rect x="496" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="504" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="512" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="520" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="528" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="536" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="544" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="552" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.35"/>
<rect x="560" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.25"/>
<rect x="568" y="106" width="8" height="8" class="fill-ink" fill-opacity="0.31"/>
<rect x="464" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.63"/>
<rect x="472" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="480" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.69"/>
<rect x="488" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="496" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="504" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.67"/>
<rect x="512" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.67"/>
<rect x="520" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.46"/>
<rect x="528" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="536" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.44"/>
<rect x="544" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.42"/>
<rect x="552" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.36"/>
<rect x="560" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.26"/>
<rect x="568" y="114" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="464" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.51"/>
<rect x="472" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="480" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.69"/>
<rect x="488" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="496" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.56"/>
<rect x="504" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="512" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.55"/>
<rect x="520" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.49"/>
<rect x="528" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="536" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.34"/>
<rect x="544" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="552" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.49"/>
<rect x="560" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.38"/>
<rect x="568" y="122" width="8" height="8" class="fill-ink" fill-opacity="0.30"/>
<rect x="464" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.53"/>
<rect x="472" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.62"/>
<rect x="480" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.58"/>
<rect x="488" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.73"/>
<rect x="496" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.48"/>
<rect x="504" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.73"/>
<rect x="512" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.41"/>
<rect x="520" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="528" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="536" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.50"/>
<rect x="544" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.34"/>
<rect x="552" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.29"/>
<rect x="560" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.34"/>
<rect x="568" y="130" width="8" height="8" class="fill-ink" fill-opacity="0.33"/>
<rect x="464" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.43"/>
<rect x="472" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.65"/>
<rect x="480" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.45"/>
<rect x="488" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.73"/>
<rect x="496" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.60"/>
<rect x="504" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.68"/>
<rect x="512" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.52"/>
<rect x="520" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.47"/>
<rect x="528" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.54"/>
<rect x="536" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.40"/>
<rect x="544" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.28"/>
<rect x="552" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.37"/>
<rect x="560" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.26"/>
<rect x="568" y="138" width="8" height="8" class="fill-ink" fill-opacity="0.36"/>
<rect x="464" y="34" width="112" height="112" class="stroke-grid"/>
<text x="520" y="24" text-anchor="middle" class="fig-bold">good, label 0</text>
<text x="520" y="164" text-anchor="middle" class="fig-small">image 5</text>
</svg>
Figure 1. Denser ink means a higher pixel value. The texture and the noise appear on every picture, so a defect is a small change on top of a busy background, and its position differs from picture to picture. The scratch and the dent do not share any pixel.
```

Two numbers set the scale of the problem. The scratch spans 7 pixels along either axis at most, which sets the least a unit must see to recognise one whole. The 1000 training pictures with 196 pixels each are small enough that a flat network can memorise them, which the project later shows it does.

## Plan the build

Figure 2 draws the pipeline. A picture goes through a stem convolution that turns one grey channel into 8 feature channels, through a residual block that refines them, through a global average that throws away position, and through a linear head that turns 8 numbers into one score for "defective".

```figure
<svg viewBox="0 0 612 250" role="img" aria-label="Pipeline: picture, stem convolution, residual block with a shortcut, global average pooling, linear head, loss, with gradients flowing back">
<defs><marker id="p096-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker>
<marker id="p096-back" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fill-b"/></marker></defs>
<rect x="14" y="70" width="84" height="76" rx="6" class="fill-soft-a stroke-a"/>
<text x="56" y="88" text-anchor="middle" class="fig-bold fig-small">picture</text>
<text x="56" y="106" text-anchor="middle" class="fig-small">1 x 14 x 14</text>
<line x1="99" y1="108" x2="113" y2="108" class="stroke-ink" marker-end="url(#p096-arrow)"/>
<rect x="114" y="70" width="84" height="76" rx="6" class="fill-soft-a stroke-a"/>
<text x="156" y="88" text-anchor="middle" class="fig-bold fig-small">stem conv</text>
<text x="156" y="106" text-anchor="middle" class="fig-small">3x3, 8 filters</text>
<text x="156" y="120" text-anchor="middle" class="fig-small">ReLU</text>
<text x="156" y="134" text-anchor="middle" class="fig-small">8 x 14 x 14</text>
<text x="156" y="164" text-anchor="middle" class="fig-small">sees 3</text>
<line x1="199" y1="108" x2="213" y2="108" class="stroke-ink" marker-end="url(#p096-arrow)"/>
<rect x="214" y="70" width="84" height="76" rx="6" class="fill-soft-c stroke-c"/>
<text x="256" y="88" text-anchor="middle" class="fig-bold fig-small">residual block</text>
<text x="256" y="106" text-anchor="middle" class="fig-small">conv, ReLU,</text>
<text x="256" y="120" text-anchor="middle" class="fig-small">conv, then add</text>
<text x="256" y="134" text-anchor="middle" class="fig-small">the input</text>
<text x="256" y="164" text-anchor="middle" class="fig-small">sees 7</text>
<line x1="299" y1="108" x2="313" y2="108" class="stroke-ink" marker-end="url(#p096-arrow)"/>
<rect x="314" y="70" width="84" height="76" rx="6" class="fill-soft-a stroke-a"/>
<text x="356" y="88" text-anchor="middle" class="fig-bold fig-small">pool</text>
<text x="356" y="106" text-anchor="middle" class="fig-small">ReLU, then</text>
<text x="356" y="120" text-anchor="middle" class="fig-small">global average</text>
<text x="356" y="134" text-anchor="middle" class="fig-small">8 numbers</text>
<line x1="399" y1="108" x2="413" y2="108" class="stroke-ink" marker-end="url(#p096-arrow)"/>
<rect x="414" y="70" width="84" height="76" rx="6" class="fill-soft-b stroke-b"/>
<text x="456" y="88" text-anchor="middle" class="fig-bold fig-small">linear head</text>
<text x="456" y="106" text-anchor="middle" class="fig-small">8 weights</text>
<text x="456" y="120" text-anchor="middle" class="fig-small">and a bias</text>
<text x="456" y="134" text-anchor="middle" class="fig-small">1 logit</text>
<line x1="499" y1="108" x2="513" y2="108" class="stroke-ink" marker-end="url(#p096-arrow)"/>
<rect x="514" y="70" width="84" height="76" rx="6" class="fill-soft-d stroke-d"/>
<text x="556" y="88" text-anchor="middle" class="fig-bold fig-small">loss</text>
<text x="556" y="106" text-anchor="middle" class="fig-small">binary</text>
<text x="556" y="120" text-anchor="middle" class="fig-small">cross-entropy</text>
<text x="556" y="134" text-anchor="middle" class="fig-small">1 number</text>
<path d="M228,70 L228,48 L284,48 L284,68" class="stroke-c dashed" marker-end="url(#p096-arrow)"/>
<text x="256" y="40" text-anchor="middle" class="fig-small">shortcut</text>
<path d="M550,180 L56,180" class="stroke-b" marker-end="url(#p096-back)"/>
<text x="306" y="204" text-anchor="middle" class="fig-small">gradients flow back: head, gap_backward, res_block_backward, conv_backward</text>
<text x="306" y="232" text-anchor="middle" class="fig-small">a unit after the residual block sees a 7 by 7 patch, as wide as a scratch is long</text>
</svg>
Figure 2. The forward pass runs left to right. The shortcut around the residual block carries the input past two convolutions. The backward pass retraces the same boxes from the loss to the picture, and each box reuses an earlier chapter's gradient.
```

The stages come in this order for a reason. Convolutions come first because they detect local patterns wherever those occur. The residual block comes after the stem because it adds reach (two more 3 by 3 layers) while the shortcut keeps the stem's signal from being lost. Global average pooling comes next because the label does not depend on where the defect is, only on whether one exists, so you average the position away. The linear head comes last because it only has to weigh eight feature averages against each other.

### Can the network see a whole scratch?

The receptive field answers that. A stack of three 3 by 3 layers at stride 1 sees $1 + 2 + 2 + 2 = 7$ pixels, which is the length of a scratch, so some unit has the entire scratch inside its window. A two-layer stack would see 5 pixels and would have to guess the rest of the scratch from a part. Here, the stem and the two convolutions in the residual block are the three layers. The rule from [the receptive-field chapter](/lessons/037-convolution-and-receptive-field) is that each layer adds `(kernel_size - 1)` times the jump, where the jump is the product of all earlier strides:

$$
r \leftarrow r + (k - 1)\,j, \qquad j \leftarrow j \cdot s
$$

Read it aloud: the field grows by the kernel's overhang, measured in input pixels, and the jump grows by the layer's stride. With all strides equal to 1, the jump stays 1 and the field grows by 2 per 3 by 3 layer.

```python
r, jump = 1, 1
for k, s in [(3, 1), (3, 1), (3, 1)]:
    r += (k - 1) * jump
    jump *= s
```

### Why a convolution handles shifts and an MLP does not

Take a one-dimensional row with a single spike of 3, and a kernel of $(1, 2)$. For the row `[0, 3, 0, 0, 0]` the convolution gives `[3, 6, 0, 0]`. Move the spike two cells to the right, `[0, 0, 0, 3, 0]`, and the output is `[0, 0, 3, 6]`. The same pair of numbers appears, two cells over. The average of the output is $9/4 = 2.25$ in both cases. A convolution is therefore **translation-equivariant**: shifting the input shifts the output. An average over positions turns equivariance into **translation invariance**: the average ignores where the pattern sits. A flat network has a separate weight for each pixel, so a scratch in an unseen place meets weights that never saw one.

> [!KEY]
> Convolution shares one small set of weights across all positions, so a pattern learned in one place is detected in every place. Global average pooling then drops the position. Together they build in the one thing this problem needs: the label does not depend on where the defect is.

## Build it step by step

> [!RECAP]
> Hold these points before you write code:
>
> - A unit's receptive field grows by `(k - 1) * jump` per layer, and the jump multiplies by the stride.
> - `im2col` returns `(N, C * k * k, H_out * W_out)`, and the kernels flatten as `w.reshape(C_out, -1)`.
> - `conv_backward` is two products and one `col2im`: `dw` from `dY` and `col`, `dx` from `w` and `dY`.
> - A residual block's `dx` is `dY` plus the branch's gradient, and the ReLU mask uses the pre-activation `h1 > 0`.
> - Global average pooling divides the gradient by `H * W` and sends the same share to every position.

The starter ships the data function, `conv_output_shape`, `col2im`, `relu`, `bce_with_logits`, `init_cnn`, the training loop `fit`, and the MLP baseline, because earlier chapters wrote them or they carry no further idea. You write the following eleven functions. The checks test them in this order.

1. **`receptive_field(layers)`.** `layers` is a list of `(kernel_size, stride)` pairs. Return the integer receptive field of the last layer. Keep a field `r` and a jump `j`. The usual mistake is to update `j` before `r`, which uses the wrong jump for the layer.
2. **`im2col(x, kernel_size, stride, padding)`.** Pad with `np.pad`, then loop over the `k * k` kernel offsets only. Each offset is one strided slice that fills that offset's entry for every output position. Reshape to `(N, C * k * k, H_out * W_out)`. The usual mistake is a loop over output positions, which works but is far too slow.
3. **`conv_forward(x, w, b, stride, padding)`.** Multiply `w.reshape(C_out, -1)` into the columns of every image, with `np.einsum("oc,ncl->nol", ...)`, add the bias, and reshape to `(N, C_out, H_out, W_out)`. The usual mistake is a bias shaped `(C_out,)` that does not broadcast over the position axis. Use `b[None, :, None]`.
4. **`conv_backward(dY, x, w, stride, padding)`.** Return `(dx, dw, db)`. The bias gradient sums `dY` over images and positions. The kernel gradient is `dY` times `col`. The input gradient scatters `w.T` times `dY` back with `col2im`. The usual mistake is to forget that overlapping windows add, which `col2im` handles, so never write the scatter with `=`.
5. **`res_block_forward(x, params)`.** Compute `conv_forward(x, W1, b1, padding=1)`, a ReLU, a second convolution, and add `x`. Padding 1 keeps the size, so the addition is legal. Return the output and a cache. The usual mistake is a ReLU after the addition, which makes the shortcut no longer an identity.
6. **`res_block_backward(dY, cache)`.** Send `dY` back through the second convolution, the ReLU mask `h1 > 0`, and the first convolution, then add `dY` for the shortcut. Return `(dx, grads)`. The usual mistake is to return only the branch term.
7. **`gap_forward(x)`.** Return the mean over the two spatial axes, shape `(N, C)`.
8. **`gap_backward(dfeat, x_shape)`.** Every position gets `dfeat / (H * W)`. Broadcast it to `x_shape`. The usual mistake is to forget the division, which makes the gradient 196 times too large.
9. **`cnn_forward(params, X)`.** Chain the stem convolution, ReLU, residual block, ReLU, pooling, and head `feat @ Wo + bo[0]`. Return the logits, shape `(N,)`, and a cache. Store pre-activations, because the masks need them.
10. **`cnn_backward(params, cache, dlogits)`.** Retrace the chain with your earlier backward functions. Return a dict with exactly the keys of `params`. The usual mistake is a head bias gradient shaped `()` instead of `(1,)`.
11. **`adam_step(param, grad, state, ...)`.** Update `state["m"]`, `state["v"]`, and `state["t"]` in place, correct their bias, and return the updated parameter. The usual mistake is to correct the bias with the previous step count.

Check the gradients first. A convolution's gradient is the likeliest place for an off-by-one in a pad or a stride, and finite differences find it in seconds. The idea is the same as in [the residual-block chapter](/lessons/039-residual-networks): nudge one number, rerun the forward pass, and compare the change in loss with your formula.

> [!WARNING]
> Finite differences only work away from a ReLU corner. If a pre-activation lies within the nudge of zero, the loss has a kink there and the check disagrees with a correct gradient. Use random inputs, and rerun with another seed before you doubt a formula that is otherwise right.

## Read your results

The reference solution trains 20 epochs with Adam at a learning rate of 0.01, a batch of 50, and 8 filters, on seed 0. Run yours and compare with these numbers.

| Quantity | CNN | MLP |
|---|---|---|
| Parameters | 1257 | 1189 |
| Training accuracy | 0.984 | 0.989 |
| Test accuracy | 0.975 | 0.618 |
| Training loss, last epoch | 0.063 | 0.105 |

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Bar chart of training and test accuracy for the CNN and the MLP, and the CNN on shifted test pictures">
<line x1="60" y1="200" x2="580" y2="200" class="stroke-muted"/>
<line x1="60" y1="40" x2="60" y2="200" class="stroke-muted"/>
<line x1="60" y1="200.0" x2="580" y2="200.0" class="stroke-grid"/><text x="52" y="204.0" text-anchor="end" class="fig-small">0.50</text>
<line x1="60" y1="120.0" x2="580" y2="120.0" class="stroke-grid"/><text x="52" y="124.0" text-anchor="end" class="fig-small">0.75</text>
<line x1="60" y1="40.0" x2="580" y2="40.0" class="stroke-grid"/><text x="52" y="44.0" text-anchor="end" class="fig-small">1.00</text>
<rect x="80" y="45.1" width="60" height="154.9" class="fill-a"/>
<text x="110" y="39.1" text-anchor="middle" class="fig-small">0.984</text>
<text x="110" y="218" text-anchor="middle" class="fig-small">CNN train</text>
<rect x="180" y="48.0" width="60" height="152.0" class="fill-b"/>
<text x="210" y="42.0" text-anchor="middle" class="fig-small">0.975</text>
<text x="210" y="218" text-anchor="middle" class="fig-small">CNN test</text>
<rect x="280" y="72.8" width="60" height="127.2" class="fill-b"/>
<text x="310" y="66.8" text-anchor="middle" class="fig-small">0.897</text>
<text x="310" y="218" text-anchor="middle" class="fig-small">CNN test</text>
<text x="310" y="233" text-anchor="middle" class="fig-small">shifted</text>
<rect x="380" y="43.5" width="60" height="156.5" class="fill-a"/>
<text x="410" y="37.5" text-anchor="middle" class="fig-small">0.989</text>
<text x="410" y="218" text-anchor="middle" class="fig-small">MLP train</text>
<rect x="480" y="162.4" width="60" height="37.6" class="fill-b"/>
<text x="510" y="156.4" text-anchor="middle" class="fig-small">0.618</text>
<text x="510" y="218" text-anchor="middle" class="fig-small">MLP test</text>
</svg>
Figure 3. Accuracy on the training and test pictures. The MLP reaches 0.989 on pictures it has seen and 0.618 on pictures it has not, so it memorised. The CNN loses almost nothing between the two, and it loses about eight points when every test picture is shifted.
```

Read the table from the plant's point of view. With the CNN, 97.5% of 400 test parts are called correctly, and in the reference run it flags 97.5% of the defective parts and wrongly flags 2.5% of the good ones. The MLP is nearly as accurate as the CNN on the pictures it trained on and only 12 points better than a coin flip on pictures it has not seen. It has 1189 parameters for 1000 pictures, so it can store the answers, and each of its six hidden units covers one arrangement of pixels instead of one kind of defect. Weight sharing is the reason the CNN does not do the same: its 1257 parameters are spent on 8 small filters that are reused at every position, so they can only describe local patterns.

The shift test rolls every test picture by `(2, -3)` pixels with `np.roll`. The CNN's accuracy goes from 0.975 to 0.897 and its predicted probabilities move by 0.085 on average. It does not drop to the MLP's level, which shows that it learned the defect and not its position. It does not stay at 0.975 either, because `np.roll` wraps pixels around the border, which cuts the stripe texture at a seam and moves some defects across the edge. The zero padding at the border, which treats the outside as flat grey, also breaks exact equivariance. A drop of a few points is the honest price of those two effects.

What a wrong number usually points to:

- **A gradient check fails by a factor close to 196.** `gap_backward` forgot to divide by `H * W`.
- **Training accuracy stuck near 0.5 and the loss near 0.69.** A gradient has the wrong sign or shape, or `dY` is missing from the residual block's `dx`.
- **Test accuracy of the CNN far below its training accuracy.** The images and labels were shuffled separately, or you trained too long on a few hundred pictures. Compare the training and test curves before you change the model.
- **`conv_forward` agrees with the loop when the padding is 0 and disagrees when it is 1.** `im2col` pads the wrong sides, or `conv_output_shape` is called without the padding.

## Take it further

- Replace global average pooling with global max pooling. This tells you whether a defect is better detected by its average presence or by its strongest response.
- Give the MLP 32 hidden units. It gets 6337 parameters, so you learn how far extra capacity closes the gap and how much it memorises instead.
- Add a second residual block and use `receptive_field` to predict what the unit sees. This tells you whether depth pays off when the scratch is only 7 pixels long.
- Train on pictures where the defects sit only in the top half, then test on defects in the bottom half. This shows how far translation invariance holds beyond the training positions.
- Add a stride-2 convolution in front of the pooling and recompute the receptive field. This shows how stride makes units see more with fewer positions to compute.

## The exercise

You write `receptive_field`, `im2col`, `conv_forward`, `conv_backward`, `res_block_forward`, `res_block_backward`, `gap_forward`, `gap_backward`, `cnn_forward`, `cnn_backward`, and `adam_step`. The checks test them in that order. `receptive_field` must return 3, 5, 7, and 4 for the stacks the tests give, and the stem plus the residual block must see at least a scratch's length. `im2col` must lay out windows in the stated order, and `conv_forward` must match a plain loop for four stride and padding settings. `conv_backward`, `res_block_backward`, `gap_backward`, and the whole network's `cnn_backward` must match finite differences for every input and parameter. `adam_step` must follow the bias-corrected update and minimise a bowl.

The last three checks run the pipeline. They train the CNN and the MLP on `make_data(seed=0)`, require at least 90% test accuracy from the CNN and a lead of at least 20 points over the MLP with a similar parameter count, and require that shifting every test picture by two different offsets leaves accuracy and probabilities close to where they were. Running the file prints the same comparison with your functions.
