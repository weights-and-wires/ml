---
title: "Splitting a compute budget"
slug: 116-compute-optimal-split
part: "Part 7 — Large language models"
summary: With a fixed compute budget, size and data trade off, and the lowest loss sits where an extra bit of compute buys the same on each side.
minutes: 40
gpu: false
objectives:
  - Explain why, at a fixed budget, the loss along the model-versus-data trade-off is a U and not a slope.
  - Derive the best model size and token count by equating the marginal gain of size and of data, and compute them for a budget.
  - Say when the best ratio of tokens to parameters is constant, and when it drifts.
  - Explain why heavy serving pushes a team toward a smaller model trained on more tokens.
prereqs:
  - 115-counting-parameters-and-flops
  - 114-loss-from-size-and-data
exercise: 116-compute-optimal-split
---

# Splitting a compute budget

You have $10^{20}$ flops, which is about six days on a chip that does 200 trillion per second. You can train a huge model for a short time, a tiny model for a long time, or something in between. The loss law from the last chapter says a bigger model helps, and more text helps too, but it never said how to choose when the budget is fixed.

The obvious plan is to build the biggest model the budget allows. It fails because compute is the product of model size and tokens: $6ND$. A bigger $N$ at fixed compute leaves fewer tokens, the data penalty rises, and at some point the rise outweighs the fall in the size penalty. The reverse plan, a small model on mountains of text, fails the same way from the other side.

This chapter finds the split in between where the loss is lowest. You will see why there is exactly one such point, derive where it is, and find that for these constants it comes out at about 20 tokens per parameter.

## What you already know

- The loss is $E + A/N^{\alpha} + B/D^{\beta}$, a floor plus a size penalty and a data penalty.
- Training costs about $6ND$ flops, so for a fixed budget $C$ the tokens are $D = C/(6N)$.
- A power law falls by a fixed share for each doubling of its input.
- A fixed number of flops divided by a chip's speed gives seconds, and the chapter on counting a transformer's parameters and flops did this arithmetic.
- A function has its lowest point where it stops falling and starts rising.

## The picture: a U along the budget line

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Size penalty falling, data penalty rising, and their sum making a U shape along a line of fixed compute, with the lowest point at 1 billion parameters">
<line x1="70.0" y1="236.0" x2="510.0" y2="236.0" class="stroke-muted"/><line x1="70.0" y1="36.0" x2="70.0" y2="236.0" class="stroke-muted"/><line x1="120.9" y1="36.0" x2="120.9" y2="236.0" class="stroke-grid"/><text x="120.9" y="252.0" text-anchor="middle" class="fig-small">100M</text><line x1="290.0" y1="36.0" x2="290.0" y2="236.0" class="stroke-grid"/><text x="290.0" y="252.0" text-anchor="middle" class="fig-small">1B</text><line x1="459.1" y1="36.0" x2="459.1" y2="236.0" class="stroke-grid"/><text x="459.1" y="252.0" text-anchor="middle" class="fig-small">10B</text><line x1="70.0" y1="236.0" x2="510.0" y2="236.0" class="stroke-grid"/><text x="64.0" y="240.0" text-anchor="end" class="fig-small">0</text><line x1="70.0" y1="156.0" x2="510.0" y2="156.0" class="stroke-grid"/><text x="64.0" y="160.0" text-anchor="end" class="fig-small">0.5</text><line x1="70.0" y1="76.0" x2="510.0" y2="76.0" class="stroke-grid"/><text x="64.0" y="80.0" text-anchor="end" class="fig-small">1</text>
<polyline points="70.0,81.6 71.5,82.7 72.9,83.7 74.4,84.8 75.9,85.8 77.4,86.8 78.8,87.8 80.3,88.8 81.8,89.8 83.2,90.8 84.7,91.8 86.2,92.8 87.7,93.8 89.1,94.7 90.6,95.7 92.1,96.6 93.5,97.6 95.0,98.5 96.5,99.5 98.0,100.4 99.4,101.3 100.9,102.2 102.4,103.1 103.8,104.0 105.3,104.9 106.8,105.8 108.3,106.7 109.7,107.6 111.2,108.4 112.7,109.3 114.1,110.2 115.6,111.0 117.1,111.9 118.6,112.7 120.0,113.6 121.5,114.4 123.0,115.2 124.4,116.0 125.9,116.9 127.4,117.7 128.9,118.5 130.3,119.3 131.8,120.1 133.3,120.8 134.7,121.6 136.2,122.4 137.7,123.2 139.2,123.9 140.6,124.7 142.1,125.5 143.6,126.2 145.1,126.9 146.5,127.7 148.0,128.4 149.5,129.2 150.9,129.9 152.4,130.6 153.9,131.3 155.4,132.0 156.8,132.7 158.3,133.4 159.8,134.1 161.2,134.8 162.7,135.5 164.2,136.2 165.7,136.9 167.1,137.5 168.6,138.2 170.1,138.9 171.5,139.5 173.0,140.2 174.5,140.8 176.0,141.5 177.4,142.1 178.9,142.8 180.4,143.4 181.8,144.0 183.3,144.7 184.8,145.3 186.3,145.9 187.7,146.5 189.2,147.1 190.7,147.7 192.1,148.3 193.6,148.9 195.1,149.5 196.6,150.1 198.0,150.7 199.5,151.2 201.0,151.8 202.4,152.4 203.9,153.0 205.4,153.5 206.9,154.1 208.3,154.6 209.8,155.2 211.3,155.7 212.7,156.3 214.2,156.8 215.7,157.4 217.2,157.9 218.6,158.4 220.1,159.0 221.6,159.5 223.0,160.0 224.5,160.5 226.0,161.0 227.5,161.5 228.9,162.0 230.4,162.5 231.9,163.0 233.3,163.5 234.8,164.0 236.3,164.5 237.8,165.0 239.2,165.5 240.7,166.0 242.2,166.4 243.6,166.9 245.1,167.4 246.6,167.8 248.1,168.3 249.5,168.8 251.0,169.2 252.5,169.7 253.9,170.1 255.4,170.6 256.9,171.0 258.4,171.5 259.8,171.9 261.3,172.3 262.8,172.8 264.2,173.2 265.7,173.6 267.2,174.0 268.7,174.5 270.1,174.9 271.6,175.3 273.1,175.7 274.5,176.1 276.0,176.5 277.5,176.9 279.0,177.3 280.4,177.7 281.9,178.1 283.4,178.5 284.8,178.9 286.3,179.3 287.8,179.7 289.3,180.1 290.7,180.4 292.2,180.8 293.7,181.2 295.2,181.6 296.6,181.9 298.1,182.3 299.6,182.7 301.0,183.0 302.5,183.4 304.0,183.8 305.5,184.1 306.9,184.5 308.4,184.8 309.9,185.2 311.3,185.5 312.8,185.8 314.3,186.2 315.8,186.5 317.2,186.9 318.7,187.2 320.2,187.5 321.6,187.9 323.1,188.2 324.6,188.5 326.1,188.8 327.5,189.1 329.0,189.5 330.5,189.8 331.9,190.1 333.4,190.4 334.9,190.7 336.4,191.0 337.8,191.3 339.3,191.6 340.8,191.9 342.2,192.2 343.7,192.5 345.2,192.8 346.7,193.1 348.1,193.4 349.6,193.7 351.1,194.0 352.5,194.3 354.0,194.6 355.5,194.8 357.0,195.1 358.4,195.4 359.9,195.7 361.4,195.9 362.8,196.2 364.3,196.5 365.8,196.8 367.3,197.0 368.7,197.3 370.2,197.5 371.7,197.8 373.1,198.1 374.6,198.3 376.1,198.6 377.6,198.8 379.0,199.1 380.5,199.3 382.0,199.6 383.4,199.8 384.9,200.1 386.4,200.3 387.9,200.6 389.3,200.8 390.8,201.0 392.3,201.3 393.7,201.5 395.2,201.8 396.7,202.0 398.2,202.2 399.6,202.4 401.1,202.7 402.6,202.9 404.0,203.1 405.5,203.3 407.0,203.6 408.5,203.8 409.9,204.0 411.4,204.2 412.9,204.4 414.3,204.7 415.8,204.9 417.3,205.1 418.8,205.3 420.2,205.5 421.7,205.7 423.2,205.9 424.6,206.1 426.1,206.3 427.6,206.5 429.1,206.7 430.5,206.9 432.0,207.1 433.5,207.3 434.9,207.5 436.4,207.7 437.9,207.9 439.4,208.1 440.8,208.3 442.3,208.5 443.8,208.6 445.3,208.8 446.7,209.0 448.2,209.2 449.7,209.4 451.1,209.6 452.6,209.7 454.1,209.9 455.6,210.1 457.0,210.3 458.5,210.5 460.0,210.6 461.4,210.8 462.9,211.0 464.4,211.1 465.9,211.3 467.3,211.5 468.8,211.6 470.3,211.8 471.7,212.0 473.2,212.1 474.7,212.3 476.2,212.5 477.6,212.6 479.1,212.8 480.6,212.9 482.0,213.1 483.5,213.2 485.0,213.4 486.5,213.6 487.9,213.7 489.4,213.9 490.9,214.0 492.3,214.2 493.8,214.3 495.3,214.5 496.8,214.6 498.2,214.7 499.7,214.9 501.2,215.0 502.6,215.2 504.1,215.3 505.6,215.5 507.1,215.6 508.5,215.7 510.0,215.9" fill="none" class="stroke-a" stroke-width="2"/>
<polyline points="70.0,216.0 71.5,215.9 72.9,215.7 74.4,215.6 75.9,215.5 77.4,215.3 78.8,215.2 80.3,215.0 81.8,214.9 83.2,214.7 84.7,214.6 86.2,214.5 87.7,214.3 89.1,214.2 90.6,214.0 92.1,213.9 93.5,213.7 95.0,213.6 96.5,213.4 98.0,213.2 99.4,213.1 100.9,212.9 102.4,212.8 103.8,212.6 105.3,212.5 106.8,212.3 108.3,212.1 109.7,212.0 111.2,211.8 112.7,211.6 114.1,211.5 115.6,211.3 117.1,211.1 118.6,211.0 120.0,210.8 121.5,210.6 123.0,210.5 124.4,210.3 125.9,210.1 127.4,209.9 128.9,209.7 130.3,209.6 131.8,209.4 133.3,209.2 134.7,209.0 136.2,208.8 137.7,208.7 139.2,208.5 140.6,208.3 142.1,208.1 143.6,207.9 145.1,207.7 146.5,207.5 148.0,207.3 149.5,207.1 150.9,206.9 152.4,206.7 153.9,206.5 155.4,206.3 156.8,206.1 158.3,205.9 159.8,205.7 161.2,205.5 162.7,205.3 164.2,205.1 165.7,204.9 167.1,204.7 168.6,204.4 170.1,204.2 171.5,204.0 173.0,203.8 174.5,203.6 176.0,203.4 177.4,203.1 178.9,202.9 180.4,202.7 181.8,202.5 183.3,202.2 184.8,202.0 186.3,201.8 187.7,201.5 189.2,201.3 190.7,201.1 192.1,200.8 193.6,200.6 195.1,200.3 196.6,200.1 198.0,199.8 199.5,199.6 201.0,199.3 202.4,199.1 203.9,198.8 205.4,198.6 206.9,198.3 208.3,198.1 209.8,197.8 211.3,197.6 212.7,197.3 214.2,197.0 215.7,196.8 217.2,196.5 218.6,196.2 220.1,195.9 221.6,195.7 223.0,195.4 224.5,195.1 226.0,194.8 227.5,194.6 228.9,194.3 230.4,194.0 231.9,193.7 233.3,193.4 234.8,193.1 236.3,192.8 237.8,192.5 239.2,192.2 240.7,191.9 242.2,191.6 243.6,191.3 245.1,191.0 246.6,190.7 248.1,190.4 249.5,190.1 251.0,189.8 252.5,189.5 253.9,189.2 255.4,188.8 256.9,188.5 258.4,188.2 259.8,187.9 261.3,187.5 262.8,187.2 264.2,186.9 265.7,186.5 267.2,186.2 268.7,185.9 270.1,185.5 271.6,185.2 273.1,184.8 274.5,184.5 276.0,184.1 277.5,183.8 279.0,183.4 280.4,183.0 281.9,182.7 283.4,182.3 284.8,181.9 286.3,181.6 287.8,181.2 289.3,180.8 290.7,180.5 292.2,180.1 293.7,179.7 295.2,179.3 296.6,178.9 298.1,178.5 299.6,178.1 301.0,177.7 302.5,177.3 304.0,176.9 305.5,176.5 306.9,176.1 308.4,175.7 309.9,175.3 311.3,174.9 312.8,174.5 314.3,174.1 315.8,173.6 317.2,173.2 318.7,172.8 320.2,172.3 321.6,171.9 323.1,171.5 324.6,171.0 326.1,170.6 327.5,170.1 329.0,169.7 330.5,169.2 331.9,168.8 333.4,168.3 334.9,167.9 336.4,167.4 337.8,166.9 339.3,166.5 340.8,166.0 342.2,165.5 343.7,165.0 345.2,164.5 346.7,164.0 348.1,163.6 349.6,163.1 351.1,162.6 352.5,162.1 354.0,161.5 355.5,161.0 357.0,160.5 358.4,160.0 359.9,159.5 361.4,159.0 362.8,158.4 364.3,157.9 365.8,157.4 367.3,156.8 368.7,156.3 370.2,155.8 371.7,155.2 373.1,154.7 374.6,154.1 376.1,153.5 377.6,153.0 379.0,152.4 380.5,151.8 382.0,151.3 383.4,150.7 384.9,150.1 386.4,149.5 387.9,148.9 389.3,148.3 390.8,147.7 392.3,147.1 393.7,146.5 395.2,145.9 396.7,145.3 398.2,144.7 399.6,144.0 401.1,143.4 402.6,142.8 404.0,142.1 405.5,141.5 407.0,140.9 408.5,140.2 409.9,139.5 411.4,138.9 412.9,138.2 414.3,137.6 415.8,136.9 417.3,136.2 418.8,135.5 420.2,134.8 421.7,134.1 423.2,133.4 424.6,132.7 426.1,132.0 427.6,131.3 429.1,130.6 430.5,129.9 432.0,129.2 433.5,128.4 434.9,127.7 436.4,127.0 437.9,126.2 439.4,125.5 440.8,124.7 442.3,124.0 443.8,123.2 445.3,122.4 446.7,121.6 448.2,120.9 449.7,120.1 451.1,119.3 452.6,118.5 454.1,117.7 455.6,116.9 457.0,116.1 458.5,115.2 460.0,114.4 461.4,113.6 462.9,112.7 464.4,111.9 465.9,111.0 467.3,110.2 468.8,109.3 470.3,108.5 471.7,107.6 473.2,106.7 474.7,105.8 476.2,104.9 477.6,104.0 479.1,103.1 480.6,102.2 482.0,101.3 483.5,100.4 485.0,99.5 486.5,98.5 487.9,97.6 489.4,96.7 490.9,95.7 492.3,94.7 493.8,93.8 495.3,92.8 496.8,91.8 498.2,90.8 499.7,89.8 501.2,88.9 502.6,87.8 504.1,86.8 505.6,85.8 507.1,84.8 508.5,83.8 510.0,82.7" fill="none" class="stroke-d" stroke-width="2"/>
<polyline points="70.0,61.7 71.5,62.6 72.9,63.5 74.4,64.4 75.9,65.2 77.4,66.1 78.8,67.0 80.3,67.9 81.8,68.7 83.2,69.6 84.7,70.4 86.2,71.2 87.7,72.1 89.1,72.9 90.6,73.7 92.1,74.5 93.5,75.3 95.0,76.1 96.5,76.9 98.0,77.6 99.4,78.4 100.9,79.2 102.4,79.9 103.8,80.6 105.3,81.4 106.8,82.1 108.3,82.8 109.7,83.6 111.2,84.3 112.7,85.0 114.1,85.7 115.6,86.3 117.1,87.0 118.6,87.7 120.0,88.4 121.5,89.0 123.0,89.7 124.4,90.3 125.9,91.0 127.4,91.6 128.9,92.2 130.3,92.8 131.8,93.4 133.3,94.0 134.7,94.6 136.2,95.2 137.7,95.8 139.2,96.4 140.6,97.0 142.1,97.5 143.6,98.1 145.1,98.7 146.5,99.2 148.0,99.7 149.5,100.3 150.9,100.8 152.4,101.3 153.9,101.8 155.4,102.4 156.8,102.9 158.3,103.4 159.8,103.8 161.2,104.3 162.7,104.8 164.2,105.3 165.7,105.7 167.1,106.2 168.6,106.7 170.1,107.1 171.5,107.5 173.0,108.0 174.5,108.4 176.0,108.8 177.4,109.3 178.9,109.7 180.4,110.1 181.8,110.5 183.3,110.9 184.8,111.3 186.3,111.6 187.7,112.0 189.2,112.4 190.7,112.8 192.1,113.1 193.6,113.5 195.1,113.8 196.6,114.2 198.0,114.5 199.5,114.8 201.0,115.2 202.4,115.5 203.9,115.8 205.4,116.1 206.9,116.4 208.3,116.7 209.8,117.0 211.3,117.3 212.7,117.6 214.2,117.9 215.7,118.1 217.2,118.4 218.6,118.7 220.1,118.9 221.6,119.2 223.0,119.4 224.5,119.6 226.0,119.9 227.5,120.1 228.9,120.3 230.4,120.5 231.9,120.8 233.3,121.0 234.8,121.2 236.3,121.4 237.8,121.5 239.2,121.7 240.7,121.9 242.2,122.1 243.6,122.3 245.1,122.4 246.6,122.6 248.1,122.7 249.5,122.9 251.0,123.0 252.5,123.2 253.9,123.3 255.4,123.4 256.9,123.5 258.4,123.7 259.8,123.8 261.3,123.9 262.8,124.0 264.2,124.1 265.7,124.2 267.2,124.2 268.7,124.3 270.1,124.4 271.6,124.5 273.1,124.5 274.5,124.6 276.0,124.6 277.5,124.7 279.0,124.7 280.4,124.8 281.9,124.8 283.4,124.8 284.8,124.9 286.3,124.9 287.8,124.9 289.3,124.9 290.7,124.9 292.2,124.9 293.7,124.9 295.2,124.9 296.6,124.9 298.1,124.8 299.6,124.8 301.0,124.8 302.5,124.7 304.0,124.7 305.5,124.6 306.9,124.6 308.4,124.5 309.9,124.5 311.3,124.4 312.8,124.3 314.3,124.2 315.8,124.2 317.2,124.1 318.7,124.0 320.2,123.9 321.6,123.8 323.1,123.7 324.6,123.5 326.1,123.4 327.5,123.3 329.0,123.2 330.5,123.0 331.9,122.9 333.4,122.7 334.9,122.6 336.4,122.4 337.8,122.3 339.3,122.1 340.8,121.9 342.2,121.7 343.7,121.5 345.2,121.4 346.7,121.2 348.1,121.0 349.6,120.8 351.1,120.5 352.5,120.3 354.0,120.1 355.5,119.9 357.0,119.6 358.4,119.4 359.9,119.2 361.4,118.9 362.8,118.7 364.3,118.4 365.8,118.1 367.3,117.9 368.7,117.6 370.2,117.3 371.7,117.0 373.1,116.7 374.6,116.4 376.1,116.1 377.6,115.8 379.0,115.5 380.5,115.2 382.0,114.8 383.4,114.5 384.9,114.2 386.4,113.8 387.9,113.5 389.3,113.1 390.8,112.8 392.3,112.4 393.7,112.0 395.2,111.7 396.7,111.3 398.2,110.9 399.6,110.5 401.1,110.1 402.6,109.7 404.0,109.3 405.5,108.8 407.0,108.4 408.5,108.0 409.9,107.6 411.4,107.1 412.9,106.7 414.3,106.2 415.8,105.8 417.3,105.3 418.8,104.8 420.2,104.3 421.7,103.9 423.2,103.4 424.6,102.9 426.1,102.4 427.6,101.9 429.1,101.3 430.5,100.8 432.0,100.3 433.5,99.8 434.9,99.2 436.4,98.7 437.9,98.1 439.4,97.6 440.8,97.0 442.3,96.4 443.8,95.8 445.3,95.2 446.7,94.7 448.2,94.1 449.7,93.5 451.1,92.8 452.6,92.2 454.1,91.6 455.6,91.0 457.0,90.3 458.5,89.7 460.0,89.0 461.4,88.4 462.9,87.7 464.4,87.0 465.9,86.4 467.3,85.7 468.8,85.0 470.3,84.3 471.7,83.6 473.2,82.8 474.7,82.1 476.2,81.4 477.6,80.7 479.1,79.9 480.6,79.2 482.0,78.4 483.5,77.6 485.0,76.9 486.5,76.1 487.9,75.3 489.4,74.5 490.9,73.7 492.3,72.9 493.8,72.1 495.3,71.3 496.8,70.4 498.2,69.6 499.7,68.7 501.2,67.9 502.6,67.0 504.1,66.1 505.6,65.3 507.1,64.4 508.5,63.5 510.0,62.6" fill="none" class="stroke-b" stroke-width="3"/>
<circle cx="290.8" cy="124.9" r="6" class="fill-b"/>
<line x1="290.8" y1="36.0" x2="290.8" y2="236.0" class="stroke-muted dashed" />
<text x="290.8" y="30.0" text-anchor="middle" class="fig-small fig-bold">best split: N* = 1.01B, D* = 19.8B</text>
<text x="78.3" y="48.8" text-anchor="start" class="fig-bold fig-small">model too small</text>
<text x="506.0" y="48.8" text-anchor="end" class="fig-bold fig-small">too few tokens</text>
<text x="83.4" y="152.8" text-anchor="start" class="fig-small fig-bold fill-a">size penalty</text>
<text x="508.1" y="152.8" text-anchor="end" class="fig-small fig-bold fill-d">data penalty</text>
<text x="300.8" y="103.6" text-anchor="start" class="fig-small fig-bold fill-b">sum 0.694</text>
<text x="290.0" y="272.0" text-anchor="middle" class="fig-small">model size N, with D = compute / 6N so that 6ND = 1.2e20</text>
<text x="12.0" y="20.0" text-anchor="start" class="fig-small">loss above the floor</text>
</svg>
Figure 1. Fix the budget at 1.2e20 flops and slide along it. A larger model leaves fewer tokens, so the size penalty (blue) falls while the data penalty (purple) rises. The sum (orange) is a U. The bottom of the U is where the two penalties are equal, because the exponents are equal here.
```

Figure 1 fixes the budget at $1.2 \times 10^{20}$ flops, which is what last chapter's model of 1 billion parameters on 20 billion tokens cost, and slides along it. On the left the model is small and the data is plentiful, so the size penalty (blue) is big and the data penalty (purple) is small. Moving right, each larger model gets fewer tokens, the blue line falls and the purple line rises. The sum above the floor (orange) falls, bottoms out, and climbs again. The left arm is a model that is too small, and the right arm is a model that sees too few tokens.

The bottom of the U sits at $N^{*} = 1.01$ billion parameters and $D^{*} = 19.8$ billion tokens, where the two penalties cross at 0.347 each. A U has one lowest point, and the budget has one best split. The rest of the chapter is about finding it exactly.

## Finding the bottom

Stand at some split and slide along the budget line. Because $6ND$ must stay fixed, growing $N$ by 1 percent means shrinking $D$ by about 1 percent, so one side pays exactly what the other receives. The size penalty falls by about $\alpha$ percent of itself, and the data penalty rises by about $\beta$ percent of itself. If the fall is bigger than the rise, the move helped, so keep sliding toward a larger model. If the rise is bigger, slide back. You stop at the split where the fall and the rise are equal, since a move in either direction would then lose:

$$
\frac{\alpha A}{N^{\alpha}} = \frac{\beta B}{D^{\beta}}
$$

Read it aloud: the loss that 1 percent more model buys equals the loss that 1 percent more data buys, so the budget has no bargain left on either side. Next, spend the whole budget by putting $D = C/(6N)$ on the right side, so $\beta B / D^{\beta} = \beta B\,(6N/C)^{\beta}$. Collect the powers of $N$ on one side:

$$
N^{\alpha + \beta} = \frac{\alpha A}{\beta B}\left(\frac{C}{6}\right)^{\beta}
$$

Take the $(\alpha+\beta)$-th root. Writing $a = \beta/(\alpha+\beta)$ and $G = (\alpha A/\beta B)^{1/(\alpha+\beta)}$, the best size and the matching tokens are

$$
N^{*} = G \cdot 6^{-a} \cdot C^{a}, \qquad D^{*} = \frac{C}{6N^{*}}
$$

Then the numbers, now at $10^{20}$ flops, the budget from the opening. With $\alpha = \beta = 0.34$, $a = 0.5$ and $G = (400/1100)^{1/0.68} = 0.2259$. For $C = 10^{20}$, $6^{-0.5} = 0.4082$, so $N^{*} = 0.2259 \times 0.4082 \times 10^{10} = 9.22 \times 10^{8}$ and $D^{*} = 10^{20} / (6 \times 9.22 \times 10^{8}) = 1.81 \times 10^{10}$. In code:

```python
a = beta / (alpha + beta)
G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
n_star = G * 6.0 ** (-a) * compute ** a
d_star = compute / (6.0 * n_star)
```

You do not need to trust the algebra, because a search can check it. Try many sizes $N$, set $D = C/(6N)$ for each, evaluate the loss, and keep the lowest. At $10^{20}$ flops, a grid of 2000 sizes lands within 0.2 percent of $9.22 \times 10^{8}$.

> [!KEY]
> At a fixed budget the loss along the size-versus-data line is a U, and its bottom is where 1 percent more model and 1 percent more data are worth the same. For these constants that is $N^{*} = G\,6^{-a}\,C^{a}$.

## What the optimum looks like

For these constants $D^{*}/N^{*} = 19.6$, about 20 tokens for every parameter, at every budget. That holds only because $\alpha = \beta$. Then $a = 0.5$, both $N^{*}$ and $D^{*}$ grow as $\sqrt{C}$, and the ratio $D^{*}/N^{*} = 1/G^{2}$ does not change. When the exponents differ, $N^{*}$ grows as $C^{a}$ and $D^{*}$ as $C^{1-a}$, so the ratio drifts as $C^{1-2a}$. With the published data exponent of about 0.28 in place of 0.34, $a$ is about 0.45, and the ratio grows by about a quarter for each tenfold increase in compute.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Best model size and best token count against compute on log axes: two parallel solid lines of slope one half, and two dashed lines of different slope that spread away">
<line x1="70.0" y1="240.0" x2="470.0" y2="240.0" class="stroke-muted"/><line x1="70.0" y1="30.0" x2="70.0" y2="240.0" class="stroke-muted"/><line x1="70.0" y1="30.0" x2="70.0" y2="240.0" class="stroke-grid"/><text x="70.0" y="256.0" text-anchor="middle" class="fig-small">1e18</text><line x1="203.3" y1="30.0" x2="203.3" y2="240.0" class="stroke-grid"/><text x="203.3" y="256.0" text-anchor="middle" class="fig-small">1e20</text><line x1="336.7" y1="30.0" x2="336.7" y2="240.0" class="stroke-grid"/><text x="336.7" y="256.0" text-anchor="middle" class="fig-small">1e22</text><line x1="470.0" y1="30.0" x2="470.0" y2="240.0" class="stroke-grid"/><text x="470.0" y="256.0" text-anchor="middle" class="fig-small">1e24</text><line x1="70.0" y1="205.0" x2="470.0" y2="205.0" class="stroke-grid"/><text x="64.0" y="209.0" text-anchor="end" class="fig-small">100M</text><line x1="70.0" y1="170.0" x2="470.0" y2="170.0" class="stroke-grid"/><text x="64.0" y="174.0" text-anchor="end" class="fig-small">1B</text><line x1="70.0" y1="135.0" x2="470.0" y2="135.0" class="stroke-grid"/><text x="64.0" y="139.0" text-anchor="end" class="fig-small">10B</text><line x1="70.0" y1="100.0" x2="470.0" y2="100.0" class="stroke-grid"/><text x="64.0" y="104.0" text-anchor="end" class="fig-small">100B</text><line x1="70.0" y1="65.0" x2="470.0" y2="65.0" class="stroke-grid"/><text x="64.0" y="69.0" text-anchor="end" class="fig-small">1T</text>
<polyline points="70.0,206.2 78.2,204.1 86.3,201.9 94.5,199.8 102.7,197.7 110.8,195.5 119.0,193.4 127.1,191.2 135.3,189.1 143.5,186.9 151.6,184.8 159.8,182.7 168.0,180.5 176.1,178.4 184.3,176.2 192.4,174.1 200.6,171.9 208.8,169.8 216.9,167.7 225.1,165.5 233.3,163.4 241.4,161.2 249.6,159.1 257.8,156.9 265.9,154.8 274.1,152.7 282.2,150.5 290.4,148.4 298.6,146.2 306.7,144.1 314.9,141.9 323.1,139.8 331.2,137.7 339.4,135.5 347.6,133.4 355.7,131.2 363.9,129.1 372.0,126.9 380.2,124.8 388.4,122.7 396.5,120.5 404.7,118.4 412.9,116.2 421.0,114.1 429.2,111.9 437.3,109.8 445.5,107.7 453.7,105.5 461.8,103.4 470.0,101.2" fill="none" class="stroke-a" stroke-width="2.5"/>
<polyline points="70.0,161.0 78.2,158.9 86.3,156.7 94.5,154.6 102.7,152.4 110.8,150.3 119.0,148.1 127.1,146.0 135.3,143.9 143.5,141.7 151.6,139.6 159.8,137.4 168.0,135.3 176.1,133.1 184.3,131.0 192.4,128.9 200.6,126.7 208.8,124.6 216.9,122.4 225.1,120.3 233.3,118.1 241.4,116.0 249.6,113.9 257.8,111.7 265.9,109.6 274.1,107.4 282.2,105.3 290.4,103.1 298.6,101.0 306.7,98.9 314.9,96.7 323.1,94.6 331.2,92.4 339.4,90.3 347.6,88.1 355.7,86.0 363.9,83.9 372.0,81.7 380.2,79.6 388.4,77.4 396.5,75.3 404.7,73.1 412.9,71.0 421.0,68.9 429.2,66.7 437.3,64.6 445.5,62.4 453.7,60.3 461.8,58.1 470.0,56.0" fill="none" class="stroke-b" stroke-width="2.5"/>
<polyline points="70.0,206.2 78.2,203.1 86.3,200.0 94.5,196.8 102.7,193.7 110.8,190.6 119.0,187.5 127.1,184.3 135.3,181.2 143.5,178.1 151.6,174.9 159.8,171.8 168.0,168.7 176.1,165.6 184.3,162.4 192.4,159.3 200.6,156.2 208.8,153.0 216.9,149.9 225.1,146.8 233.3,143.7 241.4,140.5 249.6,137.4 257.8,134.3 265.9,131.1 274.1,128.0 282.2,124.9 290.4,121.8 298.6,118.6 306.7,115.5 314.9,112.4 323.1,109.2 331.2,106.1 339.4,103.0 347.6,99.9 355.7,96.7 363.9,93.6 372.0,90.5 380.2,87.3 388.4,84.2 396.5,81.1 404.7,78.0 412.9,74.8 421.0,71.7 429.2,68.6 437.3,65.4 445.5,62.3 453.7,59.2 461.8,56.1 470.0,52.9" fill="none" class="stroke-a dashed" stroke-width="2"/>
<polyline points="70.0,161.0 78.2,159.8 86.3,158.7 94.5,157.5 102.7,156.4 110.8,155.2 119.0,154.1 127.1,152.9 135.3,151.7 143.5,150.6 151.6,149.4 159.8,148.3 168.0,147.1 176.1,146.0 184.3,144.8 192.4,143.6 200.6,142.5 208.8,141.3 216.9,140.2 225.1,139.0 233.3,137.9 241.4,136.7 249.6,135.5 257.8,134.4 265.9,133.2 274.1,132.1 282.2,130.9 290.4,129.8 298.6,128.6 306.7,127.4 314.9,126.3 323.1,125.1 331.2,124.0 339.4,122.8 347.6,121.7 355.7,120.5 363.9,119.3 372.0,118.2 380.2,117.0 388.4,115.9 396.5,114.7 404.7,113.6 412.9,112.4 421.0,111.2 429.2,110.1 437.3,108.9 445.5,107.8 453.7,106.6 461.8,105.5 470.0,104.3" fill="none" class="stroke-b dashed" stroke-width="2"/>
<text x="478.0" y="70.0" text-anchor="start" class="fig-small fig-bold fill-b">D* (tokens)</text>
<text x="478.0" y="119.2" text-anchor="start" class="fig-small fig-bold fill-a">N* (parameters)</text>
<text x="478.0" y="46.9" text-anchor="start" class="fig-small fig-bold fill-a">N, slope 0.73</text>
<text x="478.0" y="100.3" text-anchor="start" class="fig-small fig-bold fill-b">D, slope 0.27</text>
<text x="101.8" y="214.1" text-anchor="start" class="fig-small">solid: equal exponents, slope 0.5 for both</text>
<text x="101.8" y="229.1" text-anchor="start" class="fig-small">dashed: a size-heavy fit, same start at 1e18</text>
<text x="270.0" y="276.0" text-anchor="middle" class="fig-small">training compute (flops)</text>
</svg>
Figure 2. The best N* and D* grow together as the square root of compute, so the two solid lines stay a factor of 19.6 apart. A fit that sends most extra compute to the model (dashed lines, a slope near 0.73 for N, drawn here only to show the effect) agrees with the solid lines at 1e18 and then asks for a much larger model and fewer tokens as compute grows.
```

Figure 2 shows the equal-exponent case as two parallel solid lines, the model size below and the token count above. Raising the budget a hundredfold multiplies both by ten. The dashed lines show what a different fit would claim: if the fit says $N^{*}$ grows as $C^{0.73}$, then at $10^{24}$ flops it asks for a model 24 times larger than the solid line and 24 times fewer tokens, a very different run.

Why would two studies disagree this much? Kaplan and colleagues found a size-heavy answer, close to the dashed slope. Hoffmann and colleagues, in the Chinchilla study, found one close to the solid slope. The Chinchilla authors pointed to the learning-rate schedule as a main cause. If runs of different lengths use the same schedule, a short run is stopped before its learning rate has decayed and looks worse than it should, so extra tokens look less useful than they are. A fitted law is only as good as the experiments behind it, and a scaling law is a fit to experiments, not a law of physics. The Chinchilla result, a 70-billion-parameter model on 1.4 trillion tokens that beat a 280-billion-parameter model on 300 billion tokens at about the same compute, is the practical proof: the two cost roughly $5 \times 10^{23}$ to $6 \times 10^{23}$ flops, and the better balanced one won.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Three ways to spend 1.2e20 flops, as dots on the constant-compute curve: a small model on lots of data, a balanced model, and a large model on little data">
<line x1="70.0" y1="236.0" x2="510.0" y2="236.0" class="stroke-muted"/><line x1="70.0" y1="36.0" x2="70.0" y2="236.0" class="stroke-muted"/><line x1="70.0" y1="36.0" x2="70.0" y2="236.0" class="stroke-grid"/><text x="70.0" y="252.0" text-anchor="middle" class="fig-small">100M</text><line x1="261.2" y1="36.0" x2="261.2" y2="236.0" class="stroke-grid"/><text x="261.2" y="252.0" text-anchor="middle" class="fig-small">1B</text><line x1="452.4" y1="36.0" x2="452.4" y2="236.0" class="stroke-grid"/><text x="452.4" y="252.0" text-anchor="middle" class="fig-small">10B</text><line x1="70.0" y1="222.7" x2="510.0" y2="222.7" class="stroke-grid"/><text x="64.0" y="226.7" text-anchor="end" class="fig-small">2.2</text><line x1="70.0" y1="169.3" x2="510.0" y2="169.3" class="stroke-grid"/><text x="64.0" y="173.3" text-anchor="end" class="fig-small">2.4</text><line x1="70.0" y1="116.0" x2="510.0" y2="116.0" class="stroke-grid"/><text x="64.0" y="120.0" text-anchor="end" class="fig-small">2.6</text><line x1="70.0" y1="62.7" x2="510.0" y2="62.7" class="stroke-grid"/><text x="64.0" y="66.7" text-anchor="end" class="fig-small">2.8</text>
<polyline points="70.0,113.2 71.5,114.2 72.9,115.2 74.4,116.1 75.9,117.1 77.4,118.0 78.8,118.9 80.3,119.8 81.8,120.7 83.2,121.6 84.7,122.5 86.2,123.4 87.7,124.3 89.1,125.1 90.6,126.0 92.1,126.8 93.5,127.7 95.0,128.5 96.5,129.3 98.0,130.1 99.4,130.9 100.9,131.7 102.4,132.5 103.8,133.3 105.3,134.0 106.8,134.8 108.3,135.6 109.7,136.3 111.2,137.0 112.7,137.8 114.1,138.5 115.6,139.2 117.1,139.9 118.6,140.6 120.0,141.3 121.5,142.0 123.0,142.7 124.4,143.3 125.9,144.0 127.4,144.6 128.9,145.3 130.3,145.9 131.8,146.5 133.3,147.2 134.7,147.8 136.2,148.4 137.7,149.0 139.2,149.6 140.6,150.1 142.1,150.7 143.6,151.3 145.1,151.8 146.5,152.4 148.0,152.9 149.5,153.5 150.9,154.0 152.4,154.5 153.9,155.0 155.4,155.5 156.8,156.0 158.3,156.5 159.8,157.0 161.2,157.5 162.7,158.0 164.2,158.4 165.7,158.9 167.1,159.3 168.6,159.8 170.1,160.2 171.5,160.6 173.0,161.1 174.5,161.5 176.0,161.9 177.4,162.3 178.9,162.7 180.4,163.1 181.8,163.4 183.3,163.8 184.8,164.2 186.3,164.5 187.7,164.9 189.2,165.2 190.7,165.5 192.1,165.9 193.6,166.2 195.1,166.5 196.6,166.8 198.0,167.1 199.5,167.4 201.0,167.7 202.4,168.0 203.9,168.2 205.4,168.5 206.9,168.8 208.3,169.0 209.8,169.2 211.3,169.5 212.7,169.7 214.2,169.9 215.7,170.2 217.2,170.4 218.6,170.6 220.1,170.8 221.6,171.0 223.0,171.1 224.5,171.3 226.0,171.5 227.5,171.6 228.9,171.8 230.4,171.9 231.9,172.1 233.3,172.2 234.8,172.4 236.3,172.5 237.8,172.6 239.2,172.7 240.7,172.8 242.2,172.9 243.6,173.0 245.1,173.1 246.6,173.1 248.1,173.2 249.5,173.3 251.0,173.3 252.5,173.4 253.9,173.4 255.4,173.4 256.9,173.5 258.4,173.5 259.8,173.5 261.3,173.5 262.8,173.5 264.2,173.5 265.7,173.5 267.2,173.5 268.7,173.4 270.1,173.4 271.6,173.4 273.1,173.3 274.5,173.3 276.0,173.2 277.5,173.1 279.0,173.1 280.4,173.0 281.9,172.9 283.4,172.8 284.8,172.7 286.3,172.6 287.8,172.5 289.3,172.4 290.7,172.2 292.2,172.1 293.7,172.0 295.2,171.8 296.6,171.6 298.1,171.5 299.6,171.3 301.0,171.1 302.5,171.0 304.0,170.8 305.5,170.6 306.9,170.4 308.4,170.2 309.9,169.9 311.3,169.7 312.8,169.5 314.3,169.3 315.8,169.0 317.2,168.8 318.7,168.5 320.2,168.2 321.6,168.0 323.1,167.7 324.6,167.4 326.1,167.1 327.5,166.8 329.0,166.5 330.5,166.2 331.9,165.9 333.4,165.6 334.9,165.2 336.4,164.9 337.8,164.5 339.3,164.2 340.8,163.8 342.2,163.4 343.7,163.1 345.2,162.7 346.7,162.3 348.1,161.9 349.6,161.5 351.1,161.1 352.5,160.7 354.0,160.2 355.5,159.8 357.0,159.4 358.4,158.9 359.9,158.5 361.4,158.0 362.8,157.5 364.3,157.0 365.8,156.6 367.3,156.1 368.7,155.6 370.2,155.1 371.7,154.5 373.1,154.0 374.6,153.5 376.1,153.0 377.6,152.4 379.0,151.9 380.5,151.3 382.0,150.7 383.4,150.2 384.9,149.6 386.4,149.0 387.9,148.4 389.3,147.8 390.8,147.2 392.3,146.6 393.7,145.9 395.2,145.3 396.7,144.7 398.2,144.0 399.6,143.4 401.1,142.7 402.6,142.0 404.0,141.3 405.5,140.6 407.0,139.9 408.5,139.2 409.9,138.5 411.4,137.8 412.9,137.1 414.3,136.3 415.8,135.6 417.3,134.8 418.8,134.1 420.2,133.3 421.7,132.5 423.2,131.7 424.6,130.9 426.1,130.1 427.6,129.3 429.1,128.5 430.5,127.7 432.0,126.9 433.5,126.0 434.9,125.2 436.4,124.3 437.9,123.4 439.4,122.5 440.8,121.7 442.3,120.8 443.8,119.9 445.3,118.9 446.7,118.0 448.2,117.1 449.7,116.2 451.1,115.2 452.6,114.2 454.1,113.3 455.6,112.3 457.0,111.3 458.5,110.3 460.0,109.3 461.4,108.3 462.9,107.3 464.4,106.3 465.9,105.2 467.3,104.2 468.8,103.1 470.3,102.1 471.7,101.0 473.2,99.9 474.7,98.8 476.2,97.7 477.6,96.6 479.1,95.5 480.6,94.3 482.0,93.2 483.5,92.0 485.0,90.9 486.5,89.7 487.9,88.5 489.4,87.3 490.9,86.1 492.3,84.9 493.8,83.7 495.3,82.5 496.8,81.3 498.2,80.0 499.7,78.7 501.2,77.5 502.6,76.2 504.1,74.9 505.6,73.6 507.1,72.3 508.5,71.0 510.0,69.6" fill="none" class="stroke-muted" stroke-width="2"/>
<circle cx="146.1" cy="152.2" r="6.5" class="fill-a"/>
<text x="146.1" y="178.2" text-anchor="middle" class="fig-small">N=2.5e8, D=8e10</text>
<text x="146.1" y="192.2" text-anchor="middle" class="fig-small fig-bold">loss 2.4641</text>
<circle cx="261.2" cy="173.5" r="6.5" class="fill-b"/>
<text x="261.2" y="199.5" text-anchor="middle" class="fig-small">N=1e9, D=2e10</text>
<text x="261.2" y="213.5" text-anchor="middle" class="fig-small fig-bold">loss 2.3844</text>
<circle cx="376.3" cy="152.9" r="6.5" class="fill-a"/>
<text x="376.3" y="178.9" text-anchor="middle" class="fig-small">N=4e9, D=5e9</text>
<text x="376.3" y="192.9" text-anchor="middle" class="fig-small fig-bold">loss 2.4617</text>
<text x="290.0" y="272.0" text-anchor="middle" class="fig-small">model size N, with D = 1.2e20 / 6N</text>
<text x="12.0" y="20.0" text-anchor="start" class="fig-small">loss</text>
</svg>
Figure 3. Three splits with the same compute, 6ND = 1.2e20. The balanced split sits at the bottom of the curve with loss 2.3844. A model four times larger on a quarter of the tokens, and a model four times smaller on four times the tokens, both land near 2.46. Check any dot with the formula from the previous chapter.
```

Figure 3 makes the same point with three splits at $1.2 \times 10^{20}$ flops. The balanced split, 1 billion parameters on 20 billion tokens, has loss 2.3844. A model 4 times larger on a quarter of the tokens, and one 4 times smaller on 4 times the tokens, both land near 2.46. All three cost the same compute, and the balanced one is best by about 0.08. It is worth checking one of them: for $N = 4 \times 10^9$ and $D = 5 \times 10^9$, the size penalty is 0.2175 and the data penalty is 0.5542, which gives $1.69 + 0.2175 + 0.5542 = 2.4617$.

> [!RECAP]
> - At a fixed budget, $D = C/(6N)$, so a larger model means fewer tokens, and the loss traces a U.
> - The bottom of the U is where $\alpha A/N^{\alpha} = \beta B/D^{\beta}$, which gives $N^{*} = G\,6^{-a}\,C^{a}$ and $D^{*} = C/(6N^{*})$.
> - With equal exponents, $a = 0.5$ and the ratio $D^{*}/N^{*}$ is a constant, 19.6 for these constants. With unequal exponents it drifts as $C^{1-2a}$.
> - A grid search over $N$ confirms the closed form.
> - Serving costs $2N$ per token, which pushes a team to smaller models trained longer.

## The cost of serving

The best split minimises training compute for a given loss. It ignores what happens after training. Each token a deployed model reads or writes costs about $2N$ flops, which is the forward pass alone, and that bill grows with $N$ and with how much the model is used.

Take the balanced model at $1.2 \times 10^{20}$ flops, with $N^{*} = 1.01 \times 10^{9}$. A model half the size, $5.05 \times 10^{8}$, can reach the same loss of 2.3844 if you train it on enough text: solving the loss law for $D$ gives $4.9 \times 10^{10}$ tokens, which is 97 tokens per parameter. Training it costs $6ND = 1.49 \times 10^{20}$ flops, 24 percent more than the balanced model. Serving costs half as much per token, so the totals cross.

$$
\text{total} = 6\,N\,D_{\text{train}} + 2\,N\,D_{\text{served}}
$$

Read it aloud: training at six flops per weight per token, plus serving at two. Setting the two models' totals equal gives a break-even at $2.86 \times 10^{10}$ served tokens. After a trillion served tokens, the balanced model's total is $2.1 \times 10^{21}$ flops and the half-size model's is $1.2 \times 10^{21}$. A heavily used model is therefore often trained far beyond 20 tokens per parameter.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Total flops against tokens served for a balanced model and a model half its size trained to the same loss, with the lines crossing at 28.6 billion tokens">
<line x1="80.0" y1="226.0" x2="500.0" y2="226.0" class="stroke-muted"/><line x1="80.0" y1="36.0" x2="80.0" y2="226.0" class="stroke-muted"/><line x1="80.0" y1="36.0" x2="80.0" y2="226.0" class="stroke-grid"/><text x="80.0" y="242.0" text-anchor="middle" class="fig-small">0</text><line x1="164.0" y1="36.0" x2="164.0" y2="226.0" class="stroke-grid"/><text x="164.0" y="242.0" text-anchor="middle" class="fig-small">20B</text><line x1="248.0" y1="36.0" x2="248.0" y2="226.0" class="stroke-grid"/><text x="248.0" y="242.0" text-anchor="middle" class="fig-small">40B</text><line x1="332.0" y1="36.0" x2="332.0" y2="226.0" class="stroke-grid"/><text x="332.0" y="242.0" text-anchor="middle" class="fig-small">60B</text><line x1="416.0" y1="36.0" x2="416.0" y2="226.0" class="stroke-grid"/><text x="416.0" y="242.0" text-anchor="middle" class="fig-small">80B</text><line x1="500.0" y1="36.0" x2="500.0" y2="226.0" class="stroke-grid"/><text x="500.0" y="242.0" text-anchor="middle" class="fig-small">100B</text><line x1="80.0" y1="178.5" x2="500.0" y2="178.5" class="stroke-grid"/><text x="74.0" y="182.5" text-anchor="end" class="fig-small">1e20</text><line x1="80.0" y1="131.0" x2="500.0" y2="131.0" class="stroke-grid"/><text x="74.0" y="135.0" text-anchor="end" class="fig-small">2e20</text><line x1="80.0" y1="83.5" x2="500.0" y2="83.5" class="stroke-grid"/><text x="74.0" y="87.5" text-anchor="end" class="fig-small">3e20</text><line x1="80.0" y1="36.0" x2="500.0" y2="36.0" class="stroke-grid"/><text x="74.0" y="40.0" text-anchor="end" class="fig-small">4e20</text>
<polyline points="80.0,169.0 500.0,73.0" fill="none" class="stroke-a" stroke-width="2.5"/>
<polyline points="80.0,155.3 500.0,107.3" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="199.9" cy="141.6" r="5" class="fill-ink"/>
<line x1="199.9" y1="141.6" x2="199.9" y2="226.0" class="stroke-muted dashed" />
<text x="207.9" y="218.0" text-anchor="start" class="fig-small fig-bold">break-even: 28.6B served tokens</text>
<text x="496.0" y="154.8" text-anchor="end" class="fig-small fig-bold fill-a">balanced: N = 1.01B, D = 20B</text>
<text x="496.0" y="170.8" text-anchor="end" class="fig-small fig-bold fill-b">half size: N = 0.51B, D = 49B</text>
<text x="290.0" y="262.0" text-anchor="middle" class="fig-small">tokens served after training</text>
<text x="12.0" y="20.0" text-anchor="start" class="fig-small">flops, training plus serving</text>
</svg>
Figure 4. Both models reach the same loss. The half-size model starts higher, because it needed 49 billion training tokens, 24% more compute than the balanced model. Each token served costs it half as much, so its line is flatter and crosses the other at 28.6 billion served tokens. Beyond that, it is the cheaper model.
```

## Where this breaks

The data has to exist. $D^{*}$ grows with the budget, and at $10^{24}$ flops it asks for 1.8 trillion tokens of distinct text. Distinct text is finite, and repeated data helps less than fresh data, so the law's data term is too optimistic once you run out.

A small error in an exponent grows with distance, as in the chapter on power laws and the straight line on log paper, so predicting the best split three decades beyond your experiments is a guess. The constants here are stylised, and another data mix, tokeniser, or architecture moves all five. The law also predicts loss on training-like text, while a benchmark score can jump even as that loss falls smoothly.

## Build it step by step

You write three functions, and the checks call them in this order. `predict_loss` from the last chapter is already in the starter file.

1. `chinchilla_optimal(compute)` returns `(n_star, d_star)` from the closed form. Compute `a`, then `G`, then `n_star`, then `d_star` from $C = 6ND$. The usual mistake is dropping the $6^{-a}$, which still passes the budget check but pushes the ratio far from 19.6.
2. `grid_optimal(compute, n_points=2000)` builds a log-spaced grid of sizes from $\sqrt{C/6}/100$ to $100\sqrt{C/6}$, sets `D = compute / (6 * N)` for each, evaluates `predict_loss` on the arrays, and returns the `(N, D)` with the lowest loss. Use `np.geomspace` and `np.argmin`.
3. `lifetime_flops(n_params, train_tokens, served_tokens)` returns $6ND_{\text{train}} + 2ND_{\text{served}}$ as a float. Convert each input to `float` first, so numpy integers cannot overflow.

The shape of the second function:

```python
def grid_optimal(compute, n_points=2000):
    # sizes = np.geomspace(...) * sqrt(compute / 6); tokens from the budget
    return n_best, d_best
```

> [!TIP]
> If `grid_optimal` and `chinchilla_optimal` disagree by more than a percent, the grid is usually missing the optimum, so widen the range before you blame the formula.

## The exercise

You implement `chinchilla_optimal`, `grid_optimal`, and `lifetime_flops` in `starter.py`. The checks confirm that the closed form spends the budget and gives $9.22 \times 10^{8}$ and $1.81 \times 10^{10}$ at $10^{20}$ flops, that it sits at the bottom of the loss along the budget line with equal marginal gains, that the ratio is 19.6 at every budget and $N^{*}$ grows as $\sqrt{C}$, that the grid agrees within 1 percent, and that `lifetime_flops` reproduces the break-even at $2.86 \times 10^{10}$ served tokens for a model half the size.
