# Curriculum

117 chapters in 7 parts, and 8 projects in a section of their own. Each
project lists the chapter to finish before you start it. Each row is one Markdown file under
`content/lessons/` and one directory under `content/exercises/`. The `Needs GPU`
column marks the exercises that run on RunPod rather than in the app container.

Each chapter teaches one idea. The 33 lessons of the first edition were split
into chapters, and 15 pairs of adjacent chapters were then merged into one.
`content/renames.json` maps each old lesson and chapter to the chapters that
replaced it, so readers keep their progress.

## Part 1 — Foundations

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `001-scoring-a-guess` | Scoring a guess: loss and empirical risk | Write the squared loss, average it into an empirical risk, and use that risk to choose the best slope for a rent rule. | No |
| `002-honest-scores` | Honest scores: training error, test error, and holding data back | Write a seeded train/test split and a rule that memorises its training flats, then measure the gap between its training and test risk. | No |
| `003-fitting-curves` | Fitting a curve: polynomials, least squares, and the degree dial | Build the design matrix, fit weights by least squares, and sweep the degree to see training risk fall while held-out risk turns around. | No |
| `004-bias-variance-noise` | Bias, variance, and noise: why the dial has two ways to fail | Compute the split by hand for four dart-like guesses, then run a Monte Carlo experiment that refits a polynomial on many training sets and measures bias, variance, and noise. | No |
| `005-least-squares-line` | Fit a line by minimising squared misses | Write the squared-miss loss and its gradient, check the gradient against finite differences, and solve the normal equations. | No |
| `006-residuals-and-projection` | Check a fit with right angles | Measure how far a fit is from perpendicular, project the targets onto the column space, and decide when a fit is the best one. | No |
| `007-stable-least-squares` | Lost digits, and how QR and the SVD keep them | Measure a condition number, solve the normal equations with Cholesky, then solve with QR and the SVD, return the shortest weights for repeated columns, and count the rank. | No |
| `008-walking-downhill` | Walking downhill: slope, gradient, and one step | Build the loss and gradient of a bowl, a finite-difference gradient checker, and gradient descent that returns the whole path. | No |
| `009-step-size-and-long-valleys` | How big a step? Step size and long valleys | Read the curvatures of a bowl, predict the shrink factor on each axis, pick the best step along a line, then measure the condition number, the shrink ratio of a walk, and the steps to reach a tolerance. | No |
| `010-momentum` | Momentum: a heavy ball and a look-ahead | Build heavy-ball momentum, Nesterov's look-ahead, and the best settings for a bowl, then count how many fewer steps they need on a long valley. | No |
| `011-ridge-penalty-and-closed-form` | Ridge: a fee on large weights | Write the ridge score and its gradient, check the gradient against finite differences, and solve for the bottom of the bowl with one linear system. | No |
| `012-ridge-svd-path-and-degrees-of-freedom` | Ridge for every lambda: the SVD path and degrees of freedom | Compute the per-direction shrink factors, turn them into the ridge weights for every lambda at once, and count the degrees of freedom each lambda leaves. | No |
| `013-lasso-and-soft-thresholding` | Lasso and soft thresholding: why some weights become exactly zero | Build the shrink-and-kill map for one number, score the lasso, and use the map to solve the lasso exactly when the design is the identity. | No |
| `014-ista-and-fista` | ISTA and FISTA: fitting the lasso by stepping and thresholding | Take gradient steps and soft-threshold them, add momentum, and test with the KKT rules whether you have reached the bottom. | No |
| `015-sigmoid-and-probability` | From a score to a probability | Build a sigmoid that stays finite at a score of 800, its inverse the logit, and a function that scores a whole data matrix. | No |
| `016-log-loss` | How wrong is a probability? The log loss | Build a stable softplus and the log loss softplus(z) - y z with a ridge fee, and compare it with the textbook formula where that formula still works. | No |
| `017-gradient-descent-for-logistic` | Learning the weights: gradient and descent | Build the gradient X^T(p - y), a descent loop that records the loss before each step, and an accuracy function, then watch the weights grow on a perfect split. | No |
| `018-newton-method` | Curvature and Newton's method | Build the Hessian X^T D X + lambda I, a Newton step that solves H delta = g, and a fit loop that finishes in a handful of iterations where gradient descent needs hundreds. | No |
| `019-one-dial-three-distributions` | One dial, three distributions | Write A(eta), its slope, and its bend for the Gaussian, Bernoulli, and Poisson families, and watch the mean and variance fall out of one derivative each. | No |
| `020-one-loss-one-gradient` | One loss and one gradient for every family | Write the loss A(eta) - y eta and its gradient X^T(mu - y), then fit a Poisson model by walking downhill. | No |
| `021-curvature-and-fisher-scoring` | Curvature and Fisher scoring | Build the bend table X^T D X, take Newton steps with it, and fit Poisson, Gaussian, and Bernoulli models with one loop. | No |

## Part 2 — Classical models

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `022-cross-validation` | Why training error lies, and how k-fold cross-validation fixes it | Cut the data into folds, score a model on every fold it did not train on, and use those scores to choose the ridge penalty for 60 houses. | No |
| `023-leave-one-out-shortcut` | Leave-one-out error from a single ridge fit | Build the hat matrix and its diagonal, the leverages, and use them to compute the leave-one-out error of ridge regression without refitting. | No |
| `024-learning-curves` | Reading a learning curve | Hold out one fixed validation set, train on growing prefixes of the rest, and turn the two curves into a verdict: underfit, overfit, or good. | No |
| `025-winners-curse-and-flexibility` | Why the best score among many settings is too good | Simulate why the best of many noisy scores is too good, estimate how well a set of models fits random signs, and build nested cross-validation to score a penalty choice on data it never saw. | No |
| `026-bayes-rule-classifier` | Classify by asking who made this | Write the log of a Gaussian density, a stable log-sum-exp, and the Bayes step that turns per-class log densities and priors into log posteriors. | No |
| `027-gaussian-discriminant-analysis` | Fit a Gaussian cloud to each class | Estimate class means, one shared covariance and priors by averaging, score points with a Cholesky-based Gaussian log density, and combine them with Bayes' rule. | No |
| `028-gda-is-logistic-regression` | Why two Gaussians give a straight boundary | Turn fitted means, a shared covariance and priors into the weights, the log odds and the boundary of the equivalent logistic regression, with no gradient descent. | No |
| `029-naive-bayes` | Naive Bayes for many features and words | Fit Gaussian naive Bayes with per-class variances, fit a smoothed word-count model, and score documents by summing count times log word probability. | No |
| `030-widest-road-hinge-loss` | The widest road and the hinge loss | Compute the margins, the hinge objective and its subgradient, then follow the subgradient to the widest road. | No |
| `031-the-dual-and-support-vectors` | The dual problem and the support vectors | Compute a table of pairwise similarities, score points and the dual objective from the pushes alone, sort every house into its group, and compute the number that says how badly a set of pushes breaks the best-road picture. | No |
| `032-smo-two-at-a-time` | Two at a time: the SMO solver | Clip a pair of pushes to the box, jump to the top of the one-variable parabola, and repeat on whichever houses break the checklist. | No |
| `033-kernels-as-similarity` | Kernels: a similarity score that hides a bigger feature space | Compute RBF and polynomial kernel matrices with one matrix product, then confirm that a degree-2 polynomial kernel equals a dot product of six explicit features. | No |
| `034-valid-kernels` | What makes a similarity a valid kernel | Compute the smallest eigenvalue of a Gram matrix, find the squared length of a combination of points, and use both to build a positive semidefinite test. | No |
| `035-kernel-ridge-regression` | Kernel ridge regression: a weighted sum of similarities | Solve (K + lambda I) alpha = y with a Cholesky factor, predict with a matrix of query-to-training similarities, and recover the primal weights of a linear kernel. | No |
| `036-representer-and-random-features` | The representer theorem and random Fourier features | Evaluate the regularised objective for a candidate with extra centres, build random Fourier features for the RBF kernel, and fit ridge regression on them. | No |

## Part 3 — Trees, ensembles, and unsupervised learning

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `037-leaves-and-impurity` | Boxes, leaves, and impurity: how bad is one answer for a box? | Write Gini impurity, entropy, and variance impurity, and confirm that the mean is the best single answer for a box. | No |
| `038-gain-and-best-split` | Which question helps most: gain and the sorted sweep | Write the weighted gain of a split, list the cuts worth trying on one feature, and find the best split with running sums. | No |
| `039-grow-and-predict` | Grow a tree: recursion, stopping, and overfitting | Write the walk from root to leaf, measure a tree's depth, and grow a tree by splitting each box again until a stopping rule fires. | No |
| `040-pruning` | Prune the tree: paying for each extra leaf | Write the cost of a tree with a fee per leaf, the break-even fee of a split, and the bottom-up prune that keeps a split only when it pays. | No |
| `041-averaging-models` | Why averaging models cuts noise, and where it stops | Evaluate the variance of an average, build correlated model predictions from a shared and a private part, and count the models needed to get near the floor. | No |
| `042-bagging-and-out-of-bag-error` | Bagging, and the free test set it leaves behind | Draw bootstrap samples, fit and average one stump per draw, then compute the out-of-bag error from the rows each stump never saw. | No |
| `043-gradient-boosting` | Gradient boosting: fit what is still wrong | Write the negative gradients of two losses, then run gradient boosting with stumps and check that each round lowers the loss by the closed-form amount. | No |
| `044-principal-directions` | Principal directions: where the data spreads | Centre a cloud, measure the variance and the kept and lost parts along a line, then read every direction from the SVD and the covariance eigendecomposition. | No |
| `045-compress-and-know-what-you-lost` | Compress, and know what you lost | Rebuild data from k components, confirm that the error is the sum of the dropped squared singular values, and standardise columns so units stop mattering. | No |
| `046-power-iteration` | Finding the top eigenvector with one repeated product | Read an eigenvalue with the Rayleigh quotient, find the top eigenvector by multiplying and rescaling, and deflate to expose the second one. | No |
| `047-probabilistic-pca-em` | A noisy model for PCA, fitted by guess and refit | Write the E-step, the M-step, and the log-likelihood of probabilistic PCA, then watch the supplied EM loop raise the likelihood and land on the PCA plane. | No |
| `048-k-means` | k-means: Lloyd's steps and a good start | Write the k-means objective, the nearest-centre step, and Lloyd's loop, then seed the centres by squared distance and keep the best of several restarts. | No |
| `049-gmm-soft-assignment` | Gaussian mixtures and soft assignment | Write the log density of a Gaussian, a log-sum-exp that survives huge negative scores, and the E-step that turns scores into responsibilities. | No |
| `050-em-refit-and-converge` | EM: refit the blobs and watch the likelihood rise | Write the weighted M-step, the lower bound that explains why the likelihood never falls, and the full EM loop, then watch the likelihood rise. | No |

## Part 4 — Neural networks

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `051-why-layers-need-a-bend` | Why a stack of layers needs a bend | Write ReLU, its slope, and the merge of two affine layers, then watch two ReLU units turn XOR into a problem for one straight line. | No |
| `052-forward-pass-and-gelu` | Run a network forward, with a sharp or smooth bend | Write GELU and its slope, then a forward pass that runs a list of layers, each with a named bend. | No |
| `053-width-depth-and-cost` | Width adds, depth multiplies, and what it costs | Count the straight pieces of a one-input ReLU network, build the sawtooth whose pieces double with every layer, then count the parameters, flops, and stored activations that make depth cheap. | No |
| `054-backprop-the-graph-and-reverse-order` | Walking slopes backwards through a calculation | Write the two-sided slope check, the multiply rule, and the backward walk, then confirm them on c = a*b + a and on a diamond. | No |
| `055-backprop-array-operations` | The backward rule for each array operation | Write unbroadcast, the matrix-product rule, and the ReLU rule, so slopes flow through a layer that multiplies, adds a bias, and bends. | No |
| `056-softmax-cross-entropy-gradient` | Softmax and cross-entropy: a loss with a gradient you can read | Write a stable softmax, a stable log-softmax, and the cross-entropy loss whose slope on the scores is predicted probabilities minus the truth. | No |
| `057-backprop-mlp-and-gradient-check` | Putting it together: gradients for a whole network | Wire the engine into a two-layer network, then write the finite-difference check that tells you the gradients are right. | No |
| `058-stochastic-gradients` | Noisy gradients: stochastic gradient descent and mini-batches | Compute mini-batch gradients, take SGD and momentum steps, and race them on a long valley with curvatures 1 and 100. | No |
| `059-adam-and-adamw` | Adam, and why AdamW decouples the decay | Build the moving average, its bias correction, and Adam's step, then write the coupled L2 version and AdamW and measure how much shrink decay alone causes in each. | No |
| `060-lr-schedules` | Learning-rate schedules: warmup and cosine decay | Build linear warmup, the cosine decay fraction, and the full schedule, then run Adam with it on noisy mini-batches. | No |
| `061-signal-size-through-layers` | Why a deep network's signal shrinks or explodes | Compute a layer's gain and the gain after many layers, then simulate a random stack and check the simulation against your prediction. | No |
| `062-xavier-and-he` | Xavier and He initialisation | Write the Xavier and He scales, measure what a ReLU does to a signal's size, and check on a fifty-layer network that He keeps the signal alive and Xavier does not. | No |
| `063-batch-norm-and-layer-norm` | Normalising activations: batch norm versus layer norm | Rescale a group of numbers to mean 0 and variance 1, then apply the same rule down each column for batch norm and along each row for layer norm. | No |
| `064-normalisation-backward` | Backpropagating through a normalisation layer | Compute the gradients of the scale and shift, then the compact formula for the input gradient, and check both layers against finite differences and two sanity properties. | No |
| `065-weight-decay` | Weight decay: a pull toward zero | Write the slope of the ridge fee, the decayed gradient step, and the angle a sideways step turns a weight vector through. | No |
| `066-early-stopping-ridge` | Early stopping is a ridge penalty in disguise | Run gradient descent on one coordinate, compute the ridge penalty that k steps match, and pick a stopping step from a jumpy validation curve. | No |
| `067-dropout` | Dropout: train a crowd of thinned networks | Draw a keep-mask, apply inverted dropout in the forward pass, and send the gradient back through the same mask. | No |
| `068-label-smoothing` | Label smoothing: ask for a little less certainty | Build the smoothed target, the loss against it, and its gradient, then find where the loss is lowest. | No |

## Part 5 — Vision and sequences

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `069-sliding-a-filter` | Sliding one small filter over a picture | Write the one-channel sliding filter, grow it into a batched multi-channel convolution with loops, and count the weights of dense, convolutional, and depthwise separable layers. | No |
| `070-output-size-and-receptive-field` | How big is the output and how far can a unit see | Write the output-size formula for one axis, the receptive field after every layer of a stack, and a max-pooling layer. | No |
| `071-convolution-as-matrix-multiply` | Turning a convolution into one matrix multiply | Lay every window out as a column, multiply by the flattened kernels, and check the result against a slow loop. Then write the 1 by 1 convolution as a plain matrix multiply. | No |
| `072-convolution-backward` | Sending the gradient back through a convolution | Write col2im as a scatter-add, use it to count how many windows cover each pixel, and build the full backward pass of a convolution. Gradients are checked against finite differences. | No |
| `073-the-residual-block` | The residual block, forward and backward | Write the forward pass of a block that outputs the input plus a correction, measure how far it moves its input, then send the gradient back through the plus sign and check it against finite differences. | No |
| `074-gradient-through-deep-stacks` | Why the shortcut keeps a fifty-layer gradient alive | Build orthogonal layers, then backpropagate a unit gradient through fifty of them with and without a shortcut, and compare the length that arrives with the toy scalar model. | No |
| `075-bottleneck-cost` | Doing the wide work at a narrow width: bottleneck cost | Count the multiply-adds of a convolution, build the cost of a three-step bottleneck from it, and measure how much cheaper it is than two full-width 3x3 convolutions. | No |
| `076-recurrent-cell` | A network that reads one step at a time | Write one step of a recurrent cell, run it over a sequence with the same weights at every step, and read out the last state to get a loss and the gradient on each state. | No |
| `077-backprop-through-time` | Walk the chain backward: backpropagation through time | Write the tanh gradient, then the gradient of one step of the cell, then the backward loop that adds each step's own loss to the carry from later steps. Every output is checked against finite differences. | No |
| `078-vanishing-and-exploding-gradients` | Why gradients fade or blow up | Compute the spectral radius, use it to predict how large a gradient is after many steps, and ask how many steps it takes to fall below a threshold. Then compare the prediction with the gradient your own backward pass measures. | No |
| `079-truncation-and-clipping` | Taming long sequences: truncation and clipping | Split a long sequence into chunks, backpropagate through each chunk alone, and clip a gradient by its global norm. The truncated gradient is checked against finite differences of a loss that treats each chunk's starting state as a constant. | No |
| `080-why-rnns-forget-the-belt` | Why a recurrent net forgets, and the belt that fixes it | Compute the gradient left after many steps for a plain net and for a belt with a gate near one, find how far each can reach, and build the running product of forget gates. | No |
| `081-gates-and-the-lstm-update` | Gates and the LSTM memory update | Write the gate that decides how much to let through, the repeated belt update that holds a number, and the full forward step of an LSTM cell. | No |
| `082-lstm-backward-step` | Backpropagation through one LSTM step | Collect the two paths into the cell state, then send the gradient back through the gates, the weights, the input, and the previous cell. Gradients are checked against finite differences. | No |
| `083-gru-and-parameter-counts` | The GRU: one gate that forgets and writes, and what each cell costs | Write the blend that keeps part of the old state and writes part of a candidate, the full GRU step with its reset gate, and the parameter count for a plain RNN, a GRU, and an LSTM. | No |
| `084-attention-as-soft-lookup` | Why one vector is not enough: attention as a soft lookup | Write a stable softmax, the blend that turns scores into a context vector, and a dot-product scorer with an optional matrix in the middle. | No |
| `085-additive-attention` | Additive attention: let a small network score the match | Write the additive scorer, the attention built on it, and the parameters that make it imitate a plain dot product. | No |
| `086-attention-gradients` | Backpropagation through attention | Write the backward rule for the softmax, for the blend, and for the whole additive attention, and check every gradient against finite differences. | No |
| `087-alignment-and-copy-decoder` | Where the decoder looks: the alignment matrix and a copy decoder | Stack the attention weights of every decoder step into an alignment matrix, hand-set a scorer so each step reads the right source position, and compare it with an order-blind summary. | No |

## Part 6 — Transformers

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `088-scaled-attention-scores` | Attention scores, and why we scale them | Write the pieces of a soft lookup, measure how the spread of dot products grows with the key width, and build scaled dot-product attention that undoes it. | No |
| `089-heads-and-causal-mask` | Many heads and the causal mask | Build the causal mask, split a projection into column-slice heads, and join the heads into multi-head attention. | No |
| `090-attention-backward-and-cost` | The gradient and the cost of attention | Write the softmax backward rule, chain it into the gradient of scaled dot-product attention, and count the flops of one layer. | No |
| `091-position-clocks` | Where is each token? Sinusoidal position clocks | Write the sinusoidal encoding, the matrix that moves it k positions, and the step that adds it to token vectors. | No |
| `092-rotary-embeddings` | Rotating queries and keys: rotary position embeddings | Write the cosine and sine tables, rotate queries and keys with them, and check that the score depends only on the offset. | No |
| `093-gated-feed-forward` | The feed-forward network and the gate: SwiGLU | Write a stable SiLU, the gated feed-forward network that uses it, and the weight count that explains the two-thirds width. | No |
| `094-the-residual-block` | Stacking it up: residuals, layer norm, and the pre-norm block | Write layer norm, wire attention with rotary embeddings and SwiGLU into one residual block, and count its parameters. | No |
| `095-next-token-loss` | How a language model is scored: next-token loss and perplexity | Write the per-token loss without overflow, cut a text into inputs and shifted targets, and read the loss as perplexity. | No |
| `096-safe-steps` | Taking safe steps: warmup, cosine decay, and gradient clipping | Write the warmup-then-cosine schedule, the global norm of a list of gradients, and a clip that shortens them in place without turning them. | No |
| `097-the-training-loop` | One training step and the loop around it | Write one AdamW update for a single tensor, then wire loss, backward, clip, schedule, and update into train_step, and watch a tiny transformer learn a line of text. | No |
| `098-cost-of-training` | The cost of a run: flops, memory, and number formats | Write the 6ND flop count and the four-copies memory estimate, then simulate loss scaling through float16 to see a gradient of 1e-8 vanish and come back. | No |

## Part 7 — Large language models

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `099-bytes-and-the-vocabulary-trade` | Why bytes, and the trade between vocabulary and length | Write the byte encoder and decoder that make every text representable, and measure how many words a fixed word list misses. | No |
| `100-byte-pair-encoding` | Byte-pair encoding: train it, then use it | Write the pair counter, the merger, and the training loop, then replay the learned merges to encode text and join bytes to decode it. | No |
| `101-fertility-and-fairness` | Measuring a tokeniser: fertility, fairness, and where it breaks | Measure how many tokens a tokeniser spends per word and per character, and see how the training text decides what a script and a number cost. | No |
| `102-power-laws-log-log` | Power laws and the straight line on log paper | Evaluate a power law, recover its exponent and coefficient with a least-squares line in log space, and use the fit to predict a size you never measured. | No |
| `103-loss-from-size-and-data` | Loss from size and data | Evaluate the loss law E + A/N^alpha + B/D^beta, break it into its three parts, and measure what growing the model buys. | No |
| `104-compute-optimal-models` | Spending a compute budget well | Count the weights that multiply, turn the count and a token budget into flops and days, then find the best split of a budget between model size and data and add serving cost. | No |
| `105-decoding-and-the-kv-cache` | Why decoding waits on memory, and the KV cache | Compute the flops per byte of a weight matrix, the time one token takes to read the weights, and the batch that saturates a chip, then size a KV cache, build one, and decode through it with the same answer as a full recompute. | No |
| `106-temperature-and-sampling` | From scores to a token: temperature and sampling | Build temperature, a top-k and top-p filter, and the full sampling step that draws one token from a vector of scores. | No |
| `107-speculative-decoding` | Speculative decoding: guess fast, verify once | Compute the acceptance rate and the residual distribution of a draft model, then write the accept-or-replace rule and check that its output follows the target. | No |
| `108-fine-tuning-memory-bill` | Why fine-tuning a big model runs out of memory | List the five pieces each trained parameter carries, compute the bill for a full fine-tune, and show how freezing most of the model changes it. | No |
| `109-lora` | LoRA: train a low-rank update, then merge it | Count the trained parameters, initialise the two low-rank factors, write the forward pass, then fold the update into the frozen weight and check that merged and split outputs agree. | No |
| `110-low-rank-reach-and-training` | What a rank-r update can reach, and how it is trained | Find the best rank-r approximation of an update with the SVD, measure the energy it keeps, and write the gradients of A and B for a squared loss. | No |
| `111-reward-from-preferences` | Learning a reward from human preferences | Turn a reward gap into the chance a person picks the chosen answer, build the Bradley-Terry loss without overflow, and write its gradient. | No |
| `112-kl-leash-and-optimal-policy` | Why a leash: KL divergence and the best policy | Compute KL divergence with the zero-probability rule, the best policy for a reward on a KL leash, and the penalised objective that proves nothing beats it. | No |
| `113-ppo-the-sampling-loop` | PPO: the sampling loop, in words and one clip | Compute the probability ratio, the normalised advantage, and PPO's clipped objective, then check that the clip stops pulling exactly where it should. | No |
| `114-dpo-implicit-reward` | DPO: training on the implicit reward | Build the implicit reward, the DPO loss, and its gradient on top of the Bradley-Terry functions you already wrote, and watch the unknown normaliser cancel. | No |
| `115-top-k-routing` | Send each token to a few experts | Score every expert, keep the top k, renormalise the weights, mix the chosen experts' outputs, and count stored versus used parameters. | No |
| `116-router-gradient` | Why the gate weight lets the router learn | Compute the gradient on each gate logit, assemble it into the router's weight gradient, and confirm it against finite differences. | No |
| `117-balancing-the-experts` | Keeping the experts balanced | Build the auxiliary load-balancing loss from each expert's traffic share and average probability, then compute expert capacity, run a capacity-limited forward pass that drops overflow assignments, and report the dropped fraction. | No |

## Projects

Each project puts several chapters to work on a realistic problem. The number
in its slug is the last chapter it needs, so `021p-project-price-houses` can
start once you have finished chapter 21.

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `004p-project-explore-before-you-model` | Explore before you model | Load a messy CSV of rental listings, clean it with pandas, summarise it with groupby and pivot tables, plot it with matplotlib, and show that cleaning cuts the honest test error of a straight-line fit. | No |
| `021p-project-price-houses` | Price houses honestly | Split, standardise, fit by QR, sweep ridge with one SVD, select columns with FISTA lasso, and fit a Newton logistic model for sells-in-30-days. | No |
| `036p-project-spam-filter` | Build and choose a spam filter | Fit naive Bayes and a kernel classifier, tune each with your own cross-validation, read a learning curve, and report a test score that selection never touched. | No |
| `050p-project-segment-customers` | Find customer segments and predict who leaves | Standardise, compress with PCA, cluster with k-means and a Gaussian mixture, then predict churn with a bagged ensemble of shallow trees. | No |
| `068p-project-read-digits` | Train a digit reader from scratch | Build a regularised, gradient-checked, early-stopped network in numpy that reads noisy 10 by 10 digits and clearly beats a linear baseline. | No |
| `087p-project-spot-defects` | Spot defects on a production line | Write im2col convolution and its backward pass, a residual block, global average pooling, and Adam. Train the network on noisy part photos and check that it beats a flat MLP. | No |
| `098p-project-train-a-tiny-transformer` | Train a tiny transformer to reverse digits | Build masked attention, positions, pre-norm blocks, an answer-only loss, and a training loop from torch tensors, then check the model on strings it has never seen. | No |
| `117p-project-serve-a-tiny-model` | Serve a small language model | Build a tokeniser, a KV-cached generator, a sampler, speculative decoding, and LoRA merging around a given tiny transformer, and prove each stage matches the slow, obvious version. | No |
