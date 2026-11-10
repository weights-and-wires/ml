# Curriculum

62 chapters in 8 parts, and 8 projects in a section of their own. Each
project lists the chapter to finish before you start it. Each row is one Markdown file under
`content/lessons/` and one directory under `content/exercises/`. The `Needs GPU`
column marks the exercises that run on RunPod rather than in the app container.

Each chapter teaches one topic: two or three closely related ideas in one arc,
with one exercise. The 33 lessons of the first edition were split into 132
one-idea chapters, merged into 117, and then merged again into 55. One added chapter brings the count to 56, scoring classifiers brings it to 57, and debugging and tuning a network brings it to 58. The map from your engine to PyTorch brings it to 59. A count model and word vectors bring it to 60. Autoencoders and variational autoencoders bring it to 61, in an eighth part. Diffusion models bring it to 62.
`content/renames.json` maps each old lesson and chapter to the chapter that
replaced it, so readers keep their progress.

## Part 1 — Foundations

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `001-loss-and-honest-scores` | Scoring a rule honestly: loss, risk, and held-out flats | Write the squared loss, the empirical risk, and a slope search, then a seeded train/test split and a memorising rule, and measure the gap between training and test risk. | No |
| `002-curves-and-bias-variance` | Fitting a curve, and the two ways it fails: degree, bias, and variance | Build the design matrix, fit weights by least squares, sweep the degree, then split the error into bias, variance, and noise with a hand example and a Monte Carlo experiment. | No |
| `003-least-squares-and-projection` | Least squares as a bowl and a right angle | Write the squared-miss loss and its gradient, solve the normal equations, then measure how far a fit is from perpendicular, project the targets onto the column space, and decide when a fit is the best one. | No |
| `004-stable-least-squares` | Lost digits, and how QR and the SVD keep them | Measure a condition number, solve the normal equations with Cholesky, then solve with QR and the SVD, return the shortest weights for repeated columns, and count the rank. | No |
| `005-gradient-descent-and-momentum` | Gradient descent, step size, and momentum | Build the loss and gradient of a bowl and gradient descent, read the curvatures to predict the safe step and the step count, then add heavy-ball and Nesterov momentum and measure the speedup. | No |
| `006-ridge-regression` | Ridge regression: a fee on large weights, for every lambda | Write the ridge score and its gradient, solve for the bottom with one linear system, then compute the ridge weights for every lambda from one SVD and count the degrees of freedom each lambda leaves. | No |
| `007-lasso-and-fista` | The lasso: exact zeros, soft thresholding, and FISTA | Build the shrink-and-kill map for one number, solve the lasso on the identity, then fit any design with ISTA and FISTA and test the finish with the KKT rules. | No |
| `008-sigmoid-and-log-loss` | From a score to a probability, and how wrong it is | Build a sigmoid that stays finite at a score of 800, its inverse the logit, a stable softplus, and the log loss softplus(z) - y z with a ridge fee. | No |
| `009-fitting-logistic-regression` | Fitting logistic regression: gradient descent and Newton's method | Build the gradient X^T(p - y), the Hessian X^T D X + lambda I, a descent loop, a Newton loop, and an accuracy function, then compare how many passes each loop needs. | No |
| `010-exponential-families-and-glms` | Exponential families and generalised linear models | Write A(eta), its slope and bend for three families, then the loss A(eta) - y eta, its gradient X^T(mu - y), the bend table X^T D X, and a Fisher scoring loop that fits all three. | No |
| `011-likelihood-map-and-bayesian-regression` | Likelihood, MAP, and the Bayesian view | Write the Gaussian negative log likelihood and the MAP objective, match its minimiser to ridge with lambda = sigma2/tau2, then return the posterior mean and covariance and a predictive variance that grows away from the data. | No |

## Part 2 — Classical models

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `012-cross-validation-and-leave-one-out` | Cross-validation and the leave-one-out shortcut | Cut the data into folds and choose a ridge penalty with k-fold cross-validation, then build the hat matrix and get the leave-one-out error from a single fit. | No |
| `013-learning-curves-and-winners-curse` | Learning curves and the winner's curse | Build a learning curve and read it, simulate why the best of many noisy scores is too good, and use nested cross-validation to score a penalty choice on data it never saw. | No |
| `014-scoring-classifiers` | Scoring classifiers: precision, recall, ROC, and calibration | Build confusion counts, precision, recall, and F1, trace an ROC curve and its area, choose a cut from a cost, then measure calibration error and fit a temperature. | No |
| `015-bayes-rule-and-gaussian-classes` | Bayes' rule and Gaussian classes | Write a stable log-sum-exp and the Bayes step, then fit class means, one shared covariance, and priors by averaging, and score points with a Cholesky-based Gaussian log density. | No |
| `016-gda-logistic-and-naive-bayes` | From Gaussian classes to logistic regression and naive Bayes | Turn fitted means, a shared covariance and priors into the weights, log odds and boundary of the equivalent logistic regression, then fit Gaussian and word-count naive Bayes and score documents. | No |
| `017-support-vector-machines` | Support vector machines: the widest road, its dual, and SMO | Follow the hinge subgradient to the widest road, then describe the same road by pushes, test them with the KKT checklist, and find them two at a time. | No |
| `018-kernels-and-validity` | Kernels: similarity scores and what makes one valid | Compute RBF and polynomial kernel matrices with one matrix product, confirm the kernel trick on a degree-2 polynomial, test a Gram matrix with its smallest eigenvalue, and centre a Gram matrix so kernel PCA on a linear kernel matches PCA scores. | No |
| `019-kernel-ridge-and-random-features` | Kernel ridge regression and random Fourier features | Solve (K + lambda I) alpha = y with a Cholesky factor, check that extra centres never help, then replace the kernel with random Fourier features and fit ridge regression on them. | No |

## Part 3 — Trees, ensembles, and unsupervised learning

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `020-impurity-and-best-split` | Impurity, gain, and the best split | Write the three impurity scores, the size-weighted gain of a split, the cuts worth trying on a feature, and a best-split search that scores every cut in one sorted pass. | No |
| `021-grow-and-prune-trees` | Grow a tree, then prune it | Walk rows down a tree, grow one by splitting each box again until a stopping rule fires, then charge a fee per leaf and prune bottom-up. | No |
| `022-averaging-and-bagging` | Averaging models and bagging | Evaluate the variance of an average and simulate correlated models, then draw bootstrap samples, bag stumps, compute the out-of-bag error from the rows each stump never saw, then restrict splits to random columns, measure the correlation between trees, and rank columns by permutation importance. | No |
| `023-gradient-boosting` | Gradient boosting: fit what is still wrong | Write the negative gradients of two losses and a Newton leaf with its regularised split gain, then run gradient boosting with stumps and check that each round lowers the loss by the closed-form amount. | No |
| `024-principal-components` | Principal components: compress and know what you lost | Centre a cloud, read every direction from the SVD and the covariance, rebuild from k directions and confirm the error is the dropped squared singular values, then whiten with PCA and ZCA. | No |
| `025-power-iteration-and-probabilistic-pca` | Power iteration and probabilistic PCA | Read an eigenvalue with the Rayleigh quotient, find eigenvectors by multiplying and deflating, then write the E-step, the M-step, and the log-likelihood of probabilistic PCA and of factor analysis, which gives each column its own noise variance. | No |
| `026-k-means` | k-means: Lloyd's steps and a good start | Write the k-means objective, the nearest-centre step, and Lloyd's loop, then seed the centres by squared distance and keep the best of several restarts. | No |
| `027-gaussian-mixtures-and-em` | Gaussian mixtures and EM | Write the log density, a safe log-sum-exp, the E-step, the weighted M-step, the lower bound, and the full EM loop, then watch the likelihood rise. | No |

## Part 4 — Neural networks

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `028-multilayer-perceptrons` | Multilayer perceptrons | Write ReLU, GELU, and their slopes, merge affine layers, run a forward pass, count the pieces of a folded network, build a hinge network through any knots, and count its parameters, flops, and stored activations. | No |
| `029-backpropagation` | Backpropagation | Write the two-sided slope check, the multiply rule, the backward walk, unbroadcast, the matrix-product rule, and the ReLU rule, so slopes flow through a layer that multiplies, adds a bias, and bends. | No |
| `030-softmax-and-whole-network-gradients` | Softmax and whole-network gradients | Write a stable softmax and cross-entropy whose slope is prediction minus truth, wire the engine into a two-layer network, and write the finite-difference check that proves its gradients. | No |
| `031-sgd-adam-and-schedules` | SGD, Adam, and learning-rate schedules | Compute mini-batch gradients, then build SGD, momentum, Adam, and AdamW steps, a warmup-plus-cosine schedule, and the gradient noise scale, and run them on noisy batches of a long valley. | No |
| `032-signal-size-and-initialisation` | Signal size through layers, and Xavier and He initialisation | Compute a layer's gain and the gain after many layers, simulate a random stack, then write the Xavier and He scales and check them on a fifty-layer ReLU network. | No |
| `033-normalisation-layers` | Normalisation layers: batch norm, layer norm, and their gradients | Rescale a group to mean 0 and variance 1, build batch norm and layer norm forward, then write their backward passes and check them against finite differences. | No |
| `034-weight-decay-and-early-stopping` | Weight decay and early stopping: two ways to keep weights small | Write the weight-decay step and the angle a sideways step turns a weight vector through, then turn a step count into a ridge penalty and pick a stopping step. | No |
| `035-dropout-and-label-smoothing` | Dropout and label smoothing: keep a network from trusting itself | Draw a keep-mask, apply inverted dropout and send the gradient back through it, then build the smoothed target, its loss, and its gradient. | No |
| `036-debugging-and-tuning` | Debugging and tuning a network | Predict the untrained loss and flag a bad initialisation, pick a learning rate from a seeded bowl, draw log-uniform settings, and read the update-to-weight ratio. | No |

## Part 5 — Vision and sequences

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `037-from-the-engine-to-pytorch` | From your engine to PyTorch | Write Linear and MLP modules whose outputs and gradients match torch.nn.Linear, an SGD step that zeros the gradient, one training epoch whose loss falls, an eval pass, and an einsum blend that matches the attention loop. | No |
| `038-convolution-and-receptive-field` | Convolution: slide a filter, count the weights, count the view | Write a sliding filter for one channel and for a batch of images, count the weights of three layer types, then count output sizes and receptive fields and pool. | No |
| `039-im2col-and-convolution-backward` | Convolution as a matrix multiply, forward and backward | Lay every window out as a column, multiply by the flattened kernels, then write col2im as a scatter-add and build dx, dw, and db. Gradients are checked against finite differences. | No |
| `040-residual-networks` | Residual networks: the shortcut, the gradient, and the bottleneck | Write a block that outputs the input plus a correction and send the gradient back through its plus sign. Then measure the gradient that survives fifty layers with and without a shortcut, and count the cost of a bottleneck. | No |
| `041-recurrent-cells-and-bptt` | Recurrent cells and backpropagation through time | Write one step of a recurrent cell, run it over a sequence, and read out the last state for a loss. Then write the gradient of one step and the backward loop that adds each step's own loss to the carry from later steps. Every output is checked against finite differences. | No |
| `042-vanishing-gradients-and-clipping` | Vanishing gradients, exploding gradients, and clipping | Compute the spectral radius, predict how large a gradient is after many steps, and compare with your own backward pass. Then split a long sequence into chunks, backpropagate through each chunk alone, and clip a gradient by its global norm. | No |
| `043-the-lstm` | The LSTM, a belt with three gates | Compute the gradient left after many steps for a plain net and for a belt, build the running product of forget gates, write the gate that decides how much to let through, and write the full forward step of an LSTM cell. | No |
| `044-lstm-backward-and-the-gru` | Backpropagation through the LSTM, and the GRU | Collect the two paths into the cell state and send the gradient back through the gates, weights, input, and previous cell, checked against finite differences. Then write the GRU blend, the full GRU step with its reset gate, and a parameter count for three kinds of cell. | No |
| `045-attention-as-soft-lookup` | Attention as a soft lookup, with a dot product or a learned scorer | Write a stable softmax, the blend that turns scores into a context vector, a dot-product scorer with an optional matrix, an additive scorer, and the parameters that make the additive scorer imitate a dot product. | No |
| `046-attention-gradients-and-alignment` | Attention gradients and the alignment matrix | Write the backward rule for the softmax, the blend, and the whole additive attention, checked against finite differences. Then stack the weights into an alignment matrix and hand-set a scorer so each step reads the right source position. | No |

## Part 6 — Transformers

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `047-scaled-attention-and-heads` | Scaled attention, many heads, and the causal mask | Write the pieces of a soft lookup, measure how the spread of dot products grows with the key width, and build scaled dot-product attention. Then add the causal mask and multi-head attention. | No |
| `048-attention-backward-and-cost` | The gradient and the cost of attention | Write the softmax backward rule, chain it into scaled dot-product attention and into multi-head attention with a causal mask and an output projection, and count the flops of one layer. | No |
| `049-position-encodings` | Position encodings: sinusoidal clocks and rotary embeddings | Write the sinusoidal clock encoding, the matrix that moves it k positions, and the rotary tables that make a score depend only on the offset. | No |
| `050-the-transformer-block` | The transformer block: SwiGLU, residuals, and layer norm | Write a stable SiLU, the gated feed-forward network and its weight budget, layer norm, and the residual block that wires them to rotary attention. | No |
| `051-counts-to-a-language-model` | From counts to a language model, and word vectors | Smooth a bigram table, match it with a softmax, scatter-add an embedding gradient, and train a window model and a skip-gram so a planted neighbour is nearer. | No |
| `052-next-token-loss-and-safe-steps` | Training a language model: next-token loss, schedules, and clipping | Write the per-token loss without overflow, cut shifted windows, read the loss as perplexity, then build the warmup-cosine schedule and the global-norm clip. | No |
| `053-training-loop-and-cost` | The training step, and what a run costs | Write one AdamW update, wire loss, backward, clip, schedule, and update into train_step, then estimate a run's flops and memory and simulate loss scaling through float16. | No |

## Part 7 — Large language models

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `054-tokenisation` | Tokenisation: bytes, byte-pair encoding, and what a tokeniser charges | Turn text into bytes, train a byte-pair encoder and replay it to encode and decode, then measure fertility, tokens per character, and digit cuts. | No |
| `055-scaling-laws` | Scaling laws and the compute budget | Fit and extrapolate a power law on log axes, split a loss into a floor and two penalties, count the weights that multiply, price training in flops and days, then find the best split of a budget between model size and data. | No |
| `056-decoding-and-sampling` | Decoding and sampling: cache, temperature, speculation | Count flops against bytes for decoding, size and build a KV cache, turn scores into a token with temperature, top-k, and top-p, and write the accept-or-replace rule of speculative decoding. | No |
| `057-fine-tuning-and-lora` | Fine-tuning on a budget with LoRA | Add up the memory bill with and without frozen weights, write the LoRA forward pass and merge, then truncate an update with the SVD and write the gradients of A and B. | No |
| `058-reward-models-and-the-kl-leash` | Reward models and the KL leash | Build the Bradley-Terry loss and gradient for a reward model, then compute KL divergence, the best policy for a reward on a KL leash, and the penalised objective that proves nothing beats it. | No |
| `059-ppo-and-dpo` | PPO and DPO: two ways to reach the leashed optimum | Compute PPO's probability ratio, advantage, and clipped objective, then build the DPO implicit reward, loss, and gradient on top of the Bradley-Terry functions. | No |
| `060-mixtures-of-experts` | Mixtures of experts: routing, the router's gradient, and balance | Score and pick the top k experts, compute the router's gradient through the gate weights, build the load-balancing loss, and run a capacity-limited forward pass that drops overflow. | No |

## Part 8 — Generative models

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `061-autoencoders-and-vaes` | Autoencoders and variational autoencoders | Write the diagonal-Gaussian KL and its gradient, reparameterise a sample, check the ELBO against a known log probability, recover a principal component with a linear autoencoder, and train a tiny VAE. | No |
| `062-diffusion-models` | Diffusion models: add noise, then walk it back | Write the cumulative noise schedule and the closed-form noisy sample, the noise-prediction loss and its gradient, one ancestral step and one deterministic step, ancestral samples from an exact two-bell denoiser, and a classifier-free guidance step. | No |

## Projects

Each project puts several chapters to work on a realistic problem. The number
in its slug is the last chapter it needs, so `010p-project-price-houses` can
start once you have finished chapter 10.

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `002p-project-explore-before-you-model` | Explore before you model | Load a messy CSV of rental listings, clean it with pandas, summarise it with groupby and pivot tables, plot it with matplotlib, and show that cleaning cuts the honest test error of a straight-line fit. | No |
| `010p-project-price-houses` | Price houses honestly | Split, standardise, fit by QR, sweep ridge with one SVD, select columns with FISTA lasso, and fit a Newton logistic model for sells-in-30-days. | No |
| `019p-project-spam-filter` | Build and choose a spam filter | Fit naive Bayes and a kernel classifier, tune each with your own cross-validation, read a learning curve, and report a test score that selection never touched. | No |
| `027p-project-segment-customers` | Find customer segments and predict who leaves | Standardise, compress with PCA, cluster with k-means and a Gaussian mixture, then predict churn with a bagged ensemble of shallow trees. | No |
| `036p-project-read-digits` | Train a digit reader from scratch | Build a regularised, gradient-checked, early-stopped network in numpy that reads noisy 10 by 10 digits and clearly beats a linear baseline. | No |
| `046p-project-spot-defects` | Spot defects on a production line | Write im2col convolution and its backward pass, a residual block, global average pooling, and Adam. Train the network on noisy part photos and check that it beats a flat MLP. | No |
| `053p-project-train-a-tiny-transformer` | Train a tiny transformer to reverse digits | Build masked attention, positions, pre-norm blocks, an answer-only loss, and a training loop from torch tensors, then check the model on strings it has never seen. | No |
| `060p-project-serve-a-tiny-model` | Serve a small language model | Build a tokeniser, a KV-cached generator, a sampler, speculative decoding, and LoRA merging around a given tiny transformer, and prove each stage matches the slow, obvious version. | No |
