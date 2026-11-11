# Curriculum

79 chapters in 8 parts, and 8 projects in a section of their own. Each
project lists the chapter to finish before you start it. Each row is one Markdown file under
`content/lessons/` and one directory under `content/exercises/`. The `Needs GPU`
column marks the exercises that run on RunPod rather than in the app container.

Each chapter teaches one topic: two or three closely related ideas in one arc,
with one exercise. The 33 lessons of the first edition were split into 132
one-idea chapters, merged into 117, and then merged again into 55. One added chapter brings the count to 56, scoring classifiers brings it to 57, and debugging and tuning a network brings it to 58. The map from your engine to PyTorch brings it to 59. A count model and word vectors bring it to 60. Autoencoders and variational autoencoders bring it to 61, in an eighth part. Diffusion models bring it to 62. Online softmax and tiled attention bring it to 63. Augmentation, transfer, separable convolutions, and patch embedding bring it to 64. Detection and segmentation brings it to 65. Contrastive learning brings it to 66. Learning theory brings it to 67. Feature pipelines and data leakage bring it to 68. Gaussian processes bring it to 69. Convex optimisation and Lagrange duality brings it to 70. Nearest neighbours and matrix factorisation bring it to 71. ICA and t-SNE bring it to 72. Approximation theory brings it to 73. Uncertainty and conformal prediction brings it to 74. Graph neural networks bring it to 75. Modern transformer design choices bring it to 76. Encoders and cross-attention bring it to 77. State-space models and linear attention bring it to 78. Gradient accumulation and activation checkpointing bring it to 79.
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
| `013-feature-pipelines-and-leakage` | Feature pipelines and data leakage | Build a standardiser and an imputer that store training-only statistics, a one-hot encoder with an unknown bucket, and an out-of-fold target encoder, then refit a pipeline inside every fold and show that a leaky one scores well on pure noise. | No |
| `014-learning-curves-and-winners-curse` | Learning curves and the winner's curse | Build a learning curve and read it, simulate why the best of many noisy scores is too good, and use nested cross-validation to score a penalty choice on data it never saw. | No |
| `015-learning-theory` | Learning theory: why generalisation bounds hold | Write Hoeffding's bound and check it by simulation, solve the union bound for the sample size and the gap, shatter points to find the VC dimension of thresholds, intervals, and half-planes, run a perceptron against its mistake bound, and sweep min-norm least squares across p = n to see double descent. | No |
| `016-scoring-classifiers` | Scoring classifiers: precision, recall, ROC, and calibration | Build confusion counts, precision, recall, and F1, trace an ROC curve and its area, choose a cut from a cost, then measure calibration error and fit a temperature. | No |
| `017-bayes-rule-and-gaussian-classes` | Bayes' rule and Gaussian classes | Write a stable log-sum-exp and the Bayes step, then fit class means, one shared covariance, and priors by averaging, and score points with a Cholesky-based Gaussian log density. | No |
| `018-gda-logistic-and-naive-bayes` | From Gaussian classes to logistic regression and naive Bayes | Turn fitted means, a shared covariance and priors into the weights, log odds and boundary of the equivalent logistic regression, then fit Gaussian and word-count naive Bayes and score documents. | No |
| `019-convex-duality` | Convex optimisation and Lagrange duality: bounds, certificates, and gaps | Build the Lagrangian and KKT residuals, solve an equality-constrained quadratic and ridge regression through their duals, and measure the lasso duality gap as a stopping test. | No |
| `020-support-vector-machines` | Support vector machines: the widest road, its dual, and SMO | Follow the hinge subgradient to the widest road, then describe the same road by pushes, test them with the KKT checklist, and find them two at a time. | No |
| `021-kernels-and-validity` | Kernels: similarity scores and what makes one valid | Compute RBF and polynomial kernel matrices with one matrix product, confirm the kernel trick on a degree-2 polynomial, test a Gram matrix with its smallest eigenvalue, and centre a Gram matrix so kernel PCA on a linear kernel matches PCA scores. | No |
| `022-kernel-ridge-and-random-features` | Kernel ridge regression and random Fourier features | Solve (K + lambda I) alpha = y with a Cholesky factor, check that extra centres never help, then replace the kernel with random Fourier features and fit ridge regression on them. | No |
| `023-gaussian-processes` | Gaussian processes: kernel ridge with error bars | Write the RBF kernel with a length scale, draw prior functions with a jittered Cholesky factor, return the posterior mean and variance and match the mean to kernel ridge at lambda = noise variance, then write the log marginal likelihood, its slope in log length scale, and a search that picks the length scale. | No |

## Part 3 — Trees, ensembles, and unsupervised learning

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `024-impurity-and-best-split` | Impurity, gain, and the best split | Write the three impurity scores, the size-weighted gain of a split, the cuts worth trying on a feature, and a best-split search that scores every cut in one sorted pass. | No |
| `025-grow-and-prune-trees` | Grow a tree, then prune it | Walk rows down a tree, grow one by splitting each box again until a stopping rule fires, then charge a fee per leaf and prune bottom-up. | No |
| `026-averaging-and-bagging` | Averaging models and bagging | Evaluate the variance of an average and simulate correlated models, then draw bootstrap samples, bag stumps, compute the out-of-bag error from the rows each stump never saw, then restrict splits to random columns, measure the correlation between trees, and rank columns by permutation importance. | No |
| `027-gradient-boosting` | Gradient boosting: fit what is still wrong | Write the negative gradients of two losses and a Newton leaf with its regularised split gain, then run gradient boosting with stumps and check that each round lowers the loss by the closed-form amount. | No |
| `028-principal-components` | Principal components: compress and know what you lost | Centre a cloud, read every direction from the SVD and the covariance, rebuild from k directions and confirm the error is the dropped squared singular values, then whiten with PCA and ZCA. | No |
| `029-power-iteration-and-probabilistic-pca` | Power iteration and probabilistic PCA | Read an eigenvalue with the Rayleigh quotient, find eigenvectors by multiplying and deflating, then write the E-step, the M-step, and the log-likelihood of probabilistic PCA and of factor analysis, which gives each column its own noise variance. | No |
| `030-ica-and-t-sne` | ICA and t-SNE: unmixing signals and drawing neighbourhoods | Write the FastICA update with kurtosis and log cosh contrasts and deflation, find each point's bandwidth by binary search on perplexity, symmetrise the probabilities, then check the Student-t t-SNE gradient against finite differences and run a seeded embedding of planted clusters. | No |
| `031-k-means` | k-means: Lloyd's steps and a good start | Write the k-means objective, the nearest-centre step, and Lloyd's loop, then seed the centres by squared distance and keep the best of several restarts. | No |
| `032-gaussian-mixtures-and-em` | Gaussian mixtures and EM | Write the log density, a safe log-sum-exp, the E-step, the weighted M-step, the lower bound, and the full EM loop, then watch the likelihood rise. | No |
| `033-nearest-neighbours-and-matrix-factorisation` | Nearest neighbours and matrix factorisation: predict from similar rows, then from a low-rank model | Write vectorised squared distances, k-NN classification and regression with a tie rule, a leave-one-out error that skips the row itself, and the nearest-to-farthest distance ratio. Then write the masked loss, one ALS row-by-row ridge step, a seeded ALS fit, and the global-mean and user-bias baselines, and beat them on held-out entries. | No |

## Part 4 — Neural networks

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `034-multilayer-perceptrons` | Multilayer perceptrons | Write ReLU, GELU, and their slopes, merge affine layers, run a forward pass, count the pieces of a folded network, build a hinge network through any knots, and count its parameters, flops, and stored activations. | No |
| `035-approximation-theory` | Approximation theory: how many units does a curve cost? | Build the tent map and the sawtooth from a two-unit ReLU layer, count straight pieces on a grid and check 2^k, compare deep and one-hidden-layer unit counts, measure interpolation error and its log-log slope, and count the cells a grid needs in d dimensions. | No |
| `036-backpropagation` | Backpropagation | Write the two-sided slope check, the multiply rule, the backward walk, unbroadcast, the matrix-product rule, and the ReLU rule, so slopes flow through a layer that multiplies, adds a bias, and bends. | No |
| `037-softmax-and-whole-network-gradients` | Softmax and whole-network gradients | Write a stable softmax and cross-entropy whose slope is prediction minus truth, wire the engine into a two-layer network, and write the finite-difference check that proves its gradients. | No |
| `038-sgd-adam-and-schedules` | SGD, Adam, and learning-rate schedules | Compute mini-batch gradients, then build SGD, momentum, Adam, and AdamW steps, a warmup-plus-cosine schedule, and the gradient noise scale, and run them on noisy batches of a long valley. | No |
| `039-signal-size-and-initialisation` | Signal size through layers, and Xavier and He initialisation | Compute a layer's gain and the gain after many layers, simulate a random stack, then write the Xavier and He scales and check them on a fifty-layer ReLU network. | No |
| `040-normalisation-layers` | Normalisation layers: batch norm, layer norm, and their gradients | Rescale a group to mean 0 and variance 1, build batch norm and layer norm forward, then write their backward passes and check them against finite differences. | No |
| `041-weight-decay-and-early-stopping` | Weight decay and early stopping: two ways to keep weights small | Write the weight-decay step and the angle a sideways step turns a weight vector through, then turn a step count into a ridge penalty and pick a stopping step. | No |
| `042-dropout-and-label-smoothing` | Dropout and label smoothing: keep a network from trusting itself | Draw a keep-mask, apply inverted dropout and send the gradient back through it, then build the smoothed target, its loss, and its gradient. | No |
| `043-debugging-and-tuning` | Debugging and tuning a network | Predict the untrained loss and flag a bad initialisation, pick a learning rate from a seeded bowl, draw log-uniform settings, and read the update-to-weight ratio. | No |
| `044-uncertainty-and-conformal-prediction` | Uncertainty: ensembles, MC dropout, and conformal prediction | Read the mean and variance of an ensemble and of MC dropout passes, then take the conformal quantile with the (n + 1) correction, build regression intervals whose coverage holds over 200 seeded splits, and build classifier prediction sets from 1 - p of the true class. | No |

## Part 5 — Vision and sequences

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `045-from-the-engine-to-pytorch` | From your engine to PyTorch | Write Linear and MLP modules whose outputs and gradients match torch.nn.Linear, an SGD step that zeros the gradient, one training epoch whose loss falls, an eval pass, and an einsum blend that matches the attention loop. | No |
| `046-convolution-and-receptive-field` | Convolution: slide a filter, count the weights, count the view | Write a sliding filter for one channel and for a batch of images, count the weights of three layer types, then count output sizes and receptive fields and pool. | No |
| `047-im2col-and-convolution-backward` | Convolution as a matrix multiply, forward and backward | Lay every window out as a column, multiply by the flattened kernels, then write col2im as a scatter-add and build dx, dw, and db. Gradients are checked against finite differences. | No |
| `048-residual-networks` | Residual networks: the shortcut, the gradient, and the bottleneck | Write a block that outputs the input plus a correction and send the gradient back through its plus sign. Then measure the gradient that survives fifty layers with and without a shortcut, and count the cost of a bottleneck. | No |
| `049-augmentation-transfer-and-vit` | Beyond the plain CNN: augment, transfer, separate, and patchify | Write a seeded crop-and-flip and a mixup that blends labels, count a depthwise-separable layer and match it to a full convolution, and cut an image into patches whose embedding equals a strided convolution. Then fit a linear head on a frozen extractor and write an update that skips frozen weights. | No |
| `050-detection-and-segmentation` | Detection and segmentation: IoU, NMS, average precision, and Dice | Write a pairwise box IoU, greedy non-maximum suppression, a matcher that labels detections as hits or misses, and all-point average precision. Then write pixel IoU, Dice, and a soft Dice loss whose gradient is checked against finite differences. | No |
| `051-contrastive-learning` | Contrastive learning: InfoNCE, collapse, and CLIP | Write InfoNCE as a row-wise softmax over cosine similarities and its gradient through the normalisation, then CLIP's symmetric loss, the alignment and uniformity scores, and a linear encoder trained on planted clusters and scored with k-nearest neighbours. | No |
| `052-recurrent-cells-and-bptt` | Recurrent cells and backpropagation through time | Write one step of a recurrent cell, run it over a sequence, and read out the last state for a loss. Then write the gradient of one step and the backward loop that adds each step's own loss to the carry from later steps. Every output is checked against finite differences. | No |
| `053-vanishing-gradients-and-clipping` | Vanishing gradients, exploding gradients, and clipping | Compute the spectral radius, predict how large a gradient is after many steps, and compare with your own backward pass. Then split a long sequence into chunks, backpropagate through each chunk alone, and clip a gradient by its global norm. | No |
| `054-the-lstm` | The LSTM, a belt with three gates | Compute the gradient left after many steps for a plain net and for a belt, build the running product of forget gates, write the gate that decides how much to let through, and write the full forward step of an LSTM cell. | No |
| `055-lstm-backward-and-the-gru` | Backpropagation through the LSTM, and the GRU | Collect the two paths into the cell state and send the gradient back through the gates, weights, input, and previous cell, checked against finite differences. Then write the GRU blend, the full GRU step with its reset gate, and a parameter count for three kinds of cell. | No |
| `056-attention-as-soft-lookup` | Attention as a soft lookup, with a dot product or a learned scorer | Write a stable softmax, the blend that turns scores into a context vector, a dot-product scorer with an optional matrix, an additive scorer, and the parameters that make the additive scorer imitate a dot product. | No |
| `057-attention-gradients-and-alignment` | Attention gradients and the alignment matrix | Write the backward rule for the softmax, the blend, and the whole additive attention, checked against finite differences. Then stack the weights into an alignment matrix and hand-set a scorer so each step reads the right source position. | No |
| `058-graph-neural-networks` | Graph neural networks: message passing, GCN, and over-smoothing | Write the symmetric normalised adjacency, a GCN layer and its gradients checked against finite differences, and sum, mean, and max message passing with a graph readout. Check that renumbering the nodes permutes the outputs and leaves the readout alone, measure over-smoothing as layers stack, and train a two-layer GCN on a two-community graph from six labels. | No |

## Part 6 — Transformers

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `059-scaled-attention-and-heads` | Scaled attention, many heads, and the causal mask | Write the pieces of a soft lookup, measure how the spread of dot products grows with the key width, and build scaled dot-product attention. Then add the causal mask and multi-head attention. | No |
| `060-attention-backward-and-cost` | The gradient and the cost of attention | Write the softmax backward rule, chain it into scaled dot-product attention and into multi-head attention with a causal mask and an output projection, and count the flops of one layer. | No |
| `061-online-softmax-and-tiled-attention` | Online softmax and tiled attention | Write a one-pass softmax, the rule that merges two partial states, attention over small tiles with a causal block skip, and a backward pass that rebuilds the weights from the saved log-sum-exp. Count the memory and flops against the dense layer. | No |
| `062-position-encodings` | Position encodings: sinusoidal clocks and rotary embeddings | Write the sinusoidal clock encoding, the matrix that moves it k positions, and the rotary tables that make a score depend only on the offset. | No |
| `063-the-transformer-block` | The transformer block: SwiGLU, residuals, and layer norm | Write a stable SiLU, the gated feed-forward network and its weight budget, layer norm, and the residual block that wires them to rotary attention. | No |
| `064-modern-transformer-choices` | Modern transformer choices: RMSNorm, grouped-query attention, and parameter budgets | Write RMSNorm and its backward pass, grouped-query attention, the key-value cache size, a parameter count that matches GPT-2 small and Llama 3 8B, and a pre-norm and post-norm stack whose depth behaviour you measure. | No |
| `065-encoders-and-cross-attention` | Encoders, decoders, and cross-attention | Write padding and causal masks, a masked softmax, cross-attention that ignores padded encoder positions, BERT-style token masking with a loss on the chosen positions only and its gradient, and T5-style span corruption with its lengths. | No |
| `066-state-space-models` | State-space models and linear attention | Write causal linear attention as a quadratic masked form and as a running sum, a state-space model as a recurrence and as a convolution with the kernel C A^j B, a zero-order-hold discretisation, and an associative parallel scan with its backward pass and a selective model. | No |
| `067-counts-to-a-language-model` | From counts to a language model, and word vectors | Smooth a bigram table, match it with a softmax, scatter-add an embedding gradient, and train a window model and a skip-gram so a planted neighbour is nearer. | No |
| `068-next-token-loss-and-safe-steps` | Training a language model: next-token loss, schedules, and clipping | Write the per-token loss without overflow, cut shifted windows, read the loss as perplexity, then build the warmup-cosine schedule and the global-norm clip. | No |
| `069-training-loop-and-cost` | The training step, and what a run costs | Write one AdamW update, wire loss, backward, clip, schedule, and update into train_step, then estimate a run's flops and memory and simulate loss scaling through float16. | No |
| `070-gradient-accumulation-and-checkpointing` | Gradient accumulation and activation checkpointing | Count activation bytes, accumulate micro-batch gradients into the exact full-batch gradient for equal and unequal sizes, then store only segment boundaries and recompute the rest in the backward pass, counting the peak and the rebuilds. | No |

## Part 7 — Large language models

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `071-tokenisation` | Tokenisation: bytes, byte-pair encoding, and what a tokeniser charges | Turn text into bytes, train a byte-pair encoder and replay it to encode and decode, then measure fertility, tokens per character, and digit cuts. | No |
| `072-scaling-laws` | Scaling laws and the compute budget | Fit and extrapolate a power law on log axes, split a loss into a floor and two penalties, count the weights that multiply, price training in flops and days, then find the best split of a budget between model size and data. | No |
| `073-decoding-and-sampling` | Decoding and sampling: cache, temperature, speculation | Count flops against bytes for decoding, size and build a KV cache, turn scores into a token with temperature, top-k, and top-p, search with a beam, and write the accept-or-replace rule of speculative decoding. | No |
| `074-fine-tuning-and-lora` | Fine-tuning on a budget with LoRA | Add up the memory bill with and without frozen weights, write the LoRA forward pass and merge, then truncate an update with the SVD and write the gradients of A and B. | No |
| `075-reward-models-and-the-kl-leash` | Reward models and the KL leash | Build the Bradley-Terry loss and gradient for a reward model, then compute KL divergence, the best policy for a reward on a KL leash, and the penalised objective that proves nothing beats it. | No |
| `076-ppo-and-dpo` | PPO and DPO: two ways to reach the leashed optimum | Compute PPO's probability ratio, advantage, and clipped objective, then build the DPO implicit reward, loss, and gradient on top of the Bradley-Terry functions. | No |
| `077-mixtures-of-experts` | Mixtures of experts: routing, the router's gradient, and balance | Score and pick the top k experts, compute the router's gradient through the gate weights, build the load-balancing loss, and run a capacity-limited forward pass that drops overflow. | No |

## Part 8 — Generative models

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `078-autoencoders-and-vaes` | Autoencoders and variational autoencoders | Write the diagonal-Gaussian KL and its gradient, reparameterise a sample, check the ELBO against a known log probability, recover a principal component with a linear autoencoder, and train a tiny VAE. | No |
| `079-diffusion-models` | Diffusion models: add noise, then walk it back | Write the cumulative noise schedule and the closed-form noisy sample, the noise-prediction loss and its gradient, one ancestral step and one deterministic step, ancestral samples from an exact two-bell denoiser, and a classifier-free guidance step. | No |

## Projects

Each project puts several chapters to work on a realistic problem. The number
in its slug is the last chapter it needs, so `010p-project-price-houses` can
start once you have finished chapter 10.

| Slug | Title | Exercise delivers | Needs GPU |
|---|---|---|---|
| `002p-project-explore-before-you-model` | Explore before you model | Load a messy CSV of rental listings, clean it with pandas, summarise it with groupby and pivot tables, plot it with matplotlib, and show that cleaning cuts the honest test error of a straight-line fit. | No |
| `010p-project-price-houses` | Price houses honestly | Split, standardise, fit by QR, sweep ridge with one SVD, select columns with FISTA lasso, and fit a Newton logistic model for sells-in-30-days. | No |
| `023p-project-spam-filter` | Build and choose a spam filter | Fit naive Bayes and a kernel classifier, tune each with your own cross-validation, read a learning curve, and report a test score that selection never touched. | No |
| `033p-project-segment-customers` | Find customer segments and predict who leaves | Standardise, compress with PCA, cluster with k-means and a Gaussian mixture, then predict churn with a bagged ensemble of shallow trees. | No |
| `044p-project-read-digits` | Train a digit reader from scratch | Build a regularised, gradient-checked, early-stopped network in numpy that reads noisy 10 by 10 digits and clearly beats a linear baseline. | No |
| `058p-project-spot-defects` | Spot defects on a production line | Write im2col convolution and its backward pass, a residual block, global average pooling, and Adam. Train the network on noisy part photos and check that it beats a flat MLP. | No |
| `070p-project-train-a-tiny-transformer` | Train a tiny transformer to reverse digits | Build masked attention, positions, pre-norm blocks, an answer-only loss, and a training loop from torch tensors, then check the model on strings it has never seen. | No |
| `077p-project-serve-a-tiny-model` | Serve a small language model | Build a tokeniser, a KV-cached generator, a sampler, speculative decoding, and LoRA merging around a given tiny transformer, and prove each stage matches the slow, obvious version. | No |
