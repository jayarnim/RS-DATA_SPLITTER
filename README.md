# Training Module for Latent Factor Models with Implicit Feedback

- created by @jayarnim

## data splitter

The `DATA_SPLITTER` is a package designed for latent factor model experiments in recommender systems, providing various forms of processed implicit feedback datasets.

### loaders

It divides a binary implicit feedback dataset into `trn`, `val` and `tst` in a user-level hierarchical manner, and constructs corresponding PyTorch DataLoaders for each split. Additionally, to monitor the early-stopping criterion based on evaluation metrics, it provides a `leave-one-out` dataset, which contains a single positive feedback per user.

The top-level module is `data_splitter`. Through the class `DataSplitter`, users can perform dataset splitting and generate PyTorch DataLoaders with negative sampling applied to each split. The split ratios and negative sampling ratios can be configured via the `__call__` method parameters `trn_val_tst_ratio` and `neg_per_pos_ratio`, respectively.

Depending on the learning strategy, the DataLoader provides various observation pairs. This can be specified through the `__init__` parameter `learning_type` of the `DataSplitter` class. However, note that while `pairwise` and `listwise` learning types can be used for `trn` and `val` dataloaders, the `tst` and `loo` dataloaders are strictly `pointwise`, since they are intended for evaluation, not training.

- `pointwise`: (user idx, item idx, label)
- `pairwise`: (user idx, pos idx, neg idx)
- `listwise`: (user idx, pos idx, neg indices)

### user-item interaction matrix

Models such as DMF, DeepCF, J-NCF, DNCF, and DDFL utilize history embeddings, where the user–item interaction matrix and its transpose are linearly transformed to generate vector representations of users and items. For this purpose, a user–item interaction matrix constructed from the training set only is provided.

Note that user and item indices include padding indices; since Python indexing starts from zero, the shape of the interaction matrix is (M+1) × (N+1) rather than M × N.

### interaction histories

Models such as COMET, DRNet, DELF, FSIM, and NAIS aggregate user or item history vectors in various ways. For this purpose, user histories and item histories are provided, also generated using the training set only. Again, note that user and item indices include padding indices.

The `__call__` parameter `max_hist` allows limiting the maximum number of historical interactions per user or item, and the parameter hist_selector_type determines the criterion for selecting historical items.

Currently supported options for `hist_selector_type` (`__call__` parameter) are:

- `default`: use the full history
- `tfidf`: select history items by TF-IDF score (treating a user’s history as a document and items as words)

## experiment

The `EXPERIMENT` is a package designed to train latent factor models and evaluate their ranking performance in recommender systems.

The top-level module, `runner`, serves as the main controller for training. It takes as input the target model, along with two submodules — the `trainer`, which handles single-epoch mini-batch training, and the `monitor`, which tracks performance metrics to determine early-stopping points.

The `trainer` manages the `trn` and `val` processes for a single epoch, supporting pointwise, pairwise, and listwise learning strategies. It provides several loss functions defined in `loss_fn`, which can be selected using the loss_fn_type parameter in the class’s `__init__` method.

The supported loss functions for each learning strategy are as follows:

- pointwise: `bce`
- pairwise: `bpr`
- listwise: `climf`

The `monitor` determines the early-stopping point based on performance evaluated using the `leave-one-out` dataset. The evaluation metric can be configured via the `metric_fn_type` parameter, and the currently supported metrics include: `hr`, `precision`, `recall`, `map`, and `ndcg`.

The `evaluator`, independently executable with respect to the `runner`, assesses the model’s performance using the `tst` dataset. It computes `hr`, `precision`, `recall`, `map`, and `ndcg` scores for each top-k threshold. The top-k values can be specified by the user, with the default set being `[5, 10, 15, 20, 25, 50, 100]`.