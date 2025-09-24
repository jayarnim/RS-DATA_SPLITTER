# Recommendation Model Training Module

- data splitter
    - `pointwise`: pointwise negative sampling data splitter for implict feedback
    - `pairwise`: pairwise negative sampling data splitter for implict feedback
    - `listwise`: listwise negative sampling data splitter for implict feedback

- loop
    - `loop`: trn, val, monitoring process for total epochs
    - `trainer`: trn, val process for one epoch step
    - `monitor`: early stopping monitor, based on msr
    - `predictor`: performance evaluation process for one epoch step
    - `loss_fn`: recsys loss function