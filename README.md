# RS-Data_Splitter

- data splitter
    - `explicit`
    - `pointwise`: pointwise negative sampling data splitter for implict feedback
    - `pairwise`: pairwise negative sampling data splitter for implict feedback
    - `listwise`: listwise negative sampling data splitter for implict feedback

- `dataloader`: how to configure mini-batch
    - `general`
    - `curriculum`
    - `userpair`

- `trn_val_tst`
    - `trn` be structured in the form of `pointwise`, `pairwise` or `listwise`
    - `val` be structured in the form of `pointwise`, `pairwise` or `listwise`
    - `tst` be structured in the form of `pointwise` only
    - `loo` be structured in the form of `pointwise` only