Chainer Variational Factorization Machine implementation.

Run `python run_movielens.py` to download, extract, and run the FM model
on MovieLens 1M data. 


| model     | batchsize | rank |intx term | lambda0 | lambda1 | lambda2 | RMSE   | Notes |
|-----------|-----------|------|----------|---------|---------|---------| -------| ----- |
| FM        | 8192      |  0   | N        |0        | 1e-2    | 0       | 0.9305 | Regression with regularization |
| FM        | 8192      |  0   | N        |0        | 0       | 0       | 0.9115 | Regression with no regularization |
| FM        | 8192      |  0   | N        |0        | 1e-3    | 0       | 0.9112 | Regression with less regularization |
| FM        | 8192      | 20   | Y        |0        | 0       | 1e-3    | 0.8633 | FM model w/ 20D latent vector |
| FM        | 8192      | 20   | Y        |0        | 1e-3    | 1e-3    | 0.8618 | FM model w/ 20D latent vector and regularization |
|VFM        | 8192      | 20   | Y        |0        | 1e-3    | 1e-3    | 0.8625 | Variational FM model with arbitrary reularization|
|VFM        | 8192      | 20   | Y        |1        | 1       | 1       | 0.8620 | Variational FM model with default priors|
|VFM        | 8192      | 20   | Y        |1        | 1       | 1       | 0.8580 | Variational FM model with grouping|

Yamada [1] reports the following errors on a 25% test set of the same
ML-1M dataset root mean squared errors (RMSE):

| Model             | RMSE  |
|-------------------| ------|
| libFM ALS         | 0.981 |
| libFM SGD         | 0.943 |
| libFM MCMC 0.05   | 0.877 |
| CFM               | 0.866 |
| CFM (BCD)         | 0.850 |
| libFM MCMC 0.10   | 0.846 |

[1] https://arxiv.org/pdf/1507.01073.pdf
