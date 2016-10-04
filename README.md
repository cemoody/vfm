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
|VFM        | 8192      | 20   | Y        |1        | 1       | 1       | 0.8585 | Variational FM model with grouping|
|VFM        | 8192      | 64   | Y        |1        | 1       | 1       | 0.8800 | Higher rank model does worse|

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


**Active Learning**
Non-variational FM is solving the objective function:
![objfun][objfun](https://latex.codecogs.com/gif.download?%5Cdpi%7B300%7D%20%5Csmall%20y%20%5Csim%20c%20+%20%5CSigma_i%20%5Cbeta_i%20x_i%20+%20%5CSigma_%7Bij%7D%20w_%7Bij%7D%20x_i%20x_j%5C%5C%20y%20%5Csim%20c%20+%20%5CSigma_i%20%5Cbeta_i%20x_i%20+%20%5CSigma_%7Bij%7D%20%5Cvec%7Bv_i%7D%20%5Ccdot%20%5Cvec%7Bv_i%7D%20x_i%20x_j%5C%5C)

[//]: # (y \sim c + \Sigma_i \beta_i x_i + \Sigma_{ij} w_{ij} x_i x_j\\
y \sim c + \Sigma_i \beta_i x_i + \Sigma_{ij} <v_i, v_j> x_i x_j\\
<v_i, v_j> = v_i \cdot v_j)

Within the Variational FM framework we can get more than a good point estimate,
we can can get an estimate of the mean and variance of a single feature. This
means we can estimate the variance conditioned on a few active features (e.g.
conditioned on a single user). Given two random variables X and Y with known
means (mu_x, mu_y) and variances (sigma_x, sigma_y) we can compute 

[1] https://arxiv.org/pdf/1507.01073.pdf
[objfun]: https://latex.codecogs.com/gif.latex?%5Cdpi%7B300%7D%20%5Csmall%20y%20%5Csim%20c%20&plus;%20%5CSigma_i%20%5Cbeta_i%20x_i%20&plus;%20%5CSigma_%7Bij%7D%20w_%7Bij%7D%20x_i%20x_j%5C%5C%20y%20%5Csim%20c%20&plus;%20%5CSigma_i%20%5Cbeta_i%20x_i%20&plus;%20%5CSigma_%7Bij%7D%20%5Cvec%7Bv_i%7D%20%5Ccdot%20%5Cvec%7Bv_i%7D%20x_i%20x_j%5C%5C
