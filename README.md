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

[1] https://arxiv.org/pdf/1507.01073.pdf

# Dicussion

Within the Variational FM framework we can get more than a good point estimate,
we can can get an estimate of the mean and variance of a single feature. This
means we can estimate the variance conditioned on a few active features (e.g.
conditioned on a single user) and retrieve the most uncertain trial for that
user.


For typical linear regression with interactions we have:

![eq1](https://latex.codecogs.com/gif.latex?%5Cdpi%7B300%7D%20y%20%5Csim%20c%20&plus;%20%5CSigma_i%20%5Cbeta_i%20x_i%20&plus;%20%5CSigma_%7Bij%7D%20w_%7Bij%7D%20x_i%20x_j)

[//]: # ( y \sim c + \Sigma_i \beta_i x_i + \Sigma_{ij} w_{ij} x_i x_j)


Note that `x_i` is usually a sparse feature vector (but doesn't have to be). In the land of recommenders, we're usually interested in the coefficient `w_ij` in front of an interaction such as `x_i x_j` where `x_i` might be a dummy-encoded user id and `x_j` is an item_id. The big problem here is that `w_ij` is quadratic in the number of features (e.g. # of users + # of items), so there are lots of parameters to estimate with sparse observations.  We've also left off any regularization, but might choose to L2 penalize `w_ij` or `beta_ij`.

FMs fix this by doing a low-rank approximation to `w_ij` by saying that `w_ij=v_i * v_j` where each feature `i` has a latent rank-k vector `v_i`. Instead of computing an N x N `w_ij` matrix, we compute N x k parameters in the form of N `v_i` vectors, yielding a new objective function:

![eq2](https://latex.codecogs.com/gif.latex?%5Cdpi%7B300%7D%20y%20%5Csim%20c%20&plus;%20%5CSigma_i%20%5Cbeta_i%20x_i%20&plus;%20%5CSigma_%7Bij%7D%20%5Cvec%7Bv_i%7D%20%5Ccdot%20%5Cvec%7Bv_j%7D%20x_i%20x_j)

[//]: # ( y \sim c + \Sigma_i \beta_i x_i + \Sigma_{ij} \vec{v_i} \cdot \vec{v_j} x_i x_j)

In variational FMs we impose a bit more hierarchy and swap out L2 regularization for Gauassian priors:

![eq3](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B300%7D%20%5Cbeta_i%20%5Csim%20%5Cmathcal%7BN%7D%28%20%5Cmu_%5Cbeta%2C%20%5Csigma_%5Cbeta%29)


![eq3b](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B300%7D%20%5Cvec%7Bv_i%7D%20%5Csim%20%5Cmathcal%7BN%7D%28%20%5Cvec%7B%5Cmu_v%7D%2C%20%5Cvec%7B%5Csigma%7D_v%29%29)

[//]: # (\beta_i \sim \mathcal{N}( \mu_\beta, \sigma_\beta))
[//]: # (\vec{v_i} \sim \mathcal{N}( \vec{\mu_v}, \vec{\sigma}_v)))

And then group these (hyper)priors together assuming a normal prior with unity variance.  The vectors `v_i` are drawn from a multivariate prior with a diagonal covariance matrix. The log-normal prior on the variance isn't the disciplined choice but it is convenient and amenable to Stochastic Variational Bayes inference.

![eq3](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B300%7D%20%5Cmu_%5Cbeta%20%5Csim%20%5Cmathcal%7BN%7D%280%2C%201%29%5C%5C%20log%5Csigma_%5Cbeta%20%5Csim%20%5Cmathcal%7BN%7D%280%2C%201%29)

[//]: # (\mu_\beta \sim \mathcal{N}(0, 1))
[//]: # (log\sigma_\beta \sim \mathcal{N}(0, 1))

As you can see in the results table, shrinking to the groups greatly improves test set validation scores.

This forms a 'deep' model: the hyperpriors `mu_b` and `sigma_b` pick the group mean and group variance from which individual `beta_i` and `v_i` are drawn. In variational inference, those `beta_i` and  `v_i` in turn have their own means and variances, so that we're not just point estimating `beta_i` but in fact estimate `mu_beta_i` and `sigma_beta_i`. If you're curious how this mode of inference works, read [this](http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/) or [this for the trick in 140 characters](https://twitter.com/ryan_p_adams/status/663049108689715200) -- it's at the heart of Bayesian deep learning techniques.


With estimates of `mu_v_i = E[v_i]` and `sigma_v_i = Var[v_i]` we finally get the critical ingredient to do active learning on FMs -- an uncertainty estimate around the feature vector `v_i`. But we need the uncertainty for the whole model, which is composed of interactions on `v_i`:

![eq4](https://latex.codecogs.com/gif.latex?%5Cdpi%7B300%7D%20var%5Bv_i%20v_j%5D%20%3D%20%5Csigma_%7Bv_i%7D%20%5Csigma_%7Bv_j%7D%20&plus;%20%5Csigma_%7Bv_i%7D%20%5Cmu_%7Bv_j%7D%20&plus;%20%5Csigma_%7Bv_j%7D%20%5Cmu_%7Bv_i%7D)

[//]: # (var[v_i v_j] = \sigma_{v_i} \sigma_{v_j} + \sigma_{v_i} \mu_{v_j} + \sigma_{v_j} \mu_{v_i})

Note that the above is just the identity for the product of two independent random variables. Technically `v_i` is a vector, but the components are independent so replace that above `v_i` with an arbitrary component of that vector:

![eq4b](https://latex.codecogs.com/gif.latex?%5Cdpi%7B300%7D%20var%5Bv_i%20v_j%5D%20%3D%20%5Cvec%7B%5Csigma_%7Bv_i%7D%7D%20%5Ccdot%20%5Cvec%7B%5Csigma_%7Bv_j%7D%7D%20&plus;%20%5Cvec%7B%5Csigma_%7Bv_i%7D%7D%20%5Ccdot%20%5Cvec%7B%5Cmu_%7Bv_j%7D%7D%20&plus;%20%5Cvec%7B%5Csigma_%7Bv_j%7D%7D%20%5Ccdot%20%5Cvec%7B%20%5Cmu_%7Bv_i%7D%7D)

[//]: # (var[v_i v_j] = \vec{\sigma_{v_i}} \cdot \vec{\sigma_{v_j}} + \vec{\sigma_{v_i}} \cdot \vec{\mu_{v_j}} + \vec{\sigma_{v_j}} \cdot \vec{ \mu_{v_i}})

The variances of the `beta` components do not covary with the `v_i` components, so the full model variance is decomposes into the sum of the individual variances:

![eq5](https://latex.codecogs.com/gif.latex?%5Cdpi%7B300%7D%20var%5B%5CSigma_i%5Cbeta_i%20x_i%20&plus;%20%5CSigma_%7Bij%7D%20v_i%20v_j%20x_i%20x_j%5D%20%3D%5C%5C%20%5CSigma_i%20var%5B%5Cbeta_i%5D%20x_i%20x_j%20&plus;%20%5CSigma_%7Bij%7D%20var%5Bv_i%20v_j%5D%20x_i%20x_j%20%3D%20%5C%5C%20%5CSigma_i%20%5Csigma_%7B%5Cbeta_i%7D%20x_i%20x_j%20&plus;%20%5CSigma_%7Bij%7D%20%5B%5Cvec%7B%5Csigma_%7Bv_i%7D%7D%20%5Ccdot%20%5Cvec%7B%5Csigma_%7Bv_j%7D%7D%20&plus;%20%5Cvec%7B%5Csigma_%7Bv_i%7D%7D%20%5Ccdot%20%5Cvec%7B%5Cmu_%7Bv_j%7D%7D%20&plus;%20%5Cvec%7B%5Csigma_%7Bv_j%7D%7D%20%5Ccdot%20%5Cvec%7B%20%5Cmu_%7Bv_i%7D%7D%5D%20x_i%20x_j)

[//]: # ( var[\Sigma_i\beta_i x_i + \Sigma_{ij} v_i v_j x_i x_j] =\\ \Sigma_i var[\beta_i] x_i x_j + \Sigma_{ij} var[v_i v_j] x_i x_j = \\ \Sigma_i \sigma_{\beta_i} x_i x_j + \Sigma_{ij} [\vec{\sigma_{v_i}} \cdot \vec{\sigma_{v_j}} + \vec{\sigma_{v_i}} \cdot \vec{\mu_{v_j}} + \vec{\sigma_{v_j}} \cdot \vec{ \mu_{v_i}}] x_i x_j )

We've used the fact that `beta` and `v_i` are independent to sum the variances independently.

So in picking the next question we can rank by the above measure to get the highest variance question. The observation features `x_i x_j` are known for each trial (they're just usually the user ID and item ID) and the means `mu` and variances `sigma` are easily accessible model parameters.
