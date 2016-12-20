from chainer import Chain
from chainer import links as L
from chainer import functions as F
from chainer import reporter
from chainer import cuda
import numpy as np


def dot(a, b):
    """ Simple dot product"""
    return F.sum(a * b, axis=-1)


def batch_interactions(x):
    xp = cuda.get_array_module(x.data)
    batchsize = x.shape[0]
    shape = (batchsize, x.shape[1] ** 2)
    left = xp.tile(x.data, (1, x.shape[1]))
    right = xp.repeat(x.data, x.shape[1]).reshape(shape)
    return left, right


class VFM(Chain):
    lv_floor = -100.0

    def __init__(self, n_features=None, n_dim=8, lossfun=F.mean_squared_error,
                 lambda0=1, lambda1=1, lambda2=1, init_bias_mu=0.0,
                 init_bias_lv=0.0, intx_term=True, total_nobs=1):
        self.n_dim = n_dim
        self.n_features = n_features
        self.lossfun = lossfun
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.intx_term = intx_term
        self.total_nobs = total_nobs

        # In contrast to the FM model, the slopes and latent vectors
        # will have means (mu) and log variances (lv) for each component.
        super(VFM, self).__init__(bias_mu=L.Bias(shape=(1,)),
                                  bias_lv=L.Bias(shape=(1,)),
                                  slop_mu=L.Bias(shape=(1, 1)),
                                  slop_lv=L.Bias(shape=(1, 1)),
                                  slop_delta_mu=L.EmbedID(n_features, 1,
                                                          ignore_label=-1),
                                  slop_delta_lv=L.EmbedID(n_features, 1,
                                                          ignore_label=-1),
                                  feat_mu_vec=L.Bias(shape=(1, 1, n_dim)),
                                  feat_lv_vec=L.Bias(shape=(1, 1, n_dim)),
                                  feat_delta_mu=L.EmbedID(n_features, n_dim,
                                                          ignore_label=-1),
                                  feat_delta_lv=L.EmbedID(n_features, n_dim,
                                                          ignore_label=-1))

        # Xavier initialize weights
        c = np.sqrt(n_features * n_dim) * 1e3
        d = np.sqrt(n_features) * 1e3
        self.feat_delta_mu.W.data[...] = np.random.randn(n_features, n_dim) / c
        self.feat_delta_lv.W.data[...] = np.random.randn(n_features, n_dim) / c
        self.slop_delta_mu.W.data[...] = np.random.randn(n_features, 1) / d
        self.slop_delta_lv.W.data[...] = np.random.randn(n_features, 1) / d
        self.bias_mu.b.data[...] *= 0.0
        self.bias_mu.b.data[...] += init_bias_mu
        self.bias_lv.b.data[...] *= 0.0
        self.bias_lv.b.data[...] += init_bias_lv

    def term_bias(self, bs, train=True):
        """ Compute overall bias and broadcast to shape of batchsize
        """

        shape = (bs, 1,)
        # Bias is drawn from a Gaussian with given mu and log variance
        bs_mu = F.broadcast_to(self.bias_mu.b, shape)
        bs_lv = F.broadcast_to(self.bias_lv.b, shape)
        bias = F.flatten(F.gaussian(bs_mu, bs_lv))

        # Add a very negative log variance so we're sampling
        # from a very narrow distribution about the mean.
        # Useful for validation dataset when we want to only guess
        # the mean.
        if not train:
            bs_lv += self.lv_floor

        # Compute prior on the bias, so compute the KL div
        # from the KL(N(mu_bias, var_bias) | N(0, 1))
        kld = F.gaussian_kl_divergence(self.bias_mu.b, self.bias_lv.b)
        return bias, kld

    def term_slop(self, loc, val, bs, nf, train=True):
        """ Compute the slope for each active feature.
        """
        shape = (bs, nf)

        # Reshape all of our constants
        pr_mu = F.broadcast_to(self.slop_mu.b, shape)
        pr_lv = F.broadcast_to(self.slop_lv.b, shape)
        # This is either zero or a very negative number
        # indicating to sample N(mean, logvar) or just draw
        # the mean preicsely
        if not train:
            pr_lv += self.lv_floor

        # The feature slopes are grouped together so that they
        # all share a common mean. Then individual features slop_delta_lv
        # are shrunk towards zero, which effectively sets features to fall
        # back on the group mean.
        sl_mu = F.reshape(self.slop_delta_mu(loc), shape) + pr_mu
        sl_lv = F.reshape(self.slop_delta_lv(loc), shape) + pr_lv
        coef = F.gaussian(sl_mu, sl_lv)
        slop = F.sum(coef * val, axis=1)

        # Calculate divergence between group mean and N(0, 1)
        kld1 = F.gaussian_kl_divergence(self.slop_mu.b, self.slop_lv.b)
        # Calculate divergence of individual delta means and delta vars
        args = (self.slop_delta_mu.W, self.slop_delta_lv.W)
        kld2 = F.gaussian_kl_divergence(*args)

        return slop, kld1 + kld2

    def term_feat(self, iloc, jloc, ival, jval, bs, nf, train=True):
        # Change all of the shapes to form interaction vectors
        shape = (bs, nf * 2, self.n_dim)
        feat_mu_vec = F.broadcast_to(self.feat_mu_vec.b, shape)
        feat_lv_vec = F.broadcast_to(self.feat_lv_vec.b, shape)
        if not train:
            feat_lv_vec += self.lv_floor

        # Construct the interaction mean and variance
        # iloc is (bs, nf), feat(iloc) is (bs, nf, ndim) and
        # dot(feat, feat) is (bs, nf)
        ivec = F.gaussian(feat_mu_vec + self.feat_delta_mu(iloc),
                          feat_lv_vec + self.feat_delta_lv(iloc))
        jvec = F.gaussian(feat_mu_vec + self.feat_delta_mu(jloc),
                          feat_lv_vec + self.feat_delta_lv(jloc))
        # feat is (bs, )
        feat = dot(F.sum(ivec * jvec, axis=2), ival * jval)

        # Compute the KLD for the group mean vector and variance vector
        kld1 = F.gaussian_kl_divergence(self.feat_mu_vec.b, self.feat_lv_vec.b)
        # Compute the KLD for vector deviations from the group mean and var
        kld2 = F.gaussian_kl_divergence(self.feat_delta_mu.W,
                                        self.feat_delta_lv.W)
        return feat, kld1 + kld2

    def forward(self, loc, val, y, train=True):
        """ Given the sparse feature vector defined by location
        integers for the column index and the value at that index.
        y ~ c + sum(w_i x_i) + sum_ij( <v_i, v_j> * x_i * x_j)

        Parameters
        ----------
        val : array of float
        Values in the feature array. Should of shape (batchsize, n_feat_max)

        loc : array of int
        Location of the non-zero columns in the sparse vector. Should be of
        shape (batchsize, n_feat_max)

        y : array of float
        Array of expected outcome.

        train: bool
        If True uses the reparameterization trick to estimate variables.
        If False, this sets the variance to nearly zero such that
        parameters are always set to the mean with no noise, which is useful
        at test time.

        """
        bs = val.data.shape[0]
        nf = val.data.shape[1]

        iloc, jloc = batch_interactions(loc)
        ival, jval = batch_interactions(val)

        # Compute scalar bias term
        bias, kld0 = self.term_bias(bs, train=train)
        # Compute the feature weights
        slop, kld1 = self.term_slop(loc, val, bs, nf, train=train)
        # Compute factorized weights on interaction features
        feat, kld2 = self.term_feat(iloc, jloc, ival, jval,
                                    bs, nf, train=train)

        # Optionally choose to include the interaction term
        # without this is linear regression
        pred = bias + slop
        if self.intx_term:
            pred += feat

        return pred, kld0, kld1, kld2

    def __call__(self, loc, val, y, train=True):
        bs = val.data.shape[0]
        pred, kld0, kld1, kld2 = self.forward(loc, val, y, train=train)

        # Compute MSE loss
        mse = F.mean_squared_error(pred, y)
        rmse = F.sqrt(mse)  # Only used for reporting

        # Now compute the total KLD loss
        kldt = kld0 * self.lambda0 + kld1 * self.lambda1 + kld2 * self.lambda2

        # Total loss is MSE plus regularization losses
        loss = mse + kldt * (1.0 / self.total_nobs)

        # Log the errors
        logs = {'loss': loss, 'rmse': rmse, 'kld0': kld0, 'kld1': kld1,
                'kld2': kld2, 'kldt': kldt, 'bias': F.sum(self.bias_mu.b)}
        reporter.report(logs, self)
        return loss
