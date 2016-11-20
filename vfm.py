from chainer import Chain
from chainer import links as L
from chainer import functions as F
from chainer import reporter
from chainer import cuda
from chainer import Variable
import numpy as np


def dot(a, b):
    """ Simple dot product"""
    return F.sum(a * b, axis=-1)


def interaction(loc):
    """ Compute all pairs of points in loc array.
    """
    return np.array(np.meshgrid(loc, loc)).T.reshape(-1, 2)


class VFM(Chain):
    lv_floor = -100.0

    def __init__(self, n_features=None, n_dim=8, lossfun=F.mean_squared_error,
                 lambda0=5e-3, lambda1=5e-3, lambda2=5e-3, init_bias_mu=0.0,
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
                                  feat_mu=L.Bias(shape=(1, n_dim)),
                                  feat_lv=L.Bias(shape=(1, n_dim)),
                                  feat_delta_mu=L.EmbedID(n_features, n_dim,
                                                          ignore_label=-1),
                                  feat_delta_lv=L.EmbedID(n_features, n_dim,
                                                          ignore_label=-1))

        # Xavier initialize weights
        c = np.sqrt(n_features * n_dim)
        d = np.sqrt(n_features)
        self.feat_delta_mu.W.data[...] = np.random.randn(n_features, n_dim) / c
        self.feat_delta_lv.W.data[...] = np.random.randn(n_features, n_dim) / c
        self.slop_delta_mu.W.data[...] = np.random.randn(n_features, 1) / d
        self.slop_delta_lv.W.data[...] = np.random.randn(n_features, 1) / d
        self.bias_mu.b.data[...] *= 0.0
        self.bias_mu.b.data[...] += init_bias_mu
        self.bias_lv.b.data[...] *= 0.0
        self.bias_lv.b.data[...] += init_bias_lv

    def term_bias(self, bs, is_test=None):
        """ Compute overall bias and broadcast to shape of batchsize
        """

        shape = (bs, 1,)
        # Bias is drawn from a Gaussian with given mu and log variance
        bs_mu = F.broadcast_to(self.bias_mu.b, shape)
        bs_lv = F.broadcast_to(self.bias_lv.b, shape)
        # Add a very negative log variance so we're sampling
        # from a very narrow distribution about the mean.
        # Useful for validation dataset when we want to only guess
        # the mean.
        if is_test is not None:
            bs_lv += F.reshape(is_test, shape) * self.lv_floor
        bias = F.gaussian(bs_mu, bs_lv)

        # Compute prior on the bias, so compute the KL div
        # from the KL(N(mu_bias, var_bias) | N(0, 1))
        kld = F.gaussian_kl_divergence(self.bias_mu.b, self.bias_lv.b)
        return bias, bs_lv, kld

    def term_slop(self, loc, val, is_test, bs, nf):
        """ Compute the slope for each active feature.
        """
        shape = (bs, nf)

        # Reshape all of our constants
        pr_mu = F.broadcast_to(self.slop_mu.b, shape)
        pr_lv = F.broadcast_to(self.slop_lv.b, shape)
        # This is either zero or a very negative number
        # indicating to sample N(mean, logvar) or just draw
        # the mean preicsely
        is_test = F.broadcast_to(F.reshape(is_test, (bs, 1)), shape)
        is_test *= self.lv_floor

        # The feature slopes are grouped together so that they
        # all share a common mean. Then individual features slop_delta_lv
        # are shrunk towards zero, which effectively sets features to fall
        # back on the group mean.
        sl_mu = F.reshape(self.slop_delta_mu(loc), shape) + pr_mu
        sl_lv = F.reshape(self.slop_delta_lv(loc), shape) + pr_lv + is_test
        coef = F.gaussian(sl_mu, sl_lv)
        slop = F.sum(coef * val, axis=1)

        # Calculate divergence between group mean and N(0, 1)
        kld1 = F.gaussian_kl_divergence(self.slop_mu.b, self.slop_lv.b)
        # Calculate divergence of individual delta means and delta vars
        args = (self.slop_delta_mu.W, self.slop_delta_lv.W)
        kld2 = F.gaussian_kl_divergence(*args)
        return slop, sl_lv, kld1 + kld2

    def term_feat(self, iloc, jloc, ival, jval, is_test, bs, nf):
        # Change all of the shapes to form interaction vectors
        shape = (bs, nf)
        feat_mu = F.broadcast_to(self.feat_mu.b, shape)
        feat_lv = F.broadcast_to(self.feat_lv.b, shape)
        is_test = F.broadcast_to(F.reshape(is_test, (bs, 1)), shape)
        is_test *= self.lv_floor

        # Construct the interaction mean and variance
        # iloc is (bs, nf), feat(iloc) is (bs, nf, ndim) and
        # dot(feat, feat) is (bs, nf)
        feat_mu += dot(self.feat_delta_mu(iloc), self.feat_delta_mu(jloc))
        feat_lv += dot(self.feat_delta_lv(iloc), self.feat_delta_lv(jloc))
        feat_lv += is_test
        # feat_vec is (bs, nf) with each element indicating <v_i, v_j>
        feat_vec = F.gaussian(feat_mu, feat_lv)
        # feat is (bs, )
        feat = dot(feat_vec, ival * jval)

        # Compute the KLD for the group mean vector and variance vector
        kld1 = F.gaussian_kl_divergence(self.feat_mu.b, self.feat_lv.b)
        # Compute the KLD for vector deviations from the group mean and var
        kld2 = F.gaussian_kl_divergence(self.feat_delta_mu.W,
                                        self.feat_delta_lv.W)
        return feat, feat_lv, kld1 + kld2

    def forward(self, loc, val, iloc, jloc, ival, jval, y, is_test):
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

        is_test : array of float
        Dummy array that is 0 if the batch is the training dataset and 1 if
        the example is in the test set. In the latter case, this sets the
        variance to nearly zero such that parameters are always set to the
        mean with no variance.

        """
        bs = val.data.shape[0]
        nf = val.data.shape[1]

        bias, bias_lv, kld0 = self.term_bias(bs, is_test)
        slop, slop_lv, kld1 = self.term_slop(loc, val, is_test, bs, nf)
        feat, feat_lv, kld2 = self.term_feat(iloc, jloc, ival, jval,
                                             is_test, bs, nf)

        # Optionally choose to include the interaction term
        # without this is linear regression
        pred = bias + slop
        if self.intx_term:
            pred += feat

        # Compute MSE loss
        mse = F.mean_squared_error(pred, y)
        rmse = F.sqrt(mse)  # Only used for reporting

        # Now compute the total KLD loss
        kldt = kld0 * self.lambda0 + kld1 * self.lambda1 + kld2 * self.lambda2

        # Total loss is MSE plus regularization losses
        frac = bs * 1.0 / self.total_nobs
        loss = mse + kldt * frac

        # Log the errors
        logs = {'loss': loss, 'rmse': rmse, 'kld0': kld0, 'kld1': kld1,
                'kld2': kld2, 'kldt': kldt, 'bias': F.sum(self.bias_mu.b)}
        reporter.report(logs, self)
        return loss

    def __call__(self, val, loc, y, is_test):
        bs = y.data.shape[0]
        locs = interaction(cuda.to_cpu(loc.data)).astype('int32')
        iloc = Variable(self.xp.asarray(locs[:, 0]))
        jloc = Variable(self.xp.asarray(locs[:, 1]))
        ival = Variable(self.xp.ones((bs), dtype='float32'))
        jval = Variable(self.xp.ones((bs), dtype='float32'))
        return self.forward(loc, val, iloc, jloc, ival, jval, y, is_test)
