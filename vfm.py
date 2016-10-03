from chainer import Chain
from chainer import links as L
from chainer import functions as F
from chainer import reporter
import numpy as np


class VFM(Chain):
    def __init__(self, n_features=None, n_dim=8, lossfun=F.mean_squared_error,
                 init_bias_mu=0.0, init_bias_lv=1.0, lambda0=1.0, lambda1=1.0,
                 lambda2=1.0, init_bias=0.0):
        self.n_dim = n_dim
        self.n_features = n_features
        self.lossfun = lossfun
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # In contrast to the FM model, the slopes and latent vectors
        # will have means (mu) and log variances (lv) for each component.
        super(VFM, self).__init__(bias_mu=L.Bias(shape=(1,)),
                                  bias_lv=L.Bias(shape=(1,)),
                                  slope_mu=L.EmbedID(n_features, 1),
                                  slope_lv=L.EmbedID(n_features, 1),
                                  latent_mu=L.EmbedID(n_features, n_dim),
                                  latent_lv=L.EmbedID(n_features, n_dim))

        # Xavier initialize weights
        c = np.sqrt(n_features * n_dim)
        d = np.sqrt(n_features)
        self.latent_mu.W.data[...] = np.random.randn(n_features, n_dim) / c
        self.latent_lv.W.data[...] = np.random.randn(n_features, n_dim) / c
        self.slope_mu.W.data[...] = np.random.randn(n_features, 1) / d
        self.slope_lv.W.data[...] = np.random.randn(n_features, 1) / d
        self.bias_mu.b.data[...] *= 0.0
        self.bias_mu.b.data[...] += init_bias_mu
        self.bias_lv.b.data[...] *= 0.0
        self.bias_lv.b.data[...] += init_bias_lv

    def forward(self, val, loc, y):
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
        """

        # Input shape is (batchsize, n_feat_max) and
        # v is (batchsize, n_feat_max, n_dim)
        vi_mu = self.latent_mu(loc)
        vi_lv = self.latent_lv(loc)
        vi = F.gaussian(vi_mu, vi_lv)
        # Form square latent interaction matrix of shape
        # (batchsize, n_feat_max, n_feat_max)
        vij = F.batch_matmul(vi, vi, transb=True)
        # Form square observed feature matrix of shape
        # (batchsize, n_feat_max, n_feat_max)
        xij = F.batch_matmul(val, val, transb=True)
        # Slope coupled to each active feature
        # loc & self.slope(loc) are shape (batchsize, n_feat_max)
        # val is also (batchsize, n_feat_max)
        sl_mu = self.slope_mu(loc)
        sl_lv = self.slope_lv(loc)
        coef = F.reshape(F.gaussian(sl_mu, sl_lv), val.data.shape)
        slop = F.sum(coef * val, axis=1)
        # This double sums all of the interaction terms aside
        # from the computational burden this shouldn't be a problem.
        # TODO: implement the trick in Rendle's paper
        # that makes this O(kN) instead of O(kN^2)
        intx = F.sum(vij * xij, axis=(1, 2))
        # Broadcast to shape of batchsize
        bs_mu = self.bias_mu.b
        bs_lv = self.bias_lv.b
        bs = F.gaussian(bs_mu, bs_lv)
        bias = F.broadcast_to(bs, slop.data.shape)
        # Compute MSE loss
        mse = F.mean_squared_error(bias + slop + intx, y)
        rmse = F.sqrt(mse)
        # Calculate regularization losses
        reg0 = F.gaussian_kl_divergence(bs_mu, bs_lv) * self.lambda0
        reg1 = F.gaussian_kl_divergence(sl_mu, sl_lv) * self.lambda1
        reg2 = F.gaussian_kl_divergence(vi_mu, vi_lv) * self.lambda2
        regt = reg0 + reg1 + reg2
        # Total loss is MSE plus regularization losses
        loss = mse + reg0 + reg1 + reg2
        # Log the errors
        logs = {'loss': loss, 'rmse': rmse, 'reg0': reg0, 'regt': regt,
                'reg1': reg1, 'reg2': reg2, 'bias': F.sum(self.bias_mu.b)}
        reporter.report(logs, self)
        return mse + reg0 + reg1 + reg2

    def __call__(self, val, loc, y):
        return self.forward(val, loc, y)
