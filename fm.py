from chainer import Chain
from chainer import links as L
from chainer import functions as F
from chainer import reporter
import numpy as np


class FM(Chain):
    _mask = None

    def __init__(self, n_features=None, n_dim=8, lossfun=F.mean_squared_error,
                 lambda0=5e-3, lambda1=5e-3, lambda2=5e-3, init_bias=0.0,
                 intx_term=True, total_nobs=1):
        self.n_dim = n_dim
        self.n_features = n_features
        self.lossfun = lossfun
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.intx_term = intx_term
        self.total_nobs = total_nobs

        # These are all the learned weights corresponding
        # to the overall bias, slope per feature, and latent
        # interaction vector per feature
        super(FM, self).__init__(bias=L.Bias(shape=(1,)),
                                 slope=L.EmbedID(n_features, 1),
                                 latent=L.EmbedID(n_features, n_dim))

        # Xavier initialize weights
        c = np.sqrt(n_features * n_dim)
        self.latent.W.data[...] = np.random.randn(n_features, n_dim) / c
        d = np.sqrt(n_features)
        self.slope.W.data[...] = np.random.randn(n_features, 1) / d
        self.bias.b.data[...] *= 0.0
        self.bias.b.data[...] += init_bias

    def mask(self, bs, nf):
        if self._mask is None or self._mask.shape[0] != bs:
            mask = self.xp.ones((nf, nf), dtype='float32')
            mask -= self.xp.eye(nf, dtype='float32')
            masks = self.xp.tile(mask, (bs, 1, 1))
            self._mask = masks
        return self._mask

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

        bs = val.data.shape[0]
        nf = val.data.shape[1]
        mask = self.mask(bs, nf)
        # Input shape is (batchsize, n_feat_max) and
        # v is (batchsize, n_feat_max, n_dim)
        vi = self.latent(loc)
        # Form square latent interaction matrix of shape
        # (batchsize, n_feat_max, n_feat_max)
        vij = F.batch_matmul(vi, vi, transb=True)
        # Form square observed feature matrix of shape
        # (batchsize, n_feat_max, n_feat_max)
        xij = F.batch_matmul(val, val, transb=True)
        # Slope coupled to each active feature
        # loc & self.slope(loc) are shape (batchsize, n_feat_max)
        # val is also (batchsize, n_feat_max)
        coef = F.reshape(self.slope(loc), val.data.shape)
        slop = F.sum(coef * val, axis=1)
        # This double sums all of the interaction terms aside
        # from the computational burden this shouldn't be a problem.
        # TODO: implement the trick in Rendle's paper
        # that makes this O(kN) instead of O(kN^2)
        intx = F.sum(vij * xij * mask, axis=(1, 2)) * 0.5
        # Broadcast to shape of batchsize
        bias = F.broadcast_to(self.bias.b, slop.data.shape)
        # Optionally choose to include the interaction term
        # without this is linear regression
        if self.intx_term:
            pred = bias + slop + intx
        else:
            pred = bias + slop
        # Compute MSE loss
        mse = F.mean_squared_error(pred, y)
        rmse = F.sqrt(mse)
        # Calculate regularization losses
        frac = loc.data.shape[0] * 1.0 / self.total_nobs
        reg0 = F.sum(self.bias.b)
        reg1 = F.sum(self.slope.W * self.slope.W)
        reg2 = F.sum(self.latent.W * self.latent.W)
        # Total loss is MSE plus regularization losses
        regt = reg0 * self.lambda0 + reg1 * self.lambda1 + reg2 * self.lambda2
        loss = mse + regt * frac
        # Log the errors
        logs = {'loss': loss, 'mse': mse, 'rmse': rmse, 'reg0': reg0, 
                'regt': regt, 'reg1': reg1, 'reg2': reg2,
                'bias': F.sum(self.bias.b)}
        reporter.report(logs, self)
        return loss

    def __call__(self, val, loc, y, dummy=None):
        return self.forward(val, loc, y)
