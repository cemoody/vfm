from chainer import Chain
from chainer import links as L
from chainer import functions as F
from chainer import reporter
import numpy as np


class FM(Chain):
    def __init__(self, n_features=None, n_dim=8, lossfun=F.mean_squared_error,
                 lambda0=5e-3, lambda1=5e-3, lambda2=5e-3, init_bias=0.0):
        self.n_dim = n_dim
        self.n_features = n_features
        self.lossfun = lossfun
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2

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
        intx = F.sum(vij * xij, axis=(1, 2))
        # Broadcast to shape of batchsize
        bias = F.broadcast_to(self.bias.b, slop.data.shape)
        # Compute MSE loss
        mse = F.mean_squared_error(bias + slop + intx, y)
        rmse = F.sqrt(mse)
        # Calculate regularization losses
        reg0 = F.sum(self.bias.b) * self.lambda0
        reg1 = F.sum(self.slope.W * self.slope.W) * self.lambda1
        reg2 = F.sum(self.latent.W * self.latent.W) * self.lambda2
        # Total loss is MSE plus regularization losses
        loss = mse + reg0 + reg1 + reg2
        # Log the errors
        logs = {'loss': loss, 'mse': mse, 'rmse': rmse, 'reg0': reg0,
                'reg1': reg1, 'reg2': reg2, 'bias': F.sum(self.bias.b)}
        reporter.report(logs, self)
        return mse + reg0 + reg1 + reg2

    def __call__(self, val, loc, y):
        return self.forward(val, loc, y)
