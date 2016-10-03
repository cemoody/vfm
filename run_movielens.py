import os.path
import numpy as np
import requests
from zipfile import ZipFile
from sklearn.model_selection import train_test_split

import chainer
from chainer import training
from chainer.training import extensions
from chainer.datasets import TupleDataset

from fm import FM
from vfm import VFM

# Hyperparameters
n_dim = 20
batchsize = 1024 * 8
model_type = 'FM'


# Download, unzip and read in the dataset
name = 'ml-1m.zip'
base = 'ml-1m'
if not os.path.exists(name):
    url = 'http://files.grouplens.org/datasets/movielens/' + name
    r = requests.get(url)
    with open(name, 'wb') as fh:
        fh.write(r.content)
    zip = ZipFile(name)
    zip.extractall()

# First col is user, 2nd is movie id, 3rd is rating
data = np.genfromtxt(base + '/ratings.dat', delimiter='::')
user = data[:, 0].astype('int32')
movie = data[:, 1].astype('int32')
rating = data[:, 2].astype('float32')
n_features = user.max() + 1 + movie.max() + 1

# Formatting dataset
loc = np.zeros((len(data), 2), dtype='int32')
loc[:, 0] = user
loc[:, 1] = movie + user.max()
val = np.ones((len(data), 2), dtype='float32')

# Train test split
tloc, vloc, tval, vval, ty, vy = train_test_split(loc, val, rating,
                                                  random_state=42)
train = TupleDataset(tval, tloc, ty)
valid = TupleDataset(vval, vloc, vy)

# Setup model
if model_type == 'FM':
    model = FM(n_features, n_dim, lambda0=0.0, lambda1=0.0, lambda2=0.005,
               init_bias=ty.mean())
elif model_type == 'VFM':
    mu = ty.mean()
    lv = 0.5 * np.log(ty.std())
    model = VFM(n_features, n_dim, init_bias_mu=mu, init_bias_lv=lv)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# Setup iterators
train_iter = chainer.iterators.SerialIterator(train, batchsize)
valid_iter = chainer.iterators.SerialIterator(valid, batchsize,
                                              repeat=False, shuffle=False)
updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (70, 'epoch'), out='out')

# Setup logging, printing & saving
keys = ['loss', 'rmse', 'bias', 'regt']
reports = ['iteration', 'epoch']
reports += ['main/' + key for key in keys]
reports += ['validation/main/' + key for key in keys]
trainer.extend(extensions.Evaluator(valid_iter, model))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))
trainer.extend(extensions.PrintReport(reports))
trainer.extend(extensions.ProgressBar(update_interval=10))

# Run the model
trainer.run()
