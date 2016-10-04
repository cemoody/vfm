import os.path
import argparse
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

# Hyperparameters set through CLI
parser = argparse.ArgumentParser()
parser.add_argument('--n_dim', dest='n_dim', default=20, type=int)
parser.add_argument('--batchsize', dest='batchsize', default=8192, type=int)
parser.add_argument('--model_type', dest='model_type', default='FM', type=str)
parser.add_argument('--device', dest='device', default=-1, type=int)
parser.add_argument('--lambda0', dest='lambda0', default=0.0, type=float)
parser.add_argument('--lambda1', dest='lambda1', default=1e-3, type=float)
parser.add_argument('--lambda2', dest='lambda2', default=1e-3, type=float)
parser.add_argument('--intx_term', dest='intx_term', default=1, type=int)

# Expand arguments into local variables
args = vars(parser.parse_args())
print args
n_dim = args.pop('n_dim')
batchsize = args.pop('batchsize')
model_type = args.pop('model_type')
device = args.pop('device')
lambda0 = args.pop('lambda0')
lambda1 = args.pop('lambda1')
lambda2 = args.pop('lambda2')
intx_term = args.pop('intx_term')

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
total_nobs = len(tloc)
ti = np.zeros(ty.shape, dtype='float32')
vi = np.ones(vy.shape, dtype='float32')
train = TupleDataset(tval, tloc, ty, ti)
valid = TupleDataset(vval, vloc, vy, vi)

# Setup model
print "Running model:" + model_type
if model_type == 'FM':
    model = FM(n_features, n_dim, lambda0=lambda0, lambda1=lambda1, lambda2=lambda2,
               init_bias=ty.mean(), intx_term=intx_term, total_nobs=total_nobs)
elif model_type == 'VFM':
    mu = ty.mean()
    lv = 0.5 * np.log(ty.std())
    model = VFM(n_features, n_dim, init_bias_mu=mu, init_bias_lv=lv,
                total_nobs=total_nobs)
if device >= 0:
    chainer.cuda.get_device(device).use()
    model.to_gpu(device) 
optimizer = chainer.optimizers.RMSprop()
optimizer.setup(model)

# Setup iterators
train_iter = chainer.iterators.SerialIterator(train, batchsize)
valid_iter = chainer.iterators.SerialIterator(valid, batchsize,
                                              repeat=False, shuffle=False)
updater = training.StandardUpdater(train_iter, optimizer, device=device)
trainer = training.Trainer(updater, (75, 'epoch'), out='out_' + str(device))

# Setup logging, printing & saving
keys = ['loss', 'rmse', 'bias', 'regt', 'reg1', 'reg2']
reports = ['iteration', 'epoch']
reports += ['main/' + key for key in keys]
reports += ['validation/main/rmse']
trainer.extend(extensions.Evaluator(valid_iter, model, device=device))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
trainer.extend(extensions.PrintReport(reports))
trainer.extend(extensions.ProgressBar(update_interval=10))

# If previous model detected, resume
# chainer.serializers.load_npz(args.resume, trainer)

# Run the model
trainer.run()
