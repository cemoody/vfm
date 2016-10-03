Chainer Factorization Machine ranking.

Run `python run_movielens.py` to download, extract, and run the FM model
on MovieLens 1M data. 


| batchsize | rank | lambda0 | lambda1 | lambda2 | RMSE    |
|-----------|------|---------|---------|---------|---------|
| 8192      | 20   | 0       | 0       | 1e-2    | 0.90908 |
| 8192      | 20   | 0       | 0       | 5e-3    | 0.90905 | 

Blondei [1] reports on a 25% test set of the same
ML-1M dataset root mean squared errors (RMSE):

| Model             | RMSE  |
|-------------------| ------|
| CFM               | 0.866 |
| CFM (BCD)         | 0.850 |
| libFM SGD         | 0.943 |
| libFM ALS         | 0.981 |
| libFM MCMC 0.05   | 0.877 |
| libFM MCMC 0.10   | 0.846 |

0.866 0.85 0.943 0.981 0.877 0.846 0.899
CFM CFM (BCD) FMSGD FMALS FM(0.05) MCMC FM(0.1) MCMC Ridge

