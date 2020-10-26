#%% Reproduce Table 3 for COOT
import sys
sys.path.append('../code')
import numpy as np
from sklearn.metrics import *
from blockSim import *
from cot import cot_numpy
#%%
def cot_clustering(X, ns, nv, niter_cluster=10, niter=10, algo1='emd', algo2 = 'emd', reg1 = 0, reg2 = 0, verbose = False):
    Xc = np.random.randn(ns, nv)
    old_cost = 0

    for i in range(niter_cluster):
        Ts, Tv, cost = cot_numpy(X, Xc, niter=10, algo=algo1, reg = reg1, algo2 = algo2, reg2 = reg2, verbose=False)
        Xc = Ts.T.dot(X).dot(Tv) * ns * nv

        if verbose:
            print(cost)

        if abs(old_cost - cost) == 0:
            break

        old_cost = cost

    if verbose:
        print("\n\n")

    return Ts, Tv, Xc

#%%
mu = []
sigma = []
prop_r = []
prop_c = []

# D1
mu.append(np.matrix([[4.0, 0.5, 1.5],[1.8, 4.5, 1.1], [1.5, 1.5, 5.5]]))

prop_r.append([1./3, 1./3, 1./3])
prop_c.append([1./3, 1./3, 1./3])

# D2

mu.append(np.matrix([[4.0, 0.5, 1.5],[1.8, 4.5, 5.1], [3.5, 1.5, 5.5]]))

prop_r.append([0.2, 0.3, 0.5])
prop_c.append([0.2, 0.3, 0.5])

## D3

mu.append(np.matrix([[4.0, 0.5, 7.5, 0.5],[0.5, 3.5, 7.8, 0.5]]))

prop_r.append([0.5, 0.5])
prop_c.append([0.5, 0.2, 0.1, 0.2])

# D4

mu.append(np.matrix([[1.5, 1.5, 1.5, 1.5],
	[2.5, 1.5, 1.5, 1.5],
	[2.6, 2.6, 1.5, 1.5],
	[2.6, 2.6, 2.5, 1.5],
	[2.5, 2.5, 2.6, 2.5]]))

prop_r.append([0.1, 0.2, 0.2, 0.3, 0.2])
prop_c.append([0.25, 0.25, 0.25, 0.25])

###
ns = [600, 600, 300, 300]
ds = [300, 300, 200, 300]

nb_r = [3, 3, 2, 5]
nb_c = [3, 3, 4, 4]

sigmas = [0.1, 0.15, 0.2, 0.15]

algo1 = 'sinkhorn'
algo2 = 'sinkhorn'

reg_c = [0.1, 0.3, 0.3, 0.04]
reg_v = [0.1, 0.3, 0.3, 0.04]

nrep = 100
err_cc, err_r, err_c = [], [], []
#%%
np.random.seed(42)
#%%
for i in range(len(mu)):

    print("Data set D"+str(i+1))
    dat, z, w = generatedata(n=ns[i], d=ds[i], prop_r=prop_r[i],
                             prop_c=prop_c[i], mu=mu[i], noise=sigmas[i])
    reg1 = reg_c[i]
    reg2 = reg_v[i]

    for j in range(nrep):

        Ts, Tv, Xc = cot_clustering(X = dat, ns = nb_r[i], nv = nb_c[i],
                                    niter_cluster=2, niter=10, algo1=algo1, algo2=algo2, reg1=reg1, reg2=reg2, verbose = False)

        yr = Ts.argmax(1)
        yc = Tv.argmax(1)

        err_r.append(1 - np.sum(np.max(confusion_matrix(yr, z), 1)) / ns[i])
        err_c.append(1 - np.sum(np.max(confusion_matrix(yc, w), 1)) / ds[i])
        err_cc.append(err_c[-1] + err_r[-1] - err_c[-1] * err_r[-1])

    print("Rows clustering error = "+str(np.mean(err_r))+'±'+str(np.std(err_r)))
    print("Columns clustering error = "+str(np.mean(err_c))+'±'+str(np.std(err_c)))
    print("Co-clustering error = "+str(np.mean(err_cc))+'±'+str(np.std(err_cc)))