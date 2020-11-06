import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../code/")
from cot import *

def cot_clustering(X, ns, nv, niter_cluster=10, niter=10, algo1='emd', algo2 = 'emd', reg1 = 0., reg2 = 0., verbose = False):
    Xc = np.random.randn(ns, nv)
    old_cost = 0

    for i in range(niter_cluster):
        Ts, Tv, cost = cot_numpy(X, Xc, niter=niter, algo=algo1, reg = reg1, algo2 = algo2, reg2 = reg2, verbose=verbose)
        Xc = Ts.T.dot(X).dot(Tv) * ns * nv

        if verbose:
            print(cost)

        if np.abs(old_cost - cost) < 1e-7:
            if verbose:
                print('converged at iter ', i)
            break

        old_cost = cost

    if verbose:
        print("\n\n")

    return Ts, Tv, Xc

# Import the data and get the rating matrix
df = pd.read_csv("../data/ml-100k/u.data", delimiter='\t',header=None, names=["user", "item", "rating", "timestamp"])
R_df = df.pivot(index = 'user', columns ='item', values = 'rating').fillna(0).values

movies = pd.read_csv('../data/ml-100k/u.item', sep='|',  header=None, encoding='latin-1').values[:,1]

mean_ratings = np.true_divide(R_df.sum(0),(R_df!=0).sum(0))
idx_best = np.argsort(mean_ratings)[::-1].tolist()

n_users, n_items = len(df.user.unique()), len(df.item.unique())

viz_orig = False
viz_cot = True

if viz_orig:
    plt.figure(figsize = (9,6))
    plt.imshow(R_df, cmap='Blues')
    plt.xlabel("Users", fontsize = 15)
    plt.ylabel("Movies", fontsize = 15)
    plt.title('Original MovieLens matrix', fontsize = 20)
    plt.xticks([])
    plt.yticks([])
    plt.show()

ns = 10
nv = 20
algo1 = 'emd'
algo2 = 'emd'

Ts, Tv, Xc = cot_clustering(R_df, ns = ns, nv = nv,
                niter_cluster=10, niter=300, algo1=algo1, algo2=algo2, reg1=0, reg2=0, verbose = False)

yc = Tv.argmax(1)

sum_ratings_cot = np.sum(Xc, axis = 0)
idx_mov = np.argsort(sum_ratings_cot)[::-1]
idx_user = np.argsort(np.sum(Xc[:,idx_mov], axis = 1))
Xc = Xc[:,idx_mov]
Xc = Xc[idx_user,:]

if viz_cot:
    plt.figure(figsize = (9,6))
    plt.imshow(Xc, cmap='Blues')
    plt.xlabel("Users clusters", fontsize = 15)
    plt.ylabel("Movies clusters", fontsize = 15)
    plt.title('Summarized MovieLens matrix', fontsize = 20)
    plt.xticks([])
    plt.yticks([])
    plt.show()

print("Movies in the most rated cluster")
idx_best_cluster = np.where(yc == idx_mov[0])
print(movies[idx_best_cluster])

print("\nMovies in the least rated cluster")
idx_worst_cluster = np.where(yc == idx_mov[-1])
print(movies[idx_worst_cluster])



