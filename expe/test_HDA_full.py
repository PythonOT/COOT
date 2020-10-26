#%% Reproduce HDA experiment of the paper


import os
import time
from random import shuffle
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.metrics import euclidean_distances
import sys
sys.path.append('../code')
from cot import *
from functools import reduce

import matplotlib.pylab as pl

from sklearn import svm


###############################################################################
#                   Part of code about arguments to modify                    #
#                                                                             #

#featuresToUse = ["CaffeNet4096", "GoogleNet1024"]  # surf, CaffeNet4096, GoogleNet1024
featuresToUse = ["CaffeNet4096", "GoogleNet1024"]  # surf, CaffeNet4096, GoogleNet1024
numRepetition = 10
n_samples_tab = [0,1,3,5] # nombre de samples par classe
# see function adaptData for available algorithms

#                                                                             #
#               End of part of code about arguments to modify                 #
###############################################################################


def generateSubset(X, Y, nPerClass):
    idx = []
    for c in np.unique(Y):
        idxClass = np.argwhere(Y == c).ravel()
        shuffle(idxClass)
        idx.extend(idxClass[0:min(nPerClass, len(idxClass))])
    return (X[idx, :], Y[idx])


# ---------------------------- DATA Loading Part ------------------------------

domainSourceNames = ['caltech10','amazon','webcam']
domainTargetNames = ['caltech10','amazon','webcam']

tests = []
data_source = {}
data_target = {}

min_max_scaler = preprocessing.MinMaxScaler()

for sourceDomain in domainSourceNames:
    possible_data = loadmat(os.path.join("../data/", "features", featuresToUse[0],
                                         sourceDomain + '.mat'))
    if featuresToUse == "surf":
        # Normalize the surf histograms
        feat = (possible_data['fts'].astype(float) /
                np.tile(np.sum(possible_data['fts'], 1),
                        (np.shape(possible_data['fts'])[1], 1)).T)
    else:
        feat = possible_data['fts'].astype(float)

    # Z-score
    #feat = preprocessing.scale(feat)
    #feat = min_max_scaler.fit_transform(feat)

    labels = possible_data['labels'].ravel()
    data_source[sourceDomain] = [feat, labels]

    for targetDomain in domainTargetNames:
        #if targetDomain is sourceDomain:
            possible_data = loadmat(os.path.join("../data/", "features", featuresToUse[1],
                                                 targetDomain + '.mat'))
            if featuresToUse == "surf":
                # Normalize the surf histograms
                feat = (possible_data['fts'].astype(float) /
                        np.tile(np.sum(possible_data['fts'], 1),
                                (np.shape(possible_data['fts'])[1], 1)).T)
            else:
                feat = possible_data['fts'].astype(float)
    
            # Z-score
            #feat = preprocessing.scale(feat)
            #feat = min_max_scaler.fit_transform(feat)

            #feat=np.dot(np.diag(1./np.sum(feat,axis=1)),feat)
    
            labels = possible_data['labels'].ravel()
            data_target[targetDomain] = [feat, labels]
    
            perClassSource = 20
            if sourceDomain == 'dslr':
                perClassSource = 8
            tests.append([sourceDomain, targetDomain, perClassSource])

meansAcc = {}
stdsAcc = {}
totalTime = {}

print("Feature used for source: ", featuresToUse[0])
print("Feature used for target: ", featuresToUse[1])


#%%
from sklearn.preprocessing import OneHotEncoder as onehot
from sklearn.neighbors import KNeighborsClassifier
enc = onehot(handle_unknown='ignore',sparse=False)


def comp_(v=1e6):
    def comp(x,y):
        if x==y or y==-1:
            return 0
        else:
            return v
    return comp

def compute_cost_matrix(ys,yt,v=np.inf):
    M=ot.dist(ys.reshape(-1,1),yt.reshape(-1,1),metric=comp_(v))
    return M

#%%
    
import ot 

# -------------------- Main testing loop --------------------------------------

all_results={}

for n_samples in n_samples_tab:
    dict_tmp={}
    
    for test in tests:
        Sname = test[0]
        Tname = test[1]
        perClassSource = test[2]
        testName = Sname.upper()[:1] + '->' + Tname.upper()[:1]
        print(testName, end=" ")

        dict_tmp[testName] = {} 
        
        perf_baseline= []
        perf_COT = []
        time_COT = []

        
        # --------------------II. prepare data-------------------------------------
        Sx_tot = data_source[Sname][0]
        Sy_tot = data_source[Sname][1]
        Tx_tot = data_target[Tname][0]
        Ty_tot = data_target[Tname][1]
        
        for repe in range(numRepetition):
            Sx, Sy = generateSubset(Sx_tot, Sy_tot, perClassSource)
            Tx, Ty = generateSubset(Tx_tot, Ty_tot, perClassSource)
            
            idx = np.random.permutation(Tx.shape[0])
            for i in range(Tx.shape[0]):
                Tx=Tx[idx,:]
                Ty=Ty[idx]
            
            #semi supervision
            nb_perclass = n_samples
            Sy_ss =-1*np.ones_like(Sy)    
            
            for c in np.unique(Sy):
                idx=np.where(Sy==c)[0]
                Sy_ss[idx[:nb_perclass]]=c
            
            M_lin = compute_cost_matrix(Ty,Sy_ss,v=1e2)
            # --- compuet baseline score by 1NN
            
            idx=np.where(Sy_ss!=-1)[0]
            idx_inv=np.where(Sy_ss==-1)[0]
            
            if nb_perclass!=0:
                neigh = KNeighborsClassifier(n_neighbors=3).fit(Sx[idx,:],Sy[idx])
                ys_estimated = neigh.predict(Sx[idx_inv,:])
                perf_baseline.append(100*np.mean(Sy[idx_inv]==ys_estimated))
                print('Accuracy 3NN on source (baseline): {:.2f}'.format(100*np.mean(Sy[idx_inv]==ys_estimated)))
            
            #print('mean perf',np.mean(r))
        
            # --------------------III. run experiments---------------------------------
        
 
            # ------------------- COT -----------------------------------------------
            ot.tic()
            Tv, Tc, cost = cot_numpy(Sx, Tx, niter=100,C_lin=M_lin.T, 
                                     algo='sinkhorn', reg=1e0,
                                     algo2='emd', verbose = False)
            time_COT.append(ot.toc())

            yt_onehot = enc.fit_transform(Ty.reshape(-1,1))
            ys_onehot_estimated = Tv.shape[0]*np.dot(Tv,yt_onehot)
            ys_estimated=enc.inverse_transform(ys_onehot_estimated).reshape(-1)
            
            perf=100*np.mean(Sy[idx_inv]==ys_estimated[idx_inv])
            perf_COT.append(perf)
            print('Accuracy COT labelprop: {:.2f}'.format(perf))
        
  
        if n_samples!=0:
            print('mean perf baseline= {:.2f} ({:.2f})'.format(np.mean(perf_baseline),np.std(perf_baseline)))
        print('mean perf COT= {:.2f} ({:.2f})'.format(np.mean(perf_COT),np.std(perf_COT)))

        dict_tmp[testName]['baseline']=perf_baseline
        dict_tmp[testName]['COT']=perf_COT
        dict_tmp[testName]['time_COT']=time_COT

    all_results[n_samples] = dict_tmp

np.save('results_D_to_G.npy',all_results)  