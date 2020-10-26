# Reproduce Fig 4 of the paper

import sys
sys.path.append('../lib')
import numpy as np
from scipy import ndimage
import scipy as sp
import matplotlib.pylab as pl
import ot
import scipy.io


#%%

from sklearn.datasets import fetch_olivetti_faces

dataset = fetch_olivetti_faces()
Xtot1 = dataset.data

y=dataset.target


def get_data(x,n):
    
    idx=np.random.permutation(x.shape[0])
        
    xr =x[idx[:n],:]
    return xr
        
        
#%%
n=400
x=get_data(Xtot1,n)
x=x/x.max()

#%%
from cot import cot_numpy

np.random.seed=1986

def cot_clustering(X,ns,nv,niter_cluster=10, niter=10, algo='emd'):
    
    Xc=np.random.randn(ns,nv)
    old_cost=0
    
    for i in range(niter_cluster):
        Ts,Tv,cost=cot_numpy(X, Xc, niter=10, algo=algo,verbose=False)
        
        Xc=Ts.T.dot(X).dot(Tv)*ns*nv
        
        print(cost)
        
        if abs(old_cost-cost)==0:
            break
        
        old_cost=cost
        
    return Ts,Tv,Xc
        
    
    


ns=9

nv=40
    
ot.tic()
Ts,Tv,Xc= cot_clustering(x,ns,nv,niter_cluster=50, niter=20, algo='emd')
ot.toc()




#%% plot var pos



mxc=Xc.mean(0)
isort=np.argsort(mxc)
#isort=np.arange(nv)
Iclass=np.zeros((64,64))

#pl.figure(4,(8,5))

for i in range(nv):
    #pl.subplot(5,8,i+1)
    #pl.imshow(Tv[:,isort[i]].reshape((64,64))*mxc[isort[i]]/np.max(mxc),vmax=Tv.max(),cmap='jet')
    #pl.title('V={:1.2f}'.format(Xc.mean(0)[isort[i]]))
    Iclass[Tv[:,isort[i]].reshape((64,64))>0]=i#*(Tv[:,isort[i]].reshape((64,64))>0)
    #pl.axis('off')

    
pl.figure(5)
pl.imshow(Iclass,cmap='jet')
pl.title('Pixel (variable) clustering')


#%%

pl.figure(6,(16,10))

clustrs=Tv.dot(Xc.T).T

for i in range(ns):
    pl.subplot(3,4,i+1)
    pl.imshow(clustrs[i,:].reshape((64,64)),cmap='gray')
    pl.axis('off')



#%%
    
data_rec=Ts.dot(Xc).dot(Tv.T)

idplot=np.random.permutation(y.shape[0])
nbv=100

pl.figure(7,(10,10))
for i in range(nbv):
    pl.subplot(10,10,i+1)
    pl.imshow(x[idplot[i],:].reshape((64,64)),cmap='gray')


pl.figure(8,(10,10))
for i in range(nbv):
    pl.subplot(10,10,i+1)
    pl.imshow(data_rec[idplot[i],:].reshape((64,64)),cmap='gray') 
      

#%%

nx=3
ny=3  
delta=1  
wimg=64
Icluster=np.zeros((wimg*ny+delta*(ny-1),wimg*nx+delta*(nx-1)))+clustrs.min()
for i in range(ny):
    for j in range(nx):
        Icluster[i*(wimg+delta):i*(wimg+delta)+wimg,j*(wimg+delta):j*(wimg+delta)+wimg]=clustrs[i+ny*j,:].reshape((64,64))


pl.figure(1)
pl.imshow(Icluster,cmap='gray')   
pl.title('Sample clusters')
pl.axis('off')




#%%

isel=range(80,400,10)

nx=3
ny=3  
delta=1  
wimg=64
Idata=np.zeros((wimg*ny+delta*(ny-1),wimg*nx+delta*(nx-1)))+clustrs.min()
for i in range(ny):
    for j in range(nx):
        Idata[i*(wimg+delta):i*(wimg+delta)+wimg,j*(wimg+delta):j*(wimg+delta)+wimg]=x[isel[i+ny*j],:].reshape((64,64))


pl.figure(1)
pl.imshow(Idata,cmap='gray')   
pl.title('Face dataset')
pl.axis('off')


#%% 

pl.figure(10,(12,4))
pl.clf()
pl.subplot(1,3,1)
pl.imshow(Idata,cmap='gray')   
pl.title('Face dataset')
pl.axis('off')

pl.subplot(1,3,2)
pl.imshow(Icluster,cmap='gray')   
pl.title('Centroids for sample clustering')
pl.axis('off')

pl.subplot(1,3,3)
pl.imshow(Iclass+1,cmap='nipy_spectral')
pl.title('Pixel (feature) clustering')
pl.axis('off')

left, bottom, width, height = pl.gca().get_position().bounds
cax = pl.gcf().add_axes([left+1.05*width, height*0.23, width*0.05, height])
pl.colorbar( cax=cax)

pl.savefig('./clootclustering_faces.png')
pl.savefig('./clootclustering_faces.pdf',bbox_inches='tight')
           
        

