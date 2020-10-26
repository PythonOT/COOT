import numpy as np
import ot
from scipy import stats
from scipy.sparse import random
from bregman import sinkhorn_scaling

def random_gamma_init(p,q, **kwargs):
    """ Returns random coupling matrix with marginal p,q
    """
    rvs=stats.beta(1e-1,1e-1).rvs
    S=random(len(p), len(q), density=1, data_rvs=rvs)
    return sinkhorn_scaling(p,q,S.A, **kwargs)

def init_matrix_np(X1, X2, v1, v2):
    """Return loss matrices and tensors for COOT fast computation
    Returns the value of |X1-X2|^{2} \otimes T as done in [1] based on [2] for the Gromov-Wasserstein distance. 
    Where :
        - X1 : The source dataset of shape (n,d)
        - X2 : The target dataset of shape (n',d')
        - v1 ,v2 : weights (histograms) on the columns of resp. X1 and X2
        - T : Coupling matrix of shape (n,n')
    Parameters
    ----------
    X1 : numpy array, shape (n, d)
         Source dataset
    X2 : numpy array, shape (n', d')
         Target dataset
    v1 : numpy array, shape (d,)
        Weight (histogram) on the features of X1.
    v2 : numpy array, shape (d',)
        Weight (histogram) on the features of X2.    
    
    Returns
    -------
    constC : ndarray, shape (n, n')
        Constant C matrix (see paragraph 1.2 of supplementary material in [1])
    hC1 : ndarray, shape (n, d)
        h1(X1) matrix (see paragraph 1.2 of supplementary material in [1])
    hC2 : ndarray, shape (n', d')
        h2(X2) matrix (see paragraph 1.2 of supplementary material in [1])
    References
    ----------
    .. [1] Redko Ievgen, Vayer Titouan, Flamary R{\'e}mi and Courty Nicolas
          "CO-Optimal Transport"
    .. [2] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    def f1(a):
        return (a ** 2)

    def f2(b):
        return (b ** 2)

    def h1(a):
        return a

    def h2(b):
        return 2 * b

    constC1 = np.dot(np.dot(f1(X1), v1.reshape(-1, 1)),
                     np.ones(f1(X2).shape[0]).reshape(1, -1))
    constC2 = np.dot(np.ones(f1(X1).shape[0]).reshape(-1, 1),
                     np.dot(v2.reshape(1, -1), f2(X2).T))

    constC = constC1 + constC2
    hX1 = h1(X1)
    hX2 = h2(X2)

    return constC, hX1, hX2


def cot_numpy(X1, X2, w1 = None, w2 = None, v1 = None, v2 = None,
              niter=10, algo='emd', reg=0,algo2='emd',
              reg2=0, verbose=True, log=False, random_init=False, C_lin=None):

    """ Returns COOT between two datasets X1,X2 (see [1])
    
    The function solves the following optimization problem:
    .. math::
        COOT = \min_{Ts,Tv} \sum_{i,j,k,l} |X1_{i,k}-X2_{j,l}|^{2}*Ts_{i,j}*Tv_{k,l}

    Where :
    - X1 : The source dataset
    - X2 : The target dataset
    - w1,w2  : weights (histograms) on the samples (rows) of resp. X1 and X2
    - v1,v2  : weights (histograms) on the features (columns) of resp. X1 and X2

    Parameters
    ----------
    X1 : numpy array, shape (n, d)
         Source dataset
    X2 : numpy array, shape (n', d')
         Target dataset
    w1 : numpy array, shape (n,)
        Weight (histogram) on the samples of X1. If None uniform distribution is considered.
    w2 : numpy array, shape (n',)
        Weight (histogram) on the samples of X2. If None uniform distribution is considered.
    v1 : numpy array, shape (d,)
        Weight (histogram) on the features of X1. If None uniform distribution is considered.
    v2 : numpy array, shape (d',)
        Weight (histogram) on the features of X2. If None uniform distribution is considered.
    niter : integer
            Number max of iterations of the BCD for solving COOT.
    algo : string
            Choice of algorithm for solving OT problems on samples each iteration. Choice ['emd','sinkhorn'].
            If 'emd' returns sparse solution
            If 'sinkhorn' returns regularized solution
    algo2 : string
            Choice of algorithm for solving OT problems on features each iteration. Choice ['emd','sinkhorn'].
            If 'emd' returns sparse solution
            If 'sinkhorn' returns regularized solution
    reg : float
            Regularization parameter for samples coupling matrix. Ignored if algo='emd'
    reg2 : float
            Regularization parameter for features coupling matrix. Ignored if algo='emd'
    eps : float
        Threshold for the convergence
    random_init : bool
            Wether to use random initialization for the coupling matrices. If false identity couplings are considered.
    log : bool, optional
         record log if True
    C_lin : numpy array, shape (n, n')
            Prior on the sample correspondences. Added to the cost for the samples transport

    Returns
    -------
    Ts : numpy array, shape (n,n')
           Optimal Transport coupling between the samples
    Tv : numpy array, shape (d,d')
           Optimal Transport coupling between the features
    cost : float
            Optimization value after convergence
    log : dict
        convergence information and coupling marices
    References
    ----------
    .. [1] Redko Ievgen, Vayer Titouan, Flamary R{\'e}mi and Courty Nicolas
          "CO-Optimal Transport"
    Example
    ----------
    import numpy as np
    from cot import cot_numpy
    
    n_samples=300
    Xs=np.random.rand(n_samples,2)
    Xt=np.random.rand(n_samples,1)
    cot_numpy(Xs,Xt)
    """  
    if v1 is None:
       v1 = np.ones(X1.shape[1]) / X1.shape[1]  # is (d,)
    if v2 is None:
       v2 = np.ones(X2.shape[1]) / X2.shape[1]  # is (d',)
    if w1 is None:
       w1 = np.ones(X1.shape[0]) / X1.shape[0]  # is (n',)
    if w2 is None:
       w2 = np.ones(X2.shape[0]) / X2.shape[0]  # is (n,)

    if not random_init:
        Ts = np.ones((X1.shape[0], X2.shape[0])) / (X1.shape[0] * X2.shape[0])  # is (n,n')
        Tv = np.ones((X1.shape[1], X2.shape[1])) / (X1.shape[1] * X2.shape[1])  # is (d,d')
    else:
       Ts=random_gamma_init(w1,w2) 
       Tv=random_gamma_init(v1,v2)


    constC_s, hC1_s, hC2_s = init_matrix_np(X1, X2, v1, v2)

    constC_v, hC1_v, hC2_v = init_matrix_np(X1.T, X2.T, w1, w2)
    cost = np.inf

    log_out ={}
    log_out['cost'] = []
    
    for i in range(niter):
        Tsold = Ts
        Tvold = Tv
        costold = cost

        M = constC_s - np.dot(hC1_s, Tv).dot(hC2_s.T)
        if C_lin is not None:
            M=M+C_lin           
        if algo == 'emd':
            Ts = ot.emd(w1, w2, M, numItermax=1e7)
        elif algo == 'sinkhorn':
            Ts = ot.sinkhorn(w1, w2, M, reg)

        M = constC_v - np.dot(hC1_v, Ts).dot(hC2_v.T)
        
        if algo2 == 'emd':
            Tv = ot.emd(v1, v2, M, numItermax=1e7)
        elif algo2 == 'sinkhorn':
            Tv = ot.sinkhorn(v1,v2, M, reg2)

        delta = np.linalg.norm(Ts - Tsold) + np.linalg.norm(Tv - Tvold)
        cost = np.sum(M * Tv)
        
        if log:
            log_out['cost'].append(cost)
            
        if verbose:
            print('Delta: {0}  Loss: {1}'.format(delta, cost))

        if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            if verbose:
                print('converged at iter ', i)
            break
    if log:
        return Ts, Tv, cost, log_out
    else:
        return Ts, Tv, cost


