import numpy as np

def sinkhorn_scaling(a,b,K,numItermax=1000, stopThr=1e-9, verbose=False,log=False,always_raise=False, **kwargs):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    # init data
    Nini = len(a)
    Nfin = len(b)

    if len(b.shape) > 1:
        nbb = b.shape[1]
    else:
        nbb = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if nbb:
        u = np.ones((Nini, nbb)) / Nini
        v = np.ones((Nfin, nbb)) / Nfin
    else:
        u = np.ones(Nini) / Nini
        v = np.ones(Nfin) / Nfin

    # print(reg)
    # print(np.min(K))

    Kp = (1 / a).reshape(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v
        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1. / np.dot(Kp, v)

        zero_in_transp=np.any(KtransposeU == 0)
        nan_in_dual= np.any(np.isnan(u)) or np.any(np.isnan(v))
        inf_in_dual=np.any(np.isinf(u)) or np.any(np.isinf(v))
        if zero_in_transp or nan_in_dual or inf_in_dual:
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration in sinkhorn_scaling', cpt)
            #if zero_in_transp:
                #print('Zero in transp : ',KtransposeU)
            #if nan_in_dual:
                #print('Nan in dual')
                #print('u : ',u)
                #print('v : ',v)
                #print('KtransposeU ',KtransposeU)
                #print('K ',K)
                #print('M ',M)

            #    if always_raise:
            #        raise NanInDualError
            #if inf_in_dual:
            #    print('Inf in dual')
            u = uprev
            v = vprev

            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if nbb:
                err = np.sum((u - uprev)**2) / np.sum((u)**2) + \
                    np.sum((v - vprev)**2) / np.sum((v)**2)
            else:
                transp = u.reshape(-1, 1) * (K * v)
                err = np.linalg.norm((np.sum(transp, axis=0) - b))**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
    if log:
        log['u'] = u
        log['v'] = v

    if nbb:  # return only loss
        res = np.zeros((nbb))
        for i in range(nbb):
            res[i] = np.sum(
                u[:, i].reshape((-1, 1)) * K * v[:, i].reshape((1, -1)) * M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))