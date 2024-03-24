#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks as findPeaks


plt.style.use('ggplot')             # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False






# Read in an ECG signals and show the peaks detected
# --------------------------------------------------

#for 1st set of ECG data
l1D     = pd.read_csv('ecg1D.csv',
                      header=None)
ECGs    = l1D[0].values

#for 2nd set of ECG data
#l2D     = pd.read_csv('ecg2D.csv',
#                      header=None)
#ECGs    = l2D[1].values

(Pks,_)      = findPeaks(ECGs,prominence=0.5,distance=100)

plt.figure()
plt.plot(ECGs)
plt.plot(Pks,ECGs[Pks],'x')


# Perform baseline correction and show the baseline corrected signal
# ------------------------------------------------------------------


from scipy import sparse
from scipy.sparse.linalg import spsolve

def alsbase(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z   


ecgbase     = alsbase(ECGs, 10^5,0.000005,niter=50)     # 10^5 for lam, 0.000005 for p, 50 for iter
ecgcorr     = ECGs-ecgbase

plt.figure()
plt.subplot(211)
plt.plot(ECGs)

plt.plot(ecgbase, 
         color="C1",
         linestyle='dotted')
plt.subplot(212)
plt.plot(ecgcorr)
plt.show()