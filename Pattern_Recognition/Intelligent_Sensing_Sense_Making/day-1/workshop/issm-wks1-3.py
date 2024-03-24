#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks as findPeaks


plt.style.use('ggplot')             # if want to use the default style, set 'classic'
plt.style.use('ggplot')             # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False



def computeDists(x,y):
    dists       = np.zeros((len(y),len(x)))
    
    for i in range(len(y)):
        for j in range(len(x)):
            dists[i,j]  = (y[i]-x[j])**2
            
    return dists


def pltDistances(dists,xlab="X",ylab="Y",clrmap="viridis"):
    imgplt  = plt.figure()
    plt.imshow(dists,
               interpolation='nearest',
               cmap=clrmap)
    
    plt.gca().invert_yaxis()    # This is added so that the y axis start from bottom, instead of top
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid()
    plt.colorbar()
    
    return imgplt



def computeAcuCost(dists):
    acuCost     = np.zeros(dists.shape)
    acuCost[0,0]= dists[0,0]
                                                # Calculate the accumulated costs along the first row
    for j in range(1,dists.shape[1]):           # the running number starts from 1, not 0
        acuCost[0,j]    = dists[0,j]+acuCost[0,j-1]
        
                                                # Calculate the accumulated costs along the first column
    for i in range(1,dists.shape[0]):           # the running number starts from 1, not 0
        acuCost[i,0]    = dists[i,0]+acuCost[i-1,0]    
    
                                                # Calculate the accumulated costs from second column, row onwards    
    for i in range(1,dists.shape[0]):
        for j in range(1,dists.shape[1]):
            acuCost[i,j]    = min(acuCost[i-1,j-1],
                                  acuCost[i-1,j],
                                  acuCost[i,j-1])+dists[i,j]
            
    return acuCost


def doDTW(x,y,dists,acuCost):
    
    i       = len(y)-1
    j       = len(x)-1
                                                # Do backtracking to find out the warping path
    path    = [[j,i]]                           # The last point at the top-right is the starting point
                                                # Do note that, it is '[[j,i]]', not [[i,j]]
                                                # This is because we want the correct mapping in the pair ('path' is a list of pairs): 
                                                # the first element corresponds to 'x', the second element to 'y'
    
    while (i > 0) and (j > 0):
        if i==0:
            j   = j-1                           # When the search hits the bottom border, the next point to back track
                                                # is just the point to the left
            
        elif j==0:
            i   = i-1                           # When the search hits the left border, the next point to back track
                                                # is just the point to the bottom
        
        else:
                                                # For each point [i,j], there are only three points to back track:
                                                # the point to the left
                                                # the point to the bottom
                                                # the point to the left-bottom 
                                                
                                                # Among the three, find out which has the lowest accumulated cost
                                                # then that is the point
            
            if acuCost[i-1,j] == min(acuCost[i-1,j-1],
                                     acuCost[i-1,j],
                                     acuCost[i,j-1]):
                i   = i-1
                
            elif acuCost[i,j-1] == min(acuCost[i-1,j-1],
                                       acuCost[i-1,j],
                                       acuCost[i,j-1]):
                j   = j-1
                
            else:
                i   = i-1
                j   = j-1
                
        path.append([j,i])
        
    path.append([0,0])
    
    cost        = 0
    for [j,i] in path:
        cost    = cost+dists[i,j]
        
    return path,cost


def pltCostAndPath(acuCost,path,xlab="X",ylab="Y",clrmap="viridis"):
    px      = [pt[0] for pt in path]
    py      = [pt[1] for pt in path]
    
    
    imgplt  = pltDistances(acuCost,
                           xlab=xlab,
                           ylab=ylab,
                           clrmap=clrmap)  
    plt.plot(px,py)
    
    return imgplt
    


def pltWarp(s1,s2,path,xlab="idx",ylab="Value"):
    imgplt      = plt.figure()
    
    for [idx1,idx2] in path:
        plt.plot([idx1,idx2],[s1[idx1],s2[idx2]],
                 color="C4",
                 linewidth=2)
    plt.plot(s1,
             'o-',
             color="C0",
             markersize=3)
    plt.plot(s2,
             's-',
             color="C1",
             markersize=2)
    
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    
    return imgplt




# Extract the ECG
# --------------


l2D     = pd.read_csv('ecg2D.csv',
                      header=None)
ECGs    = l2D[1].values



# Find the peaks in the ECG
# -------------------------


(Pks,_)      = findPeaks(ECGs,prominence=0.5,distance=100)

plt.figure()
plt.plot(ECGs)
plt.plot(Pks,ECGs[Pks],'x')



# Extract ECG segments
# --------------------



def extractECG(ecg,pks,offset=15):
    segs    = []
    
    if pks[0]-offset < 0:
        start   = 1
    else:
        start   = 0
    
    for i in range(start,len(pks)-1):
        seg = ecg[(pks[i]-offset):(pks[i+1]-offset)]
        segs.append(seg)
        
    return segs


EcgSegs     = extractECG(ECGs,Pks)


# Comparing the first and second ecg, calculate the accumulated cost
e12D            = computeDists(EcgSegs[0],EcgSegs[1])
e12AD           = computeAcuCost(e12D)
(e12p,e12c)     = doDTW(EcgSegs[0],EcgSegs[1],e12D,e12AD)
pltCostAndPath(e12AD,e12p)
pltWarp(EcgSegs[0],EcgSegs[1],e12p)
print('The accumulated cost between ecg 1 and 2 is %f' % e12c)



# Comparing the second and third ecg, calculate the accumulated cost
e23D            = computeDists(EcgSegs[1],EcgSegs[2])
e23AD           = computeAcuCost(e23D)
(e23p,e23c)     = doDTW(EcgSegs[1],EcgSegs[2],e23D,e23AD)
pltWarp(EcgSegs[1],EcgSegs[2],e23p)  
pltCostAndPath(e23AD,e23p)
print('The accumulated cost between ecg 2 and 3 is %f' % e23c)


# Comparing the second and sixth ecg, calculate the accumulated cost
e26D            = computeDists(EcgSegs[1],EcgSegs[5])
e26AD           = computeAcuCost(e26D)
(e26p,e26c)     = doDTW(EcgSegs[1],EcgSegs[5],e26D,e26AD)
pltWarp(EcgSegs[1],EcgSegs[5],e26p)  
pltCostAndPath(e26AD,e26p)
print('The accumulated cost between ecg 2 and 6 is %f' % e26c)

plt.show()
