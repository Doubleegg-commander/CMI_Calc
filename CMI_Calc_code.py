from scipy.special import digamma
from scipy.stats import rankdata as rank
from scipy.spatial.distance import cdist
from math import gamma, log, pi
from numpy import array, abs, max, vstack, zeros, cov
from numpy.random import normal as rnorm
from multiprocessing import Process
from numpy.linalg import det
import pandas as pd
import numpy as np
import xarray as xr
import numpy.ma as ma
import multiprocessing as mp
import os
from joblib import Parallel, delayed

'''
Computation of the conditional mutual information (CMI)
'''
def construct_empirical_copula(x):
    (N, d) = x.shape
    xc = zeros([N, d])
    for i in range(0, d):
        xc[:, i] = rank(x[:, i])/N

    return xc

def entknn(x, k=3, dtype='chebychev'):
    (N, d) = x.shape

    g1 = digamma(N) - digamma(k)

    if dtype == 'euclidean':
        cd = pi**(d/2) / 2**d / gamma(1+d/2)
    else: 
        cd = 1

    logd = 0
    dists = cdist(x, x, dtype)
    dists.sort()
    for i in range(0, N):
        logd = logd + log(2 * dists[i, k]) * d / N

    return (g1 + log(cd) + logd)

def copent(x, k=3, dtype='chebychev', log0=False):
    xarray = array(x)

    if log0:
        (N, d) = xarray.shape
        max1 = max(abs(xarray), axis=0)
        for i in range(0, d):
            if max1[i] == 0:
                xarray[:, i] = rnorm(0, 1, N)
            else:
                xarray[:, i] = xarray[:, i] + \
                    rnorm(0, 1, N) * max1[i] * 0.000005

    xc = construct_empirical_copula(xarray)

    try:
        return -entknn(xc, k, dtype)
    except ValueError:  
        return copent(x, k, dtype, log0=True)

def ci(x, y, z, k=3, dtype='chebychev'):
    xyz = vstack((x, y, z)).T
    yz = vstack((y, z)).T
    xz = vstack((x, z)).T
    return copent(xyz, k, dtype) - copent(yz, k, dtype) - copent(xz, k, dtype)

def calc_copula_transfer_entropy(Source_data, Target_data, Background_data, k=3, dtype='chebychev'):

    te = ci(Target_data, Source_data, Background_data, k, dtype)

    if te >= 0:
        return te
    else:
        return 0

def calc_copula_transfer_entropy_parallel(args):
    x_ind, y_ind, Factor_data1, Target_data2, Background_data = args
    try:
        return (x_ind, y_ind, calc_copula_transfer_entropy(Factor_data1[:, y_ind, x_ind], Target_data2[:, y_ind, x_ind], Background_data[:, y_ind, x_ind]))
    except:
        return (x_ind, y_ind, -9999.0)


def Calc_Copula_Transfer_Entropy_Array_Parallel(Factor_data1, Target_data2, Background_data):
    area_shape = np.array([Factor_data1.shape[-2], Factor_data1.shape[-1]])
    TE_Matrix = np.ones(area_shape) * -9999.0

    with mp.Pool() as pool:
        results = pool.map(calc_copula_transfer_entropy_parallel, [(
            x_ind, y_ind, Factor_data1, Target_data2, Background_data) for y_ind in range(area_shape[0]) for x_ind in range(area_shape[1])])

    for x_ind, y_ind, te in results:
        if te != -9999.0:
            TE_Matrix[y_ind, x_ind] = te

    TE_Matrix = ma.masked_where(TE_Matrix == -9999.0, TE_Matrix)
    TE_Matrix[np.where(TE_Matrix < 0)] = 0

    return TE_Matrix

'''
Significance Testing Methods for Conditional Mutual Information (CMI) 
'''
def Calc_Copula_Random_Infor_parallel(Factor1,Factor2,Background_data,ratio):
    area_shape=np.array([Factor1.shape[-2],Factor1.shape[-1]])
    TE_Matrix=np.ones(area_shape)
    TE_Matrix*=-999.0
    
    def process_pixel(y_ind, x_ind):
        try:
            random_facor=np.random.choice(Factor2[:,y_ind,x_ind],Factor2[:,y_ind,x_ind].shape[0]*ratio)
            return (y_ind,x_ind,calc_copula_transfer_entropy(Factor1[:,y_ind,x_ind],random_facor,Background_data[:,y_ind,x_ind]))
        except:
            return (y_ind,x_ind,-999.0)

    results = Parallel(n_jobs=np.cpu_count())(delayed(process_pixel)(y_ind,x_ind) for y_ind in range(area_shape[0]) for x_ind in range(area_shape[1]))
    
    for y_ind, x_ind, value in results:
        TE_Matrix[y_ind,x_ind] = value
                
    TE_Matrix=ma.masked_where(TE_Matrix==-999.0,TE_Matrix)
    TE_Matrix[np.where(TE_Matrix<0)]=0
    
    return TE_Matrix

'''
The code for calculating CMI is based on the implementation available at: https://github.com/majianthu/pycopent
'''
