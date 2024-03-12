#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 00:11:21 2023

@author: michelelupini
"""

# import os
import numpy as np
# import matplotlib.pylab as plt
# import scipy.io as sio
from scipy import linalg

# import homework1 as hm


def mean_face(f):
    L = np.shape(f)[0]
    meanface = np.sum(f, axis=0)/L
    
    return meanface


def covariance_matrix(f):
    L = np.shape(f)[0]
    mn = np.shape(f)[1]
    C = np.zeros((mn,mn))
    phi = np.transpose(f)
    C = np.dot(phi,f)/L
    
    return C


def eig_reduced(C,L,var,L_red):
    mn = np.shape(C)[0]
    
    # calcoliamo L autovettori e autovalori di C, in ordine crescente
    aval, avec = linalg.eigh(C,subset_by_index=(mn-L, mn-1))
    aval = np.flip(aval)
    for i in range(mn):
        avec[i,:] = np.flip(avec[i,:])
    
    if L_red == -1:
        sum_tot = sum(aval)
        sum_aval = 0
        for i in range(L):
            sum_aval = sum_aval + aval[i]
            if sum_aval/sum_tot > var:
                L_reduced = i
                break
    else:
        if L_red > L:
            L_reduced = L
        else:
            L_reduced = L_red
        
    aval_reduced = aval[0:L_reduced]
    avec_reduced = avec[:, 0:L_reduced]
    
    return aval_reduced, avec_reduced, L_reduced


def eig_covariance_reduced(f,var,L_red):
    L = np.shape(f)[0]
    # mn = np.shape(f)[1]
    phi = np.transpose(f)
    
    C = np.zeros((L,L))
    C = np.dot(f,phi)/L
    
    eigenvalues, eigenvectors = np.linalg.eig(C)

    # Ordina gli autovalori in ordine decrescente e gli autovettori corrispondenti
    sorted_indices = np.argsort(eigenvalues)[::-1]
    aval = eigenvalues[sorted_indices]
    avec = eigenvectors[:, sorted_indices]
    
    if L_red == -1:
        sum_tot = sum(aval)
        sum_aval = 0
        for i in range(L):
            sum_aval = sum_aval + aval[i]
            rapp = sum_aval/sum_tot
            if rapp > var:
                L_red = i
                break
    else:
        if L_red > L:
            L_red = L
        
    aval_red = aval[0:L_red]
    avec_red = avec[:,0:L_red]
    
    u = np.dot(phi,avec_red)
    u = u / np.linalg.norm(u, axis=0)
    
    return aval_red, u, L_red