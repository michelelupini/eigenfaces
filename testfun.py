#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 18:34:51 2023

@author: michelelupini
"""

# import os
import numpy as np
# import matplotlib.pylab as plt
# import scipy.io as sio
from scipy import linalg

import homework1 as hm


def face_recognition(f_new,theta):
    # variabile bool che definisce se la nuova faccia appartiene allo spazio
    #   delle facce
    IsRecognized = 0
    
    # calcoliamo lo scarto tra la nuova immagine f_new e la faccia media
    phi_new = f_new - hm.f_meanface
    
    # proiettiamo phi_new nello spazio delle autofacce
    projection_new = np.dot(phi_new,hm.avec_reduced)
    
    # calcoliamo la minima distanza euclidea tra phi_new e lo spazio generato
    #   dalle autofacce
    eps = linalg.norm(phi_new - np.dot(hm.avec_reduced,projection_new))
    
    if eps < theta:
        IsRecognized = 1
    
    return IsRecognized, eps, projection_new


def which_subject(projection_new,theta_k):
    subject = 0
    flag = 1
    eps_min = linalg.norm(projection_new - hm.projection_training[0,:])
    
    k = 1
    for proj in hm.projection_training[1:]:
        eps_temp = linalg.norm(projection_new - proj)
        if eps_temp < eps_min:
            eps_min = eps_temp
            subject = k // hm.n_faces_training
        k = k + 1
    
    if eps_min > theta_k:
        flag = 0
    
    return subject, flag, eps_min