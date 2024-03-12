#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 22:54:20 2023

@author: michelelupini
"""

import os
# import sys
import numpy as np
import matplotlib.pylab as plt
# import scipy.io as sio
# from scipy import linalg

import eigfun


#def main():
path = ".../archive"  # inserire il path fino alla cartella archive

n_individuals = 40  # numero di soggetti
n_faces = 10  # numero di immagini per ciascun soggetto

# numero di pixel di ciascuna immagine
m = 112
n = 92
mn = m*n

# definiamo il numero di individui noti da usare nelle fasi di training e test
n_individuals_known = 40  # <= n_individuals

# definiamo il numero di facce da usare nelle fasi di training e test
#   per ogni individuo
n_faces_training = 6  # > 0
n_faces_test = n_faces - n_faces_training

AllTraining = True
if n_individuals_known < n_individuals:
    AllTraining = False
    n_individuals_unknown = n_individuals - n_individuals_known


# -------------------- SCAN archive -------------------- #

# separiamo le facce training e test, 
# rappresentiamo ciascuna faccia come una matrice di mxn entrate (pixel), 
#   indicizzata da individuo e immagine dello stesso
images = np.zeros((n_individuals,n_faces,m,n,))
images_training = np.zeros((n_individuals_known,n_faces_training,m,n))
images_test = np.zeros((n_individuals_known,n_faces_test,m,n))
if not AllTraining:
    images_unknown_test = np.zeros((n_individuals_unknown,n_faces,m,n))

i = -1
#for cartella, sottocartelle, files in os.walk(os.getcwd()):           # temp
for cartella, sottocartelle, files in os.walk(path):
    #print(f"Ci troviamo nella cartella: '{cartella}'")                # DEBUG
    #print(f"Le sottocartelle presenti sono: '{sottocartelle}'")       # DEBUG
    #print(f"I file presenti sono: '{files}'")                         # DEBUG
    j = 0
    for file in files:
        if file.endswith(".pgm"):
            #print(file)                                               # DEBUG
            images[i,j,:,:] = \
                np.array(plt.imread(cartella+'/'+file), dtype='float64')[:,:]
            if i < n_individuals_known:
                if j < n_faces_training:
                    images_training[i,j,:,:] = images[i,j,:,:]
                else:
                    images_test[i,j-n_faces_training,:,:] = images[i,j,:,:]
            else:
                images_unknown_test[i-n_individuals_known,j,:,:] = images[i,j,:,:]
            j = j + 1
    #print()                                                           # DEBUG
    i = i + 1

# EDIT: si può eliminare giocando con gli indici nel ciclo di sopra (ottimizzazione)
# riduciamo le dimansioni dello spazio delle immagini accorpando gli individui
L = n_faces_training * n_individuals_known  # numero di facce di training
training_reduced = np.zeros((L,m,n))
i = 0
for tensore in images_training:
    for matrice in tensore:
        #print(matrice)                                                # DEBUG
        training_reduced[i,:,:] = matrice
        i = i + 1
        
L_test = n_faces_test * n_individuals_known
test_reduced = np.zeros((L_test,m,n))
i = 0
for tensore in images_test:
    for matrice in tensore:
        #print(matrice)                                                # DEBUG
        test_reduced[i,:,:] = matrice
        i = i + 1

if not AllTraining:
    L_unknown_test = n_faces * n_individuals_unknown
    test_unknown_reduced = np.zeros((L_unknown_test,m,n))
    i = 0
    for tensore in images_test:
        for matrice in tensore:
            #print(matrice)                                            # DEBUG
            test_unknown_reduced[i,:,:] = matrice
            i = i + 1
        
del matrice, tensore
del cartella, file, files, i, j, sottocartelle

# -------------------- fine SCAN archive -------------------- #    
    
# -------------------- TRAINING phase -------------------- #

# trasformiamo le immagini in un set di L vettori in R^(mn)
#   usiamo la "base canonica"
# oss. nella base scelta consideriamo prima i coefficienti della riga 0,
#   poi i coefficienti della riga 1, e così via fino alla riga m
f = np.reshape(training_reduced, (L,mn))
# training_reconstructed = np.reshape(f, (L,m,n))

# definiamo la faccia media, come media di tutte le facce di tutti gli 
#   individui nel training set
f_meanface = eigfun.mean_face(f)
meanface = np.reshape(f_meanface, (m,n))

# rappresentazione della faccia media
# plt.imshow(meanface)
# plt.show()

# definiamo i vettori phi, sottraendo le immagini alla faccia media
phi = np.zeros((L,mn))
phi = f - f_meanface


# # Metodo C completa
# # calcoliamo la matrice di convarianza C
# C = eigfun.covariance_matrix(phi)
# # definiamo la percentuale p di soglia rispetto al masssimo autovalore, sotto a
# #   cui filtrare le coppie L autovalori-autovettori con autovalori più piccoli
# p = 0.01
# L_red = 202  # scegliere un intero positivo < L per fissare L_reduced
# aval_reduced, avec_reduced, L_reduced = eigfun.eig_reduced(C,L,p,L_red)


# Metodo C ridotta
var = 0.5
L_red = -1 # settare -1 se si vuole usare var
aval_reduced, avec_reduced, L_reduced = eigfun.eig_covariance_reduced(phi,var,L_red)

# proiettiamo le facce di training nello spazio generato dagli autovettori rimasti
# per ognuna delle L facce di training determiniamo L_reduced coefficienti 
projection_training = np.zeros((L,L_reduced))
# projection_training = np.dot(f,avec_reduced)
projection_training = np.dot(phi,avec_reduced)

# -------------------- fine TRAINING phase -------------------- #


