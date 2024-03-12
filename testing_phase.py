#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 16:06:19 2023

@author: michelelupini
"""

# import os
import numpy as np
# import matplotlib.pylab as plt
# import scipy.io as sio
# from scipy import linalg

# import eigfun
import testfun
import homework1 as hm

# -------------------- TESTING phase -------------------- #
# -------------------- caso standard -------------------- #

# data una nuova immagine facciale determinare se il volto è presente
#   nel dataset
# se il volto è riconosciuto, determinare a quale soggetto corrisponde la 
#   faccia
 
# definiamo una soglia theta sotto a cui l'individuo NON è riconosciuto
#theta = 2500
theta = 5000

# definiamo una soglia theta_k sotto a cui l'individuo appartiene alla classe
#   delle facce del dato individuo considerato
#theta_k = 200
theta_k = 5000

print("\nUsiamo i seguenti dati:")
print(f"numero facce di training = {hm.n_faces_training}")
print(f"numero facce di test = {hm.n_faces_test}")
if hm.L_red == -1:
    print(f"var = {hm.var}, corrispondente a L' = {hm.L_reduced}")
else:
    print(f"L' = {hm.L_reduced}")
print(f"theta = {theta}")
print(f"theta_k = {theta_k}")

f_test = np.reshape(hm.test_reduced, (hm.L_test, hm.mn))

# vettore di L_test elementi, le cui entrate possono assumere i seguenti valori:
#   0: l'immagine test non è sufficientemente vicina allo spazio delle facce
#   1: la distanza dell'immagine test dallo spazio delle facce è sotto al soglia
test_IsRecognized = np.zeros(hm.L_test)

# vettore di L_test elementi contentente la distanza in norma euclidea
#   dell'immagine test dallo spazio delle facce
test_distance_face_space = np.zeros(hm.L_test)

# inizializziamo una matrice L_test x 2, tale che
#   colonna 0: associazione tra faccia test e individuo nel dataset inizializzata a -1
#   colonna 1: flag che può assumere i seguenti valori
#       -1: valore di inizializzazione
#       0: l'immagine test non è vicina ad alcuna faccia di training
#       1: la distanza dell'immagine test da una faccia di training è sotto la soglia
test_subject_link = np.full((hm.L_test,2), -1)

# vettore di L_test elementi contente la distanza in norma euclidea 
#   dell'immagine test dalla proiezione più vicina delle facce di training
#   nello spazio delle eigenfaces
test_distance_class_face = np.zeros(hm.L_test)

# vettore contentente le associazioni corrette tra faccia testata e individuo
test_subject_expected = np.zeros(hm.L_test)
for i in range(hm.L_test):
    test_subject_expected[i] = i // hm.n_faces_test

i = 0
for f_new in f_test:
    test_IsRecognized[i], test_distance_face_space[i], projection_new = \
        testfun.face_recognition(f_new, theta)
    test_subject_link[i,0], test_subject_link[i,1], test_distance_class_face[i] = \
        testfun.which_subject(projection_new,theta_k)
        
    i = i + 1

# numero di facce sufficientemente vicine allo spazio delle autofacce
test_n_recognized = int(sum(test_IsRecognized))

# frazione di facce sufficientemente vicine allo spazio delle autofacce,
#   rispetto al totale di immagini testate
test_p = test_n_recognized / hm.L_test

print("\n\n**** Testitamo le immagini di test ****\n")
print(f"Sono stati riconosciuti con successo {test_n_recognized} volti, su {hm.L_test} test")
print(f"La frazione di volti riconosciuti con successo è: {test_p}")

# numero di associazioni corrette effettuate tra faccia e individuo
test_n_correct_link = 0  # soglia 1,2 ok, associazione ok
# numero di associazioni errate
test_n_error = 0  # soglia 1,2 ok, associazione no
test_n_correct_waste = 0 # soglia 1 ok, soglia 2 no, associazione ok: scartiamo troppo
test_n_error_waste = 0  # soglia 1 ok, soglia 2 no, associazione no: vogliamo sia "grande"

# numero di immagini che superano entrambe le soglie di riconoscimeto
test_n_pass_pass = 0  # soglia 1,2 ok
# numero di potenziali associazioni falso positive
test_n_false_positive = 0  # soglia 1 no, soglia 2 ok
test_n_pass_no = 0  # soglia 1 ok, soglia 2 no
test_n_no_no = 0  # soglia 1,2 no

# numero di associazioni fuori dalla soglia 2
test_n_out = 0

i = 0
for link in test_subject_link:
    if test_IsRecognized[i]:
        if link[1] == 1:
            if link[0] == test_subject_expected[i]:
                test_n_correct_link = test_n_correct_link + 1
            else:
                test_n_error = test_n_error + 1
        else:
            test_n_out = test_n_out + 1 
            test_n_pass_no = test_n_pass_no + 1
    else:
        if link[1] == 1:
            test_n_false_positive = test_n_false_positive + 1
            if link[0] == test_subject_expected[i]:
                test_n_correct_waste = test_n_correct_waste + 1
            else:
                test_n_error_waste = test_n_error_waste + 1
        else:
            test_n_out = test_n_out + 1
            test_n_no_no = test_n_no_no + 1
            
    i = i + 1
test_n_pass_pass = test_n_correct_link + test_n_error
  
# frazione di facce sufficientemente vicine a una faccia di training 
#   correttamente associate, rispetto alle immagini riconosciute come facce
test_p_correspondance = test_n_correct_link / test_n_recognized

# frazione di facce correttamente associate, rispetto al numero immagini 
#   al di sotto delle soglie di riconosciemento
test_p_correct_pass = test_n_correct_link / test_n_pass_pass

test_mean_distance_facespace = sum(test_distance_face_space)/len(test_distance_face_space)
test_max_distance_facespace = max(test_distance_face_space)
test_min_distance_facespace = min(test_distance_face_space)

test_mean_distance_faceclass = sum(test_distance_class_face)/len(test_distance_class_face)
test_max_distance_faceclass = max(test_distance_class_face)
test_min_distance_faceclass = min(test_distance_class_face)

print(f"\nSono state determinate con successo {test_n_correct_link} associazioni faccia-individuo, su {test_n_recognized} immagini riconosciute come facce")
print(f"La frazione di volti correttamente associati, rispetto alle immagini riconosciute come facce è: {test_p_correspondance}")
print(f"\nSono state determinate con successo {test_n_correct_link} associazioni faccia-individuo, su {test_n_pass_pass} immagini riconosciute")
print(f"La frazione di volti correttamente associati, rispetto alle immagini riconosciute è: {test_p_correct_pass}")
print("\nSono stati inoltre trovati:")
print(f"  {test_n_false_positive} potenziali falsi positivi (su {test_n_error_waste} effettivi falsi positivi),")
print(f"  {test_n_error} errori nelle associazioni")
print(f"  {test_n_out} immagini fuori dalla soglia di vicinanza agli individui noti")
print(f"  {test_mean_distance_facespace} distanza media dallo spazio delle facce")
print(f"  {test_max_distance_facespace} distanza massima dallo spazio delle facce")
print(f"  {test_min_distance_facespace} distanza minima dallo spazio delle facce")
print(f"  {test_mean_distance_faceclass} distanza media dalle class face")
print(f"  {test_max_distance_faceclass} distanza massima dalle class face")
print(f"  {test_min_distance_faceclass} distanza minima dalle class face")


# -------------------- caso training -------------------- #
# testiamo le immagini usate nella fase di training

training_IsRecognized = np.zeros(hm.L)
training_distance_face_space = np.zeros(hm.L)
#training_subject_link = [-1]*hm.L
training_subject_link = np.full((hm.L,2), -1)
training_distance_class_face = np.zeros(hm.L)
training_subject_expected = np.zeros(hm.L)
for i in range(hm.L):
    training_subject_expected[i] = i // hm.n_faces_training

i = 0
for f_new in hm.f:
    training_IsRecognized[i], training_distance_face_space[i], projection_new = \
        testfun.face_recognition(f_new, theta)
    training_subject_link[i,0], training_subject_link[i,1], training_distance_class_face[i] = \
        testfun.which_subject(projection_new,theta_k)
        
    i = i + 1

training_n_recognized = int(sum(training_IsRecognized))
training_p = training_n_recognized / hm.L

print("\n\n**** Testiamo le immagini usate nella fase di training ****\n")
print(f"Sono stati riconosciuti con successo {training_n_recognized} volti su {hm.L} test")
print(f"La frazione di volti riconosciuti con successo è: {training_p}")

training_n_correct_link = 0
training_n_error = 0
training_n_correct_waste = 0
training_n_error_waste = 0

training_n_pass_pass = 0
training_n_false_positive = 0
training_n_pass_no = 0
training_n_no_no = 0

training_n_out = 0

i = 0
for link in training_subject_link:
    if training_IsRecognized[i]:
        if link[1] == 1:
            if link[0] == training_subject_expected[i]:
                training_n_correct_link = training_n_correct_link + 1
            else:
                training_n_error = training_n_error + 1
        else:
            training_n_out = training_n_out + 1 
            training_n_pass_no = training_n_pass_no + 1
    else:
        if link[1] == 1:
            training_n_false_positive = training_n_false_positive + 1
            if link[0] == training_subject_expected[i]:
                training_n_correct_waste = training_n_correct_waste + 1
            else:
                training_n_error_waste = training_n_error_waste + 1
        else:
            training_n_out = training_n_out + 1
            training_n_no_no = training_n_no_no + 1
            
    i = i + 1
training_n_pass_pass = training_n_correct_link + training_n_error
    
training_p_correspondance = training_n_correct_link / training_n_recognized
training_p_correct_pass = training_n_correct_link / training_n_pass_pass

print(f"\nSono state determinate con successo {training_n_correct_link} associazioni faccia-individuo, su {training_n_recognized} immagini riconosciute come facce")
print(f"La frazione di volti correttamente associati, rispetto alle immagini riconosciute come facce è: {training_p_correspondance}")
print(f"\nSono state determinate con successo {training_n_correct_link} associazioni faccia-individuo, su {training_n_pass_pass} immagini riconosciute")
print(f"La frazione di volti correttamente associati, rispetto alle immagini riconosciute è: {training_p_correct_pass}")
print("\nSono stati inoltre trovati:")
print(f"  {training_n_false_positive} potenziali falsi positivi (su {training_n_error_waste} effettivi falsi positivi),")
print(f"  {training_n_error} errori nelle associazioni")
print(f"  {training_n_out} immagini fuori dalla soglia di vicinanza agli individui noti")


# studiamo il caso in cui non tutti i soggetti del database non siano stati
#   usati nella fase di training
if not hm.AllTraining:
    f_unknown_test = np.reshape(hm.test_unknown_reduced, (hm.L_unknown_test,hm.mn))
