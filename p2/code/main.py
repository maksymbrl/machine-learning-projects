#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:21:05 2019

@author: maksymb
"""

# importing libraries
import os
import sys
import math as mt
import numpy as np
# for polynomial manipulation
import sympy as sp
# from sympy import *
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd

# to read parameter file
import yaml

# Scikitlearn imports to check results
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from numpy import argmax

# We'll need some metrics to evaluate our models
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor

import keras
# stochastic gradient descent
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense


# importing manually created libraries
import funclib
import neural
import regression
import data_processing

import time

# Feed Forward Neural Network
def CallKerasModel(NNArch, nLayers):
    classifier = Sequential()
    print("nHidden", nHidden)

    for layer in range(1, nLayers):
        if layer == 0:
            classifier.add(Dense(nHidden, activation=NNArch[layer]['AF'], \
                                 kernel_initializer='random_normal', input_dim=nFeatures))
        elif layer == nLayers-1:
            classifier.add(Dense(nOutput, activation=NNArch[layer]['AF'], \
                                 kernel_initializer='random_normal'))
        else:
            classifier.add(Dense(nHidden, activation=NNArch[layer]['AF'], \
                                 kernel_initializer='random_normal'))
    return classifier

'''
Main Body of the Program
'''

if __name__ == '__main__':
    
    # Estimate how much time it took for program to work
    startTime = time.time()
    
    # Getting parameter file    
    paramFile = 'ParamFile.yaml'
    # getting NN configuration - a lot of parameters to trace T_T
    dataProc = data_processing.NetworkArchitecture(paramFile)
    
    with open(paramFile) as f:
        paramData = yaml.load(f, Loader = yaml.FullLoader)
        
    NNType = paramData['type']
    
    if (NNType == 'Classification'):
        NNType, NNArch, nLayers, nFeatures, \
        nHidden, nOutput, epochs, alpha, lmbd, \
        X_train, X_test, Y_train, Y_test, Y_train_onehot, Y_test_onehot,\
        m, nInput, seed, onehotencoder,\
        BatchSize, Optimization = dataProc.CreateNetwork()
    elif (NNType == 'Regression'):
        print('Regression')
    
    #print("onehotencoder", onehotencoder)
    
    #print("nFeatures", nFeatures)
    '''
    So, If the total number of layers = 2, we can switch to Simple Logistic regression.
    However, once the number of layers is > 2, we are going to use Neural Network.
    The number of hidden layers can be specified in the parameter file above 
    (this is done for simplicity). In principle, the Neural network without hidden layers,
    should produc ethe same results as logistic regression (at least, i think so).
    '''
    if NNType == 'Regression':
        print('Linear Regression')

    elif NNType == 'Classification':
        if (nLayers == 2):
        # checking whether we have Logistic Regression or Linear Regresion
            '''
            Simplest Logistic regression
            '''
            print('''
                  Logistic Regression Via Manual Coding
                  ''')
        
            activeFuncs = funclib.ActivationFuncs()
            costFuncs = funclib.CostFuncs()
            theta = np.zeros((X_train.shape[1], 1))
            epochs1 = range(epochs)
            #if BatchSize == 0:
            m = len(Y_train)
            costs = []
            for epoch in epochs1:
                Y_pred = np.dot(X_train, theta)
                A = activeFuncs.CallSigmoid(Y_pred)#CallSigmoid(Y_pred)
                # cost function        
                J, dJ = costFuncs.CallLogistic(X_train, Y_train_onehot, A)
                # Adding regularisation
                J = J + lmbd / (2*m) * np.sum(theta**2)
                dJ = dJ + lmbd * theta / m
                # updating weights
                theta = theta - alpha * dJ #+ lmbd * theta/m
                # updating cost func history
                costs.append(J)
                # getting values of cost function at each epoch
                if(epoch % 100 == 0):
                    print('Cost after iteration# {:d}: {:f}'.format(epoch, J))
            
            # Using Logistic regression
            #logReg = LogisticRegression(n_jobs=-1, random_state=1).fit(X_train, Y_train)#, solver='lbfgs', multi_class='multinomial').fit(X, y)
            #Y_pred_scikit = logReg.predict(X_test)
            #classifier2 = LogisticRegression(n_jobs=-1, random_state=1, solver='lbfgs')
            #classifier2.fit( X_train, Y_train )
            #y_pred = classifier2.predict( X_test )
            
            #cm = confusion_matrix( Y_test, Y_pred )
            #print("Accuracy on Test Set for LogReg = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))
            #scoresLR = cross_val_score( classifier2, X_train, Y_train, cv=10)
            #print("Mean LogReg CrossVal Accuracy on Train Set %.2f, with std=%.2f" % (scoresLR.mean(), scoresLR.std() ))
            print('''
                  Logistic Regression Via Scikit Learn
                  ''')
            #lambdas=np.logspace(-5,7,13)
            #parameters = [{'C': 1./lambdas, "solver":["lbfgs"]}]#*len(parameters)}]
            #scoring = ['accuracy', 'roc_auc']
            #logReg = LogisticRegression(n_jobs=-1, random_state=1)
            # Looking for the best Hyper Parameters, and then applying Regression
            #gridSearch = GridSearchCV(logReg, parameters, cv=5, scoring=scoring, refit='roc_auc')
            
            # Fit with bst parameters
            #gridSearch.fit(X_train, Y_train.ravel())
            '''
            m = np.size(Y_train)
            neuralNet = neural.NeuralNetwork(NNType, NNArch, \
                                             nLayers, nFeatures, \
                                             nHidden, nOutput, \
                                             epochs, alpha, \
                                             lmbd, nInput, seed)
            modelParams, costs = neuralNet.TrainNetwork(X_train, Y_train, m)
            '''
        
        elif (nLayers > 2):
            '''
            Switching to Neural Network
            '''
            
            m = np.size(Y_train)
            print('Neural Network')
            # passing configuration
            neuralNet = neural.NeuralNetwork(NNType, NNArch, \
                                             nLayers, nFeatures, \
                                             nHidden, nOutput, \
                                             epochs, alpha, \
                                             lmbd, nInput, seed, BatchSize)
            #print(type(BatchSize))
            if (BatchSize==0):
                print("Gradient Descent")
                modelParams, costs = neuralNet.TrainNetworkGD(X_train, Y_train_onehot, m)
            elif (BatchSize > 0):
                print("Mini Batches")
                costs = neuralNet.TrainNetworkMBGD(X_train, Y_train_onehot, m)#TrainNetworkMBGD(X_train, Y_train_onehot, m)
            
            #modelParams, 
            print(nHidden)
            # Classify using sklearn
            clf = MLPClassifier(solver="lbfgs", alpha=alpha, hidden_layer_sizes=nHidden)
            clf.fit(X_train, Y_train)
            yTrue, yPred = Y_test, clf.predict(X_test)
            print(classification_report(yTrue, yPred))
            print("Roc auc: ", roc_auc_score(yTrue,yPred))
            
            print('''
                  Initialising Keras
                  ''')
            # decoding from keras
            #print(np.shape(Y_train))
            #Y_train = np.argmax(Y_train, axis=1)#.reshape(1,-1)
            #print(np.shape(Y_train))
            
            classifier = CallKerasModel(NNArch, nLayers)
            
            '''
            To optimize our neural network we use Adam. Adam stands for Adaptive 
            moment estimation. Adam is a combination of RMSProp + Momentum.
            '''
            #print(np.shape(Y_train))
            #Y_train = Y_train.reshape(1,-1)
            #decoded = Y_train.dot(onehotencoder.active_features_).astype(int)
            # invert the one hot encoded data
            #inverted = onehotencoder.inverse_transform([argmax(Y_train[:, :])])
            #print(Y_train)
            #Y_train = Y_train.reshape(-1,1)
            #print(inverted)
            if BatchSize == 0:
                BatchSize = 32
            # Stochatic gradient descent
            sgd = SGD(lr=alpha)
            classifier.compile(optimizer = sgd, loss='binary_crossentropy', metrics =['accuracy'])
            #Fitting the data to the training dataset
            classifier.fit(X_train, Y_train_onehot, batch_size=BatchSize, epochs = epochs)
            
            
            #def build_model(hidden_layer_sizes):
            #  model = Sequential()
            
            #  model.add(Dense(hidden_layer_sizes[0], input_dim=2))
            #  model.add(Activation('tanh'))
            
            #  for layer_size in hidden_layer_sizes[1:]:
            #    model.add(Dense(layer_size))
            #    model.add(Activation('tanh'))
            
            #  model.add(Dense(1))
            #  model.add(Activation('sigmoid'))
            
            #  return model
            
            #def build_model(hidden_layer_sizes):
            #  model = Sequential()
            
            #  model.add(Dense(hidden_layer_sizes[0], input_dim=2))
            #  model.add(Activation('tanh'))
            
            #  for layer_size in hidden_layer_sizes[1:]:
            #    model.add(Dense(layer_size))
            #    model.add(Activation('tanh'))
            
            #  model.add(Dense(1))
            #  model.add(Activation('sigmoid'))
            
            #  return model
            
            #max_epochs = 500
            #my_logger = MyLogger(n=50)
            #h = model.fit(train_x, train_y, batch_size=32, epochs=max_epochs, verbose=0, callbacks=[my_logger])
            
            np.set_printoptions(precision=4, suppress=True)
            eval_results = classifier.evaluate(X_test, Y_test_onehot, verbose=0) 
            print("\nLoss, accuracy on test data: ")
            print("%0.4f %0.2f%%" % (eval_results[0], eval_results[1]*100))
            
        else:
            '''
            Raise an exception, if number of layers is smaller than 2. It shouldn't be the case,
            because in param file I am specifying number of hidden layers and not the total layers.
            Then I add 2 to that number in the code. But better safe than sorry :) 
            '''
            raise Exception('No. of Layers should be >= {}! Check Parameter File.'.format(nLayers))
        
    #epochs1 = range(epochs)
    # Plotting results
    #fig,ax = plt.subplots(figsize=(12,8))

    #ax.set_ylabel('J(Theta)')
    #ax.set_xlabel('Iterations')
    #_ = ax.plot(epochs1, costs,'b.')
    
    # End time of the program
    endTime = time.time()
    print("-- Program finished at %s sec --" % (endTime - startTime))