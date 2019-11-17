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
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense


# importing manually created libraries
import funclib
import neural

import time

# Feed Forward Neural Network





'''
Main Body of the Program
'''

if __name__ == '__main__':
    
    # Estimate how much time it took for program to work
    startTime = time.time()
    
    # Getting parameter file    
    paramFile = 'ParamFile.yaml'
    # getting NN configuration
    arch = neural.NetworkArchitecture(paramFile)
    NNType, NNArch, nLayers, nFeatures, nHidden, nOutput, epochs, alpha, lmbd, X_train, X_test, Y_train, Y_test, m, nInput = arch.CreateNetwork()
    
    if (nLayers == 2):
        print('Logistic Regression')
        '''
        Simplest Logistic regression
        '''

        activeFuncs = funclib.ActivationFuncs()
        costFuncs = funclib.CostFuncs()
        # load the data from the file
        filename = 'marks.txt'
        path = os.getcwd() + '/data/' + filename
        data = pd.read_csv(path, header = None)
        # X = feature values, all the columns except the last column
        #X = data.iloc[:, :-1]
        # y = target values, last column of the data frame
        #y = data.iloc[:, -1]
        # insert a column of 1's as the first entry in the feature
        # vector -- this is a little trick that allows us to treat
        # the bias as a trainable parameter *within* the weight matrix
        # rather than an entirely separate variable
        #X = np.c_[np.ones((X.shape[0])), X]
        #y = y[:, np.newaxis]
        theta = np.zeros((X_train.shape[1], 1))
        # apply gradient
        #alpha = 0.01 # 0.0001
        epochs1 = range(epochs)
        #epochs = range(1000) # 100
        Y_train.ravel()
        m = len(Y_train)
        costs = []
        for epoch in epochs1:
            Y_pred = np.dot(X_train, theta)
            A = activeFuncs.CallSigmoid(Y_pred)
            # cost function        
            J, dJ = costFuncs.CallLogistic(X_train, Y_train, A)
            # updating weights
            theta = theta - alpha * dJ
            # updating cost func history
            costs.append(J)
            

        '''
        neuralNet = neural.NeuralNetwork(NNType, NNArch, nLayers, nFeatures, nHidden, nOutput, epochs, alpha, lmbd, nInput)
        neuralNet.TrainNetwork(X_train, Y_train, m)
        '''
    elif (nLayers > 2):
        print('Neural Network')
        # passing configuration
        neuralNet = neural.NeuralNetwork(NNType, NNArch, nLayers, nFeatures, nHidden, nOutput, epochs, alpha, lmbd, nInput)
        modelParams, costs = neuralNet.TrainNetwork(X_train, Y_train, m)
    else:
        raise Exception('No. of Layers should be >= {}! Check Parameter File.'.format(nLayers))
    
    epochs1 = range(epochs)
    # Plotting results
    fig,ax = plt.subplots(figsize=(12,8))

    ax.set_ylabel('J(Theta)')
    ax.set_xlabel('Iterations')
    _ = ax.plot(epochs1, costs,'b.')
    
    # End time of the program
    endTime = time.time()
    print("-- Program finished at %s sec --" % (endTime - startTime))