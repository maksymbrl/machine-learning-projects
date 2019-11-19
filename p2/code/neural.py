#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:29:25 2019

@author: maksymb
"""

import os
import sys
import numpy as np
# for polynomial manipulation
import sympy as sp
# from sympy import *
import itertools as it
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
# to read parameter file
import yaml

import time

import collections

# For Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer

# One Hot Encoder from Keras
from keras.utils import to_categorical

# initialising pretty printing with sympy 
# for Latex characters and more
from IPython.display import display, Latex, Markdown
from sympy import * #init_printing
#from sympy.printing.latex import print_latex
#init_printing(use_latex='mathjax')
import funclib

    
'''
The class used for both classification and regression
depending on the cost function and the user's desire
'''
class NeuralNetwork:
    # constructor
    def __init__(self, *args):
        # type of neuralnetwork
        self.NNType = args[0]
        # Network Architecture
        self.NNArch = args[1]
        # Total Number of layers
        self.nLayers = args[2]
        # Neurons in input layer
        self.nInputNeurons = args[3]
        # Neurons in hidden layer
        self.nHiddenNeurons = args[4]
        # Neurons in output layer
        self.nOutputNeurons = args[5]
        # number of iterations for optimization algorithm
        self.epochs = args[6]
        # learning rate
        self.alpha = args[7]
        # regularisation (hyper) parameter
        self.lambd = args[8]
        
        # Data
        self.nInput = args[9]
        #self.nFeatures = 
        seed = args[10]
        # Batch Size
        self.BatchSize = args[11]
        #print(self.BatchSize)
        #print(type(self.BatchSize))
        # random seed, to make the same random number each time
        np.random.seed(seed)
        #self.Y = args[10]
        #self.m = args[11]
        # Only for printing purpose
        if (self.BatchSize == 0):
            algorithm = 'Gradient Descent'
        elif (self.BatchSize > 0):
            algorithm = 'Mini Batch Gradient Descent'
        display(Markdown(u'''
        =========================================== 
            Start {} Neural Network 
        =========================================== 
        No. of hidden layers:        {} 
        No. of input data:           {}
        No. of input neurons:        {} 
        No. of hidden neurons:       {} 
        No. of output neurons:       {} 
        Activ. Func in Hidden Layer: {} 
        Activ. Func in Output Layer: {} 
        No. of epochs to see:        {}
        Optimization Algorithm:      {}
        Learning Rate, \u03B1:            {} 
        Regularization param, \u03BB:     {} 
                      '''.format(self.NNType,
                                 self.nLayers-2,
                                 self.nInput,
                                 self.nInputNeurons, 
                                 self.nHiddenNeurons, 
                                 self.nOutputNeurons,
                                 self.NNArch[1]['AF'],
                                 self.NNArch[self.nLayers-1]['AF'],
                                 self.epochs,
                                 algorithm,
                                 self.alpha,
                                 self.lambd)))
    
    # 
    def InitParams(self, *args):
        NNArch = args[0]
        nInput = args[1]
        # biases and weights for hidden and output layers
        # dictionary to contain all parameters for each layer
        # (i.e. "W1", "b1", ..., "WL", "bL", except inpur one)
        modelParams = {}
        for l in range(1, self.nLayers):
            # weights for each layer (except input one)
            #print(self.nInputNeurons, self.nHiddenNeurons)
            #print(self.NNArch[l-1]["LSize"])
            modelParams['W' + str(l)] = np.random.randn(NNArch[l-1]["LSize"], NNArch[l]["LSize"]) / np.sqrt(NNArch[l-1]["LSize"])
            #print(np.shape(modelParams['W' + str(l)]))
            # biases for each layer (except input one)
            #modelParams['b' + str(l)] = np.zeros((self.NNArch[l]["LSize"], self.nOutputNeurons)) + 0.01
            modelParams['b' + str(l)] = np.zeros((nInput, 
                        NNArch[l]["LSize"])) + 0.0001
            #print(np.shape(modelParams['b' + str(l)]))
            #np.random.randn
        #print(modelParams)
        
        return modelParams
    
    # Getting output for each layer
    def GetA(self, *args):
        Z = args[0]
        l = args[1]
        # (['sigmoid', 'tanh', 'relu', 'softmax'])
        #print('AF is ' + self.NNArch[l]['AF'])
        if self.NNArch[l]['AF'] == 'sigmoid':
            A = funclib.ActivationFuncs().CallSigmoid(Z)
        elif self.NNArch[l]['AF'] == 'relu':
            A = funclib.ActivationFuncs().CallReLU(Z)
        elif self.NNArch[l]['AF'] == 'tanh':
            A = funclib.ActivationFuncs().CallTanh(Z)
        elif self.NNArch[l]['AF'] == 'softmax':
            A = funclib.ActivationFuncs().CallSoftmax(Z)
        return A
    
    # Method to Feed Forward Propagation
    def DoFeedForward(self, *args):
        X = args[0]
        modelParams = args[1]
        A = {}
        Z = {}
        # Values for Input Layer
        A['0'] = X
        Z['0'] = X
        #print(X)
        # compute model for each layer and 
        # apply corresponding activation function
        for l in range(1, self.nLayers):
            # z for each layer
            #print(type(np.shape(modelParams['W' + str(l)])))
            #print('W{} shape is {}'.format(l,np.shape(modelParams['W' + str(l)])))
            #print(modelParams['W' + str(l)])
            #print('A{} shape is {}'.format(l,np.shape(A[str(l-1)])))
            #print(A[str(l-1)])
            #print(type(np.shape(A[str(l-1)])))
            #print(type(modelParams['b' + str(l)]))
            Z[str(l)] = np.matmul(A[str(l-1)], modelParams['W' + str(l)]) + modelParams['b' + str(l)]
            # applying corresponding activation function
            A[str(l)] = self.GetA(Z[str(l)], l)
            #if np.isnan(Z[str(l)]):
            #    print('NaN values spotted')
            #print('l={}: Z={}, A={}' .format(l, np.shape(Z[str(l)]), np.shape(A[str(l)])))
        # returning dictionary of outputs
        return A, Z
    
    # Method which decides, whioch 
    # derivative of cost function to use
    def GetdAF(self, *args):
        A = args[0]
        l = args[1]
        # (['sigmoid', 'tanh', 'relu', 'softmax'])
        #print('dAF is ' + self.NNArch[l]['AF'])
        if self.NNArch[l]['AF'] == 'sigmoid':
            dAF = funclib.ActivationFuncs().CallDSigmoid(A)
        elif self.NNArch[l]['AF'] == 'relu':
            dAF = funclib.ActivationFuncs().CallDReLU(A)
        elif self.NNArch[l]['AF'] == 'tanh':
            dAF = funclib.ActivationFuncs().CallDTanh(A)
        elif self.NNArch[l]['AF'] == 'softmax':
            dAF = funclib.ActivationFuncs().CallDSoftmax(A)
            
        return dAF
    
    # Method to do Back Propagation
    # (to calculate gradients of J)
    def DoBackPropagation(self, *args):
        # getting values for each layer
        #A, Z = self.DoFeedForward()
        Y = args[0]
        A = args[1]
        Z = args[2]
        modelParams = args[3]
        m = args[4]
        # errors for output layer
        delta = {}
        #delta[str(self.nLayers-1)] = A[str(self.nLayers-1)] - self.Y 
        # gradients of the cost function
        # (for each layer)
        dJ = {}
        #print(self.NNType)
        # Calculating gradients of the cost function for each layer
        # (going from last to first hidden layer)
        for l in reversed(range(1, self.nLayers)):
            #print(l)
            # calculating error for each layer
            if (l == self.nLayers - 1):
                delta[str(l)] = A[str(l)] - Y
                # gradients of output layer (+ regularization)
                # W^{l} = A^{l-1} * delta^{l}
                dJ['dW'+str(l)] = (1 / m) * np.matmul(A[str(l-1)].T, delta[str(l)]) \
                + self.lambd * modelParams['W' + str(l)] / m
                dJ['db'+str(l)] = (1 / m) * np.sum(delta[str(l)], axis=0, keepdims=True)
                #print(dJ['dW'+str(l)].shape)
            else:
                #dAF = funclib.ActivationFuncs().CallDSigmoid(A[str(l)])
                
                dAF = self.GetdAF(A[str(l)], l)
                delta[str(l)] = np.multiply(np.matmul(delta[str(l+1)], modelParams['W' + str(l+1)].T), dAF)
                # gradients of the hidden layer
                # W^{l} = A^{l-1} * delta^{l}
                dJ['dW'+str(l)] = (1 / m) * np.matmul(A[str(l-1)].T, delta[str(l)]) \
                + self.lambd * modelParams['W' + str(l)] / m
                dJ['db'+str(l)] = (1 / m) * np.sum(delta[str(l)], axis=0, keepdims=True)
            #print(dJ['dW'+str(l)])
            #if np.isnan(delta[str(l)].columns.values):
            #    print('NaN values spotted')
            
        return dJ
    
    # Method to Update Weights 
    # (on each iteration)
    def UpdateWeights(self, *args):
        dJ = args[0]
        modelParams = args[1]
        m = args[2]
        for l in range(1, self.nLayers):
            modelParams['W' + str(l)] -= dJ['dW' + str(l)] * self.alpha 
            modelParams['b' + str(l)] -= dJ['db' + str(l)] * self.alpha
            #print(modelParams['W' + str(l)])
            
        return modelParams
    
    # Train Neural Network using Gradient Descent
    def TrainNetworkGD(self, *args):
        X_train = args[0]
        Y_train = args[1]
        #print(Ytrain)
        m = args[2]
        
        # Initialising parameters
        modelParams = self.InitParams(self.NNArch, self.nInput)
        #print("self.nInput", Xtrain.shape[0])
        costs =  []
        if self.NNType == 'Classification':
            # Running Optimisation Algorithm
            for epoch in range(1, self.epochs+1, 1):
                # Propagating Forward
                A, Z = self.DoFeedForward(X_train, modelParams)
                # Calculating cost Function
                J = funclib.CostFuncs().CallNNLogistic(Y_train,\
                                     A[str(self.nLayers-1)],\
                                     modelParams,\
                                     self.nLayers,\
                                     m,\
                                     self.lambd)
                #print(J)
                # Back propagation - gradients
                dJ = self.DoBackPropagation(Y_train, A, Z, modelParams, m)
                #print(dJ)
                # updating weights
                modelParams = self.UpdateWeights(dJ, modelParams, m)
                # getting values of cost function at each epoch
                if(epoch % 100 == 0):
                    print('Cost after iteration# {:d}: {:f}'.format(epoch, J))
                costs.append(J)
            # returning set of optimal model parameters
            return modelParams, costs
        
        elif self.NNType == 'Regression':
            print('Regression has yet to be implemented')
            # returning set of optimal model parameters
            return None, None
        else:
            Exception('It is neither Regression nor Classification task! Check Parameter File.')
            
            
   # Creating a list of mini-batches
    def CreateMiniBatches(self, *args):
        X = args[0]
        Y = args[1]
        modelParams = args[2]
        batchSize = args[3]
        
        miniBatches = []
        #dataParams  = {}
        miniParams  = []
        data = np.hstack((X, Y))
        # shuffling data set randomly
        np.random.shuffle(data) 
        nMiniBatches = data.shape[0] // batchSize 
        i = 0
        for i in range(nMiniBatches + 1): 
            miniBatch = data[i * batchSize:(i + 1)*batchSize, :] 
            X_mini = miniBatch[:, :-1] 
            Y_mini = miniBatch[:, -1].reshape((-1, 1)) 
            miniBatches.append((X_mini, Y_mini)) 
        if data.shape[0] % batchSize != 0: 
            miniBatch = data[i * batchSize:data.shape[0]] 
            X_mini = miniBatch[:, :-1] 
            Y_mini = miniBatch[:, -1].reshape((-1, 1)) 
            miniBatches.append((X_mini, Y_mini))
        # Shuffling initialised model parameters
        for l in range(1, self.nLayers):
            dataParams = np.hstack((modelParams['W' + str(l)], modelParams['b' + str(l)]))
            np.random.shuffle(dataParams[l])
            for i in range(nMiniBatches + 1): 
                miniParam = dataParams[i * batchSize:(i + 1)*batchSize, :] 
                modelParams['W' + str(l)] = miniParam[:, :-1] 
                modelParams['b' + str(l)] = miniParam[:, -1].reshape((-1, 2)) 
                miniBatches.append((X_mini, Y_mini)) 
            if dataParams.shape[0] % batchSize != 0: 
                miniParam = dataParams[i * batchSize:dataParams.shape[0]] 
                modelParams['W' + str(l)] = miniParam[:, :-1] 
                modelParams['b' + str(l)] = miniParam[:, -1].reshape((-1, 2)) 
                miniParams.append((X_mini, Y_mini))
        
        return miniBatches 
    
    # Train Neural Network using Mini Batch Stochastic Gradient Descent
    def TrainNetworkMBGD(self, *args):
        X_train = args[0]
        Y_train = args[1]
        #print(Ytrain)
        m = self.BatchSize
        #print(self.CreateMiniBatches(Xtrain, Ytrain, self.BatchSize))
        indices = np.arange(self.nInput)
        costs =  []
        # sum the values with same keys 
        counter = collections.Counter()
        cost = 0
        paramsW, paramsb = [], []
        sumValue1, sumValue2 = 0, 0
        nBatches = self.nInput // self.BatchSize
        
        print(nBatches)
        # getting all the indecis from inputs
        indices = np.arange(self.nInput)
        # Initialising parameters - ensuring 
        '''
        Let theta = model parameters and max_iters = number of epochs.

        for itr = 1, 2, 3, …, max_iters:
              for mini_batch (X_mini, y_mini):
        
        Forward Pass on the batch X_mini:
        Make predictions on the mini-batch
        Compute error in predictions (J(theta)) with the current values of the parameters
        Backward Pass:
        Compute gradient(theta) = partial derivative of J(theta) w.r.t. theta
        Update parameters:
        theta = theta – learning_rate*gradient(theta)
        '''
        # that we starting at the spot in the parameter space
        modelParams = self.InitParams(self.NNArch, self.BatchSize)
        if self.NNType == 'Classification':
            # Running Optimisation Algorithm
            for epoch in range(1, self.epochs+1, 1):
                # chosing the indexes for playing around
                points = np.random.choice(indices, size=self.BatchSize, replace=False)
                # looping through all batches
                for j in range(nBatches):
                    X_train_batch = X_train[points]
                    Y_train_batch = Y_train[points]
                    #print(np.shape(X_train_batch))
                    # Propagating Forward
                    A, Z = self.DoFeedForward(X_train_batch, modelParams)
                    # Calculating cost Function
                    J = funclib.CostFuncs().CallNNLogistic(Y_train_batch,\
                                         A[str(self.nLayers-1)],\
                                         modelParams,\
                                         self.nLayers,\
                                         m,\
                                         self.lambd)
                                        # Back propagation - gradients
                    dJ = self.DoBackPropagation(Y_train_batch, A, Z, modelParams, m)
                    # updating weights
                    modelParams = self.UpdateWeights(dJ, modelParams, m)
                    # iterating key value pair
                    #for l in range(1, self.nLayers):
                    #    paramsW.append(modelParams['W' + str(l)])
                    #    paramsb.append(modelParams['b' + str(l)])
                    # sum up every lth element of the list
                    #for l in range(1, self.nLayers):
                    #    params += paramsW[::l]
                    #    print(params)
                        #for key ,value in modelParams.items(): 
                        #    if (value-modelParams['W' + str(l)]).all() in value.keys(): 
                                # Adding value of sharpness to sum 
                        #        paramsW[l] += value['W' + str(l)]  
                    #for l in range(1, self.nLayers):
                    #    paramsW[nBatches] = modelParams['W' + str(l)]
                    #    paramsb[nBatches] = modelParams['b' + str(l)]
                    #print(paramsW)
                        #params['W' + str(l)] += modelParams['W' + str(l)]/self.nInput
                        #params['b' + str(l)] += modelParams['b' + str(l)]/self.nInput
                    #for l in range(1, self.nLayers):
                    #    sumValue1 = np.mean((d['W' + str(l)] for d in paramsW.values() if d), axis=0) 
                    #    sumValue2 = np.mean((d['b' + str(l)] for d in paramsb.values() if d), axis=0)
                    #    print(sumValue1)
                        #paramsW[nBatches] = modelParams['W' + str(l)]
                        #paramsb[nBatches] = modelParams['b' + str(l)]
                    # sumValue1 = sum(d['sharpness'] for d in weapons.values() if d)
                    cost += J/self.nInput
                if(epoch % 100 == 0):
                    print('Cost after iteration# {:d}: {:f}'.format(epoch, J))
                costs.append(J)
                    #print("modelParams", modelParams)
            '''
                    # creating mini batches to run on
                    miniBatches = self.CreateMiniBatches(Xtrain, Ytrain, modelParams, self.BatchSize)
                    #print(J)

                    #print(dJ)
                    # updating weights
                    modelParams = self.UpdateWeights(dJ, modelParams, m)
                    # getting values of cost function at each epoch
                    if(epoch % 100 == 0):
                        print('Cost after iteration# {:d}: {:f}'.format(epoch, J))
                    costs.append(J)
            # returning set of optimal model parameters
            return modelParams, costs
            '''
            return costs#, modelParams
        elif self.NNType == 'Regression':
            print('Regression has yet to be implemented')
            # returning set of optimal model parameters
            return None, None
        else:
            Exception('It is neither Regression nor Classification task! Check Parameter File.')
            
            
    # function to perform mini-batch gradient descent 
#def gradientDescent(X, y, learning_rate = 0.001, batch_size = 32): 
#    theta = np.zeros((X.shape[1], 1)) 
#    error_list = [] 
#    max_iters = 3
#    for itr in range(max_iters): 
#        mini_batches = create_mini_batches(X, y, batch_size) 
#        for mini_batch in mini_batches: 
#            X_mini, y_mini = mini_batch 
#            theta = theta - learning_rate * gradient(X_mini, y_mini, theta) 
#            error_list.append(cost(X_mini, y_mini, theta)) 
  
#    return theta, error_list 
    
    