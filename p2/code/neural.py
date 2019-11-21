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
from collections import Counter

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
        #display(Markdown())
        print((u'''
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
    
    #(5624, 31)
    #(5624, 2)
    #(22497, 31)
    #(22497, 2)
    # 
    
    
    '''
    To address these issues, Xavier and Bengio (2010) proposed the 
    “Xavier” initialization which considers the size of the network 
    (number of input and output units) while initializing weights. 
    This approach ensures that the weights stay within a reasonable 
    range of values by making them inversely proportional to the square 
    root of the number of units in the previous layer (referred to as fan-in).       
    '''
    '''
    To prevent the gradients of the network’s activations from vanishing or 
    exploding, we will stick to the following rules of thumb:

        The mean of the activations should be zero.
        The variance of the activations should stay the same across every layer.
    '''
    '''
    Used for tanh, sigmoid etc.
    '''
    def CallXavier(self, *args):
        n_l = args[0]
        n_next = args[1]
        xav = np.random.uniform(-np.sqrt(6.0 / (n_l + n_next)), np.sqrt(6.0 / (n_l + n_next)), size=(n_l, n_next))# * np.sqrt(6.0 / (n_l + n_next))
        return xav
    
    '''
    Used for ReLU
    '''
    def CallKaiming(self, *args):
        m = args[0]
        h = args[1]
        kaim = np.random.rand(m, h) * np.sqrt(2./m)
        return kaim#torch.randn()
    
    def InitParams(self, *args):
        NNArch = args[0]
        nInput = args[1]
        # biases and weights for hidden and output layers
        # dictionary to contain all parameters for each layer
        # (i.e. "W1", "b1", ..., "WL", "bL", except inpur one)
        modelParams = {}
        for l in range(1, self.nLayers):
            print(self.NNArch[l]["LSize"])
            if self.NNArch[l]['AF'] == 'sigmoid':
                modelParams['W' + str(l)] = self.CallXavier(self.NNArch[l-1]["LSize"],self.NNArch[l]["LSize"])
                modelParams['b' + str(l)] = np.random.randn(NNArch[l]["LSize"])
            elif self.NNArch[l]['AF'] == 'tanh':
                modelParams['W' + str(l)] = self.CallXavier(self.NNArch[l-1]["LSize"],self.NNArch[l]["LSize"])
                modelParams['b' + str(l)] = np.random.randn(NNArch[l]["LSize"])
            elif self.NNArch[l]['AF'] == 'softmax':
                modelParams['W' + str(l)] = self.CallXavier(self.NNArch[l-1]["LSize"],self.NNArch[l]["LSize"])
                modelParams['b' + str(l)] = np.random.randn(NNArch[l]["LSize"])
            elif self.NNArch[l]['AF'] == 'relu':
                modelParams['W' + str(l)] = self.CallKaiming(self.NNArch[l-1]["LSize"],self.NNArch[l]["LSize"])
                modelParams['b' + str(l)] = np.random.randn(NNArch[l]["LSize"])
            elif self.NNArch[l]['AF'] == 'elu':
                modelParams['W' + str(l)] = self.CallKaiming(self.NNArch[l-1]["LSize"],self.NNArch[l]["LSize"])
                modelParams['b' + str(l)] = np.random.randn(NNArch[l]["LSize"])
            else:
                modelParams['W' + str(l)] = self.CallXavier(self.NNArch[l-1]["LSize"],self.NNArch[l]["LSize"])
                modelParams['b' + str(l)] = np.random.randn(NNArch[l]["LSize"])
            # weights for each layer (except input one)
            #print(self.nInputNeurons, self.nHiddenNeurons)
            #print(self.NNArch[l-1]["LSize"])
            #modelParams['W' + str(l)] = np.random.randn(NNArch[l-1]["LSize"], NNArch[l]["LSize"]) #*\

            #self.CallXavier(NNArch[l-1]["LSize"], NNArch[l]["LSize"])#/ np.sqrt(NNArch[l-1]["LSize"])
            #modelParams['W' + str(l)] = np.random.randn(NNArch[l-1]["LSize"], NNArch[l]["LSize"]) *\
            #self.CallXavier(NNArch[l-1]["LSize"], NNArch[l]["LSize"])
            #print(np.shape(modelParams['W' + str(l)]))
            # biases for each layer (except input one)
            #modelParams['b' + str(l)] = np.zeros((self.NNArch[l]["LSize"], self.nOutputNeurons)) + 0.01
            
            #modelParams['b' + str(l)] = np.zeros((nInput, NNArch[l]["LSize"])) + 0.0001
            
            # shape should be of the amount of neurons per layer
            #modelParams['b' + str(l)] = np.zeros(NNArch[l]["LSize"])# + 0.1
            
            #print("W"+str(l),np.shape(modelParams['W' + str(l)]))
            #print(np.shape(modelParams['b' + str(l)]))
            #np.random.randn
        #print(modelParams)
        #sys.exit()
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
        elif self.NNArch[l]['AF'] == 'identity':
            A = funclib.ActivationFuncs().CallIdentity(Z)
        elif self.NNArch[l]['AF'] == 'elu':
            A = funclib.ActivationFuncs().CalleLU(Z)
        return A
    
    '''
    Feed Forward seems to return correct (!) shapes
    '''
    
    # Method to Feed Forward Propagation
    def DoFeedForward(self, *args):
        X = args[0]
        modelParams = args[1]
        A = {}
        Z = {}
        # Values for Input Layer
        A['0'] = X
        Z['0'] = X
        #print(X.shape)
        #print(X)
        # compute model for each layer and 
        # apply corresponding activation function
        for l in range(1, self.nLayers):
            #print("W"+str(l)+" is", modelParams['W' + str(l)])
            #print("b"+str(l)+" is", modelParams['b' + str(l)])
            # z for each layer
            Z[str(l)] = np.matmul(A[str(l-1)],modelParams['W' + str(l)]) + modelParams['b' + str(l)]
            #if np.isnan(Z[str(l)]):
            #print("Z is", Z[str(l)])
            #print(Z[str(l)])
            # applying corresponding activation function
            A[str(l)] = self.GetA(Z[str(l)], l)
            #if np.isnan(Z[str(l)]):
            #print("A is", A[str(l)])
            
            #print("Z"+str(l), Z[str(l)].shape)
            #print("A"+str(l), A[str(l)].shape)
            
            #print(A[str(l)])
            #if np.isnan(Z[str(l)]):
            #    print('NaN values spotted')
            #print('l={}: Z={}, A={}' .format(l, np.shape(Z[str(l)]), np.shape(A[str(l)])))


        # Z1 (5000, 10), batch size 5000
        # A1 (5000, 10)
        # Z2 (5000, 1)
        # A2 (5000, 1)

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
        elif self.NNArch[l]['AF'] == 'identity':
            dAF = funclib.ActivationFuncs().CallDIdentity(A)
        elif self.NNArch[l]['AF'] == 'elu':
            dAF = funclib.ActivationFuncs().CallDeLU(A)
            
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
        X = args[5]
        # errors for output layer
        delta = {}
        #delta[str(self.nLayers-1)] = A[str(self.nLayers-1)] - self.Y 
        # gradients of the cost function
        # (for each layer)
        dJ = {}
        #print(self.NNType)
        # Calculating gradients of the cost function for each layer
        # (going from last to first hidden layer)
        if self.NNType == "Classification":
            for l in reversed(range(1, self.nLayers)):
                #print(l)
                # calculating error for each layer
                if (l == self.nLayers - 1):
                    delta[str(l)] = A[str(l)] - Y
                    # gradients of output layer (+ regularization)
                    # W^{l} = A^{l-1} * delta^{l}
                    dJ['dW'+str(l)] = (1 / m) * np.matmul(A[str(l-1)].T, delta[str(l)]) \
                    + self.lambd * modelParams['W' + str(l)] / m
                    dJ['db'+str(l)] = (1 / m) * np.sum(delta[str(l)], axis=0)#, keepdims=True)                    
                else:
                    dAF = self.GetdAF(A[str(l)], l)
                    delta[str(l)] = np.multiply(np.matmul(delta[str(l+1)], modelParams['W' + str(l+1)].T), dAF)
                    # gradients of the hidden layer
                    # W^{l} = A^{l-1} * delta^{l}
                    dJ['dW'+str(l)] = (1 / m) * np.matmul(A[str(l-1)].T, delta[str(l)]) \
                    + self.lambd * modelParams['W' + str(l)] / m
                    dJ['db'+str(l)] = (1 / m) * np.sum(delta[str(l)], axis=0)#, keepdims=True)
                #print("dW"+str(l), dJ['dW'+str(l)].shape)
                #print("db"+str(l), dJ['db'+str(l)].shape)
                #print(dJ['dW'+str(l)])
                #if np.isnan(delta[str(l)].columns.values):
            #    print('NaN values spotted')
        elif self.NNType == 'Regression':
            for l in reversed(range(1, self.nLayers)):
                #print(l)
                # calculating error for each layer
                if (l == self.nLayers - 1):
                    #delta[str(l)] = 0
                    #print("Y", Y.shape)
                    # delta_L = (A_L - Y) * A_L <= notice matrix multiplication
                    delta[str(l)] = np.matmul((A[str(l)] - Y), A[str(l)])#, A[str(l)]) #self.lambd * (A[str(l)] - Y) * A[str(l)] / m
                    #print('delta'+str(l), delta[str(l)].shape)
                    #beta = beta - alpha * (X.T.dot(X.dot(beta)-y)/m)
                    # gradients of output layer (+ regularization)
                    # W^{l} = A^{l-1} * delta^{l}
                    dJ['dW'+str(l)] = (1 / m) * np.matmul(A[str(l-1)].T, delta[str(l)]) \
                    + self.lambd * modelParams['W' + str(l)] / m
                    dJ['db'+str(l)] = (1 / m) * np.sum(delta[str(l)], axis=0)#, keepdims=True)
                else:
                    #dAF = funclib.ActivationFuncs().CallDSigmoid(A[str(l)])
                    # Calling derivative of current activation function
                    dAF = self.GetdAF(A[str(l)], l)
                    delta[str(l)] = np.multiply(np.matmul(delta[str(l+1)], modelParams['W' + str(l+1)].T), dAF)
                    #print("delta"+str(l), delta[str(l)].shape)
                    # gradients of the hidden layer
                    # W^{l} = A^{l-1} * delta^{l}
                    dJ['dW'+str(l)] = (1 / m) * np.matmul(A[str(l-1)].T, delta[str(l)]) \
                    + self.lambd * modelParams['W' + str(l)] / m
                    dJ['db'+str(l)] = (1 / m) * np.sum(delta[str(l)], axis=0)#, keepdims=True)
                #print("dW"+str(l), dJ['dW'+str(l)].shape)
                #print("db"+str(l), dJ['db'+str(l)].shape)
            
        return dJ
    
    # Method to Update Weights 
    # (on each iteration)
    def UpdateWeights(self, *args):
        dJ = args[0]
        modelParams = args[1]
        m = args[2]
        for l in range(1, self.nLayers):
            modelParams['W' + str(l)] -= dJ['dW' + str(l)] * self.alpha/m# * np.sqrt(6.0 / (n_l + n_next)) # <= this one should be correct
            modelParams['b' + str(l)] -= dJ['db' + str(l)] * self.alpha/m# * np.sqrt(6.0 / (n_l + n_next)) # <= this one should be correct
            #print(modelParams['W' + str(l)])
        #print(modelParams)
        
        return modelParams
    
    # Train Neural Network using Gradient Descent
    def TrainNetworkGD(self, *args):
        X_train = args[0]
        Y_train = args[1]
        #print(Ytrain)
        m = args[2]
        
        # Initialising parameters
        modelParams = self.InitParams(self.NNArch, self.nInput)
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
                                     self.lambd, X_train)
                #print(J)
                # Back propagation - gradients
                dJ = self.DoBackPropagation(Y_train, A, Z, modelParams, m, X_train)
                #print(dJ)
                # updating weights
                modelParams = self.UpdateWeights(dJ, modelParams, m)
                # getting values of cost function at each epoch
                if(epoch % 1 == 0):
                    print('Cost after iteration# {:d}: {:f}'.format(epoch, J))
                costs.append(J)
            # returning set of optimal model parameters
            return modelParams, costs
        
        elif self.NNType == 'Regression':
            # Running Optimisation Algorithm
            for epoch in range(1, self.epochs+1, 1):
                # Propagating Forward
                A, Z = self.DoFeedForward(X_train, modelParams)
                # Calculating cost Function
                J = funclib.CostFuncs().CallNNMSE(Y_train,\
                                     A[str(self.nLayers-1)],\
                                     modelParams,\
                                     self.nLayers,\
                                     m,\
                                     self.lambd, X_train)
                #print(J)
                # Back propagation - gradients
                dJ = self.DoBackPropagation(Y_train, A, Z, modelParams, m, X_train)
                #print(dJ)
                # updating weights
                modelParams = self.UpdateWeights(dJ, modelParams, m)
                
                #print(modelParams)
                
                # getting values of cost function at each epoch
                if(epoch % 1 == 0):
                    print('Cost after iteration# {:d}: {:f}'.format(epoch, J))
                costs.append(J)
            # returning set of optimal model parameters
            return modelParams, costs
        else:
            Exception('It is neither Regression nor Classification task! Check Parameter File.')
    
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
        #counter = collections.Counter()
        cost = 0
        #paramsW, paramsb = [], []
        #params = {}
        #for l in range(1, self.nLayers):
            #paramsW.append(modelParams['W' + str(l)])
        #    params['W' + str(l)] = np.zeros((self.NNArch[l-1]["LSize"], self.NNArch[l]["LSize"]))
        #    params['b' + str(l)] = np.zeros((self.nInput, self.NNArch[l]["LSize"]))
        #sumValue1, sumValue2 = 0, 0
        nBatches = self.nInput // self.BatchSize
        
        #print(nBatches)
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
                # looping through all batches
                for j in range(nBatches):
                    # chosing the indexes for playing around
                    points = np.random.choice(indices, size=self.BatchSize, replace=False)
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
                    dJ = self.DoBackPropagation(Y_train_batch, A, Z, modelParams, m, X_train)
                    # updating weights
                    modelParams = self.UpdateWeights(dJ, modelParams, m)
                    cost += J/self.nInput
                if(epoch % 1 == 0):
                    print('Cost after iteration# {:d}: {:f}'.format(epoch, J))
                costs.append(J)
                
            return modelParams, costs
        
        elif self.NNType == 'Regression':
            # Running Optimisation Algorithm
            for epoch in range(1, self.epochs+1, 1):
               # looping through all batches
                for j in range(nBatches):
                    # chosing the indexes for playing around
                    points = np.random.choice(indices, size=self.BatchSize, replace=False)
                    X_train_batch = X_train[points]
                    Y_train_batch = Y_train[points]
                    #print(np.shape(X_train_batch))
                    # Propagating Forward
                    A, Z = self.DoFeedForward(X_train_batch, modelParams)
                    # Calculating cost Function
                    J = funclib.CostFuncs().CallNNMSE(Y_train_batch,\
                                         A[str(self.nLayers-1)],\
                                         modelParams,\
                                         self.nLayers,\
                                         m,\
                                         self.lambd, X_train_batch)
                                        # Back propagation - gradients
                    dJ = self.DoBackPropagation(Y_train_batch, A, Z, modelParams, m, X_train_batch)
                    # updating weights
                    modelParams = self.UpdateWeights(dJ, modelParams, m)
                    cost += J/self.nInput
                if(epoch % 1 == 0):
                    print('Cost after iteration# {:d}: {:f}'.format(epoch, J))
                costs.append(J)
            # returning set of optimal model parameters
            return modelParams, costs
        else:
            Exception('It is neither Regression nor Classification task! Check Parameter File.')
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
        #elif self.NNType == 'Regression':
         #   print('Regression has yet to be implemented')
            # returning set of optimal model parameters
         #   return None, None
        #else:
        #    Exception('It is neither Regression nor Classification task! Check Parameter File.')
            
    '''
    Function which will fit the test data
    '''
    def MakePrediction(self, *args):
        # making prediction for the data set
        X_test = args[0]
        
        #print("X_test is", X_test)
        
        modelParams = args[1]
        
        #print(modelParams)
        
        
        # making a prediction
        A, Z = self.DoFeedForward(X_test, modelParams)#self.feed_forward_out(X)
        
        #print("AL is", A[str(self.nLayers-1)])
        
        
        if self.NNType == 'Classification':
            # outputting the values which corresponds to the maximum probability
            return np.argmax(A[str(self.nLayers-1)], axis=1)
        elif self.NNType == 'Regression':
            return A[str(self.nLayers-1)]
        #print(np.shape(A))

            
            
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
    
    