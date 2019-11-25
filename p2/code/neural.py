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
# for nice progress bar
from tqdm import tqdm_notebook

# For Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, mean_squared_error, log_loss

# One Hot Encoder from Keras
from keras.utils import to_categorical
import tensorflow as tf
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
        self.optimization = args[12]
        # Only for printing purpose
        #if (self.BatchSize == 0):
        #    algorithm = 'Gradient Descent'
        #elif (self.BatchSize > 0):
        #    algorithm = 'Mini Batch Gradient Descent'
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
                                 self.optimization,
                                 self.alpha,
                                 self.lambd)))
    
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
        #xav = np.random.uniform(-1, 1, size=(n_l, n_next)) * np.sqrt(6.0 / (n_l + n_next))
        xav = np.random.randn(n_l, n_next)*np.sqrt(2/(n_l+n_next))
        return xav
    
    '''
    Used for ReLU
    '''
    def CallKaiming(self, *args):
        m = args[0]
        h = args[1]
        kaim = np.random.rand(m, h) * np.sqrt(2./m)
        return kaim#torch.randn()
    
    '''
    Initialising weights' for training. I am using Xavier and He 
    (here I call it Kaiming from Kaiming He <= you got the idea :),
    but, for some reason, it doesn't help much for Linear Regression,
    so later I am trying to do batch Normalisation.
    '''
    def InitParams(self, *args):
        NNArch = args[0]
        nInput = args[1]
        # biases and weights for hidden and output layers
        # dictionary to contain all parameters for each layer
        # (i.e. "W1", "b1", ..., "WL", "bL", except inpur one)
        self.modelParams = {}
        self.update_params={}

        for l in range(1, self.nLayers):
            #print(self.NNArch[l]["LSize"])
            if self.NNArch[l]['AF'] == 'sigmoid':
                self.modelParams['W' + str(l)] = self.CallXavier(self.NNArch[l-1]["LSize"],self.NNArch[l]["LSize"])
                self.modelParams['b' + str(l)] = np.random.randn(NNArch[l]["LSize"]) #+ 1
            elif self.NNArch[l]['AF'] == 'tanh':
                self.modelParams['W' + str(l)] = self.CallXavier(self.NNArch[l-1]["LSize"],self.NNArch[l]["LSize"])
                self.modelParams['b' + str(l)] = np.random.randn(NNArch[l]["LSize"]) #+ 1
            elif self.NNArch[l]['AF'] == 'softmax':
                self.modelParams['W' + str(l)] = self.CallXavier(self.NNArch[l-1]["LSize"],self.NNArch[l]["LSize"])
                self.modelParams['b' + str(l)] = np.random.randn(NNArch[l]["LSize"]) #+ 1
            elif self.NNArch[l]['AF'] == 'relu':
                self.modelParams['W' + str(l)] = self.CallKaiming(self.NNArch[l-1]["LSize"],self.NNArch[l]["LSize"])
                self.modelParams['b' + str(l)] = np.random.randn(NNArch[l]["LSize"]) #+ 1
            elif self.NNArch[l]['AF'] == 'elu':
                self.modelParams['W' + str(l)] = self.CallKaiming(self.NNArch[l-1]["LSize"],self.NNArch[l]["LSize"])
                self.modelParams['b' + str(l)] = np.random.randn(NNArch[l]["LSize"]) #+ 1
            else:
                self.modelParams['W' + str(l)] = self.CallKaiming(self.NNArch[l-1]["LSize"],self.NNArch[l]["LSize"])
                self.modelParams['b' + str(l)] = np.random.randn(NNArch[l]["LSize"]) #+ 1
            
            self.update_params["v_w"+str(l)]=0
            self.update_params["v_b"+str(l)]=0
        #return self.modelParams
    

    
    # Getting output for each layer
    def GetA(self, *args):
        Z = args[0]
        l = args[1]
        # (['sigmoid', 'tanh', 'relu', 'softmax'])
        #print('AF is ' + self.NNArch[l]['AF'])
        if self.NNArch[l]['AF'] == 'sigmoid':
            A = funclib.ActivationFuncs().CallSigmoid(Z)
        elif self.NNArch[l]['AF'] == 'relu':
            # manual version:
            A = funclib.ActivationFuncs().CallReLU(Z)
            #print(A)
            # tensor flow version:
            #A = tf.nn.relu(Z)
            #print(A)
        elif self.NNArch[l]['AF'] == 'tanh':
            A = funclib.ActivationFuncs().CallTanh(Z)
        elif self.NNArch[l]['AF'] == 'softmax':
            A = funclib.ActivationFuncs().CallSoftmax(Z)
        elif self.NNArch[l]['AF'] == 'linear':
            A = funclib.ActivationFuncs().CallIdentity(Z)
        elif self.NNArch[l]['AF'] == 'elu':
            A = funclib.ActivationFuncs().CalleLU(Z)
        return A
    
    '''
    Doing BAtch Normalisation - in case of simple gradient 
    descent the batch size is the entire data set.
    This method is used during Feed Forward, for back propagation 
    there is another one below!
    '''
    '''
    def DoBatchNormalisation(self, *args):
        X = args[0] # <= these are not actually X, but Z values 
        gamma = args[1]
        beta = args[2]
        # parameters
        eps = 1e-5
        
        #print(X.shape)
        
        # checking for matrix to be of the shape 2
        #if X.shape == 2:
        # mean of the mini-batch
        batchMean = np.mean(X, axis=0)
        # variance of the mini-batch
        batchVar = np.var(X, axis=0)
        # normalising
        X_norm = (X - batchMean) * 1.0 / np.sqrt(batchVar + eps)
        # scaled output
        X_out = gamma * X_norm + beta
        # people are usually storing stuff in cache so I will do the same thing
        cache = (X, X_norm, batchMean, batchVar, gamma, beta)

        return X_out, cache, batchMean, batchVar
        #else:
        #    print("Batch Normalisation: Check your X shape!")
        #    sys.exit()
    '''
            
    '''
    This one to implement Batch Normalisationwith Back Propagation
    '''
    '''
    def DoDBatchNormalisation(self, *args):
        dout = args[0]
        cache = args[1]
        eps = 1e-5
        # getting values from cache
        X, X_norm, mu, var, gamma, beta = cache
    
        N, D = X.shape
    
        X_mu = X - mu
        std_inv = 1. / np.sqrt(var + eps)
    
        dX_norm = dout * gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)
    
        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
        dgamma = np.sum(dout * X_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
    
        return dX, dgamma, dbeta
    '''
    
    '''
    Feed Forward seems to return correct (!) shapes
    '''
    
    # Method to Feed Forward Propagation
    def DoFeedForward(self, *args):
        X = args[0] # <= it will be of batch size (for Mini-Batch GD)
        #self.modelParams = args[1]
        # parameters for batch normalisation
        gamma = 0.01
        beta  = 0.01
        A = {}
        Z = {}
        # Values for Input Layer
        A['0'] = X
        Z['0'] = X
        
        #cache['0'] = 0

        # compute model for each layer and 
        # apply corresponding activation function
        for l in range(1, self.nLayers):
            #print("W"+str(l)+" is", np.shape(modelParams['W' + str(l)]))
            #print("b"+str(l)+" is", modelParams['b' + str(l)])
            # z for each layer
            Z[str(l)] = np.matmul(A[str(l-1)], self.modelParams['W' + str(l)]) + self.modelParams['b' + str(l)]
            
            
            
            # Doing batch Normalisation <= updating Z values for each batch
            # We need to do normalisation only for training
            
            #Z[str(l)], cache, mu, var = self.DoBatchNormalisation(Z[str(l)], gamma, beta)
            
            #if np.isnan(Z[str(l)]):
            #print("Z is", Z[str(l)])
            #print(Z[str(l)])
            # applying corresponding activation function
            A[str(l)] = self.GetA(Z[str(l)], l)


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
        elif self.NNArch[l]['AF'] == 'linear':
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
        #Z = args[2]
        #self.modelParams = args[3]
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
                    '''
                    dJ['dW'+str(l)] = (1 / m) * np.matmul(A[str(l-1)].T, delta[str(l)]) \
                    + self.lambd * modelParams['W' + str(l)] / m
                    dJ['db'+str(l)] = (1 / m) * np.sum(delta[str(l)], axis=0)#, keepdims=True)
                    '''
                    dJ['dW'+str(l)] = (np.matmul(A[str(l-1)].T, delta[str(l)]) \
                    + self.lambd * self.modelParams['W' + str(l)]) #/ m 
                    dJ['db'+str(l)] = np.sum(delta[str(l)], axis=0) #/ m#, keepdims=True)
                else:
                    dAF = self.GetdAF(A[str(l)], l)
                    delta[str(l)] = np.multiply(np.matmul(delta[str(l+1)], self.modelParams['W' + str(l+1)].T), dAF)
                    # gradients of the hidden layer
                    # W^{l} = A^{l-1} * delta^{l}
                    '''
                    dJ['dW'+str(l)] = (1 / m) * np.matmul(A[str(l-1)].T, delta[str(l)]) \
                    + self.lambd * modelParams['W' + str(l)] / m
                    dJ['db'+str(l)] = (1 / m) * np.sum(delta[str(l)], axis=0)#, keepdims=True)
                    '''
                    dJ['dW'+str(l)] = (np.matmul(A[str(l-1)].T, delta[str(l)]) \
                    + self.lambd * self.modelParams['W' + str(l)]) #/ m
                    dJ['db'+str(l)] = np.sum(delta[str(l)], axis=0) #/ m#, keepdims=True)
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
                    dJ['dW'+str(l)] = (np.matmul(A[str(l-1)].T, delta[str(l)]) \
                    + self.lambd * self.modelParams['W' + str(l)])# / m
                    #print(dJ['dW'+str(l)])
                    dJ['db'+str(l)] = np.sum(delta[str(l)], axis=0) / m#, keepdims=True)
                    # clipping gradients:
                    #if dJ['dW'+str(l)].any() > 10 or dJ['dW'+str(l)].any() < -10:
                    #    dJ['dW'+str(l)] = np.slip(dJ['dW'+str(l)], 0, 1)
                    #if dJ['db'+str(l)].any() > 10 or dJ['db'+str(l)].any() < -10:
                    #    dJ['db'+str(l)] = np.slip(dJ['db'+str(l)], 0, 1)
                else:
                    #dAF = funclib.ActivationFuncs().CallDSigmoid(A[str(l)])
                    # Calling derivative of current activation function
                    dAF = self.GetdAF(A[str(l)], l)
                    delta[str(l)] = np.multiply(np.matmul(delta[str(l+1)], self.modelParams['W' + str(l+1)].T), dAF)
                    #print("delta"+str(l), delta[str(l)].shape)
                    # gradients of the hidden layer
                    # W^{l} = A^{l-1} * delta^{l}
                    dJ['dW'+str(l)] = (np.matmul(A[str(l-1)].T, delta[str(l)]) \
                    + self.lambd * self.modelParams['W' + str(l)]) / m
                    dJ['db'+str(l)] = np.sum(delta[str(l)], axis=0) / m#, keepdims=True)
                    
                    # clipping gradients:
                    #if dJ['dW'+str(l)].any() > 10 or dJ['dW'+str(l)].any() < -10:
                    #    dJ['dW'+str(l)] = np.slip(dJ['dW'+str(l)], 0, 1)
                    #if dJ['db'+str(l)].any() > 10 or dJ['db'+str(l)].any() < -10:
                    #    dJ['db'+str(l)] = np.slip(dJ['db'+str(l)], 0, 1)
                        
                #print("dW"+str(l), dJ['dW'+str(l)])
                #print("db"+str(l), dJ['db'+str(l)].shape)
            
        return dJ
    
    # Method to Update Weights 
    # (on each iteration)
    def UpdateWeights(self, *args):
        dJ = args[0]
        #self.modelParams = args[1]
        m = args[2]
        
        gamma = 0.9
        eps=1e-8
        
        algorithm = self.optimization
        
        # Different versions of weights updates :)
        if algorithm == 'GD':            
            for l in range(1, self.nLayers):
                self.modelParams['W' + str(l)] -= dJ['dW' + str(l)] * self.alpha / (m)# * np.sqrt(6.0 / (n_l + n_next)) # <= this one should be correct
                self.modelParams['b' + str(l)] -= dJ['db' + str(l)] * self.alpha / (m)# * np.sqrt(6.0 / (n_l + n_next)) # <= this one should be correct
        elif algorithm == 'MBGD':
            for batch in range(self.BatchSize):
                for l in range(1, self.nLayers):
                    self.modelParams['W' + str(l)] -= dJ['dW' + str(l)] * self.alpha / (m)# * np.sqrt(6.0 / (n_l + n_next)) # <= this one should be correct
                    self.modelParams['b' + str(l)] -= dJ['db' + str(l)] * self.alpha / (m)# * np.sqrt(6.0 / (n_l + n_next)) # <= this one should be correct
        elif algorithm == "Momentum":
            for l in range(1, self.nLayers):
                self.update_params["v_w"+str(l)] = gamma *self.update_params["v_w"+str(l)] + self.alpha * (dJ['dW' + str(l)]/m)
                self.update_params["v_b"+str(l)] = gamma *self.update_params["v_b"+str(l)] + self.alpha * (dJ['db' + str(l)]/m)
                self.modelParams["W"+str(l)] -= self.update_params["v_w"+str(l)]
                self.modelParams["b"+str(l)] -= self.update_params["v_b"+str(l)]
        elif algorithm == "Adagrad":
            for l in range(1, self.nLayers):
                self.update_params["v_w"+str(l)] += (dJ['dW' + str(l)]/m)**2
                self.update_params["v_b"+str(l)] += (dJ['db' + str(l)]/m)**2
                self.modelParams["W"+str(l)] -= (self.alpha/(np.sqrt(self.update_params["v_w"+str(l)])+eps)) * (dJ['dW' + str(l)]/m)
                self.modelParams["b"+str(l)] -= (self.alpha/(np.sqrt(self.update_params["v_b"+str(l)])+eps)) * (dJ['db' + str(l)]/m)

    
        #return self.modelParams
    
    # Train Neural Network using Gradient Descent
    def TrainNetworkGD(self, *args):
        X_train = args[0]
        Y_train = args[1]
        #print(Ytrain)
        m = args[2]
        
        #print("NN m is", m)
        
        # Initialising parameters
        #self.modelParams = self.InitParams(self.NNArch, self.nInput)
        self.InitParams(self.NNArch, self.nInput)
        
        costs =  []        
        if self.NNType == 'Classification':
            #tqdm_notebook(range(epochs), total=epochs, unit="epoch")
            # Running Optimisation Algorithm
            for epoch in tqdm_notebook(range(self.epochs), total=self.epochs, unit="epoch"):#range(0, self.epochs, 1):
                # Propagating Forward
                A, Z = self.DoFeedForward(X_train, self.modelParams)
                # Calculating cost Function
                J = funclib.CostFuncs().CallNNLogistic(Y_train,\
                                     A[str(self.nLayers-1)],\
                                     self.modelParams,\
                                     self.nLayers,\
                                     m,\
                                     self.lambd, X_train)
                #print(J)
                # Back propagation - gradients
                dJ = self.DoBackPropagation(Y_train, A, Z, self.modelParams, m, X_train)
                #print(dJ)
                # updating weights
                #modelParams = self.UpdateWeights(dJ, self.modelParams, m)
                self.UpdateWeights(dJ, self.modelParams, m)
                # getting values of cost function at each epoch
                if(epoch % 1 == 0):
                    print('Cost after iteration# {:d}: {:f}'.format(epoch, J))
                costs.append(J)
            # returning set of optimal model parameters
            return self.modelParams, costs
        
        elif self.NNType == 'Regression':
            # Running Optimisation Algorithm
            for epoch in tqdm_notebook(range(self.epochs), total=self.epochs, unit="epoch"):#for epoch in range(1, self.epochs+1, 1):
                # Propagating Forward
                A, Z = self.DoFeedForward(X_train, self.modelParams)
                # Calculating cost Function
                J = funclib.CostFuncs().CallNNMSE(Y_train,\
                                     A[str(self.nLayers-1)],\
                                     self.modelParams,\
                                     self.nLayers,\
                                     m,\
                                     self.lambd, X_train)
                #print(J)
                # Back propagation - gradients
                dJ = self.DoBackPropagation(Y_train, A, Z, self.modelParams, m, X_train)
                #print(dJ)
                # updating weights
                #self.modelParams = self.UpdateWeights(dJ, self.modelParams, m)
                self.UpdateWeights(dJ, self.modelParams, m)
                
                #print(modelParams)
                #print(epoch, J, dJ)
                # getting values of cost function at each epoch
                #if(epoch % 1 == 0):
                #    print('Cost after iteration# {:d}: {:f}'.format(epoch, J))
                costs.append(J)
            # returning set of optimal model parameters
            return self.modelParams, costs
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
        self.nBatches = self.nInput // m
        
        print("nBatches", self.nBatches)
        
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
        #self.modelParams = self.InitParams(self.NNArch, m)
        self.InitParams(self.NNArch, m)
        
        if self.NNType == 'Classification':
            # Running Optimisation Algorithm
            for epoch in tqdm_notebook(range(self.epochs), total=self.epochs, unit="epoch"):#for epoch in range(1, self.epochs+1, 1):
                # looping through all batches
                for j in range(self.nBatches):
                    # chosing the indexes for playing around
                    points = np.random.choice(indices, size=m, replace=False)
                    X_train_batch = X_train[points]
                    Y_train_batch = Y_train[points]
                    #print(np.shape(X_train_batch))
                    # Propagating Forward
                    A, Z = self.DoFeedForward(X_train_batch, self.modelParams)
                    # Calculating cost Function
                    J = funclib.CostFuncs().CallNNLogistic(Y_train_batch,\
                                         A[str(self.nLayers-1)],\
                                         self.modelParams,\
                                         self.nLayers,\
                                         m,\
                                         self.lambd)
                    # trying scikit leaкт <= didn't work out very well
                    #J = log_loss(np.argmax(Y_train_batch, axis=1), A[str(self.nLayers-1)])
                                        # Back propagation - gradients
                    dJ = self.DoBackPropagation(Y_train_batch, A, Z, self.modelParams, m, X_train)
                    # updating weights
                    #self.modelParams = self.UpdateWeights(dJ, self.modelParams, m)
                    self.UpdateWeights(dJ, self.modelParams, m)
                    
                    cost += J/self.nInput
                if(epoch % 1 == 0):
                    print('Cost after iteration# {:d}: {:f}'.format(epoch, J))
                costs.append(J)
                
            return self.modelParams, costs
        
        elif self.NNType == 'Regression':
            # Running Optimisation Algorithm
            for epoch in tqdm_notebook(range(self.epochs), total=self.epochs, unit="epoch"):#for epoch in range(1, self.epochs+1, 1):
                #points = np.random.permutation(indices, size=m)
                # looping through all batches
                for j in range(self.nBatches):
                    # chosing the indexes for playing around
                    points = np.random.choice(indices, size=m, replace=False)
                    X_train_batch = X_train[points]
                    Y_train_batch = Y_train[points]
                    #print(np.shape(X_train_batch))
                    # Propagating Forward
                    A, Z = self.DoFeedForward(X_train_batch, self.modelParams)
                    # Calculating cost Function
                    J = funclib.CostFuncs().CallNNMSE(Y_train_batch,\
                                         A[str(self.nLayers-1)],\
                                         self.modelParams,\
                                         self.nLayers,\
                                         m,\
                                         self.lambd, X_train_batch)
                                        # Back propagation - gradients
                    dJ = self.DoBackPropagation(Y_train_batch, A, Z, self.modelParams, m, X_train_batch)
                    # updating weights
                    #self.modelParams = self.UpdateWeights(dJ, self.modelParams, m)
                    self.UpdateWeights(dJ, self.modelParams, m)
                                        
                    cost += J/self.nInput
                    
                #print(modelParams)
                #print(epoch, dJ)
                #if(epoch % 1 == 0):
                #    print('Cost after iteration# {:d}: {:f}'.format(epoch, J))
                costs.append(J)
            # returning set of optimal model parameters
            return self.modelParams, costs
        else:
            Exception('It is neither Regression nor Classification task! Check Parameter File.')
            
    '''
    Function which will fit the test data
    '''
    def MakePrediction(self, *args):
        # making prediction for the data set
        X_test = args[0]

        modelParams = args[1]

        # making a prediction
        A, Z = self.DoFeedForward(X_test, modelParams)#self.feed_forward_out(X)
        
        if self.NNType == 'Classification':
            # outputting the values which corresponds to the maximum probability
            return np.argmax(A[str(self.nLayers-1)], axis=1)
        elif self.NNType == 'Regression':
            return A[str(self.nLayers-1)]
    
    