#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:29:25 2019

@author: maksymb
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
# to read parameter file
import yaml

import time

# For Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler

# initialising pretty printing with sympy 
# for Latex characters and more
from IPython.display import display, Latex, Markdown
from sympy import * #init_printing
#from sympy.printing.latex import print_latex
#init_printing(use_latex='mathjax')

# import manual libraries
import funclib #, data_processing

    
'''
Retrieving Parameters From Parameter File 
and Configuring our Neural Network
'''
class NetworkArchitecture:
    # constructor
    def __init__(self, *args):
        paramFile = args[0]
        # Getting values from Parameter file
        with open(paramFile) as f:
            self.paramData = yaml.load(f, Loader = yaml.FullLoader)
            #sorted = yaml.dump(paramData, sort_keys = True)
            #print(self.paramData)
    
    # Method to prepare the data
    def PrepareData(self, *args):
        # Path to data
        dataPath = self.paramData['dataPath']
        # reading in the data and displaying first values
        data = pd.read_csv(dataPath, index_col=False)#, header = None, index_col=False)
        # looking into the data
        display(data.head(3))
        # Dropping ID (because we do not need it)
        data = data.drop('ID', axis = 1)
        # Checking for missing values
        for column in data:
            if data[column].isnull().values.any():
                print("NaN value/s detected in " + column)
            else:
                continue
                #print("column {} doesn't have null values".format(column))
        # Renaming Column Pay and also the default one
        data.rename(columns={'PAY_0': 'PAY_1'}, inplace = True)
        # also making in lower case column names
        data.rename(columns=lambda x: x.lower(), inplace = True)
        data.rename(columns={'default.payment.next.month': 'default'}, inplace = True)
        '''
        Plotting correlation matrix to see
        which features will affect default the most
        '''
        corr = data.corr()#data.drop('ID', axis = 1).corr()
        f, ax = plt.subplots(figsize=(15, 15))
        # Generate a custom diverging colormap
        cmap = sbn.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        sbn.heatmap(corr, cmap = cmap, vmin=0,vmax=1, center=0, square=True, annot=True, linewidths=.5)
        '''
        We can see that PAY and BILL has the highest impact
        '''
        # Making dummy features (i.e. changing stuff to either 0 or 1)
        # for this we need to add additional columns
        '''
        From Paper: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
        So, I am going to leave these 4, however, in the data set I saw also 5, which I do not
        really understand the meaning of, so I will drop this one
        '''
        # e.g. if the person was in grad school, we will get 1, 0 otherwise (etc.)
        data['grad_school']  = (data['education']==1).astype('int')
        #data.insert((data['education']==1).astype('int'), data.columns[2], 'grad_school')
        data['university']   = (data['education']==2).astype('int')
        data['high_school']  = (data['education']==3).astype('int')
        data['others']       = (data['education']==4).astype('int')
        data.drop('education', axis=1, inplace=True)
        
        # repeating the same for sex 
        data['male'] = (data['sex']==1).astype('int')
        data.drop('sex', axis=1, inplace=True)
        # dumping all singles and others in one category
        data['married'] = (data['marriage']==2).astype('int')
        data.drop('marriage', axis=1, inplace=True)
        
        # I assume that everything which is less than 0
        # is paid on time
        paynments = ['pay_1','pay_2','pay_3','pay_4','pay_5','pay_6']
        for pay in paynments:
            data.loc[data[pay]<=0,pay] = 0
            
        # retrieving columns' names
        #dataColumns = data.columns
        # manually normalising data
        #data[dataColumns] = data[dataColumns].apply(lambda x: (x - np.mean(x)) / np.std(x))
        
        '''
        After all manipulations, the data will look like this
        '''
        display(data.head(3))
        
        # passing data to further processing
        return data    
        
    '''
    Working on Data
    '''
    def ProcessData(self, *args):
        # getting data
        data = self.PrepareData()
        # Seperate the label into different Dataframe (outcome) and the other features in (data)
        X = data.drop('default',axis=1)
        # leaving only default
        Y = data['default']
        # adding new axis to the data
        Y = Y[:, np.newaxis]
        # Scaling the dataset (we do not scale Y, because it has values either 0 or 1)
        #rs = RobustScaler()
        #X = rs.fit_transform(X)
        # Splitting our data into training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
        ss = StandardScaler()
        X_train = ss.fit_transform( X_train )
        X_test = ss.transform( X_test )
        #print(X)
        #display(data.head(3))
        #dataColumns = data.columns
        # manually normalising data
        #data[dataColumns] = data[dataColumns].apply(lambda x: (x - np.mean(x)) / np.std(x))
        
        # Path to data
        #dataPath = self.paramData['dataPath']
        #data = pd.read_csv(dataPath, header = None)
        '''
        Write code which will split data into
        Train and test Sets (?)
        '''
        # X = feature values, all the columns except the last column
        # (also ignoring first row, as it contains labels)
        #X = data.iloc[1:, 3:13].values
        # y = target values, last column of the data frame
        #Y = data.iloc[1:, 13].values
        
        #X = data[features].values    
        #y = data[ output ].values

        #from sklearn.model_selection import train_test_split
        #X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)

        #from sklearn.preprocessing import StandardScaler
        #scX = StandardScaler()
        #X_train = scX.fit_transform( X_train )
        #X_test = scX.transform( X_test )

        #print(X)
        #print(Y)
        
        '''
        Now we encode the string values in the features 
        to numerical values as a ML Algorithm can only 
        work on numbers and not on string values.
        The only 2 values are Gender and Region which 
        need to converted into numerical data
        '''

        #labelencoder_X_1 = LabelEncoder()
        # now, the 3d column became the first one
        #X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
        # to encode gender name
        #labelencoder_X_2 = LabelEncoder()
        #X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
        # creating dummy variables
        #onehotencoder = OneHotEncoder(categorical_features = [1])
        #X = onehotencoder.fit_transform(X).toarray()
        #X = X[:, 1:]
        
        #from sklearn.model_selection import train_test_split
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
        
        #from sklearn.preprocessing import StandardScaler
        #sc = StandardScaler()
        #X_train = sc.fit_transform(X_train)
        #X_test = sc.transform(X_test)
        
        # insert a column of 1's as the first entry in the feature
        # vector -- this is a little trick that allows us to treat
        # the bias as a trainable parameter *within* the weight matrix
        # rather than an entirely separate variable
        #X = np.c_[np.ones((X.shape[0])), X]
        #Y = Y[:, np.newaxis]
        #theta = np.zeros((X.shape[1], 1))
        # No. of training examples
        m = X_train.shape[1]
        
        return X_train, X_test, Y_train, Y_test, m
    
    '''
    Configuring Neural Network:
    Manual Architecture with corresponding data
    and parameter file
    '''
    def CreateNetwork(self, *args):
        # Getting processed data <= it will not be changed in this stage
        # I am using to just identify some key features of the NN
        X_train, X_test, Y_train, Y_test, m = self.ProcessData()
        # NNtype
        NNType = self.paramData['type'] #['Classification', 'Regression']
        # Layer Architecture
        NNArch = []
        # Total Number of layers, should be 2 or more
        # (2 for logistic regression)
        nLayers = self.paramData['nHiddenLayers'] + 2 #3
        # Number of Input Neurons (Input Data)
        nInput = X_train.shape[0] # <= the amount of data points per variable
        print("nInput", nInput)
        nFeatures = X_train.shape[1] # <= the amount of variables
        print('nFeatures', nFeatures)
        #self.n_inputs           = Xtrain.shape[0]   # Number of input data
        #self.n_features         = Xtrain.shape[1]   # Number of features
        # Number of Hidden Neurons
        nHidden = self.paramData['nHiddenNeurons']#len(X)
        # Number of Output Neurons
        nOutput = self.paramData['nOutputNeurons']#len(Y)
        # activation functions
        aHiddenFunc = self.paramData['hiddenFunc']
        aOutputFunc = self.paramData['outputFunc']
        #aFuncs = ['sigmoid', 'relu', 'tanh', 'step']
        # Creating NN architecture
        for l in range(0, nLayers):
            # input layer
            if l == 0:
                NNArch.append({"LSize": nFeatures, "AF": "none"})
            # output layer
            elif l == (nLayers-1):
                NNArch.append({"LSize": nOutput, "AF": aOutputFunc})
            # hidden layers
            else:
                NNArch.append({"LSize": nHidden, "AF": aHiddenFunc})
        # epochs to train the algorithm
        epochs = self.paramData['epochs']#1000
        # learning rate
        alpha = self.paramData['alpha'] #0.3
        # regularization parameter
        lmbd = self.paramData['lambda'] #0.001
        
        
        return NNType, NNArch, nLayers, nFeatures, nHidden, nOutput, epochs, alpha, lmbd, X_train, X_test, Y_train, Y_test, m, nInput
    
'''
The class used for both classification and regression
depending on the cost function and the user's desire
'''
class NeuralNetwork:
    # constructor
    def __init__(self, *args):
        # random seed, to make the same random number each time
        np.random.seed(1)
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
        #self.X = args[9]
        #self.Y = args[10]
        #self.m = args[11]
        # Only for printing purpose
        
        display(Markdown(u'''
        =========================================== 
            Start {} Neural Network 
        =========================================== 
        No. of hidden layers:        {} 
        No. of input neurons:        {} 
        No. of hidden neurons:       {} 
        No. of output neurons:       {} 
        Activ. Func in Hidden Layer: {} 
        Activ. Func in Output Layer: {} 
        No. of test features: 
        No. of epochs to see:        {} 
        Learning Rate, \u03B1:            {} 
        Regularization param, \u03BB:     {} 
                      '''.format(self.NNType, 
                                 self.nLayers-2, 
                                 self.nInputNeurons, 
                                 self.nHiddenNeurons, 
                                 self.nOutputNeurons,
                                 self.NNArch[1]['AF'],
                                 self.NNArch[self.nLayers-1]['AF'],
                                 self.epochs,
                                 #latex('$\\alpha$'),
                                 self.alpha,
                                 self.lambd)))
    
        self.nInput = args[9]
        
    # 
    def InitParams(self, *args):
        # biases and weights for hidden and output layers
        # dictionary to contain all parameters for each layer
        # (i.e. "W1", "b1", ..., "WL", "bL", except inpur one)
        modelParams = {}
        for l in range(1, self.nLayers):
            # weights for each layer (except input one)
            #print(self.nInputNeurons, self.nHiddenNeurons)
            modelParams['W' + str(l)] = np.random.randn(self.NNArch[l-1]["LSize"], self.NNArch[l]["LSize"])
            #print(np.shape(modelParams['W' + str(l)]))
            # biases for each layer (except input one)
            #modelParams['b' + str(l)] = np.zeros((self.NNArch[l]["LSize"], self.nOutputNeurons)) + 0.01
            modelParams['b' + str(l)] = np.zeros((self.nInput, self.NNArch[l]["LSize"])) + 0.01
            #print(np.shape(modelParams['b' + str(l)]))
            
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
        # Calculating gradients of the cost function for each layer
        # (going from last to first hidden layer)
        for l in reversed(range(1, self.nLayers)):
            # calculating error for each layer
            if (l == self.nLayers - 1):
                delta[str(l)] = A[str(l)] - Y
                # gradients of output layer (+ regularization)
                # W^{l} = A^{l-1} * delta^{l}
                dJ['dW'+str(l)] = np.matmul(A[str(l-1)].T, delta[str(l)]) #+ self.lambd * self.modelParams['W' + str(l)]
                dJ['db'+str(l)] = np.sum(delta[str(l)], axis=0, keepdims=True)
            else:
                #dAF = funclib.ActivationFuncs().CallDSigmoid(A[str(l)])
                dAF = self.GetdAF(A[str(l)], l)
                delta[str(l)] = np.multiply(np.matmul(delta[str(l+1)], modelParams['W' + str(l+1)].T), dAF)
                # gradients of the hidden layer
                # W^{l} = A^{l-1} * delta^{l}
                dJ['dW'+str(l)] = np.matmul(A[str(l-1)].T, delta[str(l)]) #+ self.lambd * self.modelParams['W' + str(l)]
                dJ['db'+str(l)] = np.sum(delta[str(l)], axis=0, keepdims=True)
            
        return dJ
    
    # Method to Update Weights 
    # (on each iteration)
    def UpdateWeights(self, *args):
        dJ = args[0]
        modelParams = args[1]
        for l in range(1, self.nLayers):
            modelParams['W' + str(l)] -= dJ['dW' + str(l)] * self.alpha
            modelParams['b' + str(l)] -= dJ['db' + str(l)] * self.alpha
            
        return modelParams
        
    # Train Neural Network
    def TrainNetwork(self, *args):
        Xtrain = args[0]
        Ytrain = args[1]
        m = args[2]
        # Initialising parameters
        modelParams = self.InitParams()
        costs =  []
        # Running Optimisation Algorithm
        for epoch in range(1, self.epochs+1, 1):
            # Propagating Forward
            A, Z = self.DoFeedForward(Xtrain, modelParams)
            # Calculating cost Function
            J = funclib.CostFuncs().CallNNLogistic(Ytrain, A[str(self.nLayers-1)], m)
            # Back propagation - gradients
            dJ = self.DoBackPropagation(Ytrain, A, Z, modelParams, m)
            # updating weights
            modelParams = self.UpdateWeights(dJ, modelParams)
            # getting values of cost function at each epoch
            if(epoch % 100 == 0):
                print('Cost after iteration# {:d}: {:f}'.format(epoch, J))
            costs.append(J)
        
        # returning set of optimal model parameters
        return modelParams, costs
    