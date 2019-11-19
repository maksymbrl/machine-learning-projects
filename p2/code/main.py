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


# Scikitlearn imports to check results
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# We'll need some metrics to evaluate our models
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score

import keras
# stochastic gradient descent
from keras.optimizers import SGD
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
    NNType, NNArch, nLayers, nFeatures, \
    nHidden, nOutput, epochs, alpha, lmbd, \
    X_train, X_test, Y_train, Y_test, \
    m, nInput, seed = arch.CreateNetwork()
    
    #print("nFeatures", nFeatures)
    '''
    So, If the total number of layers = 2, we can switch to Simple Logistic regression.
    However, once the number of layers is > 2, we are going to use Neural Network.
    The number of hidden layers can be specified in the parameter file above 
    (this is done for simplicity). In principle, the Neural network without hidden layers,
    should produc ethe same results as logistic regression (at least, i think so).
    '''
    #print('epochs are', epochs)
    
    if (nLayers == 2):
        print('Logistic Regression')
        '''
        Simplest Logistic regression
        '''
        print('''
              Logistic Regression Via Manual Coding
              ''')
        
        activeFuncs = funclib.ActivationFuncs()
        costFuncs = funclib.CostFuncs()
        # load the data from the file
        #filename = 'marks.txt'
        #path = os.getcwd() + '/data/' + filename
        #data = pd.read_csv(path, header = None)
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
        #Y_train.ravel()
        m = len(Y_train)
        #print(m)
        costs = []
        for epoch in epochs1:
            Y_pred = np.dot(X_train, theta)
            A = activeFuncs.CallSigmoid(Y_pred)#CallSigmoid(Y_pred)
            #print(A)
            # cost function        
            J, dJ = costFuncs.CallLogistic(X_train, Y_train, A)
            # Adding regularisation
            J = J + lmbd / (2*m) * np.sum(theta**2)
            dJ = dJ + lmbd * theta / m
            #print(J, "\n", dJ)
            # updating weights
            theta = theta - alpha * dJ #+ lmbd * theta/m
            # updating cost func history
            costs.append(J)
        
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
                                         lmbd, nInput, seed)
        modelParams, costs = neuralNet.TrainNetwork(X_train, Y_train, m)
        
        #print(NNArch[1]['AF'])
        print('''
              Initialising Keras
              ''')
        #classifier = Sequential()

        # Random normal initializer generates tensors with a normal distribution.
        #First Hidden Layer
        #classifier.add(Dense(nHidden, activation=NNArch[1]['AF'], kernel_initializer='random_normal', input_dim=nFeatures))
        #Second  Hidden Layer
        #classifier.add(Dense(nHidden, activation=NNArch[1]['AF'], kernel_initializer='random_normal'))
        #Output Layer
        #classifier.add(Dense(nOutput, activation=NNArch[2]['AF'], kernel_initializer='random_normal'))
        #Compiling the neural network
        '''
        To optimize our neural network we use Adam. Adam stands for Adaptive 
        moment estimation. Adam is a combination of RMSProp + Momentum.
        '''
        # Stochatic gradient descent
        #sgd = SGD(lr=alpha)
        #classifier.compile(optimizer = sgd, loss='binary_crossentropy', metrics =['accuracy'])
        #Fitting the data to the training dataset
        #classifier.fit(X_train, Y_train, batch_size=10, epochs = epochs)
        
        
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
        
        #np.set_printoptions(precision=4, suppress=True)
        #eval_results = classifier.evaluate(X_test, Y_test, verbose=0) 
        #print("\nLoss, accuracy on test data: ")
        #print("%0.4f %0.2f%%" % (eval_results[0], eval_results[1]*100))
        
    else:
        '''
        Raise an exception, if number of layers is smaller than 2. It shouldn't be the case,
        because in param file I am specifying number of hidden layers and not the total layers.
        Then I add 2 to that number in the code. But better safe than sorry :) 
        '''
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