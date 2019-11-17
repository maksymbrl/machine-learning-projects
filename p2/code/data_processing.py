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

import funclib

'''
Class to preprocess the data set
'''

class DataProcessing:
    # constructor
    def __init__(self, *args):
        paramFile = args[0]
        # Getting values from Parameter file
        with open(paramFile) as f:
            self.paramData = yaml.load(f, Loader = yaml.FullLoader)
    
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

        m = X_train.shape[1]
        
        return X_train, X_test, Y_train, Y_test, m

    
'''
Retrieving Parameters From Parameter File 
and Configuring our Neural Network
'''
class NetworkArchitecture:
    # constructor
    def __init__(self, *args):
        pass
    
    '''
    Configuring Neural Network:
    Manual Architecture with corresponding data
    and parameter file
    '''
    def CreateNetwork(self, *args):
        paramFile = args[0]
        # Getting values from Parameter file
        with open(paramFile) as f:
            self.paramData = yaml.load(f, Loader = yaml.FullLoader)
            #sorted = yaml.dump(paramData, sort_keys = True)
            #print(self.paramData) 
        # Getting processed data <= it will not be changed in this stage
        # I am using to just identify some key features of the NN
        X_train, X_test, Y_train, Y_test, m = DataProcessing.ProcessData()
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
    
