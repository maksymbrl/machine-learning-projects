#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:29:25 2019

@author: maksymb
"""

import os
import sys
import numpy as np
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
from sympy import * #init_printing
#from sympy.printing.latex import print_latex
#init_printing(use_latex='mathjax')

# import manual libraries
import funclib #, data_processing
import regression
import neural

    
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
        outputPath = self.paramData['outputPath']
        NNType = self.paramData['type']
        # If data is of classification type, then process corresponding data set
        if (NNType == 'Classification'):
            
            '''
            Checking the format of the file - .csv or .xls
            and renaming the last column to 'default', to access it
            more easily
            '''
            dataFormat = dataPath[-4:]
            if dataFormat == '.csv':
                data = pd.read_csv(dataPath, index_col=False)#, header = None, index_col=False)
                data.rename(columns={'default.payment.next.month': 'default'}, inplace = True)
            elif dataFormat == '.xls':
                nanDict = {}
                data = pd.read_excel(dataPath, header=1, skiprows=0, index_col=False, na_values=nanDict)
                data.rename(index=str, columns={"default payment next month": 'default'}, inplace=True)
            '''
            Dropping the 'ID' column, because it is no use.
            Also, renaming the PAY_0 column to PAY_1 for con-
            sistency. Lastly, making all the columns lower case
            (I like it more like that :).
            '''
            data = data.drop('ID', axis = 1)
            data.rename(columns={'PAY_0': 'PAY_1'}, inplace = True)
            data.rename(columns=lambda x: x.lower(), inplace = True)
            '''
            Checking if there are missing or NaN data.
            '''
            data.info()
            '''
            The data quickly checked and it seems that everything is fine.
            Now, let's take a look for the 10 random data samples.
            '''
            display(data.sample(10))#data.head(3))
            # Checking for missing values
            #for column in data:
            #    if data[column].isnull().values.any():
            #        print("NaN value/s detected in " + column)
            #    else:
            #        continue
                    #print("column {} doesn't have null values".format(column))
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
            Looking for the most dense regions, it is obvious
            that we are interested in columns: 
            "pay_i", "bil_amti" and "pay_amti".
            However, let us study the data set in a more profound fasion, i.e. individually.
            '''
            '''
            Lat's look into description of individual column. From paper:
            ID: ID of each client
            LIMIT_BAL: Amount of given credit in NT dollars (includes individual 
            and family/supplementary credit)
            SEX: Gender (1=male, 2=female)
            EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others)
            MARRIAGE: Marital status (1=married, 2=single, 3=others)
            AGE: Age in years
            PAY_0: Repayment status in September, 2005 
            (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 
            8=payment delay for eight months, 9=payment delay for nine months and above)
            PAY_2: Repayment status in August, 2005 (scale same as above)
            PAY_3: Repayment status in July, 2005 (scale same as above)
            PAY_4: Repayment status in June, 2005 (scale same as above)
            PAY_5: Repayment status in May, 2005 (scale same as above)
            PAY_6: Repayment status in April, 2005 (scale same as above)
            BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
            BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
            BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
            BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
            BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
            BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
            PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
            PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
            PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
            PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
            PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
            PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
            default.payment.next.month: Default payment (1=yes, 0=no)
            '''
            
            '''
            Trying to describe the data. Since Visualising everything at once 
            is not that reasonable(quite long table), I am going to visualize data
            by columns. 
            Index(['limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_1', 'pay_2',
                   'pay_3', 'pay_4', 'pay_5', 'pay_6', 'bill_amt1', 'bill_amt2',
                   'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6', 'pay_amt1',
                   'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6', 'default'],
                  dtype='object')
            '''
            #print(data.columns)
            # Categorical variables description
            display(data[['limit_bal', 'sex', 'education', 'marriage', 'age']].describe())
            
            '''
            The output:
            age: ranges 21-79
            marriage: ranges 0-3, what is 0? => makes sense to put it into 3 other
            education: 0-6 => 0, 5 and 6 are unlabeled data (according to the data table)
            sex: 1-2
            limit_bal: 10^4 - 10^6
            
            Let's seee how much data for an education unlabeled data we have
            '''
            vals = [0, 4, 5, 6]
            for val in vals:
                print("We have {} with education={}".format(len(data.loc[ data["education"]==val]), val))
            '''
            There are not that many of them and, I think, It wouldn't harm if I put all of them inside 4 = others
            for easier classification. <= small data cleaning
            '''
            fill = (data['education'] == 5) | (data['education'] == 6) | (data['education'] == 0)
            data.loc[fill, 'education'] = 4
            # counting education values after
            print('After cleaning')
            display(data['education'].value_counts())
            # doing the same for marriage
            fill = (data['marriage'] == 0)
            data.loc[fill, 'marriage'] = 3
            print('After cleaning')
            display(data['marriage'].value_counts())
            '''
            So, I have put everything to 4 and 3(educationand marriage), which stands for other.
            Other means education higher (or lower) than university, i.e. double PhDs or whatever, (than high school).
            Other in marriage can stand for widowed, for instance.
            '''
            
            '''
            Let's take a look into the payment data
            '''
            display(data[['pay_1', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']].describe())
            vals = [1, 2, 3, 4, 5, 6]
            for val in vals:
                display(data['pay_'+str(val)].value_counts())
            '''
            Roughly zeros are like half of the data set. I will just leave it be. However,
            I thinkit is safe to put all -2 into -1.
            '''
            for val in vals:
                fill = (data['pay_' + str(val)] == -2)
                data.loc[fill, 'pay_' + str(val)] = -1
                display(data['pay_'+str(val)].value_counts())
            
            '''
            there is -2 present, which I am not quite sure the meaning of. There is also label 0...
            '''
            
            display(data[['bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6']].describe())
            '''
            There are also negative values present. Let's see how many of these values are there
            '''
            for val in vals:
                display(data['bill_amt'+str(val)].value_counts(ascending=True))
            '''
            Negative values are what people suppose to pay back? I will just leave it as it is.
            '''
                
            display(data[['pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']].describe()) 
            for val in vals:
                display(data['pay_amt'+str(val)].value_counts(ascending=True))
            '''
            There are just some people with a lot of money
            '''
            # Making dummy features (i.e. changing stuff to either 0 or 1)
            # for this we need to add additional columns
            '''
    
            '''
            # e.g. if the person was in grad school, we will get 1, 0 otherwise (etc.)
            #data['grad_school']  = (data['education']==1).astype('int')
            #data.insert((data['education']==1).astype('int'), data.columns[2], 'grad_school')
            #data['university']   = (data['education']==2).astype('int')
            #data['high_school']  = (data['education']==3).astype('int')
            #data['others']       = (data['education']==4).astype('int')
            #data.drop('education', axis=1, inplace=True)
            
            # repeating the same for sex 
            #data['male'] = (data['sex']==1).astype('int')
            #data.drop('sex', axis=1, inplace=True)
            # dumping all singles and others in one category
            #data['married'] = (data['marriage']==2).astype('int')
            #data.drop('marriage', axis=1, inplace=True)
            
            # I assume that everything which is less than 0
            # is paid on time
            #paynments = ['pay_1','pay_2','pay_3','pay_4','pay_5','pay_6']
            #for pay in paynments:
            #    data.loc[data[pay]<=0,pay] = 0
                
            # retrieving columns' names
            #dataColumns = data.columns
            # manually normalising data
            #data[dataColumns] = data[dataColumns].apply(lambda x: (x - np.mean(x)) / np.std(x))
            
            '''
            After all manipulations, the data will look like this
            '''
            display(data.sample(10))
            
            '''
            We will need to convert some data using OneHotEncoder. Specifically These are
            "education", "sex", "marriage" and "pay_i" columns
            '''
    
            '''
            Exploratory Data Analysis
            '''
            # The frequency of defaults
            yes = data.default.sum()
            no = len(data)-yes
            
            # Percentage
            yes_perc = round(yes/len(data)*100, 1)
            no_perc = round(no/len(data)*100, 1)
            
            plt.figure(figsize=(7,4))
            sbn.set_context('notebook', font_scale=1.2)
            sbn.countplot('default',data=data, palette="cool")
            plt.annotate('Non-default: {}'.format(no), xy=(-0.3, 15000), xytext=(-0.3, 3000), size=12)
            plt.annotate('Default: {}'.format(yes), xy=(0.7, 15000), xytext=(0.7, 3000), size=12)
            plt.annotate(str(no_perc)+" %", xy=(-0.3, 15000), xytext=(-0.1, 8000), size=12)
            plt.annotate(str(yes_perc)+" %", xy=(0.7, 15000), xytext=(0.9, 8000), size=12)
            plt.title("CREDIT CARDS' DEFAULT COUNT", size=14)
            #Removing the frame
            plt.box(False);
            
            #set_option('display.width', 100)
            #set_option('precision', 2)
            
            #print("SUMMARY STATISTICS OF NUMERIC COLUMNS")
            #print()
            #print(data.describe().T)
            
            # Creating a new dataframe with categorical variables
            labels = ['sex', 'education', 'marriage', 'pay_1', 'pay_2',
                   'pay_3', 'pay_4', 'pay_5', 'pay_6', 'default']
            subset = data[labels]
            fig, axes = plt.subplots(3,3, figsize=(20,15), sharey = 'all', facecolor='white')
            #fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
            axes = axes.flatten()
            #object_bol = df.dtypes == 'object'
            fig.suptitle('FREQUENCY OF CATEGORICAL VARIABLES (BY TARGET)')
            for axe, catplot in zip(axes, labels):
                #sbn.countplot(y=catplot, data=subset, ax=ax)#, order=np.unique(subset.values))
                sbn.countplot(x=catplot, hue="default", data = subset, palette = "cool", ax=axe)
        elif (NNType == 'Regression'):
            ''' Generating Data Set '''
            n_vars = self.paramData['nVars']
            N_points = self.paramData['nPoints']
            sigma = self.paramData['noise']
            # generating an array of symbolic variables
            # based on the desired amount of variables
            x_symb = sp.symarray('x', n_vars, real = True)
            # making a copy of this array
            x_vals = x_symb.copy()
            # and fill it with values
            for i in range(n_vars):
                x_vals[i] = np.arange(0, 1, 1./N_points)#np.sort(np.random.uniform(0, 1, N_points))
            # library object instantiation
            #lib = regression.RegressionPipeline(x_symb, x_vals)
            dataFunc = self.paramData['function']
            if dataFunc == 'Franke':
                # setting up the grid
                x, y = np.meshgrid(x_vals[0], x_vals[1])
                # and getting output based on the Franke Function
                z = funclib.DataFuncs().CallFranke(x, y) + sigma * np.random.randn(N_points, N_points)
                data = (x_symb, x_vals, x, y, z)
            
        # passing data to further processing
        return data    
        
    '''
    Working on Data
    '''
    def ProcessData(self, *args):
        # getting data
        data = self.PrepareData()
        outputPath = self.paramData['outputPath']
        NNType = self.paramData['type']
        # If data is of classification type, then process corresponding data set
        if (NNType == 'Classification'):
            # Seperate the label into different Dataframe (outcome) and the other features in (data)
            #X = data.drop('default',axis=1)
            # leaving only default
            #Y = data['default']
            # adding new axis to the data
            #Y = Y[:, np.newaxis]
            X = data.loc[:, data.columns != "default"].values
            Y = data.loc[:, data.columns == "default"].values
            print(np.shape(Y))
            #features = X
            #stdX = (features - features.mean()) / (features.std())
            #print(stdX)
            # Categorical variables to one-hot's - getting dummy features
            # I could do this in pandas or Keras, but I will go with this one
            # because I can? :)
            onehotencoder = OneHotEncoder(sparse=False, categories="auto")
            X = ColumnTransformer([("", onehotencoder, [3])], remainder="passthrough").fit_transform(X)
            #X = ColumnTransformer().fit_transform(X)
            #print(np.shape(X))
            #rs = RobustScaler()
            #X = rs.fit_transform(X)
            # Splitting our data into training and test sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
            # Scaling the dataset (we do not scale Y, because it has values either 0 or 1)
            # Scale to zero mean and unit variance
            ss = StandardScaler()
            rs = RobustScaler()
            X_train = ss.fit_transform( X_train )
            X_test = ss.transform( X_test )
            # One-hot's of the target vector
            Y_train_onehot = to_categorical(Y_train)#onehotencoder.fit_transform(Y_train.reshape(len(Y_train), -1))
            Y_test_onehot =  to_categorical(Y_test)#onehotencoder.fit(Y_test.reshape(len(Y_test), -1))
            
            print("Y_train shape", np.shape(Y_train))
            
            '''
            Now we encode the string values in the features 
            to numerical values as a ML Algorithm can only 
            work on numbers and not on string values.
            The only 2 values are Gender and Region which 
            need to converted into numerical data
            '''
            '''
            # insert a column of 1's as the first entry in the feature
            # vector -- this is a little trick that allows us to treat
            # the bias as a trainable parameter *within* the weight matrix
            # rather than an entirely separate variable
            #X = np.c_[np.ones((X.shape[0])), X]
            #Y = Y[:, np.newaxis]
            #theta = np.zeros((X.shape[1], 1))
            # No. of training examples
            '''
            m = X_train.shape[1]
            
            return X_train, X_test, Y_train, Y_test, Y_train_onehot, Y_test_onehot, m, onehotencoder#Y_train, Y_test, m
        
        elif (NNType == 'Regression'):
            x_symb, x_vals, x, y, z = data
            n_vars = self.paramData['nVars']
            N_points = self.paramData['nPoints']
            sigma = self.paramData['noise']
            poly_degree = self.paramData['degree']
            lambda_par = self.paramData['lambda']
            #print(type(poly_degree))
            #nproc = args[0]
            # for plotting betas (this valu will appear in the file name <= doesn't affect calculations)
            #npoints_name = args[1]
            #curr_lambda = args[2]
            # library object instantiation
            #lib = rl.RegressionLibrary(self.x_symb, self.x_vals)
            # raveling variables (making them 1d
            x_rav, y_rav, z_rav = np.ravel(x), np.ravel(y), np.ravel(z)
            # shape of z
            zshape = np.shape(z)
            #==============================================================================================================#
            ''' Linear Regression '''
            #==============================================================================================================#
            ''' MANUAL '''
            # getting design matrix
            func = funclib.NormalFuncs()
            # getting the design matrix
            X = func.ConstructDesignMatrix(x_symb, x_vals, poly_degree)
            # Dump everything into Regression Pipeline
            regression.RegressionPipeline().DoLinearRegression(X, x, y, z, x_rav, \
                                         y_rav, z_rav, zshape, \
                                         poly_degree, lambda_par, sigma, outputPath)
            
            #''' MANUAL '''
            #ztilde_ridge, beta_ridge, beta_min, beta_max = lib.doRidgeRegression(X, z_rav, self.lambda_par, self.confidence, self.sigma)
            #ztilde_ridge = ztilde_ridge.reshape(zshape)
            #''' Scikit Learn '''
            #ridge_reg = Ridge(alpha = self.lambda_par, fit_intercept=True).fit(X_poly, z_rav)
            #ztilde_sk = ridge_reg.predict(X_poly).reshape(zshape)
            #zarray_ridge = [self.z, ztilde_ridge, ztilde_sk]
            #print('\n')
            #print("Ridge MSE (no CV) - " + str(lib.getMSE(zarray_ridge[0], zarray_ridge[1])) + "; sklearn - " + str(mean_squared_error(zarray_ridge[0], zarray_ridge[2])))
            #print("Ridge R^2 (no CV) - " + str(lib.getR2(zarray_ridge[0], zarray_ridge[1])) + "; sklearn - " + str(ridge_reg.score(X_poly, z_rav)))
            #print('\n')
            #''' Plotting Surfaces '''
            #filename = self.prefix + '_ridge_p' + str(self.poly_degree).zfill(2) + '_n' + npoints_name + '.png'
            #lib.plotSurface(self.x, self.y, zarray_ridge, self.output_dir, filename)
            
            #PlotSurface.PlotFuncs()
            #X = lib.constructDesignMatrix(self.poly_degree)
            # getting predictions
            #ztilde_lin, beta_lin, beta_min, beta_max = lib.doLinearRegression(X, z_rav, self.confidence, self.sigma)
            #ztilde_lin = ztilde_lin.reshape(zshape)
            
            return
        
    '''
    Configuring Neural Network:
    Manual Architecture with corresponding data
    and parameter file
    '''
    def CreateNetwork(self, *args):
        # NNtype
        NNType = self.paramData['type'] #['Classification', 'Regression']
        # If data is of classification type, then process corresponding data set
        if (NNType == 'Classification'):
            # Getting processed data <= it will not be changed in this stage
            # I am using to just identify some key features of the NN
            X_train, X_test, Y_train, Y_test, Y_train_onehot, Y_test_onehot, m, onehotencoder = self.ProcessData()
    
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
            # Seed
            seed = self.paramData['RandomSeed']
            # Optimization Algorithm
            Optimization = self.paramData['Optimization']
            # Batch Size
            BatchSize = self.paramData['BatchSize']
            
            
            return NNType, NNArch, nLayers, \
                   nFeatures, nHidden, nOutput, \
                   epochs, alpha, lmbd, X_train, \
                   X_test, Y_train, Y_test, Y_train_onehot, Y_test_onehot, m,\
                   nInput, seed, onehotencoder, BatchSize, Optimization
                   