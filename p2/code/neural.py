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
from sklearn.compose import ColumnTransformer

# One Hot Encoder from Keras
from keras.utils import to_categorical

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
        
        #fig = plt.figure(figsize=(20, 15), facecolor='white')
        #axes = [fig.add_subplot(3, 3, i) for i in range(1, len(labels))]
        #for i in range(0, 3):
        #    for j in range
        #    axes[i] = sbn.countplot(x=label, hue="default", data=subset, palette="Blues", ax=axes[j, i])
        
        
        fig, axes = plt.subplots(3,3, figsize=(20,15), sharey = 'all', facecolor='white')
        #fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
        axes = axes.flatten()
        #object_bol = df.dtypes == 'object'
        fig.suptitle('FREQUENCY OF CATEGORICAL VARIABLES (BY TARGET)')
        for axe, catplot in zip(axes, labels):
            #sbn.countplot(y=catplot, data=subset, ax=ax)#, order=np.unique(subset.values))
            sbn.countplot(x=catplot, hue="default", data = subset, palette = "cool", ax=axe)
        
        # passing data to further processing
        return data    
        
    '''
    Working on Data
    '''
    def ProcessData(self, *args):
        # getting data
        data = self.PrepareData()
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
        #print(X_train)
        #print(X_train)
        #print(Y_train)
        # One-hot's of the target vector
        Y_train_onehot = to_categorical(Y_train)#onehotencoder.fit_transform(Y_train.reshape(len(Y_train), -1))
        Y_test_onehot =  to_categorical(Y_test)#onehotencoder.fit(Y_test.reshape(len(Y_test), -1))
        
        print("Y_train shape", np.shape(Y_train))
        
        #print(Y_train_onehot)
        #train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))
 
        #test_y = enc.transform(test_y.reshape(len(test_y), -1))
        
        #print(Y_train_onehot)
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
        
        return X_train, X_test, Y_train, Y_test, Y_train_onehot, Y_test_onehot, m, onehotencoder#Y_train, Y_test, m
    
    '''
    Configuring Neural Network:
    Manual Architecture with corresponding data
    and parameter file
    '''
    def CreateNetwork(self, *args):
        # Getting processed data <= it will not be changed in this stage
        # I am using to just identify some key features of the NN
        X_train, X_test, Y_train, Y_test, Y_train_onehot, Y_test_onehot, m, onehotencoder = self.ProcessData()
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
        # Seed
        seed = self.paramData['RandomSeed']
        # Batch Size
        BatchSize = self.paramData['BatchSize']
        
        
        return NNType, NNArch, nLayers, \
               nFeatures, nHidden, nOutput, \
               epochs, alpha, lmbd, X_train, \
               X_test, Y_train, Y_test, Y_train_onehot, Y_test_onehot, m,\
               nInput, seed, onehotencoder, BatchSize
    
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
        # random seed, to make the same random number each time
        np.random.seed(seed)
        #self.Y = args[10]
        #self.m = args[11]
        # Only for printing purpose
        
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
                                 #latex('$\\alpha$'),
                                 self.alpha,
                                 self.lambd)))
    
        #print(self.NNArch[0]["LSize"])
                
        #print("nInput", nInput)
        #nFeatures = X_train.shape[1] # <= the amount of variables
        #print('nFeatures', nFeatures)
    # 
    def InitParams(self, *args):
        # biases and weights for hidden and output layers
        # dictionary to contain all parameters for each layer
        # (i.e. "W1", "b1", ..., "WL", "bL", except inpur one)
        modelParams = {}
        for l in range(1, self.nLayers):
            # weights for each layer (except input one)
            #print(self.nInputNeurons, self.nHiddenNeurons)
            #print(self.NNArch[l-1]["LSize"])
            modelParams['W' + str(l)] = np.random.randn(self.NNArch[l-1]["LSize"], self.NNArch[l]["LSize"]) #/ np.sqrt(self.NNArch[l-1]["LSize"])
            #print(np.shape(modelParams['W' + str(l)]))
            # biases for each layer (except input one)
            #modelParams['b' + str(l)] = np.zeros((self.NNArch[l]["LSize"], self.nOutputNeurons)) + 0.01
            modelParams['b' + str(l)] = np.zeros((self.nInput, 
                        self.NNArch[l]["LSize"])) + 0.0001
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
        
    
    # Creating a list of mini-batches
    def CreateMiniBatches(self, *args):
        X = args[0]
        Y = args[1]
        batchSize = args[2]
        
        miniBatches = [] 
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
        return miniBatches 
    
    # Train Neural Network
    def TrainNetworkGD(self, *args):
        Xtrain = args[0]
        Ytrain = args[1]
        #print(Ytrain)
        m = args[2]
        print(self.CreateMiniBatches(Xtrain, Ytrain, self.BatchSize))
        
        # Initialising parameters
        modelParams = self.InitParams()
        costs =  []
        if self.NNType == 'Classification':
            # Running Optimisation Algorithm
            for epoch in range(1, self.epochs+1, 1):
                # Propagating Forward
                A, Z = self.DoFeedForward(Xtrain, modelParams)
                # Calculating cost Function
                J = funclib.CostFuncs().CallNNLogistic(Ytrain,\
                                     A[str(self.nLayers-1)],\
                                     modelParams,\
                                     self.nLayers,\
                                     m,\
                                     self.lambd)
                #print(J)
                # Back propagation - gradients
                dJ = self.DoBackPropagation(Ytrain, A, Z, modelParams, m)
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
        
        # returning set of optimal model parameters
        #return modelParams, costs
    