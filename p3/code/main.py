"""
@author: maksymb
"""

# Library imports
import os, sys
import numpy as np
import keras
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import yaml
# libraries for plotting results
import matplotlib.pyplot as plt
import seaborn as sbn
# to calculate time
import time

# importing manually created libraries
import neural, funclib#, random_forest

'''
The main class of the program
'''
class MainPipeline:
    # constructor
    def __init__(self, *args):
        paramFile = args[0]
        # Getting values from Parameter file
        with open(paramFile) as f:
            self.paramData = yaml.load(f, Loader = yaml.FullLoader)
        # creating an output directory
        outputPath = self.paramData['OutputPath']
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)

    '''
    Method to preprocess the data set for main Research Quesrtions
    '''
    def PreProcessingMainData(self, *args):
        return
    '''
    Method to preprocess the data set for Side Research Questions
    '''
    def PreProcessing(self, *args):
        # Random Seed
        seed = self.paramData['RandomSeed']
        # Loss function - we need it to instantiate funclib variable
        loss = self.paramData['Loss']
        # Instantiating object variable from Functions Library
        self.funcs = funclib.Functions(seed, loss)
        '''
        Data preprocessing
        '''
        # getting data from parameter file
        #data = pd.read_csv(self.paramData['dataPath'], delimiter='\s+', encoding='utf-8')
        #print(self.paramData['dataPath'][0])
        # as in Knut's file
        data = pd.read_csv(self.paramData['DataPath'][0])
        meta = pd.read_csv(self.paramData['DataPath'][1], delimiter=r"\s+")
        dt   = data.replace(0, pd.np.nan).dropna(axis=1, how='any').fillna(0).astype(int)
        data = self.funcs.normalize(dt)
        # our target variable
        target_choice="TP"
        tp = meta[target_choice]
        # Simple Feed Forward Neural Network
        if self.paramData['type'] == 'ffnn_keras':
            self.Y = tp.values.reshape(-1, 1)
            self.X_norm = data.to_numpy()
            # Split into training and testing
            self.X_train, self.X_test, y_train, y_test = train_test_split(self.X_norm,
                                                                          self.Y,
                                                                          random_state=seed,
                                                                          test_size=self.paramData['TestSize'])
            y_train_l, y_test_l = self.funcs.set_category(self.Y, y_train, y_test)
            # doing one hot encoding
            oh = OneHotEncoder(sparse=False, categories="auto")
            self.Y_train_onehot = oh.fit_transform(y_train_l)
            self.Y_test_onehot = oh.fit_transform(y_test_l)
        elif self.paramData['type'] == 'snn_keras':
            self.Y = tp.values.reshape(-1, 1)
            # normalizing data
            self.X_norm = data.to_numpy()
            # Split into training and testing
            self.X_train, self.X_test, y_train, y_test = train_test_split(self.X_norm,
                                                                          self.Y,
                                                                          random_state=seed,
                                                                          test_size=self.paramData['TestSize'])
            y_train_l, y_test_l = self.funcs.set_category(self.Y, y_train, y_test)
            oh = OneHotEncoder(sparse=False,categories="auto")
            #Anchors selected by inspection. Serves as positive/negatives for
            #comparison in network. First position is a low valued representative
            #second position is a high valued representative.
            anchors = {"TMP":(data.iloc[10],data.iloc[70])}
            anchors[target_choice] = (data.iloc[15],data.iloc[67])

            # Make pairs and labels of "same" or "different"
            self.pairs_train, \
            self.labels_train, \
            self.pairs_test, \
            self.labels_test = self.funcs.make_anchored_pairs(self.X_train,
                                                   y_train_l,
                                                    self.X_test,
                                                    y_test_l,
                                                    anch = anchors[target_choice])
            # doing one hot encoding
            self.Y_train_onehot = oh.fit_transform(self.labels_train.reshape(-1,1))
            self.Y_test_onehot = oh.fit_transform(self.labels_test.reshape(-1,1))
        elif self.paramData['type'] == 'tnn_keras':
            self.Y = tp.values.reshape(-1, 1)
            # normalizing data
            self.X_norm = data.to_numpy()
            # Split into training and testing
            self.X_train, self.X_test, y_train, y_test = train_test_split(self.X_norm,
                                                                          self.Y,
                                                                          random_state=seed,
                                                                          test_size=self.paramData['TestSize'])
            y_train_l, y_test_l = self.funcs.set_category(self.Y, y_train, y_test)
            oh = OneHotEncoder(sparse=False,categories="auto")
            #Anchors selected by inspection. Serves as positive/negatives for
            #comparison in network. First position is a low valued representative
            #second position is a high valued representative.
            anchors = {"TMP":(data.iloc[10],data.iloc[70])}
            anchors[target_choice] = (data.iloc[15],data.iloc[67])

            # Make pairs and labels of "same" or "different"
            self.pairs_train, \
            self.labels_train, \
            self.pairs_test, \
            self.labels_test = self.funcs.make_anchored_pairs(self.X_train,
                                                         y_train_l,
                                                         self.X_test,
                                                         y_test_l,
                                                         anch = anchors[target_choice])
            #Create triplets for triplet network
            self.triplets_train = self.funcs.make_training_triplets(anchors[target_choice],self.X_train, y_train_l)
            self.triplets_test  = self.funcs.make_training_triplets(anchors[target_choice],self.X_test, y_test_l)
            # doing one hot encoding
            self.Y_train_onehot = y_train_l#oh.fit_transform(self.labels_train.reshape(-1,1))
            self.Y_test_onehot = y_test_l#oh.fit_transform(self.labels_test.reshape(-1,1))



        #print(self.Y_train_onehot)
        '''
        # inputs
        self.X = data.iloc[:, :10].values
        # Scale inputs - applying normalization
        ss = StandardScaler()
        rs = RobustScaler()
        mms = MinMaxScaler()
        self.X_norm = mms.fit_transform(self.X)
        # outputs - binary classification
        self.Y = data.iloc[:, 11].values.reshape(-1, 1)
        onehot = OneHotEncoder(sparse=False, categories="auto")
        self.Y_onehot = onehot.fit_transform(self.Y)
        # splitting data into train and test <= use one hot otherwis eit doesn't learn
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X_norm, self.Y_onehot,
                                                                                test_size=self.paramData['TestSize'],
                                                                                random_state=self.paramData['RandomSeed'])
        '''
        '''
        Creating Network Architecture
        '''
        # Network Type
        NNType = self.paramData['type']
        # Number of Hidden Layers
        NHiddenLayers = self.paramData['NHiddenLayers']
        # Total Number of Layers
        NTotalLayers = NHiddenLayers + 2
        # No of input data <= amount of data in a single column
        Ndata = self.X_train.shape[0] #self.X_norm.shape[0]
        # No of Input Neurons <= amount of variables
        NInputNeurons = self.X_train.shape[1] #self.X_norm.shape[1] #paramData['N']
        # No of hidden neurons
        NHiddenNeurons = self.paramData['NHiddenNeurons']
        # No of output neurons
        NOutputNeurons = self.paramData['NOutputNeurons']
        # Activation Functions
        # activation functions
        aHiddenFunc = self.paramData['HiddenFunc']
        aOutputFunc = self.paramData['OutputFunc']
        # Neural Network Layer Architecture
        NNArch = []
        # Creating NN architecture
        for layer in range(0, NTotalLayers):
            # input layer
            if layer == 0:
                NNArch.append({"LSize": NInputNeurons, "AF": aHiddenFunc})
            # output layer
            elif layer == (NTotalLayers-1):
                NNArch.append({"LSize": NOutputNeurons, "AF": aOutputFunc})
            # hidden layers
            else:
                NNArch.append({"LSize": NHiddenNeurons, "AF": aHiddenFunc})
        # weights
        weights = self.paramData['Weights']
        # epochs to train the algorithm
        epochs = self.paramData['epochs']#1000
        # learning rate
        alpha = self.paramData['alpha'] #0.3
        # regularization parameter
        lmbd = self.paramData['lambda'] #0.001
        # Optimization Algorithm
        optimization = self.paramData['Optimization']
        # batch size
        batchSize = self.paramData['BatchSize']

        NNdata = seed, NNType, NHiddenLayers, NTotalLayers,\
                      Ndata, NInputNeurons, NHiddenNeurons, \
                      NOutputNeurons, aHiddenFunc, aOutputFunc, NNArch,\
                      weights, epochs, alpha, lmbd, optimization, loss, batchSize
        # returning the NN entire data structure
        return NNdata

    # Main Method
    def Run(self, *args):
        # retrieving network data
        NNdata = self.PreProcessing()

        if self.paramData['type'] == 'ffnn_keras':
            # passing parameter file
            print(self.paramData)
            '''
            Getting Network Architecture
            '''
            network = neural.NeuralNetwork(NNdata)
            # passing network architecture and create the model
            model = network.BuildModel()
            # training model
            model, history = network.TrainModel(model, self.X_train, self.X_test, self.Y_train_onehot, self.Y_test_onehot)#self.X_norm, self.Y_onehot)
            test_loss, test_acc = model.evaluate(self.X_test, self.Y_test_onehot)
            print('Test accuracy:', test_acc)

            # Plotting results
            self.funcs.PlotResults(history,
                                   self.paramData['type'],
                                   self.paramData['OutputPath'],
                                   self.paramData['epochs'],
                                   self.paramData['Optimization'],
                                   self.paramData['BatchSize'])

        elif self.paramData['type'] == 'snn_keras':
            '''
            Getting Network Architecture
            '''
            network = neural.NeuralNetwork(NNdata)
            # passing network architecture and create the model
            model = network.BuildModel()
            # training model
            model, history = network.TrainModel(model, self.pairs_train, self.Y_train_onehot, self.pairs_test, self.Y_test_onehot)
            # Plotting results
            self.funcs.PlotResults(history,
                                   self.paramData['type'],
                                   self.paramData['OutputPath'],
                                   self.paramData['epochs'],
                                   self.paramData['Optimization'],
                                   self.paramData['BatchSize'])
        elif self.paramData['type'] == 'tnn_keras':
            '''
            Getting Network Architecture
            '''
            network = neural.NeuralNetwork(NNdata)
            # passing network architecture and create the model
            model = network.BuildModel()
            # training model
            model, history = network.TrainModel(model, self.triplets_train, self.Y_train_onehot, self.triplets_test, self.Y_test_onehot)
            # Plotting results
            self.funcs.PlotResults(history,
                                   self.paramData['type'],
                                   self.paramData['OutputPath'],
                                   self.paramData['epochs'],
                                   self.paramData['Optimization'],
                                   self.paramData['BatchSize'])
        elif self.paramData['type'] == 'xgboost':
            print('Running XGBoost')
        elif self.paramData['type'] == 'rf':
            print('Running Random Forest')

    def Normalize(self, *args):
        x = args[0]
        return (x-np.amin(x))/(np.amax(x)-np.amin(x))

'''
Entry Point of the program
'''
if __name__ == '__main__':
    # Estimate how much time it took for program to work
    startTime = time.time()
    '''
    Configuring Network via Parameter file
    '''
    # Getting parameter file
    paramFile = 'ParameterFile.yaml'
    # Class Object Instantiation - passing
    # configuration from parameter file
    pipe = MainPipeline(paramFile)
    pipe.Run()

    # End time of the program
    endTime = time.time()
    print("-- Program finished at %s sec --" % (endTime - startTime))