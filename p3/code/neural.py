"""
@author: maksymb
"""

import numpy as np
import pandas as pd
import keras
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler

import funclib

"""
Class which constructs the Neural Network
"""
class NeuralNetwork:
    # constructor
    def __init__(self, *args):
        # Getting NN data
        NNdata = args[0]
        # retrieving Network Architecture
        self.seed, self.NNType, self.NHiddenLayers, self.NTotalLayers, \
        self.Ndata, self.NInputNeurons, self.NHiddenNeurons, \
        self.NOutputNeurons, self.aHiddenFunc, self.aOutputFunc, self.NNArch, \
        self.weights, self.epochs, self.alpha, self.lmbd, self.optimization, \
        self.loss, self.batchSize = NNdata

        # Printing current network configuration
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
                                 self.NHiddenLayers,
                                 self.Ndata,
                                 self.NInputNeurons,
                                 self.NHiddenNeurons,
                                 self.NOutputNeurons,
                                 self.NNArch[1]['AF'],
                                 self.NNArch[self.NTotalLayers-1]['AF'],
                                 self.epochs,
                                 self.optimization,
                                 self.alpha,
                                 self.lmbd)))

        # Instantiating object variable from Functions Library
        self.funcs = funclib.Functions(self.seed, self.loss)


    '''
    Method to Build the Neural Network Model
    (based on the configuration)
    '''
    def BuildModel(self, *args):
        #name = args[0]
        # shape of the inputs
        inputShape = (self.NInputNeurons, )
        # customizing our model
        inputs = Input(shape = inputShape)
        # Instantiating the simplest Neural Network
        # adding (connecting) layers
        for layer in range(0, self.NTotalLayers):
            # first layer
            if layer == 0:
                # getting activation function for the first layer
                activation = self.funcs.GetActivation(self.NNArch[layer]['AF'])
                weights = self.funcs.GetWeights(self.weights)
                hidden = Dense(self.NInputNeurons, activation = activation,
                               kernel_initializer=weights, bias_initializer='zeros',
                               kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
            # last layer
            elif layer == self.NTotalLayers-1:
                # getting activation function for the last layer
                activation = self.funcs.GetActivation(self.NNArch[layer]['AF'])
                weights = self.funcs.GetWeights(self.weights)
                outputs = Dense(self.NOutputNeurons, activation = activation,\
                                kernel_initializer=weights, bias_initializer='zeros',
                                kernel_regularizer=keras.regularizers.l2(0.01))(hidden)
            # intermediate layers
            else:
                # getting activation function for the hidden layers
                activation = self.funcs.GetActivation(self.NNArch[layer]['AF'])
                weights = self.funcs.GetWeights(self.weights)
                hidden = Dense(self.NHiddenNeurons, activation=activation, \
                               kernel_initializer=weights,  bias_initializer='zeros',
                               kernel_regularizer=keras.regularizers.l2(0.01))(hidden)
        # building the model
        model = Model(inputs = inputs, outputs = outputs)
        #functions = funclib.Functions(self.seed)
        # retrieving loss function
        loss = self.funcs.GetLoss()
        # retrieving gradient
        optimizer = self.funcs.GetGradient(self.optimization, self.alpha)
        # compiling model
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        # returning the constructed model
        return model

    def TrainModel(self, *args):
        # getting model to train
        model = args[0]
        # training data
        X1 = args[1]
        X2 = args[2]
        Y1 = args[3]
        Y2 = args[4]
        # train the model using keras
        #logger = MyLogger(n=10)
        history = model.fit(X1, Y1,
                            #validation_split=0.5,
                            validation_data=(X2,Y2),
                            batch_size=self.batchSize,
                            epochs=self.epochs,
                            verbose=0)
        return model, history


