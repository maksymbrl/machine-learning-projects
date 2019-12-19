"""
@author: maksymb
"""

import sys
import numpy as np
import pandas as pd
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Subtract
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
    Getting Layer Architecture
    '''
    def GetDenseArchitecture(self, *args):
        # shape of the inputs
        inputShape = args[0] #(self.NInputNeurons, )
        # customizing our model
        inputs = Input(shape = inputShape)
        '''
        we check if the NN is siamese (or triplet), than the last layer's
        activation function should be the same for each layer,
        because we will use sigmoid (or other) when merging layers.
        If activation function is simple Feed Forward Neural Network (FFNN)
        '''
        if self.NNType == 'snn_keras':
            # simply getting activation function for the layer before this one
            lastFunc = self.NNArch[self.NTotalLayers-2]['AF']
            # the neurons in the last layer
            lastNeurons = self.NHiddenNeurons
        elif self.NNType == 'tnn_keras':
            # simply getting activation function for the layer before this one
            lastFunc = self.NNArch[self.NTotalLayers-2]['AF']
            # the neurons in the last layer
            lastNeurons = self.NHiddenNeurons
        elif self.NNType == 'ffnn_keras':
            lastFunc = self.NNArch[self.NTotalLayers-1]['AF']
            lastNeurons = self.NOutputNeurons
        else:
            print('Check Your neural.py!')
            sys.exit()
        # Instantiating the Dense Neural Network
        # adding (connecting) layers
        for layer in range(0, self.NTotalLayers):
            # first layer
            if layer == 0:
                # getting activation function for the first layer
                activation = self.funcs.GetActivation(self.NNArch[layer]['AF'])
                weights = self.funcs.GetWeights(self.weights)
                hidden = Dense(self.NInputNeurons, activation = activation,
                               kernel_initializer=weights, bias_initializer='zeros')(inputs)#,
                               #kernel_regularizer=keras.regularizers.l1_l2(self.lmbd, self.lmbd))(inputs)
                # adding drop-out layer
                hidden = Dropout(rate=self.lmbd)(hidden)
            # last layer
            elif layer == self.NTotalLayers-1:
                # getting activation function for the last layer
                activation = lastFunc
                weights = self.funcs.GetWeights(self.weights)
                outputs = Dense(lastNeurons, activation = activation,
                                kernel_initializer=weights, bias_initializer='zeros')(hidden)#,
                                #kernel_regularizer=keras.regularizers.l1_l2(self.lmbd, self.lmbd))(hidden)
                outputs = Dropout(rate=self.lmbd)(outputs)
            # intermediate layers
            else:
                # getting activation function for the hidden layers
                activation = self.funcs.GetActivation(self.NNArch[layer]['AF'])
                weights = self.funcs.GetWeights(self.weights)
                hidden = Dense(self.NHiddenNeurons, activation=activation,
                               kernel_initializer=weights,  bias_initializer='zeros')(hidden)#,
                               #kernel_regularizer=keras.regularizers.l1_l2(self.lmbd, self.lmbd))(hidden)
                # adding drop-out layer
                hidden = Dropout(rate=self.lmbd)(hidden)
        # building the model
        model = Model(inputs = inputs, outputs = outputs)

        return model
    '''
    Method to Build the Neural Network Model
    (based on the configuration)
    '''
    def BuildModel(self, *args):
        # Checking the type of Network to Build
        if self.NNType == 'ffnn_keras':
            # the shape inputs
            inputShape = (self.NInputNeurons,)
            # getting model architecture
            model = self.GetDenseArchitecture(inputShape)
            #functions = funclib.Functions(self.seed)
            # retrieving loss function
            loss = self.funcs.GetLoss
            # retrieving gradient
            optimizer = self.funcs.GetGradient(self.optimization, self.alpha)
            # compiling model
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            # returning the constructed model
            return model

        # siamese neural network
        elif self.NNType == 'snn_keras':
            # shape of the inputs
            inputShape = (self.NInputNeurons, )
            # customizing our model
            input1 = Input(shape=inputShape)
            input2 = Input(shape=inputShape)
            # getting dense layers -  for siamese the first part is the same
            model = self.GetDenseArchitecture(inputShape)
            # run through first part of network
            run1 = model(input1)
            run2 = model(input2)
            # creating merge layer
            mergeLayer = Lambda(self.funcs.euclidean_distance, output_shape=self.funcs.eucl_dist_output_shape)([run1, run2])
            # getting activation function for the output layer
            activation = self.funcs.GetActivation(self.NNArch[self.NTotalLayers-1]['AF'])
            outputLayer = Dense(self.NOutputNeurons, activation=activation)(mergeLayer)
            model = Model(inputs=[input1, input2], outputs=outputLayer)
            # retrieving loss function
            loss = self.funcs.GetLoss
            # retrieving gradient
            optimizer = self.funcs.GetGradient(self.optimization, self.alpha)
            # compiling model
            model.compile(optimizer=optimizer, loss=loss, metrics=[self.funcs.acc])

            return model

        # triplet neural network
        elif self.NNType == 'tnn_keras':
            # shape of the inputs
            inputShape = (self.NInputNeurons, )
            # customizing our model
            input1 = Input(shape=inputShape)
            input2 = Input(shape=inputShape)
            input3 = Input(shape=inputShape)
            # getting dense layers -  for triplet the first part is the same
            model = self.GetDenseArchitecture(inputShape)
            # run through first part of the network
            pos = model(input1)
            neg = model(input2)
            sam = model(input3)
            # Check distance between positive and sample
            merge_layer1 = Lambda(self.funcs.euclidean_distance, output_shape=self.funcs.eucl_dist_output_shape)([pos,sam])
            # Check distance between negative and sample
            merge_layer2 = Lambda(self.funcs.euclidean_distance, output_shape=self.funcs.eucl_dist_output_shape)([neg,sam])
            #Compare distances
            loss_layer = Subtract()([merge_layer1, merge_layer2])
            model = Model(inputs=[input1, input2, input3], outputs=loss_layer)
            # retrieving loss function
            loss = self.funcs.GetLoss
            # retrieving gradient
            optimizer = self.funcs.GetGradient(self.optimization, self.alpha)
            model.compile(loss=loss, optimizer=optimizer, metrics=[self.funcs.triplet_acc])

            return model

    '''
    Training the model
    '''
    def TrainModel(self, *args):
        # getting model to train
        model = args[0]
        # train the model using keras
        #logger = MyLogger(n=10)
        if self.NNType == 'ffnn_keras':
            # training data
            X1 = args[1]
            X2 = args[2]
            Y1 = args[3]
            Y2 = args[4]
            history = model.fit(X1, Y1,
                                #validation_split=0.5,
                                validation_data=(X2, Y2),
                                batch_size=self.batchSize,
                                epochs=self.epochs,
                                verbose=0)
        elif self.NNType == 'snn_keras':
            pairs_train  = args[1]
            labels_train = args[2]
            pairs_test   = args[3]
            labels_test  = args[4]
            history = model.fit([pairs_train[:, 0], pairs_train[:, 1]], labels_train[:],
                                validation_data=([pairs_test[:, 0], pairs_test[:, 1]], labels_test[:]),
                                batch_size=self.batchSize,
                                epochs=self.epochs,
                                verbose=0)
        elif self.NNType == 'tnn_keras':
            t_train      = args[1]
            train_labels = args[2]
            t_test       = args[3]
            test_labels  = args[4]
            history = model.fit([t_train[:,0], t_train[:,1], t_train[:,2]], train_labels[:],
                                validation_data=([t_test[:,0],t_test[:,1],t_test[:,2]],test_labels[:]),
                                batch_size=self.batchSize,
                                epochs=self.epochs,
                                verbose=0)
        return model, history


