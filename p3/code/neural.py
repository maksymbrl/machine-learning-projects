"""
@author: maksymb
"""

import sys
import numpy as np
import pandas as pd
import random
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Subtract
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

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

'''
Manual code for Neural Network
(the code left almost unchanged from the project 2)
'''
class NeuralNetworkML:  #Multiple hidden layers
    def __init__(
            self,
            X_data,
            Y_data,
            trainingShare=0.5,
            n_hidden_layers=2,
            n_hidden_neurons=[24,12],
            n_categories=10,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0,
            fixed_LR=False,
            method="classification",
            activation="sigmoid",
            seed = 1):

        self.seed = seed

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.trainingShare = trainingShare
        self.method = method
        self.split_data = self.SplitData(self.X_data_full, self.Y_data_full, self.trainingShare)
        if self.method=="classification":
            self.XTrain = self.split_data[0].toarray()
            self.XTest = self.split_data[1].toarray()
        else:
            self.XTrain = self.split_data[0]
            self.XTest = self.split_data[1]
        self.yTrain = self.split_data[2]
        self.yTest = self.split_data[3]

        self.n_inputs = self.XTrain.shape[0]
        self.n_features = self.XTrain.shape[1]
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.fixed_LR = fixed_LR
        self.activation = activation

        self.create_biases_and_weights()
        self.accuracy_list = []
        self.cost_list = []
        self.models = []
        self.current_epoch = 0
        self.w_dict = {}

    def create_biases_and_weights(self):
        for n in range(self.n_hidden_layers):
            exec("self.n_hidden_neurons_" + str(eval("n + 1")) + "=" + str(eval("self.n_hidden_neurons[n]")))
        for n in range(self.n_hidden_layers):
            if n==0:
                exec("self.hidden_weights_1 = np.random.randn(self.n_features, self.n_hidden_neurons_1)")
                exec("self.hidden_bias_1 = np.zeros((1, self.n_hidden_neurons_1)) + 0.01")
            else:
                exec("self.hidden_weights_" + str(eval("n + 1")) + "= np.random.randn(self.n_hidden_neurons_" + str(eval("n")) + ", self.n_hidden_neurons_" + str(eval("n+1")) + ")")
                exec("self.hidden_bias_" + str(eval("n + 1")) + "= np.zeros((1, self.n_hidden_neurons_" + str(eval("n + 1")) + ")) + 0.01")
            exec("self.output_weights = np.random.randn(self.n_hidden_neurons_" + str(eval("self.n_hidden_layers")) + ", self.n_categories)")
            exec("self.output_bias = np.zeros((1, self.n_categories)) + 0.01")

    def act(self, x):
        if self.activation=="sigmoid":
            return self.SigmoidFunction(x)
        elif self.activation=="ELU":
            return self.ELU(x, alpha=0.01)
        elif self.activation=="LeakyReLU":
            return self.LeakyReLU(x, alpha=0.01)

    def feed_forward(self):
        for n in range(self.n_hidden_layers):
            if n==0:
                self.z_h_1 = np.matmul(self.XTrain_batch, self.hidden_weights_1) + self.hidden_bias_1
                self.a_h_1 = self.act(self.z_h_1)
            else:
                exec("self.z_h_" + str(eval("n + 1")) + " = np.matmul(self.a_h_" + str(eval("n")) + ", self.hidden_weights_" + str(eval("n + 1")) + ") + self.hidden_bias_" + str(eval("n + 1")))
                exec("self.a_h_" + str(eval("n + 1")) + " = self.act(self.z_h_" + str(eval("n + 1")) + ")")
        exec("self.z_o = np.matmul(self.a_h_" + str(eval("self.n_hidden_layers")) + ", self.output_weights) + self.output_bias")
        self.probabilities = self.LogRegPredict(self.z_o)
        self.a_o = self.act(self.z_o)

    def feed_forward_out(self, X):
        for n in range(self.n_hidden_layers):
            if n==0:
                z_h_1 = np.matmul(X, self.hidden_weights_1) + self.hidden_bias_1
                a_h_1 = self.act(z_h_1)
            # feed-forward for output
            else:
                exec("z_h_" + str(eval("n + 1")) + " = np.matmul(a_h_" + str(eval("n")) + ", self.hidden_weights_" + str(eval("n + 1")) + ") + self.hidden_bias_" + str(eval("n + 1")))
                exec("a_h_" + str(eval("n + 1")) + " = self.act(z_h_" + str(eval("n + 1")) + ")")
        z_o=eval("np.matmul(a_h_" + str(eval("self.n_hidden_layers")) + ", self.output_weights) + self.output_bias")
        a_o = self.act(z_o)
        if self.method=="classification":
            probabilities = self.LogRegPredict(z_o)
        elif self.method=="regression":
            yPred = z_o
            probabilities = z_o
        return probabilities

    def LogRegPredict(self, z_o):
        yPred = self.SigmoidFunction(z_o)
        for i in range(0, yPred.shape[0], 1):
            if yPred[i] <= 0.5:
                yPred[i] = 0
            else:
                yPred[i] = 1
        return yPred

    def backpropagation(self):
        w_list = [self.hidden_weights_1, self.hidden_bias_1]
        for n in range(self.n_hidden_layers - 1, -1, -1):
            if n + 1 == self.n_hidden_layers:
                if self.method=="regression":
                    error_output = (self.a_o - self.yTrain_batch) #Cost function: derivative of mean squared error
                else:
                    error_output = (self.a_o - self.yTrain_batch) * self.a_o * (1 - self.a_o) #Cost function: derivative of cross-entropy
                if self.n_hidden_layers==1:
                    self.error_hidden_1 = np.matmul(error_output, self.output_weights.T) * self.a_h_1 * (1 - self.a_h_1)
                    self.output_weights_gradient = np.matmul(self.a_h_1.T, error_output)
                    self.output_bias_gradient = np.sum(error_output, axis=0)
                    self.hidden_weights_gradient_1 = np.matmul(self.XTrain_batch.T, self.error_hidden_1)
                    self.hidden_bias_gradient_1 = np.sum(self.error_hidden_1, axis=0)
                else:
                    exec("self.error_hidden_" + str(eval("self.n_hidden_layers")) + " = np.matmul(error_output, self.output_weights.T) * self.a_h_" + str(eval("self.n_hidden_layers")) + " * (1 - self.a_h_" + str(eval("self.n_hidden_layers")) + ")")
                    exec("self.hidden_weights_gradient_" + str(eval("n + 1")) + " = np.matmul(self.a_h_" + str(eval("n")) + ".T, self.error_hidden_" + str(eval("n + 1")) + ")")
                    exec("self.hidden_bias_gradient_" + str(eval("n + 1")) + " = np.sum(self.error_hidden_" + str(eval("n + 1")) + ", axis=0)")
                    exec("self.output_weights_gradient = np.matmul(self.a_h_" + str(eval("n + 1")) + ".T, error_output)")
                self.output_bias_gradient = np.sum(error_output, axis=0)
            elif n > 0:
                exec("self.error_hidden_" + str(eval("n+1")) + " = np.matmul(self.error_hidden_" + str(eval("n + 2")) + ", self.hidden_weights_" + str(eval("n + 2")) + ".T) * self.a_h_" + str(eval("n+1")) + " * (1 - self.a_h_" + str(eval("n+1")) + ")")
                exec("self.hidden_weights_gradient_" + str(eval("n + 1")) + " = np.matmul(self.a_h_" + str(eval("n")) + ".T, self.error_hidden_" + str(eval("n + 1")) + ")")
                exec("self.hidden_bias_gradient_" + str(eval("n + 1")) + " = np.sum(self.error_hidden_" + str(eval("n + 1")) + ", axis=0)")
            else:
                if self.n_hidden_layers == 1:
                    self.error_hidden_1 = np.matmul(error_output, self.output_weights.T) * self.a_h_1 * (1 - self.a_h_1)
                else:
                    self.error_hidden_1 = np.matmul(self.error_hidden_2, self.hidden_weights_2.T) * self.a_h_1 * (1 - self.a_h_1)
                    self.hidden_weights_gradient_1 = np.matmul(self.XTrain_batch.T, self.error_hidden_1)
                    self.hidden_bias_gradient_1 = np.sum(self.error_hidden_1, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            for n in range(self.n_hidden_layers - 1, -1, -1):
                exec("self.hidden_weights_gradient_" + str(eval("n + 1")) + " += self.lmbd * self.hidden_weights_" + str(eval("n + 1")))

        self.output_weights -= self.eta * self.output_weights_gradient/self.batch_size
        self.output_bias -= self.eta * self.output_bias_gradient/self.batch_size

        for n in range(self.n_hidden_layers - 1, -1, -1):
            exec("self.hidden_weights_" + str(eval("n + 1")) + " -= self.eta * self.hidden_weights_gradient_" + str(eval("n + 1")) + "/self.batch_size")
            exec("self.hidden_bias_" + str(eval("n + 1")) + " -= self.eta * self.hidden_bias_gradient_" + str(eval("n + 1")) + "/self.batch_size")

            w_list.append(eval("self.hidden_weights_" + str(eval("n + 1"))))
            w_list.append(eval("self.hidden_bias_" + str(eval("n + 1"))))

        w_list.append(self.output_weights)
        w_list.append(self.output_bias)
        self.w_dict[self.current_epoch] = w_list

    def RescaleOutputToOriginal(self, z_old, z_new): #z_old is the vector of predicted values to rescale to the original scale of response values (z_new)
        max_old = max(z_old)
        min_old = min(z_old)
        max_new = max(z_new)
        min_new = min(z_new)
        z_rescaled = z_old.copy()
        for i in range(len(z_old)):
            z_rescaled[i] = (max_new - min_new)/(max_old - min_old) * (z_old[i] - max_old) + max_new
            #value_new = (max_new - min_new)/(max_old - min_old) * (value_old - max_old) + max_new
        return z_rescaled

    def model_prediction(self, X, iter):
        m_w = self.w_dict[iter-1] #Model weights
        m_o_w = m_w[-2] #Model output weights
        m_o_b = m_w[-1] #Model output bias
        m_h_w_1 = m_w[0] #Model hidden weights 1
        m_h_b_1 = m_w[1] #Model hidden bias 1
        if self.n_hidden_layers > 1:
            m_h_w_n = m_w[-4:0:-2]#[-1::-1] #Model hidden weights n
            m_h_b_n = m_w[-3:1:-2]#[-1::-1] #Model hidden bias n
            for i in range(self.n_hidden_layers-1):
                exec("m_h_w_" + str(eval("i + 2")) + " = m_h_w_n[i+1]")
                exec("m_h_b_" + str(eval("i + 2")) + " = m_h_b_n[i+1]")

        for n in range(self.n_hidden_layers):
            if n==0:
                m_z_h_1 = np.matmul(X, m_h_w_1) + m_h_b_1
                m_a_h_1 = self.act(m_z_h_1)
            # feed-forward for output
            else:
                exec("m_z_h_" + str(eval("n + 1")) + " = np.matmul(m_a_h_" + str(eval("n")) + ", m_h_w_" + str(eval("n + 1")) + ") + m_h_b_" + str(eval("n + 1")))
                exec("m_a_h_" + str(eval("n + 1")) + " = self.act(m_z_h_" + str(eval("n + 1")) + ")")
        m_z_o=eval("np.matmul(m_a_h_" + str(eval("self.n_hidden_layers")) + ", self.output_weights) + self.output_bias")
        m_a_o = self.act(m_z_o)
        if self.method=="classification":
            probabilities = self.LogRegPredict(m_z_o)
        elif self.method=="regression":
            yPred = self.RescaleOutputToOriginal(m_z_o, self.Y_data_full)
            probabilities = self.RescaleOutputToOriginal(m_z_o, self.Y_data_full)
        return probabilities

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        #return np.argmax(probabilities, axis=1)
        return probabilities

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        t0, t1 = 5, 500
        self.accuracy_list.append(self.accuracy(self.yTest, self.predict(self.XTest)))
        for i in range(self.epochs):
            self.current_epoch = i
            self.shuffled_data = self.shuffle(self.XTrain, self.yTrain) # Rows for XTrain, yTrain are shuffled for each epoch.
            self.XTrain_shuffled = self.shuffled_data[0]
            self.yTrain_shuffled = self.shuffled_data[1]
            for batch in range(int(self.XTrain.shape[0]/self.batch_size)):
                self.XTrain_batch = self.XTrain_shuffled[self.batch_size * batch: self.batch_size * (batch + 1), :] #Minibatch training data
                self.yTrain_batch = self.yTrain_shuffled[self.batch_size * batch: self.batch_size * (batch + 1)] #Minibatch training data

                if self.fixed_LR==False:
                    t = i*int(self.XTrain.shape[0]/self.batch_size) + batch #Variable learning rate
                    self.eta = self.step_length(t, t0, t1) #Variable learning rate
                #print(self.eta)

                self.feed_forward()
                self.backpropagation()
            self.accuracy_list.append(self.accuracy(self.yTest, self.predict(self.XTest)))
            print("Epoch " + str(i + 1) + " completed")

    def shuffle(self, XTrain, yTrain):
        random.seed(self.seed)
        n_rows = list(range(0, XTrain.shape[0], 1))
        random.shuffle(n_rows)
        XTrain_post_shuffle = XTrain[n_rows,:]
        yTrain_post_shuffle = yTrain[n_rows]
        return XTrain_post_shuffle, yTrain_post_shuffle

    def step_length(self,t,t0,t1):
        return t0/(t+t1)

    def SplitData(self, X, y, trainingShare=0.5):
        seed  = 1
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values
        XTrain, XTest, yTrain, yTest = train_test_split(X, y,
                                                        train_size=trainingShare,
                                                        test_size = 1-trainingShare,
                                                        random_state=seed)
        return XTrain, XTest, yTrain, yTest

    def accuracy(self, yTest, yPred):
        if self.method=="classification":
            return  (yTest.flatten() == yPred.flatten()).sum()/len(yTest.flatten())
        if self.method=="regression":
            self.models.append(self.predict(self.X_data_full))
            return np.mean((self.yTest - self.predict(self.XTest))**2)

    def SigmoidFunction(self, x):
        sigma_fn = np.vectorize(lambda x: 1/(1+np.exp(-x)))
        return 1/(1+np.exp(-x))

    def LogRegPredict(self, z_o):
        yPred = self.SigmoidFunction(z_o)
        for i in range(0, yPred.shape[0], 1):
            if yPred[i] <= 0.5:
                yPred[i] = 0
            else:
                yPred[i] = 1
        return yPred

    def ELU(self, x, alpha=0.01):
        ao = x
        for i in range(0, x.shape[0], 1):
            for j in range(0, x.shape[1], 1):
                if x[i,j] < 0:
                    ao[i,j] = alpha*(np.exp(x[i,j]) - 1)
        return ao

    def LeakyReLU(self, x, alpha=0.01):
        ao = x
        for i in range(0, x.shape[0], 1):
            for j in range(0, x.shape[1], 1):
                if x[i,j] <= 0:
                    ao[i,j] = alpha*x[i,j]
        return ao
