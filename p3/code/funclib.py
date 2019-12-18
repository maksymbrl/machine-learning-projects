import os, sys
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense

import time

'''
This class iof used to store all functions
necessary to process data and run neural network
and/or XGBoost
'''
class Functions:
    # constructor
    def __init__(self, *args):
        # passing the random seed
        self.seed = args[0]
        self.lossName = args[1]
        np.random.seed(self.seed)

    # Initialising Weights
    def GetWeights(self, *args):
        # the value from parameter file
        name = args[0]
        # random seed
        #seed = args[1]
        if name == 'norm':
            # random normal
            return keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=self.seed)
        elif name == 'unif':
            # random uniform
            return keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=self.seed)
        elif name == 'xnorm':
            # Xavier normal
            return keras.initializers.glorot_normal(seed=self.seed)
        elif name == 'xunif':
            # Xavier uniform
            return keras.initializers.glorot_uniform(seed=self.seed)
        elif name == 'hnorm':
            # He normal
            return keras.initializers.he_normal(seed=self.seed)
        elif name == 'hunif':
            # He uniform
            return keras.initializers.he_uniform(seed=self.seed)
        else:
            print("Check your weights!")
            sys.exit()

    # Activation Functions
    def GetActivation(self, *args):
        # name of activation functions
        name = args[0]
        #
        #x = args[1]
        if name == 'linear':
            # Linear
            return keras.activations.linear#(x)
        elif name == 'exp':
            # Exponential
            return keras.activations.exponential#(x)
        elif name == 'tanh':
            # Tanh
            return keras.activations.tanh#(x)
        elif name == 'sigmoid':
            # Sigmoid
            return keras.activations.sigmoid#(x)
        elif name == 'hsigmoid':
            # Hard Sigmoid
            return keras.activations.hard_sigmoid#(x)
        elif name == 'softmax':
            # Softmax
            return keras.activations.softmax#(x, axis=-1)
        elif name == 'softplus':
            # Softplus
            return keras.activations.softplus#(x)
        elif name == 'softsign':
            # Softsign
            return keras.activations.softsign#(x)
        elif name == 'relu':
            # ReLU
            return keras.activations.relu#(x, alpha=0.0, max_value=None, threshold=0.0)
        elif name == 'elu':
            # eLU
            return keras.activations.elu#(x, alpha=1.0)
        elif name == 'selu':
            # SeLU
            return keras.activations.selu#(x)
        else:
            print('Check activation function!')
            sys.exit()

    # Regularization (L1, L2 and combined to avoid overfitting)
    def GetRegularizer(self, *args):
        name = args[0]

        if name == 'l1':
            return keras.regularizers.l1(0.)
        elif name == 'l2':
            return keras.regularizers.l2(0.)
        elif name == 'l1l2':
            return keras.regularizers.l1_l2(l1=0.01, l2=0.01)
        else:
            print('Check Regularization')
            sys.exit()

    # Optimization algorithms (to compile and run the model),
    # i.e. various gradients methods
    def GetGradient(self, *args):
        # the name of the method
        name = args[0]
        # learning rate
        alpha = args[1]
        if name == 'sgd':
            # Stochastic Gradient Descent - includes momentum and support for Nesterov momentum
            return keras.optimizers.SGD(lr=alpha, momentum=0.0, nesterov=False)
            # Nesterov
        elif name == 'nesterov':
            return keras.optimizers.SGD(lr=alpha, momentum=0.0, nesterov=True)
        elif name == 'rmsprop':
            # RMSProp
            return keras.optimizers.RMSprop(lr=alpha, rho=0.9)
        elif name == 'adagrad':
            # Adagrad
            return keras.optimizers.Adagrad(lr=alpha)
        elif name == 'adadelta':
            # Adadelta
            return keras.optimizers.Adadelta(lr=alpha, rho=0.95)
        elif name == 'adam':
            # Adam
            return keras.optimizers.Adam(lr=alpha, beta_1=0.9, beta_2=0.999, amsgrad=False)
        elif name == 'adamax':
            # Adamax
            return keras.optimizers.Adamax(lr=alpha, beta_1=0.9, beta_2=0.999)
        elif name == 'nadam':
            # Nadam
            return keras.optimizers.Nadam(lr=alpha, beta_1=0.9, beta_2=0.999)
        else:
            print('Check optimization Methods!')
            sys.exit()

    # Getting Loss function
    def GetLoss(self, *args):
        name = self.lossName
        #name   = args[0]
        #y_true = args[0]
        #y_pred = args[1]
        # Getting the correct loss function
        if name == 'mse':
            # Mean Squared Error (MSE)
            return keras.losses.mean_squared_error#(y_true, y_pred)
        elif name == 'mae':
            # Mean Absolute Error (MAE)
            return keras.losses.mean_absolute_error#(y_true, y_pred)
        elif name == 'mape':
            # Mean Absolute Percentage Error (MAPE)
            return keras.losses.mean_absolute_percentage_error#(y_true, y_pred)
        elif name == 'msle':
            # Mean Squared Logarithmic Error (MSLE)
            return keras.losses.mean_squared_logarithmic_error#(y_true, y_pred)
        elif name == 'hinge':
            # Hinge
            return keras.losses.hinge#(y_true, y_pred)
        elif name == 'shinge':
            # Squared Hinge
            return keras.losses.squared_hinge#(y_true, y_pred)
        elif name == 'chinge':
            # Categorical Hinge
            return keras.losses.categorical_hinge#(y_true, y_pred)
        elif name == 'logcosh':
            # LogCosh
            return keras.losses.logcosh#(y_true, y_pred)
        #elif name == 'huber':
        #    # Huber Loss
        #    return keras.losses.huber_loss(y_true, y_pred, delta=1.0)
        elif name == 'categorical':
            # Categorical Cross Entropy
            return keras.losses.categorical_crossentropy#(y_true, y_pred)#, from_logits=False, label_smoothing=0)
        elif name == 'sparse':
            # Sparse Categorical Cross Entropy
            return keras.losses.sparse_categorical_crossentropy#(y_true, y_pred)#, from_logits=False, axis=-1)
        elif name == 'binary':
            # Binary Cross Entropy
            return keras.losses.binary_crossentropy#(y_true, y_pred)#, from_logits=False, label_smoothing=0)
        elif name == 'kullback':
            # Kullback Leibler Divergence
            return keras.losses.kullback_leibler_divergence#(y_true, y_pred)
        elif name == 'poisson':
            # Poisson
            return keras.losses.poisson#(y_true, y_pred)
        elif name == 'proximity':
            # Cosine Proximity
            return keras.losses.cosine_proximity#(y_true, y_pred)#, axis=-1)
        elif name == 'contrasive':
            '''Contrastive loss from Hadsell-et-al.'06
            http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
            '''
            margin = 1
            square_pred = keras.backend.square(y_pred)
            margin_square = keras.backend.square(keras.backend.maximum(margin - y_pred, 0))
            return keras.backend.mean(y_true * square_pred + (1 - y_true) * margin_square)
        else:
            print("Check Loss function!")
            sys.exit()
