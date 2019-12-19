import os, sys
import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import scale
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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
    def GetLoss(self, y_true, y_pred):
        name = self.lossName
        # Getting the correct loss function
        if name == 'mse':
            # Mean Squared Error (MSE)
            return keras.losses.mean_squared_error(y_true, y_pred)
        elif name == 'mae':
            # Mean Absolute Error (MAE)
            return keras.losses.mean_absolute_error(y_true, y_pred)
        elif name == 'mape':
            # Mean Absolute Percentage Error (MAPE)
            return keras.losses.mean_absolute_percentage_error(y_true, y_pred)
        elif name == 'msle':
            # Mean Squared Logarithmic Error (MSLE)
            return keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
        elif name == 'hinge':
            # Hinge
            return keras.losses.hinge(y_true, y_pred)
        elif name == 'shinge':
            # Squared Hinge
            return keras.losses.squared_hinge(y_true, y_pred)
        elif name == 'chinge':
            # Categorical Hinge
            return keras.losses.categorical_hinge(y_true, y_pred)
        elif name == 'logcosh':
            # LogCosh
            return keras.losses.logcosh(y_true, y_pred)
        #elif name == 'huber':
        #    # Huber Loss
        #    return keras.losses.huber_loss(y_true, y_pred, delta=1.0)
        elif name == 'categorical':
            # Categorical Cross Entropy
            return keras.losses.categorical_crossentropy(y_true, y_pred)#, from_logits=False, label_smoothing=0)
        elif name == 'sparse':
            # Sparse Categorical Cross Entropy
            return keras.losses.sparse_categorical_crossentropy(y_true, y_pred)#, from_logits=False, axis=-1)
        elif name == 'binary':
            # Binary Cross Entropy
            return keras.losses.binary_crossentropy(y_true, y_pred)#, from_logits=False, label_smoothing=0)
        elif name == 'kullback':
            # Kullback Leibler Divergence
            return keras.losses.kullback_leibler_divergence(y_true, y_pred)
        elif name == 'poisson':
            # Poisson
            return keras.losses.poisson(y_true, y_pred)
        elif name == 'proximity':
            # Cosine Proximity
            return keras.losses.cosine_proximity(y_true, y_pred)#, axis=-1)
        elif name == 'contrastive':
            #This is the contrastive loss from Hadsell et al.
            #If y_true= 1, meaning that the inputs come from the same category
            #then the square of y_pred is returned and the network
            #will try to minimize y_pred
            #If y_true=0, meaning different categories, the function
            #returns, if the difference beween margin and ypred > 0,
            #the square of this difference. Thus the network will try to maximize
            #y_pred
            margin = 0.9
            square_pred = K.square(y_pred)
            margin_square = K.square((K.maximum(margin - y_pred, 0)))
            return K.mean(y_true*square_pred + (1 - y_true) * margin_square)
        elif name == 'triplet':
            #Function is FaceNets triplet loss. Y_pred is the difference
            #between the positive distance layer and the negative distance layer
            margin = 0.9
            return K.mean(K.maximum(y_pred + margin, 0))
        else:
            print("Check Loss function!")
            sys.exit()

    '''
    Method used for plotting (in Neural Networks with Keras)
    '''
    def PlotResultsKeras(self, *args):
        # getting the history to plot
        history = args[0]
        # getting network type
        type = args[1]
        # output path were to save figures
        outputPath = args[2]
        # epochs
        epochs = args[3]
        # optimization algorothm used
        optimizer = args[4]
        # batch size
        batch = args[5]
        #filename = outputPath + '/'+ 'logreg_costs_e' + str(epochs).zfill(4)+'.png'
        if type == 'ffnn_keras':
            name = 'Feed Forward Neural Network'
            metrics = ['acc', 'val_acc']
        elif type == 'snn_keras':
            name = 'Siamese Neural Network'
            metrics = ['acc', 'val_acc']
        elif type == 'tnn_keras':
            name = 'Triplet Neural Network'
            metrics = ['triplet_acc', 'val_triplet_acc']
        else:
            print('Something wrong with the plots!')
            sys.exit()
        # list all data in history
        print(history.history.keys())
        # summarizing history for accuracy and loss
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(history.history[metrics[0]])
        ax1.plot(history.history[metrics[1]])
        #ax1.set_title("Accuracy")
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        #ax2.set_title("Loss")
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epochs')
        # plot title
        fig.suptitle(name, fontsize = 16)
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        '''
        Saving figure
        '''
        filename = outputPath + '/'+ type + '_e' + str(epochs).zfill(4)+ '_l'+ self.lossName + '_o' + str(optimizer) + '_b'+str(batch)+'.png'
        #print(filename)
        fig.savefig(filename)

    '''
    Normalizing using MinMaxScaler method
    '''
    def normalize(self, x):
        return (x - np.amin(x)) / (np.amax(x) - np.amin(x))

    # accuracy for siamese neural network
    def acc(self, y_true, y_pred):
        #Accuracy function for use with siames twin network
        ones = K.ones_like(y_pred)
        return K.mean(K.equal(y_true, ones - K.clip(K.round(y_pred), 0, 1)), axis=-1)

    # accuracy for triplet neural network
    def triplet_acc(self, y_true,y_pred):
        return K.mean(K.less(y_pred,0))

    def set_category(self, data, train=None, test=None):
        # The creation of bins/categories  aims to create to categories from
        # the data set, seperated by the mean. The digitize function
        # of numpy returns an array with 1,2,3....n as labels for each of n
        # bins. The min, max cut off are  chosen to be larger/smaller than
        # max min values of the data
        bins = np.array([0, data.mean(), 100])

        if train is None:
            temp = np.digitize(data, bins)
            return temp - 1
        train_labels = np.digitize(train, bins)
        test_labels = np.digitize(test, bins)
        return train_labels - 1, test_labels - 1

    def make_anchored_pairs(self, data, target,test_data,test_target, anch):
        '''
        This function returns anchored of data points for comparison
        in siamese oneshot twin network
        '''
        #create lsits to store pairs and labels to return
        pairs = []
        labels = []
        test_pairs = []
        test_labels = []
        #Create all possible training pairs where one element is an anchor
        for i, a in enumerate(anch):
            x1 = a
            for index in range(len(data)):
                x2 = data[index]
                labels += [1] if target[index] == i else [0]
                pairs += [[x1,x2]]
                print(len(labels))
            for index in range(len(test_data)):
                x2 = test_data[index]
                test_labels += [1] if test_target[index] == i else [0]
                test_pairs += [[x1,x2]]

        return np.array(pairs),np.array(labels),np.array(test_pairs),np.array(
            test_labels)

    def make_training_triplets(self, anchors,samples,labels):
        #Creates triplets for siamese triplet network.
        triplets = []
        for s,l in zip(samples,labels):
            positive = anchors[0] if l==0 else anchors[1]
            negative = anchors[0] if l==1 else anchors[1]
            triplets += [[positive,negative,s]]
        return np.array(triplets)

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)


    def euclidean_distance(self, vectors):
        x, y = vectors
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    '''
    Plot Results for XGBoost
    '''
    def PlotResultsXGBoost(self, *args):
        # getting the inputs
        CDOM = args[0]
        CDOM_sorted = args[1]
        X_CDOM_diag_mesh = args[2]
        CDOM_pred_fine_mesh = args[3]
        CDOM_pred = args[4]
        outputPath = args[5]
        y_pred = args[6]
        y_pred_train = args[7]
        MSE_per_epoch = args[8]
        y_train = args[9]
        y_test = args[10]
        XGboost_best_model_index = args[11]

        # Making 3d plots
        fontsize = 10
        # Plot Bray distance by CDOM
        x1 = (CDOM_sorted.loc[:,"x"])
        x2 = x1.copy()
        x1, x2 = np.meshgrid(x1,x2)
        CDOM.mesh[x1-x2==0] = np.nan

        fig = plt.figure(figsize=plt.figaspect(0.25))#figsize=(15, 5))#figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.set_title("BCC Bray distances by sites' CDOM", fontsize=fontsize)
        #plt.subplots_adjust(left=0, bottom=0, right=2, top=2, wspace=0, hspace=0)
        ax.view_init(elev=30.0, azim=300.0)
        surf = ax.plot_trisurf(CDOM.loc[:,"CDOM.x1"], CDOM.loc[:,"CDOM.x2"], CDOM.loc[:,"ASV.dist"],
                               cmap='viridis', edgecolor='none')
        # Customize the z axis.
        ax.set_zlim(0.3, 1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        ax.tick_params(labelsize=8)
        ax.set_zlabel(zlabel="Bray distance")
        ax.set_ylabel(ylabel="CDOM site 2")
        ax.set_xlabel(xlabel="CDOM site 1")

        # Set up the axes for the second plot
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.set_title("Predicted BCC Bray distances by sites' CDOM, \n CDOM 0.01 step meshgrid", fontsize=fontsize)
        ax.view_init(elev=30.0, azim=300.0)


        # Plot the surface.
        ax.plot_trisurf(X_CDOM_diag_mesh.loc[:,"CDOM.x1"], X_CDOM_diag_mesh.loc[:,"CDOM.x2"], CDOM_pred_fine_mesh, #197109 datapoints
                        cmap='viridis', edgecolor='none')

        # Customize the z axis.
        ax.set_zlim(0.3, 1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        ax.tick_params(labelsize=8)
        ax.set_zlabel(zlabel="Bray distance")
        ax.set_ylabel(ylabel="CDOM site 2")
        ax.set_xlabel(xlabel="CDOM site 1")


        # Set up the axes for the fourth plot
        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.set_title("Predicted BCC Bray distances by sites' CDOM, \n dataset CDOM coordinates", fontsize=fontsize)
        ax.view_init(elev=30.0, azim=300.0)

        # Plot the surface.
        ax.plot_trisurf(CDOM.loc[:,"CDOM.x1"], CDOM.loc[:,"CDOM.x2"], CDOM_pred, #197109 datapoints
                        cmap='viridis', edgecolor='none')

        # Customize the z axis.
        ax.set_zlim(0.3, 1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        ax.tick_params(labelsize=8)
        ax.set_zlabel(zlabel="Bray distance")
        ax.set_ylabel(ylabel="CDOM site 2")
        ax.set_xlabel(xlabel="CDOM site 1")

        filename = outputPath + 'xgboost'+ '_3d'+'.png'
        fig.savefig(filename)

        # evaluate predictions
        def TruePredictedTable(T,P):
            table = pd.DataFrame(np.concatenate((T, P), axis=1))
            table.columns = ["True values", "Predicted values"]
            return table

        MSE_XGboost = mean_squared_error(y_test, y_pred)
        MSE_XGboost_train = mean_squared_error(y_train, y_pred_train)
        print("MSE on test data:", MSE_XGboost)
        print(TruePredictedTable(y_test[:, np.newaxis], y_pred[:, np.newaxis]))
        print("MSE on train data:", MSE_XGboost_train)
        print(TruePredictedTable(y_train[:, np.newaxis], y_pred_train[:, np.newaxis]))


        print("Best test MSE at epoch", XGboost_best_model_index)
        fig, ax = plt.subplots(1, 1)
        #plt.figure()
        ax.plot(list(range(0, len(MSE_per_epoch['validation_0']['rmse']), 1)), MSE_per_epoch['validation_1']['rmse'], '-', label = 'MSE test', linewidth=1)
        ax.plot(list(range(0, len(MSE_per_epoch['validation_0']['rmse']), 1)), MSE_per_epoch['validation_0']['rmse'], '-', label = 'MSE train', linewidth=1)
        ax.set_ylabel("MSE")
        ax.set_xlabel("Number of epochs")
        plt.grid(False)
        plt.legend(fontsize="medium")
        plt.show()

        filename = outputPath + 'xgboost'+ '_best_mse'+'.png'
        fig.savefig(filename)


    '''
    Plotting Results for manual Feed Forward Neural Network
    '''
    def PlotResultsManualFFNN(self, *args):
        # getting inputs
        NN_reg_original = args[0]
        CDOM = args[1]

        # getting network type
        type = args[2]
        # output path were to save figures
        outputPath = args[3]
        # epochs
        epochs = args[4]
        # batch size
        batch = args[5]


        test_predict = NN_reg_original.predict(NN_reg_original.XTest)
        #best_prediction = NN_reg_original.models[NN_reg_original.accuracy_list.index(min(NN_reg_original.accuracy_list))]
        print(NN_reg_original.accuracy_list)
        print("Minimum MSE :", min(NN_reg_original.accuracy_list), "reached at epoch ", NN_reg_original.accuracy_list.index(min(NN_reg_original.accuracy_list)))

        fig, ax = plt.subplots(1, 1)
        #plt.figure()
        ax.plot(list(range(0, len(NN_reg_original.accuracy_list), 1)), NN_reg_original.accuracy_list, '-', label = 'MSE', linewidth=1)
        ax.set_ylabel("MSE")
        ax.set_xlabel("Number of epochs")
        plt.grid(False)
        plt.legend(fontsize="medium")
        plt.legend()
        plt.yscale('log')
        plt.show()

        filename = outputPath + type + '_e' + str(epochs).zfill(4)+ '_l'+ self.lossName + '_b'+str(batch)+'_mse'+'.png'
        fig.savefig(filename)


        #Use log-transformed CDOM values for creating design matrix, then plot on original values
        x_mesh = np.log10(np.arange(min(CDOM.loc[:,"CDOM.x1"]), max(CDOM.loc[:,"CDOM.x2"]) + 0.01, 0.01)) + 1
        y_mesh = x_mesh.copy()
        x_mesh, y_mesh = np.meshgrid(x_mesh,y_mesh)
        X_CDOM_mesh = self.pdCat(x_mesh.ravel()[:, np.newaxis], y_mesh.ravel()[:, np.newaxis]).to_numpy()
        best_prediction = NN_reg_original.model_prediction(X_CDOM_mesh, NN_reg_original.accuracy_list.index(min(NN_reg_original.accuracy_list)))

        x_mesh = np.arange(min(CDOM.loc[:,"CDOM.x1"]), max(CDOM.loc[:,"CDOM.x2"]) + 0.01, 0.01)
        y_mesh = x_mesh.copy()
        x_mesh, y_mesh = np.meshgrid(x_mesh,y_mesh)

        ff_pred_original = best_prediction.copy()
        ff_pred_original = np.reshape(ff_pred_original, (363, 363))
        ff_pred_original[x_mesh-y_mesh==0] = np.nan
        ff_pred_original[x_mesh>y_mesh] = np.nan
        #print(pd.DataFrame(ff_pred_original))

        # Plot the NN-smoothed surface
        fig = plt.figure(figsize=(15, 12))
        ax = fig.gca(projection="3d")
        ax.set_title("Predicted BCC Bray distances by sites' CDOM", fontsize=12)
        ax.view_init(elev=30.0, azim=300.0)
        surf = ax.plot_surface(x_mesh, y_mesh, ff_pred_original, cmap='viridis',
                               antialiased=True, vmin=np.nanmin(ff_pred_original), vmax=np.nanmax(ff_pred_original),
                               rstride=1, cstride=1)
        # Customize the z axis.
        z_range =  (np.nanmax(ff_pred_original) - np.nanmin(ff_pred_original))
        ax.set_zlim(np.nanmin(ff_pred_original) - z_range, np.nanmax(ff_pred_original) + z_range)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.tick_params(labelsize=8)
        plt.show()

        filename = outputPath + type + '_e' + str(epochs).zfill(4)+ '_l'+ self.lossName + '_b'+str(batch)+'_3d'+'.png'
        fig.savefig(filename)


    def pdCat(self, x1, x2):
        table = pd.DataFrame(np.concatenate((x1, x2), axis=1))
        table.columns = ["CDOM.x1", "CDOM.x2"]
        return table