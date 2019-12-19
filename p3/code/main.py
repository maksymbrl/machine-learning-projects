"""
@author: maksymb
"""

# Library imports
import os, sys
import numpy as np
import keras
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBRegressor

import pandas as pd
import yaml
# libraries for plotting results
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import scale
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import seaborn as sbn
# to calculate time
import time

# importing manually created libraries
import neural, funclib, regression, xgb, random_forest, data_processing

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

        # Instantiating object variable from Functions Library
        self.funcs = funclib.Functions(self.paramData['RandomSeed'],
                                       self.paramData['Loss'])

    '''
    Method to preprocess the data set for Side Research Questions
    '''
    def PreProcessing(self, *args):
        # Random Seed
        seed = self.paramData['RandomSeed']
        # Loss function - we need it to instantiate funclib variable
        loss = self.paramData['Loss']

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

        elif self.paramData['type'] == 'rf_side':

            #ph = meta["pH"]
            #n2o = meta["N2O"]
            #temp = meta["Temperature"]
            tp = meta[target_choice]
            anchors = {"TMP": (data.iloc[10], data.iloc[70])}
            anchors[target_choice] = (data.iloc[15], data.iloc[67])
            # Some data formating
            # y = make_categories(temp,2)
            #np.random.seed(2)
            self.Y = tp#.values.reshape(-1, 1)
            # normalizing data
            self.X_norm = data.to_numpy()
            # y = to_categorical(y, num_classes=None)
            # Split into training and testing
            self.X_train, self.X_test, y_train, y_test = train_test_split(self.X_norm,
                                                                          self.Y,
                                                                          random_state=seed,
                                                                          test_size=self.paramData['TestSize'])
            self.y_train_l, self.y_test_l = self.funcs.set_category(self.Y, y_train, y_test)

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
        # getting data for main problem
        dataproc = data_processing.DataProcessing()
        data = dataproc.GetMainData()
        # getting everything we need
        CDOM, CDOM_sorted, CDOM_diag_mesh, \
        ASV, ASV_ranged, \
        metadata, metadata_scaled, \
        X_ASV, y_CDOM = data
        #XGboost with scikitlearn - data with spatial component (BCC Bray distance by CDOM)
        X_CDOM = CDOM.loc[:,["CDOM.x1", "CDOM.x2"]] #Molten meshgrid CDOM values for real data BCC Bray distances
        X_CDOM_diag_mesh = CDOM_diag_mesh.loc[:,["CDOM.x1", "CDOM.x2"]] #Molten meshgrid CDOM values for generating predicted BCC Bray distances
        y_CDOM = CDOM.loc[:,"ASV.dist"]

        if self.paramData['type'] == 'ffnn_keras':
            # retrieving network data
            NNdata = self.PreProcessing()
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
            self.funcs.PlotResultsKeras(history,
                                   self.paramData['type'],
                                   self.paramData['OutputPath'],
                                   self.paramData['epochs'],
                                   self.paramData['Optimization'],
                                   self.paramData['BatchSize'])

        elif self.paramData['type'] == 'snn_keras':
            # retrieving network data
            NNdata = self.PreProcessing()
            '''
            Getting Network Architecture
            '''
            network = neural.NeuralNetwork(NNdata)
            # passing network architecture and create the model
            model = network.BuildModel()
            # training model
            model, history = network.TrainModel(model, self.pairs_train, self.Y_train_onehot, self.pairs_test, self.Y_test_onehot)
            # Plotting results
            self.funcs.PlotResultsKeras(history,
                                   self.paramData['type'],
                                   self.paramData['OutputPath'],
                                   self.paramData['epochs'],
                                   self.paramData['Optimization'],
                                   self.paramData['BatchSize'])
        elif self.paramData['type'] == 'tnn_keras':
            # retrieving network data
            NNdata = self.PreProcessing()
            '''
            Getting Network Architecture
            '''
            network = neural.NeuralNetwork(NNdata)
            # passing network architecture and create the model
            model = network.BuildModel()
            # training model
            model, history = network.TrainModel(model, self.triplets_train, self.Y_train_onehot, self.triplets_test, self.Y_test_onehot)
            # Plotting results
            self.funcs.PlotResultsKeras(history,
                                   self.paramData['type'],
                                   self.paramData['OutputPath'],
                                   self.paramData['epochs'],
                                   self.paramData['Optimization'],
                                   self.paramData['BatchSize'])

        elif self.paramData['type'] == 'ffnn_manual':
            # Neural network with multiple layers - regression - BCC Bray distances by CDOM - original data
            X_CDOM = CDOM.loc[:,["CDOM.x1", "CDOM.x2"]].to_numpy()
            y_CDOM = CDOM.loc[:,"ASV.dist"].to_numpy()[:, np.newaxis] #Original data

            '''
            NN_reg_original = neural.NeuralNetworkML(X_CDOM, y_CDOM,
                                                     trainingShare=0.80,
                                                     n_hidden_layers=3,
                                                     n_hidden_neurons=[2000, 1000, 500],
                                                     n_categories=1,
                                                     epochs=10, batch_size=10,
                                                     eta=1e-8,
                                                     lmbd=0, fixed_LR=False,
                                                     method="regression",
                                                     activation="sigmoid",
                                                     seed = self.paramData['RandomSeed'])
            '''
            n_hidden_neurons = []
            for layer in range(self.paramData['NHiddenLayers']):
                n_hidden_neurons.append(self.paramData['NHiddenNeurons'])
            for layer in range(1, self.paramData['NHiddenLayers'], 1):
                n_hidden_neurons[layer] = int(n_hidden_neurons[layer-1]/2)
            #print(n_hidden_neurons)

            NN_reg_original = neural.NeuralNetworkML(X_CDOM, y_CDOM,
                                                     trainingShare=1-self.paramData['TestSize'],
                                                     n_hidden_layers=self.paramData['NHiddenLayers'],
                                                     n_hidden_neurons=n_hidden_neurons,
                                                     n_categories=1,
                                                     epochs=self.paramData['epochs'],
                                                     batch_size=self.paramData['BatchSize'],
                                                     eta=self.paramData['alpha'],
                                                     lmbd=0,
                                                     fixed_LR=False,
                                                     method="regression",
                                                     activation="sigmoid",
                                                     seed = self.paramData['RandomSeed'])



            NN_reg_original.train()
            # Plotting results
            self.funcs.PlotResultsManualFFNN(NN_reg_original,
                                             CDOM,
                                             self.paramData['type'],
                                             self.paramData['OutputPath'],
                                             self.paramData['epochs'],
                                             self.paramData['BatchSize'])
        elif self.paramData['type'] == 'xgb':
            X_train, X_test, y_train, y_test = train_test_split(X_CDOM, y_CDOM,
                                                                train_size=1-self.paramData['TestSize'],
                                                                test_size = self.paramData['TestSize'],
                                                                random_state=self.paramData['RandomSeed'])
            # initialising xgboosting
            xgboosting = xgb.XGBoosting()
            model = xgboosting.RunModel(X_train, X_test,
                                        y_train, y_test,
                                        X_CDOM, X_CDOM_diag_mesh,
                                        CDOM, CDOM_sorted,
                                        self.paramData['OutputPath'])
            #Get best model by test MSE
            XGboost_best_model_index = model.best_iteration
            XGboost_best_iteration = model.get_booster().best_ntree_limit
            MSE_per_epoch = model.evals_result()

            # make predictions for test data
            y_pred = model.predict(X_test, ntree_limit=XGboost_best_iteration)
            y_pred_train = model.predict(X_train)
            #predictions = [round(value) for value in y_pred]

            best_prediction = model.predict(X_CDOM, ntree_limit=XGboost_best_iteration)
            CDOM_pred = best_prediction.copy()  #CDOM_pred.shape: (2556,) CDOM_pred are the predicted BCC Bray distances for CDOM value pairs
            CDOM_pred_fine_mesh = model.predict(X_CDOM_diag_mesh, ntree_limit=XGboost_best_iteration)
            '''
            y_pred,\
            y_pred_train,\
            MSE_per_epoch,\
            CDOM_pred, \
            CDOM_pred_fine_mesh, \
            XGboost_best_model_index = xgboosting.RunModel(X_train, X_test,
                                                        y_train, y_test,
                                                        X_CDOM, X_CDOM_diag_mesh,
                                                        CDOM, CDOM_sorted,
                                                        self.paramData['OutputPath'])
            '''
            # plotting 3d plots and mse for XGBoost
            self.funcs.PlotResultsXGBoost(CDOM, CDOM_sorted,
                                          X_CDOM_diag_mesh,
                                          CDOM_pred_fine_mesh,
                                          CDOM_pred,
                                          self.paramData['OutputPath'],
                                          y_pred,
                                          y_pred_train,
                                          MSE_per_epoch,
                                          y_train, y_test,
                                          XGboost_best_model_index)
        elif self.paramData['type'] == 'rf_main':
            rf = random_forest.RandomForest()
            # Laurent
            population_size, metadata = rf.read_data(False, False)
            predictions, test_y, ML_ = rf.prepare_data(population_size,
                                                       metadata,
                                                       self.paramData['TestSize'],
                                                       self.paramData['RandomSeed'])
            all_predictions = rf.predict_all_metadata(population_size, metadata, ML_)
            # we will compare the outcome with xgboost
            def MergeTable(var_list, metadata_variables):
                table = pd.DataFrame(np.concatenate((var_list), axis=1))
                table.columns = metadata_variables
                return table

            def PredictMetadata(ASV_table, metadata_variables, train_size, test_size, seed):
                X_ASV = ASV_table
                X_ASV.columns = [''] * len(X_ASV.columns)
                X_ASV = X_ASV.to_numpy()
                metadata_list = []
                for i in metadata_variables:
                    #y_CDOM = metadata.loc[:, i][:, np.newaxis]

                    # split data into train and test sets
                    y_meta = metadata.loc[:, i] #Requires 1d array
                    X_train, X_test, y_train, y_test = train_test_split(X_ASV, y_meta,
                                                                        train_size = train_size,
                                                                        test_size  = test_size,
                                                                        random_state=seed)

                    # fit model no training data
                    model = XGBRegressor(objective='reg:squarederror')
                    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
                              eval_metric='rmse', early_stopping_rounds=100, verbose=False)

                    #Get best model by test MSE
                    XGboost_best_model_index = model.best_iteration
                    XGboost_best_iteration = model.get_booster().best_ntree_limit

                    # make predictions for full dataset
                    y_pred = model.predict(X_ASV, ntree_limit=XGboost_best_iteration)
                    metadata_list.append(y_pred[:, np.newaxis])
                return MergeTable(metadata_list, metadata_variables)

            var_list = ["Latitude","Longitude","Altitude","Area","Depth","Temperature","Secchi","O2","CH4","pH","TIC","SiO2","KdPAR"]
            train_size = 1-self.paramData['TestSize']
            test_size = self.paramData['TestSize']
            seed = self.paramData['RandomSeed']
            predicted_metadata = PredictMetadata(ASV, var_list, train_size, test_size, seed)

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                print(predicted_metadata)


        elif self.paramData['type'] == 'rf_side':
            # retrieving network data
            NNdata = self.PreProcessing()
            rf = random_forest.RandomForest()
            seed = self.paramData['RandomSeed']
            clfs, scores_test, scores_train = rf.predict_t(self.X_train, self.X_test, self.y_train_l, self.y_test_l, seed)

        elif self.paramData['type'] == 'all':
            '''
            Neural Network
            '''
            # Neural network with multiple layers - regression - BCC Bray distances by CDOM - original data
            X_CDOM = CDOM.loc[:,["CDOM.x1", "CDOM.x2"]].to_numpy()
            y_CDOM = CDOM.loc[:,"ASV.dist"].to_numpy()[:, np.newaxis] #Original data

            n_hidden_neurons = []
            for layer in range(self.paramData['NHiddenLayers']):
                n_hidden_neurons.append(self.paramData['NHiddenNeurons'])
            for layer in range(1, self.paramData['NHiddenLayers'], 1):
                n_hidden_neurons[layer] = int(n_hidden_neurons[layer-1]/2)
            #print(n_hidden_neurons)

            NN_reg_original = neural.NeuralNetworkML(X_CDOM, y_CDOM,
                                                     trainingShare=1-self.paramData['TestSize'],
                                                     n_hidden_layers=self.paramData['NHiddenLayers'],
                                                     n_hidden_neurons=n_hidden_neurons,
                                                     n_categories=1,
                                                     epochs=self.paramData['epochs'],
                                                     batch_size=self.paramData['BatchSize'],
                                                     eta=self.paramData['alpha'],
                                                     lmbd=0,
                                                     fixed_LR=False,
                                                     method="regression",
                                                     activation="sigmoid",
                                                     seed = self.paramData['RandomSeed'])

            NN_reg_original.train()

            x_mesh = np.log10(np.arange(min(CDOM.loc[:,"CDOM.x1"]), max(CDOM.loc[:,"CDOM.x2"]) + 0.01, 0.01)) + 1
            y_mesh = x_mesh.copy()
            x_mesh, y_mesh = np.meshgrid(x_mesh,y_mesh)
            X_CDOM_mesh = self.funcs.pdCat(x_mesh.ravel()[:, np.newaxis], y_mesh.ravel()[:, np.newaxis]).to_numpy()
            best_prediction = NN_reg_original.model_prediction(X_CDOM_mesh, NN_reg_original.accuracy_list.index(min(NN_reg_original.accuracy_list)))

            x_mesh = np.arange(min(CDOM.loc[:,"CDOM.x1"]), max(CDOM.loc[:,"CDOM.x2"]) + 0.01, 0.01)
            y_mesh = x_mesh.copy()
            x_mesh, y_mesh = np.meshgrid(x_mesh,y_mesh)

            ff_pred_original = best_prediction.copy()
            ff_pred_original = np.reshape(ff_pred_original, (363, 363))
            ff_pred_original[x_mesh-y_mesh==0] = np.nan
            ff_pred_original[x_mesh>y_mesh] = np.nan

            '''
            XGBoost part
            '''
            X_CDOM = CDOM.loc[:,["CDOM.x1", "CDOM.x2"]] #Molten meshgrid CDOM values for real data BCC Bray distances
            X_CDOM_diag_mesh = CDOM_diag_mesh.loc[:,["CDOM.x1", "CDOM.x2"]] #Molten meshgrid CDOM values for generating predicted BCC Bray distances
            y_CDOM = CDOM.loc[:,"ASV.dist"]

            X_train, X_test, y_train, y_test = train_test_split(X_CDOM, y_CDOM,
                                                                train_size=1-self.paramData['TestSize'],
                                                                test_size = self.paramData['TestSize'],
                                                                random_state=self.paramData['RandomSeed'])
            # initialising xgboosting
            xgboosting = xgb.XGBoosting()
            model = xgboosting.RunModel(X_train, X_test,
                                        y_train, y_test,
                                        X_CDOM, X_CDOM_diag_mesh,
                                        CDOM, CDOM_sorted,
                                        self.paramData['OutputPath'])

            #Get best model by test MSE
            XGboost_best_model_index = model.best_iteration
            XGboost_best_iteration = model.get_booster().best_ntree_limit
            MSE_per_epoch = model.evals_result()

            # make predictions for test data
            y_pred = model.predict(X_test, ntree_limit=XGboost_best_iteration)
            y_pred_train = model.predict(X_train)
            #predictions = [round(value) for value in y_pred]

            best_prediction = model.predict(X_CDOM, ntree_limit=XGboost_best_iteration)
            CDOM_pred = best_prediction.copy()  #CDOM_pred.shape: (2556,) CDOM_pred are the predicted BCC Bray distances for CDOM value pairs
            CDOM_pred_fine_mesh = model.predict(X_CDOM_diag_mesh, ntree_limit=XGboost_best_iteration)

            '''
            Simple OLS - generating design matrix out of data set etc.
            '''
            reg = regression.Regression()
            X_mesh = reg.GenerateMesh(0.21, 3.83, 0.21, 3.83, 0.01, 0.01, log_transform=True) # The low number of points on the higher end of the gradient causes distortions for linear regression
            X_mesh_degree_list = reg.DesignMatrixList(X_mesh[0], X_mesh[1], 12)[1:]
            X_degree_list = reg.DesignMatrixList(CDOM.loc[:,"CDOM.x1"], CDOM.loc[:,"CDOM.x2"], 12)[1:]
            X_degree_list_subset = []

            z = CDOM_pred #XGboost-predicted values
            z = CDOM.loc[:,"ASV.dist"] #Original data
            #ebv_no_resampling = reg.generate_error_bias_variance_without_resampling(X_degree_list, 1)
            #ebv_resampling = reg.generate_error_bias_variance_with_resampling(X_degree_list, 1, 100)
            #reg.ebv_by_model_complexity(ebv_resampling)
            #reg.training_vs_test(ebv_no_resampling)

            CDOM_pred_reg = X_mesh_degree_list[8] @ reg.beta_SVD(X_degree_list[8], CDOM_pred)
            #print(pd.DataFrame(X_mesh_degree_list[1]))
            #print(CDOM_pred_reg)
            #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            #  print(pd.DataFrame(CDOM_pred_reg))

            x_mesh_reg = np.arange(min(CDOM.loc[:,"CDOM.x1"]), max(CDOM.loc[:,"CDOM.x2"]) + 0.01, 0.01)
            y_mesh_reg = x_mesh_reg.copy()
            x_mesh_reg, y_mesh_reg = np.meshgrid(x_mesh_reg,y_mesh_reg)
            X_CDOM_mesh = self.funcs.pdCat(x_mesh_reg.ravel()[:, np.newaxis], y_mesh_reg.ravel()[:, np.newaxis])
            #print(pd.DataFrame(X_CDOM_mesh))
            #print("CDOM_pred_reg.shape", CDOM_pred_reg.shape)
            z_CDOM_mesh_pred = np.reshape(CDOM_pred_reg, (x_mesh_reg.shape[0], x_mesh_reg.shape[0]))
            z_CDOM_mesh_pred[x_mesh_reg-y_mesh_reg==0] = np.nan
            z_CDOM_mesh_pred[x_mesh_reg>y_mesh_reg] = np.nan

            '''
            Neural Network with data from XGBoost
            '''
            X_CDOM = CDOM.loc[:,["CDOM.x1", "CDOM.x2"]].to_numpy()
            y_CDOM = CDOM_pred[:, np.newaxis] #Predicted data from XGboost

            n_hidden_neurons = []
            for layer in range(self.paramData['NHiddenLayers']):
                n_hidden_neurons.append(self.paramData['NHiddenNeurons'])
            for layer in range(1, self.paramData['NHiddenLayers'], 1):
                n_hidden_neurons[layer] = int(n_hidden_neurons[layer-1]/2)
            #print(n_hidden_neurons)

            NN_reg = neural.NeuralNetworkML(X_CDOM, y_CDOM,
                                                     trainingShare=1-self.paramData['TestSize'],
                                                     n_hidden_layers=self.paramData['NHiddenLayers'],
                                                     n_hidden_neurons=n_hidden_neurons,
                                                     n_categories=1,
                                                     epochs=self.paramData['epochs'],
                                                     batch_size=self.paramData['BatchSize'],
                                                     eta=self.paramData['alpha'],
                                                     lmbd=0,
                                                     fixed_LR=False,
                                                     method="regression",
                                                     activation="sigmoid",
                                                     seed = self.paramData['RandomSeed'])

            NN_reg.train()
            test_predict = NN_reg.predict(NN_reg.XTest)
            print(NN_reg.accuracy_list)

            #Use log-transformed CDOM values for creating design matrix, then plot on original values
            x_mesh = np.log10(np.arange(min(CDOM.loc[:,"CDOM.x1"]), max(CDOM.loc[:,"CDOM.x2"]) + 0.01, 0.01)) + 1
            y_mesh = x_mesh.copy()
            x_mesh, y_mesh = np.meshgrid(x_mesh,y_mesh)
            X_CDOM_mesh = self.funcs.pdCat(x_mesh.ravel()[:, np.newaxis], y_mesh.ravel()[:, np.newaxis]).to_numpy()
            best_prediction = NN_reg.model_prediction(X_CDOM_mesh, NN_reg.accuracy_list.index(min(NN_reg.accuracy_list)))

            x_mesh = np.arange(min(CDOM.loc[:,"CDOM.x1"]), max(CDOM.loc[:,"CDOM.x2"]) + 0.01, 0.01)
            y_mesh = x_mesh.copy()
            x_mesh, y_mesh = np.meshgrid(x_mesh,y_mesh)

            ff_pred = best_prediction.copy()
            ff_pred = np.reshape(ff_pred, (363, 363))
            ff_pred[x_mesh-y_mesh==0] = np.nan
            ff_pred[x_mesh>y_mesh] = np.nan

            '''
            Plotting 3d graphs for all data
            '''
            fontsize = 6
            #Compare raw data to XGboost, neural network predicted data and XGboost predicted data smoothed with neural network
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(2, 3, 1, projection='3d')
            ax.set_title("BCC Bray distances by sites' DOM", fontsize=fontsize)
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
            ax.set_ylabel(ylabel="DOM site 2")
            ax.set_xlabel(xlabel="DOM site 1")

            # Set up the axes for the second plot
            ax = fig.add_subplot(2, 3, 2, projection='3d')
            #ax.set_title("XGboost-Predicted BCC Bray distances by sites' CDOM, dataset CDOM coordinates", fontsize=8)
            ax.set_title("XGboost-Predicted BCC \n Bray distances by sites' DOM", fontsize=fontsize)
            ax.view_init(elev=30.0, azim=300.0)

            # Plot the surface.
            ax.plot_trisurf(CDOM.loc[:,"CDOM.x1"], CDOM.loc[:,"CDOM.x2"], CDOM_pred, #197109 datapoints
                            cmap='viridis', edgecolor='none')

            # Customize the z axis.
            z_range =  (np.nanmax(CDOM_pred) - np.nanmin(CDOM_pred))
            ax.set_zlim(np.nanmin(CDOM_pred) - z_range, 1)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
            ax.tick_params(labelsize=8)
            ax.set_zlabel(zlabel="Bray distance")
            ax.set_ylabel(ylabel="DOM site 2")
            ax.set_xlabel(xlabel="DOM site 1")

            # Set up the axes for the third plot
            ax = fig.add_subplot(2, 3, 3, projection='3d')
            #ax.set_title("OLS (SVD) regression-predicted BCC Bray distances by sites' CDOM, CDOM 0.01 step meshgrid", fontsize=6)
            ax.set_title("OLS (SVD) regression-predicted \n BCC Bray distances by sites' DOM", fontsize=fontsize)
            ax.view_init(elev=30.0, azim=300.0)

            # Plot the surface.
            ax.plot_trisurf(x_mesh_reg.ravel(), y_mesh_reg.ravel(), z_CDOM_mesh_pred.ravel(), cmap='viridis', #197109 datapoints
                            vmin=np.nanmin(z_CDOM_mesh_pred), vmax=np.nanmax(z_CDOM_mesh_pred),
                            edgecolor='none')

            # Customize the z axis.
            z_range =  (np.nanmax(z_CDOM_mesh_pred) - np.nanmin(z_CDOM_mesh_pred))
            ax.set_zlim(np.nanmin(z_CDOM_mesh_pred) - z_range, 1)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
            ax.tick_params(labelsize=8)
            ax.set_zlabel(zlabel="Bray distance")
            ax.set_ylabel(ylabel="DOM site 2")
            ax.set_xlabel(xlabel="DOM site 1")

            # Set up the axes for the fourth plot
            ax = fig.add_subplot(2, 3, 4, projection='3d')
            #ax.set_title("NN-smoothed XGboost-predicted BCC Bray distances by sites' CDOM, CDOM 0.01 step meshgrid", fontsize=6)
            ax.set_title("NN-smoothed XGboost-predicted \n BCC Bray distances by sites' DOM", fontsize=fontsize)
            ax.view_init(elev=30.0, azim=300.0)

            # Plot the surface.
            ax.plot_trisurf(x_mesh.ravel(), y_mesh.ravel(), ff_pred.ravel(), #197109 datapoints
                            cmap='viridis', edgecolor='none',
                            vmin=np.nanmin(ff_pred), vmax=np.nanmax(ff_pred))

            # Customize the z axis.
            z_range =  (np.nanmax(ff_pred) - np.nanmin(ff_pred))
            ax.set_zlim(np.nanmin(ff_pred) - z_range, 1)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
            ax.tick_params(labelsize=8)
            ax.set_zlabel(zlabel="Bray distance")
            ax.set_ylabel(ylabel="DOM site 2")
            ax.set_xlabel(xlabel="DOM site 1")

            # Set up the axes for the fifth plot
            ax = fig.add_subplot(2, 3, 5, projection='3d')
            #ax.set_title("NN-predicted BCC Bray distances by sites' CDOM, CDOM 0.01 step meshgrid", fontsize=8)
            ax.set_title("NN-predicted BCC Bray \n distances by sites' DOM", fontsize=fontsize)
            ax.view_init(elev=30.0, azim=300.0)

            # Plot the surface.
            ax.plot_trisurf(x_mesh.ravel(), y_mesh.ravel(), ff_pred_original.ravel(), #197109 datapoints
                            cmap='viridis', edgecolor='none',
                            vmin=np.nanmin(ff_pred_original), vmax=np.nanmax(ff_pred_original))

            # Customize the z axis.
            z_range =  (np.nanmax(ff_pred_original) - np.nanmin(ff_pred_original))
            ax.set_zlim(np.nanmin(ff_pred_original) - z_range, 1)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
            ax.tick_params(labelsize=8)
            ax.set_zlabel(zlabel="Bray distance")
            ax.set_ylabel(ylabel="DOM site 2")
            ax.set_xlabel(xlabel="DOM site 1")

            # Set up the axes for the sixth plot
            ax = fig.add_subplot(2, 3, 6, projection='3d')
            #ax.set_title("XGboost-predicted BCC Bray distances by sites' CDOM, CDOM 0.01 step meshgrid", fontsize=8)
            ax.set_title("XGboost-predicted BCC Bray \n distances by sites' DOM", fontsize=fontsize)
            ax.view_init(elev=30.0, azim=300.0)


            # Plot the surface.
            ax.plot_trisurf(X_CDOM_diag_mesh.loc[:,"CDOM.x1"], X_CDOM_diag_mesh.loc[:,"CDOM.x2"], CDOM_pred_fine_mesh, #197109 datapoints
                            cmap='viridis', edgecolor='none')

            # Customize the z axis.
            z_range =  (np.nanmax(CDOM_pred_fine_mesh) - np.nanmin(CDOM_pred_fine_mesh))
            ax.set_zlim(np.nanmin(CDOM_pred_fine_mesh) - z_range, 1)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
            ax.tick_params(labelsize=8)
            ax.set_zlabel(zlabel="Bray distance")
            ax.set_ylabel(ylabel="DOM site 2")
            ax.set_xlabel(xlabel="DOM site 1")

            #filename = self.paramData['OutputPath']
            filename = self.paramData['OutputPath'] + '/' + 'everything_3d' + '.png'
            fig.savefig(filename)

            plt.show()



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