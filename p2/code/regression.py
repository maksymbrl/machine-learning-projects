#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:56:48 2019

@author: maksymb
"""

import numpy as np
# for polynomial manipulation
import sympy as sp
# from sympy import *
import itertools as it

import multiprocessing as mp
from joblib import Parallel, delayed

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.preprocessing import PolynomialFeatures
# regression libraries
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
# to split data for testing and training - KFold cross validation implementation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Scikitlearn imports to check results
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from numpy import argmax

# We'll need some metrics to evaluate our models
from sklearn.neural_network import MLPClassifier, MLPRegressor


import keras
# stochastic gradient descent
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense

import funclib
import data_processing

'''
Class, which handles both Logistic and Linear Regressions
'''
class RegressionPipeline:
    # constructor
    def __init__(self, *args):
        # Variables common to both of them
        pass
    
    def DoLinearRegression(self, *args):
        #=====================================================================#
        # Liner Regression variables
        #=====================================================================#
        # symbolic variables
        #x_symb = args[0]
        # array of values for each variable/feature
        #x_vals = args[1]
        # grid values
        #x = args[2]
        #y = args[3]
        #z = args[4]
        # 1.96 to calculate stuff with 95% confidence
        #confidence = args[5]
        # noise variance - for confidence intervals estimation
        #sigma = args[6]
        # k-value for Cross validation
        #kfold = args[7]
        # hyper parameter
        #lambda_par = args[8]
        # directory where to store plots
        #output_dir = args[9]
        #prefix = args[10]
        # degree of polynomial to fit
        #poly_degree = args[11]
        #=====================================================================#
        funcNormal = funclib.NormalFuncs()
        funcError = funclib.ErrorFuncs()
        funcPlot  = funclib.PlotFuncs()
        X = args[0]
        x = args[1]
        y = args[2]
        z = args[3]
        x_rav = args[4]
        y_rav = args[5]
        z_rav = args[6]
        zshape = args[7]
        poly_degree = args[8]
        lambda_par = args[9]
        sigma = args[10]
        outputPath = [11]
        # getting the design matrix
        #X = func.ConstructDesignMatrix(x_symb, x_vals, poly_degree)
        # getting the Ridge/OLS Regression via direct multiplication
        ztilde_ridge = funcNormal.CallNormal(X, z_rav, lambda_par, sigma)
        ztilde_ridge = ztilde_ridge.reshape(zshape)
        
        ''' Scikit Learn '''
        poly_features = PolynomialFeatures(degree = poly_degree)
        X_scikit = np.swapaxes(np.array([x_rav, y_rav]), 0, 1)
        X_poly = poly_features.fit_transform(X_scikit)
        ridge_reg = Ridge(alpha = lambda_par, fit_intercept=True).fit(X_poly, z_rav)
        ztilde_sk = ridge_reg.predict(X_poly).reshape(zshape)
        zarray_ridge = [z, ztilde_ridge, ztilde_sk]
        print('\n')
        print("Ridge MSE (no CV) - " + str(funcError.CallMSE(zarray_ridge[0], zarray_ridge[1])) + "; sklearn - " + str(mean_squared_error(zarray_ridge[0], zarray_ridge[2])))
        print("Ridge R^2 (no CV) - " + str(funcError.CallR2(zarray_ridge[0], zarray_ridge[1])) + "; sklearn - " + str(ridge_reg.score(X_poly, z_rav)))
        print('\n')
        ''' Plotting Surfaces '''
        filename = 'ridge_p' + str(poly_degree).zfill(2) + '_n.png'
        funcPlot.PlotSurface(x, y, zarray_ridge, outputPath, filename)
        #filename = self.prefix + '_ridge_p' + str(self.poly_degree).zfill(2) + '_n' + npoints_name + '.png'
        #lib.plotSurface(self.x, self.y, zarray_ridge, self.output_dir, filename)
        
        #PlotSurface.PlotFuncs()
        
        
        
    def DoLogisticRegression(self, *args):
        #=====================================================================#
        # Logistic Regression variables
        #=====================================================================#
        X_train = args[0]
        Y_train_onehot = args[1]
        epochs   = args[2]
        lmbd = args[3]
        alpha = args[4]
        '''
        Part 1: Implementing Logistic Regression via gradient Descent. 
        Batch gradient descent and usual GD are implemented for Part 2:
        NN Logistic Regression.
        '''
        activeFuncs = funclib.ActivationFuncs()
        costFuncs = funclib.CostFuncs()
        theta = np.zeros((X_train.shape[1], 1))
        epochs1 = range(epochs)
        #if BatchSize == 0:
        m = len(Y_train_onehot)
        costs = []
        for epoch in epochs1:
            Y_pred = np.dot(X_train, theta)
            A = activeFuncs.CallSigmoid(Y_pred)#CallSigmoid(Y_pred)
            # cost function        
            J, dJ = costFuncs.CallLogistic(X_train, Y_train_onehot, A)
            # Adding regularisation
            J = J + lmbd / (2*m) * np.sum(theta**2)
            dJ = dJ + lmbd * theta / m
            # updating weights
            theta = theta - alpha * dJ #+ lmbd * theta/m
            # updating cost func history
            costs.append(J)
            # getting values of cost function at each epoch
            if(epoch % 100 == 0):
                print('Cost after iteration# {:d}: {:f}'.format(epoch, J))
                      
        #print("Old accuracy on training data: " + str(accuracy_score(predict(X_train), Y_train_onehot)))
        
        return costs