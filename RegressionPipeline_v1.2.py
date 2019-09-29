#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:13:27 2019

@author: maksymb
"""

import numpy as np
import random as rd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# for polynimial manipulation
import sympy as sp
#from sympy import *
import itertools as it

# Machine Learning libraries
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import time


class MainPipeline():
    '''
    Class constructor
    '''
    def __init__(self, *args):
        '''
        Creating data sets, using uniform distribution
        '''
        # creating a starting value for number generator
        # so each time we will get the same random numbers
        np.random.seed(1)
        # number of points
        self.N = args[0]
        # number of independent variables
        self.n_vars = args[1]
        # polynomial degree
        self.poly_degree = args[2]
        # k-value
        self.kfold = args[3]
        # generating an array of symbolic variables 
        # based on the desired amount of variables
        self.x_symb = sp.symarray('x', self.n_vars, real=True)
        # making a copy of this array
        self.x_vals = self.x_symb.copy()
        # and fill it with values
        for i in range(self.n_vars):
            self.x_vals[i] = np.sort(np.random.uniform(0, 1, self.N))
    
    '''
    Franke function, used to generate outputs (z values)
    '''
    def FrankeFunction(self, x, y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    '''
    Main method of the class
    '''
    def main(self, *args):
        # getting inputs
        #n_vars = args[0]
        #poly_degree = args[1]
        # generating output data - first setting-up the proper grid
        x, y = np.meshgrid(self.x_vals[0], self.x_vals[1])
        # and creating an output based on the input
        z = self.FrankeFunction(x, y) + 0.1 * np.random.randn(self.N)
        z_array = [z for i in range(self.poly_degree)]
        # getting design matrix
        #X1, X2 = self.constructDesignMatrix(x, y, z)
        '''
        (a) Ordinary Least Square on the Franke function with resampling:
            Calling method to do Linear Regression (with the generated polynomial)
        '''
        ''' MANUAL '''
        # calculating design matrix for different polynomial degree (from 1 to 5)
        X_polyarray = [self.constructDesignMatrix(degree) for degree in range(1, self.poly_degree+1)]
        ztilde_polyarray = [self.doLinearRegression(X, z).reshape(-1, self.N) for X in X_polyarray]
        #print(np.shape(ztilde_polyarray))
        #ztilde1 = ztilde1.reshape(-1, self.N)
        #ztilde2, beta2, conf2 = self.doLinearRegression(X2, z)
        #ztilde2 = ztilde2.reshape(-1, self.N)
        ''' SKLEARN '''

        # so far the best thing - to swap the axes, so we will get:
        # [[x[0], y[0]],[x[1],y[1]], [x[2],y[2]], ...]
        # works mush better than the transpose and reshape
        # (so far reshape, without "F" order was giving crap)
        X_scikit = np.swapaxes(np.array([self.x_vals[0], self.x_vals[1]]), 0,1)
        # generate polynomial
#        poly_features = PolynomialFeatures(degree = self.poly_degree, include_bias = True)
        poly_features_array = [PolynomialFeatures(degree = degree) for degree in range(1, self.poly_degree+1)]
        X_poly_array = [poly_features_array[i].fit_transform(X_scikit) for i in range(len(poly_features_array))]
        lin_reg_array = [LinearRegression(normalize=True).fit(X_poly_array[i], z) for i in range(len(poly_features_array))]
        ztilde_skarray = [lin_reg_array[i].predict(X_poly_array[i]) for i in range(len(poly_features_array))]
#        print("betas are %s" %beta1 + " ± %s" %conf1)
        #print("betas are %s" %beta2 + " ± %s" %conf2)
#        print("MSE is ", self.getMSE(z, ztilde1)," and sklearn ", mean_squared_error(z, ztilde_sk))
#        print("R^2 is ", self.getR2(z, ztilde1)," and sklearn ", lin_reg.score(X_poly, z))
        ''' Plotting surfaces '''
        # drawing surface - plotting stuff in a smart way - for Linear Regression
        poly_fig = plt.figure(figsize = (10, 10))
        poly_fig.suptitle('Linear Regression', fontsize=14)
        n_plots = range(1, self.poly_degree*3+1)
        poly_surf = []
        for i in range(0,self.poly_degree):
            poly_surf.append(z_array[i])
            poly_surf.append(ztilde_polyarray[i])
            poly_surf.append(ztilde_skarray[i])
        #poly_surf = [z_array, ztilde_polyarray, ztilde_skarray]#, 0, 1)
        #print(np.shape(poly_surf))
        poly_axe = [poly_fig.add_subplot(self.poly_degree, 3, plot_id, projection='3d') for plot_id in n_plots]
        [poly_axe[i-1].plot_surface(x, y, poly_surf[i-1], alpha=0.5, cmap = 'brg_r', linewidth = 0, antialiased = False) for i in n_plots]
        '''
        Ridge Regression
        '''
        ''' MANUAL '''
#        ztilde_ridge = self.doRidgeRegression(X1, z)
#        ztilde_ridge = ztilde_ridge.reshape(-1, self.N)
        ''' SKLEARN '''
        
#        fig = plt.figure(figsize = (10, 10))
        # Linear Regression
#        ax1 = fig.add_subplot(3,3,1, projection='3d')
#        ax2 = fig.add_subplot(3,3,2, projection='3d')
#        ax3 = fig.add_subplot(3,3,3, projection='3d')
        # Ridge Regression
#        ax4 = fig.add_subplot(3,3,4, projection='3d')
#        ax5 = fig.add_subplot(3,3,5, projection='3d')
#        ax6 = fig.add_subplot(3,3,6, projection='3d')
        # Lasso regression
#        ax7 = fig.add_subplot(3,3,7, projection='3d')
#        ax8 = fig.add_subplot(3,3,8, projection='3d')
#        ax9 = fig.add_subplot(3,3,9, projection='3d')
#        surf1 = ax1.plot_surface(x, y, z, alpha=0.5, cmap = 'brg_r', linewidth = 0, antialiased = False)
        #fig.colorbar(surf1, shrink=0.5, aspect=5)
#        surf2 = ax2.plot_surface(x, y, ztilde1, alpha=0.5, cmap = 'brg_r', linewidth = 0, antialiased = False)
#        surf3 = ax3.plot_surface(x, y, ztilde_sk, alpha=0.5, cmap = 'brg_r', linewidth = 0, antialiased = False)
        
#        surf4 = ax4.plot_surface(x, y, z, alpha=0.5, cmap = 'brg_r', linewidth = 0, antialiased = False)
#        surf5 = ax5.plot_surface(x, y, ztilde_ridge, alpha=0.5, cmap = 'brg_r', linewidth = 0, antialiased = False)
        
#        surf7 = ax7.plot_surface(x, y, z, alpha=0.5, cmap = 'brg_r', linewidth = 0, antialiased = False)
        #surf3 = ax3.plot_surface(x, y, ztilde2, alpha=0.5, cmap = 'brg_r', linewidth = 0, antialiased = False)
        plt.show()
    '''
    Generating polynomials for given number of variables for a given degree
    using Newton's Binomial formula, and when returning the design matrix,
    computed from the list of all variables
    '''
    def constructDesignMatrix(self, *args):
        # the degree of polynomial to be generated
        poly_degree = args[0]
        # getting inputs
        x_vals = self.x_vals
        # using itertools for generating all possible combinations 
        # of multiplications between our variables and 1, i.e.:
        # x_0*x_1*1, x_0*x_0*x_1*1 etc. => will get polynomial 
        # coefficients
        variables = list(self.x_symb.copy())
        variables.append(1)
        terms = [sp.Mul(*i) for i in it.combinations_with_replacement(variables, poly_degree)]
        # creating desing matrix
        points = len(x_vals[0])*len(x_vals[1])
        # creating desing matrix composed of ones
        X1 = np.ones((points, len(terms)))
        # populating design matrix with values
        for k in range(len(terms)):
            f = sp.lambdify([self.x_symb[0], self.x_symb[1]], terms[k], "numpy")
            X1[:, k] = [f(i, j) for i in self.x_vals[1] for j in self.x_vals[0]]
        # returning constructed design matrix (for 2 approaches if needed)
        return X1
    '''
    Singular Value Decomposition for Linear Regression
    '''
    def doSVD(self, *args):
        # getting matrix
        X = args[0]
        # Applying SVD
        A = np.transpose(X) @ X
        U, s, VT = np.linalg.svd(A)
        D = np.zeros((len(U), len(VT)))
        for i in range(0,len(VT)):
            D[i,i] = s[i]
        UT = np.transpose(U); 
        V = np.transpose(VT); 
        invD = np.linalg.inv(D)
        invA = np.matmul(V, np.matmul(invD, UT))
        
        return invA
    
    '''
    KFold Cross validation
    '''
    def doCrossVal(self, *args):
        # getting design matrix
        X = args[0]
        # getting z values and making them 1d
        z = np.ravel(args[1])
        # Splitting and shuffling data randomly
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=1./self.kfold, shuffle = True)
        MSE = []
        z_trained = []
        for i in range(self.kfold):
            # Train The Pipeline
            invA = self.doSVD(X_train)
            beta_train = invA.dot(X_train.T).dot(z_train)
            # Testing the pipeline
            z_trained.append(X_test @ beta_train)
            # Calculating MSE for each iteration
            MSE.append(self.getMSE(z_test, z_trained[i]))
        MSE_tot = np.mean(MSE)
    '''
    #============================#
    # Regression Methods
    #============================#
    '''
    '''
    Polynomial Regression - does linear regression analysis with our generated 
    polynomial and returns the predicted values (our model) <= k-fold cross 
    validation has been implemented
    '''
    def doLinearRegression(self, *args):
        # getting design matrix
        X = args[0]
        # getting z values and making them 1d
        z = np.ravel(args[1])
        # Splitting and shuffling data randomly
#        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=1./self.kfold, shuffle = True)
        ''' Applying k-Fold cross validation '''
#        MSE = []
#        z_trained = []
#        for i in range(self.kfold):
            # Train The Pipeline
#            invA = self.doSVD(X_train)
#            beta_train = invA.dot(X_train.T).dot(z_train)
            # Testing the pipeline
#            z_trained.append(X_test @ beta_train)
            # Calculating MSE for each iteration
#            MSE.append(self.getMSE(z_test, z_trained[i]))
#        MSE_tot = np.mean(MSE)
#        print(MSE_tot)
        # and then make the prediction
        invA = self.doSVD(X)
        beta = invA.dot(X.T).dot(z)
        ztilde = X @ beta
        # calculating beta confidence
        confidence = 1.96
        sigma = 1
        SE = sigma * np.sqrt(np.diag(invA)) * confidence
        
        return ztilde#, beta#z_trained#ztilde#, beta, SE
    '''
    Ridge Regression
    '''
    def doRidgeRegression(self, *args):
        # Generating the 500 values of lambda
        # to tune our model (i.e. we will calculate scores
        # and decide which parameter lambda is best suited for our model)
        nlambdas = 500
        lambdas = np.logspace(-3, 5, nlambdas)
        lambda_par = 0.1
        # getting design matrix
        X = args[0]
        # getting z values
        z = np.ravel(args[1])
        # constructing the identity matrix
        I = np.identity(np.shape(X.T.dot(X)), dtype = float)
        # calculating parameters
        beta = np.linalg.inv(X.T.dot(X) + lambda_par * I).dot(X.T).dot(z)
        # and making predictions
        ztilde = X @ beta

        return ztilde#, beta, SE
    
    '''
    LASSO Regression
    '''
    def doLASSORegression(self, *args):
        None
    
    '''
    MSE - the smaller the better (0 is the best?)
    '''
    def getMSE(self, z_data, z_model):
        n = np.size(z_model)
        return np.sum((z_data-z_model)**2)/n
    '''
    R^2 - values should be between 0 and 1 (with 1 being the best)
    '''
    def getR2(self, z_data, z_model):
        return 1 - np.sum((z_data - z_model) ** 2) / np.sum((z_data - np.mean(z_data)) ** 2)

if __name__ == '__main__':
    # Start time of the program
    start_time = time.time()
    # number of points
    N_points = 100
    # number of independent variables (features)
    n_vars = 2
    # polynomial degree
    max_poly_degree = 5
    # the amount of folds to get from your data
    kfold = 5
    pipeline = MainPipeline(N_points, n_vars, max_poly_degree, kfold)
    pipeline.main(n_vars, max_poly_degree)
    # End time of the program
    end_time = time.time()
    print("-- Program finished at %s sec --" %(end_time - start_time))