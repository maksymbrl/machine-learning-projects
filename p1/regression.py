#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:13:27 2019

@author: maksymb
"""

import numpy as np
# for polynimial manipulation
import sympy as sp
#from sympy import *
import itertools as it
# importing my library
import reglib as rl


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
# to use latex symbols
#from matplotlib import rc


# Machine Learning libraries
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import time



#rc('text', usetex=True)
#rc('text.latex', preamble=r'\usepackage{amssymb}')

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
        # to calculate confidence intervals
        
        # generating an array of symbolic variables 
        # based on the desired amount of variables
        self.x_symb = sp.symarray('x', self.n_vars, real=True)
        # making a copy of this array
        self.x_vals = self.x_symb.copy()
        # and fill it with values
        for i in range(self.n_vars):
            self.x_vals[i] = np.sort(np.random.uniform(0, 1, self.N))
            
        
        self.pipe = rl.RegressionPipeline(self.N, self.n_vars,\
                                          self.poly_degree, self.kfold,\
                                          self.x_symb, self.x_vals, \
                                          confidence, sigma)
    
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
        confidence = args[0]
        sigma = args[1]
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
        ztilde_polyarray = [self.doLinearRegression(X, z, confidence, sigma).reshape(-1, self.N) for X in X_polyarray]
        
        MSE_lin = [self.doCrossVal(X_polyarray[i], ztilde_polyarray[i]) for i in range(0, self.poly_degree)]
        fig = plt.figure(figsize = (10, 10))
        ax1 = fig.add_subplot(3,3,1)
        surf1 = ax1.plot(range(1,self.poly_degree+1), MSE_lin)
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
        plt.show()   
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
        
        MSE_lintot = []
        z_trained = []
        for i in range(self.kfold):

            # Train The Pipeline
            invA = self.doSVD(X_train)
            beta_train = invA.dot(X_train.T).dot(z_train)
            #invA = self.doSVD(X_test)
            #beta_test = invA.dot(X_test.T).dot(z_test)
            # Testing the pipeline
            z_trained.append(X_test @ beta_train)
            # Calculating MSE for each iteration
            MSE_lintot.append(self.getMSE(z_test, z_trained[i]))
        # linear MSE
        MSE_lin = np.mean(MSE_lintot)
        #print(MSE_lin)
        #fig = plt.figure(figsize = (10, 10))
        #ax1 = fig.add_subplot(3,3,1)
        #plt.show()
        return MSE_lin

#if __name__ == '__main__':
    # Start time of the program
#    start_time = time.time()
    # number of points
#    N_points = 100
    # number of independent variables (features)
#    n_vars = 2
    # polynomial degree
#    max_poly_degree = 5
    # the amount of folds to get from your data
#    kfold = 5
    # to calculate confidence intervals
#    confidence = 1.96
#    sigma = 1
#    pipeline = MainPipeline(N_points, n_vars, max_poly_degree, kfold)
#    pipeline.main(confidence, sigma)#n_vars, max_poly_degree)
    # End time of the program
#    end_time = time.time()
#    print("-- Program finished at %s sec --" %(end_time - start_time))
    
    
    
    
    
    
    
    
    
    
    
# Initialising global variables 
# (I know it is not the best way to do things in python, but let it be for now)
kFoldMSEtest_lin    = []
kFoldMSEtrain_lin   = []
kFoldMSEtest_ridge  = []
kFoldMSEtrain_ridge = []
kFoldMSEtest_lasso  = []
    
    
    
class MainPipeline:
    ''' class constructor '''
    def __init__(self, *args):
        # number of points
        self.N = args[0]
        # number of independent variables
        self.n_vars = args[1]
        # polynomial degree
        self.poly_degree = args[2]
        # k-value
        self.kfold = args[3]
        # to calculate beta confidence intervals
        self.confidence = args[4]
        self.sigma = args[5]
        # hyperparameter
        self.lambda_par = args[6]
    
    '''
    method to work with fake data, i.e. created using uniform distribution
    '''
    def doFakeData(self, *args):
        # creating a starting value for number generator
        # so each time we will get the same random numbers
        np.random.seed(1)
        # generating an array of symbolic variables 
        # based on the desired amount of variables
        self.x_symb = sp.symarray('x', self.n_vars, real=True)
        # making a copy of this array
        self.x_vals = self.x_symb.copy()
        # and fill it with values
        for i in range(self.n_vars):
            self.x_vals[i] = np.sort(np.random.uniform(0, 1, self.N))
        # library object instantiation
        lib = rl.RegressionPipeline(self.x_symb, self.x_vals)
        # generating output data - first setting-up the proper grid
        x, y = np.meshgrid(self.x_vals[0], self.x_vals[1])
        # and creating an output based on the input
        z = lib.FrankeFunction(x, y) + 0.1 * np.random.randn(self.N)
        # getting design matrix
        X = lib.constructDesignMatrix(self.poly_degree)
        
        ''' Linear Regression '''
        ''' MANUAL '''
        # getting predictions
        ztilde_lin, beta_lin, beta_min, beta_max = lib.doLinearRegression(X, z, self.confidence, self.sigma)
        ztilde_lin = ztilde_lin.reshape(-1, self.N)
        ''' Scikit Learn '''
        # generate polynomial
        poly_features = PolynomialFeatures(degree = self.poly_degree)
        # [[x[0], y[0]],[x[1],y[1]], [x[2],y[2]], ...]
        # works mush better than the transpose and reshape
        # (so far reshape, without "F" order was giving crap)
        X_scikit = np.swapaxes(np.array([self.x_vals[0], self.x_vals[1]]), 0, 1)
        X_poly = poly_features.fit_transform(X_scikit)
        lin_reg = LinearRegression().fit(X_poly, z)
        ztilde_sk = lin_reg.predict(X_poly)
        zarray_lin = [z, ztilde_lin, ztilde_sk]
        print('\n')
        print("Linear MSE (no CV) - " + str(lib.getMSE(z, ztilde_lin)) + "; sklearn - " + str(mean_squared_error(z, ztilde_sk)))
        print("Linear R^2 (no CV) - " + str(lib.getR2(z, ztilde_lin)) + "; sklearn - " + str(lin_reg.score(X_poly, z)))
        print('\n')
        ''' Plotting Surfaces '''
        fig = plt.figure(figsize = (10, 3))
        fig.suptitle('Linear Regression', fontsize=14)
        axes = [fig.add_subplot(1, 3, i, projection='3d') for i in range(1, 4)]
        surf = [axes[i].plot_surface(x, y, zarray_lin[i], alpha=0.5, \
                cmap = 'brg_r', linewidth = 0, antialiased = False) for i in range(3)]
        # betas
        fig = plt.figure(figsize = (10, 3))
        ax1 = fig.add_subplot(1, 1, 1)
        t = []
        [t.append(i) for i in range(1, len(beta_lin)+1)]
        ax1.plot(t, beta_lin, 'bo', label = r'$\beta$')
        ax1.plot(t, beta_min, 'r--', label = r'$\beta_{min}$')
        ax1.plot(t, beta_max, 'g--', label = r'$\beta_{max}$')
        ax1.legend()
        plt.grid(True)
        plt.xlabel('number of ' + r'$\beta$')
        plt.ylabel(r'$\beta$')

        # Calculating k-Fold Cross Validation
        global kFoldMSEtest_lin, kFoldMSEtrain_lin
        kFoldMSEtest_lin.append(lib.doCrossVal(X, z, self.kfold)[0])
        kFoldMSEtrain_lin.append(lib.doCrossVal(X, z, self.kfold)[1])
        
        ''' Ridge Regression '''
        ''' MANUAL '''
        ztilde_ridge, beta_ridge, beta_min, beta_max = lib.doRidgeRegression(X, z, self.lambda_par, self.confidence, self.sigma)
        ztilde_ridge = ztilde_ridge.reshape(-1, self.N)
        ''' Scikit Learn '''
        ridge_reg = Ridge(alpha = self.lambda_par, fit_intercept = True).fit(X_poly, z)
        ztilde_sk = ridge_reg.predict(X_poly)
        zarray_ridge = [z, ztilde_ridge, ztilde_sk]
        print('\n')
        print("Ridge MSE (no CV) - " + str(lib.getMSE(z, ztilde_ridge)) + "; sklearn - " + str(mean_squared_error(z, ztilde_sk)))
        print("Ridge R^2 (no CV) - " + str(lib.getR2(z, ztilde_ridge)) + "; sklearn - " + str(ridge_reg.score(X_poly, z)))
        print('\n')
        ''' Plotting Surfaces '''
        fig = plt.figure(figsize = (10, 3))
        fig.suptitle('Ridge Regression', fontsize=14)
        axes = [fig.add_subplot(1, 3, i, projection='3d') for i in range(1, 4)]
        surf = [axes[i].plot_surface(x, y, zarray_ridge[i], alpha=0.5,\
                cmap = 'brg_r', linewidth = 0, antialiased = False) for i in range(3)]
                # betas
        print('\n')
        fig = plt.figure(figsize = (10, 3))
        ax1 = fig.add_subplot(1, 1, 1)
        t = []
        [t.append(i) for i in range(1, len(beta_lin)+1)]
        ax1.plot(t, beta_ridge, 'bo', label = r'$\beta$')
        ax1.plot(t, beta_min, 'r--', label = r'$\beta_{min}$')
        ax1.plot(t, beta_max, 'g--', label = r'$\beta_{max}$')
        ax1.legend()
        plt.grid(True)
        plt.xlabel('number of ' + r'$\beta$')
        plt.ylabel(r'$\beta$')
        print('\n')
        # Calculating k-Fold Cross Validation
        global kFoldMSEtest_ridge, kFoldMSEtrain_ridge
        kFoldMSEtest_ridge.append(lib.doCrossValRidge(X, z, self.kfold, self.lambda_par)[0])
        kFoldMSEtrain_ridge.append(lib.doCrossValRidge(X, z, self.kfold, self.lambda_par)[1])
        
        ''' LASSO Regression '''
        lasso_reg = Lasso(alpha = self.lambda_par).fit(X_poly, z)
        ztilde_sk = lasso_reg.predict(X_poly)
        zarray_lasso = [z, ztilde_sk]
        print('\n')
        print("SL Lasso MSE (no CV) - " + str(mean_squared_error(z, ztilde_sk)))
        print("SL Lasso R^2 (no CV) - " + str(lasso_reg.score(X_poly, z)))
        print('\n')
        ''' Plotting Surfaces '''
        fig = plt.figure(figsize = (10, 3))
        fig.suptitle('Lasso Regression', fontsize=14)
        axes = [fig.add_subplot(1, 3, i, projection='3d') for i in range(1, 3)]
        surf = [axes[i].plot_surface(x, y, zarray_lasso[i], alpha=0.5,\
                cmap = 'brg_r', linewidth = 0, antialiased = False) for i in range(2)]
        
        plt.show()
        
        # Calculating k-Fold Cross Validation
        global kFoldMSE_lasso
        
    '''
    method to work with REAL data, i.e. downloaded from web
    '''
    def doRealData(self, *args):
        pass
    
    
if __name__ == '__main__':
    # Start time of the program
    start_time = time.time()
    
    ''' Input Parameters '''
    # number of points
    N_points = 100
    # number of independent variables (features)
    n_vars = 2
    # polynomial degree
    max_poly_degree = 10
    # the amount of folds to get from your data
    kfold = 5
    # to calculate confidence intervals
    confidence = 1.96
    sigma = 1
    lambda_par = 0.001
    # object class instantiation
    print('Fake Data')
    for poly_degree in range(1, max_poly_degree+1):
        print('\n')
        print('Starting analysis for polynomial of degree: ' + str(poly_degree))
        pipeline = MainPipeline(N_points, n_vars, poly_degree,\
                            kfold, confidence, sigma, lambda_par)
        # Linear regression on fake data
        pipeline.doFakeData()
    
    ''' MSE as a function of model complexity '''
    # plotting MSE from test data
    fig = plt.figure(figsize = (10, 3))
    ax1 = fig.add_subplot(1, 1, 1)
    t = []
    [t.append(i) for i in range(1, max_poly_degree+1)]
    ax1.plot(t, kFoldMSEtest_lin, 'bo', label = 'test')
    ax1.plot(t, kFoldMSEtrain_lin, 'r--', label = 'train')
    ax1.legend()
    plt.grid(True)
    plt.title('MSE as a function of model complexity; Linear Regression')
    plt.xlabel('model complexity (polynomial degree)')
    plt.ylabel('MSE')
    
    fig = plt.figure(figsize = (10, 3))
    ax1 = fig.add_subplot(1, 1, 1)
    t = []
    [t.append(i) for i in range(1, max_poly_degree+1)]
    ax1.plot(t, kFoldMSEtest_ridge, 'bo', label = 'test')
    ax1.plot(t, kFoldMSEtrain_ridge, 'r--', label = 'train')
    ax1.legend()
    plt.grid(True)
    plt.title('MSE as a function of model complexity; Ridge Regression')
    plt.xlabel('model complexity (polynomial degree)')
    plt.ylabel('MSE')
    
    
    
    print('\n')
    print('Real Data')
    # End time of the program
    end_time = time.time()
    print("-- Program finished at %s sec --" %(end_time - start_time))
    
    
    
    
    
    
    
    
    
    
    
    
    