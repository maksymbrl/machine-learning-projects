#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:13:27 2019

@author: maksymb
"""

# standard imports
import os, sys
import numpy as np
# for polynomial manipulation
import sympy as sp
# from sympy import *
import itertools as it
# importing my library
import reglib as rl

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Qt5Agg')
from matplotlib import cm
# to use latex symbols
# from matplotlib import rc

# to read tif files
from imageio import imread

# Machine Learning libraries
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import time

# rc('text', usetex=True)
# rc('text.latex', preamble=r'\usepackage{amssymb}')

class MainPipeline(object):
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
        # hyper parameter
        self.lambda_par = args[6]
        # directory to save plots
        self.output_dir = args[7]

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
            self.x_vals[i] = np.arange(0, 1, 1/self.N)#np.sort(np.random.uniform(0, 1, self.N))
        # library object instantiation
        lib = rl.RegressionPipeline(self.x_symb, self.x_vals)
        # generating output data - first setting-up the proper grid
        x, y = np.meshgrid(self.x_vals[0], self.x_vals[1])
        # and creating an output based on the input
        z = lib.FrankeFunction(x, y) + 0.1 * np.random.randn(self.N, self.N)
        # raveling variables
        x_rav, y_rav, z_rav = np.ravel(x), np.ravel(y), np.ravel(z)

        ''' Linear Regression '''
        ''' MANUAL '''
        # getting design matrix
        X = lib.constructDesignMatrix(self.poly_degree)
        # getting predictions
        ztilde_lin, beta_lin, beta_min, beta_max = lib.doLinearRegression(X, z_rav, self.confidence, self.sigma)
        ztilde_lin = ztilde_lin.reshape(-1, self.N)
        ''' Scikit Learn '''
        # generate polynomial
        poly_features = PolynomialFeatures(degree = self.poly_degree)
        # [[x[0], y[0]],[x[1],y[1]], [x[2],y[2]], ...]
        # works mush better than the transpose and reshape
        # (so far reshape, without "F" order was giving crap)
        X_scikit = np.swapaxes(np.array([x_rav, y_rav]), 0, 1)
        X_poly = poly_features.fit_transform(X_scikit)
        lin_reg = LinearRegression().fit(X_poly, z_rav)
        ztilde_sk = lin_reg.predict(X_poly).reshape(-1, self.N)
        zarray_lin = [z, ztilde_lin, ztilde_sk]
        # Errors
        print('\n')
        print("Linear MSE (no CV) - " + str(lib.getMSE(zarray_lin[0], zarray_lin[1])) + "; sklearn - " + str(mean_squared_error(zarray_lin[0], zarray_lin[2])))
        print("Linear R^2 (no CV) - " + str(lib.getR2(zarray_lin[0], zarray_lin[1])) + "; sklearn - " + str(lin_reg.score(X_poly, z_rav)))
        print('\n')
        ''' Plotting Surfaces '''
        filename = 'fake_linear_p' + str(self.poly_degree).zfill(2) + '.png'
        # calling method from library to do this for us
        lib.plotSurface(x, y, zarray_lin, self.output_dir, filename)
        # betas
        filename = 'fake_linear_beta_p' + str(self.poly_degree).zfill(2) + '.png'
        t = []
        [t.append(i) for i in range(1, len(beta_lin) + 1)]
        lib.plotBeta(t, beta_lin, beta_min, beta_max, output_dir, filename)
        # Calculating k-Fold Cross Validation
        self.fake_kFoldMSEtest_lin = lib.doCrossVal(X, z, self.kfold)[0]
        self.fake_kFoldMSEtrain_lin = lib.doCrossVal(X, z, self.kfold)[1]
        ''' Ridge Regression '''
        ''' MANUAL '''
        ztilde_ridge, beta_ridge, beta_min, beta_max = lib.doRidgeRegression(X, z_rav, self.lambda_par, self.confidence, self.sigma)
        ztilde_ridge = ztilde_ridge.reshape(-1, self.N)
        ''' Scikit Learn '''
        ridge_reg = Ridge(alpha=self.lambda_par, fit_intercept=True).fit(X_poly, z_rav)
        ztilde_sk = ridge_reg.predict(X_poly).reshape(-1, self.N)
        zarray_ridge = [z, ztilde_ridge, ztilde_sk]
        print('\n')
        print("Ridge MSE (no CV) - " + str(lib.getMSE(zarray_ridge[0], zarray_ridge[1])) + "; sklearn - " + str(mean_squared_error(zarray_ridge[0], zarray_ridge[2])))
        print("Ridge R^2 (no CV) - " + str(lib.getR2(zarray_ridge[0], zarray_ridge[1])) + "; sklearn - " + str(ridge_reg.score(X_poly, z_rav)))
        print('\n')
        ''' Plotting Surfaces '''
        filename = 'fake_ridge_p' + str(self.poly_degree).zfill(2) + '.png'
        lib.plotSurface(x, y, zarray_ridge, self.output_dir, filename)
        # betas
        filename = 'fake_ridge_beta_p' + str(self.poly_degree).zfill(2) + '.png'
        t = []
        [t.append(i) for i in range(1, len(beta_lin) + 1)]
        lib.plotBeta(t, beta_ridge, beta_min, beta_max, output_dir, filename)
        # Calculating k-Fold Cross Validation
        self.fake_kFoldMSEtest_ridge = lib.doCrossValRidge(X, z, self.kfold, self.lambda_par)[0]
        self.fake_kFoldMSEtrain_ridge = lib.doCrossValRidge(X, z, self.kfold, self.lambda_par)[1]

        ''' LASSO Regression '''
        lasso_reg = Lasso(alpha=self.lambda_par).fit(X_poly, np.ravel(z))
        ztilde_sk = lasso_reg.predict(X_poly).reshape(-1, self.N)
        zarray_lasso = [z, ztilde_sk]
        print('\n')
        print("SL Lasso MSE (no CV) - " + str(mean_squared_error(zarray_lasso[0], zarray_lasso[1])))
        print("SL Lasso R^2 (no CV) - " + str(lasso_reg.score(X_poly, z_rav)))
        print('\n')
        ''' Plotting Surfaces '''
        filename = 'fake_lasso_p' + str(self.poly_degree).zfill(2) + '.png'
        lib.plotSurface(x, y, zarray_lasso, self.output_dir, filename)

        # Calculating k-Fold Cross Validation
        #global kFoldMSE_lasso

    '''
    method to work with REAL data, i.e. downloaded from web
    '''

    def doRealData(self, *args):
        z = args[0]
        #print(np.shape(z))
        #for i in range(self.n_vars):
        #    self.x_vals[i] = np.arange(0, , self.N)
        self.x_vals[0] = np.linspace(0, 1801, 1801)
        self.x_vals[1] = np.linspace(0, 3601, 3601)
        # generating output data - first setting-up the proper grid
        x, y = np.meshgrid(self.x_vals[0], self.x_vals[1])
        # raveling variables
        x_rav, y_rav, z_rav = np.ravel(x), np.ravel(y), np.ravel(z)
        # library object instantiation
        lib = rl.RegressionPipeline(self.x_symb, self.x_vals)
        ''' Linear Regression '''
        ''' MANUAL '''
        # getting design matrix
        X = lib.constructDesignMatrix(self.poly_degree)
        # getting predictions
        ztilde_lin, beta_lin, beta_min, beta_max = lib.doLinearRegression(X, z_rav, self.confidence, self.sigma)
        ztilde_lin = ztilde_lin.reshape(np.shape(z))
        #print(np.shape(ztilde_lin))
        ''' Scikit Learn '''
        # generate polynomial
        poly_features = PolynomialFeatures(degree = self.poly_degree)
        # [[x[0], y[0]],[x[1],y[1]], [x[2],y[2]], ...]
        # works mush better than the transpose and reshape
        # (so far reshape, without "F" order was giving crap)
        X_scikit = np.swapaxes(np.array([x_rav, y_rav]), 0, 1)
        X_poly = poly_features.fit_transform(X_scikit)
        lin_reg = LinearRegression().fit(X_poly, z_rav)
        ztilde_sk = lin_reg.predict(X_poly).reshape(np.shape(z))
        zarray_lin = [z, ztilde_lin, ztilde_sk]
        # Errors
        print('\n')
        print("Linear MSE (no CV) - " + str(lib.getMSE(zarray_lin[0], zarray_lin[1])) + "; sklearn - " + str(mean_squared_error(zarray_lin[0], zarray_lin[2])))
        print("Linear R^2 (no CV) - " + str(lib.getR2(zarray_lin[0], zarray_lin[1])) + "; sklearn - " + str(lin_reg.score(X_poly, z_rav)))
        print('\n')
        ''' Plotting Surfaces '''
        filename = 'real_linear_p' + str(self.poly_degree).zfill(2) + '.png'
        # calling method from library to do this for us
        lib.plotSurface(x, y, zarray_lin, self.output_dir, filename)
        # betas
        filename = 'real_linear_beta_p' + str(self.poly_degree).zfill(2) + '.png'
        t = []
        [t.append(i) for i in range(1, len(beta_lin) + 1)]
        lib.plotBeta(t, beta_lin, beta_min, beta_max, output_dir, filename)
        # Calculating k-Fold Cross Validation
        self.real_kFoldMSEtest_lin = lib.doCrossVal(X, z, self.kfold)[0]
        self.real_kFoldMSEtrain_lin = lib.doCrossVal(X, z, self.kfold)[1]
        ''' Ridge Regression '''
        ''' MANUAL '''
        ztilde_ridge, beta_ridge, beta_min, beta_max = lib.doRidgeRegression(X, z_rav, self.lambda_par, self.confidence, self.sigma)
        ztilde_ridge = ztilde_ridge.reshape(np.shape(z))
        ''' Scikit Learn '''
        ridge_reg = Ridge(alpha = self.lambda_par, fit_intercept=True).fit(X_poly, z_rav)
        ztilde_sk = ridge_reg.predict(X_poly).reshape(np.shape(z))
        zarray_ridge = [z, ztilde_ridge, ztilde_sk]
        print('\n')
        print("Ridge MSE (no CV) - " + str(lib.getMSE(zarray_ridge[0], zarray_ridge[1])) + "; sklearn - " + str(mean_squared_error(zarray_ridge[0], zarray_ridge[2])))
        print("Ridge R^2 (no CV) - " + str(lib.getR2(zarray_ridge[0], zarray_ridge[1])) + "; sklearn - " + str(ridge_reg.score(X_poly, z_rav)))
        print('\n')
        ''' Plotting Surfaces '''
        filename = 'real_ridge_p' + str(self.poly_degree).zfill(2) + '.png'
        lib.plotSurface(x, y, zarray_ridge, self.output_dir, filename)
        # betas
        filename = 'real_ridge_beta_p' + str(self.poly_degree).zfill(2) + '.png'
        t = []
        [t.append(i) for i in range(1, len(beta_lin) + 1)]
        lib.plotBeta(t, beta_ridge, beta_min, beta_max, output_dir, filename)
        # Calculating k-Fold Cross Validation
        self.real_kFoldMSEtest_ridge = lib.doCrossValRidge(X, z, self.kfold, self.lambda_par)[0]
        self.real_kFoldMSEtrain_ridge = lib.doCrossValRidge(X, z, self.kfold, self.lambda_par)[1]


    '''
    Method to return calculated values (surface plots, MSEs, betas etc.), based on the user input choice.
    '''
    def doStuff(self, *args):
        pass

if __name__ == '__main__':
    # Start time of the program
    start_time = time.time()
    # Creating output directory to save plots (in  png format)
    output_dir = 'Output'
    # checking whether the output directory already exists (if not, create one)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    '''
    Yes/no question - based on the user input will return. Basically, the user is asked whether
    he/she wants to run script on real data. Is the answer is no, then the data set will be simulated
    Is the based on uniform distribution/linear grid.
    
    The answer returns True for 'yes' and False for 'no'
    '''
    sys.stdout.write('Do you want to use real data? ([y]/[n]) ')
    value = None
    while value == None:
        # User input - making it lower case
        choice = input().lower()
        # possible answers
        yes = {'yes': True, 'ye': True, 'yeah': True, 'y': True, '': True}
        no = {'no': False, 'n': False}
        if choice in yes:
            value = True
        elif choice in no:
            value = False
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or keep the field empty): ")

    if value == True:
        print('''
        #========================#
        # Working with Real Data #
        #========================#        
        ''')
        sys.stdout.write("Please, provide path to data file (default = Data/SRTM_data_Norway_1.tif): ")
        # Load the terrain
        terrain = input()
        if terrain == '':
            terrain = imread('Data/SRTM_data_Norway_1.tif')
        else:
            terrain = imread(terrain)
        # number of independent variables (features)
        n_vars = 2
        # max polynomial degree
        sys.stdout.write("Please, choose the max polynomial degree (default = 5): ")
        max_poly_degree = input()
        if max_poly_degree == '':
            max_poly_degree = 5
        else:
            max_poly_degree = int(max_poly_degree)
        # just to save your value (e.g. png(s)) under correct prefix
        prefix = 'real'
        # the amount of folds to get from your data
        kfold = 5
        # to calculate confidence intervals
        confidence = 1.96
        sigma = 1
        # lasso very sensitive to this lambda parameter
        lambda_par = 0.000001
        # just to save your value (e.g. png(s)) under correct prefix
        prefix = 'fake'
        ''' Generating Data Set '''
        # generating an array of symbolic variables
        # based on the desired amount of variables
        x_symb = sp.symarray('x', n_vars, real = True)
        # making a copy of this array
        x_vals = x_symb.copy()
        x_vals[0] = np.linspace(0, len(terrain[0]), len(terrain[0]))
        x_vals[1] = np.linspace(0, len(terrain), len(terrain))
        # library object instantiation
        lib = rl.RegressionPipeline(x_symb, x_vals)
        # generating output data - first setting-up the proper grid
        x, y = np.meshgrid(x_vals[0], x_vals[1])
        z = terrain

    elif value == False:
        print('''
        #========================#
        # Working with Fake Data #
        #========================# 
        ''')
        ''' Input Parameters'''
        sys.stdout.write("Please, choose an amount of points to simulate data (default = 50): ")
        # Data points to simulate grid (N_points x N_points)
        N_points = input()
        if N_points == '':
            N_points = 50
        else:
            N_points = int(N_points)
        # number of independent variables (features)
        n_vars = 2
        # max polynomial degree
        sys.stdout.write("Please, choose the max polynomial degree (default = 5): ")
        max_poly_degree = input()
        if max_poly_degree == '':
            max_poly_degree = 5
        else:
            max_poly_degree = int(max_poly_degree)
        # the amount of folds to get from your data
        kfold = 5
        # to calculate confidence intervals
        confidence = 1.96
        sigma = 1
        # lasso very sensitive to this lambda parameter
        lambda_par = 0.000001
        # just to save your value (e.g. png(s)) under correct prefix
        prefix = 'fake'
        ''' Generating Data Set '''
        # generating an array of symbolic variables
        # based on the desired amount of variables
        x_symb = sp.symarray('x', n_vars, real = True)
        # making a copy of this array
        x_vals = x_symb.copy()
        # and fill it with values
        for i in range(n_vars):
            x_vals[i] = np.arange(0, 1, 1./N_points)#np.sort(np.random.uniform(0, 1, N_points))
        # library object instantiation
        lib = rl.RegressionPipeline(x_symb, x_vals)
        # setting up the grid
        x, y = np.meshgrid(x_vals[0], x_vals[1])
        # and getting output based on the Franke Function
        z = lib.FrankeFunction(x, y) + 0.1 * np.random.randn(N_points, N_points)

    # looping through all polynomial degrees

    time.sleep(100)

    ''' Working with Fake Data '''
    ''' Input Parameters '''
    # number of points
    N_points = 50
    # number of independent variables (features)
    n_vars = 2
    # polynomial degree
    max_poly_degree = 5
    # the amount of folds to get from your data
    kfold = 5
    # to calculate confidence intervals
    confidence = 1.96
    sigma = 1
    # lasso very sensitive to this lambda parameter
    lambda_par = 0.000001
    # object class instantiation
    print(
        '''
        #========================#
        # Working with Fake Data #
        #========================#        
        '''
    )

    kFoldMSEtest_lin = []
    kFoldMSEtrain_lin = []
    kFoldMSEtest_ridge = []
    kFoldMSEtrain_ridge = []
    kFoldMSEtest_lasso = []

    for poly_degree in range(1, max_poly_degree + 1):
        print('\n')
        print('Starting analysis for polynomial of degree: ' + str(poly_degree))
        pipeline = MainPipeline(N_points, n_vars, poly_degree,
                                kfold, confidence, sigma, lambda_par, output_dir)
        # Linear regression on fake data
        pipeline.doFakeData()
        # linear regression kfold
        kFoldMSEtest_lin.append(pipeline.fake_kFoldMSEtest_lin)
        kFoldMSEtrain_lin.append(pipeline.fake_kFoldMSEtrain_lin)
        # ridge regression kfold
        kFoldMSEtest_ridge.append(pipeline.fake_kFoldMSEtest_ridge)
        kFoldMSEtrain_ridge.append(pipeline.fake_kFoldMSEtrain_ridge)

    # Turning interactive mode on
    #plt.ion()
    ''' MSE as a function of model complexity '''
    # plotting MSE from test data
    # Linear Regression
    filename = 'fake_linear_mse_p' + str(poly_degree).zfill(2) + '.png'
    fig = plt.figure(figsize = (10, 3))
    ax1 = fig.add_subplot(1, 1, 1)
    t = []
    [t.append(i) for i in range(1, max_poly_degree + 1)]
    ax1.plot(t, kFoldMSEtest_lin, 'bo', label='test')
    ax1.plot(t, kFoldMSEtrain_lin, 'r--', label='train')
    ax1.set_yscale('log')
    ax1.legend()
    plt.grid(True)
    plt.title('MSE as a function of model complexity; Linear Regression')
    plt.xlabel('model complexity (polynomial degree)')
    plt.ylabel('MSE')
    fig.savefig(output_dir + '/' + filename)
    plt.close(fig)

    # Ridge Regression
    filename = 'fake_ridge_mse_p' + str(poly_degree).zfill(2) + '.png'
    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(1, 1, 1)
    t = []
    [t.append(i) for i in range(1, max_poly_degree + 1)]
    ax1.plot(t, kFoldMSEtest_ridge, 'bo', label='test')
    ax1.plot(t, kFoldMSEtrain_ridge, 'r--', label='train')
    ax1.legend()
    plt.grid(True)
    plt.title('MSE as a function of model complexity; Ridge Regression')
    plt.xlabel('model complexity (polynomial degree)')
    plt.ylabel('MSE')
    fig.savefig(output_dir + '/' + filename)
    plt.close(fig)
    #plt.show(block=False)
    # turning the interactive mode off
    #plt.ioff()
    #plt.close('all')

    ''' Working with Real Data '''
    '''
    In this way we are getting Z values, now what I need is to generate X,Y
    using linspace or similar features
    '''
    print('\n')
    print(
        '''
        #========================#
        # Working with Real Data #
        #========================#        
        '''
    )
    # Load the terrain
    terrain1 = imread('Data/SRTM_data_Norway_1.tif')  # <= getting z values, now I need to create my x and y with np.linspace
    #print(np.shape(terrain1))
    # Show the terrain
    #plt.figure()
    #plt.title('Terrain over Norway 1')
    #plt.imshow(terrain1, cmap='gray')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.close()
    #plt.show()

    pipeline.doRealData(terrain1)

    # End time of the program
    end_time = time.time()
    print("-- Program finished at %s sec --" % (end_time - start_time))












