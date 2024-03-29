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
# allowing multiprocessing (because we can? :))
import multiprocessing as mp
from joblib import Parallel, delayed

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
# to generate polynomial (for regression)
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

import time

# rc('text', usetex=True)
# rc('text.latex', preamble=r'\usepackage{amssymb}')

class MainPipeline(object):
    ''' class constructor '''
    def __init__(self, *args):
        #==============================================================================================================#
        # symbolic variables
        self.x_symb = args[0]
        # array of values for each variable/feature
        self.x_vals = args[1]
        # grid values
        self.x = args[2]
        self.y = args[3]
        self.z = args[4]
        # 1.96 to calculate stuff with 95% confidence
        self.confidence = args[5]
        # noise variance - for confidence intervals estimation
        self.sigma = args[6]
        # k-value for Cross validation
        self.kfold = args[7]
        # hyper parameter
        self.lambda_par = args[8]
        # directory where to store plots
        self.output_dir = args[9]
        self.prefix = args[10]
        # degree of polynomial to fit
        self.poly_degree = args[11]
        #==============================================================================================================#

    '''
    Method to return calculated values (surface plots, MSEs, betas etc.), based on the user input choice.
    '''
    def doRegression(self, *args):
        # amount of processors to use
        nproc = args[0]
        # for plotting betas (this valu will appear in the file name <= doesn't affect calculations)
        npoints_name = args[1]
        curr_lambda = args[2]
        # library object instantiation
        lib = rl.RegressionLibrary(self.x_symb, self.x_vals)
        # raveling variables (making them 1d
        x_rav, y_rav, z_rav = np.ravel(self.x), np.ravel(self.y), np.ravel(self.z)
        # shape of z
        zshape = np.shape(self.z)
        #==============================================================================================================#
        ''' Linear Regression '''
        #==============================================================================================================#
        ''' MANUAL '''
        # getting design matrix
        X = lib.constructDesignMatrix(self.poly_degree)
        # getting predictions
        ztilde_lin, beta_lin, beta_min, beta_max = lib.doLinearRegression(X, z_rav, self.confidence, self.sigma)
        ztilde_lin = ztilde_lin.reshape(zshape)
        ''' Scikit Learn '''
        # generate polynomial
        poly_features = PolynomialFeatures(degree = self.poly_degree)
        # [[x[0], y[0]],[x[1],y[1]], [x[2],y[2]], ...]
        # works mush better than the transpose and reshape
        # (so far reshape, without "F" order was giving crap)
        X_scikit = np.swapaxes(np.array([x_rav, y_rav]), 0, 1)
        X_poly = poly_features.fit_transform(X_scikit)
        lin_reg = LinearRegression().fit(X_poly, z_rav)
        ztilde_sk = lin_reg.predict(X_poly).reshape(zshape)
        zarray_lin = [self.z, ztilde_lin, ztilde_sk]
        # Errors
        print('\n')
        print("Linear MSE (no CV) - " + str(lib.getMSE(zarray_lin[0], zarray_lin[1])) + "; sklearn - " + str(mean_squared_error(zarray_lin[0], zarray_lin[2])))
        print("Linear R^2 (no CV) - " + str(lib.getR2(zarray_lin[0], zarray_lin[1])) + "; sklearn - " + str(lin_reg.score(X_poly, z_rav)))
        print('\n')
        ''' Plotting Surfaces '''
        filename = self.prefix + '_linear_p' + str(self.poly_degree).zfill(2) + '_n' + npoints_name +'.png'
        # calling method from library to do this for us
        lib.plotSurface(self.x, self.y, zarray_lin, self.output_dir, filename)
        # betas
        filename = self.prefix + '_linear_beta_p' + str(self.poly_degree).zfill(2) + '_n' + npoints_name + '.png'
        t = []
        [t.append(i) for i in range(1, len(beta_lin) + 1)]
        print("betas are %s" %beta_lin)
        print("beta_min are %s" %beta_min)
        print("beta_max are %s" %beta_max)
        lib.plotBeta(t, beta_lin, beta_min, beta_max, output_dir, filename)
        # write betas to a txt file
        filename = self.prefix + '_linear_beta_p' + str(self.poly_degree).zfill(2) + '_n' + npoints_name + '.txt'
        with open(output_dir + '/' + filename, 'w') as file_handler:
            file_handler.write("beta_lin: {}\n".format(beta_lin))
            print('\n')
            file_handler.write("beta_min: {}\n".format(beta_min))
            print('\n')
            file_handler.write("beta_max: {}\n".format(beta_max))
        ''' kFold Cross Validation '''
        ''' MANUAL '''
        print('kFold CV for Linear Regression - Manual \n')
        self.kFoldMSEtest_lin  = lib.doCrossVal(X, self.z, self.kfold)[0]
        self.kFoldMSEtrain_lin = lib.doCrossVal(X, self.z, self.kfold)[1]
        self.kFoldBias_lin     = lib.doCrossVal(X, self.z, self.kfold)[2]
        self.kFoldVariance_lin = lib.doCrossVal(X, self.z, self.kfold)[3]
        ''' Scikit Learn '''
        reg_type = 'linear'
        self.kFoldMSEtestSK_lin  = lib.doCrossValScikit(X_poly, z_rav, self.kfold, self.poly_degree, self.lambda_par, reg_type)[0]
        self.kFoldMSEtrainSK_lin = lib.doCrossValScikit(X_poly, z_rav, self.kfold, self.poly_degree, self.lambda_par, reg_type)[1]
        self.kFoldBiasSK_lin     = lib.doCrossValScikit(X_poly, z_rav, self.kfold, self.poly_degree, self.lambda_par, reg_type)[2]
        self.kFoldVarianceSK_lin = lib.doCrossValScikit(X_poly, z_rav, self.kfold, self.poly_degree, self.lambda_par, reg_type)[3]

        #==============================================================================================================#
        ''' Ridge Regression '''
        #==============================================================================================================#
        ''' MANUAL '''
        ztilde_ridge, beta_ridge, beta_min, beta_max = lib.doRidgeRegression(X, z_rav, self.lambda_par, self.confidence, self.sigma)
        ztilde_ridge = ztilde_ridge.reshape(zshape)
        ''' Scikit Learn '''
        ridge_reg = Ridge(alpha = self.lambda_par, fit_intercept=True).fit(X_poly, z_rav)
        ztilde_sk = ridge_reg.predict(X_poly).reshape(zshape)
        zarray_ridge = [self.z, ztilde_ridge, ztilde_sk]
        print('\n')
        print("Ridge MSE (no CV) - " + str(lib.getMSE(zarray_ridge[0], zarray_ridge[1])) + "; sklearn - " + str(mean_squared_error(zarray_ridge[0], zarray_ridge[2])))
        print("Ridge R^2 (no CV) - " + str(lib.getR2(zarray_ridge[0], zarray_ridge[1])) + "; sklearn - " + str(ridge_reg.score(X_poly, z_rav)))
        print('\n')
        ''' Plotting Surfaces '''
        filename = self.prefix + '_ridge_p' + str(self.poly_degree).zfill(2) + '_n' + npoints_name + '.png'
        lib.plotSurface(self.x, self.y, zarray_ridge, self.output_dir, filename)
        # betas
        filename = self.prefix + '_ridge_beta_p' + str(self.poly_degree).zfill(2) + '_n' + npoints_name + '.png'
        t = []
        [t.append(i) for i in range(1, len(beta_lin) + 1)]
        print("betas are %s" %beta_ridge)
        print("beta_min are %s" %beta_min)
        print("beta_max are %s" %beta_max)
        lib.plotBeta(t, beta_ridge, beta_min, beta_max, output_dir, filename)
        # write betas to a txt file
        filename = self.prefix + '_ridge_beta_p' + str(self.poly_degree).zfill(2) + '_n' + npoints_name + '.txt'
        with open(output_dir + '/' + filename, 'w') as file_handler:
            file_handler.write("beta_ridge: {}\n".format(beta_ridge))
            print('\n')
            file_handler.write("beta_min: {}\n".format(beta_min))
            print('\n')
            file_handler.write("beta_max: {}\n".format(beta_max))
        # Calculating k-Fold Cross Validation
        print('kFold CV for Ridge Regression - Manual \n')
        #curr_lambda = 0.1
        # parallel processing
        manager = mp.Manager()
        # making a dictionary of values which will save "mse mean value"
        self.kFoldMSEtest_ridge  = manager.dict()
        self.kFoldMSEtrain_ridge = manager.dict()
        self.kFoldBias_ridge     = manager.dict()
        self.kFoldVariance_ridge = manager.dict()
        lambdas = []
        # decreasing the value of lambda till certain minimal value
        # and updating the list
        while curr_lambda >= self.lambda_par:
            lambdas.append(curr_lambda)
            curr_lambda = curr_lambda/10
        # creating a method to work with dictionaries (in parallel)
        def doMultiproc1(kFoldMSEtest_ridge, kFoldMSEtrain_ridge, kFoldBias_ridge, kFoldVariance_ridge, lib, X, z, kfold, curr_lambda):
            print('Ridge Manual \n')
            print("Starting to calculate for $\lambda$=%s" %curr_lambda + '\n')
            kFoldMSEtest_ridge[curr_lambda]  = lib.doCrossValRidge(X, z, kfold, curr_lambda)[0]
            kFoldMSEtrain_ridge[curr_lambda] = lib.doCrossValRidge(X, z, kfold, curr_lambda)[1]
            kFoldBias_ridge[curr_lambda]     = lib.doCrossValRidge(X, z, kfold, curr_lambda)[2]
            kFoldVariance_ridge[curr_lambda] = lib.doCrossValRidge(X, z, kfold, curr_lambda)[3]
            # and exit!
            return
        # using library for parallel processing to loop through all lambda parameters
        Parallel(n_jobs = nproc, verbose = 2)(delayed(doMultiproc1)(self.kFoldMSEtest_ridge, self.kFoldMSEtrain_ridge,
                                                                    self.kFoldBias_ridge, self.kFoldVariance_ridge,
                                                       lib, X, self.z, self.kfold, curr_lambda) for curr_lambda in lambdas)
        ''' Scikit Learn '''
        print('kFold CV for Ridge Regression - Scikit Learn \n')
        # chosing the type of regression
        reg_type = 'ridge'
        # making a dictionary of values which will save "mse mean value"
        # parallel processing
        self.kFoldMSEtestSK_ridge  = manager.dict()
        self.kFoldMSEtrainSK_ridge = manager.dict()
        self.kFoldBiasSK_ridge     = manager.dict()
        self.kFoldVarianceSK_ridge = manager.dict()
        # creating a method to work with dictionaries (in parallel)
        def doMultiproc2(kFoldMSEtestSK_ridge, kFoldMSEtrainSK_ridge, kFoldBiasSK_ridge, kFoldVarianceSK_ridge,
                         lib, X_poly, z_rav, kfold, poly_degree, curr_lambda, reg_type):
            print('Ridge Scikit Learn \n')
            print("Starting to calculate for $\lambda$=%s" %curr_lambda + '\n')
            kFoldMSEtestSK_ridge[curr_lambda]  = lib.doCrossValScikit(X_poly, z_rav, kfold, poly_degree, curr_lambda, reg_type)[0]
            kFoldMSEtrainSK_ridge[curr_lambda] = lib.doCrossValScikit(X_poly, z_rav, kfold, poly_degree, curr_lambda, reg_type)[1]
            kFoldBiasSK_ridge[curr_lambda]     = lib.doCrossValScikit(X_poly, z_rav, kfold, poly_degree, curr_lambda, reg_type)[2]
            kFoldVarianceSK_ridge[curr_lambda] = lib.doCrossValScikit(X_poly, z_rav, kfold, poly_degree, curr_lambda, reg_type)[3]
            # and exit!
            return
        Parallel(n_jobs = nproc, verbose = 2)(delayed(doMultiproc2)(self.kFoldMSEtestSK_ridge, self.kFoldMSEtrainSK_ridge,
                                                                    self.kFoldBiasSK_ridge, self.kFoldVarianceSK_ridge, lib, X_poly,
                                                       z_rav, self.kfold, self.poly_degree, curr_lambda, reg_type)
                                                       for curr_lambda in lambdas)

        #==============================================================================================================#
        ''' LASSO Regression '''
        #==============================================================================================================#
        ''' Scikit Learn '''
        lasso_reg = Lasso(alpha=self.lambda_par).fit(X_poly, z_rav)
        ztilde_sk = lasso_reg.predict(X_poly).reshape(zshape)
        zarray_lasso = [self.z, ztilde_sk]
        print('\n')
        print("SL Lasso MSE (no CV) - " + str(mean_squared_error(zarray_lasso[0], zarray_lasso[1])))
        print("SL Lasso R^2 (no CV) - " + str(lasso_reg.score(X_poly, z_rav)))
        print('\n')
        ''' Plotting Surfaces '''
        filename = self.prefix + '_lasso_p' + str(self.poly_degree).zfill(2) + '_n'+ npoints + '.png'
        lib.plotSurface(self.x, self.y, zarray_lasso, self.output_dir, filename)
        # k-fold - studying the dependence on lambda
        print('kFold CV for LASSO Regression - Scikit Learn \n')
        reg_type = 'lasso'
        # parallel processing
        # making a dictionary of values which will save "mse mean value"
        self.kFoldMSEtestSK_lasso  = manager.dict()
        self.kFoldMSEtrainSK_lasso = manager.dict()
        self.kFoldBiasSK_lasso     = manager.dict()
        self.kFoldVarianceSK_lasso = manager.dict()
        # creating a method to work with dictionaries
        def doMultiproc3(kFoldMSEtestSK_lasso, kFoldMSEtrainSK_lasso, kFoldBiasSK_lasso, kFoldVarianceSK_lasso,
                         lib, X_poly, z_rav, kfold, poly_degree, curr_lambda, reg_type):
            print('Lasso Scikit Learn \n')
            print("Starting to calculate for $\lambda$=%s" %curr_lambda + '\n')
            kFoldMSEtestSK_lasso[curr_lambda]  = lib.doCrossValScikit(X_poly, z_rav, kfold, poly_degree, curr_lambda, reg_type)[0]
            kFoldMSEtrainSK_lasso[curr_lambda] = lib.doCrossValScikit(X_poly, z_rav, kfold, poly_degree, curr_lambda, reg_type)[1]
            kFoldBiasSK_lasso[curr_lambda]     = lib.doCrossValScikit(X_poly, z_rav, kfold, poly_degree, curr_lambda, reg_type)[2]
            kFoldVarianceSK_lasso[curr_lambda] = lib.doCrossValScikit(X_poly, z_rav, kfold, poly_degree, curr_lambda, reg_type)[3]
            # and exit!
            return
        Parallel(n_jobs = nproc, verbose = 2)(delayed(doMultiproc3)(self.kFoldMSEtestSK_lasso, self.kFoldMSEtrainSK_lasso,
                                                                    self.kFoldBiasSK_lasso, self.kFoldVarianceSK_lasso, lib, X_poly,
                                                    z_rav, self.kfold, self.poly_degree, curr_lambda, reg_type)
                                                    for curr_lambda in lambdas)


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
    sys.stdout.write('Do you want to use real data? (' + '\033[91m' + '[y]'+'\033[0m' +'/[n]) ')
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

    # Working with Real data
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
        '''
        Decide on how much of data you want to use:
        It doesn't make sense to use the entire data set, 
        because there are a lot of points => CV will give 
        identical results for test and train data. => makes
        sense to use only like 10% of the data.
        '''
        sys.stdout.write("Please, specify the percentage of the data to use (default = 0.1): ")
        cut_dat = input()
        terrain.sort()
        if cut_dat == '':
            dataset = terrain[int(len(terrain) * 0.9):]#terrain[int(len(terrain) * .05) : int(len(terrain) * .95)]
        else:
            dataset = terrain[int(len(terrain) * (1.-float(cut_dat))):]
        print(np.shape(dataset))
        #terrain.sort()
        #dataset = terrain[int(len(terrain) * .05) : int(len(terrain) * .95)]
        #print(np.shape(dataset))
        # number of independent variables (features)
        n_vars = 2
        # max polynomial degree
        sys.stdout.write("Please, choose the max polynomial degree (default = 5): ")
        max_poly_degree = input()
        if max_poly_degree == '':
            max_poly_degree = 5
        else:
            max_poly_degree = int(max_poly_degree)
        # number of processors to use <= better use this version
        sys.stdout.write("Please, choose the amount of processors to use (default = " +str(mp.cpu_count()-1) + "): ")
        nproc = input()
        if nproc == '':
            nproc = mp.cpu_count()-1
        else:
            nproc = int(nproc)
        # just to save your value (e.g. png(s)) under correct prefix
        prefix = 'real'
        # the amount of folds to get from your data
        kfold = 5
        # to calculate confidence intervals
        confidence = 1.96
        sigma = 1 # because I am generating my noise from standard normal distribution
        # lasso very sensitive to this lambda parameter
        sys.stdout.write("Please, choose the min value of hyperparameter (lambda) (default = 0.0001): ")
        lambda_par = input()
        if lambda_par == '':
            lambda_par = 0.0001
        else:
            lambda_par = float(lambda_par)
        # max lambda value (starting one)
        max_lambda = 0.1
        #lambda_par = 0.000001
        # just to save your value (e.g. png(s)) under correct prefix
        prefix = 'real'
        ''' Generating Data Set '''
        # generating an array of symbolic variables
        # based on the desired amount of variables
        x_symb = sp.symarray('x', n_vars, real = True)
        # making a copy of this array
        x_vals = x_symb.copy()
        x_vals[0] = np.linspace(0, len(dataset[0]), len(dataset[0]))
        x_vals[1] = np.linspace(0, len(dataset), len(dataset))
        # library object instantiation
        lib = rl.RegressionLibrary(x_symb, x_vals)
        # generating output data - first setting-up the proper grid
        x, y = np.meshgrid(x_vals[0], x_vals[1])
        z = dataset#terrain

    # Working with generated (Fake) data
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
        # number of processors to use <= better use this version
        sys.stdout.write("Please, choose the amount of processors to use (default = " +str(mp.cpu_count()-1) + "): ")
        nproc = input()
        if nproc == '':
            nproc = mp.cpu_count()-1
        else:
            nproc = int(nproc)
        # the amount of folds to get from your data
        kfold = 5
        # to calculate confidence intervals
        confidence = 1.96
        sigma = 0.1
        # lasso very sensitive to this lambda parameter
        sys.stdout.write("Please, choose the mean value of hyperparameter (lambda) (default = 0.0001): ")
        lambda_par = input()
        if lambda_par == '':
            lambda_par = 0.0001
        else:
            lambda_par = float(lambda_par)
        # max lambda value (starting one)
        max_lambda = 0.1
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
        lib = rl.RegressionLibrary(x_symb, x_vals)
        # setting up the grid
        x, y = np.meshgrid(x_vals[0], x_vals[1])
        # and getting output based on the Franke Function
        z = lib.FrankeFunction(x, y) + sigma * np.random.randn(N_points, N_points)

    # To plot MSE from kFold Cross Validation
    kFoldMSEtest_lin      = []#[None] * max_poly_degree
    kFoldMSEtrain_lin     = []#[None] * max_poly_degree
    kFoldMSEtestSK_lin    = []#[None] * max_poly_degree
    kFoldMSEtrainSK_lin   = []#[None] * max_poly_degree

    kFoldMSEtest_ridge    = []#[None] * max_poly_degree
    kFoldMSEtrain_ridge   = []#[None] * max_poly_degree
    kFoldMSEtestSK_ridge  = []#[None] * max_poly_degree
    kFoldMSEtrainSK_ridge = []#[None] * max_poly_degree

    kFoldMSEtestSK_lasso  = []#[None] * max_poly_degree
    kFoldMSEtrainSK_lasso = []#[None] * max_poly_degree

    # for bias-variance trade off
    kFoldBias_lin         = []
    kFoldVariance_lin     = []
    kFoldBiasSK_lin       = []
    kFoldVarianceSK_lin   = []

    kFoldBias_ridge       = []
    kFoldVariance_ridge   = []
    kFoldBiasSK_ridge     = []
    kFoldVarianceSK_ridge = []

    kFoldBiasSK_lasso     = []
    kFoldVarianceSK_lasso = []

    # to better classify output plots, I am adding
    # the amount of points, the png was generated for
    # (if it is a real data => then we get 'real' in the name)
    if prefix == 'real':
        npoints = 'real'
    elif prefix == 'fake':
        npoints = str(N_points)

    # looping through all polynomial degrees
    for poly_degree in range(1, max_poly_degree+1):
        print('\n')
        print('Starting analysis for polynomial of degree: ' + str(poly_degree))
        pipeline = MainPipeline(x_symb, x_vals, x, y, z, confidence, sigma, kfold,
        lambda_par, output_dir, prefix, poly_degree)
        pipeline.doRegression(nproc, npoints, max_lambda)

        # getting the list of dictionaries
        # linear regression kfold
        kFoldMSEtest_lin.append(pipeline.kFoldMSEtest_lin)
        kFoldMSEtrain_lin.append(pipeline.kFoldMSEtrain_lin)
        kFoldMSEtestSK_lin.append(pipeline.kFoldMSEtestSK_lin)
        kFoldMSEtrainSK_lin.append(pipeline.kFoldMSEtrainSK_lin)
        # ridge regression kfold
        kFoldMSEtest_ridge.append(dict(pipeline.kFoldMSEtest_ridge))
        kFoldMSEtrain_ridge.append(dict(pipeline.kFoldMSEtrain_ridge))
        kFoldMSEtestSK_ridge.append(dict(pipeline.kFoldMSEtestSK_ridge))
        kFoldMSEtrainSK_ridge.append(dict(pipeline.kFoldMSEtrainSK_ridge))
        # lasso regression kfold
        kFoldMSEtestSK_lasso.append(dict(pipeline.kFoldMSEtestSK_lasso))
        kFoldMSEtrainSK_lasso.append(dict(pipeline.kFoldMSEtrainSK_lasso))

        '''
        getting values for bias and variance. In case of Ridge and Lasso
        regression these values are represented by dictionaries (for each 
        values of lambda)
        '''
        kFoldBias_lin.append(pipeline.kFoldBias_lin)
        kFoldVariance_lin.append(pipeline.kFoldVariance_lin)
        kFoldBiasSK_lin.append(pipeline.kFoldBiasSK_lin)
        kFoldVarianceSK_lin.append(pipeline.kFoldVarianceSK_lin)

        kFoldBias_ridge.append(dict(pipeline.kFoldBias_ridge))
        kFoldVariance_ridge.append(dict(pipeline.kFoldVariance_ridge))
        kFoldBiasSK_ridge.append(dict(pipeline.kFoldBiasSK_ridge))
        kFoldVarianceSK_ridge.append(dict(pipeline.kFoldVarianceSK_ridge))

        kFoldBiasSK_lasso.append(dict(pipeline.kFoldBiasSK_lasso))
        kFoldVarianceSK_lasso.append(dict(pipeline.kFoldVarianceSK_lasso))

    '''
    Makins plots of MSEs
    '''
    # Colors - randomly generating colors
    np.random.seed(1)
    test_colors = [np.random.rand(3,) for i in range(max_poly_degree)] # <= will generate random colors
    train_colors = [np.random.rand(3,) for i in range(max_poly_degree)]
    # Turning interactive mode on
    #plt.ion()
    ''' MSE as a function of model complexity '''
    # plotting MSE from test data
    # Linear Regression
    filename = prefix + '_linear_mse_p' + str(max_poly_degree).zfill(2) + '_n'+ npoints + '.png'
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    t = []
    [t.append(i) for i in range(1, max_poly_degree + 1)]
    # manual
    ax1.plot(t, kFoldMSEtest_lin, color = test_colors[0], marker='o', label='test')
    ax1.plot(t, kFoldMSEtrain_lin, color = train_colors[0], linestyle='dashed', label='train')
    # scikit learn
    ax2.plot(t, kFoldMSEtestSK_lin, color = test_colors[0], marker='o', label='test')
    ax2.plot(t, kFoldMSEtrainSK_lin, color = train_colors[0], linestyle='dashed', label='train')
    if prefix == 'real':
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    # Shrink current axis by 20% - making legends appear to the right of the plot
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_title('MSE as a function of model complexity; Linear Regression')
    plt.xlabel('model complexity (polynomial degree)')
    ax1.set_ylabel('MSE')
    ax2.set_ylabel('MSE')
    fig.savefig(output_dir + '/' + filename)
    plt.close(fig)
    # write mses to a txt file
    filename = prefix + '_linear_mse_p' + str(max_poly_degree).zfill(2) + '_n'+ npoints + '.txt'
    with open(output_dir + '/' + filename, 'w') as file_handler:
        file_handler.write("MSE test (Me): {}\n".format(kFoldMSEtest_lin))
        print('\n')
        file_handler.write("MSE train (Me): {}\n".format(kFoldMSEtrain_lin))
        print('\n')
        file_handler.write("MSE test (SK): {}\n".format(kFoldMSEtestSK_lin))
        print('\n')
        file_handler.write("MSE train (SK): {}\n".format(kFoldMSEtrainSK_lin))

    # Ridge Regression
    filename = prefix + '_ridge_mse_p' + str(max_poly_degree).zfill(2) + '_n'+ npoints +'.png'
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    t = []
    [t.append(i) for i in range(1, max_poly_degree + 1)]
    keylist = []
    # creating a keylist to be able to convert
    # a list of dictionaries to a list of lists
    curr_lambda = max_lambda
    while curr_lambda >= lambda_par:
        # the list of all lambdas
        keylist.append(curr_lambda)
        curr_lambda = curr_lambda/10
    # Manual
    # converting a list of dictionaries to a list of lists
    # (index is the lambda value, i.e. we have a list: [[],[],[],...[]]
    # where first sublist corresponds to a maximum lambda value - 1 -
    # and the last sublist corresponds to the smallest lambda value - 0.001 (if default))
    test_list = [[row[key] for row in kFoldMSEtest_ridge] for key in keylist]
    train_list = [[row[key] for row in kFoldMSEtrain_ridge] for key in keylist]
    # plotting different mse values for different lambda
    for i in range(len(test_list)):
        ax1.plot(t, test_list[i], color = test_colors[i], marker='o', label='$\lambda$='+str(keylist[i]) + ', test')
        ax1.plot(t, train_list[i], color = train_colors[i], linestyle='dashed', label='$\lambda$='+str(keylist[i]) + ', train')
    # Scikit learn
    test_list = [[row[key] for row in kFoldMSEtestSK_ridge] for key in keylist]
    train_list = [[row[key] for row in kFoldMSEtrainSK_ridge] for key in keylist]
    # plotting different mse values for different lambda
    for i in range(len(test_list)):
        ax2.plot(t, test_list[i], color = test_colors[i], marker='o', label='$\lambda$='+str(keylist[i]) + ', test')
        ax2.plot(t, train_list[i], color = train_colors[i], linestyle='dashed', label='$\lambda$='+str(keylist[i]) + ', train')

    if prefix == 'real':
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    # Shrink current axis by 20%
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_title('MSE as a function of model complexity; Ridge Regression')
    plt.xlabel('model complexity (polynomial degree)')
    ax1.set_ylabel('MSE')
    ax2.set_ylabel('MSE')
    fig.savefig(output_dir + '/' + filename)
    plt.close(fig)
    # write mses to a txt file
    filename = prefix + '_ridge_mse_p' + str(max_poly_degree).zfill(2) + '_n'+ npoints +'.txt'
    with open(output_dir + '/' + filename, 'w') as file_handler:
        file_handler.write("MSE test (Me): {}\n".format(kFoldMSEtest_ridge))
        print('\n')
        file_handler.write("MSE train (Me): {}\n".format(kFoldMSEtrain_ridge))
        print('\n')
        file_handler.write("MSE test (SK): {}\n".format(kFoldMSEtestSK_ridge))
        print('\n')
        file_handler.write("MSE train (SK): {}\n".format(kFoldMSEtrainSK_ridge))

    # LASSO regression
    filename = prefix + '_lasso_mse_p' + str(max_poly_degree).zfill(2) + '_n'+ npoints +'.png'
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    t = []
    [t.append(i) for i in range(1, max_poly_degree + 1)]
    # scikit learn
    keylist = []
    # creating a keylist to be able to convert
    # a list of dictionaries to a list of lists
    curr_lambda = max_lambda
    while curr_lambda >= lambda_par:
        keylist.append(curr_lambda)
        curr_lambda = curr_lambda/10
    # converting a list of dictionaries to a list of lists
    # (index is the lambda value, i.e. we have a list: [[],[],[],...[]]
    # where first sublist corresponds to a maximum lambda value - 1 -
    # and the last sublist corresponds to the smallest lambda value - 0.001 (if default))
    test_list = [[row[key] for row in kFoldMSEtestSK_lasso] for key in keylist]
    train_list = [[row[key] for row in kFoldMSEtrainSK_lasso] for key in keylist]
    # plotting different mse values for different lambda
    for i in range(len(test_list)):
        ax1.plot(t, test_list[i], color = test_colors[i], marker='o', label='$\lambda$='+str(keylist[i]) + ', test')
        ax1.plot(t, train_list[i], color = train_colors[i], linestyle='dashed', label='$\lambda$='+str(keylist[i]) + ', train')
    if prefix == 'real':
        ax1.set_yscale('log')
    # Shrink current axis by 20%
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True)
    ax1.set_title('MSE as a function of model complexity; Lasso Regression')
    plt.xlabel('model complexity (polynomial degree)')
    ax1.set_ylabel('MSE')
    fig.savefig(output_dir + '/' + filename)
    plt.close(fig)
    # write mses to a txt file
    filename = prefix + '_lasso_mse_p' + str(max_poly_degree).zfill(2) + '_n'+ npoints +'.txt'
    with open(output_dir + '/' + filename, 'w') as file_handler:
        file_handler.write("MSE test (Me): {}\n".format(kFoldMSEtestSK_lasso))
        print('\n')
        file_handler.write("MSE train (Me): {}\n".format(kFoldMSEtrainSK_lasso))

    '''
    Bias/Variance as a function of model complexity
    '''
    # Bias-variance trade off
    # Colors - randomly generating colors
    test_colors = [np.random.rand(3,) for i in range(max_poly_degree)] # <= will generate random colors
    train_colors = [np.random.rand(3,) for i in range(max_poly_degree)]
    # Turning interactive mode on
    #plt.ion()
    ''' Linear Regression '''
    filename = prefix + '_linear_bv_p' + str(max_poly_degree).zfill(2) + '_n'+ npoints + '.png'
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    t = []
    [t.append(i) for i in range(1, max_poly_degree + 1)]
    # manual
    ax1.plot(t, kFoldBias_lin, color = test_colors[0], marker='o', label='bias')
    ax1.plot(t, kFoldVariance_lin, color = train_colors[0], linestyle='dashed', label='variance')
    # scikit learn
    ax2.plot(t, kFoldBiasSK_lin, color = test_colors[0], marker='o', label='bias')
    ax2.plot(t, kFoldVarianceSK_lin, color = train_colors[0], linestyle='dashed', label='variance')
    if prefix == 'real':
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    # Shrink current axis by 20% - making legends appear to the right of the plot
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_title('Bias/Variance as a function of model complexity; Linear Regression')
    plt.xlabel('model complexity (polynomial degree)')
    ax1.set_ylabel('Bias/Variance')
    ax2.set_ylabel('Bias/Variance')
    fig.savefig(output_dir + '/' + filename)
    plt.close(fig)
    # write bias/variance to a txt file
    filename = prefix + '_linear_bv_p' + str(max_poly_degree).zfill(2) + '_n' + npoints + '.txt'
    with open(output_dir + '/' + filename, 'w') as file_handler:
        file_handler.write("Bias (Me): {}\n".format(kFoldBias_lin))
        print('\n')
        file_handler.write("Variance (Me): {}\n".format(kFoldVariance_lin))
        print('\n')
        file_handler.write("Bias (Me): {}\n".format(kFoldBiasSK_lin))
        print('\n')
        file_handler.write("Variance (Me): {}\n".format(kFoldVarianceSK_lin))

    ''' Ridge Regression '''
    filename = prefix + '_ridge_bv_p' + str(max_poly_degree).zfill(2) + '_n'+ npoints +'.png'
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    t = []
    [t.append(i) for i in range(1, max_poly_degree + 1)]
    keylist = []
    # creating a keylist to be able to convert
    # a list of dictionaries to a list of lists
    curr_lambda = max_lambda
    while curr_lambda >= lambda_par:
        # the list of all lambdas
        keylist.append(curr_lambda)
        curr_lambda = curr_lambda/10
    # Manual
    # converting a list of dictionaries to a list of lists
    # (index is the lambda value, i.e. we have a list: [[],[],[],...[]]
    # where first sublist corresponds to a maximum lambda value - 1 -
    # and the last sublist corresponds to the smallest lambda value - 0.001 (if default))
    test_list = [[row[key] for row in kFoldBias_ridge] for key in keylist]
    train_list = [[row[key] for row in kFoldVariance_ridge] for key in keylist]
    # plotting different mse values for different lambda
    for i in range(len(test_list)):
        ax1.plot(t, test_list[i], color = test_colors[i], marker='o', label='$\lambda$='+str(keylist[i]) + ', bias')
        ax1.plot(t, train_list[i], color = train_colors[i], linestyle='dashed', label='$\lambda$='+str(keylist[i]) + ', variance')
    # Scikit learn
    test_list = [[row[key] for row in kFoldBiasSK_ridge] for key in keylist]
    train_list = [[row[key] for row in kFoldVarianceSK_ridge] for key in keylist]
    # plotting different mse values for different lambda
    for i in range(len(test_list)):
        ax2.plot(t, test_list[i], color = test_colors[i], marker='o', label='$\lambda$='+str(keylist[i]) + ', bias')
        ax2.plot(t, train_list[i], color = train_colors[i], linestyle='dashed', label='$\lambda$='+str(keylist[i]) + ', variance')

    if prefix == 'real':
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    # Shrink current axis by 20%
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_title('Bias/Variance as a function of model complexity; Ridge Regression')
    plt.xlabel('model complexity (polynomial degree)')
    ax1.set_ylabel('Bias/Variance')
    ax2.set_ylabel('Bias/Variance')
    fig.savefig(output_dir + '/' + filename)
    plt.close(fig)
    # write bias/variance to a txt file
    filename = prefix + '_ridge_bv_p' + str(max_poly_degree).zfill(2) + '_n'+ npoints +'.txt'
    with open(output_dir + '/' + filename, 'w') as file_handler:
        file_handler.write("Bias (Me): {}\n".format(kFoldBias_ridge))
        print('\n')
        file_handler.write("Variance (Me): {}\n".format(kFoldVariance_ridge))
        print('\n')
        file_handler.write("Bias (Me): {}\n".format(kFoldBiasSK_ridge))
        print('\n')
        file_handler.write("Variance (Me): {}\n".format(kFoldVarianceSK_ridge))

    '''Lasso'''
    filename = prefix + '_lasso_bv_p' + str(max_poly_degree).zfill(2) + '_n'+ npoints +'.png'
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    t = []
    [t.append(i) for i in range(1, max_poly_degree + 1)]
    # scikit learn
    keylist = []
    # creating a keylist to be able to convert
    # a list of dictionaries to a list of lists
    curr_lambda = max_lambda
    while curr_lambda >= lambda_par:
        keylist.append(curr_lambda)
        curr_lambda = curr_lambda/10
    # converting a list of dictionaries to a list of lists
    # (index is the lambda value, i.e. we have a list: [[],[],[],...[]]
    # where first sublist corresponds to a maximum lambda value - 1 -
    # and the last sublist corresponds to the smallest lambda value - 0.001 (if default))
    test_list = [[row[key] for row in kFoldBiasSK_lasso] for key in keylist]
    train_list = [[row[key] for row in kFoldVarianceSK_lasso] for key in keylist]
    # plotting different mse values for different lambda
    for i in range(len(test_list)):
        ax1.plot(t, test_list[i], color = test_colors[i], marker='o', label='$\lambda$='+str(keylist[i]) + ', bias')
        ax1.plot(t, train_list[i], color = train_colors[i], linestyle='dashed', label='$\lambda$='+str(keylist[i]) + ', variance')

    if prefix == 'real':
        ax1.set_yscale('log')
    # Shrink current axis by 20%
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True)
    ax1.set_title('Bias/Variance as a function of model complexity; Lasso Regression')
    plt.xlabel('model complexity (polynomial degree)')
    ax1.set_ylabel('Bias/Variance')
    fig.savefig(output_dir + '/' + filename)
    plt.close(fig)
    # write bias/variance to a txt file
    filename = prefix + '_lasso_bv_p' + str(max_poly_degree).zfill(2) + '_n'+ npoints +'.txt'
    with open(output_dir + '/' + filename, 'w') as file_handler:
        file_handler.write("Bias (SK): {}\n".format(kFoldBiasSK_lasso))
        print('\n')
        file_handler.write("Variance (SK): {}\n".format(kFoldVarianceSK_lasso))


    # End time of the program
    end_time = time.time()
    print("-- Program finished at %s sec --" % (end_time - start_time))












