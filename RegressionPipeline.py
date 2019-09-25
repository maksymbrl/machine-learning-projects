#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:44:18 2019

@author: maksymb
"""

# Standard imports
import os
import numpy as np
import random
#import numpy.polynomial.polynomial as poly
# 
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

import time

class LinearPipeline:
    # class constructor
    def __init__(self, N_train):
        # variables instantiation
        # Generating datasets
        self.N_train = N_train
        #self.x = np.arange(0, 1, 1/self.N_train)#np.random.rand(self.N_train)
        #self.y = np.arange(0, 1, 1/self.N_train)#np.exp(-self.x) - self.x**2 + 4 + 0.25 * np.random.randn(self.N_train)
        self.x = np.sort(np.random.uniform(0, 1, N_train))
        self.y = np.sort(np.random.uniform(0, 1, N_train))

    '''
    Main method of the class - calculates everything, returns nothing
    '''
    def main(self, *args):
        # the amount of variables (e.g. x,y => 2)
        variables = args[0]
        # the degre of polynomial we want to construct to match our data
        degree    = args[1]
        # kfold
        kfold     = args[2]
        # generating grid with the incoming data
        x, y = np.meshgrid(self.x, self.y)        
        # return z values and make it one dimensional (intead of 2 dimensional)
        z = self.FrankeFunction(x, y)# + 0.1 * np.random.randn(self.N_train)          
        z_new = np.ravel(z)

        '''
        (a) Ordinary Least Square on the Franke function with resampling:
            Calling methond to do Linear Regression (with the generated polynomial)
        '''
        ''' MANUAL '''
        # Generating polynomial to fit the data (and getting design matrix)
        X = self.generatePolynomial(variables, degree)
        # retrieving our model values
        ztilde, beta = self.doPolyRegression(X, z_new)
        ztilde = np.reshape(ztilde, (-1, self.N_train))
        ''' SKLEARN '''
        # values from scikit learn
        # generate polynomial
        poly_features = PolynomialFeatures(degree = degree, include_bias = False)
        X_scikit = np.reshape((self.x, self.y), (self.N_train, variables))
        #print(X_scikit)
        X_poly = poly_features.fit_transform(X_scikit)
        #print(len(X_poly))
        lin_reg = LinearRegression().fit(X_poly, z)
        beta_sk = lin_reg.coef_[0]
        ztilde_sk = lin_reg.predict(X_poly)
        # Printing out the values we are interested in
        #print(ztilde[0], ztilde_sk[0])
        print("betas are %s" %len(beta))
        print("sklearn betas are %s" %len(beta_sk))
        print("MSE is ", self.getMSE(z, ztilde), " and sklearn ", mean_squared_error(z, ztilde_sk))
        print("R^2 is ", self.getR2(z, ztilde), " and sklearn ", lin_reg.score(X_poly, z))
        # Plotting the resulting surface
        fig = plt.figure(figsize = (10, 10))
        # the last value is the subplot number => subplot(nrows, ncols, index, **kwargs)
        ax1 = fig.add_subplot(2,2,1, projection='3d')
        ax2 = fig.add_subplot(2,2,2, projection='3d')
        #ax1.set_zlim3d(-0.2, 1.2)
        surf1 = ax1.plot_surface(x, y, z, alpha=0.5, cmap = 'brg_r', linewidth = 0, antialiased = False)
        ax1.scatter(x, y, ztilde, alpha=1, s=1, color='black')
        #fig.colorbar(surf1, shrink=0.5, aspect=5)
        surf2 = ax2.plot_surface(x, y, z, alpha=0.5, cmap = 'brg_r', linewidth = 0, antialiased = False)
        ax2.scatter(x, y, ztilde_sk, alpha=1, s=1, color='black')
        #fig.colorbar(surf2, shrink=0.5, aspect=5)
        plt.show()
        '''
        (b) Resampling techniques:
            k-fold cross validation
        '''
        ''' MANUAL '''
        self.doKFoldCrossVal(X, z_new, kfold)
        ''' SKLEARN '''
        
    '''
    Polynomial Regression - does linear regression analysis with our generated polynomial
    (should I add Singular Value Decomposition, as done in lecture notes?) and
    returns the predicted values (our model)
    '''
    def doPolyRegression(self, *args):
        # getting z values
        z = args[1]
        # Getting the polynomial data (desing matrix)
        X = args[0]#self.generatePolynomial(args[0], args[1])
        
        # Applying Singular Value Decomposition
        A = np.transpose(X) @ X
        U, s, VT = np.linalg.svd(A)
        D = np.zeros((len(U), len(VT)))
        for i in range(0,len(VT)):
            D[i,i] = s[i]
        UT = np.transpose(U); 
        V = np.transpose(VT); 
        invD = np.linalg.inv(D)
        invA = np.matmul(V, np.matmul(invD, UT))

        # Calculate betas
        beta = invA.dot(X.T).dot(z)
        #beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
        #print("betas is ", beta)
        # and then make the prediction
        ztilde = X @ beta
        #z_new  = np.reshape(ztilde, (-1, 2))
        #print(z_new)
        #surf2 = ax.plot_surface(x, y, z_new, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
        
        # Plot the surface.
#        fig = plt.figure()
#        ax  = fig.gca(projection='3d')
#        surf1 = ax.plot_surface(x, y, z, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
        # Add a color bar which maps values to colors.
#        fig.colorbar(surf1, shrink=0.5, aspect=5)
#        plt.show()
        #print(beta)
        #print("Confidence is ", self.getConfidence(X, invA, beta))
        return ztilde, beta
        
    '''
    Generating polynomials for given number of variables for a given degree
    using Newton's Binomial formula, and when returning the design matrix,
    computed from the list of all variables
    '''
    def generatePolynomial(self, *args):
        # generating polynomial of 3 independent variables
        #x, y, z = symbols('x,y,z')
        # another way is to generate variables x_1 x_2 and so on using symarray (?)
        # generating real variables
        #D = (3, 4, 2, 3)
        #a = symarray("a", D)
        # Creating a array/list of (sympy) real varibales
        x = sp.symarray('x', args[0], real=True)
        #print(var_x, self.y)
        #init_printing(use_unicode=False, wrap_line=False)
        prod_it = it.product(x, repeat = args[1])
        # Creating homogeneous polynomial part (with Newton's Binomial)
        pol = [sp.Mul(*p) for p in prod_it]#Add(*[Mul(*p) for p in prod_it])
        # Adding inhomogeneous part
        [pol.append(x[i]**j) for j in range(0, args[1]) for i in range(0, args[0])]
        # remove duplicates from the list
        mylist = list( dict.fromkeys(pol) )
        #print(mylist)
        # we have 2 variables so, we need N^2 amount of points:
        # (for m variables we would have N^m points) 
        points = self.N_train**args[0]
        X = np.ones((points, len(mylist)), dtype = float)
        # if it is 2 variables - we have a 3d matrix
        for k in range(len(mylist)):
            #print(expr.subs(x[1], 1))
            X[:, k] = [mylist[k].subs(x[0], i).subs(x[1], j) for i in self.x for j in self.y]
        # returning an array of values (design matrix)
        return X
        
    '''
    The Franke funciton - to get data in z axis
    '''
    def FrankeFunction(self, x, y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4
    
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
    '''
    Beta confidence interval
    '''
#    def getConfidence(self, X, matrix, beta, confidence = 0.95):
#        weight = np.sqrt( np.diag( matrix ) ) * confidence
#        betamin = beta - weight
#        betamax = beta + weight
#        return betamin, betamax
    
    # (b) splitting training and test set
    def doKFoldCrossVal(self, *args):
        # getting z values
        z = args[1]
        # Getting the polynomial data (desing matrix)
        X = args[0]#self.generatePolynomial(args[0], args[1])
        # getting the k value
        kfold = args[2]
        # (b) splitting the data
        X_train, X_test, Z_train, Z_test = train_test_split(X, z, test_size = 0.2)
        #print(np.random.randint(10, size=1), random.randrange(10))
        # We calculate the size of each fold as the size of the dataset divided by the number of folds required.

# The entry point of the program        
if __name__ == '__main__':
    # Start time of the program
    start_time = time.time()
    # data points
    N_train = 50
    # number of independent variables
    variables = 2
    # polynomial degree
    degree = 5
    # kfold
    kfold  = 10
    # object instantiation
    pipe = LinearPipeline(N_train)
    # calculating stuff
    pipe.main(variables, degree, kfold)#doPolyRegression(variables, degree)
    end_time = time.time()
    print("-- Script finished at %s sec --" %(end_time - start_time))
    
    # test for polynomials of multiple variables
    #pipe.generatePolynomial(variables, degree)
        
        