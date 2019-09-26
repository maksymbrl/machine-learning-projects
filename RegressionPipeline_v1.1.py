#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:13:27 2019

@author: maksymb
"""

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# for polynimial manipulation
import sympy as sp
#from sympy import *
import itertools as it

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
        # getting design matrix
        #X1, X2 = self.constructDesignMatrix(x, y, z)
        X1 = self.constructDesignMatrix()
        ztilde1, beta1, conf1 = self.doLinearRegression(X1, z)
        ztilde1 = ztilde1.reshape(-1, self.N)
        #ztilde2, beta2, conf2 = self.doLinearRegression(X2, z)
        #ztilde2 = ztilde2.reshape(-1, self.N)
        print("betas are %s" %beta1 + " ± %s" %conf1)
        #print("betas are %s" %beta2 + " ± %s" %conf2)
        print("MSE is ", self.getMSE(z, ztilde1))#, self.getMSE(z, ztilde2))#, " and sklearn ", mean_squared_error(z, ztilde_sk))
        print("R^2 is ", self.getR2(z, ztilde1))#,self.getR2(z, ztilde2))#, " and sklearn ", lin_reg.score(X_poly, z))
        # drawing surface
        fig = plt.figure(figsize = (10, 10))
        ax1 = fig.add_subplot(3,3,1, projection='3d')
        ax2 = fig.add_subplot(3,3,2, projection='3d')
        ax3 = fig.add_subplot(3,3,5, projection='3d')
        surf1 = ax1.plot_surface(x, y, z, alpha=0.5, cmap = 'brg_r', linewidth = 0, antialiased = False)
        #fig.colorbar(surf1, shrink=0.5, aspect=5)
        surf2 = ax2.plot_surface(x, y, ztilde1, alpha=0.5, cmap = 'brg_r', linewidth = 0, antialiased = False)
        #surf3 = ax3.plot_surface(x, y, ztilde2, alpha=0.5, cmap = 'brg_r', linewidth = 0, antialiased = False)
    '''
    Generating polynomials for given number of variables for a given degree
    using Newton's Binomial formula, and when returning the design matrix,
    computed from the list of all variables
    '''
    def constructDesignMatrix(self, *args):
        # getting inputs
        x_vals = self.x_vals
        # using itertools for generating all possible combinations 
        # of multiplications between our variables and 1, i.e.:
        # x_0*x_1*1, x_0*x_0*x_1*1 etc. => will get polynomial 
        # coefficients
        variables = list(self.x_symb.copy())
        variables.append(1)
        terms = [sp.Mul(*i) for i in it.combinations_with_replacement(variables, self.poly_degree)]
        # creating desing matrix
        points = len(x_vals[0])*len(x_vals[1])
        X1 = np.ones((points, len(terms)))
        for k in range(len(terms)):
            #print(terms[k])
            f = sp.lambdify([self.x_symb[0],self.x_symb[1]], terms[k], "numpy")
            #print(f(self.x_vals[0],self.x_vals[1]))
            #X1[:, k] = [terms[k].subs(self.x_symb[0], i).subs(self.x_symb[1], j) for i in self.x_vals[0] for j in self.x_vals[1]]
            X1[:, k] = [f(i, j) for i in self.x_vals[1] for j in self.x_vals[0]]
            #X1[:, k] = f(self.x_vals[0], self.x_vals[1])
        # returning an array of values (design matrix)
        #print(X1, "\n")   
        
        '''
        Another approach (adopted from Piazza) <= gives the same result, 
        so I comment it out (you can check it if you want to)
        '''
        '''
        n = self.poly_degree
        # getting inputs
        x, y, z = args[0], args[1], args[2]
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)
        N = len(x)
        l = int((n+1)*(n+2)/2)		# Number of elements in beta
        X2 = np.ones((N,l))

        for i in range(1, n+1):
            q = int((i) * (i+1) / 2)
            for k in range(i+1):
                X2[:, q+k] = x**(i-k) * y**k
        #print(X2, "\n")
        '''
        # returning constructed design matrix (for 2 approaches if needed)
        return X1#, X2
    '''
    #============================#
    # Regression Methods
    #============================#
    '''
    '''
    Polynomial Regression - does linear regression analysis with our generated 
    polynomial and returns the predicted values (our model)
    '''
    def doLinearRegression(self, *args):
        # getting design matrix
        X = args[0]
        # getting z values
        z = np.ravel(args[1])
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
        # and then make the prediction
        ztilde = X @ beta
        
        # calculating beta confidence
        confidence = 1.96
        sigma = 1
        SE = sigma * np.sqrt(np.diag(invA)) * confidence
        
        return ztilde, beta, SE
    '''
    Ridge Regression
    '''
    def doRidgeRegression(self, *args):
        None
    
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
    n_vars = 2
    poly_degree = 5
    pipeline = MainPipeline(N_points, n_vars, poly_degree)
    pipeline.main(n_vars, poly_degree)
    # End time of the program
    end_time = time.time()
    print("-- Program finished at %s sec --" %(end_time - start_time))