#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:44:18 2019

@author: maksymb
"""

# Standard imports
import os
import numpy as np
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
from sklearn.model_selection import train_test_split

import time

class LinearPipeline:
    # class constructor
    def __init__(self, N_train):
        # variables instantiation 100
        
        # Generating datasets
        self.N_train = N_train
        self.x = np.arange(0, 1, 1/self.N_train)#np.random.rand(self.N_train)
        self.y = np.arange(0, 1, 1/self.N_train)#np.exp(-self.x) - self.x**2 + 4 + 0.25 * np.random.randn(self.N_train)
        #self.x, self.y = np.meshgrid(self.x, self.y)
        #print(self.x, self.y)
        
    '''
    Polynomial Regression - Main method of the class
    '''
    def doPolyRegression(self, *args):

        # Getting the polynomial data (desing matrix)
        X = self.generatePolynomial(args[0], args[1])
        x, y = np.meshgrid(self.x, self.y)        
        # return z values and make it one dimensional (intead of 2 dimensional)
        z = self.FrankeFunction(x, y) + 0.1 * np.random.randn(self.N_train)            
        z_new = np.ravel(z)
        # calculate betas
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z_new)
        # and then make the prediction
        ztilde = X @ beta
        # printing values - H/W (a)
        print("MSE is ", self.MSE(z_new, ztilde))
        print("R^2 is ", self.R2(z_new, ztilde))
        #z_new  = np.reshape(ztilde, (-1, 2))
        #print(z_new)
        #surf2 = ax.plot_surface(x, y, z_new, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
        
        # Plot the surface.
        fig = plt.figure()
        ax  = fig.gca(projection='3d')
        surf1 = ax.plot_surface(x, y, z, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf1, shrink=0.5, aspect=5)
        plt.show()
        
        # excersise (b)
        self.doKFold(X, z_new)
        
    '''
    Generating polynomials for given number of variables for a given degree
    using Newton Binomial formula, and when returning the list of all variables
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
    MSE
    '''
    def MSE(self, z_data, z_model):
        n = np.size(z_model)
        return np.sum((z_data-z_model)**2)/n
    '''
    R^2
    '''
    def R2(self, z_data, z_model):
        return 1 - np.sum((z_data - z_model) ** 2) / np.sum((z_data - np.mean(z_data)) ** 2)
    
    # (b) splitting training and test set
    def doKFold(self, *args):
        X = args[0]
        z = args[1]
        # (b) splitting the data
        X_train, X_test, Z_train, Z_test = train_test_split(X, z, test_size = 0.2)

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
    # object instantiation
    pipe = LinearPipeline(N_train)
    # calculating stuff
    pipe.doPolyRegression(variables, degree)
    end_time = time.time()
    print("-- Script finished at %s sec --" %(end_time - start_time))
    
    # test for polynomials of multiple variables
    #pipe.generatePolynomial(variables, degree)
        
        