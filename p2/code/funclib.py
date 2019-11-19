#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:31:57 2019

@author: maksymb
"""

'''
Library Module
'''


# library imports
import numpy as np
import math as mt
# for polynimial manipulation
import sympy as sp
import itertools as it

'''
Class which contains all activation functions
(functions are taken from lecture slides)
'''
class ActivationFuncs:
    # class constructor
    def __init__(self, *args):
        pass
    
    # Sigmoid Function
    def CallSigmoid(self, *args):
        z = args[0]
        return 1 / (1 + np.exp(-z))
    
    # Derivative of sigmoid
    def CallDSigmoid(self, *args):
        z = args[0]
        p = self.CallSigmoid(z)
        return p * (1 - p)
    
    # tanh Function
    def CallTanh(self, *args):
        z = args[0]
        return np.tanh(z)
    
    # tanh'
    def CallDTanh(self, *args):
        z = args[0]
        return 1 - self.CallTanh(z)**2
    
    
    # Rectified Linear Unit Function <= need to check this one
    def CallReLU(self, *args):
        z = args[0]
        return np.maximum(z, 0)
    
    # ReLU's derivative
    def CallDReLU(self, *args):
        z = args[0]
        return (z > 0)
    
    # Softmax function
    def CallSoftmax(self, *args):
        # We need to normalize this function, otherwise we will 
        # get nan in the output
        z = args[0]
        
        # We can choose an arbitrary value for log(C) term, 
        # but generally log(C)=−max(a) is chosen, as it shifts
        # all of elements in the vector to negative to zero, 
        # and negatives with large 
        #p = np.exp(z - np.max(z))#, axis=1, keepdims = True))
        #return p / np.sum(p, axis=0)#np.sum(np.exp(z), axis=1, keepdims=True)
        p = np.exp(z - np.max(z))#np.exp(z)
        return p / np.sum(p, axis=1, keepdims=True)
    
    # Softmax gradient (in Vectorized form, 
    # also possible to write in element wise)
    def CallDSoftmax(self, *args):
        z = args[0]
        #print('Softmax, z shape is', z.shape)
        m = z.shape[0]
        p = self.CallSoftmax(z)
        #p = p.reshape(-1,1)
        #jacobian_m = np.diag(p)
        #for i in range(len(jacobian_m)):
        #    for j in range(len(jacobian_m)):
        #        if i == j:
        #            jacobian_m[i][j] = p[i] * (1-p[i])
        #        else: 
        #            jacobian_m[i][j] = -p[i]*p[j]
        #p = p.reshape(-1,1)
        #trying something else
        return p * (1 - p) #np.diagflat(p)#jacobian_m#p * (1 - p)#np.diagflat(p) - np.matmul(p, p.T)
        
    
        # identity
    def CallIdentity(self, *args):
        z = args[0]
        return z

    def CallDIdentity(self, *args):
        return 1
        
'''
Class which contains all Gradient Methods:
Gradient Descent Method, Stochastic gradient Descent,
Batch Gradient etc.
'''
class OptimizationFuncs:
    # class constructor
    def __init__(self, *args):
        self.activeFunc = ActivationFuncs()
    
    # Gradient Descent Method
    # rewrite it so it will be possible to use it for both regressions and NN
    # (just need to pass somehow J and dJ as they are the only difference)
    def SimpleGD(self, *args): 
        # getting inputs
        X = args[0]
        y = args[1]
        theta = args[3]
        alpha = args[4]
        # total number of iterations
        epochs = args[5]
        
        # number of features
        m = len(y)
        # saving cost history (to make a plots out of it)
        costs = []
        # applying gradient descent algorithm
        for epoch in epochs:
            # our model
            y_pred = np.dot(X, theta)
            # applying sigmoid
            h = self.activeFunc.CallSigmoid(y_pred)
            # calculating cost
            #J = -np.sum(y*np.log(h) +(1-y)*np.log(1-h)) / m
            # calculating gradient of the cost function
            #dJ = np.dot(X.T, h - y) / m
            J, dJ = CostFuncs().CallLogistic(X, y, h) 
            # updating weights
            theta = theta - alpha * dJ
            # saving current cost function for future reference
            costs.append(J)
            
        return theta#, costs
    
    # Stochastic Gradient Descent Method
    def StochasticGD(self, *args):
        return
    
    # Stochastic Gradient Descent Method with Batches
    def BatchedSGD(self, *args):
        return
'''
Class which contsins all Cost functions
and their respective gradients (used in
optimization methods)
'''
class CostFuncs:
    # contsructor
    def __init__(self, *args):
        pass
    
    # Linear Regression
    def CallLinear(self, *args):
        X = args[0]
        y = args[1]
        h = args[2]
        m = np.size(y)
        # cost function
        J = 0
        # its gradient
        dJ = 0
        return J, dJ
    
    # logistic Regression
    def CallLogistic(self, *args):
        X = args[0]
        y = args[1]
        h = args[2]
        m = np.size(y)
        # cost function
        J = -np.sum(y * np.log(h) +(1-y) * np.log(1-h)) / m
        J = np.squeeze(J)
        # its gradient
        dJ = np.dot(X.T, h - y) / m
        return J, dJ
    
    # Feed Forward Neural Network
    def CallNNLogistic(self, *args):
        Y = args[0]
        AL = args[1]
        modelParams = args[2]
        nLayers = args[3]
        m = args[4]
        lambd = args[5]
        #print(m)
        #print(AL)
        #AL[AL == 1] = 0.999 # if AL=0 we get an error, alternatively, I could set J=0 in this case
        #AL[AL==0] = AL+1e-07
        #AL = np.ravel(AL)
        #print("Y is",np.shape(Y))
        #print('AL is',np.shape(AL))
        J = -np.sum(np.multiply(Y, np.log(AL+1e-10)) +  np.multiply(1-Y, np.log(1-AL+1e-10)))/m
        # sum of all weights (for all layers, except input one)
        Wtot = 0
        # Computing Regularisation Term for n layer NN
        for l in range(1, nLayers, 1):
            Wtot += np.sum(np.square(modelParams['W' + str(l)]))
        Wtot = Wtot * lambd / (2*m)
        J = J + Wtot
        #L2_regularization_cost = (np.sum(np.square(W1)) + np.sum(np.square(W2)))*(lambd/(2*m))
        #print(np.multiply(Y, np.log(AL+1e-07)))
        J = np.squeeze(J)

        return J

'''
Class which contains all testing errors (MSNE, R^2, Accuracy etc.)
'''
class ErrorFuncs:
    def __init__(self, *args):
        pass
    
    # MSNE
    def CallMSNE(self, *args):
        z_data = args[0]
        z_model = args[1]
        n = np.size(z_model)
        return np.sum((z_data - z_model)**2) / n
    
    # R^2 test
    def CallR2(self, *args):
        z_data = args[0]
        z_model = args[1]
        return 1 - np.sum((z_data - z_model)**2) / np.sum((z_data - np.mean(z_data))**2)
    
    # I will be using scikit functionalities
    # To estimate errors etc.
    # Accuracy
    #def CallAccuracy(self, *args):
    #    return
    
'''
Class which holds all Normal equations, 
i.e. simple OLS, Ridge and LASSO used in
project 1
'''
class NormalFuncs:
    # constructor
    def __init__(self, *args):
        pass
    
    '''
    Generating polynomials for given number of variables for a given degree
    using Newton's Binomial formula, and when returning the design matrix,
    computed from the list of all variables
    '''
    def ConstructDesignMatrix(self, *args):
        # the degree of polynomial to be generated
        poly_degree = args[0]
        # getting inputs
        #x_vals = self.x_vals
        x_symb = args[1]
        x_vals = args[2]
        # using itertools for generating all possible combinations
        # of multiplications between our variables and 1, i.e.:
        # x_0*x_1*1, x_0*x_0*x_1*1 etc. => will get polynomial
        # coefficients
        variables = list(x_symb.copy())
        variables.append(1)
        terms = [sp.Mul(*i) for i in it.combinations_with_replacement(variables, poly_degree)]
        # creating desing matrix
        points = len(x_vals[0]) * len(x_vals[1])
        # creating desing matrix composed of ones
        X1 = np.ones((points, len(terms)))
        # populating design matrix with values
        for k in range(len(terms)):
            f = sp.lambdify([x_symb[0], x_symb[1]], terms[k], "numpy")
            X1[:, k] = [f(i, j) for i in x_vals[1] for j in x_vals[0]]
        # returning constructed design matrix (for 2 approaches if needed)
        return X1
    '''
    Normal Equation with lambda, i.e. it is a Ridge Regression
    (set lambda = 0 to get OLS)
    '''
    def CallNormal(self, *args):
        # getting design matrix
        X = args[0]
        # getting z values
        z = np.ravel(args[1])
        # hyper parameter
        lambda_par = args[2]
        # constructing the identity matrix
        XTX = X.T.dot(X)
        I = np.identity(len(XTX), dtype=float)
        # calculating parameters
        # if we set lambda =0, we get usual OLS,
        # but we need to account for singularity, 
        # so are using SVD
        if (lambda_par == 0):
            invA = self.CallSVD(X)
        else:
            invA = np.linalg.inv(XTX + lambda_par * I)
        beta = invA.dot(X.T).dot(z)
        # and making predictions
        ztilde = X @ beta

        # calculating beta confidence
        #confidence = args[3]  # 1.96
        # calculating variance
        #sigma = args[4]#np.var(z)  # args[4] #1
        #SE = sigma * np.sqrt(np.diag(invA)) * confidence
        #beta_min = beta - SE
        #beta_max = beta + SE

        return ztilde#, beta, beta_min, beta_max 
    
    '''
    Singular Value Decomposition for Linear Regression
    '''
    def CallSVD(self, *args):
        # getting matrix
        X = args[0]
        # Applying SVD
        A = np.transpose(X) @ X
        U, s, VT = np.linalg.svd(A)
        D = np.zeros((len(U), len(VT)))
        for i in range(0, len(VT)):
            D[i, i] = s[i]
        UT = np.transpose(U)
        V = np.transpose(VT)
        invD = np.linalg.inv(D)
        invA = np.matmul(V, np.matmul(invD, UT))

        return invA
    
'''
Class to generate Data, 
for now only contains 
Franke function
'''
class DataFuncs:
    # constructor
    def __init__(self, *args):
        pass
    
    # Franke Function to generate Data Set
    def CallFranke(self, *args):
        x = args[0]
        y = args[1]
        term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
        term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
        term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
        term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
        return term1 + term2 + term3 + term4
    