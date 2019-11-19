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

import funclib

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
        x_symb = args[0]
        # array of values for each variable/feature
        x_vals = args[1]
        # grid values
        x = args[2]
        y = args[3]
        z = args[4]
        # 1.96 to calculate stuff with 95% confidence
        confidence = args[5]
        # noise variance - for confidence intervals estimation
        sigma = args[6]
        # k-value for Cross validation
        kfold = args[7]
        # hyper parameter
        lambda_par = args[8]
        # directory where to store plots
        output_dir = args[9]
        prefix = args[10]
        # degree of polynomial to fit
        poly_degree = args[11]
        #=====================================================================#
        func = funclib.NormalFuncs()
        # getting the design matrix
        X = func.ConstructDesignMatrix(x_symb, x_vals, poly_degree)
        
        
        
        
    def DoLogisticRegression(self, *args):
        #=====================================================================#
        # Logistic Regression variables
        #=====================================================================#
        pass