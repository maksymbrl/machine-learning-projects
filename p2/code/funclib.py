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
import os
import sys
import numpy as np
import math as mt
# for polynimial manipulation
import sympy as sp
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sbn

from collections import Counter

import time

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
        #if z < 0:
        #    return 0
        #elif z >= 0:
        #    return (z>0)
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
        
    # elu
    def CalleLU(self, *args):
        z = args[0]
        return np.choose(z < 0, [z, (np.exp(z)-1)])
        
    def CallDeLU(self, *args):
        z = args[0]
        return np.choose(z > 0, [1,  np.exp(z)])
    
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
    
        # Stochastic gradient descent method with batches
    '''
    def SGD_batch(self, X, y, lr = 0.01, tol=1e-4, n_iter=1, batch_size=100, n_epoch=100, rnd_seed=False, adj_lr=False, rnd_batch=False, verbosity=0,lambda_r=0.0,new_per_iter=False):

        # lambda_r = lambda value for ridge regulation term in cost function.
        print("Doing SGD for logreg")
        n = len(y) 
        costs = []                                  # Initializing cost list
        
        if (rnd_seed):
            np.random.seed(int(time.time()))
        self.beta = np.random.randn(X.shape[1],1)   # Drawing initial random beta values
        tar = X@self.beta
        min_cost = self.cost.f(tar,y) + lambda_r*np.sum(self.beta**2)  # Save cost of new beta

        best_beta=self.beta.copy()

        # adjustable learning rate
        if (adj_lr):
            t0 = 5*n
            #t0 = n
            lr0=lr

        # We do several SGD searches with new batches for each search, with new searches
        # starting from the previous endpoint
        betas=np.zeros(shape=(X.shape[1]+1,n_iter)) #array to store the best betas with corresponding cost per iteration
        for i in range(n_iter):
            if (new_per_iter):
                self.beta = np.random.randn(X.shape[1],1)
                tar = X@self.beta
                min_cost = self.cost.f(tar,y) + lambda_r*np.sum(self.beta**2) 
                best_beta=self.beta.copy()

            if (verbosity>0):

                print('  search %i of %i'%(i+1,n_iter))
            # Data is (semi) sorted on age after index ~15000,
            # dividing into batches based on index is therefore potentially not random.
            # We therefore have 2 options, (1) draw batch_size random values for each
            # iteration 'j', or (2) split data into m batches before starting
            m=int(n/batch_size)
            if (rnd_batch):
                nbatch=[]
                nbatch.append(batch_size)
                idx=0
            else:
                batch_idx,nbatch=self.split_batch(n,m)
            for k in range(n_epoch):
                if (verbosity>1):
                    print('    epoch %i of %i'%(k+1,n_epoch))
                for j in range(m):
                    #values instead
                    if (rnd_batch):
                        idx_arr = np.random.randint(0,n,batch_size) # Choose n random data rows
                    else:
                        idx=np.random.randint(0,m)
                        idx_arr = batch_idx[idx,:nbatch[idx]]
                    X_ = X[idx_arr,:].reshape(nbatch[idx],X.shape[1]) # Select batch data
                    y_ = y[idx_arr].reshape(nbatch[idx],1)            # select corresponding prediction
                    b = X_@self.beta                # Calculate current prediction
                    gradient = ( X_.T @ (self.act.f(b)-y_)) + 2.0*lambda_r*self.beta # Calculate gradient
                    if (adj_lr):
                        #as iterations increase, the step size in beta is reduced
                        lr=(lr0*t0)/(t0+k*n+j*batch_size)

                    self.beta = self.beta - lr*gradient    # Calculate perturbation to beta

                #after each epoch we compute the cost (majority of runtime)
                tar = X@self.beta
                #calculate total cost (This takes a long time!!). Has support for ridge
                cost = self.cost.f(tar,y) + lambda_r*np.sum(self.beta**2)
                costs.append(cost)                      # Save cost of new beta
                if (cost < min_cost):
                    min_cost=cost
                    best_beta=self.beta.copy()
            betas[:X.shape[1],i]=best_beta[:,0].copy()
            betas[X.shape[1],i]=min_cost.copy()
        # if we draw new initial betas per iteration, we need to find the beta giving the
        # smallest cost of all iterations. If not, then the final best_beta is the one
        # we're after 
        if (new_per_iter):
            idx=np.argmin(betas[X.shape[1],:]) #find index with lowest cost
            self.beta[:,0]=betas[:X.shape[1],idx].copy() #finally return beta with the lowest total cost
        else:
            self.beta=best_beta.copy() #finally return beta with the lowest total cost

        return best_beta, costs, betas
    '''
    
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
        J = -np.sum(y * np.log(h+1e-10) +(1-y) * np.log(1-h+1e-10)) / m
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
    
    # Cost Function For Linear Regression Problems
    def CallNNMSE(self, *args):         # Mean Squared error
        Y = args[0]
        AL = args[1]
        modelParams=args[2]
        nLayers = args[3]
        m =args[4]
        lambd = args[5]
        #print(modelParams)
        J = 0.5*np.mean(( AL.ravel() - Y.ravel() )**2)
        if np.isnan(J):
            sys.exit()
            print("J", J)
        # sum of all weights (for all layers, except input one)
        Wtot = 0
        # Computing Regularisation Term for n layer NN
        for l in range(1, nLayers, 1):
            Wtot += np.sum(np.square(modelParams['W' + str(l)]))
        Wtot = Wtot * lambd / (2*m)
        if np.isnan(Wtot):
            
            print("Wtot", Wtot)
        J = J + Wtot
        

        return  J

    def CallDMSE(self, tar, y,  lmbd = 0) :
        return (tar-y)

'''
Class which contains all testing errors (MSNE, R^2, Accuracy etc.)
'''
class ErrorFuncs:
    def __init__(self, *args):
        pass
    
    # MSNE
    def CallMSE(self, *args):
        z_data = args[0]
        z_model = args[1]
        n = np.size(z_model)
        return np.sum((z_data - z_model)**2) / n
    
    # R^2 test
    def CallR2(self, *args):
        z_data = args[0]
        z_model = args[1]
        return 1 - np.sum((z_data - z_model)**2) / np.sum((z_data - np.mean(z_data))**2)
    
    # The complete F1 score
    def CallF1(self, *args):
        Y_true = args[0]
        #print(np.shape(Y_true))
        Y_pred = args[1]
        #print(np.shape(Y_pred))
        # First Calculating False Positives(FP), True Negatives (TN)
        # True Positives (TP) and True Negatives(TN):
        #TP,TN,FP,FN = 0,0,0,0
        # Identifying all values at once as
        #counts = Counter(zip(Y_pred, Y_true))
        #TP = counts[1,1]
        #TN = counts[0,0]
        #FP = counts[1,0]
        #FN = counts[0,1]
        tp=0
        tn=0
        fp=0
        fn=0
        pred=np.where(pred>threshold,1,0)
        for i in range(len(ytrue)):
            if (Y_pred[i]==1 and Y_true[i]==1):
                tp +=1
            elif (Y_pred[i]==1 and Y_true[i]==0):
                fp +=1
            elif (Y_pred[i]==0 and Y_true[i]==0):
                tn +=1
            elif (Y_pred[i]==0 and Y_true[i]==1):
                fn +=1
        pcp=np.sum(np.where(pred==1,1,0))
        pcn=np.sum(np.where(pred==0,1,0))
        cp=np.sum(np.where(ytrue==1,1,0))
        cn=np.sum(np.where(ytrue==0,1,0))
        ppv=[tn*1.0/pcn, tp*1.0/pcp]
        trp=[tn*1.0/cn, tp*1.0/cp]
        ac=(tp+tn)*1.0/(cp+cn)
        f1=[2.0*ppv[0]*trp[0]/(ppv[0]+trp[0]), 2.0*ppv[1]*trp[1]/(ppv[1]+trp[1])]
        if return_f1:
            if return_ac:
                return (f1[0]*cn+f1[1]*cp)/(cn+cp),ac
            else:
                return (f1[0]*cn+f1[1]*cp)/(cn+cp)
        if return_ac:
            return ac
        print("              precision     recall     f1-score     true number    predicted number")
        print()
        print("           0      %5.3f      %5.3f        %5.3f        %8i    %16i"%(ppv[0],trp[0],f1[0],cn,pcn))
        print("           1      %5.3f      %5.3f        %5.3f        %8i    %16i"%(ppv[1],trp[1],f1[1],cp,pcp))
        print()
        print("    accuracy                              %5.3f        %8i"%((tp+tn)*1.0/(cp+cn),cp+cn))
        print("   macro avg      %5.3f      %5.3f        %5.3f        %8i"%((ppv[0]+ppv[1])/2.0,(trp[0]+trp[1])/2.0, (f1[0]+f1[1])/2.0,cn+cp))
        print("weighted avg      %5.3f      %5.3f        %5.3f        %8i"%((ppv[0]*cn+ppv[1]*cp)/(cn+cp),(trp[0]*cn+trp[1]*cp)/(cn+cp), (f1[0]*cn+f1[1]*cp)/(cn+cp),cn+cp))
        print()
        #print(TP)
        # Accuracy: checking for 0 occurence
        #Accuracy = (TP + TN) / float(len(Y_true)) if Y_true else 0
        # Precision:
        #Precision = TP / (TP + FP)
        # Recall (should be above 0.5 than it is good)
        #Recall = TP / (TP + FN)
        # F1, could've used 2 * (PRE * REC) / (PRE + REC), but this one doesn't suffer
        # from 0 devision issue
        #F1 = (2 * TP) / (2 * TP + FP + FN)
        
        return Accuracy, Precision, Recall, F1
    
    def own_classification_report(self,ytrue,pred,threshold=0.5,return_f1=False,return_ac=False):
        tp=0
        tn=0
        fp=0
        fn=0
        pred=np.where(pred>threshold,1,0)
        for i in range(len(ytrue)):
            if (pred[i]==1 and ytrue[i]==1):
                tp +=1
            elif (pred[i]==1 and ytrue[i]==0):
                fp +=1
            elif (pred[i]==0 and ytrue[i]==0):
                tn +=1
            elif (pred[i]==0 and ytrue[i]==1):
                fn +=1
        pcp=np.sum(np.where(pred==1,1,0))
        pcn=np.sum(np.where(pred==0,1,0))
        cp=np.sum(np.where(ytrue==1,1,0))
        cn=np.sum(np.where(ytrue==0,1,0))
        ppv=[tn*1.0/pcn, tp*1.0/pcp]
        trp=[tn*1.0/cn, tp*1.0/cp]
        ac=(tp+tn)*1.0/(cp+cn)
        f1=[2.0*ppv[0]*trp[0]/(ppv[0]+trp[0]), 2.0*ppv[1]*trp[1]/(ppv[1]+trp[1])]
        if return_f1:
            if return_ac:
                return (f1[0]*cn+f1[1]*cp)/(cn+cp),ac
            else:
                return (f1[0]*cn+f1[1]*cp)/(cn+cp)
        if return_ac:
            return ac
        print("              precision     recall     f1-score     true number    predicted number")
        print()
        print("           0      %5.3f      %5.3f        %5.3f        %8i    %16i"%(ppv[0],trp[0],f1[0],cn,pcn))
        print("           1      %5.3f      %5.3f        %5.3f        %8i    %16i"%(ppv[1],trp[1],f1[1],cp,pcp))
        print()
        print("    accuracy                              %5.3f        %8i"%((tp+tn)*1.0/(cp+cn),cp+cn))
        print("   macro avg      %5.3f      %5.3f        %5.3f        %8i"%((ppv[0]+ppv[1])/2.0,(trp[0]+trp[1])/2.0, (f1[0]+f1[1])/2.0,cn+cp))
        print("weighted avg      %5.3f      %5.3f        %5.3f        %8i"%((ppv[0]*cn+ppv[1]*cp)/(cn+cp),(trp[0]*cn+trp[1]*cp)/(cn+cp), (f1[0]*cn+f1[1]*cp)/(cn+cp),cn+cp))
        print()

        return
    
    # I will be using scikit functionalities
    # To estimate errors etc. Why? Because I can :)
    # Accuracy
    #def accuracy_score_numpy(Y_test, Y_pred):
    #    return np.sum(Y_test == Y_pred) / len(Y_test)
    
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
        poly_degree = args[2]
        # getting inputs
        #x_vals = self.x_vals
        x_symb = args[0]
        x_vals = args[1]
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
    
    # Beale's function
    def CallBeale(self, *args):
        x = args[0] 
        y = args[1]
        return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
    
'''
Class which contains ll plotting functions
'''
class PlotFuncs:
    # constructorf
    def __init__(self, *args):
        pass
    
    # Plotting Surface Just to see that we Succeeded
    def PlotSurface(self, *args):
        # passing coordinates
        x = args[0]
        y = args[1]
        # takes an array of z values
        zarray = args[2]
        # output dir
        output_dir = args[3]
        # filename
        filename = args[4]
        print(filename)
        # Turning interactive mode on
        #plt.ion()
        fig = plt.figure(figsize=(10, 3))
        axes = [fig.add_subplot(1, 3, i, projection='3d') for i in range(1, len(zarray) + 1)]
        #axes[0].view_init(5,50)
        #axes[1].view_init(5,50)
        #axes[2].view_init(5,50)
        surf = [axes[i].plot_surface(x, y, zarray[i], alpha = 0.5,
                                     cmap = 'brg_r', label="Franke function", linewidth = 0, antialiased = False) for i in range(len(zarray))]
        # saving figure with corresponding filename
        #fig.savefig(output_dir + filename)
        # close the figure window
        #plt.close(fig)
        plt.show()
        # turning the interactive mode off
        #plt.ioff()
        
        
    def plot_cumulative(self,X,y,p=[],beta=[],label='',plt_ar=True,return_ar=False):
        if (len(p)==0):
            if(len(beta)==0):
                beta=self.beta
            p=self.act.f(X@beta)
        if (not label==''):
            lab = '_'+label
        else:
            #make a date and time stamp
            t=time.ctime()
            ta=t.split()
            hms=ta[3].split(':')
            label=ta[4]+'_'+ta[1]+ta[2]+'_'+hms[0]+hms[1]+hms[2]
            lab='_'+label
        temp_p=p[:,0].copy()
        nd=len(temp_p)
        nt=np.sum(y)
        model_pred=np.zeros(nd+1)
        for i in range(len(temp_p)):
            idx=np.argmax(temp_p)
            model_pred[i+1]=model_pred[i]+y[idx,0]
            temp_p[idx]=-1.0

        x_plt=np.arange(nd+1)
        best_y=np.arange(nd+1)
        best_y[nt:]=nt
        baseline=(1.0*nt)/nd*x_plt

        ar=1.0*np.sum(model_pred-baseline)/np.sum(best_y-baseline)
        if return_ar:
            return ar

        xtick=[]
        if (nd<2000):
            j=500
            nm=2001
        else:
            j=4000
            nm=16000
        for k in range(0,nm,j):
            xtick.append(k)
            if (k>nd):
                break
        if (label=='lift'):
            xtick = [0,nt,nd]
            xtick_lab=['0',r'$N_t$',r'$N_d$']
            ytick = [0,nt]
            ytick_lab=['0',r'$N_t$']
        plt.figure(1,figsize=(7,7))
        plt.plot(x_plt,best_y,label='Best fit',color=plt.cm.tab10(0))
        plt.plot(x_plt,model_pred,label='Model',color=plt.cm.tab10(1))
        plt.plot(x_plt,baseline,label='Baseline',color=plt.cm.tab10(7))
        plt.legend(loc='lower right',fontsize=22)
        plt.xlabel('Number of total data',fontsize=22)
        if (label=='lift'):
            plt.xticks(xtick,xtick_lab,fontsize=18)
            plt.yticks(ytick,ytick_lab,fontsize=18)
        else:
            plt.xticks(xtick,fontsize=18)
            plt.yticks(fontsize=18)
        plt.ylabel('Cumulative number of target data',fontsize=22)
        if (plt_ar):
            plt.text(nd*0.55,nt*0.4,'area ratio = %5.3f'%(ar), fontsize=20)
        plt.savefig('plots/cumulative_plot'+lab+'.pdf',bbox_inches='tight',pad_inches=0.02)
        plt.clf()

        return
    