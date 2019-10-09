import numpy as np
# for polynimial manipulation
import sympy as sp
# from sympy import *
import itertools as it
# for plotting stuff
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Scikit learn utilities
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


class RegressionLibrary:
    '''
    class constructor
    '''

    def __init__(self, *args):
        # getting input values in place
        self.x_symb = args[0]
        self.x_vals = args[1]

    '''
    Franke function, used to generate outputs (z values)
    '''
    def FrankeFunction(self, x, y):
        term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
        term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
        term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
        term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
        return term1 + term2 + term3 + term4

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
        points = len(x_vals[0]) * len(x_vals[1])
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
        for i in range(0, len(VT)):
            D[i, i] = s[i]
        UT = np.transpose(U)
        V = np.transpose(VT)
        invD = np.linalg.inv(D)
        invA = np.matmul(V, np.matmul(invD, UT))

        return invA

    '''
    k-Fold Cross Validation
    '''
    # method for shuffling arrays randomly (but simultaneously <= we still have ordered pairs)
    def shuffleDataSimultaneously(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    # function to split data set manually
    def splitDataset(self, *args):
        # getting inputs
        X = args[0]
        z = args[1]
        kfold = args[2]
        iterator = args[3]
        # If the dataset does not cleanly divide by the number of folds,
        # there may be some remainder rows and they will not be used in the split.
        length = len(X) % kfold
        if length == 0:
            condition = True
        else:
            condition = False
        while condition is False:
            # removing the element <= they were shuffled randomly,
            # so it doesn't matter which one to remove
            X = np.delete(X, -1, axis = 0)
            z = np.delete(z, -1, axis = 0)
            # checking whether it is divided cleanly
            length = len(X) % kfold
            if length == 0:
                condition = True
        # 2. Split the dataset into k groups:
        X_split = np.array_split(X, kfold, axis=0)
        z_split = np.array_split(z, kfold, axis=0)
        # train data set - making a copy of the shuffled and splitted arrays
        X_train = X_split.copy()
        z_train = z_split.copy()
        # test data set - each time new element
        X_test = X_split[iterator]
        z_test = z_split[iterator]
        # deleting current element
        X_train = np.delete(X_train, iterator, 0)
        z_train = np.delete(z_train, iterator, 0)
        # and adjusting arrays dimensions (e.g. X: [4, 500, 21] => [2000, 21])
        X_train = np.concatenate(X_train, axis=0)
        z_train = z_train.ravel()

        return X_train, X_test, z_train, z_test
    '''
    Linear Cross validation (manual algorithm)
    '''
    def doCrossVal(self, *args):
        # getting design matrix
        X = args[0]
        # getting z values and making them 1d
        z = np.ravel(args[1])
        kfold = args[2]
        # Splitting and shuffling data randomly
        #X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=1. / kfold, shuffle=True)
        MSEtest_lintot = []
        MSEtrain_lintot = []
        z_tested = []
        z_trained = []
        z_t = []
        # bias
        bias = []
        # shuffling dataset randomly
        # 1. Shuffling datasets randomly:
        X, z = self.shuffleDataSimultaneously(X, z)
        # splitting data sets into the kfold and iterate over each of them
        for i in range(kfold):
            # Splitting and shuffling data randomly
            #X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=1./kfold, shuffle=True)
            X_train, X_test, z_train, z_test = self.splitDataset(X, z, kfold, i)
            z_t.append(z_test)
            # Train The Pipeline
            invA = self.doSVD(X_train)
            beta_train = invA.dot(X_train.T).dot(z_train)
            # Testing the pipeline
            z_trained.append(X_train @ beta_train)
            z_tested.append(X_test @ beta_train)
            # Calculating MSE for each iteration
            MSEtest_lintot.append(self.getMSE(z_test, z_tested[i]))
            MSEtrain_lintot.append(self.getMSE(z_train, z_trained[i]))
        # linear MSE
        MSEtest_lin = np.mean(MSEtest_lintot)
        MSEtrain_lin = np.mean(MSEtrain_lintot)
        # bias-variance trade off
        z_tested_mean = np.mean(z_tested, axis=1, keepdims=True)
        for i in range(kfold):
            bias.append((z_t[i] - z_tested_mean)**2)
        bias_mean = np.mean( bias )
        variance_mean = np.mean( np.var(z_tested, axis=1, keepdims=True) )

        return MSEtest_lin, MSEtrain_lin, bias_mean, variance_mean

    '''
    Ridge Cross validation - manual algorithm
    '''
    def doCrossValRidge(self, *args):
        # getting design matrix
        X = args[0]
        # getting z values and making them 1d
        z = np.ravel(args[1])
        kfold = args[2]
        lambda_par = args[3]
        MSEtest_ridgetot = []
        MSEtrain_ridgetot = []
        z_tested = []
        z_trained = []
        # saving test data set to calculate bias-variance trade off
        z_t = []
        # bias
        bias = []
        # shuffling dataset randomly
        # 1. Shuffling datasets randomly:
        X, z = self.shuffleDataSimultaneously(X, z)
        for i in range(kfold):
            # Splitting and shuffling data randomly
            #X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=1./kfold, shuffle=True)
            X_train, X_test, z_train, z_test = self.splitDataset(X, z, kfold, i)
            z_t.append(z_test)
            # constructing the identity matrix
            I = np.identity(len(X_train.T.dot(X_train)), dtype=float)
            # Train The Pipeline
            # calculating parameters
            invA = np.linalg.inv(X_train.T.dot(X_train) + lambda_par * I)
            beta_train = invA.dot(X_train.T).dot(z_train)
            # Testing the pipeline
            z_trained.append(X_train @ beta_train)
            z_tested.append(X_test @ beta_train)
            # Calculating MSE for each iteration
            MSEtest_ridgetot.append(self.getMSE(z_test, z_tested[i]))
            MSEtrain_ridgetot.append(self.getMSE(z_train, z_trained[i]))
        # Ridge MSE
        MSEtest_ridge = np.mean(MSEtest_ridgetot)
        MSEtrain_ridge = np.mean(MSEtrain_ridgetot)
        # bias-variance trade off
        z_tested_mean = np.mean(z_tested, axis=1, keepdims=True)
        for i in range(kfold):
            bias.append((z_t[i] - z_tested_mean)**2)
        bias_mean = np.mean( bias )
        variance_mean = np.mean( np.var(z_tested, axis=1, keepdims=True) )

        return MSEtest_ridge, MSEtrain_ridge, bias_mean, variance_mean
    '''
    Cross Validation using Scikit Learn functionalities (all at once)
    '''
    def doCrossValScikit(self, *args):
        # getting inputs
        X = args[0]
        z = args[1]
        kfold = args[2]
        poly_degree = args[3]
        lambda_par = args[4]
        # understanding the regression type to use
        reg_type = args[5]
        if reg_type == 'linear':
            model = LinearRegression(fit_intercept = False)
        elif reg_type == 'ridge':
            model = Ridge(alpha = lambda_par, fit_intercept = False)
        elif reg_type == 'lasso':
            model = Lasso(alpha = lambda_par, normalize=True)
        else:
            print("Houston, we've got a problem!")

        MSEtest = []
        MSEtrain = []
        # bias
        bias = []
        z_t = []
        z_tested = []
        # If the dataset does not cleanly divide by the number of folds,
        # there may be some remainder rows and they will not be used in the split.
        length = len(X) % kfold
        if length == 0:
            condition = True
        else:
            condition = False
        while condition is False:
            # removing the element <= they were shuffled randomly,
            # so it doesn't matter which one to remove
            X = np.delete(X, -1, axis = 0)
            z = np.delete(z, -1, axis = 0)
            # checking whether it is divided cleanly
            length = len(X) % kfold
            if length == 0:
                condition = True
        # making splits - shuffling it
        cv = KFold(n_splits = kfold, shuffle = True, random_state = 1)
        # enumerate splits - splitting the data set to train and test splits
        for train, test in cv.split(X):
            X_train, X_test = X[train], X[test]
            z_train, z_test = z[train], z[test]
            z_t.append(z_test)
            # making the prediction - comparing outputs for current and "future" datasets
            z_tilde = model.fit(X_train, z_train).predict(X_train).ravel() # z_trained
            z_pred = model.fit(X_train, z_train).predict(X_test).ravel() # z_tested
            z_tested.append(z_pred)

            MSEtest.append(mean_squared_error(z_test, z_pred))
            MSEtrain.append(mean_squared_error(z_train, z_tilde))

        # getting the mean values for errors (to plot them later)
        MSEtest_mean = np.mean(MSEtest)
        MSEtrain_mean = np.mean(MSEtrain)
        # bias-variance trade off
        z_tested_mean = np.mean(z_tested, axis=1, keepdims=True)
        for i in range(kfold):
            bias.append((z_t[i] - z_tested_mean)**2)
        bias_mean = np.mean( bias )
        variance_mean = np.mean( np.var(z_tested, axis=1, keepdims=True) )

        # returning MSE, bias and variance for  a given polynomial degree
        return MSEtest_mean, MSEtrain_mean, bias_mean, variance_mean

    '''
    MSE - the smaller the better (0 is the best?)
    '''

    def getMSE(self, z_data, z_model):
        n = np.size(z_model)
        return np.sum((z_data - z_model) ** 2) / n

    '''
    R^2 - values should be between 0 and 1 (with 1 being the best)
    '''

    def getR2(self, z_data, z_model):
        return 1 - np.sum((z_data - z_model) ** 2) / np.sum((z_data - np.mean(z_data)) ** 2)

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
        # calculating variance of data

        # and then make the prediction
        invA = self.doSVD(X)
        beta = invA.dot(X.T).dot(z)
        ztilde = X @ beta
        # calculating beta confidence
        confidence = args[2]  # 1.96
        sigma = args[3]#np.var(z)  # args[3] #1
        SE = sigma * np.sqrt(np.diag(invA)) * confidence
        beta_min = beta - SE
        beta_max = beta + SE

        return ztilde, beta, beta_min, beta_max  # z_trained#ztilde#, beta, SE

    '''
    Ridge Regression
    '''

    def doRidgeRegression(self, *args):
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
        invA = np.linalg.inv(XTX + lambda_par * I)
        beta = invA.dot(X.T).dot(z)
        # and making predictions
        ztilde = X @ beta

        # calculating beta confidence
        confidence = args[3]  # 1.96
        # calculating variance
        sigma = args[4]#np.var(z)  # args[4] #1
        SE = sigma * np.sqrt(np.diag(invA)) * confidence
        beta_min = beta - SE
        beta_max = beta + SE

        return ztilde, beta, beta_min, beta_max  # , beta, SE

    '''
    LASSO Regression
    '''

    def doLASSORegression(self, *args):
        pass

    '''
    Methods to plot data
    '''
    def plotBeta(self, *args):
        x = args[0]
        y = args[1]
        y_min = args[2]
        y_max = args[3]
        output_dir = args[4]
        filename = args[5]
        # Turning interactive mode on
        #plt.ion()
        fig = plt.figure()#figsize = (10, 3))
        axe = fig.add_subplot(1, 1, 1)
        axe.plot(x, y, 'bo', label=r'$\beta$')
        axe.plot(x, y_min, 'r--', label=r'$\beta_{min}$')
        axe.plot(x, y_max, 'g--', label=r'$\beta_{max}$')
        axe.legend()
        # setting axes to log scale (to account for very high beta?)
        #axe.set_yscale('log')
        plt.grid(True)
        plt.xlabel('number of ' + r'$\beta$')
        plt.ylabel(r'$\beta$')
        fig.savefig(output_dir + '/' + filename)
        # close the figure window
        plt.close(fig)
        # turning the interactive mode off
        #plt.ioff()

    def plotSurface(self, *args):
        # passing coordinates
        x = args[0]
        y = args[1]
        # takes an array of z values
        zarray = args[2]
        # output dir
        output_dir = args[3]
        # filename
        filename = args[4]
        # Turning interactive mode on
        #plt.ion()
        fig = plt.figure(figsize=(10, 3))
        axes = [fig.add_subplot(1, 3, i, projection='3d') for i in range(1, len(zarray) + 1)]
        surf = [axes[i].plot_surface(x, y, zarray[i], alpha = 0.5,
                                     cmap = 'brg_r', linewidth = 0, antialiased = False) for i in range(len(zarray))]
        # saving figure with corresponding filename
        fig.savefig(output_dir + '/' + filename)
        # close the figure window
        plt.close(fig)
        # turning the interactive mode off
        #plt.ioff()