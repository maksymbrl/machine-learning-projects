# -*- coding: utf-8 -*-

# TODO: https://compphysics.github.io/MachineLearning/doc/pub/Regression/html/._Regression-bs095.html#___sec94 


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(4155)

def franke_function(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4

# Polynomial
# Creates the  matrix X
def design_poly_matrix(x, y, n):
    N = len(x)
    l = int((n+1)*(n+2)/2)		
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k

    return X

# Functions of statistical measures
def MSE(x, x_):
    """
        Calculating the Mean Square Error.
        Argument (numpy array x, and \tilde{x})
        
        here x_ can either an array, or a constant.
        
        returns a double
    """
    return np.average((x - x_) ** 2)

def bias(x, x_):
    return np.average((x - np.average(x_)) ** 2)

def average(x):
    return np.average(x)

def var(x):
    return np.average((x - np.average(x)) ** 2)

def R2(x, x_):
    """
        Calcualting the R2 score:
        x, and x_ are both numpy arrays
        the output will be a double
    """
    x_avg =  np.average(x)
    return 1. - MSE(x, x_)/MSE(x, x_avg)

def test():
    # Generate the data
    nrow = 100
    ncol = 200
    ax_row = np.random.uniform(0, 1, size=nrow)
    ax_col = np.random.uniform(0, 1, size=ncol)
    
    ind_sort_row = np.argsort(ax_row)
    ind_sort_col = np.argsort(ax_col)
    
    ax_row_sorted = ax_row[ind_sort_row]
    ax_col_sorted = ax_col[ind_sort_col]
    
    colmat, rowmat = np.meshgrid(ax_col_sorted, ax_row_sorted)
    
    noise_str = .0
    noise = np.random.randn(nrow, ncol)
    
    z = franke_function(rowmat, colmat) + noise_str * noise
    
    
    
    row_arr = rowmat.ravel()
    col_arr = colmat.ravel()
    z_arr = z.ravel()
    
    X = design_poly_matrix(row_arr, col_arr, 10)
    #beta = Ridge(row_arr, col_arr, z_arr, 10, 0.1)

    #print("READDDDD", len(X), len(X[0]))
    #print("READDDDD", len(beta))
    #Z = np.matmul(X, beta)
    #Z_reshaped = Z.reshape(nrow, ncol)
    
    # Generate the design matrix
    p = 10
    poly = PolynomialFeatures(degree = p)
    X = poly.fit_transform(np.c_[row_arr, col_arr])
    print(np.shape(X)[0], np.shape(X)[1])
    
    ## Perform OLS
    linreg = LinearRegression()
    linreg.fit(X, z_arr)
    #print("READDDDD", len(X), len(X[0]))
    zpred = linreg.predict(X)
    zplot = zpred.reshape(nrow, ncol)
    
    
    # Plot the resulting fit beside the original surface
    fig = plt.figure()
    
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    surf = ax.plot_surface(colmat, rowmat, z, cmap=cm.viridis, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Franke')
    
    #ax = fig.add_subplot(1, 3, 2, projection='3d')
    #surf = ax.plot_surface(colmat, rowmat, Z_reshaped, cmap=cm.viridis, linewidth=0, antialiased=False)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    #plt.title('Maziar Franke')
    
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    surf = ax.plot_surface(colmat, rowmat, zplot, cmap=cm.viridis, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Fitted Franke')
    
    plt.show()
    #MSE_score = MSE(Z, z_arr)
    #R2_score = R2(Z, z_arr)
    
    #print("MSE_maziar ", MSE_score)
    #print("R2_maziar = ", R2_score)
    
    #print("MSE_M ", MSE(z_arr, zpred))
    #print("R2_GOD = ", R2(z_arr, zpred))
    


#test()



        
    
    