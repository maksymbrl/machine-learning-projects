# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
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

def average(x):
    return np.average(x)

def bias(x, x_):
    return np.average((x - np.mean(x_)) ** 2)

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

class OLSClass:
    def __init__(self, x, y, z, degree):
        self.x = x
        self.y = y
        self.z = z
        
        self.X = None
        self.X_test = None
        self.X_train = None
        self.beta = None
        
        self.Z_test = None
        self.Z_train = None
        self.degree = degree
        
        self.beta_OLS = None
        self.beta_Ridge = None
        self.beta_Lasso = None
        self.clf_Lasso = None
        

        
        
    def resample(self, training_portion=0.75):
        self.X_train, self.X_test, self.Z_train, self.Z_test = \
            train_test_split(self.X, self.z, test_size=1 - training_portion, random_state=42)
            
    def predict (self, x, model='OLS'):
        if x is None:
            x = self.X
        if model == 'OLS':
            return np.matmul(x, self.beta_OLS)
        
        if model == 'Ridge':
            return np.matmul(x, self.beta_Ridge)
        
        if model == 'Lasso':
            return np.matmul(x, self.beta_Lasso)
    
    # Returns a set of indexes in each folds
    def k_fold_split(self, k):
        n = len(self.x)
        print ("length of the whole data ", n)
        # making random numbers
        arr = np.arange(n)
        # making the set even
        np.random.shuffle(arr)
        print(len(arr))
        # make the lenght dividable by k:
        print(len(arr))
        k_fold = []
        for i in range(0, int(n/k)*k, int(n/k)):
            print (i, i + int(n/k))
            k_fold.append([])
            k_fold[-1] = arr[i:i + int(n/k)]
            print (len(k_fold[-1]))
        
        return k_fold
        
    def OLS(self, X, expected_value):
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(expected_value)
        #beta = np.linalg.lstsq(X, expected_value)[0]
        self.beta_OLS = beta
        return beta
    
    def Ridge(self, X, l, expected_value):
        #X = design_poly_matrix(x, y, degree)
        beta = np.linalg.inv(X.T.dot(X) + np.eye(np.shape(X)[1])*l).dot(X.T).dot(expected_value)
        self.beta_Ridge =  beta
        return beta

    def Lasso(self, X, a, expected_value):
        #self.X = design_poly_matrix(x, y, self.degree)
        clf = linear_model.Lasso(alpha=a, fit_intercept=False, max_iter=1000)
        clf.fit(X, expected_value)
        self.beta_Lasso = clf.coef_
        self.clf_Lasso = clf
        return clf.coef_
    
    def apply_k_fold (self, k):
        betas = []
        R2_scores = []
        MSE_scores = []
        
        k_fold_list = self.k_fold_split(k)
        
        for i in range(k):
            tmp_x_train, tmp_x_test = [], []
            tmp_z_train, tmp_z_test = [], []
            
            print("Now using fold, ", i)
            
            for j in range(k):
                print("Copying group ", j, " with size ", len(k_fold_list[j]))
                for ind in k_fold_list[j]:
                    if i == j:
                        #print ("making testing set")
                        tmp_x_test.append(self.X[ind])
                        tmp_z_test.append(self.z[ind])
                    else:
                        #print("making training test")
                        tmp_x_train.append(self.X[ind])
                        tmp_z_train.append(self.z[ind])
            
            tmp_x_train, tmp_x_test = np.array(tmp_x_train), np.array(tmp_x_test)
            tmp_z_train, tmp_z_test = np.array(tmp_z_train), np.array(tmp_z_test)
            
            beta_tmp = self.OLS(tmp_x_train, tmp_z_train)
            betas.append(beta_tmp)
            
            
            z_tmp = np.matmul(tmp_x_test, betas[-1])
            MSE_scores.append(MSE(z_tmp, tmp_z_test))
            R2_scores.append(R2(z_tmp, tmp_z_test))
        
        return betas, MSE_scores, R2_scores
    
def test_regression():
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
    
    test_OLS = OLSClass(row_arr, col_arr, z_arr , 10)
    betas, MSE_scores, R2_scores = test_OLS.apply_k_fold(3)
    for i in range(len(MSE_scores)):
        print(MSE_scores[i], R2_scores[i])
        
#$test_regression()

def franke_data ():
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
    
    return row_arr, col_arr, z_arr

def terrain_data(row_size, col_size):
    terrain = imread('SRTM_data_Norway_1.tif')
    row_len, col_len = np.shape(terrain)
    
    row = np.linspace(0, 1, row_size)
    col = np.linspace(0, 1, col_size)
    
    colmat, rowmat = np.meshgrid(col, row)
    
    z = terrain[:row_size, :col_size]
    
    row_arr = rowmat.ravel()
    col_arr = colmat.ravel()
    z_arr = z.ravel()
    
    return row_arr, col_arr, z_arr

def testOLS(n,  row_arr, col_arr, z_arr):
    
    bias_values = []
    var_values = []
    MSE_values = []
    R2_values = []
    for i in range(n):
        OLS_object = OLSClass(row_arr, col_arr, z_arr, i)
        OLS_object.X = design_poly_matrix(row_arr, col_arr, i)
        OLS_object.resample()
        
        beta = OLS_object.OLS(OLS_object.X_train, OLS_object.Z_train)
        z_predict = OLS_object.predict(OLS_object.X_test)
        
        bias_values.append(bias(OLS_object.Z_test, z_predict))
        var_values.append(var(z_predict))
        MSE_values.append(MSE(OLS_object.Z_test, z_predict))
        R2_values.append(R2(OLS_object.Z_test, z_predict))
        
        
    return bias_values, var_values, MSE_values, R2_values




def testOLS_kfold(n, row_arr, col_arr, z_arr):
    
    bias_values = []
    var_values = []
    MSE_values = []
    R2_values = []
    degree = 11
    for i in range(2, n):
        OLS_object = OLSClass(row_arr, col_arr, z_arr, degree)
        OLS_object.X = design_poly_matrix(row_arr, col_arr, degree)
        beta, MSE_score,R2_score =  OLS_object.apply_k_fold(i)
        MSE_values.append(min(MSE_score))
        
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(list(range(2, n)), MSE_values, label='MSE')
   
    ax.set_xlabel('k in k-fold')
    ax.legend()
    plt.show()
        
        
    return bias_values, var_values, MSE_values, R2_values

def plot_MSE(n):
    bias_values, var_values, MSE_values, R2_values = testOLS(n, franke_data())
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(list(range(n)), MSE_values, label='MSE')
    ax.plot(list(range(n)), var_values, label='var')
    ax.plot(list(range(n)), bias_values, label='bias')
    ax.set_xlabel('degree of P')
    ax.legend()
    plt.show()
 
def testRidge(n, row_arr, col_arr, z_arr):
    len_lambda = 10
    lambda_start, lambda_end = 1e-10, 10e-2
    lambda_set = np.linspace(lambda_start, lambda_end, len_lambda)
    bias_values = np.zeros((n, len_lambda))
    var_values =  np.zeros((n, len_lambda))
    MSE_values =  np.zeros((n, len_lambda))
    R2_values =  np.zeros((n, len_lambda))
    
    colmat = np.zeros((n, len_lambda))
    rowmat = np.zeros((n, len_lambda))
    for j in range(n):
        OLS_object = OLSClass(row_arr, col_arr, z_arr, j)
        OLS_object.X = design_poly_matrix(row_arr, col_arr, j)
        OLS_object.resample()
        for i in range(len_lambda):
            colmat[j][i] = j
            rowmat[j][i] = lambda_set[i]
            beta = OLS_object.Ridge(OLS_object.X_train, lambda_set[i], OLS_object.Z_train)
            z_predict = OLS_object.predict(OLS_object.X_test, 'Ridge')
            
            bias_values[j][i] = bias(OLS_object.Z_test, z_predict)
            var_values[j][i] = var(z_predict)
            MSE_values[j][i] = MSE(OLS_object.Z_test, z_predict)
            R2_values[j][i] = R2(OLS_object.Z_test, z_predict)
        
    print(MSE_values.shape)
    print(lambda_set.shape)
    print(np.array(list(range(n))).shape)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    surf = ax.plot_surface(colmat, rowmat, MSE_values, cmap=cm.viridis, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('degree of P')
    ax.set_ylabel('Lambda')
    ax.set_zlabel('MSE')
    ax.title("Optimizing Ridge Franke")
    plt.show()
    
    
    return bias_values, var_values, MSE_values, R2_values


def testLasso(n, row_arr, col_arr, z_arr):
    len_lambda = 100
    lambda_start, lambda_end = 1e-10, 1e0
    lambda_set = np.linspace(lambda_start, lambda_end, len_lambda)
    bias_values = np.zeros((n, len_lambda))
    var_values =  np.zeros((n, len_lambda))
    MSE_values =  np.zeros((n, len_lambda))
    R2_values =  np.zeros((n, len_lambda))
    
    colmat = np.zeros((n, len_lambda))
    rowmat = np.zeros((n, len_lambda))
    for j in range(n):
        OLS_object = OLSClass(row_arr, col_arr, z_arr, j)
        OLS_object.X = design_poly_matrix(row_arr, col_arr, j)
        OLS_object.resample()
        for i in range(len_lambda):
            colmat[j][i] = j
            rowmat[j][i] = lambda_set[i]
            beta = OLS_object.Lasso(OLS_object.X_train, lambda_set[i], OLS_object.Z_train)
            z_predict = OLS_object.predict(OLS_object.X_test, 'Lasso')
            
            bias_values[j][i] = bias(OLS_object.Z_test, z_predict)
            var_values[j][i] = var(z_predict)
            MSE_values[j][i] = MSE(OLS_object.Z_test, z_predict)
            R2_values[j][i] = R2(OLS_object.Z_test, z_predict)
        
    print(MSE_values.shape)
    print(lambda_set.shape)
    print(np.array(list(range(n))).shape)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    surf = ax.plot_surface(colmat, rowmat, MSE_values, cmap=cm.viridis, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('degree of P')
    ax.set_ylabel('Alpha')
    ax.set_zlabel('MSE')
    plt.title("Optimizing Lasso Franke")
    plt.show()
    
    
    return bias_values, var_values, MSE_values, R2_values

def testLassoAlpha(n, row_arr, col_arr, z_arr):
    len_lambda = 10
    lambda_start, lambda_end = 1e-10, 0.02
    lambda_set = np.linspace(lambda_start, lambda_end, len_lambda)

    MSE_values =  []
    R2_values =  []
    
    degree = n
    
    OLS_object = OLSClass(row_arr, col_arr, z_arr, degree)
    OLS_object.X = design_poly_matrix(row_arr, col_arr, degree)
    OLS_object.resample()
    for i in range(len_lambda):
        beta = OLS_object.Lasso(OLS_object.X_train, lambda_set[i], OLS_object.Z_train)
        z_predict = OLS_object.predict(OLS_object.X_test, 'Lasso')
        
        MSE_values.append(MSE(OLS_object.Z_test, z_predict))
        R2_values.append(R2(OLS_object.Z_test, z_predict))
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(lambda_set, MSE_values, label='MSE')
    #ax.plot(lambda_set, R2_values, label='R2')
    
    ax.set_xlabel('Alpha')
    ax.set_ylabel('MSE')
    
    ax.legend()
    plt.title("Optimizing Lasso Terrain data")
    plt.show()
    
    
    return MSE_values, R2_values


def testAll(n,  row_arr, col_arr, z_arr):
    
    MSE_values_Lasso = []
    MSE_values_Ridge = []
    MSE_values_OLS = []
    
    l = 0.002
    alpha = 0.001
    for i in range(n):
        OLS_object = OLSClass(row_arr, col_arr, z_arr, i)
        OLS_object.X = design_poly_matrix(row_arr, col_arr, i)
        OLS_object.resample()
        
        beta_OLS = OLS_object.OLS(OLS_object.X_train, OLS_object.Z_train)
        beta_Ridge = OLS_object.Ridge(OLS_object.X_train, l, OLS_object.Z_train)
        beta_Lasso = OLS_object.Lasso(OLS_object.X_train, alpha, OLS_object.Z_train)
        
        z_predict_ols = OLS_object.predict(OLS_object.X_test, 'OLS')
        z_predict_ridge = OLS_object.predict(OLS_object.X_test, 'Ridge')
        z_predict_lasso = OLS_object.predict(OLS_object.X_test, 'Lasso')
        
        
        MSE_values_Lasso.append(MSE(OLS_object.Z_test, z_predict_lasso))
        MSE_values_Ridge.append(MSE(OLS_object.Z_test, z_predict_ridge))
        MSE_values_OLS.append(MSE(OLS_object.Z_test, z_predict_ols))
        
    
        
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(list(range(n)), MSE_values_OLS, label='OLS')
    ax.plot(list(range(n)), MSE_values_Ridge, label='Ridge')
    ax.plot(list(range(n)), MSE_values_Lasso, label='Lasso')
    ax.set_xlabel('degree of P')
    ax.set_ylabel('MSE')
    ax.legend()
    plt.title('Compare MSE')
    plt.show()
    return MSE_values_OLS, MSE_values_Ridge, MSE_values_Lasso

def testOLS_kfold_(n, row_arr, col_arr, z_arr, ax):
    
    bias_values = []
    var_values = []
    MSE_values = []
    R2_values = []
    degree = 11
    for i in range(2, n):
        OLS_object = OLSClass(row_arr, col_arr, z_arr, i)
        OLS_object.X = design_poly_matrix(row_arr, col_arr, degree)
        beta, MSE_score,R2_score =  OLS_object.apply_k_fold(4)
        MSE_values.append(min(MSE_score))
        R2_values.append(min(R2_score))
        
    
    ax.plot(list(range(2, n)), MSE_values, label='MSE-k-fold')
    ax.plot(list(range(2, n)), R2_values, label='R2-k-fold')
   
        
    return bias_values, var_values, MSE_values, R2_values

def testOLS_(n,  row_arr, col_arr, z_arr, ax):
    
    bias_values = []
    var_values = []
    MSE_values = []
    R2_values = []
    for i in range(1, n):
        OLS_object = OLSClass(row_arr, col_arr, z_arr, i)
        OLS_object.X = design_poly_matrix(row_arr, col_arr, i)
        OLS_object.resample()
        
        beta = OLS_object.OLS(OLS_object.X_train, OLS_object.Z_train)
        z_predict = OLS_object.predict(OLS_object.X_test)
        
        bias_values.append(bias(OLS_object.Z_test, z_predict))
        var_values.append(var(z_predict))
        MSE_values.append(MSE(OLS_object.Z_test, z_predict))
        R2_values.append(R2(OLS_object.Z_test, z_predict))

    ax.plot(list(range(1, n)), MSE_values, label='MSE-split')
    ax.plot(list(range(1, n)), R2_values, label='R2-split')        
        
    return bias_values, var_values, MSE_values, R2_values

def k_fold_vs_split():
    x, y, z =  franke_data()
    x1, y1, z1 = terrain_data(400, 200)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #testOLS_(15, x, y, z, ax)
    testOLS_(15, x1, y1, z1, ax)
    #testOLS_kfold_(12, x, y, z, ax)
    ax.set_xlabel('degree of P')
    ax.set_ylabel('MSE')
    plt.title('Finding the suitable polynomial degree(OLS)')
    ax.legend()
    plt.show()
    
def plot_ridge_test():
    x, y, z =  franke_data()
    x1, y1, z1 = terrain_data(400, 200)
    #testRidge(15, x1, y1, z1)
    #testLasso(15, x1, y1, z1)
    #testAll(20, x1, y1, z1)
    testLassoAlpha(11, x1, y1, z1)
    #k_fold_vs_split()
    
    
    
    
plot_ridge_test()
#plot_MSE(14)
    
#testOLS_kfold(5)
