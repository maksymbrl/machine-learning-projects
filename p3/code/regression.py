# Use regression on predicted values to get smooth function of BCC Bray distance by CDOM

import numpy as np
import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import scale
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split


'''
Simple linear/logistic regression class -  code written as part of project 1, by Laurent et al
'''
class Regression:

    def GenerateMesh(self, x_min, x_max, y_min, y_max, step_x, step_y, log_transform=False):
        import numpy as np
        if log_transform==False:
            x = np.arange(x_min, x_max + step_x, step_x)  #Add range value to range end value due for last element being excluded from range
            y = np.arange(y_min, y_max + step_y, step_y)  #Add range value to range end value due for last element being excluded from range
        else:
            x = np.log10(np.arange(x_min, x_max + step_x, step_x)) + 1  #log1p transformation
            y = np.log10(np.arange(y_min, y_max + step_y, step_y)) + 1  #log1p transformation
        x, y = np.meshgrid(x,y)
        x = x.ravel()
        y = y.ravel()
        return x, y

    def GenerateDesignMatrix(self, x, y, order):
        X = np.c_[np.ones(len(x))]
        if order==0:
            return X
        elif order>0:
            X_str = "np.c_[np.ones(len(x)),"
            for i in range(1, order+1, 1):
                str_len = (i+1)*(2*i+1)-1
                poly_string = "x" * str_len
                poly_string_list = list(poly_string)
                for j in range(2*i-1, str_len, (2*(i+1)-1)):
                    poly_string_list[j] = ","
                for j in range(2*i, str_len, (2*(i+1)-1)):
                    poly_string_list[j] = " "
                for j in range(1, str_len, 2*i+1):
                    if i==2:
                        poly_string_list[j] = "*"
                    elif i>=3:
                        poly_string_list[j:(j+2*(i-1)):2] = (i-1)*"*"
                poly_string_list[-2:2*i-1:-(2*i+1)] = i*"y"
                for j in range(0, i, 1):
                    poly_string_list[-2*(1+j):2*i-1+j*(2*i+1):-(2*i+1)] = (i-j)*"y"
                poly_string = "".join(poly_string_list)
                X_str = X_str + " " + poly_string
            X = list(X_str)
            X[-1] = "]"
            X = eval("".join(X))
            colnames = list(X_str.replace(",", "").replace("np.c_[np.ones(len(x))", "").split(" "))
        return X, colnames

    """
    # Print design matrix formatted as pandas dataframe
    DesignMat_out = GenerateDesignMatrix(0.21, 3.83, 0.21, 3.83, 0.01, 0.01, 5) #3.83
    DesignMat = pd.DataFrame(DesignMat_out[0])
    DesignMat.columns = DesignMat_out[1]
    print(DesignMat)
    """

    def DesignMatrixList(self, x, y, order):
        X_degree_list = []
        for i in range(order+1):
            X_degree_list.append(self.GenerateDesignMatrix(x, y, i)[0])
        return X_degree_list

    def SVDinv(self, A):
        ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
        SVD is numerically more stable than the inversion algorithms provided by
        numpy and scipy.linalg at the cost of being slower.
        '''
        U, s, VT = np.linalg.svd(A)
        D = np.zeros((len(U),len(VT)))
        for i in range(0,len(VT)):
            D[i,i]=s[i]
        UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.pinv(D) #np.linalg.pinv instead of np.linalg.inv
        return np.matmul(V,np.matmul(invD,UT))

    def beta_SVD(self, X, z):
        A = np.transpose(X) @ X
        beta_SVD = self.SVDinv(A).dot(X.T).dot(z)
        #ztilde_SVD = X @ beta_SVD
        return beta_SVD

    def yPredictedSVD(self, X, z):  #Takes design matrix as X and a column vector of same number of rows as z
        z_pred_SVD = X @ self.beta_SVD(X, z)
        return z_pred_SVD

    def Fig_2_11(self, design_mat, degree, n_bootstraps):
        x_train, x_test, z_train, z_test = train_test_split(design_mat, z, test_size=0.3)
        z_pred = np.empty((z_test.shape[0], n_bootstraps))
        for i in range(n_bootstraps):
            x_, z_ = resample(x_train, z_train)
            z_pred[:,i] = x_test @ self.beta_SVD(x_, z_)
        z_test = z_test[:, np.newaxis]
        error = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
        bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )
        return error,bias,variance

    def Fig_2_11_no_resampling(self, design_mat, degree):  # Using SVD
        x_train, x_test, z_train, z_test = train_test_split(design_mat, z, test_size=0.3)
        z_pred = x_test @ self.beta_SVD(x_train, z_train)
        z_train_pred = x_train @ self.beta_SVD(x_train, z_train)
        error = np.mean((z_test - z_pred)**2)
        train_error = np.mean((z_train - z_train_pred)**2 )
        #print("z_pred")
        #print(pd.DataFrame(z_pred))
        #print("z_train_pred")
        #print(pd.DataFrame(z_train_pred))
        return error, train_error

    # With resampling
    def generate_error_bias_variance_with_resampling(self, mat_list, starting_degree, bootstrap):
        degree = starting_degree
        degree_list = []
        Fig_2_11_error_list= []
        Fig_2_11_bias_list = []
        Fig_2_11_variance_list = []
        for mat in mat_list:
            #print(pd.DataFrame(mat))
            #print(degree)
            #print("Start", degree, time.clock())
            Fig_2_11_result = self.Fig_2_11(mat, degree, bootstrap)
            Fig_2_11_error_list.append(Fig_2_11_result[0])
            Fig_2_11_bias_list.append(Fig_2_11_result[1])
            Fig_2_11_variance_list.append(Fig_2_11_result[2])
            degree_list.append(degree)
            degree = degree + 1
            #print(Fig_2_11_result)
            #print("End", time.clock())
        return Fig_2_11_error_list, Fig_2_11_bias_list, Fig_2_11_variance_list, degree_list

    # No resampling
    def generate_error_bias_variance_without_resampling(self, mat_list, starting_degree):
        degree = starting_degree
        degree_list = []
        Fig_2_11_error_list= []
        Fig_2_11_train_error_list = []
        for mat in mat_list:
            #print(pd.DataFrame(mat))
            #print(degree)
            #print("Start", degree, time.clock())
            Fig_2_11_result = self.Fig_2_11_no_resampling(mat, degree)
            Fig_2_11_error_list.append(Fig_2_11_result[0])
            Fig_2_11_train_error_list.append(Fig_2_11_result[1])
            degree_list.append(degree)
            degree = degree + 1
            #print(Fig_2_11_result)
            #print("End", time.clock())
        return Fig_2_11_error_list, Fig_2_11_train_error_list, degree_list

    # Plot error, bias and variance as a function of model complexity
    def ebv_by_model_complexity(self, metrics):
        plt.plot(metrics[3], metrics[0])
        plt.plot(metrics[3], metrics[1])
        plt.plot(metrics[3], metrics[2])
        plt.scatter(metrics[3], metrics[0], label='error')
        plt.scatter(metrics[3], metrics[1], label='bias')
        plt.scatter(metrics[3], metrics[2], label='variance')
        plt.xlabel("Model complexity (polynomial order)")
        plt.xticks(np.arange(1, len(metrics[3])+1, 1))
        #plt.yticks(np.arange(0, 0.2, 0.05))
        plt.yscale('log')
        plt.legend()
        plt.show()

    # Plot training and sample errors as functions of model complexity
    def training_vs_test(self, metrics):
        plt.plot(metrics[2], metrics[0])
        plt.plot(metrics[2], metrics[1])
        plt.scatter(metrics[2], metrics[0], label='Test sample')
        plt.scatter(metrics[2], metrics[1], label='Training sample')
        plt.xlabel("Model complexity (polynomial order)")
        plt.xticks(np.arange(1, len(metrics[2])+1, 1))
        #plt.yticks(np.arange(0, 0.2, 0.05))
        plt.yscale('log')
        plt.legend()
        plt.show()




