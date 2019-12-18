import numpy as np
import matplotlib.pyplot as plt
import time

from Project2.Code.project1_code import OLSClass
from Project2.Code.project1_code import design_poly_matrix
from Project2.Code.multilayer_perceptron import NeuralNetwork, MSE
from sklearn.model_selection import train_test_split


def franke_data():
    def franke_function(x, y):
        term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
        term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
        term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
        term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
        return term1 + term2 + term3 + term4

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


def franke_NN():
    x, y, z = franke_data()
    X = np.c_[x, y]
    Z = z.reshape(-1, 1)

    trainingShare = 0.8
    XTrain, XTest, yTrain, yTest = train_test_split(X, Z, train_size=trainingShare, test_size=1 - trainingShare)

    NN_MSE = []
    duration = []
    for i in range(10):
        start_time = time.time()
        c = 0.1 * (i + 1)
        print(i)
        neural_network = NeuralNetwork(0.01, 50)
        neural_network.new_layer({'input_size': 2, 'number_of_nodes': 3, 'activation_function': 'sigmoid'})
        neural_network.new_layer({'input_size': 3, 'number_of_nodes': 2, 'activation_function': 'sigmoid'})
        neural_network.new_layer({'input_size': 2, 'number_of_nodes': 1, 'activation_function': 'tanh'})

        neural_network.train(XTrain[:int(c * len(XTrain)) - 1], yTrain[:int(c * len(XTrain)) - 1], mse_off=True)

        y_predict = neural_network.forward(XTest)
        end_time = time.time()

        duration.append(end_time - start_time)
        NN_MSE.append(np.mean(np.square(y_predict - yTest)))

    return NN_MSE, duration


def franke_OLS():
    x, y, z = franke_data()
    OLS_MSE = []
    duration = []
    for i in range(10):
        start_time = time.time()
        c = 0.1 * (i + 1)
        size_ = int(c * len(x)) - 1
        print(i, x[:size_].shape)
        OLS_object = OLSClass(x[:size_], y[:size_], z[:size_], 10)

        OLS_object.X = design_poly_matrix(x[:size_], y[:size_], 10)
        OLS_object.resample()

        beta = OLS_object.OLS(OLS_object.X_train, OLS_object.Z_train)
        z_predict = OLS_object.predict(OLS_object.X_test)
        end_time = time.time()
        duration.append(end_time - start_time)
        OLS_MSE.append(MSE(OLS_object.Z_test, z_predict))

    return OLS_MSE, duration


def compare_OLS_NN():
    OLS_MSE, OLS_time = franke_OLS()
    NN_MSE, NN_time = franke_NN()

    plt.plot(NN_MSE, label='MLP')
    plt.plot(OLS_MSE, label='OLS')
    plt.title('OLS vs MLP')
    plt.xlabel('Training Size')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

    plt.plot(NN_time, label='MLP')
    plt.plot(OLS_time, label='OLS')
    plt.title('OLS vs MLP')
    plt.xlabel('Training Size')
    plt.ylabel('Execution time')
    plt.legend()
    plt.show()


def main():
    compare_OLS_NN()


if __name__ == '__main__':
    main()
