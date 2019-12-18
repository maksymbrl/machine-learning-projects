import os

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sn

from Project2.Code.multilayer_perceptron import NeuralNetwork
from Project2.Code.logistic_regression import LogisticRegression


def run_logistic_regression():
    XTrain, yTrain, XTest, yTest = credit_card_data()

    LR = LogisticRegression(max_iter=100, lr=0.1)
    LR.train(XTrain, yTrain, XTest, yTest)

    lr_test_result = LR.predict(XTest)

    conf_matrix = [[0, 0], [0, 0]]
    for i in range(len(lr_test_result)):
        conf_matrix[int(lr_test_result[i][0])][yTest[i][0]] += 1

    df_cm = pd.DataFrame(conf_matrix, range(2),
                         range(2))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.title("Confusion matrix LR")
    plt.show()
    print("accuracy (LR): ",
          (conf_matrix[0][0] + conf_matrix[1][1]) * 100.0 / (sum(conf_matrix[0]) + sum(conf_matrix[1])))
    return LR.mse_score


def run_mlp():
    # defining the mlp object
    XTrain, yTrain, XTest, yTest = credit_card_data()
    mlp2 = NeuralNetwork(0.1, 100)

    # Design the network
    mlp2.new_layer({'input_size': 25, 'number_of_nodes': 20, 'activation_function': 'tanh'})
    mlp2.new_layer({'input_size': 20, 'number_of_nodes': 15, 'activation_function': 'sigmoid'})
    mlp2.new_layer({'input_size': 15, 'number_of_nodes': 5, 'activation_function': 'sigmoid'})
    mlp2.new_layer({'input_size': 5, 'number_of_nodes': 1, 'activation_function': 'tanh'})

    # initiate and train the system
    mlp2.train(XTrain, yTrain)

    mlp_test_result = mlp2.forward(XTest) > 0.5

    conf_matrix = [[0, 0], [0, 0]]
    for i in range(len(mlp_test_result)):
        conf_matrix[int(mlp_test_result[i][0])][yTest[i][0]] += 1

    df_cm = pd.DataFrame(conf_matrix, range(2),
                         range(2))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.title("Confusion matrix MLP")
    plt.show()
    print("accuracy (MLP): ", (conf_matrix[0][0] + conf_matrix[1][1])*100.0/(sum(conf_matrix[0]) + sum(conf_matrix[1])))
    return mlp2.mse_score


def pre_processing(df):
    # Remove instances with zeros only for past bill statements or paid amounts
    print(df.shape)
    df = df.drop(df[(df.BILL_AMT1 == 0) &
                    (df.BILL_AMT2 == 0) &
                    (df.BILL_AMT3 == 0) &
                    (df.BILL_AMT4 == 0) &
                    (df.BILL_AMT5 == 0) &
                    (df.BILL_AMT6 == 0)].index)

    df = df.drop(df[(df.PAY_AMT1 == 0) &
                    (df.PAY_AMT2 == 0) &
                    (df.PAY_AMT3 == 0) &
                    (df.PAY_AMT4 == 0) &
                    (df.PAY_AMT5 == 0) &
                    (df.PAY_AMT6 == 0)].index)

    # Checking SEX
    df = df.drop(df[(df.SEX != 1) &
                    (df.SEX != 2)].index)

    # Checking EDUCATION
    df = df.drop(df[(df.EDUCATION != 1) &
                    (df.EDUCATION != 2) &
                    (df.EDUCATION != 3) &
                    (df.EDUCATION != 4)].index)

    # Checking MARRIAGE
    df = df.drop(df[(df.MARRIAGE != 1) &
                    (df.MARRIAGE != 2) &
                    (df.MARRIAGE != 3)].index)

    # Checking defaultPaymentNextMonth
    df = df.drop(df[(df.defaultPaymentNextMonth != 0) &
                    (df.defaultPaymentNextMonth != 1)].index)

    print(df.shape)

    # Features and targets
    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

    return X, y


def credit_card_data():
    cwd = os.getcwd()
    filename = cwd + '/../../data/credit_card_data.xls'
    nanDict = {}
    df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)

    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)
    X, y = pre_processing(df)

    # Categorical variables to one-hot's
    onehotencoder = OneHotEncoder(categories="auto")

    X = ColumnTransformer(
        [("", onehotencoder, [3]), ],
        remainder="passthrough"
    ).fit_transform(X)

    # Train-test split
    trainingShare = 0.5
    seed = 1
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=trainingShare, test_size=1 - trainingShare,
                                                    random_state=seed)

    # Input Scaling
    sc = StandardScaler()
    XTrain = sc.fit_transform(XTrain)
    XTest = sc.transform(XTest)

    # One-hot's of the target vector
    Y_train_onehot, Y_test_onehot = onehotencoder.fit_transform(yTrain).toarray(), onehotencoder.fit_transform(
        yTest).toarray()




    return XTrain, yTrain, XTest, yTest


def main():
    mlp_mse = run_mlp()
    #lr_mse = run_logistic_regression()

    #plt.plot(mlp_mse, label='MLP')
    #plt.plot(lr_mse, label='LR')
    #plt.title('Changes in MSE')
    #plt.xlabel('iteration')
    #plt.ylabel('MSE')
    #plt.legend()
    #plt.show()


if __name__ == '__main__':
    main()
