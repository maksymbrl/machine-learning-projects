import os
import random

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from Project2.Code.logistic_regression import LogisticRegression as LR


# Trying to set the seed
from Project2.Code.multilayer_perceptron import NeuralNetwork as NN

np.random.seed(0)

random.seed(0)

# Reading file into data frame
cwd = os.getcwd()
filename = cwd + '/../data/credit_card_data.xls'
nanDict = {}
df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)

df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)
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

# Categorical variables to one-hot's
onehotencoder = OneHotEncoder(categories="auto")

X = ColumnTransformer(
    [("", onehotencoder, [3]), ],
    remainder="passthrough"
).fit_transform(X)

y.shape

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
Y_train_onehot, Y_test_onehot = onehotencoder.fit_transform(yTrain).toarray(), onehotencoder.fit_transform(yTest).toarray()

lambdas = np.logspace(-5, 7, 13)
parameters = [{'C': 1. / lambdas, "solver": ["lbfgs"]}]  # *len(parameters)}]
scoring = ['accuracy', 'roc_auc']
logReg = LogisticRegression()
# TRain
logReg.fit(XTrain, yTrain)
myLR = LR()
#myLR.train(XTrain, yTrain, XTest, yTest)

print(logReg.score(XTest, yTest))

# myLR.train(XTrain, yTrain.reshape(-1, 1), XTest, yTest.reshape(-1, 1))
print("hello")
# gridSearch = GridSearchCV(logReg, parameters, cv=5, scoring=scoring, refit='roc_auc')

mlp2 = NN(0.1, 100)
mlp2.new_layer({'input_size': 25, 'number_of_nodes': 20, 'activation_function': 'tanh'})
mlp2.new_layer({'input_size': 20, 'number_of_nodes': 15, 'activation_function': 'sigmoid'})
mlp2.new_layer({'input_size': 15, 'number_of_nodes': 5, 'activation_function': 'sigmoid'})
mlp2.new_layer({'input_size': 5, 'number_of_nodes': 1, 'activation_function': 'tanh'})
mlp2.train(XTrain, yTrain)
# mse = mlp.train(XTrain, yTrain, 0.01, 100)

plt.plot(mlp2.mse_score)
# plt.plot(mse)
plt.title('Changes in MSE')
plt.xlabel('Epoch (every 10th)')
plt.ylabel('MSE')
plt.show()
