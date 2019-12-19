# XGboost with scikitlearn
# https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
# https://xgboost.readthedocs.io/en/latest/python/python_api.html
from numpy import loadtxt
from xgboost import XGBRegressor  #scikit-learn API for XGBoost regression
from xgboost import XGBRFRegressor  #scikit-learn API for XGBoost random forest regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import scale
from matplotlib.ticker import LinearLocator, FormatStrFormatter

'''
CLass To implement XGBoost
'''
class XGBoosting:

    def __init__(self, *args):
        pass

    def RunModel(self, *args):
        X_train = args[0]
        X_test  = args[1]
        y_train = args[2]
        y_test  = args[3]
        X_CDOM  = args[4]
        X_CDOM_diag_mesh = args[5]
        CDOM    = args[6]
        CDOM_sorted = args[7]
        outputPath = args[8]
        #!!!Implement Xavier initialization!!!
        # fit model no training data
        model = XGBRegressor(objective='reg:squarederror')
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
                  eval_metric='rmse', early_stopping_rounds=100, verbose=False)
        #print(model)
        return model
