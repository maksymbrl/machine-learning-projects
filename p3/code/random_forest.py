import os, sys
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as MSE

#from Project3.Code.siamese_good import X_train as X_train_siamese
#from Project3.Code.siamese_good import X_test as X_test_siamese
#from Project3.Code.siamese_good import y_train_l as y_train_siamese
#from Project3.Code.siamese_good import y_test_l as y_test_siamese
#from Project3.Code.siamese_good import set_category

import data_processing

# Trying to set the seed
class RandomForest:
    def __init__(self):
        pass

    def accuracy(self, x_1, x_2):
        conf = [[0, 0], [0, 0]]
        for i, j in zip(x_1, x_2):
            conf[i][j] += 1

        print(conf, (conf[0][0] + conf[1][1]) / len(x_1))


    def get_std(self, pandas_object):
        metadata_std = [x.std() for x in pandas_object.values.T]

        return metadata_std


    def get_corr(self, pandas_object, list_of_features=[]):
        if len(list_of_features) == 0:
            list_of_features = pandas_object.columns

        corr_list = []

        for i in list_of_features:
            corr_list.append([])
            for j in list_of_features:
                corr_list[-1].append(pandas_object[i].corr(pandas_object[j]))

        return corr_list


    def read_data(self, pop_bool=True, metadata_bool=True):
        data = data_processing.DataProcessing()
        # Reading file into data frame
        #cwd = os.getcwd()
        population_path = data.url_ASV #cwd + '/../Data/ASV_table.tsv'
        metadata_path = data.url_metadata #cwd + '/../Data/Metadata_table.tsv'
        """
        df.columns (identifier)
        df.values (population size)
        
        population_size.shape -> (72, 14991)
        """
        population_size = pd.read_csv(population_path, delimiter='\s+', encoding='utf-8')

        if pop_bool:
            # find the non-zero bioms
            population_to_drop = [x for x in population_size.columns if population_size.get(x).min() == 0]
            population_size = population_size.drop(population_to_drop, axis=1)
        """
        df.columns (properties)
        df.values (values)
        
        metadata.shape -> (71, 41)
        """
        metadata = pd.read_csv(metadata_path, delimiter='\s+', encoding='utf-8')

        # l = ["Latitude", "Longitude", "Altitude", "Area",
        if metadata_bool:
            l = ["Temperature", "Secchi", "O2", "CH4", "pH", "TIC",
                 "SiO2", "KdPAR"]

            toDrop = [x for x in metadata.columns if x not in l]
            metadata = metadata.drop(toDrop, axis=1)

        return population_size, metadata


    def prepare_data(self, population_data, metadata, test_size, seed):
        x_values = np.array(population_data.values)
        y_values = np.array(metadata.values)

        train_x, test_x, train_y, test_y = train_test_split(x_values, y_values, test_size=test_size)

        print(train_x.shape, train_y.shape)
        print(test_x.shape, test_y.shape)

        predictions = np.zeros((test_y.shape[1], test_y.shape[0]))
        ML_ = []
        for i in range(len(metadata.columns)):
            print(i)
            rf = RandomForestRegressor(n_estimators=100, random_state=seed)
            rf.fit(train_x, train_y.T[i])

            ML_.append(rf)
            predictions[i] = rf.predict(test_x)

        print(predictions.shape)

        # errors = abs(pred - test_y.T[i])
        # err = round(np.mean(errors), 2)

        for i in range(len(metadata.columns)):
            print(metadata.columns[i], MSE(test_y.T[i], predictions[i]))

        return predictions, test_y, ML_
        # print("observation", MSE(predictions.T[0], test_y[0]))


    def get_bioms(self):
        """
        To get the bioms that are in the mid 20% (to be or not to be)
        :return:
        """
        #cwd = os.getcwd()
        #population_path = cwd + '/../Data/ASV_table.tsv'
        data = data_processing.DataProcessing()
        population_path = data.url_ASV

        pop_bioms = pd.read_csv(population_path, delimiter='\s+', encoding='utf-8')

        to_keep = []
        for i in pop_bioms.columns:
            c = 0
            for j in pop_bioms.get(i):
                if j > 2:
                    c += 1
            if 0.6 > c / 72 > 0.4:
                to_keep.append(i)

        to_drop = [x for x in pop_bioms.columns if x not in to_keep]
        pop_bioms = pop_bioms.drop(to_drop, axis=1)
        return pop_bioms


    def predict_exist(self):
        bioms = self.get_bioms()
        _, metadata = self.read_data()

        bioms_array = (np.array(bioms.values) > 0) * 1
        metadata_array = np.array(metadata.values)

        train_x, test_x, train_y, test_y = train_test_split(metadata_array, bioms_array, test_size=0.30)

        print("train_x size:", train_x.shape, " train_y size:", train_y.shape, train_y.T[0].shape)

        clfs = []
        scores_test = []
        scores_train = []
        for j in range(2, 10):
            clfs.append([])
            scores_test.append([])
            scores_train.append([])
            for i in range(bioms_array.shape[1]):
                print("n_estimators : ", j * 10)
                clf = RandomForestClassifier(max_depth=j, random_state=50, n_estimators=40)
                clf.fit(train_x, train_y.T[i])
                # rint(clf.predict(test_x))
                scores_test[-1].append(clf.score(test_x, test_y.T[i]))
                scores_train[-1].append(clf.score(train_x, train_y.T[i]))
                # print(scores[-1][-1])
                # print(accuracy(test_y.T[i], clf.predict(test_x)))
                clfs[-1].append(clf)

        return clfs, scores_test, scores_train


    def predict_t(self, X_train, X_test, y_train, y_test, seed):
        _, metadata = self.read_data()
        #y_train_siamese, y_test_siamese = 0, 0
        #X_train_siamese, X_test_siamese = 0, 0
        #if X_train is None or X_test is None:
        #    train_y, test_y = y_train_siamese, y_test_siamese
        #    train_x, test_x = X_train_siamese, X_test_siamese
        #else:
        train_y, test_y = y_train, y_test
        train_x, test_x = X_train, X_test

        print("train_x size:", train_x.shape, " train_y size:", train_y.shape, train_y.T[0].shape)

        clfs = []
        scores_test = []
        scores_train = []
        for j in range(2, 10):
            print("max_depth : ", j * 10)
            clf = RandomForestClassifier(max_depth=j, random_state=seed, n_estimators=100)
            clf.fit(train_x, train_y.T)
            # rint(clf.predict(test_x))
            scores_test.append(clf.score(test_x, test_y.T))
            scores_train.append(clf.score(train_x, train_y.T))
            # print(scores[-1][-1])
            # print(accuracy(test_y.T[i], clf.predict(test_x)))
            clfs.append(clf)

        return clfs, scores_test, scores_train


    def predict_all_metadata(self, population_size, metadata, ML_):
        x_values = np.array(population_size.values)
        y_value = np.array(metadata.values)

        predictions = np.zeros((y_value.shape[1], y_value.shape[0]))

        for i in range(len(metadata.columns)):
            predictions[i] = ML_[i].predict(x_values)

        print(predictions.shape)

        # errors = abs(pred - test_y.T[i])
        # err = round(np.mean(errors), 2)

        return predictions


