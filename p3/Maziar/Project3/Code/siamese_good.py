## -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:01:12 2019

@author: Knut Engvik
"""

import os
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Dropout
from sklearn.model_selection import train_test_split
from keras import backend as K
import matplotlib.pyplot as plt

cwd = os.getcwd()
data = pd.read_csv("{}/../Data/ASV_table_mod.csv".format(cwd))
meta = pd.read_csv("{}/../Data/Metadata_table.tsv".format(cwd), delimiter=r"\s+")
dt = data.replace(0, pd.np.nan).dropna(axis=1, how='any').fillna(0).astype(int)


def normalize(x):
    return (x - np.amin(x)) / (np.amax(x) - np.amin(x))


def acc(y_true, y_pred):
    ones = K.ones_like(y_pred)
    return K.mean(
        K.equal(y_true, ones - K.clip(K.round(y_pred), 0, 1)), axis=-1)


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def euclidean_distance(vectors):
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def results(data, target, cutoff=0.5):
    total = 0
    hits = 0
    for d, t in zip(data, target):
        if d > cutoff:
            if t == 1:
                total += 1
                hits += 1
            else:
                total += 1
        else:
            if t == 1:
                total += 1
            else:
                total += 1
                hits += 1
    return hits / total


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 0.9
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def set_category(data, train=None, test=None):
    # The creation of bins/categories  aims to create to categories from
    # the data set, seperated by the mean. The digitize function
    # of numpy returns an array with 1,2,3....n as labels for each of n
    # bins. The min, max cut off are  chosen to be larger/smaller than
    # max min values of the data
    bins = np.array([0, data.mean(), 100])

    if train is None:
        temp = np.digitize(data, bins)
        return temp - 1
    train_labels = np.digitize(train, bins)
    test_labels = np.digitize(test, bins)
    return train_labels - 1, test_labels - 1


def make_anchored_pairs(data, target, test_data, test_target, anch):
    '''
    This function returns anchored of data points for comparison
    in siamese oneshot network
    '''
    # create lsits to store pairs and labels to return
    pairs = []
    labels = []
    test_pairs = []
    test_labels = []
    # Create all possible training pairs where one element is an anchor
    for i, a in enumerate(anch):
        x1 = a
        for index in range(len(data)):
            x2 = data[index]
            labels += [1] if target[index] == i else [0]
            pairs += [[x1, x2]]
            print(len(labels))
        for index in range(len(test_data)):
            x2 = test_data[index]
            test_labels += [1] if test_target[index] == i else [0]
            test_pairs += [[x1, x2]]

    return np.array(pairs), np.array(labels), np.array(test_pairs), np.array(
        test_labels)


data = normalize(dt)
ph = meta["pH"]
n2o = meta["N2O"]
temp = meta["Temperature"]
tp = meta["TP"]
anchors = {"TMP": (data.iloc[10], data.iloc[70])}
anchors["TP"] = (data.iloc[15], data.iloc[67])
# Some data formating
# y = make_categories(temp,2)
np.random.seed(2)
y = tp
X = data.to_numpy()
# y = to_categorical(y, num_classes=None)
# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
y_train_l, y_test_l = set_category(y, y_train, y_test)
# Make pairs and labels
# pairs_train, labels_train = make_pairs(X_train,y_train)
# pairs_test, labels_test = make_pairs(X_test,y_test)

pairs_train, labels_train, pairs_test, labels_test = make_anchored_pairs(
    X_train, y_train_l, X_test, y_test_l, anch=anchors["TP"])
# Set input size for network
input_size = len(X[0, :])


def base_network(input_shape):
    base_input = Input(shape=input_shape)
    x = Dense(50, activation="relu")(base_input)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(base_input)
    x = Dropout(0.1)(x)
    return Model(base_input, x)


def conv_base_network(input_shape):
    base_input = Input(shape=input_shape)
    x = Dense(100, activation="relu")(base_input)
    x = Dropout(0.1)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.1)(x)
    return Model(base_input, x)


def simple():
    base_model = base_network((input_size,))

    input1 = Input(shape=(input_size,))
    input2 = Input(shape=(input_size,))

    run1 = base_model(input1)
    run2 = base_model(input2)

    merge_layer = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [run1, run2])
    out_layer = Dense(1, activation="sigmoid")(merge_layer)
    siamese_model = Model(inputs=[input1, input2], outputs=out_layer)

    siamese_model.compile(loss=contrastive_loss,
                          optimizer="RMSprop", metrics=[acc])

    siamese_model.summary()

    history = siamese_model.fit([pairs_train[:, 0], pairs_train[:, 1]],
                                labels_train[:], validation_data=(
            [pairs_test[:, 0], pairs_test[:, 1]], labels_test[:]),
                                batch_size=5, epochs=200)
    return history


def conv():
    base_model = conv_base_network((input_size,))

    input1 = Input(shape=(input_size,))
    input2 = Input(shape=(input_size,))

    run1 = base_model(input1)
    run2 = base_model(input2)

    merge_layer = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [run1, run2])
    out_layer = Dense(1, activation="sigmoid")(merge_layer)
    siamese_model = Model(inputs=[input1, input2], outputs=out_layer)

    siamese_model.compile(loss=contrastive_loss,
                          optimizer="RMSprop", metrics=[acc])

    siamese_model.summary()

    history = siamese_model.fit([pairs_train[:, 0], pairs_train[:, 1]],
                                labels_train[:], validation_data=(
            [pairs_test[:, 0], pairs_test[:, 1]], labels_test[:]),
                                batch_size=5, epochs=200)
    return history, siamese_model


def visualize(history1, history2):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(history1.history['acc'])
    ax1.plot(history1.history['val_acc'])
    ax2.plot(history2.history['acc'])
    ax2.plot(history2.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    fig.savefig("{}plottest.png".format(path))


#simp = simple()
#comp, nn = conv()
#visualize(simp, comp)
