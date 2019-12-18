import numpy as np
from sklearn.utils import shuffle


def activation_sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def der_activation_sigmoid(values):
    return values * (1 - values)


def activation_linear(values):
    return values


def der_activation_linear(values):
    return values


def activation_tanh(values):
    return np.tanh(values)


def der_activation_tanh(values):
    return 1. - values ** 2


def MSE(x, x_):
    """
        Calculating the Mean Square Error.
        Argument (numpy array x, and \tilde{x})

        here x_ can either an array, or a constant.

        returns a double
    """
    return np.mean(np.square(x - x_))


def R2(x_, x):
    # expected, predicted
    r2 = 1 - MSE(x, x_) / MSE(np.average(x_), x)
    return r2


class Layer:
    """
    Each Layer works as a seperate object, it has the number of inputs,
    and the number of output edges.

    The activation function for each Layer can be individually selected.

    Since this class is meant to be a private class, the main program can use
    the implemented new_layer function to feed the design of the neural network,
    using dictionaries as arguments.

    Each Layer will keep track of its error rate, and delta (which are used
    to help with finding the required changes in the same iteration)
    """

    def __init__(self, input_size, node_size, activation='sigmoid'):
        # Statistics
        self.result = None
        self.error = None
        self.delta = None
        self.last_value = None

        # Features
        self.number_of_inputs = input_size
        self.number_of_nodes = node_size
        self.activation = activation

        # Values
        self.bias = np.random.rand(node_size)
        self.betas = np.random.rand(input_size, node_size)

    def forward(self, values):
        # gets the input, applied the weights, and run the activation
        tmp = np.dot(values, self.betas) + self.bias
        self.result = self.activate(tmp)
        return self.result

    def activate(self, value):
        """
        :param value: the result of the feed forward
        :return: activated result, based of the chosen activation function

        TODO: Check for lower cases
        """
        # sigmoid:
        if 'sig' in self.activation:
            return activation_sigmoid(value)

        # tanh
        elif 'tanh' in self.activation:
            return activation_tanh(value)

        # linear
        else:
            return value

    def backward(self, value):
        """
        This will apply the derivative of the selected activation function
        """

        # sigmoid:
        if 'sig' in self.activation:
            return der_activation_sigmoid(value)

        # tanh
        elif 'tanh' in self.activation:
            return der_activation_tanh(value)

        # linear
        else:
            return value


class NeuralNetwork:
    def __init__(self, learning_rate=0.01, max_iter=100, epsilon=0):
        # design:
        self.R2_score = []
        self.mse_score = []
        self.layers = []

        # early stopping
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.eta = learning_rate

    def new_layer(self, design):
        """
        :param design: is a dictionary of type
            {'input_size': number of inputs,
            'number_of_nodes': number of inputs for the next layer,
            'activation_function': activation function (sigmoid, linear, tanh)
            }
        :return: None
        """
        self.layers.append(Layer(design['input_size'], design['number_of_nodes'], design['activation_function']))

    def forward(self, x_train):
        """
        Uses the forward function in each Layer to find the final result

        :param x_train: X values
        :return: the result of the NeuralNetwork
        """
        next_input = x_train
        for layer in self.layers:
            next_input = layer.forward(next_input)

        return next_input

    def predict_class(self, x):
        """
        Should only be used if the NeuralNetwork is used for classification problems, not regression
        :param x: input
        :return: index of the most probable outcome
        """
        result = self.forward(x)

        if result.ndim == 1:
            return np.argmax(result)

        else:
            return np.argmax(result, axis=1)

    def backward(self, x_train, y_train):
        """
        :param x_train: inputs of the training set
        :param y_train: expected output
        :return: None

        Uses the backward function in each Layer to simplify the process
        """
        result = self.forward(x_train)
        number_layers = len(self.layers)
        for i in reversed(range(number_layers)):
            if self.layers[i] == self.layers[-1]:
                self.layers[i].error = y_train - result
                self.layers[i].delta = self.layers[i].error * self.layers[i].backward(result)

            else:
                # weighted error
                self.layers[i].error = np.dot(self.layers[i + 1].betas, self.layers[i + 1].delta)
                self.layers[i].delta = self.layers[i].error * self.layers[i].backward(self.layers[i].result)

        for i in range(number_layers):
            if i == 0:
                tmp_x = x_train
            else:
                tmp_x = self.layers[i - 1].result

            tmp_x = np.atleast_2d(tmp_x)
            self.layers[i].betas += self.layers[i].delta * tmp_x.T * self.eta

    def train(self, x_train, y_train, mse_off=False):
        """
        :param x_train: input values
        :param y_train: expected outcome
        :return: None

        Shuffles the data, selects 20% of the training data for validation.
        The validation set is used to find the MSE score.

        If the tolerance rate is set, it can end before going through all the iterations.
        """
        for i in range(self.max_iter):
            tmp_x_train, tmp_y_train = shuffle(x_train, y_train, random_state=0)
            n_train = len(tmp_x_train)
            n_valid = int(n_train / 5)
            x_valid, y_valid = tmp_x_train[:n_valid], tmp_y_train[:n_valid]
            for x in range(n_train):
                self.backward(tmp_x_train[x], tmp_y_train[x])

            if not mse_off:
                valid_result = self.forward(x_valid)
                self.mse_score.append(MSE(y_valid, valid_result))
                self.R2_score.append(R2(y_valid, valid_result))

                if i > 10 and abs(self.mse_score[-1] - self.mse_score[-2]) <= self.epsilon:
                    break

    def accuracy(self, x_test, y_test):
        result = self.forward(x_test) > 0.5
        return np.sum(result == y_test) / len(y_test)
