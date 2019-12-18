
import numpy as np

from sklearn import linear_model

from sklearn.model_selection import train_test_split


def design_poly_matrix(x, y, n):
    N = len(x)
    l = int((n + 1) * (n + 2) / 2)
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = x ** (i - k) * y ** k

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
        Calculating the R2 score:
        x, and x_ are both numpy arrays
        the output will be a double
    """
    x_avg = np.average(x)
    return 1. - MSE(x, x_) / MSE(x, x_avg)


def R_squared(pred, actual):
    r_sq = 1 - mse(pred, actual) / mse(actual.mean(), actual)
    return r_sq


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

    def predict(self, x, model='OLS'):
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
        print("length of the whole data ", n)
        # making random numbers
        arr = np.arange(n)
        # making the set even
        np.random.shuffle(arr)
        print(len(arr))
        # make the lenght dividable by k:
        print(len(arr))
        k_fold = []
        for i in range(0, int(n / k) * k, int(n / k)):
            print(i, i + int(n / k))
            k_fold.append([])
            k_fold[-1] = arr[i:i + int(n / k)]
            print(len(k_fold[-1]))

        return k_fold

    def OLS(self, X, expected_value):
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(expected_value)
        # beta = np.linalg.lstsq(X, expected_value)[0]
        self.beta_OLS = beta
        return beta

    def Ridge(self, X, l, expected_value):
        # X = design_poly_matrix(x, y, degree)
        beta = np.linalg.inv(X.T.dot(X) + np.eye(np.shape(X)[1]) * l).dot(X.T).dot(expected_value)
        self.beta_Ridge = beta
        return beta

    def Lasso(self, X, a, expected_value):
        # self.X = design_poly_matrix(x, y, self.degree)
        clf = linear_model.Lasso(alpha=a, fit_intercept=False, max_iter=1000)
        clf.fit(X, expected_value)
        self.beta_Lasso = clf.coef_
        self.clf_Lasso = clf
        return clf.coef_

    def apply_k_fold(self, k):
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
                        # print ("making testing set")
                        tmp_x_test.append(self.X[ind])
                        tmp_z_test.append(self.z[ind])
                    else:
                        # print("making training test")
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
