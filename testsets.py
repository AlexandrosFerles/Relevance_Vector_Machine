import csv
import numpy as np
from sklearn import datasets
import sklearn

def load(dataset,n):

    if dataset == "synthetic_noise_free":

        X = np.linspace(-10,10,n)
        Y = np.sinc(np.abs(X))/np.abs(X)

        return X,Y

    if dataset == 'Friedman1':

        X, Y = sklearn.datasets.make_friedman1(n_samples=n, n_features=10, noise=0.0, random_state=None)
        # output based on: y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4] + noise * N(0, 1)

        return X, Y

    elif dataset == 'Friedman2':

        X, Y = sklearn.datasets.make_friedman2(n_samples=n, noise=0.0, random_state=None)
        # output based on: y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4] + noise * N(0, 1)

        return X, Y

    elif dataset == 'Friedman3':

        X, Y = sklearn.datasets.make_friedman3(n_samples=n, noise=0.0, random_state=None)
        # output based on: y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4] + noise * N(0, 1)

        return X, Y

    elif dataset == 'Boston':

        # Dictionary (506, 13)

        from sklearn.datasets import load_boston
        boston = load_boston()

        return boston

    elif dataset == 'Pima Indians':

        filename = 'pima-indians-diabetes.data.csv'
        raw_data = open(filename, 'rt')
        data = numpy.loadtxt(raw_data, delimiter=",")

        return raw_data, data

    elif dataset == 'USPS':

        return




