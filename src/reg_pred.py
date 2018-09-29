import numpy as np
import matplotlib.pyplot as plt
import kernel, probability_estimators, tuning


def test():
    X = np.linspace(-10, 10, 100)
    Y = np.sin(np.abs(X)) / np.abs(X)

    # plt.plot(Y,'r')

    variance = 0.01

    A = np.zeros((X.shape[0] + 1, X.shape[0] + 1), float)
    B = np.zeros((X.shape[0], X.shape[0]), float)

    np.fill_diagonal(A, 1)
    np.fill_diagonal(B, (1 / variance))

    designMatrix = tuning.design_matrix(X.shape[0], "linear_spline", X)
    mean, Sigma = probability_estimators.second_order_statistics(designMatrix, A, B, Y)

    gamas = np.ones(X.shape[0] + 1, float) - np.diag(A) * np.diag(Sigma)
    max_iter = 0

    deleted_indexes = []
    while (True):

        A_old = np.copy(A)

        for j in range(1, A.shape[0]):

            A[j, j] = gamas[j] / (mean[j] ** 2)

            if (A[j, j] > 1e9):
                deleted_indexes.append(j)

        debug = 0

        if (len(deleted_indexes) > 0):
            debug = 0

            A = np.delete(A, deleted_indexes, 0)
            A = np.delete(A, deleted_indexes, 1)
            A_old = np.delete(A_old, deleted_indexes, 0)
            A_old = np.delete(A_old, deleted_indexes, 1)

            deleted_indexes[:] = [x - 1 for x in deleted_indexes]

            B = np.delete(B, deleted_indexes, 0)
            B = np.delete(B, deleted_indexes, 1)
            X = np.delete(Y, deleted_indexes, 0)
            Y = np.delete(Y, deleted_indexes, 0)

            deleted_indexes.clear()

            debug = 0

        # Covergence criterion suggested from RVM+Explained
        # which can be found on the Literature folder
        if (np.abs(np.trace(A) - np.trace(A_old))) and max_iter > 1:
            break

        max_iter += 1
        designMatrix = tuning.design_matrix(X.shape[0], "linear_spline", X)
        mean, Sigma = probability_estimators.second_order_statistics(designMatrix, A, B, Y)
        gamas = np.ones(X.shape[0] + 1, float) - np.diag(A) * np.diag(Sigma)

    res = np.diag(A)
    res2 = np.diag(A_old)

    debug = 0

    X_new = np.random.uniform(-10, 10, 100)
    Y_true = np.sin(np.abs(X_new)) / np.abs(X_new)
    Y_new = []
    MSE=[]

    for i in range(0, 100):
        y = 0
        for j in range(1, 5):

            Y_new.append(np.sum(y) + res[0])
            MSE.append(np.sqrt(Y_true[i] - Y_new[i]))

    plt.plot(X_new, Y_new)
    plt.show()

    debug=0

test()