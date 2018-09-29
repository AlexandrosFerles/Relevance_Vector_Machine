import numpy as np
import matplotlib.pyplot as plt
import kernel, probability_estimators, tuning
import matplotlib.pyplot as plt


def train(X, Y, kernel_mode):



    debug1=kernel.rbf(X,Y)
    t=debug1[0,0]
    debug=0
    # X = np.linspace(-10, 10, 100)
    # Y = np.sin(np.abs(X)) / np.abs(X)

    # plt.plot(Y,'r')

    variance=0.0001

    A=np.zeros((X.shape[0]+1,X.shape[0]+1),float)
    B=np.zeros((X.shape[0],X.shape[0]),float)

    np.fill_diagonal(A,1e-5)
    # A = A * np.random.normal(0, 6, A.shape[0])
    np.fill_diagonal(B,(1/variance))

    designMatrix= tuning.design_matrix(X.shape[0],kernel_mode,X)

    max_iter=0

    deleted_indexes=[]
    while(True):

        A_old=np.copy(A)

        mean, Sigma = probability_estimators.second_order_statistics(designMatrix, A, B, Y)
        gamas = np.ones(X.shape[0] + 1, float) - np.diag(A) * np.diag(Sigma)
        # gamas = 1 - np.diag(A) * np.diag(Sigma)

        for j in range(0,A.shape[0]):

            A[j,j] = gamas[j]/( mean[j]**2 )

            if (A[j,j]>1e9 and j>0):
                deleted_indexes.append(j)

        # B = np.zeros((X.shape[0], X.shape[0]), float)
        # np.fill_diagonal(B, (1 / tuning.common_noise_variance(Y, designMatrix, mean, Sigma, gamas)))

        if (len(deleted_indexes)>0):

            A= np.delete(A, deleted_indexes, 0)
            A= np.delete(A, deleted_indexes, 1)
            A_old= np.delete(A_old, deleted_indexes, 0)
            A_old= np.delete(A_old, deleted_indexes, 1)

            deleted_indexes[:] = [x - 1 for x in deleted_indexes]

            X= np.delete(X, deleted_indexes, 0)
            Y= np.delete(Y, deleted_indexes, 0)
            B= np.delete(B, deleted_indexes, 0)
            B= np.delete(B, deleted_indexes, 1)

            designMatrix = tuning.design_matrix(X.shape[0], kernel_mode, X)

            deleted_indexes.clear()

        # Convergence criterion
        if (np.abs(np.trace(A) - np.trace(A_old)))< 1e-3 or max_iter> 6000:
            break

        max_iter+=1

        print (max_iter)

    mean, _ = probability_estimators.second_order_statistics(designMatrix, A, B, Y)

    debugRes=np.diag(A)
    debug=0

    return X, mean

def predict(relevance_vectors, X, mean, kernel_choice):

    prediction =[]

    for xi in range(len(X)):

        phi_x=0
        for ri in range(len(relevance_vectors)):

            if kernel_choice=="gaussian":
                phi_x+=mean[ri+1]*(kernel.gaussian(X[xi], relevance_vectors[ri])) + mean[0]
            elif kernel_choice=="linear":
                phi_x+=mean[ri+1]*(kernel.linear_kernel(X[xi], relevance_vectors[ri])) + mean[0]
            elif kernel_choice=="polynomial":
                phi_x+=mean[ri+1]*(kernel.polynomial_kernel(X[xi], relevance_vectors[ri])) + mean[0]
            elif kernel_choice=="linear_spline":
                phi_x+=mean[ri+1]*(kernel.linear_spline(X[xi], relevance_vectors[ri])) + mean[0]
            elif kernel_choice=="rbf":
                phi_x+=mean[ri+1]*(kernel.rbf(X[xi], relevance_vectors[ri])) + mean[0]

        prediction.append(phi_x)

    return prediction

def test_ti_skata_kaname():

    X = np.asarray([[1, 2]])
    Y = np.asarray([[1, 2]])

    kernel_mode="rbf"

    relevance_vectors, mean = train(X,Y,kernel_mode)

    x = np.linspace(-10, 10, 1000)
    y = np.sin(np.abs(x)) / np.abs(x)

    y_predict = predict(relevance_vectors,x,mean,kernel_mode)

    plt.plot(x, y)
    plt.scatter(relevance_vectors, np.sin(np.abs(relevance_vectors)) / np.abs(relevance_vectors), marker='x', s=20)
    plt.scatter(x, y_predict, marker='.', s=2)
    plt.show()
    y_dif = np.abs(y-y_predict)

    debug=0

test_ti_skata_kaname()