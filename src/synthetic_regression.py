import numpy as np
import matplotlib.pyplot as plt
import kernel, probability_estimators, tuning

def test():

    X = np.linspace(-10, 10, 100)
    Y = np.sin(np.abs(X)) / np.abs(X)

    # plt.plot(Y,'r')

    variance=10**(-2)

    A=np.zeros((X.shape[0]+1,X.shape[0]+1),float)
    B=np.zeros((X.shape[0],X.shape[0]),float)

    np.fill_diagonal(A,1)
    np.fill_diagonal(B,(1/variance))

    designMatrix= tuning.design_matrix(X.shape[0],"linear_spline",X,Y)
    mean, Sigma = probability_estimators.second_order_statistics(designMatrix,A,B,Y)

    gamas = np.ones(X.shape[0]+1, float) - np.diag(A)*np.diag(Sigma)
    max_iter=0

    deleted_indexes=[]
    while(True):

        A_old=np.copy(A)
        for j in range(1,A.shape[0]):

            A[j,j] = gamas[j]/( mean[j]**2 )

            if (A[j,j]>1000):
                deleted_indexes.append(j)

        if (len(deleted_indexes)>0):

            debug=0

            A= np.delete(A, deleted_indexes, 0)
            A= np.delete(A, deleted_indexes, 1)
            A_old= np.delete(A_old, deleted_indexes, 0)
            A_old= np.delete(A_old, deleted_indexes, 1)

            deleted_indexes[:] = [x - 1 for x in deleted_indexes]

            B= np.delete(B, deleted_indexes, 0)
            B= np.delete(B, deleted_indexes, 1)
            X= np.delete(Y, deleted_indexes, 0)
            Y= np.delete(Y, deleted_indexes, 0)

            deleted_indexes.clear()

            debug=0


        # Convergence criterion
        if (np.abs(np.trace(A) - np.trace(A_old)))< 0 and max_iter > 1:
            break

        max_iter+=1
        designMatrix = tuning.design_matrix(X.shape[0], "linear_spline", X, Y)
        mean, Sigma = probability_estimators.second_order_statistics(designMatrix, A, B, Y)
        gamas = np.ones(X.shape[0] + 1, float) - np.diag(A) * np.diag(Sigma)

    res=np.diag(A)
    debug=0

test()