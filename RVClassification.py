import numpy as np
import tuning, probability_estimators
import math
import kernel
import random
import matplotlib.pyplot as plt
import pylab

def sigmoid_function(y):
    return (1+math.exp(1)**(-y))**(-1)

def Relevance_Vector_Classification_Training(X,Y,kernel_mode):

    A=np.zeros((X.shape[0]+1,X.shape[0]+1),float)

    np.fill_diagonal(A,1e-5)

    sigmoids=[sigmoid_function(y)*(1-sigmoid_function(y)) for y in Y]
    B=np.diag(sigmoids)

    designMatrix = tuning.design_matrix(X.shape[0],kernel_mode,X)

    num_iter=0

    while(True):

        weightMaxPosteriori, Sigma =  probability_estimators.second_order_statistics(designMatrix)
        gamas = np.ones(X.shape[0]+1, float) - np.diag(A)*np.diag(Sigma)

        deleted_indexes = []
        while (True):

            A_old = np.copy(A)
            for j in range(1, A.shape[0]):

                A[j, j] = gamas[j] / (weightMaxPosteriori[j] ** 2)

                if (A[j, j] > 10e8):
                    deleted_indexes.append(j)

            if (len(deleted_indexes) > 0):

                A = np.delete(A, deleted_indexes, 0)
                A = np.delete(A, deleted_indexes, 1)
                A_old = np.delete(A_old, deleted_indexes, 0)
                A_old = np.delete(A_old, deleted_indexes, 1)

                deleted_indexes[:] = [x - 1 for x in deleted_indexes]

                X = np.delete(X, deleted_indexes, 0)
                Y = np.delete(Y, deleted_indexes, 0)
                B = np.delete(B, deleted_indexes, 0)
                B = np.delete(B, deleted_indexes, 1)

                designMatrix = tuning.design_matrix(X.shape[0], kernel_mode, X)

                deleted_indexes.clear()

        # Convergence criterion
        if (np.abs(np.trace(A) - np.trace(A_old)))< 1e-3 or num_iter>20000:
            break

        num_iter+=1
    weightMaxPosteriori, _ = probability_estimators.second_order_statistics(designMatrix, A, B, Y)

    return X, weightMaxPosteriori

def Relevance_Vector_Classification_Prediction (Xtest, relevance_vectors, weightMaxPosteriori, kernel_choice):

    Psum = 0
    res = []
    for xi in range(len(Xtest)):
        for ri in range (len(relevance_vectors)):
            if kernel_choice=="gaussian":
                Psum+=weightMaxPosteriori[ri+1]*(kernel.gaussian(Xtest[xi], relevance_vectors[ri])) + weightMaxPosteriori[0]
            elif kernel_choice=="linear":
                Psum+=weightMaxPosteriori[ri+1]*(kernel.linear_kernel(Xtest[xi], relevance_vectors[ri])) + weightMaxPosteriori[0]
            elif kernel_choice=="polynomial":
                Psum+=weightMaxPosteriori[ri+1]*(kernel.polynomial_kernel(Xtest[xi], relevance_vectors[ri])) + weightMaxPosteriori[0]
            elif kernel_choice=="linear_spline":
                Psum+=weightMaxPosteriori[ri+1]*(kernel.linear_spline(Xtest[xi], relevance_vectors[ri])) + weightMaxPosteriori[0]

        y = sigmoid_function(Psum)
            
        if y >0.5:
            res.append(1)
        elif y<= 0.5:
            res.append(0)

    return res

def test():

    ClassX_train = [(random.normalvariate(0.5, 0.8), random.normalvariate(0.5, 0.8), 1.0) for i in range(50)] + \
                   [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(50)]

    ClassO_train = [(random.normalvariate(1.5, 0.5), random.normalvariate(1.5, 0.5), 0) for i in range(50)]

    train_dataset1 = ClassX_train + ClassO_train


    pylab.hold(True)
    pylab.plot([p[0] for p in ClassX_train], [p[1] for p in ClassX_train], 'bo')
    pylab.plot([p[0] for p in ClassO_train], [p[1] for p in ClassO_train], 'ro')
    pylab.show()


    random.shuffle(train_dataset1)

    relevance_vectors, weightMaxPosteriori = Relevance_Vector_Classification_Training()
    debug=0

test()



























