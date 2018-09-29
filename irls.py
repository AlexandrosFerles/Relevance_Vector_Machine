# Iterative Reweighted Least Squares Algorithm
# The following code is somewhat close to the one provided in
# https://github.com/aehaynes/IRLS/blob/master/irls.py

import numpy as np
import probability_estimators as prob
import math
import tuning
from math import pow as power

def almost_sigmoid(x):

    try:
        ret  = 1 / (1 + math.exp(-x))
    except OverflowError:
        debug=0

    if (ret==1):
      # we want to use the  logarithm of
      # the sigmoid function so we deviate from the value 1
      return 1-1e-10
    elif (ret==0):
      return 1e10
    else:
      return ret


# Theory provided by http://www.cedar.buffalo.edu/~srihari/CSE574/Chap4/4.3.3-IRLS.pdf
def iterative_reweighted_least_squares_algorithm(designMatrix, t, A, B, w, max_iter=10, tolerance=10e-5):

    temp_y = np.dot(designMatrix.T,w)
    y = np.asarray( [almost_sigmoid(temp_y[i]) for i in range(temp_y.shape[0])] )
    gradient_objective = np.dot(designMatrix.T, (y - t)) - np.dot(A, w)

    for _ in range(max_iter):

        Sigma = prob.sigma_estimator(designMatrix, A, B)
        weightMaxPosteriori = w - np.dot(Sigma, gradient_objective)

        y = np.dot(designMatrix.T, weightMaxPosteriori )

        sigmoid_estimations = [almost_sigmoid(y[j]) for j in range(y.shape[0])]
        B=np.diagflat(sigmoid_estimations)

        gradient_objective = np.dot(designMatrix.T, (y - t)) - np.dot(A, weightMaxPosteriori)
        # convergence criterion
        if np.sum(gradient_objective<tolerance):
            break


    return Sigma, weightMaxPosteriori

# introducing a second method using as prototype the code from:
# https://github.com/rapidprom/source/blob/master/RapidMiner_Unuk/src/com/rapidminer/operator/learner/functions/kernel/rvm/RVMClassification.java
def iterative_reweighted_least_squares_algorithm2(X, target, kernel_mode="rbf",
              max_iter =25, alpha_threshold = 10e8, gradient_threshold=10e-6, overshoot_criterion= power(2,-8)):

    weights = np.zeros(X.shape[0]+1)

    A=np.zeros((X.shape[0]+1,X.shape[0]+1),float)
    np.fill_diagonal(A,1e-5)

    designMatrix = tuning.design_matrix(X.shape[0],kernel_mode,X)
    deleted_indexes = []

    for it in range(1000):
        
        A_old= np.copy(A)

        for j in range(1, A.shape[0]):

            if it>0:
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
            target = np.delete(target, deleted_indexes, 0)
            B = np.delete(B, deleted_indexes, 0)

            designMatrix = tuning.design_matrix(X.shape[0], "rbf", X)
            weights = np.delete(X, deleted_indexes, 0)

            deleted_indexes.clear()


        debug=0

        y = np.dot(designMatrix, weights)
        y = np.asarray([almost_sigmoid(y[i]) for i in range(y.shape[0])])

        # Negative log function minimization
        data_term = 0
        for i in range(target.shape[0]):

            if target[i] == 1:
                data_term -= math.log(y[i])
            else:
                data_term -= math.log( 1.0 -y[i])


        regulariser = -0.5 * np.sum( np.dot( weights.T , np.dot(A, weights) ) )
        error = (data_term + regulariser / 2.0) / weights.shape[0]

        # tempB = [y[i] * (1 - y[i]) for i in range(len(y))]
        # B = np.asarray(tempB)

        for g in range(max_iter):

            temp_weight = [y[k]*(1-y[k]) for k in range(y.shape[0])]
            irls_weights = np.diagflat(temp_weight)

            Hessian = np.dot( designMatrix.T, np.dot(irls_weights, designMatrix))
            Hessian += np.diag(A)
            Sigma = np.linalg.inv(Hessian)

            e = np.asarray([target[k]-y[k] for k in range(y.shape[0])])
            gradient_error = np.dot(designMatrix.T, e) - np.dot(A, weights)
            gradient_error-= A*weights

            delta_weights = np.dot(Sigma, gradient_error)


            if (g>=2 and np.linalg.norm(gradient_error) / weights.shape[0] < gradient_threshold):
                break

            delta_weights = np.dot(Sigma, gradient_error)

            l = 1

            # Overshooting part
            while (l > overshoot_criterion):

                weightMaxPosteriori = weights +l * delta_weights

                y = np.dot(designMatrix, weightMaxPosteriori)
                y = np.asarray([almost_sigmoid(y[i]) for i in range(y.shape[0])])

                data_term = 0
                for i in range(target.shape[0]):

                    if target[i] == 1:
                        try:
                            data_term -= math.log(y[i])
                        except ValueError:
                            lo = y[i]
                            debug=0
                    else:
                        data_term -= math.log(1.0 - y[i])

                regulariser = -0.5 * np.dot(weights.T, np.dot(A, weights))
                error_update = (data_term + regulariser / 2.0) / weights.shape[0]

                if (error_update > error):
                    l/=2.0
                else:
                    weights = weightMaxPosteriori
                    break

            weights = weightMaxPosteriori
            gamas = np.ones(X.shape[0] , float) - np.diag(A) * np.diag(Sigma)
            A_old = np.copy(A)

            deleted_indexes = []

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
            target = np.delete(target, deleted_indexes, 0)
            B = np.delete(B, deleted_indexes, 0)
            # B = np.delete(B, deleted_indexes, 1)

            designMatrix = tuning.design_matrix(X.shape[0], "rbf", X)
            weights = np.delete(X, deleted_indexes, 0)

            deleted_indexes.clear()

        if (np.abs(np.trace(A) - np.trace(A_old))) < 1e-3:
            break

    return weights, X
