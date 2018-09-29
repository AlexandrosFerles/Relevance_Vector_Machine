import numpy as np
import math
import tuning, probability_estimators
from irls import iterative_reweighted_least_squares_algorithm2

import numpy as np
import kernel
import random
import pylab

def almost_sigmoid(x):

  ret  = 1 / (1 + math.exp(-x))
  if (ret==1):
      # we want to use the  logarithm of
      # the sigmoid function so we deviate from the value 1
      return 1-1e-10
  elif (ret==0):
      return 1e-10
  else:
      return ret

def train(X, Y, kernel_mode):

    weightMaxPosteriori, X = iterative_reweighted_least_squares_algorithm2(X, Y)

    # Convergence criterion
    return X, weightMaxPosteriori

def fit(relevance_vectors, X, weights, kernel_choice):

    prediction =[]

    for xi in range(len(X)):

        phi_x=0
        for ri in range(len(relevance_vectors)):

            if kernel_choice=="gaussian":
                phi_x+=weights[ri+1]*(kernel.gaussian(X[xi], relevance_vectors[ri]))
            elif kernel_choice=="linear":
                t1 = X[xi]
                phi_x+=weights[ri+1]*(kernel.linear_kernel(X[xi], relevance_vectors[ri]))
            elif kernel_choice=="polynomial":
                t1 = X[xi]
                phi_x+=weights[ri+1]*(kernel.polynomial_kernel(X[xi], relevance_vectors[ri]))
            elif kernel_choice=="linear_spline":
                phi_x+=weights[ri+1]*(kernel.linear_spline(X[xi], relevance_vectors[ri]))
            elif kernel_choice=="rbf":
                phi_x+=weights[ri+1]*(kernel.rbf(X[xi], relevance_vectors[ri]))

        phi_x += weights[0]
        prediction.append(phi_x)

    return prediction

def test():

    # now import the files
    ClassX_train = [(random.normalvariate(0.5, 0.8), random.normalvariate(0.5, 0.8), 1.0) for i in range(50)] + \
                   [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(50)]

    ClassO_train = [(random.normalvariate(1.5, 0.5), random.normalvariate(1.5, 0.5), 0) for i in range(50)]

    train_dataset1 = ClassX_train + ClassO_train

    # show synthetic data
    # pylab.hold(True)
    # pylab.plot([p[0] for p in ClassX_train], [p[1] for p in ClassX_train], 'bo')
    # pylab.plot([p[0] for p in ClassO_train], [p[1] for p in ClassO_train], 'ro')
    # pylab.show()

    random.shuffle(train_dataset1)

    X = np.asarray(train_dataset1)[:,:2]
    Y = np.asarray(train_dataset1)[:,2:]

    relevance_vectors, weightMaxPosteriori = train(X, Y, "rbf")
    debug = 0

test()