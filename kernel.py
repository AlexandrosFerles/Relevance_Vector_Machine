import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import (rbf_kernel, polynomial_kernel)

def linear_kernel(x,y):
    return np.dot(x,y.T)

def polynomial_kernel(x,y, c=1):
    return ( np.dot(x,y.T) + c )**2

def linear_spline(x,y):

    part1 = 1 + x*y + x*y*min(x, y)
    part2 = (x + y)/2*min(x, y)**2
    part3 = (min(x, y)**3)/3
    return part1 - part2 + part3

def rbf(x,y, sigma=0.5):
    num = distance.euclidean(x, y)**2
    return np.exp(-(sigma**-2)*num)

def rbf(x,y,):

    temp1 = x.shape[0]

    x= np.asarray(x).reshape(1,temp1)
    y= np.asarray(y).reshape(1,temp1)

    temp = rbf_kernel(x, y, gamma=1)
    return temp[0, 0]
