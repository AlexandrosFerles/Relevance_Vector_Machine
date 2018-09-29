import numpy as np
import math
import sklearn
import matplotlib.pyplot as plt


#from sklearn.datasets import load_boston
#boston = load_boston()
#print(boston.data.shape)

def linear_kernel(x,y):
    return x*y

def RBF_kernel(x,y,s):

    return np.exp(-(x-y)**2/2*s**2)

def linear_spline(x,y):

    part1 = 1 + x*y + x*y*min(x, y)
    part2 = (x + y)/2*min(x, y)**2
    part3 = (min(x, y)**3)/3
    return part1 - part2 + part3

def likelihood(N, t, W, designMatrix, sigma):
    return (2*np.pi*sigma)**(-N/2)*np.exp((-1/2*sigma)*np.linalg.norm(t - np.dot(designMatrix,W), 2)**2)


def second_order_statistics(designMatrix,A,B,t):

    Sigma=np.linalg.inv(np.dot(np.dot(np.transpose(designMatrix),B),designMatrix)+A)
    mean=np.dot(np.dot(Sigma,np.transpose(designMatrix)), np.dot(B,t))

    return mean,Sigma

def posterior_over_weights(N,t, W, sigma, designMatrix, A, B):

    Sigma = np.linalg.inv(np.dot( np.transpose(designMatrix), np.dot(B,designMatrix))+A)
    mean = np.dot(Sigma,np.dot(np.transpose(designMatrix), np.dot(B,t)))

    return (2*np.pi)**(-(N+1)/2)*(np.linalg.det(Sigma))**(-0.5)*np.exp(-0.5*np.dot(np.transpose(W-mean),np.dot(np.linalg.inv(Sigma), W-mean)))

def evidence(N, t, designMatrix, A, B):

    part1 = (2*np.pi)**(-N/2)*np.linalg.det(np.linalg.inv(B)+np.dot(designMatrix, np.dot(np.linalg.inv(A),np.transpose(designMatrix))))**(-0.5)
    part2 = np.exp(-0.5*np.dot(np.dot(np.transpose(t)*np.linalg.inv(np.linalg.inv(B)+np.dot(designMatrix,np.dot(np.linalg.inv(A), np.transpose(designMatrix)))))),t)
    return part1*part2

def tune_hyperparameters_first_method(mean, Sigma):

    hyperparameters=[]
    for index in range(0,len(mean)):
        hyperparameters.append(1/(Sigma[index][index]+mean[index]**2))

    return hyperparameters

def tune_hyperparameters_second_method(A, mean, Sigma):
    hyperparameters = []
    for index in range(0, len(mean)):
        hyperparameters.append((1-A[index]*Sigma[index][index]) / mean[index] ** 2)

    return hyperparameters

def common_noise_variance(t, designMatrix, mean, Sigma, A):

    # estimate sum of gamas
    gamas=0
    for index in range(0,len(mean)):
        gamas+=1-A[index]*Sigma[index][index]

    return np.linalg.norm(t-np.dot(designMatrix,mean),2)**2/(len(mean)-gamas)

def design_matrix(N,kernel_mode,X,Y):

    ret = np.ndarray(shape=(N,N+1))
    for i in range(0,N):
        ret[i,0]=1

    for i in range(0,N):
        for j in range(1,N+1):
            if (kernel_mode=="linear_spline"):
                ret[i,j]=linear_spline(X[i],Y[j-1])

    return ret


def test():

    X = np.linspace(-10, 10, 100)
    Y = np.sin(np.abs(X)) / np.abs(X)

    variance=0.01

    A=np.zeros((X.shape[0]+1,X.shape[0]+1),float)
    B=np.zeros((X.shape[0],X.shape[0]),float)

    np.fill_diagonal(A,1)
    np.fill_diagonal(B,(1/variance))

    designMatrix= design_matrix(X.shape[0],"linear_spline",X,Y)
    mean, Sigma = second_order_statistics(designMatrix,A,B,Y)

    max_iter=1000

    for i in range(max_iter):

        for j in range(A.shape[0]):
            A[j,j] = 1/(Sigma[j,j]+mean[j]**2)

        designMatrix = design_matrix(X.shape[0], "linear_spline", X, Y)
        mean, Sigma = second_order_statistics(designMatrix, A, B, Y)

    res=np.diag(A)
    debug=0

test()