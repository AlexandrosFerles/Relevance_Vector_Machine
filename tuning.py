import numpy as np
import kernel

def tune_hyperparameters_first_method(mean, Sigma):

    ret = np.array((mean.shape[0], mean.shape[0]), float)
    hyperparameters=[]
    for index in range(0,len(mean)):
        hyperparameters.append(1/(Sigma[index,index]+mean[index]**2))

    ret.fill(hyperparameters)

    return ret

def tune_hyperparameters_second_method(A, mean, Sigma):

    ret = np.array((mean.shape[0], mean.shape[0]), float)
    hyperparameters = []

    for index in range(0, len(mean)):
        hyperparameters.append((1-A[index,index]*Sigma[index,index]) / mean[index] ** 2)

    ret.fill(hyperparameters)

    return ret

def common_noise_variance(t, designMatrix, mean, Sigma, gamas):

    return np.linalg.norm(t-np.dot(designMatrix,mean),2)**2/(len(mean)-np.sum(gamas))

def design_matrix(N,kernel_mode,X):

    ret = np.ndarray(shape=(N,N+1))
    for i in range(0,N):
        ret[i,0]=1

    for i in range(0,N):
        for j in range(1,N+1):
            if (kernel_mode=="polynomial"):
                ret[i,j]=kernel.polynomial_kernel(X[i],X[j-1])
            elif (kernel_mode=="linear_spline"):
                ret[i,j]=kernel.linear_spline(X[i],X[j-1])
            elif (kernel_mode == "gaussian"):
                ret[i, j] = kernel.gaussian(X[i], X[j - 1])
            elif (kernel_mode == "rbf"):
                ret[i, j] = kernel.rbf(X[i], X[j - 1])

    return ret

def design_matrix_classification(N,kernel_mode,X):

    ret = np.ndarray(shape=(N,N))


    for i in range(0,N):
        for j in range(0,N):
            if (kernel_mode=="polynomial"):
                ret[i,j]=kernel.polynomial_kernel(X[i],X[j-1])
            elif (kernel_mode=="linear_spline"):
                ret[i,j]=kernel.linear_spline(X[i],X[j-1])
            elif (kernel_mode == "gaussian"):
                ret[i, j] = kernel.gaussian(X[i], X[j - 1])
            elif (kernel_mode == "rbf"):
                ret[i, j] = kernel.rbf(X[i], X[j - 1])

    return ret