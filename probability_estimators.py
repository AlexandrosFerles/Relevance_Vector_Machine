import numpy as np

def second_order_statistics(designMatrix,A,B,t):

    Sigma=np.linalg.inv(np.dot(np.dot(np.transpose(designMatrix),B),designMatrix)+A)
    mean=np.dot(np.dot(Sigma,np.transpose(designMatrix)), np.dot(B,t))

    return mean,Sigma

def sigma_estimator(designMatrix,A,B):
    return np.linalg.inv(np.dot(np.dot(np.transpose(designMatrix),B),designMatrix)+A)

def likelihood(N, t, W, designMatrix, sigma):
    return (2*np.pi*sigma)**(-N/2)*np.exp((-1/2*sigma)*np.linalg.norm(t - np.dot(designMatrix,W), 2)**2)

def posterior_over_weights(N,t, W, sigma, designMatrix, A, B):

    Sigma = np.linalg.inv(np.dot( np.transpose(designMatrix), np.dot(B,designMatrix))+A)
    mean = np.dot(Sigma,np.dot(np.transpose(designMatrix), np.dot(B,t)))

    return (2*np.pi)**(-(N+1)/2)*(np.linalg.det(Sigma))**(-0.5)*np.exp(-0.5*np.dot(np.transpose(W-mean),np.dot(np.linalg.inv(Sigma), W-mean)))

def evidence(N, t, designMatrix, A, B):

    part1 = (2*np.pi)**(-N/2)*np.linalg.det(np.linalg.inv(B)+np.dot(designMatrix, np.dot(np.linalg.inv(A),np.transpose(designMatrix))))**(-0.5)
    part2 = np.exp(-0.5*np.dot(np.dot(np.transpose(t)*np.linalg.inv(np.linalg.inv(B)+np.dot(designMatrix,np.dot(np.linalg.inv(A), np.transpose(designMatrix)))))),t)
    return part1*part2