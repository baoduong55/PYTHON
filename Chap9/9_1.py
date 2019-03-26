'''
Created on Mar 24, 2019

@author: BAO
'''
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
# Xbar 
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)

def perceptron(X, y):
    w = [np.array([1,2,3])]
    for i in range(N):
            if y[i]*w.dot(X[i])< 0:
                wnew = w[-1] + w[-1].dot(X[i])
            if wnew == w[-1]:
                break
            w.append(wnew)    
    return w
print("end")