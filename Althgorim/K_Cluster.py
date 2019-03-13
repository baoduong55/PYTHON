'''
Created on Mar 1, 2019
   
@author: BAO
'''
from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from mnist import MNIST
#===============================================================================
# np.random.seed(11)
#===============================================================================

#===============================================================================
# mean =[[2,3],[6,3],[3,6]]
# cov = [[1,0], [0,1]]
# N = 500
# X0 = np.random.multivariate_normal(mean[0],cov,N)
# X1 = np.random.multivariate_normal(mean[1],cov,N)
# X2 = np.random.multivariate_normal(mean[2],cov,N)
# Xk = np.concatenate((X0,X1,X2))
# K = 3
#===============================================================================

mndata = MNIST('../DATA')
mndata.load_testing()
X = mndata.test_images
X0 = np.asarray(X)[:1000,:]/256.0
X = X0

K = 10

original_label = np.asarray([0]*N + [1]*N + [2]*N).T
def display(X, lable):
    X0 = X[lable == 0,:]
    X1 = X[lable == 1,:]
    X2 = X[lable == 2,:]
    
    plt.plot(X0[:, 0], X0[:, 1],'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:,0],X1[:,1], 'go',markersize = 4, alpha = .8)
    plt.plot(X2[:,0],X2[:,1], 'rs',markersize = 4, alpha = .8)
    plt.axis('equal')
    plt.plot()   
    plt.show()
def updatelable(X, centers):
    return np.argmin(cdist(X, centers), axis = 1)
def updatecenter(X, lable):
    center = np.zeros((K,X.shape[1]))
    for x in range(K):
        center[x,:]= np.mean(X[lable == x,:], axis = 0)
    return center
def Stop(X, Y):
    return (X == Y).all()
def KClustermain(X):
    center = [X[np.random.choice(X.shape[0],K, replace = False)]]
    lable = []
    lable.append(updatelable(X, center[-1]))
    while True:
        center.append(updatecenter(X, lable[-1]))
        if Stop(center[-2], center[-1]):
            break
        lable.append(updatelable(X, center[-1]))
    return lable[-1], center[-1]


  


print("my center")
label1,center = KClustermain(X)
print("origin")
display(X,original_label)
print(center)
display(X, label1)
kclus = KMeans(n_clusters= K).fit(X)
print("skit predict ", kclus.cluster_centers_)
print("end")
