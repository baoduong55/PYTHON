'''
Created on Mar 14, 2019

@author: btran40
'''

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
img = mpimg.imread('../DATA/demo.jpg')
plt.imshow(img)
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()


X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
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


for K in [5]:
    label,center = KClustermain(X)

    img4 = np.zeros_like(X)
    # replace each pixel by its center
    for k in range(K):
        img4[label == k] = center[k]
    # reshape and display output image
    img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
    plt.imshow(img5, interpolation='nearest')
    plt.axis('off')
    plt.show()
