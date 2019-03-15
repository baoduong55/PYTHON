'''
Created on Mar 15, 2019

@author: BAO
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection._split import train_test_split
from scipy.spatial.distance import cdist
# input
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print('Number of classes:', len(np.unique(iris_y)))
print('Number of data points:', len(iris_y))

X, x, Y, y = train_test_split(iris_X, iris_y, test_size=50)

def Knearestneighbour(data, label, K):
    distance = cdist(x,X)
    index = np.argsort(distance)
    distance.sort(axis = 1)
    dataneighbour = distance[:,:K]
    newlabel[:] = max(y.tolist(), key=y.tolist().count)
    
    return label
print("end")    