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
t = np.arange(-1,6,0.1)
d = X.shape[0]
w = [np.random.randn(d, 1).reshape(3)]
print(w)
def h(w, x):    
    return np.sign(np.dot(w.T, x.T))

def has_converged(X, y, w):    
    return np.array_equal(h(w, X), -1*y)
def f(x,w):
    return -1*(x*w[1]+w[0])/w[2]
def perceptron(X, y):
    dem = 0    
    while True:
        mix_id = np.random.permutation(2*N)           
        for i in mix_id:
            dem += 1
#            plt.plot(X[i][1],X[i][2],"bo")
            print("-x: ",X[i],"-y: ",y[i],"-w: ",w[-1])
            if h(w[-1],X[i])*y[i] >= 0:
                wnew = w[-1] - y[i]*X[i]
#                plt.plot(t,f(t,wnew))
                w.append(wnew)
        if has_converged(X, y, w[-1]):
            break
    print(dem)  
    return w[-1]
plt.plot(X0[0],X0[1],'go')
plt.plot(X1[0],X1[1],'ro')
plt.plot(t,f(t,w[-1]),'k')
wnew = perceptron(X.T, y[-1])
plt.plot(t,f(t,wnew),'k')

plt.show()
print("end")