'''
Created on Mar 19, 2019

@author: BAO
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.core._multiarray_umath import concatenate
from scipy.spatial.distance import cdist
from numpy.linalg import norm
x = np.random.rand(1000, 1)
y = 4 + 3*(x) + .2*np.random.randn(1000, 1)
one = np.ones((x.shape[0],1))
X= concatenate((one,x), axis = 1)
w = np.dot(np.linalg.pinv(X),y)
print("liblary result: ", w)
plt.plot(x,y,'b.')
plt.plot(x,X.dot(w),'y')
def grad(w):
    return np.dot(X.T,X.dot(w) - y)/(X.shape[0])
def Gradian(x,y):
    result = [np.array([[2], [1]])]
    dem = 0
    while True:               
        wnew = result[-1] -1*grad(result[-1])
        if norm(wnew-result[-1]) < 1e-3:
            break
        result.append(wnew)
        dem +=1
    print(dem)
    return result[-1]
Grad = Gradian(X,y)
print("mycode: ",Grad)
plt.plot(x,Grad[0]+Grad[1]*x,'g')
plt.show()