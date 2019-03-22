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
Y = 4 + 3*(x) + .2*np.random.randn(1000, 1)
one = np.ones((x.shape[0],1))
X= concatenate((one,x), axis = 1)
w = np.dot(np.linalg.pinv(X),Y)
print("liblary result: ", w, print("lost: ", norm(X.dot(w) - Y )))
plt.plot(x,Y,'b.')
plt.plot(x,X.dot(w),'y')
def grad(w,x,y):
    return np.dot(x.T,(x.dot(w) - y)[-1])
def Gradian(X,Y):
    result = [np.array([[2], [1]])]
    dem = 0
    check = 10
    for i in range(10):
        rd_id = np.random.permutation(len(X))
        x = X[rd_id]
        y = Y[rd_id]
        for j in range(len(x)):
            wnew = result[-1] -1*grad(result[-1],x[j],y[j]).reshape((2,1))
            if j%check == 0:  
                if norm(wnew-result[-1]) < 2e-3:
                    break
            result.append(wnew)
            dem +=1
    print("ilterator number: ",dem)
    return result[-1]
Grad = Gradian(X,Y)
print("mycode: ",Grad)
print("liblary result: ", w, print("lost: ", norm(X.dot(Grad) - Y )))
plt.plot(x,Grad[0]+Grad[1]*x,'g')
plt.show()