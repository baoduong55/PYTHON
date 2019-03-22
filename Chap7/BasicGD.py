'''
Created on Mar 18, 2019

@author: btran40
'''
from cmath import cos
import numpy as np
import matplotlib.pyplot as plt
from numpy.core._multiarray_umath import concatenate
def funcx(f):
    return 4*f[0]**3 + 4*f[0]*f[1] - 26*f[0] -2*f[1] + 2
def funcy(f):
    return 2*(f[0]**2 -f[0] + 2*f[1] - 8)
def f(x):
    return x**2 + 5*np.sin(x)
def main(rate):
    result = [np.array([[3], [4]])]
    dem = 0
    while True:
        xnew = result[-1] - np.reshape(rate*concatenate((funcx(result[-1]),funcy(result[-1]))),result[-1].shape)
        if np.linalg.norm(result[-1] - xnew) < 1e-3:
            break
        result.append(xnew)
        dem += 1
    print(dem)
    return result[-1]
result = main(0.015)
print("result from my code: ",result ) 
# plot
t = np.arange(-6,6,0.1)
fx = f(t)
plt.plot(t,fx,'k',result,f(result), 'bo')
plt.show()
