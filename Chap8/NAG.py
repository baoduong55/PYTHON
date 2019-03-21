'''
Created on Mar 18, 2019

@author: btran40
'''
from cmath import cos
import numpy as np
import matplotlib.pyplot as plt
from numpy.core._multiarray_umath import concatenate

def funcx(x):
    return 2*x + 10*np.cos(x)
def funcy(f):
    return 2*(f[0]**2 -f[0] + 2*f[1] - 8)
def f(x):
    return x**2 + 10*np.sin(x)
t = np.arange(-6,6,0.1)
fx = f(t)
plt.plot(t,fx,'k')
def main(rate):
    result = [5]
    v= [-funcx(5)]
    dem = 0
    for it in range(100):
#    while True:
        xpredict = result[-1]+ 0.3*v[-1]
        #=======================================================================
        # plt.plot(xpredict,f(xpredict),'go')
        # plt.show()
        #=======================================================================
        xnew = xpredict - 0.1*funcx(xpredict)
        #=======================================================================
        # plt.plot(xnew,f(xnew), 'bo')
        # plt.show()
        #=======================================================================
        vnew = xnew - result[-1]
        if np.linalg.norm(result[-1] - xnew) < 1e-3:
            break
        result.append(xnew)
        dem += 1
        v.append(vnew)
    print(dem)
    return result[-1]
result = main(0.015)
print("result from my code: ",result ) 
# plot
plt.plot(t,fx,'k',result,f(result), 'bo')
plt.show()
