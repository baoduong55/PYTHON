'''
Created on Feb 27, 2019

@author: BAO
'''
import numpy as np
import matplotlib.pyplot as pl
# height (cm)
x = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
one = np.ones((x.shape[0], 1))
X = np.concatenate((one, x), axis =1)

# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
Y = np.concatenate((one, y), axis =1)

W = np.dot(np.linalg.pinv(X),y)
print(W)
pl.plot(x,y, 'ro')
pl.axis([140, 190, 45, 70])
pl.show()