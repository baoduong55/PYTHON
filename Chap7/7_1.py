'''
Created on Mar 18, 2019

@author: btran40
'''
from cmath import cos
def funcy(x):
    return 2*x - 5*cos(x)
def main(x,rate):
    result = [x]
    for i in [100]:
        xnew = x - rate*funcy(x)
        if result[-1] - xnew < 1e-3:
            break
        result.append(xnew)
    return result[-1]
print("result from my code: ", main(5, 0.1))  
    