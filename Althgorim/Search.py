'''
Created on Jan 26, 2019

@author: BAO
'''

import  numpy as np
import time
from _ctypes import sizeof
from _ast import Num

Array = np.arange(10000000)

if 333 in Array:
    print("Co")
else:
    print("ko")

def line(array, number):
    dem1= 0
    if len(array) == 1:
        dem1 += 1
        return "not found"
    for i in range(len(array)):
        dem1 += 1
        if number == array[i]:
            print(dem1)
            return "founded"
    return "not found"

def binary(array, number, dem2):
    dem2 += 1
    if len(array) == 1:              
        return "not found"
    if number == array[len(array)//2]:
        return "founded", dem2
    if number < array[len(array)//2]:
        return binary(array[:len(array)//2], number, dem2)
    if number > array[len(array)//2]:
        return binary(array[len(array)//2:], number, dem2)

dem2 = 0
print(line(Array,10000000-1))
binary(Array, 10000000-1, dem2)

