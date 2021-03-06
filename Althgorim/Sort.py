'''
Created on Jan 28, 2019

@author: BAO
'''
import numpy as np
import random
from numpy.core._multiarray_umath import concatenate
import time
from sklearn.cluster import KMeans
n = 1000
arr = random.sample(range(1000), n)
#arr = np.array([ 0, 12, 16,  0,  7, 17, 17,  5])
#arr = [33, 11, 76, 13, 3, 73, 9, 89, 97, 22]

# quicksort
def quicksort(arr, first, last):
    if first >= last:
        return
    f = first
    l = last
    mid = (last + first)//2
    while f < l:
        while arr[f] < arr[mid]:
            f += 1
        while arr[l] > arr[mid]:
            l -= 1
        if f <= l:
           arr[f], arr[l] = arr[l], arr[f]
           f += 1
           l -= 1
    quicksort(arr, first, l)
    quicksort(arr, f, last)
    return

# insert sort
def maininsersort(ar):
    result = np.array([ar[0]])
    def insersort(ar, result):
        for a in range(1,len(ar)):
            for b in range(len(result)):
                if ar[a] <= result[b]:
                    result = np.insert(result, b, ar[a])
                    break
                if b == len(result) - 1:
                    result = np.append(result, ar[a])
        return result
    return insersort(ar, result)

# test
def quicksortB1(ar, order):
    # neu mang co 1 phan tu khoi xap
    if(len(ar) <= 1):
        return ar
    # order: so dau tien, thu 2, thu 3,... cua tung phan tu
    classinput = [[],[],[],[],[],[],[],[],[],[],[]]
    # lap tung phan tu cua array
    for i in range(len(ar)):        
        # loop tu 0 - 9 de phan class
        for j in range(0,10):
            # neu phan tu hien tai co chieu dai 
            # ngan hon oder cho vao class 0
            if order >= len(str(ar[i])):
                classinput[0].append(ar[i])
                break
            # phan class
            if str(ar[i])[order]== str(j):                                                   
                classinput[j].append(ar[i])
                break
    ar = []
    order += 1
    # rap lai 9 class
    for i in range(9,-1, -1):
        if classinput[i] != []:
            ar = ar + quicksortB1(classinput[i], order)
    return ar

# MergeSort
def Merge(X,Y):
    result = []
    x = 0
    y = 0
    while x < len(X) or y < len(Y):
            if X[x] < Y[y]:                
                result.append(X[x])
                x += 1
                if x == len(X):
                    for j in range(y, len(Y)):
                        result.append(Y[j])
                        y = len(Y)                             
            else:
                result.append(Y[y])
                y += 1              
                if y == len(Y):
                    for j in range(x, len(X)):
                        result.append(X[j])
                        x = len(X)
    return result     
def MergeSort(A):
    if len(A) == 1:
        return A
    else:
        resulta = Merge(MergeSort(A[:len(A)//2]),  MergeSort(A[len(A)//2:]))
        return resulta
       
print("voi so phan tu: ", n,"mang ban dau: ",arr)
print(arr.sort())
print(MergeSort(arr))
           
                

    