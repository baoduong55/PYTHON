'''
Created on Feb 25, 2019

@author: BAO
'''
class A:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def method1(self,var):
        print("hello",' ', var,self.x,self.y)
a = A("troi", "du")
a.method1("bitch")
    