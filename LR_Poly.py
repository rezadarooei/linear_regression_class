# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 23:50:26 2018

@author: Reza
"""

import numpy as np
import matplotlib.pyplot as plt

#load data
X=[]
Y=[]
for line in open('data_poly.csv'):
    x,y=line.split(',')
    x=float(x)
    X.append([1,x,x*x])
    Y.append(float(y))

X=np.array(X)
Y=np.array(Y)
           
#plot what is look like)
plt.scatter(X[:,1],Y)
plt.show()

#calculate W

W=np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))

Yhat1=X.dot(W)
Yhat=np.dot(X,W)


plt.scatter(X[:,1],Y)
plt.plot(sorted(X[:,1]),sorted(Yhat))
plt.show()