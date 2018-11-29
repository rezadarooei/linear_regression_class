# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 00:26:55 2018

@author: Reza
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#read data
df=pd.read_excel("mlr02.xls")

X=df.as_matrix()

plt.scatter(X[:,1],X[:,0])
plt.show()

plt.scatter(X[:,2],X[:,0])
plt.show()

df['ones']=1

Y=df['X1']
X=df[["X2","X3","ones"]]

X2only=df[["X2","ones"]]
X3only=df[["X3","ones"]]

def get_lr2(X,Y):
    W=np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))
    Yhat=X.dot(W)
    d1=Y-Yhat
    d2=Y-Y.mean()
    lr2=1-d1.dot(d2)/d2.dot(d2)
    return lr2

print("r2 for X2 only",get_lr2(X2only,Y))
print("r2 for X3 only",get_lr2(X3only,Y))
print("r2 for X",get_lr2(X,Y))

    

