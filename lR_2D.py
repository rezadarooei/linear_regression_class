# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:47:57 2018

@author: Reza
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


data=pd.read_csv("data_2d.csv")
X=data.iloc[:,0:2]
Y=data.iloc[:,[2]]

X=np.array(X)
Y=np.array(Y)

#3D Scatter plot
fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.scatter(X[:,0],X[:,1],Y)
plt.show()

#Calculate weights

W=np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))
Yhat=np.dot(X,W)

d1=Y-Yhat
d2=Y-Y.mean()


r2=1-d1.dot(d1)/d2.dot(d2)