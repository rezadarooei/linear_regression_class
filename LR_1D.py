# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 00:37:19 2018

@author: Reza
"""

import numpy as np
import matplotlib.pyplot as plt
# load Data
import pandas as pd

data=pd.read_csv("data_1d.csv")
X=data.iloc[:,0]
y=data.iloc[:,1]

plt.scatter(X,y)
plt.show()
# turn X tu Numpy ARrray
X=np.array(X)

y=np.array(y)

#calculate denunmerator
denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(y) - y.mean()*X.sum() ) / denominator
b = ( y.mean() * X.dot(X) - X.mean() * X.dot(y) ) / denominator

# let's calculate the predicted Y
Yhat = a*X + b

# let's plot everything together to make sure it worked
plt.scatter(X, y)
plt.plot(X, Yhat)
plt.show()
