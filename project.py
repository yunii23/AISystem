# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:04:14 2021

@author: DSU
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import plot_confusion_matrix
from scipy.stats import norm, boxcox
from collections import Counter
from scipy import stats
from pandas_profiling import ProfileReport
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
        
dataset = pd.read_csv("C:/Users/dsu/Documents/spyder_test1/winequality-red.csv")
#print(dataset.head())

#print(dataset.shape)

#print(dataset.describe())

#print(dataset.info())

dataset['quality'] = np.where(dataset['quality'] > 6, 1, 0)
dataset['quality'].value_counts()

X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)
print(X.shape)
print(y.shape)






    
    
