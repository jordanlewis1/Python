#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 09:59:23 2022

@author: jlewis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics


df = pd.read_csv("apiCleaned2.csv")

df = pd.read_csv("apiCleaned2.csv")
#print(df)
#sns.pairplot(df)
#print(df.columns)
#sns.boxplot(df['Reported_as_spam'], df['Spam'])
#x = df[['Username', 'Reported_as_spam', 'String Length']]
x = df[['Reported_as_spam', 'String Length']
#x = df.drop('Spam', axis = 1)

y = df[['Spam']]

#print(x)
#print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.6)

model = LogisticRegression()
model.fit(x_train, y_train.values.ravel())

predict = model.predict(x_test)

#print(classification_report(y_test, predict))
#print(confusion_matrix(y_test, predict))
print(metrics.mean_absolute_error(y_test, predict))
print(metrics.mean_squared_error(y_test, predict))
print(np.sqrt(metrics.mean_squared_error(y_test, predict)))





