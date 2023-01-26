#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 18:06:48 2022

@author: jlewis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm

df = pd.read_csv("apiCleaned2.csv")

#print(df.info())

X = df["Text"].values
y = df["Spam"].values

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)

cv = CountVectorizer() 
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = "rbf", random_state = 0)
classifier.fit(X_train, y_train)

print(classifier.score(X_test,y_test))

