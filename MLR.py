#!/usr/bin/env python
# coding: utf-8

##################################################
### Author: Sanjoy Biswas
### Email : sanjoy.eee32@gmail.com
##################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#Reading csv file
data = pd.read_csv('taxi.csv')
data.head()

#Selection of features
x = data.drop(['Numberofweeklyriders'], axis=1).values
y = data['Numberofweeklyriders'].values

#Splitting into training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)

#Model creattion
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

#Checking model score
print("Training score: ", model.score(x_train, y_train))

#Saving model
pickle.dump(model, open('taxi.pkl', 'wb'))

#Testing model by loading it first
model1= pickle.load(open('taxi.pkl', 'rb'))
print(model1.predict([[80,1777000,10000,85]]))

