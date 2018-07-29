#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 17:33:16 2018

@author: vmueller
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rnn_utility_functions import save_model, load_model
from matplotlib.finance import candlestick_ohlc
from matplotlib import style
import matplotlib.dates as mdates
style.use('ggplot')

# First Step is to Import Dataset
dataset = pd.read_csv('A/A_Price.csv')
dataset_train = dataset.iloc[3000:4300,:]
dataset_test = dataset.iloc[4300:,:]


delta_days = 30
# Importing the training set
training_set = dataset_train.iloc[:, 5:6].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)



delta_days = 30
loaded_model = load_model()


# Getting the real stock price 
real_stock_price = dataset_test.iloc[:, 5:6].values


# Getting the predicted stock price 
dataset_total = pd.concat((dataset_train['Adj Close'], dataset_test['Adj Close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - delta_days:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(delta_days, len(dataset_test) + delta_days):
    X_test.append(inputs[i-delta_days:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price_m = loaded_model.predict(X_test)
predicted_stock_price_m = sc.inverse_transform(predicted_stock_price_m)


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Agile Technologies Stock Price')
plt.plot(predicted_stock_price_m, color = 'blue', label = 'Predicted Agile Technologies Stock Price')
plt.title('Agilent Technologies (A) Price Prediction')
plt.xlabel('Time')
plt.ylabel('Agile Technologies Stock Price')
plt.legend()
plt.show()