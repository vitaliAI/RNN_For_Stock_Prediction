"""
Created on Sat Jul 28 17:49:30 2018

@author: vmueller
"""

# Data Preprocessing
# Recurrent Neural Network



# Part 1 - Data Preprocessing
# Importing the libraries
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

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(delta_days, 1300):
    X_train.append(training_set_scaled[i-delta_days:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

save_model(regressor)

loaded_model = load_model()

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price 
real_stock_price = dataset_test.iloc[:, 5:6].values


# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Adj Close'], dataset_test['Adj Close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - delta_days:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(delta_days, len(dataset_test) + delta_days):
    X_test.append(inputs[i-delta_days:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


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