from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


mosquito_df = pd.read_csv('mosquito_active_rate/mosquito_active_rate_20160101_20220719.csv')

X = mosquito_df.drop(columns = ['date', 'mosquito_Indicator'])
Y = mosquito_df['mosquito_Indicator']

X_values = X.values

datalist = []
datalist.append(X_values)
#
#
X = np.array(datalist)
#
#
print(X)
# data_reshape = data_array.reshape(1,40)




print('x shape:', X.shape)
print('y shape:', Y.shape)

print(X)
print('-------x reshape-----------')
X = X.reshape(X.shape[1], 4,1)
print('x shape:',X.shape)
print(X)

model=Sequential()
model.add(LSTM(30,activation='relu',input_shape=(4,1)))

model.add(Dense(1))

model.summary()

model.compile(optimizer = 'adam', loss = 'mse')
model.fit(X,Y, epochs=100, batch_size = 1)
x_input = array([11.5, 27.1, 24.3, 30.7])
x_input = x_input.reshape((1,4,1))

pred = model.predict(x_input)
print(pred)
