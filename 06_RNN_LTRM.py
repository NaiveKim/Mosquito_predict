from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

mosquito_df = pd.read_csv('mosquito_active_rate/mosquito_active_rate_20160101_20220719.csv')

X = mosquito_df.drop(columns = ['date','rain(mm)', 'mosquito_Indicator'])
Y = mosquito_df['mosquito_Indicator']

rain_data = mosquito_df['rain(mm)']




minmax = MinMaxScaler()

minmax.fit(X)

X_scaled = minmax.transform(X)

X_scaled = pd.DataFrame(X_scaled)
X = pd.concat([rain_data, X_scaled], axis = 1)

X_values = X.values

datalist = []
datalist.append(X_values)
#
#
X = np.array(datalist)
#
#



#
#
print('x shape:', X.shape)
print('y shape:', Y.shape)

print(X)
print('-------x reshape-----------')
X = X.reshape(X.shape[1], 4,1)
print('x shape:',X.shape)
print(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

model=Sequential()
model.add(LSTM(30,activation='relu',input_shape=(4,1)))

model.add(Dense(1))

model.summary()

early_stop = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience = 5)
checkpoint = ModelCheckpoint('the_best.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model.compile(optimizer = 'adam', loss = 'mse', metrics=['acc'])
model.fit(x_train,y_train,validation_data= (x_test, y_test), epochs=100, batch_size = 1)
# print(model.evaluate(x_test, y_test))


x_input = array([11.5, 27.1, 24.3, 30.7])
x_input = x_input.reshape((1,4,1))

pred = model.predict(x_input)
print(pred)
