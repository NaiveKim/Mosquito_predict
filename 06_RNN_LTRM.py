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
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


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

print(x_train)
print(y_train)
print(x_train.shape)
model = Sequential()
# model.add(Embedding(9080, input_length=4, input_shape=(30,1)))
# model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
# model.add(MaxPool1D(pool_size=1))
model.add(LSTM(256, input_shape=(30, 1), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Flatten())
model.add(Dense(4, activation='tanh'))
model.add(Dense(1))
model.summary()

early_stop = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience = 5)
checkpoint = ModelCheckpoint('the_best.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model.compile(optimizer = 'adam', loss = 'mse', metrics=['acc'])
fit_hist = model.fit(x_train,y_train,validation_data= (x_test, y_test), epochs=150, batch_size = 16, shuffle = False)
# print(model.evaluate(x_test, y_test))

mse,mae = model.evaluate(x_test,y_test,batch_size=1)
print(model.predict(x_test,batch_size=1))
print(mse, mae)

plt.plot(fit_hist.history['val_acc'], label='val_acc')
plt.plot(fit_hist.history['acc'], label='accuracy')
plt.legend()
plt.show()

plt.plot(fit_hist.history['loss'], label = 'loss')
plt.plot(fit_hist.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()

pred = model.predict(X)
print(pred)

