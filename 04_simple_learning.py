from nltk import CFG
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm.auto import tqdm
import pandas as pd

import numpy as np

from tqdm.auto import tqdm
import os

mosquito_df = pd.read_csv('./mosquito_active_rate/mosquito_active_rate_20160101_20220719.csv')
mosquito_df['date'] = pd.to_datetime(mosquito_df['date'])

minmax = MinMaxScaler()

# train_data = pd.DataFrame(mosquito_df[['rain(mm)','mean_T(℃)','min_T(℃)','max_T(℃)']], columns=['rain(mm)','mean_T(℃)','min_T(℃)','max_T(℃)'])
# test_data = pd.DataFrame(mosquito_df['mosquito_Indicator'], columns=['mosquito_Indicator'])


train_data = mosquito_df.drop(columns = ['date','mosquito_Indicator'])
test_data = mosquito_df[['mosquito_Indicator']]


x_train, x_test, y_train, y_test = train_test_split(train_data, test_data, test_size=0.2, shuffle=False)


minmax.fit(x_train)
minmax.fit(x_test)
x_test_scaled = minmax.transform(x_test)
x_train_scaled = minmax.transform(x_train)


x_train.info()

x_train_scaled.astype('f')
y_train = np.asarray(y_train, dtype = int)
y_test = np.asarray(y_test, dtype = int)
print(y_train)
print(type(y_train))


model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu',
                      alpha=0.01, batch_size=32,
                      learning_rate_init=0.1, max_iter=500)

model.fit(x_train_scaled, y_train)
print(model.score(x_test_scaled, y_test))

# test_x = mosquito_df.drop(columns=['mosquito_Indicator'])


# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# def onehot_encoding(ohe, x):
#     # 학습데이터로 부터 fit된 one-hot encoder (ohe)를 받아 transform 시켜주는 함수
#     encoded = ohe.transform(x['gender'].values.reshape(-1,1))
#     encoded_df = pd.DataFrame(encoded, columns=ohe.categories_[0])
#     x = pd.concat([x.drop(columns=['gender']), encoded_df], axis=1)
#     return x
#
#
# ohe = OneHotEncoder(sparse=False)
# ohe.fit(train_x['gender'].values.reshape(-1,1))
# train_x = onehot_encoding(ohe, train_x)

