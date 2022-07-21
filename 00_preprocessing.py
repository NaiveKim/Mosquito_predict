import pandas as pd

import numpy as np

mosquito_df = pd.read_csv('./mosquito_active_rate/mosquito_active_rate_20160101_20220719.csv')

mosquito_df['date'] = pd.to_datetime(mosquito_df['date'])

print(mosquito_df)
print(mosquito_df.info())
print(mosquito_df.describe())