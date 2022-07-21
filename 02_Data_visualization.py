import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import datetime

mosquito_df = pd.read_csv('./Mosquito Indicator in Seoul, Korea/mosquito_Indicator.csv')

pd.to_datetime(mosquito_df['date'])

# bplot=sns.boxplot(data=mosquito_df)
# plt.show()
#
# for i in mosquito_df['date']:
#     numeric = i.replace("-","")
#     mosquito_df.loc[mosquito_df['date']== i,"date"] = numeric
#
# print(mosquito_df)
# for j in mosquito_df['date']:

    # mosquito_df.loc[mosquito_df['date']== i,"date"] = int(numeric)
plt.figure(figsize = (20,6))
plt.bar(mosquito_df['date'], mosquito_df['mosquito_Indicator'])
plt.title('date')

plt.show()