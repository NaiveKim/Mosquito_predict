import requests
import pandas as pd
import time
from datetime import datetime



url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
params ={'serviceKey' : '서비스키', 'pageNo' : '1', 'numOfRows' : '10', 'dataType' : 'XML', 'dataCd' : 'ASOS', 'dateCd' : 'HR',
         'startDt' : '20100101', 'startHh' : '01', 'endDt' : '20100601', 'endHh' : '01', 'stnIds' : '108' }

response = requests.get(url, params=params)

response_pd = pd.DataFrame(response)

today_date = datetime.today()

print(response.content)
print(response.headers)
print(today_date)