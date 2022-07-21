import requests, xmltodict, json
import pandas as pd
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup


yesterday = datetime.today() - timedelta(1)
yesterday_date = yesterday.strftime("%Y-%m-%d")

end_date =  str(yesterday_date).replace("-","")
service_key = "I1MxqWJqYg%2FV1XIhDYJ2RMYQImlp7CnkXyQNpy6PmFGSYeSuRdJuNQdCQZJf9PoK0l6elGZ3pIwPy9RZnpq6dQ%3D%3D"

url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
# params ={'serviceKey' : service_key,
#          'pageNo' : '1', 'numOfRows' : '10', 'dataType' : 'XML', 'dataCd' : 'ASOS', 'dateCd' : 'HR',
#          'startDt' : '20160501', 'startHh' : '01', 'endDt' : end_date, 'endHh' : '01', 'stnIds' : '108' }

response = requests.get(url)
content = response.content
dict = xmltodict.parse(content)
jsonString = json.dumps(dict)

content = requests.get(url).content
print(content)
