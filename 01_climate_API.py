import requests, xmltodict, json
import pandas as pd
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from urllib.parse import urlencode, unquote
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import parse, Element
# yesterday = datetime.today() - timedelta(1)
# yesterday_date = yesterday.strftime("%Y-%m-%d")
#
# end_date =  str(yesterday_date).replace("-","")
# service_key = "I1MxqWJqYg/V1XIhDYJ2RMYQImlp7CnkXyQNpy6PmFGSYeSuRdJuNQdCQZJf9PoK0l6elGZ3pIwPy9RZnpq6dQ=="
#
# url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
# params ={'serviceKey' : service_key,
#          'pageNo' : '1', 'numOfRows' : '10', 'dataType' : 'XML', 'dataCd' : 'ASOS', 'dateCd' : 'HR',
#          'startDt' : '20160501', 'startHh' : '01', 'endDt' : end_date, 'endHh' : '01', 'stnIds' : '108' }
#
# response = requests.get(url)
# content = response.content
# dict = xmltodict.parse(content)
# jsonString = json.dumps(dict)
#
# content = requests.get(url).content
# print(content)


url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
params ={'serviceKey' : 'IJkDF/XrzABk9OGODmat0ZCNoGNUFpawi9nRyHwErjSKfSYCdcd7Nje45MY1ESL3wH+USjpwxa3YWXK37l9yow==',
         'pageNo' : '1', 'numOfRows' : '10', 'dataType' : 'xml', 'dataCd' : 'ASOS', 'dateCd' : 'HR',
         'startDt' : '20220720', 'startHh' : '01', 'endDt' : '20220815', 'endHh' : '01', 'stnIds' : '108' }
# params ={'serviceKey' : 'IJkDF%2FXrzABk9OGODmat0ZCNoGNUFpawi9nRyHwErjSKfSYCdcd7Nje45MY1ESL3wH%2BUSjpwxa3YWXK37l9yow%3D%3D',
#          'pageNo' : '1', 'numOfRows' : '10', 'dataType' : 'XML', 'dataCd' : 'ASOS', 'dateCd' : 'HR',
#          'startDt' : '20100101', 'startHh' : '01', 'endDt' : '20100601', 'endHh' : '01', 'stnIds' : '108' }

response = requests.get(url, params=params)
print(response.content)
print(response.text)
