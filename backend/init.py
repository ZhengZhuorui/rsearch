import requests
url = 'http://10.141.223.70:8000/init_db/'
res = requests.get(url=url)
print(res.text)
