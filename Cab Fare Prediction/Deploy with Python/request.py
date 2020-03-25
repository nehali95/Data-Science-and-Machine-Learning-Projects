import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'passenger_count', 'year', 'Month', 'Hour', 'distance'})

print(r.json())