import requests

url = 'http://localhost:5000/api'
# The JSON keys here must match those expected in your Flask app's /api route
r = requests.post(url, json={
    'SepalLengthCm': 5.1,
    'SepalWidthCm': 3.5,
    'PetalLengthCm': 1.4,
    'PetalWidthCm': 0.2
})

print(r.json())




