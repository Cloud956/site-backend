import requests
import json

address = "http://18.184.42.144:80/transformations/TO_RGB"


body = {"image": "tester", "param1": 2, "param2": 3, "param3": 5}

respons = requests.post(address, json.dumps(body))
print(respons.status_code)
print(respons.content)
