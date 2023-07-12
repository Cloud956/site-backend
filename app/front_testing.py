import requests
from app.transformations import image_array_to_bytes, bytes_to_image_array, load_bibi
import cv2_rgb as cv2
ip = 'http://127.0.0.1:8000'



bibi = load_bibi()
body = {
    "name": "bibi",
    "image": image_array_to_bytes(bibi)
}
#response = requests.get(ip,body)
cv2.imwrite('bibi.png',cv2.cvtColor(bibi,cv2.COLOR_RGB2BGR))

with open('bibi.png','rb') as image:
    f = str(image.read())
    body = {

        "name": "bibi",
        "image": f

    }
    response = requests.post(ip,json = body)
    b=2
b=2