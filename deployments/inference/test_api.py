import os
import base64
import requests

with open("../../assets/test_images/img_1.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = {
    "image_base64": img_b64,
    "palette": [
        [255,0,0],[0,255,0],[0,0,255],
        [255,255,0],[255,0,255],[0,255,255]
    ]
}

r = requests.post("http://localhost:8000/recolor", json=payload)

if not os.path.exists("results"):
    os.makedirs("results")
with open("results/output.png", "wb") as f:
    f.write(r.content)
