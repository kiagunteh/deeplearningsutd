import os
import requests
import zipfile
from io import BytesIO

URL = "https://www.kaggle.com/api/v1/datasets/download/mrwellsdavid/unsw-nb15"   # replace with your URL
FILE_PATH = "./data"

os.makedirs(FILE_PATH, exist_ok=True)

response = requests.get(URL)
with zipfile.ZipFile(BytesIO(response.content)) as z:
    z.extractall(FILE_PATH)

print(f"Done! Files saved to {FILE_PATH}")
