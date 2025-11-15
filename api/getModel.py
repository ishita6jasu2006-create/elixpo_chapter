
import os 
from config import MODEL_PATH
import requests

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading Real-ESRGAN model...")

        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
        response = requests.get(url, stream=True)

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)

        print("Model downloaded:", MODEL_PATH)

if __name__ == "__main__":
    download_model()