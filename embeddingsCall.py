import requests, os
from dotenv import load_dotenv
load_dotenv()
def getEmbeddings(payload):
    API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5/pipeline/feature-extraction"
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
    }
    response = requests.post(API_URL, headers=headers, json={"inputs": payload})
    return response.json()