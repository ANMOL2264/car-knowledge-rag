import os
import requests

from dotenv import load_dotenv
load_dotenv()

def query(prompt):
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
    }

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    #system - Sets the model’s overall behavior or persona (optional)
    # user - The actual user’s question or input -> to which it will actual give answer to
    # assistant - The model’s previous reply (if continuing a conversation) -  in order to refer to previous conversations
    response = query({
        "messages": [
            {"role": "system", "content": "You are a car domain expert assistant. Use only the provided context to answer. Do not rely on external knowledge."},
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "model": "google/gemma-2-2b-it:nebius"
    })

    return response["choices"][0]["message"]['content'],response

# print(response["choices"][0]["message"])