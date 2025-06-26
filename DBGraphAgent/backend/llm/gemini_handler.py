import os
import requests

class GeminiHandler:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.api_key}"

    def chat(self, prompt: str, model: str = "gemini-pro") -> str:
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        response = requests.post(self.api_url, json=data)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"] 