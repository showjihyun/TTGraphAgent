import os
import requests

class Claude4Handler:
    def __init__(self):
        self.api_key = os.getenv("CLAUDE_API_KEY")
        self.api_url = "https://api.anthropic.com/v1/messages"

    def chat(self, prompt: str, model: str = "claude-3-opus-20240229") -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["content"] 