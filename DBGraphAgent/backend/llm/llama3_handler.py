import os
import requests

class Llama3Handler:
    def __init__(self):
        # OLLAMA_HOST_URL 환경변수 사용 (기본값: http://localhost:11434)
        host_url = os.getenv("OLLAMA_HOST_URL", "http://localhost:11434")
        if host_url.startswith("localhost"):  # 스킴 누락 시 자동 보정
            host_url = "http://" + host_url
        self.api_url = f"{host_url.rstrip('/')}/api/chat"

    def chat(self, prompt: str, model: str = "llama3") -> str:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        response = requests.post(self.api_url, json=data)
        response.raise_for_status()
        return response.json()["message"]["content"] 