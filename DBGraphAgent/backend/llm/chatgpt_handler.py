import os
import openai

class ChatGPTHandler:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def chat(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        return response["choices"][0]["message"]["content"].strip() 