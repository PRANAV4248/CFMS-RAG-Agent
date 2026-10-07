# Test code to check whether the llm api is running properly

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

model = init_chat_model(
    model="openai/gpt-oss-20b",
    model_provider="openai",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0,
)

while True:
        question = input("Ask your question about the document (enter 'q' to quit):\n").strip()
        if question.lower() == "q":
            print("Thanks for using this agent!")
            break
        if not question:
            continue
        try:
            response = model.invoke(question)
            print(response.content)
        except Exception as e:
            print(f"⚠️ Request failed: {e}\nTry again.\n")