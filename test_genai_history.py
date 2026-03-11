import os
from google import genai
from google.genai import types

api_key = os.environ.get("GEMINI_API_KEY", "NOT_SET")
if api_key == "NOT_SET":
    print("No API key, just verifying syntax is importable without errors...")

try:
    client = genai.Client(api_key="123")
    
    history = [
        types.Content(role="user", parts=[types.Part.from_text(text="hello")])
    ]
    
    config = types.GenerateContentConfig(
        system_instruction="You are a helpful assistant"
    )
    
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=config,
        history=history
    )
    print("Syntax ok for chats.create")
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()

