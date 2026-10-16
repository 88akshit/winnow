import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL_NAME") or "qwen/qwen3.8-27b"

print("=" * 60)
print("⚡ Groq API Diagnostic Test")
print("=" * 60)

if not groq_key:
    print("❌ Error: GROQ_API_KEY is not set in .env")
    print("Please set: GROQ_API_KEY=gsk_... in .env")
    sys.exit(1)

masked_key = groq_key[:8] + "..." + groq_key[-4:] if len(groq_key) > 12 else "***"
print(f"Endpoint: https://api.groq.com/openai/v1")
print(f"API Key:  {masked_key}")
print(f"Model:    {model}\n")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_key
)

print(f"Testing stream completion with '{model}'...")
try:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "hi"}
        ],
        temperature=0.6,
        max_tokens=200,
        stream=True
    )
    print("💬 Groq Response: ", end="", flush=True)
    for chunk in completion:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if delta:
            print(delta, end="", flush=True)
    print("\n\n" + "=" * 60)
    print("🎉 Success! Groq is connected and responding at ultra-low latency.")
    print("=" * 60)
except Exception as e:
    print(f"\n❌ Error during completion: {e}")
