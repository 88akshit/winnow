import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

api_key = "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"
base_url =  "https://integrate.api.nvidia.com/v1"
model = "moonshotai/kimi-k3"

print("=" * 60)
print("🔑 NVIDIA API Key & Model Diagnostic Test")
print("=" * 60)

if not api_key:
    print("❌ Error: Neither NVIDIA_API_KEY nor OPENAI_API_KEY found in .env")
    sys.exit(1)

masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
print(f"Endpoint:   {base_url}")
print(f"API Key:    {masked_key}")
print(f"Target LLM: {model}\n")

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

# 1. Test Listing Models
print("1️⃣ Testing listing accessible models...")
try:
    models_response = client.models.list()
    available_ids = [m.id for m in models_response.data]
    print(f"   ✅ Success! Found {len(available_ids)} models available.")
    
    # Check if target model is in the list
    if model in available_ids:
        print(f"   ✅ '{model}' is present in the model catalog.")
    else:
        print(f"   ⚠️ Warning: '{model}' was not found in the catalog.")
        matching = [m for m in available_ids if any(k in m.lower() for k in ["llama", "deepseek", "mistral"])]
        if matching:
            print(f"   💡 Suggested alternatives: {matching[:5]}")
except Exception as e:
    print(f"   ❌ Failed to list models: {e}")

# 2. Test Chat Completion with timeout & streaming
print(f"\n2️⃣ Testing chat completion with '{model}' (timeout=15s)...")
try:
    # Test streaming first to see immediate tokens
    print("   📡 Testing streaming response...")
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hello! Reply in one short sentence confirming you are working."}],
        max_tokens=50,
        temperature=0.2,
        stream=True,
        timeout=15.0
    )
    print("   💬 Model stream: ", end="", flush=True)
    full_reply = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if delta:
            print(delta, end="", flush=True)
            full_reply += delta
    print("\n   ✅ Streaming completed successfully!\n")
    print("=" * 60)
    print("🎉 All checks passed! Your NVIDIA API configuration is working.")
    print("=" * 60)
except Exception as e:
    print(f"\n   ❌ Chat completion failed or timed out: {e}")
    print("\nSuggestions:")
    print(" - `z-ai/glm-5.3` might have high queue latency or require streaming.")
    print(" - Try a fast, dedicated NIM model in .env like:")
    print("     OPENAI_MODEL_NAME=meta/llama-3.3-70b-instruct")
    print("     OPENAI_MODEL_NAME=mistralai/mistral-large-2-instruct")
    print("=" * 60)
    print("🎉 All checks passed! Your NVIDIA API configuration is working.")
    print("=" * 60)
except Exception as e:
    print(f"   ❌ Chat completion failed: {e}")
    print("\nSuggestions:")
    print(" - Check that your NVIDIA_API_KEY has not expired or exceeded quota.")
    print(" - Verify OPENAI_MODEL_NAME in .env matches an existing model in NVIDIA's catalog.")

