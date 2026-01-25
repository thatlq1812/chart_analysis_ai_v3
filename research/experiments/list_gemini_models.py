"""List available Gemini models."""

import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

print("Available Gemini Models:")
print("=" * 80)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"\n[MODEL] {model.name}")
        print(f"   Display Name: {model.display_name}")
        print(f"   Description: {model.description}")
        print(f"   Input Token Limit: {model.input_token_limit:,}")
        print(f"   Output Token Limit: {model.output_token_limit:,}")
        
        # Highlight thinking/research models
        name_lower = model.name.lower()
        if 'thinking' in name_lower or 'pro' in name_lower or 'research' in name_lower:
            print(f"   [RECOMMENDED FOR RESEARCH]")
