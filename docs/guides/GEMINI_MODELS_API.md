# Gemini API - Available Models Reference

| Property | Value |
| --- | --- |
| **Last Updated** | 2026-03-05 |
| **API Key Env** | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| **SDK** | `google-genai` (new SDK) |
| **API Console** | https://aistudio.google.com/apikey |

---

## SDK Usage

```python
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Text generation
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Your prompt here",
)

# Vision (image + text)
from google.genai import types

image_bytes = Path("chart.png").read_bytes()
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        "Describe this chart.",
    ],
)
```

**Note:** If both `GOOGLE_API_KEY` and `GEMINI_API_KEY` are set, the SDK prefers `GOOGLE_API_KEY`.

---

## Available Models (fetched 2026-03-05)

### Recommended for This Project

| Model ID | Display Name | Input Tokens | Output Tokens | Use Case |
| --- | --- | --- | --- | --- |
| `gemini-2.5-flash` | Gemini 2.5 Flash | 1,048,576 | 65,536 | **DEFAULT** - Best balance of speed/quality/cost |
| `gemini-2.5-pro` | Gemini 2.5 Pro | 1,048,576 | 65,536 | Complex reasoning, highest accuracy |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash-Lite | 1,048,576 | 65,536 | Cheapest option, fast |

### Stable Models (GA)

| Model ID | Display Name | Input Tokens | Output Tokens | Description |
| --- | --- | --- | --- | --- |
| `gemini-2.5-flash` | Gemini 2.5 Flash | 1,048,576 | 65,536 | Mid-size multimodal, released June 2025 |
| `gemini-2.5-pro` | Gemini 2.5 Pro | 1,048,576 | 65,536 | Stable release June 2025 |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash-Lite | 1,048,576 | 65,536 | Released July 2025 |
| `gemini-2.0-flash` | Gemini 2.0 Flash | 1,048,576 | 8,192 | Previous generation |
| `gemini-2.0-flash-001` | Gemini 2.0 Flash 001 | 1,048,576 | 8,192 | Pinned version, January 2025 |
| `gemini-2.0-flash-lite` | Gemini 2.0 Flash-Lite | 1,048,576 | 8,192 | Lightweight 2.0 |
| `gemini-2.0-flash-lite-001` | Gemini 2.0 Flash-Lite 001 | 1,048,576 | 8,192 | Pinned version |

### Preview Models (Newest)

| Model ID | Display Name | Input Tokens | Output Tokens | Description |
| --- | --- | --- | --- | --- |
| `gemini-3.1-pro-preview` | Gemini 3.1 Pro Preview | 1,048,576 | 65,536 | Latest Pro model |
| `gemini-3.1-pro-preview-customtools` | Gemini 3.1 Pro Custom Tools | 1,048,576 | 65,536 | Optimized for tool calling |
| `gemini-3-flash-preview` | Gemini 3 Flash Preview | 1,048,576 | 65,536 | Fast preview model |
| `gemini-3.1-flash-lite-preview` | Gemini 3.1 Flash Lite Preview | 1,048,576 | 65,536 | Budget preview |
| `gemini-3-pro-preview` | Gemini 3 Pro Preview | 1,048,576 | 65,536 | Pro-tier preview |

### Alias Models (Latest Pointers)

| Model ID | Display Name | Points To |
| --- | --- | --- |
| `gemini-flash-latest` | Gemini Flash Latest | Latest Flash release |
| `gemini-flash-lite-latest` | Gemini Flash-Lite Latest | Latest Flash-Lite release |
| `gemini-pro-latest` | Gemini Pro Latest | Latest Pro release |

### Image Generation Models

| Model ID | Display Name | Input Tokens | Output Tokens |
| --- | --- | --- | --- |
| `gemini-2.0-flash-exp-image-generation` | Gemini 2.0 Flash Image Gen | 1,048,576 | 8,192 |
| `gemini-2.5-flash-image` | Gemini 2.5 Flash Image | 32,768 | 32,768 |
| `gemini-3-pro-image-preview` | Gemini 3 Pro Image | 131,072 | 32,768 |
| `gemini-3.1-flash-image-preview` | Gemini 3.1 Flash Image | 65,536 | 65,536 |

### Gemma Models (Open-weight, via API)

| Model ID | Display Name | Input Tokens | Output Tokens |
| --- | --- | --- | --- |
| `gemma-3-1b-it` | Gemma 3 1B | 32,768 | 8,192 |
| `gemma-3-4b-it` | Gemma 3 4B | 32,768 | 8,192 |
| `gemma-3-12b-it` | Gemma 3 12B | 32,768 | 8,192 |
| `gemma-3-27b-it` | Gemma 3 27B | 131,072 | 8,192 |
| `gemma-3n-e4b-it` | Gemma 3n E4B | 8,192 | 2,048 |
| `gemma-3n-e2b-it` | Gemma 3n E2B | 8,192 | 2,048 |

### Other Specialized Models

| Model ID | Purpose |
| --- | --- |
| `gemini-embedding-001` | Text embeddings (2048 input) |
| `imagen-4.0-generate-001` | Image generation |
| `imagen-4.0-ultra-generate-001` | High-quality image generation |
| `veo-2.0-generate-001` / `veo-3.0-generate-001` / `veo-3.1-generate-preview` | Video generation |
| `gemini-2.5-flash-native-audio-*` | Audio processing |
| `gemini-robotics-er-1.5-preview` | Robotics |

---

## Model Selection Guide for Chart Analysis

| Task | Recommended Model | Reasoning |
| --- | --- | --- |
| Chart annotation (benchmark GT) | `gemini-2.5-flash` | Vision + structured output, good accuracy |
| OCR correction | `gemini-2.5-flash-lite` | Simple text task, minimize cost |
| Chart reasoning (Stage 4) | `gemini-2.5-flash` | Balance speed + accuracy |
| Complex chart analysis | `gemini-2.5-pro` | Highest accuracy for ambiguous charts |
| QA generation | `gemini-2.5-flash` | Diverse question generation |
| Batch processing (>1000 charts) | `gemini-2.5-flash-lite` | Cost optimization |

### Key Upgrade: 2.0 -> 2.5

| Feature | gemini-2.0-flash | gemini-2.5-flash |
| --- | --- | --- |
| Max output tokens | 8,192 | **65,536** (8x more) |
| Max input tokens | 1,048,576 | 1,048,576 |
| Vision quality | Good | **Better** |
| Cost | Baseline | Similar |
| Stability | GA (Jan 2025) | GA (Jun 2025) |

---

## Configuration in This Project

### .env

```bash
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### config/models.yaml

```yaml
ai_routing:
  providers:
    gemini:
      enabled: true
      model: "gemini-2.5-flash"
      api_key_env: "GOOGLE_API_KEY"
```

### Fetching Model List Programmatically

```python
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
for model in client.models.list():
    print(f"{model.name}: {model.display_name} (in={model.input_token_limit}, out={model.output_token_limit})")
```
