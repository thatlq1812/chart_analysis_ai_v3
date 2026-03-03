#!/usr/bin/env python3
"""
Test Qwen2.5-1.5B-Instruct for Chart Analysis tasks.

This script tests the SLM's capability for:
1. OCR error correction
2. Value extraction from chart metadata
3. Description generation

Usage:
    python scripts/test_qwen_slm.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model():
    """Load Qwen model and tokenizer."""
    print("=" * 60)
    print(f"Loading {MODEL_NAME}")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load with fp16 for GPU efficiency
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print(f"Model loaded on: {model.device}")
    print(f"Model dtype: {model.dtype}")
    
    # Check memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory allocated: {allocated:.2f} GB")
    
    return model, tokenizer


def generate_response(model, tokenizer, messages: list, max_tokens: int = 512) -> str:
    """Generate response from the model."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    from transformers import GenerationConfig

    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=gen_config,
        )
    
    generated = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    
    return response


def test_ocr_correction(model, tokenizer):
    """Test OCR error correction capability."""
    print("\n" + "=" * 60)
    print("TEST 1: OCR Error Correction")
    print("=" * 60)
    
    messages = [
        {
            "role": "system",
            "content": "You are a chart analysis expert. Fix OCR errors in the extracted text from charts. Common errors include: 'l' misread as '1', 'O' as '0', 'S' as '5', etc."
        },
        {
            "role": "user", 
            "content": """Fix OCR errors in these chart labels:
- "Sal3s R3venue" 
- "2O23" 
- "l00,000"
- "Proflt Margin"
- "Q1 2O24"

Return only the corrected text, one per line."""
        }
    ]
    
    response = generate_response(model, tokenizer, messages)
    print(f"Input: OCR text with errors")
    print(f"Output:\n{response}")


def test_value_extraction(model, tokenizer):
    """Test value extraction from chart metadata."""
    print("\n" + "=" * 60)
    print("TEST 2: Value Extraction")
    print("=" * 60)
    
    messages = [
        {
            "role": "system",
            "content": "You are a chart analysis expert. Extract structured data from chart metadata."
        },
        {
            "role": "user",
            "content": """Given this bar chart metadata:
- Chart type: vertical bar chart
- Title: "Quarterly Sales 2025"
- Y-axis label: "Revenue (millions)"
- X-axis labels: ["Q1", "Q2", "Q3", "Q4"]
- Bar heights (pixels): [120, 180, 150, 200]
- Y-axis range: 0-100 (0 at pixel 300, 100 at pixel 50)

Calculate the actual values for each bar and return as JSON:
{"Q1": value, "Q2": value, "Q3": value, "Q4": value}"""
        }
    ]
    
    response = generate_response(model, tokenizer, messages)
    print(f"Input: Chart metadata with pixel coordinates")
    print(f"Output:\n{response}")


def test_description_generation(model, tokenizer):
    """Test academic-style description generation."""
    print("\n" + "=" * 60)
    print("TEST 3: Description Generation")
    print("=" * 60)
    
    messages = [
        {
            "role": "system",
            "content": "You are a chart analysis expert. Generate concise, academic-style descriptions of charts."
        },
        {
            "role": "user",
            "content": """Generate a brief description for this chart:
- Type: Line chart
- Title: "COVID-19 Cases by Month"
- Data: 
  - Jan 2020: 100
  - Feb 2020: 500
  - Mar 2020: 2000
  - Apr 2020: 8000
  - May 2020: 5000
  - Jun 2020: 3000

The description should be 2-3 sentences, suitable for academic papers."""
        }
    ]
    
    response = generate_response(model, tokenizer, messages)
    print(f"Input: Chart data points")
    print(f"Output:\n{response}")


def test_legend_mapping(model, tokenizer):
    """Test legend-to-color mapping."""
    print("\n" + "=" * 60)
    print("TEST 4: Legend Mapping")
    print("=" * 60)
    
    messages = [
        {
            "role": "system",
            "content": "You are a chart analysis expert. Match legend items with their corresponding colors and data series."
        },
        {
            "role": "user",
            "content": """Given a grouped bar chart with:
- Legend items: ["Product A", "Product B", "Product C"]
- Legend colors (RGB): [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
- Detected bars with colors:
  - Bar 1: RGB(254, 2, 1) at x=100
  - Bar 2: RGB(1, 253, 2) at x=130
  - Bar 3: RGB(2, 1, 254) at x=160
  - Bar 4: RGB(255, 1, 0) at x=200
  - Bar 5: RGB(0, 254, 1) at x=230
  - Bar 6: RGB(1, 0, 255) at x=260

Match each bar to its legend item. Return as JSON."""
        }
    ]
    
    response = generate_response(model, tokenizer, messages)
    print(f"Input: Colors and positions")
    print(f"Output:\n{response}")


def main():
    print("=" * 60)
    print("QWEN 2.5 SLM TEST FOR CHART ANALYSIS")
    print("=" * 60)
    
    # Load model
    model, tokenizer = load_model()
    
    # Run tests
    test_ocr_correction(model, tokenizer)
    test_value_extraction(model, tokenizer)
    test_description_generation(model, tokenizer)
    test_legend_mapping(model, tokenizer)
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
