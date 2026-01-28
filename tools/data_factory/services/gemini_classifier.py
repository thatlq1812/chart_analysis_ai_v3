"""
Gemini Chart Classifier - Chart classification and QA generation using Google Gemini API

This service uses Google Gemini 3 Flash for:
1. Classifying if an image is a chart
2. Determining chart type (bar, line, pie, scatter, etc.)
3. Generating QA pairs for training data

Updated to use the new `google.genai` SDK (replaces deprecated google.generativeai)
"""

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from loguru import logger
from PIL import Image

# Load environment variables
load_dotenv()


class GeminiChartClassifier:
    """
    Chart classifier using Google Gemini API.
    
    Uses gemini-3-flash-preview for optimal speed/cost/quality tradeoff.
    """
    
    # Classification prompt template
    CLASSIFICATION_PROMPT = """Analyze this image and determine if it is a chart/graph visualization.

Respond with a JSON object only (no markdown, no explanation):
{
    "is_chart": true/false,
    "chart_type": "bar|line|pie|scatter|area|histogram|heatmap|box|other|none",
    "confidence": 0.0-1.0,
    "elements": {
        "has_title": true/false,
        "has_legend": true/false,
        "has_x_axis": true/false,
        "has_y_axis": true/false,
        "has_grid": true/false,
        "has_data_labels": true/false
    },
    "brief_description": "one sentence description of what the chart shows"
}

Rules:
- If NOT a chart (photo, diagram, logo, table, etc.), set is_chart=false and chart_type="none"
- Only classify as chart if it visualizes quantitative data
- Tables are NOT charts
- Flowcharts/diagrams are NOT charts (set chart_type="none")
- confidence should reflect certainty about the classification
"""

    # QA generation prompt template
    QA_GENERATION_PROMPT = """You are a chart analysis expert. Generate exactly 5 question-answer pairs for this {chart_type} chart.

The questions must cover these 5 categories (one question per category):
1. STRUCTURAL: About chart title, labels, legend, or axes
2. COUNTING: About counting elements (bars, points, categories, etc.)
3. COMPARISON: About comparing values (highest, lowest, differences)
4. REASONING: About trends, patterns, or insights
5. EXTRACTION: About extracting specific data values

Respond with a JSON array only (no markdown, no extra text, complete the full JSON):
[
    {{
        "question": "question text",
        "answer": "detailed answer",
        "question_type": "structural|counting|comparison|reasoning|extraction"
    }},
    ...5 items total...
]

Rules:
- Answers should be specific and accurate based on the chart
- If exact values are not readable, use approximations with "approximately"
- Each question type must appear exactly once
- Keep answers concise but complete (1-2 sentences)
- IMPORTANT: Complete the full JSON array with all 5 items
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        """
        Initialize Gemini classifier.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Model name (defaults to GEMINI_MODEL env var)
            temperature: Generation temperature (lower = more deterministic)
            max_tokens: Maximum output tokens (increased to 4096 to avoid truncation)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the Gemini client
        self._init_client()
        
        logger.info(f"GeminiChartClassifier initialized | model={self.model}")
    
    def _init_client(self) -> None:
        """Initialize Google GenAI client (new SDK)."""
        try:
            from google import genai
            from google.genai import types
            
            # Create client with API key
            self.genai_client = genai.Client(api_key=self.api_key)
            self.genai_types = types
            
            # Store config for later use
            self.generation_config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
            
        except ImportError:
            raise ImportError(
                "google-genai not installed. "
                "Run: pip install google-genai"
            )
    
    def _load_image(self, image_path: Path) -> Any:
        """Load image for Gemini API."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load with PIL first to validate
        img = Image.open(image_path)
        
        # Upload to Gemini using new SDK
        uploaded_file = self.genai_client.files.upload(file=str(image_path))
        return uploaded_file
    
    def _load_image_as_base64(self, image_path: Path) -> Dict[str, Any]:
        """Load image as base64 for inline embedding."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Determine mime type
        suffix = image_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/png")
        
        return {
            "mime_type": mime_type,
            "data": base64.b64encode(image_data).decode("utf-8"),
        }
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response, handling markdown code blocks and truncation."""
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Try to fix truncated JSON
            fixed_text = self._try_fix_truncated_json(text)
            if fixed_text:
                try:
                    return json.loads(fixed_text)
                except json.JSONDecodeError:
                    pass
            
            logger.warning(f"JSON parse error: {e} | text={text[:100]}")
            return {}
    
    def _try_fix_truncated_json(self, text: str) -> Optional[str]:
        """Attempt to fix truncated JSON by closing open brackets/braces."""
        if not text:
            return None
        
        # Count open brackets
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        # If it looks like an array, try to close it
        if text.strip().startswith('[') and open_brackets > 0:
            # Find the last complete object
            last_complete = text.rfind('}')
            if last_complete > 0:
                # Check if there's a comma after
                remaining = text[last_complete + 1:].strip()
                if remaining.startswith(','):
                    text = text[:last_complete + 1]
                else:
                    text = text[:last_complete + 1]
                # Close remaining structures
                text += ']'
                return text
        
        # If it looks like an object
        if text.strip().startswith('{') and open_braces > 0:
            text += '}' * open_braces
            return text
        
        return None
    
    def classify_image(
        self,
        image_path: Path,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Classify an image as chart or non-chart.
        
        Args:
            image_path: Path to image file
            retry_count: Number of retries on API error
            retry_delay: Delay between retries (seconds)
            
        Returns:
            Classification result dict with is_chart, chart_type, confidence, etc.
        """
        image_path = Path(image_path)
        
        for attempt in range(retry_count):
            try:
                # Load image inline (more reliable than upload for small images)
                image_data = self._load_image_as_base64(image_path)
                
                # Create content with image and prompt using new SDK
                response = self.genai_client.models.generate_content(
                    model=self.model,
                    contents=[
                        self.genai_types.Part.from_bytes(
                            data=base64.b64decode(image_data["data"]),
                            mime_type=image_data["mime_type"],
                        ),
                        self.CLASSIFICATION_PROMPT,
                    ],
                    config=self.generation_config,
                )
                
                # Parse response
                result = self._parse_json_response(response.text)
                
                if result:
                    result["image_path"] = str(image_path)
                    result["model_used"] = self.model
                    logger.debug(
                        f"Classification complete | image={image_path.name} | "
                        f"is_chart={result.get('is_chart')} | type={result.get('chart_type')}"
                    )
                    return result
                
            except Exception as e:
                logger.warning(
                    f"Classification attempt {attempt + 1} failed | "
                    f"image={image_path.name} | error={e}"
                )
                if attempt < retry_count - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
        
        # Return failed result
        return {
            "is_chart": False,
            "chart_type": "none",
            "confidence": 0.0,
            "error": "Classification failed after retries",
            "image_path": str(image_path),
        }
    
    def generate_qa_pairs(
        self,
        image_path: Path,
        chart_type: str,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> List[Dict[str, str]]:
        """
        Generate QA pairs for a chart image.
        
        Args:
            image_path: Path to chart image
            chart_type: Type of chart (bar, line, pie, etc.)
            retry_count: Number of retries on API error
            retry_delay: Delay between retries
            
        Returns:
            List of QA pair dicts with question, answer, question_type
        """
        image_path = Path(image_path)
        prompt = self.QA_GENERATION_PROMPT.format(chart_type=chart_type)
        
        for attempt in range(retry_count):
            try:
                image_data = self._load_image_as_base64(image_path)
                
                # Use new SDK for generation
                response = self.genai_client.models.generate_content(
                    model=self.model,
                    contents=[
                        self.genai_types.Part.from_bytes(
                            data=base64.b64decode(image_data["data"]),
                            mime_type=image_data["mime_type"],
                        ),
                        prompt,
                    ],
                    config=self.generation_config,
                )
                
                # Parse response
                qa_pairs = self._parse_json_response(response.text)
                
                if isinstance(qa_pairs, list) and len(qa_pairs) > 0:
                    logger.debug(
                        f"QA generation complete | image={image_path.name} | "
                        f"pairs={len(qa_pairs)}"
                    )
                    return qa_pairs
                
            except Exception as e:
                logger.warning(
                    f"QA generation attempt {attempt + 1} failed | "
                    f"image={image_path.name} | error={e}"
                )
                if attempt < retry_count - 1:
                    time.sleep(retry_delay * (attempt + 1))
        
        # Return empty list on failure
        return []
    
    def process_image_full(
        self,
        image_path: Path,
    ) -> Dict[str, Any]:
        """
        Full processing: classify + generate QA if it's a chart.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Complete result with classification and QA pairs (if chart)
        """
        # Step 1: Classify
        classification = self.classify_image(image_path)
        
        result = {
            "image_path": str(image_path),
            "image_id": image_path.stem,
            "classification": classification,
            "qa_pairs": [],
            "metadata": {
                "model_used": self.model,
                "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        }
        
        # Step 2: Generate QA if it's a chart
        if classification.get("is_chart", False):
            chart_type = classification.get("chart_type", "unknown")
            qa_pairs = self.generate_qa_pairs(image_path, chart_type)
            result["qa_pairs"] = qa_pairs
            
        return result


class GeminiRateLimiter:
    """
    Rate limiter for Gemini API calls.
    
    Handles both RPM (requests per minute) and TPM (tokens per minute) limits.
    """
    
    def __init__(
        self,
        rpm: int = 60,
        tpm: int = 32000,
    ):
        """
        Initialize rate limiter.
        
        Args:
            rpm: Maximum requests per minute
            tpm: Maximum tokens per minute
        """
        self.rpm = rpm
        self.tpm = tpm
        self.request_times: List[float] = []
        self.token_counts: List[Tuple[float, int]] = []
    
    def wait_if_needed(self, estimated_tokens: int = 1000) -> None:
        """
        Wait if rate limits would be exceeded.
        
        Args:
            estimated_tokens: Estimated tokens for next request
        """
        now = time.time()
        minute_ago = now - 60
        
        # Clean old entries
        self.request_times = [t for t in self.request_times if t > minute_ago]
        self.token_counts = [(t, c) for t, c in self.token_counts if t > minute_ago]
        
        # Check RPM
        if len(self.request_times) >= self.rpm:
            wait_time = self.request_times[0] - minute_ago
            logger.debug(f"Rate limit (RPM): waiting {wait_time:.1f}s")
            time.sleep(wait_time + 0.1)
        
        # Check TPM
        current_tokens = sum(c for _, c in self.token_counts)
        if current_tokens + estimated_tokens >= self.tpm:
            oldest_time = min(t for t, _ in self.token_counts) if self.token_counts else now
            wait_time = oldest_time - minute_ago
            logger.debug(f"Rate limit (TPM): waiting {wait_time:.1f}s")
            time.sleep(wait_time + 0.1)
    
    def record_request(self, tokens_used: int = 1000) -> None:
        """Record a completed request."""
        now = time.time()
        self.request_times.append(now)
        self.token_counts.append((now, tokens_used))


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gemini_classifier.py <image_path>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    classifier = GeminiChartClassifier()
    
    result = classifier.process_image_full(image_path)
    print(json.dumps(result, indent=2))
