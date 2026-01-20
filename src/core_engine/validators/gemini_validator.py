"""
Gemini Vision API Validator

Validate chart detections using Google's Gemini Vision model.
Requires GEMINI_API_KEY in environment variables.
"""

import os
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class GeminiValidator:
    """
    Validate detected charts using Gemini Vision API.
    
    Usage:
        validator = GeminiValidator()
        result = validator.validate_image(image_path)
        # Returns:
        # {
        #     "is_chart": bool,
        #     "chart_type": str,  # bar, line, pie, scatter, text, blank, etc
        #     "confidence": float,
        #     "description": str,
        #     "error": str (if any)
        # }
    """
    
    API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini validator.
        
        Args:
            api_key: Gemini API key. If None, uses GEMINI_API_KEY env var.
        
        Raises:
            ValueError: If no API key found
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass it directly."
            )
        
        logger.info("Gemini validator initialized")
    
    def validate_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Validate an image using Gemini Vision.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary with validation results
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                return {
                    "is_chart": False,
                    "chart_type": "unknown",
                    "confidence": 0.0,
                    "description": "",
                    "error": f"File not found: {image_path}"
                }
            
            # Encode image
            image_data = self._encode_image(image_path)
            
            # Call Gemini API
            response = self._call_gemini_api(image_data)
            
            # Parse response
            result = self._parse_response(response)
            
            logger.info(f"Validated {image_path.name}: {result['chart_type']} (conf={result['confidence']:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return {
                "is_chart": False,
                "chart_type": "error",
                "confidence": 0.0,
                "description": "",
                "error": str(e)
            }
    
    def _encode_image(self, image_path: Path) -> str:
        """
        Encode image to base64.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Base64 encoded image
        
        Raises:
            ValueError: If file is not a valid image
        """
        try:
            # Verify it's an image
            img = Image.open(image_path)
            img.verify()  # Verify it's a valid image
            
            # Read and encode
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                return base64.standard_b64encode(image_bytes).decode("utf-8")
        
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")
    
    def _call_gemini_api(self, image_data: str) -> Dict[str, Any]:
        """
        Call Gemini API with image.
        
        Args:
            image_data: Base64 encoded image
        
        Returns:
            API response
        
        Raises:
            Exception: If API call fails
        """
        prompt = """Analyze this image and determine:
1. Is it a chart/graph? (yes/no)
2. If yes, what type? (bar, line, pie, scatter, area, heatmap, other)
3. Confidence level (0-1)
4. Brief description (1-2 sentences)

If image is blank/mostly white/empty, say it's NOT a chart.

Respond in JSON format:
{
    "is_chart": bool,
    "chart_type": "string",
    "confidence": float,
    "description": "string"
}
"""
        
        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": image_data
                            }
                        }
                    ]
                }
            ]
        }
        
        url = f"{self.API_ENDPOINT}?key={self.api_key}"
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        return response.json()
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Gemini API response.
        
        Args:
            response: Raw API response
        
        Returns:
            Parsed validation result
        """
        try:
            # Extract text from response
            content = response.get("candidates", [{}])[0].get("content", {})
            parts = content.get("parts", [{}])
            text = parts[0].get("text", "")
            
            # Parse JSON from response
            import json
            
            # Try to extract JSON block
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                parsed = json.loads(json_str)
            else:
                # Fallback if JSON not found
                parsed = {
                    "is_chart": "chart" in text.lower(),
                    "chart_type": "unknown",
                    "confidence": 0.5,
                    "description": text
                }
            
            # Ensure all keys exist
            return {
                "is_chart": parsed.get("is_chart", False),
                "chart_type": parsed.get("chart_type", "unknown").lower(),
                "confidence": float(parsed.get("confidence", 0.0)),
                "description": parsed.get("description", ""),
                "error": None
            }
        
        except Exception as e:
            logger.error(f"Failed to parse API response: {str(e)}")
            return {
                "is_chart": False,
                "chart_type": "error",
                "confidence": 0.0,
                "description": "",
                "error": f"Parse error: {str(e)}"
            }


def validate_detection_batch(image_paths: list, output_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Validate a batch of detected charts.
    
    Args:
        image_paths: List of image paths to validate
        output_file: Optional output file to save results
    
    Returns:
        Dictionary with validation results for each image
    """
    validator = GeminiValidator()
    results = {}
    
    for img_path in image_paths:
        img_name = Path(img_path).name
        results[img_name] = validator.validate_image(img_path)
        logger.info(f"  {img_name}: {results[img_name]['chart_type']}")
    
    # Save results if requested
    if output_file:
        import json
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(results, indent=2))
        logger.info(f"Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        validator = GeminiValidator()
        result = validator.validate_image(image_path)
        print(f"Image: {image_path}")
        print(f"Result: {result}")
    else:
        print("Usage: python gemini_validator.py <image_path>")
