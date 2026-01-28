"""
Chart QA Generator V2 - Research-Grade Question Generation

Generate enhanced QA pairs for training vision-language models with:
- Multi-level reasoning (shallow to deep)
- Visual grounding references
- Structured inference chains
- Chart-type specific templates
- Uncertainty handling

Uses Gemini Pro Vision for generation.

Note: This is a DATA FACTORY tool, not a pipeline stage.
It consumes Stage 3/4 outputs to generate SLM training data.

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 2.0.0 | 2026-01-28 | That Le | Research-grade QA generator |
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from core_engine.schemas.qa_schemas import (
    ChartQASampleV2,
    ConfidenceLevel,
    InferenceInfo,
    PointReference,
    QAPairV2,
    QuestionType,
    QUESTION_DIFFICULTY,
    ReasoningMethod,
    ReasoningStep,
    VisualGrounding,
    ChartRegion,
)

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# GEMINI PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert Chart Analysis AI specialized in generating research-grade question-answer pairs for training vision-language models.

Your task is to analyze chart images and generate DIVERSE questions that test different cognitive abilities:

1. **SHALLOW (Difficulty 1-2)**: Direct reading from chart
   - Structural: titles, labels, units
   - Extraction: reading specific values
   - Counting: number of elements

2. **INTERMEDIATE (Difficulty 2-3)**: Visual understanding
   - Comparison: which is higher/lower
   - Trend: increasing/decreasing patterns
   - Range: min/max values

3. **DEEP (Difficulty 3-5)**: Reasoning required
   - Interpolation: estimate values between tick marks
   - Percentage change: calculate relative changes
   - Threshold: find where values cross a line
   - Optimal point: identify trade-off points
   - Multi-hop: combine multiple facts

4. **CONCEPTUAL (Difficulty 4-5)**: Domain knowledge
   - Why-reasoning: explain patterns
   - Caption-aware: link caption to chart content

CRITICAL RULES:
- Generate 8-12 questions per chart
- Include at least 2 DEEP reasoning questions
- For unclear charts, include questions with "cannot be determined" answers
- Provide reasoning steps for complex questions
- Reference specific visual elements (tick marks, regions, data points)
- Be precise with numerical answers (include units)
"""

GENERATION_PROMPT = """Analyze this {chart_type} chart and generate research-grade QA pairs.

{caption_context}

Generate a JSON array of QA pairs. Each pair MUST include:
1. "question": The question text
2. "answer": Detailed answer with evidence
3. "question_type": One of: structural, extraction, counting, comparison, trend, range, interpolation, extrapolation, percentage_change, threshold, optimal_point, multi_hop, why_reasoning, caption_aware, ambiguity
4. "difficulty": 1-5 scale
5. "answer_value": The extracted/computed value (number, string, or null)
6. "answer_unit": Unit if applicable (or null)
7. "is_answerable": true/false
8. "reasoning_method": One of: direct_read, interpolation, extrapolation, calculation, approximation, inference, comparison, aggregation, pattern, cannot_determine
9. "confidence": 0.0-1.0
10. "reasoning_steps": Array of steps like {{"step": 1, "action": "...", "observation": "..."}}
11. "visual_references": {{"regions": [...], "tick_marks": [...], "points": [...]}}

IMPORTANT:
- For line charts: Include interpolation, threshold detection, trend analysis
- For bar charts: Include comparison, percentage calculations, ranking
- For scatter: Include correlation, clustering, outlier detection
- For pie: Include proportion comparison, aggregation
- For heatmap: Include pattern detection, max/min location

Generate exactly 10 QA pairs with this distribution:
- 2-3 shallow (structural, extraction, counting)
- 3-4 intermediate (comparison, trend, range)
- 3-4 deep (interpolation, percentage_change, threshold, multi_hop)
- 1-2 conceptual (why_reasoning) if applicable

Return ONLY valid JSON array, no markdown formatting."""

CAPTION_CONTEXT_TEMPLATE = """
Figure Caption: "{caption}"
Context: {context}

Use this caption to generate caption-aware questions that connect the visual data to the described concepts.
"""


# =============================================================================
# CONFIGURATION
# =============================================================================

class QAGeneratorConfig(BaseModel):
    """Configuration for QA Generator."""
    
    # API Configuration
    api_key: Optional[str] = Field(
        None, 
        description="Gemini API key. If None, reads from GEMINI_API_KEY env var"
    )
    model_name: str = Field(
        "gemini-3-flash-preview",
        description="Gemini model to use"
    )
    temperature: float = Field(
        0.7, ge=0.0, le=2.0,
        description="Generation temperature (higher = more diverse)"
    )
    max_retries: int = Field(
        3, ge=1,
        description="Number of retry attempts per image"
    )
    
    # Generation Settings
    questions_per_chart: int = Field(
        10, ge=5, le=20,
        description="Target number of QA pairs per chart"
    )
    include_uncertain: bool = Field(
        False,
        description="Whether to process 'uncertain' chart type"
    )
    
    # Output Settings
    output_dir: Path = Field(
        Path("data/academic_dataset/chart_qa_v2/generated"),
        description="Output directory for generated QA"
    )
    checkpoint_every: int = Field(
        100,
        description="Save checkpoint every N samples"
    )
    shard_size: int = Field(
        1000,
        description="Number of samples per shard file"
    )


# =============================================================================
# QA GENERATOR CLASS
# =============================================================================

class ChartQAGeneratorV2:
    """
    Enhanced QA generator using Gemini Pro Vision.
    
    This is a DATA FACTORY component, NOT a pipeline stage.
    It generates training data for SLM from chart images.
    
    Example:
        config = QAGeneratorConfig(api_key="...")
        generator = ChartQAGeneratorV2(config)
        
        samples = [
            {"image_path": "path/to/chart.png", "chart_type": "line"},
            ...
        ]
        results = generator.generate_batch(samples, output_dir)
    """
    
    def __init__(self, config: Union[QAGeneratorConfig, Dict[str, Any]]):
        """Initialize the generator.
        
        Args:
            config: Generator configuration
        """
        if isinstance(config, dict):
            config = QAGeneratorConfig(**config)
        
        self.config = config
        
        # Get API key from config or environment
        api_key = config.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in config or environment")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_retries = config.max_retries
        
        logger.info(f"Initialized ChartQAGeneratorV2 with model={config.model_name}")
    
    def generate_qa_pairs(
        self,
        image_path: Path,
        chart_type: str,
        caption: Optional[str] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChartQASampleV2:
        """Generate QA pairs for a chart image.
        
        Args:
            image_path: Path to chart image
            chart_type: Type of chart (bar, line, pie, etc.)
            caption: Figure caption if available
            context: Surrounding text context
            metadata: Additional metadata
            
        Returns:
            ChartQASampleV2 with generated QA pairs
        """
        image_path = Path(image_path)
        
        # Load image
        try:
            image = Image.open(image_path)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
        
        # Build prompt
        caption_context = ""
        if caption:
            caption_context = CAPTION_CONTEXT_TEMPLATE.format(
                caption=caption,
                context=context or "No additional context available."
            )
        
        prompt = GENERATION_PROMPT.format(
            chart_type=chart_type,
            caption_context=caption_context,
        )
        
        # Generate with retries
        raw_qa_pairs = None
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Upload image for Gemini
                uploaded_file = self.client.files.upload(file=image_path)
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_uri(
                                    file_uri=uploaded_file.uri,
                                    mime_type=uploaded_file.mime_type,
                                ),
                                types.Part.from_text(text=prompt),
                            ],
                        ),
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=self.temperature,
                        max_output_tokens=4096,
                        response_mime_type="application/json",
                    ),
                )
                
                # Parse JSON response
                raw_qa_pairs = self._parse_response(response.text)
                break
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Generation attempt {attempt + 1} failed: {e}"
                )
        
        if raw_qa_pairs is None:
            logger.error(f"All generation attempts failed: {last_error}")
            # Return empty sample
            return ChartQASampleV2(
                image_id=image_path.stem,
                image_path=str(image_path),
                chart_type=chart_type,
                caption=caption,
                context_text=context,
                qa_pairs=[],
                generator_model=self.model_name,
            )
        
        # Convert to QAPairV2 objects
        qa_pairs = self._convert_to_qa_pairs(raw_qa_pairs, chart_type)
        
        # Calculate statistics
        qa_distribution = {}
        total_difficulty = 0
        for qa in qa_pairs:
            qt = qa.question_type.value
            qa_distribution[qt] = qa_distribution.get(qt, 0) + 1
            total_difficulty += qa.difficulty
        
        avg_difficulty = total_difficulty / len(qa_pairs) if qa_pairs else 0
        
        # Build sample
        image_id = image_path.stem
        paper_id = None
        if metadata and "parent_paper_id" in metadata:
            paper_id = metadata["parent_paper_id"]
        
        return ChartQASampleV2(
            image_id=image_id,
            image_path=str(image_path),
            chart_type=chart_type,
            caption=caption,
            context_text=context,
            paper_id=paper_id,
            qa_pairs=qa_pairs,
            qa_distribution=qa_distribution,
            avg_difficulty=avg_difficulty,
            generator_model=self.model_name,
        )
    
    def _parse_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse JSON response from model."""
        # Clean response
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = re.sub(r"```(?:json)?\s*", "", text)
            text = text.rstrip("`")
        
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "qa_pairs" in data:
                return data["qa_pairs"]
            else:
                logger.warning("Unexpected response structure")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Response text: {text[:500]}...")
            return []
    
    def _convert_to_qa_pairs(
        self,
        raw_pairs: List[Dict[str, Any]],
        chart_type: str,
    ) -> List[QAPairV2]:
        """Convert raw JSON to QAPairV2 objects."""
        qa_pairs = []
        
        for raw in raw_pairs:
            try:
                # Parse question type
                qt_str = raw.get("question_type", "extraction").lower()
                try:
                    question_type = QuestionType(qt_str)
                except ValueError:
                    question_type = QuestionType.EXTRACTION
                
                # Parse reasoning method
                rm_str = raw.get("reasoning_method", "direct_read").lower()
                try:
                    reasoning_method = ReasoningMethod(rm_str)
                except ValueError:
                    reasoning_method = ReasoningMethod.DIRECT_READ
                
                # Parse confidence
                confidence = float(raw.get("confidence", 0.8))
                confidence = max(0.0, min(1.0, confidence))
                
                # Determine confidence level
                if confidence >= 0.9:
                    conf_level = ConfidenceLevel.HIGH
                elif confidence >= 0.7:
                    conf_level = ConfidenceLevel.MEDIUM
                elif confidence >= 0.5:
                    conf_level = ConfidenceLevel.LOW
                else:
                    conf_level = ConfidenceLevel.UNCERTAIN
                
                # Parse reasoning steps
                reasoning_steps = []
                for step in raw.get("reasoning_steps", []):
                    if isinstance(step, dict):
                        reasoning_steps.append(ReasoningStep(
                            step_number=step.get("step", len(reasoning_steps) + 1),
                            action=step.get("action", ""),
                            observation=step.get("observation", ""),
                            intermediate_result=step.get("result"),
                        ))
                
                # Parse visual references
                visual_refs = raw.get("visual_references", {})
                regions = []
                for r in visual_refs.get("regions", []):
                    try:
                        regions.append(ChartRegion(r.lower()))
                    except ValueError:
                        regions.append(ChartRegion.PLOT_AREA)
                
                points = []
                for p in visual_refs.get("points", []):
                    if isinstance(p, dict):
                        points.append(PointReference(
                            x_value=p.get("x"),
                            y_value=p.get("y"),
                        ))
                
                visual_grounding = VisualGrounding(
                    chart_type=chart_type,
                    regions_referenced=regions or [ChartRegion.PLOT_AREA],
                    points_referenced=points,
                    tick_marks_used=visual_refs.get("tick_marks", []),
                )
                
                # Build inference info
                inference = InferenceInfo(
                    method=reasoning_method,
                    confidence=confidence,
                    confidence_level=conf_level,
                    reasoning_steps=reasoning_steps,
                )
                
                # Build QA pair
                difficulty = raw.get("difficulty", QUESTION_DIFFICULTY.get(question_type, 2))
                is_answerable = raw.get("is_answerable", True)
                
                qa_pair = QAPairV2(
                    question=raw.get("question", ""),
                    answer=raw.get("answer", ""),
                    question_type=question_type,
                    difficulty=difficulty,
                    answer_value=raw.get("answer_value"),
                    answer_unit=raw.get("answer_unit"),
                    visual_grounding=visual_grounding,
                    inference=inference,
                    is_answerable=is_answerable,
                    unanswerable_reason=raw.get("unanswerable_reason"),
                )
                
                qa_pairs.append(qa_pair)
                
            except Exception as e:
                logger.warning(f"Failed to parse QA pair: {e}")
                continue
        
        return qa_pairs
    
    def generate_batch(
        self,
        samples: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
        checkpoint_every: Optional[int] = None,
    ) -> List[ChartQASampleV2]:
        """Generate QA pairs for a batch of samples.
        
        Args:
            samples: List of sample dicts with image_path, chart_type, etc.
            output_dir: Directory to save outputs (overrides config)
            checkpoint_every: Save checkpoint every N samples (overrides config)
            
        Returns:
            List of ChartQASampleV2 with generated QA pairs
        """
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_every = checkpoint_every or self.config.checkpoint_every
        
        results: List[ChartQASampleV2] = []
        success = 0
        failed = 0
        
        for i, sample in enumerate(samples):
            image_path = Path(sample["image_path"])
            chart_type = sample.get("chart_type", "unknown")
            
            # Skip uncertain if configured
            if chart_type == "uncertain" and not self.config.include_uncertain:
                logger.debug(f"Skipping uncertain chart: {image_path.name}")
                continue
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                failed += 1
                continue
            
            try:
                qa_sample = self.generate_qa_pairs(
                    image_path=image_path,
                    chart_type=chart_type,
                    caption=sample.get("caption"),
                    context=sample.get("context_text"),
                    metadata=sample,
                )
                
                results.append(qa_sample)
                success += 1
                
                logger.info(
                    f"[{i+1}/{len(samples)}] Generated {len(qa_sample.qa_pairs)} QA pairs "
                    f"for {image_path.name} | chart_type={chart_type}"
                )
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                failed += 1
            
            # Save checkpoint
            if (i + 1) % checkpoint_every == 0:
                self._save_checkpoint(results, output_dir, i + 1)
        
        # Save final results
        self._save_results(results, output_dir)
        
        logger.info(f"Batch complete | success={success} | failed={failed}")
        
        return results
    
    def _save_checkpoint(
        self, 
        results: List[ChartQASampleV2], 
        output_dir: Path,
        index: int,
    ) -> None:
        """Save checkpoint to disk."""
        checkpoint_path = output_dir / f"checkpoint_{index}.json"
        data = [r.model_dump() for r in results]
        
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_results(
        self,
        results: List[ChartQASampleV2],
        output_dir: Path,
    ) -> Path:
        """Save final results to JSON file."""
        output_file = output_dir / "qa_pairs_v2.json"
        data = [r.model_dump() for r in results]
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Results saved: {output_file}")
        return output_file
    
    def save_results(
        self,
        results: List[ChartQASampleV2],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Public method to save results."""
        output_dir = Path(output_path or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return self._save_results(results, output_dir)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for QA generation."""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Generate Chart QA pairs v2")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input JSON file with samples or directory with images"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/academic_dataset/chart_qa_v2"),
        help="Output directory"
    )
    parser.add_argument(
        "--chart-type", "-t",
        type=str,
        default=None,
        help="Chart type (if processing directory)"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit number of samples to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model to use"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    # Initialize config
    config = QAGeneratorConfig(model_name=args.model, output_dir=args.output)
    generator = ChartQAGeneratorV2(config)
    
    # Load samples
    if args.input.is_file():
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
            samples = data if isinstance(data, list) else data.get("samples", [])
    else:
        # Process directory of images
        if not args.chart_type:
            logger.error("--chart-type required when processing directory")
            return
        
        samples = []
        for img_path in args.input.glob("*.png"):
            samples.append({
                "image_path": str(img_path),
                "chart_type": args.chart_type,
            })
    
    if args.limit:
        samples = samples[:args.limit]
    
    logger.info(f"Processing {len(samples)} samples...")
    
    # Generate QA pairs
    results = generator.generate_batch(samples)
    
    logger.info(f"Complete! Generated QA for {len(results)} charts")


if __name__ == "__main__":
    main()
