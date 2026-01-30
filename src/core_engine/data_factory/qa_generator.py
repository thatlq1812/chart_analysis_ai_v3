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
    ChartVerification,
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

GENERATION_PROMPT = """Analyze this image (labeled as '{chart_type}' chart).

{caption_context}

Return JSON with:
1. "verification": Chart validity check
2. "qa_pairs": 6-8 question-answer pairs

JSON structure:
{{
  "verification": {{
    "is_valid_chart": boolean,
    "actual_chart_type": "line|bar|scatter|pie|heatmap|histogram|area|box|table|diagram|not_a_chart|other",
    "chart_quality": "high|medium|low|unreadable",
    "verification_notes": "string or null"
  }},
  "qa_pairs": [
    {{
      "question": "string",
      "answer": "string",
      "question_type": "structural|extraction|counting|comparison|trend|range|interpolation|percentage_change|threshold|multi_hop|why_reasoning",
      "difficulty": 1-5,
      "answer_value": "value or null",
      "is_answerable": boolean,
      "confidence": 0.0-1.0
    }}
  ]
}}

Rules:
- If NOT a valid chart: is_valid_chart=false, qa_pairs=[]
- If actual type differs from '{chart_type}': note in verification_notes
- Include mix: 2 easy (structural/counting), 3-4 medium (comparison/trend), 2 hard (interpolation/reasoning)

Return ONLY valid JSON."""

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
                        max_output_tokens=8192,  # Increased for complete JSON
                        response_mime_type="application/json",
                    ),
                )
                
                # Parse JSON response
                verification_data, raw_qa_pairs = self._parse_response(response.text)
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
        
        # Build verification object
        verification = None
        if verification_data:
            actual_type = verification_data.get("actual_chart_type", chart_type)
            type_matches = actual_type.lower() == chart_type.lower()
            
            verification = ChartVerification(
                is_valid_chart=verification_data.get("is_valid_chart", True),
                actual_chart_type=actual_type,
                chart_quality=verification_data.get("chart_quality", "medium"),
                verification_notes=verification_data.get("verification_notes"),
                type_matches_folder=type_matches,
            )
            
            # Log mismatches for data cleaning
            if not type_matches:
                logger.warning(
                    f"Type mismatch | image={image_path.name} | "
                    f"folder={chart_type} | gemini={actual_type}"
                )
            if not verification.is_valid_chart:
                logger.warning(
                    f"Invalid chart | image={image_path.name} | "
                    f"notes={verification.verification_notes}"
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
            verification=verification,
            caption=caption,
            context_text=context,
            paper_id=paper_id,
            qa_pairs=qa_pairs,
            qa_distribution=qa_distribution,
            avg_difficulty=avg_difficulty,
            generator_model=self.model_name,
        )
    
    def _parse_response(self, response_text: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Parse JSON response from model.
        
        Returns:
            Tuple of (verification_dict, qa_pairs_list)
        """
        # Clean response
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = re.sub(r"```(?:json)?\s*", "", text)
            text = text.rstrip("`")
        
        # Remove trailing commas before } or ] (common LLM JSON error)
        text = re.sub(r",\s*([}\]])", r"\1", text)
        
        verification = {}
        qa_pairs = []
        
        try:
            data = json.loads(text)
            
            if isinstance(data, dict):
                # New format with verification
                verification = data.get("verification", {})
                qa_pairs = data.get("qa_pairs", [])
            elif isinstance(data, list):
                # Old format (backward compatible)
                qa_pairs = data
            else:
                logger.warning("Unexpected response structure")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            
            # Try to extract partial data using regex
            try:
                # Try to find verification block
                verif_match = re.search(r'"verification"\s*:\s*(\{[^}]+\})', text)
                if verif_match:
                    verification = json.loads(verif_match.group(1))
                    logger.info("Extracted verification from partial JSON")
                    
                # Try to find qa_pairs array
                # This is a simple approach - find array content
                qa_match = re.search(r'"qa_pairs"\s*:\s*\[', text)
                if qa_match:
                    logger.info("Found qa_pairs key but couldn't parse full JSON")
            except Exception as ex:
                logger.debug(f"Partial extraction failed: {ex}")
            
            logger.debug(f"Response text (first 1000 chars): {text[:1000]}...")
        
        return verification, qa_pairs
    
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
    
    def _process_single_sample(
        self,
        sample: Dict[str, Any],
        index: int,
        total: int,
        output_dir: Optional[Path] = None,
    ) -> Optional[ChartQASampleV2]:
        """Process a single sample and save to individual JSON file.
        
        Args:
            sample: Sample dict with image_path, chart_type, etc.
            index: Current index (for logging)
            total: Total samples (for logging)
            output_dir: Base output directory (creates chart_type subdirs)
            
        Returns:
            ChartQASampleV2 or None if failed
        """
        image_path = Path(sample["image_path"])
        chart_type = sample.get("chart_type", "unknown")
        image_id = image_path.stem
        
        # Skip uncertain if configured
        if chart_type == "uncertain" and not self.config.include_uncertain:
            logger.debug(f"Skipping uncertain chart: {image_path.name}")
            return None
        
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None
        
        # Check if already processed (skip if output file exists)
        if output_dir:
            output_file = output_dir / chart_type / f"{image_id}.json"
            if output_file.exists():
                logger.debug(f"[{index+1}/{total}] SKIP (already exists): {image_id}")
                # Load existing result to return
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return ChartQASampleV2(**data)
                except Exception as e:
                    logger.warning(f"Failed to load existing {output_file}: {e}")
                    # Continue to regenerate
        
        try:
            qa_sample = self.generate_qa_pairs(
                image_path=image_path,
                chart_type=chart_type,
                caption=sample.get("caption"),
                context=sample.get("context_text"),
                metadata=sample,
            )
            
            # Save individual JSON file immediately
            if output_dir:
                self._save_single_result(qa_sample, output_dir)
            
            logger.info(
                f"[{index+1}/{total}] Generated {len(qa_sample.qa_pairs)} QA pairs "
                f"for {image_path.name} | chart_type={chart_type}"
            )
            return qa_sample
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            return None
    
    def _save_single_result(
        self,
        sample: ChartQASampleV2,
        output_dir: Path,
    ) -> Path:
        """Save a single sample to its own JSON file.
        
        Structure:
            output_dir/
            ├── line/
            │   ├── img_001.json
            │   └── img_002.json
            ├── bar/
            │   └── img_003.json
            └── ...
        
        Args:
            sample: The QA sample to save
            output_dir: Base output directory
            
        Returns:
            Path to saved file
        """
        # Create chart_type subdirectory
        type_dir = output_dir / sample.chart_type
        type_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to individual file
        output_file = type_dir / f"{sample.image_id}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sample.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        
        return output_file
    
    def generate_batch(
        self,
        samples: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
        checkpoint_every: Optional[int] = None,
        rate_limit_delay: float = 1.0,
        max_workers: int = 1,
        start_index: int = 0,
        skip_existing: bool = True,
    ) -> List[ChartQASampleV2]:
        """Generate QA pairs for a batch of samples.
        
        Each image is saved to its own JSON file immediately after processing:
            output_dir/{chart_type}/{image_id}.json
        
        This design ensures:
        - No data loss on interruption
        - Easy resume (skip_existing=True)
        - Parallel-safe (no file conflicts)
        - No batch file overwrites
        
        Args:
            samples: List of sample dicts with image_path, chart_type, etc.
            output_dir: Directory to save outputs (overrides config)
            checkpoint_every: Log progress every N samples
            rate_limit_delay: Delay between API calls in seconds (for sequential mode)
            max_workers: Number of parallel workers (1=sequential, >1=parallel)
            start_index: Starting index of this batch (for logging)
            skip_existing: Skip images that already have output files
            
        Returns:
            List of ChartQASampleV2 with generated QA pairs
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_every = checkpoint_every or self.config.checkpoint_every
        
        results: List[ChartQASampleV2] = []
        success = 0
        failed = 0
        skipped = 0
        start_time = time.time()
        total = len(samples)
        
        # Count existing files for skip estimation
        if skip_existing:
            existing_count = 0
            for sample in samples:
                chart_type = sample.get("chart_type", "unknown")
                image_id = Path(sample["image_path"]).stem
                if (output_dir / chart_type / f"{image_id}.json").exists():
                    existing_count += 1
            logger.info(f"Found {existing_count}/{total} already processed (will skip)")
        
        logger.info(f"Output structure: {output_dir}/{{chart_type}}/{{image_id}}.json")
        
        # Choose execution mode
        if max_workers > 1:
            # ============ PARALLEL MODE ============
            logger.info(f"Starting PARALLEL processing with {max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(
                        self._process_single_sample, sample, idx, total, output_dir
                    ): idx 
                    for idx, sample in enumerate(samples)
                }
                
                # Process completed futures
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        qa_sample = future.result()
                        if qa_sample:
                            results.append(qa_sample)
                            success += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"Worker error for sample {idx}: {e}")
                        failed += 1
                    
                    # Progress with ETA
                    completed = success + failed
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed if completed else 0
                    remaining = avg_time * (total - completed)
                    
                    if completed % checkpoint_every == 0 or completed == total:
                        logger.info(
                            f"Progress: {completed}/{total} | "
                            f"success={success} failed={failed} | "
                            f"ETA: {remaining/60:.1f}min"
                        )
        else:
            # ============ SEQUENTIAL MODE ============
            logger.info("Starting SEQUENTIAL processing with rate limiting")
            
            for i, sample in enumerate(samples):
                qa_sample = self._process_single_sample(sample, i, total, output_dir)
                
                if qa_sample:
                    results.append(qa_sample)
                    success += 1
                else:
                    failed += 1
                
                # Progress with ETA
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (total - i - 1)
                
                if (i + 1) % checkpoint_every == 0 or i == total - 1:
                    logger.info(
                        f"Progress: [{i+1}/{total}] | "
                        f"success={success} failed={failed} | "
                        f"ETA: {remaining/60:.1f}min"
                    )
                
                # Rate limiting to avoid API throttling
                if rate_limit_delay > 0 and i < total - 1:
                    time.sleep(rate_limit_delay)
        
        total_time = time.time() - start_time
        logger.info(
            f"Batch complete | success={success} | failed={failed} | "
            f"time={total_time/60:.1f}min | avg={total_time/max(1,success):.1f}s/image"
        )
        
        # Summary of output structure
        self._log_output_summary(output_dir)
        
        return results
    
    def _log_output_summary(self, output_dir: Path) -> None:
        """Log summary of output files by chart type."""
        summary = {}
        for type_dir in output_dir.iterdir():
            if type_dir.is_dir():
                count = len(list(type_dir.glob("*.json")))
                if count > 0:
                    summary[type_dir.name] = count
        
        if summary:
            logger.info("Output summary by chart type:")
            for chart_type, count in sorted(summary.items(), key=lambda x: -x[1]):
                logger.info(f"  {chart_type}: {count} files")
            logger.info(f"  TOTAL: {sum(summary.values())} files")
    
    def save_results(
        self,
        results: List[ChartQASampleV2],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, int]:
        """Save multiple results to individual JSON files.
        
        Args:
            results: List of QA samples to save
            output_dir: Base output directory
            
        Returns:
            Dict with count of files saved per chart_type
        """
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_counts = {}
        for sample in results:
            self._save_single_result(sample, output_dir)
            chart_type = sample.chart_type
            saved_counts[chart_type] = saved_counts.get(chart_type, 0) + 1
        
        logger.info(f"Saved {len(results)} files to {output_dir}")
        return saved_counts
    
    def load_results(
        self,
        output_dir: Optional[Path] = None,
        chart_types: Optional[List[str]] = None,
    ) -> List[ChartQASampleV2]:
        """Load all saved results from output directory.
        
        Args:
            output_dir: Base output directory
            chart_types: Optional list of chart types to load (None = all)
            
        Returns:
            List of ChartQASampleV2 loaded from files
        """
        output_dir = Path(output_dir or self.config.output_dir)
        results = []
        
        # Find all chart type directories
        type_dirs = []
        if chart_types:
            type_dirs = [output_dir / ct for ct in chart_types if (output_dir / ct).exists()]
        else:
            type_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        
        for type_dir in type_dirs:
            for json_file in type_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    results.append(ChartQASampleV2(**data))
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")
        
        logger.info(f"Loaded {len(results)} samples from {output_dir}")
        return results
    
    def get_processed_ids(
        self,
        output_dir: Optional[Path] = None,
    ) -> set:
        """Get set of image IDs that have already been processed.
        
        Args:
            output_dir: Base output directory
            
        Returns:
            Set of image_id strings
        """
        output_dir = Path(output_dir or self.config.output_dir)
        processed = set()
        
        for type_dir in output_dir.iterdir():
            if type_dir.is_dir():
                for json_file in type_dir.glob("*.json"):
                    processed.add(json_file.stem)
        
        return processed


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
