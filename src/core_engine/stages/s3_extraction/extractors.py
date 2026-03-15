"""
Chart Extractor Backends for Stage 3

Pluggable VLM chart-to-table extraction backends designed for ablation
experiments comparing different model families on chart derendering accuracy.

Backend options (used in ExtractionConfig.extractor_backend):
    - deplot     : google/deplot (recommended default)
    - pix2struct : google/pix2struct-base (ablation baseline, no fine-tuning)
    - matcha     : google/matcha-base (enhanced math+chart reasoning)
    - svlm       : Qwen/Qwen2-VL-2B-Instruct (zero-shot large VLM)

Usage:
    from .extractors import create_extractor, BackendType

    extractor = create_extractor("deplot")
    result = extractor.extract(image_bgr, chart_id="chart_001")
    if result and result.extraction_confidence > 0:
        # result.headers, result.rows, result.records are populated

Ablation context:
    - deplot     : Best for synthetic chart data (Lin et al. 2022, ACL 2023)
    - matcha     : Enhanced for complex scales and formulas (Liu et al. 2022)
    - pix2struct : Zero-task baseline (no chart derendering fine-tuning)
    - svlm       : Zero-shot SOTA, largest model, highest VRAM requirement

References:
    DePlot:    Lin et al. 2022, https://arxiv.org/abs/2212.10505
    MatCha:    Liu et al. 2022, https://arxiv.org/abs/2212.09662
    Pix2Struct: Lee et al. 2023, https://arxiv.org/abs/2210.03347
    Qwen2-VL:  Wang et al. 2024, https://arxiv.org/abs/2409.12191
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...schemas.stage_outputs import Pix2StructResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


class BackendType(str, Enum):
    """Available chart extraction backends."""

    DEPLOT = "deplot"
    PIX2STRUCT = "pix2struct"
    MATCHA = "matcha"
    SVLM = "svlm"


_DEFAULT_MODELS: Dict[str, str] = {
    BackendType.DEPLOT: "google/deplot",
    BackendType.PIX2STRUCT: "google/pix2struct-base",
    BackendType.MATCHA: "google/matcha-base",
    BackendType.SVLM: "Qwen/Qwen2-VL-2B-Instruct",
}


# ---------------------------------------------------------------------------
# DePlot linearized table parser (shared by all Pix2Struct-family models)
# ---------------------------------------------------------------------------


def _is_numeric(s: str) -> bool:
    """Return True if the string represents a number (int, float, %, comma-sep)."""
    try:
        float(s.replace(",", "").replace("%", "").strip())
        return True
    except ValueError:
        return False


def _parse_deplot_output(text: str) -> Tuple[List[str], List[List[str]]]:
    """
    Parse DePlot / MatCha linearized table output into (headers, rows).

    All Pix2Struct-family models produce output in this format:
        TITLE | <chart_title><0x0A>
        col0  | col1 | col2 <0x0A>
        val0  | val1 | val2 <0x0A> ...

    Where <0x0A> is the special newline token in the Pix2Struct vocabulary.

    Args:
        text: Raw decoded string from the model.

    Returns:
        Tuple of (header_list, data_rows). header_list is empty if no header
        row was detected.
    """
    text = text.replace("<0x0A>", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    non_title = [line for line in lines if not line.upper().startswith("TITLE")]

    if not non_title:
        return [], []

    def _is_data_row(line: str) -> bool:
        parts = [p.strip() for p in line.split("|")]
        return any(_is_numeric(p) for p in parts)

    first_data_idx: Optional[int] = None
    for i, line in enumerate(non_title):
        if _is_data_row(line):
            first_data_idx = i
            break

    if first_data_idx is None:
        return [], []

    header_lines = non_title[:first_data_idx]
    data_lines = [line for line in non_title[first_data_idx:] if "|" in line]

    if not header_lines:
        n_cols = len(data_lines[0].split("|")) if data_lines else 0
        headers = [f"col{i}" for i in range(n_cols)]
    elif len(header_lines) == 1:
        headers = [h.strip() for h in header_lines[0].split("|")]
    else:
        # Multiple header lines: merge with spaces, then split on pipe
        merged = " ".join(header_lines)
        if "|" in merged:
            headers = [h.strip() for h in merged.split("|")]
        else:
            n_cols = len(data_lines[0].split("|")) if data_lines else 1
            headers = [merged] + [f"col{i + 1}" for i in range(n_cols - 1)]

    rows = [
        [cell.strip() for cell in line.split("|")] for line in data_lines
    ]
    return headers, rows


def _build_records(
    headers: List[str], rows: List[List[str]]
) -> List[Dict[str, str]]:
    """Build a list of {header: value} dicts from parsed table headers and rows."""
    records = []
    for row in rows:
        padded = row + [""] * max(0, len(headers) - len(row))
        records.append({headers[i]: padded[i] for i in range(len(headers))})
    return records


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseChartExtractor(ABC):
    """
    Abstract interface for all Stage 3 chart extraction backends.

    All backends implement this interface so that the pipeline orchestrator
    can swap between DePlot, MatCha, Pix2Struct-base, and SVLM without any
    changes to Stage 3 or Stage 4 logic.
    """

    @abstractmethod
    def extract(
        self,
        image_bgr: np.ndarray,
        chart_id: str = "unknown",
    ) -> Optional[Pix2StructResult]:
        """
        Run chart-to-table extraction on a BGR image.

        Args:
            image_bgr: BGR uint8 numpy array (OpenCV format).
            chart_id: Chart identifier used in log messages.

        Returns:
            Pix2StructResult with parsed table, or None on hard failure.
            When the model produces empty output, returns a result with
            extraction_confidence=0.0 so callers can handle the empty case.
        """
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True if the model is loaded and ready for inference."""
        ...

    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Backend identifier string (e.g. 'deplot', 'matcha', 'svlm')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """HuggingFace model ID currently in use."""
        ...


# ---------------------------------------------------------------------------
# Pix2Struct-family backends: DePlot, MatCha, Pix2Struct-base
# They all use AutoProcessor + AutoModelForImageTextToText and share the same
# inference loop and DePlot linearized table parser.
# ---------------------------------------------------------------------------


class _Pix2StructFamilyExtractor(BaseChartExtractor):
    """
    Shared inference engine for all AutoModelForImageTextToText models.

    Subclasses set class-level _DEFAULT_MODEL, _PROMPT, and _BACKEND_ID.
    """

    _DEFAULT_MODEL: str = ""
    _PROMPT: str = "Generate underlying data table of the figure below:"
    _BACKEND_ID: str = ""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        max_patches: int = 1024,
    ) -> None:
        self._model_name = model_name or self._DEFAULT_MODEL
        self._device_str = device
        self.max_patches = max_patches

        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False
        self._load_error: Optional[str] = None

    @property
    def backend_id(self) -> str:
        return self._BACKEND_ID

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_available(self) -> bool:
        return self._loaded

    def _ensure_loaded(self) -> bool:
        """Load model on first call. Returns True when model is ready."""
        if self._loaded:
            return True
        if self._load_error:
            return False

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            if self._device_str == "auto":
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self._device = torch.device(self._device_str)

            logger.info(
                f"{self.__class__.__name__} loading | "
                f"model={self._model_name} | device={self._device}"
            )
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._model_name
            ).to(self._device)
            self._model.eval()
            self._loaded = True
            logger.info(
                f"{self.__class__.__name__} ready | "
                f"model={self._model_name} | device={self._device}"
            )
            return True

        except ImportError:
            self._load_error = (
                "transformers not installed. "
                "Run: pip install transformers>=4.40"
            )
            logger.warning(
                f"{self.__class__.__name__} unavailable: {self._load_error}"
            )
            return False

        except Exception as exc:
            self._load_error = str(exc)
            logger.warning(
                f"{self.__class__.__name__} load failed | "
                f"model={self._model_name} | error={exc}"
            )
            return False

    def extract(
        self,
        image_bgr: np.ndarray,
        chart_id: str = "unknown",
    ) -> Optional[Pix2StructResult]:
        if not self._ensure_loaded():
            logger.debug(
                f"{self.__class__.__name__} skipped | "
                f"chart_id={chart_id} | reason={self._load_error}"
            )
            return None

        try:
            import cv2
            import torch
            from PIL import Image as PILImage

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb)

            inputs = self._processor(
                images=pil_image,
                text=self._PROMPT,
                return_tensors="pt",
                max_patches=self.max_patches,
            ).to(self._device)

            with torch.no_grad():
                predictions = self._model.generate(**inputs, max_new_tokens=512)

            raw_text: str = self._processor.decode(
                predictions[0], skip_special_tokens=True
            )

            headers, rows = _parse_deplot_output(raw_text)
            records = _build_records(headers, rows) if headers else []

            if not headers or not rows:
                logger.warning(
                    f"{self.__class__.__name__}: empty table output | "
                    f"chart_id={chart_id} | raw={raw_text[:120]!r}"
                )
                return Pix2StructResult(
                    headers=[],
                    rows=[],
                    records=[],
                    raw_html=raw_text,
                    model_name=self._model_name,
                    extraction_confidence=0.0,
                )

            logger.info(
                f"{self.__class__.__name__} ok | chart_id={chart_id} | "
                f"headers={len(headers)} | rows={len(rows)}"
            )
            return Pix2StructResult(
                headers=headers,
                rows=rows,
                records=records,
                raw_html=raw_text,
                model_name=self._model_name,
                extraction_confidence=1.0,
            )

        except Exception as exc:
            logger.error(
                f"{self.__class__.__name__} failed | "
                f"chart_id={chart_id} | error={exc}"
            )
            return None


class DeplotExtractor(_Pix2StructFamilyExtractor):
    """
    DePlot extractor using google/deplot (recommended default backend).

    DePlot is Pix2Struct fine-tuned on a curated chart-to-table dataset,
    producing linearized table output from chart images.

    Reference: Lin et al. (2022) "DePlot: One-shot visual language reasoning
    by plot-to-table translation", ACL 2023. https://arxiv.org/abs/2212.10505
    """

    _DEFAULT_MODEL = "google/deplot"
    _PROMPT = "Generate underlying data table of the figure below:"
    _BACKEND_ID = "deplot"


class MatchaExtractor(_Pix2StructFamilyExtractor):
    """
    MatCha extractor using google/matcha-base.

    MatCha extends Pix2Struct with joint math-reasoning and chart QA
    pretraining, making it more robust on charts with complex axes,
    formulas, and scientific notation.

    Reference: Liu et al. (2022) "MatCha: Enhancing Visual Language
    Pretraining with Math Reasoning and Chart Derendering", ACL 2023.
    https://arxiv.org/abs/2212.09662
    """

    _DEFAULT_MODEL = "google/matcha-base"
    _PROMPT = "Generate underlying data table of the figure below:"
    _BACKEND_ID = "matcha"


class Pix2StructBaselineExtractor(_Pix2StructFamilyExtractor):
    """
    Pix2Struct-base zero-task baseline using google/pix2struct-base.

    This backend uses the base Pix2Struct model WITHOUT chart-derendering
    fine-tuning. It serves as the ablation baseline to quantify the
    contribution of chart-specific fine-tuning (DePlot / MatCha) over the
    general-purpose screenshot parser.

    Expected behavior: partial or incorrect table extraction, since the model
    was not trained for chart-to-table translation.

    Reference: Lee et al. (2023) "Pix2Struct: Screenshot Parsing as
    Pretraining for Visual Language Understanding", ICML 2023.
    https://arxiv.org/abs/2210.03347
    """

    _DEFAULT_MODEL = "google/pix2struct-base"
    _PROMPT = "Generate underlying data table of the figure below:"
    _BACKEND_ID = "pix2struct"


# ---------------------------------------------------------------------------
# SVLM backend: Qwen2-VL-2B-Instruct
# ---------------------------------------------------------------------------

_SVLM_PROMPT = (
    "Extract all data from this chart. "
    "Return ONLY a JSON object in this exact format, no explanation:\n"
    '{"headers": ["col0", "col1", ...], '
    '"rows": [["val0", "val1", ...], ...]}\n'
    "Column 0 must be the x-axis or category label column. "
    "Remaining columns are series names followed by their numeric values."
)


class SVLMExtractor(BaseChartExtractor):
    """
    Small VLM extractor using Qwen2-VL-2B-Instruct (zero-shot).

    Uses a chat-based vision-language model to extract chart data via
    zero-shot JSON prompting. Unlike Pix2Struct-family models, this backend
    was not fine-tuned for chart derendering but leverages large-scale VLM
    pretraining to generalize across chart types and styles.

    Requirements:
        pip install transformers>=4.40 qwen-vl-utils

    Expected VRAM: ~6 GB in float16 (compatible with RTX 3060 12 GB).

    Reference: Wang et al. (2024) "Qwen2-VL: Enhancing Vision-Language
    Model's Perception of the World at Any Resolution",
    https://arxiv.org/abs/2409.12191
    """

    _DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
    _BACKEND_ID = "svlm"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        max_new_tokens: int = 512,
    ) -> None:
        self._model_name = model_name or self._DEFAULT_MODEL
        self._device_str = device
        self.max_new_tokens = max_new_tokens

        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False
        self._load_error: Optional[str] = None

    @property
    def backend_id(self) -> str:
        return self._BACKEND_ID

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_available(self) -> bool:
        return self._loaded

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._load_error:
            return False

        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            if self._device_str == "auto":
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self._device = torch.device(self._device_str)

            logger.info(
                f"SVLMExtractor loading | "
                f"model={self._model_name} | device={self._device}"
            )
            self._processor = AutoProcessor.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self._model_name,
                torch_dtype="auto",
                trust_remote_code=True,
            ).to(self._device)
            self._model.eval()
            self._loaded = True
            logger.info(
                f"SVLMExtractor ready | "
                f"model={self._model_name} | device={self._device}"
            )
            return True

        except ImportError as exc:
            self._load_error = (
                f"Missing dependency: {exc}. "
                "Run: pip install transformers>=4.40 qwen-vl-utils"
            )
            logger.warning(f"SVLMExtractor unavailable: {self._load_error}")
            return False

        except Exception as exc:
            self._load_error = str(exc)
            logger.warning(
                f"SVLMExtractor load failed | "
                f"model={self._model_name} | error={exc}"
            )
            return False

    def extract(
        self,
        image_bgr: np.ndarray,
        chart_id: str = "unknown",
    ) -> Optional[Pix2StructResult]:
        if not self._ensure_loaded():
            logger.debug(
                f"SVLMExtractor skipped | "
                f"chart_id={chart_id} | reason={self._load_error}"
            )
            return None

        try:
            import cv2
            import torch
            from PIL import Image as PILImage

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": _SVLM_PROMPT},
                    ],
                }
            ]

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens
                )

            # Strip prompt tokens from output
            generated = output_ids[0][inputs["input_ids"].shape[1] :]
            raw_text: str = self._processor.decode(
                generated, skip_special_tokens=True
            )

            headers, rows = self._parse_svlm_output(raw_text, chart_id)
            records = _build_records(headers, rows) if headers else []

            if not headers or not rows:
                logger.warning(
                    f"SVLMExtractor: empty output | "
                    f"chart_id={chart_id} | raw={raw_text[:120]!r}"
                )
                return Pix2StructResult(
                    headers=[],
                    rows=[],
                    records=[],
                    raw_html=raw_text,
                    model_name=self._model_name,
                    extraction_confidence=0.0,
                )

            logger.info(
                f"SVLMExtractor ok | chart_id={chart_id} | "
                f"headers={len(headers)} | rows={len(rows)}"
            )
            return Pix2StructResult(
                headers=headers,
                rows=rows,
                records=records,
                raw_html=raw_text,
                model_name=self._model_name,
                extraction_confidence=1.0,
            )

        except Exception as exc:
            logger.error(
                f"SVLMExtractor failed | chart_id={chart_id} | error={exc}"
            )
            return None

    def _parse_svlm_output(
        self, text: str, chart_id: str
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Parse SVLM JSON output into (headers, rows).

        Tries JSON extraction first. Falls back to the DePlot linearized
        parser if the model outputs a pipe-separated table instead of JSON.
        """
        import json
        import re

        code_block_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        candidate_text = code_block_match.group(1).strip() if code_block_match else text

        for pattern in (r"\{.*\}", r"\[.*\]"):
            json_match = re.search(pattern, candidate_text, re.DOTALL)
            if not json_match:
                continue
            try:
                data = json.loads(json_match.group())
                if isinstance(data, dict):
                    headers = [str(h) for h in data.get("headers", [])]
                    rows = [
                        [str(cell) for cell in row] for row in data.get("rows", [])
                    ]
                    if headers and rows:
                        return headers, rows

                if (
                    isinstance(data, list)
                    and len(data) >= 2
                    and all(isinstance(row, list) for row in data)
                ):
                    headers = [str(cell) for cell in data[0]]
                    rows = [[str(cell) for cell in row] for row in data[1:]]
                    if headers and rows:
                        return headers, rows
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                logger.debug(
                    f"SVLMExtractor: JSON parse failed | chart_id={chart_id} | "
                    "falling back to DePlot linearized parser"
                )

        # Fallback: DePlot linearized table format
        return _parse_deplot_output(candidate_text)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_extractor(
    backend: str,
    model_override: Optional[str] = None,
    device: str = "auto",
    max_patches: int = 1024,
) -> BaseChartExtractor:
    """
    Create a chart extractor for the specified backend.

    Args:
        backend: Backend identifier — "deplot", "pix2struct", "matcha", "svlm".
        model_override: Override the default HuggingFace model ID. Use for
            local model paths or fine-tuned variants. None = use backend default.
        device: Compute device — "auto", "cuda", or "cpu".
        max_patches: Max image patches for Pix2Struct-family models.
            Ignored for the svlm backend.

    Returns:
        BaseChartExtractor instance. The model is loaded lazily on the first
        call to .extract(), not at construction time.

    Raises:
        ValueError: If the backend identifier is not recognized.

    Examples:
        create_extractor("deplot")
        create_extractor("matcha", device="cuda")
        create_extractor("deplot", model_override="/models/my_deplot")
        create_extractor("svlm", model_override="Qwen/Qwen2-VL-7B-Instruct")
    """
    backend_norm = backend.lower().strip()

    if backend_norm == BackendType.DEPLOT:
        return DeplotExtractor(
            model_name=model_override, device=device, max_patches=max_patches
        )
    if backend_norm == BackendType.PIX2STRUCT:
        return Pix2StructBaselineExtractor(
            model_name=model_override, device=device, max_patches=max_patches
        )
    if backend_norm == BackendType.MATCHA:
        return MatchaExtractor(
            model_name=model_override, device=device, max_patches=max_patches
        )
    if backend_norm == BackendType.SVLM:
        return SVLMExtractor(
            model_name=model_override,
            device=device,
        )

    valid = ", ".join(b.value for b in BackendType)
    raise ValueError(
        f"Unknown extraction backend: {backend!r}. Valid options: {valid}"
    )
