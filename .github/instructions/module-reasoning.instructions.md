---
applyTo: 'src/core_engine/stages/s4_reasoning/**,src/core_engine/ai/**'
---

# MODULE INSTRUCTIONS - AI Reasoning & Routing

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.1.0 | 2026-03-02 | That Le | Updated to reflect implemented state (all adapters + router + tests) |
| 1.0.0 | 2026-02-28 | That Le | AI adapter pattern and multi-provider reasoning |

---

## 1. Overview

**AI Reasoning Module** implements multi-provider LLM integration for Stage 4 (Semantic Reasoning). Uses an Adapter pattern to abstract provider differences behind a common interface, with a Router that handles fallback logic and confidence-based routing.

**Key Directories:**
- `src/core_engine/ai/` - AI abstraction layer (8 files, IMPLEMENTED)
  - `adapters/` - Provider-specific implementations (base, gemini, openai, local_slm)
  - `router.py` - Task-based routing with fallback chains
  - `task_types.py` - Enum of AI task types
  - `prompts.py` - Shared prompt templates (versioned)
  - `exceptions.py` - AI-specific exception hierarchy
- `src/core_engine/stages/s4_reasoning/` - Stage 4 pipeline stage
  - `reasoning_engine.py` - Base ABC
  - `router_engine.py` - AIRouter bridge (implements ReasoningEngine ABC)
  - `gemini_engine.py` - Legacy Gemini implementation (retained for backward compatibility)
- `tests/test_ai/` - 55 unit tests (5 files, all mock-based)

---

## 2. Adapter Pattern

### 2.1. Base Adapter Interface

```python
# src/core_engine/ai/adapters/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class AIResponse:
    """Standardized response from any AI provider."""
    content: str
    model_used: str
    provider: str
    confidence: float = 0.0
    usage: Dict[str, int] = field(default_factory=dict)  # tokens, cost
    raw_response: Optional[Any] = None
    success: bool = True
    error_message: Optional[str] = None

class BaseAIAdapter(ABC):
    """Abstract base for all AI provider adapters."""
    
    provider_id: str  # "gemini", "openai", "local_slm"
    
    @abstractmethod
    async def reason(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> AIResponse:
        """Send reasoning prompt and get structured response."""
        ...
    
    @abstractmethod
    async def correct_ocr(
        self,
        texts: List[str],
        chart_context: Dict[str, Any],
        **kwargs
    ) -> AIResponse:
        """Correct OCR errors using chart context."""
        ...
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is available and healthy."""
        ...
```

### 2.2. Provider Implementations

| Adapter | File | Status |
| --- | --- | --- |
| `GeminiAdapter` | `ai/adapters/gemini_adapter.py` | IMPLEMENTED (Google Generative AI SDK, vision support) |
| `OpenAIAdapter` | `ai/adapters/openai_adapter.py` | IMPLEMENTED (Chat Completions, vision support) |
| `LocalSLMAdapter` | `ai/adapters/local_slm_adapter.py` | IMPLEMENTED (HuggingFace Transformers, 4-bit quant + LoRA; enabled=False until training complete) |

### 2.3. Adapter Implementation Rules

1. **Each adapter** handles its own authentication (API keys, tokens)
2. **Each adapter** handles its own retry logic (provider-specific errors)
3. **All adapters** MUST return `AIResponse` -- never raw provider objects
4. **All adapters** MUST implement `health_check()` for router to verify availability
5. **Never** expose provider SDK types outside the adapter
6. **Log** all API calls with provider, model, token count, latency

---

## 3. AI Router

### 3.1. Task Types

```python
# src/core_engine/ai/task_types.py
from enum import Enum

class TaskType(str, Enum):
    """AI tasks the system needs to perform."""
    CHART_REASONING = "chart_reasoning"      # Full chart analysis
    OCR_CORRECTION = "ocr_correction"        # Fix OCR errors
    DESCRIPTION_GEN = "description_gen"      # Generate descriptions
    DATA_VALIDATION = "data_validation"      # Validate extracted data
```

### 3.2. Fallback Chains

```python
# src/core_engine/ai/router.py
FALLBACK_CHAINS: Dict[TaskType, List[str]] = {
    TaskType.CHART_REASONING: ["local_slm", "gemini", "openai"],
    TaskType.OCR_CORRECTION: ["local_slm", "gemini"],
    TaskType.DESCRIPTION_GEN: ["local_slm", "gemini", "openai"],
    TaskType.DATA_VALIDATION: ["gemini", "openai"],  # needs higher capability
}
```

**Router Logic:**
1. For a given `TaskType`, get the fallback chain
2. Try the first healthy provider
3. If response confidence < threshold (configurable), try next provider
4. If provider raises exception, log error and try next
5. If all providers fail, raise `AIProviderExhaustedError`

### 3.3. Router Configuration

```yaml
# In config/models.yaml
ai_routing:
  confidence_threshold: 0.7     # Below this, try next provider
  timeout_seconds: 30           # Per-provider timeout
  max_retries_per_provider: 2
  
  # Provider-specific config
  providers:
    gemini:
      enabled: true
      model: "gemini-2.0-flash"
      api_key_env: "GOOGLE_API_KEY"
    openai:
      enabled: false             # Not using in development
      model: "gpt-4o-mini"
      api_key_env: "OPENAI_API_KEY"
    local_slm:
      enabled: false             # Enable after training
      model_path: "models/slm/qwen2.5-1.5b-chart-merged"
      device: "auto"
```

---

## 4. Stage 4 Integration

### 4.1. Current Architecture (Implemented)

**Dual-path design:**
- `router_engine.py` - AIRouter bridge, implements `ReasoningEngine` ABC, delegates to `AIRouter` with fallback chains. This is the **production path**.
- `gemini_engine.py` - Legacy Gemini implementation, retained for backward compatibility and direct testing.

**Pipeline wiring:** `pipeline.py` initializes Stage 4 with `engine="router"` by default, which uses `AIRouterEngine` -> `AIRouter` -> best available adapter.

### 4.2. Completed Migration Steps

| Step | Action | Status |
| --- | --- | --- |
| 1 | Create `src/core_engine/ai/` directory structure | DONE |
| 2 | Define `BaseAIAdapter`, `AIResponse`, `TaskType` | DONE |
| 3 | Create `GeminiAdapter` | DONE |
| 4 | Create `AIRouter` with fallback chains | DONE |
| 5 | Wire Stage 4 via `router_engine.py` | DONE |
| 6 | Create `LocalSLMAdapter` | DONE (enabled=False until training) |
| 7 | Create `OpenAIAdapter` | DONE |
| 8 | Create AI routing test suite (55 tests) | DONE |

**Rule:** `gemini_engine.py` remains functional as a fallback. Do NOT delete without updating all tests.

---

## 5. Prompt Engineering

### 5.1. System Prompts

All prompts are stored as constants in `src/core_engine/ai/prompts.py`:

```python
# Core reasoning prompt
CHART_REASONING_SYSTEM = """You are a chart analysis expert. Given raw chart metadata...

Input format:
- chart_type: {chart_type}
- ocr_texts: {texts}
- detected_elements: {elements}
- axis_info: {axes}

Output format: Strict JSON matching the RefinedChartData schema.
"""

# OCR correction prompt
OCR_CORRECTION_SYSTEM = """You are an OCR error corrector specialized in chart text..."""
```

### 5.2. Prompt Rules

1. **NEVER** hardcode prompts inside adapter classes
2. **ALWAYS** use string templates with explicit variable names
3. **VERSION** prompts with a hash/tag for reproducing results
4. **SEPARATE** system prompts from user prompts
5. **TEST** prompts with at least 3 chart types before deploying

---

## 6. Error Handling

### 6.1. Exception Hierarchy

```python
# src/core_engine/ai/exceptions.py
class AIProviderError(Exception):
    """Base exception for AI provider errors."""
    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")

class AIRateLimitError(AIProviderError):
    """API rate limit hit."""
    pass

class AIAuthenticationError(AIProviderError):
    """Invalid API key or permissions."""
    pass

class AIProviderExhaustedError(Exception):
    """All providers in fallback chain failed."""
    pass
```

### 6.2. Error Handling Rules

1. **Rate limits:** Exponential backoff within adapter, then surface to router
2. **Auth errors:** Immediately mark provider as unhealthy, skip to next
3. **Timeout:** Respect `timeout_seconds` config, then try next provider
4. **Invalid JSON response:** Retry once with stricter prompt, then surface error
5. **NEVER** swallow exceptions silently -- always log at WARNING+

---

## 7. Testing Strategy

### 7.1. Test Structure (Implemented)

```
tests/
    test_ai/
        __init__.py
        test_adapters_base.py    # BaseAIAdapter + AIResponse tests
        test_router.py           # Router fallback logic, health checks
        test_task_types.py       # TaskType enum coverage
        test_prompts.py          # Prompt templates validation
        test_exceptions.py       # Exception hierarchy tests
```

55 unit tests total, all using mock adapters (no real API calls).

### 7.2. Mock Strategy

- **Unit tests:** Mock API responses with fixtures (never call real APIs)
- **Integration tests:** Use `@pytest.mark.integration` marker, require API key
- **Router tests:** Test fallback logic with mock adapters that simulate failures

---

## 8. Rules Summary

1. **All AI calls** go through `AIRouter` -- no direct provider SDK calls from pipeline code
2. **Adapters** are the ONLY place where provider SDKs are imported
3. **AIResponse** is the ONLY return type from adapters
4. **Fallback chains** MUST be configured, not hardcoded
5. **Prompts** live in `ai/prompts.py`, not inside adapters
6. **Confidence scores** drive routing decisions
7. **Health checks** run before routing to avoid wasting time on dead providers
8. **Log** every AI call: provider, model, tokens, latency, success/fail
