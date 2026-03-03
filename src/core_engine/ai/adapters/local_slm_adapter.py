"""
Local SLM Adapter

Wraps local small language model inference (Qwen, Llama) behind the
BaseAIAdapter interface. Uses HuggingFace transformers + optional LoRA.

Status: PLACEHOLDER - Enable after SLM training is complete.
        See: .github/instructions/module-training.instructions.md

Model priority: merged model (base + LoRA merged) > LoRA adapter > base model only.

Usage:
    adapter = LocalSLMAdapter(model_path="models/slm/qwen2.5-1.5b-chart-merged")
    healthy = await adapter.health_check()
    response = await adapter.reason(system_prompt, user_prompt)
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

from .base import AIResponse, BaseAIAdapter
from ..exceptions import AIProviderError

logger = logging.getLogger(__name__)

# Default model from HuggingFace Hub (used if no local path)
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class LocalSLMAdapter(BaseAIAdapter):
    """
    Local SLM provider adapter (Qwen2.5 / Llama-3.2).

    Runs inference on the local machine using HuggingFace transformers.
    Supports 4-bit quantization (BitsAndBytes) for low VRAM environments.
    Supports LoRA adapter loading for fine-tuned chart-specialist models.

    Attributes:
        provider_id: "local_slm"
    """

    provider_id = "local_slm"

    def __init__(
        self,
        model_path: Optional[str] = None,
        lora_path: Optional[str] = None,
        device: str = "auto",
        max_tokens: int = 512,
        temperature: float = 0.3,
        load_in_4bit: bool = True,
        enabled: bool = False,
    ) -> None:
        """
        Initialize the local SLM adapter.

        Args:
            model_path: Path to merged model dir or HuggingFace model ID.
                        Defaults to Qwen/Qwen2.5-1.5B-Instruct.
            lora_path: Optional path to LoRA adapter weights directory.
            device: Target device ("auto", "cpu", "cuda", "mps")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
            enabled: Must be True to load the model. Set False to skip init
                     (useful when model weights are not yet available).
        """
        self._model_path = model_path or DEFAULT_HF_MODEL
        self._lora_path = lora_path
        self._device = device
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._load_in_4bit = load_in_4bit
        self._enabled = enabled

        self._model: Any = None
        self._tokenizer: Any = None
        self._pipeline: Any = None

        if enabled:
            self._initialize_model()
        else:
            logger.info(
                "LocalSLMAdapter | disabled | "
                "set enabled=True after model training is complete"
            )

    # -------------------------------------------------------------------------
    # BaseAIAdapter interface
    # -------------------------------------------------------------------------

    async def reason(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: Optional[str] = None,
        image_path: Optional[str] = None,
        **kwargs: Any,
    ) -> AIResponse:
        """
        Run local SLM inference.

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User-turn prompt with data
            model_id: Ignored (local model is fixed at init)
            image_path: Not supported -- local SLM is text-only
            **kwargs: Extra generation kwargs (max_new_tokens, etc.)

        Returns:
            AIResponse with generated text
        """
        if not self._enabled or self._pipeline is None:
            return AIResponse.error(
                self.provider_id,
                self._model_path,
                "LocalSLM not initialized. Set enabled=True and provide model_path.",
            )

        if image_path:
            logger.warning(
                "LocalSLMAdapter.reason | image_path ignored | "
                "local SLM is text-only"
            )

        try:
            output = await asyncio.to_thread(
                self._run_inference, system_prompt, user_prompt, **kwargs
            )
            logger.info(
                f"LocalSLMAdapter.reason | model={self._model_path} | "
                f"chars={len(output)}"
            )
            return AIResponse(
                content=output,
                model_used=self._model_path,
                provider=self.provider_id,
                confidence=0.75,  # Adjust based on validation results
                success=True,
            )
        except Exception as exc:
            msg = str(exc)
            logger.error(
                f"LocalSLMAdapter.reason | failed | "
                f"model={self._model_path} | error={msg}"
            )
            raise AIProviderError(self.provider_id, msg) from exc

    async def health_check(self) -> bool:
        """
        Check if local model is loaded and responsive.

        Returns:
            True if pipeline is initialized and can run inference
        """
        if not self._enabled or self._pipeline is None:
            return False

        try:
            await asyncio.to_thread(self._run_inference, "You are helpful.", "Respond: ok")
            return True
        except Exception as exc:
            logger.warning(f"LocalSLMAdapter.health_check | failed | error={exc}")
            return False

    def get_default_model(self) -> str:
        """Return model path or HF model ID."""
        return self._model_path

    # -------------------------------------------------------------------------
    # Private: model loading
    # -------------------------------------------------------------------------

    def _initialize_model(self) -> None:
        """Load model and tokenizer from disk or HuggingFace Hub."""
        try:
            from transformers import (  # type: ignore[import]
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                pipeline,
            )
            import torch  # type: ignore[import]

            logger.info(
                f"LocalSLMAdapter | loading model | path={self._model_path} | "
                f"4bit={self._load_in_4bit} | device={self._device}"
            )

            # Quantization config
            quant_config = None
            if self._load_in_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

            tokenizer = AutoTokenizer.from_pretrained(
                self._model_path, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                quantization_config=quant_config,
                device_map=self._device,
                trust_remote_code=True,
            )

            # Load LoRA adapter if provided
            if self._lora_path and Path(self._lora_path).exists():
                from peft import PeftModel  # type: ignore[import]

                model = PeftModel.from_pretrained(model, self._lora_path)
                logger.info(f"LocalSLMAdapter | LoRA loaded | path={self._lora_path}")

            self._tokenizer = tokenizer
            self._model = model
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )

            # Clear default max_length to avoid conflict with max_new_tokens.
            # Both model and pipeline generation_config must be patched.
            for obj in (model, self._pipeline.model, self._pipeline):
                gc = getattr(obj, "generation_config", None)
                if gc is not None and getattr(gc, "max_length", None) is not None:
                    gc.max_length = None

            logger.info(
                f"LocalSLMAdapter | ready | model={self._model_path}"
            )

        except ImportError as exc:
            logger.error(
                f"LocalSLMAdapter | missing dependency | error={exc} | "
                "install: pip install transformers torch peft bitsandbytes"
            )
        except Exception as exc:
            logger.error(
                f"LocalSLMAdapter | load failed | "
                f"model={self._model_path} | error={exc}"
            )

    # -------------------------------------------------------------------------
    # Private: inference
    # -------------------------------------------------------------------------

    def _run_inference(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> str:
        """Run synchronous inference via HuggingFace pipeline."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        from transformers import GenerationConfig  # type: ignore[import]

        gen_config = GenerationConfig(
            max_new_tokens=kwargs.get("max_new_tokens", self._max_tokens),
            temperature=kwargs.get("temperature", self._temperature),
            do_sample=self._temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        outputs = self._pipeline(
            messages,
            generation_config=gen_config,
        )

        # Extract generated text (exclude input tokens)
        generated = outputs[0]["generated_text"]
        if isinstance(generated, list):
            # Chat format: last message is the assistant's response
            return generated[-1].get("content", "")
        return str(generated)
