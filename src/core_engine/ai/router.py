"""
AI Router

Routes AI requests to the appropriate provider based on task type.
Implements the fallback chain logic: if the primary provider fails or
returns low-confidence output, the next provider in the chain is tried.

Design:
- AIRouter is the ONLY entry point for pipeline stages to call AI.
- Stages MUST NOT instantiate adapters directly.
- Router is created once per pipeline session and reused.

Configuration: config/models.yaml under the `ai_routing` key.

Usage:
    router = AIRouter.from_config(cfg.models.ai_routing)
    response = await router.route(TaskType.CHART_REASONING, system, user)
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from .adapters.base import AIResponse, BaseAIAdapter
from .adapters.gemini_adapter import GeminiAdapter
from .adapters.local_slm_adapter import LocalSLMAdapter
from .adapters.openai_adapter import OpenAIAdapter
from .exceptions import (
    AIAuthenticationError,
    AIProviderError,
    AIProviderExhaustedError,
)
from .task_types import TaskType

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Default fallback chains
# Key: TaskType, Value: ordered list of provider_ids to try
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FALLBACK_CHAINS: Dict[TaskType, List[str]] = {
    TaskType.CHART_REASONING: ["local_slm", "gemini", "openai"],
    TaskType.OCR_CORRECTION: ["local_slm", "gemini"],
    TaskType.DESCRIPTION_GEN: ["local_slm", "gemini", "openai"],
    TaskType.DATA_VALIDATION: ["gemini", "openai"],
}


class AIRouter:
    """
    Routes AI tasks to providers with automatic fallback.

    Routing algorithm:
    1. Get the fallback chain for the requested TaskType
    2. Run health check for each provider in order
    3. Attempt the first healthy provider
    4. If response.confidence < confidence_threshold, try next provider
    5. If provider raises an exception, log it and try next provider
    6. If all providers fail, raise AIProviderExhaustedError

    Attributes:
        confidence_threshold: Minimum confidence to accept a response
        max_retries_per_provider: How many times to retry a single provider
    """

    def __init__(
        self,
        adapters: Optional[Dict[str, BaseAIAdapter]] = None,
        fallback_chains: Optional[Dict[TaskType, List[str]]] = None,
        confidence_threshold: float = 0.7,
        max_retries_per_provider: int = 2,
    ) -> None:
        """
        Initialize the AI Router.

        Args:
            adapters: Dict mapping provider_id -> adapter instance.
                      Defaults to auto-instantiate from environment variables.
            fallback_chains: Override default fallback chains.
            confidence_threshold: Accept response only if confidence >= this.
            max_retries_per_provider: Max retry attempts per provider.
        """
        self._adapters: Dict[str, BaseAIAdapter] = adapters or self._default_adapters()
        self._chains: Dict[TaskType, List[str]] = (
            fallback_chains or DEFAULT_FALLBACK_CHAINS
        )
        self.confidence_threshold = confidence_threshold
        self.max_retries_per_provider = max_retries_per_provider

        # Cache of health check results to avoid repeated checks per session
        self._health_cache: Dict[str, Optional[bool]] = {}

        logger.info(
            f"AIRouter | initialized | "
            f"providers={list(self._adapters.keys())} | "
            f"confidence_threshold={confidence_threshold}"
        )

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> "AIRouter":
        """
        Create AIRouter from OmegaConf config node (models.ai_routing).

        Args:
            config: OmegaConf DictConfig or plain dict with keys:
                    confidence_threshold, max_retries_per_provider, providers

        Returns:
            Configured AIRouter instance
        """
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(config, resolve=True) if hasattr(config, "_metadata") else config

        providers_cfg = cfg.get("providers", {})

        adapters: Dict[str, BaseAIAdapter] = {}

        # Gemini
        gemini_cfg = providers_cfg.get("gemini", {})
        if gemini_cfg.get("enabled", True):
            adapters["gemini"] = GeminiAdapter(
                default_model=gemini_cfg.get("model", "gemini-2.0-flash"),
            )

        # OpenAI
        openai_cfg = providers_cfg.get("openai", {})
        if openai_cfg.get("enabled", False):
            adapters["openai"] = OpenAIAdapter(
                default_model=openai_cfg.get("model", "gpt-4o-mini"),
            )

        # Local SLM
        slm_cfg = providers_cfg.get("local_slm", {})
        if slm_cfg.get("enabled", False):
            adapters["local_slm"] = LocalSLMAdapter(
                model_path=slm_cfg.get("model_path"),
                device=slm_cfg.get("device", "auto"),
                enabled=True,
            )

        return cls(
            adapters=adapters,
            confidence_threshold=cfg.get("confidence_threshold", 0.7),
            max_retries_per_provider=cfg.get("max_retries_per_provider", 2),
        )

    # -------------------------------------------------------------------------
    # Core routing
    # -------------------------------------------------------------------------

    async def route(
        self,
        task_type: TaskType,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
        **kwargs: Any,
    ) -> AIResponse:
        """
        Route a request through the fallback chain for the given task type.

        Args:
            task_type: The AI task to perform
            system_prompt: System/instruction prompt
            user_prompt: User-turn prompt with data payload
            image_path: Optional image path (passed to vision-capable providers)
            **kwargs: Extra generation parameters forwarded to the adapter

        Returns:
            AIResponse from the first successful provider

        Raises:
            AIProviderExhaustedError: If every provider in the chain fails
        """
        chain = self._chains.get(task_type, ["gemini"])
        errors: Dict[str, str] = {}

        for provider_id in chain:
            adapter = self._adapters.get(provider_id)
            if adapter is None:
                logger.debug(
                    f"AIRouter | skipping {provider_id} | not registered"
                )
                continue

            # Health check (cached per session)
            healthy = await self._check_health(provider_id, adapter)
            if not healthy:
                logger.warning(
                    f"AIRouter | skipping {provider_id} | health check failed"
                )
                errors[provider_id] = "health check failed"
                continue

            # Attempt call with retries
            response = await self._attempt_with_retry(
                provider_id,
                adapter,
                task_type,
                system_prompt,
                user_prompt,
                image_path,
                **kwargs,
            )

            if response is None:
                errors[provider_id] = "all retries failed"
                continue

            if not response.success:
                errors[provider_id] = response.error_message or "unknown error"
                continue

            # Accept if confidence is sufficient
            if response.confidence >= self.confidence_threshold:
                logger.info(
                    f"AIRouter | accepted | provider={provider_id} | "
                    f"task={task_type.value} | confidence={response.confidence:.2f}"
                )
                return response

            # Low confidence: try next unless this is the last in chain
            next_idx = chain.index(provider_id) + 1
            if next_idx < len(chain):
                logger.info(
                    f"AIRouter | low confidence {response.confidence:.2f} | "
                    f"provider={provider_id} | trying {chain[next_idx]}"
                )
                errors[provider_id] = f"low confidence ({response.confidence:.2f})"
                continue

            # Last in chain, accept anyway
            logger.info(
                f"AIRouter | last provider, accepting | "
                f"provider={provider_id} | confidence={response.confidence:.2f}"
            )
            return response

        raise AIProviderExhaustedError(task_type.value, errors)

    def route_sync(
        self,
        task_type: TaskType,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
        **kwargs: Any,
    ) -> AIResponse:
        """
        Synchronous wrapper around route() for non-async pipeline stages.

        Uses asyncio.run() or an existing event loop depending on context.

        Args:
            Same parameters as route()

        Returns:
            AIResponse from the first successful provider
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Inside an existing event loop (e.g. Jupyter): wrap in thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.route(
                            task_type, system_prompt, user_prompt, image_path, **kwargs
                        ),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.route(
                        task_type, system_prompt, user_prompt, image_path, **kwargs
                    )
                )
        except RuntimeError:
            return asyncio.run(
                self.route(task_type, system_prompt, user_prompt, image_path, **kwargs)
            )

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    async def _check_health(
        self,
        provider_id: str,
        adapter: BaseAIAdapter,
    ) -> bool:
        """
        Run health check with session-level caching.

        Authentication failures are cached as False permanently for this session.
        Other failures are not cached (network may recover).
        """
        cached = self._health_cache.get(provider_id)
        if cached is not None:
            return cached

        try:
            healthy = await adapter.health_check()
        except AIAuthenticationError:
            logger.error(
                f"AIRouter | auth failure | provider={provider_id} | "
                "marking permanently unhealthy for this session"
            )
            self._health_cache[provider_id] = False
            return False
        except Exception as exc:
            logger.warning(
                f"AIRouter | health check exception | provider={provider_id} | error={exc}"
            )
            return False

        if not healthy:
            logger.warning(f"AIRouter | unhealthy | provider={provider_id}")

        return healthy

    async def _attempt_with_retry(
        self,
        provider_id: str,
        adapter: BaseAIAdapter,
        task_type: TaskType,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str],
        **kwargs: Any,
    ) -> Optional[AIResponse]:
        """
        Call adapter.reason() with retry logic.

        Returns:
            AIResponse on success, None if all retries exhausted.
        """
        last_error: Optional[str] = None

        for attempt in range(1, self.max_retries_per_provider + 1):
            try:
                response = await adapter.reason(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_path=image_path,
                    **kwargs,
                )
                return response

            except AIAuthenticationError as exc:
                logger.error(
                    f"AIRouter | auth error | provider={provider_id} | {exc}"
                )
                self._health_cache[provider_id] = False
                return None  # No point retrying

            except AIProviderError as exc:
                last_error = str(exc)
                logger.warning(
                    f"AIRouter | attempt {attempt}/{self.max_retries_per_provider} "
                    f"failed | provider={provider_id} | task={task_type.value} | "
                    f"error={exc}"
                )
                if attempt < self.max_retries_per_provider:
                    await asyncio.sleep(2 ** (attempt - 1))  # exponential backoff

            except Exception as exc:
                last_error = str(exc)
                logger.error(
                    f"AIRouter | unexpected error | provider={provider_id} | error={exc}"
                )
                break

        logger.warning(
            f"AIRouter | all retries failed | provider={provider_id} | "
            f"last_error={last_error}"
        )
        return None

    @staticmethod
    def _default_adapters() -> Dict[str, BaseAIAdapter]:
        """
        Auto-create adapters from environment variables.

        Gemini and OpenAI adapters read API keys from env.
        LocalSLM is disabled by default (requires training).
        """
        adapters: Dict[str, BaseAIAdapter] = {
            "gemini": GeminiAdapter(),
            "openai": OpenAIAdapter(),
            "local_slm": LocalSLMAdapter(enabled=False),
        }
        return adapters
