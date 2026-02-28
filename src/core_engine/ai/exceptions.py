"""
AI Provider Exceptions

Exception hierarchy for the AI routing layer.
All provider-specific exceptions are caught inside adapters and re-raised
as one of these types so the router can handle them uniformly.
"""

from typing import Optional


class AIProviderError(Exception):
    """
    Base exception for all AI provider errors.

    Attributes:
        provider: Provider ID that raised the error ("gemini", "openai", "local_slm")
        message: Human-readable error message
    """

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        self.message = message
        super().__init__(f"[{provider}] {message}")


class AIRateLimitError(AIProviderError):
    """
    Provider API rate limit exceeded.

    Router behavior: exponential backoff, then try next provider.

    Attributes:
        retry_after: Suggested wait time in seconds (if provided by API)
    """

    def __init__(
        self,
        provider: str,
        message: str,
        retry_after: Optional[float] = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(provider, message)


class AIAuthenticationError(AIProviderError):
    """
    Invalid API key or insufficient permissions.

    Router behavior: immediately mark provider as unhealthy, skip to next.
    """

    pass


class AITimeoutError(AIProviderError):
    """
    Provider request exceeded configured timeout.

    Router behavior: try next provider.

    Attributes:
        timeout_seconds: The timeout that was exceeded
    """

    def __init__(
        self,
        provider: str,
        message: str,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(provider, message)


class AIInvalidResponseError(AIProviderError):
    """
    Provider returned a response that could not be parsed.

    Router behavior: retry once with stricter prompt, then try next provider.

    Attributes:
        raw_response: The unparseable raw text from the provider
    """

    def __init__(
        self,
        provider: str,
        message: str,
        raw_response: Optional[str] = None,
    ) -> None:
        self.raw_response = raw_response
        super().__init__(provider, message)


class AIProviderExhaustedError(Exception):
    """
    All providers in the fallback chain have failed.

    Raised by AIRouter when no provider could handle the request.

    Attributes:
        task_type: The task type that could not be fulfilled
        errors: Dict mapping provider_id -> error message
    """

    def __init__(self, task_type: str, errors: dict[str, str]) -> None:
        self.task_type = task_type
        self.errors = errors
        providers = ", ".join(errors.keys())
        super().__init__(
            f"All providers exhausted for task '{task_type}'. "
            f"Tried: {providers}. Errors: {errors}"
        )
