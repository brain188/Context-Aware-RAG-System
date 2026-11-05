"""Provider-agnostic LLM layer.

Services import the contract from here and get clients from LLMFactory; only
the provider modules (gemini_client, groq_client, anthropic_client,
ollama_client) talk to a vendor API.
"""

from app.llm.base import (
    LLMClient,
    LLMError,
    LLMOutputError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResult,
    LLMSafetyBlockError,
    LLMTimeoutError,
    Provider,
    Role,
    ThinkingLevel,
)
from app.llm.factory import LLMFactory

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMFactory",
    "LLMOutputError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMResult",
    "LLMSafetyBlockError",
    "LLMTimeoutError",
    "Provider",
    "Role",
    "ThinkingLevel",
]
