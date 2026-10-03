from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

_ALIASES = {
    "openai": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "openai_compatible": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "google_genai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",  # common typo
    "claude": "anthropic",
    "ollama": "ollama",
    "openrouter": "openrouter",
    "open-router": "openrouter",
    "open_router": "openrouter",
}


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents (and the judge).

    Supported providers: openai, custom (OpenAI-compatible base URL), gemini,
    anthropic, ollama, openrouter.
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`. Raises ValueError for unknown providers."""

    key = (value or "").strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unsupported provider {value!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    return _ALIASES[key]


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Provider SDKs are imported lazily so the offline benchmark/tests never need them.
    """

    provider = normalize_provider(config.provider)
    kwargs: dict = {"model": config.model_name, "temperature": config.temperature}

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        if config.api_key:
            kwargs["api_key"] = config.api_key
        if provider == "custom":
            if not config.base_url:
                raise ValueError("Provider 'custom' requires CUSTOM_BASE_URL (OpenAI-compatible endpoint).")
            kwargs["base_url"] = config.base_url
            kwargs.setdefault("api_key", config.api_key or "not-needed")
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    # openrouter
    from langchain_openrouter import ChatOpenRouter

    if config.api_key:
        kwargs["api_key"] = config.api_key
    return ChatOpenRouter(**kwargs)
