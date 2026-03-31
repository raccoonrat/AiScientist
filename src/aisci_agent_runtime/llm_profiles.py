from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMProfile:
    name: str
    provider: str
    model: str
    api_mode: str
    reasoning_effort: str | None = None
    web_search: bool = False
    max_tokens: int = 32768
    context_window: int | None = None
    use_phase: bool = False


def resolve_llm_profile(profile_name: str) -> LLMProfile:
    name = profile_name.strip() or "default"
    lowered = name.lower()
    provider = "openai"
    api_mode = "responses"
    model = name
    reasoning_effort: str | None = None
    web_search = False
    max_tokens = 32768
    context_window: int | None = None
    use_phase = False

    if "completions" in lowered:
        api_mode = "completions"
    if "responses" in lowered:
        api_mode = "responses"
    if "gpt-5.4" in lowered:
        model = "gpt-5.4"
        max_tokens = 65536
        context_window = 934464
        use_phase = True
    elif "gpt-5.2" in lowered:
        model = "gpt-5.2"
        max_tokens = 32768
        context_window = 367232
    elif "glm" in lowered:
        provider = "azure-openai"
        api_mode = "completions"
        max_tokens = 20480
        context_window = 139660
    elif "deepseek" in lowered:
        provider = "azure-openai"
        api_mode = "completions"

    if "mini" in lowered:
        reasoning_effort = "medium"
    elif "high" in lowered:
        reasoning_effort = "high"

    if "web" in lowered:
        web_search = True

    return LLMProfile(
        name=name,
        provider=provider,
        model=model,
        api_mode=api_mode,
        reasoning_effort=reasoning_effort,
        web_search=web_search,
        max_tokens=max_tokens,
        context_window=context_window,
        use_phase=use_phase,
    )


def llm_env(profile_name: str) -> dict[str, str]:
    profile = resolve_llm_profile(profile_name)
    env = {
        "AISCI_MODEL": profile.model,
        "AISCI_API_MODE": profile.api_mode,
        "AISCI_WEB_SEARCH": "true" if profile.web_search else "false",
        "AISCI_MAX_TOKENS": str(profile.max_tokens),
    }
    if profile.reasoning_effort:
        env["AISCI_REASONING_EFFORT"] = profile.reasoning_effort
    if profile.context_window is not None:
        env["AISCI_CONTEXT_WINDOW"] = str(profile.context_window)
    if profile.use_phase:
        env["AISCI_USE_PHASE"] = "true"
    passthrough_keys = (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "OPENAI_API_VERSION",
    )
    for key in passthrough_keys:
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env
