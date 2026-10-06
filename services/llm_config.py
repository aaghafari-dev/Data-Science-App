"""Module duty: Llm config.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

"""Centralized, persistent and provider-aware LLM configuration."""

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import config


@dataclass
class LLMConfig:
    provider: str = "none"  # none | local | api
    model_name: str = ""
    api_key: str = ""
    local_path: str = ""
    api_base: str = ""
    # User-declared resource budget for local LLM execution. None = auto-detect.
    ram_gb: float | None = None
    gpu_vram_gb: float | None = None

    def configured(self) -> bool:
        """Perform the configured operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.provider == "api":
            return bool(self.api_key.strip() and self.model_name.strip())
        if self.provider == "local":
            return bool(self.model_name.strip() and (self.local_path.strip() or self.model_name in config.LOCAL_LLM_MODELS))
        return False

    def redacted(self) -> dict[str, Any]:
        """Perform the redacted operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        data = asdict(self)
        data["api_key"] = "***configured***" if self.api_key else ""
        return data


def infer_api_provider(model_name: str) -> str:
    """Infer an API family from a model identifier without claiming exclusivity."""
    m = (model_name or "").lower().strip()
    if m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "google"
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith("grok"):
        return "xai"
    if m.startswith(("mistral", "ministral", "codestral", "pixtral", "voxtral")):
        return "mistral"
    if m.startswith("command"):
        return "cohere"
    if m.startswith(("llama", "qwen", "glm", "kimi")):
        return "generic"
    return "generic"


def local_model_path(model_name: str, custom_path: str = "") -> str:
    """Perform the local model path operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    if custom_path.strip():
        return custom_path.strip()
    rel = config.LOCAL_LLM_MODELS.get(model_name)
    if not rel:
        return ""
    return str(Path(config.HF_CACHE_DIR) / rel)


def local_model_available(model_name: str, custom_path: str = "") -> bool:
    """Perform the local model available operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    path = local_model_path(model_name, custom_path)
    if not path:
        return False
    p = Path(path)
    if not p.exists():
        return False
    if (p / "snapshots").exists():
        return any(x.is_dir() for x in (p / "snapshots").iterdir())
    return any(p.glob("*.json")) or any(p.glob("*.safetensors")) or any(p.glob("*.bin"))


def provider_from_state(state: dict[str, Any] | None) -> LLMConfig:
    """Perform the provider from state operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    state = state or {}
    return LLMConfig(
        provider=str(state.get("llm_provider") or state.get("provider") or "none"),
        model_name=str(state.get("model_name") or ""),
        api_key=str(state.get("api_key") or ""),
        local_path=str(state.get("local_path") or ""),
        api_base=str(state.get("api_base") or ""),
        ram_gb=(float(state["ram_gb"]) if state.get("ram_gb") not in (None, "", "Auto-detect") else None),
        gpu_vram_gb=(float(state["gpu_vram_gb"]) if state.get("gpu_vram_gb") not in (None, "", "Auto-detect") else None),
    )


def preflight(cfg: LLMConfig) -> dict[str, Any]:
    """Perform the preflight operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    if not cfg.configured():
        return {"status": "blocked", "reason": "No LLM provider has been explicitly configured."}
    if cfg.provider == "local":
        if not local_model_available(cfg.model_name, cfg.local_path):
            return {
                "status": "blocked",
                "reason": f"Local model '{cfg.model_name}' is not available in the configured cache/path.",
                "path": local_model_path(cfg.model_name, cfg.local_path),
            }
        try:
            global _LOCAL_LOADER
            try:
                _LOCAL_LOADER
            except NameError:
                from agent.local_llm import LocalLLMLoader
                _LOCAL_LOADER = LocalLLMLoader()
            resource = _LOCAL_LOADER.resource_preflight(
                cfg.model_name, cfg.local_path, ram_budget_gb=cfg.ram_gb, gpu_budget_gb=cfg.gpu_vram_gb
            )
        except Exception as exc:
            resource = {"status": "review", "reason": f"Resource preflight unavailable: {exc}"}
        if resource.get("status") == "blocked":
            return {"status": "blocked", "provider": "local", "model": cfg.model_name, "path": local_model_path(cfg.model_name, cfg.local_path), **resource}
        return {"status": "ready", "provider": "local", "model": cfg.model_name, "path": local_model_path(cfg.model_name, cfg.local_path), "resource": resource}
    return {
        "status": "ready",
        "provider": "api",
        "model": cfg.model_name,
        "api_provider": infer_api_provider(cfg.model_name),
        "reason": "API key and model are configured.",
    }


def get_llm(cfg: LLMConfig):
    """Build the configured LangChain chat model with provider-aware routing.

    Optional provider packages are imported only when selected. DeepSeek/xAI
    expose OpenAI-compatible endpoints, so langchain-openai is sufficient for
    those routes. Anthropic, Gemini, Mistral and Cohere use their optional
    LangChain integrations when installed.
    """
    check = preflight(cfg)
    if check.get("status") != "ready":
        return None
    if cfg.provider == "local":
        from agent.local_llm import LocalLLMLoader
        global _LOCAL_LOADER
        try:
            _LOCAL_LOADER
        except NameError:
            _LOCAL_LOADER = LocalLLMLoader()
        return _LOCAL_LOADER.get_llm(cfg.model_name, custom_path=cfg.local_path, ram_budget_gb=cfg.ram_gb, gpu_budget_gb=cfg.gpu_vram_gb)

    family = infer_api_provider(cfg.model_name)
    if family in {"openai", "deepseek", "xai", "mistral", "generic"}:
        try:
            from langchain_openai import ChatOpenAI
            base = cfg.api_base.strip()
            if not base:
                base = {
                    "deepseek": "https://api.deepseek.com",
                    "xai": "https://api.x.ai/v1",
                    "mistral": "https://api.mistral.ai/v1",
                    "openai": "https://api.openai.com/v1",
                }.get(family, "")
            kwargs = {"model": cfg.model_name, "api_key": cfg.api_key, "temperature": 0}
            if base:
                kwargs["base_url"] = base
            return ChatOpenAI(**kwargs)
        except Exception:
            return None
    if family == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=cfg.model_name, api_key=cfg.api_key, temperature=0)
        except Exception:
            return None
    if family == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=cfg.model_name, google_api_key=cfg.api_key, temperature=0)
        except Exception:
            return None
    if family == "cohere":
        try:
            from langchain_cohere import ChatCohere
            return ChatCohere(model=cfg.model_name, cohere_api_key=cfg.api_key, temperature=0)
        except Exception:
            return None
    return None
