"""Per-agent AI provider registry with execution-time preflight checks."""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
from services.llm_config import LLMConfig, preflight

DEFAULT_AGENTS = ("Agent Data Scientist", "AI Agent Plot", "AI Agent Report", "AI Agent Presentation")

class AgentProviderRegistry:
    """Keep independent provider selections for governed AI agents."""
    def __init__(self, configs: dict[str, Any] | None = None):
        self._configs: dict[str, LLMConfig] = {}
        for agent in DEFAULT_AGENTS:
            value = (configs or {}).get(agent)
            if isinstance(value, LLMConfig):
                self._configs[agent] = value
            elif isinstance(value, dict):
                self._configs[agent] = LLMConfig(**{k: value.get(k, "") for k in ("provider","model_name","api_key","local_path","api_base")})

    def set(self, agent_name: str, config: LLMConfig) -> None:
        """Set the provider for one named agent."""
        self._configs[agent_name] = config

    def get(self, agent_name: str, fallback: LLMConfig | None = None) -> LLMConfig:
        """Return the agent-specific provider, falling back to the global selection."""
        return self._configs.get(agent_name, fallback or LLMConfig())

    def preflight(self, agent_name: str, fallback: LLMConfig | None = None) -> dict[str, Any]:
        """Preflight the provider immediately before the agent starts."""
        cfg = self.get(agent_name, fallback)
        result = preflight(cfg)
        result["agent"] = agent_name
        result["configuration"] = cfg.redacted()
        return result

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a JSON-safe redacted registry snapshot."""
        return {name: cfg.redacted() for name, cfg in self._configs.items()}
