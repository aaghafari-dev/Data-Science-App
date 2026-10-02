"""CNN Image Analysis specialist agent for the Master Agent."""
from __future__ import annotations
from typing import Any
from services.cnn_image_analysis import CNNConfig, CNNImageAnalysisEngine
from services.agent_memory import AgentMemory


def run_cnn_image_step(image_dir: str, config: dict[str, Any] | None = None, seed: int = 42, device: str = "auto") -> dict[str, Any]:
    """Execute the governed CNN specialist and return evidence plus bounded specialist memory."""
    cfg = CNNConfig(**(config or {}))
    cfg.seed = seed
    result = CNNImageAnalysisEngine.run(image_dir, cfg, device=device)
    result.pop("model", None); result.pop("model_state_dict", None); result.pop("eval_dataset", None)
    memory = AgentMemory(role="CNN Image Analysis", max_items=20)
    memory.remember("experiment", {"architecture": cfg.architecture, "fine_tune_mode": cfg.fine_tune_mode, "metrics": result.get("metrics", {}), "dataset": result.get("dataset", {})})
    result["specialist"] = "CNN Image Analysis Agent"
    result["specialist_memory"] = memory.to_dict()
    return result
