from __future__ import annotations

import os
from pathlib import Path

MODEL_DIR_ENV = "SKINTOKEN_MODEL_DIR"


def node_root() -> Path:
    return Path(__file__).resolve().parents[2]


def vendor_root() -> Path:
    return node_root() / "vendor"


def configs_root() -> Path:
    return vendor_root() / "configs"


def get_model_root() -> Path:
    configured = os.environ.get(MODEL_DIR_ENV)
    if configured:
        return Path(configured)
    return node_root() / "models" / "skintoken"

def resolve_model_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    model_root = get_model_root()
    resolved = model_root / candidate
    if resolved.exists():
        return resolved
    return resolved


def resolve_config_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()

    text = str(path)
    if text.startswith("./"):
        text = text[2:]
    return vendor_root() / text


def get_llm_local_dir() -> Path:
    model_root = get_model_root()
    return model_root / "Qwen3-0.6B"
