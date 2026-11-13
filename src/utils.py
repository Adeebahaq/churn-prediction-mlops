"""Shared utilities for paths, logging, and config loading."""
import os
import yaml
from pathlib import Path


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_env() -> None:
    """Load environment variables from .env if present."""
    env_file = get_project_root() / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
