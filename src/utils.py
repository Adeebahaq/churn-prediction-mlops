"""Shared utilities for paths, logging, and config loading."""
import io
import os
import sys
import yaml
from pathlib import Path


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def fix_stdout_encoding() -> None:
    """Force UTF-8 stdout/stderr to avoid Windows cp1252 emoji errors."""
    try:
        if sys.stdout.encoding != "utf-8":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        if sys.stderr.encoding != "utf-8":
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
