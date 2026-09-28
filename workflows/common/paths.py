"""Repository-relative path helpers used by public release entry points."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> None:
    """Load the release's simple KEY=VALUE file without overriding the shell."""
    path = path or REPO_ROOT / ".env"
    if not path.is_file():
        return
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_number}: {raw_line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if not key.replace("_", "").isalnum():
            raise ValueError(f"Invalid .env key on line {line_number}: {key!r}")
        os.environ.setdefault(key, value)


def repo_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a user path; relative values are anchored at the repository root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def env_path(name: str, default: str) -> Path:
    return repo_path(os.environ.get(name, default))


def configure_runtime_environment() -> None:
    """Set cache/data variables from relative public defaults."""
    load_dotenv()
    defaults = {
        "HF_HOME": ".cache/huggingface",
        "OPENPI_DATA_HOME": "data/openpi",
        "HF_LEROBOT_HOME": "data/lerobot",
        "ALAM_TORCH_HOME": ".cache/torch",
        "ALAM_RUNTIME_ROOT": ".runtime",
        "ALAM_OUTPUT_ROOT": "outputs",
    }
    for key, default in defaults.items():
        value = str(env_path(key, default))
        os.environ[key] = value
    os.environ["TORCH_HOME"] = os.environ["ALAM_TORCH_HOME"]
    os.environ.pop("LEROBOT_HOME", None)


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def require_dir(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path
