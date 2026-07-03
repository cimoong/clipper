"""Application configuration loaded from environment / .env.

Run standalone to inspect the resolved config (API key masked):

    python -m clipforge.config
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv


def _get_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value is not None and value != "" else default


def _get_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key!r} must be an integer, got {value!r}") from exc


@dataclass(frozen=True)
class Config:
    """Resolved runtime configuration for ClipForge."""

    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    llm_provider: str = "gemini"
    gemini_model: str = "gemini-2.5-flash"
    anthropic_model: str = "claude-sonnet-5"
    whisper_model: str = "small"
    whisper_device: str = "auto"
    num_clips: int = 8
    clip_min_s: int = 25
    clip_max_s: int = 75
    data_dir: str = "./data"

    @classmethod
    def load(cls, *, env_file: str | os.PathLike[str] | None = None) -> "Config":
        """Load config from the process environment, populated from a .env file."""
        load_dotenv(dotenv_path=env_file, override=False)
        return cls(
            gemini_api_key=_get_str("GEMINI_API_KEY", ""),
            anthropic_api_key=_get_str("ANTHROPIC_API_KEY", ""),
            llm_provider=_get_str("LLM_PROVIDER", "gemini"),
            gemini_model=_get_str("GEMINI_MODEL", "gemini-2.5-flash"),
            anthropic_model=_get_str("ANTHROPIC_MODEL", "claude-sonnet-5"),
            whisper_model=_get_str("WHISPER_MODEL", "small"),
            whisper_device=_get_str("WHISPER_DEVICE", "auto"),
            num_clips=_get_int("NUM_CLIPS", 8),
            clip_min_s=_get_int("CLIP_MIN_S", 25),
            clip_max_s=_get_int("CLIP_MAX_S", 75),
            data_dir=_get_str("DATA_DIR", "./data"),
        )

    @property
    def data_path(self) -> Path:
        """`data_dir` as an absolute Path."""
        return Path(self.data_dir).expanduser().resolve()


def _mask_secret(value: str) -> str:
    if not value:
        return "(not set)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def format_config(config: Config) -> str:
    """Render the config for printing, masking secret fields."""
    secret_fields = {"gemini_api_key", "anthropic_api_key"}
    lines = ["ClipForge config:"]
    for field in fields(config):
        raw = getattr(config, field.name)
        shown = _mask_secret(raw) if field.name in secret_fields else raw
        lines.append(f"  {field.name} = {shown}")
    return "\n".join(lines)


def main() -> None:
    config = Config.load()
    print(format_config(config))


if __name__ == "__main__":
    main()
