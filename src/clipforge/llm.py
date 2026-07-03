"""Provider-agnostic LLM call factory for ClipForge.

The analyze (scoring) and render (metadata) stages both need a plain
``prompt -> raw model text`` callable. This module builds that callable for the
configured ``LLM_PROVIDER`` so those stages never import provider SDKs
directly:

- ``gemini``    — Google Gemini via the ``google-genai`` SDK (``GEMINI_API_KEY``)
- ``anthropic`` — Claude via the ``anthropic`` SDK (``ANTHROPIC_API_KEY``);
  models: ``claude-sonnet-5`` (default) or ``claude-haiku-4-5``

SDKs are imported lazily so tests (which inject their own callable) never need
them installed. Run standalone to see the resolved provider/model:

    python -m clipforge.llm
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from .config import Config

logger = logging.getLogger(__name__)

# A callable that takes the fully-rendered prompt and returns the raw model text.
LLMCall = Callable[[str], str]

PROVIDERS = ("gemini", "anthropic")

# Claude models offered in the settings UI. Sonnet for quality, Haiku for speed.
ANTHROPIC_MODELS = ("claude-sonnet-5", "claude-haiku-4-5")

_ANTHROPIC_MAX_TOKENS = 8192


class LLMError(Exception):
    """Raised when the LLM callable cannot be built (bad provider / missing key)."""


def build_llm_call(cfg: Config, *, system_prompt: str, temperature: float) -> LLMCall:
    """Build the LLM callable for ``cfg.llm_provider``.

    ``temperature`` is honoured by Gemini; Claude 5-family models reject
    non-default sampling parameters, so it is omitted there (the prompts already
    demand deterministic JSON output).
    """
    provider = cfg.llm_provider.strip().lower()
    if provider == "gemini":
        return _gemini_call(cfg, system_prompt=system_prompt, temperature=temperature)
    if provider == "anthropic":
        return _anthropic_call(cfg, system_prompt=system_prompt)
    raise LLMError(f"Unknown LLM_PROVIDER {cfg.llm_provider!r}; expected one of: {PROVIDERS}")


def _gemini_call(cfg: Config, *, system_prompt: str, temperature: float) -> LLMCall:
    """Gemini callable via google-genai (JSON response mode)."""
    if not cfg.gemini_api_key:
        raise LLMError("GEMINI_API_KEY is not set; cannot call the Gemini model.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=cfg.gemini_api_key)

    def _call(prompt: str) -> str:
        started = time.monotonic()
        resp = client.models.generate_content(
            model=cfg.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=temperature,
            ),
        )
        logger.info(
            "Gemini call: model=%s latency=%.1fs", cfg.gemini_model, time.monotonic() - started
        )
        return resp.text or ""

    return _call


def _anthropic_call(cfg: Config, *, system_prompt: str) -> LLMCall:
    """Claude callable via the official anthropic SDK.

    The SDK retries rate limits / 5xx with backoff itself. Text blocks are
    concatenated; the callers' JSON extractors tolerate code fences.
    """
    if not cfg.anthropic_api_key:
        raise LLMError("ANTHROPIC_API_KEY is not set; cannot call the Claude model.")

    import anthropic

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    def _call(prompt: str) -> str:
        started = time.monotonic()
        resp = client.messages.create(
            model=cfg.anthropic_model,
            max_tokens=_ANTHROPIC_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        logger.info(
            "Claude call: model=%s latency=%.1fs in=%d out=%d",
            cfg.anthropic_model,
            time.monotonic() - started,
            resp.usage.input_tokens,
            resp.usage.output_tokens,
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    return _call


def main() -> None:
    cfg = Config.load()
    model = cfg.gemini_model if cfg.llm_provider == "gemini" else cfg.anthropic_model
    print(f"LLM provider: {cfg.llm_provider}")
    print(f"LLM model:    {model}")


if __name__ == "__main__":
    main()
