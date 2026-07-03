"""Tests for the provider dispatcher in clipforge.llm (no network, no SDKs)."""

from __future__ import annotations

import pytest

from clipforge.config import Config
from clipforge.llm import LLMError, build_llm_call


def test_unknown_provider_raises() -> None:
    cfg = Config(llm_provider="openai")
    with pytest.raises(LLMError, match="Unknown LLM_PROVIDER"):
        build_llm_call(cfg, system_prompt="x", temperature=0.2)


def test_gemini_without_key_raises() -> None:
    cfg = Config(llm_provider="gemini", gemini_api_key="")
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        build_llm_call(cfg, system_prompt="x", temperature=0.2)


def test_anthropic_without_key_raises() -> None:
    cfg = Config(llm_provider="anthropic", anthropic_api_key="")
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        build_llm_call(cfg, system_prompt="x", temperature=0.2)


def test_provider_is_case_insensitive() -> None:
    cfg = Config(llm_provider="Anthropic", anthropic_api_key="")
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        build_llm_call(cfg, system_prompt="x", temperature=0.2)
