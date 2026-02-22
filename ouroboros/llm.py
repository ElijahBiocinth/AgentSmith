"""
Ouroboros — LLM client.

Multi-provider edition (no OpenRouter).
- Primary agent + tools are expected to run on OpenAI (best compatibility).
- Other providers (DeepSeek/Qwen/HF/vLLM) can be used for fallback and review.

This module must remain the single place that talks to LLM APIs.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.providers import parse_model_id, get_provider

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "gpt-5-mini"


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)

    # Monetary cost is generally not returned by OpenAI-compatible endpoints.
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


class LLMClient:
    """OpenAI-compatible multi-provider wrapper."""

    def __init__(self):
        # Require OPENAI_API_KEY because core agent + web_search expect OpenAI tooling
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise RuntimeError("OPENAI_API_KEY is not set (required for core agent/web_search).")

    def _client_for(self, provider: str):
        """Create an OpenAI SDK client for a given provider (OpenAI-compatible)."""
        base_url, api_key = get_provider(provider)
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=base_url)

    def chat_any(
        self,
        messages: List[Dict[str, Any]],
        model_id: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 8192,
        tool_choice: str = "auto",
        temperature: float = 0.2,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Call OpenAI-compatible /chat/completions on provider implied by model_id.
        model_id examples:
          - "gpt-5.2"
          - "deepseek:deepseek-chat"
          - "qwen:qwen-max"
          - "hf:Qwen/Qwen2.5-Coder-32B-Instruct"
        """
        provider, model = parse_model_id(model_id)
        client = self._client_for(provider)

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        _ = normalize_reasoning_effort(reasoning_effort)

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()

        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        # Optional cached tokens extraction (provider-dependent)
        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])

        if not usage.get("cache_write_tokens"):
            prompt_details_for_write = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_for_write, dict):
                cache_write = (
                    prompt_details_for_write.get("cache_write_tokens")
                    or prompt_details_for_write.get("cache_creation_tokens")
                    or prompt_details_for_write.get("cache_creation_input_tokens")
                )
                if cache_write:
                    usage["cache_write_tokens"] = int(cache_write)

        return msg, usage

    def chat_with_fallbacks(
        self,
        messages: List[Dict[str, Any]],
        primary_model_id: str,
        fallback_list_csv: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 8192,
        tool_choice: str = "auto",
        temperature: float = 0.2,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Try primary → fallback chain on:
          - exception
          - empty content
        """
        models = [primary_model_id] + [m.strip() for m in (fallback_list_csv or "").split(",") if m.strip()]
        last_err = None

        for mid in models:
            try:
                msg, usage = self.chat_any(
                    messages=messages,
                    model_id=mid,
                    tools=tools,
                    reasoning_effort=reasoning_effort,
                    max_tokens=max_tokens,
                    tool_choice=tool_choice,
                    temperature=temperature,
                )
                content = msg.get("content")
                if content is None or (isinstance(content, str) and not content.strip()):
                    last_err = f"Empty response from {mid}"
                    continue
                return msg, usage
            except Exception as e:
                last_err = f"{mid}: {e}"
                continue

        raise RuntimeError(f"All models failed. Last error: {last_err}")

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 8192,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Backward-compatible entrypoint used by the rest of Ouroboros.
        Uses OUROBOROS_MODEL_FALLBACK_LIST if present.
        """
        fallbacks = os.environ.get("OUROBOROS_MODEL_FALLBACK_LIST", "").strip()
        if fallbacks:
            return self.chat_with_fallbacks(
                messages=messages,
                primary_model_id=model,
                fallback_list_csv=fallbacks,
                tools=tools,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                temperature=0.2,
            )
        return self.chat_any(
            messages=messages,
            model_id=model,
            tools=tools,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            temperature=0.2,
        )

    def default_model(self) -> str:
        return os.environ.get("OUROBOROS_MODEL", "gpt-5.2")

    def available_models(self) -> List[str]:
        main = os.environ.get("OUROBOROS_MODEL", "gpt-5.2")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models
