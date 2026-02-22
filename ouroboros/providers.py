# ouroboros/providers.py
import os
from typing import Tuple, Dict

PROVIDERS: Dict[str, Dict[str, str]] = {
    # OpenAI
    "openai": {
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key_env": "OPENAI_API_KEY",
    },

    # DeepSeek (OpenAI-compatible)
    "deepseek": {
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key_env": "DEEPSEEK_API_KEY",
    },

    # Qwen via DashScope (OpenAI-compatible)
    # International: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    # China: https://dashscope.aliyuncs.com/compatible-mode/v1
    "qwen": {
        "base_url": os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        "api_key_env": "DASHSCOPE_API_KEY",
    },

    # Hugging Face router (OpenAI-compatible)
    "hf": {
        "base_url": os.environ.get("HF_OPENAI_BASE_URL", "https://router.huggingface.co/v1"),
        "api_key_env": "HF_TOKEN",
    },

    # Local vLLM (OpenAI-compatible)
    "vllm": {
        "base_url": os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        "api_key_env": "VLLM_API_KEY",
    },
}


def parse_model_id(model_id: str) -> Tuple[str, str]:
    """
    Supported forms:
      1) "gpt-5.2"                     -> ("openai", "gpt-5.2")
      2) "deepseek:deepseek-chat"      -> ("deepseek", "deepseek-chat")
      3) "qwen:qwen-max"               -> ("qwen", "qwen-max")
      4) "hf:Org/Model"                -> ("hf", "Org/Model")
    """
    s = (model_id or "").strip()
    if ":" in s:
        prov, mdl = s.split(":", 1)
        return prov.strip(), mdl.strip()
    return "openai", s


def get_provider(name: str) -> Tuple[str, str]:
    p = PROVIDERS.get(name)
    if not p:
        raise KeyError(f"Unknown provider: {name}. Known: {sorted(PROVIDERS)}")
    key_env = p["api_key_env"]
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {key_env}")
    return p["base_url"].rstrip("/"), api_key
