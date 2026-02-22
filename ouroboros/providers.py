# ouroboros/providers.py
import os

PROVIDERS = {
    # OpenAI (основной)
    "openai": {
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key_env": "OPENAI_API_KEY",
    },

    # DeepSeek (OpenAI-compatible)
    "deepseek": {
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key_env": "DEEPSEEK_API_KEY",
    },

    # Qwen через DashScope (OpenAI-compatible "compatible-mode")
    "qwen": {
        "base_url": os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        "api_key_env": "DASHSCOPE_API_KEY",
    },

    # HuggingFace Router (OpenAI-compatible)
    "hf": {
        "base_url": os.environ.get("HF_OPENAI_BASE_URL", "https://router.huggingface.co/v1"),
        "api_key_env": "HF_TOKEN",
    },

    # Локальный vLLM (OpenAI-compatible)
    "vllm": {
        "base_url": os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        "api_key_env": "VLLM_API_KEY",  # может быть любым, если vLLM не проверяет
    },
}

def parse_model_id(model_id: str):
    """
    Поддержка двух форм:
      1) "gpt-5.2" -> provider="openai", model="gpt-5.2"
      2) "deepseek:deepseek-chat" -> provider="deepseek", model="deepseek-chat"
      3) "qwen:qwen3-coder-plus" -> provider="qwen", model="qwen3-coder-plus"
      4) "hf:Qwen/Qwen2.5-Coder-32B-Instruct" -> provider="hf", model="Qwen/Qwen2.5-Coder-32B-Instruct"
    """
    s = (model_id or "").strip()
    if ":" in s:
        prov, mdl = s.split(":", 1)
        return prov.strip(), mdl.strip()
    return "openai", s

def get_provider(name: str):
    p = PROVIDERS.get(name)
    if not p:
        raise KeyError(f"Unknown provider: {name}. Known: {sorted(PROVIDERS)}")
    key = os.environ.get(p["api_key_env"], "").strip()
    if not key:
        raise RuntimeError(f"Missing API key env var: {p['api_key_env']}")
    return p["base_url"].rstrip("/"), key
