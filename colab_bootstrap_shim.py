# colab_bootstrap_shim.py
# OpenAI-only bootstrap (no OpenRouter required)

import os
import subprocess
from typing import Optional

try:
    from google.colab import userdata  # type: ignore
except Exception:
    userdata = None  # running outside colab


def get_secret(name: str, required: bool = False, default: Optional[str] = None) -> Optional[str]:
    v = None
    if userdata is not None:
        try:
            v = userdata.get(name)
        except Exception:
            v = None
    if v is None or str(v).strip() == "":
        v = os.environ.get(name, default)
    if required:
        assert v is not None and str(v).strip() != "", f"Missing required secret: {name}"
    return v


def export_secret_to_env(name: str, required: bool = False, default: Optional[str] = None) -> None:
    v = get_secret(name, required=required, default=default)
    if v is None:
        v = ""
    os.environ[name] = str(v)


# Install deps that the launcher expects (keep minimal; requirements.txt is installed separately)
subprocess.run(
    ["python", "-m", "pip", "install", "-q", "openai>=1.0.0", "requests", "httpx"],
    check=False,
)

# REQUIRED for OpenAI-only mode
export_secret_to_env("OPENAI_API_KEY", required=True)

# REQUIRED runtime secrets (same as upstream except openrouter)
export_secret_to_env("TELEGRAM_BOT_TOKEN", required=True)
export_secret_to_env("TOTAL_BUDGET", required=True)
export_secret_to_env("GITHUB_TOKEN", required=True)

# Optional keys for additional providers
export_secret_to_env("DEEPSEEK_API_KEY", required=False, default="")
export_secret_to_env("DASHSCOPE_API_KEY", required=False, default="")
export_secret_to_env("HF_TOKEN", required=False, default="")

# Keep legacy variable but NOT required
export_secret_to_env("OPENROUTER_API_KEY", required=False, default="")

print("✅ bootstrap ok: OPENAI_API_KEY + TELEGRAM_BOT_TOKEN + TOTAL_BUDGET + GITHUB_TOKEN present")
