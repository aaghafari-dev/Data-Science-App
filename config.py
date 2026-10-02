# config.py
"""Central application configuration and model catalog."""

# Broad API model catalog. The combo is editable, so users can enter any
# provider/model identifier released after this catalog without waiting for
# an application update. Provider routing is resolved by services.llm_config.
CLOUD_LLM_MODELS = [
    # OpenAI
    "gpt-5.6-sol", "gpt-5.6", "gpt-5.6-terra", "gpt-5.6-luna",
    "gpt-5.5", "gpt-5.4", "gpt-5.3", "gpt-5.2", "gpt-5.1",
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini",
    "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    # Anthropic
    "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5",
    "claude-opus-4-5", "claude-sonnet-4-5", "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest",
    "claude-3-haiku-20240307",
    # Google Gemini
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.1-pro-preview",
    "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash",
    "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite",
    # DeepSeek
    "deepseek-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner",
    # xAI
    "grok-4.7", "grok-4", "grok-4-latest", "grok-3", "grok-3-mini",
    # Mistral
    "mistral-medium-3.5", "mistral-small-4", "mistral-large-3",
    "ministral-3-14b", "ministral-3-8b", "ministral-3-3b",
    # Cohere / Meta / other commonly exposed API identifiers
    "command-a", "command-r-plus", "command-r", "llama-4-maverick", "llama-4-scout",
]

LOCAL_LLM_MODELS = {
    "Qwen2.5-7B-Instruct": "models--Qwen--Qwen2.5-7B-Instruct",
    "Qwen2.5-3B-Instruct": "models--Qwen--Qwen2.5-3B-Instruct",
    "Qwen2.5-0.5B-Instruct": "models--Qwen--Qwen2.5-0.5B-Instruct",
    "DialoGPT-medium": "models--microsoft--DialoGPT-medium",
    "DialoGPT-small": "models--microsoft--DialoGPT-small",
    "flan-t5-base": "models--google--flan-t5-base",
    "gpt2": "models--gpt2",
}

ALL_LLM_MODELS = CLOUD_LLM_MODELS + list(LOCAL_LLM_MODELS.keys())
DEFAULT_API_KEY = ""
HF_CACHE_DIR = r"C:\Users\ASUS\.cache\huggingface\hub"

# API provider metadata. These are optional integrations; the application can
# still expose/edit model IDs even if a provider SDK is not installed.
API_PROVIDER_METADATA = {
    "openai": {"label": "OpenAI", "env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"},
    "anthropic": {"label": "Anthropic", "env": "ANTHROPIC_API_KEY", "base_url": ""},
    "google": {"label": "Google Gemini", "env": "GOOGLE_API_KEY", "base_url": ""},
    "deepseek": {"label": "DeepSeek", "env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com"},
    "xai": {"label": "xAI", "env": "XAI_API_KEY", "base_url": "https://api.x.ai/v1"},
    "mistral": {"label": "Mistral", "env": "MISTRAL_API_KEY", "base_url": "https://api.mistral.ai/v1"},
    "cohere": {"label": "Cohere", "env": "COHERE_API_KEY", "base_url": ""},
    "generic": {"label": "OpenAI-compatible / custom", "env": "OPENAI_API_KEY", "base_url": ""},
}

# Theme
DEFAULT_THEME = "#e3f2fd"
DARK_THEME = "#2b2b2b"

# Recent files
RECENT_FILES_FILE = "recent_files.json"
