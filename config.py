# config.py
CLOUD_LLM_MODELS = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

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

# Theme
DEFAULT_THEME = "#e3f2fd"
DARK_THEME = "#2b2b2b"

# Recent files
RECENT_FILES_FILE = "recent_files.json"