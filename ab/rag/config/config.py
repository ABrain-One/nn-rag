import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
CACHE_FILE = "search_cache.db"
MODEL_NAME = "all-MiniLM-L6-v2"
GENERATOR_MODEL = "gpt2"
