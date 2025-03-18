# config/config.py

import os

# Base directories (adjust as needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATASET_DESC_DIR = os.path.join(REPO_ROOT, "dataset_descriptions")
GITHUB_REPO_DIR = os.path.join(REPO_ROOT, "github_repos")
FAISS_INDEX_PATH = os.path.join(REPO_ROOT, "index", "faiss_index.index")
FINE_TUNED_MODEL_DIR = os.path.join(REPO_ROOT, "fine_tuned_model")

CODE_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 8
TOP_K_RETRIEVAL = 5

HF_TOKEN = "YOUR_HF_ACCESS_TOKEN_HERE"  # If needed for private model access
