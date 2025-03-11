import os

# Model and embedding settings
CODE_EMBEDDING_MODEL_NAME = "microsoft/codebert-base"
EMBEDDING_BATCH_SIZE = 8
TOP_K_RETRIEVAL = 5

# Directories for saving index and data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Folder to store FAISS index
INDEX_DIR = os.path.join(PROJECT_ROOT, "ab", "rag", "index")
os.makedirs(INDEX_DIR, exist_ok=True)
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "code_index.faiss")

# Folders for external data
DATASET_DESC_DIR = os.path.join(PROJECT_ROOT, "ab", "rag", "dataset_descriptions")
os.makedirs(DATASET_DESC_DIR, exist_ok=True)

GITHUB_REPO_DIR = os.path.join(PROJECT_ROOT, "ab", "rag", "github_repos")
os.makedirs(GITHUB_REPO_DIR, exist_ok=True)

FINE_TUNED_MODEL_DIR= os.path.join(PROJECT_ROOT, "fine_tuned_model")
os.makedirs(FINE_TUNED_MODEL_DIR, exist_ok=True)
