import os

# Embedding model to use (CodeBERT)
CODE_EMBEDDING_MODEL_NAME = "microsoft/codebert-base"

# Batch size for embedding
EMBEDDING_BATCH_SIZE = 8

# Number of top results to retrieve for a query
TOP_K_RETRIEVAL = 5

# Define project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Folder to store the FAISS index (ensure this folder exists)
INDEX_DIR = os.path.join(PROJECT_ROOT, "index")
os.makedirs(INDEX_DIR, exist_ok=True)
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "code_index.faiss")

# Data folders: adjust these paths as needed.
# Folder containing dataset description files (txt or md)
DATASET_DESC_DIR = os.path.join(PROJECT_ROOT, "dataset_descriptions")
os.makedirs(DATASET_DESC_DIR, exist_ok=True)

# Folder containing cloned GitHub repositories (each repo in its own subfolder)
GITHUB_REPO_DIR = os.path.join(PROJECT_ROOT, "github_repos")
os.makedirs(GITHUB_REPO_DIR, exist_ok=True)
