import os
import re

def detect_libraries(text):
    """
    Detect libraries referenced in the text.
    Returns a list of libraries (e.g., 'pytorch', 'tensorflow').
    """
    libs = []
    lower_text = text.lower()
    if "torch" in lower_text or "pytorch" in lower_text:
        libs.append("pytorch")
    if "tensorflow" in lower_text:
        libs.append("tensorflow")
    return libs

def chunk_code(text, min_chunk_length=50):
    """
    Splits code into chunks based on blank lines.
    (For a more robust solution, you might split by function or class definitions.)
    """
    chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) >= min_chunk_length]
    return chunks

def load_dataset_descriptions_from_folder(folder_path):
    """
    Loads dataset descriptions from text or Markdown files in folder_path.
    Each file becomes one corpus entry.
    """
    corpus = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt") or file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        description = f.read()
                    corpus.append({
                        "text": description.strip(),
                        "metadata": {
                            "type": "dataset",
                            "source": file_path,
                            "libs": []
                        }
                    })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return corpus

def load_code_from_repo(repo_path, file_extension=".py"):
    """
    Recursively loads code files from a cloned GitHub repository at repo_path.
    Returns a list of corpus entries.
    """
    corpus = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code = f.read()
                    # Chunk the code for more granular retrieval
                    chunks = chunk_code(code)
                    if not chunks:
                        chunks = [code]
                    for chunk in chunks:
                        corpus.append({
                            "text": chunk,
                            "metadata": {
                                "type": "code",
                                "source": file_path,
                                "libs": detect_libraries(code)
                            }
                        })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return corpus

def load_full_corpus(dataset_folder, repo_folder):
    """
    Loads and combines dataset descriptions and code from all GitHub repos.
    """
    corpus = []
    corpus.extend(load_dataset_descriptions_from_folder(dataset_folder))
    # Assume each subfolder in repo_folder is a cloned GitHub repo.
    for item in os.listdir(repo_folder):
        repo_path = os.path.join(repo_folder, item)
        if os.path.isdir(repo_path):
            corpus.extend(load_code_from_repo(repo_path))
    return corpus
