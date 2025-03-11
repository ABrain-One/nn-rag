import os
import re

def detect_libraries(text):
    libs = []
    lower_text = text.lower()
    if "torch" in lower_text or "pytorch" in lower_text:
        libs.append("pytorch")
    if "tensorflow" in lower_text:
        libs.append("tensorflow")
    return libs

def chunk_code(text, min_chunk_length=50):
    # Split by double newlines (simple chunking)
    chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) >= min_chunk_length]
    return chunks

def load_dataset_descriptions_from_folder(folder_path):
    corpus = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt") or file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        description = f.read().strip()
                    corpus.append({
                        "text": description,
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
    corpus = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code = f.read()
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
    corpus = []
    corpus.extend(load_dataset_descriptions_from_folder(dataset_folder))
    # Each subfolder in repo_folder is assumed to be a cloned repo
    for item in os.listdir(repo_folder):
        repo_path = os.path.join(repo_folder, item)
        if os.path.isdir(repo_path):
            corpus.extend(load_code_from_repo(repo_path))
    return corpus
