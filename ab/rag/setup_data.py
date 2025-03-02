import os
import subprocess
from config.config import GITHUB_REPO_DIR, DATASET_DESC_DIR

def ensure_repo_cloned(repo_url, local_name):
    """
    Check if a repo folder exists in GITHUB_REPO_DIR. If not, clone it.
    """
    repo_path = os.path.join(GITHUB_REPO_DIR, local_name)
    if not os.path.isdir(repo_path):
        print(f"Cloning {repo_url} into {repo_path}...")
        os.makedirs(GITHUB_REPO_DIR, exist_ok=True)
        subprocess.run(["git", "clone", repo_url, repo_path], check=True)
    else:
        print(f"Repository '{local_name}' already exists at {repo_path}.")

def ensure_dataset_file(file_name, content=None):
    """
    Ensure a dataset description file exists in DATASET_DESC_DIR.
    If not, create it with the provided content.
    """
    file_path = os.path.join(DATASET_DESC_DIR, file_name)
    if not os.path.isfile(file_path):
        print(f"Creating dataset file: {file_path}")
        os.makedirs(DATASET_DESC_DIR, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content or "No content specified.")
    else:
        print(f"Dataset file '{file_name}' already exists at {file_path}.")

def run_setup():
    """
    Orchestrates cloning/downloading all needed repos and dataset files.
    """
    # Clone GitHub repos (add more as needed)
    ensure_repo_cloned("https://github.com/pytorch/examples.git", "pytorch_examples")
    ensure_repo_cloned("https://github.com/pytorch/tutorials.git", "pytorch_tutorials")
    
    # Create dataset description file(s)
    ensure_dataset_file(
        file_name="cifar10.txt",
        content=(
            "The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, "
            "with 6,000 images per class. It is widely used for image classification."
        )
    )
    print("\nData setup complete.\n")

if __name__ == "__main__":
    run_setup()
