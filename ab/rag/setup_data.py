import os
import subprocess
from config.config import GITHUB_REPO_DIR, DATASET_DESC_DIR

def ensure_repo_cloned(repo_url, local_name):
    repo_path = os.path.join(GITHUB_REPO_DIR, local_name)
    if not os.path.isdir(repo_path):
        print(f"Cloning {repo_url} into {repo_path}...")
        os.makedirs(GITHUB_REPO_DIR, exist_ok=True)
        subprocess.run(["git", "clone", "--single-branch", "--branch", "main", repo_url, repo_path], check=True)
    else:
        print(f"Repository '{local_name}' already exists at {repo_path}.")

def ensure_dataset_file(file_name, content=None):
    file_path = os.path.join(DATASET_DESC_DIR, file_name)
    if not os.path.isfile(file_path):
        print(f"Creating dataset file: {file_path}")
        os.makedirs(DATASET_DESC_DIR, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content or "No content specified.")
    else:
        print(f"Dataset file '{file_name}' already exists at {file_path}.")

def run_setup():
    # Clone relevant GitHub repositories for NN training code
    ensure_repo_cloned("https://github.com/osmr/pytorchcv.git", "pytorchcv")
    ensure_repo_cloned("https://github.com/pytorch/vision.git", "pytorch_vision")
    
    # Create dataset description file(s)
    ensure_dataset_file(
        file_name="cifar10.txt",
        content=(
            "The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, "
            "with 6,000 images per class. It is widely used for image classification tasks."
        )
    )
    print("Data setup complete.")

if __name__ == "__main__":
    run_setup()
