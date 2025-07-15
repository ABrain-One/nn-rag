from setuptools import setup, find_packages
from pathlib import Path

# Read the dependencies from requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()
    
# Safely read the README.md file
def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""
    
def version():
    with open(Path(__file__).parent / 'version', 'r') as file:
        v = file.readline()
    return v
    
setup(
    name="nn-rag",
    version=version(),
    description="Neural Retrieval-Augmented Generation for GitHub code blocks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="ABrain One and contributors",
    author_email="AI@ABrain.one",
    url="https://github.com/ABrain-One/nn-rag",
    packages=find_packages(include=["ab.*"]),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
