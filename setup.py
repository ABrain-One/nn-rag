from setuptools import setup, find_packages
from pathlib import Path

# Read the dependencies from requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()
    
def version():
    with open(Path(__file__).parent / 'version', 'r') as file:
        v = file.readline()
    return v
    
setup(
    name="nn-rag",
    version=version(),
    description="Neural Retrieval-Augmented Generation for GitHub Search",
    author="ABrain One and contributors",
    author_email="AI@ABrain.one",
    url="https://github.com/ABrain-One/nn-rag",
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
