from setuptools import setup, find_packages

# Read the dependencies from requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="nn-rag",
    version="0.1.0",
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
