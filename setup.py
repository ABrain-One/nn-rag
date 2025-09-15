from setuptools import setup, find_packages
from setuptools.command.install import install
from pathlib import Path
import subprocess
import sys

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

class PostInstallCommand(install):
    """Post-installation command to setup package data."""
    def run(self):
        install.run(self)
        # Package data setup is now handled by the path resolver in __init__.py
        print("Package installation completed. Data will be set up on first import.")

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
    package_data={
        "ab.rag": [
            "config/*.json",
            "*.py",
            "utils/*.py",
            "block/*.py",
        ],
    },
    include_package_data=True,
    install_requires=required,
    cmdclass={
        'install': PostInstallCommand,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
