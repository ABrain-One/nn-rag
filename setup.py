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
        # Try to setup package data, but don't fail installation if it fails
        try:
            print("Setting up package data...")
            
            # Find the installed package location
            import site
            import os
            
            # Get the installation directory
            install_dir = self.install_lib
            if not install_dir:
                # Fallback to site-packages
                install_dir = site.getusersitepackages()
            
            # Find the ab package directory
            ab_dir = None
            for root, dirs, files in os.walk(install_dir):
                if 'ab' in dirs and os.path.exists(os.path.join(root, 'ab', 'rag')):
                    ab_dir = os.path.join(root, 'ab')
                    break
            
            if ab_dir and os.path.exists(os.path.join(ab_dir, 'rag', 'setup_data.py')):
                setup_script = os.path.join(ab_dir, 'rag', 'setup_data.py')
                result = subprocess.run([
                    sys.executable, setup_script
                ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
                
                if result.returncode == 0:
                    print("Package data setup completed successfully!")
                else:
                    print(f"Package data setup failed: {result.stderr}")
                    print("   The package will work but will clone repos on first use.")
            else:
                print("Could not find setup script in installed package")
                print("   The package will work but will clone repos on first use.")
                
        except Exception as e:
            print(f"Package data setup failed: {e}")
            print("   The package will work but will clone repos on first use.")

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
