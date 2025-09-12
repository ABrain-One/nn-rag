from setuptools import setup
from setuptools.command.install import install
from pathlib import Path
import subprocess
import sys

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
            result = subprocess.run([
                sys.executable, 
                str(Path(__file__).parent / "ab" / "rag" / "setup_data.py")
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                print("Package data setup completed successfully!")
            else:
                print(f"Package data setup failed: {result.stderr}")
                print("   The package will work but will clone repos on first use.")
        except Exception as e:
            print(f"Package data setup failed: {e}")
            print("   The package will work but will clone repos on first use.")

# Minimal setup.py for backward compatibility
# Most configuration is now in pyproject.toml
setup(
    version=version(),
    cmdclass={
        'install': PostInstallCommand,
    },
)
