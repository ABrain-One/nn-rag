# <img src='https://abrain.one/img/lemur-nn-icon-64x64.png' width='32px'/> LLM Retrieval Augmented Generation

# nn-rag

A minimal Retrieval-Augmented Generation (RAG) pipeline for code and dataset details.  
This project aims to provide LLMs with additional context from the internet or local repos, 
then optionally fine-tune the LLM for specific tasks.

## Requirements

- **Python** 3.8+ recommended  
- **Pip** or **Conda** for installing dependencies  
- (Optional) **GPU** with CUDA if you plan to use `faiss-gpu` or do large-scale training


## Create and Activate a Virtual Environment (recommended)
For Linux/Mac:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
For Windows:
   ```bash
   python3 -m venv .venv
   .venv\Scripts\activate
   ```

All subsequent commands are provided for Linux/Mac OS. For Windows, please replace ```source .venv/bin/activate``` with ```.venv\Scripts\activate```.

Installing from GitHub to get the most recent code of NN RAG:
```bash
source .venv/bin/activate
pip install git+https://github.com/ABrain-One/nn-rag --upgrade --force
```
