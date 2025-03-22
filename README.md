# <img src='https://abrain.one/img/lemur-nn-icon-64x64.png' width='32px'/> LLM Retrieval Augmented Generation

# nn-rag

A minimal Retrieval-Augmented Generation (RAG) pipeline for code and dataset details.  
This project aims to provide LLMs with additional context from the internet or local repos, 
then optionally fine-tune the LLM for specific tasks.

## Requirements

- **Python** 3.8+ recommended  
- **Pip** or **Conda** for installing dependencies  
- (Optional) **GPU** with CUDA if you plan to use `faiss-gpu` or do large-scale training

### Installing Dependencies

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows

2. ### Latest Development Version

Install the latest version directly from GitHub:

```bash
pip install git+https://github.com/ABrain-One/nn-rag --upgrade