# <img src='https://abrain.one/img/lemur-nn-icon-64x64.png' width='32px'/> LLM Retrieval Augmented Generation
<sub><a href='https://pypi.python.org/pypi/nn-rag'><img src='https://img.shields.io/pypi/v/nn-rag.svg'/></a><br/>
short alias  <a href='https://pypi.python.org/pypi/lrag'>lrag</a></sub>

The original version of the NN RAG project was created by <strong>Waleed Khalid</strong> at the Computer Vision Laboratory, University of Würzburg, Germany.

<h3>Overview 📖</h3>

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
