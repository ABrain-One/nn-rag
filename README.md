# <img src='https://abrain.one/img/lemur-nn-icon-64x64.png' width='32px'/> LLM Retrieval Augmented Generation
<sub><a href='https://pypi.python.org/pypi/nn-rag'><img src='https://img.shields.io/pypi/v/nn-rag.svg'/></a> <a href="https://pepy.tech/project/nn-rag"><img alt="GitHub release" src="https://static.pepy.tech/badge/nn-rag"></a><br/>
short alias  <a href='https://pypi.python.org/pypi/lrag'>lrag</a></sub><br/><br/>

<img src='https://abrain.one/img/nnrag-logo.png' width='50%'/>

The original version of the NN RAG project was created by <strong>Waleed Khalid</strong> at the Computer Vision Laboratory, University of Würzburg, Germany.

<h3>📖 Overview</h3>

A minimal Retrieval-Augmented Generation (RAG) pipeline for code and dataset details.  
This project aims to provide LLMs with additional context from the internet or local repos, 
then optionally fine-tune the LLM for specific tasks.

## Requirements

- **Python** 3.8+ recommended  
- **Pip** or **Conda** for installing dependencies  
- (Optional) **GPU** with CUDA if you plan to do large-scale training

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
```

## Usage

### Command Line Interface

The package provides a command-line interface for extracting neural network blocks:

```bash
# Correct way to run (recommended)
python3 -m ab.rag --help

# Extract a specific block
python3 -m ab.rag --block ResNet

# Extract multiple blocks
python3 -m ab.rag --blocks ResNet VGG DenseNet

# Extract from JSON file (default)
python3 -m ab.rag

# Note: Avoid running 'python3 -m ab.rag.extract_blocks' as it may show warnings
```

### Python API

```python
from ab.rag import BlockExtractor, BlockValidator

# Initialize extractor
extractor = BlockExtractor()

# Warm up the index (clones repos and indexes if needed)
extractor.warm_index_once()

# Extract a single block
result = extractor.extract_single_block("ResNet")

# Extract multiple blocks
results = extractor.extract_multiple_blocks(["ResNet", "VGG"])

# Extract from JSON file (uses default nn_block_names.json)
results = extractor.extract_blocks_from_file()

# Extract with limit
results = extractor.extract_blocks_from_file(limit=10)

# Extract with custom JSON file
results = extractor.extract_blocks_from_file("custom_blocks.json")

# Extract with start_from parameter
results = extractor.extract_blocks_from_file(start_from="ResNet", limit=5)
```
## Citation

If you find this pipeline to be useful for your research, please consider citing our articles for <a target='_blank' href='https://arxiv.org/pdf/2512.04329'>extraction of algorithmic logic</a> and <a target='_blank' href='https://arxiv.org/pdf/2601.02997'>architecture design</a> with LLMs:

```bibtex
@article{ABrain.NN-RAG,
  title={A Retrieval-Augmented Generation Approach to Extracting Algorithmic Logic from Neural Networks},
  author={Khalid, Waleed and Ignatov, Dmitry and Timofte, Radu},
  journal={arXiv preprint},
  volume  = {arXiv:2512.04329},
  url = {https://arxiv.org/pdf/2512.04329}, 
  year={2025}
}

@article{ABrain.Architect,
	title={From Memorization to Creativity: LLM as a Designer of Novel Neural-Architectures},
	author={Khalid, Waleed and Ignatov, Dmitry and Timofte, Radu},
	journal={arXiv preprint},
	volume  = {arXiv:2601.02997},
	url = {https://arxiv.org/pdf/2601.02997}, 
	year={2026}
}
```
#### The idea and leadership of Dr. Ignatov
