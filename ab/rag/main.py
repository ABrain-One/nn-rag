# main.py

#!/usr/bin/env python
"""
Interactive Inference Pipeline with Retrieval-Augmented Generation (RAG)
and Explicit Prompt Instructions

Key Updates:
- Provides two few-shot examples (basic, MobileNet adaptation).
- Uses multi-turn fallback if 'Predicted Accuracy:' or 'Epoch:' is missing.
- Allows up to 4096 tokens in generation and 1024 tokens in the prompt.
"""

import os
import json
import torch

# Set environment variables for DeepSpeed and CUDA memory management
os.environ["DS_CUDA_VERSION"] = "12.4"
os.environ["WANDB_MODE"] = "disabled"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

from utils.data_loader import load_full_corpus
from utils.retrieval import CodeRetrieval
from setup_data import run_setup
from config.config import (
    DATASET_DESC_DIR, GITHUB_REPO_DIR, CODE_EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE, FAISS_INDEX_PATH, TOP_K_RETRIEVAL, 
    FINE_TUNED_MODEL_DIR
)

# ---------------------------------------------------------------------
# Increased generation parameters:
# ---------------------------------------------------------------------
MAX_NEW_TOKENS = 4096  
PROMPT_MAX_TOKENS = 1024  

print("Loading fine-tuned model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_DIR, trust_remote_code=True, torch_dtype="auto")

llm_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS
)

def truncate_prompt(prompt, max_tokens=PROMPT_MAX_TOKENS):
    tokenized = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokenized) > max_tokens:
        tokenized = tokenized[:max_tokens]
        return tokenizer.decode(tokenized, skip_special_tokens=True)
    return prompt

def call_llm(prompt):
    """
    Generates a response from the fine-tuned model using controlled parameters.
    If the response is missing performance metrics, do a multi-turn approach.
    """
    prompt = truncate_prompt(prompt, PROMPT_MAX_TOKENS)
    output = llm_generator(
        prompt,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        num_beams=5,
        early_stopping=True,
        max_new_tokens=MAX_NEW_TOKENS
    )
    generated_text = output[0]["generated_text"]

    # Multi-turn fallback if "Predicted Accuracy:" or "Epoch:" is missing
    if "Predicted Accuracy:" not in generated_text or "Epoch:" not in generated_text:
        system_reminder = (
            "\nSYSTEM: Your answer did not include 'Predicted Accuracy:' or 'Epoch:'. "
            "Please revise your response to include both fields in the format:\n"
            "Predicted Accuracy: <some number>\nEpoch: <some number>\n"
        )
        second_prompt = truncate_prompt(prompt + system_reminder, PROMPT_MAX_TOKENS)
        second_output = llm_generator(
            second_prompt,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            num_beams=5,
            early_stopping=True,
            max_new_tokens=MAX_NEW_TOKENS
        )
        generated_text = second_output[0]["generated_text"]

    return generated_text

def build_prompt(query, retrieved_results):
    """
    Constructs a final prompt by combining retrieved context with user query.
    Two few-shot examples are provided for demonstration:
     1) Basic: "How to define a basic neural network in PyTorch?"
     2) Adaptation: "How to adapt MobileNet to CIFAR-100?"
    """
    few_shot = (
        # Example 1 (basic)
        "Example #1:\n"
        "User Query: How to define a basic neural network in PyTorch?\n"
        "Answer:\n"
        "```python\n"
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class BasicNN(nn.Module):\n"
        "    def __init__(self, input_size, hidden_size, output_size):\n"
        "        super(BasicNN, self).__init__()\n"
        "        self.fc1 = nn.Linear(input_size, hidden_size)\n"
        "        self.fc2 = nn.Linear(hidden_size, output_size)\n\n"
        "    def forward(self, x):\n"
        "        x = torch.relu(self.fc1(x))\n"
        "        return self.fc2(x)\n"
        "```\n"
        "Predicted Accuracy: 92.1%\n"
        "Epoch: 125\n\n"
        # Example 2 (adaptation)
        "Example #2:\n"
        "User Query: How to adapt the PyTorchCV MobileNet model to CIFAR-100?\n"
        "Answer:\n"
        "```python\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "from pytorchcv.model_provider import get_model as ptcv_get_model\n\n"
        "# Load MobileNet from pytorchcv, then modify the final layer for 100 classes\n"
        "class MobileNetCIFAR100(nn.Module):\n"
        "    def __init__(self, pretrained=False):\n"
        "        super(MobileNetCIFAR100, self).__init__()\n"
        "        self.base = ptcv_get_model('mobilenet_w1', pretrained=pretrained)\n"
        "        # replace final classifier\n"
        "        self.base.output = nn.Linear(in_features=1024, out_features=100)\n"
        "    def forward(self, x):\n"
        "        return self.base(x)\n"
        "```\n"
        "Predicted Accuracy: 85.4%\n"
        "Epoch: 45\n\n"
    )

    context = "\n\n".join(
        [f"[Source: {r['metadata']['source']}]:\n{r['text']}" for r in retrieved_results]
    )

    prompt = (
        f"{few_shot}"
        "You are a knowledgeable assistant. Based on the retrieved context below, provide a short, concise answer to the user query.\n"
        "Your answer must include a brief neural network model code snippet, 'Predicted Accuracy:', and 'Epoch:'.\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        "Answer:"
    )
    return prompt

def filter_results_by_keywords(results, keywords):
    filtered = []
    for r in results:
        src = r["metadata"]["source"].lower()
        if "github_repos" in src:
            filtered.append(r)
        else:
            text = r["text"].lower()
            if any(keyword.lower() in text for keyword in keywords):
                filtered.append(r)
    return filtered

def interactive_session():
    print("=== Automated Data Setup ===")
    run_setup()

    print("=== Building Retrieval Index ===")
    corpus_data = load_full_corpus(DATASET_DESC_DIR, GITHUB_REPO_DIR)
    print(f"Loaded {len(corpus_data)} items into the corpus.")
    if not corpus_data:
        print("Corpus is empty. Please check your data sources.")
        return

    retrieval_system = CodeRetrieval(
        model_name=CODE_EMBEDDING_MODEL_NAME,
        batch_size=EMBEDDING_BATCH_SIZE,
        index_path=FAISS_INDEX_PATH
    )
    rebuild_index = True
    if rebuild_index:
        retrieval_system.build_index(corpus_data)
    else:
        retrieval_system.load_index(FAISS_INDEX_PATH, corpus_data)

    print("Entering interactive mode. Type 'exit' to quit.")
    while True:
        user_query = input("\nEnter your query: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        results = retrieval_system.search(user_query, top_k=TOP_K_RETRIEVAL)
        print(f"\nTop {TOP_K_RETRIEVAL} retrieval results:")
        for i, res in enumerate(results, start=1):
            snippet_preview = res["text"][:80].replace("\n", " ")
            print(f"{i}) distance={res['distance']:.2f} | source={res['metadata']['source']}")
            print(f"   Snippet: {snippet_preview}...\n")

        keywords = ["nn.module", "def forward", "class", "model"]
        filtered_results = filter_results_by_keywords(results, keywords)
        final_results = filtered_results if filtered_results else results

        final_prompt = build_prompt(user_query, final_results)
        print("\n=== Final Prompt for LLM (for debugging) ===")
        print(final_prompt)

        answer = call_llm(final_prompt)
        print("\n=== LLM Answer ===")
        print(answer)

if __name__ == "__main__":
    interactive_session()
