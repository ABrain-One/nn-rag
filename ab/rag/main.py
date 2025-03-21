#!/usr/bin/env python
"""
Interactive Inference Pipeline

This script loads the fine-tuned DeepSeek model and integrates it with a retrieval system.
It builds a FAISS index over external data sources (dataset descriptions and GitHub code)
and uses that context to construct a rich prompt for the model.
The user can interactively input queries via the terminal.
"""

# ========================== All Imports ==========================
import os
import json
import torch

# Environment configuration for DeepSpeed and CUDA memory management
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
# ========================== End Imports ==========================

# --------------------------
# Helper Functions
# --------------------------
def call_llm(prompt):
    """
    Generate a response from the fine-tuned model given a prompt.
    Adjust max_new_tokens if longer outputs are desired.
    """
    output = llm_generator(prompt)
    return output[0]["generated_text"]

def build_prompt(query, retrieved_results):
    """
    Build a final prompt by combining the user query with the retrieved context.
    The retrieved context is formatted as a series of sources and text snippets.
    """
    context = "\n\n".join(
        [f"[Source: {r['metadata']['source']}]:\n{r['text']}" for r in retrieved_results]
    )
    prompt = f"""
You are a knowledgeable assistant. Based on the following retrieved context, answer the user query in detail.

Retrieved Context:
{context}

User Query: {query}

Answer:
"""
    return prompt

def filter_results_by_keywords(results, keywords):
    """
    Optionally filter retrieval results to only include those containing at least one keyword.
    """
    filtered = []
    for r in results:
        text = r["text"].lower()
        if any(keyword.lower() in text for keyword in keywords):
            filtered.append(r)
    return filtered

# --------------------------
# Main Interactive Session
# --------------------------
def interactive_session():
    """
    Runs the interactive inference session:
      1. Runs setup to ensure external data is present.
      2. Builds the FAISS retrieval index.
      3. Prompts the user for a query.
      4. Retrieves relevant context, builds the final prompt, and generates an answer.
    """
    # Run setup to clone GitHub repositories and create dataset description files if needed
    print("=== Automated Data Setup ===")
    run_setup()

    # Build the corpus from external data sources (dataset descriptions & GitHub code)
    print("=== Building Retrieval Index ===")
    corpus_data = load_full_corpus(DATASET_DESC_DIR, GITHUB_REPO_DIR)
    print(f"Loaded {len(corpus_data)} items into the corpus.")

    if not corpus_data:
        print("Corpus is empty. Please check your data sources.")
        return

    # Initialize the retrieval system (uses CodeBERT embeddings)
    retrieval_system = CodeRetrieval(
        model_name=CODE_EMBEDDING_MODEL_NAME,
        batch_size=EMBEDDING_BATCH_SIZE,
        index_path=FAISS_INDEX_PATH
    )
    rebuild_index = True  # Set to False if you want to reuse an existing index
    if rebuild_index:
        retrieval_system.build_index(corpus_data)
    else:
        retrieval_system.load_index(FAISS_INDEX_PATH, corpus_data)

    # Interactive loop: user enters queries and receives model answers.
    print("Entering interactive mode. Type 'exit' to quit.")
    while True:
        user_query = input("\nEnter your query: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        # Retrieve context for the query
        results = retrieval_system.search(user_query, top_k=TOP_K_RETRIEVAL)
        print(f"\nTop {TOP_K_RETRIEVAL} retrieval results:")
        for i, res in enumerate(results, start=1):
            snippet_preview = res["text"][:80].replace("\n", " ")
            print(f"{i}) distance={res['distance']:.2f} | source={res['metadata']['source']}")
            print(f"   Snippet: {snippet_preview}...\n")

        # Optionally, further filter results by keywords
        keywords = ["nn.module", "def forward", "class", "model"]
        filtered_results = filter_results_by_keywords(results, keywords)
        final_results = filtered_results if filtered_results else results

        # Build the final prompt by combining user query and retrieved context
        final_prompt = build_prompt(user_query, final_results)
        print("\n=== Final Prompt for LLM ===")
        print(final_prompt)

        # Generate answer using the fine-tuned model
        answer = call_llm(final_prompt)
        print("\n=== LLM Answer ===")
        print(answer)

# --------------------------
# Load Fine-Tuned Model and Setup Generation Pipeline
# --------------------------
print("Loading fine-tuned model and tokenizer...")
# Directory where your fine-tuned model is saved
FINE_TUNED_MODEL_DIR = "./fine_tuned_model"
# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_DIR, trust_remote_code=True, torch_dtype="auto")

# Create the text-generation pipeline.
# Increase max_new_tokens if you want longer responses.
llm_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)

# --------------------------
# Directory Constants (from config)
# --------------------------
# Assuming these are defined in your config file
from config.config import DATASET_DESC_DIR, GITHUB_REPO_DIR, CODE_EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE, FAISS_INDEX_PATH, TOP_K_RETRIEVAL

# --------------------------
# Start the Interactive Session
# --------------------------
if __name__ == "__main__":
    interactive_session()
