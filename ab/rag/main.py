"""
Interactive Inference Pipeline with Retrieval-Augmented Generation (RAG)
and Explicit Prompt Instructions

This script loads the fine-tuned DeepSeek model and integrates it with a retrieval system.
It builds a FAISS index over external data (GitHub code and dataset descriptions) and
constructs a prompt that explicitly instructs the model to generate a short, concise answer
that includes:
  - A brief NN model code snippet.
  - Predicted performance metrics (accuracy and epoch).

If the generated answer is incomplete or needs refinement, you can iterate by adjusting the prompt.
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
from config.config import DATASET_DESC_DIR, GITHUB_REPO_DIR, CODE_EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE, FAISS_INDEX_PATH, TOP_K_RETRIEVAL, FINE_TUNED_MODEL_DIR
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
    Constructs a final prompt by combining retrieved context with the user query.
    The prompt explicitly instructs the model to provide a short answer that includes:
      - A brief NN model code snippet.
      - The predicted accuracy.
      - The predicted epoch.
    """
    context = "\n\n".join(
        [f"[Source: {r['metadata']['source']}]:\n{r['text']}" for r in retrieved_results]
    )
    prompt = f"""
You are a knowledgeable assistant. Based on the following retrieved context, provide a **short, concise answer** to the user query.
Your answer must include:
- A brief neural network (NN) model code snippet.
- Predicted Accuracy (e.g., 'Predicted Accuracy: 92.1%').
- Predicted Epoch (e.g., 'Epoch: 125').

Retrieved Context:
{context}

User Query: {query}

Answer (include 'Predicted Accuracy:' and 'Epoch:' in your response):
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
# Interactive Session Function
# --------------------------
def interactive_session():
    """
    Runs the interactive inference session:
      1. Runs setup to ensure external data (GitHub repos, dataset descriptions) is available.
      2. Builds the FAISS retrieval index from the corpus.
      3. Prompts the user for a query.
      4. Retrieves relevant context and builds a final prompt.
      5. Generates an answer from the fine-tuned model.
    """
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
    # Build (or load) the FAISS index
    rebuild_index = True  # Set to False to reuse an existing index
    if rebuild_index:
        retrieval_system.build_index(corpus_data)
    else:
        retrieval_system.load_index(FAISS_INDEX_PATH, corpus_data)

    # Interactive loop: prompt user, retrieve context, and generate answer.
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

        # Optionally filter results by keywords for more relevant context
        keywords = ["nn.module", "def forward", "class", "model"]
        filtered_results = filter_results_by_keywords(results, keywords)
        final_results = filtered_results if filtered_results else results

        # Build the final prompt with explicit instructions
        final_prompt = build_prompt(user_query, final_results)
        print("\n=== Final Prompt for LLM ===")
        print(final_prompt)

        # Generate answer from the fine-tuned model
        answer = call_llm(final_prompt)
        print("\n=== LLM Answer ===")
        print(answer)
        # If the answer doesn't include the predicted metrics, you can iteratively refine the prompt.

# --------------------------
# Load Fine-Tuned Model and Set Up Inference Pipeline
# --------------------------
print("Loading fine-tuned model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_DIR, trust_remote_code=True, torch_dtype="auto")

# Create a text-generation pipeline; adjust max_new_tokens as needed.
llm_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)

# --------------------------
# Start the Interactive Session
# --------------------------
if __name__ == "__main__":
    interactive_session()
