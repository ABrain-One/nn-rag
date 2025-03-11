#!/usr/bin/env python
import os
from config.config import CODE_EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE, FAISS_INDEX_PATH, TOP_K_RETRIEVAL, DATASET_DESC_DIR, GITHUB_REPO_DIR
from setup_data import run_setup
from utils.data_loader import load_full_corpus
from utils.retrieval import CodeRetrieval
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the fine-tuned model from our fine-tuning step
FINE_TUNED_MODEL_DIR = "./fine_tuned_model"

# Load tokenizer and model from the fine-tuned checkpoint
tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_DIR, trust_remote_code=True, torch_dtype="auto")

# Create a text-generation pipeline
llm_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

def call_llm(prompt):
    output = llm_generator(prompt)
    return output[0]["generated_text"]

def build_prompt(query, retrieved_results):
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
    filtered = []
    for r in results:
        text = r["text"].lower()
        if any(keyword.lower() in text for keyword in keywords):
            filtered.append(r)
    return filtered

def interactive_session():
    # Run the data setup and build the retrieval index (if needed)
    print("Setting up external data sources...")
    run_setup()
    corpus_data = load_full_corpus(DATASET_DESC_DIR, GITHUB_REPO_DIR)
    print(f"Loaded {len(corpus_data)} items into the corpus.")

    retrieval_system = CodeRetrieval(
        model_name=CODE_EMBEDDING_MODEL_NAME,
        batch_size=EMBEDDING_BATCH_SIZE,
        index_path=FAISS_INDEX_PATH
    )
    # Build (or load) the FAISS index
    rebuild_index = True  # Set this to False if you want to reuse an existing index
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
        # Optionally filter results further by keywords
        keywords = ["nn.module", "def forward", "class", "model"]
        filtered_results = filter_results_by_keywords(results, keywords)
        final_results = filtered_results if filtered_results else results

        final_prompt = build_prompt(user_query, final_results)
        print("\n=== Final Prompt for LLM ===")
        print(final_prompt)
        answer = call_llm(final_prompt)
        print("\n=== LLM Answer ===")
        print(answer)

if __name__ == "__main__":
    interactive_session()
