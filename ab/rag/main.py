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
llm_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

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

def main():
    print("=== Automated Data Setup ===")
    run_setup()

    print("=== RAG Pipeline Implementation ===")
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
    rebuild_index = True  # Change to False if the index exists
    if rebuild_index:
        retrieval_system.build_index(corpus_data)
    else:
        retrieval_system.load_index(FAISS_INDEX_PATH, corpus_data)

    user_query = "How to define a basic neural network in PyTorch?"
    results = retrieval_system.search(user_query, top_k=TOP_K_RETRIEVAL)
    print(f"\nTop {TOP_K_RETRIEVAL} results for query: '{user_query}'")
    for i, res in enumerate(results, start=1):
        snippet_preview = res["text"][:80].replace("\n", " ")
        print(f"{i}) distance={res['distance']:.2f} | source={res['metadata']['source']}")
        print(f"Snippet: {snippet_preview}...\n")

    # Optionally filter the results further by keywords
    keywords = ["nn.module", "def forward", "class", "model"]
    filtered_results = filter_results_by_keywords(results, keywords)
    if filtered_results:
        final_results = filtered_results
        print(f"After filtering by keywords {keywords}, {len(filtered_results)} results remain.")
    else:
        final_results = results
        print("No additional filtering applied.")

    final_prompt = build_prompt(user_query, final_results)
    print("=== Prompt for LLM ===")
    print(final_prompt)

    answer = call_llm(final_prompt)
    print("=== LLM Answer ===")
    print(answer)

if __name__ == "__main__":
    main()
