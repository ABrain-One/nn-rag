import os
from config.config import CODE_EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE, FAISS_INDEX_PATH, TOP_K_RETRIEVAL, DATASET_DESC_DIR, GITHUB_REPO_DIR
from setup_data import run_setup
from utils.data_loader import load_full_corpus
from utils.retrieval import CodeRetrieval
import os
from config.config import CODE_EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE, FAISS_INDEX_PATH, TOP_K_RETRIEVAL, DATASET_DESC_DIR, GITHUB_REPO_DIR
from setup_data import run_setup
from utils.data_loader import load_full_corpus
from utils.retrieval import CodeRetrieval
import os
# Increase the Hugging Face download timeout
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Load configuration with trust_remote_code
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
print("Original quantization config:", getattr(config, "quantization_config", None))

# Override quantization if 'quant_method' is set to "fp8"
if hasattr(config, "quantization_config"):
    qc = config.quantization_config
    if not isinstance(qc, dict):
        try:
            qc = qc.to_dict()
        except Exception as e:
            print("Error converting quantization_config to dict:", e)
    if qc.get("quant_method", None) == "fp8":
        print("Overriding quant_method from fp8 to bitsandbytes_4bit")
        qc["quant_method"] = "bitsandbytes_4bit"
        config.quantization_config = qc
    else:
        print("Quantization method is:", qc.get("quant_method", None))
else:
    print("No quantization config found.")

# Set up BitsAndBytes configuration for 4-bit quantization
quant_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Load the model (force re-download if needed)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    trust_remote_code=True,
    quantization_config=quant_config,
    torch_dtype="auto",
    force_download=True  # Force re-download of files
)

llm_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

output = llm_generator("Explain how to define a basic neural network in PyTorch.")
print("LLM Output:", output[0]["generated_text"])

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
    """
    Filter results to keep only those that contain one or more of the specified keywords.
    """
    filtered = []
    for r in results:
        text = r["text"].lower()
        if any(keyword.lower() in text for keyword in keywords):
            filtered.append(r)
    return filtered

def main():
    print("=== Automated Data Setup ===")
    # Automatically clone repositories and create dataset files if missing.
    run_setup()

    print("=== RAG Pipeline Implementation ===")
    # Load corpus from dataset descriptions and cloned GitHub repos.
    corpus_data = load_full_corpus(DATASET_DESC_DIR, GITHUB_REPO_DIR)
    print(f"Loaded a total of {len(corpus_data)} items into the corpus.")

    if not corpus_data:
        print("Corpus is empty. Please check your data sources.")
        return

    # Build or load the FAISS index.
    retrieval_system = CodeRetrieval(
        model_name=CODE_EMBEDDING_MODEL_NAME,
        batch_size=EMBEDDING_BATCH_SIZE,
        index_path=FAISS_INDEX_PATH
    )
    rebuild_index = True  # Set to False if index already exists.
    if rebuild_index:
        retrieval_system.build_index(corpus_data)
    else:
        retrieval_system.load_index(FAISS_INDEX_PATH, corpus_data)

    # Query the index.
    user_query = "How to define a basic neural network in PyTorch?"
    results = retrieval_system.search(user_query, top_k=TOP_K_RETRIEVAL)
    print(f"\nTop {TOP_K_RETRIEVAL} results for query: '{user_query}'")
    for i, res in enumerate(results, start=1):
        snippet_preview = res["text"][:80].replace("\n", " ")
        print(f"{i}) distance={res['distance']:.2f} | source={res['metadata']['source']}")
        print(f"Snippet: {snippet_preview}...\n")

    # Optional: Further filter results by keywords relevant to neural network definitions.
    keywords = ["nn.module", "def forward", "class", "model"]
    filtered_results = filter_results_by_keywords(results, keywords)
    if filtered_results:
        final_results = filtered_results
        print(f"After filtering by keywords {keywords}, {len(filtered_results)} results remain.")
    else:
        final_results = results
        print("No additional filtering applied.")

    # Build prompt for the LLM.
    final_prompt = build_prompt(user_query, final_results)
    print("=== Prompt for LLM ===")
    print(final_prompt)

    # Call the LLM and print its answer.
    answer = call_llm(final_prompt)
    print("=== LLM Answer ===")
    print(answer)

if __name__ == "__main__":
    main()
