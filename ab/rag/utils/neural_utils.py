import time
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from config.config import MODEL_NAME, GENERATOR_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = SentenceTransformer(MODEL_NAME)
generator = pipeline("text-generation", model=GENERATOR_MODEL)

def neural_rank_repositories(repo_results: list[dict], search_query: str) -> list[dict]:
    logger.info("Starting neural ranking for query: '%s'", search_query)
    start_time = time.time()
    query_embedding = model.encode([search_query])
    texts = [f"{repo['full_name']}. {repo['description']}" for repo in repo_results]
    embeddings = model.encode(texts)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    for repo, score in zip(repo_results, similarities):
        repo["neural_score"] = score
    ranked = sorted(repo_results, key=lambda x: x["neural_score"], reverse=True)
    elapsed = time.time() - start_time
    logger.info("Neural ranking completed in %.2f seconds.", elapsed)
    return ranked

def generate_response_from_results(
    search_query: str, 
    ranked_results: list[dict], 
    num_results: int = 3,
    temperature: float = 0.6,
    top_k: int = 40,
    top_p: float = 0.8,
    max_new_tokens: int = 120
) -> str:
    """
    Build a detailed prompt from the top-ranked results and generate a final response using
    the generative model. The prompt now instructs the model to provide:
      - A concise technical summary on implementing dynamic filtering.
      - A simple pseudo-code example.
      - Key design decisions highlighted.
    
    Generation parameters are adjusted for a bit less randomness (lower temperature) and 
    more focused sampling (top_k, top_p).
    """
    context = f"Search Query: {search_query}\n\nTop relevant repositories:\n"
    for repo in ranked_results[:num_results]:
        context += (
            f"- {repo['full_name']} (Stars: {repo['stars']})\n"
            f"  Description: {repo['description']}\n"
            f"  URL: {repo['url']}\n\n"
        )
    # Updated prompt instructing for a concise summary and pseudo-code
    context += (
        "Based on the above information, provide a concise technical summary that explains "
        "how one might implement dynamic filtering in a software project. Include a simple pseudo-code "
        "example and explicitly mention key design decisions (such as choice of data structures, performance considerations, "
        "and modular design)."
    )
    
    # Log prompt length for debugging
    logger.info("Prompt built with length: %d characters", len(context))
    
    response = generator(
        context,
        max_new_tokens=max_new_tokens,
        truncation=True,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )[0]['generated_text']
    
    return response
