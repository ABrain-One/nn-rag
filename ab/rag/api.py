from pandas import DataFrame
from utils.github_utils import build_query, search_repositories_with_cache
from utils.neural_utils import neural_rank_repositories, generate_response_from_results

def retrieve_and_generate(
    keyword: str,
    language: str = None,
    owner: str = None,
    stars: str = None,
    max_results: int = 100,
    num_top: int = 3
) -> dict:
    """
    Retrieve GitHub repositories based on search parameters, perform neural ranking,
    and generate a final response using a generative model.
    
    Parameters:
      - keyword: Base search keyword.
      - language: (Optional) Filter by programming language.
      - owner: (Optional) Restrict search to a specific repository owner.
      - stars: (Optional) Filter by stars (e.g., ">100", "<50").
      - max_results: Maximum number of repositories to retrieve.
      - num_top: Number of top results to use for generation.
    
    Returns:
      A dictionary with the built query, top-ranked results, and generated response.
    """
    qualifiers = {}
    if language:
        qualifiers["language"] = language
    if owner:
        qualifiers["user"] = owner
    if stars:
        qualifiers["stars"] = stars

    query = build_query(keyword, qualifiers)
    repo_results = search_repositories_with_cache(query, max_results=max_results)
    if not repo_results:
        return {"error": "No results found."}
    ranked_repos = neural_rank_repositories(repo_results, keyword)
    generated_response = generate_response_from_results(keyword, ranked_repos, num_results=num_top)
    return {
        "query": query,
        "results": ranked_repos[:10],
        "generated_response": generated_response
    }

def retrieve_and_generate_df(
    keyword: str,
    language: str = None,
    owner: str = None,
    stars: str = None,
    max_results: int = 100,
    num_top: int = 3
) -> DataFrame:
    """
    Return the top repository results as a pandas DataFrame.
    """
    data = retrieve_and_generate(keyword, language, owner, stars, max_results, num_top)
    if "error" in data:
        return DataFrame()
    from pandas import DataFrame
    return DataFrame(data["results"])
