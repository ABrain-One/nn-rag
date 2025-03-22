import time
import shelve
from github import Github, Auth, RateLimitExceededException, GithubException
from config.config import GITHUB_TOKEN, CACHE_FILE

# Create a GitHub instance using the token from config
g = Github(auth=Auth.Token(GITHUB_TOKEN))

def wait_for_rate_limit():
    core_limit = g.get_rate_limit().core
    reset_timestamp = core_limit.reset.timestamp()
    current_timestamp = time.time()
    wait_time = reset_timestamp - current_timestamp + 5  # extra buffer
    print(f"Rate limit exceeded. Waiting for {wait_time:.0f} seconds...")
    time.sleep(wait_time)

def build_query(base_keyword: str, qualifiers: dict) -> str:
    query = f"{base_keyword} in:readme in:description"
    for key, value in qualifiers.items():
        if value:
            query += f" {key}:{value}"
    return query

def search_repositories_with_cache(query: str, max_results: int = 100) -> list[dict]:
    with shelve.open(CACHE_FILE) as cache:
        if query in cache:
            print("Using cached repository results.")
            return cache[query]
        else:
            while True:
                try:
                    repos = g.search_repositories(query, sort="stars", order="desc")
                    break
                except RateLimitExceededException:
                    wait_for_rate_limit()
                except GithubException as e:
                    print("GitHub Exception:", e)
                    return []
            repo_list = []
            count = 0
            for repo in repos:
                repo_list.append({
                    "full_name": repo.full_name,
                    "stars": repo.stargazers_count,
                    "url": repo.html_url,
                    "description": repo.description or ""
                })
                count += 1
                if count >= max_results:
                    break
            cache[query] = repo_list
            return repo_list
