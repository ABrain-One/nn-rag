# For running the application from the command line (local testing)
from api import retrieve_and_generate

if __name__ == "__main__":
    keyword = input("Enter search keyword: ").strip()
    language = input("Filter by language (press enter to skip): ").strip() or None
    owner = input("Restrict search to a specific owner (press enter to skip): ").strip() or None
    stars = input("Filter by stars (e.g., >100, <50) (press enter to skip): ").strip() or None
    
    result = retrieve_and_generate(keyword, language, owner, stars)
    
    if "error" in result:
        print("Error:", result["error"])
    else:
        print("\nFinal Generated Response:")
        print(result.get("generated_response"))
        print("\nTop Ranked Repositories:")
        for repo in result.get("results", []):
            print(f"{repo['full_name']} - {repo['stars']} stars - Neural Score: {repo['neural_score']:.4f} - URL: {repo['url']}")
