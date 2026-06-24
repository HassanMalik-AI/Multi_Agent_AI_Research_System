from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from dotenv import load_dotenv
from rich import print
import os
import sys

# Set standard output encoding to UTF-8 to handle unicode characters gracefully on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Use this tool when you need to search the web for current information, return a string of results,urls,titles"""
    result = tavily.search(query=query,max_results=3)
    out = []
    for res in result['results']:
        out.append(f"""Title: {res['title']}\nContent: {res['content']}\nURL: {res['url']}\n""")
    return "\n\n".join(out)


if __name__ == "__main__":
    print(web_search.invoke({"query": "who is the current prime minister of pakistan?"}))



@tool
def get_url_content(url: str, max_length: int = 1000) -> str:
    """
    Fetches and returns the main text content of a given URL.
    
    Args:
        url: The URL to fetch content from.
        max_length: The maximum number of characters to return (default 10000).
        
    Returns:
        The extracted text content or an error message.
    """
    try:
        # Adding a User-Agent header to mimic a browser and avoid 403 Forbidden errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3' 
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an exception for bad responses (4xx or 5xx)
        
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove unwanted elements
    for script in soup(["script", "style", "header", "footer", "nav"]):
        script.decompose()
    
    text = soup.get_text(separator=" ", strip=True)[:max_length]

    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text
    
if __name__ == "__main__":
    print(get_url_content.invoke({"url": "https://en.wikipedia.org/wiki/Prime_Minister_of_Pakistan"}))