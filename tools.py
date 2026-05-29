from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from dotenv import load_dotenv
from rich import print
import os
load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Use this tool when you need to search the web for current information"""
    result = tavily.search(query=query,max_results=3)
    if result and result['results']:
        return result['results'][0]['content']

query = "What is the latest news on AI?"
query2 = "What is the capital of France?"

print(web_search.invoke({"query": query}))
print(web_search.invoke({"query": query2}))
