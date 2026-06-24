from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from tools import web_search,get_url_content
from dotenv import load_dotenv
from rich import print
import os
load_dotenv()

llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),model_name="openai/gpt-oss-120b",temperature=0.1,max_tokens=1000,top_p=0.9,frequency_penalty=0.5,presence_penalty=0.5)


# first agent
def search_agent():
    agent = create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="You are an expert research search assistant. Use the web_search tool to find detailed information on the given topic. Return a summary of search results."
    )
    return (
        RunnableLambda(lambda x: {"messages": [("user", f"Search for detailed information on this topic: {x['topic']}")]})
        | agent
        | RunnableLambda(lambda x: x["messages"][-1].content)
    )

# second agent
def reader_agent():
    agent = create_agent(
        model=llm,
        tools=[get_url_content],
        system_prompt="You are a research reader agent. Based on the topic, determine the best Wikipedia URLs or other URLs, fetch their content using get_url_content tool, and read them. Then, extract and synthesize the key findings."
    )
    return (
        RunnableLambda(lambda x: {"messages": [("user", f"Read and extract information on this topic: {x['topic']}")]})
        | agent
        | RunnableLambda(lambda x: x["messages"][-1].content)
    )


writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()


#checking_chain 

checking_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

checking_chain = checking_prompt | llm | StrOutputParser()