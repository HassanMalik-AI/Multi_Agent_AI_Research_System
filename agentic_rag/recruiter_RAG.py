import os
import sys
from typing import List

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from core_agents.tools import web_search

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)

RECRUITER_KB = [
    {
        "title": "Candidate Sourcing and Outreach",
        "content": (
            "Top recruiters build talent pipelines by combining proactive sourcing, targeted outreach, "
            "referrals, and employer branding. They use role-based messaging, focus on candidate fit, "
            "and balance passive and active candidate engagement."
        ),
    },
    {
        "title": "Job Description and Hiring Strategy",
        "content": (
            "A strong recruiter defines clear role requirements, realistic skills, and outcomes. "
            "They align the job description with culture, diversity goals, and market compensation "
            "while prioritizing candidate experience throughout the hiring funnel."
        ),
    },
    {
        "title": "Screening and Interview Best Practices",
        "content": (
            "Effective screening relies on structured interviews, competency-based questions, "
            "skills assessment, and consistent scorecards. Recruiters should avoid bias, ask "
            "behavioral questions, and validate both technical and cultural fit."
        ),
    },
    {
        "title": "Retention and Talent Intelligence",
        "content": (
            "Recruiters who partner with hiring managers also monitor retention signals, "
            "career pathing, and workforce planning. They gather market intelligence on salary, "
            "skill demand, and competitor hiring trends to advise on recruitment strategy."
        ),
    },
]


def build_recruiter_retriever():
    """Create a recruiter-focused local retriever from internal knowledge snippets."""
    docs = [Document(page_content=item["content"], metadata={"title": item["title"]}) for item in RECRUITER_KB]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


retriever = build_recruiter_retriever()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0.1,
    max_tokens=1200,
    top_p=0.9,
    frequency_penalty=0.2,
    presence_penalty=0.2,
)


def retrieve_internal_context(query: str) -> List[Document]:
    """Retrieve recruiter knowledge that is most relevant to the query."""
    return retriever.get_relevant_documents(query)


def grade_internal_relevance(query: str, retrieved_text: str) -> str:
    """Assess whether the local recruiter knowledge is sufficient for the query."""
    prompt = (
        "You are a recruitment intelligence gatekeeper. Evaluate whether the following recruiter knowledge "
        "is relevant enough to answer the user query accurately.

User Query:\n"
        f"{query}\n\n"
        "Retrieved Recruiter Context:\n"
        f"{retrieved_text}\n\n"
        "Answer with either 'relevant' or 'irrelevant', and include one brief sentence explaining the decision."
    )
    result = llm.predict(prompt)
    return result.strip().lower()


def generate_recruiter_response(query: str, context: str, extra_research: str, used_web: bool) -> str:
    """Generate the final recruiter-specific answer using the available context."""
    prompt = (
        "You are a high-performing recruiter intelligence agent. Use the provided recruiter knowledge and any "
        "external research to answer the user's query clearly, professionally, and with practical recruiting guidance.\n\n"
        f"User Query:\n{query}\n\n"
        f"Internal Recruiter Context:\n{context}\n\n"
    )
    if extra_research:
        prompt += f"External Research / Web Search Results:\n{extra_research}\n\n"
        prompt += "If external research is included, combine it with the internal recruiter context to produce the best possible answer.\n\n"
    prompt += (
        "Output a structured response with:\n"
        "1. A short summary of hiring guidance.\n"
        "2. Key recruiting recommendations or steps.\n"
        "3. Practical next actions for a recruiter or hiring manager.\n"
        "If external research was used, mention that web search data was incorporated."
    )
    return llm.predict(prompt)


async def run_recruiter_rag(query: str) -> dict:
    """Run the recruiter-specific conditional RAG flow."""
    retrieved_docs = retrieve_internal_context(query)
    internal_context = "\n\n".join([f"- {doc.metadata.get('title', 'Internal Knowledge')}: {doc.page_content}" for doc in retrieved_docs])

    relevance = grade_internal_relevance(query, internal_context)
    fallback_required = "irrelevant" in relevance
    external_research = ""
    source = "internal"

    if fallback_required:
        source = "web_fallback"
        external_research = web_search.invoke({"query": query})

    answer = generate_recruiter_response(query, internal_context, external_research, fallback_required)

    return {
        "agent_type": "Recruiter Intelligence RAG",
        "rag_style": "Conditional Recruitment RAG",
        "query": query,
        "context_source": source,
        "relevance_evaluation": relevance,
        "internal_context": internal_context,
        "external_research": external_research,
        "answer": answer,
    }


if __name__ == "__main__":
    import asyncio

    user_query = input("Enter a recruiter query or hiring problem: ")
    result = asyncio.run(run_recruiter_rag(user_query))
    print("\n--- Recruiter RAG Result ---\n")
    print(f"Agent Type: {result['agent_type']}")
    print(f"RAG Style: {result['rag_style']}")
    print(f"Context Source: {result['context_source']}")
    print(f"Relevance Evaluation: {result['relevance_evaluation']}\n")
    print(result['answer'])
