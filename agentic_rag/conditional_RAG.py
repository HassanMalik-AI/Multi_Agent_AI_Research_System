import os
import sys
from typing import TypedDict, Annotated, List, Literal

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

# =======================================================================
# 1. Setup Models and Embeddings
# =======================================================================
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)   

llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# =======================================================================
# 2. Setup Retriever
# =======================================================================
def build_retriever():
    pdf_path = os.path.join(os.getcwd(), "agentic_rag", "machine-learning.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    docs = text_splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(docs, embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever  

retriever = build_retriever()

# =======================================================================
# 3. Define State
# =======================================================================
class State(TypedDict):
    question: str
    documents: List[str]
    generation: str

# =======================================================================
# 4. Define Nodes
# =======================================================================
def retrieve(state: State):
    """
    Retrieve documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    
    docs = retriever.invoke(question)
    doc_texts = [d.page_content for d in docs]
    
    return {"documents": doc_texts, "question": question}


def grade_documents(state: State):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide only the score ('yes' or 'no').""",
        input_variables=["question", "document"],
    )
    
    grader_llm = llm.with_config({"run_name": "Grader"})
    chain = prompt | grader_llm | StrOutputParser()
    
    filtered_docs = []
    for d in documents:
        score = chain.invoke({"question": question, "document": d})
        if "yes" in score.lower():
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            
    return {"documents": filtered_docs, "question": question}


def web_search(state: State):
    """
    Web search based on the re-phrased question.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        documents = []
        
    try:
        web_search_tool = TavilySearchResults(k=3)
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        documents.append(web_results)
    except Exception as e:
        print(f"---WEB SEARCH FAILED (Ensure TAVILY_API_KEY is set)---: {e}")
        
    return {"documents": documents, "question": question}


def generate(state: State):
    """
    Generate answer
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    context = "\n\n".join(documents)
    
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:""",
        input_variables=["question", "context"],
    )
    
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})
    
    return {"documents": documents, "question": question, "generation": generation}


# =======================================================================
# 5. Define Conditional Edges
# =======================================================================
def route_question(state: State) -> Literal["retrieve", "web_search"]:
    """
    Route question to web search or RAG.
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    
    # We use a simple router here. 
    # For now, we route everything to retrieve first.
    return "retrieve"

def decide_to_generate(state: State) -> Literal["generate", "web_search"]:
    """
    Determines whether to generate an answer, or fallback to web search.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    
    if not filtered_documents:
        print("---DECISION: ALL DOCUMENTS NOT RELEVANT TO QUESTION, WEB SEARCH---")
        return "web_search"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


# =======================================================================
# 6. Build Graph
# =======================================================================
workflow = StateGraph(State)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "retrieve": "retrieve",
        "web_search": "web_search",
    },
)

workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate",
    },
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()


if __name__ == "__main__":
    print("\n\n--- TESTING CONDITIONAL RAG ---")
    inputs = {"question": "What is machine learning?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}' execution finished.")
    
    print("\n\n--- FINAL OUTPUT ---")
    print(value["generation"])
