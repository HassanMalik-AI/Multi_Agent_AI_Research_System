import os
import sys
from typing import TypedDict,Annotated

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)   
def build_retreiver():
    pdf_path = os.path.join(os.getcwd(), "agentic_rag", "machine-learning.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs=loader.load()
    
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    docs=text_splitter.split_documents(docs)
    
    vectorstore=FAISS.from_documents(docs,embedding)
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever  

retriever=build_retreiver()

llm = ChatGroq(model_name="llama3-70b-8192",temperature=0,api_key=os.getenv("GROQ_API_KEY"),temperature=0.5)

# state for history

class State(TypedDict):
    programme : str
    add_messages : Annotated[list,add_messages]
    query_type : str
    retriever_context : str

# node generation

def classifier_node(state : State) -> dict:
    """Look at the latest use message and decide wich path to take."""
    last_messge = state['messages'][-1].content
    prompt = (
        
    )
