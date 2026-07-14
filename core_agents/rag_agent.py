import os
import sys
from typing import Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0.1,
    max_tokens=1200,
    top_p=0.9,
    frequency_penalty=0.2,
    presence_penalty=0.2,
)


def build_document_retriever(text: str, title: Optional[str] = None):
    document_title = title or "uploaded_document"
    docs = [Document(page_content=text, metadata={"source": document_title})]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def generate_rag_answer(question: str, document_text: str, document_name: Optional[str] = None) -> str:
    retriever = build_document_retriever(document_text, title=document_name)
    retrieved_docs = retriever.get_relevant_documents(question)

    if not retrieved_docs:
        return "No relevant document context could be found for this question."

    context = "\n\n".join(
        [f"Source: {doc.metadata.get('source', 'document')}\n{doc.page_content}" for doc in retrieved_docs]
    )

    prompt = (
        "You are an expert research assistant that answers user questions from the given document content. "
        "Use only the provided document context and do not hallucinate new information. If the answer is not in the text, say that the information is not available in the provided document.\n\n"
        f"Document Context:\n{context}\n\n"
        f"User Question:\n{question}\n\n"
        "Answer clearly, reference the document when possible, and keep the response concise."
    )

    return llm.predict(prompt)


def run_rag_qa(question: str, document_text: str, document_name: Optional[str] = None) -> dict:
    answer = generate_rag_answer(question, document_text, document_name)
    return {
        "query": question,
        "document_name": document_name or "uploaded_document",
        "answer": answer,
    }


if __name__ == "__main__":
    question = input("Enter your question about the document: ")
    print("Paste the document content below, then press Enter:")
    document_text = sys.stdin.read().strip()
    result = run_rag_qa(question, document_text)
    print(result["answer"])
