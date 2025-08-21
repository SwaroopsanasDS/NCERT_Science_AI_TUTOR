# rag_pipeline.py
import os
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# -------------------
# Load environment variables
# -------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is missing! Set it in .env or Streamlit secrets. "
        "Without it, HuggingFace models cannot be accessed."
    )

# -------------------
# Config
# -------------------
FAISS_DIR = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "google/flan-t5-base"  # public model

# -------------------
# Load FAISS
# -------------------
def load_vectorstore(persist_directory: str = FAISS_DIR) -> FAISS:
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"FAISS index not found at '{persist_directory}'. Run preprocessing first."
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

# -------------------
# RAG QA
# -------------------
def rag_qa(query: str) -> Tuple[str, list]:
    if not query.strip():
        return "❌ Query is empty.", []

    # Load vector store
    try:
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        return f"❌ Error loading FAISS: {e}", []

    # Prompt template
    prompt_template = """
You are an NCERT Class 8 Science AI Tutor.
Use the provided context to answer clearly, concisely, and in simple words.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # HuggingFaceHub LLM
    try:
        llm = HuggingFaceHub(
            repo_id=LLM_REPO_ID,
            model_kwargs={"temperature": 0.2, "max_new_tokens": 256},
            huggingfacehub_api_token=HF_TOKEN,
            task="text2text-generation"  # must specify task for Flan-T5
        )
    except Exception as e:
        return f"❌ Error initializing HuggingFaceHub LLM: {e}", []

    # Build RAG chain
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    except Exception as e:
        return f"❌ Error building RAG QA chain: {e}", []

    # Run query
    try:
        result = qa({"query": query})
        return result["result"], result.get("source_documents", [])
    except Exception as e:
        return f"❌ Error running RAG QA: {e}", []
