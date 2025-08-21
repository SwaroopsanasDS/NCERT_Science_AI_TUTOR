# rag_pipeline.py
import os
from typing import List, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# =========================
# CONFIG
# =========================
FAISS_DIR = "data/faiss_index"

# Hugging Face token (set in Streamlit secrets or environment variable)
HF_TOKEN = os.getenv("HF_TOKEN") or "hf_your_token_here"
if not HF_TOKEN or HF_TOKEN.startswith("hf_your_"):
    raise ValueError("Please set a valid HF token in rag_pipeline.py (HF_TOKEN).")

# Embeddings model (must match preprocessing)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM model
LLM_REPO_ID = "google/flan-t5-base"

# =========================
# Load FAISS vector store
# =========================
def load_vectorstore(persist_directory: str = FAISS_DIR) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

# =========================
# RAG QA
# =========================
def rag_qa(query: str) -> Tuple[str, list]:
    # Load vectorstore
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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

    # HuggingFace LLM
    llm = HuggingFaceHub(
        repo_id=LLM_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 256},
        task="text2text-generation"
    )

    # Retrieval QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    result = qa({"query": query})
    answer = result["result"]
    sources = result.get("source_documents", [])
    return answer, sources
