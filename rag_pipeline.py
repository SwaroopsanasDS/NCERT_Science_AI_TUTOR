# rag_pipeline.py
import os
from pathlib import Path
from typing import Tuple

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from preprocess_pipeline import build_faiss_index, CLEANED_TXT, FAISS_DIR, EMBEDDING_MODEL

# -----------------------------
# CONFIG
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN") or "hf_your_token_here"
if not HF_TOKEN:
    raise ValueError("Set your Hugging Face token in HF_TOKEN env variable or here.")

LLM_REPO_ID = "google/flan-t5-base"


# -----------------------------
# Load or rebuild FAISS index
# -----------------------------
def load_vectorstore() -> FAISS:
    # Rebuild FAISS if folder doesn't exist or is empty
    if not Path(FAISS_DIR).exists() or not any(Path(FAISS_DIR).iterdir()):
        print("[INFO] FAISS index not found. Building new index...")
        build_faiss_index(txt_file=CLEANED_TXT, faiss_dir=FAISS_DIR)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu", "device_map": None}  # force CPU
    )

    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


# -----------------------------
# RAG QA function
# -----------------------------
def rag_qa(query: str) -> Tuple[str, list]:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = """
You are an NCERT Class 8 Science AI Tutor.
Use the provided context to answer
