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

# Use small LLM to avoid memory issues on Streamlit Cloud
LLM_REPO_ID = "google/flan-t5-small"

# -----------------------------
# Load or rebuild FAISS index
# -----------------------------
def load_vectorstore() -> FAISS:
    if not Path(FAISS_DIR).exists() or not any(Path(FAISS_DIR).iterdir()):
        print("[INFO] FAISS index not found. Building new index...")
        build_faiss_index(txt_file=CLEANED_TXT, faiss_dir=FAISS_DIR)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}  # only CPU
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
Use the provided context to answer clearly, concisely, and in simple words.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = HuggingFaceHub(
        repo_id=LLM_REPO_ID,
        model_kwargs={
            "temperature": 0.2,
            "max_new_tokens": 256,
            "task": "text2text-generation"
        },
        huggingfacehub_api_token=HF_TOKEN
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    result = qa({"query": query})
    return result["result"], result.get("source_documents", [])
