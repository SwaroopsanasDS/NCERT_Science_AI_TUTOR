# rag_pipeline.py
import os
from typing import Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# =========================
# CONFIG
# =========================
FAISS_DIR = "data/faiss_index"
LLM_REPO_ID = "google/flan-t5-base"  # small, CPU-friendly text2text model
HF_TOKEN = os.getenv("HF_TOKEN")  # Set this in Streamlit secrets
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =========================
# Load FAISS Vector Store
# =========================
def load_vectorstore(persist_directory: str = FAISS_DIR) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}  # CPU-only for Streamlit Cloud
    )
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


# =========================
# RAG (Retrieval + Generation)
# =========================
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
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = HuggingFaceHub(
        repo_id=LLM_REPO_ID,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 256},
        huggingfacehub_api_token=HF_TOKEN,
        task="text2text-generation"  # required for Flan-T5
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
