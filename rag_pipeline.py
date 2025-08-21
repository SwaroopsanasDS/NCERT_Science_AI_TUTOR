# rag_pipeline.py
import os
from typing import List, Tuple
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS

from huggingface_hub import InferenceClient

# =========================
# CONFIG
# =========================
FAISS_DIR = "data/faiss_index"
HF_TOKEN = os.getenv("HF_TOKEN") or "YOUR_HF_TOKEN_HERE"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "google/flan-t5-small"  # small model for Streamlit Cloud

# =========================
# API-based embeddings
# =========================
class HFAPIEmbeddings:
    """Embeddings via Hugging Face Inference API feature_extraction endpoint."""
    def __init__(self, model_name: str, api_key: str):
        self.client = InferenceClient(model=model_name, token=api_key)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for t in texts:
            out = self.client.feature_extraction(t, truncate=True)
            # mean-pool token embeddings
            if isinstance(out, list) and isinstance(out[0], list):
                vec = sum(out) / len(out)
            else:
                vec = out
            vectors.append(vec)
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

# =========================
# Load FAISS index (already built locally & pushed to repo)
# =========================
def load_vectorstore(persist_directory: str = FAISS_DIR) -> FAISS:
    embeddings = HFAPIEmbeddings(model_name=EMBEDDING_MODEL, api_key=HF_TOKEN)
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
        model_kwargs={"temperature": 0.2, "max_new_tokens": 256},
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
