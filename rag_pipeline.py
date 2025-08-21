# rag_pipeline.py
import os
import numpy as np
from typing import List, Tuple

from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub


# =========================
# CONFIG
# =========================
FAISS_DIR = "data/faiss_index"

# 🔑 Put your Hugging Face token here (or set it in Streamlit secrets as HF_TOKEN)
HF_TOKEN = os.getenv("HF_TOKEN") or "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"
if not HF_TOKEN or HF_TOKEN.startswith("hf_your_"):
    raise ValueError("Please set a valid HF token in rag_pipeline.py (HF_TOKEN).")

# Use the SAME model you used to build the FAISS index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Small, public, reliable text2text model
LLM_REPO_ID = "google/flan-t5-base"


# =========================
# Minimal HF Inference Embeddings
# (no extra packages; matches LangChain's Embeddings protocol)
# =========================
class HFAPIEmbeddings:
    """Embeddings via Hugging Face Inference API feature_extraction endpoint."""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = InferenceClient(model=model_name, token=api_key)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for t in texts:
            out = self.client.feature_extraction(t, truncate=True)
            # out can be (tokens x dim) or (dim,)
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
                # mean-pool token embeddings
                arr = np.array(out, dtype=np.float32)
                vec = arr.mean(axis=0).tolist()
            else:
                vec = list(out)
            vectors.append(vec)
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]


# =========================
# Load FAISS Vector Store
# =========================
def load_vectorstore(persist_directory: str = FAISS_DIR) -> FAISS:
    """Load FAISS index using API-based embeddings (no local downloads)."""
    embeddings = HFAPIEmbeddings(model_name=EMBEDDING_MODEL, api_key=HF_TOKEN)
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
