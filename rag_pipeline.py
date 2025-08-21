# rag_pipeline.py
import os
import numpy as np
from typing import List, Tuple
from huggingface_hub import InferenceClient, HfApi
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
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
# Validate Hugging Face repo access
# -------------------
def validate_hf_token_and_repo(token: str, repo_id: str):
    try:
        api = HfApi(token=token)
        model_info = api.model_info(repo_id)
        print(f"✅ Hugging Face token works. Model '{model_info.modelId}' is accessible.")
    except Exception as e:
        raise ValueError(
            f"❌ Cannot access Hugging Face model '{repo_id}'. Check your HF_TOKEN and repo ID.\nDetails: {e}"
        )

validate_hf_token_and_repo(HF_TOKEN, LLM_REPO_ID)

# -------------------
# Minimal HF API Embeddings (no local downloads)
# -------------------
class HFAPIEmbeddings:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = InferenceClient(model=model_name, token=api_key)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for t in texts:
            out = self.client.feature_extraction(t, truncate=True)
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
                vec = np.array(out).mean(axis=0).tolist()
            else:
                vec = list(out)
            vectors.append(vec)
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

# -------------------
# Load FAISS
# -------------------
def load_vectorstore(persist_directory: str = FAISS_DIR) -> FAISS:
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"FAISS index not found at '{persist_directory}'. Run preprocessing first."
        )
    embeddings = HFAPIEmbeddings(model_name=EMBEDDING_MODEL, api_key=HF_TOKEN)
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

# -------------------
# RAG QA (latest LangChain compatible)
# -------------------
def rag_qa(query: str) -> Tuple[str, list]:
    if not query.strip():
        return "❌ Query is empty.", []

    # Load vector store
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Prompt
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

    # HuggingFaceHub LLM (latest LangChain)
    try:
        llm = HuggingFaceHub.from_model_id(
            model_id=LLM_REPO_ID,
            model_kwargs={"temperature": 0.2, "max_new_tokens": 256},
            huggingfacehub_api_token=HF_TOKEN
        )
    except Exception as e:
        return f"❌ Error initializing HuggingFaceHub LLM: {e}", []

    # Build RAG chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    # Run query
    try:
        result = qa({"query": query})
        return result["result"], result.get("source_documents", [])
    except Exception as e:
