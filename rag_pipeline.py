# rag_pipeline.py
import os
from typing import Tuple, List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}  # CPU-only for Streamlit Cloud
        )
        
        # Check if the FAISS index exists
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(f"FAISS index directory not found at {persist_directory}")
            
        # Check for required files
        required_files = ["index.faiss", "index.pkl"]
        for file in required_files:
            if not os.path.exists(os.path.join(persist_directory, file)):
                raise FileNotFoundError(f"Required file {file} not found in FAISS directory")
        
        vectorstore = FAISS.load_local(
            persist_directory,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"Error loading FAISS index: {str(e)}")
        raise

# =========================
# RAG (Retrieval + Generation)
# =========================
def rag_qa(query: str) -> Tuple[str, List[str]]:
    try:
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

        # Check if HF_TOKEN is available
        if not HF_TOKEN:
            raise ValueError("Hugging Face token not found. Please set HF_TOKEN in your environment variables.")
            
        llm = HuggingFaceHub(
            repo_id=LLM_REPO_ID,
            model_kwargs={"temperature": 0.2, "max_length": 512, "max_new_tokens": 256},
            huggingfacehub_api_token=HF_TOKEN
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

        result = qa({"query": query})
        
        # Extract source information
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    sources.append(doc.metadata['source'])
                else:
                    sources.append("Unknown source")
        
        return result["result"], sources
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        raise
