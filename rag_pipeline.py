# rag_pipeline.py - Fix the embedding import
import os
import logging
from typing import Tuple, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import with error handling
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # Fixed import name
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain_community.llms import HuggingFaceHub
    from transformers import pipeline
    logger.info("✅ All imports successful")
except ImportError as e:
    logger.error(f"❌ Import failed: {e}")
    raise

# Configuration
FAISS_DIR = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_vectorstore():
    """Load the FAISS vector store"""
    try:
        if not os.path.exists(FAISS_DIR):
            raise FileNotFoundError(f"FAISS index not found at {FAISS_DIR}")
            
        embeddings = HuggingFaceEmbeddings(  # Fixed class name
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        
        vectorstore = FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("✅ FAISS index loaded successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"❌ Error loading FAISS: {e}")
        raise

def rag_qa(query: str) -> Tuple[str, List[str]]:
    """Main RAG QA function"""
    try:
        # Load vector store
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = f"""
You are an NCERT Class 8 Science AI Tutor. Answer the question based only on the provided context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {query}

Answer:
"""
        # Use a simpler model that works reliably
        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",  # Smaller, more reliable version
            max_length=512,
            temperature=0.2
        )
        
        # Generate answer
        result = generator(prompt, max_new_tokens=256)
        answer = result[0]['generated_text']
        
        # Extract sources
        sources = []
        for doc in docs:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                sources.append(doc.metadata['source'])
            else:
                sources.append("Textbook reference")
        
        return answer, sources
        
    except Exception as e:
        logger.error(f"❌ Error in RAG pipeline: {e}")
        raise
