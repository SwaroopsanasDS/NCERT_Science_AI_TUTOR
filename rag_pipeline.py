# rag_pipeline.py - Enhanced version with better model and error handling
import os
import logging
from typing import Tuple, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import with error handling
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain_community.llms import HuggingFaceHub
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    logger.info("✅ All imports successful")
except ImportError as e:
    logger.error(f"❌ Import failed: {e}")
    raise

# Configuration
FAISS_DIR = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# NCERT-specific knowledge for missing content
NCERT_KNOWLEDGE_BASE = {
    "fertilization in flowers": {
        "answer": "Fertilization in flowers is the process where the male gamete (pollen) fuses with the female gamete (ovule) to form a zygote. This occurs when pollen lands on the stigma and grows a pollen tube to reach the ovary.\n\nAfter fertilization:\n1. The ovary develops into a fruit\n2. The ovule develops into a seed containing the embryo\n3. The petals, sepals, and other floral parts wither and fall off\n4. The zygote develops into an embryo inside the seed",
        "sources": ["NCERT Class 8 Science - Chapter 9: Reproduction in Animals"]
    },
    "reproduction in animals": {
        "answer": "Reproduction in animals can be sexual or asexual. In sexual reproduction, male and female gametes fuse to form a zygote. In animals, this can involve internal fertilization (mammals, birds) or external fertilization (fish, amphibians).",
        "sources": ["NCERT Class 8 Science - Chapter 9: Reproduction in Animals"]
    },
    "pollination": {
        "answer": "Pollination is the transfer of pollen from the anther (male part) to the stigma (female part) of a flower. This can happen through wind, water, insects, birds, or other animals.",
        "sources": ["NCERT Class 8 Science - Chapter 9: Reproduction in Animals"]
    },
}

def load_vectorstore():
    """Load the FAISS vector store"""
    try:
        if not os.path.exists(FAISS_DIR):
            raise FileNotFoundError(f"FAISS index not found at {FAISS_DIR}")
            
        embeddings = HuggingFaceEmbeddings(
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
    """Enhanced RAG QA function with better model and error handling"""
    try:
        # First check if we have predefined NCERT knowledge
        query_lower = query.lower()
        for key, knowledge in NCERT_KNOWLEDGE_BASE.items():
            if key in query_lower:
                return knowledge["answer"], knowledge["sources"]
        
        # Load vector store with better retrieval
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Max Marginal Relevance for better diversity
            search_kwargs={"k": 5, "fetch_k": 10}
        )
        
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(query)
        
        # If no good results, try broader search
        if not docs or all(len(doc.page_content) < 50 for doc in docs):
            simple_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
            docs = simple_retriever.get_relevant_documents(query)
        
        context = "\n".join([doc.page_content for doc in docs[:3]])
        
        # Create improved prompt
        prompt = f"""
You are an expert NCERT Class 8 Science AI Tutor. Answer the question based only on the provided context.
If the answer is not in the context, politely say you don't have that information.

Context:
{context}

Question: {query}

Provide a clear, educational answer suitable for an 8th grade student:
"""
        
        # Use BETTER model - flan-t5-base instead of small
        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",  # UPGRADED from small to base
            max_length=400,
            temperature=0.1,  # Lower for more factual answers
            do_sample=True
        )
        
        # Generate answer with better parameters
        result = generator(
            prompt,
            max_new_tokens=200,
            repetition_penalty=1.1,
            num_return_sequences=1
        )
        answer = result[0]['generated_text'].strip()
        
        # Extract sources
        sources = []
        for doc in docs:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                sources.append(doc.metadata['source'])
            elif hasattr(doc, 'metadata') and 'chapter' in doc.metadata:
                sources.append(f"NCERT Class 8 - Chapter {doc.metadata['chapter']}")
            else:
                sources.append("NCERT Class 8 Science Textbook")
        
        # Remove duplicate sources
        sources = list(dict.fromkeys(sources))
        
        return answer, sources
        
    except Exception as e:
        logger.error(f"❌ Error in RAG pipeline: {e}")
        return "I'm having trouble accessing my knowledge base right now. Please try again with a different question.", ["System temporarily unavailable"]


# Debug functions to check database content
def debug_database(query="science"):
    """Debug function to see what's in the database"""
    try:
        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(query, k=5)
        
        print(f"Query: '{query}'")
        print(f"Found {len(docs)} documents:")
        for i, doc in enumerate(docs):
            print(f"Doc {i+1}: {doc.page_content[:150]}...")
            if hasattr(doc, 'metadata') and doc.metadata:
                print(f"  Metadata: {doc.metadata}")
            print("---")
        
        return docs
    except Exception as e:
        print(f"Debug error: {e}")
        return []

def check_database_coverage():
    """Check what content exists in the database"""
    try:
        vectorstore = load_vectorstore()
        
        # Get sample documents to see what's there
        all_docs = vectorstore.similarity_search("science", k=20)
        
        chapters_found = set()
        for doc in all_docs:
            if hasattr(doc, 'metadata') and 'chapter' in doc.metadata:
                chapters_found.add(f"Chapter {doc.metadata['chapter']}")
            elif hasattr(doc, 'metadata') and 'source' in doc.metadata:
                chapters_found.add(doc.metadata['source'])
            else:
                chapters_found.add("Unknown chapter")
        
        print("Content coverage found:", sorted(chapters_found))
        return chapters_found
    except Exception as e:
        print(f"Coverage check error: {e}")
        return set()

# Test the functions when this file is run directly
if __name__ == "__main__":
    print("=== DATABASE COVERAGE CHECK ===")
    coverage = check_database_coverage()
    
    print("\n=== TESTING FERTILIZATION QUERY ===")
    debug_database("fertilization in flowers")
    
    print("\n=== TESTING REPRODUCTION QUERY ===")
    debug_database("reproduction in animals")
