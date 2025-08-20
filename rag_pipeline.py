import os
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# CONFIG
# =========================
FAISS_DIR = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.getenv("HF_TOKEN")  # Set this in .env file or Streamlit secrets


# =========================
# LOAD FAISS INDEX
# =========================
def load_vectorstore():
    """Load FAISS index with HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


# =========================
# BUILD PROMPT
# =========================
def build_prompt(query, docs):
    """Builds a structured prompt for the LLM using retrieved docs."""
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""
You are a helpful science tutor for class 8 students.
Answer the question using the provided NCERT textbook context.
Keep the explanation simple, accurate, and engaging.
If the answer is not in the context, say you don't know.

Question: {query}

Context:
{context}

Answer:
"""
    return prompt


# =========================
# RUN QA PIPELINE
# =========================
def rag_qa(query: str, top_k: int = 3):
    """Main RAG QA pipeline."""
    # Load vector DB
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)

    # Build prompt
    prompt = build_prompt(query, docs)

    # Query Hugging Face inference endpoint
    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
    response = client.text_generation(
        prompt,
        max_new_tokens=300,
        temperature=0.2,
        do_sample=True
    )

    # Extract sources
    sources = []
    for d in docs:
        if "source" in d.metadata:
            sources.append(d.metadata["source"])
        else:
            sources.append(d.page_content[:150] + "...")

    return response, sources
