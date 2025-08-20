# rag_pipeline.py
import os
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# =========================
# Load environment variables
# =========================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Login to Hugging Face Hub if token is available
if HF_TOKEN:
    try:
        login(HF_TOKEN)
        print("✅ Hugging Face login successful")
    except Exception as e:
        print(f"⚠️ Hugging Face login failed: {e}")
else:
    print("⚠️ No HF_TOKEN found in .env file. Some models may fail to load.")

# =========================
# Load FAISS Vector Store
# =========================
def load_vectorstore(persist_directory="data/faiss_index"):
    """Load FAISS index with HuggingFace embeddings."""
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Can swap for lighter one if needed
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# =========================
# Setup RetrievalQA
# =========================
def rag_qa(query: str):
    """Retrieve and answer a query using FAISS + HuggingFaceHub LLM."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = """
    You are an NCERT Science AI Tutor. Use the provided context to answer questions clearly and accurately.

    Context: {context}

    Question: {question}

    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # free & small LLM
        model_kwargs={"temperature": 0.2, "max_length": 512}
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa({"query": query})
    return result["result"], result.get("source_documents", [])
