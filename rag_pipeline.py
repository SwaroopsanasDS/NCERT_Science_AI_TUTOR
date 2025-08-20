# rag_pipeline.py
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# =========================
# CONFIG
# =========================
FAISS_DIR = "data/faiss_index"

# ⚠️ Hardcode your Hugging Face token here
HF_TOKEN = "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"

if not HF_TOKEN or HF_TOKEN.startswith("hf_xxxx"):
    raise ValueError("❌ Please set your Hugging Face API key in rag_pipeline.py!")

# =========================
# Load FAISS Vector Store
# =========================
def load_vectorstore(persist_directory=FAISS_DIR):
    embedding_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        cache_folder="model_cache",
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

# =========================
# RAG QA
# =========================
def rag_qa(query: str):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = """
You are an NCERT Class 8 Science AI Tutor.
Use the provided context to answer clearly, concisely, and in simple words.
If the answer is not in the context, say "I don’t know."

Context: {context}

Question: {question}

Answer:
"""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_kwargs={"temperature": 0.2, "max_new_tokens": 512},
        huggingfacehub_api_token=HF_TOKEN
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    result = qa({"query": query})
    return result["result"], result["source_documents"]
