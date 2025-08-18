# rag_pipeline.py

import os
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# =========================
# CONFIG
# =========================
FAISS_DIR = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# HuggingFace model for answering
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.getenv("HF_TOKEN")  # <-- set your token in environment


# =========================
# LOAD FAISS INDEX
# =========================
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        FAISS_DIR, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore


# =========================
# BUILD PROMPT
# =========================
def build_prompt(query, docs):
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""
You are a helpful science tutor for class 8 students. 
Answer the question using the provided textbook context. 
Keep the answer simple, accurate, and engaging. 
If you don't know, say you don't know.

Question: {query}

Context:
{context}

Answer:
"""
    return prompt


# =========================
# RUN QA
# =========================
def answer_query(query, top_k=3):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)

    prompt = build_prompt(query, docs)

    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
    response = client.text_generation(prompt, max_new_tokens=300)

    return response


# =========================
# MAIN (interactive mode)
# =========================
if __name__ == "__main__":
    if not HF_TOKEN:
        raise ValueError(
            "Missing HuggingFace token! Please set it with:\nexport HF_TOKEN=your_api_key"
        )

    print("Class 8 Science Tutor (type 'exit' to quit)\n")

    while True:
        query = input("Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            break
        print("\nAnswer:", answer_query(query), "\n")
