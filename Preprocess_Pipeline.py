# preprocess_pipeline.py

import os
import re
from tqdm import tqdm
import PyPDF2
from PyPDF2 import PdfMerger, PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================
# CONFIG
# =========================
RAW_DIR = "data/NCERT_Class8_Science"
MERGED_PDF = "data/Class8_Science_Curiosity_Merged.pdf"
CLEANED_TXT = "data/Class8_Science_Curiosity_Cleaned.txt"
FAISS_DIR = "data/faiss_index"

# Free HuggingFace embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =========================
# STEP 1: MERGE PDFs
# =========================
def merge_pdfs(input_dir=RAW_DIR, output_file=MERGED_PDF):
    merger = PdfMerger()

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    skip_keywords = ["prelims", "qr"]  # skip extra PDFs
    chapter_pdfs = [f for f in pdf_files if not any(key in f.lower() for key in skip_keywords)]
    chapter_pdfs.sort()

    print(f"Merging {len(chapter_pdfs)} PDFs...")

    for pdf in tqdm(chapter_pdfs):
        merger.append(os.path.join(input_dir, pdf))

    merger.write(output_file)
    merger.close()
    print(f"[INFO] Merged PDF saved at: {output_file}")


# =========================
# STEP 2: EXTRACT & CLEAN TEXT
# =========================
def clean_text(text: str) -> str:
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # remove page numbers
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    text = text.replace("Science - Class VIII", "")  # remove headers
    return text.strip()


def extract_and_clean_text(pdf_file=MERGED_PDF, output_file=CLEANED_TXT):
    pdf_reader = PdfReader(pdf_file)
    all_text = ""
    for page in tqdm(pdf_reader.pages, desc="Extracting text"):
        if page.extract_text():
            all_text += page.extract_text() + "\n"

    cleaned = clean_text(all_text)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned)
    print(f"[INFO] Cleaned text saved at: {output_file}")


# =========================
# STEP 3: BUILD FAISS INDEX
# =========================
def build_faiss_index(txt_file=CLEANED_TXT, faiss_dir=FAISS_DIR):
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    print(f"[INFO] Total chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    vectorstore.save_local(faiss_dir)
    print(f"[INFO] FAISS index saved at: {faiss_dir}")


# =========================
# MAIN PIPELINE
# =========================
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    merge_pdfs()
    extract_and_clean_text()
    build_faiss_index()
    print("[SUCCESS] Preprocessing complete ✅")
