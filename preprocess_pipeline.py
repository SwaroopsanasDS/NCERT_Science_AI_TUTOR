import os, re
from tqdm import tqdm
from PyPDF2 import PdfMerger, PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

RAW_DIR = "data/NCERT_Class8_Science"
MERGED_PDF = "data/Class8_Science_Curiosity_Merged.pdf"
CLEANED_TXT = "data/Class8_Science_Curiosity_Cleaned.txt"
FAISS_DIR = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def merge_pdfs():
    merger = PdfMerger()
    pdf_files = sorted([f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pdf") and "prelims" not in f.lower() and "qr" not in f.lower()])
    for pdf in tqdm(pdf_files): merger.append(os.path.join(RAW_DIR, pdf))
    merger.write(MERGED_PDF)
    merger.close()
    print(f"Merged PDF: {MERGED_PDF}")

def clean_text(text: str) -> str:
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("Science - Class VIII", "")
    return text.strip()

def extract_and_clean_text():
    pdf_reader = PdfReader(MERGED_PDF)
    all_text = ""
    for page in tqdm(pdf_reader.pages):
        if page.extract_text(): all_text += page.extract_text() + "\n"
    with open(CLEANED_TXT, "w", encoding="utf-8") as f:
        f.write(clean_text(all_text))
    print(f"Cleaned text saved: {CLEANED_TXT}")

def build_faiss_index():
    with open(CLEANED_TXT, "r", encoding="utf-8") as f: text = f.read()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local(FAISS_DIR)
    print(f"FAISS index saved: {FAISS_DIR}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    merge_pdfs()
    extract_and_clean_text()
    build_faiss_index()
    print("[SUCCESS] Preprocessing complete ✅")
