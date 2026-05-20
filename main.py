from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import os
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter


model='mistral'

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

def process_pdf_to_chromadb(pdf_path, base_id="pdf_chunk_"):
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)

    ids = [f"{base_id}{i}" for i in range(len(chunks))]
    add_documents_to_collection(chunks, ids)
    print(f"✅ Added {len(chunks)} chunks from {pdf_path} to ChromaDB")


process_pdf_to_chromadb(r"C:\Users\yashi_dxyfmod\Downloads\cdt-summary.pdf")