import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from PyPDF2 import PdfReader
import pdfplumber



from transformers import pipeline
import re

# load pdf
def load_pdf_text(pdf_path,max_pages=10):
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        if i > max_pages:
            break
        text += page.extract_text() or ""
    return text



# set up ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_store")#creates searchable memory system

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")#understanding in numbers
collection = chroma_client.get_or_create_collection(name="rag-pdf")#creates a section in chromadb where all chunks will be

chunks = []
with pdfplumber.open("Delta Dental Dentist Handbook 2024.pdf") as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                if row and any(cell is not None for cell in row):
                    chunks.append(" | ".join([cell.strip() if cell else "" for cell in row]))

def chunk_by_paragraph(text):
    return text.split("\n\n")

def chunk_with_context(pdf_path):
    text= load_pdf_text(pdf_path)
    paragraphs= text.split("\n\n")
    chunks=[]
    for i, paragraph in enumerate(paragraphs):
        if i>0:
            previous_paragraph= paragraphs[i-1]
            context= previous_paragraph[:50]
            chunk= f"{context}\n\n{paragraph}"
        else:
            chunk=paragraphs
        chunks.append(chunk)
    return chunks
# add pdf
def add_pdf_to_chromadb(pdf_path):
    print("loading text")
    text = load_pdf_text(pdf_path)
    print("splitting text")
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    chunks = chunks[:20]
    ids = ids[:20]
    print("Adding to chromeDB")

    print("embedding")
    embeddings = embedding_function.embed_documents(chunks)

    print("add to chromaDB")
    collection.add(documents=chunks, ids=ids, embeddings=embeddings)

    print("added")

# ollama query
def rag_query(query_text):
    results = collection.query(query_texts=[query_text], n_results=2) #searches for two chunks that are most related to question
    docs = results["documents"][0] #gives list of top matching documents for each query
    context = "\n".join(docs)
    print (context)
    prompt = f"""
You are a helpful assistant trained to interpret CDT (Current Dental Terminology) codes and Delta Dental benefit policies using the 2024 Delta Dental Dentist Handbook. 

Use only the provided context below to answer the user's question. If the context doesn't contain the answer or specific CDT code, explain that clearly and advise contacting Delta Dental.

Context:
{context}

Question: {query_text}

Answer:"""
    llm = OllamaLLM(model="phi3:mini")
    return llm.invoke(prompt)


pdf_file = r"C:\Users\yashi_dxyfmod\Downloads\Delta Dental Dentist Handbook 2024.pdf"
chunk_with_context(pdf_file)

question = "What is the ADA CDT descriptor for the D2941 code"
answer = rag_query(question)
print("Answer:\n", answer)
