import os
import re
import pdfplumber
import chromadb

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# === CONFIGURATION ===
PDF_FILE = r"C:\Users\yashi_dxyfmod\Downloads\Delta Dental Dentist Handbook 2024.pdf"
CHROMA_DB_PATH = "chroma_store"
COLLECTION_NAME = "rag-pdf"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_LIMIT = 100  # to avoid memory issues

# === INIT CHROMADB AND EMBEDDINGS ===
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
embedding_function = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# If collection already exists, drop and recreate to match dimensions
if COLLECTION_NAME in [col.name for col in chroma_client.list_collections()]:
    chroma_client.delete_collection(name=COLLECTION_NAME)

collection = chroma_client.create_collection(name=COLLECTION_NAME)

# === EXTRACT TABLE ROWS ===
def extract_table_rows_from_pdf(pdf_path, max_pages=70):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                break
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if row and any(cell is not None for cell in row):
                        row_text = " | ".join([cell.strip() if cell else "" for cell in row])
                        chunks.append(row_text)
    return chunks

# === EXTRACT CDT CODE FROM CHUNK TEXT ===
def extract_cdt_code(text):
    # Check each "cell" in the chunk (split by "|")
    for part in text.split("|"):
        part = part.strip()
        if re.match(r"^D\d{4}$", part):  # Exact CDT code pattern
            return part
    return None

# === ADD CHUNKS TO CHROMADB ===
def add_pdf_to_chromadb(pdf_path):
    print("Extracting row chunks from PDF...")
    chunks = extract_table_rows_from_pdf(pdf_path)
    print("\n🔍 Searching for D2941 in extracted raw table rows...")
    for i, chunk in enumerate(chunks):
        if "D2941" in chunk:
            print(f"✅ D2941 FOUND in chunk {i}:\n{chunk}")
    print(f"Extracted {len(chunks)} chunks.")

    chunks = chunks[:CHUNK_LIMIT]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    metadatas = []
    for chunk in chunks:
        cdt_code = extract_cdt_code(chunk)
        metadatas.append({"cdt_code": cdt_code if cdt_code else "unknown"})

    for i, md in enumerate(metadatas):
        if md["cdt_code"] == "D2941":
            print(f"✅ Found D2941 in chunk {i}:")
            print(chunks[i])

    print("Embedding chunks...")
    embeddings = embedding_function.embed_documents(chunks)

    count_d2941 = sum(1 for md in metadatas if md.get("cdt_code") == "D2941")
    print(f"✅ Chunks labeled with CDT code 'D2941': {count_d2941}")

    print("Adding to ChromaDB...")
    collection.add(documents=chunks, ids=ids, embeddings=embeddings, metadatas=metadatas)
    print("✓ Added to ChromaDB.")

# === RAG QUERY WITH OPTIONAL CDT FILTER ===
def rag_query(query_text, cdt_code=None):
    print("Embedding query...")
    query_embedding = embedding_function.embed_query(query_text)

    print("Retrieving similar documents...")
    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": 5,
    }

    if cdt_code:
        query_args["where"] = {"cdt_code": cdt_code}

    results = collection.query(**query_args)

    docs = results["documents"][0]
    if not docs:
        return " No relevant content found for that CDT code in the handbook."

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

# === RUN PIPELINE ===
if __name__ == "__main__":
    add_pdf_to_chromadb(PDF_FILE)

    # Ask your question
    question = "What is the ADA CDT descriptor for the D2941 code?"
    answer = rag_query(question, cdt_code="D2941")
    print("\nAnswer:\n", answer)


with pdfplumber.open(PDF_FILE) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text and "D2941" in text:
            print(f"\n✅ Found D2941 on page {i+1}")
            print(text)