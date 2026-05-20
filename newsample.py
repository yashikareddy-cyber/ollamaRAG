import re
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter



# load pdf
def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# set up ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_store")

existing_collections = chroma_client.list_collections()
for c in existing_collections:
    if c.name == "rag-pdf":
        chroma_client.delete_collection(name="rag-pdf")

#create a new collection
collection = chroma_client.get_or_create_collection(name="rag-pdf")

# set up embedding model
from langchain_huggingface import HuggingFaceEmbeddings
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# add pdf
def add_pdf_to_chromadb(pdf_path):
    print("loading text")
    text = load_pdf_text(pdf_path)
    print("splitting text")
    chunk_size = 1000
    overlap = 200
    chunks = []

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:  # skip short parts
            chunks.append(chunk)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = []
    for chunk in chunks:
        match = re.search(r"D\d{4}", chunk)
        code = match.group(0) if match else "unknown"
        metadatas.append({"cdt_code": code})
    print("Adding to chromeDB")
    print("embedding")
    embeddings = embedding_function.embed_documents(chunks)

    print("add to chromaDB")
    collection.add(documents=chunks, ids=ids, embeddings=embeddings, metadatas=metadatas)

    print("added")

    

# ollama query
def rag_query(query_text):
    import re
    match = re.search(r"\bD\d{4}\b", query_text)
    cdt_code = match.group(0) if match else None

    if cdt_code:
        print(f" Searching for CDT code: {cdt_code}")
        results = collection.query(
            query_texts=[query_text],
            n_results=5,
            where={"cdt_code": cdt_code},#where the cdt code is actually there
            include=["documents", "metadatas"]#tells chromaDB to give back the text and info about the chunk
        )
    else:
        results = collection.query(
            query_texts=[query_text],
            n_results=5,
            include=["documents"]
        )

    docs = results["documents"][0]
    context = "\n".join(docs)

    prompt = f"""
    You are a helpful assistant trained to interpret CDT (Current Dental Terminology) codes and Delta Dental benefit policies using the 2024 Delta Dental Dentist Handbook.

    Use only the provided context below to answer the user's question. If the context doesn't contain the answer or specific CDT code, explain that clearly and advise contacting Delta Dental.

    Context:
    {context}

    Question: {query_text}

    Answer:"""

    llm = OllamaLLM(model="llama3")
    return llm.invoke(prompt)


pdf_file = r"C:\Users\yashi_dxyfmod\Downloads\Delta Dental Dentist Handbook 2024.pdf"
add_pdf_to_chromadb(pdf_file)

question = "When is a limited oral evaluation – problem focused (D0140) allowed with definitive treatment?"
answer = rag_query(question)
print("Answer:\n", answer)