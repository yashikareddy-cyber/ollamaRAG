from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import os
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter

model='mistral'

chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))
collection= chroma_client.get_or_create_collection(
    name= "pdf_rag_demo",
    metadata= {"description": "PDF Knowledge Base"},
    embedding_function= OllamaEmbeddings(
        model= 'mistral',
        base_url= "http://localhost:11434"
    )
)


def extract_text_from_pdf(pdf_path):
    doc=fitz.open(pdf_path)
    full_text=""
    for page in doc:
        full_text+=page.extractText()
    return full_text

def split_text(text):
    splitter= RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def add_pdf_to_chromadb(pdf_path):
    text= extract_text_from_pdf(pdf_path)
    chunks= split_text(text)
    ids = [f"{os.path.basename(pdf_path)}_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    print(f"Add {len(chunks)} chunks from {pdf_path}")

def retrieve_context(query_text, n_results=3):
    result= collection.query(query_texts=[query_text], n_results=n_results)
    docs= result["documents"][0]
    return "\n". join(docs)


def rag_query(query_text):
    context= retrieve_context(query_text)
    prompt= f"use the context below to answer the question"




prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    llm = OllamaLLM(model="llama3")
    response = llm.invoke(prompt)
    return response




pdf_file = r"C:\Users\yashi_dxyfmod\Downloads\cdt-summary.pdf"
add_pdf_to_chromadb(pdf_file)

# Ask a question
question = "What is the CDT framework about?"
answer = rag_query(question)
print(answer)
