# ollamaRAG232

A Retrieval-Augmented Generation (RAG) chatbot built with Python, ChromaDB, PDFs, and Ollama.  
This project allows users to ask questions about uploaded dental documents and receive accurate, context-based answers using local AI models.

## Overview

The goal of this project is to improve question answering accuracy by grounding responses in provided documents instead of relying only on general AI knowledge.

This implementation focuses on dental procedure and billing codes, allowing users to query information directly from uploaded files and receive responses based on the actual document content.

## Features

- PDF document processing
- Text chunking for semantic search
- Vector embeddings using ChromaDB
- Local LLM inference with Ollama
- Context-aware question answering
- Retrieval-Augmented Generation (RAG) pipeline
- Dental code and document analysis

## Technologies Used

- Python
- ChromaDB
- Ollama
- LangChain
- PDF processing libraries
- Local embedding models

## How It Works

1. PDF files are loaded and processed
2. Text is split into smaller chunks
3. Chunks are embedded into vector space
4. ChromaDB stores the embeddings
5. User questions are matched with relevant chunks
6. Ollama generates an answer using retrieved context

This improves factual accuracy and reduces hallucinations.

## Installation

Clone the repository:

```bash
git clone https://github.com/yashikareddy-cyber/ollamaRAG232.git
```

Move into the project folder:

```bash
cd ollamaRAG232
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Start Ollama locally, then run:

```bash
python main.py
```

## Example Use Cases

- Looking up dental procedure codes
- Answering questions from insurance documents
- Searching large PDFs quickly
- AI-assisted document analysis

## Future Improvements

- Add a web interface
- Support multiple PDFs at once
- Improve retrieval accuracy
- Add conversation memory
- Deploy as a full-stack web app

## Author

Created by Yashika Aduma

GitHub Repository:  
:contentReference[oaicite:0]{index=0}
