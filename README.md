# 📚 RAG-based PDF Question Answering System

This project is a simple **Retrieval-Augmented Generation (RAG)** system that allows you to ask questions based on the content of PDF files.

It:
- Loads PDFs from a local folder
- Splits them into chunks
- Stores embeddings in a FAISS vector database
- Retrieves relevant chunks for a query
- Uses an LLM (via Groq API) to generate answers

---

## 🚀 Features

- 📄 Load multiple PDFs automatically
- ✂️ Intelligent text chunking
- 🔍 Fast similarity search using FAISS
- 🤖 LLM-powered answers using retrieved context
- ⚡ Lightweight and beginner-friendly RAG implementation

---

## 🛠️ Tech Stack

- Python
- FAISS (vector search)
- LangChain (document loading & splitting)
- Groq API (LLM inference)
- NumPy

---

## 📂 Project Structure

project/
│── data/ # Place your PDF files here
│── main.py # Main RAG script
│── .env # API keys (not committed)
│── requirements.txt
│── README.md


---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/rag-pdf-qa.git
cd rag-pdf-qa