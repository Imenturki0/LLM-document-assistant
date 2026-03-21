# LLM-Powered Document Assistant (RAG System)

This project is a local AI-powered document assistant that allows users to upload PDF files and ask questions about them using a Large Language Model (LLM).

## 🚀 Features

- Chat with your PDF documents
- Uses local LLM (Llama3 via Ollama)
- Retrieval-Augmented Generation (RAG)
- Semantic search with embeddings
- Streamlit web interface

## 🧠 Tech Stack

- Python
- LangChain
- Ollama (Llama3)
- Sentence Transformers
- ChromaDB
- Streamlit

## 📂 Project Structure
├── app.py
├── notebooks/
├── data/
├── vector_db/
├── requirements.txt

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/llm-document-assistant.git
cd llm-document-assistant

python -m venv env
source env/bin/activate  # or Windows: env\Scripts\activate

pip install -r requirements.txt