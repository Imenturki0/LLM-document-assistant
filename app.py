# app.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# Function to process PDF and return retriever + LLM
def load_pdf_and_create_rag(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma.from_documents(chunks, embeddings)

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    llm = Ollama(model="llama3")

    return retriever, llm


    st.title("📄 Chat with Your PDF (Local LLM)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save the uploaded PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load RAG system
    retriever, llm = load_pdf_and_create_rag("temp.pdf")

    # Input box for question
    question = st.text_input("Ask a question about the document")

    if question:
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {question}
        """

        response = llm.invoke(prompt)

        st.write("### Answer")
        st.write(response)