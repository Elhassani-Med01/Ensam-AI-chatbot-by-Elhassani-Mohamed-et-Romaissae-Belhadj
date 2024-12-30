import os
import streamlit as st
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load environment variables from a .env file
load_dotenv()

# Initialize the LLM and embedding model
llm = Ollama(model="mistral", base_url=os.getenv("BASE_URL"))
embed_model = OllamaEmbeddings(model="mistral", base_url=os.getenv("BASE_URL"))

# Streamlit Title
st.title("Système de Question-Réponse avec LangChain & Ollama")

# Sidebar for file upload
st.sidebar.header("Charger un document")
uploaded_file = st.sidebar.file_uploader("Chargez un fichier texte (format .txt)", type=["txt"])

# Process file and setup RAG pipeline
if uploaded_file:
    # Load the text
    text = uploaded_file.read().decode("utf-8")
    st.sidebar.success("Fichier chargé avec succès!")

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    chunks = text_splitter.split_text(text)

    # Create vector store from the text chunks and embeddings
    vector_store = Chroma.from_texts(chunks, embed_model)

    # Create a retriever from the vector store
    retriever = vector_store.as_retriever()

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # User input for questions
    st.header("Posez une Question")
    question = st.text_input("Entrez votre question ici")

    if question:
        # Get the answer from the pipeline
        response = qa_chain.run(question)
        st.success("Réponse:")
        st.write(response)

# Footer
st.sidebar.info("Développé avec LangChain, Ollama et Streamlit")