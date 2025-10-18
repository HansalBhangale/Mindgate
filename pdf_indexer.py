import os
# We import config to ensure environment variables are loaded
import config 
from langchain_community.document_loaders import PyPDFLoader
# --- CORRECTED IMPORT ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_vector_store():
    """
    Creates and saves a FAISS vector store from the PDF document
    using OpenAI embeddings. This only needs to be run once.
    """
    if not os.path.exists(config.PDF_PATH):
        print(f"Error: PDF file not found at {config.PDF_PATH}")
        return

    print("Loading document...")
    loader = PyPDFLoader(config.PDF_PATH)
    documents = loader.load()

    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    print("Creating embeddings with OpenAI. This may take a moment...")
    try:
        embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
    except Exception as e:
        print(f"Error initializing OpenAI Embeddings. Ensure your API key is valid. Error: {e}")
        return

    print(f"Creating and saving FAISS vector store at: {config.VECTOR_STORE_PATH_OPENAI}")
    try:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(config.VECTOR_STORE_PATH_OPENAI)
        print("\nVector store created successfully!")
        print(f"You can now run the main application with 'python app.py'")
    except Exception as e:
        print(f"An error occurred while creating the vector store: {e}")

if __name__ == "__main__":
    create_vector_store()