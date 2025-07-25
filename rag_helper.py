import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ollama import chat

# Vector Store Initialization
def get_vector_store(collection_name, persist_directory):
    # Use nomic-embed-text as the default embedding model
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create directory if it doesn't exist
    if not os.path.exists(persist_directory):
        print(f"Creating new vector store directory: {persist_directory}")
        os.makedirs(persist_directory)

    # Load or initialize the vector store
    print(f"Loading/Initializing vector store for collection: '{collection_name}'...")
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=ollama_embeddings
    )
    print("Vector store ready.")
    return vector_store

# Document Processing
def add_pdf_to_vector_store(pdf_path, vector_store, original_filename):
    #Checks if a PDF document is already in the vector store, if not, processes and adds it.
        
    existing_docs = vector_store.get(where={"source": original_filename})

    if existing_docs and existing_docs['ids']:
        print(f"Document '{original_filename}' has already been vectorized and stored.")
    else:
        print(f"Document '{original_filename}' not found in the store. Processing and adding it...")
        try:
            # Load the document
            loader = PyPDFLoader(pdf_path)
            document = loader.load()

            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=400
            )
            docs = text_splitter.split_documents(document)

            # Update metadata to use the desired source identifier
            for doc in docs:
                doc.metadata["source"] = original_filename

            # Add the new document chunks to the vector store
            vector_store.add_documents(docs)
            print(f"Successfully added '{original_filename}' to the vector store.")
        except Exception as e:
            print(f"Failed to process {pdf_path}. Error: {e}")

