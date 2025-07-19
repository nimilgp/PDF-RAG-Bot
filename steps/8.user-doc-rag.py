import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ollama import chat

# Config
knowledge_collection = input("Please enter a name for your document collection: ")
### later change to load from .env
PERSIST_DIRECTORY = "./user_chroma_db"
OLLAMA_EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text")

# Vector Store Init
# Create directory if it doesn't exist
if not os.path.exists(PERSIST_DIRECTORY):
    print(f"Creating new vector store directory: {PERSIST_DIRECTORY}")
    os.makedirs(PERSIST_DIRECTORY)

# Load or initialize the vector store
print(f"Loading Initializing vector store for collection: '{knowledge_collection}'...")
vector_store = Chroma(
    collection_name=knowledge_collection,
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=OLLAMA_EMBEDDINGS
)
print("Vector store ready.")

# --- Document Processing ---
pdf_directory_path = input("Please enter the path to the directory containing your PDF documents: ")

if not os.path.isdir(pdf_directory_path):
    print(f"Error: Directory not found at '{pdf_directory_path}'")
else:
    pdf_files = [f for f in os.listdir(pdf_directory_path) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in '{pdf_directory_path}'.")
    else:
        print(f"Found {len(pdf_files)} PDF(s). Processing them now...")
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory_path, pdf_file)

            # Check if the document has already been processed
            print(f"Checking if '{pdf_path}' is already in the vector store...")
            # PyPDFLoader adds the file path to the 'source' metadata field.
            existing_docs = vector_store.get(where={"source": pdf_path})

            if existing_docs and existing_docs['ids']:
                print(f"Document '{pdf_path}' has already been vectorized and stored.")
            else:
                print(f"Document '{pdf_path}' not found in the store. Processing and adding it...")
                
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

                    # Add the new document chunks to the vector store
                    vector_store.add_documents(docs)
                    print(f"Successfully added '{pdf_path}' to the vector store.")
                except Exception as e:
                    print(f"Failed to process {pdf_path}. Error: {e}")

# Querying
while True:
    q = input("\nEnter your query (or type 'exit' to quit): ")
    if q.lower() == 'exit':
        break

    if not q.strip():
        continue

    print(f"\nQuerying for: '{q}'")
    context = vector_store.similarity_search(query=q, k=10)

    if not context:
        print("No relevant information found in the document(s).")
        continue

    system_message_content = "You are a helpful AI assistant. Use the following information to answer the user's question. If the answer is not present in the provided information, state that you don't have enough information."
    system_message_content += "\n\nRelevant Information:\n"
    for i, chunk in enumerate(context):
        system_message_content += f"Chunk {i+1}: {chunk.page_content}\n"

    try:
        stream = chat(
            model='llama3.1:8b',
            messages=[
                {'role': 'system', 'content': system_message_content},
                {'role': 'user', 'content': q}
            ],
            stream=True,
        )

        print("\nResponse:")
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        print()
    except Exception as e:
        print(f"An error occurred while querying the model: {e}")