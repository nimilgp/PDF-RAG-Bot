import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ollama import chat

pdf_path = "./alice-in-wonderland.pdf"
persist_directory = "./chroma"
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Check if the vector store already exists and is not empty
if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
    print("Persistent vector store not found or is empty. Creating new one...")
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        
    # Load the document
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400
    )
    docs = text_splitter.split_documents(document)
    # Create the vector store and persist it
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=ollama_embeddings,
        persist_directory=persist_directory
    )
    print("Vector store created and persisted.")
else:
    print("Loading persistent vector store...")
    # Load the existing vector store
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=ollama_embeddings
    )
    print("Vector store loaded.")

# Now query the vector store
system_message_content = "You are a helpful AI assistant. Use the following information to answer the user's question. If the answer is not present in the provided information, state that you don't have enough information."

q = "Does alice have a sister?"
print(f"\nQuerying for: '{q}'")
context = vector_store.similarity_search(query=q, k=10)

if context:
   system_message_content += "\n\nRelevant Information:\n"
   for i,chunk in enumerate(context):
      system_message_content += f"Chunk {i+1}: {chunk.page_content}\n"

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