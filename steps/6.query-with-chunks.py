from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import embed
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ollama import chat

pdf_path = "./alice-in-wonderland.pdf"

loader = PyPDFLoader(pdf_path)
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  
    chunk_overlap=400 
)
chunks = [] 

ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    embedding_function=ollama_embeddings,
    #persist_directory="./chroma_db"
)

for page in document:
    pageChunks = text_splitter.split_text(page.page_content)
    vector_store.add_texts(pageChunks)

system_message_content = "You are a helpful AI assistant. Use the following information to answer the user's question. If the answer is not present in the provided information, state that you don't have enough information."

q = "Tell me about alice's family"
context = vector_store.similarity_search(q)
if context:
   system_message_content += "\n\nRelevant Information:\n"
   for i,chunk in enumerate(context):
      system_message_content += f"Chunk {i+1}: {chunk}\n"

stream = chat(
    model='llama3.1:8b',
    messages=[
       {'role': 'system', 'content': system_message_content},
       {'role': 'user', 'content': q}
    ],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
