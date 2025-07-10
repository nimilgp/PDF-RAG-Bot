from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import embed
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

pdf_path = "./alice-in-wonderland.pdf"
loader = PyPDFLoader(pdf_path)
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200 
)
chunks = [] 

ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(embedding_function=ollama_embeddings)

for page in document:
    pageChunks = text_splitter.split_text(page.page_content)
    vector_store.add_texts(pageChunks)

q = "does alice have a sister?"
res = vector_store.similarity_search(q)

for chunk in res:
    print(chunk)