from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import embeddings

pdf_path = "./alice-in-wonderland.pdf"
loader = PyPDFLoader(pdf_path)
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200 
)
chunks = [] 
for page in document:
    pageChunks = text_splitter.split_text(page.page_content)
    print("\n\n\n")
    print("#######################################################################")
    for chunk in pageChunks:
        print(chunk)
        print("//////////////////////////////////////////////////////////////////")
        embed = embeddings(model='nomic-embed-text', prompt=chunk)
        print(embed)
