from langchain_community.document_loaders import PyPDFLoader

pdf_path = "./alice-in-wonderland.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

for x in documents:
    print("\n\n\n")
    print("#######################################################################")
    print(x.page_content)
