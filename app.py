import streamlit as st
import rag_helper as rag
import tempfile
import os

st.title(":blue[_Answer_] :red[PDF]initely 🤖")

vectorStore = rag.get_vector_store("trial-collection","chromdb")

with st.sidebar:
    uploaded_files = st.file_uploader("Upload PDF Document for processing",
                                      type="pdf",
                                      accept_multiple_files=True)
    process = st.button(
        "Process ⚡"
    )

    if uploaded_files and process:
        for file in uploaded_files:
            # Store uploaded file as a temp file
            temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
            temp_file.write(file.read())
            print(file.name)
            rag.add_pdf_to_vector_store(temp_file.name,vectorStore,file.name)
            os.unlink(temp_file.name)  # Delete temp file

            q = "what is CORS?"
            rag.querry_the_llm(q, vectorStore)
            