import streamlit as st
import rag_helper as rag
import tempfile
import os

st.title(":blue[_Answer_] :red[PDF]initely 🤖")

# Initialize the vector store in the session state
if 'vectorStore' not in st.session_state:
    st.session_state.vectorStore = rag.get_vector_store("trial-collection", "chromdb")

with st.sidebar:
    uploaded_files = st.file_uploader("Upload PDF Document for processing",
                                      type="pdf",
                                      accept_multiple_files=True)
    process = st.button("Process ⚡")

    if uploaded_files and process:
        for file in uploaded_files:
            with st.spinner(f"Processing {file.name}..."):
                # Store uploaded file as a temp file
                temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
                temp_file.write(file.read())
                rag.add_pdf_to_vector_store(temp_file.name, st.session_state.vectorStore, file.name)
                os.unlink(temp_file.name)  # Delete temp file
        st.success("All files processed successfully!")

# Chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about the content of your PDFs!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(rag.querry_the_llm(prompt, st.session_state.vectorStore))
    st.session_state.messages.append({"role": "assistant", "content": response})