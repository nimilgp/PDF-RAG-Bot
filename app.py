import streamlit as st

st.title(":blue[_Answer_] :red[PDF]initely 🤖")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload PDF Document for processing",type="pdf")
    process = st.button(
        "Process ⚡"
    )
