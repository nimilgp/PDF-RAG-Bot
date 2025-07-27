# Answer PDFinitely
A bot that answers queries based on the knowledge set provided from uploaded pdfs.

# Usage
[Demo.webm](https://github.com/user-attachments/assets/9fb6cbfe-6ae9-4c65-8d97-e93169676a66)

# Running the application

## 1. Clone the Repository
```sh
git clone https://github.com/nimilgp/PDF-RAG-Bot.git
```

## 2. Navigate to the Directory
```sh
cd PDF-RAG-Bot
```

## 3. Install Ollama and the LLM models
1) llama3.1:8b
2) nomic-embed-text

## 4. Set Ollama to use ROCm for AMD GPU and CUDA for Nvidia GPU instead of CPU

## 5. Install required python packages
1) Use nix-shell
```sh
nix-shell
```

  **OR**

2) Use pip in a venv and install
```sh
pip install ollama chromadb sentence-transformers streamlit pypdf langchain-community langchain-ollama langchain-chroma
```

## 6. Run the app
```sh
streamlit run app.py
```
