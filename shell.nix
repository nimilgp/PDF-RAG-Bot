let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
      ollama
      chromadb
      sentence-transformers
      streamlit
      pypdf
      langchain-community
      langchain-ollama
      langchain-chroma
    ]))
  ];
}
