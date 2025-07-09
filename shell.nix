let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
      ollama
      chromadb
      sentence-transformers
      streamlit
      pymupdf
      langchain-community
    ]))
  ];
}
