import os
import json
import glob
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

vector_store = Chroma(
    collection_name="messages_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
print("Store initialized")