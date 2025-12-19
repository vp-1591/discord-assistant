import os
import json
import glob
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="mistral:latest")

vector_store = Chroma(
    collection_name="messages_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
print("Store initialized")