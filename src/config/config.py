import os
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# --- FILE PATHS ---
PERSIST_DIR = "./llama_index_storage"
SUMMARIES_PATH = "cache/summaries.json"
HISTORY_PATH = "cache/history.json"
OPINIONS_PATH = "cache/opinions.json"
MESSAGES_DIR = "./messages_json"
RAG_CACHE_PATH = "cache/rag_cache.json"
INDEXING_LOG_PATH = "logs/indexing_summarization.log"

# --- LLM PARAMS ---
LLM_CONTEXT_WINDOW = 8192
MAX_CHUNK_SIZE = int(LLM_CONTEXT_WINDOW * 0.75)  # 6144 tokens for summarization chunks

# --- CONFIG ---
ADMIN_IDS = ["470892009440149506"]  # Add authorized Discord user IDs here

# --- LLAMA INDEX SETTINGS ---
def configure_settings():
    Settings.embed_model = OllamaEmbedding(
        model_name="bge-m3", 
        base_url="http://localhost:11434",
        request_timeout=600.0,
        keep_alive=0,
    )

    Settings.llm = Ollama(
        model="qwen3:8b", 
        request_timeout=300.0, 
        keep_alive=0,
        context_window=LLM_CONTEXT_WINDOW,
        additional_kwargs={"stop": ["Observation:", "Observation\n"]},
    )
    Settings.embed_batch_size = 10
