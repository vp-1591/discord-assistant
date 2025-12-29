import os
import json
import re
from typing import List
from datetime import datetime
from collections import defaultdict
import Stemmer

from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage, PromptTemplate
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine

# --- 0. CONFIG ---
PERSIST_DIR = "./llama_index_storage"
FORCE_REBUILD = False

# --- 1. SETTINGS CONFIGURATION ---
Settings.embed_model = OllamaEmbedding(
    model_name="bge-m3", 
    base_url="http://localhost:11434",
    request_timeout=300.0,
    keep_alive=0,
)

Settings.llm = Ollama(
    model="mistral:latest", 
    request_timeout=120.0, 
    keep_alive=0,
    additional_kwargs={"options": {"num_ctx": 4096}} 
)
Settings.embed_batch_size = 50 

# --- 2. CUSTOM DATA INGESTION ---
def load_nodes_from_json(directory: str) -> List[TextNode]:
    if not os.path.exists(directory):
        return []

    processed_data = []
    link_pattern = re.compile(r'^https?://[^\s]+$')
    code_pattern = re.compile(r'^[A-Za-z]{3}-[A-Za-z]{4}$')

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list): continue
                    last_user, last_msg = None, None
                    for msg in data:
                        text = str(msg.get("message", "")).strip()
                        user = str(msg.get("user", "Unknown"))
                        if link_pattern.search(text) or code_pattern.search(text): continue
                        if user == last_user and text == last_msg: continue
                        last_user, last_msg = user, text
                        ts = msg.get("timestamp", "2000-01-01T00:00:00")
                        date_str = ts.split("T")[0]
                        channel = msg.get("channel", "unknown")
                        processed_data.append({"user": user, "text": text, "date": date_str, "channel": channel})
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    groups = defaultdict(list)
    for entry in processed_data:
        groups[(entry['channel'], entry['date'])].append(entry)

    nodes = []
    CHUNK_LIMIT = 600
    OVERLAP = 60

    def split_long_message(author: str, text: str, metadata: dict):
        remaining_text = text
        is_first = True
        while remaining_text:
            prefix = f"{author}: " if is_first else f"{author}: [CUT] "
            max_chunk_len = CHUNK_LIMIT - len(prefix) - 10 
            if len(remaining_text) <= max_chunk_len:
                nodes.append(TextNode(text=prefix + remaining_text, metadata=metadata))
                break
            split_idx = remaining_text.rfind(" ", 0, max_chunk_len)
            if split_idx <= 0: split_idx = max_chunk_len
            nodes.append(TextNode(text=prefix + remaining_text[:split_idx] + " [CUT]", metadata=metadata))
            overlap_anchor = split_idx - OVERLAP
            new_start_idx = split_idx if overlap_anchor <= 0 else remaining_text.rfind(" ", 0, overlap_anchor)
            if new_start_idx == -1 or new_start_idx >= split_idx: new_start_idx = split_idx
            remaining_text = remaining_text[new_start_idx:].strip()
            is_first = False

    for (channel, date), messages in groups.items():
        metadata = {"date": date, "channel": channel}
        current_accumulated = ""
        for m in messages:
            line = f"{m['user']}: {m['text']}\n"
            if len(line) > CHUNK_LIMIT:
                if current_accumulated:
                    nodes.append(TextNode(text=current_accumulated.strip(), metadata=metadata))
                    current_accumulated = ""
                split_long_message(m['user'], m['text'], metadata)
                continue
            if len(current_accumulated) + len(line) > CHUNK_LIMIT:
                if current_accumulated:
                    nodes.append(TextNode(text=current_accumulated.strip(), metadata=metadata))
                current_accumulated = line
            else:
                current_accumulated += line
        if current_accumulated:
            nodes.append(TextNode(text=current_accumulated.strip(), metadata=metadata))
    return nodes

class RAGAssistant:

    def __init__(self):
        self.index = self._load_index()
        self.fusion_retriever, self.reranker, self.query_engine = self._setup_query_engine()
        self.persona_prompt = (
            "Ты — мудрый Глава Архива. "
            "Ты отвечаешь на вопросы, основываясь на предоставленной истории чата (контексте). "
            "Твой стиль — спокойный, аналитический и слегка формальный, но дружелюбный. "
            "Всегда отвечай на русском языке."
        )

    def _load_index(self):
        if not os.path.exists(PERSIST_DIR) or FORCE_REBUILD:
            nodes = load_nodes_from_json("./messages_json")
            if not nodes:
                nodes = [TextNode(text="Добряк тут!", metadata={"date": "2023-12-23"})]
            index = VectorStoreIndex(nodes, show_progress=True)
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
        return index

    def _setup_query_engine(self):
        nodes = list(self.index.docstore.docs.values())
        vector_retriever = self.index.as_retriever(similarity_top_k=50)
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, 
            similarity_top_k=50,
            stemmer=Stemmer.Stemmer("russian"),
            language="russian"
        )
        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=20,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )
        reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=7)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        qa_prompt_tmpl = PromptTemplate(
            f"Сегодняшняя дата: {current_date}\n"
            "Ниже представлены фрагменты истории чата из Discord.\n"
            "---------------------\n{context_str}\n---------------------\n"
            "Используя предоставленную историю, ответь на вопрос: {query_str}\n"
            "ОБЯЗАТЕЛЬНО: Если вопрос касается даты, сравнивай даты сообщений.\n"
            "Если информации нет в контексте, так и скажи.\nОтвет:"
        )

        for node in nodes:
            node.excluded_llm_metadata_keys = []
            node.metadata_template = "{key}: {value}"
            node.text_template = "Metadata: {metadata_str}\nContent: {content}"

        query_engine = RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            node_postprocessors=[reranker],
            response_mode="compact",
        )
        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
        return fusion_retriever, reranker, query_engine

    def _log_results(self, query: str, initial_nodes: List, reranked_nodes: List, agent_response: str):
        log_path = "cache/logs.txt"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
            f.write(f"QUERY: {query}\n")
            f.write(f"{'-'*80}\n")
            
            f.write("1. INITIAL RAG RESULTS (Top 20 from Fusion):\n")
            for i, n in enumerate(initial_nodes):
                f.write(f"  [{i+1}] Score: {n.score:.4f} | {n.node.get_content()[:100]}...\n")
            
            f.write(f"{'-'*40}\n")
            f.write("2. RERANKER RESULTS (Top 7):\n")
            for i, n in enumerate(reranked_nodes):
                f.write(f"  [{i+1}] Score: {n.score:.4f} | {n.node.get_content()[:100]}...\n")
                
            f.write(f"{'-'*40}\n")
            f.write(f"3. 1ST AGENT RESPONSE (RAG Synthesis):\n{agent_response}\n")
            f.write(f"{'='*80}\n\n")

    async def aquery(self, query_text: str):
        print(f"🔍 Processing RAG Pipeline for: {query_text}")
        
        # Initial RAG - Use async version
        initial_nodes = await self.fusion_retriever.aretrieve(query_text)
        
        # Reranking - Most rerankers are synchronous, wrap in to_thread to keep event loop alive
        import asyncio
        reranked_nodes = await asyncio.to_thread(
            self.reranker.postprocess_nodes, initial_nodes, query_str=query_text
        )
        
        # 1st Agent Response (Synthesis) - Use async version
        from llama_index.core.schema import QueryBundle
        response = await self.query_engine.asynthesize(
            query_bundle=QueryBundle(query_text),
            nodes=reranked_nodes
        )
        
        # Log everything (can be sync, but keep it minimal)
        self._log_results(query_text, initial_nodes, reranked_nodes, str(response))
        
        return response


    async def generate_refined_response(self, query_text: str, rag_response: str, history: List[str]):
        history_str = "\n".join(history)
        refine_prompt = (
            f"{self.persona_prompt}\n\n"
            f"Контекст из базы знаний (результат RAG):\n{rag_response}\n\n"
            f"Последние сообщения в чате для контекста разговора:\n{history_str}\n\n"
            f"Вопрос пользователя: {query_text}\n\n"
            f"Сформулируй окончательный ответ на русском языке, учитывая историю общения и информацию из базы знаний. "
            "Будь кратким, но в стиле своего персонажа."
        )
        response = await Settings.llm.acomplete(refine_prompt)
        return str(response)

def main():
    import asyncio
    assistant = RAGAssistant()
    query = "Что произошло в Барановичах?"
    response = asyncio.run(assistant.aquery(query))
    print(f"\nQUERY: {query}\nANSWER: {response}")

if __name__ == "__main__":
    main()