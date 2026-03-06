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
    request_timeout=600.0,
    keep_alive=0,  # free VRAM after query-time embedding; bumped to 30s during index build
)

Settings.llm = Ollama(
    model="mistral:latest", 
    request_timeout=300.0, 
    keep_alive=0,
    context_window=8192,
)
Settings.embed_batch_size = 10  # smaller batches = less chance of Ollama dropping the connection

# --- 2. CUSTOM DATA INGESTION ---
def resolve_all_mentions(text: str, live_map: dict, fallback_map: dict) -> str:
    """
    Priority:
    1. live_map (IDs currently in Discord)
    2. fallback_map (IDs saved during export)
    3. original match string (if nothing found)
    """
    pattern = re.compile(r'<@(!|&)?(\d+)>')
    
    def replace_match(match):
        entity_id = match.group(2)
        # 1. Try live names first
        if entity_id in live_map:
            return live_map[entity_id]
        # 2. Try names that were saved when message was exported
        return fallback_map.get(entity_id, match.group(0))

    return pattern.sub(replace_match, text)

def load_nodes_from_json(directory: str, id_map: dict) -> List[TextNode]:
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
                        raw_text = str(msg.get("message", "")).strip()
                        user_id = str(msg.get("user_id", "Unknown"))
                        fallback_names = msg.get("last_known_names", {})
                        
                        # Resolve mentions: Live Map -> Fallback Map -> Raw ID
                        text = resolve_all_mentions(raw_text, id_map, fallback_names)
                        
                        # Resolve the author name: Live Map -> Fallback Map -> "Unknown"
                        author_name = id_map.get(user_id, fallback_names.get(user_id, "Unknown"))
                        
                        if link_pattern.search(text) or code_pattern.search(text): continue
                        if author_name == last_user and text == last_msg: continue
                        last_user, last_msg = author_name, text
                        ts = msg.get("timestamp", "2000-01-01T00:00:00")
                        date_str = ts.split("T")[0]
                        channel = msg.get("channel", "unknown")
                        processed_data.append({"user": author_name, "text": text, "date": date_str, "channel": channel})
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

    def __init__(self, id_map: dict = None, name: str = "Глава Архива"):
        self.id_map = id_map or {}
        self.name = name  # Default name
        self.index = self._load_index()
        self.fusion_retriever, self.reranker, self.query_engine = self._setup_query_engine()

    def _get_persona_prompt(self, name: str = None) -> str:
        current_name = name or self.name
        return (
            f"Ты — мудрый {current_name}. "
            "Твоя задача — давать ответы, опираясь на пыльные летописи (knowledge_base) и беседу (conversation_summary, chat_memory). "
            "Твоя речь должна быть степенной, слегка архаичной и исполненной достоинства. "
            "Обращайся к вопрошающему как к 'искателю истин' или 'путник'. "
            "Никогда не используй слова 'контекст', 'база данных' или 'информация'. "
            "Вместо этого говори 'мои свитки', 'предания' или 'записи в манускриптах'. "
            "Отвечай всегда на русском языке, сохраняя атмосферу древнего хранилища мудрости."
        )

    def _load_index(self):
        if not os.path.exists(PERSIST_DIR) or FORCE_REBUILD:
            # We use the live id_map passed from main.py during rebuild
            nodes = load_nodes_from_json("./messages_json", self.id_map)
            if not nodes:
                nodes = [TextNode(text="Добряк тут!", metadata={"date": "2023-12-23"})]
            # Apply metadata templates BEFORE embedding so nodes are
            # embedded with the correct format from the start.
            # This prevents RetrieverQueryEngine from detecting "changed"
            # nodes and triggering a second embedding pass.
            for node in nodes:
                node.excluded_llm_metadata_keys = []
                node.metadata_template = "{key}: {value}"
                node.text_template = "Metadata: {metadata_str}\nContent: {content}"
            print(f"📄 Building index from {len(nodes)} nodes...")
            # Temporarily keep bge-m3 warm during bulk embedding to avoid cold-start per batch
            build_embed_model = OllamaEmbedding(
                model_name="bge-m3", 
                base_url="http://localhost:11434",
                request_timeout=600.0,
                keep_alive="30s",
            )
            try:
                # insert_batch_size=len(nodes) ensures all nodes are in one batch,
                # giving a single continuous progress bar instead of one bar per 2048 nodes
                index = VectorStoreIndex(nodes, show_progress=True, insert_batch_size=len(nodes), embed_model=build_embed_model)
                index.storage_context.persist(persist_dir=PERSIST_DIR)
            finally:
                # Revert index to use the global embed_model with keep_alive=0 for inference
                index._embed_model = Settings.embed_model
                print("🧹 Embedding model keep_alive reset to 0 (VRAM freed for Mistral)")
        else:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            nodes = list(index.docstore.docs.values())
        self._nodes = nodes
        return index

    def _setup_query_engine(self):
        nodes = self._nodes  # use nodes cached by _load_index, not docstore (which strips embeddings)
        print(f"⚙️ Setting up query engine with {len(nodes)} nodes...")
        vector_retriever = self.index.as_retriever(similarity_top_k=50)
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, 
            similarity_top_k=50,
            stemmer=Stemmer.Stemmer("russian"),
            language="russian"
        )
        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=50,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )
        reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=10)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.qa_prompt_tmpl = PromptTemplate(
            f"# ИНСТРУКЦИЯ ПО АНАЛИЗУ ЛОГОВ\n"
            f"**Ты — поисковая система Архива. Твоя цель — найти в логах информацию, которая поможет {self.name} ответить на вопросы.**\n"
            f"**Сегодняшняя дата:** {current_date}\n\n"
            
            "## КОНТЕКСТ (Фрагменты истории Discord)\n"
            "Ниже приведены записи из базы знаний. Каждый фрагмент содержит метаданные и содержание.\n"
            "--- \n"
            "{context_str}\n"
            "--- \n\n"
            
            "## ЗАДАНИЕ\n"
            "Используя только предоставленные выше записи, ответь на вопрос:\n"
            "> **{query_str}**\n\n"
            
            "### ТРЕБОВАНИЯ К ОТВЕТУ:\n"
            "1. **Точность субъектов:** Четко разделяй, кто совершил действие, а кто о нем рассказывает.\n"
            "2. **Хронология:** Сравнивай даты в метаданных, если это важно для ответа.\n"
            "3. **Отсутствие данных:** Если в записях нет прямого ответа, напиши: 'В летописях об этом не сказано'.\n"
            "4. **Стиль:** Фактологический, сухой, без домыслов.\n\n"
        )

        query_engine = RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            node_postprocessors=[reranker],
            response_mode="compact",
        )
        query_engine.update_prompts({"response_synthesizer:text_qa_template": self.qa_prompt_tmpl})
        print("✅ Query engine ready.")
        return fusion_retriever, reranker, query_engine

    def _write_to_rolling_log(self, content: str, log_path: str = "cache/logs.txt", max_entries: int = 20):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        separator = "=" * 80
        entries = []
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                raw = f.read()
                entries = [e.strip() for e in raw.split(separator) if e.strip()]
        
        # Add new content
        entries.append(content.strip())
        
        # Keep only latest
        entries = entries[-max_entries:]
        
        with open(log_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(f"\n{separator}\n{e}\n")

    def _log_interaction(self, query: str, agent1_prompt: str, agent1_response: str, agent2_prompt: str, agent2_response: str):
        log_block = (
            f"TIMESTAMP: {datetime.now().isoformat()}\n"
            f"QUERY: {query}\n"
            f"{'-'*80}\n"
            f"--- AGENT 1 (RAG) PROMPT ---\n{agent1_prompt}\n"
            f"{'-'*40}\n"
            f"--- AGENT 1 RESPONSE ---\n{agent1_response}\n"
            f"{'-'*80}\n"
            f"--- AGENT 2 (SYNTHESIS) PROMPT ---\n{agent2_prompt}\n"
            f"{'-'*40}\n"
            f"--- AGENT 2 RESPONSE ---\n{agent2_response}\n"
        )
        self._write_to_rolling_log(log_block)

    async def aquery(self, query_text: str):
        print(f"🔍 Processing RAG Pipeline for: {query_text}")
        
        # Initial RAG - Use async version
        initial_nodes = await self.fusion_retriever.aretrieve(query_text)
        
        # Reranking
        import asyncio
        reranked_nodes = await asyncio.to_thread(
            self.reranker.postprocess_nodes, initial_nodes, query_str=query_text
        )
        
        # Use the template to generate the exact same prompt sent to LLM for logging
        context_str = "\n\n".join([n.node.get_content(metadata_mode="llm") for n in reranked_nodes])
        agent1_prompt = self.qa_prompt_tmpl.format(
            context_str=context_str,
            query_str=query_text
        )

        # 1st Agent Response (Synthesis)
        from llama_index.core.schema import QueryBundle
        response = await self.query_engine.asynthesize(
            query_bundle=QueryBundle(query_text),
            nodes=reranked_nodes
        )
        
        # We attach the agent1_prompt to the response object temporarily so main.py can log it later 
        # or we log it here if we had agent2 prompt. 
        # Actually, let's just return both for consolidated logging if needed, 
        # but the request asks for specific steps.
        
        # Add property to response for easier retrieval
        response.agent1_prompt = agent1_prompt
        return response


    async def generate_refined_response(self, query_text: str, rag_response: str, history: List[str], summary: str = None, agent1_prompt: str = "", bot_name: str = None):
        history_str = "\n".join(history)
        persona = self._get_persona_prompt(bot_name)
        
        refine_prompt = (
            f"## ROLE\n{persona}\n\n"
            "## CONTEXT_DATA\n"
            
            f"<knowledge_base>\n"
            f"{rag_response}\n"
            f"</knowledge_base>\n\n"
            
            f"<conversation_summary>\n"
            f"{summary if summary else 'История пуста.'}\n"
            f"</conversation_summary>\n\n"
            
            f"<chat_memory>\n"
            f"{history_str}\n"
            f"</chat_memory>\n\n"
            
            "--- \n"
            "## TASK\n"
            "1. **Анализ:** Определи, касается ли вопрос информации из <knowledge_base>. Если да — используй её как основной источник фактов.\n"
            "2. **Связность:** Используй <chat_memory>, чтобы ответ логично продолжал текущую беседу и учитывал то, о чем вы уже говорили.\n"
            "3. **Стиль:** Отвечай на русском языке, кратко, сохраняя характер своего персонажа. Обращайся к пользователю 'Искатель' или 'Путник'. Не выдумывай новых эпитетов.\n"
            "4. **Приоритет:** Если информация в базе знаний противоречит истории сообщений, приоритет отдается базе знаний.\n\n"
            f"Вопрос пользователя: {query_text}"
        )
        response = await Settings.llm.acomplete(refine_prompt)
        final_response = str(response)
        
        # Consolidated Log
        self._log_interaction(
            query=query_text,
            agent1_prompt=agent1_prompt,
            agent1_response=rag_response,
            agent2_prompt=refine_prompt,
            agent2_response=final_response
        )
        
        return final_response

    async def generate_summary(self, prev_summary: str, messages: List[str]):
        msgs_str = "\n".join(messages)
        prompt = f"""
Текущее краткое содержание беседы:
{prev_summary if prev_summary else "отсутствует"}

Новые сообщения:
{msgs_str}

Задача:
Обнови краткое содержание беседы.

Правила:
- Сохраняй только долгосрочно важную информацию.
- Удаляй детали, временные обсуждения и повторения.
- Если новая информация заменяет старую — обнови её, а не добавляй.
- Старайся сокращать текст при каждом обновлении.
- Сохраняй только цели пользователя, важный контекст и выводы.

Ответ только на русском.
"""
        response = await Settings.llm.acomplete(prompt)
        text_response = str(response).strip()
        
        # Log the summary interaction
        log_block = (
            f"TIMESTAMP: {datetime.now().isoformat()}\n"
            f"--- SUMMARY PROMPT ---\n{prompt}\n"
            f"{'-'*40}\n"
            f"--- SUMMARY RESPONSE ---\n{text_response}\n"
        )
        self._write_to_rolling_log(log_block, log_path="cache/summaries_logs.txt", max_entries=10)
        
        return text_response

def main():
    import asyncio
    assistant = RAGAssistant()
    query = "Что произошло в Барановичах?"
    response = asyncio.run(assistant.aquery(query))
    print(f"\nQUERY: {query}\nANSWER: {response}")

if __name__ == "__main__":
    main()