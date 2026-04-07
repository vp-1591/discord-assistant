import json
import asyncio
import re
import sqlite3
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime

from llama_index.core import Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import FunctionTool
import Stemmer  # type: ignore
import sys
import textwrap

if TYPE_CHECKING:
    from src.data.opinion_manager import OpinionManager

from src.utils.logger_setup import sys_logger, trace_logger, log_trace
from src.utils.context import transaction_id, new_tx_id, run_with_context, span_id, new_span_id
from src.config.config import configure_settings, RAG_CACHE_PATH
from src.config.prompts import (
    get_qa_prompt_tmpl, SUMMARY_PROMPT_TEMPLATE,
    get_persona_prompt, get_system_prompt,
    SEARCH_ARCHIVE_DESC, FETCH_USER_OPINION_DESC, UPDATE_USER_OPINION_DESC,
    PEEK_CACHED_SEARCHES_DESC, PULL_CACHED_RESULT_DESC,
    AGENT1_HYBRID_SEARCH_DESC, AGENT1_FETCH_RAW_LOGS_DESC, AGENT1_SQL_QUERY_DESC
)
from src.data.ingestion import load_or_build_index, insert_new_nodes
from src.core.agent_core import ReActAgentWorkflow
from src.core.rag_cache import RAGCache

# Initialize global settings
configure_settings()

class RAGAssistant:
    def __init__(self, id_map: dict = None, name: str = "Глава Архива", opinion_manager: "OpinionManager" = None):
        self.id_map = id_map or {}
        self.name = name
        self.opinion_manager = opinion_manager
        self.rag_cache = RAGCache(persist_path=RAG_CACHE_PATH, capacity=5)
        self.index, self._nodes = load_or_build_index(self.id_map)
        self.fusion_retriever, self.reranker, self.query_engine = self._setup_query_engine()

    async def update_index(self):
        """Dynamically fetches new messages, inserts them into the DB, and live-reloads the query engine."""
        sys_logger.info("Initiating live index update...")
        new_nodes = await insert_new_nodes(self.index, self.id_map)
        
        if new_nodes or self.query_engine is None:
            if new_nodes:
                self._nodes.extend(new_nodes)
            
            if self._nodes:
                sys_logger.info(f"Live reloading query engine with {len(self._nodes)} total nodes...")
                
                # Run the heavy initialization (BM25 indexing & Reranker load) in a background thread
                new_fusion, new_rerank, new_engine = await asyncio.to_thread(self._setup_query_engine)
                
                # Switch to the new instances. 
                # (Note: Old instances will be cleaned up by Python's GC naturally)
                self.fusion_retriever = new_fusion
                self.reranker = new_rerank
                self.query_engine = new_engine
            
            return len(new_nodes) if new_nodes else 0
        return 0

    def _setup_query_engine(self):
        if not self._nodes:
            sys_logger.warning("No nodes found in index. Delaying query engine setup.")
            return None, None, None
            
        sys_logger.info(f"Setting up query engine with {len(self._nodes)} nodes...")
        vector_retriever = self.index.as_retriever(similarity_top_k=50)
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=self._nodes, 
            similarity_top_k=50,
            stemmer=Stemmer.Stemmer("russian"),
            language="russian"
        )
        from src.utils.llama_index_utils import LoggedRetriever
        logged_vector = LoggedRetriever("VECTOR_SEARCH", vector_retriever)
        logged_bm25 = LoggedRetriever("BM25_SEARCH", bm25_retriever)

        fusion_retriever = QueryFusionRetriever(
            [logged_vector, logged_bm25],
            similarity_top_k=50,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )
        
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=10, use_fp16=False)

        # query_engine is kept for potential future use and index health checks
        query_engine = RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            node_postprocessors=[reranker],
            response_mode="compact",
        )
        sys_logger.info("Query engine ready.")
        return fusion_retriever, reranker, query_engine

    def _build_agent1_tools(self) -> list:
        """Build the three tools available to Agent1's ReAct loop."""

        # --- Tool 1: hybrid_search ---
        async def hybrid_search(query: str) -> str:
            if self.fusion_retriever is None:
                return "Архив пуст или не инициализирован."
            
            # Pin new Span ID for this standalone RAG operation
            span_token = span_id.set(new_span_id())
            try:
                initial_nodes = await self.fusion_retriever.aretrieve(query)
                
                #(DEBUG): human-readable pointer to full logs
                if initial_nodes:
                    trace_logger.debug(
                        f"[hybrid_search] Retrieved {len(initial_nodes)} nodes pre-rerank for query: {query!r}"
                    )

                # LAYER 3 (TRACE): raw metadata dicts
                log_trace(
                    trace_logger,
                    "[PRE-RERANK RAW]",
                    nodes=[
                        {"score": n.score, "metadata": n.node.metadata,
                         "text_preview": n.node.get_content()[:300]}
                        for n in initial_nodes
                    ]
                )

                reranked_nodes = await asyncio.to_thread(
                    run_with_context, self.reranker.postprocess_nodes,
                    initial_nodes, query_str=query
                )
                if not reranked_nodes:
                    return "Ничего не найдено по запросу."

                parts = []
                for n in reranked_nodes:
                    node_type = n.node.metadata.get("node_type", "summary")
                    label = "[RAW LOG]" if node_type == "raw_log" else "[SUMMARY]"
                    parts.append(f"{label}\n{n.node.get_content(metadata_mode='llm')}")

                # LAYER 1 (DEBUG): final reranked summary
                full_response = "\n\n---\n\n".join(parts)
                trace_logger.debug(
                    f"[hybrid_search] Reranked to {len(reranked_nodes)} nodes."
                )

                return full_response
            finally:
                span_id.reset(span_token)

        # --- Tool 2: fetch_raw_logs ---
        async def fetch_raw_logs(source_chunk_ids: list) -> str:
            def _query():
                conn = sqlite3.connect(
                    "file:discord_data.db?mode=ro", uri=True,
                    check_same_thread=False
                )
                try:
                    conn.row_factory = sqlite3.Row
                    cur = conn.cursor()
                    
                    if not source_chunk_ids:
                        return "Предоставлен пустой список chunk_id."
                        
                    placeholders = ",".join("?" for _ in source_chunk_ids)
                    cur.execute(
                        f"SELECT author, content, timestamp, channel "
                        f"FROM messages WHERE chunk_id IN ({placeholders}) ORDER BY timestamp ASC",
                        tuple(source_chunk_ids)
                    )
                    rows = cur.fetchmany(100 * max(1, len(source_chunk_ids)))
                    if not rows:
                        return f"Нет сообщений для данных chunk_ids."
                    lines = [f"[{r['timestamp']}] #{r['channel']} {r['author']}: {r['content']}" for r in rows]
                    full_log = "\n".join(lines)
                    trace_logger.debug(f"[fetch_raw_logs] Fetched {len(rows)} lines for {len(source_chunk_ids)} chunks.")
                    return full_log
                finally:
                    conn.close()
            # run_with_context ensures Transaction_ID survives the thread boundary
            return await asyncio.to_thread(run_with_context, _query)

        # --- Tool 3: execute_sql ---
        _BLOCKED = re.compile(
            r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH)\b",
            re.IGNORECASE
        )

        async def execute_sql(query: str) -> str:
            if _BLOCKED.search(query):
                return "ОШИБКА: Разрешены только SELECT-запросы."

            def _query():
                conn = sqlite3.connect(
                    "file:discord_data.db?mode=ro", uri=True,
                    check_same_thread=False
                )
                try:
                    conn.row_factory = sqlite3.Row
                    cur = conn.cursor()
                    # Validate syntax before execution
                    try:
                        cur.execute(f"EXPLAIN {query}")
                    except sqlite3.OperationalError as e:
                        return f"ОШИБКА СИНТАКСИСА SQL: {e}"
                    cur.execute(query)
                    rows = cur.fetchmany(100)
                    if not rows:
                        return "Запрос выполнен успешно. Результатов не найдено."
                    headers = rows[0].keys()
                    lines = [" | ".join(str(r[h]) for h in headers) for r in rows]
                    header_line = " | ".join(headers)
                    truncated = " [результаты обрезаны до 100 строк]" if len(rows) == 100 else ""
                    return f"{header_line}\n" + "\n".join(lines) + truncated
                finally:
                    conn.close()
            # run_with_context ensures Transaction_ID survives the thread boundary
            return await asyncio.to_thread(run_with_context, _query)

        return [
            FunctionTool.from_defaults(
                async_fn=hybrid_search,
                name="hybrid_search",
                description=AGENT1_HYBRID_SEARCH_DESC
            ),
            FunctionTool.from_defaults(
                async_fn=fetch_raw_logs,
                name="fetch_raw_logs",
                description=AGENT1_FETCH_RAW_LOGS_DESC
            ),
            FunctionTool.from_defaults(
                async_fn=execute_sql,
                name="execute_sql",
                description=AGENT1_SQL_QUERY_DESC
            ),
        ]

    async def aquery(self, query_text: str) -> str:
        if self.fusion_retriever is None or self.reranker is None:
            sys_logger.warning("aquery called but query_engine is not initialized (index is empty).")
            return "Архив пуст или ещё инициализируется. Пожалуйста, подождите или добавьте данные."

        # Inherit Transaction_ID if already in a session, else create new
        current_tx = transaction_id.get()
        is_new_tx = False
        if current_tx == "SYS":
            tx = new_tx_id()
            tx_token = transaction_id.set(tx)
            is_new_tx = True
        else:
            tx = current_tx

        try:
            sys_logger.info(f"[Agent1] Starting ReAct loop | TxID={tx} | query={query_text!r}")
            agent1_system = get_qa_prompt_tmpl(self.name)
            tools = self._build_agent1_tools()

            agent1 = ReActAgentWorkflow(
                llm=Settings.llm,
                tools=tools,
                system_prompt=agent1_system,
                agent_name="Agent1",
                timeout=150
            )

            result = await agent1.run(input=query_text)
            response_str = result.get("response", str(result)).strip() if isinstance(result, dict) else str(result).strip()

            return response_str
        finally:
            if is_new_tx:
                transaction_id.reset(tx_token)

    def _build_tools(self, author_id: str, author_name: str) -> list:
        om = self.opinion_manager

        async def search_archive(search_query: str) -> str:
            response_str = str(await self.aquery(search_query))
            
            # Extract and log internal reasoning before stripping
            thoughts = re.findall(r"<thought>(.*?)</thought>", response_str, flags=re.DOTALL)
            for i, thought in enumerate(thoughts):
                trace_logger.info(f"--- AGENT 1 INTERNAL THOUGHT [{i+1}] ---\n{thought.strip()}\n")
            
            # Remove <thought> blocks to provide only the relevant summary to Agent 2
            clean_response = re.sub(r"<thought>.*?</thought>", "", response_str, flags=re.DOTALL).strip()
            
            # Store in cache
            self.rag_cache.store(search_query, clean_response)
            
            return clean_response

        async def peek_cached_searches() -> str:
            results = self.rag_cache.get_recent_queries()
            return json.dumps(results, ensure_ascii=False)

        async def pull_cached_result(result_id: int) -> str:
            return self.rag_cache.get_result_by_id(result_id)

        async def fetch_user_opinion(user_display_name: str) -> str:
            if om is None: return "Opinion system is not available."
            if user_display_name.lower() == author_name.lower():
                return f"Instruction: You already HAVE your stance on {author_name} in your context. Use it."
            
            om.opinions = om._load_opinions()
            matches = om.find_targets(user_display_name, threshold=0.6)
            profile = matches[0][1] if matches else None
            return json.dumps(profile, ensure_ascii=False) if profile else f"No memories found for '{user_display_name}'."

        async def update_user_opinion(current_stance: str, new_stance: str, history_note: str) -> str:
            if om is None: return "Opinion system is not available."
            profile = om.get_user_profile(author_id)
            actual_stance = profile.get("head_of_archive_stance") if profile else "None"
            
            if current_stance.strip() != str(actual_stance).strip():
                return f"ERROR: Protocol violation. Current stance does not match internal reality."

            await om.update_user_opinion(user_id=author_id, name=author_name, stance=new_stance, interaction=history_note)
            return f"Success: Your internal thoughts about {author_name} have been updated."

        return [
            FunctionTool.from_defaults(
                async_fn=search_archive, 
                name="search_archive",
                description=SEARCH_ARCHIVE_DESC
            ),
            FunctionTool.from_defaults(
                async_fn=peek_cached_searches,
                name="peek_cached_searches",
                description=PEEK_CACHED_SEARCHES_DESC
            ),
            FunctionTool.from_defaults(
                async_fn=pull_cached_result,
                name="pull_cached_result",
                description=PULL_CACHED_RESULT_DESC
            ),
            FunctionTool.from_defaults(
                async_fn=fetch_user_opinion,
                description=FETCH_USER_OPINION_DESC
            ),
            FunctionTool.from_defaults(
                async_fn=update_user_opinion,
                description=UPDATE_USER_OPINION_DESC
            ),
        ]

    async def generate_refined_response(
        self, query_text: str, history: List[str], summary: str = None, 
        bot_name: str = None, author_id: str = "", author_name: str = "",
        replied_to_msg: str = None
    ):
        history_str = "\n".join(history)
        persona = get_persona_prompt(bot_name or self.name)
        
        author_profile = "None"
        if self.opinion_manager:
            profile_data = self.opinion_manager.get_user_profile(author_id)
            if profile_data:
                author_profile = json.dumps(profile_data, ensure_ascii=False)

        system_prompt = get_system_prompt(
            author_name=author_name,
            persona=persona,
            author_profile=author_profile,
            summary=summary,
            history_str=history_str,
            query_text=query_text,
            replied_to_msg=replied_to_msg
        )

        # Inherit Transaction_ID if already in a session, else create new
        current_tx = transaction_id.get()
        is_new_tx = False
        if current_tx == "SYS":
            tx = new_tx_id()
            tx_token = transaction_id.set(tx)
            is_new_tx = True
        else:
            tx = current_tx

        try:
            sys_logger.info(f"[Agent2] Starting persona loop | TxID={tx} | user={author_name!r}")

            tools = self._build_tools(author_id=author_id, author_name=author_name)
            agent = ReActAgentWorkflow(llm=Settings.llm, tools=tools, system_prompt=system_prompt, agent_name="Agent2", timeout=300)

            agent_response = await agent.run(input=query_text)
            final_response = agent_response.get('response', str(agent_response)).strip() if isinstance(agent_response, dict) else str(agent_response).strip()
            
            trace_logger.info(f"--- AGENT TRANSACTION COMPLETE ---\n{'='*50}")
            return final_response
        finally:
            if is_new_tx:
                transaction_id.reset(tx_token)

    async def generate_summary(self, prev_summary: str, messages: List[str]):
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            prev_summary=prev_summary if prev_summary else "отсутствует",
            msgs_str="\n".join(messages)
        )
        response = await Settings.llm.acomplete(prompt)
        text_response = str(response).strip()
        trace_logger.info(f"--- SUMMARY PROMPT ---\n{prompt}\n--- SUMMARY RESPONSE ---\n{text_response}\n")
        return text_response

if __name__ == "__main__":
    assistant = RAGAssistant()
    query = "Что произошло в Барановичах?"
    response = asyncio.run(assistant.aquery(query))
    print(f"\nQUERY: {query}\nANSWER: {response}")