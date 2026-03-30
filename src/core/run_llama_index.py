import json
import asyncio
import re
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime

from llama_index.core import Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import QueryBundle
import Stemmer

if TYPE_CHECKING:
    from src.data.opinion_manager import OpinionManager

from src.utils.logger_setup import sys_logger, trace_logger
from src.config.config import configure_settings, RAG_CACHE_PATH
from src.config.prompts import (
    get_qa_prompt_tmpl, SUMMARY_PROMPT_TEMPLATE, 
    get_persona_prompt, get_system_prompt,
    SEARCH_ARCHIVE_DESC, FETCH_USER_OPINION_DESC, UPDATE_USER_OPINION_DESC,
    PEEK_RECENT_SEARCHES_DESC, PULL_CACHED_RESULT_DESC
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
                self.fusion_retriever, self.reranker, self.query_engine = self._setup_query_engine()
            
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
        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=50,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )
        
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=10, use_fp16=False)
        
        qa_prompt_tmpl = get_qa_prompt_tmpl(self.name)

        query_engine = RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            node_postprocessors=[reranker],
            response_mode="compact",
        )
        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
        sys_logger.info("Query engine ready.")
        return fusion_retriever, reranker, query_engine

    async def aquery(self, query_text: str):
        if self.query_engine is None or self.fusion_retriever is None:
            sys_logger.warning("aquery called but query_engine is not initialized (index is empty).")
            return "Архив пуст или ещё инициализируется. Пожалуйста, подождите или добавьте данные."

        sys_logger.info(f"Processing RAG Pipeline for: {query_text}")
        initial_nodes = await self.fusion_retriever.aretrieve(query_text)
        
        reranked_nodes = await asyncio.to_thread(
            self.reranker.postprocess_nodes, initial_nodes, query_str=query_text
        )
        
        context_str = "\n\n".join([n.node.get_content(metadata_mode="llm") for n in reranked_nodes])
        qa_prompt_tmpl = get_qa_prompt_tmpl(self.name)
        agent1_prompt = qa_prompt_tmpl.format(
            context_str=context_str,
            query_str=query_text
        )

        response = await self.query_engine.asynthesize(
            query_bundle=QueryBundle(query_text),
            nodes=reranked_nodes
        )
        
        trace_logger.info(f"--- AGENT 1 (RAG) PROMPT ---\n{agent1_prompt}\n")
        return response

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

        async def peek_recent_searches() -> list:
            return self.rag_cache.get_recent_queries()

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
                async_fn=peek_recent_searches,
                name="peek_recent_searches",
                description=PEEK_RECENT_SEARCHES_DESC
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

        tools = self._build_tools(author_id=author_id, author_name=author_name)
        agent = ReActAgentWorkflow(llm=Settings.llm, tools=tools, system_prompt=system_prompt, timeout=300)

        agent_response = await agent.run(input=query_text)
        final_response = agent_response.get('response', str(agent_response)).strip() if isinstance(agent_response, dict) else str(agent_response).strip()
        
        trace_logger.info(f"--- AGENT TRANSACTION COMPLETE ---\n{'='*50}")
        return final_response

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