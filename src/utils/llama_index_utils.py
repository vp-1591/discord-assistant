from typing import Any
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle
from src.utils.logger_setup import trace_logger, log_trace

class LoggedRetriever(BaseRetriever):
    """
    A transparent retriever proxy that intercepts retrieval calls 
    and logs the intermediate nodes straight to agent_traces_full.log
    """
    def __init__(self, name: str, target_retriever: BaseRetriever, **kwargs: Any):
        super().__init__(**kwargs)
        # Using dict bypass to avoid Pydantic strict field validation errors
        self.__dict__['_name'] = name
        self.__dict__['_target_retriever'] = target_retriever

    def _retrieve(self, query_bundle: QueryBundle, **kwargs: Any):
        nodes = self._target_retriever._retrieve(query_bundle, **kwargs)
        
        log_trace(
            trace_logger, 
            f"[PRE-FUSION | {self._name} RAW]", 
            nodes=[
                {
                    "score": n.score, 
                    "type": n.node.metadata.get('node_type', '?'), 
                    "full_text": n.node.get_content()
                } 
                for n in nodes
            ]
        )
        return nodes

    async def _aretrieve(self, query_bundle: QueryBundle, **kwargs: Any):
        nodes = await self._target_retriever._aretrieve(query_bundle, **kwargs)
        
        log_trace(
            trace_logger, 
            f"[PRE-FUSION | {self._name} RAW]", 
            nodes=[
                {
                    "score": n.score, 
                    "type": n.node.metadata.get('node_type', '?'), 
                    "full_text": n.node.get_content()
                } 
                for n in nodes
            ]
        )
        return nodes
