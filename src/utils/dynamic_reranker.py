import torch
from typing import List, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, MetadataMode

from src.utils.logger_setup import trace_logger, log_trace


class DynamicGPUReranker(BaseNodePostprocessor):
    """
    A LlamaIndex-compatible reranker that performs native cross-encoder scoring
    using FlagEmbedding on the GPU, then immediately offloads the model to free VRAM.

    This achieves zero quality degradation vs the original FlagEmbeddingReranker
    (full cross-encoder tokenization with proper [SEP] tokens and classification head),
    while replicating Ollama's keep_alive=0 behavior by purging GPU memory after each call.
    """

    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 10
    use_fp16: bool = True

    @classmethod
    def class_name(cls) -> str:
        return "DynamicGPUReranker"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle.")
        if not nodes:
            return []

        log_trace(trace_logger, f"[DynamicGPUReranker] Starting rerank of {len(nodes)} nodes.")

        # Import here to avoid module-level side effects
        from FlagEmbedding import FlagReranker

        # Load onto CPU first — zero VRAM cost until we explicitly push to GPU
        reranker = FlagReranker(
            self.model_name,
            use_fp16=self.use_fp16,
            devices=["cpu"],
        )

        query_str = query_bundle.query_str
        pairs = [
            (query_str, node.node.get_content(metadata_mode=MetadataMode.EMBED))
            for node in nodes
        ]

        # Redirect compute to GPU for this call
        reranker.target_devices = ["cuda:0"]
        scores = reranker.compute_score(pairs)

        # compute_score returns a float when given a single pair
        if isinstance(scores, float):
            scores = [scores]

        for node, score in zip(nodes, scores):
            node.score = score

        new_nodes = sorted(nodes, key=lambda x: x.score if x.score is not None else 0, reverse=True)[: self.top_n]

        log_trace(
            trace_logger,
            f"[DynamicGPUReranker] Reranked to {len(new_nodes)} nodes. "
            f"Top score: {new_nodes[0].score:.3f} | Bottom score: {new_nodes[-1].score:.3f}"
        )

        # Offload: evict weights from VRAM immediately
        reranker.model.to("cpu")
        del reranker.model
        del reranker.tokenizer
        del reranker
        torch.cuda.empty_cache()

        log_trace(trace_logger, "[DynamicGPUReranker] VRAM offload complete.")

        return new_nodes
