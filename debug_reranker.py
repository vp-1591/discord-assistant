from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.schema import MetadataMode

# 1. Initialize the Reranker (Same settings as your pipeline)
# print("Loading Reranker...")
# reranker = FlagEmbeddingReranker(
#     model="BAAI/bge-reranker-v2-m3",
#     top_n=5
# )

# 2. Define the Query
query = "Кто последний Добряк?"

# 3. Define the Candidates (The "Wrong" Top Result vs. The "Right" Result)

# Candidate A: The "Spammy" long story from 2024 (Currently winning)
doc_spam_text = """August: Сегодня произошла кровавая бойня... Dodik попытался воссесть на троне...
Dodik принял предложеные условия, благодаря чему фактически стал Добряк."""

# Candidate B: The correct short answer (Raw Text)
doc_correct_raw = "STR: Dodik - Добряк"

# Candidate C: The correct answer WITH injected metadata (What we want the model to see)
doc_correct_with_meta = "[Channel: летописи] [Date: 2025-12-08]\nContent: Dodik - Добряк"

# 4. Create Nodes
nodes = [
    NodeWithScore(node=TextNode(text=doc_spam_text, metadata={"channel": "летописи", "date": "2024-08-01"}), score=0.0),      # Node 0
    NodeWithScore(node=TextNode(text=doc_correct_raw, metadata={"channel": "летописи", "date": "2025-12-08"}), score=0.0),    # Node 1
    NodeWithScore(node=TextNode(text=doc_correct_with_meta, metadata={"channel": "летописи", "date": "2025-12-08"}), score=0.0) # Node 2
]


print(nodes[0].get_content(metadata_mode=MetadataMode.EMBED))

print("\n--- RUNNING DEBUG TEST ---")

# # 5. Rerank
# results = reranker.postprocess_nodes(nodes, query_str=query)

# # 6. Print Results
# for i, res in enumerate(results):
#     print(f"Rank {i+1}: Score {res.score:.4f}")
#     print(f"Content: {res.node.text[:100]}...")
#     print("-" * 30)