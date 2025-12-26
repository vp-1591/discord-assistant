import os
import json
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.core.schema import TextNode, NodeWithScore
# ... rest of your imports ...
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
import Stemmer
from llama_index.llms.ollama import Ollama

# --- 0. CONFIG ---
PERSIST_DIR = "./llama_index_storage"
FORCE_REBUILD = False  # Set to True to re-process JSONs

# --- 1. SETTINGS CONFIGURATION ---
Settings.embed_model = OllamaEmbedding(
    model_name="bge-m3", 
    base_url="http://localhost:11434",
    request_timeout=300.0,
    keep_alive=0, # Unload after each request to save RAM/VRAM
)

Settings.llm = Ollama(
    model="mistral:latest", 
    request_timeout=120.0, 
    keep_alive=0, # Unload after each request
    additional_kwargs={"options": {"num_ctx": 4096}} 
)
Settings.embed_batch_size = 50 

# --- 2. CUSTOM DATA INGESTION ---
import re
from datetime import datetime
from collections import defaultdict

def load_nodes_from_json(directory: str) -> List[TextNode]:
    """
    Custom loading function:
    1. Filters links and NJb-bbGM codes.
    2. Filters repetitive user messages.
    3. Groups by day/channel.
    4. Chunks at 600 symbols with 75-sym overlap and [CUT] tags.
    """
    if not os.path.exists(directory):
        return []

    processed_data = [] # List of filtered dicts
    
    # regex patterns
    link_pattern = re.compile(r'^https?://[^\s]+$')
    code_pattern = re.compile(r'^[A-Za-z]{3}-[A-Za-z]{4}$')

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list): continue
                    
                    last_user = None
                    last_msg = None

                    for msg in data:
                        text = str(msg.get("message", "")).strip()
                        user = str(msg.get("user", "Unknown"))
                        
                        # -- FILTERS --
                        # 1. Links & Codes
                        if link_pattern.search(text) or code_pattern.search(text):
                            continue
                        
                        # 2. Sequential Repetition Filter
                        if user == last_user and text == last_msg:
                            continue
                        
                        last_user, last_msg = user, text
                        
                        # Extract date
                        ts = msg.get("timestamp", "2000-01-01T00:00:00")
                        date_str = ts.split("T")[0]
                        channel = msg.get("channel", "unknown")
                        
                        processed_data.append({
                            "user": user,
                            "text": text,
                            "date": date_str,
                            "channel": channel
                        })
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    # -- GROUPING BY DAY AND CHANNEL --
    groups = defaultdict(list)
    for entry in processed_data:
        groups[(entry['channel'], entry['date'])].append(entry)

    nodes = []
    CHUNK_LIMIT = 600
    OVERLAP = 60

    def split_long_message(author: str, text: str, metadata: dict):
        """Splits a single message larger than 600 symbols into multiple nodes."""
        remaining_text = text
        is_first = True
        
        while remaining_text:
            prefix = f"{author}: " if is_first else f"{author}: [CUT] "
            # Max text we can take is CHUNK_LIMIT - prefix length - buffer for " [CUT]"
            max_chunk_len = CHUNK_LIMIT - len(prefix) - 10 
            
            if len(remaining_text) <= max_chunk_len:
                node = TextNode(text=prefix + remaining_text, metadata=metadata)
                nodes.append(node)
                break
                
            # -- End of current chunk --
            # Find last space within max_chunk_len to avoid cutting words
            split_idx = remaining_text.rfind(" ", 0, max_chunk_len)
            if split_idx <= 0: 
                split_idx = max_chunk_len # No space found, force cut
                
            chunk = remaining_text[:split_idx]
            node = TextNode(text=prefix + chunk + " [CUT]", metadata=metadata)
            nodes.append(node)
            
            # -- Start of next chunk (Overlap) --
            # We move back by OVERLAP, then find the nearest space 
            # to avoid starting the next node in the middle of a word.
            overlap_anchor = split_idx - OVERLAP
            if overlap_anchor <= 0:
                new_start_idx = split_idx
            else:
                # Find space before or at the anchor
                space_idx = remaining_text.rfind(" ", 0, overlap_anchor)
                if space_idx == -1:
                    new_start_idx = overlap_anchor # Fallback
                else:
                    new_start_idx = space_idx
            
            # Safety: ensure we always make progress
            if new_start_idx >= split_idx:
                new_start_idx = split_idx
                
            remaining_text = remaining_text[new_start_idx:].strip()
            is_first = False

    for (channel, date), messages in groups.items():
        metadata = {"date": date, "channel": channel}
        current_accumulated = ""
        
        for m in messages:
            line = f"{m['user']}: {m['text']}\n"
            
            # If current single line is giant, split it
            if len(line) > CHUNK_LIMIT:
                if current_accumulated:
                    node = TextNode(text=current_accumulated.strip(), metadata=metadata)
                    nodes.append(node)
                    current_accumulated = ""
                split_long_message(m['user'], m['text'], metadata)
                continue

            # If adding this line exceeds the 600 limit, flush and start over
            if len(current_accumulated) + len(line) > CHUNK_LIMIT:
                if current_accumulated:
                    node = TextNode(text=current_accumulated.strip(), metadata=metadata)
                    nodes.append(node)
                current_accumulated = line
            else:
                current_accumulated += line
        
        if current_accumulated:
            node = TextNode(text=current_accumulated.strip(), metadata=metadata)
            nodes.append(node)

    print(f"✅ Pre-processing complete. Created {len(nodes)} nodes.")
    return nodes

def main():
    print("🚀 Initializing LlamaIndex Hybrid RAG Pipeline...")

    if not os.path.exists(PERSIST_DIR) or FORCE_REBUILD:
        print("📁 Building fresh index from JSON...")
        nodes = load_nodes_from_json("./messages_json")
        if not nodes:
            print("⚠️ Warning: No nodes loaded. Using dummy data.")
            nodes = [TextNode(text="Добряк тут!", metadata={"date": "2023-12-23"})]
            
        index = VectorStoreIndex(nodes, show_progress=True)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print(f"✅ Index persisted to {PERSIST_DIR}")
    else:
        print("💾 Loading index from storage (skipping JSON processing)...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        # Extract nodes from the persistent docstore for BM25
        nodes = list(index.docstore.docs.values())
        print(f"✅ Loaded {len(nodes)} nodes from docstore.")

    # --- 4. RETRIEVAL ENGINE (HYBRID) ---
    print("🔍 Setting up Hybrid Retrieval (Vector + BM25)...")
    # A. Vector Retriever
    vector_retriever = index.as_retriever(similarity_top_k=100)

    # B. BM25 Retriever
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes, 
        similarity_top_k=100,
        stemmer=Stemmer.Stemmer("russian"),
        language="russian"
    )

    # C. Hybrid Fusion
    fusion_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=100,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )

    # --- 5. RERANKING ---
    print("🎯 Initializing Reranker...")
    reranker = FlagEmbeddingReranker(
        model="BAAI/bge-reranker-v2-m3",
        top_n=7 # Top 10 to provide more context to the LLM
    )

    # --- 6. QUERY ENGINE ---
    from llama_index.core import PromptTemplate
    from llama_index.core.query_engine import RetrieverQueryEngine

    # Custom prompt that emphasizes the importance of dates and knows TODAY'S date
    current_date = datetime.now().strftime("%Y-%m-%d")
    qa_prompt_tmpl_str = (
        f"Сегодняшняя дата: {current_date}\n"
        "Ниже представлены фрагменты истории чата из Discord. "
        "У каждого фрагмента есть metadata (дата, канал) и содержание (content).\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Используя предоставленную историю, ответь на вопрос: {query_str}\n"
        "ОБЯЗАТЕЛЬНО: Если вопрос касается даты, например 'первый', 'последний' или 'сейчас', "
        "сравнивай даты сообщений и выбирай соответствующее упоминание.\n"
        "Ответ:"
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    # Configure the index/nodes to show date metadata to the LLM
    for node in nodes:
        node.excluded_llm_metadata_keys = [] # Don't hide anything
        node.metadata_template = "{key}: {value}"
        node.text_template = "Metadata: {metadata_str}\nContent: {content}"

    query_engine = RetrieverQueryEngine.from_args(
        retriever=fusion_retriever,
        node_postprocessors=[reranker],
        response_mode="compact",
    )
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
    )

    query = "Что произошло в Барановичах?"
    print(f"\n🔍 Querying: '{query}'")
    response = query_engine.query(query)
    
    print("\n--- FINAL ANSWER ---")
    print(str(response))

    print("\n--- SOURCE DOCUMENTS USED ---")
    for node_with_score in response.source_nodes:
        score = node_with_score.score
        node = node_with_score.node
        print(f"Score: {score:.4f} | Content: {node.get_content()}...")
        print("-" * 50)

    # --- RETRIEVAL TEST ---
    # print("\n🧪 Running retrieval validation...")
    # test_passed = any("Dodik - Добряк" in node.node.get_content() for node in response.source_nodes)
    # if test_passed:
    #     print("✅ PASS: 'Dodik - Добряк' is present in the source nodes.")
    # else:
    #     print("❌ FAIL: 'Dodik - Добряк' was NOT found in the source nodes.")

#     # --- 4. RETRIEVAL ENGINE (HYBRID) ---
#     # A. Vector Retriever (Top 50)
#     vector_retriever = index.as_retriever(similarity_top_k=100)

#     # B. BM25 Retriever (Top 20)
#     # LlamaIndex's BM25Retriever works directly on nodes
#     bm25_retriever = BM25Retriever.from_defaults(
#         nodes=nodes, 
#         similarity_top_k=100,
#         stemmer=Stemmer.Stemmer("russian"), # Use Russian stemmer
#         language="russian"
#     )

#     # C. Hybrid Fusion
#     # QueryFusionRetriever combines results using Reciprocal Rank Fusion (RRF) by default
#     fusion_retriever = QueryFusionRetriever(
#         [vector_retriever, bm25_retriever],
#         similarity_top_k=100,
#         num_queries=1, # Setting to 1 bypasses query expansion/transformation
#         mode="reciprocal_rerank", # Standard RRF
#         use_async=False,
#     )

#     # --- 5. RERANKING ---
#     # Using FlagEmbeddingReranker for the local BGE-Reranker-v2-m3
#     reranker = FlagEmbeddingReranker(
#         model="BAAI/bge-reranker-v2-m3",
#         top_n=40
#     )

#     # --- 6. EXECUTION ---
#     query = "Кто последний Добряк?"
#     print(f"\n🔍 Querying: '{query}'")

#     # Step 1: Hybrid Retrieval
#     retrieved_nodes = fusion_retriever.retrieve(query)
    
#     # --- 6.1 TEST ASSERTION ---
#     print("\n🧪 Running retrieval tests...")
#     test_found = any("Dodik - Добряк" in n.text for n in retrieved_nodes)
#     try:
#         assert test_found, "Critical test failed: 'Dodik - Добряк' not found in retrieved results!"
#         print("✅ Test Passed: 'Dodik - Добряк' found in retrieved documents.")
#     except AssertionError as e:
#         print(f"❌ {e}")

#     # Step 2: Reranking
#     final_results = reranker.postprocess_nodes(retrieved_nodes, query_str=query)

#     # --- 7. FORMATTED OUTPUT & LOGGING ---
#     header = f"\n{'='*80}\nQUERY: {query}\n{'-'*80}\n{'SCORE':<8} | {'DATE':<12} | {'CHANNEL':<15} | {'CONTENT'}\n{'-'*80}"
   
    
#     with open("logs.txt", "a", encoding="utf-8") as log_file:
#         log_file.write(header + "\n")
        
#         if not final_results:
#             msg = "No results found."
#             log_file.write(msg + "\n")
#         else:
#             for node_with_score in final_results:
#                 score = node_with_score.score
#                 node = node_with_score.node
#                 date = node.metadata.get("date", "Unknown Date")
#                 channel = node.metadata.get("channel", "Unknown Channel")
#                 clean_content = node.text.replace("\n", " ")
                
#                 line = f"{score:<8.4f} | {date:<12} | {channel:<15} | {clean_content}"
#                 log_file.write(line + "\n")
        
#         log_file.write("="*80 + "\n\n")
    
#     print("\n🧪 Running reranking tests...")
#     # Find the target node and its position
#     target_node = None
#     target_rank = -1
#     for idx, res in enumerate(final_results):
#         if "Dodik - Добряк" in res.node.text:
#             target_node = res
#             target_rank = idx + 1
#             break

#     try:
#         assert target_node is not None, "Critical test failed: 'Dodik - Добряк' not found in reranked results!"
#         print(f"✅ Test Passed: 'Dodik - Добряк' found in reranked documents.")
#         print(f"   [Rank: #{target_rank}] [Score: {target_node.score:.4f}]")
#         print(f"   Content: {target_node.node.text.strip()}")
#     except AssertionError as e:
#         print(f"❌ {e}")
#     print(f"📝 Results also saved to logs.txt")

if __name__ == "__main__":
    main()