from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document, BaseDocumentCompressor
from typing import List

# --- MANUAL CLASS DEFINITIONS (To bypass ImportErrors) ---

class EnsembleRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    weights: List[float]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        all_docs = []
        for retriever in self.retrievers:
            all_docs.extend(retriever.invoke(query))
        
        # Deduplicate
        unique_docs = {}
        for d in all_docs:
            if d.page_content not in unique_docs:
                unique_docs[d.page_content] = d
        return list(unique_docs.values())

class ManualContextualCompressionRetriever(BaseRetriever):
    base_compressor: BaseDocumentCompressor
    base_retriever: BaseRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        if not docs:
            return []
        return self.base_compressor.compress_documents(docs, query)

# Use our manual class
ContextualCompressionRetriever = ManualContextualCompressionRetriever

from sentence_transformers import CrossEncoder

class ManualCrossEncoderReranker(BaseDocumentCompressor):
    model: CrossEncoder
    top_n: int = 3

    class Config:
        arbitrary_types_allowed = True

    def compress_documents(
        self, documents: List[Document], query: str, callbacks=None
    ) -> List[Document]:
        if not documents:
            return []
        # Prepare inputs for cross encoder: pairs of (query, doc_text)
        inputs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(inputs)
        
        # Attach scores to documents
        for doc, score in zip(documents, scores):
            doc.metadata["relevance_score"] = float(score)

        # Sort by score descending
        sorted_docs = sorted(documents, key=lambda x: x.metadata["relevance_score"], reverse=True)
        return sorted_docs[:self.top_n]

CrossEncoderReranker = ManualCrossEncoderReranker

# --- IMPORTS ---
import create_store_rag
import load_documents_into_store
from datetime import datetime
import sys
import json
import os

# --- 1. INITIALIZE RETRIEVERS ---

# A. Vector Retriever (Semantic)
vector_retriever = create_store_rag.vector_store.as_retriever(search_kwargs={"k": 10})

# B. BM25 Retriever (Keyword)
def load_docs_for_bm25(directory="./messages_json"):
    docs = []
    if not os.path.exists(directory):
        print(f"WARNING: Directory {directory} not found!")
        return []
    
    print("Loading documents for BM25 index...")
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # --- ИЗМЕНЕНИЕ: ГРУППИРОВКА ДЛЯ BM25 ---
                    # Мы объединяем сообщения в "псевдо-чанки" по 5-10 штук, 
                    # чтобы у BM25 был контекст, как у вектора.
                    
                    chunk_size = 10 
                    messages = [m.get("message", "") for m in data if len(str(m.get("message", ""))) > 1]
                    timestamps = [m.get("timestamp", "") for m in data if len(str(m.get("message", ""))) > 1]
                    
                    for i in range(0, len(messages), 5): # Шаг 5 (перекрытие)
                        window_msgs = messages[i : i + chunk_size]
                        window_ts = timestamps[i : i + chunk_size]
                        
                        if not window_msgs: continue
                        
                        # Собираем текст чанка
                        chunk_text = "\n".join(window_msgs)
                        
                        # Берем дату из середины чанка
                        mid_idx = len(window_ts) // 2
                        ts = window_ts[mid_idx] if window_ts else "Unknown"
                        
                        docs.append(Document(
                            page_content=chunk_text, 
                            metadata={"timestamp": ts, "source": filename}
                        ))
                        
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return docs

print("Building BM25 Index (this may take a moment)...")
try:
    bm25_docs = load_docs_for_bm25("./messages_json")
    
    if bm25_docs:
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = 10
        
        # Force a test query immediately
        print("DEBUG: Running BM25 test query for 'Добряк'...")
        test_results = bm25_retriever.invoke("Добряк")
        print(f"DEBUG: BM25 test for 'Добряк' returned {len(test_results)} docs.")
        if not test_results:
            print("CRITICAL ERROR: BM25 is blind to the word 'Добряк'. Check path/encoding.")
    else:
        print("Warning: No documents found for BM25 index.")
        bm25_retriever = None
except ImportError as e:
    if "rank_bm25" in str(e):
        print("\n\nCRITICAL: 'rank_bm25' package is missing.")
        print("Please run: pip install rank_bm25\n")
        sys.exit(1)
    else:
        raise e
except Exception as e:
    print(f"Error building BM25 index: {e}")
    bm25_retriever = None

# C. Ensemble Retriever (Hybrid)
if bm25_retriever:
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.4, 0.6] # 40% Vector, 60% Keyword
    )
else:
    print("Warning: Using only Vector Retriever due to BM25 failure.")
    ensemble_retriever = vector_retriever

# D. Reranker (Cross-Encoder)
print("Loading Cross-Encoder Model (this may take a moment)...")
# Using BAAI/bge-reranker-v2-m3 as requested
model_name = "BAAI/bge-reranker-v2-m3"
model = CrossEncoder(model_name)
compressor = CrossEncoderReranker(model=model, top_n=5)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)

# --- 2. EXECUTE SEARCH FOR MULTIPLE QUERIES ---

queries = [
    "Отношение Klaus Adler к Barmacar",
    "Кто последний Добряк?",
    "Кто любит китайскую культуру?",
    "Кто создал совет?",
    "Что произошло при Барановичах?"
]

print(f"Starting batch processing of {len(queries)} queries...")

for i, query in enumerate(queries):
    print(f"\nProcessing Query {i+1}/{len(queries)}: '{query}'")
    
    # Execute Hybrid Search
    hybrid_docs = ensemble_retriever.invoke(query)
    
    # Rerank
    compressed_docs = compressor.compress_documents(hybrid_docs, query)

    # Log results
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{'#'*80}\n")
        f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"QUERY: {query}\n")
        f.write(f"MODE: Hybrid Search (Vector + BM25[0.6]) -> CrossEncoder ({model_name})\n")
        f.write(f"{'#'*80}\n")

        f.write(f"\n[1] HYBRID RETRIEVAL RESULTS (Top {len(hybrid_docs)})\n")
        for j, d in enumerate(hybrid_docs):
            date = d.metadata.get('date', 'Unknown')
            # Full content logging
            f.write(f"  ({j+1}) {date} | {d.page_content}\n") 
            f.write(f"  {'-'*20}\n")

        f.write(f"\n[2] RERANKED RESULTS (Top 5)\n")
        if not compressed_docs:
            f.write("  No documents to rerank.\n")
        else:
            for j, d in enumerate(compressed_docs):
                score = d.metadata.get('relevance_score', d.metadata.get('score', 'N/A'))
                score_val = f"{score:.4f}" if isinstance(score, (int, float)) else score
                date = d.metadata.get('date', 'Unknown')
                f.write(f"  ({j+1}) SCORE: {score_val} | DATE: {date}\n")
                f.write(f"  CONTENT: {d.page_content}\n")
                f.write(f"  {'-'*40}\n")
    
    print(f"Logged results for '{query}'")

print("\nAll queries processed.")