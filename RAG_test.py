from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
import create_store_rag
from datetime import datetime

retriever = create_store_rag.vector_store.as_retriever(search_kwargs={"k": 20})
compressor = FlashrankRerank(top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
last_query = 'Что произошло при Барановичах?'

# --- 1. SEARCH: SEMANTIC (VECTOR) ---
retrieved_docs = create_store_rag.vector_store.similarity_search(last_query, k=10) 

# --- 2. SEARCH: KEYWORD (LITERAL) ---
# Extract the most likely noun for keyword search (e.g., 'Добряк')
# For now, we'll try to find any part of the query that isn't a common stopword
query_words = [w.strip('?!.,') for w in last_query.split() if len(w) > 3]
main_keyword = query_words[-1] if query_words else last_query

# Use Chroma's '$contains' filter to find the exact string in the documents
keyword_results = create_store_rag.vector_store.get(
    where_document={"$contains": main_keyword},
    limit=10
)

# Convert raw Chroma results back to LangChain Document objects for consistency
keyword_docs = []
for i in range(len(keyword_results['documents'])):
    keyword_docs.append(create_store_rag.Document(
        page_content=keyword_results['documents'][i],
        metadata=keyword_results['metadatas'][i]
    ))

# Combine both for the reranker (unique results only)
all_docs_list = list({d.page_content: d for d in (retrieved_docs + keyword_docs)}.values())

# --- LOGGING RESULTS TO FILE ---

with open("logs.txt", "a", encoding="utf-8") as f:
    f.write(f"\n{'#'*80}\n")
    f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"QUERY: {last_query}\n")
    f.write(f"{'#'*80}\n")

    f.write(f"\n[1] SEMANTIC SEARCH RESULTS (Top 10)\n")
    for i, d in enumerate(retrieved_docs):
        f.write(f"  ({i+1}) {d.metadata.get('date')} | {d.page_content}\n")

    f.write(f"\n[2] KEYWORD SEARCH RESULTS (Keyword: '{main_keyword}')\n")
    if not keyword_docs:
        f.write("  No direct keyword matches found.\n")
    else:
        for i, d in enumerate(keyword_docs):
            f.write(f"  ({i+1}) {d.metadata.get('date')} | {d.page_content}\n")

    f.write(f"\n[3] RERANKING RESULTS (Top 5)\n")
    if not all_docs_list:
        f.write("  No documents to rerank.\n")
    else:
        # Use the compressor directly on our hybrid document list
        compressed_docs = compressor.compress_documents(all_docs_list, last_query)
        for i, d in enumerate(compressed_docs):
            score = d.metadata.get('relevance_score', 'N/A')
            score_val = f"{score:.4f}" if isinstance(score, (int, float)) else score
            f.write(f"  ({i+1}) SCORE: {score_val} | DATE: {d.metadata.get('date')}\n")
            f.write(f"  CONTENT: {d.page_content}\n")
            f.write(f"  {'-'*40}\n")

print(f"Results for query '{last_query}' have been logged to logs.txt")