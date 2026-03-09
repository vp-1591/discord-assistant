# Discord Assistant — Architecture

## 1. Unified System Overview & Data Flow

This chart represents the high-level architecture, encompassing data ingestion, retrieval logic, the primary messaging pipeline, and the background Agent 3 (Social Auditor).

```mermaid
flowchart TD
    subgraph Discord["Discord API"]
        D_MSG["User Message (bot mention)"]
        D_EXPORT["Admin !export command"]
        D_REPLY["Bot Reply"]
    end

    subgraph Main["main.py — aclient"]
        ON_READY["on_ready()"]
        ON_MSG["on_message()"]
        HISTORY["Scan up to 50 msgs, keep up to 5 bot-relevant ones"]
        SUMMARY_LOAD["Load channel summary"]
        SUMMARY_SAVE["Save updated summary"]
        OP_FIND["OpinionManager: find_targets & get_profile"]
    end

    subgraph Export["export_chat.py"]
        EXPORT_FN["export_chat_to_json()"]
        MERGE["Merge last_known_names with old JSON"]
    end

    subgraph Storage["Storage & Cache"]
        JSON_FILES["messages_json/channel.json"]
        SAVED_IDX["llama_index_storage/ (Vector Index)"]
        OP_DB["cache/opinions.json"]
        SUM_DB["cache/summaries.json"]
    end

    subgraph RAG["run_llama_index.py — RAGAssistant"]
        LOAD_IDX["_load_index()"]
        INGEST["load_nodes_from_json()"]
        VEC_IDX["VectorStoreIndex"]
        SETUP_QE["_setup_query_engine()"]

        subgraph Retrievers["Retrieval"]
            VEC_RET["Vector Retriever (top 50)"]
            BM25["BM25 Retriever (top 50)"]
            FUSION["QueryFusionRetriever, Reciprocal Rerank"]
        end

        RERANKER["FlagEmbeddingReranker, bge-reranker-v2-m3"]

        subgraph Agents["LLM Agents"]
            AGENT1["Agent 1 — RAG Synthesis"]
            AGENT2["Agent 2 — Persona Refinement + Profiles"]
            SUMMARIZER["Agent — generate_summary()"]
            AGENT3["Agent 3 — Social Chronicler"]
        end
    end

    subgraph Ollama["Ollama, localhost:11434"]
        EMBED_MODEL["bge-m3, embedding model"]
        LLM_MODEL["qwen3:8b, LLM"]
    end

    %% Startup Flow
    ON_READY --> LOAD_IDX
    LOAD_IDX --> INGEST
    INGEST --> JSON_FILES
    INGEST --> VEC_IDX
    VEC_IDX --> SAVED_IDX

    %% Message Flow & Sync Path
    D_MSG --> ON_MSG
    ON_MSG --> HISTORY
    ON_MSG --> SUMMARY_LOAD
    SUMMARY_LOAD -.-> SUM_DB
    
    ON_MSG --> OP_FIND
    OP_FIND <--> OP_DB
    
    ON_MSG --> SETUP_QE
    SETUP_QE --> VEC_RET & BM25
    VEC_RET & BM25 --> FUSION
    FUSION --> RERANKER
    RERANKER --> AGENT1
    
    AGENT1 --> AGENT2
    OP_FIND -->|Pass profiles & targets| AGENT2
    HISTORY -.-> AGENT2
    SUMMARY_LOAD -.-> AGENT2
    
    AGENT2 --> D_REPLY
    
    %% Background Work (Async)
    AGENT2 -.-> SUMMARIZER
    SUMMARIZER --> SUMMARY_SAVE
    SUMMARY_SAVE -.-> SUM_DB
    
    AGENT2 -.-> AGENT3
    AGENT1 -.-> AGENT3
    AGENT3 -->|Update Tone/History| OP_DB

    %% Export Flow
    D_EXPORT --> EXPORT_FN
    EXPORT_FN --> MERGE
    MERGE --> JSON_FILES

    %% LLM & Embeddings Connections
    AGENT1 & AGENT2 & SUMMARIZER & AGENT3 --> LLM_MODEL
    VEC_RET & VEC_IDX --> EMBED_MODEL
```

## 2. Detailed Query Sequence

A closer look at the step-by-step query execution, including the parallel processing and agent pipeline.

```mermaid
sequenceDiagram
    actor User
    participant Bot as main.py
    participant Opinions as OpinionManager
    participant RAG as RAGAssistant
    participant Fusion as QueryFusionRetriever
    participant Reranker as FlagEmbeddingReranker
    participant Ollama

    User->>Bot: @Bot question
    Bot->>Bot: Fetch last 50 messages (history)
    Bot->>Bot: Load channel summary
    Bot->>Opinions: get_user_profile() & find_targets()
    Opinions-->>Bot: User Profile & Target Profiles
    
    Bot->>RAG: aquery(question)
    RAG->>Fusion: aretrieve(question)
    par Parallel Retrieval
        Fusion->>Ollama: embed(question) bge-m3
        Ollama-->>Fusion: vector
        Fusion->>Fusion: BM25 search (Local CPU)
    end
    Fusion->>Fusion: RRF Re-rank (Top 50)
    Fusion-->>RAG: 50 nodes
    RAG->>Reranker: postprocess_nodes(50 nodes)
    Reranker-->>RAG: top 10 nodes
    RAG->>Ollama: asynthesize() qwen3:8b
    Ollama-->>RAG: RAG response (Agent 1)
    RAG-->>Bot: rag_response
    
    Bot->>RAG: generate_refined_response(...)
    RAG->>Ollama: acomplete(persona prompt) qwen3:8b
    Ollama-->>RAG: Final Response (Agent 2)
    RAG-->>Bot: final_response
    Bot->>User: reply(final_response)
    
    par Async Background Tasks
        Bot->>RAG: generate_summary() (if >=6 msgs)
        RAG->>Ollama: acomplete(summary prompt) qwen3:8b
        Ollama-->>RAG: new summary
        Bot->>Bot: Save to cache/summaries.json
    and
        Bot->>RAG: evaluate_interaction()
        RAG->>Ollama: acomplete(agent 3 prompt) qwen3:8b
        Ollama-->>RAG: parsed JSON (stance, history)
        Bot->>Opinions: update_user_opinion()
        Opinions->>Opinions: Save to cache/opinions.json
    end
```