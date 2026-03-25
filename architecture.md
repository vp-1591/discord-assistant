# Discord Assistant — Architecture

## 1. Unified System Overview & Data Flow

This chart represents the modular architecture, encompassing configuration, data ingestion, retrieval logic, and the messaging pipeline.
```mermaid
flowchart TD
    %% Define Styling
    classDef discord fill:#5865F2,stroke:#fff,stroke-width:2px,color:#fff;
    classDef app fill:#2B2D31,stroke:#00E5FF,stroke-width:2px,color:#fff;
    classDef logic fill:#1E1F22,stroke:#43B581,stroke-width:2px,color:#fff;
    classDef storage fill:#202225,stroke:#FAA61A,stroke-width:2px,color:#fff;
    classDef ai fill:#2f3136,stroke:#F04747,stroke-width:2px,color:#fff;

    %% 1. Interface
    subgraph Discord ["1. Discord Interface"]
        direction LR
        D_EXPORT("Admin !export"):::discord
        D_MSG("User @Mention"):::discord
        D_REPLY("Bot Reply"):::discord
    end

    %% 2. Entry & Routing
    subgraph Routing ["2. Orchestration"]
        MAIN{"Message Router<br/>(main.py)"}:::app
    end

    %% 3. Main Logic Trees
    subgraph Engine ["3. Core Functional Engine"]
        
        subgraph Pipeline ["Data Ingestion Pipeline"]
            direction TB
            EXPORT_FN["Chat Exporter<br/>(export_chat.py)"]:::logic
            INGEST["Data Vectorizer<br/>(ingestion.py)"]:::logic
        end

        subgraph Cognition ["Cognitive AI"]
            direction TB
            RAG_A{"Semantic RAG<br/>(run_llama_index.py)"}:::logic
            AGENT_C("ReAct Agent<br/>(agent_core.py)"):::logic
        end

        subgraph Context ["Context Managers"]
            direction TB
            HIST["Chat History<br/>(history_manager.py)"]:::logic
            OP_M["Social Opinions<br/>(opinion_manager.py)"]:::logic
            RAG_C["Search Cache<br/>(rag_cache.py)"]:::logic
        end
    end

    %% 4. Data Layer
    subgraph Storage ["4. Storage & State"]
        direction LR
        SAVED_IDX[("Vector Index<br/>(llama_index/)")]:::storage
        JSON_FILES[("Raw Chats<br/>(messages_json/)")]:::storage
        
        OP_DB[("opinions.json")]:::storage
        SUM_DB[("summaries.json")]:::storage
        HIST_DB[("history.json")]:::storage
        RAG_CACHE_DB[("rag_cache.json")]:::storage
    end

    %% 5. Services
    subgraph Services ["5. External Models"]
        OLLAMA[["Local Ollama<br/>(qwen3:8b / bge-m3)"]]:::ai
    end

    %% --- Connection Flow ---

    D_MSG ---> MAIN
    MAIN <--> Context
    MAIN ---> RAG_A
    RAG_A ---> D_REPLY

    D_EXPORT ---> EXPORT_FN
    EXPORT_FN --->|Writes| JSON_FILES
    INGEST -.->|Reads| JSON_FILES
    INGEST --->|Updates| SAVED_IDX

    RAG_A <--> RAG_C
    RAG_A ---> AGENT_C
    RAG_A -.->|"Triggers live update"| INGEST

    %% Context I/O
    HIST <--> HIST_DB
    HIST -.-> SUM_DB
    OP_M <--> OP_DB
    RAG_C <--> RAG_CACHE_DB

    %% Inference
    AGENT_C <==>|"Synthesis"| OLLAMA
    INGEST <==>|"Embeddings"| OLLAMA
```

## 2. Component Responsibilities

- **main.py**: Entry point, Discord client, event handling, and high-level orchestration.
- **src/config/config.py**: Centralized configuration, LlamaIndex settings, and file paths.
- **src/config/prompts.py**: Long prompt strings, system templates, and persona definitions.
- **src/data/ingestion.py**: Data cleaning, mention resolution, and LlamaIndex vector store management.
- **src/data/history_manager.py**: Logic for loading, saving, and truncating channel-specific history and summaries.
- **src/core/agent_core.py**: The custom ReAct agent workflow implementation.
- **src/core/run_llama_index.py**: Orchestrates RAG (Agent 1) and Refined Response (Agent 2) synthesis.
- **src/core/rag_cache.py**: Implements the LRU cache (RAGCache) to store and retrieve recent search results, improving efficiency for follow-up questions.
- **src/data/opinion_manager.py**: Manages long-term user profiles and stances.

## 3. Sequence Diagram (Messaging)

```mermaid
sequenceDiagram
    actor User
    participant Main as main.py
    participant Hist as src/data/history_manager.py
    participant RAG as src/core/run_llama_index.py
    participant Agent as src/core/agent_core.py
    participant Ollama

    User->>Main: @Bot message
    Main->>Hist: get_history()
    Hist-->>Main: history list
    Main->>Main: Fetch Opnions
    
    Main->>RAG: generate_refined_response()
    RAG->>Agent: workflow.run()
    
    loop ReAct Loop
        Agent->>Ollama: chat (Reasoning)
        Ollama-->>Agent: Thought/Action
        Agent->>Agent: Execute Tool (Archive/Opinion/Cache)
    end
    
    Agent-->>RAG: Final Answer
    RAG-->>Main: final_response
    Main->>User: reply
    
    Main->>Hist: add_to_history()
    Main->>RAG: generate_summary() (if needed)
    RAG->>Ollama: Summarize
    Ollama-->>RAG: Summary
    RAG->>Hist: update_summary()
```