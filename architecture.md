# Discord Assistant — Architecture

## 1. Unified System Overview & Data Flow

This chart represents the modular architecture, encompassing configuration, data ingestion, retrieval logic, and the messaging pipeline.

```mermaid
flowchart TD
    subgraph Discord["Discord API"]
        D_MSG["User Message (bot mention)"]
        D_EXPORT["Admin !export command"]
        D_REPLY["Bot Reply"]
    end

    subgraph App["Core Application"]
        MAIN["main.py (aclient)"]
        HIST["src/data/history_manager.py"]
        CONFIG["src/config/config.py"]
        PROMPTS["src/config/prompts.py"]
    end

    subgraph Logic["Business Logic"]
        RAG_A["src/core/run_llama_index.py (RAGAssistant)"]
        AGENT_C["src/core/agent_core.py (ReActAgentWorkflow)"]
        INGEST["src/data/ingestion.py"]
        OP_M["src/data/opinion_manager.py"]
    end

    subgraph Storage["Storage & Cache"]
        JSON_FILES["messages_json/channel.json"]
        SAVED_IDX["llama_index_storage/"]
        OP_DB["cache/opinions.json"]
        SUM_DB["cache/summaries.json"]
        HIST_DB["cache/history.json"]
    end

    subgraph Ollama["Ollama (localhost:11434)"]
        EMBED_MODEL["bge-m3 (Embedding)"]
        LLM_MODEL["qwen3:8b (LLM)"]
    end

    %% Startup Flow
    MAIN --> CONFIG
    MAIN --> HIST
    MAIN --> RAG_A
    RAG_A --> INGEST
    INGEST --> SAVED_IDX
    INGEST --> JSON_FILES

    %% Message Flow
    D_MSG --> MAIN
    MAIN --> HIST
    HIST <--> HIST_DB
    MAIN --> OP_M
    OP_M <--> OP_DB
    
    MAIN --> RAG_A
    RAG_A --> PROMPTS
    RAG_A --> AGENT_C
    AGENT_C --> LLM_MODEL
    
    RAG_A --> D_REPLY
    
    %% Background Work
    RAG_A -.-> HIST
    HIST -.-> SUM_DB

    %% Export Flow
    D_EXPORT --> EXPORT_FN["src/data/export_chat.py"]
    EXPORT_FN --> JSON_FILES
```

## 2. Component Responsibilities

- **main.py**: Entry point, Discord client, event handling, and high-level orchestration.
- **src/config/config.py**: Centralized configuration, LlamaIndex settings, and file paths.
- **src/config/prompts.py**: Long prompt strings, system templates, and persona definitions.
- **src/data/ingestion.py**: Data cleaning, mention resolution, and LlamaIndex vector store management.
- **src/data/history_manager.py**: Logic for loading, saving, and truncating channel-specific history and summaries.
- **src/core/agent_core.py**: The custom ReAct agent workflow implementation.
- **src/core/run_llama_index.py**: Orchestrates RAG (Agent 1) and Refined Response (Agent 2) synthesis.
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
        Agent->>Agent: Execute Tool (Archive/Opinion)
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