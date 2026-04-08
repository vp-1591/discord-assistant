## Goal
This project's goal is to create a discord bot that can answer any questions about server history, interpreting history as absolute truth for entertainment purposes.
## Architecture
Diagram of the bot's architecture. Starting point is main.py:
```mermaid
flowchart TD
    %% Define Styling
    classDef orchestrator fill:#2B2D31,stroke:#00E5FF,stroke-width:2px,color:#fff;
    classDef processing fill:#1E1F22,stroke:#43B581,stroke-width:2px,color:#fff;
    classDef storage fill:#202225,stroke:#FAA61A,stroke-width:2px,color:#fff;

    %% 1. ReAct Core (RAG)
    subgraph CognitiveCore ["1. ReAct Agent Orchestration"]
        direction TB
        ApplyingRollingSummary("Applying Rolling Session History"):::processing
        OrchestratingThinking("Agent2: ReAct Persona Loop"):::orchestrator
        QueryingUserStance("Querying User Stance"):::processing
        CheckingRAGCache("Checking RAG Cache"):::processing
        ExecutingToolSearch("Executing Search Tool"):::processing
        RetrievingSummaries("Retrieving Hybrid Summaries (Vector + BM25)"):::processing
        RerankingVectors("Reranking Top 50 to Top 10"):::processing
        SynthesizingContext("Agent1: Synthesizing RAG Report"):::processing
    end

    %% 2. Data Ingestion
    subgraph IngestionPipeline ["2. Background Data Ingestion"]
        direction TB
        ExportingHistory("Exporting Channel History"):::processing
        FilteringDuplicates("Filtering Processed Message IDs"):::processing
        ChunkingByDate("Chunking Messages By Date & Channel"):::processing
        GeneratingSummaries("Sequential Summary Generation"):::processing
        UpdatingVectorIndex("Updating Vector Storage"):::processing
        UpdatingSQLRecords("Updating SQLite Raw Records"):::processing
    end

    %% 3. Storage State
    subgraph StorageLayer ["3. Abstract Persistence Layer"]
        direction LR
        VectorDB[("Vector Storage")]:::storage
        SQLiteDB[("SQLite Database")]:::storage
        JSONLedgers[("Local JSON Caches")]:::storage
    end

    %% --- Logic Flow Connections ---

    %% ReAct Retrieval Flow
    DiscordMention("Discord @Mention") --> ApplyingRollingSummary
    ApplyingRollingSummary --> OrchestratingThinking
    
    OrchestratingThinking --> QueryingUserStance
    QueryingUserStance <-->|"Fetches Opinions"| JSONLedgers
    
    OrchestratingThinking -->|"Agent Tool Call"| CheckingRAGCache
    CheckingRAGCache <-->|"Hit/Miss"| JSONLedgers
    CheckingRAGCache -->|"Cache Miss"| ExecutingToolSearch
    
    ExecutingToolSearch --> RetrievingSummaries
    RetrievingSummaries --> RerankingVectors
    RerankingVectors --> SynthesizingContext
    
    RetrievingSummaries <--> VectorDB
    
    UpdatingVectorIndex <-->|"Stored Chunk ID Link"| UpdatingSQLRecords
    VectorDB <-->|"Chunk ID Foreign Key"| SQLiteDB

    %% Ingestion Flow
    ExportingHistory --> FilteringDuplicates
    FilteringDuplicates <-->|"Checks Processed List"| JSONLedgers
    FilteringDuplicates --> ChunkingByDate
    ChunkingByDate --> GeneratingSummaries
    
    GeneratingSummaries --> UpdatingVectorIndex
    GeneratingSummaries --> UpdatingSQLRecords
    
    UpdatingVectorIndex --> VectorDB
    UpdatingSQLRecords --> SQLiteDB
```
## WHYS
We use sequential requests to Ollama in both summarization and user messages responses to avoid OOM errors.
We use FIFO queue for message processing in main.py to avoid lost update problem. 
Agent2 is alias for persona ReAct agent that conducts conversation with user and uses tools to retrieve information from Agent1 and manage opinions.
Agent1 is alias for RAG agent that is responsible for using retrieved information from the database and writing report for Agent2.(Current)
We use qwen3:8b model for local inference with russian language proficiency. 
## Rules
Whenever you add any data storage, add it to gitignore.
When writing or modifying prompt templates in src/config/prompts.py, YOU MUST activate the prompt-engineering skill to write prompts with respect to qwen3:8b size.
Every update to the Vector Index MUST include source_chunk_id in metadata to maintain the foreign-key link to SQLite
Whenever you add any new functionality, make sure it is compliant with the SoC architecture.
Do not ask access to logs/ folder. If you need to check logs, ask me to do it.
## Dev Diary Protocol
To ensure continuous contextual alignment and maintain a precise technical record, you must curate a `DEV_DIARY.md` after every significant task or session(append to the end of the file):
* **Content**: Each entry must contain datetime and exactly two sentences:
    - **Datetime**: (e.g. 07.04.2026 23:07)
    - **The What**: A technical summary of the changes.
    - **The Why**: The architectural or performance reasoning.
* **Constraint**: Keep entries brief and devoid of fluff.