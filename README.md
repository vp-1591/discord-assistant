# Discord RAG Assistant 🤖📜

A powerful Discord bot that uses **Retrieval-Augmented Generation (RAG)** to answer questions based on your server's chat history. It features a unique "Head of Archive" persona, hybrid search, rolling conversation summaries, and a dynamic **Social Memory System**.

## ✨ Features

-   **Hybrid Search**: Combines semantic vector search (`bge-m3`) with keyword-based retrieval (BM25). Merges 100 deep-candidates and narrows down to the top 50 via Reciprocal Rank Fusion.
-   **Reranking**: Uses `BAAI/bge-reranker-v2-m3` to semantically rank the top 50 results down to the most relevant 10.
-   **Conversational Memory**: Automatically maintains a rolling summary of recent channel interactions to stay within context.
-   **Identity Resolution**: Intelligently resolves Discord user and role mentions to their display names, even for users who have since left the server.
-   **Persona-Driven**: Responds as the "Head of Archive" (Глава Архива) with a wise, archaic tone in Russian (easily customizable in `src/config/prompts.py`).
-   **Social Memory (Opinion System)**: Analyzes interactions asynchronously to form and recall "opinions" and stances on users, tailoring future responses based on past interactions.
-   **RAG Cache**: Implements an LRU cache system to store and quickly recall recent search queries, minimizing repetitive database retrievals and improving conversational continuity.
-   **Live Data Ingestion & Progress Tracking**: Simple `!export` command to crawl channel history that automatically injects new messages into the vector index with progress bars in the console.
-   **Sequential Message Processing**: Implements an asynchronous worker queue to process requests one-by-one, ensuring LLM stability and preventing local VRAM spikes.

## 🛠️ Tech Stack

-   **Discord.py**: Bot framework.
-   **LlamaIndex**: RAG orchestration and data indexing.
-   **Ollama**: Local hosting for LLM (`qwen3:8b`) and Embedding models (`bge-m3`).
-   **BM25 Retreiver**: For robust keyword matching.

## 🚀 Getting Started

### 1. Prerequisites

-   **Python 3.10+**
-   **Ollama** installed and running.
-   Pull the required models:
    ```bash
    ollama pull qwen3:8b
    ollama pull bge-m3
    ```
-   A Discord Bot Token (with Message Content and Server Members intents enabled).

### 2. Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the root directory:

```env
TOKEN=your_discord_bot_token_here
```

Adjust settings in `config.py` (e.g., LlamaIndex models, file paths, or Ollama URL).

### 4. Usage

#### First Run & Data Export
1.  Start the bot: `python main.py` or use `start.bat`
2.  In Discord, use the command `!export` in the channel you want the bot to "learn" from. (Note: Only IDs listed in `ADMIN_IDS` within `src/config/config.py` can trigger this).
3.  The bot will save the history to `messages_json/`.
4.  The bot will automatically detect new messages and incrementally live-update the vector database (showing progress bars in the console).

#### Interacting
Simply mention the bot in any channel it has access to:
`@BotName Что мы обсуждали на прошлой неделе?`

## 📁 Project Structure

-   `main.py`: Entry point and Discord event handling.
-   `src/core/run_llama_index.py`: RAG Assistant orchestrator.
-   `src/config/config.py`: Centralized LlamaIndex and file configuration.
-   `src/config/prompts.py`: System prompts and persona templates.
-   `src/data/ingestion.py`: Data cleaning and LlamaIndex ingestion logic.
-   `src/core/agent_core.py`: Custom ReAct agent workflow.
-   `src/data/history_manager.py`: Manages rolling channel history and summaries.
-   `src/data/opinion_manager.py`: Fuzzy-matching logic for user stances.
-   `src/data/export_chat.py`: Utility for exporting Discord history.
-   `src/core/rag_cache.py`: Provides LRU cache for recent queries to reduce LLM overhead.
-   `messages_json/`: Exported chat JSON files.
-   `llama_index_storage/`: Vector database persistence.
-   `cache/`: Local summaries, history, user opinions, and RAG cache.