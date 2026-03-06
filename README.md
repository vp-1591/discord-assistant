# Discord RAG Assistant 🤖📜

A powerful Discord bot that uses **Retrieval-Augmented Generation (RAG)** to answer questions based on your server's chat history. It features a unique "Head of Archive" persona, hybrid search, and rolling conversation summaries.

## ✨ Features

-   **Hybrid Search**: Combines semantic vector search (`bge-m3`) with keyword-based retrieval (BM25). Merges 100 deep-candidates and narrows down to the top 50 via Reciprocal Rank Fusion.
-   **Reranking**: Uses `BAAI/bge-reranker-v2-m3` to semantically rank the top 50 results down to the most relevant 10.
-   **Conversational Memory**: Automatically maintains a rolling summary of recent channel interactions to stay within context.
-   **Identity Resolution**: Intelligently resolves Discord user and role mentions to their display names, even for users who have since left the server.
-   **Persona-Driven**: Responds as the "Head of Archive" (Глава Архива) with a wise, archaic tone in Russian.
-   **Easy Data Ingestion**: Simple `!export` command to crawl channel history and prepare it for indexing.

## 🛠️ Tech Stack

-   **Discord.py**: Bot framework.
-   **LlamaIndex**: RAG orchestration and data indexing.
-   **Ollama**: Local hosting for LLM (`mistral`) and Embedding models (`bge-m3`).
-   **BM25 Retreiver**: For robust keyword matching in Russian.

## 🚀 Getting Started

### 1. Prerequisites

-   **Python 3.10+**
-   **Ollama** installed and running.
-   Pull the required models:
    ```bash
    ollama pull mistral
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

Adjust configuration in `run_llama_index.py` if your Ollama instance is not at `localhost:11434`.

### 4. Usage

#### First Run & Data Export
1.  Start the bot: `python main.py`
2.  In Discord, use the command `!export` in the channel you want the bot to "learn" from. (Note: Only configured admin IDs can currently trigger this).
3.  The bot will save the history to `messages_json/`.
4.  Restart the bot or set `FORCE_REBUILD = True` in `run_llama_index.py` to index the new data.

#### Interacting
Simply mention the bot in any channel it has access to:
`@ArchiveHead Что мы обсуждали на прошлой неделе?`

## 📁 Project Structure

-   `main.py`: Main bot logic and Discord event handling.
-   `run_llama_index.py`: RAG Assistant implementation (indexing, retrieval, synthesis).
-   `export_chat.py`: Utility module for exporting Discord history to JSON.
-   `messages_json/`: Storage for exported chat files.
-   `llama_index_storage/`: Vector database and index persistence.
-   `cache/`: Local logs and conversation summaries.