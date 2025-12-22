import socket
import subprocess
import time

def ensure_ollama_running():
    def is_running():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', 11434)) == 0

    if not is_running():
        print("Ollama is not running. Starting 'ollama serve'...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
            # Wait for it to start
            for i in range(10):
                if is_running():
                    print("Ollama started successfully.")
                    return
                time.sleep(1)
            print("Ollama is taking a while to start, continuing anyway...")
        except FileNotFoundError:
            print("Error: 'ollama' command not found. Please ensure Ollama is installed and in your PATH.")

ensure_ollama_running()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
import create_store_rag
from datetime import datetime
from langchain.agents.middleware import dynamic_prompt, ModelRequest

vector_store = create_store_rag.vector_store

model = init_chat_model(
    "mistral:latest",
    model_provider="ollama",
    temperature=0,
)

time = datetime.now()

formatted_time = f"{time.day} {time:%B %Y; %H:%M}"

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query, k=20) 
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i + 1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(retrieved_docs)
            ]
        )
    )
    docs_content = "\n\n".join(f"Date: {doc.metadata.get('date')}, Channel: {doc.metadata.get('channel')}\n{doc.page_content}" for doc in retrieved_docs)

    system_message = f"""
You are an intelligent Chat Log Analyst.
You will be provided with a set of chat logs in the "CONTEXT" block.

CONTEXT STRUCTURE:
- The context consists of "Date" and Channel headers (e.g., "Date: 2023-08-26, Channel: площаль").
- All messages following a Date header occurred on that specific date, until a new Date header appears.
- Messages are formatted as: [Time] <User>: Message.

CONTEXT:
{docs_content}

INSTRUCTIONS:
1. Analyze the user's question to identify if it requires temporal reasoning (e.g., "last", "first", "in 2025", "before").
2. Associate every message mentioned in the context with its corresponding Date header.
3. If the context contains conflicting information (e.g., multiple people called "Dobryak"), use the DATES to resolve the conflict based on the user's question (e.g., for "last", pick the most recent date; for "first", pick the oldest).
4. Answer ONLY using the provided context. If the answer is not in the context, say "I don't know".

Respond in Russian.
"""

    #print(system_message)
    
    return system_message


agent = create_agent(model, tools=[], middleware=[prompt_with_context])

query = "Кто последний Добряк?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()