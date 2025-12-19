# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
import create_store_rag
from datetime import datetime

@tool
def retrieve_context(search_query: str):
    """
    Search the chat logs/chronicles of the server.
    Input should be a specific search phrase or topic.
    If you are looking for a specific person, include their name in the search_query.
    """
    print(f"\n   ⚙️ [TOOL ACTIVE] Searching for: '{search_query}'")
    
    # We rely on semantic search to find the user mentions naturally
    results = create_store_rag.vector_store.similarity_search(
        search_query, 
        k=5
    )
    
    # Format results for the LLM
    serialized = "\n".join(
        [doc.page_content for doc in results]
    )
    return serialized

model = init_chat_model(
    "mistral:latest",
    model_provider="ollama",
    temperature=0.1,
)

time = datetime.now()

formatted_time = f"{time.day} {time:%B %Y; %H:%M}"

agent = create_agent(
   model=model,
   tools=[retrieve_context],
   system_prompt=f'''
   You are a specialized Retrieval Agent
   ''',
)

response = agent.invoke(
   {"messages": [{"role": "user", "content": "Кто такой Barmacar?"}]}
)

for msg in response["messages"]:
    # 1. Check if the AI decided to call a tool
    if msg.type == "ai" and msg.tool_calls:
        for tool_call in msg.tool_calls:
            print(f"\n➡️  AI Decided to Call Tool: '{tool_call['name']}'")
            print(f"    Parameters: {tool_call['args']}")
            # This answers: "what parameters does it receive?"

    # 2. Check the actual results from the tool
    elif msg.type == "tool":
        print(f"\n⬅️  Tool Returned Context:")
        # The 'content' is the string the LLM sees
        # We slice it to avoid flooding the console if it's huge
        preview = msg.content[:500].replace('\n', ' ') + "..."
        print(f"    Preview: {preview}")
        
        # If you want to see the raw documents (the artifact):
        if hasattr(msg, "artifact") and msg.artifact:
             print(f"    Raw Docs Retrieved: {len(msg.artifact)}")
             # print(msg.artifact) # Uncomment to see full raw objects

# --- FINAL RESPONSE ---
print("\n--- Final Answer ---")
print(response["messages"][-1].content)