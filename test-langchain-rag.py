# pip install -qU langchain "langchain[anthropic]"
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
    temperature=0.1,
)

time = datetime.now()

formatted_time = f"{time.day} {time:%B %Y; %H:%M}"

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query, k=5) 
    
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    print("\n<----------------->\n".join(doc.page_content for doc in retrieved_docs))
    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        "Time: " + formatted_time + "\n"
        f"\n\n{docs_content}"
    )

    return system_message


agent = create_agent(model, tools=[], middleware=[prompt_with_context])

query = "Кто такой Barmacar?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()