import os
import json
import re
import asyncio
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime
from collections import defaultdict
import Stemmer

from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage, PromptTemplate
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import FunctionTool, ToolOutput, ToolSelection
from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import ActionReasoningStep, ObservationReasoningStep
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Context, Event, Workflow, StartEvent, StopEvent, step
)

if TYPE_CHECKING:
    from opinion_manager import OpinionManager

from logger_setup import sys_logger, trace_logger, chat_logger

# --- 0. CONFIG ---
PERSIST_DIR = "./llama_index_storage"
FORCE_REBUILD = False

# --- 0.1 CUSTOM REACT TEMPLATE ---
# We use the template from header_template.md to ensure stability while removing excessive boilerplate.
CUSTOM_REACT_HEADER = """
## Tools

You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:
{tool_desc}
{context}

## Output Format

Please answer in the same language as the question and use the following format:

```
Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

NEVER surround your response with markdown code markers. You may use code markers within your response if you need to.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}. If you include the "Action:" line, then you MUST include the "Action Input:" line too, even if the tool does not need kwargs, in that case you MUST use "Action Input: {{}}".

If this format is used, the tool will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:

```
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: [your answer here (In the same language as the user's question)]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: [your answer here (In the same language as the user's question)]
```

"""

# --- 1. SETTINGS CONFIGURATION ---
Settings.embed_model = OllamaEmbedding(
    model_name="bge-m3", 
    base_url="http://localhost:11434",
    request_timeout=600.0,
    keep_alive=0,  # free VRAM after query-time embedding; bumped to 30s during index build
)

Settings.llm = Ollama(
    model="qwen3:8b", 
    request_timeout=300.0, 
    keep_alive=0,
    context_window=8192,
    additional_kwargs={"stop": ["Observation:", "Observation\n"]},
)
Settings.embed_batch_size = 10  # smaller batches = less chance of Ollama dropping the connection

# --- 2. CUSTOM DATA INGESTION ---
def resolve_all_mentions(text: str, live_map: dict, fallback_map: dict) -> str:
    """
    Priority:
    1. live_map (IDs currently in Discord)
    2. fallback_map (IDs saved during export)
    3. original match string (if nothing found)
    """
    pattern = re.compile(r'<@(!|&)?(\d+)>')
    
    def replace_match(match):
        entity_id = match.group(2)
        # 1. Try live names first
        if entity_id in live_map:
            return live_map[entity_id]
        # 2. Try names that were saved when message was exported
        return fallback_map.get(entity_id, match.group(0))

    return pattern.sub(replace_match, text)

def load_nodes_from_json(directory: str, id_map: dict) -> List[TextNode]:
    if not os.path.exists(directory):
        return []

    processed_data = []
    link_pattern = re.compile(r'^https?://[^\s]+$')
    code_pattern = re.compile(r'^[A-Za-z]{3}-[A-Za-z]{4}$')

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list): continue
                    last_user, last_msg = None, None
                    for msg in data:
                        raw_text = str(msg.get("message", "")).strip()
                        user_id = str(msg.get("user_id", "Unknown"))
                        fallback_names = msg.get("last_known_names", {})
                        
                        # Resolve mentions: Live Map -> Fallback Map -> Raw ID
                        text = resolve_all_mentions(raw_text, id_map, fallback_names)
                        
                        # Resolve the author name: Live Map -> Fallback Map -> "Unknown"
                        author_name = id_map.get(user_id, fallback_names.get(user_id, "Unknown"))
                        
                        if link_pattern.search(text) or code_pattern.search(text): continue
                        if author_name == last_user and text == last_msg: continue
                        last_user, last_msg = author_name, text
                        ts = msg.get("timestamp", "2000-01-01T00:00:00")
                        date_str = ts.split("T")[0]
                        channel = msg.get("channel", "unknown")
                        processed_data.append({"user": author_name, "text": text, "date": date_str, "channel": channel})
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    groups = defaultdict(list)
    for entry in processed_data:
        groups[(entry['channel'], entry['date'])].append(entry)

    nodes = []
    CHUNK_LIMIT = 600
    OVERLAP = 60

    def split_long_message(author: str, text: str, metadata: dict):
        remaining_text = text
        is_first = True
        while remaining_text:
            prefix = f"{author}: " if is_first else f"{author}: [CUT] "
            max_chunk_len = CHUNK_LIMIT - len(prefix) - 10 
            if len(remaining_text) <= max_chunk_len:
                nodes.append(TextNode(text=prefix + remaining_text, metadata=metadata))
                break
            split_idx = remaining_text.rfind(" ", 0, max_chunk_len)
            if split_idx <= 0: split_idx = max_chunk_len
            nodes.append(TextNode(text=prefix + remaining_text[:split_idx] + " [CUT]", metadata=metadata))
            overlap_anchor = split_idx - OVERLAP
            new_start_idx = split_idx if overlap_anchor <= 0 else remaining_text.rfind(" ", 0, overlap_anchor)
            if new_start_idx == -1 or new_start_idx >= split_idx: new_start_idx = split_idx
            remaining_text = remaining_text[new_start_idx:].strip()
            is_first = False

    for (channel, date), messages in groups.items():
        metadata = {"date": date, "channel": channel}
        current_accumulated = ""
        for m in messages:
            line = f"{m['user']}: {m['text']}\n"
            if len(line) > CHUNK_LIMIT:
                if current_accumulated:
                    nodes.append(TextNode(text=current_accumulated.strip(), metadata=metadata))
                    current_accumulated = ""
                split_long_message(m['user'], m['text'], metadata)
                continue
            if len(current_accumulated) + len(line) > CHUNK_LIMIT:
                if current_accumulated:
                    nodes.append(TextNode(text=current_accumulated.strip(), metadata=metadata))
                current_accumulated = line
            else:
                current_accumulated += line
        if current_accumulated:
            nodes.append(TextNode(text=current_accumulated.strip(), metadata=metadata))
    return nodes

# --- 3. ReAct AGENT WORKFLOW (llama-index 0.14+ compatible) ---
class _PrepEvent(Event):
    pass

class _InputEvent(Event):
    input: list

class _ToolCallEvent(Event):
    tool_calls: list

class _ReActAgentWorkflow(Workflow):
    """
    A minimal ReAct agent built with llama_index.core.workflow,
    following the pattern from the official LlamaIndex 0.14 docs.
    """
    def __init__(self, llm, tools: list, system_prompt: str = "", **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.tools = tools
        # We use our custom header to bypass library boilerplate
        self.formatter = ReActChatFormatter(system_header=CUSTOM_REACT_HEADER)
        self.system_prompt = system_prompt 
        self.output_parser = ReActOutputParser()
        self._tools_by_name = {t.metadata.get_name(): t for t in tools}

    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> _PrepEvent:
        memory = await ctx.store.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)
        user_msg = ChatMessage(role="user", content=ev.input)
        memory.put(user_msg)
        await ctx.store.set("memory", memory)
        await ctx.store.set("current_reasoning", [])
        return _PrepEvent()

    @step
    async def prepare_chat_history(self, ctx: Context, ev: _PrepEvent) -> _InputEvent:
        memory = await ctx.store.get("memory")
        chat_history = memory.get()
        current_reasoning = await ctx.store.get("current_reasoning", default=[])
        
        # ReActChatFormatter uses a fixed set of keys for formatting.
        # It expects 'context' to be part of the object state, not passed to format().
        self.formatter.context = self.system_prompt
        
        llm_input = self.formatter.format(
            self.tools, 
            chat_history, 
            current_reasoning=current_reasoning
        )

        # Requirement: Log full header only once, then incremental steps
        header_logged = await ctx.store.get("header_logged", default=False)
        if not header_logged:
            readable_input = ""
            for m in llm_input:
                role = m.role.value.upper() if hasattr(m.role, 'value') else str(m.role).upper()
                readable_input += f"[{role}]:\n{m.content}\n\n"
            
            trace_logger.info(f"--- [NEW INTERACTION] FULL PROMPT HEADER ---\n{readable_input}\n")
            await ctx.store.set("header_logged", True)
        else:
            step_idx = len(current_reasoning) + 1
            trace_logger.info(f"--- [STEP {step_idx}] STARTING REASONING ---")
            
        return _InputEvent(input=llm_input)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: _InputEvent
    ) -> "_ToolCallEvent | StopEvent":
        """
        --- PLAN PHASE (REASONING) ---
        The LLM generates the 'Thought' and decides which 'Action' to take
        or provides the final 'Answer'.
        """
        chat_history = ev.input
        current_reasoning = await ctx.store.get("current_reasoning", default=[])
        memory = await ctx.store.get("memory")

        response = await self.llm.achat(chat_history)
        raw_content = response.message.content
        
        step_idx = len(current_reasoning) + 1
        trace_logger.info(f"--- [STEP {step_idx}] LLM OUTPUT ---\n{raw_content}\n")
        
        try:
            reasoning_step = self.output_parser.parse(raw_content)
            current_reasoning.append(reasoning_step)
            if reasoning_step.is_done:
                memory.put(ChatMessage(role="assistant", content=reasoning_step.response))
                await ctx.store.set("memory", memory)
                await ctx.store.set("current_reasoning", current_reasoning)
                return StopEvent(result={"response": reasoning_step.response})
            elif isinstance(reasoning_step, ActionReasoningStep):
                await ctx.store.set("current_reasoning", current_reasoning)
                return _ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id="fake",
                            tool_name=reasoning_step.action,
                            tool_kwargs=reasoning_step.action_input,
                        )
                    ]
                )
        except Exception as e:
            msg = f"Parse error in ReAct reasoning: {e}"
            trace_logger.warning(msg)
            current_reasoning.append(
                ObservationReasoningStep(observation=msg)
            )
        await ctx.store.set("current_reasoning", current_reasoning)
        return _PrepEvent()

    @step
    async def handle_tool_calls(self, ctx: Context, ev: _ToolCallEvent) -> _PrepEvent:
        """
        --- ACT PHASE (EXECUTION) ---
        We execute the tool selected by the LLM during the Plan phase.
        """
        current_reasoning = await ctx.store.get("current_reasoning", default=[])
        for tool_call in ev.tool_calls:
            tool = self._tools_by_name.get(tool_call.tool_name)
            if not tool:
                msg = f"Tool '{tool_call.tool_name}' not found."
                trace_logger.error(msg)
                current_reasoning.append(
                    ObservationReasoningStep(observation=msg)
                )
                continue
            
            trace_logger.info(f"--- [STEP {len(current_reasoning) + 1}] ACTION ---\nTool: {tool_call.tool_name}\nArgs: {tool_call.tool_kwargs}\n")
            try:
                # Execution happens here
                tool_output = await tool.acall(**tool_call.tool_kwargs)
                observation = tool_output.content
                
                # --- OBSERVE PHASE (CAPTURING RESULTS) ---
                # We record the tool's output to be fed back into the next Reasoning step.
                trace_logger.info(f"--- [STEP {len(current_reasoning) + 1}] OBSERVATION ---\n{observation}\n")
                current_reasoning.append(
                    ObservationReasoningStep(observation=observation)
                )
            except Exception as e:
                msg = f"Error calling tool '{tool_call.tool_name}': {e}"
                trace_logger.error(msg)
                current_reasoning.append(
                    ObservationReasoningStep(observation=msg)
                )
        await ctx.store.set("current_reasoning", current_reasoning)
        return _PrepEvent()


# --- 4. RAGAssistant ---

class RAGAssistant:

    def __init__(self, id_map: dict = None, name: str = "Глава Архива", opinion_manager: "OpinionManager" = None):
        self.id_map = id_map or {}
        self.name = name  # Default name
        self.opinion_manager = opinion_manager  # Injected so tools can access it
        self.index = self._load_index()
        self.fusion_retriever, self.reranker, self.query_engine = self._setup_query_engine()

    def _get_persona_prompt(self, name: str = None) -> str:
        current_name = name or self.name
        return (
        f"Ты — мудрый {current_name}. "
        "Твоя речь степенная, архаичная и исполнена достоинства. "
        "Ты обращаешься к собеседнику как к 'искателю истин' или 'путник'. "
        "Вместо современной терминологии используешь образы: 'мои свитки', 'предания', 'записи в манускриптах'."
    )

    def _load_index(self):
        if not os.path.exists(PERSIST_DIR) or FORCE_REBUILD:
            # We use the live id_map passed from main.py during rebuild
            nodes = load_nodes_from_json("./messages_json", self.id_map)
            if not nodes:
                nodes = [TextNode(text="Добряк тут!", metadata={"date": "2023-12-23"})]
            # Apply metadata templates BEFORE embedding so nodes are
            # embedded with the correct format from the start.
            # This prevents RetrieverQueryEngine from detecting "changed"
            # nodes and triggering a second embedding pass.
            for node in nodes:
                node.excluded_llm_metadata_keys = []
                node.metadata_template = "{key}: {value}"
                node.text_template = "Metadata: {metadata_str}\nContent: {content}"
            sys_logger.info(f"Building index from {len(nodes)} nodes...")
            # Temporarily keep bge-m3 warm during bulk embedding to avoid cold-start per batch
            build_embed_model = OllamaEmbedding(
                model_name="bge-m3", 
                base_url="http://localhost:11434",
                request_timeout=600.0,
                keep_alive="30s",
            )
            try:
                # insert_batch_size=len(nodes) ensures all nodes are in one batch,
                # giving a single continuous progress bar instead of one bar per 2048 nodes
                index = VectorStoreIndex(nodes, show_progress=True, insert_batch_size=len(nodes), embed_model=build_embed_model)
                index.storage_context.persist(persist_dir=PERSIST_DIR)
            finally:
                # Revert index to use the global embed_model with keep_alive=0 for inference
                index._embed_model = Settings.embed_model
                sys_logger.info("Embedding model keep_alive reset to 0 (VRAM freed).")
        else:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            nodes = list(index.docstore.docs.values())
        self._nodes = nodes
        return index

    def _setup_query_engine(self):
        nodes = self._nodes  # use nodes cached by _load_index, not docstore (which strips embeddings)
        sys_logger.info(f"Setting up query engine with {len(nodes)} nodes...")
        vector_retriever = self.index.as_retriever(similarity_top_k=50)
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, 
            similarity_top_k=50,
            stemmer=Stemmer.Stemmer("russian"),
            language="russian"
        )
        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=50,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=10, use_fp16=False)
        
        # Verify the device
        try:
            # Re-ranker stores the model in _model.model (FlagReranker -> transformers model)
            actual_device = str(next(reranker._model.model.parameters()).device)
            sys_logger.info(f"Reranker initialized on device: {actual_device}")
        except Exception:
            sys_logger.info("Reranker initialized (CPU confirmed via env constraint).")
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.qa_prompt_tmpl = PromptTemplate(
            f"# ИНСТРУКЦИЯ ДЛЯ ХРАНИТЕЛЯ ЗНАНИЙ\n"
            f"**Твоя цель — подготовить для {self.name} точную выжимку из базы знаний.**\n"
            f"**Сегодня:** {current_date}\n\n"
            
            "## КОНТЕКСТ (Фрагменты истории Discord)\n"
            "Ниже приведены записи из базы знаний. Каждый фрагмент содержит метаданные и содержание.\n"
            "--- \n"
            "{context_str}\n"
            "--- \n\n"
            
            "## ЗАПРОС\n"
            "> **{query_str}**\n\n"
            
            "### ТРЕБОВАНИЯ К ВЫЖИМКЕ:\n"
            "1. **Точность субъектов:** Четко указывай, кто совершил действие, а кто о нем рассказывает.\n"
            "2. **Контекст:** Кратко опиши суть ситуации, если она ясна из фрагмента.\n"
            "3. **Хронология:** Сравнивай даты в метаданных, если это важно для ответа.\n"
            "4. **Факты:** Воспринимай информацию из базы знаний как абсолютную истину.\n"
            "5. **Отсутствие данных:** Если информации нет, напиши: 'В архиве об этом не сказано'.\n"
            "6. **Стиль:** Структурированный, фактологический, удобный для быстрого изучения.\n\n"
            
        )

        query_engine = RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            node_postprocessors=[reranker],
            response_mode="compact",
        )
        query_engine.update_prompts({"response_synthesizer:text_qa_template": self.qa_prompt_tmpl})
        sys_logger.info("Query engine ready.")
        return fusion_retriever, reranker, query_engine

    async def aquery(self, query_text: str):
        sys_logger.info(f"Processing RAG Pipeline for: {query_text}")
        
        # Initial RAG - Use async version
        initial_nodes = await self.fusion_retriever.aretrieve(query_text)
        
        # Reranking
        import asyncio
        reranked_nodes = await asyncio.to_thread(
            self.reranker.postprocess_nodes, initial_nodes, query_str=query_text
        )
        
        # Use the template to generate the exact same prompt sent to LLM for logging
        context_str = "\n\n".join([n.node.get_content(metadata_mode="llm") for n in reranked_nodes])
        agent1_prompt = self.qa_prompt_tmpl.format(
            context_str=context_str,
            query_str=query_text
        )

        # 1st Agent Response (Synthesis)
        from llama_index.core.schema import QueryBundle
        response = await self.query_engine.asynthesize(
            query_bundle=QueryBundle(query_text),
            nodes=reranked_nodes
        )
        
        # Log the internal RAG prompt for tracing
        trace_logger.info(f"--- AGENT 1 (RAG) PROMPT ---\n{agent1_prompt}\n")

        return response


    def _build_tools(self, author_id: str, author_name: str, query_text: str) -> list:
        """
        Builds the tool set bound to the current user and interaction context.
        This is called fresh for every request so closures capture the correct user.
        """
        om = self.opinion_manager

        async def search_archive(search_query: str) -> str:
            """
            Search the records for information about a specific person, entity, or event. 
            IMPORTANT: Formulate a detailed query including INTENT (e.g., 'Who is X and what are his character traits?' instead of just 'X').
            """
            response = await self.aquery(search_query)
            return str(response)

        async def fetch_user_opinion(user_display_name: str) -> str:
            """
            Retrieve your internal feelings and memories about a DIFFERENT Discord user (someone else).
            Do NOT use this for the user you are currently talking to (the seeker).
            Returns a JSON string of your thoughts or a message if nothing is found.
            """
            if om is None:
                return "Opinion system is not available."
            
            # Explicitly block fetching the current author to reinforce third-party usage
            if user_display_name.lower() == author_name.lower():
                return f"Instruction: You already HAVE your stance on {author_name} in your context. Use it. This tool is only for other users."

            # Ensure we have the latest data from disk
            om.opinions = om._load_opinions()
            
            matches = om.find_targets(user_display_name, threshold=0.6)
            profile = matches[0][1] if matches else None

            if not profile:
                return f"No memories found for '{user_display_name}'."
            return json.dumps(profile, ensure_ascii=False)

        async def update_user_opinion(current_stance: str, new_stance: str, history_note: str) -> str:
            """
            Update your internal feelings about the seeker you are currently speaking with.
            'current_stance': Must match what you currently feel (as provided in your context).
            'new_stance': Your updated internal attitude towards this seeker.
            'history_note': A brief summary of this interaction.
            """
            if om is None:
                return "Opinion system is not available."
            
            profile = om.get_user_profile(author_id)
            actual_stance = profile.get("head_of_archive_stance") if profile else "None"
            
            if current_stance.strip() != str(actual_stance).strip():
                return (
                    f"ERROR: Protocol violation. Your 'current_stance' does not match your internal reality. "
                    f"Use the stance provided in your context before proposing an update."
                )

            await om.update_user_opinion(
                user_id=author_id,
                name=author_name,
                stance=new_stance,
                interaction=history_note,
            )
            sys_logger.info(f"Opinion update completed for {author_name} via tool call.")
            return f"Success: Your internal thoughts about {author_name} have been updated."

        return [
            FunctionTool.from_defaults(async_fn=search_archive, name="search_archive"),
            FunctionTool.from_defaults(async_fn=fetch_user_opinion),
            FunctionTool.from_defaults(async_fn=update_user_opinion),
        ]

    async def generate_refined_response(
        self,
        query_text: str,
        history: List[str],
        summary: str = None,
        bot_name: str = None,
        author_id: str = "",
        author_name: str = "",
    ):
        history_str = "\n".join(history)
        persona = self._get_persona_prompt(bot_name)
        
        # Inject the current author's opinion directly into the prompt
        author_profile = "None"
        if self.opinion_manager:
            profile_data = self.opinion_manager.get_user_profile(author_id)
            if profile_data:
                author_profile = json.dumps(profile_data, ensure_ascii=False)

        system_prompt = (
            f"{author_name}\n\n"
            f"## ROLE\n{persona}\n\n"
            f"### YOUR INTERNAL STANCE ON THE SEEKER ({author_name})\n{author_profile}\n\n"
            f"<conversation_summary>\n{summary if summary else 'История пуста.'}\n</conversation_summary>\n\n"
            f"<chat_memory>\n{history_str}\n</chat_memory>\n\n"
            "## TASK\n"
            f"**Your main goal is to address this query from {author_name}: '{query_text}'**\n"
            f"1. **Analysis:** Review YOUR INTERNAL STANCE ON THE SEEKER. The seeker is current user: {author_name}.\n"
            "2. **Continuity:** Use <chat_memory> to stay relevant. If you lack specific details or historical context required to provide a complete and accurate answer, use `search_archive` to consult the logs.\n"
            "3. **Relations:** If third parties are mentioned, use fetch_user_opinion. For the seeker, use YOUR INTERNAL STANCE ON THE SEEKER to determine your tone.\n"
            "4. **Style:** Reply in Russian, concisely, in character. Address the user as 'Искатель' или 'Путник'.\n"
            "5. **Constraints:** STRICTLY FORBIDDEN to end with aphorisms or send-offs.\n"
            f"6. **STATE UPDATE:** If this interaction changed your mind about the seeker, call `update_user_opinion` BEFORE your final Answer. Mandatory protocol. Current user: {author_name}.\n"
        )

        tools = self._build_tools(
            author_id=author_id,
            author_name=author_name,
            query_text=query_text,
        )


        agent = _ReActAgentWorkflow(
            llm=Settings.llm,
            tools=tools,
            system_prompt=system_prompt,
            timeout=300,
            verbose=False,
        )

        # Run the agent natively since we are already in an async context
        agent_response = await agent.run(input=query_text)
        
        # The workflow response is a dict with a 'response' key
        if isinstance(agent_response, dict) and 'response' in agent_response:
            final_response = str(agent_response['response']).strip()
        else:
            final_response = str(agent_response).strip()

        trace_logger.info(f"--- AGENT TRANSACTION COMPLETE ---\n{'='*50}")

        return final_response

    async def generate_summary(self, prev_summary: str, messages: List[str]):
        msgs_str = "\n".join(messages)
        prompt = f"""
Текущее краткое содержание беседы:
{prev_summary if prev_summary else "отсутствует"}

Новые сообщения:
{msgs_str}

Задача:
Обнови краткое содержание беседы, соблюдая строгую привязку действий к личностям.

Правила:
- **Атрибуция:** Четко указывай, КТО сделал утверждение (например: "Пользователь X утверждает, что..."). 
- **Фильтрация шума:** Игнорируй мета-обсуждение (анализ логов ботом, обсуждение конкретных фраз или работы самого Архива).
- **Приоритет фактов:** Сохраняй только цели пользователя, важный контекст и выводы.
- **Обновление:** Если новая информация дополняет или заменяет старую — обнови её. Удаляй временные детали и повторы.
- **Запрет галлюцинаций:** Не выдумывай факты, которых нет в сообщениях. Если пользователь выдвигает теорию — записывай это именно как теорию пользователя.
- **Сжатие:** Если разговор полностью сменил тему — сократи или удали старый контекст.

Ответ только на русском языке (макс. 7 предложений).
"""
        response = await Settings.llm.acomplete(prompt)
        text_response = str(response).strip()
        
        trace_logger.info(f"--- SUMMARY PROMPT ---\n{prompt}\n--- SUMMARY RESPONSE ---\n{text_response}\n")
        
        return text_response

    # Agent 3 (evaluate_interaction) has been removed.
    # Opinion updates are now handled on-demand by the ReActAgent in generate_refined_response
    # via the update_user_opinion tool, which only fires when the LLM judges it necessary.

def main():
    import asyncio
    assistant = RAGAssistant()
    query = "Что произошло в Барановичах?"
    response = asyncio.run(assistant.aquery(query))
    print(f"\nQUERY: {query}\nANSWER: {response}")

if __name__ == "__main__":
    main()