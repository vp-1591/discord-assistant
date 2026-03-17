from llama_index.core.workflow import (
    Context, Event, Workflow, StartEvent, StopEvent, step
)
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import ActionReasoningStep, ObservationReasoningStep
from llama_index.core.tools import ToolSelection

from src.config.prompts import CUSTOM_REACT_HEADER
from src.utils.logger_setup import trace_logger

class _PrepEvent(Event):
    pass

class _InputEvent(Event):
    input: list

class _ToolCallEvent(Event):
    tool_calls: list

class ReActAgentWorkflow(Workflow):
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
        chat_history = ev.input
        current_reasoning = await ctx.store.get("current_reasoning", default=[])
        memory = await ctx.store.get("memory")

        response = await self.llm.achat(chat_history)
        raw_content = response.message.content
        
        # Robustness fix: LlamaIndex ReAct parser MUST have a "Thought:" line.
        # If the LLM skips it and jumps straight to "Action:", we prepend a default thought.
        if raw_content.strip().startswith("Action:") and "Thought:" not in raw_content:
            raw_content = f"Thought: I need to use a tool to fulfill the request.\n{raw_content}"
            trace_logger.info(f"--- [ROBUSTNESS FIX] PREPENDED MISSING THOUGHT ---")

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
                tool_output = await tool.acall(**tool_call.tool_kwargs)
                observation = tool_output.content
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
