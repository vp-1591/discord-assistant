"""
tests/test_factory/run_harness.py
==================================
Execution harness for the Prompt Evaluation Factory.

Usage (called by the Antigravity agent via the IDE terminal):
    python tests/test_factory/run_harness.py \\
        --scenario tests/test_factory/scenarios/agent1_historical_fact.json \\
        --prompt   tests/test_factory/.drafts/draft_prompt.txt \\
        [--iteration 1]

Outputs:
    tests/test_factory/trace_output.json  (ExecutionTrace model)
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import subprocess
import time
import re
import traceback
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

# ---------------------------------------------------------------------------
# Bootstrap: project root on sys.path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.test_factory.models import (
    ExecutionTarget, ExecutionTrace, PromptTestScenario, ToolsMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_ollama_running():
    """Tries to start ollama serve in the background. Fails silently if already bound."""
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags
        )
        # Briefly wait in case it needs to initialize
        time.sleep(1)
    except Exception as e:
        print(f"[harness] Warning: Failed to auto-start ollama serve: {e}")


def _load_scenario(scenario_path: str) -> PromptTestScenario:
    with open(scenario_path, encoding="utf-8") as f:
        return PromptTestScenario.model_validate_json(f.read())


def _load_draft_prompt(prompt_path: str | None) -> str | None:
    """Returns the raw text content of the draft prompt file, or None if no override."""
    if not prompt_path or not Path(prompt_path).exists():
        return None
    return Path(prompt_path).read_text(encoding="utf-8")


def _load_mock_fixture(fixture_path: str, scenario_dir: Path) -> Any:
    """
    Loads a mock data fixture from a .json or .md file.
    The fixture_path is resolved relative to the scenario file's directory.
    """
    full_path = (scenario_dir / fixture_path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"Mock fixture not found: {full_path}")
    if full_path.suffix == ".json":
        with open(full_path, encoding="utf-8") as f:
            return json.load(f)
    # For .md files, return raw text
    return full_path.read_text(encoding="utf-8")


def _render_react_template(system_prompt: str, tools: list) -> str:
    """
    Renders the full ReAct protocol header exactly as seen by the LLM at runtime.
    Mirrors the {tool_desc}/{context}/{tool_names} interpolation done by
    ReActChatFormatter in agent_core.py.

    Only the static system header is captured here (Header + Tools + System Prompt).
    The Thought/Observation/Action turns are recorded separately in tool_calls.
    """
    from src.config.prompts import CUSTOM_REACT_HEADER

    tool_names = ", ".join(t.metadata.get_name() for t in tools)
    tool_desc_parts = []
    for t in tools:
        name = t.metadata.get_name()
        desc = (t.metadata.description or "").strip()
        tool_desc_parts.append(f"Tool: {name}\nDescription: {desc}")
    tool_desc = "\n\n".join(tool_desc_parts)

    return CUSTOM_REACT_HEADER.format(
        tool_desc=tool_desc,
        context=system_prompt,
        tool_names=tool_names,
    )


def _extract_thought(raw_content: str) -> str:
    """
    Extracts the agent's reasoning text from a raw ReAct LLM response.
    Handles both Qwen3's native <think>...</think> blocks and the plain
    'Thought: ...' prefix convention used by LlamaIndex's ReActOutputParser.
    Returns an empty string when no thought is found (e.g. direct Answer).
    """
    think_match = re.search(r"<think>(.*?)</think>", raw_content, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    react_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nAnswer:)", raw_content, re.DOTALL)
    if react_match:
        return react_match.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------

def _build_mock_tool(fixture_value: Any) -> AsyncMock:
    """Returns an AsyncMock that returns the fixture_value for any call."""
    mock = AsyncMock(return_value=fixture_value)
    return mock


# ---------------------------------------------------------------------------
# Execution strategies per target
# ---------------------------------------------------------------------------

async def _run_agent1_rag(
    scenario: PromptTestScenario,
    draft_prompt: str | None,
    scenario_dir: Path,
) -> tuple[str, str, list, list]:
    """
    Exercises RAGAssistant.aquery().
    Returns (prompt_rendered, output_text, tool_calls).
    """
    from src.config.config import configure_settings
    configure_settings()

    from llama_index.core import Settings as _Settings

    thoughts: list[str] = []
    _llm = _Settings.llm
    # Save the BOUND method before class-level patching. Calling it later as
    # _original_bound_achat(messages) makes the dispatcher receive (llm_instance, messages)
    # internally, which correctly satisfies the original achat(self, messages, **kwargs) signature.
    _original_bound_achat = _llm.achat

    async def _capturing_achat(self, messages, **kwargs):
        # Delegate via the saved BOUND method — do NOT pass self explicitly.
        response = await _original_bound_achat(messages, **kwargs)
        raw = response.message.content or ""
        try:
            blocks = (
                getattr(response.message, "blocks", None)
                or response.message.additional_kwargs.get("blocks", [])
            )
            think_parts = [
                (b["content"] if isinstance(b, dict) else getattr(b, "content", ""))
                for b in (blocks or [])
                if (b.get("block_type") if isinstance(b, dict) else getattr(b, "block_type", "")) == "thinking"
            ]
            if think_parts:
                raw = f"<think>\n{''.join(think_parts)}\n</think>\n" + raw
        except Exception:
            pass
        thought = _extract_thought(raw)
        if thought:
            thoughts.append(thought)
        return response

    # Patch at the class level to bypass Pydantic's __setattr__ on the instance.
    # The explicit (self, messages, **kwargs) signature satisfies the dispatcher's inspect.signature check.
    _achat_patch = patch.object(type(_llm), 'achat', _capturing_achat)
    _achat_patch.start()

    tool_calls: list = []

    # Build patch context for the prompt override
    patch_targets: dict = {}
    if draft_prompt is not None and scenario.prompt_id == "get_qa_prompt_tmpl":
        # We patch the function to return the draft text regardless of bot_name arg
        patch_targets["src.config.prompts.get_qa_prompt_tmpl"] = lambda *_a, **_kw: draft_prompt
        patch_targets["src.core.run_llama_index.get_qa_prompt_tmpl"] = lambda *_a, **_kw: draft_prompt

    # Build mock patches for tools if needed
    tool_patches: dict = {}
    if scenario.tools_mode == ToolsMode.MOCK:
        for tool_name, fixture_path in scenario.mock_data_paths.items():
            fixture = _load_mock_fixture(fixture_path, scenario_dir)
            if isinstance(fixture, (dict, list)):
                fixture = json.dumps(fixture, ensure_ascii=False)
            if tool_name == "hybrid_search":
                tool_patches["hybrid_search"] = fixture
            elif tool_name == "fetch_raw_logs":
                tool_patches["fetch_raw_logs"] = fixture
            elif tool_name == "execute_sql":
                tool_patches["execute_sql"] = fixture

    from src.core.run_llama_index import RAGAssistant

    assistant = RAGAssistant()

    # Render full ReAct system header for the trace (post-instantiation so we have real tools).
    # Uses the draft system prompt if overriding, otherwise the live production prompt.
    _agent1_system_prompt = (
        draft_prompt
        if draft_prompt is not None and scenario.prompt_id == "get_qa_prompt_tmpl"
        else __import__('src.config.prompts', fromlist=['get_qa_prompt_tmpl']).get_qa_prompt_tmpl(assistant.name)
    )
    _agent1_sample_tools = assistant._build_agent1_tools()
    prompt_template_used = _render_react_template(_agent1_system_prompt, _agent1_sample_tools)

    # Wrap the real tools with recorders (and optionally replace with mocks)
    original_build = assistant._build_agent1_tools

    def _patched_build_agent1_tools():
        tools = original_build()
        patched = []
        for tool in tools:
            name = tool.metadata.get_name()
            if name in tool_patches:
                # Replace with mock fixture
                mock_result = tool_patches[name]
                recorder_async = AsyncMock(return_value=mock_result)
                # Wrap recorder
                original_async = recorder_async

                async def _recorded(rec_name=name, fn=original_async, **kwargs):
                    result = await fn(**kwargs)
                    tool_calls.append({
                        "name": rec_name,
                        "kwargs": kwargs,
                        "result_preview": str(result)[:500],
                    })
                    return result

                from llama_index.core.tools import FunctionTool
                patched.append(
                    FunctionTool.from_defaults(
                        async_fn=_recorded,
                        name=name,
                        description=tool.metadata.description,
                        fn_schema=tool.metadata.fn_schema,
                    )
                )
            else:
                # Wrap real tool with recorder
                real_async = tool.async_fn if hasattr(tool, "async_fn") else tool._async_fn

                async def _recorded_real(rec_name=name, fn=real_async, **kwargs):
                    result = await fn(**kwargs)
                    tool_calls.append({
                        "name": rec_name,
                        "kwargs": kwargs,
                        "result_preview": str(result)[:500],
                    })
                    return result

                from llama_index.core.tools import FunctionTool
                patched.append(
                    FunctionTool.from_defaults(
                        async_fn=_recorded_real,
                        name=name,
                        description=tool.metadata.description,
                        fn_schema=tool.metadata.fn_schema,
                    )
                )
        return patched

    assistant._build_agent1_tools = _patched_build_agent1_tools

    # Apply prompt patches
    patch_stack = [patch(k, v) for k, v in patch_targets.items()]
    for p in patch_stack:
        p.start()

    try:
        query_text = scenario.input_variables.get("query_text", "Тестовый запрос")
        output = await assistant.aquery(query_text)
    finally:
        _achat_patch.stop()
        for p in patch_stack:
            p.stop()

    return prompt_template_used, output, tool_calls, thoughts


async def _run_agent2_react(
    scenario: PromptTestScenario,
    draft_prompt: str | None,
    scenario_dir: Path,
) -> tuple[str, str, list, list]:
    """Exercises RAGAssistant.generate_refined_response()."""
    from src.config.config import configure_settings
    configure_settings()

    from llama_index.core import Settings as _Settings

    thoughts: list[str] = []
    _llm = _Settings.llm
    _original_bound_achat = _llm.achat

    async def _capturing_achat(self, messages, **kwargs):
        response = await _original_bound_achat(messages, **kwargs)
        raw = response.message.content or ""
        try:
            blocks = (
                getattr(response.message, "blocks", None)
                or response.message.additional_kwargs.get("blocks", [])
            )
            think_parts = [
                (b["content"] if isinstance(b, dict) else getattr(b, "content", ""))
                for b in (blocks or [])
                if (b.get("block_type") if isinstance(b, dict) else getattr(b, "block_type", "")) == "thinking"
            ]
            if think_parts:
                raw = f"<think>\n{''.join(think_parts)}\n</think>\n" + raw
        except Exception:
            pass
        thought = _extract_thought(raw)
        if thought:
            thoughts.append(thought)
        return response

    _achat_patch = patch.object(type(_llm), 'achat', _capturing_achat)
    _achat_patch.start()

    tool_calls: list = []
    patch_targets: dict = {}

    if draft_prompt is not None and scenario.prompt_id == "get_system_prompt":
        patch_targets["src.config.prompts.get_system_prompt"] = lambda *_a, **_kw: draft_prompt
        patch_targets["src.core.run_llama_index.get_system_prompt"] = lambda *_a, **_kw: draft_prompt

    from src.core.run_llama_index import RAGAssistant
    assistant = RAGAssistant()

    # Render full ReAct system header for the trace.
    # We need a concrete system_prompt to fill {context}; build a representative one
    # from scenario input_variables (mirrors what generate_refined_response does at runtime).
    iv_preview = scenario.input_variables
    _prompts = __import__('src.config.prompts', fromlist=['get_persona_prompt', 'get_system_prompt'])
    _agent2_system_prompt = (
        draft_prompt
        if draft_prompt is not None and scenario.prompt_id == "get_system_prompt"
        else _prompts.get_system_prompt(
            author_name=iv_preview.get("author_name", "Тестер"),
            persona=_prompts.get_persona_prompt(iv_preview.get("bot_name", assistant.name)),
            author_profile=iv_preview.get("author_profile", "None"),
            summary=iv_preview.get("summary", ""),
            history_str="\n".join(iv_preview.get("history", [])),
            query_text=iv_preview.get("query_text", ""),
            replied_to_msg=iv_preview.get("replied_to_msg", None),
        )
    )
    _agent2_sample_tools = assistant._build_tools(
        author_id=iv_preview.get("author_id", "0"),
        author_name=iv_preview.get("author_name", "Тестер"),
    )
    prompt_template_used = _render_react_template(_agent2_system_prompt, _agent2_sample_tools)

    patch_stack = [patch(k, v) for k, v in patch_targets.items()]
    for p in patch_stack:
        p.start()

    try:
        iv = scenario.input_variables
        output = await assistant.generate_refined_response(
            query_text=iv.get("query_text", "Расскажи о сервере"),
            history=iv.get("history", []),
            summary=iv.get("summary", ""),
            bot_name=iv.get("bot_name", "Глава Архива"),
            author_id=iv.get("author_id", "0"),
            author_name=iv.get("author_name", "Тестер"),
            replied_to_msg=iv.get("replied_to_msg", None),
        )
    finally:
        _achat_patch.stop()
        for p in patch_stack:
            p.stop()

    return prompt_template_used, output, tool_calls, thoughts


async def _run_summarization(
    scenario: PromptTestScenario,
    draft_prompt: str | None,
    scenario_dir: Path,
) -> tuple[str, str, list, list]:
    """Exercises RAGAssistant.generate_summary()."""
    from src.config.config import configure_settings
    configure_settings()

    patch_targets: dict = {}
    prompt_template_used = "<live prompt — see src/config/prompts.py::SUMMARY_PROMPT_TEMPLATE>"

    if draft_prompt is not None and scenario.prompt_id == "SUMMARY_PROMPT_TEMPLATE":
        patch_targets["src.config.prompts.SUMMARY_PROMPT_TEMPLATE"] = draft_prompt
        patch_targets["src.core.run_llama_index.SUMMARY_PROMPT_TEMPLATE"] = draft_prompt
        prompt_template_used = draft_prompt

    from src.core.run_llama_index import RAGAssistant
    assistant = RAGAssistant()

    patch_stack = [patch(k, v) for k, v in patch_targets.items()]
    for p in patch_stack:
        p.start()

    try:
        iv = scenario.input_variables
        output = await assistant.generate_summary(
            prev_summary=iv.get("prev_summary", ""),
            messages=iv.get("messages", ["Тестовое сообщение"]),
        )
    finally:
        for p in patch_stack:
            p.stop()

    return prompt_template_used, output, [], []


async def _run_ingestion_summary(
    scenario: PromptTestScenario,
    draft_prompt: str | None,
    scenario_dir: Path,
) -> tuple[str, str, list, list]:
    """
    Runs the ingestion summarization prompt directly against the LLM.
    No RAGAssistant needed — raw acomplete() call.
    """
    from src.config.config import configure_settings
    configure_settings()
    from llama_index.core import Settings
    import src.config.prompts as prompts_module

    template = draft_prompt if draft_prompt else prompts_module.INGESTION_SUMMARY_PROMPT
    chunk_text = scenario.input_variables.get("chunk", "Тест: пустой чанк")
    prompt_rendered = template.format(chunk=chunk_text)

    response = await Settings.llm.acomplete(prompt_rendered)
    return template, str(response).strip(), [], []


# ---------------------------------------------------------------------------
# Target registry
# ---------------------------------------------------------------------------

_REGISTRY = {
    ExecutionTarget.AGENT1_RAG:        _run_agent1_rag,
    ExecutionTarget.AGENT2_REACT:      _run_agent2_react,
    ExecutionTarget.SUMMARIZATION:     _run_summarization,
    ExecutionTarget.INGESTION_SUMMARY: _run_ingestion_summary,
}


# ---------------------------------------------------------------------------
# Main harness runner
# ---------------------------------------------------------------------------

async def run(
    scenario_path: str,
    prompt_path: str | None = None,
    iteration: int = 1,
    output_path: str | None = None,
) -> ExecutionTrace:
    _ensure_ollama_running()

    # Pre-flight cleanup: starting a new evaluation cycle wipes stale iteration files
    # so the evaluator agent never reads leftover data from a previous scenario.
    if iteration == 1:
        target_dir = Path(output_path).parent if output_path else Path(__file__).parent
        for stale_file in target_dir.glob("trace_iter_*.json"):
            try:
                stale_file.unlink()
                print(f"[harness] Purged old trace: {stale_file.name}")
            except Exception as e:
                print(f"[harness] Warning: Failed to clean {stale_file.name} -> {e}")

    scenario = _load_scenario(scenario_path)
    scenario_dir = Path(scenario_path).resolve().parent
    draft_prompt = _load_draft_prompt(prompt_path)

    if output_path is None:
        output_path = str(Path(__file__).parent / f"trace_iter_{iteration}.json")

    strategy = _REGISTRY.get(scenario.execution_target)
    if strategy is None:
        raise ValueError(f"Unknown execution_target: {scenario.execution_target}")

    error: str | None = None
    prompt_template_used = ""
    output_text = ""
    tool_calls: list = []
    thoughts: list = []

    try:
        prompt_template_used, output_text, tool_calls, thoughts = await strategy(
            scenario, draft_prompt, scenario_dir
        )
    except Exception:
        error = traceback.format_exc()
        output_text = ""

    trace = ExecutionTrace(
        scenario_id=scenario.scenario_id,
        iteration=iteration,
        prompt_id=scenario.prompt_id,
        prompt_template_used=prompt_template_used,
        output_text=output_text,
        thoughts=thoughts,
        tool_calls=tool_calls,
        error=error,
    )

    Path(output_path).write_text(
        trace.model_dump_json(indent=2), encoding="utf-8"
    )
    print(f"[harness] Trace written -> {output_path}")
    if error:
        print(f"[harness] EXECUTION ERROR CAPTURED - see {Path(output_path).name} error field")
    return trace


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Evaluation Harness")
    parser.add_argument(
        "--scenario", required=True,
        help="Path to a PromptTestScenario JSON file"
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Path to the draft_prompt.txt override file"
    )
    parser.add_argument(
        "--iteration", type=int, default=1,
        help="Iteration number (1-3) for the trace metadata"
    )
    parser.add_argument(
        "--output", default=None,
        help="Override path for trace_output.json (default: tests/test_factory/trace_output.json)"
    )
    args = parser.parse_args()

    asyncio.run(run(
        scenario_path=args.scenario,
        prompt_path=args.prompt,
        iteration=args.iteration,
        output_path=args.output,
    ))
