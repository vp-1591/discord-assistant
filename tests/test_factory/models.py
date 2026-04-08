"""
tests/test_factory/models.py
============================
Pydantic data models for the Prompt Evaluation Factory.
These are the shared contracts between the scenario config files,
the execution harness, and the Antigravity agent evaluator.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class ToolsMode(str, Enum):
    """Controls whether the harness uses real live services or injected mock data."""
    REAL = "real"
    MOCK = "mock"


class ExecutionTarget(str, Enum):
    """Identifies which component pipeline the harness should exercise."""
    AGENT1_RAG       = "agent1_rag"        # RAGAssistant.aquery()
    AGENT2_REACT     = "agent2_react"      # RAGAssistant.generate_refined_response()
    SUMMARIZATION    = "summarization"     # RAGAssistant.generate_summary()
    INGESTION_SUMMARY = "ingestion_summary" # Raw LLM call with INGESTION_SUMMARY_PROMPT


class PromptTestScenario(BaseModel):
    """
    Static configuration for a single prompt evaluation run.
    This file is READ-ONLY by the Antigravity agent during evaluation loops.
    """
    scenario_id: str = Field(..., description="Unique identifier, e.g. 'agent1_fetch_raw_logs_test'")
    execution_target: ExecutionTarget = Field(..., description="Which component pipeline to test")
    input_variables: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Variables injected into the pipeline. "
            "For agent1_rag / agent2_react: provide 'query_text'. "
            "For agent2_react: also provide 'history', 'summary', 'author_name', 'author_id'. "
            "For summarization: provide 'prev_summary' and 'messages' (list of str). "
            "For ingestion_summary: provide 'chunk' (raw message block as str)."
        )
    )
    rubrics: list[str] = Field(
        ...,
        description="Ordered list of pass/fail criteria evaluated by the Antigravity agent."
    )
    tools_mode: ToolsMode = Field(
        default=ToolsMode.REAL,
        description="'real' = live Ollama + DB, 'mock' = inject fixtures from mock_data_paths"
    )
    mock_data_paths: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Only used when tools_mode='mock'. "
            "Maps tool names to .json or .md fixture file paths relative to the scenario file's directory. "
            "Supported keys: 'hybrid_search', 'fetch_raw_logs', 'execute_sql', 'search_archive'. "
            "Example: {\"hybrid_search\": \"mock_data/hybrid_search_results.json\"}"
        )
    )
    prompt_id: Optional[str] = Field(
        default=None,
        description=(
            "The name of the prompt function/constant in src/config/prompts.py to override. "
            "If None, the harness will use the live prompt from prompts.py. "
            "Example: 'get_qa_prompt_tmpl' or 'SUMMARY_PROMPT_TEMPLATE'"
        )
    )


class ExecutionTrace(BaseModel):
    """
    Structured output of a single harness execution.
    Written to trace_output.json after each run for the Antigravity agent to evaluate.
    """
    scenario_id: str
    iteration: int = Field(default=1, description="Which iteration (1-3) produced this trace")
    prompt_id: Optional[str] = Field(default=None, description="The prompt override that was used")
    prompt_template_used: str = Field(
        ...,
        description=(
            "The fully-rendered ReAct system header seen by the LLM at runtime: "
            "CUSTOM_REACT_HEADER with {tool_desc}, {tool_names}, and {context} interpolated. "
            "For ingestion/summarization targets this is the raw template before chunk injection."
        )
    )
    output_text: str = Field(..., description="The final LLM response text")
    thoughts: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of the agent's Thought: blocks captured per ReAct step. "
            "Each entry is the raw reasoning text from one LLM call, stripped of "
            "'Thought:' prefix and <think> tags. Empty for non-ReAct targets."
        )
    )
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Sequence of tool calls captured during the run. "
            "Each entry: {name: str, kwargs: dict, result_preview: str}"
        )
    )
    error: Optional[str] = Field(
        default=None,
        description="Populated if the harness raised an unhandled exception"
    )
