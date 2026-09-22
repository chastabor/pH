"""`ph.tools` — the registry, the pipeline, and the batch scheduler."""

from __future__ import annotations

from .batch import BatchOutcome, execute_tool_calls, parse_arguments
from .definition import (
    Accept,
    Allow,
    Ask,
    Block,
    Deny,
    ExecutionMode,
    FailureKind,
    PostToolDecision,
    PreToolDecision,
    Respond,
    ToolDefinition,
    ToolExecution,
    ToolExecutionInput,
    ToolExecutionResult,
    ToolFailure,
    ToolModel,
    ToolOutput,
    ToolResult,
    ToolRunContext,
    aborted_result,
    define_tool,
    denied_result,
    error_result,
    text_content,
)
from .errors import (
    SPAWN_REFUSED,
    TOOL_ABORTED,
    TOOL_ABORTED_BEFORE_DISPATCH,
    TOOL_BUDGET_SPENT,
    TOOL_DENIED,
    TOOL_TURN_CONCLUDED,
    HarnessError,
    ToolNotFoundError,
    ToolOutputError,
)
from .json_schema import schema_of, unsupported_keywords, validate_json_schema_value
from .presentation import CardKind, ToolCallView, ToolResultView, simple_views
from .registry import RUN_CODE, PreparedCall, ToolGuard, ToolRestriction, ToolRuntime

TOOL_DISPATCH_EVENT_TYPES = frozenset({"tool/call", "tool/code-dispatch-start"})
"""The two records meaning "a tool was let through and is about to run".

One per transport: `batch._append_call` writes the first for a native call and
`code_mode._log_start` writes the second for a dispatch inside a cell, both at
the same point — after the pipeline decided, before the body ran (B4, P7-15).
Anything counting *work attempted* wants both, and wanting only the first is how
a Code Mode cell that read nine files and edited three scores as one (D12).

Declared here, with the two producers, rather than in the packages that fold it:
`ph_stabilize.limits` and `ph_stabilize.todo` had a copy each, and a third
transport is added by somebody with no reason to know either file exists.
"""

__all__ = [
    "RUN_CODE",
    "SPAWN_REFUSED",
    "TOOL_ABORTED",
    "TOOL_ABORTED_BEFORE_DISPATCH",
    "TOOL_BUDGET_SPENT",
    "TOOL_DENIED",
    "TOOL_DISPATCH_EVENT_TYPES",
    "TOOL_TURN_CONCLUDED",
    "Accept",
    "Allow",
    "Ask",
    "BatchOutcome",
    "Block",
    "CardKind",
    "Deny",
    "ExecutionMode",
    "FailureKind",
    "HarnessError",
    "PostToolDecision",
    "PreToolDecision",
    "PreparedCall",
    "Respond",
    "ToolCallView",
    "ToolDefinition",
    "ToolExecution",
    "ToolExecutionInput",
    "ToolExecutionResult",
    "ToolFailure",
    "ToolGuard",
    "ToolModel",
    "ToolNotFoundError",
    "ToolOutput",
    "ToolOutputError",
    "ToolRestriction",
    "ToolResult",
    "ToolResultView",
    "ToolRunContext",
    "ToolRuntime",
    "aborted_result",
    "define_tool",
    "denied_result",
    "error_result",
    "execute_tool_calls",
    "parse_arguments",
    "schema_of",
    "simple_views",
    "text_content",
    "unsupported_keywords",
    "validate_json_schema_value",
]
