"""`ph.tools`: the registry, the pipeline, and the batch scheduler."""

from __future__ import annotations

from .batch import BatchOutcome, execute_tool_calls, parse_arguments
from .definition import (
    Accept,
    Allow,
    Ask,
    Block,
    Deny,
    Done,
    ExecutionMode,
    FailureKind,
    NotDone,
    PostToolDecision,
    PreToolDecision,
    Reconciled,
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
    Unknown,
    aborted_result,
    call_id_of,
    define_tool,
    denied_result,
    error_result,
    recorded_arguments,
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
"""The two records that mean "a tool was let through and is about to run".

There is one for each transport. `batch._append_call` writes the first for a
native call, and `code_mode._log_start` writes the second for a dispatch inside
a cell. Both are written at the same point: after the pipeline decided, and
before the body ran (B4, P7-15). Anything that counts *work attempted* needs
both. With only the first, a Code Mode cell that read nine files and edited
three counts as one call (D12).

The set is declared here, with the two producers, and not in the packages that
fold it. `ph_stabilize.limits` and `ph_stabilize.todo` each had a copy, and
somebody who adds a third transport has no reason to know that either file
exists.
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
    "Done",
    "ExecutionMode",
    "FailureKind",
    "HarnessError",
    "NotDone",
    "PostToolDecision",
    "PreToolDecision",
    "PreparedCall",
    "Reconciled",
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
    "Unknown",
    "aborted_result",
    "call_id_of",
    "define_tool",
    "denied_result",
    "error_result",
    "execute_tool_calls",
    "parse_arguments",
    "recorded_arguments",
    "schema_of",
    "simple_views",
    "text_content",
    "unsupported_keywords",
    "validate_json_schema_value",
]
