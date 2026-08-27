"""Output modes: print, json, transcript, rpc. The TUI lands in Phase 2."""

from __future__ import annotations

from .json_mode import JsonResult, run_json
from .print_mode import PrintResult, run_print
from .rpc_mode import RpcServer, run_rpc
from .transcript_mode import TranscriptResult, render_transcript, run_transcript

__all__ = [
    "JsonResult",
    "PrintResult",
    "RpcServer",
    "TranscriptResult",
    "render_transcript",
    "run_json",
    "run_print",
    "run_rpc",
    "run_transcript",
]
