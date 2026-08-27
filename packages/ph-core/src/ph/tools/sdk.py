"""The `tools:sdk` prompt section: what Code Mode offers instead of schemas.

Under `mode: code` the model is handed **one** callable and a generated SDK
listing. The SDK is the thing that makes C1 work: every capability the model
could have called natively is reachable as `await tools.<name>(...)`, each of
those is a governed binding that re-enters the pipeline, and the prompt says so
in the runtime's own language so the model does not have to guess a calling
convention.

Two renderers ship (Python and TypeScript). They are registered on
`ctx.code_runtime` by the Code Mode row, and that seam is the *only* place a
renderer is looked up — a runtime whose language has none fails prompt assembly
rather than shipping a listing in the wrong syntax, since a model given
TypeScript signatures for a Python runtime writes code that cannot run and the
failure looks like a model problem.

@module ph.tools.sdk
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["code_only_rule", "render_python_sdk", "render_typescript_sdk"]

_PYTHON_TYPES = {
    "string": "str",
    "number": "float",
    "integer": "int",
    "boolean": "bool",
    "object": "dict",
    "array": "list",
    "null": "None",
}

_TS_TYPES = {
    "string": "string",
    "number": "number",
    "integer": "number",
    "boolean": "boolean",
    "object": "object",
    "array": "unknown[]",
    "null": "null",
}


def _summary(binding: Any) -> str:
    description = str(getattr(binding, "description", "") or "").strip()
    return description.splitlines()[0] if description else ""


def _properties(binding: Any) -> list[tuple[str, dict[str, Any], bool]]:
    parameters = getattr(binding, "parameters", None) or {}
    properties = parameters.get("properties") or {}
    required = set(parameters.get("required") or ())
    return [
        (name, definition if isinstance(definition, dict) else {}, name in required)
        for name, definition in properties.items()
    ]


def _type_name(definition: dict[str, Any], table: dict[str, str], fallback: str) -> str:
    declared = definition.get("type")
    if isinstance(declared, list):
        declared = next((item for item in declared if item != "null"), None)
    return table.get(str(declared), fallback)


def render_python_sdk(namespaces: Sequence[Any]) -> str:
    """Render the Python SDK block for a set of binding namespaces."""
    lines: list[str] = []
    for namespace in namespaces:
        header = f"# {namespace.name}"
        if namespace.description:
            header += f" — {namespace.description}"
        lines.append(header)
        for binding in namespace.bindings:
            signature = ", ".join(
                f"{name}: {_type_name(definition, _PYTHON_TYPES, 'Any')}"
                + ("" if required else " = ...")
                for name, definition, required in _properties(binding)
            )
            lines.append(f"async def {namespace.name}.{binding.name}({signature}) -> Any: ...")
            summary = _summary(binding)
            if summary:
                lines.append(f'    """{summary}"""')
        lines.append("")
    return "\n".join(lines).rstrip()


def render_typescript_sdk(namespaces: Sequence[Any]) -> str:
    """Render the TypeScript SDK block for a set of binding namespaces."""
    lines: list[str] = []
    for namespace in namespaces:
        header = f"// {namespace.name}"
        if namespace.description:
            header += f" — {namespace.description}"
        lines.append(header)
        lines.append(f"declare const {namespace.name}: {{")
        for binding in namespace.bindings:
            signature = ", ".join(
                f"{name}{'' if required else '?'}: {_type_name(definition, _TS_TYPES, 'unknown')}"
                for name, definition, required in _properties(binding)
            )
            summary = _summary(binding)
            if summary:
                lines.append(f"  /** {summary} */")
            lines.append(f"  {binding.name}(args: {{ {signature} }}): Promise<unknown>")
        lines.append("}")
        lines.append("")
    return "\n".join(lines).rstrip()


def code_only_rule(transport: str) -> str:
    """The prompt rule that tells the model which surface it actually has."""
    return (
        f"Every capability below is reached by writing code and passing it to "
        f"`{transport}`. Do not call these names directly as tools — only "
        f"`{transport}` is callable. Each `await` inside your program is "
        "individually governed, recorded, and may be refused; a refusal fails the "
        "whole program, so check what you are about to do before you do it."
    )
