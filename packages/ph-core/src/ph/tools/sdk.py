"""The `tools:sdk` prompt section: what Code Mode offers in place of schemas.

Under `mode: code` the model is given **one** callable and a generated SDK
listing. The SDK is what makes C1 work. Every capability that the model could
have called natively is reachable as `await tools.<name>(...)`. Each of those is
a governed binding that enters the pipeline again. The prompt says so in the
language of the runtime, so the model does not have to guess a calling
convention.

Two renderers ship, one for Python and one for TypeScript. The Code Mode row
registers them on `ctx.code_runtime`, and that seam is the *only* place where a
renderer is looked up. A runtime whose language has no renderer fails prompt
assembly, and does not ship a listing in the wrong syntax. A model that is given
TypeScript signatures for a Python runtime writes code that cannot run, and the
failure looks like a model problem.

**The TypeScript renderer is registered and currently cannot be selected.** The
`language` of the *runtime* chooses the renderer, and every runtime that pH
ships is Python. The renderer is kept for two reasons. The lookup of the seam
fails loudly, and that is useful only if the renderer that it finds is correct.
Also, P6-08 would otherwise have to derive the renderer again. But nothing in a
shipped profile can reach it today. `docs/dev-notes/typescript-code-runtime.md`
records what would make it live, and which part of that work is already done.

@module ph.tools.sdk
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..seams.code_runtime import CodeBinding, CodeBindingNamespace

__all__ = ["code_only_rule", "parameter_names", "render_python_sdk", "render_typescript_sdk"]

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


def _summary(binding: CodeBinding) -> str:
    description = binding.description.strip()
    return description.splitlines()[0] if description else ""


def _properties(binding: CodeBinding) -> list[tuple[str, dict[str, Any], bool]]:
    parameters = binding.parameters
    properties = parameters.get("properties") or {}
    required = set(parameters.get("required") or ())
    return [
        (name, definition if isinstance(definition, dict) else {}, name in required)
        for name, definition in properties.items()
    ]


def parameter_names(binding: CodeBinding) -> list[str]:
    """The parameters of a binding, in the order in which the listing declares
    them. A runtime binds positional arguments to these names, so a call that is
    written from the listing means what it says (`ph_runtime.proxies`)."""
    return [name for name, _definition, _required in _properties(binding)]


def _type_name(definition: dict[str, Any], table: dict[str, str], fallback: str) -> str:
    declared = definition.get("type")
    if isinstance(declared, list):
        declared = next((item for item in declared if item != "null"), None)
    return table.get(str(declared), fallback)


def render_python_sdk(namespaces: Sequence[CodeBindingNamespace]) -> str:
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


def render_typescript_sdk(namespaces: Sequence[CodeBindingNamespace]) -> str:
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
