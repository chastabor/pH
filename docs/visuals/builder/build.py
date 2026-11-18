"""Build the pages in `docs/visuals`: render each diagram, then add its drill-down.

A page is two sources. `src/<page>.<type>.json` is the diagram, written for
Archify, which renders it to one standalone HTML file. `src/details/<page>.yaml`
is what a reader gets by clicking a node or an arrow in it: the row behind it and
every option it takes, the event that crosses and the mode it is dispatched in,
the structure of what is sent.

**Most of a details file is a name, and this script looks the rest up in the
code.** `row: skill-steps` becomes that row's module, what it injects, where it
ships and whether it ships disabled, and its option table from its own config
model: `ph.cordis.catalog`, which is what `phern config` prints. `event:
agent/turn-stopping` becomes its dispatch mode, producer, listeners and payload
fields from the event registry, which is what `phern events` prints. `struct:
ph.llm.adapter:DecisionRequest` becomes that class's fields and their
docstrings. `frame: done` is the guest protocol's own field table, `record:
offload/spilled` the log vocabulary's entry, `events: waterfall` every event
declared in that mode, and `bundle: ph_stabilize/bundle.yaml` every row that file
mounts. So a page says what the code says
on the day it is built, and building it again is how it is brought up to date.
What is written by hand is the prose around those, and it goes stale like prose.

    uv run python docs/visuals/builder/build.py              # every page
    uv run python docs/visuals/builder/build.py ph-stabilize # one
    uv run python docs/visuals/builder/build.py --facts      # read the code again
    uv run python docs/visuals/builder/build.py --check      # resolve, render nothing

`--facts` mounts five profiles to learn who listens to each event, which takes
half a minute; the answer is kept in `builder/.facts.json` and reused until asked
for again. Archify is found at `$ARCHIFY_HOME`, else `~/.claude/skills/archify`.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import html
import importlib
import inspect
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import typing
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

BUILD = Path(__file__).resolve().parent
VISUALS = BUILD.parent
REPO = VISUALS.parents[1]
SRC = VISUALS / "src"
DETAILS = SRC / "details"
FACTS = BUILD / ".facts.json"
ARCHIFY = Path(os.environ.get("ARCHIFY_HOME", "~/.claude/skills/archify")).expanduser()

PROFILES = ("tui", "rlm", "rlm-clm", "rlm-stable", "rlm-indexed")
"""The profiles mounted to learn who listens: between them they layer every bundle."""

ARMED = ("{id: tool-todo, disabled: false}", "{id: skill-steps, disabled: false}")
"""Rows that ship disabled, switched on for one more mount so their listeners are seen."""

ARMED_LABEL = "when switched on"

DOC_LIMIT = 520
"""A field's docstring is cut here. The docstrings are essays; the panel is a table."""


class BuildError(Exception):
    """A details file names something the code does not have."""


# --- what the code says -----------------------------------------------------


def _ph_home() -> str:
    """An empty `$PH_HOME`, so a person's own profiles do not stand in for the
    shipped ones of the same name."""
    return tempfile.mkdtemp(prefix="ph-visuals-home-")


def _shipped_rows() -> dict[str, list[dict[str, Any]]]:
    """Every row a shipped YAML document mounts, by plugin name: which file, under
    which id, switched off or not, and the config it is given there."""
    found: dict[str, list[dict[str, Any]]] = {}
    files = [
        *sorted(REPO.glob("packages/*/src/**/bundle.yaml")),
        *sorted(REPO.glob("packages/ph-core/src/ph/bundles/*.yaml")),
        *sorted(REPO.glob("packages/phern/src/ph_app/profiles/*.yaml")),
    ]
    for file in files:
        try:
            documents = list(yaml.safe_load_all(file.read_text()))
        except yaml.YAMLError:
            continue
        for entry in _entries(documents):
            if "id" not in entry or "name" not in entry:
                continue
            found.setdefault(str(entry["name"]), []).append(
                {
                    "file": str(file.relative_to(REPO)),
                    "id": entry["id"],
                    "disabled": bool(entry.get("disabled", False)),
                    "config": entry.get("config"),
                }
            )
    return found


def _entries(node: object) -> list[dict[str, Any]]:
    """Every mapping in a profile document that could be a row, `insert:` included."""
    if isinstance(node, dict):
        inner = [found for value in node.values() for found in _entries(value)]
        return [node, *inner]
    if isinstance(node, list):
        return [found for value in node for found in _entries(value)]
    return []


def _consumers(home: str) -> dict[str, dict[str, list[str]]]:
    """Who listens to each event, per profile: `phern events`, asked of each as it
    ships, and once more with the rows that ship disabled switched on. A listener
    seen only then is listed under `ARMED_LABEL`."""
    listening: dict[str, dict[str, list[str]]] = {}
    runs = [(profile, profile, ()) for profile in PROFILES]
    runs.append((ARMED_LABEL, PROFILES[-1], ARMED))
    for label, profile, patches in runs:
        command = ["uv", "run", "phern", "events", "--json", "--profile", profile]
        for patch in patches:
            command += ["--patch", patch]
        done = subprocess.run(
            command,
            cwd=REPO,
            env={**os.environ, "PH_HOME": home},
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            matrix = json.loads(done.stdout)
        except json.JSONDecodeError:
            print(f"  profile {profile}: no event matrix ({done.stderr.strip()[-200:]})")
            continue
        for row in matrix:
            for module in row["consumers"]:
                seen = listening.setdefault(row["name"], {}).setdefault(module, [])
                if label != ARMED_LABEL or not seen:
                    seen.append(label)
    return listening


def gather_facts() -> dict[str, Any]:
    """Read the code: rows, events, log record types and guest frames."""
    home = _ph_home()
    os.environ["PH_HOME"] = home
    from ph.cordis.catalog import config_catalog
    from ph.cordis.events import events as registry
    from ph.cordis.loader import import_plugin_modules
    from ph.session import known_event_types as vocabulary
    from ph_runtime import protocol

    import_plugin_modules()
    shipped = _shipped_rows()
    rows = {
        entry["name"]: {**entry, "shipped": shipped.get(entry["name"], [])}
        for entry in config_catalog()
    }
    listening = _consumers(home)
    declared = {}
    for name in registry.names():
        declaration = registry.require(name)
        payload = declaration.payload
        declared[name] = {
            "name": name,
            "mode": declaration.mode,
            "producer": declaration.owner,
            "doc": declaration.doc,
            "payload": None if payload is None else f"{payload.__module__}:{payload.__qualname__}",
            "consumers": listening.get(name, {}),
        }
    records = {
        name: {
            "name": name,
            "writer": sorted(vocabulary.WRITERS.get(name, ()))
            or [vocabulary.declared_owner(name) or ""],
            "ignorable": vocabulary.is_ignorable(name),
            "auditOnly": vocabulary.is_audit_only(name),
            "exported": list(vocabulary.exported_fields(name)),
        }
        for name in sorted(vocabulary.known_log_types())
    }
    frames = {
        name: {
            "name": name,
            "from": "host" if name in protocol.HOST_FRAMES else "guest",
            "required": sorted(required),
            "optional": sorted(optional),
            "unproduced": name in protocol.UNPRODUCED_FRAMES,
        }
        for name, (required, optional) in protocol.FRAME_FIELDS.items()
    }
    return {
        "protocolVersion": protocol.PROTOCOL_VERSION,
        "rows": rows,
        "events": declared,
        "records": records,
        "frames": frames,
    }


def load_facts(*, refresh: bool) -> dict[str, Any]:
    if refresh or not FACTS.exists():
        print("reading the code (mounts five profiles; about half a minute)")
        facts = gather_facts()
        FACTS.write_text(json.dumps(facts, indent=1, default=str) + "\n")
        return facts
    return json.loads(FACTS.read_text())


# --- describing a type ------------------------------------------------------


def _short(text: str, limit: int = DOC_LIMIT) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text.rfind(". ", 0, limit)
    return text[: cut + 1] if cut > limit // 2 else text[:limit].rstrip() + "…"


def _summary(klass: type) -> str:
    """A class docstring's first paragraph: its own, not one a builtin base
    lends it (`dict`'s, for a TypedDict that has none)."""
    own = vars(klass).get("__doc__")
    doc = inspect.cleandoc(own) if isinstance(own, str) else ""
    if doc.startswith(f"{klass.__name__}("):
        doc = ""  # the signature `dataclass` writes where a class has no docstring
    return _short(doc.split("\n\n")[0], 420)


def _attribute_docs(klass: type) -> dict[str, str]:
    """Field name to the docstring written beneath it, for any class, base
    classes first. `ph.cordis.catalog.field_docs` does this for a config model;
    a log record, a frame and a request are dataclasses and TypedDicts."""
    docs: dict[str, str] = {}
    for one in reversed(klass.__mro__):
        if one.__module__ in ("builtins", "typing", "pydantic.main"):
            continue
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(one)))
        except (OSError, TypeError, SyntaxError):
            continue
        definitions = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if not definitions:
            continue
        body = definitions[0].body
        for statement, following in itertools.pairwise(body):
            if not isinstance(statement, ast.AnnAssign) or not isinstance(
                statement.target, ast.Name
            ):
                continue
            if isinstance(following, ast.Expr) and isinstance(following.value, ast.Constant):
                text = following.value.value
                if isinstance(text, str):
                    docs[statement.target.id] = " ".join(text.split())
    return docs


def _methods(klass: type) -> list[dict[str, str]]:
    """A class's own public methods, each with its signature and the first
    sentence of its docstring: what a Protocol or a service is, where a
    dataclass is its fields."""
    found: list[dict[str, str]] = []
    for name, member in vars(klass).items():
        if name.startswith("_"):
            continue
        function = member.fget if isinstance(member, property) else member
        function = getattr(function, "__func__", function)
        if not inspect.isfunction(function):
            continue
        try:
            signature = str(inspect.signature(function))
        except (TypeError, ValueError):
            signature = "(...)"
        signature = re.sub(r"^\(self(, )?", "(", signature).replace("'", "")
        doc = (inspect.getdoc(function) or "").split("\n\n")[0]
        found.append(
            {
                "name": name if isinstance(member, property) else f"{name}{signature}",
                "doc": _short(doc, 260),
            }
        )
    return found


def _written_types(klass: type) -> dict[str, str]:
    """Each field's annotation as the source writes it (`JsonValue | None`),
    which reads better than the alias expanded into every type it stands for."""
    written: dict[str, str] = {}
    for one in reversed(klass.__mro__):
        for name, annotation in vars(one).get("__annotations__", {}).items():
            # A string under postponed annotations; a TypedDict keeps a ForwardRef.
            said = getattr(annotation, "__forward_arg__", annotation)
            if isinstance(said, str):
                written[name] = said
    return written


def describe_type(ref: str) -> dict[str, Any]:
    """`module:Qual.Name` as a table of fields (name, type, default, what it is),
    and of methods where the class has no fields of its own."""
    from pydantic import BaseModel

    from ph.cordis import catalog

    module_name, _, qualname = ref.partition(":")
    try:
        target: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
    except (ImportError, AttributeError) as error:
        raise BuildError(f"struct {ref!r}: {error}") from error
    if not inspect.isclass(target):
        raise BuildError(f"struct {ref!r} is not a class")
    fields: list[dict[str, Any]]
    written = _written_types(target)
    if issubclass(target, BaseModel):
        fields = [
            {
                "name": one["name"],
                "attribute": one["attribute"],
                "type": written.get(one["attribute"], one["type"])
                if "ForwardRef" in one["type"] or len(one["type"]) > 90
                else one["type"],
                "required": one["required"],
                "default": one["default"],
                "doc": _short(one["doc"]),
            }
            for one in catalog._fields(target)
        ]
    else:
        docs = _attribute_docs(target)
        try:
            hints = typing.get_type_hints(target)
        except Exception:
            hints = dict(getattr(target, "__annotations__", {}))
        required_keys = getattr(target, "__required_keys__", None)
        defaults: dict[str, str | None] = {}
        if dataclasses.is_dataclass(target):
            for field in dataclasses.fields(target):
                if field.default is not dataclasses.MISSING:
                    defaults[field.name] = repr(field.default)
                elif field.default_factory is not dataclasses.MISSING:
                    defaults[field.name] = "(built per instance)"
                else:
                    defaults[field.name] = None
            names = [field.name for field in dataclasses.fields(target)]
        else:
            names = [name for name in hints if not name.startswith("_")]
        fields = []
        for name in names:
            default = defaults.get(name)
            if required_keys is not None:
                # A TypedDict. `__required_keys__` cannot see `NotRequired[...]` in a
                # module with postponed annotations, so the wrapper is read as written.
                said = written.get(name, "")
                wrapped = re.match(r"^(NotRequired|Required)\[(.*)\]$", said)
                if wrapped:
                    required = wrapped.group(1) == "Required"
                    written[name] = wrapped.group(2)
                else:
                    required = name in required_keys and getattr(target, "__total__", True)
            elif dataclasses.is_dataclass(target):
                required = default is None
            else:
                required = not hasattr(target, name)
            fields.append(
                {
                    "name": name,
                    "type": written.get(name) or catalog.render_annotation(hints.get(name, "")),
                    "required": required,
                    "default": default,
                    "doc": _short(docs.get(name, "")),
                }
            )
    return {
        "name": target.__qualname__,
        "module": target.__module__,
        "doc": _summary(target),
        "fields": fields,
        # A record is its fields; a Protocol or a service is its methods. Which of
        # the methods a panel shows is the block's to say (`_struct`).
        "methods": _methods(target),
    }


# --- prose ------------------------------------------------------------------

_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_EMPHASIS = re.compile(r"(?<![*\w])\*([^*\s][^*]*?)\*(?![*\w])")
_NUMBERED = re.compile(r"^\s*\d+\.\s+")


def inline(text: str) -> str:
    """One line of prose as HTML: `code`, **bold** and [links](target)."""
    kept: list[str] = []

    def keep(match: re.Match[str]) -> str:
        kept.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00{len(kept) - 1}\x00"

    text = _CODE.sub(keep, str(text))
    text = html.escape(text, quote=False)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _EMPHASIS.sub(r"<em>\1</em>", text)
    text = _LINK.sub(lambda m: f'<a href="{html.escape(_href(m.group(2)))}">{m.group(1)}</a>', text)
    return re.sub(r"\x00(\d+)\x00", lambda m: kept[int(m.group(1))], text)


def _href(target: str) -> str:
    """`node:id`, `edge:id` and `topic:id` open on this page; `page#node:id` on
    another; anything else is a link as written."""
    if re.match(r"^(node|edge|topic):", target):
        return f"?drill={target}"
    page, sep, drill = target.partition("#")
    if sep and re.match(r"^(node|edge|topic):", drill):
        return f"{page}.html?drill={drill}"
    if re.match(r"^[a-z0-9-]+$", target) and any(SRC.glob(f"{target}.*.json")):
        return f"{target}.html"
    return target


def prose(text: str) -> str:
    """Paragraphs, `- ` lists and `1.` lists as HTML."""
    out: list[str] = []
    for chunk in re.split(r"\n\s*\n", str(text).strip()):
        lines = [line.rstrip() for line in chunk.splitlines() if line.strip()]
        if lines and all(line.lstrip().startswith("- ") for line in lines):
            items = "".join(f"<li>{inline(line.lstrip()[2:])}</li>" for line in lines)
            out.append(f"<ul>{items}</ul>")
        elif lines and all(_NUMBERED.match(line) for line in lines):
            items = "".join(f"<li>{inline(_NUMBERED.sub('', line))}</li>" for line in lines)
            out.append(f"<ol>{items}</ol>")
        else:
            out.append(f"<p>{inline(' '.join(line.strip() for line in lines))}</p>")
    return "".join(out)


# --- resolving a details file -----------------------------------------------

KINDS = (
    "text",
    "code",
    "row",
    "event",
    "events",
    "bundle",
    "struct",
    "frame",
    "frames",
    "record",
    "table",
    "list",
    "links",
)


def resolve_block(block: dict[str, Any], facts: dict[str, Any], where: str) -> dict[str, Any]:
    kinds = [kind for kind in KINDS if kind in block]
    if len(kinds) != 1:
        raise BuildError(f"{where}: a block names one of {', '.join(KINDS)}; got {sorted(block)}")
    kind = kinds[0]
    value = block[kind]
    out: dict[str, Any] = {"kind": kind}
    if block.get("title"):
        out["title"] = inline(block["title"])
    if block.get("note"):
        out["note"] = prose(block["note"])
    if block.get("tone"):
        out["tone"] = str(block["tone"])
    if kind == "text":
        out["html"] = prose(value)
    elif kind == "code":
        out["text"] = str(value).rstrip("\n")
        out["lang"] = str(block.get("lang", "yaml"))
    elif kind == "row":
        row = facts["rows"].get(str(value))
        if row is None:
            raise BuildError(f"{where}: no registered row named {value!r}")
        out.update(
            name=row["name"],
            module=row["module"],
            affects=row.get("affects", ""),
            injects=row.get("injects", []),
            shipped=[
                {**one, "config": None if one["config"] is None else _yaml(one["config"])}
                for one in row["shipped"]
            ],
            options=[
                {
                    "name": one["name"],
                    "type": _clean_type(one["type"]),
                    "required": one["required"],
                    "default": one["default"],
                    "doc": inline(_short(one["doc"])),
                }
                for one in row.get("config", [])
            ],
        )
    elif kind == "event":
        event = facts["events"].get(str(value))
        if event is None:
            raise BuildError(f"{where}: no declared event named {value!r}")
        out.update(
            name=event["name"],
            mode=event["mode"],
            producer=event["producer"],
            doc=inline(event["doc"]),
            consumers=[
                {"module": module, "profiles": profiles}
                for module, profiles in sorted(event["consumers"].items())
            ],
            payload=None if event["payload"] is None else _struct(event["payload"]),
        )
        if block.get("args"):
            # What the dispatch site passes, for an event declared with no payload
            # type: `[argument, type, what it is]` rows, written by hand.
            out["args"] = [[inline(cell) for cell in row] for row in block["args"]]
    elif kind == "events":
        # Every declared event of one dispatch mode (or `all`), as a table.
        chosen = [
            event
            for event in facts["events"].values()
            if value in ("all", event["mode"]) or str(event["name"]).startswith(f"{value}/")
        ]
        if not chosen:
            raise BuildError(f"{where}: no declared event is {value!r}")
        out.setdefault("title", "Events")
        out["kind"] = "table"
        # The mode column says nothing in a table of one mode.
        with_mode = len({event["mode"] for event in chosen}) > 1
        out["columns"] = ["event", *(["mode"] if with_mode else []), "payload", "listeners"]
        out["rows"] = [
            [
                inline(f"`{event['name']}`")
                + f'<span class="ph-type">{html.escape(event["producer"])}</span>',
                *([event["mode"]] if with_mode else []),
                inline(f"`{event['payload'].partition(':')[2]}`") if event["payload"] else "",
                "<br>".join(
                    inline(f"`{module}`")
                    + (f" ({ARMED_LABEL})" if profiles == [ARMED_LABEL] else "")
                    for module, profiles in sorted(event["consumers"].items())
                ),
            ]
            for event in chosen
        ]
    elif kind == "bundle":
        # Every row one shipped YAML file mounts: on or off there, and its options.
        mounted = [
            (shipped, row)
            for row in facts["rows"].values()
            for shipped in row["shipped"]
            if shipped["file"].endswith(str(value))
        ]
        if not mounted:
            raise BuildError(f"{where}: no shipped file ending in {value!r} mounts a row")
        out.setdefault("title", "Rows it ships")
        out["kind"] = "table"
        out["columns"] = ["row", "ships", "options"]
        out["rows"] = [
            [
                inline(f"`{shipped['id']}`")
                + (
                    f'<span class="ph-type">{html.escape(row["name"])}</span>'
                    if row["name"] != shipped["id"]
                    else ""
                ),
                "off" if shipped["disabled"] else "on",
                ", ".join(inline(f"`{one['name']}`") for one in row.get("config", [])) or "none",
            ]
            for shipped, row in mounted
        ]
    elif kind == "struct":
        out.update(_struct(str(value), block.get("fields"), block.get("methods"), where))
    elif kind == "frame":
        frame = facts["frames"].get(str(value))
        if frame is None:
            raise BuildError(f"{where}: no guest protocol frame named {value!r}")
        out.update(frame)
        if block.get("example"):
            out["example"] = str(block["example"]).rstrip("\n")
    elif kind == "frames":
        # Every fd-3 frame one side sends (`host`, `guest`), or `all`.
        chosen = [frame for frame in facts["frames"].values() if value in ("all", frame["from"])]
        if not chosen:
            raise BuildError(f"{where}: frames takes all, host or guest; got {value!r}")
        out.setdefault("title", "Frames")
        out["kind"] = "table"
        out["columns"] = ["frame", "sent by", "required", "optional"]
        out["rows"] = [
            [
                inline(f"`{frame['name']}`"),
                frame["from"],
                ", ".join(inline(f"`{name}`") for name in frame["required"] if name != "type"),
                ", ".join(inline(f"`{name}`") for name in frame["optional"]),
            ]
            for frame in chosen
        ]
    elif kind == "record":
        record = facts["records"].get(str(value))
        if record is None:
            raise BuildError(f"{where}: no log record type named {value!r}")
        out.update(record)
        if block.get("example"):
            out["example"] = str(block["example"]).rstrip("\n")
    elif kind == "table":
        out["columns"] = [inline(cell) for cell in value["columns"]]
        out["rows"] = [[inline(cell) for cell in row] for row in value["rows"]]
    elif kind == "list":
        out["items"] = [inline(item) for item in value]
    elif kind == "links":
        out["items"] = [
            {"href": _href(str(item["to"])), "label": str(item["label"])} for item in value
        ]
    return out


def _clean_type(text: str) -> str:
    """`Annotated[list[float], FieldInfo(...)]` as `list[float]`: the constraint is
    in the docstring, and the wrapper is not what a profile writes."""
    while "Annotated[" in text:
        start = text.index("Annotated[")
        depth, at = 0, start + len("Annotated[")
        end = comma = None
        for index in range(at, len(text)):
            char = text[index]
            if char in "[(":
                depth += 1
            elif char in "])":
                if depth == 0:
                    end = index
                    break
                depth -= 1
            elif char == "," and depth == 0 and comma is None:
                comma = index
        if end is None:
            break
        text = (
            text[:start] + text[at : comma if comma is not None else end].strip() + text[end + 1 :]
        )
    return text


def _struct(
    ref: str,
    only: Sequence[str] | None = None,
    methods: object = None,
    where: str = "",
) -> dict[str, Any]:
    """A described type as the panel shows it.

    `only` keeps the named fields, in the order given, for a class too long to
    show whole; a name is the one a profile writes or the Python attribute.
    `methods` says which public methods to list: `True` for all of them, a list
    for those. Left out, methods show only for a class with no fields: a Protocol
    or a service, which is its methods. A name the class does not have is refused,
    so a renamed field fails the build rather than vanishing from the page."""
    described = describe_type(ref)
    if only:
        by_name = {
            name: one for one in described["fields"] for name in (one["name"], one.get("attribute"))
        }
        missing = [name for name in only if name not in by_name]
        if missing:
            raise BuildError(f"{where}: {ref} has no field {', '.join(missing)}")
        described["fields"] = [by_name[name] for name in only]
        described["partial"] = True
    if isinstance(methods, list):
        by_method = {one["name"].partition("(")[0]: one for one in described["methods"]}
        missing = [name for name in methods if name not in by_method]
        if missing:
            raise BuildError(f"{where}: {ref} has no public method {', '.join(missing)}")
        described["methods"] = [by_method[name] for name in methods]
        described["partial"] = True
    elif methods is not True and described["fields"]:
        described["methods"] = []
    for one in described["fields"]:
        one.pop("attribute", None)
        one["type"] = _clean_type(one["type"])
    for one in (*described["fields"], *described["methods"]):
        one["doc"] = inline(one["doc"])
    described["doc"] = inline(described["doc"])
    return described


def _yaml(value: object) -> str:
    return yaml.safe_dump(value, sort_keys=False, default_flow_style=False).rstrip("\n")


def resolve_entry(entry: dict[str, Any], facts: dict[str, Any], where: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("title", "transport", "mode", "kind"):
        if entry.get(key):
            out[key] = str(entry[key])
    if entry.get("summary"):
        out["summary"] = prose(entry["summary"])
    out["blocks"] = [
        resolve_block(block, facts, f"{where} block {index + 1}")
        for index, block in enumerate(entry.get("blocks") or [])
    ]
    return out


def resolve_details(page: str, spec: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """A page's details file, with every name looked up and every id checked
    against the diagram it is for."""
    file = DETAILS / f"{page}.yaml"
    authored = yaml.safe_load(file.read_text()) if file.exists() else {}
    authored = authored or {}
    node_ids, edge_ids = diagram_ids(spec)
    out: dict[str, Any] = {"page": page, "title": spec["meta"]["title"]}
    if authored.get("intro"):
        out["intro"] = prose(authored["intro"])
    out["topics"] = [
        {"id": topic["id"], **resolve_entry(topic, facts, f"{page} topic {topic['id']}")}
        for topic in authored.get("topics") or []
    ]
    for group, known in (("nodes", node_ids), ("edges", edge_ids)):
        entries = authored.get(group) or {}
        unknown = sorted(set(entries) - set(known))
        if unknown:
            raise BuildError(
                f"{page}: details name {group} the diagram does not have: {', '.join(unknown)}"
            )
        out[group] = {
            key: {"label": known[key], **resolve_entry(entry, facts, f"{page} {group} {key}")}
            for key, entry in entries.items()
        }
        missing = [key for key in known if key not in entries]
        if missing and file.exists():
            print(f"  {page}: no details for {group}: {', '.join(missing)}")
    return out


def diagram_ids(spec: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """The ids a details file may name, each with the label the diagram draws."""
    nodes: dict[str, str] = {}
    edges: dict[str, str] = {}
    for key in ("components", "nodes", "participants", "states"):
        for node in spec.get(key) or []:
            nodes[node["id"]] = node.get("label", node["id"])
    for key in ("connections", "flows", "messages", "transitions", "edges"):
        for index, edge in enumerate(spec.get(key) or []):
            if "id" in edge:
                edges[edge["id"]] = edge.get("label", edge["id"])
            else:
                print(f"  an edge without an id cannot carry details: {key}[{index}]")
    return nodes, edges


# --- rendering --------------------------------------------------------------


def pages() -> list[tuple[str, str, Path]]:
    """Every page: its name, its diagram type and its source."""
    found = []
    for file in sorted(SRC.glob("*.json")):
        page, _, kind = file.stem.partition(".")
        found.append((page, kind, file))
    return found


def render(page: str, kind: str, source: Path, details: dict[str, Any]) -> None:
    out = VISUALS / f"{page}.html"
    with tempfile.TemporaryDirectory(dir=VISUALS, prefix=".render-") as scratch:
        plain = Path(scratch) / f"{page}.html"
        done = subprocess.run(
            [
                "node",
                "bin/archify.mjs",
                "deliver",
                kind,
                str(source),
                str(plain),
                "--quality",
                "showcase",
                "--json",
            ],
            cwd=ARCHIFY,
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            raise BuildError(f"{page}: archify refused the diagram\n{done.stdout[-3000:]}")
        text = plain.read_text()
    payload = json.dumps(details, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    layer = (
        f'<style id="ph-drill-css">\n{(BUILD / "drill.css").read_text()}</style>\n'
        f'<script type="application/json" id="ph-drill-data">{payload}</script>\n'
        f'<script id="ph-drill-js">\n{(BUILD / "drill.js").read_text()}</script>\n'
    )
    at = text.rfind("</body>")
    if at < 0:
        raise BuildError(f"{page}: the rendered page has no </body>")
    out.write_text(text[:at] + layer + text[at:])


_DRILL_LINK = re.compile(
    r'(?:href=\\?"|"href": ")([a-z0-9-]*)(?:\.html)?\?drill=(node|edge|topic):([^"\\&]+)'
)


def broken_links(resolved: dict[str, dict[str, Any]]) -> list[str]:
    """Every link from one panel to another that opens nothing: a page that is not
    there, or an id its page does not have. Checked over the pages as resolved, so
    a renamed box is found where it is linked from."""
    targets = {
        page: {
            "node": set(details["nodes"]),
            "edge": set(details["edges"]),
            "topic": {topic["id"] for topic in details["topics"]},
        }
        for page, details in resolved.items()
    }
    broken: list[str] = []
    for page, details in resolved.items():
        text = json.dumps(details, ensure_ascii=False)
        for to_page, kind, target in _DRILL_LINK.findall(text):
            where = to_page or page
            if where not in targets:
                broken.append(f"{page}: links to a page that does not exist: {where}")
            elif target not in targets[where][kind]:
                broken.append(f"{page}: links to {where} {kind} {target!r}, which has no details")
    return sorted(set(broken))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pages", nargs="*", help="page names; all of them when none is given")
    parser.add_argument("--facts", action="store_true", help="read the code again first")
    parser.add_argument("--check", action="store_true", help="resolve details, render nothing")
    args = parser.parse_args()
    facts = load_facts(refresh=args.facts)
    every = pages()
    titles = []
    for page, _, source in every:
        titles.append({"page": page, "title": json.loads(source.read_text())["meta"]["title"]})
    wanted = [one for one in every if not args.pages or one[0] in args.pages]
    unknown = sorted(set(args.pages) - {page for page, _, _ in every})
    if unknown:
        print(f"no such page: {', '.join(unknown)}")
        return 2
    failed = 0
    resolved: dict[str, dict[str, Any]] = {}
    for page, kind, source in wanted:
        spec = json.loads(source.read_text())
        try:
            details = resolve_details(page, spec, facts)
            details["pages"] = titles
            if not args.check:
                render(page, kind, source, details)
        except BuildError as error:
            failed += 1
            print(f"FAILED {error}")
            continue
        resolved[page] = details
        counted = f"{len(details['nodes'])} nodes, {len(details['edges'])} edges"
        print(f"{'checked' if args.check else 'built'} {page}: {counted} with details")
    if len(resolved) == len(every):
        # Links are checked only when every page was resolved: one page alone
        # cannot say whether another has the box it points at.
        broken = broken_links(resolved)
        for line in broken:
            print(f"BROKEN LINK {line}")
        failed += len(broken)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
