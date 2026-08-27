# ph-core

The core of pH: `ph.cordis` (the plugin meta-framework subset), `ph.session`
(the append-only event log and its derived message surface), `ph.llm` (the
provider-neutral vocabulary), `ph.agent`/`ph.agent_loop` (the ReAct driver),
and the local providers for the base capability seams.

`ph-core` must stay free of Textual / Rich / Typer imports — a test enforces it.
