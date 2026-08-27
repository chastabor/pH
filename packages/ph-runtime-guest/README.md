# ph-runtime-guest

The guest half of pH's Python code runtime. It runs **inside** the runtime venv
(`$PH_CACHE/runtime-venv`), in a subprocess the host spawns per agent, and talks
to the host over a single framed-JSON channel on fd 3.

It deliberately does **not** import `ph-core` or `ph-rlm`. The host lives in a
different venv, and the whole point of the process boundary is that model code
cannot reach the harness. The two halves of the protocol are therefore written
twice — once here, once in `ph_rlm.kernel.protocol` — and a mirror test asserts
they agree on every constant and every frame's field set. That test is the
contract; there is no shared module to keep them honest.
