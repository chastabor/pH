---
name: graphify
version: 1.0.0
description: Query or rebuild the graphify knowledge graph for this codebase — ask a question of it, trace a path, explain a node, or summarise it.
argument-hint: "<query|path|explain|build|report|visualize> [args]"
allowed-tools: [bash, read, write_todos]
parameters:
  action:
    type: string
    required: true
    enum: [query, path, explain, build, report, visualize]
    hint: What to do with the graph.
  args:
    type: string
    default: ""
    hint: "For query/explain: the question or concept. For path: two node names. Empty for build/report/visualize."
steps:
  - "Confirm graphify-out/graph.json exists, and build it first if it does not"
  - "Run the graphify action and show its full output"
  - "Summarise the key findings for the question that was actually asked"
---

# Graphify

Ask the knowledge graph rather than grepping the tree. The steps above are the
procedure; this is how to carry each one out.

## Rules

- Every command runs from the working directory.
- Show the command's full output before your summary.
- If graphify is not installed, say so and stop — do not fall back to grep
  silently, because the answer would be worse and the person would not know.

## 1. Confirm the graph exists

`graphify-out/graph.json` must exist. If it does not, run the `build` action
first and say that you did.

## 2. Run the action

The action is `{{parameters.action}}` and its arguments are `{{parameters.args}}`.

| Action | Command |
|---|---|
| `query` | `graphify query "{{parameters.args}}"` — add `--budget 100` if the default answer is too shallow |
| `path` | `graphify path "<NodeA>" "<NodeB>"` — arguments are two node names |
| `explain` | `graphify explain "{{parameters.args}}"` — type, location, relationships, description |
| `build` | `graphify update .` — writes `graphify-out/{graph.json,graph.html,GRAPH_REPORT.md}` |
| `report` | read `graphify-out/GRAPH_REPORT.md` |
| `visualize` | tell the person to open `graphify-out/graph.html`, then summarise the report |

## 3. Summarise

For `report` and `visualize`, cover the top communities and what they represent,
the most connected nodes, the isolated ones, and the total node and edge count.

For everything else, answer the question that was asked. The graph output is
evidence, not an answer — say what it means.
