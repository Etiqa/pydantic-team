# Roadmap

This page is informative, not a commitment to release dates or ordering.
Priorities can change with feedback and upstream pydantic-ai work.

Team members do **not** carry LLM `message_history` across invocations today.
Hierarchical delegates and Collaborative ticks each start a fresh `agent.run`;
context between cycles lives on the **task board** (and peer messages), not in a
per-member chat thread.

## Done (through v0.3)

- [`HierarchicalTeam`](hierarchical.md) — leader delegation, nested teams, aggregated usage
- [`CollaborativeTeam`](collaborative.md) — shared board, assign-by-role, replan, phased/streaming
  dispatch, peer `send_message` / `list_messages`, `iter` / typed events
- OpenTelemetry via [`instrument_pydantic_team`](index.md#observability)
- Opt-in Collaborative review gate (`require_review` / `assign_reviewer`,
  `approve_task` / `reject_task`, `pending_review` → `done` or `needs_revision`,
  [`TaskReviewDecided`](api.md) events) — see [Collaborative teams](collaborative.md)

## Next

Highest-value gaps already called out in the docs:

### Collaborative Team of Teams

[`HierarchicalTeam`](hierarchical.md) already accepts nested `BaseTeam` members.
[`CollaborativeTeam`](collaborative.md) members must be agents only in this slice.

Goal: a Collaborative member may be another team (`HierarchicalTeam` and/or a nested
`CollaborativeTeam`), so the top board can assign work to a whole sub-team.

## Later / maybe

Features often requested in the multi-agent ecosystem; several belong partly in
pydantic-ai first:

- **Task dependencies (`blocked_by`)** — a “subtask” is another board task that a
  parent depends on (or that blocks it). No separate nested `SubTask` type. Leader
  (and optionally members with limits) add tasks and dependency edges; a task is not
  ready until its blockers are done
- **Scoped history / resume** — Hierarchical: thin `message_history` passthrough to the
  leader for multi-turn user↔team chat. Collaborative: persist/resume a **board
  snapshot** across `team.run` calls. Not AutoGen-style shared peer chat history
- Richer UI lineage / nested event streaming (we already expose `iter` and OTel spans)
- HITL / deferred approval wired through team runs (depends on pydantic-ai nested
  deferred tools)
- Per-member usage and cost limits
- Richer inbox beyond current board DM / broadcast

## Non-goals

- Sequential / branching / stateful pipelines — use
  [pydantic-graph](https://ai.pydantic.dev/graph/)
- Agno-style session DB, long-term memory, or knowledge bases in this library’s core
  (prefer pydantic-ai / the application layer when those land)
- AutoGen-style group chat as the primary coordination model
- Reimplementing native handoffs that pydantic-ai intends to ship
- A separate nested “SubTask” API unrelated to task dependencies
