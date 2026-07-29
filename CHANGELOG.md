# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Optional install extra ``pydantic-team[logfire]`` (pulls in ``logfire>=3.0``).
- `CollaborativeTeam.max_replans` and leader **replan** loop: after incomplete member
  rounds, the leader may add/assign more tasks (up to `max_replans`, default `0`)
  before synthesizing; OTel span `collaborative.replan`.
- `CollaborativeTeam.max_assignments_per_tick`: cap listed assignments per member tick
  (structural, not prompt-only); member ticks ignore `UnexpectedModelBehavior` from empty
  model finals after tool calls.
- `CollaborativeTeam.dispatch_mode`: `'phased'` (default) or `'streaming'` ready-queue
  dispatch so members start on assign and overlap seed/replan; OTel spans
  `collaborative.dispatch` / `collaborative.member_tick`.
- `TaskBoard` wakeup signals on assign/claim/complete for streaming schedulers.
- `CollaborativeTeam.iter` / `CollaborativeRun` plus `TeamTask` / `TeamEvent` types
  (`TasksScheduled`, `TaskCompleted`, `PhaseJoined`, `RunEnded`) for step-by-step
  observation inspired by pydantic-graph (without turning teams into a GraphBuilder).

### Changed

- Docs no longer describe HierarchicalTeam as “Agno coordinate”; wording uses
  pydantic-ai agent delegation only.
- Collaborative seed / replan / synthesize / member ticks each use an isolated
  `agent.run` usage budget; team `usage=` remains the aggregate sum.
- Collaborative synthesize is toolless (no `add_task` / `assign_task`), so the
  leader cannot keep mutating the board while producing the final answer.

## [0.2.0] - 2026-07-28

### Added

- `TaskBoard`, `Task`, and `TaskStatus`: in-process shared task list with atomic
  claim/assign/complete (asyncio-safe).
- `CollaborativeTeam`: leader seeds the board; members claim/complete tasks in
  parallel rounds (`max_rounds`); aggregated `RunUsage`; no peer DM yet.
- Docs page for collaborative teams.
- `instrument_pydantic_team()` / `is_instrumented()`: opt-in OpenTelemetry spans
  for team orchestration (`hierarchical.*`, `collaborative.*`), backend-agnostic
  via `opentelemetry-api` (works with Logfire after `logfire.configure()`).
- Examples: Logfire + `instrument_pydantic_ai()` + `instrument_pydantic_team()`,
  `python-dotenv` for `examples/.env`.

### Changed

- `CollaborativeTeam` defaults to **assign-by-role**: the leader must assign each
  task to a named teammate; member ticks receive per-agent prompts to complete
  only their assigned work (no cross-role claim monopolies).

### Removed

- Stdlib `pydantic_team` logger and interaction `logger.info` calls (replaced by
  OTel team spans).

## [0.1.0] - 2026-07-28

### Added

- Initial release of **pydantic-team**: type-safe team orchestration for
  [`pydantic-ai`](https://ai.pydantic.dev) Agents.
- `HierarchicalTeam`: leader agent that delegates to specialist members (or nested
  teams) via tools, aggregating `RunUsage` across nested runs.
- `BaseTeam` abstract base and `TeamResult` (`data` + `usage`) as the shared run
  outcome type.
- Support for constructing a leader from `leader_model=` or reusing an existing
  `leader_agent=`, optional `system_prompt_override`, and optional `name=` for
  nested teams.
- MkDocs documentation (guides + API reference) and a full test suite with
  100% coverage using pydantic-ai `TestModel` (no live LLM calls).

[Unreleased]: https://github.com/etiqa/pydantic-team/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/etiqa/pydantic-team/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/etiqa/pydantic-team/releases/tag/v0.1.0
