# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `TaskBoard`, `Task`, and `TaskStatus`: in-process shared task list with atomic
  claim/assign/complete (asyncio-safe).
- `CollaborativeTeam`: leader seeds the board; members claim/complete tasks in
  parallel rounds (`max_rounds`); aggregated `RunUsage`; no peer DM yet.
- Docs page for collaborative teams.

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

[Unreleased]: https://github.com/etiqa/pydantic-team/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/etiqa/pydantic-team/releases/tag/v0.1.0
