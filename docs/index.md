# pydantic-team

Type-safe team orchestration for [`pydantic-ai`](https://ai.pydantic.dev) Agents.

## What this library is

v1 provides:

- **[`HierarchicalTeam`](hierarchical.md)** — leader delegates to specialists (or nested
  teams) via tools (`usage=ctx.usage`), matching Agno **coordinate** / pydantic-ai
  [agent delegation](https://ai.pydantic.dev/multi-agent-applications/)
- **[`CollaborativeTeam`](collaborative.md)** — shared [`TaskBoard`](api.md) with
  leader create/assign and parallel member claim/complete

## What this library is not

Collaborative mode does **not** yet include peer-to-peer messaging between teammates
or a fully autonomous multi-agent “inbox” loop beyond board claim/assign rounds.

For **sequential**, branching, or stateful pipelines, use
[`pydantic-graph`](https://ai.pydantic.dev/graph/) instead of inventing another workflow type.

| Need | Use |
|------|-----|
| Leader delegates and synthesizes | [`HierarchicalTeam`](hierarchical.md) |
| Shared task board + parallel claim | [`CollaborativeTeam`](collaborative.md) |
| Ordered / branching / stateful flow | [pydantic-graph](https://ai.pydantic.dev/graph/) |
| Peer DM between teammates | Not yet |

## Install

```bash
uv add pydantic-team
# from a checkout:
uv sync --group lint --group dev
```

Requires Python 3.10+.

## Quick example

```python
import asyncio

from pydantic_ai import Agent
from pydantic_team import HierarchicalTeam

researcher = Agent('openai:gpt-4.1', name='researcher', instructions='Research briefly.')
writer = Agent('openai:gpt-4.1', name='writer', instructions='Write a short summary.')

team = HierarchicalTeam(
    leader_model='openai:gpt-4.1',
    members=[researcher, writer],
)

async def main() -> None:
    result = await team.run('Explain agent delegation.')
    print(result.data)
    print(result.usage)

asyncio.run(main())
```

See [Hierarchical teams](hierarchical.md) for nested teams, usage details, and testing.

## Observability

Team orchestration emits OpenTelemetry spans when you opt in with
[`instrument_pydantic_team`][pydantic_team.instrument_pydantic_team]. Pair it with
Logfire (or any OTel backend) and pydantic-ai instrumentation:

```python
import logfire
from pydantic_team import instrument_pydantic_team

logfire.configure(send_to_logfire='if-token-present')  # local unless LOGFIRE_TOKEN
logfire.instrument_pydantic_ai()  # agent + tool spans
instrument_pydantic_team()        # team orchestration spans
```

Span names include `hierarchical.run` / `hierarchical.delegate` and
`collaborative.run` / `.seed` / `.round` / `.synthesize`. Board tool calls are
covered by `instrument_pydantic_ai()` — they are not duplicated as team spans.

## Examples

Live-model try-it scripts in the repo (default model `openai:gpt-5.6-luna`;
optional `PYDANTIC_TEAM_MODEL`). Scripts load `examples/.env` via `python-dotenv`
(e.g. `OPENAI_API_KEY` for `openai:` models) and enable Logfire + both
`instrument_*` calls.

The library depends on [`pydantic-ai-slim`](https://ai.pydantic.dev/install/)
without provider SDKs. Sync the `examples` group to install
`pydantic-ai-slim[openai]`, `logfire`, and `python-dotenv`.

- [`examples/hierarchical_basic.py`](https://github.com/etiqa/pydantic-team/blob/main/examples/hierarchical_basic.py)
- [`examples/collaborative_basic.py`](https://github.com/etiqa/pydantic-team/blob/main/examples/collaborative_basic.py)

```bash
# examples/.env with OPENAI_API_KEY is loaded automatically
uv sync --group examples
uv run examples/hierarchical_basic.py
uv run examples/collaborative_basic.py
```
