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

researcher = Agent('openai:gpt-4o', name='researcher', instructions='Research briefly.')
writer = Agent('openai:gpt-4o', name='writer', instructions='Write a short summary.')

team = HierarchicalTeam(
    leader_model='openai:gpt-4o',
    members=[researcher, writer],
)

async def main() -> None:
    result = await team.run('Explain agent delegation.')
    print(result.data)
    print(result.usage)

asyncio.run(main())
```

See [Hierarchical teams](hierarchical.md) for nested teams, usage details, and testing.
