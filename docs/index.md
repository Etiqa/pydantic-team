# pydantic-team

Type-safe team orchestration for [`pydantic-ai`](https://ai.pydantic.dev) Agents.

## What this library is

v1 provides **[`HierarchicalTeam`](hierarchical.md)** — a leader agent that delegates to
specialists (or nested teams) via tools, preserving token/usage tracking through
`usage=ctx.usage`.

This matches:

- Agno's **coordinate** team mode
- pydantic-ai [agent delegation](https://ai.pydantic.dev/multi-agent-applications/)

## What this library is not

v1 does **not** implement the “Agent Teams” paradigm with a shared task list, peer-to-peer
messaging between teammates, or autonomous claim/assign loops. That would be a separate
orchestration model.

For **sequential**, branching, or stateful pipelines, use
[`pydantic-graph`](https://ai.pydantic.dev/graph/) instead of inventing another team type.

| Need | Use |
|------|-----|
| Leader delegates and synthesizes | [`HierarchicalTeam`](hierarchical.md) |
| Ordered / branching / stateful flow | [pydantic-graph](https://ai.pydantic.dev/graph/) |
| Shared task board + peer messages | Not in v1 |

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
