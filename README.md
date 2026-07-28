# pydantic-team

Type-safe team orchestration for [`pydantic-ai`](https://ai.pydantic.dev) Agents.

v1 provides **hierarchical** and **collaborative** teams for
[`pydantic-ai`](https://ai.pydantic.dev) Agents:

- `HierarchicalTeam` — leader delegates via tools (Agno coordinate / agent delegation)
- `CollaborativeTeam` — shared task board with parallel claim/assign

For **sequential**, branching, or stateful pipelines, use
[`pydantic-graph`](https://ai.pydantic.dev/graph/) (already pulled in by `pydantic-ai-slim`).

## Install

```bash
uv add pydantic-team
# or from a checkout:
uv sync --group lint --group dev
```

Depends on [`pydantic-ai-slim`](https://ai.pydantic.dev/install/) (core agents only — no provider SDKs).
Install a provider extra when you need a live model, e.g. `pydantic-ai-slim[openai]`.

Requires Python 3.10+.

## Documentation

Full guides and API reference (MkDocs + mkdocstrings):

```bash
uv sync --group docs
make docs-serve   # http://127.0.0.1:8000
make docs         # build into site/
```

## HierarchicalTeam

The leader registers each member as a tool and passes `usage=ctx.usage` on nested runs
so `TeamResult.usage` includes every agent involved.

```python
import asyncio

from pydantic_ai import Agent
from pydantic_team import HierarchicalTeam

researcher = Agent(
    'openai:gpt-4.1',
    name='researcher',
    instructions='Research the topic and return concise notes.',
)
writer = Agent(
    'openai:gpt-4.1',
    name='writer',
    instructions='Turn research notes into a short article.',
)

team = HierarchicalTeam(
    leader_model='openai:gpt-4.1',
    members=[researcher, writer],
    system_prompt_override=(
        'Delegate to researcher or writer based on the task, then synthesize a final answer.'
    ),
)


async def main() -> None:
    result = await team.run('Explain pydantic-ai agent delegation briefly.')
    print(result.data)
    print(result.usage)


asyncio.run(main())
```

You can also pass an existing `leader_agent=` instead of `leader_model=`. Nested
`HierarchicalTeam` instances are valid members (set `name=` for a clear tool id).

See the [hierarchical teams guide](docs/hierarchical.md) for nested teams, usage
details, and `TestModel` testing.

## CollaborativeTeam

Shared [`TaskBoard`](docs/collaborative.md): the leader creates tasks and **assigns
them by role**; members complete their assigned work in parallel rounds
(`max_rounds`). Peer messaging is not included yet.

```python
from pydantic_ai import Agent
from pydantic_team import CollaborativeTeam

researcher = Agent(
    'openai:gpt-4.1',
    name='researcher',
    instructions='Complete only research tasks assigned to you.',
)
writer = Agent(
    'openai:gpt-4.1',
    name='writer',
    instructions='Complete only writing tasks assigned to you.',
)

team = CollaborativeTeam(
    leader_model='openai:gpt-4.1',
    members=[researcher, writer],
    max_rounds=3,
)
result = await team.run('Draft a short brief on agent teams')
```

### Result type

```python
from pydantic_team import TeamResult

# result: TeamResult
# result.data  — final leader output
# result.usage — aggregated RunUsage (requests + tokens)
```

## Sequential / complex control flow

Use [`pydantic-graph`](https://ai.pydantic.dev/graph/) when you need an ordered pipeline,
branches, loops, or shared state. This library intentionally does **not** reimplement that.

## Examples

Runnable scripts (live model API — not part of the test suite). Core deps stay
`pydantic-ai-slim` only; the `examples` group pulls in the OpenAI provider extra,
`logfire`, and `python-dotenv`. Examples call
`logfire.configure(send_to_logfire='if-token-present')`,
`logfire.instrument_pydantic_ai()`, and `instrument_pydantic_team()` so spans
print locally without auth; set `LOGFIRE_TOKEN` or run `logfire auth` for cloud.

```bash
# Put OPENAI_API_KEY in examples/.env (auto-loaded) or export it.
# optional: export PYDANTIC_TEAM_MODEL=openai:gpt-5.6-luna
uv sync --group examples
uv run examples/hierarchical_basic.py
uv run examples/collaborative_basic.py
# one-shot without a prior sync:
# uv run --group examples examples/hierarchical_basic.py
```

## Development

```bash
make install      # uv sync + pre-commit
make format
make lint
make typecheck
make test         # pytest with --cov-fail-under=100
make ci           # format + lint + test
make ci-strict    # lint + typecheck + test
make docs         # MkDocs build (needs --group docs)
make docs-serve
```

Unit tests use pydantic-ai `TestModel` only — no live LLM calls.

## License

MIT
