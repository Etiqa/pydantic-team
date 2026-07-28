# pydantic-team

[![CI](https://github.com/etiqa/pydantic-team/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/etiqa/pydantic-team/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pydantic-team)](https://pypi.org/project/pydantic-team/)
[![Python](https://img.shields.io/pypi/pyversions/pydantic-team)](https://pypi.org/project/pydantic-team/)

Type-safe team orchestration for [`pydantic-ai`](https://ai.pydantic.dev) Agents.

v1 focuses on **hierarchical** (leader + specialists) teams — the same idea as Agno's
`TeamMode.coordinate` and pydantic-ai
[agent delegation](https://ai.pydantic.dev/multi-agent-applications/), without hiding
usage/token tracking.

For **sequential**, branching, or stateful pipelines, use
[`pydantic-graph`](https://ai.pydantic.dev/graph/) (already pulled in by `pydantic-ai`).

## Install

```bash
uv add pydantic-team
# or from a checkout:
uv sync --group lint --group dev
```

Requires Python 3.10+.

## Documentation

Published docs: [etiqa.github.io/pydantic-team](https://etiqa.github.io/pydantic-team/)

Local build (MkDocs + mkdocstrings):

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
    'openai:gpt-4o',
    name='researcher',
    instructions='Research the topic and return concise notes.',
)
writer = Agent(
    'openai:gpt-4o',
    name='writer',
    instructions='Turn research notes into a short article.',
)

team = HierarchicalTeam(
    leader_model='openai:gpt-4o',
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

## Release

1. Bump `version` in `pyproject.toml` and update `CHANGELOG.md`.
2. Merge to `main` and wait for CI (`check`) to pass.
3. Tag and push (tag must match the package version, e.g. `0.1.1` → `v0.1.1`):

```bash
git tag v0.1.1
git push origin v0.1.1
```

On a `v*` tag, CI publishes to PyPI (Trusted Publisher / environment `release`) and
deploys docs to [GitHub Pages](https://etiqa.github.io/pydantic-team/).

## License

MIT
