"""CollaborativeTeam Sudoku example with a live model (pure reasoning).

Demonstrates a leader + solver + verifier team solving one fixed Sudoku puzzle
using **no custom** ``@agent.tool`` — only board tools from pydantic-team
(``add_task`` / ``assign_task`` / ``complete_task``, etc.) and LLM reasoning.

``dispatch_mode='streaming'`` lets members start as soon as tasks are assigned.
``max_rounds=3`` and ``max_replans=1`` give the team room to iterate; there is
no ``max_assignments_per_tick`` starvation.

Each seed / replan / synthesize / member tick is an isolated agent run for usage
limits (default ``request_limit`` applies per cycle); team usage is aggregated.
Synthesize is toolless so the leader cannot keep adding board tasks at the end.

For step-by-step observation without a live model, prefer ``team.iter(...)``
(see docs) — this example uses ``team.run`` for a simple print of the final grid.

Env:
  PYDANTIC_TEAM_MODEL — optional override (default: ``openai:gpt-5.6-luna``)
  OPENAI_API_KEY — required only when ``PYDANTIC_TEAM_MODEL`` starts with
  ``openai:`` (loaded from ``examples/.env`` if present)

Provider SDKs are not part of the core ``pydantic-ai-slim`` dependency. Sync the
``examples`` group (``pydantic-ai-slim[openai]``, ``logfire``, ``python-dotenv``)
before running.

Logfire runs locally by default (``send_to_logfire='if-token-present'``). Set
``LOGFIRE_TOKEN`` or run ``logfire auth`` only if you want cloud export.

Run from the repo root::

    uv sync --group examples
    uv run examples/collaborative_basic.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_team import CollaborativeTeam, instrument_pydantic_team

_DEFAULT_MODEL = 'openai:gpt-5.6-luna'
_ENV_FILE = Path(__file__).resolve().parent / '.env'

# Easy fixed puzzle (`.` = empty). Known solvable with standard Sudoku rules.
_SUDOKU_PROMPT = """\
Solve this Sudoku puzzle. Digits 1–9; each row, column, and 3×3 box must contain
each digit exactly once. Empty cells are marked with `.`.

5 3 . | . 7 . | . . .
6 . . | 1 9 5 | . . .
. 9 8 | . . . | . 6 .
------+-------+------
8 . . | . 6 . | . . 3
4 . . | 8 . 3 | . . 1
7 . . | . 2 . | . . 6
------+-------+------
. 6 . | . . . | 2 8 .
. . . | 4 1 9 | . . 5
. . . | . 8 . | . 7 9

Return the completed 9×9 grid in the same layout (digits only, no `.`).
"""


def _require_api_key(model: str) -> None:
    if model.startswith('openai:') and not os.getenv('OPENAI_API_KEY'):
        raise SystemExit(
            'OPENAI_API_KEY is required when using an openai: model. '
            'Set OPENAI_API_KEY (e.g. in examples/.env), or override '
            'PYDANTIC_TEAM_MODEL for another provider.'
        )


async def main() -> None:
    load_dotenv(_ENV_FILE)
    logfire.configure(send_to_logfire='if-token-present')
    logfire.instrument_pydantic_ai()
    instrument_pydantic_team()

    model = os.getenv('PYDANTIC_TEAM_MODEL', _DEFAULT_MODEL)
    _require_api_key(model)

    solver = Agent(
        model,
        name='solver',
        instructions=(
            'You solve Sudoku by reasoning only — no external tools. '
            'Apply standard Sudoku rules step-by-step to the assigned task. '
            'Put filled cells, candidates, and conclusions in your complete_task result.'
        ),
    )
    verifier = Agent(
        model,
        name='verifier',
        instructions=(
            'You verify Sudoku progress by reasoning only — no external tools. '
            'Check assigned rows/columns/boxes for conflicts or missing singles. '
            'Propose corrections in your complete_task result.'
        ),
    )

    team = CollaborativeTeam(
        leader_model=model,
        members=[solver, verifier],
        system_prompt_override=(
            'You lead solver and verifier on one Sudoku puzzle. '
            'Break the work into assignable sub-tasks (e.g. fill naked singles, '
            'work a box/row group, cross-check consistency). '
            'For every add_task, immediately assign_task to solver or verifier by role. '
            'Never leave tasks unassigned. Do not invent domain tools — only board tools.'
        ),
        max_rounds=3,
        max_replans=1,
        dispatch_mode='streaming',
    )

    result = await team.run(_SUDOKU_PROMPT)
    print(result.data)
    print(result.usage)


if __name__ == '__main__':
    asyncio.run(main())
