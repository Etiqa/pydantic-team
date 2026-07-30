"""CollaborativeTeam Sudoku example with opt-in review (`require_review=True`).

Same puzzle setup as ``collaborative_basic.py``, but completed work goes to
``pending_review`` until the leader (default reviewer) approves or rejects.
The leader may ``assign_reviewer`` to the verifier for delegated QA.

On reject, the task returns to ``needs_revision`` with ``rejection_reason`` on
the board; the assignee reworks and completes again. ``done`` is terminal.

Watch review in:

- **Logfire** — ``approve_task`` / ``reject_task`` / ``assign_reviewer`` tool spans
- **Console** — ``TaskReviewDecided`` events from ``team.iter``

Env:
  PYDANTIC_TEAM_MODEL — optional override (default: ``openai:gpt-5.6-luna``)
  OPENAI_API_KEY — required only when ``PYDANTIC_TEAM_MODEL`` starts with
  ``openai:`` (loaded from ``examples/.env`` if present)

Run from the repo root::

    uv sync --group examples
    uv run examples/collaborative_review.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent

from pydantic_team import (
    CollaborativeTeam,
    MessagePosted,
    PhaseJoined,
    RunEnded,
    TaskReviewDecided,
    TasksScheduled,
    instrument_pydantic_team,
)

_DEFAULT_MODEL = 'openai:gpt-5.6-luna'
_ENV_FILE = Path(__file__).resolve().parent / '.env'

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
    """Run Sudoku with require_review and print TaskReviewDecided events."""
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
            'You solve Sudoku by reasoning only — no external tools beyond the board. '
            'Apply standard Sudoku rules step-by-step to the assigned task. '
            'Put filled cells, candidates, and conclusions in your complete_task result. '
            'If a task is needs_revision, read rejection_reason on the board and rework '
            'before completing again. '
            'If you need the verifier, send_message to verifier (or broadcast with to="*") '
            'and list_messages for replies.'
        ),
    )
    verifier = Agent(
        model,
        name='verifier',
        instructions=(
            'You verify Sudoku progress by reasoning only — no external tools beyond the board. '
            'When you are the reviewer on pending_review tasks, use approve_task or '
            'reject_task(reason) — do not mark work done via complete_task for reviews. '
            'When assigned solver-style work, complete_task as usual. '
            'send_message / list_messages for peer coordination.'
        ),
    )

    team = CollaborativeTeam(
        leader_model=model,
        members=[solver, verifier],
        system_prompt_override=(
            'You lead solver and verifier on one Sudoku puzzle with a review gate. '
            'Break the work into assignable sub-tasks and assign_task by role. '
            'New tasks default to you as reviewer (require_review). Prefer '
            'assign_reviewer to verifier for technical QA when useful. '
            'On pending_review: approve_task if correct, else reject_task with a clear reason. '
            'Never leave tasks unassigned. Only board tools — no invented domain tools.'
        ),
        max_rounds=4,
        max_replans=1,
        dispatch_mode='streaming',
        require_review=True,
    )

    async with team.iter(_SUDOKU_PROMPT) as run:
        async for event in run:
            if isinstance(event, TasksScheduled):
                print('scheduled', [t.kind for t in event.tasks])
            elif isinstance(event, MessagePosted):
                msg = event.message
                print(f'message {msg.sender}->{msg.to}: {msg.body}')
            elif isinstance(event, TaskReviewDecided):
                print(f'review {event.decision} task={event.task.id} status={event.task.status}')
            elif isinstance(event, PhaseJoined):
                print('joined', event.phase, 'incomplete=', event.incomplete)
            elif isinstance(event, RunEnded):
                print(event.result.data)
                print(event.result.usage)


if __name__ == '__main__':
    asyncio.run(main())
