"""Basic CollaborativeTeam example with a live model.

The leader creates tasks and assigns them by role; members complete only their
assigned work across parallel rounds (``max_rounds=3``).

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

    researcher = Agent(
        model,
        name='researcher',
        instructions='You are a researcher. Complete only research tasks assigned to you.',
    )
    writer = Agent(
        model,
        name='writer',
        instructions='You are a writer. Complete only writing tasks assigned to you.',
    )

    team = CollaborativeTeam(
        leader_model=model,
        members=[researcher, writer],
        system_prompt_override=(
            'You lead researcher and writer. For every add_task, immediately '
            'assign_task to researcher or writer by role. Never leave tasks unassigned.'
        ),
        max_rounds=3,
    )

    result = await team.run('Draft a short brief on agent teams')
    print(result.data)
    print(result.usage)


if __name__ == '__main__':
    asyncio.run(main())
