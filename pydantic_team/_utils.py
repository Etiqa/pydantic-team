"""Private helpers for team orchestration."""

from __future__ import annotations as _annotations

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage

from pydantic_team.base import BaseTeam, TeamResult

TeamMember = Agent[object, object] | BaseTeam[object]


def member_tool_name(member: TeamMember, index: int) -> str:
    """Return a stable tool name for a team member.

    Prefers `Agent.name` / a nested team's `name` attribute when set; otherwise
    falls back to `member_{index}`.
    """
    name = getattr(member, 'name', None)
    if isinstance(name, str) and name.strip():
        return _sanitize_tool_name(name)
    return f'member_{index}'


def member_tool_description(member: TeamMember, tool_name: str) -> str:
    """Build a tool description so the leader knows when to delegate."""
    description = getattr(member, 'description', None)
    if isinstance(description, str) and description.strip():
        return description
    return f'Delegate work to specialist `{tool_name}`.'


def stringify_output(output: object) -> str:
    """Convert an agent or team output into a prompt-safe string."""
    if isinstance(output, str):
        return output
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    return str(output)


async def run_member(
    member: TeamMember,
    prompt: str,
    *,
    usage: RunUsage | None = None,
) -> object:
    """Run an agent or nested team and return its primary output payload."""
    if isinstance(member, BaseTeam):
        result: TeamResult[object] = await member.run(prompt, usage=usage)
        return result.data
    if usage is None:
        agent_result = await member.run(prompt)
    else:
        agent_result = await member.run(prompt, usage=usage)
    return agent_result.output


def _sanitize_tool_name(name: str) -> str:
    """Normalize a display name into a tool-safe identifier."""
    cleaned = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in name.strip())
    if not cleaned or not any(ch.isalnum() for ch in cleaned):
        return 'member'
    return cleaned
