"""Hierarchical team: a leader agent delegates to members via tools."""

from __future__ import annotations as _annotations

from collections.abc import Callable, Coroutine, Sequence

from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import RunUsage

from pydantic_team._utils import (
    TeamMember,
    member_tool_description,
    member_tool_name,
    run_member,
    stringify_output,
)
from pydantic_team.base import BaseTeam, TeamResult

_DEFAULT_LEADER_INSTRUCTIONS = (
    'You coordinate specialist team members. '
    'Delegate using the available tools based on each specialist name and description, '
    'then synthesize their results into a final answer.'
)

DelegateTool = Callable[[RunContext[object], str], Coroutine[object, object, str]]


class HierarchicalTeam(BaseTeam[object]):
    """Leader-driven team that registers each member as a delegation tool.

    Mirrors Agno's coordinate mode and pydantic-ai
    [agent delegation](https://ai.pydantic.dev/multi-agent-applications/):
    nested member runs receive `usage=ctx.usage` so tokens aggregate on the leader run.
    """

    def __init__(
        self,
        *,
        members: Sequence[TeamMember],
        leader_agent: Agent[object, object] | None = None,
        leader_model: str | None = None,
        system_prompt_override: str | None = None,
        name: str | None = None,
    ) -> None:
        """Create a hierarchical team.

        Args:
            members: Specialist agents or nested teams (at least one required).
            leader_agent: Existing leader agent. Mutually exclusive with `leader_model`.
            leader_model: Model string used to construct a leader agent when
                `leader_agent` is not provided.
            system_prompt_override: Optional leader instructions / extra system prompt.
            name: Optional team name (used when this team is nested as a member tool).
        """
        if leader_agent is not None and leader_model is not None:
            raise ValueError('Provide leader_agent or leader_model, not both')
        if leader_agent is None and leader_model is None:
            raise ValueError('Provide leader_agent or leader_model')
        if not members:
            raise ValueError('members must be a non-empty sequence')

        self.name = name
        self._members: list[TeamMember] = list(members)

        if leader_agent is not None:
            self._leader = leader_agent
            if system_prompt_override is not None:

                def _leader_override_prompt() -> str:
                    return system_prompt_override

                self._leader.system_prompt(_leader_override_prompt)

        else:
            assert leader_model is not None
            self._leader = Agent(
                leader_model,
                instructions=system_prompt_override or _DEFAULT_LEADER_INSTRUCTIONS,
            )

        self._register_member_tools()

    @property
    def leader(self) -> Agent[object, object]:
        """The coordinating leader agent."""
        return self._leader

    @property
    def members(self) -> Sequence[TeamMember]:
        return self._members

    async def run(self, user_prompt: str, *, usage: RunUsage | None = None) -> TeamResult[object]:
        if usage is None:
            result = await self._leader.run(user_prompt)
        else:
            result = await self._leader.run(user_prompt, usage=usage)
        return TeamResult(data=result.output, usage=result.usage)

    def _register_member_tools(self) -> None:
        for index, member in enumerate(self._members):
            tool_name = member_tool_name(member, index)
            description = member_tool_description(member, tool_name)
            self._leader.tool(self._make_delegate(member, tool_name, description))

    def _make_delegate(
        self,
        member: TeamMember,
        tool_name: str,
        description: str,
    ) -> DelegateTool:
        async def delegate(ctx: RunContext[object], request: str) -> str:
            output = await run_member(member, request, usage=ctx.usage)
            return stringify_output(output)

        delegate.__name__ = tool_name
        delegate.__qualname__ = tool_name
        delegate.__doc__ = description
        return delegate
