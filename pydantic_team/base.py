"""Core team types shared by all team strategies."""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from pydantic_ai.usage import RunUsage
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from pydantic_team._utils import TeamMember

OutputT = TypeVar('OutputT', default=object)
"""Type of the final team output stored on [`TeamResult.data`][pydantic_team.base.TeamResult]."""


@dataclass
class TeamResult(Generic[OutputT]):
    """Outcome of a team run.

    Attributes:
        data: Final output produced by the team (leader output for hierarchical teams).
        usage: Aggregated token/request usage for all agents involved in the run.
    """

    data: OutputT
    usage: RunUsage


class BaseTeam(ABC, Generic[OutputT]):
    """Abstract base for team orchestration strategies."""

    @property
    @abstractmethod
    def members(self) -> Sequence[TeamMember]:
        """Agents or nested teams that participate in this team."""

    @abstractmethod
    async def run(self, user_prompt: str, *, usage: RunUsage | None = None) -> TeamResult[OutputT]:
        """Execute the team against `user_prompt`.

        Args:
            user_prompt: User input for the team run.
            usage: Optional usage accumulator shared with nested agent runs.

        Returns:
            A [`TeamResult`][pydantic_team.base.TeamResult] with final data and aggregated usage.
        """
