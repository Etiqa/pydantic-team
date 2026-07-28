"""Type-safe team orchestration for pydantic-ai Agents."""

from __future__ import annotations as _annotations

from pydantic_team.base import BaseTeam, TeamResult
from pydantic_team.hierarchical import HierarchicalTeam

__version__ = '0.1.0'

__all__ = (
    '__version__',
    'BaseTeam',
    'HierarchicalTeam',
    'TeamResult',
)
