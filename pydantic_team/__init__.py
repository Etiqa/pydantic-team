"""Type-safe team orchestration for pydantic-ai Agents."""

from __future__ import annotations as _annotations

from pydantic_team._instrumentation import instrument_pydantic_team, is_instrumented
from pydantic_team.base import BaseTeam, TeamResult
from pydantic_team.board import BoardMessage, Task, TaskBoard, TaskStatus
from pydantic_team.collaborative import BoardDeps, CollaborativeRun, CollaborativeTeam
from pydantic_team.events import (
    MessagePosted,
    PhaseJoined,
    RunEnded,
    TaskCompleted,
    TasksScheduled,
    TeamEvent,
    TeamTask,
)
from pydantic_team.hierarchical import HierarchicalTeam

__version__ = '0.1.0'

__all__ = (
    '__version__',
    'BaseTeam',
    'BoardDeps',
    'BoardMessage',
    'CollaborativeRun',
    'CollaborativeTeam',
    'HierarchicalTeam',
    'MessagePosted',
    'PhaseJoined',
    'RunEnded',
    'Task',
    'TaskBoard',
    'TaskCompleted',
    'TaskStatus',
    'TasksScheduled',
    'TeamEvent',
    'TeamResult',
    'TeamTask',
    'instrument_pydantic_team',
    'is_instrumented',
)
