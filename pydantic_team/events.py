"""Observable run events for collaborative team orchestration."""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Literal

from pydantic_team.base import TeamResult

TaskKind = Literal['seed', 'member_tick', 'replan', 'synthesize']
PhaseName = Literal['seed', 'members', 'replan']


@dataclass(frozen=True)
class TeamTask:
    """A unit of scheduled collaborative work (inspired by pydantic-graph GraphTask)."""

    kind: TaskKind
    agent_id: str | None = None
    task_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TasksScheduled:
    """One or more tasks are about to run (or have just been spawned)."""

    tasks: tuple[TeamTask, ...]


@dataclass(frozen=True)
class TaskCompleted:
    """A scheduled task finished (board may have been mutated via tools)."""

    task: TeamTask


@dataclass(frozen=True)
class PhaseJoined:
    """No inflight work remains for this phase (join / barrier)."""

    phase: PhaseName
    incomplete: bool


@dataclass(frozen=True)
class RunEnded:
    """The collaborative run finished with a final result."""

    result: TeamResult[object]


TeamEvent = TasksScheduled | TaskCompleted | PhaseJoined | RunEnded
