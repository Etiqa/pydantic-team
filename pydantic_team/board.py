"""In-process shared task board for collaborative teams."""

from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass, field, replace
from enum import Enum


class TaskStatus(str, Enum):
    """Lifecycle status of a board task."""

    OPEN = 'open'
    CLAIMED = 'claimed'
    DONE = 'done'


@dataclass(frozen=True)
class Task:
    """A unit of work on the shared board."""

    id: str
    title: str
    description: str = ''
    status: TaskStatus = TaskStatus.OPEN
    assignee: str | None = None
    result: str | None = None


class TaskBoardError(Exception):
    """Base error for task board operations."""


class TaskNotFoundError(TaskBoardError):
    """Raised when a task id is not on the board."""


class TaskClaimError(TaskBoardError):
    """Raised when claim, assign, or complete is not allowed."""


@dataclass
class TaskBoard:
    """Thread-safe in-process task list shared by a collaborative team."""

    _tasks: dict[str, Task] = field(default_factory=lambda: {})
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _counter: int = 0
    _wakeup: asyncio.Event = field(default_factory=asyncio.Event)
    _wakeup_agents: set[str] = field(default_factory=lambda: set[str]())

    async def add_task(self, title: str, description: str = '') -> Task:
        """Create an open task and return its snapshot."""
        async with self._lock:
            self._counter += 1
            task = Task(id=f'task-{self._counter}', title=title, description=description)
            self._tasks[task.id] = task
            return task

    async def list_tasks(self, status: TaskStatus | None = None) -> list[Task]:
        """Return task snapshots, optionally filtered by status."""
        async with self._lock:
            tasks = list(self._tasks.values())
        if status is not None:
            tasks = [task for task in tasks if task.status is status]
        return tasks

    async def claim(self, task_id: str, agent_id: str) -> Task:
        """Atomically claim an open task for `agent_id`."""
        async with self._lock:
            task = self._require(task_id)
            if task.status is not TaskStatus.OPEN:
                raise TaskClaimError(f'task {task_id!r} is not open (status={task.status})')
            updated = replace(task, status=TaskStatus.CLAIMED, assignee=agent_id)
            self._tasks[task_id] = updated
        self.signal_wakeup(agent_id)
        return updated

    async def assign(self, task_id: str, agent_id: str) -> Task:
        """Force-assign a non-done task to `agent_id` (lead operation)."""
        async with self._lock:
            task = self._require(task_id)
            if task.status is TaskStatus.DONE:
                raise TaskClaimError(f'task {task_id!r} is already done')
            updated = replace(task, status=TaskStatus.CLAIMED, assignee=agent_id)
            self._tasks[task_id] = updated
        self.signal_wakeup(agent_id)
        return updated

    async def complete(self, task_id: str, *, result: str, agent_id: str) -> Task:
        """Mark a claimed task done; only the assignee may complete it."""
        async with self._lock:
            task = self._require(task_id)
            if task.status is not TaskStatus.CLAIMED:
                raise TaskClaimError(f'task {task_id!r} is not claimed (status={task.status})')
            if task.assignee != agent_id:
                raise TaskClaimError(f'task {task_id!r} is assigned to {task.assignee!r}, not {agent_id!r}')
            updated = replace(task, status=TaskStatus.DONE, result=result)
            self._tasks[task_id] = updated
        # Wake the scheduler so it can drain, replan, or synthesize.
        self.signal_wakeup()
        return updated

    def is_complete(self) -> bool:
        """Return True when there are no tasks or every task is done."""
        if not self._tasks:
            return True
        return all(task.status is TaskStatus.DONE for task in self._tasks.values())

    def snapshot(self) -> list[Task]:
        """Return a stable list copy of all tasks (asyncio-safe between awaits)."""
        return list(self._tasks.values())

    def signal_wakeup(self, agent_id: str | None = None) -> None:
        """Wake waiters; optionally record which agent gained work."""
        if agent_id is not None:
            self._wakeup_agents.add(agent_id)
        self._wakeup.set()

    async def wait_wakeup(self) -> set[str]:
        """Block until ``signal_wakeup``; return agent ids recorded since last wait."""
        await self._wakeup.wait()
        self._wakeup.clear()
        agents = set(self._wakeup_agents)
        self._wakeup_agents.clear()
        return agents

    def _require(self, task_id: str) -> Task:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise TaskNotFoundError(f'unknown task id {task_id!r}') from exc
