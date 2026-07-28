"""Collaborative team: shared task board with parallel claim/assign."""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import RunUsage

from pydantic_team.base import BaseTeam, TeamResult
from pydantic_team.board import TaskBoard, TaskBoardError, TaskStatus

_DEFAULT_LEADER_INSTRUCTIONS = (
    'You lead a collaborative team with a shared task board. '
    'Break the user goal into concrete tasks with add_task, optionally assign_task, '
    'then wait for teammates. When asked to synthesize, summarize completed task results.'
)

AnyAgent = Agent[object, object]


@dataclass
class BoardDeps:
    """Dependencies injected into leader and member agent runs."""

    board: TaskBoard
    agent_id: str


class CollaborativeTeam(BaseTeam[object]):
    """Team that coordinates work through a shared [`TaskBoard`][pydantic_team.board.TaskBoard].

    The leader creates/assigns tasks; members claim and complete them in parallel rounds.
    Peer messaging is not included in this version.
    """

    def __init__(
        self,
        *,
        members: Sequence[AnyAgent],
        leader_agent: AnyAgent | None = None,
        leader_model: str | None = None,
        system_prompt_override: str | None = None,
        name: str | None = None,
        max_rounds: int = 3,
    ) -> None:
        """Create a collaborative team.

        Args:
            members: Teammate agents (at least one). Nested teams are not supported here.
            leader_agent: Existing leader. Mutually exclusive with `leader_model`.
            leader_model: Model string used to build the leader when `leader_agent` is omitted.
            system_prompt_override: Optional leader instructions / extra system prompt.
            name: Optional team name.
            max_rounds: Parallel member ticks after the leader seeds the board.
        """
        if leader_agent is not None and leader_model is not None:
            raise ValueError('Provide leader_agent or leader_model, not both')
        if leader_agent is None and leader_model is None:
            raise ValueError('Provide leader_agent or leader_model')
        if not members:
            raise ValueError('members must be a non-empty sequence')
        if max_rounds < 1:
            raise ValueError('max_rounds must be >= 1')

        self.name = name
        self._max_rounds = max_rounds
        self._members: list[AnyAgent] = list(members)

        if leader_agent is not None:
            self._leader: AnyAgent = leader_agent
            if system_prompt_override is not None:

                def _leader_override_prompt() -> str:
                    return system_prompt_override

                self._leader.system_prompt(_leader_override_prompt)
        else:
            assert leader_model is not None
            self._leader = cast(
                AnyAgent,
                Agent(
                    leader_model,
                    name='leader',
                    instructions=system_prompt_override or _DEFAULT_LEADER_INSTRUCTIONS,
                    deps_type=BoardDeps,
                ),
            )

        _register_leader_tools(self._leader)
        for member in self._members:
            _register_member_tools(member)

    @property
    def leader(self) -> AnyAgent:
        """The team lead agent."""
        return self._leader

    @property
    def members(self) -> Sequence[AnyAgent]:
        return self._members

    async def run(self, user_prompt: str, *, usage: RunUsage | None = None) -> TeamResult[object]:
        run_usage = usage or RunUsage()
        board = TaskBoard()
        lead_deps = BoardDeps(board=board, agent_id=_agent_id(self._leader, fallback='leader'))

        lead_result = await self._leader.run(user_prompt, deps=lead_deps, usage=run_usage)
        if board.is_complete():
            return TeamResult(data=lead_result.output, usage=run_usage)

        member_prompt = (
            f'Team goal: {user_prompt}\nClaim open tasks you can handle, complete them with concise results.'
        )
        rounds = 0
        while rounds < self._max_rounds and not board.is_complete():
            await asyncio.gather(
                *[self._run_member(member, member_prompt, board, run_usage) for member in self._members]
            )
            rounds += 1

        synthesis_prompt = (
            f'Synthesize a final answer for the goal: {user_prompt}\nCompleted board:\n{_format_board(board)}'
        )
        final = await self._leader.run(synthesis_prompt, deps=lead_deps, usage=run_usage)
        return TeamResult(data=final.output, usage=run_usage)

    async def _run_member(
        self,
        member: AnyAgent,
        prompt: str,
        board: TaskBoard,
        usage: RunUsage,
    ) -> None:
        deps = BoardDeps(board=board, agent_id=_agent_id(member, fallback='member'))
        await member.run(prompt, deps=deps, usage=usage)


def _agent_id(agent: AnyAgent, *, fallback: str) -> str:
    name = agent.name
    if isinstance(name, str) and name.strip():
        return name.strip()
    return fallback


def _format_board(board: TaskBoard) -> str:
    lines = [
        f'- {task.id} [{task.status}] {task.title} (assignee={task.assignee!r}) result={task.result!r}'
        for task in board.snapshot()
    ]
    return '\n'.join(lines) if lines else '(empty)'


def _board_deps(ctx: RunContext[object]) -> BoardDeps:
    deps = ctx.deps
    if not isinstance(deps, BoardDeps):
        raise TypeError(f'expected BoardDeps, got {type(deps)!r}')
    return deps


async def add_task(ctx: RunContext[object], title: str, description: str = '') -> str:
    """Add an open task to the shared board."""
    task = await _board_deps(ctx).board.add_task(title, description)
    return f'Created {task.id}: {task.title}'


async def assign_task(ctx: RunContext[object], task_id: str, agent_id: str) -> str:
    """Assign a task to a teammate by agent id/name."""
    try:
        task = await _board_deps(ctx).board.assign(task_id, agent_id)
    except TaskBoardError as exc:
        return f'Error: {exc}'
    return f'Assigned {task.id} to {task.assignee}'


async def list_tasks(ctx: RunContext[object], status: str | None = None) -> str:
    """List tasks on the board; optional status filter: open, claimed, done."""
    filter_status: TaskStatus | None = None
    if status is not None and status.strip():
        try:
            filter_status = TaskStatus(status.strip().lower())
        except ValueError:
            return f'Error: invalid status {status!r}; use open, claimed, or done'
    tasks = await _board_deps(ctx).board.list_tasks(status=filter_status)
    if not tasks:
        return 'No tasks'
    return '\n'.join(
        f'{task.id} [{task.status}] {task.title} assignee={task.assignee!r} result={task.result!r}' for task in tasks
    )


async def claim_task(ctx: RunContext[object], task_id: str) -> str:
    """Claim an open task for yourself."""
    deps = _board_deps(ctx)
    try:
        task = await deps.board.claim(task_id, deps.agent_id)
    except TaskBoardError as exc:
        return f'Error: {exc}'
    return f'Claimed {task.id}: {task.title}'


async def complete_task(ctx: RunContext[object], task_id: str, result: str) -> str:
    """Mark a task you claimed as done, with a short result."""
    deps = _board_deps(ctx)
    try:
        task = await deps.board.complete(task_id, result=result, agent_id=deps.agent_id)
    except TaskBoardError as exc:
        return f'Error: {exc}'
    return f'Completed {task.id} with result={task.result!r}'


def _register_leader_tools(agent: AnyAgent) -> None:
    agent.tool(add_task)
    agent.tool(assign_task)
    agent.tool(list_tasks)


def _register_member_tools(agent: AnyAgent) -> None:
    agent.tool(list_tasks)
    agent.tool(claim_task)
    agent.tool(complete_task)
