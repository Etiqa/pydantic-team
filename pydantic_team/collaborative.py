"""Collaborative team: shared task board with parallel claim/assign."""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from pydantic_ai import Agent, AgentRunResult, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.usage import RunUsage

from pydantic_team._instrumentation import team_span
from pydantic_team.base import BaseTeam, TeamResult
from pydantic_team.board import Task, TaskBoard, TaskBoardError, TaskStatus
from pydantic_team.events import (
    PhaseJoined,
    RunEnded,
    TaskCompleted,
    TasksScheduled,
    TeamEvent,
    TeamTask,
)

AnyAgent = Agent[object, object]
DispatchMode = Literal['phased', 'streaming']
EmitFn = Callable[[TeamEvent], Awaitable[None]]


def default_leader_instructions(member_ids: Sequence[str]) -> str:
    """Leader instructions that require assign-by-role after each add_task."""
    roster = ', '.join(member_ids)
    return (
        'You lead a collaborative team with a shared task board. '
        f'Teammates (use these exact ids with assign_task): {roster}. '
        'Break the user goal into concrete tasks with add_task, then immediately '
        'assign_task each task to the most suitable teammate by role. '
        'Do not leave tasks open without an assignee. '
        'When asked to synthesize, summarize completed task results.'
    )


def seed_user_prompt(user_prompt: str, member_ids: Sequence[str]) -> str:
    """User prompt for the lead seeding run, including the teammate roster."""
    roster = ', '.join(member_ids)
    return (
        f'{user_prompt}\n\n'
        f'Teammates: {roster}. '
        'Create tasks with add_task and assign each to the right teammate with assign_task.'
    )


def replan_user_prompt(user_prompt: str, member_ids: Sequence[str], board: TaskBoard) -> str:
    """User prompt for a leader replan turn after member rounds."""
    roster = ', '.join(member_ids)
    return (
        f'Team goal: {user_prompt}\n'
        f'Teammates (use these exact ids with assign_task): {roster}.\n'
        'Review the board after member work. '
        'If the goal still needs work, add_task and assign_task for the missing pieces. '
        'If results already cover the goal, do not invent unnecessary tasks.\n'
        f'Current board:\n{_format_board(board)}'
    )


def synthesize_leader_instructions() -> str:
    """Instructions for the toolless synthesize cycle (no board mutation)."""
    return 'Produce a final answer from the completed board snapshot. Do not create, assign, or claim tasks.'


async def _run_agent_cycle(
    agent: AnyAgent,
    prompt: str,
    *,
    deps: BoardDeps,
    team_usage: RunUsage,
) -> AgentRunResult[object]:
    """Run one agent cycle with a fresh usage budget, then fold into team aggregate.

    pydantic-ai ``UsageLimits.request_limit`` applies per ``agent.run`` against the
    ``usage`` object passed into that run. Isolating each cycle avoids one shared
    counter exhausting the whole collaborative orchestration.
    """
    cycle_usage = RunUsage()
    try:
        return await agent.run(prompt, deps=deps, usage=cycle_usage)
    finally:
        team_usage.incr(cycle_usage)


def member_work_prompt(
    user_prompt: str,
    agent_id: str,
    board: TaskBoard,
    *,
    max_assignments: int | None = None,
) -> str:
    """Per-member tick prompt: work assigned tasks only (no cross-role claim).

    Args:
        user_prompt: Team goal text.
        agent_id: Member id whose assignments are listed.
        board: Shared task board snapshot source.
        max_assignments: If set, only the first N incomplete assignments are listed
            for this tick (structural cap; not a soft prompt request).
    """
    assigned = _incomplete_assignments(board, agent_id, max_assignments=max_assignments)
    if assigned:
        mine = '\n'.join(f'- {task.id} [{task.status}] {task.title}' for task in assigned)
    else:
        mine = '(none)'
    return (
        f'Team goal: {user_prompt}\n'
        f'Your agent id is {agent_id!r}. '
        'Complete tasks already assigned to you; do not claim tasks assigned to others. '
        'Only claim an open (unassigned) task if it clearly matches your role, one at a time.\n'
        'Work only on the tasks listed under "Tasks assigned to you" below.\n'
        f'Tasks assigned to you:\n{mine}\n'
        f'Full board:\n{_format_board(board)}'
    )


@dataclass
class BoardDeps:
    """Dependencies injected into leader and member agent runs."""

    board: TaskBoard
    agent_id: str


@dataclass
class CollaborativeRun:
    """Step-by-step collaborative run (inspired by pydantic-graph ``GraphRun``)."""

    leader: AnyAgent
    members: Sequence[AnyAgent]
    member_ids: Sequence[str]
    max_rounds: int
    max_replans: int
    max_assignments_per_tick: int | None
    dispatch_mode: DispatchMode
    run_member: Callable[[AnyAgent, str, TaskBoard, RunUsage], Awaitable[None]]
    user_prompt: str
    _usage: RunUsage
    _board: TaskBoard = field(default_factory=TaskBoard)
    _events: asyncio.Queue[TeamEvent | None] = field(default_factory=lambda: asyncio.Queue[TeamEvent | None]())
    _result: TeamResult[object] | None = None
    _driver: asyncio.Task[None] | None = None
    _error: BaseException | None = None

    @property
    def board(self) -> TaskBoard:
        """Live task board for this run."""
        return self._board

    @property
    def result(self) -> TeamResult[object] | None:
        """Final result once the run has ended; otherwise ``None``."""
        return self._result

    @property
    def usage(self) -> RunUsage:
        """Aggregated usage accumulated during the run."""
        return self._usage

    async def __aenter__(self) -> CollaborativeRun:
        self._driver = asyncio.create_task(self._drive())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        driver = self._driver
        assert driver is not None
        if not driver.done():
            # Drain leftover events so the driver can finish without cancel mid-agent-run.
            while True:
                item = await self._events.get()
                if item is None:
                    break
        await driver
        if self._error is not None and exc is None:
            raise self._error

    def __aiter__(self) -> AsyncIterator[TeamEvent]:
        return self

    async def __anext__(self) -> TeamEvent:
        item = await self._events.get()
        if item is None:
            if self._error is not None:
                raise self._error
            raise StopAsyncIteration
        return item

    async def _emit(self, event: TeamEvent) -> None:
        await self._events.put(event)

    async def _drive(self) -> None:
        try:
            with team_span(
                'collaborative.run',
                max_rounds=self.max_rounds,
                max_replans=self.max_replans,
                dispatch_mode=self.dispatch_mode,
            ):
                if self.dispatch_mode == 'streaming':
                    with team_span('collaborative.dispatch', mode='streaming'):
                        await self._drive_streaming()
                else:
                    await self._drive_phased()
        except BaseException as exc:
            self._error = exc
        finally:
            await self._events.put(None)

    async def _drive_phased(self) -> None:
        lead_deps = BoardDeps(board=self._board, agent_id=_agent_id(self.leader, fallback='leader'))
        leader_id = _agent_id(self.leader, fallback='leader')

        seed = TeamTask(kind='seed', agent_id=leader_id)
        await self._emit(TasksScheduled((seed,)))
        with team_span('collaborative.seed'):
            seed_prompt = seed_user_prompt(self.user_prompt, self.member_ids)
            lead_result = await _run_agent_cycle(
                self.leader,
                seed_prompt,
                deps=lead_deps,
                team_usage=self._usage,
            )
        await self._emit(TaskCompleted(seed))
        await self._emit(PhaseJoined(phase='seed', incomplete=not self._board.is_complete()))

        if self._board.is_complete():
            result = TeamResult(data=lead_result.output, usage=self._usage)
            self._result = result
            await self._emit(RunEnded(result))
            return

        replans_used = 0
        while True:
            await self._run_phased_member_rounds()
            await self._emit(PhaseJoined(phase='members', incomplete=not self._board.is_complete()))
            if self._board.is_complete() or replans_used >= self.max_replans:
                break
            replan = TeamTask(kind='replan', agent_id=leader_id)
            await self._emit(TasksScheduled((replan,)))
            with team_span(
                'collaborative.replan',
                replan=replans_used + 1,
                max_replans=self.max_replans,
            ):
                replan_prompt = replan_user_prompt(self.user_prompt, self.member_ids, self._board)
                await _run_agent_cycle(
                    self.leader,
                    replan_prompt,
                    deps=lead_deps,
                    team_usage=self._usage,
                )
            await self._emit(TaskCompleted(replan))
            await self._emit(PhaseJoined(phase='replan', incomplete=not self._board.is_complete()))
            replans_used += 1

        await self._synthesize(lead_deps, leader_id)

    async def _run_phased_member_rounds(self) -> None:
        rounds = 0
        while rounds < self.max_rounds and not self._board.is_complete():
            scheduled = tuple(
                TeamTask(
                    kind='member_tick',
                    agent_id=_agent_id(member, fallback=f'member-{index}'),
                    task_ids=_assignment_ids(
                        self._board,
                        _agent_id(member, fallback=f'member-{index}'),
                        max_assignments=self.max_assignments_per_tick,
                    ),
                )
                for index, member in enumerate(self.members)
            )
            await self._emit(TasksScheduled(scheduled))
            with team_span('collaborative.round', round=rounds + 1, max_rounds=self.max_rounds):
                await asyncio.gather(
                    *[self.run_member(member, self.user_prompt, self._board, self._usage) for member in self.members]
                )
            for task in scheduled:
                await self._emit(TaskCompleted(task))
            rounds += 1

    async def _drive_streaming(self) -> None:
        lead_deps = BoardDeps(board=self._board, agent_id=_agent_id(self.leader, fallback='leader'))
        members_by_id = {
            _agent_id(member, fallback=f'member-{index}'): member for index, member in enumerate(self.members)
        }
        dispatch = _StreamingDispatch(
            leader=self.leader,
            members_by_id=members_by_id,
            member_ids=self.member_ids,
            max_rounds=self.max_rounds,
            max_replans=self.max_replans,
            max_assignments_per_tick=self.max_assignments_per_tick,
            user_prompt=self.user_prompt,
            run_usage=self._usage,
            board=self._board,
            lead_deps=lead_deps,
            run_member=self.run_member,
            emit=self._emit,
        )
        early = await dispatch.run()
        if early is not None:
            self._result = early
            await self._emit(RunEnded(early))
            return
        await self._synthesize(lead_deps, _agent_id(self.leader, fallback='leader'))

    async def _synthesize(self, lead_deps: BoardDeps, leader_id: str) -> None:
        synth = TeamTask(kind='synthesize', agent_id=leader_id)
        await self._emit(TasksScheduled((synth,)))
        with team_span('collaborative.synthesize'):
            synthesis_prompt = (
                f'Synthesize a final answer for the goal: {self.user_prompt}\n'
                f'Completed board:\n{_format_board(self._board)}'
            )
            # Board tools stay registered on the leader; strip them for this cycle only.
            with self.leader.override(
                tools=[],
                toolsets=[],
                instructions=synthesize_leader_instructions(),
            ):
                final = await _run_agent_cycle(
                    self.leader,
                    synthesis_prompt,
                    deps=lead_deps,
                    team_usage=self._usage,
                )
        await self._emit(TaskCompleted(synth))
        result = TeamResult(data=final.output, usage=self._usage)
        self._result = result
        await self._emit(RunEnded(result))


@dataclass
class _StreamingDispatch:
    """Ready-queue scheduler: member ticks overlap leader seed/replan."""

    leader: AnyAgent
    members_by_id: dict[str, AnyAgent]
    member_ids: Sequence[str]
    max_rounds: int
    max_replans: int
    max_assignments_per_tick: int | None
    user_prompt: str
    run_usage: RunUsage
    board: TaskBoard
    lead_deps: BoardDeps
    run_member: Callable[[AnyAgent, str, TaskBoard, RunUsage], Awaitable[None]]
    emit: EmitFn
    inflight: set[str] = field(default_factory=lambda: set[str]())
    tick_counts: dict[str, int] = field(default_factory=lambda: dict[str, int]())
    member_tasks: set[asyncio.Task[None]] = field(default_factory=lambda: set[asyncio.Task[None]]())
    seed_output: object | None = None
    had_tasks: bool = False
    replans_used: int = 0
    replan_started: bool = False
    leader_task: asyncio.Task[object] | None = None
    _pending_team_tasks: dict[str, TeamTask] = field(default_factory=lambda: dict[str, TeamTask]())
    _members_joined: bool = False

    def __post_init__(self) -> None:
        self.tick_counts = {agent_id: 0 for agent_id in self.members_by_id}

    def _early_result_if_empty_seed(self) -> TeamResult[object] | None:
        """Return seed output when the board stayed empty after seed (skip synthesize)."""
        if not self.had_tasks and self.board.is_complete():
            assert self.seed_output is not None
            return TeamResult(data=self.seed_output, usage=self.run_usage)
        return None

    async def _emit_members_joined_if_needed(self) -> None:
        """Emit PhaseJoined(members) once when the board had work after seed."""
        if self._members_joined or not self.had_tasks:
            return
        await self.emit(PhaseJoined(phase='members', incomplete=not self.board.is_complete()))
        self._members_joined = True

    def _idle_should_finish(self, early: TeamResult[object] | None) -> bool:
        """Whether idle scheduling should return (early result or synthesize/stop)."""
        return early is not None or self.board.is_complete() or self.replans_used >= self.max_replans

    async def run(self) -> TeamResult[object] | None:
        """Drive streaming until synthesize is needed.

        Returns:
            Early ``TeamResult`` when seed left an empty complete board; otherwise ``None``
            so the caller can synthesize.
        """
        leader_id = _agent_id(self.leader, fallback='leader')
        seed = TeamTask(kind='seed', agent_id=leader_id)
        await self.emit(TasksScheduled((seed,)))
        self.leader_task = asyncio.create_task(self._leader_seed())
        while True:
            if self.board.snapshot():
                self.had_tasks = True

            leader_running = await self._collect_finished_leader(seed)
            await self._spawn_ready_members(leader_running=leader_running)

            if leader_running or self.inflight:
                await self._wait_for_progress()
                continue

            await self._emit_members_joined_if_needed()

            early = self._early_result_if_empty_seed()
            if self._idle_should_finish(early):
                return early

            self.replans_used += 1
            self.replan_started = True
            self._members_joined = False
            for agent_id in self.tick_counts:
                self.tick_counts[agent_id] = 0
            replan = TeamTask(kind='replan', agent_id=leader_id)
            await self.emit(TasksScheduled((replan,)))
            self.leader_task = asyncio.create_task(self._leader_replan(self.replans_used, replan))
            await self._wait_for_progress()

    async def _leader_seed(self) -> object:
        with team_span('collaborative.seed'):
            seed_prompt = seed_user_prompt(self.user_prompt, self.member_ids)
            result = await _run_agent_cycle(
                self.leader,
                seed_prompt,
                deps=self.lead_deps,
                team_usage=self.run_usage,
            )
        return result.output

    async def _leader_replan(self, replan_index: int, replan_task: TeamTask) -> None:
        try:
            with team_span(
                'collaborative.replan',
                replan=replan_index,
                max_replans=self.max_replans,
            ):
                replan_prompt = replan_user_prompt(self.user_prompt, self.member_ids, self.board)
                await _run_agent_cycle(
                    self.leader,
                    replan_prompt,
                    deps=self.lead_deps,
                    team_usage=self.run_usage,
                )
        finally:
            await self.emit(TaskCompleted(replan_task))
            await self.emit(PhaseJoined(phase='replan', incomplete=not self.board.is_complete()))

    def _member_has_incomplete_work(self, agent_id: str) -> bool:
        return any(task.assignee == agent_id and task.status is not TaskStatus.DONE for task in self.board.snapshot())

    def _board_has_open_tasks(self) -> bool:
        return any(task.status is TaskStatus.OPEN for task in self.board.snapshot())

    async def _collect_finished_leader(self, seed_task: TeamTask) -> bool:
        leader_running = self.leader_task is not None and not self.leader_task.done()
        if self.leader_task is not None and self.leader_task.done():
            if not self.replan_started and self.seed_output is None:
                self.seed_output = self.leader_task.result()
                await self.emit(TaskCompleted(seed_task))
                await self.emit(PhaseJoined(phase='seed', incomplete=not self.board.is_complete()))
            self.leader_task = None
            self.replan_started = False
            self.board.signal_wakeup()
            return False
        return leader_running

    async def _spawn_ready_members(self, *, leader_running: bool) -> None:
        for agent_id, member in self.members_by_id.items():
            if agent_id in self.inflight:
                continue
            if self.tick_counts[agent_id] >= self.max_rounds:
                continue
            has_assigned = self._member_has_incomplete_work(agent_id)
            can_claim_open = not leader_running and self._board_has_open_tasks()
            if not has_assigned and not can_claim_open:
                continue
            self.had_tasks = True
            self._members_joined = False
            self.inflight.add(agent_id)
            self.tick_counts[agent_id] += 1
            team_task = TeamTask(
                kind='member_tick',
                agent_id=agent_id,
                task_ids=_assignment_ids(self.board, agent_id, max_assignments=self.max_assignments_per_tick),
            )
            self._pending_team_tasks[agent_id] = team_task
            await self.emit(TasksScheduled((team_task,)))
            self.member_tasks.add(asyncio.create_task(self._member_tick(member, agent_id)))

    async def _member_tick(self, member: AnyAgent, agent_id: str) -> None:
        team_task = self._pending_team_tasks[agent_id]
        try:
            with team_span('collaborative.member_tick', agent_id=agent_id):
                await self.run_member(member, self.user_prompt, self.board, self.run_usage)
        finally:
            self.inflight.discard(agent_id)
            await self.emit(TaskCompleted(team_task))
            self._pending_team_tasks.pop(agent_id, None)
            self.board.signal_wakeup()

    async def _wait_for_progress(self) -> None:
        wait_set: set[asyncio.Task[object]] = set()
        wakeup_task = asyncio.create_task(self.board.wait_wakeup())
        wait_set.add(cast(asyncio.Task[object], wakeup_task))
        if self.leader_task is not None:
            wait_set.add(self.leader_task)
        wait_set.update(cast(asyncio.Task[object], task) for task in self.member_tasks)

        done, _pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
        if not wakeup_task.done():
            wakeup_task.cancel()
            try:
                await wakeup_task
            except asyncio.CancelledError:
                pass
        for finished in done:
            if finished in self.member_tasks:
                self.member_tasks.discard(cast(asyncio.Task[None], finished))
                exc = finished.exception()
                if exc is not None:
                    raise exc


class CollaborativeTeam(BaseTeam[object]):
    """Team that coordinates work through a shared [`TaskBoard`][pydantic_team.board.TaskBoard].

    The leader creates and assigns tasks by role; members complete their assigned work
    in parallel (phased rounds or streaming dispatch). When the board remains incomplete,
    the leader may replan (up to ``max_replans``) before synthesizing. Peer messaging is
    not included.
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
        max_replans: int = 0,
        max_assignments_per_tick: int | None = None,
        dispatch_mode: DispatchMode = 'phased',
    ) -> None:
        """Create a collaborative team.

        Args:
            members: Teammate agents (at least one). Nested teams are not supported here.
            leader_agent: Existing leader. Mutually exclusive with `leader_model`.
            leader_model: Model string used to build the leader when `leader_agent` is omitted.
            system_prompt_override: Optional leader instructions / extra system prompt.
            name: Optional team name.
            max_rounds: In ``phased`` mode, parallel member ticks per phase (after seed and
                after each replan). In ``streaming`` mode, max ticks per member per phase.
            max_replans: How many times the leader may replan after incomplete member phases.
                ``0`` preserves seed → work → synthesize with no replan.
            max_assignments_per_tick: If set, each member tick only lists this many incomplete
                assignments (caps work per round without relying on soft prompt wording).
            dispatch_mode: ``phased`` (default) runs seed then member rounds; ``streaming``
                starts member ticks as soon as tasks are assigned (overlap with seed/replan).
        """
        if leader_agent is not None and leader_model is not None:
            raise ValueError('Provide leader_agent or leader_model, not both')
        if leader_agent is None and leader_model is None:
            raise ValueError('Provide leader_agent or leader_model')
        if not members:
            raise ValueError('members must be a non-empty sequence')
        if max_rounds < 1:
            raise ValueError('max_rounds must be >= 1')
        if max_replans < 0:
            raise ValueError('max_replans must be >= 0')
        if max_assignments_per_tick is not None and max_assignments_per_tick < 1:
            raise ValueError('max_assignments_per_tick must be >= 1')
        if dispatch_mode not in ('phased', 'streaming'):
            raise ValueError("dispatch_mode must be 'phased' or 'streaming'")

        self.name = name
        self._max_rounds = max_rounds
        self._max_replans = max_replans
        self._max_assignments_per_tick = max_assignments_per_tick
        self._dispatch_mode: DispatchMode = dispatch_mode
        self._members: list[AnyAgent] = list(members)
        self._member_ids: list[str] = [
            _agent_id(member, fallback=f'member-{index}') for index, member in enumerate(self._members)
        ]

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
                    instructions=system_prompt_override or default_leader_instructions(self._member_ids),
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

    @property
    def dispatch_mode(self) -> DispatchMode:
        """Scheduling mode: ``phased`` or ``streaming``."""
        return self._dispatch_mode

    def iter(self, user_prompt: str, *, usage: RunUsage | None = None) -> CollaborativeRun:
        """Start an observable collaborative run (async context + async iterator)."""
        return CollaborativeRun(
            leader=self._leader,
            members=self._members,
            member_ids=self._member_ids,
            max_rounds=self._max_rounds,
            max_replans=self._max_replans,
            max_assignments_per_tick=self._max_assignments_per_tick,
            dispatch_mode=self._dispatch_mode,
            run_member=self._run_member,
            user_prompt=user_prompt,
            _usage=usage or RunUsage(),
        )

    async def run(self, user_prompt: str, *, usage: RunUsage | None = None) -> TeamResult[object]:
        async with self.iter(user_prompt, usage=usage) as run:
            async for _event in run:
                pass
        assert run.result is not None
        return run.result

    async def _run_member(
        self,
        member: AnyAgent,
        user_prompt: str,
        board: TaskBoard,
        usage: RunUsage,
    ) -> None:
        agent_id = _agent_id(member, fallback='member')
        deps = BoardDeps(board=board, agent_id=agent_id)
        prompt = member_work_prompt(
            user_prompt,
            agent_id,
            board,
            max_assignments=self._max_assignments_per_tick,
        )
        # Member ticks only mutate the board via tools; empty final model text is ignored.
        try:
            await _run_agent_cycle(member, prompt, deps=deps, team_usage=usage)
        except UnexpectedModelBehavior:
            return

    @property
    def member_ids(self) -> Sequence[str]:
        """Stable teammate ids used with ``assign_task`` (agent names)."""
        return self._member_ids


def _incomplete_assignments(
    board: TaskBoard,
    agent_id: str,
    *,
    max_assignments: int | None,
) -> list[Task]:
    assigned = [task for task in board.snapshot() if task.assignee == agent_id and task.status is not TaskStatus.DONE]
    if max_assignments is not None:
        return assigned[:max_assignments]
    return assigned


def _assignment_ids(
    board: TaskBoard,
    agent_id: str,
    *,
    max_assignments: int | None,
) -> tuple[str, ...]:
    return tuple(task.id for task in _incomplete_assignments(board, agent_id, max_assignments=max_assignments))


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
    deps = _board_deps(ctx)
    task = await deps.board.add_task(title, description)
    return f'Created {task.id}: {task.title}'


async def assign_task(ctx: RunContext[object], task_id: str, agent_id: str) -> str:
    """Assign a task to a teammate by agent id/name."""
    deps = _board_deps(ctx)
    try:
        task = await deps.board.assign(task_id, agent_id)
    except TaskBoardError as exc:
        return f'Error: {exc}'
    return f'Assigned {task.id} to {task.assignee}'


async def list_tasks(ctx: RunContext[object], status: str | None = None) -> str:
    """List tasks on the board; optional status filter: open, claimed, done."""
    deps = _board_deps(ctx)
    filter_status: TaskStatus | None = None
    if status is not None and status.strip():
        try:
            filter_status = TaskStatus(status.strip().lower())
        except ValueError:
            return f'Error: invalid status {status!r}; use open, claimed, or done'
    tasks = await deps.board.list_tasks(status=filter_status)
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
