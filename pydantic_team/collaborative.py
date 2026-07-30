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
    MessagePosted,
    PhaseJoined,
    RunEnded,
    TaskCompleted,
    TaskReviewDecided,
    TasksScheduled,
    TeamEvent,
    TeamTask,
)

AnyAgent = Agent[object, object]
DispatchMode = Literal['phased', 'streaming']
EmitFn = Callable[[TeamEvent], Awaitable[None]]
RunMemberFn = Callable[[AnyAgent, str, TaskBoard, RunUsage, EmitFn], Awaitable[None]]


def default_leader_instructions(member_ids: Sequence[str]) -> str:
    """Leader instructions that require assign-by-role after each add_task."""
    roster = ', '.join(member_ids)
    return (
        'You lead a collaborative team with a shared task board. '
        f'Teammates (use these exact ids with assign_task): {roster}. '
        'Break the user goal into concrete tasks with add_task, then immediately '
        'assign_task each task to the most suitable teammate by role. '
        'Do not leave tasks open without an assignee. '
        'Teammates may message each other directly; use list_messages to observe. '
        'Use assign_reviewer to delegate review; approve_task / reject_task for pending_review. '
        'When asked to synthesize, summarize completed task results and relevant messages.'
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
        f'Current board:\n{_format_board(board)}\n'
        f'Messages:\n{_format_messages(board)}'
    )


def synthesize_leader_instructions() -> str:
    """Instructions for the toolless synthesize cycle (no board mutation)."""
    return (
        'Produce a final answer from the completed board snapshot and peer messages. '
        'Do not create, assign, or claim tasks.'
    )


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
    """Per-member tick prompt: work assigned tasks and pending reviews."""
    assigned = _incomplete_assignments(board, agent_id, max_assignments=max_assignments)
    if assigned:
        mine = '\n'.join(
            (
                f'- {task.id} [{task.status}] {task.title}'
                + (
                    f' prior_result={task.result!r} rejection={task.rejection_reason!r}'
                    if task.status is TaskStatus.NEEDS_REVISION
                    else ''
                )
            )
            for task in assigned
        )
    else:
        mine = '(none)'
    reviews = _pending_reviews(board, agent_id)
    if reviews:
        review_lines = '\n'.join(
            f'- {task.id} [{task.status}] {task.title} result={task.result!r} assignee={task.assignee!r}'
            for task in reviews
        )
    else:
        review_lines = '(none)'
    return (
        f'Team goal: {user_prompt}\n'
        f'Your agent id is {agent_id!r}. '
        'Complete tasks already assigned to you; do not claim tasks assigned to others. '
        'Only claim an open (unassigned) task if it clearly matches your role, one at a time.\n'
        'If you are the reviewer on pending_review tasks, use approve_task or reject_task(reason).\n'
        'You may send_message to teammates (or broadcast with to="*") and list_messages.\n'
        'Work only on the tasks listed under "Tasks assigned to you" below.\n'
        f'Tasks assigned to you:\n{mine}\n'
        f'Tasks awaiting your review:\n{review_lines}\n'
        f'Full board:\n{_format_board(board)}\n'
        f'Messages visible to you:\n{_format_messages(board, agent_id=agent_id)}'
    )


def review_work_prompt(user_prompt: str, agent_id: str, board: TaskBoard) -> str:
    """Prompt for a review-only tick (typically the leader as default reviewer)."""
    reviews = _pending_reviews(board, agent_id)
    if reviews:
        review_lines = '\n'.join(
            f'- {task.id} [{task.status}] {task.title} result={task.result!r} assignee={task.assignee!r}'
            for task in reviews
        )
    else:
        review_lines = '(none)'
    return (
        f'Team goal: {user_prompt}\n'
        f'Your agent id is {agent_id!r}. '
        'Review pending_review tasks: approve_task if acceptable, or reject_task with a clear reason.\n'
        f'Tasks awaiting your review:\n{review_lines}\n'
        f'Full board:\n{_format_board(board)}\n'
        f'Messages:\n{_format_messages(board)}'
    )


def review_leader_instructions() -> str:
    """Instructions for a toolless-except-review leader cycle."""
    return (
        'You are reviewing teammate work on the shared board. '
        'Use approve_task or reject_task on pending_review items only. '
        'Do not create or assign tasks in this cycle.'
    )


@dataclass
class BoardDeps:
    """Dependencies injected into leader and member agent runs."""

    board: TaskBoard
    agent_id: str
    member_ids: tuple[str, ...]
    leader_id: str = 'leader'
    emit: EmitFn | None = None
    require_review: bool = False


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
    run_member: RunMemberFn
    user_prompt: str
    require_review: bool
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
        leader_id = _agent_id(self.leader, fallback='leader')
        lead_deps = BoardDeps(
            board=self._board,
            agent_id=leader_id,
            member_ids=tuple(self.member_ids),
            leader_id=leader_id,
            emit=self._emit,
            require_review=self.require_review,
        )

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
            await self._run_phased_member_rounds(lead_deps, leader_id)
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

    async def _run_phased_member_rounds(self, lead_deps: BoardDeps, leader_id: str) -> None:
        rounds = 0
        while rounds < self.max_rounds and not self._board.is_complete():
            scheduled = tuple(
                TeamTask(
                    kind='member_tick',
                    agent_id=_agent_id(member, fallback=f'member-{index}'),
                    task_ids=_tick_task_ids(
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
                    *[
                        self.run_member(member, self.user_prompt, self._board, self._usage, self._emit)
                        for member in self.members
                    ]
                )
            for task in scheduled:
                await self._emit(TaskCompleted(task))
            await self._run_leader_review_tick(lead_deps, leader_id)
            rounds += 1

    async def _run_leader_review_tick(self, lead_deps: BoardDeps, leader_id: str) -> None:
        """Run one leader review cycle when the leader has pending_review work."""
        if not _pending_reviews(self._board, leader_id):
            return
        review = TeamTask(
            kind='review_tick',
            agent_id=leader_id,
            task_ids=tuple(task.id for task in _pending_reviews(self._board, leader_id)),
        )
        await self._emit(TasksScheduled((review,)))
        with team_span('collaborative.review_tick', agent_id=leader_id):
            with self.leader.override(instructions=review_leader_instructions()):
                try:
                    await _run_agent_cycle(
                        self.leader,
                        review_work_prompt(self.user_prompt, leader_id, self._board),
                        deps=lead_deps,
                        team_usage=self._usage,
                    )
                except UnexpectedModelBehavior:
                    pass
        await self._emit(TaskCompleted(review))

    async def _drive_streaming(self) -> None:
        leader_id = _agent_id(self.leader, fallback='leader')
        lead_deps = BoardDeps(
            board=self._board,
            agent_id=leader_id,
            member_ids=tuple(self.member_ids),
            leader_id=leader_id,
            emit=self._emit,
            require_review=self.require_review,
        )
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
            require_review=self.require_review,
        )
        early = await dispatch.run()
        if early is not None:
            self._result = early
            await self._emit(RunEnded(early))
            return
        await self._synthesize(lead_deps, leader_id)

    async def _synthesize(self, lead_deps: BoardDeps, leader_id: str) -> None:
        synth = TeamTask(kind='synthesize', agent_id=leader_id)
        await self._emit(TasksScheduled((synth,)))
        with team_span('collaborative.synthesize'):
            synthesis_prompt = (
                f'Synthesize a final answer for the goal: {self.user_prompt}\n'
                f'Completed board:\n{_format_board(self._board)}\n'
                f'Messages:\n{_format_messages(self._board)}'
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
    run_member: RunMemberFn
    emit: EmitFn
    require_review: bool = False
    inflight: set[str] = field(default_factory=lambda: set[str]())
    tick_counts: dict[str, int] = field(default_factory=lambda: dict[str, int]())
    member_tasks: set[asyncio.Task[None]] = field(default_factory=lambda: set[asyncio.Task[None]]())
    seed_output: object | None = None
    had_tasks: bool = False
    replans_used: int = 0
    replan_started: bool = False
    leader_task: asyncio.Task[object] | None = None
    leader_review_ticks: int = 0
    _pending_team_tasks: dict[str, TeamTask] = field(default_factory=lambda: dict[str, TeamTask]())
    _members_joined: bool = False
    _leader_review_inflight: bool = False

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
            await self._spawn_leader_review_if_ready(leader_running=leader_running)

            if leader_running or self.inflight or self._leader_review_inflight:
                await self._wait_for_progress()
                continue

            await self._emit_members_joined_if_needed()

            early = self._early_result_if_empty_seed()
            if self._idle_should_finish(early):
                return early

            self.replans_used += 1
            self.replan_started = True
            self._members_joined = False
            self.leader_review_ticks = 0
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
        return bool(_incomplete_assignments(self.board, agent_id, max_assignments=None)) or bool(
            _pending_reviews(self.board, agent_id)
        )

    def _board_has_open_tasks(self) -> bool:
        return any(task.status is TaskStatus.OPEN for task in self.board.snapshot())

    async def _spawn_leader_review_if_ready(self, *, leader_running: bool) -> None:
        if leader_running or self._leader_review_inflight or self.leader_task is not None:
            return
        leader_id = _agent_id(self.leader, fallback='leader')
        if not _pending_reviews(self.board, leader_id):
            return
        if self.leader_review_ticks >= self.max_rounds:
            return
        self.leader_review_ticks += 1
        self._leader_review_inflight = True
        self.had_tasks = True
        review = TeamTask(
            kind='review_tick',
            agent_id=leader_id,
            task_ids=tuple(task.id for task in _pending_reviews(self.board, leader_id)),
        )
        await self.emit(TasksScheduled((review,)))
        self.leader_task = asyncio.create_task(self._leader_review(review))

    async def _leader_review(self, review_task: TeamTask) -> None:
        leader_id = _agent_id(self.leader, fallback='leader')
        try:
            with team_span('collaborative.review_tick', agent_id=leader_id):
                with self.leader.override(instructions=review_leader_instructions()):
                    try:
                        await _run_agent_cycle(
                            self.leader,
                            review_work_prompt(self.user_prompt, leader_id, self.board),
                            deps=self.lead_deps,
                            team_usage=self.run_usage,
                        )
                    except UnexpectedModelBehavior:
                        pass
        finally:
            self._leader_review_inflight = False
            await self.emit(TaskCompleted(review_task))
            self.board.signal_wakeup()

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
                task_ids=_tick_task_ids(self.board, agent_id, max_assignments=self.max_assignments_per_tick),
            )
            self._pending_team_tasks[agent_id] = team_task
            await self.emit(TasksScheduled((team_task,)))
            self.member_tasks.add(asyncio.create_task(self._member_tick(member, agent_id)))

    async def _member_tick(self, member: AnyAgent, agent_id: str) -> None:
        team_task = self._pending_team_tasks[agent_id]
        try:
            with team_span('collaborative.member_tick', agent_id=agent_id):
                await self.run_member(member, self.user_prompt, self.board, self.run_usage, self.emit)
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
    the leader may replan (up to ``max_replans``) before synthesizing. Members may
    message each other directly via board tools (``send_message`` / ``list_messages``).
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
        require_review: bool = False,
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
            require_review: When True, new tasks get ``reviewer=leader`` so ``complete`` goes
                to ``pending_review`` until approve/reject. When False, tasks complete to
                ``done`` unless ``assign_reviewer`` sets a reviewer.
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
        self._require_review = require_review
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
            require_review=self._require_review,
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
        emit: EmitFn,
    ) -> None:
        agent_id = _agent_id(member, fallback='member')
        leader_id = _agent_id(self._leader, fallback='leader')
        deps = BoardDeps(
            board=board,
            agent_id=agent_id,
            member_ids=tuple(self._member_ids),
            leader_id=leader_id,
            emit=emit,
            require_review=self._require_review,
        )
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
    assigned = [
        task
        for task in board.snapshot()
        if task.assignee == agent_id and task.status in (TaskStatus.CLAIMED, TaskStatus.NEEDS_REVISION)
    ]
    if max_assignments is not None:
        return assigned[:max_assignments]
    return assigned


def _pending_reviews(board: TaskBoard, reviewer_id: str) -> list[Task]:
    return [
        task for task in board.snapshot() if task.reviewer == reviewer_id and task.status is TaskStatus.PENDING_REVIEW
    ]


def _tick_task_ids(
    board: TaskBoard,
    agent_id: str,
    *,
    max_assignments: int | None,
) -> tuple[str, ...]:
    assigned = _incomplete_assignments(board, agent_id, max_assignments=max_assignments)
    reviews = _pending_reviews(board, agent_id)
    return tuple(task.id for task in assigned) + tuple(task.id for task in reviews)


def _agent_id(agent: AnyAgent, *, fallback: str) -> str:
    name = agent.name
    if isinstance(name, str) and name.strip():
        return name.strip()
    return fallback


def _format_board(board: TaskBoard) -> str:
    lines = [
        (
            f'- {task.id} [{task.status}] {task.title} '
            f'(assignee={task.assignee!r} reviewer={task.reviewer!r}) '
            f'result={task.result!r} rejection={task.rejection_reason!r}'
        )
        for task in board.snapshot()
    ]
    return '\n'.join(lines) if lines else '(empty)'


def _format_messages(board: TaskBoard, *, agent_id: str | None = None) -> str:
    messages = board.messages_snapshot()
    if agent_id is not None:
        messages = [
            message for message in messages if message.to == agent_id or message.to == '*' or message.sender == agent_id
        ]
    if not messages:
        return '(none)'
    lines: list[str] = []
    for message in messages:
        task_bit = f' task={message.task_id}' if message.task_id else ''
        lines.append(f'- {message.id} {message.sender}->{message.to}{task_bit}: {message.body}')
    return '\n'.join(lines)


def _board_deps(ctx: RunContext[object]) -> BoardDeps:
    deps = ctx.deps
    if not isinstance(deps, BoardDeps):
        raise TypeError(f'expected BoardDeps, got {type(deps)!r}')
    return deps


async def add_task(ctx: RunContext[object], title: str, description: str = '') -> str:
    """Add an open task to the shared board."""
    deps = _board_deps(ctx)
    reviewer = deps.leader_id if deps.require_review else None
    task = await deps.board.add_task(title, description, reviewer=reviewer)
    if reviewer is not None:
        return f'Created {task.id}: {task.title} (reviewer={reviewer})'
    return f'Created {task.id}: {task.title}'


async def assign_task(ctx: RunContext[object], task_id: str, agent_id: str) -> str:
    """Assign a task to a teammate by agent id/name."""
    deps = _board_deps(ctx)
    try:
        task = await deps.board.assign(task_id, agent_id)
    except TaskBoardError as exc:
        return f'Error: {exc}'
    return f'Assigned {task.id} to {task.assignee}'


async def assign_reviewer(ctx: RunContext[object], task_id: str, reviewer_id: str) -> str:
    """Set the reviewer for a task (leader or teammate id)."""
    deps = _board_deps(ctx)
    rid = reviewer_id.strip()
    allowed = {deps.leader_id, *deps.member_ids}
    if rid not in allowed:
        return f'Error: unknown reviewer {rid!r}; use one of {sorted(allowed)}'
    try:
        task = await deps.board.assign_reviewer(task_id, rid)
    except TaskBoardError as exc:
        return f'Error: {exc}'
    return f'Reviewer for {task.id} set to {task.reviewer}'


async def list_tasks(ctx: RunContext[object], status: str | None = None) -> str:
    """List tasks; optional status: open, claimed, pending_review, needs_revision, done."""
    deps = _board_deps(ctx)
    filter_status: TaskStatus | None = None
    if status is not None and status.strip():
        try:
            filter_status = TaskStatus(status.strip().lower())
        except ValueError:
            return f'Error: invalid status {status!r}; use open, claimed, pending_review, needs_revision, or done'
    tasks = await deps.board.list_tasks(status=filter_status)
    if not tasks:
        return 'No tasks'
    return '\n'.join(
        (
            f'{task.id} [{task.status}] {task.title} assignee={task.assignee!r} '
            f'reviewer={task.reviewer!r} result={task.result!r} rejection={task.rejection_reason!r}'
        )
        for task in tasks
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
    """Submit completed work (done, or pending_review when a reviewer is set)."""
    deps = _board_deps(ctx)
    try:
        task = await deps.board.complete(task_id, result=result, agent_id=deps.agent_id)
    except TaskBoardError as exc:
        return f'Error: {exc}'
    if task.status is TaskStatus.PENDING_REVIEW:
        return f'Submitted {task.id} for review by {task.reviewer!r} with result={task.result!r}'
    return f'Completed {task.id} with result={task.result!r}'


async def approve_task(ctx: RunContext[object], task_id: str) -> str:
    """Approve a pending_review task (reviewer only)."""
    deps = _board_deps(ctx)
    try:
        task = await deps.board.approve(task_id, agent_id=deps.agent_id)
    except TaskBoardError as exc:
        return f'Error: {exc}'
    if deps.emit is not None:
        await deps.emit(TaskReviewDecided(task, 'approved'))
    return f'Approved {task.id}'


async def reject_task(ctx: RunContext[object], task_id: str, reason: str) -> str:
    """Reject a pending_review task with a reason (reviewer only)."""
    deps = _board_deps(ctx)
    try:
        task = await deps.board.reject(task_id, reason=reason, agent_id=deps.agent_id)
    except TaskBoardError as exc:
        return f'Error: {exc}'
    if deps.emit is not None:
        await deps.emit(TaskReviewDecided(task, 'rejected'))
    return f'Rejected {task.id}: {task.rejection_reason}'


async def send_message(ctx: RunContext[object], to: str, body: str, task_id: str = '') -> str:
    """Send a direct or broadcast message to teammates (members only)."""
    deps = _board_deps(ctx)
    recipient = to.strip()
    if not recipient:
        return 'Error: to must be a teammate id or "*"'
    if recipient == deps.agent_id:
        return 'Error: cannot send a message to yourself'
    if recipient != '*' and recipient not in deps.member_ids:
        return f'Error: unknown teammate {recipient!r}; use one of {list(deps.member_ids)} or "*"'
    link = task_id.strip() or None
    try:
        message = await deps.board.post_message(deps.agent_id, recipient, body, task_id=link)
    except TaskBoardError as exc:
        return f'Error: {exc}'
    if deps.emit is not None:
        await deps.emit(MessagePosted(message))
    return f'Sent {message.id} to {message.to}'


async def list_messages(ctx: RunContext[object]) -> str:
    """List peer messages visible to you (members) or the full log (leader)."""
    deps = _board_deps(ctx)
    if deps.agent_id in deps.member_ids:
        messages = await deps.board.list_messages(agent_id=deps.agent_id)
    else:
        messages = await deps.board.list_messages()
    if not messages:
        return 'No messages'
    return '\n'.join(
        (
            f'{message.id} {message.sender}->{message.to}'
            f'{f" task={message.task_id}" if message.task_id else ""}: {message.body}'
        )
        for message in messages
    )


def _register_leader_tools(agent: AnyAgent) -> None:
    agent.tool(add_task)
    agent.tool(assign_task)
    agent.tool(assign_reviewer)
    agent.tool(list_tasks)
    agent.tool(list_messages)
    agent.tool(approve_task)
    agent.tool(reject_task)


def _register_member_tools(agent: AnyAgent) -> None:
    agent.tool(list_tasks)
    agent.tool(claim_task)
    agent.tool(complete_task)
    agent.tool(approve_task)
    agent.tool(reject_task)
    agent.tool(send_message)
    agent.tool(list_messages)
