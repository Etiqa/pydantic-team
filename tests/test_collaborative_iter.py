"""Tests for CollaborativeTeam.iter and team run events."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_team import CollaborativeTeam, TeamResult
from pydantic_team.board import TaskStatus
from pydantic_team.collaborative import BoardDeps
from pydantic_team.events import (
    PhaseJoined,
    RunEnded,
    TaskCompleted,
    TasksScheduled,
    TeamTask,
)


@dataclass
class _FakeRunResult:
    output: object


def _deps_from_kwargs(kwargs: dict[str, object]) -> BoardDeps | None:
    deps = kwargs.get('deps')
    return deps if isinstance(deps, BoardDeps) else None


def _usage_from_kwargs(kwargs: dict[str, object]) -> RunUsage | None:
    usage = kwargs.get('usage')
    return usage if isinstance(usage, RunUsage) else None


async def test_iter_empty_seed_ends_without_member_ticks() -> None:
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1)

    events: list[object] = []
    with leader.override(model=TestModel(call_tools=[], custom_output_text='solo')):
        async with team.iter('Nothing to split') as run:
            async for event in run:
                events.append(event)
            assert run.result is not None
            assert run.result.data == 'solo'
            assert run.board.is_complete()
            assert run.usage.requests >= 1

    assert any(isinstance(e, TasksScheduled) and e.tasks[0].kind == 'seed' for e in events)
    assert any(isinstance(e, TaskCompleted) and e.task.kind == 'seed' for e in events)
    assert any(isinstance(e, PhaseJoined) and e.phase == 'seed' for e in events)
    assert not any(isinstance(e, TasksScheduled) and any(t.kind == 'member_tick' for t in e.tasks) for e in events)
    assert isinstance(events[-1], RunEnded)
    assert events[-1].result.data == 'solo'


async def test_run_delegates_to_iter() -> None:
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1)

    with leader.override(model=TestModel(call_tools=[], custom_output_text='via-run')):
        with worker.override(model=TestModel(custom_output_text='unused')):
            result = await team.run('Nothing to split')

    assert isinstance(result, TeamResult)
    assert result.data == 'via-run'


async def test_iter_streaming_schedules_member_during_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    member_started = asyncio.Event()
    saw_member_scheduled_during_seed = asyncio.Event()

    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_agent=leader,
        members=[worker],
        max_rounds=2,
        dispatch_mode='streaming',
    )

    leader_calls = 0

    async def _seed_assigns_then_waits(user_prompt: str, **kwargs: object) -> _FakeRunResult:
        nonlocal leader_calls
        leader_calls += 1
        deps = _deps_from_kwargs(kwargs)
        if leader_calls == 1 and deps is not None:
            task = await deps.board.add_task('Overlapping work')
            await deps.board.assign(task.id, 'worker')
            await asyncio.wait_for(member_started.wait(), timeout=2)
            return _FakeRunResult(output='seeded')
        result = await Agent.run(
            leader,
            user_prompt,
            deps=deps,
            usage=_usage_from_kwargs(kwargs),
        )
        return _FakeRunResult(output=result.output)

    monkeypatch.setattr(leader, 'run', _seed_assigns_then_waits)

    async def _worker_run(user_prompt: str, **kwargs: object) -> object:
        member_started.set()
        return await Agent.run(
            worker,
            user_prompt,
            deps=_deps_from_kwargs(kwargs),
            usage=_usage_from_kwargs(kwargs),
        )

    monkeypatch.setattr(worker, 'run', _worker_run)

    seed_done = False
    with worker.override(model=TestModel(custom_output_text='worked')):
        async with team.iter('Ship overlapping work') as run:
            async for event in run:
                if isinstance(event, TaskCompleted) and event.task.kind == 'seed':
                    seed_done = True
                if (
                    not seed_done
                    and isinstance(event, TasksScheduled)
                    and any(t.kind == 'member_tick' for t in event.tasks)
                ):
                    saw_member_scheduled_during_seed.set()

    assert saw_member_scheduled_during_seed.is_set()


async def test_iter_completed_board_joins_then_synthesizes(monkeypatch: pytest.MonkeyPatch) -> None:
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_agent=leader,
        members=[worker],
        max_rounds=2,
        dispatch_mode='streaming',
    )

    leader_calls = 0

    async def _seed_one_task(user_prompt: str, **kwargs: object) -> _FakeRunResult:
        nonlocal leader_calls
        leader_calls += 1
        deps = _deps_from_kwargs(kwargs)
        if leader_calls == 1 and deps is not None:
            task = await deps.board.add_task('Do it')
            await deps.board.assign(task.id, 'worker')
            return _FakeRunResult(output='seeded')
        result = await Agent.run(
            leader,
            user_prompt,
            deps=deps,
            usage=_usage_from_kwargs(kwargs),
        )
        return _FakeRunResult(output=result.output)

    monkeypatch.setattr(leader, 'run', _seed_one_task)

    async def _complete_assigned(user_prompt: str, **kwargs: object) -> object:
        deps = _deps_from_kwargs(kwargs)
        assert deps is not None
        for task in deps.board.snapshot():
            if task.assignee == 'worker' and task.status is not TaskStatus.DONE:
                await deps.board.complete(task.id, result='done', agent_id='worker')
        return _FakeRunResult(output='worked')

    monkeypatch.setattr(worker, 'run', _complete_assigned)

    events: list[object] = []
    with leader.override(model=TestModel(custom_output_text='final-synth')):
        async with team.iter('Finish work') as run:
            async for event in run:
                events.append(event)
            assert run.result is not None
            assert run.result.data == 'final-synth'

    assert any(isinstance(e, PhaseJoined) and e.phase == 'members' and not e.incomplete for e in events)
    assert any(isinstance(e, TasksScheduled) and e.tasks[0].kind == 'synthesize' for e in events)
    assert isinstance(events[-1], RunEnded)


async def test_team_task_defaults() -> None:
    task = TeamTask(kind='seed')
    assert task.agent_id is None
    assert task.task_ids == ()


async def test_iter_drains_on_early_exit() -> None:
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1)

    with leader.override(model=TestModel(call_tools=[], custom_output_text='solo')):
        async with team.iter('Nothing to split') as run:
            async for event in run:
                if isinstance(event, TasksScheduled):
                    break
        assert run.result is not None
        assert run.result.data == 'solo'


async def test_iter_propagates_member_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_agent=leader,
        members=[worker],
        max_rounds=1,
        dispatch_mode='streaming',
    )

    leader_calls = 0

    async def _seed_one_task(user_prompt: str, **kwargs: object) -> _FakeRunResult:
        nonlocal leader_calls
        leader_calls += 1
        deps = _deps_from_kwargs(kwargs)
        if leader_calls == 1 and deps is not None:
            task = await deps.board.add_task('Boom')
            await deps.board.assign(task.id, 'worker')
            return _FakeRunResult(output='seeded')
        result = await Agent.run(
            leader,
            user_prompt,
            deps=deps,
            usage=_usage_from_kwargs(kwargs),
        )
        return _FakeRunResult(output=result.output)

    monkeypatch.setattr(leader, 'run', _seed_one_task)

    async def _boom(user_prompt: str, **_kwargs: object) -> object:
        raise RuntimeError('member failed')

    monkeypatch.setattr(worker, 'run', _boom)

    with pytest.raises(RuntimeError, match='member failed'):
        async with team.iter('Ship it') as run:
            async for event in run:
                if isinstance(event, TasksScheduled) and any(t.kind == 'member_tick' for t in event.tasks):
                    break
