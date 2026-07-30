from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_team import CollaborativeTeam, TeamResult, collaborative as collaborative_mod
from pydantic_team.board import TaskBoard, TaskStatus
from pydantic_team.collaborative import (
    BoardDeps,
    DispatchMode,
    add_task,
    approve_task,
    assign_reviewer,
    assign_task,
    claim_task,
    complete_task,
    default_leader_instructions,
    list_messages,
    list_tasks,
    member_work_prompt,
    reject_task,
    replan_user_prompt,
    review_work_prompt,
    seed_user_prompt,
    send_message,
)
from pydantic_team.events import MessagePosted, PhaseJoined, RunEnded, TaskReviewDecided, TasksScheduled, TeamEvent


@dataclass
class _FakeCtx:
    deps: object


def _ctx(
    board: TaskBoard,
    agent_id: str,
    *,
    member_ids: tuple[str, ...] = ('alice', 'bob'),
    emit: Callable[[TeamEvent], Awaitable[None]] | None = None,
) -> RunContext[object]:
    return cast(
        RunContext[object],
        _FakeCtx(
            deps=BoardDeps(
                board=board,
                agent_id=agent_id,
                member_ids=member_ids,
                emit=emit,
            )
        ),
    )


def test_collaborative_requires_leader() -> None:
    member = Agent(TestModel(), name='worker')
    with pytest.raises(ValueError, match='leader_agent or leader_model'):
        CollaborativeTeam(members=[member])


def test_collaborative_rejects_both_leader_sources() -> None:
    leader = Agent(TestModel(), name='leader')
    member = Agent(TestModel(), name='worker')
    with pytest.raises(ValueError, match='not both'):
        CollaborativeTeam(leader_agent=leader, leader_model='test', members=[member])


def test_collaborative_requires_members() -> None:
    with pytest.raises(ValueError, match='members'):
        CollaborativeTeam(leader_model='test', members=[])


def test_collaborative_requires_positive_rounds() -> None:
    member = Agent(TestModel(), name='worker')
    with pytest.raises(ValueError, match='max_rounds'):
        CollaborativeTeam(leader_model='test', members=[member], max_rounds=0)


def test_collaborative_requires_non_negative_replans() -> None:
    member = Agent(TestModel(), name='worker')
    with pytest.raises(ValueError, match='max_replans'):
        CollaborativeTeam(leader_model='test', members=[member], max_replans=-1)


def test_replan_user_prompt_includes_goal_roster_and_board() -> None:
    board = TaskBoard()
    text = replan_user_prompt('Ship it', ['researcher', 'writer'], board)
    assert 'Ship it' in text
    assert 'researcher' in text
    assert 'writer' in text
    assert 'add_task' in text
    assert 'assign_task' in text
    assert '(empty)' in text
    assert 'Messages:\n(none)' in text


async def test_replan_user_prompt_includes_messages() -> None:
    board = TaskBoard()
    await board.post_message('researcher', 'writer', 'draft ready')
    text = replan_user_prompt('Ship it', ['researcher', 'writer'], board)
    assert 'Messages:' in text
    assert 'researcher->writer' in text
    assert 'draft ready' in text


async def test_collaborative_run_returns_team_result() -> None:
    leader = Agent(TestModel(), name='leader', instructions='Lead the board.')
    worker = Agent(TestModel(), name='worker', instructions='Claim and complete tasks.')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=2)
    assert list(team.members) == [worker]
    assert team.leader is leader

    with leader.override(model=TestModel(custom_output_text='final-summary')):
        with worker.override(model=TestModel(custom_output_text='worker-done')):
            result = await team.run('Ship the feature')

    assert isinstance(result, TeamResult)
    assert result.data == 'final-summary'
    assert result.usage.requests >= 1


async def test_collaborative_no_tasks_returns_lead_output() -> None:
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=2)

    with leader.override(model=TestModel(call_tools=[], custom_output_text='solo-lead')):
        with worker.override(model=TestModel(custom_output_text='should-not-matter')):
            result = await team.run('Nothing to split')

    assert result.data == 'solo-lead'


async def test_collaborative_leader_tools_create_tasks() -> None:
    """Seed/replan expose board tools; empty-board early exit avoids toolless synthesize."""
    leader_model = TestModel(call_tools=[], custom_output_text='solo-lead')
    leader = Agent(leader_model, name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1)

    with leader.override(model=leader_model):
        with worker.override(model=TestModel(custom_output_text='ok')):
            await team.run('Create work items')

    assert leader_model.last_model_request_parameters is not None
    tool_names = {t.name for t in leader_model.last_model_request_parameters.function_tools}
    assert 'add_task' in tool_names
    assert 'assign_task' in tool_names
    assert 'assign_reviewer' in tool_names
    assert 'approve_task' in tool_names
    assert 'reject_task' in tool_names
    assert 'list_tasks' in tool_names


async def test_collaborative_member_tools_include_claim() -> None:
    leader = Agent(TestModel(), name='leader')
    worker_model = TestModel()
    worker = Agent(worker_model, name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1)

    with leader.override(model=TestModel(custom_output_text='lead')):
        with worker.override(model=worker_model):
            await team.run('Do parallel work')

    assert worker_model.last_model_request_parameters is not None
    tool_names = {t.name for t in worker_model.last_model_request_parameters.function_tools}
    assert 'claim_task' in tool_names
    assert 'complete_task' in tool_names
    assert 'approve_task' in tool_names
    assert 'reject_task' in tool_names


async def test_collaborative_from_leader_model() -> None:
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_model='test',
        members=[worker],
        system_prompt_override='Seed tasks then synthesize.',
        max_rounds=1,
    )
    with team.leader.override(model=TestModel(custom_output_text='done')):
        with worker.override(model=TestModel(custom_output_text='w')):
            result = await team.run('Goal')
    assert result.data == 'done'


async def test_collaborative_system_prompt_on_existing_leader() -> None:
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_agent=leader,
        members=[worker],
        system_prompt_override='Always create at least one task.',
        max_rounds=1,
    )
    with leader.override(model=TestModel(custom_output_text='ok')):
        with worker.override(model=TestModel(custom_output_text='ok')):
            result = await team.run('x')
    assert result.data == 'ok'


async def test_collaborative_unnamed_agents_use_fallback_ids() -> None:
    leader = Agent(TestModel())
    worker = Agent(TestModel())
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1)
    with leader.override(model=TestModel(call_tools=[], custom_output_text='ok')):
        with worker.override(model=TestModel(custom_output_text='ok')):
            result = await team.run('x')
    assert result.data == 'ok'


async def test_board_tools_error_and_success_paths() -> None:
    board = TaskBoard()
    ctx = _ctx(board, 'alice')

    assert await list_tasks(ctx, None) == 'No tasks'
    assert 'Error: invalid status' in await list_tasks(ctx, 'nope')
    created_msg = await add_task(ctx, 'Research notes')
    assert 'Created' in created_msg

    created = (await board.list_tasks())[0]
    listed = await list_tasks(ctx, 'open')
    assert created.id in listed

    assert 'Error' in await assign_task(ctx, 'missing', 'alice')
    assert 'Assigned' in await assign_task(ctx, created.id, 'alice')

    board2 = TaskBoard()
    open_task = await board2.add_task('U')
    ctx2 = _ctx(board2, 'bob')
    assert 'Claimed' in await claim_task(ctx2, open_task.id)
    assert 'Error' in await claim_task(ctx2, open_task.id)
    assert 'Completed' in await complete_task(ctx2, open_task.id, 'done')
    assert 'Error' in await complete_task(ctx2, open_task.id, 'again')


async def test_send_and_list_messages_tools() -> None:
    board = TaskBoard()
    emitted: list[TeamEvent] = []

    async def _emit(event: TeamEvent) -> None:
        emitted.append(event)

    alice = _ctx(board, 'alice', member_ids=('alice', 'bob'), emit=_emit)
    bob = _ctx(board, 'bob', member_ids=('alice', 'bob'))
    leader = _ctx(board, 'leader', member_ids=('alice', 'bob'))

    assert await list_messages(alice) == 'No messages'
    assert 'Error: cannot send a message to yourself' in await send_message(alice, 'alice', 'noop')
    assert 'Error: unknown teammate' in await send_message(alice, 'carol', 'hi')
    assert 'Error: to must be a teammate id or "*"' in await send_message(alice, '  ', 'hi')

    sent = await send_message(alice, 'bob', 'need outline', task_id='')
    assert sent.startswith('Sent msg-')
    assert len(emitted) == 1
    assert isinstance(emitted[0], MessagePosted)
    assert emitted[0].message.body == 'need outline'

    bob_view = await list_messages(bob)
    assert 'alice->bob' in bob_view
    assert 'need outline' in bob_view

    await send_message(alice, '*', 'standup')
    assert 'alice->*' in await list_messages(bob)

    # Leader sees the full log.
    leader_view = await list_messages(leader)
    assert 'need outline' in leader_view
    assert 'standup' in leader_view

    task = await board.add_task('Write')
    assert 'Sent' in await send_message(alice, 'bob', 'about task', task_id=task.id)
    assert f'task={task.id}' in await list_messages(bob)
    assert 'Error:' in await send_message(alice, 'bob', 'bad link', task_id='missing')

    # emit=None still posts successfully.
    silent = _ctx(board, 'bob', member_ids=('alice', 'bob'), emit=None)
    assert 'Sent' in await send_message(silent, 'alice', 'ack')


async def test_member_work_prompt_includes_visible_messages() -> None:
    board = TaskBoard()
    await board.post_message('alice', 'researcher', 'fyi')
    await board.post_message('writer', 'alice', 'secret')
    prompt = member_work_prompt('Goal', 'researcher', board)
    assert 'alice->researcher' in prompt
    assert 'secret' not in prompt


async def test_board_deps_type_error() -> None:
    with pytest.raises(TypeError, match='BoardDeps'):
        await list_tasks(cast(RunContext[object], _FakeCtx(deps='nope')), None)


def test_default_leader_instructions_require_assign_by_role() -> None:
    text = default_leader_instructions(['researcher', 'writer'])
    assert 'researcher' in text
    assert 'writer' in text
    assert 'assign_task' in text
    assert 'Do not leave tasks open without an assignee' in text
    assert 'list_messages' in text


def test_seed_user_prompt_includes_roster() -> None:
    text = seed_user_prompt('Ship it', ['researcher', 'writer'])
    assert 'Ship it' in text
    assert 'researcher' in text
    assert 'assign_task' in text


async def test_member_work_prompt_lists_only_own_assignments() -> None:
    board = TaskBoard()
    research = await board.add_task('Research notes')
    writing = await board.add_task('Draft copy')
    await board.assign(research.id, 'researcher')
    await board.assign(writing.id, 'writer')

    researcher_prompt = member_work_prompt('Brief', 'researcher', board)
    assert "Your agent id is 'researcher'" in researcher_prompt
    assert 'do not claim tasks assigned to others' in researcher_prompt
    assert research.id in researcher_prompt
    assert 'Tasks assigned to you:\n- task-1' in researcher_prompt
    assert 'send_message' in researcher_prompt
    assert 'Messages visible to you:\n(none)' in researcher_prompt

    writer_prompt = member_work_prompt('Brief', 'writer', board)
    assert writing.id in writer_prompt
    assert "Your agent id is 'writer'" in writer_prompt


async def test_member_work_prompt_none_when_unassigned() -> None:
    board = TaskBoard()
    await board.add_task('Open work')
    prompt = member_work_prompt('Goal', 'researcher', board)
    assert 'Tasks assigned to you:\n(none)' in prompt


async def test_member_work_prompt_max_assignments_caps_list() -> None:
    board = TaskBoard()
    first = await board.add_task('First')
    second = await board.add_task('Second')
    await board.assign(first.id, 'researcher')
    await board.assign(second.id, 'researcher')
    prompt = member_work_prompt('Goal', 'researcher', board, max_assignments=1)
    assert first.id in prompt
    assert second.id not in prompt.split('Tasks assigned to you:')[1].split('Full board:')[0]


def test_collaborative_rejects_invalid_assignments_per_tick() -> None:
    member = Agent(TestModel(), name='worker')
    with pytest.raises(ValueError, match='max_assignments_per_tick'):
        CollaborativeTeam(leader_model='test', members=[member], max_assignments_per_tick=0)


def test_collaborative_rejects_invalid_dispatch_mode() -> None:
    member = Agent(TestModel(), name='worker')
    with pytest.raises(ValueError, match='dispatch_mode'):
        CollaborativeTeam(leader_model='test', members=[member], dispatch_mode=cast(DispatchMode, 'nope'))


async def test_collaborative_from_leader_model_includes_member_roster() -> None:
    researcher = Agent(TestModel(), name='researcher')
    writer = Agent(TestModel(), name='writer')
    team = CollaborativeTeam(leader_model='test', members=[researcher, writer], max_rounds=1)
    assert 'researcher' in team.member_ids
    assert 'writer' in team.member_ids
    instructions = default_leader_instructions(list(team.member_ids))
    assert 'researcher' in instructions
    assert 'assign_task' in instructions


async def test_collaborative_member_swallows_unexpected_model_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1)

    async def _boom(*_args: object, **_kwargs: object) -> object:
        raise UnexpectedModelBehavior('Exceeded maximum output retries (1)')

    monkeypatch.setattr(worker, 'run', _boom)

    with leader.override(model=TestModel(custom_output_text='final')):
        result = await team.run('Ship it')
    assert result.data == 'final'


async def test_streaming_empty_seed_returns_leader_output() -> None:
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_agent=leader,
        members=[worker],
        max_rounds=1,
        dispatch_mode='streaming',
    )
    events: list[object] = []
    with leader.override(model=TestModel(call_tools=[], custom_output_text='solo-stream')):
        with worker.override(model=TestModel(custom_output_text='unused')):
            async with team.iter('Nothing to split') as run:
                async for event in run:
                    events.append(event)
                assert run.result is not None
                assert run.result.data == 'solo-stream'
                assert run.board.is_complete()
                assert not run.board.snapshot()
    assert team.dispatch_mode == 'streaming'
    assert not any(isinstance(e, TasksScheduled) and any(t.kind == 'synthesize' for t in e.tasks) for e in events)
    assert isinstance(events[-1], RunEnded)
    assert events[-1].result.data == 'solo-stream'


async def test_streaming_dispatch_early_return_emits_run_ended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force early TeamResult from dispatch to cover _drive_streaming early-exit path."""
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_agent=leader,
        members=[worker],
        max_rounds=1,
        dispatch_mode='streaming',
    )
    early_result: TeamResult[object] = TeamResult(data='forced-early', usage=RunUsage())

    async def _force_early(_self: object) -> TeamResult[object]:
        return early_result

    streaming_dispatch = getattr(collaborative_mod, '_StreamingDispatch')
    monkeypatch.setattr(streaming_dispatch, 'run', _force_early)

    events: list[object] = []
    with leader.override(model=TestModel(custom_output_text='should-not-synthesize')):
        with worker.override(model=TestModel(custom_output_text='unused')):
            async with team.iter('Forced early exit') as run:
                async for event in run:
                    events.append(event)
                assert run.result is not None
                assert run.result.data == 'forced-early'

    assert not any(isinstance(e, TasksScheduled) and any(t.kind == 'synthesize' for t in e.tasks) for e in events)
    assert isinstance(events[-1], RunEnded)
    assert events[-1].result.data == 'forced-early'


def test_early_result_if_empty_seed_returns_team_result() -> None:
    """Deterministic cover of empty-seed early exit (no asyncio scheduler)."""
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    board = TaskBoard()
    usage = RunUsage()

    async def _emit(_event: object) -> None:
        return None

    async def _run_member(
        _member: object,
        _prompt: str,
        _board: TaskBoard,
        _usage: RunUsage,
        _emit: object,
    ) -> None:
        return None

    dispatch_cls = getattr(collaborative_mod, '_StreamingDispatch')
    dispatch = cast(
        object,
        dispatch_cls(
            leader=leader,
            members_by_id={'worker': worker},
            member_ids=['worker'],
            max_rounds=1,
            max_replans=0,
            max_assignments_per_tick=None,
            user_prompt='goal',
            run_usage=usage,
            board=board,
            lead_deps=BoardDeps(board=board, agent_id='leader', member_ids=('worker',)),
            run_member=_run_member,
            emit=_emit,
        ),
    )
    setattr(dispatch, 'seed_output', 'solo-from-seed')
    setattr(dispatch, 'had_tasks', False)

    early = getattr(dispatch, '_early_result_if_empty_seed')()
    assert early is not None
    assert early.data == 'solo-from-seed'
    assert early.usage is usage
    assert getattr(dispatch, '_idle_should_finish')(early) is True

    setattr(dispatch, 'had_tasks', True)
    assert getattr(dispatch, '_early_result_if_empty_seed')() is None
    assert getattr(dispatch, '_idle_should_finish')(None) is True  # empty board is complete

    incomplete = TaskBoard()
    asyncio.run(incomplete.add_task('open'))
    setattr(dispatch, 'board', incomplete)
    setattr(dispatch, 'replans_used', 0)
    setattr(dispatch, 'max_replans', 1)
    assert getattr(dispatch, 'board').is_complete() is False
    assert getattr(dispatch, '_idle_should_finish')(None) is False
    assert getattr(dispatch, '_idle_should_finish')(TeamResult(data='x', usage=usage)) is True


async def test_emit_members_joined_if_needed_both_paths() -> None:
    """Deterministic cover of PhaseJoined(members) emit vs no-op (no asyncio scheduler)."""
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    board = TaskBoard()
    usage = RunUsage()
    emitted: list[object] = []

    async def _emit(event: object) -> None:
        emitted.append(event)

    async def _run_member(
        _member: object,
        _prompt: str,
        _board: TaskBoard,
        _usage: RunUsage,
        _emit: object,
    ) -> None:
        return None

    dispatch_cls = getattr(collaborative_mod, '_StreamingDispatch')
    dispatch = cast(
        object,
        dispatch_cls(
            leader=leader,
            members_by_id={'worker': worker},
            member_ids=['worker'],
            max_rounds=1,
            max_replans=0,
            max_assignments_per_tick=None,
            user_prompt='goal',
            run_usage=usage,
            board=board,
            lead_deps=BoardDeps(board=board, agent_id='leader', member_ids=('worker',)),
            run_member=_run_member,
            emit=_emit,
        ),
    )

    setattr(dispatch, 'had_tasks', False)
    await getattr(dispatch, '_emit_members_joined_if_needed')()
    assert emitted == []
    assert getattr(dispatch, '_members_joined') is False

    setattr(dispatch, 'had_tasks', True)
    await getattr(dispatch, '_emit_members_joined_if_needed')()
    assert len(emitted) == 1
    assert isinstance(emitted[0], PhaseJoined)
    assert emitted[0].phase == 'members'
    assert getattr(dispatch, '_members_joined') is True

    await getattr(dispatch, '_emit_members_joined_if_needed')()
    assert len(emitted) == 1


async def test_streaming_dispatch_run_returns_early_on_empty_seed() -> None:
    """Wire-up: empty board after seed returns TeamResult from run() (covers return early)."""
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    board = TaskBoard()
    usage = RunUsage()

    async def _emit(_event: object) -> None:
        return None

    async def _run_member(
        _member: object,
        _prompt: str,
        _board: TaskBoard,
        _usage: RunUsage,
        _emit: object,
    ) -> None:
        return None

    dispatch_cls = getattr(collaborative_mod, '_StreamingDispatch')
    dispatch = cast(
        object,
        dispatch_cls(
            leader=leader,
            members_by_id={'worker': worker},
            member_ids=['worker'],
            max_rounds=1,
            max_replans=0,
            max_assignments_per_tick=None,
            user_prompt='goal',
            run_usage=usage,
            board=board,
            lead_deps=BoardDeps(board=board, agent_id='leader', member_ids=('worker',)),
            run_member=_run_member,
            emit=_emit,
        ),
    )

    async def _instant_seed() -> object:
        return 'solo-seed-output'

    setattr(dispatch, '_leader_seed', _instant_seed)
    result = await getattr(dispatch, 'run')()
    assert result is not None
    assert result.data == 'solo-seed-output'
    assert not board.snapshot()


@dataclass
class _FakeRunResult:
    output: object


def _deps_from_kwargs(kwargs: dict[str, object]) -> BoardDeps | None:
    deps = kwargs.get('deps')
    return deps if isinstance(deps, BoardDeps) else None


def _usage_from_kwargs(kwargs: dict[str, object]) -> RunUsage | None:
    usage = kwargs.get('usage')
    return usage if isinstance(usage, RunUsage) else None


async def test_streaming_member_starts_during_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    member_started = asyncio.Event()
    overlap_confirmed = asyncio.Event()

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
            overlap_confirmed.set()
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
        result = await Agent.run(
            worker,
            user_prompt,
            deps=_deps_from_kwargs(kwargs),
            usage=_usage_from_kwargs(kwargs),
        )
        return result

    monkeypatch.setattr(worker, 'run', _worker_run)

    with worker.override(model=TestModel(custom_output_text='worked')):
        result = await team.run('Ship overlapping work')

    assert overlap_confirmed.is_set()
    assert result.data is not None


async def test_streaming_one_inflight_tick_per_member(monkeypatch: pytest.MonkeyPatch) -> None:
    concurrent = 0
    max_concurrent = 0

    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_agent=leader,
        members=[worker],
        max_rounds=3,
        max_assignments_per_tick=1,
        dispatch_mode='streaming',
    )

    leader_calls = 0

    async def _seed_two_tasks(user_prompt: str, **kwargs: object) -> _FakeRunResult:
        nonlocal leader_calls
        leader_calls += 1
        deps = _deps_from_kwargs(kwargs)
        if leader_calls == 1 and deps is not None:
            first = await deps.board.add_task('First')
            second = await deps.board.add_task('Second')
            await deps.board.assign(first.id, 'worker')
            await deps.board.assign(second.id, 'worker')
            return _FakeRunResult(output='seeded')
        result = await Agent.run(
            leader,
            user_prompt,
            deps=deps,
            usage=_usage_from_kwargs(kwargs),
        )
        return _FakeRunResult(output=result.output)

    monkeypatch.setattr(leader, 'run', _seed_two_tasks)

    async def _worker_run(user_prompt: str, **kwargs: object) -> object:
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.05)
        try:
            return await Agent.run(
                worker,
                user_prompt,
                deps=_deps_from_kwargs(kwargs),
                usage=_usage_from_kwargs(kwargs),
            )
        finally:
            concurrent -= 1

    monkeypatch.setattr(worker, 'run', _worker_run)

    with worker.override(model=TestModel(custom_output_text='worked')):
        await team.run('Many tasks for one worker')

    assert max_concurrent == 1


async def test_streaming_swallows_unexpected_model_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
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
        raise UnexpectedModelBehavior('Exceeded maximum output retries (1)')

    monkeypatch.setattr(worker, 'run', _boom)

    result = await team.run('Ship it')
    assert result.data is not None


async def test_streaming_replan_when_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Incomplete board after capped ticks triggers a leader replan."""
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_agent=leader,
        members=[worker],
        max_rounds=1,
        max_replans=1,
        max_assignments_per_tick=1,
        dispatch_mode='streaming',
    )

    leader_calls = 0
    saw_replan = False

    async def _seed_then_real(user_prompt: str, **kwargs: object) -> _FakeRunResult:
        nonlocal leader_calls, saw_replan
        leader_calls += 1
        deps = _deps_from_kwargs(kwargs)
        if 'Review the board after member work' in user_prompt:
            saw_replan = True
        if leader_calls == 1 and deps is not None:
            task = await deps.board.add_task('Unfinished')
            await deps.board.assign(task.id, 'worker')
            return _FakeRunResult(output='seeded')
        result = await Agent.run(
            leader,
            user_prompt,
            deps=deps,
            usage=_usage_from_kwargs(kwargs),
        )
        return _FakeRunResult(output=result.output)

    monkeypatch.setattr(leader, 'run', _seed_then_real)

    async def _idle_worker(user_prompt: str, **_kwargs: object) -> object:
        return None

    monkeypatch.setattr(worker, 'run', _idle_worker)

    with leader.override(model=TestModel(custom_output_text='synth')):
        result = await team.run('Need replan')

    assert saw_replan
    assert leader_calls >= 3  # seed + replan + synthesize
    assert result.data == 'synth'


async def test_streaming_completed_board_synthesizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """When members finish all tasks, streaming drains then synthesizes."""
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

    with leader.override(model=TestModel(custom_output_text='final-synth')):
        result = await team.run('Finish work')

    assert result.data == 'final-synth'
    assert leader_calls >= 2  # seed + synthesize


async def test_per_cycle_usage_isolation_with_prefilled_aggregate() -> None:
    """Each agent cycle uses a fresh RunUsage; aggregate at the limit must not block seed."""
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1)
    usage = RunUsage()
    usage.requests = 50

    with leader.override(model=TestModel(call_tools=[], custom_output_text='solo-lead')):
        with worker.override(model=TestModel(custom_output_text='unused')):
            result = await team.run('Nothing to split', usage=usage)

    assert result.data == 'solo-lead'
    assert usage.requests > 50
    assert result.usage is usage


async def test_synthesize_strips_leader_board_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthesize must not offer add_task/assign_task (toolless leader cycle)."""
    leader_model = TestModel(custom_output_text='final-from-board')
    leader = Agent(leader_model, name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(leader_agent=leader, members=[worker], max_rounds=1)

    leader_calls = 0
    board_ref: TaskBoard | None = None
    tasks_before_synth = 0

    async def _seed_then_real(user_prompt: str, **kwargs: object) -> _FakeRunResult:
        nonlocal leader_calls, board_ref, tasks_before_synth
        leader_calls += 1
        deps = _deps_from_kwargs(kwargs)
        if leader_calls == 1 and deps is not None:
            board_ref = deps.board
            task = await deps.board.add_task('Do it')
            await deps.board.assign(task.id, 'worker')
            return _FakeRunResult(output='seeded')
        assert deps is not None
        tasks_before_synth = len(deps.board.snapshot())
        result = await Agent.run(
            leader,
            user_prompt,
            deps=deps,
            usage=_usage_from_kwargs(kwargs),
        )
        return _FakeRunResult(output=result.output)

    monkeypatch.setattr(leader, 'run', _seed_then_real)

    async def _complete_assigned(user_prompt: str, **kwargs: object) -> object:
        deps = _deps_from_kwargs(kwargs)
        assert deps is not None
        for task in deps.board.snapshot():
            if task.assignee == 'worker' and task.status is not TaskStatus.DONE:
                await deps.board.complete(task.id, result='done', agent_id='worker')
        return _FakeRunResult(output='worked')

    monkeypatch.setattr(worker, 'run', _complete_assigned)

    with leader.override(model=leader_model):
        result = await team.run('Finish work')

    assert result.data == 'final-from-board'
    assert leader_calls >= 2
    assert board_ref is not None
    assert len(board_ref.snapshot()) == tasks_before_synth == 1
    assert leader_model.last_model_request_parameters is not None
    tool_names = {t.name for t in leader_model.last_model_request_parameters.function_tools}
    assert 'add_task' not in tool_names
    assert 'assign_task' not in tool_names


async def test_streaming_propagates_member_errors(monkeypatch: pytest.MonkeyPatch) -> None:
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
        await team.run('Ship it')


async def test_require_review_add_task_stamps_leader_as_reviewer() -> None:
    board = TaskBoard()
    ctx = _ctx(board, 'leader', member_ids=('worker',))
    cast(BoardDeps, ctx.deps).require_review = True
    cast(BoardDeps, ctx.deps).leader_id = 'leader'
    msg = await add_task(ctx, 'Draft')
    assert 'reviewer=leader' in msg
    task = (await board.list_tasks())[0]
    assert task.reviewer == 'leader'


async def test_assign_approve_reject_tools() -> None:
    board = TaskBoard()
    emitted: list[TeamEvent] = []

    async def _emit(event: TeamEvent) -> None:
        emitted.append(event)

    leader = _ctx(board, 'leader', member_ids=('worker',), emit=_emit)
    cast(BoardDeps, leader.deps).leader_id = 'leader'
    worker = _ctx(board, 'worker', member_ids=('worker',), emit=_emit)
    cast(BoardDeps, worker.deps).leader_id = 'leader'

    await add_task(leader, 'Write')
    task = (await board.list_tasks())[0]
    assert 'Error: unknown reviewer' in await assign_reviewer(leader, task.id, 'ghost')
    assert 'Reviewer' in await assign_reviewer(leader, task.id, 'leader')
    await assign_task(leader, task.id, 'worker')
    assert 'Submitted' in await complete_task(worker, task.id, 'draft v1')
    assert 'Error' in await approve_task(worker, task.id)
    assert 'Rejected' in await reject_task(leader, task.id, 'expand')
    assert any(isinstance(e, TaskReviewDecided) and e.decision == 'rejected' for e in emitted)
    assert 'Submitted' in await complete_task(worker, task.id, 'draft v2')
    assert 'Approved' in await approve_task(leader, task.id)
    assert any(isinstance(e, TaskReviewDecided) and e.decision == 'approved' for e in emitted)
    assert board.is_complete()


async def test_member_work_prompt_includes_revision_and_reviews() -> None:
    board = TaskBoard()
    task = await board.add_task('Write', reviewer='leader')
    await board.assign(task.id, 'writer')
    await board.complete(task.id, result='v1', agent_id='writer')
    await board.reject(task.id, reason='short', agent_id='leader')
    assignee_prompt = member_work_prompt('Goal', 'writer', board)
    assert 'short' in assignee_prompt
    assert 'v1' in assignee_prompt
    assert TaskStatus.NEEDS_REVISION.value in assignee_prompt or 'NEEDS_REVISION' in assignee_prompt
    review_prompt = review_work_prompt('Goal', 'leader', board)
    # still needs_revision until complete again
    assert 'awaiting your review' in review_prompt.lower() or 'Tasks awaiting' in review_prompt
    await board.complete(task.id, result='v2', agent_id='writer')
    review_prompt2 = review_work_prompt('Goal', 'leader', board)
    assert task.id in review_prompt2
    assert 'v2' in review_prompt2


async def test_phased_require_review_leader_approves(monkeypatch: pytest.MonkeyPatch) -> None:
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_agent=leader,
        members=[worker],
        max_rounds=2,
        max_replans=0,
        require_review=True,
        dispatch_mode='phased',
    )

    async def _leader_run(user_prompt: str, **kwargs: object) -> object:
        deps = _deps_from_kwargs(kwargs)
        assert deps is not None
        if 'awaiting your review' in user_prompt.lower() or 'Review pending' in user_prompt:
            for task in deps.board.snapshot():
                if task.status is TaskStatus.PENDING_REVIEW:
                    await deps.board.approve(task.id, agent_id='leader')
            return _FakeRunResult(output='reviewed')
        if 'Synthesize' in user_prompt:
            return _FakeRunResult(output='final')
        task = await deps.board.add_task('Do work', reviewer='leader')
        await deps.board.assign(task.id, 'worker')
        return _FakeRunResult(output='seeded')

    async def _worker_run(user_prompt: str, **kwargs: object) -> object:
        deps = _deps_from_kwargs(kwargs)
        assert deps is not None
        for task in deps.board.snapshot():
            if task.assignee == 'worker' and task.status in (TaskStatus.CLAIMED, TaskStatus.NEEDS_REVISION):
                await deps.board.complete(task.id, result='done', agent_id='worker')
        return _FakeRunResult(output='worked')

    monkeypatch.setattr(leader, 'run', _leader_run)
    monkeypatch.setattr(worker, 'run', _worker_run)

    async with team.iter('Ship with review') as run:
        async for _event in run:
            pass
    assert run.result is not None
    assert run.result.data == 'final'
    assert run.board.is_complete()


async def test_member_prompt_lists_pending_reviews_for_reviewer() -> None:
    board = TaskBoard()
    task = await board.add_task('Check', reviewer='verifier')
    await board.assign(task.id, 'writer')
    await board.complete(task.id, result='draft', agent_id='writer')
    prompt = member_work_prompt('Goal', 'verifier', board)
    assert task.id in prompt
    assert 'draft' in prompt


async def test_assign_reviewer_board_error() -> None:
    board = TaskBoard()
    ctx = _ctx(board, 'leader', member_ids=('worker',))
    cast(BoardDeps, ctx.deps).leader_id = 'leader'
    assert 'Error' in await assign_reviewer(ctx, 'missing', 'leader')


async def test_approve_reject_without_emit() -> None:
    board = TaskBoard()
    task = await board.add_task('X', reviewer='leader')
    await board.assign(task.id, 'worker')
    await board.complete(task.id, result='v1', agent_id='worker')
    leader = _ctx(board, 'leader', member_ids=('worker',), emit=None)
    cast(BoardDeps, leader.deps).leader_id = 'leader'
    assert 'Rejected' in await reject_task(leader, task.id, 'fix')
    await board.complete(task.id, result='v2', agent_id='worker')
    assert 'Approved' in await approve_task(leader, task.id)


async def test_phased_review_tick_ignores_empty_model(monkeypatch: pytest.MonkeyPatch) -> None:
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_agent=leader,
        members=[worker],
        max_rounds=2,
        require_review=True,
        dispatch_mode='phased',
    )
    review_calls = 0

    async def _leader_run(user_prompt: str, **kwargs: object) -> object:
        nonlocal review_calls
        deps = _deps_from_kwargs(kwargs)
        assert deps is not None
        if 'awaiting your review' in user_prompt.lower():
            review_calls += 1
            if review_calls == 1:
                raise UnexpectedModelBehavior('empty')
            for task in deps.board.snapshot():
                if task.status is TaskStatus.PENDING_REVIEW:
                    await deps.board.approve(task.id, agent_id='leader')
            return _FakeRunResult(output='ok')
        if 'Synthesize' in user_prompt:
            return _FakeRunResult(output='final')
        task = await deps.board.add_task('T', reviewer='leader')
        await deps.board.assign(task.id, 'worker')
        return _FakeRunResult(output='seed')

    async def _worker_run(user_prompt: str, **kwargs: object) -> object:
        deps = _deps_from_kwargs(kwargs)
        assert deps is not None
        for task in deps.board.snapshot():
            if task.assignee == 'worker' and task.status is TaskStatus.CLAIMED:
                await deps.board.complete(task.id, result='x', agent_id='worker')
        return _FakeRunResult(output='w')

    monkeypatch.setattr(leader, 'run', _leader_run)
    monkeypatch.setattr(worker, 'run', _worker_run)
    result = await team.run('goal')
    assert result.data == 'final'
    assert review_calls >= 1


async def test_streaming_require_review_leader_approves(monkeypatch: pytest.MonkeyPatch) -> None:
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_agent=leader,
        members=[worker],
        max_rounds=3,
        max_replans=0,
        require_review=True,
        dispatch_mode='streaming',
    )

    async def _leader_run(user_prompt: str, **kwargs: object) -> object:
        deps = _deps_from_kwargs(kwargs)
        assert deps is not None
        if 'awaiting your review' in user_prompt.lower():
            for task in deps.board.snapshot():
                if task.status is TaskStatus.PENDING_REVIEW:
                    await deps.board.approve(task.id, agent_id='leader')
            return _FakeRunResult(output='reviewed')
        if 'Synthesize' in user_prompt:
            return _FakeRunResult(output='final')
        task = await deps.board.add_task('Do work', reviewer='leader')
        await deps.board.assign(task.id, 'worker')
        return _FakeRunResult(output='seeded')

    async def _worker_run(user_prompt: str, **kwargs: object) -> object:
        deps = _deps_from_kwargs(kwargs)
        assert deps is not None
        for task in deps.board.snapshot():
            if task.assignee == 'worker' and task.status is TaskStatus.CLAIMED:
                await deps.board.complete(task.id, result='done', agent_id='worker')
        return _FakeRunResult(output='worked')

    monkeypatch.setattr(leader, 'run', _leader_run)
    monkeypatch.setattr(worker, 'run', _worker_run)

    async with team.iter('Ship streaming review') as run:
        async for _event in run:
            pass
    assert run.result is not None
    assert run.result.data == 'final'
    assert run.board.is_complete()


async def test_streaming_review_empty_model_then_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    leader = Agent(TestModel(), name='leader')
    worker = Agent(TestModel(), name='worker')
    team = CollaborativeTeam(
        leader_agent=leader,
        members=[worker],
        max_rounds=1,
        max_replans=0,
        require_review=True,
        dispatch_mode='streaming',
    )
    review_calls = 0

    async def _leader_run(user_prompt: str, **kwargs: object) -> object:
        nonlocal review_calls
        deps = _deps_from_kwargs(kwargs)
        assert deps is not None
        if 'awaiting your review' in user_prompt.lower():
            review_calls += 1
            raise UnexpectedModelBehavior('empty review')
        if 'Synthesize' in user_prompt:
            return _FakeRunResult(output='final-incomplete')
        task = await deps.board.add_task('Do work', reviewer='leader')
        await deps.board.assign(task.id, 'worker')
        return _FakeRunResult(output='seeded')

    async def _worker_run(user_prompt: str, **kwargs: object) -> object:
        deps = _deps_from_kwargs(kwargs)
        assert deps is not None
        for task in deps.board.snapshot():
            if task.assignee == 'worker' and task.status is TaskStatus.CLAIMED:
                await deps.board.complete(task.id, result='done', agent_id='worker')
        return _FakeRunResult(output='worked')

    monkeypatch.setattr(leader, 'run', _leader_run)
    monkeypatch.setattr(worker, 'run', _worker_run)

    async with team.iter('Ship') as run:
        async for _event in run:
            pass
    assert run.result is not None
    assert run.result.data == 'final-incomplete'
    assert review_calls == 1
    assert not run.board.is_complete()
